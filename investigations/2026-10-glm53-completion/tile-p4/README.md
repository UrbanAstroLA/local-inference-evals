# tile-p4: two-rank MoE tile intervention and its propagation to the target distribution

**State.** Built, CPU-tested and run once (2026-10-24; 2.53 GPU-hours with the single-GPU checks). **Run 1 is INCOMPLETE**:
see `RESULT.md` and `CRITIQUE-run1.md`. This page describes the package as built, before the run. Design: `PROTOCOL.md`. Authorization:
`AUTHORIZATION.txt` (4.0 occupied GPU-hours in total, retries and single-GPU checks included). Changes forced by the
records and the engine: `PROPOSAL-AMENDMENTS.md`. The session runner will not start until the operator acknowledges
that file. CPU test results: `tests/RESULTS.txt`.

## Layout
| Path | What |
|---|---|
| `build/stock/` | Stock 0.9.1 sources extracted from the local base image a9e985c0 (read-only inputs) |
| `build/make_hooked.py` | Generates the five hooked engine files; gates: additive-only, one allowed changed line, stock-equivalence after removal, forcewin call sites == the GPU-validated attempt-2 copies |
| `build/port_forcewin.py`, `forcewin-port.diff` | Stock port of the validated forcewin module (15e93e23 → d6b10341), anchored replacements only |
| `build/files/vllm/p4_hooks.py` | Recorder, graph-safe stages, per-request stores, step records, flush at removal |
| `build/make_build.sh`, `Dockerfile.p4`, `post-build-gates.txt`, `image-id.txt` | Image `local/glm53-v091-p4-tile:local` = `sha256:894bd469…20045` |
| `windows/` | Window rule + fixture build (`select_windows.py`, `extract_sched.py`, `run_select.sh`); public manifest `windows/out/` |
| `PRIVATE-fixtures/` (0700) | `windows.json` (token ids, positions, batch plan), `sched.json` |
| `precheck/` | Single-GPU graph-capture pre-check (`run_precheck.sh`, `precheck_graph.py`) |
| `g4/` | G4 offline single-GPU MoE reproduction (`run_g4.sh`, `g4_moe.py`, `in_container_g4.sh`, `g4_verdict.py`) |
| `runner/` | `run_p4.sh` (sessions), `client_p4.py`, `gates_p4.py` (G1-G3), `budget_p4.py` (ledger, order, caps), `outcome_p4.py`, `cost_p4.py`/`.json`, `images.txt` |
| `analysis/` | `analyze_p4.py` (E1-E5, decision rules), `run_analysis.sh` (CPU container) |
| `tests/` | CPU suites (`run_cpu_tests.sh`), mock server, fabricated sessions; logs in `tests/out/` |
| `SHA256SUMS`, `make_sums.sh` | Frozen package hashes (checked by the session runner) |

Created at run time: `costs.jsonl` (shared ledger); `runs/<step>-attempt-N/` (public logs, gates, outcome);
`PRIVATE-runs/session-<S>-attempt-N/` (0700: forcewin records, P4 stores, step rings).

## 1. The image (CPU-built; 7 of 2,436 vLLM files differ from stock)
`local/glm53-v091-p4-tile:local` = `sha256:894bd46976b8ef0b61e6fa1d7846c930e0575c6300713f5242f1570a79b20045`.
- Built with `docker build --network none --pull=false` FROM the local stock image.
- Of 2,436 vLLM `.py` files, exactly 7 differ from stock: 2 new modules and 5 hooked files (`post-build-gates.txt`
  gate (a)).

### 1.1 The intervention
`exl3.py:2480` `tile_config = None` becomes `tile_config = _p4_forced_tile()  # local-p4`.
- The helper is defined in exl3.py itself as marked lines, so arm B never depends on another module importing.
- It returns `(64, 256, 64, 256)` only when `GLM53_P4_TILE` is exactly `64,256,64,256`. Otherwise it returns None.
- Post-build gate (b2) checks it with the env set to `64,256,64,256`, `64,256,64,256 `, `64,256`, `128,128,128,128`
  and `1`.

### 1.2 Source reading: with the env unset, every code path is stock
Checked line by line. `make_hooked.py` proves it mechanically: dropping every inserted line and restoring line 2480
gives the stock bytes.

| File | Marked lines | With GLM53_P4, GLM53_P4_TILE and FORCEWIN unset |
|---|---|---|
| `exl3.py` | Import block (try-import of `vllm.p4_hooks`; `_P4_ON = False`; a tuple constant; the def of `_p4_forced_tile`) | Module-level definitions only. `p4_hooks` at import reads env and defines functions: no tensor, no thread, no patching (gate b1) |
| `exl3.py:2495` (stock 2480) | `tile_config = _p4_forced_tile()` | Evaluates to `None`, the stock literal. The following statements are unchanged |
| `exl3.py` 3 sites before `runtime["api"].run(binding=binding)` | `if _P4_ON: _p4.hook_bind(...)` | One boolean test each, false |
| `model.py` import block and 3 sites (after the MoE block, after each layer in the model loop, after the final norm) | `if _P4_ON: ...` | One boolean test each, false. No tensor op is issued, so captured CUDA graphs are identical |
| `model_runner.py`, `rejection_sampler.py`, `sampler.py` | forcewin call sites, `if _FW_ON:` | Byte-identical to the GPU-validated attempt-2 copies (8d276b3a, f5a875ef, bbb1df81); `_FW_ON` is false |
| `forcewin_hooks.py`, `p4_hooks.py` | New modules | Imported only through the guarded try-imports. Inert flags verified in the image (gate b1) |

**With the env set (sessions):**
- The P4 hooks only read engine tensors and write their own buffers.
  - In-graph: fixed-shape copies; at layers 12 and 34 also `hc_post` + `hc_contract`, the read-only
    materialisation the DFlash tap already runs.
  - Eager: copies from the stages to the stores.
- The only engine-state writes are forcewin's forcing of tokens and drafts. It is identical in both arms by design.

### 1.3 The forcewin port: differences from B1 and how each is handled
| Piece | B1 (validated) | Stock 0.9.1 | Handling |
|---|---|---|---|
| `model_runner.py`, `rejection_sampler.py`, `sampler.py` | Hashes 101e456b / 308d4171 / 76070739 | **Byte-identical** to B1's | The validated `make_patch.py` edits applied unchanged. Outputs byte-identical to the attempt-2 hooked copies (checked) |
| `indexer.py` D2 injection (`d2_tail_mapping`, `note_tail_mapping`) | Anchor = B1's persistent tail-buffer copy | The anchor does not exist: stock has D2 natively | Not applied. `D2_ALLOWED` is hard false |
| `sparse_attn_indexer_kpool.py` exposure counters (`on_selection`) | Window exposure | File differs; not a P4 endpoint | Not applied (fewer in-graph hooks) |
| Shadow ledger (`SHADOW`) | Models B1's corrected tail ring | Stock ring collapses to block 0 | Hard false |
| Snapshots (`SNAP`) | Read B1's mamba metadata (`positions` passed) | `mamba_hybrid.py` differs (no positions) | Hard false. Any of the three env switches set is logged as an error and fails the start-up gate |
| Scheduler | Mapguard hook | Stock | Not needed |
| Router GEMM, routed SwiGLU clamp | B1 changes in `gate_linear.py`, `exl3.py` | Stock | Numerics differ from B1. No B1 number is reused as a reference |
| Draft modes | natural / oracle / poison / reject / sched | — | Adds `list` (fixed per-step accepted counts, key `local_fw_sched`) for the proposal's `sched` |
| Bridge to P4 | — | — | 8 guarded calls (`register`, `add`, `remove`, `on_batch`, `on_logits`, `_commit_rows`, `override`, `override_single`), each exception-isolated |

### 1.4 What local-p4 records (rank 0 unless stated)
**Bound-launch recorder (both ranks, all 42 MoE layers).**
- One record per (rank, layer, plan, m): size_m, fc1/fc2 tile K/N, moe_block_size, fc2 block, max_m_blocks, plan,
  capacity, min/max trellis m, and the weight-plan tile. A changed signature is logged again.
- It is host-only. Under FULL graphs it records at warm-up and capture, which are the launches every replay re-issues.
- Draft layers are labelled. The settings needed to rebuild the plans are plain JSON (captured-p3b lesson 2).

**Stages.** Fixed buffers allocated in `capture_model` before capture.
- Contents: MoE output after the all-reduce at layers 3, 23, 44; contracted post-layer hidden state at layers 12
  and 34; final norm.
- Written in-graph by copies (m ≤ 64). Prefill chunks are not staged.

**Stores.** 8 request slots.
- Filled by eager device ops from forcewin's recorded rows (each output index once, on the intended prefix).
- MoE output at every position; hidden states at every 8th; full fp32 logits at every 64th (staged before the
  sampler).

**Step records.** A host step log (time, counts, graph mode) and a device ring of per-request composition.

**Flush.** At request removal, the slot is written to `PRIVATE/p4_<tag>_r0.npz` (0600, O_EXCL).

**No poller, no thread, no pinned memory, no stream or event** (captured-p3b lesson 1).
- Device-to-host copies happen only at removal.
- They are refused while a stream is capturing and before the first real batch (tests S3 and C1).

## 2. Windows (`windows/`, amendments A1-A5)
**No stock 0.9.1 record keeps generated ids beyond 5k tokens.** The two stock 0.9.1 passes with full text
(long-reply d091, d091b) give:
- **8 finished after 100k: all available, all used.**
- **14 capped, of which 8 chosen** (seed 20261023).

Reasoning is re-tokenized with the served tokenizer and template, as in B+/D/E. All checks pass: prompt length ==
logged prompt_tokens; no stop id or `</think>` inside; windows disjoint.

Windows, W = 512:
- early and deep at P ≡ 1 (mod 4,352) nearest 16k and 96k;
- near-close: the 512 tokens before `</think>`; for capped replies, before output index 131,072 (A2; the
  records' cap is 327,680).

Batches and totals:
- 48 windows in 6 batches of 8, composed reply by reply to fit the prefix-cache horizon.
- 2.14M warm-prefill tokens and 0.23M uncached remainder tokens per pass.
- `sched`: 1,024 accepted counts from hook-validation's natural forcewin runs (mean 3.06 tokens per step).

## 3. GPU steps (operator only; exact order, enforced by `runner/budget_p4.py`)
```bash
cd <runs>/tile-p4
ACK=$(sha256sum PROPOSAL-AMENDMENTS.md | cut -c1-12)     # acknowledges the amendments
# 1. single-GPU pre-check, about 0.01-0.02 GPU-h (cap 300 s)
ATTEMPT_REASON="P4 pre-check" precheck/run_precheck.sh
# 2. G4, single GPU, about 0.05 GPU-h (cap 900 s): patched image, env unset, reproduces captured-p3b's served MoE
#    output bit for bit (gate); G4b with the tile env (reported)
ATTEMPT_REASON="P4 G4" g4/run_g4.sh
# 3-5. sessions, 2 GPUs each, wall cap 2,350 s = 1.306 GPU-h each; G1-G3 evaluated before traffic
SESSION=A  ATTEMPT_REASON="P4 session A"  P4_AMENDMENTS_ACK=$ACK runner/run_p4.sh
SESSION=B  ATTEMPT_REASON="P4 session B"  P4_AMENDMENTS_ACK=$ACK runner/run_p4.sh
SESSION=A2 ATTEMPT_REASON="P4 session A'" P4_AMENDMENTS_ACK=$ACK runner/run_p4.sh
# 6. analysis (CPU container)
analysis/run_analysis.sh        # -> analysis/out/result.md, result.json
```
Every command accepts `DRY_RUN=1`, which runs the gates and prints the plan, starting and writing nothing.

**Estimate** (`runner/cost_p4.py`):

| Case | Per session |
|---|---|
| Planning (start 874 s, prefill 2,400 tok/s, forced decode 34 tok/s/request) | 2,289 s = 1.27 GPU-h |
| Typical (740 s, 2,750 tok/s) | 2,017 s = 1.12 GPU-h |

- Checks: 0.067 GPU-h.
- **Total: 3.88 GPU-h planning, 3.43 typical, against 4.0.** A full retry of a session does not fit at planning
  rates.
- `budget_p4.py` reserves 1.27 GPU-h for each later session and refuses a session whose cap would fall below
  1,300 s.

**What each session does:**
- **Pre-flight:** no glm53 or check container; GPUs idle with re-sampled utilization; no foreign GPU client.
- **Start-up:** start; at readiness, check that the hooks registered on both ranks with no error.
- **Gates before any traffic:** G3 (A, against captured-p3b), G1 + G2 (B, against A), or the A' control.
- **Client:** 6 batches × (warm, 8 forced requests twice).
- **Flush and stop:** wait for the flush, then stop.
- **Bookkeeping:** one cost line counted to GPU release; copy-out; final gates on every bound launch;
  `outcome.json` (COMPLETE / PARTIAL / INCOMPLETE).

## 4. CPU test results
See `tests/RESULTS.txt`. Summary at the end of the build:

| Suite | Result |
|---|---|
| gen | Hooked files regenerate; additive gates; stock-equivalence; forcewin call sites == validated copies |
| units | Stage hooks, graph-static op sequence, flush rule, lint, register, recorder |
| list | End-to-end forcing with the `list` schedule through the real sampler, rejection sampler, RequestState and post_update, with stores verified bit for bit |
| modes | The same with the oracle and poison schedules |
| forcing | hook-validation's validated forcing suite against the ported module, 9 shards plus the bridge on |
| inert | Identical output hashes: stock / hooked env unset / hooked env on unkeyed / built image |
| precheck | Dry run on CPU: C1-C3; N1 is GPU-only |
| gates | 13 planted cases on bind logs built from captured-p3b's real launches |
| runner | Refusals; budget and order; DRY_RUN; MOCK sessions A → B → A' including a start-up gate failure; client deadline; precheck and G4 wrappers with fake docker |
| analysis | 7 fabricated scenarios, each giving its planted reading; exclusions exact |
| image | units, list, forcing rest and inertness in the built image |

## 5. Open risks (what the CPU tests cannot cover)
1. **Real graph capture.** Capture of the real model with the P4 hooks inside is first exercised at session A's
   start-up.
   - The pre-check uses synthetic tensors.
   - A capture failure ends the attempt at start-up (about 0.45 GPU-h).
2. **GPU memory.** About 0.2 GB of P4 stores plus forcewin's transient rows are allocated after KV sizing, at
   `gpu_memory_utilization` 0.95. captured-p3b's 0.12 GB worked.
3. **Prefix cache.** The cache behaviour (the EAGLE rule and the about 0.66M-token horizon) was diagnosed on B1. The
   stock KV-manager files are the same; `cached_tokens` is recorded per request.
   - A miss costs time, which can drop the trailing batch in every session. Validity is not affected.
4. **Bit-stability of A.** Stock D2 tail-ring races at 8 in flight may make A not bit-stable between repeats. The
   analysis then uses its distributional fallback.
   - Shape-matched positions also depend on the batch's arrival order. Mismatches are excluded and counted.
5. **Budget margin.** At planning rates a session uses 2,289 s of its 2,350 s share, so there is no room for a full
   retry.
6. **What the forcewin port rests on.** The B1 GPU validation covered byte-identical call-site files. The stock-only
   behaviour (D2, causal length) is outside what forcewin touches, but the port itself has only CPU evidence on stock.
7. **Re-tokenized histories** (A1). Out-of-nucleus history tokens are flagged.
8. **Draft model.** If the DFlash2 drafter has EXL3 projection-mixed MoE layers, the tile env changes them too. Drafts
   are overridden by the schedule, so target inputs are unaffected; draft binds are labelled in the recorder.
