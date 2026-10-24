# tile-p4: two-rank MoE tile intervention and its propagation to the target distribution

**State.** Built, CPU-tested and run once (2026-10-24; 2.53 GPU-hours with the single-GPU checks). **Run 1 is INCOMPLETE**
(re-analysed with the corrected rules: still INCOMPLETE). **The 7.0 ceiling is VOID; see the 2026-10-24 update below
(analysis fixes, Stage-0 package, new approval needed).** Run 1:
see `RESULT.md` and `CRITIQUE-run1.md`. This page describes the package as built, before the run. Design: `PROTOCOL.md`. Authorization:
`AUTHORIZATION.txt` (4.0 occupied GPU-hours in total, retries and single-GPU checks included). Changes forced by the
records and the engine: `PROPOSAL-AMENDMENTS.md`. The session runner will not start until the operator acknowledges
that file. CPU test results: `tests/RESULTS.txt`.

## Update 2026-10-24 (22az): exact-byte capture and version comparison

- **`capture-p0/`** (protocol, package, CPU tests, two attempts; 0.897 GPU-h under a 2.0 GPU-h operator approval):
  attempt 1 read INCOMPLETE (two package defects, owned in `capture-p0/PROTOCOL.md` amendments A1-A2); attempt 2
  passed every gate. Pre-registered reading: **operation located**. Identical warm prefills first part at layer 3's
  indexer top-k output from token 2,051, where the top-k holds the same pools in a different order
  (`capture-p0/RESULT.md`).
- **`version-cmp/`** (no GPU): the prefill top-k code and binary are identical in v0.7.0 and v0.9.1, its output
  order is not specified, nothing between top-k and attention fixes the order, and the sparse-MLA prefill kernel
  walks the index list in the given order in both versions. The deployed configurations differ (v0.7.0 DCP=2,
  v0.9.1 DCP=1 with MLA layer ownership). Source-level only; v0.7.0 was not run.

## Update 2026-10-24 (22ay): Stage 0 ran; zero-GPU audit

- **Stage 0** was approved (1.34 GPU-h) and run: 0.717 GPU-h, result `RESULT-STAGE0.md`. No P4 GPU step is
  authorized now.
- **Audit** of the retained Stage 0 records: `audit-s0/AUDIT.md` (scripts in `audit-s0/scripts/`, run on CPU with
  `--network none`; hashes in `audit-s0/SHA256SUMS`). Valid cached state differs from layer 4 on; the operation is not
  located.
- **Wording corrected, marked "22ay":** `RESULT-STAGE0.md`, `STAGE0-PROTOCOL.md`. The protocol edit changes its sha256,
  so the `S0_PROTOCOL_ACK` value printed below no longer matches; Stage 0 is complete and is not re-run.
  Previous hash list: `SHA256SUMS.pre-22ay`.
- The section below describes the package as it stood before Stage 0 ran.

## Update 2026-10-24: analysis fixes and the Stage-0 package (CPU only; nothing has run on a GPU since run 1)

**State:**
- **Run 1 remains INCOMPLETE.**
- **The 7.0 GPU-h ceiling is VOID** (`AUTHORIZATION.txt`, last line). No P4 GPU step is authorized.
- The run-1 runner (`runner/run_p4.sh`), its pre-check and its G4 now refuse every step (`runner/budget_p4.py`, `VOID = True`).
- Stage 0 needs a new explicit approval: **APPROVED_GPU_H ≥ 1.34**.
  - That covers the caps: Stage-0 session 1.25 GPU-h, pre-check 0.083.
  - Expected use is 0.92-1.13 GPU-h. See `STAGE0-PROTOCOL.md` section 7.

Governing requirements: `../HANDOFF-NEXT/ASSESSOR-DISPOSITION-22AV-2026-10-24.md`, sections 2, 4, 5 and 6.

Copies of every file changed by this update are in `build/stage0-prev/`. The previous hash list is `SHA256SUMS.pre-stage0`.

### A. Analysis and recording fixes (assessor section 2)

1. **Immutable request identity (local-p4 schema 2).**
   - What `on_add` does:
     - gives every keyed request a monotonic uid (never reused) and a tag hash;
     - logs them once (public `req` record);
     - writes them to device tables at add time.
   - `on_step` gathers them into every step-ring row (eager), so attribution is captured **at record time**. Removal resets the tables.
   - Graph-safe: no in-graph op was added for identity, and there is no host sync in capture.
   - The analysis labels compositions only through uid → tag, and checks every tag-keyed forcewin row-0 measurement against the ring (attribution check).
   - **Run-1 path:** schema-1 rings are ignored. Compositions are rebuilt from the per-request, tag-keyed forcewin records (per-position step, row, n_rows, accepted), and host step facts are joined by those step ids. The reading is INCOMPLETE by rule.
2. **`decide()`** (`analysis/analyze_p4.py`) applies rules 1-6 in order:
   1. gates or session missing → INCOMPLETE;
   2. any failed forcing, attribution or measurement check (records missing on either rank, a store missing, not measured once, rank disagreement, invariants, forcewin errors, acceptance, returned ids) → INCOMPLETE;
   3. run-1 records → INCOMPLETE;
   4. coverage below 50 % of planned positions for AB_1, AB_2, AA, AA2_1, AA2_2 or the joint e1_moe23 / e3_lp_forced families → INCOMPLETE;
   5. A not bit-stable on ≥ 50 % of compared positions → INCOMPLETE;
   6. only then an effect reading.
   - A negative reading exists only at rule 6.
   - The `ab > null` threshold-exceedance fallback is removed as an effect criterion. It remains a table labelled DESCRIPTIVE ONLY.
3. **Tests:** `tests/make_fake_sessions.py` and `check_analysis.py`, 15 scenarios. **46/46 PASS.**
   - Each of these reads INCOMPLETE and never a negative finding: zero coverage, partial coverage (104 of 2,048 positions), reused indices without uids (schema 1), run-1-style records, a missing store, a missing rank-1 record, a bad forced request, gate failure, and A not bit-stable.
   - Reused indices WITH uids read correctly.
   - Full-coverage null data is the only scenario that gives the negative reading.
4. **Run 1 re-analysed:** `analysis/out-run1-corrected/` (aggregates only; `OUT_DIR=analysis/out-run1-corrected analysis/run_analysis.sh`).
   - **Reading: INCOMPLETE; descriptive only.**
   - Even with the corrected, tag-keyed composition key, only 35-80 of 24,576 positions per pair have matching step keys, and the joint families have 0. The step compositions genuinely differ between repeats.
   - All measurement checks pass: 96/96 forced ids per session, 13,140 acceptance checks with 0 deviations, 0 rank mismatches.

**Corrected wording of the run-1 description** (assessor section 3; replaces the corresponding sentences of `CRITIQUE-run1.md` and `RESULT.md`, which stay unchanged as records):

> Repeated executions under this harness produced different outputs at identical forced token histories. Their
> execution schedules differed; the contributions of cached state, remainder prefill, current-batch arithmetic, D2
> and instrumentation have not been isolated.

- **Host step facts** match on 21,906 of 24,576 positions (89.1 %) for A repeat 1 vs repeat 2 (`diagnostics/diag.out`).
- **Within that subset,** the median |Δ ln P(`</think>`)| is 0.168 for A1-A2 and 0.184 for A1-B1. The two distributions look similar descriptively. That does not establish equivalence and does not exclude a systematic tile effect.
- **Withdrawn:**
  - the "2-3 orders of magnitude" comparison (it compared different measurements);
  - "scheduling and packing alone".
- **Position-0 disagreement** does not prove that the incoming state differed: the position-0 computations ran in different mixed batches.
- **The logprob is pre-nucleus.** Terminal-token eligibility and the post-filter probability need assessing before it is read as a change in the immediate chance of termination.
- **The run-1 client:**
  - released 8 separate HTTP requests from a barrier, which does not fix their arrival order;
  - warmed replies concurrently, under one salt, with repeat 2 reusing repeat 1's cache;
  - used different seeds per repeat;
  - joined all threads between passes.
- **Cache contamination** is a candidate to inspect, not a finding.

### B. Stage-0 package (assessor section 5; `STAGE0-PROTOCOL.md`)

| Path | What |
|---|---|
| `build/files/vllm/p4_hooks.py` | Schema 2 identity, plus the stage-0 recording (`GLM53_P4_S0=on`): scheduler execution log, per-layer state digests (in-graph, fixed shapes), logits-row staging of eager steps above 64 tokens, window-entry KV-cache digests on both ranks, block provenance. With `GLM53_P4_S0` unset, the run-1 suites pass unchanged |
| `build/port_forcewin.py` (P14-P16) → `build/files/vllm/forcewin_hooks.py` (fe0a0b17) | Multi-prompt selection by prompt key (`local_fw_multi`); a missing or duplicate key leaves the request unforced |
| `build/make_build_s0.sh`, `build/stage0/` | Image **`local/glm53-v091-p4-s0:local` = `sha256:7408f893dfe35f1fbd330153e34ea81e83460fe5e2fff200ab87ea992565d3c7`** (`docker build --network none --pull=false` from the stock base). Run-1 gates (a)-(c) pass, plus (b4) stage-0 on with nothing allocated before registration, and (b5) `GLM53_P4_S0` alone is inert |
| `runner/stage0/` | `run_stage0.sh` (session), `client_stage0.py` (V1 / V2, salts, seeds, warm order), `budget_stage0.py` (APPROVED_GPU_H, `costs-stage0.jsonl`), `outcome_stage0.py`, `cost_stage0.py` / `.json`, `run_precheck_s0.sh`, `images.txt` |
| `precheck/precheck_s0.py` | Single-GPU pre-check, to be run later by the coordinator: graph capture with the new recording (C1-C4, N1) |
| `analysis/analyze_stage0.py`, `run_stage0_analysis.sh` | G-enq → G-entry → G-out, stop at the first failure; first-difference descriptors; the assessor's interpretation table |
| `tests/test_stage0_cpu.py`, `test_precheck_s0_cpu.py`, `make_fake_stage0.py` / `check_stage0.py`, `mock_server_stage0.py` / `test_stage0_runner.sh` | CPU suites (`tests/run_cpu_tests.sh s0 s0image`; results in `tests/RESULTS-stage0.txt`) |

**Launch order** (operator, after a new approval):

```bash
cd <runs>/tile-p4
APPROVED_GPU_H=<approved> ATTEMPT_REASON="stage-0 pre-check" runner/stage0/run_precheck_s0.sh
APPROVED_GPU_H=<approved> ATTEMPT_REASON="stage-0 probe" S0_PROTOCOL_ACK=$(sha256sum STAGE0-PROTOCOL.md | cut -c1-12) \
  runner/stage0/run_stage0.sh            # analysis runs automatically -> runs-stage0/stage0-attempt-1/stage0-result.md
```

### Open risks of the Stage-0 package (what the CPU tests cannot cover)

1. **Entry-digest layout on the real engine.**
   - The code assumes each KV tensor's first dimension is `num_blocks × pages-per-block`, block-major.
   - This was verified on synthetic caches only. A violating tensor is listed as uncovered at start-up.
   - G-instr stops the session before traffic only if a rank has no covered tensor at all (cost: the start-up time, about 0.45 GPU-h).
   - Uncovered groups are listed and narrow G-entry; they do not stop the session.
2. **GPU memory.**
   - Added after KV sizing: about 30 MB on rank 0 and about 25 MB on rank 1.
   - Rank 1 had no P4 buffers in run 1.
   - Entry digests also use one block-sized temporary per tensor (a few MB).
3. **Timing.**
   - The per-layer digests add about 10 small kernels per layer per step inside the graphs.
   - The entry digests add an estimated 0.1-0.25 s per digest point.
   - Both are the same in every repeat, but they make Stage 0 a different execution from run 1.
4. **The multi-prompt path on the real server is exercised only by the mock.**
   - The real OpenAI server's per-prompt ordering, and the choice index ↔ prompt mapping, are read from the stock source (`serving.py`: request ids `<id>-<i>`, results by index).
   - A mismatch shows up as failed ids checks (INCOMPLETE), never as a silently wrong window.
5. **The estimate's digest-time and solitary-decode rates are planning values.** The cap leaves an 11 % margin over the planning case. The client skips trailing replicates when needed, which gives INCOMPLETE for V2.

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
