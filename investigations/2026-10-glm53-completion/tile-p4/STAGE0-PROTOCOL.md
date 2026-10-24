# P4 Stage 0: determinism probe (protocol, written before any Stage-0 GPU step; approved and run 2026-10-24, result `RESULT-STAGE0.md`)

**Status: not run. No GPU use is authorized.**
- The 7.0 GPU-h ceiling is VOID (`AUTHORIZATION.txt`, 2026-10-24).
- Stage 0 needs a new, explicit operator approval for a stated number of GPU-hours (section 7).
- This protocol follows the assessor's 22av disposition, sections 2, 4, 5 and 6 (`HANDOFF-NEXT/ASSESSOR-DISPOSITION-22AV-2026-10-24.md`).

## 1. Question and scope

**Question:** under this harness, is matched-history forced decode on stock 0.9.1 reproducible?
- **V1, solitary:** one window at a time.
- **V2, co-batched:** the 8 windows of a batch together.

**Where it is not reproducible,** Stage 0 reports the earliest trustworthy divergence. That report separates:
- incoming-state differences, from
- current-step execution differences.

**Scope:**
- Arm A only (`GLM53_P4_TILE` unset). No arm B and no A/B reading.
- ONE representative batch: batch 1 of the frozen plan.
  - 8 windows: R01-early/deep/close, R02-early/deep/close, R03-deep/close.
  - 3 warm prefixes, 350,550 warm tokens.
  - Remainders of 4,353-5,823 tokens.
- The analysis stops at the first failed gate (section 4).
- Not part of Stage 0:
  - repeating the 48-window experiment;
  - any propagation or behavioural test.
- Further GPU use after Stage 0 is decided by the operator, with the assessor.
  - The approximately 2.5 GPU-h P4b is NOT committed by a Stage-0 pass.

## 2. Construction (`runner/stage0/client_stage0.py`)

**Session:**
- One serving session on 2 GPUs.
- Image `local/glm53-v091-p4-s0:local` with `GLM53_P4=on GLM53_P4_S0=on FORCEWIN=on`.
- The recipe environment is unchanged: KDA mode, DFlash2 with 3 draft tokens, chunked prefill, prefix caching (align mode) and FULL_AND_PIECEWISE graphs are as in run 1.

**Four independently warmed replicates, in this fixed order:** V1-r1, V1-r2, V2-r1, V2-r2.

| Element | Rule |
|---|---|
| Cache salt | `s0-<variant>-r<rep>`, one per replicate: no replicate can reuse another replicate's cached prefix |
| Warm | One request per reply, in plan order, **each awaited before the next** (solitary chunked prefill). Prompt = the reply's longest prefix in the batch; `max_tokens` 1; seed 72999. Keyed with an empty forced list, so its steps and block ids are recorded. Nothing is forced or stored for it |
| Forced, V1 | The 8 windows **one at a time**, each awaited, so the remainder prefill runs alone and the decode runs alone. Order: replies in plan order; within a reply, descending P (close, deep, early). No forced request then computes a block inside a later window's cached span. This is checked (G-entry provenance) |
| Forced, V2 | **ONE multi-prompt HTTP request** with the 8 prompts in plan order. Each engine request picks its own forced list and tag by its prompt key (length, first 8 and last 8 ids; `local_fw_multi`, forcewin port P14-P16). A missing or duplicate key leaves the request unforced, and its ids check fails. It is never silently mis-forced |
| Sampler | T 1.0, top_p 0.95; draft mode `list` with the fixed acceptance list (`sched`, 1,024 counts); `max_tokens` 512 |
| Seed | 73000 for **every** forced request of every replicate, so exact repeats share the seed |
| Between replicates | An unkeyed 4-token sentinel (own salt). It makes the engine remove and flush the last finished requests. The next replicate starts after it returns |

**What the separate salt does and does not do:**
- It prevents cross-replicate prefix reuse.
- It does **not** reset:
  - allocation history (physical block ids, free-list order and state-slot indices differ between replicates; they are recorded);
  - D2's shared tail block.
- **A measured repeat never becomes the next repeat's state builder:**
  - every replicate rebuilds its prefixes under its own salt;
  - within V1, the descending-P order and the provenance check (section 4) cover window-to-window reuse.

**On the multi-prompt submission:**
- It fixes the order in which the server issues the engine requests.
- It does not by itself guarantee deterministic scheduling.
- The engine-side enqueue order and the scheduler's execution are therefore recorded and gated (G-enq).

## 3. What is recorded (local-p4 schema 2, `GLM53_P4_S0=on`; `build/files/vllm/p4_hooks.py` sections 6-7)

| Record | Where | Content |
|---|---|---|
| Request identity | public `req`; step ring columns `uid`, `taghash` | A per-request monotonic uid assigned at add. It is captured into every step-ring row at record time and reset at removal. It is never reused |
| Scheduler execution | public `sched_r0_*.jsonl` | Per real step, the batch rows in batch order: engine request id, req_idx, uid, tag, scheduled tokens, computed tokens, computed prefill tokens, prefill length, prefilling flag, draft tokens. Also per step: real and padded token and request counts, graph mode, and the MoE plans bound by eager launches |
| Enqueue order | `PRIVATE-fw/fw_r0_*.jsonl` `add` records | The order in which the runner added the requests |
| Window-entry state | `PRIVATE/entry_<tag>_r<rank>.npz`, both ranks | Before the forward of each prefill step of a forced request (at most 7), and of its first non-prefilling step ("entry"). Every KV-cache tensor of every group is digested per **logical** block of the request (MLA KV, indexer K, KDA conv and recurrent state, draft caches), plus logical block 0 (the shared null block). Logical block ids are recorded too |
| Per-layer state | `ldig` in `PRIVATE/p4_<tag>_r0.npz` | Two int64 digests of each token row of the state after every target layer (hidden, residual, mHC post, comb), at every recorded position. The state after layer L is the state entering layer L + 1 |
| E1 / E2 / E4 | `PRIVATE/p4_<tag>_r0.npz` | As in run 1. In addition, positions measured in eager steps with more than 64 tokens (mixed prefill and verify steps, including o = 0) are now stored, from the step's logits rows |
| E3 | `PRIVATE-fw/req_<tag>_r<rank>.pt` | The forcewin per-position metrics, unchanged |
| Block provenance | public `blocks` | The logical block ids per group of every keyed request, at its removal |

**How each assessor item maps to these records:**
- **Prefill chunk boundaries:** `n_computed_prefill` and `n_sched` per row.
- **Start of verification:** the first step whose row is not prefilling and carries drafts.
- **Bound launch including padding:**
  - padded token count plus graph mode;
  - eager steps: the plans bound in the step;
  - graph replays: the capture-time bind record at the padded size (the bind recorder).

## 4. Gates (`analysis/analyze_stage0.py`)

**Scope of the comparison:** each variant separately, repeat 1 against repeat 2 of the same window. The gates are applied **in order**, and the first failure ends that variant's reading. Later gates are marked "not evaluated".

1. **G-enq: execution identical.**
   - The forced requests' enqueue order is identical.
   - Step by step from the replicate's first keyed step, every field of the scheduler record is identical: rows in batch order, chunk sizes, computed counts, prefilling flags, draft counts, real and padded sizes, graph mode and eager plans.
   - Labels are replicate-free. Engine ids, req_idx, uids and physical block ids are not compared.
   - Applied to both variants. For V2, this is the assessor's requirement that the prefill chunk boundaries and the start of verification match, not only the submission order.
2. **G-entry: window-entry state identical.** For every window and both ranks:
   - the same digest points (kind, computed and scheduled counts);
   - equal digests for every covered tensor and logical block at every point;
   - every cached block of the forced request was computed by **its own replicate's warm request** for that reply (provenance; groups with the largest block size).
   - The null block is reported but not required: it is shared scratch written by every request.
3. **G-out: outputs identical.** For output positions 0-63 of every window:
   - E1 (MoE output at layers 3, 23 and 44) is bit-equal;
   - E3 (every forcewin column except the absolute step id) is bit-equal;
   - every position is measured exactly once and staged in both repeats. **Missing coverage fails the gate; it never passes it.**

**The first differing input, state or operation is reported for the failed gate:**

| Failed gate | Reported |
|---|---|
| G-enq | The step offset and field (or the enqueue index) |
| G-entry | The point (prefill or entry), tensor and logical block, and whether the cached-prefix state at point 0 was equal |
| G-out | The position and endpoint; the first layer whose state digest differs at that position; whether that step's recorded composition matched |

How these reports separate the two sources of difference:
- **Incoming state:** a G-entry difference at point 0 lies in the cached prefix, built by the warm request.
- **Remainder prefill:** a difference first appearing at a later point was produced by the remainder prefill, the current-step arithmetic of that request's prefill steps.
- **Current-step execution:** if G-entry passes and G-out fails, the incoming state was equal and the difference arose in the current step's execution. The per-layer digests localize the first layer.

## 5. Interpretation (assessor 22av section 5; printed by the analysis)

| Stage-0 result | Reading |
|---|---|
| V1 reproducible, V2 not | A batch/concurrency-dependent execution difference is implicated. Its mechanism remains unassigned |
| Neither reproducible | Investigate the first differing input, state or operation. Do not immediately label stock kernels nondeterministic |
| V2 reproducible | Proceed only after also establishing a common A/B starting construction |
| V2 reproducible, V1 not | Unexpected. Investigate V1's first difference before any A/B |
| A replicate missing, or not 8/8 forced ok | INCOMPLETE for that variant. No interpretation |

**Section 6 of the disposition holds for anything after Stage 0:**
- Within-arm repeatability cannot prove that A and B start from equivalent state.
- After the intervention begins, state divergence can be part of the tile's effect.
- ≤ 8-row steps are a clean local control only with matched inputs and state.
- 10-32-row launches are the direct-exposure stratum, not independent observations.

## 6. Checks before and at the session

**Single-GPU pre-check** (`precheck/precheck_s0.py`, `runner/stage0/run_precheck_s0.sh`; synthetic tensors, no model). It must PASS before Stage 0:
- **C1:** CUDA-graph capture with the per-layer digests in-graph. A flush and the entry digests are refused inside the capture.
- **C2:** replays plus the eager bridge. Stores and per-layer digests equal a host recomputation. The step ring carries the uid and tag hash.
- **C3:** a 300-token eager step stores its logits rows.
- **C4:** entry digests equal a host recomputation on a synthetic KV cache with 4 tensors in 2 groups.
- **N1:** the negative control.

**At start-up, before any traffic:**
- **G-instr:** the hooks and the stage-0 recording are registered on both ranks (schema 2, `s0`, digest layers > 0), and each rank has at least one covered KV-cache tensor. Uncovered tensors and groups are listed in the start-up record and in the result's health section; G-entry then covers only the covered tensors (section 8, item 3). A layout surprise therefore narrows G-entry rather than costing the start-up.
- **G3:** stock launch geometry against captured-p3b.
- If either fails, the session stops at start-up.

**After the client** (in the runner's cleanup, after the GPUs are released):
- `outcome.json`;
- the final G3;
- the analysis, in a CPU container.

## 7. Cost and budget

**Estimate** (`runner/stage0/cost_stage0.py`, `cost_stage0.json`):

| Case | Start-up | Per V1 replicate | Per V2 replicate | Total | GPU-h |
|---|---|---|---|---|---|
| Planning (2,400 tok/s, 34 tok/s solitary decode) | 874 s | 320 s | 202 s | 2,022 s | 1.12 |
| Typical (2,750 tok/s, 70 tok/s) | 740 s | 227 s | 171 s | 1,640 s | 0.91 |

- **Recalculated, as the assessor asked.**
  - Rebuilding the warm prefixes independently per replicate gives 4 × 350,550 = 1.40M warm tokens.
  - That is about 146 s per replicate at the planning rate, and it dominates the work after start-up.
  - The earlier "about 0.6 GPU-h" assumed one shared warm and is superseded.
- **Wall caps:**
  - Stage 0: 2,250 s × 2 GPUs = **1.25 GPU-h**.
  - Pre-check: 300 s × 1 GPU = **0.083 GPU-h**; typical about 15 s.
- **Approval needed** for one pre-check attempt and one Stage-0 attempt: **APPROVED_GPU_H ≥ 1.34**.
  - The gate refuses an attempt whose full cap does not fit, so the approval must cover the caps, not the estimate.
  - Expected use is 0.92-1.13 GPU-h.

**Budget enforcement** (`runner/stage0/budget_stage0.py`):
- `APPROVED_GPU_H` must be set explicitly. There is no default, and nothing falls back to 7.0.
- An attempt is refused when its cap plus the GPU-h already in **`costs-stage0.jsonl`** would exceed the approval. That file is the new approval's own ledger.
- Run-1 `costs.jsonl` is neither read nor written.
- The run-1 runner (`runner/run_p4.sh` via `budget_p4.py`), its pre-check and its G4 refuse every step.
- Every Stage-0 attempt writes one cost line, counted to the GPU release.
- A Stage-0 attempt whose client completed is never re-run by the gate.

## 8. What Stage 0 cannot certify

1. **Outputs matching for 64 positions** are an operational gate. They do not certify every persistent-state byte, or positions 64-511.
2. **Digests are wrap-free integer sums** of the raw words.
   - A difference in digests proves a difference in bytes.
   - Equal digests make a difference very unlikely, but do not prove bitwise equality.
3. **Coverage of the KV-cache tensors is what `entry_init` lists.**
   - A tensor whose first dimension is not a multiple of `num_blocks` is not covered. It is named in the start-up record.
   - The null tail block (D2's target) is shared scratch: it is reported, not gated.
   - Allocator state, physical addresses and the scheduler's internal queues are not digested.
4. **D2 tail-ring races and allocation history are not removed.** The separate salt only prevents prefix reuse.
5. **The per-layer digests cover the target model's state at the forced requests' rows.** Draft-model hidden states are not digested; their KV caches are covered by the entry digests.
6. **The stage-0 recording itself changes the captured graphs and the step timing.** It is identical in both repeats, but it is not the run-1 graph or a production server. Instrumentation remains a candidate contributor.
7. **The multi-prompt submission does not guarantee deterministic scheduling.** G-enq tests the recorded execution, not the arrival times inside the engine.
8. **One batch of 8 windows** is a construction test, not a sample of all windows.
9. **Stage 0 does not establish a common A/B starting construction,** and it says nothing about the tile's effect.

## 9. Commands (operator; in this order; no GPU step without a new approval)

```bash
cd <runs>/tile-p4
# 0. CPU only (already run for this package; re-run any time):
OUT_OVERRIDE=$PWD/tests/out/stage0-cpu tests/run_cpu_tests.sh s0 s0image
# 1. single-GPU pre-check (cap 300 s = 0.083 GPU-h)
APPROVED_GPU_H=<approved> ATTEMPT_REASON="stage-0 pre-check" runner/stage0/run_precheck_s0.sh
# 2. Stage 0 (2 GPUs, cap 2,250 s = 1.25 GPU-h); the analysis runs automatically after the GPUs are released
ACK=$(sha256sum STAGE0-PROTOCOL.md | cut -c1-12)
APPROVED_GPU_H=<approved> ATTEMPT_REASON="stage-0 probe" S0_PROTOCOL_ACK=$ACK runner/stage0/run_stage0.sh
#    -> runs-stage0/stage0-attempt-1/{outcome.json, stage0-result.md}; PRIVATE-runs-stage0/ (0700)
# re-analysis (CPU container):  analysis/run_stage0_analysis.sh runs-stage0/stage0-attempt-1
```

Every GPU command accepts `DRY_RUN=1`: the gates run and the plan is printed; nothing is started or written.
