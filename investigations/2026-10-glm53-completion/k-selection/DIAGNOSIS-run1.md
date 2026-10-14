# K selection validation, run 1: diagnosis of the 0.7.0 ERROR and the `persp` FAILs

2026-10-13. CPU only. No GPU was used, no GPU or glm53 container was started or stopped, and no `sudo` was run.
CPU containers ran from the pinned images with `--network none` and no `--gpus`. The frozen files in `k-selection/`
are unchanged (`sha256sum -c SHA256SUMS` passes). Scripts and outputs are in `diagnosis-run1/`.

Run: `run-20261013T234310Z`.

- **B1:** 34/34 PASS. Nothing below affects that result.
- **0.7.0:** 2 FAIL (`persp`), 1 ERROR, 19 NOT-RUN. The run stopped after 40.6 s.

## Summary

| Item | Finding | Class |
|---|---|---|
| ERROR in `v070/tiled48/b512` | The plan contains a row with token length **L = −1**, so P = −1 (not 0). `if pj:` lets −1 through, and `s[j, :-1]` on a one-pool request is empty. | Harness/plan defect. It reveals no engine condition. |
| 19 cells NOT-RUN | The per-cell `try/finally` re-raises the exception, and the outer `except Exception` ends the cell loop. | Harness defect (error handling). |
| `persp` FAIL, 42 + 42 | The count is **42 of 42 batch × repeat records** in every family, not 42 of 2,016 rows. Both perspectives fed `run_row_topk` bit-identical inputs. The kernel's output order is set by shared-memory atomics, so two launches order the result differently. | **Not a stub artefact.** It is launch nondeterminism, which the real engine would also show across its two GPUs (detail below). |

The `persp` FAIL stands: C5 requires bit-identical perspectives, and they are not identical. The criterion is not
loosened here. Under the proposal's own ordering analysis, most of the difference does not matter to the consumer.
One part does: **column 511**, which stock 0.7.0 drops separately on each rank.

## 1. The ERROR

**Cause.** `kplan.targets` (`kplan.py:177`, `:183`) sets the short control's lowest target as
`max(1, 2·512 − 1024) = 1` pool, so Llast = 4. For 0.7.0 (K = 5), `row_lengths` (`kplan.py:204-209`) gives that
request the rows L = **−1**, 0, 1, 2, 3, 4.

In `run_k.py:243`, `P = L // 4` floors to **−1**. At `run_k.py:263-266`:

- `pj = −1`, which is truthy, so `poison_rows += 1`;
- `s[0, :-1]` on the request's (6, 1) score matrix is empty, so `max()` raises.

Three things confirm this:

- The cell's `summary-v070.json` records `poison_rows: 1`, exactly one increment before the raise.
- `diagnosis-run1/validate_fixes-frozen.out` (E1) reproduces the exception on CPU: the frozen `run_batch` on
  `v070/tiled48/b512` batch 0 gives `RuntimeError at run_k.py:266` before any driver call.
- The failure happens in the fp64-reference phase, so no kernel ran.

**Scope.** `diagnosis-run1/diag_rows.out`:

- All four 0.7.0 short cells (tiled48, tiled24, fused16, fused8 `/b512`) contain the L = −1 row and an L = 0 row of a
  real request.
- B1 is not affected: with K = 3, Llast = 4 gives L = 1..4.
- The plan's coverage assertion only looks at rows with `L > 0` (`kplan.py:227`), so the plan builder did not catch it.
- The CPU suite did not catch it either: T8 and T9 use target slices `[2:8]` and `[3:6]`, which skip the clamped
  minimum.

**Harness or engine?** Harness. A decode row has L = position + 1 ≥ 1, so the engine never forms a negative length.

Guarding only the descriptive counter would not have been enough. On rank 1 the same row gets a local seq_len of −1
(`local_pools(−1, 2, 1) = −1`). That is undefined for the kernel, and `row_ref` would have checked it against
`s[j, :-1]`. The invalid row has to be removed from the plan.

**Why the other 19 cells did not run.** At `run_k.py:190-209` the per-cell `finally` records ERROR, but the exception
propagates. The outer `except Exception` at `:212-214` then leaves the loop, and `:216-219` marks every remaining cell
NOT-RUN. The REPORT legend says "NOT-RUN = time guard", which is misleading here: `stopped_by_time_guard` is false.

**Minimal fix.** The changes are post-inspection changes, recorded and re-hashed; see
`diagnosis-run1/post-inspection-fixes.diff`. No criterion and no `verdict()` logic changes.

- **PI-1 (plan).** `kplan.targets` uses the lowest target `cdiv(K+1, KPOOL)` instead of 1, so every request row has
  L ≥ 1. `kplan.build` asserts this. Regenerating `plan.json` changes **only the four v070 short cells** (Llast 4 → 8
  and 5 → 9). The other 52 cells and `meta` are byte-identical as JSON, and `kplan.out` is unchanged. `run_batch`
  raises a plan error if a request row with L < 1 ever appears.
- **PI-2 (counter).** `run_k.py:264` changes `if pj:` to `if pj > 0:`. The counter is descriptive only.
- **PI-3 (isolation).** Each cell gets its own `except Exception`, which records verdict ERROR plus the traceback and
  continues with the next cell. The time guard still stops the run. If the CUDA context no longer synchronises after a
  cell error, the run stops and the remaining cells are NOT-RUN.

**Validation on CPU** (`diagnosis-run1/validate_fixes-*.out`):

| Check | Frozen code | Patched code |
|---|---|---|
| E1: short-cell batch | `RuntimeError at run_k.py:266` | Reference phase completes and the first driver call is reached |
| E2: middle cell faulted | [PASS, ERROR, NOT-RUN], reproducing run 1 | [PASS, ERROR, PASS] |
| Full CPU suite (patched code, regenerated plan) | | **71/71** in both images (`diagnosis-run1/cpu-tests-patched/`) |

## 2. The `persp` FAILs

### What was observed

**The 42 is a count of records, not rows.** `persp` is bumped once per batch × repeat when the whole (48 × 512)
tensors differ (`run_k.py:371-374`). Each cell has 2 batches × 7 families × 3 repeats = 42 records. Every one of them
differed, in every family, the exact ones included (`results-v070.jsonl`: 84/84 records have `persp_equal: false`).

The rest of the record is clean:

- `allgather_ok` is true in 84/84 records.
- No other counter fired: C1, C2, C2b, the per-rank checks and C4 all passed on perspective 0.

**The run did not keep the differences.** `failures-v070.pt` is an empty list. `persp` is not a row flag, and only
rows with row flags are stored (`run_k.py:407-411`). Perspective 1 is never stored, and C1/C2 are never applied to it
(`run_k.py:390` checks `persp[0]` only). So which ids or columns differed **cannot be read back** from run 1. Below,
the kernel source settles where the difference comes from, and a CPU model of the kernel's claim structure gives the
per-row exposure. PI-4 records the actual differences in the rerun.

### Are the two perspectives' inputs the same?

Yes, bit for bit.

- **Engine.** `_merge_b12x_dcp_topk` (`sparse_attn_indexer.py` v070:1139-1218):
  - converts local ids to global and packs `candidates` (:1163-1188);
  - calls `_dcp_all_gather_first_dim_into`, which uses PyNCCL `all_gather` (:421). NCCL concatenates in rank order,
    so slot r holds rank r's candidates;
  - unpacks rank-major into a 1,024-wide row (:486-516, rank r → columns r·512..r·512+511);
  - runs `run_row_topk` (:1207-1213) and gathers by position (:1214-1218).
- **Stub.** `kdrive.py:125-131` writes the call's own candidates to slot r and the other rank's to slot 1 − r, which
  is also rank order. It records whether the own slot equals the independently produced candidates. That check was
  true in 84/84 records.
- **The rest of the call.** The workspace pattern is the same for both perspectives. With 1,024 > 512 candidates the
  kernel takes the radix path and writes all 512 output slots (`tiled_topk.py` v070:1180-1240), so the
  perspective-specific initial contents of the output buffers are overwritten.

So the stub is faithful, and **the difference is not a harness artefact**.

### Where the difference comes from

`run_row_topk` is a port of sglang's radix select (`tiled_topk.py` v070:1-9, :1593). Every selected candidate gets
its output slot from a shared-memory atomic counter, `pos = _smem_xadd(ctr, 0, 1); s_out[pos] = idx`, in each pass:

- coarse pass, :894-895, :913-914;
- buffered pass, :946-947, :973-974;
- refine rounds, :1076-1077, :1114-1115.

When the 512th and 513th keys are exactly equal, the tied candidates fill the last slots from the end through a
second counter, `old_rem = _smem_xadd(lr, …); s_out[topk_static − old_rem]` (:1044, :1118-1124). With 1,024 threads
in 32 warps, the order of the atomic claims depends on scheduling. So:

- **Order:** the output order differs from launch to launch. This explains why all 42/42 records differ.
- **Set:** the selected set is deterministic except among candidates whose fp32 keys equal the boundary key. Those
  take the round-3 tie fill, and which of them win is a race.
- **Column 511:** this is the last slot claimed. It belongs to the final claim group, which is the bottom G pools of
  the selected set (or, on the tie path, one of the tied pools).

The CPU suite could not see any of this, because T5 and T8 stub `run_row_topk` with `torch.topk`, which is
deterministic and sorted (`test_cpu.py:435-441`).

### Model of the exposure

`diagnosis-run1/diag_persp.py` and `diag_persp.out` rebuild the exact inputs of both FAIL cells (same seeds and code)
and model the deterministic part of the claim structure per row. The model covers the coarse bin, the threshold
search and the four 8-bit rounds, with their quick exits and the tie path. It cannot model the atomic order.

- **Exact families** (plateau, zero, ulp, edge): the model is exact, because their kernel scores were bit-exact.
- **real_L3, real_L43:** fp64 → fp32 is used as a proxy for the kernel's scores.
- **gauss:** not modelled. Its GPU inputs come from a CUDA generator.

| Family (96 rows per cell) | Final claim path | Column 511 race-exposed (group > 1), b16384 / b32768 | Set can differ (each cell) |
|---|---|---|---|
| real_L3 | mostly quick exit in round 2 (median group 10) | 86 / 87 | 0 (no exact tie at the merged boundary) |
| real_L43 | round 1-2 (median 5-8); 2 rows coarse-quick (group 512) in b16384 | 81 / 86 | 0 |
| plateau | round-3 tie: 38-40 pools tied at t for 12 slots | 96 / 96 | 96, ties only |
| zero | round-3 tie: 724-743 exact zeros for 212-231 slots | 96 / 96 | 96, ties only |
| ulp | quick exit in round 3, group 8 | 96 / 96 | 0 |
| edge | round 2-3, group 24 | 52 / 52 (of 96) | 0 |
| **All six** | | **507/576 (b16384), 513/576 (b32768)** | **192/576**, every one among exact ties at t (accepted by C2) |

**Cross-check against the run's own GPU counter.** `stock_col511_rank_lt480` was measured on perspective 0 on the
GPU: 313 and 307 row-repeats. The model predicts:

- exactly 288 from the zero family, where column 511 is a tied zero ranked about 296;
- about 19 and 11.5 from the real families;
- leaving 6 and 7.5 of 288 row-repeats for gauss.

The model is consistent with the observed counts (`diagnosis-run1/crosscheck_col511.out`). The check is not
decisive, because the zero family dominates it.

### Does the order matter to the consumer?

For L ≥ 2,048 the MLA does not depend on order (PROPOSAL §6). The tail columns come from L, not from the order. What
matters is the per-rank truncation `pool_ids[:, :511]` (`sparse_attn_indexer_kpool.py` v070:1008-1009).

Each rank runs its own merge on identical gathered candidates and then truncates its own list. The MLA then keeps only
that rank's own tokens from that rank's list:

- `triton_filter_and_convert_dcp_index`, `b12x_mla_sparse.py` v070:2753-2779;
- ownership rule `owning_rank = tok % 2`, `sparse_utils.py`:74.

A pool's four tokens sit two on each rank. So if rank 0 drops pool A and rank 1 drops pool B, the attended set is 510
whole pools plus half of A and half of B. No single rank's list gives that set. The CKV-gather path
(:2709-2752) also consumes each rank's own list.

### Classification

1. **Harness artefact: no.** The inputs are bit-identical, and the stub reproduces NCCL's rank-order layout.
2. **Order-only in columns 0..510: most of each difference.** It is irrelevant to the consumer. On this part the
   bit-identity C5 criterion is stricter than the frozen intent in §6, but it stays as frozen.
3. **Set differences: only among exact ties at t.** That is plateau and zero, 192/576 modelled rows per cell, with up
   to 12 or about 220 swapped pools respectively. C2 accepts any tied subset, so these are tie-order differences.
4. **Column 511: a genuine 0.7.0 cross-rank inconsistency.** It is bounded to the final claim group, the lowest
   median 5-10 selected pools for real inputs, but any of the 512 in coarse-quick rows. In the shipped engine the two
   ranks can drop different pools, giving two half-dropped pools instead of one whole one. It is new in that it
   differs per rank. It sits inside the already-known 0.7.0 column-511 class (stage 2) and is not a set-selection
   defect.

How often the two ranks actually pick different pools is **not measured**. The model gives exposure, not outcomes.
PI-4 measures it.

**PI-4 (recording only).** No new criterion and no change to `verdict()`.

- On a perspective mismatch, each row is classified, with counts per family: order-only (same set, same column 511);
  set differs (tie or non-tie at t, or in the fp32 band for the general families); column 511 differs (tie or
  non-tie, with the min rank of the differing pools).
- Both perspectives of the first 120 differing rows are kept in `failures-v070.pt`.
- A **relaunch control** reruns the same merge on the same inputs (`persp0_relaunch_equal`). If perspective 0 differs
  from its own relaunch, the difference is shown directly to be launch nondeterminism rather than a rank effect.
- `report.py` prints the breakdown and labels `persp` as a count of batch × repeat records.
- Synthetic checks (E3) classify an order-only pair, a non-tie column-511 swap and a tie-only set swap correctly, and
  leave the verdict counters at 0.

## 3. Rerun: 0.7.0 only

**Scope.** All 22 v070 cells, including the two FAIL cells, which are rerun with PI-4 recording. B1 is not rerun: its
34/34 PASS was produced with the original hashes, and none of the B1 computation changes.

The post-inspection changes are PI-1 to PI-5:

- PI-1 regenerates `plan.json`.
- PI-5 is in the runner: `K_ENGINES` defaults to `v070`, the v070 container cap is 1,200 s, `WALL_CAP_S` is 1,260 s,
  and `CEILING_GPU_H` is 0.35.

Record them in PLAN.md, write a new `SHA256SUMS`, and re-run the refusal and cleanup tests with the shims before
the GPU stage. The patched hashes are in `diagnosis-run1/patched-sha256.txt`. Verdicts stay unchanged in meaning: C5
will very probably FAIL again in every v070 cell, because the kernel's order nondeterminism does not go away. The
reading must then quote the PI-4 breakdown (order / tie / column 511) next to the verdict. It must not reclassify the
verdict.

**Cost** (`diagnosis-run1/rerun_cost.py`, `rerun_cost.out`). The estimate uses run 1's measured rates:

- 0.7.0 tiled checks: 4.6 ms per row;
- 0.7.0 tiled kernels: 31 ms per batch × repeat;
- B1 fused kernel: 85-125 ms per call. A 0.7.0 fused batch × repeat makes two calls; this is assumed, not measured
  on 0.7.0.

The plan has 3,171 batch × repeat records and 45,864 rows.

| | GPU-hours |
|---|---|
| Expected | 0.26-0.32 (about 15-19 min wall) |
| Pessimistic (×1.5) | 0.475 |

The proposal's estimate for 0.7.0 (273 s) was low by about 3-4×, because the fused kernel calls dominate.

The frozen 850 s container cap (inner time guard about 775 s) would leave the last fused cells NOT-RUN, hence PI-5:

- The 0.35 GPU-h ceiling keeps the item within the original approval: 0.1417 + 0.35 = **0.4917 ≤ 0.5 GPU-h**. The
  budget gate needs `APPROVED_GPU_H=0.5`.
- In the pessimistic case the last fused8 cells end PARTIAL or NOT-RUN under the time guard. They are reported, not
  dropped.
- If the operator prefers that every cell run to completion, a cumulative approval of about 0.65 GPU-h with a 0.5
  ceiling would cover it.

## Files (`diagnosis-run1/`)

| File | Contents |
|---|---|
| `diag_rows.py`, `.out` | Rows with L ≤ 0 per cell; the PI-1 plan delta |
| `diag_persp.py`, `.out`, `diag_persp_rows.json` | CPU model of the merge's claim structure per row (FAIL cells; 6 families) |
| `crosscheck_col511.py`, `.out` | The model against the GPU-measured column-511 counter |
| `validate_fixes.py`, `validate_fixes-{frozen,patched}.out` | E1 reproduction and fix; E2 cell isolation; E3 PI-4 classification |
| `post-inspection-fixes.diff` | PI-1 to PI-5 against `kplan.py`, `run_k.py`, `report.py` and `run_k_selection.sh` |
| `patched-sha256.txt` | Hashes of the patched files and the regenerated `plan.json` |
| `cpu-tests-patched/` | CPU suite on the patched code: 71/71 in both images |
| `rerun_cost.py`, `.out` | GPU-time estimate for the 0.7.0-only rerun |
