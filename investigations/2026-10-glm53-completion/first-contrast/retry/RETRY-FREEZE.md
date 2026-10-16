# First-contrast retry: frozen package under PLAN.md amendments 22d, 22f and 22h (CPU only; NOT launched)

2026-10-15. Design: `RETRY-PROPOSAL.md`. Changes are labelled R-n (`CHANGES-R.json`). This file maps each amendment-22d
requirement, and each amendment-22f correction (R-10 to R-15, applied before any retry data), to its implementation and
its test evidence. It also maps each amendment-22h change (R-16 to R-18, the operator's decision "both") to the same.
Nothing here authorizes GPU time.

For R-10 onward, `CHANGES-R.json` records before and after sha256 per file. The pre-22f files are kept beside the new
ones as `*.pre22f`, and the pre-22h files as `*.pre22h`.

How it was prepared:
- **No GPU.** No GPU or glm53* container; nothing used `--gpus`; no `sudo`; only DRY_RUN was launched.
- **CPU containers only**, all `--network none`, `CUDA_VISIBLE_DEVICES=` and the host uid:
  - the serving image (replay);
  - the B1 image (analyses, synthetic runs, F6 calibration, F8 regression, unit tests).
- **Attempt-1 files are unchanged** and imported read-only. `../SHA256SUMS` still verifies, and the retry runner
  checks it before any launch.
- **The hook, the image and the served configuration are unchanged** (environment as attempt 1, WMAX 8448). No D2-arm
  termination value was read.

## Pins

| What | Value |
|---|---|
| Serving image | `local/glm53-v0.9.1-b1-forcehook-mg2:local` sha256:ac05230bd94b… (`runner/images.txt`, as attempt 1) |
| Hook module | `vllm/forcewin_hooks.py` sha256 15e93e236067…, checked in the running container before traffic |
| Scheduler / KV-manager sources driven by the replay | In-container `v1/core/*.py`, `v1/core/sched/*.py`, `v1/kv_cache_interface.py`, `v1/kv_cache_spec_registry.py`, `v1/request.py`, `v1/outputs.py`, `config/vllm.py`, `config/speculative.py`, `v1/engine/core.py`: 22 files, byte-identical across the b0-mapguard, b1-mapguard and forcehook-mg2 images (`replay/out/image_sources.sha256`) |
| CPU analysis image | sha256:799d3cf8e5d1… (B1) |
| Attempt-1 package imported | `../SHA256SUMS` (186 files; amendment 22a) |
| This package | `SHA256SUMS`, see "Freeze" |

---

## Requirement → implementation → evidence

### 22d item 7 (F7): CPU replay [R-1]

| Requirement | Implementation | Evidence |
|---|---|---|
| Drives the image's ACTUAL KV-cache manager, coordinator and single-type managers: hybrid groups, 1,275-block shared pool, block 4,352, prefix caching with block-aligned lookups, mamba align, the DFlash / EAGLE last-block drop, `GLM53_DFLASH_BOUNDARY_LOOKUP=0` | `replay/engine_replay.py`: the image's `AsyncScheduler`, `KVCacheManager`, `HybridKVCacheCoordinator`, `FullAttention` / `KpoolTail` / `Mamba` / `SlidingWindow` managers and `BlockPool`, in a CPU container of the serving image. The config is built through the image's `generate_scheduler_kv_cache_config` | Placement summary equal to the logged line; coordinator reports block-aligned lookups (alignment 4,352), EAGLE groups and the 2-group spec split |
| First reproduce run-fc1 exactly: the one-block drop on 7 fixtures, M3's eviction at the recorded step, the sentinel's private recompute | `replay/trace_replay.py`, `reproduce_fc1.py` | **5,609 / 5,609 steps identical** (scheduled set and order, num_computed, tokens, drafts, finished, group-0/1 block ids, every group's list length, free blocks, guard fingerprint, live census); 32 / 32 cached counts; drop on 7 fixtures; M3's boundary state evicted at step 1,963 by C1's own admissions (M3 admitted fifth, lookup at 1,973); attempt-1 sentinel range in the privately recomputed block for 7 / 7 hit fixtures (`out/reproduce_fc1.json`) |
| The comparison is sensitive | `replay/neg_controls.py` | 4 / 4 wrong inputs detected: 1,274 blocks (step 1), salts dropped (1,165), probe max_tokens (654), boundary lookup 1 (1,961). The EAGLE flag on the KDA groups is not identifiable (identical) |
| Historical records' cached counts | `replay/historical.py` | b0 run-a3: 2,482 / 2,482 steps, 72 / 72 counts. d7 run-b1-a1: exact to step 2,440 (mismatches only inside phase 2d's aborted streams), 72 / 72 counts. phase2-stage2: 49 / 49 and phase3: 28 / 28 (model mode). **253 measured counts total** |
| Exercise the proposed session under plausible interleavings and resource pressure; conservative occupancy bounds | `replay/session_sim.py`, `run_scenarios.sh`, `schedule_report.py` | **25 design scenarios** (seeds, staggers, jitter, admission orders including reverse and random, pools 1,100-1,275, sequential and longest-first builds, long gate wait, A0) and 3 ablations. **Every lookup is a hit in all 25.** Survival margin ≥ 359 blocks (≥ 478 at 1,275). All-8 lead ≤ 95 primary, ≤ 159 LP (`out/F3_schedule.json`, `.out`) |
| Counterfactual on the real history | `replay/fc1_counterfactual.py` | Recorded admission order gives M3 a miss (reproduced); completion order gives 8 / 8 hits; reverse gives 2 misses |

### 22d item 1 (F1): the cache rule, scoped [R-2]

| Requirement | Implementation | Evidence |
|---|---|---|
| Corrected rule for exact repeats without eviction; the actual lookup elsewhere | `analysis/fcr_geometry.rule_c` and `expected_cache` (prefill coverage of the stream under its salt); every other case from the replay before the session and the recorded value during it | — |
| Tests: exact repeats, warm-prompt extensions, lengths below / at / above block boundaries, cold / partial / full eviction, the actual C, D, CV and sentinel forms | `replay/cache_rule_tests.py` (engine-driven, real fixture ids for T7) | **70 / 70 PASS** (`out/cache_rule_tests.out`) |
| Validated against measured historical records | The 253 replayed counts; unit U6 | 253 / 253 equal; the repeat rule holds on all 99 measured exact-repeat hits |

### 22d item 3 (F3): the feasible schedule under strategy (a) [R-3]

| Requirement | Implementation | Evidence |
|---|---|---|
| Actual request and warm order | `fcr_geometry.phases`: split builds (4 longest → touch → 4 shortest → touch all); LP runs on the primary builds before any primary decode; C2 / D2 across the gate; D3 / C3 | `F3_schedule.out` s.1, the base scenario's submission orders |
| Live and cached block occupancy by group per step | Occupancy sampled every 8 steps | `F3_schedule.json` occupancy_by_phase_base: batches peak at about 450 live, free ≥ 819; builds about 175 live, free ≥ 1,099 |
| Which boundary states survive each lookup | Tracker provenance per lookup | The KDA state at c in 7 groups and the MLA chain to index c / 4,352, written by the salt's build (prefill); LP second users also restore the first LP user's prefill |
| Expected hits and permitted misses | — | All 224 requests hit in 25 / 25 scenarios. Misses are permitted only after the gate (safe, balance-gated); none expected |
| All eight decode together before the lead-in ends | — | Lead ≤ 95 (primary) / 159 (LP) < 512 |
| Survival of the sentinels' shared state | — | PS-a / b / c, C2 and D2 always restore it; minimum margin 438 blocks |
| Adjacent C/D pairs tested for reduced churn | Shared builds (C2, D2) and (D3, C3) | Identical physical blocks (base 8 / 8); the across-gate share relies on an idle gate |
| Never 4-request batches | Plan rule Q10 | Every run holds the 8 fixtures |
| Partial-cache design if eviction cannot be eliminated | Not needed: eviction avoided in the replayed scenarios, with measured survival margins; the actual-session gates are authoritative [R-15] | Single-build ablation lost M3 at 1,250 blocks; LW builds lose M3: both rejected (`out/design_single_build/`) |

### 22d item 2 (F2): safety gates and construction accounting [R-2]

| Requirement | Implementation | Evidence |
|---|---|---|
| Block alignment; no hit beyond the permissible prefix; no decode-written history as clean prefill state; restored-state and request attribution | `analysis/cache_gate.py` S1-S4 from the client records and the guard's scheduler log; GATE-1R R-cache-safe; per run `<phase>-cache-safe`; post-session R-cache-safe | Units U1 (run-fc1 real records: 32 / 32 mapped, C1's 7 hits attributed to their own builds, M3 a safe miss) and U2a-f (each violation detected); synthetic r-cache-unsafe, r-cache-attrib |
| Hit / partial / miss per request | Class on every client record and gate row | R-cache-class |
| Construction differences reported, their distribution assessed, reconstruction variability in the clean null, no "identical prefix" claim | R-cache-balance (gated post-session), construction-matched sensitivity, build identity, pair wording | Synthetic r-balance → NO READING; good → C2 / D2 and D3 / C3 verified identical, C1 / D1 "NOT an identical prefix" |
| Cache behaviour does not differ by treatment | R-cache-balance | Unit U3b; synthetic r-balance |

### 22d item 4 (F4): submission [R-4]

| Requirement | Implementation | Evidence |
|---|---|---|
| Fixed staggered submission; actual admission and scheduling recorded; gates on properties, not timing | `client_fcr.py` (builds 1.0 s in the fixed order; runs 0.5 s in oldest-freed-first order; passes sequential; submission times and resp_id recorded); admission from the scheduler log | Plan rule Q16 (tampers T15, T16, T18); replay jitter and order scenarios; counterfactual |

### 22d item 5 (F5): sentinel [R-5]

| Requirement | Implementation | Evidence |
|---|---|---|
| Defined from the actual restored boundary c and the shared physical block [c − 4,352, c); block identity AND bytes; a missing shared-block measurement = incomplete coverage | `analysis/sentinel.py`: R-PS-coverage, -identity (scheduler ids, writer's id, dump block tables), -bytes (MLA and index-K pages), -state-ssm and -state-conv (R-10, below), -limits (R-13); in-session rc 14 for PS-c; plan rule Q15 | Unit U4 (all 8 fixtures); tamper T14; synthetic a-ps (byte flip) → R-PS-bytes; r-ps-cover → R-PS-coverage; r-ps-ident → R-PS-identity |

### 22d item 6 (F6): numerical criteria [R-6]

| Requirement | Implementation | Evidence |
|---|---|---|
| Frozen matched long-context null comparisons (prefill vs prefill, plus the in-session decode null) by fixture × layer × page and pooled | `analysis/f6_criteria.py`: R-idx-pages and R-idx-early against the matched prefill null (N_dd, which contains the tested run, is reported beside them); pooled G1-content with both nulls; tables and flags | Unit U9; `calibration/f6_calibration.json`; synthetic r-f6-page (a one-run page fault that max(N_dd, N_pp) would have absorbed) |
| Exact structural requirements kept | R-idx-struct plus the unchanged attempt-1 structural rows | Synthetic g-* variants |
| Flags and leave-flagged-out; originals and failures kept beside; the wording | G1-idx-struct-orig reported; R3 flags; R2 sensitivity; `f6.WORDING` | Calibration: original FAIL (8 pages, 3 early) beside the new PASS |
| Planted wrong-page, foreign-request and stale-state controls still detectable, on run-fc1's real C1 / LW1 / LW2 | `f6_criteria.py` main (calibration only, N_dd absent) | **6 / 6 planted faults detected**: wrong page ×3, foreign request, stale state, unwritten page. The pooled rule alone catches 1 / 6 |
| Gate-asrun kept as the F8 fixture; C1 shown on the new checks, labelled post-inspection, not pooled or paired | `tests/test_f8_regression.sh` | Frozen `gate_fc.py` re-run equals `gate-asrun/gate1.json` (27 rows, FAIL). C1 post-inspection: F2 safe, F6 pass, 9 attempt-1 rows passed (`calibration/f8_regression.json`) |

### 22d item 9: integrated gate ordering [R-7]

| Requirement | Implementation | Evidence |
|---|---|---|
| References and clean checks before injected traffic; stops on unsafe restoration, insufficient concurrent decoding, alignment failure, uninterpretable numerical content, missing boundary or shared-prefix coverage | `client_fcr.py` (rc 9 / 12 / 13 / 14 / 10), `gate_fcr.py` STOP_CLASS, `run_fcr.sh` pre-traffic geometry check (rc 16) | Plan rule Q11 (tampers T6, T8); synthetic gate variants report their stop class; unit U5 (geometry) |

### 22d item 8: runner, client, analysis, tests, DRY_RUN [R-8]

| Requirement | Implementation | Evidence |
|---|---|---|
| No change to B1, the hook or the served config; WMAX 8448 | `run_fcr.sh` (DEBUG environment as attempt 1) | Runner diff scope in its header |
| Refusal and tampered-plan tests | `tests/test_runner_fcr.sh`, `tests/test_plan_fcr.py` | See the test summary |
| Synthetic and real-data analysis tests | `tests/make_synth_fcr.py` (attempt-1 generator plus explicit patches), `run_all_synth_fcr.sh`, `check_synth_fcr.py`; `test_units_fcr.py`; F8 | See the test summary |
| DRY_RUN | `DRY_RUN=1 run_fcr.sh` | `DRYRUN.out` |

### 22d deliverable: costing, ceiling, attempt policy [R-9]

| Requirement | Implementation | Evidence |
|---|---|---|
| Measured-rate costing including expected misses, cold work and early-stop costs; explicit ceiling and attempt policy | `cost/cost_fcr.py` → `cost_fcr.json`, `.out`; runner constants equal (unit U8) | B 3.00 / 4.18 / ceiling **4.18**; A0 2.75 / 3.72 / **3.72** GPU-h; early-stop table; policies single and invalid-start |

### Amendment 22f corrections [R-10 to R-15] (before any retry data)

| 22f item | Implementation | Evidence |
|---|---|---|
| s.2 Sentinel: SSM and convolution evaluated separately, each with its own PS-a / PS-b baseline; each PS-c comparison meets its own criterion; only the intended committed state and the relevant conv entries; analysis only, no hook change [R-10] | `analysis/sentinel.py`: `committed_ssm`, `relevant_conv` (positions P − 3 .. P − 1 via `fc_common.conv_entries`), `state_rows`; rows R-PS-state-ssm and R-PS-state-conv replace R-PS-state. A frontier other than P − 1, or a missing component, fails. Hook unchanged (15e93e23…) | `tests/test_sentinel_fcr.py` on REAL attempt-2 KDA prefill snapshots: **14 / 14**. Clean 5: identical; non-identical baseline within 1.5 ×; unwritten rows only; dim-first layout; a non-committed slot. Negative 9, each failing exactly its component: conv-only; SSM-only; non-identical SSM baseline then excessive conv; non-identical both then excessive conv; then excessive SSM; committed slot; wrong frontier; missing conv; dim-first conv-only. The pre-22f module passes 4 of the 9 negatives (N3, N4, N7, N9). Synthetic end to end: r-ps-conv → R-PS-state-conv; r-ps-ssm → R-PS-state-ssm; r-ps-nonid → R-PS-state-conv; r-ps-nonid-ok → clean reading (`tests/out/sentinel-tests.log`, `synth-check.log`) |
| s.3 Acceptance stress: single-token, maximum and strongly unequal advance, primary and LP batches; cache survival and all-8 before the lead-in ends; runtime in the cost; whether 4.18 holds [R-11] | `replay/session_sim.py` acceptance modes (default unchanged); `replay/run_stress.sh` (10 scenarios: single, max, unequal-a / -b at 1,275 and 1,150 blocks, unequal with reverse admission) and a recheck; `replay/stress_report.py`; `cost/cost_fcr.py` stress section with the client's own time rules under the B wall cap; `schedule_report.py` excludes stress runs from the 25-scenario table | The recheck equals `sim_base.json` exactly. **All 10: every lookup a full hit**; minimum margin 478 blocks at 1,275, 393 at 1,150 (primary 393, LP 409, sentinel 409); max lead 93 primary, 193 LP, < 512. C / D batches 674-694 s (single / unequal) against about 320 s. **4.18 does NOT hold** for single or unequal advance: complete session 4.33-4.40 central, 6.08-6.18 pessimistic. Under the cap, rc 11 before C3 (central) or a cut during D1 (pessimistic), primary incomplete. Max advance: 2.52 / 3.50, holds. Break-even 17.6 tok/s per request. Base at pessimistic rates also stops before C3 under the client rules (pre-existing). Reported, not changed (`replay/out/F3_stress.out`, `cost/cost_fcr.out`) |
| s.4 F6 pages whose null makes the threshold non-discriminating reported, not called independently verified; a pass = compatibility with measured long-context variation [R-12] | `analysis/f6_criteria.py`: bound 0.5 on the page limit, defined before data (limit ≥ 1.0 also flagged); `page_rule` reports them; GATE-1R row R-idx-pages-nd, `gate1r.json` `not_independently_verified` and a `gate1r.txt` line; `fcr_result.json` / `.txt` for GATE-1R C1 / C2 and C3 | Units U10a-b; calibration rerun: run-fc1 results otherwise identical, **7 of 2,486** C1 pages non-discriminating (all M4, limits 0.56-0.91), the planted M4 fault still detected on 3 of them; synthetic r-f6-nd: GATE-1R PASS, 2 pages reported for C1, C2 and C3 |
| s.5 Sentinel limits stated [R-13] | `sentinel.LIMITS`; row R-PS-limits; `gate1r.json` / `.txt`, `fcr_result.json` / `.txt`, notes; RETRY-PROPOSAL s.7, s.12 | Unit U12; synthetic content checks on every variant |
| s.6 LP same-depth comparison (LC1 / LD1, LC2 / LD2), LC3 / LD3 separately as depth-mismatched; no three-repeat rule on the matched pairs; "unresolved construction sensitivity" if the conclusion depends on the mismatched pair [R-14] | `analysis/analyze_fcr.lp_same_depth`; dependence rule fixed before data (beyond-null status or sign of LD − LC and of the interaction, recomputed with the matched mean) | Units U11a-d (independent; beyond-null needing the mismatched pair; null produced by it; incomplete pair); synthetic good / null / r-f6-nd / r-ps-nonid-ok carry `same_depth` |
| s.1 Wording: "eviction avoided in the replayed scenarios, with measured survival margins"; the actual-session gates authoritative [R-15] | RETRY-PROPOSAL s.2.2, s.4.3; this file; `analyze_fcr.CACHE_WORDING` in `fcr_result.json` / `.txt`; `F3_stress.out` | Synthetic content checks |
| s.7 Recommended GPU scope | Unchanged: variant B, ceiling 4.18, single attempt, existing early-stop gates, no automatic retry | Runner constants = cost model (U8). The 22f s.3 stress result above is the operator's to weigh |

### Amendment 22h changes [R-16 to R-18] (before any retry data; runner, client and costing only)

| 22h item | Implementation | Evidence |
|---|---|---|
| s.1 Ceiling 6.2 occupied GPU-hours, variant B, single attempt, no automatic retry; 4.18 values recorded as superseded [R-16] | `runner/run_fcr.sh`: B CEILING=6.2, WALL_MAX=11160 (6.2 × 3,600 / 2), with the superseded 4.18 / 7,523 in the line and header. `cost/cost_fcr.py`: B ceiling = the operator's decision, the computed 4.18 under `superseded`. The R-11 section is still evaluated against the superseded values, as reported under 22f. A0 unchanged | U8 (runner = cost model: 6.2 / 11,160 / 4,942); U13g. Runner refusals 27 / 27, including: DRY_RUN B at 6.2; budget line "cap 6.2 (wall 11160s)"; 6.7 refused (above 6.2 + 0.46); 6.21 and 6.2001 refused without invalid-start; 4.18 and 5.0 accepted with lower caps; A0 still 3.72 |
| s.2 Batch guard: a batch starts only if its projected completion fits in the remaining budget and wall time, from the session's measured decode rate with a conservative margin; never cut mid-way; a guard stop recorded as an incomplete contrast [R-17] | `runner/batch_guard_fcr.py` (pure; also used by the cost replay); `client_fcr.py` calls it before every build, touch pass, sentinel pass and run batch. Records `guard` / `rate`, rc 17; rc 11 and the 25 tok/s never-truncate rule superseded. `run_fcr.sh`: the client timeout coincides with the hard watchdog. `analyze_fcr.py`: reading "incomplete contrast (batch guard)", NOT RUN rows, `batch_guard` block | Units U13a-f: first-batch prior; fits → start; does not fit → stop before (rc 17); measured rates replace the prior per form; the slowest rate counts; the first D batch uses the prior; LP reserve incl. GATE-1R, skip and LP-off propagation. Synthetic r-guard-stop: GATE-1R PASS, reading "incomplete contrast (batch guard)", C3-residue NOT RUN, no FAIL row. Plan unchanged (plan check, 19 / 19 tampered plans) |
| s.3 Stress replay under 6.2 + guard, all 10 scenarios and the base, central and pessimistic [R-18] | `cost/cost_fcr.py` `guard_timeline` runs the guard's own code on every modelled timeline under the 11,160 s wall cap | **Never mid-phase; no guard stop in any case.** Base: complete with LP, 3.00 GPU-h central (expected spend unchanged), 4.18 pessimistic (no longer stops before C3). Max: complete, 2.52 / 3.50. Single: complete, 4.33 central; pessimistic primary complete with LD1-LD3 / LC3 skipped, 5.64. Unequal (6): complete, 4.39-4.40; pessimistic primary complete with LP after the gate skipped, 5.73. Actual / projected ≤ 0.78, except the first batch of a form at pessimistic stress rates, ≤ 1.04, with ≥ 4,400 s of slack (`cost/cost_fcr.out`) |

---

## Test summary (all CPU)

| Suite | Result | Log |
|---|---|---|
| F7 run-fc1 reproduction | 5,609 / 5,609 steps exact; 32 / 32 counts | `replay/out/reproduce_fc1.json` |
| F7 negative controls | 4 / 4 detected (1 non-identifiable assignment reported) | `replay/out/neg_controls.jsonl` |
| F7 historical sessions | 253 / 253 counts; b0 exact; d7 exact to phase 2d | `replay/out/historical.json` |
| F7 counterfactual (admission order on run-fc1's history) | 3 / 3 as stated | `replay/out/fc1_counterfactual.out` |
| F7 / F3 scenario sweep | 25 / 25 design scenarios all hits; 3 ablations | `replay/out/F3_schedule.out` |
| F1 cache-rule tests | 70 / 70 | `replay/out/cache_rule_tests.out` |
| F6 calibration and planted faults | C1 passes the new rules; 6 / 6 faults detected; 7 non-discriminating pages reported [R-12] (results otherwise identical to pre-22f) | `calibration/f6_calibration.out` |
| F8 regression | identical (27 rows); C1 post-inspection recorded; re-run after 22f, byte-identical outputs | `calibration/f8_regression.out` |
| Unit and real-data tests | 39 / 39 (25 + U10a-b, U11a-d, U12 + U13a-g) | `tests/out/units.log` |
| Batch guard under 6.2, stress replay [R-18] | base + 10 scenarios × central / pessimistic: never mid-phase, no guard stop, primary complete in every case | `cost/cost_fcr.out` |
| Sentinel boundary-state tests [R-10] | 14 / 14 (5 clean, 9 negative; the pre-22f module misses 4 of 9) | `tests/out/sentinel-tests.log`, `sentinel_tests.json` |
| Acceptance stress scenarios [R-11] | 10 / 10 all hits, all-8 leads < 512; recheck = base exactly; 4.18 not held under single / unequal advance (reported) | `replay/out/F3_stress.out`, `cost/cost_fcr.out` |
| Plan check on the session plans | B: 224 requests, 0 failures; A0: 176, 0 | `tests/out/plan-check.log` |
| Tampered plans | 19 / 19 (untampered passes; 18 refused with their rules) | `tests/out/plan-tamper-tests.log` |
| Runner refusals with shims | 27 / 27 (DRY_RUN B at 6.2 and A0, budget lines, approval and policy refusals at the 6.2 ceiling, lower approvals, A0 unchanged, attempt rules incl. invalid-start, min wall, variant, image pin, disk, both frozen-hash gates, ids tamper, existing run dir) | `tests/out/runner-tests.log` |
| Synthetic dry runs (attempt 1's 20 variants under the retry pipeline + 7 retry plants + 5 22f plants + r-guard-stop) | 33 / 33 behave as expected (GATE-1R verdict, exactly the planted rows, reading class, and output content: limits, cache wording, non-discriminating pages, LP same-depth) | `tests/out/synth-check.log` |
| Final DRY_RUN | B with APPROVED_GPU_H=6.2, and A0, pass (line and SHA256SUMS sha256 in DRYRUN.out) | `DRYRUN.out` |

## Freeze

`SHA256SUMS` (`make_sums.sh`) covers every design, code, replay, calibration, cost and evidence file. It excludes:
- the item ledger `costs.jsonl`;
- session outputs `run-fcr*/` and `DRYRUN.out`;
- bytecode;
- the synthetic runs' raw data (only each variant's `gate1r.json`, `fcr_result.json` and logs are hashed).

Its own sha256 and the final DRY_RUN line are in `DRYRUN.out`. Attempt-1 files are pinned by `../SHA256SUMS`, and the
runner verifies both.

**22h re-freeze.** The 22f freeze (358 files, sha256 0b1c7927…) is kept as `SHA256SUMS.pre22h` and
`DRYRUN.out.pre22h`. The files changed by R-16 to R-18 have `*.pre22h` copies; before and after sha256 are in
`CHANGES-R.json`.

**22f re-freeze.** The pre-22f freeze (amendment 22e: 287 files, sha256 c7374d75…) is kept as `SHA256SUMS.pre22f`,
with its DRY_RUN line in `DRYRUN.out.pre22f`. Every file changed by R-10 to R-15 has a `*.pre22f` copy, and
`CHANGES-R.json` gives the before and after sha256. The regenerated `SHA256SUMS` also covers the `*.pre22f` copies,
the new tests, the stress outputs and their logs. Its sha256 is in `DRYRUN.out`.

## Not met, limits and deviations

1. **KDA block identity** is not available from the guard's records. The sentinel certifies the boundary state by
   restored length plus snapshot bytes. Exact KDA ids would need a guard change (not made).
2. **The replay is step-exact on every recorded session except** phase 2d's aborted streams in d7 (cached counts still
   exact). The EAGLE flag on KDA groups is behaviour-equivalent and not identifiable.
3. **GATE-1R idles the GPUs** (270 s central, 600 s pessimistic; priced) to keep s2's build for D2.
4. **LP restore depth** differs between the LP arms (LC 1·c + 2·c_L, LD 2·c + 1·c_L); reported, both prefill
   construction.
5. **The pessimistic cost** assumes one cold fixture in every batch although the replay predicts none. The ceiling is
   the pessimistic total (4.18 B, 3.72 A0), below amendment 22's 4.29.
6. **The long-context numerical gates** are first in-session measurements with N_dd. The F6 calibration had no N_dd
   (C2 never ran).
7. **The decode acceptance model** comes from C1 of run-fc1 only; step timing is modelled to within about 5 %. The
   22f stress scenarios bound acceptance from both sides for cache survival and lead-in [R-11].
8. **Superseded by 22h:** the 4.18 ceiling did not complete the session under single-token or strongly unequal advance
   (RETRY-PROPOSAL s.11.1). Under 6.2 with the batch guard the primary completes in every stress scenario (s.11.2). At
   pessimistic stress rates the LP batches after the gate are skipped, so there is no LP reading.
9. **Sentinel limits (22f s.5):** MLA / index-K shared-block identity is directly checked; KDA restoration and
   captured-state consistency are checked; physical KDA block identity and the complete unsampled state are not
   established.
10. **LP attribution** is limited to the descriptive same-depth pairs [R-14].
11. **Non-discriminating F6 pages** are reported, not called independently verified [R-12].
12. **The batch guard's decode prior** (11.0 tok/s per request) is below the central single-token rate but above the
    pessimistic one (about 8.4). A first batch of a form can run a few percent past its projection; the session's
    early slack covers this. The hard watchdog stays the last resort and is the only thing that can cut a phase.
