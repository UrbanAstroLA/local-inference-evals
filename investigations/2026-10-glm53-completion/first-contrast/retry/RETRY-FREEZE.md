# First-contrast retry: frozen package under PLAN.md amendment 22d (CPU only; NOT launched)

2026-10-15. Design: `RETRY-PROPOSAL.md`. Changes are labelled R-n (`CHANGES-R.json`). This file maps each amendment-22d
requirement to its implementation and its test evidence. Nothing here authorizes GPU time.

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
| Partial-cache design if eviction cannot be eliminated | Not needed: eviction eliminated with margin | Single-build ablation lost M3 at 1,250 blocks; LW builds lose M3: both rejected (`out/design_single_build/`) |

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
| Defined from the actual restored boundary c and the shared physical block [c − 4,352, c); block identity AND bytes; a missing shared-block measurement = incomplete coverage | `analysis/sentinel.py`: R-PS-coverage, -identity (scheduler ids, writer's id, dump block tables), -bytes (MLA and index-K pages), -state; in-session rc 14 for PS-c; plan rule Q15 | Unit U4 (all 8 fixtures); tamper T14; synthetic a-ps (byte flip) → R-PS-bytes; r-ps-cover → R-PS-coverage; r-ps-ident → R-PS-identity |

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
| F6 calibration and planted faults | C1 passes the new rules; 6 / 6 faults detected | `calibration/f6_calibration.out` |
| F8 regression | identical (27 rows); C1 post-inspection recorded | `calibration/f8_regression.out` |
| Unit and real-data tests | 25 / 25 | `tests/out/units.log` |
| Plan check on the session plans | B: 224 requests, 0 failures; A0: 176, 0 | `tests/out/plan-check.log` |
| Tampered plans | 19 / 19 (untampered passes; 18 refused with their rules) | `tests/out/plan-tamper-tests.log` |
| Runner refusals with shims | 23 / 23 (DRY_RUN B and A0, budget lines, approval and policy refusals, attempt rules incl. invalid-start, min wall, variant, image pin, disk, both frozen-hash gates, ids tamper, existing run dir) | `tests/out/runner-tests.log` |
| Synthetic dry runs (attempt 1's 20 variants under the retry pipeline + 7 retry plants) | 27 / 27 behave as expected (GATE-1R verdict, exactly the planted rows, reading class) | `tests/out/synth-check.log` |
| Final DRY_RUN | B and A0 pass (line and SHA256SUMS sha256 in DRYRUN.out) | `DRYRUN.out` |

## Freeze

`SHA256SUMS` (`make_sums.sh`) covers every design, code, replay, calibration, cost and evidence file. It excludes:
- the item ledger `costs.jsonl`;
- session outputs `run-fcr*/` and `DRYRUN.out`;
- bytecode;
- the synthetic runs' raw data (only each variant's `gate1r.json`, `fcr_result.json` and logs are hashed).

Its own sha256 and the final DRY_RUN line are in `DRYRUN.out`. Attempt-1 files are pinned by `../SHA256SUMS`, and the
runner verifies both.

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
7. **The decode acceptance model** comes from C1 of run-fc1 only; step timing is modelled to within about 5 %.
