# First long-context contrast: retry package (proposal; CPU only; NOT launched)

2026-10-15. Prepared under PLAN.md amendment 22d, the binding specification. Changes are labelled **R-n**
(`CHANGES-R.json`). `RETRY-FREEZE.md` maps every 22d requirement to its implementation and test evidence. Running this
needs an amendment that records the frozen package and a new operator authorization with a GPU-hour total, a variant
and an attempt policy.

How it was prepared:
- **No GPU.** No GPU or glm53* container was started; nothing used `--gpus`; no `sudo`; nothing was launched except
  DRY_RUN.
- **CPU containers only**, all `--network none`, `CUDA_VISIBLE_DEVICES=` and the host uid:
  - the serving image `local/glm53-v0.9.1-b1-forcehook-mg2:local` (sha256:ac05230b…), for the replay;
  - the B1 image (sha256:799d3cf8…), for the analyses and the synthetic runs.
  - Container names were `fcretry-replay-*`, `fcr-*`, never glm53*.
- **Attempt-1 files are untouched.** The retry imports the frozen attempt-1 geometry and analysis modules read-only;
  the attempt-1 SHA256SUMS still verifies.
- **B1, the hook and the served configuration are unchanged.** The environment is attempt 1's, with WMAX 8448.
- No D2 termination value was read. C1 of run-fc1 is used only as a labelled instrument run (amendment 22d s.10).

---

## 0. What changed

| Label | 22d item | Change | Section |
|---|---|---|---|
| R-1 | F7 | CPU replay that drives the image's own scheduler and KV-cache manager; reproduces run-fc1 step for step; exercises the proposed session | 2 |
| R-2 | F1, F2 | Scoped cache rule with engine tests; per-request safety gates, construction accounting, balance gate | 3, 5 |
| R-3 | F3 | Feasible schedule under strategy (a): split builds with touch passes, LP runs on the primary builds, two shared C/D builds | 4 |
| R-4 | F4 | Fixed staggered submission; each pass ordered oldest-freed first | 6 |
| R-5 | F5 | Sentinel on the shared block [c − 4,352, c): identity and bytes | 7 |
| R-6 | F6 | Null-relative page and early-layer criteria; originals kept beside; planted faults stay detectable | 8 |
| R-7 | s.9 | GATE-1R and in-session early stops | 9 |
| R-8 | F8, runner | Regression fixture, runner, client, plan check, analysis; tests and DRY_RUN | 10 |
| R-9 | costing | Measured-rate costing, early-stop costs, ceiling and attempt policy | 11 |

---

## 1. The question is unchanged

The primary contrast is clean B1 ×3 against B1 with decode-window D2 injection ×3. It uses 8 fixtures, 8 requests in
flight, and separate batches for each condition. Variant **B** adds the long-prompt / short-window control; variant
**A0** is the primary contrast only.

These parts are unchanged: the primary statistic and its limits, the screens, the readings, the direction statements,
every sensitivity, the exposure and selection rule, and the LP comparison. `analysis/analyze_fcr.py` calls attempt 1's
frozen functions for all of them. No probability difference is claimed to explain GPQA completion, and no GPQA pass
follows from instrument validity or from a local logit effect.

---

## 2. F7: replay of the image's own cache machinery [R-1]

`replay/engine_replay.py` constructs the image's own classes inside a CPU container of the serving image and drives
them. Nothing is reimplemented:
- `AsyncScheduler` (`Scheduler.schedule` / `update_from_output`);
- `KVCacheManager`, `HybridKVCacheCoordinator`;
- the single-type managers `FullAttentionManager`, `KpoolTailManager`, `MambaManager` and `SlidingWindowManager`;
- `BlockPool`, `Request`, and the image's block hasher.

The loop mirrors `EngineCore.step_with_batch_queue` with a batch queue of 2.

Only three inputs are supplied from outside:
1. **The KV-cache config.** It is built the way the image builds it (`generate_scheduler_kv_cache_config`):
   - 1,275 shared block ids;
   - MLA and index-K, block 4,352;
   - the kpool tail, block 8, outside prefix caching;
   - 7 identical KDA Mamba groups in align mode, with 3 speculative blocks;
   - the DFlash2 draft window, 2,048 tokens in 1,088-token blocks, outside prefix caching.

   Recomputed this way, the image's `glm53-placement` summary equals the logged line.
2. **The served values.** max_num_batched_tokens 2,048; max_num_seqs 16; async scheduling; Model Runner V2; DFlash
   K = 3; prefix caching on; `GLM53_DFLASH_BOUNDARY_LOOKUP=0`; dense retention.
3. **The "model".** For each scheduled request, the tokens the step commits and their ids. Forced requests use the
   frozen forced ids.

### 2.1 Reproduction

**run-fc1 is reproduced exactly** (`replay/out/reproduce_fc1.json`). All 5,609 of 5,609 scheduler steps are identical
to the guard's record: the scheduled set and order, num_computed, scheduled tokens, drafts, finished ids, the group-0
and group-1 block ids, every group's block-list length, free blocks, the guard's fingerprint and its live-block census.
All 32 client cached counts are reproduced.

- **The one-block drop.** C1 restores c = (⌊(P−1)/4,352⌋ − 1)·4,352 on 7 fixtures, attributed to warm-s1's
  prefill-written blocks.
- **M3's miss is refined.** Warm-M3's boundary state at 56,576 survived the whole warm phase. **C1's own admissions
  evicted it**: the KDA state went at step 1,963 and the first needed MLA block at step 1,964, while the HTTP race had
  admitted C1-M3 fifth (lookup at step 1,973). The diagnosis attributed the loss to the warm phase alone.
  - Counterfactual on the real history (`fc1_counterfactual.json`): replayed exactly to step 1,960, then C1 submitted
    (a) in the recorded admission order gives M3 a miss, reproducing the record; (b) in warm-completion order gives
    **all 8 hits**; (c) in reverse order gives 2 misses.
- **The sentinel's private recompute.** Attempt 1's range [c0 − 1,024, c0) lies in block index c / 4,352, which every
  restoring request recomputes privately. The last shared block is [c − 4,352, c), owned by the build (prefill).

**The step comparison detects a wrong configuration** (`neg_controls.jsonl`):

| Changed input | First mismatching step |
|---|---|
| 1,274 blocks | 1 |
| Salts dropped | 1,165 |
| V0 max_tokens 95 | 654 |
| `GLM53_DFLASH_BOUNDARY_LOOKUP=1` | 1,961 (C1 would restore c0) |

Whether the 7 KDA groups carry the EAGLE flag cannot be identified from the records: both assignments give identical
steps. The replay uses the image's own fallback.

**Earlier sessions** (`historical.json`):

| Session | Steps reproduced | Cached counts |
|---|---|---|
| b0-mapguard run-a3 | 2,482 of 2,482, exact | 72 / 72 |
| d7 step2 run-b1-a1 | exact to step 2,440; mismatches only inside phase 2d's aborted streams | 72 / 72 |
| phase2-stage2 (model mode, no scheduler log) | — | 49 / 49 |
| phase3 (model mode) | — | 28 / 28 |

In total, **253 measured cached counts** are reproduced across 5 sessions. They include exact repeats, warm-prompt
extensions (r2 on r3's prefix, r0 on r1's), lengths on both sides of boundaries, staggered submission and EOS-stopped
requests.

### 2.2 The proposed session under interleavings and pressure

`session_sim.py` runs the plan of section 4 through the image in model mode:
- The client follows the retry's phase order and submission policy.
- Acceptance per decode step is drawn from run-fc1 C1's measured distribution, per fixture.
- Step times come from a linear model fitted to run-fc1's 5,608 steps.
- The start state is run-fc1's exact post-startup state, replayed. A model-mode start gives identical results.

There are 25 design scenarios (`replay/out/sim_*.json`, summarised in `F3_schedule.json` / `.out`):
- acceptance seeds 1-5;
- run and build staggers ×0.1 / ×2 / ×4, and random arrival jitter of 1 s and 3 s;
- admission order: completion order, plan order, reverse (adversarial), random ×2;
- KV pool sizes 1,275 / 1,250 / 1,225 / 1,200 / 1,150 / 1,100 blocks;
- sequential builds, longest-first builds, a 600 s gate wait;
- variant A0.

**Result: every lookup is a full hit in all 25.** The conservative occupancy bound is the minimum survival margin:
the number of further blocks the schedule could allocate before the first block a pending lookup needs would be
evicted. It is ≥ 359 blocks (at 1,100 blocks) and ≥ 478 at the served 1,275. The maximum all-8 lead is 95 output
positions (primary) and 159 (LP), against 512.

### 2.3 Ablations that shaped the design

Archived in `replay/out/design_single_build/`:
- **One 8-fixture build per salt (0.74 M tokens), as attempt 1:**
  - every hit survives at 1,275 blocks, but the first-finished fixture keeps only 30 blocks of margin;
  - **a 25-block smaller pool loses M3** in every first batch, and reverse or random admission loses it as well.
- **The LW reference builds (0.81 / 0.85 M) lose M3 even with completion-order admission.** They are therefore used
  only as cold references, and no run restores them.
- **Split builds without the final touch pass:** all hits, but margins of 49-100 blocks.

---

## 3. F1: the cache rule, scoped [R-2]

`analysis/fcr_geometry.py rule_c`, one definition for client, plan check, gate and replay:

> c = max(0, (min(⌊P_build / 4,352⌋, ⌊(P_req − 1) / 4,352⌋) − 1) · 4,352)

This holds for a request whose prompt's full blocks were all written by prefill of an earlier request of the same
fixture under the same salt, with no eviction in between. For an exact repeat it is the run-fc1 rule. P_build is the
longest earlier prompt of the stream under the salt, so a second LP user restores c_L = rule(P_L), written by the first
LP run's prefill.

**Everywhere else the image's own lookup decides**, through the replay before the session and through the recorded
value during it.

`replay/cache_rule_tests.py` passes **70 / 70** against the image's lookup:
- T1, exact repeats at k·4,352 − 1, k·4,352, k·4,352 + 1 and + 2,727 for k = 1, 2, 3, 15, 25. The rule also holds at
  exact multiples.
- T2, extensions of a warm prompt: +1, to the boundary − 1, to the boundary, +1 / +2 / +10 blocks, +7,168.
- T3, prompts shorter than the build, including the LW2-built → P_L form.
- T4, cold use, other salt, no salt.
- T5, eviction:
  - a 0.88 M-token flood evicts everything (0);
  - a growing flood moves the hit straight from the rule value to 0, because FIFO eviction of one build is
    all-or-nothing;
  - targeted evictions through the image's `evict_blocks` give the partial hits the image actually returns
    (78,336 / 17,408 / 78,336), always block-aligned and below the rule.
- T6, a prompt reaching a forced decode window restores decode-written blocks. Provenance flags them, and the plan
  invariant refuses such a prompt.
- T7, the actual retry forms on real ids: touch, C, D, CV, PS, first and second LP user, and LP on an LW2 build. C and D
  restore the identical physical blocks.

**Validation against measured records:** the 253 counts above. The repeat rule holds on all 99 measured exact-repeat
hits.

---

## 4. F3: the feasible schedule under strategy (a), configuration unchanged [R-3]

### 4.1 Order

The client re-orders every run, touch and sentinel pass by the recorded completion time of each fixture's most recent
request under that salt (R-4).

| Stage | Phases | Notes |
|---|---|---|
| A | V0 · LW1 (l1) · LW2 (l2) | Cold prefill references; no run restores them |
| A | **s1:** warm-s1-a (4 longest) → T-s1-a → warm-s1-b (4 shortest) → T-s1 (all 8) → LC1 → LC2 → C1 → CV | LP runs first: their prompts may not reach any primary decode-written block |
| A | **s2:** warm-s2-a → T-s2-a → warm-s2-b → T-s2 → C2 → PS-a → PS-b | |
| — | **GATE-1R** | The engine stays idle, so s2's build survives to D2 |
| B | D2 (s2) → PS-c | |
| B | **s4:** split build → LD1 → LD2 → D1 | |
| B | **s3:** split build → LD3 → LC3 → D3 → C3 | C3 is the residue check |

- Primary positions: C at 1, 2, 6 and D at 3, 4, 5, as in attempt 1. LP positions are the same.
- **No 4-request batches.** Every run is the 8 fixtures together.
- **Why the builds are split.** No volume of more than one half (≤ 0.43 M tokens) is built after a fixture's last
  refresh, against a measured reach of about 0.75 M.
- **Why the touch passes.** Each is 8 sequential unkeyed 1-token requests. Each restores a prefix and frees it to the
  tail of the FIFO free queue; non-cached blocks are reused first, so a pass consumes almost nothing at the head.

### 4.2 Construction

| Pair | Restores from | Wording in reports |
|---|---|---|
| C1 / D1 | builds s1 / s4 | "separate prefix builds of the same tokens; NOT an identical prefix" |
| C2 / D2 | build s2 (C2 before the gate, D2 after) | "identical physical prefix blocks (verified)", only if block identity holds |
| D3 / C3 | build s3 (D3 then C3) | as C2 / D2 |

- **Each arm has one run whose build no run of the other arm restores (C1 on s1, D1 on s4), one first user and one
  second user of a C/D-shared build.**
- The within-arm nulls all span different builds, so prefix-reconstruction variability is inside the clean null.
- LP restore depth is c for the first LP user of a build and c_L for the second. LC is 1·c + 2·c_L and LD is
  2·c + 1·c_L; both are prefill construction, and this is reported.
- The adjacent C/D pairs were tested for churn: (C2, D2) and (D3, C3) restore the identical physical blocks. The
  across-the-gate share saves a build only because the engine is idle during GATE-1R.

### 4.3 Occupancy, survival, hits and lead

From the base scenario (`F3_schedule.out` gives every phase):
- **Batches:** live blocks peak at about 450 (MLA and index-K 188, each KDA group 33, draft 24-28). Free blocks never
  fall below 819.
- **Builds:** about 175 live; free ≥ 1,099.
- **Boundary state at each lookup:** the KDA state at c (7 groups) and the MLA chain through index c / 4,352 are both
  cached, written by the salt's build (prefill). LP second users additionally restore the first LP user's prefill.
- **Expected hits:** all 224 requests restore their F1 value in all 25 scenarios. **Permitted misses:** none are
  expected (section 5).
- **All 8 decode together** before output index 95 (primary) or 159 (LP), against 512.
- **The sentinels' shared state survives.** PS-a, PS-b, PS-c, D2 and C2 restore it in every scenario, with a minimum
  margin of 438 blocks.

**No partial-cache design is needed.** Eviction is eliminated, not balanced. The fallback rules for an unexpected miss
are in section 5.

---

## 5. F2: safety gates and construction accounting [R-2]

`analysis/cache_gate.py` uses the client records and the guard's EngineCore scheduler log. The runner already copies
the log, and the client records each request's response id.

**For every request:** cached tokens, expected value, class (cold / hit / partial / miss / OVER), engine id and
restore source.

**Exact safety, structural, gated:**

| Check | Requirement |
|---|---|
| S1 | Block-aligned |
| S2 | Never beyond the stream's prefill coverage under its salt |
| S3 | Every restored group-0 block is one an earlier request of the same fixture and salt held at the same index **inside its prompt** (no decode-written history) |
| S4 | The scheduler's num_computed equals cached, and the restored ids are attributed |

**Construction:**
- **Every class is recorded.** Hits on stage A are required (R-cache-model: the replay's prediction must be confirmed
  before injected traffic).
- **After the gate, a miss is permitted and safe.** It is recorded, and its fixture is flagged.
- **R-cache-balance (gated post-session).** C and D have equal non-hit counts, as do LC and LD.
- **Construction-matched sensitivity.** The primary is repeated on the fixtures that hit in all six runs.
- **Block identity** across the users of each build is reported. A pair is called identical only when identity holds.

Real-data check: on run-fc1, all 32 requests map. C1's 7 hits are attributed to their own warm-s1 builds, M3 is a safe
miss, and every request is safe.

---

## 6. F4: fixed staggered submission [R-4]

- **Builds:** the plan's fixed order (half a: L1 A8 L8 L5; half b: M3 L4 M4 L3), staggered by 1.0 s.
- **Runs:** staggered by 0.5 s, in ascending completion time of each fixture's most recent request under that salt.
- **Touch and sentinel passes:** strictly sequential, in the same order.

The client records submission order and times. The gates use the scheduler log for actual admission, test the needed
properties (hits, safety, all-8 lead, identity), and never test agreement with a predicted timing. Determinism is not
claimed: the replay sweep includes jitter, other orders and pool sizes.

---

## 7. F5: the sentinel [R-5]

The sentinel is on s2, the build that C2 and D2 share. PS-a and PS-b come before the gate (PS-b is the determinism
null); PS-c comes after D2.

- **Each request:** prompt P, one forced token, a dump of exactly **[c − 4,352, c)** with c the restored boundary,
  and a snapshot after the prefill step.
- **Coverage.** Restored exactly c, dump range exact on both ranks, snapshot present. A missing measurement means
  incomplete coverage, never a successful sentinel. In-session, PS-c missing coverage stops the session (rc 14).
- **Identity.** The group-0 physical block at index c / 4,352 − 1 is the same in every sentinel request, in C2 and D2,
  and in each dump's block table. It is also the id the s2 build wrote.
- **Bytes.** Every MLA and index-K page of that block is identical across PS-a, PS-b and PS-c on both ranks.
- **Boundary state.** The KDA state rebuilt from the cached state at c is bitwise equal across the sentinels, or within
  2 × the PS-a / PS-b difference.
  - The guard does not log KDA block ids. Restoration of the KDA state at c is certified by the restored length: the
    coordinator returns c only when all 7 groups hold that state. This limit is stated.

---

## 8. F6: numerical criteria made relative to matched long-context nulls [R-6]

`analysis/f6_criteria.py`. Units and the 0.25 pool threshold are attempt 1's. Nulls are matched on the same units:
N_pp = LW1 vs LW2 and N_dd = C1 vs C2.

| Rule | Criterion |
|---|---|
| R-idx-pooled (G1-content, unchanged) | gross ≤ max(2 × max(N_dd, N_pp), 0.1 %) |
| R-idx-pages | Every judged page (fixture × rank × layer × 64 pools): fraction ≤ max(2 × N_pp page, 0.25). The attempt-1 0.25 survives as the floor; a page's own noisy prefill null widens only that page |
| R-idx-early | DSA layers 3-15 as one stratum: gross ≤ max(2 × N_pp, 0.1 %) |

The page and early rules use the **prefill null only**. N_dd = C1 vs C2 contains the tested run, so a fault confined to
one decode run inflates its own N_dd and would widen its own limit; this is 21h's recorded limit of the pooled rule. The
synthetic plant r-f6-page exposed it before freezing (a swapped page in C1 passed against max(N_dd, N_pp)). N_dd stays
in the pooled rule and is reported beside the page and early rules.

- **Exact rows stay exact:** dump attribution, token position, population, mapping, ledger, prohibited writes, stable
  committed bytes.
- **Reported beside:** tables by fixture, fixture × layer and page; R3 flags with the leave-flagged-out sensitivity;
  the original criteria and their failures.
- **Wording of a pass:** "compatible with measured long-context numerical variation".

**Calibration on run-fc1's real C1 / LW1 / LW2** (post-inspection, calibration only, N_dd absent;
`calibration/f6_calibration.json`):
- C1 passes the new rules: pooled 0.765 % against a limit of 1.538 %; 0 of 2,486 judged pages failing; early layers
  3 / 57,344.
- The original criteria still fail (8 pages above 25 %, 3 early pools) and are kept beside.
- **All 6 planted faults are detected.** The pooled rule alone catches only the foreign request; the page rule catches
  all six:

| Planted fault | Failing pages |
|---|---|
| Wrong page, all layers | 12 |
| Wrong page, one layer | 2 |
| Wrong page in M4's fragile region | 10 |
| Foreign request (A8 ← L8, tag and P kept) | 308 |
| Stale state | 10 |
| Unwritten page | 6 |

---

## 9. Integrated gate and early stops [R-7]

Reference builds and clean checks come before any injected traffic. Stops, in session order:

| Point | Stop (client rc) | Class (22d s.9) |
|---|---|---|
| Before traffic | Geometry check: the served KV geometry and cache configuration equal the replayed ones, num_blocks ≥ 1,150 (rc 16); readiness or instrumentation (rc 5-8) | — |
| After every phase | Errors or forced ids (rc 9); S1-S4 (rc 12) | Unsafe restoration |
| After every run batch | All-8 lead ≥ 512 (rc 13) | Insufficient concurrent decoding |
| GATE-1R, after stage A (rc 10) | R-cache-safe, R-cache-model | Unsafe restoration / replay not confirmed |
| | G1-inflight | Insufficient concurrent decoding |
| | G1-align, G1-cross, G1-regime | Alignment failure |
| | G1-content, R-idx-pages, R-idx-early, CV numerics | Uninterpretable numerical content |
| | R-PS-coverage, R-PS-identity, G1-CV-cross | Missing boundary or shared-prefix coverage |
| | Unchanged structural rows | Structural |
| PS-c (rc 14) | Missing sentinel coverage | Shared-prefix coverage |

Post-session (`analyze_fcr.py`): GATE-1R recorded and recomputed, per-run validity (F2 safety in place of exact
equality), injection and guard, ledger against bytes, R-cache-safe and R-cache-balance, the sentinel on
PS-a / PS-b / PS-c, and C3 residue with F6. Readings follow amendment 22 s.8 exactly.

---

## 10. Runner, client, analysis and tests [R-8]

| File | Role |
|---|---|
| `runner/run_fcr.sh` | Attempt 1's runner with: item fcr1 and its own ledger; VARIANT B / A0; ceilings from the cost model; attempt policy; pre-traffic geometry check; both SHA256SUMS verified |
| `runner/client_fcr.py` | Executes the geometry plan; R-4 submission; quick checks and stops; GATE-1R wait |
| `runner/gate_step_fcr.sh` | Copies stage-A records and the scheduler log; runs `gate_fcr.py` in a CPU container |
| `runner/plan_check_fcr.py` | Rules Q1-Q16 |
| `runner/geometry_check_fcr.py` | Pre-traffic geometry check |
| `analysis/gate_fcr.py`, `analyze_fcr.py`, `cache_gate.py`, `sentinel.py`, `f6_criteria.py`, `fcr_geometry.py` | Gate, analysis and helpers |

Tests (results in RETRY-FREEZE.md):
- replay reproduction, negative controls, historical sessions and counterfactual;
- F1 rule tests;
- unit and real-data tests;
- F8 regression;
- tampered plans and runner refusals;
- synthetic dry runs of 27 variants: attempt 1's 20 under the new pipeline, plus 7 retry plants;
- final DRY_RUN.

**F8.** The frozen attempt-1 `gate_fc.py`, re-run unchanged on run-fc1, reproduces `gate-asrun/gate1.json` exactly:
27 rows, verdict FAIL, so the original verdict stands. C1, shown post-inspection against the new checks: F2 safe (7
hits, 1 miss), F6 pass, and 9 attempt-1 rows passed. **C1 is not pooled or paired** with any retry run.

---

## 11. Costing, ceiling and attempt policy [R-9]

**Measured inputs** (run-fc1): startup 732 s; V0 6 s; phase gap 6.3 s; prefill 2,770 tok/s; hooked decode
36.6 tok/s per request at 8 in flight; end overhead 19 s.

**Step counts** come from the replay. Its step-time model agrees with the measured phases to 5 %:

| Phase | Modelled (s) | Measured + gap (s) |
|---|---|---|
| LW1 | 289 | 304 |
| 8-fixture build | 276 | 280 |
| C1 | 319 (all hits) | 306 (one cold fixture) |

**GATE-1R** is 270 s central (copy plus analysis) and 600 s pessimistic, with the engine idle.

**Pessimistic case:** prefill 2,410 tok/s; decode 25 tok/s; **one cold fixture in every run batch** (+40 s each), i.e.
priced misses and cold work; startup 800 s.

`cost/cost_fcr.json`, occupied GPU-hours (2 GPUs):

| | Central | Pessimistic | Ceiling max(1.3 × central, pess.) | Wall cap | Min wall |
|---|---:|---:|---:|---:|---:|
| **B** (option B) | 3.00 | 4.18 | **4.18** | 7,523 s | 4,942 s |
| **A0** (primary only) | 2.75 | 3.72 | **3.72** | 6,696 s | 4,945 s |

Option B's decode-rate sensitivity: 20 tok/s → 3.91, 25 → 3.51, 30 → 3.24, 36.6 → 3.00, 41.6 → 2.87.

**Early-stop costs**, central / pessimistic GPU-hours, option B (A0 is similar until its first batch):

| Stop point | GPU-h |
|---|---|
| Pre-traffic (rc 5-8, 16) | 0.42 / 0.46 |
| After the references | 0.75 / 0.83 |
| First touch pass (rc 12) | 0.93 / 1.03 |
| First batch lead (rc 13) | 0.97 / 1.12 |
| After C1 (rc 9) | 1.19 / 1.47 |
| GATE-1R not PASS (rc 10) | 1.75 / 2.34 |
| PS-c coverage (rc 14) | 1.94 / 2.64 |
| Complete | 3.00 / 4.18 |

**Attempt policy (operator's choice; the runner enforces whichever is authorized):**
1. **single:** one attempt. **Recommended**, as amendment 22a.
2. **invalid-start:** one further attempt only if every earlier attempt stopped before any client traffic (rc 5-8 or
   16). The approval may then exceed the variant's ceiling by at most the 0.46 GPU-h pre-traffic allowance, i.e. 4.64
   (B) or 4.18 (A0). Never automatic.
3. Anything else needs a new amendment after a diagnosis.

**Proposed authorization:** variant B, ceiling 4.18 GPU-h, policy single.

---

## 12. Limits, infeasible items and open risks

1. **KDA block identity is not in the guard's scheduler record.** The sentinel certifies the KDA state at c by restored
   length plus snapshot bytes; exact KDA block ids would need a guard change, which was not made.
2. **Not step-exact (2 cases):**
   - d7's phase-2d aborted streams; their cached counts are still exact;
   - whether the KDA groups carry the EAGLE flag, which is behaviour-equivalent on every recorded workload.
3. **The replay's timing is modelled.** The cache outcome depends on order, which is recorded and gated, not timed.
   Acceptance is drawn from C1 only.
4. **GATE-1R idles the GPUs for about 270 s** (priced) so that s2 survives to D2. Giving D2 its own build instead would
   cost a build (about 280 s) and break the construction balance.
5. **LP restore-depth imbalance** (section 4.2) is reported.
6. **The pool floor is 1,150 blocks**, refused before traffic. Five sessions have all logged 1,275.
7. **The long-context decode-window instrument is validated in-session only.** Every GATE-1R numerical row is a first
   long-context measurement with C2 (N_dd); C1 had N_dd absent.
8. **Unchanged from attempt 1:** coarse resolution with 3 runs per arm, operational screens only; what a null result
   cannot exclude; doc 79's three fixtures; disk (≥ 200 GB free required).

## 13. Preconditions before any launch

1. The package SHA256SUMS is recorded in an amendment and the tests pass (RETRY-FREEZE.md).
2. Independent review of the revised package (22d).
3. Operator authorization: item fcr1, a GPU-hour total, the variant and the attempt policy.
4. GPUs idle, and desktop graphics clients on GPU 1 closed if possible. The pre-flight reports them.
