# First long-context contrast: retry package (proposal; CPU only; NOT launched)

2026-10-15. Prepared under PLAN.md amendment 22d, the binding specification, and corrected under amendment 22f
(changes R-10 to R-15, before any retry data), and changed under amendment 22h, the operator's decision "both" (R-16
to R-18: ceiling 6.2 GPU-hours and a batch guard; runner, client and costing only). Changes are labelled **R-n** (`CHANGES-R.json`, with before / after
hashes for R-10 onward; the pre-22f files are kept as `*.pre22f`). `RETRY-FREEZE.md` maps every 22d requirement to its implementation and test evidence. Running this
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
| R-10 | 22f s.2 | Sentinel boundary state: SSM and convolution evaluated separately, each against its own PS-a / PS-b baseline; only the committed state and the relevant conv entries compared. Analysis only; no hook change | 7 |
| R-11 | 22f s.3 | Acceptance stress scenarios in the replay (single-token, maximum and strongly unequal advance), their cost, and whether the 4.18 ceiling holds | 2.4, 11.1 |
| R-12 | 22f s.4 | F6 pages whose null makes the page limit non-discriminating are reported as "not independently verified by the numerical check" | 8 |
| R-13 | 22f s.5 | The sentinel's limits stated in every output and here | 7 |
| R-14 | 22f s.6 | LP: descriptive same-depth comparison (LC1 / LD1, LC2 / LD2); LC3 / LD3 reported as depth-mismatched; "unresolved construction sensitivity" rule | 4.2 |
| R-15 | 22f s.1 | Wording: eviction avoided in the replayed scenarios, with measured survival margins; the actual-session gates are authoritative | 2.2, 4.3 |
| R-16 | 22h s.1 | Variant-B ceiling 6.2 occupied GPU-hours (APPROVED_GPU_H ≤ 6.2, wall cap 11,160 s); 4.18 / 7,523 s kept as superseded; single attempt, no automatic retry | 11 |
| R-17 | 22h s.2 | Batch guard in the client: every phase starts only if its completion, projected from this session's measured rates (a conservative prior before the first of each form), fits before the deadline; stop before a batch = rc 17, "incomplete contrast (batch guard)"; replaces the 25 tok/s never-truncate rule | 9, 11.2 |
| R-18 | 22h s.3 | The stress replay re-run under 6.2 with the guard, base and all 10 scenarios, central and pessimistic | 11.2 |

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

**Result: every lookup is a full hit in all 25**, so eviction is avoided in the replayed scenarios, with measured
survival margins [R-15]. The actual-session gates (R-cache-safe, R-cache-model, R-cache-balance) remain authoritative.
The conservative occupancy bound is the minimum survival margin:
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

### 2.4 Acceptance stress scenarios [R-11; 22f s.3]

The 25 scenarios draw decode acceptance from run-fc1's C1. `replay/run_stress.sh` adds stress scenarios, which are
**not predictions of D2**. Each applies to every natural (non-CV) decode step, primary and LP batches alike:
- **single:** single-token advance throughout (poison-like: no draft accepted; 1.0 committed token per step);
- **max:** maximum accepted advance throughout (oracle-like: every scheduled draft accepted; 4.0 per step);
- **unequal-a / unequal-b:** strongly unequal between fixtures. The 4 longest fixtures run at max and the 4 shortest
  at single (a), or the reverse (b).

Each runs at the served 1,275 blocks from run-fc1's exact start state and at the 1,150-block refusal floor. The two
unequal assignments also run with reverse (adversarial) admission order, giving 10 scenarios. The base scenario
re-run with the R-11 code reproduces `sim_base.json` exactly. `replay/stress_report.py` writes
`replay/out/F3_stress.json` / `.out`:

| Scenario (1,275 / 1,150 blocks) | Non-hits | Min margin, primary | Min margin, LP | Min margin, sentinel | Max lead, primary | Max lead, LP | C / D batch (s) | LP batch (s) | Session (s, modelled) |
|---|---|---|---|---|---|---|---|---|---|
| base (measured, 2.84 per step) | 0 | 518 | 534 | 534 | 84 | 156 | 319-321 | 63-90 | 4,528 |
| single | 0 / 0 | 518 / 393 | 534 / 409 | 534 / 409 | 24 | 49 | 674 | 108-135 | 6,925 |
| max | 0 / 0 | 518 / 393 | 534 / 409 | 534 / 409 | 93 | 193 | 191 | 47-75 | 3,661 |
| unequal-a (reverse order) | 0 / 0 (0) | 526 / 401 (478) | 534 / 409 (478) | 534 / 409 (478) | 93 | 193 | 692-694 | 109-138 | 7,042-7,053 |
| unequal-b (reverse order) | 0 / 0 (0) | 518 / 393 (478) | 534 / 409 (478) | 534 / 409 (478) | 89 (37) | 81 | 692-693 | 109-136 | 7,037-7,042 |

- **Cache survival:** every lookup is a full hit in all 10 scenarios. The minimum survival margin is 478 blocks at
  1,275 and 393 at the 1,150 floor. The sentinels' shared state survives in every scenario.
- **Lead-in:** all 8 requests decode before output index 93 (primary) and 193 (LP), against 512.
- **Runtime:** single-token or strongly unequal advance roughly doubles every C / D batch (674-694 s against about
  320 s) and lengthens the session by about 2,400-2,500 s. Maximum advance shortens it by about 870 s. The cost
  consequences, including whether 4.18 holds, are in section 11.1.

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
- **LP same-depth comparison [R-14; 22f s.6].** The primary contrast is unaffected, but LP attribution is limited.
  - `analyze_fcr.lp_same_depth` adds a descriptive comparison of the same-depth pairs: LC1 against LD1 (both restore
    c) and LC2 against LD2 (both restore c_L).
  - LC3 (c_L) against LD3 (c) is reported separately as depth-mismatched.
  - The three-repeat rule is not applied to the two matched pairs; no beyond-null or repeatability verdict is drawn
    from them.
  - **Dependence rule, fixed before data.** Take the LP conclusion: the short-context injection contrast LD − LC and
    the interaction (D − C) − (LD − LC), each judged by its frozen threshold. Recompute each with the matched pairs'
    mean difference in place of the three-run difference. If that changes the beyond-null status, or the sign while
    beyond the null, the conclusion depends on the mismatched pair. It is then reported as **"unresolved construction
    sensitivity"**.
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

**No partial-cache design is needed.** Eviction is avoided in the replayed scenarios, with measured survival margins
(section 2.2, and section 2.4 under acceptance stress) [R-15]. The actual-session gates remain authoritative. The
fallback rules for an unexpected miss are in section 5.

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
- **Boundary state [R-10; 22f s.2].** The KDA state rebuilt from the cached state at c is checked per fixture × rank ×
  KDA layer. The SSM and convolution components are evaluated **separately**, each against its own PS-a / PS-b
  baseline:
  - **R-PS-state-ssm** compares only the intended committed state: the snapshot's single non-speculative slot.
  - **R-PS-state-conv** compares only the relevant convolution entries: the committed rows at positions P − 3 .. P − 1
    (the kernel window), layout-normalised through `fc_common.conv_entries`. Rows the prefill kernel does not write are
    not compared.
  - **Per component:** where PS-a and PS-b are bitwise equal, PS-c must be bitwise equal to PS-a. Otherwise PS-c must
    be within 2 × that component's own PS-a / PS-b difference. One component's baseline never licenses the other
    component's change. A snapshot at another frontier, or a missing component, fails.
  - This is an analysis change only: no hook change and no GPU revalidation.
  - **Before the correction,** the rule combined both components into one equality and used only the SSM difference
    as the tolerance. A non-identical SSM baseline therefore let any convolution change through, and the conv rows were
    taken by raw index.
  - **Tests** (`tests/test_sentinel_fcr.py`, on real attempt-2 KDA prefill snapshots as templates): 14 / 14. Five clean
    cases pass. Nine negatives fail as expected: convolution-only corruption, SSM-only corruption, a non-identical
    baseline followed by excessive convolution change, and others. The pre-22f module passes 4 of the 9 negatives.
    End to end, the synthetic plants r-ps-conv, r-ps-ssm and r-ps-nonid fail their component and r-ps-nonid-ok passes.
- **Limits [R-13; 22f s.5],** stated in every output (row R-PS-limits, `gate1r.txt`, `fcr_result.txt`):
  - MLA / index-K shared-block identity is directly checked;
  - KDA restoration and captured-state consistency are checked;
  - physical KDA block identity and the complete unsampled state are not established.

  The guard does not log KDA block ids. Restoration of the KDA state at c is certified by the restored length: the
  coordinator returns c only when all 7 groups hold that state.

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
- **Non-discriminating pages [R-12; 22f s.4], defined before any retry data.**
  - A judged page whose limit max(2 × N_pp page, 0.25) is at least **0.5** is non-discriminating. At that point its own
    prefill null is at least 0.25, the attempt-1 page value. A fault confined to half of the page's pools or fewer
    cannot then be distinguished from that page's null.
  - At a limit of 1.0 or more, no fault on the page can fail it. Planted whole-page faults give a fraction of 1.0.
  - Such a page is still judged; the rule's verdict does not change. It is reported in every output (row
    R-idx-pages-nd, `gate1r.json` / `.txt`, and C3's residue in `fcr_result.json` / `.txt`) as **"not independently
    verified by the numerical check"**. Its pass rests on the exact structural rows alone.
  - In the calibration, 7 of 2,486 judged C1 pages are non-discriminating, all M4 (limits 0.56-0.91; none ≥ 1.0). The
    planted M4 whole-page fault is still detected on 3 of those pages.
  - Synthetic plant r-f6-nd swaps two LW2 pages: GATE-1R passes and the pages are reported for C1, C2 and C3.

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
| Before every build, touch pass, sentinel pass and run batch (rc 17) [R-17] | Batch guard: its projected completion does not fit before the deadline (an LP batch is skipped instead) | Incomplete contrast (batch guard) |
| Wall cap − 75 s | Hard watchdog, the last resort (may cut mid-phase) | — |

rc 11 ("a required phase does not fit", the 25 tok/s never-truncate rule) is superseded by rc 17 and no longer emitted.

GATE-1R also reports, without gating:
- R-idx-pages-nd, the non-discriminating pages [R-12];
- R-PS-limits, the sentinel limits [R-13].

R-PS-state-ssm and R-PS-state-conv [R-10] gate as structural rows on PS-a / PS-b at GATE-1R (presence and
identifiability) and on PS-c after the session.

Post-session (`analyze_fcr.py`):
- GATE-1R recorded and recomputed;
- per-run validity (F2 safety in place of exact equality);
- injection and guard; ledger against bytes;
- R-cache-safe and R-cache-balance;
- the sentinel on PS-a / PS-b / PS-c (SSM and conv separately);
- C3 residue with F6, including its non-discriminating pages;
- the LP same-depth comparison [R-14].

Readings follow amendment 22 s.8 exactly.

---

## 10. Runner, client, analysis and tests [R-8]

| File | Role |
|---|---|
| `runner/run_fcr.sh` | Attempt 1's runner with: item fcr1 and its own ledger; VARIANT B / A0; ceilings from the cost model; attempt policy; pre-traffic geometry check; both SHA256SUMS verified |
| `runner/client_fcr.py` | Executes the geometry plan; R-4 submission; quick checks and stops; GATE-1R wait; batch guard [R-17] |
| `runner/batch_guard_fcr.py` | Batch guard: measured-rate projection and the start / skip / stop decision [R-17] |
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
- synthetic dry runs of 33 variants: attempt 1's 20 under the new pipeline, 7 retry plants, 5 22f plants (r-ps-conv,
  r-ps-ssm, r-ps-nonid, r-ps-nonid-ok, r-f6-nd) and the 22h plant r-guard-stop, with output-content checks (limits, cache wording, non-discriminating
  pages, LP same-depth);
- sentinel boundary-state tests [R-10] and acceptance stress scenarios [R-11];
- batch-guard unit tests (U13a-g), the guard replay under 6.2 [R-18], refusal tests for the 6.2 budget, and a
  synthetic run stopped by the guard before C3 (r-guard-stop) [R-16, R-17];
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
| **B** (option B) | 3.00 | 4.18 | **6.2** (22h; computed 4.18 superseded) | 11,160 s (superseded 7,523 s) | 4,942 s |
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

### 11.1 Acceptance stress: runtime and the ceiling [R-11; 22f s.3] (as reported under 22f; superseded by 11.2)

`cost/cost_fcr.py` applies the same costing to the stress scenarios of section 2.4. Central uses the replay-modelled
phase seconds; pessimistic uses the same transformation as above. These are stress scenarios, not predictions of D2.
It also replays the client's own time rules on each timeline under the B wall cap (7,523 s):
- the never-truncate need rule at 25 tok/s;
- LP batches skipped first;
- rc 11 when a primary phase does not fit;
- the client killed at the deadline + 15 s.

**The ceiling, wall cap and client rules are unchanged here.**

| Scenario | Complete session, GPU-h central / pess. | Within 4.18? | C / D decode, tok/s per request | Under the 4.18 cap, central | Under the cap, pessimistic |
|---|---|---|---|---|---|
| measured (base) | 3.00 / 4.18 | yes | 36.6 (measured) | complete | stops before C3 (rc 11); primary incomplete |
| max advance | 2.52 / 3.50 | yes | 44.3 | complete | complete |
| single-token advance | 4.33 / 6.08 | **no** | 12.3 | stops before C3 (rc 11), LC3 skipped; 3.90 GPU-h; primary incomplete | client killed during D1 (ran 1,008 s against a need of 431 s); LD1 / LD2 skipped; 4.07 GPU-h; primary incomplete |
| strongly unequal (a, b, reverse) | 4.39-4.40 / 6.17-6.18 | **no** | 11.9-12.0 | stops before C3 (rc 11), LD3 / LC3 skipped; 3.88 GPU-h; primary incomplete | client killed during D1; 4.07 GPU-h; primary incomplete |

**The 4.18 ceiling does not hold** for single-token or strongly unequal advance. This is reported, not changed:
- **A complete session needs 4.33-4.40 GPU-h centrally,** and 6.1-6.2 at pessimistic rates.
- **Spending stays bounded by the authorized total** (wall cap, runner-enforced). The consequence is an incomplete
  primary contrast:
  - centrally, rc 11 before C3, which means fewer than 3 valid C runs and an INCOMPLETE reading;
  - at pessimistic rates, a mid-phase cut of D1. The client's never-truncate rule assumes at least 25 tok/s per
    request, and single-token advance runs at about 8-12.
- **The break-even:** a complete session stays within 4.18 GPU-h at central rates only while C / D / LP decode stays at
  or above 17.6 tok/s per request. That is about 1.36 committed tokens per step, against the measured 2.84;
  single-token advance is 1.0.
- **Found with this section (pre-existing, unchanged):** at the pessimistic rates the base session itself (4.18 = the
  ceiling) stops before C3 under the client's 240 s reserve and need rule. Only the central case completes inside the
  wall cap.
- **No in-session rule stops early on slow decode** before these points. A C1 batch taking about 670 s instead of
  about 320 s, visible before GATE-1R, would be the first sign.

Options for the operator (none adopted here):
- authorize a higher total for the stress case;
- add a decode-rate-aware need rule or stop;
- accept the risk of an incomplete session.

### 11.2 Ceiling 6.2 GPU-hours and the batch guard [R-16 to R-18; PLAN.md amendment 22h]

**Ceiling [R-16].** Variant B's ceiling is 6.2 occupied GPU-hours: APPROVED_GPU_H ≤ 6.2, and the wall cap is
6.2 × 3,600 / 2 = 11,160 s. The 22e-22g values are kept as superseded: 4.18 GPU-hours and a 7,523 s wall cap, from the
computed max(1.3 × central, pessimistic). A0 is unchanged. Single attempt, no automatic retry. The invalid-start
policy keeps its 0.46 GPU-hour allowance but needs its own authorization.

**Batch guard [R-17]** (`runner/batch_guard_fcr.py`, called by `client_fcr.py`):
- **When it runs:** before every build, touch pass, sentinel pass and C / D / LC / LD / CV batch.
- **Projection:** 1.25 × (uncached prompt tokens / prefill rate + max_tokens / decode rate) + 30 s.
- **Rates:** the slowest rate measured in this session's completed phases.
  - Prefill rates come from the builds.
  - Decode rates are kept per run form (C, D, LC, LD and CV separately). An injected arm may decode differently from a
    clean one.
  - **Priors**, used before the first phase of each kind: 2,410 tok/s prefill, and 11.0 tok/s per request decode. The
    decode prior is below the single-token-advance rate of the 22f stress replay (11.9-12.3).
- **Deadline:** T0 + min(wall cap, remaining approved budget) − the 240 s cleanup reserve. One comparison covers both
  the remaining budget and the wall cap.
- **A primary phase** starts only if it fits; otherwise the session stops **before** it. That is rc 17, read as
  **"incomplete contrast (batch guard)"**; phases not started are reported NOT RUN, never as failures.
- **An LP batch** also keeps every later primary phase, plus 600 s for a GATE-1R still to run, in reserve. If it does
  not fit it is skipped, and so is every later LP batch.
- **Every decision and measured rate is recorded** (client kinds `guard`, `rate`).
- **The client is never stopped mid-phase.** Its timeout now coincides with the hard wall-cap watchdog (wall cap − 75
  s), which stays the last resort.
- **This replaces** the 25 tok/s never-truncate rule.
- **The design, order, gates and analysis are unchanged.** The analysis only labels a guard stop.

**Stress replay under 6.2 with the guard [R-18]** (`cost/cost_fcr.py` runs the guard's own code on the modelled
timelines). These are stress scenarios, not predictions of D2:

| Scenario | Central | Pessimistic |
|---|---|---|
| base (measured acceptance) | complete, LP complete; **3.00 GPU-h (expected spend, unchanged)** | complete, LP complete; 4.18 (no longer stops before C3) |
| max advance (1,275 / 1,150) | complete, LP complete; 2.52 | complete, LP complete; 3.50 |
| single-token (1,275 / 1,150) | complete, LP complete; 4.33 | primary complete; LP after the gate skipped (LD1, LD2, LD3, LC3); 5.64 |
| strongly unequal a / b, incl. reverse order (6) | complete, LP complete; 4.39-4.40 | primary complete; LP after the gate skipped; 5.73 |

- **No mid-phase cuts and no guard stops.** No scenario reaches the watchdog, no phase is cut, and no primary phase is
  stopped by the guard.
- **The projections are conservative:** actual / projected ≤ 0.78 in every central and pessimistic case, with one
  exception.
- **The exception:** the first batch of a form (LC1, C1, D2) under pessimistic single-token or unequal advance runs up
  to 4 % beyond its projection. Those batches start with at least 4,400 s of slack, so nothing is cut.
- **What this leaves open:** under pessimistic stress, the LP control is incomplete. LC1 and LC2 run, LD does not, so
  there is no LP reading.

**Attempt policy (operator's choice; the runner enforces whichever is authorized):**
1. **single:** one attempt. **Recommended**, as amendment 22a.
2. **invalid-start:** one further attempt only if every earlier attempt stopped before any client traffic (rc 5-8 or
   16). The approval may then exceed the variant's ceiling by at most the 0.46 GPU-h pre-traffic allowance, i.e. 4.64
   (B) or 4.18 (A0). Never automatic.
3. Anything else needs a new amendment after a diagnosis.

**Authorization (22g, changed by 22h):** variant B, ceiling **6.2** GPU-h, policy single, existing early-stop gates
plus the batch guard, no automatic retry; DRY_RUN with APPROVED_GPU_H=6.2. The superseded 4.18 did not complete the
session under single-token or strongly unequal advance (section 11.1).

---

## 12. Limits, infeasible items and open risks

1. **Sentinel limits (22f s.5).** MLA / index-K shared-block identity is directly checked; KDA restoration and
   captured-state consistency are checked; physical KDA block identity and the complete unsampled state are not
   established. KDA block ids are not in the guard's scheduler record (a guard change, not made); the sampled SSM state
   is a 256-value sketch.
2. **Not step-exact (2 cases):**
   - d7's phase-2d aborted streams; their cached counts are still exact;
   - whether the KDA groups carry the EAGLE flag, which is behaviour-equivalent on every recorded workload.
3. **The replay's timing is modelled.** The cache outcome depends on order, which is recorded and gated, not timed.
   The design scenarios draw acceptance from C1 only. The acceptance stress scenarios (section 2.4) bound it from both
   sides, with every lookup a hit and every lead below 512. The step time is modelled as independent of acceptance.
4. **GATE-1R idles the GPUs for about 270 s** (priced) so that s2 survives to D2. Giving D2 its own build instead would
   cost a build (about 280 s) and break the construction balance.
5. **LP restore-depth imbalance** (section 4.2) is reported. LP attribution is limited to the descriptive same-depth
   pairs; a conclusion that depends on the depth-mismatched pair is "unresolved construction sensitivity" [R-14].
6. **The pool floor is 1,150 blocks**, refused before traffic. Five sessions have all logged 1,275.
7. **The long-context decode-window instrument is validated in-session only.** Every GATE-1R numerical row is a first
   long-context measurement with C2 (N_dd); C1 had N_dd absent.
8. **Unchanged from attempt 1:** coarse resolution with 3 runs per arm, operational screens only; what a null result
   cannot exclude; doc 79's three fixtures; disk (≥ 200 GB free required).
9. **Non-discriminating F6 pages [R-12]** are reported as "not independently verified by the numerical check". In the
   calibration these were 7 of 2,486 C1 pages, all M4.
10. **Acceptance-stress cost [R-11]:** the superseded 4.18 ceiling did not complete the session under single-token or
    strongly unequal advance (section 11.1). Under 22h (6.2 with the batch guard, section 11.2) the primary completes
    in every stress scenario. At pessimistic rates the LP control after the gate is skipped.
11. **The batch guard's decode prior** (11.0 tok/s per request) sits below the central single-token rate but above
    the pessimistic single-token rate (about 8.4). A first batch of a form can therefore run a few percent past its
    projection. This is covered by the slack early in the session and, at worst, by the hard watchdog. A prior of 8.0
    would have skipped every LP batch even at measured acceptance.

## 13. Preconditions before any launch

1. The package SHA256SUMS is recorded in an amendment and the tests pass (RETRY-FREEZE.md).
2. Independent review of the revised package (22d).
3. Operator authorization: item fcr1, a GPU-hour total, the variant and the attempt policy.
4. GPUs idle, and desktop graphics clients on GPU 1 closed if possible. The pre-flight reports them.
