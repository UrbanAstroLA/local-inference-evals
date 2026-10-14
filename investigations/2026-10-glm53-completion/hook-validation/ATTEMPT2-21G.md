# Hook validation, attempt 2: amendment 21g implemented (CPU only; NOT launched)

2026-10-14. This brings the attempt-2 package in line with PLAN.md amendment 21g (and 21e/21f). It sits on top of
`ATTEMPT2-PREP.md`. Where the two differ, this document wins.

How it was done:
- **No GPU work.** No GPU or glm53 serving container was started, and nothing used `--gpus`. Only CPU containers ran,
  with `--network none` and `CUDA_VISIBLE_DEVICES=`. The runner ran only as `DRY_RUN`, inside the shim tests.
- **No `sudo`.**
- **Attempt-1 artefacts untouched.** `run-a1/`, `diagnosis-a1/`, `tests/out/`, `build/` and every `*.a1` copy still
  hash as frozen. `prep-a2/pa2_changes.py` checks this.
- **The hook and the image did not change.**
  - `hook/forcewin_hooks.py` is `15e93e23…`, the same as before 21g.
  - The image is `sha256:ac05230bd94b3114cb6fb202b06402f9e2d45a61b20f041f314e045f967e97ee`, as before.
- **Every change is post-inspection and labelled PA2-16 to PA2-25**, with before/after hashes in
  `prep-a2/PA2-changes.json`.
  - "Before" means the package the 21g review saw. Its sums are kept as `SHA256SUMS.pre21g` (sha256 `2ab92f14…`).
  - Each file 21g modified keeps its pre-21g version as `<file>.pre21g`.
  - New outputs go to new directories (`prep-a2/out-21g`, `prep-a2/replay-a1-21g`, `prep-a2/faultinj-a1`,
    `tests/out-21g`), so the pre-21g evidence stays as it was.

**All affected CPU suites pass.** `tests/run_21g.sh` → `tests/out-21g/run_21g.log`: S0-S5 all PASS. The runner gate
tests also pass (`tests/out-a2/runner-tests.log`).

## 1. Index-K content gate (21g-1)

**The rule.** gross ≤ L = max(2 × max(N_dd, N_pp), 0.1 %).
- The 0.1 % floor is an operational tolerance, not a measured noise floor.
- N_dd = V4-b1 vs V4n-b1b: decode against decode, batch of 8 against batches of 4.
- N_pp = V4-rebuild vs V4n-rebuild2: prefill against prefill, with the build 124-200 ids longer.

**Do the nulls match the tested comparison? Yes, for every fixture of every gated comparison.**
- `tests/check_v4n_matched.py` checks the client plan: **169/169** checks pass.
  - For each of the 8 D2 fixtures, the decode arms (b1, b1b, d2, and solo for cbac17) have equal P, W, prefix and window
    hashes.
  - Both rebuilds dump exactly [P, P+W].
  - All of them hold the same token ids at those positions. This is the new `dump_src_sha` record.
  - b1b runs as two batches of 4.
  - A tampered plan, with the second rebuild misplaced by 4 positions, fails the check (negative control).
- Layers and ranks are matched in the analysis, one unit at a time.
- **No V4n extension was needed**, so the cost is unchanged.
- **Gap: V3 has no matched null.** Its traces and positions are not in the D2 set, and adding one would need new GPU
  requests. V3 content (`H4c-content`) is therefore descriptive only and not gated, as before. V3 stays under the
  structural checks.

**Frozen definitions** (`analysis/analyze_hookval.py` docstring; constants in PA2-16):

| Item | Definition |
|---|---|
| Unit | (fixture, rank, DSA index layer, pool), counted once |
| Pools compared | Those wholly inside the window and processed: (P+3)//4 ≤ i and 4i+3 ≤ P+W−2 |
| rel and gross | rel = ‖x−y‖/‖y‖ on the dequantised fp8 records; gross means rel > 0.25 |
| Matching | A tested unit is evaluated only if the same unit exists in **both** nulls of its fixture. The limit is computed from the nulls restricted to exactly the evaluated units |
| Aggregation | Pooled over units: gross units divided by evaluated units. Never a mean of per-fixture rates |
| Exclusions | Only the window-end pool, the straddling first pool, and units without a matched null (counted) |
| Null coverage | Evaluated/tested ≥ 99 %, and every fixture has evaluated units. Otherwise the content row FAILS at the inconclusive level |

**What is gated:**
- **IDX-content:** both B1 arms, each against its own matched limit.
- **H5e-r** (changed by PA2-26, amendment 21h): the solo request, over its matched units, against the **pooled**
  matched null of the eight-fixture set. That null uses the same unit definition, pooled denominators and exclusions
  as IDX-content. The solo fixture's own matched null (cbac17) is now only a **flag** (`H5e-fixture`, REPORTED, not a
  failure). The pre-21g pooled form (`H5e-pooled`) is still reported beside it.
- **H5d-r:** the D2 arm above 30 %, also reported as a multiple of its matched limit.
- **H5f-D2:** clean-labelled D2 pools against the limit of the nulls on those same units.

**What is reported:**
- Gross counts and rates per fixture × layer and per fixture, pooled, for every tested arm and both nulls
  (`verdict-a2.json` → `idx_content_table`).
- The pre-21g unmatched form, kept beside the gated rows (`content_draft_unmatched`, `H5e-pooled`).
- **Fixture flags** (`IDX-fixture-flags`, REPORTED). A flagged fixture cannot carry a mechanism claim. Both flags were
  defined before any data:
  - **Outside its own null:** gross_f > max(2 × max(N_dd_f, N_pp_f), 0.1 %), with at least 10 gross units.
  - **Arm asymmetry:** on the units both B1 arms and N_pp share, the grosser arm has at least 10 gross units, more than
    2× the other arm's, and more than max(2 × N_pp_f, 0.1 %).

**Structural checks (exact):**

| Check | What it requires |
|---|---|
| IDX-attrib (new) | Every window dump carries its request's tag, P and window range. A rebuild's range must equal its fixture's B1 window. Every V3 snapshot carries its request's tag, P and F. Every comparison compares ≥ 99 % of its expected units |
| IDX-page | No judged page above 25 % gross; unique pools; pages with ≥ 16 compared pools |
| IDX-early | Zero gross units in layers 3-15 |
| IDX-stable | Committed pools byte-identical between consecutive V3 snapshots |
| H5f-B1 | The B1 ledger is all clean |

The original criteria and their results stay beside the revised ones:
- `verdict.json` holds the frozen attempt-1 code, run verbatim.
- Every revised row lists its original checks and their results.

**Evidence:**
- **Synthetic `bad`.** Two plants target the content rule:
  - u4 and u5: the same 26 pools are wrong in both B1 arms. This fails IDX-content (b1 4.42 % vs L 2.53 %) and flags
    u4 and u5.
  - u3: one arm only. This raises the arm-asymmetry flag for u3.
- **Synthetic `good`.** Content passes and nothing is flagged.
- **Attempt-1 replay.**
  - The content rows FAIL at the inconclusive level with "no matched units". That is the coverage rule working:
    attempt 1 had no V4n.
  - IDX-attrib finds 0 attribution violations, and population is 100 % in every comparison present.
- **Negative controls on attempt-1 captures under the revised analysis** (`prep-a2/run_faultinj_a1.sh` →
  `prep-a2/faultinj-a1/summary.txt`: ALL AS EXPECTED):
  - The base passes every targeted check: IDX-attrib, IDX-page, IDX-early, IDX-stable, H5f-B1, H5f-D2, H2a-r, H2c-r,
    H4-slot, H4b-r and H4b-sep.
  - Each fault fails its targets:

| Fault | Fails |
|---|---|
| Wrong page in a B1 dump | IDX-page, IDX-early |
| Wrong page in a V3 snapshot | IDX-stable, IDX-early |
| B1 dump carrying another fixture's pages | IDX-page, IDX-early |
| V2a record carrying another request's measurements | H2c-r |
| V3 snapshot carrying another request's state | H4b-r |
| Attempt 1's real (request-shifted) B1 ledger | H5f-B1 |
| B1 ledger shifted by one pool | H5f-B1 |
| D2 ledger shifted by one pool | H5f-D2 |
| D2 ledger shifted by one request | H5f-D2 |

**Limits of this evidence (for the reviewer):**
1. **Attempt 1 has no V4n.** The fault-injection base therefore uses proxy V4n arms (byte copies, so the nulls are
   0), and the content gate cannot be shown on real captures. It is shown on the synthetic runs only.
2. **The base's ledgers are constructed.**
   - B1 is repaired to its by-construction all-clean value.
   - D2 uses oracle labels taken from the bytes.

   The ledger cases therefore show that a **shifted** ledger is detected. They do not show that the hook's ledger is
   correct; that is the GPU positive control H5f-B1, with CPU test D7p.
3. **A one-pool shift of the B1 ledger is caught only at the window boundary.** All-clean labels stay clean when
   shifted. The request shift attempt 1 actually had is caught broadly (881 records).
4. **Design property of the adopted formula: N_dd shares an arm with each B1 comparison.**
   - A defect confined to one decode arm inflates N_dd, so the limit absorbs it.
   - A localised one-arm fault inflates the **pooled** limit for every fixture. In an earlier synthetic layout, a
     wrong page in b1b raised L to 10 %.
   - Such faults are caught by the structural checks and the arm-asymmetry flag, not by the content gate. The bad
     run's wrong-page plant was moved to the solo request for this reason (PA2-19).
5. **H5e-r risk resolved by amendment 21h (PA2-26).**
   - Before 21h, H5e-r gated the solo request against cbac17's own matched null. On attempt-1 figures (solo 0.18 %,
     cbac17 decode null 0.11 %, prefill null 0.02 %) that risked a false INVALID.
   - The gate now uses the pooled eight-fixture null, and the per-fixture comparison is a flag.
   - Consequence, consistent with item 4: a localised fault in one decode arm also raises the pooled limit H5e-r uses.
     In the fault-injection cases with a wrong or foreign B1 page (F1-wrongpage-b1, F2-attrib-b1), the pooled N_dd
     rose to 19.6 % and H5e-r turned from FAIL to PASS. Those faults still fail IDX-page and IDX-early, their
     targets, and `H5e-fixture` flags the solo request.

## 2. KDA (21g-2)

**(a) H4-slot, exact, zero tolerance.**
- num_accepted equals the hook's own accepted count at the snapshot step + 1.
- 1 ≤ num_accepted ≤ the number of slots.
- Every KDA layer is speculative, with one slot count.
- The snapshot is attributed to its request (tag, P, F).
- Schedule prescription is reported, not gated, because H1j owns the schedules:
  - oracle: K;
  - poison: 0;
  - reject:j: j−1.

**(b) Nearest-reference identification as a diagnostic screen.**

| Row | Rule | Status |
|---|---|---|
| H4b-r | 100 % in layers 0-17; ≥ 99 % in layers 18 and up; rel ≤ 0.02 and margin ≥ 10 in **layers 0-2 only** | Gated |
| H4b-disagree | Every disagreement, with all slot distances, the committed and nearest slot, the margin, and the separation | REPORTED |
| H4b-sep | A wrong slot nearer than the committed slot by ≥ 1.25× is a comfortably separated wrong-slot match and requires investigation | Inconclusive level |
| H4b-draft100 | The pre-21g "100 % in every layer", kept beside | REPORTED |
| verdict.json | The original H4b | Original |

H4b-sep's 1.25 was fixed before any data. In attempt 1, correct-slot margins at depth fell to 1.04, with 14 below 1.25.

**Evidence:**
- **Attempt-1 replay:**
  - H4-slot: 106 snapshots, 0 violations; schedule prescription matched in 70 of 70.
  - H4b-r: PASS. 3,604 comparisons; 100 % in layers 0-17 (n 1,484) and in layers 18+ (n 2,120); layers 0-2 rel max
    0.0089, margin min 18.7.
  - 0 disagreements and 0 flags.
- **Synthetic `bad`:**
  - a bookkeeping mismatch whose identification still agrees fails H4-slot only;
  - a deep wrong slot about 20× nearer fails H4b-sep;
  - the layer-0 wrong slot fails H4b-r.
- **Fault injection:** a V3 snapshot carrying another request's state fails H4b-r.
- **Diagnostic only, not gated.** The KDA layers' metadata num_accepted equals the previous step's accepted count + 1
  in 104 of 106 snapshots. The two exceptions are V3-reject:2-57703691f509 at F521, frontier 4,351, one position
  before the 4,352 block boundary. They are unexplained and reported as `H4-meta`.

## 3. H2/H3 carry-over (21g-3)

**Image scope.** `prep-a2/scope_a1_a2.sh` → `prep-a2/out-21g/image-scope.json`.
- Every file of the installed vllm package was hashed in both images: **7,265** each.
- **The only source that differs is `vllm/forcewin_hooks.py`.**
- The five hooked engine files' `.pyc` differ only in the 16-byte header (source mtime). Their marshalled code is
  identical.

**Carried rows**, labelled "carried-over evidence", with provenance in `analysis/carryover-a1.json`:
- **H2-null-r:** the attempt-1 V1 null.
- **H2-a1-carried:** attempt-1 H2 under the **corrected join** (PI-1), top-1 0.983. The frozen join is not used.
- **H3-r:** carried unless the poison phase completed.

Provenance: source sha256 of run-a1 and diagnosis-a1 files, the evidence class, the join statement, and the scope.

**Admissibility rows:**
- **CARRY-scope:** the scope holds and this run used the attempt-2 image (`run-a2/image-id.txt`).
- **CARRY-admissible (invalid level):** the scope, plus these current-image checks:
  - V0;
  - forcing: H1a-e and H1g-j;
  - attribution: H5h-r and IDX-attrib;
  - ledger: H5f-B1;
  - state: H4a and H4-slot.

**Current-image recomputation.** Alignment (H2a-r) and the argmax shift control (H2b-r, H2c-r) are computed from the
overlapping (finite) V2a/V2c reference rows. Each row states which request families and how many rows it used. There
is no V1 rerun.

**Evidence:**
- **Attempt-1 replay:** H2a-r, H2b-r and H2c-r pass.
- **CARRY-scope and CARRY-admissible correctly FAIL on the attempt-1 data**, because of the attempt-1 image and its
  hook defects.
- **Synthetic `good`:** both pass.
- **Fault injection:** a swapped V2a record fails H2c-r.

## 4. Poison phase and coverage (21g-3)

**The poison phase is retained.** It runs last, after every required phase, and only if the time left covers 240 s.
- 240 s comes from attempt 1's 154 s for this phase, plus the dump and a margin.
- It is scaled up if this session's V2a-natural phase runs slower than attempt 1's 58.6 s.
- If skipped, the client records `optional_skip` with the time needed and the time left.
- An incomplete poison phase counts as not run: H3 stays CARRIED.

**Required coverage is never dropped silently:**
- The client writes a final `coverage` record (required phases run and not run; optional status).
- V3-refs writes a `coverage_gap` record when the hook log yields no frontier or a V3 request has no snapshot.
- The analysis's `coverage-a2` requires:
  - every mandatory phase;
  - every expected mandatory request, without error (from the start record's matrix and D2 set);
  - every V3 snapshot to have its reference;
  - no skip or coverage-gap record;
  - the poison phase to be ran, `optional_skip`, or incomplete. Absent with no record fails.

**Request payloads are unchanged.** The plan regenerated from the PA2-20 client matches the pre-21g plan for all 90
records, keyed by tag. Only two things differ: the new record fields, and the seed order, which follows thread start
order as before.

**Cost** (`runner/cost_a2.json`, recomputed; identical to before):

| Scope | GPU-hours | Item cumulative |
|---|---:|---:|
| Mandatory | 0.700 | 1.912 |
| + poison | 0.789 | 2.000 |
| Worst case at the 2,138 s wall cap | 1.188 | 2.3995 |

- Client room at the cap is 1,236 s, against 514 s mandatory and 674 s with the poison phase.
- The ceiling of 2.4 GPU-hours and the 2,138 s wall cap both hold.

## 5. top_p edge case (21g-4)

**H1f-r.** The gate is stated as applying to the **tracked** ids: the 4 termination ids and the forced token. A
violation is any of them above 1e-6.

**H1f-unmasked** (REPORTED) is the separate class of rows the served kernel left unmasked.
- It reports their tracked kept mass.
- It reports the **total pre-mask mass outside the intended nucleus**, where the existing records determine it.
  - The records determine it when the forced token is the argmax and mass_lt ≤ 1 − top_p. Then the nucleus is the
    top token alone, and the outside mass is mass_lt of the forced token, summed over the full vocabulary by the hook.
  - Elsewhere it is reported as "not determinable".
- No instrumentation was added, and the sampler and forcing path are unchanged.

**Attempt-1 replay:**
- 56 unmasked rows, all determinable.
- Outside-nucleus mass per row: median 2.6e-8, maximum 1.04e-7. The sum over the 56 rows is 1.7e-6.
- Tracked kept mass is at most 2.6e-9.

The bound in H1f-r therefore holds for the tracked ids, while the whole row retained up to 1.0e-7 outside the
nucleus. DIAGNOSIS-a1 §6 quoted a tracked-id tail quantity, not this total.

## 6. Suites run (all CPU)

| Suite | Result |
|---|---|
| S0 hook unchanged | PASS |
| S1 synthetic analysis | **good: revised VALID** (original INVALID by design). **bad: revised INVALID, failing exactly the 16 planted checks**: H1c, H1f-r, H2c-r, H3-r, H4-slot, H4b-r, H4b-sep, IDX-attrib, IDX-page, IDX-early, IDX-content, H5e-r, H5f-B1, H5g-r, H5h-r, CARRY-admissible |
| S2 plan | Client plan dry run: 89 requests, 72,692 forced tokens. V4n matched nulls 169/169. Tampered plan fails. Coverage record present |
| S3 attempt-1 replay (`prep-a2/replay-a1-21g`) | **Original verdict reproduced check-for-check (28 checks).** Revised: as below |
| S4 fault injection | ALL AS EXPECTED (section 1) |
| S5 cost | PASS (section 4) |
| Runner gates (`tests/test_runner_a2.sh`) | 16/16: refusals plus DRY_RUN |

S3, the revised checks on attempt-1 data:

| Group | Checks | Result |
|---|---|---|
| Diagnosis non-defects | H1f-r, H2a-r, H2b-r, H2c-r, H3-r, H4a, H4-slot, H4b-r, H4b-sep, IDX-stable, H5g-r, H5d-r | PASS |
| Hook defects | H5h-r, H5f-B1 | FAIL |
| No V4n in attempt 1 | coverage-a2, IDX-attrib, IDX-page, IDX-early, IDX-content, H5e-r, H5f-D2 | FAIL |
| Attempt-1 image | CARRY-scope, CARRY-admissible | FAIL |

**Not rerun: the hook suites** (forcing 66/66, D2 22/22, exposure 12/12, inertness). They test the hook and the image,
and neither changed (S0; image id unchanged).

## 7. Change log (PA2-16 to PA2-25)

Full before/after hashes are in `prep-a2/PA2-changes.json`.

| Id | Change |
|---|---|
| PA2-16 | Analysis: the 21g criteria above |
| PA2-17 | Image-scope script and output |
| PA2-18 | Carry-over record |
| PA2-19 | Synthetic runs |
| PA2-20 | Client records and poison need; plan regenerated; V4n plan check |
| PA2-21 | Fault injection |
| PA2-22 | 21g suites driver; attempt-1 replay |
| PA2-23 | Cost recomputed |
| PA2-24 | This document, and the notes in ATTEMPT2-PREP and CRITERIA-DRAFT |
| PA2-25 | Change log, sums, runner tests |
| PA2-26 | Amendment 21h: H5e-r gates on the pooled eight-fixture matched null; the per-fixture comparison becomes the `H5e-fixture` flag. Dry runs, replay and fault injection rerun (section 8) |

SHA256SUMS covers this document, so its own hash is recorded outside the package.

## 8. PA2-26 (amendment 21h): what changed in the reruns

`tests/run_21g.sh`: S0-S5 all PASS again. Row statuses were compared before and after PA2-26 for the synthetic good
and bad runs, the attempt-1 replay and all 10 fault-injection cases:
- **Every verdict is unchanged:** good VALID; bad INVALID failing the same 16 checks; replay INVALID; fault
  injection ALL AS EXPECTED.
- **The new row `H5e-fixture` appears everywhere.**
- **H5e-r changed status in two non-targeted places:** F1-wrongpage-b1 and F2-attrib-b1 go FAIL → PASS (section 1,
  limit 5).
- **H5e-r details changed** to show the pooled null.
- **Synthetic KDA details differ** (for example the margin minimum). That comes from the generator's unseeded
  `torch.randn` and predates 21h; the statuses are identical.

Before/after hashes are in `prep-a2/PA2-changes.json`, against `SHA256SUMS.pre21h`, the 21g package (sha256
`160015e5…`). The `*.pre21h` copies are kept.


**Launch:** unchanged, and only after an amendment records this package and the operator authorizes it:

```bash
APPROVED_GPU_H=2.4 <runs>/hook-validation/runner/run_hookval.sh 2
```
