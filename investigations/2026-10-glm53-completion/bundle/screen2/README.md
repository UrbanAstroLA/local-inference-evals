# Correctness bundle: second GPQA pass, seed 1235 (amendments 15-15c)

**Result.** The unchanged frozen bundle (D1-D4 on tpurtell 0.9.1 K3.25) ran one more GPQA pass at seed 1235, all 198
questions, original protocol. The pass was valid and left **3 of 198 replies empty, all at the 327,680-token cap** (0
premature stops), against 5 in the stock 0.9.1 record's seed-1235 pass (the primary comparator). Under rules frozen
before generation: no material or accuracy concern on the family standard (raw 168, stated 169); category **MODEST
IMPROVEMENT, NO CLEAR SEPARATION**. Against the matching stock pass, raw correct is 8 lower and stated 9 lower; the
matched-offset readout, reported alongside but not deciding, shows "concern". An **exploratory** combination of both
bundle passes against the matching stock passes gives 6 vs 10 empty replies of 396 (reduction 1.01 points, 95% 0.00 to
2.27; one-sided p = 0.11). Per amendment 15 the direction and uncertainty are reported and no further pass follows
automatically. Cost 4.541 occupied GPU-hours (ceiling 7.5).

Plan and rules: [`../../PLAN.md`](../../PLAN.md#dated-amendments), amendments 15 (proposal), 15a (authorization), 15b
(freeze) and 15c (result). The first screen (seed 1234) and the bundle itself: [`../README.md`](../README.md). Investigation
page: [second pass](../../README.md#second-pass-seed-1235).

## What this pass is

An independent assessment proposed one more pass of the same bundle to see whether the first screen's result depended on
its seed (amendment 15). The seed-1234 screen keeps its classification, **SIMILAR / NO CLEAR SEPARATION**; nothing is
reclassified. Seed 1235 is the next protocol seed, chosen by protocol order rather than by scores. This pass was chosen
after the first screen had been inspected, so any analysis that combines the two passes is labelled exploratory.

## Frozen before generation (amendment 15b)

| Item | Value |
|---|---|
| Image | `sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548` (the first screen's candidate image; the container inspected after the run reports it) |
| Recipe | `glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234`, `env.sh` sha256 `3125683b…` (checked by the runner) |
| Combined diff | `949b2301…` ([`../freeze/combined-d1234.diff`](../freeze/combined-d1234.diff)) |
| Runner | [`gpqa-bundle-screen-s1235.sh`](gpqa-bundle-screen-s1235.sh), local sha256 `39ff7616e385546c35939eed2c914b7fc8478145347314ee6f4a014391b2c0f7` |
| Analysis | [`screen2_analysis.py`](screen2_analysis.py), local sha256 `cc3d300032cd831deea233c7e2251477b807170cbee4f80a4bd611efafef11c7`; the runner refuses to start on any other hash |
| Arm loaders | [`analysis_col511.py`](analysis_col511.py), local sha256 `b5b13ce2…`, unchanged from the column-511 arm (amendment 11b) |
| Budget | estimate 5.0-6.0 occupied GPU-hours; watchdog ceiling 7.5 (13,500 s wall on 2 GPUs); no automatic retry, no third pass |

The hashes above are of the local files that ran. The public copies replace local paths and the local eval-chain script
name with placeholders, so they are not byte-identical; [`../../PUBLIC-COPIES-15.txt`](../../PUBLIC-COPIES-15.txt) lists
both hashes. The result files' `script_sha256` and `analysis_col511_sha256` refer to the local files.

## Session conditions: this pass and its comparators

The eval chain skips an existing pass directory and starts a fresh server unless told otherwise. With `GPQA_PASSES=2`
and the bundle's pass 1 present, it skipped pass 1 and ran protocol pass 2 as the first pass of a fresh server. The
comparators' seed-1235 passes were also protocol pass 2, each the first pass after a fresh server start.

| | Stock 0.9.1 record, seed 1235 | Column-511 arm, seed 1235 | Bundle, this pass |
|---|---|---|---|
| Date | 2026-10-09 | 2026-10-11 (session 2) | 2026-10-12 |
| Session | fresh server; "pass 1 exists, skip" | fresh server; "pass 1 exists, skip" | fresh server; "pass 1 exists, skip" |
| Server ready after | 709 s | 740 s | 770 s |
| Request seed / lm-eval `--seed` | 1235 / 1236 | 1235 / 1236 | 1235 / 1236 |
| Task hash `gpqa_diamond_thinking` | 87e807b3b27c | 87e807b3b27c | unchanged (prompts identical to both comparators on all 198 questions, by `messages_sha256`) |
| Client | 8 concurrent, timeout 10,800 s, 3 retries, cap 327,680 | same | same |
| Afterwards | pass 3 on the same server | pass 3 on the same server | stopped (no third pass) |

Documented differences: the comparators continued into pass 3 after pass 2 had finished, which cannot affect pass 2; the
repository commit differs, which is not part of the request or protocol; the stock record's pass 1 carried no explicit
seed (the model default, 1234), while every pass 2 carried 1235 explicitly. The server's non-default arguments and logged
environment variable names hash to the same values as in the first screen and in the column-511 arm's seed-1235 session.
Desktop graphics occupy about 2.2 GiB on the second GPU (accepted in amendment 11b); the runner's pre-flight found no
other compute process.

## Rules

**Primary outcome.** E = empty final replies of 198, split into cap exhaustion, premature stop and other.

**Comparators.** Primary: the stock 0.9.1 record's seed-1235 pass (E = 5, all at the cap: docs 55, 79, 81, 121, 147; raw
176; stated 178). Secondary: the column-511 arm's seed-1235 pass (E = 5; raw 170; stated 176). Context: 0.7.0's
seed-1235 pass (E = 3; raw 175; stated 178).

**Categories (amendment 15).**

| Category | Condition |
|---|---|
| SIMILAR OR WORSE | a material concern (any E), or E ≥ 5 |
| PROMISING FOLLOW-UP SIGNAL | E ≤ 2 and no accuracy concern |
| MODEST IMPROVEMENT, NO CLEAR SEPARATION | E = 3-4, or E ≤ 2 with an accuracy concern |

These names are amendment 15's. The first screen's rules (amendment 14c) called E = 3 against a baseline of 5 "similar /
no clear separation"; the same count here carries a different label because the rule set is different, not because the
two passes differ.

**Accuracy and stopping thresholds (deciding): an absolute standard from the stock 0.9.1 family.** The six stock-0.9.1
passes on all 198 questions (record passes 1-3, column-511 arm passes 1-3) scored raw 169, 176, 171, 164, 170, 171 (mean
170.2, SD 3.9) and stated 170, 178, 174, 165, 176, 174 (mean 172.8, SD 4.6).
- Material concern: raw ≤ 158 (mean − 3 SD, 6 below the family minimum), stated ≤ 159 (mean − 3 SD), or 2 or more
  premature stops.
- Accuracy concern: raw 159-163 (below every family pass).

**Matched offsets: reported, not deciding.** The first screen's offsets applied to the matched pass (flag raw ≤ 165 or
stated ≤ 167; concern raw 166-170) are computed in every result as a readout. They do not set the category because:
1. Record pass 2 is the family's best pass (176 raw). Anchoring on one pass adds its sampling error to the threshold and
   builds in regression to the mean: with accuracy unchanged, a pass near the family mean of 170 lands in the concern
   band. Under a normal approximation over the six passes (inferred), P(concern or flag | no change) is about 0.53 for
   the matched offsets, against about 0.04 (concern) and 0.001 (flag) for the family standard.
2. The dry run shows it: the column-511 arm's own seed-1235 pass (raw 170) is labelled "concern" by the matched offsets,
   although that correction showed no accuracy difference over 594 question-passes (−1.85 points, 95% −5.05 to +1.35).
3. The offsets were set against a seed-1234 baseline of 169, near the family mean; at seed 1235 they are stricter.

The family standard has its own limit: six passes give a coarse SD. The matched differences are always reported, so a
large drop against the matched pass stays visible.

**Stated adjudication.** Every row where the stated adjudicator differs from flexible-extract is read by the existing
rules before the final category; until then the result is PROVISIONAL.

**Exploratory combined analysis.** Both bundle passes (seeds 1234 and 1235) against the matching stock passes (record
passes 1-2; secondary, arm passes 1-2), with the question as the unit: d_q = comparator empties − bundle empties over the
two seeds; an exact sign-flip test (one-sided for fewer bundle empties; two-sided for reference) and a 20,000-replicate
question bootstrap (seed 20261012) of the empty-rate difference; raw and stated differences with bootstrap intervals.
Labelled EXPLORATORY: historical controls, exchangeability assumed, and the second pass was chosen after the first had
been inspected.

**Dry run before generation** ([`dryrun-arm-pass2.txt`](dryrun-arm-pass2.txt), with the arm's own hand reads standing in
for the adjudication file, [`dryrun-adjudication-arm-pass2.json`](dryrun-adjudication-arm-pass2.json)): the column-511
arm's seed-1235 pass in place of the bundle's. Valid (30 of 30 checks); E = 5 → SIMILAR OR WORSE; no family-standard
concern; matched-offset readout "concern".

## Results

[`run-s1235/screen2-result-final.json`](run-s1235/screen2-result-final.json) (after adjudication);
[`run-s1235/screen2-result.json`](run-s1235/screen2-result.json) (before: PROVISIONAL, same category pending). The
result files call the new pass "pass 1" (the loader's index for a single pass); it is protocol pass 2 with request seed
1235, as the checks record.

**Validity.** 30 of 30 checks: 198 requests, none re-sent, every request HTTP 200 with seed 1235, cap 327,680,
temperature 1.0 and top_p 0.95; no parse errors; finish reason stop or length on every request; prompts identical to both
comparators; one server start, no preemption, illegal-memory-access or NaN lines in the request window. The watchdog did
not fire ([`run-s1235/watchdog.log`](run-s1235/watchdog.log) is empty).

**Stated adjudication** ([`run-s1235/adjudication.json`](run-s1235/adjudication.json); ids and verdicts only): 5 rows
(docs 64, 94, 110, 113, 179); doc 94 hedged, the other four commit to a single option.

### The seed-1235 pass, reported separately

| Seed 1235 (protocol pass 2) | Bundle (this pass) | 0.9.1 record (primary) | Column-511 arm (secondary) | 0.7.0 (context) |
|---|---:|---:|---:|---:|
| Empty replies | **3** (3 cap, 0 premature) | 5 (5 cap) | 5 (5 cap) | 3 (3 cap) |
| Rescued by the bundle | | q55, q81 | q74, q108, q120, q127, q186 | q81 |
| Newly empty with the bundle | | none | q79, q121, q147 | q147 |
| Empty in both | | q79, q121, q147 | none | q79, q121 |
| Raw correct of 198 (secondary) | 168 | 176 (bundle −8) | 170 (−2) | 175 (−7) |
| Stated correct of 198 (secondary) | 169 | 178 (−9) | 176 (−7) | 178 (−9) |
| Raw right only in one of the pair | | bundle 6, record 14 | bundle 13, arm 15 | bundle 5, 0.7.0 12 |

- **Category: MODEST IMPROVEMENT, NO CLEAR SEPARATION** (E = 3). Family standard: no material concern, no accuracy
  concern (raw 168, stated 169, 0 premature stops).
- **Matched-offset readout (not deciding): "concern"** (raw 168 lies in 166-170; no flag).
- Raw and stated counts are a single pass each and secondary; no significance claim is made for them. Against the matched
  stock pass the bundle is 8 lower raw and 9 lower stated; that pass is the family's highest, and 168 lies inside the
  family's range (164-176).
- Answered completion tokens (195 replies): median 3,145, 90th percentile 31,953, maximum 116,281; 19 above 32,000, 3
  above 64,000, none above 128,000.
- Request window 2.05 h; 3,040,269 completion tokens, 411.7 tok/s aggregate, median 58.0 tok/s per answered request;
  server mean 437.2 tok/s with 8 running; peak KV use 31.9%; 84% of busy time with 5 or more requests in flight.
- Cost: **4.541 occupied GPU-hours** (2 GPUs × 8,174 s, server start to stop; [`costs.jsonl`](costs.jsonl)); below the
  5.0-6.0 estimate, ceiling 7.5 not reached.

### Both bundle passes against the matching stock passes (EXPLORATORY)

EXPLORATORY: historical controls; the follow-up pass was chosen after the first screen had been inspected; each seed is
reused from the comparators' passes (a matched seed is not a paired draw once the computation diverges; see
[confounds](../../../../CONFOUNDS.md#18-seed-reuse-across-engines)). 396 question-passes per side.

| Bundle seeds 1234 + 1235 vs | Empty: bundle vs comparator | Empty-rate reduction, points (95%) | Sign-flip p, one-sided (two-sided) | Questions fewer / more bundle empties | Raw difference, points (95%) | Stated difference, points (95%) |
|---|---|---|---|---|---|---|
| 0.9.1 record, passes 1-2 | 6 vs 10 | 1.01 (0.00 to 2.27) | 0.11 (0.22) | 5 / 1 | −0.76 (−3.79 to +2.27) | −0.76 (−3.54 to +2.02) |
| Column-511 arm, passes 1-2 | 6 vs 9 | 0.76 (−0.76 to +2.27) | 0.25 (0.51) | 6 / 3 | +2.02 (−1.77 to +6.06) | +1.01 (−2.53 to +4.55) |

By seed: bundle 3 and 3, stock record 5 and 5, column-511 arm 4 and 5; stock 0.7.0 (context) 0 and 3.

## Limits

- One pass at one more seed; a low-rate outcome (E of 3 to 5 of 198) cannot separate modest differences in one pass.
- Historical controls with exchangeability assumed; the comparators ran on 2026-10-09 and 2026-10-11.
- The combined analysis is exploratory, not a confirmation: the follow-up was chosen after inspecting the first screen,
  and both seeds are reused from the comparators.
- Reusing a seed across engines matches the sampling stream only until the first computational divergence; D3 alone
  changes about 3% of expert sets, and long-context greedy decoding of the stock release is not repeatable, so the
  bundle's replies are not paired copies of the comparators'.
- The bundle changes four things at once; nothing here attributes an effect to an individual correction.
- This pass does not establish that seed 1234 was unrepresentative, a general improvement, an explanation of 0.7.0's
  lower rate, or equivalence. The plan notes that any confirmation would prefer a contemporaneous stock control on new
  prespecified seeds (amendment 15).

## Files

- [`gpqa-bundle-screen-s1235.sh`](gpqa-bundle-screen-s1235.sh): runner (pre-flight checks, watchdog, recording proxy,
  analysis).
- [`screen2_analysis.py`](screen2_analysis.py), [`analysis_col511.py`](analysis_col511.py): the frozen analysis and the
  arm loaders it imports (the latter byte-identical to the existing public copy in
  [`../../arm-col511-plan/`](../../arm-col511-plan)).
- [`dryrun-arm-pass2.json`](dryrun-arm-pass2.json), [`dryrun-arm-pass2.txt`](dryrun-arm-pass2.txt),
  [`dryrun-adjudication-arm-pass2.json`](dryrun-adjudication-arm-pass2.json): the dry run.
- [`run-s1235/`](run-s1235): the result before (`screen2-result.*`) and after (`screen2-result-final.*`) adjudication,
  the adjudication ids and verdicts, the watchdog log and the 5-minute GPU memory and utilization log.
- [`screen2-run.log`](screen2-run.log): runner console output. [`costs.jsonl`](costs.jsonl): the cost line as logged.
- [`RAW-SHA256SUMS-screen2.txt`](RAW-SHA256SUMS-screen2.txt): sha256 of files kept local (server log, container
  inspection, the local README written before the run, lm-eval samples and results, recording-proxy request log).
