# B1 checkpoint pass at seed 1236: frozen interpretation

Written 2026-10-13, before any B1 GPQA generation (PLAN.md amendment 19). No B1 GPQA data exists at the time of writing.
`b1screen_analysis.py` implements these rules. Changing either file after freeze changes `SHA256SUMS`, and the runner then
refuses to start.

## What is run

**Checkpoint B1:**
- recipe `glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234+d7`;
- image `sha256:799d3cf8…` (= B0 image `08d1cb90…` plus only the D7 file);
- one GPQA Diamond pass of all 198 questions at request seed 1236 (protocol pass 3, lm-eval `--seed` 1237);
- protocol gpqa-diamond/v1 unchanged: concurrency 8, 327,680-token cap, temperature 1.0, top_p 0.95, thinking on,
  lm-eval's parser;
- fresh server, release warm-up gate, shipped graph configuration, recording proxy.

This is checkpoint characterisation, not a D7 attribution experiment. B0 has no seed-1236 pass, so no individual D7 effect
is inferred from this pass.

## References: seeds confirmed from the recorded data

| role | pass | request seed | lm-eval `--seed` | session position | E (empty / 198) | raw | stated |
|---|---|---|---|---|---|---|---|
| **primary reference** | stock 0.9.1 record pass 3 (`+v0.9.1+ps__pass3`, 2026-10-09) | 1236 (198/198 logged) | 1237 | 2nd pass of a fresh server | 6, all cap (docs 79, 81, 88, 127, 147, 170) | 171 | 174 |
| context | column-511 arm pass 3 (`+col511fix__pass3`, 2026-10-11) | 1236 (198/198) | 1237 | 2nd pass of a fresh server | 6, all cap | 171 | 174 |
| context | stock 0.7.0 record pass 3 (`+ps__pass3`, 2026-10-10) | 1236 (198/198) | 1237 | 2nd pass of a fresh server | 1, cap | 175 | 177 |
| B1 (this pass) | `+bundle-d1234+d7__pass3` | 1236 | 1237 | 1st and only pass of a fresh server | — | — | — |

Sources:
- request seeds: the recording-proxy logs;
- lm-eval seeds: `results_*.json` (`model_args.seed`, `random_seed`);
- published rows: `runs/2026-10-08_…v0.9.1…` and `runs/2026-09-29_…v0.7.0…`, where pass 3 carries seed 1236. The analysis
  reproduces both published files row by row.

## Primary measures

- **Empty replies E of 198**, split into:
  - cap exhaustion: finish `length` at 327,680 completion tokens;
  - premature stopping: empty with finish `stop`;
  - other.
- **Rescued, newly empty and empty-in-both questions** against each reference.
- **Stated-answer correctness over all 198 requests.**
  - Every row where the stated adjudicator (export_evals `final_answer`, sha256 731f27d4…) differs from flexible-extract
    is read by hand.
  - A reply is hedged (not correct) only if it commits to no single option (the rule in `arm-col511/adjudications.json`).
  - Until every such row is read, the category is PROVISIONAL.
- **Raw (flexible-extract) correctness**, kept for continuity.
- **Generated tokens over all requests:** total, quantiles, and the paired difference against the primary reference.
- **Occupied GPU-hours** (2 × server start-to-stop wall) and **throughput** (request window, aggregate tok/s).
- **Validity checks:**
  - execution errors and preemption, illegal-memory and NaN lines;
  - one server start;
  - effective settings equal to B0's sessions apart from the image.

## Categories (amendment 15b scheme, unchanged; baseline E = 6 from the matched stock pass)

**Accuracy standard (unchanged from 15b).** The six-pass stock-0.9.1 family: record passes 1-3 and arm passes 1-3.
- Raw 169, 176, 171, 164, 170, 171.
- Stated 170, 178, 174, 165, 176, 174.
- E 5, 5, 6, 4, 5, 6.

Flags:
- **Material concern:** raw ≤ 158 or stated ≤ 159 (mean − 3 SD), or 2 or more premature stops.
- **Accuracy concern:** raw 159-163 (below every family pass).

| 15b category | condition |
|---|---|
| SIMILAR OR WORSE | material concern (any E), or E ≥ 6 |
| PROMISING FOLLOW-UP SIGNAL | E ≤ 2 and no accuracy concern |
| MODEST IMPROVEMENT, NO CLEAR SEPARATION | otherwise: E = 3-5, or E ≤ 2 with an accuracy concern |

**Amendment-19 reading.** This is the deciding statement for this checkpoint. It is derived from the 15b category, not
from a new test.

| reading | condition | consequence |
|---|---|---|
| **FAVOURABLE** | PROMISING FOLLOW-UP SIGNAL | Supports further evaluation and may be cited in publication drafts as one checkpoint pass. Any additional checkpoint pass is a separate operator decision. |
| **SIMILAR** | anything not favourable or unfavourable: E = 3-6 without a material concern, or E ≤ 2 with an accuracy concern | Reported as similar to the stock 0.9.1 family on completion, with direction and the paired counts. An accuracy concern, if present, is listed as a review item. |
| **UNFAVOURABLE** | a material concern, or E ≥ 7 (above every stock-family pass, max 6) | Triggers a review: the run's validity, the stopping and accuracy rows, and whether any correction interacts badly. It does not by itself invalidate the independently verified corrections (D1-D4 gates, D7 serving verification). |

**Sensitivity readouts.** These are reported but never decide the category:
- **Matched offsets against record pass 3**, as in 15b: flag at raw ≤ 160 or stated ≤ 163; concern at raw 161-165.
- **Amendment 14c's absolute empty-count rule:**
  - E ≤ 2: clear promising;
  - E = 3: similar;
  - E ≥ 4: similar or worse;
  - a flag or concern with E ≤ 3: mixed.

**Why baseline E = 6 and the family standard.**
- 15b anchors the empty count on the matched stock pass and the accuracy rules on the six-pass family. The same choice is
  kept here, with no new thresholds.
- Record pass 3's E = 6 is also the family's maximum.
- The UNFAVOURABLE empty-count limit uses the family range (above every stock pass), in the same way that 15b's accuracy
  concern uses "below every family pass".

## What is not claimed

- No significance, equivalence or non-inferiority claim. This is a single pass.
- A favourable result is not a demonstrated completion improvement.
- A similar result is not evidence that the corrections lack behavioural effects.
- No D7 effect, and no effect of any individual correction.
- No explanation of the 0.7.0 advantage. 0.7.0's seed-1236 pass (E = 1) is context only.
- **Seed reuse across engines (CONFOUNDS #18).**
  - Seed 1236 was already used by the stock record, the arm and 0.7.0.
  - At the same seed, a different engine is a different draw, not a paired copy, once logits differ.
  - Rescued and newly empty replies are descriptive pairings, not matched trajectories.
- **Historical controls.** The comparators ran on 2026-10-09 to 10-11, each as the second pass of its server session.
  This pass is the first pass of a fresh server. Exchangeability is assumed, not designed.
- **Reporting.**
  - Checkpoint records stay separate: B0 is seeds 1234 and 1235; B1 is seed 1236.
  - Pooled scores are never labelled a three-pass B1 result.
  - The part-B corrected-family view (B0 1234 + B0 1235 + B1 1236 against record and arm passes 1-3) is labelled
    EXPLORATORY: it mixes two implementations, and the B1 pass was chosen after B0's passes were seen.

## Invalid or truncated runs

A run is not a completed one-pass result if any of the following happens:
- any validity check fails;
- the watchdog fires (7.5 occupied GPU-hours);
- the pass is incomplete.

Such a run gets no category and is reported as invalid, with its cost. There is no automatic retry and no further pass.
