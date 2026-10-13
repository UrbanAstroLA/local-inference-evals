# B0 checkpoint pass at seed 1236: frozen interpretation

Written 2026-10-13, before any B0 seed-1236 GPQA generation. No B0 seed-1236 data exists at the time of writing.
This is the review step that amendment 19 triggers after B1's seed-1236 pass came out UNFAVOURABLE (E = 9) under its
frozen reading. `b0s1236_analysis.py` implements these rules. Changing either file after the freeze changes
`SHA256SUMS`, and the runner then refuses to start.

## What is run

**Checkpoint B0:**
- recipe `glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234` (env.sh `3125683b…`);
- image `sha256:08d1cb90…`, the D1-D4 bundle. B1 is this image plus the D7 file only;
- one GPQA Diamond pass of all 198 questions at request seed 1236 (protocol pass 3, lm-eval `--seed` 1237);
- protocol gpqa-diamond/v1 unchanged: concurrency 8, 327,680-token cap, temperature 1.0, top_p 0.95, thinking on,
  lm-eval's parser;
- fresh server, release warm-up gate, shipped graph configuration, recording proxy.

**How seed 1236 is reached:** B0's real `__pass1` (seed 1234) and `__pass2` (seed 1235) exist, so the chain skips them
and runs pass 3 alone. No placeholders are used. B0's pass is the first and only pass of a fresh server, as B1's was.

## Why this pass

The D7 reconstruction (`d7/exposure/REPORT.md`) finds no D7 exposure on GPQA prompts in the 0.9.1 family. Where D7 does
not fire, B1's writes are byte-identical to B0's (CPU-verified). On GPQA, B0 and B1 should therefore compute the same
thing.

A B0-versus-B1 difference at seed 1236 would then measure run-to-run variation at a fixed seed: batch composition depends
on timing, and batched computation need not be batch-invariant. The other explanation is that the reconstruction is wrong.

**One pair of passes cannot separate D7 from run-to-run variation.** The reading below says which explanation the data
leave open. It does not decide between them.

## References

| role | pass | request seed | session position | E (empty / 198) | raw | stated |
|---|---|---|---|---|---|---|
| **primary for this review** | B1 `+bundle-d1234+d7__pass3` (2026-10-13) | 1236 | 1st and only pass of a fresh server | 9, all cap | 173 | 174 (doc 37 read, not hedged) |
| primary for continuity (15b) | stock 0.9.1 record pass 3 (`+v0.9.1+ps__pass3`) | 1236 | 2nd pass of a fresh server | 6, all cap | 171 | 174 |
| context | column-511 arm pass 3 | 1236 | 2nd pass of a fresh server | 6, all cap | 171 | 174 |
| context | stock 0.7.0 pass 3 | 1236 | 2nd pass of a fresh server | 1, cap | 175 | 177 |
| B0 (this pass) | `+bundle-d1234__pass3` | 1236 | 1st and only pass of a fresh server | — | — | — |

The analysis reproduces B1's final result from its files (E 9, all cap; raw 173; stated 174) as a validity check, and
reproduces the published stock and 0.7.0 rows.

## Primary comparison: B0 versus B1 at seed 1236

**Measures:**
- **Empty replies** for both passes, split into cap exhaustion, premature stop, and other.
- **Rescued, newly empty and empty in both:**
  - rescued = empty in B1, not in B0;
  - newly empty = empty in B0, not in B1.
- **Per-question agreement of empty status:** a 2×2 table (both, B0 only, B1 only, neither).
- **Raw and stated correctness.** Stated uses the same adjudicator and the same hand-read rule: a reply counts as hedged
  only if it commits to no single option.
- **Generated tokens:** totals, quantiles, and the paired per-question B0 − B1 difference with a question bootstrap.
  These are descriptive.

**Reply identity:**
- A question counts as identical when its final reply has the same `content_sha256` **and** the same
  `completion_tokens` in both passes.
- The count is split in two:
  - **I_ne:** identical non-empty replies. This is the count that is read.
  - **Both empty:** identical by construction, since both replies hash the empty string, and at the cap the token count
    is the same as well. This part is reported but not read.
- A stricter sub-count also requires equal reasoning characters and reasoning tokens, from the request logs.

The split is needed because of what existing data show (measured before freezing, from the request logs). Every
"identical reply" in the earlier same-seed pairs is a pair of cap-exhausted empty replies, and the reasoning lengths
differ in every one of them. Non-empty identical replies (I_ne):

| pair | identical by `content_sha256` + `completion_tokens` | of which non-empty (I_ne) | both empty at the cap |
|---|---|---|---|
| B1 vs stock record, seed 1236 | 5 | 0 | 5 |
| B1 vs col511 arm, seed 1236 | 5 | 0 | 5 |
| B0 vs stock record, seed 1234 | 2 | 0 | 2 |
| B0 vs stock record, seed 1235 | 3 | 0 | 3 |
| stock record vs col511 arm, seed 1236 | 4 | 0 | 4 |
| stock record vs col511 arm, seed 1235 | 0 | 0 | 0 |

These pairs all compare different builds. B0/B1 is the first same-seed pair that, under the reconstruction, compares the
same computation. The same counts are recomputed in the output (`identity_context_pairs`).

**First divergence cannot be measured.** No token-level logs exist: each request records only the final content hash,
the token counts and the reasoning length. Where two replies differ, the token at which they diverged is unknown.

### Frozen reading (deciding statement of this review)

The input is E, the B0 pass's empty count. B1's is 9.

| reading | condition | statement |
|---|---|---|
| **(a)** | E ≥ 8 | **The seed-1236 excess is shared by B0.** It is a property of the corrected family (B0 and B1) at this seed, and not attributable to D7: B0, without D7, shows it too. |
| **(b)** | E ≤ 6 | **B0 is within the stock range.** The B0/B1 difference is either D7, which would contradict the zero-exposure reconstruction, or run-to-run variation at a fixed seed. One pair of passes cannot separate them; I_ne gives only a weak lean (below). |
| **(c)** | E = 7 | **No reading.** |

**Why these thresholds:**
- The stock 0.9.1 family has E = 5, 5, 6, 4, 5, 6 over six passes (seeds 1234-1236, two builds), so its range is 4-6.
  Both stock-family passes at seed 1236 have E = 6. B0's earlier passes have E = 3 and 3.
- **E ≤ 6** puts B0 inside the stock range, including the stock value at this seed. B0 then does not share B1's excess.
- **E ≥ 8** puts B0 at least 2 above every stock pass and within 1 of B1. B0 then shares the excess.
- **E = 7** is one above the stock maximum, one below (a)'s threshold and two below B1. It cannot be assigned to either
  side.
- Differences of 2-3 empties between same-seed passes are ordinary in existing data. Examples: B0 3 against stock 5 at
  seeds 1234 and 1235; B1 9 against stock 6 at seed 1236.

**Identity lean, applied only in (b):**
- **I_ne ≥ 20, near-deterministic.** The B0/B1 difference is less readily explained by run-to-run variation. The
  predefined follow-up is a CPU re-check of the zero-exposure reconstruction. No D7 effect is claimed.
- **I_ne ≤ 2, batching nondeterminism dominates.** Same-seed passes behave as distinct draws, so a difference of this
  size is expected from run-to-run variation alone. No D7 effect is inferred.
- **I_ne 3-19:** no lean.

**How the identity count is read on its own** (reported in every case):
- I_ne ≥ 20 (about 10% of questions): **NEAR-DETERMINISTIC AT A FIXED SEED**. Single-pass differences between
  same-computation builds are then more meaningful.
- I_ne ≤ 2: **BATCHING NONDETERMINISM DOMINATES**. Single-pass comparisons at a shared seed are then comparisons of
  distinct draws, and seed matching does not pair trajectories.
- In between: **PARTIAL**.

The cut-offs are judgement, not derived. The only prior information is that every cross-build pair so far has I_ne = 0,
and 2 allows for a coincidental short reply.

**Review items, reported alongside any reading, never changing it:**
- a material concern for the B0 pass (raw ≤ 158, stated ≤ 159, or 2 or more premature stops);
- an accuracy concern (raw 159-163).

**Adjudication.** The reading depends only on E and I_ne, so it is final once the pass is valid. Stated counts and the
15b category stay PROVISIONAL until every row where the stated adjudicator differs from flexible-extract has been read
by hand.

## Continuity: B0 against stock 0.9.1 at seed 1236 (part A)

This uses amendment 15b's scheme unchanged, as in the B1 package:
- baseline E = 6, the matched stock pass;
- the six-pass family accuracy standard: material concern at raw ≤ 158, stated ≤ 159 or 2 or more premature stops;
  accuracy concern at raw 159-163;
- categories: SIMILAR OR WORSE (a material concern, or E ≥ 6) / PROMISING FOLLOW-UP SIGNAL (E ≤ 2 with no accuracy
  concern) / MODEST IMPROVEMENT, NO CLEAR SEPARATION (otherwise).

Comparisons: rescued and newly empty against record pass 3 (primary for continuity), the arm's pass 3 and 0.7.0's pass 3
(context), with the matched-offset and 14c readouts.

The amendment-19 FAVOURABLE/SIMILAR/UNFAVOURABLE reading is also computed, but it was defined for B1 and is reported
here **for continuity only**. It decides nothing about B0.

## Exploratory views (part B)

**B-1, new: the B0 three-seed record.**
- B0 at seeds 1234, 1235 and 1236 is compared, as one implementation, with stock record passes 1-3 and with the
  column-511 arm's passes 1-3 at the same seeds.
- It uses the same machinery as before: per-question sums with questions as the unit, an exact sign-flip test, and a
  20,000-replicate question bootstrap.
- It is labelled **EXPLORATORY** for three reasons:
  - the controls are historical;
  - seeds are reused across engines (CONFOUNDS #18), so the same seed on a different engine is a different draw;
  - the seed-1236 B0 pass was chosen after B1's seed-1236 pass had been seen.
- B1 stays a separate checkpoint record.

**B-2, continuity:** the B1 package's two-implementation view (B0 1234 + B0 1235 + B1 1236), recomputed and checked
against the B1 package's final output. It is unchanged and exploratory.

## What is not claimed

- No significance, equivalence or non-inferiority claim. B0 versus B1 is one pair of single passes.
- **No D7 attribution** beyond the predefined reading:
  - (a) says the excess does not need D7;
  - (b) leaves D7 and run-to-run variation unseparated;
  - (c) says nothing.
- Reading (a) does not show that D7 has no behavioural effect anywhere. Reading (b) does not show that D7 has one.
- No explanation of the 0.7.0 advantage; 0.7.0's seed-1236 pass is context only.
- **Historical controls:** the stock comparators were the second pass of their server sessions. B0 and B1 were each the
  first pass of a fresh server. Exchangeability is assumed, not designed.
- **Reporting:**
  - checkpoint records stay separate: B0 at seeds 1234, 1235 and 1236; B1 at seed 1236;
  - pooled numbers are never labelled a three-pass B1 result;
  - the B0 three-seed view is exploratory.

## Invalid or truncated runs

A run is not a completed one-pass result if any of the following happens:
- any validity check fails, including the image being something other than B0 or settings that differ from B0's
  seed-1235 session;
- the watchdog fires (7.5 occupied GPU-hours);
- the pass is incomplete.

Such a run gets no category and no reading, and is reported as invalid with its cost. There is no automatic retry and no
further pass.
