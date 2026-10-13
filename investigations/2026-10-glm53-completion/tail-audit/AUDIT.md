# Tail-accounting audit (PLAN.md amendment 20, section 2)

2026-10-13. Zero GPU, CPU only. This audits the exploratory tail analysis of amendment 19f
(`../tail-analysis/tail_analysis.py`, `.out`). It holds counts, question ids, flags and hashes only, with no GPQA
question, answer or reply text. `audit.py` reproduces every number (deterministic; 124 of 124 input and consistency
checks pass), `audit.json` holds every number and one record per request, and `audit.out` holds the generated tables
that are reproduced below.

**Status.** This audit is exploratory. The 32k/64k/100k/200k thresholds and the bands were chosen by 19f *after* the
results were seen, and this audit keeps them only so that 19f can be checked. **No change point is inferred.** The
intervals describe the data and are not tests. Seeds are reused across engines, so passes at the same seed are not
independent draws of the same distribution. Fixtures and NLL are not involved.

## What changes against 19f

1. **Reproduction.** Re-running 19f's script reproduces `tail_analysis.out` byte for byte. Every count 19f reports was
   also re-derived independently: request logs were joined to the lm-eval samples, all 2,178 content hashes match, and
   the published `results.jsonl` flags are reproduced for stock 0.7.0 and stock 0.9.1. The recorded headline counts are
   reproduced for the column-511 arm, B0 and B1. Every 19f threshold count, pooled count and band count matches. 19f's
   intervals differ slightly from this audit's (for example 0.7.0 at 100k: 19f [0.46, 0.93], here [0.44, 0.94]). The
   reason is that 19f resamples only the questions that reached L, while this audit resamples all 198 questions and
   carries all of each question's requests. No 19f replicate was undefined.
2. **Counting error (corrected).** The 19f claim "0.9.1-family replies almost never finish beyond about 150k tokens (6
   of about 100 still running)" adds up the at-risk counts of three bands (36 + 33 + 31 = 100), which counts the same
   requests up to three times. In unique terms, **36 requests over 18 questions were at risk at 150k, and 6 of them
   finished (all valid final answers): 6 of 36 (0.17), not 6 of about 100.** Stock 0.9.1 alone: 3 of 18 requests (12
   questions). Stock 0.7.0: 5 of 9 requests (8 questions). The direction of the contrast holds, but the 0.9.1 rate is
   about three times what 19f implied.
3. **The "finished" definition.** In 19f, "finished" means "not cap-exhausted". The audit splits it into a valid
   final-answer completion and other stopping (definitions below). Only four logged requests stopped in some other way:
   - stock 0.9.1 seed 1234, doc 12: empty reply at 38 tokens (the known premature stop);
   - stock 0.7.0 seed 1235, docs 25 (10,602 tokens) and 94 (45,192 tokens): ruled hedged by hand;
   - B0 seed 1235, doc 94 (40,798 tokens): ruled hedged by hand.

   So 19f's "finished" at 32k contains one non-valid stop for 0.7.0 (53 finished, 52 valid) and one for B0+B1 (55
   finished, 54 valid). **No 19f number at 64k or above changes**, because every finishing event at 64k or above, in
   every configuration, is a valid final-answer completion.
4. **Capped replies.** 19f counts cap-exhausted replies in "reached L" and never as finished. That is the correct risk
   set under administrative censoring at the cap, and nothing changes.
5. **Pooling.** 19f's headline compares 0.7.0 against the pooled "0.9.1 family" (stock plus column-511 arm) and against
   B0+B1 pooled. Stock-only figures are reported first below.
   - At 100k, stock 0.9.1 alone gives 0.40 (10 of 25), the same as the pool.
   - At 200k, stock 0.9.1 alone gives 0.12 (2 of 17), against the pool's 0.09 (3 of 33).
   - The 19f "B0+B1 0.12 at 100k" is B0 2 of 8 plus B1 0 of 9; B1 alone accounts for most of the censoring in that pool.
6. **Question as the unit.** 19f's "8 common questions" counts the arm on the 0.9.1 side. With stock only there are 7
   common questions at 100k, because doc 186 reached 100k on the 0.9.1 side only in the arm.
   - 0.7.0 finished a larger share on 3 questions (127, 147, 120), a smaller share on 1 (79), and the same share on 3.
   - With seeds and session positions matched (stock 0.9.1 seeds 1235-1236), 0.7.0 is higher on 2 (127, 147), lower on
     none, and equal on 5.
7. **Session position, which 19f ignored.**
   - Second pass of a server session: 0.7.0 s1236, stock 0.9.1 s1236 and arm s1236.
   - First pass of a session: every other logged pass, including B0 s1234, B0 s1235 and B1 s1236.
   - The descriptive split shows no consistent position pattern. B1, a first-of-session pass, has the most censored
     replies (9).
   - The 19a seed-1236 references (stock, arm, 0.7.0) were all second-of-session passes, while B1 was first-of-session.
8. **Localisation.** 19f's reading pointed to "decode behaviour at contexts of about 64-100k tokens and beyond". The
   stock-only bands below do not localise anything. Per band (still generating at band start → ending inside the band),
   stock 0.7.0 against stock 0.9.1:

   | band | stock 0.7.0 | stock 0.9.1 |
   |---|---:|---:|
   | 64-100k | 15/29 | 16/41 |
   | 100-150k | 5/14 | 7/25 |
   | 150-200k | 1/9 | 1/18 |
   | 200-260k | 4/8 | 1/17 |
   | 260k-cap | 0/4 | 1/16 |

   The cumulative difference grows with length. However, each band holds at most 16 events from few questions, and the
   200-260k contrast rests on 4 events from 3 questions (120, 127 twice, 147). The thresholds are post hoc and no change
   point is claimed.
9. **Reach.** 19f's statement that long replies are similarly frequent holds for stock alone. The share still
   generating at 100k is 0.035 (0.7.0) against 0.042 (stock 0.9.1), a difference of −0.007 [−0.026, 0.015].

## Definitions (fixed in `audit.py` before the tables were read)

| term | definition | measured / derived |
|---|---|---|
| T | `completion_tokens` from the recording proxy: reasoning, final answer and stop token together | measured |
| censored | `finish_reason` `length` with T = 327,680 (= `max_tokens`); administrative censoring at the cap | measured |
| finishing event | generation ended before the cap (`finish_reason` `stop`; no other finish reason occurs) | measured |
| valid final-answer completion | finishing event **and** non-empty content (`results.jsonl` `empty` = False, i.e. reply.strip() ≠ '') **and** the frozen stated-answer extractor (`export_evals.py` `final_answer`, sha256 731f27d4… of lines 233-259) returns a letter A-D **and** the reply was not ruled hedged by hand. Correctness plays no part in validity. | measured flags |
| other stopping | finishing event that is not valid: `empty_stop` (empty content, finish `stop`), `no_letter` (content without an extractable letter; none occurred) or `hedged` (ruled hedged by hand) | measured flags |
| at risk at L | T ≥ L (still generating when token L was due; same as 19f) | measured |
| band [lo, hi) | at risk T ≥ lo; events lo ≤ T < hi. The last band, [260,000, 327,680], holds every censored request | measured |
| reasoning tokens | T minus the content re-tokenized with the K3.25 tokenizer (`d7/exposure/render_tokens.py` BPE); includes the reasoning-close marker and the stop token. Re-tokenization can differ from the generated ids. | **derived** |
| session position | order of the passes inside one server log containing exactly one engine start, by first request time | measured (logs) |

Hand rulings on hedged replies come from these sources (ids and verdicts only):
- 0.7.0: `export_evals.py` HEDGED (recipe `+ps` pass 2, docs 25 and 94);
- stock 0.9.1: none;
- arm: `arm-col511/adjudications.json` (none hedged);
- B0 s1234: `bundle/screen/run-d1234/adjudication.json` (none hedged);
- B0 s1235: `bundle/screen2/run-s1235/adjudication.json` (doc 94 hedged);
- B1: `b1-screen/run-s1236/adjudication.json` (none hedged).

**Survival framing.** Every censoring time equals the cap. Below the cap, the Kaplan-Meier estimate is therefore
exactly the plain fraction of requests still generating, and `audit.py` checks this. Censoring affects only what can be
said at and beyond the cap: whether, or when, a censored reply would have stopped is unknown. Intervals come from 2,000
question-clustered bootstrap replicates over all 198 questions.

## Logged passes, seeds and session positions

The B0 seed-1236 pass (`bundle-d1234.s3`) was still running and is excluded. Stock 0.7.0 pass 1 (seed 1234, 2026-09-29)
has no request log, so only its published empty flag is available.

| configuration | seed | lm-eval pass | session (server log) | position | empty | censored at cap | empty stop | no-letter stop | hedged stop | valid | at risk / valid / other / censored at 100k | at 200k |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| stock 0.7.0 | 1235 | 2 | glm53-flash-exl3-k3.25@tpurtell+ps.server.log | 1 of 2 | 3 | 3 | 0 | 0 | 2 | 193 | 8 / 5 / 0 / 3 | 5 / 2 / 0 / 3 |
| stock 0.7.0 | 1236 | 3 | glm53-flash-exl3-k3.25@tpurtell+ps.server.log | 2 of 2 | 1 | 1 | 0 | 0 | 0 | 197 | 6 / 5 / 0 / 1 | 3 / 2 / 0 / 1 |
| stock 0.9.1 | 1234 | 1 | glm53-flash-exl3-k3.25@tpurtell+v0.9.1.server.log | 1 of 1 | 5 | 4 | 1 | 0 | 0 | 193 | 7 / 3 / 0 / 4 | 5 / 1 / 0 / 4 |
| stock 0.9.1 | 1235 | 2 | glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.server.log | 1 of 2 | 5 | 5 | 0 | 0 | 0 | 193 | 9 / 4 / 0 / 5 | 6 / 1 / 0 / 5 |
| stock 0.9.1 | 1236 | 3 | glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.server.log | 2 of 2 | 6 | 6 | 0 | 0 | 0 | 192 | 9 / 3 / 0 / 6 | 6 / 0 / 0 / 6 |
| col511 arm | 1234 | 1 | server-s1.log | 1 of 1 | 4 | 4 | 0 | 0 | 0 | 194 | 8 / 4 / 0 / 4 | 5 / 1 / 0 / 4 |
| col511 arm | 1235 | 2 | server-s2.log | 1 of 2 | 5 | 5 | 0 | 0 | 0 | 193 | 9 / 4 / 0 / 5 | 5 / 0 / 0 / 5 |
| col511 arm | 1236 | 3 | server-s2.log | 2 of 2 | 6 | 6 | 0 | 0 | 0 | 192 | 8 / 2 / 0 / 6 | 6 / 0 / 0 / 6 |
| B0 | 1234 | 1 | server-s1.log | 1 of 1 | 3 | 3 | 0 | 0 | 0 | 195 | 4 / 1 / 0 / 3 | 3 / 0 / 0 / 3 |
| B0 | 1235 | 2 | server-s2.log | 1 of 1 | 3 | 3 | 0 | 0 | 1 | 194 | 4 / 1 / 0 / 3 | 3 / 0 / 0 / 3 |
| B1 | 1236 | 3 | server-s1.log | 1 of 1 | 9 | 9 | 0 | 0 | 0 | 189 | 9 / 0 / 0 / 9 | 9 / 0 / 0 / 9 |
| stock 0.7.0 | 1234 | 1 | unlogged | - | 0 | ? | ? | - | - | - | unlogged | unlogged |


## 1. Stock releases (reported first and separately)

### stock 0.7.0  [STOCK; 2 passes, 396 requests]

Thresholds (T = logged completion tokens; at risk = T >= L):

| L | unique requests at risk | questions at risk | finishing events | valid final answer | other stop | censored at cap | P(event before cap \| at risk) [95% q-boot] | P(valid \| at risk) [95% q-boot] |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 32,000 | 57 | 35 | 53 | 52 | 1 | 4 | 0.93 [0.86, 0.98] | 0.91 [0.84, 0.98] |
| 64,000 | 29 | 19 | 25 | 25 | 0 | 4 | 0.86 [0.74, 0.97] | 0.86 [0.74, 0.97] |
| 100,000 | 14 | 11 | 10 | 10 | 0 | 4 | 0.71 [0.44, 0.94] | 0.71 [0.44, 0.94] |
| 200,000 | 8 | 7 | 4 | 4 | 0 | 4 | 0.50 [0.00, 0.89] | 0.50 [0.00, 0.89] |

Question identities (doc ids):

- >= 64,000: at risk [7, 8, 48, 71, 76, 79, 81, 88, 118, 120, 121, 127, 130, 138, 145, 147, 164, 170, 186]; with a valid finish [7, 8, 48, 71, 76, 79, 81, 88, 118, 120, 127, 130, 138, 145, 147, 164, 170, 186]
- >= 100,000: at risk [7, 8, 76, 79, 81, 120, 121, 127, 145, 147, 186]; with a valid finish [7, 8, 76, 120, 127, 145, 147, 186]
- >= 200,000: at risk [79, 81, 120, 121, 127, 145, 147]; with a valid finish [120, 127, 147]

Bands (at risk = still generating at band start; events end inside the band):

| band (tokens) | at risk | events | valid | other | censored in band | conditional fraction ending in band | questions with an event |
|---|---:|---:|---:|---:|---:|---:|---|
| 0-16,000 | 396 | 304 | 303 | 1 | 0 | 0.77 | 164 questions |
| 16,000-32,000 | 92 | 35 | 35 | 0 | 0 | 0.38 | 30 questions |
| 32,000-64,000 | 57 | 28 | 27 | 1 | 0 | 0.49 | 22 questions |
| 64,000-100,000 | 29 | 15 | 15 | 0 | 0 | 0.52 | [8, 48, 71, 79, 81, 88, 118, 130, 138, 147, 164, 170] |
| 100,000-150,000 | 14 | 5 | 5 | 0 | 0 | 0.36 | [7, 8, 76, 145, 186] |
| 150,000-200,000 | 9 | 1 | 1 | 0 | 0 | 0.11 | [186] |
| 200,000-260,000 | 8 | 4 | 4 | 0 | 0 | 0.50 | [120, 127, 147] |
| 260,000-327,680 | 4 | 0 | 0 | 0 | 4 | 0.00 | [] |

Still generating, S(t) = P(T > t), Kaplan-Meier with censoring at the cap, 95% question-clustered bootstrap (B = 2000, all 198 questions resampled):

| t | 16k | 32k | 64k | 100k | 150k | 200k | 260k | cap- |
|---|---|---|---|---|---|---|---|---|
| S(t), completion tokens (measured) | 0.232 [0.179, 0.290] | 0.144 [0.101, 0.192] | 0.073 [0.043, 0.109] | 0.035 [0.015, 0.058] | 0.023 [0.008, 0.040] | 0.020 [0.008, 0.038] | 0.010 [0.003, 0.020] | 0.010 [0.003, 0.020] |
| S(t), reasoning tokens (derived) | 0.227 [0.174, 0.285] | 0.139 [0.096, 0.187] | 0.073 [0.043, 0.109] | 0.035 [0.015, 0.058] | 0.023 [0.008, 0.040] | 0.020 [0.008, 0.038] | 0.010 [0.003, 0.020] | 0.010 [0.003, 0.020] |

### stock 0.9.1  [STOCK; 3 passes, 594 requests]

Thresholds (T = logged completion tokens; at risk = T >= L):

| L | unique requests at risk | questions at risk | finishing events | valid final answer | other stop | censored at cap | P(event before cap \| at risk) [95% q-boot] | P(valid \| at risk) [95% q-boot] |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 32,000 | 77 | 38 | 62 | 62 | 0 | 15 | 0.81 [0.71, 0.91] | 0.81 [0.71, 0.91] |
| 64,000 | 41 | 25 | 26 | 26 | 0 | 15 | 0.63 [0.47, 0.80] | 0.63 [0.47, 0.80] |
| 100,000 | 25 | 15 | 10 | 10 | 0 | 15 | 0.40 [0.24, 0.60] | 0.40 [0.24, 0.60] |
| 200,000 | 17 | 11 | 2 | 2 | 0 | 15 | 0.12 [0.00, 0.29] | 0.12 [0.00, 0.29] |

Question identities (doc ids):

- >= 64,000: at risk [8, 18, 27, 48, 55, 69, 71, 79, 81, 88, 90, 94, 109, 110, 118, 120, 121, 127, 130, 138, 145, 147, 164, 170, 186]; with a valid finish [8, 18, 27, 48, 69, 71, 79, 81, 90, 94, 110, 118, 120, 127, 130, 138, 145, 147, 164, 186]
- >= 100,000: at risk [8, 55, 69, 71, 79, 81, 88, 90, 109, 120, 121, 127, 147, 164, 170]; with a valid finish [8, 69, 71, 79, 90, 120, 127, 147, 164]
- >= 200,000: at risk [55, 79, 81, 88, 109, 120, 121, 127, 147, 164, 170]; with a valid finish [79, 164]

Bands (at risk = still generating at band start; events end inside the band):

| band (tokens) | at risk | events | valid | other | censored in band | conditional fraction ending in band | questions with an event |
|---|---:|---:|---:|---:|---:|---:|---|
| 0-16,000 | 594 | 462 | 461 | 1 | 0 | 0.78 | 170 questions |
| 16,000-32,000 | 132 | 55 | 55 | 0 | 0 | 0.42 | 42 questions |
| 32,000-64,000 | 77 | 36 | 36 | 0 | 0 | 0.47 | 23 questions |
| 64,000-100,000 | 41 | 16 | 16 | 0 | 0 | 0.39 | [8, 18, 27, 48, 71, 81, 94, 110, 118, 130, 138, 145, 186] |
| 100,000-150,000 | 25 | 7 | 7 | 0 | 0 | 0.28 | [8, 69, 71, 90, 120, 127, 147] |
| 150,000-200,000 | 18 | 1 | 1 | 0 | 0 | 0.06 | [71] |
| 200,000-260,000 | 17 | 1 | 1 | 0 | 0 | 0.06 | [79] |
| 260,000-327,680 | 16 | 1 | 1 | 0 | 15 | 0.06 | [164] |

Still generating, S(t) = P(T > t), Kaplan-Meier with censoring at the cap, 95% question-clustered bootstrap (B = 2000, all 198 questions resampled):

| t | 16k | 32k | 64k | 100k | 150k | 200k | 260k | cap- |
|---|---|---|---|---|---|---|---|---|
| S(t), completion tokens (measured) | 0.222 [0.172, 0.280] | 0.130 [0.089, 0.173] | 0.069 [0.042, 0.099] | 0.042 [0.022, 0.067] | 0.030 [0.013, 0.049] | 0.029 [0.012, 0.047] | 0.027 [0.012, 0.044] | 0.025 [0.010, 0.042] |
| S(t), reasoning tokens (derived) | 0.217 [0.167, 0.273] | 0.130 [0.089, 0.173] | 0.067 [0.040, 0.099] | 0.042 [0.022, 0.067] | 0.030 [0.013, 0.049] | 0.029 [0.012, 0.047] | 0.027 [0.012, 0.044] | 0.025 [0.010, 0.042] |

### stock 0.9.1 seeds 1235-1236 (seed- and session-matched to 0.7.0)  [STOCK; 2 passes, 396 requests]

Thresholds (T = logged completion tokens; at risk = T >= L):

| L | unique requests at risk | questions at risk | finishing events | valid final answer | other stop | censored at cap | P(event before cap \| at risk) [95% q-boot] | P(valid \| at risk) [95% q-boot] |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 32,000 | 53 | 34 | 42 | 42 | 0 | 11 | 0.79 [0.65, 0.92] | 0.79 [0.65, 0.92] |
| 64,000 | 28 | 21 | 17 | 17 | 0 | 11 | 0.61 [0.39, 0.82] | 0.61 [0.39, 0.82] |
| 100,000 | 18 | 13 | 7 | 7 | 0 | 11 | 0.39 [0.14, 0.67] | 0.39 [0.14, 0.67] |
| 200,000 | 12 | 9 | 1 | 1 | 0 | 11 | 0.08 [0.00, 0.31] | 0.08 [0.00, 0.31] |

Question identities (doc ids):

- >= 64,000: at risk [8, 27, 48, 55, 69, 71, 79, 81, 88, 94, 118, 120, 121, 127, 130, 138, 145, 147, 164, 170, 186]; with a valid finish [8, 27, 48, 69, 71, 94, 118, 120, 127, 130, 138, 145, 164, 186]
- >= 100,000: at risk [8, 55, 69, 71, 79, 81, 88, 120, 121, 127, 147, 164, 170]; with a valid finish [8, 69, 71, 120, 127, 164]
- >= 200,000: at risk [55, 79, 81, 88, 121, 127, 147, 164, 170]; with a valid finish [164]

Bands (at risk = still generating at band start; events end inside the band):

| band (tokens) | at risk | events | valid | other | censored in band | conditional fraction ending in band | questions with an event |
|---|---:|---:|---:|---:|---:|---:|---|
| 0-16,000 | 396 | 305 | 305 | 0 | 0 | 0.77 | 162 questions |
| 16,000-32,000 | 91 | 38 | 38 | 0 | 0 | 0.42 | 32 questions |
| 32,000-64,000 | 53 | 25 | 25 | 0 | 0 | 0.47 | 20 questions |
| 64,000-100,000 | 28 | 10 | 10 | 0 | 0 | 0.36 | [27, 48, 94, 118, 130, 138, 145, 186] |
| 100,000-150,000 | 18 | 5 | 5 | 0 | 0 | 0.28 | [8, 69, 71, 120, 127] |
| 150,000-200,000 | 13 | 1 | 1 | 0 | 0 | 0.08 | [71] |
| 200,000-260,000 | 12 | 0 | 0 | 0 | 0 | 0.00 | [] |
| 260,000-327,680 | 12 | 1 | 1 | 0 | 11 | 0.08 | [164] |

Still generating, S(t) = P(T > t), Kaplan-Meier with censoring at the cap, 95% question-clustered bootstrap (B = 2000, all 198 questions resampled):

| t | 16k | 32k | 64k | 100k | 150k | 200k | 260k | cap- |
|---|---|---|---|---|---|---|---|---|
| S(t), completion tokens (measured) | 0.230 [0.174, 0.288] | 0.134 [0.091, 0.179] | 0.071 [0.040, 0.103] | 0.045 [0.023, 0.073] | 0.033 [0.013, 0.056] | 0.030 [0.013, 0.053] | 0.030 [0.013, 0.053] | 0.028 [0.010, 0.051] |
| S(t), reasoning tokens (derived) | 0.225 [0.169, 0.283] | 0.134 [0.091, 0.179] | 0.068 [0.040, 0.101] | 0.045 [0.023, 0.073] | 0.033 [0.013, 0.056] | 0.030 [0.013, 0.053] | 0.030 [0.013, 0.053] | 0.028 [0.010, 0.051] |

### Stock 0.7.0 minus stock 0.9.1, joint question-clustered bootstrap (same resampled questions on both sides)

| comparison | P(event \| risk 32k) | P(event \| risk 64k) | P(event \| risk 100k) | P(event \| risk 200k) | S(100k) | S(200k) | S(cap-) |
|---|---|---|---|---|---|---|---|
| stock 0.7.0 minus stock 0.9.1 (all logged seeds) | +0.12 [0.03, 0.23] | +0.23 [0.05, 0.40] | +0.31 [0.03, 0.55] | +0.38 [-0.15, 0.81] | -0.007 [-0.026, 0.015] | -0.008 [-0.021, 0.003] | -0.015 [-0.030, -0.003] |
| stock 0.7.0 minus stock 0.9.1 (seeds 1235-1236) | +0.14 [0.03, 0.26] | +0.25 [0.05, 0.45] | +0.33 [0.03, 0.57] | +0.42 [-0.10, 0.80] | -0.010 [-0.033, 0.013] | -0.010 [-0.025, 0.005] | -0.018 [-0.035, -0.003] |


**Plain statements, stock only (measured counts; intervals are descriptive):**
- **Reaching long.** Similar on both releases: 3.5% of 0.7.0 requests and 4.2% of stock 0.9.1 requests were still
  generating at 100k.
- **Finishing once long.** Among requests still generating at 100k, 0.7.0 finished 10 of 14 (11 questions) and stock
  0.9.1 finished 10 of 25 (15 questions), a difference of +0.31 [0.03, 0.55]. At 200k the figures are 4 of 8 (7
  questions) against 2 of 17 (11 questions), +0.38 [−0.15, 0.81]. Every one of these finishes is a valid final answer.
- **Seed and session matched** (seeds 1235-1236, first and second pass of one session on both releases): at 100k, 10 of
  14 against 7 of 18, +0.33 [0.03, 0.57]. At 200k, 4 of 8 against 1 of 12.
- **Censoring.** 0.7.0 has 4 cap exhaustions in 396 logged requests (1.0%), against 15 in 594 for stock 0.9.1 (2.5%):
  −0.015 [−0.030, −0.003].
- **Concentration.** The stock difference sits in few questions. Seven questions reached 100k on both releases.
  - 0.7.0 finished a larger share on docs 127 and 147, both with all seeds and with matched seeds.
  - It also finished a larger share on doc 120, but only when stock 0.9.1 seed 1234 is included.
  - It finished a smaller share on doc 79 when seed 1234 is included: stock 0.9.1 s1234 finished doc 79 at 203,164
    tokens.

## 2. Modified configurations (each separately)

### col511 arm  [MODIFIED; 3 passes, 594 requests]

Thresholds (T = logged completion tokens; at risk = T >= L):

| L | unique requests at risk | questions at risk | finishing events | valid final answer | other stop | censored at cap | P(event before cap \| at risk) [95% q-boot] | P(valid \| at risk) [95% q-boot] |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 32,000 | 73 | 38 | 58 | 58 | 0 | 15 | 0.79 [0.67, 0.90] | 0.79 [0.67, 0.90] |
| 64,000 | 46 | 27 | 31 | 31 | 0 | 15 | 0.67 [0.50, 0.83] | 0.67 [0.50, 0.83] |
| 100,000 | 25 | 18 | 10 | 10 | 0 | 15 | 0.40 [0.20, 0.62] | 0.40 [0.20, 0.62] |
| 200,000 | 16 | 12 | 1 | 1 | 0 | 15 | 0.06 [0.00, 0.20] | 0.06 [0.00, 0.20] |

Question identities (doc ids):

- >= 64,000: at risk [7, 8, 32, 48, 55, 69, 71, 74, 76, 79, 81, 88, 90, 94, 108, 110, 118, 120, 127, 130, 145, 147, 157, 164, 170, 185, 186]; with a valid finish [7, 8, 32, 48, 55, 71, 76, 79, 81, 88, 90, 94, 110, 118, 130, 145, 147, 164, 185, 186]
- >= 100,000: at risk [48, 69, 71, 74, 79, 81, 88, 90, 94, 108, 110, 120, 127, 130, 147, 157, 170, 186]; with a valid finish [48, 71, 79, 90, 94, 110, 130, 147, 186]
- >= 200,000: at risk [69, 74, 79, 81, 88, 108, 120, 127, 147, 157, 170, 186]; with a valid finish [186]

Bands (at risk = still generating at band start; events end inside the band):

| band (tokens) | at risk | events | valid | other | censored in band | conditional fraction ending in band | questions with an event |
|---|---:|---:|---:|---:|---:|---:|---|
| 0-16,000 | 594 | 458 | 458 | 0 | 0 | 0.77 | 169 questions |
| 16,000-32,000 | 136 | 63 | 63 | 0 | 0 | 0.46 | 43 questions |
| 32,000-64,000 | 73 | 27 | 27 | 0 | 0 | 0.37 | 20 questions |
| 64,000-100,000 | 46 | 21 | 21 | 0 | 0 | 0.46 | [7, 8, 32, 48, 55, 71, 76, 81, 88, 118, 130, 145, 147, 164, 185] |
| 100,000-150,000 | 25 | 7 | 7 | 0 | 0 | 0.28 | [48, 71, 90, 94, 130, 147] |
| 150,000-200,000 | 18 | 2 | 2 | 0 | 0 | 0.11 | [79, 110] |
| 200,000-260,000 | 16 | 1 | 1 | 0 | 0 | 0.06 | [186] |
| 260,000-327,680 | 15 | 0 | 0 | 0 | 15 | 0.00 | [] |

Still generating, S(t) = P(T > t), Kaplan-Meier with censoring at the cap, 95% question-clustered bootstrap (B = 2000, all 198 questions resampled):

| t | 16k | 32k | 64k | 100k | 150k | 200k | 260k | cap- |
|---|---|---|---|---|---|---|---|---|
| S(t), completion tokens (measured) | 0.229 [0.178, 0.283] | 0.123 [0.086, 0.165] | 0.077 [0.047, 0.109] | 0.042 [0.022, 0.066] | 0.030 [0.013, 0.051] | 0.027 [0.012, 0.045] | 0.025 [0.012, 0.042] | 0.025 [0.012, 0.042] |
| S(t), reasoning tokens (derived) | 0.226 [0.177, 0.280] | 0.120 [0.083, 0.162] | 0.076 [0.047, 0.108] | 0.042 [0.022, 0.066] | 0.030 [0.013, 0.051] | 0.027 [0.012, 0.045] | 0.025 [0.012, 0.042] | 0.025 [0.012, 0.042] |

### B0  [MODIFIED; 2 passes, 396 requests]

Thresholds (T = logged completion tokens; at risk = T >= L):

| L | unique requests at risk | questions at risk | finishing events | valid final answer | other stop | censored at cap | P(event before cap \| at risk) [95% q-boot] | P(valid \| at risk) [95% q-boot] |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 32,000 | 44 | 30 | 38 | 37 | 1 | 6 | 0.86 [0.74, 0.97] | 0.84 [0.72, 0.95] |
| 64,000 | 17 | 14 | 11 | 11 | 0 | 6 | 0.65 [0.42, 0.89] | 0.65 [0.42, 0.89] |
| 100,000 | 8 | 6 | 2 | 2 | 0 | 6 | 0.25 [0.00, 0.60] | 0.25 [0.00, 0.60] |
| 200,000 | 6 | 5 | 0 | 0 | 0 | 6 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |

Question identities (doc ids):

- >= 64,000: at risk [8, 48, 76, 79, 81, 88, 90, 120, 121, 127, 130, 145, 147, 185]; with a valid finish [8, 48, 76, 81, 88, 90, 120, 130, 145, 147, 185]
- >= 100,000: at risk [79, 88, 90, 121, 127, 147]; with a valid finish [88, 90]
- >= 200,000: at risk [79, 88, 121, 127, 147]; with a valid finish []

Bands (at risk = still generating at band start; events end inside the band):

| band (tokens) | at risk | events | valid | other | censored in band | conditional fraction ending in band | questions with an event |
|---|---:|---:|---:|---:|---:|---:|---|
| 0-16,000 | 396 | 309 | 309 | 0 | 0 | 0.78 | 168 questions |
| 16,000-32,000 | 87 | 43 | 43 | 0 | 0 | 0.49 | 37 questions |
| 32,000-64,000 | 44 | 27 | 26 | 1 | 0 | 0.61 | 25 questions |
| 64,000-100,000 | 17 | 9 | 9 | 0 | 0 | 0.53 | [8, 48, 76, 81, 120, 130, 145, 147, 185] |
| 100,000-150,000 | 8 | 2 | 2 | 0 | 0 | 0.25 | [88, 90] |
| 150,000-200,000 | 6 | 0 | 0 | 0 | 0 | 0.00 | [] |
| 200,000-260,000 | 6 | 0 | 0 | 0 | 0 | 0.00 | [] |
| 260,000-327,680 | 6 | 0 | 0 | 0 | 6 | 0.00 | [] |

Still generating, S(t) = P(T > t), Kaplan-Meier with censoring at the cap, 95% question-clustered bootstrap (B = 2000, all 198 questions resampled):

| t | 16k | 32k | 64k | 100k | 150k | 200k | 260k | cap- |
|---|---|---|---|---|---|---|---|---|
| S(t), completion tokens (measured) | 0.220 [0.169, 0.273] | 0.111 [0.073, 0.154] | 0.043 [0.023, 0.068] | 0.020 [0.005, 0.038] | 0.015 [0.003, 0.030] | 0.015 [0.003, 0.030] | 0.015 [0.003, 0.030] | 0.015 [0.003, 0.030] |
| S(t), reasoning tokens (derived) | 0.217 [0.167, 0.270] | 0.111 [0.073, 0.154] | 0.043 [0.023, 0.068] | 0.020 [0.005, 0.038] | 0.015 [0.003, 0.030] | 0.015 [0.003, 0.030] | 0.015 [0.003, 0.030] | 0.015 [0.003, 0.030] |

### B1  [MODIFIED; 1 passes, 198 requests]

Thresholds (T = logged completion tokens; at risk = T >= L):

| L | unique requests at risk | questions at risk | finishing events | valid final answer | other stop | censored at cap | P(event before cap \| at risk) [95% q-boot] | P(valid \| at risk) [95% q-boot] |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 32,000 | 26 | 26 | 17 | 17 | 0 | 9 | 0.65 [0.46, 0.84] | 0.65 [0.46, 0.84] |
| 64,000 | 12 | 12 | 3 | 3 | 0 | 9 | 0.25 [0.00, 0.53] | 0.25 [0.00, 0.53] |
| 100,000 | 9 | 9 | 0 | 0 | 0 | 9 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| 200,000 | 9 | 9 | 0 | 0 | 0 | 9 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |

Question identities (doc ids):

- >= 64,000: at risk [8, 69, 71, 79, 81, 88, 118, 127, 130, 145, 147, 179]; with a valid finish [71, 118, 179]
- >= 100,000: at risk [8, 69, 79, 81, 88, 127, 130, 145, 147]; with a valid finish []
- >= 200,000: at risk [8, 69, 79, 81, 88, 127, 130, 145, 147]; with a valid finish []

Bands (at risk = still generating at band start; events end inside the band):

| band (tokens) | at risk | events | valid | other | censored in band | conditional fraction ending in band | questions with an event |
|---|---:|---:|---:|---:|---:|---:|---|
| 0-16,000 | 198 | 153 | 153 | 0 | 0 | 0.77 | 153 questions |
| 16,000-32,000 | 45 | 19 | 19 | 0 | 0 | 0.42 | 19 questions |
| 32,000-64,000 | 26 | 14 | 14 | 0 | 0 | 0.54 | 14 questions |
| 64,000-100,000 | 12 | 3 | 3 | 0 | 0 | 0.25 | [71, 118, 179] |
| 100,000-150,000 | 9 | 0 | 0 | 0 | 0 | 0.00 | [] |
| 150,000-200,000 | 9 | 0 | 0 | 0 | 0 | 0.00 | [] |
| 200,000-260,000 | 9 | 0 | 0 | 0 | 0 | 0.00 | [] |
| 260,000-327,680 | 9 | 0 | 0 | 0 | 9 | 0.00 | [] |

Still generating, S(t) = P(T > t), Kaplan-Meier with censoring at the cap, 95% question-clustered bootstrap (B = 2000, all 198 questions resampled):

| t | 16k | 32k | 64k | 100k | 150k | 200k | 260k | cap- |
|---|---|---|---|---|---|---|---|---|
| S(t), completion tokens (measured) | 0.227 [0.172, 0.288] | 0.131 [0.086, 0.182] | 0.061 [0.030, 0.096] | 0.045 [0.020, 0.076] | 0.045 [0.020, 0.076] | 0.045 [0.020, 0.076] | 0.045 [0.020, 0.076] | 0.045 [0.020, 0.076] |
| S(t), reasoning tokens (derived) | 0.222 [0.167, 0.283] | 0.131 [0.086, 0.182] | 0.061 [0.030, 0.096] | 0.045 [0.020, 0.076] | 0.045 [0.020, 0.076] | 0.045 [0.020, 0.076] | 0.045 [0.020, 0.076] | 0.045 [0.020, 0.076] |


## 3. Pooled views (labelled; they mix implementations)

### POOLED: stock 0.9.1 + col511 arm (19f '0.9.1 family')  [POOLED; 6 passes, 1188 requests]

Thresholds (T = logged completion tokens; at risk = T >= L):

| L | unique requests at risk | questions at risk | finishing events | valid final answer | other stop | censored at cap | P(event before cap \| at risk) [95% q-boot] | P(valid \| at risk) [95% q-boot] |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 32,000 | 150 | 44 | 120 | 120 | 0 | 30 | 0.80 [0.70, 0.89] | 0.80 [0.70, 0.89] |
| 64,000 | 87 | 32 | 57 | 57 | 0 | 30 | 0.66 [0.51, 0.80] | 0.66 [0.51, 0.80] |
| 100,000 | 50 | 23 | 20 | 20 | 0 | 30 | 0.40 [0.25, 0.58] | 0.40 [0.25, 0.58] |
| 200,000 | 33 | 16 | 3 | 3 | 0 | 30 | 0.09 [0.00, 0.20] | 0.09 [0.00, 0.20] |

Question identities (doc ids):

- >= 64,000: at risk [7, 8, 18, 27, 32, 48, 55, 69, 71, 74, 76, 79, 81, 88, 90, 94, 108, 109, 110, 118, 120, 121, 127, 130, 138, 145, 147, 157, 164, 170, 185, 186]; with a valid finish [7, 8, 18, 27, 32, 48, 55, 69, 71, 76, 79, 81, 88, 90, 94, 110, 118, 120, 127, 130, 138, 145, 147, 164, 185, 186]
- >= 100,000: at risk [8, 48, 55, 69, 71, 74, 79, 81, 88, 90, 94, 108, 109, 110, 120, 121, 127, 130, 147, 157, 164, 170, 186]; with a valid finish [8, 48, 69, 71, 79, 90, 94, 110, 120, 127, 130, 147, 164, 186]
- >= 200,000: at risk [55, 69, 74, 79, 81, 88, 108, 109, 120, 121, 127, 147, 157, 164, 170, 186]; with a valid finish [79, 164, 186]

Bands (at risk = still generating at band start; events end inside the band):

| band (tokens) | at risk | events | valid | other | censored in band | conditional fraction ending in band | questions with an event |
|---|---:|---:|---:|---:|---:|---:|---|
| 0-16,000 | 1188 | 920 | 919 | 1 | 0 | 0.77 | 174 questions |
| 16,000-32,000 | 268 | 118 | 118 | 0 | 0 | 0.44 | 58 questions |
| 32,000-64,000 | 150 | 63 | 63 | 0 | 0 | 0.42 | 30 questions |
| 64,000-100,000 | 87 | 37 | 37 | 0 | 0 | 0.43 | [7, 8, 18, 27, 32, 48, 55, 71, 76, 81, 88, 94, 110, 118, 130, 138, 145, 147, 164, 185, 186] |
| 100,000-150,000 | 50 | 14 | 14 | 0 | 0 | 0.28 | [8, 48, 69, 71, 90, 94, 120, 127, 130, 147] |
| 150,000-200,000 | 36 | 3 | 3 | 0 | 0 | 0.08 | [71, 79, 110] |
| 200,000-260,000 | 33 | 2 | 2 | 0 | 0 | 0.06 | [79, 186] |
| 260,000-327,680 | 31 | 1 | 1 | 0 | 30 | 0.03 | [164] |

Still generating, S(t) = P(T > t), Kaplan-Meier with censoring at the cap, 95% question-clustered bootstrap (B = 2000, all 198 questions resampled):

| t | 16k | 32k | 64k | 100k | 150k | 200k | 260k | cap- |
|---|---|---|---|---|---|---|---|---|
| S(t), completion tokens (measured) | 0.226 [0.177, 0.279] | 0.126 [0.087, 0.167] | 0.073 [0.045, 0.103] | 0.042 [0.024, 0.065] | 0.030 [0.015, 0.049] | 0.028 [0.013, 0.045] | 0.026 [0.013, 0.042] | 0.025 [0.012, 0.041] |
| S(t), reasoning tokens (derived) | 0.221 [0.173, 0.273] | 0.125 [0.087, 0.166] | 0.072 [0.045, 0.102] | 0.042 [0.024, 0.065] | 0.030 [0.015, 0.049] | 0.028 [0.013, 0.045] | 0.026 [0.013, 0.042] | 0.025 [0.012, 0.041] |

### POOLED: B0 + B1 (19f 'corrected')  [POOLED; 3 passes, 594 requests]

Thresholds (T = logged completion tokens; at risk = T >= L):

| L | unique requests at risk | questions at risk | finishing events | valid final answer | other stop | censored at cap | P(event before cap \| at risk) [95% q-boot] | P(valid \| at risk) [95% q-boot] |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 32,000 | 70 | 37 | 55 | 54 | 1 | 15 | 0.79 [0.67, 0.90] | 0.77 [0.66, 0.88] |
| 64,000 | 29 | 18 | 14 | 14 | 0 | 15 | 0.48 [0.32, 0.68] | 0.48 [0.32, 0.68] |
| 100,000 | 17 | 11 | 2 | 2 | 0 | 15 | 0.12 [0.00, 0.29] | 0.12 [0.00, 0.29] |
| 200,000 | 15 | 10 | 0 | 0 | 0 | 15 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |

Question identities (doc ids):

- >= 64,000: at risk [8, 48, 69, 71, 76, 79, 81, 88, 90, 118, 120, 121, 127, 130, 145, 147, 179, 185]; with a valid finish [8, 48, 71, 76, 81, 88, 90, 118, 120, 130, 145, 147, 179, 185]
- >= 100,000: at risk [8, 69, 79, 81, 88, 90, 121, 127, 130, 145, 147]; with a valid finish [88, 90]
- >= 200,000: at risk [8, 69, 79, 81, 88, 121, 127, 130, 145, 147]; with a valid finish []

Bands (at risk = still generating at band start; events end inside the band):

| band (tokens) | at risk | events | valid | other | censored in band | conditional fraction ending in band | questions with an event |
|---|---:|---:|---:|---:|---:|---:|---|
| 0-16,000 | 594 | 462 | 462 | 0 | 0 | 0.78 | 172 questions |
| 16,000-32,000 | 132 | 62 | 62 | 0 | 0 | 0.47 | 45 questions |
| 32,000-64,000 | 70 | 41 | 40 | 1 | 0 | 0.59 | 31 questions |
| 64,000-100,000 | 29 | 12 | 12 | 0 | 0 | 0.41 | [8, 48, 71, 76, 81, 118, 120, 130, 145, 147, 179, 185] |
| 100,000-150,000 | 17 | 2 | 2 | 0 | 0 | 0.12 | [88, 90] |
| 150,000-200,000 | 15 | 0 | 0 | 0 | 0 | 0.00 | [] |
| 200,000-260,000 | 15 | 0 | 0 | 0 | 0 | 0.00 | [] |
| 260,000-327,680 | 15 | 0 | 0 | 0 | 15 | 0.00 | [] |

Still generating, S(t) = P(T > t), Kaplan-Meier with censoring at the cap, 95% question-clustered bootstrap (B = 2000, all 198 questions resampled):

| t | 16k | 32k | 64k | 100k | 150k | 200k | 260k | cap- |
|---|---|---|---|---|---|---|---|---|
| S(t), completion tokens (measured) | 0.222 [0.173, 0.276] | 0.118 [0.081, 0.160] | 0.049 [0.027, 0.076] | 0.029 [0.012, 0.049] | 0.025 [0.010, 0.044] | 0.025 [0.010, 0.044] | 0.025 [0.010, 0.044] | 0.025 [0.010, 0.044] |
| S(t), reasoning tokens (derived) | 0.219 [0.170, 0.271] | 0.118 [0.081, 0.160] | 0.049 [0.027, 0.076] | 0.029 [0.012, 0.049] | 0.025 [0.010, 0.044] | 0.025 [0.010, 0.044] | 0.025 [0.010, 0.044] | 0.025 [0.010, 0.044] |

### POOLED: all modified 0.9.1 configurations (arm + B0 + B1)  [POOLED; 6 passes, 1188 requests]

Thresholds (T = logged completion tokens; at risk = T >= L):

| L | unique requests at risk | questions at risk | finishing events | valid final answer | other stop | censored at cap | P(event before cap \| at risk) [95% q-boot] | P(valid \| at risk) [95% q-boot] |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 32,000 | 143 | 45 | 113 | 112 | 1 | 30 | 0.79 [0.69, 0.89] | 0.78 [0.68, 0.88] |
| 64,000 | 75 | 29 | 45 | 45 | 0 | 30 | 0.60 [0.45, 0.75] | 0.60 [0.45, 0.75] |
| 100,000 | 42 | 21 | 12 | 12 | 0 | 30 | 0.29 [0.15, 0.48] | 0.29 [0.15, 0.48] |
| 200,000 | 31 | 16 | 1 | 1 | 0 | 30 | 0.03 [0.00, 0.11] | 0.03 [0.00, 0.11] |

Question identities (doc ids):

- >= 64,000: at risk [7, 8, 32, 48, 55, 69, 71, 74, 76, 79, 81, 88, 90, 94, 108, 110, 118, 120, 121, 127, 130, 145, 147, 157, 164, 170, 179, 185, 186]; with a valid finish [7, 8, 32, 48, 55, 71, 76, 79, 81, 88, 90, 94, 110, 118, 120, 130, 145, 147, 164, 179, 185, 186]
- >= 100,000: at risk [8, 48, 69, 71, 74, 79, 81, 88, 90, 94, 108, 110, 120, 121, 127, 130, 145, 147, 157, 170, 186]; with a valid finish [48, 71, 79, 88, 90, 94, 110, 130, 147, 186]
- >= 200,000: at risk [8, 69, 74, 79, 81, 88, 108, 120, 121, 127, 130, 145, 147, 157, 170, 186]; with a valid finish [186]

Bands (at risk = still generating at band start; events end inside the band):

| band (tokens) | at risk | events | valid | other | censored in band | conditional fraction ending in band | questions with an event |
|---|---:|---:|---:|---:|---:|---:|---|
| 0-16,000 | 1188 | 920 | 920 | 0 | 0 | 0.77 | 177 questions |
| 16,000-32,000 | 268 | 125 | 125 | 0 | 0 | 0.47 | 61 questions |
| 32,000-64,000 | 143 | 68 | 67 | 1 | 0 | 0.48 | 34 questions |
| 64,000-100,000 | 75 | 33 | 33 | 0 | 0 | 0.44 | [7, 8, 32, 48, 55, 71, 76, 81, 88, 118, 120, 130, 145, 147, 164, 179, 185] |
| 100,000-150,000 | 42 | 9 | 9 | 0 | 0 | 0.21 | [48, 71, 88, 90, 94, 130, 147] |
| 150,000-200,000 | 33 | 2 | 2 | 0 | 0 | 0.06 | [79, 110] |
| 200,000-260,000 | 31 | 1 | 1 | 0 | 0 | 0.03 | [186] |
| 260,000-327,680 | 30 | 0 | 0 | 0 | 30 | 0.00 | [] |

Still generating, S(t) = P(T > t), Kaplan-Meier with censoring at the cap, 95% question-clustered bootstrap (B = 2000, all 198 questions resampled):

| t | 16k | 32k | 64k | 100k | 150k | 200k | 260k | cap- |
|---|---|---|---|---|---|---|---|---|
| S(t), completion tokens (measured) | 0.226 [0.178, 0.278] | 0.120 [0.084, 0.162] | 0.063 [0.037, 0.092] | 0.035 [0.018, 0.056] | 0.028 [0.013, 0.045] | 0.026 [0.013, 0.043] | 0.025 [0.012, 0.041] | 0.025 [0.012, 0.041] |
| S(t), reasoning tokens (derived) | 0.222 [0.174, 0.274] | 0.119 [0.083, 0.159] | 0.062 [0.037, 0.091] | 0.035 [0.018, 0.056] | 0.028 [0.013, 0.045] | 0.026 [0.013, 0.043] | 0.025 [0.012, 0.041] | 0.025 [0.012, 0.041] |


## Requests at risk at 100k (all configurations; stock first)

| configuration | seed | session position | doc | T (completion tokens) | outcome | reasoning tokens (derived) | content tokens (re-tokenized) | reasoning chars |
|---|---:|---|---:|---:|---|---:|---:|---:|
| stock 0.7.0 | 1235 | 1 of 2 | 145 | 100,868 | valid | 100,606 | 262 | 332,536 |
| stock 0.7.0 | 1235 | 1 of 2 | 7 | 110,545 | valid | 109,962 | 583 | 399,651 |
| stock 0.7.0 | 1235 | 1 of 2 | 186 | 152,970 | valid | 152,376 | 594 | 406,460 |
| stock 0.7.0 | 1235 | 1 of 2 | 127 | 230,448 | valid | 230,069 | 379 | 533,378 |
| stock 0.7.0 | 1235 | 1 of 2 | 120 | 233,223 | valid | 232,451 | 772 | 582,576 |
| stock 0.7.0 | 1235 | 1 of 2 | 79 | 327,680 | censored_empty | 327,680 | 0 | 655,590 |
| stock 0.7.0 | 1235 | 1 of 2 | 81 | 327,680 | censored_empty | 327,680 | 0 | 859,057 |
| stock 0.7.0 | 1235 | 1 of 2 | 121 | 327,680 | censored_empty | 327,680 | 0 | 1,062,353 |
| stock 0.7.0 | 1236 | 2 of 2 | 8 | 102,001 | valid | 101,764 | 237 | 344,368 |
| stock 0.7.0 | 1236 | 2 of 2 | 186 | 102,493 | valid | 101,936 | 557 | 281,168 |
| stock 0.7.0 | 1236 | 2 of 2 | 76 | 103,553 | valid | 103,204 | 349 | 313,244 |
| stock 0.7.0 | 1236 | 2 of 2 | 127 | 220,231 | valid | 219,721 | 510 | 484,958 |
| stock 0.7.0 | 1236 | 2 of 2 | 147 | 250,709 | valid | 250,223 | 486 | 641,323 |
| stock 0.7.0 | 1236 | 2 of 2 | 145 | 327,680 | censored_empty | 327,680 | 0 | 882,831 |
| stock 0.9.1 | 1234 | 1 of 1 | 90 | 106,387 | valid | 105,936 | 451 | 324,346 |
| stock 0.9.1 | 1234 | 1 of 1 | 147 | 136,344 | valid | 135,984 | 360 | 372,918 |
| stock 0.9.1 | 1234 | 1 of 1 | 79 | 203,164 | valid | 202,796 | 368 | 498,412 |
| stock 0.9.1 | 1234 | 1 of 1 | 88 | 327,680 | censored_empty | 327,680 | 0 | 822,104 |
| stock 0.9.1 | 1234 | 1 of 1 | 109 | 327,680 | censored_empty | 327,680 | 0 | 919,329 |
| stock 0.9.1 | 1234 | 1 of 1 | 127 | 327,680 | censored_empty | 327,680 | 0 | 731,822 |
| stock 0.9.1 | 1234 | 1 of 1 | 120 | 327,680 | censored_empty | 327,680 | 0 | 802,260 |
| stock 0.9.1 | 1235 | 1 of 2 | 71 | 104,835 | valid | 104,549 | 286 | 340,352 |
| stock 0.9.1 | 1235 | 1 of 2 | 127 | 106,687 | valid | 106,488 | 199 | 281,919 |
| stock 0.9.1 | 1235 | 1 of 2 | 120 | 123,709 | valid | 123,045 | 664 | 330,234 |
| stock 0.9.1 | 1235 | 1 of 2 | 164 | 274,150 | valid | 273,659 | 491 | 801,040 |
| stock 0.9.1 | 1235 | 1 of 2 | 55 | 327,680 | censored_empty | 327,680 | 0 | 848,590 |
| stock 0.9.1 | 1235 | 1 of 2 | 79 | 327,680 | censored_empty | 327,680 | 0 | 666,262 |
| stock 0.9.1 | 1235 | 1 of 2 | 81 | 327,680 | censored_empty | 327,680 | 0 | 850,888 |
| stock 0.9.1 | 1235 | 1 of 2 | 121 | 327,680 | censored_empty | 327,680 | 0 | 870,947 |
| stock 0.9.1 | 1235 | 1 of 2 | 147 | 327,680 | censored_empty | 327,680 | 0 | 845,242 |
| stock 0.9.1 | 1236 | 2 of 2 | 8 | 102,222 | valid | 102,015 | 207 | 358,783 |
| stock 0.9.1 | 1236 | 2 of 2 | 69 | 109,684 | valid | 109,290 | 394 | 297,895 |
| stock 0.9.1 | 1236 | 2 of 2 | 71 | 175,275 | valid | 175,023 | 252 | 573,944 |
| stock 0.9.1 | 1236 | 2 of 2 | 79 | 327,680 | censored_empty | 327,680 | 0 | 646,808 |
| stock 0.9.1 | 1236 | 2 of 2 | 81 | 327,680 | censored_empty | 327,680 | 0 | 843,526 |
| stock 0.9.1 | 1236 | 2 of 2 | 88 | 327,680 | censored_empty | 327,680 | 0 | 1,191,199 |
| stock 0.9.1 | 1236 | 2 of 2 | 127 | 327,680 | censored_empty | 327,680 | 0 | 772,860 |
| stock 0.9.1 | 1236 | 2 of 2 | 147 | 327,680 | censored_empty | 327,680 | 0 | 923,013 |
| stock 0.9.1 | 1236 | 2 of 2 | 170 | 327,680 | censored_empty | 327,680 | 0 | 869,863 |
| B0 | 1234 | 1 of 1 | 90 | 101,893 | valid | 101,406 | 487 | 302,341 |
| B0 | 1234 | 1 of 1 | 79 | 327,680 | censored_empty | 327,680 | 0 | 673,269 |
| B0 | 1234 | 1 of 1 | 88 | 327,680 | censored_empty | 327,680 | 0 | 842,437 |
| B0 | 1234 | 1 of 1 | 127 | 327,680 | censored_empty | 327,680 | 0 | 730,384 |
| B0 | 1235 | 1 of 1 | 88 | 116,281 | valid | 115,976 | 305 | 379,128 |
| B0 | 1235 | 1 of 1 | 79 | 327,680 | censored_empty | 327,680 | 0 | 679,879 |
| B0 | 1235 | 1 of 1 | 121 | 327,680 | censored_empty | 327,680 | 0 | 880,751 |
| B0 | 1235 | 1 of 1 | 147 | 327,680 | censored_empty | 327,680 | 0 | 802,719 |
| B1 | 1236 | 1 of 1 | 8 | 327,680 | censored_empty | 327,680 | 0 | 1,129,585 |
| B1 | 1236 | 1 of 1 | 69 | 327,680 | censored_empty | 327,680 | 0 | 1,109,825 |
| B1 | 1236 | 1 of 1 | 79 | 327,680 | censored_empty | 327,680 | 0 | 622,696 |
| B1 | 1236 | 1 of 1 | 81 | 327,680 | censored_empty | 327,680 | 0 | 872,808 |
| B1 | 1236 | 1 of 1 | 88 | 327,680 | censored_empty | 327,680 | 0 | 1,255,472 |
| B1 | 1236 | 1 of 1 | 127 | 327,680 | censored_empty | 327,680 | 0 | 483,514 |
| B1 | 1236 | 1 of 1 | 130 | 327,680 | censored_empty | 327,680 | 0 | 1,003,004 |
| B1 | 1236 | 1 of 1 | 147 | 327,680 | censored_empty | 327,680 | 0 | 807,574 |
| B1 | 1236 | 1 of 1 | 145 | 327,680 | censored_empty | 327,680 | 0 | 896,146 |
| col511 arm | 1234 | 1 of 1 | 48 | 102,281 | valid | 101,774 | 507 | 395,511 |
| col511 arm | 1234 | 1 of 1 | 90 | 109,128 | valid | 108,714 | 414 | 315,658 |
| col511 arm | 1234 | 1 of 1 | 110 | 158,242 | valid | 157,494 | 748 | 618,384 |
| col511 arm | 1234 | 1 of 1 | 186 | 213,174 | valid | 212,645 | 529 | 575,934 |
| col511 arm | 1234 | 1 of 1 | 79 | 327,680 | censored_empty | 327,680 | 0 | 768,827 |
| col511 arm | 1234 | 1 of 1 | 88 | 327,680 | censored_empty | 327,680 | 0 | 848,013 |
| col511 arm | 1234 | 1 of 1 | 127 | 327,680 | censored_empty | 327,680 | 0 | 513,312 |
| col511 arm | 1234 | 1 of 1 | 170 | 327,680 | censored_empty | 327,680 | 0 | 868,893 |
| col511 arm | 1235 | 1 of 2 | 147 | 112,368 | valid | 111,845 | 523 | 293,619 |
| col511 arm | 1235 | 1 of 2 | 130 | 117,982 | valid | 117,639 | 343 | 397,929 |
| col511 arm | 1235 | 1 of 2 | 71 | 144,948 | valid | 144,754 | 194 | 489,082 |
| col511 arm | 1235 | 1 of 2 | 79 | 194,679 | valid | 194,069 | 610 | 424,208 |
| col511 arm | 1235 | 1 of 2 | 74 | 327,680 | censored_empty | 327,680 | 0 | 664,237 |
| col511 arm | 1235 | 1 of 2 | 108 | 327,680 | censored_empty | 327,680 | 0 | 900,022 |
| col511 arm | 1235 | 1 of 2 | 120 | 327,680 | censored_empty | 327,680 | 0 | 859,062 |
| col511 arm | 1235 | 1 of 2 | 127 | 327,680 | censored_empty | 327,680 | 0 | 881,862 |
| col511 arm | 1235 | 1 of 2 | 186 | 327,680 | censored_empty | 327,680 | 0 | 865,235 |
| col511 arm | 1236 | 2 of 2 | 94 | 108,490 | valid | 107,930 | 560 | 426,903 |
| col511 arm | 1236 | 2 of 2 | 71 | 120,738 | valid | 120,520 | 218 | 390,726 |
| col511 arm | 1236 | 2 of 2 | 69 | 327,680 | censored_empty | 327,680 | 0 | 1,226,044 |
| col511 arm | 1236 | 2 of 2 | 79 | 327,680 | censored_empty | 327,680 | 0 | 770,906 |
| col511 arm | 1236 | 2 of 2 | 81 | 327,680 | censored_empty | 327,680 | 0 | 851,559 |
| col511 arm | 1236 | 2 of 2 | 127 | 327,680 | censored_empty | 327,680 | 0 | 758,373 |
| col511 arm | 1236 | 2 of 2 | 157 | 327,680 | censored_empty | 327,680 | 0 | 1,629,359 |
| col511 arm | 1236 | 2 of 2 | 147 | 327,680 | censored_empty | 327,680 | 0 | 696,790 |


## Question as the unit

- stock 0.7.0 vs stock 0.9.1 (all logged seeds) @ 64000: 17 common questions; first side higher on 7, lower on 1, equal on 9; mean difference of event fractions 0.216; only first side [7, 76]; only second side [18, 27, 55, 69, 90, 94, 109, 110]
- stock 0.7.0 vs stock 0.9.1 (all logged seeds) @ 100000: 7 common questions; first side higher on 3, lower on 1, equal on 3; mean difference of event fractions 0.214; only first side [7, 76, 145, 186]; only second side [55, 69, 71, 88, 90, 109, 164, 170]
    - doc 8: 1/1/1 vs 1/1/1
    - doc 79: 0/0/1 vs 1/1/3
    - doc 81: 0/0/1 vs 0/0/2
    - doc 120: 1/1/1 vs 1/1/2
    - doc 121: 0/0/1 vs 0/0/1
    - doc 127: 2/2/2 vs 1/1/3
    - doc 147: 1/1/1 vs 1/1/3
- stock 0.7.0 vs stock 0.9.1, seeds 1235-1236 only @ 64000: 17 common questions; first side higher on 6, lower on 1, equal on 10; mean difference of event fractions 0.235; only first side [7, 76]; only second side [27, 55, 69, 94]
- stock 0.7.0 vs stock 0.9.1, seeds 1235-1236 only @ 100000: 7 common questions; first side higher on 2, lower on 0, equal on 5; mean difference of event fractions 0.214; only first side [7, 76, 145, 186]; only second side [55, 69, 71, 88, 164, 170]
    - doc 8: 1/1/1 vs 1/1/1
    - doc 79: 0/0/1 vs 0/0/2
    - doc 81: 0/0/1 vs 0/0/2
    - doc 120: 1/1/1 vs 1/1/1
    - doc 121: 0/0/1 vs 0/0/1
    - doc 127: 2/2/2 vs 1/1/2
    - doc 147: 1/1/1 vs 0/0/2
- 19f replica: stock 0.7.0 vs stock 0.9.1 + col511 arm @ 64000: 19 common questions; first side higher on 7, lower on 1, equal on 11; mean difference of event fractions 0.185; only first side []; only second side [18, 27, 32, 55, 69, 74, 90, 94, 108, 109, 110, 157, 185]
- 19f replica: stock 0.7.0 vs stock 0.9.1 + col511 arm @ 100000: 8 common questions; first side higher on 4, lower on 1, equal on 3; mean difference of event fractions 0.283; only first side [7, 76, 145]; only second side [48, 55, 69, 71, 74, 88, 90, 94, 108, 109, 110, 130, 157, 164, 170]


## Session position

- stock 0.7.0 session position 1: seeds [1235], 198 requests, at risk at 100k 8, events 5, censored 3
- stock 0.7.0 session position 2: seeds [1236], 198 requests, at risk at 100k 6, events 5, censored 1
- stock 0.9.1 session position 1: seeds [1234, 1235], 396 requests, at risk at 100k 16, events 7, censored 9
- stock 0.9.1 session position 2: seeds [1236], 198 requests, at risk at 100k 9, events 3, censored 6
- col511 arm session position 1: seeds [1234, 1235], 396 requests, at risk at 100k 17, events 8, censored 9
- col511 arm session position 2: seeds [1236], 198 requests, at risk at 100k 8, events 2, censored 6


## Cap exhaustions: censoring for survival, unanswered replies for the benchmark

| configuration | passes (seeds) | empty replies (unanswered) | of which cap exhaustion | of which early empty stop |
|---|---|---:|---:|---:|
| stock 0.7.0 | 1234 (unlogged) | 0 | n/a (no log) | n/a |
| stock 0.7.0 | 1235, 1236 | 4 | 4 | 0 |
| stock 0.9.1 | 1234, 1235, 1236 | 16 | 15 | 1 (s1234 doc 12, 38 tokens) |
| col511 arm | 1234, 1235, 1236 | 15 | 15 | 0 |
| B0 | 1234, 1235 | 6 | 6 | 0 |
| B1 | 1236 | 9 | 9 | 0 |

In survival terms, the 49 cap exhaustions in the logged passes are administrative censoring at 327,680 tokens: the
reply was still generating when the budget ran out. All 49 have empty content (measured), so none produced any
final-answer text, and as benchmark replies they are unanswered and scored incorrect. These are two separate
statements. The survival curves say nothing about whether a censored reply would have terminated with a larger budget.

## Reasoning fields (measured vs derived)

- **`reasoning_tokens`** is null in all 2,178 logged requests, on every engine (measured). It is not populated, so
  reasoning length in tokens cannot be read from the logs.
- **`reasoning_chars` and `content_chars`** are populated in every request. `content_chars` equals the lm-eval reply
  length in all 2,178 (measured).
- **Derived reasoning length.** The final-answer content of the 2,128 non-empty finished replies re-tokenizes to a
  median of 372 tokens (p90 632, max 1,403). Using derived reasoning tokens (T minus content tokens) as the time axis
  moves 4 requests across the 32k threshold and 2 across 64k, and none at 100k or 200k. The Kaplan-Meier fractions on
  the reasoning axis differ from the completion-token axis by at most 0.0051, which is two requests in 396 (tables above). At these lengths the
  reasoning close and the end of generation are a few hundred tokens apart, so the 19f conclusions do not depend on
  which axis is used. Reasoning close and generation stop remain different events; the token-level semantics are
  amendment 20's separate "termination semantics" item.
- **Censored replies** (measured characters, derived ratios). Every one has empty content. Their reasoning runs from
  483,514 to 1,629,359 characters for the same 327,680 tokens, that is 1.48-4.97 characters per token. Finished replies
  of 100k tokens or more run 2.19-3.96 characters per derived reasoning token. This heterogeneity is recorded for the
  fixture inventory and is not interpreted here.

## Limits

- Few events: at 100k or more there are 10 finishing events per stock release, from 8-9 questions.
- 0.7.0 has only two logged passes, and its pass 1 is unlogged.
- The same three seeds are reused across engines.
- Thresholds and bands are post hoc.
- Session position is confounded with seed: every second-of-session pass is seed 1236.
- The hand rulings on hedged replies are those already recorded; no new reply was read for this audit.
- Derived reasoning tokens rest on re-tokenization, not on the generated ids.

## Files

- `audit.py`: the analysis (reads only; writes `audit.json` and `audit.out` in this directory).
- `audit.json`: all results, the 124 checks, input sha256s, and per-request records (ids, counts, flags, hashes).
- `audit.out`: the generated tables reproduced above.
- `SHA256SUMS`: hashes of the four files above.

Input pins (sha256, abbreviated; full values in `audit.json` `inputs_sha256`):
- `tail_analysis.py` 196dcfb3…
- `tail_analysis.out` c912c857…
- stated-answer extractor copy `b1-screen/analysis_col511.py` b5b13ce2…
- tokenizer implementation `d7/exposure/render_tokens.py` 7c6a9010…
