# GPQA Diamond: three passes per configuration

Three configurations, three passes each of 198 questions, with request seeds 1234, 1235 and 1236 (pass *p* sends
1233 + *p*; every configuration sends the same seed in the same pass, so passes pair across configurations):

- `3.25bpw · tpurtell 0.7.0 · DFlash2 ×5` (the release as shipped; 8 concurrent requests)
- `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3` (8 concurrent requests)
- `4bpw TR3 (Brandon) · tpurtell 0.9.1 · DFlash2 ×3 · concurrency 4` (4 requests at once, because its KV pool of
  1,377,179 tokens holds 4.17 requests at the token cap; no request waited, peak KV usage 91.7%)

`tools/analyze.py gpqa-passes`, `gpqa-records`, `gpqa-empty`. Raw = lm-eval `flexible-extract` (the headline); stated =
the audited `correct_stated` score. Intervals: 95% bootstrap over questions, each question resampled with all its passes.

| Configuration | Raw, passes 1 / 2 / 3 | Raw, mean (95% interval) | Spread | Stated, passes 1 / 2 / 3 | Empty, passes 1 / 2 / 3 | Questions ever empty |
|---|---|---|---|---|---|---|
| `3.25bpw · tpurtell 0.7.0 · DFlash2 ×5` | 87.9 / 88.4 / 88.4% | 88.2% (84.5-91.6) | 0.5 | 89.4 / 89.4 / 89.4% | 0 / 3 / 1 | 4 |
| `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3` | 85.4 / 88.9 / 86.4% | 86.9% (83.0-90.4) | 3.5 | 85.9 / 89.9 / 87.9% | 5 / 5 / 6 | 11 |
| `4bpw TR3 (Brandon) · tpurtell 0.9.1 · DFlash2 ×3 · concurrency 4` | 84.8 / 86.4 / 85.9% | 85.7% (81.5-89.6) | 1.5 | 86.9 / 86.9 / 89.4% | 8 / 2 / 7 | 14 |

**Comparisons.** Paired by question and pass. A question answered in three passes is one unit: "questions A / B" counts
the questions on which A had more such outcomes than B over the three passes, and fewer; the clustered p is an exact
sign-flip test over questions; the interval resamples questions. The pooled McNemar p counts every question-pass
separately and is shown for reference only. No correction for multiple comparisons.

| A vs B | What differs | Outcome | A vs B (of 594) | B - A, points (95% interval) | Questions A / B | Clustered p | Pooled p |
|---|---|---|---|---|---|---|---|
| 0.7.0 vs 0.9.1, 3.25bpw | release as shipped: draft depth (5 vs 3), layout (DCP2 + EP2 vs DCP1 with MLA layer ownership), kernels and engine code, vision, KV pool | raw right | 524 vs 516 | -1.3 (-4.2 to +1.5) | 23 / 19 | 0.41 | 0.39 |
| | | empty | 4 vs 16 | +2.0 (+0.7 to +3.5) | 1 / 10 | 0.009 | 0.002 |
| 3.25bpw vs 4bpw, 0.9.1 | weights, and 8 vs 4 concurrent requests | raw right | 516 vs 509 | -1.2 (-4.2 to +1.7) | 19 / 16 | 0.51 | 0.47 |
| | | empty | 16 vs 17 | +0.2 (-1.3 to +1.7) | 9 / 9 | 1.00 | 1.00 |
| 0.7.0 3.25bpw vs 0.9.1 4bpw | all of the above | raw right | 524 vs 509 | -2.5 (-5.4 to +0.5) | 25 / 14 | 0.12 | 0.07 |
| | | empty | 4 vs 17 | +2.2 (+0.8 to +3.7) | 2 / 13 | 0.006 | 0.004 |

What this shows:
- **Pass-to-pass noise.** One configuration's passes differ by 0.5 to 3.5 points (descriptive). A single pass cannot
  separate configurations whose accuracy differs by less than that.
- **tpurtell 0.7.0 as shipped left fewer questions unanswered than tpurtell 0.9.1** (supported by the question-clustered
  test, p = 0.009). The question was raised by pass 1 (0 vs 5 empty answers); passes 2 and 3 alone, run after it was
  raised, give 4 vs 11 empty answers (1 vs 7 questions; clustered p = 0.06). Accuracy does not differ measurably. The two differ in
  several ways at once, so which difference matters is open.
- **3.25bpw and 4bpw TR3 (Brandon) on 0.9.1 show no measurable difference** in accuracy or in empty answers
  (descriptive). The 4bpw record ran at 4 concurrent requests, the 3.25bpw record at 8.
- **Which questions.** Questions 79 and 81 came back empty in all three records; 88, 127 and 147 in both 0.9.1 records and
  in no pass of 0.7.0. Where a request log exists, every empty answer but one ran to the 327,680-token cap.

Provenance per pass is in each run's notes: pass 1 of the tpurtell 0.7.0 record ran on 2026-09-29 without a request log
(its seed, 1234, is lm-eval's default); every other pass has a passive request log recording each request's seed. The
stated-answer judgement of passes the 2026-10-09 audit did not cover was made on 2026-10-10 by the same script and hand
check (protocol).
