# GPQA Diamond across GLM-5.3-Flash configurations (pass 1)

Ten configurations, pass 1 of 198 questions each, the same prompts, answer order and request seed (1234):
3.25bpw on tpurtell 0.7.0 (with and without the kpool fixes), 0.8.0 (DFlash2 ×3; ×5 with sharing off),
0.8.0 + kpool fixes ≈ 0.9.0 (×5) and 0.9.1; 4bpw TR3 (Brandon) on tpurtell 0.8.0, 0.9.0 and 0.9.1, the last both at 8
concurrent requests (KV-saturated) and at 4. Three of these runs also hold passes 2 and 3 (request seeds 1235, 1236);
they are compared in [`glm53-flash-gpqa-records`](../glm53-flash-gpqa-records). This comparison uses pass 1 only.

Raw = lm-eval `flexible-extract`, the headline (protocol v1). Stated = `correct_stated`, the secondary, audited score:
whether the reply's stated final answer is the target. "Answered" excludes empty answers. `tools/analyze.py gpqa-table`.

| Configuration | Raw (95% interval) | Raw, answered | Stated | Stated, answered | Empty answers |
|---|---|---|---|---|---|
| `3.25bpw · tpurtell 0.7.0 · DFlash2 ×5` | 87.9% (82.8-91.9) | 87.9% | 89.4% | 89.4% | 0 |
| `3.25bpw · tpurtell 0.7.0 + kpool fixes · DFlash2 ×5` | 85.9% (80.8-90.4) | 86.7% | 87.4% | 88.3% | 2 |
| `3.25bpw · tpurtell 0.8.0 · DFlash2 ×3` | 85.9% (80.8-90.4) | 88.1% | 86.9% | 89.1% | 5 |
| `3.25bpw · tpurtell 0.8.0 · DFlash2 ×5, sharing off` | 86.9% (81.8-91.4) | 89.1% | 88.4% | 90.7% | 5 |
| `3.25bpw · tpurtell 0.8.0 + kpool fixes ≈ 0.9.0 · DFlash2 ×5` | 86.9% (82.3-91.4) | 88.2% | 88.4% | 89.7% | 3 |
| `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3` | 85.4% (80.3-89.9) | 87.6% | 85.9% | 88.1% | 5 (4 at the token cap, 1 stopped after 38 tokens) |
| `4bpw TR3 (Brandon) · tpurtell 0.8.0 · DFlash2 ×3` (a) | 85.9% (80.8-90.4) | 87.2% | 87.9% | 89.2% | 3 |
| `4bpw TR3 (Brandon) · tpurtell 0.9.0 · DFlash2 ×3` (a) | 84.3% (78.8-88.9) | 88.4% | 85.4% | 89.4% | 9 |
| `4bpw TR3 (Brandon) · tpurtell 0.9.1 · DFlash2 ×3` (b) | 84.8% (79.8-89.4) | 87.5% | 86.9% | 89.6% | 6 (all at the token cap) |
| `4bpw TR3 (Brandon) · tpurtell 0.9.1 · DFlash2 ×3 · concurrency 4` | 84.8% (79.8-89.4) | 88.4% | 86.9% | 90.5% | 8 (all at the token cap) |

(a) 8 concurrent requests against a KV pool of about 1.38 million tokens, which holds about 4 requests at the token cap;
no server log was kept, so whether requests waited for KV is not recorded. (b) **KV-saturated**: the server log shows
requests waiting for KV memory in 653 of 1,417 status lines, so fewer than 8 ran at once for much of the run. The record
at 4 concurrent requests, where no request waited, replaces it as this configuration's record; it is kept, labelled, as
the like-for-like partner of the 0.9.0 run.

Question-paired comparisons of configurations that differ in one named respect (`tools/analyze.py gpqa-pairs`; exact
McNemar tests on the questions where the two runs disagree; no correction for multiple comparisons):

| A vs B | What differs | Only A right / only B right (p) | B - A, points (95% interval) | Only A empty / only B empty (p) |
|---|---|---|---|---|
| 3.25bpw 0.7.0 ×5 vs 0.8.0 ×3 | engine release as shipped | 10 / 6 (0.45) | -2.0 (-6.1 to +2.0) | 0 / 5 (0.06) |
| 3.25bpw 0.8.0 ×3 vs ×5, sharing off | draft depth and slot sharing | 10 / 12 (0.83) | +1.0 (-3.5 to +5.6) | 3 / 3 (1.00) |
| 3.25bpw 0.7.0 vs 0.7.0 + kpool fixes | kpool fixes | 10 / 6 (0.45) | -2.0 (-6.1 to +2.0) | 0 / 2 (0.50) |
| 4bpw 0.8.0 vs 0.9.0 | kpool fixes (release) | 11 / 8 (0.65) | -1.5 (-6.1 to +3.0) | 2 / 8 (0.11) |
| 4bpw 0.9.0 vs 0.9.1 | DCP1 tail fix (release) | 13 / 14 (1.00) | +0.5 (-4.5 to +5.6) | 5 / 2 (0.45) |
| 3.25bpw vs 4bpw on 0.8.0 | weights | 10 / 10 (1.00) | 0.0 (-4.5 to +4.5) | 4 / 2 (0.69) |
| 3.25bpw vs 4bpw on 0.9.1 | weights (4bpw KV-saturated) | 13 / 12 (1.00) | -0.5 (-5.6 to +4.5) | 4 / 5 (1.00) |
| 4bpw 0.9.1, 8 vs 4 concurrent | concurrency (KV-saturated vs not) | 13 / 13 (1.00) | 0.0 (-5.1 to +5.1) | 3 / 5 (0.73) |

What one pass per configuration shows: no pair differs detectably in accuracy (raw scores; the stated-answer score is
0.5 to 2.0 points higher per run); each interval spans about ±5 points, so differences smaller than that cannot be seen.
Empty answers are 0 to 9 of 198 per run; with one pass each, no pair differs at p < 0.05.

Read with care:
- One draw per question per configuration, all with request seed 1234, so the comparisons are paired by question. Passes
  of one configuration with their own request seeds differ by 0.5 to 3.5 points
  ([`glm53-flash-gpqa-records`](../glm53-flash-gpqa-records)), about as much as the 84.3-87.9% range across these runs.
- No higher-precision reference (BF16 or NVFP4) was run on this hardware, so these runs do not measure what the EXL3
  quants cost in accuracy.
- Raw `flexible-extract` scores run 0.5 to 3.5 points low per pass (see the protocol's known limitation).
