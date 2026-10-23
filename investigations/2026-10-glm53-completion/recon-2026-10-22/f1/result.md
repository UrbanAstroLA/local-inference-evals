# F1: stock 0.7.0 vs stock 0.9.1, original / fresh-only / pooled (post hoc, CPU only)

**Status: POST HOC, descriptive.** Every input had been inspected before. The frozen readings stand: rep070 O3 (22p),
B+ R-I (22w), E R-I (22ak). The rules (`stock_compare.py` RULES) were written before the first run. That run was
stopped for speed before writing any output. The rerun changed only the computation of the same tests (count-based
hypergeometric permutation, the same bootstrap), not the rules or the seeds.

**Inputs:** `manifest.json` lists sha256, image digests, sessions, seeds and coverage. Summary:
- Images: 0.7.0 `48e254d9…`; 0.9.1 `a9e985c0…`.
- Weights: all 18 shards, config, tokenizer and tokenizer_config are identical blobs in revisions `0490d2f7` and
  `701cd745`. Only the chat template and README differ; prompt hashes are equal for all 198 questions in all 11 passes.
- Draft: `incoai/GLM-5.3-Flash-DFlash2@bf582e4e` in every logged session. 0.7.0 record pass 1 has no server log;
  the stack pins `bf582e4e` by default.
- Missing data: 0.7.0 record pass 1 has no request log, so the capped / reach / finish endpoints use 5 logged passes
  per engine. The empty endpoint uses all 11 passes.
- Seeds: record passes 1234-1236, used on both engines; 0.7.0 rep070 70701-70703; 0.9.1 D 70704, D2 70705.
- Sessions: 0.7.0 rec-A (p1), rec-B (p2-3), rep-s1 (p1), rep-s2 (p2-3); 0.9.1 rec-A (p1), rec-B (p2-3), D, D2.

**Unequal pass counts:** each engine's rate is computed per question (events / passes on that question) and averaged
with equal weight per question. The within-question permutation keeps each engine's pass count on every question.

## Empty replies (all 11 passes)

| Comparison | 0.7.0 | 0.9.1 | Δ rate, 0.7.0 − 0.9.1 (95% question bootstrap) | Rate ratio (95%) | Within-question permutation, two-sided p (Monte Carlo, 20,000 draws) | Pass-level exact permutation, one-sided / two-sided p | Session-level exact permutation, one-sided / two-sided p |
|---|---|---|---|---|---:|---:|---:|
| Original (3 vs 3 passes, same seeds) | 4 / 594 (0, 3, 1) | 16 / 594 (5, 5, 6) | −2.02 pp (−3.54 to −0.67) | 0.25 (0.06–0.55) | 0.005 | 0.050 / 0.10 (20 arrangements) | 0.17 / 0.33 (6) |
| Fresh seeds only (3 vs 2) | 10 / 594 (2, 4, 4) | 14 / 396 (6, 8) | −1.85 pp (−4.12 to +0.17) | 0.48 (0.15–1.09) | 0.040 | 0.10 / 0.10 (10) | 0.17 / 0.33 (6) |
| Pooled (6 vs 5) | 14 / 1,188 | 30 / 990 | −1.85 pp (−3.23 to −0.66) | 0.39 (0.19–0.65) | 0.001 | 0.0022 / 0.0065 (462) | 0.0143 / 0.0286 (70) |

- **Fresh seeds alone:** the interval runs from 4.1 points fewer to 0.2 points more empties on 0.7.0, and the ratio
  from 0.15 to 1.09. A ~2-point reduction is as compatible with these data as a small one. This is not evidence that
  the gap is absent.
- **Pooled:** every 0.7.0 pass (0-4 empties) is below every 0.9.1 pass (5-8), and every 0.7.0 session's rate is
  below every 0.9.1 session's.
- **Assumptions, stated** (corrected 22an: the within-question test is Monte Carlo, not exact; the pass- and session-level tests are exact under exchangeability of historical passes and sessions):
  - Passes and sessions are exchangeable between engines under the null.
  - Questions are independent units.
  - Same-seed passes on the two engines are coupled (log reply-length correlation 0.66-0.91, 22m). That coupling
    makes the original comparison partly paired and is not an inflation.
  - The choice to pool was made after all counts were known.

## Where the difference sits (logged passes: 5 per engine pooled)

| Endpoint | Original | Fresh | Pooled |
|---|---|---|---|
| Reach 100k: rate ratio 0.7.0 / 0.9.1 | 0.84 (0.45–1.50) | 0.88 (0.55–1.38) | **0.92 (0.67–1.26)**: 43 / 990 vs 47 / 990 |
| Finish once past 100k | 10/14 vs 10/25: +0.31 (+0.04 to +0.55) | 19/29 vs 8/22: +0.29 (−0.02 to +0.60) | **29/43 vs 18/47: +0.29 (+0.10 to +0.51)** |
| … question-matched (questions at risk on both engines) | 7 q: +0.21 (−0.05 to +0.48) | 11 q: +0.14 (−0.26 to +0.53) | 13 q: +0.18 (−0.09 to +0.46) |
| Finish by 165,536 tokens once past 100k (the B+/E horizon) | 6/14 vs 7/25 | 16/29 vs 8/22 | 22/43 vs 15/47: +0.19 (−0.05 to +0.45) |

**Readings, descriptive:**
- There is no clear difference in reaching 100k (pooled ratio 0.92, 0.67-1.26). A difference within that interval is not excluded (wording corrected, 22an). The clear difference is in what happens after 100k.
- Part of the finishing difference arises after 165k tokens: 7 of 0.7.0's 29 finishes against 3 of 0.9.1's 18.
  The continuation experiments' +64k horizon could not see that part. Their own rates are consistent with the
  natural +64k rates (B+ 0.7.0 on 0.7.0 prefixes; E 0.9.1 on 0.9.1 prefixes).
- The question-matched estimates are smaller and wider. Which questions go long explains part, not obviously all,
  of the pooled difference.

## Secondary, descriptive only (single passes, seed 1234, no request logs)
- **0.8.0:** ×3 left 5; ×5 with draft-slot sharing off left 5; ×5 with the kpool fixes left 3.
- **0.7.0 with the kpool fixes, ×5:** 2.
- Not pooled into anything above.

## What this does and does not support
- **Supports, descriptively and post hoc:** in all stock data, 0.7.0 leaves fewer empty replies than 0.9.1. The
  pooled reduction is about 1.9 points (ratio about 0.4), with the clear difference in finishing after 100k and no clear difference in reaching 100k.
- **Does not support:**
  - a mechanism;
  - a population effect free of session confounding (4 sessions per engine);
  - an accuracy effect;
  - any conclusion from the fresh-only interval touching 0.
