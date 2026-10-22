# E prefix-source x engine continuation result: COMPLETE

- stock 0.7.0: complete
- stock 0.9.1: complete

**Reading: R-I.** Inconclusive under the prespecified rules: no engine continuation effect is supported in either direction from the 0.9.1-built states. With 13 prefixes this is the likeliest reading unless the effect is large (PROPOSAL.md 5.3).

## Primary: P(finish within +64k) on the 13 0.9.1-built prefixes, prefix as the unit

| engine | P | finished / 52 | capped-origin (8) | finished-origin (5) |
|---|---:|---:|---:|---:|
| stock 0.7.0 | 0.500 | 26 | 0.594 | 0.350 |
| stock 0.9.1 | 0.327 | 17 | 0.375 | 0.250 |

| contrast | mean | 95% interval | 90% interval | sign-flip p | prefixes + / - / 0 | screen |
|---|---:|---|---|---:|---|---|
| E1 v070 - v091 | +0.173 | -0.038 to +0.385 | +0.000 to +0.365 | 0.2109 | 8 / 4 / 1 | unresolved |

Prefix share of outcome variance (eta^2, 104 outcomes): 0.1029355699580633; PREFIX_HIGH False. B+'s frozen C1-ONLY rule on E1: R-E if EFFECT, R-E- if REVERSE, else R-I; R-S is not available with two engines. SMALL(E1) and PREFIX_HIGH are reported as screens, not a reading.

## Cross (descriptive, prespecified): prefix source x engine

| source x engine | P | 95% interval | finished / n |
|---|---:|---|---:|
| 0.7.0-built (B+) x v070 | 0.417 | 0.188 to 0.646 | 20 / 48 |
| 0.7.0-built (B+) x v091 | 0.417 | 0.188 to 0.646 | 20 / 48 |
| 0.9.1-built (E) x v070 | 0.500 | 0.385 to 0.635 | 26 / 52 |
| 0.9.1-built (E) x v091 | 0.327 | 0.173 to 0.462 | 17 / 52 |

- Source contrast (0.9.1-built - 0.7.0-built, pooled over engines; **confounded with question**): -0.003 (95% -0.248 to +0.231; permutation p 1.0)
- Interaction (E1 - C1): +0.173 (95% -0.056 to +0.410; permutation p 0.223); E1 +0.173, C1 +0.000

| stratum cell | P | 95% interval | prefixes |
|---|---:|---|---:|
| 0.7.0-built (B+) x v070 x originally capped | 0.125 | 0.000 to 0.292 | 6 |
| 0.7.0-built (B+) x v070 x originally finished | 0.708 | 0.417 to 0.958 | 6 |
| 0.7.0-built (B+) x v091 x originally capped | 0.167 | 0.000 to 0.417 | 6 |
| 0.7.0-built (B+) x v091 x originally finished | 0.667 | 0.375 to 0.917 | 6 |
| 0.9.1-built (E) x v070 x originally capped | 0.594 | 0.438 to 0.750 | 8 |
| 0.9.1-built (E) x v070 x originally finished | 0.350 | 0.250 to 0.550 | 5 |
| 0.9.1-built (E) x v091 x originally capped | 0.375 | 0.188 to 0.562 | 8 |
| 0.9.1-built (E) x v091 x originally finished | 0.250 | 0.050 to 0.450 | 5 |
| pooled sources x v070 x originally capped | 0.393 | 0.232 to 0.554 | 14 |
| pooled sources x v070 x originally finished | 0.545 | 0.341 to 0.750 | 11 |
| pooled sources x v091 x originally capped | 0.286 | 0.143 to 0.446 | 14 |
| pooled sources x v091 x originally finished | 0.477 | 0.273 to 0.682 | 11 |

Question-matched sub-table (same question in both prefix sets; different replies):

| doc | E prefix (orig.) | B+ prefix (orig.) | 0.9.1-built v070 / v091 | 0.7.0-built v070 / v091 |
|---|---|---|---|---|
| q69 | P09 (capped) | P03 (capped) | 0.75 / 0.25 | 0.25 / 0.00 |
| q71 | P13 (finished) | P10 (finished) | 0.25 / 0.50 | 1.00 / 0.75 |
| q79 | P01 (capped) | P01 (capped) | 0.75 / 0.50 | 0.00 / 0.25 |
| q81 | P02 (capped) | P09 (finished) | 0.50 / 0.00 | 1.00 / 1.00 |
| q88 | P12 (finished) | P02 (capped) | 0.25 / 0.00 | 0.00 / 0.00 |
| q120 | P10 (capped) | P04 (capped) | 0.50 / 0.75 | 0.00 / 0.00 |
| q145 | P06 (finished) | P12 (finished) | 0.75 / 0.25 | 1.00 / 1.00 |
| q186 | P11 (capped) | P11 (finished) | 0.50 / 0.50 | 0.75 / 0.75 |

Caveats: Prefixes differ by question between the sources (8 of 17 questions shared), so the source contrast is confounded with question; the question-matched sub-table is the only within-question comparison. B+ ran 2026-10-20/21 and E runs later: session date, thermal state and seeds differ between sources. Descriptive only: no reading is drawn from the cross.

## Secondary (descriptive)

| engine | any stop | median finish tokens | loop onsets in continuation (median after cut) | loop types (short / long / fragmentary / none) | cpt | unanimous prefixes | ICC | decode tok/s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| stock 0.7.0 | 0.500 | 30812.5 | 12 (20998.0) | 12 / 0 / 0 / 40 | 3.0 | 1 / 13 | -0.006 | 41.67 |
| stock 0.9.1 | 0.327 | 40988 | 17 (15617) | 17 / 0 / 0 / 35 | 3.0 | 4 / 13 | 0.121 | 41.86 |

Covariates at 100k vs pooled prefix rate (Spearman, n = 13, descriptive): {'chars_per_token': -0.43, 'interjection_per1k': 0.205, 'digit_share': 0.55, 'R50_backward_at_cut': 0.236}

Caveats: one 0.9.1-built state per question; prefixes rebuilt by prefill (decode-built state absent); re-tokenized ids (drift 0-6 tokens per reply); 'engine' bundles decode path, D2/D7 exposure, layout and speculation depth.
exit 0
