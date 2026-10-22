# E prefix-source x engine continuation result: COMPLETE

- stock 0.7.0: complete
- stock 0.9.1: complete

**Reading: R-E.** From identical 0.9.1-built 100k-token states, stock 0.7.0's continuation finished within +64k more often than stock 0.9.1's, by at least the prespecified 0.15 with the prefix-level interval and test excluding no difference. This does not identify which component of 'engine' is responsible, and holds for these 13 0.9.1-built states only.

## Primary: P(finish within +64k) on the 13 0.9.1-built prefixes, prefix as the unit

| engine | P | finished / 52 | capped-origin (8) | finished-origin (5) |
|---|---:|---:|---:|---:|
| stock 0.7.0 | 0.731 | 38 | 0.719 | 0.750 |
| stock 0.9.1 | 0.250 | 13 | 0.312 | 0.150 |

| contrast | mean | 95% interval | 90% interval | sign-flip p | prefixes + / - / 0 | screen |
|---|---:|---|---|---:|---|---|
| E1 v070 - v091 | +0.481 | +0.346 to +0.615 | +0.365 to +0.596 | 0.0005 | 12 / 0 / 1 | EFFECT |

Prefix share of outcome variance (eta^2, 104 outcomes): 0.1006289308176101; PREFIX_HIGH False. B+'s frozen C1-ONLY rule on E1: R-E if EFFECT, R-E- if REVERSE, else R-I; R-S is not available with two engines. SMALL(E1) and PREFIX_HIGH are reported as screens, not a reading.

## Cross (descriptive, prespecified): prefix source x engine

| source x engine | P | 95% interval | finished / n |
|---|---:|---|---:|
| 0.7.0-built (B+) x v070 | 0.417 | 0.188 to 0.646 | 20 / 48 |
| 0.7.0-built (B+) x v091 | 0.417 | 0.188 to 0.646 | 20 / 48 |
| 0.9.1-built (E) x v070 | 0.731 | 0.615 to 0.827 | 38 / 52 |
| 0.9.1-built (E) x v091 | 0.250 | 0.135 to 0.365 | 13 / 52 |

- Source contrast (0.9.1-built - 0.7.0-built, pooled over engines; **confounded with question**): +0.074 (95% -0.169 to +0.310; permutation p 0.5775)
- Interaction (E1 - C1): +0.481 (95% +0.324 to +0.638; permutation p 0.0); E1 +0.481, C1 +0.000

| stratum cell | P | 95% interval | prefixes |
|---|---:|---|---:|
| 0.7.0-built (B+) x v070 x originally capped | 0.125 | 0.000 to 0.292 | 6 |
| 0.7.0-built (B+) x v070 x originally finished | 0.708 | 0.417 to 0.958 | 6 |
| 0.7.0-built (B+) x v091 x originally capped | 0.167 | 0.000 to 0.417 | 6 |
| 0.7.0-built (B+) x v091 x originally finished | 0.667 | 0.375 to 0.917 | 6 |
| 0.9.1-built (E) x v070 x originally capped | 0.719 | 0.562 to 0.844 | 8 |
| 0.9.1-built (E) x v070 x originally finished | 0.750 | 0.600 to 0.900 | 5 |
| 0.9.1-built (E) x v091 x originally capped | 0.312 | 0.156 to 0.469 | 8 |
| 0.9.1-built (E) x v091 x originally finished | 0.150 | 0.050 to 0.250 | 5 |
| pooled sources x v070 x originally capped | 0.464 | 0.286 to 0.643 | 14 |
| pooled sources x v070 x originally finished | 0.727 | 0.545 to 0.886 | 11 |
| pooled sources x v091 x originally capped | 0.250 | 0.125 to 0.393 | 14 |
| pooled sources x v091 x originally finished | 0.432 | 0.227 to 0.659 | 11 |

Question-matched sub-table (same question in both prefix sets; different replies):

| doc | E prefix (orig.) | B+ prefix (orig.) | 0.9.1-built v070 / v091 | 0.7.0-built v070 / v091 |
|---|---|---|---|---|
| q69 | P09 (capped) | P03 (capped) | 0.75 / 0.00 | 0.25 / 0.00 |
| q71 | P13 (finished) | P10 (finished) | 0.75 / 0.00 | 1.00 / 0.75 |
| q79 | P01 (capped) | P01 (capped) | 0.75 / 0.75 | 0.00 / 0.25 |
| q81 | P02 (capped) | P09 (finished) | 0.75 / 0.50 | 1.00 / 1.00 |
| q88 | P12 (finished) | P02 (capped) | 1.00 / 0.00 | 0.00 / 0.00 |
| q120 | P10 (capped) | P04 (capped) | 1.00 / 0.50 | 0.00 / 0.00 |
| q145 | P06 (finished) | P12 (finished) | 0.50 / 0.25 | 1.00 / 1.00 |
| q186 | P11 (capped) | P11 (finished) | 0.75 / 0.25 | 0.75 / 0.75 |

Caveats: Prefixes differ by question between the sources (8 of 17 questions shared), so the source contrast is confounded with question; the question-matched sub-table is the only within-question comparison. B+ ran 2026-10-20/21 and E runs later: session date, thermal state and seeds differ between sources. Descriptive only: no reading is drawn from the cross.

## Secondary (descriptive)

| engine | any stop | median finish tokens | loop onsets in continuation (median after cut) | loop types (short / long / fragmentary / none) | cpt | unanimous prefixes | ICC | decode tok/s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| stock 0.7.0 | 0.731 | 27699.0 | 7 (19514) | 7 / 0 / 0 / 45 | 3.0 | 2 / 13 | -0.092 | 39.6 |
| stock 0.9.1 | 0.250 | 22537 | 14 (25362.5) | 14 / 0 / 0 / 38 | 3.0 | 4 / 13 | 0.029 | 42.8 |

Covariates at 100k vs pooled prefix rate (Spearman, n = 13, descriptive): {'chars_per_token': -0.635, 'interjection_per1k': 0.14, 'digit_share': 0.734, 'R50_backward_at_cut': 0.763}

Caveats: one 0.9.1-built state per question; prefixes rebuilt by prefill (decode-built state absent); re-tokenized ids (drift 0-6 tokens per reply); 'engine' bundles decode path, D2/D7 exposure, layout and speculation depth.
exit 0
