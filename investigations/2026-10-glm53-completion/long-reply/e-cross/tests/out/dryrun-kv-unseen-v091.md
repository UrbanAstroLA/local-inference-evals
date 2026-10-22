# E prefix-source x engine continuation result: COMPLETE

- stock 0.7.0: complete
- stock 0.9.1: complete

**Reading: R-E.** From identical 0.9.1-built 100k-token states, stock 0.7.0's continuation finished within +64k more often than stock 0.9.1's, by at least the prespecified 0.15 with the prefix-level interval and test excluding no difference. This does not identify which component of 'engine' is responsible, and holds for these 13 0.9.1-built states only.

## Primary: P(finish within +64k) on the 13 0.9.1-built prefixes, prefix as the unit

| engine | P | finished / 52 | capped-origin (8) | finished-origin (5) |
|---|---:|---:|---:|---:|
| stock 0.7.0 | 0.712 | 37 | 0.688 | 0.750 |
| stock 0.9.1 | 0.154 | 8 | 0.188 | 0.100 |

| contrast | mean | 95% interval | 90% interval | sign-flip p | prefixes + / - / 0 | screen |
|---|---:|---|---|---:|---|---|
| E1 v070 - v091 | +0.558 | +0.442 to +0.654 | +0.462 to +0.635 | 0.0005 | 12 / 0 / 1 | EFFECT |

Prefix share of outcome variance (eta^2, 104 outcomes): 0.06478342749529191; PREFIX_HIGH False. B+'s frozen C1-ONLY rule on E1: R-E if EFFECT, R-E- if REVERSE, else R-I; R-S is not available with two engines. SMALL(E1) and PREFIX_HIGH are reported as screens, not a reading.

## Cross (descriptive, prespecified): prefix source x engine

| source x engine | P | 95% interval | finished / n |
|---|---:|---|---:|
| 0.7.0-built (B+) x v070 | 0.417 | 0.188 to 0.646 | 20 / 48 |
| 0.7.0-built (B+) x v091 | 0.417 | 0.188 to 0.646 | 20 / 48 |
| 0.9.1-built (E) x v070 | 0.712 | 0.654 to 0.750 | 37 / 52 |
| 0.9.1-built (E) x v091 | 0.154 | 0.058 to 0.288 | 8 / 52 |

- Source contrast (0.9.1-built - 0.7.0-built, pooled over engines; **confounded with question**): +0.016 (95% -0.221 to +0.244; permutation p 0.9347)
- Interaction (E1 - C1): +0.558 (95% +0.418 to +0.681; permutation p 0.0); E1 +0.558, C1 +0.000

| stratum cell | P | 95% interval | prefixes |
|---|---:|---|---:|
| 0.7.0-built (B+) x v070 x originally capped | 0.125 | 0.000 to 0.292 | 6 |
| 0.7.0-built (B+) x v070 x originally finished | 0.708 | 0.417 to 0.958 | 6 |
| 0.7.0-built (B+) x v091 x originally capped | 0.167 | 0.000 to 0.417 | 6 |
| 0.7.0-built (B+) x v091 x originally finished | 0.667 | 0.375 to 0.917 | 6 |
| 0.9.1-built (E) x v070 x originally capped | 0.688 | 0.594 to 0.750 | 8 |
| 0.9.1-built (E) x v070 x originally finished | 0.750 | 0.750 to 0.750 | 5 |
| 0.9.1-built (E) x v091 x originally capped | 0.188 | 0.031 to 0.375 | 8 |
| 0.9.1-built (E) x v091 x originally finished | 0.100 | 0.000 to 0.200 | 5 |
| pooled sources x v070 x originally capped | 0.446 | 0.286 to 0.607 | 14 |
| pooled sources x v070 x originally finished | 0.727 | 0.545 to 0.864 | 11 |
| pooled sources x v091 x originally capped | 0.179 | 0.054 to 0.321 | 14 |
| pooled sources x v091 x originally finished | 0.409 | 0.182 to 0.636 | 11 |

Question-matched sub-table (same question in both prefix sets; different replies):

| doc | E prefix (orig.) | B+ prefix (orig.) | 0.9.1-built v070 / v091 | 0.7.0-built v070 / v091 |
|---|---|---|---|---|
| q69 | P09 (capped) | P03 (capped) | 0.75 / 0.75 | 0.25 / 0.00 |
| q71 | P13 (finished) | P10 (finished) | 0.75 / 0.00 | 1.00 / 0.75 |
| q79 | P01 (capped) | P01 (capped) | 0.50 / 0.00 | 0.00 / 0.25 |
| q81 | P02 (capped) | P09 (finished) | 0.75 / 0.25 | 1.00 / 1.00 |
| q88 | P12 (finished) | P02 (capped) | 0.75 / 0.00 | 0.00 / 0.00 |
| q120 | P10 (capped) | P04 (capped) | 0.75 / 0.25 | 0.00 / 0.00 |
| q145 | P06 (finished) | P12 (finished) | 0.75 / 0.00 | 1.00 / 1.00 |
| q186 | P11 (capped) | P11 (finished) | 0.75 / 0.00 | 0.75 / 0.75 |

Caveats: Prefixes differ by question between the sources (8 of 17 questions shared), so the source contrast is confounded with question; the question-matched sub-table is the only within-question comparison. B+ ran 2026-10-20/21 and E runs later: session date, thermal state and seeds differ between sources. Descriptive only: no reading is drawn from the cross.

## Secondary (descriptive)

| engine | any stop | median finish tokens | loop onsets in continuation (median after cut) | loop types (short / long / fragmentary / none) | cpt | unanimous prefixes | ICC | decode tok/s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| stock 0.7.0 | 0.712 | 35487 | 10 (13988.5) | 10 / 0 / 0 / 42 | 3.0 | 0 / 13 | -0.276 | 42.16 |
| stock 0.9.1 | 0.154 | 23840.0 | 24 (21203.5) | 24 / 0 / 0 / 28 | 3.0 | 7 / 13 | 0.138 | 43.64 |

Covariates at 100k vs pooled prefix rate (Spearman, n = 13, descriptive): {'chars_per_token': -0.131, 'interjection_per1k': 0.274, 'digit_share': 0.21, 'R50_backward_at_cut': 0.206}

Caveats: one 0.9.1-built state per question; prefixes rebuilt by prefill (decode-built state absent); re-tokenized ids (drift 0-6 tokens per reply); 'engine' bundles decode path, D2/D7 exposure, layout and speculation depth.
exit 0
