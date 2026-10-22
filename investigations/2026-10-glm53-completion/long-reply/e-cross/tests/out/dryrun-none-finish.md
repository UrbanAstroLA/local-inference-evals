# E prefix-source x engine continuation result: COMPLETE

- stock 0.7.0: complete
- stock 0.9.1: complete

**Reading: R-I.** Inconclusive under the prespecified rules: no engine continuation effect is supported in either direction from the 0.9.1-built states. With 13 prefixes this is the likeliest reading unless the effect is large (PROPOSAL.md 5.3).

## Primary: P(finish within +64k) on the 13 0.9.1-built prefixes, prefix as the unit

| engine | P | finished / 52 | capped-origin (8) | finished-origin (5) |
|---|---:|---:|---:|---:|
| stock 0.7.0 | 0.000 | 0 | 0.000 | 0.000 |
| stock 0.9.1 | 0.000 | 0 | 0.000 | 0.000 |

| contrast | mean | 95% interval | 90% interval | sign-flip p | prefixes + / - / 0 | screen |
|---|---:|---|---|---:|---|---|
| E1 v070 - v091 | +0.000 | +0.000 to +0.000 | +0.000 to +0.000 | 1.0000 | 0 / 0 / 13 | SMALL |

Prefix share of outcome variance (eta^2, 104 outcomes): None; PREFIX_HIGH False. B+'s frozen C1-ONLY rule on E1: R-E if EFFECT, R-E- if REVERSE, else R-I; R-S is not available with two engines. SMALL(E1) and PREFIX_HIGH are reported as screens, not a reading.

## Cross (descriptive, prespecified): prefix source x engine

| source x engine | P | 95% interval | finished / n |
|---|---:|---|---:|
| 0.7.0-built (B+) x v070 | 0.417 | 0.188 to 0.646 | 20 / 48 |
| 0.7.0-built (B+) x v091 | 0.417 | 0.188 to 0.646 | 20 / 48 |
| 0.9.1-built (E) x v070 | 0.000 | 0.000 to 0.000 | 0 / 52 |
| 0.9.1-built (E) x v091 | 0.000 | 0.000 to 0.000 | 0 / 52 |

- Source contrast (0.9.1-built - 0.7.0-built, pooled over engines; **confounded with question**): -0.417 (95% -0.646 to -0.198; permutation p 0.0005)
- Interaction (E1 - C1): +0.000 (95% -0.083 to +0.083; permutation p 1.0); E1 +0.000, C1 +0.000

| stratum cell | P | 95% interval | prefixes |
|---|---:|---|---:|
| 0.7.0-built (B+) x v070 x originally capped | 0.125 | 0.000 to 0.292 | 6 |
| 0.7.0-built (B+) x v070 x originally finished | 0.708 | 0.417 to 0.958 | 6 |
| 0.7.0-built (B+) x v091 x originally capped | 0.167 | 0.000 to 0.417 | 6 |
| 0.7.0-built (B+) x v091 x originally finished | 0.667 | 0.375 to 0.917 | 6 |
| 0.9.1-built (E) x v070 x originally capped | 0.000 | 0.000 to 0.000 | 8 |
| 0.9.1-built (E) x v070 x originally finished | 0.000 | 0.000 to 0.000 | 5 |
| 0.9.1-built (E) x v091 x originally capped | 0.000 | 0.000 to 0.000 | 8 |
| 0.9.1-built (E) x v091 x originally finished | 0.000 | 0.000 to 0.000 | 5 |
| pooled sources x v070 x originally capped | 0.054 | 0.000 to 0.143 | 14 |
| pooled sources x v070 x originally finished | 0.386 | 0.136 to 0.659 | 11 |
| pooled sources x v091 x originally capped | 0.071 | 0.000 to 0.196 | 14 |
| pooled sources x v091 x originally finished | 0.364 | 0.136 to 0.614 | 11 |

Question-matched sub-table (same question in both prefix sets; different replies):

| doc | E prefix (orig.) | B+ prefix (orig.) | 0.9.1-built v070 / v091 | 0.7.0-built v070 / v091 |
|---|---|---|---|---|
| q69 | P09 (capped) | P03 (capped) | 0.00 / 0.00 | 0.25 / 0.00 |
| q71 | P13 (finished) | P10 (finished) | 0.00 / 0.00 | 1.00 / 0.75 |
| q79 | P01 (capped) | P01 (capped) | 0.00 / 0.00 | 0.00 / 0.25 |
| q81 | P02 (capped) | P09 (finished) | 0.00 / 0.00 | 1.00 / 1.00 |
| q88 | P12 (finished) | P02 (capped) | 0.00 / 0.00 | 0.00 / 0.00 |
| q120 | P10 (capped) | P04 (capped) | 0.00 / 0.00 | 0.00 / 0.00 |
| q145 | P06 (finished) | P12 (finished) | 0.00 / 0.00 | 1.00 / 1.00 |
| q186 | P11 (capped) | P11 (finished) | 0.00 / 0.00 | 0.75 / 0.75 |

Caveats: Prefixes differ by question between the sources (8 of 17 questions shared), so the source contrast is confounded with question; the question-matched sub-table is the only within-question comparison. B+ ran 2026-10-20/21 and E runs later: session date, thermal state and seeds differ between sources. Descriptive only: no reading is drawn from the cross.

## Secondary (descriptive)

| engine | any stop | median finish tokens | loop onsets in continuation (median after cut) | loop types (short / long / fragmentary / none) | cpt | unanimous prefixes | ICC | decode tok/s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| stock 0.7.0 | 0.000 | None | 23 (22631) | 23 / 0 / 0 / 29 | 3.0 | 13 / 13 | None | 35.52 |
| stock 0.9.1 | 0.000 | None | 31 (19710) | 31 / 0 / 0 / 21 | 3.0 | 13 / 13 | None | 42.52 |

Covariates at 100k vs pooled prefix rate (Spearman, n = 13, descriptive): {'chars_per_token': None, 'interjection_per1k': None, 'digit_share': None, 'R50_backward_at_cut': None}

Caveats: one 0.9.1-built state per question; prefixes rebuilt by prefill (decode-built state absent); re-tokenized ids (drift 0-6 tokens per reply); 'engine' bundles decode path, D2/D7 exposure, layout and speculation depth.
exit 0
