# B+ continuation result: COMPLETE

- stock 0.7.0: complete
- stock 0.9.1: complete
- B1 (0.9.1 + D1-D4 + D7): complete

**Reading: R-E.** From identical 100k-token states, stock 0.7.0's continuation finished within +64k more often than stock 0.9.1's, by at least the prespecified 0.15 with the prefix-level interval and test excluding no difference. The engines' continuations differ; this does not identify which component of 'engine' (decode path, D2/D7, layout, speculation depth) is responsible, and holds for 0.7.0-built states only.

## Primary: P(finish within +64k), prefix as the unit

| engine | P | finished / 48 | capped-origin | finished-origin |
|---|---:|---:|---:|---:|
| stock 0.7.0 | 0.708 | 34 | 0.542 | 0.875 |
| stock 0.9.1 | 0.167 | 8 | 0.125 | 0.208 |
| B1 (0.9.1 + D1-D4 + D7) | 0.167 | 8 | 0.208 | 0.125 |

| contrast | mean | 95% interval | 90% interval | sign-flip p | prefixes + / - / 0 | screen |
|---|---:|---|---|---:|---|---|
| C1 v070 - v091 | +0.542 | +0.375 to +0.708 | +0.396 to +0.688 | 0.0020 | 10 / 0 / 2 | EFFECT |
| C2 b1 - v091 | +0.000 | -0.125 to +0.167 | -0.104 to +0.125 | 1.0000 | 1 / 3 / 8 | SMALL |
| C3 v070 - b1 | +0.542 | +0.354 to +0.729 | +0.375 to +0.708 | 0.0020 | 10 / 0 / 2 | EFFECT |

Prefix share of outcome variance (eta^2, 144 outcomes): 0.09106382978723405; PREFIX_HIGH False; secondary: C2 SMALL, C3 EFFECT.

## Secondary (descriptive)

| engine | any stop | median finish tokens | loop onsets in continuation (median after cut) | loop types (short / long / fragmentary / none) | cpt | unanimous prefixes | ICC | decode tok/s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| stock 0.7.0 | 0.708 | 26677.0 | 7 (19514) | 7 / 0 / 0 / 41 | 3.0 | 3 / 12 | 0.013 | 39.98 |
| stock 0.9.1 | 0.167 | 43718.5 | 22 (22146.0) | 22 / 0 / 0 / 26 | 3.0 | 7 / 12 | 0.126 | 38.34 |
| B1 (0.9.1 + D1-D4 + D7) | 0.167 | 29164.0 | 24 (19243.0) | 24 / 0 / 0 / 24 | 3.0 | 7 / 12 | 0.228 | 45.24 |

Covariates at 100k vs pooled prefix rate (Spearman, n = 12, descriptive): {'chars_per_token': 0.0, 'interjection_per1k': -0.805, 'digit_share': -0.075, 'R50_backward_at_cut': 0.427}

Caveats: one 0.7.0-built state per question; prefixes rebuilt by prefill (decode-built state absent); re-tokenized ids (drift 0-8 tokens per reply); 'engine' bundles decode path, D2/D7 exposure, layout and speculation depth.
exit 0
