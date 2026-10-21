# B+ continuation result: COMPLETE

- stock 0.7.0: complete
- stock 0.9.1: complete
- B1 (0.9.1 + D1-D4 + D7): complete

**Reading: R-I.** Inconclusive under the prespecified rules: neither an engine continuation effect nor state dominance is supported.

## Primary: P(finish within +64k), prefix as the unit

| engine | P | finished / 48 | capped-origin | finished-origin |
|---|---:|---:|---:|---:|
| stock 0.7.0 | 0.417 | 20 | 0.125 | 0.708 |
| stock 0.9.1 | 0.417 | 20 | 0.167 | 0.667 |
| B1 (0.9.1 + D1-D4 + D7) | 0.521 | 25 | 0.292 | 0.750 |

| contrast | mean | 95% interval | 90% interval | sign-flip p | prefixes + / - / 0 | screen |
|---|---:|---|---|---:|---|---|
| C1 v070 - v091 | +0.000 | -0.083 to +0.083 | -0.062 to +0.062 | 1.0000 | 2 / 2 / 8 | SMALL |
| C2 b1 - v091 | +0.104 | +0.000 to +0.208 | +0.021 to +0.188 | 0.1875 | 5 / 1 / 6 | - |
| C3 v070 - b1 | -0.104 | -0.208 to -0.021 | -0.188 to -0.042 | 0.1250 | 0 / 4 / 8 | - |

Prefix share of outcome variance (eta^2, 144 outcomes): 0.6658227848101264; PREFIX_HIGH True; secondary: C2 unresolved, C3 unresolved.

## Secondary (descriptive)

| engine | any stop | median finish tokens | loop onsets in continuation (median after cut) | loop types (short / long / fragmentary / none) | cpt | unanimous prefixes | ICC | decode tok/s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| stock 0.7.0 | 0.417 | 28621.0 | 5 (24956) | 5 / 0 / 0 / 43 | 2.704 | 8 / 12 | 0.624 | 34.73 |
| stock 0.9.1 | 0.417 | 26764.5 | 6 (35405.0) | 4 / 0 / 2 / 42 | 2.722 | 7 / 12 | 0.569 | 48.44 |
| B1 (0.9.1 + D1-D4 + D7) | 0.521 | 26127 | 12 (31366.5) | 9 / 2 / 1 / 36 | 2.721 | 9 / 12 | 0.714 | 48.12 |

Covariates at 100k vs pooled prefix rate (Spearman, n = 12, descriptive): {'chars_per_token': 0.601, 'interjection_per1k': -0.523, 'digit_share': -0.35, 'R50_backward_at_cut': 0.136}

Caveats: one 0.7.0-built state per question; prefixes rebuilt by prefill (decode-built state absent); re-tokenized ids (drift 0-8 tokens per reply); 'engine' bundles decode path, D2/D7 exposure, layout and speculation depth.
exit 0
