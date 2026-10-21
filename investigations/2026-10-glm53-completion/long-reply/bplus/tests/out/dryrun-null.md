# B+ continuation result: COMPLETE

- stock 0.7.0: complete
- stock 0.9.1: complete
- B1 (0.9.1 + D1-D4 + D7): complete

**Reading: R-I.** Inconclusive under the prespecified rules: neither an engine continuation effect nor state dominance is supported.

## Primary: P(finish within +64k), prefix as the unit

| engine | P | finished / 48 | capped-origin | finished-origin |
|---|---:|---:|---:|---:|
| stock 0.7.0 | 0.521 | 25 | 0.500 | 0.542 |
| stock 0.9.1 | 0.333 | 16 | 0.458 | 0.208 |
| B1 (0.9.1 + D1-D4 + D7) | 0.438 | 21 | 0.375 | 0.500 |

| contrast | mean | 95% interval | 90% interval | sign-flip p | prefixes + / - / 0 | screen |
|---|---:|---|---|---:|---|---|
| C1 v070 - v091 | +0.188 | -0.062 to +0.458 | -0.021 to +0.417 | 0.2559 | 7 / 5 / 0 | - |
| C2 b1 - v091 | +0.104 | -0.083 to +0.292 | -0.062 to +0.250 | 0.4141 | 7 / 4 / 1 | - |
| C3 v070 - b1 | +0.083 | -0.083 to +0.229 | -0.042 to +0.208 | 0.4688 | 5 / 2 / 5 | - |

Prefix share of outcome variance (eta^2, 144 outcomes): 0.04642014162077106; PREFIX_HIGH False; secondary: C2 unresolved, C3 unresolved.

## Secondary (descriptive)

| engine | any stop | median finish tokens | loop onsets in continuation (median after cut) | loop types (short / long / fragmentary / none) | cpt | unanimous prefixes | ICC | decode tok/s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| stock 0.7.0 | 0.521 | 30833 | 10 (20998.0) | 10 / 0 / 0 / 38 | 3.0 | 1 / 12 | -0.126 | 41.11 |
| stock 0.9.1 | 0.333 | 42135.0 | 18 (15716.5) | 18 / 0 / 0 / 30 | 3.0 | 5 / 12 | 0.341 | 41.23 |
| B1 (0.9.1 + D1-D4 + D7) | 0.438 | 21476 | 14 (20121.5) | 14 / 0 / 0 / 34 | 3.0 | 0 / 12 | -0.204 | 38.77 |

Covariates at 100k vs pooled prefix rate (Spearman, n = 12, descriptive): {'chars_per_token': -0.029, 'interjection_per1k': 0.145, 'digit_share': -0.327, 'R50_backward_at_cut': -0.279}

Caveats: one 0.7.0-built state per question; prefixes rebuilt by prefill (decode-built state absent); re-tokenized ids (drift 0-8 tokens per reply); 'engine' bundles decode path, D2/D7 exposure, layout and speculation depth.
exit 0
