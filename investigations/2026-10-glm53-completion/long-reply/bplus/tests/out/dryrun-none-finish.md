# B+ continuation result: COMPLETE

- stock 0.7.0: complete
- stock 0.9.1: complete
- B1 (0.9.1 + D1-D4 + D7): complete

**Reading: R-I.** Inconclusive under the prespecified rules: neither an engine continuation effect nor state dominance is supported.

## Primary: P(finish within +64k), prefix as the unit

| engine | P | finished / 48 | capped-origin | finished-origin |
|---|---:|---:|---:|---:|
| stock 0.7.0 | 0.000 | 0 | 0.000 | 0.000 |
| stock 0.9.1 | 0.000 | 0 | 0.000 | 0.000 |
| B1 (0.9.1 + D1-D4 + D7) | 0.000 | 0 | 0.000 | 0.000 |

| contrast | mean | 95% interval | 90% interval | sign-flip p | prefixes + / - / 0 | screen |
|---|---:|---|---|---:|---|---|
| C1 v070 - v091 | +0.000 | +0.000 to +0.000 | +0.000 to +0.000 | 1.0000 | 0 / 0 / 12 | SMALL |
| C2 b1 - v091 | +0.000 | +0.000 to +0.000 | +0.000 to +0.000 | 1.0000 | 0 / 0 / 12 | SMALL |
| C3 v070 - b1 | +0.000 | +0.000 to +0.000 | +0.000 to +0.000 | 1.0000 | 0 / 0 / 12 | SMALL |

Prefix share of outcome variance (eta^2, 144 outcomes): None; PREFIX_HIGH False; secondary: C2 SMALL, C3 SMALL.

## Secondary (descriptive)

| engine | any stop | median finish tokens | loop onsets in continuation (median after cut) | loop types (short / long / fragmentary / none) | cpt | unanimous prefixes | ICC | decode tok/s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| stock 0.7.0 | 0.000 | None | 21 (22631) | 21 / 0 / 0 / 27 | 3.0 | 12 / 12 | None | 35.52 |
| stock 0.9.1 | 0.000 | None | 30 (19759.5) | 30 / 0 / 0 / 18 | 3.0 | 12 / 12 | None | 41.48 |
| B1 (0.9.1 + D1-D4 + D7) | 0.000 | None | 17 (24577) | 17 / 0 / 0 / 31 | 3.0 | 12 / 12 | None | 41.49 |

Covariates at 100k vs pooled prefix rate (Spearman, n = 12, descriptive): {'chars_per_token': None, 'interjection_per1k': None, 'digit_share': None, 'R50_backward_at_cut': None}

Caveats: one 0.7.0-built state per question; prefixes rebuilt by prefill (decode-built state absent); re-tokenized ids (drift 0-8 tokens per reply); 'engine' bundles decode path, D2/D7 exposure, layout and speculation depth.
exit 0
