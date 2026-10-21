# B+ continuation result: COMPLETE

- stock 0.7.0: complete
- stock 0.9.1: complete
- B1 (0.9.1 + D1-D4 + D7): complete

**Reading: R-S.** State-dominated under these rules: all three engine contrasts were small (|mean| <= 0.10, 90% interval inside +-0.20) and the prefix explained at least 30% of the outcome variance. The finishing gap is then more plausibly set before 100k than by the continuation; small contrasts here are not proof of equality.

## Primary: P(finish within +64k), prefix as the unit

| engine | P | finished / 48 | capped-origin | finished-origin |
|---|---:|---:|---:|---:|
| stock 0.7.0 | 0.500 | 24 | 0.000 | 1.000 |
| stock 0.9.1 | 0.500 | 24 | 0.000 | 1.000 |
| B1 (0.9.1 + D1-D4 + D7) | 0.500 | 24 | 0.000 | 1.000 |

| contrast | mean | 95% interval | 90% interval | sign-flip p | prefixes + / - / 0 | screen |
|---|---:|---|---|---:|---|---|
| C1 v070 - v091 | +0.000 | +0.000 to +0.000 | +0.000 to +0.000 | 1.0000 | 0 / 0 / 12 | SMALL |
| C2 b1 - v091 | +0.000 | +0.000 to +0.000 | +0.000 to +0.000 | 1.0000 | 0 / 0 / 12 | SMALL |
| C3 v070 - b1 | +0.000 | +0.000 to +0.000 | +0.000 to +0.000 | 1.0000 | 0 / 0 / 12 | SMALL |

Prefix share of outcome variance (eta^2, 144 outcomes): 1.0; PREFIX_HIGH True; secondary: C2 SMALL, C3 SMALL.

## Secondary (descriptive)

| engine | any stop | median finish tokens | loop onsets in continuation (median after cut) | loop types (short / long / fragmentary / none) | cpt | unanimous prefixes | ICC | decode tok/s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| stock 0.7.0 | 0.500 | 40090.5 | 15 (25214) | 15 / 0 / 0 / 33 | 3.0 | 12 / 12 | 1.0 | 42.27 |
| stock 0.9.1 | 0.500 | 27066.0 | 10 (27169.0) | 10 / 0 / 0 / 38 | 3.0 | 12 / 12 | 1.0 | 36.23 |
| B1 (0.9.1 + D1-D4 + D7) | 0.500 | 35519.5 | 11 (15646) | 11 / 0 / 0 / 37 | 3.0 | 12 / 12 | 1.0 | 40.03 |

Covariates at 100k vs pooled prefix rate (Spearman, n = 12, descriptive): {'chars_per_token': 0.483, 'interjection_per1k': -0.386, 'digit_share': -0.338, 'R50_backward_at_cut': 0.3}

Caveats: one 0.7.0-built state per question; prefixes rebuilt by prefill (decode-built state absent); re-tokenized ids (drift 0-8 tokens per reply); 'engine' bundles decode path, D2/D7 exposure, layout and speculation depth.
exit 0
