# B+ continuation result: C1-ONLY

- stock 0.7.0: complete
- stock 0.9.1: complete
- B1 (0.9.1 + D1-D4 + D7): budget stop (not started)

**Reading: R-E.** From identical 100k-token states, stock 0.7.0's continuation finished within +64k more often than stock 0.9.1's, by at least the prespecified 0.15 with the prefix-level interval and test excluding no difference. The engines' continuations differ; this does not identify which component of 'engine' (decode path, D2/D7, layout, speculation depth) is responsible, and holds for 0.7.0-built states only.

## Primary: P(finish within +64k), prefix as the unit

| engine | P | finished / 48 | capped-origin | finished-origin |
|---|---:|---:|---:|---:|
| stock 0.7.0 | 0.750 | 36 | 0.792 | 0.708 |
| stock 0.9.1 | 0.188 | 9 | 0.042 | 0.333 |

| contrast | mean | 95% interval | 90% interval | sign-flip p | prefixes + / - / 0 | screen |
|---|---:|---|---|---:|---|---|
| C1  | +0.562 | +0.333 to +0.750 | +0.375 to +0.729 | 0.0029 | 11 / 1 / 0 | EFFECT |

## Secondary (descriptive)

| engine | any stop | median finish tokens | loop onsets in continuation (median after cut) | loop types (short / long / fragmentary / none) | cpt | unanimous prefixes | ICC | decode tok/s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| stock 0.7.0 | 0.750 | 30983.5 | 6 (21700.5) | 6 / 0 / 0 / 42 | 3.0 | 3 / 12 | -0.017 | 43.56 |
| stock 0.9.1 | 0.188 | 25047 | 15 (18047) | 15 / 0 / 0 / 33 | 3.0 | 6 / 12 | 0.161 | 43.47 |

Covariates at 100k vs pooled prefix rate (Spearman, n = 12, descriptive): {'chars_per_token': 0.226, 'interjection_per1k': -0.058, 'digit_share': -0.102, 'R50_backward_at_cut': -0.045}

Caveats: one 0.7.0-built state per question; prefixes rebuilt by prefill (decode-built state absent); re-tokenized ids (drift 0-8 tokens per reply); 'engine' bundles decode path, D2/D7 exposure, layout and speculation depth.
exit 5
