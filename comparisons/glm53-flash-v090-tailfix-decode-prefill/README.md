# tpurtell 0.9.0 with and without the DCP1 tail fix: decode vs prefill

`3.25bpw · tpurtell 0.9.0 · no speculation` vs `3.25bpw · tpurtell 0.9.0 + DCP1 tail fix ≈ 0.9.1 · no speculation`, both with
prefix caching off. 6 GPQA prompts × 2,600 decoded tokens, each position compared with prefill re-scoring of the same
token ids (`protocols/decode-prefill-consistency/v1.md`).

| Positions | Mean KL, stock | Mean KL, with the fix | Top-1 agreement, stock | With the fix |
|---|---|---|---|---|
| i < 2,044 (11,169) | 0.0656 | 0.0103 | 93.8% | 97.6% |
| i ≥ 2,048 (4,407) | 0.0307 | 0.0187 | 96.1% | 97.1% |

Per prompt, mean KL below 2,044 is lower with the fix in 6 of 6 prompts, and from 2,048 in 5 of 6. The disagreement
persists past 2,048, where the fix changes no selection, because keys and values written earlier stay in the cache.
The predicted i mod 4 pattern did not separate the two builds (see the investigation's gate record).
