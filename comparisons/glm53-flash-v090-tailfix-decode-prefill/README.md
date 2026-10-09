# tpurtell 0.9.0 with and without the DCP1 tail fix: decode vs prefill

`3.25bpw · tpurtell 0.9.0 · no speculation` vs `3.25bpw · tpurtell 0.9.0 + DCP1 tail fix ≈ 0.9.1 · no speculation`, both with
prefix caching off. 6 GPQA prompts × 2,600 decoded tokens, each position compared with prefill re-scoring of the same
token ids (`protocols/decode-prefill-consistency/v1.md`).

| Positions | Mean KL, stock | Mean KL, with the fix | Top-1 agreement, stock | With the fix |
|---|---|---|---|---|
| i < 2,044 (11,169) | 0.0656 | 0.0103 | 93.8% | 97.6% |
| i ≥ 2,048 (4,407) | 0.0307 | 0.0187 | 96.1% | 97.1% |

Per prompt, mean KL below 2,044 is lower with the fix in 6 of 6 prompts. The predicted i mod 4 pattern did not
separate the two builds (see the investigation's gate record).

**Each build was measured once.** A later repeat of the same measurement on tpurtell 0.9.1 (which ships the fix),
speculation and prefix caching off, 12 prompts with the same seeds, run twice on 2026-10-08/09 (receipts not
published), varied by up to about 2x between runs: mean KL from 2,048 tokens 0.0053 in one run and 0.0098 in the
other, and single prompts by more; below 2,044 tokens 0.0072 and 0.0077. Only the large effect below 2,044 tokens
(0.066 → 0.010, more than six-fold, lower in 6 of 6 prompts) is supported. The smaller difference from 2,048 tokens
(0.031 → 0.019) is within the run-to-run spread and is not claimed.
