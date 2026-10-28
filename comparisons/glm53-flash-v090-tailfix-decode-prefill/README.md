# tpurtell 0.9.0 with and without the DCP1 tail fix: decode vs prefill

**Question.** Does the DCP1 tail fix bring tpurtell 0.9.0's decode closer to prefill re-scoring of the same tokens?

**Grade.** Supported for the large effect below 2,044 tokens; from 2,048 tokens no effect is claimed
([`FINDINGS.md`](../../FINDINGS.md#1-engine-issues-found-during-evaluation-and-their-fixes), item 3).

**Configurations.** `3.25bpw · tpurtell 0.9.0 · no speculation, prefix cache off` vs
`3.25bpw · tpurtell 0.9.0 + DCP1 tail fix ≈ 0.9.1 · no speculation, prefix cache off`. 6 GPQA prompts × 2,600 decoded
tokens, each position compared with prefill re-scoring of the same token ids
(`protocols/decode-prefill-consistency/v1.md`).

This KL is the engine's agreement with itself: one model and one set of weights, its decode distribution at each
position compared with prefill re-reading the same tokens, over the top-20 tokens both return (each renormalised). It
is not a KL between a quantized model and its BF16 base, and it is not comparable with such numbers.

## Result

| Positions | Mean KL, stock | Mean KL, with the fix | Top-1 agreement, stock | With the fix |
|---|---|---|---|---|
| i < 2,044 (11,169) | 0.0656 | 0.0103 | 93.8% | 97.6% |
| i ≥ 2,048 (4,407) | 0.0307 | 0.0187 | 96.1% | 97.1% |

Per prompt, mean KL below 2,044 is lower with the fix in 6 of 6 prompts. The predicted i mod 4 pattern did not
separate the two builds (see the investigation's gate record).

### Tails

Per-position KL and |dlp| (absolute difference in the sampled token's log-probability), prompts 0-5 (the 0.9.1 runs
restricted to the same prompts, as the noise floor). p99 is the value 1% of positions exceed (type-7 quantiles).

| Positions | Run | Positions (KL defined) | KL p99 | KL p99.9 | \|dlp\| p99 | \|dlp\| p99.9 |
|---|---|---|---|---|---|---|
| i < 2,044 | stock | 11,169 | 1.025 | 3.737 | 2.061 | 10.915 |
|  | with the fix | 11,169 | 0.149 | 1.042 | 0.625 | 2.416 |
|  | 0.9.1 run 1 (floor) | 11,169 | 0.090 | 0.768 | 0.514 | 1.643 |
|  | 0.9.1 run 2 (floor) | 11,169 | 0.084 | 0.304 | 0.512 | 1.101 |
| i ≥ 2,048 | stock | 4,406 | 0.446 | 1.867 | 1.149 | 7.071 |
|  | with the fix | 4,407 | 0.312 | 2.042 | 0.843 | 4.344 |
|  | 0.9.1 run 1 (floor) | 4,407 | 0.095 | 0.323 | 0.500 | 1.606 |
|  | 0.9.1 run 2 (floor) | 4,407 | 0.179 | 0.674 | 0.708 | 1.908 |

## Caveats

- **Each build was measured once.** The noise floor comes from two runs of the 0.9.1 release (which ships the fix) with
  the same seeds ([`glm53-flash-v091-decode-prefill-repeat`](../glm53-flash-v091-decode-prefill-repeat)): on the same
  six prompts they give 0.0077 and 0.0060 below 2,044 tokens and 0.0064 and 0.0107 from 2,048.
- **Below 2,044 tokens** the run without the fix (0.0656) lies six to eleven times above all three runs with it
  (0.0060-0.0103): this large effect is supported.
- **From 2,048 tokens** the runs with the fix themselves range from 0.0064 to 0.0187, and there is one run without it
  (0.0307), so no effect is claimed there.

## Recompute

```bash
python3 tools/analyze.py decode-prefill
python3 tools/analyze.py decode-prefill-tails
python3 investigations/2026-10-glm53-looping/recompute.py
```
