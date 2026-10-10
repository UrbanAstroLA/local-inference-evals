# tpurtell 0.9.0 with and without the DCP1 tail fix: decode vs prefill

**Question.** Does the DCP1 tail fix bring tpurtell 0.9.0's decode closer to prefill re-scoring of the same tokens?

**Grade.** Supported for the large effect below 2,044 tokens; from 2,048 tokens no effect is claimed
([`FINDINGS.md`](../../FINDINGS.md#1-runtime-defects-and-their-deterministic-evidence), item 3).

**Configurations.** `3.25bpw · tpurtell 0.9.0 · no speculation, prefix cache off` vs
`3.25bpw · tpurtell 0.9.0 + DCP1 tail fix ≈ 0.9.1 · no speculation, prefix cache off`. 6 GPQA prompts × 2,600 decoded
tokens, each position compared with prefill re-scoring of the same token ids
(`protocols/decode-prefill-consistency/v1.md`).

## Result

| Positions | Mean KL, stock | Mean KL, with the fix | Top-1 agreement, stock | With the fix |
|---|---|---|---|---|
| i < 2,044 (11,169) | 0.0656 | 0.0103 | 93.8% | 97.6% |
| i ≥ 2,048 (4,407) | 0.0307 | 0.0187 | 96.1% | 97.1% |

Per prompt, mean KL below 2,044 is lower with the fix in 6 of 6 prompts. The predicted i mod 4 pattern did not
separate the two builds (see the investigation's gate record).

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
python3 investigations/2026-10-glm53-looping/recompute.py
```
