# 4bpw TR3 (Brandon): tpurtell 0.8.0 vs tpurtell 0.9.0 on GPQA Diamond (pass 1)

**Question.** Does 4bpw TR3 (Brandon) score or finish differently on tpurtell 0.9.0 (which includes the kpool fixes)
than on tpurtell 0.8.0? Pass 1 of each, paired by question.

**Grade.** Descriptive: no detectable difference at this size
([`FINDINGS.md`](../../FINDINGS.md#2-gpqa-diamond), item 9).

**Configurations.** `4bpw TR3 (Brandon) · tpurtell 0.8.0 · DFlash2 ×3` vs `4bpw TR3 (Brandon) · tpurtell 0.9.0 · DFlash2 ×3`.
Same weights (same revision), same serving settings, same prompts and request seed; only the engine release differs.
0.9.0 is 0.8.0 plus the kpool fixes (tpurtell/glm-5.3-flash-ext3-2x-rtx#5) and two changes that are off or inert at
defaults.

## Result

| | tpurtell 0.8.0 | tpurtell 0.9.0 |
|---|---|---|
| Accuracy (pass 1) | 85.9% | 84.3% |
| Empty answers / 198 | 3 | 9 |

Paired by question: 11 questions right only on 0.8.0, 8 only on 0.9.0 (exact McNemar p = 0.65; difference -1.5 points,
95% interval -6.1 to +3.0). Empty answers: 2 questions empty only on 0.8.0, 8 only on 0.9.0 (p = 0.11).

## Caveats

- One draw per question per configuration: no detectable difference, and the interval does not exclude a difference of
  several points either way. Passes of one configuration differ by 0.5 to 3.5 points
  ([`glm53-flash-gpqa-records`](../glm53-flash-gpqa-records)).
- Both runs sent 8 concurrent requests against a KV pool of about 1.38 million tokens, which holds about 4 requests at
  the token cap, and kept no server log, so whether requests waited for KV is not recorded.

## Recompute

```bash
python3 tools/analyze.py gpqa-pairs k4-v0.8.0-dflash3 k4-v0.9.0-dflash3
```
