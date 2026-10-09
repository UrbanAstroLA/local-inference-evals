# 4bpw TR3 (Brandon): tpurtell 0.9.0 vs tpurtell 0.9.1 on GPQA Diamond (pass 1)

`4bpw TR3 (Brandon) · tpurtell 0.9.0 · DFlash2 ×3` vs `4bpw TR3 (Brandon) · tpurtell 0.9.1 · DFlash2 ×3`. Same weights,
serving settings, prompts and request seed; 0.9.1 is 0.9.0 plus the DCP1 tail fix
(tpurtell/glm-5.3-flash-ext3-2x-rtx#6).

| | tpurtell 0.9.0 | tpurtell 0.9.1 |
|---|---|---|
| Accuracy (pass 1) | 84.3% | 84.8% |
| Empty answers / 198 | 9 | 6 (all at the 327,680-token cap) |

Paired by question: 13 questions right only on 0.9.0, 14 only on 0.9.1 (exact McNemar p = 1.00; difference +0.5 points,
95% interval -4.5 to +5.6). Empty answers: 5 only on 0.9.0, 2 only on 0.9.1 (p = 0.45). One draw per question per
configuration: no detectable difference at this size. The 0.9.1 run was KV-limited at times (run notes). Recompute
with `tools/analyze.py gpqa-pairs k4-v0.9.0-dflash3 k4-v0.9.1-dflash3`.
