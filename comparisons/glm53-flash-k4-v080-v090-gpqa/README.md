# 4bpw GLM-5.3-Flash: tpurtell 0.8.0 vs tpurtell 0.9.0 on GPQA Diamond

`4bpw · tpurtell 0.8.0 · 3 drafts` vs `4bpw · tpurtell 0.9.0 · 3 drafts`. Same weights (same revision), same serving
settings, same prompts and request seed; only the engine release differs. 0.9.0 is 0.8.0 plus the kpool fixes
(tpurtell/glm-5.3-flash-ext3-2x-rtx#5) and two changes that are off or inert at defaults.

| | tpurtell 0.8.0 | tpurtell 0.9.0 |
|---|---|---|
| Accuracy (3 passes) | 85.0% | 84.7% |
| Empty answers / 594 | 15 | 23 |
| Empty by pass | 3, 4, 8 | 9, 8, 6 |

Paired sign test on accuracy over all 594 answers: p = 0.90. Empty answers: Fisher exact p = 0.25; a bootstrap over
questions (each question's three passes kept together) gives a 95% interval of -3 to +19 for the difference. No
detectable difference in accuracy or completion, though the interval does not exclude a small completion cost. After two passes the gap looked larger (17 vs 7, p = 0.06); pass 3
reversed it, so treat two-pass empty counts as provisional. Recompute with `tools/analyze.py gpqa-pairs`.
