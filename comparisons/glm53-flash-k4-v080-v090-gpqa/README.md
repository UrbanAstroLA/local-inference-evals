# 4bpw GLM-5.3-Flash: tpurtell v0.8.0 vs v0.9.0 on GPQA Diamond

Same weights (same revision), same serving settings, same seeds and prompts; only the engine release differs. v0.9.0
is v0.8.0 plus the kpool fixes (tpurtell/glm-5.3-flash-ext3-2x-rtx#5) and two changes that are off or inert at
defaults.

| | v0.8.0 | v0.9.0 |
|---|---|---|
| Accuracy (3 passes) | 85.0% | 84.7% |
| Empty answers / 594 | 15 | 23 |
| Empty by pass | 3, 4, 8 | 9, 8, 6 |

Paired sign test on accuracy over all 594 answers: p = 0.90. Empty answers: Fisher exact p = 0.25; a bootstrap over
questions (each question's three passes kept together) gives a 95% interval of -3 to +19 for the difference. No
detectable difference in accuracy or completion. After two passes the gap looked larger (17 vs 7, p = 0.06); pass 3
reversed it, so treat two-pass empty counts as provisional.
