# Findings: GLM-5.3-Flash on 2x RTX PRO 6000 (2026-09 to 2026-10)

Analysis only; every number links to receipts in `runs/` and can be regenerated with `tools/analyze.py` and
`tools/verify.py`. Engines are tpurtell/glm-5.3-flash-ext3-2x-rtx releases; "patches" are the two upstream kpool
fixes ported in tpurtell/glm-5.3-flash-ext3-2x-rtx#5 and shipped in v0.9.0 ("0.8.0 + patches" = 0.9.0's kpool code,
byte for byte). Weights: 3.25bpw = wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1, 4bpw = brandonmusic/GLM-5.3-Flash-tr3-4bpw.

## GPQA Diamond, full protocol (3 passes, 594 answers; accuracy CI about +-4 points)
| Weights | Engine | Drafts | Accuracy | Empty answers (95% CI) |
|---|---|---|---|---|
| 3.25bpw | 0.7.0 | 5 | 85.5% | 6 (2-13) |
| 3.25bpw | 0.8.0 | 3 | 85.5% | 20 (12-31) |
| 3.25bpw | 0.8.0, draft-slot sharing off | 5 | 86.2% | 17 (10-27) |
| 4bpw | 0.8.0 | 3 | 85.0% | 15 (8-25) |

## GPQA Diamond, first pass only (198 answers, same seed and prompts as above)
| Weights | Engine | Drafts | Accuracy | Empty | Projected empty at 594 (95% CI) |
|---|---|---|---|---|---|
| 3.25bpw | 0.7.0 + patches | 5 | 85.9% | 2 | 6 (1-21) |
| 3.25bpw | 0.8.0 + patches | 5 | 86.9% | 3 | 9 (2-26) |

## Hard-question screen (5 hardest GPQA questions x 8 = 40 runs; failures = loops + exhaustions)
| Weights | Engine | Drafts | Failures / 40 |
|---|---|---|---|
| 3.25bpw | 0.7.0 | 5 | 12 (2026-09-30), 13 (2026-10-05) |
| 3.25bpw | 0.7.0 + patches | 5 | 13 |
| 3.25bpw | 0.8.0 | 3 | 24 (2026-09-30), 22 (2026-10-04) |
| 3.25bpw | 0.8.0 | 1 | 21 |
| 3.25bpw | 0.8.0 + patches | 3 | 20 |
| 4bpw | 0.7.0 + patches | 5 | not runnable: 437k-token KV pool; engine crashed when it filled |

## Inferences
1. **Accuracy does not depend on engine, patches, draft depth or quantization here.** Every configuration scores
   85-88%; every paired per-question comparison is consistent with noise (sign-test p >= 0.45).
2. **Finishing long reasoning depends on the engine.** On the same 3.25bpw weights, 0.7.0 fails to finish about half
   as often as 0.8.0 (screen 12-13 vs 20-24 of 40; GPQA 6 vs 20 empty, non-overlapping 95% intervals). Replicated
   across sessions.
3. **The kpool bugs do not cause the loops, but the fixes are worth having.** They correct real cache corruption
   (every prefill wrote 2 KB of keys into another block's indexer region; rejected drafts could overwrite committed
   keys), at no measurable cost. Their benefit is in long-lived servers with prefix caching and reused system prompts,
   where upstream showed the damage accumulating; fresh-server benchmarks like these rarely exercise that.
4. **Quantization shows no clear effect within an engine.** On 0.8.0, 4bpw gave 15 empty answers vs 20 for 3.25bpw,
   within noise. 4bpw cannot run on 0.7.0 at this concurrency: 13 GiB more weights per GPU leave 3.9 GiB of KV cache.
5. **The engines trade reliability for speed.** 0.7.0 decodes ~28 tok/s per request on long reasoning at 8
   concurrent, 0.8.0 ~47 tok/s; 0.8.0 fits 4bpw but abandons more hard problems.
6. **The gap to published scores (NVIDIA 92.1, Red Hat 90.6) is not a runtime problem.** Removing every empty answer
   would add 1-3 points; the remainder is quantization and/or harness, which these runs cannot separate.
7. **The engine is not bitwise reproducible at temperature 0** (identical configs diverge after ~80 tokens), so greedy
   parity cannot certify speculative exactness on this stack.

## Why this exists
Recipes for these models are widely shared and ported on the strength of speed numbers, with little end-to-end
correctness evidence. This work started from looping/no-answer reports on GPQA, found and fixed two real upstream
bugs, and showed with preregistered tests that they were not the main cause. The receipts are published so others can
check, rerun, and extend the comparisons on their own hardware.
