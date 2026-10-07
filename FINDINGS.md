# Findings: GLM-5.3-Flash on 2x RTX PRO 6000 (2026-09 to 2026-10)

Analysis only; every number links to receipts in `runs/` and can be regenerated with `tools/analyze.py` (subcommands
named below) and `tools/verify.py`.

## Labels
Configurations are named **weights · engine version · speculation**, built from the config files by one rule
([`SCHEMA.md`](SCHEMA.md#labels)). The engine name comes first because engines number their versions independently.

| Label | Meaning |
|---|---|
| **tpurtell 0.7.0** | Release v0.7.0 of [tpurtell/glm-5.3-flash-ext3-2x-rtx](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx), an engine built on vLLM, as published |
| **tpurtell 0.7.0 + kpool fixes** | v0.7.0 with the kpool fixes applied locally. A backport: not a release, and not 0.9.0 |
| **tpurtell 0.8.0** | Release v0.8.0 as published (no kpool fixes) |
| **tpurtell 0.8.0 + kpool fixes ≈ 0.9.0** | v0.8.0 with the kpool fixes applied locally. **The same engine as tpurtell 0.9.0 for every measurement here:** its kpool kernel files are byte-for-byte identical to the 0.9.0 release, and 0.9.0's other two changes (an opt-in boundary prefix-cache lookup, off by default, and usage reporting) do not affect these measurements. Measured before 0.9.0 was released |
| **tpurtell 0.9.0** | Release v0.9.0 as published, at its defaults (includes the kpool fixes) |
| kpool fixes | Upstream vLLM fixes vllm-project/vllm#57477 and #58454, ported in [tpurtell/glm-5.3-flash-ext3-2x-rtx#5](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/5) (commit 5a366b5) and shipped in v0.9.0 |
| 3.25bpw | [wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1) (EXL3, mixed K3/K4 routed experts) |
| 4bpw TR3 (Brandon) | Brandon M. Music's TR3 checkpoint [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) (EXL3, uniform K4 routed experts) |
| DFlash2 ×N | DFlash2 speculative decoding (incoai/GLM-5.3-Flash-DFlash2), N draft tokens per step; "no speculation" = plain decoding |

All engines here are tpurtell builds. 0.7.0 and the 0.8.0/0.9.0 line also differ in their default parallel layout
(0.7.0: EP2 + DCP2, vision on; 0.8.0 and 0.9.0: EP1 + DCP1, vision off), so "engine" below means the release as shipped.

## GPQA Diamond, full protocol (3 passes, 594 answers; accuracy 95% CI about ±4 points)
`analyze.py gpqa-empties` (exact 95% CI for the empty count)

| Weights | Engine | Speculation | Accuracy | Empty answers (95% CI) |
|---|---|---|---|---|
| 3.25bpw | tpurtell 0.7.0 | DFlash2 ×5 | 85.5% | 6 (2-13) |
| 3.25bpw | tpurtell 0.8.0 | DFlash2 ×3 | 85.5% | 20 (12-31) |
| 3.25bpw | tpurtell 0.8.0 | DFlash2 ×5, sharing off | 86.2% | 17 (10-27) |
| 4bpw TR3 (Brandon) | tpurtell 0.8.0 | DFlash2 ×3 | 85.0% | 15 (8-25) |
| 4bpw TR3 (Brandon) | tpurtell 0.9.0 | DFlash2 ×3 | 84.7% | 23 (15-34) |

**4bpw TR3 (Brandon), tpurtell 0.9.0 vs 0.8.0, same prompts and request seed (594 answers each).** The configs differ
only in engine version and image (checked by `tools/verify.py`, comparison `glm53-flash-k4-v080-v090-gpqa`; `analyze.py gpqa-pairs`).
Accuracy 84.7% vs 85.0% (paired sign test over all 594 answers, p = 0.90). Empty answers 23 vs 15 (Fisher exact
p = 0.25; resampling questions, 0.9.0 leaves -3 to +19 more). No detectable difference, but a small completion cost is
not excluded. The gap after two passes (17 vs 7) reversed in pass 3 (6 vs 8): two passes of a rare, clustered outcome
can mislead.

## GPQA Diamond, first pass only (198 answers, same prompts and request seed as above)
| Weights | Engine | Speculation | Accuracy | Empty | Projected empty at 594 (95% CI) |
|---|---|---|---|---|---|
| 3.25bpw | tpurtell 0.7.0 + kpool fixes | DFlash2 ×5 | 85.9% | 2 | 6 (1-21) |
| 3.25bpw | tpurtell 0.8.0 + kpool fixes ≈ 0.9.0 | DFlash2 ×5 | 86.9% | 3 | 9 (2-26) |

For comparison, pass 1 of the full runs left empty (`analyze.py gpqa-pass1`): 0 for `3.25bpw · tpurtell 0.7.0 · DFlash2 ×5`,
5 for `3.25bpw · tpurtell 0.8.0 · DFlash2 ×3`, 5 for `3.25bpw · tpurtell 0.8.0 · DFlash2 ×5, sharing off`,
3 for `4bpw TR3 (Brandon) · tpurtell 0.8.0 · DFlash2 ×3` and 9 for `4bpw TR3 (Brandon) · tpurtell 0.9.0 · DFlash2 ×3`.

## Hard-question screen (5 hardest GPQA questions x 8 = 40 runs; failures = loops + exhaustions)
Rows with the same date ran in one session. Protocol v0 has no early loop stop; v0 and v1 are not mixed in comparisons.

| Weights | Engine | Speculation | Protocol, date | Failures / 40 |
|---|---|---|---|---|
| 3.25bpw | tpurtell 0.7.0 | DFlash2 ×5 | v0, 2026-09-30 | 12 |
| 3.25bpw | tpurtell 0.8.0 | DFlash2 ×3 | v0, 2026-09-30 | 24 |
| 3.25bpw | tpurtell 0.8.0 | DFlash2 ×3 | v1, 2026-10-04 | 22 |
| 3.25bpw | tpurtell 0.8.0 | DFlash2 ×1 | v1, 2026-10-04 | 21 |
| 3.25bpw | tpurtell 0.8.0 + kpool fixes ≈ 0.9.0 | DFlash2 ×3 | v1, 2026-10-04 | 20 |
| 3.25bpw | tpurtell 0.7.0 | DFlash2 ×5 | v1, 2026-10-05 | 13 |
| 3.25bpw | tpurtell 0.7.0 + kpool fixes | DFlash2 ×5 | v1, 2026-10-05 | 13 |
| 4bpw TR3 (Brandon) | tpurtell 0.8.0 + kpool fixes ≈ 0.9.0 | DFlash2 ×3 | v1, 2026-10-05 | 15 |
| 4bpw TR3 (Brandon) | tpurtell 0.7.0 + kpool fixes | DFlash2 ×5 | v1, 2026-10-05 | not runnable: 437,563-token KV pool; engine crashed when it filled |

## Inferences
1. **Accuracy shows no dependence on engine, kpool fixes, draft depth or weights here.** Every configuration scores
   84.7-86.9% (3-pass runs 84.7-86.2%); every paired per-question comparison of first passes is consistent with noise
   (sign-test p >= 0.23), and so is every pair of full runs over all 594 answers (p >= 0.37).
2. **Finishing long reasoning depends on the engine.** On the same 3.25bpw weights, tpurtell 0.7.0 left about a third
   as many GPQA answers empty as tpurtell 0.8.0 (6 vs 20 of 594; Fisher exact p = 0.009; resampling questions,
   0.8.0 leaves 4 to 25 more), and failed about half as often on the hard-question screen (12 vs 24 of 40 in the same
   session; 13 vs 22 under protocol v1 a day apart). Draft depth alone does not explain it: 0.8.0 with 0.7.0's DFlash2 ×5
   and slot sharing off still left 17 (p = 0.033 against 6).
3. **The kpool bugs are real but are not the main cause of the loops.** The fixes correct real cache corruption (every
   prefill wrote 2 KB of keys into another block's indexer region; rejected drafts could overwrite committed keys) and
   make upstream's regression tests pass (33/33, from 29/33). Accuracy: no measurable cost. Completion: no detectable
   change on either weight set, though these data cannot exclude a small one either way. On 4bpw TR3 (Brandon), 0.9.0 left
   23 vs 15 of 594 answers empty against unpatched 0.8.0 (p = 0.25; -3 to +19). On 3.25bpw the screen moved from 22 to 20 of 40
   on 0.8.0 and stayed at 13 vs 13 on 0.7.0, each within one session, and neither met its preregistered rule. Their
   expected benefit is in long-lived servers with prefix caching and reused system prompts, where cached blocks are
   reused across requests; fresh-server benchmarks like these rarely exercise that, and this repository has no
   measurement of it.
4. **Quantization shows no clear effect within an engine.** On tpurtell 0.8.0, 4bpw TR3 (Brandon) left 15 of 594 answers empty
   vs 20 for 3.25bpw (p = 0.49), at 85.0% vs 85.5% accuracy. 4bpw TR3 (Brandon) cannot run on tpurtell 0.7.0's layout
   at this concurrency: the weights leave a 437,563-token KV pool (3.9 GiB per GPU vs 15.8 GiB with 3.25bpw), and the engine
   crashed when it filled.
5. **The engines trade completion for speed.** In the screens (8 concurrent), requests that finished ran at a median
   27-28 completion tokens per second on tpurtell 0.7.0 vs 47-49 on 0.8.0 with DFlash2 ×3, with or without the kpool
   fixes (`analyze.py screen-speed`). 0.8.0 also fits 4bpw TR3 (Brandon); it leaves more long reasoning unfinished.
6. **Empty answers explain only part of the gap to published scores (NVIDIA 92.1, Red Hat 90.6).** Scoring only
   answered questions would add 0.9-3.4 points (86.4-88.7% across the full runs), still below both. The rest mixes
   quantization, harness and other runtime effects, which these runs cannot separate.
7. **The engine is not bitwise reproducible at temperature 0.** The same configuration run twice diverges after a
   median of 318 characters (`comparisons/glm53-flash-serving-probe`), so greedy parity cannot certify speculative
   exactness on this stack.

## Reading these results
- **Concurrency changes the arithmetic.** Requests are served 8 at a time, and batch composition changes the numerics
  inside the engine. A fixed seed therefore does not reproduce a response: in the screens, eight repeats of the same
  question with the same seed (1234) produced eight different outputs that diverged within the first 0-266 characters.
  Even one greedy request at a time diverges from its own rerun after a median of 318 characters on this stack.
  Compare distributions (accuracy, failure rates), never individual transcripts.
- **Scores reproduce statistically, transcripts do not.** GPQA at temperature 1.0 is a fresh random draw each pass;
  within each of the five full runs the three passes differ by 1.0-3.5 points, and all 15 pass scores fall inside their
  run's 95% interval (`analyze.py gpqa-passes`). A rerun should land inside the interval, not on the same number. Match
  the concurrency (8 requests) as well as the sampling settings: different batch sizes take different numerical paths,
  and whether that shifts accuracy systematically is untested. The analysis itself is exactly reproducible:
  `tools/verify.py` and `tools/analyze.py` recompute every number from rows.
- **The screen's question set is not neutral.** Its five questions were chosen from empty answers in the tpurtell
  0.8.0 GPQA runs on both 3.25bpw and 4bpw TR3 (Brandon); in tpurtell 0.7.0's full GPQA run only one of them (q88)
  ever came back empty.
  Within-engine comparisons on the screen are fair; the size of the cross-engine gap on the screen is an upper-end
  estimate. The engine finding rests on the full GPQA runs.
- **Screen intervals assume independent runs.** Outcomes cluster by question (q121 failed once in 64 repeats across
  the eight valid screens; q88 failed in 4-8 of 8 repeats in every one), so the Wilson intervals in screen summaries
  are too narrow. Decision rules were count-based, not interval-based.
- **Intervals.** GPQA accuracy: 95% bootstrap over questions. Empty answers: exact (Clopper-Pearson) 95% intervals
  here, Wilson score intervals on the site; they differ slightly.

## Background
This work started from looping and no-answer reports on GPQA. It found two vLLM kpool bugs, already fixed upstream,
still present in these engine images; the fixes were ported (now part of tpurtell 0.9.0), and preregistered tests showed
that the bugs were not the main cause. Receipts, protocols and tools are here so the comparisons can be checked,
rerun and extended on other hardware.
