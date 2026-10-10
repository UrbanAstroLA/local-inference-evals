# Findings: GLM-5.3-Flash on 2x RTX PRO 6000 (2026-09 to 2026-10)

Analysis only; every number links to receipts in `runs/` and can be regenerated with `tools/analyze.py` (subcommands
named below), `tools/verify.py` and the investigation's `recompute.py`. Each statement says how strong it is:
**supported** (deterministic, or statistically clear), **descriptive** (what the data shows, without a test that
separates it from chance), **unmeasured**, or **open**. Terms (kpool, DCP1, MLA ownership, NOPE record, DFlash2,
flexible-extract, loop, exhaustion and others) are defined in the [glossary](README.md#glossary).

## Summary

Each row is the headline of the numbered statement below, with its grade; the statement itself carries the numbers,
qualifications and receipts. Open questions are [at the end](#open-questions). Charts of the same results:
[results site](https://urbanastrola.github.io/local-inference-evals/).

| # | Statement | Grade | Where |
|---|---|---|---|
| 1 | Two upstream kpool bugs were present in the tpurtell 0.7.0 and 0.8.0 images, and the fixes remove them. | **Supported** | [§1](#1-runtime-defects-and-their-deterministic-evidence) |
| 2 | Under the DCP1 layout of tpurtell 0.8.0 and 0.9.0, decode attention skipped the newest 1-3 tokens at causal lengths up to 2,043 that are not a multiple of 4; the DCP1 tail fix removes it. | **Supported** | [§1](#1-runtime-defects-and-their-deterministic-evidence) |
| 3 | With the fix, decode agrees much better with prefill re-scoring of the same tokens below 2,044 tokens. | **Supported** (for this large effect) | [§1](#1-runtime-defects-and-their-deterministic-evidence) |
| 4 | The fix's effect on answers is unmeasured. | **Descriptive** | [§1](#1-runtime-defects-and-their-deterministic-evidence) |
| 5 | One pass of one configuration varies by 0.5 to 3.5 points between passes with their own request seeds. | **Descriptive** | [§2a](#2a-three-passes-per-configuration) |
| 6 | 3.25bpw and 4bpw TR3 (Brandon) on tpurtell 0.9.1 show no measurable difference in accuracy or in completion. | **Descriptive** | [§2a](#2a-three-passes-per-configuration) |
| 7 | tpurtell 0.7.0 as shipped left fewer GPQA questions unanswered than tpurtell 0.9.1 (3.25bpw, three passes each). What produces the difference is open. | **Supported** | [§2a](#2a-three-passes-per-configuration) |
| 8 | Empty answers concentrate on a few questions. | **Descriptive** | [§2a](#2a-three-passes-per-configuration) |
| 9 | One pass per configuration does not separate these configurations in accuracy. | **Descriptive** | [§2b](#2b-pass-1-of-every-configuration) |
| 10 | Empty answers in pass 1 are 0 to 9 of 198 per run. | **Descriptive** | [§2b](#2b-pass-1-of-every-configuration) |
| 11 | Scoring limitation: the raw `flexible-extract` score reads some correct answers as wrong. | not graded (scoring note) | [§2b](#2b-pass-1-of-every-configuration) |
| 12 | Published scores, for context only. | not graded (context only) | [§2b](#2b-pass-1-of-every-configuration) |
| 13 | On tpurtell 0.9.1, question 88 fails to finish in 1-3 of 12 draws in every tested arm; question 79 in 9 of 12 on both tpurtell 0.9.1 and the tpurtell 0.7.0 image at three draft tokens. | **Descriptive** | [§3](#3-non-completion-on-hard-questions) |
| 14 | None of the tested runtime parts moved either question at this size, and only very large effects could have shown. | **Descriptive** | [§3](#3-non-completion-on-hard-questions) |
| 15 | Repeats that share one request seed vary only through batching, which is not a meaningful sample of how often a question fails. | **Supported** | [§3](#3-non-completion-on-hard-questions) |
| 16 | Loops are stopped far beyond the 2,044-token region where the tail bug acted. | **Descriptive** | [§3](#3-non-completion-on-hard-questions) |
| 17 | KV capacity depends on the layout; tpurtell's default DCP1 layout with MLA layer ownership holds the most. | **Supported** | [§4](#4-serving-facts) |
| 18 | Neither fix shows a speed cost, in single runs. | **Descriptive** | [§4](#4-serving-facts) |
| 19 | The engine is not bitwise reproducible, even one request at a time. | **Supported** | [§4](#4-serving-facts) |

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
| **tpurtell 0.9.0 + DCP1 tail fix ≈ 0.9.1** | v0.9.0 with the fix of [tpurtell/glm-5.3-flash-ext3-2x-rtx#6](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/6) applied locally, measured before the fix was merged. **The same engine as tpurtell 0.9.1 for every measurement here:** 0.9.1 (released 2026-10-08 from that merge) ships the fixed attention file byte-for-byte as built from the pull request, which differs from the build measured here only in identifier names; its other kpool and indexer files are those of 0.9.0 |
| **tpurtell 0.9.1** | Release v0.9.1 as published: v0.9.0 plus the DCP1 tail fix |
| Layout segments | Shown only when a configuration runs a parallel layout other than its release's default: `0.7.0 layout (DCP2, EP2)` on the 0.9.0 image, and the component-screen arms `EP2 experts, NOPE records off`, `EP2 experts, MLA owners tp` and `NOPE records off, MLA owners tp` on 0.9.1. Screen arms and diagnostic controls, not recommendations |
| concurrency N | Shown only when the client kept N requests in flight instead of the protocol's setting (GPQA: 8): `4bpw TR3 (Brandon) · tpurtell 0.9.1 · DFlash2 ×3 · concurrency 4`, because that KV pool holds about 4 requests at the token cap |
| kpool fixes | Upstream vLLM fixes vllm-project/vllm#57477 and #58454, ported in [tpurtell/glm-5.3-flash-ext3-2x-rtx#5](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/5) (commit 5a366b5) and shipped in v0.9.0 |
| 3.25bpw | tpurtell's K3.25 checkpoint [wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1) (EXL3, mixed K3/K4 routed experts) |
| 4bpw TR3 (Brandon) | Brandon M. Music's TR3 checkpoint [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) (EXL3, uniform K4 routed experts) |
| DFlash2 ×N | DFlash2 speculative decoding (incoai/GLM-5.3-Flash-DFlash2), N draft tokens per step; "no speculation" = plain decoding |

All engines here are tpurtell builds. 0.7.0 and the 0.8.0-0.9.1 line also differ in their default parallel layout
(0.7.0: EP2 + DCP2, vision on; 0.8.0 onward: EP1 + DCP1 with MLA layer ownership, vision off), so "engine" below means
the release as shipped.

## 1. Runtime defects and their deterministic evidence
Charts: [kernel tests](https://urbanastrola.github.io/local-inference-evals/kernels.html), [the DCP1 tail bug](https://urbanastrola.github.io/local-inference-evals/looping.html#tail-bug).

1. **Two upstream kpool bugs were present in the tpurtell 0.7.0 and 0.8.0 images, and the fixes remove them.**
   **Supported.** Upstream vLLM's own regression tests for the kpool kernels pass 29 of 33 on both images and 33 of 33
   with the fixes (vllm-project/vllm#57477: every prefill wrote 2 KB of keys into another block's indexer region;
   #58454: a rejected pool-completing draft could overwrite committed keys at 2 or more draft tokens). Ported in
   tpurtell PR #5 and shipped in tpurtell 0.9.0 (`runs/*_kpool-kernel-tests`;
   [`investigations/2026-10-glm53-kpool-tail`](investigations/2026-10-glm53-kpool-tail)). The tests ran on the 0.7.0 and
   0.8.0 release images, on local builds with the fixes, and (2026-10-09) on the 0.9.0 and 0.9.1 release images, which pass
   33 of 33 and reproduce no rejected-draft corruption at 2, 3, 5 or 7 draft tokens.
2. **Under the DCP1 layout of tpurtell 0.8.0 and 0.9.0, decode attention skipped the newest 1-3 tokens at causal lengths
   up to 2,043 that are not a multiple of 4; the DCP1 tail fix removes it.** **Supported** (index check on the image's
   own kernels: the tail is dropped in 9 of 23 packed cases without the fix, 0 with it; rows the stock code already
   handled are byte-identical). DCP1 with MLA layer ownership is tpurtell's layout choice and frees KV memory (item 17);
   the masking path already existed in the vendored attention code, and only this layout exercises it. The fix was merged
   as tpurtell PR #6 and released in tpurtell 0.9.1
   ([`investigations/2026-10-glm53-looping`](investigations/2026-10-glm53-looping)).
3. **With the fix, decode agrees much better with prefill re-scoring of the same tokens below 2,044 tokens.**
   **Supported for this large effect.** On the six prompts every run shares, mean KL is 0.066 without the fix (one run)
   against 0.010 with the local build of the fix and 0.008 and 0.006 in two runs of the 0.9.1 release; top-1 agreement
   93.8% → 97.6%; lower in 6 of 6 prompts. The two 0.9.1 runs, with the same prompts and seeds, are the noise floor:
   their means differ by up to about 2x and single prompts by up to about 6x. From 2,048 tokens the runs with the fix
   range from 0.006 to 0.019 and the one run without it is 0.031, so no effect is claimed there
   (`comparisons/glm53-flash-v090-tailfix-decode-prefill`, `glm53-flash-v091-decode-prefill-repeat`). With speculation
   on (DFlash2 ×3, one run), 0.9.1 decode sits somewhat above both speculation-off runs, within that spread
   (`glm53-flash-v091-decode-prefill-speculation`; descriptive).
4. **The fix's effect on answers is unmeasured.** In GPQA, 4bpw TR3 (Brandon) scored 84.8% on tpurtell 0.9.1 against
   84.3% on 0.9.0, one pass each at 8 concurrent requests with a KV pool that holds about 4 requests at the token cap,
   well inside the 0.5-3.5 points by which passes of one configuration differ (item 5). In tool-eval-bench, scenarios
   TC-80 and TC-88 pass in both repeats with the fix and fail in both without it (157 and 157 of 176 points → 159 and 163), but that is two repeats per
   build, and ten other scenarios flip between repeats of the same build
   (`comparisons/glm53-flash-v090-tailfix-tool-eval`). **Descriptive.**

## 2. GPQA Diamond
Charts: [GPQA Diamond](https://urbanastrola.github.io/local-inference-evals/gpqa.html).
198 questions, the same prompts and answer order in every run; pass *p* sends request seed 1233 + *p*. Raw =
`flexible-extract`, the headline (protocol v1); stated = `correct_stated`, the secondary, audited stated-answer score;
"answered" leaves out empty answers. 8 concurrent requests unless the label says otherwise.

### 2a. Three passes per configuration
`analyze.py gpqa-passes`, `gpqa-records`, `gpqa-empty`; comparison
[`glm53-flash-gpqa-records`](comparisons/glm53-flash-gpqa-records). Three configurations have three passes with request
seeds 1234, 1235 and 1236. Intervals resample questions with all their passes.

| Configuration | Raw, passes 1 / 2 / 3 | Raw, mean (95% interval) | Spread | Stated, mean | Empty, passes 1 / 2 / 3 | Questions ever empty |
|---|---|---|---|---|---|---|
| `3.25bpw · tpurtell 0.7.0 · DFlash2 ×5` | 87.9 / 88.4 / 88.4% | 88.2% (84.5-91.6) | 0.5 points | 89.4% | 0 / 3 / 1 | 4 |
| `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3` | 85.4 / 88.9 / 86.4% | 86.9% (83.0-90.4) | 3.5 points | 87.9% | 5 / 5 / 6 | 11 |
| `4bpw TR3 (Brandon) · tpurtell 0.9.1 · DFlash2 ×3 · concurrency 4` | 84.8 / 86.4 / 85.9% | 85.7% (81.5-89.6) | 1.5 points | 87.7% | 8 / 2 / 7 | 14 |

Comparisons, paired by question and pass. A question answered in three passes is one unit, not three: "questions" counts
the questions on which A had more such outcomes than B over the passes, and fewer; the clustered p is an exact sign-flip
test over questions; the interval is a bootstrap over questions. Pooled p values (exact McNemar on question-passes)
treat a question's passes as independent and are given for reference only. No correction for multiple comparisons.

| A vs B | What differs | Raw: B - A, points (95% interval); questions A / B; clustered p | Empty: A vs B of 594; questions A / B; clustered p | Pooled McNemar p (raw; empty) |
|---|---|---|---|---|
| 3.25bpw 0.7.0 ×5 vs 0.9.1 ×3 | release as shipped: draft depth, layout, kernels, vision and KV pool together | -1.3 (-4.2 to +1.5); 23 / 19; 0.41 | 4 vs 16; 1 / 10; 0.009 | 0.39; 0.002 |
| 3.25bpw vs 4bpw on 0.9.1 | weights, and 8 vs 4 concurrent requests | -1.2 (-4.2 to +1.7); 19 / 16; 0.51 | 16 vs 17; 9 / 9; 1.00 | 0.47; 1.00 |
| 3.25bpw 0.7.0 ×5 vs 4bpw 0.9.1 ×3 | all of the above | -2.5 (-5.4 to +0.5); 25 / 14; 0.12 | 4 vs 17; 2 / 13; 0.006 | 0.07; 0.004 |

5. **One pass of one configuration varies by 0.5 to 3.5 points between passes with their own request seeds.**
   **Descriptive** (three configurations, three passes each). That is the measured noise of a single pass, and it is as
   large as the spread of pass 1 across all ten configurations below (84.3-87.9%, 3.6 points). Of the 198 questions,
   154-160 are answered correctly in all three passes of a configuration, 10-12 in none, and 28-32 vary between passes.
6. **3.25bpw and 4bpw TR3 (Brandon) on tpurtell 0.9.1 show no measurable difference in accuracy or in completion.**
   **Descriptive** (no test separates them): over three passes each, raw accuracy 86.9% vs 85.7% (B - A -1.2 points,
   -4.2 to +1.7; clustered p = 0.51), stated 87.9% vs 87.7%, empty answers 16 vs 17 of 594 (9 questions each way;
   clustered p = 1.00). The 4bpw record ran 4 requests at once, the 3.25bpw record 8; with three passes, differences
   smaller than about 3 points in accuracy are neither shown nor excluded.
7. **tpurtell 0.7.0 as shipped left fewer GPQA questions unanswered than tpurtell 0.9.1 (3.25bpw, three passes each).**
   **Supported** (question-clustered test): 4 vs 16 empty answers of 594, on 4 vs 11 questions; 0.9.1 had more empty
   answers on 10 questions and fewer on 1 (exact sign-flip test over questions p = 0.009; difference +2.0 points, 95%
   interval over questions +0.7 to +3.5). Qualifications: the question was raised by pass 1 (0 vs 5); passes 2 and 3
   alone, run after it was raised, point the same way (4 vs 11 empty answers on 1 vs 7 questions, clustered p = 0.06). Three record
   comparisons were made without correction (with a Bonferroni correction over three, p = 0.03). Accuracy does not differ
   measurably (88.2% vs 86.9%, clustered p = 0.41). Where a request log exists, every one of these empty answers ran to the
   327,680-token cap except one, which ended after 38 tokens. **What produces the difference is open**: 0.7.0 differs from
   0.9.1 in draft depth (5 vs 3 tokens), parallel layout (DCP2 with EP2 experts vs DCP1 with MLA layer ownership), kernel
   and engine code, vision (on vs off), and KV pool (2,758,919 vs 4,707,515 tokens) at once (the model revisions differ only in files neither server uses: same weight
   files, and both servers load the same vendored chat template);
   none of these was varied alone in GPQA, and the component screen at three draft tokens found no tested part that moved
   questions 88 or 79 (item 14). The 4bpw TR3 (Brandon) record on 0.9.1 also left more empty answers than 0.7.0 (17, on 14
   questions; clustered p = 0.006), but it differs in weights and concurrency as well.
8. **Empty answers concentrate on a few questions.** **Descriptive.** Questions 79 and 81 came back empty in all three
   records; questions 88, 127 and 147 in both 0.9.1 records and in no pass of 0.7.0; question 88 in two of three passes
   of each 0.9.1 record (`analyze.py gpqa-empty`; questions are doc ids, the questions themselves are not published).

### 2b. Pass 1 of every configuration
`analyze.py gpqa-table`. Pass 1 (request seed 1234) is the pass every configuration has, so the configurations share
that seed and are compared question by question.

| Weights | Engine | Speculation, concurrency | Raw (95% interval) | Raw, answered | Stated | Stated, answered | Empty |
|---|---|---|---|---|---|---|---|
| 3.25bpw | tpurtell 0.7.0 | DFlash2 ×5 | 87.9% (82.8-91.9) | 87.9% | 89.4% | 89.4% | 0 |
| 3.25bpw | tpurtell 0.7.0 + kpool fixes | DFlash2 ×5 | 85.9% (80.8-90.4) | 86.7% | 87.4% | 88.3% | 2 |
| 3.25bpw | tpurtell 0.8.0 | DFlash2 ×3 | 85.9% (80.8-90.4) | 88.1% | 86.9% | 89.1% | 5 |
| 3.25bpw | tpurtell 0.8.0 | DFlash2 ×5, sharing off | 86.9% (81.8-91.4) | 89.1% | 88.4% | 90.7% | 5 |
| 3.25bpw | tpurtell 0.8.0 + kpool fixes ≈ 0.9.0 | DFlash2 ×5 | 86.9% (82.3-91.4) | 88.2% | 88.4% | 89.7% | 3 |
| 3.25bpw | tpurtell 0.9.1 | DFlash2 ×3 | 85.4% (80.3-89.9) | 87.6% | 85.9% | 88.1% | 5 |
| 4bpw TR3 (Brandon) | tpurtell 0.8.0 | DFlash2 ×3 (KV pool holds about 4 at the cap; not logged) | 85.9% (80.8-90.4) | 87.2% | 87.9% | 89.2% | 3 |
| 4bpw TR3 (Brandon) | tpurtell 0.9.0 | DFlash2 ×3 (KV pool holds about 4 at the cap; not logged) | 84.3% (78.8-88.9) | 88.4% | 85.4% | 89.4% | 9 |
| 4bpw TR3 (Brandon) | tpurtell 0.9.1 | DFlash2 ×3 (**KV-saturated**: requests waited for KV, logged) | 84.8% (79.8-89.4) | 87.5% | 86.9% | 89.6% | 6 |
| 4bpw TR3 (Brandon) | tpurtell 0.9.1 | DFlash2 ×3 · concurrency 4 | 84.8% (79.8-89.4) | 88.4% | 86.9% | 90.5% | 8 |

The 4bpw TR3 (Brandon) runs at 8 concurrent requests are kept and labelled: these weights leave a KV pool of about 1.38
million tokens, which holds about 4 requests at the 327,680-token cap. The 0.9.1 run's server log shows requests waiting
for KV in 653 of 1,417 ten-second status lines; the 0.8.0 and 0.9.0 runs kept no server log. The 0.9.1 record at 4
concurrent requests (2a), where no request waited, replaces that run as the configuration's record.

Question-paired comparisons of pass 1 (`analyze.py gpqa-pairs`; exact McNemar tests on the questions where two runs
disagree; no correction for multiple comparisons):

| A vs B | What differs | Only A right / only B right (p) | B - A, points (95% interval) | Only A empty / only B empty (p) |
|---|---|---|---|---|
| 3.25bpw 0.7.0 ×5 vs 0.8.0 ×3 | engine release as shipped | 10 / 6 (0.45) | -2.0 (-6.1 to +2.0) | 0 / 5 (0.06) |
| 3.25bpw 0.8.0 ×3 vs ×5, sharing off | draft depth, slot sharing | 10 / 12 (0.83) | +1.0 (-3.5 to +5.6) | 3 / 3 (1.00) |
| 3.25bpw 0.7.0 vs 0.7.0 + kpool fixes | kpool fixes | 10 / 6 (0.45) | -2.0 (-6.1 to +2.0) | 0 / 2 (0.50) |
| 4bpw 0.8.0 vs 0.9.0 | kpool fixes (release) | 11 / 8 (0.65) | -1.5 (-6.1 to +3.0) | 2 / 8 (0.11) |
| 4bpw 0.9.0 vs 0.9.1 | DCP1 tail fix (release) | 13 / 14 (1.00) | +0.5 (-4.5 to +5.6) | 5 / 2 (0.45) |
| 3.25bpw vs 4bpw on 0.8.0 | weights | 10 / 10 (1.00) | 0.0 (-4.5 to +4.5) | 4 / 2 (0.69) |
| 3.25bpw vs 4bpw on 0.9.1 | weights (4bpw KV-saturated) | 13 / 12 (1.00) | -0.5 (-5.6 to +4.5) | 4 / 5 (1.00) |
| 4bpw 0.9.1, 8 vs 4 concurrent | concurrency (KV-saturated vs not) | 13 / 13 (1.00) | 0.0 (-5.1 to +5.1) | 3 / 5 (0.73) |

9. **One pass per configuration does not separate these configurations in accuracy.** **Descriptive.** Every pass 1
   lands at 84.3-87.9% raw (85.4-89.4% stated), within the 0.5-3.5 points by which passes of one configuration differ
   (item 5), and every paired difference is consistent with noise; each comparison resolves only differences of about 5
   points, so smaller effects are neither shown nor excluded.
10. **Empty answers in pass 1 are 0 to 9 of 198 per run.** **Descriptive.** No paired pass-1 comparison of empty answers
    reaches p < 0.05; one pass per configuration does not separate releases (the three-pass records do for 0.7.0 vs 0.9.1,
    item 7). Of the 43 published empty answers with a request log (four runs), 42 ran to the 327,680-token cap and one
    ended after 38 tokens (`finish_reason` stop).
11. **Scoring limitation.** The raw `flexible-extract` score reads some correct answers as wrong when a reply mentions
    other options' labels or chemistry notation after its answer, or states its answer as a boxed or bold letter: over the
    16 published passes, 49 correct answers are scored wrong and 3 replies whose stated answer is not the target are
    credited, a net 0.5 to 3.5 points per pass (1.5 on average). The audited `correct_stated` score is published per row
    beside it (`protocols/gpqa-diamond/v1.md`); raw scores stay the headline so runs remain comparable.
12. **Published scores, for context only.** On the raw score, NVIDIA's 92.1 (BF16 and NVFP4) lies above every local
    run's interval; Red Hat's 90.6 (NVFP4) lies inside the intervals of three runs, all at DFlash2 ×5 (the three-pass
    tpurtell 0.7.0 record among them), and above the others. On the stated-answer score, 90.6 lies inside nine runs'
    intervals and 92.1 inside three. Those numbers used other weights and harnesses whose details are not fully published;
    no higher-precision reference was run here, so these runs cannot separate quantization, harness, scoring and runtime
    effects (`comparisons/glm53-flash-gpqa-published-context`).

## 3. Non-completion on hard questions
Charts: [hard-question screen](https://urbanastrola.github.io/local-inference-evals/screens.html), [looping investigation](https://urbanastrola.github.io/local-inference-evals/looping.html#non-completion).
Details, failure anatomy and open questions: [`investigations/2026-10-glm53-looping`](investigations/2026-10-glm53-looping).
`analyze.py screen-v2`.

13. **On tpurtell 0.9.1, question 88 fails to finish in 1-3 of 12 draws in every tested arm; question 79 in 9 of 12 on
   both tpurtell 0.9.1 and the tpurtell 0.7.0 image at three draft tokens.** **Descriptive** (component screen,
   2026-10-09, preregistered: one question per run, distinct request seeds 5001-5012). Held fixed in every arm: 3.25bpw
   weights, DFlash2 ×3, temperature 1.0 / top_p 0.95, the 327,680-token budget, 12 concurrent requests, one question per
   arm, a fresh server. Both questions were chosen as hard from GPQA runs and screens that all sent request seed 1234; with
   distinct seeds question 88 fails far less often than those screens suggested. Two questions are not a benchmark-wide
   rate.

   | Arm | Configuration | Question | Failed / 12 (95% Wilson) |
   |---|---|---|---|
   | B | `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3` | 88 | 3 (9-53%) |
   | EN | `3.25bpw · tpurtell 0.9.1 · EP2 experts, NOPE records off · DFlash2 ×3` | 88 | 2 (5-45%) |
   | EO | `3.25bpw · tpurtell 0.9.1 · EP2 experts, MLA owners tp · DFlash2 ×3, sharing off` | 88 | 1 (1-35%) |
   | NO | `3.25bpw · tpurtell 0.9.1 · NOPE records off, MLA owners tp · DFlash2 ×3, sharing off` | 88 | 2 (5-45%) |
   | B79 | `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3` | 79 | 9 (47-91%) |
   | V79 | `3.25bpw · tpurtell 0.7.0 · DFlash2 ×3` | 79 | 9 (47-91%) |
14. **None of the tested runtime parts moved either question at this size, and only very large effects could have
    shown.** **Descriptive.** Question 88: EP2 routed experts 3 vs 5 failures of 24 (Fisher p = 0.70), NOPE records off
    4 vs 4 (p = 1.00), MLA ownership tp 3 vs 5 (p = 0.70); arms EO and NO also turned draft-slot sharing off, so ownership
    and sharing are not separated; the preregistered rule found no candidate. Question 79: 9 vs 9; the rule's verdict is
    "unresolved". Minimum detectable differences (two-sided Fisher, p < 0.05, 80% power; `analyze.py screen-power`): one
    arm against another on question 88 (12 vs 12, from 25%) only a rise of about 60 points, and no drop at any size; a
    switch on vs off (24 vs 24, from 17%) a rise of about 41 points; question 79 (from 75%) a drop of about 60 points.
    Draft depth, quantization and sampling settings were not varied.
15. **Repeats that share one request seed vary only through batching, which is not a meaningful sample of how often a question fails.** **Supported.** Same configuration and question
    (88), 12 repeats each: 3 failures with distinct seeds, 11 with seed 1234 on every repeat (Fisher p = 0.003;
    `comparisons/glm53-flash-fixed-seed-control`). The earlier hard-question screens sent seed 1234 on every repeat; their
    rates, the layout-bisection statistics and the question-level observations drawn from them were withdrawn on
    2026-10-09 (notice in the investigation). Repeat 1 of each question from each configuration's first screen is kept
    as a single draw (`analyze.py screen-single`): loops or exhaustion occur in every configuration tested.
16. **Loops are stopped far beyond the 2,044-token region where the tail bug acted.** **Descriptive.** In the component
    screen the early-stop detector fired at an estimated 106,000-187,000 tokens; question 88 fails by looping, question
    79 mostly by exhaustion (varied reasoning until the budget runs out) (`analyze.py screen-anatomy`).

## 4. Serving facts
Charts: [speed, acceptance and KV capacity](https://urbanastrola.github.io/local-inference-evals/serving.html).

17. **KV capacity depends on the layout; tpurtell's default DCP1 layout with MLA layer ownership holds the most.**
    **Supported** (reported by the server at start-up, same memory setting): 3.25bpw on tpurtell 0.9.0/0.9.1 defaults
    4,707,515 tokens; with v0.7.0's layout on the 0.9.0 image 3,165,056; the component-screen arms on 0.9.1 3,992,056
    (EP2, NOPE records off), 2,215,158 (EP2, ownership tp) and 1,851,617 (NOPE records off, ownership tp); the tpurtell
    0.7.0 image 2,894,456 at three draft tokens and 2,758,919 at five. 4bpw TR3 (Brandon) on tpurtell 0.9.1 has 1,377,179
    tokens, which holds 4.17 requests at the token cap (327,680 generated plus the longest prompt, 2,795): at 8 concurrent
    GPQA requests the pool was full for much of the run and requests waited (653 of 1,417 status lines), at 4 none waited
    (peak usage 91.7%, no preemption). On tpurtell 0.7.0 + kpool fixes it has 437,563 tokens, and the engine crashed when
    the pool filled at 8 concurrent requests.
18. **Neither fix shows a speed cost, in single runs.** **Descriptive** (serving probe, one run per configuration, no
    noise floor): the kpool fixes move per-request decode speed by -3.5% to +1.7% and acceptance by at most 0.004
    (`comparisons/glm53-flash-serving-probe`); the DCP1 tail fix 147.3 vs 146.9 tok/s at concurrency 1 and 62.5 vs 64.9
    at 8, acceptance 0.534 vs 0.526 and 0.529 vs 0.550. Draft acceptance at 8 concurrent requests was 0.5285 on unpatched
    tpurtell 0.9.0 against 0.548-0.552 on the other DFlash2 ×3 builds (single runs).
19. **The engine is not bitwise reproducible, even one request at a time.** **Supported.** The same configuration run
    twice with greedy decoding, one request at a time, diverges after a median of 318 characters
    (`comparisons/glm53-flash-serving-probe`), and two decode-vs-prefill runs of tpurtell 0.9.1 with the same prompts and
    seeds, one request at a time, first differ in their per-position values after 1 to 130 generated tokens
    (`glm53-flash-v091-decode-prefill-repeat`). Batching cannot explain this; at 8 or 12 concurrent requests batching adds
    further variation. Greedy parity therefore cannot certify speculative exactness on this stack.

## Reading these results
- **Seeds.** The engine draws each request's sampling noise from its request seed. Repeats that share a seed still vary
  (the engine is not bitwise reproducible, and batching adds variation) and their texts diverge, but they draw on the same
  sampler noise, so that variation is not a statistically meaningful sample. Screens therefore use a distinct seed per
  repeat (`hard-prompt-screen/v2`) and GPQA a distinct request seed per pass (pass *p* sends 1233 + *p*;
  `tools/verify.py` rejects GPQA passes that share a seed). Every configuration has pass 1 (seed 1234), so pass-1
  comparisons are paired by question; three configurations also have passes 2 and 3, and their spread is the measured
  run-to-run noise (item 5).
- **Receipts.** Every figure is recomputed from published rows, and server-log figures from `server_log.jsonl`. Two rest
  on unpublished model output: the greedy shared-prefix lengths and the per-row `correct_stated` judgement (hashes of
  the outputs are published).
- **Questions are the unit.** Non-completion is concentrated on a few hard questions; screen rates are reported per
  question and never pooled across questions, and multi-pass GPQA records are compared with questions as clusters (a
  question answered in three passes counts once), with pooled tests shown only for reference.
- **Intervals.** GPQA accuracy: 95% bootstrap over questions, each question resampled with all its passes. Rates (empty
  answers in one pass; screen failures of one question in one configuration): 95% Wilson intervals. Differences inside
  the intervals are ties.
- **Concurrency.** GPQA runs at 8 concurrent requests unless the label says otherwise (`concurrency 4` for 4bpw TR3
  (Brandon) on tpurtell 0.9.1), and screens at 12; match it when rerunning.

## Open questions
Stated with their evidence.
- **What drives non-completion on questions 88 and 79?** Question 88 fails in 1-3 of 12 draws on every tested arm of
  tpurtell 0.9.1; question 79 in 9 of 12 on tpurtell 0.9.1 and on the tpurtell 0.7.0 image. No tested runtime part moved
  either, at a size where only very large effects could show.
- **What makes tpurtell 0.7.0 as shipped leave fewer GPQA questions unanswered than tpurtell 0.9.1?** Over three passes
  with 3.25bpw weights: 4 vs 16 empty answers of 594, on 4 vs 11 questions (question-clustered p = 0.009; passes 2 and 3
  alone 4 vs 11 empty answers on 1 vs 7 questions, p = 0.06), almost all at the 327,680-token cap; accuracy does not differ measurably. 0.7.0 differs in
  draft depth (5 vs 3), parallel layout (DCP2 with EP2 vs DCP1 with MLA layer ownership), kernel and engine code, vision,
  and KV pool at once, and none of these was varied alone in GPQA. The component screen ran the 0.7.0 image
  at three draft tokens on question 79 only, where it failed as often as 0.9.1 (9 of 12 each), and found no tested layout
  part that moved question 88 on 0.9.1. Which difference matters is inconclusive.
- **Do other releases or layouts differ in non-completion across the benchmark?** Only pass 1 exists for the other
  configurations: 0 to 9 empty answers of 198, with no paired difference at p < 0.05. 3.25bpw and 4bpw TR3 (Brandon) on
  tpurtell 0.9.1 show no measurable difference over three passes (16 vs 17 empty answers).
- **Does the DCP1 tail fix change answers or how often hard questions fail to finish?** It changes decode numerics in
  the first 2,044 tokens; the loops observed are stopped far later; GPQA and tool calling show no difference beyond
  noise at their sizes.
- **Why is the engine not bitwise reproducible one request at a time?** Greedy reruns diverge after a median of 318
  characters, and same-seed decode-vs-prefill runs first differ after 1 to 130 tokens, with no concurrent requests.
- **Do these quants cost accuracy?** No higher-precision reference (BF16 or NVFP4) was run on this hardware. The published
  90.6-92.1 used other weights and harnesses that are not fully published, so the gap to them is not a measure of
  quantization.
- Code-level questions with no measurement yet (the 511-pool slice, `swiglu_limit` on the routed experts, top-k ties
  between layouts) are listed in the investigation.
