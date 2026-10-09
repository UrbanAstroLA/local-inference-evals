# Findings: GLM-5.3-Flash on 2x RTX PRO 6000 (2026-09 to 2026-10)

Analysis only; every number links to receipts in `runs/` and can be regenerated with `tools/analyze.py` (subcommands
named below), `tools/verify.py` and the investigation's `recompute.py`. Each statement says how strong it is:
**supported** (deterministic, or statistically clear), **descriptive** (what the data shows, without a test that
separates it from chance), or **open**.

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
| kpool fixes | Upstream vLLM fixes vllm-project/vllm#57477 and #58454, ported in [tpurtell/glm-5.3-flash-ext3-2x-rtx#5](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/5) (commit 5a366b5) and shipped in v0.9.0 |
| 3.25bpw | tpurtell's K3.25 checkpoint [wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1) (EXL3, mixed K3/K4 routed experts) |
| 4bpw TR3 (Brandon) | Brandon M. Music's TR3 checkpoint [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) (EXL3, uniform K4 routed experts) |
| DFlash2 ×N | DFlash2 speculative decoding (incoai/GLM-5.3-Flash-DFlash2), N draft tokens per step; "no speculation" = plain decoding |

All engines here are tpurtell builds. 0.7.0 and the 0.8.0-0.9.1 line also differ in their default parallel layout
(0.7.0: EP2 + DCP2, vision on; 0.8.0 onward: EP1 + DCP1 with MLA layer ownership, vision off), so "engine" below means
the release as shipped.

## 1. Runtime defects and their deterministic evidence

1. **Two upstream kpool bugs were present in the tpurtell 0.7.0 and 0.8.0 images, and the fixes remove them.**
   **Supported.** Upstream vLLM's own regression tests for the kpool kernels pass 29 of 33 on both images and 33 of 33
   with the fixes (vllm-project/vllm#57477: every prefill wrote 2 KB of keys into another block's indexer region;
   #58454: a rejected pool-completing draft could overwrite committed keys at 2 or more draft tokens). Ported in
   tpurtell PR #5 and shipped in tpurtell 0.9.0 (`runs/*_kpool-kernel-tests`;
   [`investigations/2026-10-glm53-kpool-tail`](investigations/2026-10-glm53-kpool-tail)).
2. **Under the DCP1 layout of tpurtell 0.8.0 and 0.9.0, decode attention skipped the newest 1-3 tokens at causal lengths
   up to 2,043 that are not a multiple of 4; the DCP1 tail fix removes it.** **Supported** (index check on the image's
   own kernels: the tail is dropped in 9 of 23 packed cases without the fix, 0 with it; rows the stock code already
   handled are byte-identical). DCP1 with MLA layer ownership is tpurtell's layout choice and frees KV memory (item 13);
   the masking path already existed in the vendored attention code, and only this layout exercises it. The fix was merged
   as tpurtell PR #6 and released in tpurtell 0.9.1
   ([`investigations/2026-10-glm53-looping`](investigations/2026-10-glm53-looping)).
3. **With the fix, decode agrees much better with prefill re-scoring of the same tokens below 2,044 tokens:** mean KL
   0.066 → 0.010, top-1 agreement 93.8% → 97.6%, lower in 6 of 6 prompts. **Supported for this large effect.** Each build
   was measured once; a later repeat of the measurement on tpurtell 0.9.1 (not published) varied by up to about 2x
   between runs, so the smaller difference beyond 2,048 tokens is not claimed
   (`comparisons/glm53-flash-v090-tailfix-decode-prefill`).
4. **With the fix, tool-eval-bench scenarios TC-80 and TC-88 pass in both repeats instead of failing in both** (157 and
   157 of 176 points → 159 and 163). **Descriptive:** two repeats per build, and ten other scenarios vary between repeats
   of the same build (`comparisons/glm53-flash-v090-tailfix-tool-eval`).

## 2. GPQA Diamond, pass 1 per configuration
`analyze.py gpqa-table`. 198 questions, request seed 1234, the same prompts and answer order in every run.

| Weights | Engine | Speculation | Accuracy (95% interval over questions) | Empty answers |
|---|---|---|---|---|
| 3.25bpw | tpurtell 0.7.0 | DFlash2 ×5 | 87.9% (82.8-91.9) | 0 |
| 3.25bpw | tpurtell 0.7.0 + kpool fixes | DFlash2 ×5 | 85.9% (80.8-90.4) | 2 |
| 3.25bpw | tpurtell 0.8.0 | DFlash2 ×3 | 85.9% (80.8-90.4) | 5 |
| 3.25bpw | tpurtell 0.8.0 | DFlash2 ×5, sharing off | 86.9% (81.8-91.4) | 5 |
| 3.25bpw | tpurtell 0.8.0 + kpool fixes ≈ 0.9.0 | DFlash2 ×5 | 86.9% (82.3-91.4) | 3 |
| 3.25bpw | tpurtell 0.9.1 | DFlash2 ×3 | 85.4% (80.3-89.9) | 5 |
| 4bpw TR3 (Brandon) | tpurtell 0.8.0 | DFlash2 ×3 | 85.9% (80.8-90.4) | 3 |
| 4bpw TR3 (Brandon) | tpurtell 0.9.0 | DFlash2 ×3 | 84.3% (78.8-88.9) | 9 |
| 4bpw TR3 (Brandon) | tpurtell 0.9.1 | DFlash2 ×3 | 84.8% (79.8-89.4) | 6 |

Question-paired comparisons of configurations that differ in one named respect (`analyze.py gpqa-pairs`; exact McNemar
tests on the questions where two runs disagree; no correction for multiple comparisons):

| A vs B | What differs | Only A right / only B right (p) | B - A, points (95% interval) | Only A empty / only B empty (p) |
|---|---|---|---|---|
| 3.25bpw 0.7.0 ×5 vs 0.8.0 ×3 | engine release as shipped | 10 / 6 (0.45) | -2.0 (-6.1 to +2.0) | 0 / 5 (0.06) |
| 3.25bpw 0.8.0 ×3 vs ×5, sharing off | draft depth, slot sharing | 10 / 12 (0.83) | +1.0 (-3.5 to +5.6) | 3 / 3 (1.00) |
| 3.25bpw 0.7.0 vs 0.7.0 + kpool fixes | kpool fixes | 10 / 6 (0.45) | -2.0 (-6.1 to +2.0) | 0 / 2 (0.50) |
| 4bpw 0.8.0 vs 0.9.0 | kpool fixes (release) | 11 / 8 (0.65) | -1.5 (-6.1 to +3.0) | 2 / 8 (0.11) |
| 4bpw 0.9.0 vs 0.9.1 | DCP1 tail fix (release) | 13 / 14 (1.00) | +0.5 (-4.5 to +5.6) | 5 / 2 (0.45) |
| 3.25bpw vs 4bpw on 0.8.0 | weights | 10 / 10 (1.00) | 0.0 (-4.5 to +4.5) | 4 / 2 (0.69) |
| 3.25bpw vs 4bpw on 0.9.1 | weights | 13 / 12 (1.00) | -0.5 (-5.6 to +4.5) | 4 / 5 (1.00) |

5. **One pass per configuration does not separate these configurations in accuracy.** **Descriptive.** Every run lands
   at 84.3-87.9%, and every paired difference is consistent with noise; each comparison resolves only differences of
   about 5 points, so smaller effects are neither shown nor excluded.
6. **Empty answers are 0 to 9 of 198 per run.** **Descriptive.** In pass 1, tpurtell 0.7.0 left none and 0.8.0 left 5
   on the same 3.25bpw weights; that is one draw per question and does not establish a difference between the releases.
   No paired comparison of empty answers reaches p < 0.05. Where a request log exists (the two 0.9.1 runs), 10 of the 11
   empty answers ran to the 327,680-token cap and one ended after 38 tokens (`finish_reason` stop).
7. **Scoring limitation.** The raw `flexible-extract` score reads some correct answers as wrong when a reply mentions
   other options' labels or chemistry notation after its answer: 0.5 to 2.5 points low per pass, about 1.6 on average
   (`protocols/gpqa-diamond/v1.md`, known limitation). Raw scores stay the headline so runs remain comparable.
8. **Published scores, for context only.** NVIDIA's 92.1 (BF16 and NVFP4) lies above every local run's interval; Red
   Hat's 90.6 (NVFP4) lies inside the interval of `3.25bpw · tpurtell 0.7.0 · DFlash2 ×5` and above the others. Weights,
   harness and scoring differ, so these runs cannot separate quantization, harness and runtime effects
   (`comparisons/glm53-flash-gpqa-published-context`).

## 3. Non-completion on hard questions
Details, failure anatomy and open questions: [`investigations/2026-10-glm53-looping`](investigations/2026-10-glm53-looping).
`analyze.py screen-v2`.

9. **On tpurtell 0.9.1, question 88 fails to finish in 1-3 of 12 draws in every tested arm; question 79 in 9 of 12 on
   both tpurtell 0.9.1 and the tpurtell 0.7.0 image at three draft tokens.** **Descriptive** (component screen,
   2026-10-09, preregistered: one question per run, distinct request seeds 5001-5012, 12 concurrent requests).

   | Arm | Configuration | Question | Failed / 12 (95% Wilson) |
   |---|---|---|---|
   | B | `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3` | 88 | 3 (9-53%) |
   | EN | `3.25bpw · tpurtell 0.9.1 · EP2 experts, NOPE records off · DFlash2 ×3` | 88 | 2 (5-45%) |
   | EO | `3.25bpw · tpurtell 0.9.1 · EP2 experts, MLA owners tp · DFlash2 ×3, sharing off` | 88 | 1 (1-35%) |
   | NO | `3.25bpw · tpurtell 0.9.1 · NOPE records off, MLA owners tp · DFlash2 ×3, sharing off` | 88 | 2 (5-45%) |
   | B79 | `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3` | 79 | 9 (47-91%) |
   | V79 | `3.25bpw · tpurtell 0.7.0 · DFlash2 ×3` | 79 | 9 (47-91%) |
10. **None of the tested runtime parts moved either question at this size.** **Descriptive.** Question 88: EP2 routed
    experts 3 vs 5 failures of 24 (Fisher p = 0.70), NOPE records off 4 vs 4 (p = 1.00), MLA ownership tp 3 vs 5
    (p = 0.70); the preregistered rule found no candidate. Question 79: 9 vs 9; the rule's verdict is "unresolved". The
    95% intervals for these differences reach 14-33 points, so effects of that size are not excluded.
11. **Repeats that share one request seed vary, but not in a statistically meaningful way.** **Supported.** Same configuration and question
    (88), 12 repeats each: 3 failures with distinct seeds, 11 with seed 1234 on every repeat (Fisher p = 0.003;
    `comparisons/glm53-flash-fixed-seed-control`). The earlier hard-question screens sent seed 1234 on every repeat; their
    rates, the layout-bisection statistics and the question-level observations drawn from them were withdrawn on
    2026-10-09 (notice in the investigation). Repeat 1 of each question from each configuration's first screen is kept
    as a single draw (`analyze.py screen-single`): loops or exhaustion occur in every configuration tested.
12. **Loops are stopped far beyond the 2,044-token region where the tail bug acted.** **Descriptive.** In the component
    screen the early-stop detector fired at an estimated 106,000-187,000 tokens; question 88 fails by looping, question
    79 mostly by exhaustion (varied reasoning until the budget runs out) (`analyze.py screen-anatomy`).

## 4. Serving facts

13. **KV capacity depends on the layout; tpurtell's default DCP1 layout with MLA layer ownership holds the most.**
    **Supported** (reported by the server at start-up, same memory setting): 3.25bpw on tpurtell 0.9.0/0.9.1 defaults
    4,707,515 tokens; with v0.7.0's layout on the 0.9.0 image 3,165,056; the component-screen arms on 0.9.1 3,992,056
    (EP2, NOPE records off), 2,215,158 (EP2, ownership tp) and 1,851,617 (NOPE records off, ownership tp); the tpurtell
    0.7.0 image 2,894,456. 4bpw TR3 (Brandon) on tpurtell 0.9.1 has 1,377,179 tokens and ran fewer than 8 GPQA requests at
    once while the pool was full; on tpurtell 0.7.0 + kpool fixes it has 437,563 tokens, and the engine crashed when the
    pool filled at 8 concurrent requests.
14. **Neither fix costs measurable speed.** **Descriptive** (serving probe, one run per configuration): the kpool fixes
    move per-request decode speed by -3.5% to +1.7% and acceptance by at most 0.004
    (`comparisons/glm53-flash-serving-probe`); the DCP1 tail fix 147.3 vs 146.9 tok/s at concurrency 1 and 62.5 vs 64.9
    at 8, acceptance 0.534 vs 0.526 and 0.529 vs 0.550.
15. **The engine is not bitwise reproducible at temperature 0.** **Supported.** The same configuration run twice diverges
    after a median of 318 characters (`comparisons/glm53-flash-serving-probe`), so greedy parity cannot certify
    speculative exactness on this stack.

## Reading these results
- **Seeds.** The engine draws each request's sampling noise from its request seed. Repeats that share a seed still vary,
  because concurrent batching changes the arithmetic and their texts diverge, but they draw on the same sampler noise,
  so that variation is not a statistically meaningful sample. Screens therefore use a distinct seed per repeat (`hard-prompt-screen/v2`) and GPQA a distinct request
  seed per pass (pass *p* sends 1233 + *p*). All GPQA runs here are pass 1 (seed 1234), so a question's draw in two
  configurations starts from the same sampler noise; how far that correlates outcomes across configurations is not
  measured, and comparisons are paired by question.
- **Questions are the unit.** Non-completion is concentrated on a few hard questions; rates are reported per question
  and never pooled across questions.
- **Intervals.** GPQA accuracy: 95% bootstrap over questions. Rates (empty answers; screen failures of one question in
  one configuration): 95% Wilson intervals. Differences inside the intervals are ties.
- **Concurrency.** GPQA runs at 8 concurrent requests and screens at 12; match it when rerunning.

## Open questions
Stated with their evidence; how to pursue them is left to the reader.
- **What drives non-completion on questions 88 and 79?** Question 88 fails in 1-3 of 12 draws on every tested arm of
  tpurtell 0.9.1; question 79 in 9 of 12 on tpurtell 0.9.1 and on the tpurtell 0.7.0 image. No tested runtime part moved
  either at this size.
- **Do releases or layouts differ in non-completion across the benchmark?** One GPQA pass per configuration shows 0 to 9
  empty answers of 198, with no paired difference at p < 0.05.
- **Does the DCP1 tail fix change how often hard questions fail to finish?** It changes decode numerics in the first
  2,044 tokens; the loops observed are stopped far later.
- Code-level questions with no measurement yet (the 511-pool slice, `swiglu_limit` on the routed experts, top-k ties
  between layouts) are listed in the investigation.
