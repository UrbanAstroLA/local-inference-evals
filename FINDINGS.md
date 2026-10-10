# Findings: GLM-5.3-Flash on 2x RTX PRO 6000 (2026-09 to 2026-10)

Analysis only; every number links to receipts in `runs/` and can be regenerated with `tools/analyze.py` (subcommands
named below), `tools/verify.py` and the investigation's `recompute.py`. Each statement says how strong it is:
**supported** (deterministic, or statistically clear), **descriptive** (what the data shows, without a test that
separates it from chance), **unmeasured**, or **open**. Terms (kpool, DCP1, MLA ownership, NOPE record, DFlash2,
flexible-extract, loop, exhaustion and others) are defined in the [glossary](README.md#glossary).

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
   [`investigations/2026-10-glm53-kpool-tail`](investigations/2026-10-glm53-kpool-tail)). The tests ran on the 0.7.0 and
   0.8.0 release images, on local builds with the fixes, and (2026-10-09) on the 0.9.0 and 0.9.1 release images, which pass
   33 of 33 and reproduce no rejected-draft corruption at 2, 3, 5 or 7 draft tokens.
2. **Under the DCP1 layout of tpurtell 0.8.0 and 0.9.0, decode attention skipped the newest 1-3 tokens at causal lengths
   up to 2,043 that are not a multiple of 4; the DCP1 tail fix removes it.** **Supported** (index check on the image's
   own kernels: the tail is dropped in 9 of 23 packed cases without the fix, 0 with it; rows the stock code already
   handled are byte-identical). DCP1 with MLA layer ownership is tpurtell's layout choice and frees KV memory (item 13);
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
   84.3% on 0.9.0, one pass each, well inside noise (item 5). In tool-eval-bench, scenarios TC-80 and TC-88 pass in both
   repeats with the fix and fail in both without it (157 and 157 of 176 points → 159 and 163), but that is two repeats per
   build, and ten other scenarios flip between repeats of the same build
   (`comparisons/glm53-flash-v090-tailfix-tool-eval`). **Descriptive.**

## 2. GPQA Diamond, pass 1 per configuration
`analyze.py gpqa-table`. 198 questions, request seed 1234, the same prompts and answer order in every run. Raw =
`flexible-extract`, the headline (protocol v1); stated = `correct_stated`, the secondary, audited stated-answer score;
"answered" leaves out empty answers.

| Weights | Engine | Speculation | Raw (95% interval) | Raw, answered | Stated | Stated, answered | Empty |
|---|---|---|---|---|---|---|---|
| 3.25bpw | tpurtell 0.7.0 | DFlash2 ×5 | 87.9% (82.8-91.9) | 87.9% | 89.4% | 89.4% | 0 |
| 3.25bpw | tpurtell 0.7.0 + kpool fixes | DFlash2 ×5 | 85.9% (80.8-90.4) | 86.7% | 87.4% | 88.3% | 2 |
| 3.25bpw | tpurtell 0.8.0 | DFlash2 ×3 | 85.9% (80.8-90.4) | 88.1% | 86.9% | 89.1% | 5 |
| 3.25bpw | tpurtell 0.8.0 | DFlash2 ×5, sharing off | 86.9% (81.8-91.4) | 89.1% | 88.4% | 90.7% | 5 |
| 3.25bpw | tpurtell 0.8.0 + kpool fixes ≈ 0.9.0 | DFlash2 ×5 | 86.9% (82.3-91.4) | 88.2% | 88.4% | 89.7% | 3 |
| 3.25bpw | tpurtell 0.9.1 | DFlash2 ×3 | 85.4% (80.3-89.9) | 87.6% | 85.9% | 88.1% | 5 |
| 4bpw TR3 (Brandon) | tpurtell 0.8.0 | DFlash2 ×3 | 85.9% (80.8-90.4) | 87.2% | 87.9% | 89.2% | 3 |
| 4bpw TR3 (Brandon) | tpurtell 0.9.0 | DFlash2 ×3 | 84.3% (78.8-88.9) | 88.4% | 85.4% | 89.4% | 9 |
| 4bpw TR3 (Brandon) | tpurtell 0.9.1 | DFlash2 ×3 | 84.8% (79.8-89.4) | 87.5% | 86.9% | 89.6% | 6 |

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
   at 84.3-87.9% raw (85.4-89.4% stated), and every paired difference is consistent with noise; each comparison resolves
   only differences of about 5 points, so smaller effects are neither shown nor excluded. Every pass 1 sent request seed
   1234, so the configurations share sampler noise question by question, and this range likely understates how much
   independent runs of one configuration vary. Clean passes 2 and 3 with per-pass seeds are in progress.
6. **Empty answers are 0 to 9 of 198 per run.** **Descriptive.** The two tpurtell 0.7.0 builds left 0 and 2 empty in
   pass 1, every later build 3 to 9; with one draw per question this does not establish a difference between the
   releases (open question below). No paired comparison of empty answers reaches p < 0.05. Where a request log exists (the two 0.9.1 runs), 10 of the 11
   empty answers ran to the 327,680-token cap and one ended after 38 tokens (`finish_reason` stop).
7. **Scoring limitation.** The raw `flexible-extract` score reads some correct answers as wrong when a reply mentions
   other options' labels or chemistry notation after its answer, or states its answer as a boxed or bold letter: in the
   published runs 25 correct answers are scored wrong, 0.5 to 2.0 points per run, and none the other way. The audited
   `correct_stated` score is published per row beside it (`protocols/gpqa-diamond/v1.md`); raw scores stay the headline
   so runs remain comparable.
8. **Published scores, for context only.** On the raw score, NVIDIA's 92.1 (BF16 and NVFP4) lies above every local run's
   interval and Red Hat's 90.6 (NVFP4) inside the intervals of three runs (all at DFlash2 ×5); on the stated-answer score, 90.6 lies inside seven runs' intervals and 92.1 inside three. Those numbers used
   other weights and harnesses whose details are not fully published; no higher-precision reference was run here, so
   these runs cannot separate quantization, harness, scoring and runtime effects
   (`comparisons/glm53-flash-gpqa-published-context`).

## 3. Non-completion on hard questions
Details, failure anatomy and open questions: [`investigations/2026-10-glm53-looping`](investigations/2026-10-glm53-looping).
`analyze.py screen-v2`.

9. **On tpurtell 0.9.1, question 88 fails to finish in 1-3 of 12 draws in every tested arm; question 79 in 9 of 12 on
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
10. **None of the tested runtime parts moved either question at this size, and only very large effects could have
    shown.** **Descriptive.** Question 88: EP2 routed experts 3 vs 5 failures of 24 (Fisher p = 0.70), NOPE records off
    4 vs 4 (p = 1.00), MLA ownership tp 3 vs 5 (p = 0.70); arms EO and NO also turned draft-slot sharing off, so ownership
    and sharing are not separated; the preregistered rule found no candidate. Question 79: 9 vs 9; the rule's verdict is
    "unresolved". Minimum detectable differences (two-sided Fisher, p < 0.05, 80% power; `analyze.py screen-power`): one
    arm against another on question 88 (12 vs 12, from 25%) only a rise of about 60 points, and no drop at any size; a
    switch on vs off (24 vs 24, from 17%) a rise of about 41 points; question 79 (from 75%) a drop of about 60 points.
    Draft depth, quantization and sampling settings were not varied.
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
14. **Neither fix shows a speed cost, in single runs.** **Descriptive** (serving probe, one run per configuration, no
    noise floor): the kpool fixes move per-request decode speed by -3.5% to +1.7% and acceptance by at most 0.004
    (`comparisons/glm53-flash-serving-probe`); the DCP1 tail fix 147.3 vs 146.9 tok/s at concurrency 1 and 62.5 vs 64.9
    at 8, acceptance 0.534 vs 0.526 and 0.529 vs 0.550. Draft acceptance at 8 concurrent requests was 0.5285 on unpatched
    tpurtell 0.9.0 against 0.548-0.552 on the other DFlash2 ×3 builds (single runs).
15. **The engine is not bitwise reproducible, even one request at a time.** **Supported.** The same configuration run
    twice with greedy decoding, one request at a time, diverges after a median of 318 characters
    (`comparisons/glm53-flash-serving-probe`), and two decode-vs-prefill runs of tpurtell 0.9.1 with the same prompts and
    seeds, one request at a time, first differ in their per-position values after 1 to 130 generated tokens
    (`glm53-flash-v091-decode-prefill-repeat`). Batching cannot explain this; at 8 or 12 concurrent requests batching adds
    further variation. Greedy parity therefore cannot certify speculative exactness on this stack.

## Reading these results
- **Seeds.** The engine draws each request's sampling noise from its request seed. Repeats that share a seed still vary
  (the engine is not bitwise reproducible, and batching adds variation) and their texts diverge, but they draw on the same
  sampler noise, so that variation is not a statistically meaningful sample. Screens therefore use a distinct seed per repeat (`hard-prompt-screen/v2`) and GPQA a distinct request
  seed per pass (pass *p* sends 1233 + *p*). All GPQA runs here are pass 1 (seed 1234), so the configurations share
  sampler noise question by question; the spread across them likely understates independent run-to-run spread, and
  comparisons are paired by question.
- **Receipts.** Every figure is recomputed from published rows, and server-log figures from `server_log.jsonl`. Two rest
  on unpublished model output: the greedy shared-prefix lengths and the per-row `correct_stated` judgement (hashes of
  the outputs are published).
- **Questions are the unit.** Non-completion is concentrated on a few hard questions; rates are reported per question
  and never pooled across questions.
- **Intervals.** GPQA accuracy: 95% bootstrap over questions. Rates (empty answers; screen failures of one question in
  one configuration): 95% Wilson intervals. Differences inside the intervals are ties.
- **Concurrency.** GPQA runs at 8 concurrent requests and screens at 12; match it when rerunning.

## Open questions
Stated with their evidence.
- **What drives non-completion on questions 88 and 79?** Question 88 fails in 1-3 of 12 draws on every tested arm of
  tpurtell 0.9.1; question 79 in 9 of 12 on tpurtell 0.9.1 and on the tpurtell 0.7.0 image. No tested runtime part moved
  either, at a size where only very large effects could show.
- **Does tpurtell 0.7.0 as shipped leave fewer empty answers?** In pass 1, the two tpurtell 0.7.0 builds (DFlash2 ×5) left
  0 and 2 of 198 GPQA answers empty; every later build left 3 to 9. One pass each does not separate this from chance. The
  question-79 screen ran 0.7.0 at three draft tokens, not the five its GPQA runs used, so it does not bear on this.
- **Do releases or layouts differ in non-completion across the benchmark?** One GPQA pass per configuration shows 0 to 9
  empty answers of 198, with no paired difference at p < 0.05.
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
