# Findings: GLM-5.3-Flash on 2x RTX PRO 6000 (2026-09 to 2026-10)

Every statement from this evaluation, numbered and graded, with its receipt.

- Plain-language summary and result tables: [Results page](https://urbanastrola.github.io/local-inference-evals/).
- What else can move a result: [`CONFOUNDS.md`](CONFOUNDS.md). What changed and when: [`LEDGER.md`](LEDGER.md).
- Labels: [key](#labels). Terms: [glossary](README.md#glossary).

**Grades.** **Supported**: deterministic, or statistically clear. **Descriptive**: what the data shows, without a test
that separates it from chance. **Open**: not answered by the data. "Not graded" marks scoring and context notes.

<details>
<summary>How to recompute</summary>

`python3 tools/verify.py` recomputes every summary. `python3 tools/analyze.py <subcommand>` (named per row)
recomputes the analyses. `python3 investigations/2026-10-glm53-looping/recompute.py` prints every number of the
investigation. Standard library only.

</details>

## 1. Engine issues found during evaluation, and their fixes

| # | Statement | Grade | Evidence |
|---|---|---|---|
| 1 | Two upstream vLLM kpool issues were present in the tpurtell 0.7.0 and 0.8.0 images; the fixes remove them. Upstream's tests: 29 of 33 pass without, 33 of 33 with. | **Supported** | `runs/*_kpool-kernel-tests`, [kpool investigation](investigations/2026-10-glm53-kpool-tail) |
| 2 | Under the DCP1 layout of tpurtell 0.8.0 and 0.9.0, decode attention skipped the newest 1-3 tokens at causal lengths up to 2,043 not divisible by 4. The DCP1 tail fix removes it. | **Supported** | [investigation](investigations/2026-10-glm53-looping/README.md#the-dcp1-tail-issue-and-its-fix) |
| 3 | With the fix, decode agrees much better with prefill below 2,044 tokens: mean KL 0.066 → 0.010, 0.008 and 0.006. | **Supported** (for this large effect) | [`glm53-flash-v090-tailfix-decode-prefill`](comparisons/glm53-flash-v090-tailfix-decode-prefill), [`glm53-flash-v091-decode-prefill-repeat`](comparisons/glm53-flash-v091-decode-prefill-repeat) |
| 4 | The fix's effect on answers: GPQA and tool calling show no difference beyond noise at their sizes. | **Descriptive** (effect unmeasured) | [`glm53-flash-k4-v090-v091-gpqa`](comparisons/glm53-flash-k4-v090-v091-gpqa), [`glm53-flash-v090-tailfix-tool-eval`](comparisons/glm53-flash-v090-tailfix-tool-eval) |

<details>
<summary>Numbers and qualifications, items 1-4</summary>

1. vllm-project/vllm#57477: every prefill wrote 2 KB of keys into another block's indexer region. #58454: a rejected
   pool-completing draft could overwrite committed keys at 2 or more draft tokens. 29 of 33 pass on both images.
   Ported in tpurtell PR #5, shipped in tpurtell 0.9.0. The tests ran on the 0.7.0 and 0.8.0 release images, on local
   builds with the fixes, and (2026-10-09) on the 0.9.0 and 0.9.1 release images, which pass 33 of 33 and reproduce no
   rejected-draft corruption at 2, 3, 5 or 7 draft tokens.
2. Index check on the image's own kernels: the tail is dropped in 9 of 23 packed cases without the fix, 0 with it; rows
   the stock code already handled are byte-identical. DCP1 with MLA layer ownership is tpurtell's layout choice and
   frees KV memory (item 17); the masking path already existed in the vendored attention code, and only this layout
   exercises it. Merged as tpurtell PR #6, released in tpurtell 0.9.1.
3. Six prompts every run shares: 0.066 without the fix (one run), 0.010 with the local build of the fix, 0.008 and 0.006
   in two runs of the 0.9.1 release. Top-1 agreement 93.8% → 97.6%; lower in 6 of 6 prompts. The two 0.9.1 runs, same
   prompts and seeds, are the noise floor: means differ by up to about 2x, single prompts by up to about 6x. From 2,048
   tokens, runs with the fix range 0.006 to 0.019 and the one run without it is 0.031: no effect claimed there. With
   speculation on (DFlash2 ×3, one run), 0.9.1 decode sits somewhat above both speculation-off runs, within that spread
   ([`glm53-flash-v091-decode-prefill-speculation`](comparisons/glm53-flash-v091-decode-prefill-speculation);
   descriptive).
4. GPQA: 4bpw TR3 (Brandon) 84.8% on tpurtell 0.9.1 vs 84.3% on 0.9.0, one pass each at 8 concurrent requests with a
   KV pool that holds about 4 requests at the token cap; well inside the 0.5-3.5 points by which passes of one
   configuration differ (item 5). tool-eval-bench: TC-80 and TC-88 pass in both repeats with the fix and fail in both
   without it (157 and 157 of 176 points → 159 and 163), but that is two repeats per build, and ten other scenarios
   flip between repeats of the same build.

</details>

## 2. GPQA Diamond

198 questions; pass *p* sends request seed 1233 + *p*. Raw = `flexible-extract` (the headline); stated =
`correct_stated` (audited). 8 concurrent requests unless the label says otherwise. Tables:
[Results page](https://urbanastrola.github.io/local-inference-evals/#gpqa),
[`glm53-flash-gpqa-records`](comparisons/glm53-flash-gpqa-records) (three passes),
[`glm53-flash-gpqa-configs`](comparisons/glm53-flash-gpqa-configs) (pass 1).

| # | Statement | Grade | Evidence |
|---|---|---|---|
| 5 | One pass of one configuration varies by 0.5 to 3.5 points between passes with their own request seeds. | **Descriptive** | `analyze.py gpqa-passes` |
| 6 | 3.25bpw and 4bpw TR3 (Brandon) on tpurtell 0.9.1 show no measurable difference in accuracy (86.9% vs 85.7%) or completion (16 vs 17 empty of 594). | **Descriptive** | `analyze.py gpqa-records` |
| 7 | tpurtell 0.7.0 as shipped left fewer GPQA questions unanswered than tpurtell 0.9.1 (3.25bpw, three passes each): 4 vs 16 empty answers of 594. What produces the difference is open. | **Supported** (clustered p = 0.009); cause **open** | `analyze.py gpqa-records`, [confounds](CONFOUNDS.md#7-several-changes-between-releases-at-once) |
| 8 | Empty answers concentrate on a few questions. | **Descriptive** | `analyze.py gpqa-empty` |
| 9 | One pass per configuration does not separate these configurations in accuracy: pass 1 lands at 84.3-87.9% raw. | **Descriptive** | `analyze.py gpqa-table`, `gpqa-pairs` |
| 10 | Empty answers in pass 1 are 0 to 9 of 198 per run. | **Descriptive** | `analyze.py gpqa-table` |
| 11 | Scoring note: raw `flexible-extract` reads some correct answers as wrong, a net 0.5 to 3.5 points per pass. | Not graded (scoring note) | [`protocols/gpqa-diamond/v1.md`](protocols/gpqa-diamond/v1.md) |
| 12 | Published scores (92.1 NVIDIA, 90.6 Red Hat) are context only. | Not graded (context only) | [`glm53-flash-gpqa-published-context`](comparisons/glm53-flash-gpqa-published-context) |

<details>
<summary>Numbers and qualifications, items 5-12</summary>

5. Three configurations, three passes each. As large as the spread of pass 1 across all ten configurations
   (84.3-87.9%, 3.6 points). Of 198 questions, 154-160 are right in all three passes of a configuration, 10-12 in none,
   28-32 vary between passes.
6. Over three passes each: B - A -1.2 points (-4.2 to +1.7; clustered p = 0.51); stated 87.9% vs 87.7%; empty answers
   9 questions each way (clustered p = 1.00). The 4bpw record ran 4 requests at once, the 3.25bpw record 8. With three
   passes, accuracy differences smaller than about 3 points are neither shown nor excluded.
7. On 4 vs 11 questions; 0.9.1 had more empty answers on 10 questions and fewer on 1 (exact sign-flip test over
   questions p = 0.009; +2.0 points, 95% interval over questions +0.7 to +3.5). The question was raised by pass 1
   (0 vs 5); passes 2 and 3 alone, run after it was raised, point the same way (4 vs 11 on 1 vs 7 questions, clustered
   p = 0.06). Three record comparisons were made without correction (Bonferroni over three: p = 0.03). Accuracy does not
   differ measurably (88.2% vs 86.9%, clustered p = 0.41). Where a request log exists, every one of these empty answers
   ran to the 327,680-token cap except one, which ended after 38 tokens. 0.7.0 differs from 0.9.1 in draft depth (5 vs
   3 tokens), parallel layout (DCP2 with EP2 experts vs DCP1 with MLA layer ownership), kernel and engine code, vision
   (on vs off) and KV pool (2,758,919 vs 4,707,515 tokens) at once; the model revisions differ only in files neither
   server uses (same weight files; both servers load the same vendored chat template). None was varied alone in GPQA;
   the component screen at three draft tokens found no tested part that moved questions 88 or 79 (item 14). The 4bpw
   TR3 (Brandon) record on 0.9.1 also left more empty answers than 0.7.0 (17, on 14 questions; clustered p = 0.006), but
   it differs in weights and concurrency as well.
8. Questions 79 and 81 came back empty in all three records; 88, 127 and 147 in both 0.9.1 records and in no pass of
   0.7.0; question 88 in two of three passes of each 0.9.1 record. Questions are doc ids; the questions themselves are
   not published.
9. 85.4-89.4% stated. Within the 0.5-3.5 points by which passes of one configuration differ (item 5); every
   question-paired difference is consistent with noise. Each comparison resolves only differences of about 5 points, so
   smaller effects are neither shown nor excluded.
10. No paired pass-1 comparison of empty answers reaches p < 0.05; one pass per configuration does not separate
    releases (the three-pass records do for 0.7.0 vs 0.9.1, item 7). Of the 43 published empty answers with a request
    log (four runs), 42 ran to the 327,680-token cap and one ended after 38 tokens (`finish_reason` stop).
11. Misread when a reply mentions other options' labels or chemistry notation after its answer, or states its answer as
    a boxed or bold letter. Over the 16 published passes, 49 correct answers are scored wrong and 3 replies whose stated
    answer is not the target are credited: net 0.5 to 3.5 points per pass (1.5 on average). The audited
    `correct_stated` score is published per row beside it; raw scores stay the headline so runs remain comparable.
12. Raw score: NVIDIA's 92.1 (BF16 and NVFP4) lies above every local run's interval; Red Hat's 90.6 (NVFP4) lies inside
    the intervals of three runs, all at DFlash2 ×5 (the three-pass tpurtell 0.7.0 record among them), and above the
    others. Stated-answer score: 90.6 lies inside nine runs' intervals and 92.1 inside three. Those numbers used other
    weights and harnesses whose details are not fully published; no higher-precision reference was run here, so these
    runs cannot separate quantization, harness, scoring and runtime effects.

</details>

## 3. Non-completion on hard questions

Details: [investigation](investigations/2026-10-glm53-looping/README.md#non-completion-what-the-clean-data-shows).

| # | Statement | Grade | Evidence |
|---|---|---|---|
| 13 | On tpurtell 0.9.1, question 88 fails to finish in 1-3 of 12 draws in every tested arm; question 79 in 9 of 12 on both tpurtell 0.9.1 and the tpurtell 0.7.0 image at three draft tokens. | **Descriptive** | `analyze.py screen-v2`, [`glm53-flash-v091-component-screen-doc88`](comparisons/glm53-flash-v091-component-screen-doc88), [`glm53-flash-doc79-image-screen`](comparisons/glm53-flash-doc79-image-screen) |
| 14 | None of the tested runtime parts moved either question at this size, and only very large effects could have shown. | **Descriptive** | `analyze.py screen-v2`, `screen-power` |
| 15 | Repeats that share one request seed vary only through small numerical differences (batching and engine nondeterminism), which is not a meaningful sample of how often a question fails: 11 of 12 failed with seed 1234 on every repeat vs 3 of 12 with distinct seeds. | **Supported** (Fisher p = 0.003) | [`glm53-flash-fixed-seed-control`](comparisons/glm53-flash-fixed-seed-control) |
| 16 | Loops are stopped far beyond the 2,044-token region where the tail issue acted: an estimated 106,000-187,000 tokens. | **Descriptive** | `analyze.py screen-anatomy` |

<details>
<summary>Numbers and qualifications, items 13-16</summary>

13. Component screen, 2026-10-09, preregistered: one question per run, distinct request seeds 5001-5012. Held fixed in
    every arm: 3.25bpw weights, DFlash2 ×3, temperature 1.0 / top_p 0.95, the 327,680-token budget, 12 concurrent
    requests, one question per arm, a fresh server. Both questions were chosen as hard from GPQA runs and screens that
    all sent request seed 1234; with distinct seeds question 88 fails far less often than those screens suggested. Two
    questions are not a benchmark-wide rate. Per-arm counts with 95% Wilson intervals: in the investigation.
14. Question 88: EP2 routed experts 3 vs 5 failures of 24 (Fisher p = 0.70), NOPE records off 4 vs 4 (p = 1.00), MLA
    ownership tp 3 vs 5 (p = 0.70). Arms EO and NO also turned draft-slot sharing off, so ownership and sharing are not
    separated; the preregistered rule found no candidate. Question 79: 9 vs 9; the rule's verdict is "unresolved".
    Minimum detectable differences (two-sided Fisher, p < 0.05, 80% power; `analyze.py screen-power`): one arm against
    another on question 88 (12 vs 12, from 25%) only a rise of about 60 points, and no drop at any size; a switch on vs
    off (24 vs 24, from 17%) a rise of about 41 points; question 79 (from 75%) a drop of about 60 points. Draft depth,
    quantization and sampling settings were not varied.
15. Same configuration and question (88), 12 repeats each. The earlier hard-question screens sent seed 1234 on every
    repeat; their rates, the layout-bisection statistics and the question-level observations drawn from them were
    withdrawn on 2026-10-09 and replaced by the component screen ([ledger](LEDGER.md#withdrawn-and-what-replaced-it)).
    Repeat 1 of each question from each configuration's first screen is kept as a single draw
    (`analyze.py screen-single`): loops or exhaustion occur in every configuration tested.
16. Component screen, early-stop detector. Question 88 fails by looping, question 79 mostly by exhaustion (varied
    reasoning until the budget runs out).

</details>

## 4. Serving

Tables: [Results page](https://urbanastrola.github.io/local-inference-evals/#serving).

| # | Statement | Grade | Evidence |
|---|---|---|---|
| 17 | KV capacity depends on the layout; tpurtell's default DCP1 layout with MLA layer ownership holds the most: 4,707,515 tokens for 3.25bpw on 0.9.0/0.9.1. | **Supported** | server logs (`server_log.jsonl`) |
| 18 | Neither fix shows a speed cost, in single runs. | **Descriptive** | [`glm53-flash-serving-probe`](comparisons/glm53-flash-serving-probe) |
| 19 | The engine is not bitwise reproducible, even one request at a time. | **Supported** | [`glm53-flash-serving-probe`](comparisons/glm53-flash-serving-probe), [`glm53-flash-v091-decode-prefill-repeat`](comparisons/glm53-flash-v091-decode-prefill-repeat) |

<details>
<summary>Numbers and qualifications, items 17-19</summary>

17. Reported by the server at start-up, same memory setting. 3.25bpw: with v0.7.0's layout on the 0.9.0 image
    3,165,056; component-screen arms on 0.9.1 3,992,056 (EP2, NOPE records off), 2,215,158 (EP2, ownership tp) and
    1,851,617 (NOPE records off, ownership tp); the tpurtell 0.7.0 image 2,894,456 at three draft tokens and 2,758,919 at
    five. 4bpw TR3 (Brandon) on tpurtell 0.9.1: 1,377,179 tokens, which holds 4.17 requests at the token cap (327,680
    generated plus the longest prompt, 2,795). At 8 concurrent GPQA requests the pool was full for much of the run and
    requests waited (653 of 1,417 status lines); at 4 none waited (peak usage 91.7%, no preemption). On tpurtell 0.7.0 +
    kpool fixes it has 437,563 tokens, and the engine crashed when the pool filled at 8 concurrent requests.
18. Serving probe, one run per configuration, no noise floor. Kpool fixes: per-request decode speed -3.5% to +1.7%,
    acceptance within 0.004. DCP1 tail fix: 147.3 vs 146.9 tok/s at concurrency 1 and 62.5 vs 64.9 at 8, acceptance
    0.534 vs 0.526 and 0.529 vs 0.550. Draft acceptance at 8 concurrent requests was 0.5285 on unpatched tpurtell 0.9.0
    against 0.548-0.552 on the other DFlash2 ×3 builds (single runs).
19. The same configuration run twice with greedy decoding, one request at a time, diverges after a median of 318
    characters. Two decode-vs-prefill runs of tpurtell 0.9.1 with the same prompts and seeds, one request at a time,
    first differ in their per-position values after 1 to 130 generated tokens. Batching cannot explain this; at 8 or 12
    concurrent requests batching adds further variation. Greedy parity therefore cannot certify speculative exactness
    on this stack.

</details>

## Open questions

Stated with their evidence. Grade: **open**.

| Question | Evidence so far |
|---|---|
| What drives non-completion on questions 88 and 79? | Items 13-14: no tested runtime part moved either, at a size where only very large effects could show. |
| What makes tpurtell 0.7.0 as shipped leave fewer GPQA questions unanswered than 0.9.1? | Item 7: 4 vs 16 of 594, almost all at the 327,680-token cap; several differences at once, none varied alone in GPQA. The component screen ran the 0.7.0 image at three draft tokens on question 79 only, where it failed as often as 0.9.1 (9 of 12 each). Which difference matters is inconclusive. |
| Do other releases or layouts differ in non-completion across the benchmark? | Only pass 1 exists for the other configurations: 0 to 9 empty answers of 198, no paired difference at p < 0.05 (item 10). 3.25bpw and 4bpw TR3 (Brandon) on 0.9.1: no measurable difference over three passes (16 vs 17; item 6). |
| Does the DCP1 tail fix change answers or how often hard questions fail to finish? | It changes decode numerics in the first 2,044 tokens; the loops observed are stopped far later; GPQA and tool calling show no difference beyond noise at their sizes (items 3, 4, 16). |
| Why is the engine not bitwise reproducible one request at a time? | Item 19: greedy reruns diverge after a median of 318 characters; same-seed decode-vs-prefill runs first differ after 1 to 130 tokens, with no concurrent requests. |
| Do these quants cost accuracy? | No higher-precision reference (BF16 or NVFP4) was run on this hardware. The published 90.6-92.1 used other weights and harnesses that are not fully published, so the gap to them is not a measure of quantization (item 12). |
| Code-level questions: the 511-pool slice, `swiglu_limit` on the routed experts, top-k ties between layouts | Read in code, not measured ([investigation](investigations/2026-10-glm53-looping/README.md#open-questions)). |

## Labels

Configurations are named **weights · engine version · speculation**, built from the config files by one rule
([`SCHEMA.md`](SCHEMA.md#labels)). The engine name comes first because engines number their versions independently.

<details>
<summary>What each label means</summary>

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
(0.7.0: EP2 + DCP2, vision on; 0.8.0 onward: EP1 + DCP1 with MLA layer ownership, vision off), so "engine" here means
the release as shipped.

</details>
