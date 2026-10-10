# GLM-5.3-Flash on 2x RTX PRO 6000: what drives non-completion on hard questions (2026-09 to 2026-10)

**Question.** On some hard GPQA Diamond questions, GLM-5.3-Flash served by tpurtell's engine does not finish its
reasoning within the 327,680-token budget: it either repeats itself (a **loop**) or keeps producing varied reasoning
until the budget runs out (an **exhaustion**). What drives that, and which runtime defects affect what the model
computes along the way?

Much of this is not yet conclusive. This page states what the clean data supports, what it only describes, and what is
open, and it will be updated as clean data arrives. On 2026-10-09 the earlier screen rates were withdrawn; see
[section 4](#4-method-note-on-seeds-and-withdrawal-notice). Terms (MLA, kpool, DCP1, MLA ownership, NOPE record, DFlash2,
draft-slot sharing, loop, exhaustion) are defined in the repository's [glossary](../../README.md#glossary).

## 1. Summary

| Conclusion | Strength |
|---|---|
| Under the DCP1 layout of tpurtell 0.8.0 and 0.9.0, decode attention skipped the newest 1-3 tokens at causal lengths up to 2,043 that are not a multiple of 4. The fix ([tpurtell PR #6](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/6)) removes it; it shipped in tpurtell 0.9.1 | **Supported** (index check on the image's own kernels) |
| With the fix, decode agrees much better with prefill re-scoring of the same tokens below 2,044 tokens (mean KL 0.066 without the fix; 0.006-0.010 in three runs with it) | **Supported** for this large effect (one run without the fix; two runs of the 0.9.1 release give the noise floor) |
| The fix's practical effect on answers and tool calls | **Unmeasured**: GPQA +0.5 points on 4bpw TR3 (Brandon), one pass per release; tool-eval-bench TC-80 and TC-88 pass in both repeats with the fix, but ten other scenarios flip between repeats of one build |
| Repeats that share one request seed are not a meaningful sample: question 88 failed 11 of 12 with seed 1234 on every repeat vs 3 of 12 with distinct seeds, same configuration | **Supported** (Fisher p = 0.003); the reason the earlier screen rates were withdrawn |
| On tpurtell 0.9.1, question 88 fails to finish in 1-3 of 12 repeats (8-25%) in every tested arm; question 79 in 9 of 12 (75%) on both tpurtell 0.9.1 and the tpurtell 0.7.0 image at three draft tokens | **Descriptive** (12 independent draws per arm) |
| None of the tested runtime parts (EP2 routed experts, 656-byte NOPE records, MLA ownership tp with draft-slot sharing off, the tpurtell 0.7.0 image at three draft tokens) moved either question at this size | **Descriptive**: preregistered rules returned "no candidate at this size" and "unresolved"; with 12 draws per arm only very large effects (about 40-60 points) were detectable; draft depth, quantization and sampling were not varied |
| The questions screened (88 and 79) were chosen as hard from seed-1234 data; with distinct seeds question 88 fails in 1-3 of 12 draws | **Descriptive**; two questions are not a benchmark-wide rate |
| Loops are stopped far beyond the 2,044-token region where the tail bug acted; question 88 fails by looping, question 79 mostly by exhaustion | **Descriptive** |
| What drives non-completion on these questions | **Open** |

## 2. Defect found and fixed: the DCP1 tail bug

### Mechanism
Read in code and confirmed with the image's own kernels:
- For each decode step the sparse-attention indexer selects up to 2,048 earlier tokens in pools of 4. The newest pool
  is still incomplete: it holds the current token and up to two before it (the **kpool tail**).
- The kpool indexer writes the selected pools to columns 0-2043 and the tail to the fixed columns 2044-2046.
- In the DCP1 branch the selection length becomes min(causal length, 2,048), and every column at or beyond that length is
  then masked out.
- So at causal lengths up to 2,043 that are not a multiple of 4, every MLA layer missed the current token and up to two
  tokens before it. A GPQA prompt is a few hundred tokens (the six used below are 124-365), so three of every four decode
  steps in the first ~1,700-1,900 generated tokens were affected.
- Not affected: DCP2, which compacts its selection, and the dense short-prefill path. Upstream vLLM avoids the case with
  an exact causal fill for short decodes (vllm-project/vllm#53906).

**Design context.** DCP1 with MLA layer ownership is tpurtell's layout choice for 0.8.0 onward, and it frees KV memory:
with the same weights and memory setting the server reported a KV pool of 4,707,515 tokens with this layout, against
3,165,056 for v0.7.0's layout (DCP2, EP2) on the same image. The masking path already existed in the vendored attention
code; only the DCP1 layout exercises it. With the fix, the layout works as designed.

**The fix**, [tpurtell/glm-5.3-flash-ext3-2x-rtx#6](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/6),
compacts the valid entries before the existing mask, in the DCP1 branch only; rows the stock code already handled are
left untouched. It was merged on 2026-10-08 and released in **tpurtell 0.9.1**. The measurements in this section were
taken on a local build of the fix before the merge, which is why that configuration is labelled
`tpurtell 0.9.0 + DCP1 tail fix ≈ 0.9.1`: 0.9.1 ships the fixed attention file byte-for-byte as built from the pull
request, which differs from the local build only in identifier names.

### Index check (deterministic)
Runs `2026-10-07_glm53-flash_k3.25-v0.9.0-dflash3_kpool-tail-index` and its `-tailfix-` twin.

| Selection layout | Without the fix | With the fix |
|---|---|---|
| packed (as the indexer emits) | tail dropped in 9 of 23 cases (lengths 5-7, 1001-1003, 2041-2043), 18 tokens | 0 dropped |
| scattered (synthetic worst case) | tail dropped in 9 cases; 2,062 tokens dropped in all | 0 dropped |

Index rows the stock code already handled are byte-identical between the two images. With packed selections and
lengths up to 2,047, the fixed rows attend exactly tokens 0..L-1, the set the dense short-prefill path attends.

### Decode vs prefill
Prefix caching off, one request at a time, GPQA prompts × 2,600 generated tokens, each position's decode distribution
compared with prefill re-scoring of the same token ids (`protocols/decode-prefill-consistency/v1.md`;
`tools/analyze.py decode-prefill`). Mean KL on prompts 0-5, which every run shares:

| Run | i < 2,044 | i ≥ 2,048 |
|---|---|---|
| tpurtell 0.9.0, speculation off (one run) | 0.0656 | 0.0307 |
| tpurtell 0.9.0 + DCP1 tail fix ≈ 0.9.1, speculation off | 0.0103 | 0.0187 |
| tpurtell 0.9.1, speculation off, run of 2026-10-08 | 0.0077 | 0.0064 |
| tpurtell 0.9.1, speculation off, run of 2026-10-09 (same prompts and seeds) | 0.0060 | 0.0107 |
| tpurtell 0.9.1, DFlash2 ×3 on | 0.0152 | 0.0198 |

- **Below 2,044 tokens** the run without the fix lies six to eleven times above every run with it, and the fix is lower
  in 6 of 6 prompts (sign test p = 0.031). Top-1 agreement 93.8% without the fix, 97.6% with it. This effect is supported.
- **Noise floor.** The two 0.9.1 runs repeat one configuration with the same prompts and seeds. Over all 12 prompts their
  means are 0.0072 and 0.0077 below 2,044 tokens and 0.0053 and 0.0098 from 2,048; single prompts differ by up to about
  6x between them.
- **From 2,048 tokens** the runs with the fix range from 0.006 to 0.019, and there is one run without it (0.031). No
  effect is claimed there.
- **Speculation on** (one run, all 12 prompts: 0.0105 and 0.0115) sits somewhat above both speculation-off runs, within
  the spread above. Descriptive only.
- The i mod 4 split predicted before the measurement appeared in both 0.9.0 builds, so it did not separate them (gate
  record in [`preregistration/`](preregistration)).

### Practical impact: unmeasured
- **GPQA:** 4bpw TR3 (Brandon) scored 84.8% on tpurtell 0.9.1 and 84.3% on 0.9.0, one pass each (+0.5 points; 95% interval
  of the paired difference -4.5 to +5.6). No GPQA run compares 3.25bpw with and without the fix on one release.
- **tool-eval-bench** (88 scenarios, temperature 0, two repeats per build): 157 and 157 of 176 points without the fix,
  159 and 163 with it. TC-80 and TC-88 fail in both repeats without the fix and pass in both with it; ten other
  scenarios change status between repeats of the same build, so two repeats cannot separate these two from that
  variation.
- **Serving probe** (16 prompts, one run each, no noise floor): per-request decode 147.3 vs 146.9 tok/s at concurrency 1 and 62.5 vs
  64.9 at 8; acceptance rate 0.534 vs 0.526 at 1 and 0.529 vs 0.550 at 8; KV capacity unchanged (4,707,515 tokens).

### Related: the kpool fixes
Two upstream vLLM kpool bugs (vllm-project/vllm#57477 and #58454) were present in the tpurtell 0.7.0 and 0.8.0 images.
Upstream's own regression tests fail on both (29 of 33 pass) and pass with the fixes (33 of 33). The fixes were ported
in [tpurtell PR #5](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/5) and shipped in tpurtell 0.9.0
([`../2026-10-glm53-kpool-tail`](../2026-10-glm53-kpool-tail)). The tests ran on the 0.7.0 and 0.8.0 release images, on local builds with the
fixes, and (2026-10-09) on the 0.9.0 and 0.9.1 release images, which pass 33 of 33.

## 3. What clean data shows about non-completion

### Component screen, 2026-10-09 (`hard-prompt-screen/v2`, preregistered)
One question per arm, 12 repeats with distinct request seeds 5001-5012, 12 concurrent requests, a fresh server per arm;
3.25bpw weights, DFlash2 ×3. The plan, decision rules and a dated amendment are in
[`component-screen/`](component-screen). Each count below is out of 12 independent draws of one question; the interval is
the 95% Wilson interval for that question in that configuration. Counts are never pooled across questions.

| Arm | Configuration | Question | Finished | Loop | Exhaust | Failed (95% Wilson) |
|---|---|---|---|---|---|---|
| B | `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3` | 88 | 9 | 3 | 0 | 3 (9-53%) |
| EN | `3.25bpw · tpurtell 0.9.1 · EP2 experts, NOPE records off · DFlash2 ×3` | 88 | 10 | 2 | 0 | 2 (5-45%) |
| EO | `3.25bpw · tpurtell 0.9.1 · EP2 experts, MLA owners tp · DFlash2 ×3, sharing off` | 88 | 11 | 1 | 0 | 1 (1-35%) |
| NO | `3.25bpw · tpurtell 0.9.1 · NOPE records off, MLA owners tp · DFlash2 ×3, sharing off` | 88 | 10 | 2 | 0 | 2 (5-45%) |
| B79 | `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3` | 79 | 3 | 1 | 8 | 9 (47-91%) |
| V79 | `3.25bpw · tpurtell 0.7.0 · DFlash2 ×3` | 79 | 3 | 3 | 6 | 9 (47-91%) |

- **Question 88, three layout switches** (half-fraction factorial; each switch on in two arms, 24 requests, vs off in 24):
  EP2 routed experts 3 vs 5 failures (Fisher p = 0.70); NOPE records off 4 vs 4 (p = 1.00); MLA ownership tp with
  draft-slot sharing off 3 vs 5 (p = 0.70). The preregistered rule (at least 6 fewer failures and p < 0.10) found no
  candidate at this size. Arms EO and NO also turned draft-slot sharing off (the launcher pairs it with ownership `tp`),
  so ownership and sharing are not separated.
- **Question 79, the tpurtell 0.7.0 image at three draft tokens:** 9 of 12 on both. The preregistered rule's verdict is
  "unresolved". tpurtell 0.7.0 ships with five draft tokens; this arm ran three, so it says nothing about 0.7.0 as
  shipped, and it also differs from B79 in layout, vision and model revision.
- **Held fixed** in every arm: 3.25bpw weights, DFlash2 ×3, temperature 1.0 / top_p 0.95, the 327,680-token budget,
  12 concurrent requests, one question per arm, a fresh server. Draft depth, quantization and sampling were not varied.
- **Power** (`tools/analyze.py screen-power`; two-sided Fisher test, p < 0.05, 80% power): one arm against another on
  question 88 (12 vs 12, from 25% failing) detects only a rise of about 60 points, and no drop at any size; a switch on
  vs off (24 vs 24, from 17%) a rise of about 41 points; question 79 (12 vs 12, from 75%) a drop of about 60 points.
  Only very large effects could show.
- **Selection.** Questions 88 and 79 were chosen as hard from GPQA empty answers and screens that all sent request seed
  1234. With distinct seeds question 88 fails in 1-3 of 12 draws per arm, far less often than those screens suggested.
  Two questions are not a benchmark-wide rate.
- **KV capacity** (server logs: `server_log.jsonl` and each run's summary): 4,707,515 tokens at tpurtell 0.9.1's defaults; 3,992,056 with EP2 and
  NOPE records off; 2,215,158 with EP2 and ownership tp; 1,851,617 with NOPE records off and ownership tp; 2,894,456 for
  the 0.7.0 image. The V79 pool was full for part of its run, so fewer than 12 of its requests ran at once then. No arm
  logged a preemption.
- Comparisons, checked by `tools/verify.py`:
  [`glm53-flash-v091-component-screen-doc88`](../../comparisons/glm53-flash-v091-component-screen-doc88),
  [`glm53-flash-doc79-image-screen`](../../comparisons/glm53-flash-doc79-image-screen).

### Single draws from the earlier screens
The earlier screens (2026-09-30 to 2026-10-07) sent seed 1234 on every repeat, so only repeat 1 of each question from
each configuration's first screen is kept. Each row is one draw per question: it shows that loops and exhaustion occur
in every configuration tested, not how often (`tools/analyze.py screen-single`).

| Configuration | Date | q13 | q79 | q88 | q121 | q127 |
|---|---|---|---|---|---|---|
| `3.25bpw · tpurtell 0.7.0 · DFlash2 ×5` | 09-30 | ok | ok | loop | ok | exhaust |
| `3.25bpw · tpurtell 0.8.0 · DFlash2 ×3` | 09-30 | loop | loop | loop | ok | ok |
| `3.25bpw · tpurtell 0.8.0 · DFlash2 ×1` | 10-04 | ok | loop | ok | ok | exhaust |
| `3.25bpw · tpurtell 0.8.0 + kpool fixes ≈ 0.9.0 · DFlash2 ×3` | 10-04 | ok | ok | loop | ok | ok |
| `3.25bpw · tpurtell 0.7.0 + kpool fixes · DFlash2 ×5` | 10-05 | ok | ok | loop | ok | exhaust |
| `4bpw TR3 (Brandon) · tpurtell 0.8.0 + kpool fixes ≈ 0.9.0 · DFlash2 ×3` | 10-05 | ok | loop | loop | ok | ok |
| `3.25bpw · tpurtell 0.9.0 · DFlash2 ×3` | 10-07 | loop | exhaust | loop | ok | ok |
| `3.25bpw · tpurtell 0.9.0 · 0.7.0 layout (DCP2, EP2) · DFlash2 ×3` | 10-07 | ok | ok | loop | ok | ok |
| `3.25bpw · tpurtell 0.9.0 + DCP1 tail fix ≈ 0.9.1 · DFlash2 ×3` | 10-07 | ok | loop | loop | ok | loop |

`4bpw TR3 (Brandon) · tpurtell 0.7.0 + kpool fixes · DFlash2 ×5` could not run at this concurrency: the weights leave a
437,563-token KV pool and the engine crashed when it filled (run kept as INVALID).

### GPQA Diamond
GPQA is a whole-benchmark record per configuration, not an evaluator of looping: one pass per configuration gives one
draw per question. Pass 1 of each configuration is published (`comparisons/glm53-flash-gpqa-configs`): 0 to 9 of 198
answers came back empty per run, and none of the question-paired comparisons there differs in empty answers at
p < 0.05. Every pass 1 sent request seed 1234, so the configurations share sampler noise. Clean 3-pass GPQA runs with a
distinct request seed per pass are in progress for tpurtell 0.7.0 and 0.9.1, to give three records per recipe for
comparison with other recipes.

## 4. Method note on seeds, and withdrawal notice

The engine draws each request's sampling noise from its request seed. Repeats of one question that send the same seed
therefore draw on the same sampler noise. They still vary: their texts diverge within a few hundred characters,
because the engine is not bitwise reproducible even one request at a time and concurrent batching adds to that. But
that variation is small numerical noise, not fresh sampling, so it is not a statistically meaningful sample of how often a question fails, and the effective sample behind
a rate is far smaller than its request count.

The same configuration, `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3`, question 88, 12 repeats, 12 concurrent, fresh server:

| Request seeds | Failed to finish |
|---|---|
| 5001-5012, one per repeat (arm B) | 3 of 12 |
| 1234 on every repeat (arm S1234, a fixed-seed control) | 11 of 12 |

Fisher p = 0.003 ([`comparisons/glm53-flash-fixed-seed-control`](../../comparisons/glm53-flash-fixed-seed-control)).
The control was added by a dated amendment after the first arms had been seen; its reading rule (9 or more failures: the
fixed seed explains most of the difference) was written before it ran.

> **WITHDRAWAL NOTICE (2026-10-09).** Every repeat of the earlier hard-question screens (`hard-prompt-screen/v0` and
> `v1`, 2026-09-30 to 2026-10-08) sent request seed 1234. Concurrent batching made the repeats vary, but every repeat
> drew on the same sampler noise, so that variation was not statistically meaningful and the effective sample behind each
> rate was far smaller than its request count. The following are withdrawn: the screen failure rates and their intervals; the layout-bisection statistics (as released
> vs the v0.7.0-layout control, with vs without the tail fix, and the verdicts built on them); and the question-level
> observations drawn from those screens. Evidence: question 88 failed 3 of 12 times with distinct seeds and 11 of 12
> with seed 1234 on every repeat, all else equal (above); in the eight earlier fixed-seed screens of 3.25bpw DCP1
> configurations it had failed 62 of 64 repeats. Repeat 1 of each question from each configuration's first screen is
> kept as a single draw. The withdrawn rows remain in the repository's git history (commit `4fbaad2`). The
> preregistration documents of the layout bisection are kept unchanged, marked withdrawn in
> [`preregistration/WITHDRAWN.md`](preregistration/WITHDRAWN.md).

**The rule now** (`protocols/hard-prompt-screen/v2.md`): a distinct request seed per repeat (5000 + repeat), one question
per run, and the question, not the pooled screen, as the unit of analysis. `tools/verify.py` rejects repeated seeds in a
v2 run unless it is labelled a fixed-seed control, keeps only repeat 1 in v0/v1 runs, and rejects comparisons of v0/v1
runs.

## 5. Anatomy of a failure (descriptive)
From the distinct-seed component screen only; the fixed-seed control's traces are listed separately and labelled.
`tools/analyze.py screen-anatomy`.

- **Loop or exhaustion depends on the question.** Question 88: 8 failures in 48 requests, all loops, all caught by the
  early-stop detector. Question 79: 18 failures in 24 requests, 14 exhaustions (varied reasoning to the budget) and 4
  loops, 3 of which ran to the budget without the detector firing (tail compression ratios 0.003-0.107).
- **Where loops are stopped.** The detector checks the last 30,000 reasoning characters every 10,000 characters from
  30,000 on, and stops a request after three consecutive checks below 0.10. A short-period loop is therefore stopped
  within roughly 60,000 characters of its start. The distinct-seed loops were stopped at 340,007-520,009 characters
  (question 88) and 430,008 (question 79): an estimated 106,000-187,000 tokens, using each question's median of
  characters per completion token in its finished requests (3.22 for question 88, 2.29 for question 79). The fixed-seed
  control's 11 loops were stopped at 300,008-760,001 characters (about 93,000-236,000 tokens). Every one of them is far
  beyond the first 2,044 tokens, where the tail bug acted.
- **Detector margin.** No finished or exhausted request was stopped: across the v2 runs the lowest periodic check of a
  finished request was 0.2117 and of an exhausted request 0.2331, against the 0.10 threshold.

## 6. Open questions
Each item states the question and the evidence so far.

1. **What drives non-completion on questions 88 and 79?** Question 88 fails in 1-3 of 12 draws in every tested arm on
   tpurtell 0.9.1; question 79 in 9 of 12 on both tpurtell 0.9.1 and the tpurtell 0.7.0 image at three draft tokens. No
   tested runtime part moved either, at a size where only very large effects could show.
2. **Does tpurtell 0.7.0 as shipped leave fewer empty answers?** In GPQA pass 1 the two tpurtell 0.7.0 builds (DFlash2 ×5)
   left 0 and 2 of 198 answers empty; every later build left 3 to 9. One pass each does not separate this from chance, and
   the question-79 screen ran 0.7.0 at three draft tokens, not the five its GPQA runs used.
3. **Does the DCP1 tail fix change answers, or how often hard questions fail to finish?** The fix changes decode numerics
   in the first 2,044 tokens; the loops observed are stopped far later; GPQA (one pass per release) and tool calling (two
   repeats) show no difference beyond noise.
4. **Why is the engine not bitwise reproducible one request at a time?** Greedy reruns diverge after a median of 318
   characters (serving probe), and two same-seed decode-vs-prefill runs of tpurtell 0.9.1 first differ in their per-position
   values after 1 to 130 generated tokens, with no concurrent requests.
5. **Do these quants cost accuracy?** No higher-precision reference was run on this hardware. The published 90.6-92.1
   used other weights (BF16, NVFP4) and harnesses that are not fully published.
6. **Does the 511-pool slice matter?** Read in code, effect not measured: from 2,048 tokens of context the indexer keeps
   511 of the 512 pools it selected and drops the last one the top-k emits. In the kernels of these releases the top-k
   emits winners first, so the dropped pool is among the lowest-scored; a top-k that emitted in index order would drop
   the newest pool instead.
7. **Does `swiglu_limit` on the routed experts matter?** Read in code, all versions: the checkpoint sets `swiglu_limit`
   10.0; the shared expert clamps, the routed-expert path does not pass the limit.
8. **Do top-k ties matter?** Read in code: the DCP2 merge path uses a stable top-k with lowest-index ties; the DCP1 fused
   path places ties by atomic arrival order, so exact ties can resolve differently between layouts and between batch
   compositions.

## Glossary
See the repository's [glossary](../../README.md#glossary).

## Receipts and how to recompute them
Every number above comes from files in this repository. Server-log figures (KV pool, throughput, acceptance, waiting
requests) come from each run's `server_log.jsonl`, the numeric fields parsed from the engine's log, and `tools/verify.py`
recomputes them into the run's `summary.json`; the log text itself is not published. From the repository root,
standard library only:

```bash
python3 tools/verify.py                                     # recompute every summary.json; check every comparison and label
python3 investigations/2026-10-glm53-looping/recompute.py   # every number on this page, from the rows
python3 tools/analyze.py screen-v2                          # the component screen and its preregistered tests
python3 tools/analyze.py screen-anatomy                     # section 5
python3 tools/analyze.py screen-single                      # the single draws from the earlier screens
python3 tools/analyze.py screen-power                       # minimum detectable differences of the component screen
python3 tools/analyze.py decode-prefill                     # decode vs prefill, every run; noise floor; single-request divergence
```

| Runs (`runs/<id>/`) | Contents | Protocol |
|---|---|---|
| `2026-10-09_glm53-flash_k3.25-v0.9.1-dflash3_screen-doc88`, `…-ep2-nonope-…`, `…-ep2-owntp-…`, `…-nonope-owntp-…_screen-doc88` | Component screen, question 88 | [`hard-prompt-screen/v2`](../../protocols/hard-prompt-screen/v2.md) |
| `2026-10-09_glm53-flash_k3.25-v0.9.1-dflash3_screen-doc79`, `2026-10-09_glm53-flash_k3.25-v0.7.0-dflash3_screen-doc79` | Component screen, question 79 | [`hard-prompt-screen/v2`](../../protocols/hard-prompt-screen/v2.md) |
| `2026-10-09_glm53-flash_k3.25-v0.9.1-dflash3_screen-doc88-seed1234` | Fixed-seed control | [`hard-prompt-screen/v2`](../../protocols/hard-prompt-screen/v2.md) |
| `2026-10-07_glm53-flash_k3.25-v0.9.0-nospec-nocache_decode-prefill` and the `-tailfix-` twin | 15,600 positions each | [`decode-prefill-consistency/v1`](../../protocols/decode-prefill-consistency/v1.md) |
| `2026-10-08_glm53-flash_k3.25-v0.9.1-nospec-nocache_decode-prefill`, `2026-10-09_…` (its repeat), `2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill` | 12 prompts each, tpurtell 0.9.1 | [`decode-prefill-consistency/v1`](../../protocols/decode-prefill-consistency/v1.md) |
| `2026-10-07_glm53-flash_k3.25-v0.9.0-dflash3_kpool-tail-index` and the `-tailfix-` twin | 46 cases each | [`kpool-tail-index/v1`](../../protocols/kpool-tail-index/v1.md) |
| `2026-10-08_glm53-flash_k3.25-v0.9.0-dflash3_tool-eval` and the `-tailfix-` twin | 2 × 88 scenarios | [`tool-eval-bench/v1`](../../protocols/tool-eval-bench/v1.md) |
| `2026-10-08_glm53-flash_k3.25-v0.9.0-dflash3_serving-probe` and the `-tailfix-` twin | 16-prompt speed and acceptance probe | [`serving-probe/v1`](../../protocols/serving-probe/v1.md) |
| `…_screen-v0`, `…_screen-b`, `…_screen-c`, `…_screen-v07pair`, `…_screen-k4`, `…_screen-bisect1`, `…_screen-invalid` | Single draws (repeat 1) from the earlier screens | [`hard-prompt-screen/v1`](../../protocols/hard-prompt-screen/v1.md), [`v0`](../../protocols/hard-prompt-screen/v0.md) |

- Comparisons: [`glm53-flash-v091-component-screen-doc88`](../../comparisons/glm53-flash-v091-component-screen-doc88),
  [`glm53-flash-doc79-image-screen`](../../comparisons/glm53-flash-doc79-image-screen),
  [`glm53-flash-fixed-seed-control`](../../comparisons/glm53-flash-fixed-seed-control),
  [`glm53-flash-v090-tailfix-decode-prefill`](../../comparisons/glm53-flash-v090-tailfix-decode-prefill),
  [`glm53-flash-v091-decode-prefill-repeat`](../../comparisons/glm53-flash-v091-decode-prefill-repeat),
  [`glm53-flash-v091-decode-prefill-speculation`](../../comparisons/glm53-flash-v091-decode-prefill-speculation),
  [`glm53-flash-v090-tailfix-tool-eval`](../../comparisons/glm53-flash-v090-tailfix-tool-eval).
- Clients: [`tools/clients/hard_prompt_screen.py`](../../tools/clients/hard_prompt_screen.py) (v1 and v2),
  [`tools/clients/decode_prefill_consistency.py`](../../tools/clients/decode_prefill_consistency.py),
  [`tools/kernel/kpool_tail_index_check.py`](../../tools/kernel/kpool_tail_index_check.py): published copies of the
  scripts that produced these runs, differing only in local names and in how settings are passed.
- Component screen plan and amendment: [`component-screen/`](component-screen). Layout bisection plan and amendments
  (withdrawn): [`preregistration/`](preregistration).
