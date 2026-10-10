# Non-completion on hard questions: GLM-5.3-Flash on 2x RTX PRO 6000 (2026-09 to 2026-10)

On some hard GPQA Diamond questions the model does not finish its reasoning within the 327,680-token budget. It either
repeats itself (a **loop**) or keeps reasoning until the budget runs out (an **exhaustion**). This page asks what drives
that, and records an engine issue found along the way.

**Fast path:** [the DCP1 tail issue and its fix](#the-dcp1-tail-issue-and-its-fix) ·
[non-completion](#non-completion-what-the-clean-data-shows) · [open questions](#open-questions) ·
[recompute](#receipts-and-how-to-recompute-them) ·
[charts](https://urbanastrola.github.io/local-inference-evals/looping.html) · [ledger](../../LEDGER.md) ·
[confounds](../../CONFOUNDS.md) · [glossary](../../README.md#glossary)

<a id="1-summary"></a>

## Answer

| Finding | Grade |
|---|---|
| The DCP1 layout of tpurtell 0.8.0 and 0.9.0 skipped the newest 1-3 tokens in decode attention at causal lengths up to 2,043 not divisible by 4. The fix ([tpurtell PR #6](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/6)) removes it; it shipped in tpurtell 0.9.1. | **Supported** |
| With the fix, decode agrees much better with prefill below 2,044 tokens: mean KL 0.066 → 0.006-0.010. | **Supported** |
| The fix's effect on answers and tool calls is within noise at the sizes run. | **Descriptive** (effect unmeasured) |
| On tpurtell 0.9.1, question 88 fails in 1-3 of 12 draws in every tested arm; question 79 in 9 of 12, on 0.9.1 and on the 0.7.0 image at three draft tokens. | **Descriptive** |
| No tested runtime part moved either question. Only very large effects (about 40-60 points) could have shown. | **Descriptive** |
| Loops are stopped far beyond the first 2,044 tokens, where the tail issue acted. Question 88 loops; question 79 mostly exhausts. | **Descriptive** |
| Across GPQA, tpurtell 0.7.0 as shipped left fewer questions unanswered than 0.9.1: 4 vs 16 of 594 (3.25bpw, three passes each). | **Supported** (clustered p = 0.009) |
| What drives non-completion, and which of 0.7.0's differences from 0.9.1 matters. | **Open** |

<details>
<summary>Qualifications</summary>

- Decode vs prefill: one run without the fix; two runs of the 0.9.1 release give the noise floor.
- Effect on answers: GPQA +0.5 points on 4bpw TR3 (Brandon), one pass per release, both at 8 concurrent requests with a
  KV pool that holds about 4. tool-eval-bench TC-80 and TC-88 pass in both repeats with the fix, but ten other
  scenarios flip between repeats of one build.
- Component screen: 12 independent draws per arm. Preregistered rules returned "no candidate at this size" and
  "unresolved". Draft depth, quantization and sampling were not varied.
- Questions 88 and 79 were chosen as hard from seed-1234 data; two questions are not a benchmark-wide rate.
- 0.7.0 vs 0.9.1: the question was raised by pass 1; passes 2-3 alone give 4 vs 11 empty answers on 1 vs 7 questions
  (p = 0.06).
- Repeats that share one request seed are not a meaningful sample: question 88 failed 11 of 12 with seed 1234 on every
  repeat vs 3 of 12 with distinct seeds (supported, Fisher p = 0.003). This is why the earlier screens were withdrawn
  and replaced ([ledger](../../LEDGER.md#withdrawn-and-what-replaced-it)).

</details>

<a id="2-defect-found-and-fixed-the-dcp1-tail-bug"></a>

## The DCP1 tail issue and its fix

**What happened.** For each decode step, the sparse-attention indexer picks up to 2,048 earlier tokens in pools of 4.
The newest pool is incomplete: the current token and up to two before it (the **kpool tail**). In the DCP1 layout, the
tail's columns were masked out at causal lengths up to 2,043 not divisible by 4. So every MLA layer missed those tokens.

**Design context.** DCP1 with MLA layer ownership is tpurtell's layout from 0.8.0 on. It holds about 1.5x the KV cache
of v0.7.0's DCP2 layout: 4,707,515 tokens against 3,165,056 on the same image. The masking path already existed in
the vendored attention code; only this layout exercises it. With the fix, the layout works as designed.

**The fix.** [tpurtell/glm-5.3-flash-ext3-2x-rtx#6](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/6)
compacts the valid entries before the existing mask, in the DCP1 branch only. Merged 2026-10-08; released in
**tpurtell 0.9.1**.

<details>
<summary>Mechanism in detail</summary>

- The kpool indexer writes the selected pools to columns 0-2043 and the tail to the fixed columns 2044-2046.
- In the DCP1 branch the selection length becomes min(causal length, 2,048); every column at or beyond it is masked.
- A GPQA prompt is a few hundred tokens (the six used below are 124-365), so three of every four decode steps in the
  first ~1,700-1,900 generated tokens were affected.
- Not affected: DCP2, which compacts its selection, and the dense short-prefill path. Upstream vLLM avoids the case
  with an exact causal fill for short decodes (vllm-project/vllm#53906).
- The fix leaves rows the stock code already handled untouched.
- Measurements were taken on a local build of the fix before the merge, labelled
  `tpurtell 0.9.0 + DCP1 tail fix ≈ 0.9.1`. 0.9.1 ships the fixed attention file byte-for-byte as built from the pull
  request, which differs from the local build only in identifier names.

</details>

### Index check: supported

Runs `2026-10-07_glm53-flash_k3.25-v0.9.0-dflash3_kpool-tail-index` and its `-tailfix-` twin.

| Selection layout | Without the fix | With the fix |
|---|---|---|
| packed (as the indexer emits) | tail dropped in 9 of 23 cases (lengths 5-7, 1001-1003, 2041-2043), 18 tokens | 0 dropped |
| scattered (synthetic worst case) | tail dropped in 9 cases; 2,062 tokens dropped in all | 0 dropped |

Rows the stock code already handled are byte-identical between the two images. With packed selections and lengths up
to 2,047, the fixed rows attend exactly tokens 0..L-1, as the dense short-prefill path does.

### Decode vs prefill: supported below 2,044 tokens

Each position's decode distribution compared with prefill re-scoring of the same tokens. Mean KL on prompts 0-5, which
every run shares (lower is closer):

| Run | i < 2,044 | i ≥ 2,048 |
|---|---|---|
| tpurtell 0.9.0, speculation off (one run) | 0.0656 | 0.0307 |
| tpurtell 0.9.0 + DCP1 tail fix ≈ 0.9.1, speculation off | 0.0103 | 0.0187 |
| tpurtell 0.9.1, speculation off, 2026-10-08 | 0.0077 | 0.0064 |
| tpurtell 0.9.1, speculation off, 2026-10-09 (same prompts and seeds) | 0.0060 | 0.0107 |
| tpurtell 0.9.1, DFlash2 ×3 on | 0.0152 | 0.0198 |

- **Below 2,044 tokens:** without the fix, six to eleven times above every run with it. Lower with the fix in 6 of 6
  prompts (sign test p = 0.031). Top-1 agreement 93.8% → 97.6%.
- **From 2,048 tokens:** runs with the fix range 0.006-0.019; one run without it (0.031). No effect claimed.

<details>
<summary>Noise floor, speculation, method</summary>

- The two 0.9.1 runs repeat one configuration with the same prompts and seeds. Over all 12 prompts their means are
  0.0072 and 0.0077 below 2,044 tokens and 0.0053 and 0.0098 from 2,048; single prompts differ by up to about 6x.
- Speculation on (one run, all 12 prompts: 0.0105 and 0.0115) sits somewhat above both speculation-off runs, within
  that spread. Descriptive only.
- The i mod 4 split predicted before the measurement appeared in both 0.9.0 builds, so it did not separate them (gate
  record in [`preregistration/`](preregistration)).
- Method: prefix caching off, one request at a time, GPQA prompts × 2,600 generated tokens
  (`protocols/decode-prefill-consistency/v1.md`; `tools/analyze.py decode-prefill`).

</details>

<a id="impact"></a>

### Effect on answers: descriptive (effect unmeasured)

- **GPQA:** 4bpw TR3 (Brandon) 84.8% on 0.9.1 vs 84.3% on 0.9.0, one pass each (+0.5 points; 95% interval -4.5 to
  +5.6). Passes of one configuration differ by 0.5-3.5 points.
- **tool-eval-bench** (88 scenarios, temperature 0, two repeats per build): 157 and 157 of 176 points without the fix,
  159 and 163 with it.
- **Speed:** no cost in single runs.

<details>
<summary>Detail</summary>

- GPQA: both runs at 8 concurrent requests with a KV pool that holds about 4 requests at the token cap (the 0.9.1 run's
  log shows requests waiting for KV). No GPQA run compares 3.25bpw with and without the fix on one release.
- tool-eval-bench: TC-80 and TC-88 fail in both repeats without the fix and pass in both with it. Ten other scenarios
  change status between repeats of the same build, so two repeats cannot separate these two from that variation.
- Serving probe (16 prompts, one run each, no noise floor): per-request decode 147.3 vs 146.9 tok/s at concurrency 1
  and 62.5 vs 64.9 at 8; acceptance rate 0.534 vs 0.526 at 1 and 0.529 vs 0.550 at 8; KV capacity unchanged
  (4,707,515 tokens).

</details>

**Related: the kpool fixes.** Two upstream vLLM kpool issues (vllm-project/vllm#57477 and #58454) affected the
tpurtell 0.7.0 and 0.8.0 images: upstream's tests pass 29 of 33 there, 33 of 33 with the fixes. Ported in
[tpurtell PR #5](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/5), shipped in 0.9.0. The 0.9.0 and 0.9.1
release images pass 33 of 33 ([`../2026-10-glm53-kpool-tail`](../2026-10-glm53-kpool-tail)).

## Non-completion: what the clean data shows

### Component screen, 2026-10-09 (preregistered)

One question per arm, 12 draws with distinct request seeds 5001-5012, 3.25bpw weights, DFlash2 ×3. Counts are never
pooled across questions.

| Arm | Configuration | Question | Finished | Loop | Exhaust | Failed (95% Wilson) |
|---|---|---|---|---|---|---|
| B | `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3` | 88 | 9 | 3 | 0 | 3 (9-53%) |
| EN | `3.25bpw · tpurtell 0.9.1 · EP2 experts, NOPE records off · DFlash2 ×3` | 88 | 10 | 2 | 0 | 2 (5-45%) |
| EO | `3.25bpw · tpurtell 0.9.1 · EP2 experts, MLA owners tp · DFlash2 ×3, sharing off` | 88 | 11 | 1 | 0 | 1 (1-35%) |
| NO | `3.25bpw · tpurtell 0.9.1 · NOPE records off, MLA owners tp · DFlash2 ×3, sharing off` | 88 | 10 | 2 | 0 | 2 (5-45%) |
| B79 | `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3` | 79 | 3 | 1 | 8 | 9 (47-91%) |
| V79 | `3.25bpw · tpurtell 0.7.0 · DFlash2 ×3` | 79 | 3 | 3 | 6 | 9 (47-91%) |

- **Question 88:** no layout switch met the preregistered rule ("no candidate at this size").
- **Question 79:** the 0.7.0 image at three draft tokens failed as often as 0.9.1. Verdict: "unresolved".

<details>
<summary>Tests, power and scope</summary>

- **Question 88, three switches** (half-fraction factorial; each switch on in 24 requests, off in 24): EP2 routed
  experts 3 vs 5 failures (Fisher p = 0.70); NOPE records off 4 vs 4 (p = 1.00); MLA ownership tp with draft-slot
  sharing off 3 vs 5 (p = 0.70). The rule needed at least 6 fewer failures and p < 0.10. Arms EO and NO also turned
  draft-slot sharing off (the launcher pairs it with ownership `tp`), so ownership and sharing are not separated.
- **Question 79:** 9 of 12 on both. tpurtell 0.7.0 ships with five draft tokens; this arm ran three, so it says nothing
  about 0.7.0 as shipped. It also differs from B79 in layout and vision.
- **Held fixed** in every arm: 3.25bpw weights, DFlash2 ×3, temperature 1.0 / top_p 0.95, the 327,680-token budget,
  12 concurrent requests, one question per arm, a fresh server. Draft depth, quantization and sampling were not varied.
- **Power** (`tools/analyze.py screen-power`; two-sided Fisher, p < 0.05, 80% power): one arm against another on
  question 88 (12 vs 12, from 25% failing) detects only a rise of about 60 points, and no drop at any size; a switch on
  vs off (24 vs 24, from 17%) a rise of about 41 points; question 79 (12 vs 12, from 75%) a drop of about 60 points.
- **Selection.** Questions 88 and 79 were chosen as hard from GPQA empty answers and screens that all sent request
  seed 1234. With distinct seeds question 88 fails in 1-3 of 12 draws per arm, far less often than those screens
  suggested.
- **KV capacity** (server logs): 4,707,515 tokens at tpurtell 0.9.1's defaults; 3,992,056 with EP2 and NOPE records
  off; 2,215,158 with EP2 and ownership tp; 1,851,617 with NOPE records off and ownership tp; 2,894,456 for the 0.7.0
  image. The V79 pool was full for part of its run, so fewer than 12 of its requests ran at once then. No arm logged a
  preemption.
- Plan, decision rules and a dated amendment: [`component-screen/`](component-screen). Comparisons:
  [`glm53-flash-v091-component-screen-doc88`](../../comparisons/glm53-flash-v091-component-screen-doc88),
  [`glm53-flash-doc79-image-screen`](../../comparisons/glm53-flash-doc79-image-screen).

</details>

### How the failures look: descriptive

- **Loop or exhaustion depends on the question.** Question 88: 8 failures in 48 requests, all loops. Question 79: 18
  failures in 24 requests, 14 exhaustions and 4 loops.
- **Loops are stopped late.** An estimated 106,000-187,000 tokens in, far beyond the first 2,044 tokens where the tail
  issue acted.

<details>
<summary>Detail (<code>tools/analyze.py screen-anatomy</code>)</summary>

- Distinct-seed component screen only; the fixed-seed control is listed separately.
- All question-88 loops were caught by the early-stop detector. Of the 4 question-79 loops, 3 ran to the budget without
  the detector firing (tail compression ratios 0.003-0.107).
- The detector checks the last 30,000 reasoning characters every 10,000 characters from 30,000 on, and stops a request
  after three consecutive checks below 0.10. A short-period loop is stopped within roughly 60,000 characters of its
  start.
- Distinct-seed loops were stopped at 340,007-520,009 characters (question 88) and 430,008 (question 79). Tokens are
  estimated with each question's median characters per completion token in its finished requests (3.22 for question 88,
  2.29 for question 79).
- The fixed-seed control's 11 loops were stopped at 300,008-760,001 characters (about 93,000-236,000 tokens).
- Detector margin: no finished or exhausted request was stopped. The lowest periodic check of a finished request was
  0.2117, of an exhausted request 0.2331, against the 0.10 threshold.

</details>

### Across GPQA: empty answers in three passes

GPQA is a whole-benchmark record, not a looping test: each pass is one draw per question. Full tables:
[Results page](https://urbanastrola.github.io/local-inference-evals/#gpqa),
[`comparisons/glm53-flash-gpqa-records`](../../comparisons/glm53-flash-gpqa-records).

| Configuration | Empty answers, passes 1 / 2 / 3 | Of 594 | Questions ever empty | Raw accuracy, mean of 3 |
|---|---|---|---|---|
| `3.25bpw · tpurtell 0.7.0 · DFlash2 ×5` | 0 / 3 / 1 | 4 | 4 | 88.2% |
| `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3` | 5 / 5 / 6 | 16 | 11 | 86.9% |
| `4bpw TR3 (Brandon) · tpurtell 0.9.1 · DFlash2 ×3 · concurrency 4` | 8 / 2 / 7 | 17 | 14 | 85.7% |

- **0.7.0 vs 0.9.1 (3.25bpw): supported.** 0.9.1 had more empty answers on 10 questions and fewer on 1 (clustered
  p = 0.009). Accuracy does not differ measurably (clustered p = 0.41).
- **Why is open.** The two differ in several ways at once ([confounds](../../CONFOUNDS.md#7-several-changes-between-releases-at-once)).
- **Weights on 0.9.1: descriptive.** 3.25bpw and 4bpw TR3 (Brandon) leave about as many unanswered (16 vs 17).

<details>
<summary>Detail (<code>tools/analyze.py gpqa-passes</code>, <code>gpqa-records</code>, <code>gpqa-empty</code>)</summary>

- The clustered test is an exact sign-flip test over questions, each question's difference summed over its three
  passes. Difference +2.0 points (95% interval over questions +0.7 to +3.5).
- The question was raised by pass 1 (0 vs 5); passes 2 and 3 alone, run after it was raised, give 4 vs 11 empty answers
  (1 vs 7 questions, p = 0.06).
- 0.7.0 differs from 0.9.1 in draft depth (5 vs 3), parallel layout (DCP2 with EP2 experts vs DCP1 with MLA layer
  ownership), kernel and engine code, vision, and KV pool (2,758,919 vs 4,707,515 tokens). The two model revisions
  share the same weight files. Their chat templates differ (0.7.0 served the checkpoint's own template, 0.9.1 the corrected Z.ai template), but both render all 198 GPQA prompts byte-identically. The
  component screen did not test 0.7.0 at its shipped five draft tokens.
- Weights on 0.9.1: 9 questions each way; clustered p = 1.00. The 4bpw record ran 4 requests at once so that no request
  waited for KV memory.
- Which questions: 79 and 81 came back empty in all three records; 88, 127 and 147 in both 0.9.1 records and in no pass
  of 0.7.0; question 88 in two of three passes of each 0.9.1 record. Where a request log exists, every empty answer but
  one ran to the 327,680-token cap.

</details>

<a id="4-method-note-on-seeds-and-withdrawal-notice"></a>

### Seeds and the earlier screens

The earlier screens (2026-09-30 to 10-08) sent seed 1234 on every repeat. Their rates were withdrawn on 2026-10-09 and
replaced by the component screen above. Notice, evidence and the single draws kept from them:
[ledger](../../LEDGER.md#withdrawn-and-what-replaced-it). Why a shared seed is not a sample:
[confounds](../../CONFOUNDS.md#1-shared-request-seed-across-repeats).

<a id="6-open-questions"></a>

## Open questions

Each states the question and the evidence so far.

1. **What drives non-completion on questions 88 and 79?** Question 88 fails in 1-3 of 12 draws in every tested arm on
   tpurtell 0.9.1; question 79 in 9 of 12 on 0.9.1 and on the 0.7.0 image at three draft tokens. No tested runtime part
   moved either, at a size where only very large effects could show.
2. **What makes tpurtell 0.7.0 as shipped leave fewer GPQA questions unanswered than 0.9.1?** 4 vs 16 of 594 over three
   passes, on 4 vs 11 questions (clustered p = 0.009; passes 2 and 3 alone p = 0.06). Draft depth, layout, kernel and
   engine code, vision and KV pool differ at once; none was varied alone in GPQA. The question-79 screen ran the 0.7.0
   image at three draft tokens, not the five it ships with, and failed as often as 0.9.1. Which difference matters is
   inconclusive.
3. **Does the DCP1 tail fix change answers, or how often hard questions fail to finish?** It changes decode numerics in
   the first 2,044 tokens; the loops observed are stopped far later; GPQA (one pass per release) and tool calling (two
   repeats) show no difference beyond noise.
4. **Why is the engine not bitwise reproducible one request at a time?** Greedy reruns diverge after a median of 318
   characters; two same-seed decode-vs-prefill runs of 0.9.1 first differ after 1 to 130 generated tokens.
5. **Do these quants cost accuracy?** No higher-precision reference was run on this hardware. The published 90.6-92.1
   used other weights (BF16, NVFP4) and harnesses that are not fully published.

<details>
<summary>Three code-level questions (read in code, not measured)</summary>

6. **Does the 511-pool slice matter?** From 2,048 tokens of context the indexer keeps 511 of the 512 pools it selected
   and drops the last one the top-k emits. In these releases the top-k emits winners first, so the dropped pool is
   among the lowest-scored; a top-k that emitted in index order would drop the newest pool instead.
7. **Does `swiglu_limit` on the routed experts matter?** All versions: the checkpoint sets `swiglu_limit` 10.0; the
   shared expert clamps, the routed-expert path does not pass the limit.
8. **Do top-k ties matter?** The DCP2 merge path uses a stable top-k with lowest-index ties; the DCP1 fused path places
   ties by atomic arrival order, so exact ties can resolve differently between layouts and between batch compositions.

</details>

## Receipts and how to recompute them

<details>
<summary>Commands, runs and files</summary>

Every number above comes from files in this repository. Server-log figures (KV pool, throughput, acceptance, waiting
requests) come from each run's `server_log.jsonl`, the numeric fields parsed from the engine's log; `tools/verify.py`
recomputes them into the run's `summary.json`. The log text itself is not published. From the repository root,
standard library only:

```bash
python3 tools/verify.py                                     # recompute every summary.json; check every comparison and label
python3 investigations/2026-10-glm53-looping/recompute.py   # every number on this page, from the rows
python3 tools/analyze.py screen-v2                          # the component screen and its preregistered tests
python3 tools/analyze.py screen-anatomy                     # how the failures look
python3 tools/analyze.py screen-single                      # the single draws from the earlier screens
python3 tools/analyze.py screen-power                       # minimum detectable differences of the component screen
python3 tools/analyze.py decode-prefill                     # decode vs prefill, every run; noise floor; single-request divergence
python3 tools/analyze.py gpqa-passes                        # GPQA, three passes per configuration
python3 tools/analyze.py gpqa-records                       # GPQA records compared with questions as clusters
python3 tools/analyze.py gpqa-empty                         # which questions came back empty, in which passes
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
| `2026-09-29_glm53-flash_k3.25-v0.7.0-dflash5_gpqa-diamond`, `2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3_gpqa-diamond`, `2026-10-09_glm53-flash_k4-v0.9.1-dflash3-c4_gpqa-diamond` | GPQA Diamond, three passes each (request seeds 1234-1236) | [`gpqa-diamond/v1`](../../protocols/gpqa-diamond/v1.md) |

- Comparisons: [`glm53-flash-gpqa-records`](../../comparisons/glm53-flash-gpqa-records),
  [`glm53-flash-v091-component-screen-doc88`](../../comparisons/glm53-flash-v091-component-screen-doc88),
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

</details>
