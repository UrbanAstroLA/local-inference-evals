# Confounds

Things other than the configuration that can move an accuracy, completion or speed result. For each: what it does, how
the runs control for it now, and its status. Only confounds this repository has data on are listed.

| Confound | What it can do | Control now | Status |
|---|---|---|---|
| [Shared request seed](#1-shared-request-seed-across-repeats) | Makes repeats look like a sample when they are not | One request seed per repeat and per GPQA pass | Controlled |
| [Answer extraction](#2-answer-extraction-flexible-extract) | Scores some correct answers as wrong | Audited stated-answer score beside every raw score | Measured |
| [KV saturation](#3-kv-saturation) | Fewer requests run at once than the client sends | 4bpw record at 4 concurrent requests | Controlled for the record |
| [Pass-to-pass noise](#4-pass-to-pass-noise) | One pass varies by 0.5 to 3.5 points | Three passes per record; question-clustered tests | Measured |
| [Nondeterminism](#5-batching-and-single-request-nondeterminism) | Identical requests give different outputs | Compare outcomes over many draws, never transcripts | Measured; cause open |
| [Question selection](#6-hard-questions-chosen-from-seed-1234-data) | Hard questions look harder than they are | Rates per question; no benchmark-wide claim | Stated limit |
| [Changes bundled in a release](#7-several-changes-between-releases-at-once) | A release difference cannot be pinned to one change | Labels name the release as shipped; component screen; completion investigation | Open |
| [Published scores](#8-published-scores-use-other-weights-and-harnesses) | Gaps mix quantization, harness, scoring and runtime | Context only, never ranked | Stated limit |
| [Requests in flight and the top-k route](#9-requests-in-flight-and-the-top-k-route) | Decides which pool the 511-pool slice discards, so what the model attends | Concurrency fixed per protocol and stated in labels; the completion arm matched its comparator's 8 | Measured; no measurable change in completion with the correction (one arm) |
| [Kpool tail-ring positions omission](#10-kpool-tail-ring-positions-omission) | Other requests' keys enter completed pools, more with more requests in flight | Present in every record compared here; fix prepared, in no GPQA run | Measured; effect open |
| [NaN-byte index rows](#11-nan-byte-index-rows) | Selected pools whose stored bytes are not valid keys | Reported separately by the gate | Measured in one run |
| [Historical control](#12-historical-control-for-the-completion-arm) | A later run differs from a recorded baseline in more than the change | Comparator audit; analysis frozen before data | Stated limit |
| [Router precision and routed-expert clamp](#13-router-precision-and-the-routed-expert-swiglu-clamp) | Numerical departures from upstream vLLM in every release | None | Untested |
| [Graph vs eager execution](#14-graph-vs-eager-execution-for-instrumentation) | A probe that cannot run under CUDA graphs measures in eager mode | Gate and tail-ring runs checked under the shipped graphs | Controlled for the gate |
| [Post-completion continuation](#15-long-context-decode-vs-prefill-measures-continuation-after-the-reply) | A decode-vs-prefill bin can lie entirely after the reply ended | Stated with the measure; not used as evidence about completion | Stated limit |

## 1. Shared request seed across repeats

**Effect.** The engine draws sampling noise from the request seed. Repeats with one seed draw on the same noise.
They still differ, through batching and nondeterminism, but that is not a meaningful sample of how often a question fails.

**Control.** Screens: a distinct seed per repeat (5000 + repeat). GPQA: pass *p* sends 1233 + *p*.
`tools/verify.py` rejects shared seeds.

**Status.** Controlled. Earlier shared-seed results were withdrawn and replaced ([ledger](LEDGER.md#withdrawn-and-what-replaced-it)).

<!-- figure:seed-control -->

<details>
<summary>Evidence: supported (Fisher p = 0.003)</summary>

Same configuration (`3.25bpw · tpurtell 0.9.1 · DFlash2 ×3`), question 88, 12 repeats, 12 concurrent, fresh server:

| Request seeds | Failed to finish |
|---|---|
| 5001-5012, one per repeat (arm B) | 3 of 12 |
| 1234 on every repeat (arm S1234, a fixed-seed control) | 11 of 12 |

The control was added by a dated amendment after the first arms had been seen. Its reading rule (9 or more failures:
the fixed seed explains most of the difference) was written before it ran
([`comparisons/glm53-flash-fixed-seed-control`](comparisons/glm53-flash-fixed-seed-control)).

</details>

## 2. Answer extraction (flexible-extract)

**Effect.** lm-eval's raw `flexible-extract` takes the last parenthesised capital letter in a reply. A reply that
names other options or uses chemistry notation after its answer is misread. Net effect: raw scores run 0.5 to 3.5 points
low per pass, 1.5 on average.

**Control.** An audited stated-answer score (`correct_stated`) is published per row beside the raw score. Raw stays the
headline so runs stay comparable.

**Status.** Measured. The misread is deterministic: rerunning never reveals it.

<details>
<summary>Detail</summary>

- Over the 16 published passes: 49 correct answers scored wrong, 3 wrong stated answers credited.
- Causes: labels of other options mentioned after the answer, chemistry notation such as (H) or (R), answers given as a
  boxed or bold letter.
- The stated-answer judgement rests on reply text, which is not published; the per-row judgement is
  ([`protocols/gpqa-diamond/v1.md`](protocols/gpqa-diamond/v1.md)).

</details>

## 3. KV saturation

**Effect.** When running requests fill the KV pool, the rest wait, so fewer run at once than the client sends.
4bpw TR3 (Brandon) leaves a 1,377,179-token pool: 4.17 requests at the token cap.

**Control.** Its three-pass record runs 4 requests at once, labelled `concurrency 4`. No request waited.

**Status.** Controlled for the record. Earlier 4bpw runs at 8 are kept and labelled.

<details>
<summary>Evidence</summary>

- tpurtell 0.9.1 at 8 concurrent: requests waited for KV in 653 of 1,417 status lines. At 4: none waited (peak usage
  91.7%, no preemption).
- Pass 1 at 8 vs at 4, same configuration: 13 / 13 questions only one got right (p = 1.00), 0.0 points, empty answers
  3 / 5 (p = 0.73). No difference detected in one pass.
- The 4bpw runs on 0.8.0 and 0.9.0 ran at 8 and kept no server log; whether requests waited is not recorded.
- On tpurtell 0.7.0 + kpool fixes the 4bpw pool was 437,563 tokens; the engine crashed when it filled at 8 concurrent
  requests (run kept as INVALID).
- Component screen arm V79: the pool was full for part of the run.

</details>

## 4. Pass-to-pass noise

**Effect.** One configuration's passes, each with its own seed, differ by 0.5 to 3.5 points. That is as wide as pass 1
across all ten configurations (84.3-87.9%).

**Control.** Three passes per record. Questions are the unit: a question answered three times counts once.

**Status.** Measured (descriptive).

<details>
<summary>What it limits</summary>

- One pass per configuration resolves only differences of about 5 points.
- Three passes: accuracy differences smaller than about 3 points are neither shown nor excluded.
- Of 198 questions, 154-160 are right in all three passes of a configuration, 10-12 in none, 28-32 vary.

</details>

## 5. Batching and single-request nondeterminism

**Effect.** The engine is not bitwise reproducible, even one request at a time. Batching adds more variation at 8 or 12
concurrent requests.

**Control.** Compare outcomes over many draws and questions, never single transcripts. Repeat a measurement to get its
noise floor.

**Status.** Measured (supported). Why single requests vary is open.

<!-- figure:greedy -->

<details>
<summary>Evidence</summary>

- Greedy reruns of one configuration, one request at a time, diverge after a median of 318 characters
  ([`comparisons/glm53-flash-serving-probe`](comparisons/glm53-flash-serving-probe)).
- Two decode-vs-prefill runs of tpurtell 0.9.1, same prompts and seeds, one request at a time, first differ after 1 to
  130 generated tokens. Their means differ by up to about 2x; single prompts by up to about 6x
  ([`comparisons/glm53-flash-v091-decode-prefill-repeat`](comparisons/glm53-flash-v091-decode-prefill-repeat)).
- So greedy parity cannot certify that speculative decoding is exact on this stack.

</details>

## 6. Hard questions chosen from seed-1234 data

**Effect.** Questions 88 and 79 were picked as hard from runs that all sent seed 1234. With distinct seeds, question 88
fails in 1-3 of 12 draws, far less often than those runs suggested.

**Control.** Screen results are reported per question, never pooled.

**Status.** Stated limit. Two questions are not a benchmark-wide rate.

## 7. Several changes between releases at once

**Effect.** tpurtell 0.7.0 as shipped left fewer GPQA questions unanswered than 0.9.1 (4 vs 16 of 594). But the two
differ in several ways at once.

**Control.** Labels name the release as shipped. The component screen changed single layout parts on two questions.

**Status.** Open. None of the differences was varied alone in GPQA.

<details>
<summary>What differs, and what was tested</summary>

- Draft depth: 5 vs 3 tokens.
- Parallel layout: DCP2 with EP2 experts vs DCP1 with MLA layer ownership.
- Kernel and engine code; vision on vs off.
- KV pool: 2,758,919 vs 4,707,515 tokens.
- The two model revisions share the same weight files. Their chat templates differ (0.7.0 served the checkpoint's own template, 0.9.1 the corrected Z.ai template), but both render all 198 GPQA prompts byte-identically.
- Component screen, at three draft tokens: no tested layout part moved question 88 on 0.9.1, and the 0.7.0 image failed
  question 79 as often as 0.9.1 (9 of 12 each). Arms that set MLA ownership `tp` also turned draft-slot sharing off,
  so those two are not separated.
- Draft depth, quantization and sampling were not varied.
- Since 2026-10-10 the [completion investigation](investigations/2026-10-glm53-completion/README.md) has compared the two
  engines' decode against their own prefill (Phase 1), run a completion pilot and a replication of 0.9.1's default
  layout against 0.7.0's layout bundle on 0.9.1 (no useful separation), and audited the attention, DCP and indexer
  source.
- It found two localized defects: the column-511 drop ([9](#9-requests-in-flight-and-the-top-k-route)), the only one
  found that differs between the engines in the direction of the gap, and the tail-ring omission
  ([10](#10-kpool-tail-ring-positions-omission)), which affects 0.7.0 more; that does not exclude an accidental
  completion benefit from incorrect computation. A GPQA arm with the column-511 correction
  alone left 15 empty replies vs 16 on 0.9.1, so that defect is not supported as the explanation of the gap
  ([completion arm](investigations/2026-10-glm53-completion/README.md#completion-arm)). Whether the tail ring accounts
  for any of it is untested; what produces the gap is open.

</details>

## 8. Published scores use other weights and harnesses

**Effect.** Model-card GPQA scores (92.1 NVIDIA, 90.6 Red Hat) used BF16 or NVFP4 weights and harnesses not fully
published.

**Control.** Shown as context only, never ranked against local runs.

**Status.** Stated limit. No higher-precision reference was run on this hardware, so these runs cannot separate
quantization, harness, scoring and runtime effects.

## 9. Requests in flight and the top-k route

**Effect.** For each decode row the sparse-attention indexer's top-k selects 512 compressed-key pools, and this engine
keeps 511 of them (columns 0-510) to make room for the newest tokens. No top-k path sorts its output, so which pool is
discarded depends on the kernel route. On 0.9.1 the route follows the number of decode rows in the batch: up to 16 rows
(1-4 requests in flight at 3 draft tokens) take the fused route, more rows the tiled route. On real inputs the tiled
route discarded a pool ranked better than 480 of 512 in 88.1% of live rows, the fused route in 0.5%, and 0.7.0's path
on the same bytes in 6.9%. Concurrency therefore changes what the model attends. It is a confound for any comparison of
GLM-5.3 engines that use B12x top-k kernels: runs at different concurrency, or engines whose route switches at
different batch sizes, differ in this as well.

**Control.** Concurrency is fixed by each protocol and shown in labels when a run departs from it
([3](#3-kv-saturation)). The completion arm ran at 8, as its comparator did; its frozen analysis approximates route
exposure from client-side in-flight counts, which does not measure the in-engine route (76-84% of busy time with 5 or
more requests in flight). The column-511 correction
always discards the lowest-scoring selected pool, so the discarded pool no longer depends on the route.

**Status.** Measured: a localized defect (supported). A three-pass GPQA arm with the correction left 15 empty replies vs
16 of 594 (one-sided p = 0.50): a correctness fix with no measurable change in completion
([completion arm](investigations/2026-10-glm53-completion/README.md#completion-arm)).

<details>
<summary>Evidence</summary>

| Path | When it runs | Synthetic inputs, 1,024-16,384 pools | Real inputs, live rows |
|---|---|---|---|
| 0.9.1 tiled | more than 16 rows: 5-8 requests in flight at 3 draft tokens | 94-98% | 88.1% of 9,944 |
| 0.9.1 fused | up to 16 rows: 1-4 requests in flight | 0-1% | 0.5% of 1,400 |
| 0.7.0 per rank, then cross-rank merge | every decode step | 3-21% | 6.9% (replayed on the same bytes) |

- Share of rows whose discarded pool ranked better than 480 of 512. The route alone decides it: forcing fused-captured
  rows through the tiled path gives 92%; splitting tiled batches into 16-row halves gives 0.4%.
- On real inputs: 96-97% from 4,000 to 64,000 tokens of context, 13% at about 120,000.
- In the 0.9.1 GPQA record, 5 or more requests were in flight for 82-95% of each pass's busy time (client-side,
  approximate; dry run of the frozen analysis on the existing record).
- The gate found the prefill branch affected too: rows with exactly 512 candidates came back in index order, so the
  newest pool was discarded (2,200 rows).
- Receipts: [stage 1](investigations/2026-10-glm53-completion/phase2-stage1),
  [stage 2](investigations/2026-10-glm53-completion/phase2-stage2),
  [phase 3](investigations/2026-10-glm53-completion/phase3), [gate](investigations/2026-10-glm53-completion/gate),
  [dry run](investigations/2026-10-glm53-completion/arm-col511-plan/dryrun-091-vs-091.md).

</details>

## 10. Kpool tail-ring positions omission

**Effect.** tpurtell 0.7.0 through 0.9.1 build attention metadata without token positions, which upstream vLLM passes.
The kpool tail ring then takes a generic slot mapping: from position 8 every tail write goes to a shared null block, so
concurrently decoding requests share one ring, and a pool completed across two steps can be compressed from another
request's keys. Through stale ids left by the runner's start-up warm-up, a tail write occasionally lands in another
request's blocks. The cross-request share grows with the number of requests in flight on 0.9.1, and is total on 0.7.0
at every concurrency, where the two kpool issues fixed in 0.9.0 add to it.

**Control.** None in the released engines; every GLM-5.3 record compared here carries it. A fix (positions passed as
upstream does, upstream's ownership guard, a write that replayed graphs read) is prepared and verified locally. The
completion arm excludes it by design, so the arm and its comparator carry the same defect.

**Status.** Measured: a localized defect (supported). Its effect on completion or accuracy is untested. 0.7.0 is the
more affected engine; greater corruption there does not exclude an accidental completion benefit from incorrect
computation, so the defect's role in 0.7.0's better completion is untested (corrected 2026-10-11: this entry earlier
said the defect does not explain it; see the [ledger](LEDGER.md)).

<details>
<summary>Evidence</summary>

- CPU harness running each release's real kernels (synthetic scheduler and keys), share of decode-completed pools
  that differ from the compression of the request's own committed keys:

| Engine | 1 request | 2 requests | 8 requests |
|---|---|---|---|
| 0.9.1 as released | 0% | 47-57% | 78-90% |
| 0.9.1, positions passed | 0% | 0% | 0% |
| 0.7.0 as released | 100% | 100% | 100% |

- With positions passed, 0.7.0 keeps errors from its own ring-size and tail-seed issues, the two kpool issues fixed
  since 0.9.0 ([kpool investigation](investigations/2026-10-glm53-kpool-tail)).
- Shipped graph mode, unmodified 0.9.1, one run: 98% of tail writes went to the null block on each rank; 6 events
  wrote into live attention blocks. With the fix: 0 null-block, 0 foreign and 0 attention-block writes; all requests
  completed; throughput unchanged.
- The harness models neither real scheduling nor stale warm-up ids; that serving behaves the same is inferred.
- Receipts: [tail ring](investigations/2026-10-glm53-completion/README.md#kpool-tail-ring),
  [`tail-ring/`](investigations/2026-10-glm53-completion/tail-ring),
  [`tail-fix/`](investigations/2026-10-glm53-completion/tail-fix),
  [two-rank harness](investigations/2026-10-glm53-completion/tail-fix/two-rank/summary.md).

</details>

## 11. NaN-byte index rows

**Effect.** Stale tail ids ([10](#10-kpool-tail-ring-positions-omission)) write raw indexer key and gate rows into
other requests' indexer pages, where some of the bytes decode as NaN when read as compressed keys. The kernel's scores
stay finite (inferred: its max with 0 maps NaN dot products to 0), so such pools can still be selected and attended.

**Control.** None in the released engines. The gate reported them separately from its selection checks. The column-511
correction does not repair them.

**Status.** Measured in one gate run; not measurable from GPQA outputs. They limit what a negative completion result
for the column-511 correction could exclude.

<details>
<summary>Evidence</summary>

- Gate run, shipped graph mode, fp64 re-score of sampled rows: at least one selected pool with NaN bytes in 7.9% of
  decode rows on rank 0 and 32.5% on rank 1 (prefill rows 3.9% and 20.0%). Kernel NaN or infinite scores in serving: 0.
- The gate attributes them, by inference, to the tail-ring family: mainly on rank 1, persisting in prefix-cached blocks.
- In stage 2 the same family showed as 16 index-K pools byte-identical across two different requests.
- Receipts: [gate result](investigations/2026-10-glm53-completion/gate/RESULT.md).

</details>

## 12. Historical control for the completion arm

**Effect.** The completion arm is compared with the 0.9.1 record made on 2026-10-08 and 10-09. Whatever differs
between then and now (server sessions, cache state, timing and batch composition, other load on the second GPU) is
confounded with the correction.

**Control.** A comparator audit before any arm data: the arm's image is built from the release digest with one changed
file; weights, prompts, tokenizer, parsers, sampling, seeds, concurrency, admission, warm-up and server settings match;
the session split (pass 1, then passes 2-3, each on a fresh server) is mirrored; desktop graphics on the second GPU are
logged, not controlled. The analysis, frozen before data, states exchangeability as an assumption.

**Status.** Stated limit. The arm's result (15 vs 16 empty replies) was read as R3 under the frozen rules: the
localized correction stands but is not supported as the dominant completion explanation. Exchangeability with the
recorded baseline remains an assumption. The arm's rate resembles the record's, but the arm ran a modified engine, so it
is not a replication of the unmodified 0.9.1 record.

<details>
<summary>Evidence</summary>

- [Comparator audit](investigations/2026-10-glm53-completion/arm-col511-plan/COMPARATOR-AUDIT.md): no material
  unexplained configuration difference.
- [Frozen analysis](investigations/2026-10-glm53-completion/arm-col511-plan/analysis-col511.md), section 5: matching
  recipes and seeds does not make the comparison a randomized experiment.
- Dry runs reproduce the published 0.7.0 vs 0.9.1 comparison and give zero effect for 0.9.1 against itself
  ([`arm-col511-plan/`](investigations/2026-10-glm53-completion/arm-col511-plan)).

</details>

## 13. Router precision and the routed-expert SwiGLU clamp

**Effect.** Two departures from upstream vLLM, in every release. On this GPU family the MoE router's fp32-output path
is not enabled, so routing logits are computed in bf16 although the checkpoint asks for fp32 routing (upstream enabled
it for this family, vllm#54048). The routed experts never receive the checkpoint's `swiglu_limit` (10); shared experts
and dense layers are clamped.

**Control.** None. Both are the same in all releases, so neither can explain a difference between 0.7.0 and 0.9.1;
both can enter any comparison with other runtimes or with published scores ([8](#8-published-scores-use-other-weights-and-harnesses)).

**Status.** Untested. An earlier clamp comparison patched a path this quant does not run and is not evidence either
way ([upstream comparison](investigations/2026-10-glm53-completion/README.md#comparison-with-upstream-vllm)).

## 14. Graph vs eager execution for instrumentation

**Effect.** In this build the sparse indexer is not an eager-break op, so a Python probe inside it does not run under
CUDA graphs. Measurements taken with such a probe ran in eager mode (stage 2's capture, phase 3 in place), while GPQA
runs use the shipped FULL_AND_PIECEWISE graphs. A probe in eager mode adds per-step synchronisation; the top-k route
depends only on the row count.

**Control.** The gate ran the column-511 correction under the shipped graphs with an observe-only device-side checker,
and matched its calls to the runner's FULL graph replays on both ranks (no eager fallback). The tail-ring runs used a
device-side guard in shipped graph mode. Throughput figures carry each run's own instrumentation and are not clean
measures of a fix's cost.

**Status.** Controlled for the gate and the tail-ring verification. Eager-mode figures (stage 2, phase 3) are labelled
as such in the [investigation](investigations/2026-10-glm53-completion/README.md#phase-2-stage-2-real-inputs).

## 15. Long-context decode vs prefill measures continuation after the reply

**Effect.** `decode-prefill-consistency` generates with `ignore_eos`, so a trace continues after the model has ended
its reply. In all 24 traces kept by the completion investigation's steps 5-6 the reply had ended at 637-5,377
generated tokens: the 8,000-15,999-token bin is entirely forced continuation (repeated synthetic turns, up to 93
terminators). A long-context difference in that bin measures agreement on text the model would not produce in normal
use.

**Control.** Stated with the measure ([protocol v2 note](protocols/decode-prefill-consistency/v2.md)); the
long-context bins are not used as evidence about non-completion.

**Status.** Stated limit. Step 4 kept no traces (the same pattern is likely, not verified). Where the 2,600-token v1
traces cross the end of the reply was not audited
([amendment 7](investigations/2026-10-glm53-completion/PLAN.md#dated-amendments)).
