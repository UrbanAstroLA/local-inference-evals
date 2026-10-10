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
| [Changes bundled in a release](#7-several-changes-between-releases-at-once) | A release difference cannot be pinned to one change | Labels name the release as shipped; component screen | Open |
| [Published scores](#8-published-scores-use-other-weights-and-harnesses) | Gaps mix quantization, harness, scoring and runtime | Context only, never ranked | Stated limit |

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

</details>

## 8. Published scores use other weights and harnesses

**Effect.** Model-card GPQA scores (92.1 NVIDIA, 90.6 Red Hat) used BF16 or NVFP4 weights and harnesses not fully
published.

**Control.** Shown as context only, never ranked against local runs.

**Status.** Stated limit. No higher-precision reference was run on this hardware, so these runs cannot separate
quantization, harness, scoring and runtime effects.
