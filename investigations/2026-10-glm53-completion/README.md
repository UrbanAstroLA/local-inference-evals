# Completion: why tpurtell 0.7.0 left fewer GPQA answers empty than 0.9.1 (2026-10)

With the same 3.25bpw weights, tpurtell 0.7.0 (DFlash2 ×5) left 4 of 594 GPQA answers empty and tpurtell 0.9.1
(DFlash2 ×3) left 16. This investigation asks why, and whether a small, correct change to 0.9.1 recovers the
difference. Quantization is held fixed.

**Status:** Phase 0 done (existing data only). Phase 1 steps 1-6 done (decode vs prefill, 10.55 GPU-hours); a trace audit
then showed their long-context measure covers continuation after the model had already stopped. Completion pilot and
fresh-seed replication done (3.55 GPU-hours). Kernel and DCP source audit done (CPU only). Two localized defects found and verified on GPU, each
with a fix that removes the localized problem (amendments 10-10f). The column-511 correction passed a gate in the shipped
graph configuration, and a three-pass GPQA arm with that correction only has finished against the existing 0.9.1 record,
analysed with the script frozen beforehand (amendments 11-11b): 15 vs 16 empty replies of 594, so the correction stands
but is not supported as the dominant completion explanation ([completion arm](#completion-arm)); the 12 stated-answer
rows were read by hand and the stated figure is final. [Stage A verification](#stage-a-verification-d3-d4-d5) is complete
(amendments 13-14): the router's bf16 logits (D3) change top-8 expert membership on 3.15% of token × layer rows and the
fp32-output correction is verified; the routed SwiGLU limit omission (D4) is confirmed, its effect not measured; workspace
locking (D5) is not exercised in this workload. A bundle of four corrections (D1-D4) on 0.9.1
[passed its gate](#bundle-gate-d1d2d3d4-on-091) (D4 reachable in served kernels), with two criteria revised after the
gate data were inspected, and [one GPQA pass](#one-pass-gpqa-screen) with it left 3
of 198 replies empty, all at the cap: similar, no clear separation under rules frozen before generation (amendments
14-14e). A [second pass at seed 1235](#second-pass-seed-1235) (amendments 15-15c) also left 3 of 198 empty, all at the
cap, against 5 in the matching stock pass: modest improvement, no clear separation, with no accuracy concern on the
frozen family standard. An exploratory combination of both passes gives 6 vs 10 empty of 396 (one-sided p = 0.11).
The corrected build was then frozen as checkpoint B0 and three targeted invariants were tested on it, model-free
([amendment 16](#checkpoint-b0-and-targeted-invariants-amendment-16)): the two-GPU all-reduce transport is bit-exact
(2,040,800 calls, 0 mismatches), the KV record writer and query projection agree with 0.7.0's within expected rounding
(criteria errata recorded before GPU data), and cache writes and reads stay in their own pages at high offsets for the
kernels driven (after fixture faults found in the data were fixed); E1 is expected numerical variation. No defect found,
so no B1; 0.1161 GPU-hours. Cumulative 47.91 GPU-hours;
occupied GPU-hours of the three K3.25 records are
[reconstructed from start-to-stop timings](#occupied-gpu-hours-of-the-three-k325-records). A replication of stock
0.7.0's record stays deferred, not authorized (amendments 12, 14); nothing further is scheduled.
Plan: [`PLAN.md`](PLAN.md), frozen 2026-10-10, with [dated amendments](PLAN.md#dated-amendments).
**Fast path:** [answer](#answer) · [evidence](#evidence) · [configuration diff](#what-differs-between-the-two-servers) ·
[Phase 1](#phase-1) · [completion pilot](#completion-pilot-and-replication) ·
[prior probes](#prior-probes-read-before-proposing-new-work) · [kernel and DCP audit](#kernel-and-dcp-source-audit) ·
[column-511 stages](#phase-2-stage-1-which-pool-is-dropped) · [phase 3](#phase-3-column-511-intervention) · [tail ring](#kpool-tail-ring) ·
[gate](#gate-column-511-correction-in-the-shipped-graph-configuration) · [completion arm](#completion-arm) · [upstream comparison](#comparison-with-upstream-vllm) ·
[stage A](#stage-a-verification-d3-d4-d5) · [defect matrix](#defect-matrix-d1-d6) · [bundle gate](#bundle-gate-d1d2d3d4-on-091) ·
[screen](#one-pass-gpqa-screen) · [second pass](#second-pass-seed-1235) · [checkpoint B0](#checkpoint-b0-and-targeted-invariants-amendment-16) ·
[observations recorded separately](#observations-recorded-separately) ·
[open questions](#open-questions) · [limits](#limits) · [next steps](#next-steps) · [recompute](#recompute)

## Answer

| Finding | Grade |
|---|---|
| The whole 8-answer raw score difference comes from question-passes where 0.7.0 answered correctly and 0.9.1 came back empty. Where both answered (577 of 594), raw accuracy is identical: 516 vs 516 (stated 523 vs 522). | **Descriptive** |
| Every empty answer with a request log ran to the 327,680-token cap, except one that stopped after 38 tokens (0.9.1, question 12, pass 1). No server or transport errors. | **Descriptive** |
| Most of the 0.9.1 empty answers that cost points are on questions that finish well under the cap in other passes. | **Descriptive** |
| The two servers differ in draft depth, parallel layout, KV record, MLA ownership, draft-slot sharing, vision and engine code at once. Sampler, rejection test, KDA code, parsers, weights and rendered prompts are the same. | **Descriptive** |
| Decode vs prefill (Phase 1 steps 1-4): no step found 0.9.1's decode agreeing worse with its own prefill than 0.7.0's. At 8,000-15,999 tokens 0.7.0's agreed worse. Below 2,044 tokens disagreement is higher with speculation on than off, in both engines. | **Descriptive** |
| The two kpool fixes on 0.7.0 (step 5): its disagreement at 8,000-15,999 tokens stays at stock level. These repairs did not explain the difference under this protocol. | **Descriptive** (exploratory screen) |
| 0.9.1 with 0.7.0's parallel layout (step 6): its disagreement at 8,000-15,999 tokens rises to stock 0.7.0's level. The old layout bundle can reproduce the difference on newer code; which part of the bundle does is not identified. | **Descriptive** (exploratory screen) |
| Trace audit (amendment 7): in all 24 retained step-5/6 traces the model ended its reply at 637-5,377 tokens; the 8,000-15,999-token measure of steps 4-6 is entirely forced continuation after that point, so it says nothing about non-completion. | **Descriptive** |
| Completion pilot and replication (amendments 8-9): 0.9.1's default layout vs the 0.9.1 EP2/DCP2 diagnostic bundle, 24 requests per arm on three target questions and a control. Default layout 2 of 10 fresh-seed target draws unanswered at the cap, bundle 0 of 10 (one-sided p ≈ 0.24); 0 vs 0 on the 8 replication draws per arm. No useful separation. | **Descriptive** |
| Kernel and DCP source audit (amendment 10a): no defect unique to 0.9.1 found in attention, the DCP combine, record readers, graphs or scratch; the known indexer write defects are on 0.7.0's side and fixed in 0.9.1. | **Descriptive** (source reading) |
| Column-511 drop, real inputs (amendment 10b): 0.9.1's tiled top-k path discards a pool ranked better than 480 of 512 in 88.1% of 9,944 live decode rows; 0.7.0's path does so in 6.9% of rows on the identical bytes. A fix that moves the lowest-scoring selected pool into the discarded column removes it (100% of rows, offline and in place; selection unchanged). | **Supported** (localized defect; eager mode) |
| Kpool tail ring (amendments 10d-10e): this build does not pass token positions to the attention metadata, so from position 8 the tail ring is written to a shared null block (98% of tail writes in shipped graph mode) and, through stale ids left by the runner's start-up warm-up, occasionally into live attention blocks. The same omission is in 0.7.0's source. Passing positions as upstream vLLM does, with a graph-safe write, removes it (0 stray writes). | **Supported** (localized defect; graph mode) |
| Completion arm (amendments 11a-11b): 0.9.1 with the column-511 selection correction alone, three GPQA passes, against the existing 0.9.1 record (a historical control), analysis frozen before data. Empty replies 15 vs 16 of 594 (one-sided p = 0.50); raw accuracy 85.0% vs 86.9% (interval includes zero). The correction stands as a correctness fix but is not supported as the dominant completion explanation. | **Descriptive** (historical control) |
| Router logits (stage A, amendments 13-14): every release computes the MoE router's logits in bf16 on this GPU family. On 1,593 captured records (103,608 token × layer rows) of 0.9.1 in shipped graph mode, the fp32-output computation selects a different top-8 expert set on 3.15% of rows (95% over 176 requests 3.01-3.29%; over 8 histories 2.90-3.50%) and different routing weights on every row; it lies within the stated tolerance of an fp64 reference on every logit, offline and as served. | **Supported** (computational effect); completion **untested** alone |
| Correctness bundle (amendments 14-15c): D1+D2+D3+D4 on 0.9.1 passed its gate (two criteria revised after the gate data were inspected; the original criteria failed as specified). Two GPQA passes, each read by rules frozen before its generation: seed 1234, 3 of 198 empty against 5 in the matching stock 0.9.1 pass (category similar / no clear separation, amendment 14c); seed 1235, 3 of 198 against 5 (category modest improvement, no clear separation, amendment 15b; no accuracy concern on the family standard). All six empties at the cap. Column-511 arm 4 and 5; 0.7.0 (context) 0 and 3. Exploratory combination (follow-up chosen after the first pass; historical controls; seeds reused): 6 vs 10 empty of 396, reduction 1.01 points (95% 0.00 to 2.27), one-sided p = 0.11; raw −0.76 points (−3.79 to +2.27). | **Descriptive** (two passes, historical controls; combination exploratory) |
| Targeted invariants on checkpoint B0, the corrected build (amendment 16; model-free fixtures with exact or pre-stated references): the two-GPU all-reduce transport B0 dispatches under its CUDA graphs returned the exact bf16 sum on both GPUs in all 2,040,800 calls; B0's KV record writer and query projection agree with 0.7.0's within expected rounding (with criteria errata recorded before GPU data); B0's cache writers and readers stay in their own pages and address correctly above 2³¹ bytes for every kernel driven. No defect found; these do not cover the full-layer metadata paths or numerics elsewhere. | **Supported** (within the tested coverage) |
| Recorded separately: the served router's weights depend on batch size (E1, since classified as expected numerical variation of the upstream kernel); greedy decoding of the stock 0.9.1 release is not repeatable at long context (short prompts are); NaN-byte index rows are absent with the bundle, against thousands in the 2026-10-11 gate. | **Descriptive** ([observations](#observations-recorded-separately)) |
| What produces the completion gap, and whether the tail-ring fix, router precision or the SwiGLU clamp changes completion on its own. | **Open** |

Splitting the score by completion is not a causal comparison. It says where the difference sits, not what produces it.

## Evidence

The two three-pass records, joined on question and pass (the same request seed, prompt, target and question hash in
every pair):

| Outcome, of 594 question-passes | 0.7.0, DFlash2 ×5 | 0.9.1, DFlash2 ×3 |
|---|---:|---:|
| Raw correct | 524 | 516 |
| Stated correct (audited) | 531 | 522 |
| Empty answers | 4 | 16 |
| Raw correct, the 577 where both answered | 516 | 516 |
| Stated correct, the 577 where both answered | 523 | 522 |

<details>
<summary>The 8 score-driving pairs, and how each empty answer ended</summary>

0.7.0 right, 0.9.1 empty (question, pass, request seed): q12 p1 (1234), q55 p2 (1235), q81 p3 (1236), q88 p1 (1234),
q88 p3 (1236), q109 p1 (1234), q120 p1 (1234), q170 p3 (1236). None the other way. Three pairs were empty in both
(q79 p2, q81 p2, q121 p2); in six more, one side was empty and the other answered wrong.

How empty answers ended (`finish_reason` and `completion_tokens` from the passive request log):

| Record | Empty | Ran to the 327,680-token cap | Stopped early | Error |
|---|---:|---:|---:|---:|
| 3.25bpw · tpurtell 0.7.0 · DFlash2 ×5 | 4 | 4 | 0 | 0 |
| 3.25bpw · tpurtell 0.9.1 · DFlash2 ×3 | 16 | 15 | 1 | 0 |
| 4bpw TR3 (Brandon) · tpurtell 0.9.1 · DFlash2 ×3 · concurrency 4 (reference) | 17 | 17 | 0 | 0 |

- 0.7.0 pass 1 has no request log, but it had no empty answers.
- The early stop: question 12, pass 1 on 0.9.1, ended with `finish_reason` stop after 38 tokens, with reasoning and no
  answer. It was the 13th request of the pass. The server log shows nothing unusual at that time. The same question
  finished normally in its other passes on both releases (10,000-37,000 tokens).
- Seven of the eight pairs are 0.9.1 cap exhaustions. In other passes the same questions finish in 2,600-233,000 tokens,
  mostly far below the cap (q109: 12,000-55,000; q55: 2,600-33,000; q170: 42,000-90,000).
- Whether a cap exhaustion was a loop or varied reasoning cannot be told: reasoning text is not kept for GPQA runs.
- Separately from empty answers: 3 (0.7.0) and 2 (0.9.1) replies had an answer that the raw filter could not extract.
  The audited stated score covers these.
- Answered requests, passes 2-3: median length 3,192 (0.7.0) vs 3,498 (0.9.1) completion tokens; answered after more
  than 200,000 tokens: 4 vs 1.

</details>

### What differs between the two servers

From each server's start-up log (arguments, engine configuration, KV format, placement), the launch scripts, and
hashes of files read from the two images without running them.

<details>
<summary>Effective configuration, 0.7.0 vs 0.9.1 as served for these GPQA runs</summary>

| Item | 0.7.0 | 0.9.1 | | Source |
|---|---|---|---|---|
| Speculation | DFlash2, 5 draft tokens | DFlash2, 3 draft tokens | different | server arguments; engine config |
| Draft model, draft KV | same checkpoint, bfloat16 KV | same | same | server arguments |
| Draft KV placement | separate draft pages | draft-slot sharing (draft layers in MLA slots) | different | start-up log (slot sharing line) |
| Parallel layout | TP2, EP2 experts (144 of 288 per GPU), DCP2 | TP2, rank-sliced experts, DCP1 | different | server arguments; expert-map and placement lines |
| MLA ownership | none (DCP2 split by token) | `split:25` (layers 3-23 on GPU 0, 27-43 on GPU 1) | different | placement ledger |
| KV cache | `fp8_ds_mla`, 656-byte record | `fp8_ds_mla`, 528-byte NOPE record | dtype same, record different | KV format line |
| KDA recurrent state | full-state rollback, `mamba_cache_mode` align | same; KDA code byte-identical | same | server arguments; file hash |
| CUDA graphs, compilation | FULL_AND_PIECEWISE, compilation off, captures up to 96 | same, captures up to 64 | sizes differ | engine config |
| Prefix caching, chunked prefill | on, on; 2,048 batched tokens | same | same | server arguments |
| Max sequences, max length, memory | 16; 1,048,576; 0.95 | same | same | server arguments |
| KV pool, tokens | 2,758,919 (2,768,240 for pass 1) | 4,707,515 | different | start-up log |
| Stop / EOS | from `generation_config.json` (identical file) | same | same | model files; server arguments |
| Reasoning and tool parser | `glm45` reasoning, `glm47` tools | same; parser files byte-identical | same | server arguments; file hash |
| Chat template | checkpoint's own (sha256 `34d5ee66…`) | corrected Z.ai template via `--chat-template` (sha256 `0c4099f3…`) | bytes differ; all 198 GPQA prompts render byte-identically | server arguments; render check |
| Sampling defaults | temperature 1.0, top_p 0.95 from `generation_config.json`; sampler and rejection sampler byte-identical | same | same | start-up log; file hash |
| Vision | on (image limit 16) | off (`--language-model-only`) | different | server arguments |
| Weights | revision 701cd745 | revision 0490d2f7: same weight, config, tokenizer and generation files; only the template and README differ | weights same | model file listings |
| Engine code | recipe 92eec28, B12x fe054789 | recipe 84572a8, B12x 7fcc094e; same vLLM core (0.1.dev20051+g487ecf187) and DFlash2 code | different | image labels; engine config |

- 26 vLLM Python files differ between the images, among them the EXL3 MoE path, the sparse-MLA backend, the GLM-5.3
  model and attention, kpool compression, KV cache layout, the scheduler and the structured-output code.
- `swiglu_limit` is not passed to the routed experts in either image.
- 0.7.0 pass 1 (2026-09-29) kept no full server log. It used the same image digest and launch path; its start-up
  markers match passes 2-3 except the KV pool.
- Prompt token counts agree for every question in passes 2-3 of the two records.

</details>

## Phase 1

Does each engine compute the same next-token distribution when it decodes a token as when it re-reads the same tokens
by prefill? Measured with [`decode-prefill-consistency/v1`](../../protocols/decode-prefill-consistency/v1.md)
(12 prompts × 2,600 tokens) and [`v2`](../../protocols/decode-prefill-consistency/v2.md) (prompts 0-5 × 16,000
tokens). Number: mean KL over the shared top-20 tokens, all prompts of a run pooled (steps 5-6: prompts 1-5; lower =
decode agrees more closely with prefill). Each step's reading rule was written in [`PLAN.md`](PLAN.md#dated-amendments) before that step's data.
Both engines received the same prompt token ids, so the chat template does not enter.

| Step | Question | Verdict under the step's rule | GPU-hours |
|---|---|---|---:|
| [1](#step-1) | Speculation off: 0.7.0 vs 0.9.1 | Below 2,044: no difference beyond run-to-run variation. From 2,048: both 0.7.0 runs above all three 0.9.1 runs, by 0.0003 (0.7.0 agrees slightly worse) | 1.54 |
| [2](#step-2) | 0.9.1: DFlash2 ×5 vs ×3 | No difference in either region | 1.61 |
| [3](#step-3) | Speculation on: 0.7.0 ×5 vs 0.9.1 ×3 and ×5 | No difference in either region | 0.84 |
| [4](#step-4) | 16,000 tokens, both engines as shipped | 8,000-15,999: both 0.7.0 runs above both 0.9.1 runs, by 0.0081: 0.7.0 as shipped drifts further from its prefill with context. Other bins, reported only: the same order | 3.19 |
| [5](#step-5) | 16,000 tokens: 0.7.0 with the two kpool fixes vs 0.7.0 as shipped | The corrected values remain at stock level: these repairs did not explain the discrepancy under this protocol | 1.62 |
| [6](#step-6) | 16,000 tokens: 0.9.1 with 0.7.0's parallel layout vs 0.9.1 as shipped | Rise condition met: the old layout bundle can reproduce the discrepancy on newer code; no component identified | 1.75 |
| [Across steps](#across-steps) | Speculation on vs off, below 2,044 tokens | Descriptive: in both engines every speculation-on run lies above every speculation-off run | |

Total 5.27 wall hours, 10.55 GPU-hours. **Grade: descriptive.** These steps measure agreement along each run's own
trajectory; they do not measure completion. No step found 0.9.1 agreeing worse than 0.7.0. Where a rule found a
difference (step 1 from 2,048, by a hairline; step 4), 0.7.0 agreed worse. Steps 5 and 6 are exploratory screens on
prompts 1-5 at 8,000-15,999 tokens: the kpool fixes left 0.7.0's disagreement where it was, and 0.7.0's layout raised
0.9.1's.

<details>
<summary>What the measure can and cannot see</summary>

- A run is one generation per prompt. Two runs of one configuration with the same seeds diverge within a few tokens and
  differ in a region's mean by up to about 4× among these runs, so each rule compares ranges of runs, not single
  numbers.
- KL over the shared top-20 tokens ignores probability outside their intersection (PLAN.md, Phase 2 item 4).
- In every run of steps 1-4, including the 2026-10-08/09 0.9.1 runs, the engine's start-up warm-up requests (up to 16)
  ran during the first minute of prompt 0; later prompts ran alone. Steps 5-6 started measuring after the warm-up and
  leave prompt 0 out of their rules. Without prompt 0, step 1's separation from 2,048 disappears
  (ranges overlap); the other verdicts are unchanged, and the across-steps separation narrows to the fifth decimal for
  0.9.1 ([recompute](#recompute)).
- From 2,048 the reference is the server's chunked prefill over cached context (2,048-token chunks), not a single pass.
- The two engines differ in layout, KV record, vision and code at once ([configuration diff](#what-differs-between-the-two-servers)).
  A difference between them is not attributed to one component.

</details>

### Step 1

**Question.** With speculation off, does 0.7.0's decode agree with its own prefill better or worse than 0.9.1's?
**Rule (amendment 1).** "A difference counts only if every 0.7.0 run lies outside the range of all 0.9.1 runs in the
same region (and vice versa); otherwise "no difference beyond run-to-run variation"."

| Run | Group | KL below 2,044 | KL from 2,048 |
|---|---|---:|---:|
| [2026-10-10 r1](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-nospec-nocache_decode-prefill-r1/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-nospec-nocache_decode-prefill-r1/summary.json)) | 0.7.0, speculation off | 0.0061 | 0.0142 |
| [2026-10-10 r2](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-nospec-nocache_decode-prefill-r2/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-nospec-nocache_decode-prefill-r2/summary.json)) | 0.7.0, speculation off | 0.0104 | 0.0145 |
| [2026-10-08 run](../../runs/2026-10-08_glm53-flash_k3.25-v0.9.1-nospec-nocache_decode-prefill/) ([summary](../../runs/2026-10-08_glm53-flash_k3.25-v0.9.1-nospec-nocache_decode-prefill/summary.json)) | 0.9.1, speculation off | 0.0072 | 0.0053 |
| [2026-10-09 run](../../runs/2026-10-09_glm53-flash_k3.25-v0.9.1-nospec-nocache_decode-prefill/) ([summary](../../runs/2026-10-09_glm53-flash_k3.25-v0.9.1-nospec-nocache_decode-prefill/summary.json)) | 0.9.1, speculation off | 0.0077 | 0.0098 |
| [2026-10-10 r3](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-nospec-nocache_decode-prefill-r3/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-nospec-nocache_decode-prefill-r3/summary.json)) | 0.9.1, speculation off | 0.0072 | 0.0139 |
| **range** | 0.7.0, speculation off | 0.0061-0.0104 | 0.0142-0.0145 |
| **range** | 0.9.1, speculation off | 0.0072-0.0077 | 0.0053-0.0139 |

**Verdict.** Below 2,044: no difference beyond run-to-run variation. From 2,048: a difference by the rule, 0.7.0 above
0.9.1 by 0.0003 (0.7.0 agrees slightly worse); it does not hold without prompt 0. **Cost** 0.77 wall hours, 1.54 GPU-hours.

### Step 2

**Question.** On 0.9.1, does deeper speculation (DFlash2 ×5) change what the model computes along its own trajectory,
compared with ×3?
**Rule (amendment 2).** "A difference counts only if both x5 runs lie outside the range of both x3 runs in the same
region; the speculation-off runs of step 1 (0.9.1) give the reference range."

| Run | Group | KL below 2,044 | KL from 2,048 |
|---|---|---:|---:|
| [2026-10-10 r1](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash5-nocache_decode-prefill-r1/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash5-nocache_decode-prefill-r1/summary.json)) | 0.9.1, DFlash2 x5 | 0.0100 | 0.0066 |
| [2026-10-10 r2](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash5-nocache_decode-prefill-r2/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash5-nocache_decode-prefill-r2/summary.json)) | 0.9.1, DFlash2 x5 | 0.0206 | 0.0268 |
| [2026-10-08 run](../../runs/2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill/) ([summary](../../runs/2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill/summary.json)) | 0.9.1, DFlash2 x3 | 0.0105 | 0.0115 |
| [2026-10-10 r2](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-r2/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-r2/summary.json)) | 0.9.1, DFlash2 x3 | 0.0131 | 0.0186 |
| [2026-10-08 run](../../runs/2026-10-08_glm53-flash_k3.25-v0.9.1-nospec-nocache_decode-prefill/) ([summary](../../runs/2026-10-08_glm53-flash_k3.25-v0.9.1-nospec-nocache_decode-prefill/summary.json)) | 0.9.1, speculation off (reference) | 0.0072 | 0.0053 |
| [2026-10-09 run](../../runs/2026-10-09_glm53-flash_k3.25-v0.9.1-nospec-nocache_decode-prefill/) ([summary](../../runs/2026-10-09_glm53-flash_k3.25-v0.9.1-nospec-nocache_decode-prefill/summary.json)) | 0.9.1, speculation off (reference) | 0.0077 | 0.0098 |
| [2026-10-10 r3](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-nospec-nocache_decode-prefill-r3/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-nospec-nocache_decode-prefill-r3/summary.json)) | 0.9.1, speculation off (reference) | 0.0072 | 0.0139 |
| **range** | 0.9.1, DFlash2 x5 | 0.0100-0.0206 | 0.0066-0.0268 |
| **range** | 0.9.1, DFlash2 x3 | 0.0105-0.0131 | 0.0115-0.0186 |
| **range** | 0.9.1, speculation off (reference) | 0.0072-0.0077 | 0.0053-0.0139 |

**Verdict.** No difference beyond run-to-run variation in either region. **Cost** 0.80 wall hours, 1.61 GPU-hours.

### Step 3

**Question.** Is 0.7.0's speculation-on agreement (as shipped, DFlash2 ×5) better than 0.9.1's?
**Rule (amendment 3).** "A difference counts only if both 0.7.0 runs lie outside the range of all four 0.9.1
speculation-on runs (x3 and x5) in the same region."

| Run | Group | KL below 2,044 | KL from 2,048 |
|---|---|---:|---:|
| [2026-10-10 r1](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-r1/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-r1/summary.json)) | 0.7.0, DFlash2 x5 | 0.0138 | 0.0135 |
| [2026-10-10 r2](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-r2/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-r2/summary.json)) | 0.7.0, DFlash2 x5 | 0.0106 | 0.0165 |
| [2026-10-08 run](../../runs/2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill/) ([summary](../../runs/2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill/summary.json)) | 0.9.1, speculation on (x3 and x5) | 0.0105 | 0.0115 |
| [2026-10-10 r2](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-r2/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-r2/summary.json)) | 0.9.1, speculation on (x3 and x5) | 0.0131 | 0.0186 |
| [2026-10-10 r1](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash5-nocache_decode-prefill-r1/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash5-nocache_decode-prefill-r1/summary.json)) | 0.9.1, speculation on (x3 and x5) | 0.0100 | 0.0066 |
| [2026-10-10 r2](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash5-nocache_decode-prefill-r2/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash5-nocache_decode-prefill-r2/summary.json)) | 0.9.1, speculation on (x3 and x5) | 0.0206 | 0.0268 |
| **range** | 0.7.0, DFlash2 x5 | 0.0106-0.0138 | 0.0135-0.0165 |
| **range** | 0.9.1, speculation on (x3 and x5) | 0.0100-0.0206 | 0.0066-0.0268 |

**Verdict.** No difference beyond run-to-run variation in either region. **Cost** 0.42 wall hours, 0.84 GPU-hours.

### Step 4

**Question.** Does 0.9.1's decode drift from its prefill more than 0.7.0's as context grows? 16,000 tokens, prompts
0-5, both engines as shipped (0.7.0 DFlash2 ×5, 0.9.1 DFlash2 ×3), two runs each, alternating.
**Rule (amendment 4).** "A difference counts only if both runs of one engine lie outside the range of both runs of the
other in the 8,000-15,999 bin; the other bins are reported."

| Run | Group | KL 0-2,043 | KL 2,048-7,999 | KL 8,000-15,999 | KL 16,000 and above |
|---|---|---:|---:|---:|---:|
| [2026-10-10 16k-r1](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-16k-r1/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-16k-r1/summary.json)) | 0.7.0, DFlash2 x5, 16,000 tokens | 0.0109 | 0.0407 | 0.0632 | 0.1260 |
| [2026-10-10 16k-r2](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-16k-r2/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-16k-r2/summary.json)) | 0.7.0, DFlash2 x5, 16,000 tokens | 0.0160 | 0.0255 | 0.0233 | 0.0183 |
| [2026-10-10 16k-r1](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-16k-r1/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-16k-r1/summary.json)) | 0.9.1, DFlash2 x3, 16,000 tokens | 0.0106 | 0.0131 | 0.0152 | 0.0160 |
| [2026-10-10 16k-r2](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-16k-r2/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-16k-r2/summary.json)) | 0.9.1, DFlash2 x3, 16,000 tokens | 0.0103 | 0.0151 | 0.0144 | 0.0138 |
| **range** | 0.7.0, DFlash2 x5, 16,000 tokens | 0.0109-0.0160 | 0.0255-0.0407 | 0.0233-0.0632 | 0.0183-0.1260 |
| **range** | 0.9.1, DFlash2 x3, 16,000 tokens | 0.0103-0.0106 | 0.0131-0.0151 | 0.0144-0.0152 | 0.0138-0.0160 |

**Verdict.** In the 8,000-15,999 bin both 0.7.0 runs lie above both 0.9.1 runs, by 0.0081: by this measure 0.7.0 as
shipped drifts further from its prefill with context than 0.9.1 does. Reported
only: the same order from 2,048 to 7,999 (by 0.0105) and, by 0.0003, below 2,044. The two 0.7.0 runs spread widely
(0.0233 and 0.0632 in the rule's bin), the two 0.9.1 runs little (0.0144 and 0.0152). The verdict holds without prompt 0.
**Cost** 1.60 wall hours, 3.19 GPU-hours.

<details>
<summary>Step 4 notes</summary>

- Per-prompt means in the 8,000-15,999 bin: 0.7.0 run 1 0.025-0.140, run 2 0.018-0.032; 0.9.1 0.011-0.023 and
  0.009-0.027. The 0.7.0 excess is spread over prompts, not one prompt.
- Positions from 16,000 (the causal length includes the 124-365-token prompt) are outside the rule's bins and shown for
  completeness.
- A candidate mechanism, then untested: the two upstream kpool issues present in stock 0.7.0 and fixed in 0.9.0 and
  later ([`FINDINGS.md`](../../FINDINGS.md) item 1). [Step 5](#step-5) tested it.
- The bin ends at 15,999 as amendment 4 states. Taking it to the end of each trace instead (including positions from
  16,000) gives 0.0646 and 0.0232 for 0.7.0 and 0.0153 and 0.0144 for 0.9.1; the verdict is the same.
- Six prompts and two runs per engine. Whether agreement at this length relates to completion is open: 0.7.0 left fewer
  answers empty while agreeing worse here.

</details>

### Step 5

**Question.** Does applying the two kpool fixes reduce stock 0.7.0's long-context disagreement? tpurtell 0.7.0 with
only the two kpool backports, otherwise as shipped (EP2/DCP2, DFlash2 ×5), two runs at 16,000 tokens on prompts 0-5,
against the two stock 0.7.0 runs of step 4. 0.9.1 (step 4) is context only. Measurement started after the engine's
start-up warm-up had finished (first request 0.4-0.6 s after its ready file). Token ids and log-probabilities were kept
locally; only their hashes are published ([`manifest.json`](manifest.json)).
**Rule (amendment 5).** Prompts 1-5, positions 8,000-15,999 (step 4's prompt 0 overlapped the warm-up; reported
separately). "if both corrected aggregates lie below both stock aggregates and the reduction holds in most prompts (at
least 4 of 5 per repeat), nominate the kpool bundle as affecting the measured discrepancy; if ranges overlap or
reductions are inconsistent, unresolved under this screen; if corrected values remain at stock level, these repairs did
not explain the discrepancy under this protocol. Nearness to 0.9.1 is descriptive only."

KL, sampled-token log-probability difference (abs dlp) and top-1 agreement at positions 8,000-15,999; prompts 1-5
pooled, each prompt, and prompt 0 separately:

| Run | Group | KL, prompts 1-5 | Prompt 1 | Prompt 2 | Prompt 3 | Prompt 4 | Prompt 5 | Sampled-token abs dlp | Top-1 agreement | Prompt 0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [2026-10-10 16k-r1](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-kpoolfix-dflash5-nocache_decode-prefill-16k-r1/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-kpoolfix-dflash5-nocache_decode-prefill-16k-r1/summary.json)) | 0.7.0 + kpool fixes, DFlash2 x5 | 0.0439 | 0.0232 | 0.0498 | 0.1021 | 0.0236 | 0.0206 | 0.1468 | 0.9405 | 0.0256 |
| [2026-10-10 16k-r2](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-kpoolfix-dflash5-nocache_decode-prefill-16k-r2/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-kpoolfix-dflash5-nocache_decode-prefill-16k-r2/summary.json)) | 0.7.0 + kpool fixes, DFlash2 x5 | 0.0390 | 0.0601 | 0.0453 | 0.0565 | 0.0273 | 0.0057 | 0.1375 | 0.9413 | 0.0454 |
| [2026-10-10 16k-r1](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-16k-r1/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-16k-r1/summary.json)) | 0.7.0 as shipped, DFlash2 x5 (step 4) | 0.0558 | 0.0397 | 0.0396 | 0.0251 | 0.1396 | 0.0349 | 0.1707 | 0.9360 | 0.1004 |
| [2026-10-10 16k-r2](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-16k-r2/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-16k-r2/summary.json)) | 0.7.0 as shipped, DFlash2 x5 (step 4) | 0.0243 | 0.0320 | 0.0265 | 0.0187 | 0.0182 | 0.0261 | 0.0892 | 0.9566 | 0.0186 |
| [2026-10-10 16k-r1](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-16k-r1/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-16k-r1/summary.json)) | 0.9.1 as shipped, DFlash2 x3 (step 4) | 0.0142 | 0.0230 | 0.0112 | 0.0131 | 0.0123 | 0.0113 | 0.0715 | 0.9630 | 0.0206 |
| [2026-10-10 16k-r2](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-16k-r2/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-16k-r2/summary.json)) | 0.9.1 as shipped, DFlash2 x3 (step 4) | 0.0119 | 0.0092 | 0.0128 | 0.0099 | 0.0107 | 0.0169 | 0.0652 | 0.9653 | 0.0269 |
| **range** | 0.7.0 + kpool fixes, DFlash2 x5 | 0.0390-0.0439 | 0.0232-0.0601 | 0.0453-0.0498 | 0.0565-0.1021 | 0.0236-0.0273 | 0.0057-0.0206 | 0.1375-0.1468 | 0.9405-0.9413 | 0.0256-0.0454 |
| **range** | 0.7.0 as shipped, DFlash2 x5 (step 4) | 0.0243-0.0558 | 0.0320-0.0397 | 0.0265-0.0396 | 0.0187-0.0251 | 0.0182-0.1396 | 0.0261-0.0349 | 0.0892-0.1707 | 0.9360-0.9566 | 0.0186-0.1004 |
| **range** | 0.9.1 as shipped, DFlash2 x3 (step 4) | 0.0119-0.0142 | 0.0092-0.0230 | 0.0112-0.0128 | 0.0099-0.0131 | 0.0107-0.0123 | 0.0113-0.0169 | 0.0652-0.0715 | 0.9630-0.9653 | 0.0206-0.0269 |

Prompts in which a corrected run lies below both stock runs: 2 of 5 (r1), 1 of 5 (r2).

**Verdict.** By the rule, the corrected values remain at stock level; these repairs did not explain the discrepancy under this protocol.
Both corrected aggregates (0.0390, 0.0439) lie inside the stock range (0.0243-0.0558), and the reduction holds in 2 and
1 of 5 prompts where the rule asks for 4. The result matches two of amendment 5's clauses: the ranges overlap
("unresolved under this screen") and the corrected values remain at stock level ("these repairs did not explain the
discrepancy"). The second label was chosen in amendment 6, after step 5's data was seen. Under either label, no
reduction was shown. Descriptive
only: both corrected runs lie above both 0.9.1 runs. **Cost** 0.81 wall hours, 1.62 GPU-hours.

<details>
<summary>Step 5 notes</summary>

Other bins, prompts 1-5 (reported only):

| Run | Group | KL 0-2,043 | KL 2,048-7,999 | KL 8,000-15,999 | KL 16,000 and above |
|---|---|---:|---:|---:|---:|
| [2026-10-10 16k-r1](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-kpoolfix-dflash5-nocache_decode-prefill-16k-r1/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-kpoolfix-dflash5-nocache_decode-prefill-16k-r1/summary.json)) | 0.7.0 + kpool fixes, DFlash2 x5 | 0.0115 | 0.0269 | 0.0439 | 0.0519 |
| [2026-10-10 16k-r2](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-kpoolfix-dflash5-nocache_decode-prefill-16k-r2/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-kpoolfix-dflash5-nocache_decode-prefill-16k-r2/summary.json)) | 0.7.0 + kpool fixes, DFlash2 x5 | 0.0079 | 0.0189 | 0.0390 | 0.0352 |
| [2026-10-10 16k-r1](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-16k-r1/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-16k-r1/summary.json)) | 0.7.0 as shipped, DFlash2 x5 (step 4) | 0.0104 | 0.0301 | 0.0558 | 0.1318 |
| [2026-10-10 16k-r2](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-16k-r2/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-16k-r2/summary.json)) | 0.7.0 as shipped, DFlash2 x5 (step 4) | 0.0170 | 0.0244 | 0.0243 | 0.0181 |
| [2026-10-10 16k-r1](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-16k-r1/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-16k-r1/summary.json)) | 0.9.1 as shipped, DFlash2 x3 (step 4) | 0.0096 | 0.0145 | 0.0142 | 0.0138 |
| [2026-10-10 16k-r2](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-16k-r2/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-16k-r2/summary.json)) | 0.9.1 as shipped, DFlash2 x3 (step 4) | 0.0103 | 0.0145 | 0.0119 | 0.0151 |

- Abs dlp and top-1 agreement in the rule's bin give the same picture as KL: the corrected runs lie inside the stock
  range.
- Prompt 0 was warm-up gated here but not in step 4, so the two are not a like-for-like pair: corrected 0.0256 and
  0.0454, stock 0.1004 and 0.0186.
- Taking the bin to the end of each trace (8,000 and above) gives 0.0441 and 0.0389 for the corrected runs; same verdict.
- The patched image differs from stock 0.7.0 only in `attention.py` (tail-ring sizing) and `kpool_compress.py`
  (byte-identical to the 0.9.0 and 0.9.1 file); kpool kernel tests 33 of 33
  ([run](../../runs/2026-10-05_glm53-flash_k3.25-v0.7.0-kpoolfix-dflash5_kpool-kernel-tests/)).
- One configuration's runs spread widely by prompt (prompt 5: 0.0206 and 0.0057). Five prompts and two runs per arm.

</details>

### Step 6

**Question.** Does the v0.7.0 parallel layout on 0.9.1's kernels reproduce stock 0.7.0's long-context disagreement?
tpurtell 0.9.1 with v0.7.0's layout: EP2 routed experts and DCP2, which on 0.9.1 also means no MLA layer ownership,
656-byte KV records and separate draft pages. DFlash2 ×3 and vision off, as 0.9.1 ships. Two runs at 16,000 tokens on
prompts 0-5, warm-up gated (first request 0.6-0.9 s after the ready file), traces local, against 0.9.1 at its default
DCP1 layout (step 4, ×3). No new code.
**Deviation from stock 0.7.0.** `VLLM_B12X_DCP_TOPK_OWNER_EXCHANGE=0`: the top-k owner exchange 0.7.0 uses fails on
this fork, so the top-k merge gathers every rank's candidates instead. Same selection rule, different transport; as in
the looping investigation's layout-bisection arm D.
**Rule (amendment 6; interpretation replaced by amendment 6a).** Same measure as step 5. "if both step-6 aggregates lie
above both 0.9.1-default aggregates and inside or above the stock-0.7.0 range, with the increase in at least 4 of 5
prompts per repeat, nominate the layout bundle (DCP2/EP2/ownership off/record format/top-k merge path) as affecting the
discrepancy; if both stay within the 0.9.1-default range, nominate the 0.7.0 kernel/engine code instead; otherwise
unresolved. The bundle is not split into components by this step." Amendment 6a: "If step-6 disagreement rises as
defined: the old layout bundle can reproduce the discrepancy on newer code; this does not identify which layout
component causes it."

| Run | Group | KL, prompts 1-5 | Prompt 1 | Prompt 2 | Prompt 3 | Prompt 4 | Prompt 5 | Sampled-token abs dlp | Top-1 agreement | Prompt 0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [2026-10-10 16k-r1](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-ep2dcp2-dflash3-nocache_decode-prefill-16k-r1/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-ep2dcp2-dflash3-nocache_decode-prefill-16k-r1/summary.json)) | 0.9.1, 0.7.0 layout, DFlash2 x3 | 0.0604 | 0.0803 | 0.0241 | 0.0661 | 0.0306 | 0.1007 | 0.1755 | 0.9291 | 0.0067 |
| [2026-10-10 16k-r2](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-ep2dcp2-dflash3-nocache_decode-prefill-16k-r2/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-ep2dcp2-dflash3-nocache_decode-prefill-16k-r2/summary.json)) | 0.9.1, 0.7.0 layout, DFlash2 x3 | 0.0309 | 0.0153 | 0.0413 | 0.0276 | 0.0158 | 0.0544 | 0.1106 | 0.9454 | 0.0203 |
| [2026-10-10 16k-r1](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-16k-r1/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-16k-r1/summary.json)) | 0.9.1 as shipped, DFlash2 x3 (step 4) | 0.0142 | 0.0230 | 0.0112 | 0.0131 | 0.0123 | 0.0113 | 0.0715 | 0.9630 | 0.0206 |
| [2026-10-10 16k-r2](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-16k-r2/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-16k-r2/summary.json)) | 0.9.1 as shipped, DFlash2 x3 (step 4) | 0.0119 | 0.0092 | 0.0128 | 0.0099 | 0.0107 | 0.0169 | 0.0652 | 0.9653 | 0.0269 |
| [2026-10-10 16k-r1](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-16k-r1/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-16k-r1/summary.json)) | 0.7.0 as shipped, DFlash2 x5 (step 4) | 0.0558 | 0.0397 | 0.0396 | 0.0251 | 0.1396 | 0.0349 | 0.1707 | 0.9360 | 0.1004 |
| [2026-10-10 16k-r2](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-16k-r2/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-16k-r2/summary.json)) | 0.7.0 as shipped, DFlash2 x5 (step 4) | 0.0243 | 0.0320 | 0.0265 | 0.0187 | 0.0182 | 0.0261 | 0.0892 | 0.9566 | 0.0186 |
| **range** | 0.9.1, 0.7.0 layout, DFlash2 x3 | 0.0309-0.0604 | 0.0153-0.0803 | 0.0241-0.0413 | 0.0276-0.0661 | 0.0158-0.0306 | 0.0544-0.1007 | 0.1106-0.1755 | 0.9291-0.9454 | 0.0067-0.0203 |
| **range** | 0.9.1 as shipped, DFlash2 x3 (step 4) | 0.0119-0.0142 | 0.0092-0.0230 | 0.0112-0.0128 | 0.0099-0.0131 | 0.0107-0.0123 | 0.0113-0.0169 | 0.0652-0.0715 | 0.9630-0.9653 | 0.0206-0.0269 |
| **range** | 0.7.0 as shipped, DFlash2 x5 (step 4) | 0.0243-0.0558 | 0.0320-0.0397 | 0.0265-0.0396 | 0.0187-0.0251 | 0.0182-0.1396 | 0.0261-0.0349 | 0.0892-0.1707 | 0.9360-0.9566 | 0.0186-0.1004 |

Prompts in which a step-6 run lies above both 0.9.1-default runs: 5 of 5 (r1), 4 of 5 (r2).

**Verdict.** By the rule, rise condition met: the old layout bundle can reproduce the discrepancy on newer code; this does not identify which layout component causes it.
Both step-6 aggregates (0.0309, 0.0604) lie above both 0.9.1-default runs (0.0119-0.0142); one lies inside and one
above the stock 0.7.0 range (0.0243-0.0558). The bundle changes expert parallelism, context parallelism, MLA ownership,
the KV record, draft pages and the top-k merge path at once. DCP1 with MLA layer ownership is the layout tpurtell's
later releases ship; this step finds that the older layout bundle shows more generation-vs-prefill disagreement under
this protocol, not that either layout is defective. **Cost** 0.88 wall hours, 1.75 GPU-hours.

<details>
<summary>Step 6 notes</summary>

Other bins, prompts 1-5 (reported only):

| Run | Group | KL 0-2,043 | KL 2,048-7,999 | KL 8,000-15,999 | KL 16,000 and above |
|---|---|---:|---:|---:|---:|
| [2026-10-10 16k-r1](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-ep2dcp2-dflash3-nocache_decode-prefill-16k-r1/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-ep2dcp2-dflash3-nocache_decode-prefill-16k-r1/summary.json)) | 0.9.1, 0.7.0 layout, DFlash2 x3 | 0.0202 | 0.0452 | 0.0604 | 0.0667 |
| [2026-10-10 16k-r2](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-ep2dcp2-dflash3-nocache_decode-prefill-16k-r2/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-ep2dcp2-dflash3-nocache_decode-prefill-16k-r2/summary.json)) | 0.9.1, 0.7.0 layout, DFlash2 x3 | 0.0195 | 0.0215 | 0.0309 | 0.0222 |
| [2026-10-10 16k-r1](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-16k-r1/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-16k-r1/summary.json)) | 0.9.1 as shipped, DFlash2 x3 (step 4) | 0.0096 | 0.0145 | 0.0142 | 0.0138 |
| [2026-10-10 16k-r2](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-16k-r2/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.9.1-dflash3-nocache_decode-prefill-16k-r2/summary.json)) | 0.9.1 as shipped, DFlash2 x3 (step 4) | 0.0103 | 0.0145 | 0.0119 | 0.0151 |
| [2026-10-10 16k-r1](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-16k-r1/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-16k-r1/summary.json)) | 0.7.0 as shipped, DFlash2 x5 (step 4) | 0.0104 | 0.0301 | 0.0558 | 0.1318 |
| [2026-10-10 16k-r2](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-16k-r2/) ([summary](../../runs/2026-10-10_glm53-flash_k3.25-v0.7.0-dflash5-nocache_decode-prefill-16k-r2/summary.json)) | 0.7.0 as shipped, DFlash2 x5 (step 4) | 0.0170 | 0.0244 | 0.0243 | 0.0181 |

- Below 2,044 the step-6 runs (0.0202, 0.0195) also lie above both 0.9.1-default runs and both stock 0.7.0 runs.
- Abs dlp and top-1 agreement in the rule's bin follow KL: both step-6 runs lie outside the 0.9.1-default range, inside
  or beyond the stock range.
- Prompt 0 does not follow the other prompts: 0.0067 and 0.0203, below both 0.9.1-default runs (0.0206 and 0.0269,
  whose prompt 0 overlapped the warm-up).
- Taking the bin to the end of each trace gives 0.0605 and 0.0307; same verdict.
- Draft depth: ×3 here and in the 0.9.1 default; stock 0.7.0 runs ×5. On 0.9.1, step 2 found no ×5 vs ×3 difference
  at 2,600 tokens.

</details>

### Across steps

Not a rule of any amendment; recorded in amendments 3 and 4 before further data. Below 2,044 tokens, disagreement is higher
with speculation on than off, in both engines:

| Engine | Speculation on, KL below 2,044 | Speculation off, KL below 2,044 | |
|---|---|---|---|
| 0.9.1 | 0.0100-0.0206 (4 runs) | 0.0072-0.0077 (3 runs) | every on run above every off run, by 0.0023 |
| 0.7.0 | 0.0106-0.0138 (2 runs) | 0.0061-0.0104 (2 runs) | every on run above every off run, by 0.0002 |

From 2,048 tokens the speculation-on and speculation-off ranges overlap in both engines. Whether this difference bears on
completion is open.

## Completion pilot and replication

Amendments 8-9. 0.9.1 default layout (A) vs the 0.9.1 EP2/DCP2 diagnostic bundle (B), DFlash2 ×3, temperature 1.0,
top_p 0.95, normal stopping, 8 requests in flight, caps 65,536 (q55, q109, q144 control) and 131,072 (q170). The same
seed on different layouts is not a matched trajectory, so arm-to-arm pairs are separate draws. Receipts:
[`pilot/`](pilot), [`replication/`](replication).

| Run | Requests per arm | A unanswered at cap | B unanswered at cap | Note |
|---|---|---|---|---|
| Pilot, original seeds | 4 | 0 | 0 | the original 327,680-token failures did not recur |
| Pilot, fresh seeds | 4 | 2 (q109, q170) | 0 | each one draw; within chance (p ≈ 0.23); B answered later in 3 of 4 pairs where both answered |
| Replication, 4 new fresh seeds on q109 and q170 | 8 | 0 | 0 | B longer on q109, shorter on q170 |

The two capped replies were not loops (repeated 200-character windows 0-1%, steady compression): the model reached the
target answer repeatedly in its reasoning and kept reopening it. Pilot 2.01 GPU-hours, replication 1.54.

## Prior probes (read before proposing new work)

Behavioural and kernel probes already run on these engines. Screens with a fixed seed on every repeat are withdrawn and
listed only so they are not repeated as if new.

| Probe | What it varied | Result | Status |
|---|---|---|---|
| [Component screen, q88](../../comparisons/glm53-flash-v091-component-screen-doc88/README.md) (2026-10-09, seeds 5001-5012) | On 0.9.1: EP2 experts; 656 B record (NOPE off); MLA ownership tp with slot sharing off | failed to finish: 656 B 4/24 vs 528 B 4/24; EP2 3/24 vs 5/24; ownership tp 3/24 vs 5/24; no candidate at this size | valid; sized for large effects only |
| Component screen, q79 (same date and seeds; [`FINDINGS.md`](../../FINDINGS.md) items 13-14) | 0.9.1 vs the 0.7.0 image with its own layout at 3 draft tokens | 9/12 vs 9/12 failed to finish | valid; one question |
| [kpool kernel tests](../../protocols/kpool-kernel-tests) | upstream `test_kpool_decode_update_batched` on the release images | 4 cases fail on 0.7.0 and 0.8.0 (tail ring, seed stride); pass with the fixes in 0.9.1 | valid |
| Step 5 (this investigation) | 0.7.0 with both kpool fixes | decode-vs-prefill disagreement unchanged; that measure is post-completion (amendment 7) | valid, measure limited |
| Step 6, pilot, replication (this investigation) | 0.9.1 with 0.7.0's parallel layout bundle | no reproducible completion benefit | valid |
| [Bundle screen](#one-pass-gpqa-screen) (this investigation, 2026-10-12) | 0.9.1 with D1+D2+D3+D4 together, one GPQA pass at seed 1234 | 3 of 198 empty vs 5 (0.9.1 record) and 4 (column-511 arm) in pass 1; similar, no clear separation | valid; one pass, historical controls |
| [Bundle second pass](#second-pass-seed-1235) (this investigation, 2026-10-12) | the same bundle, one GPQA pass at seed 1235 | 3 of 198 empty vs 5 (0.9.1 record) and 5 (column-511 arm) at seed 1235; modest improvement, no clear separation; both passes combined (exploratory) 6 vs 10 of 396 | valid; one pass, historical controls; chosen after the first screen |
| [Targeted invariants on B0](#checkpoint-b0-and-targeted-invariants-amendment-16) (this investigation, 2026-10-12) | model-free: B0's all-reduce transport; KV record writer and query projection, B0 vs 0.7.0; cache isolation and addressing above 2³¹ bytes | no defect: 0 mismatches in 2,040,800 all-reduces; writer and projection within rounding; 80 of 80 cache checks after fixture fixes | valid; driven kernels and transport only |
| 0.8.0 GPQA, ×5 drafts with slot sharing off vs ×3 with sharing | draft depth and sharing together | no completion improvement | single passes |
| [kpool k1 screen](../2026-10-glm53-kpool-tail/preregistration-kpool-screen.md) (2026-10-04), [layout bisection](../2026-10-glm53-looping/preregistration/WITHDRAWN.md) (2026-10-07/08) | 1 vs 3 drafts; 0.9.0 vs 0.7.0 layout | withdrawn: seed 1234 on every repeat | do not cite rates |

## Kernel and DCP source audit

Amendments 10-10a. Audit reports: [`audits/`](audits) (three Phase 1 audits with file:line references, B12x kernel
notes, and the upstream comparison by area). CPU only: the installed vLLM and B12x kernel source of the three releases (0.7.0 B12x `fe054789`;
0.8.0 and 0.9.1 `7fcc094e`), read by three independent audits (indexer selection; DCP attention; kernel dispatch and
storage), each citing file and line.

| Area | Finding |
|---|---|
| Attention and DCP combine | 0.7.0 (DCP2, 656 B records) and 0.9.1 (one rank per MLA layer, 528 B records) compute the same attention up to floating-point order; the cross-rank combine has no invalid-weight path found |
| Kernel library update | changes rounding and summation order on the paths these engines run, and the top-k kernels' internal merge |
| Indexer write defects | tail ring with five drafts and tail-seed stride: 0.7.0/0.8.0 only, fixed in 0.9.1 |
| Indexer selection | every version selects the exact top 512 compressed-key pools by score (apart from ties), then keeps output columns 0-510 to fit the newest 1-3 tokens; no top-k path sorts its output, so the dropped column is not guaranteed to be the weakest pool; which kernel path supplies it differs by version and by requests in flight. Upstream vLLM keeps all 512. |
| Also noted, not tested | a default-on two-GPU all-reduce transport new in 0.8.0 without a runnable correctness test (since tested on B0: bit-exact, [amendment 16](#checkpoint-b0-and-targeted-invariants-amendment-16)); the engine's own K4 records show teacher-forced NLL about 0.007 higher from 0.8.0 on (one 0.7.1 pass) |

### Phase 2, stage 1: which pool is dropped

Each release's own top-k kernels, driven on one GPU with frozen synthetic inputs (no model, no server), against an fp64
reference computed from the same quantized bytes. Rank 0 is the best pool; the reference drops rank 511. Share of rows
where the dropped pool ranked better than 480 (well inside the selection), 5 repeats per input:

| Path | When it runs | 1,024 pools | 4,096 | 16,384 | 30,000 |
|---|---|---|---|---|---|
| 0.9.1 tiled | more than 16 rows: 5-8 requests in flight at 3 draft tokens | 94% | 97% | 98% | 2% |
| 0.9.1 fused | up to 16 rows: 1-4 requests in flight | 0% | 0% | 1% | 0.5% |
| 0.7.0 per rank, then cross-rank merge | every decode step | 17-21% | 4-5% | 3-6% | 1-3% |

- On 0.9.1's tiled path the dropped pool is effectively random within the top 512 (median rank about 250) up to 16,384
  pools (about 65,000 tokens). 0.7.0 mostly drops near the boundary at long context.
- The selected set always matched the reference apart from boundary ties (0 mismatches in 113,360 row-repeats); output
  order varied between repeats on every row.
- At exactly 512 pools (2,048-2,051 tokens) every path drops an arbitrary pool.
- Pre-set criterion (amendment 10a) met. This shows a selection-order difference on synthetic inputs only: not yet on real
  serving inputs, not a completion effect, and not an explanation of the GPQA gap. Inferred mechanism for the tiled path: a
  fold over 16,384-pool slices that, with one live slice, sees exactly 512 candidates and emits them unsorted.
- The reference rule (keep the 511 highest-scoring pools) is this investigation's; the engine's source does not state
  which pool should be dropped, and upstream vLLM keeps all 512.

0.10 GPU-hours. Receipts: [`phase2-stage1/`](phase2-stage1) (driver, analysis, per-cell summaries, hashes of the raw rows).
### Phase 2, stage 2: real inputs

Amendment 10b. One 0.9.1 server run with an observe-only probe after the decode top-k recorded, for sampled rows, the
real query bytes, head weights, index-K pools and the kernel's output order; histories were real model reasoning from the
pilot's retained traces at about 4,000-120,000 tokens (questions 55, 109, 170), decoded with 8 requests in flight (tiled
route) and with 4 or fewer (fused route). The same bytes were replayed through each release's kernels.

| Path | Rows | Dropped pool ranked better than 480 | Median rank of dropped pool |
|---|---|---|---|
| 0.9.1 live, tiled | 9,944 | 88.1% | 280 |
| 0.9.1 live, fused | 1,400 | 0.5% | 511 |
| 0.9.1 replay, tiled / fused | | 88.1% / 0.4% | 275 / 511 |
| 0.7.0 path on the same bytes | | 6.9% | 510 |

- By length: 96-97% from 4,000 to 64,000 tokens; 13% at about 120,000 (two live fold slices).
- The route alone decides it: forcing fused-captured rows through the tiled path gives 92%; splitting tiled batches
  into 16-row halves gives 0.4%. Selected sets matched the fp64 reference in every row without corrupted pools.
- Deviations: graph mode off during capture (the indexer is not an eager-break op in this build, so a probe cannot run
  under graphs; the route depends only on row count); the capture needed its one permitted retry. Rows are correlated
  (8 requests, 3 questions). 0.7.0 was replayed on 0.9.1's model state, not run.
- 1.16 GPU-hours. Receipts: [`phase2-stage2/`](phase2-stage2) (probe diff, replay driver, analysis); raw captures are
  local, hashed in [`RAW-SHA256SUMS-phase2-3.txt`](RAW-SHA256SUMS-phase2-3.txt).

### Phase 3: column-511 intervention

Amendment 10c. (i) `B12X_INDEXER_TWO_LEVEL_FOLD=0` (configuration only) and (ii) requesting the kernel's selected scores
and moving the lowest-scoring selected pool into column 511 before the slice (27 changed lines, one file).

| Intervention | Offline, stored stage-2 inputs | In place, one server run (eager) |
|---|---|---|
| (i) fold off | not met: weakest pool in column 511 in 39.8% of tiled rows (the order is still unsorted, from another path) | not run |
| (ii) move the lowest-scoring pool last | met: 100% of tiled rows, 3 repeats; selected set unchanged in 34,032 of 34,032 rows | met: 100% of 10,164 tiled and 2,332 fused rows |

- Source confirms the mechanism: with the default two-level fold the tiled path sees exactly 512 candidates in one live
  slice and takes a quick-exit branch that emits them in arbitrary order.
- Requesting scores only adds the value stores; route and output order are unchanged. Captured in a CUDA graph and
  replayed, identical to eager (offline). No measurable throughput cost at 8 requests (eager).
- "Fused route unchanged" (amendment 10c) is read as "same selection, nothing worse": on the fused route (ii) changes
  which pool is dropped in about 2.6% of rows, always to the lowest-scoring one. Under a byte-identical reading that
  part is not met.
- 0.58 GPU-hours. Receipts: [`phase3/`](phase3) (fix diff, offline and in-place analyses, throughput).

### Kpool tail ring

Amendments 10c-10f. In stage 2's second capture, 16 index-K pools on GPU 1 were byte-identical across two different
requests, some decoding as NaN. A CPU forensic pass showed they were raw indexer key and gate rows for tail-ring slots 4-7
(shared prompt positions 12-15), written through the tail-ring view into blocks then owned by other requests; the
seed-stride and spec-ring fixes behave as intended and the DCP1 tail-compaction fix writes no cache.

- Cause (CPU trace with the real kernels, then GPU unit test): this build's hybrid model state builds attention metadata
  without `positions`, which upstream vLLM has passed since GLM-5.3-Flash support landed. Without positions the tail
  group's intended slot mapping never runs; the generic mapping sends position p ≥ 8 to tail-table column p // 8,
  normally the shared null block. Stale ids come from the model runner's start-up warm-up, which gives every runner row a
  two-block tail entry that is never cleared. No block-table staging race (25,000 steps against a CPU shadow).
- Consequences on GPU (unit test): a pool completed across two steps is compressed from another request's keys
  (relative error 0.17 and 0.73); a stale column plus a prefill from position 0 reproduces the stage-2 signature exactly.
- Shipped graph mode, unmodified 0.9.1, one run: 98% of tail writes went to the null block on each rank; 6 events wrote
  into live attention blocks. With positions passed plus a two-line in-place write so replayed graphs read the mapping:
  0 null-block writes, 0 foreign writes; 104 of 104 requests completed; throughput unchanged (88.3 tok/s at 8 requests).
- The same omission is in 0.7.0's source. A CPU harness running each release's real kernels (synthetic scheduler and
  keys) finds 100% of decode-completed pools wrong on 0.7.0 at 1, 2 and 8 requests, against 0% / 47-57% / 78-90% on
  0.9.1; with positions passed, 0.9.1 is 0% and 0.7.0 keeps errors from its own ring-size and seed bugs (fixed in 0.9.1).
  0.7.0 is the more affected engine. That does not exclude a role in its better completion: greater corruption could
  still change completion by accident, through incorrect computation. The defect's effect on completion is untested
  (corrected 2026-10-11; this line earlier said the defect does not explain 0.7.0's better completion; see the
  [ledger](../../LEDGER.md)).
- Completed fix (amendment 11): positions, upstream's `own_block != 0` guard and the graph-safe write, in two variants:
  T leaves the sparse-MLA causal-length source unchanged, U also switches it as upstream does. One shipped-graph run each:
  0 null-block, 0 foreign and 0 attention-block writes on both ranks; all requests completed. Receipts:
  [`tail-fix/`](tail-fix) (diffs, 17 CPU tests, run summaries, two-rank harness).
- 1.16 GPU-hours. Receipts: [`tail-ring/`](tail-ring) (guard and fix diffs, unit test, CPU mechanism, run summaries).

### Gate: column-511 correction in the shipped graph configuration

Amendments 11-11a. One full server run of the candidate build plus an observe-only device checker, existing fixtures,
shipped FULL_AND_PIECEWISE graphs.

| Condition | Result |
|---|---|
| Observes real graph execution, no eager fallback | for every FULL batch size on both ranks, checker calls = FULL replays x DSA layers; graph counts equal the historical run's; 0 checker errors, 0 fallbacks, 0 host syncs |
| Every affected selection branch covered | the prefill branch was also affected (rows with exactly 512 candidates come back in index order, so the newest pool was dropped; 2,200 rows) and is now corrected; decode rows in mixed batches use the decode branch. After correction: 0 changed selections, 0 rows with a non-lowest pool in column 511, 0 valid pools in unfilled slots, attention's token buffer matched in 16.4 M rows; fp64 re-score of about 59,000 clean sampled rows 100% |
| Corrupted rows reported separately | at least one selected pool with NaN bytes in 7.9% (rank 0) and 32.5% (rank 1) of sampled decode rows: the tail-ring defect, not repaired by this correction; it limits what a negative completion result can exclude |
| Configuration | 23/23 API arguments, 33/33 engine fields and 27/27 VLLM_* variables identical to the historical run |

Before correction in this run: tiled route 94.9% of rows, fused 3.2-3.5%. 0.72 GPU-hours. Receipts: [`gate/`](gate)
(result, frozen manifest, correction diff, checker, analysis, configuration comparison).

### Completion arm

Amendments 11a-11b. 0.9.1 K3.25 with the column-511 selection correction only (image built from the release digest, one
changed file); three GPQA passes under the original protocol, seeds 1234/1235/1236, 8 concurrent, 327,680-token cap,
shipped graph mode, mirroring the comparator's session split. Comparator: the existing clean 0.9.1 record, a historical
control ([comparator audit](arm-col511-plan/COMPARATOR-AUDIT.md): no material unexplained difference). Analysed with the
script frozen before data ([`arm-col511-plan/`](arm-col511-plan), sha256 `b5b13ce2…`), run unchanged, and read by the
frozen rules.

**Result: the column-511 change is a correctness fix that did not move completion behaviour.** Empty replies 15 vs 16 of
594; reading R3: the localized correction stands but is not supported as the dominant completion explanation.

| Of 594 question-passes | 0.9.1 + column-511 correction | 0.9.1 (comparator) |
|---|---:|---:|
| Empty replies | 15 (2.53%) | 16 (2.69%) |
| Per pass 1 / 2 / 3 | 4 / 5 / 6 | 5 / 5 / 6 |
| Ran to the 327,680-token cap | 15 | 15 |
| Stopped early | 0 | 1 (q12 p1, 38 tokens) |
| Questions with any empty reply | 12 | 11 |
| Raw correct (all 594) | 505 (85.0%) | 516 (86.9%) |
| Stated correct (all 594, audited method) | 515 (86.7%) | 522 (87.9%) |
| Answered completion tokens, median | 3,387 | 3,508 |

- **Validity:** every frozen check passed: seeds 1234/1235/1236 and sampling as specified, no re-sent request, all 594
  (question, pass) pairs match on prompt, doc, target and message hashes and on server-rendered prompt token counts, the
  same server-argument fingerprint, and the comparator rows derived by the script reproduce the published rows.
- **Primary:** 13 questions differ in empty replies (7 more often empty on the comparator, 6 on the arm). Exact one-sided
  sign-flip p = 0.50. Empty-rate reduction +0.17 points, 95% question-bootstrap interval -1.01 to +1.35. 15 empties falls
  in the frozen band of 13 or more (little improvement or deterioration), reading R3.
- **Empties moved between questions:** 10 question-passes empty on the comparator were answered by the arm, 9 answered on
  the comparator were empty on the arm, 6 were empty in both.
- **Secondary (accuracy):** raw -1.85 points (95% -5.05 to +1.35), stated -1.18 points (95% -4.04 to +1.52). Not
  significant, which does not establish that the correction costs no accuracy; read the intervals.
- **Stated figure, final:** the 12 arm rows where the stated answer differs from raw were read by hand under the
  protocol's rules (q98p1, q102p1, q159p1, q4p2, q64p2, q72p2, q110p2, q152p2, q170p2, q70p3, q98p3, q134p3). Each
  commits to a single option, the one the adjudicator read; none is hedged. In 5 raw extraction took a parenthesised
  token that is not an option label (a stereodescriptor or a formula group), in 4 an option mentioned after the reply's
  commitment, and in 3 it found no parenthesised label. 11 raise the stated score above raw and 1 lowers it. The frozen
  script, re-run unchanged with no hedged rows, gives the same output: stated 515 of 594 (86.7%) vs 522 (87.9%),
  -1.18 points (95% -4.04 to +1.52), now final.
- **Route exposure:** 76-84% of busy time per pass had 5 or more requests in flight (client-side approximation), so the
  corrected tiled path was heavily exercised. Answered completion lengths are unchanged.
- **Cost:** 17.11 occupied GPU-hours (2 GPUs x server start to stop, two sessions; ceiling 18.5); per-session timings
  [below](#occupied-gpu-hours-of-the-three-k325-records). Cumulative for the investigation: 36.08 GPU-hours. The frozen
  script's comparator figure, 16.593, runs from chain start to chain done and leaves out each session's shutdown;
  start to stop it is 16.63.
- **Limits:** a historical control: exchangeability is assumed, not designed; matching recipes and seeds does not make
  this a randomized experiment. The tail-ring defect is present in both records, which limits what this negative result
  can exclude. Corrupted (NaN-byte) index rows are not measurable from GPQA outputs.

Receipts: [`arm-col511-result/`](arm-col511-result) (frozen script output `result-col511.md` and `.json`; the final
run after the hand read, `result-col511-final.md` and `.json`, identical in content; session costs, GPU load log). Public copies of the plan files replace local paths; [`PUBLIC-COPIES-11b.txt`](PUBLIC-COPIES-11b.txt)
lists frozen and public hashes.

### Occupied GPU-hours of the three K3.25 records

Reconstructed after the arm (amendment 12), CPU only, from logs already on disk. **Occupied** = 2 GPUs x (server start
to server stop): start is the eval chain's start, which stops any old container and launches the server; stop is the
first logged event at or after which the container was stopped. Startup, compilation, warm-up, idle time and shutdown
count. Times are local (UTC-7).

| Record, session (passes) | Start | Ready | First request | Last request ends | Stop | Wall h | GPU-h |
|---|---|---|---|---|---|---:|---:|
| tpurtell 0.7.0 · DFlash2 ×5, 2026-09-29 (1) | 06:51:39 | 07:00:54 | 07:01:03 | 10:34:37 | 10:34:58 | 3.722 | 7.44 |
| tpurtell 0.7.0 · DFlash2 ×5, 2026-10-10 (2-3) | 00:59:36 | 01:07:54 | 01:08:04 | 08:45:28 | 08:45:31 | 7.765 | 15.53 |
| **0.7.0 record, three passes** | | | | | | 11.487 | **22.97** |
| tpurtell 0.9.1 · DFlash2 ×3, 2026-10-08 (1) | 17:26:47 | 17:38:52 | 17:39:03 | 20:01:05 | 20:01:12 | 2.574 | 5.15 |
| tpurtell 0.9.1 · DFlash2 ×3, 2026-10-09 (2-3) | 12:18:31 | 12:30:20 | 12:30:31 | 18:01:59 | 18:02:53 | 5.739 | 11.48 |
| **0.9.1 record, three passes** | | | | | | 8.313 | **16.63** |
| 0.9.1 + column-511 correction, 2026-10-11 session 1 (1) | 12:59:09 | 13:10:45 | 13:10:56 | 15:51:14 | 15:51:20 | 2.870 | 5.74 |
| 0.9.1 + column-511 correction, 2026-10-11 session 2 (2-3) | 15:51:20 | 16:03:40 | 16:03:51 | 21:32:16 | 21:32:22 | 5.684 | 11.37 |
| **Completion arm, three passes** | | | | | | 8.554 | **17.11** |

Sources per session: the eval-chain log (start, "ready after" seconds, chain done), the passive request log (first
request start, last request end) and the driver log that stopped the server. 0.7.0 pass 1 has neither a server log nor
a request log: its first and last request come from lm-eval's log (generation started 07:01:03, 198 of 198 done after
3:33:34, results saved 10:34:37), its ready time from the chain that re-attached to the running server at 06:59:53
("ready after 61s"), and its stop from the next eval chain's start at 10:34:58, whose first action stops the container;
the few seconds that stop itself took are not timed (the other sessions' stop times are taken after the stop returned). The 0.9.1 passes 2-3 stop is an upper bound by up to 7 s: the driver noticed the
chain's end on its 60-second poll at 18:02:46 and stopped the server before its next log line at 18:02:53.

Estimates these replace:
- The 0.9.1 record's three passes were given as 16.59 GPU-hours (amendment 11b; the frozen script's 16.593): chain
  start to chain done, which leaves out the time to stop the server (6 s and 53 s). Start to stop: 16.63.
- Amendment 11's arm estimate (up to 13 GPU-hours) matched the 0.9.1 passes 2-3 session alone (11.48) and left out pass
  1's session (5.15); corrected before launch (amendment 11b, [ledger](../../LEDGER.md)). The arm occupied 17.11.
- A planning figure of about 15.5 GPU-hours for the 0.7.0 three-pass record came from its passes 2-3 server session
  alone (15.53). With pass 1's session the record occupied 22.97. 0.7.0's passes take longer than 0.9.1's (request
  windows 3.56 h for pass 1 and 7.62 h for passes 2-3, against 2.37 h and 5.52 h).

### Comparison with upstream vLLM

All three releases report vLLM `0.1.dev20051+g487ecf187` (2026-08-25), a commit on a GLM development branch that is not an
ancestor of upstream main; several files are vendored from another fork. A CPU audit of the GLM-5.3 paths (indexer, kpool,
sparse MLA, KDA and mamba alignment, DFlash2, model runner) against upstream found no further reachable correctness
divergence beyond the items above; about 60 upstream fixes the build lacks are unreachable under these configurations or
are speed and feature work. Two divergences outside those paths affect every release; neither has been tested alone for completion:

| Item | Detail | Status |
|---|---|---|
| Router logits | on this GPU family the MoE router's fp32-output path is not enabled, so routing logits are computed in bf16 although the checkpoint asks for fp32 routing; upstream enabled it for this family (vllm#54048) | same in all releases; computational effect measured in [stage A](#stage-a-verification-d3-d4-d5) (3.15% of top-8 sets); completion untested alone |
| SwiGLU clamp | the routed experts' path never receives `swiglu_limit` (10); shared experts and dense layers are clamped; an earlier clamp A/B patched a path this quant does not run and is not evidence either way | same in all releases; reachable in served kernels ([bundle gate](#bundle-gate-d1d2d3d4-on-091)); whether limit 10 engages not measured; completion untested alone |

Neither can explain a difference between 0.7.0 and 0.9.1.

<a id="stage-a-verification-d3-d4-d5-in-progress"></a>

### Stage A verification: D3, D4, D5

Amendments 13-14; receipts [`verify-stage-a/`](verify-stage-a). One observe-only server session in shipped graph mode on
the gate fixture (long contexts of 4,240-120,217 tokens), then offline comparisons on identical captured inputs.
**0.6929 of 2.0 GPU-hours** (smoke 2 attempts, server 1, offline 4 attempts; costs in
[`verify-stage-a/costs.jsonl`](verify-stage-a/costs.jsonl)).

| Defect | Confirmed omission | Exercised in this workload | Computational effect | Correction verified | Completion |
|---|---|---|---|---|---|
| D3 router logits in bf16 on SM12x | yes: the gate runs the bf16 fallback (`F.linear`, then cast) with all six fast-path flags off, confirmed in the server | yes (every MoE layer; eager prefill and mixed batches measured) | yes: top-8 membership differs on 3.15% of token × layer rows; routing weights on every row | yes, offline: within τ_d of fp64 on every logit (and as served, [bundle gate](#bundle-gate-d1d2d3d4-on-091)) | untested alone |
| D4 routed SwiGLU limit not passed | yes: projection-mixed path; the layer holds 10, the serving plans none | path exercised; whether activations exceed the limit not measured | not measured here (frozen inputs kept for it) | not here (see the bundle gate) | untested alone |
| D5 workspace not locked after graph capture | yes: never locked | not exercised in this workload: resizes only during profiling, before any graph existed; none after capture or in serving | — | — | untested |

**D3 controls** ([`offline-a4/`](verify-stage-a/offline-a4), all on 1,593 of 1,593 records, 103,608 rows): (a)
dispatch, bf16-valued shipped logits and weight hashes on both ranks; (b) the offline shipped computation reproduces the
captured logits bitwise; (c) the engine's router, invoked as serving does (one call on the serving batch rows, rows in
place, server alignment and router configuration), reproduces the consumed expert ids and weights bitwise; (d) the
corrected computation (`torch.mm(out_dtype=fp32)`, the upstream form) lies within τ_d = 2⁻¹⁴·S + 2⁻²⁴·|ref| of an fp64
reference on all 29,839,104 logits (maximum error/S 7.6e-7), with no top-8 disagreement. The serving-shaped invocation in
(c) was adopted in amendment 14, before any D3 comparison was computed; the criterion stayed bitwise. Offline attempt 3 had
stopped at (c) on 1,425 records: there the control called the router on the 64-row capture window, and the served fused
router computes weights differently above about 1,024 rows (E1, below); expert sets were identical on every row
([`offline-a3/`](verify-stage-a/offline-a3)).

**D3 effect** (shipped "bf16-result GEMM cast to fp32" vs corrected "fp32-output GEMM"):

| | Rows | Top-8 membership differs | 95% interval |
|---|---:|---:|---|
| All | 103,608 | 3,262 (3.15%) | 3.01-3.29% (bootstrap over 176 requests); 2.90-3.50% (over 8 histories) |
| Prefill-only batches, prefill rows | 37,356 | 3.09% | 2.92-3.27% (rows as i.i.d., descriptive) |
| Mixed batches, prefill rows | 50,456 | 3.23% | 3.08-3.39% (descriptive) |
| Mixed batches, decode rows | 15,796 | 3.03% | 2.78-3.31% (descriptive) |
| By layer (42 layers) | 2,368-4,869 each | 1.2% (layer 21) to 6.6% (layer 3) | [`d3_by_layer.csv`](verify-stage-a/offline-a4/d3_by_layer.csv) |

- 3,260 rows swap one expert, 2 rows swap two. Every flip coincides with a shipped-vs-fp64 disagreement; the shipped
  logits lie outside τ_d of fp64 on 72% of logits (maximum error/S 1.7e-3).
- Routing weights differ on every row (median L1 between routing vectors about 0.0018; on rows with an unchanged set,
  p99 of the largest weight change about 0.0016, weights summing to 2.5).
- Near-boundary margins: the corrected gap between the 8th and 9th expert score has median 0.0023-0.0103 by layer;
  on flipped rows 0.00014-0.00034, within the bf16 rounding perturbation.
- Rows are not independent serving draws; the intervals above cluster by request and by history. The fixture is 8 long
  documents, not a GPQA distribution.
- Coverage: FULL-graph decode steps are covered by source inspection only (the capture runs on eager forwards; measuring
  inside FULL graphs would add graph nodes). Decode rows were measured in mixed eager batches.
- Expert flips and weight changes do not establish a completion effect or explain the gap.

D4: the frozen inputs (1,593 records per rank, all paired, nothing missing) are sufficient for a later unclamped vs
limit-10 comparison; none was run. D5: 10 capture bindings per rank; the only resizes (rank 0 once, rank 1 twice) were
during profiling.

Stage A also found and fixed two problems in its own tooling before any D3 comparison: the capture buffer was allocated in
pageable memory in the smoke harness (fixed; pinned memory now enforced), and the offline runner mounted the captures over
the container's /run (fixed).

### Defect matrix (D1-D6)

Status after stage A, the bundle gate, the two bundle passes and amendment 16. D1 and D2 are items 4a-4b of [`FINDINGS.md`](../../FINDINGS.md).

| Defect | Implementation omission | Exercised | Computational effect | Correction verified | Completion |
|---|---|---|---|---|---|
| D1 column-511 selection | yes: the engine keeps 511 of 512 selected pools and no top-k path sorts its output | yes: 88.1% of live tiled decode rows discard a pool ranked better than 480; prefill rows with exactly 512 candidates | yes: which selected pool attention misses | yes: offline, in place, shipped graphs (gate 2026-10-11), and in the bundle gate (0 violations) | tested alone: no measurable change (15 vs 16 empty of 594); bundle passes (seeds 1234, 1235): 3 and 3 empty vs 5 and 5, no clear separation |
| D2 tail ring without positions | yes: attention metadata built without positions | yes: 98% of tail writes to the null block in shipped graph mode, 6 events writing into live attention blocks | yes: completed pools from other requests' keys (CPU harness, GPU unit test) | yes: 0 null-block, foreign or attention-block writes (two variants; bundle gate); NaN-byte index rows 0 in the bundle gate | untested alone; bundle passes (seeds 1234, 1235): 3 and 3 empty vs 5 and 5, no clear separation |
| D3 router logits in bf16 | yes | yes, every MoE layer | yes: 3.15% of top-8 sets, weights on every row | yes: offline (stage A) and as served (bundle gate, bitwise reproduction, within τ_d of fp64) | untested alone; bundle passes (seeds 1234, 1235): 3 and 3 empty vs 5 and 5, no clear separation |
| D4 routed SwiGLU limit not passed | yes | path exercised; whether activations exceed 10 not measured | limit 1 changes outputs (reachability only); effect of 10 not measured | plans carry 10 and the session's compiled kernels carry (True, 10) (bundle gate, criterion revised after data) | untested alone; bundle passes (seeds 1234, 1235): 3 and 3 empty vs 5 and 5, no clear separation |
| D5 workspace not locked after capture | yes (source) | not exercised in this workload (stage A; bundle gate s1) | — | not run; left out of the bundle | untested |
| D6 stale warm-up tail block ids | yes (source; 16 stale rows per rank measured) | the source of D2's foreign writes without D2; unreachable under D2 (stale columns never read) | — | CPU only; left out of the bundle | untested |

"Bundle passes" are two GPQA passes of D1+D2+D3+D4 together, against the matching stock 0.9.1 passes; they attribute nothing to an individual correction.

Other audited items and observations, after amendment 16 ([checkpoint B0](#checkpoint-b0-and-targeted-invariants-amendment-16)).
Item ids D2, D8 and D9 in this table are the [DCP audit](audits/phase1-dcp-attention.md)'s own, not the bundle's.

| Item | Before | After amendment 16 |
|---|---|---|
| Two-GPU all-reduce transport (B12x TP2 peer-push, default on from 0.8.0) | noted in the source audit; no runnable correctness test | bit-exact on B0's shipped graph path (2,040,800 calls, 0 mismatches); the tested transport failure modes retired |
| KV record writer and query projection, 528 vs 656 bytes (audit D2) | floating-point only, unverified | measured equal within rounding ("with errata" verdict, which governs); ruled down as a 0.7.0-vs-B0 mechanism |
| Draft slot sharing (audit D8) | no test that MLA records survive draft writes into the shared tensor | retired for the driven kernels: no write outside a group's page, no stale data across block reuse |
| Tensor offsets above 2³¹ bytes (audit D9) | 64-bit addressing inferred from source; upstream tests skip or use another path | retired for the driven kernels: blocks 934-1,274 and the straddling record correct, eager and under graph replay |
| E1 batch-size-dependent routing weights | measured (stage A) | expected numerical variation of the upstream warp kernel; selection unaffected; same in every release; no workaround |

### Bundle gate: D1+D2+D3+D4 on 0.9.1

Amendments 14-14d; receipts and the full write-up: [`bundle/`](bundle). Candidate image built from the 0.9.1 release
digest with D1 (column-511 correction), D2 (positions, ownership guard, graph-safe tail write), D3 (fp32-output router)
and D4 (`swiglu_limit` 10 on the projection-mixed path); D5 left out (not exercised; its backport reorders warm-up and
capture), D6 unreachable under D2. No limit tuning, routing heuristic or performance change. Three shipped-graph sessions
(s1 all corrections with observers; s3 D4 limit 1; a stock-release probe control) and one offline check: **1.6403 of 2.0
GPU-hours**.

- **Passed as specified (s1):** validity and errors (0 of 140 request errors, no engine, memory or stopping errors,
  configuration identical to the historical 0.9.1 record); D1 (lowest selected pool in column 511 on every checked row
  including FULL-graph replays, fp64 agreement); D2 (0 null-block, foreign, attention-block or tail collisions); D3
  (fp32-output dispatch on 42 layers × 2 ranks, 0 of 69,344 rows bf16-valued); D4 plans at 10.0; D5 not exercised.
- **Failed as specified:** "D4 compiled kernels" (a walk of the plan objects could not reach the compiled kernels, which
  the runtime does not retain) and "P1 probe repeatable" (both greedy rounds identical on all 6 probe prompts: the two
  long-context prompts diverged, in the bundle and in the stock release).
- **Revised after inspecting gate data** (amendment 14c; the original criteria are not described as met): D4 by
  session-specific compiled-kernel evidence (66 kernels with (True, 10.0) written inside s1's window by the candidate's
  plans, none with limit 10 before; s3's with (True, 1.0)); P1 by short-prompt repeatability (met in s1 and s3), with
  long-context non-repeatability reported as a limitation.
- **D4 reachable:** limit 1 vs 10 changes 6 of 6 probe outputs within 16 tokens; the 4 bitwise-repeatable short prompts
  diverge at tokens 0-2.
- **D3 as served:** on 1,000 captured records (69,344 rows), `torch.mm(x, W.T, out_dtype=fp32)` at the serving batch size
  reproduces the served logits bitwise; 0 logits outside τ_d of fp64; routing reproduces the consumed routing bitwise.
- **Before launch:** the GPQA-shaped phase's four early stops (docs 16, 19, 21, 22) have non-empty final content and no
  errors (readiness intact; that phase's throughput unusable); the production image's engine files equal the frozen
  candidate's, with no observer code.

### One-pass GPQA screen

Amendments 14c-14e; receipts [`bundle/screen/`](bundle/screen). The frozen candidate image, all 198 questions, original
protocol (temperature 1.0, top_p 0.95, 8 concurrent, 327,680-token cap, normal stopping), seed 1234, the analysis and
category rules frozen before generation, a 7.5 GPU-hour watchdog, no retry.

**Result: valid; 3 of 198 replies empty, all at the cap (no premature stop); no regression flag, no accuracy concern;
category SIMILAR / NO CLEAR SEPARATION.**

| Pass 1, seed 1234 | Bundle | 0.9.1 record | Column-511 arm | 0.7.0 (context) |
|---|---:|---:|---:|---:|
| Empty replies | 3 (3 cap) | 5 (4 cap, 1 premature) | 4 (4 cap) | 0 |
| Rescued / newly empty with the bundle | | 3 (q12, q109, q120) / 1 (q79) | 1 (q170) / 0 | 0 / 3 (q79, q88, q127) |
| Raw / stated correct (secondary) | 174 / 176 | 169 / 170 | 164 / 165 | 174 / 177 |

- Stated-answer adjudication complete before the category: 2 rows (docs 58 and 98), each commits to one option; neither
  hedged.
- Secondary figures are a single pass and not a decision input; no significance claim is made for them.
- Answered completion tokens: median 3,907, maximum 101,893. 4.84 occupied GPU-hours (ceiling 7.5).
- Under amendment 14c the bundle is deprioritised as a completion remedy within this allocation. This is not a claim that
  the included corrections lack behavioural effects or that their interactions are absent.
- Limits: one pass of a low-rate outcome (P(E ≤ 2) simulated at 0.078 with no change and 0.85-0.86 at 0.7.0-like rates);
  historical controls; matched seeds do not make paired replies once computation diverges.
- The plan records an operator note after the result: the operator considers another pass warranted, citing seed-1234
  luck seen across earlier GPQA runs, and will raise it with the independent assessor. It is not an authorization.
  The independent assessment then proposed one more pass for seed sensitivity (amendment 15), authorized as one pass
  (amendment 15a). The seed-1234 screen keeps its classification.

#### Second pass, seed 1235

Amendments 15-15c; receipts and the full write-up [`bundle/screen2/`](bundle/screen2). The unchanged frozen bundle
(same image, recipe and diff), all 198 questions, original protocol, seed 1235 (the next protocol seed, chosen by
protocol order), run as protocol pass 2 on a fresh server, the same session conditions as the comparators' seed-1235
passes (documented differences: the comparators continued into pass 3 afterwards; repository commit). Runner
(`39ff7616…`) and analysis (`cc3d3000…`) hashes recorded before generation (amendment 15b); 7.5 GPU-hour watchdog; no
retry, no third pass.

Rules frozen before generation. Primary comparator: the stock 0.9.1 record's seed-1235 pass (E = 5); categories by E:
similar or worse for a material concern or E ≥ 5; promising follow-up signal for E ≤ 2 without an accuracy concern;
modest improvement, no clear separation otherwise. Accuracy thresholds (deciding) are an absolute standard from the six
stock-0.9.1 passes (record and arm passes 1-3; raw mean 170.2, SD 3.9; stated mean 172.8, SD 4.6): material concern raw
≤ 158 or stated ≤ 159 or 2 or more premature stops; accuracy concern raw 159-163. Offsets from the single matched pass
are reported but not deciding, because that pass is the family's best (raw 176) and would put a typical pass in the
concern band about half the time with no accuracy change; the dry run labels the arm's own seed-1235 pass (raw 170) "concern".
The category names are amendment 15's: the same E = 3 against 5 was "similar / no clear separation" under the first
screen's rules (amendment 14c).

**Result: valid (30 of 30 checks); 3 of 198 replies empty, all at the cap (0 premature); no material or accuracy concern
on the family standard; category MODEST IMPROVEMENT, NO CLEAR SEPARATION.**

| Seed 1235 | Bundle | 0.9.1 record (primary) | Column-511 arm (secondary) | 0.7.0 (context) |
|---|---:|---:|---:|---:|
| Empty replies | 3 (3 cap) | 5 (5 cap) | 5 (5 cap) | 3 (3 cap) |
| Rescued / newly empty with the bundle | | 2 (q55, q81) / 0 | 5 (q74, q108, q120, q127, q186) / 3 (q79, q121, q147) | 1 (q81) / 1 (q147) |
| Raw / stated correct (secondary) | 168 / 169 | 176 / 178 | 170 / 176 | 175 / 178 |

- Stated-answer adjudication complete before the category: 5 rows (docs 64, 94, 110, 113, 179); doc 94 hedged.
- Against the matched stock pass: raw −8, stated −9 (single pass, secondary, no significance claim). The matched-offset
  readout, not deciding, shows "concern"; 168 lies inside the six-pass family's range (164-176).
- Answered completion tokens: median 3,145, maximum 116,281. 4.541 occupied GPU-hours (2 GPUs × 8,174 s; ceiling 7.5).
- Per amendment 15, direction and uncertainty are reported; no further pass follows automatically.

#### Both passes against the matching stock passes (exploratory)

EXPLORATORY: the second pass was chosen after the first had been inspected; the controls are historical; each bundle pass
reuses its comparators' seed, which matches the sampling stream only until the computation diverges, so the replies are
not paired copies ([confounds](../../CONFOUNDS.md#18-seed-reuse-across-engines)). Question-clustered; 396
question-passes per side.

| Bundle seeds 1234 + 1235 vs | Empty | Empty-rate reduction, points (95%) | Sign-flip p, one-sided | Raw difference, points (95%) |
|---|---|---|---|---|
| 0.9.1 record, passes 1-2 | 6 vs 10 | 1.01 (0.00 to 2.27) | 0.11 | −0.76 (−3.79 to +2.27) |
| Column-511 arm, passes 1-2 | 6 vs 9 | 0.76 (−0.76 to +2.27) | 0.25 | +2.02 (−1.77 to +6.06) |

By seed: bundle 3 and 3, stock record 5 and 5, column-511 arm 4 and 5, stock 0.7.0 (context) 0 and 3. Stated differences
and two-sided values are in the [frozen output](bundle/screen2/run-s1235/screen2-result-final.json).

### Checkpoint B0 and targeted invariants (amendment 16)

Amendments 16-16e; receipts and full write-ups in [`b0-allreduce/`](b0-allreduce), [`e1-kvwriter/`](e1-kvwriter) and
[`b0-cache/`](b0-cache). Amendment 16 adopts an independent assessment that moved the work from behavioural passes to
targeted invariants on the corrected build.

**Checkpoint B0** is the D1-D4 correctness bundle as gated and screened, frozen: image
`sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548`, combined diff `949b2301…`, K3.25 weights
snapshot `0490d2f7`, recipe `glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234` (only the image differs from the stock
0.9.1 recipe), and the effective configuration fingerprinted in the gate and both screens (FULL_AND_PIECEWISE graphs,
DCP1, MLA owners split 25, 528-byte NOPE record, DFlash2 ×3, KV pool 4,707,515 tokens). Its gates and two GPQA passes
stand unchanged. A future intervention is B0 plus one justified change (B1); verified corrections are not removed to
recreate stock behaviour; the stock releases stay historical references, and explaining their completion difference is a
separate question from improving B0. No ablation of the bundle's weak completion signal.

Each item needed a costed proposal before any GPU use and fixed its reference and verdict rules before data. A failure
would be minimised and its smallest correction verified before creating B1; a clean result retires only the failure modes
tested. The gate for another behavioural evaluation is a new verified intervention or a priced confirmation objective.

| Invariant | Coverage | Reference / tolerance | Result | Rules down | Not ruled down | Occupied GPU-hours |
|---|---|---|---|---|---|---:|
| TP2 all-reduce: the transport B0 dispatches under its graphs (B12x PCIe peer-push, up to 48 rows; PyNCCL above) | two GPUs, no model, B0's own communicator and adapter; every engine graph size and launch-geometry edge, FULL and breakable PIECEWISE, 91 calls per replay, address reuse, rank skew, interleaved eager prefill; 26,650 forwards | exact bitwise equality with `(a.float()+b.float()).to(bf16)`, identical on both ranks; PyNCCL cross-check arm | **CLEAN**: 0 mismatched elements in all 41 cases on both ranks over 2,040,800 peer-push calls; epoch counter advanced by exactly the scheduled calls; 900 PyNCCL forwards exact; no hangs | stale, torn, cross-step, wrong-rank, addressing and slot-parity errors and hangs on this path | contention with model kernels, the drafter's exact call sequence, epoch rollover, other communication paths, numerics elsewhere | 0.07 of 0.5 |
| KV record writer and query projection, B0 vs stock 0.7.0 | one GPU per image, no model; real-weight proxies at MLA layers 3, 23, 43 plus a synthetic edge-case set (7.1M latent elements); projection at rows 1-16 | fp64 of the bf16 inputs; a rounding model frozen before data (W1-W4, R, P1, P2), with three errata recorded before GPU data | **within expected rounding** (with errata; governs): byte differences at 6.5e-4 to 1.12e-3 of elements, each one E4M3 code, 90-99% at exact ties, all inside the frozen bracket; scales within 1 ulp; RoPE tail zero; projections bitwise equal on the shared lanes. As first written: beyond expected rounding, reproducing the dry run's defect-free counts | the writer and projection as a 0.7.0-vs-B0 mechanism beyond rounding (DCP-audit D2 now measured) | a completion effect of rounding-level differences; latents captured from a server; the record readers; the record size's effect on page geometry | 0.0069 of 0.25 |
| Cache isolation and high-offset addressing | one GPU, no model, B0's real geometry (2.93 GB shared slot tensor, pages above 2³¹ from block 935, a record straddling 2³¹) and B0's own kernels: draft, KDA, MLA and kpool writers and readers, copy, zeroer, cross-group block reuse, graph replay; addresses shown reachable in serving by simulation with the real allocator | full-size shadow image, byte-exact after every operation, sentinels intact; twins at low and high blocks bit-equal; readers vs fp32 at the upstream tests' thresholds | **all driven kernels pass** (80 of 80 checks in the final rerun) after five fixture faults, found in the data, were fixed; no high-address failure; one upstream test obsolete | 32-bit truncation at blocks 934-1,274 and on the straddling record; writes outside a group's page or into padding; padded-stride errors; stale data across block reuse; replayed graphs using capture-time addresses (DCP-audit D8 and D9 retired for these kernels) | block-table and slot-mapping construction under real scheduling; index-K readers; extend under graph replay; the mamba postprocess kernel; numerics; rank 1 | 0.0392 of 0.5 |
| E1 routing weights above 1,024 rows | CPU only: stage A captures (103,608 rows) and B0 gate captures (34,672 rows) | CPU emulation of each kernel's source arithmetic; an error bound stated before checking | **expected numerical variation**: the warp kernel's `(s+b)−b` reproduces 98% of served rows bitwise; 0 of 729,600 weights exceed its bound | E1 as a defect or as a version difference (same kernel binary in every release); any effect on expert selection | an effect on outputs or completion | 0 |

Total **0.1161 occupied GPU-hours**; cumulative 47.91.

**E1 classification.** vLLM's grouped top-k router uses a block kernel at ≤ 1,024 rows, which weights experts from the
stored sigmoid, and a warp kernel above, which rebuilds it from the biased score as `(s+b)−b` in fp32. With this model's
correction biases of 5.0-14.5 the rebuilt score sits on a grid of 2⁻²¹-2⁻²⁰, about ten times less accurate than the
block path, but exactly what that arithmetic gives: a precision limit of the upstream kernel for models with large
biases, and a batch-invariance property, not a malfunction. The same library is in every release, so it cannot explain a
version difference. No workaround ([`e1-kvwriter/`](e1-kvwriter/README.md#e1-batch-size-dependent-routing-weights)).

**KV-writer criteria: errata found before GPU data.** Finishing the CPU engineering included a dry run of the frozen
analysis on outputs synthesised from the frozen rounding model itself, defect-free by construction. It failed three
criteria, so the criteria were wrong: the P2 bound was half the correct half-ulp; the W4 tolerance was inconsistent with
the frozen error model; the W3 rate prediction omitted exact E4M3 ties, which bf16 inputs hit on 0.3-0.8% of elements.
The errata were written down and hashed in the freeze manifest before the GPU runs; the analysis reports both verdicts.
The operator chose "with errata" to govern (amendment 16b). Under the criteria as first written the verdict is "beyond
expected rounding", and its failures are exactly the dry run's defect-free counts (W3 rates 6.48e-4 to 1.12e-3; P2
5,691-104,184 elements per case, identical across images and between the two projections). Separately, the runner's
analysis step could not load the result files (torch's weights-only loading rejects a version object the investigation's
own script had written); the frozen analysis was re-run unchanged through a loader wrapper that allows that one type,
before any analysis output existed, with the frozen hashes verified.

**Cache fixture: faults found after the data, and fixes.** The first run with data passed 15 of 18 parts; its four
failures were three fixture faults and one obsolete upstream test, none specific to high addresses. Part D2 asked the
extend kernel for a width (64) it does not support (512-2,112; B0 serves 2,048). Part D3 compared the decode binding's
own scratch output with itself. Part E3 compared tail-ring slots that are never written. The upstream B12x GLM_NEXT
high-page test uses an obsolete writer signature (a repaired copy passed). The fixture code was fixed after inspecting
these data; no criterion or tolerance was loosened. Rerun 1 then exposed two more fixture faults (the extend call returns
a tuple; the fixture's own writer replays overwrote its reader sets), fixed the same way. Rerun 2 passed: part D1, D2
(decode and extend at the serving width 2,048, plus decode at 64), D3 (each replay equals an eager read of the same
selection; the control follows the new selection) and E3; 80 checks, 0 failed. Fixture part names are the fixture's own,
not the defect ids. Details: [`b0-cache/`](b0-cache/README.md#what-went-wrong-in-the-fixture-and-what-was-changed).

**Pre-flight fix (amendment 16d).** The cache fixture's first launch was refused by its runner's own pre-flight (no
container, 0 GPU-hours): the GPU-idle check counted the desktop's graphics client (type G, 8 MiB, 0% utilisation) on
GPU 0. Before any data, the check was changed to ignore type-G desktop graphics clients only; the memory and utilisation
limits and the refusal for any compute process are unchanged, as in the other fixture runners.

No item found a defect, so there is no B1, and no behavioural evaluation follows.

### Observations recorded separately

Evidence only; no completion claim.

| Observation | Evidence | Status |
|---|---|---|
| E1: the served fused router's routing weights depend on batch size | Stage A: expert ids unchanged on the captured rows; weights switch between 1,025 and 1,139 rows; against an fp64 recomputation, median relative error 1.2e-7 at ≤ 1,024 rows (12,408 rows) and 1.4e-6 above (91,200 rows; maximum 1.1e-3). Batches that large occur in prefill chunks (inferred exposure: prefill only). CPU assessment (amendment 16b): the kernel used above 1,024 rows rebuilds the unbiased score as (s+b)−b in fp32 with biases 5.0-14.5; emulating that arithmetic reproduces 98% of served rows bitwise, and 0 weights exceed its error model ([`e1-kvwriter/`](e1-kvwriter/README.md#e1-batch-size-dependent-routing-weights)). | Expected numerical variation; selection unaffected; same kernel in every release; no workaround; completion untested |
| Long-context greedy non-repeatability of the stock 0.9.1 release | Bundle gate release control, one request at a time, temperature 0, two rounds: 4 GPQA prompts of 124-158 tokens give bitwise identical 96-token outputs; prompts of 4,240 and 16,240 tokens diverge at token 3, with log-probability differences on every token before divergence; the 4,240-token prompt with identical prefix-cache coverage in both rounds. Same pattern in the bundle sessions (first divergence at tokens 3-75). Divergences occurred where the chosen token's probability was 0.30-0.66; only the chosen token's log-probability was kept, so the margin is bounded, not measured. | Mechanism not identified ([confounds](../../CONFOUNDS.md#16-long-context-greedy-non-repeatability)) |
| NaN-byte index rows | 0 in every branch in the bundle gate's s1; in the 2026-10-11 gate (column-511 correction only), selected pools with NaN bytes in 7.9% / 32.5% of sampled decode rows on ranks 0 / 1 | D2 inferred as the cause; not varied alone |

## Open questions

Stated with their evidence so far.

| Question | Evidence so far |
|---|---|
| Does generation-vs-prefill disagreement at long context bear on completion? | The 8,000-15,999-token measure covers continuation after the reply had ended (amendment 7); no positive evidence either way. |
| What produces the completion gap between 0.7.0 and 0.9.1? | The column-511 drop, the only defect found that differed between the engines in the direction of the gap, is not supported as its explanation: with the correction alone 0.9.1 left 15 empty replies vs 16 ([completion arm](#completion-arm)). The tail ring affects 0.7.0 more, which does not exclude an accidental completion benefit from incorrect computation; its role is untested. The layout bundle showed no reproducible benefit (pilot and replication); router precision and the SwiGLU clamp are the same in all releases; the KV record writer and query projection, which differ between the engines, agree within rounding ([amendment 16](#checkpoint-b0-and-targeted-invariants-amendment-16)). One GPQA pass of 0.9.1 with D1-D4 together left 3 of 198 empty against 5 and 4 in pass 1 of the 0.9.1 record and the column-511 arm (0.7.0 pass 1: 0): similar, no clear separation, one pass ([screen](#one-pass-gpqa-screen)); a second pass at seed 1235 left 3 against 5 and 5 (0.7.0: 3): modest improvement, no clear separation ([second pass](#second-pass-seed-1235)). The gap itself (4 vs 16) rests on one three-pass record per engine. The corrected 0.9.1 arm's 15 empty replies resemble the 0.9.1 record's 16, with empties moving between questions, but it ran a modified engine, so it is not a replication of the unmodified 0.9.1 record. Whether either record's rate reproduces on its unmodified engine has not been tested. Open. |
| Does the column-511 correction cost accuracy? | Raw -1.85 points (95% -5.05 to +1.35), stated -1.18 (-4.04 to +1.52), against a historical control; not significant, which does not establish no cost. Inconclusive. |
| Do the tail-ring fix, the router precision or the SwiGLU clamp change completion or accuracy? | All affect both engines. Each has a verified computational correction (D2, D3) or a reachable one (D4); none was tested alone in GPQA. Together with D1, two passes: 3 and 3 empty of 198 against 5 and 5 in the matching stock passes; raw 174 and 168, stated 176 and 169 (single passes, secondary); no clear separation in either. Untested individually; interactions untested. |
| Does the bundle's empty count differ from the baselines' beyond seed-to-seed variation? | Two passes (seeds 1234 and 1235): 3 and 3 empty against 5 and 5 in the matching stock 0.9.1 passes and 4 and 5 in the column-511 arm's. Exploratory combination (follow-up chosen after the first pass; historical controls; seeds reused): 6 vs 10 of 396, reduction 1.01 points (95% 0.00 to 2.27), one-sided p = 0.11; against the arm 6 vs 9. Rescued and newly empty replies are listed per pass ([screen](#one-pass-gpqa-screen), [second pass](#second-pass-seed-1235)). Inconclusive. |
| Do the bundle's corrections account for the historical gap to 0.7.0? | The bundle corrects verified defects and shows a modest, consistent reduction in empty replies at two seeds (3 and 3 vs 5 and 5 in the matching stock passes); not established. Stock 0.7.0's lower rate (0 and 3 at the same seeds) is not reached, and three of the four corrections (D2, D3, D4) make 0.9.1 more correct than 0.7.0 rather than more like it, so other contributors to the historical gap likely remain unidentified. A cumulative effect of several small corrections is a working hypothesis, not a result. |
| Why does greedy decoding of 0.9.1 diverge between identical long-context requests? | Short prompts repeat bitwise for 96 tokens; 4,240- and 16,240-token prompts diverge at token 3 in the stock release, also with identical prefix-cache coverage; divergences fell on low-margin steps (chosen-token probability 0.30-0.66; runner-up not recorded). Mechanism not identified ([observations](#observations-recorded-separately)). |
| Does E1 (batch-size-dependent routing weights) affect outputs? | Weights differ by up to 1.1e-3 relative above about 1,024 rows (p99 5.6e-7); expert ids unchanged, by construction and on the captured rows; inferred exposure prefill only; classified as expected numerical variation of the upstream kernel, present in every release. Untested. |
| What do the targeted invariants on B0 leave unexamined? | Not ruled down by amendment 16's fixtures: block-table and slot-mapping construction under real scheduling (the class the tail-ring defect D2 belonged to; the cache fixture bypasses it); the index-K readers (B12x paged top-k and the compiled prefill readers); the sparse-MLA extend reader under graph replay; the mamba postprocess kernel; communication paths other than the tested transport (PyNCCL above 48 rows was only cross-checked; the owner exchange; contention with model kernels); the mixed-quant MoE path, whose reference harness was deferred (amendment 16). Rank 1's cache tensors were not driven (same geometry per tensor). No evidence yet points to any of these. |
| Do legitimate numerical differences shift how often long deliberation runs past the cap? | Non-completion is a low-rate tail event (about 1 in 10 fresh draws at these caps on two hard questions); small pilots cannot resolve rates near 2.7% vs 0.7% of replies. |
| Which part of the 0.7.0 layout bundle goes with the higher disagreement? | Step 6 changed EP, DCP, MLA ownership, KV record, draft pages and the top-k merge path together, and still differs from stock 0.7.0 in the top-k merge path, draft depth (×3 vs ×5), vision and engine code. Inconclusive by component. |
| Why does prompt 0 not follow the other prompts in step 6? | One prompt, two runs; inconclusive. |

## Limits

- No reasoning text, token ids or draft-acceptance histories are kept for GPQA runs. The original trajectories cannot
  be replayed, and loops cannot be told from long varied reasoning.
- The records change several things at once (table above). None was varied alone in GPQA.
- The completion difference rests on 13 vs 1 question-passes. The question was raised by pass 1.
- Questions are doc ids; questions, replies and prompts are not published.
- The bundle passes are two single passes against historical controls; their rules allocate resources and make no
  significance or equivalence claim. The second pass was chosen after the first had been inspected, so their combination
  is exploratory, and each reuses its comparators' seed. Two gate criteria were revised after the gate data were
  inspected (amendment 14c).
- FULL-graph decode coverage of the router-precision measurement is by source inspection only.
- Amendment 16's fixtures are model-free: they drive the transport and the cache kernels directly, not the full-layer
  metadata paths, and the KV-writer comparison uses real-weight proxies, not latents captured from a server. Its criteria
  were corrected before GPU data and the cache fixture's after data (fixture faults only); both are disclosed with the
  results.

## Next steps

Nothing further is scheduled. Amendment 16's three targeted checks found no defect, so there is no new verified
intervention and no B1; the plan's gate for another behavioural evaluation (a B0 vs B1 comparison of one verified
change, or a priced confirmation objective) is not met. The first screen's frozen category (amendment 14c) stands. For the second pass's category
amendment 15 prescribes reporting direction and uncertainty with no automatic further pass. The plan notes that
confirmation, if pursued, would prefer a contemporaneous stock control on new prespecified seeds (amendment 15); none is
authorized. The stock 0.7.0 replication stays deferred and is not
authorized (amendments 12, 14). Open, with their
evidence above: what produces the completion gap; whether the bundle's lower count of empty replies holds beyond two
seeds; whether 0.7.0's rate of empty replies reproduces; whether the tail-ring
fix, router precision or the SwiGLU clamp change completion or accuracy individually; whether the column-511 correction
costs accuracy; why long-context greedy decoding is not repeatable; whether E1 affects outputs; the classes the targeted
invariants did not reach (open questions). Minimal reproductions and
patches are prepared; offering them upstream is a separate decision.

## Recompute

```bash
python3 investigations/2026-10-glm53-completion/recompute.py   # Phase 0 decomposition; Phase 1 tables, verdicts (steps 1-6), costs; checks
python3 tools/analyze.py gpqa-join                              # the joined decomposition alone
```

Files: [`PLAN.md`](PLAN.md) (frozen plan), [`manifest.json`](manifest.json) (pinned commits, images, records and
hashes; Phase 1 runs and rules; trace hashes for steps 5-6), [`checks.jsonl`](checks.jsonl) (Phase 0 and 1 checks), [`costs.jsonl`](costs.jsonl) (compute and engineering
time). Local copies of replies and logs are indexed by hash and kept out of this repository.
