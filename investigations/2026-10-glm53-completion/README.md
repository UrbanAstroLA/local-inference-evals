# Completion: why tpurtell 0.7.0 left fewer GPQA answers empty than 0.9.1 (2026-10)

With the same 3.25bpw weights, tpurtell 0.7.0 (DFlash2 ×5) left 4 of 594 GPQA answers empty and tpurtell 0.9.1
(DFlash2 ×3) left 16. This investigation asks why, and whether a small, correct change to 0.9.1 recovers the
difference. Quantization is held fixed.

**Status:** Phase 0 done (existing data only, no GPU time). Phase 1 steps 1-4 done (decode vs prefill, 7.18 GPU-hours).
Plan: [`PLAN.md`](PLAN.md), frozen 2026-10-10, with [dated amendments](PLAN.md#dated-amendments).
**Fast path:** [answer](#answer) · [evidence](#evidence) · [configuration diff](#what-differs-between-the-two-servers) ·
[Phase 1](#phase-1) · [limits](#limits) · [next steps](#next-steps) · [recompute](#recompute)

## Answer

| Finding | Grade |
|---|---|
| The whole 8-answer raw score difference comes from question-passes where 0.7.0 answered correctly and 0.9.1 came back empty. Where both answered (577 of 594), raw accuracy is identical: 516 vs 516 (stated 523 vs 522). | **Descriptive** |
| Every empty answer with a request log ran to the 327,680-token cap, except one that stopped after 38 tokens (0.9.1, question 12, pass 1). No server or transport errors. | **Descriptive** |
| Most of the 0.9.1 empty answers that cost points are on questions that finish well under the cap in other passes. | **Descriptive** |
| The two servers differ in draft depth, parallel layout, KV record, MLA ownership, draft-slot sharing, vision and engine code at once. Sampler, rejection test, KDA code, parsers, weights and rendered prompts are the same. | **Descriptive** |
| Decode vs prefill (Phase 1): no step found 0.9.1's decode agreeing worse with its own prefill than 0.7.0's. At 8,000-15,999 tokens 0.7.0's agreed worse. Below 2,044 tokens disagreement is higher with speculation on than off, in both engines. | **Descriptive** |
| Which difference causes the completion gap. | **Open** |

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
tokens). Number: mean KL over the shared top-20 tokens, all prompts of a run pooled (lower = decode agrees more closely
with prefill). Each step's reading rule was written in [`PLAN.md`](PLAN.md#dated-amendments) before that step's data.
Both engines received the same prompt token ids, so the chat template does not enter.

| Step | Question | Verdict under the step's rule | GPU-hours |
|---|---|---|---:|
| [1](#step-1) | Speculation off: 0.7.0 vs 0.9.1 | Below 2,044: no difference beyond run-to-run variation. From 2,048: both 0.7.0 runs above all three 0.9.1 runs, by 0.0003 (0.7.0 agrees slightly worse) | 1.54 |
| [2](#step-2) | 0.9.1: DFlash2 ×5 vs ×3 | No difference in either region | 1.61 |
| [3](#step-3) | Speculation on: 0.7.0 ×5 vs 0.9.1 ×3 and ×5 | No difference in either region | 0.84 |
| [4](#step-4) | 16,000 tokens, both engines as shipped | 8,000-15,999: both 0.7.0 runs above both 0.9.1 runs, by 0.0081: 0.7.0 as shipped drifts further from its prefill with context. Other bins, reported only: the same order | 3.19 |
| [Across steps](#across-steps) | Speculation on vs off, below 2,044 tokens | Descriptive: in both engines every speculation-on run lies above every speculation-off run | |

Total 3.59 wall hours, 7.18 GPU-hours. **Grade: descriptive.** These steps measure agreement along each run's own
trajectory; they do not measure completion. No step found 0.9.1 agreeing worse than 0.7.0. Where a rule found a
difference (step 1 from 2,048, by a hairline; step 4), 0.7.0 agreed worse.

<details>
<summary>What the measure can and cannot see</summary>

- A run is one generation per prompt. Two runs of one configuration with the same seeds diverge within a few tokens and
  differ in a region's mean by up to about 4× among these runs, so each rule compares ranges of runs, not single
  numbers.
- KL over the shared top-20 tokens ignores probability outside their intersection (PLAN.md, Phase 2 item 4).
- In every run, including the 2026-10-08/09 0.9.1 runs, the engine's start-up warm-up requests (up to 16) ran during the
  first minute of prompt 0; later prompts ran alone. Without prompt 0, step 1's separation from 2,048 disappears
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
- A candidate mechanism, not yet tested: the two upstream kpool issues present in stock 0.7.0 and fixed in 0.9.0 and
  later ([`FINDINGS.md`](../../FINDINGS.md) item 1).
- The bin ends at 15,999 as amendment 4 states. Taking it to the end of each trace instead (including positions from
  16,000) gives 0.0646 and 0.0232 for 0.7.0 and 0.0153 and 0.0144 for 0.9.1; the verdict is the same.
- Six prompts and two runs per engine. Whether agreement at this length relates to completion is open: 0.7.0 left fewer
  answers empty while agreeing worse here.

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

## Limits

- No reasoning text, token ids or draft-acceptance histories are kept for GPQA runs. The original trajectories cannot
  be replayed, and loops cannot be told from long varied reasoning.
- The records change several things at once (table above). None was varied alone in GPQA.
- The completion difference rests on 13 vs 1 question-passes. The question was raised by pass 1.
- Questions are doc ids; questions, replies and prompts are not published.

## Next steps

Ranked by expected information per GPU-hour. GPU-hours = 2 × allocated wall hours. Estimates, not measurements.

| # | Step | GPU-hours | Engineering | Why |
|---|---|---:|---|---|
| 1 | Decode-vs-prefill protocol on the 0.7.0 image, against the existing 0.9.1 runs | ~1 | none (existing client) | Done as [Phase 1](#phase-1) steps 1-4 (7.18 GPU-hours, with speculation and 16,000-token follow-ups) |
| 2 | Speculative commit/rollback invariants at 3 and 5 draft tokens on 0.9.1 (plan 1A) | ~2 | 1-2 days (forced-acceptance hook) | The plan's first priority; checks state after rejections across pool and page boundaries |
| 3 | Indexer selection and 511-pool slice on frozen scores (plan 1B) | ~1 | about a day (extends the kpool tools) | Selection ties and the slice interact differently in the two layouts |
| 4 | Fresh fixtures from score-driving pairs (q109 seed 1234, q55 seed 1235) and a short control, 16K cap | ~1-2 | small | Better fixtures than question 79, which fails on every configuration |
| 5 | Mixed-projection MoE fidelity, EP2 vs rank-sliced (plan 1C) | ~1 | days | Highest engineering cost |

No full GPQA pass until a candidate change exists.

## Recompute

```bash
python3 investigations/2026-10-glm53-completion/recompute.py   # Phase 0 decomposition; Phase 1 tables, verdicts, costs; checks
python3 tools/analyze.py gpqa-join                              # the joined decomposition alone
```

Files: [`PLAN.md`](PLAN.md) (frozen plan), [`manifest.json`](manifest.json) (pinned commits, images, records and
hashes; Phase 1 runs and rules), [`checks.jsonl`](checks.jsonl) (Phase 0 and 1 checks), [`costs.jsonl`](costs.jsonl) (compute and engineering
time). Local copies of replies and logs are indexed by hash and kept out of this repository.
