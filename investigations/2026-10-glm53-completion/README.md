# Completion: why tpurtell 0.7.0 left fewer GPQA answers empty than 0.9.1 (2026-10)

With the same 3.25bpw weights, tpurtell 0.7.0 (DFlash2 ×5) left 4 of 594 GPQA answers empty and tpurtell 0.9.1
(DFlash2 ×3) left 16. This investigation asks why, and whether a small, correct change to 0.9.1 recovers the
difference. Quantization is held fixed.

**Status:** Phase 0 done (existing data only, no GPU time). Plan: [`PLAN.md`](PLAN.md), frozen 2026-10-10.
**Fast path:** [answer](#answer) · [evidence](#evidence) · [configuration diff](#what-differs-between-the-two-servers) ·
[limits](#limits) · [next steps](#next-steps) · [recompute](#recompute)

## Answer

| Finding | Grade |
|---|---|
| The whole 8-answer raw score difference comes from question-passes where 0.7.0 answered correctly and 0.9.1 came back empty. Where both answered (577 of 594), raw accuracy is identical: 516 vs 516 (stated 523 vs 522). | **Descriptive** |
| Every empty answer with a request log ran to the 327,680-token cap, except one that stopped after 38 tokens (0.9.1, question 12, pass 1). No server or transport errors. | **Descriptive** |
| Most of the 0.9.1 empty answers that cost points are on questions that finish well under the cap in other passes. | **Descriptive** |
| The two servers differ in draft depth, parallel layout, KV record, MLA ownership, draft-slot sharing, vision and engine code at once. Sampler, rejection test, KDA code, parsers, weights and rendered prompts are the same. | **Descriptive** |
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
| 1 | Decode-vs-prefill protocol on the 0.7.0 image, against the existing 0.9.1 runs | ~1 | none (existing client) | Is 0.9.1's residual decode error specific to the newer engine? |
| 2 | Speculative commit/rollback invariants at 3 and 5 draft tokens on 0.9.1 (plan 1A) | ~2 | 1-2 days (forced-acceptance hook) | The plan's first priority; checks state after rejections across pool and page boundaries |
| 3 | Indexer selection and 511-pool slice on frozen scores (plan 1B) | ~1 | about a day (extends the kpool tools) | Selection ties and the slice interact differently in the two layouts |
| 4 | Fresh fixtures from score-driving pairs (q109 seed 1234, q55 seed 1235) and a short control, 16K cap | ~1-2 | small | Better fixtures than question 79, which fails on every configuration |
| 5 | Mixed-projection MoE fidelity, EP2 vs rank-sliced (plan 1C) | ~1 | days | Highest engineering cost |

No full GPQA pass until a candidate change exists.

## Recompute

```bash
python3 investigations/2026-10-glm53-completion/recompute.py   # decomposition, end of each empty answer, checks
python3 tools/analyze.py gpqa-join                              # the joined decomposition alone
```

Files: [`PLAN.md`](PLAN.md) (frozen plan), [`manifest.json`](manifest.json) (pinned commits, images, records and
hashes), [`checks.jsonl`](checks.jsonl) (Phase 0 checks), [`costs.jsonl`](costs.jsonl) (compute and engineering
time). Local copies of replies and logs are indexed by hash and kept out of this repository.
