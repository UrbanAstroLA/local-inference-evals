# Termination semantics: K3.25 on tpurtell 0.7.0, 0.9.1, B0 and B1

This is the "termination semantics" bullet of PLAN.md amendment 20 (zero GPU, 2026-10-13).

## Sources

| Source | Location |
|---|---|
| Checkpoint | `<models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1/snapshots/{0490d2f7…, 701cd745…}` |
| vLLM trees | extracted from the release images: `scratchpad/kdcp/{v070,v091}/vllm` and the B0 tree `kdcp/b0-cache/src/vllm` |
| Served configuration | the server logs' `non-default args` |
| Measured tokenizer facts | `termination_tokens.json` (`scripts/fixture_scan.py`, CPU container, no network) |

**Citations.** `vllm/...` paths cite the v091 tree.

- Every file cited below is byte-identical (md5) in the v070, v091 and B0 trees:
  - `sampling_params.py`, `v1/core/sched/utils.py`, `v1/engine/detokenizer.py`, `v1/engine/input_processor.py`
  - `config/model.py`, `entrypoints/openai/chat_completion/protocol.py`
  - `v1/worker/gpu/sample/*`, `v1/worker/gpu/spec_decode/*`, `reasoning/*`, `parser/**`
- B1 = B0 + `d7-fix.diff`, which touches only `model_executor/layers/sparse_attn_indexer_kpool.py`.
- **There is no termination-processing difference between 0.7.0, 0.9.1, B0 and B1.**

## Token table

| Event | Token id(s) | Single or sequence | Stops generation? | In output text? |
|---|---|---|---|---|
| Reasoning open | `<think>` **154841** (added, `special: false`) | single; emitted by the template, not generated (`chat_template.jinja:259-261` → `<|assistant|><think>`) | no | n/a (prompt) |
| **Reasoning close** | `</think>` **154842** (added, `special: false`) | **single token**. The text `"</think>"` encodes to `[154842]`; no ordinary vocab entry contains `</think>`. | **no**: not an eos or stop id, so generation continues into content | yes (non-special, survives `skip_special_tokens`) |
| End of turn (primary eos) | `<|endoftext|>` **154820** | single | yes: `sampling_params.eos_token_id` (tokenizer eos), `sched/utils.py:103-106` | no (special, skipped) |
| End of turn (next-turn marker) | `<|user|>` **154827** | single | yes: from `generation_config.eos_token_id`, `sampling_params.py:669-682` → `stop_token_ids`; `sched/utils.py:108-111` | no |
| Tool-observation marker | `<|observation|>` **154829** | single | yes (same path as 154827) | no |
| Client stop string (lm-eval and pilot clients) | text `"</s>"`, canonically `[522, 82, 29]` (`</`, `s`, `>`) | **multi-token text match** | yes: `finish_reason=stop`, `stop_reason="</s>"`, `detokenizer.py:129-143` | truncated |
| Token cap | `max_tokens` 327,680 (GPQA; pilot 65,536 / 131,072); `max_model_len` 1,048,576 | n/a | yes: `length`, `sched/utils.py:112-117` | n/a |

**Observed in the 24 exact traces** (steps 5-6; GPQA docs 0-5; `ignore_eos`):
- The natural ending is always `…<punctuation>` `</think>` (154842) `<content 303-632 tokens>` `<|user|>` (154827).
- The token before the close was `.`/`).`/`)`/`$.` and never whitespace.
- The first terminator was `<|user|>` in 24/24; `<|endoftext|>` and `<|observation|>` never came first.
- Saved texts contain no literal `</think>`, `</s>` or special-token strings (901/901 rows). No retained reply was
  stopped by the `</s>` string, and none closed early through a spelled-out marker.

## Processing

### 1. Reasoning close

- **Serving:** `--reasoning-parser glm45` → `Glm47MoeParserReasoningAdapter` (`reasoning/__init__.py:55-61`), backed by
  `parser/glm47_moe.py`.
  - With `enable_thinking: true` (sent by lm-eval and every client), the parser starts in `REASONING`
    (`glm47_moe.py:125`, `184-196`).
  - `THINK_END` moves it to `CONTENT` (`glm47_moe.py:110-113`).
- **How the close is detected:**
  - It is detected **by token id** when 154842 appears in the delta (`streaming_parser_engine.py:121-130`,
    `token_id_scanner.py:84-111`).
  - It is **also** detected by the text lexer on the literal string `</think>` (`glm47_moe.py:88-95`,
    `streaming_parser_engine.py:152`). A close spelled out of ordinary tokens (for example `</` `think` `>` =
    `[522, 26779, 29]`) would end reasoning in the API's reasoning/content split, but the model would have seen
    different ids.
  - A `<tool_call>` token (154843) inside reasoning also ends reasoning in the parser (`glm47_moe.py:146-149`).
  - For scoring: the primary event is P(154842). The spelled variants are rare, multi-token, and a different
    model-side event. Score them as sequences only if wanted.
- **Engine-side close test** (`is_reasoning_end` / `extract_content_ids`): token id only (`parser_engine.py:598-618`).
  It is used by structured output and the thinking budget, neither active here.
- **Stop behaviour:** the close **does not stop generation** and is not a stop token. Reasoning close and generation
  stop are separate events.
- **Saved reasoning text:** trailing whitespace before the close is withheld and then dropped
  (`strip_trailing_reasoning_whitespace=True`, `parser_engine_config.py:88`; `parser_engine.py:474-488`).
  - Saved reasoning texts therefore end without any whitespace that preceded `</think>`.
  - In the exact traces there was none, and re-tokenized closes reproduce the logged token count in many rows.
  - Still, the ids right before the close in a text fixture are reconstructed.

### 2. End of turn and EOS

- **Primary eos:** `tokenizer.eos_token_id` = 154820 (`renderers/base.py:312-319`).
- **Generation config:** `generation_config.json` (both snapshots, identical) has `eos_token_id [154820, 154827, 154829]`.
- **Merging:** `input_processor.py:326-331` → `update_from_generation_config` (`sampling_params.py:654-682`):
  - 154820 becomes `eos_token_id`;
  - 154827 and 154829 are appended to `stop_token_ids`;
  - all three go into `_all_stop_token_ids`.
- **Stop check** (`sched/utils.py:94-130`): `min_tokens` gate, then eos, then `stop_token_ids`, then the length cap.
  - Spec decoding: each accepted token is checked in order, and tokens after a stop are trimmed
    (`scheduler.py:2191-2208`).
  - The stop token itself is counted in `completion_tokens` but not emitted as text (`detokenizer.py:108-127`;
    special tokens are skipped).
- **Server and template:**
  - The server adds no stop ids: the request default is `stop_token_ids []` (`protocol.py:271`), and no
    `default_sampling_params` stop ids are set.
  - The chat template adds none.
- **`ignore_eos`** (steps 1-6 only; never in GPQA, pilot or screens):
  - It clears `eos_token_id` and skips adding the generation-config ids to `stop_token_ids`
    (`sampling_params.py:660-661`, `679-682`).
  - It **masks nothing**: masking of stop tokens happens only under `min_tokens`.
- **`stop: ["</s>"]`:**
  - It is checked on the full detokenized output text, which includes the reasoning, because detokenization precedes
    reasoning parsing (`detokenizer.py:129-143`).
  - A reply could therefore be stopped mid-reasoning by that string. It never happened in retained text.

### 3. Transition into final-answer content

- After 154842 the parser emits `content`.
- The turn ends at the next eos or stop id, in practice `<|user|>` after 300-630 content tokens in the exact traces.
- GPQA "valid final answer" therefore needs **two** events:
  1. P(close) at the end of reasoning;
  2. then P(terminator) after a short content phase.
- A reply that closes and then runs to the cap in content would be `closed_then_capped`. No retained row is.
- An eos emitted inside reasoning without a close would give `stop` with empty content. None is retained.
- The 0.7.0 GPQA pass-1 doc-180 row (190k-character *content*, no reasoning kept) shows the content phase can itself
  run long.

## Sampler and masking at the served configuration

Request: T 1.0, top_p 0.95, seed, `max_tokens` 327,680, `chat_template_kwargs {enable_thinking: true}`, stop `</s>`.

- The server's generation-config defaults are the same (`config/model.py:1693-1729`; logged as "overridden by …
  {'temperature': 1.0, 'top_p': 0.95}" on 0.7.0 and B1).
- The engine is Model Runner V2 ("Using V2 Model Runner" in both logs).

| Mechanism | Setting in the served config | Effect on termination tokens |
|---|---|---|
| **top_p 0.95** | active | **The only active mask.** `sampler.py:276` / `topk_topp_sampler.py:397-404`: sort ascending; tokens whose inclusive cumulative probability is ≤ 0.05 get −inf. A termination token is **eligible only if the probability mass of tokens no more likely than it exceeds 0.05**. Below that, its sampling probability is exactly 0 however large its logit share. |
| top_p under spec decoding | same mask | The target logits are processed with the same `apply_sampling_params` (top_p included) before rejection (`rejection_sampler.py:146-170`). Drafts are greedy (`draft_sample_method` default `greedy`), so the output distribution is the target's top_p-masked distribution. |
| Temperature 1.0 | no-op | `sampler.py:204-207`; fp32 Gumbel-max keyed by (seed, position): `gumbel.py`, `sampler.py:277-285`. The flashinfer path is excluded because an explicit seed is set (`sampler.py:262-270`). |
| top_p kernel choice | depends on rows | Triton when ≥ 8 logit rows, PyTorch sort otherwise (`topk_topp_sampler.py:355-364`). The row count depends on concurrency × (draft depth + 1), and 0.7.0 (×5) and 0.9.1 (×3) differ here. Code is identical across engines; boundary rounding could differ only for a token sitting at the 0.05 edge. |
| Repetition / presence / frequency penalty | 1.0 / 0 / 0 (request unset, generation config sets none; `protocol.py:624`, `662-666`) | inactive (`penalties.py:46-54`) |
| min_tokens | 0 | inactive (`logit_bias.py:90-105`) |
| bad_words, logit_bias, allowed_token_ids | unset | inactive |
| thinking_token_budget | unset (the request field defaults to None, `protocol.py:258`) | inactive. It would force 154842 (`sampler.py:193-202`, `thinking_budget.py:81-93`). |
| reasoning_effort | not sent | The template's default system line "Reasoning Effort: Max" is rendered for every request (`chat_template.jinja:2-3`). It is identical in both templates; GPQA prompts render byte-identically (198/198). |
| repetition_detection, structured output | unset | inactive |

**Masking:** the only masking that can remove a termination token at long context is the **top_p 0.95 nucleus**.
For the GPU probe:

- Report, at each designated position, P(154842), P(154820), P(154827) and P(154829) before masking.
- Report the tail mass below each of those tokens, which determines eligibility.
- Report their probability after masking and renormalization.
- Score the close as a single token. The `</s>` stop string and spelled-out closes are multi-token and score as
  sequences.

## 0.7.0 vs 0.9.1 (and B0/B1)

| Item | 0.7.0 (served) | 0.9.1 / B0 / B1 (served) | Termination-relevant? |
|---|---|---|---|
| vLLM base | `v0.1.dev20051+g487ecf187` | same | no |
| Stopping, parser, sampler, spec-verify code | identical md5 | identical | no difference |
| Chat template | checkpoint 701cd745 (no `--chat-template`) | `/opt/glm53/templates/glm53-zai-a5b45eb.jinja` (= 0490d2f7) | renders GPQA identically (198/198) |
| Tokenizer / generation_config | identical bytes | identical | no |
| Draft depth | DFlash2 ×5 | ×3 | changes verify row counts only (top_p kernel path), not the distribution |
| Layout | TP2 + EP2 + DCP2 | TP2, DCP1 | compute path only |
| Other args | `limit_mm_per_prompt`, cudagraph capture 96 | `language_model_only`, 64, `enable_prompt_tokens_details` | no |
| Scheduler diff | prefix-cache back-off under eagle (`scheduler.py:409-411`) | | no |
| Structured-output diff | speculative reasoning guard | | only with a grammar; GPQA uses none |

**Conclusion:** the four engines terminate under the same rules, tokens and masks. A difference in completion must come
from the logits they assign: the termination tokens' probabilities, and the mass beneath them that decides top_p
eligibility.
