# Schema

## run.json
`id` (directory name), `config` (path under `configs/` without `.json`), `protocol` (path under `protocols/` without
`.md`), `date` (first day of the run), `software` (harness/client versions), `notes` (anything a reader must know).

## configs/<family>/<id>.json
`model` {repo, revision, quant}, `engine` {project, version, image digest, patches[]}, `serving` {speculative
{method, tokens, draft_model}, draft_slot_sharing, tp, ep, dcp, kv_cache_dtype, vision, prefix_caching,
gpu_memory_utilization}, `hardware` {gpus, driver}, `notes`.

## results.jsonl by protocol
**gpqa-diamond/v1:** `doc_id`, `doc_hash`, `prompt_hash`, `target_hash` (as logged by lm-eval), `pass`, `seed` (lm-eval's
`--seed` for the pass; requests carried seed 1234 unless the run notes say otherwise),
`correct_flexible`, `correct_strict`, `empty` (no answer after reasoning), `response_chars`, `response_sha256`.

**hard-prompt-screen/v0, v1:** `doc_id`, `rep`, `seed`, `cls` (ok / loop / exhaust / error), `finish_reason`,
`stopped_early`, `completion_tokens` (null when the stream was stopped early), `secs`, `tail_zlib_ratio` (compressed /
raw size of the last 30,000 reasoning characters; low = repetitive), `reasoning_chars`, `content_chars`,
`lm_eval_prompt_hash`, `reasoning_sha256`. Rows corrected after collection carry `reclassified` with the reason.

**serving-probe/v1:** rows with `kind` = `request` (`batch`, `doc_id`, `tokens`, `finish`, `ttft_s`, `decode_tok_s`,
`secs`, `output_chars`, `output_sha256`) or `batch` (`batch`, `concurrency`, `temperature`, `max_tokens`, `wall_s`,
`spec` {accepted, drafted, drafts, per_position} from the server's speculation counters).

**kpool-kernel-tests/v1:** `suite` (upstream / rejected-draft), `test`, `outcome` (passed / failed / skipped); the
rejected-draft suite adds `ring_slots` and `key_bytes_wrong`.
