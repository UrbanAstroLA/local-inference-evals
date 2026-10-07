# Schema

## run.json
`id` (directory name), `config` (path under `configs/` without `.json`), `protocol` (path under `protocols/` without
`.md`), `date` (first day of the run), `software` (harness/client versions), `notes` (anything a reader must know).

## configs/<family>/<id>.json
`model` {repo, revision, quant, label}, `engine` {name, built_on, project, version, image digest, patches[] {name,
source, commit, ports}, equivalent_to {version, basis} or null, series}, `serving` {speculative {method, tokens,
draft_model}, draft_slot_sharing, tp, ep, dcp, kv_cache_dtype, vision, prefix_caching, gpu_memory_utilization},
`hardware` {gpus, driver}, `notes`.

## Labels
Tables, charts and text name a configuration `<weights> · <engine> · <speculation>`, built from config fields by
`config_label()` in `tools/verify.py` (the site uses the same function). Example: `4bpw · tpurtell 0.8.0 + kpool fixes ≈ 0.9.0 · 3 drafts`.

- **Weights** = `model.label`. Each label names exactly one weights repository, and each repository has one label.
- **Engine** = `engine.name`, the version without its leading `v`, then ` + <name>` for each locally applied patch set
  in `engine.patches`, then ` ≈ <version>` when `engine.equivalent_to` records that the patched build matches a later
  release for everything measured here (`basis` says why). The engine name comes first because engines number their
  versions independently: `tpurtell 0.9.0` is a tpurtell release, not vLLM 0.9.0. Each engine label names exactly one
  build (project, version, image digest, patches), and each build has one label.
- **Speculation** = `N drafts` (draft tokens per step), `no speculation`, plus `, sharing off` when draft-slot sharing
  is disabled. If configurations with different speculative methods are shown together, the method is prefixed.
- `engine.built_on` says what the engine is built on (shown in the label key). `engine.series` groups builds that share
  one code base; charts give each series one colour. tpurtell 0.8.0, 0.8.0 + kpool fixes and 0.9.0 are one series;
  tpurtell 0.7.0 (with or without the fixes) is another.

`tools/verify.py` fails if any of these fields is missing or if a label is ambiguous. Results from another engine need
only these fields in their configs: its own `name`, `built_on` and `series`, and its own version scheme.
Config and run ids (for example `k4-v0.8.0-kpoolfix-dflash3`) are stable identifiers, not labels. The existing ids
omit the engine name because every configuration so far is a tpurtell build; ids for other engines include it after the
weights (`<weights>-<engine>-<version>[-<patch>]-<speculation>`).

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
