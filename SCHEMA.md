# Schema

## run.json
`id` (directory name), `config` (path under `configs/` without `.json`), `protocol` (path under `protocols/` without
`.md`), `date` (first day of the run), `software` (harness/client versions), `notes` (anything a reader must know).

## configs/<family>/<id>.json
`model` {repo, revision, quant, label}, `engine` {name, built_on, project, version, image digest, patches[] {name,
source, commit, ports}, equivalent_to {version, basis} or null, series}, `serving` {speculative {method, label, tokens,
draft_model}, draft_slot_sharing, tp, ep, dcp, kv_cache_dtype, vision, prefix_caching, gpu_memory_utilization,
layout {label, mla_owners, nope_records, dcp_topk_owner_exchange} (newer configs only)},
`hardware` {gpus, driver}, `notes`.

## Labels
Tables, charts and text name a configuration `<weights> · <engine>[ · <layout>] · <speculation>`, built from config fields by
`config_label()` in `tools/verify.py` (the site uses the same function). Example: `4bpw TR3 (Brandon) · tpurtell 0.8.0 + kpool fixes ≈ 0.9.0 · DFlash2 ×3`.

- **Weights** = `model.label`. Each label names exactly one weights repository, and each repository has one label.
  Labels give bit width, then format family and source when two checkpoints share a bit width:
  - `3.25bpw`: wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1.
  - `4bpw TR3 (Brandon)`: Brandon M. Music's TR3 checkpoint brandonmusic/GLM-5.3-Flash-tr3-4bpw.
- **Engine** = `engine.name`, the version without its leading `v`, then ` + <name>` for each locally applied patch set
  in `engine.patches`, then ` ≈ <version>` when `engine.equivalent_to` records that the patched build matches a later
  release for everything measured here (`basis` says why). The engine name comes first because engines number their
  versions independently: `tpurtell 0.9.0` is a tpurtell release, not vLLM 0.9.0. Each engine label names exactly one
  build (project, version, image digest, patches), and each build has one label.
- **Speculation** = `<serving.speculative.label> ×<tokens>` (draft tokens per step), for example `DFlash2 ×3`, or
  `no speculation`; plus `, sharing off` when draft-slot sharing is disabled. The method's display name comes from
  the config (`DFlash2`; another method would carry its own, such as `MTP`), and each name maps to one
  `serving.speculative.method`.
- **Layout** = `serving.layout.label`, present only when the parallel layout differs from the engine release's default,
  for example `0.7.0 layout (DCP2, EP2)` on a tpurtell 0.9.0 image. `serving.layout` also records MLA layer ownership,
  NOPE record format and the DCP top-k owner exchange where the configuration sets them.
- Every configuration label names exactly one configuration.
- `engine.built_on` says what the engine is built on (shown in the label key). `engine.series` groups builds that share
  one code base; charts give each series one colour. tpurtell 0.8.0, 0.8.0 + kpool fixes, 0.9.0, 0.9.0 + DCP1 tail fix and 0.9.1 are one series;
  tpurtell 0.7.0 (with or without the fixes) is another.

`tools/verify.py` fails if any of these fields is missing or if a label is ambiguous. Results from another engine need
only these fields in their configs: its own `name`, `built_on` and `series`, its own version scheme, and a
`serving.speculative.label` for any new speculative method.
Config and run ids (for example `k4-v0.8.0-kpoolfix-dflash3`) are stable identifiers, not labels. The existing ids
omit the engine name because every configuration so far is a tpurtell build; ids for other engines include it after the
weights (`<weights>-<engine>-<version>[-<patch>]-<speculation>`).

## results.jsonl by protocol
**gpqa-diamond/v1:** `doc_id`, `doc_hash`, `prompt_hash`, `target_hash` (as logged by lm-eval), `pass`, `seed` (lm-eval's
`--seed` for the pass; requests carried seed 1234 unless the run notes say otherwise),
`correct_flexible`, `correct_strict`, `empty` (no answer after reasoning), `response_chars`, `response_sha256`.
`correct_flexible` is lm-eval's raw filter and underscores by 0.5 to 2.5 points per pass (see the GPQA protocol's known
limitation). `correct_strict` records whether the reply used the phrase "The answer is", which the prompt never asks
for; it is not an accuracy measure.

**hard-prompt-screen/v0, v1:** `doc_id`, `rep`, `seed`, `cls` (ok / loop / exhaust / error), `finish_reason`,
`stopped_early`, `completion_tokens` (null when the stream was stopped early), `secs`, `tail_zlib_ratio` (compressed /
raw size of the last 30,000 reasoning characters; low = repetitive), `reasoning_chars`, `content_chars`,
`lm_eval_prompt_hash`, `reasoning_sha256`. Rows corrected after collection carry `reclassified` with the reason.

**serving-probe/v1:** rows with `kind` = `request` (`batch`, `doc_id`, `tokens`, `finish`, `ttft_s`, `decode_tok_s`,
`secs`, `output_chars`, `output_sha256`) or `batch` (`batch`, `concurrency`, `temperature`, `max_tokens`, `wall_s`,
`spec` {accepted, drafted, drafts, per_position} from the server's speculation counters).

**decode-prefill-consistency/v1:** `doc_id`, `i` (position = causal length of the decode step), `mod4`, `region`
(`lt2044`, `2044-2047`, `ge2048`), `abs_dlp`, `top1_agree`, `kl_top20` (null when fewer than 2 shared top-20 tokens).

**kpool-tail-index/v1:** `layout` (packed / scattered), `length`, `nsa_len`, `selected`, `attended`, `tail`,
`tail_cols_before_mask`, `tail_cols_after_mask`, `tail_attended`, `n_dropped`, `equals_dense_prefix` (null above 2,047),
`row_sha256_16`.

**tool-eval-bench/v1:** `rep`, `scenario_id`, `status` (pass / partial / fail), `points` (2 / 1 / 0).

**kpool-kernel-tests/v1:** `suite` (upstream / rejected-draft), `test`, `outcome` (passed / failed / skipped); the
rejected-draft suite adds `ring_slots` and `key_bytes_wrong`.
