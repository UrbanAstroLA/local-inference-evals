# Schema

## run.json
`id` (directory name), `config` (path under `configs/` without `.json`), `protocol` (path under `protocols/` without
`.md`), `date` (first day of the run), `software` (harness/client versions), `notes` (anything a reader must know).

## configs/<family>/<id>.json
`model` {repo, revision, quant, label}, `engine` {name, built_on, project, version, image digest, patches[] {name,
source, commit, ports}, equivalent_to {version, basis} or null, series}, `serving` {speculative {method, label, tokens,
draft_model}, draft_slot_sharing, tp, ep, dcp, kv_cache_dtype, vision, prefix_caching, gpu_memory_utilization,
layout {label, mla_owners, nope_records, dcp_topk_owner_exchange} (newer configs only), concurrency (requests the client
keeps in flight; present only where it differs from the protocol's setting)},
`hardware` {gpus, driver}, `notes`.

## Labels
Tables, charts and text name a configuration `<weights> · <engine>[ · <layout>] · <speculation>[ · concurrency <n>]`, built from config fields by
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
  `no speculation`; plus `, sharing off` when draft-slot sharing is disabled, and `, prefix cache off` when
  `serving.prefix_caching` is false (for example `no speculation, prefix cache off`). The method's display name comes from
  the config (`DFlash2`; another method would carry its own, such as `MTP`), and each name maps to one
  `serving.speculative.method`.
- **Layout** = `serving.layout.label`, present only when the parallel layout differs from the engine release's default,
  for example `0.7.0 layout (DCP2, EP2)` on a tpurtell 0.9.0 image. `serving.layout` also records MLA layer ownership,
  NOPE record format and the DCP top-k owner exchange where the configuration sets them.
- **Concurrency** = `concurrency <n>` from `serving.concurrency`, present only when the client kept a number of requests in
  flight other than the protocol's (GPQA: 8), for example `4bpw TR3 (Brandon) · tpurtell 0.9.1 · DFlash2 ×3 · concurrency 4`.
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
**gpqa-diamond/v1:** `doc_id`, `doc_hash`, `prompt_hash`, `target_hash` (as logged by lm-eval), `pass`, `seed` (the
request seed: 1233 + pass), `correct_flexible`, `correct_strict`, `empty` (no answer after reasoning), `response_chars`,
`response_sha256`, `finish_reason` and `completion_tokens` (from a passive request log; null when the run had none),
`correct_stated` (secondary, audited score: the reply's stated final answer is the target; see the protocol).
One row per question per pass. Every run has pass 1; a run may hold further passes only when pass *p* carries request
seed 1233 + *p* on every row (`tools/verify.py` enforces it, so passes that share a seed cannot appear).
`correct_flexible` is lm-eval's raw filter, the headline score; it underscores by 0.5 to 3.5 points per pass in the
published passes (see the GPQA protocol's known limitation). Summaries add `accuracy_stated`, its interval, and accuracy
among answered (non-empty) questions for both scores, all over every pass of the run (intervals resample questions with
all their passes); a run of several passes adds `by_pass` (seed, raw and stated accuracy, empty answers per pass),
`pass_spread_flexible_points` and `questions_ever_empty`. `correct_strict` records whether the reply used the phrase "The answer is", which the prompt never asks
for; it is not an accuracy measure.

**hard-prompt-screen/v0, v1, v2:** `doc_id`, `rep`, `seed` (the request seed), `cls` (ok / loop / exhaust / error),
`finish_reason`, `stopped_early`, `completion_tokens` (null when the stream was stopped early), `secs`, `tail_zlib_ratio`
(compressed / raw size of the last 30,000 reasoning characters; low = repetitive), `reasoning_chars`, `content_chars`,
`lm_eval_prompt_hash`, `reasoning_sha256`; v2 adds `min_zlib_check` (the lowest of the early-stop detector's periodic
checks; null if none ran). v0/v1 runs hold repeat 1 of each question only (every repeat sent seed 1234); their
summaries give counts, never rates. A v2 run holds one question with distinct seeds, and its summary adds `seeds` and a
Wilson interval `non_ok_ci95`.

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

## server_log.jsonl (optional, per run)
Numeric fields parsed from the engine's server log, one record per relevant log line; the log text itself is not
published. `kind` = `session` (`session`, `passes`: one server start and the GPQA passes it served; a GPQA run may
hold several), `config` (`max_num_seqs`), `kv_pool` (`tokens`), `kv_memory` (`gib`, KV memory per GPU), `status`
(10-second scheduler line: `t_s` seconds since the first status line of its session, `prompt_tok_s`, `gen_tok_s`, `running`, `waiting`,
`kv_usage_pct`), `spec` (speculation counters per interval: `mean_acceptance_length`, `accepted`, `drafted`) or `event`
(`what` = `preemption` or `cuda_illegal_memory_access`, `t_s`). `tools/verify.py` recomputes the run summary's `server`
block from it: KV pool, peak KV usage, mean generation throughput over intervals with at least 8 running requests,
acceptance, and `waiting_below_max_seqs` (status lines with requests waiting while fewer than `max_num_seqs` ran, i.e.
the KV pool, not the sequence limit, held them back), `max_running`, and `sessions` where session records exist. For GPQA
runs with a request log, status, speculation and event records cover only the measured requests' time window, so the
engine's start-up warm-up requests are excluded. Run notes quote these figures.
