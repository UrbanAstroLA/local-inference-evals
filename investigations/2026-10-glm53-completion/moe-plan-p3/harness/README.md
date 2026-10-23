# P3 Stage 1 harness: decode plan vs prefill plan on identical rows

Design: `../PROTOCOL.md`. Authorization: `../AUTHORIZATION.txt` (GPU0 only, at most 0.5 occupied GPU-hours including
load, plan and compile). Line 2 of that file (operator, 2026-10-23) allows retries: all attempts together stay
within 0.5 GPU-h, and each attempt is logged with its reason. See "Running it".

**State:** built, CPU-tested, then run on GPU (attempts 1-2, 0.1458 GPU-h). The results are in `../RESULT.md`; this
page describes the harness and was written before the GPU runs (its time estimates are pre-run).

## Files

| File | What it is |
|---|---|
| `moe_plan_eq.py` | Runs inside one stock image on `cuda:0`. `--engine v091\|v070 --out OUT.json [--dry] [--smoke] [--deadline-s N]` |
| `run_p3.sh` | GPU wrapper: gates, budget, one attempt directory, cost lines |
| `compare_eversion.py` | CPU only. Compares the same slice and plan across the two images (E-version) |
| `../tests/run_cpu_tests.sh` | All CPU tests. Results: `../tests/RESULTS.txt` |
| `../SHA256SUMS` | Hashes of `harness/` and `tests/` |

## What runs where

Both images take the same EXL3 path for K3.25. The routed experts are projection-mixed (K3/K4 per projection), go
through `_rank_sliced_runtime`, and are dispatched by m: m ≤ 32 uses the decode plan, m > 32 the prefill plan.

| Engine | Image | B12x | Rank-0 slice the harness builds ("own slice") |
|---|---|---|---|
| `v091` | `a9e985c0` | `7fcc094e` | `tp2r0`: TP2 rank 0. All 288 experts, half of each expert's intermediate (1,024) |
| `v070` | `48e254d9` | `fe054789` | `ep2r0`: EP2 rank 0. Linear placement, experts 0-143, full intermediate (2,048), precomposed route map |

The B12x commits are the images' `io.tpurtell.b12x.commit` labels. The harness checks that `exl3.py` has the sha256
expected for `--engine`, and refuses with rc 3 on a mismatch (tested).

## Reused versus re-implemented

**Reused from the image (called, not copied).** These are the same objects vLLM calls at start-up and in `forward`:

1. `Exl3Config.from_config(config.json quantization_config)` and `maybe_update_config(model_dir, hf_config)`. These
   hydrate `tensor_storage` from `quantization_config.json` and build the per-projection bitrates.
2. `Exl3MoEMethod(quant_config, moe).create_weights(...)`, with the kwargs `RoutedExperts.__init__` passes. This
   gives the TP slice dims, the EP global→local map and the preallocated slabs.
3. The image's `Exl3MoEParameter.weight_loader` (`_exl3_moe_weight_loader`) for every expert tensor. It does the TP
   narrowing and the EP mapping.
4. `process_weights_after_loading`, which calls `_prepare_projection_mixed_weights` and then `api.plan_weights` and
   `api.prepare_weights` with vLLM's own arguments, plus the EP route map.
5. `_rank_sliced_runtime`. It builds both plans, the Caps, scratch, output buffers and the runtime cache, under the
   served env.
6. `_bind_rank_sliced` followed by `runtime["api"].run(binding=…)` and `.to(x.dtype)`. These are the lines
   `_apply_rank_sliced` runs for a chosen plan.
7. `Exl3MoEMethod.apply(...)` on every fixture, as a cross-check of the served dispatch. Its output must be bit-equal
   to the harness's invocation of the plan that dispatch selects (`apply_dispatch`).
8. `determine_expert_map(2, 0, 288, "linear")` and `MoEActivation.SILU`.

**Re-implemented in the harness, and why:**

- **Checkpoint iteration.** The harness goes from the safetensors index to (param, global expert id, shard id), as
  `glm5next` `load_weights` and `build_expert_params_mapping` do. This avoids building the whole model.
  - Non-local EP experts are skipped before the read. The served loader returns `False` for them.
- **A minimal `RoutedExperts` stand-in.** It is an `nn.Module` with the attributes `exl3.py` reads, plus `moe` and
  parallel-config namespaces. Building a real `RoutedExperts` needs a full `VllmConfig` and a distributed group.
- **Two module globals patched only for the duration of `create_weights`:**
  - `exl3.get_current_vllm_config_or_none` returns a scheduler with `max_num_batched_tokens=2048` and
    `max_num_seqs=16`, and `runner_type="generate"`.
  - `vllm.model_executor.parameter.get_tensor_model_parallel_rank` / `_world_size` return (0, 2).
    `BasevLLMParameter.__init__` records them (`parameter.py:65-66`). `Exl3MoEParameter` does not use them; it
    slices with its own `exl3_tp_rank` and `exl3_tp_size`.
- **Plan choice.** The bind and run lines of `_apply_rank_sliced` are called with an explicitly chosen plan instead of
  the m branch. This is the only way to run both plans on the same m.
- **Decode plan for m > 32.** Rows go to the decode plan in ≤ 32-row chunks (32, then the remainder), and the outputs
  are concatenated. This mirrors the tiling loop of `_apply_rank_sliced`. The served path never does this; it is the
  comparison the proposal asks for.
- **`MAX_M=16` check.** `_rank_sliced_runtime` is called under `VLLM_EXL3_TRELLIS_MAX_M=16` and
  `VLLM_EXL3_PREFILL_TRELLIS=0`.
  - The decode plan is built by the same `_plan_with_scratch(max_trellis_m, block_m)` call.
  - Disabling prefill only avoids a second ~300 MiB arena and its compile.
  - Run on `v091` only, the engine the switch was proposed for.
- **Fixtures, comparisons, digests.**

**The b12x tests are not used.** `tests/moe` (`test_benchmark_mixed_trellis.py`,
`test_benchmark_qsrt_checkpoint_profiles.py`, the loader tests) cover QSRT/btx and synthetic tiers. Nothing in them
builds a projection-mixed EXL3 plan from this checkpoint. `exl3.py` is the only in-image path that does, so it is
reused rather than re-derived.

**Verbatim-line checks.** `tests/check_signatures.py` asserts, in each image:

- the signatures of every callable above;
- that the served-path source lines the harness mirrors are present verbatim: the dispatch condition, the
  bind/run/`.to(x.dtype)` lines, the env parsing in `_rank_sliced_runtime`, the input normalization in `apply`, the TP
  slice dims in `create_weights`, and `tile_config = None` / `exl3_trellis_mcg` in the projection-mixed prepare.

Definition lines (from that test):

| Function | v091 | v070 |
|---|---:|---:|
| `create_weights` | 1810 | 1708 |
| `process_weights_after_loading` | 2000 | 1898 |
| `_prepare_projection_mixed_weights` | 2351 | 2249 |
| `_rank_sliced_runtime` | 2824 | 2717 |
| `_bind_rank_sliced` | 3135 | 3042 |
| `_apply_rank_sliced` | 3164 | 3076 |
| `apply` | 3349 | 3261 |

All in `vllm/model_executor/layers/quantization/exl3.py`.

## Served values and their sources

| Setting | Value | Source |
|---|---|---|
| `VLLM_EXL3_TRELLIS_MIN_M` | 1 | 0.7.0 recipe `glm53-flash-exl3-k3.25@tpurtell/recipe.json:22`; 0.9.1 recipe `…+v0.9.1/recipe.json:58`; image ENV |
| `VLLM_EXL3_TRELLIS_MAX_M` | 32 | recipes :33 / :34; image ENV; log "Trellis m=1..32" |
| `VLLM_EXL3_TRELLIS_BLOCK_M` | 8 | recipes :52 / :26; log "block_m=8" |
| `VLLM_EXL3_PREFILL_TRELLIS` | 1 | recipes :34 / :53; log "prefill trellis" |
| `VLLM_EXL3_PREFILL_BLOCK_M` | 64 | recipes :40 / :59; log "prefill trellis block_m=64" |
| `VLLM_EXL3_PREFILL_CAPACITY` | 1024 | recipes :35 / :13; log "capacity=1024" |
| `VLLM_EXL3_PREFILL_CHUNK` | unset (128) | not in either recipe; log "chunk=128" |
| `B12X_EXL3_BF16_EPILOGUE` | 0 | recipes :45 / :46; `start.sh` default `:-0` (v0.7.0 :460, v0.9.0 :527) |
| `B12X_EXL3_BF16_GEMV` | 1 | recipes :48 / :54. A `B12X_*` variable is part of B12x's compile-env key, so it is set as served |
| `max_num_batched_tokens` / `max_num_seqs` | 2048 / 16 | both served logs, non-default args (line 7) |
| Parallelism | 0.7.0 TP2 + EP2 + DCP2; 0.9.1 TP2 | same lines; 0.7.0 log :71 "Expert placement strategy: linear. Local/global 144/288" |

**Logs.** 0.7.0: `<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+ps.server.log` (sha256 `30e9e9da…`). 0.9.1:
`…@tpurtell+v0.9.1+ps.server.log` (sha256 `0e05caef…`).

**Plan lines.** 0.7.0 :238 (arena 334.8 MiB). 0.9.1 :236 (arena 262.6 MiB).

**Recipes.** Both recipes' env blocks also give `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; the wrapper sets
it.

**Weights.** Snapshot `0490d2f7`, mounted read-only.

- The 0.9.1 record served from `0490d2f7`. The 0.7.0 record served from `701cd745`.
- Every weight file in the two snapshots links to the same blob; only `chat_template.jinja` and `README.md` differ.
  Checked by comparing symlink targets.
- The wrapper mounts the whole `models--…` directory read-only, because the snapshot files are symlinks into
  `../../blobs`.

## Layers and slice checks

**Layers:** 3, 23 and 44 (early, middle, late; the MoE layers are 3-44). Each layer's experts sit in one shard file
(9, 6 and 15).

**Tier membership.** The rule `exl3.py` logs, applied to the slice the image's code builds on CPU, reproduces the
served start-up logs exactly for all six layer×slice pairs (`tests/out-dry-*.json`, `membership_matches_served_log`):

| Layer | tp2r0 (0.9.1 log) | ep2r0 (0.7.0 log) |
|---|---|---|
| 3 | (3,248,221,180),(4,40,67,108) :111 | (3,126,109,99),(4,18,35,45) :117 |
| 23 | (3,248,221,180),(4,40,67,108) :153 | (3,131,113,94),(4,13,31,50) :159 |
| 44 | (3,247,220,180),(4,41,68,108) :196 | (3,120,104,88),(4,24,40,56) :205 |

**On GPU** the harness also records the prefill arena size. For the engine's own slice it compares that size with the
log (`arena_matches_served_log`).

## Fixture panel (deterministic; identical bytes in both images)

**Seeds.** Fixtures are generated on CPU with `torch.Generator`, seeded from sha256 of the fixture key. Both images
ship torch 2.13.0. Each fixture records an `input_digest`. The two dry runs produced identical digests for every
common fixture (`tests/out-eversion-dry.json`: 0 mismatches).

**Main panel per layer:**

- **Shapes (m):** 1, 8, 16, 24, 32, 40, 48.
- **Activation scale:** ×0.25, ×1, ×4 of a reference RMS. The reference is the RMS of that layer's
  `post_attention_layernorm.weight`, which is the MoE input norm in `glm5next` (`model.py:441-444`).
- **Heavy-tailed rows:** every row r with r % 4 == 3 has 4 channels ×30.
- **Expert concentration:**
  - spread: random distinct top-8 out of 288;
  - concentrated: all rows on the same 8 experts, drawn from 0-143 so they are local to both rank-0 slices;
  - mixed: even rows concentrated, odd rows spread.
- **Routing weights:** uniform 1/8; peaked (0.9 on one route); near-zero tails (1e-4 on seven routes).
- **Seeds:** 0 and 1.
- **Count:** 378 fixtures.

**Padded cells:** (24,20), (32,28), (40,36), (48,42).

- Each cell is run with the pad rows filled two ways:
  - "runner": one synthetic row repeated, modelling zero-filled `input_ids`/positions;
  - "garbage": independent rows at ×4 with random routes and weights.
- Pad rows are excluded from every comparison.
- `pad_content_independent` checks that the real rows are bit-equal between the two pad fillings, within each plan.
- 48 fixtures. Own slice total: **426 per layer**.

**Positive controls,** on m=32 and m=48, x1, mixed, uniform, seed 0, row 2 (a concentrated row, local in both
slices):

- (a) +1 on the bf16 bit pattern at the row's largest-magnitude channel (one ulp);
- (b) route 0 replaced by a local expert not already in the row.

Each must change exactly row 2 in both plans (`pass`). GPU controls are judged on the served bf16 output. The raw
output is recorded alongside: fp32 on TP plans, bf16 on EP.

**Small-shape configuration check** (v091 own slice): m ∈ {1,2,4,8,12,16} × 3 concentrations × 2 seeds. It compares
the decode plan under `MAX_M=32` with the decode plan under `MAX_M=16`.

**E-version subset:** m ∈ {8,32,48}, x1, uniform, 3 concentrations, 2 seeds, which is 18 per layer.

## Per fixture: the plan invocations

**Order.** Decode, prefill, decode, prefill, decode, prefill, then `apply`.

**Plans.**

- **Decode:** exact-width launches at m ≤ 32, and ≤ 32-row chunks above that.
- **Prefill:** the single 1,024-capacity launch with active m.

**Launches.** The bound launch for every (plan, m) is read from the binding and recorded in `launch_map`: `size_m`,
fc1/fc2 tile K/N, `moe_block_size`, `max_m_blocks`, `blocks_per_sm`, route-id dtype, and direct or packed routing.
The tile is therefore measured, not inferred.

## Outcome fields (`out/attempt-N/<engine>.json`)

**Per fixture** (`slices.<slice>.layers.<L>.fixtures[]`):

- **Identity:** `id`, `input_digest`, `n_real`, `heavy_rows`, `conc_rows`.
- **`within.decode` / `within.prefill`:**
  - `bitequal`: all 3 repeats bit-equal;
  - `spread`: rows that differ and their max abs, per repeat.
- **`between`:** decode against prefill on the real rows.
  - `row_bitequal_frac`, `rows_differ`, `n_rows_differ`, `n_elem_differ`;
  - `max_abs`, `rel_l2` (‖D−P‖/‖P‖), `max_row_rel_l2`;
  - `per_row`: elements, max abs and rel L2 for each differing row;
  - `nonfinite`.
- **`between_raw_fp32`:** the same comparison on the fp32 kernel output before `.to(bf16)`. TP plans only.
- **`apply_dispatch`:** `apply()` against the plan the served dispatch selects (`plan`, `bitequal_all_rows`).
- **`pad_content_independent`:** padded cells only.
- **`digests`:** 8-byte blake2b per output row, per plan.
- **`zero_rows_decode`**, `t_ms`. On failure: `error` and `tb`.

**Per layer:**

- `build`: load, prepare and slice facts, membership against the log, weight plan.
- `controls[]`, `small_shape[]`, `launch_map`.
- `summary`: all-bit-equal flags, counts by m, max magnitudes, dispatch, controls, pad independence.

**Per slice:** `runtimes.default` and `runtimes.maxm16`. Each holds Caps, launch token counts, arena MiB,
`t_plan_compile_s`, and the comparison with the served arena.

**Top level:**

- `status`: `complete`, `partial` (soft deadline) or `error`;
- `stopped_reason`, `timings`, `env` / `env_matches_served`;
- versions, `code_hashes` (sha256 of every reused file), `harness_sha256`;
- `eversion_file` (`<engine>.eversion.pt`, bf16 outputs of the E-version subset).

**Reading rules (proposal):**

- A **difference** counts only if `between` differs while `within` is bit-equal; report it against `within.spread`
  otherwise. Characterize it by layer, rows, m, scale, concentration and magnitude.
- **Equality everywhere:** "equal on the tested fixtures". The MoE line stays open or deferred. It is never ruled out
  on this evidence.
- **Unfinished coverage** is INCOMPLETE, not negative.
- **Effects are never combined.** E-plan is within one engine and one slice. E-version is the same slice and plan
  across images, from `compare_eversion.py`. E-partition is not tested.

## E-version

Each image runs:

- its own slice on the full panel;
- the other engine's slice on the E-version subset, after its own slice, and only if at least 240 s of soft time
  remain.

The same slice and the same plan type are therefore planned in both images. The vLLM arguments for projection-mixed
plans are identical by source: the projection-mixed sections of `exl3.py` differ only in the API wrapper.

One difference is part of the version effect, not a mismatch:

- 0.9.1's wrapper passes an explicit `MoeDecodeConfig(w4a16, packed)`.
- 0.7.0's public `Caps` resolves the decode config through its policy. The served 0.7.0 log prints "b12x policy
  fallback: moe.decode is using a heuristic" (:232-237); look for the same line in `v070.log`.

Both results record the Caps for each plan, and `compare_eversion.py` prints them side by side. It matches fixtures
by id and input digest and compares row digests per plan. If both `.eversion.pt` files exist, it adds max abs and rel
L2. It runs CPU-only after the GPU part.

## Running it

```bash
cd <runs>/moe-plan-p3
ATTEMPT_REASON="dry check of gates" DRY_RUN=1 harness/run_p3.sh   # gates, budget, plan; no container
ATTEMPT_REASON="smoke: validate GPU path" SMOKE=1 harness/run_p3.sh
ATTEMPT_REASON="full stage 1" harness/run_p3.sh
```

**Budget gate:**

- **Budget used so far.** `costs.jsonl` is read on every start. Used = Σ `seconds` × `gpus` (`gpu_hours` × 3600 when
  `seconds` is absent).
- **Remaining** = 0.5 GPU-h − used.
- **Wall cap for this attempt** = min(1800 s, remaining). SMOKE uses min(`SMOKE_CAP`, default 420 s, remaining).
- **Refusals:** fewer than 300 s remaining; `ATTEMPT_REASON` unset; a malformed ledger line.
- **Not overridable:** the 0.5 GPU-h budget, the 1800 s ceiling and the 300 s minimum.
- **Attempt number** = count of distinct prior attempts in `costs.jsonl` + 1.
- **Attempt directory.** Each attempt writes only to `out/attempt-N/`, which is created fresh. If it already exists,
  the wrapper refuses and never overwrites it.
- **Aborted attempts.** If an attempt aborts before any container, a zero-second "aborted" cost line keeps the
  numbering in step.

**Gates before any container starts:**

- both images present locally (no pulls);
- the model snapshot present;
- no `glm53*`, `vllm`, `b0cache`, `mgtest` or `p3-*` container running;
- GPU0 memory ≤ 1024 MiB;
- GPU0 utilization ≤ 5%, sampled up to 12 × 5 s.

**Per image:**

- `docker run --rm --network none --gpus device=0 --user $(id -u):$(id -g) -e HOME=/tmp`, with the served env, the
  harness mounted `:ro`, the model repo mounted `:ro`, and the attempt directory as `/o`.
- `B12X_COMPILE_CACHE_DIR=/cc` maps to `out/compile-cache/<engine>/`. This makes a retry cheaper. The cache key
  includes the compile args and the `B12X_*` env, so it cannot change results. The first attempt is cold.

**Timing:**

- **Order:** v091 first, then v070. v070 gets whatever remains of the cap. If fewer than 300 s remain it is skipped,
  and a cost line records the reason.
- **Hard timeout per image** = cap left − 30 s, with `--kill-after=20`.
- **Harness soft deadline** = timeout − 60 s. In full mode, v091's soft deadline is additionally ≤ 55% of the cap,
  so its optional E-version work cannot starve v070's own slice.
- **The harness writes its JSON** after each layer and every ≥ 20 s.

**Cleanup and cost lines:**

- `docker rm -f` runs on every exit path (EXIT/INT/TERM trap).
- One cost line per image goes to `costs.jsonl`, including skipped and interrupted images. It carries `attempt`,
  `attempt_reason`, `mode`, `engine`, `seconds`, `gpus` 1, `gpu_hours`, `image` digest, `b12x_commit`, `rc`,
  `skipped` / `skip_reason`, `wall_cap_total` and `image_timeout`.

**SMOKE** runs v091 only, layer 3, own slice:

- 8 main fixtures: m ∈ {1,16,32,48} × spread/concentrated, x1, uniform, seed 0;
- 1 padded pair;
- the m=32 controls;
- one small-shape check.

It exercises every GPU code path (load, prepare, both plans, the `MAX_M=16` runtime, `apply`, chunked decode, digests)
at full compile cost. Its main output is the measured load, plan-compile and per-fixture time. Use it to check the
full-run estimate before spending the rest of the budget.

## CPU tests (`../tests/RESULTS.txt`; 9 of 9 passed, 2026-10-23)

1. **`check_signatures.py` in both images:** 63 checks each, covering signatures and verbatim served-path lines. 0.7.0
   needed a version-aware call for the informational tile-rule check; the rule itself is identical (K128/N128 at
   10-32 rows, K64/N256 otherwise).
2. **`--dry` in both images:**
   - 1,332 fixture comparisons on fabricated row-local outputs, with plan differences and one within-plan
     nondeterminism injected. The self-check finds exactly the injected rows; controls, dispatch, chunked shapes and
     pad independence are verified.
   - The real weight slices for all 3 layers × 2 slices are loaded on CPU through the image's loader and
     `process_weights_after_loading`, up to `api.prepare_weights`, where CUDA is required. Membership matches the
     served logs and tier tensor shapes are recorded.
3. **SMOKE dry:** 10 fixtures, one layer, one slice.
4. **Soft deadline:** gives `status=partial`, rc 4, with the JSON written.
5. **Wrong image for the engine:** refused, rc 3.
6. **`compare_eversion.py` on the two dry outputs:** 54 common fixtures per slice, 0 input mismatches.
7. **`test_wrapper.sh`:** 28 checks with fake `docker`/`nvidia-smi` on a scratch copy.
   - Refusals: no `ATTEMPT_REASON`; budget exhausted (full and smoke); existing attempt directory; malformed ledger.
   - GPU0 busy or occupied aborts.
   - DRY_RUN prints the remaining budget and starts or writes nothing.
   - SMOKE goes to attempt-1 with cap 420; a second attempt goes to attempt-2; attempt-1 is untouched.
   - Cost-line fields; the cap shrinks to the remaining budget.
   - The second image is skipped and recorded when < 300 s are left; v091's soft share is 990 s of 1800.
   - The E-version compare runs without `--gpus`.
   - The real `docker run` argument vector is checked: `--network none`, `--gpus device=0`, `--user`, `:ro` mounts,
     served env.

## What could not be matched to the served path, or tested on CPU

- **No CUDA graphs.** Served decode at 8 in flight replays FULL graphs captured at padded sizes. The harness runs the
  same plan objects and launches eagerly. Graph capture and replay are not exercised.
- **Not tested: E-partition.** That covers the EP2 replicated-input partial-sum all-reduce and the TP2 rank-sliced
  reduction. The outputs compared are rank-0 partials.
- **Synthetic inputs:**
  - Activations are Gaussian at a reference RMS from norm weights, not captured activations.
  - Routes and weights do not come from the GLM router (sigmoid, bias, renormalize). The weights sum to 1;
    `routed_scaling_factor` is not applied.
  - A difference or an equality here says nothing about real decode inputs.
- **The "runner" padding is a model.** The real padded rows are whatever the runner's persistent buffers hold.
- **Two module globals are patched during `create_weights`,** and the RoutedExperts and model objects are stand-ins
  (see "Reused versus re-implemented"). `apply_vllm_mapper` is not applied, so prefix lookups resolve through
  `exl3.py`'s own name collapsing. The match with the served membership counts shows that the slice and bitrates
  agree. It does not prove that every attribute vLLM sets is set.
- **Routed SwiGLU clamp.** The stock `exl3.py` never forwards `swiglu_limit` to B12x (checked: the string is absent
  from the file), so the stock routed path has no clamp. The harness matches stock.
- **Untested on CPU (GPU-only):**
  - `prepare_weights`, `_rank_sliced_runtime` planning and compile, the kernels, `apply`, GPU memory use;
  - whether the prefill arena equals the logged 262.6 / 334.8 MiB;
  - CUDA error handling: after 3 consecutive CUDA errors the slice is aborted.
- **Compile time is unknown cold.** The served logs show about 20-30 s from model load to "runtime planned", but the
  served containers mounted a persistent `/root/.cache`, so that figure may be warm.

## Estimated GPU wall time (unmeasured; SMOKE will measure it)

| Phase | Estimate |
|---|---|
| Per layer: load (page cache) + prepare | 5-20 s |
| Default runtime compile (33 launches) | 30-180 s, cold |
| `MAX_M=16` runtime compile (v091, 16 launches) | 15-90 s |
| Measurement | 426 fixtures × 3 layers at about 25-40 ms each, about 35-50 s, plus controls and small-shape |
| E-version slice | 3 more layers (30-60 s) + one more runtime compile (30-180 s) + about 5 s |

**Totals:**

| Run | Wall time | GPU-h |
|---|---|---|
| Per image | about 4-11 min (container start and imports about 15 s) | |
| Both images | about 8-21 min | about 0.13-0.36 |
| SMOKE | about 2-6 min | about 0.03-0.1 |

If compile turns out slow, v070's E-version subset is the first thing skipped, then late layers. Partial coverage is
reported as INCOMPLETE.
