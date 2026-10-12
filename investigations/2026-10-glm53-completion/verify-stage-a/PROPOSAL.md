# Bounded verification of D3, D4, D5 (and closing D2/D6): Stage A proposal under PLAN.md amendments 13 / 13a

Date 2026-10-11, revised the same day for the independent review (amendment 13a) **before any data**.
**Status: prepared, not run.** No GPU was used and no server was started. Every GPU step needs operator authorization.

- **Scope is Stage A only** (amendment 13a): one observe-only serving session plus small one-GPU smoke and offline
  steps, at most 2.0 occupied GPU-hours. Stage B (the D4 real-weight comparison) is **not built**; it is only priced,
  at the end. The stock 0.7.0 replication is not part of this.
- The controls, tolerances and classifications below are fixed now, before any data.

- Engine: tpurtell GLM-5.3-Flash 0.9.1, K3.25 EXL3 weights. Release image `ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c0…`.
- Configuration: the recipe `glm53-flash-exl3-k3.25@tpurtell+v0.9.1`, the stack `glm53-flash-v0.9.0`, the shipped graph
  mode (FULL_AND_PIECEWISE, breakable), TP2 with rank-sliced experts (EP off), DCP1, MLA owners split:25, DFlash2 ×3.
- File:line references are to the installed 0.9.1 package (`kdcp/v091/vllm`, `kdcp/v091/b12x`) unless stated.
- **(inferred)** marks statements that come from reading code rather than from a measurement.

## Summary

| item | design (one session plus offline work) | best classification this proposal can reach | GPU |
|---|---|---|---|
| D3 router precision | Capture the gate's bf16 input rows and the **shipped** logits (bf16-result GEMM cast to fp32) in one serving session, with the routing the experts consumed and each row's request. Offline, recompute three labelled computations on the identical inputs: **shipped** (engine `GateLinear`, tier 6), **corrected** (fp32-output GEMM, upstream #54048 form) and **reference** (fp64). Route with the engine's own router. | confirmed omission, exercised, demonstrated computational effect, correction verified offline; completion untested | session + offline |
| D4 routed SwiGLU clamp | Source confirmation is complete (below). In situ, the same session records the layer's `swiglu_limit` (10) next to the `swiglu_limit` actually held by the serving B12x plans (expected: None). B12x's in-tree GPU tests show that the projection-mixed kernel family honours the limit. The session also freezes MoE inputs, routing and per-rank outputs. | confirmed omission and kernel-level reachability. Real-weight engagement and effect are **Stage B**, priced below, not built | session + smoke |
| D5 workspace lock | `VLLM_DEBUG_WORKSPACE=1` plus an observe-only observer. It records every workspace view bound while a CUDA graph is captured, and every resize across graph capture, post-capture warm-up and serving. Each resize is classified by whether it frees an address a captured graph retained. | confirmed omission, then "captured-graph address invalidated", "resize, no captured-graph address invalidated" or "not exercised in this workload" | session |
| D2 / D6 | Existing evidence (tail-fix T/U runs and tests) settles verification. No uncovered condition needs GPU work. | D2: verified correction (T and U); D6: confirmed, made harmless by D2, its own patch is untested | none |

**Cost.** About 0.9 occupied GPU-hours expected, unchanged by the 13a revision. The additions are host-side
bookkeeping: workspace bindings during capture, request attribution of sampled rows, and a few seconds of extra smoke
checks. The hard ceiling is 2.0, enforced by the run script and including an
allowance for an invalid run.

| step | GPUs | expected | wall cap | cap in GPU-hours |
|---|---|---|---|---|
| smoke | 1 | 0.05-0.15 | 900 s | 0.25 |
| server session | 2 | 0.69-0.73 | 1,800 s | 1.00 |
| offline | 1 | 0.08-0.13 | 1,200 s | 0.33 |
| **total** | | **0.82-1.01** | | **1.58** |

The remaining 0.42 GPU-hours is the invalid-run allowance. The run script admits a step only while the logged
GPU-hours plus that step's cap stay at or below 2.0.

## Build (done, CPU only)

**Image.** `local/glm53-v0.9.1-verify313:local`, ID **`sha256:ff6c85422e2582ccea964d0ab234fa795d6b64221c38eb9ad12d97b9f39a23cf`**,
built FROM the release digest.

- Package manifests compared across the whole `vllm/` tree (`build/image-file-diff.txt`):
  - one file was added: `vllm/verify13_capture.py` (sha256 96dcd5e3…);
  - one file changed: `fused_moe/router/gate_linear.py`, release 6fca8c7f… → d892ebaf…;
  - `/opt/b12x` is byte-identical to the release.
- The `gate_linear.py` change appends 9 lines to the end of the file and removes nothing (`build/gate_linear.diff`). The
  added block is a guarded `install(GateLinear)` that runs only when `GLM53_VERIFY13=1`.
- With the variable unset, the only code that runs is `import os` and one environment check. The build gate asserts
  that the module is then never imported and that no class is patched.

**Build gates** (all passed; `build/build.log`):

- sha256 of the release file before it is replaced;
- `py_compile`;
- `cmp` of the installed file against the generated one;
- additions-only diff;
- inert by default;
- `test_verify13.py` **15/15**, `test_d3_offline.py` **5/5** and `analyze_d5.py --selftest`, all on CPU inside the image.

**What the capture module does** (`build/verify13_capture.py`):

- **D3.** Wraps `GateLinear.forward` *after* the original computation. On sampled eager forwards it copies the input
  rows [a, a+R) and the logits rows asynchronously into a pinned host slab owned by the hook. The logits are stored
  under the key `logits_shipped`, labelled "shipped: bf16-result GEMM cast to fp32"; every stored tensor carries a
  label naming its computation (`LABELS`). Each record also stores:
  - x / logits byte alignment and strides;
  - the rows' request attribution from the runner's host-side `InputBatch`: request id, prompt length (which
    identifies the history), prefill or decode, and batch kind (prefill_only / mixed / decode_only). This comes from a
    wrapper on V2 `GPUModelRunner.prepare_inputs` that only reads numpy fields. Once per layer it records
  the dispatch flags (which tier runs), the device capability and a sha256 of the gate weight and bias. The checksum is
  computed by the writer thread on a private side stream.
- **D4.** Wraps `Exl3MoEMethod.apply` *after* the original. For the same (forward, layer) it records `topk_ids`,
  `topk_weights`, the per-rank routed output (before all-reduce) and whether apply's `x` is the gate's `x`; if not, it
  also stores apply's `x`. Per record it also stores the TP rank and the B12x plan branch serving that M: decode
  trellis plan, prefill plan, or prefill plan tiled.
  Once per layer it records the serving path:
  - quant method; projection-mixed, EP and TP geometry; projection-tier membership counts;
  - the cached rank-sliced runtime's key fields and thresholds, plan ids, and `caps / launch_plan /
    _core_workspace_plan .swiglu_limit`;
  - `layer.swiglu_limit`;
  - the installed `exl3.py` sha256 and source-line anchors of `apply`, `_apply_rank_sliced`, `_rank_sliced_runtime`
    and `_prepare_projection_mixed_weights`.
  The run script copies the engine's own EXL3 log lines into `verify13/serving_path_log_lines.txt`.
- **Router.** Wraps `FusedMoERouter.select_experts` and records the router class and its parameters once per router,
  plus `VLLM_USE_FUSED_MOE_GROUPED_TOPK`.
- **D5.** Wraps `WorkspaceManager._ensure_workspace_size`.
  - While a stream is capturing, every workspace view handed out is recorded as a *capture binding*: pointer, bytes,
    runtime mode, batch size and caller (frame walk, host only).
  - Every resize is logged with its phase (graph_capture / post_capture_warmup / serving; pre_capture = profiling),
    the old and new (pointer, size), the caller and the batch size. It is classified against all bindings recorded so
    far: "captured-graph address invalidated" if the freed range holds a bound address, otherwise "resize, no
    captured-graph address invalidated".
  - It also logs `capture_model` and `warmup_kernels` begin and end, with workspace snapshots, and the full binding
    list at `capture_model_end`.

**Sampling** (`select()`; the same pure rule on both ranks):

- Only eager forwards (forward-context mode NONE) are sampled, and only after the run script creates `ARM`.
- Layer `l` is sampled on eager forward `f` when `(l + f) % 21 == 0`, which gives 2 MoE layers per forward, rotating
  through all 42.
- Batches of up to 256 rows are captured whole. Larger batches give a 64-row window; one window in three starts at
  row 0. The runner sorts decode rows first, so mixed batches contribute decode rows.
- Cap: 3 GiB per rank. Pinned pool: 192 slabs of 6.6 MB per rank (1.27 GB of host RAM per rank). The hook makes no
  device allocations, so the KV pool is unchanged.

### Graph safety and observe-only

**By design.**

- FULL and breakable-PIECEWISE replays do not re-run the model's Python (`compilation/breakable_cudagraph.py:1-20`;
  only eager-break ops are replayed as callables). CompilationMode is NONE in this build (server log `compilation_config`).
- The D3/D4 hooks return before touching a tensor when `torch.cuda.is_current_stream_capturing()` is true, and when the
  forward context's `cudagraph_runtime_mode` is not NONE.
- On the hot path, every store goes into hook-owned pinned memory (non_blocking device-to-host copies) and the code
  records one CUDA event.
- The writer thread polls `Event.query()` and never blocks the engine stream.

**Tests run on CPU** (`test_verify13.py` in the image):

- Inert without the variable (subprocess).
- All five wrappers install.
- A TorchDispatchMode op log of the hooked forward:
  - starts with exactly the unhooked op sequence;
  - every extra mutating op writes only into the hook's pool;
  - the output is bitwise equal and the input unchanged.
- The apply path writes only into the pool.
- A forced `capturing=True` records nothing; mode FULL records nothing; no forward context records nothing.
- Meta tensors are skipped without error.
- Records on disk equal their sources bitwise; `same_x` and `x_apply` behave as specified.
- Workspace resizes are classified by phase. A view bound during (forced) capture followed by post-capture growth is
  classified "captured-graph address invalidated"; growth with no binding in the freed range is "resize, no
  captured-graph address invalidated".
- Request attribution of captured rows (mixed batch: decode and prefill rows, prompt lengths) and the D4 plan-branch
  metadata are stored; no unlabelled `logits` key exists.
- AST scans: no `.item/.cpu/.tolist/.numpy/synchronize/nonzero/masked_select/is_pinned/query`, and no boolean-mask
  indexing in hot-path functions; every in-place call targets a hook-owned buffer.

**Host-sync check (planned, smoke step, 1 GPU).** `offline/smoke_gpu.py` runs the armed hooks for M=64 (full) and M=300
(window) under `torch.cuda.set_sync_debug_mode("error")`; any synchronising call inside a hook raises and is counted.
It then captures a CUDA graph of the hooked forward and apply, and checks:

- the hooks skipped (capture counters went up, nothing was sampled);
- replay on new data equals the unhooked eager forward bitwise;
- records equal their sources bitwise;
- the GPU's `GateLinear` reports tier 6;
- D5 observer on a real capture: a workspace view bound inside `torch.cuda.graph` is recorded, and a later growth is
  classified as invalidating it (`d5_capture_binding_and_invalidation`).

**In the session.**

- `gate_capturing` > 0, so the graphs were captured with the hook present and skipped.
- Every record has `mode=NONE`.
- Graph-capture counts (11 PIECEWISE, 9 FULL, 9 DFlash2) and the effective configuration are compared with the
  historical clean 0.9.1 log and with the gate run (`config_compare.py`).
- T/F decode throughput is compared with the gate run on the same fixture (`throughput_tg.py`). This is coarse: the
  gate run carried a heavier checker.

## Stage A smoke results (amendment 13b authorization)

**Smoke attempt 1** (image 23ce3291…, 0.0133 GPU-hours): rc 1. All substantive checks passed except `armed`.

- **Cause:** in the smoke harness, `GateLinear` is built on the CPU and then moved to the GPU. The `__init__` wrapper's
  `_ensure_writer(self.weight.device)` therefore made the CPU the capture device, and the writer allocated a
  **pageable** slab pool, so `pool_pinned` was False.
- **Consequences:**
  - the non_blocking device-to-host copies became synchronous;
  - no events were recorded, though the records were still correct;
  - torch's prototype sync-debug mode did not flag the pageable copies.
- **Fix** (`build/fix-smoke1.diff`; module only, smoke criterion unchanged):
  - only a CUDA device can become the capture device; the first CUDA forward sets it;
  - with CUDA present, the pool is allocated only as pinned memory for that device, otherwise not at all;
  - a non-pinned result raises;
  - the `armed` log line now records `pool_pinned` and `dev`.
- **Serving** (inferred): vLLM builds the model under the target CUDA device, so serving would likely not have hit
  this. The fix makes it fail closed regardless.
- **Regression test:** `capture_device_and_pool_target`; it fails on the old module.

**Smoke attempt 2** (image ff6c8542…, 49 s = 0.0136 GPU-hours): rc 0.

- Every check passed: `tier6_on_this_gpu` (capability 12.0), `armed` (pool_pinned true, cuda:0),
  `no_host_sync_in_hooks`, `graph_capture_skips_hooks`, `graph_replay_equals_eager`,
  `d5_capture_binding_and_invalidation` and `records_bitwise`.
- B12x `swiglu_limit` kernel tests: 3 passed.
- Cumulative Stage A cost so far: 0.0269 GPU-hours.

## Stage A offline: control (c) diagnosis (attempt 3; no D3 effect computed)

**What failed.** Offline attempt 2 ran controls (a) and (b), which passed: (b) reproduced the shipped logits bitwise on
1,593 of 1,593 records. Control (c) failed on 1,425 of 1,593 records, without the diagnostic it should have produced.

**What changed.** `d3_router_compare.py` now:

- classifies control (c) per record (`d3_control_c_records.jsonl`);
- diagnoses up to 60 failing records (`diagnose_routing`) for repeatability, batch shape, padding content, alignment,
  row shift ±1, order and scaling.

The criterion is unchanged. Image `sha256:3254a415…`; the vllm tree is identical to the serving image ff6c8542… (only
`/opt` scripts differ). Offline attempt 3: 24 s = 0.0067 GPU-hours, d3 rc 3 (stopped in phase 1 as specified), d5 rc 0.

**Findings** (per-category counts over all 103,608 rows):

- **(i) Id sets differ:** 0 rows. **(iii) Row misalignment:** none; captured rows match engine rows exactly at shift 0
  and at no row at shift ±1.
- **(iv) Transformation:** none. The id order is identical, and the captured/engine weight-sum ratio is 1 ± 2e-7. The
  served path is the fused `ops.grouped_topk` (`router_info`: `GroupedTopKRouter`, `VLLM_USE_FUSED_MOE_GROUPED_TOPK`
  true, no EPLB, int64 ids), the same call the offline script makes.
- **(ii) Same sets, weights not bitwise:** 91,200 rows, all in the 1,425 failing records. In the 60 diagnosed records:
  up to 139 ulp, max relative 8.7e-6. Over all failing records: max relative 1.1e-3.
- **(v) Repeatability and batch shape:**
  - The engine router is repeatable: window and serving-M runs are identical on repeat, 60/60.
  - With the **serving row count M** (captured rows at [a, a+R)), results equal the capture **bitwise** on 60/60
    sampled failing records. They are independent of the padding content (zero, random or tiled) and of alignment.
  - All 168 passing records have M ≤ 1,024, and all 1,425 failing records have M ≥ 1,139. The batch-size sweep is
    exact only from M upward: e.g. a record with M = 1,244 is exact at 1,244 and 2,048, not at ≤ 1,024.

**Cause.** The served fused grouped_topk computes the routing **weights** (not the ids, on these rows) differently
depending on the batch row count. It switches somewhere between 1,025 and 1,139 rows; the exact boundary is not pinned
and the kernel source is not in the image **(inferred: two kernel variants or launch configurations)**. The offline
control invoked the router on the R-row window, not on the M-row serving batch.

- Against fp64 (unbiased sigmoid, renormalised × 2.5), the captured weights from batches above 1,024 rows are about
  10× less accurate: median relative error 1.4e-6, p99 3.9e-6, max 1.1e-3. Captured weights from batches of 1,024 rows
  or fewer: median 1.2e-7, p99 3.3e-7, max 7.5e-7.
- This is a property of the served router: batch-size-dependent weights. It is deterministic, and it is reported here
  rather than corrected.

**Proposal, NOT applied (needs operator/assessor sign-off).**

- Invoke the router in control (c) **and** in phase 2 (corrected routing) as in serving: one call on the M-row batch,
  captured rows at [a, a+R), zero padding, server alignment. This is exactly the treatment control (b) already uses for
  the GEMM.
- The criterion stays bitwise. Expected: (c) passes on every record (verified on the 60 sampled failing records; the
  168 passing records are in the ≤ 1,024 regime) **(inferred for the rest until run)**.
- Shipped and corrected routing then go through the same kernel variant, so the D3 comparison is not confounded by the
  batch-size switch.
- The batch-size dependence is documented as a separate engine finding (ids unaffected on these rows; weight
  difference up to 1.1e-3 relative), not folded into D3.
- Prepared and CPU-tested (5/5): `offline/d3_router_compare.proposed-servingM.py`,
  `offline/d3-proposed-servingM-routing.UNAPPLIED.diff`. It is **not** installed in the image.
- Cost if approved: one offline run, about 0.007 GPU-hours.

Stage A total in costs.jsonl after attempt 3: 0.6823 GPU-hours: smoke 0.0133 + 0.0136, server 0.6417, offline 0.0003 + 0.0067 + 0.0067.

## Stage A results: offline attempt 4 (amendment 14, serving-shaped router invocation)

**Run.** Image `sha256:648c2764549368ec6a0b5d89c0c8e56113fa37c5ef84e4f9871e100265bc7d52`; its vllm tree is identical
to the serving image ff6c8542… (`build/image-file-diff.txt`). The script diff is `offline/d3-amendment14-servingM.diff`.
CPU tests: 15/15 and 5/5. 38 s = 0.0106 GPU-hours, rc 0. Stage A total in `costs.jsonl`: **0.6929 of 2.0 GPU-hours**.

**Router invocation (amendment 14).** Control (c) and the corrected routing in phase 2 call the engine's router once
on the M-row batch:

- the captured rows sit at [a, a+R), the other rows are zero, and the buffer has the server's alignment;
- the router configuration is the server's (`router_info`);
- the bitwise criterion and the stop rule are unchanged.

**Normalisation and the ×2.5 routed_scaling_factor, like-for-like** (also written into `d3_summary.json`,
`normalisation`).

- **Shipped and corrected routing** both come from the same engine call: `GroupedTopKRouter._compute_routing`
  (`fused_moe/router/grouped_topk_router.py:337-349`) → `grouped_topk` (:80-108, with
  `VLLM_USE_FUSED_MOE_GROUPED_TOPK`=1, `envs.py:1625`) → `fused_grouped_topk` (:28-71) → `ops.grouped_topk(logits,
  1, 1, 8, renormalize=True, routed_scaling_factor=2.5, e_score_correction_bias, scoring_func=1)` (:43-52;
  `_custom_ops.py:2470-2500`).
  - Sigmoid, bias for selection only, renormalisation and the ×2.5 all happen inside the fused kernel.
  - Nothing is applied afterwards: the `*= routed_scaling_factor` at :314-315 belongs to the invalid-grouping fallback,
    which is not taken.
  - There is no output-side scale (`apply_routed_scale_to_output=False`, `model.py:170, 252`).
  - Measured: every captured row's weights sum to 2.5.
- **The fp64 reference** (used for selection and for E1 only): sigmoid + bias selects the top-8; the weights are the
  unbiased sigmoids, renormalised and ×2.5.
- Whether the fused kernel renormalises then scales, or the reverse, is not visible.

### Controls (all pass on all 1,593 records, 103,608 rows)

| control | result |
|---|---|
| (a) dispatch, bf16-valued shipped logits, weight/bias sha256 (both ranks) | pass |
| (b) offline shipped == captured shipped logits, bitwise | 1,593 / 1,593 records |
| (c) engine routing at serving M == consumed ids/weights, bitwise | **1,593 / 1,593 records**; 103,608 rows exact (0 set differences, 0 weight differences) |
| (d) corrected vs fp64 within τ_d = 2⁻¹⁴·S + 2⁻²⁴·\|ref\| | pass: 0 of 29,839,104 logits beyond τ_d (and 0 beyond τ_worst); max \|corrected − ref\| / S = 7.6e-7 (τ_d/S ≈ 6.1e-5); 0 corrected-vs-fp64 top-8 disagreements; ran as `torch.mm(out_dtype=fp32)` |
| shipped vs fp64 (contrast) | 21,553,853 of 29,839,104 logits (72%) beyond τ_d; max \|shipped − ref\| / S = 1.7e-3; 3,262 top-8 disagreements with fp64 |

### D3 effect (shipped "bf16-result GEMM cast to fp32" vs corrected "fp32-output GEMM, #54048 form")

- **Membership flips:** 3,262 of 103,608 token × layer rows, i.e. **3.15%**. 3,260 rows swap one expert and 2 rows
  swap two. Every flip coincides with a shipped-vs-fp64 disagreement, and corrected agrees with fp64 everywhere.
- **Uncertainty** (95%, percentile bootstrap over clusters, 2,000 replicates):

  | clusters | n | flip rate CI | mean routing-L1 CI |
  |---|---|---|---|
  | histories (prompt lengths 4,240 … 120,217 tokens) | 8 | 2.90-3.50% | 0.0132-0.0159 |
  | requests | 176 | 3.01-3.29% | 0.0138-0.0150 |

  With only 8 histories, the history interval is coarse.
- **Routing-weight changes:** every row's weights change (L1 > 0 on 103,608 rows).
  - Median L1 between dense routing vectors: about 0.0018.
  - On rows whose expert set is unchanged, the per-row max |Δw| has p99 about 0.0016 (weights sum to 2.5).
- **By execution type** (`d3_by_type.csv`):

  | batch kind | row kind | rows | flips | rate | i.i.d. Wilson 95% (descriptive) |
  |---|---|---|---|---|---|
  | prefill_only | prefill | 37,356 | 1,153 | 3.09% | 2.92-3.27% |
  | mixed | prefill | 50,456 | 1,630 | 3.23% | 3.08-3.39% |
  | mixed | decode | 15,796 | 479 | 3.03% | 2.78-3.31% |

  Decode-only eager batches did not occur.
- **By layer** (`d3_by_layer.csv`; about 2,368-4,869 rows per layer): 1.22% (layer 21) to 6.63% (layer 3). Layers 3
  and 7-9 are highest (5.4-6.6%), and layers 20-23 and 43-44 lowest (1.2-2.2%). `d3_by_layer_type.csv` crosses layer
  with execution type.
- **Near-boundary margins** (corrected 8th−9th biased-score gap): median 0.0023-0.0103 by layer, about 0.0056-0.0059
  by execution type. On flipped rows the median is 0.00014-0.00034, which is within the bf16-rounding perturbation.

### Finding E1 (engine, separate from D3)

The served fused grouped_topk computes routing **weights** differently by batch row count. The switch lies between
1,025 and 1,139 rows; the ids are unaffected on the captured rows. Served weights compared with the fp64 re-statement:

| batch rows | rows | median relative error | p99 | max |
|---|---|---|---|---|
| M ≤ 1,024 | 12,408 | 1.2e-7 | 3.3e-7 | 7.5e-7 |
| M > 1,024 | 91,200 | 1.4e-6 | 3.9e-6 | 1.1e-3 |

No workaround was applied beyond invoking at the serving M. Completion effect: untested.

### Coverage and limits

- FULL-graph decode steps are covered by **source inspection only (not measured)**. Decode rows were measured only in
  mixed eager batches.
- The fixture is the col511-gate workload over 8 long documents, not a GPQA distribution.
- **D3 classification:**
  - confirmed omission, exercised (eager prefill and mixed batches), demonstrated computational effect, and verified
    correction (offline): all yes;
  - **completion effect: untested**; flips and weight changes do not establish a completion benefit or explain the gap.

### D5 and D4 in the same run

- **D5:** "not exercised in this workload". 10 capture bindings per rank. The only resizes (r0: 1, r1: 2) were during
  profiling, before any graph existed.
- **D4:** the frozen inputs are sufficient for Stage B: 1,593 records per rank, all paired, nothing missing. Plan
  branches: 2,850 "prefill_plan tiled by 1024" and 336 "prefill_plan".

## D3 router precision

### Source confirmation

1. **The model builds the router at `models/glm5next/nvidia/model.py:192-198`.**
   - The call is `GateLinear(hidden, n_routed_experts, out_dtype=self.router_dtype, prefix=…)`, with no
     `force_fp32_compute` and no `params_dtype`.
   - `_get_moe_router_dtype` (`model_executor/models/deepseek_v2.py:123-133`) returns fp32 because the checkpoint has
     `moe_router_dtype: "float32"`.
   - The weight takes the model dtype, bf16: in the checkpoint `…layers.N.mlp.gate.weight` is BF16 [288, 4096] and
     `e_score_correction_bias` is F32 [288] (shard 9 header).
   - Server placement ledger: "router" 0.092 GiB, which is the bf16 size for 42 × 288 × 4096 elements.
2. **Dispatch on SM120 (`fused_moe/router/gate_linear.py`).**
   - `can_use_specialized_kernels = is_cuda and (is_hopper or is_blackwell) and not bias` (:59-63). Hopper is exactly
     (9,0); "blackwell" is `is_device_capability_family(100)`, which compares `cap//10` (`platforms/interface.py:481-493`):
     12 ≠ 10, so the result is False on SM120 and SM121.
   - `force_fp32_compute` defaults to False and is not passed, so the weight stays bf16 (:67-68).
   - Tier 1 ll_bf16 needs `can_use_specialized_kernels` (:135-145, :168-177). Tier 2 dsv3 likewise (:81-88).
   - Tiers 3 and 4 need an fp32 weight (:99-113).
   - Tier 5 (cuBLAS `torch.mm(…, out_dtype=fp32)`, :221-224) needs `allow_specialized_router_gemm` or ROCm (:123-130, :157-165).
   - **Tier 6 therefore runs** (:226-232). `x` is already bf16, so `ReplicatedLinear.forward` → `UnquantizedLinearMethod.apply`
     (`layers/linear.py:379-385`, :204-212) → `default_unquantized_gemm` = `F.linear` (`layers/utils.py:86-92`) produces
     bf16 logits, and then `.to(torch.float32)`.
   - The start-up line "Warming up ll_bf16 router GEMM kernels" does not contradict this:
     `model_executor/warmup/kernel_warmup.py:57-99` warms by weight dtype on any capability ≥ 9.0 (:137-138); it does
     not consult `allow_ll_bf16_gemm`.
3. **Upstream's correction.** vllm-project/vllm#54048 (b5707bf99, "Enable cuBLAS out_dtype router GEMM on all CUDA
   archs (fixes family-120/GB10)") makes tier 5 eligible on all CUDA devices, so SM120 then computes
   `torch.mm(x, W.T, out_dtype=torch.float32)`, the expression at :223. The other agent's untested port is
   `defect-repros/D3-router-logits-bf16/proposed-UNTESTED.diff`; it is not used here.
4. **Routing path.**
   - `FusedMoEFactory(use_grouped_topk=True, n_group 1, topk_group 1, scoring "sigmoid", routed_scaling_factor 2.5,
     e_score_correction_bias fp32, renormalize = norm_topk_prob True)` (`model.py:199-204, 237-260`).
   - `router_factory.py:141-188`: the grouping is degenerate, but the routing method is not preserved (`config.py:148-156`
     gives DeepSeekV3 for n_group 1 and Unspecified for None with scale 2.5), so the factory returns `GroupedTopKRouter`.
   - `grouped_topk_router.py:337-347` → `grouped_topk` (:80-108). With `VLLM_USE_FUSED_MOE_GROUPED_TOPK` at its default
     of 1 (`envs.py:1625`) this is the fused `ops.grouped_topk(…, scoring_func=1)` (:41-52): sigmoid, + bias for
     selection, top-8, unbiased weights renormalised, × 2.5.
   - `moe_runner.py:596-607` → `routed_experts.py:1224` → `Exl3MoEMethod.apply` (`exl3.py:3349`), with
     `topk_indices_dtype` = `torch.long` (`exl3.py:2680-2681`).

### Labels (used everywhere: captures, scripts, outputs)

| label | computation | where |
|---|---|---|
| **shipped** | "shipped: bf16-result GEMM cast to fp32": `F.linear` on bf16 x and bf16 W with a bf16 result, then `.to(fp32)` (tier 6) | captured as `logits_shipped`; recomputed offline by the engine's `GateLinear` |
| **corrected** | "corrected: fp32-output GEMM (upstream #54048 form)": `torch.mm(x, W.T, out_dtype=torch.float32)` on the same bf16 x and W (the `gate_linear.py:223` tier-5 expression) | offline only |
| **reference** | "reference: fp64": `x.double() @ W.double().T` | offline only |

No output uses an unlabelled "fp32 logits". If the offline build rejects `out_dtype`, the run records a fallback (fp32
GEMM of the same bf16 values, TF32 off), and "verified correction" is withheld.

### Captured, per sampled (eager forward, MoE layer), on both ranks

- `x` rows [a, a+R) (bf16) and `logits_shipped` rows;
- `topk_ids` (int64) and `topk_weights` (fp32) as consumed by `Exl3MoEMethod.apply`;
- metadata: M, a, R, forward index, runtime mode (always NONE), byte alignment and strides, and per row the request
  id, prompt length, prefill/decode and batch kind.
- Once: gate dispatch flags, weight/bias sha256 per layer (both ranks), router class and parameters.

### Offline comparison (`offline/d3_router_compare.py`, 1 GPU, inside the verify13 image with the hooks inert)

**Phase 1, controls (a)-(c)**, over every record before any effect is computed:

- Per layer, the checkpoint weight and bias are loaded into the engine's own `GateLinear(4096, 288, out_dtype=fp32)`
  (tier 6 on SM120) and `GroupedTopKRouter`. The router parameters come from the in-server `router_info`; on the GPU
  this is the same fused `ops.grouped_topk` kernel.
- **shipped** is recomputed on an M-row input: the captured rows go at [a, a+R) and the other rows are zero. The
  buffer is placed at the in-server byte alignment (mod 256), so GEMM shape and alignment match serving.

**Phase 2, effect and control (d)**, only if (a)-(c) all pass:

- **corrected** and **reference** are computed on the same inputs.
- **shipped** routing = the consumed ids/weights, which control (c) shows equal the engine router on the shipped
  logits. **corrected** routing = the engine router on the corrected logits.
- **reference** selection is a plain fp64 re-statement: sigmoid + bias top-8, unbiased weights renormalised × 2.5.

### D3 controls and tolerances (fixed before data)

| control | criterion | on failure |
|---|---|---|
| (a) dispatch and identity | Every layer's in-server `gate_info`, on both ranks, has all six `allow_*` flags False, so tier 6. The offline `GateLinear` on this GPU reports the same, with a bf16 weight. Every captured `logits_shipped` value is bf16-representable (the tier-6 signature). Gate weight and bias sha256 equal the in-server sha256 on both ranks. | **stop** (exit 3), `d3_controls_failed.json` |
| (b) exact reproduction | Offline **shipped** == captured `logits_shipped`, **bitwise** (int32 view), on every captured row. No tolerance. | **stop** and emit a diagnostic per failing record (up to 20); see below |
| (c) exact routing | The engine router on the captured `logits_shipped` reproduces the consumed `topk_ids` (as sets) and `topk_weights` (bitwise, dense by expert) on every row | **stop**, diagnostic |
| (d) corrected approaches reference | Let S = Σₖ \|xₖ·wₖ\| (fp64), K = 4096. Every **corrected** logit satisfies \|corrected − reference\| ≤ **τ_d = 2⁻¹⁴·S + 2⁻²⁴·\|reference\|**. Every corrected-vs-reference top-8 disagreement lies inside the margin band: fp64 8th−9th biased-score gap ≤ 0.5·maxₑ τ_d,ₑ. The corrected path ran as `torch.mm(out_dtype=fp32)`, not the fallback. | no stop. Within the hard bound τ_worst = K·2⁻²³·S + 2⁻²⁴·\|reference\| but not τ_d: "inconclusive". Beyond τ_worst: "wrong computation". Either way "verified correction" is withheld. |

**Diagnostic for (b)** (`diagnose()`). For each failing record it reports:

- rows not exact, maximum absolute difference and maximum ulp difference;
- **repeatability**: three offline recomputations equal each other;
- **batching / row independence**: random instead of zero padding rows gives the same window result;
- **alignment**: the result at alignment 0 differs from the in-server alignment, and whether alignment 0 matches the
  capture;
- **shape**: the window-only GEMM (M' = R) matches the capture;
- dispatch flags, runtime mode, batch kind, x strides, and whether the capture is bf16-valued.

The tolerance is not relaxed after data. A failure stops D3 at "confirmed omission / exercised" until the diagnostic
explains it.

**Why these tolerances** (also written into the script, `TAU_NOTE`):

- Products of bf16 values are exact in fp32: 8 + 8 significand bits fit in 24.
- **τ_worst** (hard bound, any summation algorithm): K − 1 fp32 additions, each off by at most 1 ulp even if the
  accumulator truncates, give |error| ≤ K·2⁻²³·S to first order (Higham γ_{K−1}), plus the final rounding.
- **τ_d** (verification tolerance): a tensor-core GEMM with fp32 accumulators updates the accumulator once per
  16-product MMA chunk, i.e. K/16 = 256 times. That bounds the error by 256·2⁻²³·S = 2⁻¹⁵·S. τ_d doubles this to
  allow for error inside each chunk and any split-K reduction, giving 2⁻¹⁴·S, 8× tighter than τ_worst.
- **Band**: sigmoid is ¼-Lipschitz and the bias is common to both paths, so a logit error ≤ τ moves a biased score by
  at most τ/4, and a top-8 boundary needs a gap ≤ 2·τ/4.
- **shipped** is reported against the same τ_d for contrast. Its bf16 output rounding (up to 2⁻⁹·\|logit\|) is
  expected to exceed τ_d where logits are large relative to S. In the CPU synthetic test: corrected max error/S = 2.0e-8,
  shipped 3.1e-4, τ_d = 6.1e-5.

### Reporting (`d3_summary.json`, `d3_by_layer.csv`, `d3_by_type.csv`, `d3_by_layer_type.csv`)

**Rows of interest:**

- a **membership flip** is a token × layer whose top-8 set differs between **shipped** and **corrected** routing;
- a **routing-weight change** is the L1 distance between dense 288-vectors, plus, for rows with the same set, the
  maximum |Δw|.

**Breakdowns:**

- by layer;
- by execution type = (batch kind: prefill_only / mixed / decode_only) × (row kind: prefill / decode / padding /
  unknown);
- by layer × execution type.

**For each cell:**

- descriptive row counts and the flip count;
- the L1 median and p99, and the same-set max-|Δw| p99;
- the near-boundary margin: median corrected 8th−9th gap, over all rows and over flipped rows;
- a histogram of swapped experts;
- shipped-vs-reference and corrected-vs-reference selection disagreements;
- control (d) statistics.

**Uncertainty:**

- Rows are **not** independent serving draws. Rows share requests and histories: prompts are prefix-cached, and
  phases repeat over 8 histories.
- Intervals are therefore percentile **bootstraps over clusters**, 2,000 replicates, seed 20261012:
  - primary: clusters = histories (prompt length, 8 clusters, plus "unknown");
  - secondary: clusters = requests (vLLM request id).
- With only 8 histories, the history interval is coarse and is reported as such.
- Per-cell Wilson intervals are labelled "rows treated as i.i.d.", for description only.

**Coverage statement** (written into the summary): FULL-graph decode steps are covered by **source inspection only,
not measured**. Sampling runs on eager forwards. Measuring inside FULL graphs would add nodes to the captured graphs,
i.e. modify execution. Decode rows are measured only where they occur in mixed eager batches.

### Classification (fixed before data)

- **Confirmed implementation omission:** the source (above) plus control (a).
- **Exercised in the tested workload (measured):** control (a) on every captured layer-step of eager forwards, i.e.
  prefill and mixed batches. FULL-graph decode: source only.
- **Demonstrated computational effect:** (a)-(c) pass, and **corrected** routing differs from **shipped** routing in at
  least one token × layer (set or weights). Reported with the cluster-bootstrap intervals.
- **Verified correction (offline):** (a)-(c) and (d) pass. The patched engine is not run in serving.
- **Completion effect:** untested. Flips and weight changes do not establish a completion benefit or explain the gap
  (amendment 13a).
- **Scope:**
  - The fixture is the col511-gate workload over 8 long documents. It is not a GPQA distribution.
  - Rates are fixture-specific.

## D4 routed-expert SwiGLU clamp

### Source confirmation (complete)

1. **The model provides the limit.**
   - `swiglu_limit = config.swiglu_limit` = 10.0 (`model.py:220`).
   - It reaches the shared experts (`Glm5NextMLP` → `SiluAndMulWithClamp`, `model.py:150-152, 226-235`) and the dense
     MLP (:365-372).
   - It reaches the routed experts' `FusedMoEFactory(…, swiglu_limit=…)` (:259); `RoutedExperts` stores it
     (`fused_moe/routed_experts.py:113`).
2. **The K3.25 path is projection-mixed rank-sliced EXL3 via B12x, not the expert-mixed path.**
   - Server log: "EXL3 standard fused MoE enabled for glm5_next: projection tiers (3, 4) via b12x" (`exl3.py:727`), then
     "EXL3 projection-mixed Trellis …layers.N.mlp.experts: tiers=(bits,gate,up,down)((3,248,221,180),(4,40,67,108))"
     (`exl3.py:2536`), and "EXL3 rank-sliced runtime planned: Trellis m=1..32 block_m=8, prefill trellis block_m=64
     arena=262.6MiB capacity=1024" (`exl3.py:3118`).
   - `_apply_rank_sliced` sends projection-mixed layers to `_rank_sliced_runtime`, not to `_apply_mixed_rank_sliced`
     (`exl3.py:3164-3181`).
   - That runtime's plans come from `_plan_with_scratch` → `api.Caps(max_tokens, num_topk, route_num_experts, device,
     weight_plan, quant_mode="w4a16", w4a16_block_size_m, full_rotation_output_dtype, mixed_trellis caps)`
     (`exl3.py:2939-2970`), which has **no `swiglu_limit`**. `exl3.py` contains no occurrence of "swiglu".
   - The 2026-10-08 A/B patched `compile_mixed_trellis` at `exl3.py:2737`, the expert-mixed runtime that K3.25 never
     runs, which explains its no-op result (`upstream-diff/notes-main.md`).
3. **The B12x plumbing would carry the limit if given.**
   - `api.Caps` → `_impl.TPMoEScratchCaps(swiglu_limit=…)` (`exl3.py:213-273`; `b12x/moe/fused_moe/_impl.py:840-901`,
     normalised by `normalize_swiglu_limit_for_activation`, which accepts any positive finite limit for "silu",
     `moe/_shared/kernels/activations.py:38-55`). Limit 1 is therefore legal for a smoke test.
   - → core plan `swiglu_limit=caps.swiglu_limit` (`_impl.py:9116, 9149`)
     → `_plan_projection_mixed_trellis_launches` (`_impl.py:8568-8700`). The EXL3 source format gives the direct two-tier
     path (:8613-8616) → `compile_mixed_trellis(…, swiglu_limit=core_plan.swiglu_limit)` (:8657, :8672).
   - → `moe/_shared/kernels/w4a16/mixed_trellis.py:1883-1960`: `W4A16FusedMoeKernel(swiglu_limit=…)` (:1951) with
     `w13_layout="trellis_t256_proj"`, `intermediate_rotation=True`, `full_rotation=True`.
   - (`mixed_trellis4.py:914` is the four-tier variant, which this path does not use.)
4. **The projection kernel honours the limit, so the patch is not a no-op.**
   - `has_swiglu_limit` is set at `w4a16/kernel.py:6281-6282`, and `_clamp_swiglu_inputs` is defined at :6546-6559.
   - Every activation epilogue reached by `_moe_body` (:7009; dispatch :7153 / :7169 / :7185) applies it:
     - `_run_activation` with rotation, after un-rotating gate/up with `had128 × svh` (:8103-8166; comment :8160-8162);
     - `_run_activation_compact` (:7833-7836);
     - `_run_activation_coupled` (:7669-7672);
     - the plain arm (:8198).
   - The mixed kernels run the driver's `_moe_body` (`mixed_trellis.py:1001, 1718`).
   - **Inferred:** a correct `Caps` patch reaches the computation. Not yet measured with real weights.
5. **Semantics match the model's intent.**
   - `_clamp_swiglu_inputs`: gate ← min(gate, L); up ← clamp(up, −L, L), applied to the logical (un-rotated) values,
     before silu(gate)·up.
   - This equals `SiluAndMulWithClamp` (`model_executor/layers/activation.py:201-240`; gate max-only, up ±L) and the
     upstream Glm5Next routed/shared clamp (upstream main `models/glm5next/common/model.py:285, 314`, per
     `upstream-diff/notes-main.md`; not re-read for this proposal).
   - The other agent's untested patch passes `getattr(layer, "swiglu_limit", None)` at both `exl3.py` call sites
     (`defect-repros/D4-routed-swiglu-clamp/proposed-UNTESTED.diff`). This proposal is consistent with it and does not
     use it.

### What Stage A establishes for D4 (omission and reachability only; amendment 13a)

- **In situ, same session, no extra cost** (`experts_info`):
  - `layer.swiglu_limit` together with `caps.swiglu_limit`, `launch_plan.swiglu_limit` and
    `_core_workspace_plan.swiglu_limit` of the serving runtime plans. Expected: 10.0 vs None/None/None, i.e. the
    omission in the running engine, not only in source.
  - The serving path that produced the frozen inputs (below).
- **Kernel-level reachability (smoke step, 1 GPU).** B12x's own tests
  `tests/moe/test_w4a16_mixed_trellis.py::test_mixed_trellis_swiglu_limit_matches_clipped_serial[False|True]` (:1966-1972)
  and the three-tier variant, run in the release B12x.
  - With synthetic weights at geometry 5120×640, the two-tier projection kernel with `swiglu_limit=10` must equal the
    clipped serial reference and must differ from the unclipped result by more than 1% relative (:503-510).
  - This shows that the kernel family honours the limit. It is **not** the real-weight effect.

**Frozen inputs for Stage B** (`d4_inventory.json`, `sufficient_for_stage_b`). Per sampled (forward, layer), on both
ranks:

- `x` = the routed experts' input as consumed by `Exl3MoEMethod.apply`. This is the post-attention-norm MoE input. It
  is stored once when apply's `x` is the gate's tensor (`same_x`), otherwise additionally as `x_apply`.
- `topk_ids` / `topk_weights` as consumed.
- `out` = that rank's routed-expert partial, before the all-reduce: the control target for an unclamped offline replay.
- M and the window [a, a+R), the TP rank, and the plan branch.

**Serving path** (recorded once per layer and rank, plus the log lines):

- quant method `Exl3MoEMethod`; projection-mixed True; EP off; TP size and intermediate size per partition; tier
  membership counts;
- runtime thresholds, Trellis m window, prefill capacity, chunk;
- plan ids, and the swiglu_limit in each plan;
- `exl3.py` sha256 and the source anchors of `_apply_rank_sliced` / `_rank_sliced_runtime` /
  `_prepare_projection_mixed_weights`;
- `serving_path_log_lines.txt`: the engine's "EXL3 standard fused MoE … projection tiers (3, 4)" (`exl3.py:727`),
  "EXL3 projection-mixed Trellis …" (`:2536`) and "EXL3 rank-sliced runtime planned …" (`:3118`) lines.

The inventory checks that rank 0 and rank 1 records pair up and that no tensor is missing.

**D4 classification from Stage A:**

- confirmed omission: source, plus in situ;
- kernel-level reachability: smoke;
- real-weight engagement, computational effect and verified correction: **Stage B, not approved**, priced below;
- completion effect: untested.

## Stage B (not approved, not built): pricing of the smallest D4 comparison

**Objective.** On the Stage A frozen inputs, run the routed-expert computation unclamped vs `swiglu_limit` 10, with
limit 1 as a reachability control. Limits 7 and 5 are deferred.

**Existing facilities that suffice; no new kernel or harness design.** All of the following already exist:

- the engine's own EXL3 code:
  - `Exl3Config.from_config` + `maybe_update_config`, which hydrate the per-projection bit map from the checkpoint's
    quantization config (`exl3.py:551-560, 576-600, 643-730`);
  - `Exl3MoEParameter.load_exl3_weight`, which does the engine's own TP slicing (`exl3.py:1710-1769`);
  - `Exl3MoEMethod._prepare_projection_mixed_weights` (:2351-2541), `_rank_sliced_runtime` (:2824-3134) and
    `_apply_rank_sliced` (:3164-3245);
- B12x's planner (`Caps` / `plan` / `bind` / `run`) with its `swiglu_limit` field.

B12x's in-tree harness (`_prepared`, `_serial_tier`) builds synthetic weights and cannot load EXL3 checkpoints, so it
is not the route. What remains is glue:

- a stand-in layer namespace of about 20 attributes;
- per-layer loading of the 288 × 3 projections for each TP rank;
- a wrapper on `api.Caps` that adds `swiglu_limit`, with the runtime cache cleared between arms, because the cache key
  does not contain the limit (`exl3.py:2904-2926`);
- the replay loop, using the captured M, padding rows with route id −1 **(inferred from B12x tests)**, and the plan
  branch recorded per record.

**Controls.**

- The unclamped offline output per rank equals the captured `out` **bitwise**. If not, stop and diagnose, as for D3 (b).
- Limit 1 must change essentially every row, which shows the limit reaches the computation.

**Outputs.** Per layer and limit: rows whose output changes, relative L2 and max |Δ| of the per-rank and summed outputs,
and the share of rows that change at limit 10. This is row-level engagement only. Element-level engagement and an EXL3
decoder stay deferred (amendment 13a).

**Engineering hours (separate from GPU):**

| task | hours |
|---|---|
| glue: config hydration, per-layer weight load and TP slicing, stand-in layer, `Caps` limit wrapper, runtime-cache handling; CPU tests up to the CUDA boundary (config, slicing, tier stacking) | 4-5 |
| replay and controls: same-M padding, plan branch, bitwise control against `out`, metrics, report | 2-3 |
| first-GPU-contact debugging (wall-clock; the GPU part is costed below) | 2-3 |
| write-up | 1 |
| **total** | **9-12 h** |

**GPU-hours (one GPU, separate allowance; not part of Stage A's 2.0):**

| item | GPU-hours | basis |
|---|---|---|
| measurement | 0.15-0.25 | 6-8 layers × 3 arms (∞ / 10 / 1). Per arm, a plan and JIT compile with the limit as a compile-time constant; the gate run's server log shows about 74 s for the first runtime plan (18:56:02 → 18:57:16). Weight load is about 3 GB per layer per rank. The replay of the recorded inputs takes minutes. |
| first-contact debugging | 0.2-0.4 | |
| one invalid attempt | 0.15 | |
| **total, proposed cap** | **0.5-0.8 (cap 1.0)** | |

**What its output would change.** Which concrete next experiment D4 belongs to.

- **Limit 10 changes no row's output (bitwise) while limit 1 changes nearly all.** D4 is closed as "omission confirmed,
  computationally inert on the tested inputs". It leaves the candidate list for a corrected 0.9.1 arm, and no
  element-level or decoder work follows.
- **Limit 10 changes a non-trivial share of rows beyond bf16 noise.** D4 becomes a candidate for the narrowly defined
  corrected 0.9.1 arm, singly or explicitly bundled with D3, with the loss of attribution stated. Element-level
  engagement becomes worth pricing.

Without Stage B, the next large experiment must treat D4's effect as unknown. Stage B needs Stage A's frozen inputs
first.

## D5 workspace locking

### Source confirmation

- `v1/worker/gpu_worker.py` runs `kernel_warmup` (:715), then `capture_model()` (:719), then, for V2,
  `warmup_kernels(model_runner, execute_model, sample_tokens)` *after* capture (:796-797).
- `lock_workspace()` is called only in V1 (`v1/worker/gpu_model_runner.py:7180`). V2's
  `v1/worker/gpu/model_runner.py:977-1035` `capture_model` has no lock.
- `WorkspaceManager._ensure_workspace_size` (`v1/worker/workspace.py:119-191`):
  - when locked, growth raises (:156-162);
  - when unlocked, growth drops the old tensor, calls `empty_cache()` and allocates a new one (:164-178);
  - growth is logged only with `VLLM_DEBUG_WORKSPACE=1` (:181-189), which is why no historical log shows resizes.
- Users on this model's path:
  - the B12X sparse MLA `_borrow_workspaces` (`v1/attention/backends/mla/b12x_mla_sparse.py:2155-2158`);
  - the kpool indexer K-gather (`model_executor/layers/sparse_attn_indexer_kpool.py:296-300` reservation during
    profiling, :508-516 per call, sized by gathered tokens);
  - `sparse_attn_indexer.py:1114`.
  - Long contexts (r0 is 120k tokens) are the likely trigger for growth after capture **(inferred)**.
- Upstream fix: vllm-project/vllm#55341 (685074cd6): warm up before capture and lock after it (the other agent's
  untested port is in `defect-repros/D5-workspace-lock/`).

### Observation (same session; log only, no behaviour change)

- `VLLM_DEBUG_WORKSPACE=1` produces the engine's own resize and lock lines.
- The observer records:
  - every **capture binding**: a workspace view handed out while a stream is capturing, with pointer, bytes, runtime
    mode and batch size (graph size), and caller (e.g. `b12x_mla_sparse.py` `_borrow_workspaces`, the kpool indexer
    gather);
  - every resize, with old/new (pointer, size), phase and caller.
- **Phases:**
  - graph capture: between graphs, inside `capture_model`;
  - post-capture warm-up: V2 `warmup_kernels`;
  - serving: split into the release start-up warm-up requests before ARM, the fixture, and after DISARM;
  - pre_capture (profiling, no graph yet) is listed separately.
- `offline/analyze_d5.py` tabulates the resizes by rank × phase × classification, the bindings per (mode, batch size),
  and the workspace pointer at `capture_model_end`.

### Classification (fixed before data)

- **Per event:**
  - "**captured-graph address invalidated**": the freed range [old_ptr, old_ptr + old_size) contains an address bound
    during any earlier capture;
  - otherwise "**resize, no captured-graph address invalidated**".
- **Overall:**
  - any invalidating event: **exercised: captured-graph address invalidated**;
  - else any resize in capture, warm-up or serving: **exercised: resize, no captured-graph address invalidated**;
  - else **not exercised in this workload** (never "ruled out");
  - if the observer or the capture markers are missing: **invalid**.
- **Confirmed omission:** from source.
- **Caveat:** this is an address-level result. A resize alone does not establish graph corruption. An invalidated
  address shows that a graph retains freed memory, not a corrupted replay; that would need the freed range reused while
  a replay touches it, which is not measured.
- Bindings cover only views handed out by `WorkspaceManager` during stream capture **(inferred: these are the only
  workspace addresses graphs retain)**.
- **Demonstrated computational effect:** not measured. **Verified correction:** not run. **Completion effect:**
  untested.

## D2 / D6: closing with existing evidence

### Already settled

- **D2 tail ring, verified correction in shipped graph mode.** Runs `tailfix/T-graph-a1` and `U-graph-a1`, device guard
  on, 8 × (T, F) phases:
  - on both ranks: null-block alias 0, foreign 0, attention-block collisions 0, tail collisions 0, shadow divergence 0,
    guard errors 0;
  - 104/104 requests finished by length in each run (`tailfix/*/summary.txt`, `client.jsonl`);
  - unmodified 0.9.1 under the same guard (`tailguard/graph-a1`): 98.2% / 98.1% of tail writes went to the null block,
    with 96 / 40 token × layer writes into live attention blocks.
- **Tests:** `test_fix.py` 17/17, covering:
  - G1-G6, the upstream 240785b82 / #59528 cases ported to ring 8;
  - B1-B3, builder output and the persistent-buffer write;
  - K1-K3, the real decode kernel on a padded FULL batch and pool completion;
  - M1-M2, the B12X causal-length source;
  - I1-I2, the real `build_attn_metadata`.
  - Also: CPU scenarios 6/6 and the two-rank harness (`defect-repros/D2-kpool-tail-ring/`).
- **Final patch, as documented.** U (`fix-U.diff`: positions passed to `build_attn_metadata`, the `own_block != 0`
  guard, the circular mapping written into the persistent slot-mapping buffer) is the upstream-equivalent form. T adds
  H4, which keeps the non-pure-decode causal length on the scheduler's CPU bound.
- **D6 stale entries.** Warm-up writes leave 16 stale rows per rank: column 1 holds real block ids
  (`tailguard/RESULT.md`).
  - With D2 unfixed they caused the attention-block writes.
  - With D2 = T or U they remain (stale_col_rows 16) but are never read: 0 foreign / 0 attention writes. Source
    agrees: the circular mapping reads column 0 only (`defect-repros/D6…/expected/release.txt` lines C/D).
  - The D6 patch (one warm-up tail block, matching upstream #53896 / #57169) is CPU-checked only.

### Uncovered conditions

None that needs GPU work for D2/D6 verification.

- **T vs U.** H4 changes results only where `seq_lens_cpu_upper_bound` deviates from the true length *and* the context
  is shorter than the 2,176-token index width (`b12x_mla_sparse.py:3036-3037`; **inferred**, `tailfix/RESULT.md`).
  - Every fixture prompt is at least 4,240 tokens, so the T/U runs could not distinguish the variants. M2 covers the
    mechanism on CPU.
  - A serving test would be needed only to claim T ≡ U in serving. That would be a short-context, multi-token-verify
    workload; it is not needed to verify D2.
- **DCP > 1.** Not the shipped configuration; out of scope (D1 and D2 at DCP > 1 remain unverified, as in INDEX.md).
- **D6 patch on GPU.** Its only observable effect would be stale_col_rows 16 → 0 at start-up. If wanted, it could ride
  on any later tail-guard session; it is not proposed here.

## Run procedure (requires authorization; nothing has run)

```bash
R=<runs>/verify13
$R/run_verify13.sh smoke 1     # 1 GPU (GPU0): hook host-sync and graph smoke, B12x swiglu_limit kernel tests
$R/run_verify13.sh server 1    # 2 GPUs: one server, shipped graph mode, capture armed after readiness
$R/run_verify13.sh offline 1   # 1 GPU (GPU0): D3 comparison, D4 inventory, D5 analysis
```

**Every step:**

- pre-flight: no glm53* container; no non-desktop compute process; GPU0 ≤ 1 GiB used; GPU1 ≤ 4 GiB used (desktop
  graphics accepted, amendment 11b); utilisation ≤ 10%;
- the image ID must equal `image-id.txt`;
- the budget gate (logged GPU-hours plus this step's cap ≤ 2.0);
- an EXIT/INT/TERM trap that stops the container;
- one cost line in `verify13/costs.jsonl`, counted up to the moment the GPUs are released.

**Server step.**

- Launch: stack and recipe unchanged, `IMAGE` = verify13 image, `EXTRA_SERVE_ARGS` unset (shipped graph mode),
  `DEBUG_ENV` = the `GLM53_VERIFY13_*` sampling variables plus `VLLM_DEBUG_WORKSPACE=1`.
- Start-up gate: 1,200 s. Abort unless both ranks logged `installed`.
- Then `ARM`, then `client_v13.py`: the col511-gate fixture with REPEATS=6 (warm, then T/F/D/S), the same requests and
  seeds as `gate/server-a1`, distinct per request and repeat.
- Then `DISARM`, waiting for `FLUSHED_r0/_r1` (at most 90 s).
- Then `docker logs`, stop, and the cost line. The capture is copied out of the retained stopped container *after* the
  GPUs are freed.
- Post-steps: D5 analysis, configuration comparison against the historical log and the gate run, T/F throughput
  against the gate run, and the summary.
- Wall cap: 1,800 s.

**Measured basis for the estimates** (`gate`, `tailfix`, `tailguard` `costs.jsonl` and `gate.txt`):

- start-up to ready: 757 / 748 / 727 / 726 / 757 s (mean 743 s);
- the gate session on this exact fixture with 6 repeats: 1,252 s in total = 0.696 GPU-hours;
- T/U/T1/T3 sessions (8 × T/F): 1,021-1,052 s.

### Invalid-run rules (fixed before data)

The server run is **invalid**, and may be repeated once within the budget, if any of these holds:

- hooks are not installed on both ranks;
- either rank fails to flush;
- `gate_capturing` = 0;
- any record has mode ≠ NONE;
- `hook_error` > 0 affecting more than 1% of sampled records;
- client errors occur, or finishes other than `length`;
- the effective configuration differs from the gate or historical run beyond the intended differences (image,
  `GLM53_VERIFY13_*`, `VLLM_DEBUG_WORKSPACE`);
- the graph-capture counts are not 11 / 9 / 9.

If the smoke step fails `no_host_sync_in_hooks` or `graph_capture_skips_hooks`, the server step is **not** run, and the
fix gets a new proposal. (Rules unchanged by 13a. The new smoke check `d5_capture_binding_and_invalidation` does not
gate the server step; if it fails, D5 is reported as "observer not validated on GPU".)

**D3 offline stop.** If control (a), (b) or (c) fails, `run_verify13.sh offline` exits non-zero after writing
`d3_controls_failed.json`. That is an analysis outcome, not an invalid server run: the captured data stand, and the
diagnostic decides the next step. Any re-run of the offline step counts against the same 2.0 budget.

### Expected data volume (inferred from the gate run's eager-call counts)

- About 780 eager forwards per rank, × 2 layers, gives about 1,560 records per rank.
- Rows: 64 per window, up to 256 for whole small batches, so about 0.1-0.15 M token × layer rows per rank. That is
  about 2-3 GB per rank under the 3 GiB cap, roughly 2,900 rows per layer.

## Files

- `run_verify13.sh` (smoke | server | offline) and `client_v13.py`
- `build/`:
  - `Dockerfile`, `verify13_capture.py`, `gate_linear.py`, `gate_linear.diff`, `make_patch.py`;
  - `test_verify13.py`;
  - `build.log`, `image-file-diff.txt`, `image-id.txt`;
  - `../release/gate_linear.py` (the release file the patch was generated from).
- `offline/`:
  - `d3_router_compare.py` and `test_d3_offline.py` (5 tests: helpers, tolerance and attribution, end to end with
    labels/tables/bootstrap, stop on a 1-ulp logit mismatch with diagnostic, stop on a routing mismatch);
  - `smoke_gpu.py`;
  - `analyze_d5.py`;
  - `config_compare.py` (copy of `kdcp/gate/`) and `throughput_tg.py` (copy of `kdcp/tailguard/`).
- Scratch originals: `kdcp/verify13/`. The pre-13a version of this proposal is `kdcp/verify13/PROPOSAL.pre13a.md`.
