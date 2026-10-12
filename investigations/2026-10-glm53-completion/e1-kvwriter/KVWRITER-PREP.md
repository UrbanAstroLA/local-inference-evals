# KV writer and query projection: cheapest discriminating comparison (prepared, not run)

PLAN.md amendment 16, parallel CPU work. Prepared 2026-10-12. **No GPU was used and the comparison has not been run.**
The only things executed were CPU input preparation and an emulator self-test on synthetic data.

## Question

Do B0's KV writer and query projection differ from the 0.7.0 alternatives by more than expected rounding?

| component | B0 (0.9.1 bundle, NOPE record) | alternative (0.7.0) |
|---|---|---|
| **KV writer** | B12x `concat_and_cache_glm_next_mla`: 528 B record = 512 E4M3 + 4 × fp32 group scales | vLLM `concat_and_cache_mla(kv_cache_dtype="fp8_ds_mla")`: 656 B record = the same 528 B + 128 B bf16 RoPE tail, which is always zero (`qk_rope_head_dim = 0`) |
| **query absorption** | `torch.bmm(q_nope (64,B,256), W_UK_T (64,256,512))`, 512 wide | B12x GLM H64 projection for 2 ≤ rows ≤ 16, 576 wide with 64 zero lanes; `torch.bmm` otherwise |

The sources are the phase-1 audits: `report-dcp.md` D2, `report-dispatch.md` Q1 and Q2, and `audits/phase1-*`.

## Already settled at zero GPU cost

### The writers

**vLLM `fp8_ds_mla` writer: fully determined, so it can be emulated on CPU bit for bit.** Source:
`csrc/libtorch_stable/cache_kernels.cu:445-545` at `487ecf187`.

- **Arithmetic:**
  - `scale = max(amax/448, FLT_MIN)`;
  - `byte = e4m3_rn_satfinite(v/scale)`.
- **Both divisions are IEEE round-to-nearest.** The image's sm_120 SASS shows `FCHK` plus the `div_rn` slow path, then
  `F2FP.SATFINITE.E4M3` (`sass/vllm_concat_and_cache_ds_mla_bf16_e4m3.sass`).
- **Same library in both images.** `_C_stable_libtorch.abi3.so` has sha256 prefix `2b096e4b…` in stock 0.7.0
  (`48e254d9`) and in B0, so the 0.7.0 writer can run inside the B0 image unchanged.

**B12x GLM_NEXT writer: one approximate step.** Source: B12x `attention/_shared/mla/kv_cache.py`,
`ConcatAndCacheGlmNextMlaKernel`.

- **Arithmetic:**
  - `scale = amax · fl(1/448)`, or `1.0` for an all-zero group;
  - `byte = e4m3_rn_satfinite(v · rcp.approx.ftz(scale))`.
- `rcp.approx` is a hardware approximation within 1 ulp. A CPU emulation therefore brackets each byte: `inv` is taken
  as RN(1/scale) and ±1 ulp, and the GPU byte must be one of the three outcomes.

**Expected rounding between the two writers (model fixed now).**

- The values before rounding differ by a relative δ: typically about 2⁻²³ (1.2e-7), at most 7·2⁻²⁴ (4.2e-7). This
  combines `fl(1/448)`, the product rounding, the 1-ulp `rcp`, and the two IEEE roundings.
- A stored byte differs only when `v/scale` lies within δ of an E4M3 rounding midpoint. Midpoints are 1/16 to 1/8 of the
  value apart, so the predicted rate is **2e-6 to 1.3e-5 per element** (0.1-0.7% of tokens with one differing byte).
  Each such difference is exactly **one E4M3 code**.
- Exact ties (a group whose amax is 448·2ᵏ) are decided by the RNE tie rule in vLLM but by the `rcp` value in B12x.
- Group scales differ by at most 1 fp32 ulp.
- All-zero groups store FLT_MIN (vLLM) or 1.0 (B12x). Both decode to exact zeros.
- Dequantisation error is at most ½ E4M3 step for both writers.
- **Emulator self-test** (synthetic set with planted ties, CPU, `kvw-inputs/selftest.json`):
  - 0 elements beyond ½ step for either writer;
  - cross-writer differences 1,409 of 2.1M elements, mostly at the planted midpoints;
  - 0 differences larger than one code;
  - scales within 1 ulp.

### The projection

**The B12x test suite claims bitwise equality with `torch.bmm`.** Both versions assert it, with production views:

- `tests/gemm/test_mla_query_projection.py::test_glm_h64_bf16_matches_staged_projection_with_production_views`
  (m = 1, 2, 16, 32);
- `…nope_writes_exact_zero_suffix` and `…nope_cuda_graph_replay` both use `torch.equal(out[..., :512], torch.bmm(...))`.

If that holds on this GPU with real weights and nope_dim 256, the two paths are identical in the 512 latent lanes. The
576-wide query's extra 64 lanes are zero and multiply a zero RoPE tail, so they contribute exactly 0.

**The kernel file differs between images.** The H64 kernel source is `_bf16.py`: sha256 prefix `b8023ca8…` in 0.7.0
and `5a31d620…` in B0. The projection therefore runs in both images.

## Inputs

`kvw-inputs/kvw_inputs.pt`, sha256 `ed4498e63d369969d25b57ceff8e7758ee852e9fd4b7f1f26f987e15633bc71a`, built by
`kvw_prep_inputs.py` (CPU, about 3 s).

**No MLA latents or queries were ever captured.** The Stage A captures hold the MoE gate input; the phase-2/3 probes
hold indexer pages.

**Real-weight proxy, MLA layers 3, 23 and 43:**

- **Source rows:** the captured bf16 hidden states of real serving rows: prefill and mixed batches, contexts 4k-120k.
- **Attention input approximation:** `h = x / w_post · w_in`, using the same layer's two RMSNorm weights.
- **Latents:** the checkpoint's bf16 `kv_a_proj_with_mqa` then `kv_a_layernorm` give `kv_c`: 4,869 / 2,368 / 2,501
  tokens; absmax 2.2 / 4.9 / 4.5; RMS 0.43 / 0.99 / 0.79.
- **Queries:** `q_a_proj` → `q_a_layernorm` → `q_b_proj` give `q_nope`, 64 heads × 64 rows × 256.
- **Projection weight:** `W_UK_T` = `kv_b_proj` split as the engine does (`mla_attention.py:1158-1166`): (64, 256, 512).
- **bf16 rounding** after each linear and norm. This is realistic in distribution, not bitwise to serving.

**Synthetic set:** 4,096 tokens.

- Gaussian with per-channel spreads taken from the proxy latents.
- Edge cases:
  - all-zero groups;
  - single large outliers, which push the rest of the group toward E4M3 subnormals;
  - groups around 1e-6;
  - groups placed exactly on E4M3 midpoints with amax 448.

**Not used, and why:**

- **Purely random latents:** the engine's own `scripts/test-glm-next-records.py` uses them.
- **A fresh in-server capture of real `kv_c` and `q_nope`:** this would be the strictly real alternative, at the cost of
  one server session (about 0.3-0.5 GPU-h plus about 3 h of hooking). It is not proposed unless the proxy result is
  ambiguous.

## References

- **Writers:** fp64 values of the bf16 latents themselves, not the dequantised records of either writer. The engine
  script's reference, built from the 528 B records, would favour B0.
- **Projection:** fp64 `bmm` of the identical bf16 `q_nope` and `W_UK_T`.

## Metrics and decision rule (frozen here, before any GPU data)

| id | metric | expected if only rounding |
|---|---|---|
| W1 | vLLM GPU record bytes and scales vs the CPU emulation | **bitwise equal**, 100% |
| W2 | B12x GPU bytes inside the rcp bracket; scales equal to `fl(amax·fl(1/448))` | 100% inside; scales equal |
| W3 | cross-writer latent bytes: differing fraction, size of difference, scale ulp | ≤ 1.3e-5 per element; every difference = 1 code; scales ≤ 1 ulp (non-zero groups); RoPE tail all zero |
| W4 | each writer's dequant error vs fp64 input: max in E4M3 steps, fraction > ½ step, RMS / group amax | ≤ ½ step (+1e-6) for both; the two RMS errors equal within 1% |
| R  | repeatability (each op twice) and slot independence (identity vs permuted slots) | all equal |
| P1 | H64 `[..., :512]` vs `torch.bmm`, rows 1, 2, 4, 8, 16 | bitwise equal (as the B12x tests assert); tail lanes exactly 0 |
| P2 | each projection vs fp64: \|out − ref\| ≤ 2⁻⁹·\|ref\| + 256·2⁻²⁴·Σ\|q\|\|w\| | 0 elements outside |

**Verdict:**

- **"Beyond expected rounding"** means any of:
  - a W1 mismatch;
  - a W2 byte outside the bracket;
  - a W3 difference larger than one code, or a rate above 1.3e-4 (10× the predicted upper bound);
  - a W4 element beyond ½ step;
  - a P2 element outside the bound;
  - non-repeatability.
- **A P1 inequality with P2 clean** is reported as a rounding-level difference: it would falsify the B12x test claim,
  not establish a defect.
- **All within** retires writer and projection as a source of 0.7.0-vs-B0 difference beyond rounding, for these inputs
  and shapes. It does not rule out a completion effect of rounding-level differences.
- **Any failure** is first localised: which element, which group, which code path. If a correction is warranted, it
  becomes a B1 candidate. Note the caveat: swapping the record format is not a single change. It alters page size,
  pool size and the >2³¹ threshold (D9).

Analysis: `kvw_analyze.py analyze <inputs> <run files>` on CPU writes `kvw_analysis.json`.

## Procedure (when authorised)

Single GPU, model-free: no server, no checkpoint load. Check that nothing else is on the GPU first, using the Live tab;
`nvidia-smi --query-compute-apps` misses graphics clients.

```bash
W=<runs>/e1-kvwriter
# B0: both writers (vLLM _C is shared) + bmm + B0's H64
docker run --rm --gpus '"device=0"' --network none -e IMAGE_TAG=b0 \
  -e IMAGE_ID=sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548 \
  -v $W:/w --entrypoint python3 local/glm53-v0.9.1-bundle-d1234:local \
  /w/kvw_gpu_compare.py /w/kvw-inputs/kvw_inputs.pt /w/kvw-runs/b0.pt
# stock 0.7.0: vLLM writer + bmm + 0.7.0's H64 (GLM_NEXT writer skipped automatically if absent)
docker run --rm --gpus '"device=0"' --network none -e IMAGE_TAG=v070 -e IMAGE_ID=48e254d94f58 \
  -v $W:/w --entrypoint python3 48e254d94f58 \
  /w/kvw_gpu_compare.py /w/kvw-inputs/kvw_inputs.pt /w/kvw-runs/v070.pt
# CPU analysis
docker run --rm --network none -v $W:/w -w /w --entrypoint python3 local/glm53-v0.9.1-bundle-d1234:local \
  /w/kvw_analyze.py analyze /w/kvw-inputs/kvw_inputs.pt /w/kvw-runs/b0.pt /w/kvw-runs/v070.pt
```

**Script status:**

- `kvw_gpu_compare.py` has not been executed. Only its syntax has been checked.
- Its API calls exist in the B0 image: `ops.concat_and_cache_mla`, `backend._prepare_glm_next_writer_plan`,
  `sparse_mla.concat_and_cache_glm_next_mla`, `mla_query_projection.run_glm_h64_bf16`. They are the same calls the
  engine's `test-glm-next-records.py` makes.
- Before the run, do a CPU-side dry run of the 0.7.0 image's `run_glm_h64_bf16` signature, and add the per-run hashes
  to the freeze line.

**Optional extension** (+5 min per image, same session): the engine's `scripts/test-glm-next-records.py --heads 64`
carries the records through the prepared sparse-MLA decode/extend plans. It compares NSA-576 against NEXT-512 against
fp32. Report its numbers, not its pass/fail: its tolerances (cos ≥ 0.9995, max abs 3e-2) are far looser than rounding.

## Cost estimate

**GPU, one GPU per image:**

| step | time |
|---|---|
| container start and CUDA init | about 0.5 min |
| CuTe-DSL / Triton JIT of the writer and the H64 kernel (no cache in a fresh container) | 1-3 min |
| the ops themselves | seconds |

That is about 3-5 minutes per image, or **0.10-0.17 occupied GPU-hours** for both images, startup included. Proposed
ceiling **0.25 GPU-h**, which also covers the optional extension. An invalid attempt may be repeated within the ceiling.

**Engineering:**

- **Done in this preparation:** about 1.5 h (inputs, emulators, GPU script).
- **Remaining:** about 3 h:
  - pre-run dry checks of the 0.7.0 API, 0.5 h;
  - the run and its supervision, 0.5 h;
  - extending the analysis with a QK-logit downstream readout if wanted, 1 h;
  - review and write-up, 1 h.

**What it can change:**

- **Clean:** writer and projection drop off the list of 0.7.0-vs-B0 mechanisms (D2 moves from "FP only, unverified" to
  "measured equal within rounding"). No B1 follows from it.
- **Localised discrepancy:** the next smallest test targets that code path only.

## Files

| file | role | status |
|---|---|---|
| `kvw_prep_inputs.py` | input builder (CPU) | run |
| `kvw-inputs/kvw_inputs.pt` (+ `.sha256`) | prepared inputs | built |
| `kvw_gpu_compare.py` | GPU step | not run |
| `kvw_analyze.py` | emulators and frozen metrics (CPU) | `selftest` run on synthetic data only |
| `kvw-inputs/selftest.json` | emulator self-test output | written |
| `sass/vllm_concat_and_cache_ds_mla_bf16_e4m3.sass` | writer SASS evidence | saved |
