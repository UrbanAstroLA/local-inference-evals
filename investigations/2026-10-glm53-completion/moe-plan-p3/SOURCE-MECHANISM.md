# P3: where the decode-vs-prefill plan difference comes from (source analysis, 2026-10-23)

No GPU was used, and nothing under `PRIVATE/` was read. The CPU checks ran as `docker run --rm --network none -e CUDA_VISIBLE_DEVICES=`
on image `a9e985c0` (torch on CPU). Scripts and outputs are in `mechanism/`.

**Labels:** **S** means read in source, with file:line. **I** means inferred. **C** means a CPU emulation result (synthetic weights).

**Files.** The recon copies are byte-identical to the image files (sha256 checked in both images):

| File | v091 image `a9e985c0` (B12x 7fcc094e) | v070 image `48e254d9` (B12x fe054789) |
|---|---|---|
| `b12x/moe/_shared/kernels/w4a16/kernel.py` (`K`) | `49479e2d…76b9b` | `75dfa21f…e2a71` |
| `b12x/moe/_shared/kernels/w4a16/mixed_trellis.py` (`MT`) | `bd1538b9…12003` | `daa2e89f…5189d` |
| `b12x/moe/fused_moe/_impl.py` (`FI`) | `5511938e…252cd` | `96d189a4…a4c67` |
| `vllm/…/quantization/exl3.py` (`X`) | `3b4ca47c…444d9` | `0560a6ca…1d1ad` |

Line numbers below are v091. In v070 every cited construct is present, shifted by about 60-180 lines. Examples:
`schedule_whole_tiles=True` is MT:1913, the tile rule is FI:7760, `red_off` is K:5135 and the fp16 activation store is
K:7701.

## Answer in one paragraph

**Scope (added in 22au).** The forced-tile GPU result (22at) establishes tile choice as the cause of the measured
output difference on the tested inputs. The magnitude explanation below, through fp16 staging, rests on source
analysis and CPU emulation. It is not a direct GPU ablation of each staging operation.


The two plans run the same arithmetic per element, with one exception: **the fp32 summation order inside each GEMM.**
The CTA tile sets how many warp rows split K and how their partial sums are folded:

- **K64/N256:** 2 warp rows.
- **K128/N128:** 4 warp rows.

In both plans the fp32 result is then **rounded to fp16 at three staging points:** the FC1 output, the activated
intermediate and the FC2 output per route. A Hadamard transform sits between stages and spreads every 1-ulp fp16 flip
across 128 channels. Each rounding therefore turns a small dense difference δ into a larger one, about √(c·δ), with
c ≈ 5e-4 for fp16. Three stacked roundings take an fp32 reordering difference of about 2e-7 to 2e-6 to about 1-2e-4.

The final `.to(bf16)` does the same once more and gives about 8e-4. A CPU emulation of this data path, with random
weights, gives:

| Rounding model for MMA accumulation | fp32 rel L2 | bf16 rel L2 | bf16 elements differing |
|---|---|---|---|
| round-to-nearest | 1.1e-4 | 6.4e-4 | 6.3 % |
| round-toward-zero | 2.1e-4 | 8.6e-4 | 10.7 % |
| **Measured** (954 fixtures, median) | **2.0e-4** | **8.4e-4** | **9.8 %** (range 7.2-11.7 %) |

The same emulation with fp32 intermediates (no fp16 staging) gives 6e-7 fp32 and 0.07 % bf16 elements differing. That
cannot produce the measured numbers.

## Data path of one routed token (both plans), stage by stage

| # | Stage | Arithmetic | Plan-dependent? |
|---|---|---|---|
| 0 | dispatch / tile choice | Mixed Trellis takes `force_tile_config=_projection_mixed_tile_config(...)` (FI:8649). The projection-mixed prepare passes `tile_config = None` (X:2480), so the rule applies: K128/N128 if `not direct_topk_routes and 10 <= token_count <= 32`, otherwise K64/N256 (FI:8310-8331). Decode launches have token_count = m. Prefill is one launch with token_count = 1024, so K64/N256. **S** | Yes: this is the switch |
| 1 | input rotation | bf16 x → fp16 → ×suh (fp16-rounded) → H128 butterfly in fp32 → fp16 store, per (route, 128-block) warp unit (K:7296-7400; comment K:7311-7317; stores K:7392). **S** | No. Per-route element-wise, fixed order. **S** |
| 2 | FC1 GEMM (K = 4096, N = 2 × 1024 on tp2r0) | Trellis codes are decoded to fp16 weights (K:4082-4133). `mma.m16n8k16` with f16 operands and an **fp32 accumulator** (K:1445-1461). The m8 path swaps the operands (K:1463-1483, K:4472); the large-m path does not (K:4513). Accumulators are fp32 (K:2335, K:2695-2719). **S** | **Yes, see "What differs"** |
| 3 | CTA fold of the warp-row partials | fp32 through shared memory (K:5308-5450 m8; K:5748-5950 large-m). **S** | **Yes:** the fold tree differs |
| 4 | split-K across CTAs | Exists: fp32 `c_tmp` with locks (K:3414-3441). Mixed Trellis compiles with `schedule_whole_tiles=True` (MT:1973; rationale MT:1923-1926), so one CTA owns the full K of each MN tile (K:1638-1643). **S** | No: not used in either plan |
| 5 | FC1 epilogue | `acc * global_scale` (fp32, K:5660-5661), then **rounded to fp16** (`_st_shared_elem_from_f32` K:1438-1442; large-m `_pack_f32x2_to_elem2` K:1417, K:5950-5960). There is no top-k multiply: FC1 has `mul_topk_weights=apply_router_weight_on_input=False` (K:6409). **S** | Same op; its fp32 input differs |
| 6 | activation | fp16 FC1 → fp32 → H128 → ×svh (gate, up) → silu(g)·u·s_down → H128 → **fp16 store** (`_run_activation_compact` K:7748-7889; `_cast_elem` → Float16 K:6540-6543; `element_dtype="fp16"` MT:1962). Rows are indexed by logical route, so "packing order cannot change arithmetic" (K:7765-7767). **S** | Same op; its input differs |
| 7 | FC2 GEMM (K = 1024 tp2r0 / 2048 ep2r0, N = 4096) | As stages 2-3. **S** | **Yes** |
| 8 | FC2 epilogue | `acc * global_scale` → **fp16 per route**. No top-k weight here: `mul_topk_weights = not apply_router_weight_on_input and not full_rotation` = False (K:6439-6441; K:6380). **S** | Same op; its input differs |
| 9 | top-k sum and output rotation | Separate kernel, fully fp32: fp16 → fp32 → H128 → ×svh → weighted sum over the 8 top-k slots in slot order → fp32 store (K:8775-8873; `full_rotation_output_dtype="fp32"` because `B12X_EXL3_BF16_EPILOGUE=0`, X:2899-2901, X:2964-2968; output dtype check MT:3168). **S** | No: per token, fixed order |
| 10 | vLLM | `output.to(x.dtype)`, fp32 → bf16 (X:3200, X:3216). **S** | Same op; its input differs |

Stages 2-3 and 7 are the only places where the two plans compute a value differently. Stages 5, 6, 8 and 10 are the
roundings that amplify the difference. **I**, from the table above.

## What differs between K128/N128 and K64/N256 (stages 2, 3 and 7)

**Geometry (S):**

- `cta_threads = tile_n*tile_k//64` gives 256 for both (K:1062).
- `tb_n_warps = cta_n_blocks//4`: 2 for N128, 4 for N256 (K:1280).
- So 8 warps form **4 K-rows × 2 N-warps for K128/N128**, and **2 K-rows × 4 N-warps for K64/N256**.
- `b_sh_wr_iters` = 2 in both (K:1243-1247).
- Warp row r reads k16 slabs `tile*cta_k_blocks + 2r + kk`, with kk ∈ {0, 1} (K:4414-4418, A side K:2266-2276).

**Summation order per output element:**

- **K64/N256:** row r ∈ {0,1} accumulates k16 chunks c with c mod 4 ∈ {2r, 2r+1}.
  - FC1 (K = 4096): 128 MMAs per row. FC2 (K = 1024): 32 MMAs per row.
  - Fold: `acc0 + acc1` (`red_off = 256/128/2 = 1`, K:5309).
- **K128/N128:** row r ∈ {0..3} accumulates chunks with c mod 8 ∈ {2r, 2r+1}.
  - FC1: 64 MMAs per row. FC2: 16 MMAs per row.
  - Fold: `acc0 + ((acc1 + acc3) + acc2)` (`red_off = 2`, K:5316-5446).

Both plans add the same products. They differ only in fp32 association. **S** for the structure, **I** for the exact
fold expression, read from the store and load offsets.

**What does not differ:**

- No partial sum is rounded below fp32 (the fold and split-K are fp32; **S**).
- The accumulator precision does not differ.
- The epilogue does not differ: the same `global_scale` multiply and fp16 cast.
- The MMA instruction shape does not differ: it is m16n8k16 in both plans.

The m8 path (decode, `moe_block_size` 8) swaps operand roles relative to the large-m path (prefill, block 48). The
m ≤ 8 bit-equality shows this swap is numerically inert (see the next section). **I**

**Size of the raw fp32 difference (C):** FC1 output before rounding, rel L2 between partitions:

- 2.4e-7 with round-to-nearest accumulate;
- 1.6e-6 with round-toward-zero.

The true tensor-core accumulate rounding is not specified here. The RZ model matches the measurement better. **I**

## Why ~2e-4 and not ~1e-6: amplification by fp16 staging (I + C)

**Rounding rule.** Two values that differ by a small relative δ (δ ≪ ulp) round to the same fp16 value except with
probability ≈ |δx|/ulp(x). When they do not, the difference is one ulp. After rounding, the rel L2 of the difference is
therefore about √(c·δ), with c ≈ (mean relative ulp) ≈ 5e-4 for fp16 and ≈ 4e-3 for bf16.

**Spreading.** The H128 transforms in stages 6 and 9, and the GEMM in stage 7, turn the sparse 1-ulp flips into a
dense δ for the next rounding.

**The chain:**

| Stage | Before | After |
|---|---|---|
| FC1 fp32 difference | δ0 ≈ 2e-7 to 2e-6 | |
| fp16 FC1 store | | flips in 0.17-0.6 % of elements (C); dense ~1.4-1.7e-5 after H128 |
| fp16 activated | | ~4e-5 (C) |
| FC2, then fp16 store | | 1.1-2.1e-4 (C) |
| fp32 top-k sum | | carried through unchanged |
| bf16 cast | √(4e-3 × 2e-4) | ≈ 9e-4. **Measured 8.4e-4** |

**Two consequences:**

- The final size goes roughly as δ0^(1/8). It is therefore nearly independent of how large the underlying fp32 change
  is. **I**
- It is also independent of input scale and routing (fp16 rounding is relative, and the rotations are orthonormal),
  as measured. **I**

**Ablation (C, seeds 0 and 1):**

| fp16 roundings kept | fp32 rel L2 |
|---|---|
| FC1 only | 1.4-1.7e-5 |
| FC1 + activation | 3.9-4.1e-5 |
| FC2 only | 1.5-1.7e-5 |
| all three | 1.1e-4 (RNE) / 2.1e-4 (RZ) |
| none | 6.3e-7 |

**Dominant source:** the *stacking* of the three fp16 staging roundings, seeded by the tile-dependent fp32 K-reduction
order. The last stage (FC2 output fp16) contributes the largest single step. **I + C**

**No accuracy advantage is established (C).** The CPU emulation gives similar reference errors for the two summation
orders (wording corrected in 22au). Against an un-staged reference (fp32 intermediates), each plan's error is the
same, 4.19e-4 fp32 and 1.2e-3 bf16, for both tiles. The decode-vs-prefill difference is about ¼ to ½ of each plan's own
fp16-staging error. Any change of K order would produce a difference of this size: another tile, a warp split, a
`B12X_W4A16_SMALL_M_SPLITK`-style split-K, or another kernel. **I**

## Is the m ≤ 8 bit-equality fully explained by identical tiles?

**Yes, for the arithmetic.** At m ≤ 8 both plans launch fc1/fc2 K64/N256 (`launch_map`), so stages 2-3 and 7 use the
same warp-row partition and fold. All other stages are per-route element-wise with a fixed order (S, table rows 1, 6
and 9) and no split-K (S, row 4).

**Other things differ at m ≤ 8 and are evidently inert:**

- `moe_block_size` 8 vs 48: the m8 path with swapped MMA operands vs the large-m path with `cta_m_blocks = 3`.
- `max_m_blocks`, shared memory, grid waves and route packing.
- The `size_m==1` single-token fast path (K:6426, K:6458).

**Supporting evidence:**

- **m = 40:** about 81 % of rows differ. This matches 32 rows at K128 against 8 rows at K64/N256, which agree (I).
- **E-version:** fe054789 and 7fcc094e are bit-equal. In both versions the cited constructs are the same.

**Caveat (I):** the operand swap could in principle change the summation inside an MMA. It did not here, at least for
the m ≤ 8 fixtures measured.

## Smallest confirming checks (not run on GPU)

1. **Decisive, GPU, about 1 minute after the plan compiles.**
   - **Setup:** build the decode plan with a forced `trellis_tile_config=(64,256,64,256)`. This is the
     `configured` argument of `_projection_mixed_tile_config`, set via the weight plan's `trellis_tile_config`, today
     `None` at X:2480.
   - **Run:** one m=16 fixture and one m=32 fixture.
   - **Prediction:** bit-equal to the prefill plan.
   - **Converse:** with the stock K128 decode tile, the fp16 scratch buffers can be compared stage by stage (FC1,
     activated and FC2 regions of the arena). Predicted differing fractions: FC1 ≈ 0.2-0.6 %, mostly ±1 ulp
     (up to a few ulps near zero); then activated; then FC2. This rises from stage to stage.
2. **CPU, already done (C):** `mechanism/sim.py`, with ablations. It reproduces the magnitude and
   the bf16 flip fraction from source-level dtypes. It cannot prove the GPU's MMA rounding mode.
3. **Optional, GPU:** rerun one fixture with fp32 staging, if a debug toggle exists. None was found in this source for
   the mixed-Trellis path: `element_dtype="fp16"` is hard-coded at MT:1962. Expected result: about 1e-6.

## Limits

- The emulation uses Gaussian fp16 weights, ±1 sign vectors for suh and svh, uniform 1/8 top-k weights and 8
  concentrated experts. It does not use the real trellis codebook values or the real scales.
- Tensor-core fp32 accumulate rounding is modelled two ways (RNE and RZ). The real hardware behaviour is unverified.
- **EP slice (v070 `ep2r0`):** FC2 K = 2048, and the harness records a bf16 raw output. Not traced separately. Its
  measured bf16 numbers match tp2r0 (median 10.0 % elements, 8.42e-4).
- This explains the *size* of the per-layer plan difference. It says nothing about its effect on decode behaviour.
