# E1 assessment: batch-size dependence of the served routing weights

PLAN.md amendment 16, parallel CPU work. Written 2026-10-12. No GPU was used.

- **Inputs:** read-only Stage A captures (`verify13/server-a1/verify13/rec_r0`: 1,593 records, 103,608 rows). These came
  from the stock 0.9.1 image (`verify313`) with bf16-valued fp32 logits.
- **Confirmation on B0:** the bundle gate's s1 captures (`bundle/gate-runs/s1-d1234-a1/verify13/rec_r0`: 500 records,
  34,672 rows, fp32-output logits).
- **Containers:** all computation ran in CPU-only containers of the B0 image (`--network none`, no `--gpus`).

## Verdict

**Expected numerical variation, not a defect.** The weight difference above 1,024 rows is the arithmetic of a
different kernel. The engine picks one of two kernels by the number of rows:

- **≤ 1,024 rows:** the routing weights come from the stored sigmoid `s`.
- **> 1,024 rows:** the kernel rebuilds the unbiased sigmoid from the biased selection score, as `(s + b) − b` in fp32.

This model's correction biases `b` are large (5.0 to 14.5), so the rebuilt `s` is rounded to an absolute grid of
2⁻²¹ to 2⁻²⁰. That costs about 20 bits of absolute precision. The larger-M path:

- **matches its own arithmetic:** a CPU emulation of the source reproduces the served weights bitwise on 98% of rows;
- **stays inside its own error model:** 0 of 729,600 weights exceed the bound, worst ratio 0.83;
- **is less accurate than the ≤ 1,024 path:** about 10× larger error, and above that path's bound on 35% of weights.

It is not below the accuracy its own documented arithmetic should give. Expert selection cannot be affected (see
below). The effect is roughly three orders of magnitude smaller than D3's: p99 |Δw| is 5.6e-7 here against about
1.6e-3 for D3.

**No workaround is proposed.** "Invoke at the serving M", already used for control (c), stays the only accommodation.

## Mechanism (source and binary)

The kernel source is vLLM `csrc/libtorch_stable/moe/grouped_topk_kernels.cu` at the image's commit `487ecf187`
(`vllm 0.1.dev20051+g487ecf187`). The routing call is `ops.grouped_topk(n_group=1, topk_group=1, topk=8, 288 experts,
renormalize, ×2.5, sigmoid)`. It takes the `single_group_topk` path and the tier `<384 experts, top-8>`.

**Dispatch** (`launch`, source lines ~1199-1228):

```
use_block_kernel = ... || (num_experts >= 256 && num_tokens <= 1024);
```

- **Block kernel** (`single_group_topk_block_kernel`, rows ≤ 1,024): it keeps the unbiased sigmoid in shared memory.
  - `lane_unbiased = unbiased_scores[lane_expert]`
- **Warp kernel** (`single_group_topk_warp_kernel`, rows > 1,024): it does not keep the unbiased sigmoid.
  - `lane_unbiased = lane_score - bias[lane_expert]`, where `lane_score = fl(s + b)`.
- **Both kernels then run `write_outputs`:**
  - `S` = fp32 warp-butterfly sum of the 8 lanes, order ((0+4)+(2+6)) + ((1+5)+(3+7));
  - `scale = 2.5f / (S + 1e-20f)`, an IEEE division;
  - `w_i = u_i * scale`.
  - The weights therefore sum to 2.5, and the ×2.5 is applied to the reciprocal of the sum.
- **Sigmoid in both kernels:** `0.5f*tanhf(0.5f*x)+0.5f`, computed with the hardware approximations MUFU.EX2 and
  MUFU.RCP.

**Binary check.** The image's own sm_120 SASS (disassembled on CPU with the image's `nvdisasm`; excerpts in `sass/`)
matches this source:

- **warp kernel:** `/*2c60*/ FADD R2, -R7, R0`, which is `lane_score − bias` loaded by `LDG` from the bias pointer;
- **block kernel:** `/*2270*/ LDS R6`, which reads the stored unbiased score;
- **both:** the same butterfly, the same `+1e-20`, IEEE division and `FMUL`.

**Same kernel in every engine version tested.** `_moe_C_stable_libtorch.abi3.so` has sha256 prefix `b34e1e3c…` in
these images: stock 0.7.0 (`48e254d9`), 0.7.0-kpoolring, v0.8.0, 0.9.1 verify313 and B0. `grouped_topk_router.py` is
identical in the 0.7.0 and B0 images, and the default is `VLLM_USE_FUSED_MOE_GROUPED_TOPK=1`. B0's diff does not touch the router top-k.

E1 therefore exists identically in 0.7.0 and cannot by itself explain a version difference. The 0.7.0 model-level
router call was not re-traced.

**Observed switch.** Offline attempt 3 placed it between 1,025 and 1,139 rows. The captured M values fall on either
side of 1,024: at most 1,002 below, at least 1,139 above. This is consistent with the threshold of 1,024. The exact
boundary was not observed.

## Is the arithmetic reproduced? (`e1_emulate.py`, `e1_candidates.py` → `e1_emulation.json`, `e1_candidates.json`)

The sigmoid is emulated in fp32 numpy as CUDA's tanhf instruction sequence, with exp2 and the reciprocal correctly
rounded. Hardware MUFU cannot be emulated bitwise. The selected expert ids, and therefore the lane order, come from
the capture.

| Stage A rows | emulation | rows bitwise equal | weights bitwise | max ulp |
|---|---|---|---|---|
| M > 1,024 (91,200) | warp arithmetic `(s+b)−b` | **89,389 (98.0%)** | 98.0% | 110 |
| M > 1,024 | block arithmetic `s` | 0 | 5.7% | 17,254 |
| M ≤ 1,024 (12,408) | block arithmetic | **7,703 (62.1%)** | 87.7% | 16 |
| M ≤ 1,024 | warp arithmetic | 0 | 5.5% | 278 |

On B0's fp32-logit captures:

- **M > 1,024:** the warp arithmetic matches 25,648 of 26,112 rows (98.2%);
- **M ≤ 1,024:** the block arithmetic matches 5,613 of 8,560 rows (65.6%);
- **crossed emulations** (each path's arithmetic applied to the other's rows): 0 rows.

**Why the residual mismatches remain.** They come from the hardware's approximate EX2/RCP inside the sigmoid. On the
warp path, the implied change of the rebuilt score is at most 1.08 grid steps. That means a hardware sigmoid landed on
the other side of a rounding boundary of `s+b`. The block path, which has no grid, shows ≤ 16 ulp.

**Requested alternative arithmetics, tested on the M > 1,024 rows.** None reproduces the served weights on any row:

| candidate | median ulp from served |
|---|---|
| sequential summation | 6 |
| `(u·2.5)/S` | 6 |
| `(u/S)·2.5` | 6 |
| fp64 renormalisation | 6 |
| fp16 sigmoid | about 1,700 |
| bf16 scale factor | about 15,700 |
| bf16 output weights | about 16,400 |

The four fp32 variants equal the plain block arithmetic: median 6 ulp, 0 bitwise rows. bf16 or fp16 intermediates
would give errors of 2⁻⁹ to 2⁻¹¹, which is excluded.

## Accuracy against fp64

The reference is the fp64 sigmoid of the same fp32 logits, over the selected ids, renormalised to 1, then ×2.5. Weights
sum to 2.5. Each row has 8 weights; error = |served − reference|.

| batch rows | rows | abs median | abs p99 | abs max | rel median | rel p99 | rel max | ulp median / p99 / max |
|---|---|---|---|---|---|---|---|---|
| ≤ 256 | 2,552 | 1.4e-8 | 7.5e-8 | 1.7e-7 | 5.2e-8 | 2.3e-7 | 4.4e-7 | |
| 257-512 | 7,488 | 1.4e-8 | 7.1e-8 | 2.3e-7 | 4.9e-8 | 2.2e-7 | 6.0e-7 | |
| 513-1,024 | 2,368 | 1.3e-8 | 6.9e-8 | 1.6e-7 | 4.8e-8 | 2.1e-7 | 7.5e-7 | |
| **all ≤ 1,024** | 12,408 | 1.4e-8 | 7.2e-8 | 2.3e-7 | 5.0e-8 | 2.2e-7 | 7.5e-7 | 1 / 3 / 11 |
| **1,025-2,048 (all > 1,024)** | 91,200 | 1.5e-7 | 5.6e-7 | **1.3e-4** | 5.0e-7 | 2.7e-6 | **1.1e-3** | 6 / 34 / 16,561 |

**Matches the earlier per-row summary.** The per-row maxima reproduce the d3_summary `finding_E1`:

| batch rows | per-row max rel, median | p99 | max |
|---|---|---|---|
| ≤ 1,024 | 1.2e-7 | 3.3e-7 | 7.5e-7 |
| > 1,024 | 1.36e-6 | 3.9e-6 | 1.1e-3 |

No batch above 2,048 rows occurred in this workload. That is the prefill chunk size, so finer bins above 1,024 are not
informative.

**Per expert slot** (slot 0 = highest biased score):

- **> 1,024:** the absolute error is flat across slots, median 1.43-1.48e-7. This is the signature of an absolute grid.
  The relative error rises from 2.5e-7 (slot 0) to 7.9e-7 (slot 7), because lower slots have smaller `s`.
- **≤ 1,024:** absolute median 2.3e-8 to 1.1e-8, relative 4.0e-8 to 5.9e-8.

Full tables are in `e1_emulation.json`.

**Same tokens, both paths** (`e1_counterfactual.py`). On the 91,200 > 1,024 rows, the block arithmetic would have given:

- **abs error:** median 1.35e-8, p99 7.0e-8, max 3.7e-6.

Against that:

- **served warp path:** median 1.45e-7, p99 5.6e-7, max 1.3e-4;
- **paired |warp − block|:** median 1.5e-7, p99 5.7e-7, max 1.34e-4;
- **per-row L1:** median 1.25e-6, p99 3.1e-6, max 3.5e-4.

**B0 captures:** the same picture.

| batch rows | abs median | abs p99 | abs max | rel max |
|---|---|---|---|---|
| > 1,024 | 1.5e-7 | 5.8e-7 | 1.1e-4 | 2.3e-3 |
| ≤ 1,024 | 1.4e-8 | 7.7e-8 | 1.0e-6 | 3.4e-5 |

**Where the tail comes from.** Of the 91,200 rows above 1,024:

- 289 rows (0.32%) have a weight off by more than 1e-6;
- 9 rows have one off by more than 1e-5;
- 31 rows have a relative error above 1e-4.

In the 9 rows, the 8 selected sigmoids sum to only 0.009-0.18 (median 0.019), against a typical 3.1. These are
tokens whose router scores are all low, so the bias decides the selection. Dividing by a small sum magnifies the fixed
absolute grid error. The worst element has `s = 3.8e-4` and `b = 11.6`, which gives a relative error of 1.1e-3.

## Expected error of each path (model stated before checking)

**Notation:**

- `S` = Σ of the 8 selected `s`;
- `h_i` = ½ulp(fl(s_i + b_i)).
- Because `b ∈ [5.0, 14.5]` and `s ∈ (0, 1)`, `s + b` lies in [4, 16). So `h_i` = 2⁻²² (s+b < 8) or 2⁻²¹ (s+b ≥ 8),
  that is 2.4e-7 or 4.8e-7.

**Path ≤ 1,024 (block):** `w_i = fl(s_i · fl(2.5 / fl(S)))`.

- **Sigmoid:** `s = (1 + t)/2`, with `t` from tanhf near ±1. Its error is absolute, about 2⁻²⁵, not relative.
- **Allowance:** 2⁻²⁴ absolute per `s`, plus 6 fp32 roundings (3 in the sum, the division, the product, 1 margin).
- **Bound:** `|Δw_i| ≤ (2.5/S)(a_i + (w_i/2.5)Σa_j) + 6·2⁻²⁴·w_i`, with `a = 2⁻²⁴`. For a typical row (S ≈ 3), this is
  about 1e-7.

**Path > 1,024 (warp):** the same, with `a_i = 2⁻²⁴ + h_i`.

- The rebuilt `u_i` is `s_i` snapped to the grid, error `≤ h_i`. The renormalisation then spreads that error by
  `(2.5/S)`.
- For S ≈ 3, |Δw| ≲ 8e-7. For S ≈ 0.009, |Δw| ≲ 2e-4.
- In relative terms, the error is about `h_i / s_i`: 1e-6 at s = 0.3, 1.6e-3 at s = 3e-4.

**Check** (`e1_bound_check.py` → `e1_bound_check.json`): served error divided by the bound.

| rows | against | median | p99 | max | exceeding |
|---|---|---|---|---|---|
| ≤ 1,024 | block model | 0.07 | 0.26 | 0.42 | 0 |
| > 1,024 | **warp model** | 0.17 | 0.51 | **0.83** | **0** |
| > 1,024 | block model | 0.71 | 2.75 | 7.4 | **34.9%** |

B0 gives the same: 0 exceeding the warp model (max 0.83), and 36.3% above the block model.

**Classification under the conclusion rules.** The larger-M path is less accurate than the other path. It is not less
accurate than its own documented arithmetic allows, and it is bitwise-reproduced by that arithmetic. So it is expected
numerical variation of a design choice: the warp kernel rebuilds the unbiased score from the biased score rather than
storing it. This loses precision when |b| ≫ s.

It is a precision-design limitation of the upstream warp kernel, specific to models with large correction biases. It
is not a malfunction. It is also a batch-invariance property: the same token's weights depend on how many rows share
its batch. Both facts could go upstream as an observation with this evidence. That is not a B0 change and nothing here
requests one.

## Can it change expert selection?

**No, by construction.** Both kernels select from the same fp32 biased scores `fl(s+b)`, and only after selection is
the unbiased score rebuilt for weighting. The ids therefore cannot depend on the path, except through tie-breaking
between two `reduceTopK` implementations, which needs exactly equal fp32 biased scores.

**Measured:**

- **Selection:** 0 differences between captured ids and an fp32 emulation of the selection, on 103,608 rows (Stage A)
  and 34,672 rows (B0).
- **Ranks:** ids and weights are bitwise equal on both ranks for all 1,593 Stage A records.

**Weight-only, and small against the dominant precision limits:**

- E1 p99 |Δw| is 5.6e-7 and the maximum is 1.3e-4.
- D3's weight changes have a per-row p99 of about 1.6e-3, plus 3.15% membership flips.
- bf16 expert activations resolve about 2⁻⁸ ≈ 3.9e-3 relative.

**Completion effect:** untested. Nothing here suggests one.

## Limits

- **Where the data comes from:** the col511-gate long-document fixture (8 histories), eager prefill and mixed batches
  only. Decode-only FULL-graph batches are at most 64 rows and always take the block path. That last point is by
  dispatch rule, not measured.
- **Batches above 2,048 rows:** none in this configuration.
- **Sigmoid emulation:** only to about 1-2 ulp, because hardware MUFU results are not reproducible on CPU. Bitwise
  claims are therefore "98% of rows", with the residual explained.
- **0.7.0:** its kernel library is identical, but its router invocation was not re-traced.

## Files

**Scripts:**

| file | purpose |
|---|---|
| `e1_emulate.py` | both-path emulation, accuracy, per slot, per regime |
| `e1_supplement.py` | M bins, tails, residual analysis, rank agreement |
| `e1_counterfactual.py` | same tokens through both paths |
| `e1_candidates.py` | alternative arithmetics |
| `e1_bound_check.py` | error model |

**Outputs:**

- Stage A: matching `*.json` next to each script.
- B0 gate: `b0-gate-s1/`.
- SASS excerpts: `sass/` (block384, warp384, and the vLLM ds_mla writer used in KVWRITER-PREP).

**Re-run** each script in the same CPU-only container:

```
docker run --rm --network none --user $(id -u):$(id -g) -e HOME=/tmp -v <capture>:/capture:ro \
  -v <models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1:/model-repo:ro -v <this dir>:/out \
  --entrypoint python3 local/glm53-v0.9.1-bundle-d1234:local /out/<script>.py /capture \
  /model-repo/snapshots/0490d2f708b12145f6516555ab066aaeb401cd21 /out
```

Each script takes about 10 s.
