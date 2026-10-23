# P3 Stage 1: real-weight MoE-plan comparison on synthetic inputs (protocol)

Status: **the protocol as approved and run** (operator approval 2026-10-23, ceiling 0.5 occupied GPU-hours, retries
within that ceiling). It was not published before the run; it is published with the results (amendment 22as). The text
below is the approved design, revised per the independent assessment of 22ap before approval; a section on separate
future work is omitted.

## Question
On identical rows, do the two routed-expert launch plans produce different outputs?
- **The plans:** the decode plan (capacity 32, block_m 8, exact width launch, K128/N128 at 10-32 rows) and the
  prefill plan (capacity 1,024, block_m 64, one 1,024-wide launch, K64/N256).
- **What runs where:** 0.9.1 uses the decode plan at 8 in flight (32 rows). 0.7.0 uses the prefill plan at 6-8 in
  flight (40-48 rows).
- **Effects kept separate (never combined into one delta):**
  - **E-plan (within engine):** decode plan vs prefill plan, same engine, same weights, slice and inputs.
  - **E-version (B12x version):** the same plan type and slice through 0.7.0's B12x (`fe054789`, image
    `48e254d9`) and 0.9.1's (`7fcc094e`, image `a9e985c0`). Done only if one weight slice can be planned
    identically in both images; otherwise reported as not tested.
  - **E-partition (EP2 vs TP2):** **not tested.** A one-GPU comparison does not exercise the EP2 replicated-input
    partial-sum all-reduce or the TP2 rank-sliced reduction. Out of scope; stated as such in every report.

## Method
- **Images:** the stock images by digest, `docker run --rm --network none --gpus device=0`. No server, no model
  serving, no download.
- **Weights:** real K3.25 routed-expert tensors for 3 MoE layers (early, middle, late), read from the local snapshot
  `0490d2f7`.
- **Slices:** each engine's own rank-0 slice (0.9.1: TP2 half-intermediate of every expert; 0.7.0: EP2 experts
  0-143, full intermediate). Built through each image's own EXL3 weight-plan code (`exl3.py`
  `_rank_sliced_runtime` → `api.Caps` / `api.plan`), with the served env values (`TRELLIS_MAX_M=32`, block_m 8,
  `PREFILL_CAPACITY=1024`, prefill block_m 64, projection-mixed routes, `bf16_epilogue` as served).
- **Comparison:** each fixture runs through both plans, 3 repeats each, then:
  - the within-plan spread;
  - the per-row bitwise-equal fraction between plans;
  - max abs and relative L2 per row, per layer.
- **Above 32 rows:** the decode plan cannot take more than 32 rows. A 40- or 48-row batch is compared as prefill-plan
  (m rows) against decode-plan chunks of ≤ 32 rows. The source states that tiling a batch does not change results;
  this tests that claim directly.

## Input panel
- **Shapes m:** 1, 8, 16, 24, 32 (the 0.9.1 cell), 40, 48 (the 0.7.0 cell). Padded rows are filled as the runner
  pads them, and also with non-zero garbage, with the garbage rows excluded from the comparison.
- **Activation scale:** rows drawn at ×0.25, ×1 and ×4 the layer's typical RMS (a fixed reference RMS per layer,
  taken from the checkpoint's norm weights, since no live capture exists yet), plus heavy-tailed rows (a few
  channels × 30).
- **Expert concentration:**
  - spread: random distinct top-k per row;
  - concentrated: all rows on the same 8 experts, which fills tiles;
  - mixed.
- **Routing weights:** uniform 1/k; peaked (one weight about 0.9); near-zero tails (1e-4).
- **Seeds:** 2 per cell.
- **Total:** about 7 shapes × 3 scales × 3 concentrations × 3 weightings × 2 seeds × 3 layers, run through both plans
  with 3 repeats. Kernel time is seconds; load, plan and compile dominate.

## Controls
- **Positive control:** in a copy of one fixture, change one element of one row by one bf16 ulp, and separately one
  expert id. The output must change in that row, in both plans. This verifies detection and row attribution.
- **Determinism:** the 3 within-plan repeats must be bit-equal. Otherwise the plan comparison is reported against the
  measured spread.
- **Small-shape configuration check** (assessor): for m ≤ 16, the default configuration (`MAX_M=32`) against the
  switched one (`MAX_M=16`), both through the decode plan's exact launches. Expected bit-equal because this is the
  same launch, not because m is small. A difference is a finding about the switch.

## Outcome rules
- **Reproducible differences** between plans, beyond within-plan spread: characterize them (layers, rows, shapes,
  scales, magnitude) and prioritize a captured-input test (a small one first, not the full Stage 2).
- **Equality** on every fixture: record "equal on the tested fixtures". This does not exclude differences on real
  decode inputs. The MoE line stays **open**, or **deferred** if further testing is postponed for cost. It is never
  "ruled out" on this evidence.
- **Coverage not completed within the ceiling:** INCOMPLETE, not negative.
- **Not affected either way:** the D2 decision. A synthetic equality does not license the D2 experiment or replace
  real-input evidence.

## Cost and limits
- **Engineering (CPU), about 1-1.5 days:**
  - the per-image harness (weight-slice load, plan construction as served, fixture generator, comparison);
  - CPU tests: fixture layout, comparison logic on fabricated outputs, refusals, a DRY_RUN with no GPU;
  - a frozen `SHA256SUMS`.
- **GPU:** one GPU (GPU0), ceiling **0.5 occupied GPU-hours**, counting load, planning, compilation and measurement.
  The estimate is about 0.15-0.35 for two images. The wrapper enforces a wall cap and writes a cost line on every
  exit. One attempt; a retry needs a new approval.
- **Pre-flight:** GPU0 idle and nothing serving.
