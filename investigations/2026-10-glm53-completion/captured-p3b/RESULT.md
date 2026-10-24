# captured-p3b result: MoE plans and attention counts on captured serving inputs (2026-10-23)

## Runs
Stock 0.9.1 plus observe-only capture hooks, then offline replay on GPU0 in the stock images. Total 0.8031 GPU-h of the
1.0 ceiling (`costs.jsonl`).

| Run | Time | GPU-h | Outcome |
|---|---|---|---|
| Capture attempt 1 | 608 s × 2 GPUs | 0.3378 | failed at start-up (cause below) |
| GPU checks of the fix (graph-capture tests, hook probe) | 10 s × 1 GPU | 0.0028 | fix confirmed |
| Capture attempt 2 | 765 s × 2 GPUs | 0.425 | CAPTURE COMPLETE |
| Replay 1 | 98 s (v091) + 23 s (v070) | 0.0336 | MoE and forced tile complete; attention failed |
| Replay 2, attention only | — | 0.0039 | complete |

**Why attempt 1 failed.** The capture module's poller started at registration, before CUDA-graph capture. Its
pinned allocation, side-stream copies and event syncs invalidated the global-mode capture. Reproduced on one GPU
(`tests/gpu_graph_test.py` G1).

**The fix.** The poller now starts after the first non-dummy forward, as the 22ao guard's did. On GPU the hooks
capture cleanly and the replayed entries are bit-exact (G2). Image `3814fa6c`; the attempt-1 image is kept as
`:attempt1`.

**Why replay 1's attention part failed.** The capture recorded the decode plan's Caps as `None`, because the served
plan object does not expose `.caps`. Replay 2 rebuilt Caps from the captured impl fields exactly as vLLM `_make_plan`
does for this configuration. Validity is decided by the baseline gate below, not by the rebuild.

## Captured
GPQA panel traffic on rank 0, seeds 72301-72316: 8 requests sent before the quotas were met.

- **MoE:** layers 3, 23 and 44.
  - Traffic batches: m = 4 (4), 8 (4), 16 (6), 24 (6), 32 (10).
  - Start-up warm-up batches: m = 32-64 (non-GPQA prompts, labelled "startup").
- **Attention:** MLA layers 3 and 23, decode kernel; 38 entries per layer, 152 rows replayed per layer.
- 0 torn or abandoned entries, 0 capture errors.
- Captured tensors are under `PRIVATE-capture/` (0700), local only.

## MoE: decode plan vs prefill plan on captured rows (v091, own `tp2r0` slice)
**Baseline:** the served output is reproduced bit-equal from the captured inputs on all three layers.

| m | Rows differing | rel L2 (bf16) |
|---|---|---|
| 4 and 8 (traffic) | 0 | — |
| 16, 24, 32 (traffic) | 98.6-100 % | median by layer and m 6.2e-4 to 8.3e-4; max 8.8e-4 |
| 32-64 (start-up) | 88.9-100 % | same range |

- At m = 40 about 89 % of rows differ. The decode side is run as 32 + 8 rows.
- Max abs is up to 0.125 on start-up rows and up to 0.0156 on traffic rows.
- **Agreement with P3 Stage 1 (synthetic):** synthetic medians are 8.3-8.4e-4. Captured rows sit at or slightly
  below that; layer 3 is lowest at about 6.5e-4.
- **v070, own `ep2r0` slice, captured rows:** the same pattern (bit-equal at m ≤ 8; at m ≥ 16, rel L2 median by layer
  and m 5.9e-4 to 8.4e-4, max 9.6e-4).
  - There is no served baseline here: the rows were served by 0.9.1.

## Forced-tile confirmation (v091)
The decode plan was rebuilt through the same `exl3.py` path with the tile forced to (64, 256, 64, 256). The bound launch
was checked to be K64/N256 at m = 16-32.

- Compared with the prefill plan: **bit-equal on all 90 compared rows** (30 per layer: captured m = 16-32 rows and P3's
  synthetic m = 16 and 32 fixtures).
- This establishes tile choice as the cause of the measured difference on the tested inputs. The magnitude
  explanation (the tile's fp32 K-reduction order, amplified by fp16 staging) rests on source analysis and CPU emulation
  (`../moe-plan-p3/SOURCE-MECHANISM.md`), not on a GPU ablation of each staging step (wording corrected in 22au).

## Attention: exact count p + 1 vs the stock inflated count, all else fixed (v091, layers 3 and 23)
**Baseline:** 152 of 152 rows per layer reproduce the served output bit-equal ("anchored"). The positive control is
detected (1.0).

| Regime (per layer) | Anchored rows | Count-only | Through compaction | Reading |
|---|---|---|---|---|
| dense, r ≥ 1 | 57 (r = 1: 16, 2: 15, 3: 26) | 0 differ | 0 differ | EQUAL ON CAPTURED ROWS |
| dense, r = 0 (control) | 52 | 0 | 0 | informational |
| clamped | 40 | 0 | 0 | EQUAL ON CAPTURED ROWS |
| 64-boundary crossing | 3 | 0 | 0 | INCOMPLETE (minimum 8) |

This confirms level 3 on actual served inputs for the dense and clamped regimes (assessor item 6). The boundary regime
is INCOMPLETE: its 3 rows are equal, but that is below the required minimum.

## Not tested
- E-partition (EP2 vs TP2 reductions);
- E-version on captured rows;
- extend-kernel (mixed prefill + verify) attention rows;
- rank 1 (MLA layers ≥ 25);
- any effect on generation.

## Reading
- **MoE:** the plan difference measured on synthetic inputs (P3 Stage 1) is reproduced on real decode rows, at
  similar magnitude, with the tile identified as the cause.
  - This is a reproducible numerical difference between the two engines' routed-expert launches at the batch sizes
    long replies run at.
  - Its behavioural relevance (token choices, completion) is not measured.
- **Attention:** the stock inflated count leaves the output unchanged on captured dense and clamped rows.
  - Together with 22aq and 22ar, this supports dropping the causal-length-only arm of the D2 design for these regimes.
  - Boundary crossing remains covered only by the synthetic level-3 test.
