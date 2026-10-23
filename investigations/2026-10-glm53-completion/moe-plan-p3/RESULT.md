# P3 Stage 1 result: real-weight MoE-plan comparison on synthetic inputs (2026-10-23)

**Runs:** GPU0 only, `--network none`, stock images by digest, model snapshot read-only. The total spend was 0.1458 of
0.5 GPU-h.

| Attempt | Mode | Engine | Time | GPU-h | Outcome |
|---|---|---|---|---|---|
| 1 | smoke | v091, layer 3 | 146 s | 0.0406 | complete |
| 2 | full | v091 | 141 s | 0.0392 | complete |
| 2 | full | v070 | 238 s | 0.0661 | complete |

Attempt 2 then ran the version comparison on CPU only. Records: `out/attempt-{1,2}/`, `costs.jsonl`,
`out-*.console`.

**Faithfulness:**
- Each image used its own served EXL3 code: weight slice, weight plan, `_rank_sliced_runtime` under the served env
  (`env_matches_served` True, `exl3.py` hash matches the engine).
- The planned prefill arena reproduces the served log (262.6 MiB on v091).
- Every fixture's plan output matches `Exl3MoEMethod.apply` through the dispatch the engine selects
  (`apply_dispatch_all_match` True on every layer and slice).

## E-plan (within engine): decode plan vs prefill plan, same weights, slice and inputs
Layers 3, 23 and 44; 426 fixtures per layer on the engine's own slice (v091 `tp2r0`, v070 `ep2r0`); 0 errors.

- **Determinism:** 3 repeats per plan, bit-equal in every fixture, for both plans and both engines.
- **m = 1 and 8:** bit-equal between plans (0 of 486 rows differ per layer). At m ≤ 8 the decode launch uses the
  same tile as the prefill launch (fc1/fc2 K64/N256; `launch_map`).
- **m = 16, 24, 32, 48:** essentially every row differs. Per layer, v091 has 9,671-9,672 of 10,638 compared rows
  differing over all m, and v070 9,653-9,667.
  - Decode uses K128/N128 for 10-32 rows; prefill is one 1,024-wide launch at K64/N256 (moe block 48).
- **m = 40:** about 81 % of rows differ. Compared as prefill(40) against decode(32 + 8); the 8-row chunk takes the
  K64/N256 tile.
- **Magnitude**, m ≥ 16 (v091 `tp2r0`; v070 within ±5 %):
  - bf16 output: rel L2 per fixture median 8.4e-4, max 1.01e-3; max abs 0.0039 at ×1 scale, up to 0.25-1.0 on ×4 /
    heavy-tailed rows;
  - fp32 kernel output before the bf16 cast: rel L2 median 2.0e-4, max 2.6e-4 (954 fixtures).
- **Independence from the panel factors:** the magnitude is essentially the same across activation scale, expert
  concentration, routing-weight pattern, m and layer (medians 8.2-8.5e-4).
- **Interpretation (I):** a reproducible, plan-dependent rounding difference, not run-to-run noise.
  - The fp32 difference (2e-4) is larger than reduction-order effects alone would suggest. The plans probably differ
    in where intermediates are rounded.
  - The source mechanism is not established here.

**Controls:**
- **Positive controls:** a one-bf16-ulp input change and a changed expert id change exactly the targeted row in both
  plans, on every layer.
- **Padding:** outputs are independent of pad content.
- **Small-shape configuration check (v091):** for m ≤ 16, `MAX_M=32` against `MAX_M=16` decode launches is bit-equal.
  The `MAX_M=16` runtime was planned with the prefill plan off, to avoid a second arena. This check uses only decode
  launches.

## E-version: same slice and plan through B12x `fe054789` (v070 image) and `7fcc094e` (v091 image)
54 common fixtures per slice. Both slices (`tp2r0`, `ep2r0`) and both plans give **bit-equal outputs on all 1,584
rows**, with 0 input mismatches (`out/attempt-2/eversion.json`). No B12x-version effect on these fixtures.

## Not tested
- **E-partition (EP2 vs TP2):** the cross-GPU reductions are not exercised on one GPU.
- **Inputs:** synthetic inputs and routes, not captured decode rows.
- **Served-path details:** `routed_scaling_factor` not applied (it applies equally to both plans); no CUDA-graph
  replay.

## Reading against the approved outcome rules (PROTOCOL.md; set and approved before the run, not published before it)
**Reproducible differences** between plans beyond within-plan spread, which is zero, at every shape where the tiles
differ (m ≥ 16). They are present in both engines' B12x and both slices.

Per the rules: characterize them (done above) and prioritize a small captured-input test.
- This does not show behavioural relevance.
- It does not explain the completion difference.
- The size of a 8e-4 relative perturbation per MoE layer, compounded over decode steps, on terminal-token decisions
  is unmeasured.
