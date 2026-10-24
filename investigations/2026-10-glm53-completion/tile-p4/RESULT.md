# P4 run 1 result: INCOMPLETE (2026-10-24)

**Reading: INCOMPLETE.** This is not a negative result.
- The sessions ran and every launch gate passed.
- The harness did not produce matched states, so it could not detect a tile effect of the expected size.
- The automated analysis reading ("neither E1 nor E3 shows a reproducible A-vs-B difference",
  `analysis/out/result.md`) is **invalid**: it rests on 10-35 compared positions out of about 24,600 per pair.
- Details: `CRITIQUE-run1.md`. Evidence: `diagnostics/`.

## What ran
| Step | Outcome | GPU-h |
|---|---|---|
| Pre-check (single GPU, synthetic) | PASS on attempt 4, after fixes in the capture module and the test harness (attempts 1-3 recorded) | ~0.01 |
| G4 (single GPU, captured stock rows) | PASS: switch off reproduces stock output 44/44 per layer. G4b: switch on, decode = prefill on 66 entries | ~0.01 |
| Session A (default tile) | COMPLETE: G3 pass; 6/6 batches; 96/96 forced requests | 0.8228 |
| Session B (forced K64/N256) | COMPLETE: G1 + G2 pass on both ranks; 6/6; 96/96 | 0.8511 |
| Session A' (default tile) | COMPLETE: session-control gate pass; 6/6; 96/96 | 0.8256 |

Total 2.53 GPU-hours (`costs.jsonl`).

**The intervention reached the engine as specified.** Arm B bound K64/N256 for every 10-32-row decode launch on both
ranks and was identical to arm A everywhere else.

## Why the reading is invalid
1. **Analysis defect.** The step key labelled each step's forced requests through a map read at flush time, while
   request indices are reused. About 99.9 % of positions were excluded as "step key mismatch".
2. **Harness defect.** Windows did not start from the same state, even within one arm. Repeat 1 vs repeat 2 of the same
   window in session A, with the same forced tokens:
   - MoE output at layer 23 differs on 100 % of positions (rel L2 median 0.15);
   - layer 3 differs by 0.6-1 % from the first window position;
   - the forced-token ln P is identical on only 15 % of positions;
   - |Δ ln P(`</think>`)| has median 0.18.

   A vs B shows the same magnitudes. The step log shows why: in-batch remainder prefill (4,353-8,704 tokens in
   2,048-token chunks) was packed differently in each repeat, decode started with different co-batches, and passes
   were not in lock step.
3. **Gates and tests** verified launches and fixed-input arithmetic, but not reproducibility of the state at window
   start. The CPU mocks did not model the real scheduler's chunked prefill and co-batching.

## What the data supports (descriptive)
- The tile switch acts as designed on both ranks.
- **Matched-history forced decode built this way is not reproducible on stock 0.9.1.** Scheduling and packing alone
  perturb ln P(`</think>`) by about 0.18 (median) at identical tokens. That is 2-3 orders of magnitude above the
  tile's effect at fixed input (MoE rel L2 about 8e-4).
- No statement about the tile's effect on the target distribution is supported by this run.

Benchmark-derived data (histories, logits, hidden states, per-request stores) stays under `PRIVATE-*` and is not
published.
