# P4 run 1 result: INCOMPLETE (2026-10-24)

**Reading: INCOMPLETE.** This is not a negative result.
- The sessions ran and every launch gate passed.
- Repeated executions disagreed at identical forced token histories, so this run cannot attribute any A-vs-B
  difference to the tile (cause of the disagreement not isolated).
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
2. **Repeat disagreement.** Repeated executions under this harness produced different outputs at identical forced token
   histories. Repeat 1 vs repeat 2 of the same window in session A, same forced tokens; coarse host step facts match on
   21,906 of 24,576 positions (89.1 %). On that subset:
   - MoE output at layer 23 differs on every compared position (rel L2 median 0.15);
   - layer 3 differs by 0.6-1 % from the first window position;
   - the forced-token ln P is identical on 15 % of positions;
   - |Δ ln P(`</think>`)| has median 0.168. The corresponding A-vs-B median is 0.184.

   These distributions look similar descriptively. They do not establish equivalence, nor exclude a systematic tile
   effect. The execution schedules differed between repeats: remainder prefill was packed into different steps, and
   requests started and ended decoding at different steps within each pass. The contributions of cached state,
   remainder prefill, current-batch arithmetic, D2 and instrumentation have **not** been isolated. Position-zero
   disagreement does not by itself show that the incoming state differed, because the position-zero computations also
   ran in different batches. (Wording corrected in 22aw.)
3. **Gates and tests** verified launches and fixed-input arithmetic, but not reproducibility of the state at window
   start. The CPU mocks did not model the real scheduler's chunked prefill and co-batching.

## What the data supports (descriptive)
- The tile switch acts as designed on both ranks.
- **Repeated executions under this harness disagree at identical forced token histories** (median |Δ ln P(`</think>`)|
  0.168 within arm A). These are pre-nucleus log-probabilities; terminal-token eligibility and post-filter probability
  have not been assessed for these differences. No comparison of their size with the tile's fixed-input MoE effect is
  made: the two are different measurements. (Corrected in 22aw.)
- No statement about the tile's effect on the target distribution is supported by this run.

Benchmark-derived data (histories, logits, hidden states, per-request stores) stays under `PRIVATE-*` and is not
published.
