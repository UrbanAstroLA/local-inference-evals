# Critique of P4 run 1 (sessions A, B, A2 of 2026-10-24) and of its harness and tests

**Status of the run:** the gates passed and the sessions completed (2.53 GPU-h). **The analysis reading is invalid**:
"neither E1 nor E3 shows a reproducible A-vs-B difference". It rests on 10-35 compared positions out of about 24,600
per pair. The underlying cause is that the harness did not produce matched states. **Reading: INCOMPLETE**, not a
negative result.

## 1. Analysis defect: the step key
- **Symptom:** `analysis/out/result.md` excludes about 24,550 of about 24,560 positions per pair as "step key
  mismatch".
- **Cause:** the key's composition part comes from `steprec_*.npz`, whose `tags` map `req_idx -> tag` **at flush
  time**.
  - Request indices are reused across passes.
  - Rows recorded for earlier requests are therefore labelled with the wrong window, or as "unforced".
- **Effect:** the few positions that did match are near-coincidences. They are not a sample of matched states.

## 2. Harness defect: windows do not start from the same state, even within one arm
Direct comparison of the recorded data, ignoring the key (`diagnostics/diag*.py` with outputs `diagnostics/*.out`; CPU only). Session A, repeat 1
vs repeat 2, same window, same forced tokens; the host step facts are equal on about 94 % of positions:

| Measure | A1 vs A2 | A1 vs B1 |
|---|---|---|
| MoE output layer 23, rel L2 median | 0.15 (q90 0.45); 0 % bit-equal | 0.16 |
| MoE output layer 3, rel L2 | 0.006-0.01 from window position 0 | — |
| Forced-token ln P equal | 15 % of positions | — |
| \|Δ ln P(`</think>`)\| median | 0.18 (q90 0.67) | 0.18 |
| Forced-token ln P differs at o = 0 | yes, e.g. -0.03038 vs -0.03058 | — |

The difference exists at the first window position, so the **state at window start differs between repeats**.

**Mechanism observed in the step log** (window R01-close):
- In repeat 1 its first decode step is step 17: 8 requests in flight, a mixed step (2,024 tokens, 2 still
  prefilling).
- In repeat 2 it is step 165: 4 requests in flight, a different mixed packing.
- The previous pass's requests drain one by one (steps 149-157: 4, 3, 2, 1 requests), so passes are not in lock step
  either.

**Why the starts differ.** Each forced request still prefills a 4,353-8,704-token remainder in 2,048-token chunks
(amendment A4), co-scheduled with whatever else is running. Chunk packing, the co-batch at decode start, and therefore
the prefill-built KV, index-K, tail-ring and KDA state differ by repeat.

**Consequence for decode.** Batch composition during each window differs by repeat. At ≤ 2 requests in flight (≤ 8
rows), arm A also runs K64/N256, the same tile as B. Stock D2 tail-ring races may add further run-to-run variation.
They are not separable here.

**Net effect.** Run-to-run state variation is 2-3 orders of magnitude larger than the tile's measured effect at fixed
input (about 8e-4). A-vs-B is indistinguishable from A-vs-A. This run cannot detect a tile effect of the expected size,
in either direction.

## 3. Why the tests and gates did not catch it
- **The gates check the wrong thing for this question.** G1-G3 and G4 check bound launches and the fixed-input
  arithmetic. None checks that the state at window start is reproducible. A determinism gate was missing: "A rep 1 ==
  A rep 2 bit-for-bit on the first positions, else stop before B".
- **The mock sessions** (MOCK server) and the forcing tests run fixed, idealised schedules. They never exercise the
  real scheduler's chunked prefill, staggered completion or co-batching.
- **The analysis tests use fabricated records** whose step composition is consistent by construction. They therefore
  never hit the tag-at-flush defect.
- **The proposal assumed** "the prefix is built by prefill, the same arithmetic in both arms". That is true of
  the tile, but it is not a reproducible state: prefill with cache-hit remainders is packing-dependent on this engine.
  Amendment A4 noted mixed steps at batch start and handled them as a matching problem. It did not treat them as a
  state-reproducibility problem.

## 4. What the data still supports (descriptive)
- **The intervention reached the engine:** G1/G2 pass on both ranks; G4b forced-tile equality.
- **Matched-history decode on stock 0.9.1 under this construction is not reproducible.** Repeat-to-repeat perturbations
  of ln P(`</think>`) are about 0.18 (median) at identical forced tokens. These are perturbations from scheduling and
  packing alone, at the scale relevant to terminal decisions. *Inference, not tested:* natural serving also varies
  batch composition continuously, so perturbations of this kind are likely ever-present in both engines. That bears on
  how a per-layer 8e-4 tile effect could matter, but it does not measure it.
