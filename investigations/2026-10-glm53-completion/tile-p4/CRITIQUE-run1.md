# Critique of P4 run 1 (sessions A, B, A2 of 2026-10-24) and of its harness and tests

**Status of the run:** the gates passed and the sessions completed (2.53 GPU-h). Sections 2 and 4 were corrected in
22aw after the independent assessment of 22av. **The analysis reading is invalid**:
"neither E1 nor E3 shows a reproducible A-vs-B difference". It rests on 10-35 compared positions out of about 24,600
per pair. Repeated executions under the harness also disagreed at identical forced token histories (section 2). **Reading: INCOMPLETE**, not a
negative result.

## 1. Analysis defect: the step key
- **Symptom:** `analysis/out/result.md` excludes about 24,550 of about 24,560 positions per pair as "step key
  mismatch".
- **Cause:** the key's composition part comes from `steprec_*.npz`, whose `tags` map `req_idx -> tag` **at flush
  time**.
  - Request indices are reused across passes.
  - Rows recorded for earlier requests are therefore labelled with the wrong window, or as "unforced".
- **Effect:** the few positions that did match are near-coincidences. They are not a sample of matched states.
- **Second defect (identified by the independent assessment of 22av):** `decide()` turned missing evidence into a
  negative reading. Every joint reproducibility family had zero positions, yet it emitted "the tile intervention did not
  change the measured outputs".
  - Its fallback also declares propagation from a count of threshold exceedances (`ab > null`). That is a descriptive
    heuristic, not a calibrated test.
  - Both are being corrected: insufficient coverage must yield INCOMPLETE before any effect reading.

## 2. Repeat disagreement within one arm (wording corrected in 22aw)
Direct comparison of the recorded per-request records, ignoring the defective key (`diagnostics/diag*.py`, outputs
`diagnostics/*.out`; CPU only). Session A, repeat 1 vs repeat 2, same window, same forced tokens. Coarse host step facts
match on 21,906 of 24,576 positions (89.1 %); the figures below are on that subset.

| Measure | A1 vs A2 | A1 vs B1 |
|---|---|---|
| MoE output layer 23, rel L2 median | 0.15 (q90 0.45); 0 % bit-equal | 0.16 |
| MoE output layer 3, rel L2 | 0.006-0.01 from window position 0 | — |
| Forced-token ln P equal | 15 % of positions | — |
| \|Δ ln P(`</think>`)\| median (pre-nucleus) | 0.168 (q90 0.68) | 0.184 |
| Forced-token ln P differs at o = 0 | yes, e.g. -0.03038 vs -0.03058 | — |

**Reading.** Repeated executions under this harness produced different outputs at identical forced token histories.
Their execution schedules differed, for example for window R01-close:
- in repeat 1 its first measured decode step ran with 8 requests in flight, a mixed step with 2 still prefilling;
- in repeat 2 it ran with 4 requests in flight, a different mixed step.

Within each pass, requests started and ended decoding at different steps. (The client joins all request threads
before starting the next repeat, so this does not show that passes overlapped.)

**Not established.** The contributions of cached state, remainder prefill, current-batch arithmetic, D2 and
instrumentation have not been isolated. In particular:
- position-zero disagreement does not show that the incoming persistent state differed, because the position-zero
  computations themselves ran in different batches;
- no window-entry state comparison was retained.

**Consequences.** The A-vs-A and A-vs-B distributions look similar descriptively. They do not establish equivalence or
exclude a systematic tile effect. At ≤ 2 requests in flight (≤ 8 rows), arm A also runs K64/N256; such steps are a
clean local control only with matched inputs and state.

**Construction issues in the client** (identified by the independent assessment of 22av):
- separate HTTP requests released from a thread barrier do not fix arrival order, and the published timestamps show
  different orderings between repeats;
- warm requests for different replies run concurrently, with one cache salt throughout;
- repeat 2 reuses the cache left by repeat 1, and windows share a reply's prefixes;
- sampling seeds differ between repeats.

Equal cache-hit lengths do not establish equal cached bytes or state provenance. Cache contamination or replacement is
a candidate to inspect, not an established finding.

## 3. Why the tests and gates did not catch it
- **The gates check the wrong thing for this question.** G1-G3 and G4 check bound launches and the fixed-input
  arithmetic. None checks that the state at window start is reproducible. A determinism gate was missing: "A rep 1 ==
  A rep 2 bit-for-bit on the first positions, else stop before B".
- **The mock sessions** (MOCK server) and the forcing tests run fixed, idealised schedules. They never exercise the
  real scheduler's chunked prefill, staggered completion or co-batching.
- **The analysis tests use fabricated records** whose step composition is consistent by construction. They therefore
  never hit the tag-at-flush defect.
- **The proposal assumed** "the prefix is built by prefill, the same arithmetic in both arms". That is true of
  the tile, but reproducibility of the resulting window-start state was assumed, not verified.
  Amendment A4 noted mixed steps at batch start and handled them as a matching problem. It did not treat them as a
  state-reproducibility problem.

## 4. What the data still supports (descriptive)
- **The intervention reached the engine:** G1/G2 pass on both ranks; G4b forced-tile equality.
- **Repeated executions under this harness disagree at identical forced token histories:** median
  |Δ ln P(`</think>`)| 0.168 within arm A, pre-nucleus. The cause is not isolated (section 2). Terminal-token eligibility
  and post-filter probability have not been assessed. (Corrected in 22aw: an earlier version attributed the differences
  to "scheduling and packing alone" and compared their size with the tile's fixed-input effect, which is a different
  measurement.)
