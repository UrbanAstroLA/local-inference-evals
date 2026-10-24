# P4 Stage 0 result: neither construction reproducible; first difference in the independently warmed prefix state (2026-10-24)

**Protocol:** `STAGE0-PROTOCOL.md`. It was written before any Stage 0 GPU step, following the independent assessment
of 22av (section 5).

**Approval:** the operator approved 1.34 GPU-h for one pre-check and one probe attempt; the earlier 7.0 GPU-h ceiling
had been voided. Used: 0.717 GPU-h (`costs-stage0.jsonl`).

## What ran
| Step | Outcome | GPU-h |
|---|---|---|
| Stage 0 pre-check (one GPU, synthetic) | PASS: capture with per-layer digests, 504 rows bit-exact, 220 entry digests equal to host recomputation, negative control | 0.0039 |
| Probe, first launch | stopped by the frozen-hash gate before any container (the approval note had changed a hashed file); hash entry refreshed | 0 |
| Probe (2 GPUs) | COMPLETE: 4 replicates (V1 r1/r2, V2 r1/r2); 32/32 forced requests returned the intended tokens; entry digests on both ranks for all 32; full KV coverage (91 + 88 tensors); 0 hook errors; G3 pass | 0.7133 |

**Construction:**
- arm A only, one batch of 8 windows;
- each replicate warms its prefixes one request at a time, under its own cache salt, with the same prompt ids and
  seed;
- **V1:** windows one at a time. **V2:** the 8 windows as one multi-prompt request;
- the same seed and the same fixed acceptance list for every forced request.

## Gates (stop at the first failure)
| Variant | G-enq (recorded execution identical) | G-entry (window-entry state equal) | G-out |
|---|---|---|---|
| V1 solitary | PASS | **FAIL** | not reached |
| V2 co-batched | PASS | **FAIL** | not reached |

The scheduler execution was identical between repeats in both variants: enqueue order, prefill chunks, start of
verification, bound launches.

## First difference
- **Where:** in the cached prefix state at the start of the forced request (digest point 0, kind "prefill"). This is
  the state built by each replicate's own warm-up prefill.
- **How widespread:** comparing repeat 1 with repeat 2 (`diagnostics/s0map.py`, `diagnostics/s0map.out`):

  | State | Blocks differing |
  |---|---|
  | MLA KV | 1,460 of 1,774 |
  | indexer K cache | 1,460 of 1,694 |
  | KDA state at layer 0 | 80 of 80 (conv) and 44-64 of 80 (recurrent) |
  | most other layers | similar |
  | indexer tail cache | 0 of 88 |

  V1 and V2 show the same counts at point 0, as expected: point 0 precedes the variant-specific steps.
- **Inputs were controlled.** Both warm-ups were solitary prefills of identical prompt token ids, with the same seed
  and chunking. Separate cache salts prevented any reuse between replicates. Prompt lengths and cached-token counts
  matched expectations.

## Reading (per the assessor's interpretation table)
- **Neither construction is reproducible.** Investigate the first differing input, state or operation; do not label
  stock kernels nondeterministic.
- **Established:** two solitary warm-up prefills of identical prompt ids on stock 0.9.1, in this harness, left
  different cached state, already in layer 0's KDA state.
- **Not established:**
  - the cause;
  - the size of the differences (digests show only equality);
  - whether the difference arises in a kernel, in data movement, or in the instrumentation.

  A source scan found no atomic operations in the FLA/KDA, mamba or GLM-5 model kernels.
- **Not certified:** Stage 0 does not certify every persistent-state byte, and the recording changes graphs and timing,
  identically in every replicate.
- **Not supported:** no statement about the tile's effect.

Benchmark-derived records (digests joined to windows, per-request stores) stay under `PRIVATE-runs-stage0/` and are
not published.
