# P4 Stage 0 result: neither construction reproducible; first difference in the independently warmed prefix state (2026-10-24)

**Corrected in 22ay** after the zero-GPU audit `audit-s0/AUDIT.md` (requested by the independent assessment of 22ax).
The layer-0 KDA statement, the block counts, the checksum wording and the "no atomics" sentence are corrected below; the
22ax text is retained locally.

**Protocol:** `STAGE0-PROTOCOL.md`. It was written before any Stage 0 GPU step, following the independent assessment
of 22av (section 5).

**Approval:** the operator approved 1.34 GPU-h for one pre-check and one probe attempt; the earlier 7.0 GPU-h ceiling
had been voided. Used: 0.717 GPU-h (`costs-stage0.jsonl`).

## What ran
| Step | Outcome | GPU-h |
|---|---|---|
| Stage 0 pre-check (one GPU, synthetic) | PASS: capture with per-layer digests, 504 rows bit-exact, 220 entry digests equal to host recomputation, negative control | 0.0039 |
| Probe, first launch | stopped by the frozen-hash gate before any container (the approval note had changed a hashed file); hash entry refreshed | 0 |
| Probe (2 GPUs) | COMPLETE: 4 replicates (V1 r1/r2, V2 r1/r2); 32/32 forced requests returned the intended tokens; entry digests on both ranks for all 32; every KV-cache tensor covered (91 + 88), for the first 48 logical blocks of each group (22ay: draft-model K/V was therefore not compared in the 6 deep and close windows); 0 hook errors; G3 pass | 0.7133 |

**Construction:**
- arm A only, one batch of 8 windows;
- each replicate warms its prefixes one request at a time, under its own cache salt, with the same prompt ids and
  seed;
- **V1:** windows one at a time. **V2:** the 8 windows as one multi-prompt request;
- the same seed and the same fixed acceptance list for every forced request.

## Gates (stop at the first failure)
| Variant | G-enq (recorded execution identical) | G-entry (window-entry checksums equal: two sums per allocated block, not a byte comparison and not restricted to the state read by the next step) | G-out |
|---|---|---|---|
| V1 solitary | PASS | **FAIL** | not reached |
| V2 co-batched | PASS | **FAIL** | not reached |

The scheduler execution was identical between repeats in both variants: enqueue order, prefill chunks, start of
verification, bound launches.

## First difference
- **Where:** in the cached prefix state at the start of the forced request (digest point 0, kind "prefill"). This is
  the state built by each replicate's own warm-up prefill. For the MLA, indexer and KDA blocks compared, the cached
  block ids match the replicate's own warm-up request. The draft cached-span blocks do not, and are all-zero in both
  replicates. The first difference printed by the analysis follows its enumeration order (window, rank, tensor,
  block), not the order in which the warm-up computed the state.
- **How widespread:** comparing repeat 1 with repeat 2, restricted in 22ay to state that is valid at point 0
  (`audit-s0/recount_point0.out`, `audit-s0/unique_pairs.out`):

  | State at point 0 (per variant) | Blocks differing |
  |---|---|
  | MLA KV, fully cached blocks | 1,460 of 1,606; layer 3 none, every layer from 7 to 43 all |
  | Indexer K, fully cached blocks | 1,460 of 1,606, same pattern. Each block also holds two pool records that the warm-up never wrote |
  | KDA recurrent state at the cached boundary | 496 of 544; layers 0-2 none, every KDA layer from 4 to 44 all |
  | KDA convolution state at the cached boundary | 534-538 of 544. This digest also covers three inactive speculative rows, so it cannot show whether the rows that are read differ |
  | Allocated but not yet written, and scratch | Not part of the cached state; listed separately in the audit |

  The published 1,774 also counted 80 draft-model blocks and 88 blocks beyond the cached span. None of them differed.
  V1 and V2 show the same counts at point 0, as expected: point 0 precedes the variant-specific steps.
- **Inputs were controlled.** Both warm-ups were solitary prefills of identical prompt token ids, with the same seed
  and chunking. Separate cache salts prevented any reuse between replicates. Prompt lengths and cached-token counts
  matched expectations. Physical block ids and their earlier contents differed between replicates by design. The
  separate salt prevents prefix reuse, not allocation differences.

## Reading (per the assessor's interpretation table)
- **Neither construction is reproducible.** Investigate the first differing input, state or operation; do not label
  stock kernels nondeterministic.
- **Established:** two solitary warm-up prefills of identical prompt ids on stock 0.9.1, in this harness, left
  different cached state in blocks that were fully written by each replicate's own warm-up and are read by the next
  step: the MLA KV of every MLA layer from 7 on, and the KDA recurrent state of every KDA layer from 4 on. Layers 0-3
  compared equal by checksum (layer 0-2 KDA state; layer 3 MLA KV and indexer K). The layer-0 differences reported
  earlier lie in scratch columns and in a checksum that includes inactive rows. Layer order is not the order of
  computation, and equal checksums do not prove equal bytes.
- **Not established:**
  - the cause;
  - the size and location of the differences: the checksums show only whether two sums differ, per whole block;
  - whether the difference arises in a kernel, in data movement, or in the instrumentation.

  A source scan found no atomic operations in the FLA/KDA, mamba or GLM-5 model kernels. This does not establish
  that they are deterministic. Different inputs or state, incomplete writes, aliasing, execution order and library
  operations (MoE, GEMM, collectives, top-k) remain possible sources. The scan's stated scope does not name the MoE
  (EXL3/b12x), top-k or collective-communication kernels.
- **Not certified:** Stage 0 does not certify every persistent-state byte, and the recording changes graphs and timing,
  identically in every replicate.
- **Not supported:** no statement about the tile's effect.

## Audit outcome (22ay, zero GPU)
- **Valid state differs; the operation that produces the difference has not been located** (`audit-s0/AUDIT.md`).
- In layer order, the earliest valid difference is the KDA recurrent state of layer 4. Layer 3, the first MoE layer
  and the first sparse-MLA/indexer layer, compared equal by checksum. This is a layer-order observation, not a
  temporal trace.
- The reported first mismatch (rank 0, layer-7 indexer K, logical block 0) was checked against the cache addressing,
  record layout and warm provenance: it is a valid cached-prefix difference, and the same block's MLA KV also differs.
- At point 0 no physical storage was digested through two tensor views.
- The retained checksums cannot resolve which tokens or bytes differ. That needs a small new capture.
- **Lifecycle findings outside the question:**
  - the prefill indexer reads two pool records per cached block that the warm-up never wrote (probably zero in this
    probe; stale bytes in general serving);
  - the drafter's cached-span K/V blocks were not built by the warm-up and are all-zero.

Benchmark-derived records (digests joined to windows, per-request stores) stay under `PRIVATE-runs-stage0/` and are
not published.
