# Retained-input inventory for a real-input replay (assessor 22ba, sections D and F.1) — 2026-10-24, zero GPU

Read: `../PRIVATE-capture-p0/session-attempt-2/` (array names, dtypes, shapes and stored positions only; no values),
`../capture-p0/PROTOCOL.md` (what the instrument keeps), the model's `config.json` and weight index (format of the
layer-3 attention weights).

## What exists (capture-p0 attempt 2, rank 0 = layer 3's attention owner)

| Retained | Coverage | Real? |
|---|---|---|
| Per-row sha256 at every layer-3/4 boundary (incl. `L3.attn_in.x`, `L3.topk.ids`, `L3.attn_out.x`) | all 13,056 warm rows, 4 replicates | real |
| Raw rows at every boundary (`raw:`) | 81 positions per replicate: 0-63 and the chunk boundaries 2044, 2045, 4089, 4090, 4351, 4352, 6396, 6397, 8441, 8442, 8703, 8704, 10748, 10749, 12793, 12794, 13055 | real |
| — of which above the selection threshold (p ≥ 2,051) | **15 positions** | real |
| Raw rows of the first 8 differing positions (`diffcur:` / `diffref:`) | 2051-2058, 3 non-reference replicates plus the reference, **only for boundaries that differ** (top-k ids, attention output, MoE output, layer-4 rows) | real |
| `L3.topk.ids` raw (2,048 int32 token ids after the pool expand) | the 81 positions per replicate, plus 2051-2058 | real orderings |
| Layer-3 MLA KV, logical blocks 0-2 (the whole 13,056-token cached span) | each replicate, at point 0 of the capture request | real; written once by the warm |
| Layer-3 indexer K pool cache, blocks 0-2 | each replicate | real |
| Layer-7 MLA KV / indexer K, KDA state slices | each replicate | real |

## What does not exist

- **The indexer score matrix** (`fp8_fp4_mqa_logits` output) for any row.
- **The query side:** MLA q (after q_a_proj, q_a_layernorm, q_b_proj, rope), the indexer q and its head weights.
- **The attention input for rows 2051-2058:** it was equal across replicates, so only its hash was kept (raw rows are
  kept for the fixed 81 positions and for differing rows only).

## Against the assessor's three tests

| Test | Needs | Status |
|---|---|---|
| 1. Real top-k repeatability / ties | actual score rows, candidate ranges, served strides | **Not retained.** Reconstructable only for the 15 selecting positions with a retained attention input, from that input + the layer-3 indexer weights (bf16, not quantised: only routed experts are EXL3 in this checkpoint) + the retained indexer K cache. Validation: the reconstructed top-k set must equal the retained set at every one of the 15 positions in every replicate. 15 rows is a small sample for tie frequency. |
| 2. Real attention replay | actual Q, referenced KV, indices, lengths, config | **Not retained.** Q reconstructable for the same 15 positions from the retained input + bf16 q_a/q_b weights; KV, indices (4 real orderings each) retained. Validation: reproduce the retained `L3.attn_out.x` of each replicate bit-exactly with its own retained order before interpreting any alternative. Bit-exact reproduction needs the served projection, norm and rope path, not a reference re-implementation. |
| 3. Real KV / orderings + synthetic Q | — | **Available now** (hybrid-input sensitivity test). |

## Correction to the progress report of 2026-10-24 (owned)

The report to the assessor said a PRIVATE replay on captured real layer-3 inputs was possible because "both [are]
available from capture-p0". That was wrong: the query side and the score rows were never retained, as
`replay-topk/SELF-CRITIQUE.md` and `capture-p0/PROTOCOL.md` section 8 already said.

## Reading for the next decision (per assessor D)

- Complete real inputs do not exist. The scoped execution-nondeterminism finding stands without a further replay.
- A reconstruction-based replay of 15 rows is possible but is its own validation project (served projection path,
  bit-exact baseline). It is not needed for the MoE tile A/B.
- The open question that real data would decide is whether **real selection ties** occur at the threshold (which an
  order-only patch cannot remove). If that question becomes decision-relevant (i.e. before an order-normalisation
  behavioural arm), the cheapest direct evidence is a capture of the layer-3 score rows for a few chunks
  (2,045 × 1,022 fp32 ≈ 8.4 MB per chunk) in a serving session, priced then; or the 15-row reconstruction above.
