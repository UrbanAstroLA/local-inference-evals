# capture-p0 result (2026-10-24)

**Pre-registered reading: operation located** (assessor 22ax table row "valid state differs, reproducible local
operation defect found"; that is the table's wording, and no correctness defect is claimed: the top-k's output order
is unspecified, and an unspecified order can be valid while still causing numerical variation in its consumer; note
added in 22bb). Analysis output: `runs/session-attempt-2/cap0-result.md` and `cap0-result.json`; protocol
and amendments A1-A4: `PROTOCOL.md`.

## What ran

| Step | Outcome | GPU-h |
|---|---|---|
| Pre-check, attempt-1 image | PASS | 0.005 |
| Session attempt 1 | Completed, analysis INCOMPLETE: 0 of 6 replicates counted. Two package defects: the warm cached one block too few for the capture's cache hit (8,704 cached tokens, not 13,056), and the step/scheduler log was never flushed for requests without a forced slot. The CPU mock had assumed both. Fixed in amendments A1-A2 | 0.465 |
| Pre-check, attempt-2 image | PASS | 0.005 |
| Session attempt 2 | COMPLETE: 6 of 6 replicates passed every gate, 0 hook errors | 0.422 |

Total 0.897 GPU-h.

## Result (attempt 2)

- **Where the replicates first part:** in all 6 localisation pairs, at the layer-3 indexer top-k output during the
  warm prefill, from token 2,051 (rank 0, which owns layer 3's attention). Every earlier recorded boundary was
  equal, including all of chunk 0 (tokens 0-2,044) and tokens 2,045-2,050.
- **What differs there:** in all 24 sampled rows, the top-k holds the same pool set in a different order. No
  selection differed.
- **Why 2,051:** at position p a row sees (p+1)//4 complete pools of 4 tokens. Real selection (more than 512 pools
  for 512 slots) first applies at p = 2,051; earlier rows take an index-ordered path without top-k.
- **What follows:** layer 3's attention output differs on the same rows, then the MoE output, the layer-7 MLA KV
  (block 0: 2,301 of 4,352 records, first position 2,051; blocks 1-2: all records) and the layer-4 KDA recurrent
  state and live conv rows on both ranks.
- **Controls:** layer-3 MLA KV and layer-2 KDA state byte-identical in all 15 pairs.
- **Layer-0 conv:** live rows 0-2 equal; inactive rows 3-5 differ (the audit's stale-row inference holds).
- **Instrumentation:** the pair without localisation hooks also differs.
- **Lifecycle:** the warm's last write of the layer-4 cached state equals the point-0 bytes in every replicate.

## Limits

- Every pair landed on different physical blocks: allocation dependence and execution nondeterminism are not
  separated by this run.
- The located boundary is where the difference first appears in this serving run. The top-k operation has not been
  replayed in isolation, and the rest of the serving pipeline is not cleared.
- One prompt, one chunk geometry, stock 0.9.1 only. Nothing here measures 0.7.0 or connects this to completion
  behaviour.
