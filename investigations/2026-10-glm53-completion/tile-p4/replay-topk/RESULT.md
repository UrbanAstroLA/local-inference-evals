# replay-topk result (2026-10-24; attempt 1; synthetic inputs)

Protocol: `PROTOCOL.md`. Analysis output: `runs/attempt-1/replay-result.md` and `replay-result.json`. One GPU,
0.0144 GPU-h (v0.9.1 25 s, v0.7.0 27 s). Every input is synthetic; served shapes, strides, plans and call paths.

## R1: the layer-3 prefill top-k (`_C.top_k_per_row_prefill`; binary identical in both images)

| Regime | Order varies between calls | Set varies | Reading |
|---|---|---|---|
| K: logits from the image's own `fp8_fp4_mqa_logits` (v0.9.1; served stride [1280, 1]) | every selecting row, same allocation and relocated (24 + 24 calls; 23-24 calls differ from the first) | never; no ties at the threshold | A-ORDER |
| C: continuous values | every selecting row (both images) | never | A-ORDER |
| T: values rounded to force ties (471 rows tied at the threshold) | every selecting row (both images) | 171 rows (same allocation), 199 (relocated) | A-SET |

- The variation occurs **with identical inputs in the same memory**: in the assessor's section-5 terms, execution
  nondeterminism, not allocation dependence.
- Output pattern: the last column is always the lowest-scoring selected pool (so the expand that drops it keeps a
  stable set when there are no ties); the rest is in no fixed order.
- Rows with 512 or fewer pools take the index-ordered path and never vary.
- v0.7.0: run on a contiguous stride (its served logits producer was not available in the replay), and in its served
  DCP=2 configuration this kernel takes the index-ordered path for this chunk; its actual selection happens in the
  DCP merge, which was not replayed.

## R2: the b12x sparse-MLA prefill kernel, both images

- **Deterministic for a fixed order:** 8 repeats bit-identical; scratch and unwritten-cache poisoning and three
  relocations leave the output unchanged.
- **Order-sensitive:** every permutation of the same index set (ascending, descending, three random, and the top-k's
  own pattern) changes the output on every selecting row; rows on the index-ordered path never change.
  - v0.9.1: max |Δ| 0.0039 (flat scores) and 0.0156 (peaked); median relative L2 1.1e-3.
  - v0.7.0 (DCP rank-0 kernel call, without the cross-rank combine): max |Δ| 0.0078 and 0.016-0.031.
- For scale: each order's output differs from an fp32 reference by a relative L2 of about 3.2e-2; the order effect is
  about 30 times smaller, and no order is closer to the reference than another.

## R3: composition (v0.9.1)

Feeding R1's regime-K top-k outputs through the served index path into the kernel: 8 calls; in each, about 2,038 rows
receive a different order and the same set, and the attention output differs on those rows (median relative L2
9.4e-4). No output row differs without a top-k order difference.

## Reading

- The layer-3 divergence located by capture-p0 is reproduced in isolation: the prefill top-k returns its selection in
  an order that varies from call to call with identical inputs and memory, and the sparse-MLA prefill turns each order
  into a slightly different result.
- With ties at the selection threshold the selected set itself can vary (synthetic stress regime; the 24 sampled
  serving rows showed no set difference).
- The attention kernel of both versions is order-sensitive in the same way; the top-k binary is shared.

## Limits

- Synthetic inputs; real logits need the model weights. The tie structure of real logits is not measured.
- One idle GPU: no contention.
- v0.7.0's served selection (the DCP merge) is not replayed, and v0.7.0 has not been run end to end.
- Nothing here measures an effect on completion behaviour.
