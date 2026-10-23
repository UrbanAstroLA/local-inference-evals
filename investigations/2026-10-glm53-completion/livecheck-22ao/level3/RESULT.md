# Level 3 result (22ap): inflated consumed count vs p + 1, stock kernels, synthetic inputs

Run 2026-10-23T18:17Z on GPU0 (RTX PRO 6000 Blackwell Max-Q, SM 12.0), stock 0.9.1 base image `a9e985c0`,
`--network none`, no model. 8 s x 1 GPU = 0.0022 GPU-h. Script `kernel_eq.py`, wrapper `run_level3.sh`, output
`result.json`, `run.log`.

Plan as served: decode `num_splits 32`, `chunks_per_split 1`, `per_token_len True`, topk 2048, 64 heads, 64 rows
(`result.json` info.last_decode_plan). 64 rows with p + 1 valid slots front-packed and -1 after; p + 1 spans the
64-boundaries and 2043-2047. B uses nsa = min(p + 1 + r, 2048) and the stock cache_lengths p + 1 + r. Rows crossing
a 64 boundary under inflation: 10 / 25 / 33 for r = 1 / 2 / 3.

| Check | Result |
|---|---|
| D0 determinism, decode and extend (A twice) | bit-equal |
| E, A vs B, decode and extend, r = 1, 2, 3, slot 0 zero (served: null block, zero-initialized cache) | **bit-equal on all 64 rows, output and LSE** (12 comparisons) |
| E, same with slot 0 holding a real record | **bit-equal on all 64 rows** |
| P positive control: one live non-causal slot planted at column p + 1 in B | differs on 64 / 64 rows, both kernels (max abs 0.67) |
| N probe, slot 0 = NaN bytes (diagnostic) | decode: A and B NaN on the same 54 rows, bit-equal. extend: A NaN on 54 rows, B on 64; the 10 extra rows are the r = 1 boundary-crossing rows |

Reading:
- The inflated length on stock 0.9.1 does not change the attention output in either kernel, with slot 0 zero or
  holding any finite record: the three facts left open by the source reading (ex2(0) = 1 exactly, an all-zero-P MMA
  adds 0, slot-0 finiteness) hold in this configuration.
- The only dependence found is through slot 0's content when it is non-finite, and only in the extend kernel (used
  for verify rows in steps mixed with a prefill): inflation that crosses a 64 boundary adds one all-masked tile that
  multiplies slot 0's V by P = 0. Stock already performs the same product on every row whose last tile is partial
  (54 / 64 here), so a non-finite slot 0 would corrupt baseline rows too. The condition the masking argument needs is
  that the slot-0 record read by these kernels is finite at consumption (a real finite record also gives bit-equal
  outputs); a bounded audit of that storage is pending.
- Limits: synthetic inputs and one random seed; plans prepared with the library's default configuration (decode
  plan matches the served split plan); extend plan at 64 rows rather than the served max_batched rows; DCP1 / GLM_NEXT
  / topk 2048 route only.
