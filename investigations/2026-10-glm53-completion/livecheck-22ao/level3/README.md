# Level 3: does the inflated consumed count change the attention output? (amendment 22aq)

**Status.** A subsequent investigation, run after the 22ao live check had been read. It is **not** part of the frozen
22ao measurement and was not pre-registered. Its design, harnesses and outputs are all in this folder.

**Question.** On stock 0.9.1 (DCP1, GLM_NEXT, top-k 2,048), the live check found that speculative verify rows consume a
per-token causal length L = p + 1 + r, where r (0-3) is the previous step's rejected-draft count, and that the extra
in-count columns `[p + 1, nsa)` were empty (-1). Does the attention output differ from the output with `nsa = p + 1`?

## Three separate conclusions

1. **Compaction (consumed inputs).** In the tested layouts, `_compact_dropped_selection` and
   `_mask_page_table_after_nsa_len` produce consumed tables whose attended set per 64-slot chunk is identical for
   `nsa = p + 1` and `nsa = p + 1 + r`.
   - Method: the stock Triton kernels on CPU under `TRITON_INTERPRET=1` (`e6_compaction_harness.sh`,
     `e6_compaction.out`): indexer seq 1-2,047 × r = 1-3, sorted and shuffled pools, 12,282 cases, 0 mismatches.
   - Not covered: the stock indexer's own tail anchoring (D2) and non-front-packed pool outputs. The live level-2
     data covers these for the served rows only.
2. **Kernels (outputs).** For the tested alternative counts, the stock decode and extend kernels produce bit-identical
   outputs.
   - Method: synthetic inputs on one GPU (`kernel_eq.py`, `run_level3.sh`, `result.json`, `run.log`), 0.0022
     GPU-hours.
   - Plan: decode 32 splits, 1 chunk per split, per-row lengths, as served (`result.json` → `last_decode_plan`).
   - Rows: 64, with p + 1 at and around each 64-slot boundary and at 2,043-2,047. Under inflation 10 / 25 / 33 rows
     cross a boundary (r = 1 / 2 / 3).
   - **A vs B bit-equal** on output and LSE, both kernels, r = 1-3, with cache slot 0 zero and with slot 0 holding a
     real record (12 comparisons: 2 kernels × r = 1-3 × 2 slot-0 contents). Repeats are bit-equal.
   - **Positive control:** one live non-causal slot planted at column p + 1 changes 64 / 64 rows in both kernels.
   - **Diagnostic** (slot 0 set to NaN bytes):
     - decode: A and B are NaN on the same 54 rows and bit-equal;
     - extend: B is NaN on 10 more rows, the r = 1 boundary-crossing rows.
     - Inflation can therefore matter only through slot 0's content. Stock already multiplies slot 0 by P = 0 on
       every row whose last tile is partial.
3. **Mechanism.** The causal-length discrepancy is **ruled down as a computational contributor in the tested 0.9.1
   configuration and under the verified masking/compaction conditions.**
   - The masking argument requires the slot-0 record read by these kernels to be finite at consumption. A bounded
     audit of that storage is pending.
   - Not extended to DCP2 (0.7.0 uses rank-local lengths), untested kernel configurations or plan settings, or
     arbitrary cache contents.
   - It does not bear on the 0.7.0-0.9.1 completion difference beyond removing this candidate.

## Source argument
[`SOURCE-READING.md`](SOURCE-READING.md) (zero GPU) gives the call chain and the cited file hashes. In summary:
- the kernels receive only the consumed count;
- empty in-count slots and slots past the count are gathered and masked identically;
- an extra all-masked decode split gets merge weight +0 and is skipped;
- an extra all-masked extend tile adds P = 0;
- the split count, kernel variant, workspace and merge order do not depend on L.

The GPU test settles the hardware facts the reading inferred (`ex2` behaviour, a zero-P MMA adding exactly 0) in the
tested configuration.

## Limits
- Synthetic inputs and one seed.
- Plans prepared with the library's default configuration. The decode split plan matches the served one; the extend
  plan was sized at 64 rows.
- One GPU model: RTX PRO 6000 Blackwell Max-Q, SM 12.0.
- Stock 0.9.1 base image `a9e985c0` only.

## Note on the 22ap bounds
The 95% "rule-of-three" figures reported for the live check (3/N per row) assume independent rows. Rows from the
same steps, requests and ranks are dependent, so the figure is nominal. It is not a demonstrated population bound,
and not the probability that an engine defect remains. The zero-violation counts and the coverage stand as reported.
