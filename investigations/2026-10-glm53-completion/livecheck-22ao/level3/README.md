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
   - The masking argument requires the slot-0 record read by these kernels to be finite at consumption. The bounded
     audit and writer census below (22ar) find that it stays all zero on the served path, with two allocator
     assumptions left as inference.
   - Not extended to DCP2 (0.7.0 uses rank-local lengths), untested kernel configurations or plan settings, or
     arbitrary cache contents.
   - It does not bear on the 0.7.0-0.9.1 completion difference beyond removing this candidate.

## Slot-0 audit and writer census (amendment 22ar)
**Question** (as narrowed by the independent assessment): can the exact slot-0 storage read by the tested attention
kernels contain values that invalidate the masking argument at consumption?

- **Audit** ([`P2-SLOT0-AUDIT.md`](P2-SLOT0-AUDIT.md), zero GPU). Slot 0 is bytes [0, 528) of each MLA layer's own
  KV tensor: 11 ranges, 6 layers on rank 0 and 5 on rank 1.
  - The tensors are zero-filled at allocation by the served V2 runner, and block 0 is the shared null block.
  - The same bytes are also the conv state of co-owning KDA layers and token 0 of the slot-shared draft layer. Every
    writer found skips index -1 / block 0, or writes elsewhere in the page.
  - The 1,077,076 tail-stash "null" events from the live check are writes into the indexer tensors' own null block,
    a separate allocation.
- **Census** ([`P2-WRITER-CENSUS.md`](P2-WRITER-CENSUS.md), scripts and output in [`p2-census/`](p2-census/), zero
  GPU). A static taint pass over 736 served-scope files classified all 760 call sites that receive these tensors:

  | Class | Call sites |
  |---|---|
  | guarded, region-excluded, read-only or disjoint | 334 |
  | conditional on an allocator assumption | 11 |
  | not reached in the served configuration | 143 |
  | writes zeros only (the warm-up block zeroer) | 9 |
  | not a data write | 263 |
  | can write non-finite or arbitrary bytes, or undetermined | **0** |

  It found two writers the audit's trace had missed: a copy-on-write page copy, whose destination is never block 0,
  and the warm-up zeroer, which writes zeros.
- **Verdict:** slot 0 stays all zero (E4M3 +0, FP32 scale 0.0) at every attention read on the served 0.9.1 path. That
  is the "slot 0 zero" case measured bit-equal above. No ordering assumption is needed.
- **Residual inference (not closed by source):**
  - the mamba align-mode copy destinations are never block 0;
  - every real draft query position falls in an allocated draft block, and context writes stay in the window.
  - Neither copy kernel checks `dst != 0`.
  - The census is name-based static analysis.
  - Compiled kernels absent from the image (FA2, `reshape_and_cache_flash`) are assumed to write only their outputs.
- **Side finding:** slot 0 is safe because of null-block avoidance in four co-owner writers, three of them without an
  in-kernel check. A regression there would put non-E4M3 bytes (likely NaN codes) into MLA slot 0.
- **Notes:**
  - The audit cites the guard file as revised in 22ap (`b41e8581`). Arm A ran the 22ao guard (`e2dcd964`), which is
    identical up to line 943; later line numbers are 17 lower in that file.
  - Erratum: `SOURCE-READING.md` cited the V1 runner (`gpu_model_runner.py`) for the zero-filled cache. The served
    runner is V2 (`gpu/attn_utils.py:214`), which also uses `torch.zeros`. The conclusion is unchanged; the line is
    annotated there.

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
