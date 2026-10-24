# capture-p0 result: operation located

- Assessor 22ax table row: **Valid state differs, reproducible local operation defect found**
- Next action: the producing operation is located in serving but not yet shown repeatable: replay that operation with the captured inputs per assessor section 5 before attributing propagation
- Replicates: ['r1', 'r2', 'r3', 'r4', 'r5', 'r6'] (localisation: {'r1': True, 'r2': True, 'r3': False, 'r4': False, 'r5': True, 'r6': True}); counted: ['r1', 'r2', 'r3', 'r4', 'r5', 'r6']

## Gates

| Replicate | Counted | Failures |
|---|---|---|
| r1 | yes | none |
| r2 | yes | none |
| r3 | yes | none |
| r4 | yes | none |
| r5 | yes | none |
| r6 | yes | none |

## Pairs (exact bytes)

| Pair | Same allocation | Valid slices differing | Controls differing | Other differing | First differing boundary |
|---|---|---|---|---|---|
| r1-r2 | False | r0:mla7_b0, r0:mla7_b1, r0:mla7_b2, r0:kda4_rec, r0:kda4_conv_live, r1:kda4_rec, r1:kda4_conv_live | none | r0:kda0_conv_inactive, r0:kda4_conv_inactive, r1:kda0_conv_inactive, r1:kda4_conv_inactive, r0:idx7_b0, r0:idx7_b1, r0:idx7_b2 | L3.topk (chunk 1) |
| r1-r3 | False | r0:mla7_b0, r0:mla7_b1, r0:mla7_b2, r0:kda4_rec, r0:kda4_conv_live, r1:kda4_rec, r1:kda4_conv_live | none | r0:kda0_conv_inactive, r0:kda4_conv_inactive, r1:kda0_conv_inactive, r1:kda4_conv_inactive, r0:idx7_b0, r0:idx7_b1, r0:idx7_b2 | n/a |
| r1-r4 | False | r0:mla7_b0, r0:mla7_b1, r0:mla7_b2, r0:kda4_rec, r0:kda4_conv_live, r1:kda4_rec, r1:kda4_conv_live | none | r0:kda0_conv_inactive, r0:kda4_conv_inactive, r1:kda0_conv_inactive, r1:kda4_conv_inactive, r0:idx7_b0, r0:idx7_b1, r0:idx7_b2 | n/a |
| r1-r5 | False | r0:mla7_b0, r0:mla7_b1, r0:mla7_b2, r0:kda4_rec, r0:kda4_conv_live, r1:kda4_rec, r1:kda4_conv_live | none | r0:kda0_conv_inactive, r0:kda4_conv_inactive, r1:kda0_conv_inactive, r1:kda4_conv_inactive, r0:idx7_b0, r0:idx7_b1, r0:idx7_b2 | L3.topk (chunk 1) |
| r1-r6 | False | r0:mla7_b0, r0:mla7_b1, r0:mla7_b2, r0:kda4_rec, r0:kda4_conv_live, r1:kda4_rec, r1:kda4_conv_live | none | r0:kda0_conv_inactive, r0:kda4_conv_inactive, r1:kda0_conv_inactive, r1:kda4_conv_inactive, r0:idx7_b0, r0:idx7_b1, r0:idx7_b2 | L3.topk (chunk 1) |
| r2-r3 | False | r0:mla7_b0, r0:mla7_b1, r0:mla7_b2, r0:kda4_rec, r0:kda4_conv_live, r1:kda4_rec, r1:kda4_conv_live | none | r0:kda0_conv_inactive, r0:kda4_conv_inactive, r1:kda0_conv_inactive, r1:kda4_conv_inactive, r0:idx7_b0, r0:idx7_b1, r0:idx7_b2 | n/a |
| r2-r4 | False | r0:mla7_b0, r0:mla7_b1, r0:mla7_b2, r0:kda4_rec, r0:kda4_conv_live, r1:kda4_rec, r1:kda4_conv_live | none | r0:kda0_conv_inactive, r0:kda4_conv_inactive, r1:kda0_conv_inactive, r1:kda4_conv_inactive, r0:idx7_b0, r0:idx7_b1, r0:idx7_b2 | n/a |
| r2-r5 | False | r0:mla7_b0, r0:mla7_b1, r0:mla7_b2, r0:kda4_rec, r0:kda4_conv_live, r1:kda4_rec, r1:kda4_conv_live | none | r0:kda0_conv_inactive, r0:kda4_conv_inactive, r1:kda0_conv_inactive, r1:kda4_conv_inactive, r0:idx7_b0, r0:idx7_b1, r0:idx7_b2 | L3.topk (chunk 1) |
| r2-r6 | False | r0:mla7_b0, r0:mla7_b1, r0:mla7_b2, r0:kda4_rec, r0:kda4_conv_live, r1:kda4_rec, r1:kda4_conv_live | none | r0:kda0_conv_inactive, r0:kda4_conv_inactive, r1:kda0_conv_inactive, r1:kda4_conv_inactive, r0:idx7_b0, r0:idx7_b1, r0:idx7_b2 | L3.topk (chunk 1) |
| r3-r4 | False | r0:mla7_b0, r0:mla7_b1, r0:mla7_b2, r0:kda4_rec, r0:kda4_conv_live, r1:kda4_rec, r1:kda4_conv_live | none | r0:kda0_conv_inactive, r0:kda4_conv_inactive, r1:kda0_conv_inactive, r1:kda4_conv_inactive, r0:idx7_b0, r0:idx7_b1, r0:idx7_b2 | n/a |
| r3-r5 | False | r0:mla7_b0, r0:mla7_b1, r0:mla7_b2, r0:kda4_rec, r0:kda4_conv_live, r1:kda4_rec, r1:kda4_conv_live | none | r0:kda0_conv_inactive, r0:kda4_conv_inactive, r1:kda0_conv_inactive, r1:kda4_conv_inactive, r0:idx7_b0, r0:idx7_b1, r0:idx7_b2 | n/a |
| r3-r6 | False | r0:mla7_b0, r0:mla7_b1, r0:mla7_b2, r0:kda4_rec, r0:kda4_conv_live, r1:kda4_rec, r1:kda4_conv_live | none | r0:kda0_conv_inactive, r1:kda0_conv_inactive, r0:idx7_b0, r0:idx7_b1, r0:idx7_b2 | n/a |
| r4-r5 | False | r0:mla7_b0, r0:mla7_b1, r0:mla7_b2, r0:kda4_rec, r0:kda4_conv_live, r1:kda4_rec, r1:kda4_conv_live | none | r0:kda0_conv_inactive, r0:kda4_conv_inactive, r1:kda0_conv_inactive, r1:kda4_conv_inactive, r0:idx7_b0, r0:idx7_b1, r0:idx7_b2 | n/a |
| r4-r6 | False | r0:mla7_b0, r0:mla7_b1, r0:mla7_b2, r0:kda4_rec, r0:kda4_conv_live, r1:kda4_rec, r1:kda4_conv_live | none | r0:kda0_conv_inactive, r0:kda4_conv_inactive, r1:kda0_conv_inactive, r1:kda4_conv_inactive, r0:idx7_b0, r0:idx7_b1, r0:idx7_b2 | n/a |
| r5-r6 | False | r0:mla7_b0, r0:mla7_b1, r0:mla7_b2, r0:kda4_rec, r0:kda4_conv_live, r1:kda4_rec, r1:kda4_conv_live | none | r0:kda0_conv_inactive, r0:kda4_conv_inactive, r1:kda0_conv_inactive, r1:kda4_conv_inactive, r0:idx7_b0, r0:idx7_b1, r0:idx7_b2 | L3.topk (chunk 1) |
- r0:mla7_b0 (first pair): 2301 of 4352 token records differ; first position 2051; by warm chunk {'1': 2039, '2': 262}; latent-only 305, scale-only 0, both 1996; max |decoded diff| 0.2949218973517418
- r0:mla7_b1 (first pair): 4352 of 4352 token records differ; first position 4352; by warm chunk {'3': 2045, '4': 2045, '5': 262}; latent-only 590, scale-only 0, both 3762; max |decoded diff| 0.294642873108387
- r0:mla7_b2 (first pair): 4352 of 4352 token records differ; first position 8704; by warm chunk {'6': 2045, '7': 2045, '8': 262}; latent-only 597, scale-only 0, both 3755; max |decoded diff| 0.29603796475566924
- **conv_rows_test**: `{"pairs_live_rows_0_2_differ": [], "pairs_inactive_rows_3_5_differ": ["r1-r2", "r1-r3", "r1-r4", "r1-r5", "r1-r6", "r2-r3", "r2-r4", "r2-r5", "r2-r6", "r3-r4", "r3-r5", "r3-r6", "r4-r5", "r4-r6", "r5-r6"], "reading": "stale-row inference supported: layer-0 conv differences only in rows 3-5"}`
- **allocation**: `{"same_allocation_pairs": [], "different_allocation_pairs": ["r1-r2", "r1-r3", "r1-r4", "r1-r5", "r1-r6", "r2-r3", "r2-r4", "r2-r5", "r2-r6", "r3-r4", "r3-r5", "r3-r6", "r4-r5", "r4-r6", "r5-r6"], "same_alloc_differ": [], "diff_alloc_differ": ["r1-r2", "r1-r3", "r1-r4", "r1-r5", "r1-r6", "r2-r3", "r2-r4", "r2-r5", "r2-r6", "r3-r4", "r3-r5", "r3-r6", "r4-r5", "r4-r6", "r5-r6"], "reading": "every pair has a different allocation: allocation dependence and execution nondeterminism are not separated by this run"}`
- **instrumentation**: `{"loc_pairs": ["r1-r2", "r1-r5", "r1-r6", "r2-r5", "r2-r6", "r5-r6"], "loc_pairs_differ": ["r1-r2", "r1-r5", "r1-r6", "r2-r5", "r2-r6", "r5-r6"], "noloc_pairs": ["r3-r4"], "noloc_pairs_differ": ["r3-r4"], "reading": "loc and no-loc pairs agree"}`
- **located**: `{"boundary": "L3.topk", "operation": "layer-3 kpool indexer (q/k projections, MQA logits over the gathered pool cache, top-k)", "chunk": 1, "step_pos0": 2045, "hits": [{"rank": 0, "tensor": "ids", "n_tokens": 2039, "first_pos": 2051, "positions": [2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082]}], "topk_sets": {"rows": 24, "same_set_other_order": 24, "different_selection": 0, "reading": "the indexer emits the SAME token set in a different ORDER; the sparse-MLA consumer is order-sensitive (the replay must test the top-k kernel's output order and the attention's dependence on it)"}}`
- **loc_reading**: `{"differing_loc_pairs": ["r1-r2", "r1-r5", "r1-r6", "r2-r5", "r2-r6", "r5-r6"], "first_point": {"r1-r2": "L3.topk", "r1-r5": "L3.topk", "r1-r6": "L3.topk", "r2-r5": "L3.topk", "r2-r6": "L3.topk", "r5-r6": "L3.topk"}}`
- **lifecycle**: `{"r1:r0": {"last_write_equals_point0": true, "same_column": true}, "r1:r1": {"last_write_equals_point0": true, "same_column": true}, "r2:r0": {"last_write_equals_point0": true, "same_column": true}, "r2:r1": {"last_write_equals_point0": true, "same_column": true}, "r5:r0": {"last_write_equals_point0": true, "same_column": true}, "r5:r1": {"last_write_equals_point0": true, "same_column": true}, "r6:r0": {"last_write_equals_point0": true, "same_column": true}, "r6:r1": {"last_write_equals_point0": true, "same_column": true}}`

Limits: equal bytes in one pair do not establish repeatability; a located boundary is where the difference first appears in this serving run, not a replay of that operation and not a clearance of the rest of the serving pipeline (PROTOCOL.md section 8).
