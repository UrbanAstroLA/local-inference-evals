# 22ao live check: result (frozen analysis)

## Arm A
- instrument ok: True (guard errors 0, plants {'a': {'planted': 15521, 'det': 15521}, 'b': {'planted': 12639, 'det': 12639}, 'c': {'planted': 7986, 'det': 7986}, 'd': {'planted': 8899, 'det': 8899}}, attribution 0.9985, async True, blocked waits 0/6084)
- L1 consumed length == p + 1 (signed): **POSITIVE FINDING (672155)**
- L2 sel_causal: **CLEAN WITHIN COVERAGE (95% upper bound 0.0001 per row, N = 48782)**
- L2 sel_foreign: **CLEAN WITHIN COVERAGE (95% upper bound 0.0001 per row, N = 48782)**
- L2 masked_live: **CLEAN WITHIN COVERAGE (95% upper bound 0.0001 per row, N = 48782)**
- L2 sel_len: **POSITIVE FINDING (99495)**
- L2 nsa_pos: **POSITIVE FINDING (99495)**
- coverage: {"dense_after_reject": {"count": 18050, "min": 300, "met": true}, "dense_after_accept": {"count": 12520, "min": 100, "met": true}, "clamped": {"count": 18212, "min": 300, "met": true}, "recycles": {"count": 76, "min": 10, "met": true}, "recycles_shorter": {"count": 76, "min": 5, "met": true}, "transitions": {"count": 20, "min": 20, "met": true}, "full_per_size": {"4": {"count": 119, "min": 20, "met": true}, "8": {"count": 50, "min": 20, "met": true}, "16": {"count": 87, "min": 20, "met": true}, "24": {"count": 221, "min": 20, "met": true}, "32": {"count": 2184, "min": 20, "met": true}}}
- level 1: {"len_checked": 1506571, "len_pos": 672155, "len_neg": 0, "d_verify_rows": {"0": 18410, "1": 6892, "2": 9750, "3": 13906}, "d_by_prev_rej": {"0": {"0": 18230}, "1": {"1": 6892}, "2": {"2": 9750}, "3": {"3": 13906}, "None": {"0": 180}}, "d_on_recycled_rows": {}}
- level 2: {"sel_causal": 0, "sel_foreign": 0, "masked_live": 0, "sel_len": 99495, "nsa_pos": 99495, "nsa_neg": 0, "checked": 1506571, "sel_hole": 1827408, "inact_read": 15114}
- known D2 findings: {"tail_w_stash.mism": 1077362, "tail_w_stash.null": 1077076, "tail_w_stash.foreign": 286, "tail_w_seed.mism": 9080}
- steady / drain steps: 2135 / 340; graph modes {'PIECEWISE': 7, 'FULL': 2989, 'none': 44}; client {'requests': 24, 'errors': 0, 'skipped': 0}

## Arm C
- instrument ok: True (guard errors 0, plants {'a': {'planted': 15554, 'det': 15554}, 'b': {'planted': 12672, 'det': 12672}, 'c': {'planted': 8965, 'det': 8965}, 'd': {'planted': 8965, 'det': 8965}}, attribution 0.9985, async True, blocked waits 0/6120)
- L1 consumed length == p + 1 (signed): **CLEAN WITHIN COVERAGE (95% upper bound 0.0001 per row, N = 48964)**
- L2 sel_causal: **CLEAN WITHIN COVERAGE (95% upper bound 0.0001 per row, N = 48964)**
- L2 sel_foreign: **CLEAN WITHIN COVERAGE (95% upper bound 0.0001 per row, N = 48964)**
- L2 masked_live: **CLEAN WITHIN COVERAGE (95% upper bound 0.0001 per row, N = 48964)**
- L2 sel_len: **CLEAN WITHIN COVERAGE (95% upper bound 0.0001 per row, N = 48964)**
- L2 nsa_pos: **CLEAN WITHIN COVERAGE (95% upper bound 0.0001 per row, N = 48964)**
- coverage: {"dense_after_reject": {"count": 18570, "min": 300, "met": true}, "dense_after_accept": {"count": 12282, "min": 100, "met": true}, "clamped": {"count": 18112, "min": 300, "met": true}, "recycles": {"count": 76, "min": 10, "met": true}, "recycles_shorter": {"count": 76, "min": 5, "met": true}, "transitions": {"count": 20, "min": 20, "met": true}, "full_per_size": {"4": {"count": 146, "min": 20, "met": true}, "8": {"count": 61, "min": 20, "met": true}, "16": {"count": 97, "min": 20, "met": true}, "24": {"count": 166, "min": 20, "met": true}, "32": {"count": 2201, "min": 20, "met": true}}}
- level 1: {"len_checked": 1510575, "len_pos": 0, "len_neg": 0, "d_verify_rows": {"0": 49140}, "d_by_prev_rej": {"0": {"0": 18090}, "1": {"0": 6818}, "2": {"0": 10176}, "3": {"0": 13876}, "None": {"0": 180}}, "d_on_recycled_rows": {}}
- level 2: {"sel_causal": 0, "sel_foreign": 0, "masked_live": 0, "sel_len": 0, "nsa_pos": 0, "nsa_neg": 0, "checked": 1510575, "sel_hole": 1601259, "inact_read": 11022}
- known D2 findings: {}
- steady / drain steps: 2175 / 295; graph modes {'PIECEWISE': 7, 'FULL': 3007, 'none': 44}; client {'requests': 24, 'errors': 0, 'skipped': 0}

