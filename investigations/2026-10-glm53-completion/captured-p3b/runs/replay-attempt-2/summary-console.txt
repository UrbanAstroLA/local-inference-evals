# captured-p3b replay summary

Capture: CAPTURE COMPLETE 

## MoE v091 (None): not run
## MoE v070 (None): not run
## Attention L3: plan vllm _prepare_b12x_sparse_mla_plan (as served), caps equal None
- rows 152, not anchored 0, P1 [152, 152], P2 detected 1.0
  - dense_r>=1: EQUAL ON CAPTURED ROWS (anchored 57, Bc differ 0, Bf differ 0, r {'1': 16, '2': 15, '3': 26})
  - dense_r0: informational (control rows) (anchored 52, Bc differ 0, Bf differ 0, r {'0': 52})
  - clamped: EQUAL ON CAPTURED ROWS (anchored 40, Bc differ 0, Bf differ 0, r {'0': 4, '1': 12, '2': 12, '3': 12})
  - boundary_crossing: INCOMPLETE (3 anchored rows < 8) (anchored 3, Bc differ 0, Bf differ 0, r {'2': 1, '3': 2})
## Attention L23: plan vllm _prepare_b12x_sparse_mla_plan (as served), caps equal None
- rows 152, not anchored 0, P1 [152, 152], P2 detected 1.0
  - dense_r>=1: EQUAL ON CAPTURED ROWS (anchored 57, Bc differ 0, Bf differ 0, r {'1': 16, '2': 15, '3': 26})
  - dense_r0: informational (control rows) (anchored 52, Bc differ 0, Bf differ 0, r {'0': 52})
  - clamped: EQUAL ON CAPTURED ROWS (anchored 40, Bc differ 0, Bf differ 0, r {'0': 4, '1': 12, '2': 12, '3': 12})
  - boundary_crossing: INCOMPLETE (3 anchored rows < 8) (anchored 3, Bc differ 0, Bf differ 0, r {'2': 1, '3': 2})

Not tested: E-partition (EP2 vs TP2 reductions); E-version on captured rows; extend-kernel (mixed prefill+verify) attention rows; attention on rank 1 (MLA layers >= 25)
