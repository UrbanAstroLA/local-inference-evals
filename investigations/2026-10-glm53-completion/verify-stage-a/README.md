# Stage A verification receipts (amendments 13-14)

D3 router precision, D4 routed SwiGLU clamp, D5 workspace locking on tpurtell 0.9.1 K3.25, shipped graph mode. Complete:
0.6929 of 2.0 occupied GPU-hours. `PROPOSAL.md` is the frozen plan with the stage A results appended (offline attempt 4
last); `costs.jsonl` the per-step cost lines.

**Result.** D3: controls (a)-(d) pass on all 1,593 records (103,608 token × layer rows). Control (c) invokes the router
as serving does (one call on the serving batch rows), a change adopted in amendment 14 before any D3 comparison was
computed; its bitwise criterion is unchanged. Shipped ("bf16-result GEMM cast to fp32") vs corrected ("fp32-output
GEMM"): top-8 expert membership differs on 3,262 rows (3.15%; 95% bootstrap over 176 requests 3.01-3.29%, over 8
histories 2.90-3.50%), routing weights differ on every row; by layer 1.2-6.6%, by execution type 3.0-3.2%. The corrected
computation lies within the stated tolerance τ_d of an fp64 reference on all 29,839,104 logits. FULL-graph decode is
covered by source inspection only. D5: not exercised in this workload. D4: the frozen inputs are sufficient for a later
comparison; the effect of the limit was not measured here. E1 (batch-size-dependent routing weights) is recorded
separately. Completion effect of every item: untested here.

- `smoke-a2/`: smoke checks (all passed on attempt 2) and B12x's swiglu_limit kernel tests.
- `server-a1/`: server session summary, configuration comparisons against the historical 0.9.1 run and the gate run,
  throughput, D5 summary. Raw capture records (3,112 per rank) are local; hashes in `RAW-SHA256SUMS-stage-a.txt`.
- `offline-a2/`, `offline-a3/`: D3 control results (stopped at control (c)), the per-record control (c) classification,
  D4 inventory, D5 summary.
- `offline-a4/` (amendment 14; the final attempt): `d3_summary.json` (controls, effect, tolerances, normalisation,
  E1, clustered bootstrap), `d3_by_layer.csv`, `d3_by_type.csv`, `d3_by_layer_type.csv`, the per-record control (c)
  classification (`d3_control_c_records.jsonl`, all 1,593 exact), `d4_inventory.json`, `d5_summary.json`, `d3.log`.
- `code/`: the capture module, offline analysis (`d3_router_compare.py` as run in attempt 3,
  `d3_router_compare.a4.py` as run in attempt 4, `d3-amendment14-servingM.diff` between them, `test_d3_offline.py`), smoke
  test, runner, the smoke-1 fix, the control (c) diagnostics diff and the earlier proposed serving-shaped invocation.
  Local paths and a local name prefix in identifiers are replaced in these public copies, so they are not byte-identical
  to the code that ran ([`../PUBLIC-COPIES-14.txt`](../PUBLIC-COPIES-14.txt) lists both hashes where a file added on
  2026-10-12 differs).
