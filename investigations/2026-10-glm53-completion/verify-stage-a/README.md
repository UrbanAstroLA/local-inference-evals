# Stage A verification receipts (amendments 13-13b), in progress

D3 router precision, D4 routed SwiGLU clamp, D5 workspace locking on tpurtell 0.9.1 K3.25, shipped graph mode.
`PROPOSAL.md` is the frozen plan with the stage A results appended; `costs.jsonl` the per-step cost lines.

- `smoke-a2/`: smoke checks (all passed on attempt 2) and B12x's swiglu_limit kernel tests.
- `server-a1/`: server session summary, configuration comparisons against the historical 0.9.1 run and the gate run,
  throughput, D5 summary. Raw capture records (3,112 per rank) are local; hashes in `RAW-SHA256SUMS-stage-a.txt`.
- `offline-a2/`, `offline-a3/`: D3 control results (stopped at control (c)), the per-record control (c) classification,
  D4 inventory, D5 summary.
- `code/`: the capture module, offline analysis, smoke test, runner, the smoke-1 fix, the control (c) diagnostics diff and
  the proposed (not applied) serving-shaped router invocation. Local paths and a local name prefix in identifiers are
  replaced in these public copies, so they are not byte-identical to the code that ran.
