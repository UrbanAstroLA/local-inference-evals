#!/usr/bin/env python3
"""GPU-time estimate for a 0.7.0-only rerun (all 22 v070 cells), from run-1 measured rates.
Measured (summary-*.json, log-*.txt of run-20261013T234310Z):
  v070 tiled48: check 4.4-4.6 ms/row; kernel 31 ms per batch-repeat (2 rank calls + 2 merges) after compile
               (cell 2: 1.3 s / 42); build+ref ~0.1 s per batch-family; container start-up 12.9 s.
  B1 fused: kernel 84-125 ms per CALL (fused16 125, fused8 123, fused4 84), vs B1 tiled ~17 ms per call.
Assumptions (not measured on 0.7.0): a 0.7.0 fused call costs the same 85-125 ms (two per batch-repeat);
PI-4 adds 2 merges (~10 ms) per batch-repeat and ~1.5 ms per row."""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
plan = json.load(open(os.path.join(H, "..", "plan.json")))
cells = [c for c in plan["cells"] if c["engine"] == "v070"]
F, R = 7, 3
tot = {}
for lo_hi in ("low", "high"):
    fused_call = 0.085 if lo_hi == "low" else 0.125
    s = 12.9 + 4 * 4.0                                   # start-up + compile per route shape
    for c in cells:
        br = len(c["batches"]) * F * R
        rows = br * c["rows"]
        k = 0.031 if c["route"] == "tiled" else 2 * fused_call
        s += br * (k + 0.010) + rows * (0.0046 + 0.0015) + len(c["batches"]) * F * 0.1
    tot[lo_hi] = s
for k, s in tot.items():
    print(f"{k:4s}: ~{s:5.0f} s in-container, +~25 s docker/pre-flight -> {(s + 25) / 3600:.3f} GPU-h")
print(f"x1.5 pessimistic on high: {(1.5 * tot['high'] + 25) / 3600:.3f} GPU-h")
print("frozen per-container cap 850 s (inner guard ~775 s) is below the high estimate -> PI-5 raises it to 1200 s; "
      "ceiling 0.35 GPU-h keeps the item total at <= 0.1417 + 0.35 = 0.4917 of the 0.5 approved")
