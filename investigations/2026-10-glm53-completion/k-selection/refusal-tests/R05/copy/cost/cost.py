#!/usr/bin/env python3
"""K cost estimate (stdlib only) from plan.json and stated per-operation timing expectations.

Assumptions (sources):
  start-up per container: image start, imports, CUDA context, bank load: 45 s. The stage-2 replay
      selftest (same B12x kernels, fresh container, JIT included) took 9.8-12.2 s end to end
      (phase2-stage2/costs.jsonl); CPU import of the driver here took 11 s (cpu-tests). 45 s is ~4x.
  CuTe/Triton compile: 20 s per distinct (route, rows) shape plus 20 s for each image's Triton consumer
      kernels (stage-2: the whole v091 replay incl. compile of 3 route variants ran in 89 s).
  selection kernel per call: tiled scores the full page-table capacity (active_width is None, SOURCE-
      REVIEW section 1), so its cost is length-independent: <= 1.5 ms (B12x docstring: r64@131K
      tiled 1,043 us; r1@131K fused 41 us). Fused <= 0.5 ms. +3 ms launch/plan/bind/poison per call.
  merge (0.7.0): run_row_topk over 1,024 candidates + 3 Triton kernels: 1 ms per perspective.
  fp64 reference: rows*32*P*128*2 flops, twice (score and bound), at 0.5 TFLOP/s effective fp64
      (RTX PRO 6000 fp64 is 1/64 of fp32; ~1.5 TFLOP/s peak, einsum taken at a third).
  input build: 25 ms per batch (CPU-side exact-family construction, randperm over P <= 83k, packing).
  checks (sync-light checkers, one host transfer per check call): per row x repeat ~120 small kernel
      launches at 6 us + 5 host transfers at 30 us for B1 (C1-C3, first-511), ~170 launches + 6 transfers
      for 0.7.0 (two ranks + merged); per batch x repeat one vectorised C4 (20 launches + 1 transfer).
Pessimistic case: every term x3.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
plan = json.load(open(os.path.join(os.path.dirname(HERE), "plan.json")))
FAM = len(plan["meta"]["FAMILIES"])
REP = plan["meta"]["REPEATS"]
STARTUP, COMPILE_SHAPE, COMPILE_CONSUMER = 45.0, 20.0, 20.0
K_TILED, K_FUSED, K_OVER, MERGE = 1.5e-3, 0.5e-3, 3e-3, 1e-3
FP64 = 0.5e12
BUILD = 25e-3
ROW_COST = {"b1": 120 * 6e-6 + 5 * 30e-6, "v070": 170 * 6e-6 + 6 * 30e-6}
BATCH_CHECK = 20 * 6e-6 + 30e-6

out = {}
lines = []
for eng in ("b1", "v070"):
    e = plan["meta"]["engines"][eng]
    cells = [c for c in plan["cells"] if c["engine"] == eng]
    shapes = {(c["route"], c["rows"]) for c in cells}
    t = dict(startup=STARTUP, compile=COMPILE_SHAPE * len(shapes) + COMPILE_CONSUMER, build=0.0, ref=0.0,
             kernel=0.0, check=0.0)
    for c in cells:
        rows = c["rows"]
        for batch in c["batches"]:
            P = [L // 4 for L in batch]
            t["build"] += FAM * BUILD
            t["ref"] += FAM * sum((e["K"] + 1) * 32 * p * 128 * 2 * 2 for p in P) / FP64
            per_call = (K_TILED if c["route"] == "tiled" else K_FUSED) + K_OVER
            ranks = e["dcp"]
            t["kernel"] += FAM * REP * (ranks * per_call + (2 * MERGE if ranks == 2 else 0))
            t["check"] += FAM * REP * (rows * ROW_COST[eng] + BATCH_CHECK)
    tot = sum(t.values())
    out[eng] = dict(seconds={k: round(v, 1) for k, v in t.items()}, total_s=round(tot, 1),
                    pessimistic_s=round(3 * tot, 1), cells=len(cells),
                    batches=sum(len(c["batches"]) for c in cells) * FAM)
    lines.append(f"{eng}: {out[eng]['batches']} batches x {REP} repeats; estimate {tot:.0f} s "
                 f"({', '.join(f'{k} {v:.0f}' for k, v in t.items())}); pessimistic x3 {3 * tot:.0f} s")
est = sum(v["total_s"] for v in out.values()) + 30          # + host-side runner, pre-flight, clean-up
pes = sum(v["pessimistic_s"] for v in out.values()) + 60
out["total"] = dict(estimate_s=round(est), estimate_gpu_h=round(est / 3600, 3),
                    pessimistic_s=round(pes), pessimistic_gpu_h=round(pes / 3600, 3),
                    ceiling_gpu_h=0.5, per_container_caps_s=dict(b1=850, v070=850), wall_cap_s=1800)
lines.append(f"total (1 GPU): estimate {est:.0f} s = {est / 3600:.3f} GPU-h; pessimistic {pes:.0f} s = "
             f"{pes / 3600:.3f} GPU-h; ceiling 0.5 GPU-h (wall cap 1,800 s; containers 850 + 850 s)")
for e in ("b1", "v070"):
    cap = out["total"]["per_container_caps_s"][e] - 75
    if out[e]["pessimistic_s"] > cap:
        lines.append(f"  note: {e} pessimistic {out[e]['pessimistic_s']:.0f} s exceeds its inner time guard {cap} s; "
                     "cells run in priority order (tiled first), the rest would be NOT-RUN, not silently dropped")
json.dump(out, open(os.path.join(HERE, "cost.json"), "w"), indent=1)
open(os.path.join(HERE, "cost.out"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
