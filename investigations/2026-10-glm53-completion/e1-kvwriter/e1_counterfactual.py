#!/usr/bin/env python3
"""E1 same-token counterfactual on the M > 1024 rows: emulated block path (u = s) vs emulated warp path
(u = (s+b)-b), both with the tanh-sequence sigmoid; each vs fp64, and the paired path difference.
Usage: e1_counterfactual.py CAPTURE_DIR CKPT_SNAPSHOT OUT_DIR"""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_emulate import SCALE, ckpt_getter, f32, q, sigmoid_tanh, weights  # noqa: E402

cap, ck, out = sys.argv[1:4]
get = ckpt_getter(ck); bias = {}
A = {"block_vs_fp64_abs": [], "block_vs_fp64_rel": [], "warp_vs_fp64_abs": [], "warp_vs_fp64_rel": [],
     "paired_abs": [], "paired_rel": [], "paired_row_l1": [], "served_vs_fp64_abs": []}
for f in sorted(glob.glob(os.path.join(cap, "rec_r0", "*.pt"))):
    rec = torch.load(f, map_location="cpu"); m = rec["meta"]["m"]; L = rec["meta"]["layer"]
    if m <= 1024:
        continue
    if L not in bias:
        bias[L] = get(f"model.language_model.layers.{L}.mlp.gate.e_score_correction_bias").float().numpy()
    x = rec["logits_shipped"].float().numpy().astype(f32); ids = rec["topk_ids"].long().numpy()
    w = rec["topk_weights"].float().numpy().astype(np.float64)
    b = bias[L][ids].astype(f32)
    s32 = np.take_along_axis(sigmoid_tanh(x), ids, 1)
    wb = weights(s32).astype(np.float64)
    ww = weights(((s32 + b).astype(f32) - b).astype(f32)).astype(np.float64)
    s64 = np.take_along_axis(1 / (1 + np.exp(-x.astype(np.float64))), ids, 1)
    ref = s64 / s64.sum(1, keepdims=True) * SCALE
    for k, v in (("block", wb), ("warp", ww)):
        A[f"{k}_vs_fp64_abs"].append(np.abs(v - ref).ravel()); A[f"{k}_vs_fp64_rel"].append((np.abs(v - ref) / ref).ravel())
    A["paired_abs"].append(np.abs(ww - wb).ravel()); A["paired_rel"].append((np.abs(ww - wb) / wb).ravel())
    A["paired_row_l1"].append(np.abs(ww - wb).sum(1)); A["served_vs_fp64_abs"].append(np.abs(w - ref).ravel())
res = {k: q(np.concatenate(v), (0.5, 0.9, 0.99, 0.999, 1.0)) for k, v in A.items()}
res["rows"] = int(np.concatenate(A["paired_row_l1"]).size)
res["note"] = ("Same tokens (the 91,200 M>1024 rows). block = what the M<=1024 kernel would return, warp = what the "
               "M>1024 kernel returns (emulated; 98% of rows bitwise equal to served). paired_* = |warp - block|.")
json.dump(res, open(os.path.join(out, "e1_counterfactual.json"), "w"), indent=1); print(json.dumps(res, indent=1))
