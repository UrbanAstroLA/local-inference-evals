#!/usr/bin/env python3
"""E1: served error vs the a-priori error model of each kernel path (E1-ASSESSMENT.md, 'Expected error').
bound_i = (2.5/S) * (a_i + (w_i/2.5) * sum_j a_j) + 6 * 2^-24 * w_i
  a_i = 2^-24 (sigmoid absolute error allowance, tanhf-based sigmoid)            -- both paths
      + 1/2 ulp(fl(s_i + b_i))  (unbiased score re-derived as (s+b)-b)          -- M > 1024 path only
  6 * 2^-24 * w_i: fp32 roundings of the sum (3), the division (1), the product (1), margin (1).
Usage: e1_bound_check.py CAPTURE_DIR CKPT_SNAPSHOT OUT_DIR"""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_emulate import SCALE, ckpt_getter, f32, q  # noqa: E402
cap, ck, out = sys.argv[1:4]
get = ckpt_getter(ck); bias = {}; R = {}
for f in sorted(glob.glob(os.path.join(cap, "rec_r0", "*.pt"))):
    rec = torch.load(f, map_location="cpu"); m = rec["meta"]["m"]; L = rec["meta"]["layer"]
    if L not in bias:
        bias[L] = get(f"model.language_model.layers.{L}.mlp.gate.e_score_correction_bias").float().numpy()
    x = rec["logits_shipped"].float().numpy().astype(f32); ids = rec["topk_ids"].long().numpy()
    w = rec["topk_weights"].float().numpy().astype(np.float64)
    s64 = np.take_along_axis(1 / (1 + np.exp(-x.astype(np.float64))), ids, 1)
    S = s64.sum(1, keepdims=True); ref = s64 / S * SCALE
    b = bias[L][ids].astype(f32)
    a = np.full_like(s64, 2.0 ** -24)
    for reg, extra in (("block-model", 0.0), ("warp-model", 1)):
        aa = a + (0.5 * np.spacing((s64.astype(f32) + b).astype(f32)).astype(np.float64) if extra else 0.0)
        bound = (SCALE / S) * (aa + (w / SCALE) * aa.sum(1, keepdims=True)) + 6 * 2.0 ** -24 * w
        r = np.abs(w - ref) / bound
        key = ("M<=1024" if m <= 1024 else "M>1024") + " vs " + reg
        R.setdefault(key, []).append(r.ravel())
res = {k: {"elems": int(np.concatenate(v).size), "ratio_err_over_bound": q(np.concatenate(v), (0.5, 0.99, 1.0)),
           "frac_exceeding_bound": float((np.concatenate(v) > 1).mean())} for k, v in R.items()}
json.dump(res, open(os.path.join(out, "e1_bound_check.json"), "w"), indent=1); print(json.dumps(res, indent=1))
