#!/usr/bin/env python3
"""E1 alternative-arithmetic candidates (requested in the task) tested against the served weights:
bitwise row/element match and ulp distribution per batch regime. CPU only.
Usage: e1_candidates.py CAPTURE_DIR CKPT_SNAPSHOT OUT_DIR"""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_emulate import ckpt_getter, f32, sigmoid_tanh, ulps, weights, butterfly_sum  # noqa: E402

def seq_sum(u):
    s = u[:, 0].copy()
    for j in range(1, u.shape[1]):
        s = (s + u[:, j]).astype(f32)
    return s

def cand(s, b):
    w = {}
    w["A block: u=s, butterfly, 2.5/S then *u (fp32)"] = weights(s)
    uw = ((s + b).astype(f32) - b).astype(f32)
    w["B warp: u=(s+b)-b, butterfly, 2.5/S then *u (fp32)"] = weights(uw)
    S = butterfly_sum(s)
    w["C u=s, sequential sum"] = (s * (f32(2.5) / seq_sum(s))[:, None]).astype(f32)
    w["D u=s, (u*2.5)/S"] = ((s * f32(2.5)).astype(f32) / S[:, None]).astype(f32)
    w["E u=s, (u/S)*2.5"] = ((s / S[:, None]).astype(f32) * f32(2.5)).astype(f32)
    w["F u=s, fp64 renormalisation (fl32(s*2.5)/(S+1e-20) in fp64)"] = (
        (s * f32(2.5)).astype(f32).astype(np.float64) / (S.astype(np.float64)[:, None] + 1e-20)).astype(f32)
    sh = s.astype(np.float16).astype(f32)
    w["G fp16 sigmoid, else as A"] = weights(sh)
    sc = (f32(2.5) / S).astype(f32)
    sc_bf16 = torch.from_numpy(sc).to(torch.bfloat16).float().numpy()
    w["H bf16 scale factor"] = (s * sc_bf16[:, None]).astype(f32)
    wb = torch.from_numpy(weights(s)).to(torch.bfloat16).float().numpy()
    w["I bf16 output weights"] = wb
    return w

cap, ck, out = sys.argv[1:4]
get = ckpt_getter(ck); bias = {}; R = {}
for f in sorted(glob.glob(os.path.join(cap, "rec_r0", "*.pt"))):
    rec = torch.load(f, map_location="cpu"); m = rec["meta"]["m"]; L = rec["meta"]["layer"]
    if L not in bias:
        bias[L] = get(f"model.language_model.layers.{L}.mlp.gate.e_score_correction_bias").float().numpy()
    x = rec["logits_shipped"].float().numpy().astype(f32); ids = rec["topk_ids"].long().numpy()
    w = rec["topk_weights"].float().numpy().astype(f32)
    s = np.take_along_axis(sigmoid_tanh(x), ids, 1); b = bias[L][ids].astype(f32)
    reg = "M<=1024" if m <= 1024 else "M>1024"
    for k, v in cand(s, b).items():
        c = R.setdefault(reg, {}).setdefault(k, {"rows": 0, "rows_bitwise": 0, "elems": 0, "elems_bitwise": 0, "ulps": []})
        u = ulps(v.astype(f32), w)
        c["rows"] += u.shape[0]; c["rows_bitwise"] += int((u == 0).all(1).sum())
        c["elems"] += u.size; c["elems_bitwise"] += int((u == 0).sum()); c["ulps"].append(u.ravel())
for reg in R:
    for k, c in R[reg].items():
        u = np.concatenate(c.pop("ulps"))
        c["ulp_p50"], c["ulp_p99"], c["ulp_max"] = int(np.quantile(u, 0.5)), int(np.quantile(u, 0.99)), int(u.max())
        c["rows_bitwise_frac"] = round(c["rows_bitwise"] / c["rows"], 4)
json.dump(R, open(os.path.join(out, "e1_candidates.json"), "w"), indent=1)
for reg in R:
    print(reg)
    for k, c in R[reg].items():
        print(f"  {k:62s} rows_bitwise {c['rows_bitwise_frac']:.4f}  ulp p50/p99/max {c['ulp_p50']}/{c['ulp_p99']}/{c['ulp_max']}")
