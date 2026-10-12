#!/usr/bin/env python3
"""E1 assessment (PLAN.md amendment 16, parallel CPU work). CPU only, read-only on captures.

Question: why do the served routing weights of the fused grouped_topk differ between batches of
M <= 1024 and M > 1024 rows, how large is the difference against fp64, and is it a defect?

Mechanism (from the kernel source at the image's vLLM commit 487ecf187 and the image's own sm_120
SASS; see E1-ASSESSMENT.md): with n_group = topk_group = 1, 288 experts, top-8,
  M <= 1024 -> single_group_topk_block_kernel<float,float,int,SIGMOID,384,8>
               u_i = s_i                       (unbiased sigmoid kept in shared memory)
  M  > 1024 -> single_group_topk_warp_kernel <float,float,int,SIGMOID,384,8>
               u_i = fl(fl(s_i + b_i) - b_i)   (unbiased score re-derived from the biased one)
  both:  S = butterfly fp32 sum over the warp (lanes 0..7 = u in output order, others 0)
         scale = fl(2.5f / fl(S + 1e-20f));  w_i = fl(u_i * scale)
  s_i = 0.5f * tanhf(0.5f * x) + 0.5f (CUDA tanhf: MUFU.EX2 / MUFU.RCP for |y| >= 0.6).

This script emulates both paths in IEEE fp32 on CPU (numpy) and compares each with the captured
served weights, bitwise and in ulps, and reports served accuracy against an fp64 restatement.
Sigmoid variants: 'cr' = correctly rounded fp32 sigmoid of the fp32 logit; 'tanh' = the CUDA tanhf
instruction sequence with exp2 and reciprocal correctly rounded (MUFU.EX2/RCP are approximate on
hardware and cannot be emulated bitwise on CPU).
Usage: e1_emulate.py CAPTURE_DIR CKPT_SNAPSHOT OUT_DIR
"""
import glob
import json
import os
import struct
import sys
import time

import numpy as np
import torch

E, K, SCALE = 288, 8, 2.5
f32 = np.float32


def ckpt_getter(d):
    mp = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]

    def get(key):
        f = os.path.join(d, mp[key])
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
            m = h[key]
            s, e = m["data_offsets"]
            fh.seek(8 + n + s)
            buf = bytearray(fh.read(e - s))
        dt = {"F32": torch.float32, "BF16": torch.bfloat16}[m["dtype"]]
        return torch.frombuffer(buf, dtype=dt).reshape(m["shape"]).clone()
    return get


def fma32(a, b, c):
    # a*b exact in fp64 for fp32 inputs; one fp64 rounding of the sum then fp32 rounding
    # (double rounding possible but rare; noted as an emulation limit)
    return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(f32)


def sigmoid_cr(x32):
    x = x32.astype(np.float64)
    return (1.0 / (1.0 + np.exp(-x))).astype(f32)


def sigmoid_tanh(x32):
    """CUDA sigmoid_accurate = 0.5f*tanhf(0.5f*x)+0.5f as compiled (SASS of both kernels)."""
    y = (x32 * f32(0.5)).astype(f32)
    ay = np.abs(y)
    # |y| >= 0.6 branch: t = 1 - 2/(2^(|y|*2*log2e) + 1); saturate to 1 for |y| >= 9.0109
    a = (ay * f32(2.8853900432586669922)).astype(f32)
    e = np.exp2(a.astype(np.float64)).astype(f32)              # MUFU.EX2 (approx on HW)
    d = (e + f32(1)).astype(f32)
    r = (1.0 / d.astype(np.float64)).astype(f32)                # MUFU.RCP (approx on HW)
    t_big = fma32(r, np.full_like(r, -2), np.ones_like(r))
    t_big = np.where(ay >= f32(9.010913848876953125), f32(1), t_big).astype(f32)
    t_big = np.copysign(t_big, y).astype(f32)
    # |y| < 0.6 branch: odd polynomial
    y2 = (y * y).astype(f32)
    c0 = np.frombuffer(struct.pack("<I", 0x3c80f082), dtype=f32)[0]
    p = fma32(y2, np.full_like(y2, c0), np.full_like(y2, f32(-0.052303962409496307373)))
    p = fma32(y2, p, np.full_like(y2, f32(0.1331529766321182251)))
    p = fma32(y2, p, np.full_like(y2, f32(-0.33332768082618713379)))
    p = (y2 * p).astype(f32)                                    # FFMA y2, p, RZ
    t_small = fma32(p, y, y)
    t = np.where(ay >= f32(0.60000002384185791016), t_big, t_small).astype(f32)
    return fma32(t, np.full_like(t, f32(0.5)), np.full_like(t, f32(0.5)))


def butterfly_sum(u):
    """u: (r, 8) fp32 in lane order; xor 16/8 add zeros; then xor 4, 2, 1."""
    a = (u[:, 0:4] + u[:, 4:8]).astype(f32)        # lane i: u_i + u_{i^4}
    b = (a[:, 0:2] + a[:, 2:4]).astype(f32)        # lanes 0,1 after xor 2
    return (b[:, 0] + b[:, 1]).astype(f32)


def weights(u):
    S = butterfly_sum(u)
    den = (S + f32(1e-20)).astype(f32)
    scale = (f32(SCALE) / den).astype(f32)
    return (u * scale[:, None]).astype(f32)


def ulps(a, b):
    ia = a.view(np.int32).astype(np.int64)
    ib = b.view(np.int32).astype(np.int64)
    return np.abs(ia - ib)


def mbin(m):
    for hi, name in ((256, "M<=256"), (512, "257-512"), (1024, "513-1024"), (2048, "1025-2048"),
                     (4096, "2049-4096"), (8192, "4097-8192")):
        if m <= hi:
            return name
    return "M>8192"


def q(v, ps=(0.5, 0.9, 0.99, 0.999, 1.0)):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return None
    return {f"p{int(p*1000)/10:g}" if p < 1 else "max": float(np.quantile(v, p)) for p in ps}


def main():
    cap, ck, out = sys.argv[1:4]
    get = ckpt_getter(ck)
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(cap, "rec_r0", "*.pt")))
    bias = {}
    acc = {}          # regime -> lists
    match = {}        # regime -> emulation -> counters
    msizes = {}
    near = {"rows": 0, "sel_mismatch_vs_emul_cr": 0, "sel_mismatch_vs_emul_tanh": 0}
    worst = []
    for f in files:
        rec = torch.load(f, map_location="cpu")
        meta = rec["meta"]
        L, m, r = meta["layer"], meta["m"], meta["r"]
        if L not in bias:
            bias[L] = get(f"model.language_model.layers.{L}.mlp.gate.e_score_correction_bias").float().numpy()
        b = bias[L]
        x = rec["logits_shipped"].float().numpy().astype(f32)
        ids = rec["topk_ids"].long().numpy()
        w = rec["topk_weights"].float().numpy().astype(f32)
        reg = "M<=1024" if m <= 1024 else "M>1024"
        msizes.setdefault(mbin(m), set()).add(m)
        bsel = b[ids].astype(f32)
        # fp64 reference: unbiased sigmoid of the same fp32 logits, renormalised, x2.5
        s64 = 1.0 / (1.0 + np.exp(-x.astype(np.float64)))
        s64sel = np.take_along_axis(s64, ids, 1)
        ref = s64sel / s64sel.sum(1, keepdims=True) * SCALE
        # emulations
        em = {}
        for sname, sfun in (("cr", sigmoid_cr), ("tanh", sigmoid_tanh)):
            s32 = sfun(x)
            ssel = np.take_along_axis(s32, ids, 1)
            # selection check with fp32 biased scores (ids order = lane order = descending score)
            biased = (s32 + b[None, :].astype(f32)).astype(f32)
            top = np.argsort(-biased.astype(np.float64), axis=1, kind="stable")[:, :K]
            mism = (np.sort(top, 1) != np.sort(ids, 1)).any(1)
            near[f"sel_mismatch_vs_emul_{sname}"] += int(mism.sum())
            u_block = ssel
            u_warp = ((ssel + bsel).astype(f32) - bsel).astype(f32)
            em[f"block/{sname}"] = weights(u_block)
            em[f"warp/{sname}"] = weights(u_warp)
            if sname == "cr":
                u_warp_cr, s_cr = u_warp, ssel
        near["rows"] += r
        mt = match.setdefault(reg, {})
        for k, v in em.items():
            c = mt.setdefault(k, {"rows": 0, "rows_bitwise": 0, "elems": 0, "elems_bitwise": 0,
                                  "elem_ulp_le1": 0, "ulp_max": 0})
            u = ulps(v, w)
            c["rows"] += r
            c["rows_bitwise"] += int((u == 0).all(1).sum())
            c["elems"] += u.size
            c["elems_bitwise"] += int((u == 0).sum())
            c["elem_ulp_le1"] += int((u <= 1).sum())
            c["ulp_max"] = max(c["ulp_max"], int(u.max()))
        # accuracy vs fp64
        a = acc.setdefault(reg, {"abs": [], "rel": [], "row_maxabs": [], "row_maxrel": [], "slot_abs": [[] for _ in range(K)],
                                 "slot_rel": [[] for _ in range(K)], "ulp_vs_ref": [], "w": [], "s": [], "absb": [],
                                 "pred_rel_warp_term": []})
        d = w.astype(np.float64) - ref
        a["abs"].append(np.abs(d).ravel())
        a["rel"].append((np.abs(d) / ref).ravel())
        a["row_maxabs"].append(np.abs(d).max(1))
        a["row_maxrel"].append((np.abs(d) / ref).max(1))
        for j in range(K):
            a["slot_abs"][j].append(np.abs(d[:, j]))
            a["slot_rel"][j].append(np.abs(d[:, j]) / ref[:, j])
        a["ulp_vs_ref"].append(ulps(w, ref.astype(f32)).ravel())
        a["w"].append(w.ravel())
        a["s"].append(s64sel.ravel())
        a["absb"].append(np.abs(bsel).ravel())
        # size of the warp-path reconstruction error |u' - s| / s with the cr sigmoid
        a["pred_rel_warp_term"].append((np.abs(u_warp_cr.astype(np.float64) - s_cr) / s_cr).ravel())
        rel = (np.abs(d) / ref)
        i, j = np.unravel_index(np.argmax(rel), rel.shape)
        worst.append((float(rel[i, j]), os.path.basename(f), int(m), int(L), int(i), int(j), float(w[i, j]),
                      float(ref[i, j]), float(s64sel[i, j]), float(bsel[i, j]), float(np.abs(d).max())))
    # per-M-bin accuracy (rebuild cheaply from a second light pass would duplicate I/O; compute here)
    res = {"files": len(files), "seconds": None, "selection_check": near,
           "m_values_by_bin": {k: sorted(v) for k, v in msizes.items()},
           "emulation_match": match, "accuracy_vs_fp64": {}}
    for reg, a in acc.items():
        cat = {k: np.concatenate(v) for k, v in a.items() if k not in ("slot_abs", "slot_rel")}
        rr = {"rows": int(cat["row_maxabs"].size), "elems": int(cat["abs"].size),
              "abs_err": q(cat["abs"]), "rel_err": q(cat["rel"]),
              "row_max_abs_err": q(cat["row_maxabs"]), "row_max_rel_err": q(cat["row_maxrel"]),
              "ulp_vs_fp64_rounded": q(cat["ulp_vs_ref"]),
              "per_slot": [{"slot": j, "abs_err": q(np.concatenate(a["slot_abs"][j]), (0.5, 0.99, 1.0)),
                            "rel_err": q(np.concatenate(a["slot_rel"][j]), (0.5, 0.99, 1.0))} for j in range(K)],
              "warp_reconstruction_rel_term_|u'-s|/s": q(cat["pred_rel_warp_term"]),
              "frac_elems_reconstruction_inexact": float((cat["pred_rel_warp_term"] > 0).mean()),
              "selected_sigmoid": q(cat["s"], (0.0, 0.01, 0.5, 1.0)),
              "selected_abs_bias": q(cat["absb"], (0.0, 0.5, 0.99, 1.0)),
              "weight": q(cat["w"], (0.0, 0.01, 0.5, 1.0))}
        # relation of the error to |b|/s (the warp-path amplification factor)
        ratio = cat["absb"] / cat["s"]
        bins = [0, 1, 4, 16, 64, 1e9]
        rr["rel_err_by_absb_over_s"] = [
            {"absb_over_s": f"[{lo},{hi})", "elems": int(((ratio >= lo) & (ratio < hi)).sum()),
             "rel_err": q(cat["rel"][(ratio >= lo) & (ratio < hi)], (0.5, 0.99, 1.0))}
            for lo, hi in zip(bins[:-1], bins[1:]) if ((ratio >= lo) & (ratio < hi)).any()]
        res["accuracy_vs_fp64"][reg] = rr
    worst.sort(reverse=True)
    res["worst_elements"] = [dict(zip(("rel_err", "file", "m", "layer", "row", "slot", "w_served", "w_ref64",
                                       "s64", "bias", "row_max_abs_err"), t)) for t in worst[:12]]
    res["bias_summary"] = {"layers": len(bias),
                           "abs_bias_max": float(max(np.abs(v).max() for v in bias.values())),
                           "bias_range": [float(min(v.min() for v in bias.values())), float(max(v.max() for v in bias.values()))]}
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(out, "e1_emulation.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("files", "seconds", "selection_check", "emulation_match", "bias_summary")}, indent=1))


if __name__ == "__main__":
    main()
