#!/usr/bin/env python3
"""E1 supplement: tail rows, residual emulation mismatches, rank agreement, M-bin accuracy.
Reuses the emulation functions of e1_emulate.py. CPU only, read-only on captures.
Usage: e1_supplement.py CAPTURE_DIR CKPT_SNAPSHOT OUT_DIR"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_emulate import SCALE, ckpt_getter, f32, mbin, q, sigmoid_tanh, ulps, weights  # noqa: E402


def main():
    cap, ck, out = sys.argv[1:4]
    get = ckpt_getter(ck)
    bias = {}
    bins = {}
    tail = {"rows_M>1024": 0, "row_max_rel_gt_1e-5": 0, "row_max_rel_gt_1e-4": 0, "row_max_abs_gt_1e-5": 0,
            "row_max_abs_gt_1e-6": 0, "sum_s_of_rows_row_max_abs_gt_1e-5": []}
    resid = {"elems_mismatch": 0, "implied_du_over_grid": []}
    ranks = {"records": 0, "weights_bitwise_equal_r0_r1": 0, "ids_equal_r0_r1": 0}
    sumS = {"M<=1024": [], "M>1024": []}
    for f in sorted(glob.glob(os.path.join(cap, "rec_r0", "*.pt"))):
        rec = torch.load(f, map_location="cpu")
        meta = rec["meta"]
        L, m = meta["layer"], meta["m"]
        if L not in bias:
            bias[L] = get(f"model.language_model.layers.{L}.mlp.gate.e_score_correction_bias").float().numpy()
        f1 = f.replace("rec_r0", "rec_r1")
        if os.path.exists(f1):
            r1 = torch.load(f1, map_location="cpu")
            ranks["records"] += 1
            ranks["weights_bitwise_equal_r0_r1"] += int(torch.equal(r1["topk_weights"], rec["topk_weights"]))
            ranks["ids_equal_r0_r1"] += int(torch.equal(r1["topk_ids"], rec["topk_ids"]))
        x = rec["logits_shipped"].float().numpy().astype(f32)
        ids = rec["topk_ids"].long().numpy()
        w = rec["topk_weights"].float().numpy().astype(f32)
        bsel = bias[L][ids].astype(f32)
        s64 = np.take_along_axis(1.0 / (1.0 + np.exp(-x.astype(np.float64))), ids, 1)
        ref = s64 / s64.sum(1, keepdims=True) * SCALE
        d = np.abs(w.astype(np.float64) - ref)
        rel = d / ref
        reg = "M<=1024" if m <= 1024 else "M>1024"
        sumS[reg].append(s64.sum(1))
        bb = bins.setdefault(mbin(m), {"rows": 0, "abs": [], "rel": []})
        bb["rows"] += x.shape[0]
        bb["abs"].append(d.ravel())
        bb["rel"].append(rel.ravel())
        if m > 1024:
            tail["rows_M>1024"] += x.shape[0]
            rmr, rma = rel.max(1), d.max(1)
            tail["row_max_rel_gt_1e-5"] += int((rmr > 1e-5).sum())
            tail["row_max_rel_gt_1e-4"] += int((rmr > 1e-4).sum())
            tail["row_max_abs_gt_1e-5"] += int((rma > 1e-5).sum())
            tail["row_max_abs_gt_1e-6"] += int((rma > 1e-6).sum())
            tail["sum_s_of_rows_row_max_abs_gt_1e-5"] += s64.sum(1)[rma > 1e-5].tolist()
            # residual mismatches of the warp/tanh emulation: implied change of u in units of the
            # (s+b) grid step ulp(s+b) -- one step means the hardware sigmoid landed on the other
            # side of a rounding boundary of s+b
            s32 = np.take_along_axis(sigmoid_tanh(x), ids, 1)
            u = ((s32 + bsel).astype(f32) - bsel).astype(f32)
            we = weights(u)
            mm = ulps(we, w) > 0
            if mm.any():
                S = u.astype(np.float64).sum(1, keepdims=True)
                scale = SCALE / S
                du = (w.astype(np.float64) - we.astype(np.float64)) / scale
                grid = np.spacing((s32 + bsel).astype(f32)).astype(np.float64)
                resid["elems_mismatch"] += int(mm.sum())
                resid["implied_du_over_grid"] += (np.abs(du[mm]) / grid[mm]).tolist()
    r = np.asarray(resid["implied_du_over_grid"])
    res = {
        "accuracy_by_M_bin": {k: {"rows": v["rows"], "abs_err": q(np.concatenate(v["abs"]), (0.5, 0.99, 1.0)),
                                  "rel_err": q(np.concatenate(v["rel"]), (0.5, 0.99, 1.0))}
                              for k, v in sorted(bins.items())},
        "tail_M>1024": {**{k: v for k, v in tail.items() if not isinstance(v, list)},
                        "sum_s_of_rows_row_max_abs_gt_1e-5": q(tail["sum_s_of_rows_row_max_abs_gt_1e-5"], (0.0, 0.5, 1.0))},
        "sum_of_selected_sigmoids": {k: q(np.concatenate(v), (0.0, 0.001, 0.01, 0.5)) for k, v in sumS.items()},
        "warp_tanh_residual_mismatch": {
            "elems": resid["elems_mismatch"],
            "implied_|du|/grid_step": q(r, (0.0, 0.5, 0.99, 1.0)) if r.size else None,
            "frac_within_0.5_to_1.5_steps": float(((r > 0.5) & (r < 1.5)).mean()) if r.size else None},
        "rank_agreement": ranks,
    }
    json.dump(res, open(os.path.join(out, "e1_supplement.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
