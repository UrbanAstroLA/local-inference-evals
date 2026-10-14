#!/usr/bin/env python3
"""Within-instrument B1-vs-B1 null for termination measures, from EXISTING attempt-2 records (CPU only; no GPU).

PURPOSE (first-contrast PROPOSAL.md section 6): size the run-to-run variation the first contrast must resolve,
BEFORE its thresholds are frozen. Uses ONLY the two clean B1 decode arms of attempt 2 on the same 8 short windows:
V4-b1 (one batch of 8) and V4n-b1b (two batches of 4: a different batch composition, so this null is, if anything,
LARGER than a same-composition repeat). The D2 arm (V4-d2, V4-d2solo) is deliberately NOT read: no D2-vs-B1
termination difference is computed or looked at before the first-contrast design is frozen.

Reads run-a2/forcewin/req_V4-b1-*_r0.pt and req_V4n-b1b-*_r0.pt (rank 0; ranks are identical by H1). Writes
b1_null_a2.json: per window and pooled, |delta ln P_pre(</think>)| per position (raw and floored at ln 1e-9), the
window-mean difference (the first contrast's window statistic), and the same for the stop tokens and the forced
token. Positions: output indices with nmeas == 1 in both records, after a 64-position lead-in (short windows).

usage (CPU container, no network):
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v RUN_A2:/run:ro -v HERE:/w --entrypoint python3 \
      sha256:799d3cf8... /w/b1_null_a2.py /run /w/b1_null_a2.json
"""
import glob
import json
import math
import os
import sys

import torch

RUN, OUT = sys.argv[1], sys.argv[2]
FLOOR = math.log(1e-9)
LEAD = 64
NAMES = ["think_close_154842", "eot_154820", "user_154827", "observation_154829", "forced"]


def load(p):
    d = torch.load(p, weights_only=False)
    m = d["metrics"].float()
    cols = d["cols"]
    return d, m, {c: i for i, c in enumerate(cols)}


def q(v, f):
    v = sorted(v)
    return v[min(len(v) - 1, int(f * (len(v) - 1) + 0.5))] if v else None


res = {"windows": {}, "note": "B1 vs B1 only (V4-b1 vs V4n-b1b); the D2 arm is not read"}
pooled = {n: [] for n in NAMES}
pooled_floor = []
win_stats = []
for pa in sorted(glob.glob(os.path.join(RUN, "forcewin", "req_V4-b1-*_r0.pt"))):
    tid = os.path.basename(pa)[len("req_V4-b1-"):-len("_r0.pt")]
    pb = os.path.join(RUN, "forcewin", f"req_V4n-b1b-{tid}_r0.pt")
    if not os.path.exists(pb):
        continue
    da, ma, ca = load(pa)
    db, mb, cb = load(pb)
    na, nb = da["nmeas"], db["nmeas"]
    n = min(ma.shape[0], mb.shape[0])
    ok = [o for o in range(LEAD, n) if int(na[o]) == 1 and int(nb[o]) == 1]
    w = {"W": int(da["W"]), "positions": len(ok)}
    for k, name in enumerate(NAMES):
        a = ma[ok, ca["lp_pre_%d" % k]]
        b = mb[ok, cb["lp_pre_%d" % k]]
        d = (a - b).abs().tolist()
        pooled[name] += d
        w[name] = {"abs_delta_median": q(d, 0.5), "abs_delta_p99": q(d, 0.99), "abs_delta_max": max(d) if d else None,
                   "window_mean_delta": float((a - b).mean()) if d else None}
    a0 = ma[ok, ca["lp_pre_0"]].clamp(min=FLOOR)
    b0 = mb[ok, cb["lp_pre_0"]].clamp(min=FLOOR)
    w["think_close_floored"] = {"window_mean_a": float(a0.mean()), "window_mean_b": float(b0.mean()),
                                "window_mean_delta": float((a0 - b0).mean()),
                                "share_positions_both_at_floor": float(((a0 <= FLOOR) & (b0 <= FLOOR)).float().mean())}
    pooled_floor += (a0 - b0).abs().tolist()
    # the frozen primary window statistic of the first contrast: ln of the window-mean P_pre(</think>)
    for k, name in enumerate(NAMES[:4]):
        la = torch.logsumexp(ma[ok, ca["lp_pre_%d" % k]], 0) - math.log(len(ok))
        lb = torch.logsumexp(mb[ok, cb["lp_pre_%d" % k]], 0) - math.log(len(ok))
        w.setdefault("ln_mean_p", {})[name] = {"a": float(la), "b": float(lb), "delta": float(la - lb)}
    win_stats.append(abs(w["think_close_floored"]["window_mean_delta"]))
    # argmax agreement and entropy difference
    am = (ma[ok, ca["argmax"]] == mb[ok, cb["argmax"]]).float().mean()
    w["argmax_agreement"] = float(am)
    w["entropy_abs_delta_median"] = q((ma[ok, ca["entropy"]] - mb[ok, cb["entropy"]]).abs().tolist(), 0.5)
    res["windows"][tid] = w

res["ln_mean_p_abs_delta"] = {name: sorted(abs(w["ln_mean_p"][name]["delta"]) for w in res["windows"].values())
                              for name in NAMES[:4]}
res["pooled"] = {name: {"n": len(v), "median": q(v, 0.5), "p90": q(v, 0.9), "p99": q(v, 0.99),
                        "max": max(v) if v else None} for name, v in pooled.items()}
res["pooled_think_close_floored_abs_delta"] = {"median": q(pooled_floor, 0.5), "p99": q(pooled_floor, 0.99),
                                               "max": max(pooled_floor) if pooled_floor else None}
res["window_mean_abs_delta_think_close_floored"] = {"per_window": win_stats, "max": max(win_stats) if win_stats else None,
                                                    "mean": sum(win_stats) / len(win_stats) if win_stats else None}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("ln_mean_p_abs_delta", "pooled", "pooled_think_close_floored_abs_delta",
                                      "window_mean_abs_delta_think_close_floored")}, indent=1))
