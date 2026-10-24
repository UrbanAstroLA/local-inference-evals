import sys, os, json, collections
import numpy as np
sys.path.insert(0, "/p4/analysis")
import analyze_p4 as A
root = "/p4"
S = {s: A.Session(s, A.pick_attempt(root, s)) for s in ("A", "B")}
wins = sorted({f.split("req_A-")[1].rsplit("-r", 1)[0] for f in os.listdir(os.path.join(S["A"].priv, "fw")) if f.startswith("req_A-") and f.endswith("_r0.pt")})
bins = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
prof = collections.defaultdict(lambda: collections.defaultdict(list))
for w in wins:
    cls = w.rsplit("-", 1)[-1]
    x, y = A.WinData(S["A"], w, 1), A.WinData(S["A"], w, 2)
    if not (x.ok and y.ok): continue
    for o in range(A.W):
        b = max(i for i in range(len(bins)) if bins[i] <= o)
        for k, nm in ((0, "moe3"), (1, "moe23"), (2, "moe44")):
            mx, my = x.moe(k, o), y.moe(k, o)
            if mx is not None and my is not None:
                prof[(cls, nm)][bins[b]].append(A.rel_l2(A.bf16_to_f32(mx), A.bf16_to_f32(my)))
        c = x.c
        if x.nmeas[o] == 1 and y.nmeas[o] == 1:
            prof[(cls, "lpforced_eq")][bins[b]].append(float(x.m[o, c["lp_pre_4"]] == y.m[o, c["lp_pre_4"]]))
for key in sorted(prof):
    row = " ".join(f"{b}:{np.median(v):.3g}" for b, v in sorted(prof[key].items()))
    print(key, row)
# also: window 0 first steps detail for one window
w = wins[0]; x, y = A.WinData(S["A"], w, 1), A.WinData(S["A"], w, 2)
print("window", w)
for o in range(0, 12):
    mx, my = x.moe(1, o), y.moe(1, o)
    print(o, "nmeas", x.nmeas[o], y.nmeas[o], "step", x.m[o, x.c["step"]], y.m[o, y.c["step"]], "moe23 rel", None if mx is None or my is None else round(A.rel_l2(A.bf16_to_f32(mx), A.bf16_to_f32(my)), 5), "lp_f", x.m[o, x.c["lp_pre_4"]], y.m[o, y.c["lp_pre_4"]])
