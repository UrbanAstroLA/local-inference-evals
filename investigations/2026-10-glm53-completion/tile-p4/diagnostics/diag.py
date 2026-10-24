import sys, os, json, collections
import numpy as np
sys.path.insert(0, "/p4/analysis")
import analyze_p4 as A
root = "/p4"
S = {s: A.Session(s, A.pick_attempt(root, s)) for s in ("A", "B", "A2")}
wins = sorted({c.get("wid") for c in S["A"].client if c.get("wid")} - {None})
if not wins:
    wins = sorted({f.split("req_A-")[1].rsplit("-r", 1)[0] for f in os.listdir(os.path.join(S["A"].priv, "fw")) if f.startswith("req_A-")})
print("windows", len(wins), wins[:4])
def wd(s, w, r): return A.WinData(S[s], w, r)
def host(sess, step):
    h = sess.steps.get(int(step)); return None if h is None else (h["num_reqs"], h["num_tokens"], h["num_tokens_padded"], h["num_draft_tokens"], h["n_prefilling"], h["graph_mode"])
res = collections.defaultdict(list); cnt = collections.Counter()
for w in wins:
    d = {(s, r): wd(s, w, r) for s in ("A", "B", "A2") for r in (1, 2)}
    if not all(x.ok for x in d.values()): cnt["window_missing"] += 1; continue
    c = d[("A",1)].c
    for pair in (("A",1,"A",2), ("A",1,"B",1), ("A",2,"B",2), ("B",1,"B",2), ("A",1,"A2",1)):
        x, y = d[(pair[0],pair[1])], d[(pair[2],pair[3])]
        name = f"{pair[0]}{pair[1]}-{pair[2]}{pair[3]}"
        for o in range(A.W):
            if x.nmeas[o] != 1 or y.nmeas[o] != 1: cnt[name+":not_once"] += 1; continue
            hx = host(S[pair[0]], x.m[o, c["step"]]); hy = host(S[pair[2]], y.m[o, c["step"]])
            same_host = hx == hy and hx is not None
            cnt[name+(":host_eq" if same_host else ":host_ne")] += 1
            lx, ly = x.m[o, c["lp_pre_4"]], y.m[o, c["lp_pre_4"]]
            res[(name, "lp_forced_eq", same_host)].append(float(lx == ly))
            res[(name, "lp_forced_absd", same_host)].append(abs(float(lx - ly)))
            res[(name, "lp_close_absd", same_host)].append(abs(float(x.m[o, c["lp_pre_0"]] - y.m[o, c["lp_pre_0"]])))
            for k in (1,):
                mx, my = x.moe(k, o), y.moe(k, o)
                if mx is not None and my is not None:
                    res[(name, "moe23_bitequal", same_host)].append(float(np.array_equal(mx, my)))
                    res[(name, "moe23_rel", same_host)].append(A.rel_l2(A.bf16_to_f32(mx), A.bf16_to_f32(my)))
print(json.dumps(cnt, indent=0)[:1500])
for k in sorted(res, key=str):
    v = np.array(res[k]); print(k, "n", len(v), "mean", round(float(v.mean()), 6), "q50", round(float(np.quantile(v, .5)), 6), "q90", round(float(np.quantile(v, .9)), 6))
