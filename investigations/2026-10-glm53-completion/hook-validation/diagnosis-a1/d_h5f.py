"""H5f: measured relative L2 (window pool vs prefill rebuild) by shadow-ledger class, both injected arms and B1.
The frozen H5f joins the class array pool_class[k], k = pool - P//4 (the hook's own index), for d2 and d2solo."""
import json, collections, numpy as np, torch
R = [json.loads(l) for l in open("/d/out/h5_pools.jsonl")]
cache = {}
def pc(tag):
    if tag not in cache:
        a = torch.load(f"/run/forcewin/req_{tag}_r0.pt", weights_only=False)
        cache[tag] = (a["pool_class"].numpy(), a.get("shadow_classes", {}), a["P"])
    return cache[tag]
CN = {0: "none", 1: "clean", 2: "corrupt", 3: "race", 4: "seed"}
by = collections.defaultdict(list)
mism_dict = 0
for r in R:
    arr, dct, P = pc(r["tag"])
    k = r["pool"] - P // 4
    c = int(arr[k]) if 0 <= k < arr.size else -1
    c2 = dct.get(r["pool"], -1) if isinstance(dct, dict) else -1
    if c2 != -1 and c2 != c: mism_dict += 1
    by[(r["arm"], CN.get(c, "?"))].append(r["rel"])
print("pool_class array vs shadow_classes dict disagreements:", mism_dict)
res = {}
for (arm, c), v in sorted(by.items()):
    a = np.array(v)
    res[f"{arm}|{c}"] = dict(n=len(v), gross=float(np.mean(a > .25)), q10=float(np.quantile(a, .1)), q50=float(np.median(a)), q90=float(np.quantile(a, .9)))
    print("%-7s %-8s n %6d gross %.3f  rel q10 %.3f q50 %.3f q90 %.3f | frac in (0.10,0.25] %.3f" % (arm, c, len(v), np.mean(a > .25),
          np.quantile(a, .1), np.median(a), np.quantile(a, .9), np.mean((a > .1) & (a <= .25))))
json.dump(res, open("/d/out/h5f_classes.json", "w"), indent=1)
