"""H5f restricted (post-inspection, labelled): ledger class vs measured bytes only for pools whose completion step
and member-stash steps were all unpadded (the ledger mis-slices padded steps). Under D2, other requests' writes to
the shared block 0 in padded steps are still invisible to the ledger, so the 'corrupt'/'clean' split stays biased."""
import json, collections, glob, numpy as np, torch
R = [json.loads(l) for l in open("/d/out/h5_pools.jsonl")]
FW = [json.loads(l) for p in glob.glob("/run/forcewin/fw_r0*.jsonl") for l in open(p)]
st = {}
for r in FW:
    if r.get("kind") == "steps":
        for row in r["rows"]: st[row[0]] = row
pad = lambda s: s in st and st[s][5] > 0
info = {}
for tag in sorted({r["tag"] for r in R}):
    a = torch.load(f"/run/forcewin/req_{tag}_r0.pt", weights_only=False)
    C = {n: i for i, n in enumerate(a["cols"])}; M = a["metrics"].numpy(); P = a["P"]; W = a["W"]; pc = a["pool_class"].numpy()
    step_of = {P + o - 1: int(M[o, C["step"]]) for o in range(1, W)}
    for i in range((P + 3) // 4, (P + W - 1) // 4):
        ss = [step_of.get(q) for q in range(4 * i, 4 * i + 4)]
        k = i - P // 4
        info[(tag, i)] = (int(pc[k]) if 0 <= k < pc.size else -1, all(s is not None and not pad(s) for s in ss))
c = collections.defaultdict(list)
for r in R:
    cl, ok = info.get((r["tag"], r["pool"]), (-1, False))
    c[(r["arm"], {0: "none", 1: "clean", 2: "corrupt", 3: "race", -1: "?"}[cl], ok)].append(r["rel"])
res = {}
for k, v in sorted(c.items()):
    a = np.array(v); res["|".join(map(str, k))] = dict(n=len(v), gross=float(np.mean(a > .25)), q50=float(np.median(a)))
    print("%-7s %-8s all-unpadded=%-5s n %6d gross %.3f q50 %.3f" % (k[0], k[1], k[2], len(v), np.mean(a > .25), np.median(a)))
json.dump(res, open("/d/out/h5f_restricted.json", "w"), indent=1)
