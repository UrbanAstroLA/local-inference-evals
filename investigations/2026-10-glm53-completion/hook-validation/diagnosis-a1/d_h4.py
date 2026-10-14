"""H4b/H4c: KDA committed-slot identification and index-K pools near the frontier, V3 snapshots vs prefill-built
references at the same frontier. Per layer, per slot, with the one-token-update scale (distance between slots)."""
import glob, json, os, re, collections
import numpy as np, torch
SNAP = {}
for p in glob.glob("/run/forcewin/snap_V3*_r*.pt"):
    m = re.search(r"snap_(.*)_F(\d+)(win)?_r(\d+)\.pt$", os.path.basename(p))
    SNAP.setdefault((m.group(1), int(m.group(2)), bool(m.group(3))), {})[int(m.group(4))] = p
pairs = []
for (tag, F, win), ranks in SNAP.items():
    if win or not tag.startswith("V3-"): continue
    for (rtag, rF, rwin), rr in SNAP.items():
        if rtag == f"V3ref-F{F}-{tag}" and not rwin:
            pairs.append((tag, F, ranks, rr))
print("snapshot pairs", len(pairs))
def attn_blocks(rec):
    for g in rec["groups"]:
        if g["bpk"] > 1:
            ids = g["ids"].tolist(); bpk = g["bpk"]
            return lambda c, ids=ids, bpk=bpk: ids[c * bpk] // bpk if c * bpk < len(ids) else -1
    return lambda c: -1
def idx_pool(pages, blk_of, pool):
    blk = blk_of((4 * pool + 3) // 4352); pg = blk * 17 + (pool % 1088) // 64
    t = pages.get(pg)
    if t is None: return None
    raw = t.reshape(-1).view(torch.uint8); rr = pool % 64
    v = raw[rr * 128:(rr + 1) * 128].view(torch.float8_e4m3fn).float(); s = raw[8192 + rr * 4: 8192 + rr * 4 + 4].view(torch.float32)
    return v * s
K = []; I = []
for tag, F, ranks, rr in sorted(pairs):
    for rank, p in ranks.items():
        if rank not in rr: continue
        a = torch.load(p, weights_only=False); b = torch.load(rr[rank], weights_only=False)
        na = a.get("num_accepted"); fr = a["frontier_pos"]
        for name, La in a["layers"].items():
            Lb = b["layers"].get(name)
            if La.get("kind") == "kda" and Lb is not None and "ssm_sketch" in La and "ssm_sketch" in Lb and na and La.get("is_spec"):
                ref = Lb["ssm_sketch"][0]; S_ = La["ssm_sketch"]
                dist = [float((s - ref).norm() / max(ref.norm(), 1e-12)) for s in S_]
                inter = [float((S_[k] - S_[na - 1]).norm() / max(ref.norm(), 1e-12)) for k in range(S_.shape[0])]
                K.append(dict(tag=tag, mode=tag.split("-")[1], F=F, rank=rank, layer=int(name.split(".")[3]), na=na,
                              layer_na=La.get("num_accepted"), slots=La["slots"].tolist(), dist=dist, inter=inter,
                              rel=dist[na - 1], margin=min([x for k, x in enumerate(dist) if k != na - 1]) / max(dist[na - 1], 1e-12),
                              best=int(np.argmin(dist)), nref=float(Lb["ssm_norm"][0]), ncom=float(La["ssm_norm"][na - 1]),
                              conv_rel=None))
        fa, fb = attn_blocks(a), attn_blocks(b)
        for name, La in a["layers"].items():
            if La.get("kind") != "paged" or La.get("shape", [0])[-1:] != [132]: continue
            Lb = b["layers"].get(name)
            if Lb is None: continue
            for pool in range(max(0, fr // 4 - 8), (fr + 1) // 4):
                x = idx_pool(La["pages"], fa, pool); y = idx_pool(Lb["pages"], fb, pool)
                if x is None or y is None: continue
                I.append(dict(tag=tag, F=F, rank=rank, layer=int(name.split(".")[3]), pool=pool, frontier=fr, P=a["P"],
                              last=4 * pool + 3, rel=float((x - y).norm() / max(float(y.norm()), 1e-12)),
                              cos=float((x @ y) / max(float(x.norm() * y.norm()), 1e-12)), xn=float(x.norm()), yn=float(y.norm())))
json.dump(K, open("/d/out/h4b_kda.json", "w")); json.dump(I, open("/d/out/h4c_idx.json", "w"))
rel = np.array([k["rel"] for k in K]); mar = np.array([k["margin"] for k in K])
print("KDA n", len(K), "rel q50 %.4f q90 %.4f max %.4f" % (np.median(rel), np.quantile(rel, .9), rel.max()),
      "margin q05 %.2f q50 %.2f min %.2f" % (np.quantile(mar, .05), np.median(mar), mar.min()))
print("committed slot is the closest to the reference in %.4f of comparisons" % np.mean([k["best"] == k["na"] - 1 for k in K]))
print("top-level num_accepted == layer metadata num_accepted:", collections.Counter(k["na"] == k["layer_na"] for k in K))
inter = np.array([min([x for j, x in enumerate(k["inter"]) if j != k["na"] - 1] or [np.nan]) for k in K])
print("one-token scale (nearest scratch slot to committed, rel to ref norm): q50 %.4f q10 %.4f" % (np.nanmedian(inter), np.nanquantile(inter, .1)))
by = collections.defaultdict(list)
for k in K: by[k["layer"]].append(k)
for L in sorted(by):
    r = np.array([k["rel"] for k in by[L]]); m = np.array([k["margin"] for k in by[L]])
    it = np.array([min([x for j, x in enumerate(k["inter"]) if j != k["na"] - 1] or [np.nan]) for k in by[L]])
    print(" layer %2d n %4d rel q50 %.4f max %.4f | margin q50 %.2f min %.2f | best==committed %.3f | one-token %.4f" % (
        L, len(r), np.median(r), r.max(), np.median(m), m.min(), np.mean([k["best"] == k["na"] - 1 for k in by[L]]), np.nanmedian(it)))
bm = collections.defaultdict(list)
for k in K: bm[(k["mode"], k["na"])].append(k)
for kk in sorted(bm):
    r = np.array([k["rel"] for k in bm[kk]]); m = np.array([k["margin"] for k in bm[kk]])
    print(" mode %-10s na %d n %4d rel q50 %.4f max %.4f margin q50 %.2f min %.2f best==committed %.3f" % (kk[0], kk[1], len(r), np.median(r), r.max(), np.median(m), m.min(), np.mean([k["best"] == k["na"] - 1 for k in bm[kk]])))
a = np.array([i["rel"] for i in I])
print("IDX n", len(I), "gross", int((a > .25).sum()), "q50 %.4f q99 %.4f max %.4f" % (np.median(a), np.quantile(a, .99), a.max()))
bl = collections.Counter(i["layer"] for i in I if i["rel"] > .25); print(" gross by layer", sorted(bl.items()))
bd = collections.Counter(i["frontier"] - i["last"] for i in I if i["rel"] > .25); print(" gross by frontier-last", sorted(bd.items()))
bt = collections.Counter(i["tag"] for i in I if i["rel"] > .25); print(" gross by tag", sorted(bt.items()))
bpos = collections.Counter((i["tag"].rsplit("-",1)[-1], i["pool"]) for i in I if i["rel"] > .25); print(" gross by (trace,pool)", sorted(bpos.items())[:40])
