"""Measured nulls for index-K pool content (H4c/H5b/H5c), no decode-vs-prefill involved:
 (P) prefill vs prefill: the same pool in two V3ref prefill builds of the same trace that end at different frontiers
     (each built alone; only the total prefill length differs);
 (D) decode vs decode, same request at two snapshot times (a committed pool must be byte-stable: never rewritten)."""
import glob, json, os, re, collections, itertools
import numpy as np, torch
SN = collections.defaultdict(dict)
for p in glob.glob("/run/forcewin/snap_V3*_r*.pt"):
    m = re.search(r"snap_(.*)_F(\d+)(win)?_r(\d+)\.pt$", os.path.basename(p))
    SN[m.group(1) + "|" + m.group(2)][int(m.group(4))] = p
def attn_blocks(rec):
    for g in rec["groups"]:
        if g["bpk"] > 1:
            ids = g["ids"].tolist(); bpk = g["bpk"]
            return lambda c, ids=ids, bpk=bpk: ids[c * bpk] // bpk if c * bpk < len(ids) else -1
def raw(pages, blk_of, pool):
    blk = blk_of((4 * pool + 3) // 4352); pg = blk * 17 + (pool % 1088) // 64
    t = pages.get(pg)
    if t is None: return None
    r_ = t.reshape(-1).view(torch.uint8); rr = pool % 64
    return r_[rr * 128:(rr + 1) * 128].clone(), r_[8192 + rr * 4: 8192 + rr * 4 + 4].clone()
deq = lambda b: b[0].view(torch.float8_e4m3fn).float() * b[1].view(torch.float32)
def pools(rec):
    fa = attn_blocks(rec); out = {}
    fr = rec["frontier_pos"]
    for name, L in rec["layers"].items():
        if L.get("kind") != "paged" or L.get("shape", [0])[-1:] != [132]: continue
        lay = int(name.split(".")[3])
        for pool in range(0, (fr + 1) // 4):
            x = raw(L["pages"], fa, pool)
            if x is not None: out[(lay, pool)] = x
    return out
res = {"P": [], "D": []}
# group by trace: V3ref-F{F}-V3-{mode}-{tid}  and decode V3-{mode}-{tid}
groups = collections.defaultdict(lambda: {"ref": [], "dec": []})
for k in SN:
    tag, F = k.split("|")
    if tag.startswith("V3ref-"):
        groups[tag.split("-", 2)[2]]["ref"].append(k)
    elif tag.startswith("V3-"):
        groups[tag]["dec"].append(k)
for g, d in sorted(groups.items()):
    for kind, keys in (("P", d["ref"]), ("D", d["dec"])):
        for k1, k2 in itertools.combinations(sorted(keys), 2):
            for rank in (0, 1):
                if rank not in SN[k1] or rank not in SN[k2]: continue
                a = pools(torch.load(SN[k1][rank], weights_only=False)); b = pools(torch.load(SN[k2][rank], weights_only=False))
                for key in set(a) & set(b):
                    x, y = a[key], b[key]
                    same = bool(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]))
                    xv, yv = deq(x), deq(y)
                    res[kind].append(dict(g=g, k1=k1, k2=k2, rank=rank, layer=key[0], pool=key[1], same=same,
                                          rel=float((xv - yv).norm() / max(float(yv.norm()), 1e-12))))
        print(g, kind, len(res[kind]), flush=True)
json.dump(res, open("/d/out/h4c_nulls.json", "w"))
for kind, nm in (("P", "prefill vs prefill (different build lengths)"), ("D", "decode vs decode (same request, two snapshot times)")):
    v = res[kind]
    if not v: continue
    a = np.array([x["rel"] for x in v])
    print("%s: n %d bitwise-equal %.4f  rel q50 %.4f q99 %.4f q999 %.4f max %.4f gross(>0.25) %.4f" % (
        nm, len(v), np.mean([x["same"] for x in v]), np.median(a), np.quantile(a, .99), np.quantile(a, .999), a.max(), np.mean(a > .25)))
    bl = collections.defaultdict(list)
    for x in v: bl[x["layer"]].append(x["rel"])
    print("   by layer gross:", {L: "%.4f" % np.mean(np.array(r) > .25) for L, r in sorted(bl.items())})
