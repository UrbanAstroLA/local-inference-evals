"""H5b/H5e: decode-vs-decode comparisons of the same window pools (no prefill involved).
 (1) V4-b1 vs V4-d2 (same 8-request batch). Below context 2,044 every pool is selected, so D2's index-K corruption
     cannot change attention: pools the d2 shadow ledger calls clean must equal B1's pools if decode is deterministic.
 (2) V4-b1 vs V4-d2solo for cbac17c70af8 (batch 8 vs batch 1).
Also joins each pool with its decode-vs-rebuild relative L2 (out/h5_pools.jsonl)."""
import glob, json, os, re, collections
import numpy as np, torch
SNAP = {}
for p in glob.glob("/run/forcewin/snap_V4-*win_r*.pt"):
    m = re.search(r"snap_(.*)_F(\d+)(win)?_r(\d+)\.pt$", os.path.basename(p))
    SNAP.setdefault(m.group(1), {})[int(m.group(4))] = p
REB = {}
for l in open("/d/out/h5_pools.jsonl"):
    r = json.loads(l); REB[(r["arm"], r["tid"], r["layer"], r["pool"])] = r["rel"]
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
def deq(b):
    return b[0].view(torch.float8_e4m3fn).float() * b[1].view(torch.float32)
out = []
def compare(tagA, tagB, label, cls_tag=None):
    pc = None
    if cls_tag:
        q = torch.load(f"/run/forcewin/req_{cls_tag}_r0.pt", weights_only=False); pc = q["pool_class"].numpy(); P = q["P"]
    for rank in (0, 1):
        a = torch.load(SNAP[tagA][rank], weights_only=False); b = torch.load(SNAP[tagB][rank], weights_only=False)
        fa, fb = attn_blocks(a), attn_blocks(b); lo, hi = a["range"]; P = a["P"]
        for name, La in a["layers"].items():
            if La.get("kind") != "paged" or La.get("shape", [0])[-1:] != [132]: continue
            Lb = b["layers"][name]; L = int(name.split(".")[3])
            for pool in range((lo + 3) // 4, (hi + 1) // 4):
                x = raw(La["pages"], fa, pool); y = raw(Lb["pages"], fb, pool)
                if x is None or y is None: continue
                same = bool(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]))
                xv, yv = deq(x), deq(y)
                rel = float((xv - yv).norm() / max(float(yv.norm()), 1e-12))
                k = pool - P // 4
                cls = int(pc[k]) if pc is not None and 0 <= k < pc.size else -1
                tid = tagA.rsplit("-", 1)[-1]
                out.append(dict(cmp=label, tid=tid, rank=rank, layer=L, pool=pool, last=4 * pool + 3, same=same, rel=rel, cls=cls,
                                relA_reb=REB.get((tagA.split("-")[1], tid, L, pool)), relB_reb=REB.get((tagB.split("-")[1], tid, L, pool))))
tids = sorted({t.rsplit("-", 1)[-1] for t in SNAP if t.startswith("V4-b1-")})
for tid in tids:
    compare(f"V4-b1-{tid}", f"V4-d2-{tid}", "b1_vs_d2", cls_tag=f"V4-d2-{tid}")
compare("V4-b1-cbac17c70af8", "V4-d2solo-cbac17c70af8", "b1_vs_solo", cls_tag="V4-d2solo-cbac17c70af8")
json.dump(out, open("/d/out/h5_decode_null.json", "w"))
CN = {0: "none", 1: "clean", 2: "corrupt", 3: "race", 4: "seed", -1: "?"}
for label in ("b1_vs_d2", "b1_vs_solo"):
    R = [r for r in out if r["cmp"] == label]
    for ctx in ("<2044", ">=2044"):
        S = [r for r in R if (r["last"] < 2044) == (ctx == "<2044")]
        if not S: continue
        by = collections.defaultdict(list)
        for r in S: by[CN[r["cls"]]].append(r)
        for c, v in sorted(by.items()):
            rel = np.array([r["rel"] for r in v])
            print("%-10s ctx%-6s class %-8s n %6d bitwise-equal %.4f  rel q50 %.4f q99 %.4f gross %.4f" % (
                label, ctx, c, len(v), np.mean([r["same"] for r in v]), np.median(rel), np.quantile(rel, .99), np.mean(rel > .25)))
# decode-vs-rebuild gross pools: are they reproduced by the other decode?
G = [r for r in out if r["cmp"] == "b1_vs_d2" and r["relA_reb"] is not None and r["relA_reb"] > .25]
print("B1 pools gross vs rebuild:", len(G), "| bitwise equal to the d2 arm's pool:", sum(r["same"] for r in G),
      "| d2 pool also gross vs rebuild:", sum((r["relB_reb"] or 0) > .25 for r in G),
      "| B1-vs-d2 rel > .25:", sum(r["rel"] > .25 for r in G))
G2 = [r for r in out if r["cmp"] == "b1_vs_solo" and ((r["relA_reb"] or 0) > .25 or (r["relB_reb"] or 0) > .25)]
for r in G2:
    print("  cbac17 L%d pool %d  b1-vs-reb %.3f  solo-vs-reb %.3f  b1-vs-solo %.3f same %s" % (r["layer"], r["pool"], r["relA_reb"], r["relB_reb"], r["rel"], r["same"]))
