"""Per-pool decode-vs-rebuild comparison for V4 (H5b/H5e/H5d), with annotations. CPU only.
Writes out/h5_pools.jsonl (one row per (arm, tag, rank, layer, pool)) and prints summaries."""
import glob, json, os, re, sys
import numpy as np, torch
sys.path.insert(0, "/a")
RUN = "/run"
SNAP = {}
for p in glob.glob(os.path.join(RUN, "forcewin", "snap_V4-*_r*.pt")):
    m = re.search(r"snap_(.*)_F(\d+)(win)?_r(\d+)\.pt$", os.path.basename(p))
    SNAP.setdefault((m.group(1), int(m.group(2)), bool(m.group(3))), {})[int(m.group(4))] = p
CL = [json.loads(l) for l in open(os.path.join(RUN, "client.jsonl"))]
reqs = {r["tag"]: r for r in CL if r.get("kind") == "request"}

def attn_blocks(rec):
    for g in rec["groups"]:
        if g["bpk"] > 1:
            ids = g["ids"].tolist(); bpk = g["bpk"]
            return lambda c, ids=ids, bpk=bpk: ids[c * bpk] // bpk if c * bpk < len(ids) else -1
    return lambda c: -1

def raw_pool(pages, blk_of, pool):
    blk = blk_of((4 * pool + 3) // 4352)
    pg = blk * 17 + (pool % 1088) // 64
    t = pages.get(pg)
    if t is None:
        return None
    raw = t.reshape(-1).view(torch.uint8)
    rr = pool % 64
    v = raw[rr * 128:(rr + 1) * 128].view(torch.float8_e4m3fn).float()
    s = raw[8192 + rr * 4: 8192 + rr * 4 + 4].view(torch.float32)
    return v, float(s), raw[rr * 128:(rr + 1) * 128].clone()

out = open("/d/out/h5_pools.jsonl", "w")
for (tag, F, win), ranks in sorted(SNAP.items()):
    if not win or tag.startswith("V4-rebuild"):
        continue
    tid = tag.rsplit("-", 1)[-1]
    arm = tag.split("-")[1]
    reb = [(k, v) for k, v in SNAP.items() if k[0] == f"V4-rebuild-{tid}" and k[2]]
    for rank, pth in sorted(ranks.items()):
        a = torch.load(pth, weights_only=False)
        b = torch.load(reb[0][1][rank], weights_only=False)
        fa, fb = attn_blocks(a), attn_blocks(b)
        lo, hi = a["range"]
        P, Fn = a["P"], a["F"]
        rlo, rhi = b["range"]
        for name, La in a["layers"].items():
            if La.get("kind") != "paged" or La.get("shape", [0])[-1:] != [132]:
                continue
            Lb = b["layers"].get(name)
            for pool in range((lo + 3) // 4, (hi + 1) // 4):
                X = raw_pool(La["pages"], fa, pool); Y = raw_pool(Lb["pages"], fb, pool)
                if X is None or Y is None:
                    continue
                xv, xs, xr = X; yv, ys, yr = Y
                x, y = xv * xs, yv * ys
                rel = float((x - y).norm() / max(float(y.norm()), 1e-12))
                cos = float((x @ y) / max(float(x.norm() * y.norm()), 1e-12))
                out.write(json.dumps(dict(arm=arm, tag=tag, tid=tid, rank=rank, layer=int(name.split(".")[3]), pool=pool,
                    P=P, W=Fn, first=4 * pool, last=4 * pool + 3, frontier=a["frontier_pos"], rng=[lo, hi], rrng=[rlo, rhi],
                    rel=rel, cos=cos, xn=float(x.norm()), yn=float(y.norm()), xs=xs, ys=ys,
                    xmax=float(xv.abs().max()), ymax=float(yv.abs().max()),
                    bytes_eq=float((xr == yr).float().mean()), xzero=bool((xr == 0).all()),
                    yzero=bool((yr == 0).all()))) + "\n")
        print(tag, rank, "done", flush=True)
out.close()
