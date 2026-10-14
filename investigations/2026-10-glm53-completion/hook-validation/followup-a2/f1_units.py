"""Follow-up a2, check 1 + 3 (CPU only). Recompute every index-K window unit of the V4/V4n comparisons directly from
the raw dumps, independently of analyze_hookval.py (helpers copied verbatim from analyze_hookval.py.a1: idx_pool,
attn_blocks, idx_layers; pool range from analyze_hookval.py compare_windows: range((lo+3)//4, (hi-1)//4) with
(lo, hi) = range of the FIRST dump of the pair).
Writes out/units.jsonl (one row per (fixture, rank, layer, pool)) and out/dump_meta.json."""
import glob, hashlib, json, os, re, collections
import torch

RUN = "/run"
SNAP = {}
for p in glob.glob(os.path.join(RUN, "forcewin", "snap_V4*_r*.pt")):
    m = re.search(r"snap_(.*)_F(\d+)(win)?_r(\d+)\.pt$", os.path.basename(p))
    if m.group(3):
        SNAP.setdefault(m.group(1), {})[int(m.group(4))] = p
CL = [json.loads(l) for l in open(os.path.join(RUN, "client.jsonl"))]
reqs = {r["tag"]: r for r in CL if r.get("kind") == "request"}
tids = sorted({t.rsplit("-", 1)[-1] for t in reqs if t.startswith("V4-b1-")})


def attn_blocks(rec):
    for g in rec["groups"]:
        if g["bpk"] > 1:
            ids = g["ids"].tolist(); bpk = g["bpk"]
            return lambda c, ids=ids, bpk=bpk: ids[c * bpk] // bpk if c * bpk < len(ids) else -1
    return lambda c: -1


def idx_layers(rec):
    return {n: L for n, L in rec["layers"].items() if L.get("kind") == "paged" and L.get("shape", [0])[-1:] == [132]}


def raw_pool(pages, blk_of, pool):
    blk = blk_of((4 * pool + 3) // 4352)
    pg = blk * 17 + (pool % 1088) // 64
    t = pages.get(pg)
    if t is None:
        return None
    raw = t.reshape(-1).view(torch.uint8)
    rr = pool % 64
    return raw[rr * 128:(rr + 1) * 128].clone(), raw[8192 + rr * 4: 8192 + rr * 4 + 4].clone(), pg


def deq(b):
    return b[0].view(torch.float8_e4m3fn).float() * b[1].view(torch.float32)


def rel(x, y):
    if x is None or y is None:
        return None
    a, b = deq(x), deq(y)
    return float((a - b).norm() / max(float(b.norm()), 1e-12))


def same(x, y):
    return None if x is None or y is None else bool(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]))


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 24), b""):
            h.update(ch)
    return h.hexdigest()


ARMS = {"b1": "V4-b1-{}", "b1b": "V4n-b1b-{}", "reb": "V4-rebuild-{}", "reb2": "V4n-rebuild2-{}", "d2": "V4-d2-{}",
        "solo": "V4-d2solo-{}"}
PAIRS = [("b1", "reb"), ("b1b", "reb"), ("b1", "b1b"), ("reb", "reb2"), ("b1", "reb2"), ("b1b", "reb2"),
         ("solo", "reb"), ("solo", "b1"), ("solo", "b1b"), ("d2", "reb")]
meta, out = {}, open("/f/out/units.jsonl", "w")
for tid in tids:
    for rank in (0, 1):
        D = {}
        for a, fmt in ARMS.items():
            tag = fmt.format(tid)
            if tag in SNAP and rank in SNAP[tag]:
                p = SNAP[tag][rank]
                rec = torch.load(p, weights_only=False)
                D[a] = rec
                g0 = [g for g in rec["groups"] if g["bpk"] > 1][0]
                meta[f"{tag}|r{rank}"] = dict(
                    file=os.path.basename(p), sha256=sha_file(p), tag=rec["tag"], P=rec["P"], F=rec["F"],
                    step=rec["step"], frontier_pos=rec["frontier_pos"], range=rec["range"],
                    num_accepted=rec["num_accepted"], num_computed=rec["num_computed"], total_len=rec["total_len"],
                    g0_ids_sha=hashlib.sha256(bytes(json.dumps(g0["ids"].tolist()), "ascii")).hexdigest()[:16],
                    g0_first_ids=g0["ids"].tolist()[:3:1] + g0["ids"].tolist()[68:69],
                    client=dict(seed=reqs[tag]["seed"], batch=reqs[tag].get("batch"), t0=reqs[tag]["t0"],
                                prefix_sha=reqs[tag]["prefix_sha"], forced_sha=reqs[tag]["forced_sha"],
                                dump_src_sha=reqs[tag].get("dump_src_sha"), dumprange=reqs[tag].get("dumprange")))
        lo, hi = D["b1"]["range"]
        fr = {a: D[a]["frontier_pos"] for a in D}
        blk = {a: attn_blocks(D[a]) for a in D}
        for nm, Lb1 in idx_layers(D["b1"]).items():
            lay = int(nm.split(".")[3])
            L = {a: D[a]["layers"].get(nm) for a in D}
            for pool in range((lo + 3) // 4, (hi - 1) // 4):
                X = {a: (raw_pool(L[a]["pages"], blk[a], pool) if L[a] is not None else None) for a in D}
                row = dict(tid=tid, rank=rank, layer=lay, pool=pool, first=4 * pool, P=D["b1"]["P"],
                           o=4 * pool - D["b1"]["P"], before_frontier=fr["b1"] - (4 * pool + 3))
                for a, b in PAIRS:
                    if a in X and b in X:
                        row[f"{a}~{b}"] = rel(X[a], X[b])
                        row[f"eq:{a}~{b}"] = same(X[a], X[b])
                row["page"] = {a: (X[a][2] if X[a] else None) for a in X}
                row["norm"] = {a: (float(deq(X[a]).norm()) if X[a] else None) for a in X}
                row["scale"] = {a: (float(X[a][1].view(torch.float32)) if X[a] else None) for a in X}
                out.write(json.dumps(row) + "\n")
        print(tid, rank, flush=True)
json.dump(meta, open("/f/out/dump_meta.json", "w"), indent=1)
