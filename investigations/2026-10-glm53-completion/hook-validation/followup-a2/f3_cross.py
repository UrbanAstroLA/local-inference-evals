"""Follow-up a2, check 3 (CPU only): cross-session comparisons of the same window units. For every fixture, the
attempt-1 dumps (/run1: V4-b1 batch 8, V4-rebuild) against the attempt-2 dumps (/run: V4-b1, V4n-b1b, V4-rebuild,
V4n-rebuild2) over the same units (fixture, rank, layer, pool) as f1_units.py. Same image engine code in both
sessions (only the hook module differs). -> out/cross_units.jsonl"""
import glob, json, os, re
import torch

def snaps(run):
    S = {}
    for p in glob.glob(os.path.join(run, "forcewin", "snap_V4*_r*.pt")):
        m = re.search(r"snap_(.*)_F(\d+)(win)?_r(\d+)\.pt$", os.path.basename(p))
        if m.group(3):
            S.setdefault(m.group(1), {})[int(m.group(4))] = p
    return S

S1, S2 = snaps("/run1"), snaps("/run")

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
    return raw[rr * 128:(rr + 1) * 128].clone(), raw[8192 + rr * 4: 8192 + rr * 4 + 4].clone()

deq = lambda b: b[0].view(torch.float8_e4m3fn).float() * b[1].view(torch.float32)
def rel(x, y):
    if x is None or y is None:
        return None
    a, b = deq(x), deq(y)
    return float((a - b).norm() / max(float(b.norm()), 1e-12))
def same(x, y):
    return None if x is None or y is None else bool(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]))

ARMS = {"a1b1": (S1, "V4-b1-{}"), "a1reb": (S1, "V4-rebuild-{}"), "b1": (S2, "V4-b1-{}"), "b1b": (S2, "V4n-b1b-{}"),
        "reb": (S2, "V4-rebuild-{}"), "reb2": (S2, "V4n-rebuild2-{}")}
PAIRS = [("a1b1", "a1reb"), ("a1reb", "reb"), ("a1reb", "reb2"), ("a1b1", "b1"), ("a1b1", "b1b"), ("a1b1", "reb")]
tids = sorted({t.rsplit("-", 1)[-1] for t in S2 if t.startswith("V4-b1-")})
out = open("/f/out/cross_units.jsonl", "w")
for tid in tids:
    for rank in (0, 1):
        D = {a: torch.load(S[f.format(tid)][rank], weights_only=False) for a, (S, f) in ARMS.items()
             if f.format(tid) in S and rank in S[f.format(tid)]}
        assert D["a1b1"]["range"] == D["b1"]["range"] and D["a1b1"]["P"] == D["b1"]["P"], tid
        lo, hi = D["b1"]["range"]
        blk = {a: attn_blocks(D[a]) for a in D}
        for nm, L in D["b1"]["layers"].items():
            if L.get("kind") != "paged" or L.get("shape", [0])[-1:] != [132]:
                continue
            Ls = {a: D[a]["layers"].get(nm) for a in D}
            for pool in range((lo + 3) // 4, (hi - 1) // 4):
                X = {a: raw_pool(Ls[a]["pages"], blk[a], pool) for a in D if Ls[a] is not None}
                row = dict(tid=tid, rank=rank, layer=int(nm.split(".")[3]), pool=pool)
                for a, b in PAIRS:
                    if a in X and b in X:
                        row[f"{a}~{b}"] = rel(X[a], X[b]); row[f"eq:{a}~{b}"] = same(X[a], X[b])
                out.write(json.dumps(row) + "\n")
        print(tid, rank, flush=True)
