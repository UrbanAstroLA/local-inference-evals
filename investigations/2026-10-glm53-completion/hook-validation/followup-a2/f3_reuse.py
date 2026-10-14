"""Follow-up a2, check 3 side test: V4-b1-be2e9e349395 reused the physical blocks of the earlier injected request
V4-d2-be2e9e349395 (same group-0 block table). A pool the b1 decode failed to (re)write would still hold d2's bytes.
Compare b1 and d2 bytes at every window unit, for be2e9e349395 AND, as the control, for every fixture whose b1 and d2
requests used DIFFERENT physical blocks (equality there can only come from identical computation, not stale bytes).
Also b1b vs d2 (batches of 4 vs 8). -> out/f3_reuse.json"""
import json, collections, torch
def attn_blocks(rec):
    for g in rec["groups"]:
        if g["bpk"] > 1:
            ids = g["ids"].tolist(); bpk = g["bpk"]
            return lambda c, ids=ids, bpk=bpk: ids[c * bpk] // bpk if c * bpk < len(ids) else -1
def raw_pool(pages, blk_of, pool):
    blk = blk_of((4 * pool + 3) // 4352); pg = blk * 17 + (pool % 1088) // 64
    t = pages.get(pg)
    if t is None: return None
    raw = t.reshape(-1).view(torch.uint8); rr = pool % 64
    return raw[rr * 128:(rr + 1) * 128].clone(), raw[8192 + rr * 4: 8192 + rr * 4 + 4].clone(), pg
deq = lambda b: b[0].view(torch.float8_e4m3fn).float() * b[1].view(torch.float32)
TIDS = ["1e875e978fd4", "376073921446", "4b92a6a90ec4", "68a96f264430", "6d18c40f58df", "be2e9e349395",
        "cbac17c70af8", "ddfd287416a4"]
F = {"1e875e978fd4": 256, "376073921446": 455, "4b92a6a90ec4": 1723, "68a96f264430": 1226, "6d18c40f58df": 1069,
     "be2e9e349395": 2000, "cbac17c70af8": 2000, "ddfd287416a4": 1470}
res = {}
for tid in TIDS:
    r = dict(units=0, same_physical_pages=0, eq_b1_d2=0, eq_b1b_d2=0, eq_b1_d2_by_layer=collections.Counter(),
             eq_b1_d2_d2gross=0, eq_b1_d2_d2clean=0)
    for rank in (0, 1):
        A = {k: torch.load(f"/run/forcewin/snap_{t}-{tid}_F{F[tid]}win_r{rank}.pt", weights_only=False)
             for k, t in (("b1", "V4-b1"), ("b1b", "V4n-b1b"), ("d2", "V4-d2"))}
        R = torch.load(f"/run/forcewin/snap_V4-rebuild-{tid}_F1win_r{rank}.pt", weights_only=False)
        fb = {k: attn_blocks(v) for k, v in A.items()}; fr = attn_blocks(R)
        lo, hi = A["b1"]["range"]
        for nm, L in A["b1"]["layers"].items():
            if L.get("kind") != "paged" or L.get("shape", [0])[-1:] != [132]: continue
            lay = int(nm.split(".")[3])
            for pool in range((lo + 3) // 4, (hi - 1) // 4):
                x = {k: raw_pool(A[k]["layers"][nm]["pages"], fb[k], pool) for k in A}
                y = raw_pool(R["layers"][nm]["pages"], fr, pool)
                r["units"] += 1
                r["same_physical_pages"] += x["b1"][2] == x["d2"][2]
                e = torch.equal(x["b1"][0], x["d2"][0]) and torch.equal(x["b1"][1], x["d2"][1])
                r["eq_b1_d2"] += e
                r["eq_b1b_d2"] += torch.equal(x["b1b"][0], x["d2"][0]) and torch.equal(x["b1b"][1], x["d2"][1])
                if e:
                    r["eq_b1_d2_by_layer"][lay] += 1
                    dv, yv = deq(x["d2"]), deq(y)
                    g = float((dv - yv).norm() / max(float(yv.norm()), 1e-12)) > 0.25
                    r["eq_b1_d2_d2gross" if g else "eq_b1_d2_d2clean"] += 1
    r["eq_b1_d2_by_layer"] = dict(sorted(r["eq_b1_d2_by_layer"].items()))
    res[tid] = r
    print(tid, r, flush=True)
json.dump(res, open("/f/out/f3_reuse.json", "w"), indent=1)
