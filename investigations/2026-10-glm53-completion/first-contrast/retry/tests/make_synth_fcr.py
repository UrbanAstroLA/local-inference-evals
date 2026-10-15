#!/usr/bin/env python3
"""Synthetic RETRY run directories for dry runs of analysis/gate_fcr.py and analysis/analyze_fcr.py (CPU; B1 image).

The hook-format files come from attempt 1's validated generator ../../tests/make_synth_fc.py, executed UNCHANGED except
for the explicit source patches listed in PATCHES (each must apply exactly once; the file itself is not modified):
  P1  sentinel dumps cover the plan's dumprange [c - 4352, c) instead of attempt 1's [c0 - 1024, c0)   [R-5]
  P2  'touch' requests (unkeyed refresh passes) are client records only, like warms                   [R-3]
Added here: the guard's EngineCore scheduler log (mapguard/ms_p1.jsonl) consistent with the client records (engine id
= resp_id + "-0-s"; group-0 block ids restored from the latest earlier request of the same fixture and salt for the
first cached / 4352 blocks, fresh ids after; s2's build ids = the sentinel dumps' block ids), and the retry plants.

usage: make_synth_fcr.py PLAN_JSON OUT VARIANT
VARIANT: every attempt-1 variant (good, null, uninformative, noisy, g-*, a-*) plus
  r-cache-model   C1-M3 cached 0 (a safe miss)                          -> R-cache-model (GATE-1R stops)
  r-cache-unsafe  C1-L4 cached one block beyond its prefill coverage     -> R-cache-safe
  r-cache-attrib  C2-L5 scheduler num_computed != cached                 -> R-cache-safe (S4)
  r-ps-cover      PS-a-L4 restored nothing (sentinel coverage missing)   -> R-PS-coverage (and R-cache-model)
  r-ps-ident      PS-b-L1 restored a different physical shared block     -> R-PS-identity
  r-f6-page       two index-K pages of C1-L3 swapped (rank 0, all layers)-> R-idx-pages, R-idx-early
  r-balance       D1-A8 cached 0 after the gate (miss only in D)         -> GATE-1R PASS; R-cache-balance -> NO READING
"""
import json
import os
import sys

FC = "/fc"
SRC = open(os.path.join(FC, "tests", "make_synth_fc.py")).read()
PATCHES = [
    # P1: sentinel range from the plan
    ('        P, c0 = f["P"], f["c0"]\n        M = metrics_rows(n, ph, "C", 0, 1, lambda i: base + j, lambda i: 0)',
     '        P = f["P"]\n        lo_, hi_ = d["dumprange"]\n        c0 = hi_\n'
     '        M = metrics_rows(n, ph, "C", 0, 1, lambda i: base + j, lambda i: 0)'),
    ('for p in range(c0 - 1024, c0, 64)}', 'for p in range(lo_, hi_, 64)}'),
    ('        pools = list(range((c0 - 1024) // 4, c0 // 4))\n', '        pools = list(range(lo_ // 4, hi_ // 4))\n'),
    ('        save_dump(d["tag"], P, c0 - 1024, c0, n, "PS", groups, pools, extra_layers=extra, F=1)',
     '        save_dump(d["tag"], P, lo_, hi_, n, "PS", groups, pools, extra_layers=extra, F=1)'),
    # P2: the retry plan has no CV-before-gate index assumption problem; touch roles are records only (no change
    # needed: unknown roles fall through); the attempt-1 gate record after LC2 is harmless (analysis reads gate1r/)
]
V = sys.argv[3]
base_v = V if not V.startswith("r-") else "good"
for a, b in PATCHES:
    assert SRC.count(a) == 1, ("patch does not apply exactly once", a[:60])
    SRC = SRC.replace(a, b)
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], base_v]
g = {"__name__": "__synth__", "__file__": os.path.join(FC, "tests", "make_synth_fc.py")}
exec(compile(SRC, "make_synth_fc.py[patched]", "exec"), g)

# ------------------------------------------------------------------ retry additions
import torch  # noqa: E402

OUT = sys.argv[2]
BLOCK = 4352
CL = [json.loads(x) for x in open(os.path.join(OUT, "client.jsonl"))]
reqs = [r for r in CL if r.get("kind") == "request"]
# PS dumps' block ids for s2 (identity)
ps_ids = {}
for r in reqs:
    if r["phase"] == "PS-a":
        p = os.path.join(OUT, "forcewin", "snap_%s_F1win_r0.pt" % r["tag"])
        if os.path.exists(p):
            rec = torch.load(p, weights_only=False)
            gr = [x for x in rec["groups"] if x["bpk"] > 1][0]
            ids = gr["ids"].tolist()
            ps_ids[r["fixture"]] = [ids[i * gr["bpk"]] // gr["bpk"] for i in range(len(ids) // gr["bpk"])]
nxt = [900000]


def fresh(k):
    out = list(range(nxt[0], nxt[0] + k))
    nxt[0] += k
    return out


build_ids = {}
latest = {}
for r in reqs:
    key = (r["salt_name"], r["fixture"])
    nb_prompt = r["P"] // BLOCK
    if key not in build_ids:
        build_ids[key] = (ps_ids.get(r["fixture"]) if r["salt_name"] == "s2" and r["fixture"] in ps_ids else []) or []
    ids = build_ids[key]
    cached = r["cached_tokens"]
    # plants on the client side
    if V == "r-cache-model" and r["tag"] == "C1-M3":
        cached = 0
    if V == "r-cache-unsafe" and r["tag"] == "C1-L4":
        cached = r["cached_expected"] + BLOCK
    if V == "r-ps-cover" and r["tag"] == "PS-a-L4":
        cached = 0
    if V == "r-balance" and r["tag"] == "D1-A8":
        cached = 0
    r["cached_tokens"] = cached
    nb = cached // BLOCK
    total = (r["P"] + r["W"]) // BLOCK + 2
    g0 = []
    for i in range(total):
        if i < nb:
            g0.append(ids[i] if i < len(ids) else fresh(1)[0])          # restored: the first writer's block
        elif i < nb_prompt and i >= len(ids):
            ids.append(fresh(1)[0])                                     # this request's prefill writes it first
            g0.append(ids[i])
        elif i < nb_prompt and r["form"] in ("warm", "LW") and r["salt_name"] == "s2":
            g0.append(ids[i])                                           # s2's build = the sentinel dumps' blocks
        else:
            g0.append(fresh(1)[0])                                      # recomputed copy / decode-written block
    if V == "r-ps-ident" and r["tag"] == "PS-b-L1":
        g0[nb - 1] = 777777
    r["_g0"] = g0
    r["resp_id"] = "synth-%05d" % reqs.index(r)
    latest[key] = r
with open(os.path.join(OUT, "client.jsonl"), "w") as fh:
    for r in CL:
        r = {k: v for k, v in r.items() if k != "_g0"} if r.get("kind") == "request" else r
        if r.get("kind") == "request":
            r["cls"] = None
        fh.write(json.dumps(r) + "\n")
seq = 0
with open(os.path.join(OUT, "mapguard", "ms_p1.jsonl"), "w") as fh:
    fh.write(json.dumps(dict(kind="installed_scheduler", pid=1, t=0.0)) + "\n")
    for r in reqs:
        eid = r["resp_id"] + "-0-s"
        seq += 1
        nc = r["cached_tokens"]
        if V == "r-cache-attrib" and r["tag"] == "C2-L5":
            nc = r["cached_tokens"] - BLOCK
        lens = [len(r["_g0"]), 1, 4, 4, 4, 4, 4, 4, 4, 2]
        fh.write(json.dumps(dict(kind="sched", seq=seq, reqs={eid: [nc + 1, 1, 0]}, changed={eid: [r["_g0"], [7], lens]},
                                 finished=[], free_blocks=900, t=(r.get("t0") or 0) + 0.01)) + "\n")
        if r["W"]:
            seq += 1
            fh.write(json.dumps(dict(kind="sched", seq=seq, reqs={eid: [r["P"] + 4, 4, 3]}, changed={}, finished=[],
                                     free_blocks=900, t=(r.get("t0") or 0) + 0.02)) + "\n")
        seq += 1
        fh.write(json.dumps(dict(kind="sched", seq=seq, reqs={}, changed={}, finished=[eid], free_blocks=900,
                                 t=(r.get("t1") or 0) - 0.01)) + "\n")
# r-f6-page: swap two index-K pages of C1-L3 (rank 0, every index layer)
if V == "r-f6-page":
    sys.path.insert(0, os.path.join(FC, "analysis"))
    import fc_common as fc
    import fc_geometry as geo
    G = geo.load(FC)
    f = G["fx"]["L3"]
    lo = f["ref_pools"][0]
    p = os.path.join(OUT, "forcewin", "snap_C1-L3_F%dwin_r0.pt" % (f["W"] + 1))
    import glob
    p = sorted(glob.glob(os.path.join(OUT, "forcewin", "snap_C1-L3_F*win_r0.pt")))[0]
    rec = torch.load(p, weights_only=False)
    blk = fc.attn_blocks(rec)
    for nm, L in fc.idx_layers(rec).items():
        a, b = [blk((4 * q + 3) // 4352) * 17 + (q % 1088) // 64 for q in (lo + 640, lo + 704)]
        pg = L["pages"]
        if a in pg and b in pg:
            pg[a], pg[b] = pg[b], pg[a]
    torch.save(rec, p)
print("retry additions: scheduler log with %d records, plant %s" % (seq, V))
