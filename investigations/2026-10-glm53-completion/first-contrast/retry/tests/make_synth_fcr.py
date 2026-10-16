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
[R-10..R-12, PLAN.md amendment 22f] (the sentinel's PS-c comparison is post-session, so GATE-1R passes for the R-10
plants and the analysis reads NO READING)
  r-ps-conv       PS-c-L4 rank 1, one KDA layer: one relevant conv entry changed (identical baseline)  -> R-PS-state-conv
  r-ps-ssm        PS-c-L4 rank 0, one KDA layer: the committed SSM state changed (identical baseline) -> R-PS-state-ssm
  r-ps-nonid      PS-b-L1 differs from PS-a in the SSM state only (rel 1e-3, every KDA layer), then PS-c-L1 changes its
                  conv entries by rel 5e-2 in one layer (the pre-22f rule passed this)              -> R-PS-state-conv
  r-ps-nonid-ok   PS-b-L1 differs in both components (rel 1e-3); PS-c-L1 SSM at 1.5 x, conv = PS-b's -> as good (clean)
  r-f6-nd         two index-K pages of LW2-L1 swapped (rank 0, one layer): the prefill null of those pages is 1.0, so
                  their limit is non-discriminating -> as good; the pages are reported "not independently verified by
                  the numerical check" (GATE-1R C1 / C2 and C3)
[R-17, PLAN.md amendment 22h]
  r-guard-stop    the client's batch guard stopped the session BEFORE C3 (rc 17): C3 has no record, no dump and no
                  scheduler entry; the client log carries the guard decisions, the stop, the coverage and the end -> GATE-1R
                  PASS; reading "incomplete contrast (batch guard)"; C3-residue NOT RUN, never a failure
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
# [R-10] sentinel boundary-state plants (PS-a / PS-b / PS-c non-speculative prefill snapshots, F1)
if V in ("r-ps-conv", "r-ps-ssm", "r-ps-nonid", "r-ps-nonid-ok"):
    gen = torch.Generator().manual_seed(7)

    def snap_path(tag, rank):
        return os.path.join(OUT, "forcewin", "snap_%s_F1_r%d.pt" % (tag, rank))

    def kda_names(rec):
        return sorted(n for n, L in rec["layers"].items() if L.get("kind") == "kda")

    def noisy(t, rel):
        x = t.float()
        nz = torch.randn(x.shape, generator=gen)
        return (x + nz * (rel * float(x.norm()) / max(float(nz.norm()), 1e-12))).to(t.dtype)

    def edit(tag, ranks, fn, layers=None):
        for rank in ranks:
            pth = snap_path(tag, rank)
            rec = torch.load(pth, weights_only=False)
            names = kda_names(rec)
            for nm in (layers(names) if layers else names):
                fn(rec["layers"][nm])
            torch.save(rec, pth)

    def conv_entry(L):
        c = L["conv"].clone()
        c[1, 5] = c[1, 5] + 0.5
        L["conv"] = c

    def ssm_entry(L):
        x = L["ssm_sketch"].clone()
        x[0, 9] = x[0, 9] * 1.5 + 0.25
        L["ssm_sketch"] = x

    def ssm_rel(rel):
        def f(L):
            x = L["ssm_sketch"].clone()
            x[0] = noisy(x[0], rel)
            L["ssm_sketch"] = x
        return f

    def conv_rel(rel):
        def f(L):
            c = L["conv"].clone()
            c[:3] = noisy(c[:3], rel)
            L["conv"] = c
        return f
    mid = lambda names: [names[len(names) // 2]]  # noqa: E731
    if V == "r-ps-conv":
        edit("PS-c-L4", (1,), conv_entry, mid)
    elif V == "r-ps-ssm":
        edit("PS-c-L4", (0,), ssm_entry, mid)
    elif V == "r-ps-nonid":
        edit("PS-b-L1", (0, 1), ssm_rel(1e-3))
        edit("PS-c-L1", (0, 1), ssm_rel(1e-3))
        edit("PS-c-L1", (0,), conv_rel(5e-2), mid)
    elif V == "r-ps-nonid-ok":
        # PS-b differs in both components; PS-c: SSM 1.5 x the baseline's difference (fp32: the ratio is exact), conv
        # equal to PS-b's (bf16 rounding of fresh noise on the synthetic 3-row conv makes a fresh-noise ratio random)
        edit("PS-b-L1", (0, 1), ssm_rel(1e-3))
        edit("PS-b-L1", (0, 1), conv_rel(1e-3))
        edit("PS-c-L1", (0, 1), ssm_rel(1.5e-3))
        for rank in (0, 1):
            rb = torch.load(snap_path("PS-b-L1", rank), weights_only=False)
            pc = snap_path("PS-c-L1", rank)
            rc_ = torch.load(pc, weights_only=False)
            for nm in kda_names(rc_):
                rc_["layers"][nm]["conv"] = rb["layers"][nm]["conv"].clone()
            torch.save(rc_, pc)
# [R-12] r-f6-nd: two index-K pages of LW2-L1 swapped (rank 0, the first index-K layer): a non-discriminating null
if V == "r-f6-nd":
    sys.path.insert(0, os.path.join(FC, "analysis"))
    import glob
    import fc_common as fc
    import fc_geometry as geo
    G = geo.load(FC)
    lo = G["fx"]["L1"]["ref_pools"][0]
    p = sorted(glob.glob(os.path.join(OUT, "forcewin", "snap_LW2-L1_F*win_r0.pt")))[0]
    rec = torch.load(p, weights_only=False)
    blk = fc.attn_blocks(rec)
    nm = sorted(fc.idx_layers(rec), key=fc.layer_no)[0]
    L = rec["layers"][nm]
    a, b = [blk((4 * q + 3) // 4352) * 17 + (q % 1088) // 64 for q in (lo + 640, lo + 704)]
    pg = L["pages"]
    assert a in pg and b in pg, "r-f6-nd: pages not in the dump"
    pg[a], pg[b] = pg[b], pg[a]
    torch.save(rec, p)
# [R-17] r-guard-stop: the batch guard stopped the session before C3 (nothing of C3 exists; the records end with the
# guard's decision, its stop (rc 17), the coverage and the end, as client_fcr.py writes them)
if V == "r-guard-stop":
    sys.path.insert(0, os.path.join(FC, "retry", "runner"))
    import glob
    import batch_guard_fcr as bg
    CL2 = [json.loads(x) for x in open(os.path.join(OUT, "client.jsonl"))]
    keep, phases_ran = [], []
    for r in CL2:
        if r.get("kind") in ("end", "coverage", "stop"):
            continue
        if r.get("kind") == "request" and r.get("phase") == "C3":
            continue
        if r.get("kind") == "phase" and r.get("tag") == "C3":
            continue
        if r.get("kind") == "phase":
            phases_ran.append(r["tag"])
        keep.append(r)
    t_end = max((r.get("t1") or r.get("t") or 0) for r in keep if isinstance(r.get("t1") or r.get("t"), (int, float)))
    g = dict(kind="guard", phase="C3", action="stop", projected=812.4, reserve=0.0, left=640.0,
             why="batch guard: projected completion of C3 (812 s) does not fit the 640 s left before the deadline")
    keep.append(g)
    keep.append(dict(kind="stop", why="%s: %s" % (bg.READING, g["why"]), rc=bg.GUARD_RC, phases=phases_ran,
                     t=t_end + 1))
    start = next(r for r in keep if r.get("kind") == "start")
    req = start.get("phases") or phases_ran + ["C3"]
    keep.append(dict(kind="coverage", required=req, ran=[p_ for p_ in req if p_ in phases_ran],
                     not_run=[p_ for p_ in req if p_ not in phases_ran], order=phases_ran, left=640.0))
    keep.append(dict(kind="end", t=t_end + 2, rc=bg.GUARD_RC))
    with open(os.path.join(OUT, "client.jsonl"), "w") as fh:
        for r in keep:
            fh.write(json.dumps(r) + "\n")
    for f_ in glob.glob(os.path.join(OUT, "forcewin", "*C3-*")):
        os.remove(f_)
    msp = os.path.join(OUT, "mapguard", "ms_p1.jsonl")
    c3ids = {r["resp_id"] + "-0-s" for r in reqs if r["phase"] == "C3"}
    lines_ = [x for x in open(msp) if not any(e in x for e in c3ids)]
    open(msp, "w").writelines(lines_)
print("retry additions: scheduler log with %d records, plant %s" % (seq, V))
