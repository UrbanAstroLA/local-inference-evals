#!/usr/bin/env python3
"""Synthetic run directories for a dry run of analysis/analyze_hookval.py (CPU; inside the B1 image for torch).

  make_synthetic_run.py OUT good   -> every frozen check should pass (VALID)
  make_synthetic_run.py OUT bad    -> planted defects: an off-by-one measured row, a doubly measured position, a
                                      corrupted B1-arm pool, a wrong committed KDA slot, a guard position violation
                                      -> INVALID with exactly those checks failing
The files have the hook's real formats (pack via forcewin_hooks.COLS; snapshot dict layout; guard poll layout).
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/hv/hook")
import forcewin_hooks as fw  # noqa: E402  (module only: ON is False here, nothing is hooked)

OUT, MODE = sys.argv[1], sys.argv[2]
BAD = MODE == "bad"
rng = np.random.default_rng(7 if not BAD else 8)
os.makedirs(os.path.join(OUT, "forcewin"), exist_ok=True)
os.makedirs(os.path.join(OUT, "mapguard"), exist_ok=True)
CL = open(os.path.join(OUT, "client.jsonl"), "w")
C = fw.C
K = 3


def w(d):
    CL.write(json.dumps(d) + "\n")


REF = {}


def ref_lp(key, W):
    if key not in REF:
        REF[key] = (rng.normal(-2.5, 1.5, W + 2).clip(-14, -1e-4), rng.integers(0, 150000, W + 2))
    return REF[key]


def metrics(key, W, noise=0.02, acc_pattern=None):
    lp, am = ref_lp(key, W)
    M = np.full((W, fw.NM), np.nan, dtype=np.float32)
    x = lp[:W] + rng.laplace(0, noise, W) * (rng.random(W) < 0.9) + rng.laplace(0, 0.4, W) * (rng.random(W) < 0.01)
    M[:, C["lp_pre_0"]:C["lp_pre_0"] + 4] = -12.0
    M[:, C["lp_pre_4"]] = np.minimum(x, -1e-5)
    for nm in ("mass_lt", "mass_le"):
        M[:, C[nm + "_0"]:C[nm + "_0"] + 5] = 0.1
    M[:, C["p_post_0"]:C["p_post_0"] + 5] = 0.0
    M[:, C["p_post_4"]] = np.exp(M[:, C["lp_pre_4"]])
    M[:, C["elig_lo_0"]:C["elig_lo_0"] + 5] = 0.0
    M[:, C["elig_hi_0"]:C["elig_hi_0"] + 5] = 0.0
    M[:, C["elig_lo_4"]] = 1.0
    M[:, C["elig_hi_4"]] = 1.0
    am2 = am[:W].copy()
    flip = rng.random(W) < 0.01
    am2[flip] = am2[flip] + 1
    M[:, C["argmax"]] = am2
    M[:, C["entropy"]] = 1.0
    M[:, C["post_ok"]] = 1.0
    return M


def acc_hist(mode, W):
    h = np.zeros(K + 2, dtype=np.int64)
    a = {"natural": None, "oracle": 3, "poison": 0, "reject:1": 0, "reject:2": 1, "reject:3": 2}.get(mode)
    if mode.startswith("sched"):
        h[: K + 1] = W // 10
    elif a is None:
        h[:2] = [W // 3, W // 4]
    else:
        h[a] = W // (a + 1)
    return torch.tensor(h)


def req(tag, W, mode="natural", key=None, ref=False, d2=0, P=200, pool_class=None, desig=True):
    Wm = W if not ref else W + 2
    M = metrics(key or tag, Wm, noise=0.0 if ref else 0.02)
    if ref:
        lp, am = ref_lp(key or tag, W)
        M[:, C["lp_pre_4"]] = lp[:Wm]
        M[:, C["argmax"]] = am[:Wm]
    nmeas = torch.ones(Wm, dtype=torch.int32)
    dsg = list(range(0, W, 16)) if (desig and not ref) else []
    ex = torch.full((len(dsg), 16, 8), -1, dtype=torch.int32)
    ex[:, :6, 0] = 1
    rec = dict(tag=tag, req=tag, P=P, W=W, mode=0, marg=0, d2=None, ref=None, cols=fw.COLS, xcols=fw.XCOLS,
               metrics=torch.tensor(M), nmeas=nmeas, acc_hist=acc_hist(mode, W), inv=torch.tensor([W, 0, 0, 0]),
               d2_injected_tokens=d2, pool_class=torch.tensor(pool_class if pool_class is not None else
                                                              np.zeros(W // 4 + 2), dtype=torch.int8),
               desig=dsg, exposure=ex, selpos=[], selections=torch.zeros(0), layer_slots={}, snaps=[], wall=1.0,
               errors=0)
    return rec


def save_req(rec, tag, both=True, mutate=None):
    for rank in ((0, 1) if both else (0,)):
        r = dict(rec)
        if mutate and rank == 0:
            r = mutate(r)
        torch.save(r, os.path.join(OUT, "forcewin", "req_%s_r%d.pt" % (fw._safe(tag), rank)))


SEED = [100]


def client_req(phase, tag, W, mode="natural", P=200, ids_equal=True):
    SEED[0] += 1
    w(dict(kind="request", phase=phase, tag=tag, mode=mode, seed=SEED[0], P=P, W=W, ids_equal=ids_equal,
           finish="length", error=None))


# ---------------------------------------------------------------- V0
for d in range(4):
    w(dict(kind="probe", id=f"q{d}", sha256="x", expected="x", equal=True))
# ---------------------------------------------------------------- V2a / V2c / V2b
MATRIX = ["t%d" % i for i in range(8)]
SCHED = ["natural", "oracle", "poison", "reject:1", "reject:2", "reject:3", "sched:20261013"]
for mode in SCHED:
    for tid in MATRIX:
        tag = f"V2a-{mode}-{tid}"
        W = 300
        rec = req(tag, W, mode, key=tid)
        mut = None
        if BAD and mode == "oracle" and tid == "t0":
            def mut(r):
                r = dict(r)
                m = r["metrics"].clone()
                m[1:, C["lp_pre_4"]] = m[:-1, C["lp_pre_4"]].clone()   # off by one (shifted rows)
                r["metrics"] = m
                n = r["nmeas"].clone()
                n[5] = 2
                r["nmeas"] = n
                return r
            rec = mut(rec)
            mut = None
        client_req("V2a", tag, W, mode)
        save_req(rec, tag, mutate=mut)
for tid in MATRIX:
    tag = f"V2c-{tid}"
    client_req("V2c", tag, 0)
    save_req(req(tag, 300, key=tid, ref=True), tag)
    tag = f"V2b-natural-{tid}"
    client_req("V2b", tag, 300)
    save_req(req(tag, 300, key=tid), tag)
# ---------------------------------------------------------------- V1 (free histories, B1's own logprobs)
for i in range(8):
    tag = f"free{i}"
    lp, am = ref_lp(tag, 400)
    lps = (lp[:400] + rng.laplace(0, 0.02, 400) * (rng.random(400) < 0.9)
           + rng.laplace(0, 0.4, 400) * (rng.random(400) < 0.01)).clip(max=-1e-5)
    w(dict(kind="free_hist", tag=tag, n=400, lps=lps.tolist()))
    for mode in ("natural", "poison"):
        t = f"V1-{mode}-{tag}"
        client_req("V1", t, 400, mode)
        save_req(req(t, 400, mode, key=tag), t)
    t = f"V1-ref-{tag}"
    client_req("V1-ref", t, 0)
    save_req(req(t, 400, key=tag, ref=True), t)


# ---------------------------------------------------------------- snapshots (V3) and V4 window dumps
def idx_page_set(blk, pools, values):
    """Pages (17 per block) holding the given pools' records (page-planar E4M3 + fp32 scales)."""
    pages = {}
    for pool, v in zip(pools, values):
        pg = blk * 17 + (pool % 1088) // 64
        t = pages.setdefault(pg, torch.zeros(64, 132, dtype=torch.uint8))
        raw = t.reshape(-1)
        rr = pool % 64
        q = torch.tensor(v, dtype=torch.float32).to(torch.float8_e4m3fn)
        raw[rr * 128:(rr + 1) * 128] = q.view(torch.uint8)
        raw[8192 + rr * 4: 8192 + rr * 4 + 4] = torch.tensor([1.0], dtype=torch.float32).view(torch.uint8)
    return pages


def groups(blk):
    return [dict(g=0, block_size=4352, kernel_block=64, bpk=68, ids=torch.tensor([blk * 68 + k for k in range(68)])),
            dict(g=1, block_size=8, kernel_block=8, bpk=1, ids=torch.tensor([5]))]


POOLV = {}


def pool_vals(key, pools):
    return [POOLV.setdefault((key, p), rng.normal(0, 1, 128).clip(-400, 400)) for p in pools]


def snap(tag, F, P, blk, kda_slots, na, pools, vals, rng_lo_hi=None, win=False, ints_ok=True, spec=True):
    rec = dict(tag=tag, F=F, P=P, frontier_pos=P + F - 2, num_computed=P + F - 1 if ints_ok else P + F,
               total_len=P + F, last_sampled=1, tokens_ok=True, num_accepted=na, groups=groups(blk),
               range=rng_lo_hi or [P + F - 66, P + F + 3], layers={})
    rec["layers"]["model.layers.0.attn"] = dict(kind="kda", is_spec=spec, slots=torch.arange(len(kda_slots)),
                                                ssm_sketch=torch.stack(kda_slots), num_accepted=na)
    rec["layers"]["model.layers.3.self_attn.indexer.k_cache"] = dict(
        kind="paged", shape=[1000, 64, 132], stride=[8448, 132, 1], dtype="torch.uint8",
        pages=idx_page_set(blk, pools, vals))
    for rank in (0, 1):
        torch.save(rec, os.path.join(OUT, "forcewin", "snap_%s_F%d%s_r%d.pt" % (fw._safe(tag), F,
                                                                                "win" if win else "", rank)))


for i, tid in enumerate(MATRIX):
    tag = f"V3-sched:7-{tid}"
    client_req("V3", tag, 300, "sched:7")
    save_req(req(tag, 300, "sched:7", key=tag), tag)
    F, P = 120 + i, 400
    ref = torch.randn(256)
    slots = [ref + 0.001 * torch.randn(256)] + [ref + 0.2 * torch.randn(256) for _ in range(3)]
    na = 1
    if BAD and i == 0:
        slots = [ref + 0.2 * torch.randn(256), ref + 0.001 * torch.randn(256)] + slots[2:]   # wrong slot committed
    pools = list(range((P + F - 2) // 4 - 8, (P + F - 1) // 4))
    vals = pool_vals(("v3", tid), pools)
    snap(tag, F, P, 30 + i, slots, na, pools, vals)
    rtag = f"V3ref-F{F}-{tag}"
    client_req("V3", rtag, 1)
    snap(rtag, 1, P + F - 1, 60 + i, [ref], 1, pools, [v + rng.normal(0, 0.01, 128) for v in vals], spec=False)
# V4
D2SET = ["u%d" % i for i in range(8)]
for arm in ("d2", "b1", "d2solo"):
    for i, tid in enumerate(D2SET if arm != "d2solo" else D2SET[:1]):
        tag = f"V4-{arm}-{tid}"
        W, P = 400, 128
        pools = list(range(P // 4, (P + W + 1) // 4))
        good = pool_vals(("v4", tid), pools)
        if arm == "d2":
            bad_mask = rng.random(len(pools)) < 0.85
        else:
            bad_mask = np.zeros(len(pools), dtype=bool)
        if BAD and arm == "b1" and i == 0:
            bad_mask[3] = True
        vals = [rng.normal(0, 1, 128) if b else v + rng.normal(0, 0.01, 128) for v, b in zip(good, bad_mask)]
        pc = np.zeros(W // 4 + 2, dtype=np.int8)
        for k, b in enumerate(bad_mask):
            pc[k] = (2 if b else 1) if rng.random() < 0.5 else 3
        client_req("V4", tag, W, P=P)
        save_req(req(tag, W, key=tag, d2=(W + 40) if arm != "b1" else 0, P=P, pool_class=pc), tag)
        snap(tag, W, P, 80 + i + (10 if arm == "b1" else 0) + (20 if arm == "d2solo" else 0),
             [torch.zeros(256)], 1, pools, vals, rng_lo_hi=[P, P + W], win=True)
for i, tid in enumerate(D2SET):
    tag = f"V4-rebuild-{tid}"
    W, P = 400, 128
    pools = list(range(P // 4, (P + W + 1) // 4))
    client_req("V4", tag, 1, P=P + W)
    save_req(req(tag, 1, key=tag, P=P + W, desig=False), tag)
    snap(tag, 1, P + W, 120 + i, [torch.zeros(256)], 1, pools, pool_vals(("v4", tid), pools),
         rng_lo_hi=[P, P + W], win=True)
# ---------------------------------------------------------------- guard polls
inj = 8 * 440 + 440
for pid in (11, 12):
    cnt = {s: {c: 0 for c in fw_c} for s, fw_c in
           (("tail_w_stash", ("mism", "null")), ("tail_w_seed", ("mism",)), ("idx_w_dec", ("mism",)),
            ("idx_w_pf", ("s_mism", "d7_fallback")), ("pos", ("pos_contig", "pos_bounds")))}
    cnt["tail_w_stash"]["mism"] = cnt["tail_w_stash"]["null"] = inj * (6 if pid == 11 else 5)
    if BAD and pid == 12:
        cnt["pos"]["pos_contig"] = 3
    with open(os.path.join(OUT, "mapguard", "mg_p%d.jsonl" % pid), "w") as f:
        f.write(json.dumps(dict(kind="poll", pid=pid, t=1.0, counters=cnt)) + "\n")
CL.close()
print("synthetic run written:", OUT, MODE)
