#!/usr/bin/env python3
"""Synthetic first-contrast run directories for dry runs of analysis/gate_fc.py and analysis/analyze_fc.py (CPU; in the
B1 image for torch). The request list is the client's own --plan output (runner/client_fc.py), so every tag, salt,
length, key and expected cached value is the session's. Files carry the hook's real formats (forcewin_hooks COLS /
XCOLS, request record and snapshot dict layout, guard poll layout), as hook-validation/tests/make_synthetic_run.py.

usage: make_synth_fc.py PLAN_JSON OUT VARIANT
VARIANT (each plant aims at one check; "good" holds attempt-2-like non-defects: ~1 % fragile deep-layer pools in every
build, decode noise, 1 % argmax flips, padded steps at window ends, near-close ramps):
  good           D2 lowers close propensity by 0.6 nats, keys change           -> GATE-1 PASS; reading case 1
  null           no termination effect, keys change                            -> GATE-1 PASS; reading case 2
  uninformative  injected runs select no corrupted pool                        -> reading case 3
  noisy          per-run offsets of up to 0.45 nats common to all fixtures      -> NO READING (N > 0.25)
  g-wmax         rank-1 init wmax 5120 and C1-L1 truncated to 5121 ids        -> G1-wmax (and G1-forcing)
  g-abspos       C2-M3 hook prompt length relative to the cached prefix        -> G1-abspos
  g-cache        C1-L4 cached_tokens 0 (cold) though planned c0                 -> G1-cache
  g-attrib       exposure steps misattributed in PADDED steps only (C2)        -> G1-attrib
  g-align        C1-L4 rows shifted by one output index                        -> G1-align (and G1-cross)
  g-cross        argmax/lp disagreement only around L3's second crossing (C1)  -> G1-cross only
  g-regime       a non-causal pool id in M3's selection records near 2,048     -> G1-regime
  g-content      26 deep-layer pools wrong in BOTH C1 and C2 for M4 and L5      -> G1-content (and flags)
  g-clean        a corrupt pool counted and selected in C2-L8                  -> G1-clean
  g-cvmig        oracle L5: conv rows shifted by one position after the column change (wrong migration offset)
                                                                               -> G1-CV-persist, G1-CV-conv
  g-cvslot       natural L1: spec slots of the old column after the crossing   -> G1-CV-slot
  g-cvbook       poison L5: num_accepted 2 where the hook accepted 0          -> G1-CV-book
  a-ps           PS-c: one MLA page byte of the shared cached block changed    -> PS-prefix -> NO READING
  a-guard        a mismatched tail write attributed to C3                      -> H5g-fc -> NO READING
  a-incomplete   D3-A8 ids_equal False                                         -> INCOMPLETE
  a-c3           C3 rows shifted by one output index (residue)                 -> C3-residue -> NO READING
"""
import gzip
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/hv/hook")
sys.path.insert(0, "/fc/analysis")
import forcewin_hooks as fw  # noqa: E402  (module only; ON is False here)
import fc_geometry as geo  # noqa: E402

PLANF, OUT, V = sys.argv[1], sys.argv[2], sys.argv[3]
G = geo.load("/fc")
ORDER = G["order"]
PLAN = [d for d in json.load(open(PLANF)) if d.get("kind") == "request"]
rng = np.random.default_rng(23)
os.makedirs(os.path.join(OUT, "forcewin"), exist_ok=True)
os.makedirs(os.path.join(OUT, "mapguard"), exist_ok=True)
CLF = open(os.path.join(OUT, "client.jsonl"), "w")
C = fw.C
T = [1000.0]
STEP = [100]
IDXL = {0: (3, 7, 19), 1: (27, 39)}           # DSA index layers per rank (synthetic split)
KDAL = (0, 1, 2, 10, 20, 30)                   # KDA layers (synthetic subset)
EFFECT = {"good": -0.6, "uninformative": -0.6, "noisy": 0.0}.get(V, -0.6 if V.startswith(("g-", "a-")) else 0.0)
RUN_SD = 0.01
ADDS, STEPS, POLLS = {0: [], 1: []}, [], {0: [], 1: []}


def now(dt=1.0):
    T[0] += dt
    return round(T[0], 3)


def w(d):
    CLF.write(json.dumps(d) + "\n")


DUMPS = []
CUM = {0: dict(mism=0, null=0), 1: dict(mism=0, null=0)}
PHASE_INJ = {}


def dump(tag):
    t = now(0.1)
    w(dict(kind="dump", tag=tag, t=t, rc=0))
    DUMPS.append((tag, t))
    inj = PHASE_INJ.get(tag, 0)
    if V == "a-guard" and tag == "C3":
        inj = 1
    for rank in (0, 1):
        nl = len(IDXL[rank])
        CUM[rank]["mism"] += nl * inj
        CUM[rank]["null"] += nl * inj
        POLLS[rank].append(dict(kind="poll", rank=rank, reason="trigger", t=t + 0.5, layers={str(L): 1 for L in IDXL[rank]},
                                counters=dict(tail_w_stash=dict(CUM[rank]), idx_w_dec=dict(mism=0),
                                              tail_w_seed=dict(mism=0), idx_w_pf=dict(s_mism=0, d7_fallback=0),
                                              pos=dict(pos_contig=0, pos_bounds=0))))


def phase(tag):
    t1 = now(5.0)
    w(dict(kind="phase", tag=tag, t0=t1 - 5.0, t1=t1, secs=5.0))
    dump(tag)


# ------------------------------------------------------------------ world
W_ = {}


def world(n):
    if n in W_:
        return W_[n]
    f = G["fx"][n]
    L = f["L"] + 8200
    lp0 = rng.normal(-13.0, 3.0, L).clip(-30, -1e-3)
    if f["near_close"]:
        lp0[f["W"] - 63: f["W"] + 1] = np.linspace(-6.0, -0.4, 64)
    W_[n] = dict(lp0=lp0, lp4=-np.abs(rng.normal(0, 1.5, L)) - 1e-4, am=rng.integers(0, 150000, L),
                 lps=rng.normal(-20, 2, (3, L)), off={})
    return W_[n]


NOISY_RUN = {"C1": 0.0, "C2": 0.45, "C3": -0.3, "D1": 0.2, "D2": -0.25, "D3": 0.4}


def run_offset(ph, n):
    if V == "noisy":                     # a common per-run offset: the run aggregate itself varies (N > 0.25)
        return NOISY_RUN.get(ph, 0.0)
    k = (ph, n)
    w_ = world(n)
    if k not in w_["off"]:
        w_["off"][k] = rng.normal(0, RUN_SD)
    return w_["off"][k]


def metrics_rows(n, ph, arm, o_lo, nrows, step_of, acc_of, ref=False, shift=0):
    """rows for output indices o_lo .. o_lo + nrows - 1 (o = absolute output index of the long window)."""
    w_ = world(n)
    M = np.full((nrows, fw.NM), np.nan, dtype=np.float32)
    o = np.arange(o_lo, o_lo + nrows) + shift
    o = np.clip(o, 0, len(w_["lp0"]) - 1)
    lp0 = w_["lp0"][o].copy()
    if not ref:
        lp0 = lp0 + run_offset(ph, n) + rng.normal(0, 0.02, nrows)
        if arm in ("D", "LD"):
            lp0 = lp0 + EFFECT
    lp0 = np.minimum(lp0, -1e-5)
    M[:, C["lp_pre_0"]] = lp0
    M[:, C["lp_pre_1"]:C["lp_pre_1"] + 3] = w_["lps"][:, o].T
    lp4 = w_["lp4"][o] + (0 if ref else rng.laplace(0, 0.004, nrows))
    M[:, C["lp_pre_4"]] = np.minimum(lp4, -1e-6)
    am = w_["am"][o].copy()
    if not ref:
        fl = rng.random(nrows) < 0.01
        am[fl] += 1
    M[:, C["argmax"]] = am
    for nm in ("mass_lt", "mass_le"):
        M[:, C[nm + "_0"]:C[nm + "_0"] + 5] = 0.1
    elig = (lp0 > math.log(0.05)).astype(np.float32)
    M[:, C["p_post_0"]] = np.exp(lp0) * elig
    M[:, C["p_post_1"]:C["p_post_1"] + 3] = 0.0
    M[:, C["p_post_4"]] = np.exp(M[:, C["lp_pre_4"]])
    M[:, C["elig_lo_0"]:C["elig_lo_0"] + 5] = 0.0
    M[:, C["elig_hi_0"]:C["elig_hi_0"] + 5] = 0.0
    M[:, C["elig_lo_0"]] = elig
    M[:, C["elig_hi_0"]] = elig
    M[:, C["elig_lo_4"]] = 1.0
    M[:, C["elig_hi_4"]] = 1.0
    M[:, C["entropy"]] = 1.0
    M[:, C["post_ok"]] = 1.0
    M[:, C["step"]] = [step_of(i) for i in range(nrows)]
    M[:, C["accepted"]] = [acc_of(i) for i in range(nrows)]
    return M


# ------------------------------------------------------------------ index-K pages
POOLV = {}


def base_vals(n, L, pools):
    out = np.empty((len(pools), 128), dtype=np.float32)
    for i, p in enumerate(pools):
        k = (n, L, p)
        if k not in POOLV:
            POOLV[k] = rng.normal(0, 1, 128).astype(np.float32)
        out[i] = POOLV[k]
    return out


FRAG = {}


def build_vals(n, L, pools, kind, ph):
    v = base_vals(n, L, pools) + rng.normal(0, 0.01, (len(pools), 128)).astype(np.float32)
    if L >= 19:            # ~1 % fragile deep-layer pools: every build differs (attempt-1/2 deep-layer fragility)
        fr = FRAG.setdefault((n, L), {p for p in range(G["fx"][n]["P"] // 4, G["fx"][n]["P"] // 4 + 2200)
                                      if rng.random() < 0.01})
        for i, p in enumerate(pools):
            if p in fr:
                v[i] = rng.normal(0, 1, 128)
    if V == "g-content" and ph in ("C1", "C2") and n in ("M4", "L5") and L in (27, 39):
        lo, hi = G["fx"][n]["ref_pools"]
        for i, p in enumerate(pools):        # every 5th pool of the reference range: the same wrong content in BOTH
            if lo <= p < hi and (p - lo) % 5 == 2:          # clean decode arms (a defect common to C1 and C2)
                v[i] = -base_vals(n, L, [p])[0]
    return v


def idx_pages(ids0, pools, vals):
    pages = {}
    q = torch.tensor(vals, dtype=torch.float32).clamp(-440, 440).to(torch.float8_e4m3fn).view(torch.uint8)
    for i, pool in enumerate(pools):
        blk = ids0[((4 * pool + 3) // 4352) * 68] // 68
        pg = blk * 17 + (pool % 1088) // 64
        t = pages.get(pg)
        if t is None:
            t = pages[pg] = torch.zeros(64, 132, dtype=torch.uint8)
        raw = t.reshape(-1)
        rr = pool % 64
        raw[rr * 128:(rr + 1) * 128] = q[i]
        raw[8192 + rr * 4: 8192 + rr * 4 + 4] = torch.tensor([1.0], dtype=torch.float32).view(torch.uint8)
    return pages


BLK = [1000]


def new_groups(total_len, ncols=None, share=None):
    """g0: attention (4,352-token blocks, 64-token kernel pages, ids expanded x68); g9: 1,088-token blocks;
    g2/g3: mamba groups (one id per column + 4 spare)."""
    nb = total_len // 4352 + 2
    if share is not None:
        return share
    blocks = []
    for _ in range(nb):
        BLK[0] += 1
        blocks.append(BLK[0])
    g0 = dict(g=0, block_size=4352, kernel_block=64, bpk=68, ids=torch.tensor([b * 68 + k for b in blocks for k in range(68)]))
    g1 = dict(g=1, block_size=8, kernel_block=8, bpk=1, ids=torch.tensor([7]))
    g9 = dict(g=9, block_size=1088, kernel_block=1088, bpk=1, ids=torch.tensor([b * 4 + k for b in blocks for k in range(4)]))
    m = []
    for g in (2, 3):
        m.append(dict(g=g, block_size=4352, kernel_block=4352, bpk=1,
                      ids=torch.tensor([100000 * g + BLK[0] * 64 + j for j in range(nb + 4)])))
    return [g0, g1] + m + [g9]


def save_dump(tag, P, rng_lo, rng_hi, n, ph, groups, pools, extra_layers=None, frontier=None, F=1, step=0):
    for rank in (0, 1):
        rec = dict(tag=tag, F=F, P=P, step=step, frontier_pos=frontier if frontier is not None else P + F - 2,
                   num_computed=P + F - 1, total_len=P + F, last_sampled=1, tokens_ok=True, num_accepted=1,
                   groups=groups, range=[rng_lo, rng_hi], layers={})
        ids0 = groups[0]["ids"].tolist()
        for L in IDXL[rank]:
            vals = build_vals(n, L, pools, "x", ph)
            rec["layers"]["language_model.model.layers.%d.self_attn.indexer.k_cache" % L] = dict(
                type="Glm5NextIndexerCache", kind="paged", shape=[50000, 64, 132], stride=[8448, 132, 1],
                dtype="torch.uint8", pages=idx_pages(ids0, pools, vals))
        if extra_layers:
            rec["layers"].update(extra_layers(rank))
        torch.save(rec, os.path.join(OUT, "forcewin", "snap_%s_F%dwin_r%d.pt" % (fw._safe(tag), F, rank)))


# ------------------------------------------------------------------ request records
def save_req(tag, P, W, M, nmeas, desig, ex, selpos, sel, d2inj=0, pc=None, ref=None, F_list=None):
    for rank in (0, 1):
        rec = dict(tag=tag, req=tag, P=P, W=W, mode=0, marg=0, d2=None, ref=ref, cols=fw.COLS, xcols=fw.XCOLS,
                   metrics=torch.tensor(M), nmeas=torch.tensor(nmeas, dtype=torch.int32),
                   acc_hist=torch.zeros(5, dtype=torch.int64), inv=torch.tensor([W, 0, 0, 0]),
                   d2_injected_tokens=d2inj,
                   pool_class=torch.tensor(pc if pc is not None else np.zeros(max(W, 1) // 4 + 2), dtype=torch.int8),
                   desig=desig, exposure=ex, selpos=selpos, selections=sel,
                   layer_slots={"language_model.model.layers.%d.self_attn.indexer.k_cache" % L: i
                                for i, L in enumerate(IDXL[0] + IDXL[1])}, snaps=[], wall=1.0, errors=0)
        torch.save(rec, os.path.join(OUT, "forcewin", "req_%s_r%d.pt" % (fw._safe(tag), rank)))


def add_rec(d, W_hook=None, P_hook=None):
    for rank in (0, 1):
        ADDS[rank].append(dict(kind="add", req=d["tag"], idx=0, tag=d["tag"], P=P_hook if (P_hook is not None and
                                                                                        rank == 0) else d["P"],
                               W=W_hook if W_hook is not None else d["W"], mode=d.get("mode"), d2=d.get("d2"),
                               snap=d.get("snap"), ref=d.get("ref_from"), P_mod8=d["P"] % 8, t=now(0.001), rank=rank))


SELBASE = {}


def selections(n, ph, arm, selpos, P, o_off, step_of_o):
    sel = torch.full((len(selpos), 17, 512), -1, dtype=torch.int32)
    for k, ol in enumerate(selpos):
        o = ol + o_off
        for L in range(len(IDXL[0]) + len(IDXL[1])):
            key = (n, o, L)
            hi = (G["fx"][n]["P"] + o) // 4
            if key not in SELBASE:
                SELBASE[key] = rng.choice(hi - 600, 511, replace=False)
            s = SELBASE[key].copy()
            sw = rng.choice(511, 2, replace=False)
            s[sw] = rng.integers(0, hi - 600, 2)
            if arm in ("D", "LD") and V != "uninformative":
                s[:60] = hi - 1 - np.arange(60)            # the same corrupted recent pools in every D run
            if V == "g-regime" and ph == "C1" and n == "M3" and abs(o - 2048) < 256:
                s[0] = hi + 50                              # non-causal id
            if len(set(s.tolist())) < 511:
                s = np.unique(s)
                s = np.concatenate([s, np.setdiff1d(np.arange(hi - 600), s)[: 511 - s.size]])
            sel[k, L, 0] = step_of_o(ol)
            sel[k, L, 1:512] = torch.tensor(s[:511], dtype=torch.int32)
    return sel


def exposure(desig, step_of_o, arm, ph, W, P_off):
    ex = torch.full((len(desig), 17, 8), -1, dtype=torch.int32)
    nl = len(IDXL[0]) + len(IDXL[1])
    for k, ol in enumerate(desig):
        if ol < 1 or ol >= W:
            continue
        st = step_of_o(ol)
        if V == "g-attrib" and ph == "C2" and STEPS_PAD.get(st):
            st += 1
        nwin = min(511, (ol + P_off) // 4)
        nsel_win = min(nwin, 120)
        corrupt = arm in ("D", "LD") and V != "uninformative"
        ex[k, :nl, 0] = st
        ex[k, :nl, 1] = 511
        ex[k, :nl, 2] = nsel_win
        ex[k, :nl, 3] = int(0.5 * nsel_win) if corrupt else 0
        ex[k, :nl, 5] = nwin
        ex[k, :nl, 6] = int(0.6 * nwin) if corrupt else 0
    return ex


STEPS_PAD = {}


def run_phase(ph, reqs):
    """One forced run of 8: decode rows ~3 tokens per step; the last ~6 % of each window in padded steps."""
    base = STEP[0]
    maxW = max(d["W"] for d in reqs)
    nsteps = maxW // 3 + 2
    for s in range(nsteps):
        pad = 4 if s > 0.94 * nsteps else 0
        STEPS.append([base + s, 8 if not pad else 7, 32 if not pad else 28, 24, 0, pad])
        STEPS_PAD[base + s] = pad > 0
    STEP[0] += nsteps + 5
    for d in reqs:
        n = d["fixture"]
        f = G["fx"][n]
        role = d["role"]
        short = role in ("LC", "LD")
        arm = "D" if role in ("D", "LD") else "C"
        o_off = geo.LP_SHIFT if short else 0
        W = d["W"]
        trunc = V == "g-wmax" and ph == "C1" and n == "L1"
        Wh = 5121 if trunc else W
        step_of = lambda i: base + i // 3  # noqa: E731
        shift = 1 if ((V == "g-align" and ph == "C1" and n == "L4") or (V == "a-c3" and ph == "C3")) else 0
        M = metrics_rows(n, ph, role, o_off, W, step_of, lambda i: 2, shift=shift)
        if V == "g-cross" and ph == "C1" and n == "L3":
            ob = f["crossings_o"][1]
            M[ob - 128: ob + 128, C["argmax"]] += 7
        nmeas = np.ones(W, dtype=np.int32)
        if trunc:
            nmeas[Wh:] = 0
        desig = f["desig_lp"] if short else f["desig"]
        selpos = f["selrec_lp"] if short else f["selrec"]
        ex = exposure(desig, step_of, role, ph, W, d["P"] - f["P"])
        if V == "g-clean" and ph == "C2" and n == "L8":
            ex[5, 0, 3] = 2
            ex[5, 0, 6] = 3
        sel = selections(n, ph, role, selpos, d["P"], o_off, step_of)
        P = d["P"]
        pc = np.zeros(W // 4 + 2, dtype=np.int8)
        for i in range((P + 3) // 4, (P + W - 1) // 4):
            pc[i - P // 4] = 2 if (arm == "D" and rng.random() < 0.6) else 1
        save_req(d["tag"], P, Wh, M, nmeas, desig, ex, selpos, sel, d2inj=(W + 30) if arm == "D" else 0, pc=pc)
        PHASE_INJ[ph] = PHASE_INJ.get(ph, 0) + ((W + 30) if arm == "D" else 0)
        add_rec(d, W_hook=Wh, P_hook=(P - f["c0"]) if (V == "g-abspos" and ph == "C2" and n == "M3") else None)
        groups = new_groups(P + W)
        pools = list(range(P // 4, (P + W) // 4 + 1))
        if not short:
            save_dump(d["tag"], P, P, P + W, n, ph if arm == "C" else "D", groups, pools, F=W)
            if arm == "D":
                _corrupt_dump(d["tag"], pc, P, W)
        else:
            save_dump(d["tag"], P, P, P + W, n, ph, groups, pools[: 300], F=W)


def _corrupt_dump(tag, pc, P, W):
    """D runs: corrupt-labelled pools get unrelated content in the dump (gross)."""
    for rank in (0, 1):
        p = os.path.join(OUT, "forcewin", "snap_%s_F%dwin_r%d.pt" % (fw._safe(tag), W, rank))
        rec = torch.load(p, weights_only=False)
        ids0 = rec["groups"][0]["ids"].tolist()
        for nm, L in rec["layers"].items():
            for k in range(pc.size):
                if pc[k] != 2:
                    continue
                pool = P // 4 + k
                blk = ids0[((4 * pool + 3) // 4352) * 68] // 68
                pg = blk * 17 + (pool % 1088) // 64
                t = L["pages"].get(pg)
                if t is None:
                    continue
                raw = t.reshape(-1)
                rr = pool % 64
                raw[rr * 128:(rr + 1) * 128] = torch.tensor(rng.normal(0, 1, 128), dtype=torch.float32).to(
                    torch.float8_e4m3fn).view(torch.uint8)
        torch.save(rec, p)


def ref_phase(ph, reqs):
    base = STEP[0]
    STEP[0] += 40
    for d in reqs:
        n = d["fixture"]
        f = G["fx"][n]
        nrow = d["build_len"] - f["P"] + 1
        M = np.full((geo.WMAX + 1, fw.NM), np.nan, dtype=np.float32)
        M[:nrow] = metrics_rows(n, ph, "ref", 0, nrow, lambda i: base + i // 2048, lambda i: -1, ref=True)
        M[nrow - 1, C["lp_pre_4"]] = np.nan
        nm = np.zeros(geo.WMAX + 1, dtype=np.int32)
        nm[:nrow] = 1
        save_req(d["tag"], d["P"], 0, M, nm, [], torch.zeros(0, 17, 8, dtype=torch.int32), [],
                 torch.zeros(0, 17, 512, dtype=torch.int32), ref=f["P"])
        add_rec(d)
        groups = new_groups(d["P"])
        pools = list(range(f["P"] // 4, f["P_L"] // 4 + 1))
        save_dump(d["tag"], d["P"], f["P"], f["P_L"], n, ph, groups, pools, F=1)


# ------------------------------------------------------------------ CV and PS
CONV = {}
SSM = {}
D_CONV = 16


def conv_true(n, L, pos):
    k = (n, L, pos)
    if k not in CONV:
        CONV[k] = rng.normal(0, 1, D_CONV).astype(np.float32)
    return CONV[k]


def ssm_true(n, L, pos):
    k = (n, L, pos)
    if k not in SSM:
        prev = SSM.get((n, L, pos - 1))
        base = prev if prev is not None else rng.normal(0, 1, 256).astype(np.float32)
        SSM[k] = (base + 0.3 * rng.normal(0, 1, 256)).astype(np.float32)
    return SSM[k]


def kda_noise(L):
    return 0.002 if L <= 2 else 0.01 if L <= 17 else 0.03


def cv_phase(reqs):
    base = STEP[0]
    STEP[0] += 200
    for d in reqs:
        n, m = d["fixture"], d["cv_mode"]
        f = G["fx"][n]
        Pc, B = f["cv"]["P_cv"], f["cv"]["B"]
        targets = f["cv"]["targets"][m]
        Fs, cs = [1], [1]
        while Fs[-1] < d["W"]:
            c = {"poison": 1, "oracle": 4, "reject:2": 2}.get(m) or int(rng.integers(1, 5))
            c = min(c, d["W"] - Fs[-1])
            Fs.append(Fs[-1] + c)
            cs.append(c)
        rowstep, rowacc = np.zeros(d["W"]), np.zeros(d["W"])
        for k in range(1, len(Fs)):
            rowstep[Fs[k - 1]:Fs[k]] = base + k
            rowacc[Fs[k - 1]:Fs[k]] = cs[k] - 1
        rowstep[0] = base
        M = metrics_rows(n, "CV", "C", 0, d["W"], lambda i: rowstep[i], lambda i: rowacc[i])
        save_req(d["tag"], Pc, d["W"], M, np.ones(d["W"], np.int32), [], torch.zeros(0, 17, 8, dtype=torch.int32),
                 [], torch.zeros(0, 17, 512, dtype=torch.int32))
        add_rec(d)
        groups = new_groups(Pc + 80)
        noise = {}
        done = set()
        for t in targets:
            k = next(i for i, F in enumerate(Fs) if F >= t)
            F = Fs[k]
            if F in done:
                continue
            done.add(F)
            a = cs[k] - 1
            fr = Pc + F - 2
            p0 = fr - a
            col = (p0 + 3) // 4352
            na = a + 1
            if V == "g-cvbook" and m == "poison" and n == "L5" and t == targets[2]:
                na = 2
            rec = dict(tag=d["tag"], F=F, P=Pc, step=int(base + k), frontier_pos=fr, num_computed=fr + 1,
                       total_len=Pc + F, last_sampled=1, tokens_ok=True, num_accepted=na, groups=groups,
                       range=[fr - 64, fr + 5], layers={})
            for rank in (0, 1):
                for i, L in enumerate(KDAL):
                    g = groups[2 + (i % 2)]
                    c_ = col - 1 if (V == "g-cvslot" and m == "natural" and n == "L1" and fr >= B) else col
                    slots = g["ids"][c_: c_ + 4].clone()
                    rows = []
                    for e in range(6):
                        pos = p0 - 2 + e
                        if V == "g-cvmig" and m == "oracle" and n == "L5" and col == B // 4352:
                            pos = pos - 1
                        if e <= a + 2:
                            key = (pos, L, rank)
                            if key not in noise:
                                noise[key] = rng.normal(0, kda_noise(L), D_CONV).astype(np.float32)
                            rows.append(conv_true(n, L, pos) * (1 + noise[key]))
                        else:
                            rows.append(rng.normal(0, 1, D_CONV).astype(np.float32))
                    sk = [ssm_true(n, L, p0 + j) * (1 + rng.normal(0, kda_noise(L), 256)).astype(np.float32)
                          for j in range(4)]
                    rec["layers"]["language_model.model.layers.%d.self_attn" % L] = dict(
                        type="Glm5NextLinearAttention", kind="kda", is_spec=True, spec_rank=0, slots=slots,
                        num_accepted=1, ssm_sketch=torch.tensor(np.stack(sk)),
                        ssm_norm=torch.ones(4), conv=torch.tensor(np.stack(rows)).to(torch.bfloat16))
                torch.save(rec, os.path.join(OUT, "forcewin", "snap_%s_F%d_r%d.pt" % (fw._safe(d["tag"]), F, rank)))


PSG = {}


def ps_phase(ph, reqs):
    base = STEP[0]
    STEP[0] += 20
    for j, d in enumerate(reqs):
        n = d["fixture"]
        f = G["fx"][n]
        P, c0 = f["P"], f["c0"]
        M = metrics_rows(n, ph, "C", 0, 1, lambda i: base + j, lambda i: 0)
        save_req(d["tag"], P, 1, M, np.ones(1, np.int32), [], torch.zeros(0, 17, 8, dtype=torch.int32), [],
                 torch.zeros(0, 17, 512, dtype=torch.int32))
        add_rec(d)
        groups = PSG.setdefault(n, new_groups(P + 1))

        def extra(rank, n=n, c0=c0, groups=groups):
            out = {}
            ids0 = groups[0]["ids"].tolist()
            ids9 = groups[-1]["ids"].tolist()
            for L in IDXL[rank]:
                mpg = {ids0[p // 64]: torch.tensor(np.frombuffer(np.random.default_rng(hash((n, L, p)) % 2**32).bytes(
                    64 * 528), dtype=np.uint8).reshape(64, 528).copy()) for p in range(c0 - 1024, c0, 64)}
                if V == "a-ps" and ph == "PS-c" and n == "L4" and L == IDXL[rank][0]:
                    k0 = sorted(mpg)[3]
                    mpg[k0] = mpg[k0].clone()
                    mpg[k0][5, 7] ^= 1
                out["language_model.model.layers.%d.self_attn.attn" % L] = dict(
                    type="MLAAttention", kind="paged", shape=[86700, 64, 528], dtype="torch.uint8", pages=mpg)
                out["language_model.model.layers.%d.self_attn.indexer.tail_cache" % L] = dict(
                    type="Glm5NextTailCache", kind="paged", shape=[1275, 2, 8, 128], dtype="torch.bfloat16",
                    pages={0: torch.randn(2, 8, 128).to(torch.bfloat16)})
            apg = {ids9[(c0 - 1) // 1088]: torch.tensor(np.random.default_rng(hash((n, "a")) % 2**32).normal(
                0, 1, (4, 1088, 8)), dtype=torch.float32)}
            out["language_model.model.layers.%d.self_attn.attn_swa" % (40 + rank)] = dict(
                type="Attention", kind="paged", shape=[9000, 4, 1088, 256], dtype="torch.bfloat16", pages=apg)
            return out
        pools = list(range((c0 - 1024) // 4, c0 // 4))
        # identical shared-block content in PS-a/b/c: index-K values of the cached block are fixed per fixture
        save_dump(d["tag"], P, c0 - 1024, c0, n, "PS", groups, pools, extra_layers=extra, F=1)
        _fix_ps_idx(d["tag"], n, pools, groups)
        rec = dict(tag=d["tag"], F=1, P=P, step=base + j, frontier_pos=P - 1, num_computed=P, total_len=P + 1,
                   last_sampled=1, tokens_ok=True, num_accepted=1, groups=groups, range=[P - 65, P + 4], layers={})
        for rank in (0, 1):
            for L in KDAL:
                rows = [conv_true(n, L, P - 3 + e) for e in range(3)] + [np.zeros(D_CONV, np.float32)] * 3
                rec["layers"]["language_model.model.layers.%d.self_attn" % L] = dict(
                    type="Glm5NextLinearAttention", kind="kda", is_spec=False, non_spec_slots=torch.tensor([3]),
                    slots=torch.tensor([3]), ssm_sketch=torch.tensor(ssm_true(n, L, P - 1))[None],
                    ssm_norm=torch.ones(1), conv=torch.tensor(np.stack(rows)).to(torch.bfloat16))
            torch.save(rec, os.path.join(OUT, "forcewin", "snap_%s_F1_r%d.pt" % (fw._safe(d["tag"]), rank)))


def _fix_ps_idx(tag, n, pools, groups):
    """Overwrite the sentinel's index-K pages with the fixture's fixed cached-block content (no build noise)."""
    for rank in (0, 1):
        p = os.path.join(OUT, "forcewin", "snap_%s_F1win_r%d.pt" % (fw._safe(tag), rank))
        rec = torch.load(p, weights_only=False)
        for L in IDXL[rank]:
            nm = "language_model.model.layers.%d.self_attn.indexer.k_cache" % L
            rec["layers"][nm]["pages"] = idx_pages(groups[0]["ids"].tolist(), pools, base_vals(n, L, pools))
        torch.save(rec, p)


# ------------------------------------------------------------------ session
w(dict(kind="start", t=T[0], plan=False, deadline=9e9, scope="first-contrast fc1 SYNTHETIC " + V, order=ORDER,
       wmax=geo.WMAX))
dump("start")
for d in range(4):
    w(dict(kind="probe", id=f"q{d}", sha256="x", expected="x", equal=True))
phase("V0")
byph = {}
for d in PLAN:
    byph.setdefault(d["phase"], []).append(d)
phases = list(byph)
gate_after = phases.index("CV")
for i, ph in enumerate(phases):
    reqs = byph[ph]
    role = reqs[0]["role"]
    if role == "ref":
        ref_phase(ph, reqs)
    elif role in ("C", "D", "LC", "LD"):
        run_phase(ph, reqs)
    elif role == "CV":
        cv_phase(reqs)
    elif role == "PS":
        ps_phase(ph, reqs)
    for d in reqs:
        r = dict(d)
        cached = d["cached_expected"]
        if role == "warm" and d.get("rewarm"):
            cached = d["hit_value"]
        if V == "g-cache" and d["tag"] == "C1-L4":
            cached = 0
        ide = True if (d.get("keyed") and d["W"]) else None
        if V == "a-incomplete" and d["tag"] == "D3-A8":
            ide = False
        if V == "g-wmax" and d["tag"] == "C1-L1":
            ide = False
        r.update(t0=T[0], t1=T[0] + 1, secs=1.0, finish="length", completion_tokens=d["W"], prompt_tokens=d["P"],
                 cached_tokens=cached, returned_sha="x", ids_equal=ide, error=None)
        w(r)
    phase(ph)
    if i == gate_after:
        w(dict(kind="gate_start", t=now(0.1)))
    if ph == "LC2":
        w(dict(kind="gate", gate="GATE1", ok=True, verdict="PASS", rc=0, secs=120.0, waited=0.0))
w(dict(kind="coverage", required=[], ran=phases, not_run=[], order=phases, left=1e9))
w(dict(kind="end", t=now()))
CLF.close()
for rank in (0, 1):
    with open(os.path.join(OUT, "forcewin", "fw_r%d_p1.jsonl" % rank), "w") as fh:
        fh.write(json.dumps(dict(kind="init", K=3, vocab=154880, max_reqs=32,
                                 wmax=5120 if (V == "g-wmax" and rank == 1) else geo.WMAX, rank=rank, t=1.0)) + "\n")
        fh.write(json.dumps(dict(kind="registered", rank=rank, t=1.0)) + "\n")
        for a in ADDS[rank]:
            fh.write(json.dumps(a) + "\n")
        fh.write(json.dumps(dict(kind="steps", cols=["step", "num_reqs", "num_tokens", "num_draft_tokens",
                                                      "n_prefilling", "padded_tokens"], rows=STEPS, rank=rank)) + "\n")
    with open(os.path.join(OUT, "mapguard", "mg_p%d.jsonl" % rank), "w") as fh:
        fh.write(json.dumps(dict(kind="init", rank=rank)) + "\n")
        for p in POLLS[rank]:
            fh.write(json.dumps(p) + "\n")
open(os.path.join(OUT, "image-id.txt"), "w").write("local/glm53-v0.9.1-b1-forcehook-mg2:local synthetic\n")
print("synthetic run %s: %d requests, %d phases" % (V, len(PLAN), len(phases)))
