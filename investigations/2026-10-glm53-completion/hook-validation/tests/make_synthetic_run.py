#!/usr/bin/env python3
"""Synthetic run directories for a dry run of analysis/analyze_hookval.py, ATTEMPT 2 (PA2-8; the attempt-1 generator
is kept as make_synthetic_run.py.a1). CPU; inside the B1 image for torch.

  make_synthetic_run.py OUT good   -> the attempt-2 run shape with attempt 1's NON-defects built in: two V2c
                                      references whose window starts 5 positions before the decode window (join
                                      offset), one served row left unmasked with 2.6e-9 kept mass, deep-layer KDA drift
                                      (rel 0.1, margin 2), deep-layer index-K fragility (~2 % of layer-39 pools differ
                                      between any two builds), end-of-window pools unclassified, the guard's pre-client
                                      warm-up pos_bounds = 4 per rank, polls keyed by `rank` (no `pid`).
                                      Expected: REVISED VALID (optional poison present); ORIGINAL INVALID.
  make_synthetic_run.py OUT bad    -> the same plus planted defects, each aimed at one revised check:
                                      off-by-one rows in one request (H2c-r) and a doubly measured position (H1c);
                                      a served-mask violation with 1e-3 kept mass (H1f-r); a wrong committed KDA slot
                                      (H4b-r); one corrupted B1 pool in layer 3 (IDX-early); a whole misaddressed
                                      64-pool page, layer 39, in the solo injected request (IDX-page; PA2-19 moved it
                                      from the second B1 arm: there it inflates the decode null N_dd = b1 vs b1b and
                                      with it the content limit of every fixture; it now also fails H5e-r); a B1-arm pool labelled
                                      corrupt by the ledger (H5f-B1); a position violation on rank 1 after the start
                                      dump (H5g-r); misattributed exposure steps in one request (H5h-r).
                                      PA2-19 (amendment 21g) adds five plants:
                                      (21g-a) fixtures u4, u5: the SAME 26 layer-39 pools wrong in both B1 arms (a
                                      defect common to both decode arms; judged pages stay < 25 %) -> IDX-content,
                                      and u4, u5 flagged as outside their own null (REPORTED);
                                      (21g-b) fixture u3: 12 layer-39 pools wrong in the first B1 arm only -> flagged
                                      as an arm asymmetry (REPORTED; the decode null N_dd shares that arm, so the
                                      content limit absorbs a one-arm defect: see ATTEMPT2-21G.md);
                                      (21g-c) a deep-layer (20) wrong slot nearer than the committed one by ~20x
                                      (H4b-sep, inconclusive level);
                                      (21g-d) a snapshot whose num_accepted (2) disagrees with the hook's accepted
                                      count (0) at its step, slots arranged so identification still agrees (H4-slot);
                                      (21g-e) a second-B1-arm window dump recording a range shifted by 4 positions
                                      (IDX-attrib).
                                      Expected: REVISED INVALID failing exactly
                                      H1c, H1f-r, H2c-r, H3-r, H4-slot, H4b-r, H4b-sep, IDX-attrib, IDX-page,
                                      IDX-early, IDX-content, H5e-r, H5f-B1, H5g-r, H5h-r, CARRY-admissible
                                      (H3-r: the shifted request also disagrees with the single-token advance;
                                      CARRY-admissible: its current-image prerequisites H1c, H4-slot, IDX-attrib,
                                      H5f-B1 and H5h-r fail).
  good also carries (PA2-19): the run's image-id.txt (attempt-2 image, from carryover-a1.json), the client's start
  record with matrix / d2set, rebuild dumpranges in the client records, the hook's accepted count per row and the
  snapshot step (so H4-slot is exercised), and a determinable outside-nucleus mass on the unmasked row.
The files have the hook's real formats (pack via forcewin_hooks.COLS; snapshot dict layout; guard poll layout with
rank / reason / t / layers / counters, as in run-a1/mapguard).
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
rng = np.random.default_rng(17)          # same draws in good and bad: the plants are the only differences
os.makedirs(os.path.join(OUT, "forcewin"), exist_ok=True)
os.makedirs(os.path.join(OUT, "mapguard"), exist_ok=True)
CL = open(os.path.join(OUT, "client.jsonl"), "w")
C = fw.C
K = 3
T = [1000.0]
LAYER_NAMES = ["language_model.model.layers.%d.self_attn.indexer.k_cache" % L for L in (3, 7, 11, 15, 19, 23)]


def now(dt=1.0):
    T[0] += dt
    return round(T[0], 3)


def w(d):
    CL.write(json.dumps(d) + "\n")


DUMPS = []


def dump(tag):
    t = now(0.1)
    w(dict(kind="dump", tag=tag, t=t, rc=0))
    DUMPS.append((tag, t))


def phase(tag):
    t1 = now(5.0)
    w(dict(kind="phase", tag=tag, t0=t1 - 5.0, t1=t1, secs=5.0))
    dump(tag)


REF = {}


def ref_lp(key, W):
    if key not in REF:
        REF[key] = (rng.normal(-2.5, 1.5, W + 8).clip(-14, -1e-4), rng.integers(0, 150000, W + 8))
    return REF[key]


def metrics(key, W, noise=0.004, step0=10):
    lp, am = ref_lp(key, W)
    M = np.full((W, fw.NM), np.nan, dtype=np.float32)
    x = lp[:W] + rng.laplace(0, noise, W) * (rng.random(W) < 0.9) + rng.laplace(0, 0.2, W) * (rng.random(W) < 0.01)
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
    M[:, C["step"]] = step0 + np.arange(W) // 3
    M[:, C["accepted"]] = 0.0                      # PA2-19: the hook's accepted count per row (single advance)
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


def exposure(M, W, misattribute=False):
    dsg = list(range(0, W, 16))
    ex = torch.full((len(dsg), 17, 8), -1, dtype=torch.int32)
    for k, o in enumerate(dsg):
        if o == 0:
            continue                       # o = 0 is predicted by the prefill: never a decode record (real)
        st = int(M[o, C["step"]])
        if misattribute and k % 3 == 1:
            st += 1                        # a record from another step (the attempt-1 H5h signature)
        ex[k, :6, 0] = st
        ex[k, :6, 1:] = 1
    sel = torch.full((1, 17, 512), -1, dtype=torch.int32)
    if W > 64:
        sel[0, :6, 0] = int(M[64, C["step"]])
    return dsg, ex, ([64] if W > 64 else []), sel


def req(tag, W, mode="natural", key=None, d2=0, P=200, pool_class=None, M=None, misattribute=False, ref=None,
        nrows=None):
    if M is None:
        M = metrics(key or tag, W)
    n = nrows or W
    dsg, ex, selpos, sel = exposure(M, W, misattribute) if W > 1 and ref is None else ([], torch.zeros(0, 17, 8),
                                                                                      [], torch.zeros(0))
    return dict(tag=tag, req=tag, P=P, W=W, mode=0, marg=0, d2=None, ref=ref, cols=fw.COLS, xcols=fw.XCOLS,
                metrics=torch.tensor(M[:n]), nmeas=torch.ones(n, dtype=torch.int32), acc_hist=acc_hist(mode, W),
                inv=torch.tensor([W, 0, 0, 0]), d2_injected_tokens=d2,
                pool_class=torch.tensor(pool_class if pool_class is not None else np.zeros(W // 4 + 2),
                                        dtype=torch.int8),
                desig=dsg, exposure=ex, selpos=selpos, selections=sel,
                layer_slots={nm: i for i, nm in enumerate(LAYER_NAMES)}, snaps=[], wall=1.0, errors=0)


def save_req(rec, tag, mutate=None):
    for rank in (0, 1):
        r = dict(rec)
        if mutate and rank == 0:
            r = mutate(r)
        torch.save(r, os.path.join(OUT, "forcewin", "req_%s_r%d.pt" % (fw._safe(tag), rank)))


SEED = [100]


def client_req(phase_, tag, W, mode="natural", P=200, ref_from=None, batch=None, dumprange=None):
    SEED[0] += 1
    w(dict(kind="request", phase=phase_, tag=tag, mode=mode, seed=SEED[0], P=P, W=W,
           ids_equal=True if W else None, finish="length", error=None, ref_from=ref_from, batch=batch,
           dumprange=dumprange))


MATRIX = ["t%d" % i for i in range(8)]
D2SET = ["u%d" % i for i in range(8)]
CARRY = json.load(open("/hv/analysis/carryover-a1.json"))
open(os.path.join(OUT, "image-id.txt"), "w").write(
    "local/glm53-v0.9.1-b1-forcehook-mg2:local %s\n" % CARRY["scope"]["image_a2"])
w(dict(kind="start", t=T[0], plan=False, deadline=9e9, scope="attempt2", matrix=MATRIX, d2set=D2SET))
dump("start")
# ---------------------------------------------------------------- V0
for d in range(4):
    w(dict(kind="probe", id=f"q{d}", sha256="x", expected="x", equal=True))
phase("V0")
# ---------------------------------------------------------------- V2a natural / oracle (+ poison at the end)
JOIN_OFF = {"t0": 5, "t1": 5}            # V2c window starts 5 positions before the decode window (ref_from = P - 5)
W2, P2 = 300, 200


def v2a(mode):
    for tid in MATRIX:
        tag = f"V2a-{mode}-{tid}"
        M = metrics(tid, W2)
        if mode == "natural" and tid == "t3":       # a served row left unmasked: tracked ids carry their pre mass
            o = 77
            M[o, C["lp_pre_0"]:C["lp_pre_0"] + 4] = np.log(2.6e-9)
            M[o, C["p_post_0"]:C["p_post_0"] + 4] = 2.6e-9
            M[o, C["p_post_4"]] = np.exp(M[o, C["lp_pre_4"]])
            M[o, C["top1_p"]] = 1.0
            M[o, C["argmax_is_forced"]] = 1.0          # forced token = argmax, nucleus = {top}: mass_lt_4 determines
            M[o, C["mass_lt_4"]] = 3.1e-9              # the total pre-mask mass outside the intended nucleus
            if BAD:                                  # ... and a real violation: 1e-3 kept although not eligible
                M[o + 10, C["lp_pre_1"]] = np.log(1e-3)
                M[o + 10, C["p_post_1"]] = 1e-3
        mut = None
        if BAD and mode == "oracle" and tid == "t2":
            M[201:241, C["lp_pre_4"]] = M[200:240, C["lp_pre_4"]].copy()     # off by one on 40 rows
            M[201:241, C["argmax"]] = M[200:240, C["argmax"]].copy()
        rec = req(tag, W2, mode, key=tid, M=M, P=P2, misattribute=BAD and mode == "natural" and tid == "t5")
        if BAD and mode == "oracle" and tid == "t2":
            def mut(r):
                r = dict(r)
                n = r["nmeas"].clone()
                n[5] = 2
                r["nmeas"] = n
                return r
        client_req("V2a", tag, W2, mode, P=P2)
        save_req(rec, tag, mutate=mut)
    phase(f"V2a-{mode}")


v2a("natural")
v2a("oracle")
# ---------------------------------------------------------------- V2c references (row k predicts ref_from + k)
for tid in MATRIX:
    tag = f"V2c-{tid}"
    off = JOIN_OFF.get(tid, 0)
    lp, am = ref_lp(tid, W2)
    Mr = metrics(tid + "-pre", W2 + off + 2, noise=0.0)
    Mr[off:off + W2, C["lp_pre_4"]] = lp[:W2]
    Mr[off:off + W2, C["argmax"]] = am[:W2]
    Mr[off + W2:, C["lp_pre_4"]] = np.nan
    client_req("V2c", tag, 0, P=P2 + W2, ref_from=P2 - off)
    save_req(req(tag, 0, key=tid, M=Mr, P=P2 + W2, ref=P2 - off, nrows=W2 + off + 2), tag)
phase("V2c")


# ---------------------------------------------------------------- index-K pages and snapshots
def idx_page_set(blk, pools, values):
    pages = {}
    for pool, v in zip(pools, values):
        pg = blk * 17 + (pool % 1088) // 64
        t = pages.setdefault(pg, torch.zeros(64, 132, dtype=torch.uint8))
        raw = t.reshape(-1)
        rr = pool % 64
        qv = torch.tensor(v, dtype=torch.float32).to(torch.float8_e4m3fn)
        raw[rr * 128:(rr + 1) * 128] = qv.view(torch.uint8)
        raw[8192 + rr * 4: 8192 + rr * 4 + 4] = torch.tensor([1.0], dtype=torch.float32).view(torch.uint8)
    return pages


def groups(blk):
    return [dict(g=0, block_size=4352, kernel_block=64, bpk=68, ids=torch.tensor([blk * 68 + k for k in range(68)])),
            dict(g=1, block_size=8, kernel_block=8, bpk=1, ids=torch.tensor([5]))]


POOLV = {}
IDXL = (3, 39)


def base_vals(key, layer, pools):
    return [POOLV.setdefault((key, layer, p), rng.normal(0, 1, 128).clip(-400, 400)) for p in pools]


FRAG = {}


def fragile(key, pools):
    """~1 % of layer-39 pools of a trace are fragile: every build of them differs (deep-layer fp8 key fragility;
    attempt 1 measured ~0.5 % over all DSA layers, up to 1.7 % in layer 39)."""
    if key not in FRAG:
        FRAG[key] = {p for p in pools if rng.random() < 0.01}
    return FRAG[key]


def build_vals(key, layer, pools, noise=0.01, frag=True):
    base = base_vals(key, layer, pools)
    fr = fragile(key, pools) if (frag and layer == 39) else set()
    return [rng.normal(0, 1, 128) if p in fr else v + rng.normal(0, noise, 128) for p, v in zip(pools, base)]


def snap(tag, F, P, blk, kda, na, layer_vals, rng_lo_hi=None, win=False, spec=True, step=None):
    """kda: {layer: [slot sketches]}; layer_vals: {layer: (pools, values)}."""
    rec = dict(tag=tag, F=F, P=P, step=step, frontier_pos=P + F - 2, num_computed=P + F - 1, total_len=P + F,
               last_sampled=1,
               tokens_ok=True, num_accepted=na, groups=groups(blk), range=rng_lo_hi or [P + F - 66, P + F + 3],
               layers={})
    for L, slots in kda.items():
        rec["layers"]["language_model.model.layers.%d.self_attn" % L] = dict(
            kind="kda", is_spec=spec, slots=torch.arange(len(slots)), ssm_sketch=torch.stack(slots), num_accepted=na)
    for L, (pools, vals) in layer_vals.items():
        rec["layers"]["language_model.model.layers.%d.self_attn.indexer.k_cache" % L] = dict(
            kind="paged", shape=[1000, 64, 132], stride=[8448, 132, 1], dtype="torch.uint8",
            pages=idx_page_set(blk, pools, vals))
    for rank in (0, 1):
        torch.save(rec, os.path.join(OUT, "forcewin", "snap_%s_F%d%s_r%d.pt" % (fw._safe(tag), F,
                                                                                "win" if win else "", rank)))


# ---------------------------------------------------------------- V4 + V4n
W4, P4 = 400, 128
pools4 = list(range(P4 // 4, (P4 + W4 + 1) // 4))
inner = list(range((P4 + 3) // 4, (P4 + W4 - 1) // 4))


def v4_classes(bad_mask=None):
    pc = np.zeros(W4 // 4 + 2, dtype=np.int8)
    for i in inner:
        k = i - P4 // 4
        if bad_mask is None:
            pc[k] = 1
        else:
            b = bad_mask[pools4.index(i)]
            pc[k] = (2 if b else 1) if rng.random() < 0.5 else 3
    return pc


COMMON_BAD = {}
BAD_U4 = set(range(40, 47)) | set(range(70, 85)) | set(range(128, 132))   # 7 of page 0's 32 pools, 15 of page 1's
#                                                       64 (< 25 % each) and page 2's 4 (not judged: < 16 pools)
BAD_U3 = set(range(90, 102))                           # 12 of page 1's 64 pools
BLK = {"d2": 80, "b1": 90, "d2solo": 100, "rebuild": 120, "b1b": 140, "rebuild2": 160}
for arm in ("d2", "b1", "d2solo", "rebuild", "b1b", "rebuild2"):
    tids = D2SET[:1] if arm == "d2solo" else D2SET
    for i, tid in enumerate(tids):
        ph = "V4n" if arm in ("b1b", "rebuild2") else "V4"
        tag = f"{ph}-{arm}-{tid}"
        lv = {}
        bad_mask = rng.random(len(pools4)) < 0.85 if arm == "d2" else None    # one corruption mask per request
        for L in IDXL:
            vals = build_vals(tid, L, pools4)
            if arm == "d2":
                vals = [rng.normal(0, 1, 128) if b else v for v, b in zip(vals, bad_mask)]
            if BAD and arm == "b1" and i == 0 and L == 3:
                vals[10] = rng.normal(0, 1, 128)                         # one corrupted early-layer pool
            if BAD and arm == "d2solo" and i == 0 and L == 39:
                vals = [rng.normal(0, 1, 128) if 64 <= p < 128 else v for p, v in zip(pools4, vals)]   # wrong page
            if BAD and arm in ("b1", "b1b") and i in (4, 5) and L == 39:             # (21g-a) both B1 arms
                vals = [COMMON_BAD.setdefault((i, p), rng.normal(0, 1, 128)) if p in BAD_U4 else v
                        for p, v in zip(pools4, vals)]
            if BAD and arm == "b1" and i == 3 and L == 39:                           # (21g-b) first B1 arm only
                vals = [rng.normal(0, 1, 128) if p in BAD_U3 else v for p, v in zip(pools4, vals)]
            lv[L] = (pools4, vals)
        if arm in ("rebuild", "rebuild2"):
            Pr = P4 + W4 + (0 if arm == "rebuild" else 200)
            client_req(ph, tag, 1, P=Pr, batch=arm, dumprange=[P4, P4 + W4])
            save_req(req(tag, 1, key=tag, P=Pr), tag)
            snap(tag, 1, Pr, BLK[arm] + i, {0: [torch.zeros(256)]}, 1, lv, rng_lo_hi=[P4, P4 + W4], win=True)
            continue
        pc = v4_classes(bad_mask if arm == "d2" else None)
        if BAD and arm == "b1" and i == 2:
            pc[inner[5] - P4 // 4] = 2                                    # ledger calls a B1 pool corrupt
        client_req(ph, tag, W4, P=P4, batch=arm)
        save_req(req(tag, W4, key=tag, d2=(W4 + 40) if arm.startswith("d2") else 0, P=P4, pool_class=pc), tag)
        rlh = [P4 + 4, P4 + W4 + 4] if (BAD and arm == "b1b" and i == 6) else [P4, P4 + W4]   # (21g-e) wrong range
        snap(tag, W4, P4, BLK[arm] + i, {0: [torch.zeros(256)]}, 1, lv, rng_lo_hi=rlh, win=True)
    if arm in ("d2", "b1", "d2solo", "rebuild"):
        phase(f"V4-{arm}")
    else:
        phase(f"V4n-{arm}")
# ---------------------------------------------------------------- V3 (two snapshots per request; KDA layers 0, 20)
for i, tid in enumerate(MATRIX):
    tag = f"V3-sched:7-{tid}"
    client_req("V3", tag, 300, "sched:7", P=400)
    save_req(req(tag, 300, "sched:7", key=tag, P=400), tag)
    P = 400
    for F in (120 + i, 126 + i):
        kda = {}
        na = 2 if (BAD and i == 5 and F == 120 + i) else 1             # (21g-d) bookkeeping disagrees with the hook
        for L, near, far in ((0, 0.001, 0.2), (20, 0.1, 0.2)):
            ref = torch.randn(256)
            slots = [ref * (1 + near)] + [ref * (1 + near) + far * ref.norm() / 16 * torch.randn(256) for _ in range(3)]
            if BAD and i == 0 and L == 0:
                slots = [slots[1], slots[0]] + slots[2:]                 # wrong slot committed
            if na == 2:
                slots = [slots[1], slots[0]] + slots[2:]                 # ... the near state sits in slot na - 1
            if BAD and i == 3 and L == 20 and F == 120 + i:              # (21g-c) comfortably separated wrong slot
                slots = [slots[1], slots[2], ref * 1.005, slots[3]]
            kda[L] = (ref, slots)
        pools = list(range(0, (P + F - 1) // 4))
        lv = {L: (pools, base_vals(("v3", tid), L, pools)) for L in IDXL}
        snap(tag, F, P, 30 + i, {L: s for L, (r, s) in kda.items()}, na, lv, step=10 + (F - 2) // 3)
        rtag = f"V3ref-F{F}-{tag}"
        client_req("V3", rtag, 1, P=P + F - 1)
        lvr = {L: (pools, [v + rng.normal(0, 0.01, 128) for v in base_vals(("v3", tid), L, pools)]) for L in IDXL}
        snap(rtag, 1, P + F - 1, 60 + i, {L: [r] for L, (r, s) in kda.items()}, 1, lvr, spec=False)
phase("V3-forced")
phase("V3-refs")
v2a("poison")                                                          # optional phase, present
# ---------------------------------------------------------------- guard polls (rank, reason, t; no pid)
inj_phase = {"V4-d2": 8 * (W4 + 40), "V4-d2solo": W4 + 40}
SITES = {"tail_w_stash": ("mism", "null"), "tail_w_seed": ("mism",), "idx_w_dec": ("mism",),
         "idx_w_pf": ("s_mism", "d7_fallback"), "pos": ("pos_contig", "pos_bounds")}
for rank, nl, pid in ((0, 6, 369), (1, 5, 403)):
    cnt = {s: {c: 0 for c in cs} for s, cs in SITES.items()}
    lay = {str(3 + 4 * j + (24 if rank else 0)): {} for j in range(nl)}
    polls = []

    def poll(reason, t):
        polls.append(dict(kind="poll", reason=reason, host_step=len(polls), counters=json.loads(json.dumps(cnt)),
                          layers=lay, t=t, rank=rank))
    cnt["pos"]["pos_bounds"] = 4                                       # runner warm-up artefact, before the client
    poll("periodic", 990.0)
    for tag, t in DUMPS:
        if tag in inj_phase:
            cnt["tail_w_stash"]["mism"] += nl * inj_phase[tag]
            cnt["tail_w_stash"]["null"] += nl * inj_phase[tag]
        if BAD and rank == 1 and tag == "V3-forced":
            cnt["pos"]["pos_contig"] += 3
        poll("trigger", t + 0.05)
    poll("trigger", T[0] + 30)                                         # the runner's final DUMP at cleanup
    with open(os.path.join(OUT, "mapguard", "mg_p%d.jsonl" % pid), "w") as f:
        for p in polls:
            f.write(json.dumps(p) + "\n")
w(dict(kind="end", t=now()))
CL.close()
print("synthetic attempt-2 run written:", OUT, MODE)
