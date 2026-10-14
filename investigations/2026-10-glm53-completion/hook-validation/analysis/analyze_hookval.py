#!/usr/bin/env python3
"""Frozen analysis of the hook-validation session (VALIDATION.md sections 3-6). CPU only.

usage: analyze_hookval.py RUN_DIR      -> RUN_DIR/verdict.json, verdict.txt

Inputs (written by run_hookval.sh): client.jsonl; forcewin/req_<tag>_r<rank>.pt, snap_*.pt, fw_r*.jsonl;
mapguard/mg_p*.jsonl (guard polls, one after every phase).

Verdict precedence: INVALID (the instrument failed a necessary check or data are missing) > INCONCLUSIVE (a check
could not discriminate: tolerance not separable from the alignment-power control, or coverage short) > VALID.
Every number below is frozen here, before any GPU data (VALIDATION.md section 5 gives the justification).
"""
import glob
import json
import os
import re
import sys

import numpy as np
import torch

RUN = sys.argv[1]
TERM = (154842, 154820, 154827, 154829)
K = 3
# ---- frozen tolerances (VALIDATION.md section 5)
NULL_FACTOR = 2.0                  # hooked gap quantiles may exceed the in-session unhooked null by this factor
FLOOR = {0.5: 0.01, 0.9: 0.30, 0.99: 1.50}   # absolute floors (historical unhooked spec-on x3: 0.0011/0.162/0.745)
TOP1_MIN = 0.95                    # argmax agreement, decode vs prefill (historical 0.972)
TOP1_DEC_MIN = 0.97                # argmax agreement, decode vs decode (schedule invariance)
POWER_MIN_MEDIAN = 1.0             # alignment power: median |dlp| at offset +-1 must exceed 1 nat ...
POWER_RATIO = 20.0                 # ... and 20 x the aligned median
IDX_GROSS = 0.25                   # index-K pool: relative L2 above which a pool is "gross" (probe proposal s.8.5)
B1_GROSS_MAX = 0.001               # B1 arm and the solo injected run: gross-pool fraction at most 0.1 %
D2_GROSS_MIN = 0.30                # 8-in-flight injected arm: corruption confirmed above 30 % gross pools
SHADOW_CLEAN_OK = 0.99             # shadow-clean pools that are non-gross
SHADOW_CORRUPT_OK = 0.95           # shadow-corrupt pools that are gross
KDA_ABS = 0.05                     # committed recurrent state vs prefill-built state, relative L2 (sketch)
KDA_MARGIN = 3.0                   # ... and the committed slot is 3x closer than any scratch slot
ACC_SCHED_FRAC = 0.95              # fraction of steps that follow the prescribed accepted count

V = {"checks": [], "invalid": [], "inconclusive": []}


def check(name, ok, detail, level="invalid"):
    V["checks"].append(dict(name=name, ok=bool(ok), detail=detail, level=level))
    if not ok:
        V[level].append(name)


def q(a, p):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, p)) if a.size else float("nan")


# ------------------------------------------------------------------ load
def jl(p):
    out = []
    if os.path.exists(p):
        for ln in open(p):
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    return out


CL = jl(os.path.join(RUN, "client.jsonl"))
REQ = {}
for p in glob.glob(os.path.join(RUN, "forcewin", "req_*_r*.pt")):
    m = re.search(r"req_(.*)_r(\d+)\.pt$", os.path.basename(p))
    REQ.setdefault(m.group(1), {})[int(m.group(2))] = torch.load(p, weights_only=False)
SNAP = {}
for p in glob.glob(os.path.join(RUN, "forcewin", "snap_*_r*.pt")):
    m = re.search(r"snap_(.*)_F(\d+)(win)?_r(\d+)\.pt$", os.path.basename(p))
    SNAP.setdefault((m.group(1), int(m.group(2)), bool(m.group(3))), {})[int(m.group(4))] = p
FWLOG = [r for p in glob.glob(os.path.join(RUN, "forcewin", "fw_r*.jsonl")) for r in jl(p)]
POLLS = sorted([r for p in glob.glob(os.path.join(RUN, "mapguard", "mg_p*.jsonl")) for r in jl(p)
                if r.get("kind") == "poll"], key=lambda r: r.get("t", 0))
reqs = [r for r in CL if r.get("kind") == "request"]
C = None


def cols(rec):
    global C
    if C is None:
        C = {n: i for i, n in enumerate(rec["cols"])}
    return C


def r0(tag):
    d = REQ.get(tag.replace(":", "_"))
    if not d:
        return None
    return d.get(0) or next(iter(d.values()))


# ------------------------------------------------------------------ H1 instrument integrity
def h1():
    fr = [r for r in reqs if r.get("W", 0) > 0 and r["phase"] != "V3" or (r["phase"] == "V3" and not
                                                                           r["tag"].startswith("V3ref"))]
    fr = [r for r in fr if r.get("W", 0) > 0]
    bad_ids = [r["tag"] for r in fr if r.get("ids_equal") is not True or r.get("error")]
    check("H1a returned ids == forced ids for every forced request", not bad_ids and fr,
          f"{len(fr)} requests, mismatches {bad_ids[:5]}")
    once, nan, post, brk, inv, xr, missing = [], [], [], [], [], [], []
    for r in fr:
        d = REQ.get(r["tag"].replace(":", "_"))
        if not d:
            missing.append(r["tag"])
            continue
        a = d.get(0) or next(iter(d.values()))
        Cc = cols(a)
        W = r["W"]
        nm = a["nmeas"][:W]
        M = a["metrics"][:W].numpy()
        if not bool((nm == 1).all()):
            once.append(r["tag"])
        lp = M[:, Cc["lp_pre_0"]:Cc["lp_pre_0"] + 5]
        if not np.isfinite(lp).all():
            nan.append(r["tag"])
        if not (M[:, Cc["post_ok"]] == 1).all():
            post.append(r["tag"])
        pp = M[:, Cc["p_post_0"]:Cc["p_post_0"] + 5]
        lo = M[:, Cc["elig_lo_0"]:Cc["elig_lo_0"] + 5]
        hi = M[:, Cc["elig_hi_0"]:Cc["elig_hi_0"] + 5]
        if ((pp > 0) & (hi < 0.5)).any() or ((lo > 0.5) & (pp <= 0)).any():
            brk.append(r["tag"])
        if int(a["inv"][1]) != 0:
            inv.append(r["tag"])
        if len(d) == 2:
            b = d[1 - min(d)]
            if not np.array_equal(np.nan_to_num(a["metrics"][:W].numpy(), nan=7.0),
                                  np.nan_to_num(b["metrics"][:W].numpy(), nan=7.0)):
                xr.append(r["tag"])
        else:
            xr.append(r["tag"] + "(one rank)")
    check("H1b hook records exist for every forced request", not missing, f"missing {missing[:5]}")
    check("H1c every window position measured exactly once", not once, f"{once[:5]}")
    check("H1d no NaN in the tracked pre-mask log-probabilities", not nan, f"{nan[:5]}")
    check("H1e post-mask values read from the served processed logits on every row", not post, f"{post[:5]}")
    check("H1f served top_p mask inside the eligibility bounds", not brk, f"{brk[:5]}")
    check("H1g last committed token == history after every step", not inv, f"{inv[:5]}")
    check("H1h both ranks recorded identical metrics (same committed tokens per step)", not xr, f"{xr[:5]}")
    errs = [r for r in FWLOG if r.get("kind") == "fw_error"]
    check("H1i no hook errors", not errs, f"{[e.get('where') for e in errs[:5]]}")
    # schedules: accepted-count histograms
    bad = []
    for r in fr:
        a = r0(r["tag"])
        if a is None:
            continue
        h = a["acc_hist"].numpy()[: K + 1].astype(float)
        n = h.sum()
        mode = r["mode"]
        if n == 0:
            continue
        frac = {"oracle": h[K], "poison": h[0], "reject:1": h[0], "reject:2": h[1], "reject:3": h[2]}.get(mode)
        if frac is not None and frac / n < ACC_SCHED_FRAC:
            bad.append((r["tag"], h.tolist()))
        if mode.startswith("sched") and not (h > 0).all():
            bad.append((r["tag"], h.tolist()))
    check("H1j accepted-count histograms follow the draft schedules (bonus-token accounting)", not bad, f"{bad[:4]}")


# ------------------------------------------------------------------ H2 measurement alignment vs prefill reference
def lp_forced(a, W):
    Cc = cols(a)
    return a["metrics"][:W, Cc["lp_pre_4"]].numpy().astype(float)


def argmax(a, W):
    Cc = cols(a)
    return a["metrics"][:W, Cc["argmax"]].numpy()


def pairs_vs_ref(prefix, ref_prefix):
    out = []
    for r in reqs:
        if not r["tag"].startswith(prefix) or r.get("W", 0) == 0:
            continue
        key = r["tag"].split("-", 2)[-1]
        ref = r0(f"{ref_prefix}-{key}")
        a = r0(r["tag"])
        if ref is None or a is None:
            continue
        out.append((r["tag"], a, ref, r["W"]))
    return out


def shifted(y, W, shift):
    """y aligned so that out[o] = y[o + shift] (NaN where undefined)."""
    out = np.full(W, np.nan)
    for o in range(W):
        k = o + shift
        if 0 <= k < y.size:
            out[o] = y[k]
    return out


def gaps(pairs, shift=0, per_request=False):
    d, t1, per = [], [], {}
    for tag, a, ref, W in pairs:
        x = lp_forced(a, W)
        y = ref["metrics"][:, cols(ref)["lp_pre_4"]].numpy().astype(float)
        dd = np.abs(x - shifted(y, W, shift))
        d += list(dd)
        per[tag] = q(dd, 0.5)
        t1 += list(argmax(a, W) == ref["metrics"][:W, cols(ref)["argmax"]].numpy())
    if per_request:
        return per
    return np.array(d), float(np.mean(t1)) if t1 else float("nan")


def null_gap():
    """B1's own unhooked decode measurement (free-run returned logprobs) vs the prefill reference."""
    d = []
    for r in CL:
        if r.get("kind") != "free_hist":
            continue
        ref = r0(f"V1-ref-{r['tag']}")
        if ref is None:
            continue
        lps = np.array([x if x is not None else np.nan for x in r["lps"]], dtype=float)
        y = ref["metrics"][: len(lps), cols(ref)["lp_pre_4"]].numpy().astype(float)
        d += list(np.abs(lps - y[: len(lps)]))
    return np.array(d)


def h2():
    null = null_gap()
    have_null = null.size > 100
    nq = {p: (q(null, p) if have_null else FLOOR[p] / NULL_FACTOR) for p in FLOOR}
    tol = {p: max(NULL_FACTOR * nq[p], FLOOR[p]) for p in FLOOR}
    V["h2_null_quantiles"] = dict(n=int(null.size), **{str(p): nq[p] for p in nq})
    V["h2_tolerances"] = {str(p): tol[p] for p in tol}
    check("H2-null in-session unhooked decode-vs-prefill gap available (V1 free runs)", have_null,
          f"n={null.size}", level="inconclusive")
    allp = []
    for prefix, refp in (("V2a-", "V2c"), ("V1-natural-", "V1-ref"), ("V1-poison-", "V1-ref"), ("V2b-", "V2b-ref")):
        pr = pairs_vs_ref(prefix, refp)
        if prefix == "V2b-":   # matrix traces are referenced in V2c
            pr += [(t, a, r0(f"V2c-{t.split('-', 2)[-1]}"), W) for t, a, _, W in pairs_vs_ref(prefix, "V2c")
                   if r0(f"V2c-{t.split('-', 2)[-1]}") is not None]
        allp += pr
    if not allp:
        check("H2a decode measurements have prefill references", False, "no pairs")
        return
    g0, t0 = gaps(allp)
    gp, _ = gaps(allp, +1)
    gm, _ = gaps(allp, -1)
    stats = {str(p): q(g0, p) for p in FLOOR}
    V["h2_aligned"] = dict(n=int(g0.size), top1=t0, **stats)
    V["h2_power"] = dict(median_plus1=q(gp, 0.5), median_minus1=q(gm, 0.5))
    ok = all(q(g0, p) <= tol[p] for p in FLOOR) and t0 >= TOP1_MIN
    check("H2a forced-decode vs prefill gap within the unhooked null (x2, floors) and top-1 >= 0.95", ok,
          f"aligned {stats} top1 {t0:.4f}; tol {V['h2_tolerances']}")
    p0, pp_, pm_ = gaps(allp, 0, True), gaps(allp, +1, True), gaps(allp, -1, True)
    bad = [t for t in p0 if not (p0[t] < 0.1 * min(pp_[t], pm_[t]))]
    check("H2c per request: the aligned offset beats offsets +-1 by 10x (no misattributed request)", not bad,
          f"{len(p0)} requests; failing {bad[:5]}")
    pw = min(q(gp, 0.5), q(gm, 0.5))
    check("H2b alignment power: a one-position shift is grossly worse (median >= 1 nat and >= 20x aligned)",
          pw >= POWER_MIN_MEDIAN and pw >= POWER_RATIO * max(q(g0, 0.5), 1e-6),
          f"shift medians {V['h2_power']}, aligned median {q(g0, 0.5):.4g}", level="inconclusive")


# ------------------------------------------------------------------ H3 schedule invariance (decode vs decode)
def h3():
    by = {}
    for r in reqs:
        if r["tag"].startswith("V2a-") and r.get("W", 0):
            mode, tid = r["tag"][4:].rsplit("-", 1)
            a = r0(r["tag"])
            if a is not None:
                by.setdefault(tid, {})[mode] = (a, r["W"])
    null = []
    for r in reqs:      # decode-vs-decode null: the same history, same schedule (natural), different batch (V2b)
        if r["tag"].startswith("V2b-natural-"):
            tid = r["tag"].rsplit("-", 1)[-1]
            if tid in by and "natural" in by[tid]:
                a, W = by[tid]["natural"]
                b = r0(r["tag"])
                n = min(W, r["W"])
                null += list(np.abs(lp_forced(a, n) - lp_forced(b, n)))
    null = np.array(null)
    tol = {p: max(NULL_FACTOR * q(null, p) if null.size > 100 else 0.0, FLOOR[p]) for p in FLOOR}
    res, ok = {}, True
    for mode in ("natural", "oracle", "reject:1", "reject:2", "reject:3", "sched:20261013"):
        d, t1 = [], []
        for tid, m in by.items():
            if mode in m and "poison" in m:
                a, W = m[mode]
                b, _ = m["poison"]
                d += list(np.abs(lp_forced(a, W) - lp_forced(b, W)))
                t1 += list(argmax(a, W) == argmax(b, W))
        d = np.array(d)
        if not d.size:
            ok = False
            continue
        res[mode] = dict(n=int(d.size), top1=float(np.mean(t1)), **{str(p): q(d, p) for p in FLOOR})
        ok &= all(q(d, p) <= tol[p] for p in FLOOR) and np.mean(t1) >= TOP1_DEC_MIN
    V["h3"] = dict(null_n=int(null.size), tol={str(p): tol[p] for p in tol}, per_schedule=res)
    check("H3 every draft schedule agrees with the single-token (poison) advance within the decode-vs-decode null",
          ok, json.dumps(res)[:400])


# ------------------------------------------------------------------ H4 committed state (snapshots)
def idx_pool(pages, blk_of, pool):
    """Dequantized index-K record of logical pool `pool` (page-planar 64 x 128 E4M3 + 64 x fp32 scales)."""
    blk = blk_of((4 * pool + 3) // 4352)
    pg = blk * 17 + (pool % 1088) // 64
    t = pages.get(pg)
    if t is None:
        return None
    raw = t.reshape(-1).view(torch.uint8)
    rr = pool % 64
    v = raw[rr * 128:(rr + 1) * 128].view(torch.float8_e4m3fn).float()
    s = raw[8192 + rr * 4: 8192 + rr * 4 + 4].view(torch.float32)
    return v * s


def attn_blocks(rec):
    for g in rec["groups"]:
        if g["bpk"] > 1:
            ids = g["ids"].tolist()
            bpk = g["bpk"]
            return lambda c, ids=ids, bpk=bpk: ids[c * bpk] // bpk if c * bpk < len(ids) else -1
    return lambda c: -1


def idx_layers(rec):
    return {n: L for n, L in rec["layers"].items() if L.get("kind") == "paged" and L.get("shape", [0])[-1:] == [132]}


def h4():
    pairs = []
    for (tag, F, win), ranks in SNAP.items():
        if win or not tag.startswith("V3-"):
            continue
        for (rtag, rF, rwin), rr in SNAP.items():
            if rtag == f"V3ref-F{F}-{tag}" and not rwin:
                pairs.append((tag, F, ranks, rr))
    if not pairs:
        check("H4 committed-state snapshots with prefill references exist", False, "none", level="inconclusive")
        return
    ints, kda, idxr, nkda = [], [], [], 0
    for tag, F, ranks, rr in pairs:
        for rank, p in ranks.items():
            if rank not in rr:
                continue
            a = torch.load(p, weights_only=False)
            b = torch.load(rr[rank], weights_only=False)
            ok = a["tokens_ok"] and a["total_len"] == a["P"] + a["F"] and a["num_computed"] == a["P"] + a["F"] - 1
            ints.append((tag, F, rank, ok))
            na = a.get("num_accepted")
            for name, La in a["layers"].items():
                Lb = b["layers"].get(name)
                if La.get("kind") == "kda" and Lb is not None and "ssm_sketch" in La and "ssm_sketch" in Lb \
                        and na and La.get("is_spec"):
                    nkda += 1
                    ref = Lb["ssm_sketch"][0]
                    dist = [(float((s - ref).norm() / max(ref.norm(), 1e-12))) for s in La["ssm_sketch"]]
                    c = dist[na - 1]
                    others = [x for k, x in enumerate(dist) if k != na - 1]
                    kda.append(dict(tag=tag, layer=name, rel=c, margin=(min(others) / max(c, 1e-12)) if others else
                                    float("inf")))
            fa, fb = attn_blocks(a), attn_blocks(b)
            fr = a["frontier_pos"]
            for name, La in idx_layers(a).items():
                Lb = b["layers"].get(name)
                if Lb is None:
                    continue
                for pool in range(max(0, fr // 4 - 8), (fr + 1) // 4):
                    x = idx_pool(La["pages"], fa, pool)
                    y = idx_pool(Lb["pages"], fb, pool)
                    if x is not None and y is not None:
                        idxr.append(float((x - y).norm() / max(float(y.norm()), 1e-12)))
    check("H4a integer committed state at every snapshot (tokens, total_len, num_computed)",
          all(o for *_, o in ints) and ints, f"{sum(1 for *_, o in ints if not o)} of {len(ints)} failing")
    if kda:
        rel = np.array([k["rel"] for k in kda])
        mar = np.array([k["margin"] for k in kda])
        V["h4_kda"] = dict(n=len(kda), rel_q50=q(rel, 0.5), rel_max=float(rel.max()), margin_min=float(mar.min()))
        check("H4b KDA recurrent state: committed slot (num_accepted-1) matches the prefill-built state",
              rel.max() <= KDA_ABS and mar.min() >= KDA_MARGIN, json.dumps(V["h4_kda"]))
    else:
        check("H4b KDA recurrent state compared", False, f"kda layers matched {nkda}", level="inconclusive")
    if idxr:
        a = np.array(idxr)
        V["h4_idx"] = dict(n=int(a.size), q50=q(a, 0.5), q99=q(a, 0.99), gross=float(np.mean(a > IDX_GROSS)))
        check("H4c index-K pools completed before the frontier: no gross pool (rel L2 > 0.25) vs the prefill build",
              (a > IDX_GROSS).sum() == 0, json.dumps(V["h4_idx"]))
    else:
        check("H4c index-K pools compared", False, "no pools decoded", level="inconclusive")


# ------------------------------------------------------------------ H5 D2 decode-window injection
def window_pools(tag):
    """Per (layer, pool) relative L2 between the decode-written window and the prefill rebuild."""
    tid = tag.rsplit("-", 1)[-1]
    dec = [(k, v) for k, v in SNAP.items() if k[0] == tag and k[2]]
    reb = [(k, v) for k, v in SNAP.items() if k[0] == f"V4-rebuild-{tid}" and k[2]]
    if not dec or not reb:
        return None
    out = {}
    for rank in dec[0][1]:
        if rank not in reb[0][1]:
            continue
        a = torch.load(dec[0][1][rank], weights_only=False)
        b = torch.load(reb[0][1][rank], weights_only=False)
        fa, fb = attn_blocks(a), attn_blocks(b)
        lo, hi = a["range"]
        for name, La in idx_layers(a).items():
            Lb = b["layers"].get(name)
            if Lb is None:
                continue
            for pool in range((lo + 3) // 4, (hi + 1) // 4):
                x = idx_pool(La["pages"], fa, pool)
                y = idx_pool(Lb["pages"], fb, pool)
                if x is not None and y is not None:
                    out[(name, pool)] = float((x - y).norm() / max(float(y.norm()), 1e-12))
    return out


def h5():
    arms = {"d2": [], "b1": [], "d2solo": []}
    for r in reqs:
        if r["phase"] == "V4" and r.get("W", 0) and not r["tag"].startswith("V4-rebuild"):
            arms[r["tag"].split("-")[1]].append(r)
    if not arms["d2"]:
        check("H5 D2 injection phase ran", False, "no V4 requests", level="inconclusive")
        return
    inj = {r["tag"]: (r0(r["tag"]) or {}).get("d2_injected_tokens") for v in arms.values() for r in v}
    ok = all((inj[r["tag"]] or 0) >= r["W"] for r in arms["d2"] + arms["d2solo"]) and \
        all((inj[r["tag"]] or 0) == 0 for r in arms["b1"])
    check("H5a injected tokens: every window token of injected requests, none elsewhere", ok,
          json.dumps({k: v for k, v in list(inj.items())[:6]}))
    res = {}
    shadow_c, shadow_x = [], []
    for arm, rs in arms.items():
        rel = []
        for r in rs:
            wp = window_pools(r["tag"])
            if not wp:
                continue
            rel += list(wp.values())
            a = r0(r["tag"])
            if a is not None and arm != "b1":
                P = r["P"]
                pc = a["pool_class"].numpy()
                for (name, pool), v in wp.items():
                    k = pool - P // 4
                    if 0 <= k < pc.size:
                        if pc[k] == 1:
                            shadow_c.append(v <= IDX_GROSS)
                        elif pc[k] == 2:
                            shadow_x.append(v > IDX_GROSS)
        if rel:
            a = np.array(rel)
            res[arm] = dict(n=int(a.size), gross=float(np.mean(a > IDX_GROSS)), q50=q(a, 0.5), q999=q(a, 0.999))
    V["h5_content"] = res
    b1 = res.get("b1")
    disc = b1 is not None and b1["q999"] <= IDX_GROSS
    check("H5b B1 arm: decode-written window pools match the prefill rebuild (gross <= 0.1 %; positive control "
          "for addressing)", b1 is not None and b1["gross"] <= B1_GROSS_MAX, json.dumps(b1))
    check("H5c the 0.25 threshold discriminates (B1 99.9th percentile below it)", disc, json.dumps(b1),
          level="inconclusive")
    check("H5d injected arm at 8 in flight: corruption present (gross > 30 %)",
          res.get("d2") is not None and res["d2"]["gross"] > D2_GROSS_MIN, json.dumps(res.get("d2")))
    check("H5e injected request alone: no corruption (gross <= 0.1 %; D2's one-request signature)",
          res.get("d2solo") is not None and res["d2solo"]["gross"] <= B1_GROSS_MAX, json.dumps(res.get("d2solo")))
    if shadow_c or shadow_x:
        V["h5_shadow"] = dict(clean_n=len(shadow_c), clean_ok=float(np.mean(shadow_c)) if shadow_c else None,
                              corrupt_n=len(shadow_x), corrupt_ok=float(np.mean(shadow_x)) if shadow_x else None)
        check("H5f shadow ledger agrees with measured bytes (clean -> non-gross >= 99 %, corrupt -> gross >= 95 %)",
              (not shadow_c or np.mean(shadow_c) >= SHADOW_CLEAN_OK) and
              (not shadow_x or np.mean(shadow_x) >= SHADOW_CORRUPT_OK), json.dumps(V["h5_shadow"]))
    # guard: per-phase counter deltas (polls are triggered after every phase)
    if POLLS:
        def tot(poll, site, c):
            return poll["counters"].get(site, {}).get(c, 0)
        ph = [r for r in CL if r.get("kind") == "phase"]
        deltas = {}
        prev = None
        by_rank_last = {}
        for p in POLLS:
            by_rank_last[p.get("pid")] = p
        V["guard_final"] = {str(pid): {s: {c: p["counters"][s][c] for c in ("mism", "null", "pos_contig",
                                                                            "pos_bounds", "s_mism", "d7_fallback")
                                           if c in p["counters"].get(s, {})}
                                       for s in ("tail_w_stash", "tail_w_seed", "idx_w_dec", "idx_w_pf", "pos")}
                            for pid, p in by_rank_last.items()}
        tail = sum(tot(p, "tail_w_stash", "mism") for p in by_rank_last.values())
        null = sum(tot(p, "tail_w_stash", "null") for p in by_rank_last.values())
        other = sum(tot(p, s, c) for p in by_rank_last.values() for s, c in
                    (("idx_w_dec", "mism"), ("tail_w_seed", "mism"), ("idx_w_pf", "s_mism"),
                     ("idx_w_pf", "d7_fallback"), ("pos", "pos_contig"), ("pos", "pos_bounds")))
        injected = sum((inj[t] or 0) for t in inj)
        V["guard_d2"] = dict(tail_mism=tail, tail_null=null, injected_tokens=injected, other_violations=other)
        check("H5g guard: tail_w_stash null-block mismatches only from injected tokens (multiple of the injected "
              "count over indexer layers); no other mapping, position or D7 violation anywhere",
              injected > 0 and tail == null and tail % injected == 0 and tail > 0 and other == 0,
              json.dumps(V["guard_d2"]))
    else:
        check("H5g guard polls present", False, "no polls", level="inconclusive")
    # exposure counters present at designated positions of forced requests
    miss = 0
    tot_d = 0
    for r in reqs:
        a = r0(r["tag"])
        if a is None or not a.get("desig"):
            continue
        x = a["exposure"].numpy()
        tot_d += len(a["desig"])
        miss += int((x[:, :, 0] < 0).all(axis=1).sum())
    check("H5h window-exposure records exist at every designated position (in-graph hook captured)",
          tot_d > 0 and miss == 0, f"{miss} of {tot_d} designated positions without any layer record")


def coverage():
    probes = [r for r in CL if r.get("kind") == "probe"]
    check("V0 unkeyed probes bitwise equal to B1 run-b1-a1", probes and all(p.get("equal") for p in probes),
          f"{[(p['id'], p.get('equal')) for p in probes]}")
    sk = [r for r in CL if r.get("kind") == "skip"]
    check("coverage: no phase skipped for time", not sk, f"{sk[:5]}", level="inconclusive")


if __name__ == "__main__":
    for fn in (coverage, h1, h2, h3, h4, h5):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            check(f"{fn.__name__} raised", False, repr(exc)[:300])
    verdict = "INVALID" if V["invalid"] else ("INCONCLUSIVE" if V["inconclusive"] else "VALID")
    V["verdict"] = verdict
    json.dump(V, open(os.path.join(RUN, "verdict.json"), "w"), indent=1, default=str)
    lines = [f"VERDICT {verdict}"] + ["%s %s  [%s]" % ("PASS" if c["ok"] else "FAIL", c["name"], c["detail"])
                                      for c in V["checks"]]
    open(os.path.join(RUN, "verdict.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
