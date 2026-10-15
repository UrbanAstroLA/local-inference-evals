#!/usr/bin/env python3
"""Frozen analysis of the first long-context contrast (PROPOSAL.md s.4-6 as amended by PLAN.md amendment 22).
CPU only. Frozen before data; nothing here is revised after viewing the contrast.

usage: analyze_fc.py RUN_DIR [FC_ROOT]   -> RUN_DIR/fc_result.json, fc_result.txt

Analysis order (frozen): 1 gates (GATE-1 as recorded in-session AND recomputed on the full data; per-run structural
validity; injection, guard and ledger; prefix sentinel and C3 residue) -> 2 within-arm variation -> 3 exposure and
selection -> 4 termination -> 5 LP construction comparison -> readings.

Primary statistic [A22-1] (kept as frozen; its limits are stated in every output): per fixture f and run (a, r),
y = ln( mean over o in Q_f of P_pre(</think> | o) ) (pre-top_p, logsumexp); run aggregate Y(a, r) = mean of y over the
8 fixtures (a fixed-fixture run aggregate; fixtures share a batch and D2 couples them: never independent samples).
It measures close propensity on this counterfactual fixture panel; it is not a finishing hazard and not a GPQA
completion probability. A pre-mask change that stays outside the nucleus is a computational finding whose relevance
to sampled termination is unestablished.
Contrast: Delta = mean_r Y(D, r) - mean_r Y(C, r); the three PAIRED differences d_r = Y(D_r) - Y(C_r) are reported
explicitly (C_r and D_r share a prefix construction and salt; the order is fixed in one session) [A22-2].
Operational screens (NOT significance tests; no exact permutation level is claimed) [A22-2]:
  beyond-null   |Delta| > 2 x max(N, 0.05 nats), N = max |Y - Y'| over the six within-arm run pairs
  separation    all three Y(D) on one side of all three Y(C)
Selected keys change (s.4.5): every D run selects corrupted window pools (n_sel_win_corrupt > 0 at designated
positions, pooled over fixtures and layers) AND every D-C selection Jaccard distance exceeds every within-arm one AND
the mean D-C distance exceeds 2 x the largest within-arm distance. Exposure failing either part = "insufficient
selected-key exposure".
Readings [A22-8], exactly:
  keys change AND termination screens met  -> "supported local computational mechanism candidate; behavioural relevance
      assessed via post-top_p and eligibility; no GPQA explanation"
  keys change, termination screens not met -> "no termination-propensity effect detected under this exposure and
      design; the mechanism is not excluded" (no effect detected by the prespecified screen)
  insufficient selected-key exposure       -> "uninformative"
  direction: D2 increasing close propensity is consistent with more corrupted old-engine computation helping
  termination; decreasing it supports a harmful D2 effect, and stock 0.7.0 finishing more often would then need an
  interaction or another contributor.
No reading unless every gate holds, each arm has 3 valid runs (else INCOMPLETE; never applied to 2 runs) and N <= 0.25.
"""
import collections
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fc_common as fc  # noqa: E402
import fc_geometry as geo  # noqa: E402
import gate_fc as gt  # noqa: E402

N_MAX = 0.25             # repeatability: no reading if the within-arm null exceeds 0.25 nats
N_FLOOR = 0.05           # operational floor of the beyond-null screen
FX_MIN = 0.5             # per-fixture screen: |Delta_f| >= 0.5 nats
DIR_FRAC = 0.75          # direction: >= 6 of 8 fixtures (ceil(0.75 n) in the R2 sensitivity)
D2_GROSS_MIN = 0.30      # injected runs: gross pool share reported against the attempt-2 30 % mark (not gated)
C_RUNS, D_RUNS = ["C1", "C2", "C3"], ["D1", "D2", "D3"]
LC_RUNS, LD_RUNS = ["LC1", "LC2", "LC3"], ["LD1", "LD2", "LD3"]
# [A22-8] readings and direction statements, verbatim
READING_1 = ("supported local computational mechanism candidate; behavioural relevance assessed via post-top_p and "
             "eligibility; no GPQA explanation")
READING_2 = ("no termination-propensity effect detected under this exposure and design; the mechanism is not excluded "
             "(no effect detected by the prespecified screen)")
READING_3 = "uninformative"
DIR_UP = ("D2 INCREASED close propensity: consistent with more corrupted old-engine computation helping termination")
DIR_DOWN = ("D2 DECREASED close propensity: supports a harmful D2 effect; stock 0.7.0 finishing more often would then "
            "need an interaction or another contributor")
LIMITS = ("close propensity on this counterfactual fixture panel (re-tokenized histories, prespecified positions, "
          "pre-top_p); not a finishing hazard and not a GPQA completion probability; 3 runs per arm in one session with "
          "a fixed order; screens are operational, no significance or equivalence is claimed")


def lse_mean(lp):
    lp = np.asarray(lp, dtype=np.float64)
    if lp.size == 0 or not np.isfinite(lp).all():
        return float("nan")
    m = lp.max()
    return float(m + math.log(np.mean(np.exp(lp - m))))


def ln_mean(p, floor=1e-30):
    p = np.asarray(p, dtype=np.float64)
    if p.size == 0 or not np.isfinite(p).all():
        return float("nan")
    return float(math.log(max(float(np.mean(p)), floor)))


# ------------------------------------------------------------------ per-run validity
def run_valid(run, G, ph, clean):
    R = gt.Rows()
    gt.structural(run, G, R, [ph], [], cv_tag=None, prefix=ph, warms=False)
    if clean:
        gt.clean_checks(run, G, R, [ph], prefix=ph)
    # both ranks measured identically (TP2 logits are gathered: rank equality is exact)
    bad = []
    for n in G["order"]:
        rr = run.req_ranks(f"{ph}-{n}")
        if len(rr) == 2:
            a, b = (rr[0]["metrics"].numpy(), rr[1]["metrics"].numpy())
            c = fc.cols(rr[0])["lp_pre_0"]
            if not np.array_equal(np.nan_to_num(a[:, c], nan=9.0), np.nan_to_num(b[:, c], nan=9.0)):
                bad.append(n)
    R.add(f"{ph}-ranks", "structural", "PASS" if not bad else "FAIL", "both ranks' termination rows identical", bad)
    return not R.failing(), R.rows


def injection_checks(run, G, R, d_runs, other_runs):
    """H5a: every injected request injected at least its window's tokens, none elsewhere; guard (H5g-r adapted):
    per rank, from the client's start dump to the last poll, tail_w_stash mism == null == owned layers x injected
    tokens, attributed only to injected phases; zero decode / seed / prefill / D7 / position deltas."""
    inj, bad = 0, []
    for ph in d_runs:
        for n in G["order"]:
            c = run.by_tag.get(f"{ph}-{n}")
            for rank, a in run.req_ranks(f"{ph}-{n}").items():
                v = a.get("d2_injected_tokens") or 0
                if c is None or v < c["W"]:
                    bad.append((ph, n, rank, v))
                if rank == 0:
                    inj += v
    for ph in other_runs:
        for n in G["order"]:
            for rank, a in run.req_ranks(f"{ph}-{n}").items():
                if a.get("d2_injected_tokens"):
                    bad.append((ph, n, rank, a["d2_injected_tokens"]))
    R.add("H5a-fc", "structural", "PASS" if not bad and inj > 0 else "FAIL", "injection exactly on the intended "
          "requests: every D / LD request injected >= its forced-list tokens, no other request injected", bad[:10])
    dumps = [r for r in run.CL if r.get("kind") == "dump"]
    start = [d for d in dumps if d.get("tag") == "start"]
    if not run.POLLS or not start:
        R.add("H5g-fc", "structural", "FAIL", "guard polls and a client start dump present", "missing")
        return inj
    SITES = (("tail_w_stash", "mism"), ("tail_w_stash", "null"), ("idx_w_dec", "mism"), ("tail_w_seed", "mism"),
             ("idx_w_pf", "s_mism"), ("idx_w_pf", "d7_fallback"), ("pos", "pos_contig"), ("pos", "pos_bounds"))
    val = lambda p, s, c: p["counters"].get(s, {}).get(c, 0)  # noqa: E731
    dts = sorted((d["t"], d["tag"]) for d in dumps)
    t0 = start[0]["t"]
    res, ok = {}, True
    for rank in sorted({p.get("rank") for p in run.POLLS}):
        pr = sorted([p for p in run.POLLS if p.get("rank") == rank], key=lambda p: p["t"])
        base = [p for p in pr if p["t"] >= t0 and p.get("reason") == "trigger"]
        if not base:
            ok = False
            res[rank] = "no start-dump poll"
            continue
        b0, last = base[0], pr[-1]
        d = {f"{s}.{c}": val(last, s, c) - val(b0, s, c) for s, c in SITES}
        nl = len(last.get("layers") or {})
        trig = [p for p in pr if p.get("reason") == "trigger" and p["t"] >= t0]
        per = {}
        for p1, p2 in zip(trig, trig[1:]):
            dv = val(p2, "tail_w_stash", "mism") - val(p1, "tail_w_stash", "mism")
            if dv:
                lab = [tg for td, tg in dts if td <= p2["t"]]
                per[lab[-1] if lab else "before-start"] = per.get(lab[-1] if lab else "before-start", 0) + dv
        others = {k: v for k, v in d.items() if not k.startswith("tail_w_stash") and v}
        rok = (d["tail_w_stash.mism"] == d["tail_w_stash.null"] == nl * inj and inj > 0 and not others
               and all(t in d_runs for t in per))
        ok &= rok
        res[rank] = dict(owned_layers=nl, deltas=d, expected_tail=nl * inj, by_phase=per, ok=rok)
    R.add("H5g-fc", "structural", "PASS" if ok else "FAIL", "guard per rank: every mismatched tail write went to the "
          "null tail block (mism == null == owned layers x injected tokens), only during D / LD phases; zero decode, "
          "seed, prefill, D7 and position mismatches: the injection cannot write into a shared cached prefix "
          "(index-K and tail writes; mapping level)", res)
    return inj


def ledger_vs_content(run, G, R, d_runs, lim):
    """The ledger agrees with the bytes: D runs' clean-labelled window pools no grosser than the GATE-1 content limit;
    corrupt-labelled pools reported (gross share)."""
    out = {}
    for ph in d_runs:
        cl, cx = [], []
        for n in G["order"]:
            lo, hi = G["fx"][n]["ref_pools"]
            rows = fc.compare_pools(run, f"{ph}-{n}", f"LW1-{n}", lo, hi)
            a = run.req(f"{ph}-{n}")
            if a is None:
                continue
            pc = a["pool_class"].numpy()
            P = G["fx"][n]["P"]
            for (rank, L, pool), v in rows.items():
                k = pool - P // 4
                if 0 <= k < pc.size:
                    (cl if pc[k] == 1 else cx if pc[k] == 2 else []).append(v)
        out[ph] = dict(clean_n=len(cl), clean_gross=fc.gross(cl), corrupt_n=len(cx), corrupt_gross=fc.gross(cx),
                       all_gross=fc.gross(cl + cx))
    ok = all(v["clean_n"] and v["clean_gross"] <= lim for v in out.values()) and bool(out)
    R.add("H5f-fc", "structural", "PASS" if ok else "FAIL", "ledger agrees with the bytes: clean-labelled D-run window "
          "pools within the GATE-1 content limit (%.4f%%); corrupt-labelled gross reported" % (100 * lim), out)
    R.add("H5d-fc", "reported", "REPORTED", "injected-run gross share over all compared window pools (attempt-2 mark 30 %, "
          "reported as exposure magnitude, not gated)", {k: v["all_gross"] for k, v in out.items()})


def prefix_sentinel(run, G, R):
    """[A22-4] clean prefix state after injected traffic: PS-a / PS-b (before D3; PS-b = determinism null) and PS-c
    (after D3) on s3's cached prefix. Pages of the last cached block [c0 - 1024, c0) must be byte-identical (exact);
    the KDA state rebuilt from the cached boundary state must be bitwise equal to PS-a's when PS-b is (else within
    2 x the PS-a / PS-b difference)."""
    page_bad, page_n, bt_diff, kda = [], 0, [], []
    for n in G["order"]:
        f = G["fx"][n]
        lo, hi = f["c0"] - geo.PS_DUMP, f["c0"]
        D = {k: run.win_dump(f"{k}-{n}") for k in ("PS-a", "PS-b", "PS-c")}
        if not all(D.values()):
            page_bad.append((n, "dump missing"))
            continue
        for rank in (0, 1):
            recs = {k: run.load(v[rank]) for k, v in D.items() if rank in v}
            if len(recs) != 3:
                page_bad.append((n, rank, "rank dump missing"))
                continue
            g0 = {k: [g for g in r["groups"] if g["bpk"] > 1][0]["ids"].tolist()[: hi // 64] for k, r in recs.items()}
            if not (g0["PS-a"] == g0["PS-b"] == g0["PS-c"]):
                bt_diff.append((n, rank))
            for nm, La in fc.paged_layers(recs["PS-a"]).items():
                typ = La.get("type", "")
                if "Tail" in typ:
                    continue                       # per-request tail ring and the null tail block: not prefix state
                pages = set()
                for r_ in recs.values():
                    grp = [g for g in r_["groups"] if g["bpk"] > 1][0]
                    ids = grp["ids"].tolist()
                    if La.get("shape", [0])[-1:] == [132]:
                        pages |= {(ids[(p // 4352) * 68] // 68) * 17 + ((p // 4) % 1088) // 64 for p in range(lo, hi, 4)}
                    elif "MLA" in typ:
                        pages |= {ids[p // 64] for p in range(lo, hi, 64)}
                    else:
                        g9 = [g for g in r_["groups"] if g["block_size"] == 1088]
                        if g9:
                            ids9 = g9[0]["ids"].tolist()
                            pages |= {ids9[p // 1088] for p in range(lo, hi, 1088) if p // 1088 < len(ids9)}
                for pg in sorted(pages):
                    xs = [r_["layers"].get(nm, {}).get("pages", {}).get(pg) for r_ in recs.values()]
                    page_n += 1
                    if any(x is None for x in xs) or not all(np.array_equal(xs[0].numpy(), x.numpy()) for x in xs[1:]):
                        page_bad.append((n, rank, fc.layer_no(nm), pg))
        S = {}
        for k in ("PS-a", "PS-b", "PS-c"):
            sn = run.snaps(f"{k}-{n}")
            S[k] = {rk: run.load(p) for F, ranks in sn for rk, p in ranks.items()} if sn else {}
        for rank in (0, 1):
            if not all(rank in S[k] for k in S):
                kda.append(dict(fixture=n, rank=rank, missing=True))
                continue
            for nm, La in fc.kda_layers(S["PS-a"][rank]).items():
                Lb, Lc = fc.kda_layers(S["PS-b"][rank]).get(nm), fc.kda_layers(S["PS-c"][rank]).get(nm)
                if Lb is None or Lc is None:
                    kda.append(dict(fixture=n, rank=rank, layer=nm, missing=True))
                    continue
                eq_ab = bool(np.array_equal(La["ssm_sketch"].numpy(), Lb["ssm_sketch"].numpy())) and \
                    bool(np.array_equal(La["conv"].float().numpy()[:3], Lb["conv"].float().numpy()[:3]))
                eq_ac = bool(np.array_equal(La["ssm_sketch"].numpy(), Lc["ssm_sketch"].numpy())) and \
                    bool(np.array_equal(La["conv"].float().numpy()[:3], Lc["conv"].float().numpy()[:3]))
                r_ab = fc.rel(Lb["ssm_sketch"][0], La["ssm_sketch"][0])
                r_ac = fc.rel(Lc["ssm_sketch"][0], La["ssm_sketch"][0])
                ok = eq_ac if eq_ab else (r_ac <= 2 * r_ab + 1e-7)
                kda.append(dict(fixture=n, rank=rank, layer=fc.layer_no(nm), eq_ab=eq_ab, eq_ac=eq_ac, rel_ab=r_ab,
                                rel_ac=r_ac, ok=ok))
    kbad = [k for k in kda if k.get("missing") or not k.get("ok")]
    ok = page_n > 0 and not page_bad and not kbad and kda
    R.add("PS-prefix", "structural", "PASS" if ok else "FAIL", "prefix sentinel: last cached (shared) block's pages "
          "byte-identical before and after D3 (MLA, index-K, attention pages); KDA state rebuilt from the cached "
          "boundary state bitwise equal (or within 2 x the PS-a / PS-b null when PS-a / PS-b differ)",
          dict(pages=page_n, page_mismatches=page_bad[:10], block_tables_differ=bt_diff,
               kda_bitwise_ab=sum(1 for k in kda if k.get("eq_ab")), kda_rows=len(kda), kda_failing=kbad[:10]))


# ------------------------------------------------------------------ statistics
def run_stats(run, G, ph, construction="long", id_col="lp_pre_0", positions="Q"):
    """{fixture: y} for one run; positions: Q | Q_noend | T | end | common (LP-comparable) | all8 (the positions of Q
    measured in steps with 8 requests and no FULL-graph padding, from the hook step log)."""
    out = {}
    st = run.steps_table(0) if positions == "all8" else None
    for n in G["order"]:
        f = G["fx"][n]
        a = run.req(f"{ph}-{n}")
        if a is None:
            out[n] = float("nan")
            continue
        M = a["metrics"].numpy().astype(np.float64)
        Cc = fc.cols(a)
        pos = {"Q": f["Q"], "Q_noend": f["Q_noend"], "T": f["T"], "end": [f["end"]], "common": f["lp_common"],
               "all8": f["Q"]}[positions]
        idx = [o - (geo.LP_SHIFT if construction == "short" else 0) for o in pos]
        idx = [i for i in idx if 0 <= i < M.shape[0]]
        if st is not None:
            idx = [i for i in idx if np.isfinite(M[i, Cc["step"]]) and
                   (st.get(int(M[i, Cc["step"]])) or (0, 0, 0, 0, 1))[0] == 8 and
                   not (st.get(int(M[i, Cc["step"]])) or (0, 0, 0, 0, 1))[4]]
        if id_col.startswith("p_post"):
            out[n] = ln_mean(M[idx, Cc[id_col]])
        elif id_col.startswith(("elig_lo", "elig_hi")):
            out[n] = float(np.mean(M[idx, Cc[id_col]] > 0.5)) if idx else float("nan")
        elif id_col.startswith("floored"):
            out[n] = float(np.mean(np.maximum(M[idx, Cc["lp_pre_0"]], math.log(1e-9))))
        elif id_col.startswith("meanlog"):
            out[n] = float(np.mean(M[idx, Cc["lp_pre_0"]]))
        else:
            out[n] = lse_mean(M[idx, Cc[id_col]])
    return out


def agg(y, fixtures, weights=None):
    v = [y[n] for n in fixtures]
    if weights is None:
        return float(np.mean(v))
    return float(np.mean([np.mean([y[n] for n in grp if n in fixtures]) for grp in weights if
                          any(n in fixtures for n in grp)]))


def contrast(Yc, Yd, paired=True):
    """Yc, Yd: lists of run aggregates (index r - 1). Returns Delta, paired d_r, N, screens."""
    if len(Yc) < 3 or len(Yd) < 3 or not all(np.isfinite(Yc + Yd)):
        return dict(complete=False)
    delta = float(np.mean(Yd) - np.mean(Yc))
    N = max(abs(a - b) for a, b in itertools.combinations(Yc, 2))
    N = max(N, max(abs(a - b) for a, b in itertools.combinations(Yd, 2)))
    sep = min(Yd) > max(Yc) or max(Yd) < min(Yc)
    d = [float(b - a) for a, b in zip(Yc, Yd)] if paired else None
    beyond = abs(delta) > 2 * max(N, N_FLOOR)
    return dict(complete=True, Y_C=Yc, Y_D=Yd, delta=delta, paired_d=d,
                paired_same_sign=(all(x > 0 for x in d) or all(x < 0 for x in d)) if d else None,
                N=N, threshold=2 * max(N, N_FLOOR), beyond_null=beyond, separation=sep, screen_met=beyond and sep,
                repeatable=N <= N_MAX)


def fixture_screens(yc, yd, fixtures):
    """per fixture: Delta_f, N_f, screen (|Delta_f| >= 0.5, > 2 max(N_f, 0.05), separation); direction count."""
    out = {}
    for n in fixtures:
        c = [y[n] for y in yc]
        d = [y[n] for y in yd]
        if len(c) < 3 or len(d) < 3:
            continue
        Df = float(np.mean(d) - np.mean(c))
        Nf = max(max(abs(a - b) for a, b in itertools.combinations(c, 2)),
                 max(abs(a - b) for a, b in itertools.combinations(d, 2)))
        sep = min(d) > max(c) or max(d) < min(c)
        out[n] = dict(delta=Df, N=Nf, screen=abs(Df) >= FX_MIN and abs(Df) > 2 * max(Nf, N_FLOOR) and sep)
    signs = [np.sign(v["delta"]) for v in out.values()]
    need = math.ceil(DIR_FRAC * len(signs))
    pos, neg = sum(1 for s in signs if s > 0), sum(1 for s in signs if s < 0)
    return out, dict(n=len(signs), positive=pos, negative=neg, need=need, met=max(pos, neg) >= need)


def selection_sets(run, G, ph):
    """{(fixture, o, rank, layer_name): frozenset(pool ids)} at selection-record positions."""
    out = {}
    for n in G["order"]:
        for rank, a in run.req_ranks(f"{ph}-{n}").items():
            sel = a.get("selections")
            slots = {v: k for k, v in (a.get("layer_slots") or {}).items()}
            if sel is None:
                continue
            for k, o in enumerate(a.get("selpos") or []):
                for L in range(sel.shape[1]):
                    row = sel[k, L].numpy()
                    if row[0] < 0 or L not in slots:
                        continue
                    ids = row[1:]
                    out[(n, o, rank, slots[L])] = frozenset(ids[ids >= 0].tolist())
    return out


def jaccard(A, B):
    ks = sorted(set(A) & set(B))
    if not ks:
        return float("nan"), 0
    return float(np.mean([1 - len(A[k] & B[k]) / max(len(A[k] | B[k]), 1) for k in ks])), len(ks)


def exposure_table(run, G, runs):
    """Selected corrupted window pools by run x fixture x layer at designated positions: sums of n_sel,
    n_sel_win, n_sel_win_corrupt, n_win, n_win_corrupt (rank-0 and rank-1 layers together)."""
    out, per_run = {}, {}
    for ph in runs:
        tot = collections.Counter()
        for n in G["order"]:
            for rank, a in run.req_ranks(f"{ph}-{n}").items():
                x = a["exposure"].numpy()
                Xc = {nm: i for i, nm in enumerate(a["xcols"])}
                slots = {v: k for k, v in (a.get("layer_slots") or {}).items()}
                for L in range(x.shape[1]):
                    if L not in slots:
                        continue
                    blk = x[:, L, :]
                    blk = blk[blk[:, 0] >= 0]
                    if not blk.size:
                        continue
                    key = "%s|%s|L%d" % (ph, n, fc.layer_no(slots[L]))
                    c = {k: int(blk[:, Xc[k]].sum()) for k in ("n_sel", "n_sel_win", "n_sel_win_corrupt", "n_win",
                                                                 "n_win_corrupt")}
                    c["positions"] = int(blk.shape[0])
                    c["share_selected_corrupt"] = c["n_sel_win_corrupt"] / c["n_sel"] if c["n_sel"] else None
                    out[key] = c
                    tot.update({k: v for k, v in c.items() if isinstance(v, int)})
        per_run[ph] = dict(tot)
    return out, per_run


def main():
    rd = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(HERE)
    G = geo.load(root)
    run = fc.Run(rd)
    R = gt.Rows()
    res = dict(limits=LIMITS, labels="A22-n = PLAN.md amendment 22 item n")
    # ---------------- 1. gates
    try:
        g_sess = json.load(open(os.path.join(rd, "gate1", "gate1.json")))
    except Exception:  # noqa: BLE001
        g_sess = None
    Rg, vg, reading_g = gt.gate1(run, G)
    res["gate1"] = dict(in_session=(g_sess or {}).get("verdict"), recomputed=vg, reading=reading_g,
                        failing=dict(structural=Rg.failing("structural"), numerical=Rg.failing("numerical")))
    R.add("GATE1", "structural", "PASS" if (g_sess or {}).get("verdict") == "PASS" and vg == "PASS" else "FAIL",
          "GATE-1 passed in-session (before any injected batch) and recomputes to PASS on the full data",
          res["gate1"])
    valid = {}
    vrows = {}
    for ph in C_RUNS + LC_RUNS:
        valid[ph], vrows[ph] = run_valid(run, G, ph, clean=True)
    for ph in D_RUNS + LD_RUNS:
        valid[ph], vrows[ph] = run_valid(run, G, ph, clean=False)
    res["valid_runs"] = valid
    res["run_rows_failing"] = {ph: [r["id"] for r in rows if r["status"] == "FAIL"] for ph, rows in vrows.items()}
    injection_checks(run, G, R, [p for p in D_RUNS + LD_RUNS if p in run.by_tag or f"{p}-L1" in run.by_tag],
                     C_RUNS + LC_RUNS)
    lim = None
    for r_ in Rg.rows:
        if r_["id"] == "G1-content" and isinstance(r_["detail"], dict) and "C1" in r_["detail"]:
            lim = r_["detail"]["C1"]["limit"]
    if lim is not None and lim == lim:
        ledger_vs_content(run, G, R, [p for p in D_RUNS if f"{p}-L1" in run.by_tag], lim)
    prefix_sentinel(run, G, R)
    # C3 residue: alignment and content against the references, same rules and nulls as GATE-1
    R3 = gt.Rows()
    gt.alignment(run, G, R3, [("C3", "LW1"), ("C3", "LW2")], prefix="C3")
    gt.content_gate(run, G, R3, ["C3"], ("C1", "C2"), ("LW1", "LW2"), "LW1", prefix="C3")
    c3ok = not R3.failing() and valid.get("C3")
    R.add("C3-residue", "structural", "PASS" if c3ok else "FAIL", "C3 (after injected traffic, on s3's prefix): valid "
          "and clean run, aligned with the prefill references and within the GATE-1 content rule",
          dict(failing=R3.failing(), valid=valid.get("C3")))
    res["c3_rows"] = R3.rows
    # ---------------- 2-4. statistics
    vC = [p for p in C_RUNS if valid.get(p)]
    vD = [p for p in D_RUNS if valid.get(p)]
    order = G["order"]
    flagged = list((json.load(open(os.path.join(rd, "gate1", "gate1.json"))) if g_sess else {}).get(
        "flagged_fixtures", []) or Rg.data.get("flagged", []))
    unflagged = [n for n in order if n not in flagged]
    units = list(geo.doc_units(G).values())
    near = [n for n in order if G["fx"][n]["near_close"]]
    active = [n for n in order if not G["fx"][n]["near_close"]]
    y = {ph: run_stats(run, G, ph) for ph in C_RUNS + D_RUNS}

    complete = len(vC) == 3 and len(vD) == 3
    pr = contrast([agg(y[p], order) for p in C_RUNS], [agg(y[p], order) for p in D_RUNS]) if complete else \
        dict(complete=False)
    res["primary"] = dict(statistic="Y = mean over 8 fixtures of ln mean_{o in Q_f} P_pre(</think>)", **pr,
                          per_fixture_y={p: y[p] for p in C_RUNS + D_RUNS})
    fxs, direc = fixture_screens([y[p] for p in C_RUNS], [y[p] for p in D_RUNS], order) if complete else ({}, {})
    res["fixture_screens"], res["direction_screen"] = fxs, direc
    sens = {}
    if complete:
        def cs(ycol="lp_pre_0", pos="Q", fx=order, w=None):
            yc = [run_stats(run, G, p, "long", ycol, pos) for p in C_RUNS]
            yd = [run_stats(run, G, p, "long", ycol, pos) for p in D_RUNS]
            return contrast([agg(v, fx, w) for v in yc], [agg(v, fx, w) for v in yd])
        sens["end_position_excluded"] = cs(pos="Q_noend")
        sens["question_weighted"] = cs(w=units)
        sens["near_close"] = cs(fx=near)
        sens["active"] = cs(fx=active)
        sens["tail_T"] = cs(pos="T")
        sens["end_position_alone"] = cs(pos="end")
        sens["all_8_in_flight_positions"] = cs(pos="all8")
        sens["floored_mean_lnP"] = cs("floored")
        sens["unfloored_mean_lnP"] = cs("meanlog")
        sens["post_top_p_close"] = cs("p_post_0")
        sens["eligibility_close_share_lo"] = cs("elig_lo_0")
        sens["eligibility_close_share_hi"] = cs("elig_hi_0")
        for tid, ci in ((154820, 1), (154827, 2), (154829, 3)):
            sens["stop_%d_pre" % tid] = cs("lp_pre_%d" % ci)
            sens["stop_%d_post" % tid] = cs("p_post_%d" % ci)
            sens["stop_%d_elig_lo" % tid] = cs("elig_lo_%d" % ci)
        if flagged:
            lf = cs(fx=unflagged)
            yc_ = [y[p] for p in C_RUNS]
            yd_ = [y[p] for p in D_RUNS]
            fxs2, dir2 = fixture_screens(yc_, yd_, unflagged)
            sens["R2_leave_flagged_out"] = dict(flagged=flagged, contrast=lf, direction=dir2)
        else:
            sens["R2_leave_flagged_out"] = dict(flagged=[], note="no fixture flagged: identical to the primary")
    res["sensitivities"] = sens
    # exposure and selection
    ex, ex_run = exposure_table(run, G, C_RUNS + D_RUNS + LC_RUNS + LD_RUNS)
    res["exposure_by_run_fixture_layer"] = ex
    res["exposure_by_run"] = ex_run
    expo_ok = complete and all(ex_run.get(p, {}).get("n_sel_win_corrupt", 0) > 0 for p in D_RUNS) and \
        all(ex_run.get(p, {}).get("n_sel_win_corrupt", 1) == 0 for p in C_RUNS)
    S = {p: selection_sets(run, G, p) for p in C_RUNS + D_RUNS}
    within = [jaccard(S[a], S[b])[0] for grp in (C_RUNS, D_RUNS) for a, b in itertools.combinations(grp, 2)]
    across = [jaccard(S[a], S[b])[0] for a in C_RUNS for b in D_RUNS]
    div_ok = complete and all(np.isfinite(within + across)) and min(across) > max(within) and \
        float(np.mean(across)) > 2 * max(within)
    res["selection"] = dict(exposure_ok=bool(expo_ok), within_arm_jaccard=within, across_arm_jaccard=across,
                            divergence_ok=bool(div_ok), keys_change=bool(expo_ok and div_ok))
    # ---------------- 5. LP construction comparison (2 x 2 at common positions)
    lp = {}
    if all(f"{p}-L1" in run.by_tag for p in LC_RUNS + LD_RUNS):
        Ylong = {p: agg(run_stats(run, G, p, "long", positions="common"), order) for p in C_RUNS + D_RUNS}
        Yshort = {p: agg(run_stats(run, G, p, "short", positions="common"), order) for p in LC_RUNS + LD_RUNS}
        lp["construction_clean"] = contrast([Yshort[p] for p in LC_RUNS], [Ylong[p] for p in C_RUNS], paired=False)
        lp["construction_injected"] = contrast([Yshort[p] for p in LD_RUNS], [Ylong[p] for p in D_RUNS], paired=False)
        lp["injection_long"] = contrast([Ylong[p] for p in C_RUNS], [Ylong[p] for p in D_RUNS])
        lp["injection_short"] = contrast([Yshort[p] for p in LC_RUNS], [Yshort[p] for p in LD_RUNS])
        if lp["injection_long"].get("complete") and lp["injection_short"].get("complete"):
            inter = lp["injection_long"]["delta"] - lp["injection_short"]["delta"]
            Nint = max(lp["injection_long"]["N"], lp["injection_short"]["N"])
            lp["interaction"] = dict(delta=inter, N=Nint, beyond_null=abs(inter) > 2 * 2 * max(Nint, N_FLOOR),
                                     rule="|(D - C) - (LD - LC)| > 2 x (2 x max(N, 0.05)): the difference of two "
                                          "contrasts carries both nulls")
        lp["valid_runs"] = {p: valid.get(p) for p in LC_RUNS + LD_RUNS}
        lp["name"] = ("history-construction and accumulated-decode-state comparison at fixed total context (NOT an "
                      "output-length experiment)")
    res["lp_control"] = lp
    # ---------------- readings
    gates_ok = not R.failing() and res["gate1"]["recomputed"] == "PASS"
    if not gates_ok:
        reading = "NO READING: a gate failed (%s); results are instrument findings only" % R.failing()
    elif not complete:
        reading = "INCOMPLETE: fewer than 3 valid runs in an arm (C %s, D %s); the rule is never applied to 2 runs" % (
            vC, vD)
    elif not pr["repeatable"]:
        reading = "NO READING: within-arm variation N = %.3f nats > 0.25 (variation reported)" % pr["N"]
    elif not res["selection"]["keys_change"]:
        reading = READING_3 + " (insufficient selected-key exposure: corrupted pools selected in every D run %s, " \
            "selection divergence beyond the within-arm null %s)" % (expo_ok, div_ok)
    elif pr["screen_met"]:
        reading = READING_1
    else:
        reading = READING_2
    direction = None
    if pr.get("complete"):
        direction = DIR_UP if pr["delta"] > 0 else DIR_DOWN
    res["reading"] = reading
    res["direction_statement"] = direction
    res["flagged_fixtures"] = flagged
    res["notes"] = ["flagged fixtures carry no mechanism claim (R1); every reading is repeated in the R2 sensitivity",
                    "rejection-dependent signals are not attributed to the served D6 mechanism without the oracle-draft "
                    "bracket (21 s.7)", "a null result excludes only decode-window D2 exposure on these 8 fixtures "
                    "below 131,072; not full-history accumulation, D6, or longer ranges"]
    res["gate_rows"] = R.rows
    json.dump(res, open(os.path.join(rd, "fc_result.json"), "w"), indent=1, default=str)
    lines = ["READING: %s" % reading, "DIRECTION: %s" % direction, "LIMITS: %s" % LIMITS,
             "GATE-1 in-session %s, recomputed %s" % (res["gate1"]["in_session"], vg)]
    lines += ["%-8s %-14s %s" % (r_["status"], r_["id"], json.dumps(r_["detail"], default=str)[:200]) for r_ in R.rows]
    lines.append("valid runs: %s" % valid)
    if pr.get("complete"):
        lines.append("primary: Y_C %s Y_D %s Delta %.4f paired %s N %.4f threshold %.4f beyond %s separation %s" % (
            [round(v, 4) for v in pr["Y_C"]], [round(v, 4) for v in pr["Y_D"]], pr["delta"],
            [round(v, 4) for v in pr["paired_d"]], pr["N"], pr["threshold"], pr["beyond_null"], pr["separation"]))
    lines.append("selection: %s" % json.dumps(res["selection"]))
    open(os.path.join(rd, "fc_result.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
