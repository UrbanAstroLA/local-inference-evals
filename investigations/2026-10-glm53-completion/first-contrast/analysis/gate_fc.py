#!/usr/bin/env python3
"""GATE-1: in-session validation of the long-context instrument BEFORE any injected batch (PLAN.md amendment 22
items 3-5; PROPOSAL.md s.5). CPU only. Frozen before data.

usage: gate_fc.py RUN_DIR [FC_ROOT]   -> RUN_DIR/gate1.json, gate1.txt; rc 0 iff verdict PASS
RUN_DIR holds client.jsonl and forcewin/, mapguard/ copied out of the serving container by runner/gate_step.sh after
stage A (V0, LW1, LW2, C1, C2, CV). The same functions are re-evaluated on the full run by analyze_fc.py.

Pre-stated reading [A22-3/4]: a failing NUMERICAL row (alignment, crossings, regime, content, CV numerics) means the
long-context instrument is unvalidated under these bounds. It does not make B1 defective, and no threshold is revised
after viewing data. A failing STRUCTURAL row means the run is INVALID. Either way no injected batch is run.

Rows (status PASS / FAIL gate; REPORTED never gates):
 structural  G1-V0, G1-coverage, G1-cache (exact cache accounting), G1-wmax (WMAX = 8448 on both ranks, every keyed
             request's hook record carries its full planned list: no truncation), G1-abspos (hook P = planned prompt
             length on cached prefixes, dump ranges and snapshot frontiers exact), G1-forcing (ids_equal; every output
             index measured exactly once; references once per row), G1-errors, G1-attrib (step-matched exposure and
             selection records at every designated position, separately in flattened and padded steps), G1-inflight
             (all 8 decoding before output index 512), G1-clean (no injection, ledger clean, no corrupt pools selected
             or present in C runs), G1-idx-struct (dump attribution and population, pages <= 25 % gross, zero gross in
             DSA layers 3-15), G1-CV-slot, G1-CV-book, G1-CV-cross, G1-CV-persist
 numerical   G1-align (C1 vs LW1, C2 vs LW2), G1-cross (both block crossings of every window), G1-regime (M3's 65,536
             selection-regime crossing; with exact selection-record validity), G1-content (21g formula with the
             [A22-3] nulls: N_dd = C1 vs C2, N_pp = LW1 vs LW2 at a different build length, matched by fixture x layer x
             shared causal position), G1-CV-conv and G1-CV-ssm (vs the matched decode reference: poison)
 reported    the original 0.1 % content criterion beside the 21g rule; content against LW2; per fixture x layer tables;
             fixture flags (R3: layer-units and positions) and arm asymmetry; crossing-neighbourhood content;
             reference-vs-reference alignment (LW1 vs LW2); CV identification disagreements
"""
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fc_common as fc  # noqa: E402
import fc_geometry as geo  # noqa: E402

XW = 128                 # half-width of a crossing window (output indices)
RW = 256                 # half-width of M3's regime window


class Rows:
    def __init__(self):
        self.rows, self.data = [], {}

    def add(self, rid, kind, status, name, detail, label="A22"):
        self.rows.append(dict(id=rid, kind=kind, status=status, name=name, detail=detail, label=label))

    def failing(self, kind=None):
        return [r["id"] for r in self.rows if r["status"] == "FAIL" and (kind is None or r["kind"] == kind)]


def hook_add(run):
    """{(tag, rank): add record} from the hook logs."""
    out = {}
    for r in run.FWLOG:
        if r.get("kind") == "add":
            out[(r.get("tag"), r.get("rank", 0))] = r
    return out


def structural(run, G, R, runs, refs, cv_tag="CV", prefix="G1", warms=True):
    """runs: forced-run phase tags (C/D/LC/LD); refs: LW phase tags. Exact checks."""
    reqs = [r for r in run.reqs if r["phase"] in runs + refs + [cv_tag]
            or (warms and r["phase"].startswith(("warm", "rewarm")))]
    keyed = [r for r in reqs if r.get("keyed")]
    # ---- coverage
    exp = {f"{p}-{n}" for p in runs + refs for n in G["order"]}
    exp |= {f"CV-{m.replace(':', '_')}-{n}" for n in geo.CV_FIXTURES for m in geo.CV_MODES} if cv_tag else set()
    missing = sorted(exp - set(run.by_tag))
    errs = [r["tag"] for r in reqs if r.get("error")]
    nohook = sorted({r["tag"] for r in keyed if run.req(r["tag"]) is None})
    R.add(f"{prefix}-coverage", "structural", "PASS" if not missing and not errs and not nohook else "FAIL",
          "every planned request present, returned without error, with its hook record on disk",
          dict(missing=missing[:10], errors=errs[:10], no_hook_record=nohook[:10], expected=len(exp)))
    # ---- exact cache accounting
    bad = [(r["tag"], r.get("cached_tokens"), r.get("cached_expected")) for r in reqs
           if not r["phase"].startswith(("warm", "rewarm")) and r.get("cached_tokens") != r.get("cached_expected")]
    warms = [r for r in reqs if r["phase"].startswith(("warm", "rewarm"))]
    wbad = [(r["tag"], r.get("cached_tokens")) for r in warms if r.get("cached_tokens") is None
            or r["cached_tokens"] % geo.BLOCK or r["cached_tokens"] > r.get("hit_value", 0)
            or (not r.get("rewarm") and r["cached_tokens"] != 0)]
    hits = collections.Counter("hit" if r.get("cached_tokens") == r.get("hit_value") and r.get("rewarm") else
                               ("cold" if not r.get("rewarm") else "miss/partial") for r in warms)
    R.add(f"{prefix}-cache", "structural", "PASS" if not bad and not wbad else "FAIL",
          "exact cache accounting: every reference, run, CV and sentinel request's cached_tokens equals its planned "
          "value (0 cold; c0 / c0_L on a cached prefix); warms 0 when cold, a block multiple <= the hit value on re-warm",
          dict(violations=bad[:10], warm_violations=wbad[:10], warms=dict(hits)))
    # ---- WMAX without truncation, absolute positions
    adds = hook_add(run)
    inits = [r for r in run.FWLOG if r.get("kind") == "init"]
    wm = sorted({(r.get("rank", 0), r.get("wmax")) for r in inits})
    wbad = []
    for r in keyed:
        for rank in (0, 1):
            a = adds.get((r["tag"], rank))
            if a is None:
                wbad.append((r["tag"], rank, "no add record"))
            elif a.get("W") != r["W"]:
                wbad.append((r["tag"], rank, "hook W %s != planned %s" % (a.get("W"), r["W"])))
    R.add(f"{prefix}-wmax", "structural", "PASS" if wm and all(w == geo.WMAX for _, w in wm) and {k for k, _ in wm}
          == {0, 1} and not wbad else "FAIL", "WMAX = 8448 on both ranks; every keyed request's hook record holds its "
          "full planned forced list (no truncation)", dict(init=wm, violations=wbad[:10]))
    pbad = []
    for r in keyed:
        for rank in (0, 1):
            a = adds.get((r["tag"], rank))
            if a is not None and a.get("P") != r["P"]:
                pbad.append((r["tag"], rank, a.get("P"), r["P"]))
        rq = run.req(r["tag"])
        if rq is not None and rq.get("P") != r["P"]:
            pbad.append((r["tag"], "req", rq.get("P"), r["P"]))
        if r.get("dumpwin"):
            wd = run.win_dump(r["tag"])
            if not wd:
                pbad.append((r["tag"], "no window dump"))
            else:
                for rank, p in wd.items():
                    s = run.load(p)
                    want = list(r["dumprange"]) if r.get("dumprange") else [r["P"], r["P"] + r["W"]]
                    if s.get("tag") != r["tag"] or s.get("P") != r["P"] or list(s.get("range") or []) != want:
                        pbad.append((r["tag"], rank, "dump", s.get("tag"), s.get("P"), s.get("range"), want))
        if r.get("snap"):
            for F, ranks in run.snaps(r["tag"]):
                for rank, p in ranks.items():
                    s = run.load(p)
                    if s.get("P") != r["P"] or s.get("frontier_pos") != r["P"] + F - 2 or not s.get("tokens_ok"):
                        pbad.append((r["tag"], rank, "snap", F, s.get("P"), s.get("frontier_pos"), s.get("tokens_ok")))
    R.add(f"{prefix}-abspos", "structural", "PASS" if not pbad else "FAIL",
          "absolute positions on cached prefixes: hook prompt length = planned P (both ranks); dumps carry their tag, P "
          "and planned range; snapshots: frontier = P + F - 2 and committed tokens equal the history", pbad[:10])
    # ---- forcing
    fbad = []
    for r in keyed:
        rq = run.req_ranks(r["tag"])
        if not rq:
            continue
        if r["W"] and r.get("ids_equal") is not True:
            fbad.append((r["tag"], "ids_equal", r.get("ids_equal")))
        for rank, a in rq.items():
            nm = a["nmeas"].numpy()
            if r["role"] == "ref":
                n = r.get("build_len", r["P"]) - r["ref_from"] + 1
                n = min(n, geo.WMAX + 1)
                ok = (nm[:n] == 1).all() and (nm[n:] == 0).all()
            else:
                ok = nm.shape[0] == r["W"] and (nm == 1).all()
            if not ok:
                fbad.append((r["tag"], rank, "nmeas", int(nm.min()), int(nm.max()), int(nm.shape[0])))
            if a.get("errors"):
                fbad.append((r["tag"], rank, "errors", a.get("errors")))
    R.add(f"{prefix}-forcing", "structural", "PASS" if not fbad else "FAIL", "forced ids committed exactly (ids_equal) "
          "and every output index measured exactly once; every reference row measured once", fbad[:10])
    ferr = [r for r in run.FWLOG if r.get("kind") in ("fw_error", "guard_error")]
    R.add(f"{prefix}-errors", "structural", "PASS" if not ferr else "FAIL", "no fw_error / guard_error",
          [str(e)[:200] for e in ferr[:5]])
    # ---- attribution in flattened and padded steps
    steps = run.steps_table(0)
    cnt = collections.Counter()
    ex = []
    for r in keyed:
        if r["role"] not in ("C", "D", "LC", "LD"):
            continue
        a = run.req(r["tag"])
        if a is None or not a.get("desig"):
            continue
        Cc = fc.cols(a)
        M = a["metrics"].numpy()
        x = a["exposure"].numpy()
        nsl = len(a.get("layer_slots") or {})
        for k, o in enumerate(a["desig"]):
            if o < 1 or o >= r["W"]:
                continue
            ms = M[o, Cc["step"]]
            cls = "padded" if np.isfinite(ms) and (steps.get(int(ms), (0,) * 5)[4] or 0) > 0 else "flattened"
            cnt[cls + "_n"] += 1
            st = x[k, :, 0]
            pres = st >= 0
            if not pres.any():
                cnt[cls + "_missing"] += 1
                ex.append((r["tag"], o, "missing")) if len(ex) < 5 else None
            elif not np.isfinite(ms) or not (st[pres] == int(ms)).all() or (nsl and not (st[:nsl] == int(ms)).all()):
                cnt[cls + "_misattributed"] += 1
                ex.append((r["tag"], o, cls)) if len(ex) < 5 else None
        sel = a.get("selections")
        for k, o in enumerate(a.get("selpos") or []):
            if o < 1 or o >= r["W"] or sel is None or k >= sel.shape[0]:
                continue
            ss = sel[k, :, 0].numpy()
            ms = M[o, Cc["step"]]
            cnt["sel_n"] += 1
            if not (ss >= 0).any() or not np.isfinite(ms) or not (ss[ss >= 0] == int(ms)).all():
                cnt["sel_bad"] += 1
    okat = cnt["flattened_n"] > 0 and not any(cnt[c + s] for c in ("flattened", "padded")
                                              for s in ("_missing", "_misattributed")) and not cnt["sel_bad"]
    R.add(f"{prefix}-attrib", "structural", "PASS" if okat else "FAIL", "exact request attribution: every designated "
          "position (o >= 1) has records in every tracked layer whose step is its measurement step, in flattened and in "
          "padded (FULL-graph padding) steps alike; selection records likewise", dict(cnt, examples=ex,
                                                                                    padded_exercised=cnt["padded_n"] > 0))
    # ---- in flight
    ib = []
    for ph in runs:
        rs = [run.req(f"{ph}-{n}") for n in G["order"]]
        if any(a is None for a in rs):
            continue
        first = [a["metrics"].numpy()[1, fc.cols(a)["step"]] for a in rs]
        at512 = [a["metrics"].numpy()[min(512, a["metrics"].shape[0] - 1), fc.cols(a)["step"]] for a in rs]
        if not (np.nanmax(first) <= np.nanmin(at512)):
            ib.append((ph, float(np.nanmax(first)), float(np.nanmin(at512))))
    R.add(f"{prefix}-inflight", "structural", "PASS" if not ib else "FAIL", "all 8 requests of every run decoding "
          "before output index 512 of every request (fixed batch composition)", ib)
    return reqs


def clean_checks(run, G, R, runs, prefix="G1"):
    """C runs: no injection, ledger clean over processed window pools, no corrupt pool present or selected."""
    bad = []
    for ph in runs:
        for n in G["order"]:
            tag = f"{ph}-{n}"
            c = run.by_tag.get(tag)
            for rank, a in run.req_ranks(tag).items():
                if a.get("d2_injected_tokens"):
                    bad.append((tag, rank, "injected", a["d2_injected_tokens"]))
                P, W = c["P"], c["W"]
                pc = a["pool_class"].numpy()
                for i in range((P + 3) // 4, (P + W - 1) // 4):
                    k = i - P // 4
                    if not (0 <= k < pc.size) or pc[k] != 1:
                        bad.append((tag, rank, "ledger", i, int(pc[k]) if 0 <= k < pc.size else None))
                        break
                x = a["exposure"].numpy()
                Xc = {nm: i for i, nm in enumerate(a["xcols"])}
                v = x[:, :, [Xc["n_win_corrupt"], Xc["n_sel_win_corrupt"]]]
                if (v > 0).any():
                    bad.append((tag, rank, "corrupt pools present/selected", int((v > 0).sum())))
    R.add(f"{prefix}-clean", "structural", "PASS" if not bad else "FAIL", "clean runs: d2_injected_tokens = 0; shadow "
          "ledger clean on every processed window pool; n_win_corrupt = n_sel_win_corrupt = 0 at every designated "
          "position", bad[:10])


def alignment(run, G, R, pairs, prefix="G1"):
    """pairs: [(dec_phase, ref_phase)]. Gated H2-style rule on output indices [512, ref coverage)."""
    tol = fc.align_tol()
    allD, allA, per, cross, regime = [], [], {}, {}, {}
    for dph, rph in pairs:
        for n in G["order"]:
            f = G["fx"][n]
            dec, ref = run.req(f"{dph}-{n}"), run.req(f"{rph}-{n}")
            if dec is None or ref is None or ref.get("ref") != f["P"]:
                per[f"{dph}/{rph}-{n}"] = None
                continue
            hi = run.by_tag[f"{rph}-{n}"].get("build_len", f["P_L"]) - f["P"]
            s = fc.align_summary(dec, ref, geo.LEAD, min(hi, f["L"]))
            per[f"{dph}/{rph}-{n}"] = {k: v for k, v in s.items() if not k.startswith("_")}
            allD.append(s["_d"])
            allA.append(s["_a"])
            for j, ob in enumerate(f["crossings_o"]):
                if ob + XW <= hi:
                    c = fc.align_summary(dec, ref, max(geo.LEAD, ob - XW), ob + XW)
                    cross["%s/%s-%s-x%d" % (dph, rph, n, j + 1)] = {k: v for k, v in c.items() if not k.startswith("_")}
            for ob in f["regime_o"]:
                if ob + RW <= hi:
                    c = fc.align_summary(dec, ref, ob - RW, ob + RW)
                    regime[f"{dph}/{rph}-{n}"] = {k: v for k, v in c.items() if not k.startswith("_")}
    d = np.concatenate(allD) if allD else np.array([])
    a = np.concatenate(allA) if allA else np.array([])
    top1 = float(np.mean(a)) if a.size else float("nan")
    lpq = {str(p): fc.q(d, p) for p in fc.FLOOR}
    badr = [k for k, v in per.items() if v is None or not (v["top1"] >= fc.SHIFT_ALIGNED_MIN
                                                           and v["shifted"] <= fc.SHIFT_SHIFTED_MAX)]
    ok = bool(a.size) and top1 >= fc.TOP1_MIN and all(fc.q(d, p) <= tol[p] for p in fc.FLOOR) and not badr
    R.add(f"{prefix}-align", "numerical", "PASS" if ok else "FAIL", "long-context alignment: forced decode vs prefill "
          "reference on output indices >= 512: pooled top-1 >= 0.95 and |dlp| quantiles within max(2 x carried null, "
          "floors); per request aligned top-1 >= 0.90 and top-1 at +-1 <= 0.20", dict(
              pairs=pairs, n=int(a.size), top1=top1, lp=lpq, tol={str(k): v for k, v in tol.items()},
              failing=badr[:10], per_request=per), "21g-3 rule; A22-4")
    # crossings: every crossing of every fixture covered by at least one pair, each covered window passing
    need = {(n, j) for n in G["order"] for j in range(len(G["fx"][n]["crossings_o"]))}
    cov = {(k.split("-")[-2], int(k.split("-x")[-1]) - 1) for k in cross}
    cbad = [k for k, v in cross.items() if not (v["n"] > 0 and v["top1"] >= fc.SHIFT_ALIGNED_MIN
                                                and v["shifted"] <= fc.SHIFT_SHIFTED_MAX)]
    R.add(f"{prefix}-cross", "numerical", "PASS" if need <= cov and not cbad else "FAIL", "both block crossings of "
          "every window: in +-128 output indices around each crossing, aligned top-1 >= 0.90 and top-1 at +-1 <= 0.20 "
          "against a prefill reference; every crossing covered", dict(uncovered=sorted(need - cov), failing=cbad,
                                                                     windows=cross))
    R.data["align_raw"] = dict(d=d, a=a)
    return regime


def regime_check(run, G, R, regime, runs, prefix="G1"):
    """M3's 65,536 selection-regime crossing: alignment window (numerical) plus exact selection-record validity at the
    selection-record positions inside the window (511 distinct, causal pool ids in every tracked layer)."""
    sel_bad, n_sel = [], 0
    for n in G["order"]:
        f = G["fx"][n]
        for ob in f["regime_o"]:
            for ph in runs:
                a = run.req(f"{ph}-{n}")
                if a is None:
                    continue
                sel = a.get("selections")
                for k, o in enumerate(a.get("selpos") or []):
                    if not (ob - RW <= o < ob + RW) or sel is None:
                        continue
                    for L in range(sel.shape[1]):
                        row = sel[k, L].numpy()
                        if row[0] < 0:
                            continue
                        n_sel += 1
                        ids = row[1:]
                        ids = ids[ids >= 0]
                        lim = (f["P"] + o) // 4 + 1
                        if ids.size != 511 or len(set(ids.tolist())) != 511 or ids.max() >= lim:
                            sel_bad.append((ph, n, o, L, int(ids.size), int(ids.max()) if ids.size else None, lim))
    need = [n for n in G["order"] if G["fx"][n]["regime_o"]]
    okal = bool(regime) and all(v["n"] > 0 and v["top1"] >= fc.SHIFT_ALIGNED_MIN and v["shifted"] <=
                                fc.SHIFT_SHIFTED_MAX for v in regime.values())
    R.add(f"{prefix}-regime", "numerical", "PASS" if need and okal and n_sel and not sel_bad else "FAIL",
          "M3's 65,536 selection-regime crossing: +-256 output indices aligned with the prefill reference (top-1 >= 0.90, "
          "+-1 <= 0.20) and every recorded final selection in the window holds 511 distinct causal pool ids",
          dict(fixtures=need, windows=regime, selection_rows=n_sel, selection_violations=sel_bad[:10]))


def content_gate(run, G, R, tested_runs, dd_pair, pp_pair, ref_tag, prefix="G1", extra_ref=None):
    """21g-1 content rule on the [A22-3] matched nulls: units (fixture, rank, DSA layer, pool) over pools wholly in
    [P, P_L) (prefill-written in the references). Gate on each tested run vs ref_tag; original criterion beside."""
    attrib, pop = [], {}

    def at(tag, rank, rec):
        c = run.by_tag.get(tag)
        if c is None or rec.get("tag") != tag or rec.get("P") != c["P"]:
            attrib.append((tag, rank, rec.get("tag"), rec.get("P")))
    C = {}
    for nm, (a_, b_) in [("dd", dd_pair), ("pp", pp_pair)] + [("t_" + t, (t, ref_tag)) for t in tested_runs] + \
            ([("x_" + t, (t, extra_ref)) for t in tested_runs] if extra_ref else []):
        C[nm] = {}
        for n in G["order"]:
            lo, hi = G["fx"][n]["ref_pools"]
            C[nm][n] = fc.compare_pools(run, f"{a_}-{n}", f"{b_}-{n}", lo, hi, at, pop, nm)
    res, ok = {}, True
    for t in tested_runs:
        c = fc.content(C["t_" + t], C["dd"], C["pp"])
        res[t] = c
        ok &= c["cover_ok"] and c["gross"] <= c["limit"]
    popbad = [k for k, v in pop.items() if not v[1] or v[0] / v[1] < fc.POP_MIN or v[2]]
    R.add(f"{prefix}-content", "numerical", "PASS" if ok else "FAIL", "index-K content vs the prefill reference %s: "
          "gross <= max(2 x max(N_dd, N_pp), 0.1 %%) with N_dd = %s vs %s, N_pp = %s vs %s (different build length), "
          "matched by fixture x rank x layer x shared causal pool; pooled over units; coverage >= 99 %% and every "
          "fixture" % (ref_tag, dd_pair[0], dd_pair[1], pp_pair[0], pp_pair[1]),
          {t: {k: v for k, v in c.items() if not k.startswith("_")} for t, c in res.items()}, "21g-1; A22-3")
    R.add(f"{prefix}-content-orig", "reported", "REPORTED", "ORIGINAL attempt-1 criterion beside: gross <= 0.1 %",
          {t: dict(gross=c["gross"], passes=c["orig_pass"]) for t, c in res.items()}, "attempt-1 frozen")
    if extra_ref:
        R.add(f"{prefix}-content-ref2", "reported", "REPORTED", "the same rule against the second reference %s" % extra_ref,
              {t: {k: v for k, v in fc.content(C["x_" + t], C["dd"], C["pp"]).items() if not k.startswith("_")}
               for t in tested_runs})
    units = {t: res[t]["_units"] for t in tested_runs}
    R.add(f"{prefix}-content-table", "reported", "REPORTED", "gross counts per fixture x layer and per fixture, with "
          "both matched nulls (21g-1 reporting)", {t: fc.table(u) for t, u in units.items()})
    fl = {t: fc.flags(u) for t, u in units.items()}
    if len(tested_runs) >= 2:
        fl["arm_asymmetry"] = fc.asym_flags(units[tested_runs[0]], units[tested_runs[1]])
    R.add(f"{prefix}-flags", "reported", "REPORTED", "R3/R5 fixture flags (layer-units AND positions); a flagged fixture "
          "stays in the aggregate, carries no mechanism claim and is dropped in the R2 sensitivity", fl, "R1-R5")
    R.data["flagged"] = sorted({x["fixture"] for v in fl.values() for x in v})
    # crossing neighbourhoods (reported): pools within +-128 positions of each crossing, own matched limit
    xs = {}
    for t in tested_runs:
        for n in G["order"]:
            f = G["fx"][n]
            for j, b in enumerate(f["crossings_abs"]):
                keep = lambda k, b=b: b - XW <= 4 * k[2] < b + XW  # noqa: E731
                us = [u for u in res[t]["_units"] if u[0] == n and keep(u[1])]
                if us:
                    lim, ndd, npp = fc.limit_of(us)
                    xs["%s-%s-x%d" % (t, n, j + 1)] = dict(n=len(us), gross_n=int(sum(u[2] > fc.GROSS for u in us)),
                                                           limit=lim, N_dd=ndd, N_pp=npp)
    R.add(f"{prefix}-content-crossings", "reported", "REPORTED", "content in +-128 positions of each block crossing "
          "(own matched limit; small samples, not gated; M4's second crossing lies beyond P_L: no matched N_pp)", xs)
    rows = {t: C["t_" + t] for t in tested_runs}
    rows_by = {f"{t}-{n}": v for t in tested_runs for n, v in rows[t].items()}
    ps = fc.page_stats(rows_by)
    eg, en = fc.early_gross(rows_by)
    R.add(f"{prefix}-idx-struct", "structural", "PASS" if not attrib and not popbad and ps["over"] == 0 and eg == 0
          and en > 0 else "FAIL", "index-K structure (exact): every dump attributed to its request; every comparison "
          ">= 99 % of its expected units; no judged page > 25 % gross; zero gross pools in DSA layers 3-15",
          dict(attrib=attrib[:5], population={k: v[:2] for k, v in pop.items()}, failing_population=popbad,
               pages=ps, early=(eg, en)), "21g-1 structural")
    return res


def cv_checks(run, G, R, prefix="G1"):
    """[A22-5] conv-state boundary control on the CV snapshots: exact slot / bookkeeping / crossing / persistence, and
    identification of conv rows and recurrent state against the matched decode reference (poison)."""
    snaps = {}
    for n in geo.CV_FIXTURES:
        for m in geo.CV_MODES:
            tag = f"CV-{m.replace(':', '_')}-{n}"
            for F, ranks in run.snaps(tag):
                for rank, p in ranks.items():
                    snaps.setdefault((n, m, rank), []).append((F, run.load(p)))
    slot_bad, book_bad, cross_bad, pers = [], [], [], {}
    conv_rows, ssm_rows = [], []
    for (n, m, rank), lst in sorted(snaps.items()):
        lst.sort(key=lambda x: x[0])
        tag = f"CV-{m.replace(':', '_')}-{n}"
        rq = run.req(tag, rank)
        f = G["fx"][n]
        want = f["cv"]["targets"][m]
        if m in ("poison", "oracle", "reject:2") and [F for F, _ in lst] != want:
            book_bad.append((tag, rank, "snapshot F %s != targets %s" % ([F for F, _ in lst], want)))
        for F, s in lst:
            for b in fc.slot_check(s):
                slot_bad.append((tag, rank, F) + tuple(b))
            na = s.get("num_accepted")
            if rq is None or not isinstance(na, int):
                book_bad.append((tag, rank, F, "no hook record / num_accepted"))
                continue
            M = rq["metrics"].numpy()
            Cc = fc.cols(rq)
            rows = np.where(M[:, Cc["step"]] == s.get("step"))[0]
            accs = sorted(set(M[rows, Cc["accepted"]].tolist()))
            if len(accs) != 1 or int(accs[0]) + 1 != na:
                book_bad.append((tag, rank, F, "num_accepted %s vs hook accepted %s + 1" % (na, accs)))
        cols_ = {fc.col_of(s) for _, s in lst}
        cB = f["cv"]["B"] // geo.BLOCK
        if not {cB - 1, cB} <= cols_:
            cross_bad.append((tag, rank, sorted(c for c in cols_ if c is not None)))
        pers[(n, m, rank)] = fc.conv_persist(lst)
        if m != "poison":
            ref = dict(snaps.get((n, "poison", rank), []))
            for F, s in lst:
                for Fp, sp in ref.items():
                    for r_ in fc.conv_ident(s, sp):
                        r_.update(fixture=n, mode=m, rank=rank, F=F, F_ref=Fp)
                        conv_rows.append(r_)
                if F in ref:
                    for r_ in fc.kda_ident(s, ref[F]):
                        r_.update(fixture=n, mode=m, rank=rank, F=F)
                        ssm_rows.append(r_)
    exp_keys = {(n, m, rk) for n in geo.CV_FIXTURES for m in geo.CV_MODES for rk in (0, 1)}
    missing = sorted(exp_keys - set(snaps))
    R.add(f"{prefix}-CV-slot", "structural", "PASS" if snaps and not slot_bad and not missing else "FAIL",
          "actual conv-state slot: every KDA layer's speculative slots = the running column's block-id run of one "
          "mamba group (conv row = slots[0] = the engine's conv_state_indices)", dict(violations=slot_bad[:10],
                                                                                    missing=missing))
    R.add(f"{prefix}-CV-book", "structural", "PASS" if snaps and not book_bad else "FAIL", "bookkeeping (exact): "
          "num_accepted = hook accepted count at the snapshot step + 1; poison / oracle / reject:2 snapshots exactly at "
          "their targets (shared F with the reference)", book_bad[:10])
    R.add(f"{prefix}-CV-cross", "structural", "PASS" if snaps and not cross_bad else "FAIL", "the block crossing is "
          "exercised: every CV request has snapshots in the old and the new align-mode column", cross_bad)
    pbad = [k for k, v in pers.items() if v["identical"] != v["rows"]]
    nostr = [k for k, v in pers.items() if k[1] != "natural" and v["straddling_pairs"] < 1]
    R.add(f"{prefix}-CV-persist", "structural", "PASS" if pers and not pbad and not nostr else "FAIL",
          "migration (exact): conv rows of shared committed positions are bitwise identical between consecutive "
          "snapshots of a request, including a pair straddling the column change (poison, oracle, reject:2)",
          dict(total=dict(rows=sum(v["rows"] for v in pers.values()),
                          identical=sum(v["identical"] for v in pers.values()),
                          straddling=sum(v["straddling_pairs"] for v in pers.values())),
               mismatching=[(str(k), pers[k]["mismatches"][:3]) for k in pbad[:5]], no_straddle=[str(k) for k in nostr]))
    okc, sc = fc.ident_screen(conv_rows)
    oks, ss = fc.ident_screen(ssm_rows)
    R.add(f"{prefix}-CV-conv", "numerical", "PASS" if okc else "FAIL", "conv rows vs the matched decode reference "
          "(poison, single-token advance) at shared committed positions: nearest-row identification 100 % in layers "
          "0-17, >= 99 % deeper; layers 0-2 rel <= 0.02 with margin >= 10 (21g-2 screen)", sc, "21g-2; A22-5")
    R.add(f"{prefix}-CV-ssm", "numerical", "PASS" if oks else "FAIL", "KDA recurrent state vs the matched decode "
          "reference at the same committed count F (oracle, reject:2, natural vs poison): committed-slot identification, "
          "same screen", ss, "21g-2; A22-5")
    R.data["cv"] = dict(conv=conv_rows, ssm=ssm_rows)


def verdict(R):
    fs, fn = R.failing("structural"), R.failing("numerical")
    v = "PASS" if not fs and not fn else "FAIL"
    reading = ("PASS: the long-context instrument checks hold under these bounds; injected batches may run" if v == "PASS"
               else ("INVALID (structural): %s" % fs if fs else "") + (" ; " if fs and fn else "") +
               ("long-context instrument UNVALIDATED under these bounds (%s). This does not make B1 defective; no "
                "threshold is revised after viewing data" % fn if fn else ""))
    return v, reading


def gate1(run, G):
    R = Rows()
    pr = [r for r in run.CL if r.get("kind") == "probe"]
    R.add("G1-V0", "structural", "PASS" if len(pr) == 4 and all(p.get("equal") for p in pr) else "FAIL",
          "V0: unkeyed probes bitwise equal to B1's record (hook on, WMAX 8448)", [(p["id"], p.get("equal")) for p in pr])
    structural(run, G, R, ["C1", "C2"], ["LW1", "LW2"])
    clean_checks(run, G, R, ["C1", "C2"])
    regime = alignment(run, G, R, [("C1", "LW1"), ("C2", "LW2")])
    regime_check(run, G, R, regime, ["C1", "C2"])
    content_gate(run, G, R, ["C1", "C2"], ("C1", "C2"), ("LW1", "LW2"), "LW1", extra_ref="LW2")
    cv_checks(run, G, R)
    # reported: reference vs reference forced-token lp (prefill null at different build length)
    rr = {}
    for n in G["order"]:
        a, b = run.req(f"LW1-{n}"), run.req(f"LW2-{n}")
        if a is not None and b is not None:
            s = fc.align_summary(a, b, 0, G["fx"][n]["W_s"] + geo.LP_SHIFT - 1024)
            rr[n] = {k: v for k, v in s.items() if not k.startswith("_")}
    R.add("G1-refref", "reported", "REPORTED", "prefill vs prefill (LW1 vs LW2, build lengths P + 7,168 vs P + 8,192) "
          "on shared rows: the reference-side alignment null", rr)
    v, reading = verdict(R)
    return R, v, reading


def main():
    rd = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(HERE)
    G = geo.load(root)
    run = fc.Run(rd)
    try:
        R, v, reading = gate1(run, G)
    except Exception as exc:  # noqa: BLE001
        R, v, reading = Rows(), "FAIL", "gate raised: %r" % exc
        R.add("G1-raised", "structural", "FAIL", "gate code raised", repr(exc)[:500])
    out = dict(gate="GATE-1", verdict=v, reading=reading, rows=R.rows,
               failing=dict(structural=R.failing("structural"), numerical=R.failing("numerical")),
               flagged_fixtures=R.data.get("flagged", []))
    json.dump(out, open(os.path.join(rd, "gate1.json"), "w"), indent=1, default=str)
    lines = ["GATE-1 %s  %s" % (v, reading)] + ["%-8s %-22s %-10s %s  [%s]" % (
        r["status"], r["id"], r["kind"], r["name"][:110], json.dumps(r["detail"], default=str)[:220]) for r in R.rows]
    open(os.path.join(rd, "gate1.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if v == "PASS" else 1)


if __name__ == "__main__":
    main()
