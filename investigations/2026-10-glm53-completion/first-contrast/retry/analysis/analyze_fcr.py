#!/usr/bin/env python3
"""Frozen analysis of the first-contrast RETRY session (RETRY-PROPOSAL.md s.4-6; PLAN.md amendments 22 and 22d).
CPU only. Frozen before data; nothing here is revised after viewing the contrast.

usage: analyze_fcr.py RUN_DIR [FC_ROOT]   -> RUN_DIR/fcr_result.json, fcr_result.txt

The primary statistic, the screens, the readings, the direction statements, the sensitivities, the exposure /
selection rule and the LP comparison are attempt 1's frozen definitions, called unchanged from
../../analysis/analyze_fc.py (af) [A22-1, A22-2, A22-6, A22-8]. What changes (R-n):
  R-7  gates: GATE-1R as recorded in-session AND recomputed on the full data (gate_fcr.py); per-run validity without
       attempt 1's exact cache equality (a miss is safe) and with the F2 safety rows per run instead
  R-2  construction accounting: hit / partial / miss of every request; R-cache-balance (gated): the number of non-hit
       fixture-runs is equal in C and D (and in LC and LD); any non-hit fixture is reported and the primary contrast is
       repeated on the construction-matched fixtures (full hits in all six runs); pairs are labelled by build sharing,
       verified by block identity (C2 / D2 and D3 / C3 share a build; C1 / D1 do not) and are never described as
       sharing an identical prefix unless the identity check holds
  R-5  the sentinel PS-a / PS-b / PS-c on the shared block [c - 4352, c) (sentinel.py)
  R-6  C3 residue: alignment and the pooled content rule as GATE-1R, plus the F6 page and early-layer rules
Reading of the numerical gates when they pass: "compatible with measured long-context numerical variation".
PLAN.md amendment 22f (before any retry data):
  R-10 the sentinel's boundary state: SSM and convolution components separately, each against its own PS-a / PS-b
       baseline (sentinel.py)
  R-12 F6 pages whose null makes the page limit non-discriminating are reported as "not independently verified by the
       numerical check" (GATE-1R's C1 / C2 and C3's residue)
  R-13 the sentinel's limits are stated in every output (sentinel.LIMITS)
  R-14 LP restore depth: a DESCRIPTIVE same-depth comparison, LC1 vs LD1 (both restore c) and LC2 vs LD2 (both c_L),
       with LC3 (c_L) vs LD3 (c) reported separately as depth-mismatched. The three-repeat rule is not applied to the
       two matched pairs (no beyond-null or repeatability verdict is drawn from them). Dependence rule, fixed here: the
       LP conclusion (the short-context injection contrast LD - LC and the interaction (D - C) - (LD - LC), each by its
       frozen threshold) DEPENDS ON THE MISMATCHED PAIR when recomputing it with the matched pairs' mean difference
       in place of the three-run difference changes its beyond-null status, or its sign while beyond the null; it is
       then reported as "unresolved construction sensitivity".
  R-15 cache wording: eviction is avoided in the replayed scenarios, with measured survival margins; the actual-
       session gates (R-cache-safe, R-cache-model, R-cache-balance) are authoritative.
PLAN.md amendment 22h:
  R-17 a batch-guard stop (client rc 17: the session stopped BEFORE a phase whose projected completion did not fit)
       reads "incomplete contrast (batch guard)" when the gates pass and an arm lacks 3 runs; phases the guard did not
       start are reported NOT RUN (C3's residue row included), never as failures; LP batches the guard skipped are
       listed. Every other rule is unchanged.
"""
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RETRY = os.path.dirname(HERE)
FCROOT = os.path.dirname(RETRY)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(FCROOT, "analysis"))
import analyze_fc as af  # noqa: E402
import cache_gate as cg  # noqa: E402
import f6_criteria as f6  # noqa: E402
import fc_common as fc  # noqa: E402
import fcr_geometry as geo  # noqa: E402
import gate_fc as gt  # noqa: E402
import gate_fcr as g1r  # noqa: E402
import sentinel as ps  # noqa: E402
sys.path.insert(0, os.path.join(RETRY, "runner"))
import batch_guard_fcr as bg  # noqa: E402

C_RUNS, D_RUNS, LC_RUNS, LD_RUNS = af.C_RUNS, af.D_RUNS, af.LC_RUNS, af.LD_RUNS
PAIR_BUILD = {"1": "separate builds (s1 / s4)", "2": "one build (s2), C2 before the gate, D2 after it",
              "3": "one build (s3), D3 then C3"}


LP_DEPTH = {"LC1": "c", "LC2": "c_L", "LC3": "c_L", "LD1": "c", "LD2": "c_L", "LD3": "c"}
LP_UNRESOLVED = "unresolved construction sensitivity"
CACHE_WORDING = ("eviction avoided in the replayed scenarios, with measured survival margins; the actual-session gates "
                 "are authoritative")


def _sgn(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def lp_same_depth(Yshort, valid, inj_long, inj_short, interaction):
    """[R-14, 22f s.6] descriptive same-depth LP comparison and the dependence of the LP conclusion on the
    depth-mismatched pair. Yshort: {LP run: aggregate Y}; valid: {run: bool}; inj_*: attempt-1 contrast dicts."""
    pairs = {}
    for a, b in (("LC1", "LD1"), ("LC2", "LD2"), ("LC3", "LD3")):
        ya, yb = Yshort.get(a), Yshort.get(b)
        ok = valid.get(a) and valid.get(b) and ya is not None and yb is not None and np.isfinite(ya) and np.isfinite(yb)
        pairs[f"{a}/{b}"] = dict(depth=dict(LC=LP_DEPTH[a], LD=LP_DEPTH[b]), matched=LP_DEPTH[a] == LP_DEPTH[b],
                                 Y_LC=ya, Y_LD=yb, d_LD_minus_LC=float(yb - ya) if ok else None)
    md = [pairs[k]["d_LD_minus_LC"] for k in ("LC1/LD1", "LC2/LD2")]
    have_m = all(x is not None for x in md)
    out = dict(
        name="descriptive same-depth comparison (22f s.6): no beyond-null or repeatability verdict is drawn from the "
             "two matched pairs (the three-repeat rule is not applied to them)",
        matched_pairs={k: pairs[k] for k in ("LC1/LD1", "LC2/LD2")},
        depth_mismatched_pair=dict(pairs["LC3/LD3"], note="depth-mismatched: LC3 restores c_L, LD3 restores c; "
                                                          "reported separately"),
        matched_mean_d=float(np.mean(md)) if have_m else None,
        matched_same_sign=(all(x > 0 for x in md) or all(x < 0 for x in md)) if have_m else None,
        three_repeat_rule_applied_to_matched_pairs=False)
    checks = {}
    if have_m and inj_short.get("complete"):
        thr = inj_short["threshold"]
        full_b, m_b = bool(inj_short["beyond_null"]), abs(out["matched_mean_d"]) > thr
        dep = full_b != m_b or (full_b and _sgn(inj_short["delta"]) != _sgn(out["matched_mean_d"]))
        checks["injection_short"] = dict(full_delta=inj_short["delta"], matched_delta=out["matched_mean_d"],
                                         threshold=thr, full_beyond=full_b, matched_beyond=bool(m_b), depends=bool(dep))
        if inj_long.get("complete") and interaction:
            thr_i = 2 * 2 * max(interaction["N"], af.N_FLOOR)
            im = inj_long["delta"] - out["matched_mean_d"]
            full_b, m_b = bool(interaction["beyond_null"]), abs(im) > thr_i
            dep = full_b != m_b or (full_b and _sgn(interaction["delta"]) != _sgn(im))
            checks["interaction"] = dict(full_delta=interaction["delta"], matched_delta=im, threshold=thr_i,
                                         full_beyond=full_b, matched_beyond=bool(m_b), depends=bool(dep))
    out["dependence_checks"] = checks
    if not checks:
        out["conclusion_status"] = "not evaluated (a matched pair or the LP contrast is incomplete)"
    elif any(c["depends"] for c in checks.values()):
        out["conclusion_status"] = LP_UNRESOLVED
    else:
        out["conclusion_status"] = "the LP conclusion does not depend on the depth-mismatched pair"
    return out


def run_valid_r(run, G, ph, clean, cache_rows):
    R = gt.Rows()
    gt.structural(run, G, R, [ph], [], cv_tag=None, prefix=ph, warms=False)
    g1r.demote(R, f"{ph}-cache", f"{ph}-cache-orig", "exact equality; a miss is safe (F2)")
    if clean:
        gt.clean_checks(run, G, R, [ph], prefix=ph)
    rows = [x for x in cache_rows if x["phase"] == ph]
    uns = [x["tag"] for x in rows if not x["safe"]]
    R.add(f"{ph}-cache-safe", "structural", "PASS" if rows and not uns else "FAIL", "F2 S1-S4 for every request of "
          "the run", dict(unsafe=uns, classes={x["fixture"]: x["cls"] for x in rows}), "R-2")
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


def main():
    rd = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else FCROOT
    G = geo.load(root)
    run = fc.Run(rd)
    start = [r for r in run.CL if r.get("kind") == "start"]
    variant = (start[0].get("variant") if start else None) or "B"
    R = gt.Rows()
    res = dict(limits=af.LIMITS, labels="R-n = RETRY-PROPOSAL.md s.0; A22-n = amendment 22 item n", variant=variant)
    # ---------------- 1. gates
    try:
        g_sess = json.load(open(os.path.join(rd, "gate1r", "gate1r.json")))
    except Exception:  # noqa: BLE001
        g_sess = None
    Rg, vg, reading_g, classes, cache_rows = g1r.gate1r(run, G, rd, variant)
    res["gate1r"] = dict(in_session=(g_sess or {}).get("verdict"), recomputed=vg, reading=reading_g,
                         stop_classes=classes,
                         failing=dict(structural=Rg.failing("structural"), numerical=Rg.failing("numerical")))
    R.add("GATE-1R", "structural", "PASS" if (g_sess or {}).get("verdict") == "PASS" and vg == "PASS" else "FAIL",
          "GATE-1R passed in-session (before any injected batch) and recomputes to PASS on the full data",
          res["gate1r"], "R-7")
    lp_on = variant == "B"
    valid, vrows = {}, {}
    for ph in C_RUNS + (LC_RUNS if lp_on else []):
        valid[ph], vrows[ph] = run_valid_r(run, G, ph, True, cache_rows)
    for ph in D_RUNS + (LD_RUNS if lp_on else []):
        valid[ph], vrows[ph] = run_valid_r(run, G, ph, False, cache_rows)
    res["valid_runs"] = valid
    res["run_rows_failing"] = {ph: [r["id"] for r in rows if r["status"] == "FAIL"] for ph, rows in vrows.items()}
    af.injection_checks(run, G, R, [p for p in D_RUNS + (LD_RUNS if lp_on else []) if f"{p}-L1" in run.by_tag],
                        C_RUNS + (LC_RUNS if lp_on else []))
    lim = None
    for r_ in Rg.rows:
        if r_["id"] == "G1-content" and isinstance(r_["detail"], dict) and "C1" in r_["detail"]:
            lim = r_["detail"]["C1"]["limit"]
    if lim is not None and lim == lim:
        af.ledger_vs_content(run, G, R, [p for p in D_RUNS if f"{p}-L1" in run.by_tag], lim)
    # F2 whole session
    uns = [(x["tag"], x["cls"]) for x in cache_rows if not x["safe"]]
    R.add("R-cache-safe", "structural", "PASS" if cache_rows and not uns else "FAIL", "F2 S1-S4 for every request of the "
          "session (builds, touch passes, references, runs, CV, sentinels)", dict(n=len(cache_rows), unsafe=uns[:12]),
          "R-2")
    arms = {"C": C_RUNS, "D": D_RUNS}
    if lp_on:
        arms.update(LC=LC_RUNS, LD=LD_RUNS)
    bal = cg.balance(cache_rows, arms)
    okb = bal["per_arm"]["C"]["non_hits"] == bal["per_arm"]["D"]["non_hits"] and (
        not lp_on or bal["per_arm"]["LC"]["non_hits"] == bal["per_arm"]["LD"]["non_hits"])
    R.add("R-cache-balance", "structural", "PASS" if okb else "FAIL", "cache behaviour does not differ systematically "
          "by treatment: equal numbers of non-hit fixture-runs in C and D (and in LC and LD); non-hit fixtures are "
          "reported and analysed in the construction-matched sensitivity", bal, "R-2")
    ident = cg.identity(cache_rows, [["C2", "PS-a", "PS-b", "D2", "PS-c"], ["D3", "C3"]] +
                        ([["LC1", "LC2", "C1", "CV"], ["LD1", "LD2", "D1"], ["LD3", "LC3"]] if lp_on else []))
    res["build_identity"] = ident
    res["pair_construction"] = {}
    for r_ in ("1", "2", "3"):
        k = {"2": "C2+PS-a+PS-b+D2+PS-c", "3": "D3+C3"}.get(r_)
        same = [v["identical"] for kk, v in ident.items() if k and kk.startswith(k + ":")]
        res["pair_construction"]["C%s/D%s" % (r_, r_)] = dict(
            design=PAIR_BUILD[r_], identical_restored_blocks_verified=bool(same) and all(same) if k else False,
            wording=("restored the identical physical prefix blocks (verified)" if k and same and all(same) else
                     "separate prefix builds of the same tokens; NOT an identical prefix"))
    ps.check(run, G, R, cache_rows, phases=("PS-a", "PS-b", "PS-c"), build_users=("C2", "D2"), prefix="R-PS")
    # C3 residue
    R3 = gt.Rows()
    gt.alignment(run, G, R3, [("C3", "LW1"), ("C3", "LW2")], prefix="C3")
    gt.content_gate(run, G, R3, ["C3"], ("C1", "C2"), ("LW1", "LW2"), "LW1", prefix="C3")
    g1r.demote(R3, "C3-idx-struct", "C3-idx-struct-orig", "superseded by F6")
    PP = f6.rows_for(run, G, "LW1", "LW2")
    DD = f6.rows_for(run, G, "C1", "C2")
    e3 = f6.evaluate(f6.rows_for(run, G, "C3", "LW1"), DD, PP)
    R3.add("C3-f6", "numerical", "PASS" if e3["ok"] else "FAIL", "F6 page and early-layer rules on C3",
           dict(pages=e3["pages"], early=e3["early"]), "R-6")
    c3ok = not R3.failing() and valid.get("C3")
    # [R-17] batch guard: phases the client's guard did not start are NOT RUN, not failures
    stops = [r_ for r_ in run.CL if r_.get("kind") == "stop"]
    gstop = next((r_ for r_ in stops if r_.get("rc") == bg.GUARD_RC), None)
    cov = next((r_ for r_ in run.CL if r_.get("kind") == "coverage"), {}) or {}
    not_run = set(cov.get("not_run") or [])
    gskip = [r_["phase"] for r_ in run.CL if r_.get("kind") == "skip"]
    first_nr = next((q["tag"] for q in geo.phases(G, variant) if q["tag"] in not_run and q["tag"] not in gskip), None)
    res["batch_guard"] = dict(stop=gstop, stopped_before=first_nr if gstop else None,
        not_run=sorted(not_run), skipped=gskip,
        decisions=[r_ for r_ in run.CL if r_.get("kind") == "guard"][:80],
        rates=[r_ for r_ in run.CL if r_.get("kind") == "rate"])
    if gstop and "C3" in not_run and "C3" not in run.by_tag:
        R.add("C3-residue", "reported", "NOT RUN", "C3 not started: the batch guard stopped the session before it "
              "(rc 17)", dict(stop=gstop.get("why")), "R-17")
    else:
        R.add("C3-residue", "structural", "PASS" if c3ok else "FAIL", "C3 (after injected traffic, on s3's build): valid "
              "and clean, aligned with the prefill references, within the GATE-1R content rule and the F6 rules",
              dict(failing=R3.failing(), valid=valid.get("C3")), "R-6")
    res["c3_rows"] = R3.rows
    # ---------------- 2-4. statistics (attempt-1 functions, unchanged)
    vC = [p for p in C_RUNS if valid.get(p)]
    vD = [p for p in D_RUNS if valid.get(p)]
    order = G["order"]
    flagged = list((g_sess or {}).get("flagged_fixtures", []) or Rg.data.get("flagged", []))
    unflagged = [n for n in order if n not in flagged]
    units = list(geo.g1.doc_units(G).values())
    near = [n for n in order if G["fx"][n]["near_close"]]
    active = [n for n in order if not G["fx"][n]["near_close"]]
    y = {ph: af.run_stats(run, G, ph) for ph in C_RUNS + D_RUNS}
    complete = len(vC) == 3 and len(vD) == 3
    pr = af.contrast([af.agg(y[p], order) for p in C_RUNS], [af.agg(y[p], order) for p in D_RUNS]) if complete else \
        dict(complete=False)
    res["primary"] = dict(statistic="Y = mean over 8 fixtures of ln mean_{o in Q_f} P_pre(</think>)", **pr,
                          paired_construction=res["pair_construction"], per_fixture_y={p: y[p] for p in C_RUNS + D_RUNS})
    fxs, direc = af.fixture_screens([y[p] for p in C_RUNS], [y[p] for p in D_RUNS], order) if complete else ({}, {})
    res["fixture_screens"], res["direction_screen"] = fxs, direc
    sens = {}
    if complete:
        def cs(ycol="lp_pre_0", pos="Q", fx=order, w=None):
            yc = [af.run_stats(run, G, p, "long", ycol, pos) for p in C_RUNS]
            yd = [af.run_stats(run, G, p, "long", ycol, pos) for p in D_RUNS]
            return af.contrast([af.agg(v, fx, w) for v in yc], [af.agg(v, fx, w) for v in yd])
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
            sens["R2_leave_flagged_out"] = dict(flagged=flagged, contrast=cs(fx=unflagged),
                                                direction=af.fixture_screens([y[p] for p in C_RUNS],
                                                                             [y[p] for p in D_RUNS], unflagged)[1])
        else:
            sens["R2_leave_flagged_out"] = dict(flagged=[], note="no fixture flagged: identical to the primary")
        mm = bal["mismatch_fixtures"]
        if mm:
            keep = [n for n in order if n not in mm]
            sens["R-2_construction_matched"] = dict(dropped=mm, contrast=cs(fx=keep) if keep else None)
        else:
            sens["R-2_construction_matched"] = dict(dropped=[], note="every fixture restored as designed in all six "
                                                                    "runs: identical to the primary")
    res["sensitivities"] = sens
    ex, ex_run = af.exposure_table(run, G, C_RUNS + D_RUNS + (LC_RUNS + LD_RUNS if lp_on else []))
    res["exposure_by_run_fixture_layer"] = ex
    res["exposure_by_run"] = ex_run
    expo_ok = complete and all(ex_run.get(p, {}).get("n_sel_win_corrupt", 0) > 0 for p in D_RUNS) and \
        all(ex_run.get(p, {}).get("n_sel_win_corrupt", 1) == 0 for p in C_RUNS)
    S = {p: af.selection_sets(run, G, p) for p in C_RUNS + D_RUNS}
    within = [af.jaccard(S[a], S[b])[0] for grp in (C_RUNS, D_RUNS) for a, b in itertools.combinations(grp, 2)]
    across = [af.jaccard(S[a], S[b])[0] for a in C_RUNS for b in D_RUNS]
    div_ok = complete and all(np.isfinite(within + across)) and min(across) > max(within) and \
        float(np.mean(across)) > 2 * max(within)
    res["selection"] = dict(exposure_ok=bool(expo_ok), within_arm_jaccard=within, across_arm_jaccard=across,
                            divergence_ok=bool(div_ok), keys_change=bool(expo_ok and div_ok))
    lp = {}
    if lp_on and all(f"{p}-L1" in run.by_tag for p in LC_RUNS + LD_RUNS):
        Ylong = {p: af.agg(af.run_stats(run, G, p, "long", positions="common"), order) for p in C_RUNS + D_RUNS}
        Yshort = {p: af.agg(af.run_stats(run, G, p, "short", positions="common"), order) for p in LC_RUNS + LD_RUNS}
        lp["construction_clean"] = af.contrast([Yshort[p] for p in LC_RUNS], [Ylong[p] for p in C_RUNS], paired=False)
        lp["construction_injected"] = af.contrast([Yshort[p] for p in LD_RUNS], [Ylong[p] for p in D_RUNS],
                                                  paired=False)
        lp["injection_long"] = af.contrast([Ylong[p] for p in C_RUNS], [Ylong[p] for p in D_RUNS])
        lp["injection_short"] = af.contrast([Yshort[p] for p in LC_RUNS], [Yshort[p] for p in LD_RUNS])
        if lp["injection_long"].get("complete") and lp["injection_short"].get("complete"):
            inter = lp["injection_long"]["delta"] - lp["injection_short"]["delta"]
            Nint = max(lp["injection_long"]["N"], lp["injection_short"]["N"])
            lp["interaction"] = dict(delta=inter, N=Nint, beyond_null=abs(inter) > 2 * 2 * max(Nint, af.N_FLOOR),
                                     rule="|(D - C) - (LD - LC)| > 2 x (2 x max(N, 0.05))")
        lp["same_depth"] = lp_same_depth(Yshort, valid, lp["injection_long"], lp["injection_short"],
                                         lp.get("interaction"))
        lp["conclusion_status"] = lp["same_depth"]["conclusion_status"]
        lp["valid_runs"] = {p: valid.get(p) for p in LC_RUNS + LD_RUNS}
        lp["name"] = ("history-construction and accumulated-decode-state comparison at fixed total context (NOT an "
                      "output-length experiment)")
        lp["construction_note"] = ("[R-3] every LP run restores its prefix from the same primary builds as the C / D "
                                   "runs (restore depth c for the first LP user of a build, c_L = rule(P_L) for the "
                                   "second, from the first LP run's prefill); LC: 1 x c + 2 x c_L, LD: 2 x c + 1 x c_L; "
                                   "both depths are prefill construction and the LC-LC / LD-LD nulls each contain one "
                                   "c vs c_L pair; [R-14] same-depth pairs LC1 / LD1 (c) and LC2 / LD2 (c_L) are "
                                   "compared descriptively in same_depth, LC3 (c_L) / LD3 (c) is depth-mismatched")
    res["lp_control"] = lp
    # ---------------- readings (attempt-1 rule)
    gates_ok = not R.failing() and res["gate1r"]["recomputed"] == "PASS"
    if not gates_ok:
        reading = "NO READING: a gate failed (%s); results are instrument findings only" % R.failing()
    elif not complete and gstop:
        reading = "%s: the session stopped before %s (projected completion did not fit the remaining budget / wall " \
            "cap); fewer than 3 valid runs in an arm (C %s, D %s); the rule is never applied to 2 runs" % (
                bg.READING, res["batch_guard"]["stopped_before"], vC, vD)
    elif not complete:
        reading = "INCOMPLETE: fewer than 3 valid runs in an arm (C %s, D %s); the rule is never applied to 2 runs" % (
            vC, vD)
    elif not pr["repeatable"]:
        reading = "NO READING: within-arm variation N = %.3f nats > 0.25 (variation reported)" % pr["N"]
    elif not res["selection"]["keys_change"]:
        reading = af.READING_3 + " (insufficient selected-key exposure: corrupted pools selected in every D run %s, " \
            "selection divergence beyond the within-arm null %s)" % (expo_ok, div_ok)
    elif pr["screen_met"]:
        reading = af.READING_1
    else:
        reading = af.READING_2
    direction = (af.DIR_UP if pr["delta"] > 0 else af.DIR_DOWN) if pr.get("complete") else None
    res["reading"] = reading
    res["direction_statement"] = direction
    res["flagged_fixtures"] = flagged
    res["numerical_wording"] = f6.WORDING
    ndg = next((r_["detail"] for r_ in Rg.rows if r_["id"] == "R-idx-pages-nd"), {}) or {}
    nd3 = e3["pages"]["non_discriminating"]
    res["f6_not_independently_verified"] = dict(
        wording=f6.ND_WORDING, bound=f6.ND_BOUND,
        gate1r={t: dict(n=x["n"], of_judged=x["of_judged"], n_limit_ge_1=x["n_limit_ge_1"], pages=x["pages"])
                for t, x in ndg.items()},
        C3=dict(n=nd3["n"], of_judged=e3["pages"]["pages_judged"], n_limit_ge_1=nd3["n_undetectable"],
                pages=nd3["pages"]))
    res["sentinel_limits"] = ps.LIMITS
    res["cache_wording"] = CACHE_WORDING
    res["notes"] = ["flagged fixtures carry no mechanism claim (R1); every reading is repeated in the R2 sensitivity",
                    "C1 of run-fc1 is not pooled or paired with any run of this session (amendment 22d s.10)",
                    "a null result excludes only decode-window D2 exposure on these 8 fixtures below 131,072; not "
                    "full-history accumulation, D6, or longer ranges",
                    "cache: " + CACHE_WORDING + " (R-15)",
                    "sentinel limits: " + ps.LIMITS + " (R-13)"]
    res["gate_rows"] = R.rows
    json.dump(res, open(os.path.join(rd, "fcr_result.json"), "w"), indent=1, default=str)
    lines = ["READING: %s" % reading, "DIRECTION: %s" % direction, "LIMITS: %s" % af.LIMITS,
             "GATE-1R in-session %s, recomputed %s" % (res["gate1r"]["in_session"], vg)]
    lines += ["%-8s %-16s %s" % (r_["status"], r_["id"], json.dumps(r_["detail"], default=str)[:200]) for r_ in R.rows]
    lines.append("valid runs: %s" % valid)
    lines.append("pairs: %s" % json.dumps(res["pair_construction"]))
    if pr.get("complete"):
        lines.append("primary: Y_C %s Y_D %s Delta %.4f paired %s N %.4f threshold %.4f beyond %s separation %s" % (
            [round(v, 4) for v in pr["Y_C"]], [round(v, 4) for v in pr["Y_D"]], pr["delta"],
            [round(v, 4) for v in pr["paired_d"]], pr["N"], pr["threshold"], pr["beyond_null"], pr["separation"]))
    lines.append("selection: %s" % json.dumps(res["selection"]))
    if lp.get("same_depth"):
        sd = lp["same_depth"]
        lines.append("LP same-depth (descriptive; three-repeat rule not applied): LC1/LD1 d %s, LC2/LD2 d %s, matched "
                     "mean %s; depth-mismatched LC3/LD3 d %s; LP conclusion: %s" % (
                         sd["matched_pairs"]["LC1/LD1"]["d_LD_minus_LC"], sd["matched_pairs"]["LC2/LD2"]["d_LD_minus_LC"],
                         sd["matched_mean_d"], sd["depth_mismatched_pair"]["d_LD_minus_LC"], sd["conclusion_status"]))
    fn = res["f6_not_independently_verified"]
    lines.append("F6 pages %s (non-discriminating null, limit >= %.2f): %s; C3 %d of %d" % (
        f6.ND_WORDING, f6.ND_BOUND, "; ".join("GATE-1R %s %d of %d" % (t, x["n"], x["of_judged"])
                                             for t, x in fn["gate1r"].items()), fn["C3"]["n"], fn["C3"]["of_judged"]))
    lines.append("SENTINEL LIMITS: " + ps.LIMITS)
    lines.append("CACHE: " + CACHE_WORDING)
    bgr = res["batch_guard"]
    lines.append("BATCH GUARD: %s; LP skipped %s; not run %s" % (
        ("%s, stopped before %s" % (bg.READING, bgr["stopped_before"])) if bgr["stop"] else "no guard stop",
        bgr["skipped"], bgr["not_run"]))
    open(os.path.join(rd, "fcr_result.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
