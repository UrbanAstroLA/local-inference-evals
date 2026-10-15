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

C_RUNS, D_RUNS, LC_RUNS, LD_RUNS = af.C_RUNS, af.D_RUNS, af.LC_RUNS, af.LD_RUNS
PAIR_BUILD = {"1": "separate builds (s1 / s4)", "2": "one build (s2), C2 before the gate, D2 after it",
              "3": "one build (s3), D3 then C3"}


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
    R.add("C3-residue", "structural", "PASS" if c3ok else "FAIL", "C3 (after injected traffic, on s3's build): valid and "
          "clean, aligned with the prefill references, within the GATE-1R content rule and the F6 rules",
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
        lp["valid_runs"] = {p: valid.get(p) for p in LC_RUNS + LD_RUNS}
        lp["name"] = ("history-construction and accumulated-decode-state comparison at fixed total context (NOT an "
                      "output-length experiment)")
        lp["construction_note"] = ("[R-3] every LP run restores its prefix from the same primary builds as the C / D "
                                   "runs (restore depth c for the first LP user of a build, c_L = rule(P_L) for the "
                                   "second, from the first LP run's prefill); LC: 1 x c + 2 x c_L, LD: 2 x c + 1 x c_L; "
                                   "both depths are prefill construction and the LC-LC / LD-LD nulls each contain one "
                                   "c vs c_L pair")
    res["lp_control"] = lp
    # ---------------- readings (attempt-1 rule)
    gates_ok = not R.failing() and res["gate1r"]["recomputed"] == "PASS"
    if not gates_ok:
        reading = "NO READING: a gate failed (%s); results are instrument findings only" % R.failing()
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
    res["notes"] = ["flagged fixtures carry no mechanism claim (R1); every reading is repeated in the R2 sensitivity",
                    "C1 of run-fc1 is not pooled or paired with any run of this session (amendment 22d s.10)",
                    "a null result excludes only decode-window D2 exposure on these 8 fixtures below 131,072; not "
                    "full-history accumulation, D6, or longer ranges"]
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
    open(os.path.join(rd, "fcr_result.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
