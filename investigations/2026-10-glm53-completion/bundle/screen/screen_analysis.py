#!/usr/bin/env python3
"""One-pass GPQA screen of the correctness bundle (PLAN.md amendment 14 (4)); DRAFT rules for operator/assessor sign-off.
CPU only. Reuses the frozen arm loaders (analysis_col511.py, sha256 b5b13ce2..., copied unchanged) with N_PASSES = 1 for
the screen pass; baselines are the seed-1234 passes of the 0.9.1 record, the column-511 arm, and 0.7.0 (context only).

usage: screen_analysis.py --pass-dir DIR --reqlog FILE [--serverlog FILE] [--gpu-hours X]
                          [--adjudication JSON] [--label TEXT] [--out JSON]
Exit 0 valid, 3 invalid (no reading is made of an invalid pass).

RULES (PLAN.md amendment 14c, adopted before any screen data; resource-allocation rules, not significance or equivalence
claims) -- E = empty final replies among 198
  validity: one samples file, 198 questions, request log with 198 requests, none re-sent, all HTTP 200, seed 1234,
            max_tokens 327,680, temperature 1.0, top_p 0.95, no parse errors, messages hashes equal the baselines'
            (prompts identical), one server start, no illegal-memory/preemption/NaN lines in the window, no stopping or
            execution errors (finish not in {stop, length}); pass completed under the ceiling. An invalid or
            watchdog-truncated run is not a completed one-pass result (exit 3, no category).
  flags:    REGRESSION FLAG     raw correct <= 158/198, or stated (adjudicated) correct <= 159/198, or >= 2 premature stops
            ACCURACY CONCERN    raw correct 159-163/198 (an otherwise promising result with raw 159-163 is mixed)
  category: CLEAR PROMISING                 E <= 2, no regression flag and no accuracy concern
            MIXED                           E <= 3 with a regression flag or an accuracy concern
            SIMILAR / NO CLEAR SEPARATION   E = 3 without either
            SIMILAR OR WORSE                E >= 4 (any accompanying regression reported separately)
  Stated-answer adjudication must be completed before the final category: every row where the stated adjudicator
  differs from flexible-extract is hand-read and listed in --adjudication (JSON rows {doc, hedged}); otherwise the
  category is reported as PROVISIONAL (adjudication pending) and is not final.
  Simulation (power_bundle_onepass.py) reports P(E <= threshold) only; accuracy and stopping conditions are not modelled,
  so it does not give the probability of the complete clear-promising rule.
"""
import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis_col511 as A  # noqa: E402

G = "<gpqa-runs>"
BASE = {
    "0.9.1 record pass 1": dict(passes=[f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps__pass1",
                                        f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps__pass2",
                                        f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps__pass3"],
                                reqlog=[f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1.requests.jsonl",
                                        f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.requests.jsonl"]),
    "col511 arm pass 1": dict(passes=[f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix__pass{p}" for p in (1, 2, 3)],
                              reqlog=[f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix.s1.requests.jsonl",
                                      f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix.s2.requests.jsonl"]),
}
V070_P1 = f"{G}/glm53-flash-exl3-k3.25@tpurtell__pass1"
RAW_FLAG, STATED_FLAG, RAW_CONCERN, PREMATURE_MAX = 158, 159, 163, 1
E_CLEAR, E_NOSEP, E_SIMILAR = 2, 3, 4


def categorize(E, raw, stated, premature):
    """Amendment 14c categories -> (regression_flag, accuracy_concern, category)."""
    flag = raw <= RAW_FLAG or stated <= STATED_FLAG or premature > PREMATURE_MAX
    concern = (not flag) and RAW_FLAG < raw <= RAW_CONCERN
    if E >= E_SIMILAR:
        return flag, concern, "SIMILAR OR WORSE"
    if flag or concern:
        return flag, concern, "MIXED"
    if E <= E_CLEAR:
        return flag, concern, "CLEAR PROMISING"
    return flag, concern, "SIMILAR / NO CLEAR SEPARATION"      # E == 3 without flag or concern


def cause(r):
    if not r["empty"]:
        return None
    if not r["logged"]:
        return "not_logged"
    if r["finish_reason"] == "length" and r["completion_tokens"] == A.CAP:
        return "cap_exhaustion"
    if r["finish_reason"] == "stop":
        return "premature_stop"
    return "other_finish"


def pass1(rows):
    return {d: r for (d, p), r in rows.items() if p == 1}


def load_v070_pass1():
    """0.7.0 pass 1 (seed 1234) has no request log: samples only (context, not a comparator)."""
    saved = A.N_PASSES
    A.N_PASSES = 1
    try:
        rows, checks, _ = A.load_arm("0.7.0 pass 1 (context)", [V070_P1], [], [], None)
    finally:
        A.N_PASSES = saved
    return pass1(rows), checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-dir", required=True)
    ap.add_argument("--reqlog", required=True)
    ap.add_argument("--serverlog", default=None)
    ap.add_argument("--gpu-hours", type=float, default=None)
    ap.add_argument("--adjudication", default=None,
                    help="JSON {rows: [{doc, hedged}]}: hand reads of every row where stated differs from flexible-extract")
    ap.add_argument("--label", default="bundle screen pass (seed 1234)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    A.N_PASSES = 1
    adj_rows = json.load(open(a.adjudication))["rows"] if a.adjudication else []
    hedged = [int(r["doc"]) for r in adj_rows if r.get("hedged")]
    S_rows, checks, inputs = A.load_arm(a.label, [a.pass_dir], [a.reqlog], [f"1:{d}" for d in hedged], None)
    A.N_PASSES = 3
    S = pass1(S_rows)
    bases = {}
    for name, b in BASE.items():
        rows, bc, _ = A.load_arm(name, b["passes"], b["reqlog"], [], None)
        bases[name] = pass1(rows)
        checks += [c for c in bc if "pass 1" in c[0]]
    v070, c7 = load_v070_pass1()
    checks += c7
    for name, B in bases.items():
        mism = [d for d in S if d in B and S[d]["messages_sha256"] != B[d]["messages_sha256"]]
        checks.append((f"prompts identical to {name} (messages_sha256, 198 questions)", not mism and len(S) == 198, f"{len(mism)} differ"))
    odd = [d for d, r in S.items() if r["logged"] and r["finish_reason"] not in ("stop", "length")]
    checks.append(("no execution/stopping errors (finish reason stop or length on every request)", not odd, f"{odd[:5]}"))
    srv = None
    if a.serverlog:
        t0 = min(r["t_start"] for r in S.values() if r["t_start"])
        t1 = max(r["t_start"] + r["secs"] for r in S.values() if r["t_start"])
        srv = A.parse_serverlogs([a.serverlog], [(t0, t1)], 0)[0]
        iw = srv.get("in_window", {})
        checks.append(("server: one start, no illegal memory access / preemption / NaN lines in the request window",
                       srv.get("server_starts") == 1 and not any(iw.values()), json.dumps(iw)))
    valid = all(ok for _, ok, _ in checks)
    res = {"label": a.label, "valid": valid, "checks": [list(c) for c in checks],
           "script_sha256": A.file_sha(os.path.abspath(__file__)), "analysis_col511_sha256": A.file_sha(A.__file__)}
    if valid:
        E = sum(r["empty"] for r in S.values())
        causes = {}
        for r in S.values():
            c = cause(r)
            if c:
                causes[c] = causes.get(c, 0) + 1
        raw = sum(r["correct_flexible"] for r in S.values())
        stated = sum(r["correct_stated"] for r in S.values())
        prem = causes.get("premature_stop", 0)
        flag, concern, outcome = categorize(E, raw, stated, prem)
        read = {int(r["doc"]) for r in adj_rows}
        missing = [d for d in sorted(d for d, r in S.items() if r["stated_differs_from_flexible"]) if d not in read]
        adjudication_complete = not missing
        final = outcome if adjudication_complete else f"PROVISIONAL ({outcome}; stated adjudication pending for docs {missing})"
        cmp = {}
        for name, B in list(bases.items()) + [("0.7.0 pass 1 (context)", v070)]:
            be = {d for d, r in B.items() if r["empty"]}
            se = {d for d, r in S.items() if r["empty"]}
            br = sum(r["correct_flexible"] for r in B.values())
            bs = sum(r["correct_stated"] for r in B.values())
            disc_raw = (sum(1 for d in S if S[d]["correct_flexible"] and not B[d]["correct_flexible"]),
                        sum(1 for d in S if B[d]["correct_flexible"] and not S[d]["correct_flexible"]))
            cmp[name] = {"baseline_empties": len(be), "baseline_causes": {c: sum(1 for d in be if cause(B[d]) == c)
                                                                         for c in ("cap_exhaustion", "premature_stop", "not_logged", "other_finish")},
                         "rescued": sorted(be - se), "newly_empty": sorted(se - be), "empty_in_both": sorted(se & be),
                         "baseline_raw": br, "baseline_stated": bs, "raw_diff": raw - br, "stated_diff": stated - bs,
                         "raw_discordant_screen_only_vs_baseline_only": disc_raw,
                         "mcnemar_two_sided_raw_reference": A.signflip([s - b for s, b in zip(
                             [int(S[d]["correct_flexible"]) for d in sorted(S)], [int(B[d]["correct_flexible"]) for d in sorted(S)])])}
        ans = [r["completion_tokens"] for r in S.values() if not r["empty"] and r["completion_tokens"] is not None]
        A.N_PASSES = 1
        try:
            tim, _ = A.timing({(d, 1): r for d, r in S.items()}, "screen", None, a.gpu_hours, 2) if all(r["logged"] for r in S.values()) else ({}, None)
        finally:
            A.N_PASSES = 3
        res.update({
            "primary": {"empties": E, "causes": causes, "category": final, "category_if_adjudication_complete": outcome,
                        "adjudication_complete": adjudication_complete, "adjudicated_docs": sorted(read),
                        "flags": {"regression_flag": flag, "accuracy_concern": concern,
                                  "regression_detail": {"raw_le_158": raw <= RAW_FLAG, "stated_le_159": stated <= STATED_FLAG,
                                                        "premature_stops": prem}},
                        "versus": cmp},
            "secondary": {"raw_correct_all_198": raw, "stated_correct_all_198": stated,
                          "stated_differs_from_raw_rows": [d for d, r in S.items() if r["stated_differs_from_flexible"]],
                          "answered_completion_tokens": A.quantiles(ans) if ans else None,
                          "answered_above_32k_64k_128k_200k": [sum(t > x for t in ans) for x in A.LEN_THRESHOLDS],
                          "timing": tim, "server": srv},
        })
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({k: res[k] for k in res if k not in ("checks",)}, indent=1, default=str)[:6000])
    bad = [c for c in res["checks"] if not c[1]]
    if bad:
        print("INVALID:", json.dumps(bad, default=str)[:1500])
    return 0 if valid else 3


if __name__ == "__main__":
    sys.exit(main())
