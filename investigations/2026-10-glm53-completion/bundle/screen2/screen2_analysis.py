#!/usr/bin/env python3
"""Second bundle pass (seed 1235) of the UNCHANGED frozen bundle (PLAN.md amendment 15). DRAFT FROZEN ANALYSIS, written
before any seed-1235 bundle data. CPU only. Reuses the frozen arm loaders (analysis_col511.py, sha256 b5b13ce2..., copied
unchanged): the new pass is loaded as protocol pass 2 (request seed 1235).

usage: screen2_analysis.py --pass-dir DIR --reqlog FILE [--serverlog FILE] [--gpu-hours X] [--adjudication JSON]
                           [--label TEXT] [--out JSON]
Exit 0 valid, 3 invalid (no category is given for an invalid or watchdog-truncated pass).

PART A -- THE NEW PASS, REPORTED SEPARATELY (decision rules of amendment 15)
  Primary comparator: the stock 0.9.1 record's seed-1235 pass (+v0.9.1+ps__pass2; first pass of a fresh server, 2026-10-09):
      E = 5 (all cap exhaustion: docs 55, 79, 81, 121, 147), raw 176, stated 178 (published rows).
  Secondary: the column-511 arm's seed-1235 pass (+col511fix__pass2): E = 5 (cap), raw 170, stated 176.
  Context: 0.7.0's seed-1235 pass (+ps__pass2 of the 0.7.0 record), samples and request log.
  E = empty final replies of 198, split into cap exhaustion / premature stop / other.
  ACCURACY STANDARD (frozen; chosen and justified in README.md): the six-pass stock-0.9.1 family (record passes 1-3,
  column-511 arm passes 1-3; raw 164-176, mean 170.2, SD 3.9; stated 165-178, mean 172.8, SD 4.6), re-derived here:
      MATERIAL CONCERN (regression flag): raw <= 158 (mean - 3 SD; 6 below the family minimum) or stated <= 159
          (mean - 3 SD) or >= 2 premature stops;
      ACCURACY CONCERN: raw 159-163 (below every family pass).
  Matched-baseline offsets are reported as a SENSITIVITY readout only (not category-determining): the seed-1234 offsets
  (-11 / -11 / -6..-10) applied to record pass 2 give flag raw <= 165 or stated <= 167, concern raw 166-170.
  Categories (amendment 15):
      SIMILAR OR WORSE                 material concern (any E), or E >= 5 (baseline E)
      PROMISING FOLLOW-UP SIGNAL       E <= 2 and no accuracy concern
      MODEST IMPROVEMENT, NO CLEAR SEPARATION   otherwise (E = 3-4, or E <= 2 with an accuracy concern)
  Stated adjudication of every row where the stated adjudicator differs from flexible-extract (--adjudication) is required
  before the final category; otherwise PROVISIONAL.

PART B -- EXPLORATORY COMBINED ANALYSIS (labelled exploratory: the second pass was chosen after inspecting the first)
  Bundle pass 1 (seed 1234, screen 1) + this pass vs the matching stock passes (record passes 1-2); secondary vs the
  column-511 arm's passes 1-2. Per question d_q = comparator empties - bundle empties over the two seeds (0..2 each side);
  exact sign-flip (one-sided, fewer empties in the bundle; two-sided for reference) over questions, and a 20,000-replicate
  question bootstrap of the empty-rate difference (seed 20261012). Raw and stated correctness differences likewise
  (two-sided). Historical controls; exchangeability assumed, not designed; not a confirmation.
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
BUNDLE_R = "glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234"
SCREEN1 = "<runs>/bundle/screen/run-d1234"
REC = dict(passes=[f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps__pass{p}" for p in (1, 2, 3)],
           reqlog=[f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1.requests.jsonl", f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.requests.jsonl"],
           published="<this-repo>/runs/2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3_gpqa-diamond/results.jsonl")
ARM = dict(passes=[f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix__pass{p}" for p in (1, 2, 3)],
           reqlog=[f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix.s1.requests.jsonl",
                   f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix.s2.requests.jsonl"],
           adjudication="<runs>/arm-col511/adjudications.json")
V070 = dict(passes=[f"{G}/glm53-flash-exl3-k3.25@tpurtell__pass1", f"{G}/glm53-flash-exl3-k3.25@tpurtell+ps__pass2",
                    f"{G}/glm53-flash-exl3-k3.25@tpurtell+ps__pass3"],
            reqlog=[f"{G}/glm53-flash-exl3-k3.25@tpurtell+ps.requests.jsonl"])
FAMILY_RAW = (169, 176, 171, 164, 170, 171)        # record p1-3, arm p1-3 (raw flexible-extract, all 198)
FAMILY_STATED = (170, 178, 174, 165, 176, 174)     # record p1-3, arm p1-3 (stated, adjudicated)
RAW_FLAG, STATED_FLAG, RAW_CONCERN, PREMATURE_MAX = 158, 159, 163, 1
E_PROMISING, E_BASELINE = 2, 5
OFFSET_FLAG, OFFSET_CONCERN = 11, 6                 # sensitivity readout only (seed-1234 offsets vs its baseline 169/170)
BOOT_SEED = 20261012


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


def categorize(E, raw, stated, premature):
    """Amendment 15 categories on the family accuracy standard -> (material_concern, accuracy_concern, category)."""
    material = raw <= RAW_FLAG or stated <= STATED_FLAG or premature > PREMATURE_MAX
    concern = (not material) and RAW_FLAG < raw <= RAW_CONCERN
    if material or E >= E_BASELINE:
        return material, concern, "SIMILAR OR WORSE"
    if E <= E_PROMISING and not concern:
        return material, concern, "PROMISING FOLLOW-UP SIGNAL"
    return material, concern, "MODEST IMPROVEMENT, NO CLEAR SEPARATION"


def load_single(label, pass_dir, reqlogs, hedged_docs, seed_pass):
    """Load ONE pass directory as protocol pass `seed_pass` (request seed 1233 + seed_pass); rows keyed by doc."""
    saved = (A.N_PASSES, A.SEED_BASE)
    A.N_PASSES, A.SEED_BASE = 1, 1233 + seed_pass - 1
    try:
        rows, checks, _ = A.load_arm(label, [pass_dir], reqlogs, [f"1:{d}" for d in hedged_docs], None)
    finally:
        A.N_PASSES, A.SEED_BASE = saved
    out = {}
    for (d, _p), r in rows.items():
        r = dict(r, **{"pass": seed_pass})
        out[d] = r
    return out, checks


def load_three(label, spec, hedged=()):
    rows, checks, _ = A.load_arm(label, spec["passes"], spec["reqlog"], list(hedged), spec.get("published"))
    by = {}
    for (d, p), r in rows.items():
        by.setdefault(p, {})[d] = r
    return by, checks


def compare(S, B):
    se = {d for d, r in S.items() if r["empty"]}
    be = {d for d, r in B.items() if r["empty"]}
    raw_s = sum(r["correct_flexible"] for r in S.values())
    raw_b = sum(r["correct_flexible"] for r in B.values())
    st_s = sum(r["correct_stated"] for r in S.values())
    st_b = sum(r["correct_stated"] for r in B.values())
    return dict(baseline_empties=len(be),
                baseline_causes={c: sum(1 for d in be if cause(B[d]) == c) for c in ("cap_exhaustion", "premature_stop", "not_logged", "other_finish")},
                rescued=sorted(be - se), newly_empty=sorted(se - be), empty_in_both=sorted(se & be),
                baseline_raw=raw_b, baseline_stated=st_b, raw_diff=raw_s - raw_b, stated_diff=st_s - st_b,
                raw_only_screen_correct=sum(1 for d in S if S[d]["correct_flexible"] and not B[d]["correct_flexible"]),
                raw_only_baseline_correct=sum(1 for d in S if B[d]["correct_flexible"] and not S[d]["correct_flexible"]))


def combined(bundle_passes, cmp_passes, label):
    """EXPLORATORY: per-question sums over the paired seeds; questions are the unit."""
    docs = sorted(bundle_passes[0])
    d_e = [sum(int(c[q]["empty"]) for c in cmp_passes) - sum(int(b[q]["empty"]) for b in bundle_passes) for q in docs]
    d_raw = [sum(int(b[q]["correct_flexible"]) for b in bundle_passes) - sum(int(c[q]["correct_flexible"]) for c in cmp_passes) for q in docs]
    d_st = [sum(int(b[q]["correct_stated"]) for b in bundle_passes) - sum(int(c[q]["correct_stated"]) for c in cmp_passes) for q in docs]
    n = len(bundle_passes) * len(docs)
    sf = A.signflip(d_e)
    lo, hi = A.boot_interval(d_e, n, BOOT_SEED)
    rlo, rhi = A.boot_interval(d_raw, n, BOOT_SEED + 1)
    slo, shi = A.boot_interval(d_st, n, BOOT_SEED + 2)
    return dict(label=f"EXPLORATORY ({label}): follow-up chosen after inspecting the first screen; historical controls",
                question_passes_per_side=n,
                bundle_empties=sum(int(b[q]["empty"]) for b in bundle_passes for q in docs),
                comparator_empties=sum(int(c[q]["empty"]) for c in cmp_passes for q in docs),
                empty_signflip_cmp_minus_bundle=sf,
                empty_rate_reduction_points=round(100 * sum(d_e) / n, 2),
                empty_rate_reduction_ci95_points=[round(100 * lo, 2), round(100 * hi, 2)],
                questions_d_pos=sum(1 for x in d_e if x > 0), questions_d_neg=sum(1 for x in d_e if x < 0),
                raw_diff_points=round(100 * sum(d_raw) / n, 2), raw_diff_ci95_points=[round(100 * rlo, 2), round(100 * rhi, 2)],
                raw_signflip_two_sided_reference=A.signflip(d_raw)["p_two_sided"],
                stated_diff_points=round(100 * sum(d_st) / n, 2), stated_diff_ci95_points=[round(100 * slo, 2), round(100 * shi, 2)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-dir", required=True)
    ap.add_argument("--reqlog", required=True)
    ap.add_argument("--serverlog", default=None)
    ap.add_argument("--gpu-hours", type=float, default=None)
    ap.add_argument("--adjudication", default=None,
                    help="JSON {rows: [{doc, hedged}]}: hand reads of every row where stated differs from flexible-extract")
    ap.add_argument("--label", default="bundle second pass (seed 1235)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    adj_rows = json.load(open(a.adjudication))["rows"] if a.adjudication else []
    S, checks = load_single(a.label, a.pass_dir, [a.reqlog], [int(r["doc"]) for r in adj_rows if r.get("hedged")], 2)
    rec, c1 = load_three("0.9.1 record", REC)
    arm_adj = json.load(open(ARM["adjudication"]))["rows"]
    arm, c2 = load_three("col511 arm", ARM, [f"{r['pass']}:{r['doc']}" for r in arm_adj if r["hedged"]])
    v070, c3 = load_three("0.7.0 record (context)", V070)
    checks += [c for c in c1 + c2 + c3 if "pass 2" in c[0] or "published" in c[0]]
    b1_adj = json.load(open(os.path.join(SCREEN1, "adjudication.json")))["rows"]
    B1, c4 = load_single("bundle screen 1 (seed 1234)", f"{G}/{BUNDLE_R}__pass1", [f"{G}/{BUNDLE_R}.s1.requests.jsonl"],
                         [int(r["doc"]) for r in b1_adj if r.get("hedged")], 1)
    checks += c4
    for name, B in (("record pass 2", rec[2]), ("arm pass 2", arm[2])):
        mism = [d for d in S if S[d]["messages_sha256"] != B[d]["messages_sha256"]]
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
           "script_sha256": A.file_sha(os.path.abspath(__file__)), "analysis_col511_sha256": A.file_sha(A.__file__),
           "accuracy_standard": dict(family_raw=FAMILY_RAW, family_stated=FAMILY_STATED,
                                     raw_mean=round(statistics.mean(FAMILY_RAW), 2), raw_sd=round(statistics.stdev(FAMILY_RAW), 2),
                                     stated_mean=round(statistics.mean(FAMILY_STATED), 2), stated_sd=round(statistics.stdev(FAMILY_STATED), 2),
                                     material=f"raw <= {RAW_FLAG} or stated <= {STATED_FLAG} or premature stops >= {PREMATURE_MAX + 1}",
                                     concern=f"raw {RAW_FLAG + 1}-{RAW_CONCERN}")}
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
        material, concern, outcome = categorize(E, raw, stated, prem)
        read = {int(r["doc"]) for r in adj_rows}
        missing = [d for d in sorted(d for d, r in S.items() if r["stated_differs_from_flexible"]) if d not in read]
        final = outcome if not missing else f"PROVISIONAL ({outcome}; stated adjudication pending for docs {missing})"
        rb, sb = sum(r["correct_flexible"] for r in rec[2].values()), sum(r["correct_stated"] for r in rec[2].values())
        sens_flag = raw <= rb - OFFSET_FLAG or stated <= sb - OFFSET_FLAG or prem > PREMATURE_MAX
        sens_concern = (not sens_flag) and raw <= rb - OFFSET_CONCERN
        ans = [r["completion_tokens"] for r in S.values() if not r["empty"] and r["completion_tokens"] is not None]
        A.N_PASSES = 1
        try:
            tim, _ = (A.timing({(d, 1): dict(r, **{"pass": 1}) for d, r in S.items()}, "pass2", None, a.gpu_hours, 2)
                      if all(r["logged"] for r in S.values()) else ({}, None))
        finally:
            A.N_PASSES = 3
        res["part_A_new_pass"] = {
            "empties": E, "causes": causes, "category": final, "category_if_adjudication_complete": outcome,
            "adjudication_complete": not missing, "adjudicated_docs": sorted(read),
            "flags_family_standard": {"material_concern": material, "accuracy_concern": concern,
                                      "detail": {"raw": raw, "stated": stated, "premature_stops": prem}},
            "sensitivity_matched_offsets_vs_record_pass2": {"baseline_raw": rb, "baseline_stated": sb,
                                                            "flag": sens_flag, "concern": sens_concern,
                                                            "rule": f"flag raw <= {rb - OFFSET_FLAG} or stated <= {sb - OFFSET_FLAG} "
                                                                    f"or >= 2 premature; concern raw {rb - OFFSET_FLAG + 1}-{rb - OFFSET_CONCERN}"},
            "versus": {"record pass 2 (primary, seed 1235)": compare(S, rec[2]),
                       "col511 arm pass 2 (secondary, seed 1235)": compare(S, arm[2]),
                       "0.7.0 pass 2 (context, seed 1235)": compare(S, v070[2])},
            "secondary": {"raw_correct_all_198": raw, "stated_correct_all_198": stated,
                          "stated_differs_from_raw_rows": [d for d, r in S.items() if r["stated_differs_from_flexible"]],
                          "answered_completion_tokens": A.quantiles(ans) if ans else None,
                          "answered_above_32k_64k_128k_200k": [sum(t > x for t in ans) for x in A.LEN_THRESHOLDS],
                          "timing": tim, "server": srv}}
        res["part_B_exploratory_combined"] = {
            "vs_record_passes_1_2": combined([B1, S], [rec[1], rec[2]], "bundle seeds 1234+1235 vs stock record seeds 1234+1235"),
            "vs_arm_passes_1_2": combined([B1, S], [arm[1], arm[2]], "bundle seeds 1234+1235 vs column-511 arm seeds 1234+1235")}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({k: res[k] for k in res if k != "checks"}, indent=1, default=str)[:9000])
    bad = [c for c in res["checks"] if not c[1]]
    if bad:
        print("INVALID:", json.dumps(bad, default=str)[:1500])
    return 0 if valid else 3


if __name__ == "__main__":
    sys.exit(main())
