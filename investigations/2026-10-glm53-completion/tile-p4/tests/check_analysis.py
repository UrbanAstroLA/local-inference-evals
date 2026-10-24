#!/usr/bin/env python3
"""Dry runs of analysis/analyze_p4.py on the fabricated scenarios (tests/make_fake_sessions.py).  CPU container.
Each scenario must produce its planted reading; exclusions and acceptance must be exact.
2026-10-24: zero / partial coverage, reused request indices (schema 1 and 2), run-1-style records, missing endpoints,
a missing rank-1 record and a bad forced request must read INCOMPLETE (never a negative finding); A not bit-stable
reads INCOMPLETE (the threshold-exceedance fallback is descriptive only)."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_fake_sessions as MF  # noqa: E402

AN = os.path.join(os.path.dirname(HERE), "analysis", "analyze_p4.py")
RES = []


def check(name, ok, detail=""):
    ok = bool(ok)
    RES.append(ok)
    print(("PASS " if ok else "FAIL ") + name + (" " + detail if detail else ""), flush=True)


def run(root, scen):
    out = os.path.join(root, scen, "analysis-out")
    p = subprocess.run([sys.executable, AN, os.path.join(root, scen), out], capture_output=True, text=True)
    doc = json.load(open(os.path.join(out, "result.json")))
    return p.returncode, doc, p.stdout + p.stderr


def main():
    root = sys.argv[1]
    exp = {
        "propagation": "Computational propagation established",
        "attenuation": "No demonstrated propagation",
        "null": "No reproducible A-vs-B difference at the compared positions",
        "gatefail": "INCOMPLETE",
        "noisy": "INCOMPLETE; no reading (A is bit-stable on only",
        "noisynull": "INCOMPLETE; no reading (A is bit-stable on only",
        "shapes": "Computational propagation established",
        "zerocov": "INCOMPLETE; no reading (insufficient comparison coverage",
        "partialcov": "INCOMPLETE; no reading (insufficient comparison coverage",
        "reused": "Computational propagation established",
        "reused_s1": "INCOMPLETE; descriptive only (run-1 records",
        "run1style": "INCOMPLETE; descriptive only (run-1 records",
        "missing": "INCOMPLETE; no reading (failed forcing / attribution / measurement checks",
        "rank1missing": "INCOMPLETE; no reading (failed forcing / attribution / measurement checks",
        "idsbad": "INCOMPLETE; no reading (failed forcing / attribution / measurement checks",
    }
    docs = {}
    for scen, want in exp.items():
        rc, doc, log = run(root, scen)
        docs[scen] = doc
        check(f"{scen}: reading contains '{want}'", want.lower() in doc.get("reading", "").lower(), doc.get("reading", "")[:160] +
              ("" if rc == 0 else " | " + log[-300:]))
    d = docs["propagation"]
    r = d["result"]["reproducible"]
    check("propagation: E1/E2/E3/E4 reproducible differences present; A-stable everywhere compared",
          r["e1_moe3"]["D"] > 0 and r["e2_final"]["D"] > 0 and r["e3_lp_forced"]["D"] > 0 and r["e4"]["D"] > 0
          and r["e3_lp_forced"]["a_stable"] == r["e3_lp_forced"]["compared"])
    check("propagation: o = 0 (prefill step) is not an E1 difference (B equal there)",
          r["e1_moe3"]["D"] == 4 * (MF.W - 1), str(r["e1_moe3"]["D"]))
    check("propagation: terminal-token differences in near-close windows are reported",
          "terminal-token log-probabilities" in d["reading"])
    c = d["result"]["comparisons"]
    check("propagation: within-arm and session-control comparisons are bit-equal (E1, E3)",
          c["AA"]["e1"]["moe3"]["bitequal_frac"] == 1.0 and c["AA2_1"]["e3"]["lp_forced"]["equal_frac"] == 1.0)
    check("propagation: E4 full-vocabulary KL > 0 for A-B, == 0 for A-A2; restricted top-20 kept as diagnostic",
          c["AB_1"]["e4"]["kl_ab"]["q50"] > 0 and c["AA2_1"]["e4"]["kl_ab"]["max"] == 0
          and "restricted_top20_tv_DIAGNOSTIC_ONLY" in c["AB_1"]["e4"])
    check("propagation: acceptance follows the schedule in every session",
          all(v["not_as_scheduled"] == 0 and v["steps_checked"] > 0 for v in d["acceptance"].values()))
    check("propagation: both ranks' records identical (validity)",
          all(v["rank_mismatch"] == 0 for v in d["result"]["validity"]["per_session"].values()))
    a = docs["attenuation"]["result"]["reproducible"]
    check("attenuation: E1 D > 0, E3 D == 0, E4 D == 0", a["e1_moe3"]["D"] > 0 and a["e3_lp_forced"]["D"] == 0 and a["e4"]["D"] == 0)
    n = docs["null"]["result"]["reproducible"]
    check("null: no reproducible difference in any family", all(v["D"] == 0 for v in n.values()))
    ny = docs["noisy"]
    te = ny["result"]["threshold_exceedance_DESCRIPTIVE_ONLY"]["lp_forced"]
    check("noisy: AB exceeds the null pairing in the descriptive table, yet the reading is INCOMPLETE (no fallback "
          "effect criterion)", te["ab"] > te["null"] and ny["reading"].startswith("INCOMPLETE"))
    md = open(os.path.join(root, "noisy", "analysis-out", "result.md")).read()
    check("noisy: result.md labels the threshold-exceedance table DESCRIPTIVE ONLY", "DESCRIPTIVE ONLY" in md)
    neg = ("No reproducible", "did not change", "Neither E1 nor E3")
    for scen in ("zerocov", "partialcov", "reused_s1", "run1style", "missing", "rank1missing", "idsbad", "gatefail",
                 "noisy", "noisynull"):
        rd = docs[scen].get("reading", "")
        check(f"{scen}: INCOMPLETE and no negative finding", rd.startswith("INCOMPLETE") and not any(n in rd for n in neg),
              rd[:120])
    zc = docs["zerocov"]["result"]["coverage"]
    check("zerocov: AB coverage exactly 0, A-A coverage complete", zc["pairs"]["AB_1"]["compared"] == 0
          and zc["pairs"]["AA"]["fraction"] == 1.0, json.dumps(zc["pairs"])[:200])
    pc = docs["partialcov"]["result"]["coverage"]
    check("partialcov: AB coverage between 0 and the 50 % minimum", 0 < pc["pairs"]["AB_1"]["fraction"] < 0.5,
          str(pc["pairs"]["AB_1"]))
    ru = docs["reused"]["result"]
    check("reused: schema-2 uids attribute every step correctly although indices are reused (attribution 0 "
          "mismatches; same D counts as 'propagation')",
          all(v["attribution_mismatch"] == 0 and v["attribution_checked"] > 0 for v in ru["validity"]["per_session"].values())
          and ru["reproducible"]["e3_lp_forced"]["D"] == docs["propagation"]["result"]["reproducible"]["e3_lp_forced"]["D"])
    r1 = docs["reused_s1"]["result"]
    check("reused_s1: run-1 path (step rings ignored) still compares positions from tag-keyed records",
          r1["run1_path"] and r1["coverage"]["pairs"]["AB_1"]["compared"] > 0)
    check("missing: the missing store is counted", docs["missing"]["result"]["validity"]["per_session"]["B"]["p4_missing"] == 1)
    check("rank1missing: the missing rank-1 record is counted",
          docs["rank1missing"]["result"]["validity"]["per_session"]["A2"]["rank1_missing"] == 1)
    check("idsbad: the failed forced request is counted", docs["idsbad"]["clients"]["B"]["ok"] == docs["idsbad"]["clients"]["B"]["planned"] - 1)
    sh = docs["shapes"]["result"]["exclusions"]
    expect = 2 * sum(1 for o, v in MF.STEPMAP.items() if 10 <= v[0] < 50)
    check("shapes: positions in steps whose key differs are excluded and counted exactly (AB_1 and BB)",
          sh["AB_1"].get("step key mismatch") == expect and sh["BB"].get("step key mismatch") == expect
          and not sh["AB_2"], f"AB_1 {sh['AB_1']} BB {sh['BB']} expected {expect}")
    check("gatefail: no endpoint is read (INCOMPLETE, no result section)", "result" not in docs["gatefail"] and docs["gatefail"]["reading"].startswith("INCOMPLETE"))
    check("result.md written for every scenario", all(os.path.exists(os.path.join(root, s, "analysis-out", "result.md")) for s in exp))
    print("check_analysis: %d/%d" % (sum(RES), len(RES)))
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
