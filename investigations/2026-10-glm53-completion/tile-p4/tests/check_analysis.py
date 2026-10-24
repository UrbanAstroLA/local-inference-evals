#!/usr/bin/env python3
"""Dry runs of analysis/analyze_p4.py on the fabricated scenarios (tests/make_fake_sessions.py).  CPU container.
Each scenario must produce its planted reading; exclusions and acceptance must be exact."""
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
        "null": "Neither E1 nor E3",
        "gatefail": "INCOMPLETE",
        "noisy": "Computational propagation established",
        "noisynull": "No demonstrated propagation",
        "shapes": "Computational propagation established",
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
    check("noisy: fallback rule used and AB beyond-count exceeds the null pairing",
          "fallback rule used" in ny["reading"] and ny["result"]["beyond_fallback"]["lp_forced"]["ab"] >
          ny["result"]["beyond_fallback"]["lp_forced"]["null"])
    nn = docs["noisynull"]
    check("noisynull: fallback, AB beyond-count does not exceed the null pairing",
          "fallback rule used" in nn["reading"] and nn["result"]["beyond_fallback"]["lp_forced"]["ab"] <=
          nn["result"]["beyond_fallback"]["lp_forced"]["null"])
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
