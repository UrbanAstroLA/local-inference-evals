#!/usr/bin/env python3
"""Expected outcomes of the synthetic retry dry runs (tests/run_synth_fcr.sh): GATE-1R verdict, exactly the planted
row(s) failing (listed consequences included), and the final reading class. Host python, json only.
usage: check_synth_fcr.py OUT_DIR  -> rc 0 iff every variant behaves as expected"""
import json
import os
import sys

D = sys.argv[1]
NR = "NO READING: a gate failed"
EXP = {
    # attempt-1 variants under the retry pipeline: unchanged rows still catch their plants
    "good": dict(gate="PASS", fail=set(), reading="supported local computational mechanism candidate"),
    "null": dict(gate="PASS", fail=set(), reading="no termination-propensity effect detected"),
    "uninformative": dict(gate="PASS", fail=set(), reading="uninformative"),
    "noisy": dict(gate="PASS", fail=set(), reading="NO READING: within-arm variation"),
    "g-wmax": dict(gate="FAIL", fail={"G1-wmax", "G1-forcing"}, reading=NR),
    "g-abspos": dict(gate="FAIL", fail={"G1-abspos"}, reading=NR),
    "g-cache": dict(gate="FAIL", fail={"R-cache-model"}, reading=NR),          # a safe miss now stops via the model
    "g-attrib": dict(gate="FAIL", fail={"G1-attrib"}, reading=NR),
    "g-align": dict(gate="FAIL", fail={"G1-align", "G1-cross"}, reading=NR),
    "g-cross": dict(gate="FAIL", fail={"G1-cross"}, reading=NR),
    "g-regime": dict(gate="FAIL", fail={"G1-regime"}, reading=NR),
    "g-content": dict(gate="FAIL", fail={"G1-content"}, reading=NR),          # 26 pools spread: no page
    "g-clean": dict(gate="FAIL", fail={"G1-clean"}, reading=NR),
    "g-cvmig": dict(gate="FAIL", fail={"G1-CV-persist", "G1-CV-conv"}, reading=NR),
    "g-cvslot": dict(gate="FAIL", fail={"G1-CV-slot"}, reading=NR),
    "g-cvbook": dict(gate="FAIL", fail={"G1-CV-book", "G1-CV-persist", "G1-CV-conv"}, reading=NR),
    "a-ps": dict(gate="PASS", fail=set(), reading="NO READING: a gate failed (['R-PS-bytes'])"),
    "a-guard": dict(gate="PASS", fail=set(), reading="NO READING: a gate failed (['H5g-fc'])"),
    "a-incomplete": dict(gate="PASS", fail=set(), reading="INCOMPLETE"),
    "a-c3": dict(gate="PASS", fail=set(), reading="NO READING: a gate failed (['C3-residue'])"),
    # retry plants
    "r-cache-model": dict(gate="FAIL", fail={"R-cache-model"}, reading=NR),
    "r-cache-unsafe": dict(gate="FAIL", fail={"R-cache-safe", "R-cache-model"}, reading=NR),
    "r-cache-attrib": dict(gate="FAIL", fail={"R-cache-safe"}, reading=NR),
    "r-ps-cover": dict(gate="FAIL", fail={"R-PS-coverage", "R-cache-model"}, reading=NR),
    "r-ps-ident": dict(gate="FAIL", fail={"R-PS-identity", "R-cache-safe"}, reading=NR),
    "r-f6-page": dict(gate="FAIL", fail={"R-idx-pages", "R-idx-early"}, reading=NR),   # the swap includes DSA layers 3 and 7
    "r-balance": dict(gate="PASS", fail=set(), reading="NO READING: a gate failed (['R-cache-balance'])"),
}
ok_all, n = True, 0
only = sys.argv[2:] or list(EXP)
for v in only:
    e = EXP[v]
    p = os.path.join(D, "synth-" + v, "full")
    try:
        g = json.load(open(os.path.join(p, "gate1r", "gate1r.json")))
        r = json.load(open(os.path.join(p, "fcr_result.json")))
    except Exception as exc:  # noqa: BLE001
        print("FAIL %-15s missing outputs: %r" % (v, exc))
        ok_all = False
        continue
    failing = {x["id"] for x in g["rows"] if x["status"] == "FAIL"}
    ok = g["verdict"] == e["gate"] and failing == e["fail"] and r["reading"].startswith(e["reading"])
    n += ok
    ok_all &= ok
    print("%s %-15s GATE-1R %s failing %s stop %s | reading: %s" % ("PASS" if ok else "FAIL", v, g["verdict"],
                                                                    sorted(failing), g.get("stop_classes"),
                                                                    r["reading"][:100]))
print("synthetic variants: %d/%d behave as expected" % (n, len(only)))
sys.exit(0 if ok_all else 1)
