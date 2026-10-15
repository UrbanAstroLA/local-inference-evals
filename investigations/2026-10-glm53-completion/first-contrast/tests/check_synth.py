#!/usr/bin/env python3
"""Expected outcomes of the synthetic dry runs (tests/run_synth.sh): GATE-1 verdict, the planted row(s) failing (and no
other GATE-1 row failing unless listed as a consequence), and the final reading class. Host Python, json only.
usage: check_synth.py OUT_DIR  -> rc 0 iff every variant behaves as expected"""
import json
import os
import sys

D = sys.argv[1]
GOOD = dict(gate="PASS", fail=set())
EXP = {
    "good": dict(gate="PASS", fail=set(), reading="supported local computational mechanism candidate"),
    "null": dict(gate="PASS", fail=set(), reading="no termination-propensity effect detected"),
    "uninformative": dict(gate="PASS", fail=set(), reading="uninformative"),
    "noisy": dict(gate="PASS", fail=set(), reading="NO READING: within-arm variation"),
    "g-wmax": dict(gate="FAIL", fail={"G1-wmax", "G1-forcing"}, reading="NO READING: a gate failed"),
    "g-abspos": dict(gate="FAIL", fail={"G1-abspos"}, reading="NO READING: a gate failed"),
    "g-cache": dict(gate="FAIL", fail={"G1-cache"}, reading="NO READING: a gate failed"),
    "g-attrib": dict(gate="FAIL", fail={"G1-attrib"}, reading="NO READING: a gate failed"),
    "g-align": dict(gate="FAIL", fail={"G1-align", "G1-cross"}, reading="NO READING: a gate failed"),
    "g-cross": dict(gate="FAIL", fail={"G1-cross"}, reading="NO READING: a gate failed"),
    "g-regime": dict(gate="FAIL", fail={"G1-regime"}, reading="NO READING: a gate failed"),
    "g-content": dict(gate="FAIL", fail={"G1-content"}, reading="NO READING: a gate failed"),
    "g-clean": dict(gate="FAIL", fail={"G1-clean"}, reading="NO READING: a gate failed"),
    "g-cvmig": dict(gate="FAIL", fail={"G1-CV-persist", "G1-CV-conv"}, reading="NO READING: a gate failed"),
    "g-cvslot": dict(gate="FAIL", fail={"G1-CV-slot"}, reading="NO READING: a gate failed"),
    "g-cvbook": dict(gate="FAIL", fail={"G1-CV-book", "G1-CV-persist", "G1-CV-conv"}, reading="NO READING: a gate failed"),
    "a-ps": dict(gate="PASS", fail=set(), reading="NO READING: a gate failed (['PS-prefix'])"),
    "a-guard": dict(gate="PASS", fail=set(), reading="NO READING: a gate failed (['H5g-fc'])"),
    "a-incomplete": dict(gate="PASS", fail=set(), reading="INCOMPLETE"),
    "a-c3": dict(gate="PASS", fail=set(), reading="NO READING: a gate failed (['C3-residue'])"),
}
ok_all, n = True, 0
for v, e in EXP.items():
    p = os.path.join(D, "synth-" + v, "full")
    try:
        g = json.load(open(os.path.join(p, "gate1", "gate1.json")))
        r = json.load(open(os.path.join(p, "fc_result.json")))
    except Exception as exc:  # noqa: BLE001
        print("FAIL %-14s missing outputs: %r" % (v, exc))
        ok_all = False
        continue
    failing = {x["id"] for x in g["rows"] if x["status"] == "FAIL"}
    ok = g["verdict"] == e["gate"] and failing == e["fail"] and r["reading"].startswith(e["reading"])
    n += ok
    ok_all &= ok
    print("%s %-14s GATE-1 %s failing %s | reading: %s" % ("PASS" if ok else "FAIL", v, g["verdict"], sorted(failing),
                                                            r["reading"][:110]))
print("synthetic variants: %d/%d behave as expected" % (n, len(EXP)))
sys.exit(0 if ok_all else 1)
