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
    # [R-10] sentinel boundary state: SSM and conv separately, each its own baseline (22f s.2)
    "r-ps-conv": dict(gate="PASS", fail=set(), reading="NO READING: a gate failed (['R-PS-state-conv'])"),
    "r-ps-ssm": dict(gate="PASS", fail=set(), reading="NO READING: a gate failed (['R-PS-state-ssm'])"),
    "r-ps-nonid": dict(gate="PASS", fail=set(), reading="NO READING: a gate failed (['R-PS-state-conv'])"),
    "r-ps-nonid-ok": dict(gate="PASS", fail=set(), reading="supported local computational mechanism candidate"),
    # [R-12] non-discriminating pages reported, judgement unchanged (22f s.4)
    "r-f6-nd": dict(gate="PASS", fail=set(), reading="supported local computational mechanism candidate"),
    # [R-17] batch guard stop before C3 (22h)
    "r-guard-stop": dict(gate="PASS", fail=set(), reading="incomplete contrast (batch guard)"),
}


def extra(v, g, r):
    """[R-10..R-14] content checks of the outputs: (ok, note)."""
    lim = "physical KDA block identity and the complete unsampled state are not established"
    base_ok = lim in json.dumps(g.get("sentinel_limits", "")) and lim in r.get("sentinel_limits", "") and \
        any(x["id"] == "R-PS-limits" for x in r["gate_rows"]) and "eviction avoided in the replayed scenarios" in \
        r.get("cache_wording", "")
    nd = r.get("f6_not_independently_verified") or {}
    note = "limits+cache wording %s" % base_ok
    ok = base_ok and nd.get("wording") == "not independently verified by the numerical check"
    if v == "r-f6-nd":
        gp = (g.get("not_independently_verified") or {}).get("pages") or {}
        n_g = {t: x["n"] for t, x in gp.items()}
        ok = ok and all(n_g.get(t, 0) >= 2 for t in ("C1", "C2")) and nd["C3"]["n"] >= 2
        note += "; ND pages GATE-1R %s C3 %s" % (n_g, nd["C3"]["n"])
    elif v == "good":
        gp = (g.get("not_independently_verified") or {}).get("pages") or {}
        note += "; ND pages GATE-1R %s C3 %s" % ({t: x["n"] for t, x in gp.items()}, nd.get("C3", {}).get("n"))
    lp = r.get("lp_control") or {}
    if v in ("good", "null", "r-f6-nd", "r-ps-nonid-ok"):
        sd = lp.get("same_depth") or {}
        ok = ok and set(sd.get("matched_pairs", {})) == {"LC1/LD1", "LC2/LD2"} and \
            "depth-mismatched" in sd.get("depth_mismatched_pair", {}).get("note", "") and \
            sd.get("three_repeat_rule_applied_to_matched_pairs") is False and "conclusion_status" in lp
        note += "; LP same-depth: %s" % lp.get("conclusion_status")
    if v == "r-guard-stop":
        rows = {x["id"]: x for x in r["gate_rows"]}
        bgr = r.get("batch_guard") or {}
        ok = ok and rows.get("C3-residue", {}).get("status") == "NOT RUN" and bgr.get("stopped_before") == "C3" and \
            (bgr.get("stop") or {}).get("rc") == 17 and not any(x["status"] == "FAIL" for x in r["gate_rows"])
        note += "; C3-residue %s; stopped before %s (rc %s)" % (rows.get("C3-residue", {}).get("status"),
                                                               bgr.get("stopped_before"), (bgr.get("stop") or {}).get("rc"))
    elif v in ("good", "null"):
        ok = ok and not (r.get("batch_guard") or {}).get("stop")
    if v.startswith("r-ps-"):
        rows = {x["id"]: x for x in r["gate_rows"]}
        note += "; R-PS-state-ssm %s conv %s" % (rows.get("R-PS-state-ssm", {}).get("status"),
                                                 rows.get("R-PS-state-conv", {}).get("status"))
    return ok, note
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
    ok_x, note = extra(v, g, r)
    ok = g["verdict"] == e["gate"] and failing == e["fail"] and r["reading"].startswith(e["reading"]) and ok_x
    n += ok
    ok_all &= ok
    print("%s %-15s GATE-1R %s failing %s stop %s | reading: %s | %s" % ("PASS" if ok else "FAIL", v, g["verdict"],
                                                                         sorted(failing), g.get("stop_classes"),
                                                                         r["reading"][:100], note))
print("synthetic variants: %d/%d behave as expected" % (n, len(only)))
sys.exit(0 if ok_all else 1)
