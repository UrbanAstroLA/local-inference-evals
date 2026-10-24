#!/usr/bin/env python3
"""Dry runs of analysis/analyze_stage0.py on the fabricated attempts (tests/make_fake_stage0.py).  CPU container.
Each scenario must give its planted verdicts, first failed gate, first-difference descriptor and interpretation."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(os.path.dirname(HERE), "analysis", "analyze_stage0.py")
RES = []


def check(name, ok, detail=""):
    ok = bool(ok)
    RES.append(ok)
    print(("PASS " if ok else "FAIL ") + name + (" " + detail if detail else ""), flush=True)


def run(root, scen):
    ao = os.path.join(root, scen, "runs-stage0", "stage0-attempt-1")
    po = os.path.join(root, scen, "PRIVATE-runs-stage0", "stage0-attempt-1")
    p = subprocess.run([sys.executable, AN, ao, po], capture_output=True, text=True)
    try:
        doc = json.load(open(os.path.join(ao, "stage0-result.json")))
    except Exception:
        doc = {}
    return doc, p.stdout + p.stderr


def main():
    root = sys.argv[1]
    d = {}
    for scen in ("pass", "v2enq", "v2order", "entry", "prov", "out", "outcov", "missing", "stop"):
        d[scen], log = run(root, scen)
        check(f"{scen}: analysis ran and wrote stage0-result.json", bool(d[scen]), log[-300:])
    v = lambda s, k: (d[s].get("variants") or {}).get(k, {})  # noqa: E731
    check("pass: V1 and V2 REPRODUCIBLE; interpretation 'V2 reproducible ... common A/B starting construction'",
          v("pass", "V1").get("verdict") == "REPRODUCIBLE" and v("pass", "V2").get("verdict") == "REPRODUCIBLE"
          and "common A/B starting construction" in d["pass"].get("interpretation", ""))
    ge = (v("pass", "V1").get("gates") or {}).get("G-entry") or {}
    check("pass: null-block placeholders in a request's row are skipped and counted (not compared)",
          ge.get("null_in_row_skipped", 0) > 0)
    g = v("v2enq", "V2")
    fd = (g.get("gates", {}).get("G-enq") or {}).get("first_difference") or {}
    check("v2enq: V1 reproducible, V2 fails G-enq at the planted step (row 2, n_sched); later gates not evaluated",
          v("v2enq", "V1").get("verdict") == "REPRODUCIBLE" and g.get("first_failed") == "G-enq"
          and fd.get("field") == "rows" and fd.get("row") == 2 and fd.get("row_fields") == ["n_sched"]
          and isinstance(g["gates"].get("G-entry"), str), json.dumps(fd))
    check("v2enq: interpretation 'batch/concurrency-dependent ... mechanism remains unassigned'",
          "batch/concurrency-dependent" in d["v2enq"].get("interpretation", ""))
    g = v("v2order", "V2")
    check("v2order: V2 fails G-enq on the enqueue order", g.get("first_failed") == "G-enq"
          and g["gates"]["G-enq"].get("why") == "enqueue order differs")
    e1, e2 = v("entry", "V1"), v("entry", "V2")
    f1 = (e1.get("gates", {}).get("G-entry") or {}).get("first_difference") or {}
    f2 = (e2.get("gates", {}).get("G-entry") or {}).get("first_difference") or {}
    check("entry: V1 fails G-entry at point 1 (prefill), tensor mla.0, block 3, cached state (point 0) equal",
          e1.get("first_failed") == "G-entry" and f1.get("point") == 1 and f1.get("tensor") == "mla.0"
          and f1.get("logical_block") == 3 and f1.get("point0_cached_state_equal") is True, json.dumps(f1))
    check("entry: V2 fails G-entry at point 0 (cached state), rank 1, tensor kda.0#1",
          e2.get("first_failed") == "G-entry" and f2.get("point") == 0 and f2.get("rank") == 1
          and f2.get("tensor") == "kda.0#1" and f2.get("point0_cached_state_equal") is False, json.dumps(f2))
    check("entry: interpretation 'Neither reproducible ... do not label stock kernels nondeterministic'",
          "Neither reproducible" in d["entry"].get("interpretation", "")
          and "do not label stock kernels nondeterministic" in d["entry"]["interpretation"])
    check("entry: G-enq passed first (gates evaluated in order)",
          isinstance(e1["gates"].get("G-enq"), dict) and e1["gates"]["G-enq"].get("ok"))
    p = v("prov", "V1")
    pw = ((p.get("gates", {}).get("G-entry") or {}).get("windows") or {}).get("R02-deep", {})
    check("prov: V1 fails G-entry on provenance (one cached block in each of 2 groups not built by its warm request); digests equal",
          p.get("first_failed") == "G-entry" and pw.get("prov", [{}])[0].get("not_from_warm") == 2
          and pw.get("r0", {}).get("ok") and pw.get("r1", {}).get("ok"))
    o1 = v("out", "V1")
    fo = (o1.get("gates", {}).get("G-out") or {}).get("first_difference") or {}
    check("out: G-enq and G-entry pass; G-out fails at o = 7 of R02-deep (E3 entropy); first differing layer state 23; "
          "step composition equal",
          o1.get("first_failed") == "G-out" and fo.get("window") == "R02-deep" and fo.get("position") == 7
          and fo.get("endpoint") == "E3 differs (entropy)" and fo.get("first_differing_layer_state") == 23
          and fo.get("step_composition_equal") is True, json.dumps(fo))
    oc = v("outcov", "V2")
    foc = (oc.get("gates", {}).get("G-out") or {}).get("first_difference") or {}
    check("outcov: missing E1 coverage FAILS G-out (never a pass)", oc.get("first_failed") == "G-out"
          and foc.get("endpoint") == "E1 not staged" and foc.get("position") == 0, json.dumps(foc))
    check("missing: V2 INCOMPLETE; interpretation INCOMPLETE", v("missing", "V2").get("verdict") == "INCOMPLETE"
          and d["missing"].get("interpretation", "").startswith("INCOMPLETE"))
    s = v("stop", "V1")
    check("stop: first failed gate G-enq; G-entry and G-out not evaluated although an entry digest differs",
          s.get("first_failed") == "G-enq" and isinstance(s["gates"].get("G-entry"), str)
          and isinstance(s["gates"].get("G-out"), str))
    md = open(os.path.join(root, "out", "runs-stage0", "stage0-attempt-1", "stage0-result.md")).read()
    check("result.md: interpretation, gate table and the 64-position caveat", "Interpretation" in md and "| V1 |" in md
          and "do not certify every persistent-state byte" in md)
    print("check_stage0: %d/%d" % (sum(RES), len(RES)))
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
