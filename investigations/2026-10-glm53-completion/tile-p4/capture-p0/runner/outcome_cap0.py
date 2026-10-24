#!/usr/bin/env python3
"""capture-p0 attempt outcome (public): start-up gates, client completion, instrument health.  Host python3.

usage: outcome_cap0.py AO ATTEMPT        writes AO/outcome.json and prints a one-line verdict

verdict  COMPLETE    G-instr and G3 passed at start-up (and G3 on the final bind log); every planned replicate ran
                     with its warm and capture requests ok; a capture record on both ranks for each; a localisation
                     record on both ranks for each "-L" warm; no hook error.  The reading itself is
                     analysis/analyze_cap0.py's (cap0-result.md); COMPLETE says only that it can be read.
         PARTIAL     start-up gates passed and the client ran, but something above is missing (the analysis decides
                     whether >= 2 replicates still count)
         INCOMPLETE  a start-up gate failed or the client did not run: no reading
"client_complete" (the client reached its end record) is what budget_cap0.py treats as "this session ran".
"""
import glob
import json
import os
import sys


def jl(path):
    out = []
    for p in sorted(glob.glob(path)):
        for line in open(p):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    out.append(dict(kind="_malformed"))
    return out


def main():
    ao, attempt = sys.argv[1], int(sys.argv[2])
    gate = open(os.path.join(ao, "gate.txt")).read() if os.path.exists(os.path.join(ao, "gate.txt")) else ""
    load = lambda n: json.load(open(os.path.join(ao, n))) if os.path.exists(os.path.join(ao, n)) else {}  # noqa: E731
    gstart, gfinal = load("gates-startup.json"), load("gates-final.json")
    cl = jl(os.path.join(ao, "client.jsonl"))
    start = [d for d in cl if d.get("kind") == "start"]
    planned = [r["name"] for r in (start[-1].get("replicates", []) if start else [])]
    locs = {r["name"]: r.get("loc") for r in (start[-1].get("replicates", []) if start else [])}
    end = [d for d in cl if d.get("kind") == "end"]
    done = end[-1].get("replicates_done", []) if end else []
    reqs = {(d.get("kind"), d.get("replicate")): d for d in cl if d.get("kind") in ("warm", "cap")}
    logs = jl(os.path.join(ao, "p4log", "p4_r*_p*.jsonl"))
    caplogs = jl(os.path.join(ao, "p4log", "cap0_r*_p*.jsonl"))
    errs = [r for r in logs if r.get("kind") == "p4_error"]
    cap = {(r.get("tag"), r.get("rank")) for r in caplogs if r.get("kind") == "cap0_capture" and r.get("ok")}
    loc = {(r.get("tag"), r.get("rank")) for r in caplogs if r.get("kind") == "cap0_loc_flushed"}
    rep_ok = {}
    for r in done:
        w, c = reqs.get(("warm", r)), reqs.get(("cap", r))
        ok = bool(w and w.get("ok") and c and c.get("ok")) and all((f"CAP-{r}", k) in cap for k in (0, 1))
        if locs.get(r):
            ok = ok and all((f"CW-{r}-L", k) in loc for k in (0, 1))
        rep_ok[r] = ok
    ginstr = "G-instr: OK" in gate
    g3s, g3f = bool(gstart.get("ok")), bool(gfinal.get("ok"))
    client_complete = bool(end)
    aborted = [d for d in cl if d.get("kind") == "abort"]
    if not (ginstr and g3s) or not cl:
        verdict = "INCOMPLETE"
    elif done == planned and planned and all(rep_ok.values()) and not errs and g3f:
        verdict = "COMPLETE"
    else:
        verdict = "PARTIAL"
    doc = dict(attempt=attempt, verdict=verdict, g_instr=ginstr, g3_startup=g3s, g3_final=g3f,
               client_complete=client_complete, planned=planned, replicates_done=done, replicate_ok=rep_ok,
               captures=sorted(cap), loc_flushed=sorted(loc), hook_errors=len(errs),
               skips=[d for d in cl if d.get("kind") == "skip"], aborted=aborted[-1] if aborted else None)
    json.dump(doc, open(os.path.join(ao, "outcome.json"), "w"), indent=1, default=str)
    print("cap0 session attempt %d: %s (replicates %s of %s; ok %s; hook errors %d)%s" % (
        attempt, verdict, done, planned, sum(rep_ok.values()), len(errs),
        ("; ABORTED: " + aborted[-1].get("reason", "")) if aborted else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
