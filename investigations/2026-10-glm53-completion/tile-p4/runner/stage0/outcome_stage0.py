#!/usr/bin/env python3
"""Stage-0 attempt outcome (public): start-up gates, client completion, instrument health.  Host python3.

usage: outcome_stage0.py AO ATTEMPT        writes AO/outcome.json and prints a one-line verdict

verdict  COMPLETE    G-instr and G3 passed at start-up (and G3 on the final bind log); all four replicates ran; every
                     forced request returned the forced ids (finish "length"); every forced request was flushed on
                     rank 0 and its entry digests on both ranks; no hook error.  The gates G-enq / G-entry / G-out are
                     then read by analysis/analyze_stage0.py (stage0-result.md); COMPLETE says only that they can be.
         PARTIAL     start-up gates passed and the client ran, but not every replicate / request / flush is complete
         INCOMPLETE  a start-up gate failed or the client did not run: no reading
"client_complete" (all four replicates done) is what budget_stage0.py treats as "this probe ran".
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
                except Exception:
                    out.append(dict(kind="_malformed"))
    return out


def main():
    ao, attempt = sys.argv[1], int(sys.argv[2])
    gate = open(os.path.join(ao, "gate.txt")).read() if os.path.exists(os.path.join(ao, "gate.txt")) else ""
    gstart = json.load(open(os.path.join(ao, "gates-startup.json"))) if os.path.exists(
        os.path.join(ao, "gates-startup.json")) else {}
    gfinal = json.load(open(os.path.join(ao, "gates-final.json"))) if os.path.exists(
        os.path.join(ao, "gates-final.json")) else {}
    cl = jl(os.path.join(ao, "client.jsonl"))
    end = [d for d in cl if d.get("kind") == "end"]
    done = end[-1].get("replicates_done", []) if end else []
    forced = [d for d in cl if d.get("kind") == "forced"]
    warm = [d for d in cl if d.get("kind") == "warm"]
    logs = jl(os.path.join(ao, "p4log", "p4_r*_p*.jsonl"))
    errs = [r for r in logs if r.get("kind") == "p4_error"]
    flushed = {r.get("tag") for r in logs if r.get("kind") == "flushed"}
    ent = {}
    for r in logs:
        if r.get("kind") == "entry_flushed":
            ent.setdefault(r.get("tag"), set()).add(r.get("rank"))
    ftags = [d["tag"] for d in forced if d.get("ok")]
    ginstr = "G-instr: OK" in gate
    g3s = bool(gstart.get("ok"))
    g3f = bool(gfinal.get("ok"))
    client_complete = done == ["V1-r1", "V1-r2", "V2-r1", "V2-r2"]
    all_ok = bool(forced) and all(d.get("ok") for d in forced) and len(forced) == 32
    all_flushed = all(t in flushed for t in ftags) and all(ent.get(t) == {0, 1} for t in ftags)
    if not (ginstr and g3s) or not cl:
        verdict = "INCOMPLETE"
    elif client_complete and all_ok and all_flushed and not errs and g3f and all(d.get("ok") for d in warm):
        verdict = "COMPLETE"
    else:
        verdict = "PARTIAL"
    doc = dict(attempt=attempt, verdict=verdict, g_instr=ginstr, g3_startup=g3s, g3_final=g3f,
               client_complete=client_complete, replicates_done=done, forced=len(forced),
               forced_ok=sum(1 for d in forced if d.get("ok")), warm=len(warm), warm_ok=sum(1 for d in warm if d.get("ok")),
               flushed=len(flushed & set(ftags)), entry_flushed_both_ranks=sum(1 for t in ftags if ent.get(t) == {0, 1}),
               hook_errors=len(errs), skips=[d for d in cl if d.get("kind") == "skip"])
    json.dump(doc, open(os.path.join(ao, "outcome.json"), "w"), indent=1)
    print("stage0 attempt %d: %s (replicates %s; forced ok %d/%d; flushed %d; entry both ranks %d; hook errors %d)" % (
        attempt, verdict, done, doc["forced_ok"], doc["forced"], doc["flushed"], doc["entry_flushed_both_ranks"],
        doc["hook_errors"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
