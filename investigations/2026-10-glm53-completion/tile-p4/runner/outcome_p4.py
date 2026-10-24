#!/usr/bin/env python3
"""P4 session outcome (public): gates, coverage and instrument health of one session attempt.  Host python3.

usage: outcome_p4.py AO SESSION ATTEMPT       writes AO/outcome.json and prints a one-line verdict

verdict  COMPLETE    start-up and final gates passed; all 6 batches ran; every forced request returned the forced
                     ids with finish "length"; every completed request was flushed; no hook error
         PARTIAL     gates passed, at least one batch completed, but not all (time or an error); usable on the
                     batches it covers (the same leading batches in every session)
         INCOMPLETE  a gate failed or no batch completed: no reading (PROPOSAL.md section 6)
budget_p4.py treats COMPLETE and PARTIAL as "done" for the session order (gates_ok and batches_done >= 1).
"""
import glob
import json
import os
import sys


def jl(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    out.append(dict(kind="_malformed"))
    return out


def main():
    ao, session, attempt = sys.argv[1], sys.argv[2], int(sys.argv[3])
    gs = json.load(open(os.path.join(ao, "gates-startup.json"))) if os.path.exists(os.path.join(ao, "gates-startup.json")) else {}
    gf = json.load(open(os.path.join(ao, "gates-final.json"))) if os.path.exists(os.path.join(ao, "gates-final.json")) else {}
    cl = jl(os.path.join(ao, "client.jsonl"))
    forced = [d for d in cl if d.get("kind") == "forced"]
    passes = [d for d in cl if d.get("kind") == "pass"]
    done = [d["batch"] for d in cl if d.get("kind") == "batch_done"]
    end = [d for d in cl if d.get("kind") == "end"]
    n_batches = next((d.get("n_batches") for d in cl if d.get("kind") == "start"), 6)
    p4recs = []
    for p in glob.glob(os.path.join(ao, "p4log", "p4_r*_p*.jsonl")):
        p4recs += jl(p)
    errors = [r for r in p4recs if r.get("kind") == "p4_error"]
    flushed = {r.get("tag") for r in p4recs if r.get("kind") == "flushed"}
    refused = [r for r in p4recs if r.get("kind") in ("flush_refused", "no_slot")]
    ok_req = [d for d in forced if d.get("ok")]
    not_flushed = sorted(d["tag"] for d in forced if d.get("status") == 200 and d["tag"] not in flushed)
    cache = dict(n=sum(1 for d in forced if d.get("cached_tokens") is not None),
                 equal_expected=sum(1 for d in forced if d.get("cached_tokens") == d.get("cached_expected")),
                 below_expected=sum(1 for d in forced if (d.get("cached_tokens") or 0) < d.get("cached_expected", 0)))
    gates_ok = bool(gs.get("ok")) and bool(gf.get("ok"))
    full = (gates_ok and len(done) == n_batches and len(forced) == 16 * n_batches and len(ok_req) == len(forced)
            and not not_flushed and not errors and not refused)
    verdict = "COMPLETE" if full else ("PARTIAL" if gates_ok and done else "INCOMPLETE")
    o = dict(session=session, attempt=attempt, verdict=verdict, gates_ok=gates_ok,
             gates_startup={k: v.get("ok") for k, v in (gs.get("gates") or {}).items()},
             gates_final={k: v.get("ok") for k, v in (gf.get("gates") or {}).items()},
             batches_done=len(done), batches=done, n_batches=n_batches, forced_requests=len(forced),
             forced_ok=len(ok_req), ids_equal=sum(1 for d in forced if d.get("ids_equal")),
             passes=[dict(batch=d["batch"], rep=d["rep"], ok=d["ok"], n=d["n"], secs=d.get("secs")) for d in passes],
             flushed=len(flushed), not_flushed=not_flushed, hook_errors=len(errors), flush_refused_or_no_slot=len(refused),
             prefix_cache=cache, client_ended=bool(end),
             skips=[d for d in cl if d.get("kind") == "skip"])
    json.dump(o, open(os.path.join(ao, "outcome.json"), "w"), indent=1)
    print(f"OUTCOME session {session} attempt {attempt}: {verdict} (gates_ok {gates_ok}; batches {len(done)}/{n_batches}; "
          f"forced ok {len(ok_req)}/{len(forced)}; flushed {len(flushed)}; hook errors {len(errors)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
