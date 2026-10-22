#!/usr/bin/env python3
"""Incomplete-run handling for the stock 0.7.0 replication (PROPOSAL-v2.md section 6e; change set 22n). CPU only,
standard library; ids, hashes, counts and token lengths only.

  incomplete_rep070.py probe  --variant V --rep-pass P1 P2 P3 --ledger costs.jsonl [--watchdog-log F]
      exit 0: the run is COMPLETE (both sessions in the ledger, watchdog not fired, every pass has exactly one samples
              file with 198 questions and both filters) -> the runner calls the frozen analysis (analysis_rep070.py);
      exit 10: INCOMPLETE -> the runner calls 'report'; the causes are printed as JSON.
  incomplete_rep070.py report --variant V --rep-pass P1 P2 P3 --rep-reqlog S1 S2 --rep-events E1 E2 --ledger costs.jsonl
      [--watchdog-log F] [--artifacts PATH ...] --out OUT.json > OUT.md

The report labels the replication INCOMPLETE with its cause(s). It never applies O1/O2/O3 or any 594-based test, and it
never inserts an interrupted or unstarted request as an empty reply.
  - Fully completed passes (samples file with 198 questions): their results are reported through the frozen loader
    (empties, raw correct, provisional stated, finish reasons, that pass's own validity checks), descriptively.
  - Any other pass is counted from the proxy main log and its sidecar: completed (finish 'stop'), cap-reached (finish
    'length' at 327,680; 'empty' by the frozen definition only when the reply is empty, judged here from the proxy's
    content_chars = 0 because lm-eval writes no samples for an unfinished pass), other finishes, interrupted (started
    in the sidecar, no response logged), unstarted (198 minus started). Without a sidecar, started/interrupted/unstarted
    are reported as not determinable. Empties among completed requests are descriptive only.
"""
import argparse, glob, hashlib, json, os, sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis_rep070 as X  # noqa: E402  (checks the frozen module's hash on import)

M = X.M
CAP, NQ = M.CAP, M.N_Q


def fsha(p):
    return M.file_sha(p)


def pass_complete(d):
    fs = glob.glob(os.path.join(d, "**", "samples_*.jsonl"), recursive=True)
    if len(fs) != 1:
        return False, f"{len(fs)} samples files"
    by = {}
    try:
        for line in open(fs[0]):
            r = json.loads(line)
            by.setdefault(r["doc_id"], set()).add(r["filter"])
    except (OSError, ValueError, KeyError) as e:
        return False, f"unreadable samples ({type(e).__name__})"
    ok = sorted(by) == list(range(NQ)) and all(len(v) == 2 for v in by.values())
    return ok, f"{len(by)} questions"


def ledger(path):
    rows = []
    if path and os.path.exists(path):
        for line in open(path):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    rows.append({"unparsed": True})
    return rows


def causes_of(a):
    c = []
    L = ledger(a.ledger)
    if not L:
        c.append("no cost ledger (no session completed its bookkeeping)")
    wd = any(r.get("watchdog_fired") is True for r in L)
    if a.watchdog_log and os.path.exists(a.watchdog_log) and "CEILING REACHED" in open(a.watchdog_log, errors="replace").read():
        wd = True
    if wd:
        c.append("watchdog fired (approved GPU-hour ceiling reached)")
    if L and not any(r.get("session") == 2 for r in L):
        c.append("session 2 not started")
    for p, d in enumerate(a.rep_pass, 1):
        ok, det = pass_complete(d)
        if not ok:
            c.append(f"pass {p} incomplete or missing ({det})")
    return c, L


def interrupted_pass(seed, reqlogs, events):
    rows = [r for r in M.load_reqlogs([p for p in reqlogs if os.path.exists(p)]) if r.get("seed") == seed]
    ev = []
    have_sidecar = any(os.path.exists(p) for p in events)
    for p in events:
        if os.path.exists(p):
            ev += [json.loads(l) for l in open(p) if l.strip()]
    ev = [e for e in ev if e.get("seed") == seed]
    logged = {r.get("messages_sha256") for r in rows}
    started = {e.get("messages_sha256") for e in ev if e.get("event") == "start"}
    stop = [r for r in rows if r.get("finish_reason") == "stop"]
    cap = [r for r in rows if r.get("finish_reason") == "length" and r.get("completion_tokens") == CAP]
    other = [r for r in rows if r not in stop and r not in cap]
    out = {"request_seed": seed, "responses_logged": len(rows), "completed_finish_stop": len(stop),
           "cap_reached_finish_length_at_327680": len(cap),
           "cap_reached_with_empty_content": sum(r.get("content_chars") == 0 for r in cap),
           "cap_reached_with_content": sum((r.get("content_chars") or 0) > 0 for r in cap),
           "other_finish_or_http": len(other),
           "empty_among_completed_stop_descriptive": sum(r.get("content_chars") == 0 for r in stop),
           "http_non_200": sum(r.get("http_status") != 200 for r in rows),
           "sidecar_present": have_sidecar}
    if have_sidecar:
        out.update(started=len(started), interrupted_started_no_response=len(started - logged),
                   unstarted=NQ - len(started | logged), responses_without_start_event=len(logged - started),
                   interrupted_events=sum(e.get("event") == "interrupted" for e in ev),
                   interrupted_event_errors=sorted({e.get("error") for e in ev if e.get("event") == "interrupted"}))
    else:
        out.update(started=None, interrupted_started_no_response=None, unstarted=None,
                   note="no sidecar: started, interrupted and unstarted requests are not determinable")
    return out


def complete_pass(label, dirs, reqlogs, variant, p):
    seeds = X.VARIANTS[variant]
    d = [x if os.path.isdir(x) else os.path.join(x, "__missing__") for x in dirs]
    rows, checks, inputs = X.load_rep(label, d, [x for x in reqlogs if os.path.exists(x)], [], None, seeds)
    R = [r for (q, pp), r in rows.items() if pp == p]
    ch = [c for c in checks if c[0].startswith(f"{label} pass {p}")]
    fin = {}
    for r in R:
        fin[str(r["finish_reason"])] = fin.get(str(r["finish_reason"]), 0) + 1
    return {"questions": len(R), "empty": sum(r["empty"] for r in R), "raw_correct_flexible": sum(r["correct_flexible"] for r in R),
            "stated_correct_provisional": sum(r["correct_stated"] for r in R), "finish_reasons_logged": fin,
            "logged": sum(r["logged"] for r in R), "checks": [list(c) for c in ch], "pass_checks_ok": all(c[1] for c in ch),
            "samples_sha256": (inputs["samples"].get(f"pass{p}") or {}).get("sha256")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("probe", "report"))
    ap.add_argument("--variant", choices=sorted(X.VARIANTS), required=True)
    ap.add_argument("--rep-pass", nargs=3, required=True)
    ap.add_argument("--rep-reqlog", nargs="*", default=[])
    ap.add_argument("--rep-events", nargs="*", default=[])
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--watchdog-log", default=None)
    ap.add_argument("--artifacts", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    causes, L = causes_of(a)
    if a.mode == "probe":
        complete = not causes and {1, 2} <= {r.get("session") for r in L}
        print(json.dumps({"complete": complete, "causes": causes}))
        return 0 if complete else 10
    seeds = X.VARIANTS[a.variant]
    label = f"0.7.0 K3.25 replication variant {a.variant} (INCOMPLETE)"
    res = {"status": "INCOMPLETE", "variant": a.variant, "seeds": list(seeds), "causes": causes or ["(none detected; report requested)"],
           "rules": ("O1/O2/O3 and the 594-based tests are NOT applied to an incomplete run; interrupted and unstarted requests are "
                     "never counted as empty replies; no automatic restart or replacement run (stop and amend)."),
           "ledger": L, "gpu_hours_ledger": round(sum(r.get("gpu_hours") or 0 for r in L), 3), "passes": {}, "artifacts": []}
    for p, (d, s) in enumerate(zip(a.rep_pass, seeds), 1):
        ok, det = pass_complete(d)
        if ok:
            res["passes"][f"pass{p}"] = {"state": "completed (samples file, 198 questions)", "request_seed": s,
                                         "lm_eval_process_seed": 1234 + p, **complete_pass(label, a.rep_pass, a.rep_reqlog, a.variant, p)}
        else:
            ip = interrupted_pass(s, a.rep_reqlog, a.rep_events)
            state = "not started" if (ip["responses_logged"] == 0 and (ip["started"] in (0, None))) else "interrupted"
            res["passes"][f"pass{p}"] = {"state": state, "samples": det, "lm_eval_process_seed": 1234 + p, **ip}
    for f in a.artifacts:
        for g in sorted(glob.glob(f)):
            if os.path.isfile(g):
                res["artifacts"].append({"path": g, "bytes": os.path.getsize(g), "sha256": fsha(g)})
            elif os.path.isdir(g):
                for h in sorted(glob.glob(os.path.join(g, "**", "*"), recursive=True)):
                    if os.path.isfile(h):
                        res["artifacts"].append({"path": h, "bytes": os.path.getsize(h), "sha256": fsha(h)})
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)
    Lm = [f"# Replication of stock 0.7.0, variant {a.variant}: INCOMPLETE", "", f"- causes: {'; '.join(res['causes'])}",
          f"- {res['rules']}", f"- occupied GPU-hours (ledger): {res['gpu_hours_ledger']}", "", "## Passes"]
    for k, v in res["passes"].items():
        if v["state"].startswith("completed"):
            Lm.append(f"- {k} ({v['state']}; seed {v['request_seed']}): empty {v['empty']} of {v['questions']}, raw correct "
                      f"{v['raw_correct_flexible']}, stated (provisional) {v['stated_correct_provisional']}; finish reasons "
                      f"{v['finish_reasons_logged']}; this pass's checks {'ok' if v['pass_checks_ok'] else 'FAILED'}")
        else:
            Lm.append(f"- {k} ({v['state']}; seed {v['request_seed']}): completed {v['completed_finish_stop']}, cap-reached "
                      f"{v['cap_reached_finish_length_at_327680']} (empty content {v['cap_reached_with_empty_content']}), other "
                      f"{v['other_finish_or_http']}, interrupted {v['interrupted_started_no_response']}, unstarted {v['unstarted']}; "
                      f"empty among completed (descriptive) {v['empty_among_completed_stop_descriptive']}")
    Lm.append(f"\n{len(res['artifacts'])} artifacts preserved and hashed (see json).")
    print("\n".join(Lm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
