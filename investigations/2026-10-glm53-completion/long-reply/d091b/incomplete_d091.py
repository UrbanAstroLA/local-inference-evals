#!/usr/bin/env python3
"""Incomplete-run handling for option D (PROPOSAL.md section 5). CPU only, standard library; ids, hashes, counts and token
lengths only. Reuses incomplete_rep070.py (byte-identical copy, sha256 5a682182...) for the pass-completeness test, the
ledger reader and the per-seed interrupted-pass counter (proxy main log + start/interrupted sidecar).

  incomplete_d091.py probe  --pass-dir D --ledger costs.jsonl [--watchdog-log F]
      exit 0: COMPLETE (session 1 in the ledger, watchdog not fired, the pass has exactly one samples file with 198
              questions and both filters) -> the runner calls the frozen analysis (analysis_d091.py);
      exit 10: INCOMPLETE -> the runner calls 'report'; the causes are printed as JSON.
  incomplete_d091.py report --pass-dir D --reqlog L --events E --ledger costs.jsonl [--watchdog-log F] [--artifacts P ...]
      --out OUT.json > OUT.md

An incomplete run is labelled INCOMPLETE with its cause(s). It is never classified: no GPQA result rows, no accuracy or
empty count is presented as the pass's result, the long-reply pipeline is not run, and interrupted or unstarted requests
are never counted as empty replies. The report counts, from the proxy main log and its sidecar: completed (finish 'stop'),
cap-reached (finish 'length' at 327,680), other finishes, interrupted (started, no response), unstarted (198 minus
started). No automatic restart or replacement run: stop and amend.
"""
import argparse, glob, hashlib, json, os, sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PIN = "5a6821826af8ad11225fce7be39557dc30b46fa2033d4a84fb62a7397d4b471c"
if hashlib.sha256(open(os.path.join(HERE, "incomplete_rep070.py"), "rb").read()).hexdigest() != PIN:
    sys.exit("incomplete_rep070.py is not the frozen copy")
sys.path.insert(0, HERE)
import incomplete_rep070 as I  # noqa: E402  (imports analysis_rep070, which checks the frozen module's hash)

SEED = 70705


def causes_of(a):
    c = []
    L = I.ledger(a.ledger)
    if not L:
        c.append("no cost ledger (the session did not complete its bookkeeping)")
    wd = any(r.get("watchdog_fired") is True for r in L)
    if a.watchdog_log and os.path.exists(a.watchdog_log) and "CEILING REACHED" in open(a.watchdog_log, errors="replace").read():
        wd = True
    if wd:
        c.append("watchdog fired (approved GPU-hour ceiling reached)")
    if L and not any(r.get("session") == 1 for r in L):
        c.append("session 1 missing from the ledger")
    if any(r.get("chain_rc") not in (0, None) for r in L):
        c.append(f"eval chain exit code {[r.get('chain_rc') for r in L]}")
    if any(r.get("chain_rc") is None for r in L):
        c.append("eval chain interrupted (no exit code)")
    ok, det = I.pass_complete(a.pass_dir)
    if not ok:
        c.append(f"pass 1 incomplete or missing ({det})")
    return c, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("probe", "report"))
    ap.add_argument("--pass-dir", required=True)
    ap.add_argument("--reqlog", default=None)
    ap.add_argument("--events", default=None)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--watchdog-log", default=None)
    ap.add_argument("--artifacts", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    causes, L = causes_of(a)
    if a.mode == "probe":
        complete = not causes and any(r.get("session") == 1 for r in L)
        print(json.dumps({"complete": complete, "causes": causes}))
        return 0 if complete else 10
    ip = I.interrupted_pass(SEED, [a.reqlog] if a.reqlog else [], [a.events] if a.events else [])
    ok, det = I.pass_complete(a.pass_dir)
    state = "samples file complete (run incomplete for another cause)" if ok else (
        "not started" if (ip["responses_logged"] == 0 and ip["started"] in (0, None)) else "interrupted")
    res = {"status": "INCOMPLETE", "seed": SEED, "causes": causes or ["(none detected; report requested)"],
           "rules": ("An incomplete run is never classified: no result rows, no accuracy or empty count as the pass's result, no "
                     "long-reply pipeline; interrupted and unstarted requests are never counted as empty replies; no automatic "
                     "restart or replacement run (stop and amend)."),
           "ledger": L, "gpu_hours_ledger": round(sum(r.get("gpu_hours") or 0 for r in L), 3),
           "pass1": {"state": state, "samples": det, "lm_eval_process_seed": 1235, **ip}, "artifacts": []}
    for f in a.artifacts:
        for g in sorted(glob.glob(f)):
            if os.path.isfile(g):
                res["artifacts"].append({"path": g, "bytes": os.path.getsize(g), "sha256": I.fsha(g)})
            elif os.path.isdir(g):
                for h in sorted(glob.glob(os.path.join(g, "**", "*"), recursive=True)):
                    if os.path.isfile(h):
                        res["artifacts"].append({"path": h, "bytes": os.path.getsize(h), "sha256": I.fsha(h)})
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)
    v = res["pass1"]
    Lm = ["# Option D (stock 0.9.1, seed 70705): INCOMPLETE", "", f"- causes: {'; '.join(res['causes'])}", f"- {res['rules']}",
          f"- occupied GPU-hours (ledger): {res['gpu_hours_ledger']}", "",
          f"- pass 1 ({v['state']}): responses logged {v['responses_logged']} (finish stop {v['completed_finish_stop']}, cap-reached "
          f"{v['cap_reached_finish_length_at_327680']}, other {v['other_finish_or_http']}); started {v['started']}, interrupted "
          f"{v['interrupted_started_no_response']}, unstarted {v['unstarted']}",
          f"\n{len(res['artifacts'])} artifacts preserved and hashed (see json)."]
    print("\n".join(Lm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
