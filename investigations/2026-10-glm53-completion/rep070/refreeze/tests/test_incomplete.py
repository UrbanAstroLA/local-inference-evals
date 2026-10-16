#!/usr/bin/env python3
"""Tests of incomplete_rep070.py (and of the complete/incomplete decision) on synthetic inputs built from the 0.9.1
record relabelled to variant F's seeds (as dry run D3). Interruptions are simulated with a cut-off time T on the proxy
log: rows that ended by T stay in the main log; requests that started by T get a sidecar start event; started but
unfinished requests get an 'interrupted' event (proxy_terminated); later requests never started. Samples files are
linked, not copied; scratch inputs live in WORK and are removed at the end. Cases:
  (a) watchdog during pass 3, passes 1-2 complete;
  (b) session 2 not started;
  (c) pass 2 interrupted with both cap-reached and interrupted requests, pass 3 not started;
  (d) a complete run: the probe says COMPLETE and the frozen analysis gives the same result as dry run D3.
usage: test_incomplete.py WORK_DIR   (exit 0 only if every check passed)"""
import glob, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
INC, ANA = os.path.join(PKG, "incomplete_rep070.py"), os.path.join(PKG, "analysis_rep070.py")
V = "<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1"
SEEDS = {1234: 70701, 1235: 70702, 1236: 70703}
W = sys.argv[1]
RES = []


def ok(name, cond, detail=""):
    RES.append(bool(cond)); print(f"{'PASS' if cond else 'FAIL'} {name}" + (f" ({detail})" if detail else ""), flush=True)


def src_rows():
    r = [json.loads(l) for f in (f"{V}.requests.jsonl", f"{V}+ps.requests.jsonl") for l in open(f)]
    for x in r:
        x["seed"] = SEEDS[x["seed"]]
    return r


def build(case, complete_passes, cut=None, ledger=()):
    """complete_passes: passes with a samples dir; cut = (pass, T) interrupts that pass at time T; passes after it absent."""
    d = os.path.join(W, case); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for p in (1, 2, 3):
        if p in complete_passes:
            pd = os.path.join(d, f"pass{p}", "model"); os.makedirs(pd)
            sf = glob.glob(f"{V}+ps__pass{p}/**/samples_*.jsonl", recursive=True)[0]; rf = glob.glob(f"{V}+ps__pass{p}/**/results_*.json", recursive=True)[0]
            os.symlink(sf, os.path.join(pd, os.path.basename(sf)))
            r = json.load(open(rf)); r["config"]["model_args"]["seed"] = 70700 + p
            json.dump(r, open(os.path.join(pd, os.path.basename(rf)), "w"))
    rows = src_rows()
    for s, sess in ((1, (1,)), (2, (2, 3))):
        main, ev = [], []
        for x in rows:
            p = x["seed"] - 70700
            if p not in sess:
                continue
            if p in complete_passes:
                main.append(x); ev.append({"event": "start", "t": x["t_start"], "messages_sha256": x["messages_sha256"], "seed": x["seed"], "max_tokens": x["max_tokens"]})
            elif cut and p == cut[0]:
                T = cut[1]
                if x["t_start"] <= T:
                    ev.append({"event": "start", "t": x["t_start"], "messages_sha256": x["messages_sha256"], "seed": x["seed"], "max_tokens": x["max_tokens"]})
                    if x["t_start"] + x["secs"] <= T:
                        main.append(x)
                    else:
                        ev.append({"event": "interrupted", "t": T, "t_start": x["t_start"], "secs": round(T - x["t_start"], 3), "error": "proxy_terminated",
                                   "messages_sha256": x["messages_sha256"], "seed": x["seed"]})
        if main or ev:
            with open(os.path.join(d, f"s{s}.requests.jsonl"), "w") as f:
                f.writelines(json.dumps(x) + "\n" for x in sorted(main, key=lambda y: y["t_start"] + y["secs"]))
            with open(os.path.join(d, f"s{s}.requests.events.jsonl"), "w") as f:
                f.writelines(json.dumps(e) + "\n" for e in sorted(ev, key=lambda y: y["t"]))
    with open(os.path.join(d, "costs.jsonl"), "w") as f:
        for L in ledger:
            f.write(json.dumps(L) + "\n")
    return d, rows


def run(d, mode, extra=()):
    args = [sys.executable, INC, mode, "--variant", "F", "--rep-pass"] + [os.path.join(d, f"pass{p}") for p in (1, 2, 3)] + ["--ledger", os.path.join(d, "costs.jsonl")]
    if mode == "report":
        args += ["--rep-reqlog", os.path.join(d, "s1.requests.jsonl"), os.path.join(d, "s2.requests.jsonl"),
                 "--rep-events", os.path.join(d, "s1.requests.events.jsonl"), os.path.join(d, "s2.requests.events.jsonl"),
                 "--artifacts", os.path.join(d, "*.jsonl"), os.path.join(d, "pass*"), "--out", os.path.join(d, "inc.json")]
    p = subprocess.run(args + list(extra), capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def window(rows, p):
    r = [x for x in rows if x["seed"] == 70700 + p]
    return min(x["t_start"] for x in r), max(x["t_start"] + x["secs"] for x in r), r


def consistent(v):
    if v.get("unstarted") is None:
        return False
    return v["completed_finish_stop"] + v["cap_reached_finish_length_at_327680"] + v["other_finish_or_http"] + v["interrupted_started_no_response"] + v["unstarted"] == 198


S1 = {"session": 1, "gpus": 2, "gpu_hours": 5.0, "watchdog_fired": False, "chain_rc": 0}
# (a)
rows = src_rows(); t0, t1, _ = window(rows, 3)
d, _ = build("a", {1, 2}, cut=(3, (t0 + t1) / 2), ledger=(S1, {"session": 2, "gpus": 2, "gpu_hours": 9.0, "watchdog_fired": True, "chain_rc": 143}))
rc, out, err = run(d, "probe")
ok("(a) probe: INCOMPLETE (exit 10), causes watchdog and pass 3", rc == 10 and "watchdog fired" in out and "pass 3 incomplete" in out, out.strip()[:160])
rc, out, err = run(d, "report"); R = json.load(open(os.path.join(d, "inc.json")))
P = R["passes"]
ok("(a) report labelled INCOMPLETE with the watchdog cause", R["status"] == "INCOMPLETE" and any("watchdog" in c for c in R["causes"]))
ok("(a) passes 1-2 reported as completed, results preserved (empties 5 and 5, as the source record's passes)",
   P["pass1"]["state"].startswith("completed") and P["pass2"]["state"].startswith("completed") and (P["pass1"]["empty"], P["pass2"]["empty"]) == (5, 5)
   and P["pass1"]["pass_checks_ok"] and P["pass2"]["pass_checks_ok"], f"{P['pass1']['empty']}, {P['pass2']['empty']}")
ok("(a) pass 3 interrupted: completed + cap + other + interrupted + unstarted = 198; interrupted > 0; unstarted > 0",
   P["pass3"]["state"] == "interrupted" and consistent(P["pass3"]) and P["pass3"]["interrupted_started_no_response"] > 0 and P["pass3"]["unstarted"] > 0,
   {k: P["pass3"][k] for k in ("completed_finish_stop", "cap_reached_finish_length_at_327680", "interrupted_started_no_response", "unstarted")})
ok("(a) no outcome category, no 594-based test, no empty count over the run", not any(k in json.dumps(R) for k in ('"outcome"', '"rep_empties"', "p_one_sided")))
ok("(a) artifacts listed with hashes (logs, sidecars, pass dirs)", len(R["artifacts"]) >= 6)
# (b)
d, _ = build("b", {1}, ledger=(S1,))
rc, out, err = run(d, "probe")
ok("(b) probe: INCOMPLETE, cause session 2 not started", rc == 10 and "session 2 not started" in out)
rc, out, err = run(d, "report"); R = json.load(open(os.path.join(d, "inc.json"))); P = R["passes"]
ok("(b) pass 1 completed (empties 5); passes 2-3 not started with 198 unstarted each, none counted empty",
   P["pass1"]["empty"] == 5 and all(P[k]["state"] == "not started" and P[k]["unstarted"] == 198 and P[k]["responses_logged"] == 0 for k in ("pass2", "pass3")))
# (c)
t0, t1, r2 = window(rows, 2)
caps = sorted(x["t_start"] + x["secs"] for x in r2 if x["finish_reason"] == "length")
T = None
for i in range(1, len(caps)):   # after at least one cap finished, while another cap is still running
    tt = caps[i - 1] + 1
    if any(x["t_start"] <= tt < x["t_start"] + x["secs"] for x in r2):
        T = tt; break
d, _ = build("c", {1}, cut=(2, T), ledger=(S1, {"session": 2, "gpus": 2, "gpu_hours": 4.0, "watchdog_fired": True, "chain_rc": 143}))
rc, out, err = run(d, "report"); R = json.load(open(os.path.join(d, "inc.json"))); P = R["passes"]
v = P["pass2"]
ok("(c) pass 2 interrupted with cap-reached > 0 and interrupted > 0; counts sum to 198",
   v["state"] == "interrupted" and v["cap_reached_finish_length_at_327680"] > 0 and v["interrupted_started_no_response"] > 0 and consistent(v),
   {k: v[k] for k in ("completed_finish_stop", "cap_reached_finish_length_at_327680", "cap_reached_with_empty_content", "interrupted_started_no_response", "unstarted")})
ok("(c) cap-reached counted as cap-reached (empty only by content), interrupted never counted as empty",
   v["cap_reached_with_empty_content"] + v["cap_reached_with_content"] == v["cap_reached_finish_length_at_327680"] and "empty" not in {k for k in v if k.startswith("empty_")} - {"empty_among_completed_stop_descriptive"})
ok("(c) pass 3 not started (198 unstarted); pass 1 completed", P["pass3"]["state"] == "not started" and P["pass3"]["unstarted"] == 198 and P["pass1"]["state"].startswith("completed"))
# (d)
d, _ = build("d", {1, 2, 3}, ledger=(S1, {"session": 2, "gpus": 2, "gpu_hours": 11.0, "watchdog_fired": False, "chain_rc": 0}))
rc, out, err = run(d, "probe")
ok("(d) probe: COMPLETE (exit 0) -> frozen analysis", rc == 0 and '"complete": true' in out, out.strip())
args = [sys.executable, ANA, "--variant", "F", "--rep-label", "0.9.1 record relabelled (fresh seeds 70701/70702/70703, variant F; expect valid, O2)",
        "--rep-pass"] + [os.path.join(d, f"pass{p}") for p in (1, 2, 3)] + ["--rep-reqlog", os.path.join(d, "s1.requests.jsonl"), os.path.join(d, "s2.requests.jsonl"),
        "--rep-serverlog", f"{V}.server.log", f"{V}+ps.server.log", "--require-config-identity", "0",
        "--dry-run-label", "D3-relabel-v091-F fresh seeds 70701/70702/70703, variant F; expect valid, O2", "--out", os.path.join(d, "ana.json")]
p = subprocess.run(args, capture_output=True, text=True)
A = json.load(open(os.path.join(d, "ana.json"))); D3 = json.load(open(os.path.join(PKG, "dryruns", "D3-relabel-v091-F.json")))
strip = lambda x: json.loads(json.dumps({k: v for k, v in x.items() if k not in ("inputs",)}).replace(d, "<W>"))
same = {k: A[k] == D3[k] for k in ("valid", "reading", "C1_rep_vs_v091", "C2_rep_vs_old070", "secondary_long_reply_finishing", "repeatability_descriptive")}
ok("(d) frozen analysis on the complete run: valid, same reading, C1, C2, secondary and descriptives as dry run D3", p.returncode == 0 and all(same.values()), same)
ok("(d) same checks (names and results) as D3", [c[1] for c in A["checks"]] == [c[1] for c in D3["checks"]] and len(A["checks"]) == len(D3["checks"]))
print(f"incomplete-run tests: {sum(RES)} passed, {len(RES) - sum(RES)} failed")
shutil.rmtree(W, ignore_errors=True)
sys.exit(0 if all(RES) else 1)
