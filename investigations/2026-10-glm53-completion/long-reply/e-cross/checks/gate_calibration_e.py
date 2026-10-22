#!/usr/bin/env python3
"""[Re-run under amendment 22ah: KV pool now reported with D's functional floor (scripts/analysis_e.py KVFloor); D2's
session added as a 0.9.1 calibration input (D's was already one); attention block sizes and the longest record prompt
are read from the references.]
E gate calibration BEFORE freezing (PROPOSAL.md section 6.1; the B+ lesson of amendment 22u, d091's
checks/gate_calibration.py). CPU-only container (numpy), read only; run through checks/gate_calibration_e.sh.

Applies the UNCHANGED gate code, analysis_bplus_22v.gates() with E's configuration (scripts/analysis_e.py imported, so the
accepted sets are exactly the frozen ones), to every reference session of each engine that has a server log, and reports
per session and per configuration gate: the value the session produced, the accepted set, pass / fail. Request-level
gates (coverage, seeds, prompt hashes, raw file, order) and the runner's own gates are not configuration gates and are
not calibrated here. Each session is presented to gates() as a run folder holding its real server log (and its docker
inspect record when one was kept) plus one stand-in request spanning the session's real first request start to its last
request end, so the timing and in-window checks see the real request window.

A gate that any reference session fails is stricter than the record's own run-to-run variation, which is not allowed.
Sessions that kept only startup markers (no server log) are checked on the KV pool, the one value they record.
The same sessions are also run against B+'s 22v accepted sets, to show which gate 22v left too strict.
usage (in the container): gate_calibration_e.py /work/checks/gate-calibration.json > checks/gate-calibration.txt
"""
import glob
import json
import os
import re
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("BPLUS_WORK", "/work")
sys.path.insert(0, "/work/scripts")
import analysis_e as AE  # noqa: E402  (applies E's configuration to the unchanged B+ module)

A, M = AE.A, AE.A.M
G, RF, BR, DR, BEN = "/gpqa", "/refs/runF", "/refs/bplus_run", "/refs/d091_run", "/refs/bench"
D2R = "/refs/d091b_run"
CONFIG_PREFIXES = ("server log ", "exactly one engine start", "9 release warm-up", "exactly the release warm-up",
                   "first request sent after", "no illegal-memory", "container image id", "container ports", "docker inspect")


def window_reqlog(path, seeds=None):
    rows = [r for r in M.load_reqlogs([path]) if seeds is None or r.get("seed") in seeds]
    allr = M.load_reqlogs([path])
    return min(r["t_start"] for r in rows), max(r["t_start"] + r["secs"] for r in allr)


def window_client(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r.get("kind") == "request"]
    return min(r["t0"] for r in rows), max(r["t1"] for r in rows)


SESSIONS = [  # engine, name, server log, docker inspect or None, window function
    ("v070", "rep070-F session 1 (2026-10-16; B+'s reference)", f"{RF}/server-s1.log", f"{RF}/inspect-s1.json",
     lambda: window_reqlog(f"{G}/glm53-flash-exl3-k3.25@tpurtell+rep070-F.s1.requests.jsonl")),
    ("v070", "rep070-F session 2 (2026-10-16/17; B+'s reference)", f"{RF}/server-s2.log", f"{RF}/inspect-s2.json",
     lambda: window_reqlog(f"{G}/glm53-flash-exl3-k3.25@tpurtell+rep070-F.s2.requests.jsonl")),
    ("v070", "0.7.0 record passes 2-3 (2026-10-10, +ps)", f"{G}/glm53-flash-exl3-k3.25@tpurtell+ps.server.log", None,
     lambda: window_reqlog(f"{G}/glm53-flash-exl3-k3.25@tpurtell+ps.requests.jsonl")),
    ("v070", "B+ stock 0.7.0 session (2026-10-20)", f"{BR}/v070/server.log", f"{BR}/v070/inspect.json",
     lambda: window_client(f"{BR}/v070/requests.jsonl")),
    ("v091", "0.9.1 record session 1 (2026-10-08, pass 1)", f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1.server.log", None,
     lambda: window_reqlog(f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1.requests.jsonl", {1234})),
    ("v091", "0.9.1 record session 2 (2026-10-09, passes 2-3)", f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.server.log", None,
     lambda: window_reqlog(f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.requests.jsonl", {1235, 1236})),
    ("v091", "B+ stock 0.9.1 session (2026-10-21)", f"{BR}/v091/server.log", f"{BR}/v091/inspect.json",
     lambda: window_client(f"{BR}/v091/requests.jsonl")),
    ("v091", "d091 stock 0.9.1 session (2026-10-21; the prefix source)", f"{DR}/server-s1.log", f"{DR}/inspect-s1.json",
     lambda: window_reqlog(f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+d091.s1.requests.jsonl")),
    ("v091", "d091b (D2) stock 0.9.1 session (2026-10-21; the second prefix source)", f"{D2R}/server-s1.log", f"{D2R}/inspect-s1.json",
     lambda: window_reqlog(f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+d091b.s1.requests.jsonl")),
]
RECORD_REQLOGS = [f"{G}/glm53-flash-exl3-k3.25@tpurtell+rep070-F.s1.requests.jsonl", f"{G}/glm53-flash-exl3-k3.25@tpurtell+rep070-F.s2.requests.jsonl",
                  f"{G}/glm53-flash-exl3-k3.25@tpurtell+ps.requests.jsonl", f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1.requests.jsonl",
                  f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.requests.jsonl"]
IMAGE_DIGEST = {"v070": "48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629",
                "v091": "a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07"}


def run_gates(e, log, insp, win, tag, relabel=True):
    d = f"/tmp/calib/{tag}"
    os.makedirs(d, exist_ok=True)
    for name, src in (("server.log", log), ("inspect.json", insp)):
        p = f"{d}/{name}"
        if os.path.lexists(p):
            os.remove(p)
        if src:
            os.symlink(src, p)
    S = {e: {"session": {"preflight_ok": True, "ready": True}, "start": [], "end": [], "dir": d,
             "req": [{"k": 0, "t0": win[0], "t1": win[1]}]}}
    plan = json.load(open("/work/out/plan.json"))
    man = {m["prefix_id"]: m for m in json.load(open("/work/out/prefix_manifest.json"))["prefixes"]}
    inv = json.load(open("/work/checks/seed-inventory.json"))
    c, _, _ = A.gates(e, S, plan, man, inv)
    if relabel:
        c = AE.relabel_kv(e, c)
    out = []
    for x in c:
        n = x["check"].split(": ", 1)[1]
        if n.startswith(CONFIG_PREFIXES):
            na = insp is None and (n.startswith("container ") or n.startswith("docker inspect"))
            out.append({"gate": n, "pass": None if na else x["ok"], "detail": "no docker inspect record kept for this session" if na else x["detail"]})
    info = M.parse_serverlogs([log], [win], 0)[0]
    vals = {k: info.get(k) for k in ("non_default_args_sha256", "vllm_version", "model_snapshots", "env_names_logged_sha256",
                                     "kv_pool_tokens", "cudagraph_mode", "release_warmups_logged", "server_starts")}
    vals["attention_block_sizes"] = sorted({int(x) for x in re.findall(r"attention block size to (\d+)", open(log, errors="replace").read())})
    return out, vals


def marker_kvs():
    rows = []
    for dd in sorted(glob.glob(f"{BEN}/*k3.25@tpurtell*")):
        name = os.path.basename(dd)
        if re.search(r"k3\.25@tpurtell(\+ps|\+rep070-F)?$", name):
            e = "v070"
        elif re.search(r"k3\.25@tpurtell\+v0\.9\.1(\+ps|\+d091|\+d091b)?$", name):
            e = "v091"
        else:
            continue
        img = open(f"{dd}/image.txt").read() if os.path.exists(f"{dd}/image.txt") else ""
        if IMAGE_DIGEST[e] not in img:
            continue
        seen = set()
        for f in ("startup-markers.txt", "chain.log"):
            if not os.path.exists(f"{dd}/{f}"):
                continue
            for l in open(f"{dd}/{f}", errors="replace"):
                m = re.search(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[kv_cache_utils\.py:\d+\] GPU KV cache size: ([\d,]+) tokens", l)
                if m and m.group(1) not in seen:
                    seen.add(m.group(1))
                    rows.append({"engine": e, "session_dir": name, "start_utc": m.group(1), "kv_pool_tokens": int(m.group(2).replace(",", ""))})
    return rows


def main():
    res = {"what": "E gate calibration against the reference sessions' own run-to-run variation (values and verdicts only)",
           "accepted_sets_E": {"v070": {**{k: A.REF["v070"][k] for k in ("nd", "vllm", "snaps", "env", "cg", "warm", "warm_compl", "image")},
                                        "kv": repr(AE.KV_ACCEPT_E["v070"])},
                               "v091": {**{k: A.REF["v091"][k] for k in ("nd", "vllm", "snaps", "env", "cg", "warm", "warm_compl", "image")},
                                        "kv": repr(AE.KV_ACCEPT_E["v091"])}},
           "kv_floor_inputs": {"cap": AE.CAP, "longest_record_prompt_used": AE.MAX_RECORD_PROMPT, "attention_block": AE.ATTN_BLOCK,
                               "longest_record_prompt_measured": max(r["prompt_tokens"] for r in M.load_reqlogs(RECORD_REQLOGS) if r.get("prompt_tokens") is not None)},
           "sessions": [], "markers_only": [], "verdict": {}}
    kv22v = {"v070": [[2758919], [2768240]]}
    for i, (e, name, log, insp, wf) in enumerate(SESSIONS):
        if not os.path.exists(log):
            res["sessions"].append({"engine": e, "session": name, "error": "server log missing"})
            continue
        insp = insp if insp and os.path.exists(insp) else None
        win = wf()
        A.KV_ACCEPT_22V = AE.KV_ACCEPT_E
        gE, vals = run_gates(e, log, insp, win, f"s{i}")
        A.KV_ACCEPT_22V = kv22v
        g22, _ = run_gates(e, log, insp, win, f"s{i}b", relabel=False)
        A.KV_ACCEPT_22V = AE.KV_ACCEPT_E
        res["sessions"].append({"engine": e, "session": name, "server_log_sha256": M.file_sha(log), "values": vals,
                                "window_utc_epoch": [round(win[0], 1), round(win[1], 1)], "gates_E": gE,
                                "gates_bplus_22v": g22})
    mk = marker_kvs()
    for r in mk:
        r["accepted_E"] = [r["kv_pool_tokens"]] in AE.KV_ACCEPT_E[r["engine"]]
        r["margin_over_floor"] = r["kv_pool_tokens"] - AE.KV_FLOOR[r["engine"]]
        r["accepted_22v"] = [r["kv_pool_tokens"]] in (kv22v.get(r["engine"]) or [A.REF[r["engine"]]["kv"]])
    res["markers_only"] = mk
    gates = {}
    for s in res["sessions"]:
        for g in s.get("gates_E", []):
            gates.setdefault((s["engine"], g["gate"]), []).append((s["session"], g["pass"], g["detail"]))
    allok = True
    for (e, g), rows in sorted(gates.items()):
        ok = all(p is not False for _, p, _ in rows)
        allok &= ok
        res["verdict"][f"{e}: {g}"] = {"all_reference_sessions_pass": ok, "sessions_evaluated": sum(p is not None for _, p, _ in rows),
                                       "not_applicable": sum(p is None for _, p, _ in rows)}
    kv_ok = all(r["accepted_E"] for r in mk)
    res["markers_only_kv_all_accepted_E"] = kv_ok
    res["markers_only_kv_all_accepted_22v"] = all(r["accepted_22v"] for r in mk)
    res["bplus_22v_failures_on_reference_sessions"] = [f"{s['engine']}: {s['session']}: {g['gate']} ({g['detail']})"
                                                       for s in res["sessions"] for g in s.get("gates_bplus_22v", []) if g["pass"] is False] + \
        [f"{r['engine']}: {r['session_dir']} start {r['start_utc']}: KV pool {r['kv_pool_tokens']:,} not in the 22v set" for r in mk if not r["accepted_22v"]]
    blocks_ok = all(s.get("values", {}).get("attention_block_sizes") == [AE.ATTN_BLOCK[s["engine"]]] for s in res["sessions"] if "values" in s)
    res["attention_blocks_match_the_floor_inputs"] = blocks_ok
    res["longest_record_prompt_matches"] = res["kv_floor_inputs"]["longest_record_prompt_measured"] == AE.MAX_RECORD_PROMPT
    res["ALL_CONFIGURATION_GATES_PASS_ON_EVERY_REFERENCE_SESSION"] = allok and kv_ok and blocks_ok and res["longest_record_prompt_matches"]
    json.dump(res, open(sys.argv[1], "w"), indent=1, default=str)
    print("E gate calibration: unchanged analysis_bplus_22v.gates() with E's accepted sets, on every reference session")
    for e in ("v070", "v091"):
        print(f"\n[{e}] accepted sets: " + json.dumps({k: (v if not isinstance(v, str) else v[:16]) for k, v in res["accepted_sets_E"][e].items()}))
        for s in res["sessions"]:
            if s["engine"] != e:
                continue
            v = s.get("values", {})
            npass = sum(g["pass"] is True for g in s.get("gates_E", []))
            nfail = sum(g["pass"] is False for g in s.get("gates_E", []))
            nna = sum(g["pass"] is None for g in s.get("gates_E", []))
            print(f"  {s['session']}: nd {str(v.get('non_default_args_sha256'))[:12]} env {str(v.get('env_names_logged_sha256'))[:12]} "
                  f"kv {v.get('kv_pool_tokens')} block {v.get('attention_block_sizes')} warm {v.get('release_warmups_logged')} starts {v.get('server_starts')} -> "
                  f"{npass} pass, {nfail} fail, {nna} n/a")
            for g in s.get("gates_E", []):
                if g["pass"] is False:
                    print(f"      FAIL {g['gate']} ({g['detail']})")
        for r in mk:
            if r["engine"] == e:
                print(f"  markers only: {r['session_dir']} start {r['start_utc']} UTC: KV pool {r['kv_pool_tokens']:,} "
                      f"-> E floor {'passes' if r['accepted_E'] else 'FAILS'} (margin {r['margin_over_floor']:+,}); 22v set {'accepted' if r['accepted_22v'] else 'rejected'}")
    print("\nPer gate (all reference sessions with a server log):")
    for g, v in res["verdict"].items():
        print(f"  {'OK  ' if v['all_reference_sessions_pass'] else 'FAIL'} {g}: {v['sessions_evaluated']} evaluated, {v['not_applicable']} n/a")
    print("\nB+ 22v sets on the same sessions, failures:", res["bplus_22v_failures_on_reference_sessions"] or "none")
    print(f"KV floor inputs: {res['kv_floor_inputs']}; floors {AE.KV_FLOOR}; attention blocks match: {blocks_ok}; "
          f"longest record prompt matches: {res['longest_record_prompt_matches']}")
    print("ALL CONFIGURATION GATES PASS ON EVERY REFERENCE SESSION:", res["ALL_CONFIGURATION_GATES_PASS_ON_EVERY_REFERENCE_SESSION"])
    return 0 if res["ALL_CONFIGURATION_GATES_PASS_ON_EVERY_REFERENCE_SESSION"] else 1


if __name__ == "__main__":
    sys.exit(main())
