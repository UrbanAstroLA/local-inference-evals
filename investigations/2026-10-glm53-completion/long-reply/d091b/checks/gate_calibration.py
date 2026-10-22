#!/usr/bin/env python3
"""Gate calibration BEFORE freezing (PROPOSAL.md section 4.3; the B+ lesson of amendment 22u). CPU only, stdlib, read only.
Applies the exact gate functions of analysis_d091.py to the reference sessions and reports, per gate: the value each
reference session produced, the accepted set, and whether every reference session passes. A gate that a reference session
of the 0.9.1 record fails would be stricter than the record's own run-to-run variation and is not allowed.

Reference sessions: the 0.9.1 record's two sessions (2026-10-08 pass 1; 2026-10-09 passes 2-3), plus, as extra evidence
of the same image and configuration, B+'s stock 0.9.1 session (2026-10-21, loopback publish, /v1/completions client).
d091b addition: option D's own stock 0.9.1 GPQA session (2026-10-21, d091/run/server-s1.log, request log +d091.s1, seed
70704; the session this package copies), with its request window; every gate must also pass on it.
Precedent for the KV pool: the stock 0.7.0 family's sessions produced three different pool sizes at the same
configuration (startup markers of the K3.25 0.7.0 sessions), which is why the pool is reported, not identity-gated.
usage: gate_calibration.py OUT.json > OUT.txt
"""
import glob, json, os, re, sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import analysis_d091 as D  # noqa: E402

M, X = D.M, D.X
G = "<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1"
BP = "<runs>/long-reply/bplus/run/v091"
D091 = "<runs>/long-reply/d091/run/server-s1.log"   # d091b addition
D091_REQ = "<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+d091.s1.requests.jsonl"


def first_t(reqlog, seeds):
    rows = [r for r in M.load_reqlogs([reqlog]) if r.get("seed") in seeds or seeds is None]
    return min(r["t_start"] for r in rows)


def first_t_bplus():
    rows = [json.loads(l) for l in open(f"{BP}/requests.jsonl") if l.strip()]
    return min(r["t0"] for r in rows if "t0" in r)


def main():
    refs = [("0.9.1 record session 1 (2026-10-08, pass 1)", f"{G}.server.log", first_t(f"{G}.requests.jsonl", {1234}),
             (first_t(f"{G}.requests.jsonl", {1234}), max(r["t_start"] + r["secs"] for r in M.load_reqlogs([f"{G}.requests.jsonl"])))),
            ("0.9.1 record session 2 (2026-10-09, passes 2-3)", f"{G}+ps.server.log", first_t(f"{G}+ps.requests.jsonl", {1235, 1236}),
             (first_t(f"{G}+ps.requests.jsonl", {1235, 1236}), max(r["t_start"] + r["secs"] for r in M.load_reqlogs([f"{G}+ps.requests.jsonl"])))),
            ("B+ stock 0.9.1 session (2026-10-21; extra evidence)", f"{BP}/server.log", first_t_bplus(), None),
            ("D d091 stock 0.9.1 GPQA session (2026-10-21, seed 70704; d091b addition)", D091, first_t(D091_REQ, {70704}),
             (first_t(D091_REQ, {70704}), max(r["t_start"] + r["secs"] for r in M.load_reqlogs([D091_REQ]))))]
    rec_logs = D.REC["serverlogs"]
    acc, info = D.reference_values(rec_logs)
    floor = D.kv_floor(D.max_prompt_tokens(D.REC["reqlogs"]))
    out = {"accepted_sets_from_the_record": acc, "kv_floor": floor, "sessions": [], "gates": {}}
    for name, log, t1, win in refs:
        s = M.parse_serverlogs([log], [win] if win else [], 0)[0]
        cg, kv = D.config_gates(log, rec_logs, "session", floor)
        wu = X.warmup_checks(log, t1, "session warm-up")
        hg = D.health_gates(log, win, "session")[0] if win else []
        rec = {"session": name, "log_sha256": s["sha256"], "values": {k: s.get(k) for k in D.IDENTITY_KEYS + ("release_warmups_logged", "kv_pool_tokens")},
               "in_window": s.get("in_window"), "checks": [[n, ok, str(d)] for n, ok, d in cg + wu + hg], "kv": kv}
        out["sessions"].append(rec)
        for n, ok, d in cg + wu + hg:
            out["gates"].setdefault(n.replace("session warm-up: ", "warm-up: ").replace("session: ", ""), []).append({"session": name, "pass": ok, "detail": str(d)})
    # client-side and request-level gates against the record's own passes
    cfgs = [D.results_cfg(d) for d in D.REC["passes"]]
    strip = lambda ma: {k: v for k, v in (ma or {}).items() if k != "seed"}
    out["client_config_across_record_passes"] = {"model_args_without_seed_identical": all(strip(c["model_args"]) == strip(cfgs[0]["model_args"]) for c in cfgs),
                                                 "gen_kwargs_identical": all(c["gen_kwargs"] == cfgs[0]["gen_kwargs"] for c in cfgs),
                                                 "process_seeds": [c["random_seed"] for c in cfgs], "model_args_seeds": [c["model_args"].get("seed") for c in cfgs]}
    V, cv, _ = M.load_arm(D.REC["label"], D.REC["passes"], D.REC["reqlogs"], D.REC["hedged"], D.REC["published"])
    same = {f: all(V[(q, 1)][f] == V[(q, p)][f] for q in range(M.N_Q) for p in (2, 3)) for f in ("prompt_hash", "doc_hash", "target_hash", "messages_sha256", "prompt_tokens")}
    out["per_question_fields_identical_across_record_passes"] = same
    out["record_loader_checks_all_pass"] = all(ok for _, ok, _ in cv)
    out["task_hash_all_record_passes"] = [X.task_hash_of(d) == X.TASK_HASH for d in D.REC["passes"]]
    out["record_max_request_s_vs_lm_eval_timeout"] = [max(r["secs"] for r in M.load_reqlogs([l])) for l in D.REC["reqlogs"]]
    # KV pool precedent: every K3.25 session's startup markers, by engine family
    kvs = {}
    for f in sorted(glob.glob("<data-runs>/benchmarks/*k3.25@tpurtell*/startup-markers.txt")):
        name = os.path.basename(os.path.dirname(f))
        fam = ("0.9.x EP2/DCP2 layout (+ep2dcp2)" if "+ep2dcp2" in name else "0.9.x default DCP1 layout (the d091 configuration)") if "+v0.9" in name \
            else ("0.7.0 (DCP2 / EP2 layout)" if re.search(r"k3\.25@tpurtell(\+ps|\+rep070-F|\+kpoolring)?$", name) else "other (0.6.x / 0.8.0 variants)")
        for m in re.findall(r"GPU KV cache size: ([\d,]+) tokens", open(f, errors="replace").read()):
            kvs.setdefault(fam, []).append((name, int(m.replace(",", ""))))
    out["kv_pool_by_family_startup_markers"] = {k: {"sessions": v, "distinct_sizes": sorted({x for _, x in v})} for k, v in kvs.items()}
    verdict = {}
    for g, rows in out["gates"].items():
        rec_rows = [r for r in rows if r["session"].startswith("0.9.1 record")]
        verdict[g] = {"record_sessions_pass": all(r["pass"] for r in rec_rows), "bplus_v091_passes": all(r["pass"] for r in rows if r["session"].startswith("B+")),
                    "d091_session_passes": all(r["pass"] for r in rows if r["session"].startswith("D d091")) and any(r["session"].startswith("D d091") for r in rows)}
    out["verdict"] = verdict
    out["all_gates_pass_on_both_record_sessions"] = all(v["record_sessions_pass"] for v in verdict.values())
    out["all_gates_pass_on_d091_session"] = all(v["d091_session_passes"] for v in verdict.values())   # d091b addition
    json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
    print("Accepted sets (values the 0.9.1 record's two sessions produced):")
    for k, v in acc.items():
        print(f"  {k}: {[str(x)[:24] for x in v]}")
    print(f"KV floor {floor:,} tokens (8 x 4352 x ceil((327,680 + longest record prompt {D.max_prompt_tokens(D.REC['reqlogs'])} + 16) / 4352))")
    for g, v in verdict.items():
        print(f"  {'OK  ' if v['record_sessions_pass'] else 'FAIL'} {g}: record sessions pass {v['record_sessions_pass']}; B+ v091 {v['bplus_v091_passes']}; D d091 {v['d091_session_passes']}")
    print("client config across record passes:", out["client_config_across_record_passes"])
    print("per-question fields identical across record passes:", same, "; record loader checks pass:", out["record_loader_checks_all_pass"])
    print("task hash in all record passes:", out["task_hash_all_record_passes"], "; record max request s:", out["record_max_request_s_vs_lm_eval_timeout"])
    for k, v in out["kv_pool_by_family_startup_markers"].items():
        print(f"KV pool sizes, {k}: {len(v['sessions'])} sessions, distinct {v['distinct_sizes']}")
    print("ALL GATES PASS ON BOTH RECORD SESSIONS:", out["all_gates_pass_on_both_record_sessions"])
    print("ALL GATES PASS ON D's d091 SESSION:", out["all_gates_pass_on_d091_session"])
    return 0 if out["all_gates_pass_on_both_record_sessions"] and out["all_gates_pass_on_d091_session"] else 1


if __name__ == "__main__":
    sys.exit(main())
