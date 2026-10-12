#!/usr/bin/env python3
"""Bundle gate analysis (PLAN.md amendment 14 (3)); criteria fixed before any gate data. CPU only.
usage: analyze_bundle_gate.py RUN_DIR MODE BUNDLE      (MODE s1|s2|s3; BUNDLE d1234|d124)
Writes RUN_DIR/verdict.json and prints the report. Session-level verdicts are combined by gate_verdict.py.

Reused instruments (copies in tools/, unchanged): analyze_gate.py (col511 checker, gate 2026-10-11), summarize_tg.py and
throughput_tg.py (tail guard / tail fix runs), config_compare.py and analyze_d5.py (verify13 Stage A). D3 readout:
d3_logits_check.py (runs in the bundle check image, CPU, --network none).
"""
import glob
import json
import os
import re
import statistics
import subprocess
import sys

RUN, MODE, BUNDLE = sys.argv[1], sys.argv[2], sys.argv[3]
G = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(G, "tools")
HIST_LOG = "<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.server.log"
KV_POOL_HIST = 4707515                       # tokens; historical 0.9.1 record and col511 arm (result-col511-final.md)
REF_TF = {"tailguard graph-a1 (release + tail guard)": (88.3, 59.3), "tailfix U-graph-a1 (D2 U + tail guard)": (88.6, 59.7)}
WITH_D3 = BUNDLE == "d1234"
V = {"run": RUN, "mode": MODE, "bundle": BUNDLE, "checks": {}, "info": {}}


def ck(name, ok, detail=""):
    V["checks"][name] = {"ok": bool(ok), "detail": detail}
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")


def sh(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def jl(path):
    out = []
    for f in sorted(glob.glob(path)):
        for line in open(f, errors="replace"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


# ------------------------------------------------------------------ validity (all sessions)
reqs = [d for d in jl(f"{RUN}/client.jsonl") if d.get("kind") == "request"]
phases = [d for d in jl(f"{RUN}/client.jsonl") if d.get("kind") == "phase"]
clog = open(f"{RUN}/client.log", errors="replace").read() if os.path.exists(f"{RUN}/client.log") else ""
expected_phases = {"s1": ["warm", "T-", "F-", "D-", "S-", "Q", "Qreuse", "G", "P"], "s2": ["warm", "T-", "F-", "G", "P"],
                   "s3": ["P"], "rel": ["P"]}[MODE]
seen = {p["tag"] for p in phases}
ck("V1 client completed every planned phase", "CLIENT DONE" in clog and all(any(t.startswith(e) for t in seen) for e in expected_phases),
   f"phases seen {sorted(seen)[:12]}{'...' if len(seen) > 12 else ''}")
errs = [r for r in reqs if r.get("error")]
ck("V2 zero request errors", not errs, f"{len(errs)} of {len(reqs)}")
fixed = [r for r in reqs if re.match(r"^(warm|T-|F-|D-|S-)", r.get("phase", ""))]
bad_fixed = [r for r in fixed if r.get("finish") != "length"]
ck("V3 fixed-length requests finish by length", not bad_fixed, f"{len(bad_fixed)} of {len(fixed)} not 'length'")
free = [r for r in reqs if r.get("phase") in ("Q", "Qreuse", "G", "P1", "P2")]
odd = [r for r in free if r.get("finish") not in ("stop", "length")]
prem = [r for r in free if r.get("finish") == "stop" and (r.get("completion_tokens") or 0) == 0]
ck("V4 no stopping errors (finish not stop/length, or stop with zero tokens)", not odd and not prem,
   f"odd finishes {len(odd)}, zero-token stops {len(prem)}, of {len(free)} free-running requests")
V["info"]["empty_replies_short_context"] = sum(1 for r in free if r.get("phase") in ("Q", "Qreuse", "G") and r.get("empty_reply"))
slog = open(f"{RUN}/server.log", errors="replace").read() if os.path.exists(f"{RUN}/server.log") else ""
bad = {k: len(re.findall(p, slog, re.I | re.M)) for k, p in {
    "traceback": r"^.*Traceback \(most recent call last\)", "illegal_memory_access": r"illegal memory access",
    "cuda_error": r"CUDA error", "oom": r"out of memory", "workspace_locked": r"Workspace is locked",
    "assertion": r"AssertionError", "preemption": r"preempt", "nan_warn": r"^(?=.*\bnan\b)(?=.*(WARN|ERROR)).*$",
    "engine_dead": r"EngineDeadError|Engine core .* died"}.items()}
ck("V5 server log free of execution, memory and engine errors", not any(bad.values()), json.dumps(bad))
starts = slog.count("non-default args:")
ck("V6 one server start (no restart)", starts == 1, f"non-default args lines {starts}")
rc, cc = sh([sys.executable, f"{T}/config_compare.py", f"{RUN}/server.log", HIST_LOG])
open(f"{RUN}/config-compare-historical.txt", "w").write(cc)
ident = re.findall(r"identical (?:keys|fields): (\d+) of (\d+)", cc)
graphs = re.findall(r"- (Capturing (?:dflash2 )?CUDA graphs \((?:PIECEWISE|FULL)\)): gate completed \[(\d*)\] \| hist completed \[(\d*)\]", cc)
env_only = re.findall(r"only gate: (\[.*\])", cc)
ok_cfg = bool(ident) and all(a == b for a, b in ident) and graphs and all(g == h and g for _, g, h in graphs)
ck("V7 effective configuration = historical record (args, engine config, graph captures 11/9/9)", ok_cfg,
   f"identical {ident}; graphs {[(n, g, h) for n, g, h in graphs]}; VLLM env only here {env_only[:1]}")
kv = [int(x.replace(",", "")) for x in re.findall(r"GPU KV cache size: ([\d,]+) tokens", slog)]
V["info"]["kv_pool_tokens"] = kv
if MODE == "s2":
    ck("V8 production image: KV pool equals the historical record (memory footprint unchanged)",
       kv and all(abs(k - KV_POOL_HIST) <= 0.005 * KV_POOL_HIST for k in kv), f"{kv} vs {KV_POOL_HIST}")
else:
    print(f"[info] KV pool {kv} (historical {KV_POOL_HIST}; observers may reduce it in s1)")
ident_txt = open(f"{RUN}/container-files-env.txt", errors="replace").read() if os.path.exists(f"{RUN}/container-files-env.txt") else ""
want = {"release": {"sparse_attn_indexer_kpool.py": "e7935f2a", "indexer.py": "67449677", "mamba_hybrid.py": "82968af5",
                    "exl3.py": "3b4ca47c", "gate_linear.py": "6fca8c7f"},
        "d1234": {"sparse_attn_indexer_kpool.py": "9104be37" if MODE == "s2" else "fae1127c", "indexer.py": "00989c50",
                  "mamba_hybrid.py": "049238f9", "exl3.py": "b40344e2", "gate_linear.py": "2499ee67" if MODE == "s2" else "02ba8104"},
        "d124": {"sparse_attn_indexer_kpool.py": "9104be37" if MODE == "s2" else "fae1127c", "indexer.py": "00989c50",
                 "mamba_hybrid.py": "049238f9", "exl3.py": "b40344e2", "gate_linear.py": "6fca8c7f" if MODE == "s2" else "48f4c035"}}[BUNDLE]
got = {m.group(2).split("/")[-1]: m.group(1)[:8] for m in re.finditer(r"^([0-9a-f]{64})\s+(\S+)$", ident_txt, re.M)}
ck("V9 served engine files = the frozen bundle files (sha256 prefixes)", all(got.get(k) == v for k, v in want.items()),
   json.dumps({k: (got.get(k), v) for k, v in want.items()}))

# ------------------------------------------------------------------ throughput (s1 lower bound, s2 clean)
if MODE in ("s1", "s2"):
    rc, tp = sh([sys.executable, f"{T}/throughput_tg.py", RUN])
    open(f"{RUN}/throughput.txt", "w").write(tp)
    try:
        tj = json.loads(tp.split(" ", 1)[1])
        t8, f4 = tj["T"]["agg_tok_s_median"], tj["F"]["agg_tok_s_median"]
    except Exception:  # noqa: BLE001
        t8 = f4 = None
    ratios = {k: (round(t8 / a, 3) if t8 else None, round(f4 / b, 3) if f4 else None) for k, (a, b) in REF_TF.items()}
    V["info"]["throughput_T8_F4"] = (t8, f4)
    V["info"]["throughput_ratio_vs_reference"] = ratios
    gp = [p for p in phases if p["tag"] == "G"]
    g_tok = sum(r.get("completion_tokens") or 0 for r in reqs if r.get("phase") == "G")
    V["info"]["G_aggregate_tok_s"] = round(g_tok / gp[0]["secs"], 1) if gp and gp[0]["secs"] else None
    st8 = [float(m.group(1)) for m in re.finditer(r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs", slog)
           if int(m.group(2)) >= 8]
    V["info"]["serverlog_mean_gen_tok_s_8plus_running"] = round(statistics.mean(st8), 1) if st8 else None
    print(f"[info] T(8)/F(4) median {t8} / {f4} tok/s; ratio vs references {ratios}; G aggregate {V['info']['G_aggregate_tok_s']} "
          f"tok/s; server-log mean at >=8 running {V['info']['serverlog_mean_gen_tok_s_8plus_running']} "
          f"(GPQA history: record 459.4 / 447.7, col511 arm 444.5 / 455.0)")
    lo = min(r for pair in ratios.values() for r in pair if r) if t8 and f4 else None
    if MODE == "s1":
        V["info"]["s2_needed_for_pricing"] = (lo is None) or lo < 0.95
        print(f"[info] observers only add work, so s1 is a lower bound; s2 needed for pricing: {V['info']['s2_needed_for_pricing']}")
    else:
        ck("V10 production throughput within 15% of the release reference (else stop and report before the screen)",
           lo is not None and lo >= 0.85, f"min ratio {lo}")

# ------------------------------------------------------------------ s1: activation of each correction
if MODE == "s1":
    # D1 column-511 checker (analyze_gate.py; same criteria as the 2026-10-11 gate)
    os.makedirs(f"{RUN}/col511check", exist_ok=True)
    rc, rep = sh([sys.executable, f"{T}/analyze_gate.py", RUN])
    open(f"{RUN}/col511-report.txt", "w").write(rep)
    try:
        gs = json.load(open(f"{RUN}/gate-summary.json"))
        tot = gs.get("totals", {})
        modes = {k.split("|")[1] for k in tot}
        branches = {k.split("|")[0] for k in tot if tot[k].get("calls")}
        bad1 = {k: {m: v[m] for m in ("post_not_lowest", "post_set_changed", "consumer_mismatch", "post_unfilled_valid_in_col511",
                                      "rows_id_out_of_range", "rows_dup", "post_score_lookup_mismatch") if v.get(m)}
                for k, v in tot.items()}
        bad1 = {k: v for k, v in bad1.items() if v and not k.endswith("|capture_phase")}
        V["info"]["D1_capture_phase_counts"] = {k: v for k, v in tot.items() if k.endswith("|capture_phase")}
        # capture_phase = checker calls on the dummy batches of graph capture (no request data); reported, not judged,
        # as in the 2026-10-11 gate, whose PASS covered serving rows (eager, piecewise and FULL replays)
        fp64 = {k: (v.get("ind_post_ok_tol"), v.get("ind_rows")) for k, v in tot.items() if not k.endswith("|capture_phase")}
        ranks = gs.get("ranks", {})
        rec_ok = all(all(x.get("match") for x in (r.get("full_reconciliation") or {}).values()) for r in ranks.values())
        errs1 = sum((r.get("checker_errors") or 0) + (r.get("fallback_calls") or 0) for r in ranks.values())
        ck("D1 column-511: lowest selected pool in column 511 on every checked row, set unchanged, consumers read it",
           not bad1 and all(a == b for a, b in fp64.values()), f"violations {bad1}; fp64 post-ok/rows {fp64}")
        ck("D1 coverage: decode fused + decode tiled + prefill observed, incl. FULL-graph replays; FULL reconciliation; "
           "0 checker errors / eager fallbacks",
           {"decode_fused", "decode_tiled", "prefill"} <= branches and "full" in modes and rec_ok and errs1 == 0 and len(ranks) >= 2,
           f"branches {sorted(branches)}, modes {sorted(modes)}, reconciliation {rec_ok}, errors+fallbacks {errs1}, ranks {sorted(ranks)}")
        V["info"]["D1_nan_index_rows"] = {k: v.get("ind_nan_rows") for k, v in tot.items()}
        V["info"]["D1_unfilled_rows"] = {k: v.get("rows_unfilled") for k, v in tot.items()}
        print(f"[info] corrupted/NaN index rows (reported separately; expected ~0 with D2, inferred): {V['info']['D1_nan_index_rows']}")
    except Exception as exc:  # noqa: BLE001
        ck("D1 column-511 checker output readable", False, repr(exc))
    # D2 tail guard (summarize_tg.py; same criteria as tailfix U-graph-a1)
    rc, tg = sh([sys.executable, f"{T}/summarize_tg.py", RUN])
    open(f"{RUN}/tailguard-summary.txt", "w").write(tg)
    try:
        ts = json.load(open(f"{RUN}/tg_summary.json"))
        rk = {k: v for k, v in ts.items() if k in ("0", "1")}
        okr = {k: (v.get("tail_slot_source") == "compute_kpool_tail_slot_mapping" and v.get("counters")
                   and v["counters"].get("writes", 0) > 0
                   and all(v["counters"].get(m, 1) == 0 for m in ("null_alias", "foreign", "attn_collision", "tail_collision",
                                                                  "shadow_divergence"))
                   and not v.get("guard_errors")) for k, v in rk.items()}
        ck("D2 tail ring: circular mapping used; 0 null-block, foreign, attention-block, tail collisions, divergence, guard errors",
           len(okr) == 2 and all(okr.values()),
           json.dumps({k: (v.get("tail_slot_source"), {m: (v.get("counters") or {}).get(m) for m in
                                                       ("writes", "null_alias", "foreign", "attn_collision", "tail_collision",
                                                        "shadow_divergence", "stale_col_rows")}) for k, v in rk.items()}))
    except Exception as exc:  # noqa: BLE001
        ck("D2 tail guard output readable", False, repr(exc))
    # D3 dispatch + served logits (d3_logits_check.py inside the check image)
    img = f"local/glm53-v0.9.1-bundle-{BUNDLE}-check:local"
    rc, d3 = sh(["docker", "run", "--rm", "--network", "none", "-e", "CUDA_VISIBLE_DEVICES=", "--user", f"{os.getuid()}:{os.getgid()}",
                 "-e", "HOME=/tmp", "-v", f"{G}:/g:ro", "-v", f"{RUN}:/run:ro", "--entrypoint", "python3", img, "/g/d3_logits_check.py", "/run"])
    try:
        d3j = json.loads(d3[d3.index("{"):])
        open(f"{RUN}/d3_readout.json", "w").write(json.dumps(d3j, indent=1))
        tiers = d3j["tiers_by_rank"]
        want_tier = "5 torch.mm(out_dtype=fp32)" if WITH_D3 else "6 F.linear(bf16) then .to(out_dtype)"
        tier_ok = set(tiers) == {"0", "1"} and all(v == [want_tier] for v in tiers.values()) and all(n == 42 for n in d3j["layers_by_rank"].values())
        if WITH_D3:
            val_ok = d3j["rows"] >= 1000 and d3j["share_rows_bf16_valued"] is not None and d3j["share_rows_bf16_valued"] <= 0.001
        else:
            val_ok = d3j["rows"] >= 1000 and d3j["share_rows_bf16_valued"] == 1.0
        ck(f"D3 router: every layer on both ranks dispatches tier {want_tier.split()[0]}; served logits "
           f"{'not bf16-valued (<= 0.1% of rows)' if WITH_D3 else 'bf16-valued (D3 absent: control)'}",
           tier_ok and val_ok, f"tiers {tiers}; rows {d3j['rows']}, bf16-valued share {d3j['share_rows_bf16_valued']}; "
                               f"by batch kind {d3j['by_batch_kind']}")
    except Exception as exc:  # noqa: BLE001
        ck("D3 readout produced", False, f"{exc!r} {d3[-400:]}")
    # D4 plans and compiled kernels carry the limit
    ev = jl(f"{RUN}/verify13/v13_p*.jsonl")
    ei = [d for d in ev if d.get("kind") == "experts_info"]
    plans = [(d.get("rank"), p) for d in ei for p in (d.get("runtime_plans") or [])]
    lim_ok = bool(plans) and all(p.get(k) == 10.0 for _, p in plans for k in ("caps_swiglu_limit", "launch_plan_swiglu_limit", "core_swiglu_limit"))
    layer_ok = bool(ei) and all(d.get("swiglu_limit_on_layer") == 10.0 for d in ei)
    path_ok = bool(ei) and all(d.get("projection_mixed") is True and d.get("quant_method") == "Exl3MoEMethod" for d in ei)
    pl = [d for d in jl(f"{RUN}/d4limit/*.jsonl") if d.get("kind") == "plan_limits"]
    kern_ok = bool(pl) and all(set(d.get("values") or []) <= {"10.0", "True"} and "True" in (d.get("values") or []) for d in pl)
    ck("D4 plans: layer limit 10, every B12x plan (caps / launch plan / core) carries 10.0 on both ranks; projection-mixed EXL3 path",
       lim_ok and layer_ok and path_ok and {r for r, _ in plans} >= {0, 1},
       f"plans {[(r, p.get('plan'), p.get('caps_swiglu_limit'), p.get('launch_plan_swiglu_limit'), p.get('core_swiglu_limit')) for r, p in plans][:6]}")
    ck("D4 compiled kernels: plan-object walk finds has_swiglu_limit True and only the value 10.0 (both ranks)",
       kern_ok and len({d.get('pid') for d in pl}) >= 2, f"{[(d.get('method'), d.get('n_hits'), d.get('values')) for d in pl][:4]}")
    # D5 excluded: observation only
    rc, d5 = sh([sys.executable, f"{T}/analyze_d5.py", RUN])
    open(f"{RUN}/d5_summary.json", "w").write(d5)
    try:
        V["info"]["D5_classification"] = json.loads(d5)["classification"]["exercised"]
    except Exception:  # noqa: BLE001
        V["info"]["D5_classification"] = "unreadable"
    print(f"[info] D5 (excluded) workspace resizes after capture: {V['info']['D5_classification']}")
    # verify13 hook health (observe-only capture)
    summ = [d for d in ev if d.get("kind") == "summary"]
    herr = sum(len(d.get("errors") or []) for d in summ)
    modes_ok = all(d.get("mode") in (None, "NONE") for d in ev if d.get("kind") == "record")
    ck("O1 verify13 capture healthy (installed both ranks, records only from eager forwards, no hook errors)",
       len({d.get('rank') for d in summ}) >= 2 and modes_ok and herr == 0, f"summaries {len(summ)}, hook errors {herr}")

# ------------------------------------------------------------------ D4 probe (s1 / s2 / s3)
pr = jl(f"{RUN}/probe.jsonl")
if pr:
    by = {}
    for d in pr:
        by.setdefault(d["id"], {})[d["round"]] = d
    rep_ok = all(len(v) == 2 and v[1]["sha256"] == v[2]["sha256"] for v in by.values())
    V["info"]["probe"] = {k: {r: (v[r]["sha256"][:12], v[r]["n_tokens"]) for r in v} for k, v in by.items()}
    ck("P1 D4 probe repeatable within the session (both rounds identical on all 6 prompts)", rep_ok and len(by) == 6,
       json.dumps(V["info"]["probe"]))
if MODE == "s3":
    dl = jl(f"{RUN}/d4limit/*.jsonl")
    ap = [d for d in dl if d.get("kind") == "override_applied"]
    pl = [d for d in dl if d.get("kind") == "plan_limits"]
    ck("P2 limit-1 override applied on both ranks over the layer's 10.0, and the plans carry 1.0",
       len({d.get("pid") for d in ap}) >= 2 and all(d.get("original") == 10.0 and d.get("override") == 1.0 for d in ap)
       and pl and all(set(d.get("values") or []) <= {"1.0", "True"} for d in pl),
       f"{[(d.get('method'), d.get('original'), d.get('override')) for d in ap][:4]}; plans {[d.get('values') for d in pl][:4]}")
json.dump(V, open(f"{RUN}/verdict.json", "w"), indent=1, default=str)
fails = [k for k, v in V["checks"].items() if not v["ok"]]
print(f"\nSESSION {MODE} {BUNDLE}: {'ALL CHECKS PASS' if not fails else 'FAILED: ' + '; '.join(fails)}")
