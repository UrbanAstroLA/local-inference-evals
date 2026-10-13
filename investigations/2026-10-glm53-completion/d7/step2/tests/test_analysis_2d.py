#!/usr/bin/env python3
"""CPU tests (pure Python) of the amended analysis' phase-2d path and window cut (B1-5).  NO GPU.

Synthetic run directory: EngineCore schedule records, worker step records (ledger 'new' events with the
prompt length), polls before and after the 2d cut, client 2d case records and a server.log args line.
  T1  per case: requested D vs ACTUAL decode rows scheduled with the doc-127 first chunk, other prefill
      rows reported, chunk1/chunk2 from num_computed (post-schedule convention, measured on run-a3),
      predictions budget - sum(n_i + K) - K and 2045 - 7*D_actual, residue; requested != actual reported
      as such (not as an arithmetic contradiction)
  T2  the frozen window ends at the last readout before the 2d start: counters after the cut are
      reported as the 2d window and never enter the verdict
usage: test_analysis_2d.py OUT.json
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
import analyze_mapguard as am  # noqa: E402

RESULTS = []


def result(name, ok, **kw):
    RESULTS.append(dict(test=name, ok=bool(ok), **kw))
    print(("PASS " if ok else "FAIL ") + name + "  " + json.dumps(kw, default=str)[:600], flush=True)


def jl(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def counters(**pf):
    sites = ["mla_w", "idx_w_pf", "idx_w_dec", "tail_w_seed", "tail_w_stash", "idx_r_dec", "idx_r_pf", "mla_r", "pos"]
    names = ["exec", "armed", "checked", "written", "spec", "high", "mism", "mism_spec", "missing", "missing_pred",
             "unexp", "inact", "foreign", "stale", "wgroup", "null", "ownoff", "oor", "shared", "pos_contig",
             "pos_bounds", "qsl", "rlen", "rtable", "sel_foreign", "sel_causal", "masked_live", "sel_len", "grouppos",
             "pool_cross", "sel_hole", "inact_read", "s_events", "s_verified", "s_indet", "s_unavail", "s_missing",
             "s_mism", "pc_checked", "pc_mism", "d7_fallback", "legacy"]
    c = {s: {n: 0 for n in names} for s in sites}
    c["idx_w_pf"].update(pf)
    return c


def build(run):
    os.makedirs(os.path.join(run, "mapguard"))
    K, budget = 3, 2048
    sched, steps, cases = [], [], []
    t, seq = 1000.0, 0
    plan = [(0, 0, 0), (1, 1, 0), (3, 3, 0), (5, 4, 0), (7, 7, 1)]   # (requested D, actual decode rows, other prefill rows)
    for i, (dreq, dact, opre) in enumerate(plan):
        rid = f"chatcmpl-doc{i}"
        t += 10
        others = {f"bg{i}-{j}": [5000 + j, 1 + K, K] for j in range(dact)}
        others.update({f"pf{i}-{j}": [700, 300, 0] for j in range(opre)})
        n1 = min(2795, budget - sum(v[1] + K for v in others.values()) - K)
        seq += 1
        sched.append(dict(kind="sched", seq=seq, t=t, fp=f"f{seq}", reqs={**others, rid: [n1, n1, 0]}, dig="x", changed={},
                          finished=[], free_blocks=1))
        seq += 1
        sched.append(dict(kind="sched", seq=seq, t=t + 0.1, fp=f"f{seq}", reqs={rid: [2795, 2795 - n1, 0]}, dig="x",
                          changed={}, finished=[], free_blocks=1))
        steps.append(dict(kind="step", rank=0, t=t + 0.01, step=seq, ev=[["new", rid, 50 + i, 0, 2795]], rows=[]))
        cases.append(dict(kind="phase2d_case", case=i, requested_D=dreq, bg_in_decode=dreq, seeds=dict(bg=[], doc=i),
                          t_send=t - 0.2, t_done=t + 1.0, prompt_tokens=2795, error=None))
    init = dict(kind="init", rank=0, t=1.0, geo=dict(bs=4352, kbs=64, kpool=4, ring=8, nb=1275), issues=[],
                owned_layers_impl=[3], owned_layers_rule=[3], async_scheduling=True,
                b1_amendment=dict(scap=65536, skip_tail_env=None))
    polls = [dict(kind="poll", rank=0, t=5.0, reason="baseline", counters=counters(), layers={}, ring=[[1, 0]]),
             dict(kind="poll", rank=0, t=900.0, reason="trigger", counters=counters(pc_checked=20000, s_verified=5),
                  layers={}, ring=[[2, 0]]),
             dict(kind="poll", rank=0, t=2000.0, reason="trigger",
                  counters=counters(pc_checked=20100, s_verified=6, s_mism=1), layers={}, ring=[[3, 0]])]
    jl(os.path.join(run, "mapguard", "mg_p1.jsonl"), [init] + polls + steps)
    jl(os.path.join(run, "mapguard", "ms_p2.jsonl"), [dict(kind="installed_scheduler", t=1.0)] + sched)
    jl(os.path.join(run, "client.jsonl"), [dict(kind="phase2d_start", t=950.0)] + cases)
    open(os.path.join(run, "server.log"), "w").write(
        "non-default args: {'max_num_batched_tokens': 2048, 'speculative_config': {'method': 'dflash2', "
        "'num_speculative_tokens': 3}}\n")
    return plan


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "analysis-2d.json")
    run = tempfile.mkdtemp(prefix="a2d-")
    try:
        plan = build(run)
        p = am.phase2d(run)
        cs = p["cases"]
        ok = p["budget"] == 2048 and p["draft_slots_K"] == 3 and len(cs) == len(plan)
        exp = []
        for c, (dreq, dact, opre) in zip(cs, plan):
            n1 = min(2795, 2048 - dact * 7 - opre * 303 - 3)
            exp.append(n1)
            ok &= (c["status"] == "ok" and c["actual_D"] == dact and c["other_prefill_rows"] == opre
                   and c["chunk1_len"] == n1 and c["chunk1_start"] == 0 and c["chunk2_start"] == n1
                   and c["formula_matches"] and c["requested_eq_actual_D"] == (dreq == dact)
                   and (c["arith_matches"] is None if opre else c["arith_matches"] == (n1 == 2045 - 7 * dact))
                   and c["prompt_tokens_ok"])
        result("T1 phase 2d: actual D, other prefill rows, chunk start/length, budget formula and 2045-7D per case; "
               "requested != actual D reported as such", ok,
               cases=[{k: c.get(k) for k in ("requested_D", "actual_D", "other_prefill_rows", "chunk1_len", "chunk2_start",
                                             "pred_budget_formula", "pred_2045_minus_7D", "arith_matches",
                                             "requested_eq_actual_D", "split_residue")} for c in cs], expected=exp)
        res = am.analyze(run)
        w = res["per_rank"][0]["d7"]["counters"]
        result("T2 window cut at the 2d start: frozen window counters exclude 2d; 2d-window violations reported "
               "separately, not in the verdict",
               w["pc_checked"] == 20000 and w["s_mism"] == 0 and not res["violations"]
               and res["phase2d_window"]["violations"].get(0, {}).get("idx_w_pf.s_mism") == 1,
               window=w, p2d=res["phase2d_window"], verdict=res["verdict"], invalid=res["invalid"][:3])
    finally:
        shutil.rmtree(run, ignore_errors=True)
    n = sum(r["ok"] for r in RESULTS)
    json.dump(dict(passed=n, total=len(RESULTS), results=RESULTS), open(out, "w"), indent=1, default=str)
    print(f"{n}/{len(RESULTS)} passed -> {out}")
    sys.exit(0 if n == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
