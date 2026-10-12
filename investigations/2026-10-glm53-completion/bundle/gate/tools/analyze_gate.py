#!/usr/bin/env python3
"""Analyse the gate run: checker counters (per rank, branch, graph mode, rows, mixed), FULL-replay
reconciliation, route-transition ring, per-phase deltas, records, throughput.
Usage: analyze_gate.py RUN_DIR  -> prints a report and writes RUN_DIR/gate-summary.json"""
import collections
import glob
import json
import os
import statistics
import sys

RUN = sys.argv[1]
files = sorted(glob.glob(f"{RUN}/col511check/c511_p*.jsonl"))
KEYM = ["calls", "rows_valid", "rows_full", "rows_unfilled", "rows_nan", "rows_full_nan_excluded", "rows_dup",
        "rows_nvalid_mismatch", "rows_id_out_of_range", "rows_slot_score_inconsistent",
        "pre_lowest", "pre_tie", "pre_not_lowest", "pre_below_1_3", "pre_below_4_31", "pre_below_ge32",
        "pre_unfilled_valid_in_col511", "post_set_changed", "post_lowest", "post_tie", "post_not_lowest",
        "post_unfilled_valid_in_col511", "consumer_rows_checked", "consumer_mismatch", "stale_digest",
        "ind_rows", "ind_pre_ok_exact", "ind_pre_ok_tol", "ind_post_ok_exact", "ind_post_ok_tol",
        "ind_kernel_pools_checked", "ind_kernel_out_of_bound", "ind_nan_rows", "seq_pool_mismatch",
        "post_score_lookup_mismatch"]
summary = {"files": [], "ranks": {}}

# phase timeline from the client
dumps, phases, reqs = [], [], []
cj = f"{RUN}/client.jsonl"
if os.path.exists(cj):
    for l in open(cj):
        d = json.loads(l)
        if d["kind"] == "dump":
            dumps.append(d)
        elif d["kind"] == "phase":
            phases.append(d)
        elif d["kind"] == "request":
            reqs.append(d)


def phase_of(t):
    tag = "startup"
    for d in dumps:
        if d["t"] <= t:
            tag = d["tag"]
    return tag


def agg(slots, key=lambda s: (s["branch"], s["mode"])):
    out = collections.OrderedDict()
    for s in slots:
        k = key(s)
        a = out.setdefault(k, collections.Counter())
        for m in KEYM:
            a[m] += s.get(m, 0)
    return out


def pct(a, b):
    return f"{100.0 * a / b:.2f}%" if b else "n/a"


tot_by_branch_mode = collections.OrderedDict()
for f in files:
    recs = [json.loads(l) for l in open(f)]
    polls = [r for r in recs if r["kind"] == "poll"]
    errs = [r for r in recs if r["kind"] in ("checker_error", "checker_warn")]
    alloc = [r for r in recs if r["kind"] == "alloc"]
    if not polls:
        summary["files"].append(dict(file=os.path.basename(f), polls=0, errors=len(errs),
                                     alloc=len(alloc), kinds=collections.Counter(r["kind"] for r in recs)))
        continue
    final = [p for p in polls if p.get("reason") in ("trigger_ring", "exit")]
    final = final[-1] if final else polls[-1]
    rank = final.get("rank")
    print(f"\n==================== {os.path.basename(f)} rank={rank} polls={len(polls)} "
          f"checker_errors={final.get('checker_errors')} records={final.get('n_records')} "
          f"fallback_calls={final.get('fallback_calls')}")
    for e in errs[:10]:
        print("  ", json.dumps(e)[:300])
    slots = final["slots"]
    R = summary["ranks"].setdefault(str(rank), {})
    R["final_reason"] = final.get("reason")
    R["checker_errors"] = final.get("checker_errors")
    R["fallback_calls"] = final.get("fallback_calls")
    R["layer_calls"] = final.get("layer_calls")
    layers = sorted(int(k) for k in final.get("layer_calls", {}))
    R["layers"] = layers
    print(f"  DSA layers seen: {layers}")

    # per branch x mode table
    print("  branch/mode: calls | valid rows | full | unfilled | NaN rows | PRE lowest / tie / NOT lowest "
          "(below>=32) | POST set changed / not lowest / tie | unfilled valid in c511 pre->post | "
          "consumer mismatch / checked | stale | ind rows: pre ok tol / post ok tol / kernel oob")
    for (br, md), a in agg(slots).items():
        cf = a["rows_full"] - a["rows_full_nan_excluded"]
        print(f"  {br:13s} {md:13s}: {a['calls']:7d} | {a['rows_valid']:8d} | {a['rows_full']:8d} | "
              f"{a['rows_unfilled']:6d} | {a['rows_nan']:4d} | {a['pre_lowest']}/{a['pre_tie']}/"
              f"{a['pre_not_lowest']} ({pct(a['pre_not_lowest'], cf)}; ge32 {a['pre_below_ge32']}) | "
              f"{a['post_set_changed']}/{a['post_not_lowest']}/{a['post_tie']} | "
              f"{a['pre_unfilled_valid_in_col511']}->{a['post_unfilled_valid_in_col511']} | "
              f"{a['consumer_mismatch']}/{a['consumer_rows_checked']} | {a['stale_digest']} | "
              f"{a['ind_rows']}: {a['ind_pre_ok_tol']}/{a['ind_post_ok_tol']}/{a['ind_kernel_out_of_bound']}")
        k = (br, md)
        t = tot_by_branch_mode.setdefault(k, collections.Counter())
        t.update(a)
    other = collections.Counter()
    for s in slots:
        for m in ("rows_dup", "rows_nvalid_mismatch", "rows_id_out_of_range", "rows_slot_score_inconsistent",
                  "seq_pool_mismatch", "post_score_lookup_mismatch", "ind_nan_rows"):
            other[m] += s.get(m, 0)
    print("  integrity:", dict(other))
    R["integrity"] = dict(other)

    # rows bucket table (decode branches)
    print("  per rows (mode, branch, rows, mixed): calls, valid rows, PRE not lowest, POST not lowest, set changed")
    for s in sorted(slots, key=lambda s: (s["mode"], s["branch"], s["rows"], s["mixed"])):
        print(f"    {s['mode']:13s} {s['branch']:13s} rows={s['rows']:3d} mixed={s['mixed']} calls={s['calls']:7d} "
              f"valid={s['rows_valid']:8d} pre_not_lowest={s['pre_not_lowest']:7d} "
              f"post_not_lowest={s['post_not_lowest']} set_changed={s['post_set_changed']} "
              f"unf_post={s['post_unfilled_valid_in_col511']} consumer_mm={s['consumer_mismatch']}")

    # reconciliation: device FULL-mode decode calls vs host FULL replays x layers
    nl = len(layers)
    dev_full = collections.Counter()
    for s in slots:
        if s["mode"] == "full" and s["branch"] != "prefill":
            dev_full[s["rows"]] += s["calls"]
    reps = {int(k): v for k, v in final.get("full_replays", {}).items()}
    print(f"  FULL reconciliation (layers on rank = {nl}): rows: device calls | runner FULL replays x layers")
    recon = {}
    for r in sorted(set(dev_full) | set(reps)):
        exp = reps.get(r, 0) * nl
        ok = dev_full.get(r, 0) == exp
        recon[r] = dict(device=dev_full.get(r, 0), replays=reps.get(r, 0), expected=exp, match=ok)
        print(f"    {r:3d}: {dev_full.get(r, 0):8d} | {reps.get(r, 0):6d} x {nl} = {exp:8d}  {'MATCH' if ok else 'MISMATCH'}")
    R["full_reconciliation"] = recon
    pyc = collections.Counter()
    for p in final.get("py_calls", []):
        pyc[(p["mode"], p["branch"], p["capturing"])] += p["n"]
    print("  python-executed checker calls (mode, branch, capturing):", dict(pyc))
    R["py_calls"] = {f"{k[0]}|{k[1]}|cap={k[2]}": v for k, v in pyc.items()}
    print("  runner piecewise runs by tokens:", final.get("pw_runs"))
    R["pw_runs"] = final.get("pw_runs")
    R["full_replays"] = final.get("full_replays")

    # records
    recs_ = final.get("records", [])
    fl = collections.Counter(r["flags"] for r in recs_)
    print(f"  failure records: {final.get('n_records')} flags {dict(fl)}")
    for r in recs_[:8]:
        print("    ", r)
    R["records"] = final.get("n_records")
    R["records_sample"] = recs_[:20]

    # ring: route transitions
    ring = final.get("ring")
    if ring:
        cols = final["ring_cols"]
        rows_ = [dict(zip(cols, x)) for x in ring]
        rows_.sort(key=lambda x: x["call_idx"])
        seq = [(x["mode"], x["branch"], x["rows"], x["valid_rows"], x["mixed"]) for x in rows_]
        trans = collections.Counter()
        for a, b in zip(seq, seq[1:]):
            if (a[1], a[2]) != (b[1], b[2]):
                trans[(f"{['eager','pw','full','cap'][a[0]]}:{['fused','tiled'][a[1]]}:{a[2]}",
                       f"{['eager','pw','full','cap'][b[0]]}:{['fused','tiled'][b[1]]}:{b[2]}")] += 1
        rt = collections.Counter((['fused', 'tiled'][a[1]], ['fused', 'tiled'][b[1]]) for a, b in zip(seq, seq[1:])
                                 if a[1] != b[1])
        dist = collections.Counter((['eager', 'pw', 'full', 'cap'][x[0]], ['fused', 'tiled'][x[1]], x[2], x[4]) for x in seq)
        bad = sum(x["post_bad"] for x in rows_)
        prebad = sum(x["pre_not_lowest"] for x in rows_)
        print(f"  ring layer {rows_[0]['layer']}: {len(rows_)} decode steps; post_bad rows {bad}; pre_not_lowest rows {prebad}")
        print(f"    route switches fused<->tiled: {dict(rt)}")
        print(f"    (mode, route, rows, mixed) step counts: {sorted(dist.items())}")
        top = trans.most_common(30)
        print(f"    step-to-step transitions (top 30): {top}")
        R["ring"] = dict(layer=rows_[0]["layer"], steps=len(rows_), post_bad=bad, pre_not_lowest=prebad,
                         route_switches={f"{a}->{b}": v for (a, b), v in rt.items()},
                         dist={"|".join(map(str, k)): v for k, v in dist.items()},
                         transitions={f"{a} -> {b}": v for (a, b), v in trans.items()})

    # per-phase deltas (between consecutive trigger polls)
    tp = [p for p in polls if p.get("reason") in ("trigger", "trigger_ring", "exit")]
    prev = {}
    ph = collections.OrderedDict()
    for p in tp:
        cur = {}
        for s in p["slots"]:
            cur[(s["mode"], s["branch"], s["rows"], s["mixed"])] = s
        tag = phase_of(p["t"] - 1.0).split("-rep")[0]
        a = ph.setdefault(tag, collections.Counter())
        for k, s in cur.items():
            pv = prev.get(k, {})
            for m in ("calls", "rows_valid", "pre_not_lowest", "post_not_lowest", "post_set_changed"):
                a[f"{k[1]}|{k[0]}|mixed={k[3]}|{m}"] += s[m] - pv.get(m, 0)
        prev = cur
    print("  per-phase deltas (calls by branch|mode|mixed):")
    for tag, a in ph.items():
        calls = {k.rsplit("|", 1)[0]: v for k, v in a.items() if k.endswith("|calls") and v}
        pnl = sum(v for k, v in a.items() if k.endswith("|pre_not_lowest"))
        qnl = sum(v for k, v in a.items() if k.endswith("|post_not_lowest"))
        print(f"    {tag:8s} calls={calls} pre_not_lowest_rows={pnl} post_not_lowest_rows={qnl}")
    R["phases"] = {t: {k: v for k, v in a.items() if v} for t, a in ph.items()}

print("\n==================== totals over ranks (branch, mode)")
TOT = {}
for (br, md), a in tot_by_branch_mode.items():
    cf = a["rows_full"] - a["rows_full_nan_excluded"]
    TOT[f"{br}|{md}"] = dict(a)
    print(f"  {br:13s} {md:13s} calls={a['calls']} valid_rows={a['rows_valid']} clean_full={cf} unfilled={a['rows_unfilled']} "
          f"nan_rows={a['rows_nan']} | PRE not lowest {a['pre_not_lowest']} ({pct(a['pre_not_lowest'], cf)}), tie {a['pre_tie']}, "
          f"below>=32 {a['pre_below_ge32']} ({pct(a['pre_below_ge32'], cf)}), unfilled-valid-c511 {a['pre_unfilled_valid_in_col511']} "
          f"| POST set_changed {a['post_set_changed']} not_lowest {a['post_not_lowest']} tie {a['post_tie']} "
          f"unfilled-valid-c511 {a['post_unfilled_valid_in_col511']} | consumer {a['consumer_mismatch']}/{a['consumer_rows_checked']} "
          f"stale {a['stale_digest']} | ind {a['ind_rows']}: pre_ok_tol {a['ind_pre_ok_tol']} post_ok_tol {a['ind_post_ok_tol']} "
          f"post_ok_exact {a['ind_post_ok_exact']} kernel_oob {a['ind_kernel_out_of_bound']}/{a['ind_kernel_pools_checked']}")
summary["totals"] = TOT

# throughput (same definition as tailguard: aggregate generated tokens / phase wall)
print("\n==================== throughput (aggregate completion tokens / phase wall)")
byphase = collections.defaultdict(list)
for r in reqs:
    byphase[r["phase"]].append(r)
TP = {}
for typ in ("T", "F", "D", "S"):
    vals, per = [], []
    for p in phases:
        if not p["tag"].startswith(typ + "-"):
            continue
        rs = byphase[p["tag"]]
        toks = sum((r.get("completion_tokens") or 0) for r in rs)
        vals.append(toks / p["secs"])
        per.extend((r.get("completion_tokens") or 0) / r["secs"] for r in rs if r.get("secs"))
    if vals:
        TP[typ] = dict(n=len(vals), median=round(statistics.median(vals), 1), min=round(min(vals), 1),
                       max=round(max(vals), 1), per_request_median=round(statistics.median(per), 2))
        print(f"  {typ}: phases={len(vals)} median {TP[typ]['median']} tok/s [{TP[typ]['min']}-{TP[typ]['max']}], "
              f"per-request median {TP[typ]['per_request_median']}")
errs = [r for r in reqs if r.get("error")]
fins = collections.Counter(r.get("finish") for r in reqs)
print(f"  requests: {len(reqs)}, errors {len(errs)}, finish {dict(fins)}")
summary["throughput"] = TP
summary["requests"] = dict(n=len(reqs), errors=len(errs), finish=dict(fins))
json.dump(summary, open(f"{RUN}/gate-summary.json", "w"), indent=1, default=str)
