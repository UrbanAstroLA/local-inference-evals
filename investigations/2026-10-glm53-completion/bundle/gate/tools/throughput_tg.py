#!/usr/bin/env python3
"""Per-phase decode throughput from a tail-guard run's client.jsonl (stdlib).  Usage: throughput_tg.py RUN_DIR...
Aggregate tok/s = sum(completion_tokens) / phase wall; per-request tok/s = completion_tokens / request secs.
Requests are prefix-cached (warm phase), so phase wall is dominated by decode."""
import json, statistics, sys
for run in sys.argv[1:]:
    reqs, phases = [], []
    for line in open(f"{run}/client.jsonl"):
        d = json.loads(line)
        (reqs if d.get("kind") == "request" else phases).append(d)
    out = {}
    for tag in ("T", "F"):
        ph = [p for p in phases if p.get("kind") == "phase" and p["tag"].startswith(tag + "-")]
        agg, per, fin, err, n = [], [], {}, 0, 0
        for p in ph:
            rs = [r for r in reqs if r["phase"] == p["tag"]]
            toks = sum(r.get("completion_tokens") or 0 for r in rs)
            agg.append(toks / p["secs"] if p["secs"] else 0)
            for r in rs:
                n += 1
                fin[r.get("finish")] = fin.get(r.get("finish"), 0) + 1
                err += 1 if r.get("error") else 0
                if r.get("completion_tokens") and r.get("secs"):
                    per.append(r["completion_tokens"] / r["secs"])
        out[tag] = dict(phases=len(ph), requests=n, errors=err, finish=fin,
                        agg_tok_s_median=round(statistics.median(agg), 1) if agg else None,
                        agg_tok_s_mean=round(statistics.mean(agg), 1) if agg else None,
                        agg_tok_s_min=round(min(agg), 1) if agg else None,
                        agg_tok_s_max=round(max(agg), 1) if agg else None,
                        per_req_tok_s_median=round(statistics.median(per), 2) if per else None)
    warm = [r for r in reqs if r["phase"] == "warm"]
    out["warm"] = dict(requests=len(warm), errors=sum(1 for r in warm if r.get("error")))
    print(run, json.dumps(out))
