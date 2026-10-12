#!/usr/bin/env python3
"""Summarize a tail-guard run directory (CPU only, stdlib).  Usage: summarize_tg.py RUN_DIR
Reads RUN_DIR/tailguard/tg_p*.jsonl (one file per worker process) and writes
RUN_DIR/tg_summary.json; prints the verdict lines."""
import collections
import glob
import json
import os
import sys

run = sys.argv[1]
files = sorted(glob.glob(os.path.join(run, "tailguard", "**", "tg_p*.jsonl"), recursive=True))
per_rank = collections.defaultdict(lambda: dict(
    files=[], tail_slot_source=None, last_poll=None, violations=0, violation_events=[],
    staged_tail_anomalies=[], guard_errors=[], steps=0, row_adds=0, init=None))
for f in files:
    for line in open(f):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        r = per_rank[d.get("rank", -1)]
        if f not in r["files"]:
            r["files"].append(f)
        k = d.get("kind")
        if k == "tail_slot_source":
            r["tail_slot_source"] = d
        elif k == "poll":
            if r["last_poll"] is None or d["counters"].get("steps", 0) >= r["last_poll"]["counters"].get("steps", 0):
                r["last_poll"] = d
        elif k == "violation":
            r["violations"] += 1
            if len(r["violation_events"]) < 50:
                r["violation_events"].append(dict(step=d.get("step"), layer=d.get("layer"), call=d.get("call"),
                                                  events=d.get("events", [])[:8]))
        elif k == "staged_tail_anomaly":
            r["staged_tail_anomalies"].append(d)
        elif k == "guard_error":
            r["guard_errors"].append(d)
        elif k == "step":
            r["steps"] += 1
        elif k == "row_add":
            r["row_adds"] += 1
        elif k == "init":
            r["init"] = d

out = {}
for rank, r in sorted(per_rank.items()):
    lp = r["last_poll"] or {}
    recs = lp.get("records", [])
    by_kind = collections.Counter(x.get("kind_name") for x in recs)
    foreign = [x for x in recs if x.get("kind_name") in ("seed", "stash", "prefwd_slot")]
    out[str(rank)] = dict(
        files=r["files"], tail_slot_source=(r["tail_slot_source"] or {}).get("source"),
        counters=lp.get("counters"), n_records=lp.get("n_records"), records_by_kind=dict(by_kind),
        first_foreign_records=foreign[:20],
        stale_col_records=[x for x in recs if x.get("kind_name") == "stale_col"][:20],
        divergence_records=[x for x in recs if x.get("kind_name") == "shadow_divergence"][:20],
        eager_violation_lines=r["violations"], eager_violation_sample=r["violation_events"][:10],
        staged_tail_anomalies=r["staged_tail_anomalies"][:20], guard_errors=r["guard_errors"][:10],
        steps_logged=r["steps"], row_adds=r["row_adds"], init=r["init"])
json.dump(out, open(os.path.join(run, "tg_summary.json"), "w"), indent=1)
for rank, s in out.items():
    c = s["counters"] or {}
    print(f"rank {rank}: source={s['tail_slot_source']} writes={c.get('writes')} null_alias={c.get('null_alias')} "
          f"foreign={c.get('foreign')} attn_collision={c.get('attn_collision')} tail_collision={c.get('tail_collision')} "
          f"stale_col_rows={c.get('stale_col_rows')} shadow_div={c.get('shadow_divergence')} "
          f"prefwd dev/null/foreign={c.get('prefwd_deviate')}/{c.get('prefwd_null')}/{c.get('prefwd_foreign')} "
          f"eager_violation_lines={s['eager_violation_lines']} staged_anomalies={len(s['staged_tail_anomalies'])} "
          f"guard_errors={len(s['guard_errors'])}")
