#!/usr/bin/env python3
"""K report (stdlib only): one line per cell from run-*/<engine>/summary-<engine>.json, then the
per-engine reading under the frozen rules (PROPOSAL.md section 4).  usage: report.py <run_dir>"""
import json
import os
import sys

run = sys.argv[1]
plan = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plan.json")))
kinds = {c["id"]: (c["kind"], c["boundary_tokens_global"]) for c in plan["cells"]}
rc = 0
for eng in ("b1", "v070"):
    p = os.path.join(run, eng, f"summary-{eng}.json")
    if not os.path.exists(p):
        print(f"== {eng}: no summary ({p})")
        continue
    d = json.load(open(p))
    h = d.get("header", {})
    print(f"== {eng}: device {h.get('device')} sm{h.get('sm')} SMs {h.get('sms')} file pins ok {h.get('file_pins_ok')} "
          f"time-guard stop {d.get('stopped_by_time_guard')} secs {d.get('secs')} tally {d.get('tally')}")
    if d.get("aborted") or d.get("error"):
        print(f"   ABORTED/ERROR: {d.get('aborted') or d.get('error', '')[-400:]}")
    for c in d.get("cells", []):
        k, tok = kinds.get(c["id"], ("?", 0))
        fails = {x: n for x, n in (c.get("counts") or {}).items() if n and x != "rows"}
        desc = c.get("desc") or {}
        print(f"   {c['id']:24s} {k[:34]:34s} ~{tok:>8,d} tok  {c['verdict']:18s} rows {c.get('counts', {}).get('rows', 0):6d} "
              f"fails {fails if fails else '-'}  err/B max {desc.get('err_over_bound_max', 0):.3g} "
              f"col511 rank<480 {desc.get('stock_col511_rank_lt480', 0)}/{desc.get('stock_col511_rows', 0)}")
        if c.get("error"):
            print(f"      ERROR (confined to this cell): {c['error'].strip().splitlines()[-1][:200]}")
        if "persp_rows_differ" in desc or "persp0_relaunch_differs" in desc:
            # PI-4: persp / allgather / c4 / route count batch x repeat records, the rest count rows
            print(f"      persp (batch x repeat records) {(c.get('counts') or {}).get('persp', 0)}; rows differing "
                  f"{desc.get('persp_rows_differ', 0)} (order only, same column 511: {desc.get('persp_rows_order_only_cols_0_510', 0)}): "
                  f"set {desc.get('persp_rows_set_differ', 0)} "
                  f"(non-tie {desc.get('persp_rows_set_differ_nontie', 0)}), column 511 "
                  f"{desc.get('persp_rows_col511_differ', 0)} (non-tie {desc.get('persp_rows_col511_differ_nontie', 0)}, "
                  f"min rank {desc.get('persp_col511_rank_min', '-')}); perspective-0 relaunch differs in "
                  f"{desc.get('persp0_relaunch_differs', 0)} records")
    verdicts = [c["verdict"] for c in d.get("cells", [])]
    if verdicts and all(v == "PASS" for v in verdicts):
        print(f"   READING: every planned {eng} cell PASS -> the tested selection failures are retired for these "
              f"lengths, rows and geometries (not termination equivalence).")
    else:
        rc = 1
        print(f"   READING: not all cells PASS ({sorted(set(verdicts))}); nothing is retired for the cells that did not "
              f"PASS. FAIL = a selection defect at that cell; FAIL-NUMERIC = kernel scores not bit-exact on an exact "
              f"family; INCONCLUSIVE-BOUND = fp32 bound exceeded; INVALID-ROUTE = cell did not run the planned route; "
              f"NOT-RUN/PARTIAL = time guard; ERROR = harness exception (see log).")
sys.exit(rc)
