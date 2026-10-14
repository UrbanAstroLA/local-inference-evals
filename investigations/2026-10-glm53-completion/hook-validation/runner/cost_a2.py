#!/usr/bin/env python3
"""Attempt-2 cost estimate (PA2-12; CPU, host python, json only). Prices the scoped client's phases with attempt 1's
MEASURED phase seconds (run-a1/client.jsonl) and fixed overhead (costs.jsonl occupied seconds minus the client
span: start-up, readiness, final dump, stop), as diagnosis-a1/attempt2_cost.py did, with two conservative changes:
  * V4n-b1b (the second B1 arm, two sequential batches of 4) is priced at 2 x attempt 1's V4-b1 (one batch of 8),
    i.e. no credit for the smaller batch;
  * V4n-rebuild2 is priced at V4-rebuild x the ratio of prefill tokens (plan.json), at least 1.
Writes runner/cost_a2.json. usage: cost_a2.py [PLAN_JSON]   (default runner/plan-dryrun-a2/plan.json)"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HV = os.path.dirname(HERE)
CL = [json.loads(ln) for ln in open(os.path.join(HV, "run-a1", "client.jsonl"))]
cost = [json.loads(ln) for ln in open(os.path.join(HV, "costs.jsonl")) if ln.strip()]
a1 = [c for c in cost if c.get("item") == "hookval" and c.get("attempt") == 1][0]
PLAN = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "plan-dryrun-a2", "plan.json")))
ph = {r["tag"]: r["secs"] for r in CL if r.get("kind") == "phase"}
DUMP = 6.0                                           # client sleep after every guard DUMP
t_start = [r for r in CL if r.get("kind") == "start"][0]["t"]
t_end = max(r.get("t1", 0) for r in CL if r.get("kind") == "phase") + DUMP
client_a1 = t_end - t_start
overhead = a1["seconds"] - client_a1
APPROVED, GPUS = 2.4, 2
spent = sum(c["gpu_hours"] for c in cost if c.get("item") == "hookval")
left_h = APPROVED - spent
max_wall = int(left_h * 3600 / GPUS)
ptok = lambda pre: sum(p["P"] for p in PLAN if str(p.get("tag", "")).startswith(pre))  # noqa: E731
r2 = max(1.0, ptok("V4n-rebuild2-") / max(ptok("V4-rebuild-"), 1))
est = {
    "V0": ph["V0"], "V2a-natural": ph["V2a-natural"], "V2a-oracle": ph["V2a-oracle"], "V2c": ph["V2c"],
    "V4-d2": ph["V4-d2"], "V4-b1": ph["V4-b1"], "V4-d2solo": ph["V4-d2solo"], "V4-rebuild": ph["V4-rebuild"],
    "V4n-b1b": 2 * ph["V4-b1"], "V4n-rebuild2": round(ph["V4-rebuild"] * r2, 1),
    "V3-forced": ph["V3-forced"], "V3-refs": ph["V3-refs"],
}
optional = {"V2a-poison": ph["V2a-poison"]}


def price(phases):
    client = DUMP + sum(s + DUMP for s in phases.values())     # start dump + one dump per phase
    occ = overhead + client
    return dict(client_s=round(client), occupied_s=round(occ), gpu_h=round(GPUS * occ / 3600, 3),
                cumulative_gpu_h=round(spent + GPUS * occ / 3600, 3))


mand = price(est)
full = price(dict(est, **optional))
ready_a1 = 722                                        # run-a1/gate.txt "ready after 722s"
RESERVE = 180
client_room = max_wall - RESERVE - ready_a1
out = dict(
    basis="attempt-1 measured phase seconds (run-a1/client.jsonl) and overhead (costs.jsonl); DIAGNOSIS-a1 s.8.4",
    attempt1=dict(occupied_s=a1["seconds"], client_s=round(client_a1), overhead_s=round(overhead),
                  gpu_h=a1["gpu_hours"]),
    approved_item_gpu_h=APPROVED, spent_gpu_h=round(spent, 4), remaining_gpu_h=round(left_h, 4),
    wall_cap_s=max_wall, min_wall_s=1500, phase_estimates_s=est, optional_estimates_s=optional,
    rebuild2_token_ratio=round(r2, 3),
    mandatory=mand, mandatory_plus_optional_poison=full,
    client_room_s=dict(wall_cap=max_wall, minus_reserve=RESERVE, minus_readiness_a1=ready_a1, room=client_room,
                       mandatory_need=mand["client_s"], with_poison_need=full["client_s"]),
    worst_case_gpu_h=round(GPUS * max_wall / 3600, 3),
    worst_case_cumulative_gpu_h=round(spent + GPUS * max_wall / 3600, 4),
)
json.dump(out, open(os.path.join(HERE, "cost_a2.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
