#!/usr/bin/env python3
"""Price a scoped attempt 2 from attempt 1's MEASURED phase times (client.jsonl) and fixed overhead (costs.jsonl).
CPU only. Overhead = occupied seconds - client span (start-up, readiness, final dump, stop)."""
import json
RUN = "<runs>/hook-validation/run-a1"
CL = [json.loads(l) for l in open(RUN + "/client.jsonl")]
ph = {r["tag"]: r["secs"] for r in CL if r.get("kind") == "phase"}
DUMP = 6.0
t_start = [r for r in CL if r.get("kind") == "start"][0]["t"]
t_end = max(r.get("t1", 0) for r in CL if r.get("kind") == "phase") + DUMP
cost = json.loads(open(RUN + "/../costs.jsonl").readline())
occupied = cost["seconds"]; client = t_end - t_start; overhead = occupied - client
spent = cost["gpu_hours"]; left = 2.4 - spent
def price(phases, extra=0.0):
    s = sum(ph[p] + DUMP for p in phases) + extra + DUMP
    return dict(client_s=round(s), occupied_s=round(overhead + s), gpu_h=round(2 * (overhead + s) / 3600, 3))
A = ["V0", "V2a-natural", "V2a-oracle", "V2c", "V4-d2", "V4-b1", "V4-d2solo", "V4-rebuild"]
plans = {
    "minimal (fix validation: V0, V2a natural+oracle, V2c, V4 + 2nd rebuild + 2nd B1 arm)": price(A, extra=ph["V4-rebuild"] + DUMP + ph["V4-b1"] + DUMP),
    "minimal + V3 snapshots (H4 prospective)": price(A + ["V3-forced", "V3-refs"], extra=ph["V4-rebuild"] + DUMP + ph["V4-b1"] + DUMP),
    "minimal + V3 + V2a poison (H3 on the new image)": price(A + ["V3-forced", "V3-refs", "V2a-poison"], extra=ph["V4-rebuild"] + DUMP + ph["V4-b1"] + DUMP),
    "attempt-1 plan repeated in full": price(list(ph)),
}
out = dict(attempt1=dict(occupied_s=occupied, client_s=round(client), overhead_s=round(overhead), gpu_h=spent),
           remaining_gpu_h=round(left, 4), max_wall_s=int(left * 3600 / 2), plans=plans, phase_secs=ph)
json.dump(out, open("<runs>/hook-validation/diagnosis-a1/out/attempt2_cost.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "phase_secs"}, indent=1))
