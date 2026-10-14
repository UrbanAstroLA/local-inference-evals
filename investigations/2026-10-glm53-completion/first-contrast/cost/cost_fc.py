#!/usr/bin/env python3
"""Occupied GPU-hour estimates and ceilings for the first long-context contrast (first-contrast/PROPOSAL.md s.8).

CPU only, deterministic. Token totals come from ../fixtures/position_plan.json. Occupied GPU-hours = 2 GPUs x wall
from server start to GPU release (the convention of every earlier cost line).

Measured inputs (local receipts; nothing re-measured here):
  overhead      start -> ready 742 s (hook-validation/run-a2/gate.txt; attempt 1: 722 s); occupied minus client
                time 766 s in attempt 2 (1,431 s occupied, client 665.4 s from client.jsonl start/end) and 746 s
                in attempt 1 (runner/cost_a2.json). Planning: 766 s.
  probes        V0 (4 greedy unkeyed probes) 6.3 s + one dump pause (run-a2 client.jsonl).
  phase gap     6.05 s per phase (the client's guard DUMP trigger + 6 s sleep; run-a2 phase records).
  prefill       2,410 tok/s on the B1 guard image at long context: 63,641 tokens extending 56,576 -> 120,217 in
                26.36 s and 64,217 cold tokens in 26.37 s (d7/step2/run-b1-a1 client.log, warm r0/r1). Short
                prefills in attempt 2: 11,494 tokens in 5.0 s (V4-rebuild). Pessimistic: 1,800 tok/s.
  forced decode per request, hooked image, natural drafts, 8 in flight: 40.1-41.6 tok/s for the longest requests of
                V4-b1, V4-d2 and V2a-natural (2,000-2,500 tokens, contexts < 5k; part of their windows ran with
                fewer co-runners), 32-35 tok/s for short requests that ran wholly at 8 in flight. Free B1 decode
                shows little context dependence at 8 in flight (b1-screen server log, 53-65 tok/s/request across
                KV bands), but the hooked rate at 63-120k is NOT measured. Planning 33 tok/s; pessimistic 25.
  window dumps  dumpwin pages: 110-150 MB per rank for 256-2,000-token windows (run-a2/forcewin/snap_V4-*),
                about 108 MB + 10 KB/token -> about 190 MB per rank at 8,199 tokens. Dump seconds were not timed
                separately: planning 3 s per request (serial, at removal), pessimistic 8 s.
  short stratum S (optional; PROPOSAL.md s.3.7): the attempt-2 V4 D2 set (8 exact-id windows of 256-2,000 tokens at
                the prompt) run as C x3 / D x3 plus two prefill rebuilds. Measured phase seconds, dumps included:
                V4-d2 49.9 s, V4-b1 48.5 s, V4-rebuild 5.0 s, V4n-rebuild2 5.4 s (run-a2/client.jsonl). Pessimistic x1.3.
Cache rule (PROPOSAL.md s.4.3): one salt <-> one prompt per fixture. A round's cold warm request (unkeyed or ref,
max_tokens 1) prefills the whole prompt; each later batch with that salt prefills only the uncached remainder
(P mod 4,352). A re-warm before every batch costs a cache hit (remainder) unless the blocks were evicted.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = json.load(open(os.path.join(HERE, "..", "fixtures", "position_plan.json")))
T = PLAN["totals"]
GPUS = 2
MARGIN = 1.30
ALLOW_S = 742 + 20                    # one invalid start: start -> ready + stop

SCEN = {
    "central":     dict(overhead=766.0, prefill=2410.0, decode=33.0, dump=3.0),
    "pessimistic": dict(overhead=800.0, prefill=1800.0, decode=25.0, dump=8.0),
    "optimistic":  dict(overhead=746.0, prefill=2600.0, decode=40.0, dump=2.0),
}
GAP = 6.05
V0 = 6.3 + GAP
N = len(PLAN["fixtures"])


def comp(sc):
    pf, dec, dmp = sc["prefill"], sc["decode"], sc["dump"]
    max_list = T["max_forced_list"]
    lp_max = max(f["lp"]["forced_list_len"] for f in PLAN["fixtures"])
    c = {}
    # primary: per round one cold warm of P (unkeyed), then C_r and D_r batches (re-warm hit before each)
    c["primary_round_warm"] = GAP + T["prefix"] / pf
    c["rewarm_hit"] = GAP + T["remainder"] / pf
    c["primary_batch"] = GAP + T["remainder"] / pf + max_list / dec + N * dmp
    # LP: per LP round one cold warm of P_L that is also the long-context prefill reference (local_fw_ref = P)
    # and dumps the prefill-written pools [P, P_L): +N dumps
    c["lp_round_warm"] = GAP + T["lp_prefix"] / pf + N * dmp
    c["lp_rewarm_hit"] = GAP + T["lp_remainder"] / pf
    c["lp_batch"] = GAP + T["lp_remainder"] / pf + lp_max / dec + N * dmp
    # IH without LP: R1 = ref + dump of prompt+window (own salt, cold); R2 = dump at build +200 (own salt, cold)
    c["ih_R1"] = GAP + T["rebuild_tokens_R1"] / pf + N * dmp
    c["ih_R2"] = GAP + T["rebuild_tokens_R2"] / pf + N * dmp
    k = 1.0 if sc is SCEN["central"] or sc is SCEN["optimistic"] else 1.3
    c["short_run"] = GAP + 49.9 * k
    c["short_rebuilds"] = 2 * (GAP + 5.4 * k)
    return c


def options(sc):
    c = comp(sc)
    prim = 3 * c["primary_round_warm"] + 6 * c["rewarm_hit"] + 6 * c["primary_batch"]
    lp3 = 3 * c["lp_round_warm"] + 6 * c["lp_rewarm_hit"] + 6 * c["lp_batch"]
    lp2 = 2 * c["lp_round_warm"] + 4 * c["lp_rewarm_hit"] + 4 * c["lp_batch"]
    ih = c["ih_R1"] + c["ih_R2"]
    bridge = c["primary_round_warm"] + 2 * c["rewarm_hit"] + 2 * c["primary_batch"]
    oh = sc["overhead"] + V0
    S = 6 * c["short_run"] + c["short_rebuilds"]
    o = {
        "A0  primary only (C x3, D x3), no long-context content check": [oh + prim],
        "A   primary + long-context instrument health (R1 ref+dump, R2 dump)": [oh + prim + ih],
        "B   one session: primary + LP control x3 (instrument health folded into the LP warms)": [oh + prim + lp3],
        "B2  one session: primary + LP control x2 per condition": [oh + prim + lp2],
        "A+S primary + instrument health + short stratum S (attempt-2 D2 set)": [oh + prim + ih + S],
        "B+S one session: primary + LP x3 + short stratum S": [oh + prim + lp3 + S],
        "C   two sessions: S1 = A; S2 = LP x3 + bridge (one C/D round re-run)": [oh + prim + ih, oh + lp3 + bridge],
    }
    return c, {k: dict(session_s=[round(x) for x in v], gpu_h=round(sum(v) * GPUS / 3600.0, 3)) for k, v in o.items()}


def main():
    out = {"inputs": SCEN, "totals": T, "scenarios": {}}
    for name, sc in SCEN.items():
        c, o = options(sc)
        out["scenarios"][name] = dict(components_s={k: round(v, 1) for k, v in c.items()}, options=o)
    cen, pes = out["scenarios"]["central"]["options"], out["scenarios"]["pessimistic"]["options"]
    allow = GPUS * ALLOW_S / 3600.0
    out["ceilings"] = {}
    print("Components (seconds), central / pessimistic:")
    for k in out["scenarios"]["central"]["components_s"]:
        print("  %-20s %7.1f  %7.1f" % (k, out["scenarios"]["central"]["components_s"][k],
                                          out["scenarios"]["pessimistic"]["components_s"][k]))
    print("\n%-88s %9s %9s %9s %9s %8s" % ("Option", "estimate", "pessim.", "optim.", "ceiling", "wallcap"))
    for k in cen:
        est, p = cen[k]["gpu_h"], pes[k]["gpu_h"]
        ceil = round(max(est * MARGIN, p) + allow, 2)
        nsess = len(cen[k]["session_s"])
        wall = [round(max(s * MARGIN, ps)) for s, ps in zip(cen[k]["session_s"], pes[k]["session_s"])]
        out["ceilings"][k] = dict(estimate_gpu_h=est, pessimistic_gpu_h=p,
                                  optimistic_gpu_h=out["scenarios"]["optimistic"]["options"][k]["gpu_h"],
                                  ceiling_gpu_h=ceil, invalid_start_allowance_gpu_h=round(allow, 3),
                                  sessions=nsess, wall_cap_s_per_session=wall,
                                  central_session_s=cen[k]["session_s"])
        print("%-88s %9.2f %9.2f %9.2f %9.2f %s" % (k, est, p, out["scenarios"]["optimistic"]["options"][k]["gpu_h"],
                                                   ceil, wall))
    # the early-stop saving: structural failure of C1 stops the session after round 1
    c = out["scenarios"]["central"]["components_s"]
    stop1 = SCEN["central"]["overhead"] + V0 + c["primary_round_warm"] + c["primary_batch"]
    out["early_stop_after_C1_gpu_h"] = round(stop1 * GPUS / 3600.0, 3)
    print("\nEarly stop after C1 (structural gate fails): %.2f GPU-h" % out["early_stop_after_C1_gpu_h"])
    # LP marginal cost
    out["lp_marginal_gpu_h"] = round(cen[[k for k in cen if k.startswith("B ")][0]]["gpu_h"]
                                     - cen[[k for k in cen if k.startswith("A0")][0]]["gpu_h"], 3)
    print("LP control marginal (B - A0), central: %.2f GPU-h" % out["lp_marginal_gpu_h"])
    json.dump(out, open(os.path.join(HERE, "cost_fc.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
