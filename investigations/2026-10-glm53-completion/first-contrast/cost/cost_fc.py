#!/usr/bin/env python3
"""Occupied GPU-hours of the amended first-contrast session (PROPOSAL.md s.8; PLAN.md amendment 22 item 9).
CPU only, deterministic; pre-22 model kept as cost_fc.py.pre22 (with its .json/.out).

Every input is a MEASURED attempt-2 (or earlier local) time, except the hooked decode rate at 63-120k context, which
has never been measured; it is an explicit assumption with a sensitivity table below.

Measured inputs:
  overhead     766 s = attempt-2 occupied 1,431 s minus its client span 665 s (start -> ready 742 s, gate, final dump,
               stop; hook-validation/costs.jsonl + run-a2/client.jsonl); attempt 1: 746 s
  V0           6.3 s (run-a2 phase record); phase gap 6.05 s (guard DUMP trigger + 6 s sleep)
  prefill      2,410 tok/s at long context on the B1 guard image (d7/step2/run-b1-a1 warm r0/r1); pessimistic 1,800
  decode       hooked forced decode, natural drafts, 8 in flight, contexts < 5k: 41.6 tok/s per request (V2a-natural
               2,500 tokens in 60.1 s) and 41.2 (V4-b1 2,000 tokens in 48.5 s)
  steps        poison (one token per step) with 8 snapshots: 600 steps in 51.8 s = 11.6 steps/s (run-a2 V3-forced)
  snapshot     V3 prefill-reference requests incl. one committed-state snapshot: 0.48-0.52 s each (run-a2 V3-refs);
               planning 0.3 s per snapshot on the engine step, 0.8 s pessimistic
  dumps        window dumps 110-150 MB per rank for 256-2,000-token windows (run-a2), ~190 MB at 8.2k; not timed
               separately: 3 s per request central, 8 s pessimistic (as pre-22)
ASSUMPTION (unmeasured): hooked decode at 63-120k = 33 tok/s per request central (0.8 x the measured short-context
41.6), 25 pessimistic, 40 optimistic; CV steps scale with it (11.6 x 33 / 41.6 = 9.2 steps/s central).
Session (client_fc.py order): V0; LW1, LW2 (+8 x 1,024 tokens); warm s1, C1; warm s2, C2; CV; GATE-1 (overlapped
with LC1, LC2; pessimistic +200 s wait); re-warm s2, D2; re-warm s1, D1; re-warm l2, LD2; re-warm l1, LD1; warm s3,
PS-a, PS-b, D3, PS-c, re-warm s3, C3; LW3, LD3, re-warm l3, LC3. Pessimistic also carries one LP re-warm evicted and
rebuilt cold (+796,240 tokens).
Ceiling (as pre-22): max(estimate x 1.3, pessimistic) + one invalid-start allowance 0.42 (742 s start + 20 s stop,
x 2 GPUs). With the single-attempt runner the allowance is not spendable on a second start; it stays inside the
4.29 target and is reported separately (amendment 22: the allowance is never consumed silently).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = json.load(open(os.path.join(HERE, "..", "fixtures", "position_plan.json")))
T = PLAN["totals"]
N = len(PLAN["fixtures"])
GPUS, MARGIN, TARGET = 2, 1.30, 4.29
ALLOW_S = 742 + 20
GAP = 6.05
V0 = 6.3 + GAP
CV_PREFILL = 8 * 4289                 # CV prompts P' = B1 - 63: 4,289 uncached tokens each (c0 = B1 - 4,352)
CV_STEPS = 72                         # poison advances one token per step over the 72-id list
CV_SNAPS = 2 * (9 + 4 + 5 + 6)        # snapshots per rank (fixtures x schedules' targets)
PS_REQ = dict(central=2.2, pessimistic=4.0, optimistic=1.8)    # remainder prefill + snapshot + 1,024-position dump
SCEN = {
    "central":     dict(overhead=766.0, prefill=2410.0, decode=33.0, dump=3.0, snap=0.3, gate_wait=0.0, evict=0),
    "pessimistic": dict(overhead=800.0, prefill=1800.0, decode=25.0, dump=8.0, snap=0.8, gate_wait=200.0, evict=1),
    "optimistic":  dict(overhead=746.0, prefill=2600.0, decode=40.0, dump=2.0, snap=0.3, gate_wait=0.0, evict=0),
}


def components(sc, name):
    pf, dec, dmp = sc["prefill"], sc["decode"], sc["dump"]
    max_list = T["max_forced_list"]
    lp_max = max(f["lp"]["forced_list_len"] for f in PLAN["fixtures"])
    steps_rate = 11.6 * dec / 41.6
    c = dict(
        V0=V0,
        LW_P_L=GAP + T["lp_prefix"] / pf + N * dmp,                       # LW1, LW3
        LW2_ref2=GAP + (T["lp_prefix"] + N * 1024) / pf + N * dmp,       # [A22-3] different build length
        warm_cold=GAP + T["prefix"] / pf,
        rewarm=GAP + T["remainder"] / pf,
        lp_rewarm=GAP + T["lp_remainder"] / pf,
        batch=GAP + T["remainder"] / pf + max_list / dec + N * dmp,
        lp_batch=GAP + T["lp_remainder"] / pf + lp_max / dec + N * dmp,
        CV=GAP + CV_PREFILL / pf + CV_STEPS / steps_rate + CV_SNAPS * sc["snap"],      # [A22-5]
        PS_set=GAP + N * PS_REQ[name],                                                  # [A22-4]
        gate_wait=sc["gate_wait"],
        evict=sc["evict"] * (T["lp_prefix"] - T["lp_remainder"]) / pf,
    )
    return c


def session(sc, name, decode=None):
    sc = dict(sc, decode=decode or sc["decode"])
    c = components(sc, name)
    parts = dict(
        overhead=sc["overhead"], V0=c["V0"],
        stageA_refs=c["LW_P_L"] + c["LW2_ref2"],
        stageA_primary=2 * c["warm_cold"] + 2 * c["batch"],
        CV=c["CV"],
        stageL=2 * c["lp_rewarm"] + 2 * c["lp_batch"],
        gate_wait=c["gate_wait"],
        stageB_primary=2 * c["rewarm"] + 2 * c["batch"] + c["warm_cold"] + c["batch"] + c["rewarm"] + c["batch"],
        stageB_lp=2 * c["lp_rewarm"] + 2 * c["lp_batch"] + c["LW_P_L"] + c["lp_batch"] + c["lp_rewarm"] + c["lp_batch"],
        PS=3 * c["PS_set"],
        evict=c["evict"],
    )
    s = sum(parts.values())
    # pre-22 option B, recomputed with the same inputs (for the delta of the amendment-22 additions)
    pre = sc["overhead"] + c["V0"] + 3 * c["warm_cold"] + 6 * c["rewarm"] + 6 * c["batch"] + 3 * c["LW_P_L"] + \
        6 * c["lp_rewarm"] + 6 * c["lp_batch"]
    # early stops
    stopC1 = sc["overhead"] + c["V0"] + c["LW_P_L"] + c["LW2_ref2"] + c["warm_cold"] + c["batch"]
    stopG1 = stopC1 + c["warm_cold"] + c["batch"] + c["CV"] + 2 * c["lp_rewarm"] + 2 * c["lp_batch"] + c["gate_wait"]
    return c, parts, s, pre, stopC1, stopG1


def gh(s):
    return round(GPUS * s / 3600.0, 3)


def main():
    out = {"inputs": SCEN, "totals": T, "assumption": "hooked decode at 63-120k: 33 tok/s central (UNMEASURED)",
           "scenarios": {}}
    res = {}
    for name, sc in SCEN.items():
        c, parts, s, pre, s1, s2 = session(sc, name)
        res[name] = (s, pre, s1, s2)
        out["scenarios"][name] = dict(components_s={k: round(v, 1) for k, v in c.items()},
                                      session_parts_s={k: round(v, 1) for k, v in parts.items()},
                                      session_s=round(s), gpu_h=gh(s), pre22_B_same_inputs_gpu_h=gh(pre),
                                      stop_after_C1_gpu_h=gh(s1), stop_at_gate1_gpu_h=gh(s2))
    est, pes, opt = gh(res["central"][0]), gh(res["pessimistic"][0]), gh(res["optimistic"][0])
    allow = round(GPUS * ALLOW_S / 3600.0, 3)
    core = round(max(est * MARGIN, pes), 2)
    ceiling = round(core + allow, 2)
    wall = round(max(res["central"][0] * MARGIN, res["pessimistic"][0]))
    cp = out["scenarios"]["central"]["session_parts_s"]
    min_wall = round(sum(v for k, v in cp.items() if k not in ("stageB_lp", "evict")))
    out["ceiling"] = dict(estimate_gpu_h=est, pessimistic_gpu_h=pes, optimistic_gpu_h=opt,
                          single_attempt_ceiling_gpu_h=core, invalid_start_allowance_gpu_h=allow,
                          pre22_formula_ceiling_gpu_h=ceiling, target_gpu_h=TARGET,
                          single_attempt_within_target=core <= TARGET, pre22_formula_within_target=ceiling <= TARGET,
                          wall_cap_s=wall, min_wall_s=min_wall,
                          delta_vs_pre22_B_central_gpu_h=round(est - gh(res["central"][1]), 3))
    sens = {}
    for d in (20.0, 25.0, 33.0, 41.6):
        c, parts, s, pre, s1, s2 = session(SCEN["central"], "central", decode=d)
        sens["%.1f tok/s" % d] = dict(session_s=round(s), gpu_h=gh(s), fits_wall_cap=s <= wall)
    out["decode_sensitivity_central_other_inputs"] = sens
    json.dump(out, open(os.path.join(HERE, "cost_fc.json"), "w"), indent=1)
    print("Session parts (seconds): central / pessimistic / optimistic")
    for k in out["scenarios"]["central"]["session_parts_s"]:
        print("  %-16s %8.1f %8.1f %8.1f" % (k, out["scenarios"]["central"]["session_parts_s"][k],
                                            out["scenarios"]["pessimistic"]["session_parts_s"][k],
                                            out["scenarios"]["optimistic"]["session_parts_s"][k]))
    print("Session: central %d s = %.3f GPU-h; pessimistic %d s = %.3f; optimistic %d s = %.3f" % (
        res["central"][0], est, res["pessimistic"][0], pes, res["optimistic"][0], opt))
    print("Pre-22 option B with the same inputs: central %.3f GPU-h; amendment-22 additions +%.3f" % (
        gh(res["central"][1]), est - gh(res["central"][1])))
    print("Single-attempt ceiling = max(%.3f x 1.3, %.3f) = %.2f GPU-h (target %.2f: %s); wall cap %d s; min wall "
          "(primary + gates, central) %d s" % (est, pes, core, TARGET, "within" if core <= TARGET else "EXCEEDS", wall,
                                              min_wall))
    print("Pre-22 formula (+ invalid-start allowance %.2f) = %.2f GPU-h: %s the %.2f target by %.2f" % (
        allow, ceiling, "within" if ceiling <= TARGET else "EXCEEDS", TARGET, ceiling - TARGET))
    print("Early stops: after C1 %.3f GPU-h; at GATE-1 (no injected batch) %.3f GPU-h (central)" % (
        out["scenarios"]["central"]["stop_after_C1_gpu_h"], out["scenarios"]["central"]["stop_at_gate1_gpu_h"]))
    print("Decode-rate sensitivity (central other inputs):")
    for k, v in sens.items():
        print("  %-10s %6d s  %.3f GPU-h  %s" % (k, v["session_s"], v["gpu_h"], "fits the wall cap" if v["fits_wall_cap"]
                                                 else "EXCEEDS the wall cap (the client skips batches it cannot fit)"))


if __name__ == "__main__":
    main()
