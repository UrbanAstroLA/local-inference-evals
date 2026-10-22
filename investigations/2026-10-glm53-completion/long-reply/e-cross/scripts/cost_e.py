#!/usr/bin/env python3
"""[Recomputed under amendment 22ah: 52 continuations per engine (13 prefixes x 4 seeds).]
E costing (PROPOSAL.md section 8). Host, standard library, read only on B+'s run folder, zero GPU.

Inputs: B+'s MEASURED sessions (bplus/run, hashes recorded): per engine the session wall (costs.jsonl: v070 9,956 s =
5.531 GPU-h, v091 6,982 s = 3.879 GPU-h for 48 continuations each), start-up to ready (session.json ready_after_s), client
wall, and the 48 per-request durations (requests.jsonl secs; 28 ran to the +65,536 horizon on each engine, 20 finished).
Occupied GPU-hours = 2 GPUs x session wall (start of the launcher to server stopped), as B+'s runner bills it.

Estimates per engine, 52 continuations (13 prefixes x 4 seeds, 8 in flight, plan order):
  start-up      B+'s measured ready_after_s (pessimistic: B+'s frozen pessimistic start-up, 810 / 1,140 s).
  central       B+'s session scaled to 52 continuations: start-up + client wall x 52/48 + B+'s post-client overhead.
  simulated     list scheduling of 52 requests on 8 slots in plan order (0.2 s stagger), each duration resampled from the
                engine's 48 measured B+ durations (so outcomes as B+ observed them), times a session speed factor
                1/f, f ~ lognormal(median 1, 5th percentile 0.8); 20,000 runs; median and 95th percentile.
  pessimistic   95th percentile of the same simulation with every continuation running to the horizon (durations
                resampled from the 28 full-horizon requests only) and the pessimistic start-up.
  ceiling       99.9th percentile of the pessimistic simulation, rounded up to 0.1 GPU-h (as d091). Run ceiling = the sum of
                the two session ceilings + an inter-session gap allowance (60 s each), rounded up. For reference only: the
                slowest-observed bound (every continuation to the horizon at the engine's slowest measured full-horizon
                duration, v070 2,442 s = 26.8 tok/s, v091 1,648 s = 39.8 tok/s, ceil(52 / 8) = 7 full waves, pessimistic start-up).
  P(ceiling)    share of simulated sessions (central and pessimistic models) exceeding the per-session ceiling, and of
                simulated runs exceeding the run ceiling.
usage: cost_e.py OUT_JSON > checks/cost.out
"""
import hashlib
import json
import math
import random
import statistics
import sys

BR = "<runs>/long-reply/bplus/run"
PKG = "<runs>/long-reply/e-cross"
ENGINES = ("v070", "v091")
N_REQ, SLOTS, STAGGER, H = 52, 8, 0.2, 65536
PESS_START = {"v070": 810, "v091": 1140}          # B+ PROPOSAL.md section 8 (frozen pessimistic start-up)
GAP_S = 60
NSIM = 20000
F_SIGMA = math.log(1 / 0.8) / 1.6449


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def makespan(durs):
    slots = [STAGGER * i for i in range(SLOTS)]
    for d in durs:
        i = min(range(SLOTS), key=lambda q: slots[q])
        slots[i] += d
    return max(slots)


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def main():
    costs = {json.loads(l)["engine"]: json.loads(l) for l in open(f"{BR}/costs.jsonl") if l.strip()}
    plan = json.load(open(f"{PKG}/out/plan.json"))
    assert len(plan["order"]) == N_REQ
    out = {"what": "E costing from B+'s measured sessions (GPU-hours = 2 x session wall)", "inputs_sha256": {}, "engines": {}}
    for f in ("costs.jsonl", "v070/session.json", "v091/session.json", "v070/requests.jsonl", "v091/requests.jsonl"):
        out["inputs_sha256"][f"bplus/run/{f}"] = fsha(f"{BR}/{f}")
    rng = random.Random(20261021)
    sims_c, sims_p = {}, {}
    for e in ENGINES:
        sess = json.load(open(f"{BR}/{e}/session.json"))
        rows = [json.loads(l) for l in open(f"{BR}/{e}/requests.jsonl") if l.strip()]
        req = [r for r in rows if r.get("kind") == "request"]
        st = [r for r in rows if r.get("kind") == "start"][0]
        en = [r for r in rows if r.get("kind") == "end"][0]
        wall, ready = costs[e]["wall_s"], sess["ready_after_s"]
        client = en["t"] - st["t"]
        overhead = wall - ready - client
        durs = [r["secs"] for r in req]
        full = [r["secs"] for r in req if r["completion_tokens"] == H]
        central_s = ready + client * N_REQ / len(req) + overhead
        slow_s = PESS_START[e] + math.ceil(N_REQ / SLOTS) * max(full) + SLOTS * STAGGER + max(overhead, 0)
        sc, sp = [], []
        for _ in range(NSIM):
            f = math.exp(rng.gauss(0, F_SIGMA))
            sc.append(ready + makespan([rng.choice(durs) / f for _ in range(N_REQ)]) + overhead)
            f = math.exp(rng.gauss(0, F_SIGMA))
            sp.append(PESS_START[e] + makespan([rng.choice(full) / f for _ in range(N_REQ)]) + overhead)
        sims_c[e], sims_p[e] = sc, sp
        ceil_gh = math.ceil(2 * q(sp, 0.999) / 3600 * 10) / 10
        gh = lambda s: round(2 * s / 3600, 3)  # noqa: E731
        out["engines"][e] = {
            "bplus_measured": {"session_wall_s": wall, "gpu_hours": costs[e]["gpu_hours"], "ready_after_s": ready,
                               "client_wall_s": round(client, 1), "post_client_overhead_s": round(overhead, 1),
                               "requests": len(req), "full_horizon": len(full), "finished": len(req) - len(full),
                               "full_horizon_secs_min_median_max": [round(min(full), 1), round(statistics.median(full), 1), round(max(full), 1)],
                               "slowest_full_horizon_tok_s": round(H / max(full), 1)},
            "startup_s": {"central": ready, "pessimistic": PESS_START[e]},
            "central_scaled_gpu_h": gh(central_s), "central_scaled_wall_s": round(central_s),
            "simulated_median_gpu_h": gh(statistics.median(sc)), "simulated_p95_gpu_h": gh(q(sc, 0.95)),
            "pessimistic_p95_gpu_h": gh(q(sp, 0.95)), "pessimistic_median_gpu_h": gh(statistics.median(sp)),
            "ceiling_gpu_h": ceil_gh, "ceiling_wall_s": int(math.ceil(ceil_gh * 3600 / 2)),
            "slowest_observed_bound_gpu_h": gh(slow_s),
            "P_exceed_ceiling_central_model": sum(s > ceil_gh * 1800 for s in sc) / NSIM,
            "P_exceed_ceiling_pessimistic_model": sum(s > ceil_gh * 1800 for s in sp) / NSIM}
    E = out["engines"]
    run_ceiling = math.ceil((sum(E[e]["ceiling_gpu_h"] for e in ENGINES) + 2 * GAP_S / 3600) * 10) / 10
    tot_c = [sims_c["v070"][i] + sims_c["v091"][i] + GAP_S for i in range(NSIM)]
    tot_p = [sims_p["v070"][i] + sims_p["v091"][i] + GAP_S for i in range(NSIM)]
    out["run"] = {"central_scaled_gpu_h": round(sum(E[e]["central_scaled_gpu_h"] for e in ENGINES) + 2 * GAP_S / 3600, 2),
                  "simulated_median_gpu_h": round(2 * statistics.median(tot_c) / 3600, 2),
                  "simulated_p95_gpu_h": round(2 * q(tot_c, 0.95) / 3600, 2),
                  "pessimistic_p95_gpu_h": round(2 * q(tot_p, 0.95) / 3600, 2),
                  "ceiling_gpu_h_PLAN_MAX": run_ceiling, "watchdog_wall_s": int(run_ceiling * 3600 / 2),
                  "P_exceed_run_ceiling_central_model": sum(2 * t / 3600 > run_ceiling for t in tot_c) / NSIM,
                  "P_exceed_run_ceiling_pessimistic_model": sum(2 * t / 3600 > run_ceiling for t in tot_p) / NSIM,
                  "session_min_s_budget_rule": {e: E[e]["ceiling_wall_s"] for e in ENGINES},
                  "minimum_approval_for_the_primary": "both sessions are needed for E1: an approval below the run ceiling can "
                                                      "stop v091 by the budget rule, which reads INCOMPLETE"}
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for e in ENGINES:
        x = E[e]
        print(f"{e}: B+ measured {x['bplus_measured']['gpu_hours']} GPU-h / 48 (ready {x['startup_s']['central']} s); start-up "
              f"{x['startup_s']['central']} s (pess. {x['startup_s']['pessimistic']} s); central (scaled) {x['central_scaled_gpu_h']}; "
              f"simulated median {x['simulated_median_gpu_h']} (p95 {x['simulated_p95_gpu_h']}); pessimistic p95 "
              f"{x['pessimistic_p95_gpu_h']}; ceiling {x['ceiling_gpu_h']} ({x['ceiling_wall_s']} s); P(exceed) central "
              f"{x['P_exceed_ceiling_central_model']:.5f}, pessimistic {x['P_exceed_ceiling_pessimistic_model']:.5f}")
    print("run:", json.dumps(out["run"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
