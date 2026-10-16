#!/usr/bin/env python3
"""Budget for the stock 0.7.0 replication (PROPOSAL-v2.md section 5). CPU only, standard library, fixed seed.

1. Reconstruction [R]: the record's occupied GPU-hours from timing-reconstruction.json (2 GPUs x server start to stop) and
   the draft's ceiling rule (per session: 10 min start-up + request window x 1.15 + 1 min shutdown) -> 22.97 / 26.45, maximum
   = ceiling + one session-1-sized invalid-run allowance = 35.0.
2. Outcome dependence [I, model]: a pass's request window grows with its cap exhaustions (a capped reply runs to 327,680
   tokens; on 0.7.0 each took 100-149 minutes). Window(n caps) = base + slope x n, with two slopes:
     empirical: least squares over the record's three 0.7.0 passes (0, 3 and 1 caps; 213.6, 238.8, 218.5 min);
     token-based (conservative): a cap's 327,680 tokens at the pass's aggregate throughput (0.7.0 passes 2-3: 257-265 tok/s),
       i.e. every cap token is added time.
   Per-pass cap counts are drawn from the power model's per-question rates (power_rep070.py posteriors: S1b old 0.7.0
   rates, S2 0.9.1 record's rates), with all empties counted as caps (all four 0.7.0 empties and 15 of 16 0.9.1 empties
   were cap exhaustions). Pass-to-pass variation beyond caps: a multiplicative normal factor (sd 5%) per pass [I].
   Session overheads (start-up to first request, last request to stop, gap between passes) as recorded.
Variant F changes the seeds only; the model has no seed term (see PROPOSAL-v2.md section 5 for the evidence).

Change set 22n: --thresholds X ... reports P(total occupied GPU-hours >= X) per scenario (X = 36.5: the B-lean watchdog
would stop the run); --approved A reports P(session 2 is not started) under the runner's rule (session 2 starts only if
the approved wall A x 3600 / 2 minus session 1's wall is at least the record's session-2 wall, 27,955 s), and
P(incomplete) = P(session 2 not started or total >= A). Seeds and the token cap are never changed and no request is
dropped to fit the budget: these are the probabilities that the run is cut short and reported INCOMPLETE.

usage: budget_rep070.py [--sims 20000] [--seed 20261017] [--thresholds 36.5] [--approved 36.5] [--out budget-rep070.json]
"""
import argparse, json, math, os, random, statistics, sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "frozen"))
from power_col511 import counts, fit_bb, CMP_ROWS, V070_ROWS, N_PASSES  # noqa: E402

TIM = json.load(open(os.path.join(HERE, "timing-reconstruction.json")))
CAP_TOK = 327680
REC_PASSES = [(0, 213.57), (3, 238.80), (1, 218.48)]      # (caps, request window minutes) of the 0.7.0 record, passes 1-3
AGG_TOK_S = (265 + 257) / 2                              # passes 2-3 aggregate completion tok/s (request logs)


def iso(s):
    import datetime as dt
    return dt.datetime.fromisoformat(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--thresholds", type=float, nargs="*", default=[])
    ap.add_argument("--approved", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    S = [s for s in TIM["sessions"] if s["record"].startswith("0.7.0")]
    out = {"reconstruction": {}, "model": {}, "scenarios": []}
    # ---- 1. reconstruction and the draft rule
    ceil = []
    for s in S:
        win_h = (iso(s["last_request_end"]) - iso(s["first_request"])).total_seconds() / 3600
        c = 10 / 60 + 1.15 * win_h + 1 / 60
        ceil.append(c)
        out["reconstruction"][s["session"]] = {"passes": s["passes"], "wall_h": s["wall_h"], "gpu_h": s["gpu_hours"], "request_window_h": round(win_h, 4),
                                               "overhead_min": round((s["wall_h"] - win_h) * 60, 1), "ceiling_wall_h": round(c, 3),
                                               "ceiling_gpu_h": round(2 * c, 2)}
    exp_gh = round(2 * sum(s["wall_h"] for s in S), 2)
    ceil_gh = round(2 * sum(ceil), 2)
    max_gh = round(ceil_gh + 2 * ceil[0], 1)
    out["reconstruction"]["expected_gpu_h"] = exp_gh
    out["reconstruction"]["ceiling_gpu_h"] = ceil_gh
    out["reconstruction"]["maximum_gpu_h"] = max_gh
    # ---- 2. outcome dependence
    xs, ys = [c for c, _ in REC_PASSES], [w for _, w in REC_PASSES]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope_emp = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    slope_tok = CAP_TOK / AGG_TOK_S / 60
    over1 = out["reconstruction"]["2026-09-29"]["overhead_min"]
    over2 = out["reconstruction"]["2026-10-10"]["overhead_min"]
    out["model"] = {"slope_empirical_min_per_cap": round(slope_emp, 2), "slope_token_min_per_cap": round(slope_tok, 2),
                    "session_overhead_min": [over1, over2], "pass_noise_sd": 0.05, "agg_tok_s": AGG_TOK_S}
    k091, k070 = counts(CMP_ROWS), counts(V070_ROWS)
    f091, f070 = fit_bb(k091), fit_bb(k070)
    post091 = [(f091["a"] + x, f091["b"] + N_PASSES - x) for x in k091]
    a7b, b7b = f070["pooled_rate"] * f091["s"], (1 - f070["pooled_rate"]) * f091["s"]
    post070b = [(a7b + x, b7b + N_PASSES - x) for x in k070]
    for si, (sname, post) in enumerate((("S1b old 0.7.0 rates", post070b), ("S2 0.9.1 record's rates", post091))):
        for sl_name, slope in (("empirical", slope_emp), ("token-based", slope_tok)):
            base = statistics.mean(w - slope * c for c, w in REC_PASSES)
            rng = random.Random(a.seed + 10 * si + (sl_name == "token-based"))
            tot = []
            s1 = []
            for _ in range(a.sims):
                p = [rng.betavariate(x, y) for x, y in post]
                n = [sum(rng.random() < q for q in p) for _ in range(N_PASSES)]
                w = [max(0.0, (base + slope * n[j]) * rng.gauss(1, 0.05)) for j in range(N_PASSES)]
                h1 = (over1 + w[0]) / 60
                h2 = (over2 + w[1] + w[2]) / 60
                s1.append(h1)
                tot.append(2 * (h1 + h2))
            s1u, totu = list(s1), list(tot)
            tot.sort()
            q = lambda f: round(tot[min(len(tot) - 1, int(f * len(tot)))], 2)
            s1.sort()
            extra = {f"P_total_ge_{x}": round(sum(t >= x for t in tot) / len(tot), 4) for x in a.thresholds}
            if a.approved:
                cap_s = a.approved * 3600 / 2
                no2 = [cap_s - h * 3600 < 27955 for h in s1u]
                inc = [n or t >= a.approved for n, t in zip(no2, totu)]
                extra.update({f"P_session2_not_started_at_{a.approved}": round(sum(no2) / len(no2), 4),
                              f"P_incomplete_at_{a.approved}": round(sum(inc) / len(inc), 4)})
            out["scenarios"].append({**extra, "session1_p95_gpu_h": round(2 * s1[int(0.95 * len(s1))], 2),"scenario": sname, "slope": sl_name, "base_window_min": round(base, 1), "mean_gpu_h": round(statistics.mean(tot), 2),
                                     "p50": q(0.5), "p90": q(0.9), "p95": q(0.95), "p99": q(0.99),
                                     "P_above_valid_run_ceiling": round(sum(t > ceil_gh for t in tot) / len(tot), 4),
                                     "P_above_maximum": round(sum(t > max_gh for t in tot) / len(tot), 4),
                                     "P_session1_above_its_ceiling": round(sum(h > ceil[0] for h in s1) / len(s1), 4)})
    # ---- option B (outcome-robust): valid-run ceiling = the pessimistic scenario's 95th percentile (S2, token-based slope),
    # rounded up to 0.5; invalid-run allowance = one session 1 at that scenario's 95th percentile; maximum = their sum.
    pes = [x for x in out["scenarios"] if x["scenario"].startswith("S2") and x["slope"] == "token-based"][0]
    cb = math.ceil(pes["p95"] * 2) / 2
    ab = math.ceil(pes["session1_p95_gpu_h"] * 2) / 2
    out["option_B"] = {"valid_run_ceiling_gpu_h": cb, "invalid_run_allowance_gpu_h": ab, "maximum_gpu_h": cb + ab,
                       "rule": "ceiling = p95 of S2 (0.9.1-like rates) with the token-based slope, rounded up to 0.5; allowance = one "
                               "session 1 at its p95 in that scenario; maximum = ceiling + allowance"}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print("option B:", json.dumps(out["option_B"]))
    r = out["reconstruction"]
    print(f"reconstruction: expected {r['expected_gpu_h']} GPU-h; valid-run ceiling {r['ceiling_gpu_h']}; maximum {r['maximum_gpu_h']}")
    for k in ("2026-09-29", "2026-10-10"):
        print(f"  session {k}: {r[k]}")
    print(f"slopes: empirical {out['model']['slope_empirical_min_per_cap']} min/cap, token-based {out['model']['slope_token_min_per_cap']} min/cap")
    for s in out["scenarios"]:
        print(json.dumps(s))


if __name__ == "__main__":
    main()
