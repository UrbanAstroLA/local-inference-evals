#!/usr/bin/env python3
"""Costing for option D (PROPOSAL.md section 8). CPU only, standard library, fixed seed, read only.

1. Reconstruction [R]: the 0.9.1 record's pass-1 session (2026-10-08) from cost/timing-reconstruction.json, written by a
   byte-identical copy of rep070/refreeze/reconstruct_timing.py (2 GPUs x eval-chain start -> container stopped).
2. Outcome dependence [I, model], as rep070's budget_rep070.py: a pass's request window grows with its cap exhaustions.
   window(n caps) = base + slope x n, two slopes:
     empirical: least squares over the 0.9.1 record's three passes (caps, window minutes from the request logs);
     token-based (conservative): a cap's 327,680 tokens at the record's mean aggregate throughput.
   Per-question cap probabilities: empirical-Bayes beta-binomial fitted (frozen power_col511.fit_bb) to the record's
   per-question cap counts (finish 'length' at 327,680, out of 3 passes) -> "0.9.1-like long-reply rate". A stress scenario
   doubles every question's cap probability. Pass-to-pass variation beyond caps: multiplicative normal factor per pass.
   Session overhead (start-up to first request + last request to stop): the record pass-1 session's [R]; pessimistic +15 min.
3. Ceiling = the pessimistic scenario's 99.9th percentile of occupied GPU-hours, rounded up to 0.5 = APPROVED_GPU_H
   required and the runner's PLAN_MAX_GPU_H. P(ceiling) = P(occupied >= ceiling) per scenario: the probability that the
   watchdog stops the run and it is reported INCOMPLETE (never classified). Seeds, cap and requests are never changed.
4. Raw-body disk estimate from the record's per-pass character counts and rep070-F's measured gzip ratio.
usage: cost_d091.py [--sims 50000] [--seed 20261021] [--out cost.json]
"""
import argparse, json, math, os, random, statistics, sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "frozen"))
from power_col511 import fit_bb, N_PASSES  # noqa: E402

G = "<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell"
REC_LOGS = {1234: f"{G}+v0.9.1.requests.jsonl", 1235: f"{G}+v0.9.1+ps.requests.jsonl", 1236: f"{G}+v0.9.1+ps.requests.jsonl"}
CAP_TOK, N_Q = 327680, 198
RAW_F_S1 = "<runs>/rep070/run-F/raw/s1.raw.jsonl.gz"
REPF_S1 = f"{G}+rep070-F.s1.requests.jsonl"
PESS_EXTRA_OVERHEAD_MIN = 15.0


def passes():
    out = []
    for seed, path in REC_LOGS.items():
        R = [json.loads(l) for l in open(path) if l.strip()]
        R = [r for r in R if r["seed"] == seed]
        t0 = min(r["t_start"] for r in R); t1 = max(r["t_start"] + r["secs"] for r in R)
        out.append({"seed": seed, "caps": sum(r["finish_reason"] == "length" and r["completion_tokens"] == CAP_TOK for r in R),
                    "window_min": round((t1 - t0) / 60, 2), "tokens": sum(r["completion_tokens"] for r in R),
                    "agg_tok_s": round(sum(r["completion_tokens"] for r in R) / (t1 - t0), 1),
                    "chars": sum((r.get("reasoning_chars") or 0) + (r.get("content_chars") or 0) for r in R),
                    "max_request_s": round(max(r["secs"] for r in R), 1), "requests": len(R),
                    "cap_by_q": R})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tim = json.load(open(os.path.join(HERE, "timing-reconstruction.json")))
    s1 = [s for s in tim["sessions"] if s["record"].startswith("0.9.1 K3.25 DFlash2 x3") and s["session"] == "2026-10-08"][0]
    P = passes()
    # per-question cap counts over the record's 3 passes (messages hash identifies the question)
    qs = sorted({r["messages_sha256"] for r in P[0]["cap_by_q"]})
    assert len(qs) == N_Q
    k = []
    for q in qs:
        k.append(sum(1 for p in P for r in p["cap_by_q"] if r["messages_sha256"] == q and r["finish_reason"] == "length" and r["completion_tokens"] == CAP_TOK))
    for p in P:
        p.pop("cap_by_q")
    fit = fit_bb(k)
    post = [(fit["a"] + x, fit["b"] + N_PASSES - x) for x in k]
    xs, ys = [p["caps"] for p in P], [p["window_min"] for p in P]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope_emp = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    agg = statistics.mean(p["agg_tok_s"] for p in P)
    slope_tok = CAP_TOK / agg / 60
    win1 = (dt(s1["last_request_end"]) - dt(s1["first_request"])) / 60
    over = s1["wall_h"] * 60 - win1
    out = {"reconstruction": {"record_pass1_session": {k_: s1[k_] for k_ in ("start", "ready", "first_request", "last_request_end", "stop", "wall_h", "gpu_hours")},
                              "request_window_min": round(win1, 2), "overhead_min": round(over, 2),
                              "record_passes": P, "B+_v091_ready_after_s": 680},
           "model": {"slope_empirical_min_per_cap": round(slope_emp, 2), "slope_token_min_per_cap": round(slope_tok, 2),
                     "agg_tok_s_mean": round(agg, 1), "cap_counts_hist": {str(v): k.count(v) for v in range(4)}, "caps_total": sum(k),
                     "fit": fit, "expected_caps_per_pass": round(sum(x / (x + y) for x, y in post), 2)},
           "scenarios": []}
    scen = [("central: 0.9.1-like cap rates, empirical slope, sd 5%, record overhead", 1.0, max(slope_emp, 0.0), 0.05, over),
            ("pessimistic: 0.9.1-like cap rates, max(empirical, token) slope, sd 10%, overhead +15 min", 1.0, max(slope_emp, slope_tok), 0.10, over + PESS_EXTRA_OVERHEAD_MIN),
            ("stress: cap rates x2, max(empirical, token) slope, sd 10%, overhead +15 min", 2.0, max(slope_emp, slope_tok), 0.10, over + PESS_EXTRA_OVERHEAD_MIN)]
    draws = {}
    for i, (name, mult, slope, sd, ov) in enumerate(scen):
        base = statistics.mean(w - slope * c for c, w in zip(xs, ys))
        rng = random.Random(a.seed + i)
        tot = []
        ncap = []
        for _ in range(a.sims):
            n = 0
            for al, be in post:
                pq = min(1.0, mult * rng.betavariate(al, be))
                n += rng.random() < pq
            w = max(0.0, (base + slope * n) * rng.gauss(1, sd))
            tot.append(2 * (ov + w) / 60)
            ncap.append(n)
        draws[name] = tot
        tot_s = sorted(tot)
        q = lambda f: round(tot_s[min(len(tot_s) - 1, int(f * len(tot_s)))], 2)
        out["scenarios"].append({"scenario": name, "cap_multiplier": mult, "slope_min_per_cap": round(slope, 2), "pass_sd": sd,
                                 "overhead_min": round(ov, 1), "base_window_min": round(base, 1),
                                 "mean_caps": round(statistics.mean(ncap), 2), "p99_caps": sorted(ncap)[int(0.99 * len(ncap))],
                                 "mean_gpu_h": round(statistics.mean(tot), 2), "p50": q(0.5), "p90": q(0.9), "p95": q(0.95),
                                 "p99": q(0.99), "p999": q(0.999), "max": round(tot_s[-1], 2)})
    cen, pes = out["scenarios"][0], out["scenarios"][1]
    ceiling = math.ceil(pes["p999"] * 2) / 2
    for s_ in out["scenarios"]:
        s_["P_ge_ceiling"] = round(sum(t >= ceiling for t in draws[s_["scenario"]]) / a.sims, 5)
    out["summary"] = {"central_gpu_h": {"record_pass1_session_reconstructed": s1["gpu_hours"], "model_p50": cen["p50"]},
                      "pessimistic_gpu_h": pes["p95"], "ceiling_gpu_h": ceiling,
                      "APPROVED_GPU_H_required": ceiling,
                      "P_ceiling": {s_["scenario"].split(":")[0]: s_["P_ge_ceiling"] for s_ in out["scenarios"]},
                      "rule": "ceiling = pessimistic scenario p99.9 rounded up to 0.5 GPU-h; no invalid-run allowance (an invalid or "
                              "incomplete run means stop and amend; no retry)"}
    # raw-body disk estimate
    rs1 = [json.loads(l) for l in open(REPF_S1) if l.strip()]
    chars_f = sum((r.get("reasoning_chars") or 0) + (r.get("content_chars") or 0) for r in rs1)
    gz = os.path.getsize(RAW_F_S1)
    ratio = gz / chars_f
    mx_chars = max(p["chars"] for p in P)
    worst = N_Q * CAP_TOK * 3.5
    out["disk"] = {"rep070F_s1_raw_bytes": gz, "rep070F_s1_chars": chars_f, "gzip_bytes_per_char_measured": round(ratio, 4),
                   "central_bytes": int(statistics.mean(p["chars"] for p in P) * ratio),
                   "pessimistic_bytes": int(mx_chars * 0.45),
                   "ceiling_bytes_all_198_capped_3.5_chars_per_token_0.45": int(worst * 0.45),
                   "note": "raw file only (PRIVATE/raw, 0600); plus request log ~0.1 MB, server log ~1 MB, lm-eval samples ~12 MB; "
                           "the runner refuses to start with < 5 GiB free on /data"}
    out["timeout_margin"] = {"lm_eval_timeout_s": 10800, "record_max_request_s": max(p["max_request_s"] for p in P),
                             "note": "a request over 10,800 s would be re-sent by lm-eval (max_retries 3) and make the run INVALID ('no re-sent request')"}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    S = out["summary"]
    print(f"reconstruction [R]: 0.9.1 record pass-1 session {s1['wall_h']} h wall = {s1['gpu_hours']} GPU-h; window {win1:.1f} min; overhead {over:.1f} min")
    print(f"record passes [M]: " + "; ".join(f"seed {p['seed']}: {p['caps']} caps, {p['window_min']} min, {p['agg_tok_s']} tok/s, max request {p['max_request_s']} s" for p in P))
    print(f"slopes: empirical {slope_emp:.2f} min/cap, token-based {slope_tok:.2f} min/cap; expected caps per pass {out['model']['expected_caps_per_pass']}")
    for s_ in out["scenarios"]:
        print(json.dumps(s_))
    print(f"central {S['central_gpu_h']}; pessimistic {S['pessimistic_gpu_h']}; ceiling {S['ceiling_gpu_h']} (APPROVED_GPU_H required); P(ceiling) {S['P_ceiling']}")
    print(f"disk: {json.dumps(out['disk'])}")
    return 0


def dt(s):
    import datetime
    return datetime.datetime.fromisoformat(s).timestamp()


if __name__ == "__main__":
    sys.exit(main())
