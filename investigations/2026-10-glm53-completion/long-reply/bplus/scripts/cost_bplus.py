#!/usr/bin/env python3
"""B+ costing from measured rates (PROPOSAL.md section 8). Host, standard library, zero GPU. Writes out/cost.json.

Session model (one engine): wall = start-up to ready + all prefill (treated as stalling every slot: conservative) +
decode makespan of 48 continuations on 8 slots, list-scheduled in the plan's fixed order + shutdown. Occupied
GPU-hours = 2 x wall. A continuation decodes min(finish time, 65,536) tokens at the session's per-request rate (with
+-15% per-request lognormal scatter). Finish times: within +64k with probability p_e, then uniform over the horizon
(rep-F: 0.24 by +32k and 0.55 by +64k, i.e. 44% of the +64k finishers in the first half); otherwise 65,536.

Measured inputs [M] (sources in INPUTS):
  decode, per request, >= 7 requests running, total batch context < 200k tokens (low-context bin: least loop-inflated
    speculative acceptance), from server status lines joined to the request logs (scratch analysis, numbers below):
    0.7.0 32.7, 0.9.1 52.9, B1 52.3 tok/s; slowest lifetime rate of a finished >= 100k reply: 0.7.0 28.0, 0.9.1 47.2,
    B0 46.9 (no finished >= 100k B1 reply); B1 forcehook+mapguard image at 8 in flight on ~92k-token contexts: 36.5
    (median) / 29.6 (min) [instrumented].
  start-up to ready (first timestamped server line to the release-ready line): 0.7.0 464-493 s; 0.9.1 693-696 s; B1 675 s;
    d7 B1 session 732 s.
  prefill: B1-family cold prefill 2,770 tok/s (fcr1), pessimistic 2,410; 0.7.0 long-prompt prefill not measured.
Cost levels: central = median session at central inputs; pessimistic = 90th percentile at pessimistic inputs (no cache
  hits); ceiling = 99th percentile of the FULL-horizon session (every continuation runs 65,536 tokens) at pessimistic
  inputs, summed over engines plus inter-session gaps and rounded up to 0.1 GPU-h; P(exceed) from a run with the session
  decode rate lognormal (median central, 5th percentile pessimistic), prefill likewise, start-up up to +25%.
Assumed [I]: central decode = the low-context bin rate (sparse attention: per-token cost assumed flat in context); pessimistic
  = 0.8 x the slowest finished-long lifetime rate (0.7.0 22.4, 0.9.1 37.8, B1 37.5), which also lies at or below the
  instrumented B1 measurement at 8 x 92k; 0.7.0 prefill central 2,410 / pessimistic 1,200 tok/s.
  Finishing (central): 0.7.0 0.55, 0.9.1 0.26, B1 0.26; pessimistic: 0.26 everywhere.
usage: cost_bplus.py OUT_JSON
"""
import json
import math
import random
import sys

HERE = sys.path[0]
PKG = HERE.rsplit("/", 1)[0]
H = 65536
SLOTS = 8
GPUS = 2
SHUTDOWN = 60.0
INTER_SESSION = 120.0     # GPU release wait and pre-flight between sessions
INPUTS = {
    "v070": dict(dec_c=32.7, dec_p=22.4, pre_c=2410.0, pre_p=1200.0, start_c=540.0, start_p=810.0, p_c=0.55),
    "v091": dict(dec_c=52.9, dec_p=37.8, pre_c=2770.0, pre_p=2410.0, start_c=760.0, start_p=1140.0, p_c=0.26),
    "b1": dict(dec_c=52.3, dec_p=37.5, pre_c=2770.0, pre_p=2410.0, start_c=760.0, start_p=1140.0, p_c=0.26)}
P_PESS = 0.26
BLOCK = {"v070": 3584, "v091": 4352, "b1": 4352}
SOURCES = ["rep070/run-F/server-s{1,2}.log + rep070-F request logs; gpqa +v0.9.1(.ps).server.log + request logs; "
           "b1-screen/run-s1236/server-s1.log + B1 request log (status-line join, scratch script, numbers quoted)",
           "long-reply/discovery/out/lengths_all.jsonl (T / secs of finished >= 100k replies)",
           "first-contrast/retry/run-fcr1/client.jsonl (C/D runs: 8 in flight, ~92k contexts, B1 forcehook-mg2)",
           "first-contrast/retry/cost/cost_fcr.py (prefill 2,770 / 2,410; start-up 732 s)"]


def hit_of(L, B):
    return max(0, (L - 1) // B * B - B)


def prefill_tokens(plan, e, hits):
    seen, tot = set(), 0
    for r in plan["order"]:
        L = r["prefix_len"]
        tot += L - (hit_of(L, BLOCK[e]) if (hits and r["prefix_id"] in seen) else 0)
        seen.add(r["prefix_id"])
    return tot


def makespan(durs):
    slots = [0.0] * SLOTS
    for d in durs:            # plan order; the next request takes the earliest free slot
        i = min(range(SLOTS), key=lambda k: slots[k])
        slots[i] += d
    return max(slots)


def session(plan, e, rate, p_fin, pre_rate, start, hits, rng=None, scatter=0.15, full=False):
    durs = []
    for _ in plan["order"]:
        if full:
            tok = H
        else:
            tok = rng.uniform(1000, H) if rng.random() < p_fin else H
        r = rate * (math.exp(rng.gauss(0, scatter)) if (rng and scatter) else 1.0)
        durs.append(tok / r)
    pre = prefill_tokens(plan, e, hits) / pre_rate
    wall = start + pre + makespan(durs) + SHUTDOWN
    return wall, {"startup_s": start, "prefill_s": round(pre, 1), "decode_makespan_s": round(wall - start - pre - SHUTDOWN, 1)}


def gh(s):
    return GPUS * s / 3600.0


def main():
    plan = json.load(open(f"{PKG}/out/plan.json"))
    rng = random.Random(20261016)
    out = {"what": "B+ cost model (occupied GPU-hours = 2 x wall)", "inputs": INPUTS, "p_pess": P_PESS, "sources": SOURCES,
           "prefill_tokens": {e: {"with_cache_hits": prefill_tokens(plan, e, True), "no_hits": prefill_tokens(plan, e, False)} for e in INPUTS},
           "engines": {}}
    NS = 4000
    tot_c, tot_p, tot_ceil = 0.0, 0.0, 0.0
    sims_total = [0.0] * NS
    for e, m in INPUTS.items():
        cen = sorted(session(plan, e, m["dec_c"], m["p_c"], m["pre_c"], m["start_c"], True, rng)[0] for _ in range(NS))
        pes = sorted(session(plan, e, m["dec_p"], P_PESS, m["pre_p"], m["start_p"], False, rng)[0] for _ in range(NS))
        ceil_sims = sorted(session(plan, e, m["dec_p"], 0.0, m["pre_p"], m["start_p"], False, rng, full=True)[0] for _ in range(NS))
        ceil_w = ceil_sims[int(NS * 0.99)]
        _, ceil_parts = session(plan, e, m["dec_p"], 0.0, m["pre_p"], m["start_p"], False, None, scatter=0, full=True)
        _, cen_parts = session(plan, e, m["dec_c"], m["p_c"], m["pre_c"], m["start_c"], True, random.Random(1))
        # uncertainty run for P(exceed): session rate lognormal (median central, 5th percentile = pessimistic), outcomes at
        # the central finishing rate, cache hits as planned with prob 0.8 else none
        sd = math.log(m["dec_c"] / m["dec_p"]) / 1.645
        unc = []
        for i in range(NS):
            rate = m["dec_c"] * math.exp(rng.gauss(0, sd))
            pre = m["pre_c"] * math.exp(rng.gauss(0, math.log(m["pre_c"] / m["pre_p"]) / 1.645))
            st = m["start_c"] * math.exp(abs(rng.gauss(0, 0.25)))
            w = session(plan, e, rate, m["p_c"], pre, st, rng.random() < 0.8, rng)[0]
            unc.append(w)
            sims_total[i] += w + INTER_SESSION
        c50, p90 = cen[NS // 2], pes[int(NS * 0.9)]
        out["engines"][e] = {"central_wall_s": round(c50), "central_gpu_h": round(gh(c50), 2), "central_parts": cen_parts,
                             "pessimistic_wall_s": round(p90), "pessimistic_gpu_h": round(gh(p90), 2),
                             "ceiling_wall_s": round(ceil_w), "ceiling_gpu_h": round(gh(ceil_w), 2), "ceiling_parts": ceil_parts,
                             "uncertainty_p50_gpu_h": round(gh(sorted(unc)[NS // 2]), 2),
                             "uncertainty_p95_gpu_h": round(gh(sorted(unc)[int(NS * 0.95)]), 2),
                             "P_session_exceeds_its_ceiling": round(sum(w > ceil_w for w in unc) / NS, 4)}
        tot_c += c50 + INTER_SESSION
        tot_p += p90 + INTER_SESSION
        tot_ceil += ceil_w + INTER_SESSION
    ceiling_gh = math.ceil(gh(tot_ceil) * 10) / 10
    out["total"] = {"central_gpu_h": round(gh(tot_c), 2), "pessimistic_gpu_h": round(gh(tot_p), 2),
                    "ceiling_gpu_h": ceiling_gh, "ceiling_wall_s": int(ceiling_gh * 3600 / GPUS),
                    "P_run_exceeds_ceiling": round(sum(gh(w) > ceiling_gh for w in sims_total) / NS, 4),
                    "uncertainty_p50_gpu_h": round(gh(sorted(sims_total)[NS // 2]), 2),
                    "uncertainty_p95_gpu_h": round(gh(sorted(sims_total)[int(NS * 0.95)]), 2),
                    "session_min_wall_s": {e: out["engines"][e]["ceiling_wall_s"] for e in INPUTS},
                    "earlier_estimate_discovery_T14": {"central": 7.8, "pessimistic": 10.3,
                                                       "basis": "concurrency 16, 3 waves, lifetime decode 41 / 67 / 74 tok/s"}}
    # sensitivity: concurrency 16 (not the frozen design; for the operator's information)
    global SLOTS
    SLOTS = 16
    out["sensitivity_concurrency_16_central_gpu_h"] = round(sum(gh(session(plan, e, m["dec_c"] * 0.8, m["p_c"], m["pre_c"], m["start_c"], True,
                                                                           random.Random(5))[0] + INTER_SESSION) for e, m in INPUTS.items()), 2)
    SLOTS = 8
    # disk: raw continuation records (text + generated ids), measured per token in the session simulation
    out["disk"] = {"raw_bytes_per_generated_token_upper": 12, "requests": 144, "max_generated_tokens": 144 * H,
                   "raw_upper_bound_MB_uncompressed": round(144 * H * 12 / 1e6, 1),
                   "note": "upper bound: ~4 chars of text + ~7 bytes of JSON id per token; gzip typically 3-5x smaller; "
                           "server logs ~1-2 MB per session; prompt ids are not stored again (hash only)"}
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for e, v in out["engines"].items():
        print(e, {k: v[k] for k in ("central_gpu_h", "pessimistic_gpu_h", "ceiling_gpu_h", "uncertainty_p95_gpu_h", "P_session_exceeds_its_ceiling")},
              "central parts", v["central_parts"])
    print("total", out["total"])
    print("concurrency-16 sensitivity (central, rate x0.8):", out["sensitivity_concurrency_16_central_gpu_h"])
    print("prefill tokens", out["prefill_tokens"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
