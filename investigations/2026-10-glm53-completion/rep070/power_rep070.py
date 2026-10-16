#!/usr/bin/env python3
"""DRAFT (not frozen). Operating characteristics of the proposed reading rules for a repeatability run of stock 0.7.0
(PROPOSAL.md section 6). CPU only, standard library, fixed seed. Reuses, unchanged and by import, the
frozen test (analysis_col511.signflip, sha256 b5b13ce2...) and the beta-binomial model of power_col511.py (fit_bb,
counts; sha256 92466804...).

Design: conditional on BOTH historical records (the 0.9.1 record, 16 empties on 11 questions; the old 0.7.0 record,
4 on 4), as the real analysis will be. Only the new 0.7.0 run ('rep') is simulated: per-question rates p_q drawn from a
posterior, then three passes independent given p_q. Limitation: the run repeats the recorded seeds on the same engine,
so its draws are not independent of the old 0.7.0 record (shared sampler noise until batching makes trajectories
diverge); the model ignores that dependence, which can only make rep resemble old 0.7.0 more than modelled.

usage: power_rep070.py [--sims 20000] [--seed 20261012] [--out power-rep070.json]
"""
import argparse, hashlib, json, math, os, random, sys

FROZEN = "<scratch>/kdcp/arm-col511"
PINS = {"analysis_col511.py": "b5b13ce2155085ca46fb24877f12b7cd0a5cdd16eb4cf19012958e7b86ae0033",
        "power_col511.py": "92466804b44bd29e49f50ffeb925a0974b1c13fbdb17b48b5c307fabb546fb47"}
for f, h in PINS.items():
    assert hashlib.sha256(open(os.path.join(FROZEN, f), "rb").read()).hexdigest() == h, f"{f} changed since freeze"
sys.path.insert(0, FROZEN)
from analysis_col511 import signflip, N_Q, N_PASSES  # noqa: E402
from power_col511 import counts, fit_bb, CMP_ROWS, V070_ROWS  # noqa: E402

ALPHA = 0.05
O1_MAX, O2_MIN = 8, 12          # PROPOSED operational thresholds on rep's empties of 594 (old 0.7.0: 4; 0.9.1: 16)


def read(n_rep, p1, p2):
    """Proposed reading rules, evaluated in order. p1: one-sided, rep fewer empties than the 0.9.1 record.
    p2: one-sided, rep more empties than the old 0.7.0 record."""
    if n_rep <= O1_MAX and p1 < ALPHA:
        return "O1"   # substantially fewer empties again: replication support
    if n_rep >= O2_MIN or p2 < ALPHA:
        return "O2"   # movement toward the newer engine's count: original gap reported as not reproduced
    return "O3"       # ambiguous: report uncertainty, stop this allocation


def simulate(draw, k091, k070, rng, sims):
    R = {"O1": 0, "O2": 0, "O3": 0, "p1_sig": 0, "p2_sig": 0, "n_total": 0, "n_le8": 0, "n_9_11": 0, "n_ge12": 0}
    for _ in range(sims):
        p = draw(rng)
        k = [sum(rng.random() < x for _ in range(N_PASSES)) for x in p]
        n = sum(k)
        t1 = signflip([c - a for c, a in zip(k091, k)])          # rep fewer than 0.9.1
        t2 = signflip([a - o for a, o in zip(k, k070)])          # rep more than old 0.7.0
        R[read(n, t1["p_one_sided"], t2["p_one_sided"])] += 1
        R["p1_sig"] += t1["p_one_sided"] < ALPHA
        R["p2_sig"] += t2["p_one_sided"] < ALPHA
        R["n_total"] += n
        R["n_le8"] += n <= O1_MAX; R["n_9_11"] += O1_MAX < n < O2_MIN; R["n_ge12"] += n >= O2_MIN
    out = {}
    for key, v in R.items():
        if key == "n_total":
            out["mean_rep_empties"] = round(v / sims, 2)
        else:
            q = v / sims
            out["P_" + key] = round(q, 4); out["mc_se_" + key] = round(math.sqrt(q * (1 - q) / sims), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    k091, k070 = counts(CMP_ROWS), counts(V070_ROWS)
    f091, f070 = fit_bb(k091), fit_bb(k070)
    post091 = [(f091["a"] + x, f091["b"] + N_PASSES - x) for x in k091]
    post070a = [(f070["a"] + x, f070["b"] + N_PASSES - x) for x in k070]
    a7b, b7b = f070["pooled_rate"] * f091["s"], (1 - f070["pooled_rate"]) * f091["s"]
    post070b = [(a7b + x, b7b + N_PASSES - x) for x in k070]
    beta = lambda post: (lambda r: [r.betavariate(x, y) for x, y in post])
    scen = [
        ("S1a: rep = old 0.7.0 rates (EB, own fit: pooled 0.0067 on every question)", beta(post070a)),
        ("S1b: rep = old 0.7.0 rates (0.7.0 mean, 0.9.1 fit's concentration)", beta(post070b)),
        ("S2: rep = 0.9.1 record's rates (the old gap was chance or drift)", beta(post091)),
        ("S3: rep halfway, p_q = (0.7.0 S1b + 0.9.1) / 2", lambda r: [0.5 * (r.betavariate(*u) + r.betavariate(*v)) for u, v in zip(post070b, post091)]),
        ("S4: rep = 0.25 x 0.9.1 rates (0.7.0's overall rate on 0.9.1's question profile)", lambda r: [0.25 * r.betavariate(x, y) for x, y in post091]),
    ]
    out = {"model": {"fit_0.9.1": f091, "fit_0.7.0": f070, "fit_0.7.0_variant_b": {"a": a7b, "b": b7b},
                     "k091_total": sum(k091), "k070_total": sum(k070)},
           "rules": {"O1": f"rep empties <= {O1_MAX} and p1 < {ALPHA}", "O2": f"else rep empties >= {O2_MIN} or p2 < {ALPHA}", "O3": "otherwise",
                     "p1": "exact one-sided sign-flip, d_q = e_q(0.9.1 record) - e_q(rep)",
                     "p2": "exact one-sided sign-flip, d_q = e_q(rep) - e_q(old 0.7.0 record)"},
           "design": "conditional on both historical records", "seed": a.seed, "sims": a.sims, "results": []}
    for i, (name, f) in enumerate(scen):
        r = simulate(f, k091, k070, random.Random(a.seed + i), a.sims)
        out["results"].append({"scenario": name, **r})
        print(f"{name}: mean {r['mean_rep_empties']}; P(O1) {r['P_O1']}, P(O2) {r['P_O2']}, P(O3) {r['P_O3']}; "
              f"P(p1<.05) {r['P_p1_sig']}, P(p2<.05) {r['P_p2_sig']}; P(n<=8/9-11/>=12) {r['P_n_le8']}/{r['P_n_9_11']}/{r['P_n_ge12']}", flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
