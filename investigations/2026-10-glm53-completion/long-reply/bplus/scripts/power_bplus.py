#!/usr/bin/env python3
"""B+ power simulation for the frozen screens (PROPOSAL.md section 5.3). CPU-only container (numpy), zero GPU.
Imports bplus_stats.screens, so the simulated readings use exactly the analysis's rules.

Generative model (prefix = unit): prefix i has a latent state a_i ~ N(mu, sigma^2) on the logit scale; on engine e its
continuation finishes within +64k with p_ie = expit(a_i + delta_e); each of the 4 seeds is an independent Bernoulli(p_ie).
sigma = how strongly the 100k state fixes the outcome (0: not at all; 2.5: strongly). mu and delta_e are calibrated so
that the mean over prefixes of p_ie equals the target rate (Gauss-Hermite quadrature).
Discovery inputs: +64k finishing from 100k on stock 0.7.0, at risk: 0.55 (16 / 29, rep-F); 0.9.1 family, to the cap,
question-matched: 18 / 58 = 0.31 against 0.7.0 23 / 36 = 0.64 (T11); scaled to +64k by 0.7.0's own +64k / to-cap ratio
(0.55 / 0.66): 0.31 x 0.836 = 0.26. The full gap if all of it were continuation: 0.55 - 0.26 = 0.29.
Scenarios: gap full / half / none, for sigma 0 / 1 / 2.5; B1 = 0.9.1 unless stated. Sensitivity: the selected sample's
own expected 0.7.0 rate if the state dominated (5 of 12 originals finished within +64k: 0.42).
usage: power_bplus.py OUT_JSON [NSIM]
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/work/scripts")
import bplus_stats as BS  # noqa: E402

NSIM = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
B_SIM = 2000
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(80)
GH_W = GH_W / GH_W.sum()


def expit(x):
    return 1 / (1 + np.exp(-x))


def mean_p(mu, sigma):
    return float((GH_W * expit(mu + sigma * GH_X)).sum())


def solve(target, sigma, base=0.0):
    lo, hi = -30.0, 30.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if mean_p(base + mid, sigma) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def simulate(p070, p091, pb1, sigma, nsim, seed, eta_min=None, delta_e=None):
    rng = np.random.default_rng(seed)
    mu = solve(p070, sigma)
    d091 = solve(p091, sigma, base=mu)
    db1 = solve(pb1, sigma, base=mu)
    counts = {"R-E": 0, "R-E-": 0, "R-S": 0, "R-I": 0}
    c2 = {"EFFECT": 0, "REVERSE": 0, "SMALL": 0, "unresolved": 0}
    eta = []
    old = (BS.ETA_MIN, BS.DELTA_E)
    if eta_min is not None:
        BS.ETA_MIN = eta_min
    if delta_e is not None:
        BS.DELTA_E = delta_e
    try:
        for s in range(nsim):
            a = mu + sigma * rng.standard_normal(12)
            Y = {}
            for e, dl in (("v070", 0.0), ("v091", d091), ("b1", db1)):
                Y[e] = (rng.random((12, 4)) < expit(a + dl)[:, None]).astype(float)
            r = BS.screens(Y, B=B_SIM, seed=seed + s)
            counts[r["reading"]] += 1
            c2[r["secondary"]["C2"]] += 1
            eta.append(r["eta2_prefix"] if r["eta2_prefix"] is not None else 0.0)
    finally:
        BS.ETA_MIN, BS.DELTA_E = old
    eta = np.array(eta)
    return {"p070": p070, "p091": p091, "pb1": pb1, "sigma": sigma, "nsim": nsim,
            "reading_share": {k: round(v / nsim, 4) for k, v in counts.items()},
            "C2_share": {k: round(v / nsim, 4) for k, v in c2.items()},
            "eta2_median": round(float(np.median(eta)), 3), "eta2_p95": round(float(np.quantile(eta, 0.95)), 3)}


def main():
    P070, P091 = 0.55, round(0.31 * 0.55 / 0.66, 3)
    gap = P070 - P091
    out = {"what": "B+ power simulation of the frozen screens", "frozen": {"DELTA_E": BS.DELTA_E, "ALPHA": BS.ALPHA,
           "SMALL_MAX": BS.SMALL_MAX, "EQ_BOUND": BS.EQ_BOUND, "ETA_MIN": BS.ETA_MIN},
           "inputs": {"p070": P070, "p091_full_gap": P091, "gap": round(gap, 3)}, "bootstrap_draws_in_simulation": B_SIM,
           "main": [], "sensitivity": [], "threshold_choice": []}
    seed = 7000
    for sigma in (0.0, 1.0, 2.5):
        for lab, p91 in (("full gap", P091), ("half gap", P070 - gap / 2), ("no gap", P070)):
            seed += 1
            r = simulate(P070, p91, p91, sigma, NSIM, seed)
            r["scenario"] = lab
            out["main"].append(r)
            print(json.dumps(r))
    # B1 closes the gap (fix effect) at sigma 1: power of the C2 EFFECT screen
    seed += 1
    r = simulate(P070, P091, P070, 1.0, NSIM, seed)
    r["scenario"] = "full gap, B1 = 0.7.0 (fixes close it)"
    out["sensitivity"].append(r)
    print(json.dumps(r))
    # lower base rate (the selected sample if state dominated)
    for sigma in (1.0, 2.5):
        for lab, p70, p91 in (("full gap, base 0.42", 0.42, 0.42 - gap), ("no gap, base 0.42", 0.42, 0.42)):
            seed += 1
            r = simulate(p70, p91, p91, sigma, NSIM, seed)
            r["scenario"] = lab
            out["sensitivity"].append(r)
            print(json.dumps(r))
    # threshold choice: DELTA_E and ETA_MIN alternatives (fewer simulations)
    n2 = max(500, NSIM // 4)
    for de in (0.10, 0.15, 0.20):
        for lab, p91, sg in (("full gap", P091, 1.0), ("half gap", P070 - gap / 2, 1.0), ("no gap", P070, 1.0), ("no gap", P070, 0.0)):
            seed += 1
            r = simulate(P070, p91, p91, sg, n2, seed, delta_e=de)
            out["threshold_choice"].append({"DELTA_E": de, "scenario": lab, "sigma": sg, "P(R-E)": r["reading_share"]["R-E"]})
    for em in (0.20, 0.30, 0.40):
        for lab, p91, sg in (("no gap", P070, 2.5), ("no gap", P070, 1.0), ("no gap", P070, 0.0), ("full gap", P091, 2.5)):
            seed += 1
            r = simulate(P070, p91, p91, sg, n2, seed, eta_min=em)
            out["threshold_choice"].append({"ETA_MIN": em, "scenario": lab, "sigma": sg, "P(R-S)": r["reading_share"]["R-S"],
                                            "eta2_median": r["eta2_median"]})
    for t in out["threshold_choice"]:
        print(json.dumps(t))
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
