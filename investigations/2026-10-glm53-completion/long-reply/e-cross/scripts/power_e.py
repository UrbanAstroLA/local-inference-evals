#!/usr/bin/env python3
"""[Recomputed under amendment 22ah for 13 prefixes; thresholds unchanged.]
E power simulation of the frozen screens with 13 prefixes (PROPOSAL.md section 5.3). CPU-only container (numpy), zero GPU.
Imports bplus_stats.contrast / eta2 (B+'s frozen file, unchanged), so the simulated screens are exactly the analysis's.

Generative model: B+'s (power_bplus.py), unchanged except the design size: prefix i has a latent logit state
a_i ~ N(mu, sigma^2); on engine e its continuation finishes within +64k with p_ie = expit(a_i + delta_e); each of 4 seeds is
an independent Bernoulli(p_ie); 13 prefixes, engines v070 and v091. mu and delta_e are calibrated so that the mean over
prefixes of p_ie equals the target rate (Gauss-Hermite quadrature).
Inputs (B+'s discovery inputs, unchanged): 0.7.0 +64k finishing from 100k = 0.55; 0.9.1 = 0.31 x 0.55 / 0.66 = 0.26; full
gap 0.29; half gap 0.145; no gap. Sensitivity: base 0.42 (B+'s observed level on 0.7.0-built prefixes, 0.417 on both
engines) and 0.385 (this sample's own expectation under a pure-state reading: 5 of the 13 originals finished within +64k).
Reported per scenario: P(R-E) = P(EFFECT(E1)); P(R-E-); P(SMALL(E1)); P(SMALL(E1) and PREFIX_HIGH); eta^2 median / 95th
percentile; median 95% interval width of E1; and, for the descriptive interaction E1 - C1 with B+'s observed C1 prefix
differences held fixed, the share of runs whose two-sample 95% bootstrap interval excludes 0.
Also: the exact sign-flip floor with 13 prefixes (2 / 8192) and with 12 (2 / 4096).
usage: power_e.py OUT_JSON [NSIM]
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/work/scripts")
import bplus_stats as BS  # noqa: E402

NSIM = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
B_SIM = 2000
NP, NJ = 13, 4
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


def bplus_c1_d():
    r = json.load(open("/work/ref/bplus-result-22v.json"))
    return np.array(r["primary"]["contrasts"]["C1"]["d"], dtype=np.float64)


def simulate(p070, p091, sigma, nsim, seed, dB):
    rng = np.random.default_rng(seed)
    mu = solve(p070, sigma)
    d091 = solve(p091, sigma, base=mu)
    n = {"EFFECT": 0, "REVERSE": 0, "SMALL": 0, "SMALL_and_PREFIX_HIGH": 0, "interaction_ci_excludes_0": 0}
    eta, width = [], []
    for s in range(nsim):
        a = mu + sigma * rng.standard_normal(NP)
        Y = {e: (rng.random((NP, NJ)) < expit(a + dl)[:, None]).astype(float) for e, dl in (("v070", 0.0), ("v091", d091))}
        rA, rB = Y["v070"].mean(1), Y["v091"].mean(1)
        c = BS.contrast(rA, rB, B=B_SIM, seed=seed + s)
        e2 = BS.eta2(np.concatenate([Y["v070"], Y["v091"]], axis=1))
        n["EFFECT"] += c["EFFECT"]
        n["REVERSE"] += c["REVERSE"]
        n["SMALL"] += c["SMALL"]
        n["SMALL_and_PREFIX_HIGH"] += c["SMALL"] and e2 is not None and e2 >= BS.ETA_MIN
        eta.append(e2 if e2 is not None else 0.0)
        width.append(c["ci95"][1] - c["ci95"][0])
        dE = rA - rB
        ib = dE[rng.integers(0, NP, (B_SIM, NP))].mean(1) - dB[rng.integers(0, len(dB), (B_SIM, len(dB)))].mean(1)
        lo, hi = np.quantile(ib, 0.025), np.quantile(ib, 0.975)
        n["interaction_ci_excludes_0"] += (lo > 0) or (hi < 0)
    eta = np.array(eta)
    return {"p070": p070, "p091": round(p091, 4), "sigma": sigma, "nsim": nsim,
            "P(R-E)": round(n["EFFECT"] / nsim, 4), "P(R-E-)": round(n["REVERSE"] / nsim, 4),
            "P(SMALL E1)": round(n["SMALL"] / nsim, 4), "P(SMALL E1 and PREFIX_HIGH)": round(n["SMALL_and_PREFIX_HIGH"] / nsim, 4),
            "eta2_median": round(float(np.median(eta)), 3), "eta2_p95": round(float(np.quantile(eta, 0.95)), 3),
            "E1_ci95_width_median": round(float(np.median(width)), 3),
            "P(interaction 95% interval excludes 0 | B+ C1 fixed)": round(n["interaction_ci_excludes_0"] / nsim, 4)}


def main():
    P070, P091 = 0.55, round(0.31 * 0.55 / 0.66, 3)
    gap = P070 - P091
    dB = bplus_c1_d()
    out = {"what": "E power simulation of the frozen screens, 13 prefixes x 4 seeds x 2 engines (amendment 22ah)", "frozen": {
        "DELTA_E": BS.DELTA_E, "ALPHA": BS.ALPHA, "SMALL_MAX": BS.SMALL_MAX, "EQ_BOUND": BS.EQ_BOUND, "ETA_MIN": BS.ETA_MIN},
        "inputs": {"p070": P070, "p091_full_gap": P091, "gap": round(gap, 3), "bplus_C1_d": dB.tolist()},
        "bootstrap_draws_in_simulation": B_SIM, "signflip_floor": {"13 prefixes": 2 / 2 ** 13, "12 prefixes": 2 / 2 ** 12, "8 prefixes (first freeze)": 2 / 2 ** 8},
        "main": [], "sensitivity": []}
    seed = 9000
    for sigma in (0.0, 1.0, 2.5):
        for lab, p91 in (("full gap", P091), ("half gap", P070 - gap / 2), ("no gap", P070)):
            seed += 1
            r = simulate(P070, p91, sigma, NSIM, seed, dB)
            r["scenario"] = lab
            out["main"].append(r)
            print(json.dumps(r), flush=True)
    for base in (0.42, round(5 / 13, 3)):
        for sigma in (1.0, 2.5):
            for lab, p91 in ((f"full gap, base {base}", base - gap), (f"half gap, base {base}", base - gap / 2), (f"no gap, base {base}", base)):
                seed += 1
                r = simulate(base, p91, sigma, NSIM, seed, dB)
                r["scenario"] = lab
                out["sensitivity"].append(r)
                print(json.dumps(r), flush=True)
    try:
        bp = json.load(open("/work/ref/bplus-power.json"))
        out["bplus_12_prefix_power_for_comparison"] = [{"scenario": r["scenario"], "sigma": r["sigma"], "P(R-E)": r["reading_share"]["R-E"]}
                                                       for r in bp["main"]]
    except OSError:
        out["bplus_12_prefix_power_for_comparison"] = None
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
