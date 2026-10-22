#!/usr/bin/env python3
"""Power of the frozen primary test (analysis_col511.signflip, one-sided, alpha 0.05) for the column-511 GPQA arm.
CPU only, standard library, fixed seed. Model and design: analysis-col511.md section 6.

Per-question empty probabilities p_q: empirical-Bayes beta-binomial. A Beta(a, b) prior is fitted by maximum marginal
likelihood to the per-question empty counts k_q (of 3 passes) of a record; question q's rate is then uncertain with
posterior Beta(a + k_q, b + 3 - k_q). Each simulation draws every p_q from that posterior (question-level clustering:
the same p_q governs all three passes of q), then draws the three passes independently given p_q.

Designs:
  conditional (primary): the comparator is the observed historical record, fixed; only the arm is simulated. This is the
      operating characteristic of the experiment that will actually be analysed (predictive power given the comparator).
  resample-both (reference): comparator and arm are both drawn from the model with the same p_q draw, i.e. the power of a
      hypothetical replication with a contemporaneous comparator of the same size.

usage: power_col511.py [--sims 20000] [--seed 20261011] [--out power.json]
"""
import argparse, json, math, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_col511 import signflip, classify, ALPHA, N_Q, N_PASSES, BAND_LARGE_MAX, BAND_MODERATE_MAX  # noqa: E402

RUNS = "<this-repo>/runs"
CMP_ROWS = f"{RUNS}/2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3_gpqa-diamond/results.jsonl"
V070_ROWS = f"{RUNS}/2026-09-29_glm53-flash_k3.25-v0.7.0-dflash5_gpqa-diamond/results.jsonl"


def counts(path):
    k = [0] * N_Q
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            k[r["doc_id"]] += bool(r["empty"])
    return k


def lbeta(x, y):
    return math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)


def bb_loglik(k, mu, s, n=N_PASSES):
    a, b = mu * s, (1 - mu) * s
    return sum(math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(n - x + 1) + lbeta(x + a, n - x + b) - lbeta(a, b) for x in k)


def fit_bb(k, s_max=1e5):
    """Maximum marginal likelihood over (mu, s = a + b): log-spaced grid, then coordinate refinement. s is bounded at s_max;
    a fit at the bound means no overdispersion is detectable (the posterior then equals the pooled rate for every question)."""
    tot = sum(k) / (len(k) * N_PASSES)
    best = None
    for i in range(-40, 41):
        mu = min(0.5, max(1e-5, tot * math.exp(i * 0.05)))
        for j in range(0, 101):
            s = 10 ** (-2 + j * 0.07)
            if s > s_max:
                continue
            ll = bb_loglik(k, mu, s)
            if best is None or ll > best[0]:
                best = (ll, mu, s)
    ll, mu, s = best
    for _ in range(60):  # local refinement
        improved = False
        for dmu, ds in ((1.01, 1), (1 / 1.01, 1), (1, 1.05), (1, 1 / 1.05)):
            m2, s2 = mu * dmu, min(s_max, s * ds)
            if 0 < m2 < 1:
                l2 = bb_loglik(k, m2, s2)
                if l2 > ll + 1e-12:
                    ll, mu, s, improved = l2, m2, s2, True
        if not improved:
            break
    return {"mu": mu, "s": s, "a": mu * s, "b": (1 - mu) * s, "loglik": ll, "icc_rho": 1 / (s + 1), "at_bound": s >= s_max * 0.999,
            "pooled_rate": tot}


def simulate(name, k_cmp, draw_arm_p, rng, sims, post_cmp):
    """post_cmp: list of (alpha, beta) posterior parameters of the comparator's p_q. draw_arm_p(p_cmp_draws, rng) -> arm p_q."""
    res = {"conditional": {"sig": 0, "R1": 0, "le6": 0, "b7_12": 0, "ge13": 0, "arm_total": 0},
           "resample_both": {"sig": 0, "R1": 0, "cmp_total": 0, "arm_total": 0}}
    for _ in range(sims):
        pc = [rng.betavariate(al, be) for al, be in post_cmp]
        pa = draw_arm_p(pc, rng)
        ka = [sum(rng.random() < p for _ in range(N_PASSES)) for p in pa]
        kc_new = [sum(rng.random() < p for _ in range(N_PASSES)) for p in pc]
        ta = sum(ka)
        # conditional on the observed comparator
        t = signflip([c - a for c, a in zip(k_cmp, ka)])
        R = res["conditional"]
        R["arm_total"] += ta
        R["sig"] += t["p_one_sided"] < ALPHA
        R["R1"] += classify(ta, t["p_one_sided"])[1].startswith("R1")
        R["le6"] += ta <= BAND_LARGE_MAX
        R["b7_12"] += BAND_LARGE_MAX < ta <= BAND_MODERATE_MAX
        R["ge13"] += ta > BAND_MODERATE_MAX
        # resample both
        t2 = signflip([c - a for c, a in zip(kc_new, ka)])
        R2 = res["resample_both"]
        R2["sig"] += t2["p_one_sided"] < ALPHA
        R2["R1"] += classify(ta, t2["p_one_sided"])[1].startswith("R1")
        R2["cmp_total"] += sum(kc_new)
        R2["arm_total"] += ta
    out = {"alternative": name, "sims": sims}
    for d, R in res.items():
        o = {}
        for key, v in R.items():
            if key.endswith("_total"):
                o["mean_" + key] = round(v / sims, 2)
            else:
                p = v / sims
                o["P_" + key] = round(p, 4)
                o["mc_se_" + key] = round(math.sqrt(p * (1 - p) / sims), 4)
        out[d] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    kc, k7 = counts(CMP_ROWS), counts(V070_ROWS)
    fc, f7 = fit_bb(kc), fit_bb(k7)
    post_c = [(fc["a"] + x, fc["b"] + N_PASSES - x) for x in kc]
    post_7a = [(f7["a"] + x, f7["b"] + N_PASSES - x) for x in k7]
    a7b, b7b = f7["pooled_rate"] * fc["s"], (1 - f7["pooled_rate"]) * fc["s"]   # 0.7.0 mean, 0.9.1 concentration
    post_7b = [(a7b + x, b7b + N_PASSES - x) for x in k7]
    alts = [
        ("null: arm p_q = comparator p_q (size under the conditional design)", lambda pc, r: pc),
        ("(i-a) arm p_q = 0.7.0 EB per-question estimates, own beta-binomial fit", lambda pc, r: [r.betavariate(x, y) for x, y in post_7a]),
        ("(i-b) arm p_q = 0.7.0 per-question estimates, 0.7.0 mean with the 0.9.1 fit's concentration", lambda pc, r: [r.betavariate(x, y) for x, y in post_7b]),
        ("(ii) uniform 50% reduction of each comparator p_q", lambda pc, r: [0.5 * p for p in pc]),
        ("(iii) uniform 75% reduction of each comparator p_q", lambda pc, r: [0.25 * p for p in pc]),
    ]
    out = {"model": {"comparator_counts_hist": {str(v): kc.count(v) for v in range(4)}, "comparator_total": sum(kc),
                     "v070_counts_hist": {str(v): k7.count(v) for v in range(4)}, "v070_total": sum(k7),
                     "fit_comparator_0.9.1": fc, "fit_0.7.0": f7, "fit_0.7.0_variant_b": {"a": a7b, "b": b7b},
                     "comparator_posterior_expected_total": round(sum(N_PASSES * x / (x + y) for x, y in post_c), 2),
                     "v070_posterior_expected_total_a": round(sum(N_PASSES * x / (x + y) for x, y in post_7a), 2),
                     "v070_posterior_expected_total_b": round(sum(N_PASSES * x / (x + y) for x, y in post_7b), 2)},
           "test": "analysis_col511.signflip, one-sided, alpha 0.05; R1 = arm empties <= 6 and one-sided p < 0.05",
           "seed": a.seed, "results": []}
    for i, (name, f) in enumerate(alts):
        rng = random.Random(a.seed + i)
        out["results"].append(simulate(name, kc, f, rng, a.sims, post_c))
        r = out["results"][-1]
        print(f"{name}: conditional P(p<0.05)={r['conditional']['P_sig']} (se {r['conditional']['mc_se_sig']}), "
              f"P(R1)={r['conditional']['P_R1']}, mean arm empties {r['conditional']['mean_arm_total']}; "
              f"resample-both P(p<0.05)={r['resample_both']['P_sig']} (se {r['resample_both']['mc_se_sig']})", flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["model"], indent=1))


if __name__ == "__main__":
    main()
