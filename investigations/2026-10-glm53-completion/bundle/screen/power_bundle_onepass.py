#!/usr/bin/env python3
"""Operating characteristics of the one-pass GPQA screen of the correctness bundle (PLAN.md amendment 14 (4)).
CPU only, standard library, fixed seed. Adapts arm-col511/power_col511.py (empirical-Bayes beta-binomial over questions,
fit_bb unchanged) to ONE pass of 198 questions.

Per-question empty probabilities p_q: a Beta(a, b) prior fitted by maximum marginal likelihood to per-question empty counts
of a 3-pass record; question q's rate is drawn from its posterior Beta(a + k_q, b + n - k_q) (question-level clustering).
One simulated screen pass draws one Bernoulli(p_q) per question.

Empty-count thresholds of the amendment 14c rules (E = empty replies of 198, any cause):
  E <= 2   the empty-count component of CLEAR PROMISING
  E == 3   the empty-count component of SIMILAR / NO CLEAR SEPARATION (or MIXED with a flag)
  E >= 4   SIMILAR OR WORSE
These are P(E <= threshold) etc. ONLY. Accuracy (raw / stated thresholds) and stopping (premature stops) are NOT
modelled, so nothing here is the probability of the complete clear-promising rule.
Also reported: the exact one-sided McNemar (sign) test of the screen pass vs the OBSERVED 0.9.1 record pass 1
(seed 1234) on discordant questions, for reference only (at most 5 rescues are possible, so p >= 1/32).

usage: power_bundle_onepass.py [--sims 50000] [--seed 20261012] [--out power-onepass.json]
"""
import argparse, json, math, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # power_col511.py and analysis_col511.py, copied unchanged from arm-col511
import power_col511 as P  # noqa: E402  (fit_bb, counts, N_Q, N_PASSES)

RUNS = "<this-repo>/runs"
CMP_ROWS = f"{RUNS}/2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3_gpqa-diamond/results.jsonl"
V070_ROWS = f"{RUNS}/2026-09-29_glm53-flash_k3.25-v0.7.0-dflash5_gpqa-diamond/results.jsonl"
N_Q = 198
CP_MAX, SW_MIN = 2, 4


def pass1_vector(path):
    v = [0] * N_Q
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            if r["pass"] == 1:
                v[r["doc_id"]] = int(bool(r["empty"]))
    return v


def mcnemar_one_sided(base, new):
    """Exact one-sided sign test that the new pass has fewer empties: rescued r (base 1, new 0) vs introduced i."""
    r = sum(1 for b, n in zip(base, new) if b and not n)
    i = sum(1 for b, n in zip(base, new) if n and not b)
    m = r + i
    if m == 0:
        return 1.0
    return sum(math.comb(m, j) for j in range(r, m + 1)) / 2 ** m


def run(name, post, transform, base, rng, sims):
    c = {"cp": 0, "mid": 0, "sw": 0, "mcn_sig": 0}
    tot = 0
    hist = [0] * 12
    for _ in range(sims):
        pq = transform([rng.betavariate(a, b) for a, b in post])
        e_vec = [1 if rng.random() < p else 0 for p in pq]
        e = sum(e_vec)
        tot += e
        hist[min(e, 11)] += 1
        c["cp"] += e <= CP_MAX
        c["mid"] += CP_MAX < e < SW_MIN
        c["sw"] += e >= SW_MIN
        c["mcn_sig"] += mcnemar_one_sided(base, e_vec) < 0.05
    o = {"alternative": name, "sims": sims, "mean_empties": round(tot / sims, 2),
         "labels": {"P_cp": "P(E <= 2) only", "P_mid": "P(E == 3) only", "P_sw": "P(E >= 4)",
                    "not_modelled": "accuracy and stopping conditions; not P(clear promising rule)"},
         "P_E_hist_0..10,11+": [round(h / sims, 4) for h in hist]}
    for k, v in c.items():
        p = v / sims
        o["P_" + k] = round(p, 4)
        o["mc_se_" + k] = round(math.sqrt(p * (1 - p) / sims), 4)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    kc, k7 = P.counts(CMP_ROWS), P.counts(V070_ROWS)
    fc, f7 = P.fit_bb(kc), P.fit_bb(k7)
    post_c = [(fc["a"] + x, fc["b"] + P.N_PASSES - x) for x in kc]
    post_7 = [(f7["a"] + x, f7["b"] + P.N_PASSES - x) for x in k7]
    a7b, b7b = f7["pooled_rate"] * fc["s"], (1 - f7["pooled_rate"]) * fc["s"]
    post_7b = [(a7b + x, b7b + P.N_PASSES - x) for x in k7]
    base = pass1_vector(CMP_ROWS)
    ident = lambda pq: pq
    alts = [
        ("null: bundle p_q = 0.9.1 record p_q (EB posterior)", post_c, ident),
        ("worse: 1.5 x each 0.9.1 p_q", post_c, lambda pq: [min(1.0, 1.5 * p) for p in pq]),
        ("(ii) uniform 50% reduction of each 0.9.1 p_q", post_c, lambda pq: [0.5 * p for p in pq]),
        ("(iii) uniform 75% reduction of each 0.9.1 p_q", post_c, lambda pq: [0.25 * p for p in pq]),
        ("(i-a) 0.7.0-like: 0.7.0 EB per-question estimates, own fit", post_7, ident),
        ("(i-b) 0.7.0-like: 0.7.0 mean with the 0.9.1 fit's concentration", post_7b, ident),
        ("(iv) complete removal of empties (p_q = 0)", post_c, lambda pq: [0.0 for _ in pq]),
    ]
    out = {"design": "one pass, 198 questions; P(E<=%d), P(E=%d), P(E>=%d) only (amendment 14c empty-count thresholds); "
                     "accuracy and stopping conditions NOT modelled: these are not P(clear promising rule)" % (CP_MAX, CP_MAX + 1, SW_MIN),
           "model": {"fit_0.9.1_record": fc, "fit_0.7.0_record": f7,
                     "record_total_3_passes": sum(kc), "v070_total_3_passes": sum(k7),
                     "record_pass1_empties": sum(base),
                     "null_expected_per_pass": round(sum(x / (x + y) for x, y in post_c), 2),
                     "v070_expected_per_pass_a": round(sum(x / (x + y) for x, y in post_7), 2),
                     "v070_expected_per_pass_b": round(sum(x / (x + y) for x, y in post_7b), 2)},
           "seed": a.seed, "results": []}
    for i, (name, post, tr) in enumerate(alts):
        r = run(name, post, tr, base, random.Random(a.seed + i), a.sims)
        out["results"].append(r)
        print(f"{name}: mean E {r['mean_empties']}; P(E<=2)={r['P_cp']} (se {r['mc_se_cp']}); "
              f"P(E=3)={r['P_mid']}; P(E>=4)={r['P_sw']}; "
              f"P(McNemar vs record pass 1 p<0.05)={r['P_mcn_sig']}", flush=True)
    print(json.dumps(out["model"], indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
