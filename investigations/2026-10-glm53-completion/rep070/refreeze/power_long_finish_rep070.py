#!/usr/bin/env python3
"""Operating characteristics of the long-reply finishing secondary (PROPOSAL-v2.md section 6c), with question clustering.
CPU only, standard library, fixed seeds. Extends ../../mechanism-ranking/power_long_finish.py (sha256 in PROPOSAL-v2.md),
which modelled reaching 100k as Poisson(21) and finishing as independent Bernoulli draws (no clustering, so optimistic).

Statistic (as in analysis_rep070.py): P(finish | still generating at 100,000) = finished / at risk, at risk = completion
tokens >= 100,000, finished = finish reason not 'length'. Comparators are the records AS OBSERVED (conditional design):
  0.9.1 record, passes 1-3 (25 at risk, 10 finished, 15 questions); old 0.7.0 record, passes 2-3 (14, 10, 11 questions;
  pass 1 has no request log).
Model for the new run (three passes, distinct seeds, passes independent given the question):
  reach: per-question probability rho_q of reaching 100k; empirical-Bayes beta-binomial (prior fitted by maximum marginal
         likelihood to the record's per-question reach counts, n = passes logged), posterior Beta(a + r_q, b + n - r_q);
  finish: per-question probability phi_q of finishing once at risk; beta-binomial prior fitted to (finished_q of r_q) over
         questions at risk, posterior Beta(a + f_q, b + r_q - f_q) (prior only for questions never at risk).
  The same rho_q, phi_q govern all three passes of q (question-level clustering).
Scenarios: S1 rep = old 0.7.0 record (reach and finish); S2 rep = 0.9.1 record; S3 reach of 0.9.1 with finish of 0.7.0;
S4 reach of 0.7.0 with finish of 0.9.1.
Readouts per scenario: the joint question-clustered 95% bootstrap interval for P_rep - P_cmp (the analysis's method; B=1000
here, 2,000 in the analysis) excludes 0 in the stated direction: rep - 0.9.1 > 0 ("finishing advantage over 0.9.1
detected") and rep - old 0.7.0 < 0 ("finishing deficit against the old 0.7.0 record detected"); plus the unclustered
one-sided Fisher test of power_long_finish.py for comparison; mean at-risk count, finished count, interval width.
These are operating characteristics of a DESCRIPTIVE secondary: no reading depends on them.
Variant R shares seeds with both records (coupled draws, checks/same-seed-identity.json); this model treats rep as
independent of the records, which describes variant F; under R the rep-old070 contrast is pulled toward 0 to the extent
outcomes are coupled.

The fitted finishing prior sits at its bound (no overdispersion detectable from about 10 events per record), so finishing
then varies by question only through the posterior; --finish-s imposes a concentration (e.g. 2, intraclass 0.33) as a
sensitivity with more question-level heterogeneity in finishing.

usage: power_long_finish_rep070.py [--sims 2000] [--boot 1000] [--seed 20261016] [--procs 8] [--finish-s S] [--out FILE.json]
"""
import argparse, json, math, multiprocessing as mp, random, sys
from collections import defaultdict

G = "<gpqa-runs>/"
TOK = "<runs>/d7/exposure/gpqa_prompt_tokens.json"
P = "glm53-flash-exl3-k3.25@tpurtell"
NQ, L = 198, 100000


def per_question(logs_seeds):
    doc = {d["messages_sha256"]: d["doc_id"] for d in json.load(open(TOK))["docs"]}
    r, f, n = [0] * NQ, [0] * NQ, 0
    for path, seeds in logs_seeds:
        for x in map(json.loads, open(G + path)):
            if x["seed"] in seeds:
                q = doc[x["messages_sha256"]]
                if x["completion_tokens"] >= L:
                    r[q] += 1
                    f[q] += x["finish_reason"] != "length"
    n = sum(len(s) for _, s in logs_seeds)
    return r, f, n


def lbeta(x, y):
    return math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)


def bb_ll(ks, ns, mu, s):
    a, b = mu * s, (1 - mu) * s
    return sum(lbeta(k + a, n - k + b) - lbeta(a, b) for k, n in zip(ks, ns))


def fit_bb(ks, ns, s_max=1e5):
    """power_col511.fit_bb generalised to a per-unit number of trials (grid over mu and s = a + b, then refinement)."""
    tot = sum(ks) / sum(ns)
    best = None
    for i in range(-40, 41):
        mu = min(0.95, max(1e-5, tot * math.exp(i * 0.05)))
        for j in range(0, 101):
            s = 10 ** (-2 + j * 0.07)
            if s <= s_max:
                ll = bb_ll(ks, ns, mu, s)
                if best is None or ll > best[0]:
                    best = (ll, mu, s)
    ll, mu, s = best
    for _ in range(60):
        imp = False
        for dm, ds in ((1.01, 1), (1 / 1.01, 1), (1, 1.05), (1, 1 / 1.05)):
            m2, s2 = mu * dm, min(s_max, s * ds)
            if 0 < m2 < 1:
                l2 = bb_ll(ks, ns, m2, s2)
                if l2 > ll + 1e-12:
                    ll, mu, s, imp = l2, m2, s2, True
        if not imp:
            break
    return {"mu": mu, "s": s, "a": mu * s, "b": (1 - mu) * s, "loglik": ll, "at_bound": s >= s_max * 0.999, "pooled": tot}


def model(r, f, n, finish_s=None):
    fr = fit_bb(r, [n] * NQ)
    at = [q for q in range(NQ) if r[q]]
    ff = fit_bb([f[q] for q in at], [r[q] for q in at])
    if finish_s:   # sensitivity: the fitted finishing prior sits at the bound (no detectable overdispersion); impose s
        ff = {**ff, "s": finish_s, "a": ff["pooled"] * finish_s, "b": (1 - ff["pooled"]) * finish_s, "imposed_s": finish_s}
    post_r = [(fr["a"] + r[q], fr["b"] + n - r[q]) for q in range(NQ)]
    post_f = [(ff["a"] + f[q], ff["b"] + r[q] - f[q]) for q in range(NQ)]
    return {"fit_reach": fr, "fit_finish": ff, "post_r": post_r, "post_f": post_f}


def fisher_greater(a, b, c, d):
    n1, n2, k = a + b, c + d, a + c
    tot = math.comb(n1 + n2, k)
    return sum(math.comb(n1, x) * math.comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / tot


def boot_ci(repr_, repf, cr, cf, rng, B):
    """Joint question bootstrap of P_rep - P_cmp; only questions at risk on either side contribute, so the multinomial
    resampling counts are drawn for those questions only (sequential conditional binomials = exact multinomial marginals)."""
    rel = [q for q in range(NQ) if repr_[q] or cr[q]]
    m = len(rel)
    d = []
    for _ in range(B):
        rem, ar, af, br, bf = NQ, 0, 0, 0, 0
        for i, q in enumerate(rel):
            w = rng.binomialvariate(rem, 1.0 / (NQ - i)) if rem else 0
            rem -= w
            if w:
                ar += w * repr_[q]; af += w * repf[q]; br += w * cr[q]; bf += w * cf[q]
        if ar and br:
            d.append(af / ar - bf / br)
    d.sort()
    if len(d) < 20:
        return None
    return d[int(0.025 * len(d))], d[min(len(d) - 1, int(0.975 * len(d)))]


def run_chunk(args):
    name, rep_r_post, rep_f_post, cmp091, cmp070, sims, B, seed = args
    rng = random.Random(seed)
    acc = defaultdict(float)
    widths = []
    for _ in range(sims):
        rr, rf = [0] * NQ, [0] * NQ
        for q in range(NQ):
            rho = rng.betavariate(*rep_r_post[q]); phi = rng.betavariate(*rep_f_post[q])
            for _p in range(3):
                if rng.random() < rho:
                    rr[q] += 1
                    rf[q] += rng.random() < phi
        nr, nf = sum(rr), sum(rf)
        acc["at_risk"] += nr; acc["finished"] += nf
        if nr == 0:
            acc["no_rep_at_risk"] += 1
            continue
        for key, (cr, cf), direction in (("v091", cmp091, +1), ("old070", cmp070, -1)):
            ci = boot_ci(rr, rf, cr, cf, rng, B)
            if ci is None:
                acc[f"undefined_{key}"] += 1
                continue
            lo, hi = ci
            acc[f"detect_{key}"] += (lo > 0) if direction > 0 else (hi < 0)
            acc[f"ci_excludes0_any_{key}"] += lo > 0 or hi < 0
            widths.append((key, hi - lo))
        acc["fisher_vs_v091"] += fisher_greater(nf, nr - nf, sum(cmp091[1]), sum(cmp091[0]) - sum(cmp091[1])) < 0.05
    return dict(acc), widths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--finish-s", type=float, default=None, help="sensitivity: finishing prior concentration a+b (pooled mean)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r91, f91, n91 = per_question([(P + "+v0.9.1.requests.jsonl", {1234}), (P + "+v0.9.1+ps.requests.jsonl", {1235, 1236})])
    r70, f70, n70 = per_question([(P + "+ps.requests.jsonl", {1235, 1236})])
    obs = {"v091": {"at_risk": sum(r91), "finished": sum(f91), "questions": sum(x > 0 for x in r91), "passes": n91},
           "old070": {"at_risk": sum(r70), "finished": sum(f70), "questions": sum(x > 0 for x in r70), "passes": n70}}
    assert (obs["v091"]["at_risk"], obs["v091"]["finished"], obs["old070"]["at_risk"], obs["old070"]["finished"]) == (25, 10, 14, 10), obs
    m91, m70 = model(r91, f91, n91, a.finish_s), model(r70, f70, n70, a.finish_s)
    scen = [("S1: rep = old 0.7.0 record (reach and finish)", m70["post_r"], m70["post_f"]),
            ("S2: rep = 0.9.1 record (reach and finish)", m91["post_r"], m91["post_f"]),
            ("S3: reach of 0.9.1, finishing of 0.7.0", m91["post_r"], m70["post_f"]),
            ("S4: reach of 0.7.0, finishing of 0.9.1", m70["post_r"], m91["post_f"])]
    out = {"observed": obs, "fits": {"v091": {k: m91[k] for k in ("fit_reach", "fit_finish")}, "old070": {k: m70[k] for k in ("fit_reach", "fit_finish")}},
           "sims": a.sims, "boot": a.boot, "seed": a.seed, "finish_s": a.finish_s, "results": []}
    chunks = a.procs
    with mp.Pool(a.procs) as pool:
        for i, (name, pr, pf) in enumerate(scen):
            per = [a.sims // chunks + (1 if c < a.sims % chunks else 0) for c in range(chunks)]
            jobs = [(name, pr, pf, (r91, f91), (r70, f70), per[c], a.boot, a.seed + 1000 * i + c) for c in range(chunks)]
            acc, widths = defaultdict(float), []
            for d, w in pool.map(run_chunk, jobs):
                for k, v in d.items():
                    acc[k] += v
                widths += w
            S = a.sims
            def pr_(k):
                p = acc.get(k, 0) / S
                return {"P": round(p, 3), "mc_se": round(math.sqrt(p * (1 - p) / S), 3)}
            wv = sorted(x for k, x in widths if k == "v091"); wo = sorted(x for k, x in widths if k == "old070")
            res = {"scenario": name, "mean_at_risk": round(acc["at_risk"] / S, 2), "mean_finished": round(acc["finished"] / S, 2),
                   "P_no_rep_reply_at_risk": pr_("no_rep_at_risk"),
                   "P_detect_finishing_advantage_over_v091_(CI_lo>0)": pr_("detect_v091"),
                   "P_detect_finishing_deficit_vs_old070_(CI_hi<0)": pr_("detect_old070"),
                   "P_CI_excludes_0_either_direction_vs_v091": pr_("ci_excludes0_any_v091"),
                   "P_CI_excludes_0_either_direction_vs_old070": pr_("ci_excludes0_any_old070"),
                   "P_unclustered_fisher_one_sided_vs_v091_p<0.05": pr_("fisher_vs_v091"),
                   "median_CI_width_vs_v091": round(wv[len(wv) // 2], 3) if wv else None,
                   "median_CI_width_vs_old070": round(wo[len(wo) // 2], 3) if wo else None}
            out["results"].append(res)
            print(json.dumps(res), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"observed": obs, "fit_summary": {k: {kk: {x: round(v, 4) if isinstance(v, float) else v for x, v in vv.items()}
                                                         for kk, vv in m.items() if kk.startswith("fit")} for k, m in (("v091", m91), ("old070", m70))}}))


if __name__ == "__main__":
    sys.exit(main())
