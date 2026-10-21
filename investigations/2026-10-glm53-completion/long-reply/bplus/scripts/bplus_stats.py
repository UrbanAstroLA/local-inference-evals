"""B+ frozen statistics and operational screens (PROPOSAL.md section 5). numpy only. Imported by analysis_bplus.py and
power_bplus.py, so the power simulation evaluates exactly the rules the analysis applies.

Unit: the prefix. Per engine, a prefix's rate is the mean of its 4 seeds' finish indicators; the engine's P is the mean
of the 12 prefix rates. A contrast A - B is the mean over prefixes of d_i = rate_A,i - rate_B,i.
  interval   percentile bootstrap over prefixes (resample the 12 d_i with replacement), B_BOOT draws, fixed seed
  test       exact sign-flip over prefixes: all 2^12 sign patterns of d_i; two-sided p = share of patterns whose
             |mean| >= the observed |mean| (ties counted; 1e-12 tolerance)
Screens (operational meanings, not causal boundaries; thresholds justified by power_bplus.py):
  EFFECT(A - B)     mean >= +DELTA_E, 95% interval lower bound > 0 and sign-flip p <= ALPHA
  REVERSE(A - B)    mean <= -DELTA_E, 95% interval upper bound < 0 and sign-flip p <= ALPHA
  SMALL(A - B)      |mean| <= SMALL_MAX and the 90% interval lies inside [-EQ_BOUND, +EQ_BOUND]
  PREFIX_HIGH       eta^2 >= ETA_MIN: share of the variance of all 144 outcomes explained by the prefix (one-way, 12
                    groups of 12 = 3 engines x 4 seeds)
Reading (primary contrast C1 = 0.7.0 - 0.9.1):
  R-E   EFFECT(C1): an engine continuation effect is supported - from identical 100k states, 0.7.0 finishes more often
  R-E-  REVERSE(C1): the opposite direction is supported
  R-S   SMALL for all three contrasts and PREFIX_HIGH: state-dominated - the state at 100k sets the outcome, the engines'
        continuations do not separate
  R-I   otherwise: inconclusive under these rules
Secondary readings with the same EFFECT / REVERSE / SMALL rules: C2 = B1 - 0.9.1 (the fixes' effect from a fixed state),
C3 = 0.7.0 - B1 (what remains after the fixes). No multiplicity correction; C2 and C3 are secondary and labelled so.
"""
import numpy as np

ALPHA = 0.05
DELTA_E = 0.15
SMALL_MAX = 0.10
EQ_BOUND = 0.20
ETA_MIN = 0.30
B_BOOT = 20000
BOOT_SEED = 20261016
N_PREFIX = 12
CONTRASTS = (("C1", "v070", "v091", "primary"), ("C2", "b1", "v091", "secondary"), ("C3", "v070", "b1", "secondary"))

_SIGNS = None


def signs(n=N_PREFIX):
    global _SIGNS
    if _SIGNS is None or _SIGNS.shape[1] != n:
        k = np.arange(2 ** n)[:, None]
        _SIGNS = np.where((k >> np.arange(n)) & 1, -1.0, 1.0)
    return _SIGNS


def signflip_p(d):
    d = np.asarray(d, dtype=np.float64)
    obs = abs(d.mean())
    m = np.abs(signs(len(d)) @ d) / len(d)
    return float((m >= obs - 1e-12).mean())


def boot_ci(d, B=B_BOOT, seed=BOOT_SEED, levels=(0.95, 0.90)):
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(axis=1)
    out = {}
    for lv in levels:
        a = (1 - lv) / 2
        out[lv] = (float(np.quantile(m, a)), float(np.quantile(m, 1 - a)))
    return out


def eta2(y):
    """y: array (prefixes, observations) of 0/1 outcomes. One-way between-prefix share of the total sum of squares."""
    y = np.asarray(y, dtype=np.float64)
    g = y.mean(axis=1, keepdims=True)
    tot = ((y - y.mean()) ** 2).sum()
    if tot == 0:
        return None
    return float(((g - y.mean()) ** 2).sum() * y.shape[1] / tot)


def icc1(y):
    """ANOVA ICC(1) for equal group sizes (descriptive)."""
    y = np.asarray(y, dtype=np.float64)
    k, n = y.shape
    gm = y.mean(axis=1)
    msb = n * ((gm - y.mean()) ** 2).sum() / (k - 1)
    msw = ((y - gm[:, None]) ** 2).sum() / (k * (n - 1))
    den = msb + (n - 1) * msw
    return None if den == 0 else float((msb - msw) / den)


def contrast(rA, rB, B=B_BOOT, seed=BOOT_SEED):
    d = np.asarray(rA, dtype=np.float64) - np.asarray(rB, dtype=np.float64)
    ci = boot_ci(d, B=B, seed=seed)
    p = signflip_p(d)
    mean = float(d.mean())
    eff = mean >= DELTA_E and ci[0.95][0] > 0 and p <= ALPHA
    rev = mean <= -DELTA_E and ci[0.95][1] < 0 and p <= ALPHA
    small = abs(mean) <= SMALL_MAX and ci[0.90][0] >= -EQ_BOUND and ci[0.90][1] <= EQ_BOUND
    return {"mean": mean, "ci95": ci[0.95], "ci90": ci[0.90], "signflip_p_two_sided": p,
            "prefixes_positive": int((d > 0).sum()), "prefixes_negative": int((d < 0).sum()), "prefixes_zero": int((d == 0).sum()),
            "EFFECT": bool(eff), "REVERSE": bool(rev), "SMALL": bool(small), "d": [float(x) for x in d]}


def screens(Y, B=B_BOOT, seed=BOOT_SEED):
    """Y: dict engine -> array (12 prefixes, 4 seeds) of finish indicators, prefixes in a fixed common order."""
    rates = {e: np.asarray(Y[e], dtype=np.float64).mean(axis=1) for e in Y}
    res = {"P": {e: float(rates[e].mean()) for e in rates}, "contrasts": {}}
    for name, a, b, role in CONTRASTS:
        c = contrast(rates[a], rates[b], B=B, seed=seed)
        c.update(role=role, label=f"{a} - {b}")
        res["contrasts"][name] = c
    allY = np.concatenate([np.asarray(Y[e], dtype=np.float64) for e in ("v070", "v091", "b1")], axis=1)
    e2 = eta2(allY)
    res["eta2_prefix"] = e2
    res["icc1_prefix_pooled"] = icc1(allY)
    res["icc1_prefix_by_engine"] = {e: icc1(Y[e]) for e in Y}
    res["PREFIX_HIGH"] = bool(e2 is not None and e2 >= ETA_MIN)
    C = res["contrasts"]
    if C["C1"]["EFFECT"]:
        r = "R-E"
    elif C["C1"]["REVERSE"]:
        r = "R-E-"
    elif all(C[k]["SMALL"] for k in C) and res["PREFIX_HIGH"]:
        r = "R-S"
    else:
        r = "R-I"
    res["reading"] = r
    res["secondary"] = {k: ("EFFECT" if C[k]["EFFECT"] else "REVERSE" if C[k]["REVERSE"] else "SMALL" if C[k]["SMALL"] else "unresolved")
                        for k in ("C2", "C3")}
    return res


MEANING = {
    "R-E": "From identical 100k-token states, stock 0.7.0's continuation finished within +64k more often than stock 0.9.1's, "
           "by at least the prespecified 0.15 with the prefix-level interval and test excluding no difference. The engines' "
           "continuations differ; this does not identify which component of 'engine' (decode path, D2/D7, layout, "
           "speculation depth) is responsible, and holds for 0.7.0-built states only.",
    "R-E-": "From identical 100k-token states, stock 0.9.1's continuation finished more often than 0.7.0's under the same rule.",
    "R-S": "State-dominated under these rules: all three engine contrasts were small (|mean| <= 0.10, 90% interval inside "
           "+-0.20) and the prefix explained at least 30% of the outcome variance. The finishing gap is then more plausibly "
           "set before 100k than by the continuation; small contrasts here are not proof of equality.",
    "R-I": "Inconclusive under the prespecified rules: neither an engine continuation effect nor state dominance is supported."}
