"""H2a/H2b/H2c diagnosis: per-request decode-vs-prefill gaps, argmax agreement and its structure, shift controls.
Replicates the frozen pairing (analyze_hookval.pairs_vs_ref) and adds per-request and per-position detail. CPU only."""
import glob, json, os, re, collections
import numpy as np, torch
RUN = "/run"
CL = [json.loads(l) for l in open(os.path.join(RUN, "client.jsonl"))]
reqs = [r for r in CL if r.get("kind") == "request"]
REQ = {}
for p in glob.glob(os.path.join(RUN, "forcewin", "req_*_r0.pt")):
    m = re.search(r"req_(.*)_r(\d+)\.pt$", os.path.basename(p))
    REQ[m.group(1)] = p
_cache = {}
def r0(tag):
    k = tag.replace(":", "_")
    if k not in REQ:
        return None
    if k not in _cache:
        _cache[k] = torch.load(REQ[k], weights_only=False)
    return _cache[k]
def pairs_vs_ref(prefix, ref_prefix):
    out = []
    for r in reqs:
        if not r["tag"].startswith(prefix) or r.get("W", 0) == 0:
            continue
        key = r["tag"].split("-", 2)[-1]
        ref = r0(f"{ref_prefix}-{key}"); a = r0(r["tag"])
        if ref is None or a is None:
            continue
        out.append((r["tag"], a, ref, r["W"], f"{ref_prefix}-{key}"))
    return out
allp = []
for prefix, refp in (("V2a-", "V2c"), ("V1-natural-", "V1-ref"), ("V1-poison-", "V1-ref"), ("V2b-", "V2b-ref")):
    pr = pairs_vs_ref(prefix, refp)
    if prefix == "V2b-":
        pr += [(t, a, r0(f"V2c-{t.split('-', 2)[-1]}"), W, f"V2c-{t.split('-', 2)[-1]}") for t, a, _, W, _ in pairs_vs_ref(prefix, "V2c")
               if r0(f"V2c-{t.split('-', 2)[-1]}") is not None]
    allp += pr
print("pairs", len(allp))
# keys as the frozen code builds them
odd = [(t, rt) for t, a, ref, W, rt in allp if t.split("-", 2)[-1] != rt.split("-", 1)[-1]]
print("pairs whose reference key differs from the request key:", odd[:5])
C = {n: i for i, n in enumerate(allp[0][1]["cols"])}
per = []
POS = []
for tag, a, ref, W, rt in allp:
    M = a["metrics"][:W].numpy().astype(float); Rf = ref["metrics"].numpy().astype(float)
    x = M[:, C["lp_pre_4"]]; y = Rf[:W, C["lp_pre_4"]]
    am, ar = M[:, C["argmax"]], Rf[:W, C["argmax"]]
    t1d, t1r = M[:, C["top1_p"]], Rf[:W, C["top1_p"]]
    ent_d, ent_r = M[:, C["entropy"]], Rf[:W, C["entropy"]]
    fk = M[:, C["argmax_is_forced"]]; fr_ = Rf[:W, C["argmax_is_forced"]]
    d0 = np.abs(x - y)
    def sh(v, s):
        o = np.full(W, np.nan)
        for i in range(W):
            if 0 <= i + s < v.size: o[i] = v[i + s]
        return o
    yfull = Rf[:, C["lp_pre_4"]]; afull = Rf[:, C["argmax"]]
    dp, dm = np.abs(x - sh(yfull, 1)), np.abs(x - sh(yfull, -1))
    agree = am == ar
    agree_p1 = am == sh(afull, 1); agree_m1 = am == sh(afull, -1)
    q = lambda v: float(np.nanmedian(v))
    per.append(dict(tag=tag, ref=rt, W=W, P=a["P"], med0=q(d0), medp=q(dp), medm=q(dm), top1=float(np.mean(agree)),
                    top1_p1=float(np.nanmean(agree_p1)), top1_m1=float(np.nanmean(agree_m1)),
                    q90=float(np.nanquantile(d0, .9)), q99=float(np.nanquantile(d0, .99)),
                    frac_forced_lp_gt_m001=float(np.mean(x > -0.01)), nan_ref=int(np.isnan(y).sum()),
                    h2c_fail=not (q(d0) < 0.1 * min(q(dp), q(dm))),
                    dec_argmax_is_forced=float(np.mean(fk)), ref_argmax_is_forced=float(np.nanmean(fr_)),
                    ref_rows=int(Rf.shape[0]), ref_nmeas1=float((ref["nmeas"][:W] == 1).float().mean()),
                    ref_kind=float(np.nanmean(Rf[:W, C["kind"]]))))
    for i in range(W):
        POS.append((tag, i, x[i], y[i], am[i], ar[i], t1d[i], t1r[i], ent_d[i], ent_r[i]))
json.dump(per, open("/d/out/h2_per_request.json", "w"), indent=0)
P = np.array([p[2:] for p in POS], dtype=float)
x, y, am, ar, t1d, t1r, ed, er = P.T
agree = am == ar
print("positions", len(P), "top1 agreement %.4f" % agree.mean())
# agreement vs decode top1 prob bins
for lo, hi in ((0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, .99), (.99, 1.01)):
    s = (t1r >= lo) & (t1r < hi)
    print(" ref top1_p in [%.2f,%.2f): n %6d agree %.4f  |dlp| q50 %.2e" % (lo, hi, s.sum(), agree[s].mean() if s.any() else np.nan, np.nanmedian(np.abs(x - y)[s]) if s.any() else np.nan))
# top1 by phase
by = collections.defaultdict(list)
for p in per:
    by[p["tag"].split("-")[0] + "-" + p["tag"].split("-")[1]].append(p)
for k, v in sorted(by.items()):
    W = np.array([p["W"] for p in v])
    print("%-14s req %3d  pos %6d  top1 %.4f  top1(+1) %.3f  top1(-1) %.3f  med0 %.2e medp %.3f medm %.3f  h2c_fail %d" % (
        k, len(v), W.sum(), np.average([p["top1"] for p in v], weights=W), np.average([p["top1_p1"] for p in v], weights=W),
        np.average([p["top1_m1"] for p in v], weights=W), np.median([p["med0"] for p in v]), np.median([p["medp"] for p in v]),
        np.median([p["medm"] for p in v]), sum(p["h2c_fail"] for p in v)))
fails = [p for p in per if p["h2c_fail"]]
print("H2c failing", len(fails), "of", len(per))
for p in sorted(fails, key=lambda p: -p["med0"] / max(min(p["medp"], p["medm"]), 1e-9))[:12]:
    print("  %-34s W %5d med0 %.2e medp %.2e medm %.2e ratio %.2f top1 %.3f top1+1 %.3f frac_lp>-0.01 %.2f" % (
        p["tag"], p["W"], p["med0"], p["medp"], p["medm"], p["med0"] / max(min(p["medp"], p["medm"]), 1e-12), p["top1"], p["top1_p1"],
        p["frac_forced_lp_gt_m001"]))
print("ref argmax_is_forced mean %.3f, dec argmax_is_forced mean %.3f" % (np.mean([p["ref_argmax_is_forced"] for p in per]), np.mean([p["dec_argmax_is_forced"] for p in per])))
