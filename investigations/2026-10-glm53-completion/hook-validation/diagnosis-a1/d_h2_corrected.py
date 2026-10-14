"""H2 re-analysis (post-inspection, labelled): corrected reference joins.
 C1  V2b requests of the two block-crossing traces are re-joined to their V2c reference with the offset
     3832 - P (the V2c window starts at 3,832; V2b's at the prompt). Causal prefill makes the rows identical.
 C2  top-1 agreement only where the reference row exists (the frozen code counted NaN reference rows as
     disagreements: V2b histories longer than the V2c window of 2,500).
Everything else as frozen (NULL_FACTOR 2, floors, TOP1_MIN 0.95). Also reports a shift control on argmax."""
import glob, json, os, re, collections
import numpy as np, torch
RUN = "/run"
CL = [json.loads(l) for l in open(os.path.join(RUN, "client.jsonl"))]
reqs = [r for r in CL if r.get("kind") == "request"]
REQ = {re.search(r"req_(.*)_r0\.pt$", os.path.basename(p)).group(1): p for p in glob.glob(RUN + "/forcewin/req_*_r0.pt")}
_c = {}
def r0(tag):
    k = tag.replace(":", "_")
    if k not in REQ: return None
    if k not in _c: _c[k] = torch.load(REQ[k], weights_only=False)
    return _c[k]
PREQ = {r["tag"]: r for r in reqs}
CROSS = ["53fa5b3e066f", "57703691f509"]
pairs = []   # (tag, a, ref, W, offset): decode index o <-> ref index o - offset
for r in reqs:
    t = r["tag"]; W = r.get("W", 0)
    if not W: continue
    if t.startswith("V2a-"): key = t.split("-", 2)[-1]; pairs.append((t, r0(t), r0("V2c-" + key), W, 0))
    elif t.startswith(("V1-natural-", "V1-poison-")): key = t.split("-", 2)[-1]; pairs.append((t, r0(t), r0("V1-ref-" + key), W, 0))
    elif t.startswith("V2b-"):
        key = t.split("-", 2)[-1]
        if r0("V2b-ref-" + key) is not None: pairs.append((t, r0(t), r0("V2b-ref-" + key), W, 0))
        elif key in CROSS:
            off = PREQ["V2c-" + key]["P"] - 0  # V2c prompt = prefix + window; its fw_ref index 0 = position 3832
            off = 3832 - r["P"]
            pairs.append((t, r0(t), r0("V2c-" + key), W, off))
        else: pairs.append((t, r0(t), r0("V2c-" + key), W, 0))
pairs = [p for p in pairs if p[1] is not None and p[2] is not None]
C = {n: i for i, n in enumerate(pairs[0][1]["cols"])}
def collect(shift=0):
    d, t1, per = [], [], {}
    for tag, a, ref, W, off in pairs:
        M = a["metrics"][:W].numpy().astype(float); Rf = ref["metrics"].numpy().astype(float)
        idx = np.arange(W) - off + shift
        ok = (idx >= 0) & (idx < Rf.shape[0])
        y = np.full(W, np.nan); ar = np.full(W, np.nan)
        y[ok] = Rf[idx[ok], C["lp_pre_4"]]; ar[ok] = Rf[idx[ok], C["argmax"]]
        x = M[:, C["lp_pre_4"]]; am = M[:, C["argmax"]]
        fin = np.isfinite(y)
        dd = np.abs(x - y)[fin]
        d += list(dd); t1 += list((am == ar)[fin])
        per[tag] = (float(np.median(dd)) if dd.size else np.nan, float(np.mean((am == ar)[fin])) if fin.any() else np.nan, int(fin.sum()))
    return np.array(d), float(np.mean(t1)), per
g0, t0, p0 = collect(0); gp, tp, pp = collect(1); gm, tm, pm = collect(-1)
null = []
for r in CL:
    if r.get("kind") != "free_hist": continue
    ref = r0("V1-ref-" + r["tag"])
    lps = np.array([v if v is not None else np.nan for v in r["lps"]], dtype=float)
    y = ref["metrics"][: len(lps), C["lp_pre_4"]].numpy().astype(float)
    null += list(np.abs(lps - y[: len(lps)]))
null = np.array(null); null = null[np.isfinite(null)]
FLOOR = {0.5: 0.01, 0.9: 0.30, 0.99: 1.50}
nq = {p: float(np.quantile(null, p)) for p in FLOOR}
tol = {p: max(2 * nq[p], FLOOR[p]) for p in FLOOR}
st = {p: float(np.quantile(g0, p)) for p in FLOOR}
res = dict(n=int(g0.size), top1=t0, aligned={str(k): v for k, v in st.items()}, null={str(k): v for k, v in nq.items()},
           null_n=int(null.size), tol={str(k): v for k, v in tol.items()},
           shift={"median_plus1": float(np.median(gp)), "median_minus1": float(np.median(gm)), "top1_plus1": tp, "top1_minus1": tm},
           H2a_corrected_pass=bool(all(st[p] <= tol[p] for p in FLOOR) and t0 >= 0.95))
fails = [t for t in p0 if not (p0[t][0] < 0.1 * min(pp[t][0], pm[t][0]))]
res["H2c_lp_rule_failing"] = fails
res["H2c_argmax_rule_failing"] = [t for t in p0 if not (p0[t][1] >= 0.9 and max(pp[t][1], pm[t][1]) <= 0.2)]
# per-phase
ph = collections.defaultdict(lambda: [0, 0.0])
for t, (m, a1, n) in p0.items():
    k = "-".join(t.split("-")[:2]); ph[k][0] += n; ph[k][1] += a1 * n
res["top1_by_phase"] = {k: v[1] / v[0] for k, v in sorted(ph.items())}
# decode-vs-decode reference: V2a natural vs V2b natural, same positions (H3 null) for argmax
print(json.dumps(res, indent=1))
json.dump(res, open("/d/out/h2_corrected.json", "w"), indent=1)
# per-request separation of the argmax shift control (reported, post-inspection)
a0 = np.array([v[1] for v in p0.values()]); ash = np.array([max(pp[t][1], pm[t][1]) for t in p0])
print("per-request aligned top-1 min %.4f (n=%d); shifted top-1 max %.4f" % (np.nanmin(a0), a0.size, np.nanmax(ash)))
for t in p0:
    if any(c in t for c in CROSS) and t.startswith("V2b"):
        print("  re-joined", t, "aligned top1 %.4f n %d | lp median %.2e" % (p0[t][1], p0[t][2], p0[t][0]))
res["per_request_top1_min"] = float(np.nanmin(a0)); res["per_request_shifted_top1_max"] = float(np.nanmax(ash))
json.dump(res, open("/d/out/h2_corrected.json", "w"), indent=1)
