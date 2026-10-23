#!/usr/bin/env python3
"""Recon F1 (2026-10-22): stock tpurtell 0.7.0 vs stock 0.9.1 on K3.25 GPQA Diamond, CPU only, read-only inputs.

POST HOC. Every input below was inspected by earlier sessions; the frozen readings (rep070 O3, B+ / E R-I) stand.
The rules in RULES are written before this script was first run; outputs are descriptive with stated assumptions.

Writes only: ./manifest.json, ./rows.jsonl (ids, flags, token counts; no text), ./result.json, ./result.md
"""
import hashlib, importlib.util, json, os, random, statistics, sys
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = "<this-repo>"
INV = "<runs>"
GPQA = "<gpqa-runs>"
FROZEN = f"{INV}/rep070/refreeze/frozen/analysis_col511.py"
FROZEN_SHA = "b5b13ce2155085ca"  # prefix; same file in rep070/refreeze, d091, d091b

RULES = {
    "unit": "question (doc_id 0-197); a question-pass is one request",
    "E_empty": "lm-eval reply empty (frozen analysis_col511 definition; published 'empty' field)",
    "E_capped": "finish_reason == 'length' (ran to the 327,680-token cap); logged passes only",
    "E_reach100k": "completion_tokens >= 100,000; logged passes only",
    "E_finish_given_reach": "finish_reason == 'stop' among E_reach100k",
    "E_finish_by_165536_given_reach": "finish_reason == 'stop' and completion_tokens <= 165,536 among E_reach100k "
                                      "(horizon of the B+/E continuations: cut near 100k + 65,536)",
    "estimator": "per-question rate difference averaged over questions (each engine's rate = events / passes on that "
                 "question), i.e. equal weight per question; handles unequal pass counts",
    "conditional_estimator": "ratio of pooled events over pooled at-risk, question-cluster bootstrap; plus a question-"
                             "matched mean over questions at risk on both engines",
    "interval": "question bootstrap, 20,000 resamples, seed 20261022, 2.5/97.5 percentiles",
    "test_within_question": "stratified permutation: within each question, the pass outcomes of both engines are "
                            "reassigned to engines keeping each engine's pass count; 20,000 draws, seed 20261023",
    "test_pass_level": "exact permutation over passes (each pass a unit, statistic = mean empties per pass)",
    "test_session_level": "exact permutation over server sessions (statistic = mean empty rate per session)",
    "reporting": "effect, interval and the range of effects the data are compatible with; an interval that "
                 "includes 0 is NOT read as 'no gap' and does not decide whether correctness work continues",
}
B, P, SEED_B, SEED_P = 20000, 20000, 20261022, 20261023


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


# ------------------------------------------------------------------ sources
SRC = []  # (engine, set, label, rows, session_of_pass)


def add(engine, sset, label, rows, sessions, files, image, notes=""):
    SRC.append(dict(engine=engine, set=sset, label=label, rows=rows, sessions=sessions, files=files, image=image, notes=notes))


IMG070 = "sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629"
IMG091 = "sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07"

r = f"{PUB}/runs/2026-09-29_glm53-flash_k3.25-v0.7.0-dflash5_gpqa-diamond/results.jsonl"
add("0.7.0", "original", "0.7.0 record (seeds 1234-1236)", load_jsonl(r), {1: "070-rec-A (2026-09-29)", 2: "070-rec-B (2026-10-10)", 3: "070-rec-B (2026-10-10)"},
    {r: sha(r)}, IMG070, "pass 1 unlogged: finish_reason / completion_tokens null; no server log")
r = f"{PUB}/runs/2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3_gpqa-diamond/results.jsonl"
add("0.9.1", "original", "0.9.1 record (seeds 1234-1236)", load_jsonl(r), {1: "091-rec-A (2026-10-08)", 2: "091-rec-B (2026-10-09)", 3: "091-rec-B (2026-10-09)"},
    {r: sha(r)}, IMG091)

# rep070-F: rebuilt with the frozen loader (reads lm-eval samples in-process; only flags and hashes leave the function)
spec = importlib.util.spec_from_file_location("acol", FROZEN)
acol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acol)
assert sha(FROZEN).startswith(FROZEN_SHA), "frozen module changed"
acol.SEED_BASE = 70700
dirs = [f"{GPQA}/glm53-flash-exl3-k3.25@tpurtell+rep070-F__pass{p}" for p in (1, 2, 3)]
logs = [f"{GPQA}/glm53-flash-exl3-k3.25@tpurtell+rep070-F.s{s}.requests.jsonl" for s in (1, 2)]
rows, checks, inputs = acol.load_arm("rep070-F", dirs, logs, [], None)
bad = [c for c in checks if not c[1]]
if bad:
    sys.exit(f"rep070 loader checks failed: {bad[:3]}")
rep_rows = [dict(v) for _, v in sorted(rows.items())]
add("0.7.0", "fresh", "0.7.0 rep070-F (seeds 70701-70703)", rep_rows, {1: "070-rep-s1 (2026-10-16)", 2: "070-rep-s2 (2026-10-16/17)", 3: "070-rep-s2 (2026-10-16/17)"},
    {**{l: sha(l) for l in logs}, **{k: v["sha256"] for k, v in inputs["samples"].items()}}, IMG070, f"frozen loader {FROZEN_SHA}; {len(checks)} loader checks ok")
for tag, seed, sess in (("d091", 70704, "091-D (2026-10-21)"), ("d091b", 70705, "091-D2 (2026-10-22)")):
    r = f"{PUB}/investigations/2026-10-glm53-completion/long-reply/{tag}/run/rows-d091.jsonl"
    rr = load_jsonl(r)
    for x in rr:
        x["pass"] = 1
    add("0.9.1", "fresh", f"0.9.1 {'D' if tag == 'd091' else 'D2'} (seed {seed})", rr, {1: sess}, {r: sha(r)}, IMG091)

# ------------------------------------------------------------------ unify and check
U = []
problems = []
doc_ref = {}
for si, s in enumerate(SRC):
    for x in s["rows"]:
        key = (s["label"], x["pass"])
        sess = s["sessions"][x["pass"]]
        u = dict(engine=s["engine"], set=s["set"], src=s["label"], passkey=f"{s['label']}#p{x['pass']}", session=sess,
                 seed=x["seed"], doc=x["doc_id"], doc_hash=x["doc_hash"], prompt_hash=x["prompt_hash"], target_hash=x["target_hash"],
                 empty=bool(x["empty"]), finish_reason=x.get("finish_reason"), completion_tokens=x.get("completion_tokens"),
                 logged=x.get("completion_tokens") is not None)
        U.append(u)
        ref = doc_ref.setdefault(u["doc"], (u["doc_hash"], u["prompt_hash"], u["target_hash"]))
        if ref != (u["doc_hash"], u["prompt_hash"], u["target_hash"]):
            problems.append(f"hash mismatch doc {u['doc']} in {u['passkey']}")
passes = sorted({u["passkey"] for u in U})
for pk in passes:
    docs = sorted(u["doc"] for u in U if u["passkey"] == pk)
    if docs != list(range(198)):
        problems.append(f"{pk}: {len(docs)} docs, not 0-197")
if problems:
    sys.exit("input checks failed: " + "; ".join(problems[:5]))

with open(os.path.join(HERE, "rows.jsonl"), "w") as f:
    for u in U:
        f.write(json.dumps({k: u[k] for k in ("engine", "set", "passkey", "session", "seed", "doc", "empty", "finish_reason", "completion_tokens", "logged")}) + "\n")

# ------------------------------------------------------------------ helpers
def capped(u): return u["logged"] and u["finish_reason"] == "length"
def reach(u): return u["logged"] and u["completion_tokens"] >= 100000
def fin(u): return u["finish_reason"] == "stop"
def fin_h(u): return u["finish_reason"] == "stop" and u["completion_tokens"] <= 165536


def sel(sets, need_logged=False):
    return [u for u in U if u["set"] in sets and (u["logged"] or not need_logged)]


def per_q(rows, f):
    """{doc: {engine: [outcomes]}}"""
    d = {}
    for u in rows:
        d.setdefault(u["doc"], {"0.7.0": [], "0.9.1": []})[u["engine"]].append(1 if f(u) else 0)
    return d


def _m(v):
    return sum(v) / len(v)


def qdiff(d, docs):
    v = [_m(d[q]["0.7.0"]) - _m(d[q]["0.9.1"]) for q in docs if d[q]["0.7.0"] and d[q]["0.9.1"]]
    return sum(v) / len(v)


def rate(d, eng, docs):
    v = [_m(d[q][eng]) for q in docs if d[q][eng]]
    return sum(v) / len(v)


def boot(d, stat, B=B, seed=SEED_B):
    rng = random.Random(seed)
    docs = sorted(d)
    out = []
    for _ in range(B):
        s = [docs[rng.randrange(len(docs))] for _ in docs]
        try:
            out.append(stat(d, s))
        except ZeroDivisionError:
            pass
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1], len(out)


def boot_fast(d, B=B, seed=SEED_B):
    """question bootstrap of the per-question rate difference and of the rate ratio (same resamples as boot())"""
    rng = random.Random(seed)
    docs = sorted(d)
    a = {q: _m(d[q]["0.7.0"]) for q in docs}
    b = {q: _m(d[q]["0.9.1"]) for q in docs}
    n = len(docs)
    diffs, ratios = [], []
    for _ in range(B):
        s = [docs[rng.randrange(n)] for _ in docs]
        sa = sum(a[q] for q in s); sb = sum(b[q] for q in s)
        diffs.append((sa - sb) / n)
        if sb > 0:
            ratios.append(sa / sb)
    diffs.sort(); ratios.sort()
    q = lambda v: (v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]) if v else (None, None)
    return q(diffs), q(ratios), len(diffs), len(ratios)


def perm_within(d, P=P, seed=SEED_P):
    """stratified within-question permutation; equivalent to shuffling pass outcomes within each question:
    the 0.7.0 share of a question's events is hypergeometric given its pass counts. Questions with no events
    (or all events) contribute a constant."""
    rng = random.Random(seed)
    docs = sorted(d)
    n = len(docs)
    obs = qdiff(d, docs)
    const, var = 0.0, []
    for q in docs:
        a, b = d[q]["0.7.0"], d[q]["0.9.1"]
        e, na, nb = sum(a) + sum(b), len(a), len(b)
        if e == 0 or e == na + nb:
            const += (sum(a) / na - sum(b) / nb)
        else:
            var.append((e, na, nb))
    le = ge = ab = 0
    for _ in range(P):
        t = const
        for e, na, nb in var:
            x = sum(1 for i in rng.sample(range(na + nb), e) if i < na)
            t += x / na - (e - x) / nb
        t /= n
        le += t <= obs + 1e-12
        ge += t >= obs - 1e-12
        ab += abs(t) >= abs(obs) - 1e-12
    return {"obs": obs, "questions_with_events": len(var), "p_one_sided_070_fewer": (le + 1) / (P + 1),
            "p_one_sided_070_more": (ge + 1) / (P + 1), "p_two_sided": (ab + 1) / (P + 1), "draws": P}


def exact_units(a, b):
    """exact permutation over units; statistic = mean(a) - mean(b)"""
    allv = a + b
    obs = statistics.mean(a) - statistics.mean(b)
    n = len(a)
    tot = le = ab = 0
    for c in combinations(range(len(allv)), n):
        A = [allv[i] for i in c]
        Bv = [allv[i] for i in range(len(allv)) if i not in c]
        t = statistics.mean(A) - statistics.mean(Bv)
        tot += 1
        le += t <= obs + 1e-12
        ab += abs(t) >= abs(obs) - 1e-12
    return {"obs": obs, "p_one_sided_070_fewer": le / tot, "p_two_sided": ab / tot, "arrangements": tot}


def summarize_binary(sets, f, name, need_logged):
    rows = sel(sets, need_logged)
    d = per_q(rows, f)
    docs = [q for q in sorted(d) if d[q]["0.7.0"] and d[q]["0.9.1"]]
    d = {q: d[q] for q in docs}
    pk = {}
    for u in rows:
        pk.setdefault((u["engine"], u["passkey"], u["session"]), []).append(1 if f(u) else 0)
    per_pass = {f"{e} | {p}": f"{sum(v)}/{len(v)}" for (e, p, s), v in sorted(pk.items())}
    sess = {}
    for (e, p, s), v in pk.items():
        sess.setdefault((e, s), []).extend(v)
    r070, r091 = rate(d, "0.7.0", docs), rate(d, "0.9.1", docs)
    (lo, hi), (rlo, rhi), nb, _ = boot_fast(d)
    res = {
        "endpoint": name, "sets": sets, "logged_only": need_logged,
        "passes": {"0.7.0": len({p for (e, p, s) in pk if e == "0.7.0"}), "0.9.1": len({p for (e, p, s) in pk if e == "0.9.1"})},
        "events": {"0.7.0": sum(sum(v) for (e, p, s), v in pk.items() if e == "0.7.0"),
                   "0.9.1": sum(sum(v) for (e, p, s), v in pk.items() if e == "0.9.1")},
        "requests": {"0.7.0": sum(len(v) for (e, p, s), v in pk.items() if e == "0.7.0"),
                     "0.9.1": sum(len(v) for (e, p, s), v in pk.items() if e == "0.9.1")},
        "per_pass": per_pass,
        "per_session": {f"{e} | {s}": f"{sum(v)}/{len(v)}" for (e, s), v in sorted(sess.items())},
        "rate_070_pct": 100 * r070, "rate_091_pct": 100 * r091,
        "diff_pp": 100 * (r070 - r091), "diff_pp_ci95": [100 * lo, 100 * hi], "bootstrap_valid": nb,
        "ratio_070_over_091": r070 / r091 if r091 else None, "ratio_ci95": [rlo, rhi],
        "within_question_permutation": perm_within(d),
        "pass_level_exact": exact_units([statistics.mean(v) for (e, p, s), v in pk.items() if e == "0.7.0"],
                                        [statistics.mean(v) for (e, p, s), v in pk.items() if e == "0.9.1"]),
        "session_level_exact": exact_units([statistics.mean(v) for (e, s), v in sess.items() if e == "0.7.0"],
                                           [statistics.mean(v) for (e, s), v in sess.items() if e == "0.9.1"]),
    }
    return res


def summarize_conditional(sets, at_risk, f, name):
    rows = [u for u in sel(sets, True) if at_risk(u)]
    ev = {e: [1 if f(u) else 0 for u in rows if u["engine"] == e] for e in ("0.7.0", "0.9.1")}
    byq = {}
    for u in rows:
        byq.setdefault(u["doc"], {"0.7.0": [], "0.9.1": []})[u["engine"]].append(1 if f(u) else 0)

    def pooled(dd, s):
        a = [x for q in s for x in dd[q]["0.7.0"]]
        b = [x for q in s for x in dd[q]["0.9.1"]]
        return sum(a) / len(a) - sum(b) / len(b)

    lo, hi, nb = boot(byq, pooled)
    matched = [q for q in byq if byq[q]["0.7.0"] and byq[q]["0.9.1"]]
    mdict = {q: byq[q] for q in matched}
    mlo, mhi, mnb = boot(mdict, lambda dd, s: qdiff(dd, s)) if matched else (None, None, 0)
    return {
        "endpoint": name, "sets": sets,
        "at_risk": {e: len(v) for e, v in ev.items()}, "events": {e: sum(v) for e, v in ev.items()},
        "questions_at_risk": {e: len({u['doc'] for u in rows if u['engine'] == e}) for e in ev},
        "p_070": sum(ev["0.7.0"]) / len(ev["0.7.0"]) if ev["0.7.0"] else None,
        "p_091": sum(ev["0.9.1"]) / len(ev["0.9.1"]) if ev["0.9.1"] else None,
        "diff": (sum(ev["0.7.0"]) / len(ev["0.7.0"]) - sum(ev["0.9.1"]) / len(ev["0.9.1"])) if ev["0.7.0"] and ev["0.9.1"] else None,
        "diff_ci95_question_bootstrap": [lo, hi], "bootstrap_valid": nb,
        "question_matched": {"questions": len(matched), "mean_diff": qdiff(mdict, matched) if matched else None,
                             "ci95": [mlo, mhi], "bootstrap_valid": mnb},
    }


SETS = {"original": ["original"], "fresh": ["fresh"], "pooled": ["original", "fresh"]}
result = {"rules": RULES, "post_hoc": True, "comparisons": {}}
for nm, ss in SETS.items():
    c = {"empty": summarize_binary(ss, lambda u: u["empty"], "empty", False),
         "capped": summarize_binary(ss, capped, "capped (logged passes)", True),
         "reach100k": summarize_binary(ss, reach, "reach 100k (logged passes)", True),
         "finish_given_reach": summarize_conditional(ss, reach, fin, "finish | reach 100k"),
         "finish_by_165536_given_reach": summarize_conditional(ss, reach, fin_h, "finish by 165,536 | reach 100k")}
    result["comparisons"][nm] = c

# ------------------------------------------------------------------ manifest
cov = {}
for pk in passes:
    us = [u for u in U if u["passkey"] == pk]
    cov[pk] = {"engine": us[0]["engine"], "session": us[0]["session"], "seed": sorted({u["seed"] for u in us}),
               "requests": len(us), "logged": sum(u["logged"] for u in us),
               "missing_tokens_and_finish_reason": sum(not u["logged"] for u in us)}
manifest = {
    "created": "2026-10-22", "purpose": "recon F1 input manifest (CPU only, read-only inputs)",
    "frozen_loader": {"path": FROZEN, "sha256": sha(FROZEN)},
    "sources": [{"engine": s["engine"], "set": s["set"], "label": s["label"], "image": s["image"], "files": s["files"],
                 "sessions": s["sessions"], "notes": s["notes"]} for s in SRC],
    "coverage_by_pass": cov,
    "questions": {"ids": "0-197", "doc_prompt_target_hashes_identical_across_all_passes": True, "n": len(doc_ref)},
    "rules": RULES,
}
json.dump(manifest, open(os.path.join(HERE, "manifest.json"), "w"), indent=1)
json.dump(result, open(os.path.join(HERE, "result.json"), "w"), indent=1)
print("ok", len(U), "rows;", len(passes), "passes")
