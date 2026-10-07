#!/usr/bin/env python3
"""Verify local-inference-evals: manifests, recomputed summaries, comparison rules, label consistency.

    python3 tools/verify.py              # everything
    python3 tools/verify.py runs/<id>    # one run
Standard library only. Exit status 1 if anything fails.
"""
import json, math, random, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_KEYS = {"id", "config", "protocol", "date", "software", "notes"}
# Descriptive config fields that follow from others (repo, version, patches), so comparisons do not check them.
DESCRIPTIVE = ("id", "notes", "model.label", "engine.series", "engine.equivalent_to", "engine.built_on")
fails = []


# ---------------------------------------------------------------- labels (SCHEMA.md, "Labels"); the site uses these too

def weights_label(cfg):
    return cfg["model"]["label"]


def engine_label(cfg, with_name=True):
    """'<engine name> <version>[ + <patch name>...][ \u2248 <equivalent release>]', e.g. 'tpurtell 0.8.0 + kpool fixes \u2248 0.9.0'."""
    e = cfg["engine"]
    s = (e["name"] + " " if with_name else "") + e["version"].lstrip("v")
    s += "".join(f" + {p['name']}" for p in e.get("patches", []))
    return s + (f" \u2248 {e['equivalent_to']['version'].lstrip('v')}" if e.get("equivalent_to") else "")


def spec_label(cfg, with_method=False):
    sp = cfg["serving"]["speculative"]
    if not sp["tokens"]:
        return "no speculation"
    s = (f"{sp['method']} " if with_method else "") + f"{sp['tokens']} draft{'s' if sp['tokens'] > 1 else ''}"
    return s + (", sharing off" if cfg["serving"].get("draft_slot_sharing") is False else "")


def config_label(cfg, with_method=False):
    """'<weights> · <engine> · <speculation>'. Pass with_method=True once configs use more than one speculative method."""
    return f"{weights_label(cfg)} \u00b7 {engine_label(cfg)} \u00b7 {spec_label(cfg, with_method)}"


def check_labels(cfgs):
    """One label per thing and one thing per label: weights labels <-> model repos, engine labels <-> engine builds."""
    def one_to_one(kind, pairs):
        fwd, back = {}, {}
        for lab, thing in pairs:
            fwd.setdefault(lab, set()).add(thing); back.setdefault(thing, set()).add(lab)
        for lab, things in fwd.items():
            if len(things) > 1: fails.append(f"labels: label {lab!r} names {len(things)} different {kind}s: {sorted(things)}")
        for thing, labs in back.items():
            if len(labs) > 1: fails.append(f"labels: one {kind} has {len(labs)} labels: {sorted(labs)}")
    ok = []
    for c in cfgs:
        e = c.get("engine", {})
        miss = [k for k, v in (("model.label", c.get("model", {}).get("label")), ("engine.name", e.get("name")),
                               ("engine.series", e.get("series")), ("engine.built_on", e.get("built_on"))) if not v]
        miss += ["engine.patches[].name" for p in e.get("patches", []) if not p.get("name")]
        if miss: fails.append(f"config {c.get('id')}: missing label field(s) {miss}")
        else: ok.append(c)
    one_to_one("weights repo", [(weights_label(c), c["model"]["repo"]) for c in ok])
    build = lambda e: json.dumps([e["project"], e["version"], e["image"], [[p.get("source"), p.get("commit"), p.get("ports")] for p in e["patches"]]])
    one_to_one("engine build", [(engine_label(c), build(c["engine"])) for c in ok])
    desc = {}
    for c in ok:
        e = c["engine"]; d = json.dumps([e["series"], e["built_on"], e.get("equivalent_to")], sort_keys=True)
        if desc.setdefault(engine_label(c), d) != d:
            fails.append(f"labels: configs of engine {engine_label(c)!r} disagree on series / built_on / equivalent_to")


def load(p):
    return json.loads(Path(p).read_text())


def rows(run_dir):
    return [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 4), round((c + h) / d, 4))


def bootstrap_by_item(items, key, n_boot=2000, seed=0):
    """95% CI of the mean of `key`, resampling items (all samples of an item together)."""
    by = {}
    for r in items:
        by.setdefault(r["doc_id"], []).append(1.0 if r[key] else 0.0)
    ids = sorted(by); rng = random.Random(seed); means = []
    for _ in range(n_boot):
        pick = [by[rng.choice(ids)] for _ in ids]
        flat = [x for g in pick for x in g]; means.append(sum(flat) / len(flat))
    means.sort()
    return (round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot) - 1], 4))


def summarize(protocol, rs):
    fam = protocol.split("/")[0]
    if fam == "gpqa-diamond":
        out = {"samples": len(rs), "questions": len({r["doc_id"] for r in rs}),
               "passes": sorted({r["pass"] for r in rs}), "empty": sum(r["empty"] for r in rs)}
        for k in ("correct_flexible", "correct_strict"):
            out[k.replace("correct_", "accuracy_")] = round(sum(r[k] for r in rs) / len(rs), 4)
        out["accuracy_flexible_ci95"] = bootstrap_by_item(rs, "correct_flexible")
        answered = [r for r in rs if not r["empty"]]
        out["accuracy_flexible_answered"] = round(sum(r["correct_flexible"] for r in answered) / max(1, len(answered)), 4)
        return out
    if fam == "hard-prompt-screen":
        c = Counter(r["cls"] for r in rs)
        per = {}
        for r in rs:
            per.setdefault(str(r["doc_id"]), Counter())[r["cls"]] += 1
        return {"requests": len(rs), "ok": c["ok"], "loop": c["loop"], "exhaust": c["exhaust"], "error": c["error"],
                "non_ok": c["loop"] + c["exhaust"], "non_ok_ci95": wilson(c["loop"] + c["exhaust"], len(rs)),
                "per_doc": {d: dict(v) for d, v in sorted(per.items(), key=lambda x: int(x[0]))}}
    if fam == "kpool-kernel-tests":
        c = Counter(r["outcome"] for r in rs)
        return {"tests": len(rs), "passed": c["passed"], "failed": c["failed"], "skipped": c["skipped"],
                "failed_tests": sorted(r["test"] for r in rs if r["outcome"] == "failed")}
    if fam == "serving-probe":
        out = {}
        for b in sorted({r["batch"] for r in rs}):
            req = [r for r in rs if r["batch"] == b and r["kind"] == "request"]
            meta = next((r for r in rs if r["batch"] == b and r["kind"] == "batch"), {})
            gen = sum(r["tokens"] for r in req); spec = meta.get("spec")
            out[b] = {"requests": len(req), "median_decode_tok_s": median([r["decode_tok_s"] for r in req]),
                      "median_ttft_s": median([r["ttft_s"] for r in req]),
                      "aggregate_tok_s": round(gen / meta["wall_s"], 1) if meta.get("wall_s") else None,
                      "acceptance_rate": round(spec["accepted"] / spec["drafted"], 4) if spec else None,
                      "mean_acceptance_length": round(1 + spec["accepted"] / spec["drafts"], 3) if spec else None}
        return {"batches": out}
    raise ValueError(f"unknown protocol family {fam}")


def median(xs):
    xs = sorted(xs); n = len(xs)
    return None if n == 0 else (xs[n // 2] if n % 2 else round((xs[n // 2 - 1] + xs[n // 2]) / 2, 3))


def same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def check_run(run_dir):
    rid = run_dir.name
    try:
        m = load(run_dir / "run.json")
    except Exception as e:
        fails.append(f"{rid}: run.json unreadable ({e})"); return None
    missing = RUN_KEYS - set(m)
    if missing: fails.append(f"{rid}: run.json missing {sorted(missing)}")
    if m.get("id") != rid: fails.append(f"{rid}: id field is {m.get('id')!r}")
    proto = m.get("protocol", "")
    if not (ROOT / "protocols" / f"{proto}.md").exists(): fails.append(f"{rid}: protocol {proto!r} not found")
    cfg = m.get("config", "")
    if cfg and not (ROOT / "configs" / f"{cfg}.json").exists(): fails.append(f"{rid}: config {cfg!r} not found")
    try:
        stored = load(run_dir / "summary.json"); fresh = summarize(proto, rows(run_dir))
        diff = [k for k in fresh if not same(fresh[k], stored.get(k))]
        if diff: fails.append(f"{rid}: summary.json disagrees with results.jsonl on {diff}")
    except Exception as e:
        fails.append(f"{rid}: could not recompute summary ({e})")
    return m


def flatten(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flatten(v, f"{pre}{k}."))
        else: out[f"{pre}{k}"] = v
    return out


def check_comparison(cdir, manifests):
    c = load(cdir / "comparison.json"); cid = cdir.name
    runs = [manifests.get(r) for r in c["runs"]]
    if None in runs:
        fails.append(f"comparison {cid}: unknown run(s) {[r for r, m in zip(c['runs'], runs) if m is None]}"); return
    if not c.get("comparable", True):
        if not c.get("reason"): fails.append(f"comparison {cid}: comparable=false needs a reason")
        return
    protos = {m["protocol"] for m in runs}
    if protos != {c["protocol"]}: fails.append(f"comparison {cid}: protocols {sorted(protos)} != {c['protocol']}")
    cfgs = [flatten(load(ROOT / "configs" / f"{m['config']}.json")) for m in runs]
    keys = {k for k in set().union(*cfgs) if not any(k == d or k.startswith(d + ".") for d in DESCRIPTIVE)}
    allowed = set(c.get("varies", []))
    for k in sorted(keys):
        vals = {json.dumps(x.get(k), sort_keys=True) for x in cfgs}
        if len(vals) > 1 and not any(k == a or k.startswith(a + ".") for a in allowed):
            fails.append(f"comparison {cid}: configs differ in {k!r}, not declared in 'varies'")


def main(args):
    targets = [ROOT / a for a in args] if args else sorted((ROOT / "runs").iterdir())
    manifests = {}
    for d in targets:
        if d.is_dir() and (d / "run.json").exists():
            m = check_run(d)
            if m: manifests[d.name] = m
    if not args:
        check_labels([load(p) for p in sorted((ROOT / "configs").rglob("*.json"))])
        for cdir in sorted((ROOT / "comparisons").iterdir()):
            if (cdir / "comparison.json").exists(): check_comparison(cdir, manifests)
    print(f"checked {len(manifests)} run(s)" + ("" if args else f", {len([1 for x in (ROOT/'comparisons').iterdir() if (x/'comparison.json').exists()])} comparison(s)"))
    for f in fails: print("FAIL", f)
    print("OK" if not fails else f"{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
