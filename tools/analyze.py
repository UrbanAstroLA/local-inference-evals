#!/usr/bin/env python3
"""Analyses recomputed from published rows only (standard library).

    python3 tools/analyze.py gpqa-pass1       # per-question paired comparison of every GPQA run's pass 1
    python3 tools/analyze.py gpqa-empties     # empty answers per run, with exact 95% CIs projected to 594 answers
    python3 tools/analyze.py gpqa-passes      # per-pass accuracy of each full run, spread, and whether inside the run's CI
    python3 tools/analyze.py gpqa-pairs       # full runs pairwise: accuracy sign test over 594 answers, empty answers
                                              # (Fisher exact, and a 95% interval for the difference resampling questions)
    python3 tools/analyze.py screen-speed     # hard-question screens: median completion tokens/s of requests that finished
    python3 tools/analyze.py screen-pool CONFIG [CONFIG ...]
                                              # protocol-v1 screens pooled per config: failures, Wilson 95% CI, per question,
                                              # and pairwise Fisher exact tests (config ids without the family, e.g. k3.25-v0.9.0-dflash3)

Configurations are named by their labels (SCHEMA.md, "Labels") and config ids. Every GPQA request carried seed 1234
(see protocols/gpqa-diamond/v1.md); passes differ through batching nondeterminism.
"""
import json, math, random, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from verify import config_label  # noqa: E402


def runs(protocol_prefix):
    for d in sorted((ROOT / "runs").iterdir()):
        m = json.loads((d / "run.json").read_text())
        if m["protocol"].startswith(protocol_prefix):
            m["rows"] = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines() if l.strip()]
            m["cfg"] = json.loads((ROOT / "configs" / f"{m['config']}.json").read_text())
            yield m


def name(m):
    return f"{config_label(m['cfg'])} ({m['config'].split('/', 1)[1]})"


def gpqa_runs():
    for m in runs("gpqa-diamond/"):
        yield name(m), m["rows"]


def sign_test(a, b):
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2 ** n)


def cp(k, n, a=0.05):
    cdf = lambda p, kk: sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(kk + 1))
    def bis(cond):
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if cond(mid) else (lo, mid)
        return (lo + hi) / 2
    return (0.0 if k == 0 else bis(lambda p: 1 - cdf(p, k - 1) < a / 2), bis(lambda p: cdf(p, k) > a / 2))


def pass1():
    runs = {c: {r["doc_id"]: r for r in rows if r["pass"] == 1} for c, rows in gpqa_runs()}
    names = sorted(runs); hashes = {c: {d: r["prompt_hash"] for d, r in v.items()} for c, v in runs.items()}
    assert all(hashes[c] == hashes[names[0]] for c in names), "prompt hashes differ between runs"
    print("| config (pass 1) | accuracy | empty |"); print("|---|---|---|")
    for c in names:
        v = runs[c].values(); print(f"| {c} | {100 * sum(r['correct_flexible'] for r in v) / len(runs[c]):.1f}% | {sum(r['empty'] for r in v)} |")
    print("\n| A | B | only A right | only B right | sign-test p |"); print("|---|---|---|---|---|")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x = sum(runs[a][d]["correct_flexible"] and not runs[b][d]["correct_flexible"] for d in runs[a])
            y = sum(runs[b][d]["correct_flexible"] and not runs[a][d]["correct_flexible"] for d in runs[a])
            print(f"| {a} | {b} | {x} | {y} | {sign_test(x, y):.2f} |")


def fisher(a, b, c, d):
    """Two-sided Fisher exact test for the 2x2 table [[a, b], [c, d]]."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    p = lambda x: math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    p0 = p(a)
    return sum(p(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if p(x) <= p0 * (1 + 1e-9))


def full_runs():
    return [(n, rows) for n, rows in gpqa_runs() if len({r["pass"] for r in rows}) == 3]


def passes():
    print("| config | pass 1 | pass 2 | pass 3 | spread (points) | all inside 95% CI |"); print("|---|---|---|---|---|---|")
    for m in runs("gpqa-diamond/"):
        ps = sorted({r["pass"] for r in m["rows"]})
        if len(ps) != 3: continue
        acc = [100 * sum(r["correct_flexible"] for r in m["rows"] if r["pass"] == p) / sum(r["pass"] == p for r in m["rows"]) for p in ps]
        lo, hi = (100 * x for x in json.loads((ROOT / "runs" / m["id"] / "summary.json").read_text())["accuracy_flexible_ci95"])
        print(f"| {name(m)} | " + " | ".join(f"{a:.1f}%" for a in acc) + f" | {max(acc) - min(acc):.1f} | {all(lo <= a <= hi for a in acc)} |")


def pairs(n_boot=10000, seed=0):
    full = full_runs()
    print("| A | B | accuracy: only A right / only B right (sign p) | empty A vs B (Fisher p) | B - A empty, 95% interval |")
    print("|---|---|---|---|---|")
    for i, (a, ra) in enumerate(full):
        for b, rb in full[i + 1:]:
            A = {(r["doc_id"], r["pass"]): r for r in ra}; B = {(r["doc_id"], r["pass"]): r for r in rb}
            x = sum(A[k]["correct_flexible"] and not B[k]["correct_flexible"] for k in A)
            y = sum(B[k]["correct_flexible"] and not A[k]["correct_flexible"] for k in A)
            ea, eb = sum(r["empty"] for r in ra), sum(r["empty"] for r in rb); n = len(ra)
            by = {}
            for (d, _), r in A.items(): by.setdefault(d, [0, 0])[0] += r["empty"]
            for (d, _), r in B.items(): by[d][1] += r["empty"]
            ids = sorted(by); rng = random.Random(seed); diffs = []
            for _ in range(n_boot):
                diffs.append(sum(by[d][1] - by[d][0] for d in (rng.choice(ids) for _ in ids)))
            diffs.sort()
            print(f"| {a} | {b} | {x} / {y} ({sign_test(x, y):.2f}) | {ea} vs {eb} ({fisher(ea, n - ea, eb, n - eb):.3f}) | "
                  f"{diffs[int(0.025 * n_boot)]} to {diffs[int(0.975 * n_boot) - 1]} |")


def screen_speed():
    print("| run | protocol | config | failures / 40 | finished | median completion tok/s of finished requests |")
    print("|---|---|---|---|---|---|")
    for m in runs("hard-prompt-screen/"):
        v = [r["completion_tokens"] / r["secs"] for r in m["rows"] if r["finish_reason"] == "stop" and r.get("completion_tokens")]
        fails = sum(r["cls"] in ("loop", "exhaust") for r in m["rows"])
        print(f"| {m['id']} | {m['protocol'].split('/')[1]} | {config_label(m['cfg'])} | {fails} | {len(v)} | "
              + (f"{st.median(v):.1f}" if v else "-") + " |")


def screen_pool(cfgs):
    from verify import wilson
    pooled = {}
    for m in runs("hard-prompt-screen/v1"):
        c = m["config"].split("/", 1)[1]
        if c in cfgs and "INVALID" not in m["notes"]:
            pooled.setdefault(c, {"m": m, "runs": [], "rows": []}); pooled[c]["runs"].append(m["id"]); pooled[c]["rows"] += m["rows"]
    print("| config | screens | failures / requests | 95% Wilson | loops | exhaustions | failures by question (13, 79, 88, 121, 127) |")
    print("|---|---|---|---|---|---|---|")
    for c in cfgs:
        p = pooled[c]; rs = p["rows"]; k = sum(r["cls"] in ("loop", "exhaust") for r in rs); lo, hi = wilson(k, len(rs))
        q = [sum(r["cls"] in ("loop", "exhaust") for r in rs if r["doc_id"] == d) for d in (13, 79, 88, 121, 127)]
        print(f"| {config_label(p['m']['cfg'])} | {len(p['runs'])} | {k}/{len(rs)} | {100 * lo:.1f}-{100 * hi:.1f}% | "
              f"{sum(r['cls'] == 'loop' for r in rs)} | {sum(r['cls'] == 'exhaust' for r in rs)} | {', '.join(map(str, q))} |")
    print("\n| A | B | failures A vs B | Fisher exact p |"); print("|---|---|---|---|")
    for i, a in enumerate(cfgs):
        for b in cfgs[i + 1:]:
            ra, rb = pooled[a]["rows"], pooled[b]["rows"]
            ka, kb = (sum(r["cls"] in ("loop", "exhaust") for r in x) for x in (ra, rb))
            print(f"| {a} | {b} | {ka}/{len(ra)} vs {kb}/{len(rb)} | {fisher(ka, len(ra) - ka, kb, len(rb) - kb):.4f} |")


def empties():
    print("| config | passes | empty | answers | projected empty at 594 (exact 95% CI) |"); print("|---|---|---|---|---|")
    for c, rows in gpqa_runs():
        k, n = sum(r["empty"] for r in rows), len(rows); lo, hi = cp(k, n)
        print(f"| {c} | {len({r['pass'] for r in rows})} | {k} | {n} | {594 * k / n:.0f} ({594 * lo:.1f}-{594 * hi:.1f}) |")


if __name__ == "__main__":
    {"gpqa-pass1": pass1, "gpqa-empties": empties, "gpqa-passes": passes, "gpqa-pairs": pairs,
     "screen-speed": screen_speed}.get(sys.argv[1] if len(sys.argv) > 1 else "gpqa-pass1", lambda: screen_pool(sys.argv[2:]))()
