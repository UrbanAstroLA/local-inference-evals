#!/usr/bin/env python3
"""Analyses recomputed from published rows only (standard library).

    python3 tools/analyze.py gpqa-pass1     # per-question paired comparison of every GPQA run's pass 1 (same seed)
    python3 tools/analyze.py gpqa-empties   # empty answers per run, with exact 95% CIs projected to 594 answers
"""
import json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def gpqa_runs():
    for d in sorted((ROOT / "runs").iterdir()):
        m = json.loads((d / "run.json").read_text())
        if m["protocol"] == "gpqa-diamond/v1":
            rows = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines() if l.strip()]
            yield m["config"].split("/", 1)[1], rows


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
    print("| config (pass 1, seed 1235) | accuracy | empty |"); print("|---|---|---|")
    for c in names:
        v = runs[c].values(); print(f"| {c} | {100 * sum(r['correct_flexible'] for r in v) / len(runs[c]):.1f}% | {sum(r['empty'] for r in v)} |")
    print("\n| A | B | only A right | only B right | sign-test p |"); print("|---|---|---|---|---|")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x = sum(runs[a][d]["correct_flexible"] and not runs[b][d]["correct_flexible"] for d in runs[a])
            y = sum(runs[b][d]["correct_flexible"] and not runs[a][d]["correct_flexible"] for d in runs[a])
            print(f"| {a} | {b} | {x} | {y} | {sign_test(x, y):.2f} |")


def empties():
    print("| config | passes | empty | answers | projected empty at 594 (95% CI) |"); print("|---|---|---|---|---|")
    for c, rows in gpqa_runs():
        k, n = sum(r["empty"] for r in rows), len(rows); lo, hi = cp(k, n)
        print(f"| {c} | {len({r['pass'] for r in rows})} | {k} | {n} | {594 * k / n:.0f} ({594 * lo:.1f}-{594 * hi:.1f}) |")


if __name__ == "__main__":
    {"gpqa-pass1": pass1, "gpqa-empties": empties}[sys.argv[1] if len(sys.argv) > 1 else "gpqa-pass1"]()
