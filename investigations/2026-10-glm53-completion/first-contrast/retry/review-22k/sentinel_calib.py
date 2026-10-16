#!/usr/bin/env python3
"""Amendment 22k s.1 (post-hoc CPU review): sentinel calibration. Input: the per-row distances recomputed from the
session's snapshots by ../diagnosis-fcr1/diag_sentinel.py (sentinel.state_rows unchanged). Original gate: PS-c vs PS-a
within 2 x rel(PS-a, PS-b) unless PS-a = PS-b bitwise. Standard library only."""
import json, collections, statistics as st
rows = [r for r in json.load(open("sentinel_rows.json")) if "rel_ab" in r]
def pct(x, xs): return 100.0 * sum(v <= x for v in xs) / len(xs)
for comp in ("ssm", "conv"):
    R = [r for r in rows if r["comp"] == comp]
    ex = [r for r in R if r["eq_ab"]]; nb = [r for r in R if not r["eq_ab"]]
    print(f"== {comp}: {len(R)} rows (8 fixtures x 2 ranks x KDA layers)")
    print(f"  exact equality: PS-a = PS-b in {len(ex)} rows; in all of them PS-c = PS-a: {all(r['rel_ac'] == 0 for r in ex)}; "
          f"by fixture {dict(collections.Counter(r['fixture'] for r in ex))}; by layer {dict(sorted(collections.Counter(r['layer'] for r in ex).items()))}")
    q = lambda xs: "median %.4f, p10 %.4f, p90 %.4f, max %.4f" % (st.median(xs), sorted(xs)[len(xs)//10], sorted(xs)[9*len(xs)//10], max(xs))
    for nm in ("rel_ab", "rel_ac", "rel_bc"):
        print(f"  {nm} over the {len(nb)} non-identical rows: {q([r[nm] for r in nb])}")
    o_a = [r for r in nb if r["rel_ac"] > 2 * r["rel_ab"]]                       # the gate (reference PS-a)
    o_b = [r for r in nb if r["rel_bc"] > 2 * r["rel_ab"]]                       # reference PS-b
    both = [r for r in nb if r["rel_ac"] > 2 * r["rel_ab"] and r["rel_bc"] > 2 * r["rel_ab"]]
    mir_b = [r for r in nb if r["rel_ab"] > 2 * r["rel_ac"]]                     # PS-b judged against PS-c as baseline
    mir_a = [r for r in nb if r["rel_ab"] > 2 * r["rel_bc"]]                     # PS-a judged against PS-c as baseline
    print(f"  fail, reference PS-a (the gate): {len(o_a)}; reference PS-b: {len(o_b)}; both: {len(both)}")
    print(f"  mirror (a pre-injection phase judged against the other two): PS-b vs (PS-a, PS-c) {len(mir_b)}; PS-a vs (PS-b, PS-c) {len(mir_a)}")
    print(f"  failing rows by fixture {dict(collections.Counter(r['fixture'] for r in o_a))}; by layer {dict(sorted(collections.Counter(r['layer'] for r in o_a).items()))}; "
          f"rank pairs (same fixture and layer on both ranks): {sum(1 for k, v in collections.Counter((r['fixture'], r['layer']) for r in o_a).items() if v == 2)}")
    print("  small denominators: percentile of each failing row's rel(PS-a, PS-b) within its fixture and within its layer")
    for r in sorted(o_a, key=lambda r: r["rel_ac"] / r["rel_ab"], reverse=True):
        fx = [x["rel_ab"] for x in nb if x["fixture"] == r["fixture"]]
        ly = [x["rel_ab"] for x in nb if x["layer"] == r["layer"]]
        lyc = [x["rel_ac"] for x in nb if x["layer"] == r["layer"]]
        print("   %s r%d L%-2d ab %.4f (fixture pct %4.1f, layer pct %4.1f) ac %.4f bc %.4f ratio %.2f; ac's pct within layer %4.1f" % (
            r["fixture"], r["rank"], r["layer"], r["rel_ab"], pct(r["rel_ab"], fx), pct(r["rel_ab"], ly),
            r["rel_ac"], r["rel_bc"], r["rel_ac"] / r["rel_ab"], pct(r["rel_ac"], lyc)))
    lo = [r for r in nb if pct(r["rel_ab"], [x["rel_ab"] for x in nb if x["layer"] == r["layer"]]) <= 10]
    print(f"  rows in the lowest decile of rel(PS-a, PS-b) within their layer: {len(lo)}, of which failing {sum(r in o_a for r in lo)}; "
          f"other rows: {len(nb) - len(lo)}, failing {sum(r in o_a for r in nb if r not in lo)}")

print("\n== symmetric odd-one-out count (each phase farther than 2x from BOTH others than they are from each other)")
for comp in ("ssm", "conv"):
    nb = [r for r in rows if r["comp"] == comp and not r["eq_ab"]]
    odd = {"PS-a (before the injection)": [r for r in nb if min(r["rel_ab"], r["rel_ac"]) > 2 * r["rel_bc"]],
           "PS-b (before the injection)": [r for r in nb if min(r["rel_ab"], r["rel_bc"]) > 2 * r["rel_ac"]],
           "PS-c (after the injection)": [r for r in nb if min(r["rel_ac"], r["rel_bc"]) > 2 * r["rel_ab"]]}
    for k, v in odd.items():
        pairs = sum(1 for x, c in collections.Counter((r["fixture"], r["layer"]) for r in v).items() if c == 2)
        print(f"  {comp}: {k} odd one out in {len(v)} of {len(nb)} rows (both ranks at the same fixture and layer: {pairs}); "
              f"fixtures {dict(collections.Counter(r['fixture'] for r in v))}")
