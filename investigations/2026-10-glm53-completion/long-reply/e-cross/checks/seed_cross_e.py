#!/usr/bin/env python3
"""E seed range cross-check (PROPOSAL.md section 3.2). Host, standard library, read only. Writes seeds and counts only.

Complements seed_scan_e.py (rep070's seed_scan.py, only SKIP_DIRS changed) and raw_seed_scan_e.py (d091's, only SKIP
changed), which cannot see seeds produced by a formula at run time:
  1. the planned seeds of every earlier fixed-prefix / replication package (B+ out/plan.json, d091 and d091b (D2)
     run/seeds-planned.json, rep070-F run-F/seeds-planned.json) must not intersect E's 64 seeds;
  2. formula bases: every 7-10 digit integer in a line mentioning 'seed' in *.py / *.sh under <data-runs>, the public repo
     and <eval-tools> (this package excluded) is listed with its count, and none may fall in E's range;
  3. the known formula families (B+ section 3.2 and its seed-formula-scan.txt) are evaluated over their stated domains.
usage: seed_cross_e.py OUT.txt
"""
import glob, json, os, re, sys

EC = "<runs>/long-reply/e-cross"
B = "<runs>"
BASE = 8260000
E_SEEDS = sorted(BASE + 10000 * e + 100 * p + j for e in (1, 2) for p in range(1, 14) for j in range(1, 5))   # 22ah: 13 prefixes
LO, HI = E_SEEDS[0], E_SEEDS[-1]
ROOTS = ["<data-runs>", "<this-repo>", "<eval-tools>"]


def seeds_in(obj):
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "seed" in str(k).lower() and isinstance(v, int):
                out.add(v)
            out |= seeds_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= seeds_in(v)
    return out


def main():
    lines = [f"# E seed cross-check; E range {LO:,} - {HI:,}: seed = {BASE:,} + 10,000*engine + 100*prefix + j "
             f"(engine 1 = v070, 2 = v091; prefix 1-13; j 1-4); {len(E_SEEDS)} seeds, {len(set(E_SEEDS))} distinct"]
    ok = len(set(E_SEEDS)) == 104
    plans = {"B+ out/plan.json": f"{B}/long-reply/bplus/out/plan.json",
             "d091 run/seeds-planned.json": f"{B}/long-reply/d091/run/seeds-planned.json",
             "d091b (D2) run/seeds-planned.json": f"{B}/long-reply/d091b/run/seeds-planned.json",
             "rep070-F run-F/seeds-planned.json": f"{B}/rep070/run-F/seeds-planned.json"}
    for name, p in plans.items():
        if not os.path.exists(p):
            lines.append(f"{name}: MISSING ({p})")
            ok = False
            continue
        s = seeds_in(json.load(open(p)))
        if name.startswith("B+"):
            s |= {v for r in json.load(open(p))["order"] for v in r["seeds"].values()}
        inter = sorted(s & set(E_SEEDS))
        lines.append(f"{name}: {len(s)} seeds (min {min(s) if s else None}, max {max(s) if s else None}); intersection with E: {inter}")
        ok &= not inter
    rx = re.compile(r"\b\d[\d_,]{6,12}\b")
    found = {}
    for root in ROOTS:
        for p in glob.glob(f"{root}/**/*.py", recursive=True) + glob.glob(f"{root}/**/*.sh", recursive=True):
            if p.startswith(EC) or "/site-packages/" in p or "/venv/" in p:
                continue
            try:
                for ln in open(p, errors="replace"):
                    if "seed" not in ln.lower():
                        continue
                    for m in rx.findall(ln):
                        v = int(m.replace("_", "").replace(",", ""))
                        if 1_000_000 <= v <= 99_999_999:
                            found.setdefault(v, set()).add(p)
            except OSError:
                pass
    lines.append(f"# 7-8 digit integers on seed lines in *.py / *.sh (site-packages and venvs excluded): {len(found)} distinct")
    for v in sorted(found):
        lines.append(f"  {v:>10,}  in {len(found[v])} file(s), e.g. {sorted(found[v])[0]}")
    near = sorted(v for v in found if LO - 20000 <= v <= HI + 20000)
    lines.append(f"values within 20,000 of the E range: {near}")
    ok &= not any(LO <= v <= HI for v in found)
    fam = {
        "warm-up 7,810,000 + i (i < 100)": range(7810000, 7810100),
        "warm-up 8,810,000 + i (i < 100)": range(8810000, 8810100),
        "GPQA fixture 9,400,000 + d (d < 198)": range(9400000, 9400198),
        "phase fixtures 9,850,000 + 1000*phase + 10*fixture + extra (< 9,950,000)": range(9850000, 9950000),
        "col511 fixture 8,000,000 + 10,000*ord(case) + 10*rows + d (case a-d, rows <= 64, d < 16)":
            sorted({8000000 + 10000 * ord(c) + 10 * r + d for c in "abcd" for r in (1, 4, 16, 24, 64) for d in range(16)}),
        "col511 fixture 9,100,000 + 100*ord(case) + rows": sorted({9100000 + 100 * ord(c) + r for c in "abcd" for r in (1, 4, 16, 24, 64)}),
        "col511 seed_for 7,000,000 + 100,000*kind + 1,000*n_idx + 10*rows + d (kind, n_idx < 10)":
            sorted({7000000 + 100000 * k + 1000 * n + 10 * r + d for k in range(10) for n in range(10) for r in (1, 4, 16, 24, 64) for d in range(16)}),
        "B+ 8,160,000 + 10,000*engine + 100*prefix + j": sorted(8160000 + 10000 * e + 100 * p + j for e in (1, 2, 3) for p in range(1, 13) for j in range(1, 5)),
    }
    for name, r in fam.items():
        inter = sorted(set(r) & set(E_SEEDS))
        lines.append(f"family {name}: range {min(r):,} - {max(r):,}; intersection with E: {inter}")
        ok &= not inter
    lines.append(f"VERDICT: {'E range unused by every source checked' if ok else 'COLLISION OR MISSING INPUT'}")
    open(sys.argv[1], "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:6] + lines[-12:]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
