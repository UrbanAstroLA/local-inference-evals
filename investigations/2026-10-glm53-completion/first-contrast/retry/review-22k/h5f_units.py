#!/usr/bin/env python3
"""Amendment 22k s.1 (post-hoc CPU review, existing records only): H5f like with like.
The exact clean-labelled (fixture, rank, layer, window pool) units of the D runs, each compared on the same unit with:
the clean arm (C1-C3 vs LW1, the H5f reference), the decode null N_dd (C1 vs C2) and the prefill null N_pp (LW2 vs LW1).
Counts and identities; M4's F6 fragility flags (GATE-1R C1 / C2, C3 residue) attached by physical page."""
import sys, json, collections
sys.path.insert(0, "/fc/analysis"); sys.path.insert(0, "/fc/retry/analysis")
import fc_common as fc, fcr_geometry as geo, sentinel as sn
G = geo.load("/fc"); run = fc.Run("/run")
GROSS, BLOCK = fc.GROSS, 4352
res = json.load(open("/run/fcr_result.json"))
flags = set()
for src, v in res["f6_not_independently_verified"].items():
    if isinstance(v, dict):
        for k, x in v.items():
            if isinstance(x, dict):
                for p in x.get("pages", []):
                    flags.add((src, k, p["fixture"], p["rank"], p["layer"], p["page"]))

def page_of(tag, rank, pool):
    W = run.win_dump(tag)
    if not W or rank not in W: return None
    a = run.load(W[rank])
    ids, bpk = sn.g0_ids(a); p = 4 * pool
    i = (p // BLOCK) * bpk
    return None if i >= len(ids) else (ids[i] // bpk) * 17 + ((p // 4) % 1088) // 64

units = collections.defaultdict(dict)      # (n, rank, L, k) -> {D-run: rel}
for ph in ("D1", "D2", "D3"):
    for n in G["order"]:
        lo, hi = G["fx"][n]["ref_pools"]; P = G["fx"][n]["P"]
        a = run.req(f"{ph}-{n}")
        if a is None: continue
        pc = a["pool_class"].numpy()
        for (rank, L, pool), v in fc.compare_pools(run, f"{ph}-{n}", f"LW1-{n}", lo, hi).items():
            k = pool - P // 4
            if 0 <= k < pc.size and pc[k] == 1:
                units[(n, rank, L, k)][ph] = v
comps = {"C1": ("C1", "LW1"), "C2": ("C2", "LW1"), "C3": ("C3", "LW1"), "N_dd": ("C1", "C2"), "N_pp": ("LW2", "LW1")}
cache = {}
rows = []
for (n, rank, L, k), dv in sorted(units.items()):
    lo, hi = G["fx"][n]["ref_pools"]; P = G["fx"][n]["P"]; pool = P // 4 + k
    row = dict(fixture=n, rank=rank, layer=L, window_pool=k, D=dv)
    for name, (x, y) in comps.items():
        key = (x, y, n)
        if key not in cache: cache[key] = fc.compare_pools(run, f"{x}-{n}", f"{y}-{n}", lo, hi)
        row[name] = cache[key].get((rank, L, pool))
    pgs = {t: page_of(f"{t}-{n}", rank, pool) for t in ("C1", "C2", "C3")}
    row["f6_flag"] = sorted({f"{s}:{t}" for (s, t, f, r, l, pg) in flags if f == n and r == rank and l == L
                             and pgs.get(t) == pg})
    rows.append(row)
json.dump(rows, open("/out/h5f_units.json", "w"), indent=0)
g = lambda v: v is not None and v > GROSS
print("clean-labelled units (fixture, rank, layer, window pool):", len(rows),
      "| by fixture:", dict(collections.Counter(r["fixture"] for r in rows)),
      "| labelled clean in 1/2/3 D runs:", dict(collections.Counter(len(r["D"]) for r in rows)))
for name in ("D", "C1", "C2", "C3", "N_dd", "N_pp"):
    if name == "D":
        obs = [(r, ph) for r in rows for ph, v in r["D"].items()]
        print("  D runs: unit-runs %d, gross %d" % (len(obs), sum(g(r["D"][ph]) for r, ph in obs)))
    else:
        vals = [r[name] for r in rows]
        print("  %-5s on the same units: measured %d, gross %d" % (name, sum(v is not None for v in vals), sum(g(v) for v in vals)))
print("units gross in any D run:")
for r in rows:
    if any(g(v) for v in r["D"].values()):
        print("  %s r%d L%d pool %d  D %s | C1 %s C2 %s C3 %s | N_dd %s N_pp %s | F6 flags %s" % (
            r["fixture"], r["rank"], r["layer"], r["window_pool"], {k: round(v, 3) for k, v in r["D"].items()},
            *[None if r[c] is None else round(r[c], 3) for c in ("C1", "C2", "C3", "N_dd", "N_pp")], r["f6_flag"] or "-"))
print("units gross in a clean-arm or null comparison but in no D run:")
for r in rows:
    if not any(g(v) for v in r["D"].values()) and any(g(r[c]) for c in ("C1", "C2", "C3", "N_dd", "N_pp")):
        print("  %s r%d L%d pool %d  D %s | C1 %s C2 %s C3 %s | N_dd %s N_pp %s" % (
            r["fixture"], r["rank"], r["layer"], r["window_pool"], {k: round(v, 3) for k, v in r["D"].items()},
            *[None if r[c] is None else round(r[c], 3) for c in ("C1", "C2", "C3", "N_dd", "N_pp")]))
print("where the clean-labelled units sit: window pools", dict(sorted(collections.Counter(r["window_pool"] for r in rows).items())))
