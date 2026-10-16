#!/usr/bin/env python3
"""POST-HOC diagnostic (after data, not part of the frozen analysis): full distribution behind R-PS-state-ssm / -conv.
For every fixture x rank x KDA layer: rel(a,b), rel(a,c), rel(b,c), ratio rel(a,c)/rel(a,b). CPU only."""
import sys, json, collections
sys.path.insert(0, "/fc/analysis"); sys.path.insert(0, "/fc/retry/analysis")
import fc_common as fc, fcr_geometry as geo, sentinel as sn
G = geo.load("/fc"); run = fc.Run("/run")
have = ["PS-a", "PS-b", "PS-c"]
rows = []
for n in G["order"]:
    P = G["fx"][n]["P"]
    S = {}
    for ph in have:
        s_ = run.snaps(f"{ph}-{n}")
        S[ph] = {rk: run.load(p) for F, ranks in s_ for rk, p in ranks.items()} if s_ else {}
    r1 = sn.state_rows(n, P, have, S)                       # ref a, base b, compare c
    r2 = sn.state_rows(n, P, ["PS-b", "PS-a", "PS-c"], S)   # ref b: gives rel(b,c)
    for x, y in zip(r1, r2):
        for comp in ("ssm", "conv"):
            a = x[comp]; b = y[comp]
            if "rel_ac" not in a: rows.append(dict(fixture=n, rank=x["rank"], layer=x["layer"], comp=comp, bad=a)); continue
            rows.append(dict(fixture=n, rank=x["rank"], layer=x["layer"], comp=comp, eq_ab=a["eq_ab"], rel_ab=a["rel_ab"],
                             rel_ac=a["rel_ac"], rel_bc=b["rel_ac"], ok=a["ok"]))
json.dump(rows, open("/out/diag_sentinel.json", "w"), indent=0)
for comp in ("ssm", "conv"):
    R = [r for r in rows if r["comp"] == comp and "rel_ab" in r]
    nb = [r for r in R if not r["eq_ab"]]
    fail = [r for r in R if not r["ok"]]
    print(comp, "rows", len(R), "bitwise a=b", len(R) - len(nb), "FAIL", len(fail))
    print("  bitwise a=b rows: a=c bitwise?", collections.Counter(r["rel_ac"] == 0 for r in R if r["eq_ab"]))
    rat = sorted(r["rel_ac"] / r["rel_ab"] for r in nb if r["rel_ab"] > 0)
    q = lambda p: rat[min(len(rat) - 1, int(p * len(rat)))]
    print("  ratio rel_ac/rel_ab over non-bitwise rows: median %.2f p90 %.2f p99 %.2f max %.2f; >2: %d of %d"
          % (q(.5), q(.9), q(.99), rat[-1], sum(x > 2 for x in rat), len(rat)))
    rat2 = sorted(r["rel_bc"] / r["rel_ab"] for r in nb if r["rel_ab"] > 0)
    print("  ratio rel_bc/rel_ab: median %.2f p90 %.2f max %.2f; >2: %d" % (rat2[len(rat2)//2], rat2[int(.9*len(rat2))], rat2[-1], sum(x > 2 for x in rat2)))
    print("  failing by fixture:", dict(collections.Counter(r["fixture"] for r in fail)))
    for r in sorted(fail, key=lambda r: -r["rel_ac"] / max(r["rel_ab"], 1e-12)):
        print("   %s r%d L%d ab %.4f ac %.4f bc %.4f" % (r["fixture"], r["rank"], r["layer"], r["rel_ab"], r["rel_ac"], r["rel_bc"]))
    byfx = collections.defaultdict(list)
    for r in nb: byfx[r["fixture"]].append((r["rel_ab"], r["rel_ac"], r["rel_bc"]))
    for f, v in byfx.items():
        import statistics as st
        print("   fixture %s median ab %.4f ac %.4f bc %.4f (n=%d)" % (f, st.median(x[0] for x in v), st.median(x[1] for x in v), st.median(x[2] for x in v), len(v)))
