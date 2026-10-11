"""Aggregate over row counts: fraction of (row,rep) where col 511 holds a non-tie pool with
reference rank < 511 / < 480 / < 256, per version x path-group x pool count. CPU, stdlib."""
import json, sys
from collections import defaultdict
kind = sys.argv[1]
agg = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
for v in ["v091", "v070"]:
    for line in open(f"results/raw-{v}-full.jsonl"):
        r = json.loads(line)
        if r["kind"] != kind:
            continue
        rows = r["rows"]
        if v == "v091":
            grp = "0.9.1 fused (rows<=16)" if (r["path"] == "auto" and rows <= 16) else ("0.9.1 tiled (rows>16)" if r["path"] == "auto" else "0.9.1 tiled forced (rows<=16)")
        else:
            grp = "0.7.0 fused+merge (rows<=16)" if (r["path"] == "auto" and rows <= 16) else ("0.7.0 tiled+merge (rows>16)" if r["path"] == "auto" else "0.7.0 tiled forced+merge (rows<=16)")
        a = agg[(grp, r["pools"])]
        for m in r["per_row"]:
            a[0] += 1
            rk = m["col511_rank"]
            if rk is None or m["col511_tie_exact"]:
                continue
            a[1] += rk < 511
            a[2] += rk < 480
            a[3] += rk < 256
            a[4] += (rk < 511 and not m["col511_tie_tol"])
        a[5] += sum(1 for x in r["rtr"] if x["col511_distinct"] > 1) / max(1, len(r["rtr"])) * len(r["per_row"])
print(f"| path | pools | n | rank<511 | rank<480 | rank<256 | rank<511 (tol) | rtr col511 varies |")
print("|---|---|---|---|---|---|---|---|")
for (g, n), a in sorted(agg.items()):
    print(f"| {g} | {n} | {a[0]} | {100*a[1]/a[0]:.1f}% | {100*a[2]/a[0]:.1f}% | {100*a[3]/a[0]:.1f}% | {100*a[4]/a[0]:.1f}% | {100*a[5]/a[0]:.0f}% |")
