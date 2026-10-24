#!/usr/bin/env python3
"""audit-s0: de-duplicate recount_point0_blocks.csv to unique (tensor, rank, phys_r1, phys_r2) pairs -- windows of
one reply share cached prefix blocks, so the window-level counts repeat physical comparisons.  Writes /out/unique_pairs.out."""
import collections
import csv

rows = list(csv.DictReader(open("/out/recount_point0_blocks.csv")))
L = []
for var in ("V1", "V2"):
    for fam, cat in (("mla", "cached_full"), ("idxk", "cached_full"), ("kda_rec", "kda_cached_state"),
                     ("kda_conv", "kda_cached_state")):
        u = {}
        for r in rows:
            if r["variant"] == var and r["family"] == fam and r["category"] == cat:
                u[(r["tensor"], r["rank"], r["phys_r1"], r["phys_r2"])] = r["equal"]
        by = collections.defaultdict(lambda: [0, 0])
        for (t, rk, a, b), eq in u.items():
            lay = int(t.split("layers.")[1].split(".")[0].split("#")[0])
            by[lay][1] += 1
            by[lay][0] += int(eq == "0")
        tot = [sum(v[0] for v in by.values()), sum(v[1] for v in by.values())]
        L.append(f"{var} {fam:8s} {cat:16s} unique pairs differ/compared {tot[0]}/{tot[1]}; by layer: " +
                 " ".join(f"L{k}:{v[0]}/{v[1]}" for k, v in sorted(by.items())))
open("/out/unique_pairs.out", "w").write("\n".join(L) + "\n")
print("\n".join(L))
