#!/usr/bin/env python3
"""audit-s0: reconcile the published point-0 counts (diagnostics/s0map.out, RESULT-STAGE0.md table) with the
category split of recount_point0_blocks.csv.  Pure CSV arithmetic (public data only).  Writes /out/reconcile.out."""
import collections
import csv

rows = list(csv.DictReader(open("/out/recount_point0_blocks.csv")))
L = []


def tally(pred):
    c = collections.Counter()
    for r in rows:
        if pred(r):
            c[r["category"]] += 1
            c[r["category"] + ":differ"] += int(r["equal"] == "0")
    return c


for var in ("V1", "V2"):
    L.append(f"== {var}, point 0")
    # s0map 'attn' family = names ending '.self_attn.attn' => MLA layers AND draft layers 45-49
    attn = tally(lambda r: r["variant"] == var and r["tensor"].endswith(".self_attn.attn"))
    tot = sum(v for k, v in attn.items() if ":" not in k)
    dif = sum(v for k, v in attn.items() if k.endswith(":differ"))
    L.append(f"s0map 'attn' family = {dif}/{tot}; split: " + ", ".join(
        f"{k}={attn[k + ':differ']}/{attn[k]}" for k in sorted(k for k in attn if ':' not in k)))
    L.append("   of which draft layers 45-49: %d blocks" % sum(
        1 for r in rows if r["variant"] == var and r["family"] == "draft"))
    idx = tally(lambda r: r["variant"] == var and r["family"] == "idxk")
    L.append("s0map 'indexer.k_cache' = %d/%d; split: %s" % (
        sum(v for k, v in idx.items() if k.endswith(":differ")), sum(v for k, v in idx.items() if ":" not in k),
        ", ".join(f"{k}={idx[k + ':differ']}/{idx[k]}" for k in sorted(k for k in idx if ':' not in k))))
    for fam in ("kda_conv", "kda_rec"):
        t = tally(lambda r: r["variant"] == var and r["family"] == fam and r["layer"] == "0")
        L.append(f"layer 0 {fam}: " + ", ".join(f"{k}={t[k + ':differ']}/{t[k]}" for k in sorted(k for k in t if ':' not in k)))
open("/out/reconcile.out", "w").write("\n".join(L) + "\n")
print("\n".join(L))
