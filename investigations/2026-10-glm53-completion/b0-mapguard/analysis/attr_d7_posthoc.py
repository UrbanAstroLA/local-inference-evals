import json, collections, sys
RUN = sys.argv[1]
SITE_PF = 1  # idx_w_pf
BITS = ["mism","mism_spec","missing","missing_pred","unexp","inact","foreign","stale","wgroup","null","ownoff","oor","shared"]
for f, rank in (("mg_p370.jsonl", 0), ("mg_p404.jsonl", 1)):
    steps = {}; recs = []
    for l in open(f"{RUN}/mapguard/{f}"):
        d = json.loads(l)
        if d["kind"] == "step":
            steps[d["step"]] = {r[1]: r for r in d["rows"]}
        elif d["kind"] == "poll":
            recs += [r for r in d.get("records", []) if r["site"] == SITE_PF]
    c = collections.Counter(); starts = collections.Counter(); offs = collections.Counter(); other = []
    for r in recs:
        row = steps.get(r["step"], {}).get(r["tag"])
        if row is None:
            c["no_step_row"] += 1; continue
        lo, hi = row[2], row[3]
        unaligned = lo % 4 != 0
        first_pool = r["position"] < (lo // 4 + 1) * 4      # the pool that straddles the chunk start
        key = ("unaligned" if unaligned else "ALIGNED") + ("/first_pool" if first_pool else "/LATER_POOL")
        c[key] += 1
        starts[lo % 4352] += 1
        offs[r["position"] - lo] += 1
        if not (unaligned and first_pool) and len(other) < 5: other.append((r, row[:6]))
    print(f"rank {rank}: idx_w_pf records captured {len(recs)}; {dict(c)}")
    print("  chunk start mod 4352 (top):", starts.most_common(8))
    print("  position - chunk start:", sorted(offs.items())[:8])
    for o in other: print("  OTHER", o)
