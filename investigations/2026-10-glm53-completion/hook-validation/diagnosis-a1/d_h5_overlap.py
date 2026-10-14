"""Are the B1-arm gross pools (V4 decode vs rebuild) position-specific fragile pools that are also unstable between
two PREFILL builds (V3ref, prefill vs prefill)? Traces present in both V4 and V3: join on (trace, layer, pool)."""
import json, collections, numpy as np
P = json.load(open("/d/out/h4c_nulls.json"))["P"]
pp = collections.defaultdict(list)
for x in P:
    pp[(x["g"].rsplit("-", 1)[-1], x["layer"], x["pool"])].append(x["rel"])
ppmax = {k: max(v) for k, v in pp.items()}
R = [json.loads(l) for l in open("/d/out/h5_pools.jsonl")]
out = {}
for arm in ("b1", "d2solo"):
    A = [r for r in R if r["arm"] == arm and (r["tid"], r["layer"], r["pool"]) in ppmax]
    G = [r for r in A if r["rel"] > .25]; N = [r for r in A if r["rel"] <= .25]
    f = lambda S: (np.mean([ppmax[(r["tid"], r["layer"], r["pool"])] > .25 for r in S]) if S else float("nan"),
                   np.median([ppmax[(r["tid"], r["layer"], r["pool"])] for r in S]) if S else float("nan"))
    out[arm] = dict(joined=len(A), gross=len(G), gross_also_prefill_unstable=f(G), nongross_prefill_unstable=f(N))
    print(arm, "pools joined", len(A), "| decode-vs-rebuild gross", len(G),
          "| of those, prefill-vs-prefill max rel > .25: %.3f (median max rel %.3f)" % f(G),
          "| non-gross pools: %.4f (median %.4f)" % f(N))
    for r in sorted(G, key=lambda r: -r["rel"])[:15]:
        print("   %s L%d pool %d  dec-vs-reb %.3f  prefill-vs-prefill max %.3f (n=%d)" % (r["tid"], r["layer"], r["pool"], r["rel"],
              ppmax[(r["tid"], r["layer"], r["pool"])], len(pp[(r["tid"], r["layer"], r["pool"])])))
json.dump(out, open("/d/out/h5_overlap.json", "w"), indent=1)
