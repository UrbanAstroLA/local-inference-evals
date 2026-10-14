"""Follow-up a2, check 3 side test: are gross units over-represented at the first pool of an index-K page
(pool % 64 == 0)? Counted per POSITION (fixture, pool), since one position event spans several layers.
Poisson tail for the observed count of page-first positions among gross positions. -> out/f3_pagefirst.json"""
import json, math
D = "<runs>/hook-validation/followup-a2/out/"
U = {}
for l in open(D + "units.jsonl"):
    u = json.loads(l); U[(u["tid"], u["rank"], u["layer"], u["pool"])] = u
for l in open(D + "cross_units.jsonl"):
    u = json.loads(l); U[(u["tid"], u["rank"], u["layer"], u["pool"])].update({k: v for k, v in u.items() if "~" in k})
allpos = {(k[0], k[3]) for k in U}
base = sum(p % 64 == 0 for t, p in allpos) / len(allpos)
out = {"window_positions": len(allpos), "page_first_fraction": base}
for p in ("b1~reb", "b1b~reb", "a1b1~a1reb", "b1~b1b", "reb~reb2"):
    pos = {(k[0], k[3]) for k, u in U.items() if u.get(p) is not None and u[p] > 0.25}
    obs = sum(pp % 64 == 0 for t, pp in pos)
    lam = base * len(pos)
    tail = 1 - sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(obs))
    out[p] = dict(gross_positions=len(pos), page_first=obs, expected=round(lam, 2), P_ge_obs=round(tail, 3),
                  which=sorted([[t, pp] for t, pp in pos if pp % 64 == 0]))
json.dump(out, open(D + "f3_pagefirst.json", "w"), indent=1)
print(json.dumps(out, indent=1))
