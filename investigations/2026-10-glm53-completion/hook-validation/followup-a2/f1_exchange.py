"""Follow-up a2, check 1: exchange (label-swap) null for 'both B1 arms have equal pooled gross counts', against the
rebuild and (jointly) against the second rebuild. Under the null that b1 and b1b are exchangeable decode replicates,
swapping the two arms' data on any unit leaves the distribution unchanged; each unit's swap is an independent fair coin
(the same coin for the rebuild and rebuild2 comparisons, since it swaps the unit's data). Seeded. -> out/f1_exchange.json"""
import json, random
G = 0.25
D = "<runs>/hook-validation/followup-a2/out/"
U = [json.loads(l) for l in open(D + "units.jsonl")]
c = []
for u in U:
    d1 = int(u["b1~reb"] > G) - int(u["b1b~reb"] > G)
    d2 = int(u["b1~reb2"] > G) - int(u["b1b~reb2"] > G)
    if d1 or d2:
        c.append((d1, d2))
obs = (sum(x[0] for x in c), sum(x[1] for x in c))
# cluster form: one swap per position (fixture, pool), since one position event spans several layers / both ranks
cl = {}
for u in U:
    d1 = int(u["b1~reb"] > G) - int(u["b1b~reb"] > G)
    d2 = int(u["b1~reb2"] > G) - int(u["b1b~reb2"] > G)
    if d1 or d2:
        a = cl.setdefault((u["tid"], u["pool"]), [0, 0]); a[0] += d1; a[1] += d2
cc = [tuple(v) for v in cl.values() if v != [0, 0]]
rng = random.Random(20261014)
N = 200000
e1 = e2 = e12 = 0
for _ in range(N):
    s1 = s2 = 0
    for d1, d2 in c:
        if rng.random() < 0.5:
            s1 += d1; s2 += d2
        else:
            s1 -= d1; s2 -= d2
    e1 += s1 == 0; e2 += s2 == 0; e12 += (s1 == 0 and s2 == 0)
rng = random.Random(20261015)
f1 = f2 = f12 = 0
for _ in range(N):
    s1 = s2 = 0
    for d1, d2 in cc:
        if rng.random() < 0.5:
            s1 += d1; s2 += d2
        else:
            s1 -= d1; s2 -= d2
    f1 += s1 == 0; f2 += s2 == 0; f12 += (s1 == 0 and s2 == 0)
out = dict(units_contributing=len(c),
           cluster_form=dict(positions_contributing=len(cc), seed=20261015, P_equal_vs_reb=f1 / N,
                             P_equal_vs_reb2=f2 / N, P_equal_both=f12 / N), observed_diff=dict(reb=obs[0], reb2=obs[1]), sims=N, seed=20261014,
           P_equal_vs_reb=e1 / N, P_equal_vs_reb2=e2 / N, P_equal_both=e12 / N)
json.dump(out, open(D + "f1_exchange.json", "w"), indent=1)
print(out)
