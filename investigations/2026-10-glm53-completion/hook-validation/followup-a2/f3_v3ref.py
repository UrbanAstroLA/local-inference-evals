"""Follow-up a2, check 3: on the window units of each fixture that the attempt-1 multi-length prefill-vs-prefill null
(diagnosis-a1 out/h4c_nulls.json: pairs of V3ref builds of the same trace ending at different frontiers) also covers,
compare that null with the in-session single-length N_pp (rebuild vs rebuild2) and the decode arms.
Per unit: v3ref_max = max rel over V3ref pairs; v3ref_pairfrac = fraction of V3ref pairs with rel > 0.25.
-> out/f3_v3ref.json"""
import json, collections
HV = "<runs>/hook-validation/"
D = HV + "followup-a2/out/"
P = json.load(open(HV + "diagnosis-a1/out/h4c_nulls.json"))["P"]
pp = collections.defaultdict(list)
for x in P:
    pp[(x["g"].rsplit("-", 1)[-1], x["rank"], x["layer"], x["pool"])].append(x["rel"])
U = {}
for l in open(D + "units.jsonl"):
    u = json.loads(l); U[(u["tid"], u["rank"], u["layer"], u["pool"])] = u
for l in open(D + "cross_units.jsonl"):
    u = json.loads(l); U[(u["tid"], u["rank"], u["layer"], u["pool"])].update({k: v for k, v in u.items() if "~" in k})
out = {}
for t in sorted({k[0] for k in U}):
    J = [(k, u) for k, u in U.items() if k[0] == t and k in pp]
    if not J:
        out[t] = dict(joined=0)
        continue
    n = len(J)
    out[t] = dict(joined=n, pools=[min(k[3] for k, _ in J), max(k[3] for k, _ in J)],
                  v3ref_pairs_per_unit=round(sum(len(pp[k]) for k, _ in J) / n, 2),
                  v3ref_max_gt=sum(max(pp[k]) > 0.25 for k, _ in J),
                  v3ref_pairfrac=round(sum(sum(r > 0.25 for r in pp[k]) / len(pp[k]) for k, _ in J) / n, 5),
                  N_pp=sum(u["reb~reb2"] > 0.25 for _, u in J), b1=sum(u["b1~reb"] > 0.25 for _, u in J),
                  b1b=sum(u["b1b~reb"] > 0.25 for _, u in J), a1_b1=sum(u["a1b1~a1reb"] > 0.25 for _, u in J),
                  N_dd=sum(u["b1~b1b"] > 0.25 for _, u in J))
json.dump(out, open(D + "f3_v3ref.json", "w"), indent=1)
for t, v in out.items():
    print(t, v)
