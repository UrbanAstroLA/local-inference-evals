"""Follow-up a2, check 3 (post-inspection, descriptive): the fixture flags count layer-units, but one position event
spans several deep layers. Gross POSITIONS (fixture, pool; any layer) per fixture for each comparison, and the
units-per-position ratio. -> out/f3_positions.json"""
import json, collections
D = "<runs>/hook-validation/followup-a2/out/"
U = {}
for l in open(D + "units.jsonl"):
    u = json.loads(l); U[(u["tid"], u["rank"], u["layer"], u["pool"])] = u
for l in open(D + "cross_units.jsonl"):
    u = json.loads(l); U[(u["tid"], u["rank"], u["layer"], u["pool"])].update({k: v for k, v in u.items() if "~" in k})
C = {"b1": "b1~reb", "b1b": "b1b~reb", "N_dd": "b1~b1b", "N_pp": "reb~reb2", "a1_b1": "a1b1~a1reb"}
out = collections.defaultdict(dict)
npos = collections.Counter(k[0] for k in {(k[0], k[3]) for k in U})
for nm, p in C.items():
    for t in sorted(npos):
        units = [k for k, u in U.items() if k[0] == t and u[p] > 0.25]
        pos = {k[3] for k in units}
        out[t][nm] = dict(units=len(units), positions=len(pos))
for t in out:
    out[t]["window_positions"] = npos[t]
json.dump(out, open(D + "f3_positions.json", "w"), indent=1)
for t, v in out.items():
    print(t, v["window_positions"], {k: f"{x['units']}u/{x['positions']}p" for k, x in v.items() if k != "window_positions"})
