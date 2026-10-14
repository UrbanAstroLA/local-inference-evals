import json, collections, numpy as np
I = json.load(open("/d/out/h4c_idx.json"))
for i in I: i["mode"] = i["tag"].split("-")[1]
by = collections.defaultdict(list)
for i in I: by[i["mode"]].append(i)
for m, v in sorted(by.items()):
    a = np.array([x["rel"] for x in v]); g = [x for x in v if x["rel"] > .25]
    print("%-10s n %4d gross %3d (%.4f) q50 %.4f | gross rel: %s | cos: %s" % (m, len(v), len(g), len(g)/len(v), np.median(a),
          sorted(round(x["rel"], 2) for x in g)[-6:], sorted(round(x["cos"], 2) for x in g)[:6]))
# same pool across snapshots: is it stable (the pool is never rewritten after commit)?
key = collections.defaultdict(list)
for i in I: key[(i["tag"], i["rank"], i["layer"], i["pool"])].append((i["F"], round(i["rel"], 4)))
multi = [(k, v) for k, v in key.items() if len(v) > 1 and max(r for _, r in v) > .25]
print("pools seen in several snapshots and gross in one:", len(multi))
for k, v in multi[:10]: print("  ", k, sorted(v))
