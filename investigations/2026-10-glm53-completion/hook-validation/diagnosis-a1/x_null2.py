import json, collections, numpy as np
R = json.load(open("/d/out/h4c_nulls.json"))["P"]
by = collections.defaultdict(list)
for x in R: by[x["g"]].append(x)
for g, v in sorted(by.items()):
    a = np.array([x["rel"] for x in v])
    print("%-32s n %6d bitwise %.3f q50 %.4f q999 %.3f gross %.4f max %.3f" % (g, len(v), np.mean([x["same"] for x in v]), np.median(a), np.quantile(a, .999), np.mean(a > .25), a.max()))
# gross pools: cos-like magnitude and position spread
g = [x for x in R if x["rel"] > 1.0]
print("rel>1.0:", len(g), collections.Counter((x["g"], x["layer"]) for x in g).most_common(8))
