import json, collections
R = json.load(open("/d/out/h5h_exposure_rows.json"))
pres = [r for r in R if r["has"]]
c = collections.Counter((r["xstep"] == r["mstep"], int(r["row"]) if r["row"]==r["row"] else -1) for r in pres)
print("present by (xstep==metric step, measured row):", sorted(c.items()))
miss = [r for r in R if not r["has"]]
c2 = collections.Counter(int(r["row"]) if r["row"]==r["row"] else -1 for r in miss)
print("missing by measured row:", sorted(c2.items()))
