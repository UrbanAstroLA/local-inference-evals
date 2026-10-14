import json, collections
R = json.load(open("/d/out/h5h_exposure_rows.json"))
by = collections.defaultdict(list)
for r in R: by[r["tag"]].append(r)
for tag in ["V2a-poison-cbac17c70af8","V2a-oracle-cbac17c70af8","V2a-natural-cbac17c70af8","V2a-oracle-53fa5b3e066f","V4-b1-cbac17c70af8","V1-natural-free0-doc0"]:
    rs = by.get(tag, [])
    print(tag, "n", len(rs), "miss", sum(not r["has"] for r in rs))
    print("  first 40:", [(r["o"], int(r["has"]), r["nlay"], r["xstep"]-r["mstep"] if r["has"] else None, int(r["row"])) for r in rs[:40]])
# miss by o mod 4, by abspos mod 4
c = collections.defaultdict(lambda:[0,0])
for r in R:
    if r["o"]==0: continue
    k=(r["abspos"]%4)
    c[k][0]+=1; c[k][1]+= not r["has"]
print("by input-position mod 4:", dict(c))
c = collections.defaultdict(lambda:[0,0])
for r in R:
    if r["o"]==0: continue
    k=(r["abspos"]+1)//512
    c[k][0]+=1; c[k][1]+= not r["has"]
print("by context/512:", dict(sorted(c.items())))
# present: xstep vs mstep
d = collections.Counter(r["xstep"]-r["mstep"] for r in R if r["has"])
print("present: xstep - metric step:", sorted(d.items())[:20])
print("present nlay hist:", collections.Counter(r["nlay"] for r in R if r["has"]))
