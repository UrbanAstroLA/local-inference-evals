import json, numpy as np, collections
R=[json.loads(l) for l in open("/d/out/h5_pools.jsonl")]
for arm in ("b1","d2solo","d2"):
    A=[r for r in R if r["arm"]==arm]
    g=[r for r in A if r["rel"]>0.25]
    print("==",arm,len(A),"gross",len(g), "frac %.4f"%(len(g)/len(A)))
    # position relative to window end
    c=collections.Counter()
    for r in g:
        endpos = r["P"]+r["W"]-1  # last committed position
        c[(r["last"]-r["frontier"])]+=1
    print(" gross by last-frontier:", sorted(c.items())[:20])
    c2=collections.Counter((r["last"]-r["frontier"]) for r in A)
    print(" all by last-frontier (top):", sorted(c2.items())[-6:])
    # pools fully before frontier
    inner=[r for r in A if r["last"]<=r["frontier"]]
    gi=[r for r in inner if r["rel"]>0.25]
    print(" inner (last<=frontier) n",len(inner),"gross",len(gi),"frac %.5f"%(len(gi)/max(1,len(inner))))
    rels=np.array([r["rel"] for r in inner])
    print(" inner q50 %.4f q99 %.4f q999 %.4f max %.4f"%tuple(np.quantile(rels,[.5,.99,.999,1])))
    # by layer, rank
    bl=collections.Counter((r["layer"]) for r in gi)
    print(" inner gross by layer", sorted(bl.items()))
    bt=collections.Counter((r["tid"]) for r in gi)
    print(" inner gross by tid", sorted(bt.items()))
    print(" zero x/y", sum(r["xzero"] for r in A), sum(r["yzero"] for r in A))
    for r in sorted(gi,key=lambda r:-r["rel"])[:12]:
        print("   ",r["tid"],r["rank"],r["layer"],r["pool"],"pos",r["first"],"off_from_P",r["first"]-r["P"],"to_front",r["frontier"]-r["last"],"rel %.3f cos %.3f xn %.3g yn %.3g xs %.3g ys %.3g beq %.2f"%(r["rel"],r["cos"],r["xn"],r["yn"],r["xs"],r["ys"],r["bytes_eq"]))
