import glob, os, json, numpy as np, torch, collections
CL=[json.loads(l) for l in open("/run/client.jsonl")]
reqs=[r for r in CL if r.get("kind")=="request" and r.get("W",0)>0 and not r["tag"].startswith("V3ref")]
FW=[json.loads(l) for p in glob.glob("/run/forcewin/fw_r0*.jsonl") for l in open(p)]
steps={}
for r in FW:
    if r.get("kind")=="steps":
        for row in r["rows"]: steps[row[0]]=row
c=collections.Counter()
for r in reqs:
    p=f"/run/forcewin/req_{r['tag'].replace(':','_')}_r0.pt"
    if not os.path.exists(p): continue
    a=torch.load(p,weights_only=False); C={n:i for i,n in enumerate(a["cols"])}
    M=a["metrics"][:r["W"]].numpy().astype(np.float64)
    for i in range(M.shape[0]):
        st=steps.get(int(M[i,C["step"]])); ntok=st[2] if st else -1
        trit = ntok>=8
        top1=M[i,C["top1_p"]]
        deg = top1==1.0
        # any tracked non-argmax id with p_pre>0 : masked or not
        pre=np.exp(M[i,[C["lp_pre_%d"%k] for k in range(4)]]); post=M[i,[C["p_post_%d"%k] for k in range(4)]]
        am=M[i,C["argmax"]]
        sel=pre>0
        if not sel.any(): c[(trit,deg,"no_tail_repr")]+=1; continue
        unm=(post[sel]>0).any(); msk=(post[sel]==0).all()
        c[(trit, deg, "unmasked" if unm else "masked")]+=1
for k,v in sorted(c.items(), key=str): print(k,v)
