"""H1f: rows where the served top_p mask lies outside the pre-mask eligibility bounds; size of the violation. CPU."""
import glob, json, os, re, collections
import numpy as np, torch
RUN = "/run"
CL = [json.loads(l) for l in open(os.path.join(RUN, "client.jsonl"))]
reqs = [r for r in CL if r.get("kind") == "request" and r.get("W", 0) > 0 and not r["tag"].startswith("V3ref")]
FW = [json.loads(l) for p in glob.glob(RUN + "/forcewin/fw_r0*.jsonl") for l in open(p)]
steps = {}
for r in FW:
    if r.get("kind") == "steps":
        for row in r["rows"]: steps[row[0]] = row
out = []; tot_rows = 0; tot_cells = 0
NAMES = ["</think>", "eot", "user", "obs", "forced"]
for r in reqs:
    p = f"/run/forcewin/req_{r['tag'].replace(':', '_')}_r0.pt"
    if not os.path.exists(p): continue
    a = torch.load(p, weights_only=False)
    C = {n: i for i, n in enumerate(a["cols"])}
    M = a["metrics"][: r["W"]].numpy().astype(np.float64)
    tot_rows += M.shape[0]
    for k in range(5):
        pp, lo, hi = M[:, C[f"p_post_{k}"]], M[:, C[f"elig_lo_{k}"]], M[:, C[f"elig_hi_{k}"]]
        mlt, mle, lp = M[:, C[f"mass_lt_{k}"]], M[:, C[f"mass_le_{k}"]], M[:, C[f"lp_pre_{k}"]]
        tot_cells += np.isfinite(lp).sum()
        bad1 = (pp > 0) & (hi < 0.5)          # kept although not possibly kept
        bad2 = (lo > 0.5) & (pp <= 0)         # masked although certainly kept
        for i in np.where(bad1 | bad2)[0]:
            st = steps.get(int(M[i, C["step"]]))
            pk = float(np.exp(lp[i]))
            out.append(dict(tag=r["tag"], o=int(i), id=NAMES[k], kind="kept_not_eligible" if bad1[i] else "masked_but_eligible",
                            p_pre=pk, lp_pre=float(lp[i]), mass_lt=float(mlt[i]), mass_le=float(mle[i]), p_post=float(pp[i]),
                            # distance of the cumulative mass through this token from the cut 1 - top_p = 0.05
                            margin_hi=float(mle[i] - 0.05), margin_lo=float(mlt[i] + pk - 0.05),
                            top1_p=float(M[i, C["top1_p"]]), row=int(M[i, C["row"]]), n_rows=int(M[i, C["n_rows"]]),
                            kind_row=float(M[i, C["kind"]]), step=int(M[i, C["step"]]),
                            batch_num_reqs=st[1] if st else None, batch_tokens=st[2] if st else None))
json.dump(out, open("/d/out/h1f_rows.json", "w"), indent=0)
print("rows checked", tot_rows, "tracked cells", tot_cells, "violations", len(out))
for v in out:
    print(json.dumps({k: (round(x, 7) if isinstance(x, float) else x) for k, x in v.items()}))
