"""H5h: where the in-graph exposure records are missing (designated positions with no layer record). CPU only."""
import glob, json, os, re, collections
import numpy as np, torch
RUN = "/run"
CL = [json.loads(l) for l in open(os.path.join(RUN, "client.jsonl"))]
reqs = {r["tag"]: r for r in CL if r.get("kind") == "request"}
FW = [json.loads(l) for p in glob.glob(RUN + "/forcewin/fw_r*.jsonl") for l in open(p)]
steps = {}
for r in FW:
    if r.get("kind") == "steps" and r["rank"] == 0:
        for row in r["rows"]:
            steps[row[0]] = row
rows = []
by = collections.Counter(); tot = collections.Counter()
ls_seen = None
for p in sorted(glob.glob(RUN + "/forcewin/req_*_r0.pt")):
    a = torch.load(p, weights_only=False)
    des = a.get("desig") or []
    if not des:
        continue
    tag = a["tag"]; P = a["P"]; W = a["W"]
    x = a["exposure"].numpy()          # [ndesig, MAX_LAYERS+1, NX]
    ls_seen = a["layer_slots"]
    M = a["metrics"].numpy()
    C = {n: i for i, n in enumerate(a["cols"])}
    has = (x[:, :, 0] >= 0).any(axis=1)
    nlay = (x[:, :-1, 0] >= 0).sum(axis=1)
    ph = tag.split("-")[0] + ("-" + tag.split("-")[1] if tag.startswith(("V4", "V3ref")) else "")
    for k, o in enumerate(des):
        stp = int(M[o, C["step"]]) if o < M.shape[0] and np.isfinite(M[o, C["step"]]) else -1
        kind = float(M[o, C["kind"]]) if o < M.shape[0] else float("nan")
        nrows = float(M[o, C["n_rows"]]) if o < M.shape[0] else float("nan")
        row = float(M[o, C["row"]]) if o < M.shape[0] else float("nan")
        st = steps.get(stp)
        rows.append(dict(tag=tag, phase=ph, P=P, W=W, o=o, abspos=P + o - 1, has=bool(has[k]), nlay=int(nlay[k]),
                         xstep=int(x[k, :, 0].max()), mstep=stp, kind=kind, n_rows=nrows, row=row,
                         batch=(st[1:] if st else None)))
json.dump(rows, open("/d/out/h5h_exposure_rows.json", "w"))
print("layer_slots (rank0):", ls_seen)
R = rows
print("designated", len(R), "missing", sum(not r["has"] for r in R))
def summ(key):
    c = collections.defaultdict(lambda: [0, 0])
    for r in R:
        c[key(r)][0] += 1; c[key(r)][1] += (not r["has"])
    return {k: v for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}
print("by phase [n, missing]:", summ(lambda r: r["phase"]))
print("by o==0:", summ(lambda r: r["o"] == 0))
print("by kind (0 verify,1 single):", summ(lambda r: r["kind"]))
print("by n_rows:", summ(lambda r: r["n_rows"]))
print("by row-in-step j:", summ(lambda r: r["row"]))
print("by num_reqs in step:", summ(lambda r: r["batch"][0] if r["batch"] else None))
print("by padded tokens>0:", summ(lambda r: (r["batch"][4] > 0) if r["batch"] else None))
print("by prefilling co-scheduled>0:", summ(lambda r: (r["batch"][3] > 0) if r["batch"] else None))
print("by num_tokens in step:", summ(lambda r: r["batch"][1] if r["batch"] else None))
