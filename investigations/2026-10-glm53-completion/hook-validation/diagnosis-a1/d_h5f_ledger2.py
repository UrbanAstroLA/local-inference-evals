"""H5f, refined: skipped ledger steps predicted from the step log alone (padded steps: FULL-graph padding adds
dummy requests, so the builder's qsl has more entries than the batch's num_reqs, and _shadow_step's packed host
buffer is sliced at the wrong offsets). Prediction per pool: 'none' if it completes in a padded step; 'corrupt' if
it completes in an unpadded step but a member was stashed in a padded step; else 'clean'."""
import json, collections, glob, numpy as np, torch
R = [json.loads(l) for l in open("/d/out/h5_pools.jsonl")]
FW = [json.loads(l) for p in glob.glob("/run/forcewin/fw_r0*.jsonl") for l in open(p)]
st = {}
for r in FW:
    if r.get("kind") == "steps":
        for row in r["rows"]: st[row[0]] = row          # step, num_reqs, num_tokens, num_draft, n_prefilling, padded
pad_by_nr = collections.defaultdict(collections.Counter)
for s, row in st.items(): pad_by_nr[row[1]][row[5] > 0] += 1
print("steps by num_reqs: padded? ", {k: dict(v) for k, v in sorted(pad_by_nr.items())})
conf = collections.Counter()
for tag in sorted({r["tag"] for r in R}):
    a = torch.load(f"/run/forcewin/req_{tag}_r0.pt", weights_only=False)
    C = {n: i for i, n in enumerate(a["cols"])}; M = a["metrics"].numpy(); P = a["P"]; W = a["W"]; pc = a["pool_class"].numpy()
    step_of = {P + o - 1: int(M[o, C["step"]]) for o in range(1, W)}
    padded = lambda s: s in st and st[s][5] > 0
    for i in range((P + 3) // 4, (P + W - 1) // 4):
        k = i - P // 4
        if not (0 <= k < pc.size): continue
        cs = step_of.get(4 * i + 3)
        if cs is None: continue
        ms = [step_of.get(q) for q in range(4 * i, 4 * i + 3)]
        pred = "none" if padded(cs) else ("corrupt" if any(padded(s) for s in ms if s is not None and s != cs) else "clean")
        conf[(tag.split("-")[1], {0: "none", 1: "clean", 2: "corrupt", 3: "race", 4: "seed"}[int(pc[k])], pred)] += 1
for k, v in sorted(conf.items()): print("  ", k, v)
json.dump({"|".join(k): v for k, v in conf.items()}, open("/d/out/h5f_ledger_padded.json", "w"), indent=1)
