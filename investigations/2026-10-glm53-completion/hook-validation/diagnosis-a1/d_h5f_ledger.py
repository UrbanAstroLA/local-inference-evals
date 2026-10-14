"""H5f: why does the shadow ledger leave pools unclassified ('none') and call byte-correct B1 pools 'corrupt'?
Hypothesis S: the ledger skipped some steps (no shadow update), so pools completed in those steps stay 'none' and
pools whose earlier members were stashed in a skipped step see a stale owner and are called 'corrupt'.
Test: infer skipped steps per request as the completion steps of 'none' pools; then a pool is predicted 'corrupt'
iff it completes in a recorded step and a member (q < completing position) was processed in a skipped step."""
import json, collections, glob, numpy as np, torch
R = [json.loads(l) for l in open("/d/out/h5_pools.jsonl")]
FW = [json.loads(l) for p in glob.glob("/run/forcewin/fw_r0*.jsonl") for l in open(p)]
stepinfo = {}
for r in FW:
    if r.get("kind") == "steps":
        for row in r["rows"]: stepinfo[row[0]] = row
out = collections.Counter(); per_req = {}
for tag in sorted({r["tag"] for r in R if r["arm"] in ("b1", "d2", "d2solo")}):
    a = torch.load(f"/run/forcewin/req_{tag}_r0.pt", weights_only=False)
    C = {n: i for i, n in enumerate(a["cols"])}; M = a["metrics"].numpy(); P = a["P"]; W = a["W"]; pc = a["pool_class"].numpy()
    step_of = {}
    for o in range(W):
        q = P + o - 1                      # input position of the row predicting o
        if q >= P: step_of[q] = int(M[o, C["step"]])
    pools = range((P + 3) // 4, (P + W - 1) // 4)
    cls = {i: int(pc[i - P // 4]) for i in pools if 0 <= i - P // 4 < pc.size}
    comp_step = {i: step_of.get(4 * i + 3) for i in pools}
    skipped = {comp_step[i] for i in pools if cls.get(i) == 0 and comp_step[i] is not None}
    recorded = {comp_step[i] for i in pools if cls.get(i) in (1, 2, 3) and comp_step[i] is not None}
    both = skipped & recorded
    for i in pools:
        if comp_step[i] is None or cls.get(i) is None: continue
        mem_steps = [step_of.get(q) for q in range(4 * i, 4 * i + 3)]
        pred = "none" if comp_step[i] in skipped else ("corrupt" if any(s in skipped for s in mem_steps if s is not None) else "clean")
        out[(tag.split("-")[1], {0: "none", 1: "clean", 2: "corrupt", 3: "race", 4: "seed"}[cls[i]], pred)] += 1
    nsk = len(skipped); ntot = len(set(comp_step.values()) - {None})
    per_req[tag] = dict(steps_with_completion=ntot, inferred_skipped=nsk, conflicting=len(both),
                        skipped_steps_num_reqs=collections.Counter(stepinfo[s][1] for s in skipped if s in stepinfo).most_common(4),
                        skipped_step_parity=collections.Counter(s % 2 for s in skipped), recorded_parity=collections.Counter(s % 2 for s in recorded))
print("(arm, ledger class, predicted by hypothesis S): count")
for k, v in sorted(out.items()): print("  ", k, v)
for t, v in per_req.items(): print(t, v)
json.dump({"confusion": {"|".join(k): v for k, v in out.items()}, "per_request": {t: {k: str(x) for k, x in v.items()} for t, v in per_req.items()}},
          open("/d/out/h5f_ledger_skips.json", "w"), indent=1)
