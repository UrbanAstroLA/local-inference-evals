"""H5b: do gross B1 window pools (decode vs prefill rebuild) cluster at speculative/step/ring/block structure, and do
they coincide with decode-vs-prefill divergence of the model output at the same positions? CPU only.
For each V4-b1 pool, members q = 4i..4i+3 are the inputs of the rows predicting o = q + 1 - P. From the hook's
metric rows: the step / row j / accepted count of each member; from the prefill reference (V2c or V2b-ref) the
reference row at the same o, giving |dlp(forced)|, argmax agreement and |d entropy| at the member positions."""
import json, collections
import numpy as np, torch
R = [json.loads(l) for l in open("/d/out/h5_pools.jsonl")]
REFMAP = {"cbac17c70af8": "V2c", "be2e9e349395": "V2c", "4b92a6a90ec4": "V2c", "ddfd287416a4": "V2c",
          "68a96f264430": "V2b-ref", "376073921446": "V2b-ref", "1e875e978fd4": "V2b-ref", "6d18c40f58df": "V2b-ref"}
cache = {}
def load(tag):
    if tag not in cache: cache[tag] = torch.load(f"/run/forcewin/req_{tag}_r0.pt", weights_only=False)
    return cache[tag]
rows = []
for arm in ("b1", "d2solo"):
    for r in R:
        if r["arm"] != arm: continue
        a = load(r["tag"]); ref = load(f"{REFMAP[r['tid']]}-{r['tid']}")
        C = {n: i for i, n in enumerate(a["cols"])}
        M = a["metrics"].numpy().astype(float); Rf = ref["metrics"].numpy().astype(float)
        P = r["P"]; os_ = [q + 1 - P for q in range(r["first"], r["last"] + 1)]
        os_ = [o for o in os_ if 0 <= o < r["W"]]
        if not os_: continue
        st = [int(M[o, C["step"]]) for o in os_]; jj = [int(M[o, C["row"]]) for o in os_]; ac = [int(M[o, C["accepted"]]) for o in os_]
        dl = [abs(M[o, C["lp_pre_4"]] - Rf[o, C["lp_pre_4"]]) for o in os_]
        agree = [M[o, C["argmax"]] == Rf[o, C["argmax"]] for o in os_]
        dent = [abs(M[o, C["entropy"]] - Rf[o, C["entropy"]]) for o in os_]
        rows.append(dict(arm=arm, tid=r["tid"], layer=r["layer"], pool=r["pool"], gross=r["rel"] > .25, rel=r["rel"],
                         one_step=len(set(st)) == 1, any_draft_row=any(j > 0 for j in jj), last_step_acc=ac[-1],
                         completion_row=jj[-1], ring_half=(r["first"] % 8) // 4, block_edge=(r["first"] % 4352) < 8 or (r["first"] % 4352) > 4344,
                         ctx_ge_2044=r["last"] >= 2044, dlp_max=float(np.nanmax(dl)), disagree=not all(agree),
                         dent_max=float(np.nanmax(dent)), layer_deep=r["layer"] >= 27))
json.dump(rows, open("/d/out/h5_context.json", "w"))
def tab(arm, key, name):
    c = collections.defaultdict(lambda: [0, 0])
    for x in rows:
        if x["arm"] != arm: continue
        c[x[key]][0] += 1; c[x[key]][1] += x["gross"]
    print("  %-16s" % name, {k: "%d/%d=%.4f" % (v[1], v[0], v[1] / v[0]) for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))})
for arm in ("b1", "d2solo"):
    print("==", arm, "pools", sum(x["arm"] == arm for x in rows))
    tab(arm, "one_step", "members in one step")
    tab(arm, "any_draft_row", "member as draft row")
    tab(arm, "completion_row", "completing row j")
    tab(arm, "last_step_acc", "accepted at completion step")
    tab(arm, "ring_half", "ring half")
    tab(arm, "ctx_ge_2044", "context >= 2044")
    tab(arm, "layer_deep", "layer >= 27")
    G = [x for x in rows if x["arm"] == arm and x["gross"]]; N = [x for x in rows if x["arm"] == arm and not x["gross"]]
    for nm, S in (("gross", G), ("non-gross", N)):
        print("  %-9s n %5d  output at member positions: max|dlp| q50 %.3g mean %.3g | argmax disagreement %.3f | max|dH| q50 %.3g" % (
            nm, len(S), np.median([x["dlp_max"] for x in S]), np.mean([x["dlp_max"] for x in S]), np.mean([x["disagree"] for x in S]),
            np.median([x["dent_max"] for x in S])))
    # deep layers only, to remove the depth confound
    G = [x for x in G if x["layer_deep"]]; N = [x for x in N if x["layer_deep"]]
    for nm, S in (("gross L>=27", G), ("non-gross L>=27", N)):
        print("  %-15s n %5d  max|dlp| q50 %.3g mean %.3g | argmax disagreement %.3f | max|dH| q50 %.3g" % (
            nm, len(S), np.median([x["dlp_max"] for x in S]), np.mean([x["dlp_max"] for x in S]), np.mean([x["disagree"] for x in S]),
            np.median([x["dent_max"] for x in S])))
