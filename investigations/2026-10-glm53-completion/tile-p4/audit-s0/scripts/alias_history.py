#!/usr/bin/env python3
"""audit-s0 tasks 4/5: (a) does any digest point enumerate the same physical block id under two KV-cache groups
(the same bytes seen through two tensor views, e.g. a stale sliding-window row entry); (b) recorded history of the
focus physical blocks (which keyed requests held them, in which groups, before the warm that built the focus block).
CPU only; public-safe output (block ids, tags, step numbers, counts).  Writes /out/alias_history.out + .json.
"""
import collections
import glob
import json
import os

import numpy as np

AO = "/p4/runs-stage0/stage0-attempt-1"
PO = "/p4/PRIVATE-runs-stage0/stage0-attempt-1"


def jl(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        for line in open(p):
            if line.strip():
                out.append(json.loads(line))
    return out


logs0 = jl(os.path.join(AO, "p4log", "p4_r0_p*.jsonl"))
blocks = [r for r in logs0 if r.get("kind") == "blocks"]
reqs = {r["tag"]: r for r in logs0 if r.get("kind") == "req"}
L = []
res = dict(cross_group={}, draft_row_duplicates={}, history={})

# (a) cross-group duplicates per entry file and point
cross = collections.Counter()
examples = []
for f in sorted(glob.glob(os.path.join(PO, "p4", "entry_*_r*.npz"))):
    e = np.load(f)
    tag = os.path.basename(f)[6:-4]
    for pt in range(e["eblk"].shape[0]):
        owner = {}
        for g in range(e["eblk"].shape[1]):
            r = e["eblk"][pt, g]
            for j, x in enumerate(r.tolist()):
                if x <= 0:
                    continue
                if x in owner and owner[x][0] != g:
                    cross[(pt, owner[x][0], g)] += 1
                    if len(examples) < 12:
                        examples.append(dict(file=tag, point=pt, block=x, first=owner[x], second=(g, j)))
                owner.setdefault(x, (g, j))
res["cross_group"] = {f"point{k[0]}|g{k[1]}&g{k[2]}": v for k, v in sorted(cross.items())}
res["cross_group_examples"] = examples
L.append("(a) physical block ids enumerated under two groups at the same digest point (count of id repeats):")
L += [f"  {k}: {v}" for k, v in res["cross_group"].items()] or ["  none"]
L += [f"  e.g. {x}" for x in examples]

# duplicates / cross-group ids in the removal block records of group 9 (sliding-window draft)
dup = collections.Counter()
for r in blocks:
    g9 = [x for x in r["groups"][9] if x]
    other = {x for g, ids in enumerate(r["groups"]) if g != 9 for x in ids if x}
    dup[("draft_row_repeated_ids", r["tag"][:2])] += int(len(g9) != len(set(g9)))
    dup[("draft_row_ids_also_in_other_group", r["tag"][:2])] += int(bool(set(g9) & other))
    dup[("records", r["tag"][:2])] += 1
res["draft_row_duplicates"] = {"|".join(k): v for k, v in sorted(dup.items())}
L += ["", "removal records, group 9 (draft) row: records with repeated ids / ids also held by another group:"]
L += [f"  {k}: {v}" for k, v in res["draft_row_duplicates"].items()]

# (b) history of the focus physical blocks
order = sorted(reqs.values(), key=lambda r: r["uid"])
rem = {r["tag"]: r for r in blocks}
focus = {"V1-r1": 436, "V1-r2": 762, "V2-r1": 116, "V2-r2": 1199}
for rep, b in focus.items():
    warm = f"W-{rep}-R01"
    wuid = reqs[warm]["uid"]
    hist = []
    for r in order:
        t = r["tag"]
        if t not in rem:
            continue
        held = [g for g, ids in enumerate(rem[t]["groups"]) if b in ids]
        if held:
            hist.append(dict(tag=t, uid=r["uid"], add_step=r["step"], removal_step=rem[t]["step"], groups=held,
                             before_warm=r["uid"] < wuid))
    res["history"][f"{rep}:{b}"] = hist
    L += ["", f"(b) physical block {b} ({rep}; logical block 0 of R01, built by {warm} uid {wuid}):"]
    L += [f"  {h}" for h in hist]
    L.append("  keyed requests holding it before the warm: %d (unkeyed requests, startup warm-up and mid-life "
             "align/sliding-window frees are not recorded)" % sum(h["before_warm"] for h in hist))

json.dump(res, open("/out/alias_history.json", "w"), indent=1)
open("/out/alias_history.out", "w").write("\n".join(L) + "\n")
print("\n".join(L))
