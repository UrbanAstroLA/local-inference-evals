#!/usr/bin/env python3
"""Read-only diagnostic of the REAL attempt-1 localisation files (CPU container; numpy): does the localisation
instrument pass its own gate (G-loc) on the served model, and -- descriptively, outside the gated reading, because
attempt 1's capture failed -- where do the four localisation warms first part?  Prints counts, boundary names and
positions only.   usage: diag_attempt1_loc.py AO PO"""
import itertools
import json
import os
import sys

sys.path.insert(0, "/c/analysis")
import importlib.util  # noqa: E402
s = importlib.util.spec_from_file_location("A", "/c/analysis/analyze_cap0.py")
A = importlib.util.module_from_spec(s)
s.loader.exec_module(A)
R = A.Run(sys.argv[1], sys.argv[2])
loc = [r for r in R.reps if R.loc.get(r)]
data, out = {}, dict(replicates=loc, gate={}, pairs={})
for rep in loc:
    ok, fail, d = A.gate_loc(R, rep)
    out["gate"][rep] = dict(ok=ok, failures=fail[:8])
    data[rep] = d
ids = {rep: json.dumps(d.get("ids", {}).get(0)) for rep, d in data.items()}
out["input_ids_equal_across_replicates"] = len(set(ids.values())) == 1
for a, b in itertools.combinations(loc, 2):
    f = A.loc_first(data[a], data[b], (0, 1))
    out["pairs"][f"{a}-{b}"] = None if f is None else dict(point=f["point"], chunk=f["chunk"], step_pos0=f["step_pos0"],
                                                          hits=[dict(rank=h["rank"], tensor=h["tensor"],
                                                                     n_tokens=h.get("n_tokens"), first_pos=h.get("first_pos"))
                                                                for h in f["hits"]])
print(json.dumps(out, indent=1))


# ---- detail: every boundary per chunk for the pairs with the reference (r1), and top-k rows compared as SETS
import numpy as np  # noqa: E402
det = {}
ref = loc[0]
for rep in loc[1:]:
    rows = {}
    for s0, n in [c for c in A.CHUNKS if c[0] < A.LY.N_CACHED]:
        for point in A.COMP_ORDER:
            if point.endswith("state_end"):
                continue
            for t in A.LOC_TENSORS[point]:
                for rank in (0, 1):
                    k = f"dig:{point}.{t}"
                    aa, ab = data[ref].get(rank, ({}, {}))[0], data[rep].get(rank, ({}, {}))[0]
                    if k not in aa or k not in ab:
                        continue
                    neq = np.nonzero(np.any(aa[k][s0:s0 + n] != ab[k][s0:s0 + n], axis=1))[0]
                    if neq.size:
                        rows.setdefault(f"chunk{A.LY.chunk_of(s0, A.CHUNKS)}", []).append(
                            f"{point}.{t} r{rank}: {neq.size} rows from {s0 + int(neq[0])}")
    arr, meta = data[rep][0]
    sets = None
    if "diffcur:L3.topk.ids" in arr and "diffref:L3.topk.ids" in arr:
        cur = arr["diffcur:L3.topk.ids"].view(np.int32)
        rf = arr["diffref:L3.topk.ids"].view(np.int32)
        sets = []
        for i, p in enumerate(arr["diffpos:L3.topk.ids"]):
            a, b = set(cur[i][cur[i] >= 0].tolist()), set(rf[i][rf[i] >= 0].tolist())
            sets.append(dict(pos=int(p), same_set=a == b, only_cur=len(a - b), only_ref=len(b - a),
                             valid_ids=len(a), pools_only_cur=sorted({x // 4 for x in a - b})[:6],
                             pools_only_ref=sorted({x // 4 for x in b - a})[:6]))
    det[f"{ref}-{rep}"] = dict(differing=rows, topk_first_rows=sets)
print(json.dumps(det, indent=1))
