#!/usr/bin/env python3
"""Diagnosis run 1, item 1 (stdlib only, no GPU): which planned rows have a non-positive token length,
what run_k.py would compute for them, and what the proposed kplan.targets clamp changes.

Reads the frozen plan.json and imports the frozen kselect/kplan.py read-only (nothing is written there).
Writes nothing except stdout.
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(KS, "kselect"))
import kplan  # noqa: E402  (frozen; imported, not modified)

plan = json.load(open(os.path.join(KS, "plan.json")))

print("== rows of a REQUEST (not padding) with token length L <= 0, per cell")
bad_cells = []
for c in plan["cells"]:
    e = plan["meta"]["engines"][c["engine"]]
    K = e["K"]
    hits = []
    for bi, batch in enumerate(c["batches"]):
        Ls = kplan.row_lengths(e, batch, c["pad_rows"])
        nreq_rows = len(batch) * (K + 1)
        for ri, L in enumerate(Ls[:nreq_rows]):
            if L <= 0:
                P = L // kplan.KPOOL                       # run_k.py:243 (torch // is floor division too)
                Pr = [kplan.local_pools(P, e["dcp"], r) for r in range(e["dcp"])]
                req = ri // (K + 1)
                hits.append(dict(batch=bi, row=ri, req=req, Llast=batch[req], L=L, P=P, P_local=Pr,
                                 guard_if_pj=bool(P), slice_len=(1 + P if P < 0 else P)))
    if hits:
        bad_cells.append(c["id"])
        print(f"  {c['id']:22s} K={K} dcp={e['dcp']}")
        for h in hits:
            print(f"     batch {h['batch']} row {h['row']:2d} (request {h['req']}, Llast={h['Llast']}): L={h['L']:2d} "
                  f"P=L//4={h['P']:2d} per-rank P={h['P_local']}  `if pj:` passes={h['guard_if_pj']}  "
                  f"s[j,:pj] has {h['slice_len']} elements (s has P_req={h['Llast'] // 4} columns)")
print(f"  cells affected: {len(bad_cells)}: {bad_cells}")

print()
print("== first failing evaluation in v070/tiled48/b512 (the ERROR cell)")
c = next(x for x in plan["cells"] if x["id"] == "v070/tiled48/b512")
e = plan["meta"]["engines"]["v070"]
b0 = c["batches"][0]
Ls = kplan.row_lengths(e, b0, c["pad_rows"])
print(f"  batch 0 request 0: Llast={b0[0]} -> rows L={Ls[:6]}  P={[L // 4 for L in Ls[:6]]}")
print(f"  run_k.py:263-266: j=0 has pj={Ls[0] // 4} (truthy) -> poison_rows += 1, then s[0, :-1] over a "
      f"({e['K'] + 1}, {b0[0] // 4}) score matrix is EMPTY -> max() raises.")
print("  summary-v070.json for this cell records desc.poison_rows == 1, i.e. exactly one increment before the raise.")

print()
print("== proposed fix A (kplan.targets): lowest target T >= cdiv(K+1, KPOOL) so every request row has L >= 1")


_orig_targets = kplan.targets


def targets_fixed(e, b, kind, ctas=None):
    out = _orig_targets(e, b, kind, ctas)
    lo = kplan.cdiv(e["K"] + 1, kplan.KPOOL)
    return [max(lo, t) for t in out]


kplan.targets = targets_fixed
try:
    cells_new = kplan.build()
finally:
    kplan.targets = _orig_targets
old = {c["id"]: c for c in plan["cells"]}
changed = []
for cn in cells_new:
    co = old[cn["id"]]
    if cn["batches"] != co["batches"]:
        changed.append(cn["id"])
        print(f"  {cn['id']:22s} batches old {co['batches']}")
        print(f"  {'':22s} batches new {cn['batches']}")
same_other = all(json.dumps(cn, sort_keys=True) == json.dumps(old[cn['id']], sort_keys=True)
                 for cn in cells_new if cn["id"] not in changed)
print(f"  cells changed: {len(changed)} ({changed}); every other cell byte-identical as JSON: {same_other}")
neg = 0
for cn in cells_new:
    e = plan["meta"]["engines"][cn["engine"]]
    for batch in cn["batches"]:
        Ls = kplan.row_lengths(e, batch, cn["pad_rows"])
        neg += sum(1 for L in Ls[:len(batch) * (e["K"] + 1)] if L <= 0)
print(f"  request rows with L <= 0 after fix A: {neg}")
