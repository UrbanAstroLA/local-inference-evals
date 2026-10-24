#!/usr/bin/env python3
"""Mock root for runner tests: small fabricated fixtures (16 replies, 6 batches x 8 windows, P of a few hundred
tokens; no benchmark data), sched, and optionally passed precheck / G4 verdicts.  Host python3.
usage: make_mock_root.py ROOT [--checks-passed] [--ledger-gh X]"""
import json
import os
import sys

root = sys.argv[1]
os.makedirs(os.path.join(root, "PRIVATE-fixtures"), exist_ok=True)
os.makedirs(os.path.join(root, "runs"), exist_ok=True)
W = 512
replies = {}
order = []
for k in range(16):
    rid = f"R{k + 1:02d}"
    order.append(rid)
    ids = [1000 + ((k * 131 + i * 7) % 5000) for i in range(3000)]
    replies[rid] = dict(cls="finished" if k % 2 == 0 else "capped", ids=ids,
                        windows={"early": dict(o_start=100, P=200), "deep": dict(o_start=800, P=900),
                                 "close": dict(o_start=2300, P=2400)})
batches = []
left = []
for b in range(5):
    r0, r1, r2 = order[3 * b: 3 * b + 3]
    batches.append([(r0, "early"), (r0, "deep"), (r0, "close"), (r1, "early"), (r1, "deep"), (r1, "close"),
                    (r2, "deep"), (r2, "close")])
    left.append((r2, "early"))
batches.append([(order[15], "early"), (order[15], "deep"), (order[15], "close")] + left)
plan = []
for b, items in enumerate(batches, 1):
    warm = {}
    for rid, kind in items:
        warm[rid] = max(warm.get(rid, 0), replies[rid]["windows"][kind]["P"])
    plan.append(dict(batch=b, windows=[f"{r}-{k}" for r, k in items], warm=[dict(reply=r, P=p) for r, p in warm.items()],
                     warm_tokens=sum(warm.values()), remainder_tokens=[100] * 8))
json.dump(dict(W=W, replies=replies, plan=plan), open(os.path.join(root, "PRIVATE-fixtures", "windows.json"), "w"))
json.dump(dict(sched=[3, 2, 0, 1]), open(os.path.join(root, "PRIVATE-fixtures", "sched.json"), "w"))
if "--checks-passed" in sys.argv:
    for step, v in (("precheck", dict(verdict="PASS")), ("g4", dict(G4="PASS"))):
        d = os.path.join(root, "runs", f"{step}-attempt-1")
        os.makedirs(d, exist_ok=True)
        json.dump(v, open(os.path.join(d, "verdict.json"), "w"))
    with open(os.path.join(root, "costs.jsonl"), "a") as f:
        f.write(json.dumps(dict(item="p4-precheck", attempt=1, seconds=40, gpus=1)) + "\n")
        f.write(json.dumps(dict(item="p4-g4", attempt=1, seconds=160, gpus=1)) + "\n")
if "--ledger-gh" in sys.argv:
    gh = float(sys.argv[sys.argv.index("--ledger-gh") + 1])
    with open(os.path.join(root, "costs.jsonl"), "a") as f:
        f.write(json.dumps(dict(item="other", attempt=1, seconds=int(gh * 3600 / 2), gpus=2)) + "\n")
print("mock root", root)
