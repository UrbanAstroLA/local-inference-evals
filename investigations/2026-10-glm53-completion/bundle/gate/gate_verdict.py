#!/usr/bin/env python3
"""Combine the bundle gate sessions into one verdict (PLAN.md amendment 14 (3)); rules fixed before data. CPU only.
usage: gate_verdict.py S1_DIR S3_DIR [S2_DIR]   -> prints the verdict, writes gate-verdict.json next to S1_DIR

PASS requires: every s1 check, every s3 check, every s2 check when s2 ran, and D4 real-weight reachability:
  - cross-session control available (s2 ran and its probe equals s1's on all 6 prompts): reachable iff the limit-1
    probe (s3) differs from s1 on at least 1 of 6 prompts;
  - no control (s2 not run, or s2 differs from s1): reachable iff s3 differs on >= 5 of 6 prompts AND >= 4 of them diverge
    within the first 16 generated tokens (a gross change that cross-session numerical noise does not produce, inferred).
  s3 identical to s1 on all 6 prompts: D4 NOT verified -> the bundle is not screened with D4 (a D4-less build needs a new
  freeze and operator decision).
"""
import json
import os
import sys

s1, s3 = sys.argv[1], sys.argv[2]
s2 = sys.argv[3] if len(sys.argv) > 3 else None


def load(d):
    v = json.load(open(os.path.join(d, "verdict.json")))
    pr = {}
    for line in open(os.path.join(d, "probe.jsonl")):
        x = json.loads(line)
        if x["round"] == 1:
            pr[x["id"]] = x
    return v, pr


def first_div(a, b):
    ta, tb = a["tokens"], b["tokens"]
    for i, (x, y) in enumerate(zip(ta, tb)):
        if x[0] != y[0]:
            return i
    return None if len(ta) == len(tb) else min(len(ta), len(tb))


out = {"sessions": {}, "reasons": []}
v1, p1 = load(s1)
v3, p3 = load(s3)
sess = [("s1", v1), ("s3", v3)]
control = None
if s2:
    v2, p2 = load(s2)
    sess.append(("s2", v2))
    control = all(p2.get(k, {}).get("sha256") == p1[k]["sha256"] for k in p1) and len(p1) == 6
for name, v in sess:
    fails = [k for k, c in v["checks"].items() if not c["ok"]]
    out["sessions"][name] = {"fails": fails, "bundle": v["bundle"]}
    if fails:
        out["reasons"].append(f"{name}: {fails}")
diff = {k: (p3.get(k, {}).get("sha256") != p1[k]["sha256"], first_div(p1[k], p3[k]) if k in p3 else None) for k in p1}
n_diff = sum(d for d, _ in diff.values())
n_early = sum(1 for d, i in diff.values() if d and i is not None and i < 16)
if control:
    reach = n_diff >= 1
    rule = "cross-session control passed (s2 == s1): reachable iff >= 1 of 6 prompts differ at limit 1"
else:
    reach = n_diff >= 5 and n_early >= 4
    rule = ("no cross-session control" if control is None else "cross-session control FAILED (s2 != s1)") + \
           ": reachable iff >= 5 of 6 differ and >= 4 diverge within 16 tokens"
out["d4_probe"] = {"per_prompt": diff, "n_diff": n_diff, "n_early": n_early, "control_s2_equals_s1": control, "rule": rule,
                   "reachable": reach}
if not reach:
    out["reasons"].append("D4 real-weight reachability not shown (limit-1 probe vs limit-10)")
out["verdict"] = "PASS" if not out["reasons"] else "FAIL"
print(json.dumps(out, indent=1, default=str))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(s1)), "gate-verdict.json"), "w"), indent=1, default=str)
