#!/usr/bin/env python3
"""Zero-GPU tail analysis (operator request 2026-10-13, after the B1 seed-1236 pass; exploratory, not pre-registered).

Question: do engines differ in how often a long deliberation *finishes* before the 327,680-token cap, given that it got
long?  Inputs: recording-proxy request logs (completion_tokens, finish_reason, content_chars) of every logged K3.25 GPQA
pass, joined to doc ids by messages_sha256. Counts only; no reply text.
"""
import json, random, statistics as st
from collections import defaultdict
G = "<gpqa-runs>/"
TOK = "<runs>/d7/exposure/gpqa_prompt_tokens.json"
DOC = {d["messages_sha256"]: d["doc_id"] for d in json.load(open(TOK))["docs"]}
CAP = 327680
P = "glm53-flash-exl3-k3.25@tpurtell"
PASSES = [  # (engine, seed, request log)
    ("0.7.0", 1235, P + "+ps"), ("0.7.0", 1236, P + "+ps"),
    ("stock 0.9.1", 1234, P + "+v0.9.1"), ("stock 0.9.1", 1235, P + "+v0.9.1+ps"), ("stock 0.9.1", 1236, P + "+v0.9.1+ps"),
    ("col511 arm", 1234, P + "+v0.9.1+col511fix.s1"), ("col511 arm", 1235, P + "+v0.9.1+col511fix.s2"),
    ("col511 arm", 1236, P + "+v0.9.1+col511fix.s2"),
    ("B0", 1234, P + "+v0.9.1+bundle-d1234.s1"), ("B0", 1235, P + "+v0.9.1+bundle-d1234.s2"),
    ("B1", 1236, P + "+v0.9.1+bundle-d1234+d7.s1"),
]
def load(name, seed):
    out = {}
    for x in map(json.loads, open(G + name + ".requests.jsonl")):
        if x["seed"] == seed:
            out[DOC[x["messages_sha256"]]] = (x["completion_tokens"], x["finish_reason"] == "length")
    assert len(out) == 198, (name, seed, len(out))
    return out
data = {(e, s): load(n, s) for e, s, n in PASSES}
FAM = {"0.7.0": ["0.7.0"], "0.9.1 family (stock+arm)": ["stock 0.9.1", "col511 arm"], "corrected (B0+B1)": ["B0", "B1"],
       "stock 0.9.1": ["stock 0.9.1"]}
def rows(engines):
    return [(e, s, d, t, c) for (e, s), r in data.items() if e in engines for d, (t, c) in r.items()]
print(__doc__)
print("Per pass: replies reaching L tokens (finished at >= L or capped), and how many of those finished\n")
Ls = [32000, 64000, 100000, 200000]
print(f"{'pass':22s}" + "".join(f"  >= {L//1000:>3d}k reach/fin" for L in Ls) + "   capped")
for (e, s), r in data.items():
    line = f"{e+' '+str(s):22s}"
    for L in Ls:
        reach = [d for d, (t, c) in r.items() if t >= L]; fin = [d for d in reach if not r[d][1]]
        line += f"  {len(reach):>9d}/{len(fin):<4d}   "
    print(line + f"  {sum(c for t, c in r.values()):>4d}")
print("\nPooled by engine: P(finish | reached L), passes pooled; 95% CI by bootstrap over questions")
random.seed(20261013)
def pooled(engines, L, B=4000):
    R = [x for x in rows(engines) if x[3] >= L]
    byq = defaultdict(list)
    for e, s, d, t, c in R: byq[d].append(0 if c else 1)
    qs = list(byq); n = len(R); f = sum(sum(v) for v in byq.values())
    bs = []
    for _ in range(B):
        smp = [random.choice(qs) for _ in qs]; num = sum(sum(byq[q]) for q in smp); den = sum(len(byq[q]) for q in smp)
        bs.append(num / den if den else float("nan"))
    bs.sort()
    return n, f, len(qs), (f / n if n else float("nan")), bs[int(.025 * B)], bs[int(.975 * B)]
for L in Ls:
    print(f"  L = {L:,}")
    for name, eng in FAM.items():
        n, f, q, p, lo, hi = pooled(eng, L)
        print(f"    {name:26s} reached {n:3d} (questions {q:3d}), finished {f:3d}  P = {p:.2f}  [{lo:.2f}, {hi:.2f}]")
print("\nConclusion hazard: among replies still going at the start of each band, fraction that finish inside it")
bands = [(0, 16000), (16000, 32000), (32000, 64000), (64000, 100000), (100000, 150000), (150000, 200000),
         (200000, 260000), (260000, CAP)]
for name, eng in FAM.items():
    R = rows(eng); line = f"  {name:26s}"
    for lo, hi in bands:
        alive = [x for x in R if x[3] >= lo]; fin = [x for x in alive if not x[4] and x[3] < hi]
        line += f" {lo//1000:>3d}-{hi//1000:<3d}k {len(fin):>3d}/{len(alive):<4d}"
    print(line)
print("\nQuestion as the unit: questions that reached 100k in both a 0.7.0 pass and a 0.9.1-family pass;"
      " per question, finish fraction among its >=100k replies")
def qfrac(engines, L=100000):
    out = defaultdict(lambda: [0, 0])
    for e, s, d, t, c in rows(engines):
        if t >= L: out[d][0] += (0 if c else 1); out[d][1] += 1
    return out
a, b = qfrac(["0.7.0"]), qfrac(["stock 0.9.1", "col511 arm"])
common = sorted(set(a) & set(b)); diffs = []
for d in common:
    fa, fb = a[d][0] / a[d][1], b[d][0] / b[d][1]; diffs.append(fa - fb)
    print(f"    doc {d:3d}: 0.7.0 {a[d][0]}/{a[d][1]}  0.9.1-family {b[d][0]}/{b[d][1]}")
if diffs:
    pos = sum(x > 0 for x in diffs); neg = sum(x < 0 for x in diffs)
    print(f"  {len(common)} common questions: 0.7.0 higher on {pos}, lower on {neg}, equal on {len(diffs)-pos-neg};"
          f" mean difference {st.mean(diffs):+.2f}")
print("\nReach rate (does the engine get long more often?): replies reaching 100k per pass")
for name, eng in FAM.items():
    R = rows(eng); npass = len({(e, s) for e, s, *_ in R})
    print(f"  {name:26s} {sum(x[3] >= 100000 for x in R)/npass:.1f} per pass over {npass} passes")
