#!/usr/bin/env python3
"""Recompute every number in this investigation's README from the published rows (standard library only).

    python3 investigations/2026-10-glm53-completion/recompute.py

Reads runs/*/results.jsonl through tools/analyze.py. Phase 0 publishes no new rows: it re-reads the two three-pass 3.25bpw
GPQA records (tpurtell 0.7.0, DFlash2 x5; tpurtell 0.9.1, DFlash2 x3), and the 4bpw TR3 (Brandon) record on 0.9.1 for reference.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import analyze  # noqa: E402

HERE = Path(__file__).resolve().parent
OLD, NEW, K4 = "k3.25-v0.7.0-dflash5", "k3.25-v0.9.1-dflash3", "k4-v0.9.1-dflash3-c4"
CAP = 327680
g = analyze.gpqa()
fails = []


def check(name, ok):
    print(f"  check {name}: {'ok' if ok else 'FAILED'}")
    if not ok: fails.append(name)


print("# GLM-5.3-Flash completion investigation: Phase 0 (published rows only)\n")
man = json.loads((HERE / "manifest.json").read_text())
print("## Inputs pinned in manifest.json")
for r in man["records"]:
    m = g[r["config"].split("/", 1)[1]]
    check(f"{r['config']} is run {r['run_id']}", m["id"] == r["run_id"])
    check(f"{r['run_id']} results.jsonl sha256", __import__("hashlib").sha256((ROOT / "runs" / r["run_id"] / "results.jsonl").read_bytes()).hexdigest() == r["results_sha256"])

print("\n## Decomposition, 0.7.0 vs 0.9.1 (3.25bpw), joined on (question, pass)")
analyze.gpqa_join([OLD, NEW])
j = analyze.join(g[OLD], g[NEW])
check("594 pairs, none unpaired", j["n"] == 594 and j["unpaired"] == 0)
check("no seed, prompt, target or question hash mismatch", not any(j["mismatches"].values()))
check("raw 524 vs 516; empty 4 vs 16", (j["a_correct_flexible_all"], j["b_correct_flexible_all"], j["a_empty_all"], j["b_empty_all"]) == (524, 516, 4, 16))
check("577 both answered: raw 516 vs 516, stated 523 vs 522",
      (len(j["both"]), j["a_correct_flexible_both"], j["b_correct_flexible_both"], j["a_correct_stated_both"], j["b_correct_stated_both"]) == (577, 516, 516, 523, 522))
check("8 score-driving pairs, none the other way",
      j["a_right_b_empty"] == [(12, 1), (55, 2), (81, 3), (88, 1), (88, 3), (109, 1), (120, 1), (170, 3)] and not j["b_right_a_empty"])

print("\n## For reference: the same split for the other two record comparisons")
for a, b in ((NEW, K4), (OLD, K4)):
    x = analyze.join(g[a], g[b])
    print(f"{a} vs {b}: both answered {len(x['both'])}, raw {x['a_correct_flexible_both']} vs {x['b_correct_flexible_both']}; "
          f"A right / B empty {len(x['a_right_b_empty'])}, B right / A empty {len(x['b_right_a_empty'])}")

print("\n## How the empty answers ended (finish_reason and completion_tokens from the passive request log)")
print("| record | pass | empty | at the 327,680-token cap | stopped (finish_reason stop) | no request log |")
print("|---|---|---|---|---|---|")
for c in (OLD, NEW, K4):
    for p in (1, 2, 3):
        rs = [r for r in g[c]["rows"] if r["pass"] == p and r["empty"]]
        cap = [r for r in rs if r["finish_reason"] == "length" and r["completion_tokens"] == CAP]
        stop = [r for r in rs if r["finish_reason"] == "stop"]
        nolog = [r for r in g[c]["rows"] if r["pass"] == p and r["finish_reason"] is None]
        print(f"| {c} | {p} | {len(rs)} | {len(cap)} | {len(stop)}" + (f" (q{', q'.join(str(r['doc_id']) for r in stop)}, "
              f"{', '.join(str(r['completion_tokens']) for r in stop)} tokens)" if stop else "") + f" | {'yes' if nolog else 'no'} |")
e = [r for c in (OLD, NEW, K4) for r in g[c]["rows"] if r["empty"]]
check("every logged empty answer is at the cap or the one 38-token stop",
      all(r["finish_reason"] == "length" and r["completion_tokens"] == CAP or (r["finish_reason"] == "stop" and r["completion_tokens"] == 38)
          for r in e if r["finish_reason"]) and sum(r["finish_reason"] == "stop" for r in e) == 1)
check("0.7.0 pass 1 (no request log) has no empty answer", not any(r["empty"] for r in g[OLD]["rows"] if r["pass"] == 1))
print("\n## Phase 0 checks (checks.jsonl)")
for line in (HERE / "checks.jsonl").read_text().splitlines():
    c = json.loads(line); print(f"  {c['id']}: {c['result']}")
print("\nOK" if not fails else f"\nFAILED: {fails}")
sys.exit(1 if fails else 0)
