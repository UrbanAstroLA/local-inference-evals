#!/usr/bin/env python3
"""Recompute every number in this investigation's README from the published rows (standard library only).

    python3 investigations/2026-10-glm53-completion/recompute.py

Reads runs/*/results.jsonl through tools/analyze.py. Phase 0 publishes no new rows: it re-reads the two three-pass 3.25bpw
GPQA records (tpurtell 0.7.0, DFlash2 x5; tpurtell 0.9.1, DFlash2 x3), and the 4bpw TR3 (Brandon) record on 0.9.1 for reference.
Phase 1 (steps 1-4, decode vs prefill) reads the decode-prefill runs named in manifest.json "phase1", applies each step's
rule (PLAN.md amendments 1-4) and the costs in costs.jsonl.
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


print("# GLM-5.3-Flash completion investigation (published rows only)\n\n# Phase 0: existing GPQA records\n")
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

# ---------------------------------------------------------------- Phase 1: decode vs prefill (PLAN.md amendments 1-4)
print("\n# Phase 1: decode vs prefill (published rows of the runs in manifest.json 'phase1')\n")
P1 = man["phase1"]
BOUNDS = {"below 2,044": (0, 2043), "from 2,048": (2048, 10 ** 9), "0-2,043": (0, 2043), "2,048-7,999": (2048, 7999),
          "8,000-15,999": (8000, 15999), "16,000 and above": (16000, 10 ** 9)}
_rows = {}


def rows(rid):
    if rid not in _rows:
        _rows[rid] = [json.loads(l) for l in (ROOT / "runs" / rid / "results.jsonl").read_text().splitlines()]
    return _rows[rid]


def kl(rid, region, skip_docs=()):
    lo, hi = BOUNDS[region]
    v = [r["kl_top20"] for r in rows(rid) if lo <= r["i"] <= hi and r["kl_top20"] is not None and r["doc_id"] not in skip_docs]
    return sum(v) / len(v)


def rule(a, b):
    """Amendments 1-4: a difference counts only if every value of one group lies outside the range of the other."""
    if max(a) < min(b): return "lower", min(b) - max(a)
    if min(a) > max(b): return "higher", min(a) - max(b)
    return None, None


def verdict(step, region, skip_docs=()):
    A, B = ([kl(r, region, skip_docs) for r in step["groups"][g]] for g in (step["a"], step["b"]))
    side, gap = rule(A, B)
    if side is None:
        return "no difference beyond run-to-run variation (ranges overlap)", A, B
    return f"{step['a']} {'above' if side == 'higher' else 'below'} every {step['b']} run, by {gap:.4f}", A, B


rng = lambda v: f"{min(v):.4f}" if len(v) == 1 else f"{min(v):.4f}-{max(v):.4f}"
README = (HERE / "README.md").read_text()


def run_link(rid):
    date, _, rest = rid.partition("_")
    tag = rest.rsplit("_decode-prefill", 1)[1].lstrip("-") or "run"
    return f"[{date} {tag}](../../runs/{rid}/) ([summary](../../runs/{rid}/summary.json))"


def readme_table(st, regions):
    """The per-run table of README.md's Phase 1 section, computed here (recompute checks that README carries it verbatim)."""
    out = ["| Run | Group | " + " | ".join(f"KL {r}" for r in regions) + " |", "|---|---|" + "---:|" * len(regions)]
    for g, rids in st["groups"].items():
        out += [f"| {run_link(rid)} | {g} | " + " | ".join(f"{kl(rid, r):.4f}" for r in regions) + " |" for rid in rids]
    out += [f"| **range** | {g} | " + " | ".join(rng([kl(x, r) for x in rids]) for r in regions) + " |" for g, rids in st["groups"].items()]
    return "\n".join(out)


verdicts = {}
for st in P1["steps"]:
    regions = st["regions"] + (["16,000 and above"] if st["step"] == 4 else [])
    print(f"## Step {st['step']} (amendment {st['amendment']}): {st['question']}")
    print(f"Rule: {st['rule']}.\n")
    tab = readme_table(st, regions)
    print(tab)
    check(f"step {st['step']}: this table appears verbatim in README.md", tab in README)
    print("\n| region | " + " | ".join(st["groups"]) + " | verdict |")
    print("|---|" + "---|" * (len(st["groups"]) + 1))
    for r in st["regions"]:
        v, A, B = verdict(st, r)
        ruled = r == st.get("rule_region", r)
        verdicts[(st["step"], r)] = v
        print(f"| {r} | " + " | ".join(rng([kl(x, r) for x in rids]) for rids in st["groups"].values())
              + f" | {v if ruled else 'reported only: ' + v} |")
    print()
    for r in st["regions"]:
        for g, lohi in st["stated"].get(r, {}).items():   # the range quoted in the next amendment, written before the next step's data
            got = [kl(x, r) for x in st["groups"][g]]
            check(f"step {st['step']}, {r}, {g}: range as quoted in amendment {st['amendment'] + 1} ({lohi[0]:.4f}-{lohi[1]:.4f})",
                  abs(min(got) - lohi[0]) < 6e-5 and abs(max(got) - lohi[1]) < 6e-5)
    print()

st4 = next(st for st in P1["steps"] if st["step"] == 4)
lo4, hi4 = BOUNDS[st4["rule_region"]]
print(f"Step 4, per-prompt means in the {st4['rule_region']} bin (min-max over prompts):")
for g, rids in st4["groups"].items():
    for rid in rids:
        per = [kl(rid, st4["rule_region"], skip_docs=[d for d in range(12) if d != doc]) for doc in sorted({r["doc_id"] for r in rows(rid)})]
        print(f"  {rid}: {min(per):.3f}-{max(per):.3f}")
ends = {rid: (lambda v: sum(v) / len(v))([r["kl_top20"] for r in rows(rid) if r["i"] >= 8000 and r["kl_top20"] is not None])
        for rids in st4["groups"].values() for rid in rids}
print("Step 4, the last bin taken to the end of each trace (8,000 and above): " + ", ".join(f"{v:.4f}" for v in ends.values()))
a4, b4 = ([ends[r] for r in st4["groups"][g]] for g in (st4["a"], st4["b"]))
check("step 4: same verdict with the last bin taken to the end of each trace", rule(a4, b4)[0] == "higher")
check("step 4: in 8,000-15,999 both 0.7.0 runs above both 0.9.1 runs", verdicts[(4, "8,000-15,999")].startswith("0.7.0, DFlash2 x5, 16,000 tokens above"))
print()

# what each amendment's reading rule gives (the verdicts the next amendment recorded before the next data)
check("step 1: below 2,044 overlap; from 2,048 every 0.7.0 run above every 0.9.1 run",
      verdicts[(1, "below 2,044")].startswith("no difference") and verdicts[(1, "from 2,048")].startswith("0.7.0, speculation off above"))
check("step 2: x5 and x3 overlap in both regions", all(verdicts[(2, r)].startswith("no difference") for r in ("below 2,044", "from 2,048")))
check("step 3: 0.7.0 x5 inside the 0.9.1 speculation-on range in both regions",
      all(verdicts[(3, r)].startswith("no difference") for r in ("below 2,044", "from 2,048")))

print("## Across steps (descriptive): speculation on vs off below 2,044 tokens, within each engine")
tab = ["| Engine | Speculation on, KL below 2,044 | Speculation off, KL below 2,044 | |", "|---|---|---|---|"]
for pr in P1["cross_step"]["pairs"]:
    on, off = [kl(r, "below 2,044") for r in pr["on"]], [kl(r, "below 2,044") for r in pr["off"]]
    side, gap = rule(on, off)
    tab.append(f"| {pr['engine']} | {rng(on)} ({len(on)} runs) | {rng(off)} ({len(off)} runs) | "
               + (f"every on run above every off run, by {gap:.4f}" if side == "higher" else "ranges overlap" if side is None else "on below off") + " |")
    check(f"{pr['engine']}: every speculation-on run above every speculation-off run below 2,044", side == "higher")
tab = "\n".join(tab); print(tab)
check("across-steps table appears verbatim in README.md", tab in README)
for pr in P1["cross_step"]["pairs"]:
    on, off = [kl(r, "from 2,048") for r in pr["on"]], [kl(r, "from 2,048") for r in pr["off"]]
    print(f"  {pr['engine']}, from 2,048 (for contrast): on {rng(on)}, off {rng(off)}: "
          + ("ranges overlap" if rule(on, off)[0] is None else "separated"))

print("\n## Sensitivity (descriptive): prompt 0 overlapped the engine's start-up warm-up requests in every run")
for st in P1["steps"]:
    for r in st["regions"]:
        if r != st.get("rule_region", r): continue
        v = verdict(st, r, skip_docs=(0,))[0]
        kind = lambda x: "none" if x.startswith("no difference") else ("above" if " above " in x else "below")
        print(f"  step {st['step']}, {r}, without prompt 0: {v}" + ("" if kind(v) == kind(verdicts[(st["step"], r)]) else "  (differs from the rule's verdict)"))
for pr in P1["cross_step"]["pairs"]:
    on, off = [kl(r, "below 2,044", (0,)) for r in pr["on"]], [kl(r, "below 2,044", (0,)) for r in pr["off"]]
    side, gap = rule(on, off)
    print(f"  across steps, {pr['engine']}, below 2,044, without prompt 0: on {rng(on)}, off {rng(off)}: "
          + (f"every on run above every off run, by {gap:.5f}" if side == "higher" else "ranges overlap"))

print("\n## Cost (costs.jsonl): GPU-hours = 2 x wall hours, server start to stop, one arm at a time")
cost = {c["step"]: c for c in map(json.loads, (HERE / "costs.jsonl").read_text().splitlines()) if c.get("phase") == 1}
print("| step | arms | wall hours | GPU-hours |"); print("|---|---:|---:|---:|")
for st in P1["steps"]:
    c = cost.get(st["step"])
    if not c:
        print(f"| {st['step']} | - | not recorded | |"); fails.append(f"cost of step {st['step']}"); continue
    w = sum(a["wall_s"] for a in c["arms"]) / 3600
    print(f"| {st['step']} | {len(c['arms'])} | {w:.2f} | {c['allocated_gpus'] * w:.2f} |")
    check(f"step {st['step']} cost: arms are this step's new runs, hours add up",
          {a["run"] for a in c["arms"]} <= {r for g in st["groups"].values() for r in g}
          and abs(c["wall_hours"] - round(w, 3)) < 1e-9 and abs(c["gpu_hours"] - round(c["allocated_gpus"] * w, 3)) < 1e-9)
tw = sum(a["wall_s"] for c in cost.values() for a in c["arms"]) / 3600
print(f"| all | {sum(len(c['arms']) for c in cost.values())} | {tw:.2f} | {2 * tw:.2f} |")

print("\n## Checks (checks.jsonl)")
for line in (HERE / "checks.jsonl").read_text().splitlines():
    c = json.loads(line); print(f"  {c['id']}: {c['result']}")
print("\nOK" if not fails else f"\nFAILED: {fails}")
sys.exit(1 if fails else 0)
