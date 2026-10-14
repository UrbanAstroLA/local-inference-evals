#!/usr/bin/env python3
"""Recompute every number in this investigation's README from the published rows (standard library only).

    python3 investigations/2026-10-glm53-completion/recompute.py

Reads runs/*/results.jsonl through tools/analyze.py. Phase 0 publishes no new rows: it re-reads the two three-pass 3.25bpw
GPQA records (tpurtell 0.7.0, DFlash2 x5; tpurtell 0.9.1, DFlash2 x3), and the 4bpw TR3 (Brandon) record on 0.9.1 for reference.
Phase 1 (steps 1-4, decode vs prefill) reads the decode-prefill runs named in manifest.json "phase1", applies each step's
rule (PLAN.md amendments 1-4) and the costs in costs.jsonl. Steps 5-6 (manifest.json "phase1.screens") apply the reading
rules of amendments 5, 6 and 6a to prompts 1-5, positions 8,000-15,999, and report the secondary measures.
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

# ---------------------------------------------------------------- Phase 1 steps 5-6 (PLAN.md amendments 5, 6 and 6a)
SC = P1["screens"]
SUB, SBIN = SC["subset"]["prompts"], SC["subset"]["bin"]
print(f"\n# Phase 1 steps 5-6: prompts {SUB[0]}-{SUB[-1]}, positions {SBIN} (manifest.json 'phase1.screens')\n")


def sel(rid, docs, region):
    lo, hi = BOUNDS[region]
    return [r for r in rows(rid) if r["doc_id"] in docs and lo <= r["i"] <= hi]


def mean_of(rs, key):
    v = [r[key] for r in rs if r[key] is not None]
    return sum(v) / len(v)


agg = lambda rid, docs=tuple(SUB), region=SBIN: mean_of(sel(rid, docs, region), "kl_top20")
per_prompt = lambda rid: [agg(rid, (d,)) for d in SUB]
PLAN = " ".join((HERE / "PLAN.md").read_text().split())


def screen_table(st):
    cols = ["KL, prompts 1-5"] + [f"Prompt {d}" for d in SUB] + ["Sampled-token abs dlp", "Top-1 agreement", "Prompt 0"]
    val = lambda rid: ([agg(rid)] + per_prompt(rid) + [mean_of(sel(rid, SUB, SBIN), "abs_dlp"), mean_of(sel(rid, SUB, SBIN), "top1_agree"),
                                                       agg(rid, (0,))])
    out = ["| Run | Group | " + " | ".join(cols) + " |", "|---|---|" + "---:|" * len(cols)]
    for g, rids in st["groups"].items():
        out += [f"| {run_link(rid)} | {g} | " + " | ".join(f"{x:.4f}" for x in val(rid)) + " |" for rid in rids]
    out += [f"| **range** | {g} | " + " | ".join(rng([val(r)[k] for r in rids]) for k in range(len(cols))) + " |" for g, rids in st["groups"].items()]
    return "\n".join(out)


def bins_table(st):
    regs = ["0-2,043", "2,048-7,999", "8,000-15,999", "16,000 and above"]
    out = ["| Run | Group | " + " | ".join(f"KL {r}" for r in regs) + " |", "|---|---|" + "---:|" * len(regs)]
    for g, rids in st["groups"].items():
        out += [f"| {run_link(rid)} | {g} | " + " | ".join(f"{agg(rid, tuple(SUB), r):.4f}" for r in regs) + " |" for rid in rids]
    return "\n".join(out)


def screen_verdict(st):
    T, R = st["groups"][st["test"]], st["groups"][st["reference"]]
    t, r = [agg(x) for x in T], [agg(x) for x in R]
    pr = [per_prompt(x) for x in R]
    if st["step"] == 5:     # amendment 5: reduction = below both stock runs, per prompt
        n = [sum(per_prompt(x)[k] < min(p[k] for p in pr) for k in range(len(SUB))) for x in T]
        if max(t) < min(r) and min(n) >= 4:
            v = "the kpool bundle is nominated as affecting the measured discrepancy"
        elif min(t) >= min(r):
            v = "the corrected values remain at stock level; these repairs did not explain the discrepancy under this protocol"
        else:
            v = "unresolved under this screen"
        return v, t, r, n
    C = st["groups"][st["context"]]
    c = [agg(x) for x in C]
    n = [sum(per_prompt(x)[k] > max(p[k] for p in pr) for k in range(len(SUB))) for x in T]
    if min(t) > max(r) and min(t) >= min(c) and min(n) >= 4:      # amendment 6 rule, amendment 6a wording
        v = ("rise condition met: the old layout bundle can reproduce the discrepancy on newer code; "
             "this does not identify which layout component causes it")
    elif min(r) <= min(t) and max(t) <= max(r):
        v = ("stays within the 0.9.1-default range: the bundle alone does not reproduce the discrepancy under these conditions; "
             "old kernel code or a kernel-layout interaction remains possible")
    else:
        v = "unresolved under this screen"
    return v, t, r, n


sv = {}
for st in SC["steps"]:
    print(f"## Step {st['step']} (amendment {st['amendment']}): {st['question']}")
    check(f"step {st['step']}: rule quoted in manifest.json appears verbatim in PLAN.md", st["rule"] in PLAN)
    tab = screen_table(st); print(tab)
    check(f"step {st['step']}: primary table appears verbatim in README.md", tab in README)
    v, t, r, n = screen_verdict(st); sv[st["step"]] = (v, t, r, n)
    word = "below" if st["step"] == 5 else "above"
    line = (f"Prompts in which a {'corrected' if st['step'] == 5 else 'step-6'} run lies {word} both "
            f"{'stock' if st['step'] == 5 else '0.9.1-default'} runs: " + ", ".join(f"{k} of {len(SUB)} (r{i + 1})" for i, k in enumerate(n)) + ".")
    print("\n" + line)
    check(f"step {st['step']}: per-prompt count sentence appears in README.md", line in README)
    print(f"Verdict: {v}.")
    check(f"step {st['step']}: verdict appears in README.md", v in README)
    print("\nSecondary bins, prompts 1-5:")
    tab = bins_table(st); print(tab)
    check(f"step {st['step']}: secondary-bin table appears verbatim in README.md", tab in README)
    for key, got in (("test", t), ("reference", r)):
        if key in st["stated"]:
            check(f"step {st['step']}: {key} aggregates as quoted in amendment {st['stated']['amendment']} {st['stated'][key]}",
                  all(abs(a - b) < 6e-5 for a, b in zip(got, st["stated"][key])))
    if "context" in st["stated"]:
        got = [agg(x) for x in st["groups"][st["context"]]]
        check(f"step {st['step']}: context aggregates as quoted in amendment {st['stated']['amendment']} {st['stated']['context']}",
              all(abs(a - b) < 6e-5 for a, b in zip(got, st["stated"]["context"])))
    if "prompts_below_both_reference" in st["stated"]:
        check(f"step {st['step']}: per-prompt counts as quoted in amendment {st['stated']['amendment']}", n == st["stated"]["prompts_below_both_reference"])
    print()
check("step 5: corrected values remain at stock level", sv[5][0].startswith("the corrected values remain at stock level"))
check("step 6: rise condition met (both above both 0.9.1-default runs, inside or above the stock range, 4+ of 5 prompts per run)",
      sv[6][0].startswith("rise condition met"))
for st in SC["steps"]:
    rs = [agg(x, tuple(SUB), "8,000-15,999") for x in st["groups"][st["test"]]]
    t2 = [mean_of([r for r in rows(x) if r["doc_id"] in SUB and r["i"] >= 8000], "kl_top20") for x in st["groups"][st["test"]]]
    print(f"  step {st['step']}, the bin taken to the end of each trace (8,000 and above): " + ", ".join(f"{v:.4f}" for v in t2)
          + f" (rule bin: {', '.join(f'{v:.4f}' for v in rs)})")

print("\n## Cost (costs.jsonl): GPU-hours = 2 x wall hours, server start to stop, one arm at a time")
from decimal import Decimal, ROUND_HALF_UP
hours = lambda wall_s, gpus=1: str((Decimal(wall_s * gpus) / 3600).quantize(Decimal("0.01"), ROUND_HALF_UP))   # exact, half up
cost = {c["step"]: c for c in map(json.loads, (HERE / "costs.jsonl").read_text().splitlines()) if c.get("phase") == 1}
print("| step | arms | wall hours | GPU-hours |"); print("|---|---:|---:|---:|")
for st in P1["steps"] + SC["steps"]:
    c = cost.get(st["step"])
    if not c:
        print(f"| {st['step']} | - | not recorded | |"); fails.append(f"cost of step {st['step']}"); continue
    ws = sum(a["wall_s"] for a in c["arms"]); w = ws / 3600
    print(f"| {st['step']} | {len(c['arms'])} | {hours(ws)} | {hours(ws, c['allocated_gpus'])} |")
    check(f"step {st['step']} cost: arms are this step's new runs, hours add up",
          {a["run"] for a in c["arms"]} <= {r for g in st["groups"].values() for r in g}
          and abs(c["wall_hours"] - round(w, 3)) < 1e-9 and abs(c["gpu_hours"] - round(c["allocated_gpus"] * w, 3)) < 1e-9)
tws = sum(a["wall_s"] for c in cost.values() for a in c["arms"])
print(f"| all | {sum(len(c['arms']) for c in cost.values())} | {hours(tws)} | {hours(tws, 2)} |")
tws1 = sum(a["wall_s"] for k, c in cost.items() if isinstance(k, int) for a in c["arms"])   # Phase 1 steps
s14 = sum(a["wall_s"] for k, c in cost.items() if isinstance(k, int) and k <= 4 for a in c["arms"])
check("steps 1-4: 7.18 GPU-hours; steps 5 and 6: 1.62 and 1.75; total 10.55, within the plan's 12 GPU-hour discovery ceiling",
      (hours(s14, 2), hours(sum(a["wall_s"] for a in cost[5]["arms"]), 2), hours(sum(a["wall_s"] for a in cost[6]["arms"]), 2), hours(tws1, 2))
      == ("7.18", "1.62", "1.75", "10.55") and tws1 * 2 / 3600 <= 12)
later = {k: c for k, c in cost.items() if not isinstance(k, int)}   # Tranche 1 runs (amendments 8-9), labelled by name
for k, c in later.items():
    print(f"  {k}: {hours(sum(a['wall_s'] for a in c['arms']), 2)} GPU-hours")
gh = {k: c["gpu_hours"] for k, c in later.items()}
check("pilot 2.01 and replication 1.54 GPU-hours (two GPUs each); later runs logged with their own GPU counts",
      [hours(sum(a["wall_s"] for a in c["arms"]), 2) for k, c in later.items() if c["allocated_gpus"] == 2] == ["2.01", "1.54"]
      and all(abs(c["gpu_hours"] - round(c["allocated_gpus"] * sum(a["wall_s"] for a in c["arms"]) / 3600, 3)) < 1e-9 for c in later.values()))
allcost = [json.loads(l) for l in (HERE / "costs.jsonl").read_text().splitlines()]
print(f"  cumulative GPU-hours, every ledger entry: {sum(c['gpu_hours'] for c in allcost):.2f}")
st = lambda prefix: [c for c in allcost if str(c.get("step", "")).startswith(prefix)]
sa, bg, bs, b2 = st("stage A verification"), st("bundle gate"), st("bundle screen"), st("bundle seed-1235 pass")
print(f"  stage A {sum(c['gpu_hours'] for c in sa):.4f} ({len(sa)} entries), bundle gate {sum(c['gpu_hours'] for c in bg):.4f}, "
      f"bundle screen {sum(c['gpu_hours'] for c in bs):.3f}, bundle seed-1235 pass {sum(c['gpu_hours'] for c in b2):.3f}")
check("stage A 0.6929 of 2.0, bundle gate 1.6403 of 2.0 (arms add up), screen 4.841 of 7.5, seed-1235 pass 4.541 of 7.5 "
      "(2 GPUs x 8,174 s); cumulative 47.79 GPU-hours before amendment 16",
      round(sum(c["gpu_hours"] for c in sa), 4) == 0.6929 and round(sum(c["gpu_hours"] for c in bg), 4) == 1.6403
      and all(abs(c["gpu_hours"] - round(sum(a["gpu_hours"] for a in c["arms"]), 4)) < 1e-9 for c in bg)
      and round(sum(c["gpu_hours"] for c in bs), 3) == 4.841 and len(b2) == 1 and b2[0]["gpu_hours"] == 4.541
      and b2[0]["gpu_hours"] == round(2 * sum(a["wall_s"] for a in b2[0]["arms"]) / 3600, 3)
      and f"{sum(c['gpu_hours'] for c in allcost[:21]):.2f}" == "47.79")
ar, kv, ca = st("B0 TP2 all-reduce fixture"), st("KV-writer / query-projection comparison"), st("B0 cache isolation")
print(f"  amendment 16: all-reduce fixture {sum(c['gpu_hours'] for c in ar):.4f}, KV-writer comparison "
      f"{sum(c['gpu_hours'] for c in kv):.4f}, cache fixture {sum(c['gpu_hours'] for c in ca):.4f} "
      f"({sum(len(c['arms']) for c in ca)} launches, one refused at 0)")
arms_ok = lambda c: (abs(c["gpu_hours"] - round(sum(a["gpu_hours"] for a in c["arms"]), 4)) < 1e-9
                     and all(abs(a["gpu_hours"] - round(a["gpus"] * a["wall_s"] / 3600, 4)) < 1e-9 for a in c["arms"]))
check("amendment 16 as separate steps, arms add up (GPUs x wall): all-reduce 0.07 of 0.5 (smoke 0.0139, full 0.0561), "
      "KV writer 0.0069 of 0.25, cache fixture 0.0392 of 0.5 (incl. the refused zero-cost launch); cumulative 47.91 GPU-hours before amendment 17",
      len(ar) == len(kv) == len(ca) == 1 and all(arms_ok(c) for c in ar + kv + ca)
      and [a["gpu_hours"] for a in ar[0]["arms"]] == [0.0139, 0.0561] and ar[0]["gpu_hours"] == 0.07
      and kv[0]["gpu_hours"] == 0.0069 and ca[0]["gpu_hours"] == 0.0392
      and sum(a["wall_s"] == 0 and a["gpu_hours"] == 0.0 for a in ca[0]["arms"]) == 1
      and f"{sum(c['gpu_hours'] for c in allcost if not str(c.get('step', '')).startswith(('B0 live slot-mapping', 'B1 D7', 'B1 checkpoint GPQA', 'B0 seed-1236', 'K selection', 'Hook validation'))):.2f}" == "47.91")
mg = st("B0 live slot-mapping guard session")
print(f"  amendment 17: slot-mapping guard {sum(c['gpu_hours'] for c in mg):.4f} "
      f"({sum(len(c['arms']) for c in mg)} attempts, one refused at 0, one invalid)")
check("amendment 17 arms add up (GPUs x wall): refused 0, invalid 0.4211, complete 0.5722; 0.9933 of 1.45; cumulative 48.90 GPU-hours before amendment 18",
      len(mg) == 1 and arms_ok(mg[0]) and [a["gpu_hours"] for a in mg[0]["arms"]] == [0.0, 0.4211, 0.5722]
      and mg[0]["gpu_hours"] == 0.9933 and f"{sum(c['gpu_hours'] for c in allcost if not str(c.get('step', '')).startswith(('B1 D7', 'B1 checkpoint GPQA', 'B0 seed-1236', 'K selection', 'Hook validation'))):.2f}" == "48.90")
b1 = st("B1 D7 verification session")
print(f"  amendment 18: B1 D7 verification {sum(c['gpu_hours'] for c in b1):.4f} ({sum(len(c['arms']) for c in b1)} attempt)")
check("amendment 18 arm adds up (GPUs x wall): 0.6244 of 2.0, one attempt; cumulative 49.52 GPU-hours before amendment 19",
      len(b1) == 1 and arms_ok(b1[0]) and b1[0]["gpu_hours"] == 0.6244 and len(b1[0]["arms"]) == 1
      and f"{sum(c['gpu_hours'] for c in allcost if not str(c.get('step', '')).startswith(('B1 checkpoint GPQA', 'B0 seed-1236', 'K selection', 'Hook validation'))):.2f}" == "49.52")
b1s = st("B1 checkpoint GPQA pass")
print(f"  amendment 19: B1 checkpoint pass {sum(c['gpu_hours'] for c in b1s):.3f}")
check("amendment 19 B1 pass: 6.33 of 7.5 (2 GPUs x 11,394 s); cumulative 55.85 GPU-hours before amendment 19d results",
      len(b1s) == 1 and b1s[0]["gpu_hours"] == 6.33 == round(2 * b1s[0]["arms"][0]["wall_s"] / 3600, 2)
      and f"{sum(c['gpu_hours'] for c in allcost if not str(c.get('step', '')).startswith(('B0 seed-1236', 'K selection', 'Hook validation'))):.2f}" == "55.85")
b0s, kk, hv = st("B0 seed-1236 GPQA pass"), st("K selection-kernel validation"), st("Hook validation attempt 1")
print(f"  amendments 19d-21f: B0 seed-1236 {sum(c['gpu_hours'] for c in b0s):.3f}, K {sum(c['gpu_hours'] for c in kk):.4f}, hook validation {sum(c['gpu_hours'] for c in hv):.4f}")
check("amendments 19d-21f: B0 seed-1236 5.866 (2 x 10,559 s), K 0.2825 (0.1417 + 0.1408, 1 GPU), hook validation 1.2117 (2 x 2,181 s); cumulative 63.22 GPU-hours",
      len(b0s) == len(kk) == len(hv) == 1 and b0s[0]["gpu_hours"] == 5.866 == round(2 * 10559 / 3600, 3)
      and arms_ok(kk[0]) and kk[0]["gpu_hours"] == 0.2825 and arms_ok(hv[0]) and hv[0]["gpu_hours"] == 1.2117
      and f"{sum(c['gpu_hours'] for c in allcost):.2f}" == "63.22")

print("\n## Checks (checks.jsonl)")
for line in (HERE / "checks.jsonl").read_text().splitlines():
    c = json.loads(line); print(f"  {c['id']}: {c['result']}")
print("\nOK" if not fails else f"\nFAILED: {fails}")
sys.exit(1 if fails else 0)
