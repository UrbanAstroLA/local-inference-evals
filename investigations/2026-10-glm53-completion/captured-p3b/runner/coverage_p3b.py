#!/usr/bin/env python3
"""Capture-run coverage from the bookkeeping logs only (host python3, no numpy, no PRIVATE reads).

usage: coverage_p3b.py ATTEMPT_DIR      (reads ATTEMPT_DIR/caplog/*.jsonl|json and client.jsonl)
writes ATTEMPT_DIR/coverage.json and coverage.md

Capture-level verdict (PROPOSAL.md section 6):
  CAPTURE COMPLETE  every MoE minimum met on every captured layer, every attention category's emitted groups at or
                    above its minimum, no torn entry kept, no capture error, flush completed, every MoE bind record
                    present for the served shapes captured;
  INCOMPLETE        otherwise (the deficit is listed).
Exact attention regimes (dense / clamped / boundary-crossing, r = L - (p + 1)) need the captured positions and
lengths and are counted in the replay, inside a container, from PRIVATE-capture; the replay applies the row minima.
"""
import glob
import json
import os
import sys

MOE_MIN = {32: 8, 24: 4, 16: 4}            # traffic-phase entries per captured layer
MOE_MIN_SMALL = 2                          # entries with m <= 8, per layer
ATT_GROUP_MIN = {"dense": 8, "boundary": 10, "clamped": 6}   # emitted request groups (approximate categories)
MOE_LAYERS = (3, 23, 44)
ATT_LAYERS = (3, 23)


def load_jsonl(paths):
    out = []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    out.append(dict(kind="unparsable"))
    return out


def main():
    ao = sys.argv[1]
    cl = os.path.join(ao, "caplog")
    recs = load_jsonl(sorted(glob.glob(os.path.join(cl, "cap_*.jsonl"))))
    steps = load_jsonl(sorted(glob.glob(os.path.join(cl, "steps_*.jsonl"))))
    flushed = json.load(open(os.path.join(cl, "FLUSHED.json"))) if os.path.exists(os.path.join(cl, "FLUSHED.json")) else None
    status = json.load(open(os.path.join(cl, "status.json"))) if os.path.exists(os.path.join(cl, "status.json")) else None
    client = load_jsonl([os.path.join(ao, "client.jsonl")]) if os.path.exists(os.path.join(ao, "client.jsonl")) else []
    entries = [r for r in recs if r.get("kind") == "entry"]
    moe = {}
    for e in entries:
        if e["ring"] == "moe":
            k = (e["layer"], e.get("phase"), e["m_static"])
            moe[k] = moe.get(k, 0) + 1
    att_groups = {}
    att_entries = {}
    for e in entries:
        if e["ring"] == "att":
            att_entries[e["layer"]] = att_entries.get(e["layer"], 0) + 1
            for c in e.get("cats") or []:
                att_groups.setdefault(e["layer"], {}).setdefault(c, 0)
                att_groups[e["layer"]][c] += 1
    binds = {}
    for r in recs:
        if r.get("kind") == "site" and r.get("key", [None])[0] == "moe_bind":
            binds[(r["layer"], r["m"])] = dict(plan=r["plan"], launch=r.get("launch"))
    deficits = []
    for L in MOE_LAYERS:
        for m, need in MOE_MIN.items():
            have = moe.get((L, "traffic", m), 0)
            if have < need:
                deficits.append(f"moe L{L} m={m}: {have} < {need}")
        small = sum(v for (l_, ph, m), v in moe.items() if l_ == L and ph == "traffic" and m <= 8)
        if small < MOE_MIN_SMALL:
            deficits.append(f"moe L{L} m<=8: {small} < {MOE_MIN_SMALL}")
    for L in ATT_LAYERS:
        for c, need in ATT_GROUP_MIN.items():
            have = att_groups.get(L, {}).get(c, 0)
            if have < need:
                deficits.append(f"att L{L} {c} groups: {have} < {need}")
    for (L, ph, m) in moe:
        if (L, m) not in binds:
            deficits.append(f"moe L{L} m={m}: no bind record (plan unknown)")
    errors = [r for r in recs if r.get("kind") == "capture_error"]
    torn = [r for r in recs if r.get("kind") == "torn"]
    abandoned = [r for r in recs if r.get("kind") == "abandoned"]
    if errors:
        deficits.append(f"{len(errors)} capture error record(s)")
    if not flushed:
        deficits.append("no FLUSHED.json (flush not completed)")
    elif flushed.get("pending_at_flush"):
        deficits.append(f"{flushed['pending_at_flush']} armed entries unresolved at flush")
    cg = {}
    for s in steps:
        k = f"{s.get('cg')}:{s.get('ntok_pad')}"
        cg[k] = cg.get(k, 0) + 1
    sent = [c for c in client if c.get("kind") == "sent"]
    done = [c for c in client if c.get("kind") == "request"]
    stop = [c for c in client if c.get("kind") == "stop"]
    res = dict(
        verdict="CAPTURE COMPLETE" if not deficits else "INCOMPLETE",
        deficits=deficits,
        moe_entries={f"L{L}/{ph}/m{m}": v for (L, ph, m), v in sorted(moe.items(), key=lambda x: (x[0][0], str(x[0][1]), x[0][2]))},
        moe_bind={f"L{L}/m{m}": v for (L, m), v in sorted(binds.items())},
        att_entries={f"L{L}": v for L, v in sorted(att_entries.items())},
        att_groups={f"L{L}": v for L, v in sorted(att_groups.items())},
        abandoned=len(abandoned), torn_detected=len(torn), capture_errors=len(errors),
        errors=[e.get("where") + ": " + e.get("err", "")[:160] for e in errors[:10]],
        flushed={k: flushed.get(k) for k in ("pending_at_flush", "flush_wait_s", "emitted_moe", "emitted_att",
                                             "armed_moe", "armed_att", "torn", "errors", "dev_commits", "oob",
                                             "arm_skipped_backpressure", "arm_skipped_lock", "extra_prepare")}
        if flushed else None,
        init=[{k: r.get(k) for k in ("rank", "inert", "ring_mib_total", "issues", "async_scheduling", "file_sha256")}
              for r in recs if r.get("kind") == "init"],
        steps=dict(n=len(steps), by_cg_and_padded_size=dict(sorted(cg.items()))),
        client=dict(sent=len(sent), completed=len(done), stopped_early=bool(stop),
                    seeds=sorted({c["seed"] for c in sent}), finish=sorted({str(c.get("finish")) for c in done}),
                    errors=[c.get("error") for c in done if c.get("error")][:5]),
    )
    json.dump(res, open(os.path.join(ao, "coverage.json"), "w"), indent=1, default=str)
    md = [f"# Capture coverage: {res['verdict']}", ""]
    md += [f"- deficit: {d}" for d in deficits] or ["- no deficit"]
    md += ["", "| MoE layer/phase/m | entries |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in res["moe_entries"].items()]
    md += ["", "| attention layer | entries | groups by approximate category |", "|---|---:|---|"]
    md += [f"| {k} | {res['att_entries'].get(k, 0)} | {res['att_groups'].get(k, {})} |" for k in res["att_entries"]]
    md += ["", f"abandoned {res['abandoned']}, torn detected {res['torn_detected']}, capture errors {res['capture_errors']}",
           f"client: sent {len(sent)}, completed {len(done)}, stopped early {bool(stop)}"]
    open(os.path.join(ao, "coverage.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
