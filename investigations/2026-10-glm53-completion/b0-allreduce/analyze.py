#!/usr/bin/env python3
"""Verdict and mismatch classification for a B0 all-reduce fixture run (CPU only).

usage: analyze.py <run-dir>          writes <run-dir>/RESULT.json and RESULT.md

Verdict rules (frozen in PROPOSAL.md before any GPU run):
  INCOMPLETE  a rank did not complete (error/hang); a hang is reported as a
              synchronisation failure of the transport if it occurred in a B12x replay.
  INVALID     dispatch, transport policy, launch geometry, epoch accounting or schedule
              differs from the expectation, or the reference arm (PyNCCL graphs, PyNCCL
              eager prefill) has any mismatch.  No B12x verdict is drawn.
  FAIL        any B12x-dispatched output element differs bitwise from the reference on
              either rank, or the ranks' final outputs differ.
  CLEAN       none of the above.
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import patterns as pat  # noqa: E402

# _tp2_plain_remote_push_geometry (pcie_oneshot.py:1535-1579): rows -> transport, threads,
# blocks.  Checked against the source by cpu_tests.py.
def expected_geometry(rows: int):
    if rows == 1:
        return ("tp2_remote_push", 64, 36)
    if rows == 2:
        return ("tp2_remote_push_stream", 128, 36)
    if rows <= 4:
        return ("tp2_remote_push_stream", 256, 36)
    if rows <= 6:
        return ("tp2_remote_push", 64, 16)
    if rows <= 8:
        return ("tp2_remote_push", 256, 32)
    if rows <= 14:
        return ("tp2_remote_push", 64, 16)
    if rows <= 16:
        return ("tp2_remote_push_stream", 64, 36)
    if rows <= 22:
        return ("tp2_remote_push", 128, 16)
    if rows <= 24:
        return ("tp2_remote_push_stream", 128, 32)
    return ("tp2_remote_push", 128, 16)


EXPECTED_BACKENDS = {"ca": "B12X_PCIE_ONESHOT", "pynccl": True, "fi_ar": False,
                     "qr": False, "aiter": False, "max_size": 393216,
                     "enable_eager": False}
EXPECTED_POLICY = [False, False, True, False, True]


def load_rank(run: str, rank: int) -> dict:
    d = os.path.join(run, f"rank{rank}")
    out = {"dir": d}
    for name in ("status", "result", "hang", "progress"):
        p = os.path.join(d, f"{name}.json")
        if os.path.exists(p):
            with open(p) as f:
                out[name] = json.load(f)
    p = os.path.join(d, "schedule.jsonl")
    if os.path.exists(p):
        with open(p) as f:
            out["schedule"] = [tuple(json.loads(line)) for line in f if line.strip()]
    out["fails"] = sorted(glob.glob(os.path.join(d, "fail_g*.pt")))
    return out


def push_sequence(schedule, cases_by_gid):
    """Global order of B12x peer-push calls: (step, gid, layer)."""
    seq = []
    for s, gid, mode, transport in schedule:
        if transport != "push":
            continue
        for layer in range(cases_by_gid[gid]["layers"]):
            seq.append((s, gid, layer))
    index = {(s, layer): n for n, (s, _, layer) in enumerate(seq)}
    return seq, index


def classify_failure(rank, gid, layer, step, y, cases_by_gid, seq, index, schedule,
                     max_prev=6):
    case = cases_by_gid[gid]
    n_el = case["rows"] * pat.HIDDEN
    peer = 1 - rank
    local = pat.generate_layer(step, layer, rank, n_el)
    peer_in = pat.generate_layer(step, layer, peer, n_el)
    prev_calls_peer, prev_calls_local = [], []
    n = index.get((step, layer))
    if n is not None:
        for k in range(1, max_prev + 1):
            if n - k < 0:
                prev_calls_peer.append(None)
                continue
            s2, g2, l2 = seq[n - k]
            n2 = cases_by_gid[g2]["rows"] * pat.HIDDEN
            prev_calls_peer.append(pat.generate_layer(s2, l2, peer, n2))
            if k <= 2:
                prev_calls_local.append(pat.generate_layer(s2, l2, rank, n2))
    earlier = [s for (s, g, m, t) in schedule if g == gid and s < step][-2:][::-1]
    prev_steps_peer = [pat.generate_layer(s2, layer, peer, n_el) for s2 in earlier]
    out = pat.classify(y.view(-1)[:n_el], local, peer_in,
                       prev_calls_peer=prev_calls_peer, prev_steps_peer=prev_steps_peer,
                       prev_calls_local=prev_calls_local)
    out.update({"rank": rank, "gid": gid, "layer": layer, "step": step,
                "push_call_index": n})
    return out


def analyze(run: str) -> dict:
    ranks = [load_rank(run, r) for r in (0, 1)]
    problems, invalid, fails = [], [], []
    verdict = None
    for r, rk in enumerate(ranks):
        st = rk.get("status", {}).get("status")
        if st != "completed":
            problems.append(f"rank{r} status={st} {rk.get('status', {}).get('error', '')}")
            if "hang" in rk:
                problems.append(f"rank{r} hang at {rk['hang'].get('where')}, last "
                                f"{rk['hang'].get('last')}")
    if problems:
        verdict = "INCOMPLETE"
    res0 = ranks[0].get("result")
    res1 = ranks[1].get("result")
    report = {"run": run, "problems": problems}
    if verdict is None and (res0 is None or res1 is None):
        verdict = "INCOMPLETE"
        problems.append("missing result.json")
    if verdict is not None:
        report["verdict"] = verdict
        return report

    cases = {c["gid"]: c for c in res0["cases"]}
    # B: dispatch, policy, geometry
    for r, res in enumerate((res0, res1)):
        b = res["backends"]
        for k, v in EXPECTED_BACKENDS.items():
            if b.get(k) != v:
                invalid.append(f"rank{r} backend {k}={b.get(k)} expected {v}")
        if list(b["transport_policy"]) != EXPECTED_POLICY:
            invalid.append(f"rank{r} transport policy {b['transport_policy']}")
        for c in res["cases"]:
            if c["dispatch"] != c["expected_dispatch"]:
                invalid.append(f"rank{r} gid {c['gid']} dispatch {c['dispatch']}")
        geo = {}
        for p in res["plans"]:
            shape = p["key"][1]
            rows = int(shape.strip("()").split(",")[0])
            geo[rows] = (p["transport"], p["threads"], p["blocks"])
        for c in res["cases"]:
            if c["expected_dispatch"] == "b12x":
                g = geo.get(c["rows"])
                if g is None:
                    invalid.append(f"rank{r} no prepared plan for rows {c['rows']}")
                elif tuple(g) != expected_geometry(c["rows"]):
                    invalid.append(f"rank{r} rows {c['rows']} geometry {g} expected "
                                   f"{expected_geometry(c['rows'])}")
        # C: epoch accounting
        cap = res["epoch_after_capture"]
        if cap["delta"] != cap["expected_push_calls"]:
            invalid.append(f"rank{r} capture-warmup epoch delta {cap['delta']} != "
                           f"{cap['expected_push_calls']}")
        for ph in res["phases"]:
            if ph["epoch_delta"] != ph["expected_push_calls"]:
                invalid.append(f"rank{r} phase {ph['phase']} epoch delta "
                               f"{ph['epoch_delta']} != {ph['expected_push_calls']}")
    # D: identical schedules
    if ranks[0].get("schedule") != ranks[1].get("schedule"):
        invalid.append("rank schedules differ")
    # E: reference arm
    ref_bad = []
    for r, res in enumerate((res0, res1)):
        for c in res["cases"]:
            if c["expected_dispatch"] != "b12x" and c["mismatched_elements"]:
                ref_bad.append(f"rank{r} {c['phase']} {c['kind']} rows {c['rows']} "
                               f"({c['expected_dispatch']}): {c['mismatched_elements']}")
        for g, v in res["warmup_mismatches"].items():
            if cases[int(g)]["expected_dispatch"] != "b12x" and int(v):
                ref_bad.append(f"rank{r} warm-up gid {g} (PyNCCL): {v}")
        if res["prefill_mismatched_elements"]:
            ref_bad.append(f"rank{r} eager PyNCCL prefill: "
                           f"{res['prefill_mismatched_elements']}")
        if not res["cross_rank_reference_identical"]:
            ref_bad.append("references differ across ranks")
    invalid.extend(ref_bad)

    # F: B12x outcome
    b12x_total = {}
    for r, res in enumerate((res0, res1)):
        warm = sum(int(v) for g, v in res["warmup_mismatches"].items()
                   if cases[int(g)]["expected_dispatch"] == "b12x")
        rep = sum(c["mismatched_elements"] for c in res["cases"]
                  if c["expected_dispatch"] == "b12x")
        b12x_total[f"rank{r}"] = {"replays": rep, "warmups": warm}
        if rep or warm:
            fails.append(f"rank{r}: {rep} mismatched elements in replays, {warm} in "
                         f"warm-ups")
        if not res["cross_rank_last_output_identical"]:
            fails.append(f"rank{r}: final outputs differ across ranks")

    classifications = []
    for r, rk in enumerate(ranks):
        if not rk["fails"]:
            continue
        seq, index = push_sequence(rk["schedule"], cases)
        for path in rk["fails"]:
            rec = torch.load(path)
            gid = rec["meta"]["gid"]
            seen = (rec["fstep"] >= 0) | (rec["lstep"] >= 0)   # replays and warm-ups
            for layer in torch.nonzero(seen).view(-1).tolist():
                for which, steps, ys in (("first", rec["fstep"], rec["fy"]),
                                         ("last", rec["lstep"], rec["ly"])):
                    step = int(steps[layer])
                    if step < 0:
                        continue
                    c = classify_failure(r, gid, layer, step, ys[layer], cases, seq,
                                         index, rk["schedule"])
                    c["which"] = which
                    c["mismatched_elements_layer"] = int(rec["acc"][layer])
                    classifications.append(c)

    by_phase = {}
    for r, res in enumerate((res0, res1)):
        for c in res["cases"]:
            key = f"{c['phase']}/{c['kind']}/rows{c['rows']}/{c['expected_dispatch']}"
            by_phase.setdefault(key, [0, 0])[r] += c["mismatched_elements"]

    if invalid:
        verdict = "INVALID"
    elif fails:
        verdict = "FAIL"
    else:
        verdict = "CLEAN"
    report.update({
        "verdict": verdict, "invalid": invalid, "fails": fails,
        "b12x_mismatches": b12x_total, "mismatches_by_case": by_phase,
        "phases": {f"rank{r}": res["phases"] for r, res in enumerate((res0, res1))},
        "capture": {"seconds": res0["capture"]["seconds"],
                    "epoch_after_capture": res0["epoch_after_capture"]},
        "plans": res0["plans"], "backends": res0["backends"], "plan": res0["plan"],
        "classifications": classifications,
        "mismatch_steps": {f"rank{r}": res["mismatch_steps"][:200]
                           for r, res in enumerate((res0, res1))},
    })
    return report


def write_md(rep: dict, path: str) -> None:
    lines = [f"# B0 TP2 all-reduce fixture: {rep['verdict']}", ""]
    for key in ("problems", "invalid", "fails"):
        if rep.get(key):
            lines.append(f"## {key}")
            lines.extend(f"- {x}" for x in rep[key])
            lines.append("")
    if "phases" in rep:
        lines.append("## Phases (rank 0)")
        lines.append("| phase | forwards | seconds | B12x push calls | epoch delta |")
        lines.append("|---|---|---|---|---|")
        for ph in rep["phases"]["rank0"]:
            lines.append(f"| {ph['phase']} | {ph['forwards']} | {ph['seconds']} | "
                         f"{ph['expected_push_calls']} | {ph['epoch_delta']} |")
        lines.append("")
        lines.append("## Mismatched elements by case (rank0, rank1)")
        for k, v in sorted(rep["mismatches_by_case"].items()):
            lines.append(f"- {k}: {v[0]}, {v[1]}")
        lines.append("")
    if rep.get("classifications"):
        lines.append("## Classified failing outputs")
        for c in rep["classifications"][:60]:
            lines.append(f"- rank{c['rank']} gid {c['gid']} layer {c['layer']} step "
                         f"{c['step']} ({c['which']}): words {c['word_classes']}; "
                         f"packs {c['pack_classes']}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    run = sys.argv[1]
    rep = analyze(run)
    with open(os.path.join(run, "RESULT.json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    write_md(rep, os.path.join(run, "RESULT.md"))
    print(rep["verdict"])
    for key in ("problems", "invalid", "fails"):
        for x in rep.get(key, [])[:20]:
            print(f"  {key}: {x}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
