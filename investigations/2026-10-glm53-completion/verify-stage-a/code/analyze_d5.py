#!/usr/bin/env python3
"""D5 (workspace locking) analysis for the verify13 session: CPU only (PLAN.md amendments 13 / 13a).

Sources: verify13/v13_p*.jsonl - ws_resize events (phase, old/new pointer and size, caller, and the
module's classification against the workspace views bound while CUDA graphs were being captured),
capture_model_begin/_end (with every capture binding: pointer, bytes, runtime mode, batch size,
caller), warmup_kernels_begin/_end, armed/disarmed; server.log [WORKSPACE DEBUG] lines
(VLLM_DEBUG_WORKSPACE=1) and "Graph capturing finished".

Phases: pre_capture (profiling; no graph exists yet), graph_capture (between graph captures),
post_capture_warmup (V2 warmup_kernels after capture_model), serving (split by ARM into
serving_before_arm = release start-up warm-up requests, fixture, after_disarm).
Per event: "captured-graph address invalidated" if the freed range [old_ptr, old_ptr+old_size)
contains a workspace address bound during a capture, else "resize, no captured-graph address
invalidated".  Overall (fixed before data):
  any invalidating event            -> "exercised: captured-graph address invalidated"
  resizes in capture/warm-up/serving -> "exercised: resize, no captured-graph address invalidated"
  none                               -> "not exercised in this workload" (never "ruled out")
An invalidated address shows that a captured graph retains freed memory; it does not by itself show
corruption (that needs the freed range reused while a replay writes or reads it: not measured).
Usage: analyze_d5.py RUN_DIR [--selftest]
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys
import tempfile

CAVEAT = ("address-level result: captured bindings are workspace views handed out by WorkspaceManager "
          "while a stream was capturing; a resize alone does not establish graph corruption, and an "
          "invalidated address establishes retained freed memory, not a corrupted replay")


def load(run_dir):
    ev = []
    for f in glob.glob(os.path.join(run_dir, "**", "v13_p*.jsonl"), recursive=True):
        for line in open(f):
            try:
                ev.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    ev.sort(key=lambda d: d.get("t", 0))
    return ev


def serving_sub(d, marks):
    r, t = d.get("rank"), d.get("t", 0)
    arm, disarm = marks.get((r, "armed")), marks.get((r, "disarmed"))
    if arm is None or t < arm:
        return "serving_before_arm"
    if disarm is None or t < disarm:
        return "fixture"
    return "after_disarm"


def analyse(run_dir):
    ev = load(run_dir)
    marks = {}
    for d in ev:
        if d.get("kind") in ("armed", "disarmed"):
            marks.setdefault((d.get("rank"), d["kind"]), d.get("t"))
    rs = [d for d in ev if d.get("kind") == "ws_resize"]
    tab = collections.Counter()
    events = []
    for d in rs:
        ph = d.get("phase") or ("graph_capture" if d.get("in_capture_model") else
                                "post_capture_warmup" if d.get("in_warmup_kernels") else
                                "serving" if d.get("after_capture") else "pre_capture")
        sub = serving_sub(d, marks) if ph == "serving" else ph
        cls = d.get("classification") or "unclassified"
        if ph == "pre_capture":
            cls = "pre-capture (no captured graph yet)"
        tab[(d.get("rank"), sub, cls)] += 1
        if ph != "pre_capture":
            events.append(dict(rank=d.get("rank"), phase=sub, t=d.get("t"), caller=d.get("caller"),
                               before=d.get("before"), after=d.get("after"), required=d.get("required"),
                               mode=d.get("mode"), num_tokens=d.get("num_tokens"), classification=cls,
                               n_invalidated=d.get("n_invalidated"),
                               invalidated_bindings=(d.get("invalidated_bindings") or [])[:8]))
    caps = {}
    for d in ev:
        if d.get("kind") == "capture_model_end":
            b = d.get("capture_bindings") or []
            by = collections.Counter((x.get("mode"), x.get("num_tokens")) for x in b)
            caps[d.get("rank")] = dict(bindings=len(b), distinct_ptrs=len({x.get("ptr") for x in b}),
                                       by_mode_and_batch={f"{k[0]}:{k[1]}": v for k, v in sorted(by.items(), key=str)},
                                       callers=sorted({x.get("caller") for x in b})[:20], ws_at_end=d.get("ws"))
    srv = sorted(glob.glob(os.path.join(run_dir, "**", "server.log"), recursive=True))
    dbg = []
    if srv:
        for line in open(srv[0], errors="replace"):
            if "[WORKSPACE DEBUG]" in line or "Graph capturing finished" in line:
                dbg.append(line.strip()[:300])
    inval = [e for e in events if e["classification"] == "captured-graph address invalidated"]
    if inval:
        overall = "exercised: captured-graph address invalidated"
    elif events:
        overall = "exercised: resize, no captured-graph address invalidated"
    else:
        overall = "not exercised in this workload"
    observer = any(d.get("kind") == "installed" and "workspace" in d.get("installed", []) for d in ev)
    return dict(
        observer_present=observer, capture_model_seen=bool(caps),
        resizes_by_rank_phase_class={f"r{k[0]}:{k[1]}:{k[2]}": v for k, v in sorted(tab.items(), key=str)},
        events_after_profiling=events[:200], capture_bindings=caps,
        debug_log_lines=len(dbg), debug_log_head=dbg[:40], caveat=CAVEAT,
        classification=dict(
            confirmed_omission="from source (V2: no lock_workspace; warmup_kernels after capture_model)",
            exercised=overall if observer and caps else "invalid: observer or capture markers missing",
            demonstrated_computational_effect="not measured (needs a replay touching reused freed memory)",
            verified_correction="not run", completion_effect="untested"))


def selftest():
    tmp = tempfile.mkdtemp(prefix="d5test_")
    base = [
        dict(kind="installed", installed=["workspace"], rank=0, t=1),
        dict(kind="ws_resize", rank=0, t=2, phase="pre_capture", classification="resize, no captured-graph address invalidated"),
        dict(kind="capture_model_begin", rank=0, t=3),
        dict(kind="capture_model_end", rank=0, t=5, ws={"state": [[1, 256]]},
             capture_bindings=[dict(ptr=1, nbytes=128, mode="FULL", num_tokens=8, caller="b12x_mla_sparse.py:2156:x")]),
    ]
    later = [
        dict(kind="ws_resize", rank=0, t=6, phase="post_capture_warmup",
             classification="captured-graph address invalidated", n_invalidated=1, caller="c.py:3:h"),
        dict(kind="armed", rank=0, t=8),
        dict(kind="ws_resize", rank=0, t=9, phase="serving", classification="resize, no captured-graph address invalidated"),
        dict(kind="disarmed", rank=0, t=10),
    ]

    def run(rows):
        with open(os.path.join(tmp, "v13_p1.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return analyse(tmp)
    with open(os.path.join(tmp, "server.log"), "w") as f:
        f.write("x [WORKSPACE DEBUG] Resized workspace from 'c.py:3:h': 0.00 MB -> 1.00 MB (ubatch 0)\n")
    r = run(base + later)
    assert r["classification"]["exercised"] == "exercised: captured-graph address invalidated", r["classification"]
    ph = r["resizes_by_rank_phase_class"]
    assert ph["r0:fixture:resize, no captured-graph address invalidated"] == 1
    assert ph["r0:post_capture_warmup:captured-graph address invalidated"] == 1
    assert ph["r0:pre_capture:pre-capture (no captured graph yet)"] == 1
    assert r["capture_bindings"][0]["by_mode_and_batch"] == {"FULL:8": 1} and r["debug_log_lines"] == 1
    r = run(base + [later[1], later[2], later[3]])
    assert r["classification"]["exercised"] == "exercised: resize, no captured-graph address invalidated"
    r = run(base)
    assert r["classification"]["exercised"] == "not exercised in this workload"
    r = run(base[:2])
    assert r["classification"]["exercised"].startswith("invalid")
    print("PASS analyze_d5 selftest")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    print(json.dumps(analyse(sys.argv[1]), indent=1, default=str))
