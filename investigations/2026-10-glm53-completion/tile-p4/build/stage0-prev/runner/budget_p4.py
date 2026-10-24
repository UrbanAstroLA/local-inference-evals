#!/usr/bin/env python3
"""P4 budget, order and attempt gate shared by every GPU step (AUTHORIZATION.txt: 4.0 occupied GPU-hours in total,
retries and single-GPU checks included).  Host python3, stdlib only.

usage: budget_p4.py ROOT STEP          STEP: precheck | g4 | A | B | A2
prints  "OK <used_gh> <left_gh> <cap_s> <attempt> <gpus>"  or  "REFUSE <reason>"

Ledger: ROOT/costs.jsonl, one line per GPU occupation (and one zero line per refused attempt), keys item
("p4-<step>"), attempt, seconds, gpus (gpu_hours as fallback).  Used = sum(seconds x gpus) / 3600 over ALL lines.

Order of GPU steps (fixed): precheck (1 GPU) -> g4 (1 GPU) -> A -> B -> A2 (2 GPUs each).
  * precheck: refused once a precheck attempt has passed (runs/precheck-attempt-N/verdict.json "PASS").
  * g4: needs a passed precheck; refused once G4 passed (runs/g4-attempt-N/verdict.json G4 "PASS").
  * A: needs precheck PASS and G4 PASS.  B: needs an A attempt whose outcome.json has gates_ok and >= 1 batch.
    A2: the same for B.  A session with such an outcome is never re-run.
Caps (wall seconds of this attempt):
  precheck  min(300, left x 3600)                 g4  min(900, left x 3600)
  session   min(SHARE_S, (left - RESERVE) x 3600 / 2), RESERVE = EST_SESSION_GH x (sessions still to run after
            this one); refused below MIN_WALL_S (start-up alone took 722-874 s; below this no batch can run).
SHARE_S = 2350 s = 1.306 GPU-h = (4.0 - 0.08 single-GPU checks) / 3.  EST_SESSION_GH = 1.27 (runner/cost_p4.py, planning case).
"""
import glob
import json
import os
import sys

CEILING = 7.0   # raised from 4.0 by the operator (AUTHORIZATION.txt line 2): retries after a fix
SHARE_S = 2350
MIN_WALL_S = 1300
EST_SESSION_GH = 1.27
CHECK_CAP = {"precheck": 300, "g4": 900}
ORDER = ["A", "B", "A2"]


def ledger(root):
    used, attempts = 0.0, {}
    p = os.path.join(root, "costs.jsonl")
    if not os.path.exists(p):
        return 0.0, {}
    for n, line in enumerate(open(p), 1):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            raise SystemExit(f"REFUSE malformed ledger line {n}")
        if d.get("seconds") is not None:
            used += float(d["seconds"]) * float(d.get("gpus", 1)) / 3600.0
        else:
            used += float(d.get("gpu_hours") or 0.0)
        attempts.setdefault(d.get("item"), set()).add(d.get("attempt"))
    return used, attempts


def passed(root, step):
    for p in sorted(glob.glob(os.path.join(root, "runs", f"{step}-attempt-*", "verdict.json"))):
        try:
            v = json.load(open(p))
        except Exception:
            continue
        if step == "precheck" and v.get("verdict") == "PASS":
            return True
        if step == "g4" and v.get("G4") == "PASS":
            return True
    return False


def session_ok(root, s):
    for p in sorted(glob.glob(os.path.join(root, "runs", f"session-{s}-attempt-*", "outcome.json"))):
        try:
            o = json.load(open(p))
        except Exception:
            continue
        if o.get("gates_ok") and int(o.get("batches_done", 0)) >= 1:
            return True
    return False


def main():
    root, step = sys.argv[1], sys.argv[2]
    if step not in ("precheck", "g4", *ORDER):
        print("REFUSE unknown step"); return 0
    used, attempts = ledger(root)
    left = CEILING - used
    attempt = len(attempts.get(f"p4-{step}", set())) + 1
    if step in CHECK_CAP:
        if passed(root, step):
            print(f"REFUSE {step} already passed"); return 0
        if step == "g4" and not passed(root, "precheck"):
            print("REFUSE g4 needs a passed precheck first"); return 0
        cap = int(min(CHECK_CAP[step], max(0.0, left) * 3600.0))
        if cap < 60:
            print(f"REFUSE budget left {left:.4f} GPU-h"); return 0
        print(f"OK {used:.4f} {left:.4f} {cap} {attempt} 1"); return 0
    if not (passed(root, "precheck") and passed(root, "g4")):
        print("REFUSE sessions need a passed precheck and a passed G4 first"); return 0
    i = ORDER.index(step)
    for prev in ORDER[:i]:
        if not session_ok(root, prev):
            print(f"REFUSE session {step} needs session {prev} done first (order A, B, A2)"); return 0
    if session_ok(root, step):
        print(f"REFUSE session {step} already done"); return 0
    remaining_after = sum(1 for s in ORDER[i + 1:] if not session_ok(root, s))
    reserve = EST_SESSION_GH * remaining_after
    cap = int(min(SHARE_S, max(0.0, left - reserve) * 3600.0 / 2))
    if cap < MIN_WALL_S:
        print(f"REFUSE wall cap {cap}s < {MIN_WALL_S}s (left {left:.4f} GPU-h, reserve {reserve:.2f} for "
              f"{remaining_after} later session(s))"); return 0
    print(f"OK {used:.4f} {left:.4f} {cap} {attempt} 2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
