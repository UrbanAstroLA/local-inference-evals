#!/usr/bin/env python3
"""Stage-0 budget, order and attempt gate (STAGE0-PROTOCOL.md section 7).  Host python3, stdlib only.

usage: budget_stage0.py ROOT STEP          STEP: precheck | stage0
env:   APPROVED_GPU_H  REQUIRED.  The occupied GPU-hours the operator approved for stage 0 (the new approval that
                       replaces the VOID 7.0 ceiling, AUTHORIZATION.txt 2026-10-24).  There is no default and no
                       fallback: unset, empty, non-numeric or <= 0 -> REFUSE.
prints  "OK <used_gh> <left_gh> <cap_s> <attempt> <gpus>"  or  "REFUSE <reason>"

Ledger: ROOT/costs-stage0.jsonl ONLY (the new approval's own ledger).  Run-1 costs (ROOT/costs.jsonl) are neither read
nor written.  One line per GPU occupation and one zero line per refused-by-pre-flight attempt; used =
sum(seconds x gpus) / 3600 over all lines.

Rule (coordinator, 2026-10-24): REFUSE when used + this attempt's cap (cap_s x gpus / 3600) > APPROVED_GPU_H.  The
cap is never shrunk to fit: an attempt runs with its full cap or not at all.

Order: precheck (1 GPU, cap PRECHECK_CAP_S) -> stage0 (2 GPUs, cap STAGE0_CAP_S).
  * precheck: refused once a stage-0 pre-check attempt passed (runs-stage0/precheck-attempt-N/verdict.json PASS).
  * stage0: needs a passed stage-0 pre-check; refused once a stage-0 attempt reached the analysis
    (runs-stage0/stage0-attempt-N/outcome.json "client_complete": true) -- its result needs reading before more GPU.
"""
import glob
import json
import os
import sys

PRECHECK_CAP_S = 300
STAGE0_CAP_S = 2250          # runner/stage0/cost_stage0.py: planning 2,022 s, typical 1,640 s (cap +11 % / +37 %)
GPUS = {"precheck": 1, "stage0": 2}
LEDGER = "costs-stage0.jsonl"
RUNS = "runs-stage0"


def approved():
    raw = os.environ.get("APPROVED_GPU_H", "")
    if raw.strip() == "":
        return None, "APPROVED_GPU_H is not set: stage 0 needs an explicit operator approval in GPU-hours (no default)"
    try:
        v = float(raw)
    except ValueError:
        return None, f"APPROVED_GPU_H={raw!r} is not a number"
    if not v > 0 or v != v or v == float("inf"):
        return None, f"APPROVED_GPU_H={raw!r} must be a positive finite number"
    return v, None


def ledger(root):
    used, attempts = 0.0, {}
    p = os.path.join(root, LEDGER)
    if not os.path.exists(p):
        return 0.0, {}
    for n, line in enumerate(open(p), 1):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            raise SystemExit(f"REFUSE malformed ledger line {n} in {LEDGER}")
        if d.get("seconds") is not None:
            used += float(d["seconds"]) * float(d.get("gpus", 1)) / 3600.0
        else:
            used += float(d.get("gpu_hours") or 0.0)
        attempts.setdefault(d.get("item"), set()).add(d.get("attempt"))
    return used, attempts


def precheck_passed(root):
    for p in sorted(glob.glob(os.path.join(root, RUNS, "precheck-attempt-*", "verdict.json"))):
        try:
            if json.load(open(p)).get("verdict") == "PASS":
                return True
        except Exception:
            continue
    return False


def stage0_done(root):
    for p in sorted(glob.glob(os.path.join(root, RUNS, "stage0-attempt-*", "outcome.json"))):
        try:
            if json.load(open(p)).get("client_complete"):
                return True
        except Exception:
            continue
    return False


def main():
    if len(sys.argv) != 3:
        print("REFUSE usage: budget_stage0.py ROOT precheck|stage0"); return 0
    root, step = sys.argv[1], sys.argv[2]
    if step not in GPUS:
        print("REFUSE unknown step"); return 0
    appr, why = approved()
    if appr is None:
        print("REFUSE " + why); return 0
    used, attempts = ledger(root)
    attempt = len(attempts.get(f"s0-{step}", set())) + 1
    if step == "precheck":
        if precheck_passed(root):
            print("REFUSE stage-0 pre-check already passed"); return 0
        cap = PRECHECK_CAP_S
    else:
        if not precheck_passed(root):
            print("REFUSE stage 0 needs a passed stage-0 pre-check first (runner/stage0/run_precheck_s0.sh)"); return 0
        if stage0_done(root):
            print("REFUSE a stage-0 attempt already completed its client; read its result before any further GPU use"); return 0
        cap = STAGE0_CAP_S
    cap_gh = cap * GPUS[step] / 3600.0
    if used + cap_gh > appr + 1e-9:
        print(f"REFUSE cap {cap}s x {GPUS[step]} GPU = {cap_gh:.4f} GPU-h + used {used:.4f} > approved {appr:.4f} GPU-h "
              f"(ledger {LEDGER})"); return 0
    print(f"OK {used:.4f} {appr - used:.4f} {cap} {attempt} {GPUS[step]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
