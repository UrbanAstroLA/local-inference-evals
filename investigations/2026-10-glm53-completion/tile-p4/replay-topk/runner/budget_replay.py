#!/usr/bin/env python3
"""replay-topk budget and attempt gate (PROTOCOL.md section 8).  Host python3, stdlib only.

usage: budget_replay.py PKG IMAGES        IMAGES: comma list from {v091, v070}, run in that order
env:   APPROVED_GPU_H  REQUIRED: the occupied GPU-hours the operator approved for replay-topk.  No default, no
                       fallback: unset, empty, non-numeric, NaN, infinite or <= 0 -> REFUSE.
       REPLAY_PRIOR_ACK  needed only after an attempt whose analysis read INCOMPLETE somewhere: the first 12 hex of
                       sha256(runs/attempt-N/replay-result.json) of the latest analysed attempt.
prints "OK <used_gh> <left_gh> <attempt> <cap_v091_s> <cap_v070_s>" or "REFUSE <reason>"

Ledger: PKG/costs-replay.jsonl only.  used = sum(seconds x gpus) / 3600.  Refused when used + the FULL caps of the
requested images (1 GPU each) exceed APPROVED_GPU_H; caps are never shrunk to fit.
Attempts: a new attempt is refused when the latest analysed attempt has no INCOMPLETE unit (a valid reading is not
re-run), or read INCOMPLETE and REPLAY_PRIOR_ACK does not match.  Attempts with no analysis (aborted) do not block.
"""
import glob
import hashlib
import json
import math
import os
import sys

CAP_S = {"v091": 400, "v070": 300}      # hard wall caps per image container (cost_replay.json)
LEDGER = "costs-replay.jsonl"
RUNS = "runs"


def approved():
    raw = os.environ.get("APPROVED_GPU_H", "")
    if raw.strip() == "":
        return None, "APPROVED_GPU_H is not set: replay-topk needs an explicit operator approval in GPU-hours (no default)"
    try:
        v = float(raw)
    except ValueError:
        return None, f"APPROVED_GPU_H={raw!r} is not a number"
    if math.isnan(v) or math.isinf(v) or v <= 0:
        return None, f"APPROVED_GPU_H={raw!r} must be a positive finite number"
    return v, None


def ledger(pkg):
    used, attempts = 0.0, set()
    p = os.path.join(pkg, LEDGER)
    if not os.path.exists(p):
        return 0.0, attempts
    for n, line in enumerate(open(p), 1):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            raise SystemExit(f"REFUSE malformed ledger line {n} in {LEDGER}")
        if d.get("seconds") is not None:
            used += float(d["seconds"]) * float(d.get("gpus", 1)) / 3600.0
        else:
            used += float(d.get("gpu_hours") or 0.0)
        attempts.add(d.get("attempt"))
    return used, attempts


def attempt_dirs(pkg):
    out = []
    for d in glob.glob(os.path.join(pkg, RUNS, "attempt-*")):
        try:
            out.append((int(d.rsplit("-", 1)[1]), d))
        except ValueError:
            continue
    return sorted(out)


def has_incomplete(res):
    for im in (res.get("images") or {}).values():
        units = list((im.get("r1") or {}).values()) + list((im.get("r2") or {}).values())
        if "r3" in im:
            units.append(im["r3"])
        if not units or any(u.get("reading") == "INCOMPLETE" for u in units):
            return True
    return not res.get("images")


def main():
    if len(sys.argv) != 3:
        print("REFUSE usage: budget_replay.py PKG v091,v070")
        return 0
    pkg, images = sys.argv[1], [x for x in sys.argv[2].split(",") if x]
    if not images or any(i not in CAP_S for i in images) or len(set(images)) != len(images):
        print(f"REFUSE images must be a comma list from {sorted(CAP_S)} without repeats")
        return 0
    appr, why = approved()
    if appr is None:
        print("REFUSE " + why)
        return 0
    try:
        used, _att = ledger(pkg)
    except SystemExit as exc:
        print(str(exc))
        return 0
    dirs = attempt_dirs(pkg)
    attempt = (dirs[-1][0] + 1) if dirs else 1
    analysed = [(n, d) for n, d in dirs if os.path.exists(os.path.join(d, "replay-result.json"))]
    if analysed:
        n, d = analysed[-1]
        rp = os.path.join(d, "replay-result.json")
        try:
            res = json.load(open(rp))
        except Exception:  # noqa: BLE001
            res = {}
        if not has_incomplete(res):
            print(f"REFUSE attempt {n} produced a complete reading (no INCOMPLETE unit); read it, it is not re-run")
            return 0
        want = hashlib.sha256(open(rp, "rb").read()).hexdigest()[:12]
        if os.environ.get("REPLAY_PRIOR_ACK", "") != want:
            print(f"REFUSE attempt {n} read INCOMPLETE somewhere; a new attempt needs REPLAY_PRIOR_ACK={want} (first 12 "
                  f"hex of sha256 of runs/attempt-{n}/replay-result.json): the operator has read why")
            return 0
    cap_gh = sum(CAP_S[i] for i in images) / 3600.0
    if used + cap_gh > appr + 1e-9:
        print(f"REFUSE caps {'+'.join(str(CAP_S[i]) for i in images)} s x 1 GPU = {cap_gh:.4f} GPU-h + used {used:.4f}"
              f" > approved {appr:.4f} GPU-h (ledger {LEDGER})")
        return 0
    print(f"OK {used:.4f} {appr - used:.4f} {attempt} {CAP_S['v091'] if 'v091' in images else 0} "
          f"{CAP_S['v070'] if 'v070' in images else 0}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
