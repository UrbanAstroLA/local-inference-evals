#!/usr/bin/env python3
"""capture-p0 budget, order and attempt gate (PROTOCOL.md section 7).  Host python3, stdlib only.

usage: budget_cap0.py PKG STEP IMAGE    STEP: precheck | session      (PKG = the capture-p0 directory, or a mock root;
                                        IMAGE = the pinned image id, runner/images.txt)
env:   APPROVED_GPU_H  REQUIRED: the occupied GPU-hours the operator approved for capture-p0.  No default and no
                       fallback: unset, empty, non-numeric, NaN, infinite or <= 0 -> REFUSE.
prints "OK <used_gh> <left_gh> <cap_s> <attempt> <gpus>" or "REFUSE <reason>"

Ledger: PKG/costs-cap0.jsonl ONLY (this approval's own ledger; the Stage-0 and run-1 ledgers are neither read nor
written).  used = sum(seconds x gpus) / 3600 over its lines.  An attempt is refused when used + its FULL cap
(cap_s x gpus / 3600) exceeds APPROVED_GPU_H: the cap is never shrunk to fit.

Order: precheck (1 GPU, cap PRECHECK_CAP_S) -> session (2 GPUs, cap SESSION_CAP_S).
  A pre-check counts only for the image it ran: verdict.json "image", else the ledger line of that attempt.
  precheck: refused once a pre-check attempt passed FOR IMAGE (a new image needs its own pre-check).
  session:  needs a pre-check passed for IMAGE.  Once a session attempt's client completed (outcome.json
            "client_complete": true), a further session is refused unless (a) that attempt's analysis read
            INCOMPLETE (cap0-result.json "cls": "incomplete") and (b) env CAP0_PRIOR_ACK equals the first 12 hex of
            sha256(that cap0-result.json): the operator has read the failed result.  A valid reading is never re-run.
"""
import glob
import json
import os
import sys

PRECHECK_CAP_S = 180          # Stage-0 pre-check took 14 s; this one adds a 2,045-token hashing step (~1-3 s)
SESSION_CAP_S = 1100          # cost_cap0.json: planning 1,043 s, typical 830 s; readiness alone may take 874 s
GPUS = {"precheck": 1, "session": 2}
LEDGER = "costs-cap0.jsonl"
RUNS = "runs"


def approved():
    raw = os.environ.get("APPROVED_GPU_H", "")
    if raw.strip() == "":
        return None, "APPROVED_GPU_H is not set: capture-p0 needs an explicit operator approval in GPU-hours (no default)"
    try:
        v = float(raw)
    except ValueError:
        return None, f"APPROVED_GPU_H={raw!r} is not a number"
    if not v > 0 or v != v or v == float("inf"):
        return None, f"APPROVED_GPU_H={raw!r} must be a positive finite number"
    return v, None


def ledger(pkg):
    used, attempts = 0.0, {}
    p = os.path.join(pkg, LEDGER)
    if not os.path.exists(p):
        return 0.0, {}
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
        attempts.setdefault(d.get("item"), set()).add(d.get("attempt"))
    return used, attempts


def _sha12(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def precheck_passed(pkg, image):
    led = {}
    p = os.path.join(pkg, LEDGER)
    if os.path.exists(p):
        for line in open(p):
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if d.get("item") == "cap0-precheck":
                led[d.get("attempt")] = d.get("image")
    for v in sorted(glob.glob(os.path.join(pkg, RUNS, "precheck-attempt-*", "verdict.json"))):
        try:
            d = json.load(open(v))
            att = int(os.path.basename(os.path.dirname(v)).rsplit("-", 1)[1])
        except Exception:  # noqa: BLE001
            continue
        if d.get("verdict") == "PASS" and (d.get("image") or led.get(att)) == image:
            return True
    return False


def prior_sessions(pkg):
    """[(attempt, outcome, result path)] of session attempts whose client completed, in attempt order."""
    out = []
    for o in glob.glob(os.path.join(pkg, RUNS, "session-attempt-*", "outcome.json")):
        try:
            d = json.load(open(o))
            att = int(os.path.basename(os.path.dirname(o)).rsplit("-", 1)[1])
        except Exception:  # noqa: BLE001
            continue
        if d.get("client_complete"):
            out.append((att, d, os.path.join(os.path.dirname(o), "cap0-result.json")))
    return sorted(out, key=lambda x: x[0])


def main():
    if len(sys.argv) != 4 or not sys.argv[3].startswith("sha256:"):
        print("REFUSE usage: budget_cap0.py PKG precheck|session IMAGE_ID")
        return 0
    pkg, step, image = sys.argv[1], sys.argv[2], sys.argv[3]
    if step not in GPUS:
        print("REFUSE unknown step")
        return 0
    appr, why = approved()
    if appr is None:
        print("REFUSE " + why)
        return 0
    try:
        used, attempts = ledger(pkg)
    except SystemExit as exc:
        print(str(exc))
        return 0
    attempt = len(attempts.get(f"cap0-{step}", set())) + 1
    if step == "precheck":
        if precheck_passed(pkg, image):
            print("REFUSE the capture-p0 pre-check already passed for this image")
            return 0
        cap = PRECHECK_CAP_S
    else:
        if not precheck_passed(pkg, image):
            print("REFUSE the session needs a capture-p0 pre-check passed for image %s first (runner/run_precheck_cap0.sh)"
                  % image[:19])
            return 0
        prior = prior_sessions(pkg)
        if prior:
            att, _o, res = prior[-1]
            try:
                cls = json.load(open(res)).get("cls")
            except Exception:  # noqa: BLE001
                cls = None
            if cls != "incomplete":
                print("REFUSE session attempt %d completed its client and its analysis read %r (not INCOMPLETE): "
                      "read its result; it is not re-run" % (att, cls))
                return 0
            want = _sha12(res)
            if os.environ.get("CAP0_PRIOR_ACK", "") != want:
                print("REFUSE session attempt %d read INCOMPLETE; a new attempt needs CAP0_PRIOR_ACK=%s (first 12 hex of "
                      "sha256 of runs/session-attempt-%d/cap0-result.json): the operator has read why it failed"
                      % (att, want, att))
                return 0
        cap = SESSION_CAP_S
    cap_gh = cap * GPUS[step] / 3600.0
    if used + cap_gh > appr + 1e-9:
        print(f"REFUSE cap {cap}s x {GPUS[step]} GPU = {cap_gh:.4f} GPU-h + used {used:.4f} > approved {appr:.4f} GPU-h "
              f"(ledger {LEDGER})")
        return 0
    print(f"OK {used:.4f} {appr - used:.4f} {cap} {attempt} {GPUS[step]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
