#!/usr/bin/env python3
"""Budget / order gate tests for runner/budget_cap0.py (host python3; scratch under tests/out only)."""
import json
import os
import shutil
import subprocess
import sys

T = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(T)
B = os.path.join(PKG, "runner", "budget_cap0.py")
S = os.path.join(T, "out", "gates-scratch")
N = {"PASS": 0, "FAIL": 0}


def check(name, ok, detail=""):
    N["PASS" if ok else "FAIL"] += 1
    print(("PASS " if ok else "FAIL ") + name + ("  " + str(detail) if detail and not ok else ""))


IMG = "sha256:" + "a" * 64


def gate(step, appr, image=IMG, **extra):
    env = dict(os.environ)
    env.pop("APPROVED_GPU_H", None)
    env.pop("CAP0_PRIOR_ACK", None)
    env.update(extra)
    if appr is not None:
        env["APPROVED_GPU_H"] = appr
    return subprocess.run([sys.executable, B, S, step, image], env=env, capture_output=True, text=True).stdout.strip()


shutil.rmtree(S, ignore_errors=True)
os.makedirs(S)
check("B1 APPROVED_GPU_H unset -> REFUSE (no default)", gate("precheck", None).startswith("REFUSE APPROVED_GPU_H is not set"))
for v in ("", "abc", "0", "-1", "nan", "inf"):
    check(f"B1 APPROVED_GPU_H={v!r} -> REFUSE", gate("precheck", v).startswith("REFUSE"))
check("B2 session before a passed pre-check -> REFUSE", "needs a capture-p0 pre-check passed for image" in gate("session", "5"))
out = gate("precheck", "0.04").split()
check("B3 pre-check cap 180 s = 0.05 GPU-h > 0.04 approved -> REFUSE (cap never shrunk)", out[0] == "REFUSE")
out = gate("precheck", "0.661").split()
check("B3 pre-check OK: cap 180 s, attempt 1, 1 GPU", out[:1] == ["OK"] and out[3:] == ["180", "1", "1"], out)
os.makedirs(os.path.join(S, "runs", "precheck-attempt-1"))
json.dump(dict(verdict="PASS"), open(os.path.join(S, "runs", "precheck-attempt-1", "verdict.json"), "w"))
open(os.path.join(S, "costs-cap0.jsonl"), "w").write(json.dumps(dict(item="cap0-precheck", attempt=1, seconds=30, gpus=1,
                                                                     image=IMG)) + "\n")
check("B4 pre-check not re-run once passed for this image (image from the ledger line)", "already passed" in gate("precheck", "5"))
check("B4 a new image needs its own pre-check", gate("precheck", "5", image="sha256:" + "b" * 64).startswith("OK"))
check("B4 the session refuses a pre-check of another image", "pre-check passed for image" in gate("session", "5", image="sha256:" + "b" * 64))
check("B4 IMAGE argument required", gate("session", "5", image="none").startswith("REFUSE usage"))
out = gate("session", "0.61").split()
check("B5 used 0.0083 + session cap 0.6111 > 0.61 -> REFUSE", out[0] == "REFUSE", out)
out = gate("session", "0.661").split()
check("B5 approval 0.661 covers both caps: OK, cap 1100 s, 2 GPUs", out[0] == "OK" and out[3] == "1100" and out[5] == "2", out)
open(os.path.join(S, "costs-cap0.jsonl"), "a").write("not json\n")
check("B6 malformed ledger line -> REFUSE", gate("session", "5").startswith("REFUSE malformed"))
open(os.path.join(S, "costs-cap0.jsonl"), "w").write(json.dumps(dict(item="cap0-precheck", attempt=1, seconds=30, gpus=1,
                                                                     image=IMG)) + "\n")
other = os.path.join(S, "costs-stage0.jsonl")
open(other, "w").write(json.dumps(dict(item="s0-stage0", attempt=1, seconds=99999, gpus=2)) + "\n")
check("B7 other ledgers (costs-stage0.jsonl) are not read", gate("session", "0.661").startswith("OK"))
os.makedirs(os.path.join(S, "runs", "session-attempt-1"))
json.dump(dict(client_complete=True), open(os.path.join(S, "runs", "session-attempt-1", "outcome.json"), "w"))
check("B8 completed attempt without an analysis result: refused", "not INCOMPLETE" in gate("session", "5"))
res = os.path.join(S, "runs", "session-attempt-1", "cap0-result.json")
json.dump(dict(cls="incomplete"), open(res, "w"))
import hashlib  # noqa: E402
ack = hashlib.sha256(open(res, "rb").read()).hexdigest()[:12]
check("B8 INCOMPLETE prior attempt: refused without CAP0_PRIOR_ACK", "needs CAP0_PRIOR_ACK=" + ack in gate("session", "5"))
check("B8 INCOMPLETE prior attempt: refused with a wrong ack", "needs CAP0_PRIOR_ACK" in gate("session", "5", CAP0_PRIOR_ACK="0" * 12))
out = gate("session", "5", CAP0_PRIOR_ACK=ack).split()
check("B8 INCOMPLETE prior attempt + its ack: OK, attempt 2", out[0] == "OK" and out[4] == "1", out)
json.dump(dict(cls="located"), open(res, "w"))
ack2 = hashlib.sha256(open(res, "rb").read()).hexdigest()[:12]
check("B8 a valid prior reading is never re-run, even acknowledged", "not INCOMPLETE" in gate("session", "5", CAP0_PRIOR_ACK=ack2))
print("gates tests: %d PASS, %d FAIL" % (N["PASS"], N["FAIL"]))
sys.exit(1 if N["FAIL"] else 0)
