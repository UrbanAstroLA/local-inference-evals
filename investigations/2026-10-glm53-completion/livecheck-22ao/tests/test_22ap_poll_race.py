#!/usr/bin/env python3
"""CPU test for the 22ap guard fix (poll race). No GPU, --network none, TRITON_INTERPRET=1.

Arm C (run-1) failed because the guard's host copy buffers were allocated by whichever thread polled first. When the
engine thread's baseline poll (inside torch.inference_mode) came first, the buffers were inference tensors and every
later poll from the poller thread failed on copy_. The CPU rig has no CUDA, so _snapshot takes its clone branch; this
test replaces _snapshot with a copy of the CUDA branch (one-time buffer allocation, copy_, clone) using unpinned
buffers, which have the same inference-mode semantics, and replays arm C's order.

  R1  old allocation (torch.empty in the caller's mode): the replay produces guard errors (the test can see the bug)
  R2  fixed allocation (_alloc_pins): the same replay produces none, and the buffers are normal tensors
  R3  16 concurrent polls (half inside inference mode): no errors, every length record emitted exactly once
"""
import json
import os
import sys
import threading
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["GLM53_MAPGUARD_DIR"] = "/tmp/mg22ap"
RIG = "/rig"
sys.path.insert(0, RIG)
sys.path.insert(0, os.path.join(RIG, "..", "analysis"))

import torch  # noqa: E402
import test_22ao as t22  # noqa: E402  (the 22ao rig: emulated session driving the guard's hooks)

mg = t22.mg
assert hasattr(mg, "_alloc_pins") and hasattr(S := mg.S, "poll_lock"), "22ap guard module not mounted"
RESULTS = []


def result(name, ok, **kw):
    RESULTS.append(dict(test=name, ok=bool(ok), **kw))
    print(("PASS " if ok else "FAIL ") + name + "  " + json.dumps(kw, default=str)[:500], flush=True)


def cuda_like_snapshot(alloc, delay=0.0):
    def snap():
        dev = S.dev
        tensors = [dev.cnt, dev.rec, dev.nrec, dev.layer, dev.ring_step, dev.ring_p0,
                   dev.nslog, dev.slog, dev.nbsnap, dev.bsnap, dev.bkey,
                   dev.nlrec, dev.lrec, dev.lenhist, dev.nsahist]
        if S.side is None:
            S.side = "cpu-standin"
            S.pins = alloc(tensors)
        for p_, t in zip(S.pins, tensors):
            p_.copy_(t)
        if delay:
            time.sleep(delay)   # widen the window between reading the counters and advancing the emitted marks
        return [p_.clone() for p_ in S.pins]
    return snap


def old_alloc(tensors):
    return [torch.empty(t.shape, dtype=t.dtype) for t in tensors]


def new_alloc(tensors):
    return mg._alloc_pins(tensors, pin=False)


def reset_poll_state():
    S.side, S.pins = None, None
    S.emitted = S.slog_emitted = S.bsnap_emitted = S.lrec_emitted = 0


def replay_arm_c(alloc):
    """Baseline poll on this thread inside inference mode first, then three polls from another thread."""
    reset_poll_state()
    mg._snapshot = cuda_like_snapshot(alloc)
    e0 = S.errors
    with torch.inference_mode():
        mg.poll("baseline")
    th = threading.Thread(target=lambda: [mg.poll("periodic") for _ in range(3)])
    th.start(); th.join()
    return S.errors - e0, [p.is_inference() for p in S.pins]


def log_lrec_total(since_line):
    path = os.path.join(mg.OUT_DIR, f"mg_p{os.getpid()}.jsonl")
    for fh in getattr(S, "fh", {}).values():
        fh.flush()
    lines = open(path).read().splitlines()[since_line:]
    polls = [json.loads(x) for x in lines if '"kind": "poll"' in x]
    return sum(len(p["lrec"]) for p in polls), len(polls)


def main(out_path):
    orig = mg._snapshot
    t22.run("over", seed=3)            # a session with length records in the guard's buffers
    n_lrec = int(S.dev.nlrec[0])
    errs, inf = replay_arm_c(old_alloc)
    result("R1 old allocation reproduces the arm C failure", errs > 0 and all(inf), guard_errors=errs,
           inference_buffers=sum(inf))
    errs, inf = replay_arm_c(new_alloc)
    result("R2 fixed allocation: same replay, no guard errors, normal buffers", errs == 0 and not any(inf),
           guard_errors=errs, inference_buffers=sum(inf))
    reset_poll_state()
    mg._snapshot = cuda_like_snapshot(new_alloc, delay=0.01)
    path = os.path.join(mg.OUT_DIR, f"mg_p{os.getpid()}.jsonl")
    start = len(open(path).read().splitlines())
    e0 = S.errors

    def worker(i):
        if i % 2:
            with torch.inference_mode():
                mg.poll("concurrent")
        else:
            mg.poll("concurrent")
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    [t.start() for t in ths]; [t.join() for t in ths]
    total, npolls = log_lrec_total(start)
    result("R3 16 concurrent polls: no errors, each length record emitted once",
           S.errors == e0 and npolls == 16 and total == min(n_lrec, mg.LCAP) and n_lrec > 0,
           guard_errors=S.errors - e0, polls=npolls, lrec_emitted=total, n_lrec=n_lrec)
    mg._snapshot = orig
    ok_all = all(x["ok"] for x in RESULTS)
    json.dump(dict(results=RESULTS, ok=ok_all), open(out_path, "w"), indent=1)
    print("ALL PASS" if ok_all else "SOME FAILED", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main(sys.argv[1])
