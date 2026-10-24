#!/usr/bin/env python3
"""Engine-path test of the capture-p0 construction (CPU container): the REAL forcewin port, p4_hooks and cap0_hooks
driven by hook-validation's toy engine (real Sampler, RequestState) exactly as capture-p0 drives the server -- keyed
requests with an EMPTY forced list only, never a forced request first.  This is the path the attempt-1 harness never
exercised: every Stage-0 test that saw a sched log had a forced request flush it.

Env: FORCEWIN=on GLM53_P4=on GLM53_P4_S0=on GLM53_CAP0=on TRITON_INTERPRET=1 (mounts /hv/tests, /t, /p4pc)
     EXPECT_DEFECT=1  run against the attempt-1 module: E1 must then show the defect (no sched / steps log)

  E1  warm (CW-, F = []) then capture (CAP-, F = []), each removed: the rank-0 sched and step logs are written by the
      removal flush and carry both requests' rows; no p4 error
  E2  a forced request afterwards (its own store flush also calls flush_steps): no step row is logged twice, the
      step-ring files do not overlap, no error
  E3  rank 1: removing a keyed request writes no sched / steps log (rank-0-only behaviour, as in Stage 0)
"""
import glob
import json
import os
import sys

os.environ.setdefault("GLM53_CAP0", "on")
sys.argv = [sys.argv[0], "engine"]
import test_stage0_cpu as S0T  # noqa: E402  (stage-0 toy harness: reset_s0, add_req, finish, logs)
from fwtest_common import check, summary  # noqa: E402

p4, fw, TF, T = S0T.p4, S0T.fw, S0T.TF, S0T.T
SamplingParams = S0T.SamplingParams
DEFECT = os.environ.get("EXPECT_DEFECT") == "1"


def files(kind, rank):
    return glob.glob(os.path.join(p4.LOG_DIR, f"{kind}_r{rank}_p*.jsonl"))


def main():
    import shutil
    shutil.rmtree(p4.OUT_DIR, ignore_errors=True)
    shutil.rmtree(fw.OUT_DIR, ignore_errors=True)
    check("E0 cap0_hooks imported (GLM53_CAP0=on)", p4._cap0 is not None, str(p4._CAP0_ERR))
    S0T.reset_s0()
    p4.S.serving = True
    e = TF.Eng()
    T.CUR["eng"] = e
    prompt = [5000 + (7 * i) % 900 for i in range(40)]
    for name, tag, seed in (("warm", "CW-r1-N", 72999), ("cap", "CAP-r1", 73000)):
        sp = SamplingParams(temperature=1.0, top_p=0.95, seed=seed, max_tokens=1,
                            extra_args={"forcewin_hooks": [], "local_fw_tag": tag})
        S0T.add_req(e, name, prompt, sp, 1)
        e.prefill([name])
        S0T.finish(e, name)
    steps = [s for r in S0T.logs("sched") for s in r.get("steps", [])]
    tags = {row[3] for s in steps for row in s["rows"]}
    wrote = bool(files("sched", 0)) and bool(files("steps", 0)) and {"CW-r1-N", "CAP-r1"} <= tags
    errs = [r for r in S0T.logs() if r.get("kind") == "p4_error"]
    if DEFECT:
        check("E1-neg the attempt-1 module leaves NO sched/steps log for non-forced keyed requests (defect reproduced)",
              not files("sched", 0) and not files("steps", 0), str(files("sched", 0)))
        return summary("test_engine_path_cpu[attempt-1 module]")
    check("E1 warm + capture without a forced request: rank-0 sched and step logs written at removal, rows for both",
          wrote, str(sorted(tags)))
    check("E1 no p4 error", not errs, str(errs[:2]))
    # E2: a forced request afterwards -> its store flush calls flush_steps again (double flush)
    Fd = TF.make_history(prompt, 24, 9)
    spd = SamplingParams(temperature=1.0, top_p=0.95, seed=73000, max_tokens=24,
                         extra_args={"forcewin_hooks": Fd, "local_fw_tag": "V1-RX-early-r1", "local_fw_draft": "list",
                                     "local_fw_sched": [3, 1, 2, 0]})
    S0T.add_req(e, "d", prompt, spd, 24)
    S0T.drive(e, ["d"], {"d": 24})
    S0T.finish(e, "d")
    steps = [s["step"] for r in S0T.logs("sched") for s in r.get("steps", [])]
    srows = [row[0] for r in S0T.logs("steps") for row in r.get("rows", [])]
    rings = sorted(tuple(int(x) for x in os.path.basename(p)[8:23].split("_")) for p in
                   glob.glob(os.path.join(p4.PRIV_DIR, "steprec_*_r0.npz")))
    overlap = any(rings[i][1] >= rings[i + 1][0] for i in range(len(rings) - 1))
    errs = [r for r in S0T.logs() if r.get("kind") == "p4_error"]
    check("E2 double flush (cap0 removal + forced store flush): no step logged twice in sched or steps",
          len(steps) == len(set(steps)) and len(srows) == len(set(srows)) and bool(steps), "%d/%d" % (len(steps), len(set(steps))))
    check("E2 step-ring files do not overlap; no p4 error", bool(rings) and not overlap and not errs, str(rings) + str(errs[:1]))
    # E3: rank 1
    p4.S.rank = 1
    sp = SamplingParams(temperature=1.0, top_p=0.95, seed=72999, max_tokens=1,
                        extra_args={"forcewin_hooks": [], "local_fw_tag": "CW-r2-N"})
    S0T.add_req(e, "w2", prompt, sp, 1)
    e.prefill(["w2"])
    S0T.finish(e, "w2")
    check("E3 rank 1: no sched / steps log written (rank-0-only, as in Stage 0)", not files("sched", 1) and not files("steps", 1))
    p4.S.rank = 0
    return summary("test_engine_path_cpu")


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
