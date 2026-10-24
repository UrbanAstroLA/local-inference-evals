#!/usr/bin/env python3
"""P4 GPU-time estimate from measured rates (host python3; reads only public files).

usage: cost_p4.py [ROOT]      prints a table and writes runner/cost_p4.json

Rates (receipts):
  start-up to ready      722-874 s (captured-p3b, livecheck-22ao, hook-validation gate.txt); planning 874 s
  prefill                planning 2,400 tok/s (proposal); B1 measured 2,746-2,773 tok/s on 8 x 70k-120k prompts
                         (first-contrast run-fc1 LW1 / warm-s1)
  forced decode          8 in flight, 512 positions per request per pass, 3.06 tokens per step under `sched`
                         (windows/out/sched-summary.json) -> about 168 steps per pass.  Planning 34 tok/s/request
                         (first-contrast C1: forced decode with forcewin on B1, the slowest measured) = 15.1 s per pass;
                         natural serving ~70 tok/s/request would give 7.3 s
  stop + copy-out        90 s (stop.sh + docker stop; copy-out happens after the GPUs are released)
Token totals per session (windows/out/window-manifest.json): warm prefixes once, both passes' uncached remainders.
"""
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
man = json.load(open(os.path.join(ROOT, "windows", "out", "window-manifest.json")))
sch = json.load(open(os.path.join(ROOT, "windows", "out", "sched-summary.json")))
warm = man["totals"]["warm_prefill_tokens_per_session"]
rem = man["totals"]["remainder_tokens_per_pass"]
tps = sch["mean_tokens_per_step"]
passes = 12            # 6 batches x 2 repeats
W = 512


def session(start, rate, decode_tok_s, inter=5.0, stop=90):
    pre = (warm + 2 * rem) / rate
    dec = passes * W / decode_tok_s
    return dict(start=start, prefill=round(pre), decode=round(dec), between_batches=round(inter * passes), stop=stop,
                total=round(start + pre + dec + inter * passes + stop))


rows = {"planning (worst measured start-up, 2,400 tok/s, 34 tok/s/req)": session(874, 2400, 34),
        "typical (740 s start-up, 2,750 tok/s, 34 tok/s/req)": session(740, 2750, 34),
        "optimistic (722 s, 2,770 tok/s, 70 tok/s/req)": session(722, 2770, 70)}
# literal capped "close" window (512 tokens before the 327,680 cap) instead of amendment A2 (before 131,072)
lit_extra = 8 * (327680 - 131072)
lit = dict(session(874, 2400, 34))
lit["prefill"] = round((warm + lit_extra + 2 * rem) / 2400)
lit["total"] = round(874 + lit["prefill"] + lit["decode"] + 5 * passes + 90)
rows["literal capped window at the 327,680 cap (not adopted; A2)"] = lit
checks = dict(precheck=60, g4=180)
out = dict(tokens=dict(warm_per_session=warm, remainder_per_pass=rem, tokens_per_step=tps,
                       steps_per_pass=round(W / tps) + 1),
           sessions={k: dict(v, gpu_h=round(2 * v["total"] / 3600, 3)) for k, v in rows.items()},
           single_gpu_checks_s=checks, single_gpu_checks_gpu_h=round(sum(checks.values()) / 3600, 3))
plan = out["sessions"]["planning (worst measured start-up, 2,400 tok/s, 34 tok/s/req)"]
out["three_sessions_plus_checks_gpu_h"] = dict(
    planning=round(3 * plan["gpu_h"] + out["single_gpu_checks_gpu_h"], 3),
    typical=round(3 * out["sessions"]["typical (740 s start-up, 2,750 tok/s, 34 tok/s/req)"]["gpu_h"] +
                  out["single_gpu_checks_gpu_h"], 3))
out["caps"] = dict(session_share_s=2350, session_share_gpu_h=round(2 * 2350 / 3600, 3), ceiling_gpu_h=4.0,
                   min_wall_s=1300, precheck_cap_s=300, g4_cap_s=900)
json.dump(out, open(os.path.join(ROOT, "runner", "cost_p4.json"), "w"), indent=1)
print(f"tokens: warm {warm:,} per session; remainder {rem:,} per pass; {tps} tokens/step")
print(f"{'case':66s} {'start':>6s} {'prefill':>8s} {'decode':>7s} {'other':>6s} {'total s':>8s} {'GPU-h':>6s}")
for k, v in out["sessions"].items():
    print(f"{k:66s} {v['start']:6d} {v['prefill']:8d} {v['decode']:7d} {v['between_batches'] + v['stop']:6d} "
          f"{v['total']:8d} {v['gpu_h']:6.3f}")
print("single-GPU checks:", out["single_gpu_checks_gpu_h"], "GPU-h;  A + B + A' + checks:",
      out["three_sessions_plus_checks_gpu_h"], "(ceiling 4.0; session share cap 2350 s = 1.306 GPU-h)")
