#!/usr/bin/env python3
"""Stage-0 GPU-time estimate (STAGE0-PROTOCOL.md section 7).  Host python3, stdlib only.

usage: cost_stage0.py [WINDOWS_JSON] [BATCH]      -> prints JSON, writes runner/stage0/cost_stage0.json (counts only)

Workload of ONE session (2 GPUs), arm A, one batch of 8 windows, four independently warmed replicates in this order:
V1 rep 1, V1 rep 2, V2 rep 1, V2 rep 2.  Each replicate: its own cache salt; the batch's warm prefixes rebuilt from
scratch, one request at a time (the independent rebuild is what the assessor asked to recalculate: it multiplies the
warm prefill by 4 instead of 1 shared warm); the 8 forced requests (V1 one at a time, V2 one multi-prompt request);
a sentinel request that triggers the removal flush.  Entry digests cost device time at every prefill step and at the
entry step of each forced request (estimate per point; first measured in the session).  Per-layer digests run inside
every target-layer forward (estimated +10 % decode time).
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CHUNK = 2048
W = 512
TOK_PER_STEP = 3.0596                      # runner/cost_p4.json (the fixed acceptance list)
CASES = {
    # start-up (s), prefill tok/s, solitary forced decode tok/s/request, 8-request pass (s), entry-digest s/point
    "planning": dict(start=874, rate=2400.0, solo_tps=34.0, pass_s=20.0, dig_s=0.25),
    "typical": dict(start=740, rate=2750.0, solo_tps=70.0, pass_s=15.0, dig_s=0.10),
}
FIXED = dict(startup_gates=15, per_replicate_overhead=10, per_request_overhead=1.0, stop_copyout=90, ldig_decode=1.10)


def estimate(warm_tokens, rem):
    out = {}
    pts = sum(math.ceil(r / CHUNK) + 1 for r in rem)       # prefill chunks + the entry step, per forced request
    for name, c in CASES.items():
        warm = warm_tokens / c["rate"]
        remp = sum(rem) / c["rate"]
        dig = pts * c["dig_s"]
        v1_dec = len(rem) * (W / c["solo_tps"] * FIXED["ldig_decode"] + FIXED["per_request_overhead"])
        v2_dec = c["pass_s"] * FIXED["ldig_decode"]
        v1 = warm + remp + dig + v1_dec + FIXED["per_replicate_overhead"]
        v2 = warm + remp + dig + v2_dec + FIXED["per_replicate_overhead"]
        total = c["start"] + FIXED["startup_gates"] + 2 * v1 + 2 * v2 + FIXED["stop_copyout"]
        out[name] = dict(start=c["start"], v1_replicate_s=round(v1), v2_replicate_s=round(v2),
                         warm_prefill_per_replicate_s=round(warm), entry_digest_per_replicate_s=round(dig),
                         total_s=round(total), gpu_h=round(2 * total / 3600, 3))
    return out


def main():
    wj = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "PRIVATE-fixtures", "windows.json")
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    fx = json.load(open(wj))
    bp = next(b for b in fx["plan"] if int(b["batch"]) == batch)
    est = estimate(int(bp["warm_tokens"]), [int(x) for x in bp["remainder_tokens"]])
    sys.path.insert(0, HERE)
    import budget_stage0 as B
    doc = dict(batch=batch, windows=len(bp["windows"]), warm_tokens_per_replicate=int(bp["warm_tokens"]),
               remainder_tokens_per_replicate=sum(int(x) for x in bp["remainder_tokens"]), replicates=4,
               warm_tokens_session=4 * int(bp["warm_tokens"]),
               note_rebuild=("independent warm-prefix rebuilding per replicate: 4 x the batch's warm prefill (the "
                             "proposal's 0.6 GPU-h assumed one shared warm); this dominates the workload after start-up"),
               cases=est, cap_stage0_s=B.STAGE0_CAP_S, cap_stage0_gpu_h=round(2 * B.STAGE0_CAP_S / 3600, 4),
               cap_precheck_s=B.PRECHECK_CAP_S, cap_precheck_gpu_h=round(B.PRECHECK_CAP_S / 3600, 4),
               approval_needed_gpu_h=round(2 * B.STAGE0_CAP_S / 3600 + B.PRECHECK_CAP_S / 3600, 3),
               approval_note=("APPROVED_GPU_H must cover the caps (the gate refuses an attempt whose cap does not fit): "
                              "one pre-check attempt + one stage-0 attempt"))
    json.dump(doc, open(os.path.join(HERE, "cost_stage0.json"), "w"), indent=1)
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
