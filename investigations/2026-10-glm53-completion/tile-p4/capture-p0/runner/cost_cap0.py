#!/usr/bin/env python3
"""capture-p0 GPU-time estimate (PROTOCOL.md section 7).  Host python3, stdlib only.  Writes runner/cost_cap0.json.

Inputs, all measured or recorded (no tuning to a target):
  start-up        874 s planning / 740 s typical (Stage-0 planning figures; the Stage-0 session itself was ready
                  after 719 s, runs-stage0/stage0-attempt-1/gate.txt)
  prefill rate    2,400 tok/s planning (Stage-0 planning) / 3,500 tok/s typical (Stage-0 measured: the 113,940-token
                  R01 warm took 32.2 s including HTTP, client.jsonl)
  localisation    278,928 B per token per rank hashed (the pre-check's K3 at the served row widths); host sha256 incl.
                  the reference-replicate hold measured at 1.28 GB/s in a CPU container (precheck dry run); planning
                  0.8 GB/s; device-to-host copy planning 4 GB/s / typical 10 GB/s (pageable); 13,056 tokens per warm
  after the client  Stage-0: about 7 s from the last request to the GPU release; planning 60 s / typical 20 s
  attempt 1 (measured, 13,057-token warms): ready 769 s; whole session 837 s; warm 3.6 s (no localisation) / 6.6 s
                  (localisation); capture request 2.5 s; hashing 0.48-0.50 s per 2,045-token step per rank
Workload of the session (2 GPUs): N replicates, each = warm 17,409 tokens + capture request 4,353 tokens prefill +
1 token + sentinel; the localisation replicates add the hashing.  Both ranks hash in parallel (separate processes).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import budget_cap0 as B  # noqa: E402

WARM, CAPP = 17409, 17409 - 13056          # amendment A1: the warm is the full capture prompt
LOC_TOKENS, LOC_BPT = 13056, 278928
CASES = {
    "planning": dict(start=874, rate=2400.0, req_overhead=1.5, capture_s=1.0, hash_gbps=0.8, d2h_gbps=4.0, flush_s=1.0,
                     startup_gates=15, after=60),
    "typical": dict(start=740, rate=3500.0, req_overhead=0.5, capture_s=0.3, hash_gbps=1.28, d2h_gbps=10.0, flush_s=0.5,
                    startup_gates=10, after=20),
}
REPS_LOC, REPS_NOLOC = 4, 2


def estimate():
    out = {}
    for name, c in CASES.items():
        core = WARM / c["rate"] + CAPP / c["rate"] + 3 * c["req_overhead"] + c["capture_s"]
        gb = LOC_TOKENS * LOC_BPT / 1e9
        loc = gb / c["hash_gbps"] + gb / c["d2h_gbps"] + c["flush_s"]
        fixed = c["start"] + c["startup_gates"] + c["after"]
        core4 = fixed + 4 * core
        full = fixed + (REPS_LOC + REPS_NOLOC) * core + REPS_LOC * loc
        out[name] = dict(start_s=c["start"], core_per_replicate_s=round(core, 1), loc_extra_per_replicate_s=round(loc, 1),
                         loc_gb_per_rank_per_replicate=round(gb, 2),
                         core_only_4rep_s=round(core4), core_only_4rep_gpu_h=round(2 * core4 / 3600, 3),
                         default_6rep_4loc_s=round(full), default_6rep_4loc_gpu_h=round(2 * full / 3600, 3),
                         loc_option_marginal_gpu_h=round(2 * (full - fixed - (REPS_LOC + REPS_NOLOC) * core) / 3600, 4))
    return out


def main():
    est = estimate()
    doc = dict(
        note="one serving session on 2 GPUs: start-up dominates; replicates cost seconds each",
        replicates=dict(default=["r1:L", "r2:L", "r3:N", "r4:N", "r5:L", "r6:L"], minimum_counted=2),
        warm_tokens=WARM, capture_prefill_tokens=CAPP, loc_tokens=LOC_TOKENS, loc_bytes_per_token_per_rank=LOC_BPT,
        cases=est,
        precheck=dict(cap_s=B.PRECHECK_CAP_S, cap_gpu_h=round(B.PRECHECK_CAP_S / 3600, 4), typical_s=30,
                      typical_gpu_h=round(30 / 3600, 4)),
        session_cap_s=B.SESSION_CAP_S, session_cap_gpu_h=round(2 * B.SESSION_CAP_S / 3600, 4),
        approval_needed_gpu_h=round(2 * B.SESSION_CAP_S / 3600 + B.PRECHECK_CAP_S / 3600, 3),
        approval_note=("APPROVED_GPU_H must cover the caps (one pre-check attempt + one session attempt): the gate "
                       "refuses an attempt whose full cap does not fit"),
        why_cap_above_0_6=("the session cap has to cover the slowest measured start-up (874 s) plus start-up gates, "
                           "the client and the stop: 1,100 s x 2 GPUs = 0.611 GPU-h; with the 180 s pre-check cap "
                           "the approval is 0.661.  A 0.60 total (session cap 990 s) would leave ~100 s after a "
                           "874 s start: replicates 5-6 would likely be dropped and a start-up slower than ~900 s "
                           "would spend the attempt with no capture"))
    json.dump(doc, open(os.path.join(HERE, "cost_cap0.json"), "w"), indent=1)
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
