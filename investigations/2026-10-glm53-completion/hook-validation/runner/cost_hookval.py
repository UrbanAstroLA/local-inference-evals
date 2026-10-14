#!/usr/bin/env python3
"""Cost arithmetic for the hook-validation GPU session (PLAN.md amendment 21 section 4) and a first price for the
section-5 contrast (priced again from the validation's measured throughput before it is proposed).

Planning rates are the probe proposal's (probe-proposal/cost/cost.out), from measured receipts:
  start-to-ready + warm-up gate 13.5 min (measured 11.6-12.3), teardown 1.5 min, prefill 3,000 tok/s,
  decode 45 tok/s per request at 8 in flight with served acceptance (mean accepted length 2.8 at x3).
Forced decode is priced per STEP: a verify step at 8 in flight costs 2.8 / 45 = 62 ms (served rate / served
tokens per step); a schedule committing c tokens per step needs W / c steps. Overheads: hook metrics and the
in-graph exposure counters x1.05; the shadow ledger's per-step D2H x1.15; snapshot steps x1.5 (one sync + dumps).
"""
import json

START, STOP = 13.5 * 60, 1.5 * 60
PREFILL = 3000.0
STEP = 2.8 / 45.0          # s per verify step at 8 in flight
GPUS = 2
TPS = {"natural": 1.8, "oracle": 4.0, "poison": 1.0, "reject:1": 1.0, "reject:2": 2.0, "reject:3": 3.0,
       "sched": 2.5}       # committed tokens per step; natural on foreign histories assumed 1.8 (served 2.8)


def decode_s(W, mode, over=1.05):
    return W / TPS[mode] * STEP * over


def validation():
    ph = []
    ph.append(("V0 readiness probes (4 greedy probes, unkeyed, compared with B1 run-b1-a1)", 30))
    # V1: 8 free-run histories (B1's own samples, exact ids, engine logprobs), then forced (natural, poison)
    ph.append(("V1 free-run 8 x 2,000 tok (served acceptance)", 2000 / 45.0 + 5))
    ph.append(("V1 forced replay of the free runs: natural + poison", decode_s(2000, "natural") +
               decode_s(2000, "poison")))
    # V2a schedule matrix on 8 exact traces, windows capped at 2,500 tokens (block-crossing windows 1,000)
    ph.append(("V2a schedule matrix: 7 schedules x 1 batch of 8 (longest window 2,500)",
               sum(decode_s(2500, m) for m in TPS)))
    # V2b all 24 exact traces, natural + oracle, 3 length-sorted batches each (longest 461 / 1,469 / 4,795)
    ph.append(("V2b 24 exact traces x {natural, oracle}", sum(decode_s(L, m) for L in (461, 1469, 4795)
                                                            for m in ("natural", "oracle"))))
    ph.append(("V2c prefill references: 32 histories (every position), ~80k tokens", 80000 / PREFILL + 10))
    ph.append(("V3 committed-state snapshots: 1 batch of 8 (mixed schedules, 2,500 tok) + 24 prefill refs at "
               "the frontiers (~70k tok)", decode_s(2500, "sched", 1.5) + 70000 / PREFILL))
    ph.append(("V4 D2 window injection: 8 traces injected + same batch B1 (2,000 tok, shadow on) + 1 in flight "
               "injected + index-K dumps + prefill rebuild (20k tok)",
               2 * decode_s(2000, "natural", 1.15) + 2000 / 45.0 + 20000 / PREFILL + 20))
    ph.append(("guard readouts / idle drains between phases (8 x 8 s)", 64))
    work = sum(s for _, s in ph)
    wall = START + work + STOP
    return ph, work, wall


def contrast():
    """First long-context contrast (section 5): one session, B1 x3 and D2-window x3 interleaved in pairs on one
    fixed batch of 8 (W = 8,192 forced + 1), two cache salts (pairs 1-2 share salt A; pair 3 salt B)."""
    prefix = 8 * 95_000             # frozen batch: prefixes 64k-114k, mean ~95k
    ph = [("warm prefixes, salt A", prefix / PREFILL), ("warm prefixes, salt B", prefix / PREFILL),
          ("6 forced windows (3 B1 + 3 D2-window), remainder chunks + 8,193 forced, shadow on",
           6 * (8 * 2200 / PREFILL + decode_s(8193, "natural", 1.15))),
          ("end-of-window index-K dumps + 2 prefill rebuilds (prefix + window, fresh salts)",
           2 * (8 * 103_000) / PREFILL + 30),
          ("drains / readouts", 60)]
    work = sum(s for _, s in ph)
    return ph, work, START + work + STOP


if __name__ == "__main__":
    out = {}
    for name, fn in (("validation", validation), ("contrast", contrast)):
        ph, work, wall = fn()
        gh = GPUS * wall / 3600
        ceil = round(1.3 * gh + 0.5, 2)
        print("== %s" % name)
        for n, s in ph:
            print("   %-100s %7.0f s" % (n, s))
        print("   work %.0f s + start %.0f + stop %.0f = wall %.0f s = %.2f h -> %.2f occupied GPU-h; "
              "ceiling (x1.3 + 0.5 invalid-start allowance) %.2f" % (work, START, STOP, wall, wall / 3600, gh, ceil))
        out[name] = dict(phases=[[n, round(s, 1)] for n, s in ph], work_s=round(work), wall_s=round(wall),
                         gpu_h=round(gh, 3), ceiling_gpu_h=ceil)
    json.dump(dict(rates=dict(start_s=START, stop_s=STOP, prefill_tok_s=PREFILL, step_s=round(STEP, 4),
                              tokens_per_step=TPS), **out), open(__file__.replace(".py", ".json"), "w"), indent=1)
