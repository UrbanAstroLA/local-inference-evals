#!/usr/bin/env python3
"""Occupied GPU-hour estimates and ceilings for the forced-decode window probe (PROPOSAL.md section 7).

CPU only; deterministic. Rates are planning values derived from measured numbers (sources below); token totals come
from ../fixtures/fixtures.json. Occupied GPU-hours = 2 GPUs x wall from server start to GPU release (the convention
of every earlier cost line), except Stage K (1 GPU x container wall).

Measured inputs (all from local receipts; none re-measured here):
  start -> ready      0.9.1 family 11.6-12.3 min (rep070/timing-reconstruction.md: 12.1, 11.8, 11.6, 12.3;
                      d7/step2 B1 guard session: server start 09:59:42Z, client start 12.2 min later);
                      stock 0.7.0 8.3-9.3 min (timing-reconstruction.md: 9.25, 8.30).
  teardown            GPUs released 66 s after client end (amendment 17d, a2); 0.4 min in the B1 guard session.
  prefill, 0.9.1 fam. 64,217 cold tokens in 17.5-19.0 s and 63,641 tokens at 56-120k in 17.8-19.4 s
                      (gate/server-a1, tailguard/graph-a1, bundle/gate-runs/s1-d1234-a1, phase2-stage2 client logs:
                      3,300-3,700 tok/s); guard-instrumented images 2,400-2,600 tok/s (b0-mapguard a3, d7/step2 b1-a1).
  prefill, 0.7.0      1,630-1,820 tok/s, 10-s server-log averages at <=16k (step4-long/v070-r1-server.log).
                      NOT measured at 64-128k: the planning value is conservative and is the largest cost uncertainty.
  decode, 8 in flight aggregate generation throughput at Running: 8 (server logs, medians by KV band):
                      stock 0.9.1 (+ps pass log) 429-482 tok/s; B1 (b1-screen server-s1.log) 439-513;
                      stock 0.7.0 (+ps pass log) 258-298.  Per request: 0.9.1 family 54-64, 0.7.0 32-37 tok/s.
                      Mean acceptance length in serving: 2.8 (K=3), 3.2-3.4 (K=5) (SpecDecoding metrics lines).
Planning rates add margin for forced acceptance (draft accepted iff it equals the history token) and the hook.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = json.load(open(os.path.join(HERE, "..", "fixtures", "fixtures.json")))
T = FIX["batch_totals"]

ENG = {
    # start->ready + warm-up gate, teardown (min); prefill tok/s; decode tok/s per request at 8 in flight
    "0.9.1-family": dict(start=12.5 + 1.0, stop=1.5, prefill=3000.0, decode=45.0, gpus=2),
    "0.7.0":        dict(start=9.5 + 1.0, stop=1.5, prefill=1200.0, decode=27.0, gpus=2),
}
PHASE_OVERHEAD_S = 20.0      # client synchronisation, health checks, dumps, per phase
REMAINDER = 8 * 2176         # same-salt repeat: uncached remainder of 8 prefixes (half a 4,352-token block each)
MARGIN = 1.30                # ceiling = estimate x MARGIN + one invalid-start allowance per stage
P0_S = 30.0                  # 4 short health prompts (as in every guard session)


def phases_stage1():
    L, S = T["L"], T["S"]
    return [
        ("P0 health", 0, 0, P0_S),
        ("S full replay + rebuild at close", S["prefix_tokens"] + S["rebuild_prompt_tokens"], S["forced_predictions_max"], 0),
        ("L run 1 (salt A) + index-K dumps + rebuild R1", L["prefix_tokens"] + L["rebuild_prompt_tokens"], L["forced_predictions_max"], 0),
        ("L run 2 (salt A, decode-only repeat)", REMAINDER, L["forced_predictions_max"], 0),
        ("L run 3 (salt B, prefill+decode repeat) + rebuild R2", L["prefix_tokens"] + L["rebuild_prompt_tokens"], L["forced_predictions_max"], 0),
    ]


def phases_stage1_min():
    """Cheaper alternative first look: harness validity and index-K bytes only, no repeats."""
    return phases_stage1()[:3]


def phases_stage2():
    L2, L3 = T["L2"], T["L3"]
    comp_prefix = 2 * 112000      # two batch-L targets (L3 and L6 histories, about 89-112k prefixes, rounded up)
    return [
        ("P0 health", 0, 0, P0_S),
        ("L2 run 1 (salt A) + rebuild", L2["prefix_tokens"] + L2["rebuild_prompt_tokens"], L2["forced_predictions_max"], 0),
        ("L2 run 2 (salt A)", REMAINDER, L2["forced_predictions_max"], 0),
        ("L3 run 1 (salt A) + rebuild", L3["prefix_tokens"] + L3["rebuild_prompt_tokens"], L3["forced_predictions_max"], 0),
        ("L3 run 2 (salt A)", REMAINDER, L3["forced_predictions_max"], 0),
        ("composition: 2 targets x (1 target + 7 sampled background)", comp_prefix, 2 * 8193, 0),
        ("L oracle-draft bracket (salt A; drafts replaced by history)", REMAINDER + T["L"]["prefix_tokens"], 8193, 0),
    ]


def session(eng, phases):
    e = ENG[eng]
    rows, tot = [], e["start"] * 60 + e["stop"] * 60
    for name, ptok, dtok, extra in phases:
        s = ptok / e["prefill"] + dtok / e["decode"] + extra + PHASE_OVERHEAD_S
        rows.append((name, ptok, dtok, round(s / 60, 1)))
        tot += s
    return rows, tot / 3600.0, e["gpus"]


def main():
    out = {"rates": ENG, "batch_totals": T, "stages": {}}
    print("Planning rates:", json.dumps(ENG))
    F = "0.9.1-family"
    plan = {
        "Stage 1": [("stock 0.7.0 x5", "0.7.0"), ("B0 x3", F), ("B0-minus-D2 x3", F), ("B0 x5 (draft-depth control)", F)],
        "Stage 1-min (alternative to Stage 1)": [("stock 0.7.0 x5", "0.7.0"), ("B0 x3", F), ("B0-minus-D2 x3", F),
                                                 ("B0 x5", F)],
        "Stage 1b (optional D6 diagnostic)": [("B0-ring4 x5 (spec-ring reverted on B0)", F)],
        "Stage 2": [("stock 0.7.0 x5", "0.7.0"), ("B0 x3", F), ("B0-minus-D2 x3", F)],
        "Stage 2c (context arm)": [("stock 0.9.1 x3, Stage 1 + Stage 2 workload in one session", "0.9.1-family+")],
    }
    grand = 0.0
    for stage, arms in plan.items():
        st_total, lines, allowance = 0.0, [], 0.0
        for label, eng in arms:
            if eng == "0.9.1-family+":
                rows, wall_h, g = session("0.9.1-family", phases_stage1() + phases_stage2()[1:])
                eng = "0.9.1-family"
            else:
                ph = (phases_stage1() if stage == "Stage 1" else
                      phases_stage1_min() if stage.startswith("Stage 1-min") else phases_stage2())
                if stage.startswith("Stage 1b"):
                    ph = phases_stage1()
                rows, wall_h, g = session(eng, ph)
            gh = wall_h * g
            st_total += gh
            allowance = max(allowance, g * (ENG[eng]["start"] + ENG[eng]["stop"]) / 60.0)
            lines.append(dict(arm=label, wall_h=round(wall_h, 3), gpu_h=round(gh, 3), phases=rows))
            print("\n%s / %s: wall %.2f h, %.2f occupied GPU-h" % (stage, label, wall_h, gh))
            for r in rows:
                print("   %-58s prefill %9d tok  decode %6d tok/request  %5.1f min" % r)
        ceiling = st_total * MARGIN + allowance
        out["stages"][stage] = dict(arms=lines, estimate_gpu_h=round(st_total, 2),
                                    ceiling_gpu_h=round(ceiling, 1), invalid_start_allowance=round(allowance, 2))
        if not stage.startswith("Stage 1-min"):
            grand += st_total
        print("%s: estimate %.2f GPU-h; ceiling %.1f (x%.2f + one invalid-start allowance %.2f)" % (
            stage, st_total, ceiling, MARGIN, allowance))
    # Stage K: model-free kernel replay, one GPU (precedent: phase2-stage2 replays 0.0155-0.0489 GPU-h each)
    k_est = 1 * (3.0 + 6 * 2.5) / 60.0   # container start + six route/length configurations x ~2.5 min
    out["stages"]["Stage K (model-free selection replay, 1 GPU)"] = dict(estimate_gpu_h=round(k_est, 2), ceiling_gpu_h=0.5)
    print("\nStage K: estimate %.2f GPU-h (1 GPU); ceiling 0.5" % k_est)
    # sensitivity: 0.7.0 prefill rate is unmeasured at 64-128k
    for pf in (900.0, 1600.0):
        keep = ENG["0.7.0"]["prefill"]
        ENG["0.7.0"]["prefill"] = pf
        _, w, g = session("0.7.0", phases_stage1())
        ENG["0.7.0"]["prefill"] = keep
        print("sensitivity: stock 0.7.0 Stage 1 session at prefill %.0f tok/s -> %.2f GPU-h" % (pf, w * g))
        out.setdefault("sensitivity", {})["0.7.0 Stage 1 at prefill %d" % pf] = round(w * g, 2)
    out["total_estimate_stage1_2_2c_K"] = round(grand + k_est, 2)
    print("\nTotal estimate, Stages 1 + 2 + 2c + K (Stage 1-min excluded): %.2f GPU-h" % (grand + k_est))
    json.dump(out, open(os.path.join(HERE, "cost.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
