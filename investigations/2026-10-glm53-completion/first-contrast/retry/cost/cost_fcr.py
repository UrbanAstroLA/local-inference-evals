#!/usr/bin/env python3
"""F9 [R-9]: measured-rate costing of the retry session, including expected misses, cold work, early-stop costs at
each gate, and the proposed ceiling. Host python (json only).

Inputs
  measured (run-fc1, diagnosis-fc1/cost.json): startup to ready 732 s; V0 6 s; client gap per phase 6.3 s (dump pause);
      cold prefill 2,770 tok/s (warm-s1), 2,707 / 2,655 (LW1 / LW2); hooked decode 36.6 tok/s per request at 8 in
      flight; C1 batch 300 s (one cold fixture inside); GATE-1 CPU analysis 54 s; end-of-session overhead to GPU
      release 19 s (1,955 s billed minus the phases)
  replay (replay/out/sim_<scenario>.json): every phase's scheduler steps and modelled seconds from a step-time model
      fitted to run-fc1's 5,608 scheduler steps (checked against the measured phases below)
Expected misses: the replay predicts none in 25 design scenarios (min survival margin >= 359 blocks); a miss costs the
fixture's cold in-batch prefill (<= 112k tokens at 2,770 tok/s = 40 s) and is priced in the pessimistic case (one miss
in every run batch).
usage: cost_fcr.py  -> cost_fcr.json, cost_fcr.out
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(os.path.dirname(HERE), "replay", "out")
GPUS = 2
M = dict(startup=732.0, v0=6.0, gap=6.3, prefill=2770.0, prefill_pess=2410.0, decode=36.6, decode_pess=25.0,
         gate_cpu_fc1=54.0, end_overhead=19.0, c1_measured=300.0, lw1_measured=298.0, lw2_measured=301.0,
         warm_measured=267.0)
GATE_CENTRAL, GATE_PESS = 270.0, 600.0     # GATE-1R: copy of stage A (~25-40 GB, ~100-250 s) + analysis (3 x fc1's)
MISS_COLD_S = 112024 / 2770.0              # the longest fixture fully prefilled inside a batch


def gh(s):
    return round(GPUS * s / 3600.0, 3)


def load(name):
    return json.load(open(os.path.join(SIM, "sim_%s.json" % name)))


def phase_factor(p):
    """pessimistic scale: prefill-bound phases by the prefill rate ratio, decode-bound by the decode ratio."""
    fp = M["prefill"] / M["prefill_pess"]
    fd = M["decode"] / M["decode_pess"]
    k = p["kind"]
    if k in ("build", "touch", "ps"):
        return fp
    if k == "run" and p.get("form") in ("C", "D"):
        return 0.08 * fp + 0.92 * fd
    if k == "run" and p.get("form") in ("LC", "LD"):
        return 0.45 * fp + 0.55 * fd
    if k == "run":
        return 0.5 * fp + 0.5 * fd
    return 1.0


def session(name):
    d = load(name)
    rows, t, tp = [], M["startup"], 800.0
    runs = 0
    for p in d["phases"]:
        if p["kind"] == "probe":
            s, sp = M["v0"] + M["gap"], M["v0"] + M["gap"]
        elif p["kind"] == "gate":
            s, sp = GATE_CENTRAL, GATE_PESS
        else:
            s = p["secs"]
            sp = (p["secs"] - M["gap"]) * phase_factor(p) + M["gap"]
            if p["kind"] == "run" and p.get("form") in ("C", "D", "LC", "LD"):
                runs += 1
                sp += MISS_COLD_S          # pessimistic: one cold fixture in every run batch
        t += s
        tp += sp
        rows.append(dict(phase=p["phase"], kind=p["kind"], central_s=round(s, 1), pess_s=round(sp, 1),
                         cum_central_gh=gh(t + M["end_overhead"]), cum_pess_gh=gh(tp + M["end_overhead"])))
    t += M["end_overhead"]
    tp += M["end_overhead"]
    return dict(rows=rows, central_s=round(t, 1), pess_s=round(tp, 1), central_gh=gh(t), pess_gh=gh(tp), runs=runs)


def early_stops(sess):
    """cumulative cost (central / pessimistic) when the session stops at each gate point."""
    by = {r["phase"]: r for r in sess["rows"]}

    def at(ph):
        r = by.get(ph)
        return (r["cum_central_gh"], r["cum_pess_gh"]) if r else None
    pts = [("pre-traffic stop (readiness, instrumentation, pool geometry: rc 5-8, 16)",
            (gh(M["startup"] + M["end_overhead"]), gh(800 + M["end_overhead"]))),
           ("V0 / references (rc 9 after LW1 or LW2)", at("LW2")),
           ("unsafe restoration or replay-model deviation at the first touch pass (rc 12)", at("T-s1")),
           ("insufficient concurrent decoding in the first batch (rc 13)", at("LC1") or at("C1")),
           ("structural stop after C1 (rc 9)", at("C1")),
           ("GATE-1R not PASS (rc 10): alignment, numerical content, sentinel or CV coverage", at("GATE-1R")),
           ("missing sentinel coverage after D2 (rc 14)", at("PS-c")),
           ("complete", (sess["central_gh"], sess["pess_gh"]))]
    return [dict(point=k, central_gh=v[0], pess_gh=v[1]) for k, v in pts if v]


def main():
    res = dict(measured=M, gate=dict(central=GATE_CENTRAL, pessimistic=GATE_PESS), miss_cold_s=round(MISS_COLD_S, 1))
    # calibration of the replay's step-time model against run-fc1's measured phases (fc1 replay = base start)
    base = load("base")
    cal = {}
    ph = {p["phase"]: p for p in base["phases"]}
    cal["LW1 (modelled incl. gap / measured + gap)"] = (ph["LW1"]["secs"], M["lw1_measured"] + M["gap"])
    cal["LW2"] = (ph["LW2"]["secs"], M["lw2_measured"] + M["gap"])
    cal["8-fixture build (warm-s1-a + warm-s1-b, 2 gaps) vs warm-s1"] = (
        round(ph["warm-s1-a"]["secs"] + ph["warm-s1-b"]["secs"], 1), M["warm_measured"] + 2 * M["gap"])
    cal["C1 (all hits) vs measured C1 (one cold fixture)"] = (ph["C1"]["secs"], M["c1_measured"] + M["gap"])
    res["calibration"] = cal
    out = {}
    for v in ("base", "A0"):
        s = session(v)
        s["early_stops"] = early_stops(s)
        s["ceiling_rule"] = "max(1.3 x central, pessimistic)"
        s["ceiling_gh"] = round(max(1.3 * s["central_gh"], s["pess_gh"]), 2)
        s["wall_cap_s"] = int(s["ceiling_gh"] * 3600 / GPUS)
        lp = sum(r["central_s"] for r in s["rows"] if r["phase"].startswith(("LC", "LD")))
        s["min_wall_s"] = int(s["central_s"] - lp)       # gates + the whole primary contrast (LP skippable)
        s["invalid_start_allowance_gh"] = gh(800 + M["end_overhead"])
        out["B" if v == "base" else "A0"] = s
    res["sessions"] = out
    # decode-rate sensitivity (central other inputs) for option B
    sens = {}
    for rate in (20.0, 25.0, 30.0, 36.6, 41.6):
        t = 0.0
        for p in base["phases"]:
            if p["kind"] == "run" and p.get("form") in ("C", "D"):
                t += (p["secs"] - M["gap"]) * (0.08 + 0.92 * M["decode"] / rate) + M["gap"]
            elif p["kind"] == "run" and p.get("form") in ("LC", "LD"):
                t += (p["secs"] - M["gap"]) * (0.45 + 0.55 * M["decode"] / rate) + M["gap"]
            elif p["kind"] == "gate":
                t += GATE_CENTRAL
            elif p["kind"] == "probe":
                t += M["v0"] + M["gap"]
            else:
                t += p["secs"]
        sens[str(rate)] = gh(t + M["startup"] + M["end_overhead"])
    res["decode_rate_sensitivity_B"] = sens
    json.dump(res, open(os.path.join(HERE, "cost_fcr.json"), "w"), indent=1)
    lines = ["calibration (replay-modelled s vs measured s): %s" % json.dumps(cal)]
    for k, s in out.items():
        lines.append("option %s: central %.0f s = %.2f GPU-h; pessimistic %.0f s = %.2f GPU-h; ceiling %.2f GPU-h "
                     "(max(1.3 x central, pessimistic)); wall cap %d s; min wall %d s; invalid-start allowance %.2f "
                     "GPU-h" % (k, s["central_s"], s["central_gh"], s["pess_s"], s["pess_gh"], s["ceiling_gh"],
                                s["wall_cap_s"], s["min_wall_s"], s["invalid_start_allowance_gh"]))
        for e in s["early_stops"]:
            lines.append("   stop: %-95s %.2f / %.2f GPU-h" % (e["point"], e["central_gh"], e["pess_gh"]))
    lines.append("decode-rate sensitivity, option B (tok/s per request -> GPU-h): %s" % sens)
    open(os.path.join(HERE, "cost_fcr.out"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
