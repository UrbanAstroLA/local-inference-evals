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
[R-11, PLAN.md amendment 22f s.3] acceptance stress scenarios (replay/out/sim_stress_*.json; stress scenarios, NOT
predictions of D2): the same costing applied to each (central = replay-modelled phase seconds with the stressed
acceptance; pessimistic = the same pessimistic transformation as above), compared with the variant-B ceiling, and the
client's own time rules (runner/client_fcr.py need_s / reserve_after, deadline = wall cap - RESERVE, client killed at
deadline + 15 s) replayed on that timeline under the B wall cap, to show what a session would actually complete.
The ceiling, wall cap and client rules are NOT changed here; this section only reports whether they still hold.
[R-16, PLAN.md amendment 22h s.1] the variant-B ceiling is the operator's decision, 6.2 occupied GPU-hours (wall cap
11,160 s on 2 GPUs). The computed max(1.3 x central, pessimistic) = 4.18 / 7,523 s is kept as superseded, and the R-11
section above is evaluated against those superseded values, as reported under 22f. A0 keeps its computed ceiling.
[R-18, 22h s.2-3] the batch guard (runner/batch_guard_fcr.py, the client's own code) replayed on the modelled timeline
of the base case and of every stress scenario, central and pessimistic, under the 6.2 wall cap: whether the session
completes, where the guard stops or skips (always before a phase), and that no phase would reach the watchdog.
usage: cost_fcr.py  -> cost_fcr.json, cost_fcr.out
"""
import glob
import json
import os
import sys

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


# ------------------------------------------------------------------ [R-11] acceptance stress scenarios
CLIENT = dict(PF_RATE=2410.0, DEC_RATE=25.0, DUMP_S=6.3, RESERVE=240, KILL_AFTER_DEADLINE=15)   # runner / client values


def need_s(ph):
    """verbatim rule of runner/client_fcr.need_s (pessimistic seconds for a phase, never-truncate rule)."""
    rs = ph["requests"]
    PF, DEC, DUMP = CLIENT["PF_RATE"], CLIENT["DEC_RATE"], CLIENT["DUMP_S"]
    if ph["kind"] == "build":
        return sum(r["P"] for r in rs) / PF + 8 * DUMP + 30
    if ph["kind"] in ("touch", "ps"):
        return sum(r["P"] - r["cached_expected"] for r in rs) / PF + 8 * 2.0 + 30
    if ph["kind"] == "run":
        return sum(r["P"] - r["cached_expected"] for r in rs) / PF + max(r["max_tokens"] for r in rs) / DEC + \
            8 * DUMP + 30
    return 60.0


def client_timeline(sess, plan, wall_cap, pess):
    """the client's deadline rules on a modelled timeline: LP batches skipped first when short of time (and every later
    LP batch with them), a primary phase that does not fit stops the session (rc 11), a phase admitted but running past
    deadline + 15 s is cut by the runner's timeout (killed mid-phase)."""
    is_lp = lambda p: p.get("form") in ("LC", "LD")  # noqa: E731
    dur = {r["phase"]: (r["pess_s"] if pess else r["central_s"]) for r in sess["rows"]}
    deadline = wall_cap - CLIENT["RESERVE"]
    t = 800.0 if pess else M["startup"]
    done, skipped, outcome, lp_off = [], [], "complete", False
    for i, ph in enumerate(plan):
        tag = ph["tag"]
        if ph["kind"] in ("probe", "gate"):
            t += dur[tag]
            done.append(tag)
            continue
        if is_lp(ph) and lp_off:
            skipped.append(tag)
            continue
        need = need_s(ph)
        res = sum(need_s(q) for q in plan[i + 1:] if q["kind"] not in ("gate", "probe") and not is_lp(q)) \
            if is_lp(ph) else 0.0
        if deadline - t < need + res:
            if is_lp(ph):
                skipped.append(tag)
                lp_off = True
                continue
            outcome = "stopped before %s (rc 11: does not fit)" % tag
            break
        if t + dur[tag] > deadline + CLIENT["KILL_AFTER_DEADLINE"]:
            outcome = "client killed DURING %s at the deadline (phase admitted by the 25 tok/s need rule, ran %.0f s " \
                      "against a need of %.0f s)" % (tag, dur[tag], need)
            t = deadline + CLIENT["KILL_AFTER_DEADLINE"]
            break
        t += dur[tag]
        done.append(tag)
    primary = ["C1", "C2", "C3", "D1", "D2", "D3"]
    end = min(t + M["end_overhead"], wall_cap)
    return dict(outcome=outcome, completed=done, skipped_lp=skipped,
                primary_complete=all(p in done for p in primary), gate_reached="GATE-1R" in done,
                lp_complete=all(p in done for p in ("LC1", "LC2", "LC3", "LD1", "LD2", "LD3")),
                gpu_h=gh(end), wall_s=round(end, 1))


def central_at_rate(base, rate):
    """option B central GPU-h with the decode-bound share of every C / D / LC / LD batch rescaled to `rate` tok/s per
    request (the decode-rate sensitivity formula below)."""
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
    return gh(t + M["startup"] + M["end_overhead"])


def breakeven_rate(base, ceiling):
    lo, hi = 1.0, M["decode"]
    for _ in range(60):
        mid = (lo + hi) / 2
        if central_at_rate(base, mid) > ceiling:
            lo = mid
        else:
            hi = mid
    return round(hi, 2)


# ------------------------------------------------------------------ [R-16 / R-18] ceiling 6.2 and the batch guard
CEILING_DECIDED = {"B": dict(ceiling_gh=6.2, rule="operator decision, PLAN.md amendment 22h (single attempt)")}
SUPERSEDED_B = dict(ceiling_gh=4.18, wall_cap_s=7523, rule="max(1.3 x central, pessimistic), amendments 22e-22g")


def guard_timeline(sess, plan, wall_cap, pess):
    """the client's batch guard (runner/batch_guard_fcr.py, unchanged code) on a modelled timeline. Each phase's
    modelled seconds include the 6.3 s dump gap, which the client's measured phase time excludes. The runner's
    watchdog stops the container at wall cap - 75 s: a started phase that would end later is a mid-phase cut."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "runner"))
    import batch_guard_fcr as bg
    dur = {r["phase"]: (r["pess_s"] if pess else r["central_s"]) for r in sess["rows"]}
    deadline, watchdog = wall_cap - CLIENT["RESERVE"], wall_cap - 75
    t = 800.0 if pess else M["startup"]
    rates = bg.Rates()
    done, skipped, outcome, lp_off, worst, over_deadline = [], [], "complete", False, 0.0, []
    for i, ph in enumerate(plan):
        tag = ph["tag"]
        if ph["kind"] in ("probe", "gate"):
            t += dur[tag]
            done.append(tag)
            continue
        g = bg.decide(rates, plan, i, deadline - t, lp_off)
        if g["action"] == "skip":
            skipped.append(tag)
            lp_off = True
            continue
        if g["action"] == "stop":
            outcome = "guard stop BEFORE %s (rc 17: projected %.0f s, %.0f s left)" % (tag, g["projected"], g["left"])
            break
        d = dur[tag]
        worst = max(worst, (d - M["gap"]) / max(g["projected"], 1e-9))
        if t + d > watchdog:
            outcome = "WATCHDOG CUT DURING %s (mid-phase)" % tag
            t = watchdog
            break
        if t + d > deadline:
            over_deadline.append(tag)
        t += d
        done.append(tag)
        rates.observe(ph, d - M["gap"])
    primary = ["C1", "C2", "C3", "D1", "D2", "D3"]
    end = min(t + M["end_overhead"], wall_cap)
    return dict(outcome=outcome, completed=done, skipped_lp=skipped, primary_complete=all(p in done for p in primary),
                lp_complete=all(p in done for p in ("LC1", "LC2", "LC3", "LD1", "LD2", "LD3")),
                mid_phase_cut=outcome.startswith("WATCHDOG"), ended_past_deadline=over_deadline,
                max_actual_over_projected=round(worst, 3), gpu_h=gh(end), wall_s=round(end, 1))


def geo_plan():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "analysis"))
    import fcr_geometry as geo
    return geo.phases(geo.load(os.path.dirname(os.path.dirname(HERE))), "B")


def stress(ceiling_b, wall_b):
    plan = geo_plan()
    out = {}
    for f in sorted(glob.glob(os.path.join(SIM, "sim_stress_*.json"))):
        name = os.path.basename(f)[4:-5]
        if name == "stress_measured_recheck":
            continue
        s = session(name)
        d = load(name)
        dec = [p for p in d["phases"] if p["kind"] == "run" and p.get("form") in ("C", "D")]
        rate = [8199.0 / max(p["secs"] - M["gap"], 1e-9) for p in dec]
        out[name] = dict(
            acceptance=d["scenario"].get("acceptance"), fast=d["scenario"].get("fast"),
            num_blocks=d["scenario"].get("num_blocks", 1275), order_policy=d["scenario"].get("order_policy",
                                                                                              "completion"),
            complete_central_gh=s["central_gh"], complete_pess_gh=s["pess_gh"],
            complete_central_s=s["central_s"], complete_pess_s=s["pess_s"],
            within_ceiling_central=s["central_gh"] <= ceiling_b, within_ceiling_pess=s["pess_gh"] <= ceiling_b,
            primary_batch_decode_tok_s=(round(min(rate), 1), round(max(rate), 1)),
            under_wall_cap_central=client_timeline(s, plan, wall_b, False),
            under_wall_cap_pess=client_timeline(s, plan, wall_b, True))
    return out


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
        vk = "B" if v == "base" else "A0"
        if vk in CEILING_DECIDED:                      # [R-16] 22h: the operator's ceiling; computed one superseded
            s["superseded"] = dict(SUPERSEDED_B, computed_now=s["ceiling_gh"])
            s["ceiling_rule"] = CEILING_DECIDED[vk]["rule"]
            s["ceiling_gh"] = CEILING_DECIDED[vk]["ceiling_gh"]
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
    # [R-11] acceptance stress scenarios: the B ceiling and wall cap are reported against, not changed
    sB = out["B"]
    c_old, w_old = SUPERSEDED_B["ceiling_gh"], SUPERSEDED_B["wall_cap_s"]     # R-11 as reported under 22f
    st = stress(c_old, w_old)
    base_tl = dict(central=client_timeline(sB, geo_plan(), w_old, False),
                   pess=client_timeline(sB, geo_plan(), w_old, True))
    res["stress_B"] = dict(note="acceptance stress scenarios (22f s.3): stress scenarios, not predictions of D2; "
                                "evaluated against the SUPERSEDED 4.18 ceiling, 7,523 s wall cap and the superseded "
                                "25 tok/s client rule, as reported under 22f (R-16 / R-18 below supersede them)",
                           ceiling_gh=c_old, wall_cap_s=w_old, client_rules=CLIENT,
                           base_under_wall_cap=base_tl, scenarios=st,
                           breakeven_decode_tok_s_central=breakeven_rate(base, c_old),
                           base_advance_tokens_per_step=2.84,
                           ceiling_holds_central=all(v["within_ceiling_central"] for v in st.values()),
                           ceiling_holds_pess=all(v["within_ceiling_pess"] for v in st.values()))
    # [R-18] 6.2 + batch guard: base and every stress scenario, central and pessimistic
    plan = geo_plan()
    gsc = {"base (measured acceptance)": sB}
    for f in sorted(glob.glob(os.path.join(SIM, "sim_stress_*.json"))):
        name = os.path.basename(f)[4:-5]
        if name != "stress_measured_recheck":
            gsc[name] = session(name)
    gres = {k: dict(central=guard_timeline(v, plan, sB["wall_cap_s"], False),
                    pess=guard_timeline(v, plan, sB["wall_cap_s"], True),
                    complete_central_gh=v["central_gh"], complete_pess_gh=v["pess_gh"]) for k, v in gsc.items()}
    res["guard_B"] = dict(note="R-18: the batch guard (runner/batch_guard_fcr.py) under the 6.2 GPU-h wall cap; "
                               "stress scenarios are not predictions of D2",
                          ceiling_gh=sB["ceiling_gh"], wall_cap_s=sB["wall_cap_s"], reserve_s=CLIENT["RESERVE"],
                          scenarios=gres,
                          never_mid_phase=not any(x[c]["mid_phase_cut"] for x in gres.values() for c in ("central",
                                                                                                         "pess")),
                          expected_spend_central_normal_gh=gres["base (measured acceptance)"]["central"]["gpu_h"])
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
    lines.append("[R-11] acceptance stress scenarios, option B (stress, NOT predictions of D2; against the SUPERSEDED "
                 "ceiling %.2f GPU-h, wall cap %d s and 25 tok/s client rule, as reported under 22f). Base under that "
                 "wall cap: central %s, pessimistic %s" % (
                     c_old, w_old, base_tl["central"]["outcome"], base_tl["pess"]["outcome"]))
    for k, v in st.items():
        c, p_ = v["under_wall_cap_central"], v["under_wall_cap_pess"]
        lines.append("   %-28s blocks %4d: complete session %.2f / %.2f GPU-h (central / pess; C/D decode %s tok/s per "
                     "request) -> ceiling %s; under the cap: central %s [primary complete %s, LP skipped %s, %.2f GPU-h]; "
                     "pess %s [primary complete %s, LP skipped %s, %.2f GPU-h]" % (
                         k, v["num_blocks"], v["complete_central_gh"], v["complete_pess_gh"],
                         v["primary_batch_decode_tok_s"],
                         "holds" if v["within_ceiling_pess"] else ("holds centrally only" if v["within_ceiling_central"]
                                                                   else "EXCEEDED"),
                         c["outcome"], c["primary_complete"], c["skipped_lp"], c["gpu_h"],
                         p_["outcome"], p_["primary_complete"], p_["skipped_lp"], p_["gpu_h"]))
    be = res["stress_B"]["breakeven_decode_tok_s_central"]
    lines.append("   4.18 ceiling holds for every stress scenario: central %s, pessimistic %s. A complete session stays "
                 "within %.2f GPU-h at central rates only while C / D / LP decode stays >= %.1f tok/s per request "
                 "(about %.2f committed tokens per step against the measured 2.84; single-token advance = 1.0)" % (
                     res["stress_B"]["ceiling_holds_central"], res["stress_B"]["ceiling_holds_pess"], c_old,
                     be, 2.84 * be / M["decode"]))
    G_ = res["guard_B"]
    lines.append("[R-16 / R-18] option B under the 6.2 GPU-h ceiling (wall cap %d s, deadline = cap - %d s) with the "
                 "batch guard; never cut mid-phase: %s; expected spend at measured acceptance (central): %.2f GPU-h" % (
                     G_["wall_cap_s"], G_["reserve_s"], G_["never_mid_phase"], G_["expected_spend_central_normal_gh"]))
    for k, v in G_["scenarios"].items():
        c, p_ = v["central"], v["pess"]
        lines.append("   %-28s central: %s [primary %s, LP skipped %s, %.2f GPU-h, actual/projected <= %.2f] | pess: %s "
                     "[primary %s, LP skipped %s, %.2f GPU-h, actual/projected <= %.2f]" % (
                         k, c["outcome"], "complete" if c["primary_complete"] else "INCOMPLETE", c["skipped_lp"],
                         c["gpu_h"], c["max_actual_over_projected"], p_["outcome"],
                         "complete" if p_["primary_complete"] else "INCOMPLETE", p_["skipped_lp"], p_["gpu_h"],
                         p_["max_actual_over_projected"]))
    open(os.path.join(HERE, "cost_fcr.out"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
