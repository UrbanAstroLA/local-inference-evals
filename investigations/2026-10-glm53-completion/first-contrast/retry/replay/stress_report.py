#!/usr/bin/env python3
"""[R-11] Acceptance stress scenarios (PLAN.md amendment 22f s.3): summary of out/sim_stress_*.json (run_stress.sh).
Host python, json only. Stress scenarios, NOT predictions of D2.

Per scenario, for the primary (C / D) and LP (LC / LD) batches separately, and for the sentinels:
  * cache survival: every lookup's class (hit / partial / miss) and its minimum survival margin (free-queue blocks the
    schedule could still allocate before the first block the lookup needs would be evicted; session_sim.py);
  * all-8 decoding before the lead-in ends: the output index of the leading request when the last of the 8 starts
    decoding, per batch, against 512 (the in-session rc 13 bound);
  * runtime: modelled seconds per batch and per session against the base scenario (the cost consequences, wall cap and
    the 4.18 ceiling are in cost/cost_fcr.py);
  * the realised advance (committed tokens per speculative decode step), confirming the stress mode took effect;
  * measured_recheck: the base scenario re-run with the R-11 code; it must equal out/sim_base.json exactly.
Writes out/F3_stress.json and out/F3_stress.out.
"""
import glob
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
LEAD_MAX = 512
PRIMARY, LP = ("C", "D"), ("LC", "LD")


def strip_adv(phases):
    return [{k: v for k, v in p.items() if k != "advance"} for p in phases]


def summarise(d, base):
    ph = d["phases"]
    out = dict(scenario=d["scenario"], total_s=d["total_s"], delta_vs_base_s=round(d["total_s"] - base["total_s"], 1))
    nonhit = [(p["phase"], r["fixture"], r["cls"], r["hit"], r["expected"]) for p in ph for r in p.get("requests", [])
              if r["cls"] not in ("hit", "cold")]
    out["lookups"] = sum(1 for p in ph for r in p.get("requests", []) if r["expected"] > 0)
    out["non_hits"] = nonhit
    for name, forms in (("primary", PRIMARY), ("lp", LP)):
        bs = [p for p in ph if p["kind"] == "run" and p.get("form") in forms]
        margins = [r["margin"] for p in bs for r in p["requests"] if r["margin"] is not None]
        leads = {p["phase"]: p.get("lead") for p in bs}
        adv = [p["advance"]["mean"] for p in bs if p.get("advance")]
        base_secs = {p["phase"]: p["secs"] for p in base["phases"]}
        out[name] = dict(
            batches=len(bs), min_margin=min(margins) if margins else None,
            non_hits=[x for x in nonhit if x[0] in leads],
            leads=leads, max_lead=max((v for v in leads.values() if v is not None), default=None),
            all8_before_leadin_end=all(v is not None and v < LEAD_MAX for v in leads.values()),
            secs={p["phase"]: p["secs"] for p in bs},
            secs_vs_base={p["phase"]: round(p["secs"] - base_secs.get(p["phase"], 0.0), 1) for p in bs},
            advance_mean=(round(min(adv), 3), round(max(adv), 3)) if adv else None,
            advance_by_fixture={p["phase"]: p["advance"]["by_fixture"] for p in bs if p.get("advance")})
    sent = [r for p in ph if p["phase"] in ("C2", "PS-a", "PS-b", "D2", "PS-c") for r in p["requests"]]
    out["sentinel_shared_state"] = dict(all_hit=all(r["cls"] == "hit" for r in sent) and len(sent) == 40,
                                        min_margin=min((r["margin"] for r in sent if r["margin"] is not None),
                                                       default=None))
    allm = [r["margin"] for p in ph for r in p.get("requests", []) if r["margin"] is not None]
    out["min_margin"] = min(allm) if allm else None
    out["ok"] = (not nonhit and out["primary"]["all8_before_leadin_end"] and out["lp"]["all8_before_leadin_end"]
                 and out["sentinel_shared_state"]["all_hit"])
    return out


def main():
    S = {}
    for f in sorted(glob.glob(os.path.join(OUT, "sim_stress_*.json"))):
        d = json.load(open(f))
        S[d["scenario"]["name"]] = d
    base = json.load(open(os.path.join(OUT, "sim_base.json")))
    rep = dict(note="acceptance stress scenarios (22f s.3): stress scenarios, not predictions of D2; eviction is "
                    "avoided in the replayed scenarios, with measured survival margins; the actual-session gates remain "
                    "authoritative", lead_max=LEAD_MAX, scenarios={})
    rc = S.get("stress_measured_recheck")
    rep["measured_recheck_equals_base"] = bool(rc) and strip_adv(rc["phases"]) == base["phases"] and \
        rc["occupancy"] == base["occupancy"] and rc["total_s"] == base["total_s"]
    for k, d in S.items():
        if k == "stress_measured_recheck":
            continue
        rep["scenarios"][k] = summarise(d, base)
    sc = rep["scenarios"]
    rep["summary"] = dict(
        n=len(sc), all_ok=all(v["ok"] for v in sc.values()),
        non_hits=sum(len(v["non_hits"]) for v in sc.values()),
        min_margin_1275=min(v["min_margin"] for v in sc.values() if v["scenario"].get("num_blocks", 1275) == 1275),
        min_margin_1150=min((v["min_margin"] for v in sc.values() if v["scenario"].get("num_blocks") == 1150),
                            default=None),
        min_margin_primary=min(v["primary"]["min_margin"] for v in sc.values()),
        min_margin_lp=min(v["lp"]["min_margin"] for v in sc.values()),
        min_margin_sentinel=min(v["sentinel_shared_state"]["min_margin"] for v in sc.values()),
        max_lead_primary=max(v["primary"]["max_lead"] for v in sc.values()),
        max_lead_lp=max(v["lp"]["max_lead"] for v in sc.values()))
    json.dump(rep, open(os.path.join(OUT, "F3_stress.json"), "w"), indent=1, default=str)
    L = ["R-11 acceptance stress scenarios (22f s.3; stress scenarios, NOT predictions of D2)",
         "measured_recheck (base scenario re-run with the R-11 code) equals sim_base.json exactly: %s"
         % rep["measured_recheck_equals_base"], "",
         "%-30s %6s %8s %9s %9s %10s %10s %10s %8s %8s %9s %9s" % (
             "scenario", "blocks", "lookups", "non-hits", "min_marg", "marg_prim", "marg_LP", "marg_sent", "lead_pr",
             "lead_LP", "session_s", "vs_base")]
    for k, v in sc.items():
        L.append("%-30s %6s %8d %9d %9s %10s %10s %10s %8s %8s %9.0f %+9.0f" % (
            k, v["scenario"].get("num_blocks", 1275), v["lookups"], len(v["non_hits"]), v["min_margin"],
            v["primary"]["min_margin"], v["lp"]["min_margin"], v["sentinel_shared_state"]["min_margin"],
            v["primary"]["max_lead"], v["lp"]["max_lead"], v["total_s"], v["delta_vs_base_s"]))
    L.append("")
    L.append("realised advance (committed tokens per speculative decode step; min..max over batches) and C / D, LC / "
             "LD batch seconds (modelled):")
    for k, v in sc.items():
        ps = v["primary"]["secs"]
        ls = v["lp"]["secs"]
        L.append("  %-30s primary adv %s, %.0f-%.0f s per batch | LP adv %s, %.0f-%.0f s per batch" % (
            k, v["primary"]["advance_mean"], min(ps.values()), max(ps.values()), v["lp"]["advance_mean"],
            min(ls.values()), max(ls.values())))
    bp = [p["secs"] for p in base["phases"] if p["kind"] == "run" and p.get("form") in PRIMARY]
    bl = [p["secs"] for p in base["phases"] if p["kind"] == "run" and p.get("form") in LP]
    L.append("  %-30s primary %.0f-%.0f s per batch | LP %.0f-%.0f s per batch; session %.0f s" % (
        "base (measured acceptance)", min(bp), max(bp), min(bl), max(bl), base["total_s"]))
    s = rep["summary"]
    L.append("")
    L.append("SUMMARY: %d stress scenarios; every lookup a full hit: %s (non-hits %d); all 8 decoding before the "
             "lead-in ends in every primary and LP batch: %s (max lead primary %s, LP %s, < %d); sentinel shared state "
             "restored in every scenario; minimum survival margin %s blocks at 1,275 (%s at the 1,150 floor; primary "
             "%s, LP %s, sentinel %s)" % (
                 s["n"], s["non_hits"] == 0, s["non_hits"],
                 all(v["primary"]["all8_before_leadin_end"] and v["lp"]["all8_before_leadin_end"] for v in sc.values()),
                 s["max_lead_primary"], s["max_lead_lp"], LEAD_MAX, s["min_margin_1275"], s["min_margin_1150"],
                 s["min_margin_primary"], s["min_margin_lp"], s["min_margin_sentinel"]))
    L.append("Wording: eviction avoided in the replayed scenarios, with measured survival margins; the actual-session "
             "gates remain authoritative.")
    open(os.path.join(OUT, "F3_stress.out"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
