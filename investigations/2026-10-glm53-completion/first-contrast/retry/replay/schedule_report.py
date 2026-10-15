#!/usr/bin/env python3
"""F3 [R-3]: the feasible schedule, from the scenario sweep (out/sim_*.json; host python, no engine needed).

Writes out/F3_schedule.json and prints:
  1. the session order with the actual submission order of every pass (base scenario), steps and modelled seconds;
  2. live and cached block occupancy by group per phase (base scenario, sampled every 8 steps);
  3. per lookup: expected restore (F1), class in EVERY scenario, the boundary state restored (KDA at c, 7 groups)
     and its writer, and the minimum survival margin across scenarios (free-queue blocks the schedule could still
     absorb before the first needed block would be evicted) = the conservative occupancy bound;
  4. all-8-in-flight lead per batch, maximum across scenarios (gate: < 512);
  5. block identity: users of one build restore the same physical MLA/index-K blocks;
  6. the scenario table (non-hits, minimum margin, maximum lead, modelled seconds).
"""
import collections
import glob
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
GROUPS = ["mla+idx", "tail", "kda0", "kda1", "kda2", "kda3", "kda4", "kda5", "kda6", "draft"]


def load():
    S = {}
    for f in sorted(glob.glob(os.path.join(OUT, "sim_*.json"))):
        d = json.load(open(f))
        S[d["scenario"]["name"]] = d
    return S


def main():
    S = load()
    base = S["base"]
    rep = {}
    # 1. order
    sched = []
    for p in base["phases"]:
        sched.append(dict(phase=p["phase"], kind=p["kind"], secs=p["secs"], steps=p["steps"],
                          submission_order=p.get("order"), lead=p.get("lead")))
    rep["schedule_base"] = sched
    # 2. occupancy by group per phase
    occ = collections.defaultdict(list)
    for n, t, ph, o in base.get("occupancy", []):
        occ[ph].append(o)
    occ_tab = {}
    for ph, L in occ.items():
        occ_tab[ph] = dict(samples=len(L),
                           live_max={GROUPS[g]: max(o["live"][g] for o in L) for g in range(10)},
                           cached_free_min={GROUPS[g]: min(o["cached_free"][g] for o in L) for g in range(10)},
                           cached_free_max={GROUPS[g]: max(o["cached_free"][g] for o in L) for g in range(10)},
                           free_min=min(o["free"] for o in L), free_max=max(o["free"] for o in L),
                           live_total_max=max(sum(o["live"]) for o in L))
    rep["occupancy_by_phase_base"] = occ_tab
    # 3. lookups across scenarios
    look = {}
    ablation = {n for n in S if n.startswith("no_touch")}
    rep["ablation_scenarios"] = sorted(ablation)
    for name, d in S.items():
        if name in ablation:
            continue
        for p in d["phases"]:
            for r in p.get("requests", []):
                if p["kind"] == "build":
                    continue
                k = (p["phase"], r["fixture"], r["rid"])
                e = look.setdefault(k, dict(phase=p["phase"], fixture=r["fixture"], expected=r["expected"],
                                            classes=collections.Counter(), min_margin=None, kda_from=set(),
                                            kinds=set()))
                e["classes"][r["cls"]] += 1
                m = r.get("margin_last_window")
                if m is not None:
                    e["min_margin"] = m if e["min_margin"] is None else min(e["min_margin"], m)
                for w, k2 in r.get("restored_kda", []):
                    e["kda_from"].add("%s/%s" % (w, k2))
                for k2 in r.get("restored_kinds", []):
                    e["kinds"].add(k2)
    rows = []
    for k, e in look.items():
        rows.append(dict(phase=e["phase"], fixture=e["fixture"], expected=e["expected"],
                         classes=dict(e["classes"]), min_margin_all_scenarios=e["min_margin"],
                         restored_boundary_state_writer=sorted(e["kda_from"]), restored_kinds=sorted(e["kinds"])))
    rep["lookups"] = rows
    # 4. leads
    leads = collections.defaultdict(list)
    for name, d in S.items():
        if name in ablation:
            continue
        for p in d["phases"]:
            if p.get("lead") is not None:
                leads[p["phase"]].append(p["lead"])
    rep["lead_max_by_batch"] = {k: max(v) for k, v in leads.items()}
    # 5. block identity across users of one build (base)
    ids = collections.defaultdict(dict)
    for p in base["phases"]:
        for r in p.get("requests", []):
            if r.get("hit"):
                ids[(p["phase"], r["fixture"])] = r["mla_ids"][: r["hit"] // 4352]
    ident = {}
    for a, b in (("C2", "D2"), ("C2", "PS-a"), ("PS-a", "PS-b"), ("PS-b", "PS-c"), ("D2", "PS-c"), ("D3", "C3"),
                 ("LC1", "C1"), ("LD1", "D1"), ("LD3", "D3"), ("C1", "CV")):
        same = []
        for n in ("L1", "L4", "M3", "M4", "L3", "L5", "A8", "L8"):
            x, y = ids.get((a, n)), ids.get((b, n))
            if x is None or y is None:
                continue
            m = min(len(x), len(y))
            same.append(x[:m] == y[:m])
        ident["%s~%s" % (a, b)] = "%d/%d identical" % (sum(same), len(same))
    rep["block_identity_base"] = ident
    # 6. scenarios
    sc = []
    for name, d in S.items():
        nh = [(p["phase"], r["fixture"], r["cls"]) for p in d["phases"] for r in p.get("requests", [])
              if r["cls"] not in ("hit", "cold")]
        ms = [r.get("margin_last_window") for p in d["phases"] for r in p.get("requests", [])
              if r.get("margin_last_window") is not None]
        ld = [p["lead"] for p in d["phases"] if p.get("lead") is not None]
        sc.append(dict(name=name, scenario=d["scenario"], non_hits=len(nh), non_hit_list=nh[:12],
                       min_margin=min(ms) if ms else None, max_lead=max(ld) if ld else None,
                       modelled_s=d["total_s"]))
    rep["scenarios"] = sc
    json.dump(rep, open(os.path.join(OUT, "F3_schedule.json"), "w"), indent=1, default=str)
    # console
    print("== 1. schedule (base): phase kind secs steps lead submission order")
    for r in sched:
        print("  %-10s %-6s %7.1f %6d %5s %s" % (r["phase"], r["kind"], r["secs"], r["steps"], r["lead"],
                                                 " ".join(r["submission_order"] or [])))
    print("== 2. occupancy by group (base): live max [mla tail kda0..6 draft] | cached-free min..max mla / kda0 | "
          "free min")
    for ph, o in occ_tab.items():
        print("  %-10s live %s | mla %d..%d kda0 %d..%d | free %d..%d" % (
            ph, [o["live_max"][g] for g in GROUPS], o["cached_free_min"]["mla+idx"], o["cached_free_max"]["mla+idx"],
            o["cached_free_min"]["kda0"], o["cached_free_max"]["kda0"], o["free_min"], o["free_max"]))
    print("== 3. lookups: phase -> classes over %d design scenarios (ablations %s excluded), min margin, "
          "boundary-state writer" % (len(S) - len(ablation), sorted(ablation)))
    byph = collections.defaultdict(list)
    for r in rows:
        byph[r["phase"]].append(r)
    for ph, L in byph.items():
        cl = collections.Counter()
        for r in L:
            cl.update(r["classes"])
        print("  %-8s %s min margin %s writers %s kinds %s" % (
            ph, dict(cl), min(r["min_margin_all_scenarios"] for r in L if r["min_margin_all_scenarios"] is not None),
            sorted({w for r in L for w in r["restored_boundary_state_writer"]})[:6],
            sorted({w for r in L for w in r["restored_kinds"]})))
    print("== 4. max lead by batch (design scenarios):", rep["lead_max_by_batch"])
    print("== 5. block identity (base):", ident)
    print("== 6. scenarios")
    for r in sc:
        print("  %-22s non-hits %3d min margin %5s max lead %4s modelled %7.1f s" % (
            r["name"], r["non_hits"], r["min_margin"], r["max_lead"], r["modelled_s"]))


if __name__ == "__main__":
    main()
