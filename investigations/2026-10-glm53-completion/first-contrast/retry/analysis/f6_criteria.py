#!/usr/bin/env python3
"""F6 [R-6]: index-K numerical criteria made relative to frozen, MATCHED long-context nulls. Replaces attempt 1's
G1-idx-struct page (<= 25 % gross per judged page) and early-layer (zero gross in DSA layers 3-15) thresholds, which
failed on the prefill null itself at long context (diagnosis-fc1 s.5). The exact structural requirements stay exact
(request identity and token position of every dump, physical mapping, ledger attribution, prohibited writes, committed
bytes stable: gate_fc structural rows + R-idx-struct below). Unit definitions and the pool 'gross' threshold (rel L2 >
0.25) are attempt 1's, unchanged (fc_common).

Frozen rules (a tested comparison T = a decode run vs the prefill reference LW1 on pools wholly in [P, P_L);
nulls on the SAME units: N_pp = LW1 vs LW2 (prefill vs prefill at a different build length), N_dd = C1 vs C2 (decode
vs decode, in-session); a unit is evaluated only where both nulls exist, except in the calibration below, where run-fc1
has no C2 and N_dd is marked absent):
  R-idx-pooled  pooled gross <= max(2 x max(N_dd, N_pp), 0.1 %)                      (the 21g-1 rule, as in G1-content)
  R-idx-pages   every judged page (fixture x rank x layer x 64-pool page, >= 16 pools): gross fraction
                <= max(2 x N_pp page, 0.25)  -- the attempt-1 0.25 page value survives as the floor; a page whose OWN
                matched prefill null is noisy gets a proportionally wider limit, no other page does
  R-idx-early   DSA layers 3-15 as one stratum: gross <= max(2 x N_pp, 0.1 %) on the early-layer units
  The page and early rules use the PREFILL null only: N_dd = C1 vs C2 contains the tested decode run itself, so a
  fault confined to one decode run inflates its own N_dd (the 21h limit of the pooled rule) and would widen its own
  limit (synthetic plant r-f6-page, found before freezing). N_dd is reported beside them and stays in the pooled rule.
  reported      by fixture, by fixture x layer, by page; pooled; per-fixture flags (R3) with leave-flagged-out; the
                ORIGINAL criteria and their results beside.
Reading of a pass: "compatible with measured long-context numerical variation" (never "equivalent").

Calibration and sensitivity (main; CPU container of the B1 image, run-fc1's real C1 / LW1 / LW2 dumps; post-inspection,
calibration only, no D2 interpretation): the rules on C1 vs LW1 against N_pp, and planted faults that must stay
detectable after the recalibration: wrong page (two index-K pages swapped), foreign request (A8's window replaced by
L8's at identical positions, tag and P kept), stale state (window pages carrying the content of earlier pages), unwritten
page (zeros), and a single-layer wrong page.
usage: f6_criteria.py RUN_DIR FC_ROOT OUT_JSON
"""
import collections
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FC_ANALYSIS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "analysis")
sys.path.insert(0, FC_ANALYSIS)
import fc_common as fc  # noqa: E402

PAGE_FLOOR = fc.PAGE_MAX_GROSS          # 0.25: attempt-1 page value, now the floor of the null-relative page limit
FACTOR = fc.CONTENT_FACTOR              # 2
EARLY = (3, fc.EARLY_IDX_LAYERS)        # DSA layers 3-15
WORDING = "compatible with measured long-context numerical variation"


def rows_for(run, G, a, b):
    """{fixture: {(rank, layer, pool): rel}} for comparison a-vs-b over each fixture's [P, P_L) pools."""
    out = {}
    for n in G["order"]:
        lo, hi = G["fx"][n]["ref_pools"]
        out[n] = fc.compare_pools(run, f"{a}-{n}", f"{b}-{n}", lo, hi)
    return out


def page_table(rows):
    pg = collections.defaultdict(list)
    for n, r in rows.items():
        for (rank, layer, pool), rel in r.items():
            pg[(n, rank, layer, pool // 64)].append(rel > fc.GROSS)
    return {k: (float(np.mean(v)), len(v)) for k, v in pg.items()}


def page_rule(T, DD, PP):
    """T, DD, PP: page tables. Returns (ok, detail)."""
    judged, fails, no_null = 0, [], 0
    for k, (f, n) in T.items():
        if n < fc.PAGE_MINP:
            continue
        judged += 1
        nulls = [x[k][0] for x in (DD, PP) if x is not None and k in x]
        if not nulls:
            no_null += 1
            lim = PAGE_FLOOR
        else:
            lim = max(FACTOR * max(nulls), PAGE_FLOOR)
        if f > lim:
            fails.append(dict(fixture=k[0], rank=k[1], layer=k[2], page=k[3], frac=round(f, 4), limit=round(lim, 4),
                              nulls=[round(x, 4) for x in nulls]))
    cover = 1 - no_null / judged if judged else 0.0
    return (judged > 0 and not fails and cover >= fc.NULL_COVER_MIN,
            dict(pages_judged=judged, failing=sorted(fails, key=lambda d: -d["frac"])[:20], n_failing=len(fails),
                 null_coverage=cover))


def units(T, DD, PP):
    """matched units (fixture, key, rel, rel_dd, rel_pp); with DD absent, rel_dd is set to rel_pp (marked)."""
    out, nt = [], 0
    for n, r in T.items():
        d_, p_ = (DD or {}).get(n), PP.get(n, {})
        for k, rel in r.items():
            nt += 1
            if k in p_ and (DD is None or k in d_):
                out.append((n, k, rel, (d_ or p_)[k], p_[k]))
    return out, nt


def stratum(us, pred, prefill_only=False):
    s = [u for u in us if pred(u)]
    lim, ndd, npp = fc.limit_of(s)
    if prefill_only and s:
        lim = max(FACTOR * npp, fc.CONTENT_FLOOR)
    g = fc.gross(u[2] for u in s)
    return dict(n=len(s), gross=g, gross_n=int(sum(u[2] > fc.GROSS for u in s)), N_dd=ndd, N_pp=npp, limit=lim,
                ok=bool(s) and g <= lim)


def evaluate(T, DD, PP, dd_absent=False):
    """All F6 rows for one tested comparison (rows dicts); original criteria beside."""
    us, nt = units(T, DD, PP)
    pooled = stratum(us, lambda u: True)
    pooled["coverage"] = len(us) / nt if nt else 0.0
    early = stratum(us, lambda u: EARLY[0] <= u[1][1] <= EARLY[1], prefill_only=True)
    pT, pPP = page_table(T), page_table(PP)
    pok, pdet = page_rule(pT, None, pPP)
    by_fx = {}
    for n in T:
        s = stratum(us, lambda u, n=n: u[0] == n)
        by_fx[n] = s
    flags = fc.flags(us)
    # originals
    ps = fc.page_stats({n: r for n, r in T.items()})
    eg, en = fc.early_gross({n: r for n, r in T.items()})
    return dict(
        pooled=dict(pooled, ok=pooled["ok"] and pooled["coverage"] >= fc.NULL_COVER_MIN),
        pages=dict(pdet, ok=pok), early=early, by_fixture=by_fx, flags=flags,
        by_fixture_layer=fc.table(us)["per_fixture_layer"],
        original=dict(page_over_25=ps["over"], page_max=ps["page_max"], early_gross=eg, early_units=en,
                      orig_pass=ps["over"] == 0 and eg == 0),
        dd_absent=dd_absent,
        ok=pooled["ok"] and pok and early["ok"])


# ------------------------------------------------------------------ calibration + planted faults on run-fc1
class Faulty:
    """Wraps fc.Run.load: returns a modified copy of chosen window dumps (index-K pages only)."""

    def __init__(self, run, G):
        self.run, self.G = run, G
        self.orig = run.load
        self.plans = {}

    def set(self, plans):
        self.plans = plans
        self.run._cache.clear()
        self.run.load = self.load

    def clear(self):
        self.plans = {}
        self.run._cache.clear()
        self.run.load = self.orig

    def load(self, p):
        rec = self.orig(p)
        for match, fn in self.plans.items():
            if match in os.path.basename(p):
                rec = fn(copy.copy(rec), p)
        return rec


def idx_page_ids(rec, n, G, pools):
    """physical index-K page ids holding the given logical pools in a dump record."""
    blk = fc.attn_blocks(rec)
    return [blk((4 * pool + 3) // 4352) * 17 + (pool % 1088) // 64 for pool in pools]


def main():
    rd, root, outp = sys.argv[1:4]
    sys.path.insert(0, os.path.join(root, "analysis"))
    import fc_geometry as geo
    G = geo.load(root)
    run = fc.Run(rd)
    PP = rows_for(run, G, "LW1", "LW2")
    res = {}
    base_T = rows_for(run, G, "C1", "LW1")
    res["C1_vs_LW1_post_inspection"] = evaluate(base_T, None, PP, dd_absent=True)
    res["C1_vs_LW2_post_inspection"] = evaluate(rows_for(run, G, "C1", "LW2"), None, PP, dd_absent=True)
    res["LW1_vs_LW2_self"] = dict(note="the null against itself (every unit at its own limit): sanity")
    F = Faulty(run, G)

    def swap_pages(n, ranks, layers, k1, k2):
        f = G["fx"][n]
        lo = f["ref_pools"][0]

        def fn(rec, p):
            rank = int(p.rsplit("_r", 1)[1].split(".")[0])
            if rank not in ranks:
                return rec
            rec["layers"] = dict(rec["layers"])
            for nm, L in fc.idx_layers(rec).items():
                if layers is not None and fc.layer_no(nm) not in layers:
                    continue
                a, b = idx_page_ids(rec, n, G, [lo + 64 * k1, lo + 64 * k2])
                pg = dict(L["pages"])
                if a in pg and b in pg:
                    pg[a], pg[b] = pg[b], pg[a]
                rec["layers"][nm] = dict(L, pages=pg)
            return rec
        return fn

    def foreign(src_n, dst_n):
        def fn(rec, p):
            ps = run.win_dump(f"C1-{src_n}")
            rank = int(p.rsplit("_r", 1)[1].split(".")[0])
            other = F.orig(ps[rank])
            out = dict(other)
            out["tag"], out["P"], out["range"] = rec["tag"], rec["P"], rec["range"]
            return out
        return fn

    def stale(n, rank_, k_from, k_to, count):
        f = G["fx"][n]
        lo = f["ref_pools"][0]

        def fn(rec, p):
            rank = int(p.rsplit("_r", 1)[1].split(".")[0])
            if rank != rank_:
                return rec
            rec["layers"] = dict(rec["layers"])
            for nm, L in fc.idx_layers(rec).items():
                pg = dict(L["pages"])
                for j in range(count):
                    a, b = idx_page_ids(rec, n, G, [lo + 64 * (k_from + j), lo + 64 * (k_to + j)])
                    if a in pg and b in pg:
                        pg[b] = pg[a].clone()
                rec["layers"][nm] = dict(L, pages=pg)
            return rec
        return fn

    def zero(n, rank_, k):
        f = G["fx"][n]
        lo = f["ref_pools"][0]

        def fn(rec, p):
            rank = int(p.rsplit("_r", 1)[1].split(".")[0])
            if rank != rank_:
                return rec
            rec["layers"] = dict(rec["layers"])
            for nm, L in fc.idx_layers(rec).items():
                pg = dict(L["pages"])
                (b,) = idx_page_ids(rec, n, G, [lo + 64 * k])
                if b in pg:
                    pg[b] = pg[b].clone() * 0
                rec["layers"][nm] = dict(L, pages=pg)
            return rec
        return fn

    plants = {
        "wrong_page_L3_all_layers_rank0": {"snap_C1-L3_": swap_pages("L3", (0,), None, 10, 11)},
        "wrong_page_L1_one_layer_rank1": {"snap_C1-L1_": swap_pages("L1", (1,), {27}, 20, 21)},
        "wrong_page_M4_fragile_region_rank1": {"snap_C1-M4_": swap_pages("M4", (1,), None, 8, 9)},
        "foreign_request_A8_gets_L8": {"snap_C1-A8_": foreign("L8", "A8")},
        "stale_state_L1_rank1_2_pages": {"snap_C1-L1_": stale("L1", 1, 4, 16, 2)},
        "unwritten_page_L5_rank0": {"snap_C1-L5_": zero("L5", 0, 12)},
    }
    pl = {}
    for name, plan in plants.items():
        F.set(plan)
        T = rows_for(run, G, "C1", "LW1")
        ev = evaluate(T, None, PP, dd_absent=True)
        attrib = []
        for n in G["order"]:
            for rank, pth in (run.win_dump(f"C1-{n}") or {}).items():
                rec = run.load(pth)
                if rec.get("tag") != f"C1-{n}":
                    attrib.append((n, rank))
        F.clear()
        detected_by = [k for k in ("pooled", "pages", "early") if not ev[k]["ok"]]
        if ev["flags"]:
            detected_by.append("fixture flag (reported)")
        pl[name] = dict(detected=bool([k for k in detected_by if k != "fixture flag (reported)"]),
                        detected_by=detected_by, pooled=ev["pooled"]["gross"], limit=ev["pooled"]["limit"],
                        pages_failing=ev["pages"]["failing"][:3], n_pages_failing=ev["pages"]["n_failing"],
                        early=ev["early"]["gross_n"], flags=[x["fixture"] for x in ev["flags"]],
                        attribution_violations=attrib)
    res["planted"] = pl
    res["rules"] = dict(page_floor=PAGE_FLOOR, factor=FACTOR, early_layers=EARLY, gross=fc.GROSS,
                        content_floor=fc.CONTENT_FLOOR, page_minp=fc.PAGE_MINP, wording=WORDING)
    json.dump(res, open(outp, "w"), indent=1, default=str)
    b = res["C1_vs_LW1_post_inspection"]
    print("C1 vs LW1 (post-inspection, N_dd absent): ok", b["ok"], "| pooled %.5f limit %.5f cov %.3f | pages judged %d "
          "failing %d | early %d/%d gross, limit %.5f | ORIGINAL pass %s (pages over 25%% %d, early %d)" % (
              b["pooled"]["gross"], b["pooled"]["limit"], b["pooled"]["coverage"], b["pages"]["pages_judged"],
              b["pages"]["n_failing"], b["early"]["gross_n"], b["early"]["n"], b["early"]["limit"],
              b["original"]["orig_pass"], b["original"]["page_over_25"], b["original"]["early_gross"]))
    c = res["C1_vs_LW2_post_inspection"]
    print("C1 vs LW2 (post-inspection): ok", c["ok"], "pages failing", c["pages"]["n_failing"], "flags",
          [x["fixture"] for x in c["flags"]])
    print("flags C1 vs LW1:", [(x["fixture"], x["by_unit"], x["by_position"]) for x in b["flags"]])
    for k, v in pl.items():
        print("plant %-38s detected %-5s by %s (pooled %.5f / %.5f, pages failing %d)" % (
            k, v["detected"], v["detected_by"], v["pooled"], v["limit"], v["n_pages_failing"]))


if __name__ == "__main__":
    main()
