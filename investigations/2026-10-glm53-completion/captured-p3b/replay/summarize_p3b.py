#!/usr/bin/env python3
"""CPU only (host python3, json): one summary of a replay attempt, applying the outcome rules (PROPOSAL.md s.7).

usage: summarize_p3b.py REPLAY_ATTEMPT_DIR CAPTURE_ATTEMPT_DIR P3_ATTEMPT2_DIR
reads  attn-v091.json, moe-v091.json, moe-v070.json (public replay outputs), the capture coverage.json, and P3's
       v091.json / v070.json (synthetic-input magnitudes for comparison)
writes summary.json, summary.md (counts, verdicts, magnitudes; no tensors, positions or lengths)

Effects are reported separately and never combined: E-plan per engine (v091 tp2r0; v070 ep2r0), the forced-tile
confirmation (its own section), attention per regime.  E-version and E-partition are not tested here.
"""
import json
import os
import sys


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def p3_mag(p3, slice_, layer):
    """P3 synthetic: median / max rel L2 (bf16) and row-differ fraction over fixtures with m >= 16."""
    try:
        fx = p3["slices"][slice_]["layers"][str(layer)]["fixtures"]
    except (KeyError, TypeError):
        return None
    rl = [f["between"]["rel_l2"] for f in fx if f.get("m", 0) >= 16 and "between" in f and f["between"].get("n_rows_differ")]
    rows = sum(f["between"].get("rows", 0) for f in fx if f.get("m", 0) >= 16 and "between" in f)
    diff = sum(f["between"].get("n_rows_differ", 0) for f in fx if f.get("m", 0) >= 16 and "between" in f)
    return dict(rel_l2_median=median(rl), rel_l2_max=max(rl) if rl else None,
                row_differ_frac=round(diff / rows, 4) if rows else None, fixtures=len(fx))


def main():
    ao, cap_ao, p3dir = sys.argv[1], sys.argv[2], sys.argv[3]
    attn = load(os.path.join(ao, "attn-v091.json"))
    moe = {e: load(os.path.join(ao, f"moe-{e}.json")) for e in ("v091", "v070")}
    cov = load(os.path.join(cap_ao, "coverage.json"))
    p3 = {e: load(os.path.join(p3dir, f"{e}.json")) for e in ("v091", "v070")}
    out = dict(capture=dict(verdict=(cov or {}).get("verdict"), deficits=(cov or {}).get("deficits")), moe={}, attention={},
               forced_tile=None, not_tested=["E-partition (EP2 vs TP2 reductions)", "E-version on captured rows",
                                             "extend-kernel (mixed prefill+verify) attention rows",
                                             "attention on rank 1 (MLA layers >= 25)"])
    for e, d in moe.items():
        if d is None:
            out["moe"][e] = dict(status="not run")
            continue
        sl = "tp2r0" if e == "v091" else "ep2r0"
        lay = {}
        for L, ld in d.get("layers", {}).items():
            s = ld.get("summary", {})
            lay[L] = dict(verdict=s.get("verdict"), coverage_deficits=s.get("coverage_deficits"),
                          baseline_all_bitequal=s.get("baseline_all_bitequal"),
                          baseline_not_reproduced=s.get("baseline_not_reproduced"),
                          within_all_bitequal=s.get("within_all_bitequal"), by_m=s.get("by_m"),
                          composed=[dict(m=c["m"], rows_differ=c["between"].get("n_rows_differ"),
                                         rel_l2=c["between"].get("rel_l2")) for c in ld.get("composed", [])],
                          control_ulp_ok=(ld.get("control_ulp") or {}).get("ok"),
                          p3_synthetic=p3_mag(p3.get(e), sl, L))
        out["moe"][e] = dict(status=d.get("status"), slice=sl, layers=lay, errors=d.get("errors"))
    ft = (moe.get("v091") or {}).get("forced_tile_confirmation")
    if ft:
        out["forced_tile"] = dict(verdict=ft.get("verdict"), layers={
            L: dict(forcing=v.get("forcing"), forcing_effective=v.get("forcing_effective"), n_compared=v.get("n_compared"),
                    all_bitequal=v.get("all_bitequal"), error=v.get("error"),
                    captured_rows_differ=sum(c["n_rows_differ"] for c in v.get("captured", [])),
                    p3_fixtures_digest_match=[c.get("p3_input_digest_match") for c in v.get("p3_fixtures", [])])
            for L, v in ft.get("layers", {}).items()})
    if attn:
        for L, ld in attn.get("layers", {}).items():
            s = ld.get("summary", {})
            out["attention"][L] = dict(kernel=dict((k, ld.get("kernel", {}).get(k)) for k in
                                                   ("plan_path", "caps_equal_to_served", "caps_diff")),
                                       rows=s.get("rows"), not_anchored=s.get("not_anchored"),
                                       not_reproduced=s.get("not_reproduced"), ctrl_halve=[s.get("ctrl_halve_detected"),
                                                                                           s.get("ctrl_halve_rows")],
                                       plant_detected_frac=s.get("plant_detected_frac"),
                                       compaction_ctrl_fail=s.get("compaction_ctrl_fail"), by_regime=s.get("by_regime"))
        out["attention_status"] = attn.get("status")
    json.dump(out, open(os.path.join(ao, "summary.json"), "w"), indent=1, default=str)
    md = ["# captured-p3b replay summary", "", f"Capture: {out['capture']['verdict']} {out['capture']['deficits'] or ''}", ""]
    for e, v in out["moe"].items():
        md.append(f"## MoE {e} ({v.get('slice')}): {v.get('status')}")
        for L, l in (v.get("layers") or {}).items():
            md.append(f"- L{L}: {l['verdict']}; baseline bit-equal {l['baseline_all_bitequal']}; deficits {l['coverage_deficits']}")
            for k, b in (l.get("by_m") or {}).items():
                md.append(f"  - {k}: rows {b['rows']}, differ {b['rows_differ']} ({b['row_differ_frac']}), rel L2 median "
                          f"{b['rel_l2_median']}, max {b['rel_l2_max']}, max abs {b['max_abs']}")
            md.append(f"  - P3 synthetic (m >= 16): {l['p3_synthetic']}")
    if out["forced_tile"]:
        md.append(f"## Forced-tile confirmation: {out['forced_tile']['verdict']}")
        for L, l in out["forced_tile"]["layers"].items():
            md.append(f"- L{L}: effective {l['forcing_effective']}, compared {l['n_compared']}, all bit-equal {l['all_bitequal']}")
    for L, a in out["attention"].items():
        md.append(f"## Attention L{L}: plan {a['kernel'].get('plan_path')}, caps equal {a['kernel'].get('caps_equal_to_served')}")
        md.append(f"- rows {a['rows']}, not anchored {a['not_anchored']}, P1 {a['ctrl_halve']}, P2 detected {a['plant_detected_frac']}")
        for reg, r in (a.get("by_regime") or {}).items():
            md.append(f"  - {reg}: {r.get('verdict')} (anchored {r.get('anchored')}, Bc differ {r.get('bc_differ')}, "
                      f"Bf differ {r.get('bf_differ')}, r {r.get('r_hist')})")
    md += ["", "Not tested: " + "; ".join(out["not_tested"])]
    open(os.path.join(ao, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
