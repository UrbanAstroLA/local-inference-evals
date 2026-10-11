"""Summarize results/summary-*-full.json into markdown tables (CPU only, stdlib)."""
import json, sys
from collections import defaultdict

def load(v):
    return json.load(open(f"results/summary-{v}-full.json"))

def route_of(r):
    m = r.get("meta") or {}
    if "route" in m:
        return m["route"]
    return "+".join(sorted(set(m.get("rank_routes", []))))

def pathname(v, r):
    rt = route_of(r)
    if v == "v091":
        return {"paged_fused": "0.9.1 fused (coop merge)", "paged_tiled": "0.9.1 tiled"}[rt] + (" [B12X_FUSED_INDEXER=0]" if r["path"] == "tiled_forced" else "")
    return {"paged_fused": "0.7.0 per-rank fused", "paged_tiled": "0.7.0 per-rank tiled"}[rt] + (" [B12X_FUSED_INDEXER=0]" if r["path"] == "tiled_forced" else "") + " + DCP merge"

def pivot(kind, cellfmt):
    out = []
    for v in ["v091", "v070"]:
        S = [r for r in load(v) if r["kind"] == kind]
        groups = defaultdict(dict)
        rowsset = sorted({r["rows"] for r in S})
        for r in S:
            groups[pathname(v, r)][(r["pools"], r["rows"])] = r
        for p, cells in groups.items():
            rows_here = sorted({k[1] for k in cells})
            pools_here = sorted({k[0] for k in cells})
            out.append(f"\n**{p}** — {kind}\n")
            out.append("| pools \\ rows | " + " | ".join(str(x) for x in rows_here) + " |")
            out.append("|---|" + "---|" * len(rows_here))
            for n in pools_here:
                out.append(f"| {n} | " + " | ".join(cellfmt(cells[(n, rr)]) if (n, rr) in cells else "" for rr in rows_here) + " |")
    return "\n".join(out)

def f_bad(r):
    return f"{100*r['frac_col511_bad']:.0f}% / {r['rank_min']},{r['rank_p10']},{r['rank_median']}"

def f_rtr(r):
    return f"{100*r['rtr_frac_col511_varies']:.0f}"

def anomalies():
    lines = []
    for v in ["v091", "v070"]:
        for r in load(v):
            for k in ["set_mismatch_exact", "set_mismatch_tol", "dup", "out_of_range"]:
                if r.get(k):
                    lines.append(f"{v} {r['kind']} {r['path']} N={r['pools']} rows={r['rows']} {k}={r[k]}")
            if r.get("rtr_frac_set_varies"):
                lines.append(f"{v} {r['kind']} {r['path']} N={r['pools']} rows={r['rows']} rtr_set_varies={r['rtr_frac_set_varies']}")
    return lines

def tolsens():
    lines = []
    for v in ["v091", "v070"]:
        for r in load(v):
            if abs(r["frac_col511_bad"] - r["frac_col511_bad_tol"]) > 0.05:
                lines.append(f"{v} {r['kind']} {pathname(v,r)} N={r['pools']} rows={r['rows']} bad_exact={r['frac_col511_bad']:.2f} bad_tol={r['frac_col511_bad_tol']:.2f}")
    return lines

if __name__ == "__main__":
    what = sys.argv[1]
    if what == "bad":
        print(pivot(sys.argv[2], f_bad))
    elif what == "rtr":
        print(pivot(sys.argv[2], f_rtr))
    elif what == "anom":
        print("\n".join(anomalies()) or "none")
    elif what == "tol":
        print("\n".join(tolsens()) or "none")
    elif what == "direct":
        for v in ["v091", "v070"]:
            for r in load(v):
                if r["kind"].startswith("row_topk_direct"):
                    print(f"| {v} | {r['kind'][-1]} | {r['rows']} | {100*r['frac_col511_bad']:.0f}% | {r['rank_min']},{r['rank_p10']},{r['rank_median']} | {r['set_mismatch_exact']} | {100*r['rtr_frac_col511_varies']:.0f}% |")
    elif what == "meta":
        seen = set()
        for v in ["v091", "v070"]:
            for r in load(v):
                key = (v, r["path"], r["rows"], json.dumps({k: x for k, x in (r.get("meta") or {}).items() if k != "score_err_over_bound_max"}))
                if key not in seen and r["kind"] == "random_gauss":
                    seen.add(key); print(key)
            errs = [r["meta"].get("score_err_over_bound_max", 0) for r in load(v) if r.get("meta")]
            if v == "v070": print("v070 max score err / fp32 bound:", max(errs))
