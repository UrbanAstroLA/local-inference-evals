#!/usr/bin/env python3
"""Tabulate stage-2 replay records (CPU only).
usage: analyze.py OUTDIR  -> OUTDIR/analysis.md, OUTDIR/analysis.json
Reads OUTDIR/replay/v091.jsonl, OUTDIR/replay/v070.jsonl, OUTDIR/probe-a*/index_*.jsonl, OUTDIR/client.jsonl."""
import collections
import glob
import json
import math
import re
import statistics
import sys

OUT = sys.argv[1]
BINS = [(1500, "4k"), (3000, "8k"), (6000, "16k"), (12000, "32k"), (24000, "64k"), (10 ** 9, "120k")]


def lbin(n):
    for hi, name in BINS:
        if n < hi:
            return name


def layer_no(s):
    m = re.search(r"layers\.(\d+)\.", s)
    return int(m.group(1)) if m else -1


def load(p):
    try:
        return [json.loads(l) for l in open(p)]
    except FileNotFoundError:
        return []


v091 = {r.get("path", r["file"]): r for r in load(f"{OUT}/replay/v091.jsonl")}
v070 = {r.get("path", r["file"]): r for r in load(f"{OUT}/replay/v070.jsonl")}
files = sorted(set(v091) & set(v070))
idx = [json.loads(l) for f in glob.glob(f"{OUT}/probe-a*/index_*.jsonl") for l in open(f)]

# ---- sample rows: (path, capture-route, bin, layer) -> list of metric dicts (per row x repeat)
cells = collections.defaultdict(list)
rtr = collections.defaultdict(lambda: [0, 0])
vslive = collections.defaultdict(lambda: [0, 0, 0, 0])
per_rep = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
ties = collections.Counter()
ties_rows = 0
gaps = []
nrows_route = collections.Counter()
nrows_route_bin = collections.Counter()
layers_route = collections.defaultdict(set)
for f in files:
    a, b = v091[f], v070[f]
    route = "tiled" if "tiled" in a["route"] else ("fused" if "fused" in a["route"] else a["route"])
    L = layer_no(a["layer"])
    layers_route[route].add(L)
    nrows_route[route] += a["rows"]
    for i, n in enumerate(a["seq_lens"]):
        bn = lbin(n)
        nrows_route_bin[(route, bn)] += 1
        st = a["ref_stats"][i]
        ties_rows += 1
        ties["eq_t512>1"] += st["n_eq_t512"] > 1
        ties["tol_t512>1"] += st["n_tol_t512"] > 1
        ties["zero_any"] += st["n_zero"] > 0
        ties["zero_in_top512"] += st["zeros_in_top512"] > 0
        ties["dup_in_top520"] += (st["n_eq_any_top520"] or 0) > 0
        if st["gap_512_513"] is not None and st["t512"] != 0:
            gaps.append(abs(st["gap_512_513"] / st["t512"]))
        ties["rows_with_nan_pools"] += (st.get("n_nan") or 0) > 0
        cells[("in-situ 0.9.1 (live)", route, bn, L)].append(dict(a["insitu"][i], nan=(st.get("n_nan") or 0) > 0))
    allp = [(k, e) for k, e in a["paths"].items()] + [(k, e) for k, e in b["paths"].items()]
    for name, e in allp:
        for rep_i, rep in enumerate(e["reps"]):
            for i, m in enumerate(rep):
                bn = lbin(a["seq_lens"][i])
                cells[(name, route, bn, L)].append(dict(m, nan=(a["ref_stats"][i].get("n_nan") or 0) > 0))
                bad = m["rank"] is not None and m["rank"] < 480 and not m["tie_exact"]
                per_rep[(name, route)][rep_i][0] += bad
                per_rep[(name, route)][rep_i][1] += 1
        for x in e["rtr_col511_distinct"]:
            rtr[(name, route)][0] += x > 1
            rtr[(name, route)][1] += 1
        if name.startswith("v091"):
            for v in e["vs_live"]:
                for c, o, s in zip(v["col511_same"], v["order_same"], v["set_same"]):
                    q = vslive[(name, route)]
                    q[0] += c
                    q[1] += o
                    q[2] += s
                    q[3] += 1


def summarize(ms):
    n = len(ms)
    ranks = sorted(m["rank"] for m in ms if m["rank"] is not None)
    bad480 = sum(1 for m in ms if m["rank"] is not None and m["rank"] < 480 and not m["tie_exact"])
    bad480t = sum(1 for m in ms if m["rank"] is not None and m["rank"] < 480 and not m["tie_tol"])
    bad511 = sum(1 for m in ms if m["rank"] is not None and m["rank"] < 511 and not m["tie_exact"])
    lt256 = sum(1 for m in ms if m["rank"] is not None and m["rank"] < 256 and not m["tie_exact"])

    def q(p):
        return ranks[min(len(ranks) - 1, int(math.floor(p * (len(ranks) - 1))))] if ranks else None
    return dict(n=n, bad480=bad480 / n if n else None, bad480_tol=bad480t / n if n else None,
                bad511=bad511 / n if n else None, lt256=lt256 / n if n else None,
                rmin=ranks[0] if ranks else None, p10=q(0.10), med=q(0.5),
                set_mis_exact=sum(1 for m in ms if not m["set_ok_exact"]),
                set_mis_tol=sum(1 for m in ms if not m["set_ok_tol"]),
                dup=sum(m["dup"] for m in ms), invalid_c511=sum(1 for m in ms if m["rank"] is None))


def agg(keyf):
    g = collections.defaultdict(list)
    for k, v in cells.items():
        g[keyf(k)].extend(v)
    return {k: summarize(v) for k, v in g.items()}


PATH_ORDER = ["in-situ 0.9.1 (live)", "v091_auto", "v091_tiled_forced", "v091_fused_split16", "v070_dcp2_merge"]
PATH_DESC = {
    "in-situ 0.9.1 (live)": "0.9.1 live server output (route as captured)",
    "v091_auto": "0.9.1 replay, same batch, route by rows (= captured route)",
    "v091_tiled_forced": "0.9.1 replay, B12X_FUSED_INDEXER=0 (tiled) on fused-captured batches",
    "v091_fused_split16": "0.9.1 replay, tiled-captured batch split into 16-row halves (fused)",
    "v070_dcp2_merge": "0.7.0 replay: pools p->rank p%2, per-rank top-k (route by rows) + run_row_topk merge",
}
bins = [b for _, b in BINS]


def pct(x):
    return "n/a" if x is None else f"{100*x:.1f}%"


lines = []
A = lines.append
A("## Rows captured\n")
A(f"Capture files analysed: {len(files)} (v091 replay records {len(v091)}, v070 {len(v070)}).\n")
A("| capture route | rows | " + " | ".join(bins) + " | DSA layers |")
A("|---|---|" + "---|" * len(bins) + "---|")
for rt in sorted(nrows_route):
    A(f"| {rt} | {nrows_route[rt]} | " + " | ".join(str(nrows_route_bin[(rt, b)]) for b in bins)
      + f" | {sorted(layers_route[rt])} |")
A("")
A("## Main table: path x capture route (all lengths, all layers)\n")
A("Rows x repeats. bad480 = column-511 pool has fp64 reference rank < 480 and is not an exact boundary tie; "
  "(tol) = not within the fp32 evaluation bound of the 512th score either.\n")
A("| path | capture route | n | rank<480 non-tie | rank<480 (tol) | rank<511 non-tie | rank<256 | min | p10 | median | set mismatch exact / tol |")
A("|---|---|---|---|---|---|---|---|---|---|---|")
m1 = agg(lambda k: (k[0], k[1]))
for p in PATH_ORDER:
    for rt in ("tiled", "fused"):
        s = m1.get((p, rt))
        if s:
            A(f"| {p} | {rt} | {s['n']} | {pct(s['bad480'])} | {pct(s['bad480_tol'])} | {pct(s['bad511'])} | "
              f"{pct(s['lt256'])} | {s['rmin']} | {s['p10']} | {s['med']} | {s['set_mis_exact']} / {s['set_mis_tol']} |")
A("")
A("## Sensitivity: rows without NaN-byte pools only\n")
A("| path | capture route | n | rank<480 non-tie | median rank | set mismatch exact / tol |")
A("|---|---|---|---|---|---|")
gg = collections.defaultdict(list)
for k, v in cells.items():
    gg[(k[0], k[1])].extend(m for m in v if not m["nan"])
for p in PATH_ORDER:
    for rt in ("tiled", "fused"):
        if gg.get((p, rt)):
            s_ = summarize(gg[(p, rt)])
            A(f"| {p} | {rt} | {s_['n']} | {pct(s_['bad480'])} | {s_['med']} | {s_['set_mis_exact']} / {s_['set_mis_tol']} |")
A("")
A("## Path x capture route x length bin (pool count)\n")
A("Cell = rank<480 non-tie share (n rows x repeats); min / p10 / median rank below.\n")
m2 = agg(lambda k: (k[0], k[1], k[2]))
A("| path | route | " + " | ".join(bins) + " |")
A("|---|---|" + "---|" * len(bins))
for p in PATH_ORDER:
    for rt in ("tiled", "fused"):
        if (p, rt) not in m1:
            continue
        row = []
        for b in bins:
            s = m2.get((p, rt, b))
            row.append("—" if not s else f"{pct(s['bad480'])} (n={s['n']}; {s['rmin']}/{s['p10']}/{s['med']})")
        A(f"| {p} | {rt} | " + " | ".join(row) + " |")
A("")
A("## Path x capture route x DSA layer\n")
m3 = agg(lambda k: (k[0], k[1], k[3]))
alllayers = sorted({k[3] for k in cells})
A("| path | route | " + " | ".join(f"L{l}" for l in alllayers) + " |")
A("|---|---|" + "---|" * len(alllayers))
for p in PATH_ORDER:
    for rt in ("tiled", "fused"):
        if (p, rt) not in m1:
            continue
        row = []
        for l in alllayers:
            s = m3.get((p, rt, l))
            row.append("—" if not s else f"{pct(s['bad480'])} ({s['n']})")
        A(f"| {p} | {rt} | " + " | ".join(row) + " |")
A("")
A("## Run-to-run variation (replays, 3 repeats each)\n")
A("| path | capture route | rows whose col-511 pool differed across repeats | rank<480 non-tie share per repeat |")
A("|---|---|---|---|")
for p in PATH_ORDER[1:]:
    for rt in ("tiled", "fused"):
        if (p, rt) in rtr:
            x = rtr[(p, rt)]
            reps = per_rep[(p, rt)]
            A(f"| {p} | {rt} | {x[0]}/{x[1]} ({pct(x[0]/x[1])}) | "
              + ", ".join(pct(reps[i][0] / reps[i][1]) for i in sorted(reps)) + " |")
A("")
A("## In-situ (live) vs 0.9.1 replay of the same bytes\n")
A("| replay path | capture route | col-511 pool same as live | full 512 order same | selected set same | n |")
A("|---|---|---|---|---|---|")
for (p, rt), q in sorted(vslive.items()):
    A(f"| {p} | {rt} | {pct(q[0]/q[3])} | {pct(q[1]/q[3])} | {pct(q[2]/q[3])} | {q[3]} |")
A("")
A("## Score distribution near the boundary (fp64 reference, per captured row)\n")
A(f"Rows: {ties_rows}. Rows containing pools whose stored K bytes decode to NaN: {ties['rows_with_nan_pools']}. Rows with >1 pool exactly equal to the 512th score: {ties['eq_t512>1']}; "
  f"with >1 pool within the fp32 bound of it: {ties['tol_t512>1']}; any exactly-equal scores among the top 520: "
  f"{ties['dup_in_top520']}; any exact-zero score among all pools: {ties['zero_any']}; zeros inside the top 512: "
  f"{ties['zero_in_top512']}. Relative gap |s512 - s513| / |s512|: median "
  f"{statistics.median(gaps) if gaps else 'n/a':.3g}, p10 {sorted(gaps)[len(gaps)//10] if gaps else 'n/a':.3g}.\n")
# index meta
if idx:
    c = collections.Counter((m["tag"], m["has_prefill"], m["route"], m["captured"]) for m in idx)
    A("## Probe index (every armed decode-branch call)\n")
    A("| phase tag | mixed prefill+decode step | route | captured | calls |")
    A("|---|---|---|---|---|")
    for k in sorted(c):
        A(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {c[k]} |")
    rows_hist = collections.Counter((m["rows"], m["num_decodes"], m["next_n"], m["route"]) for m in idx)
    A("\nrows / requests / next_n / route: " + ", ".join(f"{k[0]}/{k[1]}/{k[2]}/{k[3]}: {v}" for k, v in sorted(rows_hist.items())))
    A("")

open(f"{OUT}/analysis.md", "w").write("\n".join(lines) + "\n")
json.dump(dict(main={f"{k[0]}|{k[1]}": v for k, v in m1.items()},
               bins={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in m2.items()},
               layers={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in m3.items()},
               ties=dict(ties), ties_rows=ties_rows), open(f"{OUT}/analysis.json", "w"), indent=1)
print("\n".join(lines))
