"""Per-layer gross fraction (rel L2 > 0.25) and q999: decode-vs-prefill comparisons (H4c, H5b, H5e) beside the
measured nulls (prefill vs prefill; decode vs decode at batch 8 vs batch 1; decode vs itself over time)."""
import json, collections, numpy as np
H5 = [json.loads(l) for l in open("/d/out/h5_pools.jsonl")]
H4 = json.load(open("/d/out/h4c_idx.json")); NL = json.load(open("/d/out/h4c_nulls.json"))
DN = [x for x in json.load(open("/d/out/h5_decode_null.json")) if x["cmp"] == "b1_vs_solo"]
sets = {"H5b B1 dec-vs-rebuild": [(r["layer"], r["rel"]) for r in H5 if r["arm"] == "b1"],
        "H5e solo dec-vs-rebuild": [(r["layer"], r["rel"]) for r in H5 if r["arm"] == "d2solo"],
        "H4c V3 dec-vs-prefill": [(r["layer"], r["rel"]) for r in H4],
        "NULL prefill-vs-prefill": [(r["layer"], r["rel"]) for r in NL["P"]],
        "NULL dec(b8)-vs-dec(b1)": [(r["layer"], r["rel"]) for r in DN],
        "NULL dec-vs-same-dec": [(r["layer"], r["rel"]) for r in NL["D"]],
        "H5d D2 dec-vs-rebuild": [(r["layer"], r["rel"]) for r in H5 if r["arm"] == "d2"]}
layers = sorted({L for v in sets.values() for L, _ in v})
tab = {}
hdr = "%-26s %7s %7s %7s %7s " % ("comparison", "n", "gross", "q50", "q999") + " ".join("L%-5d" % L for L in layers)
print(hdr)
for k, v in sets.items():
    a = np.array([x for _, x in v]); byL = collections.defaultdict(list)
    for L, x in v: byL[L].append(x)
    row = {L: float(np.mean(np.array(byL[L]) > .25)) if byL[L] else None for L in layers}
    tab[k] = dict(n=int(a.size), gross=float(np.mean(a > .25)), q50=float(np.median(a)), q999=float(np.quantile(a, .999)), by_layer=row)
    print("%-26s %7d %7.4f %7.4f %7.3f " % (k, a.size, np.mean(a > .25), np.median(a), np.quantile(a, .999)) +
          " ".join(("%.4f" % row[L]) if row[L] is not None else "  -   " for L in layers))
json.dump(tab, open("/d/out/null_table.json", "w"), indent=1)
