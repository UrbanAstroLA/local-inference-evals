#!/usr/bin/env python3
"""Post-inspection (diagnosis-fc1, CPU only): index-K page / early-layer structure on C1 and on the prefill-vs-prefill
null, using the FROZEN fc_common functions unchanged. Answers: are the 8 judged pages > 25 % gross and the 3 early-layer
gross pools that make G1-idx-struct fail on C1 vs LW1 specific to the decode window, or present in LW1 vs LW2 too?

usage (CPU container, --network none, no --gpus):  idx_pages.py RUN_DIR FC_ROOT OUT_JSON
"""
import collections
import json
import sys

import numpy as np

RUN, FC, OUT = sys.argv[1:4]
sys.path.insert(0, FC + "/analysis")
import fc_common as fc  # noqa: E402
import fc_geometry as geo  # noqa: E402

G = geo.load(FC)
run = fc.Run(RUN)
res = {}
for pair in (("C1", "LW1"), ("C1", "LW2"), ("LW1", "LW2")):
    rows_by, per_fx = {}, {}
    for n in G["order"]:
        lo, hi = G["fx"][n]["ref_pools"]
        r = fc.compare_pools(run, f"{pair[0]}-{n}", f"{pair[1]}-{n}", lo, hi)
        rows_by[f"{pair[0]}-{n}"] = r
        vals = list(r.values())
        g = [(k, v) for k, v in r.items() if v > fc.GROSS]
        per_fx[n] = dict(n=len(vals), gross_n=len(g), gross=fc.gross(vals),
                         early_gross=[(k[0], k[1], k[2], round(v, 3)) for k, v in g if 3 <= k[1] <= fc.EARLY_IDX_LAYERS],
                         gross_by_layer=dict(collections.Counter(str(k[1]) for k, v in g).most_common(8)),
                         rel_q=[fc.q(vals, p) for p in (0.5, 0.9, 0.99, 0.999)])
    ps = fc.page_stats(rows_by)
    # all judged pages over the limit, with the page's absolute position range and whether it straddles the window
    pg = collections.defaultdict(list)
    for tag, rows in rows_by.items():
        for (rank, layer, pool), rel in rows.items():
            pg[(tag, rank, layer, pool // 64)].append(rel > fc.GROSS)
    over = []
    for k, v in pg.items():
        if len(v) >= fc.PAGE_MINP and np.mean(v) > fc.PAGE_MAX_GROSS:
            n = k[0].split("-", 1)[1]
            P = G["fx"][n]["P"]
            over.append(dict(tag=k[0], rank=k[1], layer=k[2], page=k[3], frac=round(float(np.mean(v)), 4), n=len(v),
                             pos=[4 * 64 * k[3], 4 * 64 * (k[3] + 1)], o=[4 * 64 * k[3] - P, 4 * 64 * (k[3] + 1) - P]))
    eg, en = fc.early_gross(rows_by)
    res["%s_vs_%s" % pair] = dict(page_stats=ps, pages_over=sorted(over, key=lambda d: -d["frac"]),
                                  early=(eg, en), per_fixture=per_fx,
                                  pooled=dict(n=sum(d["n"] for d in per_fx.values()),
                                              gross_n=sum(d["gross_n"] for d in per_fx.values())))
json.dump(res, open(OUT, "w"), indent=1, default=str)
for k, v in res.items():
    print(k, "pooled", v["pooled"], "pages", {a: b for a, b in v["page_stats"].items() if a != "worst"},
          "early", v["early"])
    for d in v["pages_over"]:
        print("   over:", d)
