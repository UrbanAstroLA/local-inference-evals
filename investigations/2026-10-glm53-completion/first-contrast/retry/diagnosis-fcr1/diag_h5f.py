#!/usr/bin/env python3
"""POST-HOC diagnostic (after data, not part of the frozen analysis): where the H5f-fc clean-labelled gross pools sit
(layer, window offset), against the same breakdown for C runs. CPU only."""
import sys, os, collections, json
sys.path.insert(0, "/fc/analysis")
sys.path.insert(0, "/fc/retry/analysis")
import fc_common as fc, fcr_geometry as geo
rd = "/run"
G = geo.load("/fc")
run = fc.Run(rd)
out = {}
for ph in ["D1", "D2", "D3", "C1", "C2", "C3"]:
    bylayer = collections.defaultdict(lambda: [0, 0, 0, 0])  # clean_n, clean_gross, corrupt_n, corrupt_gross
    byoff = collections.defaultdict(lambda: [0, 0])
    clean_rel = []
    for n in G["order"]:
        lo, hi = G["fx"][n]["ref_pools"]
        rows = fc.compare_pools(run, f"{ph}-{n}", f"LW1-{n}", lo, hi)
        a = run.req(f"{ph}-{n}")
        if a is None: continue
        pc = a["pool_class"].numpy(); P = G["fx"][n]["P"]
        for (rank, L, pool), v in rows.items():
            k = pool - P // 4
            if not (0 <= k < pc.size): continue
            g = v > fc.GROSS
            if pc[k] == 1:
                bylayer[L][0] += 1; bylayer[L][1] += g; clean_rel.append((n, L, int(k), round(float(v), 4)))
                byoff[min(k // 16, 9)][0] += 1; byoff[min(k // 16, 9)][1] += g
            elif pc[k] == 2:
                bylayer[L][2] += 1; bylayer[L][3] += g
    out[ph] = dict(bylayer={str(k): v for k, v in sorted(bylayer.items())},
                   byoff={str(k): v for k, v in sorted(byoff.items())},
                   clean_gross_list=[x for x in clean_rel if x[3] > fc.GROSS])
    print(ph, "layer: [clean_n, clean_gross, corrupt_n, corrupt_gross]")
    for k, v in sorted(bylayer.items()): print("  L%-3s %s" % (k, v))
    print("  clean gross pools (fixture, layer, window pool, rel):", out[ph]["clean_gross_list"])
json.dump(out, open("/out/diag_h5f.json", "w"), indent=1)
