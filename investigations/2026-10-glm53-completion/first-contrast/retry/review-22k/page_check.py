import sys, json
sys.path.insert(0, "/fc/analysis"); sys.path.insert(0, "/fc/retry/analysis")
import fc_common as fc, fcr_geometry as geo, sentinel as sn
G = geo.load("/fc"); run = fc.Run("/run")
n = "M4"; P = G["fx"][n]["P"]; lo, hi = G["fx"][n]["ref_pools"]
for t in ("C1", "C2"):
    W = run.win_dump(f"{t}-{n}")
    for rank in sorted(W):
        a = run.load(W[rank]); ids, bpk = sn.g0_ids(a)
        pg = lambda pool: (ids[((4*pool)//4352)*bpk] // bpk) * 17 + ((pool) % 1088)//64
        m = {}
        for pool in range(lo, hi): m.setdefault(pg(pool), []).append(pool - P//4)
        print(t, "rank", rank, "window pools 0-8 on page", sorted({pg(P//4+k) for k in range(9)}),
              "| pages and window-pool ranges:", {p: (v[0], v[-1]) for p, v in sorted(m.items())})
