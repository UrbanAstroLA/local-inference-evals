"""Follow-up a2, check 1: are the two B1 arms' equal pooled gross counts (172/27,973) a coincidence or a duplicated /
aliased comparison? Pure python over out/units.jsonl and out/dump_meta.json (from f1_units.py). -> out/f1_summary.json"""
import json, collections, math
G = 0.25
U = [json.loads(l) for l in open("<runs>/hook-validation/followup-a2/out/units.jsonl")]
M = json.load(open("<runs>/hook-validation/followup-a2/out/dump_meta.json"))
S = {}
key = lambda u: (u["tid"], u["rank"], u["layer"], u["pool"])
gr = lambda u, p: u.get(p) is not None and u[p] > G
matched = [u for u in U if u.get("b1~b1b") is not None and u.get("reb~reb2") is not None]
S["units_total"] = len(U)
S["units_matched"] = len(matched)
for p in ("b1~reb", "b1b~reb", "b1~b1b", "reb~reb2", "b1~reb2", "b1b~reb2", "d2~reb"):
    S.setdefault("pooled_gross", {})[p] = sum(gr(u, p) for u in matched)
Gb1 = {key(u) for u in matched if gr(u, "b1~reb")}
Gb1b = {key(u) for u in matched if gr(u, "b1b~reb")}
inter = Gb1 & Gb1b
S["overlap"] = dict(b1=len(Gb1), b1b=len(Gb1b), both=len(inter), b1_only=len(Gb1 - Gb1b), b1b_only=len(Gb1b - Gb1),
                    jaccard=len(inter) / len(Gb1 | Gb1b))
a, b = len(Gb1 - Gb1b), len(Gb1b - Gb1)
n = a + b
S["exchange_null_P_equal_totals"] = (math.comb(n, n // 2) / 2 ** n) if n % 2 == 0 else 0.0
S["exchange_null_note"] = ("under exchangeability of the arm labels on each discordant unit (each of the %d units gross "
                           "in exactly one arm is equally likely to be b1's or b1b's), P(totals equal) = C(n,n/2)/2^n" % n)
# per fixture, per fixture x layer
pf, pfl = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
for u in matched:
    for d, k in ((pf, u["tid"]), (pfl, f'{u["tid"]}|r{u["rank"]}|L{u["layer"]}')):
        d[k]["n"] += 1
        for p, nm in (("b1~reb", "b1"), ("b1b~reb", "b1b"), ("b1~b1b", "N_dd"), ("reb~reb2", "N_pp"),
                      ("b1~reb2", "b1_vs_reb2"), ("b1b~reb2", "b1b_vs_reb2")):
            d[k][nm] += int(gr(u, p))
        if key(u) in inter:
            d[k]["both"] += 1
S["per_fixture"] = {k: dict(v) for k, v in sorted(pf.items())}
S["per_fixture_layer"] = {k: dict(v) for k, v in sorted(pfl.items())}
# byte identity between the two decode arms and between the two rebuilds
for p in ("b1~b1b", "reb~reb2", "b1~reb", "b1b~reb", "d2~reb"):
    v = [u["eq:" + p] for u in matched if u.get("eq:" + p) is not None]
    S.setdefault("byte_identical_frac", {})[p] = sum(v) / len(v)
# the b1-vs-b1b relation at gross units
for nm, s in (("both", inter), ("b1_only", Gb1 - Gb1b), ("b1b_only", Gb1b - Gb1)):
    v = sorted(u["b1~b1b"] for u in matched if key(u) in s)
    S.setdefault("rel_b1_vs_b1b_at_gross", {})[nm] = dict(n=len(v), median=v[len(v) // 2] if v else None,
                                                          min=v[0] if v else None, max=v[-1] if v else None,
                                                          byte_equal=sum(u["eq:b1~b1b"] for u in matched if key(u) in s))
# page maxima (as IDX-page: (tid, rank, layer, pool//64), pages with >= 16 pools)
for p in ("b1~reb", "b1b~reb"):
    pg = collections.defaultdict(list)
    for u in U:
        if u.get(p) is not None:
            pg[(u["tid"], u["rank"], u["layer"], u["pool"] // 64)].append(u[p] > G)
    j = {k: v for k, v in pg.items() if len(v) >= 16}
    w = sorted(((sum(v) / len(v), sum(v), len(v), k) for k, v in j.items()), reverse=True)[:3]
    S.setdefault("pages", {})[p] = dict(judged=len(j), worst=[(round(f, 4), g, n, list(k)) for f, g, n, k in w])
# stale-block check: is a b1 / b1b gross pool byte-identical to the d2 arm's pool (d2 ran first; b1 reused d2's blocks)?
S["block_reuse"] = {}
for tid in sorted({u["tid"] for u in U}):
    S["block_reuse"][tid] = {a: M.get(f"V4-{a}-{tid}|r0", M.get(f"V4n-{a}-{tid}|r0", {})).get("g0_ids_sha")
                             for a in ("b1", "b1b", "d2", "rebuild", "rebuild2")}
# dump identities
S["dumps"] = {k: dict(sha256=v["sha256"], P=v["P"], F=v["F"], step=v["step"], frontier=v["frontier_pos"], range=v["range"],
                      g0=v["g0_ids_sha"], seed=v["client"]["seed"], batch=v["client"]["batch"])
              for k, v in M.items()}
shas = [v["sha256"] for v in M.values()]
S["dump_sha_unique"] = len(set(shas)) == len(shas)
S["arm_dumps_distinct"] = {tid: len({M[f"V4-b1-{tid}|r{r}"]["sha256"] for r in (0, 1)} |
                                    {M[f"V4n-b1b-{tid}|r{r}"]["sha256"] for r in (0, 1)}) == 4
                           for tid in sorted({u["tid"] for u in U})}
json.dump(S, open("<runs>/hook-validation/followup-a2/out/f1_summary.json", "w"), indent=1)
print(json.dumps({k: S[k] for k in ("units_total", "units_matched", "pooled_gross", "overlap", "exchange_null_P_equal_totals",
                                     "byte_identical_frac", "rel_b1_vs_b1b_at_gross", "pages", "dump_sha_unique",
                                     "arm_dumps_distinct", "block_reuse")}, indent=1))
print("per fixture:")
for k, v in S["per_fixture"].items():
    print(" ", k, v)
