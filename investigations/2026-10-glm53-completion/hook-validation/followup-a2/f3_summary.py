"""Follow-up a2, check 3: characterise the fixture flags (be2e9e349395 outside its own matched null in both B1 arms;
376073921446 arm asymmetry) against every other fixture, the in-session nulls, the attempt-1 session (same windows,
same engine code) and the attempt-1 prefill-vs-prefill null of diagnosis-a1 (out/h4c_nulls.json, V3ref builds of
the same traces at different build lengths). Pure python. -> out/f3_summary.json"""
import json, collections
G = 0.25
HV = "<runs>/hook-validation/"
D = HV + "followup-a2/out/"
U = {(u["tid"], u["rank"], u["layer"], u["pool"]): u for u in map(json.loads, open(D + "units.jsonl"))}
X = {(u["tid"], u["rank"], u["layer"], u["pool"]): u for u in map(json.loads, open(D + "cross_units.jsonl"))}
assert set(U) == set(X)
for k, x in X.items():
    U[k].update({kk: v for kk, v in x.items() if "~" in kk})
# attempt-1 prefill-vs-prefill (V3ref builds of the same trace; diagnosis-a1), max rel per (tid, layer, pool)
P = json.load(open(HV + "diagnosis-a1/out/h4c_nulls.json"))["P"]
pp = collections.defaultdict(list)
for x in P:
    pp[(x["g"].rsplit("-", 1)[-1], x["layer"], x["pool"])].append(x["rel"])
ppmax = {k: max(v) for k, v in pp.items()}
g = lambda u, p: u.get(p) is not None and u[p] > G
CMP = {"b1": "b1~reb", "b1b": "b1b~reb", "N_dd": "b1~b1b", "N_pp": "reb~reb2", "a1_b1": "a1b1~a1reb",
       "a1b1_vs_a2reb": "a1b1~reb", "a1b1_vs_a2b1": "a1b1~b1", "a1b1_vs_a2b1b": "a1b1~b1b"}
S = {"cross_session_prefill_identity": {}, "per_fixture": {}, "flagged": {}}
by_t = collections.defaultdict(list)
for k, u in U.items():
    by_t[u["tid"]].append(u)
for t, us in sorted(by_t.items()):
    eq = [u["eq:a1reb~reb"] for u in us]
    S["cross_session_prefill_identity"][t] = sum(eq) / len(eq)
    row = dict(n=len(us), **{nm: sum(g(u, p) for u in us) for nm, p in CMP.items()})
    Gb1 = {(u["rank"], u["layer"], u["pool"]) for u in us if g(u, "b1~reb")}
    Gb1b = {(u["rank"], u["layer"], u["pool"]) for u in us if g(u, "b1b~reb")}
    Ga1 = {(u["rank"], u["layer"], u["pool"]) for u in us if g(u, "a1b1~a1reb")}
    row["b1&b1b"] = len(Gb1 & Gb1b)
    row["b1&a1"] = len(Gb1 & Ga1)
    row["b1b&a1"] = len(Gb1b & Ga1)
    row["all3"] = len(Gb1 & Gb1b & Ga1)
    row["union3"] = len(Gb1 | Gb1b | Ga1)
    # prefill fragility at the gross pools: in-session N_pp rel and attempt-1 V3ref max rel, vs non-gross base rate
    def frag(sel):
        a = [u for u in us if sel(u)]
        npp = [u["reb~reb2"] for u in a]
        j = [ppmax[(t, u["layer"], u["pool"])] for u in a if (t, u["layer"], u["pool"]) in ppmax]
        return dict(n=len(a), Npp_gt_0_25=sum(x > G for x in npp), Npp_gt_0_10=sum(x > 0.10 for x in npp),
                    Npp_median=sorted(npp)[len(npp) // 2] if npp else None, v3ref_joined=len(j),
                    v3ref_gt_0_25=sum(x > G for x in j), v3ref_median=sorted(j)[len(j) // 2] if j else None)
    row["at_gross_b1_or_b1b"] = frag(lambda u: g(u, "b1~reb") or g(u, "b1b~reb"))
    row["at_nongross"] = frag(lambda u: not (g(u, "b1~reb") or g(u, "b1b~reb")))
    lay = collections.Counter(u["layer"] for u in us if g(u, "b1~reb") or g(u, "b1b~reb"))
    row["layers_gross_b1_or_b1b"] = dict(sorted(lay.items()))
    pools = {u["pool"] for u in us if g(u, "b1~reb") or g(u, "b1b~reb")}
    row["distinct_pools_gross"] = len(pools)
    S["per_fixture"][t] = row
# pooled layer profile of gross units (both arms) for comparison
lay = collections.Counter(u["layer"] for u in U.values() if g(u, "b1~reb") or g(u, "b1b~reb"))
n_lay = collections.Counter(u["layer"] for u in U.values())
S["pooled_layer_gross_rate_b1_or_b1b"] = {L: round(lay[L] / n_lay[L], 5) for L in sorted(n_lay)}
# unit lists of the flagged fixtures
for t, crit in (("be2e9e349395", lambda u: g(u, "b1~reb") or g(u, "b1b~reb")),
                ("376073921446", lambda u: g(u, "b1~reb") or g(u, "b1b~reb") or g(u, "b1~b1b"))):
    rows = []
    for u in sorted(by_t[t], key=lambda u: (u["pool"], u["rank"], u["layer"])):
        if crit(u):
            rows.append(dict(rank=u["rank"], layer=u["layer"], pool=u["pool"], o=u["o"], positions=[u["first"], u["first"] + 3],
                             before_frontier=u["before_frontier"],
                             **{nm: round(u[p], 3) for nm, p in CMP.items() if u.get(p) is not None},
                             v3ref_max=round(ppmax[(t, u["layer"], u["pool"])], 3) if (t, u["layer"], u["pool"]) in ppmax else None,
                             norm_reb=round(u["norm"]["reb"], 2), scale_reb=u["scale"]["reb"]))
    S["flagged"][t] = rows
json.dump(S, open(D + "f3_summary.json", "w"), indent=1)
print("prefill rebuild, attempt 1 vs attempt 2 byte-identical fraction:", S["cross_session_prefill_identity"])
for t, r in S["per_fixture"].items():
    print(t, {k: v for k, v in r.items() if k not in ("at_gross_b1_or_b1b", "at_nongross", "layers_gross_b1_or_b1b")})
    print("    at gross:", r["at_gross_b1_or_b1b"], "\n    non-gross:", r["at_nongross"], "\n    layers:", r["layers_gross_b1_or_b1b"])
print("pooled layer rate:", S["pooled_layer_gross_rate_b1_or_b1b"])
for t, rows in S["flagged"].items():
    print(t, len(rows))
    for r in rows:
        print("  ", r)
