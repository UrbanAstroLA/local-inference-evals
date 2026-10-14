"""Follow-up a2, check 1: per fixture x layer gross counts, b1 / b1b / both (vs the prefill rebuild), plus N_dd and
N_pp, as a markdown table. Layers 3-15 are zero in every column and are folded into one column. -> out/f1_table.md"""
import json
D = "<runs>/hook-validation/followup-a2/out/"
S = json.load(open(D + "f1_summary.json"))
pfl = S["per_fixture_layer"]
deep = [19, 23, 27, 31, 35, 39, 43]
rk = {19: 0, 23: 0, 27: 1, 31: 1, 35: 1, 39: 1, 43: 1}
L = ["| Fixture | n per layer | L3-15 | " + " | ".join(f"L{x}" for x in deep) + " | Total b1 / b1b (both) | N_dd | N_pp |",
     "|---|---:|---:|" + "---:|" * len(deep) + "---:|---:|---:|"]
for t, v in S["per_fixture"].items():
    early = sum(pfl[f"{t}|r0|L{x}"].get("b1", 0) + pfl[f"{t}|r0|L{x}"].get("b1b", 0) for x in (3, 7, 11, 15))
    cells = []
    for x in deep:
        c = pfl[f"{t}|r{rk[x]}|L{x}"]
        cells.append(f"{c.get('b1', 0)} / {c.get('b1b', 0)} ({c.get('both', 0)})")
    L.append(f"| {t} | {pfl[f'{t}|r0|L19']['n']} | {early} | " + " | ".join(cells) +
             f" | **{v['b1']} / {v['b1b']}** ({v.get('both', 0)}) | {v['N_dd']} | {v['N_pp']} |")
tot = lambda k: sum(v.get(k, 0) for v in S["per_fixture"].values())
lay = []
for x in deep:
    b = sum(pfl[f"{t}|r{rk[x]}|L{x}"].get("b1", 0) for t in S["per_fixture"])
    bb = sum(pfl[f"{t}|r{rk[x]}|L{x}"].get("b1b", 0) for t in S["per_fixture"])
    both = sum(pfl[f"{t}|r{rk[x]}|L{x}"].get("both", 0) for t in S["per_fixture"])
    lay.append(f"{b} / {bb} ({both})")
L.append(f"| **Pooled** | {sum(pfl[f'{t}|r0|L19']['n'] for t in S['per_fixture']):,} | 0 | " + " | ".join(lay) + f" | **{tot('b1')} / {tot('b1b')}** ({tot('both')}) | {tot('N_dd')} | {tot('N_pp')} |")
open(D + "f1_table.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
