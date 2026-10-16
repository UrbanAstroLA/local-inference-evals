import json, glob, os, collections
D = os.path.dirname(os.path.abspath(__file__))
lines = ["| file | engine cfg | arm | accept | conc | pools | wrong | % wrong | per-seed % | shadow agrees | wrong-pool classes (pooled over seeds) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for f in sorted(glob.glob(os.path.join(D, "out", "*.json"))):
    d = json.load(open(f)); c = d["cfg"]
    g = collections.defaultdict(list)
    for r in d["results"]:
        g[(r["arm"], r["conc"])].append(r)
    for (arm, conc), rs in sorted(g.items()):
        P = sum(r["pools"] for r in rs); W = sum(r["wrong"] for r in rs)
        cls = collections.Counter()
        for r in rs:
            cls.update(r["wrong_by_shadow_class"])
        agree = all(r["shadow_agrees_with_kernel"].split("/")[0] == r["shadow_agrees_with_kernel"].split("/")[1] for r in rs)
        lines.append(f"| {os.path.basename(f)} | {d['engine']}: DCP {c['dcp']}, ring {c['ring']}, K={c['K']} | {arm} | {d['accept']} | {conc} | {P} | {W} | {100*W/P:.1f} | "
                     + "/".join(f"{100*r['wrong']/r['pools']:.0f}" for r in rs) + f" | {'all' if agree else 'NO'} | "
                     + ", ".join(f"{k} {v}" for k, v in cls.most_common()) + " |")
open(os.path.join(D, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
