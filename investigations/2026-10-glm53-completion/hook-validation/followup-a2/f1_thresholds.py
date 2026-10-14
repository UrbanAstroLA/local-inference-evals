"""Follow-up a2, check 1 side test: do the two B1 arms' pooled counts stay equal away from the frozen 0.25 threshold?
A structural (by-construction) equality would hold at every threshold; a coincidence would not. -> out/f1_thresholds.json"""
import json
D = "<runs>/hook-validation/followup-a2/out/"
U = [json.loads(l) for l in open(D + "units.jsonl")]
out = {}
for th in (0.10, 0.15, 0.20, 0.22, 0.24, 0.25, 0.26, 0.28, 0.30, 0.35, 0.50, 1.00):
    out[str(th)] = {p: sum(u[p] > th for u in U) for p in ("b1~reb", "b1b~reb", "b1~reb2", "b1b~reb2")}
json.dump(out, open(D + "f1_thresholds.json", "w"), indent=1)
for k, v in out.items():
    print(k, v)
