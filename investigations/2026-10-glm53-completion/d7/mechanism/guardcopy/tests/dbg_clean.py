import sys, os
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_mapguard as T
mg = T.mg
sc = T.scenario(max_steps=12)
for st in sc["steps"]:
    print("step", st["step"], "nd", st["nd"], "nreq", st["nreq"], "ntok", st["ntok"], "pad", st["pad"], st["viol"])
    for r in st["recs"][:3]:
        d = dict(zip(mg.REC_COLS, r)); d["flags"] = mg.flag_names(d["flags"]); d["site"] = mg.SITES[d["site"]]
        print("   ", d)
g = sc["guard"]
print("ring", list(g.ring.items())[:3], "errors", mg.S.errors)
print("ring_step nonneg", int((g.dev.ring_step >= 0).sum()))
led = g.led
for rid, st in led.reqs.items():
    print(rid, st["tag"], [len(x) for x in st["blocks"]], st["blocks"][0][:6], st["blocks"][1])
