import json, sys, collections
for f in sys.argv[1:]:
    d=json.load(open(f))
    print("==", d["scenario"]["name"], "total_s", d["total_s"])
    for p in d["phases"]:
        rq=p.get("requests",[])
        if p["kind"] not in ("run","ps","touch"): continue
        print("  %-6s lead %-4s cls %s  last-window margins %s" % (p["phase"], p.get("lead"), dict(collections.Counter(r["cls"] for r in rq)),
              [(r["fixture"], r.get("margin_last_window")) for r in rq]))
