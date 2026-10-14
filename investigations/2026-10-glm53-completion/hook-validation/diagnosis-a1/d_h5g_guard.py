"""H5g: guard counters per rank and per phase (the frozen analysis keyed polls by a 'pid' field the polls do not
carry, so it read only the single last poll of one rank). CPU only; host python (json)."""
import json, glob, collections
RUN = "<runs>/hook-validation/run-a1"
polls = []
for f in glob.glob(RUN + "/mapguard/mg_p*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r.get("kind") == "poll":
            polls.append(r)
polls.sort(key=lambda r: r["t"])
CL = [json.loads(l) for l in open(RUN + "/client.jsonl")]
phases = [(r["tag"], r["t0"], r["t1"]) for r in CL if r.get("kind") == "phase"]
dumps = [(r["tag"], r["t"]) for r in CL if r.get("kind") == "dump"]
print("polls", len(polls), "by rank", collections.Counter(p["rank"] for p in polls), "pid field present:",
      any("pid" in p for p in polls))
SITES = ("tail_w_stash", "tail_w_seed", "idx_w_dec", "idx_w_pf", "mla_w", "pos")
CNT = ("mism", "null", "pos_contig", "pos_bounds", "s_mism", "d7_fallback", "foreign", "stale", "oor", "missing", "unexp")
print("sites in counters:", sorted(polls[-1]["counters"].keys()))
last = {}
for p in polls:
    last[p["rank"]] = p
for rk, p in sorted(last.items()):
    print("rank", rk, "final poll t", p["t"], p["reason"])
    for s in sorted(p["counters"]):
        nz = {c: v for c, v in p["counters"][s].items() if v and c not in ("exec", "armed", "checked", "written", "spec", "pc_checked", "s_events", "s_verified")}
        print("   ", s, "exec", p["counters"][s].get("exec"), "checked", p["counters"][s].get("checked"), "nonzero:", nz)
# per-phase deltas per rank: attribute each counter increase to the phase whose window contains the poll time
def phase_at(t):
    for tag, t0, t1 in phases:
        if t0 <= t <= t1 + 8:
            return tag
    return "between/after:" + max([tg for tg, tt in dumps if tt <= t] or ["start"], key=lambda x: 0)
prev = {}
acc = collections.defaultdict(lambda: collections.Counter())
for p in polls:
    rk = p["rank"]
    for s in SITES:
        for c in ("mism", "null", "pos_bounds", "pos_contig"):
            v = p["counters"].get(s, {}).get(c, 0)
            d = v - prev.get((rk, s, c), 0)
            prev[(rk, s, c)] = v
            if d:
                acc[(rk, s, c)][phase_at(p["t"])] += d
for k in sorted(acc):
    print(k, dict(acc[k]))
# pos_bounds detail: look for records mentioning pos_bounds
ex = []
for p in polls:
    for rec in p.get("records", []) or []:
        if "pos_bounds" in json.dumps(rec) or "bounds" in json.dumps(rec):
            ex.append((p["rank"], p["t"], rec))
print("records mentioning bounds:", len(ex))
for e in ex[:8]:
    print("  ", e[0], e[1], json.dumps(e[2])[:600])
json.dump({"final_by_rank": {str(k): {s: {c: v["counters"].get(s, {}).get(c) for c in CNT} for s in SITES} for k, v in last.items()},
           "per_phase": {"%s|%s|%s" % k: dict(v) for k, v in acc.items()}},
          open("<runs>/hook-validation/diagnosis-a1/out/h5g_guard.json", "w"), indent=1)
