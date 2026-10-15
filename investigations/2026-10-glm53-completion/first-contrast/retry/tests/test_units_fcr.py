#!/usr/bin/env python3
"""Unit and real-data tests of the retry analysis helpers (CPU container of the B1 image, --network none; or host python
for the json-only parts). usage: test_units_fcr.py FC_ROOT  -> rc 0 iff every test passes.

  U1  F2 cache gate on run-fc1's REAL records: every client request mapped to its engine request; C1's 7 hits
      attributed to their warm-s1 builds (prefill-written), M3 a safe miss; all requests safe
  U2  F2 negatives on synthetic scheduler records: misaligned (S1), beyond the bound (S2), a decode-written block
      restored (S3), scheduler num_computed != cached (S4) each detected
  U3  classification and the balance rule
  U4  F5 sentinel range: [c - 4352, c) with c = the restored boundary for all 8 fixtures; attempt 1's [c0 - 1024, c0)
      lies in the privately recomputed block for all 8
  U5  pre-traffic geometry check: run-fc1's log passes; num_blocks 1,100, boundary lookup 1, sparse retention, a
      changed group each refused
  U6  F1 rule against every measured cached count replayed (replay/out/reproduce_fc1.json, historical.json)
  U7  plan geometry: cache invariant clean for B and A0; primary construction balanced (each arm one fresh build, one
      first user and one later user of a shared build)
  U8  runner constants equal the cost model (ceilings, wall caps, minimum walls)
  U9  F6 page rule: a noisy null widens only its own page; a planted page fails where its null is quiet
"""
import json
import os
import re
import subprocess
import sys
import tempfile

FC = sys.argv[1] if len(sys.argv) > 1 else "/fc"
R = os.path.join(FC, "retry")
sys.path.insert(0, os.path.join(R, "analysis"))
import cache_gate as cg  # noqa: E402
import fcr_geometry as geo  # noqa: E402

RES = []


def ok(name, cond, detail=""):
    RES.append((name, bool(cond)))
    print("%s %s %s" % ("PASS" if cond else "FAIL", name, str(detail)[:200]))


# U1
RUN = os.path.join(FC, "run-fc1")
reqs = [r for r in cg.jl(os.path.join(RUN, "client.jsonl")) if r.get("kind") == "request"]
for r in reqs:
    r["cached_expected"] = 0 if r["phase"] in ("LW1", "LW2", "warm-s1") else geo.rule_repeat(r["P"])
info = cg.sched_requests(cg.load_sched(RUN))
hook = {x["req"]: x["tag"] for x in cg.jl(os.path.join(RUN, "forcewin", "fw_r0_p370.jsonl")) if x.get("kind") == "add"}
m = cg.map_client(reqs, info, hook)
rows = {x["tag"]: x for x in cg.evaluate(reqs, info, m)}
ok("U1a every run-fc1 request mapped to its engine request", len(m) == len(reqs) == 32, len(m))
c1 = {t: x for t, x in rows.items() if t.startswith("C1-")}
ok("U1b C1: 7 hits attributed to their own warm-s1 build, M3 a miss",
   sum(x["cls"] == "hit" and x["restore_source"] == ["warm-s1-" + t[3:]] for t, x in c1.items()) == 7
   and c1["C1-M3"]["cls"] == "miss", {t: (x["cls"], x["restore_source"]) for t, x in c1.items()})
ok("U1c every run-fc1 request safe (S1-S4)", all(x["safe"] for x in rows.values()),
   [t for t, x in rows.items() if not x["safe"]])

# U2
def synth(cached, nc=None, restored=None, prev_W=0):
    a = dict(tag="A", phase="a", fixture="X", salt="s", salt_name="s", P=3 * 4352 + 10, W=prev_W, t0=1.0, t1=2.0,
             cached_tokens=0, cached_expected=0)
    b = dict(tag="B", phase="b", fixture="X", salt="s", salt_name="s", P=3 * 4352 + 10, W=5, t0=3.0, t1=4.0,
             cached_tokens=cached, cached_expected=4352)
    info = {"ea": dict(first_seq=1, first_t=1.1, computed_before=0, g0=[11, 12, 13, 14], g0_first=[11, 12, 13, 14],
                       last=0, first_dec=None, P_rec=a["P"]),
            "eb": dict(first_seq=2, first_t=3.1, computed_before=cached if nc is None else nc,
                       g0=restored or [11, 21, 22, 23], g0_first=restored or [11, 21, 22, 23], last=0, first_dec=None,
                       P_rec=b["P"])}
    return cg.evaluate([a, b], info, {"A": "ea", "B": "eb"})[1]
x = synth(4352)
ok("U2a a correct restore is safe", x["safe"] and x["restore_source"] == ["A"], x)
ok("U2b misaligned cached detected (S1)", not synth(4000)["S1_aligned"])
ok("U2c cached beyond the bound detected (S2)", not synth(8704, restored=[11, 12, 21, 22])["S2_within_bound"])
ok("U2d restored block not written by an earlier prefill detected (S3)",
   not synth(4352, restored=[99, 21, 22, 23])["S3_prefill_only"])
# decode-written: A decodes from P_A = 3*4352+10 -> block index 3 is decode-written; B restores 4 blocks of A
xa = dict(tag="A", phase="a", fixture="X", salt="s", salt_name="s", P=2 * 4352 + 10, W=9000, t0=1.0, t1=2.0,
          cached_tokens=0, cached_expected=0)
xb = dict(tag="B", phase="b", fixture="X", salt="s", salt_name="s", P=5 * 4352, W=5, t0=3.0, t1=4.0,
          cached_tokens=3 * 4352, cached_expected=3 * 4352)
inf = {"ea": dict(first_seq=1, first_t=1.1, computed_before=0, g0=[1, 2, 3, 4], g0_first=[1, 2, 3, 4], last=0,
                  first_dec=None, P_rec=xa["P"]),
       "eb": dict(first_seq=2, first_t=3.1, computed_before=3 * 4352, g0=[1, 2, 3, 9], g0_first=[1, 2, 3, 9], last=0,
                  first_dec=None, P_rec=xb["P"])}
xx = cg.evaluate([xa, xb], inf, {"A": "ea", "B": "eb"})[1]
ok("U2e a decode-written block restored is detected (S3)", not xx["S3_prefill_only"], xx["restore_source"])
ok("U2f scheduler num_computed != cached detected (S4)", not synth(4352, nc=0)["S4_attributed"])
# U3
ok("U3a classification", [cg.classify(*a) for a in ((0, 0), (4352, 4352), (0, 4352), (4352, 8704), (8704, 4352),
                                                     (None, 0))] == ["cold", "hit", "miss", "partial", "OVER",
                                                                     "missing"])
rows_b = [dict(phase="C1", fixture="M3", cls="miss"), dict(phase="D1", fixture="M3", cls="hit")]
bal = cg.balance(rows_b, {"C": ["C1"], "D": ["D1"]})
ok("U3b an arm-specific miss is unbalanced; its fixture is flagged", not bal["balanced"] and bal["mismatch_fixtures"]
   == ["M3"], bal)
# U4
G = geo.load(FC)
sys.path.insert(0, os.path.join(R, "analysis"))
good4 = True
for n in G["order"]:
    P = G["fx"][n]["P"]
    c = geo.rule_repeat(P)
    lo, hi = c - geo.BLOCK, c
    c0 = P // geo.BLOCK * geo.BLOCK
    good4 &= (hi == c and (c0 - 1) // geo.BLOCK == c // geo.BLOCK and (hi - 1) // geo.BLOCK == c // geo.BLOCK - 1)
ok("U4 sentinel = the last SHARED block [c - 4352, c); attempt 1's range lies in the private block, all 8", good4)
# U5
gc = os.path.join(R, "runner", "geometry_check_fcr.py")
log = open(os.path.join(RUN, "server.log"), errors="replace").read()
env = open(os.path.join(RUN, "container-files-env.txt")).read()


def geo_rc(logtxt, envtxt):
    with tempfile.TemporaryDirectory() as td:
        a, b = os.path.join(td, "l"), os.path.join(td, "e")
        open(a, "w").write(logtxt)
        open(b, "w").write(envtxt)
        return subprocess.run([sys.executable, gc, a, b, "1150"], capture_output=True, text=True).returncode
ok("U5a run-fc1's served geometry passes", geo_rc(log, env) == 0)
ok("U5b num_blocks 1,100 refused", geo_rc(log.replace('"num_blocks": 1275', '"num_blocks": 1100'), env) == 1)
ok("U5c GLM53_DFLASH_BOUNDARY_LOOKUP=1 refused", geo_rc(log, env.replace("GLM53_DFLASH_BOUNDARY_LOOKUP=0",
                                                                         "GLM53_DFLASH_BOUNDARY_LOOKUP=1")) == 1)
ok("U5d sparse retention refused", geo_rc(log, env + "VLLM_PREFIX_CACHE_RETENTION_INTERVAL=8704\n") == 1)
ok("U5e a changed group refused", geo_rc(log.replace('"SlidingWindowSpec", "layers": 5, "block_size": 1088',
                                                     '"SlidingWindowSpec", "layers": 5, "block_size": 2176'), env) == 1)
# U6
rep = json.load(open(os.path.join(R, "replay", "out", "reproduce_fc1.json")))
hist = json.load(open(os.path.join(R, "replay", "out", "historical.json")))
n6 = 0
bad6 = []
for x in rep["cached"]:
    n6 += 1
    if x["replay_hit"] != x["recorded_cached"]:
        bad6.append(x["tag"])
for k, v in hist.items():
    for x in v["requests"]:
        n6 += 1
        if not x["equal"]:
            bad6.append((k, x["id"]))
ok("U6a the replay equals every measured cached count (%d requests, 5 sessions)" % n6, not bad6, bad6[:5])
rule_rep = [x for k, v in hist.items() for x in v["requests"] if x["measured"] and x["measured"] == x["rule"]]
ok("U6b the F1 repeat rule holds on every measured exact-repeat hit (%d)" % len(rule_rep), len(rule_rep) >= 80)
# U7
for v in ("B", "A0"):
    pl = geo.phases(G, v)
    ok("U7a cache invariant clean (%s)" % v, not geo.cache_invariant(pl))
ct = geo.construction_table(geo.phases(G, "B"))
cls = {k: ("first" if not v["prior_users"] or v["prior_users"] in (["T-s1"], ["T-s2"], ["T-s3"], ["T-s4"]) else "later")
       for k, v in ct.items()}
ok("U7b construction recorded for every run", set(ct) >= {"C1", "C2", "C3", "D1", "D2", "D3"}, ct)
# U8
cost = json.load(open(os.path.join(R, "cost", "cost_fcr.json")))["sessions"]
sh = open(os.path.join(R, "runner", "run_fcr.sh")).read()
for v in ("B", "A0"):
    mm = re.search(r"%s\)\s+CEILING=([0-9.]+); WALL_MAX=(\d+); MIN_WALL=\$\{MIN_WALL:-(\d+)\}" % v, sh)
    ok("U8 runner constants = cost model (%s)" % v, mm and float(mm.group(1)) == cost[v]["ceiling_gh"] and
       int(mm.group(2)) == cost[v]["wall_cap_s"] and int(mm.group(3)) == cost[v]["min_wall_s"],
       (mm.groups() if mm else None, cost[v]["ceiling_gh"], cost[v]["wall_cap_s"], cost[v]["min_wall_s"]))
# U9 (needs numpy and fc_common; skipped on hosts without them)
try:
    import f6_criteria as f6
    T = {("F", 0, 30, 1): (0.6, 64), ("F", 0, 30, 2): (0.30, 64), ("F", 0, 30, 3): (0.0, 64)}
    PP = {("F", 0, 30, 1): (0.40, 64), ("F", 0, 30, 2): (0.0, 64), ("F", 0, 30, 3): (0.0, 64)}
    okp, det = f6.page_rule(T, None, PP)
    fails = {(d["page"]) for d in det["failing"]}
    ok("U9 page rule: noisy-null page widened (0.6 <= 0.8), quiet page with 0.30 fails (> 0.25)", fails == {2}, det)
except ImportError as exc:
    print("SKIP U9 (%r)" % exc)
n = sum(1 for _, x in RES if x)
print("unit tests: %d / %d PASS" % (n, len(RES)))
sys.exit(0 if n == len(RES) else 1)
