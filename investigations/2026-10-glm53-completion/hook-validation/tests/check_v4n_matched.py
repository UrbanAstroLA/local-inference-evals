#!/usr/bin/env python3
"""PA2-20 (amendment 21g item 1): do the client's V4n phases produce nulls MATCHED to every tested comparison?
CPU, host python, json only. Reads the client's plan dry run (runner/plan-dryrun-a2/plan.json) and checks, for every
fixture of the D2 set:
  * decode arms V4-b1, V4n-b1b, V4-d2 (and V4-d2solo for its fixture) force the same window: equal P, W, prefix and
    forced-window hashes, a window dump, and the same dump_src_sha (token ids at positions < P + W);
  * prefill arms V4-rebuild and V4n-rebuild2 dump exactly [P, P + W] of that window, from the same token ids at those
    positions (dump_src_sha equal to the decode arms'), at different build lengths;
  * V4n-b1b runs in two batches of 4 (a different batch composition from V4-b1's batch of 8);
  * the solo fixture is one of the D2 set (its matched nulls exist).
Layers and ranks are not visible in a plan: every dump holds every DSA index layer on both ranks (hook), and the
analysis matches units on (fixture, rank, layer, pool) and reports coverage (IDX-content, IDX-attrib).
usage: check_v4n_matched.py [PLAN_JSON]  -> prints PASS/FAIL lines; rc 0 iff all pass."""
import collections
import json
import os
import sys

HV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plan = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HV, "runner", "plan-dryrun-a2", "plan.json")))
R = {p["tag"]: p for p in plan if p.get("kind") == "request"}
d2set = sorted({t.rsplit("-", 1)[-1] for t in R if t.startswith("V4-b1-")})
solo = [t.rsplit("-", 1)[-1] for t in R if t.startswith("V4-d2solo-")]
bad, n = [], 0


def ck(ok, what):
    global n
    n += 1
    if not ok:
        bad.append(what)


for tid in d2set:
    arms = {a: R.get(f"{ph}-{a}-{tid}") for ph, a in (("V4", "b1"), ("V4n", "b1b"), ("V4", "d2"), ("V4", "rebuild"),
                                                      ("V4n", "rebuild2"))}
    if tid in solo:
        arms["d2solo"] = R.get(f"V4-d2solo-{tid}")
    miss = [a for a, v in arms.items() if v is None]
    ck(not miss, f"{tid}: arms missing {miss}")
    if miss:
        continue
    b1 = arms["b1"]
    P, W = b1["P"], b1["W"]
    for a in ("b1b", "d2") + (("d2solo",) if tid in solo else ()):
        x = arms[a]
        for k in ("P", "W", "prefix_sha", "forced_sha", "dump_src_sha"):
            ck(x.get(k) == b1.get(k), f"{tid}: {a}.{k} {x.get(k)} != b1 {b1.get(k)}")
        ck("local_fw_dumpwin" in x["keys"], f"{tid}: {a} has no window dump")
    for a in ("rebuild", "rebuild2"):
        x = arms[a]
        ck(x.get("dumprange") == [P, P + W], f"{tid}: {a}.dumprange {x.get('dumprange')} != {[P, P + W]}")
        ck(x.get("dump_src_sha") == b1.get("dump_src_sha"), f"{tid}: {a} dumps other token ids at [P, P+W)")
        ck("local_fw_dumpwin" in x["keys"], f"{tid}: {a} has no window dump")
    ck(arms["rebuild2"]["P"] > arms["rebuild"]["P"], f"{tid}: rebuild2 not longer than rebuild")
b1b_batches = collections.Counter(R[f"V4n-b1b-{t}"].get("batch") for t in d2set if f"V4n-b1b-{t}" in R)
ck(sorted(b1b_batches.values()) == [4, 4], f"V4n-b1b batches {dict(b1b_batches)} (want two of 4)")
ck(len(d2set) == 8, f"D2 set has {len(d2set)} fixtures (want 8)")
ck(len(solo) == 1 and solo[0] in d2set, f"solo fixture {solo} not in the D2 set")
extra = sorted(R[f"V4n-rebuild2-{t}"]["P"] - R[f"V4-rebuild-{t}"]["P"] for t in d2set if f"V4n-rebuild2-{t}" in R)
print(f"fixtures {d2set}; solo {solo}; b1b batches {dict(b1b_batches)}; rebuild2 build-length extra {extra}")
for b in bad:
    print("FAIL", b)
print(f"{'PASS' if not bad else 'FAIL'} V4n matched nulls: {n - len(bad)}/{n} checks; every tested fixture (B1 arms, "
      f"D2 arm, solo) has a decode null and a prefill null over the same window positions and token ids")
sys.exit(0 if not bad else 1)
