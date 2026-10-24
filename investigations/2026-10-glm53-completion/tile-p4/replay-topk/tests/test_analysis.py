#!/usr/bin/env python3
"""Host test of analysis/analyze_replay.py (stdlib only).

usage: test_analysis.py HARNESS_DIR WORK_DIR
HARNESS_DIR: the run directories written by test_harness_cpu.py (CPU stand-ins with known behaviour).
Part 1 checks the pre-registered reading of each stand-in case.  Part 2 plants defects into copies of the h-order
case (ORDER-SENSITIVE / A-ORDER / COMPOSED when clean) and checks each gives the pre-registered reading, never a
negative.  Prints PASS/FAIL lines; exit 1 on any FAIL.
"""
import copy
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
import analyze_replay as ar  # noqa: E402

HD, WD = sys.argv[1], sys.argv[2]
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} analysis {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def readings(res):
    out = {}
    for v, im in res["images"].items():
        for k, u in im["r1"].items():
            out[f"{v}.r1.{k}"] = u["reading"]
        for k, u in im["r2"].items():
            out[f"{v}.r2.{k}"] = u["reading"]
        if "r3" in im:
            out[f"{v}.r3"] = im["r3"]["reading"]
    return out


EXPECT = {
    "h-order": {"v091.r1.K": "A-ORDER", "v091.r1.C": "A-ORDER", "v091.r1.T": "A-ORDER", "v091.r2.flat": "ORDER-SENSITIVE",
                "v091.r2.peaked": "ORDER-SENSITIVE", "v091.r3": "COMPOSED", "v070.r1.C": "A-ORDER",
                "v070.r1.T": "A-ORDER", "v070.r2.flat": "ORDER-SENSITIVE", "v070.r2.peaked": "ORDER-SENSITIVE"},
    "h-sym": {"v091.r1.K": "REPEATABLE", "v091.r1.C": "REPEATABLE", "v091.r2.flat": "ORDER-INSENSITIVE",
              "v070.r2.peaked": "ORDER-INSENSITIVE", "v091.r3": "INCOMPLETE", "v070.r1.T": "REPEATABLE"},
    "h-noise": {"v091.r1.K": "A-SET", "v091.r2.flat": "A-EXEC", "v070.r2.peaked": "A-EXEC", "v070.r1.C": "INCOMPLETE"},
    "h-ptr": {"v091.r1.C": "B-ALLOC", "v091.r2.flat": "B-ALLOC", "v070.r2.flat": "B-ALLOC"},
    "h-bad": {"v091.r1.K": "INCOMPLETE", "v091.r1.C": "INCOMPLETE", "v070.r1.C": "INCOMPLETE", "v070.r1.T": "INCOMPLETE",
              "v091.r2.flat": "INCOMPLETE", "v091.r3": "INCOMPLETE"},
}
for case, exp in EXPECT.items():
    res = ar.analyze(os.path.join(HD, case))
    got = readings(res)
    bad = {k: (v, got.get(k)) for k, v in exp.items() if got.get(k) != v}
    check(f"stand-in case {case}", not bad, f"mismatches {bad}" if bad else "")
res = ar.analyze(os.path.join(HD, "h-bad"))
f = res["images"]["v091"]["r1"]["C"]["fails"]
check("h-bad v091 names the semantics gate", any("call semantics" in x for x in f), str(f))
f = res["images"]["v070"]["r1"]["C"]["fails"]
check("h-bad v070 names the negative control", any("negative control" in x for x in f), str(f))
md = ar.to_md(ar.analyze(os.path.join(HD, "h-order")))
check("markdown renders", "| R1 K | A-ORDER |" in md and "| R3 | COMPOSED |" in md)

# Part 2: planted defects on copies of h-order
BASE = os.path.join(HD, "h-order")


def load(ver, part):
    return json.load(open(os.path.join(BASE, ver, f"{part}.json")))


def plant(name, mutate, key, want):
    d = os.path.join(WD, name)
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(BASE, d)
    docs = {(v, p): load(v, p) for v in ("v091", "v070") for p in ("r1", "r2")}
    mutate(docs, d)
    for (v, p), doc in docs.items():
        path = os.path.join(d, v, f"{p}.json")
        if doc is None:
            if os.path.exists(path):
                os.remove(path)
        elif doc != "KEEP":
            json.dump(doc, open(path, "w"))
    got = readings(ar.analyze(d))
    check(f"planted {name} -> {key} {want}", got.get(key) == want, f"got {got.get(key)}")


def setk(doc, path, val):
    cur = doc
    for k in path[:-1]:
        cur = cur[k]
    cur[path[-1]] = val


P = []
P.append(("r1-missing", lambda D, d: D.__setitem__(("v091", "r1"), None), "v091.r1.C", "INCOMPLETE"))
P.append(("r1-status-error", lambda D, d: setk(D[("v091", "r1")], ["status"], "error"), "v091.r1.C", "INCOMPLETE"))
P.append(("r1-input-changed", lambda D, d: setk(D[("v091", "r1")], ["regimes", "C", "same_alloc", "input_changed_calls"], 1),
          "v091.r1.C", "INCOMPLETE"))
P.append(("r1-too-few-calls", lambda D, d: setk(D[("v091", "r1")], ["regimes", "C", "relocated", "calls"], 19),
          "v091.r1.C", "INCOMPLETE"))
P.append(("r1-no-warmup", lambda D, d: setk(D[("v091", "r1")], ["regimes", "C", "warmup_calls"], 0), "v091.r1.C", "INCOMPLETE"))
P.append(("r1-window", lambda D, d: setk(D[("v091", "r1")], ["window", "equals_formula"], False), "v091.r1.T", "INCOMPLETE"))
P.append(("r1-geometry", lambda D, d: setk(D[("v091", "r1")], ["regimes", "C", "first_selecting_position"], 2050),
          "v091.r1.C", "INCOMPLETE"))
P.append(("r1-posctrl", lambda D, d: setk(D[("v091", "r1")], ["regimes", "C", "pos_ctrl", "order_flagged"], False),
          "v091.r1.C", "INCOMPLETE"))
P.append(("r1-not-relocated", lambda D, d: setk(D[("v091", "r1")], ["regimes", "C", "relocated", "calls_relocated_from_ref"], 0),
          "v091.r1.C", "INCOMPLETE"))
P.append(("r1-shortcut-varied", lambda D, d: setk(D[("v091", "r1")], ["regimes", "C", "same_alloc", "rows_order_varied_shortcut"], 2),
          "v091.r1.C", "INCOMPLETE"))
P.append(("r1-stopped", lambda D, d: setk(D[("v091", "r1")], ["regimes", "C", "relocated_stopped"], "soft deadline"),
          "v091.r1.C", "INCOMPLETE"))
P.append(("r1-skippedK", lambda D, d: setk(D[("v091", "r1")], ["regimes", "K"], dict(regime="K", skipped="x")),
          "v091.r1.K", "INCOMPLETE"))
P.append(("r3-follows-r1K", lambda D, d: setk(D[("v091", "r1")], ["regimes", "K"], dict(regime="K", skipped="x")),
          "v091.r3", "INCOMPLETE"))
P.append(("r2-missing", lambda D, d: D.__setitem__(("v070", "r2"), None), "v070.r2.flat", "INCOMPLETE"))
P.append(("r2-same-set-broken", lambda D, d: setk(D[("v091", "r2")], ["regimes", "flat", "variants", "perm1", "same_set"], False),
          "v091.r2.flat", "INCOMPLETE"))
P.append(("r2-order-not-changed", lambda D, d: setk(D[("v091", "r2")], ["regimes", "flat", "variants", "asc", "rows_index_order_changed"], 0),
          "v091.r2.flat", "INCOMPLETE"))
P.append(("r2-variant-missing", lambda D, d: D[("v091", "r2")]["regimes"]["flat"]["variants"].pop("atomic2"),
          "v091.r2.flat", "INCOMPLETE"))
P.append(("r2-nan", lambda D, d: setk(D[("v091", "r2")], ["regimes", "flat", "nan_rows_base"], 3), "v091.r2.flat", "INCOMPLETE"))
P.append(("r2-few-repeats", lambda D, d: setk(D[("v091", "r2")], ["regimes", "flat", "repeat_same_alloc", "calls"], 4),
          "v091.r2.flat", "INCOMPLETE"))
P.append(("r2-posctrl-weak", lambda D, d: setk(D[("v091", "r2")], ["regimes", "flat", "pos_ctrl_set_change", "rows_differ_selecting"], 100),
          "v091.r2.flat", "INCOMPLETE"))
P.append(("r2-shortcut-differs", lambda D, d: setk(D[("v091", "r2")], ["regimes", "flat", "variants", "perm2", "rows_differ_shortcut"], 1),
          "v091.r2.flat", "INCOMPLETE"))
P.append(("r2-repeat-differs", lambda D, d: setk(D[("v091", "r2")], ["regimes", "flat", "repeat_same_alloc", "all_out_equal"], False),
          "v091.r2.flat", "A-EXEC"))
P.append(("r2-cache-poison", lambda D, d: setk(D[("v091", "r2")], ["regimes", "peaked", "cache_poison_equal"], False),
          "v091.r2.peaked", "B-ALLOC"))
P.append(("r2-scratch-poison", lambda D, d: setk(D[("v070", "r2")], ["regimes", "flat", "scratch_poison_equal"], [True, False]),
          "v070.r2.flat", "B-ALLOC"))
P.append(("r2-reloc", lambda D, d: setk(D[("v070", "r2")], ["regimes", "flat", "relocation", 1, "out_equal"], False),
          "v070.r2.flat", "B-ALLOC"))
P.append(("r2-stopped", lambda D, d: setk(D[("v091", "r2")], ["regimes", "flat", "stopped"], "soft deadline"),
          "v091.r2.flat", "INCOMPLETE"))
P.append(("r2-decode-caveat-only", lambda D, d: setk(D[("v091", "r2")], ["regimes", "flat", "record_decode_rel_err"], 0.5),
          "v091.r2.flat", "ORDER-SENSITIVE"))


def r3_anomaly(D, d):
    D[("v091", "r2")]["r3"]["calls"][0]["rows_out_diff_without_topk_diff"] = 4


def r3_none(D, d):
    for c in D[("v091", "r2")]["r3"]["calls"]:
        c["rows_topk_order_diff"] = 0
        c["rows_out_diff"] = 0


def r3_missing(D, d):
    D[("v091", "r2")].pop("r3")


P.append(("r3-anomaly", r3_anomaly, "v091.r3", "ANOMALY"))
P.append(("r3-no-variation", r3_none, "v091.r3", "NO-TOPK-VARIATION"))
P.append(("r3-missing", r3_missing, "v091.r3", "INCOMPLETE"))


def malformed(D, d):
    D[("v070", "r1")] = "KEEP"
    open(os.path.join(d, "v070", "r1.json"), "w").write("{not json")


P.append(("r1-malformed", malformed, "v070.r1.C", "INCOMPLETE"))
for name, mut, key, want in P:
    plant(name, mut, key, want)
# empty run directory: everything INCOMPLETE, nothing negative
os.makedirs(os.path.join(WD, "empty"), exist_ok=True)
got = readings(ar.analyze(os.path.join(WD, "empty")))
check("empty run dir: every unit INCOMPLETE", set(got.values()) == {"INCOMPLETE"}, str(set(got.values())))
print(f"analysis: {len(FAILS)} failure(s)")
sys.exit(1 if FAILS else 0)
