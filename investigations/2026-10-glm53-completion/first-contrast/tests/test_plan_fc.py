#!/usr/bin/env python3
"""Tampered-plan negatives for runner/plan_check.py [A22-9] (host Python, CPU, no network). The untampered plan must
pass; each tampered copy must FAIL with the named rule. usage: test_plan_fc.py PLAN_JSON  -> rc 0 iff all behave."""
import copy
import io
import json
import os
import sys
import contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "runner"))
import plan_check as pc  # noqa: E402

BASE = json.load(open(sys.argv[1]))
TMP = os.path.join(os.environ.get("TMPDIR", "/tmp"), "fc-plan-tamper.json")


def reqs(p):
    return [d for d in p if d.get("kind") == "request"]


def first(p, **kw):
    return next(d for d in reqs(p) if all(d.get(k) == v for k, v in kw.items()))


def run(p):
    json.dump(p, open(TMP, "w"))
    pc.FAIL.clear()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = pc.main(TMP)
    return rc, sorted({r for r, _ in pc.FAIL}), buf.getvalue()


def t_d2_on_C(p):
    first(p, tag="C1-L3")["d2"] = [0, first(p, tag="C1-L3")["W"] + 1]


def t_shared_salt(p):                      # an LP round salt equal to a primary salt
    for d in reqs(p):
        if d["salt_name"] == "l1":
            d["salt_name"], d["salt"] = "s1", BASE_SALT["s1"]


def t_truncated(p):                        # a list truncated to 5,121 ids (the attempt-1/2 WMAX + 1)
    first(p, tag="C2-L1")["W"] = 5121


def t_cached_mismatch(p):
    first(p, tag="D1-M4")["cached_expected"] = 0


def t_dup_seed(p):
    first(p, tag="C3-L5")["seed"] = first(p, tag="C1-L5")["seed"]


def t_inject_before_gate(p):              # move the D2 run before CV
    rs = [d for d in p if d.get("kind") == "request"]
    d2 = [d for d in rs if d["phase"] == "D2"]
    rest = [d for d in p if d not in d2]
    i = next(k for k, d in enumerate(rest) if d.get("phase") == "CV")
    p[:] = rest[:i] + d2 + rest[i:]


def t_equal_build(p):                     # LW2 at LW1's build length: the prefill null at equal build length
    for d in reqs(p):
        if d["phase"] == "LW2":
            l1 = first(p, tag="LW1-" + d["fixture"])
            d["P"], d["prompt_sha256"] = l1["P"], l1["prompt_sha256"]


def t_ps_after_order(p):                  # PS-c before D3
    rs = [d for d in p if d.get("phase") == "PS-c"]
    rest = [d for d in p if d not in rs]
    i = next(k for k, d in enumerate(rest) if d.get("phase") == "D3")
    p[:] = rest[:i] + rs + rest[i:]


def t_composition(p):                     # two fixtures swapped in one run
    rs = [d for d in p if d.get("phase") == "D3"]
    a, b = p.index(rs[0]), p.index(rs[1])
    p[a], p[b] = p[b], p[a]


def t_cv_targets(p):
    first(p, tag="CV-oracle-L5")["snap"] = [58, 62, 66, 70]


def t_cache_invariant(p):                 # a primary-salt prompt reaching a full block past the decode start
    d = first(p, tag="rewarm-s1-L1")
    d["P"] = d["P"] + 4352
    d["prompt_sha256"] = "x"


def t_missing_d2(p):
    first(p, tag="LD2-A8")["d2"] = None


CASES = [("T1 D2 key on a clean request", t_d2_on_C, "P9"), ("T2 LP salt shared with a primary salt", t_shared_salt, "P5"),
         ("T3 list truncated to 5,121", t_truncated, "P1"), ("T4 cached-count mismatch", t_cached_mismatch, "P7"),
         ("T5 duplicate seed", t_dup_seed, "P8"), ("T6 injected batch before GATE-1", t_inject_before_gate, "P11"),
         ("T7 prefill null at equal build length", t_equal_build, "P12"), ("T8 PS-c before D3", t_ps_after_order, "P11"),
         ("T9 run composition order", t_composition, "P10"), ("T10 CV snapshot targets", t_cv_targets, "P13"),
         ("T11 cache-safety invariant", t_cache_invariant, "P6"), ("T12 injected request without the D2 key", t_missing_d2, "P9")]
BASE_SALT = {d["salt_name"]: d["salt"] for d in reqs(BASE)}
ok_all = True
rc, rules, out = run(copy.deepcopy(BASE))
print("%s T0 untampered plan passes (rc %d, failures %s)" % ("PASS" if rc == 0 else "FAIL", rc, rules))
ok_all &= rc == 0
for name, fn, rule in CASES:
    p = copy.deepcopy(BASE)
    fn(p)
    rc, rules, out = run(p)
    ok = rc != 0 and rule in rules
    ok_all &= ok
    print("%s %s -> refused with %s (rules failing %s)" % ("PASS" if ok else "FAIL", name, rule, rules))
print("plan tamper tests: %s" % ("all behave" if ok_all else "FAILURES"))
sys.exit(0 if ok_all else 1)
