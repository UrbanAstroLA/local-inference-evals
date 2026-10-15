#!/usr/bin/env python3
"""Tampered-plan negatives for runner/plan_check_fcr.py [R-8] (host python, CPU, no network). The untampered plan must
pass; each tampered copy must FAIL with the named rule(s).
usage: test_plan_fcr.py PLAN_JSON  -> rc 0 iff all behave."""
import contextlib
import copy
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "runner"))
import plan_check_fcr as pc  # noqa: E402

BASE = json.load(open(sys.argv[1]))
TMP = os.path.join(os.environ.get("TMPDIR", "/tmp"), "fcr-plan-tamper-%d.json" % os.getpid())
BLOCK = 4352


def reqs(p):
    return [d for d in p if d.get("kind") == "request"]


def first(p, **kw):
    return next(d for d in reqs(p) if all(d.get(k) == v for k, v in kw.items()))


def run(p):
    json.dump(p, open(TMP, "w"))
    pc.FAIL.clear()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = pc.main(TMP, "B")
    return rc, sorted({r for r, _ in pc.FAIL}), buf.getvalue()


def t_d2_on_C(p):
    d = first(p, tag="C1-L3")
    d["d2"] = [0, d["W"] + 1]


def t_salt_shared(p):                       # a reference salt reused by a run
    s = first(p, tag="LW1-L1")
    for d in reqs(p):
        if d["phase"] == "C1":
            d["salt_name"], d["salt"] = s["salt_name"], s["salt"]


def t_truncate(p):
    d = first(p, tag="C1-L1")
    d["W"] = 5121


def t_attempt1_rule(p):                     # the attempt-1 expected value floor(P / 4352) * 4352
    d = first(p, tag="C1-L1")
    d["cached_expected"] = d["P"] // BLOCK * BLOCK


def t_dup_seed(p):
    first(p, tag="C2-L4")["seed"] = first(p, tag="C1-L4")["seed"]


def t_injected_before_gate(p):              # move D2's requests before the gate record
    g = next(i for i, d in enumerate(p) if d.get("kind") == "gate")
    d2 = [d for d in p if d.get("kind") == "request" and d["phase"] == "D2"]
    rest = [d for d in p if not (d.get("kind") == "request" and d["phase"] == "D2")]
    g = next(i for i, d in enumerate(rest) if d.get("kind") == "gate")
    p[:] = rest[:g] + d2 + rest[g:]


def t_equal_build(p):
    for d in reqs(p):
        if d["phase"] == "LW2":
            d["P"] = first(p, tag="LW1-" + d["fixture"])["P"]


def t_psc_before_d2(p):
    psc = [d for d in p if d.get("kind") == "request" and d["phase"] == "PS-c"]
    rest = [d for d in p if not (d.get("kind") == "request" and d["phase"] == "PS-c")]
    i = next(k for k, d in enumerate(rest) if d.get("kind") == "request" and d["phase"] == "D2")
    p[:] = rest[:i] + psc + rest[i:]


def t_composition(p):
    p[:] = [d for d in p if not (d.get("kind") == "request" and d.get("tag") == "D3-L8")]


def t_cv_targets(p):
    d = next(d for d in reqs(p) if d["form"] == "CV")
    d["snap"] = [1, 2, 3]


def t_lp_after_primary(p):                  # LC1 after C1 on s1: its prompt reaches C1's decode-written blocks
    lc = [d for d in p if d.get("kind") == "request" and d["phase"] == "LC1"]
    rest = [d for d in p if not (d.get("kind") == "request" and d["phase"] == "LC1")]
    i = max(k for k, d in enumerate(rest) if d.get("kind") == "request" and d["phase"] == "C1") + 1
    p[:] = rest[:i] + lc + rest[i:]


def t_missing_d2_key(p):
    first(p, tag="D1-M3")["d2"] = None


def t_no_touch(p):                          # drop the touch pass between the half-builds of s4
    p[:] = [d for d in p if not (d.get("kind") == "request" and d["phase"] == "T-s4-a")]


def t_sentinel_attempt1_range(p):           # attempt 1's [c0 - 1024, c0)
    d = first(p, tag="PS-a-L1")
    c0 = d["P"] // BLOCK * BLOCK
    d["dumprange"] = [c0 - 1024, c0]


def t_run_stagger(p):
    next(d for d in p if d.get("kind") == "phase" and d.get("tag") == "C1")["stagger"] = 0.0


def t_build_order(p):
    ph = next(d for d in p if d.get("kind") == "phase" and d.get("tag") == "warm-s2-b")
    ph["submission_order"] = list(reversed(ph["submission_order"]))


def t_d2_other_salt(p):
    s = first(p, tag="D1-L1")
    for d in reqs(p):
        if d["phase"] == "D2":
            d["salt_name"], d["salt"] = s["salt_name"], s["salt"]


def t_ps_not_sequential(p):
    next(d for d in p if d.get("kind") == "phase" and d.get("tag") == "PS-b")["sequential"] = False


CASES = [
    ("T1 D2 key on a C request", t_d2_on_C, {"Q9"}),
    ("T2 reference salt reused by a run", t_salt_shared, {"Q5", "Q14"}),
    ("T3 forced list truncated to 5,121", t_truncate, {"Q1", "Q2/Q3"}),
    ("T4 attempt-1 cache rule", t_attempt1_rule, {"Q7"}),
    ("T5 duplicate seed", t_dup_seed, {"Q8"}),
    ("T6 injected batch before GATE-1R", t_injected_before_gate, {"Q11"}),
    ("T7 LW1 / LW2 at equal build length", t_equal_build, {"Q2/Q3", "Q12"}),
    ("T8 PS-c before D2", t_psc_before_d2, {"Q11"}),
    ("T9 run composition", t_composition, {"Q10"}),
    ("T10 CV snapshot targets", t_cv_targets, {"Q13"}),
    ("T11 LP prompt after primary decode on the build", t_lp_after_primary, {"Q6", "Q7"}),
    ("T12 missing D2 key", t_missing_d2_key, {"Q9"}),
    ("T13 touch pass dropped", t_no_touch, {"Q14", "Q7"}),
    ("T14 attempt-1 sentinel range", t_sentinel_attempt1_range, {"Q15"}),
    ("T15 run submitted without stagger", t_run_stagger, {"Q16"}),
    ("T16 build submission order changed", t_build_order, {"Q16"}),
    ("T17 D2 not on C2's build salt", t_d2_other_salt, {"Q11", "Q15", "Q7", "Q14", "Q5", "Q6"}),
    ("T18 sentinel pass not sequential", t_ps_not_sequential, {"Q16"}),
]


def main():
    ok_all = True
    rc, rules, out = run(copy.deepcopy(BASE))
    ok = rc == 0 and not rules
    ok_all &= ok
    print("%s T0 untampered plan passes (rc %d, rules %s)" % ("PASS" if ok else "FAIL", rc, rules))
    for name, fn, want in CASES:
        p = copy.deepcopy(BASE)
        fn(p)
        rc, rules, out = run(p)
        ok = rc == 1 and set(rules) <= want and bool(set(rules) & want)
        ok_all &= ok
        print("%s %s: rc %d rules %s (expected within %s)" % ("PASS" if ok else "FAIL", name, rc, rules, sorted(want)))
        if not ok:
            print(out[-1500:])
    os.remove(TMP)
    print("tampered-plan tests: %s" % ("all behave" if ok_all else "FAILURES"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
