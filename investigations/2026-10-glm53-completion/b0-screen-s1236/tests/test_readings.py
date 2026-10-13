#!/usr/bin/env python3
"""Boundary cases of the frozen rules in b0s1236_analysis.py. CPU only, no GPQA data.
Part 1: the unchanged amendment-15b / amendment-19 / 14c functions give exactly the B1 package's 16 frozen cases.
Part 2: the B0-vs-B1 reading (a)/(b)/(c) and the identity reading at their boundaries.
Part 3: identity() on synthetic rows: both-empty pairs are counted apart from I_ne; a token-count or hash difference breaks
identity; the reasoning-length sub-count. Exit 0 if every case comes out as FROZEN-INTERPRETATION.md says."""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "b1-screen", "tests"))
import b0s1236_analysis as B  # noqa: E402
import test_categories as T1  # noqa: E402  (the B1 package's frozen cases, read-only import)

bad = 0
n = 0


def chk(name, got, want):
    global bad, n
    n += 1
    ok = got == want
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got}" + ("" if ok else f"   expected {want}"))


print("== part 1: amendment-15b / 19 / 14c functions against the B1 package's frozen cases")
for (E, raw, stated, prem), want in T1.CASES:
    material, concern, cat = B.categorize(E, raw, stated, prem)
    r19, _ = B.reading_a19(cat, E, material, concern)
    chk(f"E={E} raw={raw} stated={stated} premature={prem}", (cat, r19, concern, B.reading_14c(E, raw, stated, prem)), want)

print("== part 2: B0-vs-B1 reading and identity reading")
for E, i_ne, want_code, want_lean in [
        (0, 0, "(b)", "run-to-run"), (3, 0, "(b)", "run-to-run"), (6, 0, "(b)", "run-to-run"), (6, 2, "(b)", "run-to-run"),
        (6, 3, "(b)", "no lean"), (6, 19, "(b)", "no lean"), (6, 20, "(b)", "re-check"), (4, 150, "(b)", "re-check"),
        (7, 0, "(c)", None), (7, 50, "(c)", None), (8, 0, "(a)", None), (9, 198, "(a)", None), (15, 0, "(a)", None)]:
    code, _stmt, lean = B.reading_b0b1(E, i_ne)
    lean_key = None if lean is None else ("run-to-run" if "run-to-run variation alone" in lean else
                                          "re-check" if "CPU re-check" in lean else "no lean" if lean.startswith("no lean") else lean)
    chk(f"reading E={E} I_ne={i_ne}", (code, lean_key), (want_code, want_lean))
for i_ne, want in [(0, "BATCHING NONDETERMINISM DOMINATES"), (2, "BATCHING NONDETERMINISM DOMINATES"), (3, "PARTIAL"),
                   (19, "PARTIAL"), (20, "NEAR-DETERMINISTIC AT A FIXED SEED"), (198, "NEAR-DETERMINISTIC AT A FIXED SEED")]:
    chk(f"identity reading I_ne={i_ne}", B.identity_reading(i_ne), want)

print("== part 3: identity() on synthetic rows")


def row(h, tok, empty=False, fin="stop", m=None):
    return {"response_sha256": h, "completion_tokens": tok, "empty": empty, "logged": True, "finish_reason": fin, "messages_sha256": m}


E0 = B.A.sha("")
S = {0: row("a", 900, m="m0"), 1: row("b", 1200, m="m1"), 2: row(E0, B.A.CAP, True, "length", "m2"),
     3: row("d", 5000, m="m3"), 4: row(E0, 4000, True, "stop", "m4"), 5: row("f", 700, m="m5")}
R = {0: row("a", 900, m="m0"), 1: row("b", 1201, m="m1"), 2: row(E0, B.A.CAP, True, "length", "m2"),
     3: row("x", 5000, m="m3"), 4: row(E0, 4000, True, "stop", "m4"), 5: row("f", 700, m="m5")}
XS = {"m0": {"reasoning_chars": 10, "reasoning_tokens": 5}, "m2": {"reasoning_chars": 99, "reasoning_tokens": 9},
      "m4": {"reasoning_chars": 7, "reasoning_tokens": 3}, "m5": {"reasoning_chars": 8, "reasoning_tokens": 4}}
XR = {"m0": {"reasoning_chars": 10, "reasoning_tokens": 5}, "m2": {"reasoning_chars": 98, "reasoning_tokens": 9},
      "m4": {"reasoning_chars": 7, "reasoning_tokens": 3}, "m5": {"reasoning_chars": 9, "reasoning_tokens": 4}}
I = B.identity(S, R, XS, XR)
chk("identical content+tokens (docs 0, 2, 4, 5)", I["identical_content_and_tokens"], 4)
chk("I_ne counts non-empty identical only (docs 0, 5)", (I["nonempty_identical_I_ne"], I["nonempty_identical_docs"]), (2, [0, 5]))
chk("both-empty identical by construction (docs 2, 4); at cap (doc 2)", (I["both_empty_identical_by_construction"], I["both_empty_at_cap"]), (2, 1))
chk("strict: equal reasoning length (non-empty: doc 0; both-empty: doc 4)",
    (I["nonempty_identical_with_equal_reasoning_length"], I["both_empty_with_equal_reasoning_length"]), (1, 1))
chk("identity reading of I_ne = 2", I["identity_reading"], "BATCHING NONDETERMINISM DOMINATES")
A2 = B.empty_agreement(S, {**R, 3: row(E0, B.A.CAP, True, "length", "m3")})
chk("empty agreement 2x2", (A2["empty_in_both"], A2["empty_only_in_new_pass"], A2["empty_only_in_reference"], A2["empty_in_neither"]), (2, 0, 1, 3))
print(f"RESULT: {n - bad}/{n} as specified")
sys.exit(0 if not bad else 1)
