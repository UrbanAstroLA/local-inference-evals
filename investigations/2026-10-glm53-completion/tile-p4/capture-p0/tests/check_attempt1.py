#!/usr/bin/env python3
"""Checks the read-only re-analysis of the real session attempt 1 (host python3): it must read INCOMPLETE, with no
replicate counted, and every replicate's failures must name both diagnosed defects -- the capture got 8,704 cached
tokens with a 13,057-token warm (amendment A1) and no scheduler log was written (the missing flush)."""
import json
import sys

r = json.load(open(sys.argv[1]))
N = [0, 0]


def check(name, ok, detail=""):
    N[0 if ok else 1] += 1
    print(("PASS " if ok else "FAIL ") + name + ("  " + str(detail)[:400] if detail and not ok else ""))


check("A1 attempt 1 reads INCOMPLETE, 0 of 6 replicates counted", r.get("cls") == "incomplete" and r.get("counted") == []
      and len(r.get("replicates", [])) == 6, (r.get("cls"), r.get("counted")))
g = r.get("gates", {})
check("A1 every replicate: capture cached_tokens 8704 (must be 13056)",
      all(any("capture cached_tokens 8704 (must be 13056)" in f for f in v["failures"]) for v in g.values()))
check("A1 every replicate: warm prompt_tokens 13057 (must be 17409)",
      all(any("prompt_tokens 13057 (must be 17409" in f for f in v["failures"]) for v in g.values()))
check("A1 every replicate: no scheduler log at all", all(any("no scheduler log at all" in f for f in v["failures"])
                                                        for v in g.values()))
check("A1 every replicate: capture taken from n_computed 8704 on both ranks",
      all(sum("n_computed 8704" in f for f in v["failures"]) == 2 for v in g.values()))
print("attempt-1 re-analysis checks: %d PASS, %d FAIL" % tuple(N))
sys.exit(1 if N[1] else 0)
