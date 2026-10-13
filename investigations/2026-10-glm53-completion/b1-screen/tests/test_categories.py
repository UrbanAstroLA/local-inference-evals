#!/usr/bin/env python3
"""Boundary cases of the frozen categories in b1screen_analysis.py (amendment 15b scheme, baseline E 6; amendment-19
reading; 14c sensitivity readout). CPU only, no data. Exit 0 if every case comes out as FROZEN-INTERPRETATION.md says."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import b1screen_analysis as B  # noqa: E402

# (E, raw, stated, premature) -> (15b category, amendment-19 reading, accuracy concern, 14c readout)
CASES = [
    ((0, 171, 174, 0), ("PROMISING FOLLOW-UP SIGNAL", "FAVOURABLE", False, "CLEAR PROMISING")),
    ((2, 164, 166, 0), ("PROMISING FOLLOW-UP SIGNAL", "FAVOURABLE", False, "CLEAR PROMISING")),
    ((1, 171, 174, 1), ("PROMISING FOLLOW-UP SIGNAL", "FAVOURABLE", False, "CLEAR PROMISING")),
    ((2, 163, 166, 0), ("MODEST IMPROVEMENT, NO CLEAR SEPARATION", "SIMILAR", True, "MIXED")),
    ((2, 159, 166, 0), ("MODEST IMPROVEMENT, NO CLEAR SEPARATION", "SIMILAR", True, "MIXED")),
    ((2, 158, 166, 0), ("SIMILAR OR WORSE", "UNFAVOURABLE", False, "MIXED")),
    ((2, 170, 159, 0), ("SIMILAR OR WORSE", "UNFAVOURABLE", False, "MIXED")),
    ((2, 170, 160, 0), ("PROMISING FOLLOW-UP SIGNAL", "FAVOURABLE", False, "CLEAR PROMISING")),
    ((2, 171, 174, 2), ("SIMILAR OR WORSE", "UNFAVOURABLE", False, "MIXED")),
    ((3, 171, 174, 0), ("MODEST IMPROVEMENT, NO CLEAR SEPARATION", "SIMILAR", False, "SIMILAR / NO CLEAR SEPARATION")),
    ((4, 171, 174, 0), ("MODEST IMPROVEMENT, NO CLEAR SEPARATION", "SIMILAR", False, "SIMILAR OR WORSE")),
    ((5, 171, 174, 0), ("MODEST IMPROVEMENT, NO CLEAR SEPARATION", "SIMILAR", False, "SIMILAR OR WORSE")),
    ((6, 171, 174, 0), ("SIMILAR OR WORSE", "SIMILAR", False, "SIMILAR OR WORSE")),
    ((6, 160, 170, 0), ("SIMILAR OR WORSE", "SIMILAR", True, "SIMILAR OR WORSE")),
    ((7, 171, 174, 0), ("SIMILAR OR WORSE", "UNFAVOURABLE", False, "SIMILAR OR WORSE")),
    ((12, 176, 178, 0), ("SIMILAR OR WORSE", "UNFAVOURABLE", False, "SIMILAR OR WORSE")),
]


def main():
    bad = 0
    for (E, raw, stated, prem), want in CASES:
        material, concern, cat = B.categorize(E, raw, stated, prem)
        r19, why = B.reading_a19(cat, E, material, concern)
        got = (cat, r19, concern, B.reading_14c(E, raw, stated, prem))
        ok = got == want
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} E={E:2d} raw={raw} stated={stated} premature={prem}: {cat} | {r19} {why} | 14c {got[3]}"
              + ("" if ok else f"   expected {want}"))
    print(f"RESULT: {len(CASES) - bad}/{len(CASES)} as specified")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
