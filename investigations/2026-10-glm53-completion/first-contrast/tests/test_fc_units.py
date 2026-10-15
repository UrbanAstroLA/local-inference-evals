#!/usr/bin/env python3
"""Unit checks of the frozen first-contrast analysis rules (CPU container; /fc mounted read-only).
  U1 [A22-3] N_pp = 0 with N_dd > 0: the content limit is 2 x N_dd, NOT the 0.1 % floor; both zero -> the floor
  U2 [A22-2] contrast(): the three paired differences are reported; separation and beyond-null are operational screens;
     no exact permutation level or significance field exists
  U3 [A22-2] a 3-vs-3 split that is completely separated but within 2 x max(N, 0.05) is NOT beyond the null
  U4 [A22-8] the reading strings are exactly amendment 22 s.8's (cases 1-3) and the direction statements
  U5 [A22-5] conv_entries: speculative rows p0 - 2 + e (committed e <= a + 2); non-speculative rows 0..2 = frontier-2..frontier
  U6 R3 flags: a one-position event across many layers flags by units but the position count is reported beside it
"""
import sys
import types

sys.path.insert(0, "/fc/analysis")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import fc_common as fc  # noqa: E402
import analyze_fc as an  # noqa: E402

R = []


def check(name, ok, detail=""):
    R.append(ok)
    print("%s %s %s" % ("PASS" if ok else "FAIL", name, detail))


G = fc.GROSS
u = [("f", (0, 19, i), 0.0, (0.5 if i < 10 else 0.0), 0.0) for i in range(1000)]
lim, ndd, npp = fc.limit_of(u)
check("U1a N_pp = 0, N_dd = 1 %: limit = 2 x N_dd = 2 %, not the 0.1 % floor", abs(lim - 0.02) < 1e-12 and npp == 0.0,
      "limit %.4f N_dd %.4f N_pp %.4f" % (lim, ndd, npp))
u0 = [("f", (0, 19, i), 0.0, 0.0, 0.0) for i in range(1000)]
check("U1b both nulls 0: limit = the 0.1 % operational floor", fc.limit_of(u0)[0] == fc.CONTENT_FLOOR)
c = an.contrast([-6.0, -6.02, -5.99], [-6.6, -6.62, -6.58])
check("U2 paired differences reported (3), screens operational, no permutation/significance field",
      c["paired_d"] is not None and len(c["paired_d"]) == 3 and c["separation"] and c["beyond_null"]
      and not any(k for k in c if "perm" in k or "signif" in k or "p_value" in k), str(sorted(c)))
c = an.contrast([-6.0, -6.02, -5.99], [-6.06, -6.08, -6.07])
check("U3 completely separated but |Delta| = 0.067 <= 2 x max(N, 0.05) = 0.1: not beyond the null",
      c["separation"] and not c["beyond_null"] and not c["screen_met"], "delta %.3f thr %.3f" % (c["delta"], c["threshold"]))
src = open("/fc/analysis/analyze_fc.py").read()
A22 = {"READING_1": "supported local computational mechanism candidate; behavioural relevance assessed via post-top_p and "
                    "eligibility; no GPQA explanation",
       "READING_2": "no termination-propensity effect detected under this exposure and design; the mechanism is not "
                    "excluded",
       "READING_3": "uninformative",
       "DIR_UP": "consistent with more corrupted old-engine computation helping termination",
       "DIR_DOWN": "supports a harmful D2 effect; stock 0.7.0 finishing more often would then need an interaction or "
                   "another contributor"}
bad = [k for k, v in A22.items() if v not in getattr(an, k)]
check("U4 reading and direction strings as amendment 22 s.8 (verbatim); 'no effect detected by the prespecified screen' "
      "replaces 'equivalent within resolution'; no 1/20 claim", not bad and "prespecified screen" in an.READING_2
      and "equivalent within" not in src and "1/20" not in src, bad)
snap = dict(frontier_pos=100, num_accepted=3)
L = dict(is_spec=True, conv=torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4))
e = fc.conv_entries(snap, L)
check("U5a speculative: a = 2, p0 = 98, committed rows 0..4 -> positions 96..100", sorted(e) == [96, 97, 98, 99, 100]
      and torch.equal(e[100], L["conv"][4]))
L2 = dict(is_spec=False, conv=torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4))
e2 = fc.conv_entries(snap, L2)
check("U5b non-speculative: rows 0..2 -> positions 98..100", sorted(e2) == [98, 99, 100] and torch.equal(e2[98], L2["conv"][0]))
uu = [("x", (0, L_, 128), 1.0, 0.0, 0.0) for L_ in range(19, 45, 2)] + [("x", (0, 19, i), 0.0, 0.0, 0.0) for i in range(200)]
fl = fc.flags(uu)
check("U6 one position gross in 13 layers: flagged by units (>= 10), not by positions (1 < 3); both counts reported",
      len(fl) == 1 and fl[0]["by_unit"] and not fl[0]["by_position"] and fl[0]["gross_positions"] == 1, str(fl)[:200])
print("test_fc_units: %d/%d checks passed" % (sum(R), len(R)))
sys.exit(0 if all(R) else 1)
