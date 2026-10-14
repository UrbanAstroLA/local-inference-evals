#!/usr/bin/env python3
"""Cross-check the claim-structure model against the run's own descriptive counter
(summary-v070.json desc.stock_col511_rank_lt480, measured on perspective 0 on the GPU).
Model: column 511 is the last-claimed slot, i.e. a member of the final group, which holds the
bottom-G selected pools (ranks 512-G..511), or, on the round-3 tie path, a pool tied at t.
Prediction per row-repeat that rank(col511) < 480:
  tie path    : rank = number of selected pools scoring strictly above t (exact; from the tie slots)
  no-tie path : (G-32)/G if G > 32 else 0  (uniform race inside the group; an assumption)
gauss is not modelled (CUDA-generator inputs), so the remainder is attributed to it."""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(H, "diag_persp_rows.json")))
summ = json.load(open(os.path.join(H, "..", "run-20261013T234310Z", "v070", "summary-v070.json")))
obs = {c["id"]: c["desc"]["stock_col511_rank_lt480"] for c in summ["cells"] if c.get("desc")}
for cid in ("v070/tiled48/b16384", "v070/tiled48/b32768"):
    exp_exact, exp_real = 0.0, 0.0
    for m in rows:
        if m["cell"] != cid:
            continue
        if m["path"] == "tie-r3":
            above = 512 - m["tie_slots"]          # pools strictly above t
            p = 1.0 if above < 480 else 0.0
        else:
            G = m["final_group"]
            p = (G - 32) / G if G > 32 else 0.0
        if m["fam"].startswith("real_"):
            exp_real += 3 * p
        else:
            exp_exact += 3 * p
    print(f"{cid}: observed rank<480 (7 families, 3 repeats) = {obs[cid]}; model: exact families "
          f"{exp_exact:.0f} (deterministic), real families ~{exp_real:.1f}; remainder for gauss "
          f"{obs[cid] - exp_exact - exp_real:.1f} of 288 gauss row-repeats")
