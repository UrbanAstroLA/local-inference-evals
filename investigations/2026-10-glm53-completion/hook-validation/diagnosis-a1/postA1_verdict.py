#!/usr/bin/env python3
"""Post-inspection re-reading of hook-validation attempt 1 (CPU, host python, json only). LABELLED POST-INSPECTION.

Reads the frozen verdict (../run-a1/verdict.json) and the diagnosis outputs in out/ (produced by the d_*/r_* scripts,
see DIAGNOSIS-a1.md section 9), and writes out/verdict-postA1.{json,txt}: every failed check with its ORIGINAL
result beside the post-inspection reading. Nothing here changes a frozen threshold; where a criterion is judged to
lack a measured basis, the measured null is reported next to the original number and the original FAIL is kept.

Post-inspection items:
  PI-1  H2 reference joins: V2b requests of the two block-crossing traces re-joined to their V2c reference at offset
        3832 - P; top-1 counted only where a reference row exists (frozen tolerances and TOP1_MIN unchanged).
  PI-2  H2b/H2c power control re-expressed on argmax agreement at offsets +-1 (reported; no frozen rule replaced).
  PI-3  H5g guard polls read per rank (the polls carry 'rank', not 'pid'); pre-client warm-up counts separated.
  PI-4  H1f violations classified (served row left unmasked; tracked post-mask mass).
  PI-5  index-K and KDA state comparisons placed beside measured nulls from the same run.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
O = os.path.join(HERE, "out")
V0 = json.load(open(os.path.join(HERE, "..", "run-a1", "verdict.json")))
J = lambda n: json.load(open(os.path.join(O, n)))  # noqa: E731

orig = {c["name"].split(" ")[0]: c for c in V0["checks"]}
rows = []


def row(cid, cls, post_status, post_detail):
    c = orig[cid]
    rows.append(dict(id=cid, original="PASS" if c["ok"] else "FAIL", original_level=c["level"],
                     original_detail=c["detail"], classification=cls, post_inspection=post_status,
                     post_detail=post_detail))


# ---- H1f
h1f = J("h1f_rows.json")
rows_unm = {(v["tag"], v["o"]) for v in h1f}
row("H1f", "(c) engine edge case, immaterial + (b) no tolerance for underflow-level mass",
    "FAIL kept; reclassified as served-kernel skip",
    "%d violating cells on %d rows, all 'kept although not eligible', all four termination ids, never the forced "
    "token; every row has fp32 top-1 p == 1.0 and p_post == p_pre (row left unmasked by the Triton top-p path, "
    "batch >= 8 rows); max kept post-mask p %.2e; 56 of 23,751 Triton rows with fp32 top-1 == 1.0 and a "
    "representable tail (0.24%%). Emulation: fp32 overflow of exp(max_logit - max_sample) > ~88.7 -> S = inf -> "
    "final_pivot = inf -> 'skip masking'." % (len(h1f), len(rows_unm), max(v["p_post"] for v in h1f)))
# ---- H2
h2 = J("h2_corrected.json")
row("H2a", "(a) analysis defect (reference join; NaN rows counted as disagreements)",
    "PASS under PI-1 (frozen tolerances)",
    "aligned q50/q90/q99 %.2e/%.3f/%.3f vs null %.2e/%.3f/%.3f (tol 0.01/0.30/1.50); top-1 %.4f (>= 0.95); n %d"
    % (h2["aligned"]["0.5"], h2["aligned"]["0.9"], h2["aligned"]["0.99"], h2["null"]["0.5"], h2["null"]["0.9"],
       h2["null"]["0.99"], h2["top1"], h2["n"]))
row("H2c", "(a) for 4 of 6 failing requests (same join defect); (b) for 2 (forced-token lp is uninformative under "
           "a shift on near-deterministic text)",
    "lp rule: 2 of 120 still fail (V1-*-free6-doc4, ratio 0.12 vs 0.1); argmax rule: 0 of 120",
    "per-request aligned top-1 min %.4f; shifted (+-1) top-1 max %.4f"
    % (h2["per_request_top1_min"], h2["per_request_shifted_top1_max"]))
row("H2b", "(b) power control without measured basis (adjacent forced-token log-probs are both ~0 on ~90% of "
           "positions)",
    "INCONCLUSIVE kept on the lp rule; argmax shift control separates",
    "lp shift medians %.3f/%.3f nat; argmax agreement aligned %.4f vs shifted %.4f/%.4f"
    % (h2["shift"]["median_plus1"], h2["shift"]["median_minus1"], h2["top1"], h2["shift"]["top1_plus1"],
       h2["shift"]["top1_minus1"]))
# ---- H4b
import statistics as st  # noqa: E402
kda = J("h4b_kda.json")
best = sum(1 for k in kda if min(range(len(k["dist"])), key=lambda i: k["dist"][i]) == k["na"] - 1)
l0 = [k["rel"] for k in kda if k["layer"] <= 2]
row("H4b", "(b) tolerance and margin without measured basis; no slot misidentification",
    "FAIL kept; slot identification 100%",
    "committed slot nearest the prefill-built state in %d of %d comparisons; layers 0-2 (identical inputs) rel "
    "median %.4f max %.4f; rel grows with depth to ~0.13 median at layers 33-42 where one token's update is ~0.2, so "
    "a 3x margin is unattainable there by construction" % (best, len(kda), st.median(l0), max(l0)))
# ---- index-K
nt = J("null_table.json")
pp = nt["NULL prefill-vs-prefill"]
H4 = J("h4c_idx.json"); NLP = J("h4c_nulls.json")["P"]
_k = {(r["tag"], r["layer"], r["pool"]) for r in H4}
_m = [r["rel"] for r in NLP if (r["g"], r["layer"], r["pool"]) in _k]
matched = dict(n=len(_m), gross=sum(x > .25 for x in _m) / len(_m))
for cid, key in (("H4c", "H4c V3 dec-vs-prefill"), ("H5b", "H5b B1 dec-vs-rebuild"), ("H5e", "H5e solo dec-vs-rebuild"),
                 ("H5c", "H5b B1 dec-vs-rebuild")):
    x = nt[key]
    row(cid, "(b) threshold without measured basis; no misalignment; no writer defect found",
        "FAIL/INCONCLUSIVE kept; within 2x of the measured null (H4c: 1.7x the same-pool null)",
        "%s gross %.4f q999 %.3f | null prefill-vs-prefill gross %.4f q999 %.3f max 1.36%s | decode b8-vs-b1 gross "
        "%.4f | same decode over time: byte-identical 100%% | layers 3-15: 0 gross in every comparison"
        % (key, x["gross"], x["q999"], pp["gross"], pp["q999"],
           (" (same pools: %.4f, n %d)" % (matched["gross"], matched["n"])) if cid == "H4c" else "",
           nt["NULL dec(b8)-vs-dec(b1)"]["gross"]))
# ---- H5f
cls = J("h5f_classes.json")
row("H5f", "(a) hook defect: shadow ledger mis-slices FULL-graph padded steps (3/5/7 in flight)",
    "FAIL kept; not repairable offline",
    "B1 arm (must be all clean): corrupt %d, none %d of %d pools, byte-correct (q50 %.3f); every 'none' pool "
    "completes in a padded step; CPU repro r_h5f_padded.py shifts attributions by one request"
    % (cls["b1|corrupt"]["n"], cls["b1|none"]["n"], sum(v["n"] for k, v in cls.items() if k.startswith("b1|")),
       cls["b1|corrupt"]["q50"]))
# ---- H5g
g = J("h5g_guard.json")
fr = g["final_by_rank"]
row("H5g", "(a) analysis defect (polls keyed by an absent 'pid'; pre-client warm-up artefact counted)",
    "PASS under PI-3",
    "rank0 tail_w_stash mism=null=%d = 6 x 16,683; rank1 %d = 5 x 16,683; only in V4-d2/V4-d2solo; pos_bounds 4 "
    "per rank at warm-up steps 4-6 before any request (same in run-b1-a1 and b0-mapguard; b0-mapguard/DIAGNOSIS-a2 "
    "s.5); every other violation counter 0" % (fr["0"]["tail_w_stash"]["mism"], fr["1"]["tail_w_stash"]["mism"]))
# ---- H5h
row("H5h", "(a) hook defect: on_selection assumes [B, next_n] rows; SM120 indexer flattens to [B*4, 1]",
    "FAIL kept; not repairable offline (records misattributed, not just missing)",
    "server.log 'use_flattening=True ... next_n=4'; of 6,342 present records 3,457 carry a step other than the "
    "measurement step (misattributed); CPU repro r_h5h_flatten.py: native layout 8/8 correct, flattened 1/8 "
    "correct + 1 misattributed; full selections (selbuf) share the defect")

inval = [r["id"] for r in rows if r["post_inspection"].startswith("FAIL") and r["id"] in ("H5f", "H5h")]
out = dict(label="POST-INSPECTION (attempt 1); original verdict INVALID stands", original_verdict=V0["verdict"],
           post_inspection_reading="INVALID: hook defects H5f, H5h (need fixed hook + GPU data); H1f engine edge "
           "case reclassified; H2a/H5g pass under analysis fixes; H2b/H2c/H4b/H4c/H5b/H5c/H5e criteria lack a "
           "measured basis and the measured values sit within 2x of measured nulls", rows=rows)
json.dump(out, open(os.path.join(O, "verdict-postA1.json"), "w"), indent=1)
with open(os.path.join(O, "verdict-postA1.txt"), "w") as f:
    f.write(out["label"] + "\n" + out["post_inspection_reading"] + "\n\n")
    for r in rows:
        f.write("%-4s original %-4s -> %s\n     class: %s\n     %s\n" % (r["id"], r["original"], r["post_inspection"],
                                                                         r["classification"], r["post_detail"]))
print(open(os.path.join(O, "verdict-postA1.txt")).read())
