#!/usr/bin/env python3
"""Analysis tests on SYNTHETIC sessions (make_fake_cap0.build): every outcome class of the pre-registered table, and
every planted defect of the evidence -> INCOMPLETE (never a negative).  CPU container (numpy).

usage: check_analysis_cap0.py OUTDIR      (OUTDIR inside tests/out; each case's npz files are deleted after it runs)
"""
import glob
import importlib.util
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_fake_cap0 as F  # noqa: E402

_s = importlib.util.spec_from_file_location("A", os.path.join(HERE, "..", "analysis", "analyze_cap0.py"))
A = importlib.util.module_from_spec(_s)
_s.loader.exec_module(A)
OUT = sys.argv[1]
N = {"PASS": 0, "FAIL": 0}
ALL4 = [("r1", True), ("r2", True), ("r3", False), ("r4", False)]
TWO = [("r1", True), ("r2", True)]


def check(name, ok, detail=""):
    N["PASS" if ok else "FAIL"] += 1
    print(("PASS " if ok else "FAIL ") + name + ("  " + json.dumps(detail, default=str)[:700] if detail and not ok else ""), flush=True)


def diffs(reps, point=None, pos=0, mla_pos=None, kda=True, skip=()):
    """Every replicate after the first differs (its own variant): MLA-7 records from mla_pos (default pos), KDA-4
    recurrent on both ranks, localisation from (point, pos)."""
    var, core, loc = {}, {}, {}
    for k, (r, l) in enumerate(reps):
        if k == 0 or r in skip:
            continue
        var[r] = k
        core[r] = [(0, f"mla7_b{j}", pos if mla_pos is None else mla_pos) for j in range(3)]
        if kda:
            core[r] += [(0, "kda4_rec", 0), (1, "kda4_rec", 0)]
        if l and point:
            loc[r] = (point, pos)
    return dict(variant=var, core_diff=core, loc_diff=loc)


def run(name, spec, expect_cls, extra=None):
    root = os.path.join(OUT, name)
    shutil.rmtree(root, ignore_errors=True)
    ao, po = F.build(root, spec)
    try:
        res, _d = A.analyse(ao, po)
        json.dump(res, open(os.path.join(root, "result.json"), "w"), indent=1, default=str)
        open(os.path.join(root, "result.md"), "w").write(A.render(res))
        ok = res.get("cls") == expect_cls
        check(f"{name}: class {A.CLASSES[expect_cls]!r}", ok, dict(got=res.get("cls"), why=res.get("why"),
                                                                  wnl=res.get("why_not_located"), gates=res.get("gates")))
        if extra:
            for label, fn in extra:
                try:
                    v = fn(res)
                except Exception as exc:  # noqa: BLE001
                    v = False
                    label += f" (raised {exc!r})"
                check(f"{name}: {label}", bool(v), dict(res={k: res.get(k) for k in ("located", "loc_reading", "why_not_located",
                                                                                   "allocation", "conv_rows_test", "notes", "why")}))
        return res
    finally:
        for p in glob.glob(os.path.join(root, "synthetic-po", "*.npz")):
            os.remove(p)


def first_pair(res):
    return next(iter(res["pairs"].values()))


run("A_located_moe", dict(reps=ALL4, **diffs(ALL4, "L3.moe_out", 2100)), "located", [
    ("boundary L3.moe_out -> the MoE operation", lambda r: r["located"]["boundary"] == "L3.moe_out" and "MoE" in r["located"]["operation"]),
    ("located in warm chunk 1 (tokens 2,045-4,089)", lambda r: r["located"]["chunk"] == 1),
    ("MLA-7 block 0 first differing record 2,100; chunk 0 untouched", lambda r: first_pair(r)["slices"]["r0:mla7_b0"]["first_pos"] == 2100
     and "0" not in first_pair(r)["slices"]["r0:mla7_b0"]["by_chunk"]),
    ("latent lanes only (scale lanes equal)", lambda r: first_pair(r)["slices"]["r0:mla7_b0"]["latent_only"] > 0 and first_pair(r)["slices"]["r0:mla7_b0"]["scale_only"] == 0),
    ("assessor row = reproducible local operation defect found", lambda r: A.ASSESSOR[r["cls"]][0].startswith("Valid state differs, reproducible")),
    ("loc and no-loc pairs agree (instrumentation not implicated)", lambda r: r["instrumentation"]["reading"] == "loc and no-loc pairs agree"),
    ("every pair relocated: allocation and execution not separated", lambda r: "not separated" in r["allocation"]["reading"]),
    ("controls (MLA 3, KDA 2) equal", lambda r: all(not p["control_differ"] for p in r["pairs"].values())),
])
run("B_located_attn_chunk1", dict(reps=ALL4, **diffs(ALL4, "L3.attn_out", 10)), "located", [
    ("boundary L3.attn_out (sparse MLA attention)", lambda r: r["located"]["boundary"] == "L3.attn_out"),
    ("chunk 0 already differs in MLA-7 (dense path)", lambda r: "0" in first_pair(r)["slices"]["r0:mla7_b0"]["by_chunk"]),
])
NL = [("r1", False), ("r2", False), ("r3", False), ("r4", False)]
run("C_no_localisation", dict(reps=NL, **diffs(NL)), "slice", [
    ("reason: no localisation pair", lambda r: "no localisation pair passed" in r["why_not_located"]),
    ("assessor row = bounded replay does not locate it", lambda r: A.ASSESSOR[r["cls"]][0].endswith("does not locate it")),
])
run("D_enters_before_layer3", dict(reps=ALL4, **diffs(ALL4, "L3.in", 5)), "slice", [
    ("reason: enters before layer 3", lambda r: "before layer 3" in r["why_not_located"])])
R3 = [("r1", True), ("r2", True), ("r3", True)]
sp = diffs(R3, "L3.moe_out", 100)
sp["loc_diff"]["r3"] = ("L4.rec_out", 50)
run("E_pairs_disagree", dict(reps=R3, **sp), "slice", [("reason: pairs disagree", lambda r: "disagree" in r["why_not_located"])])
run("F_equal", dict(reps=ALL4), "equal", [
    ("assessor: no row, non-reproduction", lambda r: "did not reproduce" in A.ASSESSOR[r["cls"]][0]),
    ("no valid slice differs in any pair", lambda r: all(not p["valid_differ"] for p in r["pairs"].values()))])
for nm, br in (("G_rank1_capture_missing", "cap_rank1"), ("H_view_mismatch", "view"), ("I_n_cached_wrong", "ncached"),
               ("J_chunks_differ", "chunks"), ("L_cached_tokens_wrong", "cached_tokens"), ("M_read_after_precopy", "timing"),
               ("N_npz_missing", "npz_missing"), ("W_provenance", "prov"), ("X_twosum", "twosum")):
    run(nm, dict(reps=TWO, **diffs(TWO, "L3.moe_out", 2100), **{"break": {"r2" if br not in ("timing", "npz_missing") else "r1": [br]}}),
        "incomplete", [("missing / invalid evidence -> INCOMPLETE, with the gate named", lambda r: any(
            g["failures"] for g in r["gates"].values()) and r.get("why"))])
run("K_prompt_differs", dict(reps=ALL4, **diffs(ALL4, "L3.moe_out", 2100), **{"break": {"r2": ["prompt"]}}), "incomplete", [
    ("cross-replicate input check named", lambda r: any("prompt_sha256" in w for w in r["why"]))])
run("K2_one_of_four_fails_rest_counted", dict(reps=ALL4, **diffs(ALL4, "L3.moe_out", 2100), **{"break": {"r4": ["view"]}}), "located", [
    ("r4 not counted, the other three are", lambda r: r["counted"] == ["r1", "r2", "r3"])])
run("O_loc_file_missing", dict(reps=ALL4, **diffs(ALL4, "L3.moe_out", 2100), **{"break": {"r2": ["loc_missing"]}}), "slice", [
    ("not equal and not located: localisation evidence missing", lambda r: "no localisation pair passed" in r["why_not_located"]),
    ("the loc gate failure is reported for r2", lambda r: r["gates"]["r2"]["loc_ok"] is False)])
run("P_identity_violated_in_r1", dict(reps=R3, **diffs(R3, "L3.moe_out", 2100), **{"break": {"r1": ["identity"]}}), "located", [
    ("r1's localisation is excluded (identity violated)", lambda r: r["gates"]["r1"]["loc_ok"] is False and
     any("identity" in f for f in r["gates"]["r1"]["loc_failures"])),
    ("the remaining loc pair r2-r3 locates the MoE", lambda r: r["located"]["boundary"] == "L3.moe_out")])
run("Q_lifecycle", dict(reps=ALL4, **diffs(ALL4, "L3.moe_out", 2100), **{"break": {"r2": ["lifecycle"]}}), "located", [
    ("lifecycle note: last state write != point-0 bytes", lambda r: any("lifecycle" in n for n in r["notes"]))])
run("R_same_allocation_pair", dict(reps=ALL4, same_ids=["r2"], **diffs(ALL4, "L3.moe_out", 2100)), "located", [
    ("r1-r2 same allocation and differing -> execution nondeterminism reading", lambda r: "r1-r2" in r["allocation"]["same_alloc_differ"]
     and "execution nondeterminism" in r["allocation"]["reading"]),
    ("r2's blocks list r1's warm as an earlier holder", lambda r: "CW-r1-L" in r["prior_holders"]["by_replicate"]["r2"]["r0:mla7_b0"]["earlier_holders"])])
run("S_inactive_conv_rows_only", dict(reps=TWO, variant={"r2": 1}, core_diff={"r2": [(0, "kda0_conv_inactive", 0), (1, "kda0_conv_inactive", 0)]}), "equal", [
    ("stale-row inference supported (rows 3-5 differ, 0-2 equal)", lambda r: r["conv_rows_test"]["reading"].startswith("stale-row inference supported")),
    ("inactive rows are never a valid-state finding", lambda r: all(not p["valid_differ"] for p in r["pairs"].values()))])
run("S2_live_conv_rows_differ", dict(reps=TWO, variant={"r2": 1}, core_diff={"r2": [(0, "kda0_conv_live", 0)]}), "slice", [
    ("stale-row inference refuted", lambda r: "refuted" in r["conv_rows_test"]["reading"])])
run("T_unwritten_pools_differ", dict(reps=TWO, pools={"r1": 3, "r2": 5}), "equal", [
    ("pool 511 of layer-7 block 0: nonzero and unequal, reported", lambda r: first_pair(r)["slices"]["r0:idx7_b0"]["unwritten_pools"]["511"]
     == dict(zero_a=False, zero_b=False, equal=False)),
    ("the rest of the indexer block equal", lambda r: first_pair(r)["slices"]["r0:idx7_b0"]["rest_records_differ"] == 0)])
run("U_causality", dict(reps=ALL4, **diffs(ALL4, "L3.moe_out", 3000, mla_pos=100)), "slice", [
    ("causality check rejects the located boundary", lambda r: "causality" in r["why_not_located"])])
sp = diffs(TWO, None, 0)
run("V_lifecycle_only", dict(reps=TWO, **sp, **{"break": {"r1": ["loc_state_equal"], "r2": ["loc_state_equal"]}}), "slice", [
    ("no warm boundary differs; point-0 state != last write: lifecycle reading", lambda r: "lifecycle" in r["why_not_located"])])
run("V2_outside_measured_layers", dict(reps=TWO, **diffs(TWO, None, 0, kda=False)), "slice", [
    ("only MLA-7 differs, no layer-3/4 boundary: outside the measured layers", lambda r: "outside layers 3-4" in r["why_not_located"])])
run("Y1_all_zero_valid_slice", dict(reps=TWO, **diffs(TWO, "L3.moe_out", 2100), **{"break": {"r2": ["zero"]}}), "incomplete", [
    ("all-zero MLA-7 block -> addressing suspect", lambda r: any("G-content" in w for w in r["why"]))])
run("Y2_file_identity", dict(reps=TWO, **diffs(TWO, "L3.moe_out", 2100), **{"break": {"r1": ["meta_rank"]}}), "incomplete", [
    ("capture file's own rank != its name -> INCOMPLETE", lambda r: any("differ from its name" in w for w in r["why"]))])
run("Y3_loc_input_ids_differ", dict(reps=ALL4, **diffs(ALL4, "L3.moe_out", 2100), **{"break": {"r2": ["loc_ids"]}}), "slice", [
    ("localisation INCOMPLETE noted; not located", lambda r: any("localisation INCOMPLETE" in n for n in r["notes"]))])
run("Z_no_client_records", dict(reps=[]), "incomplete")
import numpy as np  # noqa: E402
a = np.arange(16, dtype=np.int32).reshape(2, 8)
arr = {"diffcur:L3.topk.ids": a[:, ::-1].copy().view(np.uint8), "diffref:L3.topk.ids": a.view(np.uint8),
       "diffpos:L3.topk.ids": np.array([2051, 2052])}
t = A.topk_sets(arr)
check("A2 topk_sets: reversed rows = same set, other order", t["same_set_other_order"] == 2 and t["different_selection"] == 0, t)
b = a.copy()
b[1, 0] = 99
arr["diffcur:L3.topk.ids"] = b.view(np.uint8)
t = A.topk_sets(arr)
check("A2 topk_sets: a changed id = different selection", t["same_set_other_order"] == 1 and t["different_selection"] == 1, t)
print("analysis tests: %d PASS, %d FAIL" % (N["PASS"], N["FAIL"]))
sys.exit(1 if N["FAIL"] else 0)
