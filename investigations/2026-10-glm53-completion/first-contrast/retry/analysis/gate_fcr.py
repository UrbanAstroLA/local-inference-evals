#!/usr/bin/env python3
"""GATE-1R: integrated validation gate of the retry session BEFORE any injected traffic (PLAN.md amendment 22d s.9;
RETRY-PROPOSAL.md s.6). CPU only. Frozen before data.

usage: gate_fcr.py RUN_DIR [FC_ROOT]   -> RUN_DIR/gate1r.json, gate1r.txt; rc 0 iff verdict PASS
RUN_DIR holds client.jsonl, forcewin/ and mapguard/ (stage-A records copied out by runner/gate_step_fcr.sh).

Rows (status PASS / FAIL gate; REPORTED never gates). Attempt-1 functions are reused unchanged where the requirement
is unchanged (../../analysis/gate_fc.py, fc_common.py); every changed row is labelled R-n and the attempt-1 row is kept
beside it as REPORTED.
  reference builds   G1-V0; G1-coverage, G1-wmax, G1-abspos, G1-forcing, G1-errors, G1-attrib (unchanged)
  cache [R-2/F2]     R-cache-safe (S1-S4 for every stage-A request: block-aligned, never beyond the prefill coverage,
                     no decode-written history restored, scheduler-attributed), R-cache-model (every stage-A lookup
                     classified as the F7 replay predicts: all hits) ; R-cache-class, R-cache-identity reported ;
                     G1-cache-orig (attempt-1 exact equality, now with the F1 values) reported
  concurrency        G1-inflight (unchanged: all 8 decoding before output index 512)
  clean              G1-clean (unchanged)
  alignment          G1-align, G1-cross, G1-regime (unchanged rules)
  numerical [R-6/F6] G1-content (21g-1 pooled, unchanged) ; R-idx-struct (exact attribution and population) ;
                     R-idx-pages, R-idx-early (null-relative, matched) ; tables, flags, G1-idx-struct-orig reported
  conv state         G1-CV-* (unchanged)
  sentinel [R-5/F5]  R-PS-coverage, R-PS-identity, R-PS-bytes, R-PS-state-ssm, R-PS-state-conv on PS-a / PS-b (the
                     PS-c comparison is post-session; SSM and conv separately, each its own baseline [R-10]);
                     R-PS-limits reported (22f s.5) [R-13]
  [R-12] R-idx-pages-nd reported: pages whose null makes the page limit non-discriminating, "not independently
                     verified by the numerical check" (22f s.4)
Early-stop class of a failure (22d s.9): unsafe restoration | insufficient concurrent decoding | long-context alignment
failure | uninterpretable numerical content | missing boundary or shared-prefix coverage | structural.
Pre-stated reading: a failing numerical row means the long-context instrument is unvalidated under these bounds; B1 is
not thereby defective and no threshold is revised after viewing data. A pass reads "compatible with measured
long-context numerical variation".
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RETRY = os.path.dirname(HERE)
FCROOT = os.path.dirname(RETRY)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(FCROOT, "analysis"))
import cache_gate as cg  # noqa: E402
import f6_criteria as f6  # noqa: E402
import fc_common as fc  # noqa: E402
import fcr_geometry as geo  # noqa: E402
import gate_fc as gt  # noqa: E402
import sentinel as ps  # noqa: E402

STOP_CLASS = {
    "R-cache-safe": "unsafe restoration", "R-cache-model": "unsafe restoration (replay model not confirmed)",
    "G1-inflight": "insufficient concurrent decoding",
    "G1-align": "long-context alignment failure", "G1-cross": "long-context alignment failure",
    "G1-regime": "long-context alignment failure",
    "G1-content": "uninterpretable numerical content", "R-idx-pages": "uninterpretable numerical content",
    "R-idx-early": "uninterpretable numerical content", "G1-CV-conv": "uninterpretable numerical content",
    "G1-CV-ssm": "uninterpretable numerical content",
    "R-PS-coverage": "missing boundary or shared-prefix coverage", "R-PS-identity": "missing boundary or shared-prefix "
    "coverage", "G1-CV-cross": "missing boundary or shared-prefix coverage", "G1-coverage": "missing coverage",
}
STAGE_A = ("V0", "LW1", "LW2", "warm-s1-a", "T-s1-a", "warm-s1-b", "T-s1", "LC1", "LC2", "C1", "CV", "warm-s2-a",
           "T-s2-a", "warm-s2-b", "T-s2", "C2", "PS-a", "PS-b")


def demote(R, rid, new_id, note):
    for r in R.rows:
        if r["id"] == rid:
            r["id"], r["kind"], r["status"] = new_id, "reported", "REPORTED"
            r["name"] = "ORIGINAL attempt-1 row kept beside (" + note + "): " + r["name"]
            return r
    return None


def cache_rows(run, rd):
    reqs = [r for r in run.reqs]
    info = cg.sched_requests(cg.load_sched(rd))
    hook = {x.get("req"): x.get("tag") for x in run.FWLOG if x.get("kind") == "add"}
    m = cg.map_client(reqs, info, hook)
    return cg.evaluate(reqs, info, m), m


def gate1r(run, G, rd, variant):
    R = gt.Rows()
    pr = [r for r in run.CL if r.get("kind") == "probe"]
    R.add("G1-V0", "structural", "PASS" if len(pr) == 4 and all(p.get("equal") for p in pr) else "FAIL",
          "V0: unkeyed probes bitwise equal to B1's record (hook on, WMAX 8448)", [(p["id"], p.get("equal")) for p in pr])
    ran = {r["phase"] for r in run.reqs}
    runs = [p for p in (["LC1", "LC2"] if variant == "B" else []) if p in ran] + ["C1", "C2"]   # LP may be skipped
    gt.structural(run, G, R, runs, ["LW1", "LW2"])
    demote(R, "G1-cache", "G1-cache-orig", "exact equality with the F1 expected values; superseded by R-cache-*")
    # ---- F2 cache rows
    rows, mapping = cache_rows(run, rd)
    rowsA = [x for x in rows if x["phase"] in STAGE_A]
    unsafe = [(x["tag"], x["cls"], x["cached"], x["expected"], x.get("S4_note")) for x in rowsA if not x["safe"]]
    R.add("R-cache-safe", "structural", "PASS" if rowsA and not unsafe else "FAIL", "cache safety (exact) for every "
          "stage-A request: S1 block-aligned, S2 never beyond the prefill coverage of its stream under its salt, S3 no "
          "decode-written history restored, S4 scheduler-attributed (num_computed == cached; restored ids written by "
          "an earlier prefill of the same fixture and salt)", dict(n=len(rowsA), unsafe=unsafe[:12],
                                                                   unmapped=[x["tag"] for x in rowsA
                                                                             if not x["engine_id"]][:8]), "R-2 / F2")
    nonhit = [(x["tag"], x["cls"], x["cached"], x["expected"]) for x in rowsA if x["cls"] not in ("hit", "cold")]
    R.add("R-cache-model", "structural", "PASS" if rowsA and not nonhit else "FAIL", "the F7 replay's prediction holds "
          "on stage A: every lookup is a full hit (every build cold); injected traffic relies on the same model (the "
          "D2 share across the gate, the touch passes, the split builds)", dict(non_hits=nonhit), "R-1 / R-3")
    R.add("R-cache-class", "reported", "REPORTED", "hit / partial / miss / cold per request and per phase (construction "
          "accounting)", cg.run_summary(rows, [p for p in STAGE_A if p != "V0"]), "R-2")
    shares = [[p for p in ("LC1", "LC2", "C1", "CV") if p in ran], ["C2", "PS-a", "PS-b"]]
    R.add("R-cache-identity", "reported", "REPORTED", "users of one build restore identical physical MLA / index-K "
          "blocks (scheduler block ids, common length)", cg.identity(rows, shares), "R-2")
    # ---- unchanged concurrency / clean / alignment
    gt.clean_checks(run, G, R, runs)
    regime = gt.alignment(run, G, R, [("C1", "LW1"), ("C2", "LW2")])
    gt.regime_check(run, G, R, regime, ["C1", "C2"])
    # ---- numerical: pooled content (21g-1) + F6
    gt.content_gate(run, G, R, ["C1", "C2"], ("C1", "C2"), ("LW1", "LW2"), "LW1", extra_ref="LW2")
    orig = demote(R, "G1-idx-struct", "G1-idx-struct-orig", "page <= 25 % and zero early-layer gross; failed on the "
                  "prefill null itself at long context (diagnosis-fc1 s.5)")
    det = (orig or {}).get("detail") or {}
    sbad = bool(det.get("attrib")) or bool(det.get("failing_population"))
    R.add("R-idx-struct", "structural", "FAIL" if sbad or orig is None else "PASS", "index-K structure (exact): every "
          "dump attributed to its request and token position; every comparison >= 99 % of its expected units",
          dict(attrib=det.get("attrib"), failing_population=det.get("failing_population")), "R-6 / F6")
    PP = f6.rows_for(run, G, "LW1", "LW2")
    DD = f6.rows_for(run, G, "C1", "C2")
    ev = {t: f6.evaluate(f6.rows_for(run, G, t, "LW1"), DD, PP) for t in ("C1", "C2")}
    R.add("R-idx-pages", "numerical", "PASS" if all(e["pages"]["ok"] for e in ev.values()) else "FAIL",
          "every judged page (fixture x rank x layer x 64 pools): gross fraction <= max(2 x N_pp page, 0.25), the "
          "prefill null on the same page (N_dd contains the tested run: reported, not used); non-discriminating pages "
          "(R-idx-pages-nd) are reported as %s" % f6.ND_WORDING, {t: e["pages"] for t, e in ev.items()}, "R-6 / F6")
    R.add("R-idx-early", "numerical", "PASS" if all(e["early"]["ok"] for e in ev.values()) else "FAIL",
          "DSA layers 3-15 as one stratum: gross <= max(2 x N_pp, 0.1 %) on matched units (N_dd reported)",
          {t: e["early"] for t, e in ev.items()}, "R-6 / F6")
    R.add("R-idx-pages-nd", "reported", "REPORTED", "pages whose matched prefill null makes the page limit "
          "non-discriminating (limit >= %.2f): %s" % (f6.ND_BOUND, f6.ND_WORDING),
          {t: dict(n=e["pages"]["non_discriminating"]["n"], of_judged=e["pages"]["pages_judged"],
                   n_limit_ge_1=e["pages"]["non_discriminating"]["n_undetectable"],
                   by_fixture=e["pages"]["non_discriminating"]["by_fixture"],
                   pages=e["pages"]["non_discriminating"]["pages"]) for t, e in ev.items()}, "R-12 / 22f s.4")
    R.add("R-idx-tables", "reported", "REPORTED", "F6 by fixture, by fixture x layer, flags (R3), originals beside",
          {t: dict(by_fixture=e["by_fixture"], flags=e["flags"], original=e["original"]) for t, e in ev.items()}, "R-6")
    # ---- conv state, sentinel
    gt.cv_checks(run, G, R)
    ps.check(run, G, R, rows, phases=("PS-a", "PS-b"), build_users=("C2",), prefix="R-PS")
    rr = {}
    for n in G["order"]:
        a, b = run.req(f"LW1-{n}"), run.req(f"LW2-{n}")
        if a is not None and b is not None:
            s = fc.align_summary(a, b, 0, G["fx"][n]["W_s"] + geo.LP_SHIFT - 1024)
            rr[n] = {k: v for k, v in s.items() if not k.startswith("_")}
    R.add("G1-refref", "reported", "REPORTED", "prefill vs prefill (LW1 vs LW2) on shared rows: the reference-side "
          "alignment null", rr)
    fs, fn = R.failing("structural"), R.failing("numerical")
    v = "PASS" if not fs and not fn else "FAIL"
    classes = sorted({STOP_CLASS.get(x, "structural") for x in fs + fn})
    reading = ("PASS: compatible with measured long-context numerical variation under these bounds; the cache and "
               "concurrency properties the contrast needs hold; injected batches may run" if v == "PASS" else
               "STOP (%s): structural %s numerical %s. A failing numerical row means the long-context instrument is "
               "unvalidated under these bounds; it does not make B1 defective; no threshold is revised after viewing "
               "data" % ("; ".join(classes), fs, fn))
    return R, v, reading, classes, rows


def main():
    rd = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else FCROOT
    G = geo.load(root)
    run = fc.Run(rd)
    start = [r for r in run.CL if r.get("kind") == "start"]
    variant = (start[0].get("variant") if start else None) or "B"
    try:
        R, v, reading, classes, _ = gate1r(run, G, rd, variant)
    except Exception as exc:  # noqa: BLE001
        R, v, reading, classes = gt.Rows(), "FAIL", "gate raised: %r" % exc, ["structural"]
        R.add("G1R-raised", "structural", "FAIL", "gate code raised", repr(exc)[:500])
    nd = next((r["detail"] for r in R.rows if r["id"] == "R-idx-pages-nd"), None)
    out = dict(gate="GATE-1R", verdict=v, reading=reading, stop_classes=classes, rows=R.rows,
               not_independently_verified=dict(wording=f6.ND_WORDING, pages=nd),
               sentinel_limits=ps.LIMITS, numerical_wording=f6.WORDING,
               failing=dict(structural=R.failing("structural"), numerical=R.failing("numerical")),
               flagged_fixtures=R.data.get("flagged", []))
    json.dump(out, open(os.path.join(rd, "gate1r.json"), "w"), indent=1, default=str)
    lines = ["GATE-1R %s  %s" % (v, reading)] + ["%-8s %-22s %-10s %s  [%s]" % (
        r["status"], r["id"], r["kind"], r["name"][:110], json.dumps(r["detail"], default=str)[:220]) for r in R.rows]
    if nd:
        lines.append("F6 pages %s (non-discriminating null, limit >= %.2f): %s" % (
            f6.ND_WORDING, f6.ND_BOUND, "; ".join("%s %d of %d judged pages (%d with limit >= 1.0)" % (
                t, x["n"], x["of_judged"], x["n_limit_ge_1"]) for t, x in nd.items())))
    lines.append("SENTINEL LIMITS: " + ps.LIMITS)
    open(os.path.join(rd, "gate1r.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if v == "PASS" else 1)


if __name__ == "__main__":
    main()
