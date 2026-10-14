#!/usr/bin/env python3
"""Hook-validation analysis, ATTEMPT 2 (PA2-7, PA2-16; attempt-1 analysis kept unchanged as analyze_hookval.py.a1;
the pre-21g attempt-2 analysis as analyze_hookval.py.pre21g). CPU only.

usage: analyze_hookval.py RUN_DIR
  -> RUN_DIR/verdict.json, verdict.txt          ORIGINAL criteria: the frozen attempt-1 code (analyze_hookval.py.a1)
                                                run verbatim on this run; nothing in it is changed.
  -> RUN_DIR/verdict-a2.json, verdict-a2.txt    REVISED criteria (post-inspection fixes PA2-7a..f and the criteria of
                                                PLAN.md amendment 21g, PA2-16) beside every original check.
HV_ANALYSIS_OUT=DIR writes the verdicts to DIR instead (replays of read-only runs).

Every original check stays computed (verdict.json) and every revised row names its original check(s) and their
result. Where 21g changed a pre-21g DRAFT rule, the draft form is kept beside it as a REPORTED row.

Post-inspection analysis fixes (DIAGNOSIS-a1 s.8.1; no frozen threshold loosened by them):
  PA2-7a (PI-1) H2 reference join: decode row o predicts position P + o; reference row k predicts ref_from + k, so
         k = o + P - ref_from (ref_from from the hook record's `ref`, else the client's ref_from); top-1 counted only
         where the reference row is finite. Frozen NULL_FACTOR, floors and TOP1_MIN.
  PA2-7b (PI-2) argmax shift control beside the frozen lp rule: per request aligned top-1 >= 0.90 and top-1 at a
         +-1 shift <= 0.20 (H2c-r, invalid level); aggregate (H2b-r, inconclusive level). The lp rule stays
         reported, labelled "uninformative when the forced lp is ~0".
  PA2-7c (PI-3) guard: polls read per `rank`; counters taken as deltas from the client's start-dump poll (the first
         "trigger" poll at or after the client's start dump) to the last poll; per-phase attribution by dumps.
  PA2-7d H5h: designated positions exclude o = 0 (predicted by prefill) and requests with W <= 1; a record counts
         only if its step equals the position's measurement step (else it is misattributed); selection records too.
  PA2-7e window comparisons exclude pools completing at P + W - 1 (committed, never processed): pools i with
         4i + 3 <= P + W - 2.
  PA2-7f H1f: rows the served kernel left unmasked (p_post == p_pre for every tracked id) are reported as their own
         class with their kept mass; gate: no kept-not-eligible cell with p_post > 1e-6 and no masked-but-eligible
         cell with p_pre > 1e-6.

PA2-16: PLAN.md amendment 21g, FROZEN HERE BEFORE ANY ATTEMPT-2 DATA (ATTEMPT2-21G.md):
 1. [21g-1] Index-K content (IDX-content, H5e-r, H5d-r, H5f-D2).
    PA2-26 (amendment 21h, 21g's "the aggregate is the gate, fixture-level deviations are flags"): H5e-r gates the
    solo request against the POOLED matched null of the eight-fixture set (units present in both nulls of each
    fixture, same exclusions, pooled denominators); its comparison with its own fixture's matched null is a FLAG
    (H5e-fixture, REPORTED); the pre-21g pooled form (H5e-pooled) stays reported. The text below describes H5e-r
    as frozen by PA2-16 (strict per-fixture limit); that form now survives only as the H5e-fixture flag.
    Unit: (fixture, rank, DSA index layer, pool); pools of a window comparison are those wholly inside the window
    and processed: (P+3)//4 <= i and 4i+3 <= P+W-2 (PA2-7e); every unit once (window dumps are one snapshot).
    rel = ||x - y|| / max(||y||, 1e-12) of the dequantised fp8 records; GROSS = rel > 0.25 (frozen attempt-1 value).
    Tested comparisons: B1 arm vs rebuild (V4-b1, V4n-b1b), solo injected vs rebuild (V4-d2solo), D2 arm vs rebuild.
    Nulls, per fixture, over the SAME windows: N_dd = V4-b1 vs V4n-b1b (decode vs decode, batch 8 vs batches of 4);
    N_pp = V4-rebuild vs V4n-rebuild2 (prefill vs prefill, build length +200 ids).
    MATCHING: a tested unit is evaluated only if the same (fixture, rank, layer, pool) exists in BOTH nulls; the
    limit of a tested comparison is computed from the nulls restricted to exactly its evaluated units:
        L = max(2 x max(gross N_dd, gross N_pp), 0.1 %)     (0.1 % = operational floor, not a measured noise floor)
    Denominators: rates are pooled over units (micro-average: gross units / evaluated units), never a mean of
    per-fixture rates. Exclusions: only PA2-7e window ends, the straddling first pool, and units missing a matched
    null (counted and reported). Null coverage: evaluated / tested units >= 0.99 and every fixture of the comparison
    has evaluated units, else the content row FAILS at the inconclusive level (nulls not measured as required).
    Reported: gross counts and rates per fixture x layer and pooled, for every tested comparison and both nulls.
    Fixture flag (REPORTED; a flagged fixture cannot carry a mechanism claim): fixture f of a tested comparison is
    "substantially outside its own null" iff gross_f > max(2 x max(N_dd_f, N_pp_f), 0.1 %) on its own matched
    units AND it has >= 10 gross units. Because N_dd = b1 vs b1b shares an arm with each B1 comparison, a defect
    confined to ONE decode arm inflates N_dd and is absorbed by the limit; such fixtures are flagged separately
    ("arm asymmetry": the grosser arm has >= 10 gross units, > 2x the other arm's and > max(2 x N_pp_f, 0.1 %),
    on the units both arms and N_pp share).
 2. [21g-1] Structural index-K checks (exact): IDX-attrib (every window dump / snapshot carries its request's tag, P and
    expected range; every comparison compares >= 99 % of its expected units), IDX-page (no judged page > 25 %
    gross; unique pools; pages judged with >= 16 compared pools), IDX-early (zero gross in DSA layers 3-15),
    IDX-stable (committed pools byte-identical between consecutive V3 snapshots), H5f-B1 (B1 ledger all clean).
 3. [21g-2] KDA. (a) H4-slot, exact: the snapshot's committed slot count num_accepted equals the hook's own accepted count
    at the snapshot step + 1, 1 <= num_accepted <= slots, every KDA layer speculative with the same slot count, and
    the snapshot belongs to its request (tag, P, F). (b) H4b-r, nearest-reference identification as a diagnostic
    screen: 100 % in layers 0-17, >= 99 % in layers >= 18; layers 0-2 rel <= 0.02 and margin >= 10 (layers 0-2
    ONLY). Every disagreement is reported with its distances and margin (H4b-disagree). H4b-sep (inconclusive
    level): a wrong slot nearer than the committed slot by a factor >= 1.25 (committed distance / nearest distance;
    attempt-1 correct-slot margins at depth fell to 1.04, 14 below 1.25) is a comfortably separated wrong-slot match
    and requires investigation. The pre-21g "100 % in every layer" is kept as H4b-draft100 (REPORTED).
 4. [21g-3] H2/H3 carry-over: CARRIED rows are carried-over evidence (attempt-1 run, corrected join, provenance in
    carryover-a1.json); CARRY-scope checks the image scope (only vllm/forcewin_hooks.py differs between the
    attempt-1 and attempt-2 images) and that this run used the attempt-2 image; CARRY-admissible requires scope plus
    the current-image V0, forcing (H1a-e, H1g-j), attribution (H5h-r, IDX-attrib), ledger (H5f-B1) and state (H4a,
    H4-slot) checks. Current-image alignment and the argmax shift control (H2a-r, H2b-r, H2c-r) are computed from the
    overlapping V2a/V2c reference rows. H3-r is recomputed only when the optional V2a-poison phase completed.
 5. [21g-3] Coverage (coverage-a2): every mandatory phase and every expected mandatory request present without error; any
    skip or coverage-gap record fails; the optional poison phase is reported as ran / optional_skip / incomplete,
    and a poison phase that is neither run nor recorded as skipped fails (no silent drop).
 6. [21g-4] top_p edge case: H1f-r gates the TRACKED ids (bound applies to them); unmasked rows are their own class
    (H1f-unmasked, REPORTED) with the total pre-mask mass outside the intended nucleus where the records determine it
    (forced token is the argmax and mass_lt <= 1 - top_p, so the nucleus is the top token alone and the
    outside-nucleus mass is mass_lt of the forced token); elsewhere "not determinable from the existing records".
Out-of-scope items (V1, V2b, reject/sched schedules) carry over from attempt 1 (carryover-a1.json): status CARRIED,
never gating by themselves.
"""
import collections
import importlib.machinery
import importlib.util
import json
import os
import re
import sys

import numpy as np
import torch

RUN = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.environ.get("HV_ANALYSIS_OUT") or RUN      # replays of a read-only run write elsewhere (default: RUN)
os.makedirs(OUTD, exist_ok=True)

# ------------------------------------------------------------------ 1. ORIGINAL: the frozen attempt-1 analysis
_loader = importlib.machinery.SourceFileLoader("analyze_hookval_a1", os.path.join(HERE, "analyze_hookval.py.a1"))
_spec = importlib.util.spec_from_loader("analyze_hookval_a1", _loader)
A1 = importlib.util.module_from_spec(_spec)
_loader.exec_module(A1)                      # module level loads RUN (sys.argv[1]) exactly as attempt 1 did
for _fn in (A1.coverage, A1.h1, A1.h2, A1.h3, A1.h4, A1.h5):     # as its __main__, verbatim
    try:
        _fn()
    except Exception as _exc:  # noqa: BLE001
        A1.check(f"{_fn.__name__} raised", False, repr(_exc)[:300])
VO = A1.V
VO["verdict"] = "INVALID" if VO["invalid"] else ("INCONCLUSIVE" if VO["inconclusive"] else "VALID")
json.dump(VO, open(os.path.join(OUTD, "verdict.json"), "w"), indent=1, default=str)
_lines = [f"VERDICT {VO['verdict']}  (ORIGINAL frozen criteria, attempt-1 code verbatim)"] + \
    ["%s %s  [%s]" % ("PASS" if c["ok"] else "FAIL", c["name"], c["detail"]) for c in VO["checks"]]
open(os.path.join(OUTD, "verdict.txt"), "w").write("\n".join(_lines) + "\n")
ORIG = {c["name"].split(" ")[0]: c for c in VO["checks"]}

# shared data (already loaded by the frozen module)
CL, REQ, SNAP, FWLOG, POLLS, reqs = A1.CL, A1.REQ, A1.SNAP, A1.FWLOG, A1.POLLS, A1.reqs
r0, q, cols = A1.r0, A1.q, A1.cols
GROSS = A1.IDX_GROSS
CARRY = json.load(open(os.path.join(HERE, "carryover-a1.json")))

# ------------------------------------------------------------------ 2. REVISED (labelled) checks
# PA2-7 thresholds (ATTEMPT2-CRITERIA-DRAFT.md; each justified there from attempt-1 measurements)
H1F_MASS = 1e-6           # PA2-7f: tracked mass that a served-mask anomaly may carry (attempt-1 max 2.6e-9)
TOP_P = 0.95              # the client's top_p (every forced request)
SHIFT_ALIGNED_MIN = 0.90  # PA2-7b: per-request aligned top-1 (attempt-1 min 0.951)
SHIFT_SHIFTED_MAX = 0.20  # PA2-7b: per-request top-1 at +-1 (attempt-1 max 0.021)
KDA_EARLY_LAYERS = (0, 1, 2)
KDA_EARLY_REL = 0.02      # attempt-1 max 0.0089
KDA_EARLY_MARGIN = 10.0   # attempt-1 min 18.7
PAGE_MINP = 16            # pages judged only with >= 16 compared pools (each pool once)
PAGE_MAX_GROSS = 0.25     # attempt-1 aligned max 0.094 (V4) / 0.10 (V3, deduplicated); planted wrong page: 1.0
EARLY_IDX_LAYERS = 15     # zero gross in DSA layers 3..15 (attempt 1: 0 in every comparison; planted: 99.9 %)
D2_GROSS_MIN = A1.D2_GROSS_MIN
SHADOW_CLEAN_OK = A1.SHADOW_CLEAN_OK
# PA2-16: amendment 21g (frozen before attempt-2 data)
CONTENT_FACTOR = 2.0      # gross <= max(2 x max(N_dd, N_pp), floor), nulls matched to the tested units
CONTENT_FLOOR = 0.001     # operational tolerance (the original 0.1 %), not a measured noise floor
NULL_COVER_MIN = 0.99     # evaluated (matched) / tested units, per tested comparison
POP_MIN = 0.99            # compared / expected units of every window comparison (population)
FLAG_MIN_GROSS = 10       # fixture flag needs >= 10 gross units ...
FLAG_FACTOR = 2.0         # ... and gross_f > max(2 x max(N_dd_f, N_pp_f), CONTENT_FLOOR)
KDA_SHALLOW_MAX = 17      # identification 100 % in layers 0..17
KDA_DEEP_IDENT_MIN = 0.99  # >= 99 % in layers >= 18
KDA_SEP_FLAG = 1.25       # wrong slot nearer by >= 1.25x: comfortably separated, requires investigation

R = {"label": "REVISED criteria: post-inspection fixes PA2-7a..f + PLAN.md amendment 21g criteria (PA2-16)",
     "rows": [], "invalid": [], "inconclusive": []}
STATUS = {}


def row(rid, orig_ids, name, status, detail, level="invalid", label=""):
    """status: PASS / FAIL (gating at `level`), CARRIED (from attempt 1, never gating), REPORTED (not gating)."""
    R["rows"].append(dict(id=rid, original=orig_ids, name=name, status=status, level=level, label=label,
                          detail=detail, original_result=[("PASS" if ORIG[o]["ok"] else "FAIL") if o in ORIG else
                                                          "absent" for o in orig_ids]))
    STATUS[rid] = status
    if status == "FAIL":
        R[level].append(rid)


def same_as_original(oid, level="invalid"):
    c = ORIG.get(oid)
    if c is None:
        row(oid, [oid], oid, "FAIL", "original check absent", level)
        return
    row(oid, [oid], c["name"], "PASS" if c["ok"] else "FAIL", c["detail"], c["level"], "unchanged")


def layer_no(name):
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def forced_reqs():
    return [r for r in reqs if r.get("W", 0) > 0 and not r["tag"].startswith("V3ref")]


def start_rec():
    s = [r for r in CL if r.get("kind") == "start"]
    return s[0] if s else {}


# ------------------------------------------------------------------ coverage / V0 / H1
def r_coverage():
    """PA2-16 item 5: required coverage is never dropped silently."""
    same_as_original("V0")
    mand = ["V0", "V2a-natural", "V2a-oracle", "V2c", "V4-d2", "V4-b1", "V4-d2solo", "V4-rebuild", "V4n-b1b",
            "V4n-rebuild2", "V3-forced", "V3-refs"]
    ph = {r["tag"] for r in CL if r.get("kind") == "phase"}
    sk = [r for r in CL if r.get("kind") in ("skip", "coverage_gap")]
    miss = [p for p in mand if p not in ph]
    st = start_rec()
    matrix, d2set = st.get("matrix") or [], st.get("d2set") or []
    exp = [f"V2a-{m}-{t}" for m in ("natural", "oracle") for t in matrix] + [f"V2c-{t}" for t in matrix]
    exp += [f"V4-{a}-{t}" for a in ("d2", "b1", "rebuild") for t in d2set] + \
        [f"V4n-{a}-{t}" for a in ("b1b", "rebuild2") for t in d2set] + ([f"V4-d2solo-{d2set[0]}"] if d2set else [])
    by = {}
    for r in reqs:
        by.setdefault(r["tag"], []).append(r)
    absent = [t for t in exp if t not in by]
    errs = [r["tag"] for r in reqs if r.get("error")]
    v3 = [t for t in matrix if not any(x.startswith("V3-") and x.endswith(t) for x in by)]
    # every V3 decode snapshot has its prefill reference (request and snapshot)
    snaps = sorted({(tag, F) for (tag, F, win) in SNAP if tag.startswith("V3-") and not win})
    v3ref_missing = [(t, F) for t, F in snaps if not any(k[0] == f"V3ref-F{F}-{t}" and not k[2] for k in SNAP)]
    # optional poison: ran / optional_skip / incomplete / silently absent
    pois_req = [t for t in by if t.startswith("V2a-poison-")]
    oskip = [r for r in CL if r.get("kind") == "optional_skip" and r.get("mode") == "poison"]
    if "V2a-poison" in ph and len(pois_req) == len(matrix):
        pstat = "ran"
    elif oskip:
        pstat = f"optional_skip (left {oskip[0].get('left')} s, needed {oskip[0].get('need')} s)"
    elif pois_req:
        pstat = f"incomplete ({len(pois_req)} requests, no phase record): treated as not run"
    else:
        pstat = "ABSENT WITHOUT A RECORD"
    covrec = [r for r in CL if r.get("kind") == "coverage"]
    ok = not sk and not miss and not absent and not errs and not v3 and not v3ref_missing and matrix and d2set \
        and pstat != "ABSENT WITHOUT A RECORD"
    R["coverage"] = dict(missing_phases=miss, skips=sk[:5], absent_requests=absent[:10], request_errors=errs[:10],
                         v3_tids_without_request=v3, v3_snapshots_without_reference=v3ref_missing[:10],
                         expected_requests=len(exp), poison=pstat, client_coverage_record=covrec[-1] if covrec else None)
    row("coverage-a2", ["coverage:"], "every mandatory phase and expected mandatory request present without error; no "
        "skip or coverage gap; every V3 snapshot referenced; optional V2a-poison reported, never silently dropped",
        "PASS" if ok else "FAIL", json.dumps(R["coverage"], default=str)[:700], "inconclusive", "PA2 scope; PA2-16 (21g-3)")


def r_h1():
    for oid in ("H1a", "H1b", "H1c", "H1d", "H1e", "H1g", "H1h", "H1i", "H1j"):
        same_as_original(oid)
    bad, unm, cells = [], [], 0
    for r in forced_reqs():
        a = r0(r["tag"])
        if a is None:
            continue
        Cc = cols(a)
        M = a["metrics"][: r["W"]].numpy().astype(np.float64)
        pp = M[:, Cc["p_post_0"]:Cc["p_post_0"] + 5]
        lo = M[:, Cc["elig_lo_0"]:Cc["elig_lo_0"] + 5]
        hi = M[:, Cc["elig_hi_0"]:Cc["elig_hi_0"] + 5]
        pre = np.exp(M[:, Cc["lp_pre_0"]:Cc["lp_pre_0"] + 5])
        kept_ne = (pp > 0) & (hi < 0.5)
        mask_e = (lo > 0.5) & (pp <= 0)
        cells += int(kept_ne.sum() + mask_e.sum())
        rows_v = np.where((kept_ne | mask_e).any(axis=1))[0]
        for i in rows_v:
            unmasked = bool(np.allclose(pp[i], pre[i], rtol=1e-4, atol=1e-12))
            mass = float(pp[i][kept_ne[i]].sum())
            # PA2-16 (21g-6): total mass outside the intended nucleus, where the records determine it
            mlt = float(M[i, Cc["mass_lt_4"]])
            forced_top = M[i, Cc["argmax_is_forced"]] == 1.0
            det = bool(forced_top and np.isfinite(mlt) and mlt <= 1.0 - TOP_P)
            rec = dict(tag=r["tag"], o=int(i), unmasked=unmasked, kept_mass=mass,
                       max_kept_p=float(pp[i][kept_ne[i]].max()) if kept_ne[i].any() else 0.0,
                       max_masked_eligible_p_pre=float(pre[i][mask_e[i]].max()) if mask_e[i].any() else 0.0,
                       top1_p=float(M[i, Cc["top1_p"]]), outside_nucleus_mass=mlt if det else None)
            (unm if unmasked else bad).append(rec)
    allv = unm + bad
    gate_bad = [v for v in allv if v["max_kept_p"] > H1F_MASS or v["max_masked_eligible_p_pre"] > H1F_MASS]
    onm = [v["outside_nucleus_mass"] for v in unm if v["outside_nucleus_mass"] is not None]
    R["h1f"] = dict(cells=cells, rows=len(allv), unmasked_rows=len(unm),
                    unmasked_max_kept_p=max([v["max_kept_p"] for v in unm], default=0.0),
                    unmasked_tracked_kept_mass_max=max([v["kept_mass"] for v in unm], default=0.0),
                    unmasked_outside_nucleus=dict(rows_determined=len(onm), rows_not_determinable=len(unm) - len(onm),
                                                  max=max(onm, default=None), sum=float(np.sum(onm)) if onm else None,
                                                  median=float(np.median(onm)) if onm else None),
                    other_rows=len(bad), gate_violations=gate_bad[:10], unmasked_examples=unm[:10])
    row("H1f-r", ["H1f"], "served top_p mask inside the eligibility bounds on the TRACKED ids (4 termination ids + the "
        "forced token) up to 1e-6; the bound applies to the tracked ids only", "PASS" if not gate_bad else "FAIL",
        json.dumps({k: v for k, v in R["h1f"].items() if k not in ("gate_violations", "unmasked_examples",
                                                                    "unmasked_outside_nucleus")})
        + f" gate {gate_bad[:3]}", "invalid", "PA2-7f; PA2-16 (21g-4)")
    row("H1f-unmasked", ["H1f"], "rows the served kernel left unmasked (separate class): tracked kept mass, and total "
        "pre-mask mass outside the intended nucleus where the records determine it (forced token = argmax, "
        "mass_lt <= 1 - top_p); not gated, no sampler change", "REPORTED",
        json.dumps(dict(rows=len(unm), tracked_kept_mass_max=R["h1f"]["unmasked_tracked_kept_mass_max"],
                        outside_nucleus=R["h1f"]["unmasked_outside_nucleus"])), "invalid", "PA2-16 (21g-4)")


# ------------------------------------------------------------------ H2 (PA2-7a, PA2-7b), H3; carry-over (21g-3)
def ref_from(tag):
    a = r0(tag)
    if a is not None and a.get("ref") is not None:
        return int(a["ref"])
    c = [r for r in reqs if r["tag"] == tag]
    return c[0].get("ref_from") if c and c[0].get("ref_from") is not None else None


def h2_pairs():
    out = []
    for r in reqs:
        t, W = r["tag"], r.get("W", 0)
        if not W:
            continue
        key = t.split("-", 2)[-1]
        if t.startswith("V2a-"):
            refs = [f"V2c-{key}"]
        elif t.startswith(("V1-natural-", "V1-poison-")):
            refs = [f"V1-ref-{key}"]
        elif t.startswith("V2b-"):
            refs = [f"V2b-ref-{key}", f"V2c-{key}"]
        else:
            continue
        for rt in refs:
            a, ref, rf = r0(t), r0(rt), ref_from(rt)
            if a is not None and ref is not None and rf is not None:
                out.append((t, a, ref, W, rf - r["P"]))           # ref row k = o - off
                break
    return out


def collect(pairs, shift=0):
    d, t1, per = [], [], {}
    for tag, a, ref, W, off in pairs:
        Cc = cols(a)
        M = a["metrics"][:W].numpy().astype(float)
        Rf = ref["metrics"].numpy().astype(float)
        idx = np.arange(W) - off + shift
        ok = (idx >= 0) & (idx < Rf.shape[0])
        y = np.full(W, np.nan)
        ar = np.full(W, np.nan)
        y[ok] = Rf[idx[ok], Cc["lp_pre_4"]]
        ar[ok] = Rf[idx[ok], Cc["argmax"]]
        fin = np.isfinite(y)
        dd = np.abs(M[:, Cc["lp_pre_4"]] - y)[fin]
        agree = (M[:, Cc["argmax"]] == ar)[fin]
        d += list(dd)
        t1 += list(agree)
        per[tag] = (q(dd, 0.5), float(np.mean(agree)) if agree.size else float("nan"), int(fin.sum()))
    return np.array(d), (float(np.mean(t1)) if t1 else float("nan")), per


def r_h2():
    FLOOR, NF = A1.FLOOR, A1.NULL_FACTOR
    null = A1.null_gap()
    null = null[np.isfinite(null)]
    if null.size > 100:
        nq = {p: q(null, p) for p in FLOOR}
        src = f"in-session V1 (n={null.size})"
        row("H2-null-r", ["H2-null"], "H2 null available", "PASS", src, "inconclusive", "PA2-7a")
    else:
        nq = {p: CARRY["h2_null"]["quantiles"][str(p)] for p in FLOOR}
        src = f"CARRIED from attempt 1 (run-a1 V1, n={CARRY['h2_null']['n']}; carryover-a1.json)"
        row("H2-null-r", ["H2-null"], "H2 null: CARRIED-OVER EVIDENCE, attempt-1 in-session unhooked null (V1 not in "
            "scope; provenance: carryover-a1.json sources, sha256 of run-a1 and diagnosis files)", "CARRIED", src,
            "inconclusive", "PA2 scope; PA2-16 (21g-3)")
    h = CARRY["h2_attempt1_PI"]
    row("H2-a1-carried", ["H2a", "H2b", "H2c"], "CARRIED-OVER EVIDENCE: attempt-1 H2 under the corrected join (PI-1; "
        "not the frozen join), image sha256:1b9f8b15 (V1, V2a, V2b vs references)", "CARRIED",
        f"n {h['n']} top1 {h['top1']:.4f} aligned {h['aligned']} shift top1 +1 {h['shift']['top1_plus1']:.4f} "
        f"-1 {h['shift']['top1_minus1']:.4f}; per-request aligned min {h['per_request_top1_min']:.4f}, shifted max "
        f"{h['per_request_shifted_top1_max']:.4f}; H2a {'PASS' if h['H2a_pass'] else 'FAIL'}; join: "
        f"{CARRY.get('h2_join', 'PI-1')}", "invalid", "PA2-16 (21g-3)")
    tol = {p: max(NF * nq[p], FLOOR[p]) for p in FLOOR}
    pairs = h2_pairs()
    if not pairs:
        row("H2a-r", ["H2a"], "decode measurements have prefill references", "FAIL", "no pairs", "invalid", "PA2-7a")
        return
    fam = collections.Counter(p[0].split("-")[0] for p in pairs)
    g0, t0, p0 = collect(pairs)
    _, tp, pp = collect(pairs, +1)
    _, tm, pm = collect(pairs, -1)
    st = {str(p): q(g0, p) for p in FLOOR}
    R["h2"] = dict(n=int(g0.size), top1=t0, aligned=st, null=nq, null_source=src, tol={str(p): tol[p] for p in tol},
                   shift_top1=dict(plus1=tp, minus1=tm), offsets=sorted({p[4] for p in pairs}), families=dict(fam),
                   per_request={t: dict(aligned_top1=p0[t][1], shifted_top1=max(pp[t][1], pm[t][1]), n=p0[t][2],
                                        lp_median=p0[t][0]) for t in p0})
    scope = "current image: " + ", ".join(f"{k} {v}" for k, v in sorted(fam.items())) + \
        f" requests vs their prefill references, {g0.size} overlapping (finite) reference rows"
    ok = all(q(g0, p) <= tol[p] for p in FLOOR) and t0 >= A1.TOP1_MIN
    row("H2a-r", ["H2a"], "alignment: forced decode vs prefill within the null (x2, floors) and top-1 >= 0.95; "
        "reference join by position, top-1 on finite reference rows (" + scope + ")", "PASS" if ok else "FAIL",
        f"aligned {st} top1 {t0:.4f}; null {src}; tol {R['h2']['tol']}; join offsets {R['h2']['offsets']}",
        "invalid", "PA2-7a (PI-1); PA2-16 (21g-3)")
    badr = [t for t in p0 if not (p0[t][1] >= SHIFT_ALIGNED_MIN and max(pp[t][1], pm[t][1]) <= SHIFT_SHIFTED_MAX)]
    amin = min(v[1] for v in p0.values())
    smax = max(max(pp[t][1], pm[t][1]) for t in p0)
    row("H2c-r", ["H2c"], "argmax shift control per request: aligned top-1 >= 0.90 and top-1 at +-1 <= 0.20 (" + scope
        + ")", "PASS" if not badr else "FAIL", f"{len(p0)} requests; aligned min {amin:.4f}; shifted max {smax:.4f}; "
        f"failing {badr[:5]}", "invalid", "PA2-7b (PI-2); PA2-16 (21g-3)")
    lp_bad = [t for t in p0 if not (p0[t][0] < 0.1 * min(pp[t][0], pm[t][0]))]
    row("H2c-lp", ["H2c"], "frozen lp shift rule (reported; uninformative when the forced lp is ~0)", "REPORTED",
        f"failing {len(lp_bad)} of {len(p0)}: {lp_bad[:5]}", "invalid", "PA2-7b")
    row("H2b-r", ["H2b"], "aggregate argmax shift control: aligned >= 0.90, +-1 shift <= 0.20 (" + scope + ")",
        "PASS" if t0 >= SHIFT_ALIGNED_MIN and max(tp, tm) <= SHIFT_SHIFTED_MAX else "FAIL",
        f"aligned {t0:.4f}, +1 {tp:.4f}, -1 {tm:.4f}", "inconclusive", "PA2-7b (PI-2); PA2-16 (21g-3)")


def r_h3():
    ph = {r["tag"] for r in CL if r.get("kind") == "phase"}
    have = {r["tag"].split("-")[1] for r in reqs if r["tag"].startswith("V2a-")}
    if "poison" not in have or "V2a-poison" not in ph:          # 21g-3: an incomplete poison phase is "not run"
        row("H3-r", ["H3"], "CARRIED-OVER EVIDENCE: schedule invariance vs the single-token advance, attempt 1 "
            "(image sha256:1b9f8b15; V2a-poison not run to completion here)", "CARRIED",
            f"attempt 1: {'PASS' if CARRY['h3_attempt1']['ok'] else 'FAIL'}; provenance run-a1/verdict.json "
            f"sha256 {CARRY['sources']['verdict_a1']['sha256'][:16]}", "invalid", "PA2 scope; PA2-16 (21g-3)")
        return
    by = {}
    for r in reqs:
        if r["tag"].startswith("V2a-") and r.get("W", 0):
            mode, tid = r["tag"][4:].rsplit("-", 1)
            a = r0(r["tag"])
            if a is not None:
                by.setdefault(tid, {})[mode] = (a, r["W"])
    res, ok = {}, True
    for mode in sorted(have - {"poison"}):
        d, t1 = [], []
        for tid, m in by.items():
            if mode in m and "poison" in m:
                a, W = m[mode]
                b, _ = m["poison"]
                d += list(np.abs(A1.lp_forced(a, W) - A1.lp_forced(b, W)))
                t1 += list(A1.argmax(a, W) == A1.argmax(b, W))
        d = np.array(d)
        if not d.size:
            ok = False
            continue
        res[mode] = dict(n=int(d.size), top1=float(np.mean(t1)), **{str(p): q(d, p) for p in A1.FLOOR})
        ok &= all(q(d, p) <= A1.FLOOR[p] for p in A1.FLOOR) and np.mean(t1) >= A1.TOP1_DEC_MIN
    row("H3-r", ["H3"], "current image: schedules present (natural, oracle) agree with poison within the frozen floors "
        "(decode-vs-decode null not measured in scope: floors only)", "PASS" if ok else "FAIL", json.dumps(res)[:400],
        "invalid", "PA2 scope")


def r_carry_scope():
    sc = CARRY.get("scope") or {}
    img = None
    p = os.path.join(RUN, "image-id.txt")
    if os.path.exists(p):
        f = open(p).read().split()
        img = f[1] if len(f) > 1 else (f[0] if f else None)
    ok = bool(sc.get("only_hook_module_changed")) and sc.get("differing_sources") == ["vllm/forcewin_hooks.py"] \
        and img is not None and img == sc.get("image_a2")
    row("CARRY-scope", ["H2a", "H3"], "carry-over scope: only vllm/forcewin_hooks.py differs between the attempt-1 "
        "and attempt-2 images (every installed vllm file compared; hooked engine files' .pyc differ in the mtime "
        "header only) and this run used the attempt-2 image", "PASS" if ok else "FAIL",
        f"run image {img}; scope {json.dumps({k: sc.get(k) for k in ('image_a1', 'image_a2', 'only_hook_module_changed', 'differing_sources', 'files_compared')})}",
        "invalid", "PA2-16 (21g-3)")


def r_carry_admissible():
    need = ["CARRY-scope", "V0", "H1a", "H1b", "H1c", "H1d", "H1e", "H1g", "H1h", "H1i", "H1j", "H5h-r", "IDX-attrib",
            "H5f-B1", "H4a", "H4-slot"]
    bad = [n for n in need if STATUS.get(n) != "PASS"]
    row("CARRY-admissible", ["H2a", "H3"], "carried-over H2 null / H2 / H3 evidence admissible: image scope plus the "
        "current-image V0, forcing (H1a-e, H1g-j), attribution (H5h-r, IDX-attrib), ledger (H5f-B1) and state (H4a, "
        "H4-slot) checks all pass", "PASS" if not bad else "FAIL", f"not passing: {[(n, STATUS.get(n)) for n in bad]}",
        "invalid", "PA2-16 (21g-3)")


# ------------------------------------------------------------------ index-K helpers
def idx_pages_pool(L, blk_of, pool):
    return A1.idx_pool(L["pages"], blk_of, pool)


def load(p):
    return torch.load(p, weights_only=False)


def win_dump(tag):
    k = [(kk, v) for kk, v in SNAP.items() if kk[0] == tag.replace(":", "_") and kk[2]]
    return k[0][1] if k else None


ATTRIB = []          # (tag, rank, what, found, expected): exact request attribution of every dump / snapshot
ATTRIB_SEEN = set()
POP = {}             # comparison -> [compared, expected, notes]


def client_req(tag):
    c = [r for r in reqs if r["tag"] == tag]
    return c[0] if c else None


def check_dump_attrib(tag, rank, rec):
    """The dump carries its request's tag and P; the window range is the request's window: [P, P+W] for a decode
    arm; for a rebuild the window of the fixture's B1 arm (V4-b1-<fixture>), which the client's dumprange (recorded
    since attempt 2) must also equal."""
    if (tag, rank) in ATTRIB_SEEN:
        return
    ATTRIB_SEEN.add((tag, rank))
    c = client_req(tag)
    if c is None:
        ATTRIB.append((tag, rank, "client record", None, "present"))
        return
    if rec.get("tag") != tag:
        ATTRIB.append((tag, rank, "tag", rec.get("tag"), tag))
    if rec.get("P") != c["P"]:
        ATTRIB.append((tag, rank, "P", rec.get("P"), c["P"]))
    exp = [c["P"], c["P"] + c["W"]]
    if tag.startswith(("V4-rebuild-", "V4n-rebuild2-")):
        w = client_req("V4-b1-" + tag.rsplit("-", 1)[-1])
        exp = [w["P"], w["P"] + w["W"]] if w else None
        if c.get("dumprange") is not None and list(c["dumprange"]) != exp:
            ATTRIB.append((tag, rank, "client dumprange", c["dumprange"], exp))
    if list(rec.get("range") or []) != exp:
        ATTRIB.append((tag, rank, "range", rec.get("range"), exp))


def compare_windows(tagA, tagB, name=None):
    """{(rank, layer, pool): rel} between two window dumps over the window's processed pools (PA2-7e: 4i+3 <= P+W-2;
    dumps carry range [P, P+W]); counts compared vs expected units (population) under `name`."""
    A, B = win_dump(tagA), win_dump(tagB)
    out, expected = {}, 0
    if not A or not B:
        if name:
            POP.setdefault(name, [0, 0, []])[2].append(f"{tagA} vs {tagB}: dump missing")
        return out
    for rank in sorted(A):
        a = load(A[rank])
        check_dump_attrib(tagA, rank, a)
        b = load(B[rank]) if rank in B else None
        if b is not None:
            check_dump_attrib(tagB, rank, b)
        fa = A1.attn_blocks(a)
        fb = A1.attn_blocks(b) if b is not None else None
        lo, hi = a["range"]
        for nm, La in A1.idx_layers(a).items():
            pools = range((lo + 3) // 4, (hi - 1) // 4)
            expected += len(pools)
            Lb = b["layers"].get(nm) if b is not None else None
            if Lb is None:
                continue
            for pool in pools:
                x, y = idx_pages_pool(La, fa, pool), idx_pages_pool(Lb, fb, pool)
                if x is None or y is None:
                    continue
                out[(rank, layer_no(nm), pool)] = float((x - y).norm() / max(float(y.norm()), 1e-12))
    if name:
        e = POP.setdefault(name, [0, 0, []])
        e[0] += len(out)
        e[1] += expected
    return out


def gross(vals):
    vals = list(vals)
    return float(np.mean([v > GROSS for v in vals])) if vals else float("nan")


def page_stats(rows_by_tag):
    pg = collections.defaultdict(list)
    for tag, rows in rows_by_tag.items():
        for (rank, layer, pool), rel in rows.items():
            pg[(tag, rank, layer, pool // 64)].append(rel > GROSS)
    judged = {k: v for k, v in pg.items() if len(v) >= PAGE_MINP}
    worst = sorted(((float(np.mean(v)), len(v), k) for k, v in judged.items()), reverse=True)[:3]
    return dict(pages_judged=len(judged), page_max=worst[0][0] if worst else None,
                over=sum(1 for v in judged.values() if np.mean(v) > PAGE_MAX_GROSS),
                worst=[(round(f, 4), n, list(map(str, k))) for f, n, k in worst])


# ------------------------------------------------------------------ H4 (KDA, V3 index-K)
def v3_pairs():
    pairs = []
    for (tag, F, win), ranks in SNAP.items():
        if win or not tag.startswith("V3-"):
            continue
        for (rtag, rF, rwin), rr in SNAP.items():
            if rtag == f"V3ref-F{F}-{tag}" and not rwin:
                pairs.append((tag, F, ranks, rr))
    return sorted(pairs, key=lambda x: (x[0], x[1]))


def prescribed_acc(mode, remaining):
    """Accepted drafts the forcing schedule prescribes (None: model-dependent or pseudo-random)."""
    a = {"oracle": A1.K, "poison": 0}.get(mode)
    if mode.startswith("reject_") or mode.startswith("reject:"):
        a = int(mode[7:]) - 1
    return None if a is None else min(a, max(remaining - 1, 0))


def r_h4():
    same_as_original("H4a")
    pairs = v3_pairs()
    kda, idx, slot_bad, sched, meta = [], {}, [], collections.Counter(), collections.Counter()
    for tag, F, ranks, rr in pairs:
        creq = [r for r in reqs if r["tag"].replace(":", "_") == tag]
        a_req = REQ.get(tag)
        for rank, p in ranks.items():
            if rank not in rr:
                continue
            a, b = load(p), load(rr[rank])
            na = a.get("num_accepted")
            # ---- 21g-2 (a): committed-slot bookkeeping and expected slot, exact
            if a.get("tag", "").replace(":", "_") != tag or a.get("F") != F or not creq or a.get("P") != creq[0]["P"]:
                slot_bad.append((tag, F, rank, "attribution", a.get("tag"), a.get("P"), a.get("F")))
                ATTRIB.append((tag, rank, "V3 snapshot tag/P/F", (a.get("tag"), a.get("P"), a.get("F")),
                               (tag, creq[0]["P"] if creq else None, F)))
            kl = [L for L in a["layers"].values() if L.get("kind") == "kda"]
            nsl = {int(L["slots"].numel()) for L in kl if "slots" in L}
            if not kl or any(not L.get("is_spec") for L in kl) or len(nsl) != 1:
                slot_bad.append((tag, F, rank, "kda layers not all speculative with one slot count", sorted(nsl)))
            elif not (isinstance(na, int) and 1 <= na <= next(iter(nsl))):
                slot_bad.append((tag, F, rank, "num_accepted out of range", na, sorted(nsl)))
            rq = None
            if a_req:
                rq = a_req.get(rank) if rank in a_req else next(iter(a_req.values()))
            if rq is not None and isinstance(na, int):
                Cc = cols(rq)
                M = rq["metrics"].numpy()
                rows = np.where(M[:, Cc["step"]] == a.get("step"))[0]
                accs = sorted(set(M[rows, Cc["accepted"]].tolist()))
                if len(accs) != 1 or int(accs[0]) + 1 != na:
                    slot_bad.append((tag, F, rank, "num_accepted != hook accepted at the step + 1", na, accs))
                elif rows.size:
                    mode = tag.split("-")[1]
                    pa = prescribed_acc(mode, int(creq[0]["W"]) - int(rows.min()) if creq else 99)
                    if pa is not None:
                        sched["match" if pa == int(accs[0]) else "differs"] += 1
                steps = sorted(set(M[:, Cc["step"]][np.isfinite(M[:, Cc["step"]])].astype(int).tolist()))
                prev = max([s for s in steps if s < a.get("step", -1)], default=None)
                if prev is not None:
                    pa_ = sorted(set(M[M[:, Cc["step"]] == prev, Cc["accepted"]].tolist()))
                    lna = {L.get("num_accepted") for L in kl}
                    meta["consistent" if len(pa_) == 1 and lna == {int(pa_[0]) + 1} else "differs"] += 1
            elif rq is None:
                slot_bad.append((tag, F, rank, "no hook request record"))
            # ---- 21g-2 (b): nearest-reference identification (diagnostic screen)
            for name, La in a["layers"].items():
                Lb = b["layers"].get(name)
                if La.get("kind") == "kda" and Lb is not None and "ssm_sketch" in La and "ssm_sketch" in Lb \
                        and na and La.get("is_spec"):
                    ref = Lb["ssm_sketch"][0]
                    dist = [float((s - ref).norm() / max(ref.norm(), 1e-12)) for s in La["ssm_sketch"]]
                    c = dist[na - 1]
                    others = [x for k, x in enumerate(dist) if k != na - 1]
                    near = int(np.argmin(dist))
                    kda.append(dict(tag=tag, F=F, rank=rank, layer=layer_no(name), rel=c, committed_slot=na - 1,
                                    nearest_slot=near, ident=near == na - 1, dists=[round(x, 6) for x in dist],
                                    margin=(min(others) / max(c, 1e-12)) if others else float("inf"),
                                    sep=(c / max(min(others), 1e-12)) if others else 0.0))
            fa, fb = A1.attn_blocks(a), A1.attn_blocks(b)
            fr = a["frontier_pos"]
            for name, La in A1.idx_layers(a).items():
                Lb = b["layers"].get(name)
                band = range(max(0, fr // 4 - 8), (fr + 1) // 4)
                e = POP.setdefault("V3", [0, 0, []])
                e[1] += len(band)
                if Lb is None:
                    continue
                for pool in band:
                    x, y = idx_pages_pool(La, fa, pool), idx_pages_pool(Lb, fb, pool)
                    if x is not None and y is not None:     # each (request, rank, layer, pool) once: latest F
                        e[0] += 1
                        idx[(tag, rank, layer_no(name), pool)] = float((x - y).norm() / max(float(y.norm()), 1e-12))
    R["h4_slot"] = dict(snapshots=sum(len(r) for _, _, r, _ in pairs), violations=slot_bad[:20],
                        schedule_prescription=dict(sched), layer_metadata_prev_step=dict(meta))
    row("H4-slot", ["H4a", "H4b"], "committed-slot bookkeeping, exact: num_accepted == hook accepted count at the "
        "snapshot step + 1, 1 <= num_accepted <= slots, every KDA layer speculative with one slot count, snapshot "
        "attributed to its request (tag, P, F)", "PASS" if pairs and not slot_bad else "FAIL",
        json.dumps(R["h4_slot"], default=str)[:600], "invalid", "PA2-16 (21g-2a)")
    row("H4-meta", ["H4b"], "diagnostic (not gated): KDA layer metadata num_accepted vs the previous step's accepted "
        "count + 1; schedule-prescribed acceptance at the snapshot step (oracle / poison / reject:j)", "REPORTED",
        json.dumps(dict(meta=dict(meta), schedule=dict(sched))), "invalid", "PA2-16 (21g-2a)")
    if kda:
        ide = float(np.mean([k["ident"] for k in kda]))
        early = [k for k in kda if k["layer"] in KDA_EARLY_LAYERS]
        erel = max([k["rel"] for k in early], default=float("nan"))
        emar = min([k["margin"] for k in early], default=float("nan"))
        shallow = [k for k in kda if k["layer"] <= KDA_SHALLOW_MAX]
        deep = [k for k in kda if k["layer"] > KDA_SHALLOW_MAX]
        i_sh = float(np.mean([k["ident"] for k in shallow])) if shallow else None
        i_dp = float(np.mean([k["ident"] for k in deep])) if deep else None
        dis = [k for k in kda if not k["ident"]]
        sepf = [k for k in dis if k["sep"] >= KDA_SEP_FLAG]
        per = collections.defaultdict(list)
        for k in kda:
            per[k["layer"]].append(k)
        R["h4_kda"] = dict(n=len(kda), identified=ide, ident_0_17=i_sh, ident_18_plus=i_dp, n_0_17=len(shallow),
                           n_18_plus=len(deep), early_n=len(early), early_rel_max=erel, early_margin_min=emar,
                           deep_margin_lt_1_25=sum(1 for k in deep if k["ident"] and k["margin"] < 1.25),
                           disagreements=dis, sep_flagged=sepf,
                           per_layer={L: dict(n=len(v), ident=float(np.mean([k["ident"] for k in v])),
                                              rel_q50=q([k["rel"] for k in v], 0.5), rel_max=max(k["rel"] for k in v),
                                              margin_min=min(k["margin"] for k in v)) for L, v in sorted(per.items())})
        ok = (i_sh == 1.0 and (i_dp is None or i_dp >= KDA_DEEP_IDENT_MIN) and bool(early)
              and erel <= KDA_EARLY_REL and emar >= KDA_EARLY_MARGIN)
        row("H4b-r", ["H4b"], "KDA nearest-reference identification (diagnostic screen): 100 % in layers 0-17, >= 99 % "
            "in layers >= 18; layers 0-2 ONLY: rel <= 0.02 with margin >= 10", "PASS" if ok else "FAIL",
            f"n {len(kda)}; layers 0-17 {i_sh} (n {len(shallow)}); layers 18+ {i_dp} (n {len(deep)}); layers 0-2 n "
            f"{len(early)} rel max {erel:.4g} margin min {emar:.3g}; disagreements {len(dis)}", "invalid",
            "PA2-16 (21g-2b)")
        row("H4b-disagree", ["H4b"], "every identification disagreement with its distances and margin (deep-layer "
            "disagreements are unresolved numerical comparisons)", "REPORTED",
            json.dumps([dict(tag=k["tag"], F=k["F"], rank=k["rank"], layer=k["layer"], committed=k["committed_slot"],
                             nearest=k["nearest_slot"], dists=k["dists"], margin=round(k["margin"], 4),
                             sep=round(k["sep"], 4)) for k in dis])[:1500], "invalid", "PA2-16 (21g-2b)")
        row("H4b-sep", ["H4b"], "no comfortably separated wrong-slot match: a wrong slot nearer than the committed "
            "slot by >= 1.25x requires investigation", "PASS" if not sepf else "FAIL",
            f"{len(sepf)} flagged: " + json.dumps([dict(tag=k["tag"], F=k["F"], rank=k["rank"], layer=k["layer"],
                                                        sep=round(k["sep"], 3), dists=k["dists"]) for k in sepf[:5]]),
            "inconclusive", "PA2-16 (21g-2b)")
        row("H4b-draft100", ["H4b"], "pre-21g DRAFT form kept beside: identification 100 % in every layer, same layers "
            "0-2 tolerance", "REPORTED", "would %s: identified %.4f" % (
                "PASS" if ide == 1.0 and bool(early) and erel <= KDA_EARLY_REL and emar >= KDA_EARLY_MARGIN else "FAIL",
                ide), "invalid", "pre-21g DRAFT")
    else:
        row("H4b-r", ["H4b"], "KDA compared", "FAIL", "no KDA comparisons", "inconclusive", "PA2-16 (21g-2b)")
    R["v3_idx_rows"] = idx
    # byte stability over time: the same decode request at two snapshot times
    by = collections.defaultdict(list)
    for (tag, F, win), ranks in SNAP.items():
        if tag.startswith("V3-") and not win:
            by[tag].append((F, ranks))
    n_pairs, n_same = 0, 0
    for tag, lst in by.items():
        lst.sort(key=lambda x: x[0])
        for (F1, r1), (F2, r2) in zip(lst, lst[1:]):
            for rank in r1:
                if rank not in r2:
                    continue
                a, b = load(r1[rank]), load(r2[rank])
                fa, fb = A1.attn_blocks(a), A1.attn_blocks(b)
                fr = a["frontier_pos"]
                for name, La in A1.idx_layers(a).items():
                    Lb = b["layers"].get(name)
                    if Lb is None:
                        continue
                    for pool in range(max(0, fr // 4 - 8), (fr + 1) // 4):
                        x, y = idx_pages_pool(La, fa, pool), idx_pages_pool(Lb, fb, pool)
                        if x is not None and y is not None:
                            n_pairs += 1
                            n_same += bool(torch.equal(x, y))
    R["idx_stable"] = dict(pairs=n_pairs, byte_identical=(n_same / n_pairs) if n_pairs else None)


# ------------------------------------------------------------------ H5 (V4 / V4n), 21g-1
C = {}


def matched(arm, tids=None, keep=None):
    """Tested units of `arm` that exist in BOTH nulls of their fixture: [(tid, key, rel, rel_dd, rel_pp)], tested,
    fixtures with tested units."""
    units, tested, fx = [], 0, set()
    for tid, rows in C.get(arm, {}).items():
        if tids is not None and tid not in tids:
            continue
        dd, pp = C["N_dd"].get(tid, {}), C["N_pp"].get(tid, {})
        for k, rel in rows.items():
            if keep is not None and not keep(tid, k):
                continue
            tested += 1
            fx.add(tid)
            if k in dd and k in pp:
                units.append((tid, k, rel, dd[k], pp[k]))
    return units, tested, fx


def limit_of(units):
    ndd, npp = gross(u[3] for u in units), gross(u[4] for u in units)
    lim = max(CONTENT_FACTOR * max(ndd, npp), CONTENT_FLOOR) if units else float("nan")
    return lim, ndd, npp


def content(arm, tids=None, keep=None):
    units, tested, fx = matched(arm, tids, keep)
    lim, ndd, npp = limit_of(units)
    g = gross(u[2] for u in units)
    fx = sorted(fx)
    fx_missing = [t for t in fx if not any(u[0] == t for u in units)]
    cov = len(units) / tested if tested else 0.0
    return dict(arm=arm, tested=tested, evaluated=len(units), coverage=cov, fixtures=fx, fixtures_without_null=fx_missing,
                gross=g, gross_n=int(sum(u[2] > GROSS for u in units)), N_dd=ndd, N_pp=npp, limit=lim,
                cover_ok=bool(units) and cov >= NULL_COVER_MIN and not fx_missing, _units=units)


def pooled_null():
    """PA2-26 (21h): the matched nulls pooled over the eight-fixture set. Units: every (fixture, rank, layer, pool)
    of the fixtures' windows (PA2-7e exclusions applied by compare_windows) present in BOTH N_dd and N_pp of its
    fixture; rates pooled over units; the fixture set is the B1 arm's (V4-b1-*)."""
    fx = sorted(C.get("b1", {}) or {r["tag"].rsplit("-", 1)[-1] for r in reqs if r["tag"].startswith("V4-b1-")})
    units = []
    for t in fx:
        dd, pp = C["N_dd"].get(t, {}), C["N_pp"].get(t, {})
        units += [(t, k, None, dd[k], pp[k]) for k in dd if k in pp]
    lim, ndd, npp = limit_of(units)
    have = {u[0] for u in units}
    return dict(n=len(units), N_dd=ndd, N_pp=npp, N_dd_n=int(sum(u[3] > GROSS for u in units)),
                N_pp_n=int(sum(u[4] > GROSS for u in units)), limit=lim, fixtures=fx,
                fixtures_without_null=[t for t in fx if t not in have], cover_ok=bool(units) and have == set(fx),
                _units=units)


def table(units):
    """per fixture x layer and per fixture: n, gross counts / rates of the test and both matched nulls; pooled."""
    by_fl, by_f = collections.defaultdict(list), collections.defaultdict(list)
    for u in units:
        by_fl[(u[0], u[1][1])].append(u)
        by_f[u[0]].append(u)

    def summ(us):
        n = len(us)
        g = [int(sum(x[i] > GROSS for x in us)) for i in (2, 3, 4)]
        return dict(n=n, gross_n=g[0], gross=g[0] / n if n else None, N_dd_n=g[1], N_dd=g[1] / n if n else None,
                    N_pp_n=g[2], N_pp=g[2] / n if n else None)
    return dict(pooled=summ(units), per_fixture={t: summ(v) for t, v in sorted(by_f.items())},
                per_fixture_layer={f"{t}|L{L}": summ(v) for (t, L), v in sorted(by_fl.items())})


def flags(units):
    out = []
    by_f = collections.defaultdict(list)
    for u in units:
        by_f[u[0]].append(u)
    for t, us in sorted(by_f.items()):
        lim_f, ndd, npp = limit_of(us)
        gn = int(sum(u[2] > GROSS for u in us))
        g = gn / len(us)
        lim_flag = max(FLAG_FACTOR * max(ndd, npp), CONTENT_FLOOR)
        if gn >= FLAG_MIN_GROSS and g > lim_flag:
            out.append(dict(fixture=t, n=len(us), gross_n=gn, gross=g, N_dd=ndd, N_pp=npp, flag_limit=lim_flag,
                            mechanism_claim=False))
    return out


def asym_flags(res):
    """Arm asymmetry (pre-data definition): on the units both B1 arms and N_pp share, a fixture is flagged when its
    grosser arm has >= 10 gross units, > 2 x the other arm's rate and > max(2 x N_pp_f, 0.1 %). N_dd is b1 vs b1b, so
    a defect confined to one decode arm inflates N_dd and the content limit absorbs it; this flag surfaces it."""
    out = []
    ua = collections.defaultdict(dict)
    for arm in ("b1", "b1b"):
        for t, k, rel, dd, pp in res[arm]["_units"]:
            ua[t].setdefault(k, {})[arm] = (rel, pp)
    for t, d in sorted(ua.items()):
        ks = [k for k, v in d.items() if len(v) == 2]
        if not ks:
            continue
        na = sum(d[k]["b1"][0] > GROSS for k in ks)
        nb = sum(d[k]["b1b"][0] > GROSS for k in ks)
        npp = float(np.mean([d[k]["b1"][1] > GROSS for k in ks]))
        hi_n, lo_n = max(na, nb), min(na, nb)
        lim = max(FLAG_FACTOR * npp, CONTENT_FLOOR)
        if hi_n >= FLAG_MIN_GROSS and hi_n > FLAG_FACTOR * lo_n and hi_n / len(ks) > lim:
            out.append(dict(fixture=t, n=len(ks), b1_gross_n=int(na), b1b_gross_n=int(nb), N_pp=npp, limit=lim,
                            arm="b1" if na > nb else "b1b", mechanism_claim=False))
    return out


def r_h5():
    same_as_original("H5a")
    tids = sorted({r["tag"].rsplit("-", 1)[-1] for r in reqs if r["tag"].startswith("V4-b1-")})
    for k in ("b1", "b1b", "d2", "d2solo", "N_dd", "N_pp"):
        C[k] = {}
    for tid in tids:
        C["b1"][tid] = compare_windows(f"V4-b1-{tid}", f"V4-rebuild-{tid}", "b1")
        C["b1b"][tid] = compare_windows(f"V4n-b1b-{tid}", f"V4-rebuild-{tid}", "b1b")
        C["d2"][tid] = compare_windows(f"V4-d2-{tid}", f"V4-rebuild-{tid}", "d2")
        C["N_dd"][tid] = compare_windows(f"V4-b1-{tid}", f"V4n-b1b-{tid}", "N_dd")
        C["N_pp"][tid] = compare_windows(f"V4-rebuild-{tid}", f"V4n-rebuild2-{tid}", "N_pp")
    for r in reqs:
        if r["tag"].startswith("V4-d2solo-"):
            tid = r["tag"].rsplit("-", 1)[-1]
            C["d2solo"][tid] = compare_windows(r["tag"], f"V4-rebuild-{tid}", "d2solo")
    for k in ("b1", "b1b", "d2", "d2solo", "N_dd", "N_pp"):
        C[k] = {t: v for t, v in C[k].items() if v}
    flat = {k: [x for v in d.values() for x in v.values()] for k, d in C.items()}
    stats = {k: dict(n=len(v), gross=gross(v), q50=q(v, 0.5), q999=q(v, 0.999),
                     per_trace={t: gross(C[k][t].values()) for t in C[k]}) for k, v in flat.items()}
    R["h5_content_unmatched"] = stats             # every comparison over all its units (pre-21g draft basis)
    # ---- structural addressing (exact)
    v3 = collections.defaultdict(dict)
    for (tag, rank, layer, pool), rel in R.get("v3_idx_rows", {}).items():
        v3[tag][(rank, layer, pool)] = rel
    sets = {"b1": C["b1"], "b1b": C["b1b"], "d2solo": C["d2solo"], "V3": dict(v3)}
    pst = {k: page_stats(v) for k, v in sets.items()}
    early = {k: sum(1 for rows in v.values() for (rk, L, p), rel in rows.items() if 3 <= L <= EARLY_IDX_LAYERS
                    and rel > GROSS) for k, v in sets.items()}
    early_n = {k: sum(1 for rows in v.values() for (rk, L, p) in rows if 3 <= L <= EARLY_IDX_LAYERS)
               for k, v in sets.items()}
    R["idx_structure"] = dict(pages=pst, early_gross=early, early_n=early_n)
    absent = [k for k in ("b1", "b1b", "V3") if not sets[k]]
    pop = {k: dict(compared=v[0], expected=v[1], frac=(v[0] / v[1]) if v[1] else None, notes=v[2][:3])
           for k, v in POP.items()}
    pop_bad = [k for k, v in pop.items() if not v["frac"] or v["frac"] < POP_MIN or v["notes"]]
    R["idx_attrib"] = dict(violations=ATTRIB[:20], n_violations=len(ATTRIB), population=pop)
    avail = "; on the sets present: " if absent else ""
    row("IDX-attrib", ["H4c", "H5b"], "exact attribution: every window dump / V3 snapshot carries its request's tag, P "
        "and window range; every comparison compares >= 99 % of its expected units", "PASS" if not ATTRIB
        and not pop_bad and not absent else "FAIL", f"attribution violations {len(ATTRIB)} {ATTRIB[:3]}; population "
        + json.dumps({k: (v["compared"], v["expected"]) for k, v in pop.items()}) + f"; failing {pop_bad}; absent {absent}",
        "invalid", "PA2-16 (21g-1 structural)")
    row("IDX-page", ["H4c", "H5b"], "structural addressing: no page with > 25 % gross pools (unique pools; pages with "
        ">= 16 compared pools; B1 arms, solo, V3)", "PASS" if not absent and all(p["over"] == 0 for p in pst.values())
        else "FAIL", (f"absent sets {absent}{avail}" if absent else "") +
        json.dumps({k: (v["pages_judged"], v["page_max"], v["over"]) for k, v in pst.items()}), "invalid",
        "DRAFT; PA2-16 (21g-1 structural)")
    row("IDX-early", ["H4c", "H5b"], "structural addressing: zero gross pools in DSA layers 3-15 (B1 arms, solo, V3)",
        "PASS" if not absent and sum(early.values()) == 0 and sum(early_n.values()) > 0 else "FAIL",
        (f"absent sets {absent}{avail}" if absent else "") + f"gross {early} of {early_n}", "invalid",
        "DRAFT; PA2-16 (21g-1 structural)")
    st = R.get("idx_stable", {})
    row("IDX-stable", ["H4c"], "committed pools byte-stable over time (V3: same request, consecutive snapshots)",
        "PASS" if st.get("pairs") and st.get("byte_identical") == 1.0 else "FAIL", json.dumps(st), "invalid",
        "DRAFT; PA2-16 (21g-1 structural)")
    # ---- content vs nulls matched to the tested units (21g-1)
    res = {"b1": content("b1"), "b1b": content("b1b"), "d2solo": content("d2solo"), "d2": content("d2")}
    tabs = {k: table(v["_units"]) for k, v in res.items()}
    tabs["N_dd_all"] = {t: dict(n=len(v), gross_n=int(sum(x > GROSS for x in v.values())), gross=gross(v.values()))
                        for t, v in C["N_dd"].items()}
    tabs["N_pp_all"] = {t: dict(n=len(v), gross_n=int(sum(x > GROSS for x in v.values())), gross=gross(v.values()))
                        for t, v in C["N_pp"].items()}
    per_layer = collections.defaultdict(lambda: collections.Counter())
    for k in ("b1", "b1b", "d2solo", "d2"):
        for u in res[k]["_units"]:
            per_layer[f"{k}|L{u[1][1]}"]["n"] += 1
            per_layer[f"{k}|L{u[1][1]}"]["gross_n"] += int(u[2] > GROSS)
    tabs["pooled_per_layer"] = {k: dict(v) for k, v in sorted(per_layer.items())}
    fl = {k: flags(res[k]["_units"]) for k in ("b1", "b1b", "d2solo")}
    fl["arm_asymmetry"] = asym_flags(res)
    R["idx_content"] = {k: {kk: vv for kk, vv in v.items() if kk != "_units"} for k, v in res.items()}
    R["idx_content_table"] = tabs
    R["fixture_flags"] = fl
    pre = dict(b1=stats["b1"]["gross"], b1b=stats["b1b"]["gross"], N_dd=stats["N_dd"]["gross"],
               N_pp=stats["N_pp"]["gross"])
    R["content_draft_unmatched"] = pre

    def fmt(x):
        return f"{x['gross']:.4%} ({x['gross_n']}/{x['evaluated']}) vs limit {x['limit']:.4%} [N_dd {x['N_dd']:.4%}, " \
               f"N_pp {x['N_pp']:.4%}; coverage {x['coverage']:.4f}, fixtures without null {x['fixtures_without_null']}]" \
            if x["evaluated"] else f"no matched units (tested {x['tested']})"
    b1, b1b = res["b1"], res["b1b"]
    if not (b1["cover_ok"] and b1b["cover_ok"]):
        row("IDX-content", ["H5b"], "matched in-session nulls (V4n) cover the tested B1 units (>= 99 %, every fixture)",
            "FAIL", f"b1: {fmt(b1)}; b1b: {fmt(b1b)}", "inconclusive", "PA2-16 (21g-1)")
    else:
        row("IDX-content", ["H5b"], "B1 arms vs rebuild: gross <= max(2 x max(N_dd, N_pp), 0.1 %), nulls restricted to "
            "the same fixtures, ranks, layers and pools as each tested arm; pooled over units",
            "PASS" if b1["gross"] <= b1["limit"] and b1b["gross"] <= b1b["limit"] else "FAIL",
            f"b1: {fmt(b1)}; b1b: {fmt(b1b)}; pre-21g unmatched draft form {json.dumps(pre)}", "invalid",
            "PA2-16 (21g-1)")
    # PA2-26 (amendment 21h): the GATE uses the pooled matched null of the eight-fixture set; the comparison with the
    # solo fixture's own matched null is a FLAG; the pre-21g pooled form and the strict per-fixture form stay reported.
    so = res["d2solo"]
    pn = pooled_null()
    R["h5e"] = dict(solo=fmt(so), pooled_null={k: v for k, v in pn.items() if k != "_units"})
    so_ok = so["cover_ok"] and pn["cover_ok"]
    row("H5e-r", ["H5e"], "injected request alone vs rebuild (its matched units): gross <= max(2 x max(N_dd, N_pp), "
        "0.1 %) with N_dd, N_pp the POOLED matched nulls of the eight-fixture set (same unit definition, pooled "
        "denominators and exclusions as IDX-content)", ("PASS" if so["gross"] <= pn["limit"] else "FAIL") if so_ok
        else "FAIL", (f"solo {so['gross']:.4%} ({so['gross_n']}/{so['evaluated']}) vs pooled limit {pn['limit']:.4%} "
                      f"[N_dd {pn['N_dd']:.4%} ({pn['N_dd_n']}/{pn['n']}), N_pp {pn['N_pp']:.4%} ({pn['N_pp_n']}/"
                      f"{pn['n']}); fixtures {len(pn['fixtures'])}, without null {pn['fixtures_without_null']}; solo "
                      f"coverage {so['coverage']:.4f}]") if so_ok else f"solo: {fmt(so)}; pooled null: "
        f"{ {k: v for k, v in pn.items() if k != '_units'} }", "invalid" if so_ok else "inconclusive",
        "PA2-26 (21h)")
    fx_flag = so["evaluated"] > 0 and so["gross"] > so["limit"]
    row("H5e-fixture", ["H5e"], "FLAG (amendment 21h; strict per-fixture form, not a failure): the solo request against "
        "its own fixture's matched null (same fixture and units)", "REPORTED",
        ("FLAGGED: " if fx_flag else "not flagged: " if so["evaluated"] else "not computable: ") + fmt(so), "invalid",
        "PA2-26 (21h)")
    R["fixture_flags_h5e_own_null"] = dict(flagged=fx_flag, solo=fmt(so))
    row("H5e-pooled", ["H5e"], "pre-21g DRAFT form kept beside: the solo request against the B1 arms' pooled limit "
        "(8 fixtures; not matched to the solo fixture)", "REPORTED",
        f"solo {so['gross']:.4%} vs b1 pooled matched limit {b1['limit']:.4%}" if so["evaluated"] and b1["evaluated"]
        else "not computable (no matched units)", "invalid", "pre-21g DRAFT")
    d2 = res["d2"]
    gd = stats["d2"]["gross"]
    row("H5d-r", ["H5d"], "injected arm at 8 in flight: gross > 30 % over all its units (also reported on its matched "
        "units as a multiple of their limit)", "PASS" if flat["d2"] and gd > D2_GROSS_MIN else "FAIL",
        f"all units {gd:.4%} (n {stats['d2']['n']}); matched: " + fmt(d2) +
        (f" = {d2['gross'] / d2['limit']:.1f} x limit" if d2["evaluated"] else ""), "invalid", "PA2-16 (21g-1)")
    nfl = {k: [f["fixture"] for f in v] for k, v in fl.items()}
    row("IDX-fixture-flags", ["H5b", "H5e"], "fixtures substantially outside their own matched null (gross_f > "
        "max(2 x max(N_dd_f, N_pp_f), 0.1 %) and >= 10 gross units), or with one B1 arm > 2x the other (arm "
        "asymmetry): flagged, cannot carry a mechanism claim",
        "REPORTED", json.dumps(dict(flagged=nfl, detail=fl))[:900], "invalid", "PA2-16 (21g-1)")
    v3g = gross(R.get("v3_idx_rows", {}).values())
    row("H4c-content", ["H4c"], "V3 decode vs prefill gross: descriptive only (V3 traces and positions have NO matched "
        "null in scope; no content reading)", "REPORTED", f"V3 {v3g:.4%} over {len(R.get('v3_idx_rows', {}))} units",
        "invalid", "PA2-16 (21g-1)")
    row("H5c-r", ["H5c"], "0.25 threshold discrimination (retired as a gate; reported)", "REPORTED",
        json.dumps({k: (round(v["q999"], 3) if v["q999"] == v["q999"] else None) for k, v in stats.items()}),
        "inconclusive", "DRAFT")
    r_h5f()
    r_h5g()
    r_h5h()


def r_h5f():
    """H5f-B1: B1 arms' ledger all clean (exact). H5f-D2: D2 clean-labelled pools no grosser than their matched
    limit; corrupt -> gross reported."""
    b1_bad, b1_n = [], 0
    for r in reqs:
        if not (r["tag"].startswith("V4-b1-") or r["tag"].startswith("V4n-b1b-")):
            continue
        for rank, a in (REQ.get(r["tag"].replace(":", "_")) or {}).items():
            pc = a["pool_class"].numpy()
            P, W = r["P"], r["W"]
            for i in range((P + 3) // 4, (P + W - 1) // 4):          # pools fully inside, processed
                k = i - P // 4
                b1_n += 1
                if not (0 <= k < pc.size) or pc[k] != 1:
                    b1_bad.append((r["tag"], rank, i, int(pc[k]) if 0 <= k < pc.size else None))
    row("H5f-B1", ["H5f"], "shadow ledger, B1 arms (no injection): every processed window pool classified clean "
        "(positive control for the PA2-2 fix)", "PASS" if b1_n and not b1_bad else "FAIL",
        f"{b1_n} pool records; not clean {len(b1_bad)}: {b1_bad[:5]}", "invalid", "DRAFT (PA2-2)")
    PC = {}
    for r in reqs:
        if r["tag"].startswith("V4-d2"):
            a = r0(r["tag"])
            if a is not None:
                PC[r["tag"]] = (a["pool_class"].numpy(), r["P"])

    def lab(arm, want):
        def f(tid, k):
            pc, P = PC.get(f"V4-{arm}-{tid}", (None, 0))
            if pc is None:
                return False
            j = k[2] - P // 4
            return 0 <= j < pc.size and pc[j] == want
        return f
    per = {}
    for arm in ("d2", "d2solo"):
        cl, cx = content(arm, keep=lab(arm, 1)), content(arm, keep=lab(arm, 2))
        per[arm] = dict(clean_n=cl["evaluated"], clean_gross=cl["gross"] if cl["evaluated"] else None,
                        clean_limit=cl["limit"], clean_cover_ok=cl["cover_ok"], corrupt_n=cx["evaluated"],
                        corrupt_gross=cx["gross"] if cx["evaluated"] else None)
    both = per["d2"]["clean_n"] + per["d2solo"]["clean_n"]
    comb = (1 - ((per["d2"]["clean_gross"] or 0) * per["d2"]["clean_n"] + (per["d2solo"]["clean_gross"] or 0)
                 * per["d2solo"]["clean_n"]) / both) if both else None
    R["h5f"] = dict(per_arm=per, combined_clean_nongross=comb)
    g, lim = per["d2"]["clean_gross"], per["d2"]["clean_limit"]
    row("H5f-D2", ["H5f"], "shadow ledger, D2 arm at 8 in flight: clean-labelled pools no grosser than their matched "
        "limit (nulls on the same units); corrupt -> gross reported, not gated",
        "PASS" if per["d2"]["clean_n"] and per["d2"]["clean_cover_ok"] and g is not None and g <= lim else "FAIL",
        json.dumps(R["h5f"]), "invalid", "DRAFT; PA2-16 (21g-1)")
    row("H5f-orig-form", ["H5f"], "original form, injected arms pooled: clean -> non-gross >= 99 % (reported)",
        "REPORTED", f"combined clean non-gross {comb}", "invalid", "DRAFT")


def r_h5g():
    """PA2-7c: per rank, counter deltas from the client's start-dump poll to the last poll; per-phase attribution."""
    if not POLLS:
        row("H5g-r", ["H5g"], "guard polls present", "FAIL", "no polls", "inconclusive", "PA2-7c")
        return
    dumps = [r for r in CL if r.get("kind") == "dump"]
    start = [d for d in dumps if d.get("tag") == "start"]
    if not start:
        row("H5g-r", ["H5g"], "client start dump present", "FAIL", "none", "inconclusive", "PA2-7c")
        return
    t_start = start[0]["t"]
    SITES = (("tail_w_stash", "mism"), ("tail_w_stash", "null"), ("idx_w_dec", "mism"), ("tail_w_seed", "mism"),
             ("idx_w_pf", "s_mism"), ("idx_w_pf", "d7_fallback"), ("pos", "pos_contig"), ("pos", "pos_bounds"))
    val = lambda p, s, c: p["counters"].get(s, {}).get(c, 0)  # noqa: E731
    inj = sum((r0(r["tag"]) or {}).get("d2_injected_tokens") or 0 for r in reqs
              if r["tag"].startswith(("V4-d2-", "V4-d2solo-")))
    res, ok = {}, inj > 0
    dts = sorted((d["t"], d["tag"]) for d in dumps)

    def label(t):                       # the latest client dump at or before a poll (time-based attribution)
        lab = [tag for td, tag in dts if td <= t]
        return lab[-1] if lab else "before-start"
    for rank in sorted({p.get("rank") for p in POLLS}):
        pr = sorted([p for p in POLLS if p.get("rank") == rank], key=lambda p: p["t"])
        base = [p for p in pr if p["t"] >= t_start and p.get("reason") == "trigger"]
        if not base:
            ok = False
            res[rank] = "no start-dump poll"
            continue
        b0, last = base[0], pr[-1]
        d = {f"{s}.{c}": val(last, s, c) - val(b0, s, c) for s, c in SITES}
        nl = len(last.get("layers") or {})
        trig = [p for p in pr if p.get("reason") == "trigger" and p["t"] >= t_start]
        per_phase = {}
        for p1, p2 in zip(trig, trig[1:]):
            dv = val(p2, "tail_w_stash", "mism") - val(p1, "tail_w_stash", "mism")
            if dv:
                lab = label(p2["t"])
                per_phase[lab] = per_phase.get(lab, 0) + dv
        dv = val(last, "tail_w_stash", "mism") - val(trig[-1], "tail_w_stash", "mism")
        if dv:
            per_phase["after-last-trigger"] = dv
        others = {k: v for k, v in d.items() if not k.startswith("tail_w_stash") and v}
        rok = (d["tail_w_stash.mism"] == d["tail_w_stash.null"] == nl * inj and d["tail_w_stash.mism"] > 0
               and not others and all(t.startswith(("V4-d2", "V4-d2solo")) for t in per_phase))
        ok &= rok
        res[rank] = dict(owned_layers=nl, deltas=d, expected_tail=nl * inj, tail_by_phase=per_phase,
                         pre_client_pos_bounds=val(b0, "pos", "pos_bounds"), ok=rok)
    R["h5g"] = dict(injected_tokens=inj, per_rank=res)
    row("H5g-r", ["H5g"], "guard per rank, deltas from the client's start dump: tail_w_stash mism == null == owned "
        "layers x injected tokens, only in V4-d2/V4-d2solo; zero idx_w_dec, tail_w_seed, D7 and position deltas",
        "PASS" if ok else "FAIL", json.dumps(R["h5g"], default=str)[:600], "invalid", "PA2-7c (PI-3)")


def r_h5h():
    """PA2-7d: designated positions o >= 1 of forced requests with W > 1 have a step-matched exposure record."""
    tot, miss, mis, allL, ntot_sel, sel_bad = 0, 0, 0, 0, 0, 0
    examples = []
    for r in forced_reqs():
        if r["W"] <= 1:
            continue
        a = r0(r["tag"])
        if a is None or not a.get("desig"):
            continue
        Cc = cols(a)
        M = a["metrics"].numpy()
        x = a["exposure"].numpy()
        nslots = len(a.get("layer_slots") or {})
        for k, o in enumerate(a["desig"]):
            if o < 1 or o >= r["W"]:
                continue
            tot += 1
            ms = M[o, Cc["step"]]
            steps = x[k, :, 0]
            present = steps >= 0
            if not present.any():
                miss += 1
                if len(examples) < 5:
                    examples.append((r["tag"], o, "missing"))
                continue
            if not np.isfinite(ms) or not (steps[present] == int(ms)).all():
                mis += 1
                if len(examples) < 5:
                    examples.append((r["tag"], o, "step %s vs %s" % (sorted(set(steps[present].tolist())), ms)))
            if np.isfinite(ms) and nslots and bool((steps[:nslots] == int(ms)).all()):
                allL += 1
        sel = a.get("selections")
        for k, o in enumerate(a.get("selpos") or []):
            if o < 1 or o >= r["W"] or sel is None or k >= sel.shape[0]:
                continue
            ntot_sel += 1
            ss = sel[k, :, 0].numpy()
            pres = ss >= 0
            ms = M[o, Cc["step"]]
            if not pres.any() or not np.isfinite(ms) or not (ss[pres] == int(ms)).all():
                sel_bad += 1
    R["h5h"] = dict(designated=tot, missing=miss, misattributed=mis, all_layers_step_matched=allL,
                    selections=ntot_sel, selections_bad=sel_bad, examples=examples)
    row("H5h-r", ["H5h"], "exposure: every designated position (o >= 1, W > 1) has a record whose step is the "
        "position's measurement step; selection records likewise", "PASS" if tot and not miss and not mis
        and not sel_bad else "FAIL", json.dumps(R["h5h"], default=str)[:500], "invalid", "PA2-7d")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    for fn in (r_coverage, r_h1, r_h2, r_h3, r_h4, r_h5, r_carry_scope, r_carry_admissible):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            row(f"{fn.__name__}-raised", [], f"{fn.__name__} raised", "FAIL", repr(exc)[:300])
    verdict = "INVALID" if R["invalid"] else ("INCONCLUSIVE" if R["inconclusive"] else "VALID")
    R["verdict"] = verdict
    R["original_verdict"] = VO["verdict"]
    R["original_failing"] = VO["invalid"] + VO["inconclusive"]
    R["v3_idx_rows"] = [dict(tag=k[0], rank=k[1], layer=k[2], pool=k[3], rel=v)
                        for k, v in (R.get("v3_idx_rows") or {}).items()]
    json.dump(R, open(os.path.join(OUTD, "verdict-a2.json"), "w"), indent=1, default=str)
    lines = [f"REVISED VERDICT {verdict}  ({R['label']})",
             f"ORIGINAL VERDICT {VO['verdict']}  (frozen attempt-1 criteria; failing {R['original_failing']})", ""]
    for r_ in R["rows"]:
        lines.append("%-8s %-17s orig %-14s %s  [%s] %s" % (r_["status"], r_["id"], ",".join(
            f"{o}:{s}" for o, s in zip(r_["original"], r_["original_result"])), r_["name"], r_["label"],
            r_["detail"][:300]))
    open(os.path.join(OUTD, "verdict-a2.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
