#!/usr/bin/env python3
"""replay-topk analysis (PROTOCOL.md section 7).  Written and tested before any GPU data.  Host python3, stdlib only.

usage: analyze_replay.py RUN_DIR [--out-json F] [--out-md F]
RUN_DIR holds v091/ and v070/ (each: env.json, r1.json, r2.json) from one runner attempt.

Rule: missing or malformed evidence, a failed gate or a failed control gives INCOMPLETE for that unit.  INCOMPLETE is
never read as a negative ("repeatable" / "order-insensitive" need every gate passed).

Units and readings (assessor 22ax section 5 distinctions):
  R1 per image x regime   INCOMPLETE | A-ORDER (same allocation, order varies: execution nondeterminism) |
                          A-SET (same allocation, set varies) | B-ALLOC (each allocation repeatable, relocation changes
                          the output) | REPEATABLE (no variation in N + N calls)
  R2 per image x q regime INCOMPLETE | A-EXEC (same order and allocation differ) | B-ALLOC (relocation / poison
                          changes the output) | ORDER-SENSITIVE (repeatable; a permutation of the same set changes
                          the output) | ORDER-INSENSITIVE (repeatable; no permutation changed any row)
  R3 (v0.9.1)             INCOMPLETE | COMPOSED (outputs differ exactly where R1's orders differ, nowhere else) |
                          NO-TOPK-VARIATION (R1's calls never differed; nothing to compose) | ANOMALY (outputs differ
                          on rows whose top-k did not)
  (c) lifecycle is outside this replay (no cache write follows the replayed operations).
"""
from __future__ import annotations

import json
import os
import sys

MIN_CALLS = 20
GEOM = dict(rows=2045, rows_selecting=2039, rows_shortcut=6, first_selecting_position=2051)
POS_CTRL_FRAC = 0.9
DECODE_REL_ERR_MAX = 0.1
R1_REGIMES = {"v091": ["K", "C", "T"], "v070": ["C", "T"]}
R2_REGIMES = ["flat", "peaked"]
VARIANTS = ["asc", "desc", "perm1", "perm2", "perm3", "atomic2"]


def load(path):
    try:
        with open(path) as fh:
            return json.load(fh), None
    except FileNotFoundError:
        return None, f"missing {os.path.basename(path)}"
    except Exception as exc:  # noqa: BLE001
        return None, f"unreadable {os.path.basename(path)}: {exc!r}"[:200]


def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def r1_unit(doc, regime, version):
    fails = []
    caveats = []
    if doc is None:
        return dict(reading="INCOMPLETE", fails=["r1.json missing"])
    if doc.get("status") not in ("complete", "partial"):
        fails.append(f"r1 status {doc.get('status')}")
    r = g(doc, "regimes", regime)
    if not isinstance(r, dict):
        return dict(reading="INCOMPLETE", fails=fails + [f"regime {regime} absent"])
    if r.get("skipped"):
        return dict(reading="INCOMPLETE", fails=fails + [f"regime {regime} skipped: {r['skipped']}"])
    errs = [e for e in doc.get("errors", []) if e.get("regime") in (regime, None)]
    if errs:
        fails.append(f"{len(errs)} error(s)")
    sa, rl = r.get("same_alloc"), r.get("relocated")
    if not isinstance(sa, dict) or not isinstance(rl, dict):
        return dict(reading="INCOMPLETE", fails=fails + ["same_alloc / relocated record missing"])
    if sa.get("_stopped") or r.get("same_alloc_stopped") or r.get("relocated_stopped"):
        fails.append("stopped by the soft deadline")
    n_sa = int(sa.get("calls", 0)) + (1 if sa.get("includes_reference_call") else 0)
    n_rl = int(rl.get("calls", 0))
    if n_sa < MIN_CALLS:
        fails.append(f"same-allocation calls {n_sa} < {MIN_CALLS}")
    if n_rl < MIN_CALLS:
        fails.append(f"relocated calls {n_rl} < {MIN_CALLS}")
    if not r.get("warmup_calls"):
        fails.append("no warm-up call")
    for k, d in (("same_alloc", sa), ("relocated", rl)):
        if d.get("input_changed_calls", 1) != 0:
            fails.append(f"{k}: input logits changed during a call")
        if d.get("shortcut_row_failures", 1) != 0 or d.get("rows_order_varied_shortcut", 1) != 0:
            fails.append(f"{k}: negative control failed (a row with <= 512 pools not index-ordered or varied)")
    for k, v in GEOM.items():
        if k in ("rows_shortcut", "rows_selecting", "first_selecting_position", "rows") and r.get(k) != v:
            fails.append(f"geometry {k} = {r.get(k)} (expected {v})")
    if r.get("neg_shortcut_rows_index_ordered") != r.get("rows_shortcut"):
        fails.append("reference call: shortcut rows not index-ordered")
    pc = r.get("pos_ctrl") or {}
    if not (pc.get("order_flagged") and pc.get("set_flagged")):
        fails.append("comparator positive control not detected")
    if int(rl.get("calls_relocated_from_ref", 0)) < max(1, n_rl // 2):
        fails.append("relocation not achieved (outputs/inputs at the reference addresses)")
    ns, nt = r.get("refset_sample_rows", 0), r.get("refset_rows_tied_at_boundary", 0)
    if not ns or r.get("refset_agree_untied") != ns - nt:
        fails.append("call semantics: kernel set != fp64 reference set on untied rows (stride/window/args wrong?)")
    if r.get("nonfinite_in_window"):
        caveats.append(f"{r['nonfinite_in_window']} non-finite logits in window")
    w = doc.get("window") or {}
    if not w.get("equals_formula"):
        fails.append("served window != (p + 1) // 4")
    if "formula" in str(w.get("source", "")):
        caveats.append("window from the formula (served builder failed)")
    if "ASSUMED" in str(doc.get("stride_source", "")):
        caveats.append("logits stride assumed (served producer unavailable)")
    sel = max(1, int(r.get("rows_selecting") or 1))
    facts = dict(
        calls_same_alloc=n_sa, calls_relocated=n_rl,
        frac_rows_order_varied_same_alloc=round(sa.get("rows_order_varied_selecting", 0) / sel, 4),
        frac_rows_order_varied_relocated=round(rl.get("rows_order_varied_selecting", 0) / sel, 4),
        rows_set_varied_same_alloc=sa.get("rows_set_varied"), rows_set_varied_relocated=rl.get("rows_set_varied"),
        rows_last_col_varied=max(sa.get("rows_last_col_varied_selecting", 0), rl.get("rows_last_col_varied_selecting", 0)),
        calls_with_order_diff=[sa.get("calls_with_order_diff"), rl.get("calls_with_order_diff")],
        distinct_orderings_max=max(sa.get("distinct_orderings_max", 1), rl.get("distinct_orderings_max", 1)),
        rows_tied_at_threshold=r.get("rows_tied_at_threshold"),
        pattern=dict(last_is_min=r.get("pattern_last_is_min"), fully_desc=r.get("pattern_fully_desc"),
                     index_ascending=r.get("pattern_index_ascending"),
                     sorted_suffix_median=r.get("pattern_sorted_suffix_len_median")),
        stride=r.get("stride"), shape=r.get("shape"))
    if fails:
        return dict(reading="INCOMPLETE", fails=fails, caveats=caveats, facts=facts)
    if sa.get("rows_set_varied", 0) > 0:
        reading = "A-SET"
    elif sa.get("rows_order_varied", 0) > 0:
        reading = "A-ORDER"
    elif rl.get("rows_order_varied", 0) > 0 or rl.get("rows_set_varied", 0) > 0:
        reading = "B-ALLOC"
    else:
        reading = "REPEATABLE"
    return dict(reading=reading, fails=[], caveats=caveats, facts=facts)


def r2_unit(doc, regime, version):
    fails, caveats = [], []
    if doc is None:
        return dict(reading="INCOMPLETE", fails=["r2.json missing"])
    if doc.get("status") not in ("complete", "partial"):
        fails.append(f"r2 status {doc.get('status')}")
    r = g(doc, "regimes", regime)
    if not isinstance(r, dict):
        return dict(reading="INCOMPLETE", fails=fails + [f"q regime {regime} absent"])
    if [e for e in doc.get("errors", []) if e.get("regime") in (regime, None) and e.get("part") is None]:
        fails.append("error(s) in this regime")
    if r.get("stopped"):
        fails.append("stopped by the soft deadline")
    if not r.get("warmup_calls"):
        fails.append("no warm-up call")
    rep = r.get("repeat_same_alloc") or {}
    if int(rep.get("calls", 0)) < 5:
        fails.append(f"same-order repeats {rep.get('calls', 0)} < 5")
    if r.get("nan_rows_base", 1) != 0:
        fails.append("NaN rows in the base output")
    if r.get("rows_selecting") != GEOM["rows_selecting"]:
        fails.append(f"geometry rows_selecting {r.get('rows_selecting')}")
    var = r.get("variants") or {}
    missing = [v for v in VARIANTS if v not in var]
    if missing:
        fails.append(f"variants missing: {missing}")
    for name, c in var.items():
        if not (c.get("same_set") and c.get("same_nsa")):
            fails.append(f"variant {name}: index set or length differs from base (construction defect)")
        if c.get("rows_index_order_changed", 0) == 0:
            fails.append(f"variant {name}: no row's index order changed (construction defect)")
    pc = r.get("pos_ctrl_set_change") or {}
    if not pc.get("selecting_rows") or pc.get("rows_differ_selecting", 0) < POS_CTRL_FRAC * pc["selecting_rows"]:
        fails.append("positive control (set change) not detected on >= 90% of selecting rows")
    dre = r.get("record_decode_rel_err")
    if dre is None or dre > DECODE_REL_ERR_MAX:
        caveats.append(f"fp32 reference not valid (record decode rel err {dre})")
    mg = g(doc, "kernel", "prefill_mg_enabled")
    if version == "v091" and mg is not True:
        caveats.append(f"prefill_mg_enabled recorded as {mg!r}")
    facts = dict(repeat=rep, scratch_poison_equal=r.get("scratch_poison_equal"),
                 cache_poison_equal=r.get("cache_poison_equal"), relocation=r.get("relocation"),
                 base_vs_fp32_ref=r.get("base_vs_fp32_ref_rel_l2_median"),
                 variants={k: dict(rows_differ_selecting=v.get("rows_differ_selecting"),
                                   selecting_rows=v.get("selecting_rows"),
                                   rows_differ_shortcut=v.get("rows_differ_shortcut"),
                                   max_abs=g(v, "mag_selecting", "max_abs"),
                                   rel_l2_median=g(v, "mag_selecting", "rel_l2_median"),
                                   vs_fp32_ref=v.get("vs_fp32_ref_rel_l2_median"),
                                   repeat_equal=v.get("repeat_equal")) for k, v in var.items()})
    if fails:
        return dict(reading="INCOMPLETE", fails=fails, caveats=caveats, facts=facts)
    if not (rep.get("all_out_equal") and rep.get("all_lse_equal", True)):
        return dict(reading="A-EXEC", fails=[], caveats=caveats, facts=facts)
    if any(not v.get("repeat_equal") for v in var.values()):
        return dict(reading="A-EXEC", fails=[], caveats=caveats + ["a variant's own repeat differed"], facts=facts)
    if pc.get("rows_differ_shortcut", 1) != 0:
        return dict(reading="INCOMPLETE", fails=["positive control changed rows whose inputs did not change"],
                    caveats=caveats, facts=facts)
    reloc_bad = [x for x in (r.get("relocation") or []) if not x.get("out_equal")]
    if (reloc_bad or r.get("cache_poison_equal") is False
            or (r.get("scratch_poison_equal") and not all(r["scratch_poison_equal"]))):
        return dict(reading="B-ALLOC", fails=[], caveats=caveats, facts=facts)
    if any(v.get("rows_differ_shortcut", 0) for v in var.values()):
        return dict(reading="INCOMPLETE", fails=["rows with identical inputs differ between variants"],
                    caveats=caveats, facts=facts)
    if any(v.get("rows_differ_selecting", 0) > 0 for v in var.values()):
        return dict(reading="ORDER-SENSITIVE", fails=[], caveats=caveats, facts=facts)
    return dict(reading="ORDER-INSENSITIVE", fails=[], caveats=caveats, facts=facts)


def r3_unit(doc, r1u):
    if doc is None or not isinstance(doc.get("r3"), dict):
        why = "r3 record missing"
        if isinstance(doc, dict) and any(e.get("part") == "r3" for e in doc.get("errors", [])):
            why = "r3 error"
        return dict(reading="INCOMPLETE", fails=[why])
    if r1u.get("reading") == "INCOMPLETE":
        return dict(reading="INCOMPLETE", fails=["its R1 input regime is INCOMPLETE"])
    calls = doc["r3"].get("calls") or []
    if len(calls) < 2:
        return dict(reading="INCOMPLETE", fails=[f"{len(calls)} composed calls"])
    if any(c.get("rows_out_diff_without_topk_diff", 0) for c in calls):
        return dict(reading="ANOMALY", fails=[], facts=dict(calls=calls))
    if not any(c.get("rows_topk_order_diff", 0) for c in calls):
        return dict(reading="NO-TOPK-VARIATION", fails=[], facts=dict(calls=calls))
    if any(c.get("rows_out_diff", 0) for c in calls):
        return dict(reading="COMPOSED", fails=[], facts=dict(calls=calls))
    return dict(reading="ORDER-VARIED-OUTPUT-EQUAL", fails=[], facts=dict(calls=calls))


def analyze(run_dir):
    out = dict(schema="replay-topk-result/1", run_dir=os.path.basename(os.path.normpath(run_dir)), images={})
    for ver in ("v091", "v070"):
        d = os.path.join(run_dir, ver)
        env, e0 = load(os.path.join(d, "env.json"))
        r1, e1 = load(os.path.join(d, "r1.json"))
        r2, e2 = load(os.path.join(d, "r2.json"))
        im = dict(load_errors=[e for e in (e0, e1, e2) if e], r1={}, r2={})
        im["gpu"] = (env or {}).get("gpu")
        for reg in R1_REGIMES[ver]:
            im["r1"][reg] = r1_unit(r1, reg, ver)
        for reg in R2_REGIMES:
            im["r2"][reg] = r2_unit(r2, reg, ver)
        if ver == "v091":
            im["r3"] = r3_unit(r2, im["r1"].get("K", {}))
        if r2 is not None and ver == "v070":
            im["filter"] = r2.get("filter")
        im["kernel"] = (r2 or {}).get("kernel")
        im["r1_window"] = (r1 or {}).get("window")
        im["r1_stride_source"] = (r1 or {}).get("stride_source")
        kp = (r1 or {}).get("K_producer")
        im["r1_K_producer"] = {k: v for k, v in kp.items() if k != "tb"} if isinstance(kp, dict) else None
        out["images"][ver] = im
    out["summary"] = summary(out)
    return out


def summary(res):
    lines = []
    for ver, im in res["images"].items():
        r1 = ", ".join(f"{k}: {v['reading']}" for k, v in im["r1"].items())
        r2 = ", ".join(f"{k}: {v['reading']}" for k, v in im["r2"].items())
        lines.append(f"{ver} R1 [{r1}]; R2 [{r2}]" + (f"; R3 {im['r3']['reading']}" if "r3" in im else ""))
    return lines


def to_md(res):
    L = ["# replay-topk result", "", "SYNTHETIC inputs throughout (no model data). Readings per PROTOCOL.md section 7.", ""]
    for s in res["summary"]:
        L.append(f"- {s}")
    for ver, im in res["images"].items():
        L += ["", f"## {ver}", ""]
        if im["load_errors"]:
            L.append(f"Load errors: {', '.join(im['load_errors'])}")
        L.append("| Unit | Reading | Key facts | Failed gates / caveats |")
        L.append("|---|---|---|---|")
        for reg, u in im["r1"].items():
            f = u.get("facts") or {}
            kf = (f"order varied: {f.get('frac_rows_order_varied_same_alloc')} same-alloc / "
                  f"{f.get('frac_rows_order_varied_relocated')} relocated of selecting rows; set varied rows "
                  f"{f.get('rows_set_varied_same_alloc')}/{f.get('rows_set_varied_relocated')}; last col varied "
                  f"{f.get('rows_last_col_varied')}") if f else ""
            L.append(f"| R1 {reg} | {u['reading']} | {kf} | {'; '.join(u.get('fails', []) + u.get('caveats', []))} |")
        for reg, u in im["r2"].items():
            f = u.get("facts") or {}
            vv = f.get("variants") or {}
            kf = "; ".join(f"{k}: {v.get('rows_differ_selecting')}/{v.get('selecting_rows')} rows, max_abs "
                           f"{v.get('max_abs')}" for k, v in vv.items())
            L.append(f"| R2 {reg} | {u['reading']} | {kf} | {'; '.join(u.get('fails', []) + u.get('caveats', []))} |")
        if "r3" in im:
            u = im["r3"]
            L.append(f"| R3 | {u['reading']} | | {'; '.join(u.get('fails', []))} |")
    return "\n".join(L) + "\n"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    run_dir = sys.argv[1]
    args = sys.argv[2:]
    oj = args[args.index("--out-json") + 1] if "--out-json" in args else os.path.join(run_dir, "replay-result.json")
    om = args[args.index("--out-md") + 1] if "--out-md" in args else os.path.join(run_dir, "replay-result.md")
    res = analyze(run_dir)
    with open(oj, "w") as fh:
        json.dump(res, fh, indent=1)
    with open(om, "w") as fh:
        fh.write(to_md(res))
    for s in res["summary"]:
        print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
