#!/usr/bin/env python3
"""Host-side: turn scan.json (container output) into inventory.json and the INVENTORY.md tables.
Metrics, positions, hashes only. Arm metadata from screen preregistrations/logs (see INVENTORY.md sources)."""
import json, sys, collections, hashlib, os

SCAN, OUTDIR = sys.argv[1], sys.argv[2]
s = json.load(open(SCAN))

# dir prefix (relative to <data-runs>) -> arm metadata
K325_070 = "K3.25@701cd745"
K325 = "K3.25@0490d2f7"
ARMS = {
 "diagnostics/empties-k4-v0.8.0-2026-09-29": dict(engine="v0.8.0 (inferred; no server log)", weights="K4 (inferred)", layout="TP2 DCP1", spec="DFlash2 x3", seeds="unseeded", cap=327680, truncation="none", conc=8),
 "diagnostics/empties-k325-v0.7.0-2026-09-30": dict(engine="v0.7.0 stock", weights=K325_070, layout="EP2 DCP2", spec="DFlash2 x5", seeds="unseeded", cap=327680, truncation="none", conc=8),
 "diagnostics/empties-k325-v0.8.0-2026-09-30": dict(engine="v0.8.0 stock", weights=K325, layout="TP2 DCP1", spec="DFlash2 x3", seeds="unseeded", cap=327680, truncation="none", conc=8),
 "diagnostics/seed-replay-2026-09-30/v0.7.0": dict(engine="v0.7.0 stock", weights=K325_070, layout="EP2 DCP2", spec="DFlash2 x5", seeds="1234 all", cap=327680, truncation="none", conc=8),
 "diagnostics/seed-replay-2026-09-30/v0.8.0": dict(engine="v0.8.0 stock", weights=K325, layout="TP2 DCP1", spec="DFlash2 x3", seeds="1234 all", cap=327680, truncation="none", conc=8),
 "diagnostics/spec-k5-v080-2026-10-01/replay": dict(engine="v0.8.0 stock", weights=K325, layout="TP2 DCP1", spec="DFlash2 x5", seeds="1234 all", cap=327680, truncation="none", conc=8),
 "diagnostics/spec-screen-v070-k3-2026-10-01/replay": dict(engine="v0.7.0 stock", weights=K325_070, layout="EP2 DCP2", spec="DFlash2 x3", seeds="1234 all", cap=327680, truncation="client cut 120k", conc=8),
 "diagnostics/spec-slot-sharing-v080-2026-10-01/replay": dict(engine="v0.8.0 stock, sharing 0", weights=K325, layout="TP2 DCP1", spec="DFlash2 x3", seeds="1234 all", cap=327680, truncation="client cut 120k", conc=8),
 "diagnostics/gpqa-retry-2026-10-01/retry-A-sharing0": dict(engine="v0.8.0 stock, sharing 0", weights=K325, layout="TP2 DCP1", spec="DFlash2 x3", seeds="1234 all (script)", cap=327680, truncation="none", conc=8),
 "diagnostics/gpqa-retry-2026-10-01/retry-B-k5": dict(engine="v0.8.0 stock", weights=K325, layout="TP2 DCP1", spec="DFlash2 x5", seeds="1234 all (script)", cap=327680, truncation="none", conc=8),
 "diagnostics/gpqa-retry-2026-10-01/retry-C-default": dict(engine="v0.8.0 stock", weights=K325, layout="TP2 DCP1", spec="DFlash2 x3", seeds="1234 all (script)", cap=327680, truncation="none", conc=8),
 "diagnostics/k4-k5-share-screen-2026-10-01/replay-C-k4-default": dict(engine="v0.8.0 stock", weights="K4@a5fee929", layout="TP2 DCP1", spec="DFlash2 x3", seeds="1234+rep-1", cap=327680, truncation="client cut 120k", conc=8),
 "diagnostics/k4-k5-share-screen-2026-10-01/replay-K5": dict(engine="v0.8.0 stock", weights="K4@a5fee929", layout="TP2 DCP1", spec="DFlash2 x5", seeds="1234+rep-1", cap=327680, truncation="client cut 120k; arm killed", conc=8),
 "diagnostics/kpool-k1-screen-2026-10-04/A": dict(engine="v0.8.0 stock", weights=K325, layout="TP2 DCP1", spec="DFlash2 x3", seeds="1234 all", cap=327680, truncation="zlib early stop", conc=8),
 "diagnostics/kpool-k1-screen-2026-10-04/B": dict(engine="v0.8.0 stock", weights=K325, layout="TP2 DCP1", spec="DFlash2 x1", seeds="1234 all", cap=327680, truncation="zlib early stop", conc=8),
 "diagnostics/kpool-k1-screen-2026-10-04/C": dict(engine="v0.8.0 local kpoolring", weights=K325, layout="TP2 DCP1", spec="DFlash2 x3", seeds="1234 all", cap=327680, truncation="zlib early stop", conc=8),
 "diagnostics/layout-bisect-2026-10-07/A1": dict(engine="v0.9.0 stock", weights=K325, layout="TP2 DCP1", spec="DFlash2 x3", seeds="1234 all", cap=327680, truncation="zlib early stop", conc=8),
 "diagnostics/layout-bisect-2026-10-07/A2": dict(engine="v0.9.0 stock", weights=K325, layout="TP2 DCP1", spec="DFlash2 x3", seeds="1234 all", cap=327680, truncation="zlib early stop", conc=8),
 "diagnostics/layout-bisect-2026-10-07/D1": dict(engine="v0.9.0 EP2/DCP2 bundle", weights=K325, layout="EP2 DCP2", spec="DFlash2 x3", seeds="1234 all", cap=327680, truncation="zlib early stop", conc=8),
 "diagnostics/layout-bisect-2026-10-07/D2": dict(engine="v0.9.0 EP2/DCP2 bundle", weights=K325, layout="EP2 DCP2", spec="DFlash2 x3", seeds="1234 all", cap=327680, truncation="zlib early stop", conc=8),
 "diagnostics/layout-bisect-2026-10-07/E1": dict(engine="v0.9.0 local tailfix", weights=K325, layout="TP2 DCP1", spec="DFlash2 x3", seeds="1234 all", cap=327680, truncation="zlib early stop", conc=8),
 "diagnostics/layout-bisect-2026-10-07/E2": dict(engine="v0.9.0 local tailfix", weights=K325, layout="TP2 DCP1", spec="DFlash2 x3", seeds="1234 all", cap=327680, truncation="zlib early stop", conc=8),
 "diagnostics/component-screen-2026-10-09/B79": dict(engine="v0.9.1 stock", weights=K325, layout="TP2 DCP1", spec="DFlash2 x3", seeds="5000+rep", cap=327680, truncation="zlib early stop", conc=12),
 "diagnostics/component-screen-2026-10-09/B": dict(engine="v0.9.1 stock", weights=K325, layout="TP2 DCP1", spec="DFlash2 x3", seeds="5000+rep", cap=327680, truncation="zlib early stop", conc=12),
 "diagnostics/component-screen-2026-10-09/EN": dict(engine="v0.9.1 EP2 NOPE0", weights=K325, layout="EP2 DCP1", spec="DFlash2 x3", seeds="5000+rep", cap=327680, truncation="zlib early stop", conc=12),
 "diagnostics/component-screen-2026-10-09/EO": dict(engine="v0.9.1 EP2 owners-tp sharing0", weights=K325, layout="EP2 DCP1", spec="DFlash2 x3", seeds="5000+rep", cap=327680, truncation="zlib early stop", conc=12),
 "diagnostics/component-screen-2026-10-09/NO": dict(engine="v0.9.1 owners-tp NOPE0 sharing0", weights=K325, layout="TP2 DCP1", spec="DFlash2 x3", seeds="5000+rep", cap=327680, truncation="zlib early stop", conc=12),
 "diagnostics/component-screen-2026-10-09/V79": dict(engine="v0.7.0 stock image, DFlash x3", weights=K325_070, layout="EP2 DCP2", spec="DFlash2 x3", seeds="5000+rep", cap=327680, truncation="zlib early stop", conc=12),
 "diagnostics/component-screen-2026-10-09/S1234": dict(engine="v0.9.1 stock", weights=K325, layout="TP2 DCP1", spec="DFlash2 x3", seeds="1234 all", cap=327680, truncation="zlib early stop", conc=12),
 "eval-matrix-2026-10-04/screens/S7p": dict(engine="v0.7.0 local kpoolring", weights=K325_070, layout="EP2 DCP2", spec="DFlash2 x5", seeds="1234 all", cap=327680, truncation="zlib early stop", conc=8),
 "eval-matrix-2026-10-04/screens/S7K4p": dict(engine="v0.7.0 local kpoolring (INVALID: engine crash)", weights="K4@a5fee929", layout="EP2 DCP2", spec="DFlash2 x5", seeds="1234 all", cap=327680, truncation="errors", conc=8),
 "eval-matrix-2026-10-04/screens/S8K4p": dict(engine="v0.8.0 local kpoolring", weights="K4@a5fee929", layout="TP2 DCP1", spec="DFlash2 x3", seeds="1234 all", cap=327680, truncation="zlib early stop", conc=8),
 "eval-matrix-2026-10-04/screens/S7": dict(engine="v0.7.0 stock", weights=K325_070, layout="EP2 DCP2", spec="DFlash2 x5", seeds="1234 all", cap=327680, truncation="zlib early stop", conc=8),
 "investigations/2026-10-glm53-completion/pilot/A": dict(engine="v0.9.1 stock", weights=K325, layout="TP2 DCP1", spec="DFlash2 x3", seeds="manifest", cap="65536/131072", truncation="none (early stop off)", conc=8),
 "investigations/2026-10-glm53-completion/pilot/B": dict(engine="v0.9.1 EP2/DCP2 bundle", weights=K325, layout="EP2 DCP2", spec="DFlash2 x3", seeds="manifest", cap="65536/131072", truncation="none (early stop off)", conc=8),
 "investigations/2026-10-glm53-completion/replication/A": dict(engine="v0.9.1 stock", weights=K325, layout="TP2 DCP1", spec="DFlash2 x3", seeds="manifest", cap="65536/131072", truncation="none (early stop off)", conc=8),
 "investigations/2026-10-glm53-completion/replication/B": dict(engine="v0.9.1 EP2/DCP2 bundle", weights=K325, layout="EP2 DCP2", spec="DFlash2 x3", seeds="manifest", cap="65536/131072", truncation="none (early stop off)", conc=8),
}
TRACE_ARMS = {
 "step5/v070fix": dict(engine="v0.7.0 + kpool fixes (local)", weights=K325_070, layout="EP2 DCP2", spec="DFlash2 x5", sampler="T1.0 top_p 1.0 ignore_eos", seeds="1234+doc"),
 "step6/v091dcp2": dict(engine="v0.9.1 EP2/DCP2 bundle", weights=K325, layout="EP2 DCP2", spec="DFlash2 x3", sampler="T1.0 top_p 1.0 ignore_eos", seeds="1234+doc"),
}


def arm_of(path):
    best = None
    for k in ARMS:
        if path.startswith(k + "/") and (best is None or len(k) > len(best)):
            best = k
    return best


def state_class(r):
    fin = r["finish_reason"] or ""
    if fin.startswith("ERROR"):
        return "error"
    end = r.get("end") or {}
    if fin == "stop":
        return "natural_close" if r["closed"] else "stop_without_close"
    if r.get("stopped_early"):
        return "loop"                      # client zlib detector (3 checks < 0.10)
    if end.get("repeating"):
        return "loop"                      # end-of-reply zlib30k < 0.15 or rep200 >= 0.5
    if r["closed"]:
        return "closed_then_capped"        # reasoning closed, content ran to the cap/cut
    if r.get("cut"):
        return "exhaustion_cut120k" if not end.get("repeating") else "loop"
    return "exhaustion"


def band(n):
    if n is None:
        return None
    return "<16k" if n < 16000 else ("16k-64k" if n < 64000 else ("64k-128k" if n < 128000 else ">=128k"))


rows = []
for r in s["text_rows"]:
    a = arm_of(r["path"])
    meta = ARMS.get(a, {})
    st = state_class(r)
    n = r["n_reasoning_tokens_retok"]
    ev = r["strong_events"]
    ev_win = [e for e in ev if 56000 <= e[0] <= 136000]
    ev_nonrep_win = [e for e in ev_win if e[4] is False]
    # windows usable for the 64k-128k probe: non-repeating at 64k / 96k / 128k checkpoints
    cps = r["checkpoints"]
    nonrep_cp = [int(k) for k, v in cps.items() if v and not v["repeating"]]
    rep_cp = [int(k) for k, v in cps.items() if v and v["repeating"]]
    last_commit_before_end = ev[-1][0] if ev else None
    out = dict(
        id=hashlib.sha256(r["path"].encode()).hexdigest()[:12], path=r["path"], arm=a, **{k: meta.get(k) for k in ("engine", "weights", "layout", "spec", "seeds", "truncation", "conc")},
        doc=r["doc"], rep=r["rep"], seed=r["seed"], finish_reason=r["finish_reason"], client_cls=r["client_cls"],
        stopped_early=r["stopped_early"], cut=r["cut"], completion_tokens=r["completion_tokens"],
        prompt_tokens_logged=r["prompt_tokens_logged"], prompt_tokens_rendered=r["prompt_tokens_rendered"],
        prompt_match=(r["prompt_tokens_logged"] == r["prompt_tokens_rendered"]) if r["prompt_tokens_logged"] is not None else None,
        ids="text-only (re-tokenized)", n_reasoning_tokens=n, length_band=band(n), retok_drift=r["retok_drift"],
        closed=r["closed"], natural_close_pos=(n if r["closed"] else None), state=st,
        censored=(None if st in ("natural_close", "stop_without_close") else (r["truncation_reason"] if "truncation_reason" in r else ("client early stop" if r["stopped_early"] else ("client cut 120k" if r["cut"] else "token cap")))),
        loop_onset_tok=r["loop_onset_tok"], first_repeating_tok=r["first_repeating_tok"], rep200_all=r["rep200_all"],
        end_metrics=r["end"], checkpoints=cps, nonrepeating_checkpoints=sorted(nonrep_cp), repeating_checkpoints=sorted(rep_cp),
        commit_counts=r["commit_counts"], strong_events_total=len(ev),
        strong_events_56k_136k=ev_win[:300], strong_events_56k_136k_nonrepeating=len(ev_nonrep_win),
        strong_events_last5=ev[-5:], last_commit_tok=last_commit_before_end,
        close_minus_last_commit=(n - last_commit_before_end) if (r["closed"] and last_commit_before_end is not None) else None,
        literal_markers_in_text=r["literal_in_text"], special_ids_in_retok=r["special_ids_in_retok"],
        sha256_reasoning_text=r["sha256_reasoning_text"], sha256_reasoning_ids=r["sha256_reasoning_ids"],
        content_chars=r["content_chars"], content_full=r["content_full"],
    )
    rows.append(out)

traces = []
for t in s["exact_traces"]:
    key = "step5/v070fix" if "/step5/v070fix" in t["path"] else "step6/v091dcp2"
    am = t["active_metrics"] or {}
    ev = am.get("strong_events", [])
    traces.append(dict(
        id=hashlib.sha256(t["path"].encode()).hexdigest()[:12], path=t["path"], **TRACE_ARMS[key], doc=int(t["prompt_id"]),
        seed=t["seed"], ids="exact generated ids", prompt_len=t["prompt_len"], prompt_equals_rendered_gpqa_doc=t["prompt_equals_rendered_gpqa_doc"],
        n_gen=t["n_gen"], natural_close_pos=t["first_close"], first_terminator_pos=t["first_terminator"],
        first_terminator_id=t["first_terminator_id"], tokens_close_to_terminator=t["tokens_close_to_terminator"],
        n_close_in_16k=t["n_close"], n_terminators_in_16k=t["n_terminators"],
        forced_continuation_from=(t["first_terminator"] + 1) if t["first_terminator"] is not None else None,
        token_before_close=t["token_before_close"], token_before_close_is_ws=t["token_before_close_text_is_ws"],
        decode_lp_close_at_close=t["decode_lp_close_at_close"], decode_lp_terminator_at_term=t["decode_lp_terminator_at_term"],
        close_in_top20_count_last2048=t.get("close_in_top20_count_last2048"),
        close_in_top20_positions_last2048=t.get("close_in_top20_positions_last2048"),
        active_roundtrip_exact=t["active_roundtrip_exact"], active_retok_len=t.get("active_retok_len"),
        state="natural_close", active_end_metrics=am.get("end"), active_rep200_all=am.get("rep200_all"),
        active_strong_events=ev, length_band=band(t["first_close"]),
        sha256_prompt_ids=t["sha256_prompt_ids"], sha256_gen_ids=t["sha256_gen_ids"], sha256_active_ids=t["sha256_active_ids"]))

inv = dict(
    note="Local only. Fixture inventory for PLAN.md amendment 20 (+20b). Metrics, positions (token index within the reasoning, 0 = first token after <think>; absolute = prompt_tokens_rendered + index) and hashes only. No reasoning or GPQA text.",
    params=s["params"], phase2_requests=s["phase2_requests"], exact_traces=traces, text_rows=rows)
json.dump(inv, open(os.path.join(OUTDIR, "inventory.json"), "w"), indent=1)

# ---- summary tables for INVENTORY.md
C = collections.Counter
by_arm = collections.defaultdict(list)
for r in rows:
    by_arm[r["arm"]].append(r)
lines = []
lines.append("| Arm (dir) | Engine | Weights | n | natural close | loop | exhaustion | stop w/o close | closed then capped | error | reached 64k | reached 128k |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for a in sorted(by_arm, key=lambda x: x or ""):
    rs = by_arm[a]
    c = C(r["state"] for r in rs)
    lines.append(f"| {a} | {rs[0]['engine']} | {rs[0]['weights']} | {len(rs)} | {c['natural_close']} | {c['loop']} | {c['exhaustion'] + c['exhaustion_cut120k']} | {c['stop_without_close']} | {c['closed_then_capped']} | {c['error']} | "
                 f"{sum(1 for r in rs if (r['n_reasoning_tokens'] or 0) >= 64000)} | {sum(1 for r in rs if (r['n_reasoning_tokens'] or 0) >= 128000)} |")
open(os.path.join(OUTDIR, "arm_table.md.part"), "w").write("\n".join(lines) + "\n")
print("rows", len(rows), "traces", len(traces))
print(C(r["state"] for r in rows))
