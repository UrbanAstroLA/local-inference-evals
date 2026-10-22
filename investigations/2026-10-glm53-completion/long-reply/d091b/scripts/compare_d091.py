#!/usr/bin/env python3
"""Side-by-side long-reply tables, d091 (stock 0.9.1, one pass, seed 70705) against rep070-F (stock 0.7.0, three passes),
under the discovery's frozen definitions (host, stdlib only, zero GPU). DESCRIPTIVE ONLY: one pass, about 8 replies past
100k expected, different seeds; no test, no threshold, no reading.

Definitions come from the discovery itself:
  - per-reply and per-window metrics, terminal onset, episodes and the PRESPECIFIED loop class are trajectory.py's outputs
    (out/replies_all.json, out/windows_all.jsonl; byte-identical copy run unchanged);
  - the POST HOC exact-loop class uses periods.py's LRS (out/periods_all.json; byte-identical copy run unchanged) with the
    rule of discovery's analyse.py (T3), reproduced here;
  - the helper functions med, cls_set, reply_medians, tw_cpt, tw and auc and the METRICS list are taken from discovery's
    analyse.py itself (byte-identical copy in scripts/, sha256 checked) by extracting those definitions with ast and
    executing only them; the set and group constructions below mirror analyse.py's T1, T3, T4 and T12 code.
Proof that nothing changed: `--reproduce` recomputes the rep070-F side from discovery/out alone and compares every
recomputed quantity with discovery's summary.json (exact equality); tests/compare-repro.txt.

usage: compare_d091.py --d091-out DIR [--analysis RESULT.json] --out-md T.md --out-json T.json [--label TEXT]
       compare_d091.py --reproduce
Writes ids, positions, counts and metrics only.
"""
import argparse, ast, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DISC = "<runs>/long-reply/discovery/out"
ANALYSE = os.path.join(HERE, "analyse.py")
ANALYSE_SHA = "1e599f2f606a31fd55be125904739a3089c83367371ade9003531b31b594410a"
DISC_SUMMARY_SHA = "021bcb013c5691f410868886213c4d72bfb0c9d34847814cf7bdb9a334e49308"
FUNCS = ("med", "f", "cls_set", "reply_medians", "tw_cpt", "tw", "auc")
CAP = 327680


def load_defs():
    src = open(ANALYSE, "rb").read()
    if hashlib.sha256(src).hexdigest() != ANALYSE_SHA:
        sys.exit("scripts/analyse.py is not discovery's frozen file")
    tree = ast.parse(src.decode())
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in FUNCS)
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "METRICS" for t in n.targets))]
    ns = {"st": __import__("statistics")}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "analyse.py (definitions)", "exec"), ns)
    missing = [x for x in FUNCS + ("METRICS",) if x not in ns]
    if missing:
        sys.exit(f"definitions not found in analyse.py: {missing}")
    return ns


D = load_defs()
med, cls_set, reply_medians, tw_cpt, tw, auc = D["med"], D["cls_set"], D["reply_medians"], D["tw_cpt"], D["tw"], D["auc"]
SHOW = ["cpt", "rep20", "rep50", "rep100", "zr", "dlr", "line_seen", "cjk", "digit", "punct", "space", "tok_le1char",
        "chk_verify_per1k", "chk_verify_rep", "chk_wait_per1k", "chk_wait_rep"]        # analyse.py T1 'show'
FEATS = ("cpt", "digit", "chk_wait", "chk_verify", "rep20", "R50_100k")              # analyse.py T12


def load(out_dir):
    reps = {r["id"]: r for r in json.load(open(f"{out_dir}/replies_all.json"))}
    per = {r["id"]: r for r in json.load(open(f"{out_dir}/periods_all.json"))}
    wins = {}
    for line in open(f"{out_dir}/windows_all.jsonl"):
        w = json.loads(line)
        wins.setdefault(w["id"], []).append(w)
    idx = json.load(open(f"{out_dir}/rep070F_index.json"))
    return reps, per, wins, idx


def group_medians(reps, wins):   # analyse.py T1
    groups = {"comparison 50-100k (finished)": [], "finished >=100k: all windows": [], "finished >=100k: windows before 100k": [],
              "capped: windows before 100k": [], "capped: pre-onset windows": [], "capped: post-onset windows": []}
    for i, r in reps.items():
        ws = wins[i]
        c = cls_set(r)
        if c == "comparison 50-100k":
            groups["comparison 50-100k (finished)"].append(reply_medians(ws))
        elif c == "finished>=100k":
            groups["finished >=100k: all windows"].append(reply_medians(ws))
            groups["finished >=100k: windows before 100k"].append(reply_medians([w for w in ws if w["g1"] <= 100000]))
        else:
            groups["capped: windows before 100k"].append(reply_medians([w for w in ws if w["g1"] <= 100000]))
            o = r["onset"] if r["onset"] is not None else r["N_reasoning"]
            pre = [w for w in ws if w["g1"] <= o]
            post = [w for w in ws if w["g0"] >= o]
            if pre:
                groups["capped: pre-onset windows"].append(reply_medians(pre))
            if post:
                groups["capped: post-onset windows"].append(reply_medians(post))
    return {g: {"n": len(rows), **{m: med([x[m] for x in rows]) for m in SHOW}} for g, rows in groups.items()}


def capped_rows(reps, per):      # analyse.py T3 (prespecified class from trajectory.py; post hoc class from the LRS)
    out = []
    for i, r in sorted(reps.items(), key=lambda kv: kv[1]["T"]):
        if not (r["finish_reason"] == "length" or r["onset"] is not None):
            continue
        pr = per[i]
        lpo = pr.get("lrs_post_onset") or {}
        lag = pr.get("lag_post_onset") or {}
        mp = None
        if lag and lpo.get("offset"):
            mp = dict((p_, v) for p_, v in lag["top_lags"]).get(lpo["offset"])
        region = r["N_reasoning"] - r["onset"] if r["onset"] is not None else None
        if r["onset"] is None:
            ph = "no terminal loop"
        elif lpo.get("len", 0) >= 0.5 * region:
            ph = f"exact loop, period {lpo['offset']}" + (" (short)" if lpo["offset"] <= 4096 else " (long)")
        else:
            ph = f"partial / fragmentary (LRS {lpo.get('len')} of {region})"
        eps = r["episodes"]
        out.append({"id": i, "T": r["T"], "finish": r["finish_reason"], "onset": r["onset"], "onset_abs": r["onset_abs"],
                    "onset_03": r["sensitivity"]["0.3"]["onset"], "onset_07": r["sensitivity"]["0.7"]["onset"],
                    "R50_100k": r["R50_at"]["100000"], "prespec_class": r["loop"]["class"], "E": r["loop"].get("E"),
                    "lrs_post": lpo.get("len"), "lrs_start": lpo.get("start_a"), "period": lpo.get("offset"), "m_period": mp,
                    "posthoc_class": ph, "episodes": len(eps), "recovered": sum(not e["terminal"] for e in eps),
                    "R50_final": r["R50_final"], "R50_max": r["R50_max"], "prompt_tokens": r["prompt_tokens"], "doc": r["doc"]})
    return out


def episode_counts(reps):        # analyse.py T4 summary, plus recovered-episode detail
    sets = ("capped", "finished>=100k", "comparison 50-100k")
    n_any = {c: sum(1 for r in reps.values() if cls_set(r) == c and r["episodes"]) for c in sets}
    n_tot = {c: sum(1 for r in reps.values() if cls_set(r) == c) for c in sets}
    rec = {c: sorted(i for i, r in reps.items() if cls_set(r) == c and any(not e["terminal"] for e in r["episodes"])) for c in sets}
    fin_inside = sorted(i for i, r in reps.items() if r["finish_reason"] != "length" and r["onset"] is not None)
    return {"with_any_episode": n_any, "replies": n_tot, "replies_with_a_recovered_episode": rec,
            "recovered_episodes": {c: sum(sum(not e["terminal"] for e in reps[i]["episodes"]) for i in rec[c]) for c in sets},
            "finished_with_a_terminal_onset (finished from inside repetition)": fin_inside}


def at100k(reps, wins):          # analyse.py T12
    atr = [i for i in reps if reps[i]["T"] >= 100000]
    feat = {}
    for i in atr:
        ws = [w for w in wins[i] if w["g1"] <= 100000]
        feat[i] = {"cpt": tw_cpt(ws), "digit": tw(ws, "digit"), "chk_wait": tw(ws, "chk_wait_per1k"), "chk_verify": tw(ws, "chk_verify_per1k"),
                   "rep20": tw(ws, "rep20"), "R50_100k": reps[i]["R50_at"]["100000"], "capped": reps[i]["finish_reason"] == "length"}
    res = {}
    for m in FEATS:
        pos = [feat[i][m] for i in atr if feat[i]["capped"] and feat[i][m] is not None]
        neg = [feat[i][m] for i in atr if not feat[i]["capped"] and feat[i][m] is not None]
        res[m] = {"capped_median": med(pos), "finished_median": med(neg), "auc": auc(pos, neg) if pos and neg else None,
                  "n_capped": len(pos), "n_finished": len(neg)}
    return res, feat


def cpt_proxy(idx):              # analyse.py T13 proxy: (reasoning + content chars) / completion tokens, request level
    cp = lambda x: (x["reasoning_chars"] + x["content_chars"]) / x["T"]
    c = sorted(cp(x) for x in idx if x["finish_reason"] == "length" and x["T"] == CAP)
    fl = [cp(x) for x in idx if not (x["finish_reason"] == "length" and x["T"] == CAP) and x["T"] >= 100000]
    mid = [cp(x) for x in idx if 50000 <= x["T"] < 100000]
    return {"capped": c, "capped_median": med(c), "capped_ge3": sum(v >= 3.0 for v in c),
            "finished_ge100k_n": len(fl), "finished_ge100k_median": med(fl), "50_100k_median": med(mid)}


def side(out_dir):
    reps, per, wins, idx = load(out_dir)
    a, feat = at100k(reps, wins)
    return {"group_medians": group_medians(reps, wins), "capped_and_terminal": capped_rows(reps, per), "episode_counts": episode_counts(reps),
            "at100k_separation": a, "at100k_features": feat, "cpt_proxy": cpt_proxy(idx),
            "sets": {c: sum(1 for r in reps.values() if cls_set(r) == c) for c in ("capped", "finished>=100k", "comparison 50-100k")},
            "at_risk_docs": sorted({r["doc"] for r in reps.values() if r["T"] >= 100000}),
            "capped_docs": sorted({r["doc"] for r in reps.values() if r["finish_reason"] == "length"}),
            "finished_at_risk": {r["doc"]: r["finish_reason"] != "length" for r in reps.values() if r["T"] >= 100000},
            "drift": {"min": min((r["drift"] for r in reps.values()), default=None), "max": max((r["drift"] for r in reps.values()), default=None),
                      "hash_verify_failures": sum(r["hash_verify_failures"] for r in reps.values())}}


def reproduce():
    s = json.load(open(f"{DISC}/summary.json"))
    if hashlib.sha256(open(f"{DISC}/summary.json", "rb").read()).hexdigest() != DISC_SUMMARY_SHA:
        sys.exit("discovery summary.json changed")
    R = side(DISC)
    ok_all = True
    def eq(name, a, b):
        nonlocal ok_all
        good = json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
        ok_all &= good
        print(("SAME " if good else "DIFF ") + name)
    for g, v in s["group_medians"].items():
        eq(f"T1 group medians: {g}", v, R["group_medians"][g])
    eq("T3 capped and terminal-onset rows (prespecified class, post hoc class, onsets, episodes)", s["capped_and_terminal"], R["capped_and_terminal"])
    eq("T4 episode counts", s["episode_counts"], {k: R["episode_counts"][k] for k in ("with_any_episode", "replies")})
    eq("T12 first-100k separation (medians, AUC)", s["at100k_separation"], {m: {k: v[k] for k in ("capped_median", "finished_median", "auc")} for m, v in R["at100k_separation"].items()})
    eng = s["engine_cpt"]["0.7.0 rep-F"]
    eq("T13 rep070-F capped chars/token proxy values and >= 3.0 count (request-log chars in discovery, raw-index chars here)",
       [eng["capped"], eng["capped_ge3"], eng["fin_long_median"], eng["mid_median"]],
       [R["cpt_proxy"]["capped"], R["cpt_proxy"]["capped_ge3"], R["cpt_proxy"]["finished_ge100k_median"], R["cpt_proxy"]["50_100k_median"]])
    eq("drift and hash verification", s["drift"], R["drift"])
    print("reproduction:", "ALL SAME" if ok_all else "DIFFERENCES FOUND")
    return 0 if ok_all else 1


def fmt(x, nd=3):
    return D["f"](x, nd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reproduce", action="store_true")
    ap.add_argument("--d091-out")
    ap.add_argument("--analysis", default=None)
    ap.add_argument("--out-md")
    ap.add_argument("--out-json")
    ap.add_argument("--label", default="stock 0.9.1 d091 (1 pass, seed 70705)")
    a = ap.parse_args()
    if a.reproduce:
        return reproduce()
    A, B = side(DISC), side(a.d091_out)
    s = json.load(open(f"{DISC}/summary.json"))
    panel = {row["doc"]: row for row in s["panel"]}
    res = {"status": "DESCRIPTIVE: one pass, about 8 replies past 100k expected, different seeds from rep070-F; no test, no threshold, no reading",
           "labels": {"A": "stock 0.7.0 rep070-F (3 passes, seeds 70701-70703)", "B": a.label}, "rep070F": A, "d091": B}
    L = [f"# Long-reply side-by-side: {res['labels']['B']} vs {res['labels']['A']}", "", f"**{res['status']}.** Definitions: discovery PRESPEC.md "
         "(sha256 97040a5c...) through trajectory.py / periods.py run unchanged; table rules as discovery's analyse.py. Labels: [P] prespecified "
         "rule, [PH] post hoc (discovery's LRS supplement).", ""]
    L.append(f"Sets: rep070-F capped {A['sets']['capped']}, finished >= 100k {A['sets']['finished>=100k']}, 50-100k {A['sets']['comparison 50-100k']}; "
             f"d091 capped {B['sets']['capped']}, finished >= 100k {B['sets']['finished>=100k']}, 50-100k {B['sets']['comparison 50-100k']}. "
             f"Re-tokenization drift d091 {B['drift']}.")
    # 1. first 100k
    L += ["", "## 1. First 100k tokens: chars/token, digit share, interjection and verification rates", "",
          "Token-weighted over full windows ending at or before 100k (discovery T12 features); medians over replies.", "",
          "| feature | rep070-F capped | rep070-F finished >= 100k | AUC | d091 capped | d091 finished >= 100k | AUC |", "|---|---:|---:|---:|---:|---:|---:|"]
    for m in ("cpt", "digit", "chk_wait", "chk_verify", "rep20", "R50_100k"):
        x, y = A["at100k_separation"][m], B["at100k_separation"][m]
        L.append(f"| {m} | {fmt(x['capped_median'])} (n {x['n_capped']}) | {fmt(x['finished_median'])} (n {x['n_finished']}) | {fmt(x['auc'], 2)} | "
                 f"{fmt(y['capped_median'])} (n {y['n_capped']}) | {fmt(y['finished_median'])} (n {y['n_finished']}) | {fmt(y['auc'], 2)} |")
    L += ["", "Per-set medians of each reply's window medians (discovery T1; chk_* per 1,000 tokens):", "",
          "| set | engine | n | cpt | digit | punct | tok_le1char | chk_verify_per1k | chk_wait_per1k | rep50 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in A["group_medians"]:
        for nm, S in (("rep070-F", A), ("d091", B)):
            v = S["group_medians"][g]
            L.append(f"| {g} | {nm} | {v['n']} | {fmt(v['cpt'])} | {fmt(v['digit'])} | {fmt(v['punct'])} | {fmt(v['tok_le1char'])} | "
                     f"{fmt(v['chk_verify_per1k'])} | {fmt(v['chk_wait_per1k'])} | {fmt(v['rep50'])} |")
    L += ["", f"Request-level chars/token proxy (discovery T13): rep070-F capped median {fmt(A['cpt_proxy']['capped_median'], 2)} "
          f"({A['cpt_proxy']['capped_ge3']} of {len(A['cpt_proxy']['capped'])} >= 3.0); d091 capped median {fmt(B['cpt_proxy']['capped_median'], 2)} "
          f"({B['cpt_proxy']['capped_ge3']} of {len(B['cpt_proxy']['capped'])} >= 3.0); finished >= 100k {fmt(A['cpt_proxy']['finished_ge100k_median'], 2)} vs "
          f"{fmt(B['cpt_proxy']['finished_ge100k_median'], 2)}."]
    # 2. capped: onset and loop type
    L += ["", "## 2. Capped replies (and finished replies with a terminal onset): onset and loop type", "",
          "| engine | id | T | outcome | onset g [P] | onset @0.3 / @0.7 | R50 at 100k | loop class [P] | E | LRS after onset | period | loop class [PH] | episodes (recovered) | R50 final / max |",
          "|---|---|---:|---|---:|---|---:|---|---:|---:|---:|---|---|---|"]
    for nm, S in (("rep070-F", A), ("d091", B)):
        for r in S["capped_and_terminal"]:
            L.append(f"| {nm} | {r['id']} | {r['T']} | {r['finish']} | {fmt(r['onset'])} | {fmt(r['onset_03'])} / {fmt(r['onset_07'])} | {fmt(r['R50_100k'])} | "
                     f"{r['prespec_class']} | {fmt(r['E'], 2)} | {fmt(r['lrs_post'])} | {fmt(r['period'])} | {r['posthoc_class']} | {r['episodes']} ({r['recovered']}) | "
                     f"{fmt(r['R50_final'], 2)} / {fmt(r['R50_max'], 2)} |")
    def tally(S, key, capped_only=True):
        c = {}
        for r in S["capped_and_terminal"]:
            if capped_only and r["finish"] != "length":
                continue
            k = r[key] if key == "prespec_class" else (r[key].split(",")[0] if r[key].startswith("exact loop") else r[key].split(" (")[0])
            c[k] = c.get(k, 0) + 1
        return c
    res["loop_tally"] = {nm: {"prespecified": tally(S, "prespec_class"), "post_hoc": tally(S, "posthoc_class"),
                              "capped_looping_at_100k_R50_ge_0.5": sum(1 for r in S["capped_and_terminal"] if r["finish"] == "length" and (r["R50_100k"] or 0) >= 0.5),
                              "capped": sum(1 for r in S["capped_and_terminal"] if r["finish"] == "length")}
                         for nm, S in (("rep070-F", A), ("d091", B))}
    for nm, t in res["loop_tally"].items():
        L.append(f"\n- {nm}: {t['capped']} capped; [P] {t['prespecified']}; [PH] {t['post_hoc']}; already repeating at 100k (R50 >= 0.5): {t['capped_looping_at_100k_R50_ge_0.5']}")
    # 3. recovered episodes
    L += ["", "## 3. High-repetition episodes and recovery [P]", "",
          "An episode is recovered when R50 falls below 0.5 again before the end of the reasoning; a terminal onset holds to the end by definition.", "", "| engine | set | replies | with any episode | with a recovered episode | recovered episodes |", "|---|---|---:|---:|---|---:|"]
    for nm, S in (("rep070-F", A), ("d091", B)):
        e = S["episode_counts"]
        for c in ("capped", "finished>=100k", "comparison 50-100k"):
            L.append(f"| {nm} | {c} | {e['replies'][c]} | {e['with_any_episode'][c]} | {', '.join(e['replies_with_a_recovered_episode'][c]) or '-'} | {e['recovered_episodes'][c]} |")
        L.append(f"| {nm} | finished from inside repetition (terminal onset, then finished) | | | {', '.join(e['finished_with_a_terminal_onset (finished from inside repetition)']) or '-'} | |")
    # 4. P(finish | at 100k)
    L += ["", "## 4. P(finish | still generating at 100,000 tokens)", ""]
    if a.analysis and os.path.exists(a.analysis):
        an = json.load(open(a.analysis))
        S = an.get("long_reply_finishing")
        res["P_finish_at_100k"] = S
        if S:
            L += ["| group | finished / at risk | questions | P | 95% question bootstrap |", "|---|---:|---:|---:|---|"]
            for k, g in S["groups"].items():
                x = g["100000"]
                ci = S["intervals_95_question_bootstrap"].get(k, {}).get("ci95")
                L.append(f"| {g['label']} | {x['finished']} / {x['at_risk']} | {x['questions_at_risk']} | {fmt(x['P_finish_given_at_risk'], 2)} | {ci if ci else '-'} |")
            for k, dd in S["differences_at_100000"].items():
                L.append(f"\n- {k}: {dd['estimate']} (95% joint question bootstrap {dd['ci95']}; {dd['undefined_replicates']} undefined replicates)")
            L.append(f"- {S['unit_note']}")
    else:
        L.append("(analysis result not given)")
    # 5. question overlap with the discovery panel
    dq, fq = B["at_risk_docs"], A["at_risk_docs"]
    ov = {"d091_at_risk_questions": dq, "d091_capped_questions": B["capped_docs"],
          "in_discovery_panel_ever_long": [d for d in dq if d in panel],
          "never_long_in_any_of_the_15_logged_passes": [d for d in dq if d not in panel],
          "also_at_risk_in_rep070F": [d for d in dq if d in fq],
          "long_in_stock_0.9.1_record": [d for d in dq if d in panel and panel[d].get("0.9.1", 0) > 0],
          "long_in_both_families_in_discovery": [d for d in dq if d in panel and panel[d]["fam070"] > 0 and panel[d]["fam091"] > 0],
          "discovery_both_family_questions_long_again_in_d091": [d for d in s["question_matched_finishing"]["questions"] if d in dq],
          "discovery_both_family_questions": s["question_matched_finishing"]["questions"]}
    res["question_overlap"] = ov
    L += ["", "## 5. Question overlap with the discovery's long-reply panel", "",
          f"- d091 questions past 100k: {len(dq)} ({', '.join(map(str, dq)) or '-'}); capped: {', '.join(map(str, B['capped_docs'])) or '-'}",
          f"- of them ever long in the panel's 15 logged passes: {len(ov['in_discovery_panel_ever_long'])}; never long before: {ov['never_long_in_any_of_the_15_logged_passes'] or '-'}",
          f"- also past 100k in rep070-F: {len(ov['also_at_risk_in_rep070F'])}; long in the stock 0.9.1 record: {len(ov['long_in_stock_0.9.1_record'])}; "
          f"long in both families in discovery: {len(ov['long_in_both_families_in_discovery'])}",
          f"- of discovery's {len(ov['discovery_both_family_questions'])} both-family questions, past 100k again in d091: {ov['discovery_both_family_questions_long_again_in_d091'] or '-'}",
          "", "| doc | d091: outcome past 100k | panel all (15) | 0.7.0 family (5) | 0.9.1 family (10) | stock 0.9.1 record (3) | rep070-F (3) |", "|---|---|---:|---:|---:|---:|---:|"]
    for d in dq:
        p = panel.get(d, {})
        L.append(f"| {d} | {'finished' if B['finished_at_risk'][d] else 'capped'} | {p.get('all', 0)} | {p.get('fam070', 0)} | {p.get('fam091', 0)} | {p.get('0.9.1', 0)} | {p.get('0.7.0 rep-F', 0)} |")
    L += ["", "All of the above is descriptive: one pass, a handful of long replies, seeds different from every comparator; it cannot "
          "adjudicate a termination mechanism, and reaching 100k selects different trajectories on each engine."]
    json.dump(res, open(a.out_json, "w"), indent=1, sort_keys=True, default=str)
    open(a.out_md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
