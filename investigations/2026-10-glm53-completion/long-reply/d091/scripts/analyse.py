#!/usr/bin/env python3
"""Aggregate tables for REPORT.md (host, stdlib only, zero GPU). Inputs: out/*.json(l) from build_index.py,
trajectory.py and periods.py, plus the lm-eval samples (for validity/correctness flags only; no text is written).
Writes out/tables.md and out/summary.json.
"""
import glob, importlib.util, json, math, os, random, statistics as st

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = f"{D}/out"
GPQA = "<gpqa-runs>"
FROZEN = "<runs>/rep070/refreeze/frozen/analysis_col511.py"
P = "glm53-flash-exl3-k3.25@tpurtell"
CAP = 327680

spec = importlib.util.spec_from_file_location("frozen", FROZEN)
fz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fz)

reps = {r["id"]: r for r in json.load(open(f"{OUT}/replies_all.json"))}
per = {r["id"]: r for r in json.load(open(f"{OUT}/periods_all.json"))}
wins = {}
for line in open(f"{OUT}/windows_all.jsonl"):
    w = json.loads(line)
    wins.setdefault(w["id"], []).append(w)
lengths = [json.loads(l) for l in open(f"{OUT}/lengths_all.jsonl")]
L = []  # markdown lines


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def f(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def cls_set(r):
    if r["finish_reason"] == "length":
        return "capped"
    return "finished>=100k" if r["T"] >= 100000 else "comparison 50-100k"


# ---------------------------------------------------------------- validity flags (no text out)
valid = {}
for p in (1, 2, 3):
    fs = glob.glob(f"{GPQA}/{P}+rep070-F__pass{p}/**/samples_*.jsonl", recursive=True)
    for line in open(fs[0]):
        r = json.loads(line)
        if r["filter"] != "flexible-extract":
            continue
        txt = str(r["resps"][0][0]) if r["resps"] else ""
        fa = fz.final_answer(txt) if txt.strip() else None
        valid[f"q{r['doc_id']}p{p}"] = {"nonempty": bool(txt.strip()), "letter": fa is not None,
                                        "correct_stated": fa is not None and fa == r["target"].strip("()")}

# ---------------------------------------------------------------- Q1 group metrics (reply = unit)
METRICS = ["cpt", "rep20", "rep50", "rep100", "zr", "dlr", "line_seen", "cjk", "nonascii", "alpha", "digit", "space",
           "punct", "tok_le1char", "chk_verify_per1k", "chk_verify_rep", "chk_wait_per1k", "chk_wait_rep"]


def reply_medians(sel_windows):
    return {m: med([w[m] for w in sel_windows if not w["partial"]]) for m in METRICS}


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
L.append("### T1. Per-window metrics by set (median over replies of each reply's median over full 4k windows)\n")
show = ["cpt", "rep20", "rep50", "rep100", "zr", "dlr", "line_seen", "cjk", "digit", "punct", "space", "tok_le1char",
        "chk_verify_per1k", "chk_verify_rep", "chk_wait_per1k", "chk_wait_rep"]
L.append("| set | replies | " + " | ".join(show) + " |")
L.append("|---|---:|" + "---:|" * len(show))
summary = {"group_medians": {}}
for g, rows in groups.items():
    vals = {m: med([x[m] for x in rows]) for m in show}
    summary["group_medians"][g] = {"n": len(rows), **vals}
    L.append(f"| {g} | {len(rows)} | " + " | ".join(f(vals[m]) for m in show) + " |")

# token-weighted cpt decomposition for capped replies
L.append("\n### T2. Characters per token: whole reply, before onset and after onset (token-weighted)\n")
L.append("| id | T | outcome | cpt whole | cpt before onset (or whole) | cpt after onset | cpt first 100k | digit share first 100k |")
L.append("|---|---:|---|---:|---:|---:|---:|---:|")


def tw_cpt(ws):
    tok = sum(w["g1"] - w["g0"] for w in ws)
    ch = sum(w["cpt"] * (w["g1"] - w["g0"]) for w in ws)
    return ch / tok if tok else None


def tw(ws, m):
    tok = sum(w["g1"] - w["g0"] for w in ws if w[m] is not None)
    return sum(w[m] * (w["g1"] - w["g0"]) for w in ws if w[m] is not None) / tok if tok else None


cpt_rows = []
for i, r in sorted(reps.items(), key=lambda kv: (kv[1]["finish_reason"] != "length", -kv[1]["T"])):
    if r["T"] < 100000:
        continue
    ws = wins[i]
    o = r["onset"]
    pre = [w for w in ws if o is None or w["g1"] <= o]
    post = [w for w in ws if o is not None and w["g0"] >= o]
    first = [w for w in ws if w["g1"] <= 100000]
    row = {"id": i, "T": r["T"], "out": "capped" if r["finish_reason"] == "length" else "finished",
           "whole": r["overall"]["cpt"], "pre": tw_cpt(pre), "post": tw_cpt(post) if post else None,
           "first100k": tw_cpt(first), "digit100k": tw(first, "digit")}
    cpt_rows.append(row)
    L.append(f"| {i} | {r['T']} | {row['out']} | {f(row['whole'],2)} | {f(row['pre'],2)} | {f(row['post'],2)} | {f(row['first100k'],2)} | {f(row['digit100k'],3)} |")
summary["cpt_rows"] = cpt_rows
for out in ("capped", "finished"):
    rr = [x for x in cpt_rows if x["out"] == out]
    summary[f"cpt_{out}"] = {"whole_median": med([x["whole"] for x in rr]), "pre_median": med([x["pre"] for x in rr]),
                             "first100k_median": med([x["first100k"] for x in rr]),
                             "whole_pooled": sum(reps[x["id"]]["reasoning_chars"] for x in rr) / sum(reps[x["id"]]["N_reasoning"] for x in rr)}

# ---------------------------------------------------------------- Q2 capped classification
L.append("\n### T3. Capped replies (and finished replies with a terminal onset): onset and loop type\n")
L.append("| id | T | outcome | onset g (R50>=0.5) | onset g @0.3 / @0.7 | onset abs | R50 at 100k | prespec class | E | LRS after onset (tokens) | LRS start g | LRS offset = period | exact lag match m(period) | post hoc class | episodes (recovered) | R50 final / max |")
L.append("|---|---:|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|")
cap_rows = []
for i, r in sorted(reps.items(), key=lambda kv: kv[1]["T"]):
    if not (r["finish_reason"] == "length" or r["onset"] is not None):
        continue
    pr = per[i]
    lpo = pr.get("lrs_post_onset") or {}
    lag = pr.get("lag_post_onset") or {}
    mp = None
    if lag and lpo.get("offset"):
        mp = dict((p_, v) for p_, v in lag["top_lags"]).get(lpo["offset"])
    # post hoc class: exact loop if post-onset LRS covers >= 50% of the post-onset region
    region = r["N_reasoning"] - r["onset"] if r["onset"] is not None else None
    if r["onset"] is None:
        ph = "no terminal loop"
    elif lpo.get("len", 0) >= 0.5 * region:
        ph = f"exact loop, period {lpo['offset']}" + (" (short)" if lpo["offset"] <= 4096 else " (long)")
    else:
        ph = f"partial / fragmentary (LRS {lpo.get('len')} of {region})"
    s3, s7 = r["sensitivity"]["0.3"]["onset"], r["sensitivity"]["0.7"]["onset"]
    eps = r["episodes"]
    row = {"id": i, "T": r["T"], "finish": r["finish_reason"], "onset": r["onset"], "onset_abs": r["onset_abs"],
           "onset_03": s3, "onset_07": s7, "R50_100k": r["R50_at"]["100000"], "prespec_class": r["loop"]["class"],
           "E": r["loop"].get("E"), "lrs_post": lpo.get("len"), "lrs_start": lpo.get("start_a"),
           "period": lpo.get("offset"), "m_period": mp, "posthoc_class": ph,
           "episodes": len(eps), "recovered": sum(not e["terminal"] for e in eps), "R50_final": r["R50_final"],
           "R50_max": r["R50_max"], "prompt_tokens": r["prompt_tokens"], "doc": r["doc"]}
    cap_rows.append(row)
    L.append(f"| {i} | {r['T']} | {r['finish_reason']} | {f(r['onset'])} | {f(s3)} / {f(s7)} | {f(r['onset_abs'])} | {f(row['R50_100k'],3)} | {row['prespec_class']} | {f(row['E'],2)} | {f(row['lrs_post'])} | {f(row['lrs_start'])} | {f(row['period'])} | {f(mp,3)} | {ph} | {len(eps)} ({row['recovered']}) | {f(r['R50_final'],2)} / {f(r['R50_max'],2)} |")
summary["capped_and_terminal"] = cap_rows

# episodes across all analysed replies
L.append("\n### T4. High-repetition episodes in every analysed reply (R50 >= 0.5 for >= 2,048 positions)\n")
L.append("| id | T | outcome | episodes | recovered | terminal | episode spans (g start-end) |")
L.append("|---|---:|---|---:|---:|---:|---|")
ep_all = []
for i, r in sorted(reps.items(), key=lambda kv: kv[1]["T"]):
    if not r["episodes"]:
        continue
    for e in r["episodes"]:
        ep_all.append({"id": i, "start": e["start"], "end": e["end"], "terminal": e["terminal"],
                       "abs_start": e["start"] + r["prompt_tokens"], "finish": r["finish_reason"]})
    sp = ", ".join(f"{e['start']}-{e['end']}" for e in r["episodes"])
    L.append(f"| {i} | {r['T']} | {r['finish_reason']} | {len(r['episodes'])} | {sum(not e['terminal'] for e in r['episodes'])} | {sum(e['terminal'] for e in r['episodes'])} | {sp} |")
n_any = {c: sum(1 for r in reps.values() if cls_set(r) == c and r["episodes"]) for c in ("capped", "finished>=100k", "comparison 50-100k")}
n_tot = {c: sum(1 for r in reps.values() if cls_set(r) == c) for c in ("capped", "finished>=100k", "comparison 50-100k")}
summary["episode_counts"] = {"with_any_episode": n_any, "replies": n_tot}
L.append(f"\nReplies with any episode: capped {n_any['capped']}/{n_tot['capped']}, finished >=100k {n_any['finished>=100k']}/{n_tot['finished>=100k']}, comparison {n_any['comparison 50-100k']}/{n_tot['comparison 50-100k']}.")

# ---------------------------------------------------------------- Q3 finished long replies
L.append("\n### T5. Finished replies >= 100k: what precedes termination\n")
L.append("| id | T | reasoning tokens | content tokens / chars | valid letter | stated correct | last rep50>=0.5 window: distance to reasoning end | same at 0.25 | episodes (recovered) | R50 max | R50 final | last-window rep50 / chk_verify per 1k vs reply median |")
L.append("|---|---:|---:|---|---|---|---:|---:|---|---:|---:|---|")
fin_rows = []
for i, r in sorted(reps.items(), key=lambda kv: kv[1]["T"]):
    if r["finish_reason"] != "stop" or r["T"] < 100000:
        continue
    ws = [w for w in wins[i] if not w["partial"]]
    last = ws[-1]
    v = valid[i]
    h5, h25 = r["last_hiwin_0.5"], r["last_hiwin_0.25"]
    row = {"id": i, "T": r["T"], "N": r["N_reasoning"], "content_tokens": r["content_tokens"], "content_chars": r["content_chars"],
           "valid": v["letter"] and v["nonempty"], "correct_stated": v["correct_stated"],
           "dist05": h5["dist_to_reasoning_end"] if h5 else None, "dist025": h25["dist_to_reasoning_end"] if h25 else None,
           "episodes": len(r["episodes"]), "recovered": sum(not e["terminal"] for e in r["episodes"]),
           "R50_max": r["R50_max"], "R50_final": r["R50_final"], "last_rep50": last["rep50"],
           "med_rep50": med([w["rep50"] for w in ws]), "last_chk": last["chk_verify_per1k"],
           "med_chk": med([w["chk_verify_per1k"] for w in ws])}
    fin_rows.append(row)
    L.append(f"| {i} | {r['T']} | {r['N_reasoning']} | {r['content_tokens']} / {r['content_chars']} | {'yes' if row['valid'] else 'no'} | {'yes' if row['correct_stated'] else 'no'} | {f(row['dist05'])} | {f(row['dist025'])} | {row['episodes']} ({row['recovered']}) | {f(r['R50_max'],3)} | {f(r['R50_final'],3)} | {f(row['last_rep50'],3)} / {f(row['last_chk'],2)} vs {f(row['med_rep50'],3)} / {f(row['med_chk'],2)} |")
summary["finished_long"] = fin_rows
summary["finished_long_content_tokens"] = {"median": med([x["content_tokens"] for x in fin_rows]),
                                           "min": min(x["content_tokens"] for x in fin_rows), "max": max(x["content_tokens"] for x in fin_rows)}
cmpc = [r["content_tokens"] for r in reps.values() if cls_set(r) == "comparison 50-100k"]
summary["comparison_content_tokens"] = {"median": med(cmpc), "min": min(cmpc), "max": max(cmpc)}
summary["finished_long_valid"] = sum(x["valid"] for x in fin_rows)

# ---------------------------------------------------------------- Q4 geometry
L.append("\n### T6. Onset positions against engine geometry (absolute positions = prompt + generated)\n")
L.append("| event | id | abs pos | mod 3,584 | dist. to nearest 3,584k | dist. to nearest 2,043k | mod 4 | nearest of 65,536/131,072/262,144 (dist.) | onset spread @0.3-0.7 |")
L.append("|---|---|---:|---:|---:|---:|---:|---|---:|")
events = []
for row in cap_rows:
    if row["onset"] is not None:
        events.append(("terminal onset (R50)", row["id"], row["onset_abs"], (row["onset_07"] or 0) - (row["onset_03"] or 0)))
    if row["lrs_start"] is not None:
        events.append(("exact-loop start (LRS, post hoc)", row["id"], row["lrs_start"] + row["prompt_tokens"], None))
for e in ep_all:
    if not e["terminal"]:
        events.append(("recovered-episode start", e["id"], e["abs_start"], None))
geo = []
for kind, i, a, spread in events:
    m3584 = a % 3584
    d3584 = min(m3584, 3584 - m3584)
    m2043 = a % 2043
    d2043 = min(m2043, 2043 - m2043)
    p2 = min((abs(a - x), x) for x in (65536, 131072, 262144))
    geo.append({"kind": kind, "id": i, "abs": a, "mod3584": m3584, "d3584": d3584, "d2043": d2043, "mod4": a % 4,
                "nearest_pow2": p2[1], "d_pow2": p2[0], "spread": spread})
    L.append(f"| {kind} | {i} | {a} | {m3584} | {d3584} | {d2043} | {a % 4} | {p2[1]} ({p2[0]}) | {f(spread)} |")


def rayleigh(phases, period, draws=20000, seed=1):
    n = len(phases)
    if n == 0:
        return None, None
    c = sum(math.cos(2 * math.pi * x / period) for x in phases) / n
    s = sum(math.sin(2 * math.pi * x / period) for x in phases) / n
    R = math.hypot(c, s)
    rng = random.Random(seed)
    ge = 0
    for _ in range(draws):
        cc = ss = 0.0
        for _ in range(n):
            t = rng.random() * 2 * math.pi
            cc += math.cos(t)
            ss += math.sin(t)
        if math.hypot(cc / n, ss / n) >= R:
            ge += 1
    return R, (ge + 1) / (draws + 1)


geo_stats = {}
for kind in ("terminal onset (R50)", "exact-loop start (LRS, post hoc)", "all events"):
    xs = [g["abs"] for g in geo if kind == "all events" or g["kind"] == kind]
    R3584, p3584 = rayleigh([x % 3584 for x in xs], 3584)
    R4, p4 = rayleigh([x % 4 for x in xs], 4)
    geo_stats[kind] = {"n": len(xs), "R_mod3584": R3584, "p_mod3584": p3584, "R_mod4": R4, "p_mod4": p4,
                       "median_d_pow2": med([g["d_pow2"] for g in geo if kind == "all events" or g["kind"] == kind])}
L.append("\nUniformity (descriptive): mean resultant length R of the phase, Monte Carlo p under uniform phase.\n")
L.append("| events | n | R (mod 3,584) | p | R (mod 4) | p | median distance to nearest power of two |")
L.append("|---|---:|---:|---:|---:|---:|---:|")
for k, v in geo_stats.items():
    L.append(f"| {k} | {v['n']} | {f(v['R_mod3584'],2)} | {f(v['p_mod3584'],2)} | {f(v['R_mod4'],2)} | {f(v['p_mod4'],2)} | {f(v['median_d_pow2'])} |")
summary["geometry"] = {"events": geo, "uniformity": geo_stats}

# ---------------------------------------------------------------- Q5 panel
ENG = [("0.7.0 orig", "stock 0.7.0 (orig)"), ("0.7.0 rep-F", "stock 0.7.0 rep070-F"), ("0.9.1", "stock 0.9.1"),
       ("col511", "col511 arm"), ("B0", "B0"), ("B1", "B1")]
npass = {g: len({r["seed"] for r in lengths if r["group"] == g}) for _, g in ENG}
long_ = {}
capd = {}
for r in lengths:
    k = (r["group"], r["doc"])
    long_[k] = long_.get(k, 0) + (r["T"] >= 100000)
    capd[k] = capd.get(k, 0) + bool(r["capped"])
docs = sorted({r["doc"] for r in lengths if r["T"] >= 100000})
tot = {d: sum(long_.get((g, d), 0) for _, g in ENG) for d in docs}
fam070 = {d: long_.get(("stock 0.7.0 (orig)", d), 0) + long_.get(("stock 0.7.0 rep070-F", d), 0) for d in range(198)}
fam091 = {d: sum(long_.get((g, d), 0) for g in ("stock 0.9.1", "col511 arm", "B0", "B1")) for d in range(198)}
# unlogged empties (capped proxy): stock 0.7.0 pass 1 and 0.7.0 kpoolfix pass 1
unl = {}
for name, path in (("0.7.0 orig pass1 (unlogged)", f"{GPQA}/{P}__pass1"), ("0.7.0 kpoolfix pass1 (unlogged)", f"{GPQA}/{P}+kpoolring__pass1")):
    fs = glob.glob(f"{path}/**/samples_*.jsonl", recursive=True)
    e = set()
    for line in open(fs[0]):
        r = json.loads(line)
        if r["filter"] == "flexible-extract" and not (str(r["resps"][0][0]) if r["resps"] else "").strip():
            e.add(r["doc_id"])
    unl[name] = sorted(e)
L.append("\n### T7. Long-reply panel: passes with T >= 100,000 per question (capped in brackets), by configuration\n")
hdr = " | ".join(f"{n} ({npass[g]})" for n, g in ENG)
L.append(f"| doc | {hdr} | 0.7.0 family (5) | 0.9.1 family (10) | all (15) | empty in unlogged 0.7.0 passes |")
L.append("|---|" + "---:|" * (len(ENG) + 3) + "---|")
panel = []
for d in sorted(docs, key=lambda d: (-tot[d], d)):
    cells = [f"{long_.get((g, d), 0)} [{capd.get((g, d), 0)}]" for _, g in ENG]
    em = ", ".join(n.split(" (")[0] for n, s in unl.items() if d in s) or ""
    panel.append({"doc": d, **{n: long_.get((g, d), 0) for n, g in ENG}, **{n + " capped": capd.get((g, d), 0) for n, g in ENG},
                  "fam070": fam070[d], "fam091": fam091[d], "all": tot[d], "unlogged_empty": em})
    L.append(f"| {d} | " + " | ".join(cells) + f" | {fam070[d]} | {fam091[d]} | {tot[d]} | {em} |")
summary["panel"] = panel
summary["unlogged_empties"] = unl
# concentration and concordance
ev = sorted(tot.values(), reverse=True)
alln = sum(ev)
top5, top10 = sum(ev[:5]), sum(ev[:10])
summary["concentration"] = {"questions_ever_long": len(docs), "long_events": alln, "top5_share": top5 / alln,
                            "top10_share": top10 / alln, "questions_with_1_event": sum(1 for v in ev if v == 1)}


def spearman(a, b):
    def rank(x):
        o = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else None


a = [fam070[d] / 5 for d in range(198)]
b = [fam091[d] / 10 for d in range(198)]
repF = {d: long_.get(("stock 0.7.0 rep070-F", d), 0) for d in range(198)}
orig = {d: long_.get(("stock 0.7.0 (orig)", d), 0) for d in range(198)}
st091 = {d: long_.get(("stock 0.9.1", d), 0) for d in range(198)}
atrisk_q = [d for d in range(198) if repF[d] > 0]
summary["concordance"] = {
    "spearman_070fam_vs_091fam_all198": spearman(a, b),
    "spearman_repF_vs_orig070_all198": spearman([repF[d] for d in range(198)], [orig[d] for d in range(198)]),
    "spearman_repF_vs_stock091_all198": spearman([repF[d] for d in range(198)], [st091[d] for d in range(198)]),
    "repF_atrisk_questions": len(atrisk_q),
    "of_which_long_in_orig070": sum(orig[d] > 0 for d in atrisk_q),
    "of_which_long_in_stock091": sum(st091[d] > 0 for d in atrisk_q),
    "of_which_long_in_any_091fam": sum(fam091[d] > 0 for d in atrisk_q),
    "questions_long_in_091fam_never_in_070fam": sorted(d for d in range(198) if fam091[d] > 0 and fam070[d] == 0),
    "questions_long_in_070fam_never_in_091fam": sorted(d for d in range(198) if fam070[d] > 0 and fam091[d] == 0),
    "base_rate_long_per_request": {n: sum(long_.get((g, d), 0) for d in range(198)) / (198 * npass[g]) for n, g in ENG}}
# expected overlap under independence (same per-question marginal), for reading the overlap
L.append("\nConcentration and concordance: see summary.json `concentration`, `concordance`.")

# ---------------------------------------------------------------- Q6 cost inputs
rates = {}
for n, g in ENG:
    xs = [r["T"] / r["secs"] for r in lengths if r["group"] == g and r["T"] >= 100000 and r["secs"]]
    rates[n] = {"n": len(xs), "median_tok_s_per_request": med(xs), "min": min(xs) if xs else None, "max": max(xs) if xs else None}
summary["decode_rates_long_requests"] = rates
# P(finish within +H | at risk at 100k) in rep070-F, for horizon choices
atr = [r for r in lengths if r["group"] == "stock 0.7.0 rep070-F" and r["T"] >= 100000]
summary["repF_finish_within_horizon_from_100k"] = {str(h): sum(1 for r in atr if r["finish_reason"] == "stop" and r["T"] < 100000 + h) / len(atr)
                                                    for h in (32000, 64000, 100000, 227680)}
summary["repF_R50_at_100k_atrisk"] = {i: reps[i]["R50_at"]["100000"] for i in reps if reps[i]["T"] >= 100000}
summary["drift"] = {"min": min(r["drift"] for r in reps.values()), "max": max(r["drift"] for r in reps.values()),
                    "hash_verify_failures": sum(r["hash_verify_failures"] for r in reps.values())}

# ---------------------------------------------------------------- extra (post hoc, descriptive)
# (a) characters-per-token proxy for every rep070-F request: (reasoning + content chars) / completion_tokens
idx = json.load(open(f"{OUT}/rep070F_index.json"))
def proxy(x):
    return (x["reasoning_chars"] + x["content_chars"]) / x["T"]
bins = [(0, 10000), (10000, 25000), (25000, 50000), (50000, 100000), (100000, CAP), (CAP, CAP + 1)]
L.append("\n### T8. Characters per completion token, every rep070-F request (proxy: (reasoning + content chars) / T)\n")
L.append("| T band | requests | median | IQR |")
L.append("|---|---:|---:|---|")
cptb = {}
for lo, hi in bins:
    xs = sorted(proxy(x) for x in idx if lo <= x["T"] < hi)
    q = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]
    name = "capped (327,680)" if lo == CAP else f"{lo:,}-{hi:,}" if hi != CAP else f"{lo:,}-cap (finished)"
    cptb[name] = {"n": len(xs), "median": med(xs), "q25": q(0.25), "q75": q(0.75)}
    L.append(f"| {name} | {len(xs)} | {f(med(xs),2)} | {f(q(0.25),2)}-{f(q(0.75),2)} |")
summary["cpt_proxy_by_band"] = cptb
# within question: questions with a capped rep070-F pass
L.append("\n### T9. Within question: rep070-F passes of every question with a capped pass (cpt proxy over the whole reply; first-100k cpt where tokenized)\n")
L.append("| doc | pass | T | outcome | cpt proxy | cpt first 100k (re-tokenized) | digit share first 100k |")
L.append("|---|---:|---:|---|---:|---:|---:|")
capdocs = sorted({x["doc"] for x in idx if x["finish_reason"] == "length"})
wq = []
for d in capdocs:
    for x in sorted((x for x in idx if x["doc"] == d), key=lambda x: x["pass"]):
        i = x["id"]
        c100 = tw_cpt([w for w in wins[i] if w["g1"] <= 100000]) if i in wins else None
        dg = tw([w for w in wins[i] if w["g1"] <= 100000], "digit") if i in wins else None
        wq.append({"id": i, "doc": d, "T": x["T"], "finish": x["finish_reason"], "cpt_proxy": proxy(x), "cpt100k": c100, "digit100k": dg})
        L.append(f"| {d} | {x['pass']} | {x['T']} | {x['finish_reason']} | {f(proxy(x),2)} | {f(c100,2)} | {f(dg,3)} |")
summary["within_question_cpt"] = wq

# (b) precursors: the 32k tokens before a terminal onset against the reply's earlier windows
L.append("\n### T10. Before the onset: last 32k tokens before a terminal onset against earlier windows (token-weighted)\n")
L.append("| id | onset g | rep20 earlier / last 32k | rep50 earlier / last 32k | chk_wait per 1k earlier / last 32k | chk_verify per 1k earlier / last 32k | cpt earlier / last 32k |")
L.append("|---|---:|---|---|---|---|---|")
prec = []
for row in cap_rows:
    o = row["onset"]
    if o is None or o < 40000:
        continue
    ws = wins[row["id"]]
    a = [w for w in ws if w["g1"] <= o - 32768]
    b = [w for w in ws if w["g0"] >= o - 32768 and w["g1"] <= o]
    rr = {m: (tw(a, m), tw(b, m)) for m in ("rep20", "rep50", "chk_wait_per1k", "chk_verify_per1k", "cpt")}
    prec.append({"id": row["id"], "onset": o, **{m: v for m, v in rr.items()}})
    L.append(f"| {row['id']} | {o} | " + " | ".join(f"{f(rr[m][0],3)} / {f(rr[m][1],3)}" for m in ("rep20", "rep50", "chk_wait_per1k", "chk_verify_per1k", "cpt")) + " |")
summary["precursors"] = prec

# (c) question-matched finishing given >= 100k, by family (questions long in both families)
L.append("\n### T11. Finishing among replies >= 100k, on questions long in both families (question as stratum)\n")
L.append("| doc | 0.7.0 family: finished / long | 0.9.1 family: finished / long |")
L.append("|---|---|---|")
fin = {}
for r in lengths:
    if r["T"] >= 100000:
        fam = "070" if r["group"].startswith("stock 0.7.0") else "091"
        k = (fam, r["doc"])
        a_, b_ = fin.get(k, (0, 0))
        fin[k] = (a_ + (not r["capped"]), b_ + 1)
both = sorted(d for d in range(198) if ("070", d) in fin and ("091", d) in fin)
s070 = [0, 0]; s091 = [0, 0]; qhi = qlo = qeq = 0
for d in both:
    a_, b_ = fin[("070", d)]; c_, e_ = fin[("091", d)]
    s070[0] += a_; s070[1] += b_; s091[0] += c_; s091[1] += e_
    if a_ / b_ > c_ / e_: qhi += 1
    elif a_ / b_ < c_ / e_: qlo += 1
    else: qeq += 1
    L.append(f"| {d} | {a_} / {b_} | {c_} / {e_} |")
L.append(f"| **pooled ({len(both)} questions)** | **{s070[0]} / {s070[1]}** | **{s091[0]} / {s091[1]}** |")
L.append(f"\nQuestions where the 0.7.0 family finishes a larger share: {qhi}; smaller: {qlo}; equal: {qeq}.")
summary["question_matched_finishing"] = {"questions": both, "070": s070, "091": s091, "070_higher": qhi, "070_lower": qlo, "equal": qeq,
                                         "per_question": {d: {"070": fin[("070", d)], "091": fin[("091", d)]} for d in both}}

# (d) at 100k: what separates capped from finished among the 29 at risk (rank AUC, descriptive)
def auc(pos, neg):
    s = 0.0
    for x in pos:
        for y in neg:
            s += 1.0 if x > y else 0.5 if x == y else 0.0
    return s / (len(pos) * len(neg))
atr_ids = [i for i in reps if reps[i]["T"] >= 100000]
feat = {}
for i in atr_ids:
    ws = [w for w in wins[i] if w["g1"] <= 100000]
    feat[i] = {"cpt": tw_cpt(ws), "digit": tw(ws, "digit"), "chk_wait": tw(ws, "chk_wait_per1k"), "chk_verify": tw(ws, "chk_verify_per1k"),
               "rep20": tw(ws, "rep20"), "R50_100k": reps[i]["R50_at"]["100000"], "capped": reps[i]["finish_reason"] == "length"}
L.append("\n### T12. First 100k tokens of the 29 at-risk replies: rank separation of capped (10) from finished (19)\n")
L.append("AUC = P(capped value > finished value); 0.5 = no separation. Replies are not independent (q79 contributes 3 capped).\n")
L.append("| feature (first 100k) | capped median | finished median | AUC |")
L.append("|---|---:|---:|---:|")
aucs = {}
for m in ("cpt", "digit", "chk_wait", "chk_verify", "rep20", "R50_100k"):
    pos = [feat[i][m] for i in atr_ids if feat[i]["capped"] and feat[i][m] is not None]
    neg = [feat[i][m] for i in atr_ids if not feat[i]["capped"] and feat[i][m] is not None]
    aucs[m] = {"capped_median": med(pos), "finished_median": med(neg), "auc": auc(pos, neg)}
    L.append(f"| {m} | {f(med(pos),3)} | {f(med(neg),3)} | {f(aucs[m]['auc'],2)} (n {len(pos)}/{len(neg)}) |")
summary["at100k_separation"] = aucs

# (e) characters per token of capped and finished-long replies on every logged engine (request-log char counts)
L.append("\n### T13. Characters per completion token by engine (request-log character counts / completion_tokens)\n")
L.append("| configuration | capped: n, median [min-max] | capped with cpt >= 3.0 | finished >= 100k: n, median | 50-100k: median |")
L.append("|---|---|---:|---|---:|")
eng_cpt = {}
for n, g in ENG:
    rows = [r for r in lengths if r["group"] == g and r["reasoning_chars"] is not None]
    cp = lambda r: ((r["reasoning_chars"] or 0) + (r["content_chars"] or 0)) / r["T"]
    c = sorted(cp(r) for r in rows if r["capped"])
    fl = [cp(r) for r in rows if not r["capped"] and r["T"] >= 100000]
    mid = [cp(r) for r in rows if 50000 <= r["T"] < 100000]
    eng_cpt[n] = {"capped": c, "capped_ge3": sum(x >= 3.0 for x in c), "fin_long_median": med(fl), "n_fin_long": len(fl), "mid_median": med(mid)}
    L.append(f"| {n} | {len(c)}, {f(med(c),2)} [{f(c[0],2)}-{f(c[-1],2)}] | {eng_cpt[n]['capped_ge3']} | {len(fl)}, {f(med(fl),2)} | {f(med(mid),2)} |")
summary["engine_cpt"] = eng_cpt

# (f) cost model for continuation designs (upper bound: every continuation runs its full horizon)
KV = {"0.7.0": 2758919, "0.9.1": 4707515, "B1": 4707515}
RATE = {"0.7.0": (rates["0.7.0 rep-F"]["median_tok_s_per_request"], rates["0.7.0 rep-F"]["min"]),
        "0.9.1": (rates["0.9.1"]["median_tok_s_per_request"], rates["0.9.1"]["min"]),
        "B1": (rates["B1"]["median_tok_s_per_request"], rates["B1"]["min"])}
START_H, PREFILL_TPS, GPUS, PREFIX = 13.5 / 60, 2410, 2, 100000
def cost(K, S, H, engines=("0.7.0", "0.9.1", "B1"), extra_h=0.0):
    out = {}
    for e in engines:
        c = min(16, KV[e] // (PREFIX + H) if H < CAP - PREFIX else KV[e] // CAP)
        waves = math.ceil(K * S / c)
        res = []
        for rate in RATE[e]:
            wall = START_H + K * PREFIX / PREFILL_TPS / 3600 + waves * H / rate / 3600 + extra_h
            res.append(round(GPUS * wall, 1))
        out[e] = {"concurrency": c, "waves": waves, "gpu_h_central": res[0], "gpu_h_pessimistic": res[1]}
    out["total_central"] = round(sum(v["gpu_h_central"] for v in out.values()), 1)
    out["total_pessimistic"] = round(sum(v["gpu_h_pessimistic"] for k, v in out.items() if k != "total_central"), 1)
    return out
designs = {
    "A: 8 prefixes x 4 seeds, continue 100k -> cap (227,680 tokens), 0.7.0 + 0.9.1 + B1": cost(8, 4, CAP - PREFIX),
    "B: 8 prefixes x 4 seeds, continue +64,000 tokens, 0.7.0 + 0.9.1 + B1": cost(8, 4, 64000),
    "B+: 12 prefixes x 4 seeds, continue +64,000 tokens, 0.7.0 + 0.9.1 + B1": cost(12, 4, 64000),
    "C: B1 only, 8 prefixes forced through decode (hook, oracle schedule) vs prefill-built, 4 seeds each, +64,000": cost(8, 8, 64000, engines=("B1",), extra_h=PREFIX / 4 * 0.062 / 3600),
}
summary["cost_designs"] = designs
L.append("\n### T14. Rough cost of continuation designs (GPU-hours, 2 GPUs; upper bound - every continuation runs its full horizon)\n")
L.append("Inputs: per-request decode rate on replies >= 100k (median / slowest observed, T14 note), KV-limited concurrency, 13.5 min start, 2,410 tok/s prefill.\n")
L.append("| design | 0.7.0 (conc., waves, GPU-h central / pess.) | 0.9.1 | B1 | total central / pessimistic |")
L.append("|---|---|---|---|---|")
for k, v in designs.items():
    cell = lambda e: f"{v[e]['concurrency']}, {v[e]['waves']}, {v[e]['gpu_h_central']} / {v[e]['gpu_h_pessimistic']}" if e in v else "–"
    L.append(f"| {k} | {cell('0.7.0')} | {cell('0.9.1')} | {cell('B1')} | {v['total_central']} / {v['total_pessimistic']} |")

open(f"{OUT}/tables.md", "w").write("\n".join(L) + "\n")
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=str)
print("\n".join(L))
print(json.dumps({k: summary[k] for k in ("concentration", "concordance", "decode_rates_long_requests",
                                            "repF_finish_within_horizon_from_100k", "drift", "cpt_capped", "cpt_finished",
                                            "finished_long_content_tokens", "comparison_content_tokens", "finished_long_valid",
                                            "episode_counts")}, indent=1, default=str))
