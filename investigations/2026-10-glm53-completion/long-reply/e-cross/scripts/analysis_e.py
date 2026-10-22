#!/usr/bin/env python3
"""E frozen analysis (PROPOSAL.md sections 5-6): the prefix-source x engine cross. CPU-only container (numpy; trajectory.py
needs tokenizers), zero GPU. Run through scripts/run_cpu.sh with RUN_DIR mounted at /run (read-only):
    RUN_DIR=<run folder> scripts/run_cpu.sh v070 scripts/analysis_e.py /work/<out>.json > <out>.md
Writes ids, seeds, counts, rates, positions and hashes only: no question, answer or reply text, no token ids.

Reuse, byte for byte (hash-checked below):
  analysis_bplus_22v.py  B+'s amended analysis (sha256 ed915a07...). Its gates(), load(), raw_records(), loop_metrics() and
                         spearman() are called UNCHANGED. Only module-level configuration data is set here, each item
                         listed in CONFIG_OVERRIDES: the engine list, and (amendment 22ah, as D under 22ad) the KV pool:
                         reported, not identity-gated, with a functional floor of 8 requests at the 327,680 cap. Before
                         22ah the v070 KV-pool accepted set was widened to every
                         size the stock 0.7.0 image's own sessions produced (checks/gate-calibration.txt). Every other
                         reference value (non-default args, vLLM version, snapshots, environment names, CUDA-graph mode,
                         warm-up counts, image ids) is B+'s, unchanged, and passes on every reference session.
  bplus_stats.py         B+'s frozen statistics: contrast() (EFFECT / REVERSE / SMALL screens, bootstrap 20,000 draws seed
                         20261016, exact sign-flip), eta2(), icc1(), boot_ci(), unchanged.
New here: probe() for two engines, the reading rule (B+'s frozen C1-ONLY rule applied to E1), and the descriptive cross.

Order of work (never classify a partial or invalid run):
  1. probe     COMPLETE = both sessions ran to the end; anything else (including a v091 budget stop) = INCOMPLETE.
  2. gates     analysis_bplus_22v.gates() per engine, plus seeds distinct across the run. Any failure = INVALID.
  3. primary   per engine P(finish within +64k) on the 0.9.1-built prefixes, prefix as the unit (13 prefixes x 4 seeds; 22ah);
               E1 = v070 - v091 paired by prefix with B+'s contrast(); reading R-E / R-E- / R-I (C1-ONLY rule).
  4. cross     descriptive, prespecified: source (0.7.0-built B+ prefixes, ref/bplus-result-22v.json; 0.9.1-built E
               prefixes) x engine (v070, v091) cells; source contrast pooled over engines (confounded with question);
               question-matched sub-table; interaction E1 - C1; the original-outcome stratum.
  5. secondary as B+: any stop, loop onsets and types, chars/token, ICC, within-prefix agreement, covariates, cache hits,
               decode rates.
"""
import hashlib
import itertools
import json
import os
import statistics
import sys

import numpy as np

WORK = os.environ.get("BPLUS_WORK", "/work")
RUN = os.environ.get("BPLUS_RUN", "/run")
sys.path.insert(0, f"{WORK}/scripts")
FROZEN = {"analysis_bplus_22v.py": "ed915a071e0faa9ca6b842a8895db13dde23d4e365ecaa7e420bf121a20da6da",
          "bplus_stats.py": "68e4bc45bc8196fdd893bd65f8bca3b3a22c68e0395c7302dfff3994225e94a6",
          "trajectory.py": "c24267b1a9641fc2a3bf4b9bca23950d9530d787002d74286bef8b717f5ac4c6",
          "analysis_col511.py": "b5b13ce2155085ca46fb24877f12b7cd0a5cdd16eb4cf19012958e7b86ae0033"}
for _f, _s in FROZEN.items():
    if hashlib.sha256(open(f"{WORK}/scripts/{_f}", "rb").read()).hexdigest() != _s:
        sys.exit(f"{_f} is not the frozen file")
REF_FILES = {"ref/bplus-result-22v.json": "513f0bfc43cbf0fae0837c8e89d17ba8204c9e7e19d55e5a52395ab8c1118518",
             "ref/bplus-prefix_manifest.json": "26f747ffd2eeeeec3cba2f1c2f5854a0c707f64e24bc694106c4dc1860cc3bbc"}
import analysis_bplus_22v as A  # noqa: E402
import bplus_stats as BS  # noqa: E402

ENGINES = ("v070", "v091")
LABEL = {"v070": "stock 0.7.0", "v091": "stock 0.9.1"}
# Configuration data set on the unchanged B+ module (the only differences from B+'s 22v analysis; PROPOSAL.md section 6):
# Amendment 22ah (KV pool as D, amendment 22ad): REPORTED, not identity-gated; gated only on a functional floor, D's formula
# (d091 analysis_d091.py kv_floor): 8 requests x attention block x ceil((cap 327,680 + longest record prompt 2,795 + 16) /
# block), with each engine's own attention block (server logs: 0.7.0 3,584; 0.9.1 4,352). The unchanged gates() tests
# `info["kv_pool_tokens"] in KV_ACCEPT_22V[e]`; KVFloor answers that membership test with D's rule (exactly one pool line,
# and pool >= floor), and analysis_e renames the check so its label states the floor rule.
CAP, MAX_RECORD_PROMPT, CONC = 327680, 2795, 8
ATTN_BLOCK = {"v070": 3584, "v091": 4352}


class KVFloor:
    def __init__(self, e):
        b = ATTN_BLOCK[e]
        self.engine, self.block = e, b
        self.floor = CONC * b * -(-(CAP + MAX_RECORD_PROMPT + 16) // b)

    def __contains__(self, v):
        return isinstance(v, list) and len(v) == 1 and isinstance(v[0], int) and v[0] >= self.floor

    def __repr__(self):
        return f"floor {self.floor:,}"

    def label(self):
        return (f"server log KV pool >= functional floor {self.floor:,} tokens (8 requests at the 327,680 cap + longest record "
                f"prompt, whole {self.block:,}-token blocks; reported, not identity-gated; amendments 22ad / 22ah)")


KV_ACCEPT_E = {e: KVFloor(e) for e in ENGINES}
KV_FLOOR = {e: KV_ACCEPT_E[e].floor for e in ENGINES}
assert KV_FLOOR == {"v070": 2666496, "v091": 2646016}, KV_FLOOR
CONFIG_OVERRIDES = {"ENGINES": ENGINES, "KV_ACCEPT_22V": {e: repr(v) for e, v in KV_ACCEPT_E.items()}}
A.ENGINES = ENGINES
A.KV_ACCEPT_22V = KV_ACCEPT_E
KV_CHECK_OLD = "server log kv_pool_tokens is one the 0.7.0 record itself produced"
N_J = 4


def relabel_kv(e, checks):
    """Rename the floor check produced by the unchanged gates() so its label states the 22ah rule."""
    for c in checks:
        if c["check"].startswith(f"{e}: {KV_CHECK_OLD}"):
            c["check"] = f"{e}: {KV_ACCEPT_E[e].label()}"
    return checks


MEANING = {
    "R-E": "From identical 0.9.1-built 100k-token states, stock 0.7.0's continuation finished within +64k more often than "
           "stock 0.9.1's, by at least the prespecified 0.15 with the prefix-level interval and test excluding no difference. "
           "This does not identify which component of 'engine' is responsible, and holds for these 13 0.9.1-built states only.",
    "R-E-": "From identical 0.9.1-built 100k-token states, stock 0.9.1's continuation finished more often than 0.7.0's under "
            "the same rule.",
    "R-I": "Inconclusive under the prespecified rules: no engine continuation effect is supported in either direction from "
           "the 0.9.1-built states. With 13 prefixes this is the likeliest reading unless the effect is large (PROPOSAL.md 5.3)."}


def fmt_ci(c):
    return "-" if c is None else f"{c[0]:.3f} to {c[1]:.3f}"


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def probe(S, plan):
    st = {}
    for e in ENGINES:
        s = S[e]["session"]
        if s is None:
            st[e] = "not started"
        elif s.get("budget_stop"):
            st[e] = "budget stop (not started)"
        elif not s.get("completed") or s.get("watchdog") or s.get("client_rc") != 0 or not S[e]["end"]:
            st[e] = f"interrupted or failed (client rc {s.get('client_rc')}, watchdog {s.get('watchdog')}, ready {s.get('ready')})"
        elif len(S[e]["req"]) != len(plan["order"]):
            st[e] = f"{len(S[e]['req'])} of {len(plan['order'])} request records"
        else:
            st[e] = "complete"
    return ("COMPLETE" if all(st[e] == "complete" for e in ENGINES) else "INCOMPLETE"), st


def matrix(rows, pids):
    y = np.full((len(pids), N_J), np.nan)
    for r in rows:
        y[pids.index(r["prefix_id"]), r["j"] - 1] = 1.0 if r["finish"] else 0.0
    assert not np.isnan(y).any(), "missing outcome"
    return y


def ci(rates, seed_off=0):
    c = BS.boot_ci(np.asarray(rates, dtype=np.float64), seed=BS.BOOT_SEED + seed_off)
    return [round(c[0.95][0], 4), round(c[0.95][1], 4)]


def two_sample(a, b, seed_off):
    """Descriptive: mean(a) - mean(b) for two independent prefix sets; percentile bootstrap resampling each set within
    itself (B_BOOT draws); exact two-sided permutation p over all splits of the pooled prefixes."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    obs = float(a.mean() - b.mean())
    rng = np.random.default_rng(BS.BOOT_SEED + seed_off)
    m = a[rng.integers(0, len(a), (BS.B_BOOT, len(a)))].mean(1) - b[rng.integers(0, len(b), (BS.B_BOOT, len(b)))].mean(1)
    pool = np.concatenate([a, b])
    n, na, tot = len(pool), len(a), pool.sum()
    hits = cnt = 0
    for idx in itertools.combinations(range(n), na):
        sa = pool[list(idx)].sum()
        d = sa / na - (tot - sa) / (n - na)
        hits += abs(d) >= abs(obs) - 1e-12
        cnt += 1
    return {"mean_diff": round(obs, 4), "ci95": [round(float(np.quantile(m, 0.025)), 4), round(float(np.quantile(m, 0.975)), 4)],
            "permutation_p_two_sided": round(hits / cnt, 4), "n": [len(a), len(b)]}


def cross(Y, man, pids):
    """Descriptive cross (PROPOSAL.md section 5.4). Y: engine -> (8, 4) finish matrix of the E run."""
    for f, s in REF_FILES.items():
        if fsha(f"{WORK}/{f}") != s:
            return {"error": f"{f} is not the frozen reference file"}
    bp = json.load(open(f"{WORK}/ref/bplus-result-22v.json"))
    bman = {m["prefix_id"]: m for m in json.load(open(f"{WORK}/ref/bplus-prefix_manifest.json"))["prefixes"]}
    if bp.get("state") != "COMPLETE" or not all(c["ok"] for c in bp["checks"]):
        return {"error": "B+ reference result is not COMPLETE with every check passing"}
    bpids = sorted(bman)
    YB = {e: matrix(bp["per_request"][e], bpids) for e in ENGINES}
    src = {"0.7.0-built (B+)": (YB, bpids, bman), "0.9.1-built (E)": (Y, pids, man)}
    out = {"labels": "source = the engine that generated the 100k prefix; engine = the engine that continued it",
           "cells": {}, "by_stratum": {}, "caveats": [
               f"Prefixes differ by question between the sources ({len(set(m['doc'] for m in man.values()) & set(m['doc'] for m in bman.values()))} "
               f"of {len(set(m['doc'] for m in man.values()) | set(m['doc'] for m in bman.values()))} questions shared), so the source contrast "
               "is confounded with question; the question-matched sub-table is the only within-question comparison.",
               "B+ ran 2026-10-20/21 and E runs later: session date, thermal state and seeds differ between sources.",
               "Descriptive only: no reading is drawn from the cross."]}
    k = 0
    for sname, (YY, pp, mm) in src.items():
        for e in ENGINES:
            r = YY[e].mean(axis=1)
            k += 1
            out["cells"][f"{sname} x {e}"] = {"P": round(float(r.mean()), 4), "ci95": ci(r, k), "prefixes": len(pp),
                                             "finished": int(YY[e].sum()), "of": int(YY[e].size)}
            for st in ("capped", "finished"):
                idx = [i for i, p in enumerate(pp) if mm[p]["stratum"] == st]
                rs = r[idx]
                k += 1
                out["by_stratum"][f"{sname} x {e} x originally {st}"] = {
                    "P": round(float(rs.mean()), 4), "ci95": ci(rs, k) if len(rs) > 1 else None, "prefixes": len(rs)}
    for e in ENGINES:
        for st in ("capped", "finished"):
            vals = np.concatenate([YY[e].mean(axis=1)[[i for i, p in enumerate(pp) if mm[p]["stratum"] == st]]
                                   for YY, pp, mm in src.values()])
            k += 1
            out["by_stratum"][f"pooled sources x {e} x originally {st}"] = {"P": round(float(vals.mean()), 4),
                                                                             "ci95": ci(vals, k), "prefixes": len(vals)}
    pooled = {s: np.mean([YY[e].mean(axis=1) for e in ENGINES], axis=0) for s, (YY, _, _) in src.items()}
    out["source_contrast_pooled_over_engines"] = dict(two_sample(pooled["0.9.1-built (E)"], pooled["0.7.0-built (B+)"], 101),
                                                      what="0.9.1-built minus 0.7.0-built, prefix rates pooled over both engines; confounded with question")
    out["source_contrast_by_engine"] = {e: two_sample(Y[e].mean(axis=1), YB[e].mean(axis=1), 102 + i) for i, e in enumerate(ENGINES)}
    dE = Y["v070"].mean(axis=1) - Y["v091"].mean(axis=1)
    dB = YB["v070"].mean(axis=1) - YB["v091"].mean(axis=1)
    out["interaction"] = dict(two_sample(dE, dB, 110), what="(E1 = v070 - v091 on 0.9.1-built) minus (B+ C1 = v070 - v091 on 0.7.0-built)",
                              E1=round(float(dE.mean()), 4), C1=round(float(dB.mean()), 4),
                              C1_from_bplus_result=round(bp["primary"]["contrasts"]["C1"]["mean"], 4))
    edoc = {man[p]["doc"]: p for p in pids}
    bdoc = {bman[p]["doc"]: p for p in bpids}
    qm = []
    for d in sorted(set(edoc) & set(bdoc)):
        pe, pb = edoc[d], bdoc[d]
        row = {"doc": d, "E_prefix": pe, "E_reply": man[pe]["id"], "E_original_stratum": man[pe]["stratum"],
               "Bplus_prefix": pb, "Bplus_reply": bman[pb]["id"], "Bplus_original_stratum": bman[pb]["stratum"]}
        for e in ENGINES:
            row[f"P_0.9.1-built_{e}"] = float(Y[e][pids.index(pe)].mean())
            row[f"P_0.7.0-built_{e}"] = float(YB[e][bpids.index(pb)].mean())
        row["source_diff_pooled"] = round(float(np.mean([row[f"P_0.9.1-built_{e}"] - row[f"P_0.7.0-built_{e}"] for e in ENGINES])), 4)
        qm.append(row)
    out["question_matched"] = {"questions": qm, "n": len(qm),
                               "mean_source_diff_pooled": round(float(np.mean([r["source_diff_pooled"] for r in qm])), 4) if qm else None,
                               "note": "same question, different reply: each source's prefix is a different trajectory; descriptive only, no test"}
    return out


def main():
    out_json = sys.argv[1]
    plan = json.load(open(f"{WORK}/out/plan.json"))
    man = {m["prefix_id"]: m for m in json.load(open(f"{WORK}/out/prefix_manifest.json"))["prefixes"]}
    inventory = json.load(open(f"{WORK}/checks/seed-inventory.json"))
    S = A.load(plan)
    state, per = probe(S, plan)
    res = {"what": "E result (ids, counts, rates, positions; no text)", "state": state, "sessions": per,
           "config_overrides_on_bplus_22v": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CONFIG_OVERRIDES.items()},
           "rules": {"DELTA_E": BS.DELTA_E, "ALPHA": BS.ALPHA, "SMALL_MAX": BS.SMALL_MAX, "EQ_BOUND": BS.EQ_BOUND,
                     "ETA_MIN": BS.ETA_MIN, "B_BOOT": BS.B_BOOT, "BOOT_SEED": BS.BOOT_SEED}}
    md = [f"# E prefix-source x engine continuation result: {state}", ""]
    for e in ENGINES:
        md.append(f"- {LABEL[e]}: {per[e]}")
    if state == "INCOMPLETE":
        res["completed_requests"] = {e: len(S[e]["req"]) for e in ENGINES}
        res["reading"] = None
        md += ["", "**INCOMPLETE: no reading, no contrast, no screen, no cross table.** Completed request records per engine: "
               + ", ".join(f"{e} {len(S[e]['req'])}" for e in ENGINES) + ". Stop and amend; no automatic retry."]
        json.dump(res, open(out_json, "w"), indent=1)
        print("\n".join(md))
        return 4
    allc, R, X, seeds_all = [], {}, {}, []
    for e in ENGINES:
        c, R[e], X[e] = A.gates(e, S, plan, man, inventory)
        c = relabel_kv(e, c)
        slog = os.path.join(S[e]["dir"], "server.log")
        X[e]["kv_pool_tokens_reported"] = A.M.parse_serverlogs([slog], [], 0)[0].get("kv_pool_tokens") if os.path.exists(slog) else None
        X[e]["kv_floor"] = KV_FLOOR[e]
        allc += c
        seeds_all += [r["seed"] for r in R[e].values()]
    allc.append({"check": "seeds distinct across every request of the run", "ok": len(seeds_all) == len(set(seeds_all)),
                 "detail": f"{len(seeds_all)} seeds, {len(set(seeds_all))} distinct"})
    res["checks"] = allc
    res["session_extras"] = X
    bad = [c for c in allc if not c["ok"]]
    if bad:
        res["state"] = "INVALID"
        res["reading"] = None
        md += ["", f"**INVALID: {len(bad)} validity check(s) failed; no reading.**", ""]
        md += [f"- FAIL {c['check']} ({c['detail']})" for c in bad]
        json.dump(res, open(out_json, "w"), indent=1)
        print("\n".join(md))
        return 3
    pids = sorted(man)
    priv = json.load(open(f"{WORK}/PRIVATE/prefixes.json"))["prefixes"]
    Y, Yany, cont = {}, {}, {}
    for e in ENGINES:
        raw = A.raw_records(e)
        y, ya, rows = np.zeros((len(pids), N_J)), np.zeros((len(pids), N_J)), []
        for k, r in R[e].items():
            i, j = pids.index(r["prefix_id"]), r["j"] - 1
            fin = r["finish_reason"] == "stop" and r.get("stop_reason") in A.STOP_IDS
            y[i, j] = 1.0 if fin else 0.0
            ya[i, j] = 1.0 if r["finish_reason"] == "stop" else 0.0
            gen, text = raw[k][0]["token_ids"], raw[k][0]["text"]
            tc = r.get("think_close_at")
            reason = gen[:tc] if tc is not None else gen
            lm = A.loop_metrics(priv[r["prefix_id"]][man[r["prefix_id"]]["prompt_len"]:], reason)
            rtext = text.split("</think>", 1)[0]
            rows.append({"k": k, "prefix_id": r["prefix_id"], "j": r["j"], "seed": r["seed"], "finish": fin,
                         "finish_reason": r["finish_reason"], "stop_reason": r.get("stop_reason"),
                         "completion_tokens": r["completion_tokens"], "think_close_at": tc, "reasoning_tokens": len(reason),
                         "content_tokens": (len(gen) - tc - 1) if tc is not None else 0,
                         "cpt_reasoning": round(len(rtext) / max(1, len(reason)), 4), "secs": r["secs"],
                         "decode_tok_s": round(r["completion_tokens"] / max(1e-9, r["secs"]), 2),
                         "cached_tokens": r.get("cached_tokens"), **{f"loop_{a}": b for a, b in lm.items()}})
        Y[e], Yany[e], cont[e] = y, ya, sorted(rows, key=lambda x: x["k"])
    res["per_request"] = cont
    rates = {e: Y[e].mean(axis=1) for e in ENGINES}
    e1 = BS.contrast(rates["v070"], rates["v091"])
    e1.update(role="primary", label="v070 - v091 (0.9.1-built prefixes)")
    screen = "EFFECT" if e1["EFFECT"] else "REVERSE" if e1["REVERSE"] else "SMALL" if e1["SMALL"] else "unresolved"
    reading = "R-E" if e1["EFFECT"] else "R-E-" if e1["REVERSE"] else "R-I"
    allY = np.concatenate([Y[e] for e in ENGINES], axis=1)
    e2 = BS.eta2(allY)
    res["primary"] = {"P": {e: float(rates[e].mean()) for e in ENGINES}, "contrasts": {"E1": e1}, "E1_screen": screen,
                      "eta2_prefix": e2, "PREFIX_HIGH": bool(e2 is not None and e2 >= BS.ETA_MIN),
                      "icc1_prefix_pooled": BS.icc1(allY), "icc1_prefix_by_engine": {e: BS.icc1(Y[e]) for e in ENGINES},
                      "reading": reading,
                      "reading_rule": "B+'s frozen C1-ONLY rule on E1: R-E if EFFECT, R-E- if REVERSE, else R-I; R-S is not "
                                      "available with two engines. SMALL(E1) and PREFIX_HIGH are reported as screens, not a reading."}
    res["reading"] = reading
    res["meaning"] = MEANING[reading]
    res["cross"] = cross(Y, man, pids)
    sec = {}
    for e in ENGINES:
        rows = cont[e]
        fin = [r for r in rows if r["finish"]]
        loops = [r for r in rows if r["loop_onset_after_cut"] is not None and r["loop_onset_after_cut"] >= 0]
        sec[e] = {"finished": len(fin), "of": len(rows), "P_any_stop": float(Yany[e].mean()),
                  "P_by_stratum": {s: float(np.mean([Y[e][pids.index(p)].mean() for p in pids if man[p]["stratum"] == s])) for s in ("capped", "finished")},
                  "finish_tokens_median": statistics.median([r["completion_tokens"] for r in fin]) if fin else None,
                  "terminal_loop_onsets_in_continuation": len(loops),
                  "loop_onset_after_cut_median": statistics.median([r["loop_onset_after_cut"] for r in loops]) if loops else None,
                  "loop_types": {c: sum(r["loop_loop"]["class"] == c for r in rows) for c in
                                 ("short-period exact loop", "long-period exact cycling", "fragmentary recycling", "no terminal loop")},
                  "finished_from_terminal_repetition": sum(r["finish"] and r["loop_onset_g"] is not None for r in rows),
                  "cpt_reasoning_median": round(statistics.median([r["cpt_reasoning"] for r in rows]), 3),
                  "decode_tok_s_median": round(statistics.median([r["decode_tok_s"] for r in rows]), 2),
                  "seeds_unanimous_prefixes": int(sum(len(set(Y[e][i])) == 1 for i in range(len(pids)))),
                  "icc1": BS.icc1(Y[e]), "cached_tokens": [r["cached_tokens"] for r in rows]}
    pooled = np.mean([rates[e] for e in ENGINES], axis=0)
    cov = {k: [man[p]["covariates_100k"][k] for p in pids] for k in ("chars_per_token", "interjection_per1k", "digit_share", "R50_backward_at_cut")}
    sec["covariates_vs_pooled_rate_spearman"] = {k: A.spearman(v, list(pooled)) for k, v in cov.items()}
    sec["per_prefix_rates"] = {p: {e: float(Y[e][pids.index(p)].mean()) for e in ENGINES} for p in pids}
    res["secondary"] = sec
    json.dump(res, open(out_json, "w"), indent=1)
    pr, cr = res["primary"], res["cross"]
    nst = {s_: sum(man[p]["stratum"] == s_ for p in pids) for s_ in ("capped", "finished")}
    md += ["", f"**Reading: {reading}.** {MEANING[reading]}", "", f"## Primary: P(finish within +64k) on the {len(pids)} 0.9.1-built prefixes, prefix as the unit", "",
           f"| engine | P | finished / {len(pids) * N_J} | capped-origin ({nst['capped']}) | finished-origin ({nst['finished']}) |", "|---|---:|---:|---:|---:|"]
    for e in ENGINES:
        md.append(f"| {LABEL[e]} | {pr['P'][e]:.3f} | {sec[e]['finished']} | {sec[e]['P_by_stratum']['capped']:.3f} | {sec[e]['P_by_stratum']['finished']:.3f} |")
    md += ["", "| contrast | mean | 95% interval | 90% interval | sign-flip p | prefixes + / - / 0 | screen |", "|---|---:|---|---|---:|---|---|",
           f"| E1 v070 - v091 | {e1['mean']:+.3f} | {e1['ci95'][0]:+.3f} to {e1['ci95'][1]:+.3f} | {e1['ci90'][0]:+.3f} to {e1['ci90'][1]:+.3f} | "
           f"{e1['signflip_p_two_sided']:.4f} | {e1['prefixes_positive']} / {e1['prefixes_negative']} / {e1['prefixes_zero']} | {screen} |",
           "", f"Prefix share of outcome variance (eta^2, {allY.size} outcomes): {pr['eta2_prefix']}; PREFIX_HIGH {pr['PREFIX_HIGH']}. {pr['reading_rule']}"]
    if "error" in cr:
        md += ["", f"**Cross table not computed: {cr['error']}**"]
    else:
        md += ["", "## Cross (descriptive, prespecified): prefix source x engine", "", "| source x engine | P | 95% interval | finished / n |", "|---|---:|---|---:|"]
        for kk, v in cr["cells"].items():
            md.append(f"| {kk} | {v['P']:.3f} | {v['ci95'][0]:.3f} to {v['ci95'][1]:.3f} | {v['finished']} / {v['of']} |")
        sc, it = cr["source_contrast_pooled_over_engines"], cr["interaction"]
        md += ["", f"- Source contrast (0.9.1-built - 0.7.0-built, pooled over engines; **confounded with question**): {sc['mean_diff']:+.3f} "
               f"(95% {sc['ci95'][0]:+.3f} to {sc['ci95'][1]:+.3f}; permutation p {sc['permutation_p_two_sided']})",
               f"- Interaction (E1 - C1): {it['mean_diff']:+.3f} (95% {it['ci95'][0]:+.3f} to {it['ci95'][1]:+.3f}; permutation p "
               f"{it['permutation_p_two_sided']}); E1 {it['E1']:+.3f}, C1 {it['C1']:+.3f}", "",
               "| stratum cell | P | 95% interval | prefixes |", "|---|---:|---|---:|"]
        for kk, v in cr["by_stratum"].items():
            md.append(f"| {kk} | {v['P']:.3f} | {fmt_ci(v['ci95'])} | {v['prefixes']} |")
        md += ["", "Question-matched sub-table (same question in both prefix sets; different replies):", "",
               "| doc | E prefix (orig.) | B+ prefix (orig.) | 0.9.1-built v070 / v091 | 0.7.0-built v070 / v091 |", "|---|---|---|---|---|"]
        for q in cr["question_matched"]["questions"]:
            md.append(f"| q{q['doc']} | {q['E_prefix']} ({q['E_original_stratum']}) | {q['Bplus_prefix']} ({q['Bplus_original_stratum']}) | "
                      f"{q['P_0.9.1-built_v070']:.2f} / {q['P_0.9.1-built_v091']:.2f} | {q['P_0.7.0-built_v070']:.2f} / {q['P_0.7.0-built_v091']:.2f} |")
        md += ["", "Caveats: " + " ".join(cr["caveats"])]
    md += ["", "## Secondary (descriptive)", "", "| engine | any stop | median finish tokens | loop onsets in continuation (median after cut) | "
           "loop types (short / long / fragmentary / none) | cpt | unanimous prefixes | ICC | decode tok/s |", "|---|---:|---:|---|---|---:|---:|---:|---:|"]
    for e in ENGINES:
        s = sec[e]
        lt = s["loop_types"]
        md.append(f"| {LABEL[e]} | {s['P_any_stop']:.3f} | {s['finish_tokens_median']} | {s['terminal_loop_onsets_in_continuation']} "
                  f"({s['loop_onset_after_cut_median']}) | {lt['short-period exact loop']} / {lt['long-period exact cycling']} / "
                  f"{lt['fragmentary recycling']} / {lt['no terminal loop']} | {s['cpt_reasoning_median']} | {s['seeds_unanimous_prefixes']} / {len(pids)} | "
                  f"{None if s['icc1'] is None else round(s['icc1'], 3)} | {s['decode_tok_s_median']} |")
    md += ["", f"Covariates at 100k vs pooled prefix rate (Spearman, n = {len(pids)}, descriptive): {sec['covariates_vs_pooled_rate_spearman']}",
           "", "Caveats: one 0.9.1-built state per question; prefixes rebuilt by prefill (decode-built state absent); re-tokenized ids "
           "(drift 0-6 tokens per reply); 'engine' bundles decode path, D2/D7 exposure, layout and speculation depth."]
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
