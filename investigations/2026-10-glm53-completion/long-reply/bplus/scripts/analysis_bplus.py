#!/usr/bin/env python3
"""B+ frozen analysis (PROPOSAL.md sections 5-6). CPU-only container (numpy; trajectory.py needs tokenizers), zero GPU.
Run through scripts/run_cpu.sh with RUN_DIR mounted at /run (read-only):
    RUN_DIR=<run folder> scripts/run_cpu.sh v070 scripts/analysis_bplus.py /work/<out>.json > <out>.md
Writes ids, seeds, counts, rates, positions and hashes only: no question, answer or reply text, no token ids.

Order of work (never classify a partial or invalid run):
  1. probe   every engine session's state (runner's session.json, client log). COMPLETE = all three sessions ran to the
             end; C1-ONLY = 0.7.0 and 0.9.1 complete and B1 not started by the runner's budget rule (frozen fallback:
             reading on C1 only, R-S unavailable); anything else = INCOMPLETE (cause and counts, no reading).
  2. gates   per engine (section 6): plan coverage, HTTP 200, seeds, prompt-id hashes (sent and server-echoed),
             sampling parameters, generated ids vs raw file, finish fields, server configuration identity with the
             engine's reference record, image id and loopback publish, release warm-up and request timing, error lines,
             submission order and concurrency, GPU pre-flight. Any failure = INVALID (no reading). Preemption is
             reported, not invalidating.
  3. primary per engine P(finish within +64k): finish = finish_reason 'stop' with stop_reason a stop-token id
             (154820 / 154827 / 154829) before max_tokens; unit = prefix (mean of 4 seeds, then of 12 prefixes);
             contrasts and screens in bplus_stats.py.
  4. secondary, descriptive: any-stop finishing; loop onset in the continuation with the discovery's frozen R50 rule
             (trajectory.py, byte-identical copy, imported) on prefix reasoning + continuation reasoning; loop type (the
             discovery's E / P rule, as in trajectory.analyse); chars per token of the continuation reasoning;
             within-prefix agreement (ICC, eta^2, seeds agreeing); outcome by stratum; covariates at 100k; prefix-cache
             hits; decode rates.
"""
import datetime
import gzip
import hashlib
import json
import os
import re
import statistics
import sys

import numpy as np

WORK = os.environ.get("BPLUS_WORK", "/work")
RUN = os.environ.get("BPLUS_RUN", "/run")
sys.path.insert(0, f"{WORK}/scripts")
FROZEN = {"trajectory.py": "c24267b1a9641fc2a3bf4b9bca23950d9530d787002d74286bef8b717f5ac4c6",
          "analysis_col511.py": "b5b13ce2155085ca46fb24877f12b7cd0a5cdd16eb4cf19012958e7b86ae0033"}
for _f, _s in FROZEN.items():
    if hashlib.sha256(open(f"{WORK}/scripts/{_f}", "rb").read()).hexdigest() != _s:
        sys.exit(f"{_f} is not the frozen file")
import analysis_col511 as M  # noqa: E402
import bplus_stats as BS  # noqa: E402
import trajectory as TJ  # noqa: E402

ENGINES = ("v070", "v091", "b1")
LABEL = {"v070": "stock 0.7.0", "v091": "stock 0.9.1", "b1": "B1 (0.9.1 + D1-D4 + D7)"}
STOP_IDS = (154820, 154827, 154829)
THINK_CLOSE = 154842
H = 65536
CUT = 100000
READY_LINE = "GLM release startup warmup complete; container is ready."
COMPL_ACCESS = '"POST /v1/completions HTTP'
REF = {  # reference records (section 6): 0.7.0 rep070-F server logs; 0.9.1 record's server log; B1 screen's server log
    "v070": dict(image="sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629",
                 nd="74f26837e0e20391bbc8e39c037bb7a44a59fcab1335032a65729cc9eb8d0b72",
                 snaps=["701cd7456c13d87bf0147ad946f828a999afb59c", "bf582e4eacc1810f76656d1811693ff6c6737d2a"],
                 env="014d31a48bdd11c1a48236e5c2e4ddcba69d037d4f7ca7bc577290afe52b6a97", kv=[2758919]),
    "v091": dict(image="sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07",
                 nd="7025a96888a67642741db00aaecb9cbf786c2ba67cc4ded7d67c1cf4e909dcc5",
                 snaps=["0490d2f708b12145f6516555ab066aaeb401cd21", "bf582e4eacc1810f76656d1811693ff6c6737d2a"],
                 env="f85d9d360488c02bc9ba2bbb33b86319201cfad14cd4c6a9ccb312539ec1313e", kv=[4707515]),
    "b1": dict(image="sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a",
               nd="7025a96888a67642741db00aaecb9cbf786c2ba67cc4ded7d67c1cf4e909dcc5",
               snaps=["0490d2f708b12145f6516555ab066aaeb401cd21", "bf582e4eacc1810f76656d1811693ff6c6737d2a"],
               env="f85d9d360488c02bc9ba2bbb33b86319201cfad14cd4c6a9ccb312539ec1313e", kv=[4707515])}
for _e in REF:
    REF[_e].update(vllm=["0.1.dev20051+g487ecf187"], cg=["FULL_AND_PIECEWISE"], warm=9, warm_compl=8)


def ids_sha(ids):
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def ts_utc(line):
    m = re.search(r" (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) ", line)
    if not m:
        return None
    return datetime.datetime(2026, *map(int, m.groups()), tzinfo=datetime.timezone.utc).timestamp()


def load(plan):
    S = {}
    for e in ENGINES:
        d = os.path.join(RUN, e)
        sess = json.load(open(f"{d}/session.json")) if os.path.exists(f"{d}/session.json") else None
        log = jl(f"{d}/requests.jsonl")
        S[e] = {"session": sess, "start": [r for r in log if r.get("kind") == "start"],
                "end": [r for r in log if r.get("kind") == "end"],
                "req": [r for r in log if r.get("kind") == "request"], "dir": d}
    return S


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
    if all(st[e] == "complete" for e in ENGINES):
        return "COMPLETE", st
    if st["v070"] == "complete" and st["v091"] == "complete" and st["b1"] == "budget stop (not started)":
        return "C1-ONLY", st
    return "INCOMPLETE", st


def raw_records(e):
    p = os.path.join(RUN, "raw", f"{e}.raw.jsonl.gz")
    out = {}
    if not os.path.exists(p):
        return out
    with gzip.open(p, "rt") as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                out.setdefault(r["k"], []).append(r)
    return out


def gates(e, S, plan, man, inventory):
    c = []

    def chk(name, ok, detail=""):
        c.append({"check": f"{e}: {name}", "ok": bool(ok), "detail": str(detail)[:300]})

    reqs = S[e]["req"]
    sess = S[e]["session"] or {}
    byk = {}
    for r in reqs:
        byk.setdefault(r["k"], []).append(r)
    ks = [r["k"] for r in plan["order"]]
    chk("every planned request has exactly one record, and no other record exists",
        sorted(byk) == sorted(ks) and all(len(v) == 1 for v in byk.values()), f"{len(reqs)} records, {len(byk)} distinct k")
    R = {k: v[0] for k, v in byk.items()}
    P = {r["k"]: r for r in plan["order"]}
    ok200 = [k for k in ks if k in R and R[k].get("http_status") == 200 and not R[k].get("error")]
    chk("HTTP 200 without error for all requests", len(ok200) == len(ks), f"{len(ok200)} / {len(ks)}")
    chk("prefix id and seed index as planned", all(k in R and R[k]["prefix_id"] == P[k]["prefix_id"] and R[k]["j"] == P[k]["j"] for k in ks))
    chk("seed as planned for this engine", all(k in R and R[k]["seed"] == P[k]["seeds"][e] for k in ks),
        [k for k in ks if k in R and R[k]["seed"] != P[k]["seeds"][e]][:5])
    elig = {int(s) for s, v in inventory["candidates"].items() if v["eligible"]}
    chk("every seed was eligible in the pre-run seed inventory (never used before)", all(k in R and R[k]["seed"] in elig for k in ks))
    chk("sent prompt ids hash and length equal the frozen prefix",
        all(k in R and R[k]["prompt_ids_sha256"] == man[P[k]["prefix_id"]]["prefix_ids_sha256"] and R[k]["prompt_len"] == man[P[k]["prefix_id"]]["prefix_len"] for k in ks))
    chk("server-echoed prompt ids hash equals the frozen prefix (byte-identical prompt received)",
        all(k in R and R[k].get("returned_prompt_ids_sha256") == man[P[k]["prefix_id"]]["prefix_ids_sha256"] for k in ks),
        [k for k in ks if k in R and R[k].get("returned_prompt_ids_sha256") != man[P[k]["prefix_id"]]["prefix_ids_sha256"]][:5])
    chk("server prompt_tokens equal the prefix length", all(k in R and R[k].get("prompt_tokens") == man[P[k]["prefix_id"]]["prefix_len"] for k in ks))
    chk("sampling parameters hash equals the plan's (max_tokens 65,536, temperature 1.0, top_p 0.95, stop </s>, return_token_ids)",
        all(k in R and R[k].get("params_sha256") == plan["params_sha256"] for k in ks))
    fin_ok = all(k in R and R[k].get("finish_reason") in ("stop", "length") for k in ks)
    len_ok = all(k in R and (R[k].get("completion_tokens") or 0) <= H and (R[k]["finish_reason"] != "length" or R[k]["completion_tokens"] == H) for k in ks)
    chk("finish_reason is stop or length; completion tokens <= 65,536; length only at 65,536", fin_ok and len_ok)
    chk("generated ids returned and their count equals completion_tokens", all(k in R and R[k].get("gen_ids_len") == R[k].get("completion_tokens") for k in ks))
    raw = raw_records(e)
    chk("raw file holds exactly one record per request whose ids and text hash to the request log",
        all(k in raw and len(raw[k]) == 1 and ids_sha(raw[k][0]["token_ids"]) == R[k]["gen_ids_sha256"]
            and hashlib.sha256(raw[k][0]["text"].encode()).hexdigest() == R[k]["text_sha256"] and raw[k][0]["seed"] == R[k]["seed"] for k in ks if k in R))
    # server configuration identity, image, loopback
    slog = os.path.join(S[e]["dir"], "server.log")
    ref = REF[e]
    if os.path.exists(slog):
        win = [(min(r["t0"] for r in reqs) - 5, max(r["t1"] for r in reqs) + 5)] if reqs else []
        info = M.parse_serverlogs([slog], win, 0)[0]
        for k_, key in (("nd", "non_default_args_sha256"), ("vllm", "vllm_version"), ("snaps", "model_snapshots"),
                        ("env", "env_names_logged_sha256"), ("kv", "kv_pool_tokens"), ("cg", "cudagraph_mode")):
            chk(f"server log {key} equals the reference record's", info.get(key) == ref[k_], f"{str(info.get(key))[:40]} vs {str(ref[k_])[:40]}")
        lines = open(slog, errors="replace").read().splitlines()
        starts = sum("non-default args:" in l for l in lines)
        warm = [i for i, l in enumerate(lines) if "GLM release startup " in l]
        ready = [i for i, l in enumerate(lines) if READY_LINE in l]
        chk("exactly one engine start in the server log", starts == 1, starts)
        chk(f"{ref['warm']} release warm-up lines ending with the ready line", len(warm) == ref["warm"] and len(ready) == 1 and warm[-1] == ready[0],
            f"{len(warm)} warm-up lines, {len(ready)} ready lines")
        if ready:
            early = sum(COMPL_ACCESS in l for l in lines[:ready[0]])
            chk(f"exactly the release warm-up's own {ref['warm_compl']} /v1/completions requests before the ready line",
                early == ref["warm_compl"], early)
            before = [t for t in (ts_utc(l) for l in lines[:ready[0]]) if t is not None]
            t_lo = before[-1] if before else None
            first = min((r["t0"] for r in reqs), default=None)
            chk("first request sent after the last timestamped line before the ready line", t_lo is not None and first is not None and first > t_lo,
                f"first {first}, last before ready {t_lo}")
        iw = info.get("in_window") or {}
        chk("no illegal-memory-access or NaN warning lines during the requests", iw.get("illegal_memory_access_lines", 1) == 0 and iw.get("nan_warning_lines", 1) == 0, iw)
        pre = iw.get("preemption_lines")
        prefix_hit = [float(x) for x in re.findall(r"Prefix cache hit rate: ([\d.]+)%", "\n".join(lines))]
    else:
        chk("server log present", False)
        pre, prefix_hit = None, []
    insp = os.path.join(S[e]["dir"], "inspect.json")
    try:
        d = json.load(open(insp))[0]
        b = (d.get("HostConfig") or {}).get("PortBindings") or {}
        hosts = sorted({x.get("HostIp") for v in b.values() for x in (v or [])})
        chk("container image id equals the pinned image", d.get("Image") == ref["image"], d.get("Image"))
        chk("container ports published on 127.0.0.1 only", hosts == ["127.0.0.1"], hosts)
    except Exception as ex:  # noqa: BLE001
        chk("docker inspect record readable", False, repr(ex))
    # submission order and concurrency (client timestamps)
    t0s = [R[k]["t0"] for k in ks if k in R]
    chk("requests submitted in the plan order", t0s == sorted(t0s))
    ev = sorted([(R[k]["t0"], 1) for k in ks if k in R] + [(R[k]["t1"], -1) for k in ks if k in R], key=lambda x: (x[0], x[1]))
    cur = mx = 0
    for _, dlt in ev:
        cur += dlt
        mx = max(mx, cur)
    chk("at most 8 requests in flight", mx <= 8, mx)
    chk("GPU pre-flight passed before the session (runner)", sess.get("preflight_ok") is True, sess.get("preflight_ok"))
    chk("server became ready through the release gate (runner)", sess.get("ready") is True, sess.get("ready"))
    rc_ = [r for r in jl(os.path.join(S[e]["dir"], "requests.jsonl")) if r.get("kind") == "render_check"]
    extra = {"preemption_lines_in_window": pre, "max_in_flight": mx,
             "server_render_check": {"prefixes": len(rc_), "equal": sum(bool(r.get("equal")) for r in rc_),
                                     "errors": sum(bool(r.get("error")) for r in rc_)},
             "prefix_cache_hit_rate_lines_max": max(prefix_hit) if prefix_hit else None,
             "cached_tokens": [R[k].get("cached_tokens") for k in ks if k in R]}
    return c, R, extra


def loop_metrics(prefix_reason_ids, cont_reason_ids):
    """The discovery's frozen R50 rule (PRESPEC.md; trajectory.py functions) on prefix reasoning + continuation reasoning."""
    ids = np.array(list(prefix_reason_ids) + list(cont_reason_ids), dtype=np.int64)
    N = len(ids)
    rep, prev = {}, {}
    for n in (20, 50):
        rep[n], prev[n] = TJ.rep_and_prev(TJ.ngram_hash(ids, n))
    r50 = rep[50].astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(r50)])
    Mn = len(r50)
    R50 = (cs[TJ.W:] - cs[:-TJ.W]) / TJ.W if Mn >= TJ.W else np.zeros(0)
    onset, eps = TJ.onset_and_episodes(R50, TJ.THR)
    out = {"N": N, "onset_g": onset, "onset_after_cut": (onset - CUT) if onset is not None else None,
           "episodes": len(eps), "episodes_in_continuation": sum(x["start"] >= CUT for x in eps),
           "R50_final": round(float(R50[-1]), 4) if len(R50) else None,
           "R50_max_continuation": round(float(R50[CUT:].max()), 4) if len(R50) > CUT else None}
    cls = {"class": "no terminal loop"}
    if onset is not None:
        s_st, s_len, s_per = TJ.spans_from(rep[20], prev[20], 20)
        big = s_len >= TJ.SPAN_MIN
        cov = np.zeros(N + 1, dtype=np.int64)
        per_tokens = {}
        for a, L, p in zip(s_st[big], s_len[big], s_per[big]):
            a2, b2 = max(a, onset), min(a + L, N)
            if b2 > a2:
                cov[a2] += 1
                cov[b2] -= 1
                per_tokens[int(p)] = per_tokens.get(int(p), 0) + int(b2 - a2)
        covered = np.cumsum(cov)[:N] > 0
        E = float(covered[onset:].mean()) if N > onset else 0.0
        Pp = max(per_tokens, key=per_tokens.get) if per_tokens else None
        if E >= 0.5 and Pp is not None and Pp <= TJ.SHORT_P:
            cl = "short-period exact loop"
        elif E >= 0.5:
            cl = "long-period exact cycling"
        else:
            cl = "fragmentary recycling"
        cls = {"class": cl, "E": round(E, 4), "P": Pp}
    out["loop"] = cls
    return out


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for t in range(i, j + 1):
                r[o[t]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    if len(set(rx)) < 2 or len(set(ry)) < 2:
        return None
    return round(float(np.corrcoef(rx, ry)[0, 1]), 3)


def main():
    out_json = sys.argv[1]
    plan = json.load(open(f"{WORK}/out/plan.json"))
    man = {m["prefix_id"]: m for m in json.load(open(f"{WORK}/out/prefix_manifest.json"))["prefixes"]}
    inventory = json.load(open(f"{WORK}/checks/seed-inventory.json"))
    S = load(plan)
    state, per = probe(S, plan)
    res = {"what": "B+ result (ids, counts, rates, positions; no text)", "state": state, "sessions": per,
           "rules": {"DELTA_E": BS.DELTA_E, "ALPHA": BS.ALPHA, "SMALL_MAX": BS.SMALL_MAX, "EQ_BOUND": BS.EQ_BOUND,
                     "ETA_MIN": BS.ETA_MIN, "B_BOOT": BS.B_BOOT, "BOOT_SEED": BS.BOOT_SEED}}
    md = [f"# B+ continuation result: {state}", ""]
    for e in ENGINES:
        md.append(f"- {LABEL[e]}: {per[e]}")
    engines_done = [e for e in ENGINES if per[e] == "complete"]
    if state == "INCOMPLETE":
        res["completed_requests"] = {e: len(S[e]["req"]) for e in ENGINES}
        res["reading"] = None
        md += ["", "**INCOMPLETE: no reading, no contrast, no screen.** Completed request records per engine: "
               + ", ".join(f"{e} {len(S[e]['req'])}" for e in ENGINES) + ". Stop and amend; no automatic retry."]
        json.dump(res, open(out_json, "w"), indent=1)
        print("\n".join(md))
        return 4
    allc, R, X = [], {}, {}
    seeds_all = []
    for e in engines_done:
        c, R[e], X[e] = gates(e, S, plan, man, inventory)
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
    Y, Yany, cont = {}, {}, {}
    priv = json.load(open(f"{WORK}/PRIVATE/prefixes.json"))["prefixes"]
    for e in engines_done:
        raw = raw_records(e)
        y = np.zeros((12, 4))
        ya = np.zeros((12, 4))
        rows = []
        for k, r in R[e].items():
            i, j = pids.index(r["prefix_id"]), r["j"] - 1
            fin = r["finish_reason"] == "stop" and r.get("stop_reason") in STOP_IDS
            y[i, j] = 1.0 if fin else 0.0
            ya[i, j] = 1.0 if r["finish_reason"] == "stop" else 0.0
            gen = raw[k][0]["token_ids"]
            text = raw[k][0]["text"]
            tc = r.get("think_close_at")
            reason = gen[:tc] if tc is not None else gen
            pre_reason = priv[r["prefix_id"]][man[r["prefix_id"]]["prompt_len"]:]
            lm = loop_metrics(pre_reason, reason)
            rtext = text.split("</think>", 1)[0]
            rows.append({"k": k, "prefix_id": r["prefix_id"], "j": r["j"], "seed": r["seed"], "finish": fin,
                         "finish_reason": r["finish_reason"], "stop_reason": r.get("stop_reason"),
                         "completion_tokens": r["completion_tokens"], "think_close_at": tc,
                         "reasoning_tokens": len(reason), "content_tokens": (len(gen) - tc - 1) if tc is not None else 0,
                         "cpt_reasoning": round(len(rtext) / max(1, len(reason)), 4), "secs": r["secs"],
                         "decode_tok_s": round(r["completion_tokens"] / max(1e-9, r["secs"]), 2),
                         "cached_tokens": r.get("cached_tokens"), **{f"loop_{a}": b for a, b in lm.items()}})
        Y[e], Yany[e], cont[e] = y, ya, sorted(rows, key=lambda x: x["k"])
    res["per_request"] = cont
    if state == "C1-ONLY":
        rates = {e: Y[e].mean(axis=1) for e in engines_done}
        c1 = BS.contrast(rates["v070"], rates["v091"])
        reading = "R-E" if c1["EFFECT"] else "R-E-" if c1["REVERSE"] else "R-I"
        res["primary"] = {"P": {e: float(rates[e].mean()) for e in rates}, "contrasts": {"C1": c1},
                          "reading": reading, "note": "C1-ONLY fallback: B1 not started (budget rule); R-S unavailable"}
    else:
        res["primary"] = BS.screens(Y)
        reading = res["primary"]["reading"]
    res["reading"] = reading
    res["meaning"] = BS.MEANING[reading]
    # secondary, descriptive
    sec = {}
    strata = {p: man[p]["stratum"] for p in pids}
    for e in engines_done:
        rows = cont[e]
        fin = [r for r in rows if r["finish"]]
        loops = [r for r in rows if r["loop_onset_after_cut"] is not None and r["loop_onset_after_cut"] >= 0]
        sec[e] = {"finished": len(fin), "of": len(rows), "P_any_stop": float(Yany[e].mean()),
                  "P_by_stratum": {s: float(np.mean([Y[e][pids.index(p)].mean() for p in pids if strata[p] == s])) for s in ("capped", "finished")},
                  "finish_tokens_median": statistics.median([r["completion_tokens"] for r in fin]) if fin else None,
                  "terminal_loop_onsets_in_continuation": len(loops),
                  "loop_onset_after_cut_median": statistics.median([r["loop_onset_after_cut"] for r in loops]) if loops else None,
                  "loop_types": {c: sum(r["loop_loop"]["class"] == c for r in rows) for c in
                                 ("short-period exact loop", "long-period exact cycling", "fragmentary recycling", "no terminal loop")},
                  "finished_from_terminal_repetition": sum(r["finish"] and r["loop_onset_g"] is not None for r in rows),
                  "cpt_reasoning_median": round(statistics.median([r["cpt_reasoning"] for r in rows]), 3),
                  "decode_tok_s_median": round(statistics.median([r["decode_tok_s"] for r in rows]), 2),
                  "seeds_unanimous_prefixes": int(sum(len(set(Y[e][i])) == 1 for i in range(12))),
                  "icc1": BS.icc1(Y[e]),
                  "cached_tokens": [r["cached_tokens"] for r in rows]}
    pooled = np.mean([Y[e].mean(axis=1) for e in engines_done], axis=0)
    cov = {k: [man[p]["covariates_100k"][k] for p in pids] for k in ("chars_per_token", "interjection_per1k", "digit_share", "R50_backward_at_cut")}
    sec["covariates_vs_pooled_rate_spearman"] = {k: spearman(v, list(pooled)) for k, v in cov.items()}
    sec["per_prefix_rates"] = {p: {e: float(Y[e][pids.index(p)].mean()) for e in engines_done} for p in pids}
    res["secondary"] = sec
    json.dump(res, open(out_json, "w"), indent=1)
    pr = res["primary"]
    md += ["", f"**Reading: {reading}.** {BS.MEANING[reading]}", "", "## Primary: P(finish within +64k), prefix as the unit", "",
           "| engine | P | finished / 48 | capped-origin | finished-origin |", "|---|---:|---:|---:|---:|"]
    for e in engines_done:
        md.append(f"| {LABEL[e]} | {pr['P'][e]:.3f} | {sec[e]['finished']} | {sec[e]['P_by_stratum']['capped']:.3f} | {sec[e]['P_by_stratum']['finished']:.3f} |")
    md += ["", "| contrast | mean | 95% interval | 90% interval | sign-flip p | prefixes + / - / 0 | screen |", "|---|---:|---|---|---:|---|---|"]
    for n, cc in pr["contrasts"].items():
        scr = "EFFECT" if cc["EFFECT"] else "REVERSE" if cc["REVERSE"] else "SMALL" if cc["SMALL"] else "-"
        md.append(f"| {n} {cc.get('label', '')} | {cc['mean']:+.3f} | {cc['ci95'][0]:+.3f} to {cc['ci95'][1]:+.3f} | "
                  f"{cc['ci90'][0]:+.3f} to {cc['ci90'][1]:+.3f} | {cc['signflip_p_two_sided']:.4f} | "
                  f"{cc['prefixes_positive']} / {cc['prefixes_negative']} / {cc['prefixes_zero']} | {scr} |")
    if "eta2_prefix" in pr:
        md += ["", f"Prefix share of outcome variance (eta^2, 144 outcomes): {pr['eta2_prefix']}; PREFIX_HIGH {pr['PREFIX_HIGH']}; "
               f"secondary: C2 {pr['secondary']['C2']}, C3 {pr['secondary']['C3']}."]
    md += ["", "## Secondary (descriptive)", "", "| engine | any stop | median finish tokens | loop onsets in continuation (median after cut) | "
           "loop types (short / long / fragmentary / none) | cpt | unanimous prefixes | ICC | decode tok/s |", "|---|---:|---:|---|---|---:|---:|---:|---:|"]
    for e in engines_done:
        s = sec[e]
        lt = s["loop_types"]
        md.append(f"| {LABEL[e]} | {s['P_any_stop']:.3f} | {s['finish_tokens_median']} | {s['terminal_loop_onsets_in_continuation']} "
                  f"({s['loop_onset_after_cut_median']}) | {lt['short-period exact loop']} / {lt['long-period exact cycling']} / "
                  f"{lt['fragmentary recycling']} / {lt['no terminal loop']} | {s['cpt_reasoning_median']} | {s['seeds_unanimous_prefixes']} / 12 | "
                  f"{None if s['icc1'] is None else round(s['icc1'], 3)} | {s['decode_tok_s_median']} |")
    md += ["", f"Covariates at 100k vs pooled prefix rate (Spearman, n = 12, descriptive): {sec['covariates_vs_pooled_rate_spearman']}",
           "", "Caveats: one 0.7.0-built state per question; prefixes rebuilt by prefill (decode-built state absent); "
           "re-tokenized ids (drift 0-8 tokens per reply); 'engine' bundles decode path, D2/D7 exposure, layout and speculation depth."]
    print("\n".join(md))
    return 0 if state == "COMPLETE" else 5


if __name__ == "__main__":
    sys.exit(main())
