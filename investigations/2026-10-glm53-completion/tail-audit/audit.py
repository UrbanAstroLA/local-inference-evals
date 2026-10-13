#!/usr/bin/env python3
"""Tail-accounting audit (PLAN.md amendment 20, section 2), zero GPU, CPU only.

Audits the exploratory tail analysis of amendment 19f (../tail-analysis/tail_analysis.py, .out).

Inputs (read only):
  - recording-proxy request logs <gpqa-runs>/*.requests.jsonl (completion_tokens, finish_reason, seed, t_start, ...)
  - lm-eval samples in <gpqa-runs>/<recipe>__passN (reply text is read IN MEMORY ONLY, to (a) verify the request-log
    content hash, (b) run the frozen stated-answer extractor, (c) re-tokenize the final-answer content; no text is written)
  - published per-question rows <this-repo>/runs/*/results.jsonl (stock 0.7.0, stock 0.9.1)
  - hand adjudications of the modified configurations (ids and verdicts only)
  - server logs (one engine start per log = one server session) for session position
Outputs: audit.json (all numbers, per-request records: ids, hashes, counts, flags) and audit.out (markdown tables).

Definitions (fixed here, before the tables were read; thresholds and bands were chosen by 19f after the results):
  T            = completion_tokens logged by the proxy (measured; includes reasoning, final answer and stop token)
  censored     = finish_reason 'length' with T == 327,680 (max_tokens): administrative censoring at the cap
  event        = generation ended before the cap (finish_reason 'stop')
  valid final-answer completion = event AND content non-empty (results flag `empty` False: reply.strip() != '')
                 AND the frozen stated-answer extractor (export_evals.py final_answer, sha256 731f27d4 of lines 233-259)
                 returns a letter A-D AND the reply was not ruled hedged by hand.  Correctness is NOT part of validity.
  other stopping = event that is not a valid final-answer completion: empty content with finish 'stop' ('empty_stop'),
                 content with no extractable letter ('no_letter'), or hand-ruled hedged ('hedged').
  at risk at L = T >= L (still generating when token L was due; same as 19f).  Band [lo, hi): at risk T >= lo,
                 events lo <= T < hi; the last band is [260,000, 327,680] and holds all censored requests.
  reasoning tokens (DERIVED, not measured) = T - content tokens, where content tokens = re-tokenization of the reply
                 content with the K3.25 tokenizer (pure-Python BPE of d7/exposure/render_tokens.py); includes the reasoning
                 close marker and the stop token, and re-tokenization can differ from the generated ids.
"""
import sys
sys.dont_write_bytecode = True  # never write __pycache__ into the read-only source directories
import glob, hashlib, importlib.util, json, math, os, random, statistics, subprocess
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
G = "<gpqa-runs>/"
INV = "<runs>/"
RUNS = "<this-repo>/runs/"
P = "glm53-flash-exl3-k3.25@tpurtell"
CAP = 327680
NQ = 198
THRESHOLDS = [32000, 64000, 100000, 200000]
BANDS = [(0, 16000), (16000, 32000), (32000, 64000), (64000, 100000), (100000, 150000), (150000, 200000),
         (200000, 260000), (260000, CAP + 1)]
GRID = [16000, 32000, 64000, 100000, 150000, 200000, 260000, CAP - 1]
BOOT_B = 2000
BOOT_SEED = 20261013_20  # amendment 20 audit
RUN070 = RUNS + "2026-09-29_glm53-flash_k3.25-v0.7.0-dflash5_gpqa-diamond/results.jsonl"
RUN091 = RUNS + "2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3_gpqa-diamond/results.jsonl"
TOKMAP = INV + "d7/exposure/gpqa_prompt_tokens.json"
ADJ_MOD = INV + "b1-screen/analysis_col511.py"        # frozen copy of the stated-answer extractor (sha b5b13ce2...)
RENDER = INV + "d7/exposure/render_tokens.py"         # pure-Python K3.25 tokenizer, verified against engine prompt counts
TAIL = INV + "tail-analysis/"
# Hand-ruled hedged rows. Published records: <eval-tools>/export_evals.py HEDGED (0.7.0 = recipe '+ps', pass2 docs 25, 94;
# none for stock 0.9.1 '+v0.9.1+ps'). Modified configurations: their adjudication files below.
HEDGED_PUBLISHED = {("stock 0.7.0", 1235): {25, 94}}

# (group, kind, seed, pass_no, request log, samples dir, published results, adjudication file, server log)
PASSES = [
    ("stock 0.7.0", "stock", 1235, 2, P + "+ps", P + "+ps__pass2", RUN070, None, G + P + "+ps.server.log"),
    ("stock 0.7.0", "stock", 1236, 3, P + "+ps", P + "+ps__pass3", RUN070, None, G + P + "+ps.server.log"),
    ("stock 0.9.1", "stock", 1234, 1, P + "+v0.9.1", P + "+v0.9.1+ps__pass1", RUN091, None, G + P + "+v0.9.1.server.log"),
    ("stock 0.9.1", "stock", 1235, 2, P + "+v0.9.1+ps", P + "+v0.9.1+ps__pass2", RUN091, None, G + P + "+v0.9.1+ps.server.log"),
    ("stock 0.9.1", "stock", 1236, 3, P + "+v0.9.1+ps", P + "+v0.9.1+ps__pass3", RUN091, None, G + P + "+v0.9.1+ps.server.log"),
    ("col511 arm", "modified", 1234, 1, P + "+v0.9.1+col511fix.s1", P + "+v0.9.1+col511fix__pass1", None,
     INV + "arm-col511/adjudications.json", INV + "arm-col511/server-s1.log"),
    ("col511 arm", "modified", 1235, 2, P + "+v0.9.1+col511fix.s2", P + "+v0.9.1+col511fix__pass2", None,
     INV + "arm-col511/adjudications.json", INV + "arm-col511/server-s2.log"),
    ("col511 arm", "modified", 1236, 3, P + "+v0.9.1+col511fix.s2", P + "+v0.9.1+col511fix__pass3", None,
     INV + "arm-col511/adjudications.json", INV + "arm-col511/server-s2.log"),
    ("B0", "modified", 1234, 1, P + "+v0.9.1+bundle-d1234.s1", P + "+v0.9.1+bundle-d1234__pass1", None,
     INV + "bundle/screen/run-d1234/adjudication.json", INV + "bundle/screen/run-d1234/server-s1.log"),
    ("B0", "modified", 1235, 2, P + "+v0.9.1+bundle-d1234.s2", P + "+v0.9.1+bundle-d1234__pass2", None,
     INV + "bundle/screen2/run-s1235/adjudication.json", INV + "bundle/screen2/run-s1235/server-s2.log"),
    ("B1", "modified", 1236, 3, P + "+v0.9.1+bundle-d1234+d7.s1", P + "+v0.9.1+bundle-d1234+d7__pass3", None,
     INV + "b1-screen/run-s1236/adjudication.json", INV + "b1-screen/run-s1236/server-s1.log"),
]
# Headline counts already recorded for the modified configurations (empties, raw flexible, stated): reproduced as a check.
EXPECTED = {("B0", 1234): (3, 174, 176), ("B0", 1235): (3, 168, 169), ("B1", 1236): (9, 173, 174),
            ("col511 arm", None): (15, 505, 515)}
# Groups, in the reporting order amendment 20 requires: stock first, then modified separately, then labelled pools.
GROUPS = [
    ("stock 0.7.0", "STOCK", [("stock 0.7.0", None)]),
    ("stock 0.9.1", "STOCK", [("stock 0.9.1", None)]),
    ("stock 0.9.1 seeds 1235-1236 (seed- and session-matched to 0.7.0)", "STOCK", [("stock 0.9.1", 1235), ("stock 0.9.1", 1236)]),
    ("col511 arm", "MODIFIED", [("col511 arm", None)]),
    ("B0", "MODIFIED", [("B0", None)]),
    ("B1", "MODIFIED", [("B1", None)]),
    ("POOLED: stock 0.9.1 + col511 arm (19f '0.9.1 family')", "POOLED", [("stock 0.9.1", None), ("col511 arm", None)]),
    ("POOLED: B0 + B1 (19f 'corrected')", "POOLED", [("B0", None), ("B1", None)]),
    ("POOLED: all modified 0.9.1 configurations (arm + B0 + B1)", "POOLED", [("col511 arm", None), ("B0", None), ("B1", None)]),
]
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append({"check": name, "ok": bool(ok), "detail": str(detail)})
    if not ok:
        print(f"CHECK FAILED: {name} {detail}", file=sys.stderr)


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ADJ = load_module("analysis_col511_ro", ADJ_MOD)
check("stated-answer extractor copy is the frozen one (analysis_col511.py sha256 b5b13ce2...)",
      fsha(ADJ_MOD).startswith("b5b13ce2155085ca"), fsha(ADJ_MOD))
check("extractor block in <eval-tools>/export_evals.py lines 233-259 still has sha256 731f27d4...",
      sha("".join(open("<eval-tools>/export_evals.py").readlines()[232:259])) == ADJ.ADJ_SHA)
RT = load_module("render_tokens_ro", RENDER)
TOK = RT.BPE(RT.TOKJSON)
INPUTS = {}

# ------------------------------------------------------------------------------------------------ load and join
DOC = {d["messages_sha256"]: d["doc_id"] for d in json.load(open(TOKMAP))["docs"]}
check("doc map: 198 distinct messages hashes", len(DOC) == NQ and sorted(DOC.values()) == list(range(NQ)))
INPUTS[TOKMAP] = fsha(TOKMAP)


def load_published(path):
    INPUTS[path] = fsha(path)
    return {(r["doc_id"], r["pass"]): r for r in map(json.loads, open(path))}


PUB = {RUN070: load_published(RUN070), RUN091: load_published(RUN091)}


def hedged_set(group, seed, pass_no, adj_path):
    if adj_path is None:
        return set(HEDGED_PUBLISHED.get((group, seed), set()))
    INPUTS[adj_path] = fsha(adj_path)
    a = json.load(open(adj_path))
    out = set()
    for r in a["rows"]:
        if "pass" in r and r["pass"] != pass_no:
            continue
        if r.get("hedged"):
            out.add(r["doc"])
    return out


def session_info(server_log):
    n = sum(1 for line in open(server_log, errors="replace") if "Application startup complete" in line)
    return n


RECORDS = []
SESSIONS = defaultdict(list)
content_tok_cache = {}
for group, kind, seed, pno, reqname, sdir, pub, adj, slog in PASSES:
    rl = G + reqname + ".requests.jsonl"
    INPUTS[rl] = fsha(rl)
    reqs = [r for r in map(json.loads, open(rl)) if r["seed"] == seed]
    cnt = Counter(r["messages_sha256"] for r in reqs)
    check(f"{group} s{seed}: 198 requests, one per question, HTTP 200, max_tokens {CAP}",
          len(reqs) == NQ and max(cnt.values()) == 1 and set(DOC[m] for m in cnt) == set(range(NQ))
          and all(r["http_status"] == 200 and r["max_tokens"] == CAP for r in reqs), len(reqs))
    sf = glob.glob(G + sdir + "/**/samples_*.jsonl", recursive=True)
    check(f"{group} s{seed}: exactly one samples file in {sdir}", len(sf) == 1, len(sf))
    INPUTS[sf[0]] = fsha(sf[0])
    by = defaultdict(dict)
    for line in open(sf[0]):
        x = json.loads(line)
        by[x["doc_id"]][x["filter"]] = x
    hed = hedged_set(group, seed, pno, adj)
    nstart = session_info(slog)
    INPUTS[slog] = fsha(slog)
    check(f"{group} s{seed}: server log has exactly one engine start ({os.path.basename(slog)})", nstart == 1, nstart)
    SESSIONS[(group, slog)].append((min(r["t_start"] for r in reqs), seed))
    for r in reqs:
        d = DOC[r["messages_sha256"]]
        fx, st = by[d]["flexible-extract"], by[d]["strict-match"]
        txt = str(fx["resps"][0][0]) if fx["resps"] else ""
        empty = not txt.strip()
        letter = None if empty else ADJ.final_answer(txt)
        hedged = d in hed
        stated = (not empty) and not hedged and letter == fx["target"].strip("()")
        check_hash = sha(txt) == r["content_sha256"]
        if not check_hash:
            check(f"{group} s{seed} doc {d}: request-log content hash equals lm-eval reply hash", False)
        if empty:
            ctok = 0
        else:
            if txt not in content_tok_cache:
                content_tok_cache[txt] = len(TOK.encode(txt))
            ctok = content_tok_cache[txt]
        T = r["completion_tokens"]
        capped = r["finish_reason"] == "length"
        if capped:
            check(f"{group} s{seed} doc {d}: length finish is at T == cap", T == CAP, T)
        if capped:
            outcome, sub = "censored", ("censored_content_nonempty" if not empty else "censored_empty")
        elif r["finish_reason"] == "stop" and not empty and letter is not None and not hedged:
            outcome, sub = "valid", "valid"
        elif r["finish_reason"] == "stop":
            outcome, sub = "other", ("empty_stop" if empty else ("hedged" if hedged else "no_letter"))
        else:
            outcome, sub = "other", "finish_" + str(r["finish_reason"])
        rec = {"group": group, "kind": kind, "seed": seed, "pass": pno, "doc": d, "T": T, "finish_reason": r["finish_reason"],
               "outcome": outcome, "outcome_detail": sub, "empty": empty, "letter_extractable": letter is not None,
               "hedged_by_hand": hedged, "flexible_invalid": fx["filtered_resps"][0] == "[invalid]",
               "correct_flexible": bool(fx["exact_match"]), "correct_strict": bool(st["exact_match"]), "correct_stated": stated,
               "prompt_tokens": r["prompt_tokens"], "content_chars": r["content_chars"], "reasoning_chars": r["reasoning_chars"],
               "reasoning_tokens_logged": r["reasoning_tokens"], "content_tokens_retok": ctok, "reply_chars": len(txt),
               "reasoning_tokens_derived": T - ctok, "t_start": r["t_start"], "secs": r["secs"],
               "messages_sha256": r["messages_sha256"], "content_sha256": r["content_sha256"],
               "reqlog": os.path.basename(rl), "server_log": slog}
        if pub is not None:
            pr = PUB[pub][(d, pno)]
            for k in ("empty", "correct_flexible", "correct_strict", "correct_stated", "seed", "finish_reason", "completion_tokens"):
                v = {"completion_tokens": T}.get(k, rec.get(k))
                if pr[k] != v:
                    check(f"{group} s{seed} doc {d}: reproduces published {k}", False, f"{pr[k]} vs {v}")
            if pr["response_sha256"] != r["content_sha256"]:
                check(f"{group} s{seed} doc {d}: published response hash equals request-log content hash", False)
        RECORDS.append(rec)
    n = [x for x in RECORDS if x["group"] == group and x["seed"] == seed]
    check(f"{group} s{seed}: all 198 content hashes match lm-eval replies",
          all(sha(str(by[x['doc']]['flexible-extract']['resps'][0][0])) == x["content_sha256"] for x in n))
    if pub is not None:
        check(f"{group} s{seed}: all 198 rows reproduce the published results.jsonl flags (empty, raw, strict, stated, finish, "
              f"tokens, hash)", not any(c["check"].startswith(f"{group} s{seed} doc") and not c["ok"] for c in CHECKS))

# session position: order passes sharing one server log (one engine start) by first request time
POS = {}
for (group, slog), lst in SESSIONS.items():
    for i, (_, seed) in enumerate(sorted(lst), 1):
        POS[(group, seed)] = (i, len(lst), os.path.basename(slog))
for x in RECORDS:
    i, n, s = POS[(x["group"], x["seed"])]
    x["session_position"] = i
    x["session_passes"] = n
    x["session"] = s

# headline reproduction for the modified configurations
for (g, s), (e, raw, stt) in EXPECTED.items():
    R = [x for x in RECORDS if x["group"] == g and (s is None or x["seed"] == s)]
    got = (sum(x["empty"] for x in R), sum(x["correct_flexible"] for x in R), sum(x["correct_stated"] for x in R))
    check(f"{g}{'' if s is None else ' s' + str(s)}: reproduces recorded empties/raw/stated {(e, raw, stt)}", got == (e, raw, stt), got)

# 0.7.0 pass 1 (seed 1234): no request log; only the published empty flag exists
P070_1 = [r for (d, p), r in PUB[RUN070].items() if p == 1]
UNLOGGED = {"group": "stock 0.7.0", "seed": 1234, "pass": 1, "logged": False, "questions": len(P070_1),
            "empty": sum(r["empty"] for r in P070_1), "empty_docs": sorted(r["doc_id"] for r in P070_1 if r["empty"]),
            "note": "no request log: completion tokens and finish reason unknown; cap exhaustion vs early stop cannot be separated"}


# ------------------------------------------------------------------------------------------------ analysis helpers
def select(members):
    return [x for x in RECORDS if any(x["group"] == g and (s is None or x["seed"] == s) for g, s in members)]


def label(x):
    return f"{x['group']} s{x['seed']} p{x['session_position']}/{x['session_passes']} doc{x['doc']}"


def km(times_events, grid):
    """General Kaplan-Meier: S(t) = prod over distinct event times u <= t of (1 - d_u / n_u). times_events: (T, event)."""
    pts = sorted(times_events)
    out, S, i, n = {}, 1.0, 0, len(pts)
    at_risk = n
    gi = 0
    grid = sorted(grid)
    ev_times = sorted(set(t for t, e in pts if e))
    # walk distinct times
    by_t = defaultdict(lambda: [0, 0])
    for t, e in pts:
        by_t[t][0 if e else 1] += 1
    surv = []
    for t in sorted(by_t):
        d, c = by_t[t]
        if d:
            S *= 1 - d / at_risk
        surv.append((t, S))
        at_risk -= d + c
    for g in grid:
        s = 1.0
        for t, v in surv:
            if t <= g:
                s = v
            else:
                break
        out[g] = s
    return out


def threshold_table(R, axis="T"):
    rows = []
    for L in THRESHOLDS:
        risk = [x for x in R if x[axis] >= L]
        ev = [x for x in risk if x["outcome"] != "censored"]
        rows.append({"L": L, "at_risk": len(risk), "events": len(ev),
                     "valid": sum(x["outcome"] == "valid" for x in ev), "other": sum(x["outcome"] == "other" for x in ev),
                     "censored": sum(x["outcome"] == "censored" for x in risk),
                     "questions_at_risk": sorted(set(x["doc"] for x in risk)),
                     "questions_with_event": sorted(set(x["doc"] for x in ev)),
                     "questions_with_valid": sorted(set(x["doc"] for x in ev if x["outcome"] == "valid")),
                     "other_detail": dict(Counter(x["outcome_detail"] for x in ev if x["outcome"] == "other")),
                     "requests_at_risk": [{"group": x["group"], "seed": x["seed"], "session_position": x["session_position"],
                                           "doc": x["doc"], "T": x["T"], "outcome": x["outcome_detail"]} for x in
                                          sorted(risk, key=lambda x: (x["group"], x["seed"], x["doc"]))]})
    return rows


def band_table(R, axis="T"):
    rows = []
    for lo, hi in BANDS:
        risk = [x for x in R if x[axis] >= lo]
        inb = [x for x in risk if x[axis] < hi]
        ev = [x for x in inb if x["outcome"] != "censored"]
        cen = [x for x in inb if x["outcome"] == "censored"]
        rows.append({"band": [lo, min(hi, CAP)], "at_risk": len(risk), "events": len(ev),
                     "valid": sum(x["outcome"] == "valid" for x in ev), "other": sum(x["outcome"] == "other" for x in ev),
                     "censored_in_band": len(cen), "hazard": (len(ev) / len(risk)) if risk else None,
                     "questions_at_risk": len(set(x["doc"] for x in risk)),
                     "questions_with_event": sorted(set(x["doc"] for x in ev)),
                     "events_detail": [{"group": x["group"], "seed": x["seed"], "session_position": x["session_position"],
                                        "doc": x["doc"], "T": x["T"], "outcome": x["outcome_detail"]} for x in
                                       sorted(ev, key=lambda x: x["T"])] if lo >= 64000 else None,
                     "other_detail": dict(Counter(x["outcome_detail"] for x in ev if x["outcome"] == "other"))})
    return rows


def boot(R, axis="T", B=BOOT_B, seed=BOOT_SEED):
    """Question-clustered bootstrap over all 198 questions (each question carries all its requests in the group).
    Statistics: S(t) on GRID (= KM, since every censoring time is the cap), P(event before cap | at risk at L),
    P(valid final answer before cap | at risk at L)."""
    per = defaultdict(list)
    for x in R:
        per[x["doc"]].append(x)
    # per-question sufficient counts
    Q = {}
    for d in range(NQ):
        xs = per.get(d, [])
        Q[d] = ([sum(x[axis] > g for x in xs) for g in GRID], len(xs),
                [sum(x[axis] >= L for x in xs) for L in THRESHOLDS],
                [sum(x[axis] >= L and x["outcome"] != "censored" for x in xs) for L in THRESHOLDS],
                [sum(x[axis] >= L and x["outcome"] == "valid" for x in xs) for L in THRESHOLDS])
    rng = random.Random(seed)
    S = [[] for _ in GRID]
    PE = [[] for _ in THRESHOLDS]
    PV = [[] for _ in THRESHOLDS]
    undefined = [0] * len(THRESHOLDS)
    for _ in range(B):
        samp = [rng.randrange(NQ) for _ in range(NQ)]
        n = sum(Q[d][1] for d in samp)
        for i in range(len(GRID)):
            S[i].append(sum(Q[d][0][i] for d in samp) / n)
        for j in range(len(THRESHOLDS)):
            den = sum(Q[d][2][j] for d in samp)
            if den == 0:
                undefined[j] += 1
                continue
            PE[j].append(sum(Q[d][3][j] for d in samp) / den)
            PV[j].append(sum(Q[d][4][j] for d in samp) / den)

    def ci(v):
        v = sorted(v)
        return [round(v[int(0.025 * len(v))], 4), round(v[min(len(v) - 1, int(0.975 * len(v)))], 4)] if v else None
    return {"S_ci": {g: ci(S[i]) for i, g in enumerate(GRID)},
            "P_event_given_risk_ci": {L: ci(PE[j]) for j, L in enumerate(THRESHOLDS)},
            "P_valid_given_risk_ci": {L: ci(PV[j]) for j, L in enumerate(THRESHOLDS)},
            "undefined_replicates": dict(zip(THRESHOLDS, undefined)), "B": B, "seed": seed}


def survival(R, axis="T"):
    te = [(x[axis] if x["outcome"] != "censored" else CAP, x["outcome"] != "censored") for x in R]
    S = km(te, GRID)
    emp = {g: sum(x[axis] > g for x in R) / len(R) for g in GRID}
    check(f"KM equals the empirical still-going fraction below the cap (axis {axis}, n={len(R)})",
          all(abs(S[g] - emp[g]) < 1e-12 for g in GRID))
    return S


# ------------------------------------------------------------------------------------------------ build results
res = {"what": "Tail-accounting audit, PLAN.md amendment 20 section 2 (zero GPU). Counts, ids, hashes and flags only.",
       "definitions": __doc__.split("Definitions")[1].strip(), "thresholds": THRESHOLDS, "bands": BANDS, "grid": GRID,
       "cap": CAP, "passes": [], "groups": {}, "unlogged": UNLOGGED}
for group, kind, seed, pno, reqname, sdir, pub, adj, slog in PASSES:
    R = [x for x in RECORDS if x["group"] == group and x["seed"] == seed]
    res["passes"].append({"group": group, "kind": kind, "seed": seed, "pass": pno, "session": POS[(group, seed)][2],
                          "session_position": POS[(group, seed)][0], "session_passes": POS[(group, seed)][1],
                          "requests": len(R), "empty": sum(x["empty"] for x in R),
                          "censored": sum(x["outcome"] == "censored" for x in R),
                          "empty_stop": sum(x["outcome_detail"] == "empty_stop" for x in R),
                          "no_letter_stop": sum(x["outcome_detail"] == "no_letter" for x in R),
                          "hedged_stop": sum(x["outcome_detail"] == "hedged" for x in R),
                          "valid": sum(x["outcome"] == "valid" for x in R),
                          "censored_docs": sorted(x["doc"] for x in R if x["outcome"] == "censored"),
                          "other_docs": sorted(x["doc"] for x in R if x["outcome"] == "other"),
                          "raw_correct": sum(x["correct_flexible"] for x in R), "stated_correct": sum(x["correct_stated"] for x in R),
                          "tokens_total": sum(x["T"] for x in R),
                          "threshold": {t["L"]: [t["at_risk"], t["valid"], t["other"], t["censored"]] for t in threshold_table(R)}})
for name, cls, members in GROUPS:
    R = select(members)
    g = {"class": cls, "members": members, "requests": len(R), "passes": len({(x["group"], x["seed"]) for x in R}),
         "thresholds": threshold_table(R), "bands": band_table(R), "S": survival(R), "boot": boot(R),
         "thresholds_reasoning_axis": [{k: v for k, v in t.items() if k in ("L", "at_risk", "events", "valid", "other", "censored")}
                                       for t in threshold_table(R, "reasoning_tokens_derived")],
         "S_reasoning_axis": survival(R, "reasoning_tokens_derived"),
         "boot_reasoning_axis": boot(R, "reasoning_tokens_derived")}
    res["groups"][name] = g

# stock-vs-stock differences, one joint question-clustered bootstrap (the same resampled questions for both sides)
def boot_diff(Ra, Rb, B=BOOT_B, seed=BOOT_SEED + 1):
    def suff(R):
        per = defaultdict(list)
        for x in R:
            per[x["doc"]].append(x)
        return {d: ([sum(x["T"] > g for x in per.get(d, [])) for g in GRID], len(per.get(d, [])),
                    [sum(x["T"] >= L for x in per.get(d, [])) for L in THRESHOLDS],
                    [sum(x["T"] >= L and x["outcome"] != "censored" for x in per.get(d, [])) for L in THRESHOLDS])
                for d in range(NQ)}
    A, Bq = suff(Ra), suff(Rb)
    def stats(samp):
        out = []
        for Q in (A, Bq):
            n = sum(Q[d][1] for d in samp)
            s = [sum(Q[d][0][i] for d in samp) / n for i in range(len(GRID))]
            p = []
            for j in range(len(THRESHOLDS)):
                den = sum(Q[d][2][j] for d in samp)
                p.append(sum(Q[d][3][j] for d in samp) / den if den else None)
            out.append((s, p))
        return out
    pt = stats(list(range(NQ)))
    rng = random.Random(seed)
    dS = [[] for _ in GRID]
    dP = [[] for _ in THRESHOLDS]
    for _ in range(B):
        (sa, pa), (sb, pb) = stats([rng.randrange(NQ) for _ in range(NQ)])
        for i in range(len(GRID)):
            dS[i].append(sa[i] - sb[i])
        for j in range(len(THRESHOLDS)):
            if pa[j] is not None and pb[j] is not None:
                dP[j].append(pa[j] - pb[j])
    def ci(v):
        v = sorted(v)
        return [round(v[int(0.025 * len(v))], 4), round(v[min(len(v) - 1, int(0.975 * len(v)))], 4)] if v else None
    (sa, pa), (sb, pb) = pt
    return {"S_diff": {g: [round(sa[i] - sb[i], 4), ci(dS[i])] for i, g in enumerate(GRID)},
            "P_event_given_risk_diff": {L: [round(pa[j] - pb[j], 4), ci(dP[j]), B - len(dP[j])] for j, L in enumerate(THRESHOLDS)},
            "B": B, "seed": seed, "note": "first minus second; [estimate, 95% interval, undefined replicates]"}


res["stock_differences"] = {
    "stock 0.7.0 minus stock 0.9.1 (all logged seeds)": boot_diff(select([("stock 0.7.0", None)]), select([("stock 0.9.1", None)])),
    "stock 0.7.0 minus stock 0.9.1 (seeds 1235-1236)": boot_diff(select([("stock 0.7.0", None)]),
                                                                 select([("stock 0.9.1", 1235), ("stock 0.9.1", 1236)])),
}

# question as the unit, stock only (19f used stock + arm on the 0.9.1 side)
def qfrac(R, L=100000):
    out = defaultdict(lambda: [0, 0, 0])
    for x in R:
        if x["T"] >= L:
            out[x["doc"]][0] += x["outcome"] != "censored"
            out[x["doc"]][1] += x["outcome"] == "valid"
            out[x["doc"]][2] += 1
    return out


QU = {}
for lab, a_m, b_m in [("stock 0.7.0 vs stock 0.9.1 (all logged seeds)", [("stock 0.7.0", None)], [("stock 0.9.1", None)]),
                      ("stock 0.7.0 vs stock 0.9.1, seeds 1235-1236 only", [("stock 0.7.0", None)],
                       [("stock 0.9.1", 1235), ("stock 0.9.1", 1236)]),
                      ("19f replica: stock 0.7.0 vs stock 0.9.1 + col511 arm", [("stock 0.7.0", None)],
                       [("stock 0.9.1", None), ("col511 arm", None)])]:
    for L in (64000, 100000):
        a, b = qfrac(select(a_m), L), qfrac(select(b_m), L)
        common = sorted(set(a) & set(b))
        diffs = [a[d][0] / a[d][2] - b[d][0] / b[d][2] for d in common]
        QU[f"{lab} @ {L}"] = {"common_questions": len(common),
                              "per_question": {d: {"a_event_valid_risk": a[d], "b_event_valid_risk": b[d]} for d in common},
                              "a_higher": sum(v > 0 for v in diffs), "a_lower": sum(v < 0 for v in diffs),
                              "equal": sum(v == 0 for v in diffs),
                              "mean_diff": round(statistics.mean(diffs), 3) if diffs else None,
                              "only_a": sorted(set(a) - set(b)), "only_b": sorted(set(b) - set(a))}
res["question_unit"] = QU

# session position: within-engine first vs second pass of a session (descriptive)
SP = {}
for grp in ("stock 0.7.0", "stock 0.9.1", "col511 arm"):
    for pos in (1, 2):
        R = [x for x in RECORDS if x["group"] == grp and x["session_position"] == pos]
        if not R:
            continue
        SP[f"{grp} session position {pos}"] = {
            "passes": sorted({x["seed"] for x in R}), "requests": len(R),
            "at_risk_100k": sum(x["T"] >= 100000 for x in R),
            "events_100k": sum(x["T"] >= 100000 and x["outcome"] != "censored" for x in R),
            "censored": sum(x["outcome"] == "censored" for x in R)}
res["session_position"] = SP

# reasoning fields
rt_types = Counter(type(x["reasoning_tokens_logged"]).__name__ for x in RECORDS)
fin = [x for x in RECORDS if x["outcome"] != "censored" and not x["empty"]]
ct = sorted(x["content_tokens_retok"] for x in fin)
moved = defaultdict(int)
for x in RECORDS:
    for L in THRESHOLDS:
        if (x["T"] >= L) != (x["reasoning_tokens_derived"] >= L):
            moved[L] += 1
cen = [x for x in RECORDS if x["outcome"] == "censored"]
res["reasoning_fields"] = {
    "reasoning_tokens_logged_types": dict(rt_types),
    "reasoning_chars_populated": all(isinstance(x["reasoning_chars"], int) for x in RECORDS),
    "content_chars_populated": all(isinstance(x["content_chars"], int) for x in RECORDS),
    "content_chars_equal_to_lm_eval_reply_length": sum(x["content_chars"] == x["reply_chars"] for x in RECORDS),
    "requests": len(RECORDS),
    "content_tokens_retok_finished_nonempty": {"n": len(ct), "min": ct[0], "median": statistics.median(ct),
                                               "p90": ct[int(0.9 * (len(ct) - 1))], "max": ct[-1]},
    "requests_changing_threshold_status_on_reasoning_axis": dict(moved),
    "censored_with_nonempty_content": sum(not x["empty"] for x in cen),
    "censored_reasoning_chars": {"min": min(x["reasoning_chars"] for x in cen), "max": max(x["reasoning_chars"] for x in cen),
                                 "median": statistics.median(x["reasoning_chars"] for x in cen)} if cen else None,
    "censored_content_chars_max": max(x["content_chars"] for x in cen) if cen else None,
}

# ------------------------------------------------------------------------------------------------ 19f re-derivation
REP = {}
p = subprocess.run([sys.executable, "-B", TAIL + "tail_analysis.py"], capture_output=True, text=True)
REP["rerun_stdout_identical_to_tail_analysis_out"] = p.stdout == open(TAIL + "tail_analysis.out").read()
REP["tail_analysis_py_sha256"] = fsha(TAIL + "tail_analysis.py")
REP["tail_analysis_out_sha256"] = fsha(TAIL + "tail_analysis.out")
check("19f script re-run reproduces tail_analysis.out byte for byte", REP["rerun_stdout_identical_to_tail_analysis_out"])


def replay_19f_bootstrap():
    """Re-implements tail_analysis.py's pooled() resampling scheme to count NaN replicates (den == 0), which its percentile
    sort would keep (NaN in a Python sort leaves the order undefined). 19f resamples only questions that reached L."""
    data = {}
    PASS19 = [("0.7.0", 1235, P + "+ps"), ("0.7.0", 1236, P + "+ps"), ("stock 0.9.1", 1234, P + "+v0.9.1"),
              ("stock 0.9.1", 1235, P + "+v0.9.1+ps"), ("stock 0.9.1", 1236, P + "+v0.9.1+ps"),
              ("col511 arm", 1234, P + "+v0.9.1+col511fix.s1"), ("col511 arm", 1235, P + "+v0.9.1+col511fix.s2"),
              ("col511 arm", 1236, P + "+v0.9.1+col511fix.s2"), ("B0", 1234, P + "+v0.9.1+bundle-d1234.s1"),
              ("B0", 1235, P + "+v0.9.1+bundle-d1234.s2"), ("B1", 1236, P + "+v0.9.1+bundle-d1234+d7.s1")]
    for e, s, n in PASS19:
        out = {}
        for x in map(json.loads, open(G + n + ".requests.jsonl")):
            if x["seed"] == s:
                out[DOC[x["messages_sha256"]]] = (x["completion_tokens"], x["finish_reason"] == "length")
        data[(e, s)] = out
    FAM = {"0.7.0": ["0.7.0"], "0.9.1 family (stock+arm)": ["stock 0.9.1", "col511 arm"], "corrected (B0+B1)": ["B0", "B1"],
           "stock 0.9.1": ["stock 0.9.1"]}
    rng = random.Random()
    rng.seed(20261013)
    out = {}
    for L in THRESHOLDS:
        for name, eng in FAM.items():
            R = [(e, s, d, t, c) for (e, s), r in data.items() if e in eng for d, (t, c) in r.items() if t >= L]
            byq = defaultdict(list)
            for e, s, d, t, c in R:
                byq[d].append(0 if c else 1)
            qs = list(byq)
            nan = 0
            bs = []
            for _ in range(4000):
                smp = [rng.choice(qs) for _ in qs]
                den = sum(len(byq[q]) for q in smp)
                if den == 0:
                    nan += 1
                bs.append(sum(sum(byq[q]) for q in smp) / den if den else float("nan"))
            out[f"{name} @ {L}"] = {"nan_replicates": nan, "questions_resampled": len(qs)}
    return out


REP["bootstrap_nan_replicates"] = replay_19f_bootstrap()


def recount_19f():
    """19f's quantities recounted from RECORDS with this audit's definitions."""
    out = {}
    fam = [("stock 0.9.1", None), ("col511 arm", None)]
    R = select(fam)
    # '6 of about 100': 19f summed band at-risk counts (150-200k 36, 200-260k 33, 260-327k 31) = 100 request-bands
    out["0.9.1 family: unique requests at risk at 150k"] = sum(x["T"] >= 150000 for x in R)
    out["0.9.1 family: finishing events at >= 150k"] = sum(x["T"] >= 150000 and x["outcome"] != "censored" for x in R)
    out["0.9.1 family: valid final answers at >= 150k"] = sum(x["T"] >= 150000 and x["outcome"] == "valid" for x in R)
    out["0.9.1 family: sum of band at-risk counts 150k-cap (19f's 'about 100')"] = sum(
        sum(x["T"] >= lo for x in R) for lo, hi in BANDS if lo >= 150000)
    for nm, mem in [("stock 0.9.1", [("stock 0.9.1", None)]), ("stock 0.7.0", [("stock 0.7.0", None)])]:
        RR = select(mem)
        out[f"{nm}: unique at risk at 150k / events beyond"] = [sum(x["T"] >= 150000 for x in RR),
                                                                  sum(x["T"] >= 150000 and x["outcome"] != "censored" for x in RR)]
    # 'finished' in 19f = not length; split into valid vs other at each threshold, for every 19f group
    fams = {"0.7.0": [("stock 0.7.0", None)], "0.9.1 family (stock+arm)": fam, "corrected (B0+B1)": [("B0", None), ("B1", None)],
            "stock 0.9.1": [("stock 0.9.1", None)]}
    for nm, mem in fams.items():
        RR = select(mem)
        for L in THRESHOLDS:
            risk = [x for x in RR if x["T"] >= L]
            out[f"{nm} @ {L}: reached / finished(19f) / valid / other"] = [
                len(risk), sum(x["outcome"] != "censored" for x in risk), sum(x["outcome"] == "valid" for x in risk),
                sum(x["outcome"] == "other" for x in risk)]
    return out


REP["recount"] = recount_19f()
res["rederivation_19f"] = REP
res["checks"] = CHECKS
res["valid"] = all(c["ok"] for c in CHECKS)
res["inputs_sha256"] = INPUTS
res["records"] = [{k: v for k, v in x.items() if k not in ("server_log",)} for x in RECORDS]
json.dump(res, open(os.path.join(HERE, "audit.json"), "w"), indent=1, default=str)

# ------------------------------------------------------------------------------------------------ markdown tables
O = []
w = O.append


def fmtci(c, nd=2):
    return f"[{c[0]:.{nd}f}, {c[1]:.{nd}f}]" if c else "n/a"


w(f"checks: {sum(c['ok'] for c in CHECKS)}/{len(CHECKS)} passed; valid = {res['valid']}\n")
w("### Per pass (logged passes; session position = order within one server session)\n")
w("| configuration | seed | lm-eval pass | session (server log) | position | empty | censored at cap | empty stop | no-letter stop | hedged stop | valid | at risk / valid / other / censored at 100k | at 200k |")
w("|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|")
for q in res["passes"]:
    t1, t2 = q["threshold"][100000], q["threshold"][200000]
    w(f"| {q['group']} | {q['seed']} | {q['pass']} | {q['session']} | {q['session_position']} of {q['session_passes']} | {q['empty']} | "
      f"{q['censored']} | {q['empty_stop']} | {q['no_letter_stop']} | {q['hedged_stop']} | {q['valid']} | "
      f"{t1[0]} / {t1[1]} / {t1[2]} / {t1[3]} | {t2[0]} / {t2[1]} / {t2[2]} / {t2[3]} |")
w(f"| stock 0.7.0 | 1234 | 1 | unlogged | - | {UNLOGGED['empty']} | ? | ? | - | - | - | unlogged | unlogged |")
w("")
for name, g in res["groups"].items():
    w(f"### {name}  [{g['class']}; {g['passes']} passes, {g['requests']} requests]\n")
    w("Thresholds (T = logged completion tokens; at risk = T >= L):\n")
    w("| L | unique requests at risk | questions at risk | finishing events | valid final answer | other stop | censored at cap | P(event before cap \\| at risk) [95% q-boot] | P(valid \\| at risk) [95% q-boot] |")
    w("|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for t in g["thresholds"]:
        L = t["L"]
        pe = t["events"] / t["at_risk"] if t["at_risk"] else float("nan")
        pv = t["valid"] / t["at_risk"] if t["at_risk"] else float("nan")
        w(f"| {L:,} | {t['at_risk']} | {len(t['questions_at_risk'])} | {t['events']} | {t['valid']} | {t['other']} | {t['censored']} | "
          f"{pe:.2f} {fmtci(g['boot']['P_event_given_risk_ci'][L])} | {pv:.2f} {fmtci(g['boot']['P_valid_given_risk_ci'][L])} |")
    w("\nQuestion identities (doc ids):\n")
    for t in g["thresholds"]:
        if t["L"] >= 64000:
            w(f"- >= {t['L']:,}: at risk {t['questions_at_risk']}; with a valid finish {t['questions_with_valid']}"
              + (f"; other stops {t['other_detail']}" if t["other"] else ""))
    w("\nBands (at risk = still generating at band start; events end inside the band):\n")
    w("| band (tokens) | at risk | events | valid | other | censored in band | conditional fraction ending in band | questions with an event |")
    w("|---|---:|---:|---:|---:|---:|---:|---|")
    for b in g["bands"]:
        lo, hi = b["band"]
        qs = b["questions_with_event"] if lo >= 64000 else f"{len(b['questions_with_event'])} questions"
        w(f"| {lo:,}-{hi:,} | {b['at_risk']} | {b['events']} | {b['valid']} | {b['other']} | {b['censored_in_band']} | "
          f"{(b['hazard'] if b['hazard'] is not None else float('nan')):.2f} | {qs} |")
    w("\nStill generating, S(t) = P(T > t), Kaplan-Meier with censoring at the cap, 95% question-clustered bootstrap "
      f"(B = {BOOT_B}, all 198 questions resampled):\n")
    w("| t | " + " | ".join(f"{x // 1000}k" if x < CAP - 1 else "cap-" for x in GRID) + " |")
    w("|---|" + "---|" * len(GRID))
    w("| S(t), completion tokens (measured) | " + " | ".join(f"{g['S'][x]:.3f} {fmtci(g['boot']['S_ci'][x], 3)}" for x in GRID) + " |")
    w("| S(t), reasoning tokens (derived) | " + " | ".join(
        f"{g['S_reasoning_axis'][x]:.3f} {fmtci(g['boot_reasoning_axis']['S_ci'][x], 3)}" for x in GRID) + " |")
    w("")
w("### Requests at risk at 100k, every configuration (ids only)\n")
w("| configuration | seed | session position | doc | T (completion tokens) | outcome | reasoning tokens (derived) | content tokens (re-tokenized) | reasoning chars |")
w("|---|---:|---|---:|---:|---|---:|---:|---:|")
for x in sorted([x for x in RECORDS if x["T"] >= 100000], key=lambda x: (x["kind"] != "stock", x["group"], x["seed"], x["T"])):
    w(f"| {x['group']} | {x['seed']} | {x['session_position']} of {x['session_passes']} | {x['doc']} | {x['T']:,} | {x['outcome_detail']} | "
      f"{x['reasoning_tokens_derived']:,} | {x['content_tokens_retok']} | {x['reasoning_chars']:,} |")
w("### Stock 0.7.0 minus stock 0.9.1, joint question-clustered bootstrap (same resampled questions on both sides)\n")
w("| comparison | " + " | ".join(f"P(event \\| risk {L // 1000}k)" for L in THRESHOLDS) + " | " + " | ".join(
    f"S({x // 1000}k)" if x < CAP - 1 else "S(cap-)" for x in GRID if x in (100000, 200000, CAP - 1)) + " |")
w("|---|" + "---|" * (len(THRESHOLDS) + 3))
for k, v in res["stock_differences"].items():
    w(f"| {k} | " + " | ".join(f"{v['P_event_given_risk_diff'][L][0]:+.2f} {fmtci(v['P_event_given_risk_diff'][L][1])}" for L in THRESHOLDS)
      + " | " + " | ".join(f"{v['S_diff'][x][0]:+.3f} {fmtci(v['S_diff'][x][1], 3)}" for x in (100000, 200000, CAP - 1)) + " |")
w("\n### Question as the unit (per question: events / valid / at risk)\n")
for k, v in QU.items():
    w(f"- {k}: {v['common_questions']} common questions; first side higher on {v['a_higher']}, lower on {v['a_lower']}, equal on "
      f"{v['equal']}; mean difference of event fractions {v['mean_diff']}; only first side {v['only_a']}; only second side {v['only_b']}")
    if "@ 100000" in k and "19f" not in k:
        for d, pq in v["per_question"].items():
            w(f"    - doc {d}: {pq['a_event_valid_risk'][0]}/{pq['a_event_valid_risk'][1]}/{pq['a_event_valid_risk'][2]} vs "
              f"{pq['b_event_valid_risk'][0]}/{pq['b_event_valid_risk'][1]}/{pq['b_event_valid_risk'][2]}")
w("\n### Session position (descriptive)\n")
for k, v in SP.items():
    w(f"- {k}: seeds {v['passes']}, {v['requests']} requests, at risk at 100k {v['at_risk_100k']}, events {v['events_100k']}, censored {v['censored']}")
w("\n### Reasoning fields\n")
w("```\n" + json.dumps(res["reasoning_fields"], indent=1) + "\n```")
w("\n### 19f re-derivation\n")
w("```\n" + json.dumps({k: v for k, v in REP.items()}, indent=1) + "\n```")
w("\n### Checks\n")
for c in CHECKS:
    if not c["ok"] or "doc" not in c["check"]:
        w(f"- [{'ok' if c['ok'] else 'FAIL'}] {c['check']} {c['detail'] if not c['ok'] else ''}")
open(os.path.join(HERE, "audit.out"), "w").write("\n".join(O) + "\n")
print("valid" if res["valid"] else "INVALID", f"{sum(c['ok'] for c in CHECKS)}/{len(CHECKS)} checks")
