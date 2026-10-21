#!/usr/bin/env python3
"""Frozen analysis of option D (PROPOSAL.md sections 4-6, amendment 22x): ONE stock tpurtell 0.9.1 GPQA Diamond pass on
the standard protocol at the fresh request seed 70704. Descriptive only: no outcome test against the records.

Reuse (byte-identical copies, hash-checked below):
  analysis_rep070.py      (rep070/refreeze, sha256 54c7fa4f...): warm-up checks, task hash, long-reply secondary
                          (tail_units / tail_summary / tail_diff), its RECORDS entry for the 0.9.1 record;
  frozen/analysis_col511.py (sha256 b5b13ce2..., imported by analysis_rep070): the loader (rows, stated adjudicator,
                          per-pass validity checks), parse_serverlogs, parse_chainlogs, inflight, quantiles.
The one pass is loaded exactly as rep070's variant-F loader loads each of its passes: the frozen module's SEED_BASE is set
to 70704 - 1 so that pass 1 is matched to request seed 70704, only pass 1's rows and checks are kept, and SEED_BASE is
restored on every exit path.

Gates (any failure = INVALID, exit 3; never a reading of an invalid run). Configuration-identity gates accept exactly the
set of values the 0.9.1 record's own two server sessions produced (checks/gate-calibration.json shows each gate applied to
the record's sessions and to the B+ v091 session): non-default args, vLLM version, model snapshots, logged env names,
CUDA-graph mode, release warm-up count. The KV pool is REPORTED, not identity-gated (PROPOSAL.md section 4.3); a
functional floor is gated instead (8 x the cap plus the record's longest prompt, rounded up to whole attention blocks,
fits in the pool, so the protocol's 8 in flight can never be KV-limited).

usage: analysis_d091.py --pass-dir D --reqlog L --serverlog S [--inspect I] [--chainlog C ...] [--gpu-hours X]
         [--hedged 1:DOC ...] [--rows-out ROWS.jsonl] [--dry-run-label TEXT] --out OUT.json > OUT.md
Exit 0 valid, 3 invalid. Outputs ids, hashes, counts and token lengths only (no question, answer, letter or reply text).
"""
import argparse, glob, hashlib, json, math, os, random, statistics, sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PINS = {"analysis_rep070.py": "54c7fa4f850a4802178931911a3ad665dd3438f6e3ab7c3c65f9d0aec2940410",
        "frozen/analysis_col511.py": "b5b13ce2155085ca46fb24877f12b7cd0a5cdd16eb4cf19012958e7b86ae0033",
        "checks/seed-inventory.json": "1f58e7b3747b769133e112bfbe64c45890a1812742e1526d682bcefcda931c8e"}   # seed inventory frozen 2026-10-21
for _f, _s in PINS.items():
    _h = hashlib.sha256(open(os.path.join(HERE, _f), "rb").read()).hexdigest()
    if _h != _s:
        sys.exit(f"pinned file changed: {_f} {_h}")
sys.path.insert(0, HERE)
import analysis_rep070 as X  # noqa: E402  (asserts the frozen module's sha256 on import)

M = X.M
SEED, PASS, PROCESS_SEED = 70704, 1, 1235
TAG = "+d091"
G = "<gpqa-runs>"
REC = X.RECORDS["v091"]                       # the 0.9.1 three-pass record (pass dirs, request logs, server logs, published rows)
REC_PASS1_SERVERLOG = REC["serverlogs"][0]    # session of record pass 1 (2026-10-08)
REPF = dict(label="stock 0.7.0 rep070-F (2026-10-16)", passes=[f"{G}/glm53-flash-exl3-k3.25@tpurtell+rep070-F__pass{p}" for p in (1, 2, 3)],
            reqlogs=[f"{G}/glm53-flash-exl3-k3.25@tpurtell+rep070-F.s1.requests.jsonl", f"{G}/glm53-flash-exl3-k3.25@tpurtell+rep070-F.s2.requests.jsonl"],
            seeds=(70701, 70702, 70703),
            final="<runs>/rep070/run-F/result-rep070-F-final.json",
            final_sha="a764806dbb8ff6e5ef013199bcd2a40f9347189058f63a766b210f1b1025188c")
IMG_ID = "sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07"
IDENTITY_KEYS = ("non_default_args_sha256", "vllm_version", "model_snapshots", "env_names_logged_sha256", "cudagraph_mode")
ATTN_BLOCK = 4352                             # 'Setting attention block size to 4352 tokens' (0.9.1 server logs)
CONCURRENCY = 8
L_SEC, L_DESC = X.L_SEC, X.L_DESC             # 100,000 (prespecified) and 200,000 (descriptive)
L_CONTEXT = 50000                             # descriptive context only (the discovery pipeline's selection threshold)
BOOT_B, BOOT_SEED = X.BOOT_B_SEC, 20261021    # question bootstrap for single-group P intervals (2,000 replicates)


def fsha(p):
    return M.file_sha(p)


def used_seeds():
    inv = json.load(open(os.path.join(HERE, "checks/seed-inventory.json")))["summary"]
    return set(inv["gpqa_seeds_used"]) | set(inv["request_seeds_any_prompt"])


def load_one_pass(label, d, reqlogs, hedged, seed=SEED):
    """rep070 variant-F loader for one pass: SEED_BASE = seed - 1, keep pass 1 only, restore SEED_BASE on every path."""
    old = M.SEED_BASE
    try:
        M.SEED_BASE = seed - PASS
        r, c, i = M.load_arm(label, [d, d, d], reqlogs, hedged, None)
    finally:
        M.SEED_BASE = old
    rows = {k: v for k, v in r.items() if k[1] == PASS}
    checks = [x for x in c if x[0].startswith(f"{label} pass {PASS}:") or x[0].startswith(f"{label} pass {PASS} q")]
    return rows, checks, {"samples": i["samples"].get(f"pass{PASS}"), "reqlogs": i["reqlogs"]}


def results_cfg(d):
    res = glob.glob(os.path.join(d, "**", "results_*.json"), recursive=True)
    if len(res) != 1:
        return None
    cfg = json.load(open(res[0])).get("config", {})
    ma = cfg.get("model_args") or {}
    if isinstance(ma, str):
        ma = dict(kv.split("=", 1) for kv in ma.split(",") if "=" in kv)
    return {"model_args": ma, "random_seed": cfg.get("random_seed"), "gen_kwargs": cfg.get("gen_kwargs")}


# ------------------------------------------------------------------ gates shared with checks/gate_calibration.py
def reference_values(ref_logs):
    """Accepted set per identity key = the values the reference sessions produced (one entry per distinct value)."""
    acc = {k: [] for k in IDENTITY_KEYS}
    info = []
    for p in ref_logs:
        s = M.parse_serverlogs([p], [], 0)[0]
        info.append(s)
        for k in IDENTITY_KEYS:
            if s.get(k) not in acc[k]:
                acc[k].append(s.get(k))
    acc["release_warmups_logged"] = sorted({s.get("release_warmups_logged") for s in info})
    acc["kv_pool_tokens"] = [list(x) for x in sorted({tuple(s.get("kv_pool_tokens") or []) for s in info})]
    return acc, info


def max_prompt_tokens(reqlogs):
    return max(r["prompt_tokens"] for r in M.load_reqlogs(reqlogs) if r.get("prompt_tokens") is not None)


def kv_floor(max_prompt):
    """Pool needed for 8 requests at the cap with the longest record prompt, each rounded up to whole attention blocks
    (+16 tokens for speculative lookahead)."""
    return CONCURRENCY * ATTN_BLOCK * math.ceil((M.CAP + max_prompt + 16) / ATTN_BLOCK)


def config_gates(serverlog, ref_logs, label, floor):
    acc, _ = reference_values(ref_logs)
    s = M.parse_serverlogs([serverlog], [], 0)[0]
    out = []
    for k in IDENTITY_KEYS:
        out.append((f"{label}: {k} is a value the 0.9.1 record's own server sessions produced", s.get(k) in acc[k],
                    f"{str(s.get(k))[:24]} in {[str(v)[:12] for v in acc[k]]}"))
    out.append((f"{label}: release warm-up count is one the record's sessions logged", s.get("release_warmups_logged") in acc["release_warmups_logged"],
                f"{s.get('release_warmups_logged')} in {acc['release_warmups_logged']}"))
    kv = s.get("kv_pool_tokens") or []
    out.append((f"{label}: KV pool >= functional floor {floor:,} tokens (8 x cap + longest record prompt, whole blocks; not an identity gate)",
                len(kv) == 1 and kv[0] >= floor, f"{kv} vs floor {floor}"))
    info = {"kv_pool_tokens": kv, "kv_pool_equals_a_record_size": list(kv) in acc["kv_pool_tokens"],
            "record_kv_pool_sizes": acc["kv_pool_tokens"], "kv_floor": floor, "kv_margin_tokens": (kv[0] - floor) if len(kv) == 1 else None}
    return out, info


def health_gates(serverlog, window, label):
    s = M.parse_serverlogs([serverlog], [window] if window else [], 0)[0]
    iw = s.get("in_window") or {}
    out = [(f"{label}: no illegal-memory-access and no NaN warning lines during the requests (the record's sessions: 0 and 0)",
            iw.get("illegal_memory_access_lines", 1) == 0 and iw.get("nan_warning_lines", 1) == 0, iw)]
    return out, s


def window_of(rows):
    ts = [(r["t_start"], r["t_start"] + r["secs"]) for r in rows.values() if r["t_start"] is not None and r["secs"] is not None]
    return (min(a for a, _ in ts), max(b for _, b in ts)) if ts else None


# ------------------------------------------------------------------ descriptive pieces
def boot_p(per, j=0, B=BOOT_B, seed=BOOT_SEED):
    """P(finish | at risk) with a question-bootstrap percentile interval (all requests of a question move together)."""
    allq = list(range(M.N_Q))
    def P(idx):
        den = sum(per[q][j][0] for q in idx)
        return (sum(per[q][j][1] for q in idx) / den) if den else None
    est = P(allq)
    rng = random.Random(seed)
    xs, undef = [], 0
    for _ in range(B):
        v = P([rng.randrange(M.N_Q) for _ in range(M.N_Q)])
        if v is None:
            undef += 1
        else:
            xs.append(v)
    xs.sort()
    ci = [round(xs[int(0.025 * len(xs))], 4), round(xs[min(len(xs) - 1, int(0.975 * len(xs)))], 4)] if xs else None
    return {"estimate": round(est, 4) if est is not None else None, "ci95": ci, "undefined_replicates": undef, "B": B, "seed": seed}


def pass_summary(rows):
    R = list(rows.values())
    fin = {}
    for r in R:
        fin[str(r["finish_reason"])] = fin.get(str(r["finish_reason"]), 0) + 1
    cats = {"cap_exhaustion": 0, "premature_stop": 0, "other_finish": 0, "not_logged": 0}
    for r in R:
        if not r["empty"]:
            continue
        if not r["logged"] or r["finish_reason"] is None:
            cats["not_logged"] += 1
        elif r["finish_reason"] == "length" or (r["completion_tokens"] or 0) >= M.CAP:
            cats["cap_exhaustion"] += 1
        elif r["finish_reason"] == "stop":
            cats["premature_stop"] += 1
        else:
            cats["other_finish"] += 1
    ans = [r["completion_tokens"] for r in R if r["logged"] and not r["empty"] and r["completion_tokens"] is not None]
    allt = [r["completion_tokens"] for r in R if r["logged"] and r["completion_tokens"] is not None]
    n = len(R)
    return {"questions": n, "empty": sum(r["empty"] for r in R), "raw_correct_flexible": sum(r["correct_flexible"] for r in R),
            "stated_correct": sum(r["correct_stated"] for r in R), "strict_correct": sum(r["correct_strict"] for r in R),
            "raw_pct": round(100 * sum(r["correct_flexible"] for r in R) / n, 2) if n else None,
            "stated_pct": round(100 * sum(r["correct_stated"] for r in R) / n, 2) if n else None,
            "finish_reasons": fin, "empties_by_end": cats,
            "answered_but_length_finish": sum(1 for r in R if r["logged"] and not r["empty"] and r["finish_reason"] == "length"),
            "completion_tokens_answered": M.quantiles(ans), "completion_tokens_all": M.quantiles(allt),
            "completion_tokens_total": sum(allt),
            "replies_ge": {str(x): sum(v >= x for v in allt) for x in (L_CONTEXT, L_SEC) + L_DESC},
            "capped_at_327680": sum(1 for r in R if r["finish_reason"] == "length" and r["completion_tokens"] == M.CAP),
            "stated_differs_from_flexible_ids": [f"q{r['doc_id']}p{r['pass']}" for r in sorted(R, key=lambda r: r["doc_id"]) if r["stated_differs_from_flexible"]],
            "hedged_by_hand": sum(r["hedged_by_hand"] for r in R)}


ROW_KEYS = ("completion_tokens", "correct_flexible", "correct_stated", "correct_strict", "doc_hash", "doc_id", "empty",
            "finish_reason", "pass", "prompt_hash", "response_chars", "response_sha256", "seed", "target_hash")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-dir", required=True)
    ap.add_argument("--reqlog", required=True)
    ap.add_argument("--serverlog", required=True)
    ap.add_argument("--inspect", default=None)
    ap.add_argument("--chainlog", nargs="*", default=[])
    ap.add_argument("--gpu-hours", type=float, default=None)
    ap.add_argument("--hedged", nargs="*", default=[], help="1:DOC replies read by hand as committing to no single option")
    ap.add_argument("--rows-out", default=None)
    ap.add_argument("--dry-run-label", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    label = "stock 0.9.1 d091 (seed 70704)"
    checks = []
    # ---- the pass, through the frozen loader
    reqlogs = [a.reqlog] if os.path.isfile(a.reqlog) else []
    checks.append(("proxy request log present", bool(reqlogs), os.path.basename(a.reqlog)))
    have_srv = os.path.isfile(a.serverlog)
    checks.append(("server log present", have_srv, os.path.basename(a.serverlog)))
    A, ca, ia = load_one_pass(label, a.pass_dir, reqlogs, a.hedged)
    checks += ca
    checks.append((f"{label}: 198 question rows, every one with its proxy row (finish reason and token counts)",
                   len(A) == M.N_Q and all(r["logged"] for r in A.values()), f"{len(A)} rows, {sum(r['logged'] for r in A.values())} logged"))
    # ---- the records
    V, cv, iv = M.load_arm(REC["label"], REC["passes"], REC["reqlogs"], REC["hedged"], REC["published"])
    checks += cv
    F, cf, if_ = X.load_rep(REPF["label"], REPF["passes"], REPF["reqlogs"], [], None, REPF["seeds"])
    checks += cf
    # ---- seeds
    rows_log = M.load_reqlogs(reqlogs)
    cnt = {}
    for r in rows_log:
        cnt[r.get("seed")] = cnt.get(r.get("seed"), 0) + 1
    used = used_seeds()
    checks.append(("request seed 70704 is fresh: in none of the frozen inventory's GPQA or request seeds (checks/seed-inventory.json)",
                   SEED not in used, f"{len(used)} used seeds"))
    checks.append(("request log: every request carries seed 70704 (198 requests, nothing else)", cnt == {SEED: M.N_Q},
                   json.dumps({str(k): v for k, v in cnt.items()})))
    cfg, rcfg = results_cfg(a.pass_dir), results_cfg(REC["passes"][0])
    ok = cfg is not None and str(cfg["model_args"].get("seed")) == str(SEED) and cfg["random_seed"] == PROCESS_SEED
    checks.append((f"lm-eval: model_args seed {SEED} (sent in every request) and process seed {PROCESS_SEED} (the records' pass-1 process seed)",
                   ok and rcfg is not None and rcfg["random_seed"] == PROCESS_SEED,
                   f"model_args seed {cfg and cfg['model_args'].get('seed')}, process seed {cfg and cfg['random_seed']}; record pass 1 process seed {rcfg and rcfg['random_seed']}"))
    strip = lambda ma: {k: v for k, v in (ma or {}).items() if k != "seed"}
    checks.append(("lm-eval client configuration equals the 0.9.1 record's pass 1 (model_args without seed: model, base_url, num_concurrent 8, "
                   "tokenized_requests, timeout 10800, max_retries 3; gen_kwargs max_gen_toks 327680)",
                   cfg is not None and rcfg is not None and strip(cfg["model_args"]) == strip(rcfg["model_args"]) and cfg["gen_kwargs"] == rcfg["gen_kwargs"],
                   f"{strip(cfg['model_args']) if cfg else None} / {cfg and cfg['gen_kwargs']}"))
    # ---- task identity
    h = X.task_hash_of(a.pass_dir)
    checks.append((f"lm-eval task hash {X.TASK_HASH[:16]} (same task, rendering and answer ordering as the records)", h == X.TASK_HASH, str(h)[:16]))
    checks.append(("0.9.1 record pass 1 and rep070-F pass 1 carry the same task hash", X.task_hash_of(REC["passes"][0]) == X.TASK_HASH
                   and X.task_hash_of(REPF["passes"][0]) == X.TASK_HASH, ""))
    # ---- per-question pairing with the 0.9.1 record's pass 1 and rep070-F's pass 1 (same process seed 1235)
    for nm, C in (("0.9.1 record pass 1", V), ("rep070-F pass 1", F)):
        have = all((q, 1) in A and (q, 1) in C for q in range(M.N_Q))
        checks.append((f"d091 vs {nm}: all 198 questions present on both sides", have, ""))
        if not have:
            continue
        for f in ("prompt_hash", "doc_hash", "target_hash", "messages_sha256"):
            bad = [q for q in range(M.N_Q) if A[(q, 1)][f] != C[(q, 1)][f]]
            checks.append((f"d091 vs {nm}: {f} equal on every question", not bad, f"{len(bad)} mismatches"))
        pt = [q for q in range(M.N_Q) if A[(q, 1)]["prompt_tokens"] is not None and C[(q, 1)]["prompt_tokens"] is not None]
        bad = [q for q in pt if A[(q, 1)]["prompt_tokens"] != C[(q, 1)]["prompt_tokens"]]
        checks.append((f"d091 vs {nm}: server-rendered prompt token counts equal", not bad and len(pt) == M.N_Q, f"{len(pt)} compared, {len(bad)} differ"))
    # ---- warm-up, configuration identity, health
    first_t = min((r["t_start"] for r in A.values() if r["t_start"] is not None), default=None)
    floor = kv_floor(max_prompt_tokens(REC["reqlogs"]))
    win = window_of(A)
    kvinfo, srv = {"kv_pool_tokens": None, "record_kv_pool_sizes": None, "kv_floor": floor}, None
    if have_srv:
        checks += X.warmup_checks(a.serverlog, first_t, "d091 session warm-up")
        cg, kvinfo = config_gates(a.serverlog, REC["serverlogs"], "d091 session", floor)
        checks += cg
        hg, srv = health_gates(a.serverlog, win, "d091 session")
        checks += hg
    if a.inspect:
        try:
            d = json.load(open(a.inspect))[0]
            img = d.get("Image")
            b = (d.get("HostConfig") or {}).get("PortBindings") or {}
            hosts = sorted({x.get("HostIp") for v in b.values() for x in (v or [])})
        except (OSError, ValueError, IndexError, KeyError, TypeError) as e:
            img, hosts = f"unreadable ({type(e).__name__})", None
        checks.append(("container image Id is the 0.9.1 record's image a9e985c0", img == IMG_ID, str(img)[:24]))
    else:
        hosts = None
    adj = M.adjudicator_status()
    checks.append(("stated adjudicator source unchanged since freeze (informative)", bool(adj["source_unchanged"]), adj["current_source_sha256"]))
    valid = all(ok for n, ok, _ in checks if not n.startswith("stated adjudicator"))
    res = {"what": "option D (amendment 22x): one stock 0.9.1 GPQA Diamond pass at request seed 70704; DESCRIPTIVE",
           "dry_run": a.dry_run_label, "seed": SEED, "process_seed": PROCESS_SEED, "script_sha256": fsha(os.path.abspath(__file__)),
           "pins": PINS, "inputs": {"d091": ia, "v091_record": iv, "rep070F": if_}, "checks": [list(c) for c in checks], "valid": valid,
           "server_config": kvinfo, "published_on": hosts}
    if len(A) == M.N_Q:
        d = pass_summary(A)
        rec = {f"pass{p}": pass_summary({k: v for k, v in V.items() if k[1] == p}) for p in (1, 2, 3)}
        rng_ = lambda key: [min(rec[p][key] for p in rec), max(rec[p][key] for p in rec)]
        where = lambda x, lohi: "below the record's per-pass range" if x < lohi[0] else ("above the record's per-pass range" if x > lohi[1] else "inside the record's per-pass range")
        fF = json.load(open(REPF["final"])) if os.path.exists(REPF["final"]) else None
        repf_ctx = None
        if fF is not None:
            acc = fF["C1_rep_vs_v091"]["accuracy_all_requests"]
            repf_ctx = {"final_result_sha256": fsha(REPF["final"]), "final_result_sha256_frozen": REPF["final_sha"],
                        "empty_per_pass": {k: v["arm"] for k, v in fF["C1_rep_vs_v091"]["primary"]["per_pass"].items()},
                        "raw_pct_per_pass": acc["correct_flexible"]["arm_pct_per_pass"], "stated_final_pct_per_pass": acc["correct_stated"]["arm_pct_per_pass"]}
        res["pass"] = d
        res["context"] = {
            "status": "descriptive context: one pass at a new seed beside the 0.9.1 record's per-pass range; no test; the stated figure is "
                      "provisional until the coordinator's hand read of every row where stated differs from flexible-extract",
            "v091_record_per_pass": {p: {k: rec[p][k] for k in ("empty", "raw_correct_flexible", "stated_correct", "raw_pct", "stated_pct", "capped_at_327680")} for p in rec},
            "v091_record_range": {"empty": rng_("empty"), "raw_pct": rng_("raw_pct"), "stated_pct": rng_("stated_pct"), "capped": rng_("capped_at_327680")},
            "d091_vs_range": {"empty": where(d["empty"], rng_("empty")), "raw_pct": where(d["raw_pct"], rng_("raw_pct")),
                              "stated_pct_provisional": where(d["stated_pct"], rng_("stated_pct"))},
            "rep070F_per_pass": repf_ctx}
        # long-reply secondary (rep070 definitions, unchanged functions): P(finish | still generating at 100k)
        sd, pd = X.tail_summary(A, (1,), "d091, pass 1 (seed 70704)")
        sv, pv = X.tail_summary(V, (1, 2, 3), "0.9.1 record, passes 1-3")
        sv1, pv1 = X.tail_summary(V, (1,), "0.9.1 record, pass 1 only")
        sf, pf = X.tail_summary(F, (1, 2, 3), "stock 0.7.0 rep070-F, passes 1-3")
        res["long_reply_finishing"] = {
            "definition": ("at risk at L = completion_tokens >= L (proxy log); finished = finish_reason not 'length'; P(finish | at risk at "
                           "100,000) = finished / at risk (rep070 secondary / tail-audit definitions, functions of analysis_rep070.py unchanged)"),
            "status": "DESCRIPTIVE: one pass, about 8 replies at risk expected, different seeds from every comparator; no test",
            "groups": {"d091": sd, "v091_record": sv, "v091_record_pass1": sv1, "rep070F": sf},
            "intervals_95_question_bootstrap": {"d091": boot_p(pd), "v091_record": boot_p(pv, seed=BOOT_SEED + 1),
                                                "rep070F": boot_p(pf, seed=BOOT_SEED + 2)},
            "differences_at_100000": {"d091_minus_v091_record": X.tail_diff(pd, pv, seed=BOOT_SEED + 3),
                                      "d091_minus_rep070F": X.tail_diff(pd, pf, seed=BOOT_SEED + 4)},
            "unit_note": "d091 has one reply per question, so its question bootstrap is a reply bootstrap; the records' intervals cluster passes by question"}
        # timing, server status (descriptive)
        if win:
            tok = sum(r["completion_tokens"] for r in A.values() if r["completion_tokens"] is not None)
            prof = M.inflight(list(A.values()))
            res["timing"] = {"window_h": round((win[1] - win[0]) / 3600, 4), "completion_tokens": tok,
                             "aggregate_tok_s": round(tok / (win[1] - win[0]), 1),
                             "occupied": M.parse_chainlogs(a.chainlog, 2), "gpu_hours_ledger": a.gpu_hours,
                             "inflight_profile": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in prof[0].items()} if prof else None}
        res["server"] = srv
        if a.rows_out:
            with open(a.rows_out, "w") as fo:
                for q in range(M.N_Q):
                    r = A[(q, 1)]
                    fo.write(json.dumps({k: r[k] for k in ROW_KEYS}, sort_keys=True) + "\n")
            res["rows_out_sha256"] = fsha(a.rows_out)
    json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)
    L = [f"# Option D: stock 0.9.1, one GPQA Diamond pass at request seed {SEED} (descriptive)" + (f" -- DRY RUN: {a.dry_run_label}" if a.dry_run_label else ""),
         "", f"valid: **{valid}**; frozen module {M.file_sha(os.path.join(HERE, 'frozen/analysis_col511.py'))[:8]}; no outcome test against the records", "", "## Checks"]
    L += [f"- [{'ok' if ok else 'FAIL'}] {nm} ({det})" for nm, ok, det in checks]
    if "pass" in res:
        d, c = res["pass"], res["context"]
        L += ["", "## The pass (descriptive; stated figure provisional until the hand read)",
              f"- empty {d['empty']} of 198 (0.9.1 record per pass {c['v091_record_range']['empty'][0]}-{c['v091_record_range']['empty'][1]}: {c['d091_vs_range']['empty']})",
              f"- raw (flexible) {d['raw_correct_flexible']} ({d['raw_pct']}%; record {c['v091_record_range']['raw_pct'][0]}-{c['v091_record_range']['raw_pct'][1]}%: {c['d091_vs_range']['raw_pct']})",
              f"- stated (provisional) {d['stated_correct']} ({d['stated_pct']}%; record {c['v091_record_range']['stated_pct'][0]}-{c['v091_record_range']['stated_pct'][1]}%); "
              f"rows for the hand read (stated differs from flexible): {', '.join(d['stated_differs_from_flexible_ids']) or '-'}",
              f"- finish reasons {d['finish_reasons']}; empties by end {d['empties_by_end']}; capped at 327,680: {d['capped_at_327680']} (record {c['v091_record_range']['capped']})",
              f"- replies >= 50k / 100k / 200k: {d['replies_ge']['50000']} / {d['replies_ge']['100000']} / {d['replies_ge']['200000']}",
              f"- completion tokens answered: median {d['completion_tokens_answered'].get('p50')}, p90 {d['completion_tokens_answered'].get('p90')}, max {d['completion_tokens_answered'].get('max')}"]
        if c["rep070F_per_pass"]:
            L.append(f"- context, stock 0.7.0 rep070-F per pass: empty {c['rep070F_per_pass']['empty_per_pass']}")
        S = res["long_reply_finishing"]
        L += ["", "## P(finish | still generating at 100,000 tokens) (descriptive)"]
        for k, g in S["groups"].items():
            x = g[str(L_SEC)]
            ci = S["intervals_95_question_bootstrap"].get(k, {}).get("ci95")
            L.append(f"- {g['label']}: {x['finished']}/{x['at_risk']} ({x['questions_at_risk']} questions; P = "
                     f"{'n/a' if x['P_finish_given_at_risk'] is None else round(x['P_finish_given_at_risk'], 2)}{'; 95% ' + str(ci) if ci else ''})")
        for k, dd in S["differences_at_100000"].items():
            L.append(f"- {k}: {dd['estimate']} (95% joint question bootstrap {dd['ci95']}, {dd['undefined_replicates']} undefined)")
        L.append(f"- {S['status']}")
        if "timing" in res:
            T = res["timing"]
            L.append(f"\n- request window {T['window_h']} h, {T['completion_tokens']:,} tokens, {T['aggregate_tok_s']} tok/s aggregate; ledger GPU-h {T['gpu_hours_ledger']}")
        L.append(f"- KV pool {kvinfo['kv_pool_tokens']} (record sizes {kvinfo['record_kv_pool_sizes']}; floor {kvinfo['kv_floor']:,}; reported, not identity-gated); "
                 f"published on {hosts}")
    print("\n".join(L))
    return 0 if valid else 3


if __name__ == "__main__":
    sys.exit(main())
