#!/usr/bin/env python3
"""RE-FREEZE CANDIDATE (2026-10-15), NOT FROZEN. Analysis of a replication of stock tpurtell 0.7.0 K3.25 (DFlash2 x5) GPQA
record (specification: PROPOSAL-v2.md section 6). Copied from ../analysis_rep070_draft.py (sha256 3c811784...) and changed for:
  - the seed policy: --variant R (recorded seeds 1234/1235/1236) or F (fresh seeds 70701/70702/70703); validity checks test
    the variant's seeds in the request logs and in lm-eval's model_args; variant F also checks that its seeds are not in
    the frozen inventory of seeds already used by GPQA passes on this machine (checks/seed-inventory.json);
  - the verified warm-up (draft section 2), implemented here: per rep server log, one engine start, 9 release warm-up
    lines with the 'container is ready' line, no GPQA chat request logged before it, and the session's first request
    later than the last timestamped line before it;
  - configuration identity: the rep server logs' non-default args, vLLM version, model snapshots and logged env names equal
    the old 0.7.0 record's passes 2-3 server log (gating; --require-config-identity 0 only for stand-in dry runs);
  - the prespecified long-reply finishing secondary (descriptive, never the reading): P(finish | still generating at
    100,000 completion tokens), definitions of tail-audit/audit.py (amendment 20c), joint question-clustered bootstrap;
  - a descriptive coupling measure (log reply-length correlation at matched vs mismatched passes) beside the draft's
    identical-hash count;
  - the frozen module is imported from ./frozen/ (byte-identical copy, hash checked as in the draft; the draft imported a
    copy in a temporary directory).
Thresholds, tests, bootstrap seeds and readings are unchanged from the draft.
Change set 22n (2026-10-15): task identity made explicit (gating: lm-eval task hash 87e807b3b27c8661... in every rep and
record pass; per-question prompt/doc/target/message hashes vs both records were already gating via the frozen pairing
checks); request and lm-eval seeds observed per pass are reported; the report states what each category means and does
not mean, the tests' assumptions, and the secondary's limits (review sections 4-5). Rules and thresholds unchanged.

Machinery reused unchanged, by import, from the frozen column-511 analysis (analysis_col511.py, sha256 b5b13ce2...,
checked at start). Roles:
  rep    = the new 0.7.0 run (three passes; proxy request logs required)
  v091   = the existing 0.9.1 K3.25 three-pass record (historical)  -> C1 (does the advantage reproduce?)
  old070 = the existing 0.7.0 K3.25 three-pass record (historical)  -> C2 (repeatability / movement toward 0.9.1)
The old 0.7.0 record's pass 1 has no request log or server log; checks that need them are skipped for that pass only, and
it is excluded from the long-reply secondary (lengths unknown; amendment 20c).

usage: analysis_rep070.py --variant R|F --rep-pass P1 P2 P3 --rep-reqlog S1 S2 --rep-serverlog S1 S2 [--rep-chainlog ...]
       [--rep-hedged P:DOC ...] [--rep-gpu-hours X] [--dry-run-label TEXT] --out OUT.json > OUT.md
Exit code 3 when a validity check fails. Outputs ids, hashes and numbers only.
"""
import argparse, datetime, glob, hashlib, json, math, os, random, re, sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "frozen")
FROZEN_SHA = "b5b13ce2155085ca46fb24877f12b7cd0a5cdd16eb4cf19012958e7b86ae0033"
assert hashlib.sha256(open(os.path.join(FROZEN, "analysis_col511.py"), "rb").read()).hexdigest() == FROZEN_SHA, "frozen module changed"
sys.path.insert(0, FROZEN)
import analysis_col511 as M  # noqa: E402

G, B, R = "<gpqa-runs>", "<data-runs>/benchmarks", "<this-repo>/runs"
V091, V070 = f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1", f"{G}/glm53-flash-exl3-k3.25@tpurtell"
RECORDS = {
    "v091": dict(label="0.9.1 K3.25 DFlash2 x3 (historical 3-pass record)",
                 passes=[f"{V091}+ps__pass{p}" for p in (1, 2, 3)], reqlogs=[f"{V091}.requests.jsonl", f"{V091}+ps.requests.jsonl"],
                 serverlogs=[f"{V091}.server.log", f"{V091}+ps.server.log"],
                 chainlogs=[f"{B}/2026-10-08-glm53-flash-exl3-k3.25@tpurtell+v0.9.1/chain.log",
                            f"{B}/2026-10-09-glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps/chain.log"],
                 hedged=[], published=f"{R}/2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3_gpqa-diamond/results.jsonl", gpu_hours=16.63),
    "old070": dict(label="0.7.0 K3.25 DFlash2 x5 (historical 3-pass record)",
                   passes=[f"{V070}+ps__pass{p}" for p in (1, 2, 3)], reqlogs=[f"{V070}+ps.requests.jsonl"],
                   serverlogs=[f"{V070}+ps.server.log"],
                   chainlogs=[f"{B}/2026-09-29-glm53-flash-exl3-k3.25@tpurtell/chain.log",
                              f"{B}/2026-10-10-glm53-flash-exl3-k3.25@tpurtell+ps/chain.log"],
                   hedged=["2:25", "2:94"], published=f"{R}/2026-09-29_glm53-flash_k3.25-v0.7.0-dflash5_gpqa-diamond/results.jsonl",
                   gpu_hours=22.97)}
# PROPOSED operational thresholds (not causal boundaries); rep empties of 594; old 0.7.0 record 4, 0.9.1 record 16. Unchanged.
ALPHA, O1_MAX, O2_MIN = 0.05, 8, 12
BOOT_SEED_C2 = M.BOOT_SEED + 10   # 20261021, C2 effect interval (draft)
# ---- seed policy (PROPOSAL-v2.md section 2a)
VARIANTS = {"R": (1234, 1235, 1236), "F": (70701, 70702, 70703)}
# Request seeds already used by any GPQA request or pass on this machine (checks/seed-inventory.json, 2026-10-15):
# full passes 1234/1235/1236; GPQA-prompt gate runs 4242; single-question screens 5001-5012.
USED_GPQA_SEEDS = frozenset([1234, 1235, 1236, 4242] + list(range(5001, 5013)))
PROCESS_SEED_BASE = 1234          # lm-eval --seed 1234 + pass, as in every record (prompt construction unchanged)
# ---- verified warm-up (draft section 2)
WARMUPS_RECORD = 9                # 'GLM release startup ' lines in the record's passes 2-3 server log (and both 0.9.1 logs)
READY_LINE = "GLM release startup warmup complete; container is ready."
CHAT_ACCESS = '"POST /v1/chat/completions HTTP'
# ---- long-reply secondary (tail-audit/audit.py definitions; amendment 20c)
L_SEC = 100000                    # prespecified threshold: still generating when completion token 100,000 was due
L_DESC = (200000,)                # descriptive only
BOOT_B_SEC, BOOT_SEED_SEC = 2000, 20261015
# ---- task identity (records: every pass of both records)
TASK = "gpqa_diamond_thinking"
TASK_HASH = "87e807b3b27c86618285efdab99466609ad12f022ae453146595fc8306b053b0"
MEANING = {
    "O1": ("the {what} sustains a low empty count and separates from the historical 0.9.1 comparator under the specified "
           "test. It does not establish equivalence to the original 0.7.0 rate, nor a contemporaneous causal engine advantage."),
    "O2": ("the {what} does not sustain the original low-empty result under the specified rules. This does not necessarily mean "
           "that no advantage over 0.9.1 remains: p2 can indicate deterioration relative to the old 0.7.0 record while some "
           "advantage over 0.9.1 persists (see C1 beside it)."),
    "O3": "the prespecified evidence is inconclusive."}
NEXT = {
    "O1": "the historical advantage merits further investigation; consider a contemporaneous comparison before another broad mechanism search.",
    "O2": "reassess whether the original 4-versus-16 contrast remains an appropriate optimization target; verified correctness fixes keep their value.",
    "O3": "report the uncertainty and make a new budget decision; no automatic additional passes."}
ASSUMPTIONS = ("Both tests are exact sign-flip tests: 'exact' refers to the calculation of the p-value (full enumeration of the "
               "sign patterns), given the assumptions. The assumptions are that, under the null, each question's pair of "
               "per-question empty counts (rep, historical record) is exchangeable, i.e. the historical comparator behaves as "
               "a draw from the same process as the new run on that question, and that questions are the independent units. "
               "Neither is designed: the comparators are historical records, so time, state and other changes since are "
               "confounded with the comparison.")
SECONDARY_LIMITS = ("Descriptive only. The small number of finishing events (about 10 per record) and the wide intervals prevent "
                    "this measure from adjudicating a termination mechanism. Reaching 100,000 tokens selects different "
                    "trajectories on different engines, so the conditional comparison is not a causal engine effect.")


def task_hash_of(d):
    res = glob.glob(os.path.join(d, "**", "results_*.json"), recursive=True)
    if len(res) != 1:
        return None
    return (json.load(open(res[0])).get("task_hashes") or {}).get(TASK)


def task_checks(rep_dirs):
    c = []
    for p, d in enumerate(rep_dirs, 1):
        h = task_hash_of(d)
        c.append((f"rep pass {p}: lm-eval task hash {TASK_HASH[:16]} (same task, rendering and answer ordering as both records)", h == TASK_HASH, str(h)[:16]))
    for k, r in RECORDS.items():
        hs = [task_hash_of(d) for d in r["passes"]]
        c.append((f"{k} record: task hash {TASK_HASH[:16]} in all three passes", all(h == TASK_HASH for h in hs), ",".join(str(h)[:8] for h in hs)))
    return c


def seeds_observed(dirs, reqlogs, seeds):
    rows = M.load_reqlogs(reqlogs)
    out = {}
    for p, (d, s) in enumerate(zip(dirs, seeds), 1):
        res = glob.glob(os.path.join(d, "**", "results_*.json"), recursive=True)
        cfg = json.load(open(res[0])).get("config", {}) if len(res) == 1 else {}
        ma = cfg.get("model_args") or {}
        if isinstance(ma, str):
            ma = dict(kv.split("=", 1) for kv in ma.split(",") if "=" in kv)
        out[f"pass{p}"] = {"variant_request_seed": s, "requests_logged_with_that_seed": sum(r.get("seed") == s for r in rows),
                           "lm_eval_model_args_seed": ma.get("seed"), "lm_eval_process_seed": cfg.get("random_seed"),
                           "task_hash": task_hash_of(d)}
    return out


def ts_epoch(line, year=2026):
    t = M.ts_utc(line, year)
    return t


def warmup_checks(path, first_t, label):
    """Verified warm-up for one rep server session. first_t = epoch of the session's first proxy request."""
    out = []
    lines = open(path, errors="replace").read().splitlines()
    starts = sum("non-default args:" in l for l in lines)
    warm = [i for i, l in enumerate(lines) if "GLM release startup " in l]
    ready = [i for i, l in enumerate(lines) if READY_LINE in l]
    out.append((f"{label}: exactly one engine start in the server log", starts == 1, starts))
    out.append((f"{label}: {WARMUPS_RECORD} release warm-up lines (record's count), ending with the ready line",
                len(warm) == WARMUPS_RECORD and len(ready) == 1 and warm[-1] == ready[0], f"{len(warm)} warm-up lines, {len(ready)} ready lines"))
    if ready:
        r0 = ready[0]
        early = sum(CHAT_ACCESS in l for l in lines[:r0])
        out.append((f"{label}: no GPQA chat-completion request logged before the ready line", early == 0, f"{early} before"))
        before = [t for t in (ts_epoch(l) for l in lines[:r0]) if t is not None]
        t_lo = before[-1] if before else None
        ok = t_lo is not None and first_t is not None and first_t > t_lo
        out.append((f"{label}: first GPQA request after the last timestamped line before the ready line",
                    ok, f"first request {first_t}, last line before ready {t_lo}"))
    return out


def config_identity(rep_logs, rec_log):
    rec = M.parse_serverlogs([rec_log], [], 0)[0]
    out = []
    for i, p in enumerate(rep_logs, 1):
        s = M.parse_serverlogs([p], [], 0)[0]
        for k in ("non_default_args_sha256", "vllm_version", "model_snapshots", "env_names_logged_sha256", "cudagraph_mode"):
            out.append((f"rep session {i}: {k} equals the old 0.7.0 record's passes 2-3 server log", s.get(k) == rec.get(k),
                        f"{str(s.get(k))[:24]} vs {str(rec.get(k))[:24]}"))
    return out


def load_rep(label, dirs, reqlogs, hedged, published, seeds):
    """Variant R: one call of the frozen loader, exactly as the draft (SEED_BASE 1233 gives 1234/1235/1236).
    Variant F: one call per pass with the frozen module's SEED_BASE set so that pass p is matched to its own seed; only that
    pass's rows and checks are kept; SEED_BASE is restored on every exit path."""
    if tuple(seeds) == tuple(M.SEED_BASE + p for p in (1, 2, 3)):
        return M.load_arm(label, dirs, reqlogs, hedged, published)
    if published:
        sys.exit("--rep-published is only meaningful for variant R stand-ins")
    rows, checks, inputs = {}, [], {"samples": {}, "reqlogs": {}}
    old = M.SEED_BASE
    try:
        for p, s in enumerate(seeds, 1):
            M.SEED_BASE = s - p
            r, c, i = M.load_arm(label, dirs, reqlogs, hedged, None)
            rows.update({k: v for k, v in r.items() if k[1] == p})
            checks += [x for x in c if x[0].startswith(f"{label} pass {p}:") or x[0].startswith(f"{label} pass {p} q")]
            inputs["reqlogs"].update(i["reqlogs"])
            if f"pass{p}" in i["samples"]:
                inputs["samples"][f"pass{p}"] = i["samples"][f"pass{p}"]
    finally:
        M.SEED_BASE = old
    return rows, checks, inputs


def seed_checks(variant, seeds, dirs, reqlogs):
    c = []
    if variant == "R":
        c.append(("variant R: rep seeds are the records' per-pass seeds 1234/1235/1236", tuple(seeds) == (1234, 1235, 1236), seeds))
    else:
        c.append(("variant F: three distinct fresh seeds, none in the frozen inventory of seeds used by GPQA passes",
                  len(set(seeds)) == 3 and not (set(seeds) & USED_GPQA_SEEDS), seeds))
    rows = M.load_reqlogs(reqlogs)
    cnt = {}
    for r in rows:
        cnt[r.get("seed")] = cnt.get(r.get("seed"), 0) + 1
    c.append((f"rep request logs: request seeds are exactly the variant's {list(seeds)}, 198 requests each, nothing else",
              cnt == {s: M.N_Q for s in seeds}, json.dumps({str(k): v for k, v in cnt.items()})))
    for p, (d, s) in enumerate(zip(dirs, seeds), 1):
        res = glob.glob(os.path.join(d, "**", "results_*.json"), recursive=True)
        ok, det = False, "no results json"
        if len(res) == 1:
            cfg = json.load(open(res[0])).get("config", {})
            ma = cfg.get("model_args") or {}
            if isinstance(ma, str):
                ma = dict(kv.split("=", 1) for kv in ma.split(",") if "=" in kv)
            got, rs = ma.get("seed"), cfg.get("random_seed")
            ok = str(got) == str(s) and rs == PROCESS_SEED_BASE + p
            det = f"model_args seed {got}, process seed {rs}"
        c.append((f"rep pass {p}: lm-eval model_args seed {s} (sent in every request) and process seed {PROCESS_SEED_BASE + p}", ok, det))
    return c


def pairing(A, rec, k, variant):
    out = []
    for n, ok, d in M.check_pairing(A, rec):
        if variant == "F" and n == "(question, pass) pairs agree on seed":
            nm = int(str(d).split()[0]) if str(d).split() and str(d).split()[0].isdigit() else -1
            out.append((f"rep vs {k}: variant F, rep seed differs from the record's on every (question, pass) pair", nm == M.N_Q * M.N_PASSES, d))
        else:
            out.append((f"rep vs {k}: {n}", ok, d))
    return out


# ------------------------------------------------------------------ long-reply secondary
def tail_units(rows, passes):
    """Per question: (at risk at L, events among them) for each L, over the logged rows of the given passes."""
    Ls = (L_SEC,) + L_DESC
    per = {q: [[0, 0] for _ in Ls] for q in range(M.N_Q)}
    n_rows = 0
    for (q, p), r in rows.items():
        if p not in passes or not r["logged"] or r["completion_tokens"] is None:
            continue
        n_rows += 1
        for j, L in enumerate(Ls):
            if r["completion_tokens"] >= L:
                per[q][j][0] += 1
                per[q][j][1] += r["finish_reason"] != "length"     # event = generation ended before the cap (audit)
    return per, n_rows


def tail_summary(rows, passes, label):
    per, n = tail_units(rows, passes)
    Ls = (L_SEC,) + L_DESC
    out = {"label": label, "passes": list(passes), "logged_rows": n}
    for j, L in enumerate(Ls):
        risk = sum(v[j][0] for v in per.values()); ev = sum(v[j][1] for v in per.values())
        out[str(L)] = {"at_risk": risk, "finished": ev, "censored_at_cap": risk - ev, "questions_at_risk": sum(v[j][0] > 0 for v in per.values()),
                       "P_finish_given_at_risk": (ev / risk) if risk else None,
                       "at_risk_per_pass": round(risk / len(passes), 2) if passes else None}
    nonempty = sum(1 for (q, p), r in rows.items() if p in passes and r["logged"] and r["completion_tokens"] is not None
                   and r["completion_tokens"] >= L_SEC and r["finish_reason"] != "length" and not r["empty"])
    out[str(L_SEC)]["finished_nonempty"] = nonempty
    return out, per


def tail_diff(perA, perB, j=0, B=BOOT_B_SEC, seed=BOOT_SEED_SEC):
    """Joint question-clustered bootstrap of P_A - P_B (the same resampled questions on both sides; audit.boot_diff)."""
    def P(per, idx):
        den = sum(per[q][j][0] for q in idx)
        return (sum(per[q][j][1] for q in idx) / den) if den else None
    allq = list(range(M.N_Q))
    pa, pb = P(perA, allq), P(perB, allq)
    rng = random.Random(seed)
    d, undef = [], 0
    for _ in range(B):
        idx = [rng.randrange(M.N_Q) for _ in range(M.N_Q)]
        a, b = P(perA, idx), P(perB, idx)
        if a is None or b is None:
            undef += 1
            continue
        d.append(a - b)
    d.sort()
    ci = [round(d[int(0.025 * len(d))], 4), round(d[min(len(d) - 1, int(0.975 * len(d)))], 4)] if d else None
    return {"estimate": round(pa - pb, 4) if pa is not None and pb is not None else None, "ci95": ci, "undefined_replicates": undef,
            "B": B, "seed": seed, "unit": "question (all its requests on both sides)"}


def secondary(A, recs, variant):
    rep, prep = tail_summary(A, (1, 2, 3), "rep, passes 1-3")
    v91, pv91 = tail_summary(recs["v091"], (1, 2, 3), "0.9.1 record, passes 1-3")
    o70, po70 = tail_summary(recs["old070"], (2, 3), "old 0.7.0 record, passes 2-3 (pass 1 unlogged: lengths unknown)")
    out = {"definition": ("at risk at L = completion_tokens >= L (proxy log); finished = finish_reason not 'length' (generation ended "
                          "before the 327,680 cap); censored = finish 'length' at the cap. P(finish | at risk at 100,000) = finished / at "
                          "risk. Definitions of tail-audit/audit.py (amendment 20c). Prespecified threshold 100,000; 200,000 descriptive."),
           "status": "SECONDARY, descriptive; never the reading; no test, no threshold",
           "groups": {"rep": rep, "v091": v91, "old070": o70},
           "differences_at_100000": {"rep_minus_v091": tail_diff(prep, pv91, seed=BOOT_SEED_SEC),
                                     "rep_minus_old070_passes_2_3": tail_diff(prep, po70, seed=BOOT_SEED_SEC + 1)}}
    if variant == "R":
        r23, pr23 = tail_summary(A, (2, 3), "rep, passes 2-3 (seed-matched to the old record's logged passes)")
        out["groups"]["rep_passes_2_3"] = r23
        out["differences_at_100000"]["rep_passes_2_3_minus_old070_passes_2_3_seed_matched"] = tail_diff(pr23, po70, seed=BOOT_SEED_SEC + 2)
    out["note"] = ("Event counts are small (the records hold about 10 finishing events each). Seeds: variant R shares seeds 1234-1236 "
                   "with both records (coupled draws); variant F does not. Old record pass 1: no request log, excluded.")
    return out


def pearson(u, v):
    if len(u) < 3:
        return None
    mu, mv = sum(u) / len(u), sum(v) / len(v)
    su = math.sqrt(sum((a - mu) ** 2 for a in u)); sv = math.sqrt(sum((b - mv) ** 2 for b in v))
    return round(sum((a - mu) * (b - mv) for a, b in zip(u, v)) / (su * sv), 3) if su and sv else None


def coupling(A, C):
    """Descriptive: correlation over questions of log reply length, rep pass p vs record pass p (matched) and vs the record's
    other passes (mismatched); questions answered in both. Same-seed pairs in the existing records: 0.66-0.91; different-seed
    pairs 0.37-0.48 (checks/same-seed-identity.json)."""
    def corr(p, p2):
        qs = [q for q in range(M.N_Q) if not A[(q, p)]["empty"] and not C[(q, p2)]["empty"]]
        return pearson([math.log(A[(q, p)]["response_chars"]) for q in qs], [math.log(C[(q, p2)]["response_chars"]) for q in qs])
    m = {f"pass{p}": corr(p, p) for p in (1, 2, 3)}
    mm = {f"pass{p}_vs_pass{p2}": corr(p, p2) for p in (1, 2, 3) for p2 in (1, 2, 3) if p2 != p}
    return {"matched_pass": m, "mismatched_pass": mm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--rep-label", default=None)
    ap.add_argument("--rep-pass", nargs=3, required=True)
    ap.add_argument("--rep-reqlog", nargs="*", default=[])
    ap.add_argument("--rep-serverlog", nargs="*", default=[], help="session 1 (pass 1) then session 2 (passes 2-3)")
    ap.add_argument("--rep-chainlog", nargs="*", default=[])
    ap.add_argument("--rep-hedged", nargs="*", default=[])
    ap.add_argument("--rep-published", default=None, help="dry runs only (variant R stand-ins)")
    ap.add_argument("--rep-gpu-hours", type=float, default=None)
    ap.add_argument("--require-rep-reqlog", type=int, default=1)
    ap.add_argument("--require-rep-serverlog", type=int, default=1, help="0 only for stand-ins without both session logs")
    ap.add_argument("--require-config-identity", type=int, default=1, help="0 only for the 0.9.1 stand-in")
    ap.add_argument("--require-seed-args", type=int, default=1, help="0 only for stand-ins whose pass 1 ran without a seed argument")
    ap.add_argument("--dry-run-label", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = VARIANTS[a.variant]
    label = a.rep_label or f"0.7.0 K3.25 DFlash2 x5, replication variant {a.variant} (stock image 48e254d9)"
    A, ca, ia = load_rep(label, a.rep_pass, a.rep_reqlog, a.rep_hedged, a.rep_published, seeds)
    recs, checks, inputs = {}, list(ca), {"rep": ia}
    for k, r in RECORDS.items():
        rows, c, i = M.load_arm(r["label"], r["passes"], r["reqlogs"], r["hedged"], r["published"])
        recs[k] = rows; checks += c; inputs[k] = i
    if a.require_rep_reqlog:
        checks.append(("rep: every pass has its proxy request log (finish reasons and token counts)",
                       all(row["logged"] for row in A.values()) and len(A) == M.N_Q * M.N_PASSES, sum(row["logged"] for row in A.values())))
    sc = seed_checks(a.variant, seeds, a.rep_pass, a.rep_reqlog)
    if not a.require_seed_args:
        sc = [(n + " (informative for this stand-in)", ok, d) if "lm-eval model_args seed" in n else (n, ok, d) for n, ok, d in sc]
    if not a.require_rep_reqlog:
        sc = [(n + " (informative for this stand-in)", ok, d) if n.startswith("rep request logs:") else (n, ok, d) for n, ok, d in sc]
    checks += sc
    checks += task_checks(a.rep_pass)
    for k in recs:
        checks += pairing(A, recs[k], k, a.variant)
    # verified warm-up and configuration identity (rep server logs: session 1 = pass 1, session 2 = passes 2-3)
    sess_passes = [(1,), (2, 3)]
    if a.require_rep_serverlog and len(a.rep_serverlog) != 2:
        checks.append(("rep: two server logs (session 1, session 2)", False, len(a.rep_serverlog)))
    for i, p in enumerate(a.rep_serverlog, 1):
        ps = sess_passes[i - 1] if len(a.rep_serverlog) == 2 else sess_passes[1]
        ts = [A[(q, pp)]["t_start"] for q in range(M.N_Q) for pp in ps if (q, pp) in A and A[(q, pp)]["t_start"] is not None]
        checks += warmup_checks(p, min(ts) if ts else None, f"rep session {i if len(a.rep_serverlog) == 2 else 2} warm-up")
    if a.rep_serverlog:
        ci = config_identity(a.rep_serverlog, RECORDS["old070"]["serverlogs"][0])
        if not a.require_config_identity:
            ci = [(n + " (informative for this stand-in)", ok, d) for n, ok, d in ci]
        checks += ci
    adj = M.adjudicator_status()
    checks.append(("stated adjudicator source unchanged since freeze (informative)", bool(adj["source_unchanged"]), adj["current_source_sha256"]))
    valid = all(ok for n, ok, _ in checks if not n.startswith("stated adjudicator") and not n.endswith("(informative for this stand-in)"))
    res = {"refreeze_candidate": True, "variant": a.variant, "seeds": list(seeds), "dry_run": a.dry_run_label, "frozen_module_sha256": FROZEN_SHA,
           "script_sha256": M.file_sha(os.path.abspath(__file__)), "inputs": inputs, "checks": [list(c) for c in checks], "valid": valid,
           "thresholds_proposed": {"alpha": ALPHA, "O1_max_empties": O1_MAX, "O2_min_empties": O2_MIN}}
    n = M.N_Q * M.N_PASSES
    if all(len(x) == n for x in [A, *recs.values()]):
        c1 = M.analyze(A, recs["v091"], label, RECORDS["v091"]["label"])   # arm = rep, cmp = v091: T = v091 - rep
        c2 = M.analyze(A, recs["old070"], label, RECORDS["old070"]["label"])
        for c in (c1, c2):   # the column-511 bands and readings do not apply here
            c["primary"].pop("band"); c["primary"].pop("reading")
        ea, eo = M.per_question(A, "empty"), M.per_question(recs["old070"], "empty")
        d2 = [x - y for x, y in zip(ea, eo)]                                      # rep - old070: movement toward more empties
        t2 = M.signflip(d2)
        lo2, hi2 = M.boot_interval(d2, n, BOOT_SEED_C2)
        p1 = c1["primary"]["test"]["p_one_sided"]
        nrep = c1["primary"]["arm_empties"]
        code, text = reading(nrep, p1, t2["p_one_sided"], a.variant)
        what = "recorded-seed repeat (variant R)" if a.variant == "R" else "fresh-seed replication (variant F)"
        keys = [(q, p) for q in range(M.N_Q) for p in (1, 2, 3)]
        same = {f"pass{p}": sum(not A[(q, p)]["empty"] and A[(q, p)]["response_sha256"] == recs["old070"][(q, p)]["response_sha256"]
                                 for q in range(M.N_Q)) for p in (1, 2, 3)}   # answered replies only (empty replies share one hash)
        res.update({
            "C1_rep_vs_v091": c1,
            "C2_rep_vs_old070": {**c2, "repeatability_test": {"statistic": "T2 = sum_q (rep empties_q - old070 empties_q)", **t2,
                                 "direction": "one-sided, rep has more empties than the old 0.7.0 record (movement toward 0.9.1)",
                                 "effect_empty_rate_increase": {"estimate": sum(d2) / n, "ci95": [lo2, hi2], "seed": BOOT_SEED_C2}}},
            "repeatability_descriptive": {"identical_answered_reply_hash_vs_old070_per_pass": same,
                                          "empty_in_both_rep_and_old070": [f"q{q}p{p}" for q, p in keys if A[(q, p)]["empty"] and recs["old070"][(q, p)]["empty"]],
                                          "log_reply_length_correlation_vs_old070": coupling(A, recs["old070"]),
                                          "log_reply_length_correlation_vs_v091": coupling(A, recs["v091"]),
                                          "note": ("variant R: same engine, same seeds as old070 and same seeds as v091 (coupled draws); variant F: "
                                                   "no shared seed. Existing same-seed pairs: 0 identical answered replies, length correlation "
                                                   "0.66-0.91; different-seed pairs 0.37-0.48. Descriptive")},
            "reading": {"rep_empties": nrep, "p1_one_sided_rep_fewer_than_v091": p1, "p2_one_sided_rep_more_than_old070": t2["p_one_sided"],
                        "outcome": code, "text": text, "variant": a.variant,
                        "meaning": MEANING[code].format(what=what), "next_step_prespecified": NEXT[code],
                        "test_assumptions": ASSUMPTIONS,
                        "limits": ("historical comparators for both C1 and C2; exchangeability assumed, not designed; thresholds are "
                                   "operational; " + ("repeatability under the recorded seeds (coupled to both records), not a fresh-seed "
                                                      "estimate" if a.variant == "R" else "fresh seeds: a population-rate replication of "
                                                      "0.7.0's record, not a repeat of its seeds"))},
            "secondary_long_reply_finishing": {**secondary(A, recs, a.variant), "limits": SECONDARY_LIMITS}})
        res["seeds_observed"] = seeds_observed(a.rep_pass, a.rep_reqlog, seeds)
        ch = M.parse_chainlogs(a.rep_chainlog, 2)
        res["timing_rep"], win = M.timing(A, "rep", ch, a.rep_gpu_hours, 2)
        res["server_rep"] = M.parse_serverlogs(a.rep_serverlog, win, 0)
    json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)
    L = [f"# Replication of stock 0.7.0, variant {a.variant} (re-freeze candidate analysis)" + (f" -- DRY RUN: {a.dry_run_label}" if a.dry_run_label else ""), "",
         f"valid: **{valid}**; seeds {list(seeds)}; frozen module {FROZEN_SHA[:8]}; thresholds (proposed) O1 <= {O1_MAX} with p1 < {ALPHA}; "
         f"O2 >= {O2_MIN} or p2 < {ALPHA}", "", "## Checks"] + [f"- [{'ok' if ok else 'FAIL'}] {nm} ({d})" for nm, ok, d in checks]
    if "reading" in res:
        r, c1p, c2p = res["reading"], res["C1_rep_vs_v091"]["primary"], res["C2_rep_vs_old070"]
        L += ["", "## Reading", f"- rep empties {r['rep_empties']} of 594 (old 0.7.0 record {c2p['primary']['cmp_empties']}, 0.9.1 record {c1p['cmp_empties']})",
              f"- C1 (rep fewer than 0.9.1): one-sided p = {M.fmt_p(r['p1_one_sided_rep_fewer_than_v091'])}; reduction {100 * c1p['effect_empty_rate_reduction']['estimate']:+.2f} points "
              f"(95% {100 * c1p['effect_empty_rate_reduction']['ci95'][0]:+.2f} to {100 * c1p['effect_empty_rate_reduction']['ci95'][1]:+.2f})",
              f"- C2 (rep more than old 0.7.0): one-sided p = {M.fmt_p(r['p2_one_sided_rep_more_than_old070'])}; increase "
              f"{100 * c2p['repeatability_test']['effect_empty_rate_increase']['estimate']:+.2f} points (95% "
              f"{100 * c2p['repeatability_test']['effect_empty_rate_increase']['ci95'][0]:+.2f} to {100 * c2p['repeatability_test']['effect_empty_rate_increase']['ci95'][1]:+.2f})",
              f"- outcome **{r['outcome']}**: {r['text']}", f"- what {r['outcome']} means: {r['meaning']}",
              f"- prespecified next step: {r['next_step_prespecified']}", f"- tests: {r['test_assumptions']}", f"- limits: {r['limits']}",
              f"- identical answered-reply hashes vs old 0.7.0 per pass: {res['repeatability_descriptive']['identical_answered_reply_hash_vs_old070_per_pass']}",
              f"- empty in both rep and old 0.7.0: {res['repeatability_descriptive']['empty_in_both_rep_and_old070'] or '-'}",
              f"- log reply-length correlation vs old 0.7.0, matched passes {res['repeatability_descriptive']['log_reply_length_correlation_vs_old070']['matched_pass']}",
              "", "## Secondary (descriptive, never the reading): P(finish | still generating at 100,000 tokens)"]
        S = res["secondary_long_reply_finishing"]
        for k, g in S["groups"].items():
            x = g[str(L_SEC)]
            L.append(f"- {g['label']}: {x['finished']}/{x['at_risk']} finished ({x['questions_at_risk']} questions; "
                     f"P = {'n/a' if x['P_finish_given_at_risk'] is None else round(x['P_finish_given_at_risk'], 2)}); at >= 200k "
                     f"{g['200000']['finished']}/{g['200000']['at_risk']}")
        for k, dd in S["differences_at_100000"].items():
            L.append(f"- {k}: {dd['estimate']} (95% question-bootstrap {dd['ci95']}, {dd['undefined_replicates']} undefined)")
        L.append(f"- limits: {S['limits']}")
    if "seeds_observed" in res:
        L += ["", "## Seeds and task identity observed per pass"] + [f"- {k}: {v}" for k, v in res["seeds_observed"].items()]
    print("\n".join(L))
    return 0 if valid else 3


def reading(n, p1, p2, variant):
    if n <= O1_MAX and p1 < ALPHA:
        return ("O1", "substantially fewer empties again: replication support for the 0.7.0 advantage relative to the 0.9.1 record "
                      + ("(repeatability under the recorded seeds, historical comparators)" if variant == "R" else
                         "(at fresh seeds: population-rate replication, historical comparators)")
                      + "; the tail-mapping/DCP interaction on 0.7.0 becomes the concrete candidate (amendment 12)")
    if n >= O2_MIN or p2 < ALPHA:
        return ("O2", "movement toward the newer engine's count: the original gap is reported as not reproduced, without claiming "
                      "equivalence; no expensive bisection without a reproducible target (amendment 12)")
    return ("O3", "ambiguous: report the uncertainty and stop this allocation (amendment 12)")


if __name__ == "__main__":
    sys.exit(main())
