#!/usr/bin/env python3
"""Frozen analysis for the column-511 GPQA completion arm (PLAN.md amendments 11 and 11a). Standard library only.

Specification: analysis-col511.md (same directory). This file implements it; the two are frozen together and identified by
their sha256 in the investigation record. Outputs ids, hashes and numbers only: no question, answer or reply text.

usage:
  analysis_col511.py \
    --arm-label L --arm-pass DIR1 DIR2 DIR3 [--arm-reqlog F ...] [--arm-serverlog F ...] [--arm-chainlog F ...] \
        [--arm-hedged P:DOC ...] [--arm-published results.jsonl] [--arm-gpu-hours X] \
    --cmp-label L --cmp-pass DIR1 DIR2 DIR3 [--cmp-reqlog F ...] [--cmp-serverlog F ...] [--cmp-chainlog F ...] \
        [--cmp-hedged P:DOC ...] [--cmp-published results.jsonl] [--cmp-gpu-hours X] \
    [--server-block 0] [--gpus 2] [--out result.json] [--dry-run-label TEXT]

Pass k of each arm is the lm-eval output directory of pass k (request seed 1233 + k). A request log may hold several passes;
rows are matched to a pass by request seed. "arm" is the candidate; "cmp" is the fixed historical comparator.
"""
import argparse, datetime, glob, hashlib, json, math, os, random, re, statistics, sys
from functools import lru_cache

# ------------------------------------------------------------------ frozen constants (analysis-col511.md section 2)
N_PASSES, N_Q, CAP, SEED_BASE = 3, 198, 327680, 1233
TEMPERATURE, TOP_P = 1.0, 0.95
ALPHA = 0.05
BOOT_B = 20000
BOOT_SEED = 20261011            # empty-rate difference; +1 raw accuracy, +2 stated accuracy, +3 relative reduction, +4 lengths
BAND_LARGE_MAX, BAND_MODERATE_MAX = 6, 12   # operational bands on the arm's total empties: <=6, 7-12, >=13 (not causal)
LEN_THRESHOLDS = (32768, 65536, 131072, 200000)
ROUTE_FUSED_MAX_INFLIGHT = 4    # 0.9.1: decode rows = requests x 4 (DFlash2 x3); <=16 rows fused, >16 tiled (amendment 10a)

# ------------------------------------------------------------------ stated-answer adjudicator
# Copied verbatim from <local exporter export_evals.py> lines 233-259 (sha256 of those lines at freeze:
# 731f27d4a436805baae367096db8b039ae2d723f7944dda52f81fd2a96d017f6). Self-check: the comparator's published correct_stated
# must be reproduced on every row (--cmp-published).
# >>> ADJUDICATOR BEGIN
NOTLAB = r'(?![A-Za-z0-9])(?!\s*=)(?!\s*\*\*\s*=)'
LET = r'(?:(?:\*\*|\\textbf\{|\$|\\\(|\\text\{)\s*){0,2}(\()?([A-D])(\))?' + NOTLAB
CO = r'(?:\*\*)?\s*(?i:choice|option)?\s*'
STRONG = [
    ("boxed", re.compile(r'\\boxed\{\s*(?:\\(?:text|mathrm|textbf)\{\s*)?(?i:choice|option)?\s*(\()?\s*([A-D])\s*(\))?' + NOTLAB)),
    ("phrase", re.compile(r'(?i:(?:final|correct|the|my)\s+answer|answer)\s*(?i:is|:|=|-)?\s*[:\-]?\s*' + CO + LET)),
    ("phrase", re.compile(r'(?i:choice|option|answer\s+choice)\s*(?i:is|:|=)\s*[:\-]?\s*' + LET)),
    ("phrase", re.compile(r'(?i:the\s+(?:correct|right|intended|matching|best)\s+(?:choice|option|pair|match)\s+is|corresponds\s+to|matches(?:\s+only)?|choose|select)\s*[:\-]?\s*' + CO + LET)),
    ("phrase", re.compile(r'(?i:therefore|thus|hence|so)\s*[:,]\s*' + r'\*\*\s*(\()?([A-D])(\))?' + NOTLAB)),
    ("boxed", re.compile(r'\\boxed\{[^\n]{0,80}?\((?i:choice|option)\s*()([A-D])()\)')),
    ("bold", re.compile(r'(?m)^[ \t]*\*\*\s*(?i:choice|option)\s*(\()?([A-D])(\))?' + NOTLAB))]
WEAKBOLD = re.compile(r'(?m)^[ \t]*(?:#+[ \t]*)?\*\*\s*(\()?([A-D])(\))?(?=[\s.:\)]|\*\*)' + NOTLAB)
BARE = re.compile(r'\(([A-D])\)')


def final_answer(c):
    """The reply's last explicit commitment (boxed letter, answer phrase, bold choice line); else the last bare (A)-(D)."""
    hits = []
    for name, rx in STRONG:
        for m in rx.finditer(c):
            op, L, cl = m.groups()[-3:]; hits.append((m.start(), name, L, bool(op and cl)))
    for m in WEAKBOLD.finditer(c):
        if len(c) - m.start() <= 300:
            op, L, cl = m.groups(); hits.append((m.start(), "bold", L, bool(op and cl)))
    if hits: return max(hits)[2]
    b = [(m.start(), m.group(1)) for m in BARE.finditer(c)]
    return b[-1][1] if b else None
# >>> ADJUDICATOR END
ADJ_SOURCE = "<local exporter export_evals.py>"
ADJ_LINES = (233, 259)
ADJ_SHA = "731f27d4a436805baae367096db8b039ae2d723f7944dda52f81fd2a96d017f6"


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def messages_hash(messages):  # = record_proxy.py's hash of the request messages
    return sha(json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


# ------------------------------------------------------------------ frozen test and interval (section 3)
@lru_cache(maxsize=None)
def _null_dist(absvals):
    """Exact null distribution of sum(s_q * |d_q|), s_q = +/-1 independently with probability 1/2, over the questions
    with d_q != 0. Returned as a tuple of (sum, count) with counts summing to 2^m."""
    dist = {0: 1}
    for x in absvals:
        nd = {}
        for s_, c in dist.items():
            nd[s_ + x] = nd.get(s_ + x, 0) + c
            nd[s_ - x] = nd.get(s_ - x, 0) + c
        dist = nd
    return tuple(sorted(dist.items()))


def signflip(d):
    """Exact sign-flip test on per-question differences d (questions as the unit). Statistic T = sum(d).
    One-sided p = P(T* >= T_obs) (alternative: E[d_q] > 0, i.e. fewer empties in the arm when d = cmp - arm);
    two-sided p = P(|T*| >= |T_obs|) for reference. Exact enumeration over the 2^m sign patterns of the m nonzero d_q."""
    nz = tuple(sorted(abs(x) for x in d if x))
    T = sum(d)
    dist = _null_dist(nz)
    tot = 2 ** len(nz)
    p1 = sum(c for s_, c in dist if s_ >= T) / tot
    p2 = sum(c for s_, c in dist if abs(s_) >= abs(T)) / tot
    return dict(T=T, m_nonzero=len(nz), p_one_sided=p1, p_two_sided=p2, null_patterns=tot)


def boot_interval(values, denom, seed, B=BOOT_B):
    """Percentile 95% interval of sum(resampled values)/denom, resampling questions with replacement (all passes of a
    question, both arms, move together). Same indexing as tools/analyze.py records(): lo = sorted[int(0.025B)],
    hi = sorted[int(0.975B) - 1]."""
    rng = random.Random(seed)
    n = len(values)
    boots = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / denom for _ in range(B))
    return boots[int(0.025 * B)], boots[int(0.975 * B) - 1]


def boot_ratio_interval(num_vals, den_vals, seed, B=BOOT_B):
    """Percentile interval for 1 - sum(num)/sum(den) (relative reduction), questions resampled; replicates with
    sum(den) = 0 are counted and excluded."""
    rng = random.Random(seed)
    n = len(num_vals)
    out, undefined = [], 0
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        dsum = sum(den_vals[i] for i in idx)
        if dsum == 0:
            undefined += 1
            continue
        out.append(1 - sum(num_vals[i] for i in idx) / dsum)
    out.sort()
    if not out:
        return None, None, undefined
    k = len(out)
    return out[int(0.025 * k)], out[int(0.975 * k) - 1], undefined


def quantiles(xs, qs=(0.1, 0.25, 0.5, 0.75, 0.9, 0.99)):
    """Nearest-rank quantiles (no interpolation)."""
    if not xs:
        return {}
    s = sorted(xs)
    return {f"p{int(q * 100)}": s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))] for q in qs} | {"max": s[-1], "n": len(s)}


# ------------------------------------------------------------------ loading
def load_reqlogs(paths):
    recs = []
    for p in paths or []:
        for i, line in enumerate(open(p)):
            if line.strip():
                r = json.loads(line)
                r["_src"] = os.path.basename(p)
                recs.append(r)
    return recs


def load_arm(label, pass_dirs, reqlog_paths, hedged, published):
    if len(pass_dirs) != N_PASSES:
        sys.exit(f"{label}: need exactly {N_PASSES} pass directories")
    checks, inputs = [], {"samples": {}, "reqlogs": {}}
    reqs = load_reqlogs(reqlog_paths)
    for p in reqlog_paths or []:
        inputs["reqlogs"][os.path.basename(p)] = file_sha(p)
    hed = set()
    for h in hedged or []:
        a, b = h.split(":")
        hed.add((int(b), int(a)))
    rows = {}
    for p, d in enumerate(pass_dirs, 1):
        seed = SEED_BASE + p
        fs = glob.glob(os.path.join(d, "**", "samples_*.jsonl"), recursive=True)
        checks.append((f"{label} pass {p}: exactly one samples file", len(fs) == 1, len(fs)))
        if len(fs) != 1:
            continue
        inputs["samples"][f"pass{p}"] = {"dir": os.path.basename(os.path.normpath(d)), "sha256": file_sha(fs[0])}
        res = glob.glob(os.path.join(d, "**", "results_*.json"), recursive=True)
        if res:
            cfg = json.load(open(res[0])).get("config", {})
            ma = cfg.get("model_args") or {}
            if isinstance(ma, str):
                ma = dict(kv.split("=", 1) for kv in ma.split(",") if "=" in kv)
            inputs["samples"][f"pass{p}"]["lm_eval"] = {"num_concurrent": ma.get("num_concurrent"), "timeout": ma.get("timeout"),
                                                        "max_retries": ma.get("max_retries"), "seed_arg": ma.get("seed"),
                                                        "random_seed": cfg.get("random_seed"), "gen_kwargs": cfg.get("gen_kwargs")}
        by = {}
        for line in open(fs[0]):
            r = json.loads(line)
            by.setdefault(r["doc_id"], {})[r["filter"]] = r
        checks.append((f"{label} pass {p}: 198 questions, ids 0-197, both filters",
                       sorted(by) == list(range(N_Q)) and all(len(v) == 2 for v in by.values()), len(by)))
        pr = [r for r in reqs if r.get("seed") == seed]
        logged = bool(pr)
        req = {}
        if logged:
            counts = {}
            for r in pr:
                counts[r.get("messages_sha256")] = counts.get(r.get("messages_sha256"), 0) + 1
                req[r.get("messages_sha256")] = r
            checks.append((f"{label} pass {p}: request log has 198 requests, no re-sent request", len(pr) == N_Q and max(counts.values()) == 1,
                           f"{len(pr)} requests, max per message {max(counts.values())}"))
            checks.append((f"{label} pass {p}: every request HTTP 200, seed {seed}, max_tokens {CAP}, temperature {TEMPERATURE}, top_p {TOP_P}",
                           all(r.get("http_status") == 200 and r.get("max_tokens") == CAP and r.get("temperature") == TEMPERATURE
                               and r.get("top_p") == TOP_P for r in pr), ""))
            checks.append((f"{label} pass {p}: no request or response parse error",
                           not any("request_parse_error" in r or "response_parse_error" in r for r in pr), ""))
        for doc, fl in sorted(by.items()):
            a = fl.get("flexible-extract")
            s = fl.get("strict-match")
            if a is None or s is None:
                continue
            txt = str(a["resps"][0][0]) if a["resps"] else ""
            mh = messages_hash(json.loads(a["arguments"]["gen_args_0"]["arg_0"][0]))
            empty = not txt.strip()
            fa = None if empty else final_answer(txt)
            stated = (not empty) and (doc, p) not in hed and fa == a["target"].strip("()")
            row = {"doc_id": doc, "pass": p, "seed": seed, "doc_hash": a["doc_hash"], "prompt_hash": a["prompt_hash"],
                   "target_hash": a["target_hash"], "messages_sha256": mh, "empty": empty,
                   "correct_flexible": bool(a["exact_match"]), "correct_strict": bool(s["exact_match"]), "correct_stated": stated,
                   "stated_differs_from_flexible": stated != bool(a["exact_match"]), "hedged_by_hand": (doc, p) in hed,
                   "response_sha256": sha(txt), "response_chars": len(txt), "logged": logged,
                   "finish_reason": None, "completion_tokens": None, "prompt_tokens": None, "t_start": None, "secs": None}
            if logged:
                q = req.get(mh)
                if q is None:
                    checks.append((f"{label} pass {p} q{doc}: request found in log", False, "missing"))
                else:
                    row.update(finish_reason=q.get("finish_reason"), completion_tokens=q.get("completion_tokens"),
                               prompt_tokens=q.get("prompt_tokens"), t_start=q.get("t_start"), secs=q.get("secs"))
                    if q.get("content_sha256") != row["response_sha256"]:
                        checks.append((f"{label} pass {p} q{doc}: request-log content hash equals lm-eval reply hash", False, "mismatch"))
            rows[(doc, p)] = row
    if published:
        pub = {(r["doc_id"], r["pass"]): r for r in (json.loads(l) for l in open(published) if l.strip())}
        inputs["published"] = {"file": os.path.basename(published), "sha256": file_sha(published)}
        keys = ("empty", "correct_flexible", "correct_strict", "correct_stated", "response_sha256", "seed", "prompt_hash", "doc_hash", "target_hash")
        bad = [(k, f) for k, r in rows.items() for f in keys if k in pub and pub[k][f] != r[f]]
        bad += [(k, f) for k, r in rows.items() for f in ("finish_reason", "completion_tokens") if k in pub and r["logged"] and pub[k][f] != r[f]]
        checks.append((f"{label}: rows derived here reproduce the published rows ({len(pub)} rows; empty, raw, strict, stated, hashes, "
                       f"finish reason, tokens)", not bad and len(pub) == len(rows), f"{len(bad)} field mismatches"))
    return rows, checks, inputs


def parse_chainlogs(paths, gpus):
    """Occupied sessions: each 'chain done' paired with the first 'chain start' after the previous 'chain done' (aborted
    starts before it count as occupied time)."""
    sessions = []
    for p in paths or []:
        pending = None
        ready = []
        for line in open(p, errors="replace"):
            m = re.match(r"=== (\S+) chain (start|done)", line)
            r = re.match(r"ready after (\d+)s", line)
            if r:
                ready.append(int(r.group(1)))
            if not m:
                continue
            t = datetime.datetime.fromisoformat(m.group(1))
            if m.group(2) == "start" and pending is None:
                pending = t
            elif m.group(2) == "done" and pending is not None:
                sessions.append({"log": os.path.basename(os.path.dirname(p)) or os.path.basename(p), "start": pending.isoformat(),
                                 "done": t.isoformat(), "wall_h": round((t - pending).total_seconds() / 3600, 4),
                                 "ready_after_s": ready[-1] if ready else None})
                pending, ready = None, []
    wall = sum(s["wall_h"] for s in sessions)
    return {"sessions": sessions, "wall_h": round(wall, 4), "gpu_hours": round(gpus * wall, 3)} if sessions else None


def ts_utc(line, year):
    m = re.search(r" (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) ", line)
    if not m:
        return None
    return datetime.datetime(year, *map(int, m.groups()), tzinfo=datetime.timezone.utc).timestamp()


def parse_serverlogs(paths, windows, block, year=2026):
    """Server-log facts. Blocks are split at each 'non-default args' line; block `block` (default 0, the first) is cited for
    configuration. Status lines are counted only inside the request windows (epoch seconds; server timestamps are UTC)."""
    out = []
    for p in paths or []:
        lines = open(p, errors="replace").read().splitlines()
        starts = [i for i, l in enumerate(lines) if "non-default args:" in l]
        blocks = []
        for j, i0 in enumerate(starts):
            i1 = starts[j + 1] if j + 1 < len(starts) else len(lines)
            blocks.append((i0, i1))
        info = {"log": os.path.basename(p), "sha256": file_sha(p), "server_starts": len(starts)}
        if not blocks:
            out.append(info)
            continue
        b0, b1 = blocks[min(block, len(blocks) - 1)]
        blk = lines[max(0, b0 - 10):b1]
        nd = lines[b0].split("non-default args: ", 1)[1]
        info["cited_block"] = min(block, len(blocks) - 1)
        info["non_default_args_sha256"] = sha(nd)
        info["model_snapshots"] = sorted(set(re.findall(r"snapshots/([0-9a-f]{40})", "\n".join(blk))))
        info["vllm_version"] = sorted(set(re.findall(r"version (\S+)", "\n".join(l for l in blk if "api_utils.py:345" in l))))
        info["env_names_logged_sha256"] = sha(",".join(sorted(set(re.findall(r"Unknown vLLM environment variable detected: (\S+)", "\n".join(blk))))))
        info["kv_pool_tokens"] = [int(x.replace(",", "")) for x in re.findall(r"GPU KV cache size: ([\d,]+) tokens", "\n".join(blk))]
        cg = re.findall(r"'cudagraph_mode': <CUDAGraphMode\.(\w+)", "\n".join(blk))
        info["cudagraph_mode"] = sorted(set(cg))
        warm = [l.split("GLM release startup ", 1)[1] for l in blk if "GLM release startup " in l]
        info["release_warmups_logged"] = len(warm)
        st_ = []
        pre = ill = nanw = 0
        for l in blk:
            t = ts_utc(l, year)
            inside = t is not None and any(w0 <= t <= w1 for w0, w1 in windows)
            if not inside:
                continue
            m = re.search(r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%", l)
            if m:
                st_.append((float(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))))
            if re.search(r"preempt", l, re.I):
                pre += 1
            if "illegal memory access" in l:
                ill += 1
            if re.search(r"\bnan\b", l, re.I) and ("WARN" in l or "ERROR" in l):
                nanw += 1
        if st_:
            g8 = [g for g, r, w, k in st_ if r >= 8]
            info["status"] = {"lines": len(st_), "max_running": max(r for _, r, _, _ in st_), "lines_waiting": sum(w > 0 for _, _, w, _ in st_),
                              "peak_kv_usage_pct": max(k for *_, k in st_),
                              "mean_gen_tok_s_8plus_running": round(statistics.mean(g8), 1) if g8 else None,
                              "share_lines_running_1to4": round(sum(1 <= r <= ROUTE_FUSED_MAX_INFLIGHT for _, r, _, _ in st_) / len(st_), 4),
                              "share_lines_running_5plus": round(sum(r > ROUTE_FUSED_MAX_INFLIGHT for _, r, _, _ in st_) / len(st_), 4)}
        info["in_window"] = {"preemption_lines": pre, "illegal_memory_access_lines": ill, "nan_warning_lines": nanw}
        out.append(info)
    return out


def inflight(rows_pass):
    """Client-side in-flight profile of one pass from request start/end times."""
    ev = []
    for r in rows_pass:
        if r["t_start"] is None:
            return None
        ev += [(r["t_start"], 1), (r["t_start"] + r["secs"], -1)]
    ev.sort()
    cur, last, acc = 0, ev[0][0], {}
    for t, dv in ev:
        acc[cur] = acc.get(cur, 0) + t - last
        cur += dv
        last = t
    busy = sum(v for k, v in acc.items() if k > 0)
    # per request: share of its lifetime with <= ROUTE_FUSED_MAX_INFLIGHT in flight
    seg = []
    cur, last = 0, ev[0][0]
    for t, dv in ev:
        seg.append((last, t, cur))
        cur += dv
        last = t
    per = {}
    for r in rows_pass:
        a, b = r["t_start"], r["t_start"] + r["secs"]
        tot = low = 0.0
        for s0, s1, c in seg:
            o = min(b, s1) - max(a, s0)
            if o > 0:
                tot += o
                low += o if c <= ROUTE_FUSED_MAX_INFLIGHT else 0
        per[(r["doc_id"], r["pass"])] = low / tot if tot else None
    return {"busy_h": busy / 3600, "share_busy_inflight_1to4": sum(v for k, v in acc.items() if 1 <= k <= ROUTE_FUSED_MAX_INFLIGHT) / busy,
            "share_busy_inflight_5plus": sum(v for k, v in acc.items() if k > ROUTE_FUSED_MAX_INFLIGHT) / busy, "max_inflight": max(acc)}, per


# ------------------------------------------------------------------ analysis
def classify(arm_empties, p_one):
    if arm_empties <= BAND_LARGE_MAX:
        band = f"<= {BAND_LARGE_MAX} (operational band: large reduction)"
        if p_one < ALPHA:
            rule = ("R1: large reduction supported by the prespecified analysis = evidence that the correction contributes to "
                    "better completion relative to the recorded baseline (historical control, not a randomized comparison)")
        else:
            rule = "R2: count in the large band without support from the prespecified test = inconclusive; no automatic further pass"
    elif arm_empties <= BAND_MODERATE_MAX:
        band = f"{BAND_LARGE_MAX + 1}-{BAND_MODERATE_MAX} (operational band: moderate reduction)"
        rule = ("R2: moderate reduction without precision = inconclusive; no automatic further pass" if p_one >= ALPHA else
                "R2b (amendment 11b, strong signal required): modest, statistically detectable reduction; not support for the "
                "correction as an explanation of the gap; reported as a contribution of modest size; no automatic further pass")
    else:
        band = f">= {BAND_MODERATE_MAX + 1} (operational band: little improvement or deterioration)"
        rule = ("R3: little improvement or deterioration = the localized correction stands but is not supported as the dominant "
                "completion explanation")
    return band, rule


def per_question(rows, key):
    return [sum(rows[(q, p)][key] for p in range(1, N_PASSES + 1)) for q in range(N_Q)]


def analyze(A, C, a_label, c_label):
    out = {}
    ps = range(1, N_PASSES + 1)
    n = N_Q * N_PASSES
    # ---- primary
    ea, ec = per_question(A, "empty"), per_question(C, "empty")
    d = [c - a for a, c in zip(ea, ec)]
    t = signflip(d)
    lo, hi = boot_interval(d, n, BOOT_SEED)
    rlo, rhi, rund = boot_ratio_interval(ea, ec, BOOT_SEED + 3)
    band, rule = classify(sum(ea), t["p_one_sided"])
    out["primary"] = {
        "arm_empties": sum(ea), "cmp_empties": sum(ec), "n_question_passes": n,
        "arm_empty_rate": sum(ea) / n, "cmp_empty_rate": sum(ec) / n,
        "per_pass": {f"pass{p}": {"arm": sum(A[(q, p)]["empty"] for q in range(N_Q)), "cmp": sum(C[(q, p)]["empty"] for q in range(N_Q))} for p in ps},
        "questions_with_any_empty": {"arm": sum(x > 0 for x in ea), "cmp": sum(x > 0 for x in ec)},
        "questions_d_positive": sum(x > 0 for x in d), "questions_d_negative": sum(x < 0 for x in d),
        "d_nonzero_by_question": {str(q): d[q] for q in range(N_Q) if d[q]},
        "test": {"statistic": "T = sum over 198 questions of d_q = (cmp empties_q - arm empties_q)", **t,
                 "direction": "one-sided, reduction (T large)", "alpha": ALPHA},
        "effect_empty_rate_reduction": {"estimate": sum(d) / n, "ci95": [lo, hi], "bootstrap": {"replicates": BOOT_B, "seed": BOOT_SEED,
                                        "unit": "question (all passes, both arms)", "method": "percentile"}},
        "relative_reduction": {"estimate": (1 - sum(ea) / sum(ec)) if sum(ec) else None, "ci95": [rlo, rhi],
                               "replicates_undefined": rund, "seed": BOOT_SEED + 3},
        "band": band, "reading": rule}
    # ---- secondary: accuracy over all requests
    acc = {}
    for i, key in enumerate(("correct_flexible", "correct_stated")):
        qa, qc = per_question(A, key), per_question(C, key)
        dd = [a - c for a, c in zip(qa, qc)]
        tt = signflip(dd)
        lo, hi = boot_interval(dd, n / 100, BOOT_SEED + 1 + i)
        acc[key] = {"arm_correct": sum(qa), "cmp_correct": sum(qc),
                    "arm_pct_per_pass": {f"pass{p}": 100 * sum(A[(q, p)][key] for q in range(N_Q)) / N_Q for p in ps},
                    "cmp_pct_per_pass": {f"pass{p}": 100 * sum(C[(q, p)][key] for q in range(N_Q)) / N_Q for p in ps},
                    "arm_pct_mean": 100 * sum(qa) / n, "cmp_pct_mean": 100 * sum(qc) / n,
                    "diff_points_arm_minus_cmp": 100 * sum(dd) / n, "ci95_points": [lo, hi], "bootstrap_seed": BOOT_SEED + 1 + i,
                    "signflip_two_sided_p_reference": tt["p_two_sided"], "questions_arm_more": sum(x > 0 for x in dd),
                    "questions_cmp_more": sum(x < 0 for x in dd)}
    acc["note"] = "A nonsignificant accuracy test does not establish no accuracy cost; read the interval."
    acc["stated_status"] = {"arm_rows_stated_differs_from_flexible": sum(r["stated_differs_from_flexible"] for r in A.values()),
                            "cmp_rows_stated_differs_from_flexible": sum(r["stated_differs_from_flexible"] for r in C.values()),
                            "arm_rows_hedged_by_hand": sum(r["hedged_by_hand"] for r in A.values()),
                            "rule": "arm correct_stated is final only after every row where it differs from flexible-extract is read by hand (protocol)"}
    out["accuracy_all_requests"] = acc
    # ---- rescued / introduced empties (paired by question and pass = same request seed; trajectories are not matched)
    keys = [(q, p) for q in range(N_Q) for p in ps]
    resc = [k for k in keys if C[k]["empty"] and not A[k]["empty"]]
    newe = [k for k in keys if A[k]["empty"] and not C[k]["empty"]]
    both = [k for k in keys if A[k]["empty"] and C[k]["empty"]]
    out["empties_paired"] = {
        "rescued_cmp_empty_arm_answered": {"n": len(resc), "ids": [f"q{q}p{p}" for q, p in resc],
                                           "arm_raw_correct": sum(A[k]["correct_flexible"] for k in resc), "arm_stated_correct": sum(A[k]["correct_stated"] for k in resc)},
        "introduced_cmp_answered_arm_empty": {"n": len(newe), "ids": [f"q{q}p{p}" for q, p in newe],
                                              "cmp_raw_correct_lost": sum(C[k]["correct_flexible"] for k in newe), "cmp_stated_correct_lost": sum(C[k]["correct_stated"] for k in newe)},
        "both_empty": {"n": len(both), "ids": [f"q{q}p{p}" for q, p in both]},
        "answered_in_both_raw": {"n": sum(1 for k in keys if not A[k]["empty"] and not C[k]["empty"]),
                                 "arm": sum(A[k]["correct_flexible"] for k in keys if not A[k]["empty"] and not C[k]["empty"]),
                                 "cmp": sum(C[k]["correct_flexible"] for k in keys if not A[k]["empty"] and not C[k]["empty"])},
        "note": "Pairs share prompt and request seed; generations under different engines are not matched trajectories. Descriptive."}
    # ---- finish categories and lengths
    def fin(R):
        cats = {"cap_exhaustion": 0, "premature_stop": 0, "other_finish": 0, "not_logged": 0}
        for r in R.values():
            if not r["empty"]:
                continue
            if not r["logged"] or r["finish_reason"] is None:
                cats["not_logged"] += 1
            elif r["finish_reason"] == "length" or (r["completion_tokens"] or 0) >= CAP:
                cats["cap_exhaustion"] += 1
            elif r["finish_reason"] == "stop":
                cats["premature_stop"] += 1
            else:
                cats["other_finish"] += 1
        allf = {}
        for r in R.values():
            if r["logged"]:
                allf[str(r["finish_reason"])] = allf.get(str(r["finish_reason"]), 0) + 1
        prem = [{"id": f"q{r['doc_id']}p{r['pass']}", "completion_tokens": r["completion_tokens"]} for r in R.values()
                if r["empty"] and r["logged"] and r["finish_reason"] == "stop"]
        ans_len = [r["completion_tokens"] for r in R.values() if r["logged"] and not r["empty"] and r["completion_tokens"] is not None]
        all_len = [r["completion_tokens"] for r in R.values() if r["logged"] and r["completion_tokens"] is not None]
        return {"empties_by_end": cats, "premature_stops": prem, "finish_reason_all_logged": allf,
                "answered_but_length_finish": sum(1 for r in R.values() if r["logged"] and not r["empty"] and r["finish_reason"] == "length"),
                "logged_rows": sum(r["logged"] for r in R.values()),
                "completion_tokens_answered": quantiles(ans_len), "completion_tokens_all_logged": quantiles(all_len),
                "answered_above": {str(x): sum(v > x for v in ans_len) for x in LEN_THRESHOLDS},
                "completion_tokens_total_logged": sum(all_len)}
    out["ends_and_lengths"] = {"arm": fin(A), "cmp": fin(C)}
    both_logged = [q for q in range(N_Q) if all(A[(q, p)]["logged"] and C[(q, p)]["logged"] for p in ps)]
    if both_logged:
        dl = [sum(A[(q, p)]["completion_tokens"] - C[(q, p)]["completion_tokens"] for p in ps) for q in both_logged]
        lo, hi = boot_interval(dl, len(both_logged) * N_PASSES, BOOT_SEED + 4)
        out["ends_and_lengths"]["paired_mean_tokens_arm_minus_cmp"] = {"questions": len(both_logged), "estimate": sum(dl) / (len(both_logged) * N_PASSES),
                                                                       "ci95": [lo, hi], "note": "all logged requests incl. cap exhaustions; descriptive"}
    return out


def timing(R, label, chain, gpu_hours_override, gpus):
    ps = range(1, N_PASSES + 1)
    per = {}
    windows = []
    route = {}
    for p in ps:
        rp = [r for r in R.values() if r["pass"] == p]
        if not all(r["logged"] and r["t_start"] is not None for r in rp):
            per[f"pass{p}"] = None
            continue
        t0 = min(r["t_start"] for r in rp)
        t1 = max(r["t_start"] + r["secs"] for r in rp)
        windows.append((t0, t1))
        tok = sum(r["completion_tokens"] for r in rp)
        rates = [r["completion_tokens"] / r["secs"] for r in rp if r["secs"] and not r["empty"]]
        per[f"pass{p}"] = {"window_h": round((t1 - t0) / 3600, 4), "completion_tokens": tok, "aggregate_tok_s": round(tok / (t1 - t0), 1),
                           "median_request_tok_s_answered": round(statistics.median(rates), 1) if rates else None}
        prof = inflight(rp)
        if prof:
            summ, perreq = prof
            emp = [v for k, v in perreq.items() if R[k]["empty"] and v is not None]
            ans = [v for k, v in perreq.items() if not R[k]["empty"] and v is not None]
            route[f"pass{p}"] = {**{k: round(v, 4) if isinstance(v, float) else v for k, v in summ.items()},
                                 "mean_share_of_request_life_inflight_1to4": {"empty": round(statistics.mean(emp), 4) if emp else None,
                                                                              "answered": round(statistics.mean(ans), 4) if ans else None}}
    win_h = sum(v["window_h"] for v in per.values() if v)
    gh = gpu_hours_override if gpu_hours_override is not None else (chain["gpu_hours"] if chain else None)
    raw = sum(r["correct_flexible"] for r in R.values())
    st_ = sum(r["correct_stated"] for r in R.values())
    return {"per_pass": per, "request_window_h_total": round(win_h, 4), "request_window_gpu_h": round(gpus * win_h, 3),
            "occupied": chain, "gpu_hours_used": gh, "gpu_hours_source": "override" if gpu_hours_override is not None else ("chain logs" if chain else None),
            "raw_correct_per_gpu_hour": round(raw / gh, 2) if gh else None, "stated_correct_per_gpu_hour": round(st_ / gh, 2) if gh else None,
            "kernel_route_exposure_client_side": route}, windows


def check_pairing(A, C):
    c = []
    keys = [(q, p) for q in range(N_Q) for p in range(1, N_PASSES + 1)]
    have = all(k in A for k in keys) and all(k in C for k in keys)
    c.append(("both arms have all 594 question-passes", have, f"arm {len(A)}, cmp {len(C)}"))
    if not have:
        return c
    for f in ("seed", "prompt_hash", "doc_hash", "target_hash", "messages_sha256"):
        bad = [k for k in keys if A[k][f] != C[k][f]]
        c.append((f"(question, pass) pairs agree on {f}", not bad, f"{len(bad)} mismatches"))
    pt = [k for k in keys if A[k]["prompt_tokens"] is not None and C[k]["prompt_tokens"] is not None]
    bad = [k for k in pt if A[k]["prompt_tokens"] != C[k]["prompt_tokens"]]
    c.append(("server-rendered prompt token counts agree where both logged", not bad, f"{len(pt)} compared, {len(bad)} differ"))
    return c


def adjudicator_status():
    try:
        lines = open(ADJ_SOURCE).read().splitlines(keepends=True)[ADJ_LINES[0] - 1:ADJ_LINES[1]]
        cur = hashlib.sha256("".join(lines).encode()).hexdigest()
        return {"source": ADJ_SOURCE, "frozen_sha256": ADJ_SHA, "current_source_sha256": cur, "source_unchanged": cur == ADJ_SHA}
    except OSError:
        return {"source": ADJ_SOURCE, "frozen_sha256": ADJ_SHA, "current_source_sha256": None, "source_unchanged": None}


def fmt_p(x):
    return f"{x:.4f}" if x >= 1e-4 else f"{x:.2e}"


def markdown(res):
    P = res["primary"]
    L = []
    L.append(f"# Column-511 arm analysis: {res['labels']['arm']} vs {res['labels']['cmp']}")
    if res.get("dry_run"):
        L.append(f"\n**DRY RUN: {res['dry_run']}. Not a result of the column-511 arm.**")
    L.append(f"\nscript sha256 {res['script_sha256']}; spec sha256 {res['spec_sha256']}; valid: **{res['valid']}**")
    L.append("\n## Checks")
    for name, ok, det in res["checks"]:
        L.append(f"- [{'ok' if ok else 'FAIL'}] {name}" + (f" ({det})" if det not in ('', None) else ""))
    L.append("\n## Primary: empty replies, 594 question-passes per arm")
    L.append(f"- arm {P['arm_empties']} ({100 * P['arm_empty_rate']:.2f}%), comparator {P['cmp_empties']} ({100 * P['cmp_empty_rate']:.2f}%); "
             f"per pass arm/cmp: " + ", ".join(f"{k} {v['arm']}/{v['cmp']}" for k, v in P["per_pass"].items()))
    L.append(f"- questions with any empty: arm {P['questions_with_any_empty']['arm']}, cmp {P['questions_with_any_empty']['cmp']}; "
             f"d_q > 0 on {P['questions_d_positive']} questions, < 0 on {P['questions_d_negative']}")
    t = P["test"]
    L.append(f"- exact sign-flip, T = {t['T']}, m = {t['m_nonzero']} nonzero questions ({t['null_patterns']} sign patterns): "
             f"one-sided p = {fmt_p(t['p_one_sided'])} (two-sided {fmt_p(t['p_two_sided'])}, reference)")
    e = P["effect_empty_rate_reduction"]
    L.append(f"- empty-rate reduction (cmp - arm) {100 * e['estimate']:+.2f} points, 95% question-bootstrap interval "
             f"{100 * e['ci95'][0]:+.2f} to {100 * e['ci95'][1]:+.2f} ({BOOT_B} replicates, seed {BOOT_SEED})")
    r = P["relative_reduction"]
    if r["estimate"] is not None:
        L.append(f"- relative reduction {100 * r['estimate']:.0f}% (interval {100 * r['ci95'][0]:.0f}% to {100 * r['ci95'][1]:.0f}%; "
                 f"{r['replicates_undefined']} replicates undefined)")
    L.append(f"- band: {P['band']}")
    L.append(f"- reading: {P['reading']}")
    L.append("\n## Secondary")
    for key, lab in (("correct_flexible", "raw (flexible-extract)"), ("correct_stated", "stated (audited method)")):
        a = res["accuracy_all_requests"][key]
        L.append(f"- {lab}, all 594 requests: arm {a['arm_correct']} ({a['arm_pct_mean']:.1f}%), cmp {a['cmp_correct']} ({a['cmp_pct_mean']:.1f}%); "
                 f"arm - cmp {a['diff_points_arm_minus_cmp']:+.2f} points (95% {a['ci95_points'][0]:+.2f} to {a['ci95_points'][1]:+.2f}); "
                 f"two-sided sign-flip p {fmt_p(a['signflip_two_sided_p_reference'])} (reference). Per pass arm: "
                 + ", ".join(f"{v:.1f}" for v in a["arm_pct_per_pass"].values()) + "; cmp: " + ", ".join(f"{v:.1f}" for v in a["cmp_pct_per_pass"].values()))
    L.append(f"- {res['accuracy_all_requests']['note']}")
    s = res["accuracy_all_requests"]["stated_status"]
    L.append(f"- stated differs from raw on {s['arm_rows_stated_differs_from_flexible']} arm rows (hand read required before the stated figure is final), "
             f"{s['cmp_rows_stated_differs_from_flexible']} comparator rows")
    E = res["empties_paired"]
    L.append(f"- rescued (cmp empty, arm answered): {E['rescued_cmp_empty_arm_answered']['n']} "
             f"({', '.join(E['rescued_cmp_empty_arm_answered']['ids']) or '-'}); arm raw correct on them {E['rescued_cmp_empty_arm_answered']['arm_raw_correct']}")
    L.append(f"- introduced (cmp answered, arm empty): {E['introduced_cmp_answered_arm_empty']['n']} "
             f"({', '.join(E['introduced_cmp_answered_arm_empty']['ids']) or '-'}); cmp raw correct lost {E['introduced_cmp_answered_arm_empty']['cmp_raw_correct_lost']}")
    L.append(f"- empty in both: {E['both_empty']['n']} ({', '.join(E['both_empty']['ids']) or '-'}); answered in both: {E['answered_in_both_raw']['n']}, "
             f"raw correct arm {E['answered_in_both_raw']['arm']} vs cmp {E['answered_in_both_raw']['cmp']}")
    for side in ("arm", "cmp"):
        f = res["ends_and_lengths"][side]
        L.append(f"- {side} empties by end: {f['empties_by_end']}; premature stops: {f['premature_stops'] or '-'}; logged rows {f['logged_rows']}; "
                 f"answered with finish 'length': {f['answered_but_length_finish']}")
        q = f["completion_tokens_answered"]
        if q:
            L.append(f"  - answered completion tokens: median {q['p50']}, p90 {q['p90']}, p99 {q['p99']}, max {q['max']} (n {q['n']}); "
                     f"answered above 32k/64k/128k/200k: " + "/".join(str(v) for v in f["answered_above"].values()))
    if "paired_mean_tokens_arm_minus_cmp" in res["ends_and_lengths"]:
        m = res["ends_and_lengths"]["paired_mean_tokens_arm_minus_cmp"]
        L.append(f"- mean completion tokens per request, arm - cmp, {m['questions']} questions logged in all passes: {m['estimate']:+.0f} "
                 f"(95% {m['ci95'][0]:+.0f} to {m['ci95'][1]:+.0f})")
    for side in ("arm", "cmp"):
        T = res["timing"][side]
        L.append(f"- {side} cost: request windows {T['request_window_h_total']:.2f} h ({T['request_window_gpu_h']:.2f} GPU-h at {res['gpus']} GPUs); "
                 f"occupied GPU-h {T['gpu_hours_used'] if T['gpu_hours_used'] is not None else 'n/a'} ({T['gpu_hours_source'] or 'not available'}); "
                 f"raw correct per GPU-h {T['raw_correct_per_gpu_hour'] if T['raw_correct_per_gpu_hour'] is not None else 'n/a'}")
        for k, v in T["per_pass"].items():
            if v:
                L.append(f"  - {k}: window {v['window_h']:.2f} h, {v['completion_tokens']:,} tokens, {v['aggregate_tok_s']} tok/s aggregate, "
                         f"median answered request {v['median_request_tok_s_answered']} tok/s")
        for k, v in T["kernel_route_exposure_client_side"].items():
            L.append(f"  - route exposure {k} (client-side in-flight, approximate): share of busy time with 1-4 in flight "
                     f"{v['share_busy_inflight_1to4']:.3f}, 5+ {v['share_busy_inflight_5plus']:.3f}; mean share of request life at 1-4: "
                     f"empty {v['mean_share_of_request_life_inflight_1to4']['empty']}, answered {v['mean_share_of_request_life_inflight_1to4']['answered']}")
    for side in ("arm", "cmp"):
        for s_ in res["server"][side]:
            L.append(f"- {side} server log {s_['log']}: starts {s_['server_starts']}, cited block {s_.get('cited_block')}, "
                     f"non-default args sha256 {str(s_.get('non_default_args_sha256'))[:16]}, cudagraph {s_.get('cudagraph_mode')}, "
                     f"KV pool {s_.get('kv_pool_tokens')}, status {s_.get('status')}, in window {s_.get('in_window')}")
    L.append("\nLimits: historical control; exchangeability is assumed, not designed; matching recipes and seeds does not make this "
             "a randomized experiment; a positive result supports improvement relative to the recorded baseline only. Corrupted/NaN "
             "index rows are not measurable from GPQA outputs and are reported separately where instrumentation exists. The "
             "unresolved tail-ring defect is present in both arms and limits what a negative result could exclude.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for side in ("arm", "cmp"):
        ap.add_argument(f"--{side}-label", required=True)
        ap.add_argument(f"--{side}-pass", nargs=N_PASSES, required=True)
        ap.add_argument(f"--{side}-reqlog", nargs="*", default=[])
        ap.add_argument(f"--{side}-serverlog", nargs="*", default=[])
        ap.add_argument(f"--{side}-chainlog", nargs="*", default=[])
        ap.add_argument(f"--{side}-hedged", nargs="*", default=[], help="PASS:DOC replies read by hand as committing to no single option")
        ap.add_argument(f"--{side}-published", default=None)
        ap.add_argument(f"--{side}-gpu-hours", type=float, default=None)
    ap.add_argument("--server-block", type=int, default=0)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run-label", default=None)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    spec = os.path.join(here, "analysis-col511.md")
    A, ca, ia = load_arm(a.arm_label, a.arm_pass, a.arm_reqlog, a.arm_hedged, a.arm_published)
    C, cc, ic = load_arm(a.cmp_label, a.cmp_pass, a.cmp_reqlog, a.cmp_hedged, a.cmp_published)
    checks = ca + cc + check_pairing(A, C)
    adj = adjudicator_status()
    checks.append(("stated adjudicator source unchanged since freeze (informative)", bool(adj["source_unchanged"]), adj["current_source_sha256"]))
    gating = [c for c in checks if not c[0].startswith("stated adjudicator")]
    valid = all(ok for _, ok, _ in gating)
    res = {"labels": {"arm": a.arm_label, "cmp": a.cmp_label}, "dry_run": a.dry_run_label,
           "script_sha256": file_sha(os.path.abspath(__file__)), "spec_sha256": file_sha(spec) if os.path.exists(spec) else None,
           "inputs": {"arm": ia, "cmp": ic}, "adjudicator": adj, "gpus": a.gpus,
           "checks": [list(c) for c in checks], "valid": valid}
    if len(A) == len(C) == N_Q * N_PASSES:
        res.update(analyze(A, C, a.arm_label, a.cmp_label))
        res["timing"], res["server"] = {}, {}
        for side, R, chainp, gho, srv in (("arm", A, a.arm_chainlog, a.arm_gpu_hours, a.arm_serverlog),
                                          ("cmp", C, a.cmp_chainlog, a.cmp_gpu_hours, a.cmp_serverlog)):
            chain = parse_chainlogs(chainp, a.gpus)
            res["timing"][side], windows = timing(R, side, chain, gho, a.gpus)
            res["server"][side] = parse_serverlogs(srv, windows, a.server_block)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, sort_keys=True)
    if "primary" in res:
        print(markdown(res))
    else:
        print(json.dumps(res["checks"], indent=1))
    return 0 if valid else 3


if __name__ == "__main__":
    sys.exit(main())
