#!/usr/bin/env python3
"""Per-window trajectory metrics for long rep070-F replies (zero GPU, CPU-only container, no network).

Definitions: ../PRESPEC.md (sha256 checked below; written before any metric was computed).
Reads the raw request/response file read-only, re-tokenizes reasoning and content with the K3.25 tokenizer
(add_special_tokens=False). Writes ids, positions, counts and metrics only -- never reasoning, content or GPQA text.

Run (see run_trajectory.sh):
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= --user UID:GID --entrypoint python3 \
    -v RAW:/raw:ro -v SNAP:/tok:ro -v DISCOVERY:/work  IMAGE /work/scripts/trajectory.py
"""
import gzip, hashlib, json, os, re, sys, zlib
from multiprocessing import Pool

import numpy as np
from tokenizers import Tokenizer

WORK = "/work"
RAW = "/raw"
TOK = "/repo/snapshots/701cd7456c13d87bf0147ad946f828a999afb59c/tokenizer.json"
PRESPEC_SHA = "97040a5c6260b2447917dfaf48c87605c066ebc57de0199f42571ea01c96c4b7"
W = 4096
THR = 0.5
SENS = (0.3, 0.7)
EPI_MIN = 2048
SPAN_MIN = 256
SHORT_P = 4096
NS = (20, 50, 100)
SEL_MIN = 50000

CJK = re.compile(r"[⺀-鿿가-힯豈-﫿＀-￯　-〿]")
# verification / self-check phrases (generic English; counts only)
CHK_VERIFY = re.compile(
    r"(?i)\b(?:let me|let's|let us|i(?:'ll| will| should| need to| want to| must)|we (?:should|need to|must))\s+"
    r"(?:double[- ]?check|re-?check|verify|check|re-?verify|confirm|re-?examine|reconsider|re-?evaluate|"
    r"re-?calculate|recompute|re-?read|re-?think|re-?derive|go back)\b|\bdouble[- ]?check(?:ing)?\b|\bsanity check\b")
CHK_WAIT = re.compile(r"(?i)\b(?:but )?wait\b|\bhmm+\b|\bhold on\b|\bactually,|\bon second thought\b")
SENT_END = re.compile(r"[.!?\n]")


def ngram_hash(t, n):
    m = len(t) - n + 1
    if m <= 0:
        return np.zeros(0, dtype=np.uint64)
    x = t.astype(np.uint64) + np.uint64(1)
    h = np.zeros(m, dtype=np.uint64)
    B = np.uint64(1000003)
    with np.errstate(over="ignore"):
        for k in range(n):
            h = h * B + x[k:k + m]
    return h


def rep_and_prev(h):
    """rep[i]: n-gram at i occurred at some j < i. prev[i]: nearest such j (-1 if none)."""
    m = len(h)
    idx = np.arange(m)
    order = np.lexsort((idx, h))
    hs = h[order]
    prev_s = np.full(m, -1, dtype=np.int64)
    same = hs[1:] == hs[:-1]
    prev_s[1:][same] = order[:-1][same]
    prev = np.empty(m, dtype=np.int64)
    prev[order] = prev_s
    return prev >= 0, prev


def spans_from(rep, prev, n):
    """Maximal runs of consecutive repeated starts with constant distance. Returns arrays start, length(tokens), period."""
    m = len(rep)
    if m == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
    d = np.where(rep, np.arange(m) - prev, 0)
    cont = np.zeros(m, dtype=bool)
    cont[1:] = rep[1:] & rep[:-1] & (d[1:] == d[:-1])
    starts = np.where(rep & ~cont)[0]
    # run end: next start of a new run or a non-rep
    brk = np.where(~cont)[0]  # positions where a run cannot continue from i-1
    nxt = np.searchsorted(brk, starts, side="right")
    ends = np.where(nxt < len(brk), brk[np.minimum(nxt, len(brk) - 1)], m)  # exclusive end of run
    run = ends - starts
    return starts, run + n - 1, d[starts]


def windows(N):
    return [(s, min(s + W, N)) for s in range(0, N, W)]


def onset_and_episodes(R, thr):
    if len(R) == 0:
        return None, []
    hi = R >= thr
    onset = None
    if hi[-1]:
        lo = np.where(~hi)[0]
        onset = int(lo[-1] + 1) if len(lo) else 0
    eps = []
    edges = np.diff(np.concatenate([[0], hi.astype(np.int8), [0]]))
    st = np.where(edges == 1)[0]
    en = np.where(edges == -1)[0]
    for a, b in zip(st, en):
        if b - a >= EPI_MIN:
            eps.append({"start": int(a), "end": int(b), "len": int(b - a), "terminal": bool(b == len(R)),
                        "max_R50": round(float(R[a:b].max()), 4)})
    return onset, eps


def analyse(rec):
    with gzip.open(f"{RAW}/{rec['file']}.raw.jsonl.gz", "rt") as fi:
        for ln, line in enumerate(fi):
            if ln == rec["line"]:
                d = json.loads(line)
                break
    msg = d["response"]["choices"][0]["message"]
    text = msg.get("reasoning") or ""
    content = msg.get("content") or ""
    tok = Tokenizer.from_file(TOK)
    enc = tok.encode(text, add_special_tokens=False)
    ids = np.array(enc.ids, dtype=np.int64)
    offs = np.array(enc.offsets, dtype=np.int64).reshape(-1, 2)
    N = len(ids)
    ctoks = len(tok.encode(content, add_special_tokens=False).ids) if content else 0
    out = {k: rec[k] for k in ("id", "doc", "pass", "seed", "T", "prompt_tokens", "finish_reason", "stop_reason")}
    out.update(N_reasoning=N, content_tokens=ctoks, content_chars=len(content), reasoning_chars=len(text),
               drift=rec["T"] - (N + ctoks), offsets_cover_text=bool(N and offs[-1, 1] == len(text)),
               text_sha256=hashlib.sha256(text.encode()).hexdigest())
    rep, prev, R = {}, {}, {}
    verify_fail = 0
    rng = np.random.default_rng(rec["line"] + 7)
    for n in NS:
        h = ngram_hash(ids, n)
        rep[n], prev[n] = rep_and_prev(h)
        hits = np.where(rep[n])[0]
        for i in rng.choice(hits, size=min(50, len(hits)), replace=False) if len(hits) else []:
            j = prev[n][i]
            if not np.array_equal(ids[i:i + n], ids[j:j + n]):
                verify_fail += 1
    out["hash_verify_failures"] = verify_fail
    r50 = rep[50].astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(r50)])
    M = len(r50)
    R50 = (cs[W:] - cs[:-W]) / W if M >= W else np.zeros(0)
    onset, eps = onset_and_episodes(R50, THR)
    sens = {}
    for t in SENS:
        o, e = onset_and_episodes(R50, t)
        sens[str(t)] = {"onset": o, "episodes": len(e), "recovered": sum(not x["terminal"] for x in e)}
    s_st, s_len, s_per = spans_from(rep[20], prev[20], 20)
    big = s_len >= SPAN_MIN
    out["longest_span"] = {"len": int(s_len.max()) if len(s_len) else 0,
                           "period": int(s_per[s_len.argmax()]) if len(s_len) else None,
                           "start": int(s_st[s_len.argmax()]) if len(s_len) else None}
    loop = {"class": "no terminal loop"}
    if onset is not None:
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
        P = max(per_tokens, key=per_tokens.get) if per_tokens else None
        top = sorted(per_tokens.items(), key=lambda kv: -kv[1])[:3]
        if E >= 0.5 and P is not None and P <= SHORT_P:
            cl = "short-period exact loop"
        elif E >= 0.5:
            cl = "long-period exact cycling"
        else:
            cl = "fragmentary recycling"
        loop = {"class": cl, "E": round(E, 4), "P": P, "top_periods": top,
                "n_distinct_periods_ge256": len(per_tokens)}
    out.update(onset=onset, onset_abs=(onset + rec["prompt_tokens"]) if onset is not None else None,
               episodes=eps, sensitivity=sens, loop=loop,
               R50_final=round(float(R50[-1]), 4) if len(R50) else None,
               R50_max=round(float(R50.max()), 4) if len(R50) else None,
               R50_at={str(g): (round(float(R50[g]), 4) if g < len(R50) else None) for g in (50000, 96000, 100000)})
    # coarse R50 trajectory every 1,024 tokens (numbers only)
    out["R50_traj_1k"] = [round(float(R50[g]), 3) for g in range(0, len(R50), 1024)]

    # per-window metrics
    starts_c = offs[:, 0]
    # lines with start char offsets
    lines, pos = [], 0
    for ln_ in text.split("\n"):
        lines.append((pos, ln_.strip()))
        pos += len(ln_) + 1
    line_tok = np.searchsorted(starts_c, [p for p, _ in lines], side="right") - 1
    seen_lines = set()
    # verification phrases
    def phrase_events(rx):
        ev, seen, outl = [], set(), []
        for m in rx.finditer(text):
            a = m.start()
            ps = text.rfind(".", 0, a), text.rfind("\n", 0, a), text.rfind("?", 0, a), text.rfind("!", 0, a)
            s0 = max(ps) + 1
            mm = SENT_END.search(text, m.end())
            s1 = mm.start() if mm else len(text)
            sent = " ".join(text[s0:s1].lower().split())
            outl.append((int(np.searchsorted(starts_c, a, side="right") - 1), sent in seen))
            seen.add(sent)
        return outl
    ver = phrase_events(CHK_VERIFY)
    wai = phrase_events(CHK_WAIT)
    ver_pos = np.array([p for p, _ in ver], dtype=np.int64)
    ver_rep = np.array([r for _, r in ver], dtype=bool)
    wai_pos = np.array([p for p, _ in wai], dtype=np.int64)
    wai_rep = np.array([r for _, r in wai], dtype=bool)
    rows = []
    li = 0
    for k, (s, e) in enumerate(windows(N)):
        c0, c1 = int(offs[s, 0]), int(offs[e - 1, 1])
        seg = text[c0:c1]
        ntok = e - s
        row = {"id": out["id"], "k": k, "g0": s, "g1": e, "partial": ntok < 1024,
               "cpt": round(len(seg) / ntok, 4)}
        for n in NS:
            m_ = len(rep[n])
            a, b = min(s, m_), min(e, m_)
            row[f"rep{n}"] = round(float(rep[n][a:b].mean()), 4) if b > a else None
        sel = (s_st >= s) & (s_st < e)
        if sel.any():
            j = np.argmax(np.where(sel, s_len, -1))
            row["span_max"], row["span_period"] = int(s_len[j]), int(s_per[j])
        else:
            row["span_max"], row["span_period"] = 0, None
        b = seg.encode("utf-8")
        row["zr"] = round(len(zlib.compress(b, 9)) / max(1, len(b)), 4)
        # lines starting in this window
        win_lines = []
        while li < len(lines) and line_tok[li] < e:
            if line_tok[li] >= s:
                win_lines.append(lines[li][1])
            li += 1
        ne = [x for x in win_lines if x]
        row["lines"] = len(ne)
        row["dlr"] = round(len(set(ne)) / len(ne), 4) if ne else None
        long_ = [x for x in ne if len(x) >= 8]
        row["line_seen"] = round(sum(x in seen_lines for x in long_) / len(long_), 4) if long_ else None
        seen_lines.update(ne)
        nch = max(1, len(seg))
        row["cjk"] = round(len(CJK.findall(seg)) / nch, 4)
        row["nonascii"] = round(sum(ord(ch) > 127 for ch in seg) / nch, 4)
        row["alpha"] = round(sum(ch.isascii() and ch.isalpha() for ch in seg) / nch, 4)
        row["digit"] = round(sum(ch.isdigit() for ch in seg) / nch, 4)
        row["space"] = round(sum(ch.isspace() for ch in seg) / nch, 4)
        row["punct"] = round(sum(ch.isascii() and not ch.isalnum() and not ch.isspace() for ch in seg) / nch, 4)
        lens = offs[s:e, 1] - offs[s:e, 0]
        row["tok_le1char"] = round(float((lens <= 1).mean()), 4)
        sv = (ver_pos >= s) & (ver_pos < e)
        sw = (wai_pos >= s) & (wai_pos < e)
        row["chk_verify_per1k"] = round(1000 * int(sv.sum()) / ntok, 3)
        row["chk_verify_rep"] = round(float(ver_rep[sv].mean()), 4) if sv.any() else None
        row["chk_wait_per1k"] = round(1000 * int(sw.sum()) / ntok, 3)
        row["chk_wait_rep"] = round(float(wai_rep[sw].mean()), 4) if sw.any() else None
        rows.append(row)
    out["n_windows"] = len(rows)
    out["chk_verify_total"], out["chk_verify_rep_share"] = len(ver), round(float(ver_rep.mean()), 4) if len(ver) else None
    out["chk_wait_total"], out["chk_wait_rep_share"] = len(wai), round(float(wai_rep.mean()), 4) if len(wai) else None
    # last high-repetition grid window (finished replies, Q3)
    for t in (THR, 0.25):
        hiw = [r for r in rows if r["rep50"] is not None and r["rep50"] >= t]
        out[f"last_hiwin_{t}"] = ({"k": hiw[-1]["k"], "end": hiw[-1]["g1"], "dist_to_reasoning_end": N - hiw[-1]["g1"]}
                                  if hiw else None)
    out["overall"] = {"cpt": round(len(text) / max(1, N), 4),
                      "rep50": round(float(rep[50].mean()), 4) if len(rep[50]) else None,
                      "cjk": round(len(CJK.findall(text)) / max(1, len(text)), 4)}
    return out, rows


def main():
    ps = hashlib.sha256(open(f"{WORK}/PRESPEC.md", "rb").read()).hexdigest()
    if ps != PRESPEC_SHA:
        sys.exit(f"PRESPEC.md changed since it was frozen: {ps}")
    idx = json.load(open(f"{WORK}/out/rep070F_index.json"))
    sel = [r for r in idx if r["T"] >= SEL_MIN]
    only = os.environ.get("ONLY")  # e.g. "capped" to process capped replies first
    if only == "capped":
        sel = [r for r in sel if r["finish_reason"] == "length"]
    with Pool(min(12, len(sel))) as pool:
        res = pool.map(analyse, sel, chunksize=1)
    tag = os.environ.get("TAG", "all")
    with open(f"{WORK}/out/replies_{tag}.json", "w") as fo:
        json.dump([r for r, _ in res], fo, indent=1)
    with open(f"{WORK}/out/windows_{tag}.jsonl", "w") as fo:
        for _, rows in res:
            for w in rows:
                fo.write(json.dumps(w) + "\n")
    for r, _ in res:
        print(r["id"], r["T"], r["finish_reason"], "N", r["N_reasoning"], "drift", r["drift"], "onset", r["onset"],
              r["loop"]["class"], "verifyfail", r["hash_verify_failures"])


if __name__ == "__main__":
    main()
