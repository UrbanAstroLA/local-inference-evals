#!/usr/bin/env python3
"""POST HOC supplement to trajectory.py (not in PRESPEC.md): exact periodicity and longest repeated substring.

Why: the pre-specified span measure (nearest earlier 20-gram, constant distance) breaks a long cycle into short spans
when the cycle repeats material internally, so "fragmentary recycling" (R50 = 1 with short spans) needed a second,
definition-free look. Two measures, ids and numbers only:
  - LRS: longest repeated substring of the token sequence (overlapping occurrences allowed), via suffix array + LCP,
    for the whole reasoning and for the post-onset region; reported with the offset between the two occurrences.
  - lag match rate m(p) = share of i in a region with t[i] == t[i-p], screened for all p <= 65,536 with a random-phase
    FFT autocorrelation and then computed exactly for the top candidates. "Exact period" = smallest p with exact
    m(p) >= 0.8.
Runs in the same CPU-only container as trajectory.py.
"""
import gzip, json, os, sys
from multiprocessing import Pool

import numpy as np
from tokenizers import Tokenizer

WORK, RAW = "/work", "/raw"
TOK = "/repo/snapshots/701cd7456c13d87bf0147ad946f828a999afb59c/tokenizer.json"
MAXLAG = 65536


def suffix_array(t):
    n = len(t)
    _, rank = np.unique(t, return_inverse=True)
    rank = rank.astype(np.int64)
    k = 1
    sa = np.argsort(rank, kind="stable")
    while True:
        r2 = np.full(n, -1, dtype=np.int64)
        r2[:n - k] = rank[k:]
        sa = np.lexsort((r2, rank))
        key = rank[sa] * (n + 2) + (r2[sa] + 1)
        new = np.empty(n, dtype=np.int64)
        new[sa] = np.concatenate([[0], np.cumsum(key[1:] != key[:-1])])
        rank = new
        if rank.max() == n - 1:
            break
        k *= 2
    return sa


def lrs(t):
    """Longest repeated substring: (length, pos_a, pos_b) with pos_a < pos_b (Kasai LCP)."""
    n = len(t)
    if n < 2:
        return 0, None, None
    sa = suffix_array(t)
    rank = np.empty(n, dtype=np.int64)
    rank[sa] = np.arange(n)
    tl, sal, rk = t.tolist(), sa.tolist(), rank.tolist()
    h, best, bi = 0, 0, -1
    for i in range(n):
        r = rk[i]
        if r > 0:
            j = sal[r - 1]
            while i + h < n and j + h < n and tl[i + h] == tl[j + h]:
                h += 1
            if h > best:
                best, bi = h, r
            if h > 0:
                h -= 1
        else:
            h = 0
    a, b = sorted((sal[bi], sal[bi - 1]))
    return best, a, b


def lag_profile(t):
    n = len(t)
    maxlag = min(MAXLAG, n // 2)
    if maxlag < 2:
        return []
    rng = np.random.default_rng(12345)
    acc = np.zeros(maxlag + 1)
    L = 1 << int(np.ceil(np.log2(2 * n)))
    for _ in range(3):
        u = rng.random(int(t.max()) + 1)
        z = np.exp(2j * np.pi * u[t])
        F = np.fft.fft(z, L)
        ac = np.fft.ifft(F * np.conj(F))[: maxlag + 1].real
        acc += ac
    acc /= 3
    m = acc[1:] / (n - np.arange(1, maxlag + 1))
    cand = np.argsort(-m)[:40] + 1
    exact = {}
    for p in sorted(set(int(c) for c in cand)):
        exact[p] = float((t[p:] == t[:-p]).mean())
    top = sorted(exact.items(), key=lambda kv: -kv[1])[:5]
    per = min([p for p, v in exact.items() if v >= 0.8], default=None)
    return {"top_lags": [[p, round(v, 4)] for p, v in top], "exact_period_m08": per}


def analyse(args):
    rec, onset = args
    with gzip.open(f"{RAW}/{rec['file']}.raw.jsonl.gz", "rt") as fi:
        for ln, line in enumerate(fi):
            if ln == rec["line"]:
                d = json.loads(line)
                break
    text = d["response"]["choices"][0]["message"].get("reasoning") or ""
    t = np.array(Tokenizer.from_file(TOK).encode(text, add_special_tokens=False).ids, dtype=np.int64)
    out = {"id": rec["id"]}
    L, a, b = lrs(t)
    out["lrs_full"] = {"len": L, "start_a": a, "start_b": b, "offset": (b - a) if a is not None else None}
    if onset is not None and len(t) - onset >= 1024:
        reg = t[onset:]
        L, a, b = lrs(reg)
        out["lrs_post_onset"] = {"len": L, "start_a": a + onset, "start_b": b + onset, "offset": b - a}
        out["lag_post_onset"] = lag_profile(reg)
    # pre-onset (or whole reply when no onset): profile of the last 32k tokens before onset/end
    end = onset if onset is not None else len(t)
    reg = t[max(0, end - 32768):end]
    out["lag_last32k_before_onset_or_end"] = lag_profile(reg) if len(reg) >= 2048 else None
    return out


def main():
    idx = {r["id"]: r for r in json.load(open(f"{WORK}/out/rep070F_index.json"))}
    reps = json.load(open(f"{WORK}/out/replies_all.json"))
    jobs = [(idx[r["id"]], r["onset"]) for r in reps]
    with Pool(12) as pool:
        res = pool.map(analyse, jobs, chunksize=1)
    json.dump(res, open(f"{WORK}/out/periods_all.json", "w"), indent=1)
    for r in res:
        print(r["id"], r["lrs_full"], r.get("lrs_post_onset"), r.get("lag_post_onset"))


if __name__ == "__main__":
    main()
