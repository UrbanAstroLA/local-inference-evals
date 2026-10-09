#!/usr/bin/env python3
"""Analyses recomputed from published rows only (standard library).

    python3 tools/analyze.py gpqa-table        # every GPQA run (pass 1): raw accuracy with 95% interval, the audited stated-answer score,
                                               # accuracy among answered questions, empty answers and their finish reasons
    python3 tools/analyze.py gpqa-pairs [A B]  # question-paired comparisons of two runs' pass 1 (default: the pairs in PAIRS below):
                                               # discordant questions, exact McNemar p, difference with a 95% interval over questions
    python3 tools/analyze.py screen-v2         # hard-prompt-screen/v2 (component screen): counts, Wilson intervals, the
                                               # preregistered per-switch Fisher tests with Newcombe intervals, the doc-79 rule,
                                               # the fixed-seed control, and the logistic fit
    python3 tools/analyze.py screen-single     # hard-prompt-screen v0/v1: the single draw (repeat 1) of each question per
                                               # configuration; outcomes only, never rates
    python3 tools/analyze.py screen-anatomy    # v2 screens: loop vs exhaustion per question, where the early stop fired,
                                               # the detector's margin on finished requests
    python3 tools/analyze.py screen-power      # component screen: smallest differences detectable with 80% power (Fisher, p < 0.05)
    python3 tools/analyze.py decode-prefill    # decode vs prefill: KL by region for every run; repeat spread; where a repeat diverges
    python3 tools/analyze.py screen-speed      # median completion tokens/s of finished requests in the v2 screens

Configurations are named by their labels (SCHEMA.md, "Labels") and config ids. GPQA runs are pass 1, request seed 1234
(protocols/gpqa-diamond/v1.md).
"""
import json, math, random, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from verify import config_label, wilson  # noqa: E402

F = "glm53-flash/"
# Question-paired GPQA comparisons reported in FINDINGS.md: configurations that differ in one named respect.
PAIRS = [("k3.25-v0.7.0-dflash5", "k3.25-v0.8.0-dflash3", "engine release as shipped (0.7.0 vs 0.8.0)"),
         ("k3.25-v0.8.0-dflash3", "k3.25-v0.8.0-dflash5-noshare", "draft depth 3 vs 5 with slot sharing off"),
         ("k3.25-v0.7.0-dflash5", "k3.25-v0.7.0-kpoolfix-dflash5", "kpool fixes on 0.7.0"),
         ("k4-v0.8.0-dflash3", "k4-v0.9.0-dflash3", "0.8.0 vs 0.9.0 (kpool fixes)"),
         ("k4-v0.9.0-dflash3", "k4-v0.9.1-dflash3", "0.9.0 vs 0.9.1 (DCP1 tail fix)"),
         ("k3.25-v0.8.0-dflash3", "k4-v0.8.0-dflash3", "weights on 0.8.0"),
         ("k3.25-v0.9.1-dflash3", "k4-v0.9.1-dflash3", "weights on 0.9.1")]
# hard-prompt-screen/v2 component screen (investigations/2026-10-glm53-looping/component-screen)
ARMS88 = {"B": "k3.25-v0.9.1-dflash3", "EN": "k3.25-v0.9.1-ep2-nonope-dflash3", "EO": "k3.25-v0.9.1-ep2-owntp-dflash3",
          "NO": "k3.25-v0.9.1-nonope-owntp-dflash3"}
SWITCHES = {"EP2 routed experts": ("EN", "EO"), "NOPE records off": ("EN", "NO"), "MLA ownership tp (+ slot sharing off)": ("EO", "NO")}
ARMS79 = {"B79": "k3.25-v0.9.1-dflash3", "V79": "k3.25-v0.7.0-dflash3"}


def runs(protocol_prefix):
    for d in sorted((ROOT / "runs").iterdir()):
        m = json.loads((d / "run.json").read_text())
        if m["protocol"].startswith(protocol_prefix):
            m["rows"] = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines() if l.strip()]
            m["cfg"] = json.loads((ROOT / "configs" / f"{m['config']}.json").read_text())
            yield m


def name(m):
    return f"{config_label(m['cfg'])} ({m['config'].split('/', 1)[1]})"


def failed(r):
    return r["cls"] in ("loop", "exhaust")


def sign_test(a, b):
    """Two-sided exact binomial test of a vs b discordant counts (= exact McNemar)."""
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2 ** n)


def fisher(a, b, c, d):
    """Two-sided Fisher exact test for the 2x2 table [[a, b], [c, d]]."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    p = lambda x: math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    p0 = p(a)
    return sum(p(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if p(x) <= p0 * (1 + 1e-9))


def newcombe(k1, n1, k2, n2):
    """95% interval for p1 - p2 from two Wilson intervals (Newcombe's method 10)."""
    p1, p2 = k1 / n1, k2 / n2; (l1, u1), (l2, u2) = wilson(k1, n1), wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


# ---------------------------------------------------------------- GPQA

def gpqa():
    return {m["config"].split("/", 1)[1]: m for m in runs("gpqa-diamond/")}


def gpqa_table():
    print("| config | raw accuracy (flexible-extract) | 95% interval over questions | raw, answered only | stated answer (audited) | "
          "stated, answered only | stated - raw, points | empty | empty by finish reason |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c, m in gpqa().items():
        s = json.loads((ROOT / "runs" / m["id"] / "summary.json").read_text())
        lo, hi = s["accuracy_flexible_ci95"]
        print(f"| {name(m)} | {100 * s['accuracy_flexible']:.1f}% | {100 * lo:.1f}-{100 * hi:.1f}% | {100 * s['accuracy_flexible_answered']:.1f}% | "
              f"{100 * s['accuracy_stated']:.1f}% | {100 * s['accuracy_stated_answered']:.1f}% | "
              f"{100 * (s['accuracy_stated'] - s['accuracy_flexible']):+.1f} | {s['empty']} | {s['empty_by_finish_reason'] or '-'} |")


def paired(a, b, n_boot=10000, seed=0):
    """Question-paired comparison of two GPQA runs (pass 1 each)."""
    A = {r["doc_id"]: r for r in a["rows"]}; B = {r["doc_id"]: r for r in b["rows"]}
    assert A.keys() == B.keys() and all(A[d]["prompt_hash"] == B[d]["prompt_hash"] for d in A)
    ids = sorted(A)
    xa = sum(A[d]["correct_flexible"] and not B[d]["correct_flexible"] for d in ids)
    xb = sum(B[d]["correct_flexible"] and not A[d]["correct_flexible"] for d in ids)
    ea = sum(A[d]["empty"] and not B[d]["empty"] for d in ids); eb = sum(B[d]["empty"] and not A[d]["empty"] for d in ids)
    diff = [B[d]["correct_flexible"] - A[d]["correct_flexible"] for d in ids]
    rng = random.Random(seed)
    boots = sorted(sum(diff[rng.randrange(len(ids))] for _ in ids) / len(ids) for _ in range(n_boot))
    return dict(only_a=xa, only_b=xb, p_acc=sign_test(xa, xb), diff=100 * sum(diff) / len(ids),
                lo=100 * boots[int(0.025 * n_boot)], hi=100 * boots[int(0.975 * n_boot) - 1],
                empty_a=sum(A[d]["empty"] for d in ids), empty_b=sum(B[d]["empty"] for d in ids),
                empty_only_a=ea, empty_only_b=eb, p_empty=sign_test(ea, eb))


def gpqa_pairs(args):
    g = gpqa()
    todo = [(args[0], args[1], "")] if len(args) == 2 else PAIRS
    print("| A | B | what differs | accuracy A, B | only A right / only B right (McNemar p) | B - A, points (95% interval) | "
          "empty A, B | only A empty / only B empty (McNemar p) |")
    print("|---|---|---|---|---|---|---|---|")
    for a, b, what in todo:
        r = paired(g[a], g[b]); acc = lambda m: 100 * sum(x["correct_flexible"] for x in m["rows"]) / len(m["rows"])
        print(f"| {config_label(g[a]['cfg'])} | {config_label(g[b]['cfg'])} | {what} | {acc(g[a]):.1f}%, {acc(g[b]):.1f}% | "
              f"{r['only_a']} / {r['only_b']} ({r['p_acc']:.2f}) | {r['diff']:+.1f} ({r['lo']:+.1f} to {r['hi']:+.1f}) | "
              f"{r['empty_a']}, {r['empty_b']} | {r['empty_only_a']} / {r['empty_only_b']} ({r['p_empty']:.2f}) |")


# ---------------------------------------------------------------- hard-prompt-screen v2 (component screen)

def v2_runs():
    out = {}
    for m in runs("hard-prompt-screen/v2"):
        arm = m["notes"].split("arm ", 1)[1].split(":", 1)[0]
        out[arm] = m
    return out


def logistic(X, y, iters=50):
    """Maximum-likelihood logistic regression by Newton-Raphson (no regularisation). Returns coefficients and SEs."""
    k = len(X[0]); b = [0.0] * k
    for _ in range(iters):
        p = [1 / (1 + math.exp(-sum(bi * xi for bi, xi in zip(b, x)))) for x in X]
        g = [sum((yi - pi) * x[j] for x, yi, pi in zip(X, y, p)) for j in range(k)]
        H = [[sum(pi * (1 - pi) * x[i] * x[j] for x, pi in zip(X, p)) for j in range(k)] for i in range(k)]
        inv = invert(H); step = [sum(inv[i][j] * g[j] for j in range(k)) for i in range(k)]
        b = [bi + si for bi, si in zip(b, step)]
        if max(abs(s) for s in step) < 1e-10: break
    return b, [math.sqrt(inv[i][i]) for i in range(k)]


def invert(M):
    n = len(M); A = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(M)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(A[r][c])); A[c], A[piv] = A[piv], A[c]
        f = A[c][c]; A[c] = [v / f for v in A[c]]
        for r in range(n):
            if r != c:
                g = A[r][c]; A[r] = [v - g * w for v, w in zip(A[r], A[c])]
    return [row[n:] for row in A]


def screen_v2():
    v = v2_runs()
    print("## Arms (one question each, 12 repeats, seeds 5001-5012 unless marked)")
    print("| arm | configuration | question | failed / 12 | 95% Wilson | loops | exhaustions | seeds |"); print("|---|---|---|---|---|---|---|---|")
    for arm, m in v.items():
        rs = m["rows"]; k = sum(map(failed, rs)); lo, hi = wilson(k, len(rs)); seeds = sorted({r["seed"] for r in rs})
        ci = "- (control, not a rate)" if len(seeds) == 1 else f"{100 * lo:.0f}-{100 * hi:.0f}%"
        print(f"| {arm} | {config_label(m['cfg'])} | {rs[0]['doc_id']} | {k} | {ci} | "
              f"{sum(r['cls'] == 'loop' for r in rs)} | {sum(r['cls'] == 'exhaust' for r in rs)} | "
              f"{seeds[0]}-{seeds[-1] if len(seeds) > 1 else seeds[0]}{' (fixed-seed control)' if len(seeds) == 1 else ''} |")
    if all(a in v for a in ARMS88):
        print("\n## Part 1, doc 88: each switch on (two arms, 24 requests) vs off (24), preregistered rule: on-arms >= 6 fewer failures and Fisher p < 0.10")
        print("| switch | failed, on | failed, off | difference on - off (95% Newcombe) | Fisher p | screening rule |"); print("|---|---|---|---|---|---|")
        k = {a: sum(map(failed, v[a]["rows"])) for a in ARMS88}
        for sw, on in SWITCHES.items():
            off = [a for a in ARMS88 if a not in on]
            kon, koff = sum(k[a] for a in on), sum(k[a] for a in off)
            d, lo, hi = newcombe(kon, 24, koff, 24); p = fisher(kon, 24 - kon, koff, 24 - koff)
            verdict = "CANDIDATE" if koff - kon >= 6 and p < 0.10 else "no candidate at this size"
            print(f"| {sw} | {kon}/24 | {koff}/24 | {100 * d:+.1f} points ({100 * lo:+.1f} to {100 * hi:+.1f}) | {p:.2f} | {verdict} |")
        X, y = [], []
        for a in ARMS88:
            for r in v[a]["rows"]:
                X.append([1.0] + [float(a in on) for on in SWITCHES.values()]); y.append(float(failed(r)))
        b, se = logistic(X, y)
        print("\nLogistic fit, three main effects (reported, no rule): " + "; ".join(
            f"{n} {bi:+.2f} (SE {si:.2f})" for n, bi, si in zip(["intercept"] + list(SWITCHES), b, se)))
        print("\nLoop onset (reasoning characters when the early stop fired) and finished lengths (completion tokens), descriptive:")
        for a in ARMS88:
            rs = v[a]["rows"]
            print(f"  {a}: onset {sorted(r['reasoning_chars'] for r in rs if r['stopped_early'])}; finished median "
                  f"{st.median([r['completion_tokens'] for r in rs if r['cls'] == 'ok']):,.0f} tokens")
    if all(a in v for a in ARMS79):
        print("\n## Part 2, doc 79: B79 vs V79, preregistered rule: V79 >= 5 fewer failures and Fisher p < 0.10 -> v0.7.0 image candidate; "
              ">= 5 more -> draft depth implicated; otherwise unresolved")
        kb, kv = (sum(map(failed, v[a]["rows"])) for a in ARMS79)
        p = fisher(kb, 12 - kb, kv, 12 - kv); d, lo, hi = newcombe(kv, 12, kb, 12)
        verdict = "CANDIDATE (v0.7.0 image)" if kb - kv >= 5 and p < 0.10 else ("draft depth implicated" if kv - kb >= 5 and p < 0.10 else "unresolved")
        print(f"B79 {kb}/12, V79 {kv}/12; V79 - B79 {100 * d:+.1f} points ({100 * lo:+.1f} to {100 * hi:+.1f}); Fisher p = {p:.2f}; {verdict}")
    if "S1234" in v and "B" in v:
        kb, ks = sum(map(failed, v["B"]["rows"])), sum(map(failed, v["S1234"]["rows"]))
        reading = ("the fixed seed explains most of the difference" if ks >= 9 else
                   "the fixed seed does not explain it" if ks <= 5 else "unresolved")
        print(f"\n## Fixed-seed control (Amendment 1): B (seeds 5001-5012) {kb}/12 vs S1234 (seed 1234 on every repeat) {ks}/12; "
              f"Fisher p = {fisher(kb, 12 - kb, ks, 12 - ks):.4f}; amendment reading (>= 9 failed: fixed seed explains most; <= 5: it does not): {reading}")


def chars_per_token(rows):
    """Median characters per completion token of the finished requests in one run (reasoning + answer characters)."""
    return st.median((r["reasoning_chars"] + r["content_chars"]) / r["completion_tokens"] for r in rows if r["cls"] == "ok")


def anatomy():
    """Failure anatomy from hard-prompt-screen/v2 rows only (distinct seeds; the fixed-seed control listed separately)."""
    v = v2_runs(); ctrl = {a: m for a, m in v.items() if len({r["seed"] for r in m["rows"]}) == 1}
    print("## Loop and exhaustion by question (distinct-seed arms)")
    for doc in sorted({m["rows"][0]["doc_id"] for a, m in v.items() if a not in ctrl}):
        rs = [r for a, m in v.items() if a not in ctrl for r in m["rows"] if r["doc_id"] == doc]
        print(f"doc {doc}: {len(rs)} requests, finished {sum(r['cls'] == 'ok' for r in rs)}, loop {sum(r['cls'] == 'loop' for r in rs)} "
              f"({sum(r['stopped_early'] for r in rs)} stopped early), exhaust {sum(r['cls'] == 'exhaust' for r in rs)}")
    print("\n## Where the early stop fired (reasoning characters; tokens estimated with the question's median characters per completion "
          "token over the finished requests of the distinct-seed arms)")
    cpq = {d: chars_per_token([r for a, m in v.items() if a not in ctrl for r in m["rows"] if r["doc_id"] == d])
           for d in {m["rows"][0]["doc_id"] for m in v.values()}}
    for a, m in v.items():
        cpt = cpq[m["rows"][0]["doc_id"]]; pos = sorted(r["reasoning_chars"] for r in m["rows"] if r["stopped_early"])
        print(f"{a}{' (fixed-seed control)' if a in ctrl else ''}: {cpt:.2f} chars/token; stops at "
              + (", ".join(f"{c:,} chars ~{c / cpt / 1000:.0f}k tokens" for c in pos) or "none"))
        caps = [r for r in m["rows"] if r["cls"] == "loop" and not r["stopped_early"]]
        if caps: print(f"   loops that ran to the 327,680-token cap without an early stop: {len(caps)} (tail ratio {[r['tail_zlib_ratio'] for r in caps]})")
    print("\n## Early-stop detector margin (lowest periodic check; the detector stops after three consecutive checks below 0.10)")
    for cls in ("ok", "exhaust"):
        x = [r["min_zlib_check"] for m in v.values() for r in m["rows"] if r["cls"] == cls and r.get("min_zlib_check") is not None]
        print(f"{cls}: {len(x)} requests, none stopped early; lowest check {min(x):.4f}")


def fisher_power(p1, p2, n1, n2, alpha=0.05):
    """Exact power of the two-sided Fisher test (p < alpha) for true rates p1, p2 with n1, n2 independent draws."""
    b = lambda n, p: [math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]
    b1, b2 = b(n1, p1), b(n2, p2)
    return sum(b1[x] * b2[y] for x in range(n1 + 1) for y in range(n2 + 1) if fisher(x, n1 - x, y, n2 - y) < alpha)


def mdd(p1, n1, n2, power=0.8, alpha=0.05):
    """Smallest true difference from p1 (down, up) that the Fisher test detects with the given power; None if none in [0, 1]."""
    grid = [i / 100 for i in range(101)]
    down = [p1 - q for q in grid if q < p1 and fisher_power(p1, q, n1, n2, alpha) >= power]
    up = [q - p1 for q in grid if q > p1 and fisher_power(p1, q, n1, n2, alpha) >= power]
    return (min(down) if down else None), (min(up) if up else None)


def power_table():
    """Minimum detectable differences for the component screen's comparisons (two-sided Fisher p < 0.05, 80% power)."""
    v = v2_runs(); k = {a: sum(map(failed, m["rows"])) for a, m in v.items()}
    base88 = sum(k[a] for a in ARMS88) / (12 * len(ARMS88))
    cases = [("one arm vs another, question 88 (12 vs 12)", k["B"] / 12, 12, 12),
             ("switch on vs off, question 88 (24 vs 24)", base88, 24, 24),
             ("B79 vs V79, question 79 (12 vs 12)", k["B79"] / 12, 12, 12)]
    print("| comparison | baseline failure rate | smallest detectable drop | smallest detectable rise |"); print("|---|---|---|---|")
    for name_, p0, n1, n2 in cases:
        d, u = mdd(p0, n1, n2)
        f = lambda x: "not detectable at any size" if x is None else f"{100 * x:.0f} points (to {100 * (p0 - x if x is d else p0 + x):.0f}%)"
        print(f"| {name_} | {100 * p0:.0f}% | {f(d)} | {f(u)} |")


def decode_prefill():
    """decode-prefill-consistency/v1: mean KL by region for every run, on prompts 0-5 (shared by all runs) and on all prompts;
    and, for two runs of one configuration with the same seeds, where their decodes first differ (single-request nondeterminism)."""
    ms = list(runs("decode-prefill-consistency/"))
    def kl(rows, reg, docs=None):
        v = [r["kl_top20"] for r in rows if r["region"] == reg and r["kl_top20"] is not None and (docs is None or r["doc_id"] in docs)]
        return sum(v) / len(v)
    print("| run | configuration | prompts | KL i < 2,044 (prompts 0-5) | KL i >= 2,048 (prompts 0-5) | KL i < 2,044 (all) | KL i >= 2,048 (all) |")
    print("|---|---|---|---|---|---|---|")
    for m in ms:
        d = sorted({r["doc_id"] for r in m["rows"]})
        print(f"| {m['id']} | {config_label(m['cfg'])} | {d[0]}-{d[-1]} | {kl(m['rows'], 'lt2044', range(6)):.4f} | {kl(m['rows'], 'ge2048', range(6)):.4f} | "
              f"{kl(m['rows'], 'lt2044'):.4f} | {kl(m['rows'], 'ge2048'):.4f} |")
    by = {}
    for m in ms: by.setdefault(m["config"], []).append(m)
    for c, pair in by.items():
        if len(pair) != 2: continue
        a, b = ({(r["doc_id"], r["i"]): r for r in m["rows"]} for m in pair)
        first = {}
        for doc in sorted({k[0] for k in a}):
            pos = sorted(i for (d, i) in a if d == doc)
            diff = [i for i in pos if (doc, i) in b and (a[(doc, i)]["abs_dlp"], a[(doc, i)]["kl_top20"]) != (b[(doc, i)]["abs_dlp"], b[(doc, i)]["kl_top20"])]
            first[doc] = (diff[0] - pos[0]) if diff else None
        print(f"\nSame configuration twice ({config_label(pair[0]['cfg'])}; {pair[0]['id']} vs {pair[1]['id']}), concurrency 1, same seeds: "
              f"generated tokens before the first position whose decode values differ, per prompt: {first}")
        for reg in ("lt2044", "ge2048"):
            per = [(kl(pair[0]["rows"], reg, [d]), kl(pair[1]["rows"], reg, [d])) for d in sorted(first)]
            print(f"  {reg}: per-prompt KL ratio between the two runs (max / min): {max(max(x, y) / min(x, y) for x, y in per):.1f}; "
                  f"run means {kl(pair[0]['rows'], reg):.4f} vs {kl(pair[1]['rows'], reg):.4f}")


def screen_single():
    print("Single draws (repeat 1 of each question); outcomes only. Every repeat of these screens sent seed 1234, so they are not rates.")
    docs = [13, 79, 88, 121, 127]
    print("| configuration | protocol, date | " + " | ".join(f"q{d}" for d in docs) + " |"); print("|---|---|" + "---|" * len(docs))
    for m in runs("hard-prompt-screen/v"):
        if m["protocol"].endswith("v2"): continue
        o = {r["doc_id"]: r["cls"] for r in m["rows"]}
        print(f"| {name(m)} | {m['protocol'].split('/')[1]}, {m['date']} | " + " | ".join(o.get(d, "-") for d in docs) + " |")


def screen_speed():
    print("| run | configuration | finished | median completion tok/s of finished requests |"); print("|---|---|---|---|")
    for m in runs("hard-prompt-screen/v2"):
        v = [r["completion_tokens"] / r["secs"] for r in m["rows"] if r["finish_reason"] == "stop" and r.get("completion_tokens")]
        print(f"| {m['id']} | {config_label(m['cfg'])} | {len(v)} | " + (f"{st.median(v):.1f}" if v else "-") + " |")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gpqa-table"
    {"gpqa-table": gpqa_table, "gpqa-pairs": lambda: gpqa_pairs(sys.argv[2:]), "screen-v2": screen_v2,
     "screen-single": screen_single, "decode-prefill": decode_prefill, "screen-power": power_table, "screen-speed": screen_speed, "screen-anatomy": anatomy}[cmd]()
