#!/usr/bin/env python3
"""Same-seed reply identity across existing GPQA records (CPU, read only; counts and hashes only, no reply text written).
For each pair of passes that sent the same request seed, count (question-matched): identical non-empty replies (sha256 of the
reply text equal), both empty, empty in exactly one, and identical empty-status. Reply text is hashed in memory only.
Also (descriptive, question-dominated): Pearson correlation of log reply length over questions answered in both, and raw-correctness
agreement. Pairs: same image + same seed + fresh server (the situation of variant R) and different builds at the same seed."""
import glob, hashlib, json, math, sys


def pearson(u, v):
    mu, mv = sum(u) / len(u), sum(v) / len(v)
    su = math.sqrt(sum((a - mu) ** 2 for a in u)); sv = math.sqrt(sum((b - mv) ** 2 for b in v))
    return round(sum((a - mu) * (b - mv) for a, b in zip(u, v)) / (su * sv), 3) if su and sv else None
G, A = "<gpqa-runs>/", "<data-runs>/archive/retired-2026-10/fixed-seed-gpqa/"
P = "glm53-flash-exl3-k3.25@tpurtell"
def load(d):
    f = glob.glob(d + "/**/samples_*.jsonl", recursive=True); assert len(f) == 1, d
    out = {}
    for line in open(f[0]):
        r = json.loads(line)
        if r["filter"] != "flexible-extract": continue
        t = str(r["resps"][0][0]) if r["resps"] else ""
        out[r["doc_id"]] = (hashlib.sha256(t.encode()).hexdigest(), not t.strip(), len(t), bool(r["exact_match"]))
    assert sorted(out) == list(range(198)), d
    return out
PAIRS = [  # (label, kind, dir a, dir b, seed)
 ("0.7.0 pass 1 (2026-09-29 session) vs 0.7.0 fixed-seed pass 2 (2026-09-30 fresh session)", "same image, same seed 1234, different server session", G + P + "__pass1", A + P + "__pass2", 1234),
 ("0.7.0 pass 1 (2026-09-29 session) vs 0.7.0 fixed-seed pass 3 (2026-09-30 session)", "same image, same seed 1234, different server session", G + P + "__pass1", A + P + "__pass3", 1234),
 ("0.7.0 fixed-seed pass 2 vs pass 3 (both 2026-09-30, one session)", "same image, same seed 1234, same server session", A + P + "__pass2", A + P + "__pass3", 1234),
 ("0.8.0 K3.25 pass 1 vs fixed-seed pass 2", "same image, same seed 1234", G + P + "+v0.8.0__pass1", A + P + "+v0.8.0__pass2", 1234),
 ("0.8.0 K3.25 pass 1 vs fixed-seed pass 3", "same image, same seed 1234", G + P + "+v0.8.0__pass1", A + P + "+v0.8.0__pass3", 1234),
 ("K4 0.9.1 pass 1 conc 8 vs K4 0.9.1 pass 1 conc 4 (+psc4)", "same image, same seed 1234, different concurrency and session", G + "glm53-flash-exl3-k4@tpurtell+v0.9.1__pass1", G + "glm53-flash-exl3-k4@tpurtell+v0.9.1+psc4__pass1", 1234),
 ("B0 vs B1 at seed 1236", "near-identical builds (D7 file only), same seed", G + P + "+v0.9.1+bundle-d1234__pass3", G + P + "+v0.9.1+bundle-d1234+d7__pass3", 1236),
 ("stock 0.9.1 vs column-511 arm at seed 1234", "different builds, same seed", G + P + "+v0.9.1+ps__pass1", G + P + "+v0.9.1+col511fix__pass1", 1234),
 ("stock 0.7.0 vs stock 0.9.1 at seed 1235", "different engines, same seed", G + P + "+ps__pass2", G + P + "+v0.9.1+ps__pass2", 1235),
 ("stock 0.7.0 vs stock 0.9.1 at seed 1236", "different engines, same seed", G + P + "+ps__pass3", G + P + "+v0.9.1+ps__pass3", 1236),
 ("stock 0.7.0 pass 2 (1235) vs pass 3 (1236), same session", "same image, different seeds (control)", G + P + "+ps__pass2", G + P + "+ps__pass3", None),
 ("stock 0.7.0 pass 1 (1234) vs pass 2 (1235)", "same image, different seeds and sessions (control)", G + P + "__pass1", G + P + "+ps__pass2", None),
 ("stock 0.7.0 pass 1 (1234) vs pass 3 (1236)", "same image, different seeds and sessions (control)", G + P + "__pass1", G + P + "+ps__pass3", None),
 ("stock 0.9.1 pass 1 (1234) vs pass 2 (1235)", "same image, different seeds and sessions (control)", G + P + "+v0.9.1+ps__pass1", G + P + "+v0.9.1+ps__pass2", None),
 ("stock 0.9.1 pass 2 (1235) vs pass 3 (1236)", "same image, different seeds, same session (control)", G + P + "+v0.9.1+ps__pass2", G + P + "+v0.9.1+ps__pass3", None),
 ("column-511 arm pass 2 (1235) vs pass 3 (1236)", "same image, different seeds, same session (control)", G + P + "+v0.9.1+col511fix__pass2", G + P + "+v0.9.1+col511fix__pass3", None),
 ("B0 pass 1 (1234) vs B0 pass 2 (1235)", "same image, different seeds and sessions (control)", G + P + "+v0.9.1+bundle-d1234__pass1", G + P + "+v0.9.1+bundle-d1234__pass2", None),
 ("stock 0.7.0 (1235) vs stock 0.9.1 (1236)", "different engines, different seeds (control)", G + P + "+ps__pass2", G + P + "+v0.9.1+ps__pass3", None),
 ("stock 0.7.0 (1236) vs stock 0.9.1 (1235)", "different engines, different seeds (control)", G + P + "+ps__pass3", G + P + "+v0.9.1+ps__pass2", None),
 ("stock 0.9.1 (1234) vs column-511 arm (1235)", "different builds, different seeds (control)", G + P + "+v0.9.1+ps__pass1", G + P + "+v0.9.1+col511fix__pass2", None),
 ("stock 0.9.1 (1235) vs column-511 arm (1234)", "different builds, different seeds (control)", G + P + "+v0.9.1+ps__pass2", G + P + "+v0.9.1+col511fix__pass1", None),
 ("stock 0.9.1 seed 1235 vs column-511 arm seed 1235", "different builds, same seed", G + P + "+v0.9.1+ps__pass2", G + P + "+v0.9.1+col511fix__pass2", 1235),
 ("stock 0.9.1 seed 1236 vs column-511 arm seed 1236", "different builds, same seed", G + P + "+v0.9.1+ps__pass3", G + P + "+v0.9.1+col511fix__pass3", 1236),
 ("stock 0.7.0 seed 1234 vs stock 0.9.1 seed 1234", "different engines, same seed", G + P + "__pass1", G + P + "+v0.9.1+ps__pass1", 1234),
]
res = []
for lab, kind, a, b, s in PAIRS:
    try:
        x, y = load(a), load(b)
    except (AssertionError, OSError) as e:
        res.append({"pair": lab, "error": str(e)}); continue
    ident_ne = sum(1 for q in range(198) if not x[q][1] and not y[q][1] and x[q][0] == y[q][0])
    both_e = sum(1 for q in range(198) if x[q][1] and y[q][1])
    one_e = sum(1 for q in range(198) if x[q][1] != y[q][1])
    res.append({"pair": lab, "kind": kind, "seed": s, "identical_nonempty": ident_ne, "both_empty": both_e,
                "empty_in_exactly_one": one_e, "empties_a": sum(v[1] for v in x.values()), "empties_b": sum(v[1] for v in y.values()),
                "answered_in_both": sum(1 for q in range(198) if not x[q][1] and not y[q][1]),
                "log_reply_chars_pearson_answered_in_both": pearson([math.log(x[q][2]) for q in range(198) if not x[q][1] and not y[q][1]],
                                                                    [math.log(y[q][2]) for q in range(198) if not x[q][1] and not y[q][1]]),
                "raw_correct_agreement": sum(x[q][3] == y[q][3] for q in range(198))})
def summ(sel):
    v = [r["log_reply_chars_pearson_answered_in_both"] for r in res if "error" not in r and sel(r)]
    return {"pairs": len(v), "min": min(v), "median": sorted(v)[len(v) // 2], "max": max(v)} if v else None
summary = {"same_seed_pairs": summ(lambda r: r["seed"] is not None), "different_seed_pairs": summ(lambda r: r["seed"] is None),
           "identical_nonempty_total": sum(r.get("identical_nonempty", 0) for r in res)}
json.dump({"pairs": res, "summary_log_length_correlation": summary}, open(sys.argv[1], "w"), indent=1)
print(json.dumps(summary))
for r in res:
    print(json.dumps(r))
