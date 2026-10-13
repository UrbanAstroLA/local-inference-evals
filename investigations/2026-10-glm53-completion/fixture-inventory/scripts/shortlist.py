#!/usr/bin/env python3
"""Add a shortlist of fixture candidates (by window class) to inventory.json. Positions only."""
import json, sys, collections
p = sys.argv[1]
inv = json.load(open(p))
R = [r for r in inv["text_rows"] if r["state"] != "error" and (r["weights"] or "").startswith("K3.25")]
def fam(r):
    e = r["engine"]
    return "0.7.0" if "v0.7.0" in e else "0.8.0" if "v0.8.0" in e else "0.9.0" if "v0.9.0" in e else "0.9.1"
def distinct(r):
    return r["seeds"] not in ("1234 all", "1234 all (script)")
def nonrep_at(r, cp):
    v = r["checkpoints"].get(str(cp))
    return bool(v) and not v["repeating"]
def brief(r, extra=None):
    d = dict(id=r["id"], arm=r["arm"], engine_family=fam(r), doc=r["doc"], seed=r["seed"], seeds_rule=r["seeds"],
             state=r["state"], n_reasoning_tokens=r["n_reasoning_tokens"], prompt_tokens=r["prompt_tokens_rendered"],
             natural_close_pos=r["natural_close_pos"], last_commit_tok=r["last_commit_tok"],
             close_minus_last_commit=r["close_minus_last_commit"], loop_onset_tok=r["loop_onset_tok"],
             end_zlib30k=(r["end_metrics"] or {}).get("zlib30k"), end_rep200=(r["end_metrics"] or {}).get("rep200"),
             retok_drift=r["retok_drift"], sha256_reasoning_ids=r["sha256_reasoning_ids"])
    if extra: d.update(extra)
    return d
sl = {}
# near-close, long: natural close at 64k-128k, not repeating at the close
nc = [r for r in R if r["state"] == "natural_close" and 64000 <= r["n_reasoning_tokens"] < 128000 and not r["end_metrics"]["repeating"]]
nc.sort(key=lambda r: (not distinct(r), fam(r), r["doc"], r["n_reasoning_tokens"]))
sl["near_close_64k_128k"] = [brief(r) for r in nc]
nc2 = [r for r in R if r["state"] == "natural_close" and r["n_reasoning_tokens"] >= 128000 and not r["end_metrics"]["repeating"]]
nc2.sort(key=lambda r: (not distinct(r), fam(r), r["doc"], r["n_reasoning_tokens"]))
sl["near_close_ge128k"] = [brief(r) for r in nc2]
# censored exhaustions, non-repeating at 64k, 96k and 128k (active reasoning across the band)
ex = [r for r in R if r["state"] in ("exhaustion", "exhaustion_cut120k") and all(nonrep_at(r, c) for c in (64000, 96000) if str(c) in r["checkpoints"])
      and r["n_reasoning_tokens"] >= 96000]
ex.sort(key=lambda r: (not distinct(r), fam(r), r["doc"]))
sl["exhaustion_active_64k_128k"] = [brief(r, dict(nonrep_commit_events_56k_136k=r["strong_events_56k_136k_nonrepeating"],
                                                 first_nonrep_commits_in_band=[e[0] for e in r["strong_events_56k_136k"] if e[4] is False][:5])) for r in ex]
# pre-onset windows of loops that start repeating only after 128k
lp = [r for r in R if r["state"] == "loop" and r["loop_onset_tok"] and r["loop_onset_tok"] > 136000 and nonrep_at(r, 64000) and nonrep_at(r, 128000)]
lp.sort(key=lambda r: (not distinct(r), fam(r), r["doc"]))
sl["loop_preonset_64k_128k"] = [brief(r) for r in lp]
# short controls with text (<16k natural close), distinct seeds first
sc = [r for r in R if r["state"] == "natural_close" and r["n_reasoning_tokens"] < 16000]
sc.sort(key=lambda r: (not distinct(r), fam(r), r["doc"]))
sl["short_control_text"] = [brief(r) for r in sc[:40]]
sl["short_control_exact_ids"] = [dict(id=t["id"], path=t["path"], engine=t["engine"], doc=t["doc"], seed=t["seed"],
                                     natural_close_pos=t["natural_close_pos"], first_terminator_pos=t["first_terminator_pos"],
                                     tokens_close_to_terminator=t["tokens_close_to_terminator"], sha256_active_ids=t["sha256_active_ids"])
                                for t in inv["exact_traces"]]
inv["shortlist"] = sl
json.dump(inv, open(p, "w"), indent=1)
for k, v in sl.items():
    c = collections.Counter((x.get("engine_family") or x.get("engine")) for x in v)
    print(k, len(v), dict(c), "docs", sorted(collections.Counter(x["doc"] for x in v).items()))
