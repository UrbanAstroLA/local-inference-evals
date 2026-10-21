#!/usr/bin/env python3
"""B+ prefix selection (PROPOSAL.md section 2.1). Host, standard library, read only. Zero GPU.

The rule is written in PROPOSAL.md before any continuation exists (none has been run). It uses only the discovery
outputs (ids, lengths, R50 at 100k, the long-reply panel counts); no reply text is read here. Writes ids, counts and
metrics only.

Rule
  pool       rep070-F replies with T >= 100,000 completion tokens (the discovery's at-risk set, 29 replies)
  E1         re-tokenized reasoning longer than the cut: N_reasoning > CUT (the cut lies strictly inside the reasoning)
  E2         the discovery's R50 at g = 100,000 (replies_all.json R50_at["100000"]) is defined and < 0.05
  strata     original outcome: capped (finish_reason length at 327,680) / finished (finish_reason stop)
  per stratum, one reply per question, questions never shared between strata (capped stratum chosen first)
  question rank   long-reply panel count over all 15 logged passes (discovery T7 'all', summary.json panel), descending;
                  ties by sha256(SALT + 'q' + doc), ascending
  reply in question   sha256(SALT + reply id), ascending (outcome-blind)
  take       the first 6 questions of each stratum
usage: select_prefixes.py OUT_JSON
"""
import hashlib
import json
import sys

DISC = "<runs>/long-reply/discovery/out"
SALT = "bplus-2026-10-16:"
CUT = 100000
R50_MAX = 0.05
PER_STRATUM = 6
INPUTS = {f"{DISC}/replies_all.json": "e2cb8516650a44771d6dbb319be603b4defc6d78df199ec62305ce0c76d7296f",
          f"{DISC}/summary.json": "021bcb013c5691f410868886213c4d72bfb0c9d34847814cf7bdb9a334e49308"}


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def main():
    for p, s in INPUTS.items():
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if got != s:
            sys.exit(f"input changed: {p} {got}")
    R = json.load(open(f"{DISC}/replies_all.json"))
    panel = {row["doc"]: row["all"] for row in json.load(open(f"{DISC}/summary.json"))["panel"]}
    pool = [r for r in R if r["T"] >= 100000]
    assert len(pool) == 29, len(pool)
    rows = []
    for r in sorted(pool, key=lambda r: (r["doc"], r["pass"])):
        r50 = r["R50_at"].get(str(CUT))
        e1 = r["N_reasoning"] > CUT
        e2 = r50 is not None and r50 < R50_MAX
        rows.append({"id": r["id"], "doc": r["doc"], "pass": r["pass"], "seed": r["seed"], "T": r["T"],
                     "stratum": "capped" if r["finish_reason"] == "length" else "finished",
                     "N_reasoning": r["N_reasoning"], "prompt_tokens": r["prompt_tokens"], "drift": r["drift"],
                     "R50_at_100k": r50, "E1_reasoning_longer_than_cut": e1, "E2_R50_lt_0.05": e2,
                     "eligible": e1 and e2, "panel_all": panel.get(r["doc"], 0),
                     "reply_hash": h(SALT + r["id"]),
                     "orig_onset_g": r["onset"],
                     "orig_reasoning_end_minus_cut": r["N_reasoning"] - CUT})
    chosen, used_q, reasons = [], set(), {}
    for stratum in ("capped", "finished"):
        el = [x for x in rows if x["eligible"] and x["stratum"] == stratum and x["doc"] not in used_q]
        qs = sorted({x["doc"] for x in el}, key=lambda d: (-panel.get(d, 0), h(f"{SALT}q{d}")))
        for rank, d in enumerate(qs[:PER_STRATUM], 1):
            cand = sorted((x for x in el if x["doc"] == d), key=lambda x: x["reply_hash"])
            pick = cand[0]
            chosen.append(pick)
            used_q.add(d)
            reasons[pick["id"]] = (f"{stratum} stratum, question rank {rank} of {len(qs)} (panel long count {panel.get(d, 0)}); "
                                   f"{len(cand)} eligible repl{'y' if len(cand) == 1 else 'ies'} in the question "
                                   f"({', '.join(x['id'] for x in cand)}), lowest reply hash")
        for d in qs[PER_STRATUM:]:
            for x in el:
                if x["doc"] == d:
                    reasons.setdefault(x["id"], f"{stratum} stratum: eligible, question rank beyond {PER_STRATUM}")
    for x in rows:
        if not x["eligible"]:
            why = []
            if not x["E1_reasoning_longer_than_cut"]:
                why.append(f"reasoning ends at {x['N_reasoning']} <= {CUT}")
            if x["R50_at_100k"] is None:
                why.append("R50 at 100k undefined (reasoning ends before the 4,096-token window fits)")
            elif x["R50_at_100k"] >= R50_MAX:
                why.append(f"R50 at 100k = {x['R50_at_100k']} >= {R50_MAX}")
            reasons[x["id"]] = "not eligible: " + "; ".join(why)
        elif x["id"] not in reasons:
            reasons[x["id"]] = "eligible, not chosen: question already used or a sibling reply had the lower hash"
    for x in rows:
        x["selected"] = any(c["id"] == x["id"] for c in chosen)
        x["reason"] = reasons[x["id"]]
    sel = [{"prefix_id": f"P{i:02d}", **{k: c[k] for k in ("id", "doc", "pass", "seed", "stratum", "T", "N_reasoning",
                                                            "prompt_tokens", "drift", "R50_at_100k", "panel_all",
                                                            "orig_onset_g", "orig_reasoning_end_minus_cut")},
            "reason": reasons[c["id"]]} for i, c in enumerate(chosen, 1)]
    out = {"what": "B+ prefix selection (ids, counts and metrics only)", "salt": SALT, "cut": CUT, "r50_max": R50_MAX,
           "inputs_sha256": INPUTS, "selected": sel, "pool": rows,
           "summary": {"pool": len(rows), "eligible": sum(x["eligible"] for x in rows),
                       "selected": len(sel), "by_stratum": {s: sum(x["stratum"] == s for x in sel) for s in ("capped", "finished")},
                       "distinct_questions": len({x["doc"] for x in sel})}}
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for s in sel:
        print(s["prefix_id"], s["id"], s["stratum"], "R50@100k", s["R50_at_100k"], "|", s["reason"])
    print(json.dumps(out["summary"]))
    return 0 if len(sel) == 2 * PER_STRATUM and out["summary"]["distinct_questions"] == len(sel) else 1


if __name__ == "__main__":
    sys.exit(main())
