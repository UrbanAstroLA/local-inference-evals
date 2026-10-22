#!/usr/bin/env python3
"""Prefix selection for a possible later cross-continuation (PROPOSAL.md section 7). Host, stdlib, read only, zero GPU.
NOTHING IS CONTINUED IN THIS PACKAGE. Writes ids, counts and metrics only.

Rule = B+'s (bplus/scripts/select_prefixes.py, sha256 b998bf67..., PROPOSAL 22s), applied to the d091 replies:
  pool       replies with T >= 100,000 completion tokens (the discovery's at-risk definition)
  E1         re-tokenized reasoning longer than the cut: N_reasoning > 100,000
  E2         trajectory.py's R50 at g = 100,000 is defined and < 0.05
  strata     original outcome: capped (finish_reason length) / finished (stop); capped stratum chosen first
  per stratum one reply per question (a single pass gives one reply per question anyway), questions never shared
  question rank   discovery's long-reply panel count over its 15 logged passes (summary.json 'all'), descending;
                  ties by sha256(SALT + 'q' + doc), ascending
  reply in question   sha256(SALT + reply id), ascending (outcome-blind)
  take       at most 6 questions per stratum (B+'s PER_STRATUM); with one pass fewer are expected, and all eligible are taken
The selection function below is select_prefixes.main()'s body with its inputs as parameters; B+'s file cannot run on
d091 unchanged because it asserts B+'s pool size (29) and pins B+'s input hashes. Proof of identity: `--reproduce-bplus`
applies this function to B+'s inputs with B+'s salt and compares the output with bplus/out/selection.json byte for byte.

usage: prefix_select_d091.py OUT_DIR                 (d091: OUT_DIR/replies_all.json in, OUT_DIR/selection.json out)
       prefix_select_d091.py --reproduce-bplus TEST_OUT.json
"""
import hashlib, json, sys

DISC = "<runs>/long-reply/discovery/out"
BPLUS_SEL = "<runs>/long-reply/bplus/out/selection.json"
SALT_D091 = "d091-2026-10-21:"
SALT_BPLUS = "bplus-2026-10-16:"
CUT = 100000
R50_MAX = 0.05
PER_STRATUM = 6
SUMMARY_SHA = "021bcb013c5691f410868886213c4d72bfb0c9d34847814cf7bdb9a334e49308"


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def select(R, panel, SALT, inputs_sha, what, expect_pool=None):
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()
    pool = [r for r in R if r["T"] >= 100000]
    if expect_pool is not None:
        assert len(pool) == expect_pool, len(pool)
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
    out = {"what": what, "salt": SALT, "cut": CUT, "r50_max": R50_MAX,
           "inputs_sha256": inputs_sha, "selected": sel, "pool": rows,
           "summary": {"pool": len(rows), "eligible": sum(x["eligible"] for x in rows),
                       "selected": len(sel), "by_stratum": {s: sum(x["stratum"] == s for x in sel) for s in ("capped", "finished")},
                       "distinct_questions": len({x["doc"] for x in sel})}}
    return out


def panel_counts():
    if fsha(f"{DISC}/summary.json") != SUMMARY_SHA:
        sys.exit("discovery summary.json changed")
    return {row["doc"]: row["all"] for row in json.load(open(f"{DISC}/summary.json"))["panel"]}


def main():
    if sys.argv[1] == "--reproduce-bplus":
        inputs = {f"{DISC}/replies_all.json": "e2cb8516650a44771d6dbb319be603b4defc6d78df199ec62305ce0c76d7296f",
                  f"{DISC}/summary.json": SUMMARY_SHA}
        for p, s in inputs.items():
            if fsha(p) != s:
                sys.exit(f"input changed: {p}")
        out = select(json.load(open(f"{DISC}/replies_all.json")), panel_counts(), SALT_BPLUS, inputs,
                     "B+ prefix selection (ids, counts and metrics only)", expect_pool=29)
        json.dump(out, open(sys.argv[2], "w"), indent=1)
        same = open(sys.argv[2], "rb").read() == open(BPLUS_SEL, "rb").read()
        print("reproduction of bplus/out/selection.json:", "BYTE-IDENTICAL" if same else "DIFFERENT")
        return 0 if same else 1
    od = sys.argv[1]
    inputs = {f"{od}/replies_all.json": fsha(f"{od}/replies_all.json"), f"{DISC}/summary.json": SUMMARY_SHA}
    out = select(json.load(open(f"{od}/replies_all.json")), panel_counts(), SALT_D091, inputs,
                 "d091 prefix selection, B+ rule (ids, counts and metrics only); nothing is continued in this package")
    json.dump(out, open(f"{od}/selection.json", "w"), indent=1)
    for s in out["selected"]:
        print(s["prefix_id"], s["id"], s["stratum"], "R50@100k", s["R50_at_100k"], "|", s["reason"])
    print(json.dumps(out["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
