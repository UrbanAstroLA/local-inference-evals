#!/usr/bin/env python3
"""22ao live causal-length check: frozen analysis (CPU; LIVE-CHECK-PROPOSAL.md sections 2, 4-8).

usage: analyze_22ao.py RUN_DIR      (RUN_DIR holds arm-a/ and arm-c/, each with mapguard/*.jsonl and client.jsonl)
Writes RUN_DIR/result.json and RUN_DIR/result.md. Reads guard and client records only (no GPQA text exists in them).

Per arm and invariant the outcome is exactly one of:
  INCOMPLETE             instrument failure (guard errors, a positive-control plant class not planted or not detected
                         at every planted token, record overflow, attribution < 99%), async scheduling off, or the
                         required exposure for that invariant not met. Blocked host waits > 1% of steps make the
                         level-1 invariant INCOMPLETE unless the discrepancy is itself observed at the required counts.
  POSITIVE FINDING       a violation counted by a validated instrument (stock's known D2 tail-ring violations are
                         reported here as known findings; they never invalidate the session)
  CLEAN WITHIN COVERAGE  required exposure met, controls passed, no violation; reported with the rule-of-three
                         95% upper bound 3/N on the per-row rate
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

MIN = dict(dense_after_reject=300, dense_after_accept=100, clamped=300, recycles=10, recycles_shorter=5,
           full_per_size=20, transitions=20)
SIZES = (4, 8, 16, 24, 32)
LREC_COLS = ["step", "batch_row", "tag", "position", "consumed_len", "d", "dense", "row_tokens"]
L1 = ["len_pos", "len_neg"]
L2 = ["sel_causal", "sel_foreign", "masked_live", "sel_len", "nsa_pos"]
D2_KNOWN = [("tail_w_stash", "mism"), ("tail_w_stash", "null"), ("tail_w_stash", "foreign"), ("tail_w_seed", "mism"),
            ("idx_w_dec", "mism"), ("idx_w_dec", "missing")]


def load(path):
    out = []
    for f in sorted(glob.glob(path)):
        for line in open(f, errors="replace"):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def arm(ad):
    res = dict(dir=ad, present=os.path.isdir(os.path.join(ad, "mapguard")))
    if not res["present"]:
        res["status"] = "INCOMPLETE: no guard records"
        return res
    ranks = {}
    for f in sorted(glob.glob(os.path.join(ad, "mapguard", "mg_p*.jsonl"))):
        recs = load(f)
        init = next((r for r in recs if r.get("kind") == "init"), None)
        polls = [r for r in recs if r.get("kind") == "poll"]
        if init is None or not polls:
            continue
        ranks[os.path.basename(f)] = dict(recs=recs, init=init, last=polls[-1], polls=polls)
    sched = load(os.path.join(ad, "mapguard", "ms_p*.jsonl"))
    client = load(os.path.join(ad, "client.jsonl"))
    res["ranks"] = list(ranks)
    # ---- instrument validity
    errors = sum(1 for d in ranks.values() for r in d["recs"] if r.get("kind") == "guard_error")
    async_on = all(d["init"].get("async_scheduling") for d in ranks.values()) if ranks else False
    cnt = Counter()
    for d in ranks.values():
        for site, cs in d["last"]["counters"].items():
            for c, v in cs.items():
                cnt[(site, c)] += int(v)
    plants = {c: dict(planted=cnt[("mla_r", f"p{c}_planted")], det=cnt[("mla_r", f"p{c}_det")]) for c in "abcd"}
    plants_ok = all(v["planted"] > 0 and v["det"] == v["planted"] for v in plants.values())
    uploads = sum(int(d["last"].get("uploads", 0)) for d in ranks.values())
    blocked = sum(int(d["last"].get("blocked_waits", 0)) for d in ranks.values())
    lrec_over = any(int(d["last"].get("n_lrec", 0)) > 131072 for d in ranks.values())
    # ---- attribution: lrec rows -> step rows -> request -> previous step's rejected count
    order = [r["fp"] for r in sched if r.get("kind") == "sched"]
    upd = {r["fp"]: r.get("rej", {}) for r in sched if r.get("kind") == "upd"}
    prev_rej = {}            # (fp, rid) -> rejected drafts of rid's previous spec step
    last = {}
    for fp in order:
        for rid in upd.get(fp, {}):
            prev_rej[(fp, rid)] = last.get(rid)
        for rid, v in upd.get(fp, {}).items():
            last[rid] = int(v[2])
    rows_l = []
    joined = total = 0
    exposure = Counter()
    for name, d in ranks.items():
        steps = {r["step"]: r for r in d["recs"] if r.get("kind") == "step"}
        lrec = [x for p in d["polls"] for x in p.get("lrec", [])]
        for x in lrec:
            row = dict(zip(LREC_COLS, x))
            total += 1
            st = steps.get(row["step"])
            if st is None or row["batch_row"] >= len(st["rows"]) or st.get("warm"):
                continue
            r = st["rows"][row["batch_row"]]
            rid, nspec = r[0], r[5]
            if r[1] != row["tag"]:
                continue
            joined += 1
            pr = prev_rej.get((st.get("fp"), rid))
            verify = nspec > 0
            row.update(rank=name, rid=rid, verify=verify, prev_rej=pr, reuse=r[10] if len(r) > 10 else 0,
                       rows_in_step=len(st["rows"]))
            rows_l.append(row)
            if verify and row["dense"] and pr is not None and pr >= 1:
                exposure["dense_after_reject"] += 1
            if verify and row["dense"] and pr == 0:
                exposure["dense_after_accept"] += 1
            if verify and not row["dense"]:
                exposure["clamped"] += 1
    attribution = joined / total if total else 0.0
    # ---- shapes, graph modes, recycles (first rank's step records)
    first = ranks[sorted(ranks)[0]]["recs"] if ranks else []
    steps0 = [r for r in first if r.get("kind") == "step" and not r.get("warm")]
    full_by = Counter(r.get("ntok_pad") for r in steps0 if r.get("cg") == "FULL")
    trans = sum(1 for a, b in zip(steps0, steps0[1:]) if a.get("ntok_pad") in SIZES and b.get("ntok_pad") in SIZES
                and a.get("ntok_pad") != b.get("ntok_pad"))
    cg_mix = Counter(r.get("cg") for r in steps0)
    occ = {}                 # runner row -> (rid, last hi)
    rec_n = rec_short = 0
    for st in steps0:
        for r in st["rows"]:
            rid, hi, rrow = r[0], r[3], r[6]
            if rrow in occ and occ[rrow][0] != rid:
                rec_n += 1
                if hi < occ[rrow][1]:
                    rec_short += 1
            occ[rrow] = (rid, hi)
    exposure.update(recycles=rec_n, recycles_shorter=rec_short, transitions=trans)
    # ---- steady state and drain
    sends = [c["t0"] for c in client if c.get("kind") == "request"]
    t_last_send = max(sends) if sends else None
    steady = sum(1 for st in steps0 if len(st["rows"]) == 8)
    drain = sum(1 for st in steps0 if len(st["rows"]) < 8 and t_last_send and st.get("t", 0) > t_last_send)
    # ---- level 1 / level 2 counts
    dvals = Counter(r["d"] for r in rows_l if r["verify"])
    lvl1 = dict(len_checked=cnt[("mla_len", "len_checked")], len_pos=cnt[("mla_len", "len_pos")],
                len_neg=cnt[("mla_len", "len_neg")], d_verify_rows=dict(sorted(dvals.items())),
                d_by_prev_rej={str(k): dict(sorted(Counter(r["d"] for r in rows_l if r["verify"] and r["prev_rej"] == k).items()))
                               for k in sorted({r["prev_rej"] for r in rows_l if r["verify"]}, key=lambda v: (v is None, v))},
                d_on_recycled_rows=dict(sorted(Counter(r["d"] for r in rows_l if r["verify"] and r["reuse"]).items())))
    lvl2 = {c: cnt[("mla_r", c)] for c in L2 + ["nsa_neg", "checked", "sel_hole", "inact_read"]}
    viol_recs = [x for d in ranks.values() for p in d["polls"] for x in p.get("records", []) if x.get("site") == 7]
    d2_known = {f"{s}.{c}": cnt[(s, c)] for s, c in D2_KNOWN if cnt[(s, c)]}
    # ---- outcomes
    instrument_ok = errors == 0 and plants_ok and not lrec_over and attribution >= 0.99 and async_on
    why = [w for w, bad in (("guard errors", errors), ("plants", not plants_ok), ("lrec overflow", lrec_over),
                            ("attribution", attribution < 0.99), ("async off", not async_on)) if bad]
    rows_ok = all(exposure[k] >= MIN[k] for k in ("dense_after_reject", "dense_after_accept", "clamped"))
    blocked_frac = blocked / uploads if uploads else 1.0
    n_exp = exposure["dense_after_reject"] + exposure["dense_after_accept"] + exposure["clamped"]

    def outcome(viol, level):
        if not instrument_ok:
            return "INCOMPLETE (instrument: " + ", ".join(why) + ")"
        if viol > 0:
            return f"POSITIVE FINDING ({viol})"
        if not rows_ok:
            return "INCOMPLETE (required exposure not met)"
        if level == 1 and blocked_frac > 0.01:
            return f"INCOMPLETE (blocked host waits {blocked_frac:.3%} > 1%)"
        return f"CLEAN WITHIN COVERAGE (95% upper bound {3 / n_exp:.4f} per row, N = {n_exp})" if n_exp else "INCOMPLETE"
    outcomes = {"L1 consumed length == p + 1 (signed)": outcome(lvl1["len_pos"] + lvl1["len_neg"], 1)}
    for c in L2:
        outcomes[f"L2 {c}"] = outcome(lvl2[c], 2)
    coverage = {k: dict(count=exposure[k], min=MIN[k], met=exposure[k] >= MIN[k]) for k in
                ("dense_after_reject", "dense_after_accept", "clamped", "recycles", "recycles_shorter", "transitions")}
    coverage["full_per_size"] = {str(s): dict(count=full_by.get(s, 0), min=MIN["full_per_size"],
                                              met=full_by.get(s, 0) >= MIN["full_per_size"]) for s in SIZES}
    res.update(instrument=dict(ok=instrument_ok, guard_errors=errors, plants=plants, plants_ok=plants_ok,
                               attribution=round(attribution, 4), joined=joined, lrec_rows=total, async_scheduling=async_on,
                               uploads=uploads, blocked_waits=blocked, blocked_frac=round(blocked_frac, 5)),
               coverage=coverage, steady_steps=steady, drain_steps=drain, cg_mix=dict(cg_mix),
               level1=lvl1, level2=lvl2, level2_records=viol_recs[:200], known_d2=d2_known, outcomes=outcomes,
               client=dict(requests=len(sends), errors=sum(1 for c in client if c.get("kind") == "request" and c.get("error")),
                           skipped=sum(1 for c in client if c.get("kind") == "skipped")))
    return res


def main():
    run = sys.argv[1]
    out = dict(rules=dict(minima=MIN, outcomes=["INCOMPLETE", "POSITIVE FINDING", "CLEAN WITHIN COVERAGE"]),
               arms={a: arm(os.path.join(run, f"arm-{a}")) for a in ("a", "c")})
    json.dump(out, open(os.path.join(run, "result.json"), "w"), indent=1, default=str)
    L = ["# 22ao live check: result (frozen analysis)", ""]
    for a, r in out["arms"].items():
        L.append(f"## Arm {a.upper()}")
        if not r.get("present"):
            L += [f"- {r.get('status')}", ""]
            continue
        L.append(f"- instrument ok: {r['instrument']['ok']} (guard errors {r['instrument']['guard_errors']}, plants "
                 f"{r['instrument']['plants']}, attribution {r['instrument']['attribution']}, async {r['instrument']['async_scheduling']}, "
                 f"blocked waits {r['instrument']['blocked_waits']}/{r['instrument']['uploads']})")
        for k, v in r["outcomes"].items():
            L.append(f"- {k}: **{v}**")
        L.append(f"- coverage: {json.dumps(r['coverage'])}")
        L.append(f"- level 1: {json.dumps(r['level1'])}")
        L.append(f"- level 2: {json.dumps(r['level2'])}")
        L.append(f"- known D2 findings: {json.dumps(r['known_d2'])}")
        L.append(f"- steady / drain steps: {r['steady_steps']} / {r['drain_steps']}; graph modes {r['cg_mix']}; client {r['client']}")
        L.append("")
    open(os.path.join(run, "result.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
