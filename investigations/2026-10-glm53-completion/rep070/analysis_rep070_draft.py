#!/usr/bin/env python3
"""DRAFT, NOT FROZEN. Analysis of a repeatability run of stock tpurtell 0.7.0 K3.25 (DFlash2 x5) under the recorded
GPQA seeds (specification: PROPOSAL.md section 6). To be frozen (sha256 recorded in a dated amendment)
before any request is sent; thresholds marked PROPOSED are for the operator and assessor to fix.

Machinery reused unchanged, by import, from the frozen column-511 analysis (analysis_col511.py, sha256 b5b13ce2...,
checked at start): row loading and validity checks (load_arm), pairing checks, the exact sign-flip test, the question
bootstrap, the stated-answer adjudicator, chain/server-log parsing and timing. Roles:
  rep    = the new 0.7.0 run (three passes, seeds 1234/1235/1236, proxy request log required)
  v091   = the existing 0.9.1 K3.25 three-pass record (historical)       -> comparison C1 (does the advantage reproduce?)
  old070 = the existing 0.7.0 K3.25 three-pass record (historical)       -> comparison C2 (repeatability)
The old 0.7.0 record's pass 1 has no request log or server log; checks that need them are skipped for that pass only.

usage: analysis_rep070_draft.py --rep-pass P1 P2 P3 --rep-reqlog S1 S2 [--rep-serverlog ...] [--rep-chainlog ...]
       [--rep-hedged P:DOC ...] [--rep-gpu-hours X] [--dry-run-label TEXT] --out OUT.json > OUT.md
Exit code 3 when a validity check fails.
"""
import argparse, hashlib, json, os, sys

FROZEN = "<scratch>/kdcp/arm-col511"
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
# PROPOSED operational thresholds (not causal boundaries); rep empties of 594; old 0.7.0 record 4, 0.9.1 record 16.
ALPHA, O1_MAX, O2_MIN = 0.05, 8, 12
BOOT_SEED_C2 = M.BOOT_SEED + 10   # 20261021, C2 effect interval


def reading(n, p1, p2):
    if n <= O1_MAX and p1 < ALPHA:
        return ("O1", "substantially fewer empties again: replication support for the 0.7.0 advantage relative to the 0.9.1 record "
                      "(repeatability under the recorded seeds, historical comparators); the tail-mapping/DCP interaction on 0.7.0 "
                      "becomes the concrete candidate (amendment 12)")
    if n >= O2_MIN or p2 < ALPHA:
        return ("O2", "movement toward the newer engine's count: the original gap is reported as not reproduced, without claiming "
                      "equivalence; no expensive bisection without a reproducible target (amendment 12)")
    return ("O3", "ambiguous: report the uncertainty and stop this allocation (amendment 12)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep-label", default="0.7.0 K3.25 DFlash2 x5, repeatability run (stock image 48e254d9)")
    ap.add_argument("--rep-pass", nargs=3, required=True)
    ap.add_argument("--rep-reqlog", nargs="*", default=[])
    ap.add_argument("--rep-serverlog", nargs="*", default=[])
    ap.add_argument("--rep-chainlog", nargs="*", default=[])
    ap.add_argument("--rep-hedged", nargs="*", default=[])
    ap.add_argument("--rep-published", default=None, help="dry runs only")
    ap.add_argument("--rep-gpu-hours", type=float, default=None)
    ap.add_argument("--require-rep-reqlog", type=int, default=1)
    ap.add_argument("--dry-run-label", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, ca, ia = M.load_arm(a.rep_label, a.rep_pass, a.rep_reqlog, a.rep_hedged, a.rep_published)
    recs, checks, inputs = {}, list(ca), {"rep": ia}
    for k, r in RECORDS.items():
        rows, c, i = M.load_arm(r["label"], r["passes"], r["reqlogs"], r["hedged"], r["published"])
        recs[k] = rows; checks += c; inputs[k] = i
    if a.require_rep_reqlog:
        checks.append(("rep: every pass has its proxy request log (finish reasons and token counts)",
                       all(row["logged"] for row in A.values()), sum(row["logged"] for row in A.values())))
    for k in recs:
        checks += [(f"rep vs {k}: {n}", ok, d) for n, ok, d in M.check_pairing(A, recs[k])]
    adj = M.adjudicator_status()
    checks.append(("stated adjudicator source unchanged since freeze (informative)", bool(adj["source_unchanged"]), adj["current_source_sha256"]))
    valid = all(ok for n, ok, _ in checks if not n.startswith("stated adjudicator"))
    res = {"draft": True, "dry_run": a.dry_run_label, "frozen_module_sha256": FROZEN_SHA,
           "script_sha256": M.file_sha(os.path.abspath(__file__)), "inputs": inputs, "checks": [list(c) for c in checks], "valid": valid,
           "thresholds_proposed": {"alpha": ALPHA, "O1_max_empties": O1_MAX, "O2_min_empties": O2_MIN}}
    n = M.N_Q * M.N_PASSES
    if all(len(x) == n for x in [A, *recs.values()]):
        c1 = M.analyze(A, recs["v091"], a.rep_label, RECORDS["v091"]["label"])   # arm = rep, cmp = v091: T = v091 - rep
        c2 = M.analyze(A, recs["old070"], a.rep_label, RECORDS["old070"]["label"])
        for c in (c1, c2):   # the column-511 bands and readings do not apply here
            c["primary"].pop("band"); c["primary"].pop("reading")
        ea, eo = M.per_question(A, "empty"), M.per_question(recs["old070"], "empty")
        d2 = [x - y for x, y in zip(ea, eo)]                                      # rep - old070: movement toward more empties
        t2 = M.signflip(d2)
        lo2, hi2 = M.boot_interval(d2, n, BOOT_SEED_C2)
        p1 = c1["primary"]["test"]["p_one_sided"]
        nrep = c1["primary"]["arm_empties"]
        code, text = reading(nrep, p1, t2["p_one_sided"])
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
                                          "note": "same engine, same seeds: shared sampler noise until batching makes trajectories diverge; descriptive"},
            "reading": {"rep_empties": nrep, "p1_one_sided_rep_fewer_than_v091": p1, "p2_one_sided_rep_more_than_old070": t2["p_one_sided"],
                        "outcome": code, "text": text,
                        "limits": "historical comparators for both C1 and C2; exchangeability assumed, not designed; repeatability under "
                                  "the recorded seeds, not a fresh-seed estimate; thresholds are operational"}})
        ch = M.parse_chainlogs(a.rep_chainlog, 2)
        res["timing_rep"], win = M.timing(A, "rep", ch, a.rep_gpu_hours, 2)
        res["server_rep"] = M.parse_serverlogs(a.rep_serverlog, win, 0)
    json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)
    L = [f"# Repeatability run, stock 0.7.0 (DRAFT analysis)" + (f" -- DRY RUN: {a.dry_run_label}" if a.dry_run_label else ""), "",
         f"valid: **{valid}**; frozen module {FROZEN_SHA[:8]}; thresholds (proposed) O1 <= {O1_MAX} with p1 < {ALPHA}; O2 >= {O2_MIN} or p2 < {ALPHA}", "",
         "## Checks"] + [f"- [{'ok' if ok else 'FAIL'}] {nm} ({d})" for nm, ok, d in checks]
    if "reading" in res:
        r, c1p, c2p = res["reading"], res["C1_rep_vs_v091"]["primary"], res["C2_rep_vs_old070"]
        L += ["", "## Reading", f"- rep empties {r['rep_empties']} of 594 (old 0.7.0 record {c2p['primary']['cmp_empties']}, 0.9.1 record {c1p['cmp_empties']})",
              f"- C1 (rep fewer than 0.9.1): one-sided p = {M.fmt_p(r['p1_one_sided_rep_fewer_than_v091'])}; reduction {100 * c1p['effect_empty_rate_reduction']['estimate']:+.2f} points "
              f"(95% {100 * c1p['effect_empty_rate_reduction']['ci95'][0]:+.2f} to {100 * c1p['effect_empty_rate_reduction']['ci95'][1]:+.2f})",
              f"- C2 (rep more than old 0.7.0): one-sided p = {M.fmt_p(r['p2_one_sided_rep_more_than_old070'])}; increase "
              f"{100 * c2p['repeatability_test']['effect_empty_rate_increase']['estimate']:+.2f} points (95% "
              f"{100 * c2p['repeatability_test']['effect_empty_rate_increase']['ci95'][0]:+.2f} to {100 * c2p['repeatability_test']['effect_empty_rate_increase']['ci95'][1]:+.2f})",
              f"- outcome **{r['outcome']}**: {r['text']}", f"- limits: {r['limits']}",
              f"- identical answered-reply hashes vs old 0.7.0 per pass: {res['repeatability_descriptive']['identical_answered_reply_hash_vs_old070_per_pass']}",
              f"- empty in both rep and old 0.7.0: {res['repeatability_descriptive']['empty_in_both_rep_and_old070'] or '-'}"]
    print("\n".join(L))
    return 0 if valid else 3


if __name__ == "__main__":
    sys.exit(main())
