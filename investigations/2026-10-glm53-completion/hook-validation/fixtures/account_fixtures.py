#!/usr/bin/env python3
"""Fixture accounting (PLAN.md amendment 21 section 8) and the frozen fixture list for the first long-context
contrast (section 5). CPU only; reads fixture-inventory/inventory.json (sha256-checked) and
probe-proposal/fixtures/fixtures.json; writes accounting.json and first_contrast.json. Ids, positions, classes and
hashes only; no reasoning text is read.

Accounting: the inventory's 901 text histories = 527 natural close + 224 loop + 92 exhaustion (the 20b classes)
+ 18 exhaustion_cut120k + 40 error. Every row is characterised by arm, engine, weights, state, censoring, length,
end-repetition metrics and K3.25 eligibility; the classified 843 are also split by weights, since the probe runs
K3.25 only.
"""
import collections
import hashlib
import json
import os

ROOT = "<runs>"
INVP = f"{ROOT}/fixture-inventory/inventory.json"
FIXP = f"{ROOT}/probe-proposal/fixtures/fixtures.json"
HERE = os.path.dirname(os.path.abspath(__file__))
W = 8192
ABS_LIMIT = 131072


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    inv = json.load(open(INVP))
    rows = inv["text_rows"]
    out = dict(inventory_sha256=sha256(INVP), fixtures_sha256=sha256(FIXP), n_rows=len(rows))
    st = collections.Counter(r["state"] for r in rows)
    out["states"] = dict(st)
    cls3 = {"natural_close", "loop", "exhaustion"}
    other = [r for r in rows if r["state"] not in cls3]
    out["n_in_three_classes"] = sum(st[s] for s in cls3)
    out["n_other"] = len(other)
    assert out["n_in_three_classes"] == 843 and len(other) == 58, (out["n_in_three_classes"], len(other))

    def char(rs):
        nt = sorted(int(r["n_reasoning_tokens"] or 0) for r in rs)
        return dict(
            n=len(rs), arms=dict(collections.Counter(r["arm"] for r in rs)),
            engines=dict(collections.Counter(r["engine"] for r in rs)),
            weights=dict(collections.Counter(r["weights"] for r in rs)),
            finish=dict(collections.Counter(str(r["finish_reason"])[:60] for r in rs)),
            censored=dict(collections.Counter(str(r["censored"]) for r in rs)),
            spec=dict(collections.Counter(str(r["spec"]) for r in rs)),
            conc=dict(collections.Counter(str(r["conc"]) for r in rs)),
            docs=dict(collections.Counter(str(r["doc"]) for r in rs)),
            seeds=dict(collections.Counter(str(r["seed"]) for r in rs)),
            reasoning_tokens=dict(min=nt[0], median=nt[len(nt) // 2], max=nt[-1],
                                  zero=sum(1 for x in nt if x == 0), ge_64k=sum(1 for x in nt if x >= 65536)),
            end_repeating=dict(collections.Counter(str((r.get("end_metrics") or {}).get("repeating"))
                                                   for r in rs)),
            any_repeating_checkpoint=sum(1 for r in rs if r.get("repeating_checkpoints")),
            closed=sum(1 for r in rs if r.get("closed")),
            special_ids_in_retok=sum(sum((r.get("special_ids_in_retok") or {}).values()) for r in rs),
            k325=sum(1 for r in rs if str(r["weights"]).startswith("K3.25")))

    out["other_by_state"] = {s: char([r for r in other if r["state"] == s]) for s in ("exhaustion_cut120k", "error")}
    out["error_rows_with_text"] = [dict(id=r["id"][:12], doc=r["doc"], n_reasoning_tokens=r["n_reasoning_tokens"],
                                        finish=str(r["finish_reason"])[:60], end=r.get("end_metrics"))
                                   for r in other if r["state"] == "error" and (r["n_reasoning_tokens"] or 0) > 0]
    # the classified 843 by weights (K4 rows are outside any K3.25 probe as well)
    out["classified_by_weights"] = {s: dict(collections.Counter(
        "K3.25" if str(r["weights"]).startswith("K3.25") else str(r["weights"]) for r in rows if r["state"] == s))
        for s in sorted(cls3)}
    k4 = [r for r in rows if not str(r["weights"]).startswith("K3.25")]
    out["k4_rows_total"] = len(k4)
    out["k4_rows_by_arm_state"] = {f"{a} | {s}": n for (a, s), n in
                                   sorted(collections.Counter((r["arm"], r["state"]) for r in k4).items())}
    out["k325_eligible_by_state"] = dict(collections.Counter(r["state"] for r in rows
                                                             if str(r["weights"]).startswith("K3.25")))
    out["reading"] = (
        "The 58 rows outside 527/224/92 are all K4-weight histories (K4@a5fee929), so none is eligible for a "
        "K3.25 probe whatever its state. 40 'error' rows = the S7K4p screen (v0.7.0 local kpoolring, K4, DFlash2 x5, "
        "8 in flight), declared INVALID for an engine crash: 36 'no stream data' with no reasoning text, 4 'stream "
        "ended without a finish reason' with partial text (925 - 198,565 tokens, not repeating at the end): "
        "unterminated by a server failure, i.e. neither close, loop nor exhaustion. 18 'exhaustion_cut120k' rows = "
        "k4-k5-share-screen replay-C-k4-default (v0.8.0 stock, K4, DFlash2 x3, 8 in flight): the client cut every "
        "reply at 120k tokens (administrative censoring by the client, not the engine cap); none was repeating at "
        "the cut (17 of 18 never repeating at any checkpoint, 1 repeating at 88k-96k but not at the end), so under "
        "the 20b rule they are client-censored exhaustions. They are kept out of the 92 exhaustions because the "
        "censoring mechanism differs (client cut at 120k, not the server's max_tokens). In addition, the three "
        "classes themselves contain K4 rows (natural close 62, loop 12, exhaustion 5); they are excluded from "
        "fixtures for the same weights reason.")
    json.dump(out, open(os.path.join(HERE, "accounting.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("states", "n_in_three_classes", "n_other", "classified_by_weights",
                                          "k4_rows_total", "k325_eligible_by_state")}, indent=1))
    for s, c in out["other_by_state"].items():
        print("==", s, json.dumps({k: c[k] for k in ("n", "arms", "weights", "finish", "censored",
                                                    "reasoning_tokens", "end_repeating", "any_repeating_checkpoint",
                                                    "docs", "k325")}))
    return inv


# ------------------------------------------------------------------ frozen first-contrast list
FIRST = [  # (fixture id in probe-proposal/fixtures.json, why)
    ("L1", "near close, doc 13, stock 0.7.0 history; window ends AT the close (positive control: </think> next)"),
    ("L4", "near close, doc 88, stock 0.9.1 history (seed 5005)"),
    ("M3", "near close, doc 121, stock 0.7.0 seed-replay history"),
    ("M4", "near close, doc 127, stock 0.7.0 S7 history (2,795-token prompt)"),
    ("L3", "near close, doc 79, stock 0.9.1 history (seed 5004)"),
    ("L5", "active exhaustion, doc 170, pilot r0 history (no close in reach)"),
    ("A8", "active, loop pre-onset (onset 184k), doc 79, stock 0.7.0 history"),
    ("L8", "active exhaustion, doc 79, stock 0.9.1 history (seed 5010)"),
]


def first_contrast(inv):
    fx = json.load(open(FIXP))
    byf = {}
    for b, v in fx["batches"].items():
        for x in v:
            byf[x["fixture"]] = dict(x, batch=b)
    rows = {r["id"][:12]: r for r in inv["text_rows"]}
    out = []
    for fid, why in FIRST:
        x = byf[fid]
        r = rows.get(x["source_id"][:12])
        assert r is not None and str(r["weights"]).startswith("K3.25"), fid
        end_idx = int(x["window_end_index"])
        P0 = int(x["prompt_tokens"])
        d = (P0 + end_idx - W) % 8
        start = end_idx - W - d
        assert (P0 + start) % 8 == 0 and P0 + end_idx < ABS_LIMIT
        out.append(dict(fixture=fid, why=why, source_id=x["source_id"], doc=x["doc"], seed=x.get("seed"),
                        source_engine=x["source_engine"], cls=x["cls"], prompt_tokens=P0,
                        history_sha256=x["history_sha256"], weights=r["weights"],
                        window_end_reasoning_index=end_idx, window_start_reasoning_index=start,
                        prefix_len_P=P0 + start, P_mod8=(P0 + start) % 8, W_forced=end_idx - start,
                        abs_window_end=P0 + end_idx, class_20b_at_window=x["class_20b_at_window"],
                        repetition_near_window_end=x["repetition_near_window_end"],
                        forced_token_at_end=x["forced_token_at_end"],
                        proposal_record={k: x[k] for k in x if k not in ("checkpoints",)}))
    docs = collections.Counter(o["doc"] for o in out)
    engines = collections.Counter("0.7.0" if "0.7.0" in o["source_engine"] else "0.9.1" for o in out)
    classes = collections.Counter(o["cls"] for o in out)
    rule = dict(
        W_forced=f"{W} + d tokens, d = (P0 + end - {W}) mod 8, so that the prefix length P = prompt + start is a "
                 "multiple of 8 (window starts on a pool and a ring boundary; no window pool has prefill-seeded "
                 "members, so the D2 injection never mixes the corrected seed with the old decode mapping); the "
                 "last forced token is the history's own next token (</think> for windows ending at a close)",
        abs_limit=f"every window ends below {ABS_LIMIT} (Stage K has not validated selection beyond)",
        composition="the same 8 fixtures in this order in ONE batch of 8 in flight, submitted together after the "
                    "warm phase; identical in every run of both arms",
        designated="primary positions fixed at the Stage-0 freeze: the end position e, the last 512 positions, "
                   "a 64-token grid after the 512-token lead-in, and the non-repeating commitment events inside the "
                   "window (positions only, from inventory.json)",
        ids="exact window ids, prefix ids and their sha256 are fixed at the Stage-0 freeze by re-tokenizing in a "
            "CPU container (prep as phase-2 prep_requests.py); this list freezes WHICH histories and WHERE")
    res = dict(fixtures=out, batch_order=[o["fixture"] for o in out], docs=dict(docs), engines=dict(engines),
               classes=dict(classes), rule=rule,
               note="Shared counterfactual fixtures (re-tokenized text), not replays; 6 questions (doc 79 three "
                    "times); K3.25 only; no forced continuation past a natural close.")
    json.dump(res, open(os.path.join(HERE, "first_contrast.json"), "w"), indent=1)
    print("first contrast:", res["batch_order"], dict(docs), dict(engines), dict(classes))


if __name__ == "__main__":
    inv = main()
    first_contrast(inv)
