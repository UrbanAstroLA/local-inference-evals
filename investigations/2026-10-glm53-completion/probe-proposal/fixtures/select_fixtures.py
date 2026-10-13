#!/usr/bin/env python3
"""Fixture selection for the forced-decode window probe (PLAN.md amendment 20 section 3; 20b).

CPU only. Reads ../../fixture-inventory/inventory.json (sha256-checked against that folder's SHA256SUMS) and
writes fixtures.json: ids, positions, classes, repetition state and hashes only. No reasoning or GPQA text is read
or written; window-level id hashes are computed at freeze time (Stage 0), when the ids are re-tokenized.

Conventions (inventory): reasoning index 0 = first token after <think> (154841). Absolute position = rendered
prompt length + reasoning index. A window "ending at e" forces reasoning indices [e - W, e] (W + 1 predictions);
the prediction recorded at index e is of reasoning token e, which for a close window is </think> (154842), the
history's own next token (positive control). Nothing after the close is forced (amendment 20 section 2).
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
INV_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "fixture-inventory"))
INV = os.path.join(INV_DIR, "inventory.json")

W_LONG = 8192          # forced history tokens in a long window (2,048 decode-written index-K pools)
LEAD_IN = 512          # first window positions excluded from primary analysis (stagger, decode start)
TAIL = 512             # dense primary positions at the window end
GRID = 64              # grid spacing elsewhere in the window
ABS_LIMIT = 131072     # B1/B0 tiled second supertile chunk starts here (SOURCE-REVIEW item 2); stay below
CLOSE_ID = 154842

# (fixture id, source row id, class, window end e or "close", note).  Frozen choice; see PROPOSAL.md section 5.
STAGE1_L = [  # batch L: 8 long windows, run together (8 in flight)
    ("L1", "ca7fd76b2b7a", "near_close", "close", "stock 0.7.0 history, doc 13"),
    ("L2", "e87744068243", "near_close", "close", "stock 0.7.0 history, doc 88"),
    ("L3", "09ba09a88d7b", "near_close", "close", "stock 0.9.1 history, doc 79 seed 5004 (pairs with L6)"),
    ("L4", "f0c2498e7b7f", "near_close", "close", "stock 0.9.1 history, doc 88"),
    ("L5", "fc3010f3a808", "exhaustion_active", 120000, "phase-2 r0 history (doc 170), active, no close in reach"),
    ("L6", "a503e4fce96a", "doc79_pair_active", 97408, "stock 0.7.0 doc 79 seed 5004, same end index as L3; closes 51.7k later"),
    ("L7", "c85856ae6b1f", "doc79_pair_exhaustion", 120000, "stock 0.7.0 doc 79 seed 5010, exhaustion (pairs with L8)"),
    ("L8", "77100ed719f5", "doc79_pair_exhaustion", 120000, "stock 0.9.1 doc 79 seed 5010, exhaustion (pairs with L7)"),
]
STAGE1_S = [  # batch S: short controls, full replay from the prompt (W = close index), 8 in flight
    ("S1", "57703691f509", "short_exact", "close", "exact ids, 0.9.1 EP2/DCP2 bundle (step 6), doc 1"),
    ("S2", "53fa5b3e066f", "short_exact", "close", "exact ids, 0.7.0 + kpool fixes (step 5), doc 1"),
    ("S3", "be2e9e349395", "short_exact", "close", "exact ids, 0.9.1 EP2/DCP2 bundle (step 6), doc 3"),
    ("S4", "4b92a6a90ec4", "short_exact", "close", "exact ids, 0.7.0 + kpool fixes (step 5), doc 3"),
    ("S5", "9143682aa636", "short_text", "close", "stock 0.7.0 doc 79 seed 5001"),
    ("S6", "7a64a6fa06d2", "short_text", "close", "stock 0.9.1 pilot doc 55"),
    ("S7", "f4f28e056fb3", "short_text", "close", "stock 0.7.0 S7 doc 13"),
    ("S8", "964a74b9edf1", "short_text", "close", "stock 0.9.1 replication doc 109"),
]
STAGE2_L2 = [  # batch L2: 8 further natural closes at 64k-128k from stock engines
    ("M1", "17d671f680b5", "near_close", "close", "stock 0.7.0, doc 79"),
    ("M2", "6800a0e38d91", "near_close", "close", "stock 0.7.0, doc 88 (re-tokenization drift 7)"),
    ("M3", "cd6c18552504", "near_close", "close", "stock 0.7.0 seed-replay, doc 121"),
    ("M4", "c253c576662a", "near_close", "close", "stock 0.7.0 S7, doc 127"),
    ("M5", "70b6b6b1ae38", "near_close", "close", "stock 0.9.1 doc 79 seed 5012"),
    ("M6", "73700fa6eb27", "near_close", "close", "stock 0.9.1 doc 88 seed 5010"),
    ("M7", "a68d65155475", "near_close", "close", "stock 0.9.1 doc 88 seed 5001"),
    ("M8", "dcaf4cdad47a", "near_close", "close", "stock 0.9.1 doc 88 seed 5011"),
]
STAGE2_L3 = [  # batch L3: the same histories away from the close, and other active windows
    ("A1", "ca7fd76b2b7a", "pre_close_active", "close-8192", "L1 history, window ending 8,192 before its close"),
    ("A2", "e87744068243", "pre_close_active", "close-8192", "L2 history"),
    ("A3", "09ba09a88d7b", "pre_close_active", "close-8192", "L3 history"),
    ("A4", "f0c2498e7b7f", "pre_close_active", "close-8192", "L4 history"),
    ("A5", "fc3010f3a808", "exhaustion_active", 64000, "phase-2 r1 end (doc 170)"),
    ("A6", "99441a83fffd", "doc79_pair_exhaustion", 96000, "stock 0.7.0 doc 79 seed 5002 (pairs with A7)"),
    ("A7", "d5bafa393d8f", "doc79_pair_exhaustion", 96000, "stock 0.9.1 doc 79 seed 5002 (pairs with A6)"),
    ("A8", "4db8bbe82d5c", "loop_preonset_active", 120000, "stock 0.7.0 doc 79 seed 5012; loop onset 184k, active here"),
]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_inventory():
    want = None
    for line in open(os.path.join(INV_DIR, "SHA256SUMS")):
        h, name = line.split()
        if name.endswith("inventory.json"):
            want = h
    got = sha256_file(INV)
    if want != got:
        sys.exit("ABORT: inventory.json sha256 %s != SHA256SUMS %s" % (got, want))
    return got


def nearest_ckpts(row, e):
    cps = sorted(int(k) for k in row["checkpoints"])
    below = [c for c in cps if c <= e]
    above = [c for c in cps if c > e]
    out = {}
    for tag, c in (("at_or_before", below[-1] if below else None), ("after", above[0] if above else None)):
        if c is not None:
            v = row["checkpoints"][str(c)]
            out[tag] = dict(checkpoint=c, zlib30k=v["zlib30k"], rep200=v["rep200"], repeating=v["repeating"])
    return out


def build(spec, rows, exact, short):
    out = []
    for fid, rid, cls, end, note in spec:
        r = rows.get(rid) or exact.get(rid)
        if r is None:
            sys.exit("ABORT: %s not in inventory" % rid)
        is_exact = rid in exact
        close = r["natural_close_pos"]
        if end == "close":
            e = close
        elif end == "close-8192":
            e = close - 8192
        else:
            e = int(end)
        if is_exact:
            prompt = r["prompt_len"]
            n_reason = r["n_gen"]
            hist_sha = r["sha256_active_ids"]
            hist_kind = "exact generated ids (active segment, up to the first close)"
            state = "natural_close"
            loop_onset = None
        else:
            prompt = r["prompt_tokens_rendered"]
            n_reason = r["n_reasoning_tokens"]
            hist_sha = r["sha256_reasoning_ids"]
            hist_kind = "re-tokenized text (shared counterfactual fixture, not a replay)"
            state = r["state"]
            loop_onset = r["loop_onset_tok"]
        if short:
            w = e                          # full replay: everything after the prompt is forced
            lead = min(LEAD_IN, e // 4)
        else:
            w = W_LONG
            lead = LEAD_IN
        start = e - w
        assert start >= 0, fid
        if close is not None:
            assert e <= close, fid         # never force past the natural close
        abs_end = prompt + e
        rec = dict(
            fixture=fid, source_id=rid, cls=cls, note=note, doc=r["doc"], seed=r.get("seed"),
            source_engine=r.get("engine") or r.get("arm"), history_ids=hist_kind, history_sha256=hist_sha,
            prompt_tokens=prompt, reasoning_tokens_in_history=n_reason,
            natural_close_index=close, window_start_index=start, window_end_index=e,
            forced_predictions=w + 1, prefix_tokens=prompt + start, abs_window_end=abs_end,
            below_131072=abs_end < ABS_LIMIT,
            forced_token_at_end=("</think> 154842 (history's own next token)" if e == close else "history token e"),
            lead_in=lead,
            primary_positions=dict(end=[e], tail=[max(start + lead, e - TAIL + 1), e],
                                   grid_every=GRID, grid_range=[start + lead, e]),
            state_of_history=state,
            distance_to_close=(close - e) if close is not None else None,
            loop_onset_index=loop_onset,
            loop_onset_after_window=(loop_onset > e) if loop_onset else None,
        )
        if not is_exact:
            rec["retok_drift"] = r["retok_drift"]
            rec["repetition_near_window_end"] = nearest_ckpts(r, e)
            rec["repetition_at_history_end"] = r["end_metrics"]
            ev = [x[0] for x in r.get("strong_events_56k_136k", []) if start + lead <= x[0] <= e and not x[4]]
            rec["nonrepeating_commitment_positions_in_window"] = ev   # positions only (class/letter withheld)
            # 20b class at the window: is the history repeating where the window sits?
            nb = rec["repetition_near_window_end"]
            rep = any(v["repeating"] for v in nb.values())
            rec["class_20b_at_window"] = ("repeating" if rep else
                                          "natural_close" if e == close else
                                          "active (history closes later)" if close is not None else
                                          "active (loop pre-onset)" if state == "loop" else
                                          "active (exhaustion)")
        else:
            rec["class_20b_at_window"] = "natural_close (exact ids; short control)"
            rec["generating_sampler"] = r.get("sampler")
            rec["decode_logprob_of_close_at_close_on_generator"] = r.get("decode_lp_close_at_close")
        out.append(rec)
    return out


def main():
    inv_sha = check_inventory()
    d = json.load(open(INV))
    rows = {r["id"]: r for r in d["text_rows"]}
    exact = {r["id"]: r for r in d["exact_traces"]}
    res = dict(
        note=("Ids, positions, classes and hashes only. Window-level id hashes (prefix ids, forced ids) are "
              "computed at freeze (Stage 0) when the ids are re-tokenized in a CPU container; they are not here."),
        inventory_sha256=inv_sha,
        params=dict(W_LONG=W_LONG, LEAD_IN=LEAD_IN, TAIL=TAIL, GRID=GRID, ABS_LIMIT=ABS_LIMIT, CLOSE_ID=CLOSE_ID),
        batches=dict(
            L=build(STAGE1_L, rows, exact, short=False),
            S=build(STAGE1_S, rows, exact, short=True),
            L2=build(STAGE2_L2, rows, exact, short=False),
            L3=build(STAGE2_L3, rows, exact, short=False),
        ),
    )
    # checks: every window below 131,072; no window past a close; same-history fixtures listed
    bad = [f["fixture"] for b in res["batches"].values() for f in b if not f["below_131072"]]
    if bad:
        sys.exit("ABORT: windows at or beyond 131,072: %s" % bad)
    seen = {}
    for bname, b in res["batches"].items():
        for f in b:
            seen.setdefault(f["source_id"], []).append("%s:%s" % (bname, f["fixture"]))
    res["same_history_across_batches"] = {k: v for k, v in seen.items() if len(v) > 1}
    for bname, b in res["batches"].items():
        ids = [f["source_id"] for f in b]
        assert len(ids) == len(set(ids)), "a history appears twice inside batch %s" % bname
        res.setdefault("batch_totals", {})[bname] = dict(
            n=len(b), prefix_tokens=sum(f["prefix_tokens"] for f in b),
            forced_predictions_max=max(f["forced_predictions"] for f in b),
            forced_predictions_sum=sum(f["forced_predictions"] for f in b),
            rebuild_prompt_tokens=sum(f["prompt_tokens"] + f["window_end_index"] for f in b))
    json.dump(res, open(os.path.join(HERE, "fixtures.json"), "w"), indent=1)
    for bname, b in res["batches"].items():
        print("batch", bname, res["batch_totals"][bname])
        for f in b:
            print("  %-3s %-13s %-22s doc %-4s e=%-7s W=%-6s prefix=%-7s absend=%-7s d2close=%-7s %s" % (
                f["fixture"], f["source_id"], f["cls"], f["doc"], f["window_end_index"], f["forced_predictions"] - 1,
                f["prefix_tokens"], f["abs_window_end"], f["distance_to_close"], f["class_20b_at_window"]))
    print("same history across batches:", res["same_history_across_batches"])


if __name__ == "__main__":
    main()
