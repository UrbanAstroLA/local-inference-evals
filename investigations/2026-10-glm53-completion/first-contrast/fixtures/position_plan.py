#!/usr/bin/env python3
"""First long-context contrast: fixture and position plan (PLAN.md amendments 21 s.5-7, 21g s.6-7). CPU only.

Reads hook-validation/fixtures/first_contrast.json (sha256-checked below) and writes position_plan.json: ids,
lengths, positions and hashes ONLY. No reasoning text, no token ids are read or written here; exact prefix/window
token ids and their sha256 are fixed at the freeze by re-tokenizing in a CPU container (as first_contrast.json's
"ids" rule states), and the freeze asserts the conventions checked below on the ids.

Conventions (from first_contrast.json "rule" and the client's forced() call):
  * prompt occupies absolute positions 0..prompt-1; reasoning index r sits at absolute position prompt + r.
  * window_start_reasoning_index s (8-aligned so P = prompt + s = 0 mod 8); W = W_forced = end - s.
  * the request's prompt is prompt + reasoning[:s] (length P); forcewin_hooks = reasoning[s : end + 1], i.e. the W
    window tokens PLUS the history's next token (</think> 154842 for near-close fixtures), length W + 1.
  * output index o predicts reasoning index s + o (absolute position P + o); o = W is the end position e
    (the prediction of the history's own next token; P(</think>) at the true close for near-close fixtures).
Long-prompt / short-output (LP) construction, 21g s.6: P_L = P + 7168 (= 0 mod 8, so the LP window also starts on a
pool and ring boundary), W_s = W - 7168 (1,025-1,030), forced list = reasoning[s+7168 : end+1]. Same final position.
Common measured positions: o >= 7168 + 512 (LP lead-in) = o in [7680, W], which contains the whole 512-position tail.
"""
import hashlib
import json
import math
import os

ROOT = "<runs>"
FC = f"{ROOT}/hook-validation/fixtures/first_contrast.json"
FC_SHA_EXPECTED = None          # recorded in the output; the freeze pins it in SHA256SUMS
HERE = os.path.dirname(os.path.abspath(__file__))
BLOCK, KPOOL, RING = 4352, 4, 8
LEAD = 512                      # lead-in excluded from the 64-grid (first_contrast.json "designated" rule)
LP_SHIFT = 7168                 # 896 x 8
TAIL = 512
NSEL = 64                       # hook limit on full-selection records per request (FORCEWIN_NSEL)
NDESIG = 1024                   # hook limit on designated exposure positions per request
WMAX = 8448                     # module default; attempt 1/2 ran 5120, which would truncate these windows
REGIME = (65536, 131072)        # B1 tiled-route selection regime changes (amendment 20d item 2)


def sha_list(x):
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()


def main():
    raw = open(FC, "rb").read()
    fc = json.loads(raw)
    out = {"source": FC, "source_sha256": hashlib.sha256(raw).hexdigest(), "conventions": __doc__.split("Conventions")[1],
           "constants": dict(BLOCK=BLOCK, LEAD=LEAD, LP_SHIFT=LP_SHIFT, TAIL=TAIL, NSEL=NSEL, NDESIG=NDESIG, WMAX=WMAX),
           "batch_order": fc["batch_order"], "fixtures": [], "checks": []}
    chk = out["checks"]
    by = {f["fixture"]: f for f in fc["fixtures"]}
    tot = dict(prefix=0, warm_cached=0, remainder=0, lp_prefix=0, lp_warm_cached=0, lp_remainder=0, forced=0, lp_forced=0)
    for name in fc["batch_order"]:
        f = by[name]
        pr = f["proposal_record"]
        prompt, s = int(f["prompt_tokens"]), int(f["window_start_reasoning_index"])
        end, P, W = int(f["window_end_reasoning_index"]), int(f["prefix_len_P"]), int(f["W_forced"])
        pp = pr["primary_positions"]
        e_abs = prompt + end
        # ---- convention checks
        chk.append((name, "P == prompt + s", P == prompt + s))
        chk.append((name, "W == end - s", W == end - s))
        chk.append((name, "P % 8 == 0", P % 8 == 0))
        chk.append((name, "abs_window_end == P + W", int(f["abs_window_end"]) == P + W))
        chk.append((name, "P + W < 131072", P + W < 131072))
        chk.append((name, "forced list W+1 <= WMAX+1", W + 1 <= WMAX + 1))
        chk.append((name, "pp end == window end", pp["end"] == [end]))
        P_L, W_s = P + LP_SHIFT, W - LP_SHIFT
        chk.append((name, "P_L % 8 == 0", P_L % 8 == 0))
        chk.append((name, "LP lead-in + tail fit", W_s >= LEAD + TAIL - 1))
        # ---- designated positions (output indices o = r - s)
        g0 = s + LEAD
        grid = sorted({o for o in range(LEAD, W + 1, 64)} | {W})                           # 64-grid after lead-in + e
        commits = sorted({int(r) - s for r in pr.get("nonrepeating_commitment_positions_in_window", [])
                          if g0 <= int(r) <= end})
        Q = sorted(set(grid) | set(commits) | {W})                                         # primary set
        T = list(range(W - TAIL + 1, W + 1))                                               # 512-position tail
        desig = sorted(set(Q) | set(T))
        common = [o for o in desig if o >= LP_SHIFT + LEAD]                                # LP-comparable positions
        Q_common = [o for o in Q if o >= LP_SHIFT + LEAD]
        pre_tail_grid = [o for o in grid if o < W - TAIL + 1]
        k48 = [pre_tail_grid[round(i * (len(pre_tail_grid) - 1) / 47)] for i in range(48)]
        sel = sorted(set(k48) | set(range(W - 480, W + 1, 32)))[:NSEL]
        sel_lp = [o for o in sel if o >= LP_SHIFT + LEAD]
        chk.append((name, "desig <= NDESIG", len(desig) <= NDESIG))
        chk.append((name, "selrec <= NSEL", len(sel) <= NSEL))
        chk.append((name, "tail inside LP common set", set(T) <= set(common)))
        # ---- cache geometry (warm prefix: unkeyed max_tokens=1 request with the run's salt; full blocks cached)
        warm, rem = (P // BLOCK) * BLOCK, P % BLOCK
        lwarm, lrem = (P_L // BLOCK) * BLOCK, P_L % BLOCK
        blocks_crossed = [b * BLOCK for b in range(P // BLOCK + 1, (P + W) // BLOCK + 1)]
        regime = [x for x in REGIME if P <= x < P + W + 1]
        rec = dict(
            fixture=name, source_id=f["source_id"], doc=f["doc"], seed=f["seed"], source_engine=f["source_engine"],
            cls=f["cls"], class_20b_at_window=f["class_20b_at_window"], history_sha256=f["history_sha256"],
            weights=f["weights"], prompt_tokens=prompt, window_start_reasoning_index=s,
            window_end_reasoning_index=end, P=P, W=W, forced_list_len=W + 1, abs_end_position=e_abs,
            repeating_near_window_end=f["repetition_near_window_end"]["at_or_before"]["repeating"],
            loop_onset_index=pr.get("loop_onset_index"),
            primary=dict(warm_prompt_len=P, warm_cached_expected=warm, remainder_prefill=rem,
                         blocks_crossed_in_window=blocks_crossed, regime_boundaries_in_window=regime,
                         regime_boundary_output_index=[x - P for x in regime]),
            lp=dict(P_L=P_L, W_s=W_s, forced_list_len=W_s + 1, warm_cached_expected=lwarm, remainder_prefill=lrem,
                    output_index_offset=LP_SHIFT,
                    note="LP output index o_L = o - 7168; dumprange [P, P_L + W_s] covers prefill-written [P, P_L)"),
            positions=dict(
                Q_primary=dict(n=len(Q), n_grid=len(grid), n_commit=len(commits), first=Q[0], last=Q[-1],
                               sha256=sha_list(Q), list=Q),
                T_tail=dict(n=len(T), first=T[0], last=T[-1], sha256=sha_list(T)),
                designated=dict(n=len(desig), sha256=sha_list(desig)),
                lp_common=dict(n=len(common), n_Q=len(Q_common), first=common[0], last=common[-1],
                               sha256=sha_list(common)),
                selrec=dict(n=len(sel), sha256=sha_list(sel), list=sel),
                selrec_lp_common=dict(n=len(sel_lp), list=sel_lp)))
        out["fixtures"].append(rec)
        tot["prefix"] += P
        tot["warm_cached"] += warm
        tot["remainder"] += rem
        tot["lp_prefix"] += P_L
        tot["lp_warm_cached"] += lwarm
        tot["lp_remainder"] += lrem
        tot["forced"] += W + 1
        tot["lp_forced"] += W_s + 1
    tot["max_forced_list"] = max(r["forced_list_len"] for r in out["fixtures"])
    tot["rebuild_tokens_R1"] = sum(r["P"] + r["W"] + 1 for r in out["fixtures"])
    tot["rebuild_tokens_R2"] = tot["rebuild_tokens_R1"] + 200 * len(out["fixtures"])
    out["totals"] = tot
    out["checks_pass"] = all(c[2] for c in chk)
    out["checks"] = [list(c) for c in chk]
    json.dump(out, open(os.path.join(HERE, "position_plan.json"), "w"), indent=1)
    print("fixture  P       W     list  rem   P_L     W_s   |Q|  |T|  |desig| |LPcommon| |sel| blocks  regime")
    for r in out["fixtures"]:
        p = r["positions"]
        print("%-6s %7d %5d %5d %5d %7d %5d %4d %4d %6d %8d %6d %-15s %s" % (
            r["fixture"], r["P"], r["W"], r["forced_list_len"], r["primary"]["remainder_prefill"], r["lp"]["P_L"],
            r["lp"]["W_s"], p["Q_primary"]["n"], p["T_tail"]["n"], p["designated"]["n"], p["lp_common"]["n"],
            p["selrec"]["n"], r["primary"]["blocks_crossed_in_window"], r["primary"]["regime_boundaries_in_window"]))
    print("totals", json.dumps(tot))
    print("checks: %d, all pass: %s" % (len(chk), out["checks_pass"]))
    bad = [c for c in chk if not c[2]]
    if bad:
        print("FAILED:", bad)


if __name__ == "__main__":
    main()
