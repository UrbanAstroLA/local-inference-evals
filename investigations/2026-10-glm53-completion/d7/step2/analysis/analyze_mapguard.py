#!/usr/bin/env python3
"""Frozen analysis for the B1 mapping-guard session (PLAN.md amendment 18a, step 2c; copy of the frozen
b0-mapguard analysis with the B1 amendment -- see "B1 AMENDMENT" below and d7/step2/FROZEN.md).

usage: analyze_mapguard.py RUN_DIR  -> RUN_DIR/verdict.json + verdict.txt

Inputs (copied out by run_mapguard.sh): RUN_DIR/mapguard/mg_p*.jsonl (one per worker rank),
RUN_DIR/mapguard/ms_p*.jsonl (EngineCore scheduler side), client.jsonl, probe.jsonl,
server.log.  Frozen BEFORE any GPU data: invariants, coverage minimums, verdict rules.

Verdicts (whole run, and per coverage condition):
  VIOLATION   any invariant counter > 0 in the workload window on either rank, or a
              scheduler-side exclusivity / transmission / exact-position failure (reported
              even if the instrumentation is otherwise invalid; snapshots in records)
  INVALID     the instrumentation did not observe what it claims: guard errors, a rank or the
              scheduler side missing, geometry/config mismatch vs B0, no FULL replays or the
              in-graph checks not executed once per forward per owned layer, client errors
  INCOMPLETE  no violation, instrumentation valid, but a frozen coverage minimum not met
              (named per condition; INCOMPLETE is never reported as CLEAN)
  CLEAN       none of the above
POST-INSPECTION CHANGES (after run-a2, 2026-10-13; each justified in POST_A2_CHANGES below and in
DIAGNOSIS-a2.md; invariants, classes and coverage minimums are unchanged):
  P1 armed forwards = forwards the device executed (in-graph ring entries in the readout window),
     not host step logs: the readout's side-stream copy is unordered with the engine stream, so a
     forward enqueued before the readout may not have run yet (a2: 362 device vs 363 host).
  P2 no client (workload never started): window = first non-warm-up step .. final readout;
     baseline = last readout before it.  Runner warm-up (_warmup_ requests) steps are excluded from
     step-level checks in every case, as the design intended (they precede the baseline).
  P3 step-level checks use only steps the device executed (step <= ring max of the final readout).
  P4 guard module 551faad6... (run-a2) checked index-K READ tables in KV-block units; B0 consumes
     index-page ids (17 per block).  For that module the idx_r_dec/idx_r_pf `rtable` counters are
     void; the stored records are re-checked against EngineCore lists; those read sites are
     INCOMPLETE (not CLEAN) for that run.
  P5 probes absent -> output identity NOT RUN (not UNVERIFIED); completions from EngineCore when
     no client ran (labelled).
  P6 (frozen before any rerun data) window = from the first non-_warmup_ request step, i.e. the
     stack's release warm-up traffic (real B0 serving on the same server, shipped graphs) is part
     of the window whether or not a client ran; baseline = the guard's exact "baseline" readout
     (taken just before that step, after a device sync) if present, else the last readout before.
B1 AMENDMENT (frozen before any B1 data; d7/step2/FROZEN.md):
  B1-1 idx_w_pf violation classes add s_missing, s_mism (straddling-pool bytes) and d7_fallback (an
       unaligned straddling pool written without a ring).  missing_pred / pool_cross stay violations
       (the amended guard raises them only for tokens whose writer had no ring).
  B1-2 Positive control: any pc_mism (an aligned pool whose oracle record differs from the stored
       record) on a rank makes that rank's straddling byte checks INVALID; s_missing / s_mism are then
       reported, not counted as violations.  d7_fallback and every slot-level class stay violations.
  B1-3 Coverage adds C13_d7_straddle_events >= 100 per rank (distinct (step, request, position)
       straddling pools byte-verified on every owned indexer layer of the rank) and
       C14_d7_positive_control >= 10,000 aligned pools per rank.  All other minimums unchanged.
  B1-4 INVALID adds: legacy prefill-hook calls (ring status unknown) in the window; straddling-event log
       overflow; VLLM_KPOOL_SKIP_TAIL_CACHE set in the container.
  B1-5 Window ends at the readout taken after the frozen phases (the last readout before the client's
       phase-2d start record); phase 2d is analysed separately (phase2d.json) and is outside every
       coverage count and the verdict.  Guard counters in the 2d window are reported separately.
  B1-6 Output identity: the four short greedy probes vs B0's attempt-3 record (run-a3/probe.jsonl; equal
       to the s1 values below).
Output identity (short greedy probes vs the s1 B0-check-image record) is reported separately:
identical -> 'guard does not alter outputs on the probed path'; different -> the mapping verdict
stands, labelled instrumentation-unverified (a guard-off control is priced as follow-up).
"""
import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

VIOLATION_COUNTERS = ["mism", "missing", "missing_pred", "pool_cross", "unexp", "inact", "shared",
                      "pos_contig", "pos_bounds", "qsl", "rlen", "rtable", "sel_foreign",
                      "sel_causal", "masked_live", "sel_len", "grouppos"]
D7_BYTE_COUNTERS = ["s_missing", "s_mism"]       # B1-1/B1-2: void if the positive control fails
D7_SLOT_COUNTERS = ["d7_fallback"]               # B1-1: always a violation
D7_OUT = {1: "verified", 2: "indet", 3: "unavail", 4: "missing", 5: "mism", 6: "fallback"}
SLOG_COLS = ["step", "layer", "tag", "position", "outcome", "rank", "row", "slot"]   # guard slog row layout
MLA_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
REF_PROBES = {   # B0 attempt 3 (b0-mapguard/run-a3/probe.jsonl, both rounds) == s1-d1234-a1 values
    "q0": "e705960ca1e45291", "q1": "1854a9629800b9c8", "q2": "3ecf3c5b523df951",
    "q3": "a997a1718d15facc"}
A2_GUARD_SHA = "551faad623901b328ba64f5bdb4fff6f8389114fbd37781321668c97a5107b15"
POST_A2_CHANGES = ["P1 device-executed armed forwards", "P2 no-client window/baseline + warm-up exclusion",
                   "P3 device-executed steps only", "P4 a2 guard index-K read unit fault handling",
                   "P5 probes NOT RUN / EngineCore completions",
                   "P6 window from first non-warm-up step; exact baseline readout"]
REF_ARGS_LOG = ("<runs>/bundle/screen/run-d1234/"
                "server-s1.log")

# ---- frozen coverage minimums (per rank where marked; both ranks must meet them) ----
COVERAGE = {
    "C1_full_replay_steps":   dict(min=300, per_rank=True, what="forwards replayed as FULL graphs with the in-graph checks armed"),
    "C2_mixed_steps":         dict(min=20, per_rank=False, what="steps with decode rows and prefill rows together"),
    "C3_row_reuse":           dict(min=20, per_rank=False, what="runner rows re-occupied by a different request"),
    "C4_prefix_hits":         dict(min=10, per_rank=False, what="new requests admitted with a prefix-cache hit"),
    "C5a_spec_writes":        dict(min=20000, per_rank=True, what="speculative MLA write checks (token x layer)"),
    "C5b_rejected_drafts":    dict(min=2000, per_rank=False, what="draft tokens later rejected (scheduler side)"),
    "C5c_accepted_drafts":    dict(min=2000, per_rank=False, what="draft tokens accepted (scheduler side)"),
    "C6a_decode_le4":         dict(min=60, per_rank=False, what="decode-only steps with <= 4 requests"),
    "C6b_decode_5to8":        dict(min=60, per_rank=False, what="decode-only steps with 5-8 requests"),
    "C6c_route_transitions":  dict(min=6, per_rank=False, what="consecutive decode steps crossing 4|5"),
    "C7_high_mla_w":          dict(min=5000, per_rank=True, what="MLA writes into blocks >= 935 (matched)"),
    "C7_high_idx_w":          dict(min=500, per_rank=True, what="index-K pool writes into blocks >= 935"),
    "C7_high_mla_r":          dict(min=5000, per_rank=True, what="MLA selected reads in blocks >= 935"),
    "C7_high_idx_r":          dict(min=500, per_rank=True, what="index-K page-table entries >= 935 read"),
    "C7_high_tail":           dict(min=50, per_rank=True, what="tail-ring writes into blocks >= 935"),
    "C8_mla_w":               dict(min=200000, per_rank=True, what="MLA write tokens checked"),
    "C8_idx_w_pf":            dict(min=100000, per_rank=True, what="index-K prefill-write tokens checked"),
    "C8_idx_w_dec":           dict(min=20000, per_rank=True, what="index-K decode-write tokens checked"),
    "C8_tail_w_seed":         dict(min=500, per_rank=True, what="tail seed tokens checked"),
    "C8_tail_w_stash":        dict(min=20000, per_rank=True, what="tail stash tokens checked"),
    "C8_idx_r_dec":           dict(min=20000, per_rank=True, what="index-K decode-read rows checked"),
    "C8_idx_r_pf":            dict(min=50, per_rank=True, what="index-K prefill-read rows checked"),
    "C8_mla_r":               dict(min=100000, per_rank=True, what="MLA read tokens checked"),
    "C9_owned_layers":        dict(min=1, per_rank=True, what="every owned MLA layer checked once per forward (exact)"),
    "C10_join":               dict(min=0.95, per_rank=True, what="fraction of worker steps joined to the scheduler record"),
    "C10_p3_rows":            dict(min=0.90, per_rank=True, what="fraction of rows with an exact offline position check"),
    "C11_completions":        dict(min=40, per_rank=False, what="completed requests"),
    "C12_short_ctx":          dict(min=8, per_rank=False, what="completed requests with < 2,176 prompt tokens"),
    # B1-3 (amendment 18a)
    "C13_d7_straddle_events": dict(min=100, per_rank=True, what="distinct straddling pools byte-verified on every owned layer"),
    "C14_d7_positive_control": dict(min=10000, per_rank=True, what="aligned pools recomputed by the oracle (positive control)"),
}


# ------------------------------------------------------------------ rules (unit-tested on CPU)
def owner_rule(spec, mla_layers, rank):
    spec = (spec or "tp").strip()
    if spec.startswith("split:"):
        b = int(spec.split(":", 1)[1])
        return [L for L in mla_layers if int(L >= b) == rank]
    if spec.startswith("map:"):
        m = {int(a): int(b) for a, b in (x.split("=") for x in spec.split(":", 1)[1].split(","))}
        return [L for L in mla_layers if m.get(L) == rank]
    return list(mla_layers)


def rank_ownership(layer_counts, spec, mla_layers, armed_forwards):
    """layer_counts[rank][layer] = armed MLA-write executions in the window; every owned layer
    must equal the rank's armed forwards exactly, and no other layer may write."""
    problems = []
    for rank, counts in layer_counts.items():
        owned = set(owner_rule(spec, mla_layers, rank))
        for L, n in counts.items():
            if n and L not in owned:
                problems.append(("non_owned_layer_writes", rank, L, n))
        for L in owned:
            n = counts.get(L, 0)
            if n != armed_forwards.get(rank, -1):
                problems.append(("owned_layer_count", rank, L, n, armed_forwards.get(rank)))
    return dict(ok=not problems, problems=problems)


def p3_exact(steps, ring, upd, async_mode=True):
    """Exact position join: first position of every row == scheduler num_computed minus the
    drafts rejected in that request's previous (consecutive) step (scheduler.py:1836-1856),
    or == num_computed under synchronous scheduling."""
    last = {}
    checked = mismatch = 0
    examples = []
    prev_fp = {}
    for st in steps:
        s = st.get("sched_step", st["step"])
        p0 = ring.get(st["step"])
        for i, row in enumerate(st["rows"]):
            rid, hi = row[0], row[3]
            exp = hi
            if async_mode and rid in last and last[rid] == s - 1:
                exp = hi - int((upd.get(prev_fp[rid]) or {}).get(rid, [0, 0, 0])[2])
            last[rid] = s
            prev_fp[rid] = st["fp"]
            if p0 is None or i >= len(p0) or p0[i] < 0:
                continue
            checked += 1
            if p0[i] != exp:
                mismatch += 1
                if len(examples) < 10:
                    examples.append(dict(step=st["step"], rid=rid, p0=p0[i], expected=exp, hi=hi))
    return dict(checked=checked, mismatch=mismatch, examples=examples)


def transmission(steps, sched_by_fp):
    joined = mismatch = 0
    ex = []
    for st in steps:
        sc = sched_by_fp.get(st["fp"])
        if sc is None:
            continue
        joined += 1
        if sc["dig"] != st["dig"]:
            mismatch += 1
            if len(ex) < 5:
                ex.append(dict(step=st["step"], fp=st["fp"]))
    return dict(joined=joined, steps=len(steps), mismatch=mismatch, examples=ex)


def recheck_index_reads(records, worker_recs, sched):
    """P4: re-check stored idx_r_* records of the a2 guard against the corrected expectation:
    consumed page at column c == 17 * g0[c // 17] + c % 17, with g0 the request's group-0 list
    from the EngineCore log (KVCacheManager), as of the step's fingerprint."""
    lists, at = {}, {}
    for r in sched:
        if r.get("kind") != "sched":
            continue
        for rid, ch in r.get("changed", {}).items():
            lists[rid] = ch[0]
        at[r["fp"]] = {rid: list(lists.get(rid, [])) for rid in r.get("reqs", {})}
    steps = {r["step"]: r for r in worker_recs if r.get("kind") == "step"}
    out = dict(checked=0, consistent=0, inconsistent=0, unjoined=0, examples=[])
    for x in records:
        if x["site"] not in (5, 6):
            continue
        st = steps.get(x["step"])
        if st is None or st.get("warm") or x["row"] >= len(st["rows"]) or st["fp"] not in at:
            out["unjoined"] += 1
            continue
        rid = st["rows"][x["row"]][0]
        g0 = at[st["fp"]].get(rid, [])
        c = int(x["aux"])
        exp = 17 * g0[c // 17] + c % 17 if c // 17 < len(g0) else None
        out["checked"] += 1
        if exp == x["consumed"]:
            out["consistent"] += 1
        else:
            out["inconsistent"] += 1
            if len(out["examples"]) < 5:
                out["examples"].append(dict(step=x["step"], rid=rid, col=c, consumed=x["consumed"],
                                            expected=exp, blocks=g0[:4]))
    return out


def d7_assess(win, rank):
    """B1-1/B1-2 for one rank's window counters: violations and INVALID items from the D7 counters."""
    pf = win.get("idx_w_pf", {})
    viol, invalid, notes = {}, [], []
    for c in D7_SLOT_COUNTERS:
        if pf.get(c, 0):
            viol[f"r{rank}.idx_w_pf.{c}"] = pf[c]
    pc_fail = pf.get("pc_mism", 0) > 0
    if pc_fail:
        invalid.append(f"rank {rank}: positive control failed (pc_mism {pf['pc_mism']} of "
                       f"{pf.get('pc_checked', 0)} aligned pools): straddling byte checks INVALID")
        for c in D7_BYTE_COUNTERS:
            if pf.get(c, 0):
                notes.append(f"r{rank}.idx_w_pf.{c} = {pf[c]} reported, not counted (byte oracle not validated)")
    else:
        for c in D7_BYTE_COUNTERS:
            if pf.get(c, 0):
                viol[f"r{rank}.idx_w_pf.{c}"] = pf[c]
    if pf.get("legacy", 0):
        invalid.append(f"rank {rank}: {pf['legacy']} prefill-hook calls without the writer's ring status")
    if pf.get("s_unavail", 0):
        notes.append(f"rank {rank}: {pf['s_unavail']} straddling pools without an oracle (shadow lacked a member)")
    return dict(violations=viol, invalid=invalid, notes=notes, pc_failed=pc_fail)


def d7_events(rows, owned_layers, step_lo, step_hi):
    """B1-3: distinct straddling events (step, tag, position) in (step_lo, step_hi]; exercised = every
    owned layer logged it, all as verified/indeterminate, at least one verified."""
    ev = defaultdict(dict)
    for r in rows:
        if step_lo < r["step"] <= step_hi:
            ev[(r["step"], r["tag"], r["position"])].setdefault(r["layer"], set()).add(r["outcome"])
    owned = set(owned_layers)
    exercised = 0
    by_outcome = defaultdict(int)
    for key, lay in ev.items():
        outs = set().union(*lay.values())
        for o in outs:
            by_outcome[D7_OUT.get(o, str(o))] += 1
        if set(lay) >= owned and outs <= {1, 2} and any(1 in lay[L] for L in owned):
            exercised += 1
    return dict(events=len(ev), exercised=exercised, by_outcome=dict(by_outcome))


def verdict(violations, invalid, coverage):
    incomplete = []
    for k, spec in COVERAGE.items():
        v = coverage.get(k)
        if v is None or (v < spec["min"]):
            incomplete.append(k)
    if violations:
        v = "VIOLATION"
    elif invalid:
        v = "INVALID"
    elif incomplete:
        v = "INCOMPLETE"
    else:
        v = "CLEAN"
    return dict(verdict=v, violations=violations, invalid=invalid, incomplete=incomplete,
                per_condition={k: ("CLEAN" if k not in incomplete else "INCOMPLETE")
                               for k in COVERAGE})


# ------------------------------------------------------------------ run analysis
def _jl(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return out


def analyze(run):
    mgd = os.path.join(run, "mapguard")
    ranks = {}
    sched = []
    byrank = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(mgd, "mg_p*.jsonl"))):
        for r in _jl(p):
            r["_file"] = os.path.basename(p)
            byrank[r.get("rank", -1)].append(r)
    for rk, recs in byrank.items():
        if rk is None or int(rk) < 0:
            continue                     # records written before the rank was known ("installed")
        inits = [r for r in recs if r.get("kind") == "init"]
        if not inits:
            continue
        recs.sort(key=lambda r: r.get("t", 0))
        ranks[int(rk)] = dict(file=sorted({r["_file"] for r in recs}), recs=recs, init=inits[-1],
                              n_inits=len(inits))
    for p in sorted(glob.glob(os.path.join(mgd, "ms_p*.jsonl"))):
        sched += _jl(p)
    client = _jl(os.path.join(run, "client.jsonl")) if os.path.exists(os.path.join(run, "client.jsonl")) else []
    t_start = min((r["t0"] for r in client if r.get("kind") == "request"), default=None)
    # B1-5: the frozen window ends before phase 2d (client record "phase2d_start")
    t_cut = min((r["t"] for r in client if r.get("kind") == "phase2d_start"), default=None)
    t_end = t_cut if t_cut is not None else float("inf")
    sched_all = None
    if t_cut is not None:
        sched_all = sched
        sched = [r for r in sched if r.get("t", 0) <= t_cut]
    invalid, viol, notes = [], {}, []
    void_sites = []
    guard_sha = None
    cfe = os.path.join(run, "container-files-env.txt")
    if os.path.exists(cfe):
        for line in open(cfe):
            if line.strip().endswith("mapguard_hooks.py"):
                guard_sha = line.split()[0]
    if guard_sha == A2_GUARD_SHA:
        void_sites = ["idx_r_dec", "idx_r_pf"]
        notes.append("P4: guard 551faad6 index-K read tables checked in block units -> idx_r_* rtable void")
    if ranks:                          # P2/P6: window starts at the first non-warm-up step
        firsts = [min((r["t"] for r in d["recs"] if r.get("kind") == "step" and not r.get("warm")),
                      default=None) for d in ranks.values()]
        firsts = [f for f in firsts if f is not None]
        if firsts:
            t_start = min(firsts) - 1e-3
            notes.append("P2/P6: window starts at the first non-warm-up step "
                         "(the stack's release warm-up traffic is included)")
    cov = {}
    if sorted(ranks) != [0, 1]:
        invalid.append(f"worker ranks reporting: {sorted(ranks)} (need [0, 1])")
    if not any(r.get("kind") == "installed_scheduler" for r in sched):
        invalid.append("scheduler-side hook not installed (no ms_p*.jsonl)")
    sched_by_fp = {r["fp"]: r for r in sched if r.get("kind") == "sched"}
    upd = {r["fp"]: r["rej"] for r in sched if r.get("kind") == "upd"}
    per_rank = {}
    phase2d_viol, phase2d_d7 = {}, {}
    for rank, d in ranks.items():
        recs, init = d["recs"], d["init"]
        errs = [r for r in recs if r.get("kind") == "guard_error"]
        if errs:
            invalid.append(f"rank {rank}: {len(errs)} guard errors (first: {errs[0].get('where')} {errs[0].get('err', '')[:120]})")
        if init.get("issues"):
            invalid.append(f"rank {rank}: geometry/config issues {init['issues']}")
        g = init["geo"]
        if (g["bs"], g["kbs"], g["kpool"], g["ring"], g["nb"]) != (4352, 64, 4, 8, 1275):
            invalid.append(f"rank {rank}: geometry {g} differs from B0 (4352/64/4/8/1275)")
        if not init.get("async_scheduling"):
            notes.append(f"rank {rank}: async_scheduling reported {init.get('async_scheduling')} (P3 uses sync rule)")
        if init.get("adaptive_verification"):
            invalid.append(f"rank {rank}: adaptive verification on (row split not scheduler-exact)")
        if sorted(init.get("owned_layers_impl", [])) != sorted(init.get("owned_layers_rule", [])):
            viol[f"r{rank}.layer_ownership_model"] = 1
        polls_all = [r for r in recs if r.get("kind") == "poll"]
        polls = [r for r in polls_all if r["t"] <= t_end]          # B1-5
        if not polls:
            invalid.append(f"rank {rank}: no guard readout")
            continue
        final = polls[-1]
        base = None
        exact = [p for p in polls if p.get("reason") == "baseline"]
        if exact:
            base = exact[0]
        elif t_start is not None:
            pre = [p for p in polls if p["t"] < t_start]
            base = pre[-1] if pre else None
        if base is None:
            invalid.append(f"rank {rank}: no pre-workload baseline readout")
            base = dict(counters={k: {c: 0 for c in v} for k, v in final["counters"].items()}, layers={})
        win = {s: {c: final["counters"][s][c] - base["counters"].get(s, {}).get(c, 0)
                   for c in final["counters"][s]} for s in final["counters"]}
        for s, cs in win.items():
            for c in VIOLATION_COUNTERS:
                if cs.get(c, 0):
                    if s in void_sites and c == "rtable":
                        notes.append(f"r{rank}.{s}.rtable = {cs[c]} void (P4)")
                        continue
                    viol[f"r{rank}.{s}.{c}"] = cs[c]
        warm_v = {f"{s}.{c}": base["counters"][s][c] for s in base["counters"] for c in VIOLATION_COUNTERS
                  if base["counters"][s].get(c, 0)} if base.get("counters") else {}
        if warm_v:
            notes.append(f"rank {rank}: before the workload (runner warm-up incl.): {warm_v}")
        records = [x for p in polls for x in p.get("records", [])]
        ring = {}
        for p in polls:
            for e in p.get("ring", []):
                ring[e[0]] = e[1:]
        ring_max_final = max(ring, default=0)          # all readouts up to the final one
        ring_max_base = max((e[0] for p in polls if p["t"] <= base.get("t", -1) for e in p.get("ring", [])),
                            default=0)
        steps = [r for r in recs if r.get("kind") == "step" and (t_start is None or r["t"] >= t_start)
                 and r["t"] <= final["t"] and r["step"] <= ring_max_final and not r.get("warm")]
        # in-graph execution: every owned layer checked once per armed forward (exact)
        lay_f = final.get("layers", {})
        lay_b = base.get("layers", {})
        # P1: forwards the device executed in the window (ring entries are written in-graph)
        armed_fw = len([k for k in ring if ring_max_base < k <= ring_max_final])
        host_fw = len([s for s in recs if s.get("kind") == "step" and s.get("ntok", 0) > 0
                       and base.get("t", -1) < s["t"] <= final["t"]])
        if host_fw != armed_fw:
            notes.append(f"rank {rank}: host-logged forwards {host_fw} vs device-executed {armed_fw} "
                         "(readout unordered with the engine stream; P1)")
        counts = {}
        for L in set(lay_f) | set(lay_b):
            a = lay_f.get(L, {}).get("mla_w", [0, 0])[1] - lay_b.get(L, {}).get("mla_w", [0, 0])[1]
            counts[int(L)] = a
        own = rank_ownership({rank: counts}, init.get("owners_env") or "split:25", MLA_LAYERS,
                             {rank: armed_fw})
        if not own["ok"]:
            non_owned = [p for p in own["problems"] if p[0] == "non_owned_layer_writes"]
            if non_owned:
                viol[f"r{rank}.rank_ownership"] = len(non_owned)
            short = [p for p in own["problems"] if p[0] == "owned_layer_count"]
            if short:
                invalid.append(f"rank {rank}: in-graph check executions != armed forwards {short[:3]}")
        cov.setdefault("C9_owned_layers", []).append(int(own["ok"]))
        full = sum(1 for s in steps if s.get("cg") == "FULL")
        cov.setdefault("C1_full_replay_steps", []).append(full)
        if full == 0:
            invalid.append(f"rank {rank}: no FULL-graph replay observed (eager fallback?)")
        cov.setdefault("C5a_spec_writes", []).append(win["mla_w"]["spec"])
        cov.setdefault("C7_high_mla_w", []).append(win["mla_w"]["high"])
        cov.setdefault("C7_high_idx_w", []).append(win["idx_w_pf"]["high"] + win["idx_w_dec"]["high"])
        cov.setdefault("C7_high_mla_r", []).append(win["mla_r"]["high"])
        cov.setdefault("C7_high_idx_r", []).append(win["idx_r_dec"]["high"] + win["idx_r_pf"]["high"])
        cov.setdefault("C7_high_tail", []).append(win["tail_w_seed"]["high"] + win["tail_w_stash"]["high"])
        for s in ("mla_w", "idx_w_pf", "idx_w_dec", "tail_w_seed", "tail_w_stash", "idx_r_dec",
                  "idx_r_pf", "mla_r"):
            cov.setdefault(f"C8_{s}", []).append(0 if s in void_sites else win[s]["checked"])
        if void_sites:
            cov["C7_high_idx_r"][-1] = 0
        tr = transmission(steps, sched_by_fp)
        if tr["mismatch"]:
            viol[f"r{rank}.transmission"] = tr["mismatch"]
        cov.setdefault("C10_join", []).append(tr["joined"] / max(tr["steps"], 1))
        p3 = p3_exact(steps, ring, upd, async_mode=bool(init.get("async_scheduling", True)))
        if p3["mismatch"]:
            viol[f"r{rank}.position_exact"] = p3["mismatch"]
        nrows = sum(len(s["rows"]) for s in steps)
        cov.setdefault("C10_p3_rows", []).append(p3["checked"] / max(nrows, 1))
        issues = [i for s in steps for i in s.get("issues", [])]
        if issues:
            viol[f"r{rank}.host_attribution"] = len(issues)
        recheck = recheck_index_reads(records, recs, sched) if void_sites else None
        # ---- B1 amendment (B1-1..B1-4)
        d7 = d7_assess(win, rank)
        viol.update(d7["violations"])
        invalid += d7["invalid"]
        notes += d7["notes"]
        owned_rule = init.get("owned_layers_rule") or owner_rule(init.get("owners_env") or "split:25", MLA_LAYERS, rank)
        slog = [dict(zip(SLOG_COLS, x)) if isinstance(x, list) else x for p in polls for x in p.get("slog", [])]
        d7ev = d7_events(slog, owned_rule, ring_max_base, ring_max_final)
        cov.setdefault("C13_d7_straddle_events", []).append(d7ev["exercised"])
        cov.setdefault("C14_d7_positive_control", []).append(win.get("idx_w_pf", {}).get("pc_checked", 0))
        scap = (init.get("b1_amendment") or {}).get("scap", 65536)
        if final.get("n_slog", 0) > scap:
            invalid.append(f"rank {rank}: straddling-event log overflow ({final['n_slog']} > {scap})")
        if "b1_amendment" not in init:
            invalid.append(f"rank {rank}: guard module without the B1 amendment")
        elif init["b1_amendment"].get("skip_tail_env") == "1":
            invalid.append(f"rank {rank}: VLLM_KPOOL_SKIP_TAIL_CACHE=1 (writer has no ring)")
        bsnap = [x for p in polls for x in p.get("bsnap", [])]
        win2d = None
        if t_cut is not None and polls_all and polls_all[-1]["t"] > final["t"]:
            last = polls_all[-1]
            win2d = {s: {c: last["counters"][s][c] - final["counters"][s].get(c, 0) for c in last["counters"][s]}
                     for s in last["counters"]}
            v2 = {f"{s}.{c}": cs[c] for s, cs in win2d.items() for c in VIOLATION_COUNTERS + D7_BYTE_COUNTERS
                  + D7_SLOT_COUNTERS if cs.get(c, 0)}
            phase2d_viol[rank] = v2
            phase2d_d7[rank] = {c: win2d["idx_w_pf"].get(c, 0) for c in
                                ("s_events", "s_verified", "s_indet", "s_unavail", "s_missing", "s_mism",
                                 "pc_checked", "pc_mism", "d7_fallback")}
        per_rank[rank] = dict(d7=dict(events=d7ev, assess=d7, counters={c: win.get("idx_w_pf", {}).get(c, 0) for c in
                                                                         ("s_events", "s_verified", "s_indet", "s_unavail",
                                                                          "s_missing", "s_mism", "pc_checked", "pc_mism",
                                                                          "d7_fallback", "legacy")},
                                      byte_snapshots=bsnap[:20]),
                              read_recheck=recheck, window=win, full_steps=full, armed_forwards=armed_fw, ownership=own,
                              transmission=tr, p3=p3, records=records[:200], n_records=len(records),
                              host_issues=issues[:20], steps=len(steps))
    # scheduler-side exclusivity (block held by two groups at once)
    ex = [r["excl"] for r in sched if r.get("kind") == "sched" and "excl" in r]   # B1-5: up to the cut
    multi = sum(e.get("multi_group", 0) for e in ex)
    if multi:
        viol["scheduler.multi_group_holds"] = multi
    # step-level coverage from rank 0's step log (identical schedule on both ranks)
    if 0 in ranks:
        st0 = [r for r in ranks[0]["recs"] if r.get("kind") == "step" and (t_start is None or r["t"] >= t_start)
               and not r.get("warm") and r["t"] <= min(t_end, max(p["t"] for p in ranks[0]["recs"] if p.get("kind") == "poll"))]
        mixed = le4 = m58 = trans = reuse = 0
        prev = None
        for s in st0:
            rows = s["rows"]
            dec = [r for r in rows if r[5] > 0]
            pre = [r for r in rows if r[5] == 0 and r[4] > 1]
            if dec and pre:
                mixed += 1
            if dec and not pre:
                n = len(rows)
                if n <= 4:
                    le4 += 1
                elif n <= 8:
                    m58 += 1
                cls = n <= 4
                if prev is not None and prev != cls:
                    trans += 1
                prev = cls
            reuse += sum(r[-1] for r in rows)
        hits = sum(1 for s in st0 for e in s.get("ev", []) if e[0] == "new" and e[3] > 0)
        cov.update(C2_mixed_steps=mixed, C3_row_reuse=reuse, C4_prefix_hits=hits,
                   C6a_decode_le4=le4, C6b_decode_5to8=m58, C6c_route_transitions=trans)
    rej = acc = 0
    for r in sched:
        if r.get("kind") == "upd":
            for v in r["rej"].values():
                acc += v[1]
                rej += v[2]
    cov.update(C5b_rejected_drafts=rej, C5c_accepted_drafts=acc)
    reqs = [r for r in client if r.get("kind") == "request" and not str(r.get("phase", "")).startswith("2d")]
    errs = [r for r in reqs if r.get("error")]
    if errs:
        invalid.append(f"client: {len(errs)} request errors")
    cov["C11_completions"] = sum(1 for r in reqs if r.get("finish") in ("stop", "length"))
    cov["C12_short_ctx"] = sum(1 for r in reqs if (r.get("prompt_tokens") or 1e9) < 2176
                               and r.get("finish") in ("stop", "length"))
    if 0 in ranks:                   # P5/P6: EngineCore finished ids in the window (labelled)
        plen = {e[1]: e[4] for r in ranks[0]["recs"] if r.get("kind") == "step" and not r.get("warm")
                for e in r.get("ev", []) if e[0] == "new"}
        fin = {f for r in sched if r.get("kind") == "sched" and (t_start is None or r["t"] >= t_start)
               for f in r.get("finished", [])}
        cov["C11_completions"] = sum(1 for f in fin if f in plen)
        cov["C12_short_ctx"] = sum(1 for f in fin if plen.get(f, 1e9) < 2176)
        notes.append("P5/P6: completions from EngineCore finished ids in the window")
    # per-rank conditions: both ranks must meet the minimum -> take the min over ranks
    covd = {}
    for k, v in cov.items():
        covd[k] = min(v) if isinstance(v, list) and v else (v if not isinstance(v, list) else None)
    # configuration identity vs B0's screen (non-default args line) and FULL graph capture
    cfg = {}
    sl = os.path.join(run, "server.log")
    if os.path.exists(sl):
        txt = open(sl, errors="replace").read()
        cfg["full_capture"] = "Capturing CUDA graphs (FULL)" in txt
        if not cfg["full_capture"]:
            invalid.append("server.log: no FULL CUDA graph capture")

        def nda(t):
            i = t.find("non-default args: ")
            return t[i + 18:t.find("\n", i)] if i >= 0 else None
        cur = nda(txt)
        ref = nda(open(REF_ARGS_LOG, errors="replace").read()) if os.path.exists(REF_ARGS_LOG) else None
        cfg["non_default_args_equal_b0_screen"] = (cur == ref) if ref else None
        if ref and cur != ref:
            invalid.append("non-default args differ from the B0 screen")
    # output identity on short greedy probes
    probes = _jl(os.path.join(run, "probe.jsonl")) if os.path.exists(os.path.join(run, "probe.jsonl")) else []
    pid = {}
    for p in probes:
        pid.setdefault(p["id"], set()).add(p["sha256"][:16])
    ident = {k: (sorted(pid.get(k, [])) == [v]) for k, v in REF_PROBES.items()}
    out = verdict(viol, invalid, covd)
    if sched_all is not None:
        ex2 = sum(r["excl"].get("multi_group", 0) for r in sched_all
                  if r.get("kind") == "sched" and "excl" in r and r.get("t", 0) > t_cut)
        if ex2:
            phase2d_viol["scheduler"] = {"multi_group_holds": ex2}
    out["phase2d_window"] = dict(cut=t_cut, violations=phase2d_viol, d7=phase2d_d7,
                                 note="outside every coverage count and the verdict (B1-5); reported separately")
    if any(phase2d_viol.values()):
        notes.append(f"PHASE 2D WINDOW VIOLATIONS (outside the verdict, reported): {phase2d_viol}")
    out.update(coverage=covd, coverage_min={k: v["min"] for k, v in COVERAGE.items()},
               config=cfg, probes_identical_to_s1=ident,
               instrumentation_output_check=("NOT RUN" if not probes else
                                             "PASS" if all(ident.values()) else "UNVERIFIED"),
               post_inspection_changes=POST_A2_CHANGES, guard_module_sha256=guard_sha,
               notes=notes, per_rank={r: {k: v for k, v in d.items() if k != "records"} for r, d in per_rank.items()},
               first_records={r: d["records"][:30] for r, d in per_rank.items()})
    return out


DOC127_TOKENS = 2795


def _server_param(txt, name):
    import re
    m = re.search(r"['\"]?" + name + r"['\"]?\s*[:=]\s*(\d+)", txt or "")
    return int(m.group(1)) if m else None


def phase2d(run):
    """Phase 2d (outside the verdict): doc-127 first-chunk arithmetic under measured decode concurrency.
    Per case: requested D, ACTUAL decode rows scheduled with doc 127's first chunk (EngineCore record),
    other prefill rows, budget and draft-slot reservation (server.log), chunk lengths and start, and the
    predictions budget - sum(other n_i + K) - K and 2045 - 7*D_actual.  Token counts and offsets only."""
    cpath = os.path.join(run, "client.jsonl")
    client = _jl(cpath) if os.path.exists(cpath) else []
    cases = [r for r in client if r.get("kind") == "phase2d_case"]
    out = dict(ran=bool(cases), cases=[], skipped=[r for r in client if r.get("kind") == "skipped"
                                                    and "2d" in str(r.get("what", ""))])
    if not cases:
        return out
    mgd = os.path.join(run, "mapguard")
    sched = [r for p in sorted(glob.glob(os.path.join(mgd, "ms_p*.jsonl"))) for r in _jl(p) if r.get("kind") == "sched"]
    sched.sort(key=lambda r: r.get("seq", 0))
    news = []
    for p in sorted(glob.glob(os.path.join(mgd, "mg_p*.jsonl"))):
        for r in _jl(p):
            if r.get("kind") == "step" and r.get("rank") == 0:
                for e in r.get("ev", []):
                    if e[0] == "new":
                        news.append(dict(t=r["t"], rid=e[1], prompt_len=e[4], nc=e[3]))
    sl = os.path.join(run, "server.log")
    txt = open(sl, errors="replace").read() if os.path.exists(sl) else ""
    budget = _server_param(txt, "max_num_batched_tokens")
    K = _server_param(txt, "num_speculative_tokens")
    out.update(budget=budget, draft_slots_K=K, source="server.log non-default args / speculative config")
    used = set()
    for c in cases:
        cand = [n for n in news if n["prompt_len"] == DOC127_TOKENS and n["rid"] not in used
                and c["t_send"] - 1.0 <= n["t"] <= c["t_done"] + 5.0]
        row = dict(case=c["case"], requested_D=c["requested_D"], bg_in_decode_at_send=c.get("bg_in_decode"),
                   prompt_tokens_server=c.get("prompt_tokens"), prompt_tokens_ok=c.get("prompt_tokens") == DOC127_TOKENS,
                   seeds=c.get("seeds"), error=c.get("error"))
        if not cand:
            row["status"] = "doc-127 request not found in the guard's step log"
            out["cases"].append(row)
            continue
        rid = cand[0]["rid"]
        used.add(rid)
        recs = [r for r in sched if rid in r.get("reqs", {})]
        if not recs:
            row["status"] = "no EngineCore schedule record for the request"
            out["cases"].append(row)
            continue
        first = recs[0]
        others = {k: v for k, v in first["reqs"].items() if k != rid}
        dec = {k: v for k, v in others.items() if v[2] > 0}
        pre = {k: v for k, v in others.items() if v[2] == 0}
        n1 = first["reqs"][rid][1]
        row.update(status="ok", sched_seq=first["seq"], actual_D=len(dec), other_prefill_rows=len(pre),
                   other_rows_tokens=sorted(v[1] for v in others.values()),
                   chunk1_start=first["reqs"][rid][0] - n1, chunk1_len=n1,
                   num_computed_after_first=first["reqs"][rid][0],
                   chunk2_start=(recs[1]["reqs"][rid][0] - recs[1]["reqs"][rid][1]) if len(recs) > 1 else None,
                   chunk2_len=recs[1]["reqs"][rid][1] if len(recs) > 1 else None)
        if budget is not None and K is not None:
            row["pred_budget_formula"] = budget - sum(v[1] + K for v in others.values()) - K
            row["formula_matches"] = row["pred_budget_formula"] == n1 or (n1 == DOC127_TOKENS and row["pred_budget_formula"] >= n1)
        row["pred_2045_minus_7D"] = 2045 - 7 * len(dec)
        row["arith_matches"] = row["pred_2045_minus_7D"] == n1 if not pre else None
        row["requested_eq_actual_D"] = len(dec) == c["requested_D"]
        row["split_residue"] = n1 % 4 if n1 < DOC127_TOKENS else None
        row["split_pool_aligned"] = (n1 % 4 == 0) if n1 < DOC127_TOKENS else None
        out["cases"].append(row)
    return out


def main():
    run = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else ""      # e.g. "-postA2" for a re-analysis
    res = analyze(run)
    p2d = phase2d(run)
    json.dump(p2d, open(os.path.join(run, f"phase2d{tag}.json"), "w"), indent=1, default=str)
    json.dump(res, open(os.path.join(run, f"verdict{tag}.json"), "w"), indent=1, default=str)
    lines = [f"VERDICT: {res['verdict']}", f"invalid: {res['invalid']}",
             f"violations: {res['violations']}", f"incomplete: {res['incomplete']}",
             f"output identity (short greedy vs s1): {res['instrumentation_output_check']} {res['probes_identical_to_s1']}"]
    for k, v in res["coverage"].items():
        lines.append(f"  {k:24s} {v} (min {COVERAGE.get(k, {}).get('min')})")
    lines += [f"note: {n}" for n in res["notes"]]
    for r, d in res.get("per_rank", {}).items():
        lines.append(f"D7 rank {r}: {d.get('d7', {}).get('counters')} events {d.get('d7', {}).get('events')}")
    lines.append(f"phase 2d window (outside the verdict): {res.get('phase2d_window')}")
    lines.append(f"phase 2d: ran={p2d['ran']} budget={p2d.get('budget')} K={p2d.get('draft_slots_K')}")
    for c in p2d.get("cases", []):
        lines.append("  2d " + json.dumps({k: c.get(k) for k in (
            "case", "requested_D", "actual_D", "other_prefill_rows", "chunk1_len", "chunk2_start",
            "pred_budget_formula", "pred_2045_minus_7D", "arith_matches", "requested_eq_actual_D",
            "split_residue", "prompt_tokens_ok", "status")}))
    open(os.path.join(run, f"verdict{tag}.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
