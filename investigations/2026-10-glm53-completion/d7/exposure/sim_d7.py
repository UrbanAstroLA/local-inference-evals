#!/usr/bin/env python3
"""D7 exposure accounting (PLAN.md amendment 18), step 1b: CORRECTED scheduler simulator, checked against guard attempt 3.

CPU only, standard library only. Derived from b0-mapguard/reach/sim_reach_mapguard.py (frozen original, NOT edited; this
is a separate re-implementation). The frozen original drove B0's real KVCacheManager for block ids; D7 exposure depends
only on chunk boundaries, which are set by the scheduler's budget arithmetic, so this simulator re-implements that
arithmetic line by line from B0's scheduler.py (byte-identical to stock 0.9.1's; 0.7.0's differs only in one eagle
back-off line, which cannot move a chunk boundary here, see mamba_clip):

  token_budget = max_num_scheduled_tokens (2048); input_budget = max_num_batched_tokens (2048)
  draft_slots K = speculative_config.max_num_new_slots_for_drafting = num_speculative_tokens for DFlash (3; 0.7.0: 5)
  RUNNING requests first, in admission order (scheduler.py 517-700):
      stop when input_budget <= K
      n = num_tokens_with_spec + placeholders - computed  (decode row: 1 + K; prefill: remaining prompt)
      n = min(n, token_budget, input_budget - K); Mamba 'align' clip (_mamba_block_aligned_split, 378-460)
      token_budget -= n; input_budget -= n + K
  then WAITING requests, FIFO (736-1116): stop when input_budget <= K or len(running) >= max_num_seqs (16);
      start = prefix hit; n = min(prompt - hit, token_budget, input_budget - K); Mamba clip; n == 0 -> stop
  Mamba 'align' clip with block 4,352: a chunk never runs past a 4,352 boundary that lies strictly inside it
      (partial-tail stops are off: B0 enable_partial_hash_hits is False -- measured, see REPORT; 0.7.0 runs DCP2,
      for which the coordinator disables them)
  exposure event: a prefill chunk whose start % 4 != 0 and n > 1 + K (rows with n <= 1 + K take the indexer's decode
      path: decode_threshold = next_n = 1 + K, indexer.py 737-738; one such row in attempt 3 was not flagged by the guard)

Modes
  replay    per measured step: given the measured rows ahead of each row, recompute its n; checks the arithmetic for
            every one of the 2,482 scheduler steps of attempt 3 (MEASURED inputs, deterministic check)
  closed    closed-loop reconstruction of attempt 3: exogenous inputs only -- step start times (measured grid), arrival
            time bound and arrival order, prompt length, prefix hit, number of decode steps per request; chunking,
            admission step and co-scheduling are simulated. Agreement reported per event.
  ablate    closed loop with the frozen original simulator's assumptions switched on one at a time (explains 73 vs 217)
  predict   workload-only prediction (no measured step grid): step-time model fitted on attempt 3, client plan,
            mean accepted tokens per decode step; ESTIMATE, sensitivity over acceptance and step-time scale
"""
import bisect
import collections
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BLOCK = 4352
POOL = 4


# ------------------------------------------------------------------ engine model
class Req:
    __slots__ = ("rid", "L", "hit", "arrival", "order", "decode_steps", "computed", "in_decode", "decode_left", "done",
                 "chunks", "admit_step", "phase", "meta")

    def __init__(self, rid, L, hit=0, arrival=0.0, order=0, decode_steps=0, phase="", meta=None):
        self.rid, self.L, self.hit, self.arrival, self.order = rid, L, hit, arrival, order
        self.decode_steps, self.phase, self.meta = decode_steps, phase, meta or {}
        self.computed, self.in_decode, self.decode_left, self.done = 0, False, 0, False
        self.chunks, self.admit_step = [], None


class Rules:
    """Scheduler rules. Defaults = B0 / stock 0.9.1 (DFlash2 x3)."""
    def __init__(self, K=3, max_batched=2048, max_sched=2048, max_running=16, reserve_draft_slots=True,
                 decode_tokens=None, mamba_align=True, decode_first=False, pool=POOL, eagle_backoff=True):
        self.K = K
        self.max_batched, self.max_sched, self.max_running = max_batched, max_sched, max_running
        self.reserve = K if reserve_draft_slots else 0
        self.decode_tokens = (1 + K) if decode_tokens is None else decode_tokens
        self.mamba_align, self.decode_first, self.pool, self.eagle_backoff = mamba_align, decode_first, pool, eagle_backoff
        self.decode_threshold = 1 + K

    def as_dict(self):
        return dict(K=self.K, max_batched=self.max_batched, reserve_per_request=self.reserve, decode_tokens=self.decode_tokens,
                    mamba_align=self.mamba_align, decode_first=self.decode_first, pool=self.pool)


def mamba_clip(start, n, L, rules):
    """_mamba_block_aligned_split for a prompt prefill (num_tokens == prompt length), partial-tail stops off."""
    if not rules.mamba_align or start >= L:
        return n
    end = start + n
    if end < L:
        aligned_end = end // BLOCK * BLOCK
        if aligned_end > start:          # block_size (4352) > max prefill tokens (2048): only when crossing a boundary
            end = aligned_end
    lcp = L - L % BLOCK
    if rules.eagle_backoff:
        lcp = max(lcp - BLOCK, 0)
    stops = ((start // BLOCK + 1) * BLOCK if start % BLOCK else 0, lcp)
    inside = [s for s in stops if start < s < end]
    if inside:
        end = min(inside)
    return max(end - start, 0)


class Engine:
    def __init__(self, rules):
        self.r = rules
        self.running, self.waiting = [], []
        self.step = 0
        self.log = []          # per step: (step, t, [(rid, start, n, kind)])

    def add(self, req):
        self.waiting.append(req)

    def schedule(self, t):
        R = self.r
        K = R.reserve
        token_budget, input_budget = R.max_sched, R.max_batched
        rows = []
        order = self.running
        if R.decode_first:
            order = sorted(self.running, key=lambda q: not q.in_decode)
        for q in order:
            if token_budget <= 0 or input_budget <= K:
                break
            if q.in_decode:
                n = R.decode_tokens
            else:
                n = q.L - q.computed
            n = min(n, token_budget, input_budget - K)
            if not q.in_decode:
                n = mamba_clip(q.computed, n, q.L, R)
            if n <= 0:
                continue
            rows.append((q, q.computed, n, "decode" if q.in_decode else "prefill"))
            token_budget -= n
            input_budget -= n + K
        self.waiting.sort(key=lambda q: (q.arrival, q.order))
        while self.waiting and token_budget > 0:
            if input_budget <= K or len(self.running) >= R.max_running:
                break
            q = self.waiting[0]
            if q.arrival > t:
                break
            start = q.hit
            n = min(q.L - start, token_budget, input_budget - K)
            n = mamba_clip(start, n, q.L, R)
            if n <= 0:
                break
            self.waiting.pop(0)
            q.computed = start
            q.admit_step = self.step
            self.running.append(q)
            rows.append((q, start, n, "prefill"))
            token_budget -= n
            input_budget -= n + K
        # ---- apply
        out = []
        for q, start, n, kind in rows:
            out.append((q.rid, start, n, kind))
            if kind == "prefill":
                q.chunks.append((self.step, start, n))
                q.computed = start + n
                if q.computed >= q.L:
                    q.in_decode = True
                    q.decode_left = q.decode_steps
                    if q.decode_left <= 0:
                        q.done = True
            else:
                q.decode_left -= 1
                if q.decode_left <= 0:
                    q.done = True
        self.running = [q for q in self.running if not q.done]
        self.log.append((self.step, t, out))
        self.step += 1
        return out

    def idle(self):
        return not self.running and not self.waiting


def events_of(reqs, rules):
    ev = []
    for q in reqs:
        for (st, start, n) in q.chunks:
            if start % rules.pool and n > rules.decode_threshold:
                ev.append((q.rid, start, n, st))
    return ev


# ------------------------------------------------------------------ attempt-3 inputs
def load_a3(path=os.path.join(HERE, "a3_measured.json")):
    return json.load(open(path))


def measured_events(a3):
    return [(e["rid"], e["start"], e["n"], e["seq"] - 1) for e in a3["events"] if not e["decode_classified"]]  # step index = seq - 1


# ------------------------------------------------------------------ mode: replay (per-step arithmetic check)
def replay(a3, rules=None):
    R = rules or Rules()
    K = R.reserve
    plen = {rid: r["prompt_len"] for rid, r in a3["requests"].items()}
    computed = {}
    in_decode = set()
    n_rows = n_ok = 0
    bad = []
    skipped_unexplained = []
    running_order = []
    for s in a3["steps"]:
        tb, ib = R.max_sched, R.max_batched
        seen_waiting = False
        sched_ids = set()
        for rid, start, n, nsp in s["rows"]:
            n_rows += 1
            new = rid not in computed
            if nsp > 0:                                  # decode row
                pred = R.decode_tokens
                kind = "decode"
            else:
                rem = plen[rid] - start
                pred = mamba_clip(start, min(rem, tb, ib - K), plen[rid], R)
                kind = "new" if new else "prefill"
            if new:
                seen_waiting = True
                running_order.append(rid)
            elif seen_waiting:
                bad.append((s["seq"], rid, "running row after a waiting row"))
            if pred == n:
                n_ok += 1
            else:
                bad.append((s["seq"], rid[-8:], kind, start, n, pred, tb, ib))
            tb -= n
            ib -= n + K
            sched_ids.add(rid)
            computed[rid] = start + n
            if nsp == 0 and start + n >= plen[rid]:
                in_decode.add(rid)
        # running requests with prefill left that were not scheduled: must be explained by an exhausted budget
        for rid in running_order:
            if rid in sched_ids or rid in in_decode or rid in s.get("finished", []):
                continue
            if computed.get(rid, 0) < plen[rid] and ib > K:
                skipped_unexplained.append((s["seq"], rid[-8:], ib))
        running_order = [r for r in running_order if r not in s.get("finished", [])]
    return dict(rows=n_rows, rows_matched=n_ok, mismatches=bad[:20], n_mismatches=len(bad),
                skipped_prefill_unexplained=skipped_unexplained[:20], n_skipped_unexplained=len(skipped_unexplained))


# ------------------------------------------------------------------ mode: closed loop on the measured step grid
def a3_requests(a3, arrival_mode="bound", delay=0.07, order_by="engine"):
    """Exogenous request inputs. arrival: client send time + frontend delay, never later than the measured first
    scheduling (a request can only be admitted after it arrived); stack warm-up traffic (no client record): its
    measured first scheduling. Arrival ORDER is taken from the measured admission order (exogenous)."""
    reqs = []
    items = sorted(a3["requests"].items(), key=lambda kv: kv[1]["first_seq"])
    first_pos = {}
    for s in a3["steps"]:
        for k, row in enumerate(s["rows"]):
            first_pos.setdefault(row[0], (s["seq"], k))
    if order_by == "engine":      # measured engine admission order (= engine arrival order, FCFS)
        ordered = sorted(items, key=lambda kv: first_pos[kv[0]])
    else:                         # client send order (what a workload-only model knows)
        ordered = sorted(items, key=lambda kv: (kv[1]["client"]["t0"] if kv[1]["client"] else kv[1]["first_t"]))
    for i, (rid, r) in enumerate(ordered):
        c = r["client"]
        if c is None or arrival_mode == "measured":
            a = r["first_t"] - 1e-4
        else:
            a = min(r["first_t"] - 1e-4, c["t0"] + delay)
        reqs.append(Req(rid, r["prompt_len"], r["hit"], a, i, r["decode_steps"], r["phase"]))
    # vLLM admits FCFS and stops at the first request it cannot admit, so the measured admission order IS the engine
    # arrival order; the frontend can reorder concurrent sends (e.g. a 120k-token prompt takes longer to render and
    # tokenize), so arrival times are made non-decreasing along that order.
    for prev, q in zip(reqs, reqs[1:]):
        q.arrival = max(q.arrival, prev.arrival)
    return reqs


def closed_loop(a3, rules=None, reqs=None, grid=None):
    R = rules or Rules()
    reqs = reqs if reqs is not None else a3_requests(a3)
    grid = grid if grid is not None else [s["t"] for s in a3["steps"]]
    eng = Engine(R)
    pend = sorted(reqs, key=lambda q: (q.arrival, q.order))
    i = 0
    k = 0
    dt_tail = 0.05
    while i < len(pend) or not eng.idle():
        t = grid[k] if k < len(grid) else grid[-1] + dt_tail * (k - len(grid) + 1)
        while i < len(pend) and pend[i].arrival <= t:
            eng.add(pend[i])
            i += 1
        eng.schedule(t)
        k += 1
        if k > 20 * len(grid) + 100000:
            raise RuntimeError("closed loop did not terminate")
    return eng, reqs


def agreement(meas, sim, a3):
    """Per-event agreement. identity = (request, chunk start, n); 'exact' also needs the same scheduler step."""
    ph = {rid: r["phase"] for rid, r in a3["requests"].items()}
    M = {(r, s, n): st for r, s, n, st in meas}
    S = {(r, s, n): st for r, s, n, st in sim}
    exact = [k for k in M if k in S and S[k] == M[k]]
    same_chunk_other_step = [k for k in M if k in S and S[k] != M[k]]
    only_m = [k for k in M if k not in S]
    only_s = [k for k in S if k not in M]
    by_phase = collections.defaultdict(lambda: [0, 0])
    for k in M:
        by_phase[ph[k[0]]][0] += 1
    for k in S:
        by_phase[ph[k[0]]][1] += 1
    return dict(measured=len(M), simulated=len(S), exact_step_and_chunk=len(exact),
                same_chunk_different_step=len(same_chunk_other_step), measured_only=len(only_m), simulated_only=len(only_s),
                by_phase_measured_vs_simulated={p: v for p, v in sorted(by_phase.items())},
                measured_only_examples=[[k[0][-8:], k[1], k[2], M[k], ph[k[0]]] for k in only_m[:12]],
                simulated_only_examples=[[k[0][-8:], k[1], k[2], S[k], ph[k[0]]] for k in only_s[:12]])


def chunk_agreement(eng_reqs, a3):
    """All prefill chunks (aligned or not): measured vs simulated."""
    meas = set()
    for s in a3["steps"]:
        for rid, start, n, nsp in s["rows"]:
            if nsp == 0 and start < a3["requests"][rid]["prompt_len"]:
                meas.add((rid, start, n, s["seq"] - 1))
    sim = set()
    for q in eng_reqs:
        for st, start, n in q.chunks:
            sim.add((q.rid, start, n, st))
    m2 = {k[:3] for k in meas}
    s2 = {k[:3] for k in sim}
    return dict(measured_chunks=len(meas), simulated_chunks=len(sim), same_chunk=len(m2 & s2),
                same_chunk_and_step=len(meas & sim))


# ------------------------------------------------------------------ mode: ablation (frozen original's assumptions)
ORIGINAL = dict(reserve_draft_slots=False, decode_tokens=4, mamba_align=False, decode_first=True)


def ablate(a3):
    """Closed loop on attempt 3's exogenous inputs with the frozen simulator's scheduler assumptions, then each
    correction added. The frozen simulator also differed in workload and timing (see REPORT section 1)."""
    out = []
    steps = [
        ("frozen-original rules: budget 2048, no draft-slot reservation, decode row 4, no Mamba clip, decodes first",
         dict(ORIGINAL)),
        ("+ Mamba 'align' clip at 4,352 boundaries", dict(ORIGINAL, mamba_align=True)),
        ("+ draft-slot reservation K=3 per scheduled request (solo budget 2045; decode row costs 4+3)",
         dict(ORIGINAL, mamba_align=True, reserve_draft_slots=True)),
        ("+ scheduler order (running in admission order, then waiting) = corrected rules",
         dict(ORIGINAL, mamba_align=True, reserve_draft_slots=True, decode_first=False)),
        ("corrected rules without the Mamba clip", dict(mamba_align=False)),
        ("corrected rules without the draft-slot reservation", dict(reserve_draft_slots=False)),
    ]
    meas = measured_events(a3)
    for label, kw in steps:
        R = Rules(**kw)
        eng, reqs = closed_loop(a3, R)
        ev = events_of(reqs, R)
        ag = agreement(meas, ev, a3)
        out.append(dict(label=label, rules=R.as_dict(), events=len(ev), exact=ag["exact_step_and_chunk"],
                        same_chunk=ag["exact_step_and_chunk"] + ag["same_chunk_different_step"],
                        by_phase=ag["by_phase_measured_vs_simulated"]))
    return out


# ------------------------------------------------------------------ mode: workload-only prediction
def fit_step_time(a3):
    """Least squares t_step = a + b*prefill_tokens + c*decode_rows over attempt-3 steps (MEASURED step spacing).
    Returns coefficients and the residual scale; used only for the 'predict' mode."""
    xs, ys = [], []
    st = a3["steps"]
    for s0, s1 in zip(st, st[1:]):
        dt = s1["t"] - s0["t"]
        if dt <= 0 or dt > 3:
            continue
        pf = sum(r[2] for r in s0["rows"] if r[3] == 0)
        dr = sum(1 for r in s0["rows"] if r[3] > 0)
        if not s0["rows"]:
            continue
        xs.append((1.0, pf, dr))
        ys.append(dt)
    # normal equations 3x3
    import itertools
    A = [[sum(x[i] * x[j] for x in xs) for j in range(3)] for i in range(3)]
    b = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(3)]
    # solve
    M = [row[:] + [bb] for row, bb in zip(A, b)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(3):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [u - f * v for u, v in zip(M[r], M[c])]
    coef = [M[i][3] / M[i][i] for i in range(3)]
    res = [y - sum(cf * xx for cf, xx in zip(coef, x)) for x, y in zip(xs, ys)]
    return dict(a=coef[0], b_per_prefill_token=coef[1], c_per_decode_row=coef[2],
                rmse=math.sqrt(sum(r * r for r in res) / len(res)), n=len(xs))


def predict(a3, acc=2.9, time_scale=1.0, turnaround=0.03, rules=None):
    """Prediction from the workload plan only (no measured step grid, no measured decode counts): the client's phases
    (sequential), widths, S-phase start offsets, max_tokens; Q/P completion tokens as recorded by the client (the GPQA
    replies' lengths are a property of the workload here); decode steps = ceil((tokens - 1) / acc); step duration from
    the fitted model x time_scale. Stack warm-up traffic before P0 is kept at its measured times (not client traffic)."""
    R = rules or Rules()
    fit = fit_step_time(a3)
    reqs_by_phase = collections.defaultdict(list)
    for rid, r in a3["requests"].items():
        reqs_by_phase[r["phase"]].append((rid, r))
    phases = a3["phases"]
    eng = Engine(R)
    allreqs = []
    t = 0.0
    order = 0
    # stack warm-up: replay at measured relative times, serial
    warm = sorted(reqs_by_phase["stack-warmup"], key=lambda kv: kv[1]["first_seq"])
    t0w = warm[0][1]["first_t"] if warm else 0
    pending = []
    for rid, r in warm:
        q = Req(rid, r["prompt_len"], r["hit"], r["first_t"] - t0w, order, r["decode_steps"], "stack-warmup")
        order += 1
        pending.append(q)
    client_t = (phases[0]["t0"] - t0w)
    # client phases: each phase starts when the previous finished (+0.05 s); within a phase, width-limited closed loop
    phase_list = [p["tag"] for p in phases]
    plan = {p["tag"]: p for p in phases}
    queue = []   # (phase, [(rid, r, start_offset)])
    for tag in phase_list:
        rs = sorted(reqs_by_phase.get(tag, []), key=lambda kv: kv[1]["client"]["t0"])
        offs = {}
        if "plan" in plan[tag]:
            for ent in plan[tag]["plan"]:
                offs[ent[0]] = ent[2]
        queue.append((tag, rs, offs))
    widths = {"P0": 1, "warm": 1, "T0": 8, "F0": 4, "T1": 8, "F1": 4, "D": 8, "S": 8, "Q": 8, "P1": 1}
    steps = 0
    now = 0.0
    phase_i = -1
    active, todo = [], []
    phase_start = None

    def dec_steps(r):
        c = r["client"]
        tokens = c["completion_tokens"]
        return 0 if tokens <= 1 else math.ceil((tokens - 1) / acc)

    while True:
        # client: start next phase when all previous done
        if not active and not todo and (not pending or now >= client_t):
            phase_i += 1
            if phase_i >= len(queue):
                if not pending and eng.idle():
                    break
            else:
                tag, rs, offs = queue[phase_i]
                phase_start = max(now, client_t) + (0.05 if phase_i else 0)
                todo = [(rid, r, offs.get(r["client"]["id"], 0)) for rid, r in rs]
                width = widths.get(tag, 8)
        # send
        while todo and len(active) < width:
            rid, r, off = todo[0]
            send = max(phase_start + off, now)
            todo.pop(0)
            q = Req(rid, r["prompt_len"], r["hit"], send + turnaround, order, dec_steps(r), r["phase"])
            order += 1
            pending.append(q)
            active.append(q)
        pending.sort(key=lambda q: (q.arrival, q.order))
        while pending and pending[0].arrival <= now:
            q = pending.pop(0)
            eng.add(q)
            allreqs.append(q)
        if eng.idle() and pending:
            now = max(now, pending[0].arrival)
            continue
        if eng.idle() and not pending and not todo and not active:
            if phase_i >= len(queue) - 1:
                break
            continue
        out = eng.schedule(now)
        pf = sum(n for _, _, n, k in out if k == "prefill")
        dr = sum(1 for _, _, n, k in out if k == "decode")
        dt = max(0.005, (fit["a"] + fit["b_per_prefill_token"] * pf + fit["c_per_decode_row"] * dr)) * time_scale
        now += dt
        steps += 1
        done_now = [q for q in active if q.done]
        for q in done_now:
            active.remove(q)
        if steps > 200000:
            raise RuntimeError("predict did not terminate")
    allreqs += [q for q in pending]
    ev = events_of(allreqs, R)
    return ev, steps, fit


# ------------------------------------------------------------------ main
def main():
    a3 = load_a3()
    out = {}
    rep = replay(a3)
    out["replay"] = rep
    print(f"[replay] rows {rep['rows']}, n reproduced {rep['rows_matched']}, mismatches {rep['n_mismatches']}, "
          f"unexplained skipped prefills {rep['n_skipped_unexplained']}")
    for b in rep["mismatches"][:8]:
        print("   ", b)
    meas = measured_events(a3)
    for mode in ("bound", "measured"):
        eng, reqs = closed_loop(a3, Rules(), a3_requests(a3, arrival_mode=mode))
        ev = events_of(reqs, Rules())
        ag = agreement(meas, ev, a3)
        ag["all_prefill_chunks"] = chunk_agreement(reqs, a3)
        ag["steps_simulated"] = eng.step
        ag["steps_measured"] = len(a3["steps"])
        out[f"closed_{mode}"] = ag
        print(f"[closed/{mode}] measured {ag['measured']} simulated {ag['simulated']} exact {ag['exact_step_and_chunk']} "
              f"same-chunk-other-step {ag['same_chunk_different_step']} measured-only {ag['measured_only']} "
              f"sim-only {ag['simulated_only']}; chunks {ag['all_prefill_chunks']}; steps {eng.step}/{len(a3['steps'])}")
        print("    by phase (measured, simulated):", ag["by_phase_measured_vs_simulated"])
        if ag["measured_only_examples"]:
            print("    measured-only:", ag["measured_only_examples"][:6])
            print("    sim-only:", ag["simulated_only_examples"][:6])
    out["ablation"] = ablate(a3)
    eng, reqs = closed_loop(a3, Rules(), a3_requests(a3, order_by="client"))
    ag = agreement(meas, events_of(reqs, Rules()), a3)
    out["ablation"].append(dict(label="corrected rules, arrival order = client send order instead of engine order",
                                events=ag["simulated"], exact=ag["exact_step_and_chunk"],
                                same_chunk=ag["exact_step_and_chunk"] + ag["same_chunk_different_step"],
                                by_phase=ag["by_phase_measured_vs_simulated"]))
    for a in out["ablation"]:
        print(f"[ablate] {a['events']:4d} events, {a['same_chunk']:4d} same chunk as measured ({a['exact']} same step) -- {a['label']}")
    preds = []
    for acc in (2.6, 2.9, 3.2):
        for ts in (0.7, 1.0, 1.4):
            ev, nsteps, fit = predict(a3, acc=acc, time_scale=ts)
            ag = agreement(meas, ev, a3)
            preds.append(dict(acc=acc, time_scale=ts, events=len(ev), same_chunk=ag["exact_step_and_chunk"] + ag["same_chunk_different_step"],
                              steps=nsteps, by_phase=ag["by_phase_measured_vs_simulated"]))
            print(f"[predict] acc {acc} time x{ts}: {len(ev)} events ({preds[-1]['same_chunk']} same chunk as measured), steps {nsteps}")
    out["predict"] = dict(step_time_fit=fit, runs=preds)
    json.dump(out, open(os.path.join(HERE, "sim_a3_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
