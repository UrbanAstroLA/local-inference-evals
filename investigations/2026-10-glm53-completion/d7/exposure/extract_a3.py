#!/usr/bin/env python3
"""D7 exposure accounting (PLAN.md amendment 18), step 1a: MEASURED schedule of guard attempt 3.

CPU only, standard library only. Reads the frozen attempt-3 records (read-only):
  run-a3/mapguard/ms_p274.jsonl  scheduler side: one 'sched' record per schedule() call; reqs = {req_id: [num_computed_tokens
                                 AFTER scheduling, num_scheduled_tokens, num_scheduled_spec_tokens]} in insertion order of
                                 SchedulerOutput.num_scheduled_tokens, i.e. the scheduler's own order (running, then waiting)
  run-a3/mapguard/mg_p370.jsonl / mg_p404.jsonl   worker side (rank 0 / rank 1): 'step' rows [req_id, tag, lo, hi, nsched,
                                 nspec, row, ...] in persistent-batch order, and 'poll' records (site 1 = idx_w_pf)
  run-a3/client.jsonl            client phases and requests (t0, prompt_tokens, cached_tokens, completion_tokens)
Writes a3_measured.json (no prompt text; request ids, offsets and counts only).

Definitions
  prefill chunk      a scheduled row of a request whose chunk start (computed_after - n) is below its prompt length
  exposure event     a prefill chunk with start % 4 != 0 that the worker classifies as prefill (n > 1 + num_spec = 4;
                     rows with n <= 4 go through the decode path, see the step-1315 row below) -- per step x request
  guard event        (step, tag) pair with idx_w_pf records in a3 (the frozen guard's own count: 217 per rank)
"""
import collections
import json
import os
import sys

RUN = sys.argv[1] if len(sys.argv) > 1 else "<runs>/b0-mapguard/run-a3"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "a3_measured.json")
DECODE_MAX = 4          # 1 + num_speculative_tokens (DFlash2 x3): rows with n <= 4 are decode-classified by the runner
POOL = 4


def jl(p):
    with open(p) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    sched = [d for d in jl(f"{RUN}/mapguard/ms_p274.jsonl") if d["kind"] == "sched"]
    client = list(jl(f"{RUN}/client.jsonl"))
    phases = [c for c in client if c["kind"] == "phase"]
    creqs = [c for c in client if c["kind"] == "request"]

    # ---- per-request prompt length from the scheduler stream: last prefill row ends at the prompt length; decode rows
    # always carry spec tokens (nspec 3) in this run, so prompt length = max(start + n) over rows with nspec == 0.
    plen, first_seq, first_t, first_start = {}, {}, {}, {}
    dec_steps = collections.Counter()
    for d in sched:
        for rid, (after, n, nsp) in d["reqs"].items():
            start = after - n
            if rid not in first_seq:
                first_seq[rid], first_t[rid], first_start[rid] = d["seq"], d["t"], start
            if nsp == 0:
                plen[rid] = max(plen.get(rid, 0), start + n)
            else:
                dec_steps[rid] += 1

    # ---- phase of each request (client windows; traffic before P0 is the stack's own release warm-up)
    def phase_of(t):
        # latest client phase that had started (server time >= client send time); before P0: the stack's warm-up
        cur = "stack-warmup"
        for p in phases:
            if t >= p["t0"] - 0.01:
                cur = p["tag"]
        return cur

    # ---- map server request -> client request (phase, prompt length, cached tokens, admission order)
    by_phase = collections.defaultdict(list)
    for c in creqs:
        by_phase[c["phase"]].append(c)
    used = set()
    cmap = {}
    for rid in sorted(first_seq, key=lambda r: first_seq[r]):
        ph = phase_of(first_t[rid])
        cands = [c for c in by_phase.get(ph, []) if id(c) not in used and c["prompt_tokens"] == plen.get(rid)
                 and c.get("cached_tokens", 0) == first_start[rid]]
        if cands:
            c = min(cands, key=lambda c: c["t0"])
            used.add(id(c))
            cmap[rid] = c

    steps = []
    for d in sched:
        rows = []
        for rid, (after, n, nsp) in d["reqs"].items():
            rows.append([rid, after - n, n, nsp])
        steps.append(dict(seq=d["seq"], t=d["t"], rows=rows, finished=d.get("finished", [])))

    # ---- exposure events from the scheduler stream
    events = []
    for s in steps:
        for k, (rid, start, n, nsp) in enumerate(s["rows"]):
            if start < plen.get(rid, 0) and start % POOL and nsp == 0:
                events.append(dict(seq=s["seq"], rid=rid, start=start, n=n, decode_classified=n <= DECODE_MAX,
                                   sched_index=k, rows_in_step=len(s["rows"]), phase=phase_of(first_t[rid])))

    # ---- cause of each event: what ended the request's PREVIOUS chunk at an unaligned offset
    last_chunk = {}
    chunk_of = {}
    for s in steps:
        for k, (rid, start, n, nsp) in enumerate(s["rows"]):
            if nsp == 0 and start < plen.get(rid, 0):
                chunk_of[(rid, start)] = (s["seq"], n, len(s["rows"]), k,
                                          sum(1 for r in s["rows"] if r[3] > 0),
                                          sum(1 for j, r in enumerate(s["rows"]) if r[3] == 0 and j < k))
                last_chunk.setdefault(rid, []).append((start, n, s["seq"]))
    for e in events:
        prevs = [c for c in last_chunk[e["rid"]] if c[0] + c[1] == e["start"]]
        if not prevs:
            e["cause"] = "first chunk (prefix hit at unaligned offset)"
            continue
        pstart, pn, pseq = prevs[0]
        _, _, nrows, k, ndec, npf_before = chunk_of[(e["rid"], pstart)]
        if nrows == 1 and pn == 2045:
            e["cause"] = "solo budget 2048-3 (2045)"
        elif ndec and not npf_before:
            e["cause"] = f"shared step: {ndec} decode row(s) ahead"
        elif npf_before and not ndec:
            e["cause"] = f"shared step: {npf_before} prefill row(s) ahead"
        elif npf_before and ndec:
            e["cause"] = "shared step: decode and prefill rows ahead"
        else:
            e["cause"] = f"other (prev n={pn}, rows={nrows})"
        e["prev_chunk"] = [pstart, pn, pseq]

    # ---- the guard's own events (worker side), joined to the scheduler stream by (req_id, chunk start, n)
    guard = {}
    for rank, f in ((0, "mg_p370.jsonl"), (1, "mg_p404.jsonl")):
        steprows, recs = {}, collections.defaultdict(list)
        for d in jl(f"{RUN}/mapguard/{f}"):
            if d["kind"] == "step":
                steprows[d["step"]] = d
            elif d["kind"] == "poll":
                for r in d.get("records", []):
                    if r["site"] == 1:
                        recs[(r["step"], r["tag"])].append(r)
        ev = {}
        for (st, tag), rs in recs.items():
            row = next(r for r in steprows[st]["rows"] if r[1] == tag)
            pf_rows = [r for r in steprows[st]["rows"] if r[4] > DECODE_MAX]
            pos_in_pf = [r[1] for r in pf_rows].index(tag) if tag in [r[1] for r in pf_rows] else -1
            flags = sorted({r["flags"] for r in rs})
            ev[(row[0], row[2], row[4])] = dict(worker_step=st, tag=tag, records=len(rs), flags=flags,
                                                kind="pool_cross" if 8388608 in flags else "missing_pred",
                                                position_among_prefill_rows=pos_in_pf, warm=steprows[st].get("warm", 0))
        guard[rank] = ev
    g0 = guard[0]
    for e in events:
        k = (e["rid"], e["start"], e["n"])
        e["guard_rank0"] = g0.get(k)
        e["guard_rank1"] = guard[1].get(k)

    reqs = {}
    for rid in first_seq:
        c = cmap.get(rid)
        reqs[rid] = dict(first_seq=first_seq[rid], first_t=first_t[rid], prompt_len=plen.get(rid), hit=first_start[rid],
                         decode_steps=dec_steps[rid], phase=phase_of(first_t[rid]),
                         client=None if c is None else dict(id=c["id"], t0=c["t0"], t1=c["t1"], max_tokens=c["max_tokens"],
                                                             completion_tokens=c["completion_tokens"],
                                                             cached_tokens=c.get("cached_tokens"), prompt_tokens=c["prompt_tokens"]))
    out = dict(run=RUN, n_sched_records=len(sched), n_requests=len(reqs), steps=steps, requests=reqs, events=events,
               phases=[{k: p[k] for k in ("tag", "t0", "t1")} | ({"plan": p["plan"]} if "plan" in p else {}) for p in phases],
               guard_events_per_rank={r: len(v) for r, v in guard.items()},
               guard_only=[dict(rank=r, key=list(k), **v) for r, g in guard.items() for k, v in g.items()
                           if not any((e["rid"], e["start"], e["n"]) == k for e in events)])
    json.dump(out, open(OUT, "w"))
    # ---- summary
    ev_pf = [e for e in events if not e["decode_classified"]]
    print(f"sched records {len(sched)}, requests {len(reqs)} (client-mapped {len(cmap)})")
    print(f"unaligned prefill-chunk starts (scheduler side): {len(events)}; decode-classified (n<=4): "
          f"{sum(e['decode_classified'] for e in events)}; exposure events (prefill-classified): {len(ev_pf)} "
          f"in {len({e['rid'] for e in ev_pf})} requests")
    print(f"guard events per rank: {out['guard_events_per_rank']}; joined to scheduler events: "
          f"{sum(1 for e in ev_pf if e['guard_rank0'])} (rank 0), {sum(1 for e in ev_pf if e['guard_rank1'])} (rank 1); "
          f"guard-only: {len(out['guard_only'])}")
    print("by phase:", dict(collections.Counter(e["phase"] for e in ev_pf)))
    print("by cause:", dict(collections.Counter(e["cause"] for e in ev_pf)))
    print("guard kind (rank 0):", dict(collections.Counter((e["guard_rank0"] or {}).get("kind") for e in ev_pf)))
    print("chunk start mod 4352:", collections.Counter(e["start"] % 4352 for e in ev_pf).most_common(12))


if __name__ == "__main__":
    main()
