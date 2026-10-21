#!/usr/bin/env python3
"""B+ D7 prefix-prefill exposure estimate (PROPOSAL.md section 4.2). Host, standard library, zero GPU. [E]

Uses the d7 exposure work's corrected scheduler simulator (d7/exposure/sim_d7.py, public repo, imported read-only and
hash-checked; its arithmetic reproduced all 14,712 scheduled rows of the B0 guard session exactly) to chunk each B+
prefix the way each engine's scheduler would, and counts the D7 exposure events: prefill chunks whose start is not a
multiple of 4 and whose length exceeds 1 + K (shorter rows take the indexer's decode path). Each event leaves one 4-token
index-K pool (the one straddling the chunk start) unwritten or written from another request's tokens, on every MLA
indexer layer (DCP1 + MLA owners on 0.9.1; on 0.7.0 the straddling pool's DCP owner rank). B1 carries the writer fix, so
its events are repaired by construction (verified CLEAN in d7 step 2) and count as 0 affected pools.

Rules per engine (from d7/exposure/REPORT.md and the server logs): 0.9.1 / B1: K = 3 draft slots (solo chunk 2,045; a
decode row costs 4 + 3), Mamba-align block 4,352; 0.7.0: K = 5 (solo chunk 2,043; a decode row costs 6 + 5), block 3,584
(server log 'Setting attention block size to 3584'). Budget 2,048. EAGLE back-off of the last cacheable block on both.
0.7.0's chunking is source-read, not measured in serving (as in d7).

Scenarios (the plan's submission order, outputs/plan.json):
  wave 1: the first 8 requests (8 different prefixes) arrive together on an idle engine; their prefills are serialised by
          the budget and each later one is chunked beside the decode rows of those admitted before it.
  later:  each later request arrives when a slot frees, beside 7 decoding requests (D = 7). j = 1 requests of prefixes
          not in wave 1 prefill cold; j >= 2 requests hit the prefix cache (hit = largest whole block below the prompt,
          minus the EAGLE back-off block; [I], measured per request on 0.9.1 / B1 by cached_tokens) and prefill only the
          remainder from a block-aligned start. Sensitivity: no cache hits at all; D = 0..7 for a cold prefill.
usage: d7_prefill_exposure.py OUT_JSON
"""
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
SIM = "<this-repo>/investigations/2026-10-glm53-completion/d7/exposure/sim_d7.py"
SIM_SHA = "c5662f1d61679f2d9299c48a36d18162173c5a367d32397ade22bfdd41577e21"
ENGINES = {"v070": dict(K=5, block=3584), "v091": dict(K=3, block=4352), "b1": dict(K=3, block=4352)}
FIXED = {"b1"}


def load_sim():
    got = hashlib.sha256(open(SIM, "rb").read()).hexdigest()
    if got != SIM_SHA:
        sys.exit(f"sim_d7.py changed: {got}")
    spec = importlib.util.spec_from_file_location("sim_d7_ro", SIM)
    m = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(m)
    return m


def hit_of(L, B):
    return max(0, (L - 1) // B * B - B)


def events(S, chunks, K):
    return [(st, n) for (_, st, n) in chunks if st % 4 and n > 1 + K]


def run_wave(S, eng, lengths, hits=None):
    """lengths: prompt lengths of requests arriving together at t=0 in this order. Returns per-request chunk lists."""
    R = S.Rules(K=eng["K"])
    E = S.Engine(R)
    reqs = []
    for i, L in enumerate(lengths):
        q = S.Req(f"r{i}", L, hit=(hits[i] if hits else 0), arrival=0.0, order=i, decode_steps=10 ** 9)
        reqs.append(q)
        E.add(q)
    steps = 0
    while any(not q.in_decode for q in reqs):
        E.schedule(float(steps))
        steps += 1
        assert steps < 10 ** 6
    return reqs


def run_beside(S, eng, L, D, hit=0):
    """One request arriving beside D requests already decoding."""
    R = S.Rules(K=eng["K"])
    E = S.Engine(R)
    for i in range(D):
        d = S.Req(f"d{i}", 8, arrival=0.0, order=i, decode_steps=10 ** 9)
        d.computed, d.in_decode, d.decode_left = 8, True, 10 ** 9
        E.running.append(d)
    q = S.Req("x", L, hit=hit, arrival=0.0, order=D, decode_steps=10 ** 9)
    E.add(q)
    steps = 0
    while not q.in_decode:
        E.schedule(float(steps))
        steps += 1
        assert steps < 10 ** 6
    return q


def main():
    S = load_sim()
    man = {m["prefix_id"]: m for m in json.load(open(f"{PKG}/out/prefix_manifest.json"))["prefixes"]}
    plan = json.load(open(f"{PKG}/out/plan.json"))
    order = [(r["prefix_id"], r["j"]) for r in plan["order"]]
    out = {"what": "D7 prefix-prefill exposure estimate [E] (ids, counts and positions only)", "sim_d7_sha256": SIM_SHA,
           "engines": {}}
    for e, eng in ENGINES.items():
        S.BLOCK = eng["block"]       # module global used by mamba_clip
        B = eng["block"]
        per = {}
        # sensitivity: cold prefill beside D decoding requests
        sens = {}
        for D in range(8):
            ev, pools = 0, 0
            for pid, m in man.items():
                q = run_beside(S, eng, m["prefix_len"], D)
                ev += len(events(S, q.chunks, eng["K"]))
                pools += m["prefix_len"] // 4
            sens[D] = {"events_12_prefixes": ev, "affected_pool_fraction": round(ev / pools, 6)}
        for scenario in ("central_cache_hits", "no_cache_hits"):
            first8 = order[:8]
            w = run_wave(S, eng, [man[p]["prefix_len"] for p, _ in first8])
            seen = {}
            rows = []
            for (p, j), q in zip(first8, w):
                ev = events(S, q.chunks, eng["K"])
                seen.setdefault(p, ev)
                rows.append({"k": len(rows) + 1, "prefix_id": p, "j": j, "D_at_admission": "wave 1 (decode rows of earlier admissions)",
                             "hit": 0, "chunks": len(q.chunks), "unaligned_starts": len(ev),
                             "affected_pools_in_its_prefix_kv": len(ev),
                             "start_residues_mod4": sorted({st % 4 for st, _ in ev})})
            for p, j in order[8:]:
                L = man[p]["prefix_len"]
                hit = hit_of(L, B) if (scenario == "central_cache_hits" and p in seen) else 0
                q = run_beside(S, eng, L, 7, hit)
                ev = events(S, q.chunks, eng["K"])
                inherited = [x for x in seen.get(p, []) if x[0] < hit] if hit else []
                if p not in seen:
                    seen[p] = ev
                rows.append({"k": len(rows) + 1, "prefix_id": p, "j": j, "D_at_admission": 7, "hit": hit, "chunks": len(q.chunks),
                             "unaligned_starts": len(ev), "affected_pools_in_its_prefix_kv": len(ev) + len(inherited),
                             "start_residues_mod4": sorted({st % 4 for st, _ in ev})})
            tot_pools = sum((man[p]["prefix_len"] // 4) for p, _ in order)
            tot_ev = sum(r["unaligned_starts"] for r in rows)
            per[scenario] = {"requests": rows, "events_total_48_requests": tot_ev,
                             "affected_pool_fraction_mean": round(tot_ev / tot_pools, 6),
                             "max_events_one_request": max(r["unaligned_starts"] for r in rows),
                             "requests_with_any_event": sum(r["unaligned_starts"] > 0 for r in rows),
                             "pools_affected_if_unfixed": tot_ev,
                             "per_continuation_affected_fraction": {
                                 "mean": round(sum(r["affected_pools_in_its_prefix_kv"] / (man[r["prefix_id"]]["prefix_len"] // 4) for r in rows) / len(rows), 6),
                                 "max": round(max(r["affected_pools_in_its_prefix_kv"] / (man[r["prefix_id"]]["prefix_len"] // 4) for r in rows), 6),
                                 "continuations_with_any": sum(r["affected_pools_in_its_prefix_kv"] > 0 for r in rows),
                                 "max_pools": max(r["affected_pools_in_its_prefix_kv"] for r in rows)},
                             "pools_affected_on_this_engine": 0 if e in FIXED else tot_ev}
        out["engines"][e] = {"K": eng["K"], "block": B, "fixed_writer": e in FIXED,
                             "cold_prefill_beside_D_decoding": sens, **per}
        c = per["central_cache_hits"]
        print(f"{e}: K={eng['K']} block={B}; central: {c['events_total_48_requests']} unaligned starts over 48 requests, "
              f"pool fraction {c['affected_pool_fraction_mean']:.5f}, requests with any {c['requests_with_any_event']}; "
              f"per continuation (inherited through the cache): {c['per_continuation_affected_fraction']}; "
              f"no-hit: {per['no_cache_hits']['events_total_48_requests']} ({per['no_cache_hits']['affected_pool_fraction_mean']:.5f}); "
              f"cold beside D=0..7: {[sens[D]['events_12_prefixes'] for D in range(8)]}"
              + ("  [B1: repaired by the D7 fix]" if e in FIXED else ""))
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
