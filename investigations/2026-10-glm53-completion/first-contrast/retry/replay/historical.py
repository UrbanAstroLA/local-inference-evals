#!/usr/bin/env python3
"""F7 / F1 [R-1, R-2]: replay of the earlier measured sessions on this stack through the image's own scheduler and
KV-cache manager (CPU container), and comparison with their measured cached-token counts.

Two kinds of evidence:
  A. STEP-EXACT trace replays (sessions with the guard's EngineCore scheduler log): b0-mapguard/run-a3 and
     d7/step2/run-b1-a1. Same comparison as run-fc1 (every step). Their workloads contain what run-fc1 lacks: exact
     repeats (T0, T1, F0, F1, D, S), warm-prompt EXTENSIONS of an earlier, shorter cached prompt (warm r2 on r3's
     prefix, warm r0 on r1's prefix), lengths just above block boundaries (4,240 / 8,240 below; 16,478 / 32,478 /
     64,217 / 64,478 / 120,217 above), staggered submission (S), EOS-terminated chat requests (Q) and aborted streams
     (phase 2d).
  B. CACHED-COUNT replays without a scheduler log (phase3, phase2-stage2): the image's scheduler is driven in
     model mode with the recorded submission order and concurrency (client t0/t1 windows); the measured cached
     counts are compared request by request. phase2-stage2 spans two server sessions (a1, a2): each is replayed
     from a fresh engine.
Writes /out/historical.json.
"""
import collections
import json
import sys

sys.path.insert(0, "/replay")
import engine_replay as E  # noqa: E402
import trace_replay as T  # noqa: E402

INV = "/inv"
STARTUP = [32, 64, 1, 1] + [256] * 64          # container/glm53-release-warmup.py (see reproduce_fc1)


def rule(P):
    return max(0, ((P - 1) // E.BLOCK - 1) * E.BLOCK)


def map_requests(recs, info, client, byid):
    S = [r for r in recs if r["kind"] == "sched"]
    t_of = {r["seq"]: r["t"] for r in S}
    t_start = min(c["t0"] for c in client)
    out = {}
    unrec = [q for q, d in info.items() if t_of[d["first"]] < t_start - 0.5]
    for q, mt in zip(unrec, STARTUP):
        out[q] = dict(prompt=None, max_tokens=mt, label="startup")
    used = set(out)
    mapped = []
    for c in sorted(client, key=lambda c: (c["t0"], c["t1"])):
        cand = [q for q, d in info.items() if q not in used and c["t0"] - 1.0 <= t_of[d["first"]] <= c["t1"] + 1.0
                and d["P_rec"] == c["prompt_tokens"]]
        if not cand:
            mapped.append((c, None))
            continue
        q = min(cand, key=lambda q: info[q]["first"])
        used.add(q)
        mapped.append((c, q))
    for c, q in mapped:
        if q is None:
            continue
        r = byid.get(c["id"])
        prompt = list(r["ids"]) if r is not None else None
        if prompt is not None:
            assert len(prompt) == c["prompt_tokens"], (c["id"], len(prompt), c["prompt_tokens"])
        stop = None
        if c.get("completion_tokens") is not None and c["completion_tokens"] < c["max_tokens"]:
            stop = info[q]["fin_step"]
        out[q] = dict(prompt=prompt, max_tokens=c["max_tokens"], salt=None, outputs=None,
                      label="%s:%s" % (c["phase"], c["id"]), stop_step=stop, client=c)
    # requests in the log that no client record describes (aborted streams of phase 2d, ...): filler, stopped
    # where the record ends them
    for q, d in info.items():
        if q not in out:
            out[q] = dict(prompt=None, max_tokens=max(d["gen_rec"] + 64, 1), label="other", stop_step=d["fin_step"])
    for q, d in out.items():
        if d["prompt"] is None:
            inf = info[q]
            P = inf["P_rec"] if inf["P_rec"] is not None else inf["steps"][-1][1][0]
            d["prompt"] = E.filler(P, hash(q) & 0xFFFFFFFF)
            d.setdefault("salt", None)
            d.setdefault("outputs", None)
    return out, mapped


def trace_session(name, rel):
    recs = [r for r in T.jl(f"{INV}/{rel}/mapguard/ms_p274.jsonl") if r["kind"] in ("sched", "upd")]
    client = [r for r in T.jl(f"{INV}/{rel}/client.jsonl") if r.get("kind") == "request"]
    byid = {r["id"]: r for r in json.load(open(f"{INV}/{rel}/requests.json"))["requests"]}
    S, U, info = T.prepass(recs)
    reqs, mapped = map_requests(recs, info, client, byid)
    tr = T.TraceReplay(recs, reqs)
    ok = tr.run(stop_on_mismatch=False)
    rows = []
    for c, q in mapped:
        hit = tr.eng.lookups.get(q, {}).get("hit") if q else None
        rows.append(dict(phase=c["phase"], id=c["id"], P=c["prompt_tokens"], measured=c["cached_tokens"],
                         replay=hit, equal=hit == c["cached_tokens"], rule=rule(c["prompt_tokens"]),
                         kind=classify(c, client)))
    last_bad = tr.mismatch[-1][0] if tr.mismatch else None
    t_of = {r["seq"]: r["t"] for r in S}
    other = [r for r in T.jl(f"{INV}/{rel}/client.jsonl") if r.get("kind") == "phase2d_start"]
    t2d = other[0]["t"] if other else None
    first_bad_t = t_of.get(tr.mismatch[0][0]) if tr.mismatch else None
    return dict(session=name, mode="trace", steps=tr.n_steps, exact=ok, mismatching_steps=len(tr.mismatch),
                first_mismatch_after_phase2d_start=(first_bad_t is not None and t2d is not None and
                                                    first_bad_t >= t2d),
                steps_exact_before_phase2d=sum(1 for r in S if t2d is None or r["t"] < t2d),
                first_mismatch=tr.mismatch[:2], last_mismatch_step=last_bad,
                unmapped=[c["id"] for c, q in mapped if q is None], requests=rows)


def classify(c, client):
    """exact repeat / extension of an earlier cached prompt / first use, from the client's prompt ids lengths."""
    earlier = [x for x in client if x["t1"] <= c["t0"] + 1e-3 and x is not c]
    same = [x for x in earlier if x["id"] == c["id"]]
    if same:
        return "exact-repeat"
    return "first-use"


# ------------------------------------------------------------------ model-mode replay of sessions without a log
def model_session(name, rel, segments, ids_from=None):
    """ids_from: the session whose requests.json holds the prompts (phase3 sent the phase2-stage2 prompts: same ids
    r0-r7, documents, history lengths and prompt token counts; checked below)."""
    client = [r for r in T.jl(f"{INV}/{rel}/client.jsonl") if r.get("kind") == "request"]
    byid = {r["id"]: r for r in json.load(open(f"{INV}/{ids_from or rel}/requests.json"))["requests"]}
    for c in client:
        if c["id"] in byid:
            assert len(byid[c["id"]]["ids"]) == c["prompt_tokens"], (name, c["id"])
    rows = []
    for seg_name, (t_lo, t_hi) in segments.items():
        cs = sorted([c for c in client if t_lo <= c["t0"] < t_hi], key=lambda c: c["t0"])
        rows += run_model(cs, byid, seg_name)
    return dict(session=name, mode="model", requests=rows)


def run_model(cs, byid, seg):
    """Submit in recorded order; a request arrives when every request whose t1 precedes its t0 has finished (the
    client's dependency), concurrent ones together. Decode: 1 + 3 drafts scheduled, accepted 3 (timing only)."""
    s = E.make_scheduler()
    eng = E.Engine(s)
    hbs = s._replay_block_sizes[1]
    pending = list(cs)
    live = {}
    done_t1 = []
    rows = []
    k = 0
    guard = 0
    while (pending or live or eng.queue) and guard < 200000:
        guard += 1
        # admit every pending request whose predecessors (t1 <= its t0) are finished
        for c in list(pending):
            preds = [x for x in cs if x["t1"] <= c["t0"] and x is not c]
            if all(id(x) in done_t1 for x in preds):
                rid = "%s-%d" % (c["id"], k)
                k += 1
                prompt = list(byid[c["id"]]["ids"])[: c["prompt_tokens"]] if c["id"] in byid else \
                    E.filler(c["prompt_tokens"], hash((c["id"], c["phase"])) & 0xFFFF)
                s.add_request(E.make_request(rid, prompt, c["max_tokens"], None, hbs))
                live[rid] = c
                pending.remove(c)
        if s.has_requests():
            eng.schedule()
        if eng.queue and (len(eng.queue) >= 2 or not s.has_requests()):
            so, n = eng.update(lambda q, so_, n_: [] if q not in s.requests or
                               s.requests[q].num_computed_tokens < s.requests[q].num_prompt_tokens and
                               s.requests[q].is_prefill_chunk else
                               E.filler(1 + len((so_.scheduled_spec_decode_tokens or {}).get(q, ())), n_))
        for rid in list(live):
            r = s.requests.get(rid)
            if r is None or r.is_finished():
                done_t1.append(id(live[rid]))
                c = live.pop(rid)
                hit = eng.lookups.get(rid, {}).get("hit")
                rows.append(dict(segment=seg, phase=c["phase"], id=c["id"], P=c["prompt_tokens"],
                                 measured=c["cached_tokens"], replay=hit, equal=hit == c["cached_tokens"],
                                 rule=rule(c["prompt_tokens"])))
    return rows


def main():
    res = {}
    res["b0-mapguard/run-a3"] = trace_session("b0-mapguard/run-a3", "b0-mapguard/run-a3")
    res["d7/step2/run-b1-a1"] = trace_session("d7/step2/run-b1-a1", "d7/step2/run-b1-a1")
    cl = [r for r in T.jl(f"{INV}/phase2-stage2/client.jsonl") if r.get("kind") == "request"]
    # phase2-stage2: two server sessions (a1 then a2); the a2 boundary is the second 'warm r7' (t0 ordering)
    w7 = sorted(c["t0"] for c in cl if c["phase"] == "warm" and c["id"] == "r7")
    segs = {"a1": (0, w7[1] - 1e-3), "a2": (w7[1] - 1e-3, 1e18)} if len(w7) > 1 else {"all": (0, 1e18)}
    res["phase2-stage2"] = model_session("phase2-stage2", "phase2-stage2", segs)
    res["phase3"] = model_session("phase3", "phase3", {"c1": (0, 1e18)}, ids_from="phase2-stage2")
    json.dump(res, open("/out/historical.json", "w"), indent=1, default=str)
    for k, v in res.items():
        rows = v["requests"]
        eq = sum(1 for r in rows if r["equal"])
        rep = [r for r in rows if r.get("kind") == "exact-repeat" or (r["measured"] or 0) > 0]
        rep_rule = sum(1 for r in rows if (r["measured"] or 0) > 0 and r["measured"] == r["rule"])
        print("%-22s mode %-5s steps %s exact %s | cached equal %d/%d | hits %d, of which = repeat rule %d" % (
            k, v["mode"], v.get("steps"), v.get("exact"), eq, len(rows), len(rep), rep_rule))
        for r in rows:
            if not r["equal"]:
                print("     DIFF", r)
        if v["mode"] == "trace" and not v["exact"]:
            print("     first mismatch", v["first_mismatch"][:1], "last mismatching step", v["last_mismatch_step"],
                  "of", v["steps"], "unmapped", v["unmapped"], "first mismatch inside phase 2d (aborted streams):",
                  v["first_mismatch_after_phase2d_start"], "steps before phase 2d:", v["steps_exact_before_phase2d"])


if __name__ == "__main__":
    main()
