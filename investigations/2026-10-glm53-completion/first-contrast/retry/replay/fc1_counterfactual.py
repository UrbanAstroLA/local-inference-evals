#!/usr/bin/env python3
"""F7 / F4 [R-1, R-4]: counterfactual on run-fc1's real history. The image's scheduler replays run-fc1 exactly up to
the step before C1's first admission (step 1,961), then C1's 8 requests are submitted in model mode
  (a) in the order the engine actually admitted them (L4 L1 L3 A8 M3 M4 L8 L5; set by eight racing HTTP threads), and
  (b) in warm-s1's completion order (M3 L1 L4 L3 A8 M4 L5 L8: the retry's R-4 rule, oldest last-freed first),
  (c) in reverse completion order (adversarial).
(a) must reproduce the recorded outcome (7 hits, M3 a miss) if the model-mode continuation is faithful; (b) shows
whether the submission rule alone would have kept M3's prefix. Writes /out/fc1_counterfactual.json.
"""
import json
import sys

sys.path.insert(0, "/replay")
import engine_replay as E  # noqa: E402
import reproduce_fc1 as R  # noqa: E402
import trace_replay as T  # noqa: E402
from session_sim import measured_inputs  # noqa: E402


def run(order_name, order):
    recs = [r for r in T.jl(R.RUN + "/mapguard/ms_p274.jsonl") if r["kind"] in ("sched", "upd")]
    S, U, info = T.prepass(recs)
    reqs, creq, salts = R.build_requests(recs, info)
    c1 = {d["label"][3:]: (q, d) for q, d in reqs.items() if d["label"].startswith("C1-")}
    first_c1 = min(info[q]["first"] for q, _ in c1.values())
    cut = []
    for r in recs:
        if r["kind"] == "sched" and r["seq"] >= first_c1:
            break
        cut.append(r)
    keep = {q: d for q, d in reqs.items() if not d["label"].startswith("C1-")}
    tr = T.TraceReplay(cut, keep, max_tokens_override=R.unrecorded_max_tokens(info, reqs))
    assert tr.run(stop_on_mismatch=True), tr.mismatch[:1]
    s, eng = tr.s, tr.eng
    M = measured_inputs()
    import random
    rng = random.Random(5)
    hbs = s._replay_block_sizes[1]
    pos = {}
    pre = {}
    objs = []
    for n in order:
        q, d = c1[n]
        rid = "cf-" + n
        objs.append(rid)
        pos[rid] = 0
        s.add_request(E.make_request(rid, d["prompt"], d["max_tokens"], d["salt"], hbs))
    outs = {"cf-" + n: c1[n][1]["outputs"] for n in order}

    def commit(q, so, k):
        if pre.get((k, q), True):
            return []
        nspec = len((so.scheduled_spec_decode_tokens or {}).get(q, ()))
        fx = q[3:]
        acc = min(nspec, rng.choice(M["acc"].get(fx) or [2])) if nspec else 0
        f = outs.get(q) or []
        p = pos.get(q, 0)
        t = list(f[p:p + 1 + acc])
        t += E.filler(1 + acc - len(t), 3 + p)
        pos[q] = p + 1 + acc
        return t
    g = 0
    while s.has_requests() or eng.queue:
        g += 1
        if s.has_requests():
            so = eng.schedule()
            for q in so.num_scheduled_tokens:
                pre[(eng.step_no, q)] = s.requests[q].is_prefill_chunk
            if len(eng.queue) >= 2:
                eng.update(commit)
        else:
            eng.update(commit)
        if all(x in eng.lookups for x in objs) and g > 50:
            break
    hits = {rid[3:]: eng.lookups.get(rid, {}).get("hit") for rid in objs}
    return dict(order=order, hits=hits, misses=[n for n, h in hits.items() if h == 0])


def main():
    res = {}
    for name, order in (("a_recorded_admission", ["L4", "L1", "L3", "A8", "M3", "M4", "L8", "L5"]),
                        ("b_warm_completion_order", ["M3", "L1", "L4", "L3", "A8", "M4", "L5", "L8"]),
                        ("c_reverse_completion", ["L8", "L5", "M4", "A8", "L3", "L4", "L1", "M3"])):
        res[name] = run(name, order)
        print(name, res[name]["hits"], "misses", res[name]["misses"])
    json.dump(res, open("/out/fc1_counterfactual.json", "w"), indent=1)


if __name__ == "__main__":
    main()
