#!/usr/bin/env python3
"""F7 [R-1] step 1: reproduce run-fc1 exactly with the image's own scheduler / KV-cache manager (CPU container).

Writes /out/reproduce_fc1.json and prints a summary:
  * step-by-step equality with the recorded scheduler log (5,609 schedule() calls);
  * the cached-token count of every client request (= num_computed before its first chunk, which the record shows
    equals the API's cached_tokens) against the client's record;
  * the one-block drop on the 7 hit fixtures, M3's miss, the eviction of warm-M3's state (which block, at which step,
    by which request), and where the attempt-1 prefix-sentinel range [c0 - 1024, c0) would have been computed;
  * group occupancy (live / cached-free per group) at every step, summarised per phase.
"""
import collections
import json
import sys

sys.path.insert(0, "/replay")
import engine_replay as E  # noqa: E402
import trace_replay as T  # noqa: E402

FC = "/inv/first-contrast"
RUN = FC + "/run-fc1"


def build_requests(recs, info):
    cl = T.jl(RUN + "/client.jsonl")
    start = next(r for r in cl if r.get("kind") == "start")
    salts = start["salts"]
    creq = [r for r in cl if r.get("kind") == "request"]
    ids = E.load_ids(FC)
    tags = {}
    for r in T.jl(RUN + "/forcewin/fw_r0_p370.jsonl"):
        if r.get("kind") == "add":
            tags[r["req"]] = r["tag"]
    bytag = {r["tag"]: r for r in creq}
    out, label = {}, {}
    # keyed requests: hook add records
    for q, tag in tags.items():
        c = bytag[tag]
        n = c["fixture"]
        pre, fo = ids[n]["prefix"], ids[n]["forced"]
        if c["phase"].startswith("LW"):
            prompt = pre + fo[: c["P"] - len(pre)]
            outs = None
        else:
            prompt = pre
            outs = fo
        assert len(prompt) == c["P"], tag
        assert E.hashlib.sha256(json.dumps(prompt).encode()).hexdigest() == c["prompt_sha256"], tag
        out[q] = dict(prompt=prompt, salt=c["salt"], max_tokens=c["max_tokens"], outputs=outs, label=tag)
    # unkeyed warm-s1 requests: scheduler requests first seen during the warm phase, matched by prompt length and,
    # for the two equal-length prompts (A8, L8), by completion order against the client's t1
    S = [r for r in recs if r["kind"] == "sched"]
    t_of = {r["seq"]: r["t"] for r in S}
    warm = [c for c in creq if c["phase"] == "warm-s1"]
    w0, w1 = min(c["t0"] for c in warm), max(c["t1"] for c in warm)
    cand = [q for q, d in info.items() if q not in tags and w0 <= t_of[d["first"]] <= w1 + 1]
    used = set()
    for c in sorted(warm, key=lambda c: c["t1"]):
        qs = sorted([q for q in cand if q not in used and info[q]["P_rec"] == c["P"]], key=lambda q: info[q]["fin"])
        q = qs[0]
        used.add(q)
        n = c["fixture"]
        assert E.hashlib.sha256(json.dumps(ids[n]["prefix"]).encode()).hexdigest() == c["prompt_sha256"]
        out[q] = dict(prompt=ids[n]["prefix"], salt=c["salt"], max_tokens=1, outputs=None, label=c["tag"])
    return out, creq, salts


def unrecorded_max_tokens(info, reqs):
    """max_tokens of the requests whose prompts were not recorded, from their senders' fixed payloads:
    the stack's release warm-up (container/glm53-release-warmup.py: greedy 32, rendered-chat 64, 8K depth 1, 16K
    prefill 1, then 4 passes of n=16 x 256, all ignore_eos) and the client's V0 probes (max_tokens 96; every probe
    returned 96 tokens, client.jsonl n_tokens)."""
    order = [q for q in info if q not in reqs]
    fixed = [32, 64, 1, 1] + [256] * 64 + [96] * 4
    assert len(order) == len(fixed), (len(order), len(fixed))
    return dict(zip(order, fixed))


def fc1_details(tr, trk, reqs, creq):
    """The three run-fc1 facts the retry must model: the one-block drop, M3's eviction, the sentinel's private block."""
    lab = {q: d["label"] for q, d in reqs.items()}
    rid = {v: k for k, v in lab.items()}
    B = E.BLOCK
    out = {"drop": [], "m3": {}, "sentinel_attempt1": []}
    for n in ["L1", "L4", "M3", "M4", "L3", "L5", "A8", "L8"]:
        c1, w = rid["C1-" + n], rid["warm-s1-" + n]
        P = len(reqs[c1]["prompt"])
        rs = trk.restores.get(c1, {})
        c = rs.get("hit", 0)
        mla = [(i, b, ow) for g, i, b, ow in rs.get("blocks", []) if g == 0]
        kda = [(g, i, b, ow) for g, i, b, ow in rs.get("blocks", []) if 2 <= g <= 8]
        c0 = P // B * B
        rec = dict(fixture=n, P=P, hit=c, blocks_restored=c // B, planned_attempt1=c0,
                   rule=max(0, ((P - 1) // B - 1) * B),
                   restored_mla_all_prefill_from_warm=all(ow and ow[0] == "warm-s1-" + n and ow[3] == "prefill"
                                                          for _, _, ow in mla) if mla else None,
                   restored_kda=[(g, i, ow and ow[0], ow and ow[3]) for g, i, b, ow in kda][:7])
        out["drop"].append(rec)
        if c:
            last_shared = c // B - 1
            sb = next(b for i, b, ow in mla if i == last_shared)
            out["sentinel_attempt1"].append(dict(
                fixture=n, restored_boundary_c=c, last_shared_block=[c - B, c], last_shared_block_id=sb,
                last_shared_owner=trk.owner.get(sb, None) and trk.owner[sb][:4],
                attempt1_dump_range=[c0 - 1024, c0], attempt1_dump_block_index=(c0 - 1) // B,
                attempt1_dump_in_shared_block=(c0 - 1) // B <= last_shared))
    # M3: which warm-M3 blocks were evicted before C1-M3's lookup, when and in what order
    st = trk.restores.get(rid["C1-M3"], {}).get("step")
    ev = [(s, b, ow[1], ow[2]) for s, b, ow in trk.evictions if ow and ow[0] == "warm-s1-M3" and s <= st]
    P = len(reqs[rid["C1-M3"]]["prompt"])
    c = max(0, ((P - 1) // B - 1) * B)
    need_kda = [e for e in ev if 2 <= e[3] <= 8 and e[2] == c // B - 1]
    need_mla = [e for e in ev if e[3] == 0 and e[2] < c // B]
    out["m3"] = dict(lookup_step=st, rule_c=c, evicted_warm_m3_blocks_before_lookup=len(ev),
                     first_needed_kda_state_evicted=need_kda[:1], needed_kda_state_evictions=len(need_kda),
                     first_needed_mla_evicted=sorted(need_mla)[:1], needed_mla_evictions=len(need_mla),
                     other_fixtures_needed_evicted=sorted({ow[0] for s, b, ow in trk.evictions
                                                          if ow and ow[0].startswith("warm-s1-") and
                                                          ow[0] != "warm-s1-M3" and s <= st}))
    return out


def main():
    recs = [r for r in T.jl(RUN + "/mapguard/ms_p274.jsonl") if r["kind"] in ("sched", "upd")]
    S, U, info = T.prepass(recs)
    reqs, creq, salts = build_requests(recs, info)
    tr = T.TraceReplay(recs, reqs, max_tokens_override=unrecorded_max_tokens(info, reqs))
    trk = E.Tracker(tr.s, tr.eng, lambda q: reqs[q]["label"] if q in reqs else "unrec:" + q[-8:])
    occ = []
    phase_of = {q: d["label"].split("-")[0] if not d["label"].startswith("warm") else "warm-s1"
                for q, d in reqs.items()}

    def on_step(t, r, so):
        if r["seq"] % 4 == 0 or so.finished_req_ids:
            occ.append((r["seq"], E.group_occupancy(t.s)))
    ok = tr.run(stop_on_mismatch=False, on_step=on_step)
    res = dict(steps=tr.n_steps, sched_records=len(S), exact=ok, n_mismatch_steps=len(tr.mismatch),
               first_mismatches=[(s, b[:4]) for s, b in tr.mismatch[:5]])
    # cached tokens per client request
    cached = []
    for q, d in reqs.items():
        lk = tr.eng.lookups.get(q, {})
        c = next(c for c in creq if c["tag"] == d["label"])
        cached.append(dict(tag=d["label"], P=c["P"], replay_hit=lk.get("hit"), replay_lookup_step=lk.get("at_step"),
                           recorded_cached=c["cached_tokens"], equal=lk.get("hit") == c["cached_tokens"],
                           eagle_rule=max(0, ((c["P"] - 1) // E.BLOCK - 1) * E.BLOCK)))
    res["cached"] = sorted(cached, key=lambda x: x["tag"])
    res["cached_all_equal"] = all(x["equal"] for x in cached)
    res["n_client_requests_replayed"] = len(cached)
    res["config"] = dict(served=E.SERVED, placement=E.placement_summary(tr.s),
                         eagle_groups=sorted(tr.s.kv_cache_manager.coordinator.eagle_group_ids),
                         block_sizes=tr.s._replay_block_sizes)
    res["occupancy_samples"] = occ[::5]
    res.update(fc1_details(tr, trk, reqs, creq))
    json.dump(res, open("/out/reproduce_fc1.json", "w"), indent=1, default=str)
    print("steps replayed", tr.n_steps, "of", len(S), "exact", ok, "mismatching steps", len(tr.mismatch))
    for s, b in tr.mismatch[:5]:
        print("  MISMATCH step", s, b[:4])
    print(json.dumps(res["m3"], default=str))
    for x in res["sentinel_attempt1"]:
        print("  sentinel", x)
    for x in res["drop"]:
        print("  drop", {k: v for k, v in x.items() if k != "restored_kda"})
    for x in res["cached"]:
        print("  %-12s P %6d replay %6s recorded %6s equal %s rule %6d" % (x["tag"], x["P"], x["replay_hit"],
                                                                         x["recorded_cached"], x["equal"],
                                                                         x["eagle_rule"]))


if __name__ == "__main__":
    main()
