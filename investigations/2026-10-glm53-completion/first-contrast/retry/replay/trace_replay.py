#!/usr/bin/env python3
"""F7 [R-1] trace-driven replay: drive the image's AsyncScheduler + KVCacheManager through a RECORDED session (the
guard's EngineCore scheduler log ms_p*.jsonl) and compare every step.

Inputs per step come only from the record where the record is the cause, never where it is the outcome:
  * arrival: a request is added just before the schedule() in which it first appears, in its recorded admission order
    (FCFS; the lookup and allocation happen at that schedule() call, exactly as recorded);
  * the model's commit count per request per step: accepted + 1 from the update record (`upd.rej`), 1 for the final
    prefill chunk or a decode step without drafts, 0 for a non-final prefill chunk (decided by the replay's own
    scheduler state);
  * the event order sched(N+1) / update(N) as recorded (batch queue 2);
  * token ids: the frozen fixture ids for every client request (prefix, forced list, salts as sent); seeded filler for
    requests whose prompts were not recorded (startup warm-up, V0 probes), whose lengths come from the record.
Outcomes compared at EVERY step (any difference is reported with the first differing step):
  scheduled set and order, num_computed after scheduling, scheduled tokens, scheduled drafts, finished ids,
  group-0 (MLA+index-K) and group-1 (kpool tail) block ids, the length of every group's block list, free blocks,
  and the guard's periodic live-block census.
"""
import collections
import json
import sys

sys.path.insert(0, "/replay")
import engine_replay as E  # noqa: E402


def jl(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def prepass(recs):
    """Per request: first seq, admission rank, prompt length and committed-token total implied by the record."""
    S = [r for r in recs if r["kind"] == "sched"]
    U = {r["fp"]: r for r in recs if r["kind"] == "upd"}
    info = collections.OrderedDict()
    for r in S:
        for q, v in r["reqs"].items():
            d = info.setdefault(q, dict(first=r["seq"], steps=[]))
            d["steps"].append((r["seq"], v, U.get(r["fp"], {}).get("rej", {}).get(q)))
        for q in r["finished"]:
            if q in info:
                info[q]["fin"] = r["seq"]
    # the update that finished a request is the last update processed before the schedule() that reports it
    fp_seq = {r["fp"]: r["seq"] for r in S}
    last_upd_before = {}
    cur = None
    for r in recs:
        if r["kind"] == "upd":
            cur = fp_seq.get(r["fp"])
        elif r["kind"] == "sched":
            last_upd_before[r["seq"]] = cur
    for q, d in info.items():
        st = d["steps"]
        fin_step = last_upd_before.get(d.get("fin")) if d.get("fin") else None
        d["computed_before"] = st[0][1][0] - st[0][1][1]
        first_dec = next((i for i, (_, v, _) in enumerate(st) if v[2] > 0), None)
        if first_dec is None:
            d["P_rec"] = st[-1][1][0]
        else:
            d["P_rec"] = st[first_dec - 1][1][0] if first_dec > 0 else None
        gen = 1 if d["P_rec"] is not None else 0
        for sq, v, rj in st[(first_dec or len(st)):]:
            if fin_step is not None and sq > fin_step:
                break
            gen += (rj[1] + 1) if rj else 0
        d["gen_rec"] = gen
        d["fin_step"] = fin_step
    return S, U, info


class TraceReplay:
    def __init__(self, recs, requests, filler_seed=1, num_blocks=E.NUM_BLOCKS, mla_prefix=None, kda_prefix=None,
                 max_tokens_override=None):
        """requests: req_id -> dict(prompt=list, salt=str|None, max_tokens=int, outputs=list|None, label=str);
        requests missing from the dict get filler prompts of the recorded length."""
        self.recs = recs
        self.S, self.U, self.info = prepass(recs)
        self.req = dict(requests)
        self.s = E.make_scheduler(num_blocks=num_blocks, mla_prefix=mla_prefix, kda_prefix=kda_prefix)
        self.eng = E.Engine(self.s)
        self.hbs = self.s._replay_block_sizes[1]
        self.filler_seed = filler_seed
        self.mto = max_tokens_override or {}
        self.out_pos = collections.Counter()
        self.prefill_at_sched = {}
        self.last_lists = {}
        self.mismatch = []
        self.n_steps = 0
        self.lookup_log = []
        self.step_occ = []

    def _req_obj(self, q):
        d = self.req.get(q)
        inf = self.info[q]
        if d is None:
            P = inf["P_rec"] if inf["P_rec"] is not None else inf["steps"][-1][1][0]
            d = dict(prompt=E.filler(P, hash((self.filler_seed, q)) & 0xFFFFFFFF), salt=None,
                     max_tokens=self.mto.get(q, max(inf["gen_rec"], 1)), outputs=None, label="unrecorded")
            self.req[q] = d
        return E.make_request(q, d["prompt"], d["max_tokens"], d["salt"], self.hbs, stop=d.get("stop_step") is not None)

    def _commit(self, q, so, rej, step=None):
        n = 0
        if q in rej:
            n = rej[q][1] + 1
        elif not self.prefill_at_sched.get((id(so), q), True):
            n = 1
        d = self.req[q]
        outs = d.get("outputs")
        k = self.out_pos[q]
        if outs is not None:
            toks = list(outs[k:k + n])
            toks += E.filler(n - len(toks), 7 + k)
        else:
            toks = E.filler(n, hash((q, k)) & 0xFFFF)
        self.out_pos[q] += n
        if n and d.get("stop_step") is not None and step == d["stop_step"]:
            toks[-1] = E.STOP_ID
        return toks

    def _compare(self, r, so):
        s = self.s
        reqs = {}
        changed = {}
        for rid, n in so.num_scheduled_tokens.items():
            ids = E.block_ids(s, rid)
            g0, g1 = ids[0], ids[1]
            key = (tuple(g0), tuple(g1))
            if self.last_lists.get(rid) != key:
                self.last_lists[rid] = key
                changed[rid] = [g0, g1, [len(x) for x in ids]]
            req = s.requests.get(rid)
            reqs[rid] = [int(req.num_computed_tokens) if req is not None else -1, int(n),
                         len((so.scheduled_spec_decode_tokens or {}).get(rid, ()))]
        for rid in so.finished_req_ids or ():
            self.last_lists.pop(rid, None)
        fin = sorted(map(str, so.finished_req_ids or ()))
        free = s.kv_cache_manager.block_pool.get_num_free_blocks()
        bad = []
        if list(reqs) != list(r["reqs"]):
            bad.append(("order/set", list(reqs)[:6], list(r["reqs"])[:6]))
        for k in r["reqs"]:
            if reqs.get(k) != r["reqs"][k]:
                bad.append(("reqs", k[-12:], reqs.get(k), r["reqs"][k]))
        if changed != r["changed"]:
            ks = set(changed) | set(r["changed"])
            for k in ks:
                if changed.get(k) != r["changed"].get(k):
                    a, b = changed.get(k), r["changed"].get(k)
                    bad.append(("blocks", k[-12:], a and (a[0][-4:], a[1], a[2]), b and (b[0][-4:], b[1], b[2])))
        if E.json_fp(so) != r["fp"]:
            bad.append(("fp", E.json_fp(so), r["fp"]))
        if fin != r["finished"]:
            bad.append(("finished", fin, r["finished"]))
        if free != r["free_blocks"]:
            bad.append(("free_blocks", free, r["free_blocks"]))
        if "excl" in r:
            live = set()
            for m in s.kv_cache_manager.coordinator.single_type_managers:
                for blks in m.req_to_blocks.values():
                    for b in blks:
                        if b.block_id != 0:
                            live.add(b.block_id)
            if len(live) != r["excl"]["live_blocks"]:
                bad.append(("live_blocks", len(live), r["excl"]["live_blocks"]))
        return bad

    def run(self, stop_on_mismatch=True, occupancy_every=0, on_step=None):
        added = set()
        for r in self.recs:
            if r["kind"] == "sched":
                new = [q for q in r["reqs"] if q not in added and self.info[q]["first"] == r["seq"]]
                for q in new:
                    self.s.add_request(self._req_obj(q))
                    added.add(q)
                so = self.eng.schedule()
                for q in so.num_scheduled_tokens:
                    self.prefill_at_sched[(id(so), q)] = self.s.requests[q].is_prefill_chunk
                self.n_steps += 1
                bad = self._compare(r, so)
                if on_step:
                    on_step(self, r, so)
                if occupancy_every and r["seq"] % occupancy_every == 0:
                    self.step_occ.append((r["seq"], E.group_occupancy(self.s)))
                if bad:
                    self.mismatch.append((r["seq"], bad))
                    if stop_on_mismatch:
                        return False
            elif r["kind"] == "upd":
                rej = r["rej"]
                so, _ = self.eng.update(lambda q, so_, n_: self._commit(q, so_, rej, n_))
                if E.json_fp(so) != r["fp"]:
                    self.mismatch.append((-1, [("upd-pairing", E.json_fp(so), r["fp"])]))
                    if stop_on_mismatch:
                        return False
        return not self.mismatch
