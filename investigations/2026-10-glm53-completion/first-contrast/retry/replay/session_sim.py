#!/usr/bin/env python3
"""F7 / F3 [R-1, R-3]: exercise the proposed retry session (analysis/fcr_geometry.py) through the image's own
scheduler and KV-cache manager (CPU container) under plausible interleavings and resource pressure.

Model mode (no record to follow): the engine loop is EngineCore.step_with_batch_queue (batch queue 2); the client is
the retry client's phase order and submission policy [R-4]; the "model" commits, per decode step, 1 + accepted drafts
drawn from the per-fixture empirical acceptance distribution measured in run-fc1's C1 batch (natural drafts on these
fixtures' forced histories; CV schedules poison/oracle/reject:2 as defined); forced requests commit their frozen
forced ids, so every decode-written block carries the real tokens and hash. Step durations follow a linear model
fitted to run-fc1's scheduler log (used for arrival interleaving and costing only; the cache outcome depends on
order, not on seconds).
[R-11, 22f s.3] Acceptance stress modes (scenario key "acceptance"; default "measured" is the distribution above and
leaves every earlier scenario unchanged): "single" = single-token advance throughout (poison-like), "max" = maximum
accepted advance throughout (oracle-like), "unequal" = the fixtures in "fast" at "max" and every other at "single".
They apply to every natural (non-CV) decode step, primary and LP batches alike; CV keeps its defined schedules. They
are stress scenarios, not predictions of D2.

Per scenario it records
  * every lookup: restored length vs the F1 expectation (hit / partial / miss), restored block provenance (owner
    request, prefill vs decode written), block identity across users of a shared build;
  * the survival margin of every pending lookup: the free-queue position of the first block the lookup needs
    (MLA/index-K chain and the KDA state at the restored boundary, all 7 groups), minimised over the window from the
    build's completion to the lookup = how many more block allocations the schedule could absorb;
  * live and cached block occupancy by group per phase; the all-8-in-flight lead (output index of the leading
    request when the last one starts decoding) per batch; steps and modelled seconds per phase.
usage: session_sim.py SCENARIO_JSON  -> /out/sim_<name>.json
"""
import collections
import json
import os
import random
import sys

sys.path.insert(0, "/replay")
sys.path.insert(0, "/inv/first-contrast/retry/analysis")
import engine_replay as E  # noqa: E402
import trace_replay as T  # noqa: E402
import fcr_geometry as geo  # noqa: E402

RUN1 = "/inv/first-contrast/run-fc1"
FC = "/inv/first-contrast"


# ------------------------------------------------------------------ measured inputs (run-fc1)
def measured_inputs():
    recs = [r for r in T.jl(RUN1 + "/mapguard/ms_p274.jsonl") if r["kind"] in ("sched", "upd")]
    S = [r for r in recs if r["kind"] == "sched"]
    U = {r["fp"]: r for r in recs if r["kind"] == "upd"}
    tag = {}
    for r in T.jl(RUN1 + "/forcewin/fw_r0_p370.jsonl"):
        if r.get("kind") == "add":
            tag[r["req"]] = r["tag"]
    acc = collections.defaultdict(list)
    for r in S:
        u = U.get(r["fp"])
        if not u:
            continue
        for q, v in u["rej"].items():
            t = tag.get(q, "")
            if t.startswith("C1-") and v[0] == 3:
                acc[t[3:]].append(v[1])
    # step-time model: dt = a + b * n_decode_requests + pre_tokens / rate + m * mixed   (least squares, dt < 5 s)
    X, Y = [], []
    for a, b in zip(S, S[1:]):
        dt = b["t"] - a["t"]
        if dt >= 5:
            continue
        pre = sum(v[1] for v in a["reqs"].values() if v[2] == 0 and v[1] > 1)
        nd = sum(1 for v in a["reqs"].values() if v[2] > 0)
        X.append([1.0, nd, pre, 1.0 if (pre and nd) else 0.0])
        Y.append(dt)
    import numpy as np
    coef, *_ = np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)
    return dict(acc={k: v for k, v in acc.items()}, step_coef=[float(x) for x in coef], n_steps_fit=len(Y))


def step_time(coef, so, prefill_rids):
    pre = sum(n for r, n in so.num_scheduled_tokens.items() if r in prefill_rids)
    nd = sum(1 for r in so.num_scheduled_tokens if r not in prefill_rids)
    if not so.num_scheduled_tokens:
        return 0.002
    return max(0.02, coef[0] + coef[1] * nd + coef[2] * pre + coef[3] * (1.0 if pre and nd else 0.0))


# ------------------------------------------------------------------ simulator
class Sim:
    def __init__(self, sc, M, ids, G):
        self.sc, self.M, self.ids, self.G = sc, M, ids, G
        self.rng = random.Random(sc.get("seed", 0))
        nb = sc.get("num_blocks", E.NUM_BLOCKS)
        if sc.get("start") == "trace" and nb == E.NUM_BLOCKS:
            self._start_from_trace()
        else:
            self.s = E.make_scheduler(num_blocks=nb)
            self.eng = E.Engine(self.s)
        self.hbs = self.s._replay_block_sizes[1]
        self.label = {}
        self.trk = E.Tracker(self.s, self.eng, lambda q: self.label.get(q, q))
        self.t = 0.0
        self.meta = {}            # rid -> dict(phase, fixture, form, P, forced ids, mode, expected, ...)
        self.out_pos = collections.Counter()
        self.adv_steps, self.adv_tok = collections.Counter(), collections.Counter()   # [R-11] speculative decode steps
        self.finish_t = {}
        self.first_dec = {}
        self.pending_lookups = {}  # rid -> (request object for hashes, expected c, build label)
        self.margins = collections.defaultdict(lambda: None)
        self.phase_log = []
        self.occ = []
        self.prefill_at = {}
        self.cur_phase = None
        self.last_finish = {}
        self.window_open = {}     # rid -> label of the phase whose end opened the current risk window
        self.win_margins = {}

    def _start_from_trace(self):
        """Exact startup state: the run-fc1 record replayed through the image up to (not including) its first
        client request after V0 (the release warm-up and the 4 V0 probes are identical in every session)."""
        import reproduce_fc1 as R
        recs = [r for r in T.jl(RUN1 + "/mapguard/ms_p274.jsonl") if r["kind"] in ("sched", "upd")]
        S, U, info = T.prepass(recs)
        reqs, creq, salts = R.build_requests(recs, info)
        first_client = min(info[q]["first"] for q in reqs)
        cut = []
        for r in recs:
            if r["kind"] == "sched" and r["seq"] >= first_client:
                break
            cut.append(r)
        tr = T.TraceReplay(cut, {}, max_tokens_override=R.unrecorded_max_tokens(
            {q: d for q, d in info.items() if q not in reqs}, {}))
        ok = tr.run(stop_on_mismatch=True)
        assert ok, tr.mismatch[:1]
        # drain: finish the queued update(s) as recorded (the cut ends on an idle engine)
        while tr.eng.queue:
            tr.eng.update(lambda q, so_, n_: [])
        self.s, self.eng = tr.s, tr.eng
        self.start_info = dict(replayed_steps=tr.n_steps, free_blocks=self.s.kv_cache_manager.block_pool
                               .get_num_free_blocks())

    # ---- requests
    def _make(self, phase, r, rid):
        n = r["fixture"]
        pre, fo = self.ids[n]["prefix"], self.ids[n]["forced"]
        kind, k = r["prompt"]
        prompt = pre + fo[:k] if kind == "prefix+forced" else list(pre)
        assert len(prompt) == r["P"], (rid, len(prompt), r["P"])
        forced = fo[r["forced"][0]: r["forced"][0] + r["forced"][1]] if r.get("forced") else None
        self.meta[rid] = dict(phase=phase["tag"], fixture=n, form=r["form"], P=r["P"], forced=forced,
                              mode=r.get("cv_mode", "natural"), expected=r["cached_expected"], salt=r["salt"],
                              build=r.get("build"))
        self.label[rid] = rid
        req = E.make_request(rid, prompt, r["max_tokens"], geo.salt(r["salt"]), self.hbs)
        return req

    def _natural_acc(self, fixture, nspec):
        """[R-11, 22f s.3] accepted drafts of a natural (non-CV) decode step. 'measured' (default, unchanged): drawn
        from run-fc1 C1's per-fixture distribution. Stress modes (not predictions of D2): 'single' = single-token advance
        throughout (poison-like, 0 accepted), 'max' = maximum accepted advance throughout (oracle-like, every scheduled
        draft accepted), 'unequal' = fixtures in sc['fast'] at 'max', every other fixture at 'single'."""
        mode = self.sc.get("acceptance", "measured")
        if mode == "unequal":
            mode = "max" if fixture in self.sc.get("fast", ()) else "single"
        if mode == "single":
            return 0
        if mode == "max":
            return nspec
        dist = self.M["acc"].get(fixture) or [2]
        return min(nspec, self.rng.choice(dist))

    def _commit(self, q, so, n):
        if self.prefill_at.get((n, q), True):
            return []
        m = self.meta.get(q)
        nspec = len((so.scheduled_spec_decode_tokens or {}).get(q, ()))
        if m is None:
            k = 1
        else:
            mode = m["mode"]
            if nspec == 0:
                acc = 0
            elif mode == "poison":
                acc = 0
            elif mode == "oracle":
                acc = nspec
            elif mode == "reject:2":
                acc = min(1, nspec)
            else:
                acc = self._natural_acc(m["fixture"], nspec)
            k = 1 + acc
        if m is not None and nspec:
            self.adv_steps[q] += 1
            self.adv_tok[q] += k
        fo = (m or {}).get("forced")
        p = self.out_pos[q]
        toks = list(fo[p:p + k]) if fo else []
        toks += E.filler(k - len(toks), 11 + p)
        self.out_pos[q] += k
        return toks

    # ---- margins
    def _needed(self, req, expected):
        """Units the lookup needs, each a list of the cached copies (any surviving copy serves): the MLA/index-K chain
        blocks 0..m-1 (m = expected / 4352 + 1: the EAGLE check matches one block past the restored boundary) and
        the KDA state at the boundary in each of the 7 groups. Returns (units, all_present)."""
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
        cache = self.s.kv_cache_manager.block_pool.cached_block_hash_to_block._cache
        if expected <= 0:
            return [], True
        m = expected // geo.BLOCK + 1
        hs = req.block_hashes
        if len(hs) < m:
            return [], False

        def copies(h, g):
            v = cache.get(make_block_hash_with_group_id(h, g))
            if v is None:
                return []
            return [v] if not isinstance(v, dict) else list(v.values())
        units = []
        for i in range(m):
            c = copies(hs[i], 0)
            if not c:
                return units, False
            units.append(c)
        j = expected // geo.BLOCK - 1
        for g in range(2, 9):
            c = copies(hs[j], g)
            if not c:
                return units, False
            units.append(c)
        return units, True

    def _update_margins(self):
        if not self.pending_lookups:
            return
        pos = {}
        q = self.s.kv_cache_manager.block_pool.free_block_queue
        b = q.fake_free_list_head.next_free_block
        i = 0
        while b is not None and b is not q.fake_free_list_tail:
            pos[b.block_id] = i
            i += 1
            b = b.next_free_block
        BIG = 10 ** 6
        for rid, (req, exp) in list(self.pending_lookups.items()):
            if rid not in self.window_open:
                continue
            units, ok = self._needed(req, exp)
            if not ok:
                cur = -1                       # already lost (partial or full eviction predicted)
            else:
                # a unit survives while any copy survives: its position is that of its youngest copy (live = BIG)
                cur = min((max((pos.get(c.block_id, BIG) if c.ref_cnt == 0 else BIG) for c in u) for u in units),
                          default=BIG)
            prev = self.margins[rid]
            self.margins[rid] = cur if prev is None else min(prev, cur)
            k = (rid, self.window_open[rid])
            pw = self.win_margins.get(k)
            self.win_margins[k] = cur if pw is None else min(pw, cur)

    # ---- engine loop
    def _run_until(self, done_fn, arrivals):
        """arrivals: list of (time, Request) sorted; runs the engine until done_fn() and the engine is drained."""
        arrivals = list(arrivals)
        guard = 0
        while True:
            guard += 1
            if guard > 400000:
                raise RuntimeError("sim guard")
            while arrivals and arrivals[0][0] <= self.t + 1e-9:
                _, req = arrivals.pop(0)
                self.s.add_request(req)
            if self.s.has_requests():
                so = self.eng.schedule()
                n = self.eng.step_no
                pre = set()
                for q, k in so.num_scheduled_tokens.items():
                    rq = self.s.requests[q]
                    self.prefill_at[(n, q)] = rq.is_prefill_chunk
                    before = rq.num_computed_tokens - k
                    if before < rq.num_prompt_tokens:
                        pre.add(q)
                    elif q not in self.first_dec:
                        self.first_dec[q] = n
                        self._check_lead(q)
                dt = step_time(self.M["step_coef"], so, pre)
                if len(self.eng.queue) >= 2:
                    self._update()
                self.t += dt
                self._update_margins()
                if self.sc.get("occ_every") and n % self.sc["occ_every"] == 0:
                    self.occ.append((n, round(self.t, 1), self.cur_phase, E.group_occupancy(self.s)))
            elif self.eng.queue:
                self._update()
            elif arrivals:
                self.t = max(self.t, arrivals[0][0])
                continue
            else:
                return
            if not arrivals and done_fn() and not self.s.has_requests():
                while self.eng.queue:
                    self._update()
                return

    def _check_lead(self, q):
        ph = self.batch_of.get(q)
        if ph is None or ph in self.lead:
            return
        mem = self.batch_members[ph]
        if all(x in self.first_dec for x in mem):
            self.lead[ph] = max(self.out_pos[x] for x in mem)

    def _update(self):
        so, n = self.eng.update(lambda q, so_, n_: self._commit(q, so_, n_))
        for q in list(so.num_scheduled_tokens):
            rq = self.s.requests.get(q)
            if (rq is None or rq.is_finished()) and q not in self.finish_t:
                self.finish_t[q] = self.t
                self.pending_lookups.pop(q, None)

    def _advance(self, rids):
        """[R-11] committed tokens per speculative decode step over the phase's requests: mean and per fixture."""
        st = sum(self.adv_steps[q] for q in rids)
        if not st:
            return None
        return dict(mean=round(sum(self.adv_tok[q] for q in rids) / st, 3), steps=st,
                    by_fixture={self.meta[q]["fixture"]: round(self.adv_tok[q] / self.adv_steps[q], 3)
                                for q in rids if self.adv_steps[q] and q in self.meta})

    # ---- client
    def _rid(self, ph, r):
        if r["form"] == "CV":
            return "CV-%s-%s" % (r["cv_mode"].replace(":", "_"), r["fixture"])
        return "%s-%s" % (ph["tag"], r["fixture"])

    def run_plan(self, plan):
        res = []
        build_finish = {}
        prebuilt = {}
        self.batch_of, self.batch_members, self.lead = {}, {}, {}
        if self.sc.get("no_touch"):
            plan = [p for p in plan if p["kind"] != "touch"]
        for ph in plan:
            self.cur_phase = ph["tag"]
            t0, step0 = self.t, self.eng.step_no
            if ph["kind"] == "gate":
                self.t += self.sc.get("gate_s", 120.0)
                res.append(dict(phase=ph["tag"], kind="gate", t0=round(t0, 1), secs=round(self.t - t0, 1), steps=0))
                continue
            if ph["kind"] == "probe":
                for i, L in enumerate((124, 134, 158, 133)):
                    rid = "V0-q%d" % i
                    req = E.make_request(rid, E.filler(L, 900 + i), 96, None, self.hbs)
                    self._run_until(lambda rid=rid: rid in self.finish_t, [(self.t, req)])
                self.t += geo.GAP_S
                res.append(dict(phase="V0", kind="probe", t0=round(t0, 1), secs=round(self.t - t0, 1),
                                steps=self.eng.step_no - step0))
                continue
            reqs = list(ph["requests"])
            if ph["kind"] == "build" and self.sc.get("build_order") == "longest":
                reqs = list(reversed(reqs))
            if ph["kind"] in ("run", "touch", "ps") and ph.get("build") and ph.get("form") != "CV":
                pol = self.sc.get("order_policy", "completion")
                fin = {r["fixture"]: self.last_finish.get((ph["salt"], r["fixture"]), 0) for r in reqs}
                if pol == "plan":
                    pass
                elif pol == "completion":
                    reqs = sorted(reqs, key=lambda r: fin.get(r["fixture"], 0))
                elif pol == "reverse":
                    reqs = sorted(reqs, key=lambda r: -fin.get(r["fixture"], 0))
                elif pol == "random":
                    self.rng.shuffle(reqs)
            objs = []
            for r in reqs:
                rid = self._rid(ph, r)
                req = prebuilt.pop(rid, None) or self._make(ph, r, rid)
                objs.append((rid, req, r))
            if ph["kind"] == "run" and ph["form"] in ("C", "D", "LC", "LD"):
                self.batch_members[ph["tag"]] = [rid for rid, _, _ in objs]
                for rid, _, _ in objs:
                    self.batch_of[rid] = ph["tag"]
            if ph.get("sequential") or (ph["kind"] == "build" and self.sc.get("build_sequential")):
                for rid, req, r in objs:
                    self._run_until(lambda rid=rid: rid in self.finish_t, [(self.t, req)])
                    self.t += 0.1
            else:
                st = ph.get("stagger", 0.0) * self.sc.get("stagger_scale", 1.0)
                jit = self.sc.get("arrival_jitter", 0.0)
                arr = sorted([(self.t + i * st + (self.rng.uniform(0, jit) if jit else 0.0), req)
                              for i, (rid, req, r) in enumerate(objs)], key=lambda x: x[0])
                rids = [rid for rid, _, _ in objs]
                self._run_until(lambda rids=rids: all(x in self.finish_t for x in rids), arr)
            # the end of any phase on this salt (re)opens the risk window of every later user on the salt: their
            # needed blocks were last freed (or first created) by this phase
            for rid2, (req2, exp2) in self.pending_lookups.items():
                if self.meta.get(rid2, {}).get("salt") == ph.get("salt"):
                    self.window_open[rid2] = ph["tag"]
            for rid, req, r in objs:
                self.last_finish[(r["salt"], r["fixture"])] = self.finish_t.get(rid, self.t)
            if ph["kind"] == "build":
                build_finish[ph["tag"]] = {r["fixture"]: self.finish_t[rid] for rid, req, r in objs}
            # the risk window of every later user on this salt opens as soon as the blocks it needs exist
            if ph.get("salt"):
                idx = plan.index(ph)
                for later in plan[idx + 1:]:
                    if later.get("salt") != ph["salt"] or later["kind"] not in ("run", "ps", "touch"):
                        continue
                    for r in later["requests"]:
                        rid = self._rid(later, r)
                        if rid in self.pending_lookups or r["cached_expected"] <= 0:
                            continue
                        req = prebuilt.get(rid) or self._make(later, r, rid)
                        prebuilt[rid] = req
                        if self._needed(req, r["cached_expected"])[1]:
                            self.pending_lookups[rid] = (req, r["cached_expected"])
                            self.window_open[rid] = ph["tag"]
            rows = []
            for rid, req, r in objs:
                rs = self.trk.restores.get(rid, {})
                hit = rs.get("hit")
                exp = r["cached_expected"]
                cls = "cold" if exp == 0 and hit == 0 else ("hit" if hit == exp else ("miss" if hit == 0 else
                                                                                        ("partial" if hit < exp else
                                                                                         "OVER")))
                mla = [(i, b, ow) for g, i, b, ow in rs.get("blocks", []) if g == 0]
                kinds = sorted({ow[3] for _, _, ow in mla if ow})
                owners = sorted({ow[0].rsplit("-", 1)[0] for _, _, ow in mla if ow})
                kda = sorted({(ow[0].rsplit("-", 1)[0], ow[3]) for g, i, b, ow in rs.get("blocks", [])
                              if 2 <= g <= 8 and ow})
                rows.append(dict(rid=rid, fixture=r["fixture"], P=r["P"], expected=exp, hit=hit, cls=cls,
                                 restored_from=owners, restored_kinds=kinds, restored_kda=kda,
                                 mla_ids=[b for _, b, _ in sorted(mla)],
                                 margin=self.margins.get(rid),
                                 margin_last_window=self.win_margins.get((rid, self.window_open.get(rid))),
                                 margin_windows={w: m for (x, w), m in self.win_margins.items() if x == rid},
                                 lookup_step=rs.get("step"),
                                 finish_t=round(self.finish_t.get(rid, -1), 2)))
            self.t += geo.GAP_S
            res.append(dict(phase=ph["tag"], kind=ph["kind"], form=ph.get("form"), t0=round(t0, 1),
                            secs=round(self.t - t0, 1), steps=self.eng.step_no - step0,
                            lead=self.lead.get(ph["tag"]), order=[r["fixture"] for r in reqs], requests=rows,
                            advance=self._advance([rid for rid, _, _ in objs])))
        return res


def main():
    sc = json.load(open(sys.argv[1])) if sys.argv[1].endswith(".json") else json.loads(sys.argv[1])
    G = geo.load(FC)
    ids = E.load_ids(FC)
    M = measured_inputs()
    plan = geo.phases(G, sc.get("variant", "B"))
    sim = Sim(sc, M, ids, G)
    res = sim.run_plan(plan)
    out = dict(scenario=sc, start=getattr(sim, "start_info", None), step_coef=M["step_coef"],
               acc_mean={k: round(sum(v) / len(v), 3) for k, v in M["acc"].items()}, phases=res,
               total_s=round(sim.t, 1), occupancy=sim.occ)
    name = sc.get("name", "scenario")
    json.dump(out, open("/out/sim_%s.json" % name, "w"), default=str)
    # console summary
    bad = []
    for p in res:
        for r in p.get("requests", []):
            if r["cls"] not in ("hit", "cold"):
                bad.append((p["phase"], r["fixture"], r["cls"], r["hit"], r["expected"]))
    mins = [r["margin"] for p in res for r in p.get("requests", []) if r["margin"] is not None]
    leads = {p["phase"]: p["lead"] for p in res if p.get("lead") is not None}
    print(json.dumps(dict(name=name, total_s=out["total_s"], non_hits=bad[:20], n_non_hits=len(bad),
                          min_margin=min(mins) if mins else None, max_lead=max(leads.values()) if leads else None,
                          leads=leads)))


if __name__ == "__main__":
    main()
