#!/usr/bin/env python3
"""CPU tests for mapguard_hooks (PLAN.md amendment 17, item 3).  NO GPU.

Run inside the derived mapguard image (or B0 with the module on PYTHONPATH):
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e TRITON_INTERPRET=1 \
     -v $PWD:/t -w /t --entrypoint python3 <image> tests/test_mapguard.py OUT.json

Scenario: the real B0 KVCacheManager hands out block ids (pool pre-cycled past block 935),
a mini async scheduler emits real SchedulerOutputs, and B0's real mapping builders (Triton
interpreter) produce every mapping the kernels consume (b0emu.py).  The guard's real hook
functions check them against its independent expectation (Ledger + build_expected).

  N*  negative controls  -- must NOT be flagged
  F*  injected faults    -- must be flagged with the stated class
  G*  graph-safety and observe-only lints
  A*  offline analysis rules (rank ownership, exact position join, transmission, verdicts)
"""
import json
import os
import random
import sys
import time
import traceback

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "guard"))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))

import torch  # noqa: E402

try:
    from vllm import mapguard_hooks as mg  # derived image
    SRC = "vllm.mapguard_hooks (image)"
except Exception:  # noqa: BLE001
    import mapguard_hooks as mg  # noqa: E402
    SRC = "guard/mapguard_hooks.py"
import b0emu  # noqa: E402
from b0emu import Emu, Fault  # noqa: E402
import analyze_mapguard as am  # noqa: E402

RESULTS = []
VIOL = mg.VIOLATION_COUNTERS


def result(name, ok, **kw):
    RESULTS.append(dict(test=name, ok=bool(ok), **kw))
    print(("PASS " if ok else "FAIL ") + name + "  " + json.dumps(kw, default=str)[:400], flush=True)


class Guard:
    """Wires one Dev + Ledger like a worker rank does, and runs the real hook functions."""

    def __init__(self, rank=0, layers=(3, 7)):
        self.geo = mg.Geo(nb=b0emu.NB, R=b0emu.R, W0=b0emu.W0)
        self.dev = mg.Dev(self.geo, torch.device("cpu"))
        self.led = mg.Ledger(self.geo)
        self.step = 0
        self.rank = rank
        self.layers = layers
        mg.ENABLED = True
        mg.S.dev, mg.S.rank, mg.S.geo, mg.S.ledger = self.dev, rank, self.geo, self.led
        self.impl = object()
        mg.S.impl_layer = {id(self.impl): layers[0]}
        mg.S.first_layer = layers[0]
        self.steps = []       # per-step host digest (as the worker step log)
        self.ring = {}

    def begin(self, so):
        self.led.update(so)
        self.fp = mg.fingerprint(so)

    def upload(self, req_ids):
        self.step += 1
        a, issues = mg.build_expected(self.dev.lay, self.led, req_ids, self.step)
        self.dev.E.copy_(torch.from_numpy(a))
        rows = []
        for rid in req_ids:
            st = self.led.cur[rid]
            rows.append([rid, st["tag"], st["lo"], st["hi"], st["nsched"], st["nspec"]])
        self.steps.append(dict(step=self.step, sched_step=self.led.step, fp=self.fp, rows=rows, dig=self.led.digest(req_ids),
                               issues=issues))
        return issues

    def run_checks(self, out, layer_list=None, trap=None):
        import contextlib
        before = self.dev.cnt.clone()
        nrec0 = int(self.dev.nrec[0])
        mg.S.pos_buf = out["pos"]
        mg.S.qsl_buf = out["qsl_consumed"]
        kv = torch.ones(1)
        ctx = trap if trap is not None else contextlib.nullcontext()
        with ctx:
            self._hooks(out, layer_list, kv)
        # disarm after the forward (as execute_model's wrapper does)
        return self._readout(before, nrec0)

    def _hooks(self, out, layer_list, kv):
        for L in (layer_list or self.layers):
            mg.S.impl_layer[id(self.impl)] = L
            mg.hook_mla_write(self.impl, kv, out["slot0"])
            name = f"model.layers.{L}.self_attn.indexer.k_cache"
            ntok, ndt = out["ntok"], out["ndt"]
            if out["nd"] < out["nreq"]:
                mg.hook_idx_prefill_write(name, out["idx_slot"][ndt:ntok], out["pos"], ndt, ntok, 4)
                mg.hook_tail_seed(name, out["tail"][ndt:ntok], out["pos"], ndt, 4)
                mg.hook_idx_prefill_read(name, out["pread"], out["pos"])
            if out["nd"]:
                ds, dp, dt = out["dec"]
                mg.hook_idx_decode_write(name, ds, dp, dt, out["pos"], ndt)
                if "dread" in out:
                    rt, rl = out["dread"]
                    mg.hook_idx_decode_read(name, rt, rl, out["pos"], int(rt.shape[0]))
            sel, nsa = out["sel"]

            class Lyr:
                layer_name = f"model.layers.{L}.self_attn.mla_attn"
            mg.hook_mla_read(Lyr(), sel, nsa, out["ntok_pad"])

    def _readout(self, before, nrec0):
        self.dev.E[mg.H_ARMED] = 0
        d = self.dev.cnt - before
        n = int(self.dev.nrec[0])
        recs = self.dev.rec[nrec0:min(n, mg.CAP)].tolist()
        slot = self.step % mg.RING
        if int(self.dev.ring_step[slot]) == self.step:
            self.ring[self.step] = self.dev.ring_p0[slot].tolist()
        return d, recs

    @staticmethod
    def viol(d):
        return {f"{mg.SITES[s]}.{c}": int(d[s, mg.CI[c]]) for s in range(len(mg.SITES))
                for c in VIOL if int(d[s, mg.CI[c]])}

    @staticmethod
    def flags(recs):
        f = 0
        for r in recs:
            f |= int(r[mg.REC_COLS.index("flags")])
        return mg.flag_names(f)


def scenario(trigger=None, seed=1, max_steps=60, pad=True, budget=2048, chunk_override=None):
    """Drive one scenario; trigger(emu, so, plan, step) -> list[Fault] (injected that step)."""
    rnd = random.Random(seed)
    emu = Emu(seed=seed, budget=budget, concurrency=3)
    g = Guard()
    A = [700000 + i for i in range(9000)]
    emu.add(A, 6, rid="A")                                  # 3 MLA blocks, finishes early
    emu.add([710000 + i for i in range(5000)], 10, rid="B")
    emu.add(A[:8704] + [720000 + i for i in range(300)], 6, rid="C")   # prefix hit (2 blocks)
    emu.add([730000 + i for i in range(300)], 12, rid="D")  # short context
    emu.add([740000 + i for i in range(4400)], 5, rid="E")  # admitted later: reused rows
    emu.add([750000 + i for i in range(132)], 8, rid="F")
    per_step = []
    injected = []
    for s in range(1, max_steps + 1):
        if not (emu.running or emu.waiting):
            break
        so, plan = emu.schedule(chunk_override=(chunk_override(s) if chunk_override else None))
        if not so.num_scheduled_tokens:
            emu.complete(plan, {})
            continue
        faults = trigger(emu, so, plan, s) if trigger else []
        if faults:
            injected.append((s, [f.kind for f in faults]))
        g.begin(so)
        req_ids = b0emu.sort_batch_req_ids(so.num_scheduled_tokens, so.scheduled_spec_decode_tokens, 4)
        ns = [so.num_scheduled_tokens[r] for r in req_ids]
        uniform_decode = all(so.scheduled_spec_decode_tokens.get(r) for r in req_ids) and len(set(ns)) == 1
        pad_to = None
        if pad and uniform_decode:
            ntok = sum(ns)
            pad_to = next(c for c in (4, 8, 16, 24, 32, 40, 48, 56, 64) if c >= ntok)
            if pad_to == ntok:
                pad_to = None
        out = emu.runner_step(so, faults=faults, pad_to=pad_to)
        issues = g.upload(out["req_ids"])
        d, recs = g.run_checks(out)
        accepted = {}
        for rid, pl in plan.items():
            if pl["k"] or (pl["st"]["prefill_done"] and not pl["st"].get("first_decode")):
                accepted[rid] = rnd.randint(1, pl["k"] + 1)
        emu.commit(out, accepted, plan)
        rej = emu.complete(plan, accepted)
        per_step.append(dict(step=g.step, fp=g.fp, viol=Guard.viol(d), flags=Guard.flags(recs),
                             faults=[f.kind for f in faults], issues=issues, rej=rej,
                             nd=out["nd"], nreq=out["nreq"], ntok=out["ntok"], pad=pad_to,
                             checked={mg.SITES[i]: int(d[i, mg.CI["checked"]]) for i in range(len(mg.SITES))},
                             high={mg.SITES[i]: int(d[i, mg.CI["high"]]) for i in range(len(mg.SITES))},
                             spec=int(d[mg.S_MLA_W, mg.CI["spec"]]), recs=recs[:4]))
    return dict(emu=emu, guard=g, steps=per_step, injected=injected)


def summarize(sc):
    tot = {}
    for st in sc["steps"]:
        for k, v in st["viol"].items():
            tot[k] = tot.get(k, 0) + v
    return tot


# ------------------------------------------------------------------ negative controls
def test_negative_controls():
    t0 = time.time()
    sc = scenario()
    steps = sc["steps"]
    tot = summarize(sc)
    checked = {}
    high = {}
    for st in steps:
        for k, v in st["checked"].items():
            checked[k] = checked.get(k, 0) + v
        for k, v in st["high"].items():
            high[k] = high.get(k, 0) + v
    n_mixed = sum(1 for st in steps if 0 < st["nd"] < st["nreq"])
    n_pad = sum(1 for st in steps if st["pad"])
    n_spec = sum(st["spec"] for st in steps)
    n_rej = sum(sum(st["rej"].values()) for st in steps)
    hits = [e for e in sc["guard"].led.events]  # last step only; use emu state below
    hit_reqs = [w["rid"] for w in sc["emu"].running] + []
    issues = [st["issues"] for st in steps if st["issues"]]
    result("N1 clean scenario: no violation at any site (real allocator + real B0 builders)",
           not tot and not issues and mg.S.errors == 0, violations=tot, issues=issues[:3],
           guard_errors=mg.S.errors, steps=len(steps),
           secs=round(time.time() - t0, 1))
    result("N1 coverage: every check site exercised", all(checked.get(s, 0) > 0 for s in mg.SITES),
           checked=checked)
    result("N2 padded rows/tokens (FULL-graph-style padding) not flagged",
           n_pad > 0 and not any(st["viol"] for st in steps if st["pad"]), padded_steps=n_pad)
    result("N3 masked selection entries / padding tokens with nsa=0 not flagged",
           not tot.get("mla_r.masked_live") and checked.get("mla_r", 0) > 0)
    result("N4 speculative slots later rejected not flagged",
           n_spec > 0 and n_rej > 0 and not any(st["viol"] for st in steps),
           spec_writes=n_spec, rejected_tokens=n_rej)
    result("N5 mixed prefill/decode steps, prefix hit, row reuse clean",
           n_mixed > 0 and not tot, mixed_steps=n_mixed)
    result("N6 high-address blocks (>= 935) checked and clean", high.get("mla_w", 0) > 0 and not tot,
           high=high)
    # A2 exact position join on the clean scenario (offline rule)
    p3 = am.p3_exact(sc["guard"].steps, sc["guard"].ring,
                     {st["fp"]: {r: [0, 0, v] for r, v in st["rej"].items()} for st in steps},
                     async_mode=True)
    result("A2a exact offline position join: clean scenario has 0 mismatches",
           p3["checked"] > 0 and p3["mismatch"] == 0, **{k: p3[k] for k in ("checked", "mismatch")})
    return sc


# ------------------------------------------------------------------ fault injections
def first_step(pred):
    st = {"done": False}

    def trig(emu, so, plan, s):
        if st["done"]:
            return []
        f = pred(emu, so, plan, s)
        if f:
            st["done"] = True
            return f
        return []
    return trig


def expect(name, sc, classes, flags_any, site_pred=None):
    inj = sc["injected"]
    steps = [st for st in sc["steps"] if st["faults"]]
    pre = [st for st in sc["steps"] if not st["faults"] and inj and st["step"] < inj[0][0]]
    tot = {}
    fl = set()
    for st in steps:
        for k, v in st["viol"].items():
            tot[k] = tot.get(k, 0) + v
        fl.update(st["flags"])
    ok_classes = all(any(k.endswith("." + c) or k == c for k in tot) for c in classes)
    ok_flags = any(f in fl for f in flags_any) if flags_any else True
    clean_before = not any(st["viol"] for st in pre)
    result(name, bool(inj) and ok_classes and ok_flags and clean_before,
           injected=inj[:2], violations=tot, flags=sorted(fl), clean_before=clean_before)


def test_faults():
    # F1 wrong-request block id in the persistent table (live block of another request)
    def p1(emu, so, plan, s):
        live = [r for r in so.num_scheduled_tokens if r in emu.row_of]
        dec = [r for r in live if so.scheduled_spec_decode_tokens.get(r)]
        if len(dec) >= 2:
            # victim's column at its current write position <- donor's first block
            return [Fault("wrong_req", s, victim=dec[0], donor=dec[1], col=None, donor_col=0)]
        return []
    expect("F1 wrong-request block id -> MLA/index-K writes and reads flagged (foreign)",
           scenario(first_step(p1)), ["mism"], ["foreign"])

    # F2 stale block after free/reuse: a reused runner row misses one append, so the column
    # still holds the previous occupant's (freed) block
    def p2(emu, so, plan, s):
        cr = so.scheduled_cached_reqs
        for i, rid in enumerate(cr.req_ids):
            nb = cr.new_block_ids[i]
            if nb is not None and len(nb[0]) and rid in ("E", "C", "B"):
                return [Fault("drop_append", s, rid=rid)]
        return []
    sc2 = scenario(first_step(p2), budget=1024)
    expect("F2 stale block id after free/reuse (missed append on a reused row) flagged",
           sc2, ["mism"], ["stale", "foreign", "null", "wgroup"])

    # F3a wrong logical position: one row's positions shifted by +1 (stale num_computed)
    def p3(emu, so, plan, s):
        pf = [r for r, pl in plan.items() if not pl["k"] and not pl["st"]["prefill_done"]]
        if pf:
            return [Fault("wrong_pos", s, rid=pf[0], delta=1)]
        return []
    expect("F3a wrong logical position (prefill row +1) -> position bounds flagged",
           scenario(first_step(p3)), ["pos_bounds"], ["posb"])

    # F3b wrong position inside the [lo, hi] window (decode row): only the exact offline join
    def p3b(emu, so, plan, s):
        dec = [r for r, pl in plan.items() if pl["k"] >= 2 and s > 8]
        if dec:
            return [Fault("wrong_pos", s, rid=dec[0], delta=-1)]
        return []
    sc3b = scenario(first_step(p3b))
    p3 = am.p3_exact(sc3b["guard"].steps, sc3b["guard"].ring,
                     {st["fp"]: {r: [0, 0, v] for r, v in st["rej"].items()} for st in sc3b["steps"]},
                     async_mode=True)
    result("F3b wrong position within the in-graph window -> exact offline join flags it",
           bool(sc3b["injected"]) and p3["mismatch"] > 0, injected=sc3b["injected"][:1],
           p3=dict(checked=p3["checked"], mismatch=p3["mismatch"], examples=p3["examples"][:2]))

    # F3c slot builder offset error (right block, wrong record)
    def p3c(emu, so, plan, s):
        if s > 3:
            return [Fault("offset_bug", s, rid=next(iter(so.num_scheduled_tokens)))]
        return []
    expect("F3c wrong record within the right block (offset error) flagged (ownoff)",
           scenario(first_step(p3c)), ["mism"], ["ownoff"])

    # F4 row-reuse error: the batch row reads the previous occupant's runner row
    state = {}

    def p4(emu, so, plan, s):
        if "freed_row" not in state:
            for rid in so.finished_req_ids:
                if rid in emu.row_of:
                    state["freed_row"] = emu.row_of[rid]
            return []
        live = [r for r in so.num_scheduled_tokens if r in emu.row_of
                and emu.row_of[r] != state["freed_row"]]
        if live:
            return [Fault("row_reuse", s, rid=live[0], stale_row=state["freed_row"])]
        return []
    expect("F4 row-reuse error (previous request's runner row) flagged",
           scenario(first_step(p4)), ["mism"], ["stale", "foreign", "null", "unexp"])

    # F5a wrong group ownership: a request's MLA column points at one of its own KDA blocks
    def p5(emu, so, plan, s):
        for rid in so.num_scheduled_tokens:
            if rid in emu.row_of and s > 2:
                kda = emu.mgr.get_block_ids(rid)[2]
                if kda:
                    return [Fault("wrong_group", s, victim=rid, col=0, block=int(kda[0]))]
        return []
    expect("F5a wrong group (MLA write into a KDA-group block) flagged (wgroup)",
           scenario(first_step(p5)), ["mism"], ["wgroup"])

    # F6 stale graph replay: the consumed slot buffer still holds the previous step's mapping
    def p6(emu, so, plan, s):
        if s > 4:
            return [Fault("stale_replay", s)]
        return []
    expect("F6 stale replay (previous step's slot mapping consumed) flagged",
           scenario(first_step(p6)), ["mism"], ["mism"])

    # F7 index-K pool completion dropped at an unaligned prefill chunk start (kernel predicate)
    expect_pred = scenario(chunk_override=lambda s: {"A": 2047} if s == 1 else None,
                           trigger=lambda emu, so, plan, s: [Fault("none", s)] if s == 2 else [])
    expect("F7 unaligned prefill chunk start: pool completion dropped (missing_pred) or pooled "
           "across the previous request (pool_cross) flagged", expect_pred,
           [], ["pred", "pcross"])

    # F8 padded token writes into a live block
    def p8(emu, so, plan, s):
        if all(so.scheduled_spec_decode_tokens.get(r) for r in so.num_scheduled_tokens) and \
                sum(so.num_scheduled_tokens.values()) % 8:
            blk = int(emu.mgr.get_block_ids(next(iter(so.num_scheduled_tokens)))[0][0])
            return [Fault("inact_write", s, block=blk)]
        return []
    expect("F8 inactive (padding) token writes a live block -> flagged (inact)",
           scenario(first_step(p8)), ["inact"], ["inact"])

    # F9 tail ring: generic mapping with a stale column (amendment 10d/10e T1 signature)
    def p9(emu, so, plan, s):
        pf = [r for r, pl in plan.items() if not pl["k"] and pl["nc"] == 0 and pl["n"] >= 16]
        others = [r for r in so.num_scheduled_tokens if r not in pf and r in emu.row_of]
        if pf and (others or s > 1):
            donor = others[0] if others else pf[0]
            blk = int(emu.mgr.get_block_ids(donor)[0][0]) if others else 3
            return [Fault("tail_t1", s, rid=pf[0], block=blk)]
        return []
    expect("F9 tail ring T1 signature (generic mapping, stale column) flagged",
           scenario(first_step(p9)), ["mism"], ["mism", "null", "unexp"])

    # F10 MLA reads: non-causal, foreign block, masked entry live
    def p10(emu, so, plan, s):
        if s > 3 and sum(so.num_scheduled_tokens.values()) > 2:
            req_ids = b0emu.sort_batch_req_ids(so.num_scheduled_tokens, so.scheduled_spec_decode_tokens, 4)
            own = set(emu.mgr.get_block_ids(req_ids[0])[0])
            for r in emu.row_of:
                cand = [b for b in emu.mgr.get_block_ids(r)[0] if b not in own]
                if r != req_ids[0] and cand:
                    return [Fault("sel_causal", s), Fault("sel_foreign", s, block=int(cand[0])),
                            Fault("masked_live", s)]
        return []
    sc10 = scenario(first_step(p10))
    expect("F10 MLA selected slots: non-causal / foreign / masked-live flagged", sc10,
           ["sel_causal", "sel_foreign", "masked_live"], ["selc", "self", "maskl"])

    # F11 consumed query_start_loc disagrees with the scheduler's row split
    def p11(emu, so, plan, s):
        if len(so.num_scheduled_tokens) >= 2:
            return [Fault("qsl_bad", s)]
        return []
    expect("F11 row split (query_start_loc) mismatch flagged", scenario(first_step(p11)),
           ["qsl"], ["qsl"])

    # F13 index-K read: wrong pool length / foreign page
    def p13(emu, so, plan, s):
        if all(so.scheduled_spec_decode_tokens.get(r) for r in so.num_scheduled_tokens) and \
                len(set(so.num_scheduled_tokens.values())) == 1 and len(so.num_scheduled_tokens) >= 2:
            other = list(so.num_scheduled_tokens)[1]
            blk = int(emu.mgr.get_block_ids(other)[0][0])
            return [Fault("idx_read_len", s, delta=1), Fault("idx_read_foreign", s, block=blk)]
        return []
    expect("F13 index-K read: wrong pool length and foreign page flagged",
           scenario(first_step(p13)), ["rlen", "rtable"], ["rlen", "rtable"])


def test_transmission_and_rank():
    # F12 transmission: the worker ledger misses one append (SchedulerOutput lost a block id)
    sc = scenario()
    g = sc["guard"]
    sched = {st["fp"]: st["dig"] for st in g.steps}
    bad = dict(sched)
    k = list(bad)[len(bad) // 2]
    bad[k] = "0" * 16
    tr_ok = am.transmission(g.steps, {fp: dict(dig=d) for fp, d in sched.items()})
    tr_bad = am.transmission(g.steps, {fp: dict(dig=d) for fp, d in bad.items()})
    result("A3 transmission join: identical digests -> 0 mismatches; one altered -> flagged",
           tr_ok["mismatch"] == 0 and tr_ok["joined"] == len(g.steps) and tr_bad["mismatch"] == 1,
           identical=tr_ok, altered=dict(mismatch=tr_bad["mismatch"]))
    # F5b rank ownership: writes from a layer owned by the other rank / a missing owned layer
    mla_layers = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
    good = {0: {L: 100 for L in mla_layers if L < 25}, 1: {L: 100 for L in mla_layers if L >= 25}}
    r_ok = am.rank_ownership(good, "split:25", mla_layers, armed_forwards={0: 100, 1: 100})
    wrong = {0: dict(good[0]), 1: dict(good[1])}
    wrong[1][23] = 5
    r_bad = am.rank_ownership(wrong, "split:25", mla_layers, armed_forwards={0: 100, 1: 100})
    missing = {0: dict(good[0]), 1: dict(good[1])}
    missing[0].pop(11)
    r_miss = am.rank_ownership(missing, "split:25", mla_layers, armed_forwards={0: 100, 1: 100})
    result("F5b rank ownership: a write from a non-owned layer and a missing owned layer flagged",
           r_ok["ok"] and not r_bad["ok"] and not r_miss["ok"],
           good_ok=r_ok["ok"], wrong=r_bad["problems"][:2], missing=r_miss["problems"][:2])


# ------------------------------------------------------------------ lints
class SyncTrap(torch.overrides.TorchFunctionMode):
    BAD = {"item", "tolist", "nonzero", "masked_select", "cpu", "numpy", "__bool__", "unique",
           "__int__", "__float__", "__index__", "argwhere", "repeat_interleave_cpu"}

    def __init__(self):
        super().__init__()
        self.hits = []
        self.writes = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", str(func))
        if name in self.BAD:
            self.hits.append(name)
        if name == "__getitem__" and len(args) > 1:
            idx = args[1]
            if isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
                self.hits.append("bool_index")
        return func(*args, **(kwargs or {}))


def test_indexer_geometry():
    """A6 (post-inspection, run-a2): the emulation's index-K read table is the real builder's
    (factor and storage derived by B0 code), and the guard's page geometry agrees."""
    geo = mg.Geo(nb=b0emu.NB, R=b0emu.R, W0=b0emu.W0)
    result("A6 indexer read geometry from B0 code: factor 4, 64-pool pages, 17 per block; guard agrees",
           (b0emu.IDX_FACTOR, b0emu.IDX_STORAGE) == (4, 64) and geo.ipage == b0emu.IDX_STORAGE
           and geo.ipb == b0emu.PPB // b0emu.IDX_STORAGE,
           factor=b0emu.IDX_FACTOR, storage=b0emu.IDX_STORAGE, guard_ipage=geo.ipage, guard_ipb=geo.ipb)


def test_lints(sc):
    """G1: the device checks use no host-sync op (graph-capture safe by construction).
       G2: hooks never modify an engine tensor (observe-only)."""
    emu, g = Emu(seed=7, concurrency=3), Guard()
    emu.add([760000 + i for i in range(5000)], 6, rid="X")
    emu.add([770000 + i for i in range(300)], 8, rid="Y")
    hits, mutated = [], []
    for s in range(1, 9):
        so, plan = emu.schedule()
        if not so.num_scheduled_tokens:
            continue
        g.begin(so)
        out = emu.runner_step(so)
        g.upload(out["req_ids"])
        engine = {k: v.clone() for k, v in out.items() if isinstance(v, torch.Tensor)}
        engine.update({f"sel{i}": t.clone() for i, t in enumerate(out["sel"])})
        if "dec" in out:
            engine.update({f"dec{i}": t.clone() for i, t in enumerate(out["dec"])})
        trap = SyncTrap()
        g.run_checks(out, trap=trap)
        hits += trap.hits
        for k, v in out.items():
            if isinstance(v, torch.Tensor) and not torch.equal(v, engine[k]):
                mutated.append((s, k))
        for i, t in enumerate(out["sel"]):
            if not torch.equal(t, engine[f"sel{i}"]):
                mutated.append((s, f"sel{i}"))
        accepted = {r: 1 for r, pl in plan.items() if pl["k"] or (pl["st"]["prefill_done"] and not pl["st"].get("first_decode"))}
        emu.commit(out, accepted, plan)
        emu.complete(plan, accepted)
    result("G1 graph safety: no host-sync op inside any device check (.item/.tolist/nonzero/bool-index/...)",
           not hits, hits=sorted(set(hits))[:10])
    result("G2 observe-only: no consumed engine tensor modified by any hook", not mutated,
           mutated=mutated[:5])


def test_verdicts():
    cov_ok = {k: v["min"] for k, v in am.COVERAGE.items()}
    v_clean = am.verdict(violations={}, invalid=[], coverage=cov_ok)
    v_inc = am.verdict(violations={}, invalid=[], coverage={**cov_ok, "C7_high_mla_w": 0})
    v_inv = am.verdict(violations={}, invalid=["eager fallback"], coverage=cov_ok)
    v_vio = am.verdict(violations={"mla_w.mism": 3}, invalid=["eager fallback"], coverage=cov_ok)
    result("A4/A5 verdicts: CLEAN / INCOMPLETE (named condition) / INVALID / VIOLATION precedence",
           v_clean["verdict"] == "CLEAN" and v_inc["verdict"] == "INCOMPLETE"
           and "C7_high_mla_w" in v_inc["incomplete"] and v_inv["verdict"] == "INVALID"
           and v_vio["verdict"] == "VIOLATION",
           clean=v_clean["verdict"], inc=v_inc["incomplete"], inv=v_inv["verdict"], vio=v_vio["verdict"])


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "cpu-results.json")
    t0 = time.time()
    for fn in (test_indexer_geometry, test_negative_controls, test_faults, test_transmission_and_rank,
               test_verdicts):
        try:
            r = fn()
            if fn is test_negative_controls:
                sc = r
        except Exception:  # noqa: BLE001
            result(fn.__name__ + " crashed", False, tb=traceback.format_exc()[-1500:])
            sc = None
    try:
        test_lints(sc)
    except Exception:  # noqa: BLE001
        result("test_lints crashed", False, tb=traceback.format_exc()[-1500:])
    npass = sum(r["ok"] for r in RESULTS)
    summary = dict(module=SRC, module_sha256=__import__("hashlib").sha256(
        open(mg.__file__, "rb").read()).hexdigest(), passed=npass, total=len(RESULTS),
        secs=round(time.time() - t0, 1), results=RESULTS)
    json.dump(summary, open(out, "w"), indent=1, default=str)
    print(f"{npass}/{len(RESULTS)} passed in {summary['secs']} s -> {out}")
    sys.exit(0 if npass == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
