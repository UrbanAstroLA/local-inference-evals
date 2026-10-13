#!/usr/bin/env python3
"""CPU tests of the B1 guard amendment (PLAN.md amendment 18a, step 2a).  NO GPU; Triton interpreter.

Drives the INSTALLED, hooked index-K op of the B1 guard image (B1's real call site: D7 ring resolution,
the amended pre-write hook, B1's writer, the post-write hook, the tail seed) step by step, as
d7/mechanism/tests/test_d7_fix.py does, with the guard's expectation buffer E filled per step from the
test's schedule (request tag, chunk start = scheduler num_computed, scheduled counts, block lists).
Faults are injected only into the WRITER's inputs or the engine's tail ring (never into the guard's
inputs), at B1's code path.

  N*  negative controls (must not be flagged): unaligned starts repaired by B1 (residues 1-3, first
      and later rows, successive boundaries, block boundary), aligned pools, spec decode rows ahead
      of the prefill slice, padding tokens; positive control clean; observe-only
  O*  oracle: straddling records equal an independent plain-torch pooling of the request's own
      tokens by logical position (d7lib.ref_record, interpreter-matched conversions)
  F*  faults that must be flagged: (1) missing repaired pool, (2) foreign keys (another request's
      ring), (3) stale keys from the same request (old ring contents / earlier position),
      (4) incorrect logical positions, (5) no-ring fallback (call without ring, env switch,
      row without tail slot) incl. B0's cross-request pooling, (6) B0's writer itself
  P*  positive control: an unfaithful oracle is caught on aligned pools (pc_mism) -> INVALID
  U*  shadow unavailable (not armed / slot collision) -> s_unavail, never a violation
  G*  graph-safety lint of the new hooks, observe-only (byte-identical caches hooks on vs off)
  A*  amended analysis rules (exercised-event count, positive-control rule, fallback, verdicts)

usage: test_d7_guard.py OUT.json
"""
import ast
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace as NS

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "guard"))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.Tensor.pin_memory = lambda self, *a, **k: self
import vllm.model_executor.layers.sparse_attn_indexer  # noqa: E402,F401
import vllm.models.glm5next.nvidia.ops.kpool_compress as kc  # noqa: E402
from vllm.model_executor.layers import sparse_attn_indexer_kpool as kp  # noqa: E402
from vllm.v1.attention.backends.mla.indexer import (  # noqa: E402
    DeepseekV32IndexerMetadata, compute_kpool_tail_slot_mapping)

try:
    from vllm import mapguard_hooks as mg  # the B1 guard image
    SRC = "vllm.mapguard_hooks (image)"
except Exception:  # noqa: BLE001
    import mapguard_hooks as mg  # noqa: E402
    SRC = "guard/mapguard_hooks.py"
import analyze_mapguard as am  # noqa: E402
from d7lib import HD, KP, make_caches, rec, ref_record  # noqa: E402

RING, PPB = 8, 1088
LAYER = 3
PFX = f"model.layers.{LAYER}.self_attn.indexer.k_cache"
TPFX = f"model.layers.{LAYER}.self_attn.indexer.tail_cache"
APE = (torch.randn(KP, HD, generator=torch.Generator().manual_seed(23)) * 0.1).float()
KP_SRC = open(kp.__file__).read()
HOOKED = "hook_idx_prefill_post" in KP_SRC and "local-d7" in KP_SRC
kp._mapguard = mg
kp._MAPGUARD_ON = True
mg.ENABLED = True
REAL_INSERT = kp._kpool_compress_insert
RESULTS = []
PF = mg.S_IDX_W_PF
VIOL_PF = ["mism", "missing", "missing_pred", "pool_cross", "unexp", "inact", "shared", "s_missing",
           "s_mism", "d7_fallback"]


def result(name, ok, **kw):
    RESULTS.append(dict(test=name, ok=bool(ok), **kw))
    print(("PASS " if ok else "FAIL ") + name + "  " + json.dumps(kw, default=str)[:500], flush=True)


class Req:
    tags = [100]

    def __init__(self, name, n, idx_blocks, tail_block, seed, tag=None):
        g = torch.Generator().manual_seed(seed)
        self.name, self.n = name, n
        self.k = torch.randn(n, HD, generator=g).to(torch.bfloat16)
        self.g = torch.randn(n, HD, generator=g).to(torch.bfloat16)
        self.idx_blocks, self.tail_block = list(idx_blocks), tail_block
        if tag is None:
            tag = Req.tags[0]
            Req.tags[0] += 1
        self.tag = tag
        self.done = set()

    def loc(self, pool):
        return self.idx_blocks[pool // PPB] * PPB + pool % PPB

    def slot(self, p):
        return self.loc(p // KP) if (p + 1) % KP == 0 else -1

    def ref(self, pool):
        return ref_record(self.k[pool * KP: pool * KP + KP], self.g[pool * KP: pool * KP + KP], APE)


class Abort(Exception):
    pass


class StopBuf:
    shape = (1, 2048)

    def __setitem__(self, *_):
        raise Abort()


class World:
    def __init__(self, nb=48):
        self.raw, self.tail, self.idx = make_caches(nb, RING)
        self.reqs = []
        self.geo = mg.Geo(nb=nb, R=16, W0=4)
        self.dev = mg.Dev(self.geo, torch.device("cpu"))
        mg.S.dev, mg.S.geo, mg.S.rank, mg.S.d7_pending = self.dev, self.geo, 0, None
        self.step_no = 0
        self.straddles = []          # (req, pool) expected straddling pools of armed steps

    def add(self, r):
        self.reqs.append(r)
        return r

    def arm(self, rows):
        lay = self.dev.lay
        a = np.zeros(lay.size, np.int64)
        v = lay.np_views(a)
        q = 0
        for i, (r, s, n, nspec) in enumerate(rows):
            q += n
            v["qsl"][i + 1] = q
            v["tag"][i] = r.tag
            v["lo"][i] = v["hi"][i] = s
            v["nspec"][i], v["nsched"][i] = nspec, n
            v["bt0"][i, :len(r.idx_blocks)] = r.idx_blocks
            v["nb0"][i] = len(r.idx_blocks)
            v["bt1"][i] = r.tail_block
        v["qsl"][len(rows) + 1:] = q
        v["col0"][:] = -1
        v["grp"][:] = -1
        for r in self.reqs:
            for c, b in enumerate(r.idx_blocks):
                v["col0"][b], v["tagx"][b], v["grp"][b] = c, r.tag, 0
            v["grp"][r.tail_block], v["tagx"][r.tail_block] = 1, r.tag
        h = v["hdr"]
        h[mg.H_ARMED], h[mg.H_STEP], h[mg.H_NROWS], h[mg.H_NTOK] = 1, self.step_no, len(rows), q
        self.dev.E.copy_(torch.from_numpy(a))

    def step(self, segs, decode=(), pad=0, tail=True, armed=True, null_tail=(), prestep=None):
        """segs: [(req, start, n)] prefill rows in batch order; decode: [(req, start)] spec decode rows
        (4 tokens, 3 drafts) placed first; pad: trailing padding tokens (slot -1, no tail slot);
        null_tail: requests whose row gets the null tail block (no ring slot) this step."""
        self.step_no += 1
        rows = [(r, s, 4, 3) for r, s in decode] + [(r, s, n, 0) for r, s, n in segs]
        pos = torch.cat([torch.arange(s, s + n) for r, s, n, _ in rows]).to(torch.int64)
        k = torch.cat([r.k[s:s + n] for r, s, n, _ in rows])
        g = torch.cat([r.g[s:s + n] for r, s, n, _ in rows])
        sm = torch.tensor([r.slot(int(p)) for (r, s, n, _) in rows for p in range(s, s + n)], dtype=torch.int64)
        qsl = torch.tensor([0] + list(np.cumsum([n for _, _, n, _ in rows])), dtype=torch.int32)
        bt = torch.zeros(len(rows), 4, dtype=torch.int32)
        bt[:, 0] = torch.tensor([0 if r in null_tail else r.tail_block for r, _, _, _ in rows], dtype=torch.int32)
        tsm = compute_kpool_tail_slot_mapping(torch.full((len(pos),), -1, dtype=torch.int64), bt, qsl, pos,
                                              len(pos), len(rows), RING)
        if pad:
            pg = torch.Generator().manual_seed(999)
            pos = torch.cat([pos, torch.zeros(pad, dtype=torch.int64)])
            k = torch.cat([k, torch.randn(pad, HD, generator=pg).to(torch.bfloat16)])
            g = torch.cat([g, torch.randn(pad, HD, generator=pg).to(torch.bfloat16)])
            sm = torch.cat([sm, torch.full((pad,), -1, dtype=torch.int64)])
            tsm = torch.cat([tsm, torch.full((pad,), -1, dtype=torch.int64)])
        ndt = 4 * len(decode)
        md = dict(seq_lens=torch.tensor([s + n for _, s, n, _ in rows], dtype=torch.int32), max_seq_len=0,
                  num_decodes=len(decode), num_decode_tokens=ndt, num_prefills=len(segs),
                  num_prefill_tokens=len(pos) - ndt)
        meta = {PFX: DeepseekV32IndexerMetadata(slot_mapping=sm, **md)}
        if tail:
            meta[TPFX] = DeepseekV32IndexerMetadata(slot_mapping=tsm, **md)
        kp.get_forward_context = lambda: NS(attn_metadata=meta)
        if armed:
            self.arm(rows)
            for r, s, n in segs:
                for p in range(s, s + n):
                    if (p + 1) % KP == 0 and p - 3 < s:
                        self.straddles.append((r, p // KP))
        else:
            self.dev.E[mg.H_ARMED] = 0
        if prestep:
            prestep(self)
        try:
            kp.sparse_attn_indexer_kpool(
                torch.zeros(len(pos), 1), PFX, self.idx, torch.zeros(len(pos), 32, HD), None, k, None, 128,
                "ue8m0", 2048, HD, 1 << 20, 1 << 16, StopBuf(), False, False, gate_score=g, compress_ape=APE,
                index_kpool=KP, positions=pos, tail_kv_cache=self.tail if tail else None,
                tail_prefix=TPFX if tail else None)
        except Abort:
            pass
        self.dev.E[mg.H_ARMED] = 0               # disarm after the forward (execute_model wrapper)
        for r, s, n in segs:
            r.done.update(range(s, s + n))

    def counters(self):
        c = self.dev.cnt[PF]
        return {n: int(c[mg.CI[n]]) for n in mg.CNT if int(c[mg.CI[n]])}

    def viol(self):
        c = self.counters()
        return {n: c[n] for n in VIOL_PF if c.get(n)}

    def slog(self):
        n = min(int(self.dev.nslog[0]), mg.SCAP)
        return [dict(zip(mg.SLOG_COLS, r)) for r in self.dev.slog[:n].tolist()]

    def bad_pools(self, reqs=None):
        out = {}
        for r in (reqs or self.reqs):
            bad = []
            for pool in range(r.n // KP):
                if all(p in r.done for p in range(pool * KP, pool * KP + KP)) and \
                        not torch.equal(rec(self.raw, r.loc(pool)), r.ref(pool)):
                    bad.append(pool)
            out[r.name] = bad
        return out


def sched_n1(w, decode_req=None):
    """Unaligned starts of every residue, first and later rows, successive boundaries, a block boundary,
    spec decode rows ahead of the prefill slice, padding tokens."""
    A = w.add(Req("A", 4200, [1, 2], 30, seed=1))
    B = w.add(Req("B", 2600, [3], 31, seed=2))
    C = w.add(Req("C", 600, [4], 32, seed=3))
    D = w.add(Req("D", 4096, [5], 33, seed=4))
    E = w.add(Req("E", 5000, [6, 7], 34, seed=5))
    w.step([(A, 0, 2045), (D, 0, 2000)])                                   # D: later row, aligned
    w.step([(A, 2045, 2045), (B, 0, 2047)], decode=[(D, 2000)], pad=5)    # A r1 first row
    w.step([(B, 2047, 553), (A, 4090, 110), (C, 0, 301)], decode=[(D, 2004)])   # B r3 first, A r2 later
    w.step([(E, 0, 4350), (C, 301, 299)], pad=3)                          # C r1 later
    w.step([(E, 4350, 650)], decode=[(D, 2008)])                          # E r2 across the 4352 boundary
    return dict(A=A, B=B, C=C, D=D, E=E)


def unaligned_pair(w, prestep=None, wrap=None, tail2=True, null_tail=(), second_later=False):
    """A: chunk 0..2045 then 2045..2600 (pool 511 straddles); B (seeded earlier) present as a donor."""
    A = w.add(Req("A", 2600, [1], 30, seed=11))
    B = w.add(Req("B", 1200, [2], 31, seed=12))
    w.step([(A, 0, 2045), (B, 0, 1000)])
    if wrap is not None:
        kp._kpool_compress_insert = wrap
    try:
        if second_later:
            w.step([(B, 1000, 200), (A, 2045, 555)], tail=tail2, null_tail=null_tail, prestep=prestep)
        else:
            w.step([(A, 2045, 555)], tail=tail2, null_tail=null_tail, prestep=prestep)
    finally:
        kp._kpool_compress_insert = REAL_INSERT
    return A, B


# ------------------------------------------------------------------ N / O: clean B1 behaviour
def test_negative_and_oracle():
    w = World()
    t0 = time.time()
    R = sched_n1(w)
    v, c = w.viol(), w.counters()
    n_str = len(w.straddles)
    sl = w.slog()
    bad = w.bad_pools()
    result("N0 hooked B1 call site installed (pre + post hooks, D7 writer)", HOOKED, kp_file=kp.__file__)
    result("N1 B1 unaligned starts (residues 1-3, first/later rows, successive chunks, 4352 block "
           "boundary, decode rows ahead, padding): zero violations at idx_w_pf",
           not v and mg.S.errors == 0, violations=v, guard_errors=mg.S.errors, secs=round(time.time() - t0, 1))
    result("N2 every straddling pool exercised and byte-verified (s_events == s_verified == expected)",
           n_str == 5 and c.get("s_events") == n_str and c.get("s_verified") == n_str
           and not c.get("s_unavail") and not c.get("s_indet"),
           expected=n_str, counters={k: c.get(k) for k in ("s_events", "s_verified", "s_indet", "s_unavail")})
    result("N3 positive control: every aligned completing pool recomputed and byte-identical",
           c.get("pc_checked", 0) > 1000 and not c.get("pc_mism"),
           pc_checked=c.get("pc_checked"), pc_mism=c.get("pc_mism", 0))
    result("N4 straddling-event log: one 'verified' row per straddling pool, positions as scheduled",
           sorted((x["tag"], x["position"]) for x in sl) == sorted((r.tag, p * KP + 3) for r, p in w.straddles)
           and all(x["outcome"] == mg.D7_OUT["verified"] and x["layer"] == LAYER for x in sl),
           rows=len(sl))
    result("N5 spec decode rows / padding tokens: no D7 outcome, no slot flag (checked tokens = real prefill)",
           all(x["tag"] != R["D"].tag for x in sl) and not v)
    # O1: independent plain-torch reference of the request's OWN tokens by logical position
    ok_ref = all(torch.equal(rec(w.raw, r.loc(p)), r.ref(p)) for r, p in w.straddles)
    result("O1 straddling records == independent plain-torch pooling of own tokens (interp conversions)",
           ok_ref and not any(bad.values()), straddles=len(w.straddles), bad=bad)
    # O2: the oracle path itself (shadow + batch members, compress kernel into scratch) reproduces the
    # reference: recompute every straddling pool through mg._oracle_compress from the request arrays
    scr = mg._scratch(w.dev, w.idx)
    ok2 = True
    for r, p in w.straddles:
        mk = r.k[p * KP:p * KP + KP][None]
        mgt = r.g[p * KP:p * KP + KP][None]
        mg._oracle_compress(scr, mk, mgt, APE, torch.zeros(1, dtype=torch.int64), KP,
                            torch.ones(1, dtype=torch.bool), True)
        ok2 &= torch.equal(mg._rec_bytes(scr, torch.zeros(1, dtype=torch.int64))[0], r.ref(p))
    result("O2 oracle compress (engine kernel, interpreter) == plain-torch reference on straddling members", ok2)
    return w


def test_aligned_only():
    w = World()
    A = w.add(Req("A", 3000, [1], 30, seed=21))
    B = w.add(Req("B", 1500, [2], 31, seed=22))
    w.step([(A, 0, 2044), (B, 0, 1000)])
    w.step([(A, 2044, 956), (B, 1000, 500)], pad=2)
    c = w.counters()
    result("N6 aligned-only schedule: no straddling event, positive control clean, no violation",
           not c.get("s_events") and c.get("pc_checked", 0) > 0 and not c.get("pc_mism") and not w.viol(),
           counters=c)


# ------------------------------------------------------------------ F: faults (writer inputs / ring)
def _fault(name, sc_kwargs, expect_any, expect_none=(), label=""):
    w = World()
    A, B = unaligned_pair(w, **sc_kwargs(w) if callable(sc_kwargs) else sc_kwargs)
    c, v = w.counters(), w.viol()
    sl = w.slog()
    ok = all(c.get(x, 0) > 0 for x in expect_any) and not any(c.get(x, 0) for x in expect_none)
    recs = int(w.dev.nrec[0])
    flags = 0
    for r in w.dev.rec[:min(recs, mg.CAP)].tolist():
        flags |= int(r[mg.REC_COLS.index("flags")])
    result(name, ok and recs > 0, violations=v, s=dict((k, c.get(k)) for k in
                                                     ("s_events", "s_verified", "s_missing", "s_mism", "s_unavail")),
           flags=mg.flag_names(flags), outcomes=sorted({x["outcome"] for x in sl}), n_bsnap=int(w.dev.nbsnap[0]))
    return w


def test_faults():
    def restore_wrap(w):
        A_loc = []

        def wrap(*a, **kw):
            loc = 1 * PPB + 511                         # A's pool 511 (positions 2044-2047)
            before = rec(w.raw, loc)
            REAL_INSERT(*a, **kw)
            pg, r = loc // 64, loc % 64
            base = pg * 8448
            w.raw[base + r * 128: base + r * 128 + 128] = before[:128]
            w.raw[base + 8192 + r * 4: base + 8192 + r * 4 + 4] = before[128:]
            A_loc.append(loc)
        return dict(wrap=wrap)
    _fault("F1 missing repaired pool (writer skips the straddling write) -> s_missing",
           restore_wrap, ["s_missing"], ["s_verified", "d7_fallback"])

    def foreign_wrap(w):
        def wrap(*a, **kw):
            t = kw["tail_slot_mapping"].clone()
            blk = torch.div(t.clamp(min=0), RING, rounding_mode="floor")
            kw["tail_slot_mapping"] = torch.where((t >= 0) & (blk == 30), 31 * RING + torch.remainder(t, RING), t)
            REAL_INSERT(*a, **kw)
        return dict(wrap=wrap)
    _fault("F2 foreign keys (writer reads another request's tail ring) -> s_mism",
           foreign_wrap, ["s_mism"], ["s_verified"])

    def stale_ring(w):
        def pre(world):
            A = world.reqs[0]
            for q in (2044,):                           # straddling member of pool 511 (start 2045)
                world.tail[30, 0, q % RING] = A.k[q - RING]
                world.tail[30, 1, q % RING] = A.g[q - RING]
        return dict(prestep=pre)
    _fault("F3a stale keys, same request (old ring contents, position - 8) -> s_mism",
           stale_ring, ["s_mism"], ["s_verified"])

    def stale_pos(w):
        def pre(world):
            A = world.reqs[0]
            world.tail[30, 0, 2044 % RING] = A.k[2043]
            world.tail[30, 1, 2044 % RING] = A.g[2043]
        return dict(prestep=pre)
    _fault("F3b stale keys, same request (earlier position's key in the member's slot) -> s_mism",
           stale_pos, ["s_mism"], ["s_verified"])

    for d in (1, -1, 4):
        def pos_wrap(w, d=d):
            def wrap(*a, **kw):
                kw["positions"] = kw["positions"] + d
                REAL_INSERT(*a, **kw)
            return dict(wrap=wrap)
        _fault(f"F4 incorrect logical positions given to the writer (shift {d:+d}) -> s_mism/s_missing",
               pos_wrap, [], ["s_verified"])
        c = RESULTS[-1]["s"]
        RESULTS[-1]["ok"] = RESULTS[-1]["ok"] and ((c.get("s_mism") or 0) + (c.get("s_missing") or 0)) > 0

    _fault("F5a no ring for the call (first row) -> d7_fallback + missing_pred + s_missing",
           dict(tail2=False), ["d7_fallback", "missing_pred", "s_missing"])
    _fault("F5b no ring for the call (later row: B0 cross-request pooling) -> d7_fallback + pool_cross + s_mism",
           dict(tail2=False, second_later=True), ["d7_fallback", "pool_cross", "s_mism"])
    os.environ["VLLM_KPOOL_SKIP_TAIL_CACHE"] = "1"
    try:
        _fault("F5c VLLM_KPOOL_SKIP_TAIL_CACHE=1 at the call site -> d7_fallback + s_missing",
               dict(), ["d7_fallback", "s_missing"])
    finally:
        os.environ.pop("VLLM_KPOOL_SKIP_TAIL_CACHE", None)
    w = World()
    A = w.add(Req("A", 2600, [1], 30, seed=11))
    w.step([(A, 0, 2045)])
    w.step([(A, 2045, 555)], null_tail=(A,))
    c = w.counters()
    result("F5d row without a tail slot (null tail block) -> d7_fallback (row-level B0 rule)",
           c.get("d7_fallback", 0) > 0 and c.get("missing_pred", 0) > 0, counters={k: c.get(k) for k in
                                                                                  ("d7_fallback", "missing_pred", "s_missing", "s_mism")})

    src = open(os.path.join(HERE, "b0_kpool.py")).read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_kpool_compress_insert")
    ns = {"torch": torch, "kpool_compress_and_write_cache": kc.kpool_compress_and_write_cache}
    exec(compile(ast.Module([fn], []), "b0_kpool.py", "exec"), ns)
    b0_insert = ns["_kpool_compress_insert"]

    def b0_wrap(*a, **kw):
        for x in ("positions", "tail_kv_cache", "tail_slot_mapping"):
            kw.pop(x, None)
        b0_insert(*a, **kw)
    _fault("F6a B0's writer behind B1's call site (ring passed, first row) -> s_missing by bytes alone",
           dict(wrap=b0_wrap), ["s_missing"], ["d7_fallback", "missing_pred", "s_verified"])
    _fault("F6b B0's writer behind B1's call site (later row) -> s_mism (cross-request) by bytes alone",
           dict(wrap=b0_wrap, second_later=True), ["s_mism"], ["d7_fallback", "pool_cross"])


def test_positive_control():
    real = mg._oracle_compress

    def unfaithful(scratch, slot_k, slot_score, ape, loc, kpool, mask, rs):
        real(scratch, slot_k, slot_score, ape * 1.01, loc, kpool, mask, rs)
    mg._oracle_compress = unfaithful
    try:
        w = World()
        A, B = unaligned_pair(w)
    finally:
        mg._oracle_compress = real
    c = w.counters()
    win = {"idx_w_pf": c}
    d7 = am.d7_assess({"idx_w_pf": {k: c.get(k, 0) for k in mg.CNT}}, rank=0)
    result("P1 unfaithful oracle (ape x1.01): aligned pools disagree (pc_mism) -> byte checks INVALID, "
           "straddling mismatches NOT counted as violations",
           c.get("pc_mism", 0) > 0 and d7["invalid"] and not any(k.endswith(("s_mism", "s_missing")) for k in d7["violations"]),
           pc_mism=c.get("pc_mism"), s_mism=c.get("s_mism"), assess=d7, win=win)


def test_unavailable():
    w = World()
    A = w.add(Req("A", 2600, [1], 30, seed=31))
    w.step([(A, 0, 2045)], armed=False)          # previous chunk not seen by the guard
    w.step([(A, 2045, 555)])
    c = w.counters()
    result("U1 straddling member not in the shadow (previous chunk unarmed) -> s_unavail, no violation",
           c.get("s_unavail") == 1 and not w.viol() and not c.get("s_verified"), counters=c)
    w = World()
    A = w.add(Req("A", 2600, [1], 30, seed=32, tag=500))
    X = w.add(Req("X", 2600, [2], 31, seed=33, tag=500 + mg.SH))   # same shadow slot
    w.step([(A, 0, 2045), (X, 0, 2045)])
    w.step([(A, 2045, 555)])
    c = w.counters()
    result("U2 shadow slot collision (tags equal mod SH): tag check -> s_unavail, never a false violation",
           c.get("s_unavail") == 1 and not w.viol(), counters=c)


# ------------------------------------------------------------------ G: lints
class SyncTrap(torch.overrides.TorchFunctionMode):
    BAD = {"item", "tolist", "nonzero", "masked_select", "cpu", "numpy", "__bool__", "unique",
           "__int__", "__float__", "__index__", "argwhere"}

    def __init__(self):
        super().__init__()
        self.hits = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", str(func))
        if name in self.BAD:
            self.hits.append(name)
        if name == "__getitem__" and len(args) > 1 and isinstance(args[1], torch.Tensor) \
                and args[1].dtype == torch.bool:
            self.hits.append("bool_index")
        return func(*args, **(kwargs or {}))


def test_lints():
    hits = []
    real = {n: getattr(mg, n) for n in ("hook_idx_prefill_write", "hook_idx_prefill_post", "hook_tail_seed")}
    real_oracle = mg._oracle_compress

    def trapped(fn):
        def f(*a, **kw):
            t = SyncTrap()
            with t:
                fn(*a, **kw)
            hits.extend(t.hits)
        return f
    proxy = NS(**{n: getattr(mg, n) for n in dir(mg) if not n.startswith("__")})
    for n, fn in real.items():
        setattr(proxy, n, trapped(fn))
    # the oracle is an engine-kernel launch (asynchronous on GPU); under the interpreter it runs on numpy,
    # so it is replaced by a stub for the lint and checked separately (N3/O2)
    mg._oracle_compress = lambda *a, **k: None
    kp._mapguard = proxy
    try:
        w = World()
        unaligned_pair(w, second_later=True)
    finally:
        kp._mapguard = mg
        mg._oracle_compress = real_oracle
    result("G1 new hooks: no host-sync op (.item/.tolist/nonzero/bool-index/cpu/numpy/...)", not hits,
           hits=sorted(set(hits))[:10])
    # G2 observe-only: identical caches and ring with hooks on and off
    w_on = World()
    unaligned_pair(w_on, second_later=True)
    kp._MAPGUARD_ON = False
    try:
        Req.tags = [100]
        w_off = World()
        unaligned_pair(w_off, second_later=True)
    finally:
        kp._MAPGUARD_ON = True
    result("G2 observe-only: index-K cache and tail ring byte-identical with the hooks on and off "
           "(unaligned first and later rows)",
           torch.equal(w_on.raw, w_off.raw), nbytes=int(w_on.raw.numel()))


# ------------------------------------------------------------------ A: amended analysis rules
def test_analysis_rules():
    owned = [3, 7, 11]
    rows = []
    for ev in range(120):
        for L in owned:
            rows.append(dict(step=10 + ev, layer=L, tag=5, position=2047 + 4352 * ev, outcome=1, rank=0, row=0, slot=0))
    rows.append(dict(step=500, layer=3, tag=6, position=2047, outcome=1, rank=0, row=0, slot=0))   # one layer only
    rows += [dict(step=501, layer=L, tag=7, position=2047, outcome=2, rank=0, row=0, slot=0) for L in owned]
    ev = am.d7_events(rows, owned, 0, 10 ** 9)
    result("A6 exercised straddling events: counted once per (step, request, position), only when every "
           "owned layer verified it (indeterminate layers allowed if one verified)",
           ev["exercised"] == 120 and ev["events"] == 122, ev={k: ev[k] for k in ("events", "exercised", "by_outcome")})
    base = {k: 0 for k in mg.CNT}
    ok = am.d7_assess({"idx_w_pf": dict(base, pc_checked=50000, s_verified=900)}, rank=0)
    bad = am.d7_assess({"idx_w_pf": dict(base, pc_checked=50000, s_mism=1)}, rank=0)
    pcf = am.d7_assess({"idx_w_pf": dict(base, pc_checked=50000, pc_mism=2, s_mism=1)}, rank=0)
    fb = am.d7_assess({"idx_w_pf": dict(base, pc_checked=50000, pc_mism=2, d7_fallback=1)}, rank=0)
    leg = am.d7_assess({"idx_w_pf": dict(base, pc_checked=50000, legacy=3)}, rank=0)
    result("A7 positive-control rule: s_mism with a clean control -> VIOLATION; with pc_mism > 0 -> INVALID "
           "(not violation); d7_fallback stays a VIOLATION; legacy hook calls -> INVALID",
           not ok["violations"] and not ok["invalid"] and bad["violations"] and not bad["invalid"]
           and pcf["invalid"] and not pcf["violations"] and fb["violations"] and leg["invalid"],
           clean=ok, bad=bad, pcf=pcf, fb=fb, leg=leg)
    cov_ok = {k: v["min"] for k, v in am.COVERAGE.items()}
    v_inc = am.verdict(violations={}, invalid=[], coverage={**cov_ok, "C13_d7_straddle_events": 99})
    v_pc = am.verdict(violations={}, invalid=[], coverage={**cov_ok, "C14_d7_positive_control": 0})
    result("A8 new coverage minimums: < 100 exercised straddling events or < 10,000 positive-control pools "
           "-> INCOMPLETE (named)", v_inc["verdict"] == "INCOMPLETE" and "C13_d7_straddle_events" in v_inc["incomplete"]
           and v_pc["verdict"] == "INCOMPLETE" and am.COVERAGE["C13_d7_straddle_events"]["min"] == 100,
           inc=v_inc["incomplete"], pc=v_pc["incomplete"])


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "d7-guard.json")
    t0 = time.time()
    for fn in (test_negative_and_oracle, test_aligned_only, test_faults, test_positive_control,
               test_unavailable, test_lints, test_analysis_rules):
        try:
            fn()
        except Exception:  # noqa: BLE001
            result(fn.__name__ + " crashed", False, tb=traceback.format_exc()[-2000:])
    npass = sum(r["ok"] for r in RESULTS)
    summary = dict(module=SRC, module_sha256=__import__("hashlib").sha256(open(mg.__file__, "rb").read()).hexdigest(),
                   kpool_file=kp.__file__, kpool_sha256=__import__("hashlib").sha256(KP_SRC.encode()).hexdigest(),
                   guard_errors=mg.S.errors, passed=npass, total=len(RESULTS), secs=round(time.time() - t0, 1),
                   results=RESULTS)
    json.dump(summary, open(out, "w"), indent=1, default=str)
    print(f"{npass}/{len(RESULTS)} passed in {summary['secs']} s, guard errors {mg.S.errors} -> {out}")
    sys.exit(0 if npass == len(RESULTS) and mg.S.errors == 0 else 1)


if __name__ == "__main__":
    main()
