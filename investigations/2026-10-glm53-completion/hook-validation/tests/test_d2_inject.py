#!/usr/bin/env python3
"""Decode-window-only D2 fault injection (PLAN.md amendment 21 section 2), CPU, inside the B1 image with the
hooked indexer.py: the REAL KpoolTailMetadataBuilder.build, the REAL generic per-group slot kernel
(_compute_slot_mappings_kernel), the REAL decode index-K writer (kpool_decode_update_and_maybe_write_cache_batched)
and compress kernel (kpool_compress_and_write_cache) under the Triton interpreter, the hook's shadow ledger, and
the FROZEN B1 mapguard's write check (check_writes, unmodified file d7/step2/guard/mapguard_hooks.py).

  D1  mapping: injected window tokens get the OLD mapping (generic kernel output: null tail block 0, slot pos % 8);
      the same request's prefill rows (D7 writer) and every other request keep B1's circular mapping; padding rows
      stay PAD; with injection off the builder output is byte-identical to B1; the persistent slot-mapping buffer
      (what FULL-graph replays read) holds the final mapping.
  D2  content, B1 arm (no injection): every committed decode pool == the compress of the request's own keys
      (positive control; includes rejected-draft redo).
  D3  content, injected arm, 3 requests: some committed pools are corrupted; every pool the shadow ledger calls
      clean is byte-correct and every pool it calls corrupt is wrong (races reported, not asserted).
  D4  one injected request alone: zero corrupted pools (the D2 signature: 0% at one request in flight).
  D5  the frozen guard: tail_w_stash flags exactly the injected tokens (mism + null-block class); index-K decode
      slots stay correct (D2 does not move pool slots); B1 arm: zero.
  D6  hook glue: d2_tail_mapping counts injected tokens per request; _shadow_step fills the device class table.

Attempt 2 (post-inspection PA2-4; attempt-1 file kept as test_d2_inject.py.a1; D1-D6 unchanged except that the
input-batch stand-in now carries query_start_loc_np, as the engine's InputBatch does):
  D7p padded FULL-graph steps (3 requests = 12 tokens run in the 16-token graph: the builder sees 4 requests) through
      the REAL hooked builder, the REAL decode writer and the hook's own ledger (_shadow_step), against the bytes:
      a  B1 arm: every committed pool classified, under its own request, clean, and byte-correct (the positive
         control the attempt-2 analysis asserts on GPU);
      b  injected arm: every committed pool classified; clean -> byte-correct, corrupt -> wrong;
      c  padded and unpadded runs of the same schedule give identical classes (both arms);
      d  a count mismatch between the stored mapping and the step is refused (raises; nothing attributed).
  D7pn negative control: the attempt-1 module (forcewin_hooks.py.a1), fed the same padded builder inputs, fails
      D7p-a (pools under the wrong request / missing), as DIAGNOSIS-a1 section 7 measured on GPU.
  D0  no unexpected hook errors in the suite.
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("FORCEWIN", "on")
os.environ.setdefault("FORCEWIN_D2", "1")
os.environ.setdefault("FORCEWIN_SHADOW", "1")
os.environ.setdefault("FORCEWIN_WMAX", "512")
os.environ.setdefault("FORCEWIN_DIR", "/tmp/fwtest")

from fwtest_common import check, summary  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from vllm import forcewin_hooks as fw  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
import vllm.model_executor.layers.sparse_attn_indexer  # noqa: E402,F401  (import order, as b0emu)
import vllm.models.glm5next.nvidia.ops.kpool_compress  # noqa: E402,F401
from vllm.models.glm5next.nvidia.ops.kpool_compress import (  # noqa: E402
    kpool_compress_and_write_cache, kpool_decode_update_and_maybe_write_cache_batched)
import vllm.v1.attention.backends.mla.indexer as IDX  # noqa: E402
from vllm.v1.attention.backend import CommonAttentionMetadata  # noqa: E402
from vllm.v1.worker.gpu.block_table import _compute_slot_mappings_kernel  # noqa: E402

assert fw.ON and fw.D2_ALLOWED and fw.SHADOW
RING, KPOOL, HD = 8, 4, 128
NB, PAGE, IDX_PAGE = 64, 143616, 64 * 132
PAD = -1
CPU = torch.device("cpu")


def builder():
    b = object.__new__(IDX.KpoolTailMetadataBuilder)
    b.kv_cache_spec = types.SimpleNamespace(block_size=RING)
    return b


def generic_slots(tab, idx_mapping, qsl, pos, n_pad):
    """The REAL per-group slot kernel on the tail group's table (only column 0 written, as KpoolTailManager)."""
    buf = torch.full((1, n_pad), PAD, dtype=torch.int64)
    ptr = torch.tensor([tab.data_ptr()], dtype=torch.uint64)
    strides = torch.tensor([tab.stride(0)], dtype=torch.int64)
    bsz = torch.tensor([RING], dtype=torch.int32)
    nreq = idx_mapping.shape[0]
    _compute_slot_mappings_kernel[(1, nreq + 1)](n_pad, idx_mapping, qsl.to(torch.int32), pos, ptr, strides, bsz,
                                                 buf, buf.stride(0), 0, CP_SIZE=1, CP_INTERLEAVE=1, PAD_ID=PAD,
                                                 TRITON_BLOCK_SIZE=1024)
    return buf[0]


class Rig:
    """Requests with own tail blocks; per-step batch -> real builder (hooked) -> tail slots."""

    def __init__(self, own, P, d2, max_reqs=8):
        fw.S.__init__()
        self.runner = types.SimpleNamespace(device=CPU, num_speculative_steps=3, max_num_reqs=max_reqs,
                                            model_config=types.SimpleNamespace(get_vocab_size=lambda: 154880))
        self.own, self.P = own, P
        self.tab = torch.zeros(max_reqs, 64, dtype=torch.int32)      # tail group table: column 0 only
        for i, b in enumerate(own):
            self.tab[i, 0] = b
        for i in range(len(own)):
            ea = {"forcewin_hooks": [1] * 400, "local_fw_tag": "r%d" % i}
            if d2[i] is not None:
                ea["local_fw_d2"] = d2[i]
            fw.add_request(self.runner, "r%d" % i, i, P[i], types.SimpleNamespace(
                sampling_params=SamplingParams(max_tokens=400, extra_args=ea)))

    def mapping(self, rows, per_req_pos, pad_reqs=0, arm=True):
        """rows: request indices in batch order; per_req_pos: list of position lists. Returns (tail slots,
        circular B1 mapping, generic mapping, persistent buffer after build, qsl, positions)."""
        pos = torch.tensor(sum(per_req_pos, []), dtype=torch.int64)
        lens = [len(p) for p in per_req_pos]
        qsl = torch.zeros(len(rows) + 1, dtype=torch.int64)
        torch.cumsum(torch.tensor(lens), 0, out=qsl[1:])
        n = int(qsl[-1])
        n_pad = n + 4 * pad_reqs
        idx_map = torch.tensor(rows, dtype=torch.int32)
        bt = self.tab[idx_map.long()].clone()
        if pad_reqs:
            bt = torch.cat([bt, torch.zeros(pad_reqs, 64, dtype=torch.int32)])
            qsl = torch.cat([qsl, qsl[-1] + 4 * torch.arange(1, pad_reqs + 1)])
            pos = torch.cat([pos, torch.zeros(4 * pad_reqs, dtype=torch.int64)])
        gen = generic_slots(self.tab, idx_map, qsl[: len(rows) + 1], pos, n_pad)
        circ = IDX.compute_kpool_tail_slot_mapping(gen.clone(), bt, qsl, pos, n_pad, len(rows) + pad_reqs, RING)
        ib = types.SimpleNamespace(num_reqs=len(rows), idx_mapping=idx_map, idx_mapping_np=np.array(rows),
                                   is_prefilling_np=np.zeros(len(rows), dtype=bool), num_tokens=n,
                                   num_draft_tokens=0, query_start_loc_np=qsl[: len(rows) + 1].numpy().copy())
        if arm:
            fw.on_batch(self.runner, ib)
        persistent = gen.clone()
        cm = CommonAttentionMetadata(
            query_start_loc=qsl, query_start_loc_cpu=qsl.clone(),
            seq_lens=torch.tensor([max(p) + 1 for p in per_req_pos] + [0] * pad_reqs, dtype=torch.int32),
            num_reqs=len(rows) + pad_reqs, num_actual_tokens=n_pad, max_query_len=4, max_seq_len=1000,
            block_table_tensor=bt, slot_mapping=persistent, positions=pos)
        out = IDX.KpoolTailMetadataBuilder.build(builder(), 0, cm).slot_mapping
        fw.S.fwd_armed = False
        return out.clone(), circ, gen, persistent, qsl, pos, ib


def t_mapping():
    # r0 injected window [0, 300) from P=40; r1 not injected; r2 injected but its rows are prefill (pos < P);
    # one FULL-graph padding request on the null block
    rig = Rig(own=[5, 6, 7], P=[40, 40, 52], d2=[[0, 300], None, [0, 300]])
    pr = [[44, 45, 46, 47], [61, 62, 63, 64], [48, 49, 50, 51]]
    out, circ, gen, persistent, qsl, pos, _ = rig.mapping([0, 1, 2], pr, pad_reqs=1)
    exp = circ.clone()
    exp[0:4] = torch.tensor([p % RING for p in pr[0]])                  # null block 0, slot pos % 8
    check("D1a injected window tokens -> OLD mapping (null tail block 0, pos % 8)",
          out[0:4].tolist() == [p % RING for p in pr[0]] and gen[0:4].tolist() == out[0:4].tolist(),
          "got %s generic %s" % (out[0:4].tolist(), gen[0:4].tolist()))
    check("D1b other request and the injected request's PREFILL rows keep B1's circular mapping",
          torch.equal(out[4:12], circ[4:12]) and out[4:8].tolist() == [6 * 8 + p % 8 for p in pr[1]]
          and out[8:12].tolist() == [7 * 8 + p % 8 for p in pr[2]], "r1 %s r2 %s" % (out[4:8].tolist(),
                                                                                   out[8:12].tolist()))
    check("D1c padding rows (null block) stay PAD", out[12:16].tolist() == [PAD] * 4, str(out[12:16].tolist()))
    check("D1d persistent slot-mapping buffer holds the final mapping (graph-safe write kept)",
          torch.equal(persistent, out))
    out2, circ2, *_ = rig.mapping([0, 1, 2], pr, pad_reqs=1, arm=False)
    check("D1e injection off (step not armed): builder output byte-identical to B1", torch.equal(out2, circ2))
    rig2 = Rig(own=[5, 6, 7], P=[40, 40, 52], d2=[None, None, None])
    out3, circ3, *_ = rig2.mapping([0, 1, 2], pr, pad_reqs=1)
    check("D1f no request keyed: builder output byte-identical to B1", torch.equal(out3, circ3))
    rig3 = Rig(own=[5, 6, 7], P=[40, 40, 52], d2=[[0, 6], None, None])
    out4, circ4, *_ = rig3.mapping([0], [[44, 45, 46, 47]])
    check("D1g window bounds respected: positions >= P+hi revert to B1",
          out4[:2].tolist() == [4, 5] and out4[2:4].tolist() == [5 * 8 + 6, 5 * 8 + 7], str(out4[:4].tolist()))


def make_kv():
    raw = torch.zeros(NB * PAGE, dtype=torch.uint8)
    tail = torch.as_strided(raw.view(torch.bfloat16), size=(NB, 2, RING, HD), stride=(PAGE // 2, RING * HD, HD, 1))
    idxc = torch.as_strided(raw, size=(NB * 17, 64, 132), stride=(IDX_PAGE, 132, 1))
    return raw, tail, idxc


def pool_rec(raw, mla_blk, pool):
    """Dequantized index-K record of a pool (page-planar: 64 x 128 E4M3 bytes, then 64 x fp32 scales)."""
    b0 = (mla_blk * 17 + pool // 64) * IDX_PAGE
    off = pool % 64
    q = raw[b0 + off * HD: b0 + off * HD + HD].view(torch.float8_e4m3fn).float()
    sc = raw[b0 + 64 * HD + off * 4: b0 + 64 * HD + off * 4 + 4].view(torch.float32)
    return q * sc


def ref_rec(K4, G4, ape):
    """The engine's own compress kernel on the request's true keys (as tailfix test K3)."""
    rk, rs = kpool_compress_and_write_cache(
        torch.zeros(1, 64, 132, dtype=torch.uint8), K4[None].contiguous(), G4[None].contiguous(), ape,
        torch.zeros(1, dtype=torch.int64), KPOOL, HD, round_scale=True, return_compressed=True, write_cache=False)
    return rk[0].float().reshape(-1) * rs[0].float().reshape(-1)[0]


def same(a, b):
    return float((a - b).abs().max()) <= 1e-6 * max(1.0, float(b.abs().max()))


def simulate(n_req, inject, steps=10, seed=0, P0=(64, 72, 100), accept_seq=None):
    """Multi-step speculative decode (K=3, 4 rows/request/step) through the hooked builder and the real decode
    writer. Rows beyond the accepted prefix carry rejected-draft keys; the next step redoes those positions with
    the true keys. Returns per-request committed-pool outcomes and shadow classes."""
    g = torch.Generator().manual_seed(seed)
    own = [5, 6, 7][:n_req]
    mla = [30, 31, 32][:n_req]
    P = list(P0[:n_req])
    d2 = [[0, 400] if inject else None] * n_req
    rig = Rig(own=own, P=P, d2=d2)
    raw, tail, idxc = make_kv()
    ape = 0.1 * torch.randn(KPOOL, HD, generator=g)
    Kt = {r: torch.randn(600, HD, generator=g).to(torch.bfloat16) for r in range(n_req)}   # true keys by pos
    Gt = {r: torch.randn(600, HD, generator=g).to(torch.bfloat16) for r in range(n_req)}
    p0 = list(P)
    committed = {r: [] for r in range(n_req)}
    shadow = fw.ShadowRing()
    tok_rows = {}
    for s in range(steps):
        rows = list(range(n_req))
        per = [list(range(p0[r], p0[r] + 4)) for r in rows]
        ts, circ, gen, persistent, qsl, pos, ib = rig.mapping(rows, per)
        a = [accept_seq[(s + r) % len(accept_seq)] if accept_seq else int(torch.randint(0, 4, (1,), generator=g))
             for r in rows]
        Kb = torch.zeros(n_req, 4, HD, dtype=torch.bfloat16)
        Gb = torch.zeros(n_req, 4, HD, dtype=torch.bfloat16)
        sm = torch.full((n_req, 4), -1, dtype=torch.int32)
        for r in rows:
            for j in range(4):
                p = per[r][j]
                if j <= a[r]:
                    Kb[r, j], Gb[r, j] = Kt[r][p], Gt[r][p]
                else:   # conditioned on a rejected draft: a different key at the same position
                    Kb[r, j] = torch.randn(HD, generator=g).to(torch.bfloat16)
                    Gb[r, j] = torch.randn(HD, generator=g).to(torch.bfloat16)
                if (p + 1) % KPOOL == 0:
                    sm[r, j] = (mla[r] * 17) * 64 + p // KPOOL
        kpool_decode_update_and_maybe_write_cache_batched(
            idxc, tail, ts.view(n_req, 4).to(torch.int32), Kb, Gb, ape, sm,
            pos.view(n_req, 4).to(torch.int32), KPOOL, HD, round_scale=True)
        toks = []
        for r in rows:
            toks.append(dict(key=r, P=P[r], pos=per[r], slot=ts.view(n_req, 4)[r].tolist(),
                             valid=[j <= a[r] for j in range(4)]))
            for j in range(a[r] + 1):
                p = per[r][j]
                if (p + 1) % KPOOL == 0:
                    committed[r].append(p // KPOOL)
            tok_rows.setdefault(s, (ts.clone(), pos.clone(), qsl.clone(), a, ib))
        shadow.step(toks)
        p0 = [p0[r] + a[r] + 1 for r in rows]
    out = {}
    for r in range(n_req):
        res = []
        for pool in committed[r]:
            got = pool_rec(raw, mla[r], pool)
            ref = ref_rec(Kt[r][4 * pool: 4 * pool + 4], Gt[r][4 * pool: 4 * pool + 4], ape)
            res.append((pool, same(got, ref), shadow.cls.get(r, {}).get(pool)))
        out[r] = res
    return out, rig, tok_rows, ts, pos


def t_content():
    b1, *_ = simulate(3, inject=False, steps=12, seed=1)
    allp = [x for r in b1.values() for x in r]
    check("D2 B1 arm: every committed decode pool == compress of the request's own keys (incl. redo)",
          all(ok for _, ok, _ in allp) and len(allp) > 10,
          "%d pools, shadow classes %s" % (len(allp), sorted({c for *_, c in allp})))
    inj, *_ = simulate(3, inject=True, steps=12, seed=1)
    allp = [x for r in inj.values() for x in r]
    clean = [ok for _, ok, c in allp if c == fw.PC_CLEAN]
    corrupt = [ok for _, ok, c in allp if c == fw.PC_CORRUPT]
    race = [ok for _, ok, c in allp if c == fw.PC_RACE]
    wrong = sum(1 for _, ok, _ in allp if not ok)
    check("D3a injected arm (3 requests): committed pools are corrupted", wrong > 0,
          "%d of %d pools wrong" % (wrong, len(allp)))
    check("D3b shadow ledger: every 'clean' pool is byte-correct, every 'corrupt' pool is wrong",
          all(clean) and not any(corrupt),
          "clean %d (all ok %s), corrupt %d (any ok %s), race %d (ok %d)"
          % (len(clean), all(clean), len(corrupt), any(corrupt), len(race), sum(race)))
    one, *_ = simulate(1, inject=True, steps=14, seed=2)
    allp = [x for r in one.values() for x in r]
    check("D4 one injected request alone: zero corrupted pools (D2's 1-in-flight signature)",
          all(ok for _, ok, _ in allp) and all(c == fw.PC_CLEAN for *_, c in allp), "%d pools" % len(allp))
    return inj


def t_guard():
    sys.path.insert(0, "/hv/guard")
    os.environ.pop("GLM53_MAPGUARD", None)
    import mapguard_hooks as G
    res = {}
    for inject in (False, True):
        _, rig, tok_rows, ts, pos = simulate(3, inject=inject, steps=1, seed=3)
        ts, pos, qsl, a, ib = tok_rows[0]
        geo = G.Geo(nb=NB, R=8, W0=4)
        dev = G.Dev(geo, CPU)
        ev = dev.ev
        ev.armed.fill_(1)
        ev.step.fill_(1)
        ev.nrows.fill_(3)
        ev.ntok.fill_(12)
        ev.qsl[:4] = qsl
        ev.qsl[4:] = 12
        ev.nsched[:3] = 4
        ev.nspec[:3] = 3
        ev.grp.fill_(-1)
        ev.tagx.fill_(-1)
        for r, (tb, mb) in enumerate(zip([5, 6, 7], [30, 31, 32])):
            ev.bt1[r] = tb
            ev.grp[tb] = geo.g_tail
            ev.tagx[tb] = r
            ev.bt0[r, 0] = mb
            ev.nb0[r] = 1
            ev.col0[mb] = 0
            ev.grp[mb] = geo.g_attn
            ev.tagx[mb] = r
            ev.tag[r] = r
        tok = torch.arange(12)
        G.check_writes(dev, G.S_TAIL_STASH, 3, 0, "tail", ts.to(torch.int64), pos, tok)
        idx_slot = torch.full((12,), -1, dtype=torch.int64)
        for t in range(12):
            r, p = t // 4, int(pos[t])
            if (p + 1) % 4 == 0:
                idx_slot[t] = [30, 31, 32][r] * 1088 + (p // 4) % 1088
        G.check_writes(dev, G.S_IDX_W_DEC, 3, 0, "idx", idx_slot, pos, tok)
        c = dev.cnt
        res[inject] = dict(tail_mism=int(c[G.S_TAIL_STASH, G.CI["mism"]]), tail_null=int(c[G.S_TAIL_STASH, G.CI["null"]]),
                           tail_checked=int(c[G.S_TAIL_STASH, G.CI["checked"]]),
                           idx_mism=int(c[G.S_IDX_W_DEC, G.CI["mism"]]), idx_written=int(c[G.S_IDX_W_DEC, G.CI["written"]]))
    check("D5a frozen guard, B1 arm: zero tail_w_stash / idx_w_dec mismatches", res[False]["tail_mism"] == 0
          and res[False]["idx_mism"] == 0, str(res[False]))
    check("D5b frozen guard, injected arm: tail_w_stash mism == null-class == every injected token; "
          "idx_w_dec slots still correct", res[True]["tail_mism"] == 12 and res[True]["tail_null"] == 12
          and res[True]["idx_mism"] == 0 and res[True]["idx_written"] == res[False]["idx_written"], str(res[True]))


def t_glue():
    rig = Rig(own=[5, 6, 7], P=[64, 72, 100], d2=[[0, 400], [0, 400], None])
    per = [[64, 65, 66, 67], [72, 73, 74, 75], [100, 101, 102, 103]]
    out, circ, gen, persistent, qsl, pos, ib = rig.mapping([0, 1, 2], per)
    cnt = fw.S.d2count[:3].tolist()
    check("D6a d2_tail_mapping counts injected tokens per request", cnt == [4, 4, 0], str(cnt))
    tokrec = fw.S.d2_tok
    check("D6b shadow input recorded for the step (final mapping, positions, qsl)",
          tokrec is not None and torch.equal(tokrec[0], out[:12]))
    sc = dict(acc=torch.tensor([3, 3, 3]), idx=torch.tensor([0, 1, 2]), nr=3)
    fw._shadow_step(ib, sc)
    # r0 and r1 write the same block-0 slots (64 % 8 == 72 % 8): their completing pools race
    pc = fw.S.pcls[:3, :4].tolist()
    k0, k1 = 64 // 4 - 64 // 4, 72 // 4 - 72 // 4
    check("D6c _shadow_step fills the device class table (colliding requests -> race; uninjected r2 clean)",
          pc[0][k0] == fw.PC_RACE and pc[1][k1] == fw.PC_RACE and pc[2][0] == fw.PC_CLEAN, str(pc))


# ---------------------------------------------------------------- attempt 2: padded steps (PA2-4)
ERRS = []
_orig_err = fw._err


def _count_err(where, exc):
    ERRS.append((where, repr(exc)[:200]))
    _orig_err(where, exc)


fw._err = _count_err


def load_a1():
    import importlib.machinery
    import importlib.util
    path = os.path.join(HERE, "..", "hook", "forcewin_hooks.py.a1")
    loader = importlib.machinery.SourceFileLoader("forcewin_hooks_a1", path)
    spec = importlib.util.spec_from_loader("forcewin_hooks_a1", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def simulate_hook(n_req, inject, steps=12, seed=0, P0=(64, 72, 100), pad_reqs=1, accept_seq=None):
    """As simulate(), but the ledger is the HOOK's own (_shadow_step on the mapping the hooked builder stored), and
    every step runs with pad_reqs FULL-graph padding requests appended to the builder's batch. Returns per-request
    committed pools (byte truth), the hook's class map, and the per-step builder inputs (for the a1 replay)."""
    g = torch.Generator().manual_seed(seed)
    own = [5, 6, 7][:n_req]
    mla = [30, 31, 32][:n_req]
    P = list(P0[:n_req])
    rig = Rig(own=own, P=P, d2=[[0, 400] if inject else None] * n_req)
    raw, tail, idxc = make_kv()
    ape = 0.1 * torch.randn(KPOOL, HD, generator=g)
    Kt = {r: torch.randn(600, HD, generator=g).to(torch.bfloat16) for r in range(n_req)}
    Gt = {r: torch.randn(600, HD, generator=g).to(torch.bfloat16) for r in range(n_req)}
    p0 = list(P)
    committed = {r: [] for r in range(n_req)}
    replay = []
    n = 4 * n_req
    for s in range(steps):
        rows = list(range(n_req))
        per = [list(range(p0[r], p0[r] + 4)) for r in rows]
        ts, circ, gen, persistent, qsl, pos, ib = rig.mapping(rows, per, pad_reqs=pad_reqs)
        a = [accept_seq[(s + r) % len(accept_seq)] if accept_seq else int(torch.randint(0, 4, (1,), generator=g))
             for r in rows]
        Kb = torch.zeros(n_req, 4, HD, dtype=torch.bfloat16)
        Gb = torch.zeros(n_req, 4, HD, dtype=torch.bfloat16)
        sm = torch.full((n_req, 4), -1, dtype=torch.int32)
        for r in rows:
            for j in range(4):
                p = per[r][j]
                if j <= a[r]:
                    Kb[r, j], Gb[r, j] = Kt[r][p], Gt[r][p]
                else:
                    Kb[r, j] = torch.randn(HD, generator=g).to(torch.bfloat16)
                    Gb[r, j] = torch.randn(HD, generator=g).to(torch.bfloat16)
                if (p + 1) % KPOOL == 0:
                    sm[r, j] = (mla[r] * 17) * 64 + p // KPOOL
        kpool_decode_update_and_maybe_write_cache_batched(
            idxc, tail, ts[:n].view(n_req, 4).to(torch.int32), Kb, Gb, ape, sm,
            pos[:n].view(n_req, 4).to(torch.int32), KPOOL, HD, round_scale=True)
        sc = dict(nr=n_req, acc=torch.tensor(a), idx=torch.tensor(rows))
        fw._shadow_step(ib, sc)
        replay.append((gen.clone(), circ.clone(), qsl.clone(), pos.clone(), int(qsl[-1]), n_req + pad_reqs, ib, a))
        for r in rows:
            for j in range(a[r] + 1):
                p = per[r][j]
                if (p + 1) % KPOOL == 0:
                    committed[r].append(p // KPOOL)
        p0 = [p0[r] + a[r] + 1 for r in rows]
    truth = {}
    for r in range(n_req):
        for pool in committed[r]:
            got = pool_rec(raw, mla[r], pool)
            ref = ref_rec(Kt[r][4 * pool: 4 * pool + 4], Gt[r][4 * pool: 4 * pool + 4], ape)
            truth[(r, pool)] = same(got, ref)
    cls = {(k, pool): c for k, d in fw.S.shadow.cls.items() for pool, c in d.items()}
    return dict(truth=truth, cls=cls, replay=replay, rig=rig, P=P)


def replay_a1(a1, res, inject):
    """Feed the attempt-1 module the SAME builder inputs (padded qsl, n_reqs incl. padding) and decisions."""
    a1.S.__init__()
    a1.S.shadow = a1.ShadowRing()
    rig = res["rig"]
    for i, Pi in enumerate(res["P"]):
        ea = {"forcewin_hooks": [1] * 400, "local_fw_tag": "r%d" % i}
        if inject:
            ea["local_fw_d2"] = [0, 400]
        a1.add_request(rig.runner, "r%d" % i, i, Pi, types.SimpleNamespace(
            sampling_params=SamplingParams(max_tokens=400, extra_args=ea)))
    for gen, circ, qsl, pos, n_pad, n_reqs, ib, a in res["replay"]:
        a1.on_batch(rig.runner, ib)
        out = a1.d2_tail_mapping(gen, circ, qsl, pos, n_pad, n_reqs)
        a1.note_tail_mapping(out, pos, qsl, n_pad, n_reqs)
        a1.S.fwd_armed = False
        a1._shadow_step(ib, dict(nr=ib.num_reqs, acc=torch.tensor(a), idx=torch.tensor(list(range(ib.num_reqs)))))
    return {(k, pool): c for k, d in a1.S.shadow.cls.items() for pool, c in d.items()}


def ledger_vs_truth(truth, cls):
    keys_t, keys_c = set(truth), set(cls)
    clean = [truth[k] for k in keys_t & keys_c if cls[k] == fw.PC_CLEAN]
    corrupt = [truth[k] for k in keys_t & keys_c if cls[k] == fw.PC_CORRUPT]
    return dict(committed=len(keys_t), classified=len(keys_t & keys_c), missing=len(keys_t - keys_c),
                foreign=len(keys_c - keys_t), clean=len(clean), clean_ok=all(clean), corrupt=len(corrupt),
                corrupt_wrong=not any(corrupt), race=sum(1 for k in keys_t & keys_c if cls[k] == fw.PC_RACE),
                byte_ok=sum(truth.values()))


def t_padded():
    b1p = simulate_hook(3, inject=False, steps=12, seed=1, pad_reqs=1)
    j = ledger_vs_truth(b1p["truth"], b1p["cls"])
    allclean = all(c == fw.PC_CLEAN for k, c in b1p["cls"].items() if k in b1p["truth"])
    check("D7p-a B1 arm, padded steps, hook ledger: every committed pool classified under its own request, all "
          "clean, all byte-correct", j["missing"] == 0 and j["foreign"] == 0 and allclean and j["clean_ok"]
          and j["byte_ok"] == j["committed"] and j["committed"] > 10, str(j))
    injp = simulate_hook(3, inject=True, steps=12, seed=1, pad_reqs=1)
    ji = ledger_vs_truth(injp["truth"], injp["cls"])
    check("D7p-b injected arm, padded steps, hook ledger: every committed pool classified; clean -> byte-correct, "
          "corrupt -> wrong", ji["missing"] == 0 and ji["foreign"] == 0 and ji["clean_ok"] and ji["corrupt_wrong"]
          and ji["corrupt"] > 0, str(ji))
    b1u = simulate_hook(3, inject=False, steps=12, seed=1, pad_reqs=0)
    inju = simulate_hook(3, inject=True, steps=12, seed=1, pad_reqs=0)
    check("D7p-c padded and unpadded runs of the same schedule: identical classes (B1 and injected arms)",
          b1u["cls"] == b1p["cls"] and inju["cls"] == injp["cls"],
          "B1 equal %s, injected equal %s" % (b1u["cls"] == b1p["cls"], inju["cls"] == injp["cls"]))
    # D7p-d: a count mismatch is refused
    gen, circ, qsl, pos, n_pad, n_reqs, ib, a = b1p["replay"][0]
    fw.S.d2_tok = (circ[:12].clone(), pos[:12].clone(), qsl[:4].clone(), 3, 12)
    try:
        fw._shadow_step(ib, dict(nr=2, acc=torch.tensor(a[:2]), idx=torch.tensor([0, 1])))
        refused = False
    except ValueError:
        refused = True
    check("D7p-d count mismatch between the stored mapping and the step is refused (nothing attributed)", refused)
    a1 = load_a1()
    a1._err = lambda where, exc: None
    ca = replay_a1(a1, b1p, inject=False)
    ja = ledger_vs_truth(b1p["truth"], ca)
    a1clean = all(c == fw.PC_CLEAN for k, c in ca.items() if k in b1p["truth"])
    a1_pass = ja["missing"] == 0 and ja["foreign"] == 0 and a1clean
    cu = replay_a1(a1, b1u, inject=False)
    check("D7pn negative control: the attempt-1 ledger fails D7p-a on the same padded inputs (and is correct "
          "unpadded)", not a1_pass and cu == b1u["cls"],
          "a1 padded: %s, all-clean %s; a1 unpadded equal to the fixed ledger: %s" % (ja, a1clean, cu == b1u["cls"]))


if __name__ == "__main__":
    t_mapping()
    t_content()
    t_guard()
    t_glue()
    t_padded()
    check("D0 no hook errors during the suite", not ERRS, str(ERRS[:3]))
    ok = summary("test_d2_inject")
    sys.exit(0 if ok else 1)
