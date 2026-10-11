#!/usr/bin/env python3
"""CPU-only tests for the kpool tail-ring fix variants (0.9.1 engine image, no GPU).

TRITON_INTERPRET=1 runs the real Triton tail kernels on CPU. The B12X sparse-MLA metadata
builder is driven on CPU with a minimally initialised builder object (its build() is pure
torch/numpy for DCP1).

Run (inside any 0.9.1-derived image):
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e TRITON_INTERPRET=1 \
     -e TAILFIX_EXPECT=T|U|release -v $PWD:/w --entrypoint python3 <image> /w/test_tailfix.py

TAILFIX_EXPECT selects the expected outcome of the variant-specific checks:
  T        positions passed + own_block != 0 guard + in-place persistent write + MLA causal
           length kept on the scheduler CPU bound
  U        positions passed + guard + in-place write (MLA causal length from positions)
  release  unmodified 0.9.1 (positions not passed; no guard; fresh-tensor mapping)
Every check prints PASS/FAIL against the expectation for that variant; the exit code is 0
only if every check met its expectation.
"""
import ast
import inspect
import os
import sys
import time
from types import SimpleNamespace as NS

os.environ.setdefault("TRITON_INTERPRET", "1")
import torch  # noqa: E402

torch.Tensor.pin_memory = lambda self, *a, **k: self  # no CUDA host allocator on CPU

import vllm.v1.attention.backends.mla.b12x_mla_sparse as bm  # noqa: E402
import vllm.v1.attention.backends.mla.indexer as idx  # noqa: E402
import vllm.v1.worker.gpu.attn_utils as au  # noqa: E402
import vllm.v1.worker.gpu.model_states.mamba_hybrid as mh  # noqa: E402
from vllm.models.glm5next.nvidia.ops.kpool_compress import (  # noqa: E402
    kpool_compress_and_write_cache,
    kpool_decode_update_and_maybe_write_cache_batched,
)
from vllm.v1.attention.backend import CommonAttentionMetadata  # noqa: E402

EXPECT = os.environ.get("TAILFIX_EXPECT", "T")
assert EXPECT in ("T", "U", "release"), EXPECT
FIXED = EXPECT in ("T", "U")
RING, KPOOL, HD = 8, 4, 128          # 0.9.1: tail ring 8 (kpool 4 + 3 drafts, rounded), head dim 128
RESULTS = []


def check(name, ok, detail="", expect=True):
    met = bool(ok) == bool(expect)
    RESULTS.append(met)
    tag = "PASS" if met else "FAIL"
    exp = "" if expect else " (expected NOT to hold in this variant)"
    print(f"{tag} {name}{exp}  [{'holds' if ok else 'does not hold'}; {detail}]")


# ------------------------------------------------------------------ helpers
def make_tail_block_table(own_blocks, width=64):
    bt = torch.zeros(len(own_blocks), width, dtype=torch.int32)
    bt[:, 0] = torch.tensor(own_blocks, dtype=torch.int32)
    return bt


def make_batch(per_req_positions, padded_len=None):
    positions = torch.cat([torch.tensor(p, dtype=torch.int64) for p in per_req_positions])
    num_actual = positions.numel()
    lens = [len(p) for p in per_req_positions]
    qsl = torch.zeros(len(lens) + 1, dtype=torch.int64)
    torch.cumsum(torch.tensor(lens, dtype=torch.int64), 0, out=qsl[1:])
    slot_mapping = torch.full((padded_len or num_actual,), -1, dtype=torch.int64)
    return positions, qsl, slot_mapping, num_actual, len(lens)


def generic_tail_slots(bt, qsl, positions, ring=RING):
    """block_table.py:305-313 (CP_SIZE == 1) applied to the tail group's table."""
    out = []
    for r in range(bt.shape[0]):
        for i in range(int(qsl[r]), int(qsl[r + 1])):
            p = int(positions[i])
            out.append(int(bt[r, p // ring]) * ring + p % ring)
    return torch.tensor(out, dtype=torch.int64)


def tail_builder():
    b = object.__new__(idx.KpoolTailMetadataBuilder)
    b.kv_cache_spec = NS(block_size=RING)
    return b


def tail_cm(per_req, own_blocks, with_positions=True, pad=4):
    positions, qsl, sm, n, nr = make_batch(per_req, padded_len=sum(map(len, per_req)) + pad)
    seq = torch.tensor([max(p) + 1 for p in per_req], dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=qsl, query_start_loc_cpu=qsl.clone(), seq_lens=seq, num_reqs=nr,
        num_actual_tokens=n, max_query_len=max(map(len, per_req)), max_seq_len=int(seq.max()),
        block_table_tensor=make_tail_block_table(own_blocks), slot_mapping=sm,
        positions=positions if with_positions else None)


# ------------------------------------------------------------------ P: plumbing (static)
def test_positions_plumbing():
    src = inspect.getsource(mh.MambaHybridModelState.prepare_attn)
    tree = ast.parse("class _X:\n" + src if src.startswith(" ") else src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", "")) == "build_attn_metadata"]
    kws = {k.arg: ast.unparse(k.value) for k in calls[0].keywords} if calls else {}
    check("P1 MambaHybridModelState.prepare_attn passes positions=input_batch.positions",
          kws.get("positions") == "input_batch.positions", f"positions kw={kws.get('positions')}",
          expect=FIXED)


# ------------------------------------------------------------------ G: guard (ported 240785b82)
def test_guard_ported():
    # G1 (upstream test_legacy_generic_mapping_collapses_onto_block_zero, ring 8)
    per_req = [list(range(10)), list(range(12))]
    positions, qsl, sm, n, nr = make_batch(per_req)
    bt = make_tail_block_table([5, 9])
    legacy = generic_tail_slots(bt, qsl, positions)
    collapse = all(int(legacy[i]) // RING == 0 for i in range(n) if int(positions[i]) >= RING)
    shared = bool(set(legacy[:10].tolist()) & set(legacy[10:].tolist()))
    check("G1 generic mapping collapses pos >= ring onto tail block 0 and requests share slots",
          collapse and shared, f"blocks={sorted(set((legacy // RING).tolist()))}")
    # G2 (test_circular_mapping_isolates_requests)
    out = idx.compute_kpool_tail_slot_mapping(sm, bt, qsl, positions, n, nr, RING)
    ok = all(int(out[i]) // RING == [5, 9][0 if i < 10 else 1] and int(out[i]) % RING ==
             int(positions[i]) % RING for i in range(n))
    check("G2 circular mapping keeps every token in its own block at pos % ring", ok)
    # G3 (test_circular_mapping_matches_generic_for_short_requests)
    ok = True
    for L in range(1, RING + 1):
        p, q, s, n1, r1 = make_batch([list(range(L))])
        b1 = make_tail_block_table([7])
        ok &= torch.equal(idx.compute_kpool_tail_slot_mapping(s, b1, q, p, n1, r1, RING),
                          generic_tail_slots(b1, q, p))
    check("G3 circular == generic for requests shorter than the ring", ok)
    # G4 (test_circular_mapping_preserves_padding_and_empty_batch, with the 240785b82 change)
    per_req = [list(range(10)), list(range(12)), [0, 1]]
    p, q, s, n4, r4 = make_batch(per_req, padded_len=sum(map(len, per_req)) + 8)
    b4 = make_tail_block_table([5, 9, 0])
    out = idx.compute_kpool_tail_slot_mapping(s, b4, q, p, n4, r4, RING)
    pad_start = int(q[-2])
    check("G4 padding request on the null block gets PAD (-1), never block 0 (own_block != 0 guard)",
          torch.equal(out[pad_start:], torch.full_like(out[pad_start:], -1)),
          f"slots of null-block request={out[pad_start:n4].tolist()}", expect=FIXED)
    empty = idx.compute_kpool_tail_slot_mapping(s, b4, q, p[:0], 0, r4, RING)
    check("G5 empty batch returns the input mapping unchanged", torch.equal(empty, s))
    # G6 dummy batch: every row on the null block (capture / dummy runs)
    p, q, s, n6, r6 = make_batch([[0], [0]], padded_len=4)
    out = idx.compute_kpool_tail_slot_mapping(s, make_tail_block_table([0, 0]), q, p, n6, r6, RING)
    check("G6 dummy batch (all rows on the null block) maps every token to PAD",
          out.tolist() == [-1] * 4, f"{out.tolist()}", expect=FIXED)


# ------------------------------------------------------------------ B: builder / persistent buffer
def test_builder():
    cm = tail_cm([list(range(10)), list(range(12))], [5, 9])
    buf_ptr = cm.slot_mapping.data_ptr()
    md = idx.KpoolTailMetadataBuilder.build(tail_builder(), 0, cm)
    sm = md.slot_mapping
    vals_ok = all(int(sm[i]) // RING == (5 if i < 10 else 9) for i in range(22)) and \
        bool((sm[22:] == -1).all())
    check("B1 builder (positions given) returns the circular mapping, padding kept at -1", vals_ok,
          f"blocks={sorted(set((sm[:22] // RING).tolist()))}")
    same = sm.data_ptr() == buf_ptr and sm.shape == cm.slot_mapping.shape
    check("B2 builder writes the circular mapping into the persistent per-group buffer (graph safe)",
          same and torch.equal(cm.slot_mapping, sm), f"same_storage={sm.data_ptr() == buf_ptr}",
          expect=FIXED)
    cm2 = tail_cm([list(range(10))], [5], with_positions=False)
    md2 = idx.KpoolTailMetadataBuilder.build(tail_builder(), 0, cm2)
    check("B3 builder without positions returns the generic mapping object untouched",
          md2.slot_mapping is cm2.slot_mapping)


# ------------------------------------------------------------------ K: real kernels, padded graph batch
def test_kernels_padded_batch():
    """FULL-graph uniform decode: 2 real requests + 2 padding rows (zeroed rows -> null block 0),
    DFlash x3 -> 4 tokens per request; padding tokens carry stale positions.  The real
    decode-update kernel stashes through the builder's mapping.  Fixed: block 0's ring untouched
    and each real request's ring holds its own keys; release: padding tokens write block 0."""
    NB, PAGE = 16, 143616
    raw = torch.zeros(NB * PAGE, dtype=torch.uint8)
    tail = torch.as_strided(raw.view(torch.bfloat16), size=(NB, 2, RING, HD),
                            stride=(PAGE // 2, RING * HD, HD, 1))
    idxc = torch.zeros(NB * 17, 64, 132, dtype=torch.uint8)
    own = [5, 6, 0, 0]
    pos_rows = [[40, 41, 42, 43], [61, 62, 63, 64], [36, 37, 38, 39], [36, 37, 38, 39]]
    positions = torch.tensor(sum(pos_rows, []), dtype=torch.int64)
    qsl = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int64)   # padded rows given 4 tokens each
    cm = CommonAttentionMetadata(
        query_start_loc=qsl, query_start_loc_cpu=qsl.clone(),
        seq_lens=torch.tensor([44, 65, 0, 0], dtype=torch.int32), num_reqs=4,
        num_actual_tokens=16, max_query_len=4, max_seq_len=65,
        block_table_tensor=make_tail_block_table(own), slot_mapping=torch.full((16,), -1, dtype=torch.int64),
        positions=positions)
    ts = idx.KpoolTailMetadataBuilder.build(tail_builder(), 0, cm).slot_mapping.view(4, 4).to(torch.int32)
    g = torch.Generator().manual_seed(11)
    K = torch.randn(4, 4, HD, generator=g).to(torch.bfloat16)
    G = torch.randn(4, 4, HD, generator=g).to(torch.bfloat16)
    ape = 0.1 * torch.randn(KPOOL, HD, generator=g)
    sm = torch.full((4, 4), -1, dtype=torch.int32)               # no pool completes in rows 2-3
    kpool_decode_update_and_maybe_write_cache_batched(
        idxc, tail, ts, K, G, ape, sm, positions.view(4, 4).to(torch.int32), KPOOL, HD, round_scale=True)
    block0_touched = bool(tail[0].abs().sum() != 0)
    own_ok = all(torch.equal(tail[own[r], 0, p % RING], K[r, t])
                 for r in (0, 1) for t, p in enumerate(pos_rows[r]))
    check("K1 real decode kernel: padding rows on the null block leave tail block 0 untouched",
          not block0_touched, f"tail block 0 written={block0_touched}; tail slots rows 2-3="
          f"{ts[2:].flatten().tolist()}", expect=FIXED)
    check("K2 real decode kernel: each real request stashes into its own ring at pos % ring",
          own_ok)


def test_kernel_pool_completion():
    """Two requests completing a pool that spans two verify steps (T2 scenario B through the
    builder with positions): the completed pools equal the kernel's compression of each
    request's own keys."""
    NB, PAGE, IDX_PAGE = 64, 143616, 64 * 132
    raw = torch.zeros(NB * PAGE, dtype=torch.uint8)
    tail = torch.as_strided(raw.view(torch.bfloat16), size=(NB, 2, RING, HD),
                            stride=(PAGE // 2, RING * HD, HD, 1))
    idxc = torch.as_strided(raw, size=(NB * 17, 64, 132), stride=(IDX_PAGE, 132, 1))
    owns, mlas = [5, 6], [30, 31]
    g = torch.Generator().manual_seed(7)
    K = torch.randn(2, 8, HD, generator=g).to(torch.bfloat16)
    Gt = torch.randn(2, 8, HD, generator=g).to(torch.bfloat16)
    ape = 0.1 * torch.randn(KPOOL, HD, generator=g)
    for p0 in (98, 102):
        j = p0 - 98
        per_req = [list(range(p0, p0 + 4))] * 2
        cm = tail_cm(per_req, owns, pad=0)
        ts = idx.KpoolTailMetadataBuilder.build(tail_builder(), 0, cm).slot_mapping.view(2, 4).to(torch.int32)
        sm = torch.full((2, 4), -1, dtype=torch.int32)
        for i, m in enumerate(mlas):
            for t in range(4):
                if (p0 + t) % KPOOL == KPOOL - 1:
                    sm[i, t] = (m * 17) * 64 + (p0 + t) // KPOOL
        pos = torch.tensor([list(range(p0, p0 + 4))] * 2, dtype=torch.int32)
        kpool_decode_update_and_maybe_write_cache_batched(
            idxc, tail, ts, K[:, j:j + 4].contiguous(), Gt[:, j:j + 4].contiguous(), ape, sm, pos,
            KPOOL, HD, round_scale=True)
    errs = []
    off = 103 // KPOOL
    for i, m in enumerate(mlas):
        base = (m * 17) * IDX_PAGE
        got = raw[base + off * HD: base + off * HD + HD].view(torch.float8_e4m3fn).float() * \
            raw[base + 64 * HD + off * 4: base + 64 * HD + off * 4 + 4].view(torch.float32)
        rk, rs = kpool_compress_and_write_cache(
            idxc, K[i, 2:6][None].contiguous(), Gt[i, 2:6][None].contiguous(), ape,
            torch.zeros(1, dtype=torch.int64), KPOOL, HD, round_scale=True,
            return_compressed=True, write_cache=False)
        ref = rk[0].float() * rs[0]
        errs.append(round(float((got - ref).abs().max() / ref.abs().max()), 4))
    check("K3 pool completed across two steps equals compression of the request's own keys",
          max(errs) < 1e-3, f"rel_err={errs}")


# ------------------------------------------------------------------ M: B12X causal-length source
def b12x_builder(max_tokens=64, max_seqs=16, topk=2176):
    b = object.__new__(bm.B12xMLASparseMetadataBuilder)
    b.dcp_world_size, b.dcp_rank, b.cp_kv_cache_interleave_size = 1, 0, 1
    b.num_speculative_tokens = 3
    b.ckv_prefetch_registry = None
    b.cache_seq_lens_per_token_buffer = torch.zeros(max_tokens, dtype=torch.int32)
    b.cache_seq_lens_per_req_buffer = torch.zeros(max_seqs, dtype=torch.int32)
    b.req_id_per_token_buffer = torch.zeros(max_tokens, dtype=torch.int32)
    b.page_table_1_buffer = torch.zeros(max_tokens, topk, dtype=torch.int32)
    b.nsa_cache_seqlens_buffer = torch.zeros(max_tokens, dtype=torch.int32)
    b.req_ids_arange = torch.arange(max_tokens, dtype=torch.int32)
    for n in ("ckv_page_table_1_buffer", "ckv_nsa_cache_seqlens_buffer", "dcp_rank_req_lens_buffer",
              "dcp_rank_req_starts_buffer", "dcp_local_cu_seq_lens_buffer"):
        setattr(b, n, None)
    b.kv_cache_spec = NS(block_size=4352)
    b.topk_tokens = topk
    return b


def mla_cm(true_lens, bound, with_positions, q=4):
    n = q * len(true_lens)
    qsl = torch.arange(0, n + 1, q, dtype=torch.int32)
    pos = torch.cat([torch.arange(L - q, L) for L in true_lens])
    return CommonAttentionMetadata(
        query_start_loc=qsl, query_start_loc_cpu=qsl.clone(),
        seq_lens=torch.tensor(true_lens, dtype=torch.int32), num_reqs=len(true_lens),
        num_actual_tokens=n, max_query_len=q, max_seq_len=max(bound),
        block_table_tensor=torch.zeros(len(true_lens), 4, dtype=torch.int32),
        slot_mapping=torch.zeros(n, dtype=torch.int64),
        seq_lens_cpu_upper_bound=torch.tensor(bound, dtype=torch.int32),
        positions=pos if with_positions else None, is_prefilling=torch.zeros(len(true_lens), dtype=torch.bool))


def per_token_lens(cm):
    return bm.B12xMLASparseMetadataBuilder.build(b12x_builder(), 0, cm).cache_seq_lens_per_token.tolist()


def test_mla_causal_length():
    true = [104, 54]
    exact = per_token_lens(mla_cm(true, true, True)), per_token_lens(mla_cm(true, true, False))
    check("M1 spec-verify batch, exact CPU bound: per-token causal lengths identical with and without "
          "positions", exact[0] == exact[1], f"{exact[0]}")
    lag = [104, 60]   # CPU bound off by 6 for request 1 (stale/optimistic bound)
    with_pos, without = per_token_lens(mla_cm(true, lag, True)), per_token_lens(mla_cm(true, lag, False))
    truth = [p + 1 for L in true for p in range(L - 4, L)]
    if EXPECT == "T":
        check("M2 bound != truth: with positions present the builder still uses the CPU bound "
              "(MLA unchanged by the tail fix)", with_pos == without,
              f"with_positions={with_pos} without={without}")
    else:
        check("M2 bound != truth: with positions present the builder uses positions + 1 "
              "(the bundled MLA change)", with_pos == truth and with_pos != without,
              f"with_positions={with_pos} without={without}")


# ------------------------------------------------------------------ I: integration through build_attn_metadata
def test_integration():
    """Real attn_utils.build_attn_metadata with two KV-cache groups (MLA via the B12X builder,
    kpool tail via KpoolTailMetadataBuilder), positions passed as the patched mamba_hybrid does."""
    q = 4
    true, bound = [104, 54], [104, 60]
    n = q * 2
    qsl = torch.arange(0, n + 1, q, dtype=torch.int32)
    pos = torch.cat([torch.arange(L - q, L) for L in true])
    slot_mappings = torch.full((2, n + 4), -1, dtype=torch.int64)     # persistent per-group buffer
    tail_bt = make_tail_block_table([5, 9])
    groups = [[NS(get_metadata_builder=lambda i: b12x_builder(), layer_names=["mla.0"])],
              [NS(get_metadata_builder=lambda i: tail_builder(), layer_names=["tail.0"])]]
    md = au.build_attn_metadata(
        attn_groups=groups, num_reqs=2, num_tokens=n, query_start_loc_gpu=qsl,
        query_start_loc_cpu=qsl.clone(), max_query_len=q, seq_lens=torch.tensor(true, dtype=torch.int32),
        max_seq_len=max(bound), block_tables=[torch.zeros(2, 4, dtype=torch.int32), tail_bt],
        slot_mappings=slot_mappings, kv_cache_config=NS(kv_cache_groups=[0, 1]),
        seq_lens_cpu_upper_bound=torch.tensor(bound, dtype=torch.int32),
        positions=pos, is_prefilling=torch.zeros(2, dtype=torch.bool))
    tail_sm = md["tail.0"].slot_mapping
    circ = [b * RING + int(p) % RING for b, p in zip([5] * 4 + [9] * 4, pos)]
    check("I1 tail group: circular mapping lands in the persistent slot_mappings row",
          slot_mappings[1, :n].tolist() == circ and tail_sm.data_ptr() == slot_mappings[1].data_ptr(),
          f"row={slot_mappings[1, :n].tolist()}", expect=FIXED)
    lens = md["mla.0"].cache_seq_lens_per_token.tolist()
    bound_lens = [c + i + 1 for L, B in zip(true, bound) for c in [B - q] for i in range(q)]
    pos_lens = (pos + 1).tolist()
    want = bound_lens if EXPECT == "T" else pos_lens
    check(f"I2 MLA group: per-token causal length source = {'CPU bound' if EXPECT == 'T' else 'positions'}",
          lens == want, f"lens={lens} bound={bound_lens} positions+1={pos_lens}")


if __name__ == "__main__":
    t0 = time.time()
    print(f"variant expectation: {EXPECT}; torch {torch.__version__}; "
          f"TRITON_INTERPRET={os.environ.get('TRITON_INTERPRET')}; cuda={torch.cuda.is_available()}")
    for fn in (test_positions_plumbing, test_guard_ported, test_builder, test_kernels_padded_batch,
               test_kernel_pool_completion, test_mla_causal_length, test_integration):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            check(f"{fn.__name__} raised", False, repr(e))
    print(f"{sum(RESULTS)}/{len(RESULTS)} met expectation in {time.time() - t0:.1f}s")
    sys.exit(0 if all(RESULTS) else 1)
