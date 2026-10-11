#!/usr/bin/env python3
"""T2 (PLAN.md amendment 10d): single-GPU unit test of the nominated tail-ring path in tpurtell
0.9.1 runner v2.  NOT RUN on GPU yet (CPU emulation mode validated only).

Nominated path (T0):
  runner-v2 MambaHybridModelState.prepare_attn calls build_attn_metadata without `positions`
  (v1/worker/gpu/model_states/mamba_hybrid.py:350-367; build_attn_metadata default
  positions=None, attn_utils.py:610,657), so KpoolTailMetadataBuilder.build skips
  compute_kpool_tail_slot_mapping (indexer.py:597-610) and hands the indexer op the GENERIC
  per-group slot mapping (block_table.py:261-328):
        tail_slot = persistent_tail_bt[req_row][pos // 8] * 8 + pos % 8
  KpoolTailManager writes only column 0 (one block per request), so for pos >= 8 the ring is
  column pos // 8 of the persistent row: the null block 0, or whatever stale id that column holds.
  The seed kernel's "token kpool ahead is in another block" rule then seeds every position with
  pos % 8 in 4..7 (slots 4-7 only) into column pos // 8's block.

Scenarios (each prints PASS/FAIL with numbers):
  A  single from-zero prefill chunk: where do the seed writes land (own block / null / other)?
  B  two requests decoding together (verify 4 tokens each): stash + pool completion through the
     shared null-block ring; compare each completed pool with the reference compressed from the
     request's own keys (cross-request contamination, quantified).
  C  stale column 1 in a reused runner row (LIFO reuse after a finish), victim block X owned by
     another request's indexer pages: from-zero prefill of the new occupant overwrites X's first
     indexer page at 256-B units 4-7 and 12-15 with raw bf16 K/gate of positions 12-15 (the
     forensic signature).
  D  candidate fix: positions passed (compute_kpool_tail_slot_mapping) -> A-C write only the own
     block; contamination in B disappears.
  E  race stress for the remaining unknown (how column 1 became non-zero on rank 1): random
     add/finish/reuse sequences through the REAL BlockTables staged writes (FusedStagedWriter,
     depth-2 UVA pools) with the CPU running ahead of a delayed GPU (spin kernel) by the engine's
     bound (1 step under async scheduling; PASS expected) and, informationally, by 3 steps (shows
     what a violated bound would do), checking the device tail table against a CPU shadow after
     every step.  GPU only.
  F  graph-capture smoke of the tail-guard device mode (capture + replay of check_call).  GPU only.

Usage:
  GPU (one device):  docker run --rm --gpus '"device=0"' --network none -v $PWD:/tg \
                       --entrypoint python3 local/glm53-v0.9.1-tailguard:local /tg/t2_single_gpu.py
  CPU emulation:     docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e TRITON_INTERPRET=1 \
                       -v $PWD:/tg --entrypoint python3 <image> /tg/t2_single_gpu.py --cpu-emulate
"""
import argparse
import os
import random
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--cpu-emulate", action="store_true")
ap.add_argument("--only", default="ABCDEF")
ap.add_argument("--e-steps", type=int, default=20000)
args = ap.parse_args()
if args.cpu_emulate:
    os.environ.setdefault("TRITON_INTERPRET", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from vllm.models.glm5next.nvidia.ops.kpool_compress import (  # noqa: E402
    kpool_decode_update_and_maybe_write_cache_batched,
    kpool_seed_tail_cache,
)
from vllm.v1.attention.backend import CommonAttentionMetadata  # noqa: E402
from vllm.v1.attention.backends.mla import indexer as idxmod  # noqa: E402

DEV = "cpu" if args.cpu_emulate else "cuda"
RING, KPOOL, HD = 8, 4, 128
PAGE = 143616             # idx page bytes per KV block (17 indexer pages x 8448 B)
IDX_PAGE = 8448           # one indexer page: 64 pools x 128 fp8 + 64 fp32 scales
BS_MLA, KBS_MLA = 4352, 64
NB = 64                   # KV blocks in this test pool
MAX_REQS = 16
TAIL_W = 131072           # tail table width in runner v2 (cdiv(1M, 8))
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


# ------------------------------------------------------------------ block tables
class EmuBlockTables:
    """CPU emulation with the same semantics as runner-v2 BlockTables (block_table.py:107-207):
    staged row writes (no clearing of columns beyond the write), gather copies num_blocks
    columns, generic slot mapping reads the PERSISTENT table at column pos // block_size."""

    def __init__(self, block_sizes, kernel_block_sizes, widths):
        self.block_sizes, self.kernel_block_sizes = block_sizes, kernel_block_sizes
        self.blocks_per_kv_block = [b // k for b, k in zip(block_sizes, kernel_block_sizes)]
        self.tables = [torch.zeros(MAX_REQS, w * bpk, dtype=torch.int32)
                       for w, bpk in zip(widths, self.blocks_per_kv_block)]
        self.input_block_tables = [torch.zeros_like(t) for t in self.tables]
        self.num_blocks = torch.zeros(len(widths), MAX_REQS, dtype=torch.int32)
        self.slot_mappings = torch.zeros(len(widths), 4096, dtype=torch.int64)
        self.staged = []

    def append_block_ids(self, row, new_ids, overwrite):
        for g, ids in enumerate(new_ids):
            start = 0 if overwrite else int(self.num_blocks[g, row])
            bpk = self.blocks_per_kv_block[g]
            ids = [b * bpk + k for b in ids for k in range(bpk)]
            if ids:
                self.staged.append((g, row, start, ids))
            self.num_blocks[g, row] = start + len(ids)

    def apply_staged_writes(self):
        for g, row, start, ids in self.staged:
            self.tables[g][row, start:start + len(ids)] = torch.tensor(ids, dtype=torch.int32)
        self.staged = []

    def persistent(self, g):
        return self.tables[g]

    def gather_block_tables(self, idx_mapping, num_reqs_padded):
        for g, t in enumerate(self.tables):
            for b in range(num_reqs_padded):
                if b < idx_mapping.numel():
                    r = int(idx_mapping[b])
                    n = int(self.num_blocks[g, r])
                    self.input_block_tables[g][b, :n] = t[r, :n]
                else:
                    self.input_block_tables[g][b].zero_()
        return tuple(t[:num_reqs_padded] for t in self.input_block_tables)

    def compute_slot_mappings(self, idx_mapping, qsl, positions, num_tokens_padded):
        sm = self.slot_mappings
        sm.fill_(-1)
        for g, t in enumerate(self.tables):
            bs = self.kernel_block_sizes[g]
            for b in range(idx_mapping.numel()):
                r = int(idx_mapping[b])
                for i in range(int(qsl[b]), int(qsl[b + 1])):
                    p = int(positions[i])
                    col = p // bs
                    if col < t.shape[1]:
                        sm[g, i] = int(t[r, col]) * bs + p % bs
        return sm[:, :num_tokens_padded]


def make_block_tables():
    widths = [241, TAIL_W]
    if args.cpu_emulate:
        return EmuBlockTables([BS_MLA, RING], [KBS_MLA, RING], widths)
    from vllm.v1.worker.gpu.block_table import BlockTables
    bt = BlockTables(block_sizes=[BS_MLA, RING], max_num_reqs=MAX_REQS, max_num_batched_tokens=4096,
                     max_num_blocks_per_group=widths, device=torch.device("cuda"),
                     kernel_block_sizes=[KBS_MLA, RING])
    bt.persistent = lambda g: bt.block_tables[g].gpu
    return bt


def tail_slots(bt, idx_mapping, qsl, pos, n_tok, pass_positions):
    """The tail group's slot mapping exactly as the indexer op receives it."""
    sm = bt.compute_slot_mappings(idx_mapping, qsl, pos, n_tok)
    gathered = bt.gather_block_tables(idx_mapping, idx_mapping.numel())
    b = object.__new__(idxmod.KpoolTailMetadataBuilder)

    class _Spec:
        block_size = RING

    b.kv_cache_spec = _Spec()
    cm = CommonAttentionMetadata(
        query_start_loc=qsl, query_start_loc_cpu=qsl.cpu(), seq_lens=torch.zeros(idx_mapping.numel(),
                                                                                  dtype=torch.int32, device=DEV),
        num_reqs=idx_mapping.numel(), num_actual_tokens=n_tok, max_query_len=n_tok, max_seq_len=n_tok,
        block_table_tensor=gathered[1], slot_mapping=sm[1],
        positions=pos[:n_tok] if pass_positions else None)
    return idxmod.KpoolTailMetadataBuilder.build(b, 0, cm).slot_mapping


def make_caches():
    raw = torch.zeros(NB * PAGE, dtype=torch.uint8, device=DEV)
    tail = torch.as_strided(raw.view(torch.bfloat16), size=(NB, 2, RING, HD),
                            stride=(PAGE // 2, RING * HD, HD, 1))
    idx = torch.as_strided(raw, size=(NB * 17, 64, 132), stride=(IDX_PAGE, 132, 1))
    return raw, tail, idx


def tagged_keys(n, base):
    """Key/gate rows encoding the token tag t = base + i + 1 exactly in bf16:
    dim0 = t // 64, dim1 = t % 64 (both exactly representable); gate = -key."""
    t = torch.arange(n, dtype=torch.int64, device=DEV) + base + 1
    v = torch.zeros(n, HD, dtype=torch.float32, device=DEV)
    v[:, 0] = (t // 64).float()
    v[:, 1] = (t % 64).float()
    v[:, 2] = 1.0
    return v.to(torch.bfloat16), (-v).to(torch.bfloat16)


def decode_tag(x):
    return int(float(x[0])) * 64 + int(float(x[1])) - 1


def seed_cells(tail, blocks):
    out = {}
    for b in blocks:
        for s in range(RING):
            if float(tail[b, 0, s, 2]) != 0.0:
                out[(b, s)] = decode_tag(tail[b, 0, s])
    return out


# ------------------------------------------------------------------ scenarios
def scenario_A(pass_positions):
    bt = make_block_tables()
    raw, tail, _ = make_caches()
    row, own = 15, 7
    bt.append_block_ids(row, ([20, 21], [own]), overwrite=True)
    bt.apply_staged_writes()
    idxm = torch.tensor([row], device=DEV)
    n = 2048
    qsl = torch.tensor([0, n], dtype=torch.int32, device=DEV)
    pos = torch.arange(n, device=DEV)
    ts = tail_slots(bt, idxm, qsl, pos, n, pass_positions)
    k, g = tagged_keys(n, 0)
    kpool_seed_tail_cache(tail, k, g, ts, KPOOL, HD)
    cells = seed_cells(tail, range(NB))
    tag = "D/fixed" if pass_positions else "A/shipped"
    own_cells = {c: p for c, p in cells.items() if c[0] == own}
    other = {c: p for c, p in cells.items() if c[0] != own}
    if pass_positions:
        check(f"{tag}: seed writes only the last kpool tokens into the own block",
              sorted(own_cells.values()) == [2044, 2045, 2046, 2047] and not other, f"{cells}")
    else:
        check(f"{tag}: seed writes pos 4-7 to own block and the chunk's last 4 tokens to null block 0",
              sorted(own_cells.values()) == [4, 5, 6, 7] and sorted(other) == [(0, 4), (0, 5), (0, 6), (0, 7)]
              and sorted(other.values()) == [2044, 2045, 2046, 2047], f"{cells}")


def scenario_B(pass_positions):
    """Two requests verify 4 tokens each per step (DFlash x3) at the same phase: step 1 positions
    98..101, step 2 positions 102..105.  Pool 25 (positions 100..103) spans the two steps, so its
    completion at 103 reads ring slots 4,5 (positions 100,101) stashed one step earlier.  The
    completed pools are compared with the kernel's own compression of each request's keys."""
    from vllm.models.glm5next.nvidia.ops.kpool_compress import kpool_compress_and_write_cache
    bt = make_block_tables()
    raw, tail, idx = make_caches()
    rows, owns, mlas = [14, 15], [5, 6], [30, 31]
    for r, o, m in zip(rows, owns, mlas):
        bt.append_block_ids(r, ([m], [o]), overwrite=True)
    bt.apply_staged_writes()
    idxm = torch.tensor(rows, device=DEV)
    gen = torch.Generator(device="cpu").manual_seed(7)
    K = torch.randn(2, 8, HD, generator=gen).to(DEV).to(torch.bfloat16)      # tokens 98..105
    Gt = torch.randn(2, 8, HD, generator=gen).to(DEV).to(torch.bfloat16)
    ape = (0.1 * torch.randn(KPOOL, HD, generator=gen)).to(DEV)
    off = 103 // KPOOL  # pool index within the request's first indexer page (25)
    for p0 in (98, 102):
        j = p0 - 98
        pos = torch.tensor([p0, p0 + 1, p0 + 2, p0 + 3] * 2, device=DEV)
        qsl = torch.tensor([0, 4, 8], dtype=torch.int32, device=DEV)
        ts = tail_slots(bt, idxm, qsl, pos, 8, pass_positions).view(2, 4).to(torch.int32)
        sm = torch.full((2, 4), -1, dtype=torch.int32, device=DEV)
        for i, m in enumerate(mlas):
            for t in range(4):
                if (p0 + t) % KPOOL == KPOOL - 1:
                    sm[i, t] = (m * 17) * 64 + (p0 + t) // KPOOL
        kpool_decode_update_and_maybe_write_cache_batched(
            idx, tail, ts, K[:, j:j + 4].contiguous(), Gt[:, j:j + 4].contiguous(), ape, sm,
            pos.view(2, 4).to(torch.int32), KPOOL, HD, round_scale=True)
    errs = []
    for i, m in enumerate(mlas):
        base = (m * 17) * IDX_PAGE
        got = raw[base + off * HD: base + off * HD + HD].view(torch.float8_e4m3fn).float() * \
            raw[base + 64 * HD + off * 4: base + 64 * HD + off * 4 + 4].view(torch.float32)
        ref_k, ref_s = kpool_compress_and_write_cache(
            idx, K[i, 2:6][None].contiguous(), Gt[i, 2:6][None].contiguous(), ape,
            torch.zeros(1, dtype=torch.int64, device=DEV), KPOOL, HD, round_scale=True,
            return_compressed=True, write_cache=False)
        ref = ref_k[0].float() * ref_s[0]
        errs.append(round(float((got - ref).abs().max() / ref.abs().max()), 4))
    tag = "D/fixed" if pass_positions else "B/shipped"
    if pass_positions:
        check(f"{tag}: completed decode pools equal the kernel's compression of each request's own keys",
              max(errs) < 1e-3, f"rel_err={errs}")
    else:
        check(f"{tag}: a pool spanning two steps is compressed from another request's keys "
              "(shared null-block ring)", max(errs) > 0.05, f"rel_err={errs}")


def scenario_C(pass_positions):
    bt = make_block_tables()
    raw, tail, idx = make_caches()
    victim = 33
    # victim's first indexer page holds valid pools (pattern 0x11)
    raw[victim * PAGE: victim * PAGE + IDX_PAGE] = 0x11
    row = 15
    # previous occupant of `row` left column 1 = victim (emulates the unexplained stale column)
    bt.append_block_ids(row, ([40], [9]), overwrite=True)
    bt.apply_staged_writes()
    bt.persistent(1)[row, 1] = victim
    # it finishes; a new request reuses the row (LIFO) and only column 0 is overwritten
    bt.append_block_ids(row, ([41], [10]), overwrite=True)
    bt.apply_staged_writes()
    idxm = torch.tensor([row], device=DEV)
    n = 2048
    qsl = torch.tensor([0, n], dtype=torch.int32, device=DEV)
    pos = torch.arange(n, device=DEV)
    ts = tail_slots(bt, idxm, qsl, pos, n, pass_positions)
    k, g = tagged_keys(n, 0)
    kpool_seed_tail_cache(tail, k, g, ts, KPOOL, HD)
    page = raw[victim * PAGE: victim * PAGE + 4096].view(16, 256)
    changed = [u for u in range(16) if not bool((page[u] == 0x11).all())]
    tag = "D/fixed" if pass_positions else "C/shipped"
    if pass_positions:
        check(f"{tag}: stale column ignored, victim page untouched", changed == [], f"units={changed}")
    else:
        vals = sorted({decode_tag(tail[victim, 0, s]) for s in range(4, 8)})
        check(f"{tag}: victim page 256-B units 4-7 and 12-15 overwritten with positions 12-15 "
              "(forensic signature)", changed == [4, 5, 6, 7, 12, 13, 14, 15] and vals == [12, 13, 14, 15],
              f"units={changed} positions={vals}")


def scenario_E(steps, runahead):
    """Real BlockTables (FusedStagedWriter, depth-2 UVA metadata pools) under CPU run-ahead.
    The CPU stages step N+1 only after the GPU finished step N+1-runahead (an event wait), as the
    engine's batch queue bounds it (runahead=1 under async scheduling).  A spin kernel delays the
    GPU so the bound is actually reached.  After every step the device tail table is snapshotted
    and compared (post hoc) with a CPU shadow of the staged writes."""
    from vllm.v1.worker.gpu.block_table import BlockTables
    rnd = random.Random(runahead)
    G = 3
    bt = BlockTables(block_sizes=[BS_MLA, RING, BS_MLA], max_num_reqs=MAX_REQS, max_num_batched_tokens=4096,
                     max_num_blocks_per_group=[241, TAIL_W, 241], device=torch.device("cuda"),
                     kernel_block_sizes=[KBS_MLA, RING, BS_MLA])
    shadow = torch.zeros(MAX_REQS, 4, dtype=torch.int64)
    free_rows = list(range(MAX_REQS))
    live = set()
    nb = [1000]
    spin = torch.zeros(1 << 22, device="cuda")
    events, snaps = [], []
    for step in range(steps):
        if len(events) >= runahead:
            events[-runahead].synchronize()
        for r in list(live):
            if rnd.random() < 0.1:
                free_rows.append(r)
                live.discard(r)
        while free_rows and rnd.random() < 0.5:
            r = free_rows.pop()  # LIFO reuse, as RequestState.free_indices
            nmla = rnd.randint(1, 6)
            ids = ([nb[0] + i for i in range(nmla)], [nb[0] + 100], [nb[0] + 200 + i for i in range(nmla)])
            nb[0] += 300
            bt.append_block_ids(r, ids, overwrite=True)
            shadow[r, 0] = ids[1][0]
            live.add(r)
        for r in list(live):
            if rnd.random() < 0.2:
                bt.append_block_ids(r, ([nb[0]], [], [nb[0] + 1]), overwrite=False)
                nb[0] += 2
        for _ in range(4):
            spin.mul_(1.0000001)
        bt.apply_staged_writes()
        rows = sorted(live) or [0]
        bt.gather_block_tables(torch.tensor(rows, device="cuda"), len(rows))
        snaps.append((step, bt.block_tables[1].gpu[:, :4].clone(), shadow.clone()))
        ev = torch.cuda.Event()
        ev.record()
        events.append(ev)
        events = events[-8:]
    torch.cuda.synchronize()
    mism = 0
    for st, dev_t, sh in snaps:
        if not torch.equal(dev_t.cpu().to(torch.int64), sh):
            mism += 1
            if mism <= 3:
                print("  E mismatch at step", st, "device", dev_t.cpu().tolist()[:4], "shadow", sh.tolist()[:4])
    nz = int((bt.block_tables[1].gpu[:, 1:] != 0).sum())
    name = (f"E(runahead={runahead}): {steps} steps of staged writes: device tail table == CPU shadow, "
            "no non-zero column >= 1")
    if runahead <= 1:
        check(name, mism == 0 and nz == 0, f"mismatched_steps={mism} nonzero_cols={nz}")
    else:
        print(f"INFO {name}: mismatched_steps={mism} nonzero_cols={nz} (run-ahead beyond the engine bound)")


def scenario_F():
    os.environ["GLM53_TAILGUARD"] = "device"
    import importlib
    import kpool_tailguard as tg
    importlib.reload(tg)
    tg.MODE, tg.ENABLED = "device", True
    tg.S.dev = tg.alloc_device_buffers("cuda", MAX_REQS)
    gathered = torch.zeros(4, 64, dtype=torch.int32, device="cuda")
    gathered[:, 0] = torch.tensor([5, 6, 7, 8], device="cuda")
    qsl = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32, device="cuda")
    tg.S.ctx = dict(tail_bt=gathered, qsl=qsl, n_reqs=4)
    ts = torch.zeros(16, dtype=torch.int64, device="cuda")
    pos = torch.zeros(16, dtype=torch.int64, device="cuda")
    _, tail, _ = make_caches()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        tg.check_call("stash", "layers.27.x", tail, ts, pos, 0, KPOOL)  # warm-up outside capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        tg.check_call("stash", "layers.27.x", tail, ts, pos, 0, KPOOL)
    # replay: rows 0-3 decode at pos 100..103; row 2 uses a foreign block 33
    pos.copy_(torch.arange(100, 104, device="cuda").repeat(4))
    tsv = gathered[:, 0].repeat_interleave(4).to(torch.int64) * RING + pos % RING
    tsv[8:12] = 33 * RING + pos[8:12] % RING
    ts.copy_(tsv)
    c0 = tg.S.dev["counters"].clone()
    g.replay()
    torch.cuda.synchronize()
    d = (tg.S.dev["counters"] - c0).tolist()
    check("F: tail-guard device check captured in a CUDA graph and replayed (4 foreign writes recorded)",
          d[tg.C_FOREIGN] == 4 and d[tg.C_WRITES] == 16, f"delta={d[:6]}")


if __name__ == "__main__":
    t0 = time.time()
    if "A" in args.only:
        scenario_A(False)
    if "B" in args.only:
        scenario_B(False)
    if "C" in args.only:
        scenario_C(False)
    if "D" in args.only:
        scenario_A(True)
        scenario_B(True)
        scenario_C(True)
    if not args.cpu_emulate:
        if "E" in args.only:
            scenario_E(args.e_steps, 1)
            scenario_E(args.e_steps // 4, 3)
        if "F" in args.only:
            scenario_F()
    print(f"{sum(RESULTS)}/{len(RESULTS)} passed in {time.time() - t0:.1f}s")
    sys.exit(0 if all(RESULTS) else 1)
