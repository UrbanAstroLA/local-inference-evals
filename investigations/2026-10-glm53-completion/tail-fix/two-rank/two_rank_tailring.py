#!/usr/bin/env python3
"""Kpool tail ring under continuous batching, per DCP rank, with the engine's real kernels (CPU).

Runs inside an engine image with no GPU: TRITON_INTERPRET=1 executes the image's own Triton kernels
on CPU tensors:
  * generic per-group slot mapping  _compute_slot_mappings_kernel   (v1/worker/gpu/block_table.py)
  * pool-granular index-K slot map  get_compressed_slot_mapping     (v1/attention/backends/mla/compressor_utils.py)
  * tail builder (positions arm)    KpoolTailMetadataBuilder.build  (v1/attention/backends/mla/indexer.py)
  * prefill tail seed               kpool_seed_tail_cache           (models/glm5next/nvidia/ops/kpool_compress.py)
  * decode stash + pool completion  kpool_decode_update_and_maybe_write_cache_batched (same file)
  * reference compression           kpool_compress_and_write_cache  (same file)

The harness (scheduler, acceptance, keys) is synthetic. Per engine configuration:
  0.7.0: DCP 2 (interleave 1), tail ring = kpool = 4, DFlash K = 5 (verify window 6), per-rank caches
  0.9.1: DCP 1, tail ring 8, K = 3 (window 4)
Arms:  shipped  = positions not passed (generic per-group tail mapping, CP-masked per rank under DCP)
       positions = positions passed to the tail builder (circular per-request mapping, rank-agnostic)

Measured per decode-completed pool (completing position >= prompt length), on the rank that owns the
pool's index-K slot (0.7.0: pool p on rank p % 2; 0.9.1: single rank): is the written fp8 pool equal
to the kernel's compression of the request's committed keys of positions 4p..4p+3?  A Python shadow
of the ring addressing (transcribed from the same kernels) classifies which inputs were wrong; it is
cross-checked against the real-kernel outcome for every pool.

Usage: python3 two_rank_tailring.py --engine 070|091 --conc 1 2 8 --seeds 0 1 2 --out result.json
"""
import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("TRITON_INTERPRET", "1")
import torch  # noqa: E402

from vllm.models.glm5next.nvidia.ops.kpool_compress import (  # noqa: E402
    kpool_compress_and_write_cache,
    kpool_decode_update_and_maybe_write_cache_batched,
    kpool_seed_tail_cache,
)
from vllm.v1.attention.backend import CommonAttentionMetadata  # noqa: E402
from vllm.v1.attention.backends.mla import indexer as idxmod  # noqa: E402
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping  # noqa: E402
from vllm.v1.worker.gpu.block_table import _compute_slot_mappings_kernel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--engine", choices=["070", "091"], required=True)
ap.add_argument("--conc", type=int, nargs="+", default=[1, 2, 8])
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--arms", nargs="+", default=["shipped", "positions"])
ap.add_argument("--gen", type=int, default=160)
ap.add_argument("--accept", type=float, default=0.6)
ap.add_argument("--out", required=True)
args = ap.parse_args()

KPOOL, HD = 4, 128
SUBPAGE_POOLS = 64                     # index-K sub-page: 64 pools x 132 B = 8448 B
SUBPAGE_BYTES = SUBPAGE_POOLS * 132
CFG = {
    # page = indexer page (pools per attention block x 132 B), measured block sizes:
    # 0.7.0 attention block 3584 tokens -> 896 pools -> 118,272 B = 14 sub-pages (server log)
    # 0.9.1 attention block 4352 tokens -> 1088 pools -> 143,616 B = 17 sub-pages
    "070": dict(dcp=2, ring=4, K=5, subpages=14),
    "091": dict(dcp=1, ring=8, K=3, subpages=17),
}[args.engine]
DCP, RING, K, SUB = CFG["dcp"], CFG["ring"], CFG["K"], CFG["subpages"]
PAGE = SUB * SUBPAGE_BYTES
NB = 64                                # pages per rank
WIDTH = 4096                           # tail block-table width (real: 131072; only column 0 is ever written)
MAXTOK = 8192


def generic_tail_slots(bt, idx_mapping, qsl, pos, rank):
    """Real _compute_slot_mappings_kernel (one group: the tail), CP masking for this rank."""
    out = torch.full((1, MAXTOK), -1, dtype=torch.int64)
    ptrs = torch.tensor([bt.data_ptr()], dtype=torch.int64)
    strides = torch.tensor([bt.stride(0)], dtype=torch.int64)
    bsz = torch.tensor([RING], dtype=torch.int32)
    _compute_slot_mappings_kernel[(1, idx_mapping.numel() + 1)](
        out.shape[1], idx_mapping, qsl, pos, ptrs, strides, bsz, out, out.stride(0), rank,
        CP_SIZE=DCP, CP_INTERLEAVE=1, PAD_ID=-1, TRITON_BLOCK_SIZE=1024)
    return out[0, : int(qsl[-1])].clone()


class Rank:
    def __init__(self, r):
        self.r = r
        self.raw = torch.zeros(NB * PAGE, dtype=torch.uint8)
        self.tail = torch.as_strided(self.raw.view(torch.bfloat16), size=(NB, 2, RING, HD),
                                     stride=(PAGE // 2, RING * HD, HD, 1))
        self.idx = torch.as_strided(self.raw, size=(NB * SUB, SUBPAGE_POOLS, 132),
                                    stride=(SUBPAGE_BYTES, 132, 1))
        self.shadow = {}               # (block, ring slot) -> tag of the K/gate row stored there


def run(conc, seed, arm):
    rng = random.Random(seed * 1000 + conc)
    gen = torch.Generator().manual_seed(seed * 1000 + conc)
    ape = 0.1 * torch.randn(KPOOL, HD, generator=gen)
    reqs = []
    for i in range(conc):
        L = rng.randint(64, 400)
        n = L + args.gen + K + 8
        reqs.append(dict(
            i=i, L=L, arrive=2 * i, own=1 + i, mla=20 + 5 * i, c=None,
            kc=torch.randn(n, HD, generator=gen).to(torch.bfloat16),      # committed keys
            gc=torch.randn(n, HD, generator=gen).to(torch.bfloat16),      # committed gate scores
            done=False))
    ranks = [Rank(r) for r in range(DCP)]
    # persistent tail block table: column 0 = the request's single tail block (KpoolTailManager)
    tail_bt = torch.zeros(16, WIDTH, dtype=torch.int32)
    for q in reqs:
        tail_bt[q["i"], 0] = q["own"]
    completions = []                   # last completion record per (req, pool)
    step = 0
    while not all(q["done"] for q in reqs):
        dec = [q for q in reqs if q["c"] is not None and not q["done"]]
        pre = [q for q in reqs if q["c"] is None and q["arrive"] <= step]
        if not dec and not pre:
            step += 1
            continue
        # ---- build the batch: decode requests first (K+1 tokens each), then whole-prompt prefills
        toks = []                      # (req, pos, key, gate, tag)
        accept = {}
        for q in dec:
            c = q["c"]                 # position of the request's newest (already sampled) token
            m = 0
            while m < K and rng.random() < args.accept:
                m += 1
            accept[q["i"]] = m
            for j in range(K + 1):
                p = c + j
                if j <= m:
                    toks.append((q, p, q["kc"][p], q["gc"][p], (q["i"], p, "c")))
                else:
                    kd = torch.randn(HD, generator=gen).to(torch.bfloat16)
                    gd = torch.randn(HD, generator=gen).to(torch.bfloat16)
                    toks.append((q, p, kd, gd, (q["i"], p, f"d{step}")))
        for q in pre:
            for p in range(q["L"]):
                toks.append((q, p, q["kc"][p], q["gc"][p], (q["i"], p, "c")))
        batch = dec + pre
        lens = [K + 1] * len(dec) + [q["L"] for q in pre]
        qsl = torch.zeros(len(batch) + 1, dtype=torch.int32)
        qsl[1:] = torch.cumsum(torch.tensor(lens), 0)
        pos = torch.tensor([t[1] for t in toks], dtype=torch.int64)
        keys = torch.stack([t[2] for t in toks])
        gates = torch.stack([t[3] for t in toks])
        seq_lens = torch.tensor([int(qsl[b + 1] - qsl[b]) + (q["c"] if q["c"] is not None else 0)
                                 for b, q in enumerate(batch)], dtype=torch.int32)
        idx_mapping = torch.tensor([q["i"] for q in batch], dtype=torch.int32)
        n_dec_tok = (K + 1) * len(dec)
        ibt = torch.zeros(len(batch), 64, dtype=torch.int32)     # index-K sub-page table per batch row
        for b, q in enumerate(batch):
            ibt[b] = torch.arange(64, dtype=torch.int32) + q["mla"] * SUB
        for rk in ranks:
            ts = generic_tail_slots(tail_bt, idx_mapping, qsl, pos, rk.r)
            if arm == "positions":
                gathered = tail_bt[idx_mapping.long()]
                b = object.__new__(idxmod.KpoolTailMetadataBuilder)
                b.kv_cache_spec = type("S", (), {"block_size": RING})()
                cm = CommonAttentionMetadata(
                    query_start_loc=qsl, query_start_loc_cpu=qsl.clone(), seq_lens=seq_lens,
                    num_reqs=len(batch), num_actual_tokens=len(toks), max_query_len=max(lens),
                    max_seq_len=int(seq_lens.max()), block_table_tensor=gathered,
                    slot_mapping=ts.clone(), positions=pos)
                ts = idxmod.KpoolTailMetadataBuilder.build(b, 0, cm).slot_mapping.clone()
            csm = get_compressed_slot_mapping(len(toks), qsl, seq_lens, ibt, SUBPAGE_POOLS, KPOOL,
                                              dcp_world_size=DCP, dcp_rank=rk.r,
                                              cp_kv_cache_interleave_size=1).clone()
            # ---- decode slice (uniform [B, K+1]): stash + completion, real kernel + shadow
            if dec:
                B = len(dec)
                dts = ts[:n_dec_tok].view(B, K + 1).to(torch.int32).contiguous()
                dsm = csm[:n_dec_tok].view(B, K + 1).to(torch.int32).contiguous()
                dpos = pos[:n_dec_tok].view(B, K + 1).to(torch.int32).contiguous()
                for bi in range(B):    # shadow first (kernel programs run in request order here)
                    for t in range(K + 1):
                        k = bi * (K + 1) + t
                        tslot, cl, p = int(dts[bi, t]), int(dsm[bi, t]), int(dpos[bi, t])
                        blk = max(tslot, 0) // RING
                        if cl >= 0 and p >= 0 and p % KPOOL == KPOOL - 1:
                            start = p - KPOOL + 1
                            read = [rk.shadow.get((blk, (start + s) % RING), None) for s in range(KPOOL - 1)]
                            completions.append(dict(req=dec[bi]["i"], pool=p // KPOOL, rank=rk.r, step=step,
                                                    tail_block=blk, read=read,
                                                    want=[(dec[bi]["i"], start + s, "c") for s in range(KPOOL - 1)],
                                                    current=toks[k][4]))
                        if p >= 0 and tslot >= 0:
                            rk.shadow[(blk, p % RING)] = toks[k][4]
                kpool_decode_update_and_maybe_write_cache_batched(
                    rk.idx, rk.tail, dts, keys[:n_dec_tok].view(B, K + 1, HD).contiguous(),
                    gates[:n_dec_tok].view(B, K + 1, HD).contiguous(), ape, dsm, dpos, KPOOL, HD,
                    round_scale=True)
            # ---- prefill slice: tail seed (prompt pools are compressed by the prefill path; not modelled)
            if pre:
                tsp = ts[n_dec_tok:].contiguous()
                n = tsp.numel()
                for i in range(n):     # shadow of _kpool_tail_seed_kernel addressing
                    t = int(tsp[i])
                    if t < 0:
                        continue
                    ahead = int(tsp[i + KPOOL]) if i + KPOOL < n else -1
                    if ahead >= 0 and ahead // RING == t // RING:
                        continue
                    if args.engine == "070":
                        # 0.7.0 seed kernel addresses the tail as a dense [blocks, 2, kpool, hd] array
                        off = ((t // KPOOL) * 2 * KPOOL + t % KPOOL) * HD * 2     # bytes
                        pg, rem = divmod(off, PAGE)
                        if rem < KPOOL * HD * 2:                               # lands in a ring's K half
                            rk.shadow[(pg, rem // (HD * 2))] = toks[n_dec_tok + i][4]
                    else:
                        rk.shadow[(t // RING, t % RING)] = toks[n_dec_tok + i][4]
                kpool_seed_tail_cache(rk.tail, keys[n_dec_tok:].contiguous(), gates[n_dec_tok:].contiguous(),
                                      tsp, KPOOL, HD)
        # ---- commit
        for q in dec:
            q["c"] += accept[q["i"]] + 1
            if q["c"] >= q["L"] + args.gen:
                q["done"] = True
        for q in pre:
            q["c"] = q["L"]
        step += 1
    # ---- evaluate: final content of every decode-completed, committed pool on its owning rank
    last = {}
    for rec in completions:
        last[(rec["req"], rec["pool"])] = rec
    rows = []
    for q in reqs:
        for p in range(q["L"] // KPOOL, (q["c"]) // KPOOL):
            if KPOOL * p + KPOOL - 1 < q["L"] or KPOOL * p + KPOOL - 1 >= q["c"]:
                continue
            owner = p % DCP
            local = p // DCP
            rk = ranks[owner]
            sub = q["mla"] * SUB + local // SUBPAGE_POOLS
            off = local % SUBPAGE_POOLS
            base = sub * SUBPAGE_BYTES
            got = rk.raw[base + off * HD: base + off * HD + HD].view(torch.float8_e4m3fn).float() * \
                rk.raw[base + SUBPAGE_POOLS * HD + off * 4: base + SUBPAGE_POOLS * HD + off * 4 + 4].view(torch.float32)
            rkk, rss = kpool_compress_and_write_cache(
                rk.idx, q["kc"][KPOOL * p: KPOOL * p + KPOOL][None].contiguous(),
                q["gc"][KPOOL * p: KPOOL * p + KPOOL][None].contiguous(), ape,
                torch.zeros(1, dtype=torch.int64), KPOOL, HD, round_scale=True,
                return_compressed=True, write_cache=False)
            ref = rkk[0].float() * rss[0]
            err = float((got - ref).abs().max() / ref.abs().max())
            rec = last.get((q["i"], p))
            if rec is None:
                cls = "never-completed"
            else:
                bad = [r for r, w in zip(rec["read"], rec["want"]) if r != w]
                if not bad:
                    cls = "ok"
                else:
                    kinds = set()
                    for r in bad:
                        if r is None:
                            kinds.add("missing")
                        elif r[0] != q["i"]:
                            kinds.add("foreign")
                        elif r[2] != "c":
                            kinds.add("own-rejected-draft")
                        else:
                            kinds.add("own-other-position")
                    cls = "+".join(sorted(kinds))
            rows.append(dict(req=q["i"], pool=p, rank=owner, boundary=KPOOL * p < q["L"],
                             wrong=err > 1e-3, rel_err=round(err, 4), shadow=cls,
                             tail_block=None if rec is None else rec["tail_block"]))
    return rows, step


def summarize(rows):
    n = len(rows)
    w = sum(r["wrong"] for r in rows)
    agree = sum((r["shadow"] != "ok") == r["wrong"] for r in rows)
    per_rank = {}
    for rk in range(DCP):
        rr = [r for r in rows if r["rank"] == rk]
        per_rank[rk] = dict(pools=len(rr), wrong=sum(r["wrong"] for r in rr))
    cls = {}
    for r in rows:
        if r["wrong"]:
            cls[r["shadow"]] = cls.get(r["shadow"], 0) + 1
    bnd = [r for r in rows if r["boundary"]]
    return dict(pools=n, wrong=w, frac_wrong=round(w / n, 4) if n else None, per_rank=per_rank,
                boundary_pools=len(bnd), boundary_wrong=sum(r["wrong"] for r in bnd),
                shadow_agrees_with_kernel=f"{agree}/{n}", wrong_by_shadow_class=cls)


out = dict(engine=args.engine, cfg=dict(CFG, page_bytes=PAGE), gen=args.gen, accept=args.accept,
           note="prompts U[64,400], arrivals every 2 steps, tail blocks 1..c, index pages from block 20",
           results=[])
t0 = time.time()
for arm in args.arms:
    for conc in args.conc:
        for seed in args.seeds:
            ts0 = time.time()
            rows, steps = run(conc, seed, arm)
            s = summarize(rows)
            s.update(arm=arm, conc=conc, seed=seed, steps=steps, secs=round(time.time() - ts0, 1))
            out["results"].append(s)
            print(json.dumps(s), flush=True)
json.dump(out, open(args.out, "w"), indent=1)
print(f"done in {time.time() - t0:.0f}s", flush=True)
