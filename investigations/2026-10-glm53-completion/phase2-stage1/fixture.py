"""Phase 2 / Stage 1 fixture: which pool lands in output column 511 of the
sparse-MLA indexer pool top-k (select_k = 512), per shipped decode path.

Runs inside a release image (0.7.0 or 0.9.1) against the image's own
/opt/b12x.  No vLLM import, no model, no server, no network.

Paths driven (all through the real B12x entry points the engine calls):
  v091 : index_topk_fp8 (paged.py) after plan_indexer_scratch(...).bind(...)
         exactly as vllm sparse_attn_indexer.py:1054-1136 (_run_b12x_paged_topk)
         route auto  -> 'paged_fused' (rows<=16, SM120) or 'paged_tiled' (rows>16)
         route forced tiled via B12X_FUSED_INDEXER=0 for rows<=16
  v070 : same per-rank call on each DCP rank's local pools (pool p on rank p%2,
         local id p//2), then _merge_b12x_dcp_topk emulation
         (sparse_attn_indexer.py:1139-1218): local->global ids + -inf scores for
         invalid (torch transcription of b12x_dcp_topk.py:10-40), all-gather
         replaced by torch.cat of rank0|rank1 candidates, REAL run_row_topk over
         1024, gather-by-position (torch transcription of b12x_dcp_topk.py:176-210).
  both : run_row_topk direct on constructed score rows (score-input kernel).

Usage: python3 fixture.py --version v070|v091 --mode cpucheck|smoke|full --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

import torch

TOPK = 512            # select_k = index_topk 2048 / index_kpool 4
KEEP = TOPK - 1       # engine keeps pool_ids[:, :511]
HEADS = 32
HD = 128
PAGE = 64             # PAGED_INDEX_PAGE_SIZE (pools per page)
PAGE_BYTES = PAGE * (HD + 4)
FP8_MAX = 448.0
ROWS_PER_REQ = 4      # rows grouped into "requests" sharing one K cache (same for both versions)
REPEATS = 5
# Page-table width (columns) as the engine would size it: max_model_len 1,048,576 tokens
# / (64 pools * 4 tokens) = 4096 (DCP1); per-rank half under DCP2 = 2048 (inferred).
WIDTH = {"v091": 4096, "v070": 2048}
EPS32 = 2.0 ** -24

POOLS = [512, 513, 520, 1024, 2048, 4096, 16384, 30000]
ROWS = [1, 4, 8, 16, 24, 32, 48, 64]


# --------------------------------------------------------------------------------------
# Input construction (engine-style quantization, emulated in torch)
# --------------------------------------------------------------------------------------
def hadamard128(device):
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < HD:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h.to(device)


def fwht_quant(x_bf16, H):
    """Emulates fwht128_quant_fp8 (kpool_compress.py v091:79-152): Hadamard-128 in
    fp32, *1/sqrt(128), bf16 round, ue8m0 absmax scale, clamp +-448, e4m3."""
    x = x_bf16.float() @ H
    x = (x * 0.08838834764831845).to(torch.bfloat16).float()
    absmax = x.abs().amax(-1).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(absmax * (1.0 / FP8_MAX))))
    y = (x / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return y, scale


def draw(shape, dist, gen, device):
    z = torch.randn(shape, generator=gen, device=device)
    if dist == "gauss":
        return z
    if dist == "heavy":  # Student-t, df=3
        c = torch.randn((*shape, 3), generator=gen, device=device).pow(2).sum(-1) / 3.0
        return z / c.sqrt()
    raise ValueError(dist)


def make_random_inputs(n_pools, rows, dist, seed, device):
    """q (rows,32,128) fp8, w (rows,32) fp32, per-request K (N,128) fp8 + scale (N,)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    H = hadamard128(device)
    nreq = (rows + ROWS_PER_REQ - 1) // ROWS_PER_REQ
    ks, kscales = [], []
    for _ in range(nreq):
        kb = draw((n_pools, HD), dist, gen, device).to(torch.bfloat16)
        k8, kscale = fwht_quant(kb, H)
        ks.append(k8)
        kscales.append(kscale)
    qb = draw((rows, HEADS, HD), dist, gen, device).to(torch.bfloat16)
    q8, qscale = fwht_quant(qb.reshape(rows * HEADS, HD), H)
    q8 = q8.reshape(rows, HEADS, HD)
    qscale = qscale.reshape(rows, HEADS)
    wraw = draw((rows, HEADS), dist, gen, device).to(torch.bfloat16).float()
    # attention.py v091:63-67 / :391-396: weights * q_scale * softmax_scale * n_head**-0.5
    w = (wraw * qscale * (HD ** -0.5) * (HEADS ** -0.5)).float().contiguous()
    req_of_row = [i // ROWS_PER_REQ for i in range(rows)]
    return dict(q=q8.contiguous(), w=w, ks=ks, kscales=kscales, req_of_row=req_of_row)


def _fp8_combo_values(device):
    """512 distinct exactly-representable v = a + b/16 + c/256, a,b,c in e4m3 [1,2)."""
    m = torch.tensor([1.0 + i / 8.0 for i in range(8)], dtype=torch.float32)
    a, b, c = torch.meshgrid(m, m, m, indexing="ij")
    return torch.stack([a.flatten(), b.flatten(), c.flatten()], 1).to(device)  # (512,3)


def make_constructed_inputs(case, n_pools, rows, seed, device):
    """Controlled exact scores: head 0 only (w=[1,0..]), q0=[1, 2^-4, 2^-8, 0...].
    score_p = kscale_p * ReLU(k0 + k1/16 + k2/256) -- exact in fp32.
      a: 512 distinct pools in [2.13,3.98] (above fp16 coarse boundary 2.0), rest < 2.0
      b: N=512 valid pools only (rest of the row is beyond seq_len), distinct scores
      c: 500 distinct pools > 4, 24-pool plateau at exactly 4.0 (ranks 500..523), rest < 4
      d: 300 distinct positive pools, the rest exactly 0 (ReLU plateau)
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    combos = _fp8_combo_values("cpu")
    nreq = (rows + ROWS_PER_REQ - 1) // ROWS_PER_REQ
    ks, kscales = [], []
    for _ in range(nreq):
        k = torch.zeros((n_pools, HD), dtype=torch.float32)
        sc = torch.ones((n_pools,), dtype=torch.float32)
        perm = torch.randperm(n_pools, generator=gen)  # which pool ids get which role
        rest_combo = combos[torch.randint(0, 512, (n_pools,), generator=gen)]
        k[:, :3] = rest_combo  # default role: "rest", scale 1 -> [1.07, 1.99]
        if case == "a":
            top = perm[:512]
            k[top, :3] = combos[torch.randperm(512, generator=gen)]
            sc[top] = 2.0
        elif case == "b":
            assert n_pools == 512
            k[:, :3] = combos[torch.randperm(512, generator=gen)]
        elif case == "c":
            top = perm[:500]
            plat = perm[500:524]
            k[top, :3] = combos[torch.randperm(512, generator=gen)[:500]]
            sc[top] = 4.0                         # [4.26, 7.97]
            k[plat, :3] = torch.tensor([1.0, 0.0, 0.0])
            sc[plat] = 4.0                        # exactly 4.0
            sc[perm[524:]] = 2.0                  # rest [2.13, 3.98]
        elif case == "d":
            pos = perm[:300]
            k[pos, :3] = combos[torch.randperm(512, generator=gen)[:300]]
            zero = perm[300:]
            k[zero, :3] = torch.tensor([-1.0, 0.0, 0.0])  # q.k = -1 -> ReLU 0
        else:
            raise ValueError(case)
        ks.append(k.to(torch.float8_e4m3fn).to(device))
        kscales.append(sc.to(device))
    q = torch.zeros((rows, HEADS, HD), dtype=torch.float32)
    q[:, 0, 0] = 1.0
    q[:, 0, 1] = 2.0 ** -4
    q[:, 0, 2] = 2.0 ** -8
    w = torch.zeros((rows, HEADS), dtype=torch.float32)
    w[:, 0] = 1.0
    req_of_row = [i // ROWS_PER_REQ for i in range(rows)]
    return dict(q=q.to(torch.float8_e4m3fn).to(device).contiguous(), w=w.to(device).contiguous(),
                ks=ks, kscales=kscales, req_of_row=req_of_row)


# --------------------------------------------------------------------------------------
# Reference (float64 from the exact dequantized kernel inputs) + fp32 error bound
# --------------------------------------------------------------------------------------
def reference_scores(inp, device):
    """Return per-row (score64 (N,), bound64 (N,)) where score = kscale*sum_h w_h relu(q_h.k)
    and bound = 162*2^-24*kscale*sum_h |w_h| (|q_h|.|k|) (fp32 evaluation-error bound
    for any summation order over 128 dims + 32 heads + 2 multiplies)."""
    out = []
    for i, rq in enumerate(inp["req_of_row"]):
        q = inp["q"][i].double()                     # (32,128)
        w = inp["w"][i].double()                     # (32,)
        k = inp["ks"][rq].double()                   # (N,128)
        ksc = inp["kscales"][rq].double()            # (N,)
        dots = q @ k.T                               # (32,N)
        s = (torch.relu(dots) * w[:, None]).sum(0) * ksc
        absd = q.abs() @ k.abs().T
        b = (absd * w.abs()[:, None]).sum(0) * ksc * (162 * EPS32)
        out.append((s, b))
    return out


def per_rank_split(inp, rank, world=2):
    ks = [k[rank::world].contiguous() for k in inp["ks"]]
    sc = [s[rank::world].contiguous() for s in inp["kscales"]]
    return dict(q=inp["q"], w=inp["w"], ks=ks, kscales=sc, req_of_row=inp["req_of_row"])


# --------------------------------------------------------------------------------------
# Paged cache packing (planar page: 64x128 e4m3 bytes, then 64 fp32 scales)
# --------------------------------------------------------------------------------------
def pack_cache(inp, rows, width, device):
    """Page 0 is a null page (vLLM pads block tables with 0). Each request gets its own
    contiguous run of pages; every row of a request points at that run."""
    pages_per_req = [(k.shape[0] + PAGE - 1) // PAGE for k in inp["ks"]]
    total = 1 + sum(pages_per_req)
    cache = torch.zeros((total, PAGE_BYTES), dtype=torch.uint8, device=device)
    data = cache[:, : PAGE * HD].view(total, PAGE, HD)
    scales = cache[:, PAGE * HD:].view(torch.float32)  # (total, 64)
    starts = []
    nxt = 1
    for k, s, p in zip(inp["ks"], inp["kscales"], pages_per_req):
        n = k.shape[0]
        kb = torch.zeros((p * PAGE, HD), dtype=torch.uint8, device=device)
        kb[:n] = k.view(torch.uint8)
        sb = torch.zeros((p * PAGE,), dtype=torch.float32, device=device)
        sb[:n] = s.float()
        data[nxt: nxt + p] = kb.view(p, PAGE, HD)
        scales[nxt: nxt + p] = sb.view(p, PAGE)
        starts.append(nxt)
        nxt += p
    bt = torch.zeros((rows, width), dtype=torch.int32, device=device)
    for i, rq in enumerate(inp["req_of_row"]):
        p = pages_per_req[rq]
        assert p <= width
        bt[i, :p] = torch.arange(starts[rq], starts[rq] + p, dtype=torch.int32, device=device)
    seqlens = torch.tensor([inp["ks"][rq].shape[0] for rq in inp["req_of_row"]],
                           dtype=torch.int32, device=device)
    return cache, bt.contiguous(), seqlens


# --------------------------------------------------------------------------------------
# Kernel drivers
# --------------------------------------------------------------------------------------
def b12x_paged_topk(q, w, cache, bt, seqlens, with_scores, force_tiled):
    """Mirror of vllm sparse_attn_indexer.py:1054-1136 (_run_b12x_paged_topk), with
    scratch from torch.empty instead of the workspace manager, schedule_metadata=None
    and active_width=None (engine: getattr(decode_metadata,'active_width',None) -> None)."""
    from b12x.attention.dsa_indexer.paged import PAGED_INDEX_PAGE_SIZE, index_topk_fp8
    from b12x.attention.dsa_indexer.scratch import (
        INDEXER_SOURCE_LAYOUT_PAGED, B12XIndexerScratchCaps, plan_indexer_scratch)
    rows = int(q.shape[0])
    old = os.environ.get("B12X_FUSED_INDEXER")
    if force_tiled:
        os.environ["B12X_FUSED_INDEXER"] = "0"
    try:
        plan = plan_indexer_scratch(B12XIndexerScratchCaps(
            device=q.device, source_layout=INDEXER_SOURCE_LAYOUT_PAGED, num_q_heads=HEADS,
            max_q_rows=rows, max_page_table_width=int(bt.shape[1]), topk=TOPK,
            mode="decode", shared_page_table=False))
        scratch = [torch.empty(s, dtype=d, device=q.device) for s, d in plan.shapes_and_dtypes()]
        binding = plan.bind(scratch=scratch, real_page_table=bt, cache_seqlens_int32=seqlens,
                            active_width=None, schedule_metadata=None,
                            expected_num_q_heads=HEADS, shared_page_table=False,
                            output_physical_slots=False)
    finally:
        if force_tiled:
            if old is None:
                os.environ.pop("B12X_FUSED_INDEXER", None)
            else:
                os.environ["B12X_FUSED_INDEXER"] = old
    out = torch.full((rows, TOPK), -1, dtype=torch.int32, device=q.device)  # pool_topk init (kpool.py)
    sc = torch.empty((rows, TOPK), dtype=torch.float32, device=q.device) if with_scores else None
    index_topk_fp8(q_fp8=q, weights=w, index_k_cache=cache, binding=binding,
                   page_size=PAGED_INDEX_PAGE_SIZE, expected_num_q_heads=HEADS,
                   out_indices=out, out_scores=sc)
    route = str(getattr(binding, "route", "?"))
    s = getattr(binding, "scratch", None)
    meta = dict(route=route,
                ctas_per_group=getattr(s, "fused_ctas_per_group", None),
                merge_threshold=getattr(s, "fused_merge_threshold", None))
    return out, sc, meta


def dcp2_merge(idx_r, sc_r):
    """_merge_b12x_dcp_topk (sparse_attn_indexer.py:1139-1218) with all-gather -> cat."""
    from b12x.attention.dsa_indexer.tiled_topk import run_row_topk
    gl, gs = [], []
    for r, (idx, sc) in enumerate(zip(idx_r, sc_r)):
        valid = idx >= 0
        gl.append(torch.where(valid, idx * 2 + r, torch.full_like(idx, -1)))
        gs.append(torch.where(valid, sc, torch.full_like(sc, float("-inf"))))
    cand_idx = torch.cat(gl, 1).contiguous()
    cand_sc = torch.cat(gs, 1).contiguous()
    rows = cand_idx.shape[0]
    lengths = torch.full((rows,), cand_idx.shape[1], dtype=torch.int32, device=cand_idx.device)
    pos = torch.empty((rows, TOPK), dtype=torch.int32, device=cand_idx.device)
    vals = torch.empty((rows, TOPK), dtype=torch.float32, device=cand_idx.device)
    run_row_topk(row_logits=cand_sc, lengths=lengths, topk=TOPK, output_values=vals,
                 output_indices=pos)
    p = pos.long()
    ok = (p >= 0) & (p < cand_idx.shape[1])
    out = torch.where(ok, torch.gather(cand_idx, 1, p.clamp(0, cand_idx.shape[1] - 1)),
                      torch.full_like(pos, -1))
    return out, pos


def _packed(inp, key, builder):
    memo = inp.setdefault("_packed", {})
    if key not in memo:
        memo[key] = builder()
    return memo[key]


TIMING = {"kernel_path_s": 0.0, "calls": 0}


def run_path(version, path, inp, rows, device):
    """Returns final (rows,512) pool ids (global for v070) + meta. Cache packing is done
    once per input draw; plan/bind/launch are repeated every call (as the engine does)."""
    t0 = time.time()
    if version == "v091":
        cache, bt, sl = _packed(inp, "v091", lambda: pack_cache(inp, rows, WIDTH["v091"], device))
        out, _, meta = b12x_paged_topk(inp["q"], inp["w"], cache, bt, sl, False,
                                       force_tiled=(path == "tiled_forced"))
        res = (out, meta)
    else:
        # v070 DCP2
        idx_r, sc_r, metas = [], [], []
        for r in range(2):
            def build(r=r):
                rin = per_rank_split(inp, r)
                return (rin,) + pack_cache(rin, rows, WIDTH["v070"], device)
            rin, cache, bt, sl = _packed(inp, ("v070", r), build)
            out, sc, meta = b12x_paged_topk(rin["q"], rin["w"], cache, bt, sl, True,
                                            force_tiled=(path == "tiled_forced"))
            idx_r.append(out)
            sc_r.append(sc)
            metas.append(meta)
        merged, _ = dcp2_merge(idx_r, sc_r)
        res = (merged, dict(rank_routes=[m["route"] for m in metas],
                            ctas_per_group=metas[0]["ctas_per_group"],
                            merge_threshold=metas[0]["merge_threshold"],
                            rank_scores=sc_r, rank_idx=idx_r))
    if device.type == "cuda":
        torch.cuda.synchronize()
    TIMING["kernel_path_s"] += time.time() - t0
    TIMING["calls"] += 1
    return res


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def prep_ref(ref):
    """Stack per-row reference (score, bound) and pre-sort (on the run device)."""
    S = torch.stack([s for s, _ in ref])          # (R,N) float64
    B = torch.stack([b for _, b in ref])
    srt = torch.sort(S, dim=1, descending=True, stable=True)
    n = S.shape[1]
    k512 = min(TOPK, n) - 1
    k511 = min(KEEP, n) - 1
    i512 = srt.indices[:, k512]
    i511 = srt.indices[:, k511]
    return dict(S=S, B=B, ss=srt.values, n=n,
                t512=srt.values[:, k512], t511=srt.values[:, k511],
                b512=B.gather(1, i512[:, None])[:, 0], b511=B.gather(1, i511[:, None])[:, 0])


def _sel_stats(sel, S, B, t, bt, expected):
    """sel (R,K) int64 -> (ok_exact, ok_tol, n_valid, dup, oor, n_missing_strict, n_missing_tol)."""
    R, n = S.shape
    valid = sel >= 0
    inr = valid & (sel < n)
    oor = (valid & ~inr).sum(1)
    n_valid = valid.sum(1)
    sv = torch.sort(torch.where(inr, sel, torch.full_like(sel, -1)), dim=1).values
    dup = ((sv[:, 1:] == sv[:, :-1]) & (sv[:, 1:] >= 0)).sum(1)
    safe = sel.clamp(0, n - 1)
    mask = torch.zeros((R, n), dtype=torch.bool, device=S.device)
    mask.scatter_(1, torch.where(inr, safe, torch.zeros_like(safe)), inr)
    vs = S.gather(1, safe)
    vb = B.gather(1, safe)
    must_ex = S > t[:, None]
    must_tol = S > (t[:, None] + B + bt[:, None])
    miss_ex = (must_ex & ~mask).sum(1)
    miss_tol = (must_tol & ~mask).sum(1)
    low_ex = (inr & (vs < t[:, None])).sum(1)
    low_tol = (inr & (vs < t[:, None] - vb - bt[:, None])).sum(1)
    base = (n_valid == expected) & (dup == 0) & (oor == 0)
    ok_ex = base & (miss_ex == 0) & (low_ex == 0)
    ok_tol = base & (miss_tol == 0) & (low_tol == 0)
    return ok_ex, ok_tol, n_valid, dup, oor, miss_ex, miss_tol


def batched_metrics(P, sel):
    """P: prep_ref dict; sel: (R,512) int64 on the same device. Returns list of dicts."""
    S, B, n = P["S"], P["B"], P["n"]
    R = S.shape[0]
    ok_ex, ok_tol, n_valid, dup, oor, miss_ex, miss_tol = _sel_stats(
        sel, S, B, P["t512"], P["b512"], min(TOPK, n))
    f_ex, f_tol, _, _, _, _, _ = _sel_stats(sel[:, :KEEP], S, B, P["t511"], P["b511"], min(KEEP, n))
    c = sel[:, KEEP]
    cvalid = (c >= 0) & (c < n)
    cs = S.gather(1, c.clamp(0, n - 1)[:, None])[:, 0]
    cb = B.gather(1, c.clamp(0, n - 1)[:, None])[:, 0]
    crank = (S > cs[:, None]).sum(1)
    tie_ex = cs == P["t512"]
    tie_tol = (cs - P["t512"]).abs() <= cb + P["b512"]
    gap = (P["ss"][:, TOPK - 1] - P["ss"][:, TOPK]) if n > TOPK else None
    n_eq = (S == P["t512"][:, None]).sum(1)
    n_zero = (S == 0).sum(1)
    cols = [x.cpu().tolist() for x in (ok_ex, ok_tol, n_valid, dup, oor, miss_ex, miss_tol, f_ex, f_tol,
                                        c, cvalid, crank, tie_ex, tie_tol, cs - P["t512"], n_eq, n_zero)]
    gapl = gap.cpu().tolist() if gap is not None else [None] * R
    out = []
    for i in range(R):
        (a_ok_ex, a_ok_tol, a_nv, a_dup, a_oor, a_mex, a_mtol, a_fex, a_ftol, a_c, a_cv, a_cr, a_te,
         a_tt, a_gap, a_neq, a_nz) = (col[i] for col in cols)
        out.append(dict(
            n_pools=n, n_valid=a_nv, dup=a_dup, out_of_range=a_oor,
            set_ok_exact=bool(a_ok_ex), set_ok_tol=bool(a_ok_tol),
            n_missing_strict=a_mex, n_missing_tol=a_mtol,
            col511_id=a_c, col511_rank=(a_cr if a_cv else None),
            col511_tie_exact=bool(a_te and a_cv), col511_tie_tol=bool(a_tt and a_cv),
            col511_gap=(a_gap if a_cv else None),
            first511_ok_exact=bool(a_fex), first511_ok_tol=bool(a_ftol),
            gap_512_513=gapl[i], n_eq_t512=a_neq, n_zero=a_nz))
    return out


def summarize(cell_rows):
    """cell_rows: list of per-(row,rep) metric dicts."""
    n = len(cell_rows)
    viol = [r for r in cell_rows
            if r["col511_rank"] is not None and r["col511_rank"] < KEEP and not r["col511_tie_exact"]]
    viol_tol = [r for r in cell_rows
                if r["col511_rank"] is not None and r["col511_rank"] < KEEP and not r["col511_tie_tol"]]
    ranks = sorted(r["col511_rank"] for r in cell_rows if r["col511_rank"] is not None)

    def q(p):
        if not ranks:
            return None
        return ranks[min(len(ranks) - 1, int(math.floor(p * (len(ranks) - 1))))]
    return dict(
        n=n,
        frac_col511_bad=len(viol) / n if n else None,
        frac_col511_bad_tol=len(viol_tol) / n if n else None,
        frac_col511_rank_lt_256=sum(1 for r in viol_tol if r["col511_rank"] < 256) / n if n else None,
        rank_min=ranks[0] if ranks else None, rank_p10=q(0.10), rank_median=q(0.5),
        set_mismatch_exact=sum(1 for r in cell_rows if not r["set_ok_exact"]),
        set_mismatch_tol=sum(1 for r in cell_rows if not r["set_ok_tol"]),
        first511_mismatch_exact=sum(1 for r in cell_rows if not r["first511_ok_exact"]),
        first511_mismatch_tol=sum(1 for r in cell_rows if not r["first511_ok_tol"]),
        dup=sum(r["dup"] for r in cell_rows), out_of_range=sum(r["out_of_range"] for r in cell_rows),
    )


# --------------------------------------------------------------------------------------
# Experiment loops
# --------------------------------------------------------------------------------------
def paths_for(version, rows):
    if rows <= 16:
        return ["auto", "tiled_forced"]
    return ["auto"]


def seed_for(kind, n_idx, rows, draw_i):
    return 7_000_000 + kind * 100_000 + n_idx * 1_000 + rows * 10 + draw_i


def run_cell(version, kind_name, inp_builder, n_pools, rows, ndraws, device, log, raw):
    """Generate ndraws inputs, run every path REPEATS times, record metrics."""
    cells = {}
    for d in range(ndraws):
        seed, inp = inp_builder(d)
        ref = reference_scores(inp, device)
        P = prep_ref(ref)
        for path in paths_for(version, rows):
            key = (kind_name, path, n_pools, rows)
            per_rep_ids = []
            route_meta = None
            for rep in range(REPEATS):
                out, meta = run_path(version, path, inp, rows, device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                sel = out.long()
                if rep == 0:
                    route_meta = {k: v for k, v in meta.items() if k not in ("rank_scores", "rank_idx")}
                    if version == "v070":
                        # per-rank kernel scores vs fp64 reference, in units of the fp32 bound
                        worst = 0.0
                        for r in range(2):
                            ridx = meta["rank_idx"][r].long()
                            rsc = meta["rank_scores"][r].double()
                            Sl, Bl = P["S"][:, r::2], P["B"][:, r::2]
                            v = (ridx >= 0) & (ridx < Sl.shape[1])
                            gi = ridx.clamp(0, Sl.shape[1] - 1)
                            e = (rsc - Sl.gather(1, gi)).abs() / Bl.gather(1, gi).clamp_min(1e-300)
                            e = torch.where(v, e, torch.zeros_like(e))
                            worst = max(worst, e.max().item())
                        route_meta["score_err_over_bound_max"] = worst
                per_rep_ids.append(sel.cpu())
                if d == 0 and rep < 2:
                    # audit example: row 0's 512 ids with their fp64 reference scores
                    ids0 = sel[0]
                    okm = (ids0 >= 0) & (ids0 < P["n"])
                    sc0 = torch.where(okm, P["S"][0].gather(0, ids0.clamp(0, P["n"] - 1)),
                                      torch.full_like(P["S"][0][:1].expand(ids0.shape[0]), float("nan")))
                    cells.setdefault(key, {"rows": [], "meta": route_meta, "rtr": []}).setdefault(
                        "examples", []).append(dict(rep=rep, seed=seed, ids=ids0.cpu().tolist(),
                                                    ref_scores=sc0.cpu().tolist(),
                                                    t512=P["t512"][0].item(),
                                                    top512_sorted=P["ss"][0, :min(TOPK + 2, P["n"])].cpu().tolist()))
                for i, m in enumerate(batched_metrics(P, sel)):
                    m.update(draw=d, seed=seed, row=i, rep=rep)
                    cells.setdefault(key, {"rows": [], "meta": route_meta, "rtr": []})["rows"].append(m)
            # run-to-run variation per input row
            ids = torch.stack(per_rep_ids)  # (REPEATS, rows, 512)
            for i in range(rows):
                col = ids[:, i, KEEP]
                order_var = bool((ids[:, i, :] != ids[0, i, :]).any().item())
                sets = [tuple(sorted(x.tolist())) for x in ids[:, i, :]]
                cells[key]["rtr"].append(dict(
                    col511_distinct=int(torch.unique(col).numel()),
                    order_varies=order_var,
                    set_varies=len(set(sets)) > 1))
            cells[key]["meta"] = route_meta
    out = []
    for key, c in cells.items():
        kind_name, path, n, r = key
        summ = summarize(c["rows"])
        rtr = c["rtr"]
        summ.update(
            rtr_rows=len(rtr),
            rtr_frac_col511_varies=sum(1 for x in rtr if x["col511_distinct"] > 1) / len(rtr),
            rtr_frac_order_varies=sum(1 for x in rtr if x["order_varies"]) / len(rtr),
            rtr_frac_set_varies=sum(1 for x in rtr if x["set_varies"]) / len(rtr))
        rec = dict(version=version, kind=kind_name, path=path, pools=n, rows=r, meta=c["meta"], **summ)
        out.append(rec)
        raw.write(json.dumps(dict(version=version, kind=kind_name, path=path, pools=n, rows=r,
                                  per_row=c["rows"], rtr=rtr, examples=c.get("examples", []))) + "\n")
        log(json.dumps({k: v for k, v in rec.items()}))
    return out


def row_topk_direct(version, device, log):
    """run_row_topk on constructed score rows (width 1024, the 0.7.0 merge shape)."""
    from b12x.attention.dsa_indexer.tiled_topk import run_row_topk
    W = 1024
    res = []
    gen = torch.Generator(device="cpu")
    for case in ["a", "b", "c", "d"]:
        for rows in [1, 6, 24, 48]:
            gen.manual_seed(9_100_000 + ord(case) * 100 + rows)
            x = torch.empty((rows, W), dtype=torch.float32)
            for i in range(rows):
                perm = torch.randperm(W, generator=gen)
                row = torch.empty(W, dtype=torch.float32)
                base = (torch.arange(512, dtype=torch.float32) + 1.0) / 512.0  # distinct
                if case == "a":   # 512 in [2,4) (distinct), 512 in [1,2) -> coarse fp16 boundary 2.0
                    row[perm[:512]] = 2.0 + 1.9 * base[torch.randperm(512, generator=gen)]
                    row[perm[512:]] = 1.0 + 0.9 * base[torch.randperm(512, generator=gen)]
                elif case == "b":  # 512 finite + 512 -inf (padding)
                    row[perm[:512]] = 1.0 + base[torch.randperm(512, generator=gen)]
                    row[perm[512:]] = float("-inf")
                elif case == "c":  # 500 distinct high, plateau of 24 at 4.0, rest below
                    row[perm[:500]] = 4.5 + base[torch.randperm(512, generator=gen)[:500]]
                    row[perm[500:524]] = 4.0
                    row[perm[524:]] = 1.0 + 2.9 * base[torch.randperm(512, generator=gen)[:500]]
                elif case == "d":  # 300 positive, rest exactly 0
                    row[perm[:300]] = 1.0 + base[torch.randperm(512, generator=gen)[:300]]
                    row[perm[300:]] = 0.0
                x[i] = row
            xs = x.to(device).contiguous()
            lengths = torch.full((rows,), W, dtype=torch.int32, device=device)
            reps = []
            for rep in range(REPEATS):
                pos = torch.empty((rows, TOPK), dtype=torch.int32, device=device)
                vals = torch.empty((rows, TOPK), dtype=torch.float32, device=device)
                run_row_topk(row_logits=xs, lengths=lengths, topk=TOPK, output_values=vals,
                             output_indices=pos)
                reps.append(pos.long().cpu())
            ms = []
            S = torch.where(torch.isinf(x), torch.full_like(x, -1e30), x).double().to(device)
            P = prep_ref([(S[i], torch.zeros_like(S[i])) for i in range(rows)])
            for rep, pos in enumerate(reps):
                for i, m in enumerate(batched_metrics(P, pos.to(device))):
                    m.update(row=i, rep=rep)
                    ms.append(m)
            summ = summarize(ms)
            ids = torch.stack(reps)
            summ["rtr_frac_col511_varies"] = sum(
                1 for i in range(rows) if torch.unique(ids[:, i, KEEP]).numel() > 1) / rows
            rec = dict(version=version, kind=f"row_topk_direct_{case}", path="run_row_topk",
                       pools=W, rows=rows, **summ)
            res.append(rec)
            log(json.dumps(rec))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["v070", "v091"])
    ap.add_argument("--mode", required=True, choices=["cpucheck", "smoke", "full"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-rows", type=int, default=64)
    ap.add_argument("--max-seconds", type=float, default=330.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.version}-{args.mode}"
    logf = open(os.path.join(args.out, f"log-{tag}.txt"), "w")
    rawf = open(os.path.join(args.out, f"raw-{tag}.jsonl"), "w")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    t0 = time.time()
    import b12x
    log(f"python {sys.version.split()[0]} torch {torch.__version__} b12x {b12x.__file__}")
    try:
        import subprocess
        log("b12x git: " + subprocess.run(["git", "-C", "/opt/b12x", "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip())
    except Exception as e:  # noqa
        log(f"git rev-parse failed: {e}")

    if args.mode == "cpucheck":
        cpu_check(log)
        return

    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    log(f"device {props.name} sm{props.major}{props.minor} SMs={props.multi_processor_count}")
    summary = []
    try:
        if args.mode == "smoke":
            pools_list, rows_list, target = [1024], [1, 4, 24], 4
            constructed = [("a", 1024)]
            direct = False
        else:
            pools_list, rows_list, target = POOLS, ROWS, args.target_rows
            constructed = [("a", 1024), ("a", 16384), ("b", 512), ("c", 1024), ("c", 16384),
                           ("d", 1024), ("d", 16384)]
            direct = True
        def random_block(dists):
          for dist in dists:
            kind = 1 if dist == "gauss" else 2
            for n_idx, n in enumerate(pools_list):
                for rows in rows_list:
                    tgt = target if dist == "gauss" else max(1, target // 2)
                    ndraws = max(1, (tgt + rows - 1) // rows)
                    tc = time.time()
                    if tc - t0 > args.max_seconds:
                        raise TimeoutError("time guard hit before random cell")

                    def builder(d, n=n, rows=rows, n_idx=n_idx, kind=kind, dist=dist):
                        sd = seed_for(kind, n_idx, rows, d)
                        return sd, make_random_inputs(n, rows, dist, sd, device)
                    summary.extend(run_cell(args.version, f"random_{dist}", builder, n, rows, ndraws,
                                        device, log, rawf))
                    log(f"cell dist={dist} N={n} rows={rows} draws={ndraws} {time.time()-tc:.1f}s timing={TIMING}")
        random_block(["gauss"])
        for case, n in constructed:
            for rows in ([1, 4, 16, 24, 64] if args.mode == "full" else [1, 24]):
                ndraws = max(1, (16 + rows - 1) // rows) if args.mode == "full" else 1

                def builder(d, n=n, rows=rows, case=case):
                    sd = 8_000_000 + ord(case) * 10_000 + rows * 10 + d
                    return sd, make_constructed_inputs(case, n, rows, sd, device)
                tc = time.time()
                summary += run_cell(args.version, f"constructed_{case}", builder, n, rows, ndraws,
                                    device, log, rawf)
                log(f"cell constructed {case} N={n} rows={rows} {time.time()-tc:.1f}s")
        if direct:
            summary += row_topk_direct(args.version, device, log)
        if args.mode == "full":
            random_block(["heavy"])
    except TimeoutError as e:
        log(f"STOPPED EARLY: {e}")
    except Exception:
        log("ERROR\n" + traceback.format_exc())
        raise
    finally:
        with open(os.path.join(args.out, f"summary-{tag}.json"), "w") as f:
            json.dump(summary, f, indent=1)
        log(f"done in {time.time()-t0:.1f}s")


def cpu_check(log):
    """No kernels: validate packing + reference against b12x's own torch references."""
    from b12x.attention.dsa_indexer.paged import paged_index_logits_reference
    from b12x.attention.dsa_indexer.scratch import (
        INDEXER_SOURCE_LAYOUT_PAGED, B12XIndexerScratchCaps, plan_indexer_scratch)  # noqa
    dev = torch.device("cpu")
    for dist in ["gauss", "heavy"]:
        inp = make_random_inputs(700, 5, dist, 123, dev)
        cache, bt, sl = pack_cache(inp, 5, 64, dev)
        ref = reference_scores(inp, dev)
        lg = paged_index_logits_reference(
            q_fp8=inp["q"], weights=inp["w"], index_k_cache=cache, real_page_table=bt,
            query_row_to_batch=torch.arange(5, dtype=torch.int32), seqlens_per_query=sl)
        worst = 0.0
        for i in range(5):
            s, b = ref[i]
            e = (lg[i, :700].double() - s).abs() / b
            worst = max(worst, e.max().item())
            assert torch.isinf(lg[i, 700:]).all()
        log(f"cpucheck {dist}: b12x torch reference vs fp64 ref max err/bound = {worst:.3g}; "
            f"score range [{min(r[0].min().item() for r in ref):.3g},"
            f"{max(r[0].max().item() for r in ref):.3g}]")
        r0 = ref[0][0]
        srt = torch.sort(r0, descending=True).values
        log(f"  row0 top={srt[0]:.4g} s512={srt[511]:.4g} gap512_513={srt[511]-srt[512]:.3g} "
            f"zeros={(r0 == 0).sum().item()} neg={(r0 < 0).sum().item()}")
        # metric self-test: a perfect selection sorted descending has col511 = rank 511
        ids = torch.sort(r0, descending=True, stable=True).indices[:512]
        P0 = prep_ref([ref[0]])
        m = batched_metrics(P0, ids[None])[0]
        assert m["set_ok_exact"] and m["first511_ok_exact"] and m["col511_rank"] == 511, m
        ids2 = ids.clone()
        ids2[[0, 511]] = ids2[[511, 0]]
        m2 = batched_metrics(P0, ids2[None])[0]
        assert m2["set_ok_exact"] and not m2["first511_ok_exact"] and m2["col511_rank"] == 0, m2
        # v070 split sanity
        r_in = per_rank_split(inp, 1)
        assert r_in["ks"][0].shape[0] == 350
    for case, n in [("a", 1024), ("b", 512), ("c", 1024), ("d", 1024)]:
        inp = make_constructed_inputs(case, n, 4, 5, dev)
        cache, bt, sl = pack_cache(inp, 4, 64, dev)
        ref = reference_scores(inp, dev)
        lg = paged_index_logits_reference(
            q_fp8=inp["q"], weights=inp["w"], index_k_cache=cache, real_page_table=bt,
            query_row_to_batch=torch.arange(4, dtype=torch.int32), seqlens_per_query=sl)
        s = ref[0][0]
        exact = bool(torch.equal(lg[0, :n].double(), s))
        srt = torch.sort(s, descending=True).values
        log(f"cpucheck constructed {case}: fp32 ref == fp64 ref: {exact}; distinct={torch.unique(s).numel()} "
            f"s[0]={srt[0]:.6g} s[499]={srt[499]:.6g} s[500]={srt[500]:.6g} s[511]={srt[511]:.6g} "
            f"s[512]={srt[512] if n > 512 else float('nan'):.6g} zeros={(s == 0).sum().item()}")
    log("cpucheck OK")


if __name__ == "__main__":
    main()
