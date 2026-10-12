#!/usr/bin/env python3
"""APPLIED under PLAN.md amendment 14 (before any D3 comparison): routing in control (c) and the
corrected routing in phase 2 are invoked with the serving batch row count M (captured rows at
[a, a+r), zero padding, server alignment, same router configuration), exactly as the GEMM in
control (b), because the served fused grouped_topk's weights depend on the batch row count
(finding E1; offline attempt 3).  Criterion of control (c) unchanged (bitwise); stop rule unchanged.

D3 offline comparison (PLAN.md amendments 13 / 13a): router logits from three labelled
computations on the identical captured gate inputs, routed by the engine's own router.

  SHIPPED   "shipped: bf16-result GEMM cast to fp32"  - the engine's GateLinear on SM120 (tier 6)
  CORRECTED "corrected: fp32-output GEMM (upstream #54048 form)" - torch.mm(x, W.T, out_dtype=fp32)
  REFERENCE "reference: fp64"                          - x.double() @ W.double().T

Controls, fixed before data (PROPOSAL.md, "D3 controls"):
  (a) dispatch: in-server GateLinear flags select tier 6 on every layer, the offline GateLinear on
      this GPU selects the same tier, gate weight / bias sha256 equal the in-server sha256 (both ranks);
  (b) exact reproduction: the offline SHIPPED computation on a zero-padded M-row input placed at the
      in-server byte alignment equals the captured SHIPPED logits BITWISE on every captured row;
  (c) exact routing: the engine's router on the captured SHIPPED logits reproduces the consumed
      topk ids (as sets) and weights (bitwise, dense by expert) on every captured row;
  If (a), (b) or (c) fails the script STOPS (exit 3) after writing d3_controls_failed.json with a
  diagnostic of batching / shape / alignment / dispatch / repeatability.  No tolerance is relaxed.
  (d) CORRECTED vs REFERENCE within tau = K * 2^-23 * sum_k |x_k w_k| + 2^-24 * |ref| per logit
      (K = 4096) and selection agreement except inside the margin band 0.5 * max_e tau_e;
      SHIPPED vs REFERENCE reported for contrast.  (d) failing does not stop; it withholds
      "verified correction".
Usage: d3_router_compare.py RUN_DIR CKPT_DIR OUT_DIR [--max-records N] [--device cuda|cpu]
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import os
import random
import struct
import sys
import time

import torch

E, K, H = 288, 8, 4096
SCALE = 2.5
LABEL_SHIPPED = "shipped: bf16-result GEMM cast to fp32"
LABEL_CORRECTED = "corrected: fp32-output GEMM (upstream #54048 form)"
LABEL_REFERENCE = "reference: fp64"
U_ACC = 2.0 ** -23          # one fp32 ulp per accumulation step (covers truncating accumulators)
U_OUT = 2.0 ** -24          # final fp32 rounding
MMA_K = 16                  # bf16 tensor-core MMA k-chunk
TAU_NOTE = ("Two fixed tolerances per logit, S = sum_k |x_k*w_k| (fp64). Hard bound, any summation "
            "algorithm: tau_worst = K*2^-23*S + 2^-24*|ref| with K=4096 (bf16 x bf16 products are exact "
            "in fp32, 8+8 significand bits <= 24; K-1 fp32 additions, each <= 1 ulp even if truncating, "
            "first-order Higham gamma_{K-1}). Verification tolerance: tau_d = 2^-14*S + 2^-24*|ref|: a "
            "tensor-core GEMM with fp32 accumulators updates the accumulator K/16 = 256 times "
            "(<= 2^-15*S, one ulp each incl. truncation), doubled for the error inside each 16-product "
            "chunk and any split-K reduction. Pass = every corrected logit within tau_d and every "
            "corrected-vs-reference top-8 disagreement inside the margin band 0.5*max_e tau_d,e "
            "(sigmoid is 1/4-Lipschitz, the bias is common); within tau_worst but not tau_d = "
            "inconclusive; beyond tau_worst = wrong computation.")
NORM_NOTE = (
    "Normalisation and x2.5 scaling, like-for-like. Served path (= shipped routing, and the corrected "
    "routing, both through the same engine call at serving M): GroupedTopKRouter._compute_routing "
    "(fused_moe/router/grouped_topk_router.py:337-349) -> grouped_topk (:80-108; "
    "VLLM_USE_FUSED_MOE_GROUPED_TOPK=1, envs.py:1625) -> fused_grouped_topk (:28-71) -> "
    "ops.grouped_topk(logits, n_group=1, topk_group=1, topk=8, renormalize=True, "
    "routed_scaling_factor=2.5, e_score_correction_bias, scoring_func=1) (:43-52; _custom_ops.py:2470-2500): "
    "sigmoid, bias used for selection only, renormalisation and the x2.5 scaling all inside the fused "
    "kernel; nothing is applied afterwards on this path (the post-hoc '*= routed_scaling_factor' at "
    ":314-315 belongs to the invalid-grouping fallback, not taken: 288 experts, 1 group), and no "
    "output-side scaling (apply_routed_scale_to_output=False, models/glm5next/nvidia/model.py:170,252). "
    "Measured: every captured row's weights sum to 2.5 (within 1e-6). Reference (fp64, selection and E1 "
    "only): sigmoid(logits)+bias selects the top-8, weights = unbiased sigmoid of the selected, "
    "renormalised to 1, x2.5 (route_ref64). Order inside the fused kernel (renormalise then scale, or "
    "the reverse) is not visible (kernel source not in the image); both give weights summing to 2.5.")
E1_NOTE = ("Finding E1 (engine, separate from D3): the served fused grouped_topk computes routing weights "
           "differently by batch row count (switch between 1,025 and 1,139 rows, offline attempt 3); ids "
           "unaffected on the captured rows. Reported as served-weight accuracy against the fp64 "
           "re-statement, split at M <= 1024 / M > 1024; no workaround beyond invoking at the serving M.")
FULL_DECODE_NOTE = ("FULL-graph decode steps are covered by source inspection only (not measured): the "
                    "capture runs on eager forwards; measuring inside FULL graphs would add nodes to the "
                    "captured graphs, i.e. modify execution")


# ------------------------------------------------------------------ pure helpers (CPU-tested)
def sets_differ(ids_a: torch.Tensor, ids_b: torch.Tensor) -> torch.Tensor:
    """Per row: number of experts in a not in b (0 = same set)."""
    a = torch.sort(ids_a.long(), dim=1).values
    b = torch.sort(ids_b.long(), dim=1).values
    eq = (a[:, :, None] == b[:, None, :]).any(dim=2)
    return (~eq).sum(dim=1)


def dense(ids: torch.Tensor, w: torch.Tensor, n: int = E) -> torch.Tensor:
    out = torch.zeros(ids.shape[0], n, dtype=torch.float64)
    out.scatter_(1, ids.long(), w.double())
    return out


def route_ref64(logits64: torch.Tensor, bias: torch.Tensor, k: int = K, scale: float = SCALE):
    """fp64 re-statement (reference only): sigmoid + bias selection, unbiased weights renormalised x scale."""
    s = torch.sigmoid(logits64.double())
    sel = s + bias.double()[None, :]
    ids = torch.topk(sel, k, dim=1).indices
    w = s.gather(1, ids)
    w = w / w.sum(dim=1, keepdim=True) * scale
    return ids, w


def gap_k(logits: torch.Tensor, bias: torch.Tensor, k: int = K) -> torch.Tensor:
    """Selection margin: biased score of the k-th minus the (k+1)-th expert."""
    sel = torch.sigmoid(logits.double()) + bias.double()[None, :]
    top = torch.topk(sel, k + 1, dim=1).values
    return top[:, k - 1] - top[:, k]


def tau(x: torch.Tensor, W: torch.Tensor, ref64: torch.Tensor, kdim: int = H, which: str = "d") -> torch.Tensor:
    """Per-logit tolerance (see TAU_NOTE): which='worst' (any algorithm) or 'd' (MMA-chunk model)."""
    sxw = x.double().abs() @ W.double().abs().T
    steps = kdim if which == "worst" else 2 * (kdim // MMA_K)
    return steps * U_ACC * sxw + U_OUT * ref64.abs()


def bf16_representable(t: torch.Tensor) -> bool:
    return bool(torch.equal(t.float(), t.to(torch.bfloat16).float()))


def bitwise_equal_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.contiguous().view(torch.int32) == b.contiguous().view(torch.int32)).all(dim=1)


def ulp_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    ia = a.contiguous().view(torch.int32).long()
    ib = b.contiguous().view(torch.int32).long()
    return int((ia - ib).abs().max()) if a.numel() else 0


def wilson(x: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cluster_boot(num_by_cluster, den_by_cluster, reps=2000, seed=20261012):
    """95% percentile interval of sum(num)/sum(den), resampling whole clusters with replacement."""
    rng = random.Random(seed)
    keys = [k for k in den_by_cluster if den_by_cluster[k] > 0]
    if not keys:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(reps):
        f = n = 0.0
        for _k in range(len(keys)):
            kk = keys[rng.randrange(len(keys))]
            f += num_by_cluster[kk]
            n += den_by_cluster[kk]
        vals.append(f / n if n else 0.0)
    vals.sort()
    return (vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1])


def row_attribution(meta: dict, r: int):
    """Per captured row: (row_kind, batch_kind, history key, request key)."""
    rows = meta.get("rows") or {}
    if not rows or rows.get("mismatch") or rows.get("error") or "row_req" not in rows:
        return [("unknown", "unknown", "unknown", "unknown")] * r
    reqs = {int(k): v for k, v in (rows.get("reqs") or {}).items()}
    out = []
    for i in rows["row_req"][:r]:
        if i is None or i < 0:
            out.append(("padding", rows.get("batch_kind") or "unknown", "padding", "padding"))
            continue
        q = reqs.get(int(i), {})
        kind = "prefill" if q.get("is_prefilling") else ("decode" if q.get("is_prefilling") is False else "unknown")
        out.append((kind, rows.get("batch_kind") or "unknown", f"prompt_len={q.get('prefill_len')}",
                    str(q.get("req_id"))))
    return out


def aligned_padded(x_rows: torch.Tensor, m: int, a: int, ptr_mod256: int, device, fill="zero", seed=0):
    """M-row bf16 input with x_rows at [a, a+r) whose base address has the in-server residue mod 256."""
    off = (int(ptr_mod256) // 2) if ptr_mod256 is not None else 0
    buf = torch.zeros(m * H + 256, dtype=torch.bfloat16, device=device)
    xs = buf[off:off + m * H].view(m, H)
    if fill == "random":
        g = torch.Generator(device="cpu").manual_seed(seed)
        xs.copy_((torch.randn(m, H, generator=g) * 3).to(torch.bfloat16).to(device))
    xs[a:a + x_rows.shape[0]] = x_rows.to(device)
    return xs


# ------------------------------------------------------------------ checkpoint access
class Ckpt:
    def __init__(self, d):
        self.d = d
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]

    def get(self, key) -> torch.Tensor:
        f = os.path.join(self.d, self.map[key])
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
            meta = hdr[key]
            s, e = meta["data_offsets"]
            fh.seek(8 + n + s)
            buf = bytearray(fh.read(e - s))
        dt = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}[meta["dtype"]]
        return torch.frombuffer(buf, dtype=dt).reshape(meta["shape"]).clone()


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


# ------------------------------------------------------------------ engine objects
FLAG_KEYS = ("allow_ll_bf16_gemm", "allow_specialized_router_gemm", "allow_dsv3_router_gemm",
             "allow_fp32_router_gemm", "allow_bf16x3_router_gemm", "allow_cublas_router_gemm")


def stub_tp():
    import types
    import vllm.distributed.parallel_state as ps
    if getattr(ps, "_TP", None) is None:
        ps._TP = types.SimpleNamespace(rank_in_group=0, world_size=1)


def engine_gate(weight: torch.Tensor, device):
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)   # the engine builds the model under the model dtype
    try:
        g = GateLinear(H, E, out_dtype=torch.float32, prefix="d3.offline.gate")
    finally:
        torch.set_default_dtype(prev)
    g = g.to(device)
    with torch.no_grad():
        g.weight.copy_(weight.to(device))
    flags = {k: getattr(g, k, None) for k in FLAG_KEYS}
    flags["weight_dtype"] = str(g.weight.dtype)
    return g, flags


def engine_router(bias: torch.Tensor, rinfo: dict, device):
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import GroupedTopKRouter
    p = torch.nn.Parameter(bias.to(device=device, dtype=torch.float32), requires_grad=False)
    return GroupedTopKRouter(top_k=rinfo.get("top_k", K), global_num_experts=rinfo.get("global_num_experts", E),
                             num_expert_group=rinfo.get("num_expert_group", 1),
                             topk_group=rinfo.get("topk_group", 1),
                             renormalize=rinfo.get("renormalize", True),
                             scoring_func=rinfo.get("scoring_func", "sigmoid"),
                             routed_scaling_factor=rinfo.get("routed_scaling_factor", SCALE),
                             e_score_correction_bias=p)


CORRECTED_IMPL = {"impl": None}


def corrected(xs: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """CORRECTED: torch.mm(x, W.T, out_dtype=fp32), gate_linear.py:223 tier-5 expression (#54048).
    If this torch/cuBLAS build rejects out_dtype on the device the run records the fallback
    (fp32 GEMM of the same bf16 values, TF32 off) and the D3 'verified correction' label is withheld."""
    if xs.is_cuda:
        try:
            out = torch.mm(xs, W.T, out_dtype=torch.float32)
            CORRECTED_IMPL["impl"] = CORRECTED_IMPL["impl"] or "torch.mm(out_dtype=fp32)"
            return out
        except Exception as exc:  # noqa: BLE001
            CORRECTED_IMPL["impl"] = f"FALLBACK fp32 mm ({exc!r})"[:200]
            torch.backends.cuda.matmul.allow_tf32 = False
            return xs.float() @ W.float().T
    CORRECTED_IMPL["impl"] = "cpu fp32 mm (test only)"
    return xs.float() @ W.float().T


def load_log(run_dir):
    recs = []
    for f in glob.glob(os.path.join(run_dir, "**", "v13_p*.jsonl"), recursive=True):
        for line in open(f):
            try:
                recs.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return recs


def logits_key(rec):
    if "logits_shipped" in rec:
        return "logits_shipped"
    raise KeyError("record has no 'logits_shipped' tensor (unlabelled logits are not accepted)")


# ------------------------------------------------------------------ control (c) diagnostics
CREC = []          # per-record control (c) classification (all records)
E1 = collections.defaultdict(list)   # served-weight relative error vs fp64, by batch-size regime
CDIAG_FILES = []   # failing records selected for the detailed routing diagnostic


def classify_routing(ci, cw, ei, ew):
    """Captured (consumed) routing vs engine routing of the same logits: (i) id sets differ,
    (ii) same sets but weights not bitwise (ulp stats), positional order, weight-sum stats."""
    nd = sets_differ(ci, ei)
    same = nd == 0
    Dc = torch.zeros(ci.shape[0], E, dtype=torch.float32).scatter_(1, ci.long(), cw.float())
    De = torch.zeros(ci.shape[0], E, dtype=torch.float32).scatter_(1, ei.long(), ew.float())
    beq = (Dc.view(torch.int32) == De.view(torch.int32)).all(dim=1)
    wrows = same & ~beq
    out = dict(rows=int(ci.shape[0]), rows_set_differ=int((nd > 0).sum()),
               members_differ=int(nd.sum()), rows_same_set_weights_not_bitwise=int(wrows.sum()),
               rows_exact=int((same & beq).sum()),
               rows_same_order=int((ci.long() == ei.long()).all(dim=1).sum()),
               captured_wsum_minus_2p5_max=float((cw.double().sum(1) - SCALE).abs().max()),
               engine_wsum_minus_2p5_max=float((ew.double().sum(1) - SCALE).abs().max()))
    if bool(wrows.any()):
        u = (Dc[wrows].view(torch.int32).long() - De[wrows].view(torch.int32).long()).abs()
        u = u[u > 0]
        a = (Dc[wrows] - De[wrows]).abs()
        out.update(ulp_max=int(u.max()), ulp_median=float(u.double().median()),
                   ulp_hist={str(k): int((u == k).sum()) for k in (1, 2, 3, 4)} | {">4": int((u > 4).sum())},
                   w_max_abs=float(a.max()), w_max_rel=float((a / De[wrows].abs().clamp_min(1e-30)).max()))
    return out


def aligned_logits(win: torch.Tensor, m: int, a: int, ptr_mod256, device, fill="zero", seed=0):
    """M-row fp32 logits with the captured rows at [a, a+r), base at the in-server residue mod 256."""
    off = (int(ptr_mod256) // 4) if ptr_mod256 is not None else 0
    buf = torch.zeros(m * E + 64, dtype=torch.float32, device=device)
    lg = buf[off:off + m * E].view(m, E)
    if fill == "random":
        g = torch.Generator(device="cpu").manual_seed(seed)
        lg.copy_((torch.randn(m, E, generator=g) * 2.0).to(device))
    elif fill == "tile":
        reps = (m + win.shape[0] - 1) // win.shape[0]
        lg.copy_(win.to(device).repeat(reps, 1)[:m])
    lg[a:a + win.shape[0]] = win.to(device)
    return lg


def route_serving_shape(router, logits_rows, m, a, ptr_mod256, ids_dtype, device):
    """The engine router invoked as in serving: one call on the M-row batch (model.py:276-279 ->
    moe_runner.py:596), the rows of interest at [a, a+r), returning those rows."""
    r = logits_rows.shape[0]
    lg = aligned_logits(logits_rows, m, a, ptr_mod256, device)
    hs = torch.zeros(m, H, dtype=torch.bfloat16, device=device)
    w, i = router.select_experts(hidden_states=hs, router_logits=lg, topk_indices_dtype=ids_dtype)
    return w[a:a + r], i[a:a + r]


def diagnose_routing(f, router, device):
    """Why does the engine router on the captured shipped logits not reproduce the consumed routing?
    Repeatability, batch shape (window vs serving M, other sizes), padding content, alignment,
    row misalignment (+-1), and transformations (order, scaling)."""
    rec = torch.load(f)
    meta = rec["meta"]
    m, a, r = meta["m"], meta["a"], meta["r"]
    cap = rec["logits_shipped"]
    ci, cw = rec["topk_ids"], rec["topk_weights"]
    dt = ci.dtype
    x = rec["x"].to(device)

    def route(lg, rows=None):
        hs = torch.zeros(lg.shape[0], H, dtype=torch.bfloat16, device=device) if lg.shape[0] != r else x
        w, i = router.select_experts(hidden_states=hs, router_logits=lg, topk_indices_dtype=dt)
        if rows is not None:
            w, i = w[rows], i[rows]
        return i.cpu(), w.cpu()

    out = dict(file=os.path.basename(f), m=m, a=a, r=r, logits_ptr_mod256=meta.get("logits_ptr_mod256"),
               batch_kind=(meta.get("rows") or {}).get("batch_kind"))
    with torch.no_grad():
        i1, w1 = route(cap.to(device).contiguous())
        i2, w2 = route(cap.to(device).contiguous())
        out["repeatable_window"] = bool(torch.equal(i1, i2) and torch.equal(w1, w2))
        out["window_vs_captured"] = classify_routing(ci, cw, i1, w1)
        sl = slice(a, a + r)
        pm = meta.get("logits_ptr_mod256")
        iz, wz = route(aligned_logits(cap, m, a, pm, device), sl)
        out["serving_M_zero_pad_vs_captured"] = classify_routing(ci, cw, iz, wz)
        iz2, wz2 = route(aligned_logits(cap, m, a, pm, device), sl)
        out["repeatable_serving_M"] = bool(torch.equal(iz, iz2) and torch.equal(wz, wz2))
        ir, wr = route(aligned_logits(cap, m, a, pm, device, fill="random", seed=1), sl)
        out["serving_M_random_pad_equals_zero_pad"] = bool(torch.equal(iz, ir) and torch.equal(wz, wr))
        it, wt = route(aligned_logits(cap, m, a, pm, device, fill="tile"), sl)
        out["serving_M_tiled_pad_equals_zero_pad"] = bool(torch.equal(iz, it) and torch.equal(wz, wt))
        i0, w0 = route(aligned_logits(cap, m, a, 0, device), sl)
        out["serving_M_alignment0_equals_server_alignment"] = bool(torch.equal(iz, i0) and torch.equal(wz, w0))
        sizes = {}
        for mm in sorted({r, 2 * r, 128, 256, 512, 1024, 2048, m}):
            if mm < r:
                continue
            ii, ww = route(aligned_logits(cap, mm, 0, pm, device), slice(0, r))
            c = classify_routing(ci, cw, ii, ww)
            sizes[str(mm)] = dict(exact_rows=c["rows_exact"], set_differ=c["rows_set_differ"],
                                  weights_not_bitwise=c["rows_same_set_weights_not_bitwise"])
        out["by_batch_rows_window_at_0"] = sizes
        # (iii) row misalignment: does captured row i match engine row i+-1 better?
        for sh in (-1, 1):
            if r > 1:
                a_ = ci[max(0, sh):r + min(0, sh)]
                b_ = i1[max(0, -sh):r + min(0, -sh)]
                out[f"shift{sh:+d}_set_match_rows"] = int((sets_differ(a_, b_) == 0).sum())
        # (iv) transformations
        out["captured_order_equals_engine_order_rows"] = int((ci.long() == i1.long()).all(dim=1).sum())
        same = sets_differ(ci, i1) == 0
        if bool(same.any()):
            Dc = dense(ci, cw)[same]
            De = dense(i1, w1)[same]
            ratio = (Dc.sum(1) / De.sum(1))
            out["captured_over_engine_weight_sum_ratio_minmax"] = [float(ratio.min()), float(ratio.max())]
        sel = torch.sigmoid(cap.double()) + 0.0
        top = torch.topk(sel, K + 1, dim=1).values
        nd = sets_differ(ci, i1)
        if bool((nd > 0).any()):
            out["unbiased_sigmoid_gap8_9_at_set_differ_rows"] = (top[:, K - 1] - top[:, K])[nd > 0].tolist()[:20]
    return out


# ------------------------------------------------------------------ phase 1: controls (a)-(c)
def diagnose(rec, gate, device, cap):
    """Why does the offline SHIPPED computation differ from the captured one?  Never relaxes (b)."""
    m, a, r = rec["meta"]["m"], rec["meta"]["a"], rec["meta"]["r"]
    pm = rec["meta"].get("x_ptr_mod256")
    x = rec["x"]
    out = dict(m=m, a=a, r=r, mode=rec["meta"].get("mode"), x_ptr_mod256=pm,
               x_stride=rec["meta"].get("x_stride"), batch_kind=(rec["meta"].get("rows") or {}).get("batch_kind"))
    with torch.no_grad():
        base = gate(aligned_padded(x, m, a, pm, device))[0][a:a + r]
        out["rows_not_exact"] = int((~bitwise_equal_rows(base, cap)).sum())
        out["max_abs_diff"] = float((base - cap).abs().max())
        out["max_ulp_diff"] = ulp_diff(base, cap)
        reps = [gate(aligned_padded(x, m, a, pm, device))[0][a:a + r] for _ in range(3)]
        out["repeatable_offline"] = all(torch.equal(base, t) for t in reps)
        rnd = gate(aligned_padded(x, m, a, pm, device, fill="random", seed=1))[0][a:a + r]
        out["row_independent_of_padding"] = bool(torch.equal(base, rnd))
        al0 = gate(aligned_padded(x, m, a, 0, device))[0][a:a + r]
        out["alignment_sensitive"] = not torch.equal(base, al0)
        out["alignment0_matches_captured"] = bool(bitwise_equal_rows(al0, cap).all())
        win = gate(x.to(device).contiguous())[0]
        out["window_only_shape_matches_captured"] = bool(bitwise_equal_rows(win, cap).all())
        out["bf16_valued_captured"] = bool(torch.equal(cap.float(), cap.to(torch.bfloat16).float()))
    return out


def controls(files0, ck, log, device, rinfo0):
    ginfo = {(d["rank"], d["layer"]): d for d in log if d.get("kind") == "gate_info"}
    gsha = {(d["rank"], d["layer"]): d for d in log if d.get("kind") == "gate_weight_sha"}
    by_layer = collections.defaultdict(list)
    for f in files0:
        by_layer[int(os.path.basename(f).split("_l")[1][:2])].append(f)
    fails = dict(a=[], b=[], c=[])
    diag = []
    cache = {}
    nrows = 0
    for layer in sorted(by_layer):
        pfx = f"model.language_model.layers.{layer}.mlp.gate"
        W = ck.get(pfx + ".weight")
        b = ck.get(pfx + ".e_score_correction_bias")
        gate, flags = engine_gate(W, device)
        router = engine_router(b, rinfo0, device)
        # (a) dispatch + hashes
        for rank in (0, 1):
            gi, gs = ginfo.get((rank, layer)), gsha.get((rank, layer))
            if gi is None and rank == 1:
                continue
            bad = []
            if gi is None or not str(gi.get("tier_from_flags", "")).startswith("6"):
                bad.append(f"server tier {None if gi is None else gi.get('tier_from_flags')}")
            if gi is not None and any(gi.get(k) for k in FLAG_KEYS):
                bad.append("server flags not all False")
            if gs is None or gs.get("weight_sha256") != sha(W) or gs.get("bias_sha256") != sha(b.float()):
                bad.append("weight/bias sha256 mismatch or missing")
            if bad:
                fails["a"].append(dict(layer=layer, rank=rank, why=bad))
        if any(flags[k] for k in FLAG_KEYS) or flags["weight_dtype"] != "torch.bfloat16":
            fails["a"].append(dict(layer=layer, rank="offline", why=[f"offline flags {flags}"]))
        cache[layer] = (W, b, gate, router)
        for f in by_layer[layer]:
            rec = torch.load(f)
            m, a, r = rec["meta"]["m"], rec["meta"]["a"], rec["meta"]["r"]
            cap = rec[logits_key(rec)].to(device)
            nrows += r
            with torch.no_grad():
                ship = gate(aligned_padded(rec["x"], m, a, rec["meta"].get("x_ptr_mod256"), device))[0][a:a + r]
            if not bf16_representable(cap):
                fails["a"].append(dict(layer=layer, rank=0, file=os.path.basename(f),
                                       why=["captured shipped logits not all bf16-representable (not tier 6)"]))
            okb = bool(bitwise_equal_rows(ship, cap).all())
            if not okb:
                fails["b"].append(os.path.basename(f))
                if len(diag) < 20:
                    diag.append(dict(file=os.path.basename(f), layer=layer, **diagnose(rec, gate, device, cap)))
            if "topk_ids" in rec:
                sg = torch.sigmoid(rec[logits_key(rec)].double()).gather(1, rec["topk_ids"].long())
                ref_w = sg / sg.sum(dim=1, keepdim=True) * SCALE
                E1[("M<=1024" if m <= 1024 else "M>1024")].append(
                    ((rec["topk_weights"].double() - ref_w).abs() / ref_w).max(dim=1).values)
                w_e, i_e = route_serving_shape(router, cap, m, a, rec["meta"].get("logits_ptr_mod256"),
                                               rec["topk_ids"].dtype, device)
                cls_c = classify_routing(rec["topk_ids"], rec["topk_weights"], i_e.cpu(), w_e.cpu())
                okc = cls_c["rows_set_differ"] == 0 and cls_c["rows_same_set_weights_not_bitwise"] == 0
            else:
                okc, cls_c = False, dict(missing_routing=True)
            CREC.append(dict(file=os.path.basename(f), layer=layer, m=m, a=a, r=r, ok=okc,
                             batch_kind=(rec["meta"].get("rows") or {}).get("batch_kind"),
                             logits_ptr_mod256=rec["meta"].get("logits_ptr_mod256"), **cls_c))
            if not okc:
                fails["c"].append(os.path.basename(f))
                if len(CDIAG_FILES) < 60:
                    CDIAG_FILES.append((f, layer))
    return by_layer, cache, fails, diag, nrows


# ------------------------------------------------------------------ phase 2: effect, (d), reporting
class Acc:
    def __init__(self):
        self.c = collections.Counter()
        self.l1 = []
        self.wmax_same = []
        self.gap = []
        self.gap_flip = []

    def add(self, nd, l1, wmax_same, gap):
        self.c["rows"] += int(nd.numel())
        self.c["flips"] += int((nd > 0).sum())
        self.l1.append(l1)
        if wmax_same.numel():
            self.wmax_same.append(wmax_same)
        self.gap.append(gap)
        if bool((nd > 0).any()):
            self.gap_flip.append(gap[nd > 0])


def q(ts, ps=(0.5, 0.9, 0.99, 1.0)):
    t = torch.cat(ts) if ts else torch.zeros(0)
    return {str(p): (float(torch.quantile(t.double(), p)) if t.numel() else None) for p in ps}


def effect(by_layer, cache, device, nrows_total, out_dir):
    tot = collections.Counter()
    acc_layer, acc_type, acc_lt = (collections.defaultdict(Acc) for _ in range(3))
    clus = {k: (collections.Counter(), collections.Counter(), collections.Counter())
            for k in ("history", "request")}      # flips, rows, l1 sum
    swapped = collections.Counter()
    err_c, err_s = [], []
    band_out = 0
    for layer in sorted(by_layer):
        W, b, gate, router = cache[layer]
        Wd = gate.weight.data
        for f in by_layer[layer]:
            rec = torch.load(f)
            meta = rec["meta"]
            m, a, r = meta["m"], meta["a"], meta["r"]
            x = rec["x"]
            ship = rec[logits_key(rec)].to(device)                    # SHIPPED (captured; == offline by (b))
            with torch.no_grad():
                xs = aligned_padded(x, m, a, meta.get("x_ptr_mod256"), device)
                corr = corrected(xs, Wd)[a:a + r].contiguous()          # CORRECTED
            ref = x.double() @ W.double().T                             # REFERENCE
            t = tau(x, W, ref)
            tw = tau(x, W, ref, which="worst")
            ec = (corr.double().cpu() - ref).abs()
            es = (ship.double().cpu() - ref).abs()
            sxw = x.double().abs() @ W.double().abs().T
            err_c.append((ec / sxw.clamp_min(1e-300)).max(dim=1).values)
            err_s.append((es / sxw.clamp_min(1e-300)).max(dim=1).values)
            tot["d_corr_violations"] += int((ec > t).sum())
            tot["d_corr_worst_violations"] += int((ec > tw).sum())
            tot["d_ship_violations"] += int((es > t).sum())
            tot["logits"] += int(ec.numel())
            ids_s, w_s = rec["topk_ids"].cpu(), rec["topk_weights"].cpu()   # consumed (== engine on SHIPPED by (c))
            w_c, i_c = route_serving_shape(router, corr, m, a, meta.get("logits_ptr_mod256"),
                                           rec["topk_ids"].dtype, device)
            ids_c, w_c = i_c.cpu(), w_c.cpu()
            ids64, _ = route_ref64(ref, b)
            nd = sets_differ(ids_s, ids_c)
            nd_s64, nd_c64 = sets_differ(ids_s, ids64), sets_differ(ids_c, ids64)
            g = gap_k(corr.cpu(), b)
            g64 = gap_k(ref, b)
            band = 0.5 * t.max(dim=1).values
            band_out += int(((nd_c64 > 0) & (g64 > band)).sum())
            tot["c64_flips"] += int((nd_c64 > 0).sum())
            tot["s64_flips"] += int((nd_s64 > 0).sum())
            Ds, Dc = dense(ids_s, w_s), dense(ids_c, w_c)
            l1 = (Ds - Dc).abs().sum(dim=1)
            same = nd == 0
            wmax = (Ds[same] - Dc[same]).abs().max(dim=1).values if bool(same.any()) else torch.zeros(0)
            for v in nd[nd > 0].tolist():
                swapped[int(v)] += 1
            attr = row_attribution(meta, r)
            acc_layer[layer].add(nd, l1, wmax, g)
            groups_t, groups_lt, groups_c = (collections.defaultdict(list) for _ in range(3))
            for i, (rk, bk, hist, req) in enumerate(attr):
                groups_t[(bk, rk)].append(i)
                groups_lt[(layer, bk, rk)].append(i)
                groups_c[("history", hist)].append(i)
                groups_c[("request", req)].append(i)
            for key, idx in groups_t.items():
                ix = torch.tensor(idx)
                sm = same[ix]
                wm = (Ds[ix][sm] - Dc[ix][sm]).abs().max(dim=1).values if bool(sm.any()) else torch.zeros(0)
                acc_type[key].add(nd[ix], l1[ix], wm, g[ix])
            for key, idx in groups_lt.items():
                ix = torch.tensor(idx)
                acc_lt[key].add(nd[ix], l1[ix], torch.zeros(0), g[ix])
            for (ck_, val), idx in groups_c.items():
                ix = torch.tensor(idx)
                clus[ck_][0][val] += int((nd[ix] > 0).sum())
                clus[ck_][1][val] += len(idx)
                clus[ck_][2][val] += float(l1[ix].sum())
            tot["l1_nonzero_rows"] += int((l1 > 0).sum())
            tot["rows"] += r
            tot["flips"] += int((nd > 0).sum())
    flips, n = tot["flips"], tot["rows"]

    def table(acc, keyname):
        rows = []
        for k, v in sorted(acc.items(), key=lambda kv: str(kv[0])):
            rows.append(dict(**({keyname: k} if not isinstance(k, tuple) else dict(zip(keyname.split("|"), k))),
                             rows=v.c["rows"], flips=v.c["flips"],
                             flip_rate=v.c["flips"] / max(1, v.c["rows"]),
                             wilson95_rows_iid=wilson(v.c["flips"], v.c["rows"]),
                             l1_median=q(v.l1)["0.5"], l1_p99=q(v.l1)["0.99"],
                             w_maxdiff_same_set_p99=q(v.wmax_same)["0.99"],
                             gap8_9_corrected_median=q(v.gap)["0.5"],
                             gap8_9_on_flips_median=q(v.gap_flip)["0.5"]))
        return rows

    by_layer_t = table(acc_layer, "layer")
    by_type_t = table(acc_type, "batch_kind|row_kind")
    by_lt_t = table(acc_lt, "layer|batch_kind|row_kind")
    for name, rows in (("d3_by_layer.csv", by_layer_t), ("d3_by_type.csv", by_type_t),
                       ("d3_by_layer_type.csv", by_lt_t)):
        with open(os.path.join(out_dir, name), "w") as fh:
            if rows:
                fh.write(",".join(rows[0].keys()) + "\n")
                for r_ in rows:
                    fh.write(",".join(json.dumps(v) if isinstance(v, (tuple, list)) else str(v)
                                      for v in r_.values()) + "\n")
    boot = {}
    for ck_, (fl, rw, l1s) in clus.items():
        boot[ck_] = dict(clusters=len([k for k in rw if rw[k]]),
                         flip_rate_ci95=cluster_boot(fl, rw), mean_l1_ci95=cluster_boot(l1s, rw),
                         rows_per_cluster=dict(sorted(rw.items(), key=lambda kv: -kv[1])[:20]))
    return dict(rows=n, flips=flips, flip_rate=flips / max(1, n), l1_nonzero_rows=tot["l1_nonzero_rows"],
                swapped_experts_hist=dict(sorted(swapped.items())),
                d_corrected_vs_reference=dict(violations=tot["d_corr_violations"],
                                              violations_of_tau_worst=tot["d_corr_worst_violations"],
                                              logits=tot["logits"],
                                              max_rel_err_over_sum_abs_xw=q(err_c)["1.0"],
                                              selection_disagreements=tot["c64_flips"],
                                              disagreements_outside_band=band_out),
                shipped_vs_reference_contrast=dict(violations_of_tau=tot["d_ship_violations"],
                                                   max_rel_err_over_sum_abs_xw=q(err_s)["1.0"],
                                                   selection_disagreements=tot["s64_flips"]),
                bootstrap_clustered=boot, by_layer=by_layer_t, by_type=by_type_t)


# ------------------------------------------------------------------ main
def analyse(run_dir, ckpt_dir, out_dir, max_records=None, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    stub_tp()
    log = load_log(run_dir)
    rinfo = [d for d in log if d.get("kind") == "router_info" and d.get("cls", "").endswith("TopKRouter")]
    rinfo0 = rinfo[0] if rinfo else {}
    files0 = sorted(glob.glob(os.path.join(run_dir, "**", "rec_r0", "*.pt"), recursive=True))
    if max_records:
        files0 = files0[:max_records]
    ck = Ckpt(ckpt_dir)
    t0 = time.time()
    by_layer, cache, fails, diag, nrows = controls(files0, ck, log, device, rinfo0)
    ctl = dict(a_dispatch_and_hashes=dict(ok=not fails["a"], failures=fails["a"][:20]),
               b_shipped_reproduced_bitwise=dict(ok=not fails["b"], records_failed=len(fails["b"]),
                                                 examples=fails["b"][:20]),
               c_routing_reproduced_exactly=dict(ok=not fails["c"], records_failed=len(fails["c"]),
                                                 examples=fails["c"][:20]),
               records=len(files0), rows=nrows, seconds=round(time.time() - t0, 1))
    labels = dict(shipped=LABEL_SHIPPED, corrected=LABEL_CORRECTED, reference=LABEL_REFERENCE)
    e1 = dict(note=E1_NOTE)
    for k, v in sorted(E1.items()):
        t = torch.cat(v)
        e1[k] = dict(rows=int(t.numel()), rel_err_median=float(t.median()),
                     rel_err_p99=float(torch.quantile(t, 0.99)), rel_err_max=float(t.max()))
    ctl["router_invocation"] = "serving-shaped: one call on the M-row batch, captured rows at [a, a+r), zero padding, server alignment (amendment 14)"
    if fails["c"] and CDIAG_FILES:
        for f, layer in CDIAG_FILES:
            try:
                diag.append(dict(control="c", **diagnose_routing(f, cache[layer][3], device)))
            except Exception as exc:  # noqa: BLE001
                diag.append(dict(control="c", file=os.path.basename(f), error=repr(exc)))
    if CREC:
        with open(os.path.join(out_dir, "d3_control_c_records.jsonl"), "w") as fh:
            for row in CREC:
                fh.write(json.dumps(row, default=str) + "\n")
        agg = collections.Counter()
        for row in CREC:
            for k in ("rows", "rows_set_differ", "members_differ", "rows_same_set_weights_not_bitwise", "rows_exact"):
                agg[k] += int(row.get(k, 0) or 0)
            agg["records_ok" if row["ok"] else "records_failed"] += 1
            agg[("full_ok" if row["ok"] else "full_failed") if row["m"] == row["r"] else
                ("window_ok" if row["ok"] else "window_failed")] += 1
        ctl["c_routing_reproduced_exactly"]["classification"] = dict(agg)
    if fails["a"] or fails["b"] or fails["c"]:
        json.dump(dict(stopped=True, labels=labels, controls=ctl, diagnostics=diag, finding_E1=e1, normalisation=NORM_NOTE,
                       note="exact-reproduction control failed: investigate batching, shape, alignment, "
                            "dispatch and repeatability (diagnostics); tolerances are not relaxed"),
                  open(os.path.join(out_dir, "d3_controls_failed.json"), "w"), indent=1, default=str)
        print(json.dumps(dict(stopped=True, controls=ctl, diagnostics=diag[:3]), indent=1, default=str))
        return dict(stopped=True, controls=ctl, diagnostics=diag)
    eff = effect(by_layer, cache, device, nrows, out_dir)
    d = eff["d_corrected_vs_reference"]
    impl_ok = CORRECTED_IMPL["impl"] in ("torch.mm(out_dtype=fp32)", "cpu fp32 mm (test only)")
    d_ok = d["violations"] == 0 and d["disagreements_outside_band"] == 0 and impl_ok
    d_state = ("pass" if d_ok else
               "wrong computation (beyond tau_worst)" if d["violations_of_tau_worst"] else
               "inconclusive (within tau_worst only, or corrected-path fallback)")
    cls = dict(confirmed_omission=True, exercised=True, controls_abc_pass=True,
               demonstrated_computational_effect=eff["flips"] > 0,
               routing_weights_changed=eff["l1_nonzero_rows"] > 0,
               verified_correction_offline=bool(d_ok),
               completion_effect="untested; expert flips and weight changes are not evidence of a completion effect")
    summary = dict(stopped=False, labels=labels, tolerance_d=TAU_NOTE, full_graph_decode=FULL_DECODE_NOTE,
                   normalisation=NORM_NOTE, finding_E1=e1,
                   corrected_impl=CORRECTED_IMPL["impl"], router_info=rinfo0, controls=ctl,
                   control_d=dict(ok=d_ok, state=d_state, **d), classification=cls,
                   **{k: v for k, v in eff.items() if k not in ("d_corrected_vs_reference",)})
    json.dump(summary, open(os.path.join(out_dir, "d3_summary.json"), "w"), indent=1, default=str)
    print(json.dumps(dict(classification=cls, rows=eff["rows"], flips=eff["flips"], control_d=summary["control_d"],
                          finding_E1=e1, controls=ctl,
                          bootstrap=eff["bootstrap_clustered"]), indent=1, default=str))
    return summary


def d4_inventory(run_dir, out_dir):
    """What the session froze for a later unclamped-vs-limit comparison (no B12x here)."""
    inv = collections.Counter()
    layers, branches, missing = set(), collections.Counter(), collections.Counter()
    for rank in (0, 1):
        for f in glob.glob(os.path.join(run_dir, "**", f"rec_r{rank}", "*.pt"), recursive=True):
            r = torch.load(f)
            inv[f"r{rank}_records"] += 1
            inv[f"r{rank}_rows"] += r["meta"]["r"]
            inv[f"r{rank}_with_out"] += int("out" in r)
            inv[f"r{rank}_same_x"] += int(bool(r["meta"].get("same_x")))
            for k in ("x", "topk_ids", "topk_weights", "out"):
                missing[k] += int(k not in r)
            branches[(r["meta"].get("d4") or {}).get("plan_branch", "unknown")] += 1
            layers.add(r["meta"]["layer"])
    log = load_log(run_dir)
    ei = [d for d in log if d.get("kind") == "experts_info"]
    pairs = 0
    for f in glob.glob(os.path.join(run_dir, "**", "rec_r0", "*.pt"), recursive=True):
        pairs += int(os.path.exists(f.replace("rec_r0", "rec_r1")))
    res = dict(inventory=dict(inv), rank_pairs=pairs, layers=sorted(layers), missing_tensors=dict(missing),
               plan_branches=dict(branches),
               swiglu_limit_on_layer=sorted({str(d.get("swiglu_limit_on_layer")) for d in ei}),
               serving_path=None if not ei else {k: ei[0].get(k) for k in (
                   "quant_method", "projection_mixed", "use_ep", "tp_size", "intermediate_per_partition",
                   "projection_tier_counts", "exl3_py_sha256", "source_anchors", "runtimes", "runtime_plans")},
               sufficient_for_stage_b=bool(inv["r0_records"] and pairs == inv["r0_records"]
                                           and not any(missing.values())))
    json.dump(res, open(os.path.join(out_dir, "d4_inventory.json"), "w"), indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("inventory", "rank_pairs", "missing_tensors", "plan_branches",
                                           "swiglu_limit_on_layer", "sufficient_for_stage_b")}, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("ckpt_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    res = analyse(a.run_dir, a.ckpt_dir, a.out_dir, a.max_records, a.device)
    d4_inventory(a.run_dir, a.out_dir)
    sys.exit(3 if res.get("stopped") else 0)
