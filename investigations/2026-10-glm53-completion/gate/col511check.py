# SPDX-License-Identifier: Apache-2.0
"""col511check: observe-only, graph-capture-safe checker for the column-511 correction
in the GLM5Next kpool sparse indexer (decode fused/tiled and prefill/extend branches).

Inert unless GLM53_COL511CHECK=device.  Installed as ``vllm/col511check.py`` in a
diagnostic image and called from the patched ``sparse_attn_indexer_kpool.py``.  It never
writes to an engine tensor: every store goes to checker-owned buffers; engine tensors are only
read.  Inside the indexer op it uses fixed-shape device ops only (no .item(), no boolean-mask
indexing, no host<->device copies), so the same code runs eagerly and inside captured FULL
CUDA graphs, where it is replayed with the op.  Readout: a background thread copies the
checker's device buffers on a guard-private side stream into pinned memory (on a DUMP trigger
file, periodically, at exit) - never inside a captured region and never on the engine stream.

Per call (branch = decode-fused | decode-tiled | prefill; keyed by graph mode, rows, mixed):
  pre  : the kernel's column-511 pool vs the other valid selected pools (kernel scores)
  post : (a) selected multiset unchanged; (b) column 511 = lowest kernel score (ties counted);
         (c) unfilled rows: an invalid slot, never a valid pool, ends in column 511;
         (e) NaN/inf-score rows counted separately and excluded from (b);
         (f) consumer check (expanded token buffer == corrected pools[:511] + tail),
             per-layer stale-digest check, id range check;
  ind  : 2 sampled valid rows per call re-scored in fp64 from the cache bytes (decode) or the
         gathered index-K workspace (prefill): column 511 lowest within the documented fp32 bound
         (|q|.|k| . |w| . scale . 162 . 2^-24), pre and post; kernel score within bound.
Host side: python-executed call counts (by mode, capture or not) and the runner's FULL /
piecewise replay counts, so device call counts can be reconciled with replayed steps.

Env: GLM53_COL511CHECK=device | "" ; GLM53_COL511CHECK_DIR (default /tmp/col511check);
     GLM53_COL511CHECK_POLL_SECS (default 30); GLM53_COL511CHECK_CAP (records, default 2048);
     GLM53_COL511CHECK_RING (per-step route ring, default 65536).
"""
from __future__ import annotations

import atexit
import json
import os
import threading
import time

import torch

MODE = os.environ.get("GLM53_COL511CHECK", "").strip().lower()
ENABLED = MODE == "device"
OUT_DIR = os.environ.get("GLM53_COL511CHECK_DIR", "/tmp/col511check")
POLL_SECS = float(os.environ.get("GLM53_COL511CHECK_POLL_SECS", "30"))
CAP = int(os.environ.get("GLM53_COL511CHECK_CAP", "2048"))
RING = int(os.environ.get("GLM53_COL511CHECK_RING", "65536"))
NSAMPLE = 2
EPS32 = 2.0 ** -24
TOPK = 512

MODES = ["none", "piecewise", "full", "capture_phase"]
BRANCHES = ["decode_fused", "decode_tiled", "prefill"]
NRB = 66  # rows bucket: 0..64 exact, 65 = more than 64
METRICS = [
    "calls", "rows_valid", "rows_full", "rows_unfilled", "rows_nan", "rows_dup",
    "rows_nvalid_mismatch", "rows_id_out_of_range", "rows_slot_score_inconsistent",
    "pre_lowest", "pre_tie", "pre_not_lowest", "pre_below_1_3", "pre_below_4_31",
    "pre_below_ge32", "pre_unfilled_valid_in_col511",
    "post_set_changed", "post_lowest", "post_tie", "post_not_lowest",
    "post_unfilled_valid_in_col511",
    "consumer_rows_checked", "consumer_mismatch", "stale_digest",
    "ind_rows", "ind_pre_ok_exact", "ind_pre_ok_tol", "ind_post_ok_exact", "ind_post_ok_tol",
    "ind_kernel_pools_checked", "ind_kernel_out_of_bound", "ind_nan_rows",
    "seq_pool_mismatch", "post_score_lookup_mismatch", "rows_full_nan_excluded",
]
M = {k: i for i, k in enumerate(METRICS)}
NM = len(METRICS)
REC_COLS = ["flags", "call_idx", "layer", "mode", "branch", "rows", "row", "c511_post",
            "below_post", "n_valid", "n_cand"]
F_SET, F_NOTLOW, F_UNFILLED, F_CONSUMER, F_NVALID, F_RANGE = 1, 2, 4, 8, 16, 32
RING_COLS = ["call_idx", "mode", "branch", "rows", "valid_rows", "mixed", "pre_not_lowest",
             "post_bad", "layer"]
NLAYER = 128


class _State:
    def __init__(self):
        self.dev = None
        self.device = None
        self.rank = None
        self.errors = 0
        self.fh = None
        self.logged = set()
        self.poller = None
        self.side = None
        self.pin = None
        self.route_cache = {}
        self.py_calls = {}       # (mode, branch, rows, mixed, capturing) -> python executions
        self.full_replays = {}   # num_tokens -> runner FULL replays
        self.pw_runs = {}        # num_tokens -> runner piecewise runs
        self.ring_layer = None   # designated layer for the per-step route ring
        self.last_key = None
        self.lock = threading.Lock()
        self.capture_depth = 0   # >0 while the runner captures / profiles graphs


S = _State()


def _capturing() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False


def _rank() -> int:
    if S.rank is None:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            S.rank = int(get_tensor_model_parallel_rank())
        except Exception:  # noqa: BLE001
            return -1
    return S.rank


def _log(rec: dict) -> None:
    try:
        with S.lock:
            if S.fh is None:
                os.makedirs(OUT_DIR, exist_ok=True)
                S.fh = open(os.path.join(OUT_DIR, f"c511_p{os.getpid()}.jsonl"), "a")
            rec.setdefault("t", round(time.time(), 4))
            rec.setdefault("rank", _rank())
            S.fh.write(json.dumps(rec) + "\n")
            S.fh.flush()
    except Exception:  # noqa: BLE001
        S.errors += 1


def _once(key, rec):
    if key not in S.logged:
        S.logged.add(key)
        _log(rec)


def _err(where, exc):
    S.errors += 1
    if S.errors <= 30:
        _log(dict(kind="checker_error", where=where, err=repr(exc)[:800]))


def _layer_idx(name) -> int:
    try:
        parts = str(name).split(".")
        return int(parts[parts.index("layers") + 1])
    except Exception:  # noqa: BLE001
        return NLAYER - 1


def alloc(device) -> None:
    """Persistent checker buffers; allocated outside any capture (BlockTables init)."""
    if S.dev is not None:
        return
    S.device = device
    S.dev = dict(
        counters=torch.zeros(len(MODES) * len(BRANCHES) * NRB * 2, NM, dtype=torch.int64,
                             device=device),
        records=torch.zeros(CAP + 1, len(REC_COLS), dtype=torch.int64, device=device),
        nrec=torch.zeros(1, dtype=torch.int64, device=device),
        layer_calls=torch.zeros(NLAYER, dtype=torch.int64, device=device),
        last_digest=torch.zeros(NLAYER, len(BRANCHES), dtype=torch.int64, device=device),
        ring=torch.full((RING, len(RING_COLS)), -1, dtype=torch.int64, device=device),
        fallback_calls=torch.zeros(1, dtype=torch.int64, device=device),
    )
    _log(dict(kind="alloc", device=str(device), cap=CAP, ring=RING, metrics=METRICS))


def _mode() -> int:
    """2 = FULL (inside a torch.cuda.graph capture: the ops recorded here are what FULL
    replays run); 1 = piecewise (eager break inside breakable piecewise replay); 0 = eager;
    3 = eager warm-up calls made by the runner while capturing graphs (dummy batches)."""
    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
        if _capturing() and BreakableCUDAGraphCapture.current() is None:
            return 2
    except Exception as exc:  # noqa: BLE001
        _err("mode_capture", exc)
    if S.capture_depth > 0:
        return 3
    try:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context, is_forward_context_available
        if not is_forward_context_available():
            return 0
        m = get_forward_context().cudagraph_runtime_mode
        if m == CUDAGraphMode.FULL:
            return 2
        if m == CUDAGraphMode.PIECEWISE:
            return 1
        return 0
    except Exception:  # noqa: BLE001
        return 0


def decode_route(q_fp8, block_table, rows, select_k) -> int:
    """0 = fused, 1 = tiled.  Host-side B12x plan (no launch), cached; rule fallback."""
    key = (int(rows), int(block_table.shape[1]), int(q_fp8.shape[1]), int(select_k))
    if key in S.route_cache:
        return S.route_cache[key]
    r = None
    if not _capturing():
        try:
            from b12x.attention.dsa_indexer.scratch import (
                INDEXER_SOURCE_LAYOUT_PAGED as _LP,
                B12XIndexerScratchCaps as _Caps,
                plan_indexer_scratch as _plan,
            )
            plan = _plan(_Caps(device=q_fp8.device, source_layout=_LP,
                               num_q_heads=int(q_fp8.shape[1]), max_q_rows=int(rows),
                               max_page_table_width=int(block_table.shape[1]),
                               topk=int(select_k), mode="decode", shared_page_table=False))
            name = str(getattr(plan.layout, "route", "?"))
            r = 0 if "fused" in name else 1
            _log(dict(kind="route", rows=int(rows), width=key[1], route=name))
        except Exception as exc:  # noqa: BLE001
            _err("route_plan", exc)
    if r is None:
        r = 0 if int(rows) <= 16 else 1
        _log(dict(kind="route_rule", rows=int(rows), route=r, capturing=_capturing()))
    S.route_cache[key] = r
    return r


# ------------------------------------------------------------------ device core
def _slot(mode, branch, rows, mixed) -> int:
    rb = int(rows) if int(rows) <= 64 else 65
    return ((mode * len(BRANCHES) + branch) * NRB + rb) * 2 + int(bool(mixed))


def _below_eq(ids, s, col):
    """For each row: number of OTHER valid selected pools scoring below / equal to column col."""
    K = ids.shape[1]
    vid = ids >= 0
    cs = s[:, col:col + 1]
    other = torch.arange(K, device=ids.device) != col  # no host->device scalar copy
    m = vid & other[None, :]
    below = (m & (s < cs)).sum(1)
    eq = (m & (s == cs)).sum(1)
    return below, eq


class Ctx:
    pass


def pre(layer_name, branch, mixed, ids, scores, n_cand, valid_row):
    """Snapshot the kernel output before the correction.  ids (R,512) int32 engine tensor,
    scores (R,512) fp32 aligned with ids, n_cand (R,) candidate count, valid_row (R,) bool."""
    if ENABLED and S.dev is None and not _capturing():
        try:
            alloc(ids.device)  # fallback if BlockTables was built before install()
        except Exception as exc:  # noqa: BLE001
            _err("alloc_lazy", exc)
    if not ENABLED or S.dev is None:
        if ENABLED and S.dev is None:
            _once("nodev", dict(kind="checker_warn", msg="no device buffers", capturing=_capturing()))
        return None
    try:
        c = Ctx()
        c.layer = _layer_idx(layer_name)
        c.mode = _mode()
        c.branch = int(branch)
        c.mixed = bool(mixed)
        c.R = int(ids.shape[0])
        c.capturing = _capturing()
        key = (c.mode, c.branch, c.R, c.mixed, c.capturing)
        S.py_calls[key] = S.py_calls.get(key, 0) + 1
        c.slot = _slot(c.mode, c.branch, c.R, c.mixed)
        c.ids = ids.to(torch.int64).clone()
        c.s = scores.to(torch.float32).clone()
        c.n = n_cand[: c.R].to(torch.int64).clone()
        c.valid = valid_row[: c.R].clone()
        dev = S.dev
        if not c.capturing:
            ensure_poller()
        dev["layer_calls"][c.layer] += 1
        c.call_idx = dev["layer_calls"][c.layer].clone()
        return c
    except Exception as exc:  # noqa: BLE001
        _err("pre", exc)
        return None


def post(c, ids_post, scores_direct=None, sample=None):
    """After the correction.  ids_post = the engine's corrected pool list (read only).
    scores_direct: scores gathered directly for ids_post (prefill), cross-checked against the
    lookup.  sample: callable(rows_idx (NSAMPLE,) int64, ids (NSAMPLE,512) int64)
    -> (s64, bound) fp64 (NSAMPLE,512) for the independent re-score."""
    if c is None:
        return
    try:
        _post(c, ids_post, scores_direct, sample)
    except Exception as exc:  # noqa: BLE001
        _err("post", exc)


def _post(c, ids_post, scores_direct, sample):
    dev = S.dev
    ids, s, n, valid = c.ids, c.s, c.n, c.valid
    R, K = ids.shape
    vid = ids >= 0
    nv = vid.sum(1)
    cnt = torch.zeros(NM, dtype=torch.int64, device=ids.device)

    def add(name, mask):
        cnt[M[name]] += (mask & valid).sum()

    finite = torch.isfinite(s)
    nanrow = (vid & ~finite).any(1)
    exp_nv = torch.clamp(n, max=K)
    full = nv == K
    unfilled = nv < K
    sp, perm = torch.sort(ids, dim=1)
    dup = ((sp[:, 1:] == sp[:, :-1]) & (sp[:, 1:] >= 0)).any(1)
    rng = (vid & (ids >= n[:, None])).any(1)
    slot_bad = ((vid & torch.isneginf(s)) | (~vid & ~torch.isneginf(s))).any(1)
    cnt[M["calls"]] += 1
    add("rows_valid", torch.ones_like(valid))
    add("rows_full", full)
    add("rows_unfilled", unfilled)
    add("rows_nan", nanrow)
    add("rows_dup", dup)
    add("rows_nvalid_mismatch", nv != exp_nv)
    add("rows_id_out_of_range", rng)
    if c.branch != 2:  # prefill scores are masked by construction
        add("rows_slot_score_inconsistent", slot_bad)
    clean_full = full & ~nanrow
    add("rows_full_nan_excluded", full & nanrow)

    # pre: kernel's column 511
    b0, e0 = _below_eq(ids, s, K - 1)
    add("pre_lowest", clean_full & (b0 == 0) & (e0 == 0))
    add("pre_tie", clean_full & (b0 == 0) & (e0 > 0))
    add("pre_not_lowest", clean_full & (b0 > 0))
    add("pre_below_1_3", clean_full & (b0 >= 1) & (b0 <= 3))
    add("pre_below_4_31", clean_full & (b0 >= 4) & (b0 <= 31))
    add("pre_below_ge32", clean_full & (b0 >= 32))
    add("pre_unfilled_valid_in_col511", unfilled & (ids[:, K - 1] >= 0))

    # post: corrected list, scores by lookup into the pre snapshot (multiset check first)
    ip = ids_post[:R].to(torch.int64)
    spost_sorted = torch.sort(ip, dim=1).values
    set_changed = ~(spost_sorted == sp).all(1)
    s_sorted = s.gather(1, perm)
    pos = torch.searchsorted(sp.contiguous(), ip.contiguous()).clamp(max=K - 1)
    s_post = s_sorted.gather(1, pos)
    if scores_direct is not None:
        sd = scores_direct[:R].to(torch.float32)
        mism = ((sd != s_post) & ~(torch.isnan(sd) & torch.isnan(s_post))).any(1)
        add("post_score_lookup_mismatch", mism)
    b1, e1 = _below_eq(ip, s_post, K - 1)
    add("post_set_changed", set_changed)
    add("post_lowest", clean_full & (b1 == 0) & (e1 == 0))
    add("post_tie", clean_full & (b1 == 0) & (e1 > 0))
    notlow = clean_full & (b1 > 0)
    add("post_not_lowest", notlow)
    unf_bad = unfilled & (ip[:, K - 1] >= 0)
    add("post_unfilled_valid_in_col511", unf_bad)

    # (f) stale digest of the kernel output consumed by the correction (per layer, branch)
    col = torch.arange(K, device=ids.device, dtype=torch.int64)[None, :]
    rowi = torch.arange(R, device=ids.device, dtype=torch.int64)[:, None]
    sbits = torch.where(vid & finite, s.view(torch.int32).to(torch.int64), torch.zeros_like(ids))
    term = torch.where(vid & valid[:, None], (ids + 1) * 1000003 + col * 7919 + rowi * 104729 + sbits,
                       torch.zeros_like(ids))
    dg = term.sum()
    any_valid = (valid & (nv > 0)).any()
    ld = dev["last_digest"][c.layer, c.branch]
    cnt[M["stale_digest"]] += (any_valid & (dg == ld)).to(torch.int64)
    dev["last_digest"][c.layer, c.branch] = dg

    # independent fp64 re-score of NSAMPLE valid rows
    if sample is not None:
        nvr = valid.to(torch.int64).sum().clamp(min=1)
        cum = torch.cumsum(valid.to(torch.int64), 0)
        j = torch.arange(NSAMPLE, device=ids.device, dtype=torch.int64)
        step = torch.div(nvr, NSAMPLE, rounding_mode="floor").clamp(min=1)
        k = torch.remainder(c.call_idx * 13 + j * step, nvr)
        rows_s = torch.searchsorted(cum, k + 1).clamp(max=R - 1)
        ids_s = ids.index_select(0, rows_s)              # pre order
        s64, bnd = sample(rows_s, ids_s)                 # (NS,K) fp64, -inf / 0 for invalid
        vrow = valid.index_select(0, rows_s) & clean_full.index_select(0, rows_s)
        nan64 = (torch.isnan(s64) & (ids_s >= 0)).any(1)
        ok = vrow & ~nan64
        cnt[M["ind_nan_rows"]] += (vrow & nan64).sum()
        cnt[M["ind_rows"]] += ok.sum()

        def lowest(xpos):
            sx = s64.gather(1, xpos[:, None]).squeeze(1)
            bx = bnd.gather(1, xpos[:, None]).squeeze(1)
            other = torch.ones_like(s64, dtype=torch.bool)
            other.scatter_(1, xpos[:, None], False)
            so = torch.where(other, s64, torch.full_like(s64, float("inf")))
            mo, am = so.min(1)
            bo = bnd.gather(1, am[:, None]).squeeze(1)
            return sx <= mo, sx <= mo + bx + bo

        pre_pos = torch.full((NSAMPLE,), K - 1, dtype=torch.int64, device=ids.device)
        cpost = ip.index_select(0, rows_s)[:, K - 1]
        post_pos = torch.argmax((ids_s == cpost[:, None]).to(torch.int8), dim=1)
        pe, pt = lowest(pre_pos)
        qe, qt = lowest(post_pos)
        cnt[M["ind_pre_ok_exact"]] += (ok & pe).sum()
        cnt[M["ind_pre_ok_tol"]] += (ok & pt).sum()
        cnt[M["ind_post_ok_exact"]] += (ok & qe).sum()
        cnt[M["ind_post_ok_tol"]] += (ok & qt).sum()
        sk = s.index_select(0, rows_s).double()
        vp = (ids_s >= 0) & ok[:, None]
        oob = vp & ((sk - s64).abs() > bnd + 1e-30)
        cnt[M["ind_kernel_pools_checked"]] += vp.sum()
        cnt[M["ind_kernel_out_of_bound"]] += oob.sum()

    dev["counters"][c.slot] += cnt

    # records of post failures
    flags = (set_changed.to(torch.int64) * F_SET + notlow.to(torch.int64) * F_NOTLOW
             + unf_bad.to(torch.int64) * F_UNFILLED + (nv != exp_nv).to(torch.int64) * F_NVALID
             + rng.to(torch.int64) * F_RANGE)
    c.flags = flags
    c.ip = ip
    c.b1 = b1
    c.nv = nv
    _record(c, valid & (flags != 0), flags)

    # per-step route ring (designated layer: first layer seen in this process)
    if S.ring_layer is None and not c.capturing:
        S.ring_layer = c.layer
        _log(dict(kind="ring_layer", layer=c.layer))
    if c.layer == S.ring_layer and c.branch != 2:
        bad_rows = (valid & (flags != 0)).sum()
        entry = torch.stack([
            c.call_idx.reshape(()), torch.full((), c.mode, dtype=torch.int64, device=ids.device),
            torch.full((), c.branch, dtype=torch.int64, device=ids.device),
            torch.full((), R, dtype=torch.int64, device=ids.device),
            valid.to(torch.int64).sum(), torch.full((), int(c.mixed), dtype=torch.int64,
                                                    device=ids.device),
            (valid & clean_full & (b0 > 0)).sum(), bad_rows,
            torch.full((), c.layer, dtype=torch.int64, device=ids.device)])
        ridx = torch.remainder(c.call_idx.reshape(1), RING)
        dev["ring"].index_copy_(0, ridx, entry[None, :])


def _record(c, mask, flags, extra_flag=0):
    dev = S.dev
    R = mask.shape[0]
    m = mask.to(torch.int64)
    base = dev["nrec"][0]
    idx = base + torch.cumsum(m, 0) - 1
    okm = mask & (idx < CAP)
    tgt = torch.where(okm, idx, torch.full_like(idx, CAP))
    ar = torch.arange(R, device=mask.device, dtype=torch.int64)
    ip = getattr(c, "ip", None)
    rows = torch.stack([
        flags | extra_flag, c.call_idx.expand(R), torch.full_like(ar, c.layer),
        torch.full_like(ar, c.mode), torch.full_like(ar, c.branch), torch.full_like(ar, R), ar,
        ip[:, -1] if ip is not None else torch.full_like(ar, -2),
        c.b1 if hasattr(c, "b1") else torch.full_like(ar, -2),
        c.nv if hasattr(c, "nv") else torch.full_like(ar, -2), c.n], dim=1)
    dev["records"].index_copy_(0, tgt, rows)
    dev["nrec"] += m.sum()


def consumer(c, buf_rows, seq_tok, kpool=4):
    """(f) the expanded token buffer the sparse attention reads must equal the corrected
    pools[:, :511] (ids in [0, seq//kpool)) expanded to tokens, plus the 0..kpool-1 tail."""
    if c is None:
        return
    try:
        R = c.R
        K = c.ids.shape[1]
        ip = c.ip[:, : K - 1]
        seq = seq_tok[:R].to(torch.int64)
        pool_len = torch.div(seq, kpool, rounding_mode="floor")
        keep = (ip >= 0) & (ip < pool_len[:, None])
        tail_start = pool_len * kpool
        tail_cnt = seq - tail_start
        exp_cnt = kpool * keep.sum(1) + tail_cnt
        exp_sum = (torch.where(keep, ip * (kpool * kpool) + (kpool * (kpool - 1)) // 2,
                               torch.zeros_like(ip)).sum(1)
                   + tail_cnt * tail_start + torch.div(tail_cnt * (tail_cnt - 1), 2,
                                                       rounding_mode="floor"))
        tok = buf_rows[:R].to(torch.int64)
        got_cnt = (tok >= 0).sum(1)
        got_sum = torch.where(tok >= 0, tok, torch.zeros_like(tok)).sum(1)
        bad = c.valid & ((got_cnt != exp_cnt) | (got_sum != exp_sum))
        cnt = torch.zeros(NM, dtype=torch.int64, device=tok.device)
        cnt[M["consumer_rows_checked"]] += c.valid.sum()
        cnt[M["consumer_mismatch"]] += bad.sum()
        if c.branch != 2:
            pm = c.valid & (pool_len != c.n)
            cnt[M["seq_pool_mismatch"]] += pm.sum()
        S.dev["counters"][c.slot] += cnt
        _record(c, bad, torch.zeros_like(seq), extra_flag=F_CONSUMER)
    except Exception as exc:  # noqa: BLE001
        _err("consumer", exc)


def fallback_call():
    """Counts decode calls that take the non-B12x (DeepGEMM + persistent_topk) path."""
    if ENABLED and S.dev is not None:
        try:
            S.dev["fallback_calls"] += 1
            _once("fallback", dict(kind="fallback_path_seen", capturing=_capturing()))
        except Exception as exc:  # noqa: BLE001
            _err("fallback", exc)


# ---------------------------------------------------------- fp64 samplers (device)
def decode_sampler(q_fp8, weights, kv_cache_raw, block_table, page=64, hd=128):
    """Independent re-score from the paged index-K cache bytes (page: 64x128 e4m3, 64 fp32)."""
    def f(rows_s, ids_s):
        nb = kv_cache_raw.shape[0]
        kvf = kv_cache_raw.reshape(nb, -1)
        idc = ids_s.clamp(min=0)
        bt = block_table.index_select(0, rows_s).to(torch.int64)
        pg = bt.gather(1, torch.div(idc, page, rounding_mode="floor"))
        off = torch.remainder(idc, page)
        ar = torch.arange(hd, device=idc.device, dtype=torch.int64)
        ar4 = torch.arange(4, device=idc.device, dtype=torch.int64)
        kb = kvf[pg[:, :, None], off[:, :, None] * hd + ar[None, None, :]]
        sb = kvf[pg[:, :, None], page * hd + off[:, :, None] * 4 + ar4[None, None, :]]
        K = kb.view(torch.float8_e4m3fn).double()
        sc = sb.contiguous().view(torch.float32).squeeze(-1).double()
        q = q_fp8.index_select(0, rows_s)
        q = (q.view(torch.float8_e4m3fn) if q.dtype == torch.uint8 else q).double()
        w = weights.index_select(0, rows_s).double()[:, : q.shape[1]]
        dots = torch.einsum("shd,skd->shk", q, K)
        s64 = (torch.relu(dots) * w[:, :, None]).sum(1) * sc
        bnd = (torch.einsum("shd,skd->shk", q.abs(), K.abs()) * w.abs()[:, :, None]).sum(1) \
            * sc.abs() * (162 * EPS32)
        vid = ids_s >= 0
        s64 = torch.where(vid, s64, torch.full_like(s64, float("-inf")))
        bnd = torch.where(vid & ~torch.isnan(bnd), bnd, torch.zeros_like(bnd))
        return s64, bnd
    return f


def prefill_sampler(q_fp8, weights, k_quant, k_scale, row_starts):
    """Independent re-score from the gathered index-K workspace of this chunk."""
    def f(rows_s, ids_s):
        T = k_quant.shape[0]
        colm = (ids_s.clamp(min=0) + row_starts.index_select(0, rows_s).to(torch.int64)[:, None]
                ).clamp(max=T - 1)
        K = k_quant[colm]
        K = (K.view(torch.float8_e4m3fn) if K.dtype == torch.uint8 else K).double()
        sc = k_scale.reshape(-1)[colm].double()
        q = q_fp8.index_select(0, rows_s)
        q = (q.view(torch.float8_e4m3fn) if q.dtype == torch.uint8 else q).double()
        w = weights.index_select(0, rows_s).double()[:, : q.shape[1]]
        dots = torch.einsum("shd,skd->shk", q, K)
        s64 = (torch.relu(dots) * w[:, :, None]).sum(1) * sc
        bnd = (torch.einsum("shd,skd->shk", q.abs(), K.abs()) * w.abs()[:, :, None]).sum(1) \
            * sc.abs() * (162 * EPS32)
        vid = ids_s >= 0
        s64 = torch.where(vid, s64, torch.full_like(s64, float("-inf")))
        bnd = torch.where(vid & ~torch.isnan(bnd), bnd, torch.zeros_like(bnd))
        return s64, bnd
    return f


# ------------------------------------------------------------------ readout
def _snapshot():
    dev = S.dev
    keys = ["counters", "records", "nrec", "layer_calls", "ring", "fallback_calls"]
    if dev["counters"].is_cuda:
        if S.side is None:
            S.side = torch.cuda.Stream(device=dev["counters"].device)
            S.pin = {k: torch.empty(dev[k].shape, dtype=dev[k].dtype, pin_memory=True) for k in keys}
        with torch.cuda.stream(S.side):
            for k in keys:
                S.pin[k].copy_(dev[k], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(S.side)
        ev.synchronize()
        return {k: S.pin[k].clone() for k in keys}
    return {k: dev[k].clone() for k in keys}


def poll(reason: str, only_if_changed: bool = False, with_ring: bool = False) -> None:
    if S.dev is None:
        return
    try:
        snap = _snapshot()
        cnt = snap["counters"]
        key = (int(cnt[:, 0].sum()), int(snap["nrec"][0]))
        if only_if_changed and key == S.last_key:
            return
        S.last_key = key
        rows = []
        nz = torch.nonzero(cnt[:, 0] > 0).flatten().tolist()
        for slot in nz:
            mixed = slot % 2
            rb = (slot // 2) % NRB
            br = (slot // 2 // NRB) % len(BRANCHES)
            md = slot // 2 // NRB // len(BRANCHES)
            rows.append(dict(mode=MODES[md], branch=BRANCHES[br], rows=rb, mixed=mixed,
                             **{m: int(cnt[slot, i]) for i, m in enumerate(METRICS)}))
        nrec = min(int(snap["nrec"][0]), CAP)
        d = dict(kind="poll", reason=reason, slots=rows, n_records=int(snap["nrec"][0]),
                 records=[dict(zip(REC_COLS, r)) for r in snap["records"][:nrec].tolist()],
                 layer_calls={i: int(v) for i, v in enumerate(snap["layer_calls"].tolist()) if v},
                 fallback_calls=int(snap["fallback_calls"][0]),
                 py_calls=[dict(mode=MODES[k[0]], branch=BRANCHES[k[1]], rows=k[2], mixed=k[3],
                                capturing=k[4], n=v) for k, v in sorted(list(S.py_calls.items()))],
                 full_replays={str(k): v for k, v in sorted(list(S.full_replays.items()))},
                 pw_runs={str(k): v for k, v in sorted(list(S.pw_runs.items()))},
                 ring_layer=S.ring_layer, checker_errors=S.errors)
        if with_ring:
            r = snap["ring"]
            used = r[:, 0] >= 0
            d["ring"] = r[used].tolist()
            d["ring_cols"] = RING_COLS
        _log(d)
    except Exception as exc:  # noqa: BLE001
        _err("poll", exc)


def _poller_loop():
    try:
        if S.dev is not None and S.dev["counters"].is_cuda:
            torch.cuda.set_device(S.dev["counters"].device)
    except Exception:  # noqa: BLE001
        pass
    trig = os.path.join(OUT_DIR, "DUMP")
    ring_trig = os.path.join(OUT_DIR, "DUMP_RING")
    last_trig = last_ring = None
    last = time.time()
    while True:
        time.sleep(2.0)
        try:
            t = os.path.getmtime(trig) if os.path.exists(trig) else None
            tr = os.path.getmtime(ring_trig) if os.path.exists(ring_trig) else None
        except OSError:
            t = tr = None
        if tr is not None and tr != last_ring:
            last_ring = tr
            poll("trigger_ring", with_ring=True)
            last = time.time()
        elif t is not None and t != last_trig:
            last_trig = t
            poll("trigger")
            last = time.time()
        elif time.time() - last >= POLL_SECS:
            poll("periodic", only_if_changed=True)
            last = time.time()


def ensure_poller():
    if S.poller is None and S.dev is not None and not _capturing():
        S.poller = threading.Thread(target=_poller_loop, name="col511check-poll",
                                    daemon=True)
        S.poller.start()


# ------------------------------------------------------------------ install
def install() -> None:
    """Allocate buffers at BlockTables init (before any capture); count runner FULL replays
    and piecewise runs host-side (observe-only wrappers that call the originals)."""
    from vllm.v1.worker.gpu import block_table as btmod
    from vllm.v1.worker.gpu import cudagraph_utils as cgu

    BT = btmod.BlockTables
    if getattr(BT, "_col511check", False):
        return
    BT._col511check = True
    orig_init = BT.__init__

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        try:
            alloc(self.device)
        except Exception as exc:  # noqa: BLE001
            _err("alloc", exc)

    BT.__init__ = __init__

    MG = cgu.ModelCudaGraphManager
    orig_full = MG.run_fullgraph
    orig_pw = cgu.CudaGraphManager.run_pw_graph

    def run_fullgraph(self, desc):
        try:
            S.full_replays[int(desc.num_tokens)] = S.full_replays.get(int(desc.num_tokens), 0) + 1
            ensure_poller()
        except Exception as exc:  # noqa: BLE001
            _err("count_full", exc)
        return orig_full(self, desc)

    def run_pw_graph(self, model, model_inputs):
        if isinstance(self, MG):
            try:
                n = -1
                for k in ("input_ids", "positions", "inputs_embeds"):
                    v = model_inputs.get(k)
                    if isinstance(v, torch.Tensor):
                        n = int(v.shape[0])
                        break
                S.pw_runs[n] = S.pw_runs.get(n, 0) + 1
                ensure_poller()
            except Exception as exc:  # noqa: BLE001
                _err("count_pw", exc)
        return orig_pw(self, model, model_inputs)

    MG.run_fullgraph = run_fullgraph
    cgu.CudaGraphManager.run_pw_graph = run_pw_graph

    def _phase_wrap(orig):
        def wrapped(self, *a, **kw):
            S.capture_depth += 1
            try:
                return orig(self, *a, **kw)
            finally:
                S.capture_depth -= 1
        return wrapped

    for cls in (cgu.CudaGraphManager, MG):
        for name in ("capture", "profile_memory"):
            if name in cls.__dict__:
                setattr(cls, name, _phase_wrap(cls.__dict__[name]))
    atexit.register(lambda: poll("exit", with_ring=True))
    _log(dict(kind="installed", mode=MODE, pid=os.getpid()))


if ENABLED:
    try:
        install()
    except Exception as _exc:  # noqa: BLE001
        _err("install", _exc)
