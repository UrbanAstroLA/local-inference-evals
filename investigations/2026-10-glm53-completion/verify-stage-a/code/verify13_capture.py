# SPDX-License-Identifier: Apache-2.0
"""verify13: observe-only capture for PLAN.md amendment 13 (D3, D4 inputs, D5).

Inert unless GLM53_VERIFY13=1.  Installed as ``vllm/verify13_capture.py`` in a
diagnostic image; imported only from a guarded block appended to
``fused_moe/router/gate_linear.py``.  It never writes to an engine tensor:
engine tensors are only read (async device->pinned-host copies into
hook-owned slabs); every store goes to hook-owned memory or files.

What it records (every tensor carries a label naming its computation, see LABELS)
  D3  per sampled (eager forward, MoE layer): the gate input hidden states
      (bf16, rows [a, a+R) of the M-row batch) and "logits_shipped" = the shipped
      computation's output (bf16-result GEMM cast to fp32), the rows' request
      attribution from the runner's host-side InputBatch (request id, prompt
      length, prefill/decode, batch kind), alignment/stride of x and logits; once
      per layer the GateLinear dispatch flags and sha256 of gate weight / bias.
  D4  for the same (forward, layer): topk_ids / topk_weights as consumed by
      Exl3MoEMethod.apply, its per-rank routed-expert output (pre all-reduce),
      whether apply's x is the gate's x (else apply's x too), the plan branch
      (decode trellis plan / prefill plan / tiled) and tp rank; once per layer the
      shard geometry, projection-tier membership, the cached runtime's key fields
      and plan ids, the swiglu_limit on the layer and inside the plans, the
      installed exl3.py sha256 and source-line anchors of the serving path.
  D5  every WorkspaceManager resize with phase (graph_capture, post_capture_warmup,
      serving), old/new pointer and size, caller; every workspace view bound while
      a CUDA graph is being captured (pointer, bytes, mode, batch size); for each
      resize, whether the freed range holds an address a captured graph retained.
      Host-only bookkeeping; no device work.

Graph safety (by design)
  * Python in this module runs only where the engine runs Python.  FULL and
    breakable-PIECEWISE replays do not re-run the model's Python, so the hooks
    cannot add work to a replay.  During any capture (``is_current_stream_
    capturing``) and whenever the forward context's runtime mode is not NONE,
    the D3/D4 hooks return before touching a tensor.
  * Sampling happens only on eager (cudagraph_runtime_mode NONE) forwards after
    the run script arms it (file <dir>/ARM); profile, dummy and capture-phase
    runs happen before arming.
  * Hot path: host-int slicing, ``pinned.copy_(src, non_blocking=True)`` and
    ``torch.cuda.Event().record()``.  No .item/.cpu/.tolist/synchronize,
    no boolean-mask indexing, no device allocation of hook state.  The writer
    thread polls ``Event.query()`` (no blocking sync) before reading a slab.
  * Pinned slabs are allocated by the writer thread when ARM appears, never by
    the engine thread.

Env
  GLM53_VERIFY13           "1" enables (anything else: inert)
  GLM53_VERIFY13_DIR       output dir (default /tmp/verify13)
  GLM53_VERIFY13_EVERY     sample every k-th eager forward (default 1)
  GLM53_VERIFY13_STRIDE    layer l sampled on forward f when (l + f) % STRIDE == 0 (default 14)
  GLM53_VERIFY13_FULL_ROWS capture all rows when M <= this (default 256)
  GLM53_VERIFY13_WIN_ROWS  otherwise a window of this many rows (default 64)
  GLM53_VERIFY13_MAX_MB    per-rank cap on captured bytes (default 4096)
  GLM53_VERIFY13_SLABS     pinned slabs per rank (default 192)
  GLM53_VERIFY13_OUT       "1" also capture the routed-expert output (default 1)
  GLM53_VERIFY13_WS        "1" workspace observer (default 1)
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import queue
import re
import sys
import threading
import time

import numpy as np
import torch

ENABLED = os.environ.get("GLM53_VERIFY13", "").strip() == "1"
OUT_DIR = os.environ.get("GLM53_VERIFY13_DIR", "/tmp/verify13")
EVERY = max(1, int(os.environ.get("GLM53_VERIFY13_EVERY", "1")))
STRIDE = max(1, int(os.environ.get("GLM53_VERIFY13_STRIDE", "14")))
FULL_ROWS = int(os.environ.get("GLM53_VERIFY13_FULL_ROWS", "256"))
WIN_ROWS = int(os.environ.get("GLM53_VERIFY13_WIN_ROWS", "64"))
MAX_BYTES = int(float(os.environ.get("GLM53_VERIFY13_MAX_MB", "4096")) * (1 << 20))
NSLABS = int(os.environ.get("GLM53_VERIFY13_SLABS", "192"))
CAP_OUT = os.environ.get("GLM53_VERIFY13_OUT", "1") == "1"
WS_ON = os.environ.get("GLM53_VERIFY13_WS", "1") == "1"

FIRST_MOE_LAYER = 3          # mlp_layer_types: layers 0-2 dense, 3-44 sparse
HIDDEN, NEXP, TOPK = 4096, 288, 8
# Every captured logits tensor is labelled by the computation that produced it (amendment 13a).
LABELS = {
    "x": "gate input hidden states as passed to GateLinear (bf16), rows [a, a+r) of the M-row batch",
    "logits_shipped": "shipped: bf16-result GEMM cast to fp32 (GateLinear output as consumed by the router)",
    "topk_ids": "routing ids as consumed by Exl3MoEMethod.apply",
    "topk_weights": "routing weights as consumed by Exl3MoEMethod.apply",
    "out": "routed-expert output of this rank (rank-sliced partial, before the TP all-reduce)",
    "x_apply": "Exl3MoEMethod.apply input when it is not the gate input tensor",
}
_ALIGN = 256
_GATE_RE = re.compile(r"(?:^|\.)language_model\.model\.layers\.(\d+)\.mlp\.gate$")
_EXPERTS_RE = re.compile(r"(?:^|\.)language_model\.model\.layers\.(\d+)\.mlp\.experts$")
# Fallback for checkpoints without the language_model prefix (tests).
_GATE_RE2 = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate$")
_EXPERTS_RE2 = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts$")


def _aligned(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


# Upper bound of one record: rows * (x bf16 + logits fp32 + ids int64 + weights fp32
# + out bf16) + a second x if apply's x is a different tensor.
_ROW_BYTES = HIDDEN * 2 + NEXP * 4 + TOPK * 8 + TOPK * 4 + HIDDEN * 2 + HIDDEN * 2
SLAB_BYTES = _aligned(max(FULL_ROWS, WIN_ROWS) * _ROW_BYTES + 8 * _ALIGN)


class _State:
    def __init__(self):
        self.rank = None
        self.fwd = 0                  # eager forwards seen (counted at the first MoE gate)
        self.armed = False
        self.disarmed = False
        self.pool = None              # pinned uint8 [NSLABS, SLAB_BYTES] (writer-allocated)
        self.pool_pinned = False
        self.free = None              # queue of free slab indices
        self.pending = {}             # layer -> record being filled
        self.q = queue.Queue()        # completed records -> writer
        self.writer = None
        self.bytes = 0
        self.seq = 0
        self.fh = None
        self.lock = threading.Lock()
        self.layer_info_done = set()
        self.router_info_done = set()
        self.weight_jobs = []         # (layer, weight, bias) for writer-side checksums
        self.captured_done = False    # capture_model returned (D5)
        self.in_capture_model = False
        self.in_warmup_kernels = False
        self.ws_calls = 0
        self.batch = None             # last real InputBatch (V2 runner prepare_inputs), host fields only
        self.batch_seq = 0
        self.cap_bindings = {}        # (ptr, nbytes, mode, num_tokens, caller) -> first t; workspace views bound in capture
        self.runtime_info = None
        self.c = dict(gate_calls=0, gate_capturing=0, gate_mode_not_none=0, gate_no_ctx=0,
                      gate_other_device=0, fwd_eager=0, sampled=0, drop_pool=0, drop_cap=0,
                      drop_not_ready=0, apply_calls=0, apply_capturing=0, apply_matched=0,
                      apply_unmatched=0, incomplete=0, written=0, errors=0, ws_resizes=0,
                      ws_resizes_after_capture=0, ws_capture_bindings=0, ws_invalidating_resizes=0,
                      batch_info_missing=0)
        self.errors = []
        self.installed = set()
        self.dev = None


S = _State()


# ------------------------------------------------------------------ utils
def _capturing() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False


def _runtime_mode() -> str | None:
    """'NONE' for eager forwards, other names for graph modes, None if no context."""
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available
        if not is_forward_context_available():
            return None
        m = get_forward_context().cudagraph_runtime_mode
        return getattr(m, "name", str(m))
    except Exception:  # noqa: BLE001
        return None


def _rank() -> int:
    if S.rank is None:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            S.rank = int(get_tensor_model_parallel_rank())
        except Exception:  # noqa: BLE001
            return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1")))
    return S.rank


def _log(rec: dict) -> None:
    try:
        with S.lock:
            if S.fh is None:
                os.makedirs(OUT_DIR, exist_ok=True)
                S.fh = open(os.path.join(OUT_DIR, f"v13_p{os.getpid()}.jsonl"), "a")
            rec.setdefault("t", round(time.time(), 4))
            rec.setdefault("rank", _rank())
            rec.setdefault("pid", os.getpid())
            S.fh.write(json.dumps(rec, default=str) + "\n")
            S.fh.flush()
    except Exception:  # noqa: BLE001
        pass


def _err(where: str, exc: BaseException) -> None:
    S.c["errors"] += 1
    if len(S.errors) < 50:
        S.errors.append(f"{where}: {exc!r}")
        _log(dict(kind="hook_error", where=where, err=repr(exc)))


def _layer_from(name: str, rx, rx2) -> int:
    m = rx.search(name or "") or rx2.search(name or "")
    return int(m.group(1)) if m else -1


def select(fwd: int, layer: int, m: int) -> tuple[bool, int, int]:
    """Pure sampling rule shared by every rank: (sampled, row start a, rows R)."""
    if fwd % EVERY != 0 or (layer + fwd) % STRIDE != 0 or m <= 0:
        return False, 0, 0
    if m <= FULL_ROWS:
        return True, 0, m
    r = min(WIN_ROWS, m)
    h = fwd * 7919 + layer * 104729
    # one window in three starts at row 0: this runner orders decode rows first, so mixed
    # batches contribute decode rows; the others rotate over the batch.
    a = 0 if h % 3 == 0 else h % (m - r + 1)
    return True, a, r


def _num_decode_tokens():
    """Host-side batch composition if the forward context exposes it (rows are
    decode-first in this runner).  Never reads a device tensor."""
    try:
        from vllm.forward_context import get_forward_context
        md = get_forward_context().attn_metadata
        vals = md.values() if isinstance(md, dict) else (md if isinstance(md, list) else [md])
        for v in vals:
            if isinstance(v, list):
                v = v[0] if v else None
            for attr in ("num_decode_tokens", "num_decodes"):
                n = getattr(v, attr, None)
                if isinstance(n, int):
                    return {attr: n, "num_prefills": getattr(v, "num_prefills", None)
                            if isinstance(getattr(v, "num_prefills", None), int) else None}
    except Exception:  # noqa: BLE001
        pass
    return None


def _batch_rows(a: int, r: int, m: int):
    """Request attribution of captured rows from the runner's host-side InputBatch fields
    (numpy, no device reads).  Returns None if the stashed batch is not this forward's."""
    ib = S.batch
    if ib is None:
        return None
    try:
        nt = int(getattr(ib, "num_tokens_after_padding", -1))
        if nt != m:
            return dict(mismatch=True, batch_tokens_after_padding=nt, m=m)
        n = int(ib.num_reqs)
        ntok = int(ib.num_tokens)
        qsl = np.asarray(ib.query_start_loc_np[:n + 1]).astype(np.int64)
        rows = np.arange(a, a + r)
        ridx = np.searchsorted(qsl, rows, side="right") - 1
        ridx = np.where(rows < ntok, ridx, -1)
        used = sorted({int(i) for i in ridx.tolist() if i >= 0})

        def at(arr, i):
            try:
                return int(np.asarray(arr)[i])
            except Exception:  # noqa: BLE001
                return None
        isp = np.asarray(ib.is_prefilling_np[:n]).astype(bool) if getattr(ib, "is_prefilling_np", None) is not None else None
        kind = None
        if isp is not None and len(isp) == n:
            kind = "prefill_only" if isp.all() else ("decode_only" if not isp.any() else "mixed")
        reqs = {}
        for i in used:
            reqs[i] = dict(req_id=str(ib.req_ids[i]), prefill_len=at(ib.prefill_len_np, i),
                           num_computed=at(ib.num_computed_tokens_np, i),
                           num_scheduled=int(qsl[i + 1] - qsl[i]),
                           is_prefilling=None if isp is None else bool(isp[i]))
        return dict(mismatch=False, batch_kind=kind, num_reqs=n, num_tokens=ntok, m=m,
                    row_req=ridx.tolist(), reqs=reqs, batch_seq=S.batch_seq)
    except Exception as exc:  # noqa: BLE001
        return dict(error=repr(exc))


class _Rec:
    __slots__ = ("slab", "off", "spec", "meta", "x_ptr", "x_shape", "a", "r")

    def __init__(self, slab, meta, a, r):
        self.slab = slab
        self.off = 0
        self.spec = []
        self.meta = meta
        self.a = a
        self.r = r
        self.x_ptr = None
        self.x_shape = None


def _put(rec: _Rec, name: str, src: torch.Tensor) -> None:
    """Async copy of an engine tensor (rows already sliced) into the record's
    pinned slab.  Reads src only."""
    n = src.numel() * src.element_size()
    off = rec.off
    if off + n > SLAB_BYTES:
        raise RuntimeError(f"slab overflow {name}")
    dst = S.pool[rec.slab, off:off + n].view(src.dtype).view(src.shape)
    dst.copy_(src, non_blocking=True)
    rec.spec.append((name, str(src.dtype).replace("torch.", ""), list(src.shape), off, n))
    rec.off = _aligned(off + n)


# ------------------------------------------------------------------ D3 hook
def on_gate(gate, x: torch.Tensor, out) -> None:
    S.c["gate_calls"] += 1
    layer = getattr(gate, "_v13_layer", None)
    if layer is None:
        layer = _layer_from(getattr(gate, "prefix", ""), _GATE_RE, _GATE_RE2)
        gate._v13_layer = layer
    if layer < 0:
        return
    if layer not in S.layer_info_done:
        _layer_info(gate, layer, out)
    if _capturing():
        S.c["gate_capturing"] += 1
        return
    mode = _runtime_mode()
    if mode is None:
        S.c["gate_no_ctx"] += 1
        return
    if mode != "NONE":
        S.c["gate_mode_not_none"] += 1
        return
    if x.device.type not in ("cuda", "cpu"):
        S.c["gate_other_device"] += 1
        return
    if layer == FIRST_MOE_LAYER:
        S.fwd += 1
        S.c["fwd_eager"] += 1
    if not S.armed:
        return
    logits = out[0] if isinstance(out, tuple) else out
    m = int(x.shape[0])
    ok, a, r = select(S.fwd, layer, m)
    if not ok:
        return
    if S.bytes >= MAX_BYTES:
        S.c["drop_cap"] += 1
        return
    if S.pool is None:
        S.c["drop_not_ready"] += 1
        return
    old = S.pending.pop(layer, None)
    if old is not None:          # gate seen twice without apply: close the old one
        S.c["incomplete"] += 1
        _submit(old, complete=False)
    try:
        slab = S.free.get_nowait()
    except queue.Empty:
        S.c["drop_pool"] += 1
        return
    rows = _batch_rows(a, r, m)
    if rows is None or rows.get("mismatch") or rows.get("error"):
        S.c["batch_info_missing"] += 1
    meta = dict(fwd=S.fwd, layer=layer, m=m, a=a, r=r, mode=mode,
                x_dtype=str(x.dtype), logits_dtype=str(logits.dtype),
                x_ptr_mod256=int(x.data_ptr() % 256), x_stride=list(x.stride()),
                logits_ptr_mod256=int(logits.data_ptr() % 256), logits_stride=list(logits.stride()),
                labels={k: LABELS[k] for k in ("x", "logits_shipped")},
                batch=_num_decode_tokens(), rows=rows, t_gate=round(time.time(), 6))
    rec = _Rec(slab, meta, a, r)
    try:
        _put(rec, "x", x[a:a + r])
        _put(rec, "logits_shipped", logits[a:a + r])
    except Exception as exc:  # noqa: BLE001
        S.free.put(slab)
        _err("gate_put", exc)
        return
    rec.x_ptr = x.data_ptr()
    rec.x_shape = tuple(x.shape)
    S.pending[layer] = rec
    S.c["sampled"] += 1


def _layer_info(gate, layer: int, out) -> None:
    S.layer_info_done.add(layer)
    try:
        w = gate.weight
        b = getattr(gate, "e_score_correction_bias", None)
        logits = out[0] if isinstance(out, tuple) else out
        info = dict(kind="gate_info", layer=layer, prefix=getattr(gate, "prefix", None),
                    cls=type(gate).__name__, weight_dtype=str(w.dtype), weight_shape=list(w.shape),
                    out_dtype=str(getattr(gate, "out_dtype", None)),
                    logits_dtype=str(getattr(logits, "dtype", None)),
                    bias_dtype=None if b is None else str(b.dtype),
                    bias_ptr=None if b is None else b.data_ptr(),
                    weight_ptr_mod256=int(w.data_ptr() % 256), weight_stride=list(w.stride()),
                    capturing_at_info=_capturing())
        for k in ("allow_ll_bf16_gemm", "allow_specialized_router_gemm", "allow_dsv3_router_gemm",
                  "allow_fp32_router_gemm", "allow_bf16x3_router_gemm", "allow_cublas_router_gemm",
                  "_dsv3_max_batch"):
            info[k] = getattr(gate, k, None)
        try:
            from vllm.platforms import current_platform
            cap = current_platform.get_device_capability()
            info["device_capability"] = None if cap is None else [cap.major, cap.minor]
        except Exception:  # noqa: BLE001
            pass
        # Expected tier from the shipped dispatch (gate_linear.py forward):
        if info["allow_ll_bf16_gemm"] or info["allow_dsv3_router_gemm"]:
            tier = "1-2 (shape-dependent)"
        elif info["allow_fp32_router_gemm"]:
            tier = "3"
        elif info["allow_bf16x3_router_gemm"]:
            tier = "4"
        elif info["allow_cublas_router_gemm"]:
            tier = "5 torch.mm(out_dtype=fp32)"
        else:
            tier = "6 F.linear(bf16) then .to(out_dtype)"
        info["tier_from_flags"] = tier
        _log(info)
        S.weight_jobs.append((layer, w, b))
    except Exception as exc:  # noqa: BLE001
        _err("layer_info", exc)


# ------------------------------------------------------------------ D4 hook
def on_apply(layer_mod, x, topk_weights, topk_ids, out) -> None:
    S.c["apply_calls"] += 1
    layer = getattr(layer_mod, "_v13_layer", None)
    if layer is None:
        layer = _layer_from(getattr(layer_mod, "layer_name", ""), _EXPERTS_RE, _EXPERTS_RE2)
        try:
            layer_mod._v13_layer = layer
        except Exception:  # noqa: BLE001
            pass
    if layer < 0:
        return
    if ("apply", layer) not in S.layer_info_done:
        _apply_info(layer_mod, layer, topk_ids, topk_weights, out)
    if _capturing():
        S.c["apply_capturing"] += 1
        return
    rec = S.pending.pop(layer, None)
    if rec is None:
        return
    S.c["apply_matched"] += 1
    a, r = rec.a, rec.r
    try:
        rec.meta["same_x"] = bool(x.data_ptr() == rec.x_ptr and tuple(x.shape) == rec.x_shape)
        rec.meta["ids_dtype"] = str(topk_ids.dtype)
        rec.meta["labels"].update({k: LABELS[k] for k in ("topk_ids", "topk_weights", "out", "x_apply")})
        rec.meta["d4"] = _d4_record_meta(layer_mod, rec.meta["m"])
        rec.meta["w_dtype"] = str(topk_weights.dtype)
        _put(rec, "topk_ids", topk_ids[a:a + r])
        _put(rec, "topk_weights", topk_weights[a:a + r])
        if CAP_OUT and isinstance(out, torch.Tensor):
            rec.meta["out_dtype"] = str(out.dtype)
            _put(rec, "out", out[a:a + r])
        if not rec.meta["same_x"]:
            _put(rec, "x_apply", x[a:a + r])
    except Exception as exc:  # noqa: BLE001
        _err("apply_put", exc)
    _submit(rec, complete=True)


def _runtimes():
    """The cached rank-sliced EXL3 runtimes (exl3.py _RANK_SLICED_RUNTIMES); host objects only."""
    try:
        from vllm.model_executor.layers.quantization import exl3 as _e
        return list(getattr(_e, "_RANK_SLICED_RUNTIMES", {}).items())
    except Exception:  # noqa: BLE001
        return []


def _runtime_for(x_device_index):
    for key, rt in _runtimes():
        if isinstance(rt, dict) and isinstance(key, tuple) and len(key) > 1 and key[1] == x_device_index:
            return key, rt
    rts = _runtimes()
    return rts[0] if rts else (None, None)


def _plan_branch(rt, m: int) -> str:
    """Which planned B12x launch serves an m-row call (mirrors exl3.py _apply_rank_sliced)."""
    if not isinstance(rt, dict):
        return "unknown"
    if rt.get("min_trellis_m", 1) <= m <= rt.get("max_trellis_m", 32):
        return "trellis_plan (decode capacity)"
    if rt.get("prefill_plan") is not None and m > rt.get("max_trellis_m", 32):
        return ("prefill_plan" if m <= rt.get("max_batched_tokens", 0)
                else f"prefill_plan tiled by {rt.get('max_batched_tokens')}")
    return "eager parity path"


def _d4_record_meta(layer_mod, m: int) -> dict:
    """Per-record D4 provenance: serving path, plan branch and shard (host attributes only)."""
    try:
        w = getattr(layer_mod, "exl3_trellis_weights", None)
        key, rt = _runtime_for(S.dev.index if S.dev is not None else None)
        return dict(tp_rank=getattr(layer_mod, "exl3_tp_rank", None),
                    tp_size=getattr(layer_mod, "exl3_tp_size", None),
                    intermediate_per_partition=getattr(layer_mod, "exl3_intermediate_size_per_partition", None),
                    projection_mixed=getattr(layer_mod, "exl3_projection_mixed", None),
                    plan_branch=_plan_branch(rt, m),
                    trellis_plan_id=None if not isinstance(rt, dict) else id(rt.get("trellis_plan")),
                    prefill_plan_id=None if not isinstance(rt, dict) else id(rt.get("prefill_plan")),
                    weights_obj_id=None if w is None else id(w))
    except Exception as exc:  # noqa: BLE001
        return dict(error=repr(exc))


def _src_anchor(obj) -> dict:
    try:
        lines, start = inspect.getsourcelines(obj)
        return dict(file=os.path.basename(inspect.getsourcefile(obj)), first=start, last=start + len(lines) - 1)
    except Exception as exc:  # noqa: BLE001
        return dict(error=repr(exc))


def _apply_info(layer_mod, layer, topk_ids, topk_weights, out) -> None:
    S.layer_info_done.add(("apply", layer))
    try:
        bitr = getattr(layer_mod, "exl3_projection_bitrates", None)
        tiers = None
        if bitr:
            tiers = {}
            for vals in bitr:
                for proj, b in zip(("gate", "up", "down"), vals):
                    tiers.setdefault(str(b), {}).setdefault(proj, 0)
                    tiers[str(b)][proj] += 1
        info = dict(kind="experts_info", layer=layer, layer_name=getattr(layer_mod, "layer_name", None),
                    cls=type(layer_mod).__name__,
                    quant_method=type(getattr(layer_mod, "quant_method", None)).__name__,
                    swiglu_limit_on_layer=getattr(layer_mod, "swiglu_limit", "absent"),
                    projection_mixed=getattr(layer_mod, "exl3_projection_mixed", None),
                    mixed_bitrate=getattr(layer_mod, "exl3_mixed_bitrate", None),
                    use_ep=getattr(layer_mod, "use_ep", None),
                    tp_rank=getattr(layer_mod, "exl3_tp_rank", None),
                    tp_size=getattr(layer_mod, "exl3_tp_size", None),
                    intermediate_per_partition=getattr(layer_mod, "exl3_intermediate_size_per_partition", None),
                    hidden_size=getattr(layer_mod, "exl3_hidden_size", None),
                    local_num_experts=getattr(layer_mod, "local_num_experts", None),
                    global_num_experts=getattr(layer_mod, "exl3_global_num_experts", None),
                    max_num_batched_tokens=getattr(layer_mod, "exl3_max_num_batched_tokens", None),
                    params_dtype=str(getattr(layer_mod, "exl3_params_dtype", None)),
                    trellis_tile_config=str(getattr(layer_mod, "exl3_trellis_tile_config", None)),
                    projection_bitrates_sha256=None if not bitr else hashlib.sha256(
                        json.dumps([list(v) for v in bitr]).encode()).hexdigest(),
                    projection_tier_counts=tiers,
                    ids_dtype=str(topk_ids.dtype), weights_dtype=str(topk_weights.dtype),
                    out_dtype=str(getattr(out, "dtype", None)))
        # serving path: the cached runtime (key fields), plan ids, swiglu_limit actually in the plans
        try:
            from vllm.model_executor.layers.quantization import exl3 as _e
            src = inspect.getsourcefile(_e)
            info["exl3_py_sha256"] = hashlib.sha256(open(src, "rb").read()).hexdigest()
            M = _e.Exl3MoEMethod
            info["source_anchors"] = {n: _src_anchor(getattr(M, n)) for n in
                                      ("apply", "_apply_rank_sliced", "_rank_sliced_runtime",
                                       "_prepare_projection_mixed_weights", "_apply_mixed_rank_sliced")
                                      if hasattr(M, n)}
            plans, rts = [], []
            for key, rt in _runtimes():
                if not isinstance(rt, dict):
                    continue
                rts.append(dict(key=[str(k) for k in key] if isinstance(key, tuple) else str(key),
                                **{k: rt.get(k) for k in ("use_ep", "projection_ep", "min_trellis_m",
                                                          "max_trellis_m", "max_batched_tokens",
                                                          "parity_rows", "topk", "chunk")}))
                for pk in ("trellis_plan", "prefill_plan"):
                    p = rt.get(pk)
                    if p is None:
                        continue
                    caps = getattr(p, "caps", None)
                    lp = getattr(p, "launch_plan", None)
                    plans.append(dict(plan=pk, plan_id=id(p), caps_swiglu_limit=getattr(caps, "swiglu_limit", "absent"),
                                      launch_plan_swiglu_limit=getattr(lp, "swiglu_limit", "absent"),
                                      core_swiglu_limit=getattr(getattr(p, "_core_workspace_plan", None),
                                                                "swiglu_limit", "absent"),
                                      max_tokens=getattr(caps, "max_tokens", None),
                                      w4a16_block_size_m=getattr(caps, "w4a16_block_size_m", None),
                                      implementation=getattr(lp, "implementation", None)))
            info["runtimes"] = rts
            info["runtime_plans"] = plans
        except Exception as exc:  # noqa: BLE001
            info["runtime_plans_error"] = repr(exc)
        _log(info)
    except Exception as exc:  # noqa: BLE001
        _err("apply_info", exc)


def _submit(rec: _Rec, complete: bool) -> None:
    rec.meta["complete"] = complete
    ev = None
    try:
        if S.pool_pinned:
            ev = torch.cuda.Event()
            ev.record()
    except Exception as exc:  # noqa: BLE001
        _err("event", exc)
    S.bytes += rec.off
    S.q.put((rec, ev))


def on_router(router, topk_indices_dtype) -> None:
    key = id(router)
    if key in S.router_info_done:
        return
    S.router_info_done.add(key)
    try:
        b = getattr(router, "e_score_correction_bias", None)
        info = dict(kind="router_info", cls=type(router).__name__,
                    topk_indices_dtype=str(topk_indices_dtype),
                    bias_ptr=None if b is None else b.data_ptr())
        for k in ("top_k", "global_num_experts", "num_expert_group", "topk_group", "renormalize",
                  "scoring_func", "routed_scaling_factor", "num_fused_shared_experts"):
            info[k] = getattr(router, k, None)
        try:
            import vllm.envs as envs
            info["VLLM_USE_FUSED_MOE_GROUPED_TOPK"] = bool(envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK)
            info["VLLM_BATCH_INVARIANT"] = bool(envs.VLLM_BATCH_INVARIANT)
            info["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"] = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
        except Exception:  # noqa: BLE001
            pass
        info["eplb"] = getattr(router, "eplb_state", None) is not None
        _log(info)
    except Exception as exc:  # noqa: BLE001
        _err("router_info", exc)


# ------------------------------------------------------------------ writer thread
def _writer_loop() -> None:
    arm = os.path.join(OUT_DIR, "ARM")
    disarm = os.path.join(OUT_DIR, "DISARM")
    side = None
    last_summary = 0.0
    while True:
        try:
            if (not S.armed and not S.disarmed and os.path.exists(arm)
                    and pool_target(S.dev, torch.cuda.is_available()) is not None):
                _alloc_pool()
                S.armed = S.pool is not None
                _log(dict(kind="armed", ok=S.armed, fwd=S.fwd, slabs=NSLABS, slab_bytes=SLAB_BYTES,
                          pool_pinned=S.pool_pinned, dev=str(S.dev),
                          every=EVERY, stride=STRIDE, full_rows=FULL_ROWS, win_rows=WIN_ROWS,
                          max_bytes=MAX_BYTES, cap_out=CAP_OUT))
            if S.armed and os.path.exists(disarm):
                S.armed = False
                S.disarmed = True
                _log(dict(kind="disarmed", fwd=S.fwd))
            if S.weight_jobs and (S.dev is not None or not torch.cuda.is_available()):
                side = _do_weight_jobs(side)
            try:
                rec, ev = S.q.get(timeout=0.2)
            except queue.Empty:
                rec = None
            if rec is not None:
                while ev is not None and not ev.query():
                    time.sleep(0.001)
                _write(rec)
            if S.disarmed and S.q.empty() and not S.pending:
                flushed = os.path.join(OUT_DIR, f"FLUSHED_r{_rank()}")
                if not os.path.exists(flushed):
                    _summary("flushed")
                    open(flushed, "w").write(json.dumps(S.c) + "\n")
            if time.time() - last_summary > 60:
                last_summary = time.time()
                _summary("periodic")
        except Exception as exc:  # noqa: BLE001
            _err("writer", exc)
            time.sleep(0.5)


def pool_target(dev, cuda_available: bool):
    """Where the slab pool may live.  With CUDA present it must be pinned host memory for the
    device the gate actually runs on (a pageable destination would make every non_blocking
    device->host copy synchronous); until a CUDA forward has been seen there is no target."""
    if cuda_available:
        return "pinned" if (dev is not None and getattr(dev, "type", None) == "cuda") else None
    return "pageable"   # CPU-only (tests): copies are synchronous anyway


def _alloc_pool() -> None:
    if S.pool is not None:
        return
    try:
        target = pool_target(S.dev, torch.cuda.is_available())
        if target is None:
            return
        if target == "pinned":
            torch.cuda.set_device(S.dev)
            pool = torch.empty((NSLABS, SLAB_BYTES), dtype=torch.uint8, pin_memory=True)
            if not pool.is_pinned():
                raise RuntimeError("pinned slab pool allocation returned pageable memory")
        else:
            pool = torch.empty((NSLABS, SLAB_BYTES), dtype=torch.uint8)
        free = queue.Queue()
        for i in range(NSLABS):
            free.put(i)
        S.free = free
        S.pool_pinned = bool(pool.is_pinned())
        S.pool = pool
    except Exception as exc:  # noqa: BLE001
        _err("alloc_pool", exc)


def _do_weight_jobs(side):
    """sha256 of static gate weight / bias bytes, copied on a writer-private side
    stream (engine stream untouched; weights are never written after load)."""
    jobs, S.weight_jobs = S.weight_jobs, []
    for layer, w, b in jobs:
        try:
            out = {}
            for name, t in (("weight", w), ("bias", b)):
                if t is None:
                    continue
                if t.is_cuda:
                    if side is None:
                        torch.cuda.set_device(t.device)
                        side = torch.cuda.Stream(device=t.device)
                    host = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                    with torch.cuda.stream(side):
                        host.copy_(t.detach(), non_blocking=True)
                        ev = torch.cuda.Event()
                        ev.record(side)
                    while not ev.query():
                        time.sleep(0.001)
                else:
                    host = t.detach().clone()
                out[name + "_sha256"] = hashlib.sha256(
                    host.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            _log(dict(kind="gate_weight_sha", layer=layer, **out))
        except Exception as exc:  # noqa: BLE001
            _err("weight_sha", exc)
    return side


def _write(rec: _Rec) -> None:
    try:
        tensors = {}
        for name, dtype, shape, off, n in rec.spec:
            v = S.pool[rec.slab, off:off + n].view(getattr(torch, dtype)).view(shape)
            tensors[name] = v.clone()
        S.free.put(rec.slab)
        S.seq += 1
        meta = dict(rec.meta, seq=S.seq, rank=_rank(), spec=[s[:3] for s in rec.spec])
        d = os.path.join(OUT_DIR, f"rec_r{_rank()}")
        os.makedirs(d, exist_ok=True)
        fn = f"f{meta['fwd']:07d}_l{meta['layer']:02d}.pt"
        torch.save(dict(meta=meta, **tensors), os.path.join(d, fn))
        _log(dict(kind="record", file=fn, **{k: meta[k] for k in
                  ("seq", "fwd", "layer", "m", "a", "r", "mode", "complete") if k in meta},
                  same_x=meta.get("same_x"), batch=meta.get("batch")))
        S.c["written"] += 1
    except Exception as exc:  # noqa: BLE001
        try:
            S.free.put(rec.slab)
        except Exception:  # noqa: BLE001
            pass
        _err("write", exc)


def _summary(reason: str) -> None:
    _log(dict(kind="summary", reason=reason, counters=dict(S.c), bytes=S.bytes, fwd=S.fwd,
              armed=S.armed, disarmed=S.disarmed, errors=S.errors[:10], ws_calls=S.ws_calls,
              installed=sorted(S.installed)))


def _ensure_writer(dev) -> None:
    # Only a CUDA device may become the capture device.  GateLinear can be constructed on
    # the CPU and moved later (smoke attempt 1: S.dev became cpu at __init__ and the slab
    # pool was then allocated pageable); the first CUDA forward sets the device instead.
    if S.dev is None and dev is not None and getattr(dev, "type", None) == "cuda":
        S.dev = dev
    if S.writer is None:
        S.writer = threading.Thread(target=_writer_loop, name="verify13-writer", daemon=True)
        S.writer.start()


# ------------------------------------------------------------------ D5 workspace observer
def _ws_state(mgr):
    out = []
    for ws in getattr(mgr, "_current_workspaces", []) or []:
        out.append(None if ws is None else (int(ws.data_ptr()), int(ws.numel() * ws.element_size())))
    return out


def _caller() -> str:
    """First frame outside workspace.py / this module (frame walk only: no source reads)."""
    try:
        f = sys._getframe(2)
        for _ in range(12):
            if f is None:
                break
            fn = os.path.basename(f.f_code.co_filename)
            if fn not in ("workspace.py", "verify13_capture.py"):
                return f"{fn}:{f.f_lineno}:{f.f_code.co_name}"
            f = f.f_back
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _batch_desc():
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available
        if not is_forward_context_available():
            return None
        bd = get_forward_context().batch_descriptor
        return None if bd is None else int(getattr(bd, "num_tokens", -1))
    except Exception:  # noqa: BLE001
        return None


def _phase() -> str:
    if S.in_capture_model:
        return "graph_capture"
    if S.in_warmup_kernels:
        return "post_capture_warmup"
    if S.captured_done:
        return "serving"
    return "pre_capture"


def classify_resize(before, after, bindings) -> dict:
    """For each workspace slot whose buffer changed: the freed range [ptr, ptr+size) and every
    workspace view bound while a CUDA graph was being captured that lies in it.  A captured graph
    keeps the device address it recorded, so such a binding is invalidated by the free."""
    freed, hits = [], []
    for b, a in zip(before, after):
        if b is not None and b != a:
            freed.append(b)
    for (ptr, nbytes, mode, ntok, caller) in bindings:
        for fptr, fsize in freed:
            if fptr <= ptr < fptr + fsize:
                hits.append(dict(ptr=ptr, nbytes=nbytes, mode=mode, num_tokens=ntok, caller=caller))
    return dict(freed=freed, invalidated_bindings=hits[:64], n_invalidated=len(hits),
                classification=("captured-graph address invalidated" if hits
                                else "resize, no captured-graph address invalidated"))


def wrap_workspace(WM) -> None:
    if getattr(WM, "_v13", False):
        return
    orig = WM._ensure_workspace_size

    def _ensure_workspace_size(self, required_bytes):
        before = _ws_state(self)
        res = orig(self, required_bytes)
        S.ws_calls += 1
        try:
            if _capturing():
                # a view of `res` bound inside a captured graph: the graph replays this address
                key = (int(res.data_ptr()), int(required_bytes), _runtime_mode(), _batch_desc(), _caller())
                if key not in S.cap_bindings:
                    S.cap_bindings[key] = round(time.time(), 4)
                    S.c["ws_capture_bindings"] += 1
            after = _ws_state(self)
            if after != before:
                S.c["ws_resizes"] += 1
                if S.captured_done:
                    S.c["ws_resizes_after_capture"] += 1
                cls = classify_resize(before, after, list(S.cap_bindings))
                if cls["n_invalidated"]:
                    S.c["ws_invalidating_resizes"] += 1
                _log(dict(kind="ws_resize", required=int(required_bytes), before=before, after=after,
                          phase=_phase(), after_capture=S.captured_done, in_capture_model=S.in_capture_model,
                          in_warmup_kernels=S.in_warmup_kernels, capturing=_capturing(),
                          locked=bool(getattr(self, "_locked", False)), mode=_runtime_mode(),
                          num_tokens=_batch_desc(), caller=_caller(), calls=S.ws_calls,
                          n_capture_bindings_so_far=len(S.cap_bindings), **cls))
        except Exception as exc:  # noqa: BLE001
            _err("ws", exc)
        return res

    WM._ensure_workspace_size = _ensure_workspace_size
    WM._v13 = True
    S.installed.add("workspace")


def _ws_snapshot():
    try:
        from vllm.v1.worker import workspace as wsmod
        m = getattr(wsmod, "_manager", None)
        return None if m is None else dict(state=_ws_state(m), locked=bool(getattr(m, "_locked", False)))
    except Exception as exc:  # noqa: BLE001
        return repr(exc)


def wrap_runner(Runner) -> None:
    if getattr(Runner, "_v13", False):
        return
    orig = Runner.capture_model

    def capture_model(self, *a, **kw):
        S.in_capture_model = True
        _log(dict(kind="capture_model_begin", ws=_ws_snapshot()))
        try:
            return orig(self, *a, **kw)
        finally:
            S.in_capture_model = False
            S.captured_done = True
            _log(dict(kind="capture_model_end", ws=_ws_snapshot(),
                      capture_bindings=[dict(ptr=k[0], nbytes=k[1], mode=k[2], num_tokens=k[3], caller=k[4])
                                        for k in S.cap_bindings]))

    Runner.capture_model = capture_model
    Runner._v13 = True
    S.installed.add("runner_v2_capture_model")


def wrap_prepare_inputs(Runner) -> None:
    """Stash the real InputBatch of each step (host fields read later, at sampling time)."""
    if getattr(Runner.prepare_inputs, "_v13", False):
        return
    orig = Runner.prepare_inputs

    def prepare_inputs(self, *a, **kw):
        ib = orig(self, *a, **kw)
        S.batch = ib
        S.batch_seq += 1
        return ib

    prepare_inputs._v13 = True
    Runner.prepare_inputs = prepare_inputs
    S.installed.add("runner_v2_prepare_inputs")


def wrap_warmup_kernels(worker_mod) -> None:
    orig = getattr(worker_mod, "warmup_kernels", None)
    if orig is None or getattr(orig, "_v13", False):
        return

    def warmup_kernels(*a, **kw):
        S.in_warmup_kernels = True
        _log(dict(kind="warmup_kernels_begin", after_capture=S.captured_done, ws=_ws_snapshot()))
        try:
            return orig(*a, **kw)
        finally:
            S.in_warmup_kernels = False
            _log(dict(kind="warmup_kernels_end", ws=_ws_snapshot()))

    warmup_kernels._v13 = True
    worker_mod.warmup_kernels = warmup_kernels
    S.installed.add("worker_warmup_kernels")


# ------------------------------------------------------------------ installers
def wrap_apply(Method) -> None:
    """Wrap a quant method's apply(layer, x, topk_weights, topk_ids, ...)."""
    if getattr(Method, "_v13", False):
        return
    orig = Method.apply
    sig = inspect.signature(orig)

    def apply(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        try:
            ba = sig.bind(self, *args, **kwargs)
            p = ba.arguments
            on_apply(p["layer"], p["x"], p["topk_weights"], p["topk_ids"], out)
        except Exception as exc:  # noqa: BLE001
            _err("apply", exc)
        return out

    Method.apply = apply
    Method._v13 = True
    S.installed.add(f"apply:{Method.__name__}")


def wrap_router(Router) -> None:
    if getattr(Router, "_v13", False):
        return
    orig = Router.select_experts

    def select_experts(self, hidden_states, router_logits, topk_indices_dtype=None, **kw):
        res = orig(self, hidden_states, router_logits, topk_indices_dtype=topk_indices_dtype, **kw)
        try:
            on_router(self, topk_indices_dtype)
        except Exception as exc:  # noqa: BLE001
            _err("router", exc)
        return res

    Router.select_experts = select_experts
    Router._v13 = True
    S.installed.add(f"router:{Router.__name__}")


def _deferred_install() -> None:
    """Runs once, at the first GateLinear construction (model build, worker process)."""
    if "deferred" in S.installed:
        return
    S.installed.add("deferred")
    for what, fn in (
        ("exl3", lambda: wrap_apply(__import__(
            "vllm.model_executor.layers.quantization.exl3", fromlist=["Exl3MoEMethod"]).Exl3MoEMethod)),
        ("router", lambda: wrap_router(__import__(
            "vllm.model_executor.layers.fused_moe.router.fused_moe_router",
            fromlist=["FusedMoERouter"]).FusedMoERouter)),
        ("workspace", lambda: WS_ON and wrap_workspace(__import__(
            "vllm.v1.worker.workspace", fromlist=["WorkspaceManager"]).WorkspaceManager)),
        ("runner", lambda: WS_ON and wrap_runner(__import__(
            "vllm.v1.worker.gpu.model_runner", fromlist=["GPUModelRunner"]).GPUModelRunner)),
        ("batch", lambda: wrap_prepare_inputs(__import__(
            "vllm.v1.worker.gpu.model_runner", fromlist=["GPUModelRunner"]).GPUModelRunner)),
        ("warmup", lambda: WS_ON and wrap_warmup_kernels(__import__(
            "vllm.v1.worker.gpu_worker", fromlist=["warmup_kernels"]))),
    ):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            _err(f"install_{what}", exc)
    _log(dict(kind="installed", installed=sorted(S.installed), every=EVERY, stride=STRIDE,
              full_rows=FULL_ROWS, win_rows=WIN_ROWS, max_bytes=MAX_BYTES, slabs=NSLABS,
              slab_bytes=SLAB_BYTES, cap_out=CAP_OUT, ws=WS_ON))


def install(GateLinear) -> None:
    """Called from the guarded block at the end of gate_linear.py when enabled."""
    if not ENABLED or getattr(GateLinear, "_v13", False):
        return
    orig_init = GateLinear.__init__
    orig_forward = GateLinear.forward

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        try:
            _deferred_install()
            _ensure_writer(self.weight.device if self.weight.device.type != "meta" else None)
        except Exception as exc:  # noqa: BLE001
            _err("gate_init", exc)

    def forward(self, x):
        out = orig_forward(self, x)
        try:
            if (S.dev is None or S.dev.type != "cuda") and x.device.type == "cuda":
                S.dev = x.device
            on_gate(self, x, out)
        except Exception as exc:  # noqa: BLE001
            _err("gate", exc)
        return out

    GateLinear.__init__ = __init__
    GateLinear.forward = forward
    GateLinear._v13 = True
    GateLinear._v13_orig_forward = orig_forward
    S.installed.add("gate_linear")
