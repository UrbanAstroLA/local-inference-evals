# SPDX-License-Identifier: Apache-2.0
"""local-cap0: point-0 exact-byte capture and warm-prefill localisation (investigation 2026-10-glm53-completion,
package tile-p4/capture-p0/, PROTOCOL.md).  Observe-only.

INERT unless GLM53_P4=on, GLM53_P4_S0=on and GLM53_CAP0=on.  It is imported by p4_hooks only when GLM53_CAP0=on
(the marked "# local-cap0" lines in p4_hooks.py); with the switch off this module is never imported and
p4_hooks.py minus its marked lines is byte-identical to the stage-0 module.

1. Core capture (both ranks; GLM53_CAP0=on).  For a keyed request whose tag starts with "CAP-", at its FIRST step
   (point 0: p4_hooks._s0_batch -> forcewin on_batch -> GPUModelRunner.prepare_attn, before
   model_state.preprocess_state, i.e. before the mamba align pre-copy and the forward), copy to host the exact bytes
   of these slices of the KV-cache storage, addressed by the published formulas (cap0_hooks_layout.py) from the
   request's block-table row:
     MLA KVCacheTensor of layers 3 and 7, logical blocks 0..n_cached/4352-1 (rank 0 owns them: MLA owner map)
     indexer (pool) KVCacheTensor of layers 3 and 7, same blocks (pool records 511 / 1022 also split out)
     KDA layers 2 and 4 recurrent state and layers 0 and 4 conv state (rows 0-2 / 3-5 separately) in the cached-state
       column n_cached/4352-1; the layer-4 recurrent state of the pre-copy destination column (sha256 only: a
       timing check; it must still differ from the cached column at point 0)
   Each slice is copied twice: (a) as a raw byte range of the tensor's untyped storage at the formula offset and
   (b) through the engine's own view of that block; (a) == (b) is required (G-addr).  The stage-0 two-sum digest of
   the same block is also computed on device with p4_hooks._block_digest and compared with the host two-sum of the
   copied bytes.  Bytes -> PRIVATE/cap0_<tag>_r<rank>.npz; offsets, ids, sha256 per slice -> log/cap0_r<rank>.jsonl.
2. Localisation (GLM53_CAP0_LOC=on; both ranks).  For a solitary eager warm step (> 64 tokens, graph mode "none")
   of a keyed request whose tag starts with "CW-" and ends with "-L", for token positions < GLM53_CAP0_LOC_TOKENS:
   sha256 of every token row of the tensors at the boundaries listed in cap0_hooks_layout.LOC_ORDER (forward
   hooks on layers 3 and 4 and their submodules, plus a wrapper of kda.causal_conv1d_fn that records only while
   layer 4's KDA forward runs), and per step the running KDA column of layers 2 and 4 after the state write.
   Raw rows are kept for the first GLM53_CAP0_RAW_FIRST positions and the chunk boundaries; the first warm of the
   session ("reference") is held in host memory (<= GLM53_CAP0_HOLD_MB per rank), and for later warms the first
   RAW_DIFF positions per tensor whose sha256 differs from the reference are kept raw for both.  Hashing runs on the
   host from read-only copies: rows are copied in blocks of <= 8 MiB (device temporaries only for non-contiguous
   inputs), then .cpu() (a stream sync inside an eager step; never inside a capture: every hook returns at once
   while a stream is capturing or when the step is not armed).  -> PRIVATE/loc_<tag>_r<rank>.npz at removal.

Env: GLM53_CAP0 ("on"), GLM53_CAP0_LOC ("on"), GLM53_CAP0_LOC_TOKENS (13056), GLM53_CAP0_RAW_FIRST (64),
     GLM53_CAP0_HOLD_MB (4096)
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import time

import numpy as np
import torch

from vllm import cap0_hooks_layout as LY

_on = lambda k: os.environ.get(k, "").strip().lower() == "on"  # noqa: E731
ON = _on("GLM53_P4") and _on("GLM53_P4_S0") and _on("GLM53_CAP0")
LOC = ON and _on("GLM53_CAP0_LOC")
LOC_TOKENS = int(os.environ.get("GLM53_CAP0_LOC_TOKENS", str(LY.N_CACHED)))
RAW_FIRST = int(os.environ.get("GLM53_CAP0_RAW_FIRST", "64"))
HOLD_MB = int(os.environ.get("GLM53_CAP0_HOLD_MB", "4096"))
RAW_DIFF = 8
ROWBLK = 8 << 20
MMAX = 64
CAP_PREFIX = "CAP-"
WARM_PREFIX = "CW-"
LOC_SUFFIX = "-L"
MLA_LAYERS = (3, 7)
IDX_LAYERS = (3, 7)
REC_LAYERS = (2, 4)
CONV_LAYERS = (0, 4)
STATE_LAYERS = (2, 4)
SCHEMA = "local-cap0/1"
BOUNDARY = sorted({p for s, n in LY.warm_chunks(LOC_TOKENS) for p in (s, s + n - 1)})


class _State:
    def __init__(self):
        self.p4 = None
        self.runner = None
        self.rank = None
        self.registered = False
        self.targets = {}           # key -> dict(kind, layer, name, group, views, raw, ok, checks)
        self.seen = set()           # uids seen in any on_batch
        self.captured = set()
        self.arm = None
        self.cur_layer = None
        self.loc = {}               # uid -> accumulator
        self.ref_uid = None
        self.ref_dig = {}           # (point, tensor) -> [LOC_TOKENS, 32] uint8
        self.ref_have = {}          # (point, tensor) -> [LOC_TOKENS] bool
        self.hold = {}              # (point, tensor) -> [LOC_TOKENS, rowbytes] uint8 (reference rows)
        self.hold_bytes = 0
        self.hold_refused = []
        self.hooks = []
        self.conv_patched = False
        self.layers = {}
        self.n_capture = 0
        self.errors = 0


S = _State()


# ------------------------------------------------------------------ helpers
def _capturing() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False


def _log(rec):
    if S.p4 is not None:
        S.p4._log(rec, prefix="cap0")


def _err(where, exc):
    S.errors += 1
    if S.p4 is not None:
        S.p4._err("cap0." + where, exc)


def _sha(a) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast("B")).hexdigest()


def _host_u8(t: torch.Tensor) -> np.ndarray:
    """Read-only host copy of a contiguous device tensor's bytes (flat uint8)."""
    t = t.detach()
    if not t.is_contiguous():
        t = t.contiguous()
    return t.reshape(-1).view(torch.uint8).cpu().numpy()


def _twosum_host(u8: np.ndarray, esize: int):
    """The stage-0 digest (p4_hooks._block_digest) of these bytes: sums of the signed words, all and [::3]."""
    w = u8.view({1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[esize]).astype(np.int64)
    return [int(w.sum()), int(w[::3].sum())]


def raw_u8(t: torch.Tensor) -> torch.Tensor:
    """A flat uint8 tensor over the whole untyped storage of t (byte 0 = storage byte 0)."""
    st = t.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=t.device).set_(st, 0, (st.nbytes(),), (1,))


# ------------------------------------------------------------------ registration (capture_model, before capture)
def _match(names, suffix):
    hits = [n for n in names if n.endswith(suffix) and (n == suffix or n[-len(suffix) - 1] == ".")]
    return hits[0] if len(hits) == 1 else (None if not hits else hits)


def layout_checks(kind, views, nblk):
    """Compare a target's live views with the published layout.  Returns (ok, checks)."""
    c = {}
    try:
        if kind == "mla":
            t = views[0]
            c.update(dtype=str(t.dtype), shape=list(t.shape), stride0_bytes=int(t.stride(0)) * t.element_size(),
                     offset_bytes=int(t.storage_offset()) * t.element_size(), storage_bytes=int(t.untyped_storage().nbytes()))
            ok = (t.dtype == torch.uint8 and list(t.shape[1:]) == [LY.KERNEL_TOKENS, LY.MLA_REC]
                  and int(t.shape[0]) == nblk * LY.MLA_PPB and c["stride0_bytes"] == LY.KERNEL_TOKENS * LY.MLA_REC
                  and c["offset_bytes"] == 0 and c["storage_bytes"] == nblk * LY.P)
        elif kind == "idx":
            t = views[0]
            c.update(dtype=str(t.dtype), shape=list(t.shape), stride0_bytes=int(t.stride(0)) * t.element_size(),
                     offset_bytes=int(t.storage_offset()) * t.element_size(), storage_bytes=int(t.untyped_storage().nbytes()))
            ok = (t.dtype == torch.uint8 and list(t.shape[1:]) == [LY.IDX_ROWS, LY.IDX_VAL + 4]
                  and int(t.shape[0]) == nblk * LY.IDX_PPB and c["stride0_bytes"] == LY.IDX_POOL_PAGE
                  and c["offset_bytes"] == 0 and c["storage_bytes"] == nblk * LY.IDX_PAGE)
        else:
            conv, rec = views
            c.update(conv_shape=list(conv.shape), conv_dtype=str(conv.dtype), rec_shape=list(rec.shape),
                     rec_dtype=str(rec.dtype), conv_stride0_bytes=int(conv.stride(0)) * conv.element_size(),
                     rec_stride0_bytes=int(rec.stride(0)) * rec.element_size(),
                     conv_offset_bytes=int(conv.storage_offset()) * conv.element_size(),
                     rec_offset_bytes=int(rec.storage_offset()) * rec.element_size(),
                     storage_bytes=int(conv.untyped_storage().nbytes()),
                     same_storage=conv.untyped_storage().data_ptr() == rec.untyped_storage().data_ptr(),
                     conv_block_contiguous=bool(conv[0].is_contiguous()), rec_block_contiguous=bool(rec[0].is_contiguous()))
            ok = (conv.dtype == torch.bfloat16 and list(conv.shape[1:]) == [LY.CONV_ROWS, LY.CONV_ROW_BYTES // 2]
                  and rec.dtype == torch.float32 and int(rec[0].numel()) * 4 == LY.REC_BYTES
                  and int(conv.shape[0]) == nblk and int(rec.shape[0]) == nblk
                  and c["conv_stride0_bytes"] == LY.P and c["rec_stride0_bytes"] == LY.P
                  and c["conv_offset_bytes"] == 0 and c["rec_offset_bytes"] == LY.REC_OFF
                  and c["storage_bytes"] == nblk * LY.P and c["same_storage"]
                  and c["conv_block_contiguous"] and c["rec_block_contiguous"])
        return bool(ok), c
    except Exception as exc:  # noqa: BLE001
        c["exception"] = repr(exc)[:200]
        return False, c


def resolve_targets(runner):
    """{key: target} for every capture slice present on this rank (MLA layers absent on a non-owner rank)."""
    kvc = runner.kv_cache_config
    cc = getattr(runner, "compilation_config", None) or runner.vllm_config.compilation_config
    ctx = cc.static_forward_context
    group_of = {}
    for g, grp in enumerate(kvc.kv_cache_groups):
        for name in grp.layer_names:
            group_of[name] = g
    names = sorted(group_of)
    nblk = int(kvc.num_blocks)
    out, missing = {}, []
    want = [("mla", L, f"layers.{L}.self_attn.attn") for L in MLA_LAYERS] + \
           [("idx", L, f"layers.{L}.self_attn.indexer.k_cache") for L in IDX_LAYERS] + \
           [("kda", L, f"layers.{L}.self_attn") for L in sorted(set(REC_LAYERS) | set(CONV_LAYERS))]
    for kind, L, suf in want:
        name = _match(names, suf)
        if name is None or isinstance(name, list):
            missing.append(dict(kind=kind, layer=L, why="no layer" if name is None else "ambiguous", hits=name))
            continue
        mod = ctx.get(name)
        parts = S.p4._kv_parts(getattr(mod, "kv_cache", None)) if mod is not None else []
        views = [t for _s, t in parts]
        if (kind == "kda" and len(views) != 2) or (kind != "kda" and len(views) != 1):
            missing.append(dict(kind=kind, layer=L, name=name, why="kv_cache parts %d" % len(views)))
            continue
        ok, checks = layout_checks(kind, views, nblk)
        out[f"{kind}{L}"] = dict(kind=kind, layer=L, name=name, group=group_of[name], views=views,
                                 raw=raw_u8(views[0]), ok=ok, checks=checks)
    return out, missing, nblk


def register(runner):
    """p4_hooks.register (GPUModelRunner.capture_model, before any CUDA graph is captured), both ranks."""
    if not ON or S.registered:
        return
    try:
        from vllm import p4_hooks as p4
        S.p4 = p4
        S.runner = runner
        S.rank = p4._rank()
        S.targets, missing, nblk = resolve_targets(runner)
        hooks = install_loc(runner) if LOC else []
        S.registered = True
        vp = os.path.dirname(os.path.abspath(__file__))
        _log(dict(kind="cap0_init", schema=SCHEMA, rank=S.rank, loc=LOC, loc_tokens=LOC_TOKENS, raw_first=RAW_FIRST,
                  hold_mb=HOLD_MB, num_blocks=nblk, page=LY.P, idx_page=LY.IDX_PAGE,
                  targets={k: dict(kind=v["kind"], layer=v["layer"], name=v["name"], group=v["group"], ok=v["ok"],
                                   checks=v["checks"]) for k, v in S.targets.items()},
                  missing=missing, hooks=hooks, conv_patched=S.conv_patched,
                  mla_owners=os.environ.get("VLLM_GLM53_MLA_OWNERS", ""),
                  file_sha256={f: S.p4._sha_file(os.path.join(vp, f)) for f in ("cap0_hooks.py",
                                                                              "cap0_hooks_layout.py", "p4_hooks.py")}))
    except Exception as exc:  # noqa: BLE001
        _err("register", exc)


# ------------------------------------------------------------------ per step (p4_hooks._s0_batch; eager, not capturing)
def on_batch(input_batch, step, rows):
    """rows: p4_hooks sched rows [req_id, req_idx, uid, tag, n_sched, n_computed, n_computed_prefill,
    prefill_len, prefilling, n_draft] in batch order.  Never called while capturing."""
    if not S.registered:
        return
    S.arm = None
    S.cur_layer = None
    try:
        nt = int(input_batch.num_tokens)
        ntp = int(getattr(input_batch, "num_tokens_after_padding", nt))
        for b, row in enumerate(rows):
            uid, tag, pf = int(row[2]), str(row[3]), bool(row[8])
            first = uid >= 0 and uid not in S.seen
            if uid >= 0:
                S.seen.add(uid)
            if tag.startswith(CAP_PREFIX) and uid not in S.captured:
                if first and pf:
                    S.captured.add(uid)
                    capture(row, step, nt, ntp, len(rows))
                elif first:
                    _log(dict(kind="cap0_missed", tag=tag, uid=uid, step=step, why="first step not prefilling"))
        if LOC and len(rows) == 1:
            row = rows[0]
            uid, tag, n_sched, n_comp, pf = int(row[2]), str(row[3]), int(row[4]), int(row[5]), bool(row[8])
            if tag.startswith(WARM_PREFIX) and tag.endswith(LOC_SUFFIX) and pf and n_comp < LOC_TOKENS:
                if ntp == nt and ntp > MMAX:
                    _arm(input_batch, step, uid, tag, n_comp, n_sched, nt)
                else:
                    _log(dict(kind="cap0_loc_skip", tag=tag, uid=uid, step=step, nt=nt, ntp=ntp, why="not an eager step"))
    except Exception as exc:  # noqa: BLE001
        S.arm = None
        _err("on_batch", exc)


def after_forward():
    """p4_hooks.on_logits (after the forward, before the sampler): close the armed step."""
    a, S.arm, S.cur_layer = S.arm, None, None
    if a is None:
        return
    try:
        acc = S.loc.get(a["uid"])
        if acc is not None:
            acc["steps"].append(dict(step=a["step"], pos0=a["pos0"], n=a["n"], m=a["m"], mode=S.p4.S.mode,
                                     ids_sha256=a["ids_sha256"], pos_sha256=a["pos_sha256"],
                                     secs=round(a["secs"], 4), wall=round(time.time() - a["t0"], 4),
                                     points=sorted(set(a["points"]))))
    except Exception as exc:  # noqa: BLE001
        _err("after_forward", exc)


# ------------------------------------------------------------------ 1. core capture
def _rows_of(req_idx):
    bt = S.runner.block_tables
    out = []
    for g in range(bt.num_kv_cache_groups):
        nbk = int(bt.num_blocks.np[g, req_idx])
        bpk = max(int(bt.blocks_per_kv_block[g]), 1)
        ent = bt.block_tables[g].gpu[req_idx, :nbk].cpu().numpy().astype(np.int64)
        out.append(dict(nbk=nbk, bpk=bpk, entries=ent))
    return out


def plan_slices(targets, groups, n_cached, n_sched, chunks):
    """The capture slices for this rank: list of dicts (key, target key, kind, logical, physical, lo, hi, view
    selector).  Pure given the block-table rows (tested on CPU)."""
    out, problems = [], []
    nb = n_cached // LY.BLOCK_TOKENS
    jstate = LY.cached_state_col(n_cached)
    jdst = LY.precopy_dst_col(n_cached, n_sched)
    for key, tg in sorted(targets.items()):
        g = groups[tg["group"]]
        bpk, ent = g["bpk"], g["entries"]
        if tg["kind"] in ("mla", "idx"):
            for j in range(nb):
                if (j + 1) * bpk > len(ent):
                    problems.append(dict(key=key, j=j, why="row shorter than the cached span"))
                    continue
                b = LY.physical(ent, j, bpk)
                contiguous = bool(np.array_equal(ent[j * bpk:(j + 1) * bpk], b * bpk + np.arange(bpk)))
                lo, hi = LY.mla_block(b) if tg["kind"] == "mla" else LY.idx_block(b)
                pools = sorted(LY.never_written_pools(chunks, j)) if tg["kind"] == "idx" else None
                out.append(dict(key=f"{key}_b{j}", target=key, kind=tg["kind"], j=j, b=b, lo=lo, hi=hi,
                                kernel_pages_contiguous=contiguous, pools=pools))
        else:
            if (jstate + 1) * bpk > len(ent):
                problems.append(dict(key=key, j=jstate, why="row shorter than the cached-state column"))
                continue
            b = LY.physical(ent, jstate, bpk)
            L = tg["layer"]
            if L in CONV_LAYERS:
                (l0, h0), (l1, h1) = LY.kda_conv(b)
                out.append(dict(key=f"kda{L}_conv_live", target=key, kind="conv_live", j=jstate, b=b, lo=l0, hi=h0))
                out.append(dict(key=f"kda{L}_conv_inactive", target=key, kind="conv_inactive", j=jstate, b=b, lo=l1, hi=h1))
            if L in REC_LAYERS:
                lo, hi = LY.kda_rec(b)
                out.append(dict(key=f"kda{L}_rec", target=key, kind="rec", j=jstate, b=b, lo=lo, hi=hi))
            if L == 4 and (jdst + 1) * bpk <= len(ent):
                bd = LY.physical(ent, jdst, bpk)
                lo, hi = LY.kda_rec(bd)
                out.append(dict(key="kda4_rec_dst", target=key, kind="rec", j=jdst, b=bd, lo=lo, hi=hi, hash_only=True))
    return out, problems


def _view_bytes(tg, sl):
    """The same slice read through the engine's view (contiguous block selections; no index_select)."""
    v = tg["views"]
    b = sl["b"]
    if sl["kind"] == "mla":
        return _host_u8(v[0][b * LY.MLA_PPB:(b + 1) * LY.MLA_PPB]), v[0], torch.arange(LY.MLA_PPB) + b * LY.MLA_PPB
    if sl["kind"] == "idx":
        return _host_u8(v[0][b * LY.IDX_PPB:(b + 1) * LY.IDX_PPB]), v[0], torch.arange(LY.IDX_PPB) + b * LY.IDX_PPB
    if sl["kind"] == "conv_live":
        return _host_u8(v[0][b, :LY.CONV_LIVE_ROWS]), None, None
    if sl["kind"] == "conv_inactive":
        return _host_u8(v[0][b, LY.CONV_LIVE_ROWS:]), None, None
    return _host_u8(v[1][b]), v[1], torch.tensor([b])


def capture(row, step, nt, ntp, nrows):
    t0 = time.time()
    req_idx, uid, tag = int(row[1]), int(row[2]), str(row[3])
    n_sched, n_comp, n_comp_pf, pf_len = int(row[4]), int(row[5]), int(row[6]), int(row[7])
    rec = dict(kind="cap0_capture", tag=tag, uid=uid, step=step, rank=S.rank, req_idx=req_idx, n_sched=n_sched,
               n_computed=n_comp, n_computed_prefill=n_comp_pf, prefill_len=pf_len, nt=nt, ntp=ntp, batch_rows=nrows,
               n_cached_expected=LY.N_CACHED, mode_at_capture=S.p4.S.mode)
    try:
        if n_comp <= 0 or n_comp % LY.BLOCK_TOKENS:
            rec.update(ok=False, why="n_computed %d is not a positive multiple of %d" % (n_comp, LY.BLOCK_TOKENS))
            _log(rec)
            return
        groups = _rows_of(req_idx)
        rec["groups"] = [dict(nbk=g["nbk"], bpk=g["bpk"],
                              logical=[int(g["entries"][j * g["bpk"]]) // g["bpk"] for j in range(min(8, g["nbk"] // g["bpk"]))])
                         for g in groups]
        try:   # engine-side prompt ids of this request (UVA buffer; .cpu() is stream-ordered after its staged write)
            ids = S.runner.req_states.all_token_ids.gpu[req_idx, :pf_len].cpu().numpy().astype(np.int32)
            rec["prompt_ids_sha256"] = _sha(ids)
            rec["cached_ids_sha256"] = _sha(ids[:n_comp])
        except Exception as exc:  # noqa: BLE001
            rec["prompt_ids_sha256"] = None
            rec["prompt_ids_err"] = repr(exc)[:200]
        chunks = LY.warm_chunks(LY.WARM_TOKENS)
        slices, problems = plan_slices(S.targets, groups, n_comp, n_sched, chunks)
        arrays, out = {}, {}
        for sl in slices:
            tg = S.targets[sl["target"]]
            raw = tg["raw"][sl["lo"]:sl["hi"]].cpu().numpy()
            vb, vt, pages = _view_bytes(tg, sl)
            d = dict(b=sl["b"], j=sl["j"], lo=sl["lo"], hi=sl["hi"], nbytes=int(raw.size), sha256=_sha(raw),
                     view_equal=bool(raw.size == vb.size and np.array_equal(raw, vb)), layout_ok=tg["ok"],
                     kernel_pages_contiguous=sl.get("kernel_pages_contiguous"))
            if vt is not None:
                dev = S.p4._block_digest(vt, pages.to(vt.device)).cpu().tolist()
                d["twosum_device"] = [int(x) for x in dev]
                d["twosum_host"] = _twosum_host(vb, vt.element_size())
            if sl.get("pools"):
                for q in sl["pools"]:
                    (v0, v1), (s0, s1) = LY.idx_pool(q)
                    pr = np.concatenate([raw[v0:v1], raw[s0:s1]])
                    d[f"pool{q}_sha256"] = _sha(pr)
                    d[f"pool{q}_zero"] = bool(not pr.any())
                    if not sl.get("hash_only"):
                        arrays[f"{sl['key']}_pool{q}"] = pr
            if not sl.get("hash_only"):
                arrays[sl["key"]] = raw
            out[sl["key"]] = d
        meta = dict(rec, slices=out, problems=problems, chunks=chunks)
        arrays["meta"] = np.array(json.dumps(meta, default=str))
        rec.update(ok=not problems and all(d["view_equal"] and d["layout_ok"] for d in out.values()),
                   slices=out, problems=problems, file_sha256=S.p4._write_private(f"cap0_{S.p4._safe(tag)}_r{S.rank}.npz", arrays),
                   secs=round(time.time() - t0, 3))
        S.n_capture += 1
        _log(rec)
    except Exception as exc:  # noqa: BLE001
        rec.update(ok=False, why=repr(exc)[:300])
        _log(rec)
        _err("capture", exc)


# ------------------------------------------------------------------ 2. localisation
def _new_acc(tag):
    return dict(tag=tag, steps=[], dig={}, pos={}, raw={}, diff={}, states=[], shapes={}, ndiff={}, errors=[])


def _arm(input_batch, step, uid, tag, n_comp, n_sched, nt):
    t0 = time.time()
    acc = S.loc.setdefault(uid, _new_acc(tag))
    if S.ref_uid is None:
        S.ref_uid = uid
    try:   # the step's model inputs (eager step, before the forward): engine-side identical-input check
        ids = input_batch.input_ids[:nt].cpu().numpy().astype(np.int32)
        pos = input_batch.positions[:nt].cpu().numpy().astype(np.int64)
        ids_sha, pos_sha = _sha(ids), _sha(pos)
        pos_ok = bool(pos[0] == n_comp and np.array_equal(pos, n_comp + np.arange(nt)))
    except Exception as exc:  # noqa: BLE001
        ids_sha = pos_sha = None
        pos_ok = False
        acc["errors"].append("ids: " + repr(exc)[:200])
    if not pos_ok:
        acc["errors"].append("positions of step %d are not %d..%d" % (step, n_comp, n_comp + nt - 1))
    S.arm = dict(uid=uid, tag=tag, step=step, pos0=n_comp, n=min(n_sched, LOC_TOKENS - n_comp), m=nt,
                 t0=t0, secs=0.0, ids_sha256=ids_sha, pos_sha256=pos_sha, points=[])


def _rows2d(t, m):
    """[m, -1] view of a [m, ...] or [1, m, ...] tensor (no copy for contiguous or 2-D inputs)."""
    if t.dim() >= 3 and int(t.shape[0]) == 1 and int(t.shape[1]) == m:
        t = t[0]
    if t.dim() >= 1 and int(t.shape[0]) == m:
        return t if t.dim() == 2 else t.reshape(m, -1)
    if t.dim() == 2 and int(t.shape[1]) == m:
        return t.t()
    return None


def _rec(point, named):
    """Hash the token rows of the named tensors at one boundary (armed eager steps only)."""
    a = S.arm
    if a is None or _capturing() or S.p4 is None or S.p4.S.mode != "none":
        return
    t0 = time.time()
    try:
        acc = S.loc[a["uid"]]
        a["points"].append(point)
        is_ref = a["uid"] == S.ref_uid
        for tname, t in named:
            if t is None:
                continue
            key = (point, tname)
            x = _rows2d(t.detach(), a["m"])
            if x is None:
                acc["errors"].append("%s.%s shape %s vs m %d" % (point, tname, list(t.shape), a["m"]))
                continue
            n = a["n"]
            rowbytes = int(x.shape[1]) * x.element_size()
            acc["shapes"][key] = dict(shape=list(t.shape), dtype=str(t.dtype), rowbytes=rowbytes)
            dig = np.empty((n, 32), dtype=np.uint8)
            per = max(1, ROWBLK // max(rowbytes, 1))
            hold = _hold(key, rowbytes) if is_ref else None
            for r0 in range(0, n, per):
                r1 = min(n, r0 + per)
                h = x[r0:r1].contiguous().view(torch.uint8).cpu().numpy().reshape(r1 - r0, rowbytes)
                for i in range(r1 - r0):
                    dig[r0 + i] = np.frombuffer(hashlib.sha256(h[i]).digest(), dtype=np.uint8)
                    pos = a["pos0"] + r0 + i
                    if pos < RAW_FIRST or pos in BOUNDARY:
                        acc["raw"].setdefault(key, {})[pos] = h[i].copy()
                if hold is not None:
                    hold[a["pos0"] + r0:a["pos0"] + r1] = h
            acc["dig"].setdefault(key, []).append(dig)
            acc["pos"].setdefault(key, []).append(np.arange(a["pos0"], a["pos0"] + n, dtype=np.int32))
            _vs_ref(acc, key, a["pos0"], dig, x, is_ref)
    except Exception as exc:  # noqa: BLE001
        _err("rec." + point, exc)
    a["secs"] += time.time() - t0


def _hold(key, rowbytes):
    if key in S.hold:
        return S.hold[key]
    need = LOC_TOKENS * rowbytes
    if S.hold_bytes + need > HOLD_MB * (1 << 20):
        S.hold_refused.append("%s.%s" % key)
        return None
    S.hold[key] = np.zeros((LOC_TOKENS, rowbytes), dtype=np.uint8)
    S.hold_bytes += need
    return S.hold[key]


def _vs_ref(acc, key, pos0, dig, x, is_ref):
    n = dig.shape[0]
    if is_ref:
        if key not in S.ref_dig:
            S.ref_dig[key] = np.zeros((LOC_TOKENS, 32), dtype=np.uint8)
            S.ref_have[key] = np.zeros(LOC_TOKENS, dtype=bool)
        S.ref_dig[key][pos0:pos0 + n] = dig
        S.ref_have[key][pos0:pos0 + n] = True
        return
    if key not in S.ref_dig:
        return
    have = S.ref_have[key][pos0:pos0 + n]
    neq = np.nonzero(have & np.any(S.ref_dig[key][pos0:pos0 + n] != dig, axis=1))[0]
    acc["ndiff"][key] = acc["ndiff"].get(key, 0) + int(neq.size)
    d = acc["diff"].setdefault(key, {})
    for i in neq[: max(0, RAW_DIFF - len(d))]:
        pos = pos0 + int(i)
        cur = x[int(i):int(i) + 1].contiguous().view(torch.uint8).cpu().numpy().reshape(-1)
        ref = S.hold[key][pos].copy() if key in S.hold else None
        d[pos] = (cur, ref)


def _rec_state(L, mod):
    """After layer L's KDA forward: the running column it wrote (non_spec_state_indices_tensor[0])."""
    a = S.arm
    if a is None or _capturing() or S.p4.S.mode != "none":
        return
    t0 = time.time()
    try:
        from vllm.forward_context import get_forward_context
        md = get_forward_context().attn_metadata[mod.prefix]
        col = int(md.non_spec_state_indices_tensor[0].item())
        parts = S.p4._kv_parts(getattr(mod, "kv_cache", None))
        conv, rec = parts[0][1], parts[1][1]
        rb = _host_u8(rec[col])
        cb = _host_u8(conv[col, :LY.CONV_LIVE_ROWS])
        end = a["pos0"] + a["n"]
        d = dict(point=f"L{L}.state_end", step=a["step"], pos_end=end, col=col, rec_sha256=_sha(rb),
                 conv_live_sha256=_sha(cb), full_step=a["n"] == a["m"])
        if L == 4 and end % LY.BLOCK_TOKENS == 0 and a["n"] == a["m"]:
            d["rec_raw"], d["conv_raw"] = rb, cb
        S.loc[a["uid"]]["states"].append(d)
    except Exception as exc:  # noqa: BLE001
        _err("rec_state", exc)
    a["secs"] += time.time() - t0


def install_loc(runner):
    """Forward hooks on layers 3 and 4 (and 2 for the state control) of the target model, and the conv wrapper.
    Installed before graph capture; every hook returns immediately unless an eager warm step is armed."""
    layers = {}
    for name, mod in runner.model.named_modules():
        if any(w in name.lower() for w in ("draft", "mtp", "dflash", "eagle", "spec")):
            continue
        if type(mod).__name__ == "Glm5NextDecoderLayer":
            L = int(getattr(mod, "layer_idx", -1))
            if L in (2, 3, 4):
                layers.setdefault(L, []).append(name)
                S.layers[L] = mod
    made = []
    if any(len(v) != 1 for v in layers.values()) or set(layers) != {2, 3, 4}:
        _log(dict(kind="cap0_hook_problem", layers=layers))
        return made

    def H(m, kind, fn, **kw):
        h = getattr(m, "register_forward_pre_hook" if kind == "pre" else "register_forward_hook")(fn, **kw)
        S.hooks.append(h)
        made.append("%s:%s" % (kind, type(m).__name__))

    l3, l4, l2 = S.layers[3], S.layers[4], S.layers[2]
    H(l3, "pre", lambda m, args: _rec("L3.in", list(zip(("x", "residual", "post", "comb"), args[1:5]))))
    H(l4, "pre", lambda m, args: _rec("L4.in", list(zip(("x", "residual", "post", "comb"), args[1:5]))))

    def fused_post(layer, tag):
        def f(m, args, kwargs, out):
            if S.arm is None:
                return
            fn = kwargs.get("fn")
            side = "attn" if fn is layer.hc_attn_fn else ("ffn" if fn is layer.hc_ffn_fn else None)
            if side is None or (tag == "L4" and side != "attn"):
                return
            _rec(f"{tag}.pre_{side}", list(zip(("residual", "post", "comb", "y"), out)))
        return f

    def fused_pre(layer):
        def f(m, args, kwargs):
            if S.arm is not None and kwargs.get("fn") is layer.hc_ffn_fn:
                _rec("L3.attn_out", [("x", kwargs.get("x"))])
        return f

    H(l3.mhc_fused_post_pre_op, "post", fused_post(l3, "L3"), with_kwargs=True)
    H(l3.mhc_fused_post_pre_op, "pre", fused_pre(l3), with_kwargs=True)
    H(l4.mhc_fused_post_pre_op, "post", fused_post(l4, "L4"), with_kwargs=True)
    H(l3.self_attn, "pre", lambda m, args, kwargs: _rec("L3.attn_in", [("x", kwargs.get("hidden_states",
                                                                                   args[0] if args else None))]),
      with_kwargs=True)
    idx = getattr(l3.self_attn, "indexer", None)
    if idx is not None:
        def topk(m, args, out):
            if S.arm is not None and torch.is_tensor(out) and args:
                _rec("L3.topk", [("ids", out[: int(args[0].shape[0])])])
        H(idx, "post", topk)
    H(l3.mlp, "post", lambda m, args, out: _rec("L3.moe_out", [("x", out)]))

    def l4_pre(m, args, kwargs):
        S.cur_layer = 4 if S.arm is not None else None

    def l4_post(m, args, kwargs, out):
        S.cur_layer = None
        if S.arm is not None:
            _rec("L4.attn_out", [("x", out)])
            _rec_state(4, m)
    H(l4.self_attn, "pre", l4_pre, with_kwargs=True)
    H(l4.self_attn, "post", l4_post, with_kwargs=True)
    H(l2.self_attn, "post", lambda m, args, out: _rec_state(2, m) if S.arm is not None else None)
    onorm = getattr(l4.self_attn, "o_norm", None)
    if onorm is not None:
        H(onorm, "pre", lambda m, args: _rec("L4.rec_out", [("o", args[0])]) if S.arm is not None else None)
    try:
        from vllm.models.glm5next.nvidia import kda as kmod
        patch_conv(kmod)
    except Exception as exc:  # noqa: BLE001
        _err("patch_conv", exc)
    return made


def patch_conv(kmod):
    """Wrap kmod.causal_conv1d_fn (looked up at call time by kda._forward): record the output while layer 4's
    KDA forward runs in an armed step; otherwise a pass-through."""
    if S.conv_patched:
        return
    orig = kmod.causal_conv1d_fn

    @functools.wraps(orig)
    def causal_conv1d_fn(*args, **kwargs):
        out = orig(*args, **kwargs)
        if S.arm is not None and S.cur_layer == 4:
            _rec("L4.conv_out", [("qkv", out)])
        return out
    causal_conv1d_fn._cap0_hooks_orig = orig
    kmod.causal_conv1d_fn = causal_conv1d_fn
    S.conv_patched = True


# ------------------------------------------------------------------ removal (p4_hooks.on_remove; never in capture)
def on_remove(req_idx, tag, uid):
    acc = S.loc.pop(uid, None)
    if acc is None:
        return
    try:
        arrays, summ = {}, {}
        for key, digs in acc["dig"].items():
            nm = "%s.%s" % key
            arrays["dig:" + nm] = np.concatenate(digs)
            arrays["pos:" + nm] = np.concatenate(acc["pos"][key])
            raw = acc["raw"].get(key, {})
            if raw:
                ps = sorted(raw)
                arrays["rawpos:" + nm] = np.array(ps, dtype=np.int32)
                arrays["raw:" + nm] = np.stack([raw[p] for p in ps])
            dd = acc["diff"].get(key, {})
            if dd:
                ps = sorted(dd)
                arrays["diffpos:" + nm] = np.array(ps, dtype=np.int32)
                arrays["diffcur:" + nm] = np.stack([dd[p][0] for p in ps])
                if all(dd[p][1] is not None for p in ps):
                    arrays["diffref:" + nm] = np.stack([dd[p][1] for p in ps])
            summ[nm] = dict(rows=int(arrays["dig:" + nm].shape[0]), ndiff_vs_ref=acc["ndiff"].get(key),
                            **acc["shapes"].get(key, {}))
        st = []
        for k, d in enumerate(acc["states"]):
            if "rec_raw" in d:
                arrays[f"state{k}_rec"] = d.pop("rec_raw")
                arrays[f"state{k}_conv"] = d.pop("conv_raw")
                d["raw_index"] = k
            st.append(d)
        meta = dict(tag=tag, uid=uid, rank=S.rank, ref_uid=S.ref_uid, is_ref=uid == S.ref_uid, steps=acc["steps"],
                    states=st, points=summ, errors=acc["errors"], hold_mb=round(S.hold_bytes / 2 ** 20, 1),
                    hold_refused=S.hold_refused, loc_tokens=LOC_TOKENS, raw_first=RAW_FIRST, boundary=BOUNDARY)
        arrays["meta"] = np.array(json.dumps(meta, default=str))
        sha = S.p4._write_private(f"loc_{S.p4._safe(tag)}_r{S.rank}.npz", arrays)
        _log(dict(kind="cap0_loc_flushed", tag=tag, uid=uid, rank=S.rank, is_ref=uid == S.ref_uid, steps=acc["steps"],
                  n_points=len(summ), points=summ, states=st, errors=acc["errors"][:20], file_sha256=sha,
                  hold_mb=meta["hold_mb"], hold_refused=S.hold_refused))
    except Exception as exc:  # noqa: BLE001
        _err("on_remove", exc)
