# SPDX-License-Identifier: Apache-2.0
"""tailguard: observe-only guard for the GLM5Next kpool tail ring.

Inert unless GLM53_TAILGUARD is set to "eager" or "device".  Installed as
``vllm/kpool_tailguard.py`` in a diagnostic image; imported by the patched
``sparse_attn_indexer_kpool.py``.  It never writes to an engine tensor: every
store goes to guard-owned buffers; engine tensors are only read.

What it checks (per step and rank):
  * at every tail SEED (prefill) and STASH (decode/verify) call, the tail block
    each writing token actually uses (``tail_slot // ring``, with the kernels'
    own write predicates) against
      - the row's own tail block (gathered tail-group block table, column 0),
      - 0 (the reserved null block: "null alias"),
      - the step's attention (MLA/indexer group) block ids of every row,
      - every other row's own tail block;
  * before the forward: the tail group's persistent block-table rows - any
    non-zero column >= 1 (the generic slot mapping reads column pos // ring for
    every token, so a non-zero stale column is a pending foreign write), and a
    CPU shadow of the staged writes vs the device table (GPU-side divergence);
  * the CPU-staged tail-group writes (any write with start != 0 or length != 1
    is an anomaly: KpoolTailManager allocates exactly one block per request).

Modes:
  eager  : host-side check (device->host copies, only when not capturing),
           detailed JSONL per violation plus a per-step digest.  Use with
           cudagraph_mode NONE.
  device : graph-capture safe; fixed-shape device ops only (no host sync):
           counters + first-N records in device buffers, polled to JSONL
           outside any captured region by a background thread (guard-private
           side stream, pinned buffers), on a DUMP trigger file and at exit.

Env:
  GLM53_TAILGUARD             "" (off) | "eager" | "device"
  GLM53_TAILGUARD_DIR         output dir (default /tmp/tailguard)
  GLM53_TAILGUARD_CAP         record capacity (default 4096)
  GLM53_TAILGUARD_POLL_SECS   seconds between background polls (default 30; a poll
                              is also taken whenever <dir>/DUMP is touched, and at exit)
  GLM53_TAILGUARD_MAX_BLOCKS  size of block-id maps (default 65536)
  GLM53_TAILGUARD_STEPLOG     1 = per-step digest line (eager default 1, device 0)
"""
from __future__ import annotations

import atexit
import json
import os
import time

import numpy as np
import torch

MODE = os.environ.get("GLM53_TAILGUARD", "").strip().lower()
ENABLED = MODE in ("eager", "device")
OUT_DIR = os.environ.get("GLM53_TAILGUARD_DIR", "/tmp/tailguard")
CAP = int(os.environ.get("GLM53_TAILGUARD_CAP", "4096"))
POLL_SECS = float(os.environ.get("GLM53_TAILGUARD_POLL_SECS", "30"))
MAX_BLOCKS = int(os.environ.get("GLM53_TAILGUARD_MAX_BLOCKS", "65536"))
STEPLOG = os.environ.get("GLM53_TAILGUARD_STEPLOG", "1" if MODE == "eager" else "0") == "1"
SHADOW_COLS = 4

# record kinds
K_SEED, K_STASH, K_PREFWD, K_STALE, K_DIVERGE = 1, 2, 3, 4, 5
KIND_NAME = {K_SEED: "seed", K_STASH: "stash", K_PREFWD: "prefwd_slot",
             K_STALE: "stale_col", K_DIVERGE: "shadow_divergence"}
# flags
F_NULL, F_FOREIGN, F_ATTN, F_TAILCOLL = 1, 2, 4, 8
# counters
C_WRITES, C_NULL, C_FOREIGN, C_ATTN, C_TAILCOLL, C_REC, C_STALE, C_DIVERGE, \
    C_PREFWD_TOK, C_PREFWD_DEV, C_PREFWD_NULL, C_PREFWD_FOREIGN, C_STEPS = range(13)
COUNTER_NAMES = ["writes", "null_alias", "foreign", "attn_collision", "tail_collision",
                 "records", "stale_col_rows", "shadow_divergence", "prefwd_tokens",
                 "prefwd_deviate", "prefwd_null", "prefwd_foreign", "steps"]
REC_COLS = ["kind", "rank", "layer", "step", "token", "row", "req_idx", "position",
            "used_block", "own_block", "flags", "aux"]
NREC = len(REC_COLS)


class _State:
    def __init__(self):
        self.bt = None              # BlockTables instance
        self.device = None
        self.tail_gid = None
        self.mla_gid = None
        self.rank = None
        self.step = 0
        self.ctx = {}               # per-step tensors (views of persistent buffers)
        self.shadow = None          # list[np.ndarray (max_reqs, SHADOW_COLS)]
        self.step_tail_writes = []  # CPU-staged tail writes this step
        self.dev = None             # dict of guard-owned device buffers
        self.fh = None
        self.logged_once = set()
        self.errors = 0
        self.poller = None
        self.side = None
        self.pin_c = None
        self.pin_r = None
        self.last_key = None


S = _State()


# --------------------------------------------------------------------- utils
def _capturing() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False


def _rank() -> int:
    if S.rank is None:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            S.rank = int(get_tensor_model_parallel_rank())  # cached only once resolvable
        except Exception:  # noqa: BLE001
            return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1")))
    return S.rank


def _log(rec: dict) -> None:
    try:
        if S.fh is None:
            os.makedirs(OUT_DIR, exist_ok=True)
            S.fh = open(os.path.join(OUT_DIR, f"tg_p{os.getpid()}.jsonl"), "a")
        rec.setdefault("t", round(time.time(), 4))
        rec.setdefault("rank", _rank())
        rec.setdefault("step", S.step)
        S.fh.write(json.dumps(rec) + "\n")
        S.fh.flush()
    except Exception:  # noqa: BLE001
        S.errors += 1


def _once(key: str, rec: dict) -> None:
    if key not in S.logged_once:
        S.logged_once.add(key)
        _log(rec)


def _layer_idx(name) -> int:
    try:
        parts = str(name).split(".")
        return int(parts[parts.index("layers") + 1])
    except Exception:  # noqa: BLE001
        return -1


def alloc_device_buffers(device, max_reqs: int) -> dict:
    d = dict(
        counters=torch.zeros(16, dtype=torch.int64, device=device),
        records=torch.zeros(CAP + 1, NREC, dtype=torch.int64, device=device),  # row CAP = dump
        att_owner=torch.zeros(MAX_BLOCKS + 1, dtype=torch.int32, device=device),  # row+1, 0 = none
        tail_owner=torch.zeros(MAX_BLOCKS + 1, dtype=torch.int32, device=device),
        stale_seen=torch.zeros(max_reqs, dtype=torch.bool, device=device),
        step=torch.zeros(1, dtype=torch.int64, device=device),
    )
    return d


# ------------------------------------------------------------ core (pure torch)
def write_mask_and_blocks(kind: str, tslot: torch.Tensor, pos: torch.Tensor | None,
                          ring: int, kpool: int):
    """Tokens that the real kernel writes, and the tail block / slot used.

    seed  (_kpool_tail_seed_kernel): t >= 0 and (ahead < 0 or ahead//ring != t//ring)
          with ahead = tslot[i + kpool] inside the same call (else -1); slot = t % ring.
    stash (_kpool_decode_update_batched_kernel): pos >= 0 and tail_slot >= 0;
          block = max(tail_slot, 0) // ring; slot = pos % ring.
    """
    t = tslot.to(torch.int64)
    n = t.shape[0]
    tpos = t.clamp(min=0)
    blk = torch.div(tpos, ring, rounding_mode="floor")
    if kind == "seed":
        ahead = torch.full_like(t, -1)
        if n > kpool:
            ahead[: n - kpool] = t[kpool:]
        ablk = torch.div(ahead.clamp(min=0), ring, rounding_mode="floor")
        w = (t >= 0) & ((ahead < 0) | (ablk != blk))
        slot = torch.remainder(tpos, ring)
    else:
        assert pos is not None
        p = pos.to(torch.int64)
        w = (p >= 0) & (t >= 0)
        slot = torch.remainder(p.clamp(min=0), ring)
    return w, blk, slot


def classify(w, blk, row, own, att_owner, tail_owner):
    """Per-token flags (int64) for written tokens; 0 for clean or non-writing."""
    nb = att_owner.shape[0] - 1
    bidx = blk.clamp(max=nb)
    dev = w & (blk != own.to(torch.int64))
    null = dev & (blk == 0)
    foreign = dev & (blk != 0)
    attn = w & (blk != 0) & (att_owner[bidx] != 0)
    towner = tail_owner[bidx].to(torch.int64)
    tcoll = w & (blk != 0) & (towner != 0) & (towner != row + 1)
    flags = (null.to(torch.int64) * F_NULL + foreign.to(torch.int64) * F_FOREIGN
             + attn.to(torch.int64) * F_ATTN + tcoll.to(torch.int64) * F_TAILCOLL)
    return flags, dev, null, foreign, attn, tcoll


def record_rows(dev: dict, mask: torch.Tensor, rows: torch.Tensor) -> None:
    """Fixed-shape append of rows[mask] into dev['records'] (no host sync).

    Overflowing rows go to the dump row CAP; dev['counters'][C_REC] counts all.
    """
    m = mask.to(torch.int64)
    base = dev["counters"][C_REC]
    idx = base + torch.cumsum(m, 0) - 1
    ok = mask & (idx < CAP)
    tgt = torch.where(ok, idx, torch.full_like(idx, CAP))
    dev["records"].index_copy_(0, tgt, rows)
    dev["counters"][C_REC] += m.sum()


def token_rows(qsl: torch.Tensor, n_reqs: int, tok: torch.Tensor) -> torch.Tensor:
    q = qsl[: n_reqs + 1].to(torch.int64)
    r = torch.searchsorted(q, tok, right=True) - 1
    return r.clamp(min=0, max=max(n_reqs - 1, 0))


def build_att_owner(dev: dict, mla_bt: torch.Tensor, seq_lens: torch.Tensor, n_reqs: int,
                    kernel_block_size: int, bpk: int) -> None:
    """att_owner[kv_block] = row+1 for every MLA-group block a scheduled row holds
    (columns < cdiv(seq_len, kernel_block_size); kernel ids // bpk).  Fixed shapes."""
    nb = dev["att_owner"].shape[0] - 1
    dev["att_owner"].zero_()
    if n_reqs > 0:
        bt = mla_bt[:n_reqs].to(torch.int64)
        W = bt.shape[1]
        ncols = torch.div(seq_lens[:n_reqs].to(torch.int64) + kernel_block_size - 1,
                          kernel_block_size, rounding_mode="floor")
        col = torch.arange(W, device=bt.device)[None, :]
        valid = (col < ncols[:, None]) & (bt > 0)
        kv = torch.div(bt, bpk, rounding_mode="floor").clamp(max=nb)
        kv = torch.where(valid, kv, torch.full_like(kv, nb))
        rowid = (torch.arange(n_reqs, device=bt.device, dtype=torch.int32)[:, None] + 1).expand_as(kv)
        dev["att_owner"].index_put_((kv.reshape(-1),), rowid.reshape(-1))
        dev["att_owner"][nb] = 0


def build_tail_owner(dev: dict, tail_bt: torch.Tensor, tail_n_reqs: int) -> None:
    """tail_owner[kv_block] = row+1 for every row's own tail block (column 0)."""
    nb = dev["tail_owner"].shape[0] - 1
    dev["tail_owner"].zero_()
    if tail_n_reqs > 0:
        t0 = tail_bt[:tail_n_reqs, 0].to(torch.int64).clamp(min=0, max=nb)
        t0 = torch.where(t0 > 0, t0, torch.full_like(t0, nb))
        dev["tail_owner"].index_put_(
            (t0,), torch.arange(1, tail_n_reqs + 1, device=t0.device, dtype=torch.int32))
        dev["tail_owner"][nb] = 0


# ------------------------------------------------------------ in-op check
def check_call(kind: str, layer_name, tail_kv_cache: torch.Tensor, tslot: torch.Tensor,
               pos: torch.Tensor | None, token_offset: int, kpool: int) -> None:
    """Called at the tail seed (kind='seed') and stash (kind='stash') call sites,
    before the kernel launch.  Observe-only."""
    if not ENABLED:
        return
    try:
        _check_call(kind, layer_name, tail_kv_cache, tslot, pos, token_offset, kpool)
    except Exception as exc:  # noqa: BLE001
        S.errors += 1
        if S.errors <= 20:
            _log(dict(kind="guard_error", where=kind, err=repr(exc)))


def _check_call(kind, layer_name, tail_kv_cache, tslot, pos, token_offset, kpool):
    c = S.ctx
    if "tail_bt" not in c or "qsl" not in c:
        _once("noctx", dict(kind="guard_warn", msg="no tail metadata context registered"))
        return
    ring = int(tail_kv_cache.shape[2])
    n = int(tslot.shape[0])
    if n == 0:
        return
    if pos is not None:
        pos = pos[:n]
    w, blk, slot = write_mask_and_blocks(kind, tslot, pos, ring, kpool)
    tok = torch.arange(n, device=tslot.device, dtype=torch.int64) + int(token_offset)
    n_reqs = int(c["n_reqs"])
    row = token_rows(c["qsl"], n_reqs, tok)
    own = c["tail_bt"][:, 0].to(torch.int64)[row]
    if S.dev is None:
        _once("nodev", dict(kind="guard_warn", msg="no device buffers"))
        return
    dev = S.dev
    flags, devm, null, foreign, attn, tcoll = classify(
        w, blk, row, own, dev["att_owner"], dev["tail_owner"])
    kcode = K_SEED if kind == "seed" else K_STASH
    layer = _layer_idx(layer_name)

    if MODE == "eager":
        if _capturing():
            return
        summ = torch.stack([w.sum(), null.sum(), foreign.sum(), attn.sum(), tcoll.sum()])
        cnt = dev["counters"]
        cnt[C_WRITES:C_TAILCOLL + 1] += summ
        nw, nnull, nfor, natt, ntc = summ.cpu().tolist()  # one host sync per call
        if nfor or natt or ntc:
            w_c, blk_c, flags_c = w.cpu(), blk.cpu(), flags.cpu()
            bad = (flags_c & (F_FOREIGN | F_ATTN | F_TAILCOLL)).ne(0)
            row_c = row.cpu()
            pos_c = pos.cpu() if pos is not None else None
            qsl_c = c["qsl"][: n_reqs + 1].cpu()
            idxm = c.get("idx_mapping")
            idxm_c = idxm.cpu() if idxm is not None else None
            own_c = own.cpu()
            att_c = dev["att_owner"][blk.clamp(max=MAX_BLOCKS)].cpu()
            persist = None
            if S.bt is not None and S.tail_gid is not None and idxm_c is not None:
                persist = S.bt.block_tables[S.tail_gid].gpu[idxm_c.long(), :SHADOW_COLS].cpu()
            ev = []
            for i in torch.nonzero(bad).flatten().tolist()[:64]:
                r = int(row_c[i])
                ev.append(dict(
                    token=int(tok[i]), row=r,
                    req_idx=None if idxm_c is None else int(idxm_c[r]),
                    pos=None if pos_c is None else int(pos_c[i]),
                    tslot=int(tslot[i]), used_block=int(blk_c[i]), slot=int(slot[i]),
                    own_block=int(own_c[i]), flags=int(flags_c[i]),
                    att_owner_row=int(att_c[i]) - 1,
                    chunk_len=int(qsl_c[r + 1] - qsl_c[r]),
                    persist_cols=None if persist is None else persist[r].tolist()))
            _log(dict(kind="violation", call=kind, layer=layer, n_tokens=n, n_writes=nw,
                      n_null=nnull, n_bad=int(bad.sum()), events=ev))
        return

    # device mode: fixed shapes, no host sync (capture safe)
    cnt = dev["counters"]
    cnt[C_WRITES] += w.sum()
    cnt[C_NULL] += null.sum()
    cnt[C_FOREIGN] += foreign.sum()
    cnt[C_ATTN] += attn.sum()
    cnt[C_TAILCOLL] += tcoll.sum()
    bad = (flags & (F_FOREIGN | F_ATTN | F_TAILCOLL)) != 0
    # idx_mapping is not a capture-stable buffer: req_idx is resolved post hoc
    # from the per-step log (step, row) in device mode.
    req_idx = torch.full_like(row, -1)
    p = pos.to(torch.int64) if pos is not None else torch.full_like(row, -1)
    aux = dev["att_owner"][blk.clamp(max=MAX_BLOCKS)].to(torch.int64) - 1
    rows = torch.stack([
        torch.full_like(row, kcode), torch.full_like(row, _rank()), torch.full_like(row, layer),
        dev["step"].expand_as(row), tok, row, req_idx, p, blk, own, flags, aux], dim=1)
    record_rows(dev, bad, rows)


# ------------------------------------------------------------ pre-forward hooks
def _prefwd_checks(bt) -> None:
    """After apply_staged_writes + gather: persistent tail table vs shadow, and
    non-zero stale columns.  Device ops only (no sync); not during capture."""
    if S.dev is None or S.tail_gid is None or _capturing():
        return
    dev = S.dev
    g = S.tail_gid
    persist = bt.block_tables[g].gpu
    R = persist.shape[0]
    # stale non-zero columns >= 1 (first appearance per row)
    nz = persist[:, 1:] != 0
    has = nz.any(dim=1)
    firstcol = torch.argmax(nz.to(torch.int8), dim=1).to(torch.int64) + 1
    val = persist.gather(1, firstcol[:, None]).squeeze(1).to(torch.int64)
    new = has & ~dev["stale_seen"]
    dev["stale_seen"].copy_(has)
    dev["counters"][C_STALE] += new.sum()
    ar = torch.arange(R, device=persist.device, dtype=torch.int64)
    rows = torch.stack([
        torch.full_like(ar, K_STALE), torch.full_like(ar, _rank()), torch.full_like(ar, -1),
        dev["step"].expand_as(ar), firstcol, torch.full_like(ar, -1), ar,
        torch.full_like(ar, -1), val, persist[:, 0].to(torch.int64),
        nz.sum(dim=1).to(torch.int64), torch.full_like(ar, -1)], dim=1)
    record_rows(dev, new, rows)
    # CPU shadow vs device (columns 0..SHADOW_COLS-1)
    if S.shadow is not None:
        sh_cpu = torch.from_numpy(S.shadow[g].copy())
        if persist.is_cuda:
            sh_cpu = sh_cpu.pin_memory()
        sh = sh_cpu.to(persist.device, non_blocking=True)
        cur = persist[:, :SHADOW_COLS].to(torch.int64)
        diff = cur != sh
        anyd = diff.any(dim=1)
        dcol = torch.argmax(diff.to(torch.int8), dim=1).to(torch.int64)
        dev["counters"][C_DIVERGE] += anyd.sum()
        rows = torch.stack([
            torch.full_like(ar, K_DIVERGE), torch.full_like(ar, _rank()), torch.full_like(ar, -1),
            dev["step"].expand_as(ar), dcol, torch.full_like(ar, -1), ar,
            torch.full_like(ar, -1), cur.gather(1, dcol[:, None]).squeeze(1),
            sh.gather(1, dcol[:, None]).squeeze(1), torch.full_like(ar, 0),
            torch.full_like(ar, -1)], dim=1)
        record_rows(dev, anyd, rows)


def _prefwd_slot_check(slot_mappings: torch.Tensor, qsl: torch.Tensor, positions: torch.Tensor,
                       idx_mapping: torch.Tensor) -> None:
    """Generic tail slot mapping vs the intended own_block*ring + pos%ring, all tokens."""
    if S.dev is None or S.tail_gid is None or S.bt is None or _capturing():
        return
    dev = S.dev
    g = S.tail_gid
    ring = int(S.bt.kernel_block_sizes[g])
    n_reqs = int(idx_mapping.shape[0])
    if n_reqs == 0:
        return
    sm = slot_mappings[g]
    n = min(int(sm.shape[0]), int(positions.shape[0]))
    sm = sm[:n].to(torch.int64)
    tok = torch.arange(n, device=sm.device, dtype=torch.int64)
    row = token_rows(qsl, n_reqs, tok)
    own = S.bt.block_tables[g].gpu[idx_mapping.to(torch.int64), 0].to(torch.int64)[row]
    p = positions[:n].to(torch.int64)
    valid = (sm >= 0) & (tok < qsl[n_reqs].to(torch.int64))
    intended = own * ring + torch.remainder(p.clamp(min=0), ring)
    blk = torch.div(sm.clamp(min=0), ring, rounding_mode="floor")
    devm = valid & (sm != intended)
    null = devm & (blk == 0)
    foreign = devm & (blk != 0) & (blk != own)
    cnt = dev["counters"]
    cnt[C_PREFWD_TOK] += valid.sum()
    cnt[C_PREFWD_DEV] += devm.sum()
    cnt[C_PREFWD_NULL] += null.sum()
    cnt[C_PREFWD_FOREIGN] += foreign.sum()
    req_idx = idx_mapping.to(torch.int64)[row]
    rows = torch.stack([
        torch.full_like(row, K_PREFWD), torch.full_like(row, _rank()), torch.full_like(row, -1),
        dev["step"].expand_as(row), tok, row, req_idx, p, blk, own,
        foreign.to(torch.int64) * F_FOREIGN, torch.full_like(row, -1)], dim=1)
    record_rows(dev, foreign, rows)


def _snapshot():
    """Copy counters + records to host.  On CUDA: a guard-private side stream,
    non_blocking into pinned memory, then wait on that stream's event only (no
    sync of the engine's stream; reads may be one step stale)."""
    dev = S.dev
    if dev["counters"].is_cuda:
        if S.side is None:
            S.side = torch.cuda.Stream(device=dev["counters"].device)
            S.pin_c = torch.empty(dev["counters"].shape, dtype=torch.int64, pin_memory=True)
            S.pin_r = torch.empty(dev["records"].shape, dtype=torch.int64, pin_memory=True)
        with torch.cuda.stream(S.side):
            S.pin_c.copy_(dev["counters"], non_blocking=True)
            S.pin_r.copy_(dev["records"], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(S.side)
        ev.synchronize()
        return S.pin_c.clone(), S.pin_r.clone()
    return dev["counters"].clone(), dev["records"].clone()


def poll(reason: str, only_if_changed: bool = False) -> None:
    """Host readout of the guard's device buffers.  Never touches engine tensors."""
    if S.dev is None:
        return
    try:
        c_t, r_t = _snapshot()
        cnt = c_t.tolist()
        key = tuple(cnt[:C_STEPS])
        if only_if_changed and key == S.last_key:
            return
        S.last_key = key
        nrec = min(int(cnt[C_REC]), CAP)
        recs = r_t[:nrec].tolist()
        _log(dict(kind="poll", reason=reason, mode=MODE,
                  counters={k: cnt[i] for i, k in enumerate(COUNTER_NAMES)},
                  n_records=int(cnt[C_REC]),
                  records=[dict(zip(REC_COLS, r), kind_name=KIND_NAME.get(r[0], "?")) for r in recs],
                  guard_errors=S.errors))
    except Exception as exc:  # noqa: BLE001
        S.errors += 1
        _log(dict(kind="guard_error", where="poll", err=repr(exc)))


def _poller_loop():
    try:
        if S.dev is not None and S.dev["counters"].is_cuda:
            torch.cuda.set_device(S.dev["counters"].device)
    except Exception:  # noqa: BLE001
        pass
    trig = os.path.join(OUT_DIR, "DUMP")
    last_trig = None
    last = time.time()
    while True:
        time.sleep(2.0)
        try:
            t = os.path.getmtime(trig) if os.path.exists(trig) else None
        except OSError:
            t = None
        if t is not None and t != last_trig:
            last_trig = t
            poll("trigger")
            last = time.time()
        elif time.time() - last >= POLL_SECS:
            poll("periodic", only_if_changed=True)
            last = time.time()


def _ensure_poller() -> None:
    """Start the poll thread after the first real step (i.e. after any graph
    capture has finished)."""
    if S.poller is None and S.dev is not None:
        import threading
        S.poller = threading.Thread(target=_poller_loop, name="tailguard-poll",
                                    daemon=True)
        S.poller.start()


# ------------------------------------------------------------ patches
def _shadow_apply(bt) -> None:
    """Mirror the CPU-staged writes into the shadow (cols < SHADOW_COLS) and log
    tail-group anomalies.  Runs before the original apply_staged_writes."""
    if S.shadow is None:
        S.shadow = [np.zeros((t.num_rows, SHADOW_COLS), dtype=np.int64) for t in bt.block_tables]
    S.step_tail_writes = []
    for g, t in enumerate(bt.block_tables):
        cu_prev = 0
        for k, (r, st) in enumerate(zip(t._staged_write_indices, t._staged_write_starts)):
            cu = t._staged_write_cu_lens[k]
            cont = t._staged_write_contents[cu_prev:cu]
            cu_prev = cu
            for j, v in enumerate(cont):
                if st + j < SHADOW_COLS:
                    S.shadow[g][r, st + j] = int(v)
            if g == S.tail_gid:
                S.step_tail_writes.append((int(r), int(st), [int(x) for x in cont[:4]], len(cont)))
                if st != 0 or len(cont) != 1:
                    _log(dict(kind="staged_tail_anomaly", row=int(r), start=int(st),
                              length=len(cont), contents=[int(x) for x in cont[:8]]))


def install() -> None:
    from vllm.v1.attention.backends.mla import indexer as idxmod
    from vllm.v1.worker.gpu import block_table as btmod

    BT = btmod.BlockTables
    if getattr(BT, "_kpool_tailguard", False):
        return
    BT._kpool_tailguard = True
    orig_init = BT.__init__
    orig_apply = BT.apply_staged_writes
    orig_gather = BT.gather_block_tables
    orig_slots = BT.compute_slot_mappings

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        S.bt = self
        S.device = self.device
        S.dev = alloc_device_buffers(self.device, self.max_num_reqs)
        S.shadow = None
        S.mla_gid = None
        S.ctx = {}
        small = [g for g, bs in enumerate(self.block_sizes) if bs < 64]
        S.tail_gid = small[0] if len(small) == 1 else None
        _log(dict(kind="init", mode=MODE, block_sizes=list(self.block_sizes),
                  kernel_block_sizes=list(self.kernel_block_sizes),
                  blocks_per_kv_block=list(self.blocks_per_kv_block),
                  widths=[int(t.gpu.shape[1]) for t in self.block_tables],
                  max_num_reqs=int(self.max_num_reqs), tail_gid_guess=S.tail_gid,
                  cap=CAP, max_blocks=MAX_BLOCKS))

    def apply_staged_writes(self):
        if S.bt is self:
            try:
                _shadow_apply(self)
            except Exception as exc:  # noqa: BLE001
                S.errors += 1
                _log(dict(kind="guard_error", where="shadow", err=repr(exc)))
        return orig_apply(self)

    def gather_block_tables(self, idx_mapping, num_reqs_padded, out=None, out_ptrs=None):
        res = orig_gather(self, idx_mapping, num_reqs_padded, out=out, out_ptrs=out_ptrs)
        if S.bt is self and out is None:
            try:
                S.step += 1
                S.ctx["idx_mapping"] = idx_mapping
                S.ctx["n_reqs_sched"] = int(idx_mapping.shape[0])
                if S.dev is not None and not _capturing():
                    S.dev["step"] += 1
                    S.dev["counters"][C_STEPS] += 1
                _prefwd_checks(self)
                if STEPLOG and not _capturing():
                    d = dict(kind="step", n_reqs=int(idx_mapping.shape[0]),
                             tail_writes=S.step_tail_writes)
                    if MODE == "eager":
                        d["idx_mapping"] = idx_mapping.cpu().tolist()
                        if S.tail_gid is not None:
                            d["own_tail"] = self.block_tables[S.tail_gid].gpu[
                                idx_mapping.long(), 0].cpu().tolist()
                    _log(d)
                elif S.step_tail_writes:
                    _log(dict(kind="step_tail_writes", tail_writes=S.step_tail_writes))
                _ensure_poller()
            except Exception as exc:  # noqa: BLE001
                S.errors += 1
                if S.errors <= 20:
                    _log(dict(kind="guard_error", where="gather", err=repr(exc)))
        return res

    def compute_slot_mappings(self, idx_mapping, query_start_loc, positions,
                              num_tokens_padded, out=None):
        res = orig_slots(self, idx_mapping, query_start_loc, positions,
                         num_tokens_padded, out=out)
        if S.bt is self and out is None:
            try:
                _prefwd_slot_check(res, query_start_loc, positions, idx_mapping)
            except Exception as exc:  # noqa: BLE001
                S.errors += 1
                if S.errors <= 20:
                    _log(dict(kind="guard_error", where="prefwd_slot", err=repr(exc)))
        return res

    BT.__init__ = __init__
    BT.apply_staged_writes = apply_staged_writes
    BT.gather_block_tables = gather_block_tables
    BT.compute_slot_mappings = compute_slot_mappings

    TB = idxmod.KpoolTailMetadataBuilder
    orig_tail_build = TB.build

    def tail_build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = orig_tail_build(self, common_prefix_len, common_attn_metadata, fast_build)
        try:
            cm = common_attn_metadata
            S.ctx["tail_bt"] = cm.block_table_tensor
            S.ctx["qsl"] = cm.query_start_loc
            S.ctx["n_reqs"] = int(cm.num_reqs)
            if S.bt is not None and S.tail_gid is None:
                for g, t in enumerate(S.bt.input_block_tables):
                    if t.data_ptr() == cm.block_table_tensor.data_ptr():
                        S.tail_gid = g
            if S.dev is not None and not _capturing():
                build_tail_owner(S.dev, cm.block_table_tensor, int(cm.num_reqs))
            src = ("generic-persistent (positions=None)" if cm.positions is None
                   else "compute_kpool_tail_slot_mapping")
            _once("tailsrc", dict(kind="tail_slot_source", source=src,
                                  same_storage_as_input=bool(
                                      md.slot_mapping.data_ptr() == cm.slot_mapping.data_ptr()),
                                  tail_gid=S.tail_gid))
        except Exception as exc:  # noqa: BLE001
            S.errors += 1
            _log(dict(kind="guard_error", where="tail_build", err=repr(exc)))
        return md

    TB.build = tail_build

    # Row (request-state index) history: which request occupies which runner row,
    # so stale columns and foreign ids can be attributed to a previous occupant.
    from vllm.v1.worker.gpu import states as stmod
    RS = stmod.RequestState
    orig_add, orig_remove = RS.add_request, RS.remove_request

    def add_request(self, req_id, prompt_len, all_token_ids, num_computed_tokens, max_tokens):
        orig_add(self, req_id, prompt_len, all_token_ids, num_computed_tokens, max_tokens)
        try:
            _log(dict(kind="row_add", req_id=str(req_id), row=int(self.req_id_to_index[req_id]),
                      prompt_len=int(prompt_len), prefill_len=len(all_token_ids),
                      num_computed=int(num_computed_tokens)))
        except Exception:  # noqa: BLE001
            S.errors += 1

    def remove_request(self, req_id):
        r = orig_remove(self, req_id)
        if r is not None:
            _log(dict(kind="row_free", req_id=str(req_id), row=int(r)))
        return r

    RS.add_request = add_request
    RS.remove_request = remove_request

    IB = idxmod.DeepseekV32IndexerMetadataBuilder
    orig_idx_build = IB.build

    def idx_build(self, *a, **kw):
        md = orig_idx_build(self, *a, **kw)
        try:
            cm = kw.get("common_attn_metadata", a[1] if len(a) > 1 else None)
            if cm is not None and S.bt is not None and S.dev is not None:
                if S.mla_gid is None:
                    for g, t in enumerate(S.bt.input_block_tables):
                        if t.data_ptr() == cm.block_table_tensor.data_ptr():
                            S.mla_gid = g
                    _once("mlagid", dict(kind="mla_gid", mla_gid=S.mla_gid))
                if S.mla_gid is not None and not _capturing():
                    g = S.mla_gid
                    build_att_owner(
                        S.dev, cm.block_table_tensor, cm.seq_lens, int(cm.num_reqs),
                        int(S.bt.kernel_block_sizes[g]), int(S.bt.blocks_per_kv_block[g]))
        except Exception as exc:  # noqa: BLE001
            S.errors += 1
            if S.errors <= 20:
                _log(dict(kind="guard_error", where="idx_build", err=repr(exc)))
        return md

    IB.build = idx_build
    atexit.register(lambda: poll("exit"))
    _log(dict(kind="installed", mode=MODE, pid=os.getpid()))


if ENABLED:
    try:
        install()
    except Exception as _exc:  # noqa: BLE001
        S.errors += 1
        _log(dict(kind="guard_error", where="install", err=repr(_exc)))
