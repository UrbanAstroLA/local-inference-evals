# SPDX-License-Identifier: Apache-2.0
"""local-p3b: observe-only, CUDA-graph-safe capture of served MoE and sparse-MLA inputs/outputs.

Captured-input follow-up to P3 Stage 1 (investigation 2026-10-glm53-completion, package captured-p3b/).
Inert unless GLM53_P3B=capture.  Installed as ``vllm/p3b_capture_hooks.py`` in a derived image; called from
marked additive hook lines in ``quantization/exl3.py`` (MoE) and ``attention/backends/mla/b12x_mla_sparse.py``
(sparse MLA), and it wraps three GPUModelRunner methods (initialize_kv_cache, execute_model, prepare_attn) and the
two CUDA-graph run methods (mode label only), as the 22ao mapguard did.

What it records (rank 0 only; TP rank 1 stays inert):
  * MoE, layers GLM53_P3B_MOE_LAYERS (default 3,23,44), at Exl3MoEMethod.apply after the rank-sliced output:
    x (m x 4096 bf16, all m rows incl. padding), topk_ids, topk_weights (as apply normalised them), and the
    served output (apply's return value, the rank-local partial before the TP all-reduce).  Host-side, once per
    (layer, m, plan): the plan actually bound in _apply_rank_sliced (decode / prefill) and its launch fields.
  * Sparse MLA, layers GLM53_P3B_ATT_LAYERS (default 3,23; both MLA layers owned by rank 0 under split:25),
    on the DCP1 kpool branch and the decode kernel only, for NA host-chosen batch rows:
      H1 (after _logical_topk_to_physical_slots, before nsa/compaction): logical top-k, the PRE-compaction
         physical table, the consumed causal length L (cache_seq_lens_per_token) and the position p;
      H2 (just before the decode bind): q as bound, the POST-compaction table, the consumed nsa and
         cache_lengths, a device gather of every KV record the PRE table references (-1 -> slot 0), and the
         slot-0 record;
      H3 (just before forward_mqa returns on the decode branch): the attention output (and LSE if returned).

Graph safety (FULL decode graphs replay without Python):
  * every hook is a fixed sequence of device ops whose shapes depend only on static facts (layer, m, n);
    no .item()/.tolist()/nonzero/boolean indexing/host tensor creation inside a hook;
  * which ring slot a hook writes, and whether it writes a real slot or the sink slot R, is computed ON DEVICE
    from a guard-owned meta vector that the host refreshes by an H2D copy before every real forward
    (prepare_attn) and disarms after it (execute_model), exactly as the mapguard's E buffer;
  * every capture writes a header begin word first and an end word last (seqlock); the host poller copies the
    header, the data, then the header again on a side stream, and keeps an entry only if both copies show
    begin == end == the expected sequence number;
  * the host assigns sequence numbers and never arms sequence s while entry s - R of the same ring is
    unresolved (not yet copied out, or known not committed), so a slot is never rewritten before it is read.
It never writes an engine tensor: engine tensors are only read (index_select / copy into guard buffers).

Disk: captured arrays go ONLY to <GLM53_P3B_DIR>/PRIVATE-capture (dirs 0700, files 0600, created with explicit
modes; the process umask is not changed).  <GLM53_P3B_DIR>/log holds records with shapes, digests, counts,
categories and step bookkeeping only; no tensor values, no positions or lengths.

Env:
  GLM53_P3B              "" (off) | "capture"
  GLM53_P3B_DIR          output root (default /tmp/p3b)
  GLM53_P3B_MOE_LAYERS   default "3,23,44"
  GLM53_P3B_ATT_LAYERS   default "3,23"
  GLM53_P3B_POLL_SECS    default 1.0
Trigger files in GLM53_P3B_DIR: PHASE_TRAFFIC (runner, after readiness), FLUSH (runner, at the end; the poller
then resolves every armed entry or gives up after 30 s and writes FLUSHED).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import time

import numpy as np
import torch

MODE = os.environ.get("GLM53_P3B", "").strip().lower()
ENABLED = MODE == "capture"
OUT_DIR = os.environ.get("GLM53_P3B_DIR", "/tmp/p3b")
PRIV_DIR = os.path.join(OUT_DIR, "PRIVATE-capture")
LOG_DIR = os.path.join(OUT_DIR, "log")
POLL_SECS = float(os.environ.get("GLM53_P3B_POLL_SECS", "1.0"))
MOE_LAYERS = tuple(int(v) for v in os.environ.get("GLM53_P3B_MOE_LAYERS", "3,23,44").split(",") if v.strip())
ATT_LAYERS = tuple(int(v) for v in os.environ.get("GLM53_P3B_ATT_LAYERS", "3,23").split(",") if v.strip())

SCHEMA = "local-p3b-capture/1"
HIDDEN_DEFAULT = 4096
TOPK_DEFAULT = 8
MMAX = 64          # largest MoE batch captured (= max_cudagraph_capture_size of the served config)
RM = 8             # MoE ring slots per layer (+1 sink)
NA = 8             # attention rows per armed step
RA = 4             # attention ring slots per layer (+1 sink)
W_DEFAULT = 2048   # sparse-MLA top-k width (checked against the impl at registration)

# meta vector (int64, device): written by the host before each real forward
M_ARM_MOE, M_SEQ_MOE, M_ARM_ATT, M_SEQ_ATT, M_STEP, M_NTOK, M_NTOK_PAD, M_NREQS = range(8)
M_ROWS = 8
META_N = M_ROWS + NA
# ring header columns
HDR_COLS = ["seq_begin", "step", "m_static", "ntok", "ntok_pad", "kind", "aux", "seq_end"]   # aux: att = cache slots
H_BEGIN, H_STEP, H_M, H_NTOK, H_NPAD, H_KIND, H_AUX, H_END = range(8)
HDRW = len(HDR_COLS)
ATT_SCAL = ["row", "valid", "pos", "L", "nsa_post", "cache_len"]
S_ROW, S_VALID, S_POS, S_L, S_NSA, S_CLEN = range(6)

# Arming policy, fixed before data (PROPOSAL.md section 5).  Targets count ARMED entries; the poller reports
# EMITTED entries; quotas_met uses emitted entries of the traffic phase.
MOE_TARGET = {
    "traffic": {32: 10, 24: 6, 16: 6, 8: 4, 4: 4, 2: 2, 1: 2, 40: 2, 48: 2, 56: 2, 64: 2},
    "startup": {40: 3, 48: 3, 56: 3, 64: 3, 32: 2},
}
MOE_REQUIRED = {32: 10, 24: 6, 16: 6, 8: 4, 4: 4}      # traffic phase, for quotas_met
MOE_STRIDE = 6       # steps between armed MoE captures of the same (phase, padded size)
ATT_TARGET = {"dense": 12, "boundary": 16, "clamped": 10}   # request groups (<= 4 rows each), per run
ATT_STRIDE = 4
ATT_MAX_ARMED = 48
MOE_MAX_ARMED = 96
BOUNDARY_RES = (0, 61, 62, 63)   # (p + 1) % 64 values whose inflated count can cross a 64 boundary
APPROX_SLACK = 3                 # host positions are within [num_computed - 3, num_computed] under async scheduling
FLUSH_WAIT_S = 30.0


# ------------------------------------------------------------------ device structures
class Ring:
    """Fixed-shape device ring: R data slots + 1 sink slot (index R) for disarmed writes."""

    def __init__(self, name: str, layer: int, R: int, specs: dict, device):
        self.name, self.layer, self.R = name, layer, R
        self.t = {k: torch.zeros((R + 1,) + tuple(shape), dtype=dt, device=device) for k, (shape, dt) in specs.items()}
        self.t["hdr"] = torch.full((R + 1, HDRW), -1, dtype=torch.int64, device=device)
        self.cnt = torch.zeros(1, dtype=torch.int64, device=device)     # committed captures (device counter)
        self.oob = torch.zeros(1, dtype=torch.int64, device=device)     # gathered slot ids clamped into range
        self.end_col = self.t["hdr"][:, H_END]                          # strided column view

    def nbytes(self) -> int:
        return sum(v.numel() * v.element_size() for v in self.t.values())


def target_slot(meta: torch.Tensor, arm_idx: int, seq_idx: int, R: int) -> torch.Tensor:
    """(1,) int64 device index: (seq - 1) % R when armed, else the sink R.  Pure device ops."""
    arm = meta[arm_idx:arm_idx + 1]
    seq = meta[seq_idx:seq_idx + 1]
    slot = torch.remainder(seq - 1, R)
    return torch.where(arm > 0, slot, torch.full_like(slot, R))


def hdr_begin(ring: Ring, tgt: torch.Tensor, meta: torch.Tensor, seq_idx: int, m: int, kind: int) -> None:
    dev = meta.device
    row = torch.cat([meta[seq_idx:seq_idx + 1], meta[M_STEP:M_STEP + 1],
                     torch.full((1,), int(m), dtype=torch.int64, device=dev),
                     meta[M_NTOK:M_NTOK + 1], meta[M_NTOK_PAD:M_NTOK_PAD + 1],
                     torch.full((1,), int(kind), dtype=torch.int64, device=dev),
                     torch.zeros(1, dtype=torch.int64, device=dev),
                     torch.full((1,), -1, dtype=torch.int64, device=dev)])
    ring.t["hdr"].index_copy_(0, tgt, row.unsqueeze(0))


def hdr_end(ring: Ring, tgt: torch.Tensor, meta: torch.Tensor, seq_idx: int, arm_idx: int) -> None:
    ring.end_col.index_copy_(0, tgt, meta[seq_idx:seq_idx + 1])
    ring.cnt.add_(meta[arm_idx:arm_idx + 1].clamp(0, 1))


def put(buf: torch.Tensor, tgt: torch.Tensor, src: torch.Tensor) -> None:
    """buf[tgt, :src.shape[0], ...] = src for a (R+1, cap, ...) ring tensor; fixed shape, device index."""
    view = buf
    for d, n in enumerate(src.shape):
        if n != buf.shape[d + 1]:
            view = view.narrow(d + 1, 0, int(n))
    view.index_copy_(0, tgt, src.unsqueeze(0).to(buf.dtype))


# ------------------------------------------------------------------ process state
class _State:
    def __init__(self):
        self.runner = None
        self.rank = -1
        self.device = None
        self.meta = None
        self.done_step = None
        self.moe: dict[int, Ring] = {}
        self.att: dict[int, Ring] = {}
        self.moe_mod: dict[int, int] = {}       # id(RoutedExperts module) -> layer index (target model only)
        self.att_impl: dict[int, int] = {}      # id(B12xMLASparseImpl) -> layer index (target model only)
        self.att_info: dict[int, dict] = {}
        self.last_moe_layer = -1
        self.pos_buf = None
        self.staging = None
        self.events = None
        self.sidx = 0
        self.step = 0
        self.in_exec = False
        self.uploaded_this_exec = False
        self.extra_prepare = 0
        self.armed_host = False
        self.mode = "?"
        self.pending_step = None
        self.phase = "startup"
        self.errors = 0
        self.fh: dict = {}
        self.log_lock = threading.Lock()
        self.sites_logged: set = set()
        self.att_flags: dict[int, int] = {}     # capture-time Python: 1 after H1, 2 after H2 (per layer)
        self.att_tgt: dict[int, torch.Tensor] = {}
        self.att_rc: dict[int, torch.Tensor] = {}
        # host bookkeeping (engine thread arms; poller thread resolves)
        self.lock = threading.Lock()
        self.seq_moe = 0
        self.seq_att = 0
        self.armed_moe: dict[int, dict] = {}
        self.armed_att: dict[int, dict] = {}
        self.resolved_moe: set = set()
        self.resolved_att: set = set()
        self.moe_armed_count: dict = {}
        self.moe_last_step: dict = {}
        self.att_armed_groups = {k: 0 for k in ATT_TARGET}
        self.att_last_step = -10 ** 9
        self.att_rot = 0
        self.arm_skipped_lock = 0
        self.arm_skipped_backpressure = 0
        # poller
        self.poller = None
        self.stop = threading.Event()       # set only by tests (the served poller runs until process exit)
        self.side = None
        self.pins: dict = {}
        self.emitted = {"moe": {}, "att": {}}          # (seq, layer) -> True
        self.abandoned = {"moe": {}, "att": {}}
        self.torn = 0
        self.flushed = False
        self.emit_counts = {"moe": {}, "att": {}}


S = _State()


def _capturing() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False


def _mkdir_private(path: str) -> None:
    """Create path with every component at or below PRIV_DIR at mode 0700 (makedirs' mode covers only the leaf)."""
    path = os.path.abspath(path)
    root = os.path.abspath(PRIV_DIR)
    os.makedirs(os.path.dirname(root), exist_ok=True)
    cur = root
    rel = os.path.relpath(path, root)
    parts = [] if rel == "." else rel.split(os.sep)
    if parts and parts[0] == "..":
        raise ValueError(f"private path outside {root}: {path}")
    for comp in [None] + parts:
        if comp is not None:
            cur = os.path.join(cur, comp)
        if not os.path.isdir(cur):
            os.mkdir(cur, 0o700)
        os.chmod(cur, 0o700)


def _log(rec: dict, prefix: str = "cap") -> None:
    """log/<prefix>_p<pid>.jsonl: bookkeeping only (shapes, digests, counts, categories)."""
    try:
        fh = S.fh.get(prefix)
        if fh is None:
            os.makedirs(LOG_DIR, exist_ok=True)
            with S.log_lock:
                fh = S.fh.get(prefix) or open(os.path.join(LOG_DIR, f"{prefix}_p{os.getpid()}.jsonl"), "a")
                S.fh[prefix] = fh
        rec.setdefault("t", round(time.time(), 4))
        rec.setdefault("rank", S.rank)
        line = json.dumps(rec, default=str) + "\n"
        with S.log_lock:
            fh.write(line)
            fh.flush()
    except Exception:  # noqa: BLE001
        S.errors += 1


def _err(where: str, exc: BaseException) -> None:
    S.errors += 1
    if S.errors <= 50:
        _log(dict(kind="capture_error", where=where, err=repr(exc)[:400]))


def _ok() -> bool:
    return ENABLED and S.meta is not None


def _site_once(key, **info) -> None:
    if key in S.sites_logged:
        return
    S.sites_logged.add(key)
    _log(dict(kind="site", key=list(key) if isinstance(key, tuple) else key, capturing=_capturing(), **info))


def _layer_idx(name) -> int:
    try:
        parts = str(name).split(".")
        return int(parts[parts.index("layers") + 1])
    except Exception:  # noqa: BLE001
        return -1


def serialize_caps(caps) -> dict:
    """Dataclass Caps -> JSON-safe dict that replay/replay_attn.py turns back into Caps exactly."""
    import dataclasses
    import enum
    out = {}
    if caps is None or not dataclasses.is_dataclass(caps):
        return dict(_repr=repr(caps)[:400])
    for f in dataclasses.fields(caps):
        v = getattr(caps, f.name, None)
        if isinstance(v, enum.Enum):
            val = v.value if isinstance(v.value, (int, str, bool)) else repr(v.value)
            out[f.name] = dict(enum=type(v).__name__, name=v.name, value=val)
        elif isinstance(v, bool) or v is None or isinstance(v, (int, str)):
            out[f.name] = v
        elif isinstance(v, float):
            out[f.name] = dict(float_hex=float(v).hex())
        elif isinstance(v, torch.dtype):
            out[f.name] = dict(dtype=str(v))
        elif isinstance(v, torch.device):
            out[f.name] = dict(device=str(v))
        else:
            out[f.name] = dict(_repr=repr(v)[:200])
    return out


def _sha256_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


# ------------------------------------------------------------------ registration (initialize_kv_cache)
def register(runner) -> None:
    from vllm.distributed import get_tensor_model_parallel_rank
    S.runner = runner
    S.rank = int(get_tensor_model_parallel_rank())
    vp = os.path.dirname(os.path.abspath(__file__))
    base = dict(kind="init", schema=SCHEMA, mode=MODE, pid=os.getpid(), rank=S.rank,
                moe_layers=list(MOE_LAYERS), att_layers=list(ATT_LAYERS), mmax=MMAX, rm=RM, na=NA, ra=RA,
                moe_target={p: {str(k): v for k, v in d.items()} for p, d in MOE_TARGET.items()},
                moe_required={str(k): v for k, v in MOE_REQUIRED.items()}, att_target=ATT_TARGET,
                moe_stride=MOE_STRIDE, att_stride=ATT_STRIDE,
                async_scheduling=bool(getattr(runner.scheduler_config, "async_scheduling", None)),
                torch=torch.__version__,
                env={k: v for k, v in sorted(os.environ.items())
                     if k.startswith(("B12X", "VLLM_EXL3", "VLLM_B12X", "VLLM_GLM53", "GLM53", "PYTORCH_CUDA"))},
                file_sha256={p: _sha256_file(os.path.join(vp, p)) for p in (
                    "p3b_capture_hooks.py", "model_executor/layers/quantization/exl3.py",
                    "v1/attention/backends/mla/b12x_mla_sparse.py")})
    if S.rank != 0:
        _log(dict(base, inert="rank != 0"))
        return
    dev = runner.device
    S.pos_buf = runner.input_buffers.positions
    issues = []
    # target-model MoE modules and MLA impls (runner.model only; the drafter is not under it)
    try:
        from vllm.model_executor.layers.quantization import exl3 as exl3mod
        is_draft = getattr(exl3mod, "_is_draft_layer", lambda _m: False)
    except Exception:  # noqa: BLE001
        is_draft = lambda _m: False  # noqa: E731
    try:
        from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseImpl
    except Exception as exc:  # noqa: BLE001
        B12xMLASparseImpl = None
        issues.append(("impl_import", repr(exc)[:160]))
    moe_names, att_names = {}, {}
    hidden, topk_seen = HIDDEN_DEFAULT, TOPK_DEFAULT
    for name, mod in runner.model.named_modules():
        L = _layer_idx(name)
        if any(w in name.lower() for w in ("draft", "mtp", "spec", "eagle", "dflash")):
            continue
        # the module Exl3MoEMethod.apply receives is the one that holds the prepared EXL3 weights
        if "mlp.experts" in name and L in MOE_LAYERS and hasattr(mod, "exl3_trellis_weights"):
            try:
                if is_draft(mod):
                    continue
            except Exception:  # noqa: BLE001
                pass
            if L in moe_names:          # two candidates for one layer: refuse to guess (the gate fails)
                issues.append(("moe_duplicate", L, moe_names[L], name))
                continue
            S.moe_mod[id(mod)] = L
            moe_names[L] = name
            hidden = int(getattr(mod, "hidden_size", hidden) or hidden)
        impl = getattr(mod, "impl", None)
        if B12xMLASparseImpl is not None and isinstance(impl, B12xMLASparseImpl) and L in ATT_LAYERS:
            if L in att_names:
                issues.append(("att_duplicate", L, att_names[L], name))
                continue
            S.att_impl[id(impl)] = L
            att_names[L] = name
            S.att_info[L] = _impl_info(impl)
    _setup(dev, moe_names, hidden, topk_seen, S.pos_buf)
    if len(moe_names) != len(MOE_LAYERS):
        issues.append(("moe_layers_found", sorted(moe_names)))
    if len(att_names) != len(ATT_LAYERS):
        issues.append(("att_layers_found", sorted(att_names)))
    for L, ai in S.att_info.items():
        if int(ai["topk_tokens"]) != W_DEFAULT:
            issues.append(("att_width", L, ai["topk_tokens"]))
    ring_bytes = {f"moe.{L}": r.nbytes() for L, r in S.moe.items()}
    ring_bytes.update({f"att.{L}": r.nbytes() for L, r in S.att.items()})
    _log(dict(base, moe_modules=moe_names, att_modules=att_names, att_info=S.att_info, hidden=hidden,
              ring_bytes=ring_bytes, ring_mib_total=round(sum(ring_bytes.values()) / 2 ** 20, 1), issues=issues,
              cuda_graph_note="rings and meta allocated before graph capture"))
    _mkdir_private(PRIV_DIR)
    _ensure_poller()


def _setup(dev, moe_names: dict, hidden: int, topk: int, pos_buf) -> None:
    """Allocate meta, done_step and the rings (before any graph capture).  Shared by register() and the CPU tests;
    S.moe_mod / S.att_impl / S.att_info must already be filled."""
    S.device = dev
    S.pos_buf = pos_buf
    S.last_moe_layer = max(moe_names) if moe_names else -1
    S.meta = torch.zeros(META_N, dtype=torch.int64, device=dev)
    S.meta[M_ROWS:].fill_(-1)
    S.done_step = torch.full((1,), -1, dtype=torch.int64, device=dev)
    for L in sorted(moe_names):
        S.moe[L] = Ring("moe", L, RM, {
            "x": ((MMAX, hidden), torch.bfloat16),
            "ids": ((MMAX, topk), torch.int64),
            "w": ((MMAX, topk), torch.float32),
            "out": ((MMAX, hidden), torch.bfloat16)}, dev)
    for L in sorted(S.att_info):
        ai = S.att_info[L]
        W = int(ai["topk_tokens"])
        specs = {
            "q": ((NA, ai["q_heads_cap"], ai["q_head_dim"]), torch.bfloat16),
            "pre": ((NA, W), torch.int32),
            "post": ((NA, W), torch.int32),
            "logical": ((NA, W), torch.int32),
            "rec": ((NA, W, ai["record_bytes"]), torch.uint8),
            "slot0": ((1, ai["record_bytes"]), torch.uint8),
            "out": ((NA, ai["q_heads_cap"], ai["v_head_dim"]), torch.bfloat16),
            "scal": ((NA, len(ATT_SCAL)), torch.int64)}
        if ai.get("returns_lse"):
            specs["lse"] = ((NA, ai["q_heads_cap"]), torch.float32)
        S.att[L] = Ring("att", L, RA, specs, dev)
    pin = torch.cuda.is_available() and S.meta.is_cuda
    with torch.inference_mode(False):
        S.staging = [torch.zeros(META_N, dtype=torch.int64, pin_memory=pin) for _ in range(4)]
    S.events = [None] * 4


def _impl_info(impl) -> dict:
    g = lambda a, d=None: getattr(impl, a, d)  # noqa: E731
    in_heads, k_heads = int(g("_input_num_heads", 0) or 0), int(g("_kernel_num_heads", 0) or 0)
    info = dict(
        input_num_heads=in_heads, kernel_num_heads=k_heads, q_heads_cap=max(in_heads, k_heads),
        q_head_dim=int(g("q_head_dim", 0) or 0), v_head_dim=int(g("kv_lora_rank", 0) or 0),
        record_bytes=int(g("_kv_record_bytes", 0) or 0), topk_tokens=int(g("topk_tokens", 0) or 0),
        block_size=int(g("block_size", 0) or 0), pad_heads=bool(g("_pad_heads", False)),
        returns_lse=bool(g("need_to_return_lse_for_decode", False)),
        head_major_output=bool(g("_head_major_mla_output", False)),
        decode_max_rows=int(g("_decode_max_rows", 0) or 0), num_splits_cap=int(g("_num_splits_cap", 0) or 0),
        spec_decode_max_q=int(g("spec_decode_max_q", 0) or 0),
        spec_extend_as_decode=bool(g("spec_extend_as_decode", False)),
        indexer_outputs_physical_slots=bool(g("_indexer_outputs_physical_slots", False)),
        glm_next_record=bool(g("_glm_next_record", False)), dcp_world_size=int(g("dcp_world_size", 1) or 1),
        scale_hex=float(g("scale", 0.0) or 0.0).hex())
    try:
        info["decode_caps"] = serialize_caps(getattr(g("_decode_plan"), "caps", None))
    except Exception as exc:  # noqa: BLE001
        info["decode_caps"] = dict(_error=repr(exc)[:200])
    return info


# ------------------------------------------------------------------ host arming (prepare_attn, engine thread)
def _resolved(kind: str, seq: int) -> bool:
    if seq <= 0:
        return True
    return seq in (S.resolved_moe if kind == "moe" else S.resolved_att)


def _att_choose(input_batch) -> tuple[list[int], list[str]]:
    """Up to NA rows in whole request groups, targeting the categories still below target.
    Categories are approximate (host positions are bounds under async scheduling); final classification is
    done offline from the captured exact positions and lengths."""
    nreq = int(input_batch.num_reqs)
    qsl = np.asarray(input_batch.query_start_loc_np)[: nreq + 1]
    nc = np.asarray(input_batch.num_computed_tokens_np)[:nreq]
    cands = {k: [] for k in ATT_TARGET}
    for i in range(nreq):
        lo, hi = int(qsl[i]), int(qsl[i + 1])
        k = hi - lo
        if k < 2:
            continue
        c = int(nc[i])
        if c + 1 - APPROX_SLACK >= W_DEFAULT + 4:
            cat = "clamped"
        elif c + k < W_DEFAULT - 4:
            near = any(((c - s + j + 1) % 64) in BOUNDARY_RES for s in range(APPROX_SLACK + 1) for j in range(k))
            cat = "boundary" if near else "dense"
        else:
            continue
        cands[cat].append((i, lo, hi))
    rows, cats = [], []
    used = set()
    for _ in range(2):
        order = sorted(ATT_TARGET, key=lambda c_: -(ATT_TARGET[c_] - S.att_armed_groups[c_] -
                                                    cats.count(c_)))
        pick = None
        for cat in order:
            if ATT_TARGET[cat] - S.att_armed_groups[cat] - cats.count(cat) <= 0:
                continue
            avail = [g for g in cands[cat] if g[0] not in used]
            if avail:
                pick = (cat, avail[(S.att_rot + len(cats)) % len(avail)])
                break
        if pick is None:
            break
        cat, (i, lo, hi) = pick
        take = list(range(lo, min(hi, lo + NA - len(rows))))
        if not take:
            break
        used.add(i)
        rows += take
        cats.append(cat)
        if len(rows) >= NA:
            break
    S.att_rot += 1
    return rows, cats


def _upload(runner, input_batch) -> None:
    S.step += 1
    step = S.step
    n = int(input_batch.num_tokens)
    npad = int(getattr(input_batch, "num_tokens_after_padding", n))
    nreq = int(input_batch.num_reqs)
    has_prefill = bool(getattr(input_batch, "has_prefill", False))
    ndraft = int(getattr(input_batch, "num_draft_tokens", 0) or 0)
    internal = any(str(r).startswith("_warmup") for r in list(getattr(input_batch, "req_ids", []) or []))
    phase = "internal-warmup" if internal else S.phase
    arm_moe = arm_att = 0
    seq_moe = seq_att = 0
    rows: list[int] = []
    cats: list[str] = []
    got = S.lock.acquire(timeout=0.005)
    if not got:
        S.arm_skipped_lock += 1
    else:
        try:
            # MoE
            tgt = MOE_TARGET.get(phase, {}).get(npad, 0)
            key = (phase, npad)
            if (S.moe and npad <= MMAX and tgt and S.moe_armed_count.get(key, 0) < tgt
                    and step - S.moe_last_step.get(key, -10 ** 9) >= MOE_STRIDE
                    and len(S.armed_moe) < MOE_MAX_ARMED):
                if _resolved("moe", S.seq_moe + 1 - RM):
                    S.seq_moe += 1
                    seq_moe, arm_moe = S.seq_moe, 1
                    S.moe_armed_count[key] = S.moe_armed_count.get(key, 0) + 1
                    S.moe_last_step[key] = step
                    S.armed_moe[seq_moe] = dict(step=step, phase=phase, npad=npad, ntok=n, nreq=nreq,
                                                prefill=has_prefill)
                else:
                    S.arm_skipped_backpressure += 1
            # attention (verify steps only: decode kernel path)
            if (S.att and phase == "traffic" and not has_prefill and ndraft > 0
                    and step - S.att_last_step >= ATT_STRIDE and len(S.armed_att) < ATT_MAX_ARMED
                    and any(S.att_armed_groups[c] < ATT_TARGET[c] for c in ATT_TARGET)):
                if _resolved("att", S.seq_att + 1 - RA):
                    rows, cats = _att_choose(input_batch)
                    if rows:
                        S.seq_att += 1
                        seq_att, arm_att = S.seq_att, 1
                        S.att_last_step = step
                        for c in cats:
                            S.att_armed_groups[c] += 1
                        S.armed_att[seq_att] = dict(step=step, phase=phase, ntok=n, npad=npad, nreq=nreq,
                                                    nrows=len(rows), cats=cats)
                else:
                    S.arm_skipped_backpressure += 1
        finally:
            S.lock.release()
    k = S.sidx
    S.sidx = (k + 1) % len(S.staging)
    ev = S.events[k]
    if ev is not None:
        ev.synchronize()   # only waits if the GPU is >= 4 steps behind the host
    buf = S.staging[k].numpy()
    buf[:] = 0
    buf[M_ARM_MOE], buf[M_SEQ_MOE], buf[M_ARM_ATT], buf[M_SEQ_ATT] = arm_moe, seq_moe, arm_att, seq_att
    buf[M_STEP], buf[M_NTOK], buf[M_NTOK_PAD], buf[M_NREQS] = step, n, npad, nreq
    buf[M_ROWS:] = -1
    for j, r in enumerate(rows[:NA]):
        buf[M_ROWS + j] = r
    S.meta.copy_(S.staging[k], non_blocking=True)
    if S.meta.is_cuda:
        e = torch.cuda.Event()
        e.record()
        S.events[k] = e
    S.armed_host = True
    S.pending_step = dict(kind="step", step=step, phase=phase, ntok=n, ntok_pad=npad, nreq=nreq,
                          prefill=has_prefill, ndraft=ndraft, arm_moe=arm_moe, seq_moe=seq_moe, arm_att=arm_att,
                          seq_att=seq_att, att_rows=len(rows), att_cats=cats)


# ------------------------------------------------------------------ hooks (graph-safe device ops only)
def hook_moe_bind(layer, which: str, m: int, binding, runtime) -> None:
    """Host only: the plan actually bound for (layer, m).  Logged once per key (capture time under graphs)."""
    if not _ok():
        return
    try:
        L = S.moe_mod.get(id(layer))
        if L is None:
            return
        key = ("moe_bind", L, int(m), which)
        if key in S.sites_logged:
            return
        mb = getattr(binding, "mixed_trellis_binding", None)
        launch = getattr(mb, "launch", None)
        fields = {}
        for f in ("size_m", "fc1_tile_k", "fc1_tile_n", "fc2_tile_k", "fc2_tile_n", "moe_block_size",
                  "fc2_moe_block_size", "max_m_blocks", "blocks_per_sm", "route_ids_dtype", "direct_topk_routes"):
            v = getattr(launch, f, None)
            fields[f] = v if isinstance(v, (int, bool, str)) or v is None else str(v)
        _site_once(key, layer=L, m=int(m), plan=which, launch=fields,
                   min_trellis_m=runtime.get("min_trellis_m"), max_trellis_m=runtime.get("max_trellis_m"),
                   tile_config=str(getattr(layer, "exl3_trellis_tile_config", None)))
    except Exception as exc:  # noqa: BLE001
        _err("moe_bind", exc)


def hook_moe(layer, x2d, weights, ids, out) -> None:
    """After _apply_rank_sliced in Exl3MoEMethod.apply: copy inputs and the served output into the ring."""
    if not _ok():
        return
    try:
        L = S.moe_mod.get(id(layer))
        if L is None:
            return
        ring = S.moe.get(L)
        m = int(x2d.shape[0])
        ok = (ring is not None and m <= MMAX and x2d.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
              and int(x2d.shape[1]) == ring.t["x"].shape[2] and int(ids.shape[1]) <= ring.t["ids"].shape[2])
        _site_once(("moe", L, m), layer=L, m=m, captured=bool(ok), x=[list(x2d.shape), str(x2d.dtype)],
                   ids=[list(ids.shape), str(ids.dtype)], w=[list(weights.shape), str(weights.dtype)],
                   out=[list(out.shape), str(out.dtype)])
        if ok:
            tgt = target_slot(S.meta, M_ARM_MOE, M_SEQ_MOE, ring.R)
            hdr_begin(ring, tgt, S.meta, M_SEQ_MOE, m, 0)
            put(ring.t["x"], tgt, x2d)
            put(ring.t["ids"], tgt, ids)
            put(ring.t["w"], tgt, weights)
            put(ring.t["out"], tgt, out.reshape(m, -1))
            hdr_end(ring, tgt, S.meta, M_SEQ_MOE, M_ARM_MOE)
        if L == S.last_moe_layer:
            S.done_step.copy_(S.meta[M_STEP:M_STEP + 1])   # every forward, any m: marks the step complete
    except Exception as exc:  # noqa: BLE001
        _err("moe", exc)


def _att_layer(impl):
    if not _ok():
        return None
    return S.att_impl.get(id(impl))


def hook_attn_pre(impl, topk_logical, selected_pre, per_token_cache, n: int) -> None:
    """H1: after _logical_topk_to_physical_slots, before nsa and compaction (DCP1 kpool branch)."""
    L = _att_layer(impl)
    if L is None:
        return
    S.att_flags[L] = 0
    try:
        ring = S.att[L]
        n = int(n)
        W = ring.t["pre"].shape[2]
        if n <= 0 or int(selected_pre.shape[1]) != W:
            _site_once(("att_skip", L, n), layer=L, n=n, width=int(selected_pre.shape[1]))
            return
        dev = S.meta.device
        tgt = target_slot(S.meta, M_ARM_ATT, M_SEQ_ATT, ring.R)
        rows = S.meta[M_ROWS:M_ROWS + NA]
        valid = (rows >= 0) & (rows < n)
        rc = torch.where(valid, rows, torch.zeros_like(rows))
        hdr_begin(ring, tgt, S.meta, M_SEQ_ATT, n, 1)
        put(ring.t["logical"], tgt, topk_logical.index_select(0, rc))
        put(ring.t["pre"], tgt, selected_pre.index_select(0, rc))
        scal = torch.full((NA, len(ATT_SCAL)), -1, dtype=torch.int64, device=dev)
        scal[:, S_ROW] = rows
        scal[:, S_VALID] = valid.to(torch.int64)
        scal[:, S_POS] = S.pos_buf.index_select(0, rc).to(torch.int64)
        scal[:, S_L] = per_token_cache.index_select(0, rc).to(torch.int64)
        put(ring.t["scal"], tgt, scal)
        S.att_tgt[L], S.att_rc[L] = tgt, rc
        S.att_flags[L] = 1
        _site_once(("att_pre", L, n), layer=L, n=n, logical=[list(topk_logical.shape), str(topk_logical.dtype)],
                   pre=[list(selected_pre.shape), str(selected_pre.dtype)], L_dtype=str(per_token_cache.dtype))
    except Exception as exc:  # noqa: BLE001
        _err("attn_pre", exc)


def hook_attn_in(impl, q, kv_cache, selected_post, nsa_post, cache_seqlens, n: int) -> None:
    """H2: decode branch, just before bind: q, post table, nsa, cache_lengths, KV records by the PRE table."""
    L = _att_layer(impl)
    if L is None or S.att_flags.get(L) != 1:
        return
    try:
        ring = S.att[L]
        tgt, rc = S.att_tgt[L], S.att_rc[L]
        qh, qd = int(q.shape[1]), int(q.shape[2])
        rb = int(kv_cache.shape[-1])
        if qh > ring.t["q"].shape[2] or qd != ring.t["q"].shape[3] or rb != ring.t["rec"].shape[3]:
            _site_once(("att_in_skip", L), layer=L, q=list(q.shape), rec_bytes=rb)
            S.att_flags[L] = 0
            return
        put(ring.t["q"], tgt, q.index_select(0, rc))
        put(ring.t["post"], tgt, selected_post.index_select(0, rc))
        scal = ring.t["scal"]
        # scal[tgt, :, S_NSA] = nsa_post[rc]; scal[tgt, :, S_CLEN] = cache_seqlens[rc]  (fixed-shape device ops)
        upd = scal.index_select(0, tgt)[0]
        upd[:, S_NSA] = nsa_post.index_select(0, rc).to(torch.int64)
        ncl = int(cache_seqlens.shape[0])
        rcc = rc.clamp(0, max(ncl - 1, 0))
        upd[:, S_CLEN] = cache_seqlens.index_select(0, rcc).to(torch.int64)
        scal.index_copy_(0, tgt, upd.unsqueeze(0))
        flat = kv_cache.reshape(-1, rb)
        nslots = int(flat.shape[0])
        # header aux = number of cache slots (static): the replay flags rows whose PRE table points past it
        ring.t["hdr"][:, H_AUX].index_copy_(0, tgt, torch.full((1,), nslots, dtype=torch.int64, device=flat.device))
        pre = ring.t["pre"].index_select(0, tgt)[0].reshape(-1).to(torch.int64)
        oob = (pre >= nslots).to(torch.int64).sum().reshape(1)
        idx = pre.clamp(0, nslots - 1)
        W = ring.t["pre"].shape[2]
        put(ring.t["rec"], tgt, flat.index_select(0, idx).reshape(NA, W, rb))
        put(ring.t["slot0"], tgt, flat[0:1])
        ring.oob.add_(oob * S.meta[M_ARM_ATT:M_ARM_ATT + 1].clamp(0, 1))
        S.att_flags[L] = 2
        _site_once(("att_in", L, int(q.shape[0])), layer=L, n=int(q.shape[0]), q=[list(q.shape), str(q.dtype)],
                   post=[list(selected_post.shape), str(selected_post.dtype)], nsa=str(nsa_post.dtype),
                   cache_seqlens=[list(cache_seqlens.shape), str(cache_seqlens.dtype)], kv=[list(kv_cache.shape),
                                                                                             str(kv_cache.dtype)])
    except Exception as exc:  # noqa: BLE001
        S.att_flags[L] = 0
        _err("attn_in", exc)


def hook_attn_out(impl, out, lse) -> None:
    """H3: decode branch, the value forward_mqa returns (after the head-padding copy)."""
    L = _att_layer(impl)
    if L is None or S.att_flags.get(L) != 2:
        return
    try:
        ring = S.att[L]
        tgt, rc = S.att_tgt[L], S.att_rc[L]
        if (out.dim() != 3 or int(out.shape[1]) > ring.t["out"].shape[2]
                or int(out.shape[2]) != ring.t["out"].shape[3]):
            _site_once(("att_out_skip", L), layer=L, out=list(out.shape))
            return
        put(ring.t["out"], tgt, out.index_select(0, rc))
        if lse is not None and "lse" in ring.t and lse.dim() == 2 and int(lse.shape[1]) <= ring.t["lse"].shape[2]:
            put(ring.t["lse"], tgt, lse.index_select(0, rc).float())
        hdr_end(ring, tgt, S.meta, M_SEQ_ATT, M_ARM_ATT)
        _site_once(("att_out", L, int(out.shape[0])), layer=L, out=[list(out.shape), str(out.dtype),
                                                                    list(out.stride())],
                   lse=None if lse is None else [list(lse.shape), str(lse.dtype)])
    except Exception as exc:  # noqa: BLE001
        _err("attn_out", exc)
    finally:
        S.att_flags[L] = 0


# ------------------------------------------------------------------ poller (host thread)
def _ensure_poller() -> None:
    if S.poller is None and S.meta is not None:
        S.poller = threading.Thread(target=_poller_loop, name="local-p3b-poll", daemon=True)
        S.poller.start()


def _alloc_pins(tensors, pin):
    """Host copy buffers as normal tensors whichever thread / inference mode allocates them (22ap lesson)."""
    with torch.inference_mode(False):
        return [torch.empty(t.shape, dtype=t.dtype, pin_memory=pin) for t in tensors]


def _copy_rows(tensors, rows):
    """Copy t[rows] (each a 0-dim-indexed slot view) to host on the side stream, in list order."""
    cuda = tensors[0].is_cuda
    key = tuple((tuple(t.shape), t.dtype) for t in tensors)
    if cuda:
        if S.side is None:
            S.side = torch.cuda.Stream(device=tensors[0].device)
        pins = S.pins.get(key)
        if pins is None:
            pins = S.pins[key] = _alloc_pins([t[rows[0]] for t in tensors], pin=True)
        with torch.cuda.stream(S.side):
            for p_, t, r in zip(pins, tensors, rows):
                p_.copy_(t[r], non_blocking=True)
            e = torch.cuda.Event()
            e.record(S.side)
        e.synchronize()
        return [p_.clone() for p_ in pins]
    return [t[r].clone() for t, r in zip(tensors, rows)]


def _write_private(path: str, arrays: dict) -> str:
    """Write an .npz with mode 0600 (O_EXCL: never overwrite).  Returns sha256 of the file bytes."""
    _mkdir_private(os.path.dirname(path))
    bio = io.BytesIO()
    np.savez(bio, **arrays)
    data = bio.getvalue()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return hashlib.sha256(data).hexdigest()


def _np(t: torch.Tensor) -> np.ndarray:
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(np.uint16)
    return t.numpy()


def _poll_kind(kind: str, rings: dict, armed: dict, resolved: set, R: int, done: int, hdrs: dict) -> None:
    for seq in sorted(s for s in armed if s not in resolved):
        info = armed[seq]
        slot = (seq - 1) % R
        all_done = True
        for L, ring in rings.items():
            key = (seq, L)
            if key in S.emitted[kind] or key in S.abandoned[kind]:
                continue
            h = hdrs[L][slot]
            if int(h[H_BEGIN]) == seq and int(h[H_END]) == seq:
                names = [k for k in ring.t if k != "hdr"]
                tensors = [ring.t["hdr"]] + [ring.t[k] for k in names] + [ring.t["hdr"]]
                got = _copy_rows(tensors, [slot] * len(tensors))
                h1, h2 = got[0], got[-1]
                if not (torch.equal(h1, h2) and int(h1[H_BEGIN]) == seq and int(h1[H_END]) == seq):
                    S.torn += 1
                    _log(dict(kind="torn", ring=kind, layer=L, seq=seq))
                    all_done = False
                    continue
                arrays = {k: _np(v) for k, v in zip(names, got[1:-1])}
                arrays["hdr"] = h1.numpy()
                path = os.path.join(PRIV_DIR, kind, f"L{L}", f"s{seq:05d}.npz")
                digest = _write_private(path, arrays)
                S.emitted[kind][key] = True
                m = int(h1[H_M])
                _log(dict(kind="entry", ring=kind, layer=L, seq=seq, step=int(h1[H_STEP]), m_static=m,
                          ntok=int(h1[H_NTOK]), ntok_pad=int(h1[H_NPAD]), phase=info.get("phase"),
                          cats=info.get("cats"), nrows=info.get("nrows"),
                          file=os.path.relpath(path, PRIV_DIR), sha256=digest,
                          shapes={k: list(v.shape) for k, v in arrays.items()}))
                ck = (info.get("phase"), m) if kind == "moe" else "att"
                d = S.emit_counts[kind].setdefault(str(L), {})
                d[str(ck)] = d.get(str(ck), 0) + 1
            elif done >= int(info["step"]):
                S.abandoned[kind][key] = True
                _log(dict(kind="abandoned", ring=kind, layer=L, seq=seq, step=int(info["step"]),
                          hdr_begin=int(h[H_BEGIN]), hdr_end=int(h[H_END]), done_step=done))
            else:
                all_done = False
        if all_done:
            with S.lock:
                resolved.add(seq)
                # an entry abandoned on every layer captured nothing: give its quota back so the host re-arms
                if all((seq, L) in S.abandoned[kind] for L in rings):
                    if kind == "moe":
                        key = (info.get("phase"), info.get("npad"))
                        S.moe_armed_count[key] = max(0, S.moe_armed_count.get(key, 0) - 1)
                    else:
                        for c in info.get("cats") or []:
                            S.att_armed_groups[c] = max(0, S.att_armed_groups[c] - 1)


def _status() -> dict:
    with S.lock:
        armed_att = dict(S.armed_att)
        n_armed_moe = len(S.armed_moe)
        n_res_moe, n_res_att = len(S.resolved_moe), len(S.resolved_att)
        armed_groups = dict(S.att_armed_groups)
    moe_ok = True
    for L in S.moe:
        d = S.emit_counts["moe"].get(str(L), {})
        for m, need in MOE_REQUIRED.items():
            if d.get(str(("traffic", m)), 0) < need:
                moe_ok = False
    att_groups = {c: 0 for c in ATT_TARGET}
    for seq, info in armed_att.items():
        if all((seq, L) in S.emitted["att"] for L in S.att):
            for c in info.get("cats") or []:
                att_groups[c] += 1
    att_ok = all(att_groups[c] >= ATT_TARGET[c] for c in ATT_TARGET) if S.att else False
    return dict(t=round(time.time(), 2), step=S.step, phase=S.phase, armed_moe=n_armed_moe,
                armed_att=len(armed_att), resolved_moe=n_res_moe, resolved_att=n_res_att,
                emitted_moe=len(S.emitted["moe"]), emitted_att=len(S.emitted["att"]),
                abandoned_moe=len(S.abandoned["moe"]), abandoned_att=len(S.abandoned["att"]), torn=S.torn,
                errors=S.errors, extra_prepare=S.extra_prepare, arm_skipped_lock=S.arm_skipped_lock,
                arm_skipped_backpressure=S.arm_skipped_backpressure, moe_emit=S.emit_counts["moe"],
                att_groups_emitted=att_groups, att_groups_armed=armed_groups,
                moe_quotas_met=moe_ok, att_quotas_met=att_ok, quotas_met=bool(moe_ok and att_ok),
                flushed=S.flushed)


def _write_status(st: dict, name: str = "status.json") -> None:
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        tmp = os.path.join(LOG_DIR, f".{name}.tmp")
        with open(tmp, "w") as fh:
            json.dump(st, fh)
        os.replace(tmp, os.path.join(LOG_DIR, name))
    except Exception as exc:  # noqa: BLE001
        _err("status", exc)


def poll(reason: str) -> dict:
    """One resolution pass: copy out committed entries, abandon entries whose step finished uncommitted."""
    if S.meta is None:
        return {}
    try:
        rings = [("done", S.done_step)] + [(("moe", L), r.t["hdr"]) for L, r in S.moe.items()] + \
                [(("att", L), r.t["hdr"]) for L, r in S.att.items()] + \
                [(("cnt_moe", L), r.cnt) for L, r in S.moe.items()] + [(("cnt_att", L), r.cnt) for L, r in S.att.items()] + \
                [(("oob", L), r.oob) for L, r in S.att.items()]
        # done_step FIRST: every header write of a step precedes that step's done_step write on the engine stream
        got = _copy_rows([t for _, t in rings], [slice(None)] * len(rings))
        snap = dict(zip([k for k, _ in rings], got))
        done = int(snap["done"][0])
        with S.lock:
            armed_moe = dict(S.armed_moe)
            armed_att = dict(S.armed_att)
        _poll_kind("moe", S.moe, armed_moe, S.resolved_moe, RM, done,
                   {L: snap[("moe", L)] for L in S.moe})
        _poll_kind("att", S.att, armed_att, S.resolved_att, RA, done,
                   {L: snap[("att", L)] for L in S.att})
        st = _status()
        st.update(reason=reason, done_step=done,
                  dev_commits={**{f"moe.{L}": int(snap[("cnt_moe", L)][0]) for L in S.moe},
                               **{f"att.{L}": int(snap[("cnt_att", L)][0]) for L in S.att}},
                  oob={str(L): int(snap[("oob", L)][0]) for L in S.att})
        _write_status(st)
        return st
    except Exception as exc:  # noqa: BLE001
        _err("poll", exc)
        return {}


def _poller_loop():
    try:
        if S.meta is not None and S.meta.is_cuda:
            torch.cuda.set_device(S.meta.device)
    except Exception:  # noqa: BLE001
        pass
    flush_t0 = None
    while not S.stop.is_set():
        time.sleep(POLL_SECS)
        try:
            if S.phase != "traffic" and os.path.exists(os.path.join(OUT_DIR, "PHASE_TRAFFIC")):
                S.phase = "traffic"
                _log(dict(kind="phase", phase="traffic", step=S.step))
            st = poll("periodic")
            if os.path.exists(os.path.join(OUT_DIR, "FLUSH")) and not S.flushed:
                flush_t0 = flush_t0 or time.time()
                with S.lock:
                    pending = (len(S.armed_moe) - len(S.resolved_moe)) + (len(S.armed_att) - len(S.resolved_att))
                if pending == 0 or time.time() - flush_t0 > FLUSH_WAIT_S:
                    S.flushed = True
                    st = poll("flush")
                    st["pending_at_flush"] = pending
                    st["flush_wait_s"] = round(time.time() - flush_t0, 1)
                    _write_status(st)
                    _write_status(st, "FLUSHED.json")
                    _log(dict(kind="flushed", **{k: v for k, v in st.items() if k != "moe_emit"}))
        except Exception as exc:  # noqa: BLE001
            _err("poller", exc)


# ------------------------------------------------------------------ worker install
def install() -> None:
    from vllm.v1.worker.gpu import cudagraph_utils as cgmod
    from vllm.v1.worker.gpu import model_runner as mrmod

    MR = mrmod.GPUModelRunner
    if getattr(MR, "_local_p3b", False):
        return
    MR._local_p3b = True
    o_init_kv, o_exec, o_prep = MR.initialize_kv_cache, MR.execute_model, MR.prepare_attn

    def initialize_kv_cache(self, *a, **kw):
        r = o_init_kv(self, *a, **kw)
        try:
            register(self)
        except Exception as exc:  # noqa: BLE001
            _err("register", exc)
        return r

    def execute_model(self, scheduler_output, *a, **kw):
        dummy = kw.get("dummy_run", a[1] if len(a) > 1 else False)
        S.in_exec = not dummy
        S.uploaded_this_exec = False
        S.mode = "none"
        try:
            return o_exec(self, scheduler_output, *a, **kw)
        finally:
            S.in_exec = False
            if S.armed_host and S.meta is not None:
                try:
                    S.meta[M_ARM_MOE:M_ARM_MOE + 1].zero_()   # stream-ordered after the forward
                    S.meta[M_ARM_ATT:M_ARM_ATT + 1].zero_()
                except Exception as exc:  # noqa: BLE001
                    _err("disarm", exc)
                S.armed_host = False
            if S.pending_step is not None:
                S.pending_step["cg"] = S.mode
                _log(S.pending_step, prefix="steps")
                S.pending_step = None

    def prepare_attn(self, input_batch):
        r = o_prep(self, input_batch)
        if S.meta is not None and not _capturing() and S.in_exec:
            if S.uploaded_this_exec:
                S.extra_prepare += 1
            else:
                S.uploaded_this_exec = True
                try:
                    _upload(self, input_batch)
                except Exception as exc:  # noqa: BLE001
                    _err("upload", exc)
        return r

    MR.initialize_kv_cache = initialize_kv_cache
    MR.execute_model = execute_model
    MR.prepare_attn = prepare_attn
    for cls in (cgmod.CudaGraphManager, getattr(cgmod, "ModelCudaGraphManager", None)):
        if cls is None:
            continue
        if "run_fullgraph" in cls.__dict__:
            o_full = cls.__dict__["run_fullgraph"]

            def run_fullgraph(self, *a, _o=o_full, **kw):
                S.mode = "FULL"
                return _o(self, *a, **kw)
            cls.run_fullgraph = run_fullgraph
        if "run_pw_graph" in cls.__dict__:
            o_pw = cls.__dict__["run_pw_graph"]

            def run_pw_graph(self, *a, _o=o_pw, **kw):
                S.mode = "PIECEWISE"
                return _o(self, *a, **kw)
            cls.run_pw_graph = run_pw_graph
    _log(dict(kind="installed", mode=MODE, pid=os.getpid()))


def ensure_worker_install() -> None:
    """Called by the patched b12x_mla_sparse module at import."""
    if not ENABLED:
        return
    try:
        install()
    except Exception as exc:  # noqa: BLE001
        _err("install", exc)
