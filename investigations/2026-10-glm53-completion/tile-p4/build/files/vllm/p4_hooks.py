# SPDX-License-Identifier: Apache-2.0
"""local-p4: observe-only instrumentation for the P4 two-rank MoE tile intervention (investigation
2026-10-glm53-completion, package tile-p4/, PROPOSAL.md sections 3-5).

INERT unless GLM53_P4=on.  The tile switch itself is NOT here: it is the marked helper ``_p4_forced_tile()`` in
``quantization/exl3.py`` (env GLM53_P4_TILE), so arm B never depends on this module importing.

What it does when GLM53_P4=on:
  1. Bound-launch recorder (both ranks, every routed-expert MoE layer, no registration needed):
     ``hook_bind`` runs before each ``runtime["api"].run(binding=...)`` in ``Exl3MoEMethod._apply_rank_sliced``
     (decode plan, prefill plan, tiled prefill).  One record per (rank, layer, plan, m) with the launch fields
     (size_m, fc1/fc2 tile K/N, moe_block_size, ...), the plan identity, capacity and the layer's weight-plan
     tile; a later bind of the same key with a different signature is logged again ("changed").  Host only; under
     FULL CUDA graphs the bind runs at capture (and warm-up) time, which is the launch every replay re-issues.
  2. Graph-safe stages (rank 0): fixed device buffers, allocated in ``register`` (GPUModelRunner.capture_model,
     BEFORE graph capture), written by in-graph hooks with fixed-shape copies only:
       MoE output after the TP all-reduce (Glm5NextMoE output, routed + shared) at layers 3, 23, 44;
       the contracted post-layer hidden state at layers 12 and 34 (hc_post + hc_contract, exactly the read-only
       materialisation the DFlash tap uses); the final-norm hidden state (input of compute_logits).
     A step with more than MMAX = 64 tokens (prefill chunks) is not staged (its stage counter is set to 0).
  3. Per-request stores (rank 0, NSLOT forced requests in flight): the forcewin port calls ``on_commit`` with the
     rows it records (each output index once, on the intended prefix); those rows are copied from the stages into
     per-slot stores by EAGER device ops (sampling runs eagerly): MoE output at every recorded position, hidden
     states at every 8th (o % 8 == 0), full fp32 logits at every 64th (o % 64 == 0; staged at on_logits, before
     the sampler can modify the logits).
  4. Step records: a host step log (time, counts, graph mode) and a device ring of per-request step composition
     (step, req_idx, row-0 output index, rows, accepted) written by eager device ops in the verify path.
  5. Flush: at request removal (start of the next execute_model, i.e. never during CUDA-graph capture) the slot is
     copied to host and written to <GLM53_P4_DIR>/PRIVATE/p4_<tag>_r<rank>.npz (0600, O_EXCL; dirs 0700).

Graph-safety rules (captured-p3b RESULT.md lessons):
  * NO background thread, no poller, no pinned allocation, no side stream, no event sync anywhere in this module.
    Device-to-host copies happen only in ``on_remove`` / ``flush_steps``, which refuse while a stream is capturing
    and before the first real (non-dummy) batch after registration (``S.serving``).
  * In-graph hooks use only fixed-shape device ops on buffers allocated before capture; per-step values are never
    read on the host inside them.
  * Every plan / layer setting the offline tools need is logged as plain JSON (bind records, registration record).

Env:
  GLM53_P4          "" (off) | "on"
  GLM53_P4_DIR      output root (default /tmp/p4)
  GLM53_P4_NSLOT    forced requests stored at once (default 8)
  GLM53_P4_WPOS     positions per request store (default 513)
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

ENABLED = os.environ.get("GLM53_P4", "").strip().lower() == "on"
OUT_DIR = os.environ.get("GLM53_P4_DIR", "/tmp/p4")
PRIV_DIR = os.path.join(OUT_DIR, "PRIVATE")
LOG_DIR = os.path.join(OUT_DIR, "log")
NSLOT = int(os.environ.get("GLM53_P4_NSLOT", "8"))
WPOS = int(os.environ.get("GLM53_P4_WPOS", "513"))
SCHEMA = "local-p4/1"
MOE_LAYERS = (3, 23, 44)
HID_LAYERS = (12, 34)
STAGE_NAMES = ("moe3", "moe23", "moe44", "hid12", "hid34", "final")
K_MOE = {3: 0, 23: 1, 44: 2}
K_HID = {12: 3, 34: 4}
K_FINAL = 5
MMAX = 64
HID_EVERY = 8
LOGIT_EVERY = 64
NSTEPREC = 16384
STEPREC_COLS = ("step", "req_idx", "o0", "n_rows", "accepted")
TILE_ARM_B = (64, 256, 64, 256)
LAUNCH_FIELDS = ("size_m", "fc1_tile_k", "fc1_tile_n", "fc2_tile_k", "fc2_tile_n", "moe_block_size",
                 "fc2_moe_block_size", "max_m_blocks", "blocks_per_sm", "route_ids_dtype", "direct_topk_routes")


class _State:
    def __init__(self):
        self.rank = None
        self.dev = None
        self.registered = False
        self.serving = False
        self.pending_reset = []     # req_idx whose device slot_of reset was deferred (removal refused inside a capture)
        self.errors = 0
        self.fh = {}
        self.log_lock = threading.Lock()
        self.bind_sig = {}          # (layer, plan, m, draft) -> signature
        self.bind_n = {}            # same key -> bind calls seen (host)
        self.ids = {}               # id(module) -> stage index (target model only)
        self.stage = None
        self.stage_n = None
        self.store = None
        self.have = None
        self.lstage = None
        self.lstage_o = None
        self.lstore = None
        self.lhave = None
        self.slot_of = None         # device [max_reqs + 1] -> slot (NSLOT = sink)
        self.slot_host = {}         # req_idx -> slot
        self.free = list(range(NSLOT))
        self.tag_of = {}            # req_idx -> tag
        self.tok = None
        self.mode = "?"
        self.steplog = []
        self.steprec = None
        self.step_flushed = 0
        self.last_step = 0
        self.flushed = []
        self.no_slot = 0
        self.max_reqs = 0
        self.vocab = 0


S = _State()


# ------------------------------------------------------------------ helpers
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
            try:
                import torch.distributed as dist
                return int(dist.get_rank()) if dist.is_initialized() else 0
            except Exception:  # noqa: BLE001
                return 0
    return S.rank


def _log(rec: dict, prefix: str = "p4") -> None:
    """log/<prefix>_r<rank>_p<pid>.jsonl: shapes, digests, counts, launch fields only (no tensor values)."""
    try:
        r = _rank()
        key = (prefix, r)
        with S.log_lock:
            fh = S.fh.get(key)
            if fh is None:
                os.makedirs(LOG_DIR, exist_ok=True)
                fh = S.fh[key] = open(os.path.join(LOG_DIR, f"{prefix}_r{r}_p{os.getpid()}.jsonl"), "a")
            rec = dict(rec)
            rec.setdefault("t", round(time.time(), 4))
            rec.setdefault("rank", r)
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
    except Exception:  # noqa: BLE001
        S.errors += 1


def _err(where: str, exc: BaseException) -> None:
    S.errors += 1
    if S.errors <= 50:
        _log(dict(kind="p4_error", where=where, err=repr(exc)[:400]))


def _layer_idx(name) -> int:
    try:
        parts = str(name).split(".")
        return int(parts[parts.index("layers") + 1])
    except Exception:  # noqa: BLE001
        return -1


def _phase() -> str:
    if S.serving:
        return "serve"
    if S.registered:
        return "capture"
    return "pre-register"


def _mkdir_private(path: str) -> None:
    """Every component from PRIV_DIR down at mode 0700 (makedirs' mode covers only the leaf)."""
    path = os.path.abspath(path)
    root = os.path.abspath(PRIV_DIR)
    os.makedirs(os.path.dirname(root), exist_ok=True)
    rel = os.path.relpath(path, root)
    parts = [] if rel == "." else rel.split(os.sep)
    if parts and parts[0] == "..":
        raise ValueError(f"private path outside {root}: {path}")
    cur = root
    for comp in [None] + parts:
        if comp is not None:
            cur = os.path.join(cur, comp)
        if not os.path.isdir(cur):
            os.mkdir(cur, 0o700)
        os.chmod(cur, 0o700)


def _write_private(name: str, arrays: dict) -> str:
    _mkdir_private(PRIV_DIR)
    path = os.path.join(PRIV_DIR, name)
    bio = io.BytesIO()
    np.savez(bio, **arrays)
    data = bio.getvalue()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return hashlib.sha256(data).hexdigest()


def _np(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(np.uint16)
    return t.numpy()


def _safe(s) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))[:80]


# ------------------------------------------------------------------ 1. bound-launch recorder (host only)
def _launch_fields(binding) -> dict:
    mb = getattr(binding, "mixed_trellis_binding", None)
    launch = getattr(mb, "launch", None)
    out = {}
    for f in LAUNCH_FIELDS:
        v = getattr(launch, f, None)
        out[f] = v if isinstance(v, (int, bool, str)) or v is None else str(v)
    out["_has_launch"] = launch is not None
    return out


def bind_signature(layer, which: str, m: int, binding, runtime) -> dict:
    """Plain-JSON signature of one bound launch (pure; tested on CPU)."""
    plan_kind = which
    try:
        if which.endswith("_tile"):
            plan_kind = which[:-5]
        cap = runtime.get("max_trellis_m") if plan_kind == "decode" else runtime.get("max_batched_tokens")
    except Exception:  # noqa: BLE001
        cap = None
    sig = dict(launch=_launch_fields(binding), plan_kind=plan_kind, capacity=cap,
               min_trellis_m=runtime.get("min_trellis_m") if hasattr(runtime, "get") else None,
               max_trellis_m=runtime.get("max_trellis_m") if hasattr(runtime, "get") else None,
               max_batched_tokens=runtime.get("max_batched_tokens") if hasattr(runtime, "get") else None,
               weight_tile_config=str(getattr(layer, "exl3_trellis_tile_config", None)))
    return sig


def hook_bind(layer, which: str, m: int, binding, runtime) -> None:
    """exl3._apply_rank_sliced, just before runtime["api"].run(binding=binding).  Host only; never raises."""
    if not ENABLED:
        return
    try:
        name = getattr(layer, "layer_name", None) or ""
        L = _layer_idx(name)
        try:
            from vllm.model_executor.layers.quantization import exl3 as _e
            draft = bool(_e._is_draft_layer(layer))
        except Exception:  # noqa: BLE001
            draft = any(w in str(name).lower() for w in ("draft", "mtp", "dflash"))
        key = (L, which, int(m), draft)
        S.bind_n[key] = S.bind_n.get(key, 0) + 1
        sig = bind_signature(layer, which, int(m), binding, runtime)
        prev = S.bind_sig.get(key)
        if prev == sig:
            return
        S.bind_sig[key] = sig
        _log(dict(kind="bind", layer=L, layer_name=str(name), plan=which, m=int(m), draft=draft,
                  changed=prev is not None, capturing=_capturing(), phase=_phase(), **sig), prefix="bind")
    except Exception as exc:  # noqa: BLE001
        _err("hook_bind", exc)


# ------------------------------------------------------------------ 2. registration (capture_model, before capture)
def register(runner) -> None:
    """Called by the forcewin port's register (GPUModelRunner.capture_model, before any graph is captured)."""
    if not ENABLED or S.registered:
        return
    try:
        r = _rank()
        vp = os.path.dirname(os.path.abspath(__file__))
        base = dict(kind="init", schema=SCHEMA, pid=os.getpid(), rank=r, nslot=NSLOT, wpos=WPOS, mmax=MMAX,
                    moe_layers=list(MOE_LAYERS), hid_layers=list(HID_LAYERS), hid_every=HID_EVERY,
                    logit_every=LOGIT_EVERY, tile_env=os.environ.get("GLM53_P4_TILE", ""),
                    env={k: v for k, v in sorted(os.environ.items())
                         if k.startswith(("B12X", "VLLM_EXL3", "VLLM_GLM53", "GLM53", "FORCEWIN", "VLLM_ADAPTIVE"))},
                    file_sha256={p: _sha_file(os.path.join(vp, p)) for p in (
                        "p4_hooks.py", "forcewin_hooks.py", "model_executor/layers/quantization/exl3.py",
                        "models/glm5next/nvidia/model.py", "v1/worker/gpu/model_runner.py",
                        "v1/worker/gpu/spec_decode/rejection_sampler.py", "v1/worker/gpu/sample/sampler.py")})
        issues = []
        names = {}
        tile_by_layer = {}
        for name, mod in runner.model.named_modules():
            low = name.lower()
            if any(w in low for w in ("draft", "mtp", "dflash", "eagle", "spec")):
                continue
            cls = type(mod).__name__
            if cls == "Glm5NextDecoderLayer":
                L = int(getattr(mod, "layer_idx", _layer_idx(name)))
                k = K_MOE.get(L) if getattr(mod, "_mlp_is_moe", False) else None
                if k is None:
                    k = K_HID.get(L)
                if k is not None:
                    if k in names.values():
                        issues.append(("duplicate", L, name))
                        continue
                    if getattr(mod, "is_sequence_parallel", False):
                        issues.append(("sequence_parallel", L))
                    if not getattr(mod, "mhc", False):
                        issues.append(("no_mhc", L))
                    S.ids[id(mod)] = k
                    names[name] = k
            elif cls == "Glm5NextModel":
                if K_FINAL in names.values():
                    issues.append(("duplicate_model", name))
                    continue
                S.ids[id(mod)] = K_FINAL
                names[name] = K_FINAL
            if hasattr(mod, "exl3_trellis_weights") and "mlp.experts" in name:
                tile_by_layer[_layer_idx(name)] = str(getattr(mod, "exl3_trellis_tile_config", None))
        if sorted(names.values()) != list(range(len(STAGE_NAMES))):
            issues.append(("stages_found", sorted(names.values())))
        hidden = int(getattr(getattr(runner, "model_config", None), "get_hidden_size", lambda: 4096)())
        dev = runner.device
        S.dev = dev
        S.max_reqs = int(runner.max_num_reqs)
        try:
            S.vocab = int(runner.model_config.get_vocab_size())
        except Exception:  # noqa: BLE001
            S.vocab = 154880
        if r == 0:
            _alloc(dev, hidden, S.vocab, S.max_reqs)
        S.registered = True
        install_graph_label()
        _log(dict(base, stage_modules=names, issues=issues, hidden=hidden, vocab=S.vocab, max_reqs=S.max_reqs,
                  weight_tile_by_layer=tile_by_layer, n_moe_layers_with_weights=len(tile_by_layer),
                  stores_mib=round(_bytes() / 2 ** 20, 1),
                  note="stages and stores allocated before CUDA-graph capture; no poller (flush at removal only)"))
        if r == 0:
            _mkdir_private(PRIV_DIR)
    except Exception as exc:  # noqa: BLE001
        _err("register", exc)


def _sha_file(path):
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()
    except OSError:
        return None


def _alloc(dev, hidden: int, vocab: int, max_reqs: int) -> None:
    """Fixed device buffers (shared by register() and the CPU tests)."""
    ns = len(STAGE_NAMES)
    S.stage = torch.zeros((ns, MMAX, hidden), dtype=torch.bfloat16, device=dev)
    S.stage_n = torch.zeros(ns, dtype=torch.int64, device=dev)
    S.store = torch.zeros((3, NSLOT + 1, WPOS, hidden), dtype=torch.bfloat16, device=dev)
    S.have = torch.zeros((3, NSLOT + 1, WPOS), dtype=torch.int32, device=dev)
    nh = (WPOS + HID_EVERY - 1) // HID_EVERY
    S.hstore = torch.zeros((3, NSLOT + 1, nh, hidden), dtype=torch.bfloat16, device=dev)
    S.hhave = torch.zeros((3, NSLOT + 1, nh), dtype=torch.int32, device=dev)
    nl = (WPOS + LOGIT_EVERY - 1) // LOGIT_EVERY
    S.lstage = torch.zeros((NSLOT + 1, vocab), dtype=torch.float32, device=dev)
    S.lstage_o = torch.full((NSLOT + 1,), -1, dtype=torch.int64, device=dev)
    S.lstore = torch.zeros((NSLOT + 1, nl, vocab), dtype=torch.float32, device=dev)
    S.lhave = torch.zeros((NSLOT + 1, nl), dtype=torch.int32, device=dev)
    S.slot_of = torch.full((max_reqs + 1,), NSLOT, dtype=torch.int64, device=dev)
    S.steprec = torch.full((NSTEPREC, max_reqs, len(STEPREC_COLS)), -1, dtype=torch.int32, device=dev)
    S.free = list(range(NSLOT))


def _bytes() -> int:
    n = 0
    for k in ("stage", "stage_n", "store", "have", "hstore", "hhave", "lstage", "lstage_o", "lstore", "lhave",
              "slot_of", "steprec"):
        t = getattr(S, k, None)
        if torch.is_tensor(t):
            n += t.numel() * t.element_size()
    return n


def _ready() -> bool:
    return ENABLED and S.stage is not None


# ------------------------------------------------------------------ in-graph hooks (fixed-shape device ops only)
def _put(k: int, x: torch.Tensor) -> None:
    m = int(x.shape[0])
    if m > MMAX or x.dim() != 2 or int(x.shape[1]) != int(S.stage.shape[2]) or x.dtype != S.stage.dtype:
        # prefill chunk (always eager: capture sizes are <= MMAX) or an unexpected shape: not staged
        S.stage_n[k:k + 1].fill_(0)
        return
    S.stage[k].narrow(0, 0, m).copy_(x)
    S.stage_n[k:k + 1].fill_(m)


def hook_moe_out(layer, x) -> None:
    """Glm5NextDecoderLayer.forward, right after the MoE block (output after the TP all-reduce)."""
    if not _ready():
        return
    k = S.ids.get(id(layer))
    if k is None or k > 2:
        return
    try:
        _put(k, x)
    except Exception as exc:  # noqa: BLE001
        _err("hook_moe_out", exc)


def hook_hidden(layer, hidden_states, residual, post, comb) -> None:
    """Glm5NextModel.forward, after each decoder layer: contracted post-layer hidden at layers 12 and 34
    (read-only materialisation, as the DFlash aux tap does)."""
    if not _ready():
        return
    k = S.ids.get(id(layer))
    if k is None or k not in (3, 4):
        return
    try:
        m = int(hidden_states.shape[0])
        if m > MMAX:
            S.stage_n[k:k + 1].fill_(0)
            return
        if getattr(layer, "mhc", False) and post is not None:
            from vllm.model_executor.layers.mhc import hc_contract
            h = hc_contract(layer.hc_post(hidden_states, residual, post, comb), layer.n)
        else:
            h = hidden_states
        _put(k, h)
    except Exception as exc:  # noqa: BLE001
        _err("hook_hidden", exc)


def hook_final(model, hidden_states) -> None:
    """Glm5NextModel.forward, right after the final norm."""
    if not _ready():
        return
    if S.ids.get(id(model)) != K_FINAL:
        return
    try:
        _put(K_FINAL, hidden_states)
    except Exception as exc:  # noqa: BLE001
        _err("hook_final", exc)


# ------------------------------------------------------------------ graph-mode label (as captured-p3b install())
def install_graph_label() -> None:
    try:
        from vllm.v1.worker.gpu import cudagraph_utils as cgmod
    except Exception as exc:  # noqa: BLE001
        _err("install_graph_label", exc)
        return
    for cls in (getattr(cgmod, "CudaGraphManager", None), getattr(cgmod, "ModelCudaGraphManager", None)):
        if cls is None or getattr(cls, "_p4_hooks", False):
            continue
        cls._p4_hooks = True
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


# ------------------------------------------------------------------ 3/4. bridge from the forcewin port (eager)
def on_add(req_idx: int, tag: str, W: int) -> None:
    if not _ready():
        return
    try:
        old = S.slot_host.pop(req_idx, None)
        if old is not None and old < NSLOT:
            S.free.append(old)
        slot = S.free.pop(0) if S.free else NSLOT
        if slot == NSLOT:
            S.no_slot += 1
            _log(dict(kind="no_slot", req_idx=req_idx, tag=tag))
        S.slot_host[req_idx] = slot
        S.tag_of[req_idx] = tag
        if slot < NSLOT:
            S.have[:, slot].zero_()
            S.hhave[:, slot].zero_()
            S.lhave[slot].zero_()
            S.lstage_o[slot] = -1
        S.slot_of[req_idx] = slot
        while req_idx in S.pending_reset:          # a deferred reset must not clobber this new assignment
            S.pending_reset.remove(req_idx)
    except Exception as exc:  # noqa: BLE001
        _err("on_add", exc)


def on_batch(input_batch, step: int) -> None:
    """prepare_attn of a real batch (forcewin on_batch): host step log; first real batch => serving."""
    if not ENABLED:
        return
    try:
        if S.registered and not _capturing():
            S.serving = True
            while S.pending_reset:                     # deferred slot_of resets (refused removals inside a capture)
                S.slot_of[S.pending_reset.pop()] = NSLOT
        nr = int(input_batch.num_reqs)
        npf = 0
        try:
            if input_batch.is_prefilling_np is not None:
                npf = int(np.sum(input_batch.is_prefilling_np[:nr]))
        except Exception:  # noqa: BLE001
            npf = -1
        S.last_step = int(step)
        S.mode = "none"
        S.steplog.append([int(step), round(time.time(), 4), nr, int(input_batch.num_tokens),
                          int(getattr(input_batch, "num_tokens_after_padding", input_batch.num_tokens)),
                          int(getattr(input_batch, "num_draft_tokens", 0) or 0), npf, ""])
    except Exception as exc:  # noqa: BLE001
        _err("on_batch", exc)


def on_logits(sc: dict, logits: torch.Tensor, input_batch) -> None:
    """forcewin on_logits (BEFORE the sampler): remember token rows; stage full logits rows with o % 64 == 0."""
    if not ENABLED:
        return
    try:
        if S.steplog and S.steplog[-1][0] == S.last_step:
            S.steplog[-1][7] = S.mode
        if not _ready():
            return
        L = int(sc["L"])
        S.tok = input_batch.logits_indices[:L].to(torch.int64).clone()
        slot = S.slot_of[sc["ridx"]]
        o = sc["o"]
        cand = sc["fwin"] & (torch.remainder(o, LOGIT_EVERY) == 0) & (slot < NSLOT)
        tgt = torch.where(cand, slot, torch.full_like(slot, NSLOT))
        S.lstage.index_copy_(0, tgt, logits.float())
        S.lstage_o.index_copy_(0, tgt, torch.where(cand, o, torch.full_like(o, -1)))
    except Exception as exc:  # noqa: BLE001
        _err("on_logits", exc)


def on_commit(sc: dict, valid_flat: torch.Tensor) -> None:
    """forcewin _commit_rows: copy the recorded rows (valid_flat; each output index once) into the stores."""
    if not _ready() or S.tok is None:
        return
    try:
        o = sc["o"]
        slot = S.slot_of[sc["ridx"]]
        tok = S.tok
        base = valid_flat & (slot < NSLOT) & (o >= 0) & (o < WPOS)
        tokc = tok.clamp(0, MMAX - 1)
        sink = torch.full_like(slot, NSLOT)
        zero = torch.zeros_like(o)
        for k in range(3):
            ok = base & (tok < S.stage_n[k])
            ts, to = torch.where(ok, slot, sink), torch.where(ok, o, zero)
            S.store[k][ts, to] = S.stage[k][tokc]
            S.have[k][ts, to] += ok.to(torch.int32)
        oh = torch.div(o, HID_EVERY, rounding_mode="floor")
        hsel = base & (torch.remainder(o, HID_EVERY) == 0)
        for j, k in enumerate((3, 4, 5)):
            ok = hsel & (tok < S.stage_n[k])
            ts, to = torch.where(ok, slot, sink), torch.where(ok, oh, zero)
            S.hstore[j][ts, to] = S.stage[k][tokc]
            S.hhave[j][ts, to] += ok.to(torch.int32)
        ol = torch.div(o, LOGIT_EVERY, rounding_mode="floor")
        okl = base & (torch.remainder(o, LOGIT_EVERY) == 0) & (S.lstage_o[slot.clamp(max=NSLOT)] == o)
        ts, to = torch.where(okl, slot, sink), torch.where(okl, ol, zero)
        S.lstore[ts, to] = S.lstage[slot.clamp(max=NSLOT)]
        S.lhave[ts, to] += okl.to(torch.int32)
    except Exception as exc:  # noqa: BLE001
        _err("on_commit", exc)


def on_step(step: int, idx: torch.Tensor, o0: torch.Tensor, n_rows: torch.Tensor, acc: torch.Tensor) -> None:
    """forcewin override / override_single: per-request composition of this step (device ring, eager)."""
    if not _ready():
        return
    try:
        nr = int(idx.shape[0])
        rec = torch.stack([torch.full_like(idx, int(step)), idx.to(torch.int64), o0.to(torch.int64),
                           n_rows.to(torch.int64), acc.to(torch.int64)], dim=1).to(torch.int32)
        row = S.steprec[int(step) % NSTEPREC]
        row.fill_(-1)
        row[:nr] = rec
    except Exception as exc:  # noqa: BLE001
        _err("on_step", exc)


# ------------------------------------------------------------------ 5. flush (request removal; never in capture)
def flush_allowed() -> bool:
    return S.serving and not _capturing()


def flush_steps() -> None:
    """Write the host step log (public: counts, times, graph mode) and the device step ring rows since the last
    flush (PRIVATE: they join output indices with tags)."""
    if not ENABLED or not flush_allowed():
        return
    try:
        if S.steplog:
            _log(dict(kind="steps", cols=["step", "t", "num_reqs", "num_tokens", "num_tokens_padded",
                                          "num_draft_tokens", "n_prefilling", "graph_mode"], rows=S.steplog),
                 prefix="steps")
            S.steplog = []
        if S.steprec is None:
            return
        lo, hi = S.step_flushed + 1, S.last_step
        if hi < lo:
            return
        if hi - lo + 1 > NSTEPREC:
            _log(dict(kind="steprec_overrun", lo=lo, hi=hi))
            lo = hi - NSTEPREC + 1
        rows = [(s % NSTEPREC) for s in range(lo, hi + 1)]
        host = S.steprec[torch.tensor(rows, device=S.steprec.device)].cpu().numpy()
        S.step_flushed = hi
        _write_private(f"steprec_{lo:07d}_{hi:07d}_r{_rank()}.npz",
                       dict(rows=host, cols=np.array(STEPREC_COLS), tags=np.array(
                           [f"{k}:{v}" for k, v in sorted(S.tag_of.items())])))
    except Exception as exc:  # noqa: BLE001
        _err("flush_steps", exc)


def on_remove(req_idx: int, tag: str, W: int) -> None:
    """forcewin remove_request: write this request's stores and free its slot."""
    if not ENABLED:
        return
    if _capturing():   # precheck-1/2 fix: refuse before any device write (slot_of is a device tensor; an H2D write
        # inside a capture fails).  Release the slot host-side now; defer the device reset to the next batch.
        _log(dict(kind="flush_refused", tag=tag, serving=S.serving, capturing=True))
        slot = S.slot_host.pop(req_idx, None)
        if slot is not None and slot < NSLOT:
            S.free.append(slot)
            S.pending_reset.append(req_idx)
        S.tag_of.pop(req_idx, None)
        return
    slot = S.slot_host.pop(req_idx, None)
    try:
        if slot is None or not _ready():
            return
        S.slot_of[req_idx] = NSLOT
        if slot >= NSLOT:
            return
        if not flush_allowed():
            _log(dict(kind="flush_refused", tag=tag, serving=S.serving, capturing=_capturing()))
            return
        n = max(1, min(int(W), WPOS))
        nh = (n + HID_EVERY - 1) // HID_EVERY
        nl = (n + LOGIT_EVERY - 1) // LOGIT_EVERY
        arrays = dict(moe=_np(S.store[:, slot, :n]), moe_have=_np(S.have[:, slot, :n]),
                      hid=_np(S.hstore[:, slot, :nh]), hid_have=_np(S.hhave[:, slot, :nh]),
                      logits=_np(S.lstore[slot, :nl]), logits_have=_np(S.lhave[slot, :nl]),
                      stage_names=np.array(STAGE_NAMES), hid_names=np.array(("hid12", "hid34", "final")))
        digest = _write_private(f"p4_{_safe(tag)}_r{_rank()}.npz", arrays)
        hv = arrays["moe_have"]
        S.flushed.append(tag)
        _log(dict(kind="flushed", tag=tag, W=int(W), sha256=digest,
                  shapes={k: list(v.shape) for k, v in arrays.items()},
                  moe_have={"1": int((hv == 1).sum()), ">1": int((hv > 1).sum()), "0": int((hv == 0).sum())},
                  hid_have_1=int((arrays["hid_have"] == 1).sum()), logits_have_1=int((arrays["logits_have"] == 1).sum())))
        flush_steps()
        write_status()
    except Exception as exc:  # noqa: BLE001
        _err("on_remove", exc)
    finally:
        if slot is not None and slot < NSLOT:
            S.free.append(slot)
        S.tag_of.pop(req_idx, None)


def write_status() -> None:
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        st = dict(t=round(time.time(), 2), rank=_rank(), registered=S.registered, serving=S.serving,
                  flushed=len(S.flushed), flushed_tags=list(S.flushed), errors=S.errors, no_slot=S.no_slot,
                  slots_in_use=NSLOT - len(S.free), bind_keys=len(S.bind_sig), last_step=S.last_step)
        tmp = os.path.join(LOG_DIR, f".status_r{_rank()}.tmp")
        with open(tmp, "w") as fh:
            json.dump(st, fh)
        os.replace(tmp, os.path.join(LOG_DIR, f"status_r{_rank()}.json"))
    except Exception as exc:  # noqa: BLE001
        _err("status", exc)
