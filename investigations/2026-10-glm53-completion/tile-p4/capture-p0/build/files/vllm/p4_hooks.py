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
  6. Immutable request identity (schema local-p4/2, stage-0 package): ``on_add`` gives every keyed request a
     per-process monotonic id (uid, never reused) and a 31-bit tag hash, logged once as a public "req" record
     (uid, tag, req_idx).  Both are written to device tables indexed by req_idx at add time (eager) and gathered
     into every step-ring row by ``on_step`` (eager), i.e. captured AT RECORD TIME, so a step row stays attributable
     after the req_idx is reused.  Removal resets the tables to -1 (a later unkeyed occupant of the index reads -1).
     The run-1 flush-time ``tags`` map is still written for compatibility, but the analysis no longer uses it.
  7. Stage-0 recording (GLM53_P4_S0=on, in addition to GLM53_P4=on; STAGE0-PROTOCOL.md).  Off: none of this runs and
     every captured graph is the run-1 graph.
     a. Scheduler execution log (rank 0, host only, in on_batch / on_logits): per real step the batch rows in batch
        order (engine request id, req_idx, uid, tag, scheduled tokens, computed tokens, computed prefill tokens,
        prefill length, prefilling flag, draft tokens), the step's real / padded token and request counts, the graph
        mode, and the MoE plans bound by eager launches in the step (graph replays: resolved offline from the
        capture-time bind records at the padded size).  Flushed with the step log (log/sched_r0_*.jsonl).
     b. Per-layer hidden-state digests (rank 0): ``hook_hidden`` (after every target decoder layer) reduces each
        token row of the layer's output state (hidden_states, residual, mHC post, comb) to two int64 digests
        (wrap-free int64 sums of the raw 16/32-bit words; the second weights the hidden words by a fixed sequence).
        Fixed-shape device ops into buffers allocated before capture.  The state after layer L is the state
        entering layer L+1; layer 0's input is the embedding of the (forced, identical) token.  Stored per recorded
        position (every position) like the MoE stores.
     c. Eager prefill / mixed steps (more than MMAX tokens; never captured, capture sizes <= 64): the stages, the
        MoE / hidden stores and the layer digests take the step's logits rows (gathered by logits_indices, at most
        8 x 4 rows), so positions measured in mixed steps (o = 0 and the first decode steps of a co-batched pass)
        are stored too.  Run 1 left them unstaged.
     d. Window-entry state digests (BOTH ranks): for each forced request, before the forward of every step in which
        it is prefilling and of its first non-prefilling step ("entry"), every KV-cache tensor of every KV-cache
        group (MLA KV, indexer K, KDA conv / recurrent state, draft caches) is digested per LOGICAL block of the
        request's block-table row (blocks addressed by index, so digests compare across replicates whose physical
        block ids differ) plus logical block 0 (the shared null block); the logical block ids are recorded too.
        At most NE_POINT points per request: the first NE_POINT - 1 prefill steps and always the entry step (further
        prefill steps are counted, not digested).  Device ops issued eagerly in on_batch (stream-ordered after the
        previous forward); copied to host at removal (PRIVATE/entry_<tag>_r<rank>.npz).  A tensor whose first dimension is not a multiple of num_blocks is not
        covered and is listed in the public "entry_init" record.
     e. Block provenance (rank 0): at the removal of every keyed request (warm requests are keyed with an empty
        forced list in stage 0) its logical block ids per KV-cache group are logged publicly ("blocks" record), so
        the analysis can tell which earlier request computed each cached block a forced request reused.

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
  GLM53_P4_S0       "" (off) | "on": stage-0 recording (section 7)
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
SCHEMA = "local-p4/2"
S0 = ENABLED and os.environ.get("GLM53_P4_S0", "").strip().lower() == "on"
_cap0 = None  # local-cap0
_CAP0_ERR = None  # local-cap0
if S0 and os.environ.get("GLM53_CAP0", "").strip().lower() == "on":  # local-cap0
    try:  # local-cap0
        from vllm import cap0_hooks as _cap0  # local-cap0
        if not _cap0.ON:  # local-cap0
            _cap0 = None  # local-cap0
    except Exception as _exc:  # noqa: BLE001  # local-cap0
        _cap0, _CAP0_ERR = None, repr(_exc)[:300]  # local-cap0
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
STEPREC_COLS = ("step", "req_idx", "o0", "n_rows", "accepted", "uid", "taghash")   # uid/taghash: schema 2
NE_POINT = 8        # stage 0: entry-digest points per forced request (prefill steps + the entry step)
NE_BLK = 48         # stage 0: logical blocks digested per request and group (+1 column: logical block 0)
NE_TENS = 256       # stage 0: KV-cache tensors digested per rank
NE_SLOT = 16        # stage 0: forced requests with entry digests in flight
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
        # schema 2: immutable request identity
        self.uid_ctr = 0
        self.uid_host = {}          # req_idx -> uid of the keyed request now holding it
        self.uid_of = None          # device [max_reqs + 1] int32 (-1 = none), rank 0
        self.th_of = None           # device [max_reqs + 1] int32 tag hash (-1 = none), rank 0
        self.runner = None
        # stage 0
        self.big = False            # this step is an eager step with more than MMAX tokens (stage its logits rows)
        self.big_m = -1             # its padded token count (a later unlogged step of another size never gathers)
        self.lrows = None           # device logits_indices of the step (eager big steps only)
        self.nlog = 0
        self.step_binds = set()
        self.sched = []
        self.cur_sched = None
        self.lid = {}               # id(target decoder layer) -> row in the layer-digest buffers
        self.lid_layers = []
        self.ldig = None
        self.ldig_n = None
        self.ldstore = None
        self.ldhave = None
        self.wdig = None
        self.eready = False
        self.e_tens = []            # (name, group, tensor, pages per logical block)
        self.e_ar = {}
        self.edig = None
        self.eblk = None
        self.eslot_host = {}
        self.efree = list(range(NE_SLOT))
        self.epoints = {}           # req_idx -> list of point metadata (host)
        self.e_entry_done = set()
        self.e_dropped = 0
        self.e_dropped_pts = 0      # prefill points beyond NE_POINT - 1 (not digested; counted)


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
        if S0 and S.serving and not draft:
            S.step_binds.add((which, int(m)))
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
        lid_all = {}
        for name, mod in runner.model.named_modules():
            low = name.lower()
            if any(w in low for w in ("draft", "mtp", "dflash", "eagle", "spec")):
                continue
            cls = type(mod).__name__
            if cls == "Glm5NextDecoderLayer":
                L = int(getattr(mod, "layer_idx", _layer_idx(name)))
                lid_all[id(mod)] = L
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
        S.runner = runner
        if r == 0:
            _alloc(dev, hidden, S.vocab, S.max_reqs)
            if S0:
                order = sorted(set(lid_all.values()))
                S.lid_layers = order
                S.lid = {k: order.index(v) for k, v in lid_all.items()}
                _alloc_ldig(dev, len(order), hidden)
        e_cov = _entry_setup(runner) if S0 else None
        if _cap0 is not None:  # local-cap0
            _cap0.register(runner)  # local-cap0
        elif _CAP0_ERR is not None:  # local-cap0
            _log(dict(kind="p4_error", where="cap0.import", err=_CAP0_ERR))  # local-cap0
        S.registered = True
        install_graph_label()
        _log(dict(base, stage_modules=names, issues=issues, hidden=hidden, vocab=S.vocab, max_reqs=S.max_reqs,
                  weight_tile_by_layer=tile_by_layer, n_moe_layers_with_weights=len(tile_by_layer),
                  stores_mib=round(_bytes() / 2 ** 20, 1), s0=S0, steprec_cols=list(STEPREC_COLS),
                  n_digest_layers=len(S.lid_layers),
                  note="stages and stores allocated before CUDA-graph capture; no poller (flush at removal only)"))
        if S0:
            _log(dict(kind="entry_init", rank=r, covered=sum(1 for c in e_cov if c.get("covered")),
                      uncovered=[c for c in e_cov if not c.get("covered")][:64], tensors=e_cov[:NE_TENS + 64],
                      n_groups=_ngroups(runner), num_blocks=_num_blocks(runner), ne_blk=NE_BLK, ne_point=NE_POINT))
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
    S.uid_of = torch.full((max_reqs + 1,), -1, dtype=torch.int32, device=dev)
    S.th_of = torch.full((max_reqs + 1,), -1, dtype=torch.int32, device=dev)
    S.free = list(range(NSLOT))


def _alloc_ldig(dev, nl: int, hidden: int) -> None:
    """Stage 0: per-layer row digests (in-graph stage) and their per-request stores (rank 0, before capture)."""
    S.ldig = torch.zeros((max(nl, 1), MMAX, 2), dtype=torch.int64, device=dev)
    S.ldig_n = torch.zeros(max(nl, 1), dtype=torch.int64, device=dev)
    S.ldstore = torch.zeros((NSLOT + 1, WPOS, max(nl, 1), 2), dtype=torch.int64, device=dev)
    S.ldhave = torch.zeros((NSLOT + 1, WPOS), dtype=torch.int32, device=dev)
    # fixed odd weights for the second digest (any fixed sequence works; this one is reproducible everywhere)
    S.wdig = (torch.remainder(torch.arange(hidden, dtype=torch.int64) * 2654435761, 2 ** 31) * 2 + 1).to(dev)


def _ngroups(runner) -> int:
    try:
        return len(runner.kv_cache_config.kv_cache_groups)
    except Exception:  # noqa: BLE001
        return 0


def _num_blocks(runner) -> int:
    try:
        return int(runner.kv_cache_config.num_blocks)
    except Exception:  # noqa: BLE001
        return 0


def _kv_parts(kv, suf=""):
    """The tensors of one layer's kv_cache: a tensor, a one-element (virtual-engine) list, or a tuple of tensors
    (KDA conv / recurrent state)."""
    if torch.is_tensor(kv):
        return [(suf, kv)]
    if isinstance(kv, (list, tuple)):
        if len(kv) == 1:
            return _kv_parts(kv[0], suf)
        out = []
        for k, x in enumerate(kv):
            out += _kv_parts(x, f"{suf}#{k}")
        return out
    return []


def _entry_setup(runner) -> list:
    """Stage 0 (both ranks, register time, before capture): enumerate the KV-cache tensors by group and allocate
    the entry-digest buffers.  Returns the coverage list for the public entry_init record."""
    cov = []
    try:
        kvc = runner.kv_cache_config
        nblk = int(kvc.num_blocks)
        cc = getattr(runner, "compilation_config", None) or runner.vllm_config.compilation_config
        ctx = cc.static_forward_context
        tens = []
        for g, grp in enumerate(kvc.kv_cache_groups):
            for name in grp.layer_names:
                mod = ctx.get(name)
                parts = _kv_parts(getattr(mod, "kv_cache", None)) if mod is not None else []
                if not parts:
                    cov.append(dict(layer=name, group=g, covered=False, why="no kv tensor"))
                    continue
                for suf, t in parts:
                    n0 = int(t.shape[0]) if t.dim() > 0 else 0
                    rec = dict(layer=name + suf, group=g, shape=list(t.shape), dtype=str(t.dtype))
                    if nblk > 0 and n0 > 0 and n0 % nblk == 0 and t.element_size() in (1, 2, 4, 8) \
                            and len(tens) < NE_TENS:
                        ppb = n0 // nblk
                        tens.append((name + suf, g, t, ppb))
                        cov.append(dict(rec, covered=True, ppb=ppb))
                    else:
                        cov.append(dict(rec, covered=False, why="first dim %d vs num_blocks %d (or tensor cap)" % (
                            n0, nblk)))
        S.e_tens = tens
        dev = runner.device
        S.edig = torch.zeros((NE_SLOT, NE_POINT, max(len(tens), 1), NE_BLK + 1, 2), dtype=torch.int64, device=dev)
        S.eblk = torch.full((NE_SLOT, NE_POINT, max(_ngroups(runner), 1), NE_BLK), -1, dtype=torch.int32, device=dev)
        S.e_ar = {ppb: torch.arange(ppb, dtype=torch.int64, device=dev) for ppb in {t[3] for t in tens}}
        S.efree = list(range(NE_SLOT))
        S.eready = True
    except Exception as exc:  # noqa: BLE001
        _err("entry_setup", exc)
        cov.append(dict(layer="*", covered=False, why=repr(exc)[:200]))
    return cov


_INT_OF = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _block_digest(t: torch.Tensor, pages: torch.Tensor) -> torch.Tensor:
    """[2] int64 digest of the pages of one logical block (eager; one block-sized temporary)."""
    blk = t.index_select(0, pages)
    w = blk.view(_INT_OF[blk.element_size()]).reshape(-1)
    return torch.stack([w.sum(dtype=torch.int64), w[::3].sum(dtype=torch.int64)])


def _entry_digest(req_idx: int, eslot: int, point: int) -> None:
    """Stage 0 (eager, in on_batch, before the step's forward): digest the request's KV-cache blocks."""
    bt = S.runner.block_tables
    lids = []
    S.eblk[eslot, point].fill_(-1)
    for g in range(bt.num_kv_cache_groups):
        nbk = int(bt.num_blocks.np[g, req_idx])
        bpk = max(int(bt.blocks_per_kv_block[g]), 1)
        nlog = min(nbk // bpk, NE_BLK)
        lid = torch.div(bt.block_tables[g].gpu[req_idx, 0:nlog * bpk:bpk], bpk, rounding_mode="floor").to(torch.int64)
        if nlog:
            S.eblk[eslot, point, g, :nlog] = lid.to(torch.int32)
        lids.append(lid)
    zero = torch.zeros((), dtype=torch.int64, device=S.edig.device)
    for ti, (_name, g, t, ppb) in enumerate(S.e_tens):
        lid = lids[g] if g < len(lids) else lids[0][:0]
        ar = S.e_ar[ppb]
        n = int(lid.shape[0])
        for j in range(n + 1):
            b = lid[j] if j < n else zero
            S.edig[eslot, point, ti, j if j < n else NE_BLK] = _block_digest(t, b * ppb + ar)


def _bytes() -> int:
    n = 0
    for k in ("stage", "stage_n", "store", "have", "hstore", "hhave", "lstage", "lstage_o", "lstore", "lhave",
              "slot_of", "steprec", "uid_of", "th_of", "ldig", "ldig_n", "ldstore", "ldhave", "wdig", "edig", "eblk"):
        t = getattr(S, k, None)
        if torch.is_tensor(t):
            n += t.numel() * t.element_size()
    return n


def _ready() -> bool:
    return ENABLED and S.stage is not None


# ------------------------------------------------------------------ in-graph hooks (fixed-shape device ops only)
def _big_rows(m: int):
    """Stage 0, eager steps with more than MMAX tokens only: the step's logits rows (device index), else None.
    Never during a capture (capture sizes are <= MMAX anyway); host flags were set by on_batch before the forward."""
    if S0 and S.big and m > MMAX and m == S.big_m and S.lrows is not None and 0 < S.nlog <= MMAX \
            and not _capturing():
        return S.lrows[:S.nlog]
    return None


def _put(k: int, x: torch.Tensor) -> None:
    m = int(x.shape[0])
    rows = _big_rows(m)
    if rows is not None and x.dim() == 2:
        x = x.index_select(0, rows)
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
    if S.ldig is not None:
        L = S.lid.get(id(layer))
        if L is not None:
            try:
                _ldig_put(L, hidden_states, residual, post, comb)
            except Exception as exc:  # noqa: BLE001
                _err("hook_hidden.ldig", exc)
    k = S.ids.get(id(layer))
    if k is None or k not in (3, 4):
        return
    try:
        m = int(hidden_states.shape[0])
        rows = _big_rows(m)
        if rows is not None:
            hidden_states, residual, post, comb = (None if t is None else t.index_select(0, rows)
                                                   for t in (hidden_states, residual, post, comb))
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


def _rowsum(t: torch.Tensor, m: int) -> torch.Tensor:
    """[m] int64: per token row, the sum of the row's raw 16/32-bit words (fixed-shape, no host read)."""
    x = t.reshape(m, -1)
    return x.view(_INT_OF[x.element_size()]).sum(dim=1, dtype=torch.int64)


def _ldig_put(L: int, hidden_states, residual, post, comb) -> None:
    """Stage 0, after target layer L: two int64 digests per token row of the state entering layer L + 1."""
    m = int(hidden_states.shape[0])
    rows = _big_rows(m)
    if rows is not None:
        hidden_states, residual, post, comb = (None if t is None else t.index_select(0, rows)
                                               for t in (hidden_states, residual, post, comb))
        m = int(hidden_states.shape[0])
    if m > MMAX:
        S.ldig_n[L:L + 1].fill_(0)
        return
    d0 = _rowsum(hidden_states, m)
    for t in (residual, post, comb):
        if t is not None:
            d0 = d0 + _rowsum(t, m)
    h = hidden_states.reshape(m, -1)
    hw = h.view(_INT_OF[h.element_size()])
    d1 = (hw * S.wdig[: hw.shape[1]]).sum(dim=1)
    S.ldig[L].narrow(0, 0, m).select(1, 0).copy_(d0)
    S.ldig[L].narrow(0, 0, m).select(1, 1).copy_(d1)
    S.ldig_n[L:L + 1].fill_(m)


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
def taghash(tag: str) -> int:
    return int(hashlib.sha256(str(tag).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def on_add(req_idx: int, tag: str, W: int) -> None:
    if not ENABLED:
        return
    try:                                    # schema 2: immutable identity, both ranks (host); device tables rank 0
        S.uid_ctr += 1
        uid = S.uid_ctr
        S.uid_host[req_idx] = uid
        S.tag_of[req_idx] = tag
        th = taghash(tag)
        if S.uid_of is not None:
            S.uid_of[req_idx] = uid
            S.th_of[req_idx] = th
        _log(dict(kind="req", uid=uid, tag=tag, req_idx=int(req_idx), W=int(W), taghash=th, step=S.last_step))
        if S.eready and int(W) > 0:         # stage 0: entry-digest slot for a forced request
            old = S.eslot_host.pop(req_idx, None)
            if old is not None:
                S.efree.append(old)
            if S.efree:
                es = S.efree.pop(0)
                S.eslot_host[req_idx] = es
                S.edig[es].zero_()
                S.eblk[es].fill_(-1)
                S.epoints[req_idx] = []
                S.e_entry_done.discard(req_idx)
            else:
                S.e_dropped += 1
                _log(dict(kind="no_eslot", req_idx=req_idx, tag=tag))
    except Exception as exc:  # noqa: BLE001
        _err("on_add.identity", exc)
    if not _ready():
        return
    try:
        old = S.slot_host.pop(req_idx, None)
        if old is not None and old < NSLOT:
            S.free.append(old)
        if int(W) <= 0:                     # keyed request with nothing to store (stage-0 warm requests)
            S.slot_host[req_idx] = NSLOT
            S.slot_of[req_idx] = NSLOT
            while req_idx in S.pending_reset:
                S.pending_reset.remove(req_idx)
            return
        slot = S.free.pop(0) if S.free else NSLOT
        if slot == NSLOT:
            S.no_slot += 1
            _log(dict(kind="no_slot", req_idx=req_idx, tag=tag))
        S.slot_host[req_idx] = slot
        if slot < NSLOT:
            S.have[:, slot].zero_()
            S.hhave[:, slot].zero_()
            S.lhave[slot].zero_()
            S.lstage_o[slot] = -1
            if S.ldhave is not None:
                S.ldhave[slot].zero_()
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
                ri = S.pending_reset.pop()
                if S.slot_of is not None:
                    S.slot_of[ri] = NSLOT
                if S.uid_of is not None and ri not in S.uid_host:
                    S.uid_of[ri] = -1
                    S.th_of[ri] = -1
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
    if S0 and S.serving and not _capturing():
        try:
            _s0_batch(input_batch, int(step))
        except Exception as exc:  # noqa: BLE001
            _err("on_batch.s0", exc)


def _s0_batch(input_batch, step: int) -> None:
    """Stage 0 (host bookkeeping; device digests for forced requests): before the step's forward."""
    nr = int(input_batch.num_reqs)
    ntp = int(getattr(input_batch, "num_tokens_after_padding", input_batch.num_tokens))
    S.step_binds = set()
    S.big = ntp > MMAX
    S.big_m = ntp if S.big else -1
    S.lrows = input_batch.logits_indices if S.big else None
    try:
        S.nlog = int(input_batch.cu_num_logits_np[nr])
    except Exception:  # noqa: BLE001
        S.nlog = 0
    idx = [int(x) for x in input_batch.idx_mapping_np[:nr]]
    g = lambda name, b, d=0: (int(getattr(input_batch, name)[b])  # noqa: E731
                              if getattr(input_batch, name, None) is not None else d)
    rows = []
    for b, i in enumerate(idx):
        rows.append([str(input_batch.req_ids[b]) if b < len(input_batch.req_ids) else "", i, S.uid_host.get(i, -1),
                     S.tag_of.get(i, ""), g("num_scheduled_tokens", b), g("num_computed_tokens_np", b),
                     g("num_computed_prefill_tokens_np", b), g("prefill_len_np", b), g("is_prefilling_np", b),
                     g("num_draft_tokens_per_req", b)])
    epts = []
    if S.eready:
        for b, i in enumerate(idx):
            es = S.eslot_host.get(i)
            if es is None or i in S.e_entry_done:
                continue
            pts = S.epoints.setdefault(i, [])
            pre = bool(rows[b][8])
            if pre and len(pts) >= NE_POINT - 1:     # the last point is reserved for the entry step
                S.e_dropped_pts += 1
                continue
            _entry_digest(i, es, len(pts))
            pts.append([step, 0 if pre else 1, rows[b][5], rows[b][6], rows[b][4]])
            epts.append([S.uid_host.get(i, -1), len(pts) - 1, 0 if pre else 1])
            if not pre:
                S.e_entry_done.add(i)
    if _cap0 is not None:  # local-cap0
        _cap0.on_batch(input_batch, step, rows)  # local-cap0
    if _rank() == 0:
        S.cur_sched = dict(step=step, t=round(time.time(), 4), nt=int(input_batch.num_tokens), ntp=ntp, nr=nr,
                           nrp=int(getattr(input_batch, "num_reqs_after_padding", nr)),
                           ndraft=int(getattr(input_batch, "num_draft_tokens", 0) or 0), big=S.big, nlog=S.nlog,
                           rows=rows, epoints=epts, mode=None, binds=None)
        S.sched.append(S.cur_sched)


def on_logits(sc: dict, logits: torch.Tensor, input_batch) -> None:
    """forcewin on_logits (BEFORE the sampler): remember token rows; stage full logits rows with o % 64 == 0."""
    if not ENABLED:
        return
    try:
        if S.steplog and S.steplog[-1][0] == S.last_step:
            S.steplog[-1][7] = S.mode
        if S.cur_sched is not None and S.cur_sched.get("step") == S.last_step:
            S.cur_sched["mode"] = S.mode
            S.cur_sched["binds"] = sorted(S.step_binds)
        big = S.big
        S.big, S.big_m, S.lrows = False, -1, None        # the forward is over: no later step may gather by these rows
        if _cap0 is not None:  # local-cap0
            _cap0.after_forward()  # local-cap0
        if not _ready():
            return
        L = int(sc["L"])
        if S0 and big and L == S.nlog and 0 < L <= MMAX:       # stage 0: big eager step staged by logits row
            S.tok = torch.arange(L, dtype=torch.int64, device=logits.device)
        else:
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
        if S.ldig is not None:                                   # stage 0: per-layer digests, every position
            okd = base & (tok < S.ldig_n.min())
            ts, to = torch.where(okd, slot, sink), torch.where(okd, o, zero)
            S.ldstore[ts, to] = S.ldig[:, tokc].permute(1, 0, 2)
            S.ldhave[ts, to] += okd.to(torch.int32)
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
        ii = idx.to(torch.int64)
        rec = torch.stack([torch.full_like(idx, int(step)).to(torch.int64), ii, o0.to(torch.int64),
                           n_rows.to(torch.int64), acc.to(torch.int64), S.uid_of[ii].to(torch.int64),
                           S.th_of[ii].to(torch.int64)], dim=1).to(torch.int32)
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
        if S.sched:
            _log(dict(kind="sched", cols=["req_id", "req_idx", "uid", "tag", "n_sched", "n_computed",
                                          "n_computed_prefill", "prefill_len", "prefilling", "n_draft"],
                      steps=S.sched), prefix="sched")
            S.sched = []
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


def _log_blocks(req_idx: int, tag: str, uid: int) -> None:
    """Stage 0, rank 0, at removal (D2H allowed here): the request's logical block ids per KV-cache group."""
    bt = S.runner.block_tables
    groups = []
    for g in range(bt.num_kv_cache_groups):
        nbk = int(bt.num_blocks.np[g, req_idx])
        bpk = max(int(bt.blocks_per_kv_block[g]), 1)
        ids = bt.block_tables[g].gpu[req_idx, 0:nbk:bpk].cpu().tolist()
        groups.append([int(x) // bpk for x in ids])
    _log(dict(kind="blocks", tag=tag, uid=uid, req_idx=int(req_idx), step=S.last_step, groups=groups,
              block_sizes=[int(x) for x in getattr(bt, "block_sizes", [])]))


def _flush_entry(req_idx: int, es: int, tag: str, uid: int) -> None:
    """Stage 0, both ranks, at removal: this request's entry digests -> PRIVATE/entry_<tag>_r<rank>.npz."""
    pts = S.epoints.get(req_idx, [])
    n = max(len(pts), 1)
    arrays = dict(edig=_np(S.edig[es, :n]), eblk=_np(S.eblk[es, :n]),
                  points=np.array(pts if pts else [[-1, -1, -1, -1, -1]], dtype=np.int64),
                  point_cols=np.array(["step", "kind", "n_computed", "n_computed_prefill", "n_sched"]),
                  tensors=np.array([t[0] for t in S.e_tens] or [""]),
                  tensor_group=np.array([t[1] for t in S.e_tens] or [-1], dtype=np.int64),
                  tensor_ppb=np.array([t[3] for t in S.e_tens] or [0], dtype=np.int64))
    digest = _write_private(f"entry_{_safe(tag)}_r{_rank()}.npz", arrays)
    _log(dict(kind="entry_flushed", tag=tag, uid=uid, points=len(pts), kinds=[int(p[1]) for p in pts],
              n_tensors=len(S.e_tens), sha256=digest, points_dropped_total=S.e_dropped_pts))


def on_remove(req_idx: int, tag: str, W: int) -> None:
    """forcewin remove_request: write this request's stores and free its slot."""
    if not ENABLED:
        return
    uid = S.uid_host.pop(req_idx, -1)
    if _capturing():   # precheck-1/2 fix: refuse before any device write (slot_of is a device tensor; an H2D write
        # inside a capture fails).  Release the slot host-side now; defer the device reset to the next batch.
        _log(dict(kind="flush_refused", tag=tag, serving=S.serving, capturing=True))
        slot = S.slot_host.pop(req_idx, None)
        if slot is not None and slot < NSLOT:
            S.free.append(slot)
        S.pending_reset.append(req_idx)
        es = S.eslot_host.pop(req_idx, None)
        if es is not None:
            S.efree.append(es)
        S.epoints.pop(req_idx, None)
        S.e_entry_done.discard(req_idx)
        S.tag_of.pop(req_idx, None)
        return
    try:                                         # schema 2 identity tables; stage-0 entry digests and provenance
        if S.uid_of is not None:
            S.uid_of[req_idx] = -1
            S.th_of[req_idx] = -1
        es = S.eslot_host.pop(req_idx, None)
        if es is not None:
            try:
                if flush_allowed():
                    _flush_entry(req_idx, es, tag, uid)
                else:
                    _log(dict(kind="entry_flush_refused", tag=tag, serving=S.serving))
            finally:
                S.efree.append(es)
                S.epoints.pop(req_idx, None)
                S.e_entry_done.discard(req_idx)
        if S0 and S.runner is not None and _rank() == 0 and flush_allowed():
            _log_blocks(req_idx, tag, uid)
        if _cap0 is not None and flush_allowed():  # local-cap0
            _cap0.on_remove(req_idx, tag, uid)  # local-cap0
        if _cap0 is not None and flush_allowed() and _rank() == 0:  # local-cap0
            flush_steps()  # step + sched logs: a request without a forced slot never reaches the store flush  # local-cap0
    except Exception as exc:  # noqa: BLE001
        _err("on_remove.s0", exc)
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
        if S.ldstore is not None:
            arrays.update(ldig=_np(S.ldstore[slot, :n]), ldig_have=_np(S.ldhave[slot, :n]),
                          ldig_layers=np.array(S.lid_layers or [-1], dtype=np.int64))
        digest = _write_private(f"p4_{_safe(tag)}_r{_rank()}.npz", arrays)
        hv = arrays["moe_have"]
        S.flushed.append(tag)
        _log(dict(kind="flushed", tag=tag, uid=uid, W=int(W), sha256=digest,
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
                  slots_in_use=NSLOT - len(S.free), bind_keys=len(S.bind_sig), last_step=S.last_step,
                  s0=S0, entry_ready=S.eready, entry_dropped=S.e_dropped, entry_points_dropped=S.e_dropped_pts,
                  uid_max=S.uid_ctr)
        tmp = os.path.join(LOG_DIR, f".status_r{_rank()}.tmp")
        with open(tmp, "w") as fh:
            json.dump(st, fh)
        os.replace(tmp, os.path.join(LOG_DIR, f"status_r{_rank()}.json"))
    except Exception as exc:  # noqa: BLE001
        _err("status", exc)
