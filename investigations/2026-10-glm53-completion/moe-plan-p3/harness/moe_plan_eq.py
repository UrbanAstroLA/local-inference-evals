#!/usr/bin/env python3
"""P3 Stage 1: routed-expert decode plan vs prefill plan on identical rows.

Runs INSIDE one stock image (0.9.1 = a9e985c0 / B12x 7fcc094e, or
0.7.0 = 48e254d9 / B12x fe054789) on one GPU (cuda:0), launched by
run_p3.sh.  Design: ../PROPOSAL.md.  Reuse map and sources: README.md.

Usage:
  moe_plan_eq.py --engine v091|v070 --out OUT.json [--dry] [--smoke]
                 [--deadline-s N] [--model-dir DIR] [--dry-skip-weights]

What it does (GPU mode), per slice and layer:
  * builds the engine's rank-0 routed-expert slice through the image's own
    vLLM EXL3 code: Exl3Config.from_config + maybe_update_config,
    Exl3MoEMethod.create_weights, the image's Exl3MoEParameter weight loader,
    Exl3MoEMethod.process_weights_after_loading (-> api.plan_weights /
    api.prepare_weights exactly as vLLM calls them);
  * builds both runtime plans with Exl3MoEMethod._rank_sliced_runtime under
    the served env (decode cap 32 block_m 8; prefill cap min(2048,1024)
    block_m 64);
  * invokes a chosen plan exactly as _apply_rank_sliced does
    (Exl3MoEMethod._bind_rank_sliced + runtime["api"].run + .to(x.dtype));
  * cross-checks every fixture against Exl3MoEMethod.apply (the served
    dispatch: m<=32 decode, m>32 prefill).

--dry: CPU only, no kernels.  Builds fixtures, runs the comparison logic on
fabricated row-local outputs with injected differences (self-check), and
loads the real weight slices on CPU through the same image code up to the
point where B12x needs CUDA (prepare_weights), to verify slice/tier
bookkeeping against the served start-up logs.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from types import SimpleNamespace

T_PROCESS_START = time.time()

SCHEMA = "p3-moe-plan-eq/1"
DEFAULT_MODEL_DIR = (
    "/model-repo/snapshots/0490d2f708b12145f6516555ab066aaeb401cd21"
)
LAYERS = (3, 23, 44)  # early / middle / late MoE layers (MoE layers are 3..44)
HIDDEN = 4096
MOE_INTER = 2048
GLOBAL_EXPERTS = 288
TOPK = 8
SHAPES = (1, 8, 16, 24, 32, 40, 48)
SCALES = {"x0.25": 0.25, "x1": 1.0, "x4": 4.0}
CONCS = ("spread", "concentrated", "mixed")
WMODES = ("uniform", "peaked", "neartail")
SEEDS = (0, 1)
HEAVY_EVERY = 4        # rows r with r % 4 == 3 are heavy-tailed
HEAVY_CHANNELS = 4     # ... with 4 channels x30
HEAVY_FACTOR = 30.0
DECODE_CHUNK = 32      # decode plan capacity (VLLM_EXL3_TRELLIS_MAX_M)
# Padded cells (proposal: "padded rows are filled as the runner pads them,
# and also with non-zero garbage"): (m, real rows).  0.9.1 at 5/7 in flight:
# 20->24, 28->32.  0.7.0 at 6/7 in flight: 36->40, 42->48 (DISPATCH.md s.2).
PAD_CELLS = ((24, 20), (32, 28), (40, 36), (48, 42))
PAD_MODES = ("runner", "garbage")
SMALL_SHAPE_MS = (1, 2, 4, 8, 12, 16)
EVERSION_MS = (8, 32, 48)

# Served env.  Sources (README "Served values"): recipe.json env blocks of
# glm53-flash-exl3-k3.25@tpurtell (0.7.0) and ...+v0.9.1, both images'
# baked ENV, and the served start-up logs' plan line.
SERVED_ENV = {
    "VLLM_EXL3_TRELLIS_MIN_M": "1",
    "VLLM_EXL3_TRELLIS_MAX_M": "32",
    "VLLM_EXL3_TRELLIS_BLOCK_M": "8",
    "VLLM_EXL3_PREFILL_TRELLIS": "1",
    "VLLM_EXL3_PREFILL_BLOCK_M": "64",
    "VLLM_EXL3_PREFILL_CAPACITY": "1024",
    "B12X_EXL3_BF16_EPILOGUE": "0",
    "B12X_EXL3_BF16_GEMV": "1",
}
SERVED_UNSET = ("VLLM_EXL3_PREFILL_CHUNK",)  # unset -> 128 (log: chunk=128)
SERVED_MAX_NUM_BATCHED_TOKENS = 2048  # both logs, non-default args line 7
SERVED_MAX_NUM_SEQS = 16

SLICES = {
    # 0.9.1 served partition: TP2 rank 0, all 288 experts, half intermediate.
    "tp2r0": dict(use_ep=False, tp_size=2, tp_rank=0, ep_size=1, ep_rank=0,
                  inter=MOE_INTER // 2),
    # 0.7.0 served partition: EP2 rank 0 (linear placement: experts 0-143),
    # full intermediate, replicated input, precomposed route map.
    "ep2r0": dict(use_ep=True, tp_size=1, tp_rank=0, ep_size=2, ep_rank=0,
                  inter=MOE_INTER),
}
ENGINES = {
    "v091": dict(own="tp2r0", other="ep2r0",
                 image="sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07",
                 b12x="7fcc094edcc93af61fdfbe14300100e3204363ea",
                 exl3_sha256="3b4ca47cb1168bffebcfe3cf1db48daf86691c140cbf40f9c1f3aa6ad6a444d9",
                 small_shape_check=True),
    "v070": dict(own="ep2r0", other="tp2r0",
                 image="sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629",
                 b12x="fe054789069579e19ae5ec21f880b397bcf6575b",
                 exl3_sha256="0560a6caedd3bb0198c04cd5a1816c92872c9b4c5e2b059c54110aaf9d59d1ad",
                 small_shape_check=False),
}
# Served start-up log references (rank 0).  Membership = (bits, gate, up,
# down) expert counts per tier, as logged by exl3._prepare_projection_mixed_weights.
SERVED_LOG = {
    "v070": dict(
        file="<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+ps.server.log",
        sha256="30e9e9da2a19b4c2d1c54433cccbdd78a11fdbaf89fbce39881946f2a75bcd6f",
        slice="ep2r0",
        plan_line=238, prefill_arena_mib=334.8,
        membership={3: ((3, 126, 109, 99), (4, 18, 35, 45)),
                    23: ((3, 131, 113, 94), (4, 13, 31, 50)),
                    44: ((3, 120, 104, 88), (4, 24, 40, 56))},
        membership_lines={3: 117, 23: 159, 44: 205}),
    "v091": dict(
        file="<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.server.log",
        sha256="0e05caefa4ab0538957cbf2d90acf296a11bf5e5204e6f0d9fb2b373d35b622e",
        slice="tp2r0",
        plan_line=236, prefill_arena_mib=262.6,
        membership={3: ((3, 248, 221, 180), (4, 40, 67, 108)),
                    23: ((3, 248, 221, 180), (4, 40, 67, 108)),
                    44: ((3, 247, 220, 180), (4, 41, 68, 108))},
        membership_lines={3: 111, 23: 153, 44: 196}),
}

VLLM_ROOT = "/usr/local/lib/python3.12/dist-packages/vllm"
B12X_ROOT = "/opt/b12x/b12x"
REUSED_FILES = (
    VLLM_ROOT + "/model_executor/layers/quantization/exl3.py",
    VLLM_ROOT + "/model_executor/layers/fused_moe/expert_map_manager.py",
    VLLM_ROOT + "/model_executor/layers/fused_moe/activation.py",
    VLLM_ROOT + "/model_executor/layers/fused_moe/fused_moe_method_base.py",
    VLLM_ROOT + "/transformers_utils/repo_utils.py",
    B12X_ROOT + "/moe/fused_moe/__init__.py",
    B12X_ROOT + "/moe/fused_moe/_impl.py",
    B12X_ROOT + "/moe/fused_moe/trellis.py",
    B12X_ROOT + "/moe/_shared/kernels/w4a16/mixed_trellis.py",
    B12X_ROOT + "/moe/_shared/kernels/w4a16/kernel.py",
    B12X_ROOT + "/_lib/compiler.py",
)

REUSE_MAP = {
    "reused_from_image": [
        "exl3.Exl3Config.from_config + maybe_update_config (tensor_storage, per-projection bitrates)",
        "exl3.Exl3MoEMethod.__init__ / create_weights (parameters, TP slice dims, EP global->local map)",
        "exl3._exl3_moe_weight_loader via Exl3MoEParameter.weight_loader (TP narrowing / EP mapping)",
        "exl3.Exl3MoEMethod.process_weights_after_loading -> _prepare_projection_mixed_weights -> api.plan_weights / api.prepare_weights (+ EP route map)",
        "exl3.Exl3MoEMethod._rank_sliced_runtime (decode + prefill plans, Caps, scratch, output buffers, runtime cache)",
        "exl3.Exl3MoEMethod._bind_rank_sliced + runtime['api'].run (the plan invocation used by _apply_rank_sliced)",
        "exl3.Exl3MoEMethod.apply (served dispatch, used as a cross-check on every fixture)",
        "vllm fused_moe.expert_map_manager.determine_expert_map (EP linear placement)",
        "vllm fused_moe.MoEActivation.SILU",
    ],
    "reimplemented_in_harness": [
        "checkpoint iteration (safetensors index -> (param, expert_id, shard_id)); mirrors glm5next load_weights + build_expert_params_mapping; non-local EP experts are skipped before the read (the served loader returns False for them)",
        "a minimal RoutedExperts stand-in (nn.Module with the attributes exl3 reads) and the moe/parallel config namespaces",
        "get_current_vllm_config_or_none patched in the exl3 module during create_weights only (scheduler max_num_batched_tokens=2048, max_num_seqs=16, runner_type=generate)",
        "plan selection: the 4 lines of _apply_rank_sliced that bind+run a plan are called with an explicitly chosen plan (decode or prefill) instead of the m-based branch",
        "decode plan on m>32: rows split into <=32-row chunks, each bound/run through the decode plan, outputs concatenated (mirrors _apply_rank_sliced's tiling loop)",
        "MAX_M=16 check: _rank_sliced_runtime called under VLLM_EXL3_TRELLIS_MAX_M=16 and VLLM_EXL3_PREFILL_TRELLIS=0 (decode plan built identically; the prefill arena is not duplicated)",
        "synthetic inputs (fixtures), comparisons, digests",
    ],
}


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------

def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def seed_of(*parts) -> int:
    raw = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(raw[:8], "little") & ((1 << 63) - 1)


def short_tb(exc: BaseException, lines: int = 8) -> str:
    return "".join(traceback.format_exception(exc)[-lines:])


@contextlib.contextmanager
def env_override(values: dict[str, str | None]):
    saved = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class JsonOut:
    """Atomic incremental JSON writer (tmp + os.replace)."""

    def __init__(self, path: str, doc: dict):
        self.path = path
        self.doc = doc
        self.last = 0.0

    def write(self, force: bool = True) -> None:
        now = time.time()
        if not force and now - self.last < 20.0:
            return
        self.doc["updated_utc"] = utc()
        self.doc["elapsed_s"] = round(now - T_PROCESS_START, 1)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.doc, fh, indent=1, sort_keys=False, default=str)
        os.replace(tmp, self.path)
        self.last = now


class Deadline:
    def __init__(self, seconds: float | None):
        self.seconds = seconds

    def left(self) -> float:
        if not self.seconds:
            return float("inf")
        return self.seconds - (time.time() - T_PROCESS_START)

    def passed(self) -> bool:
        return self.left() <= 0


# --------------------------------------------------------------------------
# fixtures (CPU, deterministic, identical across both images: same torch)
# --------------------------------------------------------------------------

def fixture_id(f: dict) -> str:
    base = f"L{f['layer']}.m{f['m']}.{f['scale']}.{f['conc']}.{f['wmode']}.s{f['seed']}"
    if f.get("pad_mode"):
        base += f".pad{f['n_real']}-{f['pad_mode']}"
    return base


def panel(layers, smoke: bool) -> list[dict]:
    out: list[dict] = []
    for layer in layers:
        if smoke:
            for m in (1, 16, 32, 48):
                for conc in ("spread", "concentrated"):
                    out.append(dict(layer=layer, m=m, scale="x1", conc=conc,
                                    wmode="uniform", seed=0, kind="main"))
            for mode in PAD_MODES:
                out.append(dict(layer=layer, m=48, scale="x1", conc="mixed",
                                wmode="uniform", seed=0, kind="pad",
                                n_real=42, pad_mode=mode))
            continue
        for m in SHAPES:
            for scale in SCALES:
                for conc in CONCS:
                    for wmode in WMODES:
                        for seed in SEEDS:
                            out.append(dict(layer=layer, m=m, scale=scale,
                                            conc=conc, wmode=wmode, seed=seed,
                                            kind="main"))
        for m, n_real in PAD_CELLS:
            for conc in CONCS:
                for seed in SEEDS:
                    for mode in PAD_MODES:
                        out.append(dict(layer=layer, m=m, scale="x1", conc=conc,
                                        wmode="uniform", seed=seed, kind="pad",
                                        n_real=n_real, pad_mode=mode))
    return out


def eversion_subset(f: dict) -> bool:
    return (f["kind"] == "main" and f["m"] in EVERSION_MS and f["scale"] == "x1"
            and f["wmode"] == "uniform")


def make_fixture(torch, f: dict, ref_rms: float) -> dict:
    """x [m,H] bf16, ids [m,8] int64 (global ids), w [m,8] float32."""
    m = f["m"]
    g = torch.Generator().manual_seed(
        seed_of("p3", f["layer"], m, f["scale"], f["conc"], f["wmode"], f["seed"]))
    x = torch.randn(m, HIDDEN, generator=g, dtype=torch.float32)
    x *= ref_rms * SCALES[f["scale"]]
    heavy = []
    for r in range(m):
        if r % HEAVY_EVERY == HEAVY_EVERY - 1:
            ch = torch.randperm(HIDDEN, generator=g)[:HEAVY_CHANNELS]
            x[r, ch] *= HEAVY_FACTOR
            heavy.append(r)
    # concentrated set: 8 experts inside 0..143, local to BOTH rank-0 slices
    conc_set = torch.randperm(GLOBAL_EXPERTS // 2, generator=g)[:TOPK].to(torch.int64)
    ids = torch.empty(m, TOPK, dtype=torch.int64)
    row_conc = []
    for r in range(m):
        concentrated = f["conc"] == "concentrated" or (f["conc"] == "mixed" and r % 2 == 0)
        if concentrated:
            ids[r] = conc_set
        else:
            ids[r] = torch.randperm(GLOBAL_EXPERTS, generator=g)[:TOPK]
        row_conc.append(bool(concentrated))
    w = torch.empty(m, TOPK, dtype=torch.float32)
    for r in range(m):
        if f["wmode"] == "uniform":
            w[r] = 1.0 / TOPK
        elif f["wmode"] == "peaked":
            w[r] = 0.1 / (TOPK - 1)
            w[r, int(torch.randint(TOPK, (1,), generator=g))] = 0.9
        else:  # neartail
            w[r] = 1e-4
            w[r, int(torch.randint(TOPK, (1,), generator=g))] = 1.0 - (TOPK - 1) * 1e-4
    n_real = m
    if f.get("pad_mode"):
        n_real = f["n_real"]
        gp = torch.Generator().manual_seed(seed_of("pad", fixture_id({**f, "pad_mode": None}), f["pad_mode"]))
        if f["pad_mode"] == "runner":
            # one synthetic row repeated: models the runner's zero-filled
            # input_ids/positions producing identical hidden states
            xr = torch.randn(HIDDEN, generator=gp) * ref_rms
            ir = torch.randperm(GLOBAL_EXPERTS, generator=gp)[:TOPK]
            for r in range(n_real, m):
                x[r] = xr
                ids[r] = ir
                w[r] = 1.0 / TOPK
        else:  # garbage: independent non-zero rows at x4
            for r in range(n_real, m):
                x[r] = torch.randn(HIDDEN, generator=gp) * ref_rms * 4.0
                ids[r] = torch.randperm(GLOBAL_EXPERTS, generator=gp)[:TOPK]
                ww = torch.rand(TOPK, generator=gp) + 0.05
                w[r] = ww / ww.sum()
    xb = x.to(torch.bfloat16).contiguous()
    h = hashlib.sha256()
    h.update(xb.view(torch.int16).numpy().tobytes())
    h.update(ids.numpy().tobytes())
    h.update(w.numpy().tobytes())
    return dict(x=xb, ids=ids.contiguous(), w=w.contiguous(), n_real=n_real,
                heavy=heavy, row_conc=row_conc, digest=h.hexdigest()[:24])


# --------------------------------------------------------------------------
# comparison logic (device-agnostic)
# --------------------------------------------------------------------------

def row_digests(torch, t) -> list[str]:
    a = t.detach().to("cpu").contiguous()
    if a.dtype == torch.bfloat16:
        a = a.view(torch.int16)
    elif a.dtype == torch.float32:
        a = a.view(torch.int32)
    arr = a.numpy()
    return [hashlib.blake2b(arr[r].tobytes(), digest_size=8).hexdigest()
            for r in range(arr.shape[0])]


def bit_rows_equal(torch, a, b, rows):
    """bool tensor [len(rows)]: bitwise row equality (dtype-aware)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        raise ValueError(f"shape/dtype mismatch {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
    iv = torch.int16 if a.dtype in (torch.bfloat16, torch.float16) else torch.int32
    return (a[rows].view(iv) == b[rows].view(iv)).all(dim=1)


def compare(torch, a, b, rows: list[int], max_list: int = 64) -> dict:
    """a vs b on the given rows: bit equality and magnitudes (fp32 math)."""
    if not rows:
        return dict(rows=0)
    idx = torch.tensor(rows, dtype=torch.long, device=a.device)
    eq = bit_rows_equal(torch, a, b, idx)
    af = a[idx].float()
    bf = b[idx].float()
    d = af - bf
    absd = d.abs()
    row_l2 = d.norm(dim=1)
    ref_l2 = bf.norm(dim=1)
    row_rel = torch.where(ref_l2 > 0, row_l2 / ref_l2.clamp_min(1e-30),
                          torch.where(row_l2 > 0, torch.full_like(row_l2, float("inf")),
                                      torch.zeros_like(row_l2)))
    iv = torch.int16 if a.dtype in (torch.bfloat16, torch.float16) else torch.int32
    elem_diff = (a[idx].view(iv) != b[idx].view(iv)).sum(dim=1)
    differ = [rows[i] for i in range(len(rows)) if not bool(eq[i])]
    tot_ref = float(bf.norm())
    tot_d = float(d.norm())
    res = dict(
        rows=len(rows),
        row_bitequal_frac=round(float(eq.float().mean()), 6),
        rows_differ=differ[:max_list],
        n_rows_differ=len(differ),
        n_elem_differ=int(elem_diff.sum()),
        max_abs=float(absd.max()),
        rel_l2=(tot_d / tot_ref) if tot_ref > 0 else (0.0 if tot_d == 0 else float("inf")),
        max_row_rel_l2=float(row_rel.max()),
        nonfinite=int((~torch.isfinite(af)).sum() + (~torch.isfinite(bf)).sum()),
    )
    if differ:
        res["per_row"] = [
            dict(row=rows[i], elems=int(elem_diff[i]), max_abs=float(absd[i].max()),
                 rel_l2=float(row_rel[i]))
            for i in range(len(rows)) if not bool(eq[i])
        ][:max_list]
    return res


def within(torch, reps, rows) -> dict:
    """Repeats must be bit-equal; report spread against repeat 1."""
    out = dict(bitequal=True, spread=[])
    for k in range(1, len(reps)):
        c = compare(torch, reps[k], reps[0], rows, max_list=16)
        out["spread"].append(dict(rep=k + 1, n_rows_differ=c.get("n_rows_differ", 0),
                                  max_abs=c.get("max_abs", 0.0),
                                  rows_differ=c.get("rows_differ", [])))
        if c.get("n_rows_differ", 0):
            out["bitequal"] = False
    return out


# --------------------------------------------------------------------------
# the engine side: vLLM EXL3 objects built from the image's own code
# --------------------------------------------------------------------------

class Engine:
    def __init__(self, args, doc):
        import torch
        self.torch = torch
        self.args = args
        self.doc = doc
        self.dry = args.dry
        self.device = torch.device("cpu" if self.dry else "cuda:0")
        t0 = time.time()
        import vllm.model_executor.layers.quantization.exl3 as exl3
        from vllm.model_executor.layers.fused_moe import MoEActivation
        from vllm.model_executor.layers.fused_moe.expert_map_manager import (
            determine_expert_map,
        )
        self.exl3 = exl3
        self.MoEActivation = MoEActivation
        self.determine_expert_map = determine_expert_map
        doc["timings"]["import_s"] = round(time.time() - t0, 2)
        self.model_dir = args.model_dir
        with open(os.path.join(self.model_dir, "model.safetensors.index.json")) as fh:
            self.weight_map = json.load(fh)["weight_map"]
        self.quant_config = self._build_quant_config()
        self.runtimes: dict = {}

    # -- quant config: the served entry points ------------------------------
    def _build_quant_config(self):
        with open(os.path.join(self.model_dir, "config.json")) as fh:
            cfg = json.load(fh)
        qc = self.exl3.Exl3Config.from_config(cfg["quantization_config"])
        text = cfg.get("text_config", {})
        hf = SimpleNamespace(model_type=cfg.get("model_type"),
                             num_hidden_layers=int(text.get("num_hidden_layers", 0)))
        qc.maybe_update_config(self.model_dir, hf_config=hf)
        self.doc["quant_config"] = dict(
            standard_fused_moe=bool(qc.standard_fused_moe),
            codebook=qc.codebook, n_tensor_storage=len(qc.tensor_storage),
            layers_with_projection_bitrates=len(qc.standard_projection_bits_by_layer),
        )
        if not qc.standard_fused_moe:
            raise RuntimeError("checkpoint did not configure standard fused MoE")
        return qc

    def ref_rms(self, layer: int) -> float:
        from safetensors import safe_open
        key = f"model.language_model.layers.{layer}.post_attention_layernorm.weight"
        with safe_open(os.path.join(self.model_dir, self.weight_map[key]),
                       framework="pt", device="cpu") as fh:
            w = fh.get_tensor(key).float()
        return float(w.pow(2).mean().sqrt())

    # -- one layer's rank-0 slice --------------------------------------------
    def build_layer(self, slice_name: str, layer_idx: int) -> tuple:
        torch = self.torch
        spec = SLICES[slice_name]
        info: dict = dict(slice=slice_name, layer=layer_idx)
        par = SimpleNamespace(use_ep=spec["use_ep"], enable_eplb=False,
                              tp_rank=spec["tp_rank"], tp_size=spec["tp_size"],
                              ep_rank=spec["ep_rank"], ep_size=spec["ep_size"])
        moe = SimpleNamespace(moe_parallel_config=par, has_bias=False)
        method = self.exl3.Exl3MoEMethod(self.quant_config, moe)
        layer = torch.nn.Module()
        layer.layer_name = f"language_model.model.layers.{layer_idx}.mlp.experts"
        layer.activation = self.MoEActivation.SILU
        layer.ckpt_gate_proj_name = "gate_proj"
        layer.ckpt_up_proj_name = "up_proj"
        layer.ckpt_down_proj_name = "down_proj"
        layer.apply_router_weight_on_input = False
        layer.hidden_size = HIDDEN
        layer.use_ep = spec["use_ep"]
        if spec["use_ep"]:
            local_n, emap, _ = self.determine_expert_map(
                spec["ep_size"], spec["ep_rank"], GLOBAL_EXPERTS, "linear")
            layer.expert_map = emap.to(self.device)
        else:
            local_n, layer.expert_map = GLOBAL_EXPERTS, None
        layer.local_num_experts = int(local_n)
        info["local_num_experts"] = int(local_n)
        fake_cfg = SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=SERVED_MAX_NUM_BATCHED_TOKENS,
                max_num_seqs=SERVED_MAX_NUM_SEQS),
            model_config=SimpleNamespace(runner_type="generate", enforce_eager=False))
        # BasevLLMParameter.__init__ records the TP group rank/size on every
        # parameter (vllm/model_executor/parameter.py:65-66).  There is no TP
        # group in this one-process harness; the served values are rank 0 of a
        # TP2 group in both engines.  Exl3MoEParameter does not use them (it
        # slices with its own exl3_tp_rank/exl3_tp_size from the moe config).
        import vllm.model_executor.parameter as vparam
        orig = self.exl3.get_current_vllm_config_or_none
        orig_rank = vparam.get_tensor_model_parallel_rank
        orig_world = vparam.get_tensor_model_parallel_world_size
        self.exl3.get_current_vllm_config_or_none = lambda: fake_cfg
        vparam.get_tensor_model_parallel_rank = lambda: 0
        vparam.get_tensor_model_parallel_world_size = lambda: 2
        try:
            with torch.device(self.device):
                method.create_weights(
                    layer=layer, num_experts=int(local_n), hidden_size=HIDDEN,
                    unpadded_hidden_size=HIDDEN,
                    intermediate_size_per_partition=spec["inter"],
                    params_dtype=torch.bfloat16, weight_loader=None,
                    global_num_experts=GLOBAL_EXPERTS)
        finally:
            self.exl3.get_current_vllm_config_or_none = orig
            vparam.get_tensor_model_parallel_rank = orig_rank
            vparam.get_tensor_model_parallel_world_size = orig_world
        info["projection_mixed"] = bool(getattr(layer, "exl3_projection_mixed", False))
        info["exl3_max_num_batched_tokens"] = int(layer.exl3_max_num_batched_tokens)
        info["exl3_tp"] = [int(layer.exl3_tp_rank), int(layer.exl3_tp_size)]
        # membership as exl3 will log it (same rule as _prepare_projection_mixed_weights)
        pb = tuple(tuple(int(v) for v in vals) for vals in layer.exl3_projection_bitrates)
        tiers = sorted({b for vals in pb for b in vals})
        info["membership"] = [[b, sum(v[0] == b for v in pb), sum(v[1] == b for v in pb),
                               sum(v[2] == b for v in pb)] for b in tiers]
        # load weights through the image's loader
        t0 = time.time()
        n_loaded, n_skipped, bytes_read = self._load_layer(layer, layer_idx)
        info.update(n_tensors_loaded=n_loaded, n_tensors_skipped_nonlocal=n_skipped,
                    bytes_read=bytes_read, t_load_s=round(time.time() - t0, 2))
        return method, layer, info

    def _load_layer(self, layer, layer_idx: int):
        from safetensors import safe_open
        torch = self.torch
        proj = {"gate_proj": ("w13", "w1"), "up_proj": ("w13", "w3"),
                "down_proj": ("w2", "w2")}
        by_file: dict[str, list] = {}
        for e in range(GLOBAL_EXPERTS):
            for pname, (group, shard) in proj.items():
                for suffix in ("trellis", "suh", "svh", "mcg"):
                    key = f"model.language_model.layers.{layer_idx}.mlp.experts.{e}.{pname}.{suffix}"
                    by_file.setdefault(self.weight_map[key], []).append(
                        (key, f"{group}_{suffix}", e, shard))
        n_loaded = n_skipped = nbytes = 0
        for fname, items in sorted(by_file.items()):
            with safe_open(os.path.join(self.model_dir, fname), framework="pt",
                           device="cpu") as fh:
                for key, pattr, gid, shard in items:
                    param = getattr(layer, pattr)
                    if param.map_global_expert_id(gid) < 0:
                        n_skipped += 1  # served loader returns False here
                        continue
                    t = fh.get_tensor(key)
                    nbytes += t.numel() * t.element_size()
                    param.weight_loader(param, t, f"{layer.layer_name}.{pattr}",
                                        expert_id=gid, shard_id=shard)
                    n_loaded += 1
        if not self.dry:
            torch.cuda.synchronize()
        return n_loaded, n_skipped, nbytes

    def prepare_layer(self, method, layer, info: dict) -> None:
        torch = self.torch
        t0 = time.time()
        method.process_weights_after_loading(layer)
        torch.cuda.synchronize()
        info["t_prepare_s"] = round(time.time() - t0, 2)
        tw = layer.exl3_trellis_weights
        plan = getattr(tw, "plan", None)
        info["weight_plan"] = plan_summary(plan)
        info["route_map"] = (None if layer.exl3_projection_route_map is None else
                             dict(numel=int(layer.exl3_projection_route_map.numel()),
                                  n_local=int((layer.exl3_projection_route_map >= 0).sum())))

    def runtime(self, method, layer, x, ids, name: str, env: dict | None = None) -> dict:
        torch = self.torch
        if name in self.runtimes:
            return self.runtimes[name]
        t0 = time.time()
        with env_override(env or {}):
            rt = method._rank_sliced_runtime(layer, x, ids)
        torch.cuda.synchronize()
        t = time.time() - t0
        meta = dict(t_plan_compile_s=round(t, 2),
                    min_trellis_m=rt["min_trellis_m"], max_trellis_m=rt["max_trellis_m"],
                    max_batched_tokens=rt["max_batched_tokens"],
                    parity_rows=rt["parity_rows"], chunk=rt["chunk"], topk=rt["topk"],
                    use_ep=rt["use_ep"], projection_ep=rt["projection_ep"],
                    decode_plan=scratch_plan_summary(rt["trellis_plan"], rt["trellis_scratch"]),
                    prefill_plan=(None if rt["prefill_plan"] is None else
                                  scratch_plan_summary(rt["prefill_plan"], rt["prefill_scratch"])),
                    env=dict(env or {}))
        if rt["prefill_scratch"] is not None:
            meta["prefill_arena_mib"] = round(
                rt["prefill_scratch"].numel() * rt["prefill_scratch"].element_size() / (1 << 20), 1)
        self.runtimes[name] = dict(rt=rt, meta=meta)
        return self.runtimes[name]

    # -- invocation: exactly the bind/run lines of _apply_rank_sliced --------
    def run_plan(self, method, layer, rt: dict, which: str, x, w, ids, launch_log=None):
        if which == "decode":
            plan, scratch, output = rt["trellis_plan"], rt["trellis_scratch"], rt["trellis_output"]
        elif which == "prefill":
            plan, scratch, output = rt["prefill_plan"], rt["prefill_scratch"], rt["prefill_output"]
        else:
            raise ValueError(which)
        binding = self.exl3.Exl3MoEMethod._bind_rank_sliced(
            rt, layer, plan=plan, scratch=scratch, output=output,
            x=x, topk_weights=w, topk_ids=ids)
        if launch_log is not None:
            record_launch(launch_log, which, int(x.shape[0]), binding)
        raw = rt["api"].run(binding=binding)
        raw_c = raw.clone()           # outputs alias plan-owned buffers
        final = raw.to(x.dtype).clone()
        return final, raw_c

    def run_decode_any(self, method, layer, rt, x, w, ids, launch_log=None):
        """decode plan; m>32 is split into <=32-row chunks (decode capacity)."""
        torch = self.torch
        m = int(x.shape[0])
        if m <= rt["max_trellis_m"]:
            return self.run_plan(method, layer, rt, "decode", x, w, ids, launch_log)
        finals, raws = [], []
        for s in range(0, m, rt["max_trellis_m"]):
            e = min(s + rt["max_trellis_m"], m)
            f_, r_ = self.run_plan(method, layer, rt, "decode", x[s:e].contiguous(),
                                   w[s:e].contiguous(), ids[s:e].contiguous(), launch_log)
            finals.append(f_)
            raws.append(r_)
        return torch.cat(finals, 0), torch.cat(raws, 0)

    def run_apply(self, method, layer, x, w, ids):
        out = method.apply(layer, x, w, ids, None, None)
        return out.clone()


def plan_summary(plan) -> dict | None:
    if plan is None:
        return None
    out = {}
    if dataclasses.is_dataclass(plan):
        for fld in dataclasses.fields(plan):
            v = getattr(plan, fld.name, None)
            if isinstance(v, (int, float, str, bool, type(None))):
                out[fld.name] = v
            elif isinstance(v, (tuple, list, frozenset, set)) and len(v) < 16:
                out[fld.name] = [str(i) for i in v]
            else:
                out[fld.name] = str(v)[:120]
    else:
        out["repr"] = repr(plan)[:400]
    return out


def scratch_plan_summary(plan, scratch) -> dict:
    out: dict = dict(scratch_bytes=int(scratch.numel() * scratch.element_size())
                     if scratch is not None else None)
    caps = getattr(plan, "caps", None)
    if caps is not None and dataclasses.is_dataclass(caps):
        c = {}
        for fld in dataclasses.fields(caps):
            if fld.name in ("weight_plan",):
                continue
            v = getattr(caps, fld.name, None)
            c[fld.name] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)[:160]
        out["caps"] = c
    launches = getattr(plan, "_mixed_trellis_launches", None)
    if launches:
        out["n_launches"] = len(launches)
        out["launch_token_counts"] = sorted({int(item[0]) for item in launches})
        out["launch_variants"] = sorted({(str(item[1]), bool(item[2]), bool(item[3]))
                                         for item in launches})
    return out


LAUNCH_FIELDS = ("size_m", "fc1_tile_k", "fc1_tile_n", "fc2_tile_k", "fc2_tile_n",
                 "moe_block_size", "fc2_moe_block_size", "fc2_schedule_route_block_factor",
                 "max_m_blocks", "blocks_per_sm", "sms", "shared_memory_bytes",
                 "rotation_input_dtype", "route_ids_dtype", "direct_topk_routes",
                 "broadcast_suh", "broadcast_svh", "tier0_num_experts", "intermediate_size")


def record_launch(log: dict, which: str, m: int, binding) -> None:
    key = f"{which}:m{m}"
    if key in log:
        return
    mb = getattr(binding, "mixed_trellis_binding", None)
    launch = getattr(mb, "launch", None)
    if launch is None:
        log[key] = dict(error="binding has no mixed_trellis_binding.launch")
        return
    log[key] = {f: (getattr(launch, f) if isinstance(getattr(launch, f, None), (int, bool, str))
                    else str(getattr(launch, f, None))) for f in LAUNCH_FIELDS}


# --------------------------------------------------------------------------
# dry-mode fake kernel (row-local; plan difference and nondeterminism injected)
# --------------------------------------------------------------------------

DRY_NONDET_FIXTURE = "L3.m24.x1.spread.uniform.s1"
DRY_NONDET_ROW = 5


def dry_expected_plan_rows(m: int, n_real: int) -> list[int]:
    return [r for r in range(n_real) if m > 16 and r % 5 == 4]


def dry_kernel(torch, slice_name, which, x, w, ids, fid=None, rep=1, row_offset=0):
    local = ids < (GLOBAL_EXPERTS // 2) if SLICES[slice_name]["use_ep"] else torch.ones_like(ids, dtype=torch.bool)
    coef = (w * local.float() * (1.0 + (ids % 251).float() / 251.0)).sum(dim=1, keepdim=True)
    raw = x.float() * coef
    m = int(x.shape[0])
    if which == "prefill" and m > 16:
        for r in range(m):
            if (r + row_offset) % 5 == 4:
                raw[r, 0] = raw[r, 0] + raw[r, 0].abs() * 0.05 + 1e-3
    if which == "decode" and fid == DRY_NONDET_FIXTURE and rep == 3:
        raw[DRY_NONDET_ROW, 1] = raw[DRY_NONDET_ROW, 1] + raw[DRY_NONDET_ROW, 1].abs() * 0.05 + 1e-3
    return raw.to(torch.bfloat16), raw


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

class Runner:
    def __init__(self, args, doc, out: JsonOut, deadline: Deadline):
        import torch
        self.torch = torch
        self.args = args
        self.doc = doc
        self.out = out
        self.deadline = deadline
        self.engine = None
        self.eversion_store: dict = {}

    # --- per-slice, per-layer driver ----------------------------------------
    def run_slice(self, slice_name: str, role: str, layers, fixtures_all) -> None:
        torch = self.torch
        sdoc = self.doc["slices"].setdefault(slice_name, dict(role=role, layers={},
                                                             runtimes={}, errors=[]))
        for layer_idx in layers:
            if self.deadline.passed():
                self.doc["stopped_reason"] = "soft_deadline"
                return
            ldoc = sdoc["layers"].setdefault(str(layer_idx), dict(
                build=None, fixtures=[], controls=[], small_shape=[], errors=[],
                launch_map={}))
            fixtures = [f for f in fixtures_all if f["layer"] == layer_idx]
            if role == "eversion":
                fixtures = [f for f in fixtures if eversion_subset(f)]
            try:
                self._run_layer(slice_name, role, layer_idx, ldoc, sdoc, fixtures)
            except Exception as exc:  # layer-level failure: record, continue
                ldoc["errors"].append(dict(stage="layer", error=repr(exc), tb=short_tb(exc)))
            self.out.write()
            if not self.args.dry:
                torch.cuda.empty_cache()
            if self.doc.get("stopped_reason"):
                return

    def _run_layer(self, slice_name, role, layer_idx, ldoc, sdoc, fixtures) -> None:
        torch = self.torch
        eng = self.engine
        ref = eng.ref_rms(layer_idx)
        ldoc["ref_rms"] = ref
        method, layer, info = eng.build_layer(slice_name, layer_idx)
        ldoc["build"] = info
        served = SERVED_LOG[self.args.engine]
        slice_log = next(v for v in SERVED_LOG.values() if v["slice"] == slice_name)
        if info.get("membership") is not None:
            exp = [list(t) for t in slice_log["membership"][layer_idx]]
            info["membership_served_log"] = exp
            info["membership_served_log_ref"] = (
                f"{slice_log['file']}:{slice_log['membership_lines'][layer_idx]}")
            info["membership_matches_served_log"] = info["membership"] == exp
        self.out.write()
        if self.args.dry:
            if method is not None:
                self._dry_prepare_probe(method, layer, info)
            del method, layer
            for f in fixtures:
                if self.deadline.passed():
                    self.doc["stopped_reason"] = "soft_deadline"
                    break
                self._fixture(None, None, None, slice_name, f, ref, ldoc)
            self._controls(None, None, None, slice_name, layer_idx, ref, ldoc)
            if role == "own" and ENGINES[self.args.engine]["small_shape_check"]:
                self._small_shape(None, None, None, None, slice_name, layer_idx, ref, ldoc)
            ldoc["summary"] = summarize(ldoc)
            return
        eng.prepare_layer(method, layer, info)
        self.out.write()
        # runtimes: built once per slice (served: one runtime per model scope)
        f0 = make_fixture(torch, fixtures[0] if fixtures else dict(
            layer=layer_idx, m=1, scale="x1", conc="spread", wmode="uniform", seed=0,
            kind="main"), ref)
        x0 = f0["x"].to(eng.device)
        ids0 = f0["ids"].to(eng.device)
        rtname = f"{slice_name}:default"
        rtd = eng.runtime(method, layer, x0, ids0, rtname)
        sdoc["runtimes"]["default"] = rtd["meta"]
        if served["slice"] == slice_name and "prefill_arena_mib" in rtd["meta"] \
                and "arena_matches_served_log" not in rtd["meta"]:
            rtd["meta"]["prefill_arena_mib_served_log"] = served["prefill_arena_mib"]
            rtd["meta"]["arena_matches_served_log"] = (
                rtd["meta"]["prefill_arena_mib"] == served["prefill_arena_mib"])
        rt16 = None
        if role == "own" and ENGINES[self.args.engine]["small_shape_check"]:
            try:
                rt16d = eng.runtime(method, layer, x0, ids0, f"{slice_name}:maxm16",
                                    env={"VLLM_EXL3_TRELLIS_MAX_M": "16",
                                         "VLLM_EXL3_PREFILL_TRELLIS": "0"})
                sdoc["runtimes"]["maxm16"] = rt16d["meta"]
                rt16 = rt16d["rt"]
            except Exception as exc:
                sdoc["errors"].append(dict(stage="runtime_maxm16", error=repr(exc), tb=short_tb(exc)))
        self.out.write()
        t0 = time.time()
        consecutive_cuda_errors = 0
        for f in fixtures:
            if self.deadline.passed():
                self.doc["stopped_reason"] = "soft_deadline"
                break
            ok = self._fixture(method, layer, rtd["rt"], slice_name, f, ref, ldoc)
            if ok == "cuda_error":
                consecutive_cuda_errors += 1
                if consecutive_cuda_errors >= 3:
                    ldoc["errors"].append(dict(stage="abort", error="3 consecutive CUDA errors"))
                    self.doc["stopped_reason"] = "cuda_error"
                    break
            else:
                consecutive_cuda_errors = 0
            self.out.write(force=False)
        if not self.doc.get("stopped_reason"):
            self._controls(method, layer, rtd["rt"], slice_name, layer_idx, ref, ldoc)
            if rt16 is not None:
                self._small_shape(method, layer, rtd["rt"], rt16, slice_name, layer_idx, ref, ldoc)
        ldoc["t_measure_s"] = round(time.time() - t0, 2)
        ldoc["summary"] = summarize(ldoc)
        if self.eversion_store.get((slice_name, layer_idx)):
            self._save_eversion()
        del method, layer

    def _dry_prepare_probe(self, method, layer, info) -> None:
        """Run process_weights_after_loading on CPU up to api.prepare_weights."""
        exl3 = self.engine.exl3
        real = exl3._load_b12x_fused_moe()
        captured: dict = {}

        class _DryStop(Exception):
            pass

        def prepare_weights(*, plan, params_dtype, projection_tiers=None, **kw):
            captured["params_dtype"] = str(params_dtype)
            captured["tiers"] = [dict(bits=int(t.bits), w13=list(t.w13.shape), w2=list(t.w2.shape),
                                      gate=len(t.gate_experts), up=len(t.up_experts),
                                      down=len(t.down_experts)) for t in (projection_tiers or ())]
            captured["rotations"] = {k: list(v.shape) for k, v in kw.items()
                                     if hasattr(v, "shape")}
            raise _DryStop()

        proxy = SimpleNamespace(plan_weights=real.plan_weights, prepare_weights=prepare_weights)
        saved = exl3._SPARKINFER_FUSED_MOE_API
        exl3._SPARKINFER_FUSED_MOE_API = proxy
        orig_plan_weights = real.plan_weights

        def plan_weights(**kw):
            captured["plan_weights_kwargs"] = {k: (v if isinstance(v, (int, str, type(None))) else str(v))
                                               for k, v in kw.items()}
            p = orig_plan_weights(**kw)
            captured["weight_plan"] = plan_summary(p)
            return p

        proxy.plan_weights = plan_weights
        try:
            method.process_weights_after_loading(layer)
            captured["reached"] = "complete (unexpected on CPU)"
        except _DryStop:
            captured["reached"] = "api.prepare_weights (CUDA required beyond this point)"
        except Exception as exc:
            captured["reached"] = "error"
            captured["error"] = repr(exc)
            captured["tb"] = short_tb(exc)
        finally:
            exl3._SPARKINFER_FUSED_MOE_API = saved
            for prefix in ("w13", "w2"):
                for suffix in ("suh", "svh", "trellis", "mcg", "mul1"):
                    p = getattr(layer, f"{prefix}_{suffix}", None)
                    if p is not None:
                        p.exl3_tensors.clear()
                        p.exl3_backing = None
        info["dry_prepare_probe"] = captured

    # --- one fixture ---------------------------------------------------------
    def _invoke_all(self, method, layer, rt, slice_name, f, fx, ldoc):
        """3 repeats per plan, interleaved D P D P D P, then apply."""
        torch = self.torch
        fid = fixture_id(f)
        if self.args.dry:
            x, w, ids = fx["x"], fx["w"], fx["ids"]
            dec, pre = [], []
            for rep in (1, 2, 3):
                m = int(x.shape[0])
                if m <= DECODE_CHUNK:
                    d = dry_kernel(torch, slice_name, "decode", x, w, ids, fid, rep)
                else:
                    parts = [dry_kernel(torch, slice_name, "decode", x[s:s + DECODE_CHUNK],
                                        w[s:s + DECODE_CHUNK], ids[s:s + DECODE_CHUNK], fid, rep)
                             for s in range(0, m, DECODE_CHUNK)]
                    d = (torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]))
                dec.append(d)
                pre.append(dry_kernel(torch, slice_name, "prefill", x, w, ids, fid, rep))
            app = dec[0][0].clone() if int(x.shape[0]) <= DECODE_CHUNK else pre[0][0].clone()
            return dec, pre, app
        dev = self.engine.device
        x, w, ids = fx["x"].to(dev), fx["w"].to(dev), fx["ids"].to(dev)
        dec, pre = [], []
        for _ in range(3):
            dec.append(self.engine.run_decode_any(method, layer, rt, x, w, ids, ldoc["launch_map"]))
            pre.append(self.engine.run_plan(method, layer, rt, "prefill", x, w, ids, ldoc["launch_map"]))
        app = self.engine.run_apply(method, layer, x, w, ids)
        torch.cuda.synchronize()
        return dec, pre, app

    def _fixture(self, method, layer, rt, slice_name, f, ref, ldoc):
        torch = self.torch
        fid = fixture_id(f)
        rec: dict = dict(id=fid, **{k: f[k] for k in ("m", "scale", "conc", "wmode", "seed", "kind")})
        t0 = time.time()
        try:
            fx = make_fixture(torch, f, ref)
            rec.update(n_real=fx["n_real"], input_digest=fx["digest"], heavy_rows=fx["heavy"],
                       conc_rows=[r for r, c in enumerate(fx["row_conc"]) if c][:64])
            if f.get("pad_mode"):
                rec.update(pad_mode=f["pad_mode"])
            rows = list(range(fx["n_real"]))
            dec, pre, app = self._invoke_all(method, layer, rt, slice_name, f, fx, ldoc)
            D = [d[0] for d in dec]
            P = [p[0] for p in pre]
            rec["within"] = dict(decode=within(torch, D, rows), prefill=within(torch, P, rows))
            rec["between"] = compare(torch, D[0], P[0], rows)
            if D[0].shape == dec[0][1].shape and dec[0][1].dtype == torch.float32 \
                    and pre[0][1].dtype == torch.float32:
                rb = compare(torch, dec[0][1], pre[0][1], rows, max_list=16)
                rec["between_raw_fp32"] = {k: rb[k] for k in ("row_bitequal_frac", "n_rows_differ",
                                                              "max_abs", "rel_l2") if k in rb}
            dispatch = "decode" if f["m"] <= DECODE_CHUNK else "prefill"
            target = D[0] if dispatch == "decode" else P[0]
            rec["apply_dispatch"] = dict(plan=dispatch, bitequal_all_rows=bool(
                bit_rows_equal(torch, app, target, torch.arange(f["m"], device=app.device)).all()))
            rec["zero_rows_decode"] = int((D[0][rows].float().abs().sum(dim=1) == 0).sum())
            rec["digests"] = dict(decode=row_digests(torch, D[0]), prefill=row_digests(torch, P[0]))
            if f.get("pad_mode"):
                key = fid.rsplit("-", 1)[0]
                store = ldoc.setdefault("_pad_pairs", {})
                cur = dict(decode=D[0][rows].clone(), prefill=P[0][rows].clone())
                if key in store:
                    other = store.pop(key)
                    rec["pad_content_independent"] = {
                        p: bool(bit_rows_equal(torch, cur[p], other[p],
                                               torch.arange(len(rows), device=cur[p].device)).all())
                        for p in ("decode", "prefill")}
                else:
                    store[key] = cur
            if eversion_subset(f):
                self.eversion_store.setdefault((slice_name, f["layer"]), {})[fid] = dict(
                    decode=D[0].detach().cpu(), prefill=P[0].detach().cpu(),
                    input_digest=fx["digest"])
            status = "ok"
        except Exception as exc:
            rec["error"] = repr(exc)
            rec["tb"] = short_tb(exc)
            status = "cuda_error" if "CUDA" in repr(exc) or "cuda" in repr(exc) else "error"
        rec["t_ms"] = round((time.time() - t0) * 1000, 1)
        ldoc["fixtures"].append(rec)
        return status

    # --- positive controls ---------------------------------------------------
    def _controls(self, method, layer, rt, slice_name, layer_idx, ref, ldoc):
        torch = self.torch
        bases = [dict(layer=layer_idx, m=32, scale="x1", conc="mixed", wmode="uniform", seed=0, kind="main")]
        if not self.args.smoke:
            bases.append(dict(layer=layer_idx, m=48, scale="x1", conc="mixed", wmode="uniform", seed=0, kind="main"))
        for base in bases:
            fx = make_fixture(torch, base, ref)
            r = 2  # even row in a mixed fixture: concentrated, experts local to both slices
            c = int(fx["x"][r].float().abs().argmax())
            x_ulp = fx["x"].clone()
            x_ulp.view(torch.int16)[r, c] += 1  # one bf16 ulp (bit pattern +1)
            ids_e = fx["ids"].clone()
            present = set(int(v) for v in ids_e[r])
            new_id = next(e for e in range(GLOBAL_EXPERTS // 2) if e not in present)
            ids_e[r, 0] = new_id
            for name, xx, ii in (("bf16_ulp", x_ulp, fx["ids"]), ("expert_id", fx["x"], ids_e)):
                rec = dict(base=fixture_id(base), control=name, row=r, channel=c if name == "bf16_ulp" else None,
                           new_expert=new_id if name == "expert_id" else None)
                try:
                    # GPU: judged on the served observable (bf16 output after
                    # .to(x.dtype)); raw (fp32 on TP plans) recorded alongside.
                    # Dry: the fake kernel's bf16 rounding can absorb a 1-ulp
                    # input change, so dry controls are judged on its fp32 raw.
                    judge = "raw" if self.args.dry else "final"
                    for plan in ("decode", "prefill"):
                        b_fin, b_raw = self._single(method, layer, rt, slice_name, plan, fx["x"], fx["w"], fx["ids"])
                        c_fin, c_raw = self._single(method, layer, rt, slice_name, plan, xx, fx["w"], ii)
                        rows_all = list(range(base["m"]))
                        res = {}
                        for lvl, (cc, bb) in (("final", (c_fin, b_fin)), ("raw", (c_raw, b_raw))):
                            cmpd = compare(torch, cc, bb, rows_all)
                            changed = cmpd.get("rows_differ", [])
                            res[lvl] = dict(dtype=str(cc.dtype), rows_changed=changed,
                                            detected=r in changed, attribution_exact=changed == [r],
                                            n_elem_changed=cmpd.get("n_elem_differ"))
                        rec[plan] = dict(judged_on=judge, **res)
                    rec["pass"] = all(rec[p][judge]["attribution_exact"] for p in ("decode", "prefill"))
                except Exception as exc:
                    rec["error"] = repr(exc)
                    rec["tb"] = short_tb(exc)
                    rec["pass"] = False
                ldoc["controls"].append(rec)

    def _single(self, method, layer, rt, slice_name, plan, x, w, ids, rt_override=None):
        torch = self.torch
        if self.args.dry:
            if plan == "decode" and int(x.shape[0]) > DECODE_CHUNK:
                parts = [dry_kernel(torch, slice_name, "decode", x[s:s + DECODE_CHUNK],
                                    w[s:s + DECODE_CHUNK], ids[s:s + DECODE_CHUNK])
                         for s in range(0, int(x.shape[0]), DECODE_CHUNK)]
                return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
            return dry_kernel(torch, slice_name, "prefill" if plan == "prefill" else "decode", x, w, ids)
        dev = self.engine.device
        x, w, ids = x.to(dev), w.to(dev), ids.to(dev)
        use_rt = rt_override if rt_override is not None else rt
        if plan == "prefill":
            out = self.engine.run_plan(method, layer, use_rt, "prefill", x, w, ids)
        else:
            out = self.engine.run_decode_any(method, layer, use_rt, x, w, ids)
        torch.cuda.synchronize()
        return out

    # --- small-shape configuration check (MAX_M=32 vs MAX_M=16) ---------------
    def _small_shape(self, method, layer, rt, rt16, slice_name, layer_idx, ref, ldoc):
        torch = self.torch
        ms = (8,) if self.args.smoke else SMALL_SHAPE_MS
        concs = ("spread",) if self.args.smoke else CONCS
        seeds = (0,) if self.args.smoke else SEEDS
        for m in ms:
            for conc in concs:
                for seed in seeds:
                    f = dict(layer=layer_idx, m=m, scale="x1", conc=conc, wmode="uniform",
                             seed=seed, kind="small_shape")
                    rec = dict(id=fixture_id(f) + ".maxm16")
                    try:
                        fx = make_fixture(torch, f, ref)
                        a = self._single(method, layer, rt, slice_name, "decode", fx["x"], fx["w"], fx["ids"])[0]
                        b = (a.clone() if self.args.dry else
                             self._single(method, layer, rt, slice_name, "decode", fx["x"], fx["w"],
                                          fx["ids"], rt_override=rt16)[0])
                        c = compare(torch, a, b, list(range(m)))
                        rec.update(row_bitequal_frac=c["row_bitequal_frac"],
                                   rows_differ=c["rows_differ"], max_abs=c["max_abs"])
                        if not self.args.dry:
                            log = ldoc["launch_map"]
                            for tag, r_ in (("maxm32", rt), ("maxm16", rt16)):
                                bnd = self.engine.exl3.Exl3MoEMethod._bind_rank_sliced(
                                    r_, layer, plan=r_["trellis_plan"], scratch=r_["trellis_scratch"],
                                    output=r_["trellis_output"], x=fx["x"].to(self.engine.device),
                                    topk_weights=fx["w"].to(self.engine.device),
                                    topk_ids=fx["ids"].to(self.engine.device))
                                record_launch(log, f"decode_{tag}", m, bnd)
                    except Exception as exc:
                        rec["error"] = repr(exc)
                        rec["tb"] = short_tb(exc)
                    ldoc["small_shape"].append(rec)

    def _save_eversion(self) -> None:
        if self.args.dry:
            return
        torch = self.torch
        path = os.path.splitext(self.args.out)[0] + ".eversion.pt"
        payload = {f"{s}|{l}": v for (s, l), v in self.eversion_store.items()}
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        self.doc["eversion_file"] = os.path.basename(path)


def summarize(ldoc: dict) -> dict:
    fx = [f for f in ldoc["fixtures"] if "error" not in f]
    errs = [f for f in ldoc["fixtures"] if "error" in f]
    s = dict(n_fixtures=len(ldoc["fixtures"]), n_ok=len(fx), n_error=len(errs))
    s["within_decode_all_bitequal"] = all(f["within"]["decode"]["bitequal"] for f in fx)
    s["within_prefill_all_bitequal"] = all(f["within"]["prefill"]["bitequal"] for f in fx)
    s["n_within_nondeterministic"] = sum(1 for f in fx if not (f["within"]["decode"]["bitequal"]
                                                              and f["within"]["prefill"]["bitequal"]))
    s["n_fixtures_plans_differ"] = sum(1 for f in fx if f["between"].get("n_rows_differ", 0))
    s["rows_compared"] = sum(f["between"].get("rows", 0) for f in fx)
    s["rows_differ"] = sum(f["between"].get("n_rows_differ", 0) for f in fx)
    s["max_abs_between"] = max([f["between"].get("max_abs", 0.0) for f in fx] or [0.0])
    s["max_rel_l2_between"] = max([f["between"].get("rel_l2", 0.0) for f in fx] or [0.0])
    s["apply_dispatch_all_match"] = all(f["apply_dispatch"]["bitequal_all_rows"] for f in fx)
    by_m: dict = {}
    for f in fx:
        d = by_m.setdefault(str(f["m"]), dict(n=0, differ=0, rows=0, rows_differ=0))
        d["n"] += 1
        d["rows"] += f["between"].get("rows", 0)
        d["rows_differ"] += f["between"].get("n_rows_differ", 0)
        d["differ"] += 1 if f["between"].get("n_rows_differ", 0) else 0
    s["by_m"] = by_m
    pads = [f for f in fx if "pad_content_independent" in f]
    s["pad_content_independent_all"] = all(all(f["pad_content_independent"].values()) for f in pads) if pads else None
    s["controls_all_pass"] = all(c.get("pass") for c in ldoc["controls"]) if ldoc["controls"] else None
    ss = [c for c in ldoc["small_shape"] if "error" not in c]
    s["small_shape_all_bitequal"] = (all(c["row_bitequal_frac"] == 1.0 for c in ss)
                                     if ldoc["small_shape"] else None)
    ldoc.pop("_pad_pairs", None)
    return s


# --------------------------------------------------------------------------
# dry self-check: the comparison logic must find exactly what was injected
# --------------------------------------------------------------------------

def dry_selfcheck(doc: dict) -> dict:
    checks = dict(plan_rows_exact=True, within_injection_found=False, within_false_positive=False,
                  controls_pass=True, dispatch_match=True, shapes_ok=True, small_shape_equal=True,
                  pad_independent=True, n_fixtures=0, failures=[])
    for sname, sdoc in doc["slices"].items():
        for lk, ldoc in sdoc["layers"].items():
            for f in ldoc["fixtures"]:
                checks["n_fixtures"] += 1
                if "error" in f:
                    checks["failures"].append(f"{sname}/{f['id']}: {f['error']}")
                    checks["shapes_ok"] = False
                    continue
                exp = dry_expected_plan_rows(f["m"], f["n_real"])
                got = f["between"].get("rows_differ", [])
                if got != exp:
                    checks["plan_rows_exact"] = False
                    checks["failures"].append(f"{sname}/{f['id']}: plan rows {got} != {exp}")
                if len(f["digests"]["decode"]) != f["m"]:
                    checks["shapes_ok"] = False
                nd = not f["within"]["decode"]["bitequal"]
                if f["id"] == DRY_NONDET_FIXTURE and nd:
                    rows = f["within"]["decode"]["spread"][1]["rows_differ"]
                    checks["within_injection_found"] = rows == [DRY_NONDET_ROW]
                elif nd or not f["within"]["prefill"]["bitequal"]:
                    checks["within_false_positive"] = True
                if not f["apply_dispatch"]["bitequal_all_rows"]:
                    checks["dispatch_match"] = False
                if "pad_content_independent" in f and not all(f["pad_content_independent"].values()):
                    checks["pad_independent"] = False
            for c in ldoc["controls"]:
                if not c.get("pass"):
                    checks["controls_pass"] = False
                    checks["failures"].append(f"{sname}/control {c}")
            for c in ldoc["small_shape"]:
                if c.get("row_bitequal_frac") != 1.0:
                    checks["small_shape_equal"] = False
    layer_errs = [e for s in doc["slices"].values() for l in s["layers"].values() for e in l["errors"]]
    probes = [l["build"].get("dry_prepare_probe", {}).get("reached") for s in doc["slices"].values()
              for l in s["layers"].values() if l.get("build") and "skipped" not in l["build"]]
    memb = [l["build"].get("membership_matches_served_log") for s in doc["slices"].values()
            for l in s["layers"].values() if l.get("build") and "skipped" not in l["build"]]
    checks["layer_errors"] = len(layer_errs)
    checks["weights_probe_layers"] = len(probes)
    checks["weights_probe_reached_prepare"] = all(p and p.startswith("api.prepare_weights") for p in probes)
    checks["membership_matches_served_logs"] = all(memb) if memb else None
    checks["pass"] = (checks["n_fixtures"] > 0 and not layer_errs
                      and checks["weights_probe_reached_prepare"]
                      and checks["membership_matches_served_logs"] is not False
                      and checks["plan_rows_exact"] and not checks["within_false_positive"]
                      and checks["controls_pass"] and checks["dispatch_match"] and checks["shapes_ok"]
                      and checks["small_shape_equal"] and checks["pad_independent"]
                      and (checks["within_injection_found"] or not any(
                          f["id"] == DRY_NONDET_FIXTURE for s in doc["slices"].values()
                          for l in s["layers"].values() for f in l["fixtures"])))
    checks["failures"] = checks["failures"][:40]
    return checks


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=sorted(ENGINES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry", action="store_true", help="CPU only, fabricated outputs")
    ap.add_argument("--smoke", action="store_true", help="one layer, own slice, few fixtures")
    ap.add_argument("--deadline-s", type=float, default=None,
                    help="soft wall: stop starting new work after N s from process start")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--dry-skip-weights", action="store_true",
                    help="dry only: skip the CPU weight-slice bookkeeping probe")
    ap.add_argument("--no-eversion", action="store_true", help="skip the other engine's slice")
    args = ap.parse_args()
    eng_cfg = ENGINES[args.engine]
    mode = "dry" if args.dry else ("smoke" if args.smoke else "full")
    harness_sha = sha256_file(os.path.abspath(__file__))
    doc = dict(schema=SCHEMA, engine=args.engine, mode=mode, status="running",
               stopped_reason=None, started_utc=utc(), argv=sys.argv[1:],
               expected=dict(image=eng_cfg["image"], b12x_commit=eng_cfg["b12x"],
                             exl3_sha256=eng_cfg["exl3_sha256"]),
               harness_sha256=harness_sha, layers=list(LAYERS[:1] if args.smoke else LAYERS),
               model_dir=args.model_dir, reuse=REUSE_MAP, served_log=SERVED_LOG[args.engine],
               served_env_expected=SERVED_ENV, timings={}, slices={}, errors=[])
    out = JsonOut(args.out, doc)
    deadline = Deadline(args.deadline_s)
    # environment / versions / code hashes
    doc["env"] = {k: os.environ.get(k) for k in list(SERVED_ENV) + list(SERVED_UNSET) +
                  ["CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_LAUNCH_BLOCKING",
                   "CUDA_MODULE_LOADING", "B12X_COMPILE_CACHE_DIR", "HOME"]}
    env_mismatch = {k: (os.environ.get(k), v) for k, v in SERVED_ENV.items() if os.environ.get(k) != v}
    env_mismatch.update({k: (os.environ.get(k), None) for k in SERVED_UNSET if os.environ.get(k)})
    doc["env_matches_served"] = not env_mismatch
    doc["env_mismatch"] = env_mismatch
    doc["code_hashes"] = {p: sha256_file(p) for p in REUSED_FILES}
    exl3_sha = doc["code_hashes"][REUSED_FILES[0]]
    doc["exl3_sha256_matches_engine"] = exl3_sha == eng_cfg["exl3_sha256"]
    out.write()
    if not doc["exl3_sha256_matches_engine"]:
        doc["status"] = "error"
        doc["errors"].append(f"exl3.py sha256 {exl3_sha} does not match engine {args.engine}; wrong image?")
        out.write()
        return 3
    if env_mismatch and not args.dry:
        doc["status"] = "error"
        doc["errors"].append(f"served env not set: {env_mismatch}")
        out.write()
        return 3
    try:
        import torch
        doc["versions"] = dict(python=platform.python_version(), torch=torch.__version__,
                               torch_cuda=torch.version.cuda)
        try:
            import b12x
            doc["versions"]["b12x"] = getattr(b12x, "__version__", None)
        except Exception as exc:  # noqa: BLE001
            doc["versions"]["b12x_import_error"] = repr(exc)
        if not args.dry:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available (GPU mode)")
            props = torch.cuda.get_device_properties(0)
            doc["device"] = dict(name=props.name, sm=f"{props.major}.{props.minor}",
                                 sms=props.multi_processor_count,
                                 total_mib=props.total_memory // (1 << 20),
                                 count=torch.cuda.device_count())
        runner = Runner(args, doc, out, deadline)
        runner.engine = Engine(args, doc)
        fixtures = panel(doc["layers"], args.smoke)
        doc["panel"] = dict(n_fixtures=len(fixtures), shapes=list(SHAPES), scales=SCALES,
                            concentrations=list(CONCS), weightings=list(WMODES), seeds=list(SEEDS),
                            heavy_rows=f"r % {HEAVY_EVERY} == {HEAVY_EVERY - 1}: {HEAVY_CHANNELS} channels x{HEAVY_FACTOR}",
                            pad_cells=[list(p) for p in PAD_CELLS], pad_modes=list(PAD_MODES),
                            small_shape_ms=list(SMALL_SHAPE_MS), eversion_ms=list(EVERSION_MS))
        out.write()
        t0 = time.time()
        if args.dry and args.dry_skip_weights:
            runner.engine.build_layer = _dry_fake_build(runner.engine)
        runner.run_slice(eng_cfg["own"], "own", doc["layers"], fixtures)
        doc["timings"]["own_slice_s"] = round(time.time() - t0, 1)
        eversion_min_s = 240  # load+prepare 3 layers + one runtime compile, conservatively
        if not args.smoke and not args.no_eversion and not doc.get("stopped_reason") \
                and deadline.left() < eversion_min_s:
            doc["eversion_skipped"] = f"soft time left {deadline.left():.0f} s < {eversion_min_s} s"
        elif not args.smoke and not args.no_eversion and not doc.get("stopped_reason"):
            t1 = time.time()
            runner.run_slice(eng_cfg["other"], "eversion", doc["layers"], fixtures)
            doc["timings"]["eversion_slice_s"] = round(time.time() - t1, 1)
        layer_errors = [f"{s}/{l}: {e.get('stage')}: {e.get('error')}"
                        for s, sd in doc["slices"].items() for l, ld in sd["layers"].items()
                        for e in ld["errors"]]
        doc["layer_errors"] = layer_errors[:40]
        if args.dry:
            doc["dry_selfcheck"] = dry_selfcheck(doc)
        if layer_errors:
            doc["status"] = "error"
        else:
            doc["status"] = "partial" if doc.get("stopped_reason") else "complete"
    except Exception as exc:
        doc["status"] = "error"
        doc["errors"].append(dict(error=repr(exc), tb=short_tb(exc, 14)))
    doc["finished_utc"] = utc()
    out.write()
    print(json.dumps(dict(engine=args.engine, mode=mode, status=doc["status"],
                          stopped_reason=doc.get("stopped_reason"),
                          elapsed_s=round(time.time() - T_PROCESS_START, 1),
                          dry_selfcheck=doc.get("dry_selfcheck", {}).get("pass"))))
    return 0 if doc["status"] == "complete" else (4 if doc["status"] == "partial" else 5)


def _dry_fake_build(engine):
    """--dry-skip-weights: bookkeeping-free layer stand-in."""
    def build_layer(slice_name, layer_idx):
        return None, None, dict(slice=slice_name, layer=layer_idx, skipped="dry-skip-weights",
                                membership=None)
    return build_layer


if __name__ == "__main__":
    sys.exit(main())
