"""Shared CPU-test scaffolding (inside the B1 image as a CPU container; Triton interpreter for engine kernels).

The only shim: vLLM's UVA buffers need pinned memory and a CUDA view, which a CPU container lacks. UvaBuffer is
replaced by plain CPU tensors whose "uva" view is the same tensor. Every kernel and Python function under test is
the installed B1 code (the hooked copies for the five patched files).
"""
import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import vllm.v1.worker.gpu.buffer_utils as BU  # noqa: E402


class _CpuUva:
    def __init__(self, size, dtype):
        self.cpu = torch.zeros(size, dtype=dtype)
        self.np = self.cpu.numpy()
        self.uva = self.cpu


BU.UvaBuffer = _CpuUva
import vllm.utils.torch_utils as _TU  # noqa: E402
_TU.PIN_MEMORY = False                   # pinned host memory needs a driver; CPU copies are synchronous anyway
import vllm.v1.worker.gpu.states as _st  # noqa: E402,F401
# Triton's CPU interpreter lacks libdevice.log1p (used by the gumbel noise and one rejection-ratio term).
# Test-only substitute log(1 + x): it perturbs random draws, never the logits the hook measures or the
# forced commits (which do not depend on draws).
import types as _types  # noqa: E402
from vllm.triton_utils import tl as _tl  # noqa: E402
import vllm.v1.worker.gpu.sample.gumbel as _gum  # noqa: E402
import vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils as _rsu  # noqa: E402
_LD = _types.SimpleNamespace(log1p=lambda x: _tl.log(1.0 + x))
_gum.tldevice = _LD
_rsu.tldevice = _LD
# apply_top_k_top_p dispatches to a Triton kernel that needs a device SM count; on CPU use the engine's own
# PyTorch implementation (the path the GPU takes for batches of fewer than 8 rows).
import vllm.v1.sample.ops.topk_topp_sampler as _tts  # noqa: E402
import vllm.v1.worker.gpu.sample.sampler as _smp  # noqa: E402
import vllm.v1.worker.gpu.sample.states as _sst  # noqa: E402


def _topkp_cpu(logits, k, p):
    if p is None and k is None:
        return logits
    return _tts.apply_top_k_top_p_pytorch(logits, k, p, allow_cpu_sync=True)


_tts.apply_top_k_top_p = _topkp_cpu
for _m in (_smp, _sst):
    if hasattr(_m, "apply_top_k_top_p"):
        _m.apply_top_k_top_p = _topkp_cpu

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("%s %s %s" % ("PASS" if cond else "FAIL", name, detail), flush=True)


def summary(tag):
    n = sum(1 for _, ok in RESULTS if ok)
    print("%s: %d/%d checks passed" % (tag, n, len(RESULTS)), flush=True)
    return n == len(RESULTS)


V = 154880
TERM = (154842, 154820, 154827, 154829)


def toy_logits(prefix):
    """Deterministic next-token logits for a token prefix (independent of batch, row and step)."""
    key = (hash(tuple(prefix[-6:])) ^ (len(prefix) * 1000003)) & 0x7FFFFFFF
    g = torch.Generator().manual_seed(key)
    x = torch.randn(V, generator=g)
    pref = 1000 + key % 50000
    x[pref] += 9.0
    x[pref + 1] += 7.5
    x[pref + 2] += 6.0
    if len(prefix) % 29 == 0:
        x[TERM[0]] += 10.0
    if len(prefix) % 41 == 0:
        x[TERM[2]] += 8.0
    return x
