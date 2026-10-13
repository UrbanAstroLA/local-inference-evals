"""Shared helpers for the D7 CPU tests (no GPU; Triton interpreter for the engine kernels).

Independent reference: ``ref_record`` pools four tokens in plain torch (softmax(gate + ape)-weighted
sum -> bf16 -> Hadamard-128 by Sylvester matrix -> bf16 -> per-vector absmax E4M3 with a power-of-two
scale) and returns the 132-byte index-K record (128 E4M3 bytes + fp32 scale bytes). It never calls an
engine kernel. The members are chosen by LOGICAL POSITION from the request's own token arrays.

Cache geometry follows the B0 layout used by the guard and the D2 tests: a KV block = 17 indexer pages
of 8,448 B (64 pools x 128 B fp8 + 64 x 4 B scales); 1,088 pools per 4,352-token block; the kpool tail
ring aliases the same raw buffer with the block stride (b0-mapguard tests, D2 tailring_scenarios.py).
"""
import math
import os

import numpy as np
import torch

from vllm.triton_utils import tl, triton

KP, HD = 4, 128
IDX_PAGE = 8448
PAGE = 17 * IDX_PAGE  # bytes per KV block
FP8_MAX = 448.0
SENT = 0xA5


def make_caches(nb, ring, sentinel=SENT):
    raw = torch.full((nb * PAGE,), sentinel, dtype=torch.uint8)
    tail = torch.as_strided(raw.view(torch.bfloat16), size=(nb, 2, ring, HD),
                            stride=(PAGE // 2, ring * HD, HD, 1))
    idx = torch.as_strided(raw, size=(nb * 17, 64, 132), stride=(IDX_PAGE, 132, 1))
    return raw, tail, idx


def rec(raw, loc):
    """132-byte index-K record of pool slot ``loc`` (page-planar)."""
    pg, r = loc // 64, loc % 64
    base = pg * IDX_PAGE
    return torch.cat([raw[base + r * 128: base + r * 128 + 128],
                      raw[base + 8192 + r * 4: base + 8192 + r * 4 + 4]]).clone()


def _sylvester(n=128):
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


_H = _sylvester()


def fwht128(x):
    """Unnormalised Walsh-Hadamard transform by radix-2 butterflies, strides 1,2,...,64 (a+b, a-b).
    Same transform as x @ _H; the butterfly order makes the fp32 sums round like the engine's kernel
    (a 128-term matmul can differ by one fp32 ulp, which bf16 truncation can expose)."""
    x = x.float().reshape(128)
    for groups, stride in ((64, 1), (32, 2), (16, 4), (8, 8), (4, 16), (2, 32), (1, 64)):
        x3 = x.view(groups, 2, stride)
        a, b = x3[:, 0], x3[:, 1]
        x = torch.stack([a + b, a - b], 1).reshape(128)
    return x


@triton.jit
def _store_cast_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < N
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m), mask=m)


def store_cast_e4m3(x):
    """Element-wise fp32 -> E4M3 conversion performed by a Triton store into an fp8 pointer -- the same
    implicit conversion the engine's compress kernel uses for its record bytes. Under TRITON_INTERPRET
    this conversion is emulated and is NOT torch's round-to-nearest-even (see REPORT, measured), so the
    reference uses it for the final cast only; all pooling arithmetic stays in plain torch."""
    x = x.contiguous().float()
    y = torch.empty(x.numel(), dtype=torch.float8_e4m3fn)
    _store_cast_kernel[(1,)](x, y, N=x.numel(), BLOCK=triton.next_power_of_2(x.numel()))
    return y


# Element-wise conversions. On the GPU both are round-to-nearest-even. Under TRITON_INTERPRET (CPU) the
# images' Triton interpreter converts fp32 -> bf16 by TRUNCATION and fp32 -> E4M3 with a rounding-carry
# error (measured; REPORT section "CPU numerics"). The reference therefore applies the platform's
# element-wise conversion: "interp" = bf16 truncation (plain torch bit mask) + the Triton store cast for
# E4M3; "rne" = torch's round-to-nearest-even for both. Pooling arithmetic is plain torch in both modes.
MODE = os.environ.get("D7_REF_MODE", "interp" if os.environ.get("TRITON_INTERPRET") == "1" else "rne")


def _bf16(x, mode):
    if mode == "interp":
        return (x.float().contiguous().view(torch.int32) & ~0xFFFF).view(torch.float32)
    return x.to(torch.bfloat16).float()


def ref_record(k4, g4, ape, round_scale=True, mode=None):
    """Plain-torch pooled record of four tokens (rows of k4/g4 in logical order)."""
    mode = mode or MODE
    k4 = k4.float()
    s = g4.float() + ape.float()
    m = s.max(dim=0).values
    acc = torch.zeros(HD, dtype=torch.float32)
    den = torch.zeros(HD, dtype=torch.float32)
    for j in range(KP):
        # exp in float32 via numpy: the interpreter evaluates tl.exp with numpy, whose float32 exp can
        # differ from torch's by one ulp (measured); either is a correct float32 exp
        p = torch.from_numpy(np.exp((s[j] - m).numpy())) if mode == "interp" else torch.exp(s[j] - m)
        den = den + p
        acc = acc + k4[j] * p
    x = _bf16(acc / den, mode)
    x = _bf16(fwht128(x) * torch.tensor(0.08838834764831845, dtype=torch.float32), mode)
    absmax = torch.clamp(x.abs().max(), min=1e-4)
    inv = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32)
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(absmax * inv)))
    else:
        scale = absmax * inv
    qf = torch.clamp(x / scale, -FP8_MAX, FP8_MAX)
    q = store_cast_e4m3(qf) if mode == "interp" else qf.to(torch.float8_e4m3fn)
    return torch.cat([q.view(torch.uint8), scale.reshape(1).view(torch.uint8)])


def dequant(r):
    q = r[:128].clone().view(torch.float8_e4m3fn).float()
    s = r[128:132].clone().view(torch.float32)
    return q * s


class Req:
    """A request's own token data by logical position, plus its block ids."""

    def __init__(self, name, n_tokens, idx_blocks, tail_block, gen, ppb=1088):
        self.name = name
        self.k = torch.randn(n_tokens, HD, generator=gen).to(torch.bfloat16)
        self.g = torch.randn(n_tokens, HD, generator=gen).to(torch.bfloat16)
        self.idx_blocks = list(idx_blocks)  # index-K block id per 4*ppb tokens
        self.tail_block = tail_block
        self.ppb = ppb

    def pool_loc(self, pool):
        return self.idx_blocks[pool // self.ppb] * self.ppb + pool % self.ppb

    def slot(self, p):
        return self.pool_loc(p // KP) if (p + 1) % KP == 0 else -1


def classify(raw, req, pool, ape, cross_members=None):
    """'ok' (== own-token reference), 'unwritten' (sentinel bytes), 'cross' (== pooling of the given
    batch members), or 'other'."""
    got = rec(raw, req.pool_loc(pool))
    ref = ref_record(req.k[pool * KP: pool * KP + KP], req.g[pool * KP: pool * KP + KP], ape)
    if torch.equal(got, ref):
        return "ok"
    if bool((got == SENT).all()):
        return "unwritten"
    if cross_members is not None and torch.equal(got, ref_record(*cross_members, ape)):
        return "cross"
    return "other"


def fwht_matches_sylvester(n=64):
    """Sanity: the butterfly equals the Sylvester-matrix transform (max abs difference)."""
    x = torch.randn(n, 128, generator=torch.Generator().manual_seed(1))
    return max(float((fwht128(r) - r @ _H).abs().max()) for r in x)
