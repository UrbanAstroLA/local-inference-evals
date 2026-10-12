"""Build the B0 kernel views of one MLA slot tensor and one idx tensor with the REAL B0
code paths (no hand-written strides):

  vllm.v1.worker.gpu.attn_utils._reshape_attention_kv_cache   (attention views, padded pages)
  backend.get_kv_cache_shape / get_kv_cache_stride_order       (B12X_MLA_SPARSE, FLASH_ATTN,
                                                                DEEPSEEK_V32_INDEXER, KPOOL_TAIL)
  MambaBase.bind_kv_cache + MambaStateShape/DtypeCalculator    (KDA conv|ssm packing)

Requires VLLM_GLM53_NOPE_RECORD=1 in the environment before vLLM import (528-B record).
Works on CPU tensors (CPU test) and CUDA tensors (fixture).
"""
from __future__ import annotations

import os

assert os.environ.get("VLLM_GLM53_NOPE_RECORD") == "1", "set VLLM_GLM53_NOPE_RECORD=1"

import torch

import b0geom as G


def _backends():
    # B0 has no kv_transfer_config, so get_kv_connector_cache_layout() returns "NHD"
    # (distributed/kv_transfer/kv_connector/utils.py:38-52); pin it without a VllmConfig.
    from vllm.v1.attention.backends.utils import set_kv_cache_layout
    set_kv_cache_layout("NHD")
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseBackend
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerBackend, KpoolTailBackend,
    )
    return B12xMLASparseBackend, FlashAttentionBackend, DeepseekV32IndexerBackend, KpoolTailBackend


def specs():
    from vllm.v1.kv_cache_interface import (
        KpoolTailSpec, MLAAttentionSpec, SlidingWindowSpec,
    )
    mla = MLAAttentionSpec(block_size=G.MLA_BLOCK_TOKENS, num_kv_heads=1, head_size=512,
                           dtype=torch.uint8, cache_dtype_str="fp8_ds_mla")
    idx = MLAAttentionSpec(block_size=G.MLA_BLOCK_TOKENS, num_kv_heads=1, head_size=G.IDX_RECORD,
                           dtype=torch.uint8, compress_ratio=4)
    tail = KpoolTailSpec(block_size=G.TAIL_SLOTS, num_kv_heads=1, head_size=2 * G.TAIL_HEAD,
                         head_size_v=0, dtype=torch.bfloat16, sliding_window=G.TAIL_SLOTS,
                         page_size_padded=G.IDX_PAGE)
    draft = SlidingWindowSpec(block_size=G.DRAFT_BLOCK_TOKENS, num_kv_heads=G.DRAFT_HEADS,
                              head_size=G.DRAFT_HEAD, dtype=torch.bfloat16, sliding_window=2048,
                              page_size_padded=G.MLA_PAGE)
    return mla, idx, tail, draft


def _attn_view(raw, spec, backend, kernel_num_blocks, kernel_block, num_kv_heads, head_size,
               cache_dtype_str):
    from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache
    shape = backend.get_kv_cache_shape(kernel_num_blocks, kernel_block, num_kv_heads, head_size,
                                       cache_dtype_str=cache_dtype_str)
    try:
        order = backend.get_kv_cache_stride_order()
        assert len(order) == len(shape)
    except (AttributeError, NotImplementedError):
        order = tuple(range(len(shape)))
    return _reshape_attention_kv_cache(raw, spec, shape, order, kernel_num_blocks, None)


def mla_view(raw):
    B12x, _, _, _ = _backends()
    mla, _, _, _ = specs()
    nb = G.NUM_BLOCKS * G.MLA_KPAGES_PER_BLOCK
    v = _attn_view(raw, mla, B12x, nb, G.MLA_KPAGE, 1, 512, "fp8_ds_mla")
    assert tuple(v.shape) == (nb, 64, 528), v.shape
    return v


def draft_view(raw):
    """FlashAttention logical (B, H, N, 2D); padded block stride = MLA page."""
    _, FA, _, _ = _backends()
    _, _, _, draft = specs()
    return _attn_view(raw, draft, FA, G.NUM_BLOCKS, G.DRAFT_BLOCK_TOKENS, G.DRAFT_HEADS,
                      G.DRAFT_HEAD, "auto")


def draft_kv_split(kv):
    """Exactly FlashAttentionImpl.do_kv_cache_update / forward (flash_attn.py:989, 1201)."""
    return kv.transpose(1, 2).split(G.DRAFT_HEAD, dim=-1)


def idx_view(raw_idx):
    _, _, IDX, _ = _backends()
    _, idx, _, _ = specs()
    nb = G.NUM_BLOCKS * G.IDX_KPAGES_PER_BLOCK
    v = _attn_view(raw_idx, idx, IDX, nb, G.IDX_KPAGE, 1, G.IDX_RECORD, "auto")
    assert tuple(v.shape) == (nb, 64, 132), v.shape
    return v


def tail_view(raw_idx):
    _, _, _, TAIL = _backends()
    _, _, tail, _ = specs()
    return _attn_view(raw_idx, tail, TAIL, G.NUM_BLOCKS, G.TAIL_SLOTS, 1, 2 * G.TAIL_HEAD, "auto")


class _FakeKDA:
    """Carries only what MambaBase.bind_kv_cache needs (abstract.py:29-43)."""

    def __init__(self):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateDtypeCalculator, MambaStateShapeCalculator,
        )
        self._shape = MambaStateShapeCalculator.kda_state_shape(
            2, 64, 128, conv_kernel_size=4, num_spec=3)
        self._dtype = MambaStateDtypeCalculator.kda_state_dtype(torch.bfloat16, "auto")

    def get_state_shape(self):
        return self._shape

    def get_state_dtype(self):
        return self._dtype


def kda_views(raw):
    """(conv_state as the kernels see it (B, dim, width), ssm (B, 32, 128, 128) fp32)."""
    from vllm.model_executor.layers.mamba.abstract import MambaBase
    f = _FakeKDA()
    page = raw[: G.NUM_BLOCKS * G.MLA_PAGE].view(G.NUM_BLOCKS, 1, 1, G.MLA_PAGE)  # attn_utils.py:447-451
    MambaBase.bind_kv_cache(f, page)
    conv, ssm = f.kv_cache
    # kda.py:450 — SD layout (VLLM_SSM_CONV_STATE_LAYOUT unset in B0) is transposed for kernels
    from vllm.model_executor.layers.mamba import mamba_utils as MU
    dim_first = getattr(MU, "is_conv_state_dim_first", lambda: False)()
    conv_k = conv if dim_first else conv.transpose(-1, -2)
    return conv, conv_k, ssm, f


def byte_offset(t: torch.Tensor, index: tuple, base: torch.Tensor) -> int:
    """Byte offset of element `index` of view `t` relative to raw tensor `base`."""
    off = t.storage_offset() + sum(i * s for i, s in zip(index, t.stride()))
    return off * t.element_size() - base.storage_offset() * base.element_size()
