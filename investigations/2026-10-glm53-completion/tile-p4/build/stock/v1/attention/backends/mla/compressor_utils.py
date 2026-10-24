# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _compressed_slot_mapping_kernel(
    # [num_tokens]
    slot_mapping_ptr,
    # [num_reqs + 1]
    query_start_loc_ptr,
    # [num_reqs]
    seq_lens_ptr,
    # [num_reqs, max_num_blocks]
    block_table_ptr,
    block_table_stride,
    block_size,
    COMPRESS_RATIO: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    DCP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    start_pos = seq_len - query_len

    for i in range(0, query_len, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < query_len

        pos = start_pos + i + tl.arange(0, TRITON_BLOCK_SIZE)
        completes_pool = (pos + 1) % COMPRESS_RATIO == 0
        global_pool = pos // COMPRESS_RATIO
        pool_group = global_pool // DCP_INTERLEAVE
        pool_offset = global_pool % DCP_INTERLEAVE
        is_local = pool_group % DCP_WORLD_SIZE == DCP_RANK
        is_valid = completes_pool & is_local
        local_pool = (
            pool_group // DCP_WORLD_SIZE * DCP_INTERLEAVE + pool_offset
        )

        block_ids = local_pool // block_size
        block_numbers = tl.load(
            block_table_ptr + batch_idx * block_table_stride + block_ids,
            mask=mask & is_valid,
        )
        slot_ids = block_numbers * block_size + local_pool % block_size

        # NOTE
        slot_ids = tl.where(is_valid, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + query_start + offset, slot_ids, mask=mask)


def get_compressed_slot_mapping(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert dcp_world_size >= 1
    assert 0 <= dcp_rank < dcp_world_size
    assert cp_kv_cache_interleave_size >= 1
    assert block_size % cp_kv_cache_interleave_size == 0, (
        f"compressed cache block_size ({block_size}) must be divisible by "
        f"DCP interleave ({cp_kv_cache_interleave_size})"
    )
    if out is not None:
        # Guard: for padded / invalid sequences.
        # Negative positions produce bogus block indices that lead to illegal memory
        # accesses inside the block_table load.
        # NOTE: Fill -1 to the whole tensor, not just the first `num_tokens`.
        out.fill_(-1)
        slot_mapping = out[:num_tokens]
    else:
        slot_mapping = torch.full(
            (num_tokens,), -1, dtype=torch.int64, device=query_start_loc.device
        )

    num_reqs = block_table.shape[0]
    _compressed_slot_mapping_kernel[(num_reqs,)](
        slot_mapping,
        query_start_loc,
        seq_lens,
        block_table,
        block_table.stride(0),
        block_size,
        compress_ratio,
        DCP_WORLD_SIZE=dcp_world_size,
        DCP_RANK=dcp_rank,
        DCP_INTERLEAVE=cp_kv_cache_interleave_size,
        PAD_ID=-1,
        TRITON_BLOCK_SIZE=1024,
    )
    return slot_mapping
