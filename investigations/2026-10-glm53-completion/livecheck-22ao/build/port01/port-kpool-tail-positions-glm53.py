#!/usr/bin/env python3
"""Address the kpool tail ring by token position (upstream vLLM #53906 and #59528).

MambaHybridModelState.prepare_attn builds the attention metadata without
``positions`` (the default model state passes them). KpoolTailMetadataBuilder
builds the circular tail mapping ``own_block * ring + pos % ring`` only when
positions are present; otherwise the generic per-group mapping is used, which
sends a tail write at position p >= ring to column p // ring of the tail block
table: the shared null block 0, or a block id left in column 1 by the start-up
warm-up. Decoding requests then share block 0's ring, so a pool completed during
decode can be compressed from another request's keys, and a stale column can
receive raw key/gate rows of positions 12-15 inside another request's index-K
page.

Three changes, as upstream: pass ``positions=input_batch.positions``; give rows
on the null block (padding and dummy rows) PAD instead of block 0's ring
(``own_block != 0``, #59528); and copy the circular mapping into the persistent
per-group slot-mapping buffer, because the indexer op is not an eager break under
FULL CUDA graphs and replays read the capture-time address (upstream uses a
builder-owned buffer; either works here). With positions visible, the B12X
sparse-MLA metadata builder also takes causal lengths from ``positions + 1`` for
batches that are not pure single-token decode, as upstream does.
Usage: port-kpool-tail-positions-glm53.py VLLM_DIR
"""

import os
import sys
import tempfile
from pathlib import Path

vllm = Path(sys.argv[1])
PENDING: list[tuple[Path, str]] = []


def patch(rel: str, edits: list[tuple[str, str, int]], marker: str) -> None:
    """Validate every edit before anything is written (see the end of the file)."""
    path = vllm / rel
    source = path.read_text()
    if marker in source:
        print(f"{rel}: already applied")
        return
    for old, new, count in edits:
        if source.count(old) != count:
            raise SystemExit(f"{rel}: refusing, source drift ({source.count(old)} != {count}):{chr(10)}{old}")
        source = source.replace(old, new)
    compile(source, str(path), "exec")
    PENDING.append((path, source))


patch('v1/worker/gpu/model_states/mamba_hybrid.py', [
    (
"""            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
""",
"""            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            positions=input_batch.positions,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
""",
        1,
    ),
], marker='positions=input_batch.positions,')

patch('v1/attention/backends/mla/indexer.py', [
    (
'''    ``slot = own_block * kpool + pos % kpool`` instead, which is the layout
    the tail kernels and ``KpoolTailManager`` are designed around.

    Pure torch (no Triton, no device sync): the indexer op consumes the tail
    slot mapping on its eager break, so the returned tensor need not be the
    persistent ``BlockTables`` buffer.
    """
    out = slot_mapping.clone()
    if num_actual_tokens == 0:
''',
'''    ``slot = own_block * kpool + pos % kpool`` instead, which is the layout
    the tail kernels and ``KpoolTailManager`` are designed around.

    Pure torch (no Triton, no device sync). The indexer op is not an eager
    break under FULL CUDA graphs, so the caller copies the result into the
    persistent per-group slot-mapping buffer that replayed graphs read.
    Requests on the null block (dummy and padding rows) get PAD (-1).
    """
    out = slot_mapping.clone()
    if num_actual_tokens == 0:
''',
        1,
    ),
    (
"""    req = req.clamp_(min=0, max=num_reqs - 1)
    own_block = block_table[:num_reqs, 0].index_select(0, req).to(torch.int64)
    pos = positions[:num_actual_tokens].to(torch.int64)
    out[:num_actual_tokens] = own_block * kpool + torch.remainder(pos, kpool)
    return out
""",
"""    req = req.clamp_(min=0, max=num_reqs - 1)
    own_block = block_table[:num_reqs, 0].index_select(0, req).to(torch.int64)
    pos = positions[:num_actual_tokens].to(torch.int64)
    # Every scheduled request owns a tail block; the runners give dummy and
    # padding requests the null block, which must never be written.
    slots = own_block * kpool + torch.remainder(pos, kpool)
    out[:num_actual_tokens] = torch.where(own_block != 0, slots, -1)
    return out
""",
        1,
    ),
    (
"""                common_attn_metadata.num_reqs,
                self.kv_cache_spec.block_size,
            )
        return DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
""",
"""                common_attn_metadata.num_reqs,
                self.kv_cache_spec.block_size,
            )
            # FULL CUDA graph replay reads the capture-time address, so the
            # circular mapping must live in the persistent per-group buffer
            # (a view of BlockTables.slot_mappings), not in a fresh tensor.
            common_attn_metadata.slot_mapping.copy_(slot_mapping)
            slot_mapping = common_attn_metadata.slot_mapping
        return DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
""",
        1,
    ),
], marker='torch.where(own_block != 0, slots, -1)')



for path, source in PENDING:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".port-")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(source)
        os.chmod(tmp, os.stat(path).st_mode & 0o777)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

print('GLM kpool tail ring addressed by position, null-block guard, graph-safe mapping (vLLM #53906, #59528)')
