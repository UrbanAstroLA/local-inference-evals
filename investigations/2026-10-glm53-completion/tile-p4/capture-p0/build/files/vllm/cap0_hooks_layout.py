# SPDX-License-Identifier: Apache-2.0
"""local-cap0 layout: the byte addressing of the point-0 capture slices (capture-p0/PROTOCOL.md section 2).

Pure python (no torch, no numpy).  One source for the served hook module (installed as vllm/cap0_hooks_layout.py)
and for the offline analysis (imported by path).  Every constant is a served value recorded in the Stage-0 records
or read in source (audit-s0/AUDIT.md tasks 2-4); the hook module re-checks the ones it can against the live views at
registration (storage size, storage offset, page stride) and refuses to call a slice valid when they disagree.
"""

BLOCK_TOKENS = 4352            # KV-cache block (MLA group and KDA groups), server.log block sizes
KERNEL_TOKENS = 64             # MLA kernel page (tokens)
MLA_PPB = BLOCK_TOKENS // KERNEL_TOKENS          # 68 kernel pages per logical block
MLA_REC = 528                  # bytes per token record: [0, 512) E4M3 latent (4 x 128), [512, 528) 4 fp32 scales
MLA_LATENT = 512
MLA_GROUP = 128
MLA_NGROUP = 4
P = MLA_PPB * KERNEL_TOKENS * MLA_REC            # 2,297,856 B: page of one MLA KVCacheTensor block
KPOOL = 4                      # index_kpool (config.json)
IDX_POOLS = BLOCK_TOKENS // KPOOL                # 1,088 pool records per block
IDX_ROWS = 64                  # records per pool page
IDX_PPB = IDX_POOLS // IDX_ROWS                  # 17 pool pages per block
IDX_VAL = 128                  # E4M3 values per record
IDX_POOL_PAGE = IDX_ROWS * IDX_VAL + IDX_ROWS * 4               # 8,448 B
IDX_SCALE_OFF = IDX_ROWS * IDX_VAL                              # 8,192
IDX_PAGE = IDX_PPB * IDX_POOL_PAGE                              # 143,616 B per block of an indexer KVCacheTensor
CONV_ROWS = 6                  # conv_kernel - 1 + num_speculative_tokens = 3 + 3 (SD layout [6, 12288] bf16)
CONV_LIVE_ROWS = 3
CONV_ROW_BYTES = 12288 * 2
CONV_BYTES = CONV_ROWS * CONV_ROW_BYTES                         # 147,456
CONV_LIVE_BYTES = CONV_LIVE_ROWS * CONV_ROW_BYTES               # 73,728
REC_OFF = CONV_BYTES                                            # recurrent state follows the conv state
REC_BYTES = 32 * 128 * 128 * 4                                  # 2,097,152 (fp32 [32, 128, 128])
KDA_USED = REC_OFF + REC_BYTES                                  # 2,244,608 of P (rest is page padding)
CHUNK_BUDGET = 2045            # max_num_batched_tokens 2048 minus 3 draft slots (Stage-0 sched rows)
N_CACHED = 13056               # R01-early: P = 17,409 -> ((P - 1) // 4352 - 1) * 4352
P_EARLY = 17409
# Amendment A1 (2026-10-24, after session attempt 1): the warm is the FULL capture prompt (17,409 tokens, 4 cached
# blocks).  The attempt-1 warm of 13,057 tokens cached 3 blocks, and a request hitting a cached prefix gets back one
# block fewer than it matches (PROPOSAL-AMENDMENTS.md A3, the EAGLE-type drop), so the capture request got 8,704,
# not 13,056.  See cache_hit().
WARM_TOKENS = P_EARLY


def cache_hit(warm_tokens, prompt_tokens, block=BLOCK_TOKENS):
    """Tokens a request of prompt_tokens gets from the prefix cache after a warm of warm_tokens identical ids (same
    salt): min(full blocks the warm cached, floor((P - 1) / block)) minus the one dropped block.  Measured: Stage 0
    (warm 113,940, P 17,409) -> 13,056; capture-p0 attempt 1 (warm 13,057) -> 8,704."""
    usable = min(warm_tokens // block, (prompt_tokens - 1) // block)
    return max(0, usable - 1) * block


def mla_block(b):
    """Byte range of physical block b in an MLA KVCacheTensor."""
    return b * P, (b + 1) * P


def mla_record(b, pos):
    """Byte range of the 528-B record of in-block token position pos (0 <= pos < 4352) of physical block b:
    kernel page b * 68 + pos // 64, row pos % 64 -> b * P + pos * 528."""
    assert 0 <= pos < BLOCK_TOKENS
    lo = b * P + (pos // KERNEL_TOKENS) * KERNEL_TOKENS * MLA_REC + (pos % KERNEL_TOKENS) * MLA_REC
    return lo, lo + MLA_REC


def idx_block(b):
    """Byte range of physical block b in an indexer (pool) KVCacheTensor."""
    return b * IDX_PAGE, (b + 1) * IDX_PAGE


def idx_pool(q):
    """In-block byte ranges of pool record q (0 <= q < 1088): (values lo, hi), (scale lo, hi).
    Pool page q // 64, row q % 64; values [p*8448 + r*128, +128), fp32 scale at p*8448 + 8192 + 4r."""
    assert 0 <= q < IDX_POOLS
    p, r = divmod(q, IDX_ROWS)
    v = p * IDX_POOL_PAGE + r * IDX_VAL
    s = p * IDX_POOL_PAGE + IDX_SCALE_OFF + 4 * r
    return (v, v + IDX_VAL), (s, s + 4)


def kda_conv(b):
    """(live rows 0-2, inactive rows 3-5) byte ranges of the KDA conv state of physical block b (co-owning tensor)."""
    return (b * P, b * P + CONV_LIVE_BYTES), (b * P + CONV_LIVE_BYTES, b * P + CONV_BYTES)


def kda_rec(b):
    """Byte range of the KDA recurrent state of physical block b (co-owning tensor)."""
    return b * P + REC_OFF, b * P + REC_OFF + REC_BYTES


def cached_state_col(n_cached):
    """Logical KDA column holding the cached state at point 0 (align mode): n_cached / 4352 - 1."""
    assert n_cached % BLOCK_TOKENS == 0 and n_cached > 0
    return n_cached // BLOCK_TOKENS - 1


def precopy_dst_col(n_cached, n_sched):
    """Logical KDA column the align-mode pre-copy writes before the first remainder step:
    ceil((n_cached + n_sched) / 4352) - 1 (w_mamba_utils.py:425-436)."""
    return (n_cached + n_sched + BLOCK_TOKENS - 1) // BLOCK_TOKENS - 1


def physical(entries, j, bpk):
    """Physical block id of logical block j from a block-table row of kernel-block entries (bpk per block)."""
    return int(entries[j * bpk]) // bpk


def warm_chunks(n_tokens, budget=CHUNK_BUDGET, block=BLOCK_TOKENS, start=0):
    """Align-mode chunked prefill of one solitary request: each step takes min(budget, tokens to the next block
    boundary, tokens left).  Reproduces the Stage-0 warm rows (2,045 / 2,045 / 262 per 4,352-token block)."""
    out, s = [], start
    while s < n_tokens:
        n = min(budget, block - s % block, n_tokens - s)
        out.append((s, n))
        s += n
    return out


def never_written_pools(chunks, block_index=0, kpool=KPOOL):
    """In-block pool indices of logical block `block_index` that the prefill pool writer never writes: a pool is
    written only by the chunk holding its last token, and only when that token sits at batch position >= kpool - 1
    (kpool.py:97-103), i.e. when the chunk starts at or before the pool's first token."""
    lo, hi = block_index * BLOCK_TOKENS, (block_index + 1) * BLOCK_TOKENS
    out = []
    for q in range(lo // kpool, hi // kpool):
        last = q * kpool + kpool - 1
        for s, n in chunks:
            if s <= last < s + n:
                if last - s < kpool - 1:
                    out.append(q - lo // kpool)
                break
    return out


def chunk_of(pos, chunks):
    """Index of the chunk that computed absolute token position pos."""
    for k, (s, n) in enumerate(chunks):
        if s <= pos < s + n:
            return k
    return -1


def e4m3_table():
    """Decoded value of every E4M3 (fn) byte; 0x7f / 0xff (NaN) -> None."""
    out = []
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0xF and m == 0x7:
            out.append(None)
        elif e == 0:
            out.append(s * (m / 8.0) * 2.0 ** -6)
        else:
            out.append(s * (1.0 + m / 8.0) * 2.0 ** (e - 7))
    return out


# Localisation boundaries in computation order within one warm step (PROTOCOL.md section 3).  Each entry:
# (point, tensors, what it is).  The analysis reads the first point (chunk-major) whose rows differ.
LOC_ORDER = (
    ("L3.in", ("x", "residual", "post", "comb"), "state entering layer 3 (layer 2's output, deferred mHC post)"),
    ("L3.pre_attn", ("residual", "post", "comb", "y"), "layer 3 fused mHC post(2)+pre(attn) outputs; y = attention input"),
    ("L3.attn_in", ("x",), "attention module input (identical tensor to L3.pre_attn.y)"),
    ("L3.topk", ("ids",), "indexer top-k token ids of layer 3 (rank 0; the MLA owner)"),
    ("L3.attn_out", ("x",), "attention output after the TP all-reduce (input of the ffn-side fused mHC op)"),
    ("L3.pre_ffn", ("residual", "post", "comb", "y"), "layer 3 fused mHC post(attn)+pre(ffn) outputs; y = MoE input"),
    ("L3.moe_out", ("x",), "MoE output after the TP all-reduce (routed + shared)"),
    ("L4.in", ("x", "residual", "post", "comb"), "state entering layer 4 (= L3.moe_out, L3.pre_ffn residual/post/comb)"),
    ("L4.pre_attn", ("residual", "post", "comb", "y"), "layer 4 fused mHC post(3)+pre(attn): materialised residual into layer 4"),
    ("L4.conv_out", ("qkv",), "layer 4 KDA short-conv output (causal_conv1d_fn, prefill path)"),
    ("L4.rec_out", ("o",), "layer 4 KDA core output (chunk_kda, before o_norm)"),
    ("L4.attn_out", ("x",), "layer 4 KDA output after o_proj (and its all-reduce)"),
)
STATE_POINTS = ("L2.state_end", "L4.state_end")   # per warm step: the running column after scatter (rec, conv rows 0-2)
# Identities that hold by source (model.py Glm5NextDecoderLayer.forward); a violation means the instrument is wrong.
LOC_IDENTITIES = (
    (("L3.pre_attn", "y"), ("L3.attn_in", "x")),
    (("L4.in", "x"), ("L3.moe_out", "x")),
    (("L4.in", "residual"), ("L3.pre_ffn", "residual")),
    (("L4.in", "post"), ("L3.pre_ffn", "post")),
    (("L4.in", "comb"), ("L3.pre_ffn", "comb")),
)
# The operation that lies between a boundary and the one before it (the reading when that boundary is the first to
# differ and every earlier boundary of the same step, and every earlier step, is equal).
LOC_OPERATION = {
    "L3.in": "before layer 3 (layers 0-2 or the embedding): not inside the measured boundaries",
    "L3.pre_attn": "layer-3 fused mHC post+pre (attention side; tilelang mhc_fused_post_pre)",
    "L3.attn_in": "instrument inconsistency (identical tensor to L3.pre_attn.y)",
    "L3.topk": "layer-3 kpool indexer (q/k projections, MQA logits over the gathered pool cache, top-k)",
    "L3.attn_out": "layer-3 sparse MLA attention (KV write, gather, attention, o_proj, TP all-reduce)",
    "L3.pre_ffn": "layer-3 fused mHC post+pre (ffn side)",
    "L3.moe_out": "layer-3 MoE (router, EXL3 routed experts, shared expert, TP all-reduce)",
    "L4.in": "instrument inconsistency (identity with layer-3 outputs)",
    "L4.pre_attn": "layer-4 fused mHC post+pre (materialises layer 3's residual)",
    "L4.conv_out": "layer-4 KDA short conv (causal_conv1d_fn with the initial conv state)",
    "L4.rec_out": "layer-4 KDA chunked recurrence (gather_initial_states + chunk_kda)",
    "L4.attn_out": "layer-4 o_norm + o_proj",
    "L4.state_end": "layer-4 recurrent/conv state write (scatter_states) or the align-mode pre-copy",
    "L2.state_end": "layer-2 state write (control)",
}
