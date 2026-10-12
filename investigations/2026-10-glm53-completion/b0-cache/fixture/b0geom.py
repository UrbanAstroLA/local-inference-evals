"""B0 cache geometry (checkpoint B0, image 08d1cb90..., rank 0) and byte footprints.

Pure Python (no torch needed). Every number is derived, then cross-checked against the
B0 server log line `glm53-placement kv-capacity` (screen run-d1234/server-s1.log:241) by
check_against_log().

Sources (B0 vLLM tree):
  v1/core/kv_cache_utils.py:1419-1471  draft slot sharing (block 1088, page padded to MLA page)
  v1/core/kv_cache_utils.py:1483-1580  groups: [MLA+idx] [tail] [7 x mamba] [draft]
  v1/core/kv_cache_utils.py:1745-1795  tensors: per MLA slot (MLA + KDA_g[i] + draft[i]); per idx (idx + tail)
  v1/worker/gpu/attn_utils.py:271-343  padded pages -> as_strided block stride = page_size_padded
  model_executor/layers/mamba/abstract.py:29-43, mamba_utils.py:396-419  KDA conv|ssm packing
  v1/attention/backends/flash_attn.py:138-170  draft view (B, H, N, 2D), NHD order -> physical (B, N, H, 2D)
"""
from __future__ import annotations

NUM_BLOCKS = 1275
NULL_BLOCK = 0
TWO31 = 1 << 31

# MLA (NOPE / GLM_NEXT record, fp8_ds_mla, 528 B per token, DCP1)
MLA_BLOCK_TOKENS = 4352
MLA_RECORD = 528
MLA_PAGE = MLA_BLOCK_TOKENS * MLA_RECORD  # 2,297,856
MLA_TENSOR = MLA_PAGE * NUM_BLOCKS  # 2,929,766,400

# kpool indexer (compress ratio 4): 1088 pools x (128 fp8 + 4 B scale)
IDX_POOLS = MLA_BLOCK_TOKENS // 4
IDX_RECORD = 132
IDX_PAGE = IDX_POOLS * IDX_RECORD  # 143,616
IDX_TENSOR = IDX_PAGE * NUM_BLOCKS
# kpool tail ring: [2 (K|gate), 8 slots, 128] bf16 at the start of the idx page (KpoolTailSpec block 8)
TAIL_SLOTS = 8
TAIL_HEAD = 128
TAIL_REAL = 2 * TAIL_SLOTS * TAIL_HEAD * 2  # 4,096 B = idx kernel page 17b, K rows 0..31

# KDA (34 layers, 7 mamba groups, TP2 -> 32 heads x 128): conv (6, 12288) bf16 | ssm (32,128,128) fp32
KDA_CONV_W = 4 - 1 + 3  # conv_kernel - 1 + num_spec
KDA_CONV_DIM = 3 * 32 * 128
KDA_CONV_BYTES = KDA_CONV_W * KDA_CONV_DIM * 2  # 147,456
KDA_SSM_BYTES = 32 * 128 * 128 * 4  # 2,097,152
KDA_REAL = KDA_CONV_BYTES + KDA_SSM_BYTES  # 2,244,608 (padded 2.37% to MLA_PAGE)

# DFlash2 draft (5 sliding layers, 8 KV heads / TP2 = 4, head 128, bf16), block 1088
DRAFT_BLOCK_TOKENS = 1088
DRAFT_HEADS = 4
DRAFT_HEAD = 128
DRAFT_TOKEN_BYTES = DRAFT_HEADS * 2 * DRAFT_HEAD * 2  # K|V packed per head: 2,048
DRAFT_REAL = DRAFT_BLOCK_TOKENS * DRAFT_TOKEN_BYTES  # 2,228,224

RANK0_MLA_LAYERS = [3, 7, 11, 15, 19, 23]
RANK1_MLA_LAYERS = [27, 31, 35, 39, 43]
NUM_MAMBA_GROUPS = 7  # cdiv(34, min owned MLA count 5)
NUM_DRAFT_LAYERS = 5


def first_high_block(page: int = MLA_PAGE) -> int:
    """First block whose page starts at or above 2^31 bytes."""
    return -(-TWO31 // page)


def first_high_mla_slot() -> int:
    """First MLA slot whose record END passes 2^31 (the straddling record)."""
    return (TWO31 - 1) // MLA_RECORD


def mla_record_range(slot: int) -> tuple[int, int]:
    o = slot * MLA_RECORD
    return o, o + MLA_RECORD


def mla_slot(block: int, tok: int) -> int:
    return block * MLA_BLOCK_TOKENS + tok


def draft_token_range(block: int, tok: int) -> tuple[int, int]:
    """Bytes of all K and V heads of one draft token (NHD: (B, N, H, 2D))."""
    o = block * MLA_PAGE + tok * DRAFT_TOKEN_BYTES
    return o, o + DRAFT_TOKEN_BYTES


def draft_k_range(block: int, tok: int, head: int) -> tuple[int, int]:
    o = block * MLA_PAGE + tok * DRAFT_TOKEN_BYTES + head * 2 * DRAFT_HEAD * 2
    return o, o + DRAFT_HEAD * 2


def draft_v_range(block: int, tok: int, head: int) -> tuple[int, int]:
    s, e = draft_k_range(block, tok, head)
    return s + DRAFT_HEAD * 2, e + DRAFT_HEAD * 2


def kda_conv_range(block: int) -> tuple[int, int]:
    o = block * MLA_PAGE
    return o, o + KDA_CONV_BYTES


def kda_ssm_range(block: int) -> tuple[int, int]:
    o = block * MLA_PAGE + KDA_CONV_BYTES
    return o, o + KDA_SSM_BYTES


def page_range(block: int, page: int = MLA_PAGE) -> tuple[int, int]:
    return block * page, (block + 1) * page


# Kernel views (B12x backend kernel block 64; attn_utils.py:388-430 virtual split):
#   MLA   uint8 (86,700, 64, 528) contiguous, stride (33,792, 528, 1)   [b12x_mla_sparse.py:1012-1016, 1074-1090]
#   idx   uint8 (21,675, 64, 132) contiguous, page-planar: 64x128 E4M3 K, then 64 fp32 scales at byte 8,192
#   tail  bf16 as_strided (1275, 2, 8, 128), stride (71,808, 1,024, 128, 1) elements
MLA_KPAGE = 64
MLA_KPAGES_PER_BLOCK = MLA_BLOCK_TOKENS // MLA_KPAGE  # 68
IDX_KPAGE = 64
IDX_KPAGES_PER_BLOCK = IDX_POOLS // IDX_KPAGE  # 17
IDX_KPAGE_BYTES = IDX_KPAGE * IDX_RECORD  # 8,448


def idx_k_range(block: int, pool: int) -> tuple[int, int]:
    """E4M3 index-K bytes of pool `pool` (0..1087) of storage block `block`."""
    kp = block * IDX_KPAGES_PER_BLOCK + pool // IDX_KPAGE
    o = kp * IDX_KPAGE_BYTES + (pool % IDX_KPAGE) * 128
    return o, o + 128


def idx_scale_range(block: int, pool: int) -> tuple[int, int]:
    kp = block * IDX_KPAGES_PER_BLOCK + pool // IDX_KPAGE
    o = kp * IDX_KPAGE_BYTES + IDX_KPAGE * 128 + (pool % IDX_KPAGE) * 4
    return o, o + 4


def tail_range(block: int) -> tuple[int, int]:
    o = block * IDX_PAGE
    return o, o + TAIL_REAL


def per_block_bytes(n_mla: int) -> int:
    return n_mla * MLA_PAGE + n_mla * IDX_PAGE


def check_against_log(log_line: str | None = None) -> dict:
    """Cross-check derived numbers with the B0 log (server-s1.log:240-242)."""
    out = {
        "mla_page": MLA_PAGE,
        "mla_tensor_bytes": MLA_TENSOR,
        "idx_page": IDX_PAGE,
        "kda_real": KDA_REAL,
        "kda_padding_pct": round(100 * (MLA_PAGE / KDA_REAL - 1), 2),
        "draft_real": DRAFT_REAL,
        "rank0_tensor_bytes": NUM_BLOCKS * per_block_bytes(len(RANK0_MLA_LAYERS)),
        "rank1_tensor_bytes": NUM_BLOCKS * per_block_bytes(len(RANK1_MLA_LAYERS)),
        "first_high_block": first_high_block(),
        "first_high_slot_straddling": first_high_mla_slot(),
        "straddle_bytes": mla_record_range(first_high_mla_slot()),
        "high_blocks": NUM_BLOCKS - first_high_block(),
        "kv_tokens": None,
    }
    assert out["kda_padding_pct"] == 2.37, out
    assert out["rank0_tensor_bytes"] == 18_677_260_800, out
    assert DRAFT_REAL == 2_228_224
    assert out["first_high_block"] == 935
    assert out["first_high_slot_straddling"] == 4_067_203
    assert IDX_KPAGES_PER_BLOCK * IDX_KPAGE_BYTES == IDX_PAGE and MLA_KPAGES_PER_BLOCK * MLA_KPAGE * MLA_RECORD == MLA_PAGE
    s, e = out["straddle_bytes"]
    assert s < TWO31 < e
    if log_line:
        import json, re
        m = re.search(r"kv-capacity (\{.*\})", log_line)
        d = json.loads(m.group(1))
        assert d["num_blocks"] == NUM_BLOCKS and d["tensor_bytes"] == out["rank0_tensor_bytes"]
        pages = {g["spec"]: g["page_bytes"] for g in d["groups"]}
        assert pages["MambaSpec"] == MLA_PAGE and pages["SlidingWindowSpec"] == MLA_PAGE
        out["log_groups"] = [(g["spec"], g["layers"], g["block_size"], g["page_bytes"]) for g in d["groups"]]
    return out


if __name__ == "__main__":
    import json, sys
    line = None
    if len(sys.argv) > 1:
        for l in open(sys.argv[1], errors="replace"):
            if "glm53-placement kv-capacity" in l:
                line = l
                break
    print(json.dumps(check_against_log(line), indent=1))
