#!/usr/bin/env python3
"""Context-length engagement points for the GLM-5.3-Flash K3.25 serving configs.

CPU only, no imports beyond the stdlib. Every constant is copied from the source
cited next to it (paths relative to the extracted trees kdcp/{v070,v091}/ and
b0-cache/src/; B1 = B0 + d7-fix.diff, which touches only the index-K prefill
writer). Output: thresholds.out (same directory).

Model geometry (config.json, snapshot 0490d2f7): index_kpool 4, index_topk 2048
-> select_k 512 pools; qk_rope_head_dim 0 (no RoPE anywhere in the target).
"""

SMS = 188                    # RTX PRO 6000 Blackwell Max-Q (HANDOFF.md:55)
KPOOL = 4                    # config.json index_kpool
TOPK_POOLS = 2048 // KPOOL   # 512 (sparse_attn_indexer_kpool.py:574/:855)
PAGE = 64                    # b12x paged index page, pool rows (paged.py:48)
SLICE = 16384                # _TWO_LEVEL_SLICE_TOKENS, pool rows (paged.py:36)
MAX_SLICES = 32              # _TWO_LEVEL_MAX_SLICES (paged.py:37)
SUPERTILE = 32768            # _PAGED_INDEX_SUPERTILE_K_DEFAULT (paged.py:51; scratch.py:32)
FUSED_MAX_ROWS = 16          # _C4_B12X_FUSED_MAX_ROWS, topk 512 / 32 heads / SM120 (fused_indexer.py v091:140, v070:136)
SLACK = 448                  # _BATCH_SLACK (fused_indexer.py v091:1137, v070:743)
CARRY_CAP = TOPK_POOLS + SLACK   # carry_cap = topk + slack (v091:1605, v070:1197)
MAX_MODEL_LEN = 1_048_576
CAPTURE = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96]  # start-up logs
LOGITS_MB = 512              # VLLM_SPARSE_INDEXER_MAX_LOGITS_MB default (envs.py:1081)

ENGINES = {
    # name: (K drafts, DCP size, attention block tokens per rank, capture max)
    "0.7.0": dict(K=5, dcp=2, block_local=3584, cap_max=96, mamba_block=3584),
    "0.9.1/B0/B1": dict(K=3, dcp=1, block_local=4352, cap_max=64, mamba_block=4352),
}


def cdiv(a, b):
    return -(-a // b)


def padded(rows, cap_max):
    for c in CAPTURE:
        if c >= rows and c <= cap_max:
            return c
    return rows


out = []
P = out.append
for name, e in ENGINES.items():
    K, dcp, blk = e["K"], e["dcp"], e["block_local"]
    global_block = blk * dcp
    w_blocks = cdiv(MAX_MODEL_LEN, global_block)
    pools_per_block_local = blk // KPOOL
    pages_per_block = pools_per_block_local // PAGE
    width_pages = w_blocks * pages_per_block
    width_rows = width_pages * PAGE
    # tokens per local pool row: one pool = 4 tokens; under DCP2 pools interleave
    # by p % 2, so one local pool row stands for dcp*4 global tokens.
    tok_per_row = KPOOL * dcp
    P(f"== {name}: K={K}, DCP{dcp}, block {blk} local tokens ({global_block} global)")
    P(f"  attention block-table width W0 = {w_blocks}; index-K pages/request = {width_pages}"
      f" ({width_rows} local pool rows); mamba block = {e['mamba_block']} tokens")
    supertile_pages = SUPERTILE // PAGE
    n_chunks = cdiv(width_pages, supertile_pages)
    min_slice = cdiv(width_rows, MAX_SLICES)
    slice_rows = SLICE if min_slice <= SLICE else cdiv(min_slice, PAGE) * PAGE
    total_slices = 0
    for c in range(n_chunks):
        pages = min(supertile_pages, width_pages - c * supertile_pages)
        total_slices += max(1, cdiv(pages * PAGE, slice_rows))
    P(f"  tiled route: {n_chunks} supertile chunks of {SUPERTILE} rows, two-level fold"
      f" {'ON' if width_rows >= 2 * SLICE else 'off'}, slice {slice_rows} rows, {total_slices} slices")
    P(f"    slice k live from L > {slice_rows * tok_per_row:,} * k tokens"
      f" (2nd slice at {slice_rows * tok_per_row:,}; 3rd at {2 * slice_rows * tok_per_row:,})")
    P(f"    supertile chunk c live from L > {SUPERTILE * tok_per_row:,} * c tokens"
      f" (2nd chunk at {SUPERTILE * tok_per_row:,}; 3rd at {2 * SUPERTILE * tok_per_row:,})")
    P("  fused route (padded rows <= 16): per-CTA local trims")
    for req in range(1, 9):
        rows = req * (K + 1)
        prow = padded(rows, e["cap_max"])
        if prow > FUSED_MAX_ROWS:
            P(f"    {req} req: rows {rows} -> padded {prow}: TILED")
            continue
        ctas = max(1, min(width_pages, SMS // prow))
        # local trim at the end once a CTA's slice holds > topk rows (>= 9 pages)
        final_trim_rows = 8 * PAGE * ctas
        # mid-loop trims once carry_count + 64 > carry_cap (>= 15 pages)
        mid_trim_rows = (CARRY_CAP // PAGE) * PAGE * ctas
        merge = "cooperative (always)" if dcp == 1 else None
        if dcp == 2:
            crossover = max(4096, 22000 + 117 * ctas)       # v070 fused_indexer.py:612
            cap = ctas * TOPK_POOLS
            merge = ("cooperative above %d rows" % crossover) if cap > crossover else \
                "serial last-CTA (cap %d <= crossover %d: never cooperative)" % (cap, crossover)
        P(f"    {req} req: rows {rows} -> padded {prow}: ctas/group {ctas}; "
          f"local trim from L > {final_trim_rows * tok_per_row:,}; "
          f"repeated trims from L > {mid_trim_rows * tok_per_row:,}; merge {merge}")
    # prefill (all engines use the vLLM fp8_mqa_logits + top_k_per_row_prefill path)
    max_elems = LOGITS_MB * 1024 * 1024 // 4
    chunk_q = 2048 - K
    n_lim = max_elems // chunk_q
    P(f"  prefill: solo chunk {chunk_q} tokens (2048 - K draft slots); unaligned starts"
      f" (chunk % 4 = {chunk_q % 4}); logits budget sub-chunks the query dim once"
      f" pools > {n_lim:,} -> L > {n_lim * KPOOL:,} tokens")
    P(f"  KDA align migrations per 327,680-token reply: {327_680 // e['mamba_block']}")
    P("")

open(__file__.replace("thresholds.py", "thresholds.out"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
