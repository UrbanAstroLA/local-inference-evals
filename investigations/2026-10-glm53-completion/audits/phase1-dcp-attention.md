# Phase 1, area 2: DCP attention execution and DCP1 counterpart — audit report (agent hand-back, saved by coordinator 2026-10-11)

CPU only; nothing outside scratch modified. Kernel-level notes: $S/notes-b12x.md. Paths relative to $S/v070|v080|v091/.

## Short answer
No smoking gun. Attention math is the same function in all versions (0.7.0 DCP2 + 656 B GLM_NSA; 0.9.1 DCP1 + ownership
+ 528 B GLM_NEXT) up to fp reordering (bf16 partial rounding, split-merge order, PV interleave, query-projection GEMM).
B12x byte addressing is Int64 in both versions. Differences in WHAT is attended exist but cut the wrong way or are small:
(1) 0.7.0's indexer is more corrupted (tail-ring wrap with 5 drafts, tail-seed stride), both fixed by 0.9.0/0.9.1;
(2) DCP1 kpool-tail drop (0.8.0/0.9.0, fixed 0.9.1; 0.7.0 DCP2 never had it); (3) which one of 512 pools is discarded
by the [:511] slice differs (4 of ~2,047 tokens per row at contexts ≥ 2,048).
Only 0.9.1-unique structural risk: the shared, >2 GiB MLA slot tensor (2.93 GB; MLA + KDA state + a DFlash draft layer),
B12x reads it with Int64 offsets, other writers not audited, existing >2^31 tests skip when VRAM is short. 0.7.0 and the
DCP2 bundle stay under 2 GiB per tensor.

## Partitioning
- DCP2 (0.7.0; bundle): token t on rank t%2, local index t//2 (v1/attention/backends/utils.py:964-1000, all versions);
  indexer refuses interleave >1 (mla/indexer.py:700-707); logs "compress_ratio=4, world_size=2, interleave=1". Pool p
  (tokens 4p..4p+3) on rank p%2 (indexer.py:1289-1305); MLA tokens of a pool split 2+2; both ranks compute indexer K/gate
  per token into their own tail ring; only owner writes the compressed pool (pos_valid gating). Queries: TP shard 32
  heads; B12x A2A all_gather_heads → 64 heads, geometry pinned (64,512,512) (b12x_dcp_a2a.py:188), padded to 576 with 64
  zero RoPE lanes. Each rank: all 64 heads over its half → (O bf16, LSE fp32 natural log); combine reduce-scatters to 32.
- DCP1 + MLA ownership (0.9.1 split:25): whole MLA layer incl. indexer, MLA KV, indexer/tail caches exists only on owner
  rank, all 64 heads unsharded (models/glm5next/nvidia/placement.py:120-136; attention.py v091 442-522 owner_local,
  disable_tp, reduce_results=False). Non-owner returns zeros_like (placement.py:105-106); TP all-reduce completes the layer
  (109-117); x+0 exact. Log: rank 1 owns 27, 31, 35, 39, 43. Ownership refuses DCP≠1 (placement.py:76-77).

## Difference matrix
D1 Cross-rank combine. 0.7.0: B12x PCIe lse_reduce_scatter; fallback vLLM cp_lse_ag_out_rs when can_combine fails
  (partials not bf16, batch > captured 176); mask_dcp_empty_shards_ sets LSE -inf only for global seq_len 0 and padding.
  0.8.0/0.9.1: none (ownership). Bundle: DCP2, same kernel. Source: v1/attention/ops/dcp_utils.py:650-713 (mask 706);
  ops/common.py:9-35, 38-122; B12x comm/pcie/_dcp_a2a_cute.py:685-722 identical v070/v091 (non-finite LSE → -inf; max
  non-finite → 0; w=exp(lse-max); inv=1/max(sum,1e-10); fp32 accumulation of bf16 partials); reference
  pcie_dcp_a2a.py lse_reduce_scatter_reference (v070:239, v091:216); vLLM b12x_dcp_a2a.py v070→v091 prepared plans + bf16
  only (141). FP only (bf16 partial rounding; deterministic per head). Invariants: zero-length rank → -inf, weight 0
  (interleave 1 → only seq_len=1); all -1 indices → finite LSE ≈ -6e28, O = mean V at slot 0, weight underflows to 0 when
  the other rank has a real token (both B12x versions, notes §1b). No NaN path. Tests: B12x tests/comm/test_pcie_dcp_a2a.py
  (+_gpu, v070); tests/attention/test_attention_mla_merge.py:125-219 (-inf, empty splits). No end-to-end vLLM combine test.
  Mechanism: none found (a mis-weight would bias 0.7.0). Test: 1 GPU, two rank-local halves (t%2) + reference combine vs
  full-cache run vs FP32.
D2 Record/query path, 656 B GLM_NSA vs 528 B GLM_NEXT. 0.7.0/bundle: 512 E4M3 + 4×fp32 scales (bytes 512-527) + 128 B
  bf16 RoPE tail held at 0; writer vLLM concat_and_cache_mla (C++, not in trees); query 576-wide (H64 projection).
  0.8.0/0.9.1: first 528 bytes only; writer B12x concat_and_cache_glm_next_mla (scale amax/448, v*rcp.approx(scale));
  query 512-wide, H64 disabled → stock bmm. Source: model_executor/layers/mla_cache_format.py v091 16-41;
  b12x_mla_sparse.py v091 1740-1759 (DCP>1 raises 1752-1758), 2050-2063 (writer), 2855-2860 (512 query);
  mla_attention.py v091:929, 1085; B12x notes §1b (io.py, kernel.py:455-479, kv_cache.py:487-586). FP only (zero RoPE
  tail; writer rounding ties, inferred; GEMM path). Tests: B12x test_glm_next_mla.py (v091 :1004, :1306, :1524) vs record
  oracle; no GLM_NEXT↔GLM_NSA equivalence test; engine scripts/test-glm-next-records.py (4,096-token cache, 8 rows,
  cos ≥ 0.9995, max abs ≤ 3e-2; no large caches, incremental writes or query projection change). Test: F2.
D3 B12x fe054789 → 7fcc094e sparse-MLA decode/extend. 0.7.0 per-call run_decode/run_extend forced_num_splits;
  0.8.0/0.9.1 plan/prepare/bind/run, splits pinned 32 (log "rows<=64 splits=32/32"); f5394625 two-pass split merge,
  interleaved HIGH/LOW PV, permuted K axis; Q inverse scale from UE8M0 exponent (exact). Masking unchanged
  (selected_lengths only; cache_seqlens unused by decode). New extend kernels ignore head-major output (strides baked at
  prime: sparse_mla/_preparation.py:191-245, prefill_mg.py:2146-2148, 4108) — real kernel bug, worked around by forcing
  token-major extend (engine 43c0014; b12x_mla_sparse v091 1862-1866); decode head-major honoured. FP only for decode.
  Tests: test_attention_mla_merge.py (+:325, :380 v091); test_attention_mla_sm120.py:976; test_sparse_mla_decode_regimes.py
  :86, :120 (v091, GLM_NSA, whole-tensor cos); no head-major sparse test; engine scripts/test-prompt-nll.py (prefill only).
D4 Discarded pool from top-512 (all versions [:511], sparse_attn_indexer_kpool.py:644/:1009). 0.7.0 fused: serial arm
  below ~22,000 + 117·ctas pools (~176k tokens one row) keeps arrival order in lowest refined bucket; above, cooperative
  exact pivot. 0.9.1: always cooperative, rewritten (02a19e65) to 12/12/8-bit rounds with early exit → col 511 arbitrary
  member of lowest 12-bit bucket (~1/8 octave). DCP2: run_row_topk merge over [r0 512 | r1 512]
  (sparse_attn_indexer.py:1139-1215). Prefill top_k_per_row_prefill (compiled, ordering not inspectable); >16 rows tiled.
  Source: B12x v091 dsa_indexer/fused_indexer.py:2577-2725; v070 593-621, 2123-2262; scratch.py v091 1010-1031 (notes §3).
  Implementation difference, not violation; intended "top-511 by score" rule not stated in source. Tests: set-only
  (test_fused_indexer.py:306-350); engine test-dcp1-kpool-tail.py synthetic ascending. Test: indexer fixture on real
  index-K/queries vs exact fp32 top-511.
D5 DCP1 kpool-tail drop: 0.7.0 DCP1 branch has it (v070:2796-2808) but shipped DCP2 compacts; 0.8.0/0.9.0 DCP1 drop newest
  1-3 tokens when context < ~2,044 and len%4≠0; 0.9.1 fixed (_compact_dropped_selection; v091 724-770, 3036-3047; engine
  patches/port-dcp1-kpool-tail-glm53.py, f51ace6/v0.9.1; tail cols 2044-2046 kpool_compress v091 955-961). Invariant,
  fixed in 0.9.1; cannot explain 0.9.1 worse. Tests: engine scripts/test-dcp1-kpool-tail.py (lengths 1-2050).
D6 Kpool tail-ring under speculation. 0.7.0 (K=5, next_n=6) and 0.8.0 ring = kpool 4 slots: verify tokens wrap; a
  rejected pool-completing draft's redo reads slots overwritten by later drafts → permanently wrong compressed index-K for
  that pool (inferred frequent with K=5); MLA values unaffected. 0.9.1 (K=3) ring 8, no wrap. Source: v070
  models/glm5next/nvidia/attention.py:190-198; v070 kpool_compress 579, 587, 615, 633; v091 attention.py:188-208; engine
  patches/port-kpool-spec-ring-glm53.py (vllm#58454). Invariant violation in 0.7.0, fixed 0.9.1. Tests: upstream
  tests/kernels/test_kpool_decode_update_batched.py (4 fail on 0.7.0/0.8.0, pass with fix).
D7 Kpool tail-seed addressing (prefill). 0.7.0/0.8.0 dense [blocks,2,4,128] addressing (kpool_compress v070:470, 481) but
  tail aliases indexer page stride → request's own tail unseeded (first decode-built pool uses stale ring) and ~2 KB raw
  K/gate written into another block's index-K. 0.9.1 strided, fixed (v091 449-521; patches/port-kpool-seed-stride-glm53.py,
  vllm#57477). Invariant violation in 0.7.0, fixed. Same test file as D6.
D8 Draft slot sharing (0.9.1 shipped only): 5 DFlash sliding-window layers co-own MLA slot tensors, draft block_size 1088,
  pages padded to MLA page 2,297,856 B; block-ID exclusivity via shared BlockPool. 0.7.0/bundle separate tensors (DCP2
  disables sharing). Source: v091 v1/core/kv_cache_utils.py:1419-1481, 1752-1793 (shared_by 1770); engine
  patches/port-glm53-draft-slots.py; log "5 draft layers in MLA slots, block_size=[1088]". No test that MLA records
  survive draft writes into a shared tensor. Mechanism (inferred): a draft writer/reader ignoring the padded page stride
  would land draft K/V in other blocks' MLA records — long-context corruption after blocks recycle. Test: F1.
D9 Tensor size vs 2^31 bytes. 0.9.1 shipped: 1,275 blocks × 2,297,856 B = 2.93 GB per slot tensor (log tensor_bytes
  18,677,260,800 rank 0); slots to 5.55M; byte offsets > 2^31 for slot ≥ 4,067,203 (block ≥ 935). 0.7.0: block 3,584 local
  tokens (log), 656 B/token, 2.76M pool, ~0.9 GB/rank (inferred). Bundle: 650 × 2,351,104 B = 1.53 GB. B12x Int64 in both
  (notes §4: v091 io.py:590-599, kv_cache.py:197-210; _cache_block_stride_bytes uses real stride(0)); CuTe pointer+Int64
  lowering to 64-bit inferred (DSL not installed). Not audited: DFlash FlashAttention paging, KDA state kernels on the same
  tensors. Tests: B12x v091 test_glm_next_mla.py:1646, test_attention_mla_sm120.py:1742 (#312, page ids > 2^31) — both
  skip when free VRAM is short. Mechanism: wrong records only in high blocks, reached once the LRU pool cycles in long
  GPQA runs (inferred) → sporadic, grows with run length. Test: F1; serving discriminator --num-gpu-blocks-override ≤ 930
  on 0.9.1 with everything else fixed.
D10 Per-token causal bound. DCP1 exact GPU positions + 1; DCP2 CPU seq_lens_cpu_upper_bound localized (optimistic under
  async spec decode; async scheduling not visible in logs); applied as min(valid_count, bound), passed as cache_lengths
  which the kernel ignores (v091 b12x_mla_sparse 1438-1480, 3007-3011; v1/attention/backend.py:488-492). Overestimate
  harmless (inferred); underestimate would drop arbitrary tokens on DCP2 (compaction order unspecified, sparse_utils.py:307)
  — 0.7.0 side. Test: assert bound ≥ valid_count in a debug run.
D11 DCP2 summation order: atomic compaction → nondeterministic order (sparse_utils.py:284-307); DCP1 deterministic. FP only.

Hold equal in fixtures: snapshot 701cd745 (0.7.0) vs 0490d2f7 (0.9.1) (weights identical blobs; template differs);
explicit chat template in 0.9.1; K5 vs K3 (next_n, plan rows 96 vs 64, tail-ring size, indexer route).

## Bundles
EP touches no attention code. 528 B NOPE requires DCP=1 (b12x_mla_sparse v091:1752). Ownership requires DCP=1, PP=1
(placement.py:76-79), each rank ≥1 MLA layer, mamba group count fits smallest owned count (kv_cache_utils.py:1385-1394).
Draft slot sharing requires DCP=1 and ≥5 MLA slots/rank else silently separate (kv_cache_utils.py:1428-1441); changes pool
size and block size (4,352 vs 3,584). DCP2 requires interleave 1; B12x A2A needs (64,512,512) + bf16 partials else ag_rs
fallback. Owner top-k exchange only with DCP2 and prefill chunks ≥1,024 rows at top-512 (sparse_attn_indexer.py:1310-1328).
Kpool ring derived from num_speculative_tokens (0.9.1), must divide block_size. Record size changes block geometry/pool
size and so the >2^31 threshold.

## Oracles
Neither prefill nor 0.7.0 is an oracle (different top-k and extend kernels; 0.7.0 carries D6/D7). Use FP32 reference from
dequantized records and exact fp32 indexer scores. A newly built cache tests reads/selection only; incremental writes
(tail ring, rollback, shared-tensor writers) need replayed decode steps.

## Ranked fixtures
F1 Shared-tensor isolation + high-offset addressing, 0.9.1 geometry (1 GPU, ~3-6 GB, minutes): build 1,275 blocks, MLA
  page 2,297,856 B, draft block 1088, shared_by MLA + KDA + draft; write known 528 B records into MLA blocks below and
  above 935 (GLM_NEXT writer); run DFlash draft KV writer + FlashAttention read and a KDA state update on other block IDs;
  decode vs FP32 selecting slots > 4,067,203. Actionable: any byte change outside own block or decode mismatch at high
  slots = 0.9.1-unique invariant violation; clean removes D8/D9.
F2 Layer-level attention equivalence at 4k/64k/256k (1 GPU; 2 for real A2A): identical latents/queries and 2,047-column
  selections incl. tail cols and -1 padding; compare 0.9.1 DCP1 GLM_NEXT stock projection, GLM_NSA 656 B H64, DCP2 halves +
  combine, 0.7.0 B12x kernel, each vs FP32 (sets, rank-local LSE/O, combined O). Actionable: beyond fp envelope (cos <
  0.999 at long N, split-dependent deviation).
F3 Indexer incremental-write and rollback replay (1 GPU): recorded stream with real DFlash accept/reject; K=5 on 0.7.0
  kernels (ring 4, dense seed) vs K=3 on 0.9.1; compare index-K pools and top-511 sets vs no-spec FP32 reference incl.
  discarded pool. Actionable: corruption/selection-change rate per 1k tokens per version.

## Uncertainties
Async scheduling on/off; compiled top_k_per_row_prefill ordering and fp8_ds_mla C++ writer rounding; CuTe pointer+Int64
lowering; DFlash FlashAttention and KDA addressing of >2 GiB shared tensors (not audited).
