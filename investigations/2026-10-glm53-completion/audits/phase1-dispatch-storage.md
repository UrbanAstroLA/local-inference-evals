# Phase 1, area 3: kernel dispatch and storage — audit report (agent hand-back, saved by coordinator 2026-10-11)

No GPU/server/container; torch not on host, so no CPU tests. Working notes: $S/work/ (banners-*.txt, b12x-diff-notes.md,
upstream/ read-only copies of vLLM sampler.cu, test_top_k_per_row.py, glm5next sparse_indexer.py, kpool_compress.py via gh api).
L7 = <gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+ps.server.log (0.7.0); L9 = ...+v0.9.1+ps.server.log (0.9.1);
LB = step6-layout/v091dcp2-r1-server.log (bundle); LF = step5-kpoolfix/v070fix-r1-server.log.

## Bottom line
- No B12x fe054789 → 7fcc094e change alters attention/indexer/MoE results on the deployed paths beyond summation/rounding order.
- One new untested mechanism in 0.8.0/0.9.1: default-on TP2 "plain remote push" all-reduce transport carries every
  captured TP all-reduce at GPQA batch sizes; its correctness protocol has no runnable test against the shipped API.
- Most differences are configuration (DCP2 vs ownership, 656 vs 528 record + writer, query projection, spec depth,
  indexer route mix), bundled — a 0.7.0 vs 0.9.1 comparison does not isolate the kernel library.
- Defect in both versions: 512 pools selected, column 511 dropped for the tail; no top-k path sorts; the dropped pool is
  not guaranteed lowest; the supplying path differs by version and concurrency.
- 0.7.0-only defects fixed in 0.9.x: kpool tail-seed stride (2 KB into another block's indexer region); 4-slot tail ring
  with K5. Cannot explain 0.9.1 worse; confounds any 0.7.0 baseline.
- Teacher-forced NLL (engine repo, K4 not K3.25): 0.7.1 0.8109 → 0.8.0/0.9.0/0.9.1 0.8181/0.8177/0.8180 on the 17,500-token
  long-backref doc (+~0.007; newer three agree within 0.0004; one 0.7.1 pass). git show v0.9.1:benchmarks/v0.8.0/nll-frozen/
  {v071-published,default}.json, benchmarks/v0.9.0/nll-frozen.json, benchmarks/v0.9.1/nll-frozen.json. Prefill-path signal.

## Q1 What runs (runtime-selected)
- MLA backend B12X_MLA_SPARSE all (L7:63). Record 656 (L7:65) / 528 (L9:71) / bundle 656 (LB:72).
- Partition: 0.7.0 TP2+DCP2 (32 local heads → 64 gathered, half KV, LSE combine); 0.9.1 owner split:25, mla_owned rank0
  [3..23], rank1 [27..43] (L9:211); bundle DCP2 owners=tp.
- Sparse-MLA decode: 0.7.0 old run_decode GLM_NSA 64 heads 96 rows (16×6) forced 32 splits (v070 b12x_mla_sparse.py:1569,
  2969); 0.9.1 decode.h64.rows64.w2048.rec528 splits 32/32 (L9:74); bundle rec656 (LB:75). Decode-vs-extend rule same in
  all (decode if max_query_len ≤ 1 or spec verify q ≤ 8 within capacity; mixed batches send decode rows through extend;
  v070:2935, v091:3170).
- Extend output: 0.7.0 head-major consistent (old prefill_mg.py:2084-2086, 3630-3701); 0.8.0+ token-major (43c0014;
  v091:1864).
- Query absorption: 0.7.0 B12x H64 fused when 2 ≤ rows ≤ 16 (v070 mla_query_projection/api.py:109) else torch.bmm, 576
  with zero RoPE lanes; 0.9.1 torch.bmm 512 (v091 mla_attention.py:925-930); bundle H64 as 0.7.0.
- KV writer: 0.7.0 stock compiled concat_and_cache_mla (v070:1840-1861); 0.9.1 B12x concat_and_cache_glm_next_mla
  (v091:2085-2118); bundle stock.
- Indexer decode: B12x paged index_topk_fp8 (L7:62); 0.7.0 per rank then DCP merge (all-gather 2×512 → run_row_topk;
  sparse_attn_indexer.py:1139-1218); 0.9.1 single rank. Prefill: DeepGEMM fp8_mqa_logits + vLLM top_k_per_row_prefill;
  0.7.0 + owner exchange ≥1024 rows (L7:406) or CuTe StableTopK merge. requires_padding fallback (kpool.py:933-974) rare.
- kpool compression: Triton _kpool_decode_update_batched_kernel / _kpool_tail_seed_kernel; 0.7.0/0.8.0 ring 4 + dense seed
  (bugs); 0.9.1 ring 8 + strided seed (kpool_compress.py:489, 594).
- MoE: 0.7.0 EP2 144 local, replicated input, all-reduce (L7:71, 238-239), B12x MoE policy-fallback fired (L7:232); 0.9.1
  TP2 rank-sliced 288/rank (L9:236); bundle EP2 (LB:80).
- TP all-reduce: 0.7.0 B12X_PCIE_ONESHOT old pull protocol then PYNCCL (L7:50); 0.8.0/0.9.1 B12X_PCIE_ONESHOT with new TP2
  remote-push transport default on (b12x new pcie_oneshot.py:137-148, 214-227; old default off v070 pcie_oneshot.py:120),
  PYNCCL above 384 KB (48 rows × 4096 bf16).
- mHC fused M=1 (L7:255, L9:254); 0.9.1 graph first seen during capture keeps TileLang (v091 mhc.py:~64).
- Draft: FLASH_ATTN v2; 0.7.0 own pages; 0.9.1 slot-shared in MLA pages, block 1088 (L9:240).
- Route mix: rows = requests × (1+K), graph-padded; fused ≤16 rows else tiled (fused_indexer.py v091:921-927,
  v070:550-556); no B12X_FUSED_INDEXER set. 0.9.1 K3: 1-4 requests fused (cooperative any length, v091:964-982); 5-8
  tiled (1024 buffer). 0.7.0 K5: 1-2 fused (serial below ~22k pools); 3-8 tiled (8192 buffer); final order from DCP
  run_row_topk over 1024. Time share (status lines): 0.9.1 1-4 running 318/1,990 (16%), 8 running 1,616; 0.7.0 1-2
  running 169/2,748 (6%), 8 running 2,391. Long unanswered replies spend their final stretch with few requests left →
  0.9.1 fused-cooperative vs 0.7.0 tiled-then-merge (inferred).

## Q2 KV record
656 B: [0,512) e4m3 latent 4×128; [512,528) 4×fp32 scales; [528,656) 64×bf16 RoPE always zero (v070:1856-1861; rope_pad=64
v070:1465); query padded 576 with zero lanes → extra QK term exactly 0. 528 B: same latent+scales, no tail
(mla_cache_format.py v091:16-42); B12x writer scale = amax/448 (1.0 all-zero), × rcp.approx.ftz, saturating rn e4m3, skips
slots < 0 (b12x new kv_cache.py:444-586). qk_rope_head_dim=0 → tail removal drops only zeros. Writer equivalence
unverified (0.7.0 writer compiled C++; inferred ~1 ulp); engine scripts/test-glm-next-records.py exists, no result artifact.
Readers/layout agree: vLLM block 4352 (0.9.1) / 3584 (0.7.0) = 68/56 kernel blocks of 64; MLA page = unified page:
4352×528 = 2,297,856 (L9:262), 3584×656 = 2,351,104; writer slot//64, slot%64 via strides; reader requires contiguous
(blocks,64,rec) (v091:3088-3112), block = slot // page; single GLM53_NOPE_RECORD import. Draft slot sharing: disjoint block
IDs, page 2,228,224 of 2,297,856 (v091 kv_cache_utils.py:1419-1470, L9:240), one BlockPool, DCP=1 required. Kpool tail ring
co-owns the indexer tensor via padded stride; 0.7.0/0.8.0 dense seed base=(blk*2*KPOOL+t%KPOOL)*HEAD_DIM (v070
kpool_compress.py:481) → own tail unseeded, 2 KB into another block (vllm#57477); 0.9.1 strided (:489). Selection layout
as area 1; upstream vLLM keeps 512 pools, 2051-wide (work/upstream/glm5next_sparse_indexer.py:392, 646) → drop is
fork-specific. -1 indices within selected length: loader clamps to slot 0, score -1e30, slot-0 V still in P·V (NaN would
propagate; b12x io.py old 189-200; new zeroes masked V only for DSV4, kernel.py:1565-1588); same both; catastrophic not subtle.

## Q3 Graph vs eager
Capture 0.7.0 [1,2,4,8,16,24,…,96] (L7:41); 0.9.1/bundle […,64] (L9:47, LB:47); both cover uniform verify batches ≤16
requests; larger (incl. 2048-token prefill chunks) eager. PIECEWISE: eager-break ops (unified_mla_attention_with_output,
sparse_attn_indexer_kpool, KDA _forward, draft attention) run eagerly at replay; FULL (uniform decode/verify,
cudagraph_dispatcher.py:143): wrapper calls op inside capture (breakable_cudagraph.py:102) → MLA, indexer, KDA baked in.
Frozen per graph: use_decode_kernel, fused vs tiled, ctas_per_group, use_uniform/lmax, indexer plan max_q_rows, B12x plan
objects and scratch addresses, split count 32, A2A/DCP collectives (0.7.0) — all depend only on padded rows; device lengths
re-read per replay; no stale pointer/length path found. Unverified (all versions): is_spec_decode computed on host from
is_prefilling at build (v091:1317-1323) — if the dummy capture batch lacks is_prefilling, FULL verify graphs would bake
extend instead of decode. Capture-sync patch (0.8.0+) synchronize() during capture only (breakable_cudagraph.py:207, 378).

## Q4 Scratch
Workspace (v1/worker/workspace.py:92-190, identical): one torch.empty uint8 per micro-batch, never zeroed, views from
offset 0, locked after warmup; sequential lifetimes (inferred safe); MLA O in scratch read by next torch.bmm before reuse.
Split-K: empty split writes LSE -inf, skips mid_out; old merge skips -inf; new merge (f5394625) zero-weights -inf/NaN,
normalises before summing; no stale reads; not bit-identical. Indexer: topk_indices_buffer reset -1 per call (kpool.py:461);
pool_topk torch.full(-1) per call; fused merge state zeroed per call (old 2613 / new 3062); DCP1 page_table_1 row fully
rewritten (v091:773-814); nsa lengths recomputed. Plans shared across layers with equal Caps; per-rank buffers; draft runs
after target; rows independent.

## Q5 B12x fe054789 → 7fcc094e
Neutral: prepared-plan API (Caps fix scale, LSE convention, traits; GLM index rows = planned width; no torch-reference
fallback in prepared execution). Rounding/order only: split-K merge rewrite (f5394625, ignores NaN LSE), W·V hi/lo
interleave + K-axis permutation, vectorised Q staging, packed record copies, BF16 GEMV for m ≤ 8. GLM_NEXT vs GLM_NSA
equivalent apart from zero RoPE lanes (inferred). Top-k: candidate buffer at 512 cut 8192 → 1024 (02a19e65) — same set
except tied members kept, branch frequency and order change; fused cooperative merge rewritten (357576e6) and default at
any length, early stop can place any pivot-bin member in column 511. No path sorts (atomic emission: higher bins first,
ties last; vLLM CuTe StableTopK same, dcp_indexer_cutedsl.py:162-400); upstream top_k_per_row (work/upstream/sampler.cu:
271-567) sorts final bin descending, placed last; fork's compiled version unverified; no test asserts order. Comm: new TP2
remote-push (f46fee91 et al.), default on, world 2, bf16, last dim 4096, arithmetic identical (one bf16 add of 2); tag/
generation protocol relies on 8-byte store atomicity across PCIe; its GPU tests call removed APIs; evidence JSON only
checks output == 3.0. DCP A2A, owner exchange plumbing only. MoE projection-mixed Trellis same; K4 codeword rewrite
bitwise-equal by hand; routing unchanged; swiglu_limit never passed (side note: swiglu-clamp A/B probably tested a path
K3.25 never takes; inferred). vLLM 0.8.0 → 0.9.1 only: seed stride (vllm#57477), spec ring (vllm#58454), DCP1 tail
compaction (f51ace6), boundary-lookup hooks (off). Indexer files byte-identical.

## Q6 Partitioning and bundles
EP2 (144/rank, replicated input) or TP (slices of 288) independent of attention layout; both end in TP all-reduce;
observed EP2 with owner split (EN) and owners=tp (EO). NOPE_RECORD=1 ⇒ DCP=1, fp8_ds_mla, H64 projection off
(v091:1745-1759). MLA_OWNERS ⇒ TP=2, DCP=1, PP=1 (placement.py:27, 77, 79). DRAFT_SLOT_SHARING=1 ⇒ DCP=1 (else warns,
ignored, kv_cache_utils.py:1433-1435), draft layers ≤ smallest owned count (5 ≤ 5). Owner exchange only DCP=2 prefill
≥1024 rows (sparse_attn_indexer.py:1354). Spec ring ⇒ block_size % ring == 0 (attention.py:194-201). Capture max =
MAX_NUM_SEQS × (K+1). Observed layouts: DCP2+EP2+656 (L7, LB); owner+528 (L9); owner+656 (EN); owners=tp+DCP1+528/656
(h32 plans; EO, NO in component-screen-2026-10-09).

## Matrix (FP = legitimate fp/perf; DEF = potential/actual defect)
1 Attention partition — FP (rounding; screened EO/NO/EN).
2 KV record + writer 656 vLLM / 528 B12x — FP unverified; test: test-glm-next-records --heads 64; NLL NOPE_RECORD 0 vs 1.
3 Query absorption H64 vs bmm — FP; no parity test found.
4 Decode kernel internals — FP (merge tests atol 1.5e-2; decode regimes cos > 0.999).
5 Extend output layout — OK (consistent in each version).
6 Column-511 pool drop — DEF both versions; supplying path 0.7.0 run_row_topk; 0.9.1 fused-coop ≤4 requests, tiled 5-8;
  no ordering test; mechanism: sometimes drops a well-ranked history pool at every MLA layer at long context; test: GPU
  microtest on recorded indexer logits, rank of dropped pool by path.
7 Indexer route/candidate mix — FP (tie membership); test B12X_FUSED_INDEXER=0.
8 DCP2 indexer/owner exchange — FP (exact merge inferred).
9 Tail seed stride — DEF 0.7.0/0.8.0 (upstream test_kpool_decode_update_batched.py 4 fail before fix).
10 Spec ring — DEF 0.7.0/0.8.0.
11 DCP1 tail loss — DEF 0.8.0; fixed (PROVENANCE: KL 0.066 → 0.010).
12 Draft slot sharing — OK (disjoint block IDs; no aliasing found).
13 Graph capture — OK (none found).
14 TP2 all-reduce transport — DEF-candidate 0.8.0/0.9.1: no runnable test; stale/torn peer read in captured all-reduce
  would silently perturb a layer's hidden state; test: stress (below); serving A/B B12X_PCIE_TP2_PLAIN_REMOTE_PUSH=0.
15 MoE EP2 vs TP — FP (rounding; screened).
16 Prefill NLL drift (K4) — open; NLL ×2 per arm.

## Fixtures
1 Captured TP2 all-reduce stress, 0.9.1 image, both GPUs, ~0.1-0.2 GPU-h: replay captured one-shot all-reduce (bf16,
  4096, 1-48 rows) thousands of times with varying inputs, bitwise vs PyNCCL. Then serving A/B with remote push off (~2
  GPU-h). Actionable: any mismatch = defect in a 0.8.0+ mechanism touching every layer; clean retires it.
2 Rank of dropped column per route (~0.2 GPU-h): fixed/recorded pool logits (32 heads, topk 512, 4k-60k pools) through
  0.7.0 tiled + run_row_topk and 0.9.1 fused-coop and tiled; record score rank of column 511. Then serving A/B with
  deterministic drop or B12X_FUSED_INDEXER=0 on q109/q170 (~2 GPU-h). Actionable: dropped pool often well inside top-512
  (e.g. rank < 450) → real in both, version-dependent rate, especially ≤4 concurrent in 0.9.1.
3 Record/writer/projection via teacher-forced NLL (~1 GPU-h): test-glm-next-records --heads 64; frozen-text NLL
  (long-backref) ×2 on 0.9.1 NOPE_RECORD=1 vs 0 and on 0.7.0. Actionable: reproducible gap > ±0.002 tracking NOPE_RECORD.
