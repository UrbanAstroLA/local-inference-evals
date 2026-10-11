# Phase 1, area 1: sparse MLA indexer (top-k) selection — audit report (agent hand-back, saved by coordinator 2026-10-11)

CPU only; no GPU used. One pure-Python transcription: $S/cpu/slice_demo.py.
S=<scratch>; engine repo <engine-repo>.

**Bottom line:** no 0.9.1-specific selection defect that would drop decisive long-range context. Selection is exact top-512
by score in all versions (modulo ties); scoring math unchanged. Real issues here are shared with 0.7.0 and differ only in
which arbitrary choice is made, so this area can explain 4 vs 16 only through how often those choices bite.

## 0. Corrections / facts
- f51ace6 (DCP1 kpool-tail fix) is in shipped v0.9.1 (tag --contains → v0.9.1; tag 0e4a582 "preserve newest tokens in
  sparse attention"; image 84572a8 = merge of that PR; marker `dcp1-kpool-tail` ×3 in v091 b12x_mla_sparse.py, absent in v080).
- Model (snapshot 0490d2f7): index_kpool=4, index_topk=2048, always_select_tail=True, 32 indexer heads, 11 DSA layers each
  with own indexer → 512 pools selected in all versions.
- Byte-identical in all three images (shared base cstechdev/vllm:glm53-flash-nope…r1): sparse_attn_indexer_kpool.py
  (sha e7935f2a9b8e: pool selection, the 511 slice, the tail), indexer.py, b12x_dcp_topk.py, sparse_utils.py,
  dcp_indexer_cutedsl.py. kpool_compress.py identical v070/v080; v091 adds only spec-ring + seed-stride ports (expand
  kernel unchanged, +17-line shift).
- Version differences in this area come only from: B12x fe054789 vs 7fcc094e; vendored sparse_attn_indexer.py (owner
  exchange plan plumbing); topology DCP2 (0.7.0) vs DCP1+layer ownership (0.8.0/0.9.1).
- GPQA exercises decode selection only: prompts median 204, max 2795 tokens (both *+ps.requests.jsonl); prefill ≤2048
  takes the dense path sparse_attn_indexer_kpool.py:475-506 (no top-k).
- Decode always B12x path (sparse_attn_indexer_kpool.py:881-931). SM120 flattening on (indexer.py:736-738; both logs) →
  requires_padding=False.
- Route by rows = requests × next_n: ≤16 rows fused (fused_indexer.py v091:921-927, v070:550-556) = ≤2 requests in 0.7.0
  (next_n 6), ≤4 in 0.9.1 (next_n 4); >16 rows tiled. GPQA ran up to 8 concurrent → mostly tiled.

## 1. Pipeline (same structure in all versions)
1. Eligibility: pool seq_lens = floor(L/4) (indexer.py:1289-1295); DCP2 localizes per rank after per-token expansion
   (indexer.py:1298-1305, interleave 1). In-progress pool never scored.
2. Pool top-k: _run_b12x_paged_topk(topk=512) (sparse_attn_indexer.py:1054-1136) → index_topk_fp8 fused or tiled
   (paged.py v091:870-899 / 1172-1236). DCP2 adds _merge_b12x_dcp_topk (sparse_attn_indexer.py:1139-1218): local→global
   ids (b12x_dcp_topk.py:10-40), all-gather 2×512, run_row_topk over 1024, gather by position (b12x_dcp_topk.py:176-210).
3. 511 slice + tail: expand_pools_and_append_tail(pool_ids[:, :select_k-1], …) at sparse_attn_indexer_kpool.py:1008-1010
   (decode), :643-645 (prefill); kernel kpool_compress.py v091:925-964 / v070:908-947. Cols 0..2043 = first 511 pool
   columns keeping 0 <= pid < L//4 (v091:955); cols 2044..2046 tail tokens [floor(L/4)*4, L); col 2047 = -1. Width
   round_up(2048,128)=2048 (model.py v091:628-637).
4. MLA consumption (b12x_mla_sparse.py): DCP2 stable-compacts each rank's owned tokens to a prefix, "prefix order is
   unspecified" (v091:2981-3010 = v070:2751-2781); DCP1 positional mask at nsa_len=min(L,2048); 0.9.1 adds
   _compact_dropped_selection (v091:3012-3044), absent v070:2782-2810 and v080.

## 2. Answers
- Q1 eligibility/tail: same in all versions (complete pools only; 0–3 tail tokens at cols 2044..2046). 0.7.0 keeps the
  tail via DCP2 compaction. 0.8.0 (DCP1, no fix) erases the tail for L ≤ 2043 when L mod 4 ≠ 0 (known, fixed in 0.9.1;
  does not touch long replies).
- Q2 511 slice: select_k = 2048/4 = 512 (sparse_attn_indexer_kpool.py:574, :855); kernels pick 512 then [:, :511] keeps
  the first 511 COLUMNS (:644, :1009) so 511×4 + 3 tail = 2047 fits 2048. Positional, not by score; no top-k kernel emits
  score-sorted output, so the dropped pool is whatever lands in column 511. Upstream B12x 7fcc094e uses all 512 + tail,
  width 2051 (b12x/testing/startup/attention.py v091:306-317 "C4 pooled512"; sparse_mla/pooled_selection.py:111-127).
  Which matches training not checkable locally. CPU demo: L ≤ 2047 dense-equivalent; L = 2048..2051 drops one pool (if
  index order: tokens 2044–2047, incl. the current token at L=2048).
- Q3 local vs global: 0.7.0 decode per-rank exact top-512 then exact global top-512 of 1024 (exact: every global winner
  is in its owner's local top-512). 0.8.0/0.9.1 single top-512. Same set modulo ties.
- Q4 ties/order/duplicates: all decode selectors are radix selects writing via atomic counters → unsorted, nondeterministic
  order. tiled_topk/run_row_topk (0.7.0 merge, 0.9.1 tiled fold): col 511 normally lowest tier (ties fill
  s_out[topk-old_rem], v091:1165 / v070:1124) EXCEPT coarse quick exit: when exactly 512 candidates sit above a coarse fp16
  bin boundary all 512 are written in atomic order (v091:913-955, v070:872-914) → col 511 can be ANY pool incl. the best;
  under DCP2 this always happens at L=2048..2051. 0.9.1 fused cooperative merge (default every multi-CTA group, threshold
  0, v091:964-982; code 2520-2742): ≤512 total pools → CTA-arrival order (2577-2598); >512 → winners first, pivot ties at
  end in arrival order (2690-2722) → col 511 lowest tier. 0.7.0 fused uses serial merge (cooperative only above ~22k pool
  rows/rank, v070:593-620), and the DCP merge reorders anyway. CuteDSL merge (0.7.0 prefill <1024 rows): deterministic,
  lowest id wins ties (dcp_indexer_cutedsl.py:209-220), output order still atomic. No path emits duplicate valid ids;
  0.9.1 B12x pre-inits output before overflow fallback (tiled_topk v091:1194-1203, commit 02a19e65); no reachable gap
  found in 0.7.0 (inferred). MLA is a softmax over a set → the 511 slice is the only place order changes what is attended.
- Q5 owner exchange vs global merge (0.7.0 only): start.sh:447-449 exchange ON, min_rows 128 → effective max(128,1024)=1024
  (0.7.0 log). Prefill only: DCP=2, ≥1024 even rows → owner exchange (sparse_attn_indexer.py v070:1331-1410, run_row_topk);
  else _merge_dcp_topk_global (:802-852, CuteDSL stable). Decode uses neither. Equivalent sets for distinct scores; diverge
  on exact ties (arbitrary vs lowest id) and output order (changes which pool the slice drops).
- Q6 window guarantees: only the 0–3 tail tokens; no sliding window (sliding_window=ring only sizes the tail ring,
  attention.py v091:191-208). The newest complete pool must win on score; when L mod 4 = 0 it contains the query token.
  All versions: L ≤ 2047 dense-equivalent; L = 2048..2051 drops one arbitrary pool. 0.8.0 only: tail drop L ≤ 2043.

## 3a. Different but legitimately equivalent (set preserved modulo ties)
| # | Item | 0.7.0 | 0.8.0 | 0.9.1 | Location | Tests |
|---|---|---|---|---|---|---|
| E1 | Merge topology | per-rank top-512 → all-gather → run_row_topk/1024 | single top-512 | same | sparse_attn_indexer.py:1139-1218; _kpool.py:901-931 | engine scripts/test-b12x-dcp-score-gather.py (remap only); B12x test_row_topk_* (set vs torch.topk) |
| E2 | Fused merge arm (≤16 rows) | serial | cooperative any length | same | fused_indexer.py v070:593-620; v091:964-982, 2520-2742 | test_fused_indexer.py:320-350, 491-518, 791-879 random-input sets; :348 "no rank-k ties here"; no tie/zero-score test |
| E3 | Candidate buffer top-k 512 | 8192 | 1024 (exact fallback far more often) + output pre-init | same | tiled_topk.py v070:55; v091:52-68, 1039, 1194-1203 | test_paged_prefill_topk_long_context.py (all-equal, carry-fold overflow, dups 209-216); test_attention_dsa_indexer_api.py:1554 |
| E4 | Route mix | fused ≤2 req | fused ≤4 req | same | fused_indexer.py:921-927; scratch.py v091:875-956 | test_fused_indexer.py:163-255 |
| E5 | Scoring math | same | same | same | kernel.py diff plumbing only | — |
| E6 | Prefill merge | owner exchange or CuteDSL | — | — | Q5 | — |
| E7 | MLA consumption | DCP2 compaction | DCP1 mask (D4) | DCP1 + compaction fix | b12x_mla_sparse.py v070:2751-2810; v091:2981-3044 | scripts/test-dcp1-kpool-tail.py; protocol kpool-tail-index/v1 |

## 3b. Potential defects
- D1 positional 511 slice of an unsorted 512 top-k (all versions). Col 511 may hold a high-ranked pool; at long lengths
  needs a coarse quick exit (possible on every 0.7.0 decode merge and on 0.9.1 batches >16 rows); at exactly 512 pools
  always drops an arbitrary pool. Could silently remove the strongest 4-token span for that layer/step; frequency unknown.
  No test uses real kernel order (test-dcp1-kpool-tail.py feeds pre-sliced arange(511)). Test: F2.
- D2 tie resolution at the boundary: nondeterministic everywhere, each draws differently; score = Σ w·ReLU(q·k) so
  exact-zero plateaus plausible (inferred); attention.py comment (v091 ~305-313) warns near-tie flips hurt hard
  long-context tasks. Test: tie census in F1.
- D3 only 0–3 tail tokens guaranteed (shared design; newest pool and sometimes the query token are score-dependent). Not a
  differentiator; measured by F1.
- D4 DCP1 tail erasure L < 2044: 0.8.0 only; fixed in 0.9.1; covered.
- D5 latent DCP2 decode fallback without merge (sparse_attn_indexer_kpool.py:932-979): would expand rank-local ids as
  global; unreachable in shipped 0.7.0 since flattening forces requires_padding=False (inferred from indexer.py:736-738,
  891-984); a CPU assertion would confirm.

## 4. Bundles
- DCP2 ↔ selection path (decode merge, compaction, prefill merges). VLLM_GLM53_NOPE_RECORD=1 refuses DCP>1 → NOPE 528 B
  implies DCP1 + ownership; a DCP2 selection A/B on 0.9.1 also flips record to 656 B and drops ownership (= step 6).
- next_n ↔ route: rows = requests × next_n; B12X_FUSED_INDEXER=0 forces tiled (scratch.py v091:892, 939); no env knob for
  serial vs cooperative (policy fused_merge).
- B12x 7fcc094e ↔ port-b12x-latest-api.py: selection kernels not swappable without that patch.
- Expert partitioning untouched by these files; indexer replicated (32 heads, ReplicatedLinear); hold MoE fixed.

## 5. Ranked fixtures
1. F1 frozen-input selection parity + tie census (1 GPU, ~10–20 min/image, no server): dump q_fp8, weights, index-K pages,
   block table, pool seq_lens, positions for ~64 decode rows per DSA layer at L ∈ {2048, 2049, 2051, 4k, 16k, 64k, ~120k}
   from one long GPQA reply; replay 0.9.1 fused-cooperative and tiled (B12X_FUSED_INDEXER=0), 0.7.0 per-rank on
   parity-split pools + real _merge_b12x_dcp_topk math (all-gather → concat), and a torch reference (fp32 scores, exact
   top-511 lowest-id ties + tail). Compare 512-sets, attended token sets, rank of the col-511 pool, tie mass, exact-zero
   count, newest-pool selected. Actionable: set mismatch beyond ties = defect; frequent high-rank drops or big tie
   plateaus justify a shared fix (drop min-score column, or 512+tail 2051 contract); all match = area cleared.
2. F2 slice-order unit test (1 GPU, seconds): run_row_topk at both B12x commits, cooperative arm, CuteDSL,
   top_k_per_row_prefill on constructed rows: (a) exactly 512 above a coarse fp16 bin boundary; (b) 512 valid + −inf
   padding; (c) equal-score plateau straddling rank 512. Check col 511 = min selected score, then slice+expand.
   Actionable: best pool can land in col 511 → D1 real in every version; one-line fix.
3. F3 route A/B on the 16 unanswered items (2 GPUs, hours): 0.9.1 vs 0.9.1 B12X_FUSED_INDEXER=0, distinct seeds, question
   as unit. Only if F1/F2 show a route-dependent difference.

## 6. Limits
- top_k_per_row_prefill / persistent_topk ordering (compiled in base image) not visible; inferred unsorted with
  index-order path when row ≤ k; identical across versions; rarely reached in GPQA.
- DCP2 nsa_len cap uses CPU upper-bound per-token lengths (v070:1217-1251); harmless if a true upper bound (inferred);
  MLA area.
