# v0.7.0 vs v0.9.1: top-k order and sparse-MLA order sensitivity (source comparison, no GPU)

Date: 2026-10-24. No GPU, no network, no model download. Every image command was
`docker run --rm --network none -e CUDA_VISIBLE_DEVICES= --entrypoint bash <image> ...`.

| label | image | digest |
|---|---|---|
| **070** | v0.7.0 | `sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629` |
| **091** | v0.9.1 | `sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07` |

Evidence labels: **S** means read directly in source (or in a binary or log). **I** means
inferred. Line numbers are `file:line` inside the image. `vllm/` means
`/usr/local/lib/python3.12/dist-packages/vllm/`, and `b12x/` means `/opt/b12x/b12x/`.
When one number is given, the file is byte-identical in both images. When two are given,
they are written `070:L / 091:L`. Copies of every cited file are under `src-070/` and
`src-091/`, and their hashes are in `sources-sha256.txt`.

## The finding being explained

On v0.9.1 (stock, TP=2, prefill in 2,045-token chunks, prefix caching on), two identical
prefills first diverge at layer 3 at token position 2,051. The indexer top-k rows hold the
same pools in a different order, and the sparse-MLA output then differs on exactly those
rows.

## Short answers

| Q | Answer |
|---|---|
| 1. Does the prefill indexer path differ? | **No (S).** The code for indexer to top-k to merge to expand is identical: `sparse_attn_indexer_kpool.py` and `_C_stable_libtorch.abi3.so` are byte-identical, and `sparse_attn_indexer.py` changes only b12x import paths and plan staging. **The deployed configuration does differ (S).** v0.7.0 runs DCP=2, so the DCP merge stages run. v0.9.1 runs DCP=1 with MLA layer ownership (`split:25`), so the merge returns immediately and the raw `top_k_per_row_prefill` order goes straight to attention. |
| 2. What order does `top_k_per_row_prefill` produce? | **Not documented in the image (S).** The image has no source for the op, no docstring and no test for it. A local copy of upstream vLLM `csrc/sampler.cu` (commit not recorded; **not** from either image) gives two cases. When `rowLen <= topK`, it writes indices in index order. Otherwise it writes above-threshold elements to atomically allocated slots, then the threshold-bin elements sorted by logit (descending). So output order is not specified and depends on thread scheduling (I). **Nothing between top-k and attention normalises order, in either image (S).** v0.7.0's DCP stages add more atomic-slot reordering. |
| 3. Is sparse-MLA prefill order-sensitive? | **Yes, in both images (S for structure, I for bit-level effect).** The b12x MG prefill kernel gives each token row to one CTA. That CTA walks the index row in the order given, in tiles of 64 candidates. It carries a running softmax max and fp32 accumulators across tiles. It quantises P to FP8 using a scale computed per tile and per head. Different orders therefore produce different tile contents, different rescale sequences and different P quantisation scales. |
| 4. Is the mechanism in v0.7.0 too? | **Present (S for each part; I that it actually varies from run to run on v0.7.0, because no v0.7.0 run has measured it).** Both images use the same top-k binary, the same kpool selection threshold and an order-sensitive MLA prefill kernel of the same structure. Nothing in v0.7.0 makes the order deterministic. Its DCP=2 path adds atomic reordering instead of removing it. |
| 5. Does the decode path have the same mechanism? | **Yes, in both images.** Decode top-k (b12x tiled radix-select by default) writes to atomically allocated slots (S). Nothing normalises the order (S). The decode MLA kernel splits the index row in the order given into 64-candidate chunks, then into split-K ranges, and merges them, so it is order-sensitive (S for structure, I for bit-level effect). |

---

## Q1. Path from the model's attention module to the indexer, and what TP=2 actually reaches

### 1a. Call chain (identical logic in both images)

1. **The Indexer is always the kpool indexer.** `Glm5NextMLAAttention` builds `Indexer(...)`
   (`vllm/models/glm5next/nvidia/attention.py` 070:549 / 091:568). `Indexer.__init__` always
   builds `self.indexer_op = SparseAttnIndexerKpool(...)` (070:304 / 091:315). No switch
   selects a non-kpool indexer. `Indexer.forward` returns `self.indexer_op(...)` (070:403 /
   091:414) (S).
2. **The MLA wrapper calls the indexer.** `vllm/model_executor/layers/mla.py:224-225`
   (identical) calls `self.indexer(...)` when `is_sparse and not skip_topk`. `skip_topk`
   depends on `use_index_cache` (070:560-572 / 091:579-591), and that key is absent from the
   model config, so it is False (S).
3. **`SparseAttnIndexerKpool.forward_cuda` dispatches to the custom op.** It calls
   `torch.ops.vllm.sparse_attn_indexer_kpool` (`sparse_attn_indexer_kpool.py:1167`, identical
   file) (S).
4. **The prefill branch** (`sparse_attn_indexer_kpool.py`, identical in both):
   - `:461` sets the top-k buffer to -1.
   - `:481-506` handles short prefill (the `_full_topk_for_short_sequence` equivalent). The
     test is per batch, not per row. When `max_prefill_seq_len <= topk_tokens` (2048), every
     prefill row gets an index-ordered causal arange and **no top-k runs**. Otherwise every
     row in every chunk goes through top-k (`:531`).
   - `:561-568` computes DeepGEMM `fp8_fp4_mqa_logits`. These are pool-granular because
     `compress_ratio == index_kpool`.
   - `:574` sets `select_k = topk_tokens // index_kpool` = 2048 // 4 = **512 pools**. The
     model config has `index_topk=2048` and `index_kpool=4`.
   - `:576-582` allocates the `pool_topk` buffer, filled with -1.
   - `:600-609` calls `torch.ops._C.top_k_per_row_prefill(logits, cu_seqlen_ks, cu_seqlen_ke, pool_topk, num_rows, stride0, stride1, 512)`.
   - `:611-631` runs `_merge_b12x_kpool_dcp_topk_by_owner`. If that returns False, it runs
     `_merge_dcp_topk_global`.
   - `:633-653` runs `expand_pools_and_append_tail(pool_ids[:, :select_k-1], q_seq, 4)` and
     writes the result to `topk_indices_buffer` (S).
5. **Env switches** (all S):
   - `VLLM_USE_B12X_KPOOL_INDEXER` (`sparse_attn_indexer_kpool.py:63-67`) is used only at
     `:328` (profiling scratch), `:881-888` (decode b12x paged top-k) and `:1104-1107`. The
     **prefill** path always uses DeepGEMM logits plus `_C.top_k_per_row_prefill`, whatever
     this switch is set to.
   - `VLLM_USE_B12X_SPARSE_INDEXER=1` is baked into both images' environment, and both stacks
     set `VLLM_USE_B12X_KPOOL_INDEXER=1` (`stacks/glm53-flash-v0.7.0/start.sh:98-99`,
     `stacks/glm53-flash-v0.9.0/start.sh:142-143`).
   - `VLLM_DCP_TOPK_OWNER_MERGE` and `VLLM_B12X_DCP_TOPK_OWNER_EXCHANGE`
     (`vllm/envs.py:1096-1101`) only matter when DCP > 1.
   - `VLLM_B12X_MLA_CKV_GATHER` defaults to 0 (`vllm/envs.py:1108-1110`), and neither stack
     sets it.

### 1b. Differences that matter: the deployed parallel layout (S)

| | v0.7.0 recipe / stack | v0.9.1 recipe / stack |
|---|---|---|
| TP | 2 (`stacks/glm53-flash-v0.7.0/start.sh:32`) | 2 (`stacks/glm53-flash-v0.9.0/start.sh:32`) |
| DCP | **2** (`v0.7.0/start.sh:34`) | **1** (`v0.9.0/start.sh:34`) |
| MLA layers | TP/DCP-sharded | **Owner-local**: `GLM53_MLA_OWNERS=split:25` (`v0.9.0/start.sh:36-42`; recipe env `VLLM_GLM53_MLA_OWNERS=split:25`). Layer 3 lives on GPU 0 unsharded (`attention.py` 091:445-461, `owner_local`). |
| prefix caching | on (`v0.7.0/start.sh:61`) | on (`v0.9.0/start.sh:85`) |
| max-num-batched-tokens | 2048 (profile, `v0.7.0/start.sh:76,82`) | 2048 (`v0.9.0/start.sh:99,105`) |
| server log confirms | `decode_context_parallel_size=2`, `tensor_parallel_size=2` (`runs/gpqa/glm53-flash-exl3-k3.25@tpurtell+ps.server.log`) | `decode_context_parallel_size=1`, `tensor_parallel_size=2` (`runs/gpqa/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.server.log`) |

Both images share the same entrypoint (`/usr/local/bin/glm53-entrypoint`, byte-identical,
runs `vllm serve "$@"`). Neither image sets a default `Cmd`, so the served flags come only
from the stack `start.sh`.

### 1c. Which merge code is actually reached at layer 3 prefill

- **v0.9.1 (DCP=1)** (S). `_merge_b12x_kpool_dcp_topk_by_owner` returns False at
  `sparse_attn_indexer.py` 091:1380 (`dcp_world_size <= 1`). `_merge_dcp_topk_global` then
  returns at 091:823. The `top_k_per_row_prefill` output reaches the expand step untouched.
- **v0.7.0 (DCP=2)** (S for the code, I for which branch fires). The owner exchange runs only
  if all three hold (070:1342-1351, 1316-1328):
  - rows are divisible by 2;
  - `dcp_world_size == 2`;
  - rows >= `max(VLLM_B12X_DCP_TOPK_MIN_ROWS=128, qualified[512]=1024)` = 1024.

  It then reselects through b12x `run_row_topk` (070:1392). Otherwise it falls back to
  `_merge_dcp_topk_global`, which runs a CuteDSL "stable" top-k over the all-gathered
  candidates (070:829-852). With an odd chunk row count, such as 2,045, the fallback is taken
  (I: the chunk size on v0.7.0 has not been measured).

### 1d. Version differences on this path that are *not* order-related (S)

- `attention.py` adds the tail ring sized for spec drafts (091:191-209) and `owner_local`.
- `sparse_attn_indexer.py` adds b12x import-path adapters and a per-row-count plan for the
  owner exchange (091:1310-1345).
- `kpool_compress.py` changes the tail-ring addressing (`RING`) only. The expand kernel is
  identical (070:909-981 / 091:926-998).
- `b12x_mla_sparse.py` adds `_compact_dropped_selection` (091:725-773, called at 091:3040).
  It is a stable cumsum compaction, not a sort.
- The scheduler diff is one line in `_mamba_block_aligned_split` (091:410, the Eagle
  back-off). It can move chunk boundaries (I), but not the selection threshold.
- The threshold where sparse selection starts is the same: the identical kpool file plus the
  same model config.

---

## Q2. Order semantics of `top_k_per_row_prefill`, and anything that normalises order downstream

### 2a. What the images contain (S)

- **The binary is the same.** `_C_stable_libtorch.abi3.so` is byte-identical
  (`2b096e4b...`). All 17 `.so` files under `vllm/` are identical between the images.
- **No source is shipped.** There is no `.cu`, `.cuh` or `.cpp` for `top_k_per_row` in either
  image. The only `*topk*` CUDA sources are `vllm/third_party/fmha_sm100/...sparse_topk_select.*`,
  which belong to a different op.
- **No documentation.** The wrapper `vllm/_custom_ops.py:3018-3037` (identical) has no
  docstring and states no order contract.
- **No vLLM tests ship.** The b12x tests that compare indexer top-k results sort both sides
  first (`/opt/b12x/tests/attention/test_paged_indexer_integration.py:374-375,403-404`, both
  images). That suggests order is not part of the contract (I).
- **The compiled kernels match the upstream prefill signature.** The sm_120 cubin contains
  `vllm::topKPerRowPrefill<512,false>` and `<512,true>` with the 8-argument signature, and
  both contain shared-memory atomics (`ATOMS.ADD`, `ATOMS.POPC.INC`). See
  `evidence/topk-per-row-sm120-sass-atomics.txt`. This is consistent with atomic slot
  assignment, but does not prove it (I).

### 2b. Upstream reference (local copy, not from either image)

`upstream-ref/sampler.cu` is vLLM `csrc/sampler.cu`, fetched in an earlier session with the
commit not recorded. Its decode kernels include variants the image binary lacks
(`DeviceLengthAware`, `Decode512`), so it is **newer** than the compiled code. Its prefill
launcher and kernel signature match the binary's symbols. On that basis only (I), the
prefill kernel behaves as follows:

- **Short rows.** When `rowLen <= topK`, it writes `0..rowLen-1` in index order and pads the
  rest with -1 (`sampler.cu:403-424`; the comment says "Indices are not sorted by their
  corresponding logit"). This is deterministic.
- **Long rows.** It runs a radix histogram. Elements in bins above the threshold are written
  to `smemOutput[atomicAdd(&smemFoundTopKValues,1)]` (`:278-290`), so their **order depends
  on thread scheduling**. Threshold-bin elements are collected through a second `atomicAdd`
  (`:292-304`). They are then insertion-sorted (the launcher uses the insertion variant for
  the first 12,288 rows, `:988-1007`) by logit descending, with ties broken by their
  atomically assigned slot (`:479-500`), and placed after the above-threshold block
  (`:556-569`). If the threshold bin needs all four radix steps, its elements go out through
  a further `atomicAdd` with no sort (`:305-318`).
- **Fit with the finding (I).** Under kpool, a row at token position p sees `(p+1)//4`
  complete pools. 512 is exceeded first when p+1 = 2052, which is **p = 2,051**: exactly the
  first divergent position. Rows 2,045 to 2,050 in the second chunk do go through top-k,
  because the short-prefill test is per batch. But they have 512 or fewer pools, so they take
  the index-ordered shortcut. This assumes `cu_seqlen_ke - cu_seqlen_ks == (p+1)//4` per row;
  the metadata builder was not traced to confirm it.
- **The "same set, different order" pattern also fits (I).** The final column is a
  threshold-bin element, sorted last, so it is the lowest-logit selected pool regardless of
  how the above-threshold block was ordered. `pool_ids[:, :select_k-1]` drops that column
  (`sparse_attn_indexer_kpool.py:644`; this is the known D1 col-511 issue in
  `defect-repros/D1-col511-selection/`). The dropped pool is therefore the same from run to
  run when logits are equal and there are no ties at the threshold. That leaves the set
  stable while the order of the first ~510 columns varies.

### 2c. Stages between top-k and the attention kernel

None of these stages sorts or canonicalises order. A grep for `sort`, `argsort` and `unique`
over the traced files finds only an error-message `sorted()` at `b12x_mla_sparse.py` 091:313.

| stage | v0.9.1 (reached, DCP=1) | v0.7.0 (reached, DCP=2) |
|---|---|---|
| DCP merge | Skipped (091:1380, 091:823) (S) | Owner path: `run_row_topk`, a CuTe port of the sglang radix-select. Output slots come from `s_out[_smem_xadd(ctr,...)]` (`b12x/attention/dsa_indexer/tiled_topk.py` 070:895-896, 914-915) (S). Fallback: CuteDSL `StableTopKFromGatheredCandidatesKernel`. Its key (score, ~token_id) fixes the **set**, but output position is `atomic_add(committed_count)` (`dcp_indexer_cutedsl.py:320-335`) (S). Either way the output is order-arbitrary (I). |
| pool to token expand | `_expand_pools_and_append_tail_kernel`: column g maps to `pool_ids[row, g]`, so order is preserved (091:950-955) (S) | Same kernel (070:933-938) (S) |
| logical to physical slots | `_logical_topk_to_physical_slots`, elementwise (091:773-866) (S), then `_compact_dropped_selection`, a stable cumsum compaction (091:725-747) (S) | DCP branch (070:2751-2781): `triton_filter_and_convert_dcp_index` compacts each rank's winners with an atomic slot allocator. Its docstring says "Prefix order is unspecified (only the set matters)" (`sparse_utils.py:299-308`) (S). |
| design assumption stated in code | A comment says "Attention is a softmax over the selected set, so order does not affect the result" (`b12x_mla_sparse.py` 070:721-723 / 091:922-924) (S). This is true in exact arithmetic but not bit-exact in fp (Q3). | Same |

---

## Q3. Is the b12x sparse-MLA prefill order-sensitive?

### 3a. Route (S)

- **v0.7.0.** `b12x_mla_sparse.py` 070:3000-3048 calls `sparse_mla.run_extend`, which is
  `_shared/mla/api.sparse_mla_extend_forward` (`b12x/attention/sparse_mla/api.py` 070:19-20).
  That goes to `_run_sm120_prefill` (`_shared/mla/api.py` 070:788-805, 855-900), then to
  `prefill.run_unified_prefill` and `run_unified_prefill_mg` (`_shared/mla/prefill.py`
  070:295, 332).
- **v0.9.1.** `b12x_mla_sparse.py` 091:3219-3248 calls `sparse_mla.bind/run`
  (`sparse_mla/api.py` 091:61-129). That goes to `_preparation.State.run`, then
  `prefill.run_unified_prefill(...)` (`sparse_mla/_preparation.py` 091:232-259), then
  `run_unified_prefill_mg` (`_shared/mla/prefill.py` 091:311, 368).
- **No prefill split-K in either.** `_apply_mla_prefill_strategy` is defined
  (`_shared/mla/api.py` 070:186 / 091:189) but has no callers in either image.

### 3b. Kernel structure (`b12x/attention/_shared/mla/prefill_mg.py`), the same in both (S)

- **Grid.** `grid=(num_tokens * replicate_h, 1, 1)` (070:2177, 2232 / 091:2239, 2294). Each
  query token (and head-group replica) is handled by one CTA, with no cross-CTA reduction.
- **Tiles in given order.** Tile width is `_CAND_WINDOW = 64` (070:83 / 091:90). The row
  length is `topk_length[token]`, with `actual_tiles = ceil(len/64)` (070:2468-2474 /
  091:2540-2546). The sequential loop `for lc in cutlass.range(loop_tiles)` (070:2906 /
  091:3004) reads candidates `[ci*64, ci*64+64)` **in the index order it was given**:
  `split_cand_start = ci*64` and `index_base_ptr = topk_row + split_cand_start` (070:2922,
  2938 / 091:3020, 3036).
- **Online softmax.** The running max persists in shared memory across tiles (070:2455-2465 /
  091:2527-2537). The fp32 accumulators `acc0/acc1`, `rope`, and `gsum` carry across tiles
  (070:2886-2901 / 091:2984-2999). `s4_online_softmax_mg2` rescales them by
  `alpha = exp2(m_old - m_new)` when the max rises (070:1473, 1594-1626 / 091:1519,
  1640-1672).
- **P to FP8 with a per-tile scale.** On the GLM fp8 cache branch (`elif is_glm`, 070:3347 /
  091:3496), w_head_sc is zeroed every tile and then `s6_xv_nope` runs. That function takes a
  per-(vc, head) absmax of |P·V_scale| over the **current 64-candidate tile**, then quantises
  P to an e4m3 HIGH value plus a LOW residual (`_shared/mla/decode_math.py` 070:1741,
  1774-1799, 1855-1868 / 091:2038, 2072-2097, 2156-2169).

### 3c. Consequence (I)

Permuting the index row changes which candidates share a tile. That changes:

- the sequence of running-max updates and `alpha` rescales;
- the order of fp32 accumulation;
- each tile's FP8 quantisation scale for P.

The output is mathematically invariant but not bit-invariant. So "same set, different order"
gives different bits in both images. The two images' `prefill_mg.py` differ by about 494
changed lines: DSV4.1 FP8-PV and NVFP4 arms, record-format paths, and a partial-tile
length guard. None of these adds a sort or canonical gather, and the loop and quantisation
structure above is unchanged (S, by reading the anchors listed).

---

## Q4. Bottom line for v0.7.0

**The mechanism is present in v0.7.0** (S per component, I overall):

1. **Same top-k.** `_C_stable_libtorch.abi3.so` is byte-identical, so the same
   `top_k_per_row_prefill` runs, with the same atomic slot assignment in the compiled sm_120
   kernel (S, binary). Its order is not specified (I, from the upstream copy).
2. **Same selection start.** The kpool indexer file is byte-identical and the model config is
   the same (`index_topk=2048`, `index_kpool=4`). Selection starts at the same context length
   (more than 512 complete pools, first at p = 2,051) and passes through the same
   `[:select_k-1]` expand (S).
3. **Nothing normalises order.** There is no sort, unique or canonicalising gather between
   top-k and attention (S). v0.7.0's default DCP=2 inserts two more stages whose **output
   position is atomically assigned**: the merge re-top-k and the DCP filter-and-compact. So
   v0.7.0 has more reordering, not less (S for the code, I for the effect).
4. **Same kernel structure.** The MG prefill kernel is order-sensitive in the same way (S for
   structure).

Nothing in v0.7.0 would make the order deterministic. Two DCP=2 details differ, and neither
removes the mechanism (I):

- Each rank attends only to its local share of the selected tokens, and the outer layer
  combines the two partials by LSE (`b12x_mla_sparse.py` 070:2751-2781, comment at 070:2752-2754).
  The per-rank order sensitivity is the same.
- The token positions where divergence first appears may differ, because of DCP sharding and
  possibly different chunk boundaries.

**Not determined from source:**

- whether atomic slot order actually differs between two identical runs on v0.7.0 hardware
  (needs a measurement);
- the v0.7.0 prefill chunk row count, which decides whether the owner exchange or the
  CuteDSL fallback fires;
- the exact upstream commit compiled into `_C`.

---

## Q5. Decode path, briefly

- **Top-k (S).** With `VLLM_USE_B12X_KPOOL_INDEXER=1`, the conditions at
  `sparse_attn_indexer_kpool.py:881-887` hold, and decode uses b12x `index_topk_fp8` through
  `_run_b12x_paged_topk` (`:908-923`). The b12x side is `dsa_indexer/paged.py`, which calls
  `run_tiled_topk` / `run_row_topk` (070:28-30 / 091:27-29). That radix select writes output
  slots through `_smem_xadd` (`tiled_topk.py` 070:895-915, 947-975 / 091:936-956,
  988-1016), so its order is arbitrary (I). Padded or non-uniform batches fall back to
  `_C.persistent_topk` or `_C.top_k_per_row_decode` (`:944-979`); their order semantics were
  not checked (I).
- **Merge (S).** `_merge_b12x_dcp_topk` returns immediately at DCP=1 (`sparse_attn_indexer.py`
  1148-1149, both images). At v0.7.0 DCP=2 it runs `run_row_topk` (1207), which also uses
  atomic slots.
- **Expand (S).** `expand_pools_and_append_tail(pool_ids[:, :select_k-1], ...)` at `:1008-1010`
  preserves order, the same as prefill.
- **Decode MLA (S for structure).** v0.7.0 calls `run_unified_decode` (`_shared/mla/kernel.py`
  070:2734); v0.9.1 calls `run_prepared_unified_decode` (091:2720). Both split the index row
  in the given order into 64-candidate chunks (`_CAND_WINDOW`, 070:70-73 / 091:77-80),
  assign contiguous chunk ranges to split-K CTAs (loop at 070:1851-1857 / 091:1521-1527),
  apply per-tile online softmax and `s6_xv_nope` P quantisation inside each split, then
  merge the splits (070:3229-3232 / 091:3233-3235).
- **Result (I).** Decode is order-sensitive in both images in the same way as prefill, with
  split boundaries as an extra order-dependent partition.

---

## What remains undetermined

1. **No order contract.** The image has no source for `_C.top_k_per_row_prefill` or
   `_C.persistent_topk`. The order behaviour comes from an upstream copy whose commit is
   unknown and which is newer than the binary on the decode side. The binary only shows that
   shared atomics are present.
2. **Not measured on v0.7.0.** No GPU run on v0.7.0 has shown order variation. Section 4 is
   a source-level "present", not a measured one.
3. **The window assumption is unconfirmed.** That `cu_seqlen_ks/ke` gives each row exactly
   `(p+1)//4` complete pools (the basis for "first at p = 2,051") was inferred; the indexer
   metadata builder (`vllm/v1/attention/backends/mla/indexer.py`, identical in both) was not
   traced.
4. **Size of the effect is unknown.** How large the order-induced output differences are, and
   whether they matter for completion behaviour, cannot be read from source.
5. **v0.7.0 chunk geometry under DCP=2 is unknown**, and with it which merge branch
   (owner exchange or CuteDSL global) actually runs.

## Files

- `src-070/`, `src-091/`: copies of every cited in-image source file, at their in-image paths.
- `sources-sha256.txt`: per-image sha256 for every cited file (marked SAME or DIFF), plus the
  host-side files (recipes, stack scripts, server logs, model config) and the upstream copy.
- `upstream-ref/sampler.cu`: local copy of upstream vLLM `csrc/sampler.cu` (not from either
  image; commit not recorded).
- `evidence/topk-per-row-sm120-sass-atomics.txt`: sm_120 SASS check of the compiled top-k
  kernels in both images.
