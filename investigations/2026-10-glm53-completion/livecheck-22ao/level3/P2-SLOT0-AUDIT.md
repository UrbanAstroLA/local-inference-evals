# P2 slot-0 audit: what the sparse-MLA kernels read when an index is -1 (2026-10-23, zero GPU)

## (a) Question, as narrowed by the assessor

> Can the exact slot-zero storage read by the tested attention kernels contain values that invalidate the
> Level-3 masking argument at consumption?

The masking argument needs one thing from that storage: the 512 E4M3 latent bytes and the 4 FP32 group scales of the
528-byte GLM_NEXT record must be finite, so that P = 0 times the value is 0. It also needs any ordering assumption
behind that to hold. The question is not whether the storage is "never written".

**Evidence labels:**
- **S**: read in source (hashes below).
- **O**: observed in the 22ao run-1 arm-A records or server log.
- **I**: inferred.

**Scope:** stock 0.9.1, DCP1, TP2, GLM_NEXT / topk 2048, as served in 22ao arm A. Only the paths that can address the
MLA layers' KV storage are traced. This is not an engine-wide review.

**Sources:**
- Extracted read-only from base image `a9e985c0` (`docker run --rm --network none`).
- Every vllm file below is byte-identical in the arm A image `dd51ed73…`, which was re-hashed for this note.
- Arm A differs from stock only in `b12x_mla_sparse.py` and `sparse_attn_indexer_kpool.py`, which carry the
  observe-only guard (SOURCE-READING §1).
- Paths are relative to `dist-packages/` (vllm) and `/opt/b12x/` (b12x).

| sha256 (prefix) | File |
|---|---|
| `1dd3dd2826a2cc73` | vllm/v1/worker/gpu/attn_utils.py |
| `101e456b28fbebd0` | vllm/v1/worker/gpu/model_runner.py |
| `61ceb57c8e626b89` | vllm/v1/worker/gpu/block_table.py |
| `3929c92e42ae90e4` | vllm/v1/worker/gpu/input_batch.py |
| `82968af5e104f99f` | vllm/v1/worker/gpu/model_states/mamba_hybrid.py |
| `6c950a5854ef0c31` | vllm/v1/core/kv_cache_utils.py |
| `ddee56dccb220841` | vllm/v1/core/block_pool.py |
| `4d2a153eb8c1f3b4` | vllm/v1/core/single_type_kv_cache_manager.py |
| `3eab4c55d500aeb4` | vllm/config/vllm.py |
| `6e74c42cf4385bb0` | vllm/v1/attention/backends/mla/b12x_mla_sparse.py |
| `1eb8faecf4920a28` | vllm/v1/attention/backends/gdn_attn.py |
| `cb9a34eb45a94847` | vllm/v1/attention/backends/utils.py |
| `8391ef277d824174` | vllm/v1/attention/backends/flash_attn.py |
| `8e7ac8b29d002c6e` | vllm/model_executor/layers/mamba/abstract.py |
| `044d005cfe59fd08` | vllm/model_executor/layers/mamba/ops/causal_conv1d.py |
| `f89669f0427c1190` | vllm/model_executor/layers/mamba/ops/scatter_states.py |
| `371e7d733e90a714` | vllm/third_party/flash_linear_attention/ops/fused_recurrent.py |
| `6a20d0c77a4c32de` | vllm/third_party/flash_linear_attention/ops/kda.py |
| `8346d232ce3d0ffe` | vllm/models/glm5next/nvidia/kda.py |
| `487bb3e6696155e2` | vllm/v1/worker/mamba_utils.py |
| `482b4511561a902c` | vllm/v1/worker/gpu/spec_decode/dflash/speculator.py |
| `b983d8bf776b1a29` | vllm/model_executor/models/qwen3_dflash.py |
| `e7935f2a9b8ecbc7` | vllm/model_executor/layers/sparse_attn_indexer_kpool.py |
| `858e2de18cfc29b5` | b12x/attention/_shared/mla/kv_cache.py |
| `cdf14e155db81476` | b12x/attention/_shared/mla/io.py |
| `b582b693ca62fa71` | b12x/attention/_shared/mla/io_mg.py |
| `b41e8581147c1c5c` | livecheck-22ao/guard/mapguard_hooks.py |

---

## (b) Findings

### 1. The exact tensor and byte range

**Runner and model layout (O)**
- **Model runner:** the served runner is the V2 runner (`server.log:61` "Using V2 Model Runner"). KV allocation
  therefore goes through `gpu/attn_utils.py`. The V1 `gpu_model_runner.py:7605-7616` cited in SOURCE-READING is the
  same pattern, but it is not the served code.
- **Groups:**
  - MLA group: block 4352 tokens;
  - KpoolTail group: block 8;
  - 7 Mamba (KDA) groups: page 2,297,856 B;
  - one SlidingWindow DFlash2 draft group: block 1088.
  - Source: `server.log:241`.
- **Draft slot sharing:** active. `VLLM_GLM53_DRAFT_SLOT_SHARING=1`; "5 draft layers in MLA slots …
  page=2228224 of 2297856" (`server.log:240`).
- **MLA layer ownership:** rank 0 owns MLA layers 3-23 (6 layers); rank 1 owns 27-43 (5 layers) (`server.log:211,216`).

**Tensor sharing (S)**
- In the GLM5-Next layout there is one KV tensor per MLA layer (`kv_cache_utils.py:1765-1773`). Its `shared_by` list
  is:
  - that MLA layer;
  - one KDA layer from each mamba group;
  - one draft layer, if slot-shared.
- Every co-owner's page is padded to the MLA page (`:1460` draft, `:1561` mamba). Block id b therefore names the same
  bytes `[b·2,297,856, (b+1)·2,297,856)` in every co-owner of that tensor.
- Indexer layers get their own tensors, co-owned only by their sibling kpool-tail layer (`:1778-1784`).
- Each `KVCacheTensor` is a separate `torch.zeros` allocation (`attn_utils.py:200-216`, `:214`). The packed backing
  at `:209` is not used, because this layout never sets `block_stride`.

**The view the kernels read (S)**
- `get_kv_cache_shape` gives `(num_blocks·68, 64, 528)` (`b12x_mla_sparse.py:1074-1089`). This is the 528-byte
  record when `VLLM_GLM53_NOPE_RECORD=1`, which is set (O). The view is contiguous (`attn_utils.py:341`).
- `forward_mqa` passes it on unchanged as a contiguous uint8 `(blocks, 64, 528)` view (`b12x_mla_sparse.py:3054-3074`).
- The CKV-gather branch is not taken: its info_once line is absent from `server.log` (O).
- A -1 index is substituted with row 0 (`io.py:576-584`, `io_mg.py:264-274`). With block stride 64·528, row 0 is
  **bytes [0, 528) of each MLA layer's tensor**.

**Answer to item 1:**
- 11 distinct 528-byte ranges: one per owned MLA layer, per rank. 6 sit on rank 0 and 5 on rank 1.
- Each range sits inside block id 0, the null block.
- In the same physical bytes, block 0 is also:
  - **KDA conv state** of each co-owning KDA layer. The page is conv state first, at offset 0, then the fp32
    recurrent state (`mamba/abstract.py:29-43`, shapes `glm5next/nvidia/kda.py:138-175`);
  - **draft token 0 of draft block 0**. FLASH_ATTN is logical `(B, H, N, 2D)` with NHD stride order, so physically
    `(B, N, H, 2D)` (`flash_attn.py:139-173`). That is 2,048 B per token, so token 0 occupies `[0, 2048)`. No
    layout override is logged (O).
- The conv region is `[0, 147,456)` of the page: 1 + 2.37 % padding of 147,456 + 2,097,152, which fits H = 64,
  D = 128 (I).

### 2. Initialization and lifetime

- **Zero-fill (S):**
  - `torch.zeros` per tensor (`attn_utils.py:214`), called once from `init_kv_cache` (`model_runner.py:618`).
  - The profiling-only minimal cache is a separate throwaway allocation, torn down before the real init
    (`model_runner.py` `_init_minimal_kv_cache_for_profiling` / `_cleanup_profiling_kv_cache`).
  - No `zero_`/`fill_` on KV storage was found in the runner, model-state, KDA, GDN or b12x backend files (S, grep).
- **Block 0 is the null block (S):** it is popped from the free queue at pool creation and marked `is_null`
  (`block_pool.py:190-191`).
  - There is one block pool for all groups, so block 0 is null for **every** co-owning group.
  - The KV managers place it only as a placeholder: SWA out-of-window blocks, freed align-mode blocks
    (`single_type_kv_cache_manager.py:1543-1570`).
  - The upstream code itself states the contract: "null_block KV is all-zero" (`:860`).
- **Ordering (O):** allocation precedes warm-up and capture (`server.log:239-241`, then 352+).

### 3. Writers and aliases that can reach block 0 of an MLA tensor

**a. GLM_NEXT writer** (`concat_and_cache_glm_next_mla`)
- **Negative slots are skipped (S):** `kv_cache.py:34`, `:291-292`, `:504-505`.
- **Real tokens (S):** `slot = block·4352 + pos%4352` comes from the request's MLA-group block table
  (`block_table.py:301-318`). Full-attention tables hold no null entries. Slot 0 would need block 0, which is never
  allocated.
- **Padding (S):**
  - padded tokens get `PAD_SLOT_ID = -1`, starting from the actual token count (`block_table.py:284-291`);
  - dummy, warm-up and capture runs fill all slots with PAD (`:213`);
  - DFlash capture uses the same dummy PAD slots (`dflash/cudagraph.py:35-36`).
- **Observed (O):** `mla_w` checked 821,766 (rank 0) + 684,805 (rank 1) written tokens. All matched the expected own
  slot: mism / unexp / null / inact = 0.

**b. KDA (mamba) writers on the co-owned tensor** (`glm5next/nvidia/kda.py`)
- **Conv state, the region that contains slot 0 (S):**
  - `causal_conv1d_update` (`:513`, `:547`) and `causal_conv1d_fn` (`:530`) use the default
    `null_block_id = NULL_BLOCK_ID = 0` (`causal_conv1d.py:491`, `:1106`; `utils.py:46`).
  - Both return before any load or store when the state index is 0 (`causal_conv1d.py:143-148`, `:834-838`).
  - The stores go to the same index column (`:225-244`, `:946-957`).
  - `causal_conv1d_fn` also skips PAD (`:86`).
- **Recurrent state (S):** `fused_recurrent_kda` (`:613`, `:705`) skips loads for `state_idx <= 0` and stores only
  when the index is > 0 (`fused_recurrent.py:120-126`, `:178-183`).
- **`scatter_states` (`kda.py:691`, prefill) has no guard (S, `scatter_states.py:31-33`):**
  - it writes the **recurrent** state, which sits at page offset ≥ 147,456, not slot 0;
  - dummy runs zero the block tables, so their mamba state goes to block 0 by design (`block_table.py:165-176`);
  - that write lands in block-0 bytes that no reader uses.
- **Padded rows under FULL graphs (S):** they get `NULL_BLOCK_ID` (`gdn_attn.py:707`, `:753`), and the kernels above
  then skip them.
- **Align-mode pre-copy / post-copy (S):** the copy kernel has no 0-guard on its destination
  (`v1/worker/mamba_utils.py:174-181`). Its destination is either:
  - the new running column, for pre-copy (`:493-499`); or
  - the just-completed block, for post-copy (`:330-350`).
  Both are allocated for the step, and the post-copy destination is not the "two steps ago" block that align mode
  frees (I).
  - Pre-copy runs on real batches only (`mamba_hybrid.py:211-258`).

**c. DFlash2 draft writer on the co-owned tensor** (`dflash/speculator.py`)
- It would reach `[0, 528)` only by writing draft slot 0, meaning block 0 at offset 0.
- **Context K/V (S):**
  - skipped in dummy runs (`:411-412`, `qwen3_dflash.py:626-627`);
  - in real steps it is written only for the actual tokens (`:334`), at their scheduled positions;
  - those positions are inside the sliding window by construction (`single_type_kv_cache_manager.py:1059-1084`).
- **Query K/V (S):** `q_slot = table[q_pos//1088]·1088 + q_pos%1088` (`:563-575`), with padding set to PAD (`:630-635`).
  - In real steps the query positions are covered by the DFlash lookahead `num_speculative_tokens + 1 = 4`
    (`config/vllm.py:603-620`). The block is therefore allocated.
  - In dummy runs the tables are zeroed and the positions are 0 (`input_batch.py:163`). So `q_pos = 1…4`, which
    gives draft slots 1-4 = bytes `[2048, 10240)`. That is past slot 0, at startup only.

**d. Kpool tail stash/seed** (the "null" events)
- **Which storage (S):** the tail group co-owns the **indexer** tensor (`kv_cache_utils.py:1778-1784`, padding
  `:1545`). The decode stash writes `tail_kv_cache` through the tail group's slot mapping
  (`sparse_attn_indexer_kpool.py:768-797`). That storage is disjoint from every MLA tensor.
- **What "null" means (S):** at site `tail_w_stash`, a write whose consumed tail block `slot//8 == 0` and that
  mismatched or was unexpected (`mapguard_hooks.py:323`, `:1549-1553`).
- **Totals (O):** 587,496 (rank 0) + 489,580 (rank 1) = 1,077,076.
- **Retained records (O):** these cover only steps 2-78 because of the CAP: 5,158 + 5,459 rows. All have
  `consumed == position % 8` (tail slots 0-7), owner-group -1, and expected a live tail block.
- **Conclusion:** these are writes into the **indexer tensor's** null block, so they do not overlap MLA slot 0 (S).
  `tail_w_seed` null events (2,616 + 2,180) are in the same storage.
- **Out of scope:** whether the indexer reads its own block 0.

**e. Other**
- No MTP, KV connector, CKV gather, or sleep/reset path is active (O: launch args, log).
- No `fill_`/`zero_` on KV storage was found (S).

### 4. Ordering relative to attention reads

- No writer found addresses `[0, 528)` after the zero-fill, so the argument needs no within-step ordering assumption.
- The only block-0 writes found are both at startup, before any request, and never read by attention:
  - KDA recurrent-state scatter, at offset ≥ 147,456;
  - draft query K/V at `[2048, 10240)`.
- Attention reads block 0 only through the -1 → row 0 substitution. Valid ids never map into block 0, because
  `_logical_topk_to_physical_slots` maps an unmapped block to -1 (SOURCE-READING §2).
- The masking argument's precondition (finite slot 0) therefore holds from allocation onward, with no race to rule
  out (S + I).

### 5. What the guard records can support (bounded)

**Supported (O)**
- No real-token MLA write landed outside its own slot over 3,046 steps on both ranks. That includes none landing in
  block 0.
- The 1,077,076 `tail_w_stash` null events are indexer-tensor writes, as the record fields show (site 4, group
  `g_tail = 1`, slots 0-7).

**Not supported**
- **Inactive-token writes into block 0 are invisible to the guard:** `inact` requires `cb != 0`
  (`mapguard_hooks.py:412`).
- **The KDA and draft writers are not hooked at all.**
- **No record reads slot-0 bytes.**
- **Record sampling is limited:** retained records cover steps ≤ 78; counters cover every step.

**Indirect only (I)**
- The Level-3 N-probe showed that a non-finite slot 0 turns about 85 % of rows to NaN.
- Arm A served normal spec-decode acceptance, mean acceptance length 2.5-3.3 (`server.log:477-506`). Wholesale NaN attention
  on any layer is hard to reconcile with that.
- This is corroboration, not proof.

---

## (c) Verdict

**Slot-0 storage cannot hold non-finite values at consumption in the served 0.9.1 path (S).**
- It is the first 528 bytes of each of the 11 MLA layer tensors, zero-filled at allocation. Every writer that can
  address that tensor either never targets those bytes or skips index 0 / -1:
  - the GLM_NEXT writer;
  - 7 co-owning KDA layers × 4 kernels;
  - the align-mode copies;
  - the slot-shared DFlash2 draft layer.
- Slot 0 therefore stays all-zero: E4M3 +0 and FP32 scale 0.0. Those are finite, so `0 · value = 0` holds. This is the
  "slot 0 zero" case that Level 3 measured as bit-equal.

**Residual inference (I):**
- the writer census is complete. It was a targeted trace, not a mechanical enumeration of every kernel that receives
  these tensors;
- three allocator invariants: real requests never hold block 0 in the MLA, mamba-running or draft-in-window
  columns; align-mode copy destinations are allocated blocks; the DFlash lookahead covers every query position.

**Side finding (outside the question):**
- Under slot sharing, any future change that lets a co-owner write block 0 would land non-E4M3 data in MLA slot 0.
  Bf16 or fp32 bytes give E4M3 NaN codes (0x7F/0xFF) with high probability.
- Examples: a mamba kernel without the null-block skip; a draft write to draft slot 0; an NHD → HND switch (draft
  token 1 / head 0 would sit at `[512, 1024)`, over the scale bytes).
- The safety is a property of those guards, not of the masking.

## (d) Smallest further checks (optional; the verdict does not depend on them)

1. **CPU, about 15 min, zero GPU:** a mechanical census. List every call site that receives `layer.kv_cache` or
   `kv_caches[...]` for a layer named in any MLA tensor's `shared_by`. Confirm each site carries one of the guards
   cited above. This turns the completeness inference into source.
2. **Piggy-back on the next authorized serving run (suggestion only, never run here).** Add two observe-only items:
   - at each guard poll, a copy of bytes `[0, 528)` of the 11 MLA tensors to the host (5.8 KB), recording "all zero";
   - in the `mla_w` site, drop `cb != 0` from `inact`, so inactive writes into block 0 are counted.
   - **Cost:** zero added GPU-hours if folded into an already-approved run. Standalone it would be about 0.5 GPU-h
     on 2 GPUs (startup plus a few requests).
