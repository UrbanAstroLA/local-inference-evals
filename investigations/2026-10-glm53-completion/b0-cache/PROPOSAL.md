# B0 cache isolation and high-offset addressing: fixture proposal (PLAN amendment 16, second priority)

Status: **prepared, not run.** This item is priced separately and needs its own operator budget decision. CPU preparation only: no GPU, no server, no image changes. The runner refuses to start without `APPROVED_GPU_H`.

Checkpoint B0:
- image `sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548`;
- recipe `glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234`;
- configuration: TP2, DCP1, owners split:25, NOPE 528-B record, DFlash2 ×3, `fp8_ds_mla`, prefix caching, mamba `align`, FULL_AND_PIECEWISE.

Source:
- The vLLM tree, `/opt/b12x` and `/opt/glm53` were extracted from the image.
- vLLM equals the release tree plus the five bundle files (`diff -rq`); B12x is byte-identical to release.
- Paths below are relative to the image's `vllm/` package unless prefixed `b12x/`.
- **(inf.)** marks an inference not verified by execution.

---

## 1. Geometry on the shipped configuration (rank 0)

Cross-checked three ways:
- derived from source;
- against the B0 log line `glm53-placement kv-capacity` (`bundle/screen/run-d1234/server-s1.log:240-242`);
- by building the views with B0's own view code on full-size meta tensors (`cpu-tests/cpu-tests.json`, T2: 107 offset checks, all pass).

**Pool and tensors**
- 1,275 blocks per tensor, one shared BlockPool, 10 KV groups.
- Rank 0 owns MLA layers 3, 7, 11, 15, 19 and 23; rank 1 owns 27, 31, 35, 39 and 43.
- Rank 0 total: 18,677,260,800 B = 1,275 × (6 × 2,297,856 + 6 × 143,616). This matches the log.

| Tensor (per owned MLA layer i) | Bytes | Co-owners (`shared_by`, `kv_cache_utils.py:1745-1795`) | Kernel view (from B0 code) |
|---|---|---|---|
| **MLA slot tensor** | 2,929,766,400 (> 2^31) | MLA layer i; KDA layer i of each of the 7 mamba groups (34 KDA layers: groups of 5,5,5,5,5,5,4); DFlash2 draft layer i (5 draft layers; rank 0's 6th slot has none) | see the three rows below |
| ↳ MLA (group 0) | page 2,297,856 = 4,352 tok × 528 B | – | uint8 `(86,700, 64, 528)`, stride `(33,792, 528, 1)`; storage block split into 68 kernel pages of 64 (`b12x_mla_sparse.py:1012-1016, 1074-1090`; `attn_utils.py:388-430`) |
| ↳ KDA (groups 2-8) | real 2,244,608, padded to 2,297,856 (+2.37 %) | – | conv bf16 SD `(6, 12288)` at page byte 0 (kernel view `(1275, 12288, 6)`, strides `(1,148,928, 1, 12,288)`). ssm fp32 `(32, 128, 128)` at byte 147,456 (stride 574,464). Bytes 2,244,608 to 2,297,855 are never touched. Sources: `mamba/abstract.py:29-43`, `mamba_utils.py:396-419`, `kda.py:450` |
| ↳ draft (group 9) | real 2,228,224 = 1,088 tok × 2,048 B, padded to the MLA page | – | FlashAttention logical `(B, H, N, 2D)`, NHD, so physical `(1275, 1088, 4, 256)`. As_strided block stride 1,148,928 elements (`flash_attn.py:138-170`; `attn_utils.py:295-322`). K/V split at `flash_attn.py:989, 1201` |
| **idx tensor** | 183,110,400 | index-K layer i (group 0, same block table as MLA) + kpool tail layer i (group 1) | index-K uint8 `(21,675, 64, 132)`, page-planar: 64×128 E4M3, then 64 fp32 scales at byte 8,192; 17 pages per block. Tail bf16 `(1275, 2, 8, 128)`, strides `(71,808, 1,024, 128, 1)`; tail block b = bytes `[b·143,616, +4,096)` |

**Where 2^31 falls**
- Block b's page starts at b × 2,297,856. Pages start above 2^31 for **b ≥ 935**: 340 of 1,275 blocks, 26.7 % of the pool.
- The MLA record at **slot 4,067,203** (block 934, token 2,435; bytes 2,147,483,184 to 2,147,483,711) straddles 2^31. 2^31 lands at record byte 464, inside latent group 3; all 16 scale bytes lie above it.
- The idx tensor never reaches 2^31.

**Unit matters** **(inf., arithmetic)**
- In element units, the bf16 draft and KDA views and the fp32 ssm view stay below 2^31 elements up to block 1,274 (largest element offset about 1.46e9). Even an int32 element-offset kernel would not overflow there.
- The 2^31 hazard is real only for **byte-unit** (uint8) addressing: the MLA record writer and readers, and the byte-copy kernels.

## 2. Writers and readers, and how they address the cache

All block ids and slots are int32 on the host and device side; the largest slot, 5,548,799, fits. The "Offset width" column is what matters at 2^31.

| Path | Kernel (B0) | Offset width | Stride source | Notes / citation |
|---|---|---|---|---|
| **MLA record writer** | B12x `ConcatAndCacheGlmNextMlaKernel` via `concat_and_cache_glm_next_mla(plan=…)` | **Int64**: `base + (slot//64)·stride0 + (slot%64)·stride1` | runtime Int64 from the tensor | `b12x/…/mla/kv_cache.py:197-210, 444-586, 684-725`. Plan is primed on 1-page dummies (`b12x_mla_sparse.py:105-142`); slot < 0 is skipped |
| **MLA readers** (decode / extend) | B12x unified decode (`io.py:563-612`, packed GLM_NEXT arm, `kernel.py:455-479`); prefill `io_mg.py:272-274` | **Int64** `Int64(blk)·stride_kv_block + Int64(loc)·528`; indices Int32 | computed 33,792 (contiguous) or real stride0 (`kernel.py:2037-2067`) | Indexer emits logical positions; `_logical_topk_to_physical_slots` maps them int32 (`b12x_mla_sparse.py:773-845`). **(inf.)** CuTe pointer + Int64 lowers to a 64-bit GEP; the static descriptor from the prime dummy is a residual marshalling risk |
| **Draft K/V writer** (context precompute + query block) | `reshape_and_cache_flash` (compiled `_C`, **no source in image**) via `FlashAttentionImpl.do_kv_cache_update` (`flash_attn.py:1190-1220`; `qwen3_dflash.py:616-636`) | **not auditable**; upstream uses int64 slot and strides **(inf.)** | `key_cache.stride()` of the padded view | Slots: `_prepare_dflash_inputs_kernel`, int64 `block_id·1088 + pos%1088` (`dflash/speculator.py:476-575`). Query lookahead covered: `num_lookahead_tokens = K+1 = 4` (`config/vllm.py:603-620`) |
| **Draft reader** | vllm_flash_attn **FA2** (`_vllm_fa2_C`, no source) with `block_table` | not auditable; element-unit offsets stay below 2^31 at this geometry **(inf.)** | `k.stride(0)` = 1,148,928 elements | log `Using FlashAttention version 2` (`server-s1.log`). Window 2048, block 1088 |
| **KDA conv** (decode / spec / prefill) | `causal_conv1d_update`, `causal_conv1d_fn` (Triton) | index `.to(int64)` × constexpr stride, elements | `conv_state.stride()` | `causal_conv1d.py:139-152, 830-952`; null block guarded |
| **KDA recurrent** (spec / decode) | `fused_recurrent_kda` (Triton) | `state_idx.to(int64)·stride0`, elements | `initial_state.stride(0)` | `fla/ops/fused_recurrent.py:116-126, 176-185`; writes 4 columns per spec row; skips `idx ≤ 0` |
| **KDA prefill gather / scatter** | `gather_initial_states`, `scatter_states` (Triton) | int64 | `stride(0)` | `ops/gather_initial_states.py:35-38`, `ops/scatter_states.py:31-33`. Not null-guarded |
| **KDA align copies** (precopy, postprocess) | `precopy_mamba_align_fused_kernel`, `postprocess_mamba_fused_kernel` (Triton) | **bytes, Int64** (comment names the 2^31 hazard) | int64 tables from `data_ptr` and `stride(0)·elsize` = 2,297,856 | `v1/worker/mamba_utils.py:117-251, 393-523, 792-840`. Not null-guarded |
| **Partial prefix-hit copy-on-write** | `copy_kv_cache_blocks_inplace` (torch, whole page) | torch int64 | page = storage / num_blocks | `v1/worker/utils.py:622-656`; prefix hit rate was 0.0 % in every B0 log line, so dormant in GPQA |
| **New-block zeroing** | `KVBlockZeroer` / `_zero_kv_blocks_kernel` (Triton) | `block_id.to(int64)·stride`, int32-word units | from the view strides | `v1/worker/utils.py:71-260`. Active because the config has mamba layers (`kv_cache_interface.py:1165-1173`). Zeroes whole MLA and idx pages of every **new attention-group block id**; the draft and tail views dedupe onto the same storage |
| **Index-K writers** | `kpool_compress_and_write_cache`, decode-update batched (Triton) | elements, int64 `loc`; constexpr page stride | `kv_cache.stride(0)` (8,448) | `glm5next/nvidia/ops/kpool_compress.py:206-445, 543-811` |
| **Tail ring writers** | `kpool_seed_tail_cache`, decode-update stash (Triton) | int64 | `tail.stride(0)`, `stride(1)` (padded 71,808) | `kpool_compress.py:448-526`. B0's D2 maps null-block requests to −1 (`indexer.py:532-573, 621-634`) |
| **Index-K reader** | B12x fused paged score + top-k (`fused_indexer.py:237-247, 3069-3070`) | Int64 `page_id·page_stride` | tensor `stride(0)` baked as constexpr | Prefill readers `cp_gather_indexer_k_quant_cache` and DeepGEMM `fp8_fp4_paged_mqa_logits` are compiled with no source (**not auditable**) |

**Isolation by construction.** Each group has its own block table and all draw from one BlockPool, so a block id belongs to at most one group at a time. Within a tensor, co-owners therefore use disjoint pages. Every view whose page is padded uses the padded page as its block stride.

**What reading cannot settle:**
1. the compiled draft writer and FA2 reader, the only cache touches in the shared tensor with no source;
2. CuTe lowering of the Int64 MLA addressing on the live 2.93 GB tensor, with a descriptor primed on a dummy;
3. Triton constexpr-stride specialisation on the real views;
4. replay of captured graphs.

## 3. Reachability in serving

**Allocation order (source):**
- The free queue starts in block-id order: `FreeKVCacheBlockQueue`, `kv_cache_utils.py:186-205`; block 0 is the null block (`block_pool.py:188-191`).
- New blocks are popped from the front (`block_pool.py:647-677`).
- In this fork, freed blocks without a hash are **prepended** (LIFO reuse), and blocks with a hash (full, prefix-cacheable) are **appended** (FIFO / LRU) (`block_pool.py:719-743`).
- So the queue head advances into never-used, higher ids at the rate at which hashed full blocks are retired. High ids are reached without high KV usage.

**Log evidence (B0 screens):**
- KV usage peaked at 38.9 % (pass 1, during warm-up with 16 requests) and 31.9 % (pass 2). It is about 20 % at concurrency 8.
- Prefix cache hit rate was 0.0 % on all 805 and 744 logger lines.
- The logs do not record block ids, so they cannot show which ids were used. That is answered by the simulation below.

**Simulation with the real allocator**
- Method: CPU only, inside the B0 image, no GPU (`reachability/sim_pool.py`, outputs `out-*.json`). The real `KVCacheManager`, `HybridKVCacheCoordinator`, BlockPool and single-type managers are given B0's 10-group scheduler config (built through `generate_scheduler_kv_cache_config`; `hash_block_size` 1,088; lookahead 4). They are driven by the per-request prompt and completion token counts of both B0 GPQA passes (`<gpqa-runs>/…bundle-d1234.s{1,2}.requests.jsonl`), with concurrency 8 and chunked prefill 2,048.
- **(inf.)** The scheduler loop is approximated: a fixed mean of 2.6 accepted tokens per step, with 2.0 and 3.2 tested for sensitivity.
- Results:
  - **Every non-null id, 1 to 1,274, is handed out** in both passes and at all three acceptance rates.
  - Ids ≥ 935 are first handed out about **9 % into a pass** (pass 1: KDA 8.7 %, draft 8.9 %, MLA 9.1 %; pass 2: about 9.5 %). That is about 12 minutes into a 2.2-hour pass **(inf.)**.
  - Per pass, ids ≥ 935 were held by MLA+idx 157-172, draft 294-315, each KDA group 141-209, and tail 36-42.
  - Cross-group reuse of the same block id is routine: about 10,700-11,400 cross-group reallocations per pass against about 1,800-2,000 same-group ones. The most frequent are KDA→KDA (about 650-700 per pair), draft→MLA (640) and MLA→draft (298), followed by draft→KDA and KDA→draft. These are exactly the "same bytes, new owner and new view" events the fixture reproduces.
  - **No block id was ever held by two groups at once** (exclusivity check at every simulated step, 0 violations). This confirms dispatch audit item 12 with the real allocator, not only by reading.
- **Null block 0**:
  - It is written by dummy and padding rows on the MLA and index-K paths. Those slots come from table entries of 0, not −1, so the writes are confined to page 0.
  - The KDA kernels guard against block 0. The KDA copy and gather/scatter kernels do not; they see only scheduler-managed columns **(inf.)**.
  - `_logical_topk_to_physical_slots` does not reject block 0 (`b12x_mla_sparse.py:773-845`). It is harmless only if valid positions never map to block 0 **(inf.)**.

## 4. Fixture (model-free, one GPU, ready)

**Files and hashes.** Everything is under `fixture/` and is frozen by `fixture.sha256`:
- `b0geom.py`: geometry and footprints;
- `b0shadow.py`: sentinel pattern, shadow image and payload tags;
- `b0views.py`: views built by B0's own code;
- `run_fixture.py`: the parts listed below;
- `inner.sh` and `run_b0_cache_fixture.sh`: the runner;
- `test_repaired_high_page.py`.

**Method**
- Allocate one MLA slot tensor and one idx tensor at real size.
- Build every view with B0's code: `_reshape_attention_kv_cache` with each backend's `get_kv_cache_shape` and stride order, plus `MambaBase.bind_kv_cache`.
- Fill both tensors with a per-8-byte-word sentinel derived from (offset, salt) and keep a same-size **shadow image** of the bytes that must be present.
- After **every** operation, require `raw == shadow` over the full 2.93 GB. A differing byte is reported with its block, region (MLA token, draft token, KDA conv/ssm/pad, index-K/scale/tail) and decoded payload tag.
- Payloads carry group, block, token and epoch tags, so a stray write identifies its owner:
  - bf16 and fp32 values are exact;
  - MLA latents are E4M3-exact, with amax 448·2^-6 per group.

| Part | Drives (real B0 kernel) | Addresses | Pass criterion (frozen) |
|---|---|---|---|
| A | draft writer `reshape_and_cache_flash` on the padded view, with PAD rows | blocks 1, 934, 935, 1274; tokens 0, 1, 543, 1087 | exact bytes in footprint; sentinel elsewhere |
| B | **FA2 paged reader**, non-causal, window 2047 | L = 2,100 context in blocks {2,3}, {1273,1274}, {934,935}; different stale bytes beyond `seq_len` in each set | high sets bit-equal low; vs fp32 torch: cos ≥ 0.999, max abs ≤ 3e-2; reads write nothing |
| C1 | `scatter_states` / `gather_initial_states` | 1, 934, 935, 1274 | exact |
| C2 / C2b | `causal_conv1d_update` (decode and spec: 4 tokens, accepted 2) and `causal_conv1d_fn` (prefill, T = 37) | twins (2, 1274), (3, 935), (4, 934) | outputs and states bit-equal across twins; only the conv region changes |
| C3 | `fused_recurrent_kda` spec (4 columns per row, h0 from column 1; gate and beta computed as shipped) | rows [1-4] vs [935, 1100, 1273, 1274]; [5-8] vs [930, 933, 934, 1271] | outputs and all 4 states bit-equal; only the ssm region changes |
| C4 | `precopy_mamba_align_fused_kernel` (direct launch, metadata as in `mamba_utils.py:792-840`) | rows [10-13] vs [1271-1274] | exact conv shift and ssm copy |
| C5 | `copy_kv_cache_blocks_inplace` | (2→1270), (1274→7) | exact page copies |
| D1 | **GLM_NEXT writer** with vLLM's primed plan (`_prepare_glm_next_writer_plan`), PAD rows | low block 2; 64 records around the **straddling slot 4,067,203**; block 935; the **last 64 records of block 1274**; a kernel-page crossing in block 1100 | all sets byte-equal the low twin; low decodes to the latents (B12x pure-torch unpack); bytes vs the reference pack reported |
| D2 | **B12x sparse MLA decode and extend readers**, 4 rows × 64 heads × width 64 | the same sets, permuted | high sets bit-equal low; vs fp32 reference cos > 0.995, max abs < 3e-2 (the B12x test's thresholds) |
| D3 | **graph replay**: writer and decode reader captured once, replayed with new slots and selections (high, straddle, reuse) | as D1 | replayed writes byte-equal eager twins; replayed reads equal eager; a **control** selection must change the output (rules out bind-time copies) |
| E1-E3 | `kpool_seed_tail_cache` (exact; PAD rows); `kpool_compress_and_write_cache` (twins); decode-update batched (spec 4, pool completion) | tail and index-K at blocks 1-6, 934, 935, 1273, 1274; pools 0, 63, 64, 1023, 1087 | exact or twin bytes; only the tail-ring slots and the target pool's K and scale change |
| F1 | real `KVBlockZeroer`, with segments built from fake groups in serving order | 5, 935, 1274 | exactly the MLA and idx pages zeroed |
| F2 | **cross-group reuse sequence** on 935 and 1274: draft→MLA→KDA→KDA→draft→KDA→MLA, with zeroing for attention owners, each owner's write plus read-back | 935, 1274 | each step: the new owner reads only its own tagged data; sentinel and shadow intact |
| G | draft writer captured in a CUDA graph, replayed with high, PAD and reused slots | 6, 935, 1274 | exact |
| P | existing B12x > 2^31 tests (pytest; image has pytest 9.1.1): `test_attention_mla_sm120.py -k high` (GLM_NSA, current API); `test_glm_next_mla.py::…live_high_page` as shipped and **repaired** | padded-stride path, page 56,681 | recorded outcome. The shipped GLM_NEXT test calls the writer **without the required `plan=`** (`b12x/attention/sparse_mla/api.py:133-141`), so it is expected to fail with TypeError: an obsolete test interface **(inf.)**. The repaired copy supplies the plan |

**CPU validation already done** (B0 image, no GPU; `cpu-tests/`):
- **T1** geometry derivation matches the log line.
- **T2** the B0 view code, on full-size meta tensors, gives exactly the footprint model's byte offsets: 107 checks over blocks 1, 934, 935 and 1274 and the straddling record.
- **T3** on a small tensor, the sentinel and shadow detect a misplaced write, a 2-byte overrun and a missing write, and locate each.
- **T4** at **full 2.93 GB size** on CPU, tracking is exact for high-offset writes, including the straddling record and the last record. Stray writes at exactly 2^31 and at the tensor's last word are located.
- **T5** payload tags round-trip, and the designed MLA latent packs and unpacks exactly with B12x's pure-torch reference.
- **Import smoke test** (`cpu-import-smoke.json`, device = cpu): setup, full fill and checks pass. Every part reaches its CUDA kernel call, then fails as expected on CPU (no name or import errors other than CUDA-gated modules).

**Not yet executed on a GPU.** The C4 precopy direct launch and the D2/D3 B12x plan/bind glue are the least certain. Fixing a glue error on the first run would take minutes of engineering time; it does not change the price below.

**Runner** (`run_b0_cache_fixture.sh`):
- **Image pin**: by id.
- **Budget gate**: sums `item = b0-cache-fixture` in `costs.jsonl`; refuses unless spent + 0.5 ≤ `APPROVED_GPU_H`.
- **Pre-flight**: refuses if an inference container is running, or if GPU 0 shows more than 1 GiB used, more than 5 % utilisation, or any process in the plain `nvidia-smi` table. The plain table includes graphics clients, which `--query-compute-apps` misses.
- **Container**: one, with `--gpus device=0` and no network.
- **Caps**: wall cap 1,800 s (timeout, then docker kill); inner caps 1,380 s for the fixture and 360 s per pytest step. Results JSON is rewritten after each part, so partial results survive the cap.
- **Cleanup**: on every exit path (EXIT, INT and TERM traps), a cost line is appended and the container is removed.
- **Dry-run verified**: refuses with no approval and with 0.3; passes the gate and image check at 0.5 with `DRY_RUN=1`.

Command (needs no sudo):

```bash
APPROVED_GPU_H=0.5 <runs>/b0-cache/fixture/run_b0_cache_fixture.sh
```

## 5. Price (separate from the TP2 all-reduce item)

- **Expected:** 10-20 minutes on **one GPU**, about 0.17-0.33 GPU-hours. The time is dominated by first-time JIT: Triton (conv, recurrent, scatter/gather, copy, kpool, zeroer) and the B12x CuTe writer, decode and extend kernels. Data movement is seconds; about 60 full-tensor comparisons cost milliseconds each.
- **Ceiling: 0.5 GPU-hours** (1 GPU × 30-minute wall cap), startup, compilation and cleanup included. No automatic retry.
- Memory: about 6.3 GB (tensor plus shadow) plus kernels.
- The second GPU is idle but cannot serve meanwhile. If the operator counts machine time, the ceiling is 1.0 GPU-hour.

## 6. What the outcomes rule down

**Clean result (all parts pass).** This retires, for B0's shipped geometry and kernels:
- 32-bit truncation of byte or element offsets in every driven writer and reader at blocks ≥ 935, the last block, and the record straddling 2^31 (D8/D9);
- writes outside a group's page, or into padding, by MLA, draft, KDA, kpool, the zeroer and the copy kernels;
- padded-stride errors in the draft, KDA and tail views;
- stale-owner data surviving reuse of a block id across groups;
- replayed graphs writing or reading capture-time addresses (draft writer; MLA writer and reader).

It does **not** exclude:
- the full-layer metadata paths, which the fixture bypasses: block-table and slot-mapping construction under real scheduling (D2's class);
- `postprocess_mamba_fused_kernel` (it shares its copy body with the tested precopy);
- the index-K readers (B12x paged top-k glue; prefill `cp_gather` and DeepGEMM, compiled);
- numerical effects;
- rank 1. Its geometry is the same per tensor; only its tensor count differs.

**Failure.** The shadow diff names the block, region and owner tag. The next steps are:
1. minimise to the single kernel and address;
2. check the smallest correction against the same fixture;
3. create B1 from B0 with that one change;
4. propose an end-to-end test.

A useful cheap serving discriminator also exists: `--num-gpu-blocks-override ≤ 934` on B0 with everything else fixed. It is not proposed here.

## 7. Stopped for engineering cost (not in this price)

- **B12x paged top-k index-K reader** (decode). It needs `plan_indexer_scratch` / bind / `index_topk_fp8` glue and a set-based exact top-k reference: about half a day of CPU work, then about 5 GPU-minutes. Its offsets use Int64 page × tensor stride on a 183 MB tensor, so the priority is low.
- **Compiled prefill indexer readers** (`cp_gather_indexer_k_quant_cache`, DeepGEMM `fp8_fp4_paged_mqa_logits`). They need schedule metadata and have no source: about half a day to a day.
- **Full-layer drives** (MLA, KDA, DFlash speculator with real metadata builders). These need a model config, weights or random weights, and a forward context: substantial instrumentation, estimated 2-4 days. The cache-touching kernels are all covered above.

## 8. Secondary findings from the CPU work (no action taken)

1. The shipped B12x GLM_NEXT tests use an obsolete writer signature. All 21 writer calls in `b12x/tests/attention/test_glm_next_mla.py` (e.g. :765, 801, 850, …, 1695 in the high-page test) omit the keyword-only `plan` that both shipped definitions require (`b12x/attention/sparse_mla/api.py:133-141`; `b12x/attention/_shared/mla/kv_cache.py:860-869`). So the high-page test cannot test what it names **(inf.: not executed)**. A repaired copy of the high-page test is in `fixture/`.
2. The existing > 2^31 tests skip unless about 4.3 GB is free. They exercise a **padded** page stride (37,888 B) or 656-B GLM_NSA records, not B0's contiguous 528-B path.
3. Two block-0 behaviours are inferred harmless but not verified: `_logical_topk_to_physical_slots` does not reject the null block, and the KDA copy and gather/scatter kernels are not null-guarded.

Files: `fixture/`, `fixture.sha256`, `reachability/` (`sim_pool.py`, `out-s1*.json`, `out-s2.json`), `cpu-tests/`, `costs.jsonl` (CPU prep line, 0 GPU-hours).
