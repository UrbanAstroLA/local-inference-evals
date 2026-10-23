# P2 writer census: every call site that can reach MLA slot 0 (2026-10-23, zero GPU)

**Purpose.** P2-SLOT0-AUDIT.md concluded (S) that bytes [0, 528) of each MLA layer's KV tensor ("slot 0", the record
the b12x sparse-MLA kernels read for a -1 index) stay zero after allocation. Its open gap was "the writer list is
complete": that was a targeted trace. This note replaces it with a mechanical, bounded census.

**Scope:** stock tpurtell 0.9.1, base image `a9e985c0` (vllm under `dist-packages/`, b12x under `/opt/b12x/`). Served
configuration as in 22ao run-1 arm A: V2 runner, TP2, DCP1, GLM_NEXT 528-byte record, DFlash2 K3 with draft slot
sharing, `mamba_cache_mode=align`, prefix caching on. Arm A's observe-only guard files are not analysed; they are
stock plus hooks (P2 audit, sources).
**Labels:** S = read in source (hashes in §7), O = served log/env, I = inferred, M = produced by the census scripts.
**Resources:** zero GPU. Every `docker run` was `--rm --network none`. Temp files are in
`<census-work>/`.

---

## 1. What is bound to each MLA raw tensor

**Allocation (S).** `_glm5_next_tensor_layout` emits one `KVCacheTensor` per MLA layer i (`kv_cache_utils.py:1766-1777`)
with `shared_by = [mla_names[i]] + [g.layer_names[i] for the 7 mamba groups] + [draft.layer_names[i]]`; mamba layers
are dealt `k % 7` (`:1571-1573`; O groups 5,5,5,5,5,5,4, `server.log:241`). Each tensor is its own `torch.zeros`
(`attn_utils.py:214`; packed branch `:206-213` unused, `block_stride` = 0). Indexer tensors are separate, co-owned
only with their kpool tail (`kv_cache_utils.py:1778-1792`).

**Layer names (S + O).** MLA `language_model.model.layers.L.self_attn.attn` (`model.py:322-343`, `mla.py:125`); KDA
`language_model.model.layers.L.self_attn` (`kda.py:302-305`); draft `model.layers.(45+i).self_attn.attn`
(`qwen3_dflash.py:423,695`; drafter top-level prefix I). Owned MLA (O): rank 0 L3,7,11,15,19,23; rank 1 L27-43 step 4.
With KDA order k over the 34 non-MLA layers 0-44, tensor i's KDA co-owners are k = 7i … 7i+6, and its draft co-owner
is draft layer i for i < 5 (I: S rule plus layer order). The census treats every KDA and draft layer as a possible
co-owner, so it does not depend on this list.

| Tensor i | Rank 0 MLA | Rank 1 MLA | KDA co-owners (model layer) | Draft |
|---|---|---|---|---|
| 0 | L3 | L27 | 0, 1, 2, 4, 5, 6, 8 | 45 |
| 1 | L7 | L31 | 9, 10, 12, 13, 14, 16, 17 | 46 |
| 2 | L11 | L35 | 18, 20, 21, 22, 24, 25, 26 | 47 |
| 3 | L15 | L39 | 28, 29, 30, 32, 33, 34, 36 | 48 |
| 4 | L19 | L43 | 37, 38, 40, 41, 42, 44 | 49 |
| 5 | L23 | – | none | none |

**Block-0 byte map (S).**
- MLA row 0 is [0, 528) of the `(blocks·68, 64, 528)` view.
- KDA conv state is [0, 147,456); fp32 recurrent state is [147,456, 2,244,608) (`abstract.py:29-43`,
  `layers/mamba/mamba_utils.py:177-183,396-419`).
- Draft token 0 (FLASH_ATTN, NHD) is [0, 2048) (`flash_attn.py:139-173`; padded strided view `attn_utils.py:295-321`).

**Carriers (S, M).**
- `kv_cache_raw_tensors` and `kv_caches` dicts (`attn_utils.py:216-468`).
- `GPUModelRunner.kv_caches` (`model_runner.py:617-620`; `utils.py:612-613`).
- `layer.kv_cache` on every `static_forward_context` layer (`utils.py:619-620`); for KDA it is the tuple
  `(conv, recurrent)`.
- Pointer tables `KVBlockZeroer._meta` (`utils.py:183-233`) and `MambaSpecDecodeGPUContext.state_base_addrs`
  (`mamba_utils.py:792`).
- `_CKVPrefetchState.layer_caches` (`b12x_mla_sparse.py:536`).
- `kv_caches_dict` passed to `get_kv_connector` (`model_runner.py:628`, a no-op here).

The census found exactly 5 classes that store a carrier on `self`: `AttentionLayerBase`, `MambaBase`,
`_CKVPrefetchState`, `MambaSpecDecodeGPUContext`, `KVBlockZeroer`. No attention-metadata object, metadata builder or
speculator holds one. The speculators pass slot mappings only.

---

## 2. Census method (reproducible)

**Extraction (read-only).** `docker run --rm --network none --entrypoint /bin/sh <a9e985c0> -c 'cd
/usr/local/lib/python3.12/dist-packages && tar cf - --exclude=*.pyc --exclude=*.so --exclude=__pycache__ vllm' >
src/vllm.tar`, and the same with `cd /opt/b12x … b12x`.
- Hashes: vllm.tar `e529b077342e93e2`, b12x.tar `2f98230c61b523f0`. Spot hashes match the P2 audit.
- The image holds no vllm `csrc`. `/opt/glm53` holds only the chat template. No `.pth` hook injects code.

**Universe and scope (M).**
- 2,931 `.py` files (vllm 2,434, b12x 497).
- **Served scope: 736 files** (include rules in `census4.py`): `v1/worker/gpu/**`, `v1/worker/*.py`,
  `v1/attention/*.py`; backends `b12x_mla_sparse`, `b12x_dcp_topk`, `mla/indexer`, `sparse_utils`, `flash_attn`,
  `fa_utils`, `gdn_attn`, `mamba_attn`, `utils`, `registry`; `layers/attention/**`, `mla*`, `sparse_attn_indexer*`,
  `b12x_preparation`; mamba `abstract`, `mamba_utils`, `gdn/**`, `ops/*.py`; `warmup/**`; `models/glm5next/**`,
  `models/common/**`; `qwen3_dflash{,2}`, `qwen3`; `third_party/flash_linear_attention/**`; `forward_context`;
  `compilation/**`; `_custom_ops`; `vllm_flash_attn/*.py`; **all of b12x**.
- **Excluded by configuration:** the V1 `gpu_model_runner.py` (O: V2 runner, `server.log:61`); CPU, XPU and TPU
  workers; non-selected speculators (`spec_decode/__init__.py:11-17` picks `DFlash2Speculator`); other model families
  and backends; kv_transfer; kv_offload.
- **Seed regex** `\bkv_caches?\b|kv_cache_raw_tensors|\.kv_cache\b`: 53 files / 741 hits in scope; 133 / 976 out of
  scope. The largest excluded sources are `distributed/kv_transfer` 224, other backends 171, `v1/attention/ops` 103,
  other mamba layers 61, kimi_k3 55, V1 runner 51.

**Taint pass** (`census4.py`).
- **Seeds:** names `kv_cache`, `kv_caches`, `kv_cache_raw_tensors`, `runner_kv_caches`, `kv_caches_dict`,
  `kv_raw_tensor`, `packed_backing`; attributes `.kv_cache` and `.kv_caches`.
- **Propagation:** assignment, unpacking, `for`, `with`, walrus; subscripts and pointer arithmetic; view methods
  (`view`, `reshape`, `permute`, `narrow`, `split`, `unbind`, `as_strided`, `untyped_storage`, `data_ptr`, …);
  container insertion (`x[i]=c`, `append`, `add`, `update`, `set_`); `self.X = carrier` taints X in its class and, for
  non-generic names, globally (fixpoint: `kv_cache`, `layer_caches`, `state_base_addrs`); resolved calls pass taint
  from arguments to parameters and from return values to results.
- **Resolution:** same-file def; `from m import f [as g]`; module alias; `self.f`/`cls.f` through in-scope bases;
  `Class()` → `__init__`; `kernel[grid]()`. Any other `obj.f` resolves only if `f` has ≤ 2 defs in scope and is not a
  generic name. Otherwise the sink is kept and flagged UNRES; every UNRES callee on a served path is resolved by hand
  in §3.
- **Sinks:** every call with a carrier argument; every in-place method on a carrier (`copy_`, `fill_`, `zero_`,
  `index_put_`, `scatter_*`, `set_`, `as_strided_`, …); every subscript store or aug-assign on a carrier.
- **Result (M):** 14 fixpoint iterations; 213 functions with tainted parameters; **760 sinks** (683 argument, 74
  store, 3 in-place). `classify.py` assigns each sink to one group G1-G21 (first matching rule wins), **0 unassigned**.
- Run: `python3 census4.py && python3 classify.py` (about 5 minutes on CPU).

**b12x read-kernel store sub-census** (`b12x_stores.py`). Taint stops at the compiled CuTe launch, so this checks the
kernel bodies.
- Files: `_shared/mla/{kernel,prefill,prefill_mg,io,io_mg,decode_math,merge,smem,smem_mg}.py`.
- AST listing of the target of every global-store intrinsic from `_lib/intrinsics.py` (`st.global`, `red`, `atom`,
  `store_bf16x2`, …) and every subscript store on a non-register tensor.
- **Global targets:** `out_o` (`decode_math.py:3156-3240`), `out_lse` (`:3277-3279`), `mid_out` and `mid_lse`
  (`kernel.py:931-933,1215`), merge `dst`/`out` (`merge.py:472-481`).
- Everything else goes to shared memory or registers. **No store targets the KV operand.**

**Supplementary greps.**
- Lambdas that name carriers: 4. Three are b12x primes on local dummy caches; one is `b12x/testing`.
- String `getattr`/`hasattr` on carriers: 4. Profiling cleanup, plus the Qwen-gated warmup.
- `swap_blocks`/`copy_blocks`: only `_custom_ops` and `simple_kv_offload`, neither reached.
- No sleep mode or kv_transfer configured (O, `server.log:7`).

**Method limits.**
- The pass is name-based static analysis. It can miss a carrier that escapes through an unseeded name via closures,
  computed `setattr`, or C++ ops.
- Kernel guards (§3) were read, not proved by taint.
- FA2 and `reshape_and_cache_flash` are compiled code absent from the image.

---

## 3. Census table (every sink, by group)

Column "[0,528)?" = can the site store to bytes [0, 528) of block 0 of an MLA tensor. Classes: **A** cannot (cited
guard, region or unreachable); **A\*** cannot given a §6 allocator invariant; **B** finite only; **C** arbitrary or
non-finite; **D** undetermined; **–** not a data write.

| G | Sinks | Call sites (file:line) | Passed / view | Can write? | Index source | Writes [0,528)? Guard | Class |
|---|---|---|---|---|---|---|---|
| G1 | 23 | `b12x_mla_sparse.py:2108` → `sparse_mla/api.py:141` → `_preparation.py:296-306` → `kv_cache.py:684-774,862-887`; kernel `ConcatAndCacheGlmNextMlaKernel` `:444-588` | `kv_cache.view(uint8)` (blocks·68, 64, 528) | yes: the GLM_NEXT record writer | MLA-group slot mapping `block·4352+pos` (`block_table.py:301-318`); PAD = -1 past actual tokens (`:284-291`) and in dummy/capture runs (`:208-213`) | No. It skips slot < 0 (`kv_cache.py:504-505`). Slot 0 needs MLA table entry 0 at a written position: block 0 is popped as null (`block_pool.py:190-191`) and never re-queued (`:713,732`), and FullAttentionManager never nulls (`single_type_kv_cache_manager.py:663-674`, `:2054`). The prime runs on a dummy page with slot -1 (`b12x_mla_sparse.py:121-139`). | A |
| G2 | 23 | `b12x_mla_sparse.py:2120,2136,2147`; `backend.py:1131`; `_custom_ops.py:2788,2852`; NVFP4 writer `kv_cache.py:213-443,903-1330`; `_preparation.py:304,350` | same | yes | – | Not reached: the GLM branch returns at `b12x_mla_sparse.py:2101-2114` (O: `kv_gmem_stride=528`, `NOPE_RECORD=1`); `_WriterState.nvfp4` is False for 528-byte records (`_preparation.py:302-306`) | A (unreached) |
| G3 | 95 | `b12x_mla_sparse.py:3082,3091,3189,3236` and the b12x `sparse_mla/*`, `_shared/mla/*`, `dense_mla/*` bind/run/plan chain | kv view | no (read) | top-k physical slots; -1 → row 0 | Read-only: §2 store sub-census. Primes use fresh zero pages (`:170-215`). The 2 "stores" are Python kwargs dicts. | A |
| G4 | 9 | `b12x_mla_sparse.py:536,2607,2738-2755,3106,3120,3139,3153` | kv (gather source) | writes the workspace only | – | CKV/DCP gather branch not taken (O: info line absent; DCP1). `kv_cache` is rebound to a workspace view at `:3101-3103`. | A (unreached; read) |
| G5 | 12 | `mla_attention.py:672-691,862,1000,1315-1322,1374`; `pcp.py:26-60` | `self.kv_cache` | dispatch | – | Dispatches to G1/G3. `forward_mha` is unused (O: "no dense-MHA prefill path", `server.log:75`). PCP gathers inputs only. | – |
| G6 | 37 | `kda.py:513,530,547` → `causal_conv1d.py:712,1232`; kernel stores `:244,289,316,393,957` | `conv_state` = page bytes [0, 147,456) | **yes: covers [0,528)** | `non_spec_state_indices` / `spec_state_indices[:,0]` from the mamba block table | No. Null index is skipped (`causal_conv1d.py:143-148`, `:834-838`) and PAD is skipped (`:86`). With APC off (no `block_idx_last_scheduled_token` from kda.py, so `IS_APC_ENABLED=False` at `:750,1270`; `current_last_index=0` at `:120-125,826-827`), every store coordinate equals the guarded input coordinate. Dummy tables are zero, i.e. null (`block_table.py:165-176`). | A |
| G7 | 20 | `kda.py:613,642,691,705` → `fla/ops/kda.py:95,167` → `fused_recurrent.py:130-189`; `scatter_states.py:33,61`; `gather_initial_states` | recurrent state | yes | same indices | No, by region: offset ≥ 147,456 (`abstract.py:35-43`; shapes as §1). Also guarded at `fused_recurrent.py:119-126,176-183`. `scatter_states` is unguarded but region-excluded; its dummy write lands in block-0 recurrent bytes. | A |
| G8 | 19 | `kda.py:591,657` → `kda_replayssm_spec_decode.py:*` | recurrent + ring | yes | – | Not reached: `use_replayssm_spec` False (`kda.py:433`; O: `GLM53_REPLAYSSM_ACTIVE=0`, "baseline full-state rollback"; page = 1.0237 × (147,456 + 2,097,152), i.e. two states) | A (unreached) |
| G9 | 5 | `kda.py:733-743` | `core_attn_out` | – | – | False positive: a fresh `torch.empty` (`kda.py:367`), not the cache | – |
| G10 | 7 | `mamba_utils.py:162,369,502,792,919,968,1019` (align pre-copy and post-copy) | raw `state_base_addrs` + block id × page stride | **yes: conv copy covers [0, 147,456)** | dst = `block_table[dst_col]` (`mamba_utils.py:179-180`). Pre-copy: new running column (skips `src<0` or `src==dst`, `:498`). Post-copy: the just-completed block (`:355-357`). | No in-kernel dst ≠ 0 guard. Both run on real batches only (`mamba_hybrid.py:211-258` via `model_runner.py:1601`; post via `postprocess_sampled` `:1521`; not in `_dummy_run` `:643-777`). V2 warmup ids start at 1 (`gpu/warmup.py:284`). **If dst were 0, this would be C** (bf16 conv bytes). | A\* |
| G11 | 2 | `mamba_utils.py:1116,1121` | – | yes | – | V1 only: `preprocess_mamba` is called only from `gpu_model_runner.py:4589` | A (unreached) |
| G12 | 5 | `model_runner.py:1170` → `utils.py:637-659` `blocks[dst]=blocks[src]` over **every** storage | whole pages | **yes: whole page** | `KVCacheBlockCopy` from CoW | No. dst = `get_new_blocks` (`single_type_kv_cache_manager.py:356,1738-1741`), never block 0 (`block_pool.py:190-191,713,732`). **If dst were 0, this would be C.** | A |
| G13 | 9 | `utils.py:72-112,183-221,247-263` (`KVBlockZeroer`) | raw addresses of attention-spec layers | yes, zeros only (`tl.zeros`, `:110-111`) | `new_block_ids_to_zero` (never 0, as G12) and the warmup's block 0 (`utils.py:265-268` ← `gpu/warmup.py:293`) | Yes at warmup, with zeros. Active because `needs_kv_cache_zeroing` = has mamba (`kv_cache_interface.py:1165-1173`, `gpu_worker.py:675-678`). | B |
| G14 | 19 | `attn_utils.py:216-581`; `utils.py:589-620`; `abstract.py:41`; `model_runner.py:618,628,894`; `kv_connector_model_runner_mixin.py:253` | views, binding | no (metadata) | – | Allocation `torch.zeros` (`attn_utils.py:214`). `as_strided_` (`:544`) is stride-only and unreached: it needs a block-dim-1 co-owner (`:519`), and both b12x and FLASH_ATTN are block-dim 0 (`flash_attn.py:149`). The profiling cache is a separate allocation, torn down (`model_runner.py:856-910`). | – |
| G15 | 3 | `gpu/kv_connector.py:56,123`; `kv_transfer_utils.py:57` | `kv_caches_dict` | – | – | No-op connector: `has_kv_transfer_group()` is false (`kv_connector.py:116-121`; O: no `kv_transfer_config`) | A (unreached) |
| G16 | 4 | `attention.py:745` (draft query K/V); `qwen3_dflash.py:630` (context K/V) → `flash_attn.py:1210` → `_custom_ops.py:2641` `reshape_and_cache_flash` | draft (B, N, H, 2D) view; slot 0 = [0, 2048) | **yes: covers [0,528)** | draft SWA table. Query: `q_slot=table[q_pos//1088]·1088+q_pos%1088` (`dflash/speculator.py:563-575`, PAD past real `:628-635`). Context: real positions only. | No slot 0 found. Context is skipped in dummy runs (`speculator.py:411-412`, `qwen3_dflash.py:626-627`). Dummy queries go to slots 1-4 = [2048, 10240). Real query blocks are reserved by lookahead `num_spec+1` (`config/vllm.py:603-620`); context positions are inside the window (`single_type_kv_cache_manager.py:1059-1084`). The table read has `other=0` and a clamp (`:563-569`). **If the lookahead failed, this would be C.** | A\* |
| G17 | 12 | `flash_attn.py:993-1776`; `cp_utils.py:233,270`; `attention.py:791` | draft view | no (FA2 read; stride canonicalisation only) | – | read | A |
| G18 | 142 | `sparse_attn_indexer{,_kpool}.py`, `kpool_compress.py`, `b12x/dsa_indexer/*`, `_custom_ops.py:2963-3069` | indexer and kpool-tail tensors | yes | – | Separate allocation from every MLA tensor (`kv_cache_utils.py:1778-1792`; `attn_utils.py:214`) | A |
| G19 | 84 | `layers/mamba/gdn/{qwen,kimi,olmo}*`, `static_sink`/`cross_attention`, `warmup/{qwen,kimi_k3,jit}*`, fla GDN/replay ops, `_custom_ops.py:174,2707` | – | – | – | Not instantiated or gated. GLM KDA subclasses `gdn/base.py`, which has no cache access. Warmups: kimi is `isinstance`-gated (`kimi_k3_triton_warmup.py:21-37`), qwen is gated by model_type, jit is compile-only. | A (unreached) |
| G20 | 3 | `compilation/passes/fusion/*kvcache*` | – | – | – | Not reached: `CompilationMode.NONE`; enabled fusions are `norm_quant, act_quant` only (O, `server.log:47,46`) | A (unreached) |
| G21 | 227 | `b12x/_lib/compiler.py`, `compile_plan.py` | tensors as compile-key inputs | no | – | Cache-key and manifest code. All 51 stores go into Python dicts. Launches forward arguments to the kernel owned by the calling group. | – |

---

## 4. Init, profile, dummy, warmup, capture, align and reset paths

| Path | Block-0 writes that occur (S) | Effect on [0, 528) |
|---|---|---|
| Allocation (`attn_utils.py:214`) | whole tensor zero-filled | zero |
| Profiling minimal cache (`model_runner.py:856-910`) | separate throwaway tensors | none |
| `_dummy_run` / CUDA-graph capture (`model_runner.py:643-777,1025`) | MLA writer PAD (G1); KDA conv skips null (G6); `scatter_states` → recurrent ≥ 147,456 (G7); draft queries → [2048, 10240) (G16); no align copy (G10); no CoW (G12) | none |
| V2 warmup (`gpu/warmup.py:284-293`) | real block ids ≥ 1; zeroer warmup zeros block 0 (G13) | zero (re-zeroed) |
| `kernel_warmup` (`kernel_warmup.py:102-163`) | zeroer branch is V1-only (`:107-115`); others gated or use own buffers (G19) | none |
| b12x prime/prewarm (`b12x_mla_sparse.py:121-139,170-215,2828-2846`) | local zero pages, slot -1 | none |
| Align pre-copy / post-copy, `scatter_states` | G10 (A\*), G7 (region) | none (given §6 i) |
| Reset/clear | no state-reset writer in V2 found. Zeroing of new blocks = G13 (zeros). `reset_prefix_cache` is scheduler-only. | zero / none |

---

## 5. Classification summary (760 sinks)

| Class | Groups | Sinks |
|---|---|---|
| A, served (guard, region, read-only or disjoint) | G1, G3, G6, G7, G12, G17, G18 | 334 |
| A\*, conditional on a §6 allocator invariant | G10, G16 | 11 |
| A, not reached in the served configuration | G2, G4, G8, G11, G15, G19, G20 | 143 |
| B, finite (zeros) only | G13 | 9 |
| –, not a data write (dispatch, binding, metadata, false positive) | G5, G9, G14, G21 | 263 |
| C | none | 0 |
| D | none | 0 |

**Writers that can address [0, 528) of block 0, and what stops each:**
- G1 GLM_NEXT writer: slot ≥ 0, and block 0 is never in the MLA table (S).
- G6 KDA conv: in-kernel null skip (S).
- G10 align copies: allocator (I).
- G12 CoW copy: destination is a new block (S).
- G13 zeroer: writes zeros only.
- G16 draft writer: dummy skip (S), allocator and lookahead (I).

Every other writer is region-excluded, read-only, on a disjoint allocation, or not reached.

---

## 6. Verdict

**The census confirms the P2 audit's conclusion.**
- Over 736 served-scope files and 760 carrier sinks, no served path stores a non-zero byte into [0, 528) of block 0 of
  any MLA tensor after the `torch.zeros` allocation.
- The only block-0 [0, 528) writer that executes is the `KVBlockZeroer` warmup, and it writes zeros.
- Slot 0 therefore stays all-zero (E4M3 +0, FP32 scale 0.0) at every attention read, which is the "slot 0 zero" case
  Level 3 measured as bit-equal (S + M, with the residual below).

**What this adds to the audit.**
- Two writers missing from the targeted trace:
  - **CoW `copy_kv_cache_blocks_inplace` (G12)**, a whole-page copy across every storage. It is safe because its
    destination comes from `get_new_blocks`, which never returns block 0 (S).
  - **`KVBlockZeroer` (G13)**, which writes zeros to block 0 at warmup (B).
- Corrections:
  - The served speculator is `DFlash2Speculator` (`dflash2/speculator.py`), which subclasses the DFlash speculator
    the audit cited. Its KV paths are the inherited ones.
  - Of the KDA writers, the conv kernels are the only ones that address the slot-0 region. The recurrent writers are
    excluded by region.

**Residual inference (I), not closed by source.**
1. **Align copies (G10).** The pre-copy destination (the new running column) and the post-copy destination (the
   just-completed block) are never `null_block`. Align mode nulls only the block allocated two steps earlier
   (`single_type_kv_cache_manager.py:1555-1571`). Prefix-hit nulls (`:1453,1478,1713,1725`) sit before the running
   column. No in-kernel guard backs this.
2. **Draft queries (G16).** Every real DFlash query position falls inside an allocated draft block (lookahead
   `num_spec+1`), and context writes stay inside the sliding window. If either failed, a draft K/V write at draft
   slot 0 would put bf16 bytes into MLA slot 0.
3. **Precision of the method.** It is name-based static analysis (§2 limits). Compiled kernels outside the image (FA2,
   `reshape_and_cache_flash`) are assumed to write only their output or slot arguments.
4. **Scope.** The result covers the stock image only. Arm A's observe-only guard hooks are not included.

**Side finding (unchanged from the audit).** Safety rests on block-0 avoidance and guards in four co-owner writers:
G6, G10, G12 and G16. G10, G12 and G16 have no in-kernel `dst ≠ 0` check. A regression there would land
non-E4M3 bytes in MLA slot 0.

**Optional cheap closure of (1)-(2)** (suggestion only, not run): the P2 audit's (d)2 item. On the next authorised
serving run, read back bytes [0, 528) of the 11 tensors at each guard poll.

---

## 7. Hashes (sha256 prefix; stock `a9e985c0`; full trees in `<census-work>/src/py`)

| sha256 | File | sha256 | File |
|---|---|---|---|
| `1dd3dd2826a2cc73` | vllm/v1/worker/gpu/attn_utils.py | `101e456b28fbebd0` | vllm/v1/worker/gpu/model_runner.py |
| `5b0964afbdab59a0` | vllm/v1/worker/gpu/warmup.py | `61ceb57c8e626b89` | vllm/v1/worker/gpu/block_table.py |
| `3dcd6ad34ee1d1db` | vllm/v1/worker/utils.py | `487bb3e6696155e2` | vllm/v1/worker/mamba_utils.py |
| `82968af5e104f99f` | vllm/v1/worker/gpu/model_states/mamba_hybrid.py | `b698dcbe1815e9dc` | vllm/v1/worker/gpu/kv_connector.py |
| `35a29e38d34c6ccd` | vllm/v1/worker/gpu_worker.py | `9d0949eb408f25e1` | vllm/v1/worker/gpu/spec_decode/__init__.py |
| `482b4511561a902c` | vllm/v1/worker/gpu/spec_decode/dflash/speculator.py | `6c950a5854ef0c31` | vllm/v1/core/kv_cache_utils.py |
| `ddee56dccb220841` | vllm/v1/core/block_pool.py | `4d2a153eb8c1f3b4` | vllm/v1/core/single_type_kv_cache_manager.py |
| `f0704eb8bdbf7058` | vllm/v1/kv_cache_interface.py | `3eab4c55d500aeb4` | vllm/config/vllm.py |
| `6e74c42cf4385bb0` | vllm/v1/attention/backends/mla/b12x_mla_sparse.py | `8391ef277d824174` | vllm/v1/attention/backends/flash_attn.py |
| `944829836d9b2db8` | vllm/model_executor/layers/attention/mla_attention.py | `8e7ac8b29d002c6e` | vllm/model_executor/layers/mamba/abstract.py |
| `98a69379d675ae3a` | vllm/model_executor/layers/mamba/mamba_utils.py | `044d005cfe59fd08` | vllm/model_executor/layers/mamba/ops/causal_conv1d.py |
| `f89669f0427c1190` | vllm/model_executor/layers/mamba/ops/scatter_states.py | `371e7d733e90a714` | vllm/third_party/flash_linear_attention/ops/fused_recurrent.py |
| `8346d232ce3d0ffe` | vllm/models/glm5next/nvidia/kda.py | `b983d8bf776b1a29` | vllm/model_executor/models/qwen3_dflash.py |
| `00b239d250ae180e` | vllm/model_executor/warmup/kernel_warmup.py | `d480b9662d5ee8e0` | vllm/model_executor/warmup/kimi_k3_triton_warmup.py |
| `858e2de18cfc29b5` | b12x/attention/_shared/mla/kv_cache.py | `86ae220dbe5f680f` | b12x/attention/sparse_mla/_preparation.py |
| `105c0f738f0f3ed6` | b12x/attention/_shared/mla/decode_math.py | `36413bfca40e4aac` | b12x/attention/_shared/mla/kernel.py |
| `c2fb8882edc96e06` | b12x/attention/_shared/mla/merge.py | `52b8ad31dc95fa86` | b12x/_lib/intrinsics.py |
| `28c93ec6efa760fb` | vllm/models/glm5next/nvidia/model.py (`model.py`) | `936b06c4671d52fc` | vllm/model_executor/layers/mla.py |
| `301c76c90d5f26cd` | vllm/v1/attention/backend.py | `655b5884939b5582` | b12x/attention/sparse_mla/api.py (`api.py`) |
| `3983823640d5a35e` | vllm/model_executor/layers/attention/attention.py (`attention.py`) | `1f6ff5ca9c8f38ff` | vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py |

**Census artefacts** (in `<census-work>/`): census4.py `a718f79c9d50404d`; classify.py `5fbf5ba6dc0c1611`;
b12x_stores.py `b01c65ef9ef3dc5c`; out/census4.json `346c00ab06905f50`; out/groups.json `c14616849380f1ed`.
