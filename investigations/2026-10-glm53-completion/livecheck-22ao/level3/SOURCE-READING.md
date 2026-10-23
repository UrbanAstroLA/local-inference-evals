# Level 3 from source: does an inflated causal length L change the sparse-MLA output? (2026-10-23, zero GPU)

**Question.** On stock tpurtell 0.9.1 (`a9e985c0`, DCP1), speculative verify rows receive a per-token causal length
L = (p + 1) + r, where r (0-3) is the previous step's rejected-draft count. The backend uses
`nsa = min(L, 2048)`, then runs `_compact_dropped_selection` and `_mask_page_table_after_nsa_len`. The live check
(22ao, level 2) found that the extra in-count columns `[p+1, nsa)` hold only -1 entries. Does the attention output
differ from the output with `nsa = p + 1`?

**Evidence labels:**
- **S**: read in source. Files and hashes are in section 1.
- **H**: CPU harness. The stock Triton kernels ran under `TRITON_INTERPRET=1` in the base image with
  `--network none` and no GPU (section 5a).
- **I**: inferred. Hardware semantics or data properties not read in this source.
- **O**: observed in served logs.

**Short answer.** No source path makes the output depend on the -1 entries inside `nsa`, or on L itself, beyond
masking.
- **Decode kernel** (all uniform verify steps): **bit-identical by construction (S)**. This rests on one PTX
  property: `ex2.approx.ftz` of about -6.4e28 returns +0 (I).
- **Extend/prefill kernel** (verify rows in mixed prefill + verify steps): **bit-identical by construction (S)**,
  conditional on three hardware/data properties (I). A small single-GPU test settles them (section 5b).
- **Compaction** never diverged on any dense-regime layout (H, 12,282 cases).
- **L has no other consumer:** not the split count, kernel variant, workspace or merge order.

---

## 1. Files read and their hashes

The cited files were extracted to `recon-2026-10-22/dispatch/src/v091/`. For this note each was re-hashed inside
both images with `docker run --rm --network none` (vllm under `/usr/local/lib/python3.12/dist-packages/`, b12x
under `/opt/b12x/`, b12x 1.3.0, nvidia-cutlass-dsl 4.6.2).
- **Base image `a9e985c0`:** all 15 files are identical to the extracted copies.
- **Arm A image `local/glm53-v091-22ao-a:local`:** the 11 b12x files and kpool_compress.py / model.py are identical.
  `b12x_mla_sparse.py` (A: `490cab51…`) and `sparse_attn_indexer_kpool.py` (A: `c2fa0777…`) differ.
- **Arm A diff in `b12x_mla_sparse.py`:** only the four `local-mapguard` observe-only blocks (import, `hook_mla_len`
  at the `per_token_cache` site, and two `hook_mla_read` calls; the GLM one sits after compact and mask).
  Line numbers below are the **stock** file's.

| File (v091) | sha256 |
|---|---|
| `vllm/v1/attention/backends/mla/b12x_mla_sparse.py` | `6e74c42cf4385bb0451243098ec8b4a84005675fa776169fddb2fb230bab0157` |
| `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` | `e7935f2a9b8ecbc75410defff571d9940ad2abf511286bf0936d0c324936df39` |
| `vllm/models/glm5next/nvidia/ops/kpool_compress.py` | `7fb8eab0d71c2c576f995cc91ffc79c5055269b9bed3b094247f36bc503b241d` |
| `vllm/models/glm5next/nvidia/model.py` | `28c93ec6efa760fbf05dd6c36184c2a2ef1cba2902cc422b702c41f282c61ecb` |
| `b12x/attention/sparse_mla/api.py` | `655b5884939b55822c8d847bb541fbd3976eec345d52caeb085dc0b862d6518b` |
| `b12x/attention/sparse_mla/_preparation.py` | `86ae220dbe5f680f22c9e7726d0c3a8845b57ad492c744ef1cac9d4baa885ced` |
| `b12x/attention/sparse_mla/_scratch.py` | `d6d3398257cbd2efd140936754266633eef0502625596683c730383059a537f9` |
| `b12x/attention/_shared/mla/kernel.py` | `36413bfca40e4aaca82b14f0bd1c5b3412d32f65738c4106e6aefa3796ef6515` |
| `b12x/attention/_shared/mla/decode_math.py` | `105c0f738f0f3ed6bf6a72c2f7db3a89a19b6e0813b84c0100541ff36307e178` |
| `b12x/attention/_shared/mla/io.py` | `cdf14e155db81476ff58901d3dbedbcf5507f5068557637b937aacf3fe797aca` |
| `b12x/attention/_shared/mla/merge.py` | `c2fb8882edc96e06276834130d5efb94d4d06c7c64ec2e9badf7674d74637d02` |
| `b12x/attention/_shared/mla/prefill.py` | `1197f081741a2b1977586ccdf519df0855ee710d31d3efc56b2b1f4028d1cb4c` |
| `b12x/attention/_shared/mla/prefill_mg.py` | `73fdfab3d46fdbdf303ad98665d74ce76f42c13c99287338a1ae607d414b93eb` |
| `b12x/attention/_shared/mla/io_mg.py` | `b582b693ca62fa71de581dfca6ddb61088e4c889ddac5a62f76dce78ccaaa871` |
| `b12x/attention/_shared/mla/traits.py` | `4bcb502c64f8432d7388c915318c0258cefa26250208f6805675199242f98258` |

**Served configuration:**
- The dominant 0.9.1 decode plan line in `<data-runs>` logs (62 occurrences) is
  `vllm.b12x_mla_sparse.decode.h64.rows64.w2048.rec528: rows<=64 splits=32/32` (O). That is 64 heads, width
  2048 and 32 splits of 64 candidates, so `chunks_per_split = 1`.
- The model type is GLM_NEXT with the 528-byte record. The scale format is ARBITRARY_FP32 (`traits.py:302-355`) (S).

## 2. Call chain from the backend to the kernel (S)

**Metadata builder:** `B12xMLASparseMetadataBuilder.build`, `b12x_mla_sparse.py`.
- `:1439-1445`: with `cm.positions` present (port 01), `per_token_lens_t = cm.positions + 1`.
- `:1446-1491` (stock DCP1, no positions):
  - `context_len = seq_lens_cpu_upper_bound[req] - q_len` (`:1462`);
  - lengths `context_len+1 … context_len+q_len` (`:1477-1481`);
  - copied to `cache_seq_lens_per_token` (`:1494-1500`).
- `cache_seq_lens_per_req` = `cm.seq_lens` (`:1497-1501`).

**`forward_mqa`, DCP1 GLM branch** (logical indexer output):
- `:2938` `per_token_cache = attn_metadata.cache_seq_lens_per_token[:num_actual_toks]`
- `:3026-3032` `_logical_topk_to_physical_slots`: a logical -1, or an unmapped block, becomes physical -1
  (`:799-812`).
- `:3036-3037` `nsa_cache_seqlens.copy_(per_token_cache); nsa_cache_seqlens.clamp_max_(2048)`
- `:3040` `_compact_dropped_selection(selected_indices, nsa_cache_seqlens)` (kernel `:724-770`)
- `:3043` `_mask_page_table_after_nsa_len(selected_indices, nsa_cache_seqlens)` (kernel `:683-722`)

**Kernel choice** (`:3167-3175`):
- A uniform verify batch takes the decode kernel when `is_spec_decode` holds and `rows ≤ 64`.
- A mixed prefill + verify batch has `is_spec_decode = False` (`:1317-1323`) and takes the extend kernel.

**Decode path:**
- **Binding** (`:3176-3197`): `cache_lengths = cache_seq_lens_per_token` (verify rows) and
  `selected_lengths = nsa_cache_seqlens`, passed to `b12x.attention.sparse_mla.bind`.
- **Binding storage** (`sparse_mla/api.py:61-116`, `_scratch.py:370-438`): `cache_seqlens_int32` and
  `nsa_cache_seqlens_int32` are stored in the binding.
- **Run** (`api.py:119-129`, then `_preparation.py:211-231`): `run_prepared_unified_decode(...,
  swa_topk_lengths=binding.nsa_cache_seqlens_int32, ...)`. **`cache_seqlens_int32` is not passed.**
- **Kernel** (`kernel.py:2725-3275`): `run_unified_decode` launches the grid kernel `UnifiedDecodeKernel` (body
  `:880-1987`), grid `(rows, h_blocks, num_splits)`. Then come the split merge (`merge.py:339-484`, launched at
  `kernel.py:3238-3252`) and, only with `return_lse`, the final-LSE kernel.

**Extend path:**
- `b12x_mla_sparse.py:3218-3245` → `_preparation.py:232-259` → `prefill.run_unified_prefill` (`prefill.py:131-561`).
- For GLM, ARBITRARY_FP32, topk 2048, it routes to `_run_partitioned_mg` (`:437-442`), then to
  `prefill_mg.run_unified_prefill_mg` (`:4300…`), kernel body `_body` (`:2377…`).
- `topk_length = nsa` is the only length passed. `cache_lengths` is again not passed (`_preparation.py:234-259`).

**Selection producer** (indexer, DCP1, for the tail-column question; `sparse_attn_indexer_kpool.py`):
- `:461`: `topk_indices_buffer[: hidden_states.shape[0]] = -1` runs on every call. This resets all 2,048 columns,
  including column 2047, which no expansion writes.
- `:1008-1010` decode, `:643-645` prefill: `expand_pools_and_append_tail(pool_ids[:, :511], seq, 4)`
  (`kpool_compress.py:925-998`).
- That produces 2,047 columns:
  - **History** `[0, 2044)`: pool g gives `pid*4+o`, valid iff `0 ≤ pid < seq//4`.
  - **Tail** `[2044, 2044 + seq%4)`: positions `seq//4*4 …`.
  - Everything else is -1.
- The tail columns therefore depend on the indexer's own `seq`, not on L.

## 3. How an invalid (-1) entry inside `nsa` is handled

### 3a. Decode kernel (S)

**Length.** It is loaded per token and clamped (`kernel.py:939-952`):
```
section_len = Int32(topk_length[token_idx]); clamp to [0, topk]
```

**Which split CTAs work** (`:1166-1225`, `chunks_per_split = 1`):
```
main_valid_chunks = (section_len + 63) // 64 ; … active_chunks = main_chunk_count + …
if active_chunks == 0: mid_lse[token, head, split] = -inf ; exit
```
So nsa decides how many 64-wide chunk CTAs do work. This is the only geometric effect of nsa.

**Producer gather** (`io.py:615-736`, packed GLM_NEXT IO):
- An entry with `cand >= g_end` (with `g_end = min(start+64, section_len)`) is set to -1 (`:670-673`).
- A raw -1 from the table also stays -1.
- Either way the entry copies the record of **physical slot 0**: `io.py:580` reads "a negative index copies row 0
  so the byte count is always met".
- -1 is written to the smem validity buffer (`:732-733`).
- An in-count -1 and a past-the-length entry therefore produce byte-identical smem state.

**S3 mask** (`decode_math.py:1280-1325`):
```
if abs_c0 >= split_cand_end or idx0 < Int32(0):  qk[0] = qk[2] = _QK_MASK   # -1e30
```
- The value is finite: -1e30 × sm_scale·log2e, about -6.4e28.
- The two masking reasons (past the length, or index < 0) give the same value.
- **There is no V masking for GLM.** The V-zeroing block runs only for `scale_format == 0` (DSV4) (`kernel.py:1566`).
  Masked candidates therefore keep slot 0's V bytes in smem and rely on P = 0.

**S4 online softmax** (`decode_math.py:1685-1833`):
- **Warp with at least one valid candidate:** a masked p is `ex2(-6.4e28 - local_max) = 0`.
- **All-masked warp inside a chunk that has valid entries:**
  - its local p = 1 (spurious);
  - its warp sum is weighted by `ex2(wmax - block_max) = 0` (`:1792-1797`);
  - its `w_pre = p · warp_rescale = 1 · ex2(-6.4e28 - gmax) = 0` (`:1810`, kernel `:1823-1828`).
  - The denominator contribution is exactly 0, and P = 0 enters S5/S6.
- **All-masked chunk (= split here), the only new case under inflation.** It is reached only when nsa crosses a
  multiple of 64:
  - `block_local_max = -6.4e28`;
  - `alpha = 0` because `global_max (-1e30) > -1e29` is false (`:1804`);
  - `global_sum = 64`, `warp_rescale = 1`, so W = 1 on 64 copies of the slot-0 record.
  - S7 (`decode_math.py:3111-3116, 3253-3258`) writes `O = acc/64` (slot-0 V, or NaN if slot 0 were non-finite)
    and **lse = log2(64) + (-6.4e28) ≈ -6.4e28**. This is finite; it is not the -inf sentinel.

**Merge** (`merge.py:390-417, 279-336`):
```
m = fmax over lse_j ;  if lse_j > -inf: w_j = ex2(lse_j - m)  ;  d += w_j
…  if w != Float32(0.0): acc += w * partial
```
- For the all-masked split, `m` is a real row LSE, so `w_j = ex2(≈-6.4e28 - m) = +0` (I: PTX
  `ex2.approx.ftz.f32`, `decode_math.py:98-115`).
- `d + 0.0 = d` bit-for-bit.
- The `w != 0` guard **skips the split's partial entirely**, so even a NaN partial cannot enter.
- Slots are fixed lanes and the number of slots is fixed (32), so the summation order is unchanged.

### 3b. Extend/prefill kernel (MG), used for verify rows in mixed steps (S)

- **Tile count:** `section_len = topk_length[token]` clamped; `actual_tiles = ceil(section_len/64)`
  (`prefill_mg.py:2540-2546`). One CTA walks all tiles in a single online softmax, with no split-K.
- **Gather:** `io_issue_gather_glm_mg` (`io_mg.py:144-283`). `cand >= g_end` gives -1, and a negative index is
  clamped to 0 (`:268-272`). This is the same slot-0 substitution as decode.
- **Mask:** `s3_mask_and_scale_global` (`prefill_mg.py:125-161`). Index < 0, or a position past the end, gives
  `-1.0e30`.
- **Softmax** (`s4_online_softmax_mg2`, `:1519-1756`):
  - `new_m = fmax(old_m, bmax)`, `alpha = ex2(old_m - new_m)`; accumulators are rescaled only
    `if alpha < 1.0` (`:1611, :1641-1682`);
  - `p = ex2(qk - new_m)` (`:1708-1720`).
- **Extra trailing tile with every entry masked:**
  - `new_m = old_m`, so `alpha = ex2(0)`. The source states this is exactly 1 (`:1640-1644` comment, I);
  - p = 0 for all 64 candidates, so `gsum += 0`;
  - XV (`decode_math.s6_xv_nope`, GLM ARBITRARY_FP32 two-pass W, `:2038-…`): W quantizes 0 to e4m3 0, so the MMA
    adds 0·V(slot 0).
- **No GLM V masking** here either. The "Zero masked V rows" block (`:1886`) is the DSV4 function only.

## 4. Every place L or nsa influences something other than masking

| # | Site | Depends on L / nsa? | Effect |
|---|---|---|---|
| 1 | `cache_lengths` (= L for verify rows) passed to the decode bind (`b12x_mla_sparse.py:3176-3195`) | L | Shape-checked only (`_scratch.py:385-390`) and stored. **Never passed to the decode or prefill kernels** (`_preparation.py:216-259`). No effect (S). |
| 2 | Split count / chunks | no | Fixed at plan prime: vLLM pins `B12X_MLA_SM120_NUM_SPLITS=32` around `state.prime` (`b12x_mla_sparse.py:203-215`, cap `:1774`), and `prepared.num_splits` is reused at run (`kernel.py:3046-3049`). Served: `splits=32/32` (O). Split boundaries are a function of topk width only (`kernel.py:295-365`). |
| 3 | Kernel variant (`per_token_len`) | no | On CUDA `_length_tensor` always returns `(lt, False)` (`kernel.py:3013-3014`), so the per-token variant always runs. No host readback of lengths, so the PTX does not depend on the length values (S). |
| 4 | Workspace / mid_out / mid_lse sizing | no | From Caps (`max_width`, `max_chunks_per_row`, rows). Merge `num_chunks = num_splits` (`kernel.py:3238-3245`) (S). |
| 5 | Active split CTAs (decode) | nsa | Crossing a 64 boundary activates one more split. That split is all-masked and its partial gets merge weight +0 and is skipped (section 3a). More work; no change in arithmetic (S + I for `ex2` underflow). |
| 6 | Speculative first-chunk copies (decode) | nsa | `spec_valid_chunks` (`kernel.py:1142-1164`) decides only whether slot-0/selected records are prefetched. Data path only (S). |
| 7 | Prefill tile count | nsa | One extra all-masked trailing tile when crossing a 64 boundary (section 3b). |
| 8 | `_compact_dropped_selection` trigger | nsa | Fires iff some valid id sits at a column ≥ nsa (`b12x_mla_sparse.py:738-747`). When it fires, its output (every valid id in the row stable-compacted to the front, `nsa := count`) **does not depend on the input nsa**. See below. |
| 9 | `_mask_page_table_after_nsa_len` | nsa | Writes -1 at columns ≥ nsa. Writing -1 over holes is a no-op. |
| 10 | Other consumers of `cache_seq_lens_per_token` | — | None in vllm (`grep` over the tree: only this backend) (S). DCP LSE output is not used at DCP1 (`need_to_return_lse_for_decode = dcp>1`, `:1719`). |

**Compaction and tail columns (S + H).**

*Fires in both cases:*
- In the dense regime (`p+1 ≤ 2044`) with a tail present (`seq % 4 ≠ 0`), the tail ids at 2044…2046 lie beyond
  both `p+1` and `p+1+r`.
- Compaction therefore fires in both cases and yields the identical row and `nsa = count`.
- On these rows the inflated L is **erased** before the kernel sees it.

*Fires in neither case:*
- With no tail (`seq % 4 = 0`), or with `p+1 ∈ [2045, 2047]` (tail inside both lengths), compaction fires in neither
  case.
- The extra columns are -1, including column 2047, which is reset at `indexer_kpool.py:461`.

*Clamped regime:* for `p+1 ≥ 2048`, nsa is 2048 for both lengths.

*Only possible divergence from source:* a valid id at a column in `[p+1, nsa_infl)` with none at or beyond
`nsa_infl`.
- Then compaction would fire for `p+1` but not for the inflated length.
- The attended **set** would be the same, because compaction keeps every valid id. The column order, and so the
  chunk/split partition, would differ.
- That makes it "numerically different, mathematically equal".
- The 22ao level-2 record (holes only in `[p+1, nsa)`, after compaction) excludes this on the measured rows.
  The front-packed pool layout excludes it by construction (H, section 5a).

## 5. Verdicts per effect

| Effect | Verdict | Label |
|---|---|---|
| E1. Decode: extra -1 columns inside the same 64-chunk (no boundary crossed) | **bit-identical by construction**: identical table, identical smem (slot-0 substitute for both "past length" and "-1"), identical `qk = _QK_MASK`, everything downstream identical | S |
| E2. Decode: nsa crosses a 64 boundary, so one extra all-masked split | **bit-identical by construction**: split lse ≈ -6.4e28 gives merge weight +0; the `w != 0` guard drops its partial (NaN-safe); `d + 0 = d`; slot order fixed | S (I: `ex2.approx.ftz(≈-6.4e28 - m) = +0`, standard PTX FTZ underflow) |
| E3. Extend/prefill (mixed steps): extra all-masked trailing tile | **bit-identical by construction, conditional on**: (a) `ex2.approx(0) == 1.0` exactly (the code relies on this, `prefill_mg.py:1640-1644`); (b) a tensor-core MMA with A = 0 adds exactly 0; (c) the slot-0 record is finite | S (control flow) + I (a-c). (c): the KV cache is `torch.zeros`-initialized (`gpu_model_runner.py:7605-7616`; **erratum 22ar:** the served runner is V2, allocation at `gpu/attn_utils.py:214`, also `torch.zeros`) and block 0 is the null block (`block_pool.py:190`). Also, the same slot-0 × P = 0 product already occurs in the partial last tile of almost every baseline row, so a non-finite slot 0 would already corrupt the baseline. |
| E4. `cache_lengths` = L reaching the kernels | **no effect**: never read by either kernel | S |
| E5. Split count, kernel variant, workspace, merge slot count / reduction order | **independent of L and nsa** | S (+O `splits=32/32`) |
| E6. `_compact_dropped_selection` / tail columns 2044-2046 | **bit-identical** on every dense-regime layout tested: when the tail is present it fires in both cases with identical output (inflation erased); otherwise in neither, with -1 holes. Divergence would need a valid id in `[p+1, nsa_infl)` and none beyond, which is absent from the indexer's layout and from the 22ao level-2 data. In that hypothetical the result would be "numerically different, mathematically equal" (same set, different chunk order). | S + H |
| E7. Mathematically different outcome | **none found** in source | S |
| E8. Undetermined from source | the three hardware/data facts under E2/E3 (`ex2` underflow and `ex2(0) = 1`, zero-A MMA, slot-0 finiteness). They are not verifiable without the GPU. | I |

### 5a. CPU harness already run for E6 (H)

- **Run:** base image `a9e985c0`, `--network none`, `TRITON_INTERPRET=1`, no GPU. The script was piped on stdin and
  nothing was written on the host.
- **Code under test:** the stock `_compact_dropped_selection_kernel`, `_mask_page_table_after_nsa_len_kernel` and
  `_expand_pools_and_append_tail_kernel`, AST-extracted from the installed files (hash prefixes `6e74c42c…`,
  `7fb8eab0…` checked in-run).
- **Inputs:**
  - every indexer seq `p+1` from 1 to 2047 × r ∈ {1, 2, 3}, with `nsa = min(p+1+r, 2048)`;
  - pool order sorted, and randomly shuffled within the front-packed prefix;
  - buffer column 2047 reset to -1 as at `:461`.
- **Comparisons against r = 0:**
  - the post-compaction/mask table;
  - the per-64-chunk attended set (columns < nsa with id ≥ 0), modelling the decode kernel's validity rule.
- **Result: 12,282 cases, 0 mismatches.**
  - Post-compaction nsa was equal in 9,198 cases: tail present, so compaction reset nsa to the count.
  - It differed in 3,084 cases (holes only). In 186 of those, nsa crossed a 64 boundary (case E2/E3).
  - Run time 530 s.
- **Not covered:**
  - the stock indexer's own tail anchoring (D2) and non-front-packed pool outputs. The level-2 data covers these
    for the served rows;
  - the kernel arithmetic itself.

### 5b. Smallest test that settles E8 (single GPU, synthetic tensors, no model load)

**There is no CPU path for the kernels:**
- `sparse_mla.is_supported` requires SM120/121 (`api.py:163-165`);
- compile needs `torch.cuda.get_device_properties`;
- the pure-torch `_shared/mla/reference.py:148-261` defines the semantics (an entry is valid iff `col < count` and
  `idx ≥ 0`) but is not bit-faithful to the kernel. It can serve as a sanity reference only.

**Test design:**
- **Image:** the base `a9e985c0`, `docker run --rm --network none --gpus device=0`. No model weights, no server.
- **Plans:** build the two plans exactly as `_make_plan` does (`b12x_mla_sparse.py:1839-1880`): `Caps(model_type=GLM_NEXT,
  head_dim=512, v_head_dim=512, cache_record_bytes=528, page_size=64, max_width=2048, max_chunks_per_row=32,
  num_q_heads=64, kv_dtype=uint8, dtype=bf16, lse_scale="natural", return_lse=False)`.
  - decode: `max_q_rows=64`, `max_batch=64`, primed with `B12X_MLA_SM120_NUM_SPLITS=32`;
  - extend: rows 64, `max_batch=8`;
  - assert the decode launch reports splits 32/32.
- **KV:**
  - a 64-page × 64-slot uint8 cache, written through the stock GLM_NEXT cache writer
    (`sparse_mla.plan_cache_writer` / `concat_and_cache_glm_next_mla`) from random bf16 latents, so records are
    well-formed;
  - slot 0 left zero (the null block). Variant: slot 0 filled with real latents.
- **Rows (64 query rows, random bf16 q):**
  - row t has `p+1` valid distinct slots (≠ 0), front-packed in columns `[0, p+1)`, and -1 elsewhere;
  - `p+1` spans the 64-boundaries and edges: {1, 2, 63, 64, 65, 127, 128, 129, 191, 192, 1023, 1024, 1025, 1983,
    1984, 1985, 2043, 2044, 2045, 2046, 2047}, plus random values.
- **Two settings per row:** `nsa_A = p+1` and `nsa_B = min(p+1+r, 2048)` for r = 1, 2, 3 (separate runs). For E6 in
  situ, also run the stock compact and mask kernels on GPU first.
- **Compare:**
  - `torch.equal` on the bf16 output (decode and extend), with the table unchanged;
  - report max |Δ| and the row count of any inequality;
  - extend path: also compare the returned base-2 LSE.
- **Positive control:** in the same session, plant one live non-causal slot (a valid id) at column `p+1` for B only.
  The output must differ, which shows the comparison can see a one-entry change.
- **Optional probe** (diagnostic, not a pass criterion): set slot 0 to NaN e4m3 bytes (0x7F).
  - Expected (I): decode A and B are both NaN on rows whose last chunk is partial. E2's extra split is still dropped.
  - Expected (I): extend is NaN on partial last tiles in both A and B.
  - This makes the baseline hazard visible and confirms it is not L-specific.
- **Expected run time:**
  - about 3-6 min wall for the CuTe compiles of two plans (cached afterwards);
  - under 1 s per kernel comparison;
  - total about 10 min of one GPU, well under 0.25 GPU-hours.
- **Authorization:** needs operator authorization (GPU).

**Expected outcome from source:** bit-equal for every row and every r, in both kernels; and the positive control
differs.

## 6. Limits
- Source-level conclusions cover the stock 0.9.1 files hashed above and the GLM_NEXT / DCP1 / topk 2048 route only.
  The DCP2 (0.7.0) and CKV-gather branches were not analysed.
- **Port 01 (C):** lengths become `positions + 1`, which removes E1, E2 and E3 at the source. The kernels and
  compaction are the same files.
- E6's harness models the indexer layout for the DCP1 tail rule with a correct indexer seq. Stock-specific index
  content (D2) is outside this question and is covered only by the 22ao level-2 measurement.
