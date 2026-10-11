# B12x sparse-MLA / indexer audit: OLD fe054789 (v070) vs NEW 7fcc094e (v091)

Paths below are relative to `kdcp/{v070,v091}/b12x/`. "O:" = OLD, "N:" = NEW.
This was a CPU-only source read. The CuTe DSL package is not installed on the host, so DSL lowering (for example whether `ptr + Int64` stays 64-bit) could not be checked and is marked INFERRED.

## 0. Headline findings

1. **Int32 overflow: not found.** In both versions, every decode, prefill and writer KV byte address is computed in Int64 before any multiply. The selected slot itself is Int32, and 4.71M fits in Int32. NEW also has live GPU tests above 2^31 bytes; they skip when free VRAM is short (see 4).
2. **NEW extend ignores a head-major output (real bug, it explains the cos 0.76).**
   - Where: `_SparseMlaState.prime` for extend (N: `b12x/attention/sparse_mla/_preparation.py:191-208`) calls `run_unified_prefill` without `output=`.
   - Effect: prefill allocates a contiguous token-major buffer (N: `_shared/mla/prefill.py:293-296`). The kernel bakes `output_stride` into the compiled program as Python ints (N: `prefill_mg.py:2146-2148`, `3990`).
   - At run time the prepared program is replayed by `run_compiled` (N: `prefill_mg.py:4108-4109`). It receives the head-major `scratch.output_buffer` pointer (N: `_preparation.py:241-245`) but writes with token-major strides.
   - OLD compiled per call with the real output tensor, and its strides are in the compile key (O: `prefill_mg.py:3836`, tensor_key "output" `~3903`), so OLD honored head-major on extend.
   - NEW decode is fine. Its prime uses the plan's own scratch (head-major `output_buffer`, `mid_out` strides), and the merge uses dynamic layouts (N: `merge.py:154-177`). See A.
3. **Top-k output is unordered in both versions (radix select), never position-ascending or score-sorted.**
   - The dropped column 511 is an element of the lowest selected score bucket, placed by atomic arrival order.
   - OLD vs NEW changes the merge arm and the bucket resolution, so *which* low-score pool is dropped differs. The selected 512-set is exact in both (modulo exact ties). See 3.
4. **The attention math in NEW GLM_NEXT 528 B equals OLD GLM_NSA 656 B with a zero RoPE tail, up to fp32 summation order.**
   - No masking or normalization semantics changed in decode or merge.
   - The changes found are FP reorderings: two-pass PV interleave, permuted K axis, two-pass merge.

## 1. Code paths

- **OLD decode/extend:** `sparse_mla/api.py:17-21` → `_shared/mla/api.py` `sparse_mla_decode_forward`/`_run_sparse_mla` (`:776-822`) → `kernel.py:run_unified_decode` (`:2734`), or `_run_sm120_prefill` (`api.py:~865`) → `prefill.py` → `prefill_mg.py`. OLD resolves splits per call (`forced_num_splits`).
- **NEW:** `sparse_mla/api.py:50-129` plan/bind/run → `_preparation.py:_SparseMlaState.prime/run` (`:154-259`).
  - Decode: `kernel.py:prepare_unified_decode_launch` (`:2557`), `compile_unified_decode_launch` (`:2684`), `run_prepared_unified_decode` (`:2720`) → `run_unified_decode` (`:2725`).
  - Extend: `prefill.py:run_unified_prefill`.
- **Split planning:** byte-identical, O: `kernel.py:140-357` = N: `148-365`. NEW calls prepare with `forced_num_splits=None`, so the env value `B12X_MLA_SM120_NUM_SPLITS` is read once at prime (N: `_preparation.py:162-176`). With width W, splits = ceil(W/64) means one chunk per split. For W=2051 that is 33 chunks; the vLLM cap is `_cdiv(topk,64)`.
- **Lengths:**
  - Both versions mask only by `nsa_cache_seqlens`/`selected_lengths`, passed as per-row `topk_length` (O: `api.py:449,507`; N: `_preparation.py:182,223`).
  - `cache_seqlens` is only validated (batch size) and never used by the decode kernel.
  - CUDA always takes the per-token path (O/N `_length_tensor`: O `kernel.py:2991`, N `:2995`).

## 1b. Per-topic semantics (OLD → NEW)

**Invalid (-1) indices.**
- Producer: idx<0 is clamped to slot 0 for the payload copy, and the raw idx is staged for validity.
  - O: `io.py:189-196`, `226-244`, `318-325`.
  - N: same in `io.py:330/377`; the GLM_NEXT packed producer is `io.py:563-610` and `615-...`.
- S3 masks qk to `-1e30` (finite) when `abs_c >= split_end` or `idx<0` (N: `decode_math.py:1280-1325`; O identical; `_QK_MASK` O `:267`, N `:273`).
- Unchanged.

**Empty split, judged by length.**
- `active_chunks==0` writes `mid_lse=-inf` and exits (O: `kernel.py:1512`; N: `kernel.py:1213-1225`).
- NEW additionally arrives and waits on the speculative stage before exit.

**Active split whose in-length indices are all -1 (both versions, unchanged).**
- The LSE is finite (≈ -6e28 + log2(64)), not -inf. O is the mean of V(slot 0), because every masked p = exp2(0) = 1 (S4 N: `decode_math.py:1712-1834`; S7 LSE N: `3247-3252`).
- The merge weight underflows to 0 when any real split exists.
- If a whole row has only -1 within its length, the output is the slot-0 V average instead of 0. This happens in both versions, so it is not a regression.

**Merge.**
- **OLD:** online pairwise merge. It skips `lse==-inf` and outputs 0 when all splits are -inf (O: `merge.py:259-330`).
- **NEW:** two-pass rewrite (commit f5394625).
  - Warp 0 computes the max, then per-split weights `exp2(lse-m)/d`. A -inf split, or a slot beyond `num_chunks`, gets weight 0. `d==0` gives scale 0 and output 0 (N: `merge.py:339-484`, `404`, `411`).
  - Zero-weight slots are skipped (`315/331`).
- Both: fp32 accumulation of bf16 partials, base-2 LSE, `exp2.approx.ftz`. No NaN path (no `-inf - -inf`). The difference is legitimate FP only (reduction order).

**Base-2 vs natural log.**
- Decode final LSE: natural via `_final_lse_from_split_workspace` (O: `kernel.py` tail; N: `new run_unified_decode` 534-551). With DCP=1 it is not requested.
- Extend returns base-2; the NEW caller multiplies by ln2.
- Unchanged.

**Q quantization.**
- Per (head, 128-tile): absmax with eps 1e-4, scale = pow2ceil(absmax/448), E4M3 satfinite.
- NEW computes `inv_scale` from the UE8M0 exponent instead of `1/rounded`, which is exact for powers of two (dm diff `@@ -400`). The vector Q stager `s0_quantize_q_vec` is numerically identical (N: `decode_math.py:559-701`).

**KV dequant.**
- GLM ARBITRARY_FP32: 4 inline fp32 scales at record bytes 512..527, one per 128 dims, read in PV as `_vsc` (N: `decode_math.py:2138-2154`; O same).
- QK uses `s1_qk_nope_block_scaled` with the same scales.
- RoPE (S2) loops `d_rope//16`, which is zero iterations for GLM_NEXT (N: `decode_math.py:1155-1200`).

**PV.**
- GLM two-pass W (E4M3 HIGH plus LOW residual): OLD did HIGH then LOW in separate passes; NEW stages both behind one barrier and interleaves HIGH/LOW MMAs per kstep (N: `decode_math.py:2324-2421`). GLM_NEXT gets 4 W buffers, and smem allocates 4 (N: `smem.py:303-311`).
- Permuted K axis: `_w_perm_col` (N: `3389`) is the exact inverse of `_d2_load_b_fp8_perm` (N: `3316`). I checked every W staging site in the decode arms; all of them use `wcol`.
- Permutation is a reordering of a sum. Legitimate FP difference only.

**656 B vs 528 B records.**
- Same first 528 bytes in the same layout (O/N `io.py:69-73` / `74-77`). The 656 B record adds a 128 B BF16 RoPE tail.
- OLD generic arm, used for 32 heads/rank (TP2): `io_issue_gather` copies NoPE+scales (528) and RoPE (128) separately.
- NEW GLM_NEXT uses `packed_glm_next` (N: `kernel.py:455-479`), one 528 B bulk copy per candidate.
- With a zero RoPE tail, the OLD extra term is Q_rope·0 = 0, so the attention is the same function.
- **Writers.**
  - NEW GLM_NEXT writer (N: `kv_cache.py:487-586`): fp32 scale = amax/448, with amax==0 giving scale 1.0; quantizes `v * rcp.approx.ftz(scale)` with satfinite E4M3.
  - OLD 656 B writer: vLLM's `ops.concat_and_cache_mla` (fp8_ds_mla). This is CUDA C++ that is not in these trees; b12x's reference packer uses amax/448, zero gives 1, divide, clamp (`reference.py:64-66`).
  - INFERRED: the vLLM writer is the same formula with exact division. Then the stored bytes differ only at rare rounding ties (rcp.approx vs div). This is FP-level, not an invariant change.

## 2. All-invalid split and LSE (task 2)

- **Length-empty split:** in both versions, `mid_lse=-inf`, no output written, skipped by the merge. NEW zero-weights it, OLD skips it; both are NaN-safe.
- **Index-invalid but in-length split:** finite LSE, weight 0 after the merge (see 1b).
- **No behavioral change found** between OLD and NEW.
- **NEW-only speculative first-chunk copy** (`kernel.py:1121-1163`, gated by `pre_ci < spec_valid_chunks` at `:1151`): copies records for `topk_row[pc]` before the length is applied. Validity is still masked by `g_end` (`io.py` `io_issue_gather_packed` `cand>=g_end → -1`). INFERRED: stale indices beyond the length only cost bandwidth, and garbage out-of-pool ids would fault rather than corrupt.

## 3. Top-k ordering (tasks 3 and B)

**vLLM (both trees, kpool decode)** (`sparse_attn_indexer_kpool.py:908-1009`, identical in O and N):
- DCP1 path: `_run_b12x_paged_topk` writes the pool ids.
- DCP2 path: adds `_merge_b12x_dcp_topk` (`sparse_attn_indexer.py:1139-1215`): local→global ids, all-gather of 2×512 candidates, `run_row_topk` over 1024 candidates, then gather by position.
- Then `pool_ids[:, :select_k-1]`, which drops column 511.

**b12x route** (identical in O and N, `scratch.py` O `840` / N `875`):
- `resolve_fused_indexer_path` (N: `fused_indexer.py:901-930`, unchanged) selects the fused route for topk 512, 32 heads, SM120/121 and rows ≤ 16.
- More than 16 rows (spec-verify) takes the tiled route: `run_tiled_topk` + `run_row_topk`.

**All selection kernels are radix select (unordered).**
- Winners are written at atomic-counter positions in arrival order; pivot ties, or the last-refined bucket, fill the tail.
- Tests compare as sets / after sorting (N: `tests/attention/test_fused_indexer.py:306-350`; `tiled_topk.py:1982` reference uses `sorted=False`).
- **Nothing is position-ascending.**

**OLD fused path, multi-CTA.**
- `merge_threshold` = crossover ≈ 22000 + 117·ctas_pg pools (O: `fused_indexer.py:593-621`). Below it the serial last-CTA arm runs `_fused_radix_select`: coarse 8-bit key bucket, then byte refine. The tail holds the lowest refined bucket in arrival order.
- Above it the cooperative arm runs 4 full byte rounds to an exact pivot (O: `2123-2262`). Positions [0, count_gt) are winners and the tail is elements exactly equal to the 512th score. So the dropped column is the true minimum (up to exact ties).

**NEW fused path.**
- Plan sets `ctas_per_group = min(width, sms/rows)` and threshold 0 (N: `scratch.py:1010-1031`; `fused_indexer.py:964-1012`), so every multi-CTA row takes the cooperative arm.
- That arm was rewritten (commit 02a19e65): 12/12/8-bit rounds with early exit when `bin_count == remaining_k` (N: `2577-2725`, exits at `2608`/`2640`).
- On early exit, [winners, 512) holds the whole lowest prefix bucket in CTA/thread arrival order. Column 511 is then an arbitrary member of that bucket, not necessarily the minimum.
  - A 12-bit bucket is about 1/8 octave wide, so this matters.
  - A 24-bit bucket is negligible.
- `total <= 512` (short rows): arrival order with -1 padding at the end (`2577-2598`).

**DCP2 `run_row_topk`** (O and N): radix over the [rank0 512 | rank1 512] candidates. The tail is the lowest bucket. INFERRED: the order within a pass is roughly ascending candidate index, so there is some bias toward rank-1 / later candidates in column 511.

**Conclusion.**
- The dropped pool is "one of the lowest-scoring" in every version, never deliberately the newest.
- OLD (serial arm below the crossover, or exact-pivot coop above it) and NEW (coop with bucket early exit) choose differently among low-score pools.
- The selected set is exact. Only the identity of the one discarded pool changes. This is a legitimate implementation difference, not an invariant violation, but it does change what is attended (one pool = 4 tokens).

## 4. Int32 overflow (task 4, high priority)

**Decode gather.**
- OLD `io.py:194-196`, `240-244`: `block_idx = idx // pbs` (Int32), then `Int64(block_idx)*stride_kv_block(Int64) + Int64(local)*_IOS(Int64)`.
- NEW `io.py:330/377` (generic) and `io.py:590-599` (packed, used by GLM_NEXT): the same pattern, with `Int64(record_bytes)`.
- `stride_kv_block` is a host Int64 from `_cache_block_stride_bytes` (O `kernel.py:2320`; N `2037`), using the real `stride(0)` for padded or pooled-tail pages.
- Address = `get_ptr_as_int64(tensor, off)` = `iterator + off` → `ptrtoint i64` (O `_lib/intrinsics.py:1501`; N `:1528`, identical). INFERRED: CuTe pointer + Int64 lowers to a 64-bit GEP; no source was available to confirm.
- The flat `kv_flat` u8 view (`_cache_base_tensor`, O `2299` / N `2016`) is >2^31 elements in NEW (2.49–2.8 GB). It is passed with a static layout, but addressing never uses that layout.

**Prefill gather:** `io_mg.py:96-132`, `243-274` (NEW) are all Int64.

**NEW GLM_NEXT writer:** `kv_cache.py:197-210`, with slot Int64 (`slot_mapping` loaded as Int64, `:510`) and Int64 strides; bounds check `0 <= slot < pages*page_size`.

**Indexer K cache:** `Int64(page_id)*page_stride` (N `fused_indexer.py:237`, `1991`; `kernel.py:3292`). Pool row counts are about 1.2M, far from the limit.

**Remaining Int32 values:** the slot index, plus `pid*64+slot` global ids in the indexer. These are fine below 2.1e9 tokens.

**Tests.**
- NEW `tests/attention/test_glm_next_mla.py:1646` (and OLD `:962`) is a live >2^31-byte GLM_NEXT writer plus prepared decode test.
- NEW `test_attention_mla_sm120.py:1742` (#312) covers GLM_NSA rescale-late-maximum with high page ids beyond 2^31.
- Host-mirror checks: N `test_glm_next_mla.py:674`, O `:213`.
- The live tests **skip** if free VRAM is below roughly required + 2 GB, so it is unknown whether they ran upstream.

## 4b. Tests by topic

| Topic | OLD | NEW |
|---|---|---|
| GLM_NEXT vs record oracle | `test_glm_next_mla.py:485` (decode), `:702`/`:824` (prefill) | `:1004`, `:1306`, `:1524` |
| GLM_NEXT vs GLM_NSA equivalence | none | none |
| Merge with -inf / empty splits | `test_attention_mla_merge.py:125-219` | `test_attention_mla_merge.py:125-219`; adds `:325` (capacity) and `:380` |
| Padded -1 rows and short lengths under one wide plan | none | `test_sparse_mla_decode_regimes.py:86`, `:120`. Uses GLM_NSA 576 and only a whole-tensor cos > 0.999. |
| Multi-split equals single split | `test_attention_mla_sm120.py` | `:976` |
| Head-major sparse-MLA output | none | none (only compressed_sparse) |
| Prefill top-k long context | `test_paged_prefill_topk_long_context.py` | `test_paged_prefill_topk_long_context.py` |
| Fused indexer | set equality | set equality |

## 5. Other semantics at long context (task 5)

- **Decode:** behavior depends on live length only via `selected_lengths` (chunk compaction plus speculative gating). No `cache_seqlens`-dependent path. Width is fixed at plan time; runtime width ≠ plan raises (N: `new run_unified_decode` checks, `:144-156`).
- **Indexer:** the merge-arm switch is live-length dependent in OLD (serial ↔ coop at ≈22k + 117·ctas_pg pools, ≈ 44k pools ≈ 176k tokens for 1 row on 188 SMs). NEW is always coop for multi-CTA. Only output order changes (see 3).
- **Head-major (A):**
  - NEW decode honors head-major for any row count. `output_buffer` has stride (D, maxQ·D, 1) fixed by `max_q_rows`, not by live rows (N: `_scratch.py:539-560`; `workspace.py:198-268`).
  - Decode partial strides are baked from the same workspace, and strides are exact in the compile key (`_lib/compiler.py` `tensor_compile_fact`).
  - The merge uses sym-int shapes and strides.
  - OLD decode was also head-major (OLD vLLM `b12x_mla_sparse.py:1491,1638`) and was honored.
  - LSE is always token-major `[rows, heads]` contiguous (`final_lse`), which is logically consistent with output[row, head].
  - NEW extend: bug in §0.2; the caller's token-major workaround is correct.
- **Prepared-program ABI hazards (NEW, INFERRED low risk):** q strides, `vector_q` alignment, and merge vector flags are baked at prime. q must be contiguous (`_scratch.py:299`), so strides match. Misalignment would fault, not corrupt.
- **Pooled selection helper:** `sparse_mla/pooled_selection.py` (commit 107ec861) is not used by NEW vLLM, which uses its own `expand_pools_and_append_tail`.
