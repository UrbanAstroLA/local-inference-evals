# replay-topk: bounded single-GPU replay of the layer-3 top-k and sparse-MLA prefill (protocol)

**Status: prepared on CPU, not run. No GPU use is authorized by this document.** The analysis, its tests and this
protocol were written before any GPU data. The operator approves GPU time separately (section 8).

**All inputs are synthetic** (labelled in every output): no model weights, prompt text, token ids or captured
activations are used, so nothing here is written under a `PRIVATE-` path.

## 1. Question

capture-p0 (`../capture-p0/RESULT.md`) located where two identical warm prefills on stock v0.9.1 first part:

- at layer 3's indexer top-k output, from token 2,051 of warm chunk 1;
- in all 24 sampled rows, the same pool set in a different order;
- layer-3 sparse-MLA attention output then differs on those rows.

Every pair landed on different physical blocks, so allocation dependence and execution nondeterminism were not
separated. Following the assessor's section-5 replay rules (`../../HANDOFF-NEXT/ASSESSOR-DISPOSITION-22AX-2026-10-24.md`):

- **R1.** Is the served prefill top-k repeatable on fixed inputs? Separately for the same allocation and for relocated
  copies; set and order separately.
- **R2.** Is the served sparse-MLA prefill order-sensitive? Same index set, different order. Asked in both images,
  because this kernel differs between them.
- **R3.** Do R1's own orderings, passed through the served index path, reproduce the output difference at the
  attention output?

## 2. Served geometry (stock v0.9.1, warm chunk 1)

| Item | Value | Source |
|---|---|---|
| Query rows | positions 2,045-4,089 (2,045 rows); request seq_len 4,090 | capture-p0 chunking 2,045 / 2,045 / 262 |
| Indexer window | row p sees pools `[0, (p + 1) // 4)`: `cu_seqlen_ks = 0`, `cu_seqlen_ke = (p + 1) // 4` | `indexer.py:342-343` (`global_ctx = start_pos + 1 + offset`; `// COMPRESS_RATIO`), `COMPRESS_RATIO = index_kpool = 4`, DCP_WORLD = 1 |
| Logits | `[2,045, 1,022]` fp32, `clean_logits=False` | `sparse_attn_indexer_kpool.py:561-568`; width = compressed seq len 4,090 // 4 |
| Top-k | `select_k = 2048 // 4 = 512` pools | `kpool:574` |
| Rows that select | 2,039 (p ≥ 2,051); 6 rows (2,045-2,050) hold ≤ 512 pools | window rule |
| Expand | `pool_ids[:, :511]` → 2,044 token ids, plus the tail (seq_len % 4) at columns 2,044-2,046 | `kpool:633-653`, `kpool_compress.py` 091:967 / 070:950 |
| MLA heads | v0.9.1: 64, owner-local on GPU 0 (`attention.py` 091:461). v0.7.0: 32 local × DCP 2, gathered = 64 (`b12x_mla_sparse.py` 070:1537) | |
| Scheduler | max_num_batched_tokens 2,048, max_num_seqs 16 | capture-p0 attempt-2 server log |

The window rule was traced rather than assumed (VERSION-CMP listed it as untraced). Two checks:

- `tests/test_served_args.py` S3 runs the image's own `build_prefill_chunk_metadata` under the Triton interpreter and
  compares it with the formula.
- `scripts/rt_common.py:served_window` uses that same builder on the GPU to make the replayed `cu_seqlen_ks/ke`.

## 3. R1: top-k repeatability (v0.9.1 image; short confirmation in v0.7.0)

**The call, as served** (`sparse_attn_indexer_kpool.py:600-609`, byte-identical in both images, as is the `_C` binary):
```
topk_dst = torch.full((num_rows, 512), -1, dtype=torch.int32)          # kpool:576-581, fresh per call
torch.ops._C.top_k_per_row_prefill(logits, chunk.cu_seqlen_ks, chunk.cu_seqlen_ke, topk_dst, num_rows,
                                   logits.stride(0), logits.stride(1), select_k)
```
- The schema comes from the binary's registration string (test S2):
  `top_k_per_row_prefill(Tensor logits, Tensor rowStarts, Tensor rowEnds, Tensor! indices, int numRows, int stride0, int stride1, int topK)`.
- `ks` and `ke` come from the served builder.
- Calls run on the default stream and are synchronised after each one.

**Input regimes (all SYNTHETIC):**

| Regime | Logits | Basis |
|---|---|---|
| K, served producer | The image's DeepGEMM `fp8_fp4_mqa_logits` (`vllm/utils/deep_gemm.py`), called as at `kpool:561-568`. Inputs: synthetic e4m3 q `[2045, 32, 128]` (`q_scale` None); e4m3 k `[1022, 128]`; power-of-two k scales; fp32 weights `[2045, 32]` | The served path: FP8 cache (`use_fp4_cache` False), `scale_fmt = "ue8m0"` (`attention.py` 091:284). This gives the served tensor's actual shape, strides (padded or not; recorded) and value granularity |
| C, continuous | fp32 standard normal, copied into a tensor with K's strides | No exact ties (probability ~1) |
| T, tied | C rounded to bf16 precision, stored as fp32 | Many exact ties at the 512th value (counted per row). This is a stress case: real logits are weighted sums over 32 heads and rarely tie exactly |

If K fails (for example a DeepGEMM JIT failure):
- C and T use a contiguous tensor, and the output says "stride ASSUMED";
- R3 is not run (pre-registered; no substitute regime).

**Per regime, after 3 warm-up calls:**

1. **Same allocation.** 24 calls on one logits tensor and one output buffer. The buffer is re-filled with -1 before
   each call (production allocates it fresh with -1). All calls are compared with call 1.
2. **Relocation.** 24 calls, each with:
   - a fresh strided copy of the logits;
   - fresh `ks` / `ke` and a fresh output buffer;
   - the allocator shifted by a dummy allocation.

   The addresses are recorded; at least half the calls must differ from the reference addresses.
3. **Per row:** whether the order varies, whether the set varies, whether the last column (dropped by the 511
   expand) varies, and the number of distinct orderings.
4. **Controls:**
   - *Negative, in-kernel:* the 6 rows with ≤ 512 pools must be `0..n-1` then -1, in every call.
   - *Input unchanged:* the logits are bitwise identical after every call.
   - *Comparator positive control:* a planted swap and a planted replacement are flagged on exactly those rows.
   - *Semantics:* on 128 sampled rows, the kernel's set equals an fp64 reference top-k wherever the 512th and 513th
     values do not tie. A wrong stride, window or argument fails this.
5. **Informational:**
   - the order pattern: last column = minimum logit; fully sorted; index-ascending; median sorted-suffix length (the
     threshold-bin block);
   - rows tied at the threshold;
   - non-finite in-window values.

**v0.7.0 confirmation:** regimes C and T, 20 + 20 calls.
- It is a **binary confirmation at the v0.9.1 geometry only**.
- On v0.7.0's served geometry (DCP=2), the window builder localises each rank to ≤ 511 pools for p < 4,096
  (`indexer.py`, DCP_WORLD > 1 branch). The per-rank top-k therefore takes the index-ordered shortcut in this chunk,
  and the selection happens in the DCP merge (section 9).

## 4. R2: sparse-MLA prefill order sensitivity (both images)

The kernel, its plan, its cache writer and every index step are the image's own code, with the served arguments of
layer 3 (`scripts/r2_mla.py`).
- The Caps keyword set equals the served `_make_plan` call's (test S4, by AST of each image's file).
- The values follow the served configuration (S4 / S6).

| | v0.9.1 (DCP=1, split:25, NoPE record) | v0.7.0 (DCP=2, TP=2), call of DCP rank 0 |
|---|---|---|
| Plan | `b12x_mla_sparse._prepare_b12x_sparse_mla_plan(sparse_mla, Caps(...), forced_num_splits=32)` (091:151, 1839-1890) | `sparse_mla.plan(Caps(...))` (070:1621-1648) |
| Caps | extend, max_q_rows 2048, 64 heads, max_batch 16, head_dim 512, v 512, GLM_NEXT, scale_format None, record 528, fp8_rope None, max_chunks 32, page 64, head_major False, return_lse True, natural, softmax_scale 256^-0.5 | extend, max_q_rows 2048, 64 heads, max_batch 16, head_dim 576, v 512, max_chunks 32, page 64, head_major True |
| KV cache | `[80 pages, 64, 528]` uint8; records written by the served GLM_NEXT writer (`concat_and_cache_glm_next_mla`, 091:2108) from synthetic bf16 latents of all 4,090 tokens; pages in a shuffled order | `[48, 64, 656]`; rank 0's tokens (even positions, interleave 1) written by `ops.concat_and_cache_mla(..., "fp8_ds_mla", scale)` (`backend.py:1118`) with a zero 64-lane RoPE tail |
| Q | bf16 `[2045, 64, 512]`, a view of a `[2048, 64, 512]` workspace | bf16 `[2045, 64, 576]` workspace view, lanes 512-575 zero (070:1465) |
| Index | `topk_indices_buffer [2048, 2048]` reset to -1 (kpool:461), then `[:, :2047] = expand_pools_and_append_tail(pool_ids[:, :511], p + 1, 4)`; then `_logical_topk_to_physical_slots`, `nsa = min(p + 1, 2048)`, `_compact_dropped_selection` and `_mask_page_table_after_nsa_len` (091:3026-3048) | same expand; then `triton_filter_and_convert_dcp_index(..., dcp_size=2, dcp_rank=0, interleave 1, BLOCK_SIZE 64, out=, valid_counts=)`, `nsa = min(valid, local p + 1)`, `_mask_page_table_after_nsa_len` (070:2764-2781) |
| Run | `sparse_mla.bind(plan, scratch, q, kv_cache, selected_indices, cache_lengths=[4090], selected_lengths=nsa)`, then `sparse_mla.run` (091:3236-3248) | `plan.bind(scratch, q, selected_indices, cache_seqlens_int32=[2045], nsa_cache_seqlens_int32=nsa)`, then `run_extend(binding=, kv_cache, sm_scale, v_head_dim=512, return_lse=True, lse_scale="natural")` (070:3017-3041) |

The scratch is allocated with `torch.empty`, as the served workspace is: it is not zeroed. The prefill kernel is MG
in both images; `prefill.py` has no other prefill path, and v0.9.1 records `prefill_mg_enabled`.

**Synthetic selections:**
- Rows with ≤ 512 pools take the shortcut layout.
- Each selecting row gets a fixed random set of 512 pools. Its lowest synthetic score is the fixed last column, which
  the expand drops.

**Order variants** (columns 0-510 only, so the expanded token SET and `nsa` are identical to base; both are checked
per variant):

| Variant | v0.9.1 (built through the served index path) | v0.7.0 |
|---|---|---|
| base | Above-threshold block (495) in a random order, then the 16 lowest sorted descending. This is the upstream kernel layout; the block size of 16 is ASSUMED | the filter's output for base |
| atomic2 | the same, with another above-threshold order (what two served runs differ by) | random permutation of the valid prefix |
| asc / desc | ascending pool id / score descending | sorted slots / reversed |
| perm1-3 | uniform random | uniform random |

On v0.7.0 the variants permute the filter's output, which is the kernel's direct input. This isolates the kernel from
the filter's own ordering, which is measured separately.

**Q regimes:** flat (q std 1) and peaked (q std 4). The KV latent std is 1.

**Measurements per Q regime, after 2 warm-up calls:**

| Measurement | What it shows |
|---|---|
| Same order, same allocation, 8 calls | (a) |
| Scratch poisoned (0xFF, 0x00) before a call | scratch read before write |
| Relocation × 3: fresh q workspace, cache, index buffers and scratch; same pages, then other pages | (b) |
| Unwritten cache records poisoned differently | hidden reads |
| Each variant twice | its own repeatability; rows that differ from base (selecting and shortcut rows separately); max abs and relative L2 in the bf16 output |
| fp32 reference attention on 64 rows, from the decoded records | puts the order effect next to the kernel's quantisation error |
| Positive control: every selected pool p → (p + 1) mod pool_len | must change ≥ 90% of selecting rows and no shortcut row |

**v0.7.0 only, the filter:**
- 8 repeated `triton_filter_and_convert_dcp_index` calls on the same input: is the output identical?
- Is the input order preserved (rows sampled)?

## 5. R3: composition (v0.9.1, only with regime K)

R1's regime-K outputs are used:
- call 1, and the next 8 same-allocation calls: the same logits, the orderings the served kernel produced;
- each goes through the v0.9.1 index path and kernel (flat Q, the same cache);
- each output is compared with call 1's output.

Per call, recorded:
- rows whose top-k order differs, and rows whose set differs;
- rows whose output differs;
- rows whose output differs without a top-k difference (must be 0);
- magnitudes.

The logits are not computed from the Q / KV (synthetic), so R3 shows the mechanism on the served path. It does not
reproduce serving values.

## 6. Execution

- One container per image: `--gpus device=0 --network none`, stock image by digest (`runner/images.txt`), no pulls.
- Mounts: `scripts/` read-only, the output directory, and the image's compile cache.
- Driver: `scripts/in_container.sh VERSION SOFT` runs `scripts/replay.py`. The configuration is frozen in
  `replay.py: CONFIG`; there are no command-line knobs besides the soft deadline.
- **Compile caches** (`B12X_COMPILE_CACHE_DIR`, `DG_JIT_CACHE_DIR`, `TRITON_CACHE_DIR`) live under
  `replay-topk/cache/<version>/`. They are seeded once from the serving caches (read-only there), are regenerable,
  and are withheld from publication. A cache hit cannot change results; warm-up calls run before any comparison.
- **Outputs:** `runs/attempt-N/<version>/{env,r1,r2}.json` (synthetic data, counts and magnitudes only), then
  `runs/attempt-N/replay-result.{json,md}`.

## 7. Pre-registered readings (`analysis/analyze_replay.py`)

**Missing or malformed evidence, a failed gate or a failed control gives INCOMPLETE for that unit. INCOMPLETE is
never read as a negative.**

| Unit | Gates (all required) | Readings | Assessor section 5 |
|---|---|---|---|
| R1, per image × regime | status ok; ≥ 20 same-allocation and ≥ 20 relocated calls; warm-up; input unchanged; shortcut rows index-ordered and never varying; served geometry (2,039 selecting rows, first at 2,051); comparator control; relocation achieved; fp64 set agreement on untied rows; window = rule | **A-SET**: same allocation, the set varies. **A-ORDER**: same allocation, the order varies. **B-ALLOC**: same allocation repeatable, relocated calls differ. **REPEATABLE**: no variation in N + N calls | A-*: execution nondeterminism (or incomplete initialisation). B-ALLOC: allocation / layout dependence |
| R2, per image × Q regime | status ok; warm-up; ≥ 5 repeats; no NaN; geometry; all 6 variants present, each with the same set and nsa and a changed order; positive control ≥ 90% | **A-EXEC**: same order and allocation differ. **B-ALLOC**: relocation, scratch or cache poison changes the output. **ORDER-SENSITIVE**: repeatable, and some variant changes selecting rows only. **ORDER-INSENSITIVE**: repeatable, and no variant changes any row. A shortcut row changing gives INCOMPLETE | A-EXEC: (a). B-ALLOC: (b) or hidden reads. ORDER-SENSITIVE: the operation's output depends on R1's order |
| R3 | R1 regime K not INCOMPLETE; ≥ 2 composed calls | **COMPOSED**: outputs differ where top-k orders differ and nowhere else. **NO-TOPK-VARIATION**: R1 produced no differing order. **ANOMALY**: output differs without a top-k difference. ORDER-VARIED-OUTPUT-EQUAL | the end-to-end mechanism on synthetic inputs |

Also reported, never gating:
- fractions of selecting rows whose order varied; distinct orderings;
- whether the last column varied;
- the order pattern; ties at the threshold;
- the logits stride;
- R2 magnitudes, and each variant against the fp32 reference;
- the v0.7.0 filter's repeatability and order preservation.

**(c) Lifecycle** (operation equal, later cache read differs) is outside this replay: no cache write follows the
replayed operations. capture-p0 found layer-3 MLA KV byte-identical.

**How the pieces combine.** capture-p0's pattern ("same set, other order", attention differs) is explained by this
replay only if both of these hold:
- R1 reads A-ORDER or B-ALLOC with the set stable;
- R2 (v0.9.1) reads ORDER-SENSITIVE.

R3 COMPOSED then shows the path end to end on synthetic inputs.

## 8. Cost, budget and commands

| Image | Start + import | R1 | JIT / plan (warm cache - cold) | R2 (+R3) calls | Planning | Hard wall cap |
|---|---|---|---|---|---|---|
| v0.9.1 | 25 s | 15 s | 11-290 s (DeepGEMM, B12x plan + writer, Triton) | 38 s | 89-368 s | 400 s |
| v0.7.0 | 25 s | 8 s | 8-170 s (B12x plan, Triton) | 35 s | 76-238 s | 300 s |
| **Total** | | | | | **0.046-0.168 GPU-h** | **700 s = 0.194 GPU-h** |

- **Basis** (`runner/cost_replay.json`): the captured-p3b replay on the same image ran its attention part in 14 s,
  including start-up, with the cache seeded. A cold CuTe compile is an assumption (60-150 s per plan); no log
  measures it.
- **Hard caps:** each container runs under `timeout` at its cap minus 20 s, with a soft deadline 40 s earlier inside.
- **Budget gate** (`runner/budget_replay.py`):
  - `APPROVED_GPU_H` is required (no default);
  - the ledger is `costs-replay.jsonl` only;
  - an attempt is refused when used + the full caps exceed the approval;
  - a complete reading is not re-run;
  - after an INCOMPLETE attempt, `REPLAY_PRIOR_ACK` must equal the first 12 hex of that attempt's
    `replay-result.json` sha256.
- **Runner gates** (`runner/run_replay_topk.sh`):
  - `ATTEMPT_REASON`;
  - `sha256sum -c SHA256SUMS`;
  - image digests;
  - no inference or check container running;
  - GPU-0 preflight (`runner/preflight_gpu.py`). It reads the full `nvidia-smi` process table, so graphics clients
    are listed. It refuses on: any compute process on GPU 0; a graphics client above 64 MiB; > 1,024 MiB used; or
    > 5% utilisation over 12 samples.
- **Ledger:** one line per container, on every exit path, including interrupt.

The remaining operator approval for the investigation is 1.103 GPU-h. This package asks for 0.2.

```bash
cd tile-p4/replay-topk
APPROVED_GPU_H=0.2 ATTEMPT_REASON="replay-topk attempt 1: R1-R3 per PROTOCOL.md" DRY_RUN=1 runner/run_replay_topk.sh
APPROVED_GPU_H=0.2 ATTEMPT_REASON="replay-topk attempt 1: R1-R3 per PROTOCOL.md" runner/run_replay_topk.sh
```

## 9. What this cannot show, and what was infeasible

- **Synthetic inputs only.**
  - Real logits need the model's indexer weights and the prompt.
  - Regime K has the served producer's granularity but not the real value distribution or tie structure.
  - Q / KV are random, so magnitudes are not serving magnitudes.
- **Single GPU, one call per operation.** This is not the serving pipeline:
  - no CUDA graphs (prefill > 64 tokens runs eager in serving too);
  - no concurrent kernels from other layers or the other rank;
  - no all-reduce.
- **v0.7.0's served selection is not replayed.**
  - With DCP=2, the per-rank `top_k_per_row_prefill` sees ≤ 511 pools in this chunk, so it takes the index-ordered
    shortcut.
  - The selection happens in the DCP merge: the owner exchange (b12x `run_row_topk`) or the CuteDSL global top-k
    (`sparse_attn_indexer.py` 070:1331-1392 / 802-852). Both need the two ranks' all-gathered candidates.
  - The closest faithful alternatives, each needing its own estimate: a two-GPU replay of the merge; or a
    single-process call of the post-gather kernels on synthetic gathered candidates.
  - R1 in the v0.7.0 image is therefore labelled a binary confirmation. R2 there covers rank 0's kernel call only,
    without the cross-rank LSE combine.
- **v0.7.0 chunk geometry under DCP=2 is not measured.** R2 uses the v0.9.1 chunk geometry in both images, so
  results are comparable across images, not served-exact for v0.7.0.
- **The base / atomic2 threshold-block size (16) is an assumption.** R3 uses the kernel's own orderings instead.
- **No completion-behaviour claim.** Nothing here connects order effects to replies.

## 10. Files

| Path | What |
|---|---|
| `scripts/rt_common.py` | geometry, served window, comparisons |
| `scripts/r1_topk.py` | R1 |
| `scripts/r2_mla.py` | R2, R3 |
| `scripts/replay.py` | the driver, with the frozen config |
| `scripts/in_container.sh` | the container entry point |
| `analysis/analyze_replay.py` | the readings |
| `runner/run_replay_topk.sh`, `budget_replay.py`, `preflight_gpu.py`, `images.txt`, `cost_replay.json` | runner and gates |
| `tests/run_cpu_tests.sh` | the CPU suite; `tests/RESULTS.txt` holds its results |
| `tests/test_served_args.py` | the replay's calls against each image's code |
| `tests/test_harness_cpu.py` | the harness on CPU stand-ins of known behaviour |
| `tests/test_analysis.py` | readings, including planted defects |
| `tests/test_runner.sh` | runner refusals, ledger and interrupt (fake docker / nvidia-smi) |
| `SELF-CRITIQUE.md` | what could make each reading spurious |
| `SHA256SUMS`, `make_sums.sh` | the frozen package |
