# D7: mechanism, release and upstream comparison, and a prepared correction (CPU only)

PLAN.md amendment 18, steps 0a, 0b and 1. Text written by the mechanism agent; saved to this file by the coordinator
because the agent could not write report files. GPQA exposure accounting is in `../exposure/REPORT.md`.

No GPU was used: every container ran with `--network none`, `CUDA_VISIBLE_DEVICES=`, no `--gpus`, and
`TRITON_INTERPRET=1`. No `glm53*` container or serving stack was touched. B0's image, recipes, the public repo and the
frozen b0-mapguard files were not modified; the guard tests ran on a copy in `guardcopy/`. There is no GPQA text in any
output.

Evidence labels: **[measured]** = produced by running code here; **[source]** = read from source; **[inferred]** =
reasoning from the two. Engine paths are relative to the installed `vllm/` package; the extracted trees are
`kdcp/{v070,v080,v091}/vllm` and `kdcp/b0-cache/src/vllm`.

Images:
- **0.7.0:** `48e254d9` (recipe `glm53-flash-exl3-k3.25@tpurtell`)
- **0.8.0:** `e4d37a01`
- **0.9.1:** `a9e985c0`
- **B0:** `08d1cb90` (bundle-d1234)

## Step 0a: release comparison

**Writer identity [measured, sha256 inside each image].**
- `sparse_attn_indexer_kpool.py` is byte-identical in 0.7.0, 0.8.0 and 0.9.1 (`e7935f2a9b8e`). B0 (`9104be37c8cb`) adds
  only the col-511 helpers; `_kpool_compress_insert` is unchanged.
- The compress kernel `_kpool_softmax_rotate_write_cache_kernel` is identical in all four. The `kpool_compress.py`
  differences (0.7.0/0.8.0 `52959ea3`, 0.9.1/B0 `7fb8eab0`) touch only the tail-ring kernels.
- 0.7.0 and 0.8.0 share `scheduler.py` (`a509e42c`). 0.9.1 and B0 share `e405e84e`. The only change inside
  `_mamba_block_aligned_split` is the eagle back-off condition at line 409.

**Served configuration [measured from server logs].**

| | 0.7.0 | 0.8.0 | 0.9.1 / B0 |
|---|---|---|---|
| Log | `gpqa/…tpurtell+ps.server.log` | `eval-matrix-2026-10-04/R0-server.log` | `gpqa/…v0.9.1+ps.server.log` |
| Batched tokens | 2048 | 2048 | 2048 |
| DFlash draft slots K | 5 | 3 | 3 |
| Attention block | 3,584 | 4,352 | 4,352 |
| Mamba split | align | align | align |
| Prefix caching | on | on | on |
| Parallelism | DCP2, interleave 1 ("Using DCP-sharded sparse indexer KV compression … world_size=2, interleave=1") | DCP1, `VLLM_GLM53_MLA_OWNERS=split:25` | DCP1, `VLLM_GLM53_MLA_OWNERS=split:25` |

Draft slots are K for dflash [source: `config/speculative.py:1457-1471`].

| Item | stock 0.7.0 | stock 0.8.0 | stock 0.9.1 / B0 |
|---|---|---|---|
| **1. Faulty assumption in the actual prefill writer** | **Yes** [source]: `_kpool_compress_insert` at `sparse_attn_indexer_kpool.py:73-118`. Docstring "Assumes pool-aligned chunk starts" at :91; `if n < kpool: return` at :97; `write_mask = valid & (pos >= kpool-1)` at :103; batch-row gather t-3..t at :105; call site :398, before the tail seed at :442. Behaviour is the same in every release [measured, `out/release/writer-v070.json`]. | **Yes**, the same file and lines [measured, `writer-v080.json`] | **Yes**. 0.9.1 at the same lines; B0 at `:102-147`, call :427, seed :471 [measured, `writer-v091.json`, `writer-b0.json`]. All four writer logs are byte-identical. |
| **2. Can execution reach an unaligned start** | **Yes** [measured: real `_mamba_block_aligned_split` from the image plus budget lines emulated from `scheduler.py:562,689,915,1115`; `out/release/chunking-v070.json`]. A long prompt alone starts chunks at 2,043 + 3,584k, always residue 3. With D running decodes the chunk is 2043 − 11D, which gives every residue (D=1 aligned, D=2 → r1). Shared-step prefills give r1-r3. Prefix hits are block-aligned; continuations after them are unaligned. | **Yes** [measured, `chunking-v080.json`]. Starts at 2,045 (r1) and 4,090 (r2) in every 4,352 block; with D decodes the chunk is 2045 − 7D; shared steps give any residue. | **Yes**, the same as 0.8.0 [measured]. This matches attempt 3's measured starts (2045, 4090, 2031, 2017, 2003, 2010, …). |
| **3. How affected pools differ (order / chunking / rank)** | First prefill row of the step: the straddling pool is **unwritten**. Later row: it is **compressed from the previous request's last r tokens** [measured]. A single long prompt has 1 affected pool per 3,584 tokens: pool 510 of each block, positions 2040-2043, 3 earlier tokens. **DCP2:** both ranks run the writer on the full batch, but only the owner rank of the straddling pool (pool % 2) has a valid slot. So each event hits one rank, in all 11 MLA layers [measured with the real `get_compressed_slot_mapping`; e.g. start 2043 → rank 0, 2045/2046/2047 → rank 1, 5627 → rank 0]. | Same writer outcomes [measured]. A long prompt has 2 affected pools per 4,352-token block (pools 511 and 1022 of each block; 1 or 2 earlier tokens). DCP1 with MLA owners: each layer's index-K lives on its owner GPU, so every event hits every MLA layer, each on its own GPU [source: `start.sh:34-41`]. | Same as 0.8.0 [measured]. B0 attempt 3: 217 events per rank-layer set (1302 = 6 × 217 on GPU0, 1085 = 5 × 217 on GPU1) [measured earlier, `run-a3/d7-attribution.txt`]. |

**Reading.** This is a shared source defect: the same writer in every release, reachable in every served configuration.
It still produces a release difference in pattern:
- **Frequency per long prompt:** 0.7.0 has about 1 affected pool per 3,584 tokens; 0.8.0/0.9.1 have about 2 per 4,352.
- **Straddle shape:** 0.7.0 pools 3 previous-chunk tokens with 1 new; 0.9.1 pools 1-2 with 3-2.
- **Where it lands:** 0.7.0 hits only the owner rank of a DCP2 shard; 0.8.0/0.9.1 hit the owner GPU of each layer.
- **Residues under concurrency:** these differ because K differs (11-token versus 7-token budget steps per decode).

Whether these pattern differences matter for selection is unmeasured.

For the doc-127-length prompt (2,795 tokens, alone): 0.7.0 starts its second chunk at 2,043, so pool 510 is unwritten on
rank 0. 0.9.1/B0 start it at 2,045, so pool 511 is unwritten [measured, chunking emulation].

## Step 0b: upstream comparison

**vLLM main** (`c5422f4c46`, fetched 2026-10-13):
- **Writer:** unchanged. `vllm/models/glm5next/nvidia/sparse_indexer.py:54` has the same `_kpool_compress_insert` with
  "Assumes pool-aligned chunk starts" at :71; `amd/sparse_indexer.py:49/66` is the same. Per `git log`, none of the
  commits that touched these files (#53906 … #54951) change the assumption.
- **Scheduler:** no kpool alignment. `_mamba_block_aligned_split` only adds Mamba checkpoint stops. Its shared-prefix
  junction stop can even be unfloored (`junction_stop`), so it is also unaligned-capable [source].
- **Open PR [#57161](https://github.com/vllm-project/vllm/pull/57161)** (kpool compress rework) keeps the assumption:
  rows too early to own a full pool return [source, PR body].
- **Issues/PRs:** searches for kpool unaligned/chunk/`_kpool_compress_insert`/pool-aligned found nothing for D7.
- **Possibly related symptom, unverified:** [#61157](https://github.com/vllm-project/vllm/issues/61157) reports GLM
  hybrid + MTP degrading exactly when a concurrent request arrives. That matches the shared-step pattern, but it comes
  from a different fork (sm80) and is not evidence of D7.

**sglang main** (`b4822bae`) handles this writer-side, from its first GLM-5.3 kernels (`c66a285c94`, 2026-09-02):
- `python/sglang/srt/layers/attention/dsa/kpool_plan.py:152-210`: `_decompose_compress` sets `first_slot = start %
  kpool`; `_append_compress_rows` emits `n_from_tail=first_slot` and `tail_logical_base`.
- `kpool_fp8_index.py:1217-1264`: the kernel reads `tail[(tail_base + slot) % TAIL_SIZE]` for `slot < n_tail`, and
  chunk rows otherwise.
- So the fork's docstring "same invariant as sglang" is not accurate for sglang main.

**Compatibility of that approach with B0 [source + measured]:**
- **Ring contents.** B0's tail ring holds each request's last 4 prefill tokens at `pos % ring`. The writer is
  `kpool_seed_tail_cache`, called after the insert at `:471`; its kernel is `kpool_compress.py:449-526`. Ring = 8 for
  K=3 (`attention.py:196`). The slot is `own_block*ring + pos%ring` (`indexer.py:571`, the D2 fix). There is one
  never-shared, never-prefix-cached block per request (`single_type_kv_cache_manager.py:1097`).
- **Serving verification of the ring.** Attempt 3 checked tail seed (2,784,260 writes) and stash (284,640) ownership and
  positions with no violation [measured earlier, `run-a3/verdict.txt`].
- **What an unaligned start needs.** Positions s−r..s−1 (r ≤ 3) are among the previous chunk's last 4 tokens, or were
  stashed by a decode-classified short chunk.
- **Prefix hits:** block-aligned (attempt 3: 32 hits, all 4,352-aligned), so they never need the ring.
- **Preemption:** resume restarts at an aligned hit or at 0.
- **Speculation:** drafts do not exist while a request is still prefilling.

## Step 1: chosen correction

**Writer-side repair, `d7-fix.diff`.** One file, 114 lines: `sparse_attn_indexer_kpool.py`.
- **Ring reads.** `_kpool_compress_insert` gains optional `positions`, `tail_kv_cache` and `tail_slot_mapping`. A pool
  member is "own" when its batch row has the same tail block and position `pos − j`; otherwise it is read from the
  request's tail ring at `pos % ring`.
- **Write mask.** It requires every member to be own or at position ≥ 0; the old `pos >= 3` rule no longer applies to
  rows with a ring. Only the `n < kpool` early return is lifted, and only when the ring is present.
- **Rows without a ring.** Rows with no tail slot, and calls without the ring (`VLLM_KPOOL_SKIP_TAIL_CACHE=1` / no tail
  cache), keep B0's rule unchanged.
- **Call site.** It resolves the tail metadata before the insert; the existing seed stays after the insert, so ring reads
  precede overwrites.
- **No device sync:** no `.item`/`nonzero`; the prefill path is the op's eager break.

**Why writer-side:** amendment 18 prefers it when the preceding 1-3 keys and gates are available with verified ownership
and positions, and they are. It needs no scheduler change and keeps unaligned starts present, so Step 2 can exercise
them.

**Test image:** `local/glm53-v0.9.1-b1-d7-cpu:local`, `sha256:093ddce64f91…` = B0 plus the patched file
(`build/Dockerfile`; file sha `146a7c6b…`). For CPU testing only.

**CPU numerics finding [measured].** The images' Triton interpreter converts fp32→bf16 by truncation (519 of 1,024 values
differ from round-to-nearest-even) and mis-rounds the fp32→E4M3 carry (31.75 → 16 instead of 32). CPU record bytes are
therefore not GPU-faithful.
- The independent reference (`tests/d7lib.py`) pools in plain torch: numpy float32 exp, a butterfly Hadamard, torch
  softmax/scale. It applies the interpreter's element-wise conversions (bf16 truncation by bit mask; the E4M3 store cast
  via a one-line Triton store).
- Result: 0 of 6,000 random pools differ from the engine kernel.
- Earlier kernel-versus-kernel CPU tests are unaffected.

## Test matrix and results

**`tests/test_d7_fix.py`** drives the installed op end to end: the real call site, the real
`compute_kpool_tail_slot_mapping` and real seeding, stopping after the writes. Every computed pool is compared to the
plain-torch reference of the request's own tokens by logical position, and stray writes are checked.

| Check | B1 (patched) | B0 (unpatched) |
|---|---|---|
| M1 residues 0-3, first request alone | ok ×4 | r0 ok; r1-r3 pool 511 unwritten |
| M2 residues 0-3, later request in a packed batch | ok ×4 | r0 ok; r1-r3 pool 511 wrong (cross-request) |
| M2b first and later both unaligned, order swapped (×3) | ok | unwritten + wrong |
| M3/M3b short chunks of 1-4 tokens alone and packed, successive boundaries | ok | unwritten / wrong |
| M4 served 9,000-token chunking (2045/4090/6397/8442); M4b starts 4349-4355 at the 4,352-token block boundary | ok | pools 511/1022/… and 1087/1088 unwritten |
| M5 prefix hit at 4,352 on a reused stale ring block, then continuation at 5,479 | ok | pool 1369 unwritten |
| M6 preemption: resume on a new ring block, recompute, unaligned continuation | ok | unwritten |
| M7 ring absent → B0 rule (documented fallback) | as B0 | as B0 |
| M8 decode rows ahead of the prefill slice | ok | wrong |
| M9 random shared-step schedules (6 trials, 53 unaligned starts, 20,799 pools) | 0 bad | 53 bad, one per unaligned start |
| M10 aligned starts versus B0's verbatim function (24 packed batches, stale ring) | 24/24 byte-identical | 24/24 byte-identical (B0 against itself) |
| M11 op-level aligned workload: raw cache + ring SHA-256 after each of 4 steps, compared across images | identical to B0 (`out/fix/m11-hash-compare.txt`) | identical to B1 |
| M12 DCP2 geometry, owner rank (informative only) | 3/3 ok | 0/3 |
| **Total** | **28/28 met expectation** | **28/28 met expectation (defect documented)** |

**Regressions against the patched code** (`out/regress/summary.txt`, `summary-rerun23.txt`):

| Suite | Result |
|---|---|
| D1 col511 cpu | ALL PASS |
| D1 fixture cpucheck | OK |
| D2 test_fix U | 17/17 |
| D2 scenarios | 6/6 |
| D3 patched | PASS |
| D4 patched | PASS |
| D5/D6 excluded, still reproduced | as expected |
| Check-image tests with observer file + D7: c511check | 7/7 |
| tailguard | 22/22 |
| verify13 | 15/15 |
| d4limit | 0 failed |
| test_fix U | 17/17 |
| all-observers-together | 0 failed |
| Inert import | ok |
| Mapguard suite (guard file + D7): test_mapguard | 28/28 |
| a2 regression | ok |
| image smoke unset/device | PASS |
| dry-run r0/r1 | 0 guard errors |
| Dry-run analysis | INVALID for "no FULL CUDA graph / args differ", identical to the original `out-postA2` |
| `test_kpool_unaligned.py` | still `ok`: it calls the writer without ring arguments, so the B0 path, by design |

Two harness notes:
- The first "import_inert" failure was a circular import that is also present in the unmodified check image (`import
  sparse_attn_indexer_kpool` first). The import order in the runner was fixed.
- The first mapguard pass lacked `reach/sim_pool.py`, so it was copied into `guardcopy/`.

## Open items

1. **Blocks Step 2 as frozen.** The guard's idx_w_pf predicate models B0's rule: `pred = loc >= (kpool-1)`, at
   `b0-mapguard/guard/mapguard_hooks.py:1049`. On B1 it would still count `missing_pred`/`pool_cross` at every
   unaligned start, even when the pool was repaired. The guard needs an amendment modelling B1's rule (a member from the
   ring when it is not own), preferably with a byte-level check of straddling records, plus re-freezing and re-testing
   before any GPU run.
2. Because of the interpreter numerics, CPU tests cannot show GPU bytes. The fix's GPU behaviour is verified only in
   Step 2.
3. **Not covered by the writer tests:**
   - Decode-classified short prefill chunks (≤ next_n = 4 on SM120; `utils.py:698`, `indexer.py:738`) take the
     ring-based decode kernel. It is unchanged here and covered only by the D2 tests.
   - KV-connector/PD external-token starts are not configured and not covered.
4. **Overhead [inferred].** The fix adds two gathers of n×4×128 bf16 plus about 15 small elementwise ops per indexer
   layer per prefill step, with no host sync. Unmeasured on GPU; Step 2 records throughput.
5. The release DCP2 tests use an assumed per-rank geometry (896 pools per block, inferred from 3,584 × DCP2). Pool
   ownership comes from the real slot-mapping kernel.

## Files

All in this folder (hashes in `SHA256SUMS`):
- `d7-fix.diff`: the correction
- `patch/b0`, `patch/b1`: B0's file and the patched file
- `patch/guard/*.d7`, `patch/check/*.d7`: guard and check-image variants with the patch applied
- `build/`: Dockerfile and patched file for the CPU-test image
- `tests/`: `d7lib.py`, `test_release_writer.py`, `test_release_chunking.py`, `test_d7_fix.py`, `run_release.sh`,
  `run_regressions.sh`
- `out/release/`, `out/fix/`, `out/regress/`: logs and results
- `guardcopy/`: the copied mapguard suite and its outputs
