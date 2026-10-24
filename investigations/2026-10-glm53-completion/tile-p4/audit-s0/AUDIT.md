# Stage 0 zero-GPU audit (assessor 22ax, section 3) — 2026-10-24

**Governing:** `HANDOFF-NEXT/ASSESSOR-DISPOSITION-22AX-2026-10-24.md`, sections 3 and 4. No GPU was used. No container
had `--gpus` or network. Existing files were not modified.

**Labels:** **S** read in source (file:line, hashes below). **O** observed in the retained Stage-0 records. **I**
inferred.

## Outcome (assessor decision table)

**Valid state differs.** The operation that produces the difference has not been located.

- Fully written cached MLA KV blocks differ between the two warm-ups of identical prompt ids. Each block was built
  by its own replicate's warm request, and no other tensor view aliases it at point 0. In unique physical pairs
  (O, `unique_pairs.out`):
  - 770 of 847 differ in each variant;
  - layer 3: 0 of 77 differ;
  - each of the 10 MLA layers 7-43: 77 of 77 differ.
- The KDA recurrent state in the cached-state column is entirely live state. 496 of 544 differ:
  - layers 0, 1 and 2: 0 of 16 differ;
  - each of the 31 KDA layers from 4 to 44: 16 of 16 differ.
- The reported first mismatch (layer-7 indexer K, logical block 0) checks out against the addressing formula and its
  provenance. The same logical block's MLA KV, which has no never-written bytes, also differs. It is the first
  mismatch under a valid-only scope (`recount_point0.out`).
- **Not** "confined to inactive storage". **Not** "instrumentation or addressing invalid" for the point-0 valid
  scope. The gate still needs scope corrections (tasks 2, 5 and 6).
- **Next row of the table:** a bounded targeted test (assessor section 5) at the earliest justified boundary. Read
  across layers, that boundary is the state leaving layer 3 during the warm prefill:
  - layer 3 is the first MoE layer and the first sparse-MLA/indexer layer (`config.json`:
    `first_k_dense_replace` 3; `layer_types`);
  - **S/O:** layers 0-3 match by digest and layers 4 and up differ.
  - This is a layer-order observation, not a temporal trace.

### Inputs and hashes

| sha256 (prefix) | File |
|---|---|
| `9fbde27c1f657812` | tile-p4/analysis/analyze_stage0.py |
| `93f4147a49caf2e2` | tile-p4/build/files/vllm/p4_hooks.py (= served copy in `local/glm53-v091-p4-s0:local`) |
| `fe0a0b17b5414c9d` | forcewin_hooks.py (served) |
| `4e6dc18ba654def3` / `a4c088113af55fda` | tile-p4/diagnostics/s0map.py / s0map.out |
| `9efb2b5787016dcc` / `d4cc2900f9264953` | tile-p4/RESULT-STAGE0.md / STAGE0-PROTOCOL.md |
| `e42348b44158ed59` / `bc45d2b78429f135` | runs-stage0/stage0-attempt-1/p4log/p4_r0_p340.jsonl / sched_r0_p340.jsonl |
| `25fb7db506336200` / `997b454635755989` | runs-stage0/stage0-attempt-1/client.jsonl / server.log |
| `6b477cfc1fbf8cdf` | model `config.json` (snapshot 0490d2f7…) |

- **Engine sources** were read from base image `a9e985c0…` and the served image `7408f893…`. Every cited stock file
  is byte-identical in both. Only `v1/worker/gpu/model_runner.py` differs: `101e456b` stock, `8d276b3a` served
  (hooked). The full list is in `source-sha256.txt`.
- **Paths:** `vllm/…` is relative to `dist-packages/`, and `b12x/…` to `/opt/b12x/`.

**Short names for the frequently cited files:**

| Short name | Path |
|---|---|
| kda.py | `vllm/models/glm5next/nvidia/kda.py` |
| gdn_attn.py | `vllm/v1/attention/backends/gdn_attn.py` |
| be_utils.py | `vllm/v1/attention/backends/utils.py` |
| w_mamba_utils.py | `vllm/v1/worker/mamba_utils.py` |
| mamba_hybrid.py | `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` |
| l_mamba_utils.py | `vllm/model_executor/layers/mamba/mamba_utils.py` |
| causal_conv1d.py | `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` |
| kpool.py | `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` |
| kpool_compress.py | `vllm/models/glm5next/nvidia/ops/kpool_compress.py` |
| compressor_utils.py | `vllm/v1/attention/backends/mla/compressor_utils.py` |
| kv_cache_utils.py | `vllm/v1/core/kv_cache_utils.py` |
| attn_utils.py | `vllm/v1/worker/gpu/attn_utils.py` |
| runner | served `vllm/v1/worker/gpu/model_runner.py` |

**Scripts** (`scripts/`, run with `scripts/run_cpu.sh` = `docker run --rm --network none -e CUDA_VISIBLE_DEVICES=` on
the stock image; tile-p4 is mounted read-only):

| Script | Output |
|---|---|
| `recount_point0.py` | `recount_point0.{out,json}`, `recount_point0_blocks.csv`, `../PRIVATE-audit-s0/focus_digests_point0.json` |
| `alias_history.py` | `alias_history.{out,json}` |
| `reconcile.py` | `reconcile.out` |
| `unique_pairs.py` | `unique_pairs.out` |

---

## Task 1. Point-0 counts restricted to fully computed, valid cached state

**When point 0 is taken (S):**
- The entry digest runs in `prepare_attn`, through `_fw.on_batch` (runner `:1464-1465`, forcewin `:649-650`,
  `p4_hooks.py:835-878`, call `:867`).
- That is **before** `model_state.preprocess_state`, the mamba align pre-copy (runner `:1615-1625`), and before the
  forward.

**Boundary geometry (O):**
- n_cached is a multiple of 4352 and of 1088 in all 16 windows (13,056 / 91,392 / 100,096 / 108,800 / 126,208).
- No cached block is partial.
- Block sizes: MLA group 4352; tail 8; KDA groups 4352; draft (sliding window) 1088 (`server.log:241`).

**Valid at point 0:**
- **MLA KV and indexer K:** logical blocks j with (j+1)·4352 ≤ n_cached.
- **KDA:** only the cached-state column, j = n_cached/4352 − 1 (task 3).
  - The recurrent view is entirely live.
  - Conv rows 0-2 are live and rows 3-5 inactive. That digest is therefore mixed.
- **Draft:** in-window blocks below n_cached.

Each variant compares repeat 1 with repeat 2. Counts run over windows × ranks × blocks. `unique` de-duplicates
the physical pairs that windows of one reply share.

| Family (valid scope) | V1 differ / compared | V2 differ / compared | Unique physical pairs | By layer (O) |
|---|---|---|---|---|
| MLA KV, fully cached | 1,460 / 1,606 | 1,460 / 1,606 | 770 / 847 (each variant) | L3 0/77; L7 … L43 77/77 each |
| Indexer K, fully cached (2 never-written pool records per block; task 2) | 1,460 / 1,606 | 1,460 / 1,606 | 770 / 847 | same pattern as MLA |
| KDA recurrent, cached-state column (all live) | 496 / 544 | 496 / 544 | 496 / 544 | L0, L1, L2 0/16; the 31 layers L4 … L44 16/16 each |
| KDA conv, cached-state column (rows 0-2 live, 3-5 inactive) | 538 / 544 | 534 / 544 | same | L0 16/16; L1 16/16 (V2 8/16); L2 10/16 (V2 14/16); L4 and up 16/16 |
| Draft K/V (layers 45-49), cached span inside the window | 0 / 40 | 0 / 40 | — | all-zero in both repeats (task 2) |

**Provenance (O):** every MLA, indexer and KDA valid block (1,606 + 1,606 + 544 + 544 per variant) has, at the
same logical index, the physical id recorded for its own replicate's warm request of the same reply. Draft blocks
fail this check (task 2).

**Earliest layer with a valid-state difference (O):** layer 4 (KDA recurrent) in all 16 window × rank cases of
each variant.
- In layer order: the valid state of layers 0-2 (KDA recurrent) and layer 3 (MLA KV, indexer K) compared equal.
  Everything from layer 4 differs.
- **Caution:** equal two-sum digests are not proof of equality (task 6).

## Task 2. Everything else that the published counts included

| Category (point 0) | V1 | V2 | What it is (S/O) |
|---|---|---|---|
| MLA KV, block beyond the cached span (logical n_cached/4352) | 0 / 88 | 0 / 88 | Allocated for the step, not yet written. Digest (0,0) in both repeats: never written since allocation (O) |
| Indexer K, beyond the cached span | 0 / 88 | 0 / 88 | Same; (0,0) both |
| Kpool tail ring (one block per request) | 0 / 88 | 0 / 88 | Fresh per-request block. n_cached % 4 = 0, so no in-progress pool. (0,0) both |
| KDA running column (pre-copy destination, (n_cached+n_sched−1)//4352) | conv 516/544, rec 516/544 | conv 539/544, rec 497/544 | Holds stale bytes from earlier owners at point 0; overwritten by the pre-copy before it is read (task 3) |
| KDA speculative columns (num_speculative_blocks = 3) | conv 1,560/1,632, rec 1,502/1,632 | conv 1,616/1,632, rec 1,490/1,632 | Spec-verify scratch. Not read by a prefill step |
| KDA conv rows 3-5 of the cached-state column | inside the mixed row above | | The spec extra columns, `conv_kernel−1+num_spec` = 3+3 (`l_mamba_utils.py:396-419`, `gdn/base.py:43-47`). Copied by the pre-copy; not read or written by `causal_conv1d_fn` (task 3) |
| Draft cached-span blocks | 0 / 40 | 0 / 40 | Digest (0,0) in both. The ids were **not** built by the warm: e.g. 709 (V1-r1) was R01-deep-r1's tail block until that request was removed (O, `alias_history`) |
| Null placeholders (physical block 0 in the row, both repeats) | KDA 9,384 per family; draft 3,080 | same | Skipped by `g_entry` (`analyze_stage0.py:249`). Counted only |
| Partial cached blocks | 0 | 0 | — |

**Reconciliation (O, `reconcile.out`):** this split reproduces the published point-0 numbers exactly.
- **"MLA KV 1,460 of 1,774":**
  - s0map's `attn` family also takes the 5 draft layers (`model.layers.45-49.self_attn.attn`), 80 blocks;
  - it also takes 88 beyond-span blocks.
  - So: 1,460/1,606 cached + 0/88 new + 0/80 draft.
- **"Indexer K 1,460 of 1,694":** 1,460/1,606 + 0/88.
- **"KDA layer 0: 80/80 conv, 44-64/80 recurrent":** the layer-0 recurrent differences are all in running and
  speculative scratch columns. The cached-state recurrent column is 0/16 in both variants.

**Regions inside the digest that the next operation does not read, or does not read as written state (S/I):**
1. **KDA conv rows 3-5** of the cached-state column:
   - `causal_conv1d_fn` uses `state_len = KERNEL_WIDTH − 1` (`causal_conv1d.py:69-71`);
   - it reads rows 2, 1 and 0 (`:160-181`) and writes rows 0-2 (`:205-244`, mask `:242`).
2. **The KDA running column and the 3 speculative columns.** Point 0 precedes the pre-copy, which overwrites the
   whole running column before the forward (task 3). A prefill step does not use the spec columns.
3. **Beyond-span MLA, indexer and tail blocks.** All zero here.
4. **Page padding** is excluded by construction: index_select on the views.
   - KDA page: 2,244,608 B used of 2,297,856 (`abstract.py:29-43`).
   - Tail view: 4,096 B of 143,616.
   - Draft view: 2,228,224 of 2,297,856 (`server.log:240`).
5. **Read, but never written by the warm (consumed-unwritten), indexer K:**
   - The prefill pool writer assumes pool-aligned chunk starts (`kpool.py:91`). It masks any pool whose last token
     sits at batch position < kpool−1 (`kpool.py:97-103`; `kpool_compress.py:236-241`, `:303-318`).
   - The slot is emitted only on the pool-completing token (`compressor_utils.py:41-59`).
   - **O:** warm chunks are 2,045 / 2,045 / 262 per 4,352-token block (sched rows; chunk starts ≡ 1, 2 mod 4).
   - **Consequence:** in every cached block, pools 511 (tokens 2044-2047) and 1022 (tokens 4088-4091) are never
     written by that block's warm (all 1,606/1,606 blocks).
   - **S:** the prefill indexer gathers and scores every pool of the context (`kpool.py:531-568`, top-k `:574-609`).
   - **O/I:** here these records are probably zero in both repeats:
     - layer-3 indexer blocks compared equal in all 77 pairs;
     - no recorded earlier indexer-group use of the focus blocks.
   - **I:** they are therefore unlikely to explain this probe's divergence. In general serving, the selection would
     read allocation-dependent stale bytes. This is outside this audit's question and recorded as a lifecycle
     finding.
6. **Draft cached-span blocks** are all zero and were not built by the warm (O). The drafter's in-window context
   for the cached span is therefore unwritten storage (I). They are equal across repeats, so they are not a
   Stage-0 divergence source.

## Task 3. KDA state selected by the first remainder-prefill step (align mode; example V1 R01-early)

| Step | Effect (S) | R01-early (O) |
|---|---|---|
| add_request | `state_idx = (num_computed − 1) // block_size` (`mamba_hybrid.py:134-138`) | (13,056−1)//4,352 = 2 |
| point 0 digest | `prepare_attn` → entry digest (runner `:1464-1465`) | KDA column 2 = cached state; column 3 still stale |
| preprocess | `src_col = state_idx`, `src_off = max(num_accepted−1, 0)` = 0 (num_accepted reset to 1 at add, `mamba_hybrid.py:130`); `new_state_idx = ceil(computed_after/4352) − 1` (`w_mamba_utils.py:425-436`); skip if src == dst (`:493-499`) | src 2 → dst (13,056+2,045+4,351)//4,352 − 1 = 3 |
| pre-copy | SD conv: `state[bt[src], token_bias:] → state[bt[dst], :conv_width−token_bias]` with conv_width = `state.size(1)` = 6 (`w_mamba_utils.py:208-229`, `:827-830`): **all 6 rows**. Temporal: full state copy (`:232-251`, `:838-841`) | logical col 2 → col 3; V1-r1 group 2: 440 → 715; V1-r2: 761 → 351 |
| metadata | `block_table_tensor = gather(bt, (seq_lens−1)//block_size + arange(1+num_spec_blocks))` (`be_utils.py:1022-1042`); `non_spec_state_indices_tensor = block_table_tensor[:, 0]` (`gdn_attn.py:250-255`, `:353`; mixed spec batches `:427-429`); `has_initial_state = context_lens > 0` (`gdn_attn.py:569-572`) | col 3; has_initial_state = True |
| conv (prefill) | `causal_conv1d_fn(..., conv_states, has_initial_state, cache_indices=non_spec_state_indices_tensor)` (`kda.py:528-540`); SD layout transposed to (dim, state_len) (`kda.py:449-450`; default "SD", `l_mamba_utils.py:26-48`); IS_APC_ENABLED False, so the input and output column are the same (`causal_conv1d.py:98-128`, `:139-141`, `:228-232`, `:750`); reads rows `state_len−1 … 0` = 2, 1, 0 (`:165-181`); writes rows 0-2 (`:242-244`) | rows 0-2 of col 3 (= col 2's rows 0-2) |
| recurrent (prefill) | `gather_initial_states(recurrent_state, non_spec_state_indices_tensor, has_initial_state)` reads the full row (`kda.py:642-644`; `gather_initial_states.py:32-42`); `chunk_kda_with_fused_gate` (`kda.py:668-689`); `scatter_states` writes the full row back (`kda.py:691-695`; `scatter_states.py:31-33`) | full state of col 3 (= col 2's) |

**Live state at the boundary**, attributed to the cached-state column (logical n_cached/4352 − 1). Page size
P = 2,297,856 B; offsets are within the co-owning MLA KVCacheTensor (`abstract.py:29-43`):

| Region | Bytes | Status |
|---|---|---|
| Conv rows 0-2 | `[b·P, b·P + 73,728)` | Live |
| Conv rows 3-5 | `[b·P + 73,728, b·P + 147,456)` | Inactive (spec extra columns) |
| Recurrent state | `[b·P + 147,456, b·P + 2,244,608)` | Live |

Shapes (O, `entry_init`): conv `[6, 12288]` bf16 (SD); recurrent `[32, 128, 128]` fp32. No replayssm (2-tuple
kv_cache; `kda.py:443-445`).

**I:** the layer 0-2 recurrent states are equal while their conv digests differ. That is consistent with the conv
differences lying in rows 3-5, which hold stale bytes carried forward by every pre-copy. The digests cannot show it.

## Task 4. The reported first mismatch: rank 0, layer 7 indexer K, logical block 0, point 0, R01-early

| Check | Result |
|---|---|
| Physical ids | V1: 436 (r1), 762 (r2). V2: 116, 1,199 (O, eblk) |
| Inside the cached span | Yes: tokens 0-4,351 of n_cached 13,056 (O) |
| Addressing | Entry lid = `bt[g][req, 0:n·bpk:bpk] // bpk` (`p4_hooks.py:558-578`). Pages `b·17 … b·17+16` of the `[21675, 64, 132]` uint8 view (21,675 = 1,275 × 17; O). The writer's slot is `page·64 + pool%64`, with page = the block-table entry at pool granularity (`compressor_utils.py:41-56`) and pool = pos//4 (index_kpool 4, `config.json`; storage 4,352/4 = 1,088 = 17 × 64; O). The same block id gives the same bytes (S/O) |
| Byte range (layer-7 indexer KVCacheTensor, idx_page 143,616) | r1 `[62,616,576, 62,760,192)`; r2 `[109,435,392, 109,579,008)` |
| Record layout | Per 8,448-B pool page p: E4M3 values `[p·8448 + r·128, +128)` for r = 0-63, then fp32 scales at `p·8448 + 8192 + 4r` (`kpool_compress.py:391-392`, `:303-318`). Pool q of the block → page q//64, row q%64. No padding (17 × 8,448 = 143,616) |
| Tail ring vs pools | The tail view (group 1, `[1275, 2, 8, 128]` bf16) is an `as_strided` view at offset 0 of the same pages (`kv_cache_utils.py:1778-1784`; `attn_utils.py:295-320`). It would alias bytes `[0, 4096)` of a block **only** when that block id is held by the tail group. At point 0, 436/762 are group-0 blocks and no id appears in two groups (task 5). The digest covers the pool view only |
| Never-written bytes inside | Pools 511 (page 7, row 63: values `[67,200, 67,328)`, scale `[67,580, 67,584)`) and 1,022 (page 15, row 62: `[134,656, 134,784)`, `[135,160, 135,164)`) — 264 B of 143,616 (task 2) |
| Warm provenance | Built by W-V1-r1-R01 (uid 1, steps 1-3: chunks 0-2,044 / 2,045-4,089 / 4,090-4,351) and W-V1-r2-R01 (uid 12, steps 1,380-1,382). Same ids at logical 0 in the warm's and forced requests' rows. In between, read (not written) by V1-R01-close and -deep of the same replicate (O) |
| Prior holders (recorded keyed requests only) | 436: none before the warm. 762: V1-R03-close-r1 (KDA group 8) and V1-R03-deep-r1 (KDA group 2). Those are MLA-co-owned tensors, not indexer tensors (O, `alias_history.out`) |
| Same storage via another view at point 0 | No (task 5) |
| Same-block MLA KV | `layers.7.self_attn.attn`, logical 0, ids 436/762, also differs. Layer-7 MLA KVCacheTensor, co-owned by KDA layers 9, 10, 12, 13, 14, 16, 17 and draft layer 46 (`kv_cache_utils.py:1765-1773`). Bytes `[b·2,297,856, (b+1)·2,297,856)`: 4,352 records × 528 B (512 E4M3 + 4 fp32 scales; `b12x kv_cache.py:1-14`), all written by the warm (slot ≥ 0 for every token, `b12x kv_cache.py:504-505`). No never-written bytes. 762's earlier KDA-9 state in this tensor is fully overwritten |

**Conclusion (S/O):** the first mismatch is a valid cached-prefix difference by addressing, layout and provenance.
- It is the first mismatch in the published enumeration and also under the valid-only scope (MLA, same block).
- **Not concluded:** that layer 7 is where divergence starts. Enumeration order is window → rank → tensor → block;
  it is not a temporal trace. In layer order, layer 4's KDA state is the first valid difference (task 1).

## Task 5. Same physical storage through several tensor views

| Family | Shares storage with | Double counting in the digests? |
|---|---|---|
| MLA KV (layer L) | KDA conv/recurrent of one layer per mamba group, and one draft layer, all on the same KVCacheTensor (`kv_cache_utils.py:1765-1773`). Block b means the same bytes in every co-owner | **Point 0: no.** No id is in two groups' rows (O: 0 repeats in 64 files) |
| KDA conv #0 and recurrent #1 (same layer) | Disjoint slices of one page (`abstract.py:29-43`) | No |
| Indexer K and tail ring (same layer) | The same indexer tensor; the tail rides at offset 0 of the page | **Point 0: no** (different groups and ids) |
| Draft (layers 45-49) | The MLA tensor i, for i < 5 | **Points 2-4: yes** (O, `alias_history.out`) |
| Across ranks | Separate GPUs | No |

**Draft row detail (O):**
- At points 2-4, the draft row of the early windows still lists sliding-window blocks that are by then allocated to
  the MLA group (16 repeats) or to KDA groups 2 and 3 (16 and 12).
- Example: block 714 is MLA logical 4 and draft logical 10 in V1-R01-early.
- Those points therefore digest the same bytes through two views.
- **Removal records:** every group-9 row (44/44) contains ids that the same record also lists for another group.
- **I:** the worker-side table keeps stale ids for freed window blocks.
- This does not affect point 0.

## Task 6. What `_block_digest` computes (`p4_hooks.py:551-555`)

**The computation (S):**
1. `blk = t.index_select(0, pages)` copies exactly that view's elements for the ppb pages of the block.
   - Strided page padding and co-owner bytes outside the view are excluded.
   - ppb: MLA 68, indexer 17, others 1.
2. `w = blk.view(int8|int16|int32)` reinterprets the copy as signed integers of the element width:
   - uint8 → int8;
   - bf16 → int16;
   - fp32 → int32.
3. It returns **two int64 sums**: Σ w over every word, and Σ w[::3] over the words at flat index ≡ 0 (mod 3). It
   is wrap-free at these sizes.

**What it proves:**
- **Different sums ⇒ different bytes (S).**
- **Equal sums do not establish equal bytes.** These changes preserve both sums:
  - any permutation of words within the same residue class;
  - any +d/−d pair in the same class;
  - permutations of whole records or rows whose word count is a multiple of 3. Examples:
    - two 528-B MLA token records swapped (528 ≡ 0 mod 3);
    - conv rows (12,288 words ≡ 0) swapped or shifted.
- Slot-permutation addressing errors in MLA KV, and row shifts in conv state, are therefore exactly what this
  digest cannot see.
- The checksum is linear, not collision-resistant. No probability bound applies to equal sums.

**Where the published text over-claims:**
- `STAGE0-PROTOCOL.md:185`: "Equal digests make a difference very unlikely".
- The phrase "entry-digest equality" used as "state identical", in §4 G-entry ("equal digests for every covered
  tensor") and in the RESULT gate table "window-entry state equal". It means equal two-sum checksums of the
  allocated blocks.
- `STAGE0-PROTOCOL.md:71`: "Every KV-cache tensor of every group is digested per logical block". In fact:
  - only the first `NE_BLK` = 48 logical blocks are digested (`p4_hooks.py:113`, `:566`);
  - draft rows have 89-106 entries, so in the 6 deep/close windows the first 48 (all null) were digested and draft
    K/V was not compared at all (O);
  - null placeholders are skipped.
- Logical block 0 vs the physical null block (wording, task 8).
- Layers 0-3 "equal" in this audit means equal two-sum checksums only.

## Task 7. Can a valid differing slice be resolved from the retained digests?

**No.**
- For each (tensor, block, point) only the two sums are retained.
- A cached block is read-only after its warm, so the later points add nothing for it.
- The per-layer row digests (`ldig`) exist only for forced-request rows, not for the warm rows that built the
  blocks.
- Which tokens or bytes of MLA L7 block 0 or of the KDA L4 state differ cannot be recovered.
- Nor can it be decided whether the KDA conv differences lie in live rows 0-2.

**Resolution:** the live slice cannot be resolved from the retained records without a small new capture (assessor 22ax, section 3). Any such capture is proposed and approved separately; its design is not part of this audit.

## Task 8

Corrected sentences are in `WORDING-CORRECTIONS.md`; the originals were not edited.

## Residual inferences and limits

- **(I)** Layers 0-3 equal and 4+ different relies on equal two-sum digests.
- **(I)** The stale-row explanation for the layer-0/1/2 conv differences.
- **(I)** "Probably zero" for the never-written indexer pools in this probe.
- **History:** prior holders come from keyed requests' removal records only. These are not recorded:
  - unkeyed or sentinel requests;
  - start-up warm-up;
  - mid-life align and sliding-window frees.
- **Instrumentation** is identical in both repeats, but it is not excluded as a contributor (assessor priority 4).
- **Not examined:** points 1-4 beyond the aliasing check; ranks other than the stated ones for the focus slice;
  operation-level causes.
