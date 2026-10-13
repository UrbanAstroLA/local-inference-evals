# B0 live mapping guard: proposal (PLAN.md amendments 17 and 17a)

**Status: prepared on CPU and ready to launch. Not run on a GPU.**
- The guard, the independent expected mapping, the CPU tests, the frozen invariants and coverage minimums, the instrumentation checks and the runner are all in place.
- Budget gate: 1.0 GPU-hour total (amendment 17a), which must also cover one invalid run.
- The coordinator verifies and launches (section 10). No GPU time has been used: only CPU containers and a `docker build`.

**Notation.**
- `(inf.)` marks an inference that has not been verified by execution.
- Source citations are to the B0 tree as extracted from the image: `vllm/…` = `b0-cache/src/vllm`, identical to release 0.9.1 plus the D1-D4 files.
- `S` = `<runs>/b0-mapguard` (this directory).

---

## 0. Read this first: a CPU-derived prediction for the session

While building the index-K write check, a hazard in B0's index-K prefill write turned up. It is confirmed with B0's real kernels on CPU; whether serving actually reaches it is inferred.

**What the kernel does.**
- `_kpool_compress_insert` (`sparse_attn_indexer_kpool.py:102-147`) pools the batch tokens `t-3..t` of the prefill slice.
- It masks local index < 3. Its docstring states the assumption behind this: "Assumes pool-aligned chunk starts".
- The scheduler does not align prefill chunk boundaries to 4 tokens. Mamba "align" splitting aligns only to 4,352-token blocks (`scheduler.py:378-440`), and no kpool alignment exists (grep; (inf.) from source).
- Leftover budget after another request's final chunk can therefore leave a chunk ending, and the next chunk starting, at `p0 % 4 != 0`.

**Confirmed on CPU** with the real B0 functions under the Triton interpreter (`tests/test_kpool_unaligned.py` → `tests/out/kpool-unaligned.json`, `ok: true`):
1. **First prefill row of the slice, unaligned start.** The index-K pool that completes in its first three tokens is never written. Here: pool 511 at positions 2,044-2,047.
2. **A later row starting unaligned.** Its first completing pool is compressed from the previous request's last tokens. The stored record is byte-equal to pooling batch tokens 4..7 and differs from pooling the request's own tokens.
3. **Control.** Aligned pools are written correctly.

**Reachability.** The allocator simulation of this workload (`reach/out-mapguard-acc2.6.json`, approximate scheduler, (inf.)) produces 73 unaligned prefill chunk starts, about 10-17 in each long-context phase. Example: T0, step 128, start 57,703. That is the r1 pattern: 56,576 cached tokens + 1,127.

**The guard measures both cases.**
- `missing_pred`: a pool that should be written but is masked by the kernel predicate.
- `pool_cross`: a pool fed from another request's tokens.

**Pre-registered expectation.** If serving reproduces the simulation, the session returns **VIOLATION** on these two counters (site `idx_w_pf`), whatever the mapping-builder classes show. They are reported as their own class: "index-K pool records at unaligned chunk starts". The effect on selection or completion is **not** claimed.

---

## 1. Design

**Hook sites.** Eight device check sites plus host hooks.

The device sites are graph-safe and run at the point of consumption, with the exact tensors the kernels receive:

| Site | Consumed mapping (what the kernel reads) | Hook location |
|---|---|---|
| `mla_w` | MLA writer slot mapping (`forward_context.slot_mapping[layer]` = persistent `BlockTables.slot_mappings[0]` view) | wrapper on `B12xMLASparseImpl.do_kv_cache_update` (monkeypatch, no diff) |
| `pos` | positions and query_start_loc buffers consumed by the forward (persistent `InputBuffers`); per-row first position into a device ring | inside `mla_w`, first owned layer per rank |
| `idx_w_pf` | index-K pool slots for the prefill slice + the kernel's own write predicate + the pooled-token window | before `_kpool_compress_insert` (`sparse_attn_indexer_kpool.py:427`) |
| `tail_w_seed` | kpool tail slots for prefill + the seed kernel's predicate | before `kpool_seed_tail_cache` (`:471`) |
| `idx_w_dec`, `tail_w_stash` | grouped `[B, lmax]` `dec_slot` / `dec_pos` / `dec_tail_slot` passed to the batched decode kernel | before `kpool_decode_update_and_maybe_write_cache_batched` (`:825`) |
| `idx_r_dec` | B12x paged top-k page table and pool lengths (DeepGEMM fallback too) | before `_run_b12x_paged_topk` (`:942`) and `fp8_fp4_paged_mqa_logits` (`:968`) |
| `idx_r_pf` | prefill `cp_gather` chunk page table and per-request pool lengths | before `ops.cp_gather_indexer_k_quant_cache` (`:566`) |
| `mla_r` | B12x selected physical slots (`page_table_1` after `_compact_dropped_selection` and masking) and `nsa_cache_seqlens` | after `_mask_page_table_after_nsa_len` (`b12x_mla_sparse.py:3043`) |

**Host hooks.**
- *Worker:* the runner's `execute_model` (ledger update; disarm and step log after the forward), `prepare_attn` (expectation upload), `initialize_kv_cache` (registration), the FULL/PIECEWISE replay wrappers, and `RequestState` add/remove for row history.
- *EngineCore:* `Scheduler.schedule` and `update_from_output`.

**How the device checks work (D2 tail-guard architecture).**
- One guard-owned expectation buffer `E` lives at a fixed address. It is refreshed by one H2D copy, from a ring of four pinned staging buffers, in `prepare_attn` before every real forward. It is disarmed (`E.armed = 0`) on the same stream after the forward.
- Checks are fixed-shape torch ops only: no host sync, no host tensor creation, no boolean indexing.
- Counters `[site × class]`, per-layer execution counters and bounded violation records (capacity 8,192) live in guard buffers. A background thread reads them out on a side stream every 20 s and on a DUMP trigger.
- **Stale graph replay.** A replayed graph that consumed a stale mapping reads the same stale buffer the kernel reads. That buffer differs from the current expectation, so a stale replay shows up as a mismatch.
- **Inert by default.** `GLM53_MAPGUARD` unset → nothing is patched. Tested: `tests/out/image-smoke-unset.json`.
- **Write volume.** Engine tensors are never written (test G2). Records are written only on violation.

**Files and image.**
- Guard: `guard/mapguard_hooks.py`. It extends the D2 tail guard's mechanisms: `record_rows`, poller, snapshot and DUMP trigger, and the seed-kernel predicate from `kpool_tailguard.write_mask_and_blocks`.
- Hook lines: `build/mapguard.diff`, 135 lines in three files. Every line is behind `_MAPGUARD_ON`.
- Derived image: `local/glm53-v0.9.1-b0-mapguard:local`, built FROM the B0 tag after checking that the tag is image `sha256:08d1cb90…`.
  - Build gates: B0 file hashes before; patched-file hashes, `py_compile` and marker counts after.
  - Ancestry check: B0's layers are a prefix of the derived image's layers.
  - The D2 files (`indexer.py`, `mamba_hybrid.py`) are B0's, byte-identical.

## 2. Independence: how scheduler truth reaches the workers, and why it is not the mapping builder

**1. EngineCore holds the truth.**
- The KV cache manager allocates ids per request and group.
- `SchedulerOutput` carries them to both workers (`scheduler.py:1176-1185, 1490-1530`):
  - new or resumed requests: the full per-group lists from `KVCacheManager.get_blocks(..).get_block_ids()`;
  - running requests: the ids appended this step.
- The guard's EngineCore hook logs, per `schedule()` call:
  - a step fingerprint (scheduled counts, finished ids and the scheduler's `num_computed` per request, so decode streaks do not collide);
  - a digest of every scheduled request's group-0/1 lists, read **directly** from `kv_cache_manager.get_block_ids` (full lists when they change);
  - every 16 steps, a pool-wide exclusivity snapshot from the managers' `req_to_blocks` (a block held by two groups at once is a violation).
- `update_from_output` logs draft and accepted counts per request.

**2. The worker rebuilds the lists from the stream alone.**
- A guard-private Ledger rebuilds request → per-group block lists from the `SchedulerOutput` stream only: finished and preempted → free; new and resumed → overwrite; running → append.
- It never reads the runner's `RequestState` rows, `BlockTables` (staged writes, persistent or gathered tables), the slot-mapping or compressed-mapping kernels, the metadata builders or `idx_mapping`.
- The only runner value used is the batch **order** of request ids (`input_batch.req_ids`). That order is a sort of the scheduler's own keys (`sort_batch_req_ids`). Per-row token counts are compared with the scheduler's counts on the host.

**3. The transmission is checked offline.**
- The worker digest is joined to the EngineCore digest by fingerprint.
- Test I3 (`tests/out/image-smoke-device.json`): the guard's real EngineCore hook, run against the real B0 `KVCacheManager` over a 12-step scenario with prefix hits and completions, equals the worker ledger digest at every step (12/12 joined, 12/12 equal).

**4. How the expectation is built.**
- It combines the ledger lists, logical positions, the frozen geometry (verified at registration against `BlockTables` and `kv_cache_config`) and the independently recomputed rank-ownership rule (`split:N → int(layer ≥ N)`, as in `placement.py:_parse`).
- *Positions* are consumed values, checked three ways:
  - contiguity per row;
  - in-graph bounds `[num_computed − previous-step drafts, num_computed]`, which is the async-scheduling window (`scheduler.py:1836-1856`);
  - exactly offline: first position == `num_computed − drafts rejected in that request's previous step`, joined to the EngineCore rejection log.
- *Speculative tokens* (the last `nspec` of a verify row) are counted separately. They must land in the request's own allocated blocks at the exact slot, like committed tokens. Rejection is classified offline from the next step's first position, so a draft later rejected is never a "committed-state" claim (N4).
- Mamba/sliding-window lists in the ledger can be stale (those managers drop blocks without a `SchedulerOutput` delta). They are used only to label a wrong block id (wrong group), never as an expectation.

## 3. Invariants (frozen)

**Writes** (`mla_w`, `idx_w_pf`, `idx_w_dec`, `tail_*`), for every active token:

| Counter | Meaning |
|---|---|
| `mism` | consumed slot ≠ expected slot. Labelled by where the consumed block belongs: another live request in the same group (foreign), no live holder (stale, after free or reuse), another group (wgroup), the null block, the right block at the wrong record (ownoff), or out of range |
| `missing` | an expected write is absent |
| `missing_pred` | an expected write is dropped by the kernel's own predicate |
| `pool_cross` | an index-K pool is fed from tokens outside the request's own `p-3..p` |
| `unexp` | a write the expectation does not call for: unallocated column, non-completing pool, null tail block, tail token outside the seeded set |
| `inact` | a padding or inactive token writes a non-null block |
| `shared` | a write into a block shared by several live requests (prefix-cache block) |

**Positions:** `pos_contig`, `pos_bounds` (in-graph); the exact join offline; `qsl` (consumed `query_start_loc` ≠ the scheduler's row split).

**Reads:**

| Counter | Meaning |
|---|---|
| `rtable` | an index-K page-table column needed for the row's length ≠ the request's own block |
| `rlen` | index-K length ≠ `(p+1)//4` pools |
| `sel_foreign` | a selected MLA slot lies outside the request's own group-0 blocks |
| `sel_causal` | a selected MLA slot's logical position is > p |
| `masked_live` | an entry ≥ `nsa_len` is not −1 |
| `sel_len` | `nsa_len` > p+1 |

**Grouping and rank:**
- `grouppos`: a grouped decode position ≠ the token's consumed position.
- Rank ownership: each rank's MLA-write executions come only from its owned layers, and every owned layer executes exactly once per armed forward.

**Scheduler side:** a block held by two groups at once; transmission digest mismatch.

**Informational, not violations:** `sel_hole` (an unfilled selection inside `nsa_len`); `inact_read` (padding rows reading; outputs discarded). Steps before the workload (the release's own warm-up) are excluded and reported. The tail guard (T1) found the runner warm-up's stale columns there.

**Snapshot on violation.** Each record carries site, rank, layer, step, batch token, row, request tag, position, consumed, expected, flags and holder tag/group. Row add/free history and per-step rows (request id, runner row, first position, bounds, block counts, max block) are in the step log.

## 4. CPU tests

**Where they run.** In the derived image, `--network none`, `CUDA_VISIBLE_DEVICES=` and `TRITON_INTERPRET=1`; no GPU. Runner: `tests/run_cpu_tests.sh`; outputs in `tests/out/`.

**The scenario harness (`tests/b0emu.py`).**
- Block ids come from the **real** B0 `KVCacheManager` (10-group B0 config), pre-cycled so that ids ≥ 935 are handed out.
- Real `SchedulerOutput` dataclasses, with async semantics emulated.
- Every consumed mapping is produced by **B0's real code**: `_prepare_pos_seq_lens_kernel`, `_gather_block_tables_kernel`, `_compute_slot_mappings_kernel`, `get_compressed_slot_mapping`, `compute_kpool_tail_slot_mapping`, `_prepare_uniform_decode_kernel`, `_scatter_decode_tokens_by_request`, `_logical_topk_to_physical_slots`, `_compact_dropped_selection`, `_mask_page_table_after_nsa_len` and `sort_batch_req_ids`.
- The guard's real hook functions check the result.
- Faults are injected only into the runner emulation.

**Results: 27/27 in `test_mapguard.py` (`tests/out/cpu-results.json`, 1,534 s), plus image smoke 3/3 (I1 with the variable unset; I2 and I3 with it set) and the kernel check 1/1. Total 31/31 passed, 0 failed.**

| Group | Tests |
|---|---|
| Negative controls (must **not** flag) | N1 clean multi-step scenario at every site: prefill/decode interleaving, prefix hit, completion and row reuse, ids ≥ 935, 0 guard errors; N2 FULL-graph-style padded rows and tokens; N3 masked selection entries and `nsa = 0` padding; N4 speculative slots later rejected (96 speculative writes, 20 rejected tokens); N5 mixed steps with a prefix hit and row reuse; N6 high-address blocks; A2a exact offline position join (33 rows, 0 mismatches) |
| Injected faults (must flag the stated class) | F1 wrong-request block (mism, foreign); F2 stale block after free/reuse, i.e. a missed append on a reused row; F3a wrong position, prefill row (`pos_bounds`); F3b wrong position inside the in-graph window, caught by the exact offline join; F3c right block, wrong record (ownoff); F4 row-reuse error, i.e. the previous occupant's runner row; F5a wrong group (an MLA write into the request's own KDA block → wgroup); F5b wrong rank ownership (a non-owned layer writing; an owned layer missing); F6 stale replay (the previous step's mapping consumed); F7 unaligned chunk start (`missing_pred` / `pool_cross`); F8 a padding token writing a live block; F9 the tail-ring T1 signature (generic mapping + stale column); F10 MLA selection non-causal, foreign and masked-live; F11 row-split mismatch; F13 index-K read with a wrong length and a foreign page; A3 transmission digest mismatch |
| Lints | G1: no host-sync op (`.item`, `.tolist`, `nonzero`, boolean indexing, `.cpu`, …) inside any device check, so graph-capture safe by construction; G2: no consumed engine tensor modified |
| Analysis rules | A4/A5: CLEAN / INCOMPLETE naming the condition / INVALID / VIOLATION precedence |
| Image | I1 inert when unset; I2 all hooks installed when set; I3 EngineCore digest (real `KVCacheManager`) == worker ledger digest at every step |
| Kernel | `kpool-unaligned`: section 0, real `_kpool_compress_insert` |
| Pipeline dry run | `tests/test_pipeline_dryrun.py` + the frozen analysis on its output (`tests/out/dryrun/verdict.txt`) |

**Pipeline dry run.** It exercises the guard's real `_register_runner`, `_upload`, step-log and `poll` paths, plus the EngineCore hook, for both ranks.
- No violations.
- Exact per-layer ownership counts.
- Transmission join 1.0; exact position join on 100 % of rows.
- The only INVALID items are the expected CPU artefacts: no server log, and rank 1's baseline (the dry run ran the two ranks one after the other).

**Bugs found and fixed during the CPU work** (before any GPU data):
- Stale draft/KDA ledger lists polluted the ownership maps; fixed with a two-pass map.
- A bool-subtraction error in the ring write.
- Scheduler and worker records went to the same file when both hooks ran in one process; now separate files.
- Test-harness bugs: injections in F1 and F10 cancelling out; the record layout used by the kernel test.

## 5. Workload (existing fixtures only) and reachability

**Client.** `run/client_mapguard.py` reuses `bundle/gate/client_bundle.py`'s request functions. Seeds are distinct and recorded per request.
- **Before any traffic:** a guard baseline readout.
- **P0:** short greedy probes q0-q3: chat, temperature 0, seed 4242, 96 tokens, each alone.
- **warm:** the 8 stage-2 active-reasoning prefixes, 1 token each.
- **T0 F0 T1 F1:** T = 8 × 200 tokens (5-8 in flight); F = 4 × 128 (≤ 4 in flight).
- **D:** drain 8 → 1.
- **S:** staggered admission, mixed prefill/decode.
- **Q:** 16 GPQA prompts, 8 in flight, continuous admission, `max_tokens 1024` (bounded continuations, short contexts).
- **P1:** the probes again.
- After every phase: a DUMP trigger. The client skips a phase rather than overrun the deadline; a missing phase shows as INCOMPLETE.

**Reachability without changing B0.**
- Driven by the real allocator with this workload (`reach/sim_reach_mapguard.py`; real stage-2 token ids, so prefix hits are the real ones), group 0 (MLA + index-K) first receives ids ≥ 935 at T1, about 20 steps in (step 311 of 1,517). The tail group follows one step later.
- The same holds at mean acceptance 2.0 and 3.2: steps 350 and 288, both in T1.
- MLA + index-K hold 5-11 blocks ≥ 935 in T1-Q; the tail holds 2-8. Ids reach 1,265 (MLA) and 1,274 (draft).
- Wall time: warm, T0, F0 and 20 steps of T1 add up to about 1.8 minutes of workload (s1 phase timings, (inf.)).
- **`--num-gpu-blocks-override` is therefore not needed and is not used.** It would also change B0's effective configuration (the non-default-args fingerprint) and its scheduling capacity.
- Prefix hits: 32 requests, all aligned to the 4,352-token block (mamba align). Partial-block copy-on-write is **not reachable** in this configuration (sim: 0; (inf.) from `MambaManager`), so it is not covered.

## 6. Coverage minimums (frozen in `analysis/analyze_mapguard.py: COVERAGE`)

**Rules.** Every per-rank condition must be met on **both** ranks. A condition not met makes the result **INCOMPLETE for that condition**, never CLEAN.

| Condition | Minimum | Expected by design |
|---|---|---|
| C1 forwards replayed as FULL graphs, checks armed | ≥ 300 / rank | about 1,000 decode-only steps (sim) |
| C2 mixed prefill + decode steps | ≥ 20 | T/S/Q admissions |
| C3 runner rows re-occupied by another request | ≥ 20 | 72 requests on 16 rows |
| C4 requests admitted with a prefix hit | ≥ 10 | 32 (sim) |
| C5a speculative MLA write checks (token × layer) | ≥ 20,000 / rank | about 90 k (inf.) |
| C5b / C5c drafts rejected / accepted (EngineCore log) | ≥ 2,000 each | about 8 k / 10 k (inf.) |
| C6a / C6b decode steps with ≤ 4 / 5-8 requests; C6c transitions across 4\|5 | ≥ 60 / ≥ 60 / ≥ 6 | F and D tails / T, Q |
| C7 matched accesses in blocks ≥ 935: MLA writes / index-K writes / MLA reads / index-K page reads / tail writes | ≥ 5,000 / 500 / 5,000 / 500 / 50 per rank | ids ≥ 935 from T1 onward |
| C8 tokens or rows checked per site: `mla_w` / `idx_w_pf` / `idx_w_dec` / `tail_w_seed` / `tail_w_stash` / `idx_r_dec` / `idx_r_pf` / `mla_r` | 200 k / 100 k / 20 k / 500 / 20 k / 20 k / 50 / 100 k per rank | about 0.5 M prefill tokens, about 24 k decode tokens |
| C9 every owned layer checked exactly once per armed forward | exact | — |
| C10 worker steps joined to the EngineCore record / rows with the exact position check | ≥ 0.95 / ≥ 0.90 | 1.0 in the dry run |
| C11 completed requests / C12 completed requests with < 2,176-token prompts | ≥ 40 / ≥ 8 | 72 / 24 |

**Not covered by design (stated, not priced):**
- the draft group (DFlash2 KV) and the KDA states (out of the amendment's MLA/index-K scope);
- partial-block copy-on-write (unreachable);
- DeepGEMM prefill logits bounds `ke` beyond the gathered rows;
- record **contents**, whether numerics or bytes (b0-cache covered the kernels);
- contention-dependent races beyond one session.

## 7. Verdicts and instrumentation checks (frozen)

**Verdict precedence.**
- **VIOLATION** if any invariant counter is > 0 in the workload window on either rank, or the scheduler side, transmission or exact-position checks fail. Violations are reported even if instrumentation is invalid.
- **INVALID** if any of the following:
  - guard errors;
  - a rank or the scheduler hook missing;
  - geometry ≠ 4,352 / 64 / 4 / 8 / 1,275;
  - adaptive verification on;
  - no FULL capture or replay;
  - in-graph check executions ≠ armed forwards × owned layers (proves the checks ran inside every replayed graph);
  - non-default args ≠ the B0 screen;
  - client errors.
- **INCOMPLETE** if a coverage minimum is not met (named).
- **CLEAN** otherwise.

**Does the guard alter outputs?**
- *By design:* engine tensors are read only (G2); there are no host syncs inside the forward (G1); the guard writes only guard buffers; one small H2D copy happens before each forward on the same stream. The guard allocates after the KV pool is sized, so `num_blocks` stays 1,275, and the analysis checks that.
- *Measured:* the P0/P1 probe hashes must equal the s1 B0-check-image record (q0 `e705960c…`, q1 `1854a962…`, q2 `3ecf3c5b…`, q3 `a997a171…`). Short greedy outputs were repeatable within every earlier session.
  - Identical → PASS.
  - Different → the mapping verdict stands but is labelled *instrumentation-unverified*. The follow-up would be a guard-off probe-only session of the same image, about 0.25 GPU-hours, start-up dominated, needing separate authorization.
  - Limitation: cross-session repeatability of short greedy outputs has not been measured for an identical image (inf.).

**No forced eager fallback.** These are INVALID conditions:
- FULL replays are counted on the host;
- in-graph execution counters must equal armed forwards per owned layer;
- `server.log` must show `Capturing CUDA graphs (FULL)`.

**Expected overhead (inf.).**
- *Decode (FULL):* about 230 extra small kernels per layer, about 1,400 per step on rank 0, roughly 3-4 ms per step. That is about 1.5 % at the long-context decode step (~0.25 s) and up to about 6-7 % at short-context decode (~0.055 s).
- *Prefill (piecewise and eager):* about 30-40 ms of host launch overhead per 2,048-token step, about 6 %.
- *Host:* ledger plus expectation build plus 74 KB H2D, under 1 ms per step.
- *Overall:* ≤ 8 % throughput. It is measured as T/F decode tokens per second against s1 (88.3 / 59.3 tok/s with s1's observers) and reported, not used as a criterion.
- *Memory:* guard buffers about 3 MB; eager transients under 100 MB, outside the profiled budget (headroom about 4.7 GiB per GPU).

## 8. Cost (occupied GPU-hours, two GPUs)

| Item | Estimate |
|---|---|
| Start-up to readiness (weights, graph capture, release warm-up): measured 11-13 min; s1 799 s with observers | 11-13.3 min |
| Workload (s1 phase timings: P0 5 s, warm 73 s, 2 × (T 19.5 + F 9.3) s, D 21 s, S 20.5 s, Q ~45 s, P1 5 s, readouts ~20 s), +5-8 % guard overhead | 4.5-4.8 min |
| Snapshot, stop and GPU release | about 1 min |
| **Expected per session** | **16.5-19 min = 0.55-0.63 GPU-h** (amendment 17: about 0.6) |
| **Per-attempt wall cap** (watchdog stops the server) | 24 min = **0.80 GPU-h** |
| **Budget gate** (all `b0-mapguard` cost lines, invalid runs included) | **1.0 GPU-h** |

**Invalid-run allowance, stated explicitly.**
- The runner refuses an attempt whose remaining budget allows less than 18 minutes of wall time (0.60 GPU-hours), because a complete session cannot fit in less.
- A second complete session therefore fits only if the first attempt consumed ≤ 0.40 GPU-hours. Examples: a start-up or capture failure, or a guard error. The runner aborts on any guard error **during start-up**, not only at readiness.
- A first attempt that runs to completion but is INVALID (about 0.6) leaves less than 0.6. **Any retry then needs new authorization.**
- A second attempt after an early failure gets at most an 18-minute wall cap. If start-up takes the full 13 minutes, that leaves about 2 minutes of workload, so the second attempt may return INCOMPLETE.
- No retry is automatic.

**Engineering.**
- Spent, all CPU: about 2.5 hours of agent wall time, including about 1.5 hours of interpreter test runs.
- Remaining before launch: the coordinator's verification, about 0.5 hour.
- The analysis runs automatically after the GPUs are released.

**Reused:**
- the D2 tail guard (architecture, readout, seed predicate);
- the bundle-gate client and runner (request functions, phase designs, pre-flight, budget, cleanup and identity patterns);
- the amendment 16d type-G pre-flight rule;
- `b0-cache` geometry and `sim_pool.build_config` (allocator simulation and CPU harness);
- verify13's idea of attributing requests through runner prepare hooks.

**New:** the independent ledger and expectation, the eight device check sites and the EngineCore hook. Each is an extension of the guard pattern, not a new tracing framework.

No part needed instrumentation beyond extending the guard.

## 9. Outcome mapping (amendment 17)

- **VIOLATION.** The records name site, rank, layer, step, request and row, position, consumed vs expected, and the owner of the consumed block.
  - Next: minimize and reproduce (for the section 0 class the CPU reproduction already exists: `tests/test_kpool_unaligned.py`), verify a correction, then B1 = B0 + that correction only, then propose a behavioural comparison.
  - Plausible corrections for section 0, not evaluated: align prefill chunk ends to kpool in the scheduler, or complete boundary pools from the tail ring inside `_kpool_compress_insert`.
- **CLEAN with the required coverage.** Retire the tested scheduling and mapping failure classes on both ranks under shipped graphs. No automatic completion pass. Next: the smallest mixed-quant MoE fidelity fixture. A clean run cannot exclude every contention-dependent race.
- **INCOMPLETE.** Identify the exact missing exposure from the per-condition table and price only that.
- **INVALID.** Report the cause. Retry only within the 1.0 gate as above, otherwise ask.

## 10. Files, hashes and launch

**Image:** `local/glm53-v0.9.1-b0-mapguard:local` = `sha256:1f55fe653b7910bbb8eb907ec8d09ad7b7ba64813d89097a95c07aff91b9fbc1`, pinned in `run/images.txt`. It contains B0 plus `vllm/mapguard_hooks.py` and the three hooked files (hashes in `SHA256SUMS`).

**Frozen file hashes:** see `SHA256SUMS` in this directory; key lines are in the summary.

**Launch command** (no sudo; the coordinator runs it):

```bash
APPROVED_GPU_H=1.0 <runs>/b0-mapguard/run/run_mapguard.sh 1
```

**What the command does:**
1. Budget gate: spent + this attempt's cap ≤ 1.0. Image-id pin. Pre-flight: no glm53 container; GPU0 ≤ 1 GiB, GPU1 ≤ 4 GiB, utilisation ≤ 5 %; refuse any GPU client other than desktop type-G gnome-shell, Xwayland or xdg-desktop-portal.
2. Starts B0's stack with the derived image and `GLM53_MAPGUARD=device`. Shipped graphs, no extra serve args, DFlash2 ×3.
3. Aborts on an engine or guard error during start-up. At readiness, requires the guard on both ranks and the scheduler.
4. Runs the client to a deadline of cap − 150 s.
5. On every exit path (EXIT/INT/TERM trap, plus a wall-cap watchdog): readout, logs, stop, cost line in `costs.jsonl`, `docker cp` of `/tmp/mapguard`, then the frozen analysis → `run-a1/verdict.{json,txt}`.

**Dry check without GPUs:**

```bash
APPROVED_GPU_H=1.0 DRY_RUN=1 <runs>/b0-mapguard/run/run_mapguard.sh 1
```
