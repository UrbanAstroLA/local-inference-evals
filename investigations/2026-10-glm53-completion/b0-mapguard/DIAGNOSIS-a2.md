# run-a2 diagnosis (CPU only, from stored outputs and source; no GPU)

Inputs: `run-a2/` (guard logs `mapguard/mg_p371.jsonl` rank 0, `mg_p405.jsonl` rank 1, `ms_p275.jsonl`
EngineCore), `run-a2.out`, B0 source (`b0-cache/src/vllm`). `(inf.)` = inference not verified by execution.
The original `run-a2/verdict.{json,txt}` are untouched; the re-analysis is `run-a2/verdict-postA2.{json,txt}`.

## 1. Why the workload never started: a runner fault (mine)

- `run-a2.out:23`: `INVALID: guard error during start-up -> abort`, so rc=8.
- The run produced **no `guard_error` record** in any file.
- **What happened.** The early-abort test was `grep -h "guard_error" /tmp/mapguard/*.jsonl`. That also matches the key `"guard_errors": 0`, which every readout carries.
- **When.** The first periodic readout at 01:04:19 UTC (host step 6) contained the key, so the 10 s readiness loop aborted at about 01:04:27 (cleanup at 01:04:27).
- **Result.** No client ran: no `client.jsonl`, no probes, no phase ran. The wall cap was not involved.
- **What the guard did see** is the stack's own release warm-up, which runs before `/tmp/glm53-release-ready`:
  - 84 requests: 82 short, one of about 8 k tokens and one of about 16 k;
  - up to 16 in flight;
  - 570 steps per rank between 01:03:59 and 01:05:26;
  - it kept running during the about 1 min `stop.sh`.
- The last readout was at host step 363/364 (01:05:08). No exit readout happened (SIGTERM), so steps 364-570 are unread.
- **Fix (post-inspection, runner).** Match the quoted record kind `"\"guard_error\""` in all three places.
  - On the a2 logs: old pattern 8 hits, new pattern 0.
  - A positive control line gives 1 hit.

## 2. `idx_r_dec.rtable` (86,736) and `idx_r_pf.rtable` (84): guard expectation fault, not an engine finding

**Every stored record has `consumed == 17 × (request's block)` at column 0.** Samples (rank 0, layer 3; blocks from the EngineCore `KVCacheManager` log):

| Site | Step | Request | Blocks | Length (pools) | Column | Consumed | a2 expected (block id) | Corrected expected (17·blk[c//17] + c%17) |
|---|---|---|---|---|---|---|---|---|
| idx_r_dec | 8 | cmpl-8282a307… | [1] | 10 | 0 | 17 | 1 | 17 |
| idx_r_pf | 45 | cmpl-81ceba7f… | [1] | 1,022 | 0 | 17 | 1 | 17 |

Single-block requests over the whole run show (consumed, block) = (527, 31), (1156, 68), (595, 35), (1615, 95), … (4250, 250): always 17×.

**What the reader consumes (source).**
- `AttentionGroup.create_metadata_builders` (`v1/worker/utils.py:302-311`): for a compressed spec, the builder's spec block = `compressed_kernel_block_size(spec) × compress_ratio`.
  - `compressed_kernel_block_size` (`utils.py:55-68`) with `PAGED_MQA_PAGE_SIZES = (32, 64)` (`utils/deep_gemm.py:35`) gives 64 pools for a 1,088-pool storage block. So the builder's block is 256 tokens and its storage is 64 pools.
- `DeepseekV32IndexerMetadataBuilder.build` (`indexer.py:1089-1096` prefill/write, `:1268-1290` decode) translates the shared 64-token table with `factor = 256 // 64 = 4`: `block_table[:, ::4] // 4` = **17·block + j**. That is one entry per 64-pool index page, 17 pages per 4,352-token block.
- The B12x paged top-k (`sparse_attn_indexer_kpool.py:942-955`) and `cp_gather` (`:566`) consume this table.

**Why writes were unaffected.** The compressed slot mapping uses the same page table with storage 64: (17b + j)·64 + p%64 = 1088·b + pool. That equals the guard's write expectation, and the writes showed 0 mismatches.

**Why the CPU tests missed it.** My emulation hard-coded the translation (factor 68, storage 1,088), copying the guard's own assumption, instead of deriving it with B0's builder code. The pool-length expectation (`rlen`) was right and recorded 0 violations at both sites.

**Fix (post-inspection, guard).**
- `check_table_read` now expects page ids (`Geo.ipage = 64`, `Geo.ipb = 17`); the high-address threshold is scaled to 935 × 17.
- Registration verifies the page with B0's own `compressed_kernel_block_size` on the real index spec; any mismatch is a registration issue, which is INVALID.
- The emulation now derives `IDX_FACTOR = 4` and `IDX_STORAGE = 64` from B0's `MLAAttentionSpec.copy_with_new_block_size` and `compressed_kernel_block_size` (test A6).

**Regression test (`test_a2_regression.py`)** on the real-geometry clean scenario: the a2 guard module flags `idx_r_dec.rtable` with every consumed value a multiple of 17 (the a2 signature); the fixed guard flags nothing.

**Re-check of the stored data.**
- Every stored idx_r record checked against the corrected expectation agrees: rank 0 7,672/7,672, rank 1 7,758/7,758. Records from the runner's own warm-up are not joinable and were not checked.
- The device counters themselves cannot be recomputed. Only the first wrong column per row was stored, and only 8,192 records were kept. **So the index-K read sites stay INCOMPLETE for a2, not CLEAN.**

## 3. The INVALID items

**In-graph executions 362 vs 363 (rank 0) and 364 (rank 1): analysis fault.**
- At the last readout every owned layer's armed MLA-write count is 362 on both ranks. That equals the number of device ring entries (written in-graph at the first owned layer), so the counts are exactly consistent on the device.
- The host step logs reach 363 and 364 because the readout's side-stream copy is unordered with the engine stream (by design, as in the tail guard's `_snapshot`). A forward that was enqueued and logged had not yet executed.
- **Fix P1:** count armed forwards from the device ring in the window. P3: step-level checks use only device-executed steps.
- With the fix, ownership is exact on both ranks: 356 forwards in the window × every owned layer, and no write from a non-owned layer.

**No pre-workload baseline readout: consequence of item 1.**
- With no client the frozen rule had no start time.
- **Fix P2:** with no client, the window starts at the first non-`_warmup_` step, with the last readout before it as baseline. In a2 that is readout 1 at host step 6, 6.5 s before the first real step and 12.5 s after the last runner warm-up step.
- **Fix P6 (frozen now, before any rerun data):** the window always starts there, so the stack's release warm-up counts as serving traffic. The guard now takes an exact `baseline` readout, after a device sync, just before the first real request (test I4).

**Output identity: not a measurement.** The probes never ran, so it reads NOT RUN (fix P5), not UNVERIFIED.
- On cross-session repeatability: short greedy outputs were identical *within* every earlier session (s1, s3, rel; the gate's `probe.jsonl`). The release's long-context greedy non-repeatability is documented (gate P1).
- Cross-session identity for an identical image and configuration has never been measured. The s1 reference was also taken on a different image (the check image with other observers).
- So a mismatch in a rerun would be ambiguous between session-level non-determinism and a guard effect. The frozen rule already handles that: the label becomes instrumentation-unverified and a guard-off probe session is the follow-up. (inf.) Batch-1 short greedy outputs on identical graphs are expected to repeat, but this is not established.

## 4. D7 (index-K pool completion at unaligned prefill chunk starts): confirmed in serving

- **10 events per rank, all ranks and layers alike:** rank 0 has 60 = 10 × 6 layers, rank 1 has 50 = 10 × 5.
- All were in eager steps (`cg none`) with a single prefill request (the first prefill row of the slice), so no `pool_cross` was possible here.

| Request | Chunk start (start % 4) | Missing pool (positions) |
|---|---|---|
| tag 19 (about 8 k) | 2045 (1), 4090 (2), 6397 (1) | 2044-2047, 4088-4091, 6396-6399 |
| tag 20 (about 16 k) | 2045, 4090, 6397, 8442, 10749, 12794, 15101 | the same two offsets (2,044 and 4,088) in every 4,352-token block |

- **Mechanism.** Every long-prompt chunk is **2,045** tokens: `input_budget − draft_slots` = 2,048 − 3 (`scheduler.py:563`, also `:690, 916, 1116`). Mamba alignment adds a 262-token chunk to reach each 4,352 boundary (`scheduler.py:378-440`). So chunk starts are 4352k + {0, 2045, 4090}.
- `_kpool_compress_insert` masks local index < 3 (`sparse_attn_indexer_kpool.py:129-130`), so the pool completing in the first 1-3 tokens of the chunk is never written. That is 2 of 1,088 index-K pools per 4,352-token block of every long prompt.
- **Every unaligned first-row start in the run produced an event; aligned starts (4,352k) produced none.**
- The same step's prefill read (`idx_r_pf`, step 45: length 1,022 pools) includes the unwritten pool 511. (inf.) The record still holds the zeroed contents of a new block.
- The serving cause (the 2,045-token budget) differs from the simulation's assumed leftover budget; the effect is the one predicted, case (a).
- **Scope (inf.):** this affects long prompts and any resumed or long prefill. GPQA prompts (124-600 tokens) are one chunk, so they are unaffected except after preemption.
- **Effect on selection or completion: not measured.**

## 5. `pos_bounds` (4), `position_exact` (6), `host_attribution` (38): runner warm-up artefacts, not findings

- All of them are in steps 1-6, whose requests are `_warmup_0_`…`_warmup_2_`. Those are fake `SchedulerOutput`s hand-built by the runner (`v1/worker/gpu/warmup.py`).
- Examples:
  - step 4: p0 = 8 vs lo/hi = 10/13;
  - exact join at step 3: p0 7 vs 9;
  - `tail_gt_1`: 2-3 tail blocks per fake request, the same warm-up block-count behaviour found in amendment 10d/T1.
- The frozen design excluded these steps through the pre-client baseline. With no client they leaked in.
- With P2/P6, real traffic shows:
  - 0 `pos_bounds`;
  - an exact position join on 3,661 rows per rank with 0 mismatches;
  - transmission 356/356 steps joined with 0 digest mismatches;
  - 0 host attribution issues;
  - no block held by two groups at once in the EngineCore snapshots.

## 6. Classification

| Item | Class | Evidence |
|---|---|---|
| Abort at readiness (rc=8) | runner fault | substring grep on `"guard_errors": 0`; 0 guard_error records |
| `idx_r_dec.rtable`, `idx_r_pf.rtable` | guard expectation fault (unit) | consumed = 17 × block in all 15,430 stored records; `utils.py:302-311`, `indexer.py:1089-1096` |
| 362 vs 363/364 executions | analysis fault (readout timing) | device ring = 362 = every layer's count |
| missing baseline | consequence of the abort; analysis rule gap | no client |
| output identity | not measured | no probes ran |
| `idx_w_pf.missing_pred` (D7) | **engine finding, confirmed in serving** | 10 events, all at unaligned chunk starts 2045/4090 mod 4352 |
| pos_bounds / position_exact / host_attribution | runner warm-up artefacts | all in `_warmup_` steps 1-6 |

## 7. Re-analysis of the stored data (`verdict-postA2.*`, analysis P1-P6, labelled)

| | Original (frozen analysis) | Corrected (post-inspection) |
|---|---|---|
| Verdict | VIOLATION + INVALID items | **VIOLATION** (D7 only), no INVALID |
| Violations | missing_pred 60/50; rtable 86,736/72,280 and 84/70; pos_bounds 4/4; position_exact 6/6; host_attribution 38/38 | `idx_w_pf.missing_pred` 60 / 50 |
| Instrumentation | invalid (ownership counts, baseline) | valid: exact ownership, FULL replays 329, joins 1.0 |
| Incomplete | C2, C4, C6b, all C7, C9, C11, C12 | C2 (6/20), C4 (0/10), C6b (22/60), C6c (5/6), all C7 (0; ids ≤ 528), C8 idx_r_dec/idx_r_pf (void) |
| Met | — | C1 329, C3 52, C5a 53,925, C5b/C5c 7,993/10,475, C6a 100, C8 mla_w 205 k, idx_w_pf 133 k, idx_w_dec 72 k, tail 133 k/72 k, mla_r 205 k, C9, C10 1.0/1.0, C11 68, C12 66 |

On the real release warm-up traffic, both ranks, shipped graphs: MLA writes and reads, index-K writes, tail seed and stash, positions, row split, transmission and rank ownership are **clean**. That is about 205 k MLA token writes and reads and 133 k index-K prefill tokens per rank. **Not covered:** high-address blocks (the warm-up reached block 528), prefix hits, 5-8 in flight, and the index-K read tables.

## 8. Rerun: what it must cover and what it costs (remaining 0.5789 GPU-h = 1,042 s on 2 GPUs)

**It must cover only what is missing:**
- C7, all high-address sites: ids ≥ 935;
- C4 prefix hits;
- C2 mixed steps;
- C6b/C6c (5-8 in flight, transitions);
- the corrected index-K read checks (C8 idx_r_*, C7_high_idx_r);
- the probes (output identity).

The release warm-up traffic in the window (P6) re-supplies C1, C3, C5, C6a, C11, C12.

**Reachability (`reach/out-rerun-acc2.6-pre528.json`; allocator simulation, (inf.)).** With the free-queue head advanced to 528 by the release warm-up (as observed in a2):
- group 0 and the tail get ids ≥ 935 at step 92 of the warm phase (about 55 s in);
- MLA + index-K hold 14-21 high blocks through T0-T1;
- 20 prefix hits.

**Option A: within 0.58 GPU-h, no invalid-run allowance.**
- Command:

```bash
APPROVED_GPU_H=1.0 MIN_WALL=960 RESERVE=95 PHASES="P0 warm T0 F0 T1 S P1" \
  <runs>/b0-mapguard/run/run_mapguard.sh 3
```

- Timeline:
  - cap 1,042 s;
  - readiness about 800 s (s1: 799 s);
  - client until 947 s, about 140 s: P0 5 s, warm 73 s, T0 20 s, F0 9 s, T1 20 s; S and P1 only if time remains;
  - watchdog stop at 967 s, GPUs released by about 1,035 s (a2 stop: 66 s).
- Cost: about 0.575 GPU-h, reaching the 1.0 total.
- **Risk.** If readiness takes ≥ about 850 s, T0/T1 are cut and C4/C7 can come out INCOMPLETE. No retry would be possible.

**Option B (recommended): add 0.25 GPU-h (cumulative 1.25).**
- Command:

```bash
APPROVED_GPU_H=1.25 <runs>/b0-mapguard/run/run_mapguard.sh 3
```

- The full frozen workload with the standard 24 min cap: expected 0.55-0.63, hard maximum about 0.83 within the gate.
- It covers every condition with the margins of the original proposal; C2/C6c especially benefit from the D/S/Q phases.
