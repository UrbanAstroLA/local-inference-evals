# P4 capture-p0: exact bytes of the differing cached state, and where they first differ (protocol)

**Status: attempt 1 ran on 2026-10-24 and read INCOMPLETE (see the amendments below); attempt 2 is prepared, not run. No GPU use is authorized by this document.** It is written, with its analysis and tests, before
any capture-p0 GPU step. The operator approves GPU time separately (section 7).

It follows the Stage-0 zero-GPU audit (`audit-s0/AUDIT.md`, task 7) and the independent assessor's 22ax disposition
(sections 3 and 5 and its decision table). Where the Stage-0 protocol wording and the audit differ, the audit holds.

## Amendments after session attempt 1 (2026-10-24; owner: the capture-p0 package author; diagnosis: the parent session)

Session attempt 1 ran on 2026-10-24 (pre-check PASS 0.005 GPU-h; session 0.465 GPU-h, rc 0, start-up gates passed,
0 hook errors). Its analysis read **INCOMPLETE: 0 of 6 replicates counted**, for two package defects named below.
The records of attempt 1 are kept unchanged (`runs/session-attempt-1/`, `../PRIVATE-capture-p0/session-attempt-1/`).
Text below that these amendments supersede is kept and marked **[A1]**-**[A4]**; nothing was silently rewritten.

**A1. The warm must be the full capture prompt (17,409 tokens), not 13,057.**
- A request that hits a cached prefix gets back one block fewer than it matches
  (`PROPOSAL-AMENDMENTS.md` A3, the EAGLE-type drop): `min(blocks cached, ⌊(P − 1)/4352⌋) − 1`.
- A 13,057-token warm caches 3 blocks, so the capture request got 2 (8,704 tokens; measured in all six replicates).
- The audit's sketch ("a warm of R01 truncated to its first 13,057 tokens is enough", `audit-s0/AUDIT.md` task 7)
  was wrong on this point: it computes blocks 0-2 identically, but it does not leave them hittable.
- Stage 0 worked because its warms were the reply's longest prefix.
- Now: the warm prompt is R01 ids [0, 17,409), identical to the capture prompt. It caches 4 blocks; the capture gets
  3 = 13,056 tokens, which is Stage 0's R01-early construction.
  - Warm chunking: 2,045 / 2,045 / 262 for each of 4 blocks, then 1 token (checked against the Stage-0 sched rows).
  - The localisation range stays tokens 0-13,055 (blocks 0-2, the cached slice), so the host-memory hold is
    unchanged (3.5 GB per rank).
- The cache rule is in `cap0_hooks_layout.cache_hit()` and is tested against Stage 0 and attempt 1.
- **Early abort:** if the first replicate's warm is not uncached, or its capture request does not get exactly 13,056
  cached tokens, the client ends the session at once and records why.

**A2. Step and scheduler logs are flushed at every keyed removal on rank 0.**
- In the Stage-0 module, `flush_steps()` (step log, sched log, step ring) runs only after a FORCED request's store
  flush. capture-p0's requests are keyed with an empty forced list, so in attempt 1 nothing ever flushed them, and
  G-exec had no chunk rows to check.
- Now: two more marked lines in `p4_hooks.on_remove` (21 in all) call `flush_steps()` after the cap0 removal hook,
  on rank 0 only, when a flush is allowed.
  - A second call in the same removal (a forced request's own flush) finds nothing new: no row is logged twice.
  - Without the marked lines the module is still byte-identical to the Stage-0 module.
- New image `local/glm53-v091-p4-cap0-a2:local` (`sha256:8cb49fcabcfe…`). The attempt-1 image keeps its tag; its build
  record is in `build/attempt1/`.

**A3. Top-k set or order (an added sub-reading; written before any attempt-2 data).**
- Attempt 1's capture failed, but its four localisation warms were valid. Their localisation files pass G-loc on the
  served model: every boundary row present, identities holding, input ids equal across ranks and replicates.
- Read descriptively, outside the gated table: every pair first parts at `L3.topk` from token 2,051 in warm chunk 1.
  The saved first differing top-k rows hold the same token SET in a different ORDER, and `L3.attn_out` differs on
  exactly those rows. Chunk 0 is equal (`tests/out/attempt1-loc-diag.json`).
- When the located boundary is `L3.topk`, the analysis now also reports, for the saved first differing rows, "same set,
  other order" against "different selection". This decides what the replay must test: the top-k kernel's output order
  and the attention's dependence on it, or the selection itself (the never-written pools, allocation).
- The decision table and its classes are unchanged.

**A4. Re-running after an INCOMPLETE attempt; the pre-check per image.**
- The budget gate counts a pre-check only for the image it ran, so the new image needs its own pre-check
  (cap 180 s = 0.050 GPU-h; attempt 1's took 18 s).
- After a completed attempt, a new session needs:
  - that attempt's analysis to have read INCOMPLETE;
  - `CAP0_PRIOR_ACK` = the first 12 hex of the sha256 of its `cap0-result.json`.
- A valid reading is never re-run.
- The ledger and the approval are unchanged: used 0.470 GPU-h of the operator's 2.0.

## 1. Question and scope

Stage 0 found that two independently warmed replicates of identical prompt ids leave different valid cached state.
- By two-sum checksum, layers 0-3 compared equal and every layer from 4 on differed.
- Layer 3 is the first MoE and first sparse-MLA/indexer layer; layer 4 is the first KDA layer after it.
- Checksums cannot say which bytes differ, and they are not proof of equality.

**Goal (assessor):** identify one valid differing state slice and the operation that produces or consumes it.

- **Core capture (always on):** the exact bytes of the decisive slices at point 0 of the R01-early capture
  request, compared exactly between replicates.
- **Localisation option (on by default, switchable):** per-token sha256 of every row at the layer-3 and layer-4
  boundaries during each warm prefill. It shows the first boundary at which the replicates part.

**Not in scope:**
- any forced decode, propagation or behavioural test;
- the tile switch (arm B);
- any engine other than stock 0.9.1.

## 2. Exact slices (core capture)

**When the slices are read:** in `prepare_attn` of the capture request's first step, through the forcewin
`on_batch` hook and `p4_hooks._s0_batch`.
- That is before `model_state.preprocess_state` (the align-mode pre-copy) and before the forward.
- In the runner source, `on_batch` is at `model_runner.py:1464-1465` and the pre-copy at `:1615-1625`.

**How they are addressed:**
- `b` is the replicate's own physical block id, read from the request's block-table row.
- P = 2,297,856 B is the page of one MLA KVCacheTensor block (68 kernel pages × 64 tokens × 528 B).
- The indexer page is 143,616 B (17 pool pages × 8,448 B).
- Formulas: `build/files/vllm/cap0_hooks_layout.py`.

| Slice | Rank | Bytes | Kind |
|---|---|---|---|
| MLA KV, layer 7, logical blocks 0, 1, 2 | 0 (the MLA owner of layers < 25) | `[b·P, (b+1)·P)`: 4,352 records of 528 B (`[0,512)` E4M3 latent in 4 groups of 128; `[512,528)` 4 fp32 scales) | valid state |
| MLA KV, layer 3, blocks 0-2 | 0 | same | control (equal by checksum in Stage 0) |
| KDA layer 4 recurrent, cached-state column `n_cached/4352 − 1` = 2 | 0 and 1 | `[b·P + 147,456, +2,097,152)` | valid state |
| KDA layer 2 recurrent, same column | 0 and 1 | same | control |
| KDA layer 0 conv, same column | 0 and 1 | rows 0-2 `[b·P, +73,728)`; rows 3-5 `[+73,728, +147,456)`, stored separately | rows 0-2 valid; rows 3-5 inactive |
| KDA layer 4 conv, same column | 0 and 1 | as layer 0 | rows 0-2 valid |
| Indexer K (pool cache), layers 7 and 3, blocks 0-2 | 0 | `[b·143,616, +143,616)`; pool records 511 and 1,022 (264 B) also stored apart | consumed; records 511 / 1,022 never written by the warm |
| KDA layer 4 recurrent, pre-copy destination column 3 | 0 and 1 | sha256 only | timing check |

**Each slice is copied twice:**
- (a) as a raw byte range of the tensor's untyped storage at the formula offset;
- (b) through the engine's own view of that block.

**G-addr requires** (a) == (b), the stage-0 two-sum digest computed on the device equal to the host two-sum of the
copied bytes, and kernel pages `b·68 + k` in the block-table row.

**Size:** about 19 MB per replicate on rank 0 and 4.4 MB on rank 1. This is more than the audit's 9 MB: blocks 1 and
2, and the controls, are included.

## 3. Construction (`runner/client_cap0.py`)

**Session:**
- One serving session on 2 GPUs (TP 2), image `local/glm53-v091-p4-cap0:local` **[A2: now `local/glm53-v091-p4-cap0-a2:local`]**.
- Debug environment: `GLM53_P4=on GLM53_P4_S0=on GLM53_CAP0=on GLM53_CAP0_LOC=on`, plus the Stage-0 forcewin
  settings.
- The recipe environment is Stage 0's: KDA, DFlash2 with 3 draft tokens, chunked prefill, prefix caching in align
  mode, FULL_AND_PIECEWISE graphs, MLA owner map `split:25`.
- With the switch off, the image is the Stage-0 image (section 6).

**Replicates:** the default plan is six, in this order: r1 L, r2 L, r3 N, r4 N, r5 L, r6 L.
- L means localisation on for that replicate's warm; N means off.
- Four localisation replicates give six localisation pairs.
- The two N replicates are an instrumentation contrast: same hooks installed but inert, as close to Stage-0
  conditions as this image allows.
- At least 2 counted replicates are required. Each replicate costs seconds; start-up dominates.

**Per replicate, each request awaited before the next:**

| Request | Prompt | Settings | Expected |
|---|---|---|---|
| warm `CW-rk-L/N` | ~~R01 ids [0, 13,057)~~ **[A1] R01 ids [0, 17,409)**, the capture prompt | max_tokens 1, T 1.0, top_p 0.95, seed 72,999 | cached_tokens 0 |
| capture `CAP-rk` | R01 ids [0, 17,409) (the R01-early window prompt) | max_tokens 1, seed 73,000 | cached_tokens 13,056; first step = point 0 |
| sentinel | 4 ids, its own salt | max_tokens 1 | makes the engine remove the capture request |

- Cache salt `cap0-rk` per replicate: no prefix reuse across replicates.
- Same seeds and same ids in every replicate.
- Requests are keyed with an empty forced list (nothing is forced). The keys exist only so the recorders see them.

**Warm chunking:** the ~~13,057-token~~ **[A1] 17,409-token** warm prefills in align-mode chunks of 2,045 / 2,045 / 262 per
4,352-token block (**[A1]** four blocks), then 1 token.
- This is the Stage-0 recorded geometry for the first 13,056 tokens (checked against the Stage-0 sched rows in
  `tests/test_layout.py`).
- A shorter warm does not change how blocks 0-2 are computed. **[A1] But it changes what the capture request can
  hit: a 13,057-token warm leaves only 2 hittable blocks.**

**Localisation option (`build/files/vllm/cap0_hooks.py`, section 2; both ranks):**

Recorded only in solitary eager warm steps: more than 64 tokens, graph mode "none" (`cudagraph_utils.py:483-499`
dispatches > 64 tokens to eager). Tokens 0-13,055 are covered.

For every token row, the sha256 of the raw bytes at these boundaries, in computation order:

| Boundary | What it is |
|---|---|
| `L2.state_end` | Layer-2 running KDA column after its state write (control) |
| `L3.in` | x, residual (4 mHC streams), post, comb entering layer 3 |
| `L3.pre_attn` | Fused mHC post+pre outputs |
| `L3.attn_in` | Attention input |
| `L3.topk` | Indexer top-k token ids (rank 0) |
| `L3.attn_out` | Attention output after the TP all-reduce |
| `L3.pre_ffn` | Fused mHC post+pre (ffn side; y = MoE input) |
| `L3.moe_out` | MoE output after the all-reduce |
| `L4.in` | State entering layer 4 |
| `L4.pre_attn` | The materialised residual into layer 4 |
| `L4.conv_out` | Short-conv output |
| `L4.rec_out` | Chunked-recurrence output |
| `L4.attn_out` | Layer-4 KDA output |
| `L4.state_end` | Running column after the state write: recurrent and conv rows 0-2, sha256; raw at block ends |

- **The mHC residual is recorded as the model carries it:** 4 residual streams (`[m, 4, 4096]` bf16) with post
  (`[m, 4, 1]` fp32) and comb (`[m, 4, 4]` fp32). Each layer defers its hc_post to the next layer's fused op.
  Nothing is contracted or materialised by the instrument; `L4.pre_attn` is the engine's own materialisation.
- **Identities by source** (L3.attn_in = L3.pre_attn.y; L4.in = layer-3 outputs) are recorded twice and must agree.
  This is the instrument's self-test.
- **Raw rows are kept** for positions 0-63 and the chunk boundaries.
- The first warm of the session is held in host memory as the reference (3.5 GB per rank). For later warms, the raw
  rows of the first 8 positions per tensor whose sha256 differs from the reference are kept for both.
- **Read-only:**
  - module forward hooks and one wrapper of `kda.causal_conv1d_fn`;
  - rows are copied in blocks of 8 MiB or less, so device temporaries appear only for non-contiguous inputs;
  - `.cpu()` synchronises the stream inside the eager step;
  - no hook does anything while a stream is capturing, or when the step is not armed.
- **What this changes:** timing and allocator state in the localisation warms. It does not change the computation.
  The N replicates are the contrast.

**Recorded besides:**
- the scheduler rows (Stage-0 sched log);
- block ids of every keyed request at removal;
- engine-side sha256 of the cached prompt ids at point 0 (both ranks) and of each localisation step's input ids and
  positions.
- No token ids or text leave PRIVATE storage. `client.jsonl` holds only sha256 of the prompts.

## 4. Gates (`analysis/analyze_cap0.py`; written and tested before data)

**At start-up, before traffic (runner):**
- **G-instr:** local-p4 schema 2 with stage-0 recording on both ranks.
- cap0 targets resolved with the served layout:
  - rank 0: MLA and indexer 3 and 7, KDA 0/2/4;
  - rank 1: KDA 0/2/4.
- Layout checks pass: storage size `num_blocks·P`, storage offset 0 / 147,456, page stride P.
- MLA owner map `split:25`.
- Hooks: 12 on rank 0, 11 on rank 1, plus the conv wrapper.
- forcewin registered on both ranks.
- **G3:** stock launch geometry against captured-p3b.

**A replicate counts only if every gate passes.** Missing evidence fails the gate; it is never read as a negative.

| Gate | Requires |
|---|---|
| G-client | warm ok, cached_tokens 0; capture ok, cached_tokens 13,056; prompt_tokens as sent |
| G-exec | The warm's steps over tokens 0-13,055 are exactly 2,045 / 2,045 / 262 × 3, solitary, eager, prefilling. The capture request's first step is solitary, from n_computed 13,056, n_sched 2,045 |
| G-capture | One capture record per rank, ok, at that first step; every slice present; npz sha256 = its log record |
| G-addr | layout_ok, view == raw, device two-sum == host two-sum, kernel pages contiguous for every slice |
| G-timing | The pre-copy destination column (block and bytes) differs from the cached column; otherwise "read after the pre-copy" cannot be excluded |
| G-prov | Physical ids at the captured logical columns = the ids this replicate's warm recorded at its removal (each block was built by its own warm) |
| G-input (cross-replicate) | Identical prompt sha256 and identical engine-side cached-ids sha256 in every counted replicate and on both ranks; distinct salts |
| G-loc (localisation replicates; affects only the localisation reading) | Both ranks' files present with matching sha256; steps cover exactly the warm chunks, eager; every boundary row 0-13,055 present; identities hold; a state record at every step end; input ids hashed and equal across ranks; no recording error |

**Fewer than 2 counted replicates gives INCOMPLETE.**

## 5. Pre-registered readings and the assessor's 22ax table

**Valid slices:** MLA-7 blocks 0-2; KDA-4 recurrent and conv rows 0-2; KDA-0 conv rows 0-2.
- Controls: MLA-3, KDA-2.
- Reported, not valid state: conv rows 3-5; indexer pool records 511 and 1,022; the rest of the indexer blocks.

| Outcome | When | Assessor row → next action |
|---|---|---|
| **INCOMPLETE** (instrumentation/addressing invalid) | Fewer than 2 counted replicates; a cross-replicate input gate fails; or the first differing boundary is an identity of an earlier one | "Instrumentation or addressing remains invalid" → repair that measurement before interpreting anything |
| **valid state equal this time (non-reproduced)** | No valid slice differs in any counted pair | No row (the Stage-0 difference did not reproduce here) → report it. One equal pair does not establish repeatability. Do not expand |
| **slice located, operation not located** | A valid slice differs, but: no localisation pair passed its gates; or the pairs disagree on the first boundary; or the first boundary is `L3.in` / `L2.state_end` (it enters before layer 3); or no boundary differs (outside layers 3-4, or after the warm's last write: lifecycle); or the causality check fails (an MLA-7 record differs before the located boundary's first differing token) | "Valid state differs, bounded replay does not locate it" → preserve the slice; stop expanding; plan the behavioural A/B |
| **operation located** | Every localisation pair with a valid difference has the same first differing boundary B (chunk-major, then computation order, either rank), all earlier boundaries equal | "Valid state differs, reproducible local operation defect found" → the operation is located in serving, not yet shown repeatable: replay it with the captured inputs (assessor section 5) before attributing propagation |

**Operation between boundaries** (`LOC_OPERATION` in the layout module):

| First differing boundary | Operation |
|---|---|
| `L3.pre_attn` | mHC fused post+pre |
| `L3.topk` | kpool indexer |
| `L3.attn_out` | sparse MLA attention, including its KV write and the all-reduce with the non-owner rank's zeros |
| `L3.pre_ffn` | mHC ffn side |
| `L3.moe_out` | MoE |
| `L4.pre_attn` | Layer-4 mHC |
| `L4.conv_out` | Short conv |
| `L4.rec_out` | Gather of the initial state + chunked recurrence |
| `L4.attn_out` | o_norm / o_proj |
| `L4.state_end` | State write or pre-copy |

**Also reported:**
- **MLA-7 per-token diff:** which records differ, the first position, and the count per warm chunk.
  - Does chunk 1 (tokens 0-2,044) already differ? Positions below 2,048 use the dense short-prefill path, so the
    indexer selection cannot be the cause there.
  - Latent-only vs scale-only lanes, and the decoded magnitude.
- **KDA-4:** which rank or ranks differ.
- **Conv rows 0-2 vs 3-5:** tests the audit's stale-row inference.
- **Pool records 511 / 1,022:** zero or not, equal or not.
- **Allocation (assessor section 5):** the reading depends on which pairs differ.
  - Same allocation and different bytes: execution nondeterminism or incomplete initialisation.
  - Same-allocation pairs equal and relocated pairs differ: allocation or layout dependence.
  - Only relocated pairs (the likely case): the two are not separated.
- **Instrumentation:** do the localisation pairs and the no-localisation pairs disagree on whether valid state
  differs?
- **Lifecycle:** the warm's last write of the layer-4 cached state compared with the point-0 bytes, in the same
  replicate.

## 6. Build, tests and pre-check

**Image:** `local/glm53-v091-p4-cap0:local` (**[A2]** `-a2`), built FROM the Stage-0 image (`build/make_build_cap0.sh`, CPU,
`--network none`). Only `p4_hooks.py` changes; `cap0_hooks.py` and `cap0_hooks_layout.py` are added.
- `p4_hooks.py` minus its 19 (**[A2]** 21) lines marked `# local-cap0` is byte-identical to the Stage-0 module.
- With `GLM53_CAP0` unset, `cap0_hooks` is never imported.
- Post-build gates and the Stage-0 CPU suite, run against the new image with the switch off, check this
  (`tests/run_cpu_tests.sh s0regress`).

**CPU tests (`tests/run_cpu_tests.sh`):**
- addressing, including the audit's task-4 byte ranges for ids 436 / 762 and the Stage-0 warm geometry;
- hook capture and localisation on served-layout synthetic tensors;
- negative controls;
- the analysis on synthetic sessions: every outcome, every planted defect → INCOMPLETE;
- the runner end to end against a mock server;
- the budget gate.

**Single-GPU pre-check (`precheck/precheck_cap0.py`; cap 180 s; must PASS on the GPU before the session):**
- capture exactness on a device KV cache in the served layout;
- CUDA-graph capture with every hook installed (unarmed, and with a stale arm set), recording nothing, replay
  equal to eager;
- an armed 2,045-token step at the served row widths: hashes equal to host, time measured;
- the negative control.

## 7. Cost and budget (`runner/cost_cap0.py`, `cost_cap0.json`)

| Case | Start-up | Per replicate (core) | + localisation per replicate | Core only, 4 replicates | Default: 6 replicates, 4 with localisation |
|---|---|---|---|---|---|
| Planning (2,400 tok/s; host hash 0.8 GB/s) | 874 s | ~~12.8 s~~ **[A1] 14.6 s** | 6.5 s | ~~0.556~~ **0.560 GPU-h** | ~~0.584~~ **0.590 GPU-h** |
| Typical (3,500 tok/s; 1.28 GB/s measured) | 740 s | ~~6.8 s~~ **[A1] 8.0 s** | 3.7 s | ~~0.443~~ **0.446 GPU-h** | ~~0.459~~ **0.463 GPU-h** |

**[A1]** The warm is now 17,409 tokens per replicate. Attempt 1 measured: ready after 769 s; 837 s for the whole
session (0.465 GPU-h); about 0.5 s of hashing per 2,045-token step per rank.

- **Pre-check:** typical about 30 s, i.e. 0.008 GPU-h.
- **Localisation option:** adds 0.008-0.014 GPU-h. It hashes 3.64 GB per rank per warm (278,928 B per token) on the
  host, in parallel on both ranks.
- **Wall caps:**
  - session 1,100 s × 2 GPUs = 0.611 GPU-h;
  - pre-check 180 s × 1 GPU = 0.050 GPU-h.
- **Approval needed for one pre-check attempt and one session attempt: `APPROVED_GPU_H ≥ 0.661`.**
  - The gate refuses an attempt whose full cap does not fit.
  - The ledger is `capture-p0/costs-cap0.jsonl` only.
- **Why the cap is above the audit's "about 0.6":** start-up alone is 740-874 s.
  - A 0.60 total (session cap 990 s) leaves about 100 s after a slow start: replicates 5-6 would likely be dropped.
  - A start-up beyond about 900 s would spend the attempt with nothing captured.
  - The estimate itself (0.46-0.58 with the pre-check) is in the audit's range.

## 8. What this cannot show

1. **Equal bytes in one pair do not establish repeatability.** Six replicates are more pairs, but only of this
   construction (one prompt, one block-size geometry, a fresh server).
2. **A located boundary is where two serving runs first part.**
   - It is not a replay of that operation and not a cause.
   - It does not clear the rest of the serving pipeline.
   - A replay of the operation with the captured inputs (assessor section 5) is the next step.
   - The raw rows kept (the first 64 positions, the chunk boundaries and the first 8 differing positions) are not
     a full input set for a whole-chunk replay.
3. **Localisation covers layers 3 and 4 and tokens 0-13,055 of the warm.**
   - A difference that arises in layers 5-6 (before MLA-7) is reported as "outside the measured layers".
   - So is one in another part of the step.
4. **The localisation hooks change timing and allocator state in their warms.** The N replicates are the contrast.
   The installed-but-inert hooks, the Stage-0 recording and the point-0 copy itself remain in every replicate.
   Instrumentation is not excluded as a contributor (assessor priority 4).
5. **Physical allocation is not controlled.** Pairs are classed same-allocation or relocated after the fact.
   - If every pair is relocated, allocation dependence and execution nondeterminism stay unseparated.
   - Prior holders of a block are known only from keyed requests' removal records. Start-up warm-up and sentinels are
     not recorded.
6. **The capture is of rank 0 for MLA and indexer state** (the owner of layers < 25), **and of both ranks for KDA
   state.** MLA layers owned by rank 1 are not captured.
7. **No statement** about the tile's effect, the version gap, or behaviour.

## 9. Commands (operator; in order; no GPU step without an approval)

**[A4] For attempt 2** (approval 2.0 GPU-h, of which the ledger shows 0.470 used):

```bash
cd <investigation>/tile-p4/capture-p0
# 0. CPU only (already run for this package; re-run any time):
tests/run_cpu_tests.sh layout image enginepath precheck analysis runner gates attempt1 s0regress
# 1. single-GPU pre-check of the new image (cap 180 s = 0.050 GPU-h)
APPROVED_GPU_H=2.0 ATTEMPT_REASON="capture-p0 pre-check, image a2" runner/run_precheck_cap0.sh
# 2. the session (2 GPUs, cap 1,100 s = 0.611 GPU-h); the analysis runs in a CPU container after the GPUs are released
ACK=$(sha256sum PROTOCOL.md | cut -c1-12)
PRIOR=$(sha256sum runs/session-attempt-1/cap0-result.json | cut -c1-12)
APPROVED_GPU_H=2.0 ATTEMPT_REASON="capture-p0 session attempt 2 (A1 warm, A2 flush)" CAP0_PROTOCOL_ACK=$ACK \
  CAP0_PRIOR_ACK=$PRIOR runner/run_cap0.sh
#    -> runs/session-attempt-2/{outcome.json, cap0-result.md}; ../PRIVATE-capture-p0/session-attempt-2/ (0700)
```

- Every GPU command accepts `DRY_RUN=1`: the gates run and the plan is printed; nothing starts.
- `CAP0_LOC=off` runs the core capture alone.
