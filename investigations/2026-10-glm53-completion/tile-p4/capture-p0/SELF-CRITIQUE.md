# capture-p0 self-critique (written before any GPU step)

P4 run 1 failed for three reasons:
- its gates verified launches, not what the question needed;
- its CPU mocks idealised the scheduler;
- its analysis turned missing evidence into a negative.

This page takes each pre-registered reading of `PROTOCOL.md` section 5 and asks what could produce it spuriously, and
which gate or test catches it. "Fixed" means changed in this package before freezing. "Open" means it remains.

## Attempt 1 (2026-10-24): why the harness missed both defects

Attempt 1 read INCOMPLETE (0 of 6 replicates counted). The gates did their job: nothing was misread. But the CPU
harness should have caught both defects before any GPU time was spent. It did not, for the same root cause as P4
run 1: **the tests encoded my assumptions about the engine instead of exercising the engine path.**

| Defect | Why no CPU test caught it | Now |
|---|---|---|
| **Warm too short: the capture got 8,704 cached tokens, not 13,056** (amendment A1) | The mock answered `cached_tokens = 13,056` for every capture request, a constant I wrote from the protocol's expectation. The analysis fixtures did the same. Nothing modelled how the prefix cache answers. The audit's "13,057 is enough" was taken on trust; the EAGLE-type drop (`PROPOSAL-AMENDMENTS.md` A3) was already documented in this investigation | `cache_hit()` encodes the measured rule and is tested against Stage 0 (113,940 → 13,056) and attempt 1 (13,057 → 8,704, all six replicates). The mock applies the same rule per cache salt. Runner test S10 replays the attempt-1 construction against it: the mock drops a block, the client aborts after r1 with the reason, and the analysis names it |
| **No scheduler / step log** (amendment A2) | The mock never wrote one, and the runner test accepted the mock's INCOMPLETE as "expected, a mock has no bytes" **without checking why** it was INCOMPLETE. The engine-side tests (`test_hook_cpu`) called `p4_hooks._s0_batch` directly and never went through a removal. The Stage-0 suite's U6 saw a sched log only because a forced request flushed it afterwards | `tests/test_engine_path_cpu.py` drives the real forcewin port → `p4_hooks` → `cap0_hooks` with the hook-validation toy engine, using only keyed, non-forced requests. It **fails on the attempt-1 image** (negative control in the suite) and passes on the fixed module. It also checks the double flush (no row twice) and that rank 1 writes nothing. Runner test S7 now requires that the mock's INCOMPLETE is for the right reasons: G-client passes, and "no scheduler log at all" is named |

**Rule adopted:** a test that expects INCOMPLETE must assert the exact reasons, and a mock must not answer
with the value the protocol expects; it must compute it by the engine's rule.

**Other places checked where the engine path could differ from what the tests assume** (all keyed on the forced slot
in the Stage-0 module):
- `write_status()` (status_r0.json) runs only on the forced path. The runner no longer reads it: the flush wait
  counts cap0 records.
- Entry digests are allocated only for forced requests (W > 0). capture-p0 does not use them, by design: the
  capture replaces them.
- The forcewin port keys a request with `forcewin_hooks []`: it arms `on_batch`/`on_logits`, so `p4_hooks.on_logits`
  (which closes the localisation step) is reached. This is confirmed on the real records: every attempt-1
  localisation step recorded mode "none" and closed.
- The forcewin per-request metrics file is written for every keyed request (as in Stage 0). It is harmless.
- **The localisation instrument itself is validated on the real attempt-1 files:** G-loc passes for all four
  localisation warms (every boundary row, identities, step coverage, input ids equal across ranks and replicates).
  `tests/diag_attempt1_loc.py` is the read-only diagnostic.

**Remaining exposure:** the engine-path test uses the toy engine, not the server's scheduler. The real prefix-cache
answer is checked only by `cached_tokens` at run time. The client now aborts after the first replicate if it is wrong
(this costs one replicate's seconds instead of a session's worth of replicates).

## How the three run-1 failures are addressed

| Run-1 failure | Here |
|---|---|
| Gates verified launches, not the question | Each reading has its own evidence gate: G-addr, G-prov, G-timing, G-exec, G-input, G-loc and G-content. Start-up G-instr checks the resolved targets and their live layout (storage size, offset, page stride), not only that hooks registered |
| Mocks idealised the scheduler | Nothing in the client or analysis assumes a schedule. The analysis reads the engine's own sched rows and requires the recorded Stage-0 geometry (2,045 / 2,045 / 262 × 3), checked against the real Stage-0 rows in `tests/test_layout.py`. Anything else gives INCOMPLETE. The mock tests only the wrapper. Its byte-less output must read INCOMPLETE, and that is tested |
| Missing evidence turned into a negative | Every gate fails on absence. Fewer than 2 counted replicates gives INCOMPLETE. Missing or invalid localisation can never yield "equal" or "located": it yields "slice located, operation not located" with the reason. `tests/check_analysis_cap0.py` plants 16 evidence defects (a missing record, file or rank; wrong geometry; mismatched inputs; a failed address check; a broken identity). Each gives INCOMPLETE, a not-located reading, or excludes only the defective replicate. None gives "equal" |

## Reading by reading

### R1. "MLA-7 records differ (at these token positions)"

| Spurious cause | Caught by |
|---|---|
| Wrong physical id: the wrong block-table column, or kernel-page vs block id confusion | G-prov: the captured ids at logical 0-2 must equal the ids the replicate's own warm recorded at removal. G-addr: kernel pages are `b·68 + k`. Tests: `test_layout` A6 (the audit's ids 436 / 762), `test_hook_cpu` C1 / N2 |
| Formula offset wrong (page size, record layout) | G-addr: raw formula bytes == the engine's view of the block, and device two-sum == host two-sum. G-content: a written slice is never all zero. The layout constants are re-checked against the live views at registration (storage `num_blocks·P`, stride, offset) |
| A view that includes or excludes padding | The MLA page has no padding (68 × 64 × 528 = P). Raw == view is required. The KDA page padding `[2,244,608, P)` is outside every slice |
| Reading after the pre-copy or the forward | Order in source: `on_batch` (runner `:1464`) runs before `preprocess_state` (`:1620`) and the forward. `test_hook_cpu` P1 (AST) checks that the capture call is in `_s0_batch` only. G-timing: the pre-copy destination column must still differ from the cached column. MLA blocks 0-2 are not written by the capture request at all (it writes from block 3 on) |
| Capture inside a CUDA graph | The capture runs only in `_s0_batch`, which p4_hooks calls only when serving and not capturing. Point 0 is a 2,045-token eager step |
| The hook perturbs what it reads | The copy is taken after the warm finished. Blocks 0-2 are read-only afterwards |
| Rank mismatch | MLA 3 / 7 exist only on rank 0 (owner map `split:25`, checked at G-instr). Each file records its rank and tag and must match its name (G-capture; synthetic test Y2) |
| Replicates sharing a prefix-cache entry | Distinct salts (G-input). Warm `cached_tokens` must be 0 and capture `cached_tokens` 13,056 (G-client). G-prov |
| Stale bytes from an earlier holder | Every MLA record of blocks 0-2 is written by the warm (audit task 4: slot ≥ 0 for every token). Earlier holders are listed anyway (`prior_holders`) |

### R2. "Control (MLA-3, KDA-2) equal"

- **Spurious equality from reading the wrong, coincidentally equal bytes (for example zeros):** G-content rejects all-zero
  written slices. Raw == view.
- **Spurious inequality:** the same gates as R1. A differing control is reported as a note that refutes the
  layer-order premise. It is not hidden.

### R3. "KDA-4 recurrent differs on rank 0, 1 or both"

| Spurious cause | Caught by |
|---|---|
| Wrong state column (the align-mode semantics) | The column formula `n_cached/4352 − 1` comes from source (audit task 3). G-prov: the id equals the warm's column 2. **Lifecycle cross-check (localisation replicates):** the warm's last state write at token 13,056 (column id, recurrent and conv rows 0-2 sha256) must equal the point-0 bytes in the same replicate. If not, the reading becomes "changed after the write" (synthetic test V), never a silent pass |
| Reading after the pre-copy | G-timing (above) |
| Rank-local shards confused | Each rank reads its own views. The files are per rank. Rank is part of every comparison key |

### R4. "Layer-0 conv rows 0-2 vs rows 3-5" (the audit's stale-row inference)

- **Row layout misread** (SD `[6, 12288]` vs DS): G-instr checks the conv view's shape and dtype. Rows 0-2 are the
  first 73,728 B, and raw == view on that exact sub-slice.
- **The "live" label itself rests on source** (`causal_conv1d_fn` reads and writes rows 0-2; `state_len =
  KERNEL_WIDTH − 1`).
- Open: this is source reading (audit task 3), not measured. No test executes the served conv kernel.

### R5. "Indexer pool records 511 / 1,022 zero or not, equal or not"

- **Wrong pools:** derived from the measured chunking by `never_written_pools()`. G-exec requires that chunking.
  `test_layout` A2 / A4 reproduce the audit's offsets.
- Open: an earlier holder that wrote those exact pools is reported only for keyed requests. The start-up warm-up
  (`GLM53_STARTUP_WARMUP=1`) and the sentinels are invisible.

### R6. "First differing boundary is B, therefore operation O"

| Spurious cause | Caught by |
|---|---|
| A hook labelled or attached wrongly | Identities by source are recorded twice and must agree (G-loc: L3.attn_in = L3.pre_attn.y; L4.in = layer-3 outputs). Per-rank hook counts are checked at start-up (a shortfall degrades the option, see below). `test_hook_cpu` L1 checks every boundary's row sha256 against the toy's own trace |
| A boundary read before its producer finished (streams) | Each hook reads on the current stream at the point where the model's next op reads the same tensor. `.cpu()` blocks until that copy completes. Open: a producer on a side stream that the model itself does not join would race for the model too. That would be an engine defect, not an instrument artefact, but the reading could not tell them apart |
| The armed state leaking into another forward (drafter, dummy run, graph replay) | Armed only in `on_batch` for a solitary "-L" warm step of more than 64 tokens. Cleared after the forward and at every `on_batch`. Hooks also require graph mode "none" and no capture. `test_hook_cpu` L3; pre-check K2 (graph capture with a stale arm records nothing and replays equal) |
| Inputs not identical across replicates | Client prompt sha256 (G-input). Engine-side cached-ids sha256 on both ranks. Per-step input-ids and positions sha256 in every localisation step, equal across ranks and across localisation replicates (synthetic test Y3) |
| Cross-chunk dependence misread as a local operation | Chunk-major order: an earlier chunk's later boundary is found first. The state carried between chunks is itself a boundary (L2 / L4 state_end), and the layer-3 KV / pool cache is consumed inside the boundaries that read it (attention, indexer) |
| Disagreement between pairs, or a cause downstream of a different first token | All differing localisation pairs must agree. Causality check: no MLA-7 record may differ before the located boundary's first differing token (synthetic test U) |
| The localisation hooks cause the difference | The N replicates (hooks installed, inert) are the contrast. The analysis reports whether localisation and non-localisation pairs disagree |
| Missing or invalid localisation | G-loc fails → that replicate's localisation is excluded → "operation not located (no localisation pair passed its gates)" (synthetic tests O, P) |

### R7. "Valid state equal this time (non-reproduced)"

- **The construction differs from Stage 0 in a way that removes the cause.**
  - The 13,057-token warm computes blocks 0-2 identically (same tokens, same chunks).
  - But a fresh server with fewer blocks in use has a different allocation history from Stage 0. A cause that
    depends on stale memory could vanish.
  - Open: reported as non-reproduction, with the caution that one construction's equality is not repeatability.
- **The point-0 syncs or the localisation syncs mask a race:** the N replicates have no syncs inside their warm, so a
  masked race would show as a loc / no-loc disagreement.

### R8. "INCOMPLETE"

INCOMPLETE could be over-triggered by a too-strict gate. The gates were checked against the real Stage-0 records where
they exist (chunking, block ids, offsets). Hook counts per rank are derived from source: rank 0 has 12; rank 1 has 11,
having no indexer.

- **Fixed:** a hook-count mismatch no longer aborts the session. It degrades localisation only (runner test S9).
  Spending a start-up on the optional part was the wrong trade.

## Fixed during this critique

- The base-image module hash in the build script was copied from the run-1 Dockerfile; gate 0 caught it before any
  build.
- Gate (a) compared the file list under locale collation; it is now `LC_ALL=C`. The post-build gates were re-run
  (`GATES_ONLY=1`) on the unchanged image.
- Added:
  - G-content (all-zero valid slice);
  - the file self-identity check (rank / tag);
  - localisation input-id equality across replicates;
  - the conv rows in the lifecycle cross-check;
  - the prior-holder report.
- The start-up gate now separates core failures (stop) from localisation-hook shortfalls (degrade).
- "No localisation pair" and "the differing pairs lack localisation" are now distinct reasons.
- The precheck's verdict counts as PASS only when it ran on the GPU (a CPU fallback is not a pass).

## Open (not fixable on CPU, or out of scope)

1. **Real hook attachment** is checked only by counts at start-up and by the identities in the first localisation
   record. The CPU tests use a toy with the same call pattern (keyword calls of the fused mHC op, positional decoder
   calls, the indexer called through `__call__`, the module-level `causal_conv1d_fn` looked up at call time).
   - If the served model differs, G-loc fails and localisation reads INCOMPLETE. The core capture is unaffected.
2. **The localisation hooks add stream syncs, host time (about 0.5 s per 2,045-token step per rank) and device
   temporaries (8 MiB or less) to the localisation warms.** The computation is unchanged by construction (read-only
   copies), but allocator state and timing differ. Only the N contrast addresses this. Instrumentation is not excluded
   (assessor priority 4).
3. **Every replicate runs with the Stage-0 recording and the capture hooks installed.** Neither the Stage-0 run nor
   this one is a production server.
4. **Allocation is not controlled.** If every pair is relocated (likely), allocation dependence and execution
   nondeterminism are not separated. A same-allocation pair happens only by chance.
5. **Localisation covers layers 3-4 and the warm tokens only.** A difference produced in layers 5-6 is reported as
   "outside the measured layers". Raw rows suffice for magnitudes at the first differing tokens, not for a
   whole-chunk replay. A replay of the located operation needs its own capture of that operation's full inputs.
6. **Rank-1 MLA layers (≥ 25) are not captured.**
7. **The pre-check's graph check uses a toy model.** The real hooks meet the real graphs only at the session's own
   capture (start-up). A hook that broke capture would show as a start-up failure (the session is spent; the cost
   line records it).
8. **The 1,100 s session cap leaves the client about 120 s after a 874 s start-up** (client deadline at cap − 90 s; planning needs about 103 s for all six replicates). The client skips trailing replicates when
   short (2 replicates are the minimum). A start-up beyond the 960 s readiness gate spends the attempt.
