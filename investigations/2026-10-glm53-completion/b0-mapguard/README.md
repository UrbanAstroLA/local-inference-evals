# Live slot mappings on checkpoint B0 (amendments 17-17e)

Question: under real scheduling, do the MLA and index-K kernels of checkpoint B0 consume the cache slots they should?
B0 is the D1-D4 correctness bundle on tpurtell 0.9.1 (image `sha256:08d1cb90…`), unchanged. One serving session, both
GPUs, shipped CUDA graphs, DFlash2 ×3, existing fixtures only; no GPQA scoring.

**Result: VIOLATION, all of it one defect, D7.** Every attention (MLA) write and read, index-K decode write and read,
index-K prefill read, tail-ring write and position check was clean over the frozen coverage. At the index-K prefill write,
every flagged access is the first pool after a prefill chunk start that is not a multiple of 4. Whether this changes
selection or completion was not measured.

## Method

- **Guard** ([`guard/mapguard_hooks.py`](guard/mapguard_hooks.py)): an observe-only module added to B0 by hook lines in
  three engine files ([`build/mapguard.diff`](build/mapguard.diff)), inert unless `GLM53_MAPGUARD` is set. It extends the
  earlier tail-ring guard.
- **Independent expectation:** a worker-side ledger rebuilt from the scheduler's output stream alone. It is cross-checked
  offline against digests read from the KV cache manager in the engine-core process, never copied from the mapping
  builder the kernels use.
- **Check sites:** eight graph-safe device check sites at the point of consumption, so mappings are checked as the replayed
  graphs consume them, not only as prepared on the host. They cover MLA write and read, index-K prefill and decode writes,
  index-K prefill and decode reads, tail-ring seed and stash writes, and positions.
- **Invariants per access:** the physical block belongs to the intended request and KV group; the logical position maps
  to the intended record; inactive rows and masked accesses stay inactive; speculative slots stay separate from committed
  state.
- **CPU tests before any GPU use**
  ([`tests/`](tests), outputs in [`tests/out-postA2/`](tests/out-postA2)), run under the Triton interpreter with B0's real
  builders and allocator:
  - 28 of 28 pass;
  - injected wrong-request and stale block ids, wrong positions, row-reuse errors and wrong rank or group ownership are
    flagged;
  - valid padding, masked rows and rejected speculative slots are not flagged.
- **Frozen before running:** 28 coverage minimums, in [`PROPOSAL.md`](PROPOSAL.md) and
  [`analysis/analyze_mapguard.py`](analysis/analyze_mapguard.py). A condition below its minimum is INCOMPLETE, not clean.
- **Workload** ([`run/client_mapguard.py`](run/client_mapguard.py)): eight active-reasoning prefixes of 4,000-120,000
  tokens with bounded continuations at 1 to 8 in flight, a staggered admission, 16 GPQA prompts capped at 1,024 tokens,
  and four short greedy probes before and after. Every sampled request has its own seed.

## Attempts

| Attempt | What happened | Occupied GPU-hours |
|---|---|---:|
| 1 | Refused by the runner's pre-flight (no container): it counted the desktop's graphics clients on GPU 1. The filter was widened to the clients the other runners ignore (amendment 17c). | 0 |
| 2 | **Invalid as a test of the workload.** The client never ran: the runner's start-up abort searched for `guard_error`, which also matches the `"guard_errors": 0` field of every healthy readout, and stopped the session. The guard saw only the stack's own warm-up traffic. D7 appeared in that traffic. | 0.4211 |
| 3 | Full workload; every coverage minimum met; nothing invalid. | 0.5722 |

Total **0.9933 occupied GPU-hours** of a ceiling of 1.0 (amendment 17a), raised to 1.45 before attempt 3 (amendment 17e).
The cost lines are in [`costs.jsonl`](costs.jsonl).

### What went wrong in attempt 2, and what was changed

These faults were in the guard, the analysis and the runner, not in the engine. They were found by inspecting attempt 2's
data, fixed before attempt 3, and are labelled P1-P6 in [`DIAGNOSIS-a2.md`](DIAGNOSIS-a2.md). No coverage minimum,
invariant or verdict rule was loosened.

- **Runner (start-up abort pattern):** it now matches only guard-error records (`"guard_error"` in quotes).
- **Guard (index-K read expectation):** the guard expected block ids, but the reader consumes index-page ids 17·block + j.
  This caused the 86,736 decode and 84 prefill read flags. Those sites count as not covered for attempt 2. Records
  re-checked against the corrected expectation are consistent, and a regression test reproduces the a2 signature with the
  old guard and none with the new one.
- **Analysis:** an off-by-one in-graph count (a host-versus-device timing artefact) is fixed. The analysis window now
  starts at the first step that is not runner warm-up, with an exact baseline readout.
- **Attempt 2 re-analysed with the fixed tools:** VIOLATION (D7 only), nothing invalid, six conditions incomplete
  ([`run-a2/verdict-postA2.txt`](run-a2/verdict-postA2.txt)). The original verdict is kept beside it.

Before attempt 3, the following were checked with zero GPU use (amendment 17e):

- file hashes against [`SHA256SUMS-postA2`](SHA256SUMS-postA2);
- the image id;
- the CPU tests, and the regression test rerun on the rebuilt image
  ([`tests/out-prelaunch/`](tests/out-prelaunch)), because a logged rc=1 from that test did not reproduce;
- the client end to end against a local mock server: every phase, no errors, distinct seeds for all 64 sampled requests.

## Attempt 3

Rebuilt guard image `sha256:e48738c6…` (B0 plus the guard; B0's own files are unchanged). Readiness took 713 s and the
client 302 s. The verdict is in [`run-a3/verdict.txt`](run-a3/verdict.txt) and
[`run-a3/verdict.json`](run-a3/verdict.json), and the guard's records in [`run-a3/mapguard/`](run-a3/mapguard) (gzip).

**Coverage, all at or above the frozen minimums:**

| Condition | Observed | Minimum |
|---|---:|---:|
| graph-replay steps | 2,076 | 300 |
| mixed prefill/decode steps | 132 | 20 |
| row reuse | 140 | 20 |
| prefix-cache hits | 32 | 10 |
| decode steps at ≤ 4 in flight | 851 | 60 |
| decode steps at 5-8 in flight | 889 | 60 |
| fused/tiled route transitions | 18 | 6 |
| speculative writes | 213,465 | 20,000 |
| accepted drafts | 26,016 | 2,000 |
| rejected drafts | 16,677 | 2,000 |
| completions | 140 | 40 |
| short contexts | 90 | 8 |
| MLA writes / reads at blocks ≥ 935 (offsets above 2³¹ bytes) | 68,640 / 95.2 M | 5,000 / 5,000 |
| index-K writes / reads at blocks ≥ 935 | 17,140 / 113,180 | 500 / 500 |
| tail-ring writes at blocks ≥ 935 | 14,480 | 50 |
| index-K prefill reads | 1,910 | 50 |
| index-K decode reads | 284,640 | 20,000 |
| ledger join with the engine-core digests | 1.0 | 0.95 |

**Instrumentation effect:**

- The four short greedy probes gave output identical to the earlier B0 record, before and after the workload.
- Full CUDA-graph capture was kept, with no eager fallback.
- The prefix phases ran about 30-35% slower than the same client functions on B0 without the guard (T phases 26.2 s vs
  19.4-19.6 s; F phases 12.1 s vs 9.2-9.6 s).

**Violations, at the index-K prefill write only:**

| Rank | Unwritten pool (`missing_pred`) | Pool from the previous request's tokens (`pool_cross`) |
|---|---:|---:|
| 0 (layers 3-23) | 1,116 | 186 |
| 1 (layers 27-43) | 930 | 155 |

The counts are per layer: rank 0 owns six MLA layers and rank 1 five, so both ranks show the same **217 events** (step ×
request) in 50 requests: 186 unwritten pools and 31 cross-request pools.

A post-hoc attribution, written after the data and not part of the frozen analysis
([`analysis/attr_d7_posthoc.py`](analysis/attr_d7_posthoc.py), output
[`run-a3/d7-attribution.txt`](run-a3/d7-attribution.txt)), joins every record to its request's chunk start. All 2,387
records sit in the pool that straddles a chunk start that is not a multiple of 4, at 0-2 tokens after the start; none is
at an aligned start, and none is a later pool.

- Most such starts are at 2,045 and 4,090 tokens into a 4,352-token block. The scheduler's prefill budget is 2,048 tokens
  less 3 draft slots.
- Others (2,003, 2,010, 2,017, 2,031, 2,082, 4,078, …) occur in steps shared with other requests. What sets those
  offsets was not traced.
- The allocator simulation run before GPU use predicted 73 unaligned starts for this workload; the difference from 217
  was not analysed.

Before the analysis window, the runner's own synthetic warm-up batches produced 4 position-bounds flags per rank; they are
outside the window by the rule above and are listed in the verdict's notes.

## D7

The index-K prefill writer (`_kpool_compress_insert`, `sparse_attn_indexer_kpool.py:102-147` in B0's engine) pools
tokens t−3..t of the batch and writes a pool only when the token completes it. It assumes every prefill chunk starts on
a pool boundary, but the scheduler does not align prefill chunks to 4 tokens. For a chunk starting at an unaligned
position:

- **(a)** in the first request of a step, the pool that completes within the chunk's first three tokens is never written;
- **(b)** in a later request of the same step, that pool is written from tokens of the request before it in the batch.

The CPU test [`tests/test_kpool_unaligned.py`](tests/test_kpool_unaligned.py) reproduces both cases with B0's real
kernel. The behaviour sits in the engine's index-K writer, not in a kernel specific to the GPU architecture; whether 0.7.0
shares it was not checked here.

**Not measured:** how often index-K selection reads the affected pools, and any effect on outputs or completion. GPQA
prompts are mostly shorter than one 2,045-token chunk and start at position 0 (an aligned start), so most GPQA prefills
likely never reach an unaligned start; that is an inference.

## Not covered

- copy-on-write prefix paths, which are unreachable here because all prefix hits are block-aligned;
- contention-dependent races that one clean run cannot exclude;
- other model paths, such as the mixed-quant MoE.

## Files

| Path | Contents |
|---|---|
| [`PROPOSAL.md`](PROPOSAL.md) | design, invariants, coverage minimums, costing (before any GPU use) |
| [`DIAGNOSIS-a2.md`](DIAGNOSIS-a2.md) | attempt-2 diagnosis and fixes P1-P6 |
| `guard/`, `build/` | guard module (current and `.a2`), patch builder, hook diff, Dockerfile, build log, image id |
| `analysis/` | frozen analysis (current and `.a2`), post-hoc D7 attribution |
| `run/` | runner and client (current and `.a2`), pinned image ids |
| `tests/` | CPU tests and their outputs (`out/` before attempt 2, `out-postA2/` after the fixes, `out-prelaunch/` before attempt 3) |
| `reach/` | allocator simulation of the workload's reachable block ids |
| `run-a1/`, `run-a2/`, `run-a3/` | per-attempt budget, image id, script hashes, gate output, client log, guard records, verdicts |
| `SHA256SUMS`, `SHA256SUMS-postA2` | local file hashes before attempt 2 and after the fixes |
| `RAW-SHA256SUMS-mapguard.txt` | hashes of local files not published (fixture token ids, probe outputs, container logs, pre-flight listings, full patched engine files) |
