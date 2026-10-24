# Captured-input MoE-plan test and attention confirmation (captured-p3b)

**Status.** The protocol as built, CPU-tested and approved on 2026-10-23, then run (results: `RESULT.md`). It was not
published before the run. After capture attempt 1 failed at start-up, the operator raised the ceiling from 0.65 to 1.0
occupied GPU-hours (`AUTHORIZATION.txt`). The poller fix and the replay Caps rebuild are described in `RESULT.md`; the
text below is otherwise the pre-run design.

**Authorization.** `AUTHORIZATION.txt`: the operator approved "do all three":
- a small captured-input test;
- capture as a standalone run of about 0.6 GPU-h;
- replay of about 0.05 GPU-h.

The ceiling is **0.65 occupied GPU-hours in total, including retries**. Captured tensors derive from GPQA prompts and
stay local.

**Assessor requirements** (`HANDOFF-NEXT/ASSESSOR-DISPOSITION-22AP-2026-10-23.md` items 4-6):
- hold rows, weights, expert ids and routing weights fixed;
- under FULL graphs, capture actual serving inputs in a graph-compatible way;
- reproduce the captured baseline output with the recorded plan and shape before reading any cross-plan difference;
- on served attention inputs, compare inflated against exact counts with all else fixed, covering the dense, clamped
  and 64-boundary cases.

## 1. Questions and what is not claimed

**Questions:**
- **MoE.** On rows the 0.9.1 engine actually served, do the decode plan (K128/N128 at 10-32 rows) and the prefill
  plan (K64/N256) produce different routed-expert outputs, and by how much against P3's synthetic inputs?
- **Forced tile** (added by the coordinator after `moe-plan-p3/SOURCE-MECHANISM.md`). Does a decode plan forced to
  the prefill tile reproduce the prefill plan bit for bit?
- **Attention.** On served verify rows, does the consumed (inflated) count change the attention output compared with
  the exact count p + 1, all else fixed?

**Not claimed, whatever the result:**
- an explanation of the 0.7.0-0.9.1 completion difference;
- any effect of a per-layer difference on decode behaviour;
- anything about the EP2-vs-TP2 reductions (E-partition), which are not exercised;
- anything beyond the captured layers, rows, shapes and regimes.

Equality is reported as **"equal on captured rows"**, never as "ruled out".

## 2. Design in one paragraph

**Step A, capture (2 GPUs while serving).** The 22ao arm A stack and recipe are served on a new image:
- stock 0.9.1 plus observe-only capture hooks, tagged `local/glm53-v091-p3b-capture:local`, id `sha256:5be4b853…`;
- built `--network none` from the local stock image `a9e985c0`.

The hooks record, inside the served CUDA graphs:
- **MoE:** the inputs and outputs of `Exl3MoEMethod.apply` on layers 3, 23 and 44;
- **attention:** the full input set of the B12x sparse-MLA decode kernel on MLA layers 3 and 23, including the KV
  records it reads.

The workload is the GPQA panel at 8 outstanding. Capture stops as soon as fixed quotas are met.

**Step B, replay (GPU0 only, stock images).** The replay first reproduces every served output from the captured
inputs with the served plan. Only then does it run the comparisons.

## 3. Capture hooks (`build/files/vllm/p3b_capture_hooks.py`)

The hooks are inert unless `GLM53_P3B=capture`. They are injected as marked additive lines:
- 13 lines in `quantization/exl3.py`;
- 20 lines in `attention/backends/mla/b12x_mla_sparse.py`.

`build/make_hooked.py` generates both files from the stock copies by exact-anchor insertion. Its gate requires:
- insertions only;
- every inserted line marked `local-p3b`;
- exact line counts.

Of the image's 2,435 vLLM `.py` files, only these three differ from stock (`build/post-build-gates.txt`).

### 3.1 Sites (rank 0 only; rank 1 stays inert)

| Site | Where (stock line) | Recorded |
|---|---|---|
| MoE | `Exl3MoEMethod.apply`, after `_apply_rank_sliced` (exl3.py:3381) | x (all m rows, padding included, bf16), topk_ids (served dtype restored in replay), topk_weights (fp32 as apply normalised them), and the **served output**: apply's return value, the rank-local partial before the TP all-reduce. Layers 3, 23 and 44; m ≤ 64. |
| MoE bind | `_apply_rank_sliced`, decode and prefill branches (:3199, :3215) | Host only, once per (layer, m): the plan actually bound and its launch fields (tile K/N, block, …). Under FULL graphs this runs at capture, which is the binding every replay uses. |
| H1 | b12x_mla_sparse.py, right after `_logical_topk_to_physical_slots` (:3032), DCP1 kpool branch | Logical top-k, the **pre-compaction physical table**, the consumed causal length L (`cache_seq_lens_per_token`) and the position p (`input_buffers.positions`). |
| H2 | Decode branch, just before `bind` (:3186) | q as bound; the post-compaction table; consumed nsa and cache_lengths; a **device gather of every KV record the pre table references** (-1 maps to slot 0; out-of-range ids are clamped and counted); the slot-0 record; the cache slot count. |
| H3 | Before the decode branch's two `return`s (:3210, :3217) | The attention output that `forward_mqa` returns (and LSE when returned). |

Attention rows: up to 8 batch rows per armed step, in whole request groups (all verify rows of 1-2 requests).

### 3.2 Graph safety (the mapguard pattern, 22ao/22ap)

**Fixed shapes, no host sync.** Every hook is a fixed sequence of device ops: index_select, index_copy_, where,
remainder, cat and fills. Shapes depend only on static facts (layer, m, n). There is no `.item()`, `nonzero`, boolean
indexing or host tensor creation.

**Guard-owned meta vector.** The host refreshes it with an H2D copy in `prepare_attn` before every real forward:
- 4 pinned staging buffers, with an event wait only when the GPU is 4 or more steps behind;
- it holds the arm flags, sequence numbers, step id, token counts and the chosen attention rows;
- it is disarmed (stream-ordered) after `execute_model`.

The device picks the ring slot `(seq − 1) % R`, or a **sink slot** when disarmed. Replays therefore read current
values. Dummy runs and graph capture write only the sink.

**Seqlock and counter.** Each capture writes a header begin word first and an end word last, and bumps a device
commit counter.

**Poller (host thread, 1 s, side stream).** For each entry it copies the header, then the data, then the header again.
It keeps the entry only if both headers show begin = end = seq; a torn copy is counted and re-read.

**Order.** `done_step` (written by the layer-44 MoE hook every forward) is copied first. An armed entry whose step has
finished without committing is **abandoned**: it is resolved, and its quota is returned if every layer abandoned it.

**No overwrite before read.** The host assigns sequence numbers and never arms seq s while entry s − R is unresolved
(backpressure). A slot is therefore never rewritten before it is copied out.

**The 22ap lesson.** Host buffers are allocated with inference mode off, inside the poller thread.

**Numerics.** Engine tensors are only read; every store goes to guard-owned buffers. The cost per forward is about
20 MB of extra device copies (the sink writes included), which is microseconds against a ~60 ms step.

**Device memory.** 121.5 MiB on rank 0 at the served dimensions (`tests` G1/G2):
- MoE: 3 layers × 9 slots × 1.03 MiB;
- attention: 2 layers × 5 slots × 9.4 MiB.

The rings are allocated in `initialize_kv_cache`, after the KV cache and before graph capture. The served log shows
about 3.7 GiB below the 0.95 utilization target.

### 3.3 What cannot be captured faithfully, and the fallback

| Requirement | Status |
|---|---|
| Served MoE output inside FULL graphs | **Captured.** apply's return value is copied in-graph. |
| KV records in-graph | **Captured** as a device gather by the pre table at H2, immediately before `bind` on the same stream, so these are the bytes the kernel reads. A post-table slot not present in the pre table cannot be replayed; such a row is flagged and excluded (the CPU tests show compaction only moves pre-table entries). |
| Extend kernel (verify rows in steps mixed with a prefill) | **Not captured.** The extend bind takes per-request lengths and the whole batch structure. Attention is armed only on pure verify steps, where the decode kernel runs. Level 3 covered the extend kernel on synthetic inputs only. |
| MLA layers ≥ 25 (rank 1) and the rank-1 MoE half | Not captured. Rank 0 only. |
| m = 40 / 48 | Not served at 8 in flight on 0.9.1. Two sources, both labelled: **startup** (the stack's own release warm-up runs 16 parallel samples before readiness: real served steps at m = 40-64, non-GPQA prompts, captured as their own stratum); **composed** (real traffic rows from two m = 32 captures concatenated in replay). The vLLM-internal `_warmup` steps are never armed. |
| Exact positions under async scheduling | The device positions at consumption are captured. The host's row choice uses approximate categories; the final regime is computed offline from the captured p and L. |

## 4. Workload (`runner/client_p3b.py`)

**Panel.** The 22ao panel fixture (12 GPQA questions):
- read at run time from its PRIVATE path;
- its sha256 is checked against `livecheck-22ao/SHA256SUMS` before the GPUs are touched;
- sampling settings are taken from the fixture; stopping is natural.

**Requests.** 16, with fresh seeds 72301-72316 (no prior use under `<data-runs>`: test 8).
- 8 long requests: `max_tokens` 2,816 down to 1,024 in steps of 256. The longest prompt (q127, about 2.8k tokens) is
  first, so clamped rows exist from the start.
- 8 short requests (512).

**In flight.** A ramp of 1, 2, 4, 6, 8 at 4 s per level, so that padded sizes 4, 8, 16 and 24 are served early. Then
exactly 8 outstanding with backfill, then the drain. The ramp is the only change from 22ao's shape; it exists so
small-m coverage does not depend on reaching the drain inside the wall cap.

**Early stop.** The runner stops the client as soon as the poller reports `quotas_met`.

## 5. Arming policy (fixed before data; `MOE_TARGET`, `ATT_TARGET`)

**MoE.** Armed entries per padded size, traffic phase:

| Padded size | 32 | 24 | 16 | 8 | 4 | 1, 2, 40, 48, 56, 64 |
|---|---:|---:|---:|---:|---:|---:|
| Target | 10 | 6 | 6 | 4 | 4 | 2 each |

- Startup phase: 3 each at 40-64, and 2 at 32.
- At least 6 steps between captures of the same (phase, size).
- At most 96 armed entries in all.

**Attention.** Request groups per category:

| Category | Groups |
|---|---:|
| dense | 12 |
| boundary (some row with (p + 1) mod 64 in {0, 61, 62, 63}) | 16 |
| clamped (p + 1 > 2,052) | 10 |

- Verify steps only.
- At least 4 steps between armed steps; at most 48 armed steps.

**`quotas_met`.** The emitted traffic MoE entries meet 32: 10, 24: 6, 16: 6, 8: 4 and 4: 4 on every layer, and every
attention category reaches its target.

## 6. Coverage minima (fixed before data)

**Capture verdict** (`runner/coverage_p3b.py`): CAPTURE COMPLETE or INCOMPLETE with the deficit listed.
- MoE, traffic entries per layer: m = 32 ≥ 8; m = 24 ≥ 4; m = 16 ≥ 4; m ≤ 8 ≥ 2.
- Attention: emitted groups dense ≥ 8, boundary ≥ 10, clamped ≥ 6.
- Also required: no capture error, flush complete, and a bind record for every captured shape.

**Replay minima, anchored attention rows per layer.** These are computed from the exact captured p and L:

| Regime | Minimum rows |
|---|---:|
| dense with r = L − (p + 1) ≥ 1 | 32 |
| clamped (p + 1 ≥ 2,048) | 16 |
| boundary-crossing (⌈min(L, 2048)/64⌉ > ⌈(p + 1)/64⌉) | 8 |

Dense rows with r = 0 are reported as controls. A shortfall is **INCOMPLETE**, never negative.

## 7. Replay and outcome rules

### 7.1 MoE (`replay/replay_moe.py`)

The replay reuses P3's `moe_plan_eq.py` (sha256 pinned): the image's own EXL3 code builds the slice and plans under
the served env, and invokes them as `_apply_rank_sliced` does. The effects below are never combined.

1. **BASELINE (v091, tp2r0).** Every entry goes through the plan bound for (layer, m), with all m rows as served,
   twice, and through `apply`. It must be **bit-equal to the served output**. If it is not, that layer is reported as
   BASELINE NOT REPRODUCED, and its plan comparison is shown but not interpreted.
2. **E-plan (within engine).** Decode against prefill on the real rows, 2 repeats each.
   - A difference counts only if repeats are bit-equal.
   - It is characterized by layer, m and phase: per-row bit equality, rel L2, max abs and the fp32 raw difference,
     set against P3's synthetic magnitudes (`summarize_p3b.py`).
   - At m ≤ 8 the rows are expected equal (same tile); this is reported, not assumed.
3. **Composed m = 40 / 48.** Prefill(m) against decode in ≤ 32-row chunks. Labelled.
4. **Control.** A one-ulp input change must change exactly that row in both plans.
5. **v070 image, ep2r0 slice.** E-plan only, on the same captured rows. There is no baseline, because v070 did not
   serve them.
6. **Forced-tile confirmation (v091, separate section).** How the tile is forced:
   - a second copy of each layer is built through the same image path;
   - `exl3._load_b12x_fused_moe` returns a proxy whose `plan_weights` turns the stock `trellis_tile_config=None`
     (exl3.py:2480) into (64, 256, 64, 256);
   - `layer.exl3_trellis_tile_config` is set to the same tuple (it is part of the runtime cache key);
   - the decode runtime is planned by `_rank_sliced_runtime` under the served env with `VLLM_EXL3_PREFILL_TRELLIS=0`.

   The bound launch must show K64/N256 at m = 16-32; otherwise the result is FORCING NOT EFFECTIVE. It is compared
   with the stock prefill plan on the captured m = 16-32 rows and on P3's m = 16 / 32 synthetic fixtures (inputs
   regenerated bit-identically, digests checked).
   - **Prediction:** bit-equal.
   - **Outcomes:** PREDICTION HOLDS, PREDICTION FAILS, or INCOMPLETE.

**Verdicts per layer:**

| Verdict | Condition |
|---|---|
| INCOMPLETE | coverage deficit, an error, or a failed control |
| BASELINE NOT REPRODUCED | the served output was not reproduced |
| DIFFERENCES (characterize) | the plans differ on the captured rows |
| EQUAL ON CAPTURED ROWS | no difference on the captured rows |

### 7.2 Attention (`replay/replay_attn.py`, v091 image)

**Compact cache.** The pre-table records and the slot-0 record are re-addressed to compact slot ids. A slot gathered
twice must hold the same bytes.

**Plan.** The served decode plan is rebuilt from the logged Caps, with splits pinned by vLLM's own
`_prepare_b12x_sparse_mla_plan`. The rebuilt Caps are checked equal to the served ones.

**Runs and controls:**

| Run | What it is |
|---|---|
| A | The served inputs. Must reproduce the served output bit for bit, row by row. A row that does not reproduce is **NOT ANCHORED** and excluded (counted). A is also run twice for determinism. |
| Bc (count only) | Same post table; nsa = min(p + 1, 2048); cache_lengths = p + 1. |
| Bf (full path) | The physical pre table through the image's own `_compact_dropped_selection` and `_mask_page_table_after_nsa_len` with the exact count, then the kernel. Control: the pre table with the served count must reproduce the served post table and nsa. |
| P1 (required) | Halving nsa must change every row with nsa ≥ 2. |
| P2 (informational) | The level-3 plant: one foreign live slot appended. Reported as a detected fraction; a low-scoring key can vanish in bf16 rounding. |

**Verdicts per layer and regime:**

| Verdict | Condition |
|---|---|
| INCOMPLETE | anchored rows below the minimum, a P1 miss, or a kernel error |
| DIFFERENCES (characterize) | Bc or Bf differs from A on any anchored row |
| EQUAL ON CAPTURED ROWS | otherwise |

EQUAL ON CAPTURED ROWS extends the level-3 statement ("ruled down … under the verified masking/compaction conditions")
to served inputs in the captured regimes only.

## 8. Budget, GPU time per step, retries

**Shared ledger.** `costs.jsonl` holds the items `p3b-capture` and `p3b-replay`. Both wrappers gate on the total:
used + this attempt's cap ≤ 0.65 GPU-h.

| Step | Wall cap | Estimate |
|---|---|---|
| A capture (2 GPUs) | min(1,080 s, (0.65 − used − 0.05 replay reserve) × 1,800 s); refused below 990 s | Ready 743-874 s (22ao); traffic until the quotas are met about 60-120 s; flush and stop 40-70 s. Total 850-1,060 s = **0.47-0.59 GPU-h** |
| B replay, v091 (GPU0) | min(600 s, budget left) for the whole replay; v070 runs only with ≥ 140 s left | Import, load and plan about 40 s (P3 compile cache copied in); attention about 20 s; MoE baseline and E-plan about 20 s; forced-tile compile **unknown** (a new K64/N256 decode specialisation for 10-32 rows; up to about 90 s if cold). **0.02-0.05 GPU-h** |
| B replay, v070 | (see above) | About 60-120 s (P3: 238 s with its E-version slice). **0.02-0.03 GPU-h** |
| Total | | about **0.52-0.65 GPU-h**. If the capture uses its full 0.6, the replay has 180 s: v091 runs and v070 is skipped (recorded). |

**Retries.** Allowed: each attempt gets its own `attempt-N` directory and needs `ATTEMPT_REASON`. A capture retry
needs ≥ 990 s of wall time after the replay reserve, so in practice only an attempt that failed within its first
minutes leaves room for one.

**Cost lines.** Every container and every exit path writes one (skipped and interrupted ones included). The capture's
cost line counts up to the moment the GPUs are released. Copy-out happens afterwards.

## 9. Gates (capture runner, as `run_22ao.sh` with the 22ap re-sampled pre-flight)

- `ATTEMPT_REASON`;
- the budget gate;
- attempt numbering, never overwriting;
- frozen hashes (`SHA256SUMS`);
- the image pin (`runner/images.txt`);
- the fixture hash;
- pre-flight: no `glm53*` / `p3*` container; GPUs idle by memory and by re-sampled utilization; no foreign GPU
  clients;
- readiness by 930 s;
- **capture gate before traffic:**
  - rank 0 registered with rings and no issues (exactly one candidate module per captured layer; draft-named modules
    excluded);
  - rank 1 inert;
  - every hook site seen at capture time;
  - `async_scheduling` true.
- coverage watch;
- the FLUSH handshake (≤ 45 s);
- a watchdog at the wall cap.

## 10. Privacy

| What | Where |
|---|---|
| Captured tensors | Only `/tmp/p3b/PRIVATE-capture` in the container (dirs 0700, files 0600, explicit modes), copied after the stop to `captured-p3b/PRIVATE-capture/capture-attempt-N/` (0700) |
| Replay per-row positions and lengths | `PRIVATE-capture/replay-attempt-N/` |
| Logs, coverage and replay JSON | Shapes, digests, counts, categories, flags and comparison magnitudes only (test T10 / A6) |

The stopped container keeps its own copy under `/tmp/p3b` until the stack's next `start.sh` replaces it, as 22ao's
mapguard data did.

## 11. Open risks and what the CPU tests could not cover

**Not exercised on CPU** (no CUDA available):
- real `torch.cuda.graph` capture and replay of the hooks. The CPU test shows the op sequence is identical for every
  meta value (T6), which is the property capture relies on;
- side-stream copies racing a live engine stream;
- pinned-memory behaviour;
- headroom for the 121.5 MiB of rings plus graph-pool temporaries (about 20 MB per captured graph);
- the B12x decode kernel and the MoE kernels (the replays used reference or `dry_kernel` stand-ins).

**Possible start-up failures,** caught by the capture gate before traffic (cost about 0.45 GPU-h):
- served module names differ from the expected ones (MoE modules are matched as "mlp.experts" holding
  `exl3_trellis_weights`; attention by `B12xMLASparseImpl` on `runner.model`);
- the decode-kernel attention path is not reached before readiness.

**Whether the attention baseline reproduces with fewer rows than served.** The replay runs ≤ 8 rows, while the served
batch had up to 32 under a pinned 32-split plan. If rows interact through the plan, A does not reproduce: rows are NOT
ANCHORED and the regime is INCOMPLETE, which is visible rather than silent.

**Forced tile:**
- `prepare_weights` and the planner have not been run on GPU with an explicit tile on the projection-mixed path;
- the compile time of the new specialisation is unknown;
- the soft deadline turns a slow compile into INCOMPLETE.

**Quota timing.** Boundary groups need r ≥ 1 at specific positions. Clamped groups need contexts over 2,052 (q127
from the start, long requests after about 30-45 s). If traffic ends first, the capture is INCOMPLETE.
