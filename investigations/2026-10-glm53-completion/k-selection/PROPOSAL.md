# K: model-free validation of the B12x sparse-selection kernels at long context (PLAN.md amendment 21, section 3)

2026-10-13. Prepared on CPU only: no GPU was used, no GPU or glm53 container was started or stopped,
`<stacks>` was not touched, and no `sudo` was run. CPU containers ran with `--network none` and no `--gpus`.

**Status: PREPARED, NOT RUN.** The GPU stage needs its own authorization. The proposed ceiling is 0.5 occupied
GPU-hours on 1 GPU. The estimate is 0.16 GPU-hours.

## 1. Question and scope

Source review item 2 (`source-review-lc/SOURCE-REVIEW.md`) found that the selection kernels change regime with
context length. These regime changes are exact by design, but they have never been tested on live data past about
120k tokens:

- **B1, tiled route:** a second slice is live from 65,536 tokens, a second supertile chunk from 131,072, and further
  slices from 196,608 and 262,144.
- **B1, fused route:** per-CTA trims start near 22.5k, 47k and 96k (for 4, 2 and 1 requests), and repeated trims
  near 42k, 88k and 180k.
- **0.7.0:** the thresholds sit at about twice these lengths (DCP2 halves the pools per rank), and the fused merge
  is serial.
- **Existing coverage:** real-input checks reached about 120k. The B12x tests stop at 32,768 rows.

K asks one question: at context lengths on both sides of every one of these boundaries, up to the 330k-token
equivalent, do the actual kernels do two things?

- Select the correct top-512 pool set.
- Produce the correct final attended selection after the column-511 truncation and tail policy.

K is model-free. No model weights are loaded, no server runs and no sampling happens.

## 2. What is driven

Each engine runs inside its own pinned image. Nothing on the selection path is re-implemented: the fixture calls
the image's own code (`kselect/kdrive.py`).

| Step | Function called from the image | Engine call site |
|---|---|---|
| Top-k | `vllm…sparse_attn_indexer._run_b12x_paged_topk`: `plan_indexer_scratch` → `bind` → B12x `index_topk_fp8`, tiled or fused route as the plan decides | `sparse_attn_indexer_kpool.py` b0:928-954 |
| DCP2 merge (0.7.0) | `_merge_b12x_dcp_topk`: the image's Triton local→global convert, candidate unpack, B12x `run_row_topk` over 1,024 candidates, Triton gather-by-position | kpool v070:924-931 |
| Column-511 policy (B1) | `_move_lowest_selected_pool_last` | kpool b0:63-76, :966 |
| Consumer | `expand_pools_and_append_tail(pool_ids[:, :511], dec_seq, 4)` | kpool b0:1043-1044 |

Only three things are replaced, and only where no GPU or second rank exists:

- **Workspace manager.** It returns `torch.empty` buffers pre-filled with a poison byte: 0x00, 0xFF and 0x7F on
  repeats 0, 1 and 2. A route that depends on stale or zeroed scratch therefore shows up as a failure.
- **DCP group and all-gather.** A two-rank stub replaces them. Rank r's slot holds the call's own packed candidates.
  The other slot holds the other rank's candidates, produced by the same image kernel. Both rank perspectives are run
  and must agree bit for bit.
- **`index_topk_fp8` wrapper.** It passes everything through unchanged and only reads the binding's route, CTAs per
  group and merge threshold.

The output score buffer is pre-filled with NaN, so any slot a route leaves unwritten is visible.

The routing environment variables are absent, as they were on the served containers:

- `B12X_FUSED_INDEXER`;
- `B12X_INDEXER_TWO_LEVEL_FOLD*`;
- `B12X_PAGED_INDEX_SUPERTILE_K`.

Their absence on the served containers is recorded in `b1-screen/run-s1236/inspect-s1.json` and, for a 0.7.0-based
image, in `gate/server-a1/container-inspect.txt`. The fixture aborts if any of them is set.

**Pins.** The images are pinned by id:

- B1: `sha256:799d3cf8e5d1…`;
- stock 0.7.0: `sha256:48e254d94f58…`.

Inside each container, the nine selection-path files are checked against `kselect/pins.json`. The B12x files and
`sparse_attn_indexer.py` agree with the source review's `sources.sha256`. B1's kpool file is B0 plus `d7-fix.diff`.
The run aborts on any mismatch. It also aborts unless the device is sm120 with 188 SMs, because the fused CTA
boundaries depend on the SM count.

**Geometry.** Rows are requests × (K+1) at the captured sizes, with the engine's page-table widths:

| Engine | Width | Tiled rows | Fused rows (CTAs per group) | Other |
|---|---|---|---|---|
| B1 | 4,097 pages | 32, 24 | 16 (11), 8 (23), 4 (47) | |
| 0.7.0 | 2,058 pages per rank | 48, 24 | 16 (11), 8 (23) | interleave 1; local pools per rank from `get_dcp_local_seq_lens`; zero-length padding rows where rows exceed requests × 6 |

Each request has K+1 rows at consecutive token lengths L−K..L, so rows of one request sit on both sides of a pool
boundary. Pool count is L // 4, as in the indexer metadata (`indexer.py` b0:1110).

## 3. Boundaries covered

The boundaries are computed from the image constants in `kselect/kplan.py`. Test T6 checks them against the image's
own `_plan_two_level_fold`, fused routing, carry slack and B12x constants.

A boundary b is in local pool rows. A regime is live in a row when that row's local pool count exceeds b.

Each boundary gets 12 request end-lengths:

- **DCP1:** b−512, b−64, b−2…b+3, b+64, b+65, b+512 and b+1024 pools.
- **DCP2:** the same pattern around 2b, so that rank 0 crosses at 2b+1 and rank 1 at 2b+2.
- **Token phase:** cycles 0..3 so that every tail count occurs.
- **Fused trim boundaries:** 4 more end-lengths at ±64·CTAs pools. These bracket the neighbouring per-CTA page counts
  in case the source reading of the deferred trim condition is off by one page.

Coverage on both sides is asserted for every boundary and every rank (`kplan.build`).

| Engine / route (rows) | Boundaries, in tokens (both sides tested) | Controls |
|---|---|---|
| **B1 tiled** (32, 24) | slice 2 at 65,536; slice 3 + chunk 2 at 131,072; slice 4 at 196,608; slice 5 + chunk 3 at 262,144; slice 6 at 327,680 | P≈512 (2k), 120k (stage-2 overlap), 330,000 (max) |
| **B1 fused 16** (11 CTAs) | final trim 22,528; mid-loop trim 1 at 42,240; trim 2 at 61,952 | same |
| **B1 fused 8** (23 CTAs) | 47,104; 88,320; 129,536 | same |
| **B1 fused 4** (47 CTAs) | 96,256; 180,480; 264,704 | same |
| **0.7.0 tiled** (48, 24), per rank | slice 2 at 131,072; slice 3 + chunk 2 at 262,144 | 4k, 120k, 330,000 |
| **0.7.0 fused 16** (11 CTAs, serial merge) | 45,056; 84,480; 123,904 | same |
| **0.7.0 fused 8** (23 CTAs, serial merge) | 94,208; 176,640; 259,072 | same |

The plan has 56 cells, 2,310 batches × 3 repeats and 6,930 selection calls (0.7.0's are per rank, plus 2 merges).
Tested token lengths run from 1 to 331,779. `kselect/kplan.out` lists every cell.

## 4. Inputs (`kselect/kinputs.py`)

**Families.** Every batch runs all seven families.

- **real_L3, real_L43.** Real index-K pools and query/weight rows from the stage-2 decode captures (stock 0.9.1,
  `phase2-stage2/probe-a1,a2`), built into `bank/` by `kselect/build_bank.py`.
  - The bank holds 58,599 distinct live pools per layer, from 6 requests (the longest is 30,051 pools), and 1,160
    query rows.
  - Sanitisation: NaN K bytes were replaced by zero (0, 0 and 16 in layers 3, 23 and 43), and 96 pools per layer
    whose scale was not positive and finite were dropped. The rest is byte-identical.
  - Past 58,599 pools the stream wraps. Reused pools are derived by a fixed per-wrap e4m3 sign flip on 64 dims, so
    no exact duplicates are manufactured (checked in T2). At 330k, about 29% of pools are derived.
  - The bytes come from stock 0.9.1 decode, so some were built by the inv-D2 path. For a selection test only the
    score distribution matters: gaps, ReLU zeros and exact ties.
  - `bank_L23.pt` was built but is not used by the frozen families.
- **gauss.** The stage-1 engine-style generator: Hadamard, bf16, ue8m0 scale, e4m3 (`phase2-stage1/fixture.py:56-104`).
- **Exact-arithmetic adversarial families: plateau, zero, ulp, edge.**
  - Construction: head 0 only, `q0 = [8, 2⁻⁹×6]`. Each pool encodes `8·k0 + m·2⁻¹⁸`, with m written in six base-8
    e4m3 digits and scale 1.
  - Exactness: every partial sum is a multiple of 2⁻¹⁸ below 64, so the scores are exact in any fp32 order. In the
    top band, consecutive m values are exactly 1 ulp apart.
  - **plateau:** 40 pools tie exactly across rank 512.
  - **zero:** fewer than 512 positive scores; the rest is exactly 0.
  - **ulp:** 600 consecutive 1-ulp scores, with rank 512 between neighbours that differ by 1 ulp.
  - **edge:** the newest pools and the pools at and after the boundary, slice and chunk starts and CTA-share edges
    take ranks 0-9 and 505-519, interleaved.
  - plateau, zero and edge space m by 128, so they stay exact even with a 17-bit accumulator. ulp needs full fp32
    accumulation.

**Poison (every family).** Pools that outscore every live pool are written in three places:

- the unused slots of each request's last live page;
- two extra pages appended to its block-table row;
- the null page 0, which every later entry and every padding row point to.

All of them are ineligible, so selecting any of them fails C1. The tiled scorer covers the whole capacity
(`active_width` is None), so it does read these pages.

## 5. Reference and its validation

The fp64 reference is the score `scale·Σ_h w_h·relu(q_h·k)` over the decoded e4m3/fp32 values (`kselect/kref.py`).

The fp32 evaluation bound is `B = 162·2⁻²⁴·|scale|·Σ_h|w_h|·Σ_d|q||k|`, unchanged from stage 1. On stage 1 the
kernel score error was at most 0.0047 B.

CPU validation passed in both images (`cpu-tests/cpu-{b1,v070}.json`, 71/71 each):

| Check | Result |
|---|---|
| Reference vs the image's own B12x torch reference, on packed caches | worst 0.0042 B (real), 0.0010 B (gauss), 0 (exact); everything beyond the live length is −inf |
| fp64 reference vs exact rationals (`fractions`) | exact families equal; general families within 0.0005 of the fp64 rounding bound; ordering preserved |
| Three fp32 summation orders | within 0.0037 B; bit-exact for the exact families, including a full 82,500-pool row |
| fp64 rounding vs the band | fp64 rounding (≤ 2⁻²⁹ B) is negligible beside the band, so it is ignored |

## 6. Frozen criteria

The criteria are in `kref.py`. They are enforced by the runner's `SHA256SUMS` check: an edited file refuses to run.

For each row, with P the row's live pool count, n = min(512, P), and t the n-th largest reference score:

- **C1, structure and eligibility.** Exactly n ids lie in [0, P), with no duplicates, the rest −1 and none below −1.
  Applied per rank and to the merged list on 0.7.0.
- **C2, selected set.**
  - Exact families: every s > t is selected and nothing with s < t is.
  - Other families: every s > t + B + B_t is selected, and nothing with s < t − B − B_t is.
- **C2b, kernel scores of the selected ids.** Bit-equal to the reference for the exact families, within B otherwise.
- **C3, B1 column-511 policy.**
  - The multiset is unchanged by the swap.
  - The column-511 id carries the minimum kernel score.
  - Its reference score is the selected minimum: exact, or within B of it.
  - When P < 512, column 511 is −1.
  - Columns 0..510 pass C1 and C2 as a top-min(511, P) set. This is the final attended pool set.
- **C4, final attended selection.** The image's expanded output (rows × 2047) equals an independent reference: pool p
  expands to tokens 4p..4p+3 only if p < L//4, and the tail columns 2044-2046 hold 4·(L//4)+t for t < L % 4. Every
  token is below L, no token appears twice, and there are exactly L % 4 tail tokens.
- **C5, 0.7.0 merge.** Both rank perspectives give bit-identical merged ids and scores, and the stub's own-rank slot
  equals the independently produced candidates. The global C1 and C2 also apply.
- **C7, route as planned.** The binding route, CTAs per group and the image's own fold plan (slices and total slices)
  must match the plan.

**Ties and rounding.**

- Ties at t are resolved by no route by design, so any subset of the tied pools is accepted.
- −0 and +0 compare equal (IEEE).
- Within the fp32 band the choice is free. The band size per family is reported, so that a wide band cannot pass
  silently.
- NaN-valued inputs are excluded by construction: all inputs are finite.

**Verdicts per cell**, applied to every row × repeat:

| Verdict | Meaning |
|---|---|
| PASS | All criteria hold |
| FAIL | Any C1, C3 or C4 failure; any C2 failure where the scores were exact or within the bound; or a C5 failure |
| FAIL-NUMERIC | An exact family's kernel scores were not bit-exact. C2 is then judged with the bound, and a failure there is still FAIL |
| INCONCLUSIVE-BOUND | A general family's kernel score exceeded B |
| INVALID-ROUTE | The cell did not run the planned route |
| PARTIAL / NOT-RUN | Stopped by the time guard |
| ERROR | Harness exception |

**Descriptive only** (not criteria):

- The rank of the pool that column 511 drops under the shipped policy: stock 0.9.1 for B1's kernel output, and 0.7.0
  as merged. The 0.7.0 column-511 behaviour is known (stage 2).
- Kernel scores of unfilled slots.
- Whether poison would have won in the general families.
- How many pools were derived after a wrap.

**Ordering where the consumer depends on it.** For L ≥ 2,048 the MLA reads all 2,048 columns and masks −1, so it
does not depend on order (SOURCE-REVIEW section 3; `b12x_mla_sparse.py` v091:3033-3046). Order matters only through
two things, and both are checked:

- the `[:, :511]` truncation (C3 and C4);
- the fixed tail columns (C4).

## 7. CPU tests (`kselect/test_cpu.py`; `cpu-tests/`)

The suite runs in both images, CPU only, with `TRITON_INTERPRET=1`. Each image passed 71/71, with code hashes equal
to the frozen `SHA256SUMS`.

| Test | What it checks |
|---|---|
| T1 | The reference, as in section 5 |
| T2 | The families have their designed properties (plateau straddles rank 512, zero has < 512 positive, ulp neighbours are exactly 1 ulp, edge ranks land at hot positions); poison outranks the live pools; bank wraps create no duplicates |
| T3 | Checker mutation tests: duplicate, ineligible id, −2, a missing slot, a lower pool swapped in, plateau subsets accepted, ±0, tolerance band in and out, 1-ulp score mismatch, column 511 not the minimum, P < 512 placement, corrupted or missing tail. T3b: the sync-light checkers used on GPU equal the straightforward ones on 336 random and mutated cases |
| T4 | The image's `expand_pools_and_append_tail` (Triton, interpreted) equals the image's torch composition and the independent reference. B1's `_move_lowest_selected_pool_last` satisfies C3. 0.7.0 has no column-511 fix |
| T5 | `get_dcp_local_seq_lens` equals the plan; the Triton convert, unpack and gather kernels (interpreted) are correct; the real `_merge_b12x_dcp_topk` through the stub gives the global top-512 with equal perspectives (`run_row_topk` stubbed on CPU only) |
| T6 | Fold plan, fused routing, carry slack 448 and constants match the image. Merge arm: B1 cooperative at 11/23/47 CTAs; 0.7.0 serial (forced last-CTA) at 11/23 |
| T7 | Packing round trip; poison only beyond the live count; null page poisoned |
| T8 | End-to-end `run_batch` with a fake kernel built on the image's B12x torch reference (reduced width). The honest fake passes every family. Four faults per image are caught by the intended criterion: eligibility ignored → C1; newly live region dropped → C2; bf16-rounded ranking → FAIL; and either the column-511 fix skipped (B1, → C3) or merge perspectives differing (0.7.0) |
| T9 | `run_cells`: verdicts and output files; the time guard leaves cells NOT-RUN and still writes the summary |

**Triton interpreter caveat.** The interpreted kernels do only integer index arithmetic and float copies, which numpy
executes exactly. The interpreter does not model GPU memory ordering, races or faults, so T4 and T5 validate
semantics, not the compiled kernels. The CuTe DSL selection kernels cannot run on CPU; the GPU stage is their only
test.

**In-container guards** (`refusal-tests/in-container-guards.txt`), run on CPU in the real images:

- the pins pass and the run aborts at the device check;
- the wrong engine for the image aborts at the file pins;
- `B12X_FUSED_INDEXER` set aborts at the environment check.

## 8. Runner (`run_k_selection.sh`)

**Usage.**

```bash
APPROVED_GPU_H=0.5 ./run_k_selection.sh                # full run: B1 container, then 0.7.0 container
APPROVED_GPU_H=0.5 K_SMOKE=1 ./run_k_selection.sh      # 1 batch of 3-4 cells per engine, same budget item
APPROVED_GPU_H=0.5 DRY_RUN=1 ./run_k_selection.sh      # frozen files, budget gate and image pins only
```

**Guards and failure handling.**

- One GPU (`K_GPU_INDEX`, 0 or 1), `--network none`, read-only mounts.
- Frozen-file check against `SHA256SUMS`, then image-id pins.
- Budget gate: approved ≥ spent (from `costs.jsonl`, item `k-selection`) + 0.5.
- GPU-idle pre-flight, repeated before the second container. It refuses if:
  - any glm53, vllm or fixture container is running;
  - memory is above 1 GiB or utilisation above 5%;
  - any process other than a desktop graphics client appears in the plain `nvidia-smi` report.
- Wall caps: 850 s per container inside a 1,800 s total. An inner time guard of 775 s ends cleanly, and cells run in
  priority order (tiled first).
- Every exit path runs `docker rm -f` and writes one cost line. `docker run` runs in the background behind an
  interruptible `wait`, so INT or TERM cleans up at once.
- No `set -e`.

**Refusal and cleanup tests** (`refusal-tests/run_refusal_tests.sh`; `results.txt`) passed 16/16. Shims stand in for
`docker` and `nvidia-smi`, so no real container is started; only R07 uses the real docker, for `image inspect`.

| Group | Scenarios |
|---|---|
| Refused before any GPU use | no approval; approval below the ceiling; non-numeric approval; 0.3 GPU-h already spent; an edited frozen file; image id not matching the pin; an inference container running; GPU memory or utilisation busy; a non-desktop graphics client; a bad GPU index; a test root outside `refusal-tests` |
| DRY_RUN with the real images | passes and writes no cost line |
| Run path | idle GPU with a desktop client: one "done" cost line, both containers removed, REPORT written |
| Cumulative budget | a second run after a billed run, read back in the runner's own cost-line format, is refused |
| Interrupt | TERM mid-run: exit 130 within 4 s, an "interrupted" cost line, the container removed |
| Wall cap | `wall-cap` status, both containers removed |

**Outputs.** Each run writes these files to `run-<UTC>/<engine>/`:

- `summary-<engine>.json`, with per-cell verdicts and counts by family;
- `results-<engine>.jsonl`, one line per batch × repeat;
- `failures-<engine>.pt`, holding the ids of failing rows (capped);
- `log`, `gpu.txt` and `container.log`.

It also writes `REPORT.txt` (`kselect/report.py`).

## 9. Cost (`cost/cost.py`, `cost/cost.out`)

| Term | Expectation | Source |
|---|---|---|
| Start-up | 45 s per container | the stage-2 selftest ran end to end in 10-12 s |
| Compile | 20 s per route shape | |
| Selection call | ≤ 1.5 ms tiled, length-independent because the scorer covers the full capacity; ≤ 0.5 ms fused; plus 3 ms of plan, bind and poison per call | B12x docstring: tiled r64@131K 1,043 µs, fused r1@131K 41 µs |
| fp64 reference | 0.5 TFLOP/s effective | |
| Checks | sync-light: about 1 ms per row × repeat | |

| | Seconds |
|---|---|
| B1 | 262 |
| 0.7.0 | 273 |
| **Total** | 565 (0.157 GPU-h) |
| Pessimistic (every term ×3) | 1,664 (0.462 GPU-h) |
| **Ceiling** | **0.5 GPU-h** |

In the pessimistic case the last fused cells of each engine would end NOT-RUN under the time guard. They would be
reported as such, not dropped.

## 10. What a clean result retires, and what it does not

**Retires**, for the tested lengths, row counts and geometries:

- Kernel-level selection failures on B1's tiled and fused routes, and on 0.7.0's DCP2 per-rank routes with the
  cross-rank merge:
  - a wrong or incomplete top-512 set across slice, supertile-chunk, CTA-share and trim transitions up to the
    330k-token equivalent;
  - a missed newly live slice or chunk;
  - an ineligible, padding or poison id;
  - dependence on stale scratch;
  - mishandled exact ties or 1-ulp neighbours;
  - a broken DCP2 merge.
- For B1, also: a final attended selection that is not the top-511 plus tail.

Together these close source-review item 2's untested invariant ("UI beyond about 120k") for these kernels.

**Does not retire, and does not show:**

- **Termination equivalence,** or any claim about reasoning-close or EOS probabilities, or about completion rates.
- **Whether the index-K bytes the engines write at those lengths are correct.** That is inv-D2, a writer defect that
  K bypasses by construction.
- **Prefill selection.** It uses a different kernel, `top_k_per_row_prefill`.
- **The MLA attention computation itself, KDA state, speculative verification or sampling.**
- **CUDA-graph-captured execution.** The same kernels and route are chosen capture-statically, but K calls them
  eagerly with fresh scratch, not under replay with a shared workspace.
- **Row counts other than those tested.** Rows 40-64 (more than 8 requests) are not tested.
- **The engine's actual padding-row representation.** K assumes zero-length rows.
- **Behaviour under concurrent kernels.**
- **Real score distributions beyond the bank.** At over 58,599 pools the "real" families include derived pools.
- **The shipped column-511 drop on stock 0.9.1 and 0.7.0.** It is reported descriptively, not judged; it is a known
  property.
- **Geometry assumptions.** The 0.7.0 per-rank width of 2,058 pages is source-derived (geometry table), not observed in
  a log. Only the fold plan depends on it: 9 slices at 2,058 against 8 at 2,048.

A FAIL is a selection defect at that cell and nominates nothing beyond it. Under amendment 20/21 rules it would be
classed and reported before any other use.

## 11. Blocking items and assumptions

- **Authorization.** The GPU stage needs explicit authorization of 0.5 GPU-h on 1 GPU.
- **GPU must be idle.** At preparation time a GPQA pass held the GPUs, and the real pre-flight would refuse.
- **SM count.** The fused CTA boundaries assume 188 SMs on sm120. The fixture aborts on any other device.
- **Optional smoke.** A smoke run (`K_SMOKE=1`) would confirm route introspection (C7) and timing on the GPU before
  the full run, under the same budget item. It is optional.

## Files

| File | Contents |
|---|---|
| `kselect/` | `kplan.py` (+ `kplan.out`), `kref.py`, `kinputs.py`, `kdrive.py`, `run_k.py`, `inner.sh`, `report.py`, `test_cpu.py`, `build_bank.py`, `pins.json` |
| `plan.json` | The frozen cell plan |
| `bank/` | Built by `build_bank.py`; `build_bank.out` |
| `cpu-tests/` | CPU test results for both images |
| `refusal-tests/` | Shims, scenarios, `results.txt`, `in-container-guards.txt` |
| `cost/` | `cost.py`, `cost.out`, `cost.json` |
| `costs.jsonl` | A CPU-preparation line only (item `k-selection-cpu-prep`; the GPU item `k-selection` has spent 0) |
| `SHA256SUMS` | The frozen inputs the runner verifies |
| `RECEIPTS.sha256` | Test outputs and this proposal |
