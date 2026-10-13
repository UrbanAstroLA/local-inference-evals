# D7 step 2 — B1 serving verification: frozen rules, criteria and files

PLAN.md amendments 18, 18a (binding spec) and 18b (budget). Prepared on CPU only; frozen before any B1 data.
The GPU session is launched by the coordinator, only after the operator's go-ahead.

Evidence labels: **[M]** measured by running code here; **[S]** read from source; **[I]** inferred.

## 1. What runs

| Item | Value |
|---|---|
| B0 | `local/glm53-v0.9.1-bundle-d1234:local` = `sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548` (unchanged) |
| B1 (serving) | `local/glm53-v0.9.1-b1-d7:local` = `sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a` |
| B1 build check [M] | staged file == B0's file + `d7-fix.diff` (`146a7c6b…`); full filesystem diff against B0: 150,806 entries each, 2 changed (`sparse_attn_indexer_kpool.py` and its `.pyc`), 0 added, 0 removed (`b1-build/fs-diff-summary.txt`) |
| Guard image | `local/glm53-v0.9.1-b1-mapguard:local` = `sha256:4d7bd73e6f2ef6485289dcb5bcb2f2a65f1846a5959cde7cbd9c275471c72e16` = B1 + amended guard module + hook lines (same method as b0-mapguard: `guard/make_patch.py`, `build/make_build.sh`; `b12x_mla_sparse.py` and `scheduler.py` hook copies byte-identical to b0-mapguard's) |
| Guard module / hooked index-K file | `mapguard_hooks.py` `a8e59fa53590253584ccb1057ddc57b1cbf3946a1b9c143a3f76ab7ac65aa663`; hooked `sparse_attn_indexer_kpool.py` `30ff8bdea2bfba185bf0bf5b591e7999d451baad12238761681f60d56b55b35f` (B1 file `146a7c6b…` + 9 hook-marker lines) |
| Recipe fork | `<recipes>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234+d7` (`local-recipe fork`; only `IMAGE` changed to B1's id; status `draft-untested`; not captured, not started) |
| Runner | `run/run_mapguard.sh` (attempt dirs `run-b1-aN`, item `b1-mapguard`, costs `costs.jsonl`, image pin `run/images.txt`) |
| Workload | the frozen amendment-17 workload (P0 warm T0 F0 T1 F1 D S Q P1: same requests, seeds, order as run-a3), both ranks, shipped graphs, DFlash2 ×3, then phase 2d |

## 2. Guard amendment (2a)

Changes against the frozen B0 module (`b0-mapguard/guard/mapguard_hooks.py`, `3fa979c8…`). Every other site,
invariant, class, coverage minimum and verdict rule is unchanged.

**B1 write rule at idx_w_pf.** The call site now tells the pre-write hook what the writer receives:
`ring_mode="ring"` with the writer's per-token tail slots, or `"none"` (call without ring). For a token whose writer has a
ring, every completing pool is EXPECTED written (no `missing_pred`, no `pool_cross` from the predicate model). For a token
without a ring (call without ring, or tail slot < 0), B0's predicate and B0's batch gathering apply as frozen, and a
completing pool that straddles the chunk start is counted **`d7_fallback`**. Legacy calls without `ring_mode` (the frozen CPU
suite) keep B0's predicate exactly and are counted `legacy`.

**Independent content oracle.**
- Logical positions come from the ledger (scheduler `num_computed` + offset in the row, rebuilt from SchedulerOutput only;
  prefill rows with lo == hi and no drafts), never from the engine's positions buffer or the ring.
- After every prefill write, the post-write hook snapshots each row's trailing incomplete-pool raw keys and gates (the last
  ≤3 positions of the chunk) into a guard-owned per-layer shadow keyed by (layer, request tag, logical position).
- A straddling pool's earlier members are taken from that shadow, with tag and position verified on read. Its own members
  come from the row's batch tokens, selected by ledger position. The tail ring that B1's writer reads is never used.
- Missing members are counted `s_unavail` (not checked, never a violation). This covers a previous chunk not seen by the
  armed guard, a shadow slot collision, and members stashed by the decode path.

**Byte semantics.** The expected record is produced by the engine's own compress kernel (`kpool_compress_and_write_cache`)
on the oracle's members, written into a guard-owned scratch page set.
- The scratch has the engine cache's page size, row width and page stride, so the kernel gets the same constexpr
  specialisation as the engine call.
- On GPU its bf16 RNE, softmax, Hadamard, E4M3 and scale encoding are therefore the GPU kernel's by construction.
- The independent part is member identity, position and content. The compression arithmetic is shared, and is not the
  subject of D7: the kernel is byte-identical in 0.7.0–B0 and unchanged by the fix.
- On CPU the same kernel runs under the Triton interpreter (truncating bf16, mis-rounded E4M3 carry), so CPU tests use
  interpreter-matched conversions. They also check the oracle against the independent plain-torch reference
  (`d7lib.ref_record`). **GPU faithfulness is established only by the positive control in serving.**

**Positive control (frozen rule).** Every ALIGNED completing pool of an exact prefill row (all four members inside the
chunk, written by the path B1 leaves unchanged) is recomputed by the same oracle and compared byte for byte (`pc_checked`,
`pc_mism`).
- Any `pc_mism > 0` on a rank makes that rank's straddling byte checks **INVALID**: `s_missing` and `s_mism` are then
  reported but not counted as violations, and the run cannot be CLEAN.
- Oracle disagreement on aligned pools is never counted as a D7 violation. It is reported with byte snapshots, and its
  cause (oracle infidelity or a new non-D7 defect) is diagnosed separately.
- `d7_fallback` and every slot-level class stay violations regardless.

**Outcome per straddling pool.** The pre-write bytes at the expected slot are snapshotted before the writer runs.

| Outcome | Condition |
|---|---|
| `s_verified` | stored == expected != pre |
| `s_indet` | stored == expected == pre (write not observable) |
| `s_missing` | stored == pre != expected (**violation**) |
| `s_mism` | stored != expected and != pre: foreign, stale or wrong-position members (**violation**) |
| `s_unavail` | no oracle (info) |

Every straddling event is logged (`slog`: step, layer, tag, position, outcome, rank, row, slot). Failing pools keep
byte snapshots (expected / stored / pre) and the standard record (request/row, rank, layer, step, position, slot,
ownership).

## 3. No-ring fallback (reachability, frozen decision)

B1 falls back to B0's rule when the call gets no ring or a row has no tail slot. Under B1's served configuration this is
**unreachable** for unaligned prefill rows. The decision is nonetheless frozen as a violation.

| Fallback condition | Why unreachable |
|---|---|
| Call-level: `d7_tail is None` (`b1-build/sparse_attn_indexer_kpool.py:473-480`) | **`tail_kv_cache` / `tail_prefix` None:** the GLM-5.3 indexer always builds `Glm5NextTailCache` (`models/glm5next/nvidia/attention.py:303-309`, passed at `:324`), and the op receives `self.tail_cache.kv_cache` / `.prefix` (`sparse_attn_indexer_kpool.py:1282-1283`) [S]. |
| | **`positions` None:** `Indexer.forward` passes `positions=positions` (`attention.py:422`, and `sparse_attn_indexer_kpool.py:1233`) [S]. |
| | **`VLLM_KPOOL_SKIP_TAIL_CACHE=1`:** unset in the recipe env and in the stack [M: grep]; absent from run-a3's container env dump [M]; not in B1's image ENV [M]. |
| | **Tail metadata absent:** the same condition gates the ring seed, and the tail builder supports all graph modes (`indexer.py:584-630`) [S]. |
| Row-level: tail slot < 0 | `compute_kpool_tail_slot_mapping` returns −1 only for the null block (`indexer.py:568-572`). `KpoolTailManager` gives every request exactly one block at admission, never prunes and never prefix-caches it (`single_type_kv_cache_manager.py:1097-1187`). The null block is removed from the free queue at pool init (`block_pool.py:190`), so `get_new_blocks` never returns it [S]. |

**Assertion (frozen before data).** The guard counts **`d7_fallback`**: a completing straddling pool whose writer had no ring.
It is a **VIOLATION** (it means a missing or cross-request pool, and falsifies "unreachable"). The analysis also marks the
run **INVALID** if `VLLM_KPOOL_SKIP_TAIL_CACHE=1` is reported in the guard's init record.

One residual case is not observable by the guard: `positions is None` skips the guard hooks as well as the ring. It is
unreachable per the source lines above.

**Out of claim.** KV-connector / PD external-token starts (not configured; no `kv_transfer_config` in the recipe).
Decode-classified short prefill chunks (≤ 4 tokens; ring decode kernel, D2 tests) appear only as `s_unavail` when a later
chunk needs their members.

## 4. Criteria and verdict (frozen)

**Unchanged from amendment 17.**
- Invariants, violation classes and verdict precedence: VIOLATION > INVALID > INCOMPLETE > CLEAN.
- All 28 coverage minimums C1–C12, per rank where marked, in the window from the first non-warm-up step.
- Transmission, exact-position and rank-ownership rules.
- FULL-graph capture and non-default args equal to the B0 screen.

**Added (B1-1 … B1-6, `analysis/analyze_mapguard.py`).**
- **B1-1 Violations:** `s_missing`, `s_mism`, `d7_fallback`. `missing_pred` and `pool_cross` remain violations; the
  amended guard raises them only for no-ring tokens.
- **B1-2 Positive-control rule:** as in section 2.
- **B1-3 Coverage:**
  - **C13_d7_straddle_events ≥ 100 per rank:** distinct (step, request, position) straddling pools byte-verified on every
    owned indexer layer of the rank (all layers `s_verified`/`s_indet`, at least one verified). For reference, attempt 3
    had 217 events per rank-layer set.
  - **C14_d7_positive_control ≥ 10,000 aligned pools per rank.**
- **B1-4 INVALID:**
  - legacy prefill-hook calls in the window;
  - straddling-event log overflow (65,536 rows);
  - the guard module lacks the amendment;
  - `VLLM_KPOOL_SKIP_TAIL_CACHE=1`.
- **B1-5 Window:** ends at the readout taken after P1 (the last readout before the client's `phase2d_start` record; the
  client waits 6 s with the engine idle). Phase 2d is outside every coverage count and the verdict. Guard counters and
  scheduler exclusivity in the 2d window are reported separately; any 2d-window violation is reported prominently and
  diagnosed before B1 is called verified.
- **B1-6 Output identity:** the four short greedy probes q0–q3 (P0 and P1) must be bitwise identical to B0's attempt-3
  record (`run-a3/probe.jsonl`: e705960ca1e45291, 1854a9629800b9c8, 3ecf3c5b523df951, a997a1718d15facc, identical to s1).
  The probe records now hold sha256, n_tokens and per-token hashes only, with no token text.

**D1–D4 gates.**
- D1–D4 corrections are present unchanged in B1: the filesystem diff shows only the kpool file changed.
- The D1–D6 CPU regressions pass on the B1 serving image (section 7).
- In serving, the probes are the gate.

**Not expected:** long-prompt transcript parity with B0.

**Outcomes** (amendment 18a):
- CLEAN with C13 met: the D7 repair is verified for this serving configuration, and B1 is frozen as B0 + D7 only.
- VIOLATION: the failing inputs are preserved and minimised; no verified B1.
- INCOMPLETE: the missing condition is named and narrowly priced.
- INVALID: diagnosed and returned to the operator. One retry is allowed (18b), only after diagnosis, with no rule
  loosened and any changed file re-hashed.

## 5. Phase 2d recording spec (appended; outside the verdict)

**Procedure.** For D = 0..7:
1. Start D streaming completions of fixture prefix r7. It has 4,240 tokens, less than one 4,352-token block, so it never
   gets a prefix hit. Each has max_tokens 1024, temperature 1.0, top_p 0.95 and seed 9700000 + 10·D + j.
2. Wait until each background request has streamed ≥ 2 events (in decode).
3. Send the doc-127 GPQA chat prompt with max_tokens 8 and seed 9710000 + D, using the GPQA request parameters
   (`chat_template_kwargs`, top_p).
4. Close the background streams (server-side abort) and drain the server (`/metrics` running + waiting == 0).

**Doc-127 source.** The messages are read locally from the pass-1 lm-eval samples (`samples_…2026-10-08T20-01-05…jsonl`,
sha256 `be5c567d…`) exactly as the GPQA client sent them. They are verified by messages sha256 `b41ce17b…` before sending.
The exact rendered count is 2,795 (d7/exposure).

**Client records.** Requested D, background requests in decode at send, all seeds, times, server `prompt_tokens` (must be
2,795), finish and errors. There is no prompt or reply text in any output.

**Analysis** (`phase2d.json`), per case:
- doc 127 is identified by the ledger's new-request event with prompt length 2,795 in the case's time window;
- from the EngineCore record of its first scheduled step: the ACTUAL scheduled decode rows (D_actual), other prefill rows
  and their token counts;
- budget (`max_num_batched_tokens`) and draft slots K from server.log;
- chunk-1 start and length, and chunk-2 start (post-schedule `num_computed` convention, measured on run-a3);
- predictions `budget − Σ_others(n_i + K) − K` and `2045 − 7·D_actual`;
- split residue.

**Reporting rules.**
- Requested ≠ actual D is reported as such, not as an arithmetic contradiction.
- Shared steps (other prefill rows) are reported, and the 2045 − 7D formula is not applied to them.
- This validates the scheduling arithmetic under measured conditions. It does not measure historical GPQA exposure.
  Measured arithmetic, the reconstructed historical decode count and the inferred exposure are kept separate.

## 6. Budget (amendment 18b)

**Item and gate.**
- Item `b1-mapguard`: total ≤ **2.0 occupied GPU-h** over all its cost lines, invalid attempts included, with **at most 2
  attempts** (one retry).
- Runner gate: `APPROVED_GPU_H` is required, refused if > 2.0, and refused when 2 attempts are already recorded.

**Per-attempt cap.**
- **WALL_CAP 1,740 s** × 2 GPUs = **0.967 GPU-h**, which covers start-up, warm-up, the frozen phases, 2d and shutdown.
- The watchdog stops the server at cap − 75 s.
- The attempt is refused if the remaining budget leaves < MIN_WALL 1,260 s.
- The client deadline is cap − 150 s. Phase 2d starts only with ≥ 240 s left, and each case only with ≥ 40 s left, so 2d
  is skipped rather than the frozen phases.

**Expected spend [I, from attempt 3's measured 1,030 s].**

| Part | Seconds |
|---|---|
| Readiness | about 713–800 |
| Frozen phases | about 300 |
| 2d (28 × 4,240 background prefill tokens ≈ 58 prefill steps, plus 8 doc-127 cases and drains) | about 60–120 |
| Guard overhead | small |
| Shutdown | about 70 |
| **Total per attempt** | **about 1,150–1,300 s = 0.64–0.72 GPU-h** |

Worst case is two attempts at the cap (1.93 GPU-h).

## 7. CPU results before freezing [M]

`tests/run_cpu_tests.sh`, all in containers with `--network none`, `CUDA_VISIBLE_DEVICES=` and `TRITON_INTERPRET=1`
(`tests/out/summary.txt`).

| Suite | Image | Result |
|---|---|---|
| D7 fix tests (`d7/mechanism/tests/test_d7_fix.py`, read-only) | B1 serving | 28/28 met expectation |
| D1–D6 regressions (`tests/run_d1d4_b1image.sh` = mechanism part 1) | B1 serving | D1 col511 ALL PASS; D1 fixture cpucheck OK; D2 17/17 and 6/6; D3 and D4 PATCH CHECK PASS; D5/D6 still reproduced as expected |
| Image smoke (unset / device) | guard | I1 inert; I2–I4 pass |
| Frozen guard suite (unmodified copy, 28 tests) | guard | 28/28 |
| a2 regression | guard | ok (a2 module flags rtable; amended guard none) |
| kpool-unaligned (B0 path, no ring args) | guard | ok (documents B0's rule) |
| **B1 amendment (`tests/test_d7_guard.py`)** | guard | **30/30, 0 guard errors** |
| Frozen pipeline dry run + amended analysis | guard | runs; INVALID only for legacy hook calls (by design) and CPU-unsatisfiable items |
| B1 pipeline dry run (real B1 writer and seed, stand-in compress kernel) + analysis | guard | 0 violations; per rank 9 straddling events, all verified on every owned layer; positive control 34,974 / 29,145 pools, 0 mismatches; INVALID only for no FULL capture and args (CPU) |
| Analysis phase-2d / window-cut tests | host | 2/2 |
| Client against a mock OpenAI server (127.0.0.1:8001) | host | every phase incl. 2d (8 cases); 0 request errors; no text keys in the records |

**Runner [M].** `APPROVED_GPU_H=2.0 DRY_RUN=1 run/run_mapguard.sh 1` passed the budget gate (spent 0.0 + cap 0.967), the
attempt count, the frozen-hash check and the image pin. Gate checks on a scratch costs file: unset → REFUSED; 2.5 →
REFUSED; one attempt of 0.967 or 0.40 recorded → allowed (full 1,740 s cap); two attempts recorded → REFUSED; 1.40
spent → REFUSED (1,080 s < MIN_WALL 1,260 s).

**What `test_d7_guard.py` checks.** It drives B1's real hooked call site.

Negative controls (no flags):
- residues 1–3, first and later rows, successive chunks and the 4,352-token block boundary: 5/5 straddling pools
  `s_verified`;
- aligned pools: positive control 3,595 pools, 0 mismatches;
- spec decode rows ahead, padding tokens and an aligned-only schedule.

Oracle checks:
- straddling records equal the independent plain-torch reference of the request's own tokens;
- the oracle path equals that reference.

Faults, each detected:

| Fault | Flagged as |
|---|---|
| F1 missing repaired pool | `s_missing` |
| F2 foreign ring | `s_mism` |
| F3a stale keys, ring at position − 8 | `s_mism` |
| F3b stale keys, earlier position's key | `s_mism` |
| F4 writer positions +1 / −1 / +4 | `s_mism` |
| F5a/b/c/d no ring for the call, later-row cross pooling, the env switch, a row without a tail slot | `d7_fallback` + `missing_pred` / `pool_cross` + `s_missing` / `s_mism` |
| F6a/b B0's own writer behind B1's call site | `s_missing` / `s_mism` by bytes alone |

Positive control and shadow:
- An unfaithful oracle (ape × 1.01) gives `pc_mism` 894/899, so byte checks are INVALID and not violations.
- Shadow unavailable (unarmed previous chunk, slot collision) gives `s_unavail` and no violation.

Lints and analysis rules:
- G1: no host-sync op in the new hooks.
- G2: caches byte-identical with the hooks on and off.
- A6–A8: the event-count, positive-control and coverage rules.

## 8. Notes

- **Seed collision in the frozen workload [M]:** phases T1 and D both use seeds 8850000–8850007 on the same prefixes
  r0–r7 (T1 base 8840000 + 10000·1 = D base 8850000). The collision was also present in run-a3, so the "distinct seeds
  for all 64 sampled requests" in amendment 17e was not accurate. The workload is kept identical to attempt 3 for
  comparability; it affects sampled tokens only, not the mapping checks. Phase-2d seeds are distinct and disjoint from all
  other seeds.
- Drafter layers never reach the prefill hook (run-a3: only target MLA layers 3–43 counted at idx_w_pf) [M].
- The interpreter-based tests are slow: `test_d7_guard.py` 2,306 s, the frozen 28-test suite 1,538 s [M].

## 9. Files and hashes

`SHA256SUMS` (this directory) lists every file that runs (guard, hooked files, Dockerfiles and build scripts, analysis,
runner, client, images.txt, tests) and the test outputs. The runner refuses to start if any listed file differs.
Image ids are in section 1, `IMAGE-IDS.txt`, `run/images.txt` and `b1-build/image-id.txt`.
