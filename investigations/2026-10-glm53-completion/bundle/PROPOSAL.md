# Minimal verified correctness bundle for 0.9.1 K3.25: build, gate design, one-pass GPQA screen (PLAN.md amendment 14)

Date 2026-10-12. **Status: built and CPU-tested; nothing has run on a GPU and no server was started.** The gate and the screen
each need operator authorization. Values are measured unless marked **(inferred)**. File:line references are to the
installed 0.9.1 package (`kdcp/v091/vllm`, `kdcp/v091/b12x`) unless they name a bundle file. The plan was not changed.

## Summary

| item | result |
|---|---|
| Components | D1 column-511 (frozen gate diff), D2 tail ring variant U, D4 swiglu_limit 10, D3 fp32-output router (conditional). **D5 left out** (section 1.5). D6 is not separately included because it is unreachable under D2. |
| D3 condition | **Met.** Offline attempt 4 passed controls (a)-(d) on 1,593/1,593 records, and the corrected computation agrees with fp64 everywhere (`verify13/PROPOSAL.md` "offline attempt 4"). The proposed screen image is therefore **d1234**. d124 is the fallback; it is already built and tested, so no rebuild is needed. |
| Images (FROM release a9e985c0) | `local/glm53-v0.9.1-bundle-d1234:local` **sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548**; `…-bundle-d124:local` **sha256:0e970204890e4f353ff375b53f2307c8738e5fa15d8195029f950d990c9ebbfe** |
| Patches | All apply with zero fuzz and touch disjoint files, so no overlap needed resolving. The result does not depend on the order in which they are applied. Combined diffs: d1234 `949b2301…`, d124 `1a30a9f8…` |
| CPU tests | Each correction's check passes on both bundles, and release behaviour reproduces on the release image (D1-D6). Every output matches the earlier run's `expected/` apart from timings. Observer suites pass in both verification images; 9 new D4-override tests and an all-observers-together test pass. |
| Gate | Three sessions. **s1** (verification image, observers on) checks D1-D4 activation, short and mixed batches, request reuse, errors and a throughput lower bound. **s3** (limit-1 override) is the D4 real-weight reachability check. **s2** (production image) is conditional and gives clean throughput and a start-up smoke. Runner, client and analysis are written; not run. Required **1.25-1.69 GPU-h**; with s2 **1.95-2.44**; ceiling **3.0** (budget gate). |
| D4 check | Fixed greedy probe of 6 prompts × 2 rounds. Limit 1 (s3) is compared with limit 10 (s1 and s2), with in-session repeatability and cross-session controls. Engineering is done: a new verification-only module, 9 CPU checks. GPU cost **0.46-0.68 GPU-h** (s3). A Stage B harness, 9-12 engineering hours, is not needed. |
| Screen | One pass, 198 questions, **seed 1234**, original protocol. Expected **5.2-6.0 GPU-h** if production throughput is at least 95% of the release; **ceiling 7.5 GPU-h**. Invalid runs are handled separately, with no automatic re-run. Draft decision rules, with their power, are in section 7.4. |

---

## 1. Components and decisions

| id | source (sha256) | files | decision |
|---|---|---|---|
| D1 | `gate/freeze/candidate.diff` `1ca405f2…` | `sparse_attn_indexer_kpool.py` → **9104be37…** (= the frozen arm image 047ea60d file, `cmp` identical) | included |
| D2 | `tailfix/tailfix-U.diff` = `defect-repros/D2…/fix-U.diff` `9d5a7514…` | `indexer.py` → 00989c50…, `mamba_hybrid.py` → 049238f9… (both byte-identical to the GPU-run U-guard image 301baa73) | included |
| D3 | `defect-repros/D3…/proposed-UNTESTED.diff` `7af5968f…` (port of vllm-project/vllm#54048) | `gate_linear.py` → 2499ee67… | included in d1234 only; condition met |
| D4 | `defect-repros/D4…/proposed-UNTESTED.diff` `0091e194…` | `exl3.py` → b40344e2… | included (subject to the gate's real-weight check) |
| D5 | `defect-repros/D5…/proposed-UNTESTED.diff` `e7720d78…` (port of vllm-project/vllm#55341) | `gpu_worker.py`, `model_runner.py` | **left out** (section 1.5) |
| D6 | `defect-repros/D6…` | `warmup.py` | not separately included: unreachable under D2 (section 1.6) |

No other change is included: no limit tuning, routing heuristic, performance change or E1 batch-shape workaround.

### 1.1 D1 and D2 together

D1 and D2 do not share a file in the production bundle. D2 changes `v1/attention/backends/mla/indexer.py` and
`mamba_hybrid.py`; D1 changes `model_executor/layers/sparse_attn_indexer_kpool.py`.

D1's prefill gate reads `positions` from the kpool op's own argument. The model forward passes it at
`sparse_attn_indexer_kpool.py:1167-1191`. D2's `positions=` goes to `build_attn_metadata` instead (bundle
`mamba_hybrid.py:367`), so D2 does not change D1's branch conditions.

The shared file appears only in the verification image. There the tail-guard hooks and the col511 checker are both
added to D1's file; see section 4 and the merged-file tests.

The data-level interaction is that D2 fixes the tail-pool contents whose scores D1 orders. It is untested on a GPU. The
s1 gate measures both corrections in one server, and reports NaN-byte index rows as an interaction readout. The
2026-10-11 gate attributed those rows to the tail-ring family **(inferred)**.

### 1.2 D3 (conditional; condition met)

Attempt 4 verified the **expression** `torch.mm(x, W.T, out_dtype=fp32)` offline. The **patched dispatch** (tier 5 on
SM120, bundle `gate_linear.py:125-131, 161, 222`) has run only on CPU: `D3_installed_expect_patched` PASS. The s1 gate
verifies it in serving on two counts:
- dispatch: `tier_from_flags` must be tier 5 on all 42 layers × 2 ranks;
- output: the served logits must not be bf16-valued.

This readout was validated on the Stage A release capture: tier 6 on 84/84 layer-ranks, and 207,216/207,216 rows
bf16-valued (`cpu-tests/d3_logits_check_on_stageA_release_capture.json`).

**Label note.** verify13 stores the router output under the key `logits_shipped`. The gate re-labels it per record as
"served (tier from gate_info)" and never calls it fp32 without the dispatch evidence.

E1 (batch-size-dependent routing weights) is recorded separately and does not enter the bundle.

### 1.3 D4 and the runtime cache key

The rank-sliced runtime cache key omits the limit (bundle `exl3.py:2905-2929`). With D4, the planners read
`getattr(layer, "swiglu_limit", None)` only inside `_rank_sliced_runtime` (:2972) and `_mixed_rank_sliced_runtime` (:2754);
an AST check is in `test_d4limit.py`. Three facts make the patched path build every plan with limit 10:

- **One runtime, one limit.** Stage A measured a single runtime key per rank, owner `(0, False)`, serving layers 3-44,
  all with `layer.swiglu_limit` 10.0 (`verify13/offline-a2/d4_inventory.json`). The first plan therefore uses 10, and
  any layer that shares the runtime also has 10.
- **The runtime is per process.** `_RANK_SLICED_RUNTIMES` is created fresh at each start-up, so no plan from a release
  run can be reused.
- **The compiled kernels are keyed on the limit.** `W4A16FusedMoeKernel.__cache_key__` includes `has_swiglu_limit` and
  `swiglu_limit` (`b12x/moe/_shared/kernels/w4a16/kernel.py:6493-6508`). That key feeds both the in-memory cache and
  the on-disk compile spec (`mixed_trellis.py:2002-2024, 2109-2116`). A stale unclamped kernel from the stack's
  persistent cache (`/root/.cache`, mounted by `start.sh`) cannot be reused. The first bundle start therefore compiles
  new kernels; this is budgeted as cold-cache time, 0-300 s **(inferred)**.

The gate verifies the limit in two ways:
- **Plans:** caps, launch plan and core plan must each carry 10.0 on both ranks (verify13 `experts_info`).
- **Compiled kernels:** a new host-only walk of the plan object graph (`d4limit.py`, `GLM53_D4_INSPECT=1`)
  must find `has_swiglu_limit=True` and only the value 10.0. That walk is CPU-tested only on stand-in objects; it is
  unverified on real B12x plans until s1.

### 1.4 What the bundle does not change

The release's start-up order, workspace handling and warm-up are unchanged, because D5 and D6 are not in the bundle.

### 1.5 D5: left out (decision and justification)

Amendment 14 (2) includes D5 only if its backport is straightforward and verified against the V2 runner's actual
workspace lifecycle. Neither holds; four reasons:

1. **Not exercised (measured).** Stage A ran with `VLLM_DEBUG_WORKSPACE=1` and the workspace observer. The only resizes
   were during profiling, before any graph existed: rank 0 once, rank 1 twice, reaching 2,113 MB. There were none in
   capture, post-capture warm-up or serving, on contexts up to 120k tokens (`verify13/offline-a4/d5_summary.json`:
   "not exercised in this workload").
2. **The lifecycle reaches its maximum before capture (source; inferred for unseen shapes).**
   - The dominant user, the kpool K-gather, reserves `_gather_workspace_shapes(total_seq_lens=max_total_seq_len)`
     during profiling (`sparse_attn_indexer_kpool.py:294-304`). Every later call asks for the same full size (:508-516).
   - The B12X MLA borrows fixed specs (`b12x_mla_sparse.py:2155-2158`).
   - Growth after capture would need a paged-top-k scratch plan larger than about 2.1 GB (`sparse_attn_indexer.py:1105-1114`).
3. **Not straightforward.** The backport is not just a lock. It moves V2 `warmup_kernels` from after capture
   (`gpu_worker.py:796-797`) to before it (:715-719). That changes what has run, and what is cached, when
   `capture_model` empties the cache (`model_runner.py:998`). It also changes the order relative to the
   adaptive-verification profiling inside capture (:1018-1026) and DFlash2 capture. None of those interactions has run
   on a GPU.
4. **No benefit, real risk.** The lock turns any unforeseen growth into `AssertionError` (`workspace.py:156-162`), an
   execution error that would void a 5-6 GPU-h screen. In the measured lifecycle it would correct nothing.

Documented as: **"D5: confirmed omission (source); not exercised in Stage A; left out of the bundle."** The s1 gate
re-observes workspace resizes with short contexts and mixed batches (verify13 workspace observer), extending that
evidence to the bundle's workload. The bundle's CPU check confirms the release behaviour is unchanged
(`D5_excluded_expect_release`).

### 1.6 D6

D6 stays active only through D2. With D2 = U, the stale warm-up columns (16 rows per rank) are never read, because the
circular mapping reads column 0 only (bundle `indexer.py:567`). Measured in `tailfix/U-graph-a1`: `stale_col_rows` 16,
foreign writes 0, attention-block writes 0. The D6 patch stays CPU-checked only, and the bundle check reproduces the
release behaviour (`D6_excluded_expect_release`).

---

## 2. Build (CPU, done)

Workflow: extract the release files, apply the component diffs with `patch -F0`, and build `FROM` the release digest.

**Build gates** (`freeze/build-*/Dockerfile`, all passed):
- sha256 of each release file before it is replaced;
- byte-exact sha256 of each installed file;
- `py_compile` (bytecode regenerated);
- correction markers;
- no observer modules or hooks present;
- no `lock_workspace` in the V2 runner.

| file | release | d124 | d1234 |
|---|---|---|---|
| `model_executor/layers/sparse_attn_indexer_kpool.py` | e7935f2a… | 9104be37c8cb876a… | 9104be37c8cb876a… |
| `v1/attention/backends/mla/indexer.py` | 67449677… | 00989c503028216c… | 00989c503028216c… |
| `v1/worker/gpu/model_states/mamba_hybrid.py` | 82968af5… | 049238f99b3c77cb… | 049238f99b3c77cb… |
| `model_executor/layers/quantization/exl3.py` | 3b4ca47c… | b40344e2b4bd583a… | b40344e2b4bd583a… |
| `model_executor/layers/fused_moe/router/gate_linear.py` | 6fca8c7f… | unchanged | 2499ee679a3a8816… |

**Whole-image check.** The sha256 of every file under `dist-packages`, `/opt`, `/usr/local/bin` and `/etc` (117,696
files) was compared for release, d124 and d1234 (`freeze/manifests/`). The only differences are:
- the changed `.py` files above;
- their regenerated `.pyc` files;
- `/etc/hostname`, which is generated per container.

d124 and d1234 differ only in `gate_linear.py` and its `.pyc`.

**Diffs.**
- Combined: `freeze/combined-d1234.diff` (5 files, +66/−22) and `freeze/combined-d124.diff` (4 files). Each reproduces
  its tree when applied to the release files.
- Components: `freeze/patches/`. The order check applied D3, D4, D2, D1 and got a tree identical to D1, D2, D4, D3.

## 3. CPU tests (done; `cpu-tests/summary.txt`, runner `freeze/cpu_tests.sh`)

Everything ran in CPU-only containers (`--network none`, `CUDA_VISIBLE_DEVICES=`) against the **installed** files of
each image; nothing was mounted over them.

| check | release | d124 | d1234 |
|---|---|---|---|
| D1 `test_col511_cpu.py` (release slice vs installed helpers) + fixture cpucheck | n/a | ALL PASS / cpucheck OK | ALL PASS / cpucheck OK |
| D2 `test_fix.py` + scenarios (+ mechanism on release) | 17/17 release expectation; 6/6; MECHANISM_REPRODUCED | 17/17 (U); 6/6 | 17/17 (U); 6/6 |
| D3 `repro_router_logits.py` with layer 5's real router weights | REPRODUCED (tier 6) | REPRODUCED (D3 absent, as intended) | PATCH CHECK PASS (tier 5) |
| D4 `check_swiglu_plumbing.py` with the checkpoint config | REPRODUCED | PATCH CHECK PASS | PATCH CHECK PASS |
| D5 / D6 (excluded) | n/a | REPRODUCED (release behaviour) | REPRODUCED |

Every output equals the earlier run's `defect-repros/*/expected/` apart from timings and timestamps.

**Interaction checks.**
- The D1 and D2 suites ran in images where both changes are installed; their outputs are identical to the
  single-change runs.
- In both verification images (section 4), the suites pass on the merged kpool file:
  - col511 checker 7/7;
  - tail guard 22/22;
  - verify13 15/15;
  - `test_fix.py` U 17/17;
  - the new D4 override 9/9;
  - an all-observers-enabled-in-one-process test 5/5 (both `BlockTables` wrappers chained, verify13 and D4 wrappers
    installed, hooked forward runs).

## 4. Verification images (gate only, not the screen)

`…-bundle-d1234-check:local` **sha256:b61793817388bc42b2477df4bbde5aca61df0567cf7c88ab29061633b06282ba** and
`…-bundle-d124-check:local` **sha256:5dd7f8a3466a6fa21e579b7eb282e6cb20caa564ac816fea826cdba3c41443c1** are built FROM the
bundle image ids. Each adds the following, all additions-only and each inert unless its variable is set (build gate:
inert-by-default check):

- `col511check.py` 90ae3113… (gate/freeze; ran in the PASS gate, image 3e3d887b);
- `kpool_tailguard.py` 414f5b5c… (ran in tailfix U-graph-a1, image 301baa73);
- `verify13_capture.py` 96dcd5e3… (ran in Stage A, images ff6c8542 / 648c2764);
- **new** `d4limit.py` 49597e45… (`GLM53_D4_INSPECT=1` plan walk; `GLM53_D4_LIMIT_OVERRIDE=1|none` sets the
  limit the EXL3 planners see during their own call, then restores the layer attribute). CPU-tested only.
- `sparse_attn_indexer_kpool.py` fae1127c… = the D1 file + the frozen checker additions + the tail-guard hooks
  (+88/−0 overall). The tail-guard hook lines are identical to the U-guard image's.
- `gate_linear.py`: the bundle file + the verify13 guarded block + the d4limit guarded block (+16/−0). The result is
  02ba8104… for d1234 and 48f4c035… for d124.

Builder: `freeze/make_check_files.py`; Dockerfiles and observer diffs are in `freeze/check-*`. Every test's exit code
gates the build. (A first draft let earlier test failures pass; it was found and fixed before any image was recorded.)

## 5. Gate design (not run)

### 5.1 Sessions

All sessions use the stack `glm53-flash-v0.9.0` with recipe `…k3.25@tpurtell+v0.9.1` env, `IMAGE` overridden, shipped
graph mode (no `EXTRA_SERVE_ARGS`) and DFlash2 ×3.

| session | image | observers / env | client phases | purpose |
|---|---|---|---|---|
| **s1** | d1234-check | col511 checker (device), tail guard (device), verify13 (`EVERY=4, STRIDE=21, OUT=0`, 1 GiB cap), D4 inspect, `VLLM_DEBUG_WORKSPACE=1` | warm, T/F ×4, D/S ×2, Q, Qreuse, G, P | D1-D4 activation, short contexts / mixed batches / reuse, errors, throughput lower bound, D5 observation |
| **s3** | d1234-check | `GLM53_D4_LIMIT_OVERRIDE=1` (+ inspect) only | P | D4 real-weight reachability |
| **s2** (conditional) | d1234 production | none | warm, T/F ×8, G, P | clean throughput for pricing, production start-up smoke, cross-session probe control, KV pool unchanged |

Run s2 only if s1's T/F throughput falls below 95% of the release reference, or if the operator wants the screen image
started first. Observers only add work, so s1 is a lower bound. **(inferred)** s2 is likely to be needed, because the
col511 checker alone cost about 4% in the 2026-10-11 gate (`gate/RESULT.md:238-239`).

**Client phases** (`gate/client_bundle.py`; every request's seed is recorded):
- **warm, T/F, D/S:** byte-identical workload and seeds to `tailguard/graph-a1`, `tailfix/*-graph-a1` and the col511
  gate (long contexts of 4,240-120,217 tokens).
- **Q:** 16 GPQA prompts (124-600 tokens), chat template with thinking, temperature 1.0, top_p 0.95, 2,048 tokens,
  continuous admission at 8 in flight. This gives contexts below the 2,176-token index width and mixed prefill/decode
  batches.
- **Qreuse:** the same 16 prompts with new seeds, so the prompts hit the prefix cache.
- **G:** 8 GPQA prompts at once, 6,144 tokens: GPQA-shaped throughput.
- **P (D4 probe):** 4 GPQA prompts plus 2 long prompts (r7, r5), each sent **alone**, greedy, 96 tokens, logprobs, two
  rounds. Repeating a prompt here is a determinism control, not repeated sampling.

**Fixture.** The GPQA prompts are the exact messages lm-eval sent in the 0.9.1 record's pass 1. 24 of 24 match a
seed-1234 request-log hash. The fixture is `gate/gpqa_fixture.LOCAL.json`, kept local and never published.

### 5.2 Criteria (fixed before data; `gate/analyze_bundle_gate.py`, `gate/gate_verdict.py`)

**Validity and errors (every session).**
- V1-V4: every planned phase completed; 0 request errors; fixed-length requests finish by length; no stopping error
  (finish outside {stop, length}, or a stop with zero tokens).
- V5: the server log has no Traceback, illegal memory access, CUDA error, out-of-memory, "Workspace is locked",
  AssertionError, preemption, NaN warning or engine death.
- V6: one server start.
- V7: effective configuration equals the historical record: API arguments, engine config, graph captures 11 / 9 / 9
  (`config_compare.py`).
- V9: the served engine files have the frozen sha256s.
- s2 only, V8: KV pool equals the historical 4,707,515 tokens (±0.5%); V10: production throughput at least 85% of the
  release reference, otherwise stop and report before the screen.

**s1 activation.**

| correction | criterion (both ranks) |
|---|---|
| D1 | On every serving row (eager, piecewise, FULL replays): column 511 holds the lowest selected pool, the set is unchanged and consumers read it (`post_not_lowest`, `post_set_changed`, `consumer_mismatch`, `post_unfilled_valid_in_col511`, out-of-range, duplicates all 0); fp64 re-score agrees on every clean sampled row. Coverage: decode fused, decode tiled and prefill, with FULL replays observed and FULL reconciliation exact; 0 checker errors or eager fallbacks. Capture-phase dummy rows are reported, not judged, as in the 2026-10-11 gate. |
| D2 | Tail slot source is `compute_kpool_tail_slot_mapping`; writes > 0; null-block, foreign, attention-block and tail collisions, shadow divergence and guard errors all 0. |
| D3 | `tier_from_flags` = "5 torch.mm(out_dtype=fp32)" on 42 layers × 2 ranks; at least 1,000 sampled rows, of which ≤ 0.1% are bf16-valued. For d124 the control is the reverse: tier 6 and 100% bf16-valued. |
| D4 | Layer limit 10.0; trellis and prefill plans carry 10.0 in caps, launch plan and core plan; the plan walk finds `has_swiglu_limit=True` and only 10.0. |
| O1 | verify13 is healthy: installed on both ranks, records only from eager forwards, 0 hook errors. |

Reported but not judged:
- NaN-byte index rows: expected about 0 with D2 **(inferred)**; a non-zero count is investigated before the screen.
- D5 workspace resizes after capture.
- KV pool size under the observers.

**D4 reachability** (`gate_verdict.py`). P1: within-session repeats are identical on all 6 prompts.
- If s2 ran and its probe equals s1's (cross-session determinism, and the observers are numerically inert): reachable
  when s3 (limit 1) differs from s1 on at least 1 of 6 prompts.
- Without that control: reachable when at least 5 of 6 differ and at least 4 diverge within the first 16 tokens.
- If s3 is identical to s1, D4 is **not verified**. The bundle is then not screened with D4; a build without it needs a
  new freeze and an operator decision.

Overall **PASS** = every s1 check, every s3 check, every s2 check if s2 ran, and D4 reachable. If D3 fails in s1, the
fallback is to run s1 on d124 (about 0.8-1.0 GPU-h, needs re-authorization).

### 5.3 Runner (`gate/run_bundle_gate.sh s1|s2|s3 [d1234|d124] [ATTEMPT]`)

Modelled on the `$S/gate` and `$S/tailguard` runners and `verify13/run_verify13.sh`:
- **Budget gate:** logged `bundle/costs.jsonl` GPU-h plus this step's cap must be ≤ `BUDGET` (3.0).
- **Image-id check:** against `gate/images.txt`.
- **Pre-flight:** no glm53* container; no non-desktop compute process; GPU0 ≤ 1 GiB; GPU1 ≤ 4 GiB (desktop accepted);
  utilization ≤ 10%.
- **Readiness gate:** release warm-up marker and API, with an engine-error abort.
- **Observer-install gate** (s1: all four observers on both ranks; s3: override applied on both ranks); otherwise the
  run is invalid and stops at once.
- An EXIT/INT/TERM trap stops the server on every exit path.
- **Cost line** counted to GPU release; observer output is copied out after the GPUs are free; then the analysis runs.

Invalid-run rule: an observer missing, a readiness timeout, or a host or infrastructure failure → invalid. One repeat
per session is allowed while inside the budget. An engine error caused by the bundle (V5) is a gate **FAIL**, not
invalid.

Dry runs on archived data (`kdcp/bundle/dryrun/`):
- The analysis passes D1 on the frozen gate's checker data.
- It passes D2 on `tailfix/U-graph-a1` and fails D2 on the unfixed `tailguard/graph-a1` (null alias 301,272).
- It fails D3 and D4 on the release Stage A capture (tier 6; plans `None`), as it should.
- `gate_verdict.py` passes and fails synthetic probes as specified.
- The client ran end to end against a mock server: 140 requests, all phases, distinct recorded seeds.

### 5.4 Cost

Basis: measured start-up 726-818 s; phase durations from `gate/server-a1`, `tailfix/U-graph-a1` and `verify13/server-a1`
(warm 64-69 s, T 17.5-18.9 s, F 8.3-9.0 s, D and S 18-20 s per repeat). Q, G and P are estimated at about 57 tok/s per
request at 8 in flight (`arm-col511/result-col511-final.md`) **(inferred)**. Cold-cache compile of the new limit-10 and
limit-1 kernels is 0-300 s **(inferred)**.

| session | expected wall | expected GPU-h (2 GPUs) | wall cap → GPU-h cap |
|---|---|---|---|
| s1 | 1,426-1,818 s | 0.79-1.01 | 2,400 s → 1.33 |
| s3 | 826-1,218 s | 0.46-0.68 | 1,800 s → 1.00 |
| s2 (conditional, likely) | 1,256-1,348 s | 0.70-0.75 | 2,000 s → 1.11 |
| **required s1 + s3** | | **1.25-1.69** | |
| **with s2** | | **1.95-2.44** | **ceiling 3.0** (budget gate; invalid attempts count) |

Engineering for the gate is done in this task: runner, client, analysis, verdict, fixture, D4 module and tests. What
remains is about 1-2 h to write up the gate result.

## 6. D4 real-weight execution check (design and cost)

**Question.** Does the checkpoint's limit reach the K3.25 projection-mixed computation in shipped graph serving? Limit 1
establishes reachability; 10 is the candidate (amendment 14 (2)).

**Design.** The cheapest real-weight check uses existing runtime fixtures and needs no new harness:

1. **s1, limit 10 (free; rides on the gate session).** Every plan and compiled kernel carries 10.0, measured on the live
   runtime objects. The probe P is recorded twice.
2. **s3, limit 1.** The check image with `GLM53_D4_LIMIT_OVERRIDE=1`. It must log that the override replaced 10.0 on
   both ranks and that the plans carry 1.0. Same probe.
3. **Rule** (section 5.2). A difference at limit 1 shows that the plan's limit reaches the real-weight serving
   computation. The kernel-level semantics were already shown by B12x's own tests on this GPU (Stage A smoke: 3
   passed).

**Controls.** In-session repeat; cross-session equality of s2 with s1 when s2 runs. The override cannot leak into the
screen image, because the module is absent from the production image (manifest check).

**Cost.** GPU **0.46-0.68 GPU-h** (s3; cap 1.0). Engineering about 2 h, already done: `d4limit.py`, 9 CPU
checks, probe phase, verdict rule. The Stage B offline harness (9-12 engineering hours + 0.5-0.8 GPU-h,
`verify13/PROPOSAL.md`) is not needed.

**Not answered.** Whether limit 10 changes outputs, i.e. whether the clamp engages. An optional s4
(`GLM53_D4_LIMIT_OVERRIDE=none` against limit 10) would answer that for the 6 probe prompts at 0.46-0.68 GPU-h. It is not
proposed, because the limit is the checkpoint's intended setting and its effect does not decide the screen.

## 7. One-pass GPQA screen (not run; needs separate authorization)

### 7.1 Protocol and recipe

**Protocol** (gpqa-diamond/v1, unchanged):
- all 198 questions, lm-eval `gpqa_diamond_thinking`, chat template with thinking;
- temperature 1.0, top_p 0.95, concurrency 8, cap 327,680, normal stopping;
- the recording proxy, a fresh server and the release warm-up marker gate (`eval-chain.sh`);
- traces kept locally in `<gpqa-runs>` (samples, request log, server log).

**Recipe fork** (described here, **not created**): `glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234`.
- Created with `recipe-tool fork` from `glm53-flash-exl3-k3.25@tpurtell+v0.9.1`. The only change is `env.sh`
  `IMAGE=sha256:08d1cb90…`, following the `+col511fix` pattern; status draft-untested until captured.
- The fallback, if D3 fails the gate, is `…+bundle-d124` with `IMAGE=sha256:0e970204…`.

**Runner draft:** `screen/gpqa-bundle-screen.sh`. It aborts unless the recipe exists with that image id, the image is
present and no pass directory exists. It runs one session (`GPQA_PASSES=1 GPQA_SEED_PER_PASS=1`), keeps the GPU log,
stops at the ceiling (watchdog), writes `screen/costs.jsonl` and runs the frozen analysis.

### 7.2 Seed: 1234

- **Same seed as all three pass-1 baselines.** Seed 1234 is pass 1 of the 0.9.1 record, of the column-511 arm and of
  0.7.0 (`local-fixture-index.json`). Each of those passes ran first on a freshly started server, so the server
  lifecycle matches too.
- **Cause can be classified against both comparators.** Request logs exist for the record (`…+v0.9.1.requests.jsonl`)
  and the arm (`…+col511fix.s1.requests.jsonl`), which classify each empty reply as cap exhaustion or premature stop.
  0.7.0's pass 1 has no request log, but it has 0 empties, so it serves as context only.
- **A budget basis exists.** Reconstructed pass-1 timings exist for both 0.9.1 baselines.
- **Prompts are identical.** The analysis checks all 198 message hashes; the fixture subset already matches 24/24.

Seeds 1235 and 1236 match passes 2 and 3, but pass 3 ran on a warm server in both 0.9.1 records.

**Limit.** A matched seed matches only the sampling stream up to the first computational divergence. D3 alone flips
about 3% of expert sets, so the outputs are not paired copies. The comparison is a historical control with
exchangeability assumed, not a randomized one (amendment 11a). The record's and the arm's pass 1 at the same seed show
the churn: 5 vs 4 empties, 3 rescued and 2 new.

### 7.3 Budget

The reconstructed pass-1 sessions (`rep070/timing-reconstruction.md`) are:
- 0.9.1 record: **5.15 GPU-h** (2.574 h wall; request window 2.37 h);
- col511 arm: **5.74 GPU-h** (2.870 h; window 2.67 h).
Across the six 0.9.1-family passes, the request windows run 2.37-2.79 h.

**Expected.** Session = 2 × (about 0.2 h start-up and warm-up + window × f), where f = 1/r and r is production
throughput relative to the release reference from the gate (s2, or s1's bound when it is at least 0.95). With
r ≥ 0.95: **5.2-6.0 GPU-h**. The worst observed window scaled by 1/0.95 gives 6.3. Identical runtime is not assumed:
fewer cap exhaustions shorten the pass, and longer answered completions lengthen it.

**Ceiling: 7.5 GPU-h** (3.75 h wall), enforced by the watchdog and not raised after data. If the gate's V10 finds
r < 0.85, the screen is not started.

**Invalid runs** (handled separately, never automatic):
- An infrastructure failure (host, another GPU process, proxy, external kill) makes the pass invalid. One repeat at seed
  1234 on a fresh server needs explicit operator authorization, up to 7.5 GPU-h.
- An execution, memory or stopping error caused by the bundle is **not** re-run. It is a screen outcome: an execution
  regression.
- Reaching the ceiling leaves the pass incomplete and invalid. Partial counts are reported only descriptively, with no
  extension.

### 7.4 Draft screening rules (for operator/assessor sign-off; frozen before generation)

These are encoded in `screen/screen_analysis.py`. They reuse the frozen arm loaders `analysis_col511.py` (b5b13ce2…)
with N_PASSES = 1. A dry run, with the arm's pass 1 standing in, reproduces its known figures: 4 empties, raw 164,
stated 165, window 2.67 h at 405 tok/s, and a server mean of 444.5 tok/s at 8 running.

- **Validity.** One samples file with 198 questions; 198 logged requests, none re-sent; all HTTP 200; seed 1234;
  cap 327,680; temperature 1.0; top_p 0.95; no parse errors; message hashes equal to both baselines'; one server start;
  no illegal-memory, preemption or NaN lines in the window; every finish is stop or length; completed under the ceiling.
  An invalid pass gets no reading.
- **Primary.** E = empty final replies of 198, split into cap exhaustion (length at 327,680) and premature stop (stop
  with empty content). It is reported as rescued, newly empty and empty in both against:
  - the 0.9.1 record's pass 1: 5 empties = 4 cap + 1 premature (q12); q88, q109, q120, q127;
  - the column-511 arm's pass 1: 4 cap (q79, q88, q127, q170);
  - 0.7.0's pass 1 for context: 0 empties.
- **Flags.**
  - OBVIOUS REGRESSION: raw (flexible-extract) correct ≤ 158/198, stated (adjudicated) ≤ 159/198, or 2 or more
    premature stops.
  - ACCURACY CONCERN: raw 159-163, which is below each of the six 0.9.1-family passes (164-176; mean 170.2, SD 3.9).
    **(inferred, normal approximation over 6 passes)**: P(raw ≤ 163 | no change) ≈ 4%, P(raw ≤ 158) ≈ 0.1%.
- **Outcome.**

  | outcome | condition |
  |---|---|
  | **CLEAR PROMISING** | E ≤ 1 and no flag |
  | **MIXED** | E ≤ 3 and any flag |
  | **INTERMEDIATE** | E = 2-3 and no flag; not clearly promising, allocated as similar or worse |
  | **SIMILAR OR WORSE** | E ≥ 4 |

**Power of one pass** (`screen/power_bundle_onepass.py`; the empirical-Bayes beta-binomial of `power_col511.py`, fitted
to the 0.9.1 record, question-clustered; 50,000 simulations, seed 20261012):

| true behaviour of the bundle | mean E | P(E ≤ 1: clear promising) | P(E 2-3) | P(E ≥ 4: similar or worse) | P(McNemar vs record p1 < 0.05) |
|---|---|---|---|---|---|
| no change (0.9.1 rates) | 5.27 | **0.020** | 0.176 | 0.805 | 0.003 |
| 1.5× worse | 7.82 | 0.001 | 0.022 | 0.977 | 0.000 |
| 50% fewer empties | 2.63 | 0.245 | 0.486 | 0.268 | 0.061 |
| 75% fewer | 1.31 | 0.619 | 0.342 | 0.039 | 0.260 |
| 0.7.0-like per-question rates (two fits) | 1.33 | 0.61-0.62 | 0.34-0.35 | 0.035-0.047 | 0.24-0.26 |
| no empties | 0 | 1.000 | 0 | 0 | 1.000 |

One pass of a rare outcome has limited power. The paired McNemar test against the record's pass 1 cannot go below
p = 1/32, because at most 5 replies can be rescued, so it is reported only as a reference.

An alternative threshold for the assessor: clear promising at E ≤ 2. That raises P(clear promising) to 0.078 under no
change, 0.85 for 0.7.0-like rates and 0.50 for a 50% reduction.

**Secondary.**
- Raw and adjudicated correctness on all 198 requests (not answered-only), with differences and discordant counts
  against both baselines. Stated-vs-raw disagreements are hand-read before the stated figure is final.
- Answered completion lengths: quantiles and counts above 32k, 64k, 128k and 200k.
- Occupied GPU-h, validity, aggregate throughput and server-log tok/s at 8 or more running.

### 7.5 Follow-ups per outcome (amendment 14 (5))

- **CLEAR PROMISING → confirm before attribution.** Choose on measured cost between:
  - (a) bundle passes 2-3 at seeds 1235/1236 on a fresh server, mirroring the baselines' session 2 (11.37-11.48 GPU-h
    × f), enabling the frozen 594-question-pass analysis against both records;
  - (b) a contemporaneous stock-0.9.1 control at seed 1234 plus a bundle pass at a new prespecified seed (about 11-12
    GPU-h).
  Then ablate by removing one component at a time (−D1, −D2, −D3, −D4; at least 4 passes, about 22 GPU-h × f). D1's
  value depends on D2-corrected contexts, so −D2 also changes D1's inputs; interactions are reported, not assumed away.
- **SIMILAR OR WORSE or INTERMEDIATE.**
  - Deprioritise the bundle as the next completion remedy, and start other source and frozen-input investigations
    (for example E1 on the Stage A inputs).
  - No serial small completion pilots.
  - No conclusion that the included corrections lack effects, that interactions are absent, or that a new kernel or MoE
    defect exists.
  - The bundle stays as a corrected-engine candidate. The stock 0.7.0 replication stays deferred (amendment 14 (6)).
- **MIXED.** Report the trade-off and its uncertainty, not success. Any confirmation makes accuracy co-primary and is
  priced separately.
- **Invalid.** Section 7.3.

## 8. Not verified / limits

- Nothing has run on a GPU. Not yet seen in serving:
  - the D3 patched dispatch;
  - D4's plans and kernels;
  - the D4 override module;
  - the plan walk on real B12x objects;
  - the three observers together in one server (CPU-tested together only).
- The amount of cold-cache compile time at the first bundle start is unknown **(inferred 0-300 s)**.
- D1 and D2 at DCP > 1 remain out of scope; neither is the shipped configuration. D1's prefill part is DCP1-only by
  design.
- FULL-graph decode coverage for D3 is by source plus the dispatch flags. verify13 samples eager forwards only, and in
  serving the tier is fixed per layer at construction, so it does not vary between graph and eager **(inferred)**.
- The gate's GPQA fixture is 24 questions with capped lengths. It exercises execution paths; it does not estimate
  completion.
- Fixtures and scores are not comparable across model families, and nothing here compares models.

## 9. Files

`<runs>/bundle/`:

- `PROPOSAL.md`
- `freeze/`:
  - `combined-d124.diff`, `combined-d1234.diff`, `patches/`;
  - `build-d124/`, `build-d1234/` (Dockerfile, build.log, image-id, files);
  - `check-d124/`, `check-d1234/` (Dockerfile, build.log, merged files, observer diffs, `d4limit.py`,
    `test_d4limit.py`);
  - `make_check_files.py`, `cpu_tests.sh`, `test_all_observers_together.py`;
  - `manifests/` (release file sha256s, whole-image manifest diffs).
- `cpu-tests/`: `summary.txt`, per-image outputs, `d3_logits_check_on_stageA_release_capture.json`.
- `gate/`:
  - `run_bundle_gate.sh`, `client_bundle.py`, `analyze_bundle_gate.py`, `gate_verdict.py`, `d3_logits_check.py`;
  - `make_gpqa_fixture.py`, `gpqa_fixture.LOCAL.json` (local only), `images.txt`;
  - `tools/` (unchanged copies of `analyze_gate.py`, `summarize_tg.py`, `throughput_tg.py`, `config_compare.py`,
    `analyze_d5.py`).
- `screen/`:
  - `gpqa-bundle-screen.sh` (draft);
  - `screen_analysis.py` (draft rules) and `dryrun-arm-pass1.{json,txt}`;
  - `power_bundle_onepass.py`, `power-onepass.{json,txt}`;
  - `analysis_col511.py` and `power_col511.py` (unchanged copies).

Hashes: runner `e768dee0…`, client `e62ad0bc…`, gate analysis `cfd133dd…`, verdict `1ce791ed…`, screen analysis
`0af25821…`, power `a8dabdb2…`. Scratch working copies are in `kdcp/bundle/`.

---

## Addendum 2026-10-12: gate review after s1 / s3 and the release control (for operator review)

Gate runs (coordinator, amendments 14a-14b): s1 d1234 a1 0.7711 GPU-h, s3 d1234 a1 0.4506. Release control
`rel release a1` (this review; runner mode `rel` added: stock release image, no observers or override, probe phase
only): 0.4111 GPU-h (ready 709 s). Gate total **1.6328 of 2.0** occupied GPU-h. Data: `gate-runs/`.

**(1) "D4 compiled kernels" failed because the check was mis-specified, not because the limit is missing.**
- The projection-mixed runtime does not retain the kernel objects that carry `has_swiglu_limit`. The limit lives on
  `W4A16FusedMoeKernel` (`kernel.py:6281-6282`) as a `const_expr` used in `_clamp_swiglu_inputs` (:6546-6559).
  `compile_mixed_trellis` builds that kernel, compiles it, and returns a `MixedTrellisCompileResult` that copies only
  scalar fields (`mixed_trellis.py:61-86`, constructed at :2117-2153). The kernel instance is dropped.
- The walk could therefore only reach plan-level objects. It found exactly 6 hits per rank (`TPMoEPlan`,
  `TPMoEScratchCaps` and `_TPCoreWorkspacePlan` × 2 plans, all 10.0). Its CPU test used a stand-in kernel object placed
  in the plan; that placement was my unverified assumption.
- The intended kernel-level fact is measured in the b12x on-disk compile cache, whose spec records the kernel cache key
  (`mixed_trellis.py:2002-2010, 2109-2116`; `gate-runs/d4-compile-cache-evidence.{txt,json}`):

  | compiled during | per GPU | (has_swiglu_limit, swiglu_limit), driver / tier0 / tier1 |
  |---|---|---|
  | s1 (08 h UTC) | 33 kernels, size_m 1-1024 | (True, 10.0) on both GPUs |
  | s3 (09 h) | 33 | (True, 1.0) |
  | earlier release runs | 33 | (False, 0.0) |

  Together with the plan check (pass) and the s3 limit-1 output change, the limit is compiled into the served kernels.
- Proposed replacement criterion: every mixed-trellis kernel compiled during the session, on both GPUs, has
  (True, 10.0) in all three kernel keys. **Met.**

**(2) P1 failed in the release too, so the criterion was mis-specified for long contexts.**
- The release control repeats the pattern of the bundle sessions:
  - q0-q3 (124-158-token prompts) are bitwise identical between rounds;
  - r7 (4,240 tokens) and r5 (16,240) diverge, with nonzero logprob differences on every token before divergence;
  - r7 diverges even with identical cache coverage (0/4240 in both rounds, in every session).
- Correction to the earlier description: r5's coverage differed between rounds in s3 and in the release (0 vs 8,704
  cached). It was identical only in s1 (8,704 both). r7's coverage was identical in every session.
- Long-context greedy non-repeatability is therefore a **pre-existing engine property** of 0.9.1, not introduced by the
  bundle. **(inferred)** Candidates are nondeterministic sparse-index selection or ordering, or reduction order, beyond
  the short-context regime; not identified.

  | session | r7 first divergence | r5 first divergence | max \|Δlogprob\| before divergence | chosen-token p at divergence (round 1 / round 2) |
  |---|---|---|---|---|
  | release | 3 | 3 | 0.053 / 0.126 | r7 0.49 / 0.66; r5 0.41 / 0.31 |
  | s1 (bundle, limit 10) | 75 | 3 | 0.160 / 0.147 | r7 0.56 / 0.58; r5 0.30 / 0.35 |
  | s3 (bundle, limit 1) | 23 | 3 | 0.256 / 0.136 | r7 0.32 / 0.32; r5 0.33 / 0.33 |

  Divergence happens at low-margin steps: two tokens each carry about 0.3-0.6 probability, for example the same
  two tokens at r5's divergence in all three sessions (token strings withheld: the long prompts are
  model reasoning on GPQA questions). Only the chosen token's logprob was recorded, so the margin is bounded, not measured.
  `gate-runs/probe-divergence.json` holds the table.
- Proposed corrected P1: within-session repeatability on the short prompts (q0-q3), where the release is repeatable.
  **Met** in s1 and s3. Long-context non-repeatability is reported as a finding.

**D4 reachability** under the frozen no-control rule: 6 of 6 probes differ between limit 1 and limit 10, all within the
first 16 tokens. The informative ones are q0-q3. They are bitwise repeatable within each session, and diverge at tokens
0-2 with token-0 logprob shifts of 0.03-0.71. The r7/r5 differences carry no weight given (2). **Reachable.**

**Other s1 results.** All checks passed:
- validity and errors;
- D1, with 0 violations and fp64 agreement;
- D2, with 0 null, foreign, attention-block and tail writes;
- D3, with tier 5 on 42 layers × 2 ranks and 0 of 69,344 sampled rows bf16-valued;
- D4 plans at 10.0.

Further readouts:
- NaN-byte index rows were 0 in every branch, against thousands in the 2026-10-11 gate. This supports D2 removing them
  **(inferred link)**.
- D5 was again "not exercised".
- Throughput with all observers on: T/F 82.2 / 55.1 tok/s (0.93 of the release-with-guard reference). This is a lower
  bound and cannot separate observer cost from bundle cost.
- The G phase gives no GPQA-shaped number: 3 of 8 requests stopped within 18 s, so the phase mostly ran 5 or fewer in
  flight (244.7 tok/s aggregate). The screen's own measurement and its 7.5 GPU-h watchdog (amendment 14b) cover pricing.

**Proposed verdict (operator review):** PASS with two criterion corrections, each justified above from source or a
release control and decided after the data, which the write-up must say. If the corrections are not accepted, the gate
stays FAILED on a criterion the stock release also fails.

## Addendum 2026-10-12 (2): amendment 14c pre-launch items, from existing records (screen not launched)

1. **Early stops in the gate's G phase: all valid stops with a non-empty final reply.**
   - The four `stop` finishes (q19 726 tokens, q21 902, q16 852, q22 4,096) all have non-empty content (785-2,305
     characters), reasoning present, no request error and HTTP OK.
   - The same questions in the 0.9.1 record and the column-511 arm stop at similar lengths, with every reply correct:
     docs 16/19/21 at 429-2,863 tokens, doc 22 at 4,019-17,375.
   - The gate client stored lengths only, not text, so the GPQA answer parser could not be applied. "Answered" here
     means non-empty final content.
   - Consequence: G-phase throughput is unusable; readiness is intact.
2. **D3 as served: PASS** (`gate-runs/d3-served-a2/`; `gate/d3/d3_served_check.py`, stage-A helpers unchanged).
   - Scope: 1,000 records over both ranks, 69,344 rows, 19,971,072 logits.
   - A0: tier 5 and weight/bias sha256 on 84/84 layer-ranks.
   - A': `torch.mm(out_dtype=fp32)` at the serving M reproduces the served logits bitwise on 1,000/1,000 records; the
     patched GateLinear does too.
   - D: 0 τ_d and 0 τ_worst violations; maximum error/S 6.8e-7; 0 top-8 disagreements.
   - C: routing at the serving M reproduces the consumed routing bitwise, 69,344/69,344 rows.
   - Cost: 0.0075 GPU-h, plus 0.0 for attempt 1, whose container failed to start because of a mount path.
3. **Production image (08d1cb90…).**
   - All five engine files match the frozen candidate files. There are 0 observer modules, and 0 files reference
     observer code.
   - The verification image differs from it only by additions: kpool +88/−0, gate_linear +16/−0, other engine files
     identical, plus `/opt` test files and the observer modules. The hooks are inert with their variables unset
     (`gate-runs/production-image-check.txt`).
4. **D4 cache provenance.** `gate-runs/d4-cache-provenance.txt` shows:
   - 66 (True, 10.0) mixed-trellis kernels (33 per GPU, size_m 1-32 and 1024 = s1's trellis and prefill plans), written
     at 08:43:01-08:44:34 UTC, inside s1's window. s1 logged "rank-sliced runtime planned" at 08:44:36.
   - 0 of those keys, and no (True, 10.0) key at all, existed before s1.
   - With the limit fields neutralised, all 66 equal pre-gate release keys on the same GPU.
   - The 66 s3 keys are the same shapes with 1.0, written 09:06-09:07 UTC.
5. **`screen/screen_analysis.py` follows the amendment 14c categories** (`categorize()`; 13/13 boundary cases).
   - Without a complete stated-answer adjudication file the category is PROVISIONAL.
   - Dry run on the arm's pass 1: SIMILAR OR WORSE (E = 4), and PROVISIONAL without its three hand reads.
   - The power output is relabelled as P(E ≤ 2), P(E = 3) and P(E ≥ 4) only; accuracy and stopping are not modelled.
     P(E ≤ 2) is 0.078 with no change, 0.85-0.86 for 0.7.0-like rates and 0.50 for a 50% reduction.
6. **`screen/gpqa-bundle-screen.sh` is executable and ready, not run.** It aborts unless:
   - the recipe IMAGE is the bundle id;
   - the image is present;
   - the GPUs are idle;
   - `screen_analysis.py` is the frozen version;
   - no pass directory exists.
   The 7.5 GPU-h watchdog is in place and there is no retry.
