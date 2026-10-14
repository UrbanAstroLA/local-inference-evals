# Hook validation: plan, references, tolerances, frozen criteria, runner (PLAN.md amendment 21, section 4)

2026-10-13. **Prepared on CPU; nothing has run on a GPU.** The GPU session runs only after the operator authorizes
it. The design of the hook is in PLAN-NOTES.md section 2.

Evidence labels: **[M]** measured by running code here; **[S]** read from source; **[I]** inferred.

## 1. What must be shown before any probability is interpreted

The amendment-21 section-4 list, and where each item is covered:

| Requirement | Covered by |
|---|---|
| Short exact-id histories | 24 step-5/6 traces (< 5k) plus 8 histories B1 samples itself in-session |
| Zero accepted drafts | `poison` schedule (a = 0 every step) |
| Fully matching drafts | `oracle` schedule (drafts = next history tokens; a = K) |
| Rejection at each draft position | `reject:1`, `reject:2`, `reject:3` (a = 0, 1, 2) and `sched` (mixed) |
| Bonus-token accounting | Every step commits a + 1 tokens, the last clamped: per-step commit counts, accepted-count histograms, `num_computed` / `total_len` exact |
| Block boundary (4,352) | Two windows prefilled to P = 3,832 and forced across position 4,352 (traces 53fa5b3e066f and 57703691f509); snapshots at processed frontiers 4,350-4,354 |
| Ring boundary (8) | Every verify step crosses `pos % 8` wraps; snapshots at all 8 residues; windows starting unaligned (P = 124-365, so the first pools straddle the prefill-seeded ring) and aligned (P ≡ 0 mod 8) |
| Single-token history-advance reference | Section 2 |
| Committed state (positions, KDA recurrent and convolution, MLA, index-K, tail ring) versus permitted scratch | Section 4 |
| Justified tolerances | Section 5 |
| Returned ids == fixture is necessary, not sufficient | It is check H1a, one of about 25 |
| Measured on the intended prefix, once, before forcing changes the decision | PLAN-NOTES 2.3, tests F2, F3, O1-O3 and N1-N2, GPU checks H1c and H2a-c |

## 2. References: which are valid, and for what

1. **Single-token history advance: the `poison` schedule on the served path.**
   - Every draft is made unequal to the history, so every step commits exactly one history token, the row-0
     prediction.
   - The measured row is always row 0, conditioned only on committed state. The committed state advances one token
     per step.
   - It runs through the same verify forward (K + 1 rows, CUDA graphs, tail ring, KDA rollback) as every other
     schedule, so it isolates the effect of *acceptance bookkeeping*. The design claims that committed state
     depends only on committed tokens; every other schedule must agree with poison within decode-vs-decode noise.
   - The reference is itself validated against B1's own unhooked measurements (reference 3) and the prefill
     reference (reference 2).
2. **Prefill reference at every committed boundary.**
   - One causal prefill of prompt + history per history. The hook records every computed prompt row
     (`local_fw_ref`), so row o − 1 gives P(· | prompt + F[:o]) for every o at once. Causality makes this
     identical to a separate prefill ending at each boundary.
   - It is valid for the **conditioning**: is the decode-measured row on the intended prefix?
   - It is **not** a decode-fidelity reference: the prefill path builds index-K, KDA and tail state differently.
     So its tolerance is the engine's own decode-vs-prefill gap (section 5), never zero.
   - For committed *state*, prefill-built requests ending exactly at each snapshot frontier serve as the reference
     (section 4), with the same caveat.
3. **B1's own unhooked decode measurement.** In phase V1, B1 samples 8 histories freely (T 1.0, top_p 0.95,
   distinct seeds) with `logprobs=1`. The engine reports the log-probability of each sampled token from its own
   decode and verify path, without forcing. Two comparisons follow:
   - forced replay of the same tokens against those logprobs (decode against decode);
   - the free run's logprobs against the prefill reference: the **in-session null** for the decode-vs-prefill gap
     (section 5).
4. **Speculation off: not valid here.**
   - Without DFlash2, `VllmConfig.use_v2_model_runner` returns False for the hybrid target: the default-V2 check
     fails, so it falls back to the V1 model runner (`config/vllm.py:630-681`) **[S]**.
   - That runner has none of the hook's call sites. Its decode path differs, and at K = 0 the tail ring is 4 slots
     and the capture sizes differ.
   - Forcing V2 with `VLLM_USE_V2_MODEL_RUNNER=1` is a never-served configuration. It would itself need
     validation, and it costs a second server start (about 0.45 GPU-hours).
   - The historical "speculation off" runs of step 1 (amendment 1) are kept only as context for tolerance floors.

## 3. Phases of the GPU session (priority order; `runner/client_hookval.py`)

| Phase | Content | Checks |
|---|---|---|
| V0 | 4 greedy chat probes, unkeyed, hook ON | Bitwise equal to B1's run-b1-a1 probe hashes (the hooked image computes like B1 for unkeyed requests) |
| V2a | 8 traces × 7 schedules, one batch of 8 per schedule. Windows: 2 across the block boundary (P = 3,832); 6 from the prompt, capped at 2,500 tokens. Full selections recorded at 64/256/520/1,024 | H1, H2, H3, H5h |
| V2c | Prefill references of the V2a histories | H2 |
| V1 | 8 free runs (2,000 tokens), forced back with natural and poison, plus prefill references | H2 null and H2 |
| V4 | D2 window injection: 8 windows injected; the same 8 uninjected; one injected request alone; prefill rebuilds of prompt + window. Every request dumps its window pages at removal | H5 |
| V3 | 8 snapshot requests (frontiers 4,350-4,354 across the block; 8 residues mod 8 elsewhere; mixed schedules). Then prefill-built references at the **actual** frontiers, one at a time | H4 |
| V2b | All 24 traces × {natural, oracle}, full histories to the natural close; prefill references for the 16 not in V2a | H1, H2, H3 null |

The mapping guard (`GLM53_MAPGUARD=device`) runs throughout. A readout after every phase gives per-phase counter
deltas. The guard independently verifies:
- positions against scheduler truth (`pos_contig`, `pos_bounds`);
- MLA, index-K and tail slots;
- the D7 straddle byte oracle.

So the forcing's position accounting is checked by code that never reads the hook.

**Request plan [M]** (`runner/client_hookval.py --plan`, CPU): 193 forced or reference requests plus 8 free runs;
202,863 forced tokens; 156,950 prompt tokens (`runner/plan-dryrun/plan.json`).

## 4. Committed state against permitted scratch

At a snapshot after the step whose commit first reaches output count F (the actual F is recorded):

- The last processed position is f = P + F − 2.
- The token F[F − 1] is committed but not yet processed. It is the next step's row-0 input.

| State | Committed (checked) | Permitted scratch (reported, not checked) | Reference and check |
|---|---|---|---|
| Token ids and positions | `all_token_ids[P:P+F] == F[:F]`, `total_len == P + F`, `num_computed == P + F − 1`, last token == F[F − 1], after **every** step (device counters) and at snapshots | none | Exact |
| KDA recurrent state | Slot `spec_state_indices[r, num_accepted − 1]` (baseline full-state rollback reads this slot next) | The other K slots (states after rejected drafts) | Prefill-built state at the same f: relative L2 of a fixed 256-direction sketch ≤ 0.05, **and** the committed slot at least 3× closer than any scratch slot (identification) |
| KDA convolution state | The window ending at f (offset by `num_accepted`) | Entries past it | Prefill-built conv row: reported; same identification rule |
| MLA records | Positions ≤ f | Positions > f (written by rejected-draft rows) | Prefill build: byte-identical fraction reported; the frontier record must match its own position better than its neighbours (addressing) |
| Index-K pools | Pools completed at ≤ f | Pools completed by rejected drafts at > f | Prefill build: no pool with relative L2 > 0.25 (gross) |
| Tail ring | Slots of positions in the incomplete pool ending at f | Slots of draft positions > f; old pool members | Prefill build (B1's prefill seeds the trailing incomplete pool): reported |

**Positive controls for page addressing.** A wrong page address compares unrelated records: relative L2 near 1.4,
byte agreement near chance.
- The B1-arm decode-vs-rebuild comparison in V4 (H5b) must be clean. If it is not, the addressing or the
  reference is broken and the run is INVALID, not a finding.
- The KDA identification rule (H4b) catches a wrong spec-row mapping.

## 5. Tolerances and their justification

**Logits: decode against prefill (H2).**
- The hooked forced-decode gap |Δ ln P(forced token)| between the decode-measured row and the prefill reference
  must stay within NULL_FACTOR = 2 × the in-session unhooked null at q50, q90 and q99, with absolute floors.
- The null is B1's own free-run logprobs against prefill, from phase V1.
- Floors: 0.01 / 0.30 / 1.50 nats, about 2× the historical unhooked spec-on ×3 gap on 0.9.1
  (`step2-spec-depth/k3-r2.json` **[M]**: q50 0.0011, q90 0.162, q99 0.745, q99.9 3.19, top-1 agreement 0.972).
  The spec-off runs gave 0.0017 / 0.147 / 0.592 / 0.972.
- Argmax agreement ≥ 0.95.
- **Why a distributional criterion.** The engine's own decode-vs-prefill gap is heavy-tailed: 1.4-2.1 % of
  positions exceed 0.5 nats, with a maximum of 18 nats. These come from selection flips near ties under fp8 index-K
  and bf16 accumulation. A per-position tolerance would either reject a correct instrument or pass a broken one.
- **Power control (H2b, H2c).**
  - The gap at a one-position shift must have median ≥ 1 nat and ≥ 20× the aligned median, aggregated.
  - For **every request**, the aligned median must be < 0.1 × the shifted medians.
  - A misattributed row (off by one, or conditioned on a rejected draft) is therefore detected per request. If the
    shift is not grossly worse, the check is INCONCLUSIVE, not passed.
  - On CPU the synthetic dry run shows the per-request check catching a single shifted request among 80 that the
    aggregate quantiles miss **[M]**.

**Logits: decode against decode (H3).**
- Every schedule against `poison`, same position: q50 / q90 / q99 within 2× the decode-vs-decode null (same
  floors), and argmax agreement ≥ 0.97.
- The null is the same history and schedule in a different batch (V2a natural against V2b natural). Batch
  composition changes MoE grouping and therefore numerics.

**State (H4).**
- KDA: ≤ 0.05 relative L2 with a 3× identification margin. The fp32 recurrent state is driven by bf16 inputs. A
  wrong slot differs by one token's full update.
- Index-K: gross threshold 0.25 relative L2, frozen in the probe proposal (section 8.5). The B1-arm distribution
  must show it discriminates: 99.9th percentile below 0.25, else INCONCLUSIVE.

**D2 injection (H5).**

| Arm | Gross-pool fraction |
|---|---|
| B1 arm | ≤ 0.1 % |
| Solo injected request | ≤ 0.1 % |
| Injected at 8 in flight | > 30 %. The CPU emulation of stock D2 gives 78-90 % at 8; 30 % is a deliberately low bar for "corruption present" |

- Shadow clean → non-gross ≥ 99 %; shadow corrupt → gross ≥ 95 %.
- Guard: `tail_w_stash` mism equals the null class, is a non-zero multiple of the injected-token count (one per
  owned indexer layer), and comes only from injected requests. Zero `idx_w_dec`, `tail_w_seed`, D7 (`s_mism`,
  `d7_fallback`) and position violations.

**Schedules (H1j).** Each prescribed accepted count holds on ≥ 95 % of steps (the window end clamps the last
steps), and `sched` covers every a in 0..K.

## 6. Frozen criteria (`analysis/analyze_hookval.py`)

Verdict precedence: **INVALID > INCONCLUSIVE > VALID**.

| Id | Check | Level |
|---|---|---|
| V0 | Unkeyed probes bitwise equal to B1's run-b1-a1 | invalid |
| H1a | Returned ids == forced ids and finish "length", for every forced request | invalid |
| H1b | A hook record exists for every forced request | invalid |
| H1c | Each window position measured exactly once (`nmeas == 1`) | invalid |
| H1d | No NaN in the tracked pre-mask log-probabilities | invalid |
| H1e | Post-mask values read from served processed logits on every row (`post_ok`) | invalid |
| H1f | Served top_p mask inside the eligibility bounds (p_post > 0 ⇒ elig_hi; elig_lo ⇒ p_post > 0) | invalid |
| H1g | Last committed token == history after every step | invalid |
| H1h | Both ranks recorded identical metrics (identical committed tokens per step) | invalid |
| H1i | No hook errors | invalid |
| H1j | Accepted-count histograms follow the schedules (bonus accounting) | invalid |
| H2-null | In-session null available (> 100 positions) | inconclusive |
| H2a | Forced-decode vs prefill gap within the null tolerances; top-1 ≥ 0.95 | invalid |
| H2b | Aggregate alignment power | inconclusive |
| H2c | Per-request alignment (no misattributed request) | invalid |
| H3 | Schedule invariance against the single-token advance | invalid |
| H4a | Integer committed state at every snapshot | invalid |
| H4b | KDA committed slot matches the prefill-built state with identification margin | invalid |
| H4c | No gross index-K pool before the frontier | invalid |
| H5a | Injected tokens ≥ W for injected requests, 0 elsewhere | invalid |
| H5b | B1 arm gross ≤ 0.1 % (addressing positive control) | invalid |
| H5c | 0.25 discriminates on B1 | inconclusive |
| H5d | Injected at 8: gross > 30 % | invalid |
| H5e | Injected alone: gross ≤ 0.1 % | invalid |
| H5f | Shadow ledger agrees with measured bytes | invalid |
| H5g | Guard pattern as section 5 | invalid |
| H5h | Exposure records exist at every designated position (in-graph capture works) | invalid |
| coverage | No phase skipped for time | inconclusive |

**Reported, not gating:**
- distributions behind every check;
- per-schedule acceptance;
- step time per schedule (needed to price the contrast);
- out-of-nucleus counts;
- padded tokens;
- ReplaySSM: must not appear in the server log.

**Dry run [M].** On synthetic runs with the hook's real record formats (`tests/make_synthetic_run.py`):
- "good" gives **VALID**.
- "bad" gives **INVALID**, failing exactly the planted defects: H1c, H2c, H3, H4b, H5b (and H5c) and H5g.

## 7. Runner, pins, budget

**Files.** `runner/run_hookval.sh` (copy of the frozen B1 guard runner's structure), `runner/client_hookval.py`,
`runner/images.txt`.

**Image.** `local/glm53-v0.9.1-b1-forcehook-mg:local` = `sha256:1b9f8b15…` **[M]**.
- FROM the B1 mapguard image (`sha256:4d7bd73e…`, verified).
- Gate 0 checks the replaced files' pre-hashes; the build then checks post-hashes, `py_compile`, marker counts and
  layer ancestry.
- The `.py` manifest differs from its base in exactly the 6 files (`build/manifest-diff.txt`).
- New tag only; no tag was replaced.

**Server.**
- Recipe `glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234+d7` with IMAGE overridden; shipped graph mode;
  DFlash2 ×3; both ranks.
- `DEBUG_ENV`: `FORCEWIN=on`, `_D2=1`, `_SHADOW=1`, `_SNAP=1`, `_WMAX=5120`, `GLM53_MAPGUARD=device`.

**Gates.**
- `APPROVED_GPU_H` is required and ≤ 2.4.
- At most 2 attempts.
- Frozen-hash check of this directory's `SHA256SUMS`.
- Image-id pin.
- Client plan dry run.
- Pre-flight:
  - no glm53* container;
  - GPU memory and utilization idle;
  - the process table is checked, because `--query-compute-apps` misses graphics clients.
- Readiness:
  - no engine error;
  - no hook or guard error during start-up;
  - **the hook "registered" on both ranks** (it must be allocated before graph capture);
  - guard init on both ranks.
- Watchdog at the wall cap.
- EXIT/INT/TERM trap:
  - server log and container file hashes and env;
  - stop;
  - one cost line to the moment the GPUs are released;
  - `docker cp` of the hook and guard outputs;
  - the frozen analysis in a CPU container.
- No `set -e`.

**Cost** (`runner/cost_hookval.py` **[M]**; planning rates from measured receipts: start 13.5 min, prefill
3,000 tok/s, verify step 62 ms at 8 in flight):

| Part | Seconds |
|---|---:|
| V0 | 30 |
| V1 (free runs plus forced replay) | 252 |
| V2a schedule matrix | 660 |
| V2b | 354 |
| V2c and references | 37 |
| V3 snapshots | 117 |
| V4 D2 | 230 |
| Readouts | 64 |
| **Work** | **1,744** |
| Start-up plus stop | 900 |
| **Wall** | **2,644 s = 0.73 h → 1.47 occupied GPU-hours** |

- **Proposed item ceiling: 2.4 GPU-hours**, = 1.3 × 1.47 + 0.5 invalid-start allowance.
- Per-attempt wall cap 3,240 s (1.80 GPU-hours); MIN_WALL 2,000 s.
- Phases run in priority order, so a stopped session keeps V0, V2a, V2c, V1 and V4.

**Launch (after authorization only):**

```bash
APPROVED_GPU_H=2.4 <runs>/hook-validation/runner/run_hookval.sh 1
```

## 8. CPU test results [M]

All runs used CPU containers of the B1 image (or of the built validation image where marked), with
`--network none`, no `--gpus`, `CUDA_VISIBLE_DEVICES=` and `TRITON_INTERPRET=1`. Logs are in `tests/out/`.

| Suite | Image | Result |
|---|---|---|
| `test_forcing.py`, 9 parallel shards (`run_cpu_tests.sh forcing-par`; logs `tests/out/test_forcing_*.log`): F1-F6 per schedule; F6b-c; F7, F7n, F7b; O1-O3; N1-N2; R1. Installed B1 rejection sampler, sampler, `RequestState`, `post_update`; toy model | B1 + hooked copies | 66/66: natural 7/7, oracle 8/8 (incl. F6b), poison 7/7, reject:1 7/7, reject:2 7/7, reject:3 7/7, sched:7 7/7, seed 8/8 (incl. O3), rest 8/8 (F7, F7n, F7b, O1, O2, N1, N2, R1). Measured metrics equal the single-token reference with max \|Δ\| 0 for every schedule; `nmeas == 1` everywhere |
| `test_d2_inject.py`: D1a-g, D2, D3a-b, D4, D5a-b, D6a-c | B1 + hooked copies; **built image** | 16/16; 16/16 |
| `test_exposure.py`: X1, X2a-d, X3 | B1 + hooked copies; **built image** | 6/6; 6/6 |
| Inertness I1: original B1 / hooked env unset / hooked env on, unkeyed / built image env unset | four installs | identical output hashes |
| MG1: hooked mapguard index-K file imports with both instrumentations | hooked copy; **built image** | PASS; PASS |
| IMG1: installed files == reviewed hook copies; IMG2: built image inert with env unset | **built image** | PASS; PASS |
| Analysis dry run, synthetic good / bad | B1 (CPU, torch) | VALID / INVALID with exactly the planted defects |
| Client plan (`--plan`) | host | 193 requests built; ids from the 24 exact traces verified (close token at the recorded position) |

**Shims, test-only.**
- vLLM's UVA buffers become plain CPU tensors, and pinned memory is off (no driver).
- Triton-interpreter `libdevice.log1p` becomes log(1 + x). It is used by the gumbel noise and one rejection-ratio
  term, so it changes random draws only.
- `apply_top_k_top_p` uses the engine's own PyTorch implementation (the GPU path below 8 rows); the Triton top-p
  kernel needs a device SM count.
- The model forward is a deterministic toy.

None of these touches the hook's logic or the forced commits. On GPU, H1f checks the served Triton top-p mask
against the eligibility bounds.

### Post-inspection change to test F7 (2026-10-14, after the first sharded run; hook code unchanged)

- **Before:** F7 compared the co-runner's **whole** sampler output row, `sampled_token_ids[1]` (K + 1 entries),
  between a hook-on and a hook-off call of the same step, plus `num_sampled`.
- **After:** F7 compares what the engine **consumes** for the co-runner, over 4 steps: the first `num_sampled`
  tokens, `num_sampled` and `num_rejected`. Two checks are added:
  - **F7n**, a negative control: a planted change of the co-runner's committed token must be detected (it is);
  - F7b, as before: the forced request commits history tokens.
- **Cause of the failure.** Entries past `num_sampled` are uninitialised scratch from `torch.empty` in
  `rejection_sample`. They differ between two **hook-off** calls of the same step (`tests/out/diag_f7.log`):
  - trial 1, both hook off: `[59093, 20, 21, 22]` vs `[59093, 2, 2, 2]`;
  - trial 2: pointer-like values such as `502838656`.
  - The committed token (59093), `num_sampled` (1) and `num_rejected` (3) were identical in every call.
  - The earlier full run passed F7 only because the allocator happened to reuse the same memory.
- **Why the new criterion is not a loosening.** Nothing the engine consumes reads the scratch:
  - `post_update` loops `i < num_sampled`;
  - the host output deletes `token_ids[num_tokens:]` (`async_utils.py:169-172`);
  - the proposer takes `num_sampled`, `num_rejected` and `last_sampled_tokens`.
- **Evidence that every consumed value is bit-identical** (`tests/out/diag_f7b.log`):
  - Same step, hook on vs off: the co-runner's processed (top_p) logits rows are bitwise equal, and so are the
    forced rows. The raw logits are unchanged by the hooked call. Committed tokens, `num_sampled` and
    `num_rejected` are equal at all 3 steps.
  - Two engines over 6 steps, one with the hook forcing request 0 and one with nobody forced: the co-runner's
    committed sequence, `num_computed`, `total_len`, last token, `all_token_ids` and draft tokens are identical
    after every step.
- The I1 inertness check, rerun on the unchanged hook, gives identical hashes across all four installs.
