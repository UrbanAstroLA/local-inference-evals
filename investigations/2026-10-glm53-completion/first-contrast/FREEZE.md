# First long-context contrast: frozen package under PLAN.md amendment 22 (CPU only; NOT launched)

2026-10-14. Option B, amended. Design: `PROPOSAL.md` (pre-22 text kept as `PROPOSAL.md.pre22`; changes labelled
[A22-n] = amendment 22 item n). This file maps each amendment-22 requirement to its implementation and its test
evidence. Nothing here authorizes GPU time.

How it was prepared:
- **No GPU.** No GPU or glm53* container was started; nothing used `--gpus`; no `sudo`.
- **CPU containers only**, all `--network none` and `CUDA_VISIBLE_DEVICES=`:
  - the 0.9.1 release image (tokenizer, for the id freeze);
  - the B1 image `sha256:799d3cf8…` (analyses, synthetic dry runs, the GATE-1 stand-in);
  - the attempt-2 image `sha256:ac05230b…` (hook CPU tests);
  - the GATE-1 step test's stand-in container ran `sleep` under the name `fcgate-test-*`, never glm53*, and was removed.
- **File ownership.** Files that the first container runs wrote as root were re-owned to the operator's user through a
  CPU container (`chown`); every later run used `--user`.
- **The hook, the image and every hook-validation file are unchanged.** The session sets only the environment
  `FORCEWIN_WMAX=8448`. No D2-arm termination value was read.

## Pins

| What | Value |
|---|---|
| Serving image | `local/glm53-v0.9.1-b1-forcehook-mg2:local` sha256:ac05230bd94b… (`runner/images.txt`, the attempt-2 pin) |
| Hook module | `vllm/forcewin_hooks.py` sha256 15e93e236067… (= hook-validation/hook/forcewin_hooks.py); checked in the running container before any traffic |
| CPU analysis image | sha256:799d3cf8e5d1… (B1) |
| Validated package this derives from | hook-validation/SHA256SUMS sha256 4ae4ad2206ad… (amendment 21i) |
| Fixture ids | `fixtures/ids_manifest.json` sha256 3f6a0838b70d…; 128 / 128 convention checks; first_contrast.json 8df042ec7c01… |
| This package | `SHA256SUMS` — see "Freeze" at the end |

---

## Requirement → implementation → evidence

### 1. Analysis [A22-1, A22-2, A22-6, A22-8]

| Requirement | Implementation | Evidence |
|---|---|---|
| Frozen primary statistic kept (ln window-mean pre-top_p P(`</think>`), fixed-fixture run aggregate) with explicit limits | `analysis/analyze_fc.py` `run_stats`, `agg`, `contrast`; the `LIMITS` text is written into every result | Synthetic `good` reproduces the planted −0.6 nats (Δ −0.603, paired −0.604 / −0.604 / −0.600, N 0.011) |
| End-position-excluded sensitivity (added) | `sens["end_position_excluded"]` (Q_f without o = W) | Computed in every complete synthetic run |
| Question-weighted, near-close vs active, common-tail construction comparison, post-top_p / eligibility, each stop token separately | `sens[...]` (question_weighted, near_close, active, post_top_p_close, eligibility_close_share_lo/hi, stop_<id>_pre/post/elig_lo); `res["lp_control"]` | Same |
| Selected corrupted pools by fixture × layer | `exposure_table` → `exposure_by_run_fixture_layer` (C, D, LC, LD) | Synthetic `uninformative` vs `good` |
| 1/20 claim removed; three paired differences; separation and beyond-null as operational screens; no significance | `contrast()` returns paired d_r and the screens; no permutation or significance field | Unit U2 (fields), U3 (separated but not beyond the null), U4 ("1/20" absent from the analysis) |
| Readings exactly as §8 (three cases + direction; "no effect detected by the prespecified screen") | `READING_1/2/3`, `DIR_UP/DOWN` constants | Unit U4 (verbatim); synthetic good → case 1, null → case 2, uninformative → case 3, noisy → no reading (N 0.74 > 0.25), a-incomplete → INCOMPLETE |

### 2. Prefill null at a different build length [A22-3]

| Requirement | Implementation | Evidence |
|---|---|---|
| A prefill reference at a different build length, compared only on shared causal positions; adapt a planned reference warm if cache safety is demonstrable | LW2 (the LP round-2 warm) at build length P + 8,192 vs LW1 at P + 7,168; both dump [P, P_L]; N_pp on pools wholly in [P, P_L). **Adapted, no added rebuild** (+8 × 1,024 prefill tokens, about 3 s) | Plan rule P12 (equal build lengths refused: tamper T7); synthetic `good` G1-content (N_pp from LW1 vs LW2) |
| Cache safety demonstrated | Restated invariant (PROPOSAL 2.5) enforced over the ordered plan: P6. LW2 runs before any decode under l2; the later l2 prompts (P_L) read only blocks LW2 prefilled | Plan check 0 failures on the session plan; tamper T11 (P6) and T2 (shared salt) refused |
| Matching by fixture × layer × shared positions; exact cache accounting; per-fixture flags; pooled gate with the original criteria beside | `fc_common.matched/content/limit_of/table/flags/asym_flags`; G1-cache; G1-content with G1-content-orig, -ref2, -table, -flags (layer-units AND positions, R3) | Synthetic g-content → G1-content FAIL only; g-cache → G1-cache FAIL only; unit U6 (position vs layer-unit flag) |
| N_pp = 0 wording corrected | `limit_of` docstring; PROPOSAL 5.1 | Unit U1a: N_pp = 0, N_dd = 1 % → limit 2 %, not 0.1 %; U1b: both 0 → 0.1 % |
| Pre-stated: a failing numerical gate means "unvalidated under these bounds"; B1 not automatically defective; no post-hoc revision | `gate_fc.verdict()` reading text; PROPOSAL 5.3 | Synthetic g-align / g-cross / g-regime / g-content verdict text |

### 3. In-session validation before the injected batches [A22-4]

| Requirement | Implementation | Evidence |
|---|---|---|
| Session order: alignment and content from clean runs and reference builds computed BEFORE any D batch, by a CPU gate step between batches from records already copied out; reference warms earlier; stop before injected runs on failure | `client_fc.py` stage A (V0, LW1, LW2, C1, C2, CV) → `runner/gate_step.sh` (docker exec + tar copy of stage A's records; `analysis/gate_fc.py` in a CPU container) overlapping stage L (LC1, LC2) → **no D batch unless GATE-1 = PASS** (client rc 10). LW3 stays the round-3 warm (moving it earlier would risk evicting its prefix) | Plan rule P11 (tamper T6: D2 before CV refused; T8: PS-c before D3 refused); GATE-1 step test against a stand-in container: good → PASS rc 0, g-content → FAIL rc 1, 236 / 236 stage-A files copied, no later-phase file |
| WMAX 8,448 without truncation | G1-wmax (init on both ranks; every keyed request's hook record holds its full planned list); runner checks the in-container hook sha and init WMAX before traffic | Synthetic g-wmax → G1-wmax + G1-forcing; hook tests W1, W2, W3 |
| Absolute positions on cached prefixes | G1-abspos (hook P = planned prompt length, dump ranges, snapshot frontiers); alignment against cold references | Synthetic g-abspos → G1-abspos only; hook tests W4, W4b, W5, W6 |
| Exact attribution in flattened and padded steps | G1-attrib: step-matched exposure records in every tracked layer at every designated position, counted separately in FULL-graph-padded steps (hook step log); selection records step-matched. Stricter than attempt-2's H5h-r (all layers required; attempt 2 had 7,922 / 7,922) | Synthetic g-attrib (misattribution only in padded steps) → G1-attrib only |
| Both block crossings | G1-cross: ±128 indices around each crossing aligned with a prefill reference; every crossing covered (M4's second only via LW2's longer reference) | Synthetic g-cross (second crossing of L3 only) → G1-cross only |
| M3's selection-regime crossing | G1-regime: ±256 indices aligned; every recorded final selection there has 511 distinct causal pool ids | Synthetic g-regime → G1-regime only |
| Clean prefix state after injected traffic (C3 residue) | Post-session rows C3-residue (C3 valid, clean, aligned with LW1 / LW2, within the GATE-1 content rule) and PS-prefix | Synthetic a-c3 → C3-residue; a-ps → PS-prefix |
| Guard establishes the injection cannot write into the shared cached prefix or its committed recurrent state; if incomplete, sampled before / after state checks | H5g-fc: tail mism == null == owned layers × injected tokens, only in D / LD phases; zero decode, seed, prefill, D7 and position mismatches (index-K and tail writes, mapping level). **The guard does not monitor MLA or KDA writes, so the protection is incomplete** → prefix sentinel PS-a / PS-b / PS-c (existing snapshot capability): shared cached block pages byte-identical; KDA state rebuilt from the cached boundary state bitwise equal (or within the PS-a / PS-b null) | Synthetic a-guard → H5g-fc; a-ps → PS-prefix; plan rule P11 |

### 4. Convolution-state boundary control [A22-5]

| Requirement | Implementation | Evidence |
|---|---|---|
| Reuse existing snapshot capability around a block crossing; check the actual conv-state slot and migration with KDA state and bookkeeping; against a matched decode reference | CV phase: L5, L1 × poison / oracle / reject:2 / natural, prompt P' = B1 − 63, snapshots at frontiers B1 − 8 … B1 + 4 (identical F for poison, oracle, reject:2; a consecutive poison run straddles the column change). Gated: G1-CV-slot, -book, -cross, -persist (exact), -conv, -ssm (21g-2 screen vs poison) | Synthetic g-cvmig (wrong migration offset) → CV-persist + CV-conv; g-cvslot → CV-slot; g-cvbook → CV-book (+ persist, conv); good → all pass |
| Mapping and functions validated on real engine data | `conv-a2/conv_a2.py` on attempt-2's 53 V3 pairs × 2 ranks: mapped row offset best in 106 / 106; conv vs prefill identification passes the screen (100 % layers 0-17, 99.7 % deeper; layers 0-2 rel ≤ 0.009, margin ≥ 101); slots 0 violations; persistence 408 / 408 bitwise | `conv-a2/conv_a2.json`, `.out` |
| Hook change needs costing | **None needed**: the existing `local_fw_snap` records the engine's conv slot (`conv[spec_state_indices[:, 0]]` = `conv_state_indices`). Cost of CV about 43 s central (0.024 GPU-hours) | — |
| Limitation | No prefill reference at the long-context CV frontiers (a cold ~110k prefill each, about 50 s); decode-vs-prefill conv at a crossing rests on attempt-2 data at 4,352. Attempt 2 never observed a pre-migration snapshot; CV is the first observation of the migration itself | — |

### 5. Runner and client [A22-4, A22-9]

| Requirement | Implementation | Evidence |
|---|---|---|
| `run_fc.sh` + `client_fc.py` from the validated runner / client; new phase set; WMAX 8,448 env; no hook change | `runner/run_fc.sh` (diff scope in its header), `runner/client_fc.py` (phases of PROPOSAL 3.4; seeds fixed per phase / fixture / schedule; never truncates a batch; keeps the primary phases' reserve before LP stage-B batches) | DRY_RUN; plan 240 requests |
| Budget gate (4.29 incl. any added reference; reprice and report if exceeded), single attempt | CEILING 4.29, single attempt (refused if any fc1 line exists), wall cap min(7,714 s, approved × 1,800), MIN_WALL 4,477 s | Runner tests R2-R9, R12 |
| Image / hook pins, frozen sums | Image id pin; SHA256SUMS check; in-container hook sha and init WMAX before traffic | R10 (wrong image id), R13 (tampered analysis), R14 (tampered ids file) |
| Pre-flight GPU idle; report desktop Firefox / Electron on GPU 1 | Attempt-2 idle limits kept; every desktop graphics client is printed and logged in `preflight.txt` (tolerated within the limits, never silently) | Code (nvidia-smi is shimmed to refuse in the tests; the pre-flight is not reached in DRY_RUN) |
| Watchdog, cleanup on all exit paths, cost line, no `set -e` | Inherited unchanged: EXIT/INT/TERM trap, watchdog at the wall cap, cost line at GPU release | Inherited (hookval attempts 1-2) |
| Disk check for 45-60 GB dumps, fail early | ≥ 200 GB free on /data and on the docker root (<docker-root>), before anything starts | R11 (150 GB planted → refused) |
| Refusal / tampered-plan tests with shims | `tests/test_runner_fc.sh` (shims: docker image inspect / info only, nvidia-smi and curl refuse); `tests/test_plan_fc.py` | Runner tests and plan tamper tests (counts below) |
| CPU hook tests for WMAX 8,448 and forcing on a cached prefix | `tests/test_fc_hook.py` via `tests/run_fc_cpu_tests.sh` (attempt-2 image, validated scaffolding) | Hook tests (counts below) |

### 6. Repricing, sums, DRY_RUN [A22-9]

| Requirement | Implementation | Evidence |
|---|---|---|
| Reprice from attempt-2 measured times; state the decode-speed assumption and its sensitivity | `cost/cost_fc.py` → `cost_fc.json`, `.out` (pre-22 kept) | Central 2.86, pessimistic 4.29, optimistic 2.52 GPU-hours; sensitivity 20 / 25 / 33 / 41.6 tok/s → 3.46 / 3.15 / 2.86 / 2.66 |
| Regenerate SHA256SUMS, final DRY_RUN | `make_sums.sh`; `DRYRUN.out` | Below |

---

## Test summary (all CPU)

| Suite | Result | Log |
|---|---|---|
| Fixture id freeze (re-tokenization conventions, hashes) | 128 / 128 checks | `fixtures/freeze_ids.out`, `ids_manifest.json` |
| Plan check on the session plan (240 requests, rules P1-P13) | 0 failures | `tests/out/plan-check.log` |
| Tampered-plan negatives (D2 key on a C request, shared salt, list truncated to 5,121, cached-count mismatch, duplicate seed, injected batch before GATE-1, equal build lengths, PS-c before D3, composition order, CV targets, cache invariant, missing D2 key) | 13 / 13 (untampered passes; 12 refused with their rule) | `tests/out/plan-tamper-tests.log` |
| Runner refusals with shims (DRY_RUN pass, budget line, no approval, over ceiling, negative approval, attempt 2, attempt recorded, below MIN_WALL, other items ignored, wall cap capped, wrong image id, disk 150 GB, tampered analysis, tampered ids, existing run dir) | 15 / 15 | `tests/out/runner-tests.log` |
| Hook CPU tests, fast (W1, W3, W4 natural / reject:2, W4b, W5a, W5b, W6, W7) | 9 / 9 | `tests/out/test_fc_hook_fast.log` |
| Hook CPU tests, long (W2: the 8,199-id list under oracle, every index once and equal to the reference, 3,882 s; W2n: natural drafts at absolute positions 7,636-8,235 on a cached prefix) | 2 / 2 | `tests/out/test_fc_hook_long.log` (run before a docstring-only edit of test_fc_hook.py) |
| Analysis unit checks (U1 N_pp = 0 limit, U2-U3 screens, U4 verbatim readings, U5 conv mapping, U6 R3 flags) | 8 / 8 | `tests/out/test_fc_units.log` |
| Synthetic dry runs: GATE-1 verdict, the planted rows exactly, the reading class | 20 / 20 variants | `tests/out/synth-check.log`, `synth-*/` |
| GATE-1 step against a stand-in container (copy completeness, verdict, rc) | 2 / 2 | `tests/out/gate-step-tests.log` |
| Conv-state comparison on attempt-2 data | screen passed; 106 / 106 mapping; 408 / 408 bitwise; 0 slot violations | `conv-a2/conv_a2.json`, `.out` |

## Freeze

`SHA256SUMS` covers every design, code, fixture, cost and evidence file (`make_sums.sh`; excluded: the item ledger
`costs.jsonl`, session outputs `run-fc*/`, `DRYRUN.out`, bytecode, and the synthetic runs' raw data apart from each
variant's `gate1.json` and `fc_result.json`). Its own sha256 and the final DRY_RUN line are in `DRYRUN.out` (a
file cannot carry its own hash). `SHA256SUMS.pre22` is the pre-22 package list.

## Not met, limits and deviations

1. **Budget.**
   - Under the pre-22 ceiling formula, which includes the 0.42 GPU-hour invalid-start allowance, the repriced ceiling
     is **4.71, above the 4.29 target**.
   - The single-attempt ceiling max(estimate × 1.3, pessimistic) is 4.29 and equals the target. The runner is frozen
     at 4.29.
   - The new pessimistic items (one LP prefix eviction 0.24, a GATE-1 wait 0.11, CV and sentinel 0.06) are what move
     the pre-22-formula total; the central increase is only +0.035.
   - Authorization decides whether the allowance is wanted as a reserve (4.71). That would need the runner constant
     changed and the package re-frozen.
2. **Conv-state control:** decode reference only at long context; no prefill reference at the CV frontiers (section 4).
3. **The long-context decode rate is still unmeasured;** every long-context gate is a first measurement.
4. **GATE-1 attribution is stricter than attempt 2** (all tracked layers required at every designated position).
   Attempt 2 would have passed (7,922 / 7,922).
5. **The guard's coverage** is index-K / tail only. MLA and KDA state of the shared prefix are checked by sampling (the
   sentinel on s3's last cached block and the rebuilt KDA state), not exhaustively.
6. **The all-8-in-flight sensitivity** is per run (each run's own subset of Q_f), not a common position set.
7. **Synthetic dry runs test the code paths and plants, not engine behaviour.** The real-data validation of the new
   analysis code is limited to the attempt-2 conv-state comparison.
