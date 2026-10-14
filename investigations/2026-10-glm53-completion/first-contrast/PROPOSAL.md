# First long-context contrast: B1 against decode-window D2 injection (proposal; CPU only; NOT launched)

2026-10-14. This document designs and prices the first long-context contrast under PLAN.md amendments 20, 21
(sections 5-7), 21g (sections 6-7) and 21h-21j. It is for review. Running it needs an amendment that records the
frozen package and an operator authorization with a GPU-hour total.

How it was prepared:
- No GPU was used. No GPU or glm53 container was started, and nothing used `--gpus`. No `sudo`.
- One CPU container (`--network none`, `CUDA_VISIBLE_DEVICES=`) read existing attempt-2 records to measure a prior
  B1-vs-B1 null (section 2.6). It read the two clean B1 arms only. The D2 arm was not read, so no D2-vs-B1 termination
  difference was seen before this design was written.
- The hook, the image and every hook-validation file are unchanged.

Files:

| File | Contents |
|---|---|
| `fixtures/position_plan.py` → `position_plan.json`, `.out` | Lengths, cache geometry and designated positions per fixture: ids, positions and hashes only (96 checks pass) |
| `fixtures/exclusion_rule.json` | The flagged-fixture rules R1-R5 (section 5.4), ready to freeze |
| `fixtures/known_fragile_be2e9e349395.json` | be2e9e349395's 13 three-way-reproducible gross index-K units (rank, layer, pool, positions), with the source hashes |
| `cost/cost_fc.py` → `cost_fc.json`, `.out` | Per-component costs, options and ceilings |
| `prior-null/b1_null_a2.py` → `b1_null_a2.json` | The prior B1-vs-B1 null from attempt-2 records |

---

## 1. What is being asked

1. **Primary.** At fixed long-context histories, does D2 injected only during the forced decode window change the
   selected keys and the reasoning-close (`</think>`) probabilities beyond the matched in-session null?
   - Arms: clean B1 ×3 against B1 + D2 window injection ×3.
   - Same 8 fixtures, 8 requests in flight, clean and injected conditions in separate batches.
2. **Long-prompt / short-output control (21g §6).** At the same token history and final context position, does the
   result depend on how the history state was built?
   - Long construction: a shorter prefill followed by the planned 8.2k forced window.
   - Short construction: a long prefill followed by a short (about 1k) forced window.
   - Compared over common final positions, clean and injected, with matched composition and concurrency.
   - **This is not a pure output-length test** (section 6).

The contrast makes no claim that a probability difference explains GPQA completion rates. No full GPQA pass is
purchased because of it (21g §7).

---

## 2. Findings from the instrument that shape the design

### 2.1 The two flagged fixtures are not in the first-contrast set

`be2e9e349395` (outside its own null) and `376073921446` (arm asymmetry) are exact-id short traces under 5k tokens
from attempt 2's V4 D2 set (`runner/client_hookval.py`, `D2SET`).

The first-contrast set in `fixtures/first_contrast.json` is a different set:
- **Fixtures:** L1, L4, M3, M4, L3, L5, A8, L8.
- **Source ids:** ca7fd76b2b7a, f0c2498e7b7f, cd6c18552504, c253c576662a, 09ba09a88d7b, fc3010f3a808, 4db8bbe82d5c and
  77100ed719f5.
- **Format:** re-tokenized text histories of 63k-120k tokens.

**Update (coordinator input, from `hook-validation/followup-a2/FOLLOWUP-a2.md`, sha256 c579b91a…):** neither flag
is an instrument defect.
- **be2e9e349395** shows deep-layer numerical fragility common to clean and injected decodes: 13 of its gross units
  reproduce across three independent decodes.
- **376073921446**'s asymmetry is a single-position event: pool 128 inflates the layer-unit counts against a zero
  prefill null, and the arms are symmetric by position (5 against 6).

The coordinator's frozen decisions, and how this design applies them:

| Decision | Applied as |
|---|---|
| (1) Keep both in the primary aggregate, flagged, with no mechanism claim | They are not long fixtures, so they can only be in a primary aggregate if their set is run. The design adds an **optional short stratum S** (section 3.7): the attempt-2 D2 set (8 exact-id windows of 256-2,000 tokens, including both ids) under the same arms and rules. Its own primary aggregate keeps both fixtures in. S is never pooled with the long stratum, because the context range differs |
| (2) Leave-flagged-out sensitivity beside the primary | Rule R2: for both strata, every statistic, screen and reading |
| (3) Count flags by position as well as by layer-unit | Rule R3 |
| (4) be2e9e349395's 13 reproducible gross units as a known fragile set, reported separately | Rule R4; `known_fragile_be2e9e349395.json` |

The same rules govern any in-session flag on the 8 long fixtures (R5). **No fixture is excluded.**

**Coordinator decision needed:** run S or not. Without S, decisions (1) and (4) have nothing to act on in this
contrast, and R2, R3 and R5 still apply to the long fixtures. S adds about 0.20 GPU-hours (section 8).

### 2.2 The validated sessions would have truncated these windows

Both validated sessions ran `FORCEWIN_WMAX=5120`:
- `run-a1` and `run-a2` `container-files-env.txt`;
- `run_hookval.sh` `DEBUG`.

The hook silently truncates a forced list to WMAX+1 (`forcewin_hooks.py` line 474: `r.F = ...[: WMAX + 1]`). The
first contrast forces lists of 8,194-8,199 ids. The runner must therefore set `FORCEWIN_WMAX=8448`, the
module default.
- This is an environment change, not a hook change.
- It enlarges two hook buffers by about 8 MB in total.
- The plan check refuses any forced list longer than WMAX+1.
- WMAX=8448 has never run on GPU. Every run is gated in-session on exact forcing (section 5.5).

### 2.3 The first contrast leaves the validated envelope

| | Validated (attempt 2) | First contrast |
|---|---|---|
| Contexts | Under 5k | 63,488-120,217 |
| Windows | 2,500 tokens or fewer | 8,193-8,198 tokens (lists 8,194-8,199) |
| Block crossings (4,352) | At most 1 per window | 2 per window |
| Selection regime boundary (65,536) | Not crossed | Crossed by M3 at output index 2,048 |
| Prefix-cache hits | None (cached_tokens 0 throughout) | Every batch (section 2.4) |

The hook's validity at long context is therefore measured again in-session, not assumed:
- exact structural gates on every run;
- long-context decode-vs-prefill alignment and index-K content against prefill-written pools (section 5.5).

K validated selection model-free on both sides of 65,536 and up to the 330k equivalent.

### 2.4 Cold long prefills would break the fixed batch composition

If eight 63k-112k prompts are submitted together:
- the scheduler prefills them in chunks, roughly one after another (about 5 minutes in total);
- the first request decodes thousands of window tokens alone, or beside prefill chunks, before the last one starts.

D2 couples injected requests through the shared null tail block, so concurrency is part of the treatment.

**Fix: one warm request per round.** An unkeyed request with max_tokens 1, using the round's `cache_salt`, prefills
each prefix once. Every batch then prefills only the uncached remainder:
- P mod 4,352: 728-3,224 tokens per request, 20,816 in total, about 9 s;
- all 8 requests enter decode within the 512-position lead-in.

B1's prefix-hit path is inside its verified envelope: the D7 guard session covered 32 prefix hits of up to 113,152
cached tokens (d7/step2/run-b1-a1, C4_prefix_hits). A keyed forced request on a cached prefix is new for the hook.
It is covered by a CPU test and the in-session gates.

### 2.5 Cache-safety invariant: one salt, one prompt per fixture

vLLM caches full blocks of generated tokens too. A later, longer prompt with the same salt and the same leading
tokens would therefore read decode-written (in D runs, D2-corrupted) blocks as if they were prefill state.

Rules:
- Each salt is used with exactly one prompt per fixture.
- **Primary rounds** share a salt between C_r and D_r. That is safe, because no prompt under that salt ever extends
  past P.
- **Long-prompt (LP) rounds** have their own salts.
- **Rebuilds** in option A have their own salts.
- **Gate:** every request's `cached_tokens` must equal the expected prefix-only value exactly (`position_plan.json`,
  `warm_cached_expected`). Anything else makes the run INVALID.

### 2.6 Prior null: per-position noise is large, window-level noise is small

`prior-null/b1_null_a2.json` compares B1 against B1 on the same 8 short windows: V4-b1 against V4n-b1b, a different
batch composition. This null is therefore, if anything, larger than a same-composition repeat.

| Quantity | B1 vs B1 |
|---|---|
| ln P_pre(`</think>`) per position, abs Δ (9,687 positions) | median 0.19, p90 0.62, p99 1.66, max 5.17 nats |
| Positions with P(`</think>`) below 1e-9 in both arms | 72-94 % per window |
| Window mean of ln P floored at ln 1e-9, abs Δ | ≤ 0.06 nats |
| Window ln of mean P(`</think>`), abs Δ (8 windows) | 0.0001, 0.003, 0.028, 0.029, 0.051, 0.062, 0.127, 0.249 nats |
| Stop tokens (window ln-mean-P, abs Δ) | up to 0.22-0.28 nats, at ln P of −14 to −24 |
| Forced-token ln P, abs Δ | median 1e-4 |
| Argmax agreement | 0.974-0.990 |

Consequences for the design:
- **The window statistic is ln of the window-mean probability** (section 4). A mean of per-position logs is diluted
  by below-floor positions and driven by noise at irrelevant probabilities.
- **At window level, 3 runs per arm can resolve the 0.5-nat operational screen.** The worst prior window null is
  0.25 nats, at short context with a different composition.
- These are short windows, so the long-context null is measured in-session; the prior only informs the design.

### 2.7 Other inputs

- **Open check from 21j: resolved.** Both B1 arms showed the same pooled gross count (172 of 27,973). FOLLOWUP-a2 §1
  found a coincidence, not a duplicated comparison: the per-layer counts differ and the comparisons reproduce from
  raw bytes.
- **The served (natural) draft schedule** leaves finishing times unequal. The last positions of the slowest requests
  may run with fewer than 8 in flight. In-flight count is derived per position from the hook's step column and
  reported. A sensitivity analysis restricts to positions with all 8 in flight.
- **A8 and L8** share P, W and the doc-79 prompt, so their designated positions are identical. Doc 79 appears 3 times
  (L3, A8, L8). A question-weighted sensitivity is reported (section 4.4).

---

## 3. Design

### 3.1 Server

One server, as attempt 2:
- **Image:** `local/glm53-v0.9.1-b1-forcehook-mg2:local`, sha256:ac05230b…, pinned.
- **Hook module:** `vllm/forcewin_hooks.py`, 15e93e23…, checked in-container before traffic.
- **Recipe:** glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234+d7.
- **Serving:** DFlash2 ×3, TP2, shipped graphs, guard on (`GLM53_MAPGUARD=device`).
- **Environment:** identical to attempt 2's `DEBUG` (FORCEWIN on, D2 allowed, SHADOW 1, SNAP 1, guard) except
  `FORCEWIN_WMAX=8448`.

Both arms run on the same server with the same environment. Only the per-request `local_fw_d2` key differs.

### 3.2 Fixtures

The 8 fixtures of `first_contrast.json`, unchanged, in `batch_order` L1 L4 M3 M4 L3 L5 A8 L8:

| Group | Fixtures |
|---|---|
| Near-close | L1, L4, M3, M4, L3 (5) |
| Active | L5 exhaustion, A8 loop pre-onset (onset at 184k, after the window), L8 exhaustion (3) |

- No fixture is repeating at the window (20b).
- All are re-tokenized shared counterfactual fixtures, not replays.
- All windows end below 131,072.
- Exact prefix and window ids, and their sha256, are produced at the freeze by re-tokenizing in a CPU container, as the
  `ids` rule of first_contrast.json states.
  - The freeze asserts every convention in `position_plan.py`.
  - The forced list is the W window tokens plus the history's next token, and is `</think>` 154842 at output index W
    for the 5 near-close fixtures.

### 3.3 Requests

Every forced request carries:
- temperature 1.0 and top_p 0.95 (the served sampler, overridden by forcing);
- a distinct seed, recorded;
- `local_fw_draft = natural`;
- `local_fw_desig` = the designated set (625-677 positions; at most 1,024);
- `local_fw_selrec` = 64 selection-record positions;
- `local_fw_dumpwin = 1`.

D requests also carry `local_fw_d2 = [0, len(forced list) + 1]`, the same convention as attempt 2's V4: the whole
window is injected.

### 3.4 Primary: 3 rounds, 6 runs (run = one batch of the 8 fixtures)

| Order | Phase | Salt | Notes |
|---|---|---|---|
| 0 | V0 probes | — | Unkeyed, bitwise equal to B1's record |
| 1 | warm s1 → **C1** | s1 | Cold prefix prefill, then the batch |
| 2 | warm s2 → **C2** | s2 | Repeatability pair. Structural gate after C2 (session stops on failure) |
| 3 | re-warm s2 → **D2** | s2 | Shares C2's prefix state |
| 4 | re-warm s1 → **D1** | s1 | A re-warm is a cache hit unless evicted; `cached_tokens` is gated |
| 5 | warm s3 → **D3** | s3 | |
| 6 | re-warm s3 → **C3** | s3 | The last run: post-injection clean check (ledger clean, no carry-over) |

Properties of this order:
- **Clean and injected runs are always separate batches.**
- **Each round's C and D share an identical prefix state.** Between rounds, the prefix is rebuilt, so the within-arm
  null includes prefill variation and is conservative for the paired contrast.
- **Order positions:** C at 1, 2, 6 and D at 3, 4, 5.

### 3.5 Long-prompt / short-output control (LP): 3 rounds, 6 runs

The construction:
- **P_L = P + 7,168**, which is 0 mod 8, so the LP window also starts on a pool and ring boundary.
- **W_s = W − 7,168**, which is 1,025-1,030; the forced list is reasoning[s + 7,168 : end + 1].
- The final position and the token history are the same as the long construction.
- **Common measured positions:** output index o ∈ [7,680, W] (the LP lead-in is 512). This contains the whole
  512-position tail and 513-514 designated positions per fixture.

Each LP round's warm is a **keyed reference request**:
- prompt P_L, max_tokens 1;
- `local_fw_ref = P`, `dumpwin`, `dumprange = [P, P_L)`.

This one prefill therefore also supplies, at no extra prefill cost:
- the long-context **prefill reference** for output indices [0, 7,168) of the primary window (decode-vs-prefill
  alignment);
- **prefill-written index-K pools** for 87.5 % of the primary window (the content gate and flags).

Order, mirroring the primary: LW1 → LC1, LW2 → LC2 (repeatability), LD2, LD1, LW3 → LD3, LC3. The LP salts are
separate from the primary salts (section 2.5).

### 3.6 Instrument health without the LP control (option A)

- **RB1:** prompt P + W + 1 under its own salt, cold, with `local_fw_ref = P` and dumpwin over [P, P + W].
- **RB2:** the same build plus 200 further ids of the history (attempt 2's N_pp construction), under its own salt.

### 3.7 Optional short stratum S (the attempt-2 D2 set; coordinator decision)

S is the only way the two attempt-2 flagged fixtures enter this contrast (section 2.1).

- **Fixtures:** the 8 exact-id traces of attempt 2's `D2SET`: cbac17c70af8, be2e9e349395, 4b92a6a90ec4, ddfd287416a4,
  68a96f264430, 376073921446, 1e875e978fd4 and 6d18c40f58df.
  - Windows of 256-2,000 tokens start at the prompt (124-365 tokens), payloads exactly as attempt 2's V4.
  - Exact generated ids, so these are replays, unlike the long fixtures.
  - No warm request is needed: cold prefills are under 400 tokens.
- **Runs:**
  - SC ×3 and SD ×3, as one batch of 8 per run, clean and injected in separate batches.
  - Order SC1 SC2 SD1 SD2 SD3 SC3, run after the long primary and before the LP control.
  - Plus two prefill rebuilds, SR1 (prompt + window) and SR2 (+200 ids), for the content nulls.
- **Analysis:** S's primary aggregate uses the same statistics as the long stratum, over all 8 S fixtures, **both
  flagged fixtures included** (R1).
  - S's Q set is every 16th output index from 64 (a 64-position lead-in, as in the prior null) to the window end,
    plus the end. Designated exposure positions are the hook default: the 16-grid plus the last 64.
  - Six windows are whole histories ending at the natural close, so `</think>` is the last forced token; two are
    capped at 2,000 tokens.
  - S is reported beside the long stratum, **never pooled with it**: the context range, fixture type and
    replay status differ.
  - The leave-flagged-out sensitivity (R2), position-level flags (R3) and the known fragile set (R4) apply.
- **What S adds:**
  - an in-session, short-context replication of attempt 2's injection with three runs per arm;
  - a short-context termination contrast on exact-id histories;
  - a check that the fragility at be2e9e349395 is common to both arms (FOLLOWUP-a2 §3.3: the arms agree with each
    other at those positions).
- **Cost:** about 0.20 GPU-hours, from attempt 2's measured phase times (section 8).

---

## 4. Measures and frozen primary statistic

### 4.1 What is recorded

Everything below is already recorded by the hook. Nothing is added.

| Group | Recorded |
|---|---|
| Per output index (all positions) | lp_pre and mass_lt/le for `</think>`, the three stop tokens (154820, 154827, 154829) and the forced token; p_post and elig_lo/hi (nucleus eligibility after top_p); entropy; argmax; top1_p; argmax_is_forced; argmax_is_term; step; accepted; out_of_nucleus |
| Per designated position and indexer layer | n_sel, n_sel_win (selected decode-written pools), n_sel_win_corrupt, n_sel_win_race, n_win (decode-written pools), n_win_corrupt, n_win_race |
| Per request | Shadow-ledger class of every decode-written pool; final 511-pool selections at the 64 selrec positions; d2_injected_tokens; acceptance history; invariant counters; window page dump |
| Client | Seed, salt, cached_tokens, timings, ids_equal |
| Guard | Write mapping per rank |

### 4.2 Primary statistic

This is frozen before data.

1. **Positions.** Q_f is the designated active-reasoning set of fixture f (`position_plan.json`, `Q_primary`; 122-180
   positions):
   - the 64-grid after the 512 lead-in;
   - the non-repeating commitment events in the window;
   - the end position e (o = W).

   For near-close fixtures this includes the positions approaching the natural close and the close itself.
2. **Window.** y(a, r, f) = ln( mean over o in Q_f of exp(lp_pre_`</think>`(o)) ).
   - This is ln of the window-mean pre-mask close probability, computed with logsumexp.
   - Effects are log ratios of close propensity, in nats.
3. **Run (the unit).** Y(a, r) = mean of y(a, r, f) over f in F_primary.
   - F_primary is all 8 fixtures. No fixture is excluded (section 5.4, R1); flagged fixtures stay in and are dropped
     only in the R2 sensitivity.
   - Fixtures within a run share a batch, and D2 couples them, so they are never treated as independent samples.
4. **Arm contrast.** Δ = mean over r of Y(D, r) − mean over r of Y(C, r).
5. **Matched null.** N = the maximum over the six within-arm run pairs (C-C ×3, D-D ×3) of |Y − Y′|, measured
   in-session.

### 4.3 Termination criterion

The termination criterion ("beyond the matched null") is met if all of these hold:
- |Δ| > 2 × max(N, 0.05 nats);
- complete separation: all three Y(D) lie on one side of all three Y(C).

The 0.05-nat floor is operational, because a deterministic null can be exactly 0.

With 3 runs against 3, complete separation is the smallest attainable exact permutation level: 1/20, one-sided. It is
reported as that bound, not as significance or equivalence.

### 4.4 Operational screens

These are labelled operational. They are not statistical tests.

- **Per fixture:**
  - Δ_f is the difference of the arm means of y.
  - N_f is the per-fixture within-arm maximum.
  - **Screen:** |Δ_f| ≥ 0.5 nats, |Δ_f| > 2 × max(N_f, 0.05), and complete separation.
- **Direction:** at least 6 of 8 fixtures share the sign of Δ (in the R2 sensitivity, ⌈0.75 n⌉ of the n remaining).

### 4.5 Selected-keys criterion

"D2 changes the selected keys" requires both of the following:
1. **Exposure.**
   - In every D run, selected corrupted window pools (n_sel_win_corrupt) are greater than 0 at designated positions.
     They are reported as a share of selected pools by fixture and layer, with n_win and n_win_corrupt.
   - In C runs, n_sel_win_corrupt and n_win_corrupt must be exactly 0. That is a gate.
2. **Divergence beyond the null.**
   - The quantity is the mean Jaccard distance between final selection sets at the selrec positions, averaged over
     layers and fixtures.
   - Every one of the 9 D-C run pairs must exceed every one of the 6 within-arm pairs.
   - The mean D-C distance must exceed 2 × the largest within-arm distance.

### 4.6 Reported separately

These definitions are fixed; none of them is primary.
- The floored mean of ln P over Q_f (floor ln 1e-9), and the unfloored mean.
- The tail T_f (512 positions) and the end position e alone.
- Near-close (5) against active (3) fixtures.
- A question-weighted aggregate, with doc 79's three fixtures averaged first: 6 question units.
- Positions with all 8 in flight only.
- **`</think>` after top_p:** p_post and the eligibility share (positions where `</think>` is in the nucleus).
- **Each stop token separately** (154820, 154827, 154829): lp_pre, p_post and eligibility.
  - Reasoning close and generation stop are never combined into one score.
  - Premature stopping is not successful completion.
- Forced-token likelihood, entropy, argmax agreement (within and across arms), and argmax_is_term.
- Index-K content gross by fixture and layer, plus pooled (section 5.5).

### 4.7 Frozen readings (21g §7)

Each reading applies only if the instrument gates pass and the repeatability step passes (section 5.6).

| Selected keys (4.5) | Termination (4.3) | Reading |
|---|---|---|
| Yes | Yes | **Supported local mechanism candidate.** The direction is stated: D2 lowering P(`</think>`) is the direction of the stock-0.9.1 completion hypothesis. Not proof, and not an explanation of GPQA rates |
| Yes | No | This decode-window exposure is excluded as a cause of termination-probability change **at this resolution, on these fixtures and positions**. Accumulated full-history D2 effects are **not** excluded |
| No | Any | The injection did not reach selection beyond the null, so the contrast is uninformative about the mechanism. Check D gross and exposure first |
| No | Yes | Unexplained. No mechanism claim; investigate |

Flagged fixtures never carry a mechanism claim; every reading is repeated in the leave-flagged-out sensitivity
(section 5.4). Rejection-dependent
signals are not attributed to the served D6 mechanism without the oracle-draft bracket (21 §7).

---

## 5. Validity, nulls, flags and repeatability

### 5.1 Nulls measured in-session

| Null | Source |
|---|---|
| Termination null | C-C and D-D run pairs (section 4.2) |
| Selection null | Within-arm Jaccard (section 4.5) |
| Index-K decode null N_dd | C1 against C2 window dumps (same composition) |
| Index-K prefill null N_pp | Option B: LW1 against LW2, the same build length, so possibly exactly 0. Option A: RB1 against RB2, build +200 |

When N_pp is 0, the content limit falls to the 0.1 % operational floor.

### 5.2 Index-K content gate (21g §1 formula)

C1 and C2 decode-written pools are compared against prefill-written pools over the matched units.

- **Rule:** gross ≤ max(2 × max(N_dd, N_pp), 0.1 %).
- **Unit, matching, aggregation and exclusions:** as frozen in attempt 2.
- **D runs:** gross ≥ 30 %, also reported as a multiple of the limit.

### 5.3 Long-context alignment (H2-style)

- C1 decode is compared with the prefill reference over o ∈ [512, 7,168).
- **Requirement:** top-1 agreement ≥ 0.95.
- **Argmax shift control:** aligned ≥ 0.90 and shifted ≤ 0.20.

### 5.4 Flagged fixtures (rules R1-R5)

The full rule is in `fixtures/exclusion_rule.json`.

These rules apply the coordinator's decision on FOLLOWUP-a2. **No fixture is excluded**; the exclusion list is
empty and immutable once GPU data exists.

- **R1. Keep flagged fixtures in.** A flagged fixture stays in its stratum's primary aggregate but carries no
  mechanism claim.
  - In S, that covers be2e9e349395 and 376073921446, flagged from attempt 2.
  - In either stratum, it covers any fixture flagged in-session.
- **R2. Leave-flagged-out sensitivity.** Every statistic, screen and reading is recomputed without the flagged
  fixtures and reported beside the primary, never instead of it. The direction screen becomes ⌈0.75 n⌉ of the
  remaining n.
- **R3. Flags by position as well as by layer-unit.** A gross position is a (fixture, pool) that is gross in at least
  one DSA layer on either rank.
  - Attempt 2's two definitions are evaluated with layer-units (at least 10 gross units) and again with positions (at
    least 3 gross positions):
    - outside its own null: gross_f > max(2 × max(N_dd_f, N_pp_f), 0.1 %);
    - arm asymmetry.
  - Both counts are reported. A flag on either count marks the fixture.
  - Reason: 376073921446's flag came from one position (pool 128) counted in many deep layers.
- **R4. Known fragile set.** be2e9e349395's 13 units that are gross in all three independent decodes (a2-b1, a2-b1b
  and a1-b1) are listed in `known_fragile_be2e9e349395.json`.
  - They are reported separately, and S's content rows are given with and without them.
  - Any pool-level effect located on them is attributed first to pre-existing decode-vs-prefill numerics.
- **R5. In-session flags (after data).** The attempt-2 definitions with R3 counting are applied to the long fixtures
  (C1 against C2, and against the prefill-written pools). Flagged fixtures follow R1 and R2.
- Option A0 cannot compute in-session flags.

### 5.5 Gates per run (exact; a failing run is INVALID, reported, and excluded)

Common to every run:
- ids_equal for every request;
- every output index measured exactly once (nmeas);
- post-mask values present;
- no fw_error or guard_error;
- exposure records at every designated position;
- cached_tokens equal to the expected value;
- no request error;
- all 8 requests decoding before output index 512 of every request;
- V0 equality for the session.

C runs additionally:
- d2_injected_tokens = 0;
- shadow ledger all clean;
- n_win_corrupt = n_sel_win_corrupt = 0.

D runs additionally:
- injection exactly on the intended tokens (H5a-equivalent);
- guard pattern exact per rank (H5g-r);
- the ledger agreeing with the guard.

**Session level:**
- An arm with fewer than 3 valid runs gives an INCOMPLETE contrast. The rule is never applied to 2 runs.
- The alignment (section 5.3) and content (section 5.2) gates failing makes the session **not interpretable at long
  context**. Results are reported as instrument findings only.

### 5.6 Repeatability first (analysis order, frozen)

1. Gates.
2. Within-arm variation. If N > 0.25 nats, the contrast cannot resolve the 0.5-nat screen. Variation is reported and
   no reading is made.
3. Exposure and selection.
4. Termination.
5. The LP control.

In-session, the client stops the session if C1 or C2 fails a structural gate. That is checkable from client records
and the hook log, and costs at most about 0.77 GPU-hours when it triggers after C1.

---

## 6. Long-prompt / short-output control: what it measures

The statistic is y^L = ln-mean-P(`</think>`) over the common positions L_f (tail ∪ Q_f in [7,680, W]), computed for
both constructions in each of the clean and injected conditions: a 2 × 2 design, with 3 runs per cell and a fixed
composition of 8.

| Effect | Definition | Null |
|---|---|---|
| Construction effect (clean) | Long-construction C runs (from the primary runs, restricted to L_f) against LC | The within-cell run pairs, same rule as section 4.3 |
| Construction effect (injected) | D against LD | As above |
| Interaction | Whether the injection effect depends on construction | As above |

Recorded per cell at the common positions:

| Quantity | Long construction | Short construction |
|---|---|---|
| Decode-written pools (n_win), about | 1,900-2,050 per request and layer | 128-257 per request and layer |
| In the injected condition | n_win_corrupt and n_sel_win_corrupt | n_win_corrupt and n_sel_win_corrupt |

Also recorded: `</think>` before and after top_p with eligibility, and EOS and the other stops separately.

What it distinguishes:
- At fixed total context and the same token history, the two constructions differ in:
  - how 7,168 positions of state were built (KDA recurrent and convolution state; MLA and index-K pools written by
    decode or by prefill);
  - the speculative history;
  - the number of decode-written, and in D, corrupted, pools available to selection.
- So it measures sensitivity to history construction and to accumulated decode-written state. **It is not a pure
  test of output length.**

Readings:
- **A clean construction effect beyond the null:** termination probability at a fixed position depends on how the
  history state was built. This is reported as a construction sensitivity, not as an engine defect.
- **An injection effect larger in the long construction beyond the null:** accumulated corrupted decode-written state
  matters beyond recent corruption.
- **No effects:** the constructions are equivalent at these positions within resolution. Nothing beyond the window
  is excluded.

---

## 7. Runner, client, budget gate and refusal tests (plan; to be written at the freeze)

### 7.1 Runner: `first-contrast/runner/run_fc.sh`

Derived from `hook-validation/runner/run_hookval.sh` (attempt-2 variant).

| Kept as is | Changed |
|---|---|
| Image pin; frozen SHA256SUMS check; preflight (no glm53* container; GPUs idle) | Item `fc1` with its own `first-contrast/costs.jsonl` |
| Readiness gate; hook and guard registered on both ranks | In-container sha256 of `vllm/forcewin_hooks.py` = 15e93e23… before traffic |
| Watchdog at the wall cap; EXIT/INT/TERM trap that stops the server | `FORCEWIN_WMAX=8448` |
| Cost line to GPU release; docker cp after release; frozen analysis in a CPU container | Wall cap from the authorized total; at most 2 attempts (one invalid-start allowance) |

The preflight lists graphics clients: attempt 2's preflight showed Firefox at 578 MiB, an Electron app at 390 MiB and
gnome-shell on GPU 1. The operator closes them first.

### 7.2 Client: `first-contrast/runner/client_fc.py`

Derived from `client_hookval.py`.

Reused: `forced()` (with explicit `salt`), `dump()`, `phase()`, `batch()`, and the coverage, skip and gap records.

New phases:
- `V0`;
- `warm(r)`: unkeyed, max_tokens 1, the round salt;
- `C(r)` and `D(r)`;
- `LW(r)`: keyed reference warm;
- `LC(r)` and `LD(r)`;
- the rebuilds of section 3.6 (option A only);
- `SC(r)`, `SD(r)`, `SR1` and `SR2` (stratum S only). These use attempt 2's V4 payloads and seeds scheme with new
  seeds; payloads are identical otherwise.

Other behaviour:
- Ids are read from the frozen re-tokenized fixture file, sha-checked.
- Phases run in the priority order of sections 3.4-3.5.
- A batch never starts unless the time left covers remainder + max list ÷ 25 tok/s + dumps. Runs are never
  truncated.
- Structural stop after C1 and C2.
- A final coverage record.
- **The hook is not changed.**

### 7.3 Budget gate (refusals)

The runner refuses to start if any of these holds:
- APPROVED_GPU_H is unset, or above the ceiling of the authorized option;
- the item's spent total plus this attempt's cap exceeds the authorized total;
- the attempt count is wrong;
- the wall cap is below MIN_WALL (the central estimate of the primary-only part);
- the image id or hook hash does not match;
- SHA256SUMS differ;
- the output directory exists;
- a glm53* container is running, or a GPU is busy.

### 7.4 CPU tests before the freeze

- **Plan check:**
  - every forced list ≤ WMAX+1;
  - P and P_L are 0 mod 8;
  - P + W < 131,072;
  - designated ≤ 1,024 and selrec ≤ 64;
  - `</think>` at o = W for the near-close fixtures;
  - the frozen ids hash;
  - salts: one prompt per salt, LP and rebuild salts never shared with primary salts, warm and batch salts equal
    within a round;
  - distinct recorded seeds;
  - the d2 key present on every D request and absent from every C request;
  - identical composition and order in every batch.
- **Tampered-plan negatives**, each of which must fail:
  - a d2 key on a C request;
  - an LP salt equal to a primary salt;
  - a list truncated to 5,121;
  - a wrong expected cached value.
- **Runner refusals:** every case in section 7.3, plus a DRY_RUN pass.
- **Hook CPU tests at WMAX=8448:**
  - the existing forcing tests with an 8,199-id list;
  - `row_plan` and forcing with a cached-prefix offset (query rows starting at the cached position).
- **Analysis on synthetic runs:**
  - `good` gives a reading;
  - `bad` plants each gate failure, a C-run corruption and a cached mismatch.
- **Offline fault injection on the attempt-2 captures:** the content and flag rules stay sensitive under the
  long-window analysis.

---

## 8. Cost

`cost/cost_fc.py` produces `cost_fc.json` and `cost_fc.out`. Occupied GPU-hours are 2 GPUs × wall from server start to
release.

### 8.1 Measured inputs

| Input | Value | Source |
|---|---|---|
| Fixed overhead (start → ready, gate, stop) | 766 s; attempt 1: 746 s | 742 s to ready in attempt 2 |
| Prefill, B1 guard image, long context | 2,410 tok/s | d7/step2 warm r0/r1 |
| Hooked forced decode, natural drafts, 8 in flight | 40.1-41.6 tok/s per request for the longest attempt-2 requests; 32-35 for short ones | Contexts under 5k |
| Planning decode rate at 63-120k | 33 tok/s; pessimistic 25 | Not measured |
| Window dumps | 3 s each; pessimistic 8 | About 190 MB per rank at 8.2k, extrapolated |
| Phase gap | 6.05 s | |

### 8.2 Components (seconds, central / pessimistic)

| Component | Central | Pessimistic |
|---|---:|---:|
| Primary round warm (738,896-token cold prefill) | 313 | 417 |
| Re-warm (cache hit) | 15 | 18 |
| Primary batch (8 × up to 8,199 forced tokens + remainder + dumps) | 287 | 410 |
| LP round warm (796,240-token keyed reference prefill + 8 dumps) | 360 | 512 |
| LP batch (8 × up to 1,031 forced) | 67 | 118 |
| RB1 / RB2 cold rebuild (option A) | 364 each | 517 each |

### 8.3 Options (GPU-hours)

The ceiling is max(estimate × 1.3, pessimistic) plus one invalid-start allowance of 0.42.

| Option | Estimate | Pessimistic | Optimistic | **Ceiling** | Wall cap per session |
|---|---:|---:|---:|---:|---|
| A0: primary only (C ×3, D ×3), no long-context content check | 1.96 | 2.57 | 1.74 | 2.99 | 4,625 s |
| A: primary + instrument health (RB1, RB2) | 2.36 | 3.14 | 2.10 | **3.57** | 5,660 s |
| B: one session, primary + LP ×3 (health folded into the LP warms) | 2.82 | 3.86 | 2.50 | **4.29** | 6,952 s |
| B2: one session, primary + LP ×2 per condition | 2.53 | 3.43 | 2.24 | 3.85 | 6,177 s |
| A+S: A + short stratum S | 2.56 | 3.40 | 2.30 | 3.82 | 6,112 s |
| B+S: B + short stratum S | 3.02 | 4.11 | 2.69 | **4.54** | 7,404 s |
| C: two sessions, S1 = A, then S2 = LP ×3 + a bridge round (C + D re-run) | 4.17 | 5.59 | 3.72 | 6.02 | 5,660 + 4,410 s |

- **The LP control costs about 0.86 GPU-hours more** than the primary alone (B − A0, central). About 0.40 of that (A − A0) is
  A's instrument-health work, which B absorbs.
- **The short stratum S costs about 0.20 GPU-hours** (6 × 56 s runs plus 23 s of rebuilds, from attempt 2's measured
  phases; pessimistic ×1.3).
- **An early structural stop after C1** costs about 0.77 GPU-hours.
- **The dominant uncertainty is the hooked decode rate at long context.** At 25 tok/s instead of 33, B rises from 2.82
  to 3.86.

### 8.4 One session or two

| | One session (B) | Two sessions (C) |
|---|---|---|
| Server instance | The LP and primary constructions are compared in the same instance, with matched composition and no session effect | LP and primary sit in different instances; a bridge round is needed to estimate the session effect |
| Overhead | One startup | A second startup (about 0.43) plus the bridge (about 0.5) |
| Total estimate | 2.82 | 4.17 (+1.35 GPU-hours) |
| Decision point | The primary runs first by priority, and the structural stop covers the main early failure | A real decision point after the primary result |

The design favours B. C only buys a decision point, at about 1.35 GPU-hours.

**Recommendation:** option B at a ceiling of 4.3 GPU-hours, one session, wall cap 6,952 s. If the coordinator wants
the flagged fixtures in this contrast (section 2.1), use B+S at a ceiling of 4.6 and a wall cap of 7,404 s. If the LP control is not
wanted now, the recommendation is A at 3.6, not A0: A0 cannot compute the long-context content gate or the in-session
fixture flags.

---

## 9. Preconditions and open risks

### 9.1 Preconditions before any launch

1. **The 21j open check: resolved by FOLLOWUP-a2 §1.** The two arms' equal 172/27,973 is a coincidence: their
   per-layer counts differ (for example 26 against 23), and IDX-content and N_dd reproduce exactly from the raw bytes.
   The amendment should record this.
2. **`exclusion_rule.json` (R1-R5) and `known_fragile_be2e9e349395.json` are frozen**, and the decision on S is
   recorded.
3. **Re-tokenized ids are frozen** and every convention assert passes.
4. **The CPU tests in section 7.4 pass**, and the package SHA256SUMS is recorded in an amendment.
5. **The operator authorizes an option and a GPU-hour total.** The GPUs are idle, with browsers and Electron apps
   closed.

### 9.2 Open risks

1. **Long-context hooked decode rate is unmeasured.** It is the main cost risk; the ceiling covers the pessimistic
   case.
2. **WMAX=8448, 8.2k forced windows, two block crossings, and keyed requests on cached prefixes are new for the
   hook.** Mitigated by CPU tests and exact per-run gates. If they fail, the cost is spent with no reading.
3. **The instrument is validated only below 5k.** The long-context alignment and content gates are first
   measurements, and may make the session uninterpretable.
4. **Concurrency falls at window ends** under natural drafts. It is recorded and gets a sensitivity analysis. The
   oracle schedule would equalise it, but is not the served path.
5. **With 3 runs against 3, resolution is coarse.** The smallest permutation level is 0.05 one-sided. No
   significance or equivalence is claimed, and the screens are operational.
6. **N_pp at the same build length (option B) may be 0**, putting the content limit on its 0.1 % floor.
7. **A null result excludes only decode-window D2 exposure** on these 8 re-tokenized fixtures at positions below
   131,072. It does not exclude full-history accumulation, D6, or ranges beyond 131k.
8. **Doc 79 contributes 3 fixtures**, and A8 and L8 have identical positions. **M3 crosses the 65,536 selection
   regime** inside its window. All three are covered by sensitivities and per-fixture reporting.
9. **KDA metadata disagreement: explained by FOLLOWUP-a2.** The align-mode column change at frontier 4,351 sets
   num_accepted = 1, so the expectation was mis-specified, not a commit issue. **Limit:** the KDA conv state at a
   block boundary was not compared. The long windows cross two block boundaries each, and this design takes no
   snapshots. Conv-state fidelity at boundaries is covered only indirectly, by the long-context alignment check.
10. **Disk:** about 45-60 GB of window dumps per session in the container layer under <docker-root>, copied after GPU
    release. /data has 2.1 TB free.
11. **Stock 0.7.0, D6/ring-4, depth-matched arms and whole-history replay are out of scope.** Each needs its own
    proposal (21 §8).
