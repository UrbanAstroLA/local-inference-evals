# First long-context contrast: B1 against decode-window D2 injection (amended proposal; CPU only; NOT launched)

2026-10-14. Option B of the pre-22 proposal, amended under PLAN.md amendment 22. The pre-22 text is kept unchanged as
`PROPOSAL.md.pre22`. Every change is labelled **[A22-n]** (n = the amendment-22 item it implements). `FREEZE.md` maps
each item to its implementation and test evidence. Running this needs an amendment that records the frozen package and
an operator authorization with a GPU-hour total.

How it was prepared:
- No GPU was used. No GPU or glm53 container was started; nothing used `--gpus`; no `sudo`.
- CPU containers only, all `--network none` and `CUDA_VISIBLE_DEVICES=`: the fixture-id freeze (0.9.1 release image,
  tokenizer only), the conv-state comparison on existing attempt-2 snapshots, the hook CPU tests (attempt-2 image) and
  the synthetic analysis dry runs (B1 image).
- The hook, the image and every hook-validation file are unchanged. No D2-arm termination value was read.

## 0. What changed under amendment 22

| Item | Change | Section |
|---|---|---|
| [A22-1] | Primary statistic kept, with its limits stated. Added: end-position-excluded sensitivity. Kept: question-weighted, near-close / active, common-tail construction comparison, post-top_p and eligibility, each stop token | 1, 4.2, 4.6 |
| [A22-2] | The exact 1/20 claim is removed. The three paired differences are reported. Separation and the beyond-null rule are operational screens; no significance is claimed | 4.3 |
| [A22-3] | Prefill null at a different build length: LW2, the planned LP round-2 reference warm, is adapted to build length P + 8,192 (cache safety shown). Matching by fixture × layer × shared causal position; exact cache accounting; N_pp = 0 wording corrected; the failure reading is stated before data | 3.6, 5.1-5.2 |
| [A22-4] | Validation inside the session BEFORE any injected batch. A CPU gate step (GATE-1) runs between batches on records already copied out; the reference warms run first; the session stops before any D batch on failure. The guard plus a prefix sentinel establish a clean shared prefix after injected traffic | 3.4, 3.8, 5.3, 5.6 |
| [A22-5] | Convolution-state boundary control (CV): existing snapshot capability, no hook change, against a matched decode reference (poison). Validated on attempt-2 data | 2.8, 3.7 |
| [A22-6] | LP control kept and named a history-construction and accumulated-decode-state comparison; selected corrupted pools reported by fixture × layer | 1, 4.5, 6 |
| [A22-7] | Short stratum S removed | 2.1, 3 |
| [A22-8] | Readings exactly as amendment 22 s.8 | 4.7 |
| [A22-9] | Frozen package, stop on failure, no automatic additional attempt: `runner/run_fc.sh` (single attempt), `runner/client_fc.py`, `runner/gate_step.sh`, `runner/plan_check.py`; refusal and tampered-plan tests; hook CPU tests at WMAX 8,448 and on cached prefixes | 7 |
| [A22-9] | Repriced from attempt-2 measured times (any added reference or boundary check repriced; the invalid-start allowance not consumed silently); decode-speed assumption and its sensitivity stated | 8 |

---

## 1. What is being asked

1. **Primary.** At fixed long-context histories, does D2 injected only during the forced decode window change the
   selected keys and the reasoning-close (`</think>`) probabilities beyond the matched in-session null?
   - Arms: clean B1 ×3 against B1 + D2 window injection ×3.
   - Same 8 fixtures, 8 requests in flight, clean and injected conditions in separate batches.
2. **Long-prompt / short-window control (21g §6).** At the same token history and final context position, does the
   result depend on how the history state was built? **[A22-6]** This is a
   **history-construction and accumulated-decode-state comparison**, not an output-length experiment.

**[A22-1] What the primary statistic measures.** Close propensity on this counterfactual fixture panel (re-tokenized
histories, prespecified positions, pre-top_p). It is not a finishing hazard and not a GPQA completion probability. A
pre-mask change that stays outside the nucleus is a computational finding whose relevance to sampled termination is
unestablished.

No probability difference is claimed to explain GPQA completion rates. No full GPQA pass follows from instrument
validity or from a local logit effect (22 §9).

---

## 2. Findings that shape the design

### 2.1 Flagged attempt-2 fixtures and stratum S [A22-7]

`be2e9e349395` and `376073921446` (FOLLOWUP-a2: numerical fragility, not instrument defects) are exact-id short traces
of attempt 2's D2 set. They could enter only through the optional short stratum S, which amendment 22 omits. A labelled
exploratory short-context comparison may later use attempt-2 records, after the first-contrast definitions are frozen.
`fixtures/exclusion_rule.json` is amended accordingly (pre-22 copy kept): R1-R3 and R5 apply to in-session flags on the
long fixtures; R4 has nothing to act on.

### 2.2 WMAX

Both validated sessions ran `FORCEWIN_WMAX=5120`; the hook silently truncates a forced list to WMAX + 1. The
runner sets 8,448, the module default (an environment change, not a hook change).
- **[A22-9]** The plan check refuses any list longer than 8,449 or shorter than planned; a CPU test shows the
  truncation that would otherwise happen (W3).
- **[A22-4]** GATE-1 requires `wmax` 8,448 in the hook's init record on both ranks and the full planned list in every
  keyed request's hook record (G1-wmax); the runner also checks it in the running container before any traffic.

### 2.3 The contrast leaves the validated envelope

| | Validated (attempt 2) | First contrast |
|---|---|---|
| Contexts | Under 5k | 63,488-120,217 |
| Windows | 2,500 tokens or fewer | 8,193-8,198 tokens (lists 8,194-8,199) |
| Block crossings (4,352) | At most 1 per window | 2 per window |
| Selection regime boundary (65,536) | Not crossed | Crossed by M3 at output index 2,048 |
| Prefix-cache hits | None | Every batch |

**[A22-4]** Each of these is now a gated check in-session, before any injected batch (section 5.3).

### 2.4 Cold long prefills would break the fixed batch composition

Unchanged: one warm request per round prefills each prefix once, so every batch prefills only the remainder
(P mod 4,352: 728-3,224 tokens per request) and all 8 requests enter decode within the 512-position lead-in.

### 2.5 Cache-safety invariant [A22-3, restated]

vLLM caches full blocks, decode-written ones included. A later prompt with the same salt and leading tokens would read
decode-written (in D runs, corrupted) blocks as if they were prefill state. The pre-22 rule "one prompt per salt per
fixture" is replaced by the exact condition it protected:

> Under one salt, a request issued after a decode window started under that salt may only cover full blocks that end
> at or before that window's start: floor(prompt length / 4,352) × 4,352 ≤ every earlier decode start under the salt.

- Primary salts s1-s3 carry prompts of length P (C, D, PS) and P' = B1 − 63 (CV, below its first crossing B1).
- LP salts l1-l3 carry P_L. LW2's longer prompt P + 8,192 is issued under l2 **before any decode under l2**, and every
  later l2 prompt (P_L) covers only blocks LW2 itself prefilled.
- Salts are partitioned by role and never shared; **exact cache accounting** is gated: every reference, run, CV and
  sentinel request's `cached_tokens` equals its planned value (0 cold; c0 = floor(P/4,352) × 4,352 or c0_L on a cached
  prefix). Warm and re-warm values are recorded and classified (cold, hit, or miss / partial).
- `runner/plan_check.py` rule P6 enforces the invariant over the whole ordered plan; tampered-plan test T11 shows it
  firing.

### 2.6 Prior null

Unchanged (`prior-null/b1_null_a2.json`): per-position noise is large, window-level noise small (window ln-mean-P
|Δ| ≤ 0.25 nats on 8 short windows). It informs the design only.

### 2.7 Other inputs

Unchanged: the 21j open check is closed (FOLLOWUP-a2 §1). A8 and L8 share P, W and the doc-79 prompt; doc 79 appears 3
times (question-weighted sensitivity).

### 2.8 Conv state at a block crossing on existing attempt-2 data [A22-5]

`conv-a2/conv_a2.py` → `conv_a2.json`, `.out` (post-inspection, CPU, attempt-2 verdict unchanged), over the 53
attempt-2 V3 snapshot / prefill-reference pairs × 2 ranks (two requests cross 4,352):
- **Mapping.** Speculative snapshot after a step that accepted a drafts: conv-state rows e = 0..5 hold the inputs of
  positions p0 − 2 + e (p0 = frontier − a); rows 0..a + 2 are committed. Prefill reference: rows 0..2 hold the last three
  prompt positions. The mapped offset is the best layer-0 match in 106 of 106 pairs (layer-0 relative L2 0.003 at the
  mapped rows; 0.8 or more elsewhere).
- **Conv rows against the prefill reference**, the 21g-2 identification screen: 100 % in layers 0-17, 99.7 % deeper
  (18 deep-layer disagreements at single positions, wrong row nearer by at most 1/0.84), layers 0-2 relative L2
  ≤ 0.009 with margin ≥ 101. Screen passed. At the block crossing alone: 100 % in all layers.
- **Slots.** Every KDA layer's speculative slots equal the running column's block-id run of one mamba group (0
  violations); the conv row recorded is slot 0, the engine's `conv_state_indices`.
- **Persistence.** Conv rows of shared committed positions are bitwise identical between consecutive snapshots
  (408 / 408), including across the crossing step. Attempt 2 took no snapshot before the crossing, so the migration
  itself (old column → new column) was **not** observed: that is what the CV control adds.
- **KDA recurrent state** (reproduced): identification 100 %, early relative L2 ≤ 0.009.

---

## 3. Design

### 3.1 Server

Unchanged: image `local/glm53-v0.9.1-b1-forcehook-mg2:local` (sha256:ac05230b…), hook module
`vllm/forcewin_hooks.py` 15e93e23…, recipe glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234+d7, DFlash2 ×3, TP2,
shipped graphs, guard on. Environment = attempt 2's `DEBUG` with `FORCEWIN_WMAX=8448`.

### 3.2 Fixtures (ids frozen)

The 8 fixtures of `first_contrast.json`, unchanged, in `batch_order` L1 L4 M3 M4 L3 L5 A8 L8 (near-close: L1 L4 M3 M4
L3; active: L5, A8, L8). The exact ids are now frozen (`fixtures/freeze_ids.py` → `fixtures/ids/*.json.gz`,
`fixtures/ids_manifest.json`; 128 of 128 convention checks pass):
- re-tokenized in a CPU container of the 0.9.1 release image with the K3.25 tokenizer, as the fixture inventory did;
- every reasoning-id sha256 equals `history_sha256`; every rendered prompt equals the inventory's rendered prompt hash;
- the forced list is the W window ids plus the history's next token: `</think>` 154842 at o = W for the 5 near-close
  fixtures; no `</think>` or stop id inside any window;
- also hashed: the LP prompt (P_L), the LP list and LW2's prompt (P + 8,192 ≤ P + W for every fixture).

### 3.3 Requests

Unchanged: temperature 1.0, top_p 0.95, a distinct recorded seed (now deterministic per phase, fixture and schedule
from a fixed list), `local_fw_draft = natural`, designated set (625-677 positions), 64 selection-record positions,
`dumpwin`. D requests also carry `local_fw_d2 = [0, len(list) + 1]`.

### 3.4 Session order [A22-4: validation before the injected batches]

| Stage | Phase | Salt | Purpose |
|---|---|---|---|
| A | V0 | — | Unkeyed probes, bitwise equal to B1's record |
| A | **LW1** | l1 | Prefill reference, cold, build length P_L = P + 7,168; dumps [P, P_L] (moved first) |
| A | **LW2** | l2 | Prefill reference, cold, build length P + 8,192 **[A22-3]**; same dump range (moved first) |
| A | warm s1 → **C1** | s1 | Clean run 1; client structural stop on failure |
| A | warm s2 → **C2** | s2 | Clean run 2; client structural stop on failure |
| A | **CV** | s1 | Conv-state boundary control **[A22-5]**, clean |
| — | **GATE-1** | — | `runner/gate_step.sh`: copy stage A's records out of the container, run `analysis/gate_fc.py` in a CPU container, in the background |
| L | re-warm l1 → **LC1**; re-warm l2 → **LC2** | l1, l2 | Clean LP runs, overlapping the gate computation |
| — | wait for GATE-1 | — | **No injected batch runs unless GATE-1 = PASS**; otherwise the session stops (client rc 10) |
| B | re-warm s2 → **D2**; re-warm s1 → **D1** | s2, s1 | Injected; share C2's / C1's prefix |
| B | re-warm l2 → **LD2**; re-warm l1 → **LD1** | l2, l1 | Injected LP; run only if the remaining primary phases still fit afterwards |
| B | warm s3 → **PS-a**, **PS-b** → **D3** → **PS-c** → re-warm s3 → **C3** | s3 | Prefix sentinel before / after injected traffic **[A22-4]**; C3 = residue check |
| B | **LW3** → **LD3** → re-warm l3 → **LC3** | l3 | LP round 3 |

Properties:
- Clean and injected runs are always separate batches. C at primary positions 1, 2, 6; D at 3, 4, 5 (as pre-22).
- C_r and D_r share a salt and therefore a prefix state (a re-warm hit, recorded); between rounds the prefix is rebuilt.
- The order keeps every needed prefix cached: before LD1 the resident sets (l1, l2, s1, s2 and the decode blocks of 7
  runs) total about 3.4 M of the 4.7 M-token KV cache; salts evicted later are no longer needed. A miss is still only
  a recomputation: re-warms rebuild, and `cached_tokens` stays gated on every batch.

### 3.5 LP control

Unchanged construction: P_L = P + 7,168, W_s = W − 7,168 (lists 1,026-1,031), common positions o ∈ [7,680, W]. LW1
and LW3 are the round warms of l1 and l3; LW2 (round 2) is adapted (section 3.6). Designated and selection-record sets of
LP requests are the primary sets at o ≥ 7,168, re-based.

### 3.6 Prefill references and the different-build-length prefill null [A22-3]

- **LW1** (P + 7,168) and **LW2** (P + 8,192) are keyed references (`local_fw_ref = P`, every prompt row from P on
  measured once) under their own salts, both cold, both dumping the prefill-written pages over the identical range
  [P, P_L].
- **The prefill null N_pp = LW1 vs LW2** on the shared causal positions [P, P_L) (attention is causal: tokens after
  P_L change only the build length, the chunk layout and the batch shape of the shared positions' computation).
- **Adapted, not added.** LW2 is the planned round-2 LP warm. Its cache safety is shown in 2.5: it is issued before any
  decode under l2, LC2/LD2's prompt P_L then reads only blocks LW2 prefilled, and their expected `cached_tokens` is
  unchanged (c0_L). Cost: 8 × 1,024 extra prefill tokens (about 3 s), no added rebuild.
- **Coverage gained.** LW2's reference rows reach o = 8,192, so the alignment checks cover M4's second block crossing
  (o = 7,976), which LW1 does not reach.
- **The decode null N_dd = C1 vs C2** (same composition, different rounds).
- **Matching** (21g-1): a tested unit (fixture, rank, DSA layer, pool) is evaluated only where both nulls exist for
  that same unit; the limit is computed from the nulls restricted to exactly the evaluated units; rates pooled over
  units; coverage ≥ 99 % and every fixture, else the row fails.

### 3.7 Convolution-state boundary control (CV) [A22-5]

Existing snapshot capability (`local_fw_snap`), no hook change:
- **Fixtures** L5 and L1. Prompt P' = B1 − 63 under s1 (its first crossing B1; the cached prefix c0 as in C1; the
  4,289-token remainder prefilled). Forced list: the next 72 history ids. One batch of 8, clean.
- **Schedules** poison (single-token advance, the **matched decode reference**), oracle, reject:2 and natural.
  P' ≡ 1 mod 4, so oracle (4 per step) and reject:2 (2 per step) commit counts land exactly on frontiers
  B1 − 8, B1 − 4, (B1 − 2), B1, B1 + 4, where poison also snapshots: identical committed counts F across schedules.
- **Snapshots** at frontiers B1 − 8 … B1 + 4: poison also at B1 − 6 … B1 − 1, consecutive, so a pair straddles the
  align-mode column change (the column switches when the step's furthest query position reaches B1).
- **Gated checks** (5.3): the actual conv-state slot (slots = the running column's block-id run; conv row = slot 0 =
  `conv_state_indices`); bookkeeping (num_accepted = hook accepted + 1; snapshots exactly at their targets); the crossing
  exercised (both columns per request); **migration exact** (conv rows of shared committed positions bitwise identical
  across consecutive snapshots, including a straddling pair for poison, oracle and reject:2); conv rows and KDA
  recurrent state identified against poison with the 21g-2 screen.
- **Limitation.** No prefill reference is taken at the CV frontiers (each would be a cold ~110k prefill, about 50 s);
  decode-vs-prefill of the conv state at a crossing is covered by the attempt-2 result (2.8), and the long-context
  alignment checks cover it indirectly. No hook change was needed, so none is costed.

### 3.8 Prefix sentinel (PS) [A22-4]

The guard verifies index-K and tail-ring writes at the mapping level (H5g): every mismatched tail write lands in the
null tail block, and decode, seed, prefill and position mismatches are zero. That shows the injection cannot write
index-K or tail data into a shared cached block. It does **not** monitor MLA or KDA writes, so the protection is
incomplete for the committed recurrent state. Sampled before / after state checks with the existing snapshot capability
close that gap:
- **PS-a, PS-b** (before D3) and **PS-c** (after D3) on s3's cached prefix, one request at a time (a fixed batch
  shape): prompt P, one forced token, a committed-state snapshot after the prefill step (KDA conv rows and recurrent
  sketch at P − 1, rebuilt from the cached boundary state), and a dump of the last cached shared block's pages
  [c0 − 1,024, c0).
- **Exact:** MLA, index-K and attention pages of that block are byte-identical in PS-a, PS-b and PS-c.
- **KDA:** PS-c equals PS-a bitwise when PS-a equals PS-b (the in-session determinism null); otherwise within
  2 × the PS-a / PS-b difference.

### Removed

- The option-A rebuilds RB1 and RB2: the references are folded into LW1-LW3, and the null into LW2.
- Stratum S [A22-7].

---

## 4. Measures and the frozen statistic

### 4.1 What is recorded

Unchanged (hook columns; nothing added).

### 4.2 Primary statistic [A22-1: kept, with explicit limits]

1. Q_f: the designated active-reasoning set of fixture f (`position_plan.json`; 122-180 positions: the 64-grid after
   the 512 lead-in, the non-repeating commitment events, and the end position e = W).
2. y(a, r, f) = ln( mean over o in Q_f of P_pre(`</think>` | o) ) (logsumexp).
3. **Fixed-fixture run aggregate** Y(a, r) = mean over the 8 fixtures. Fixtures share a batch and D2 couples them, so
   they are never treated as independent samples.
4. Δ = mean_r Y(D, r) − mean_r Y(C, r).
5. N = the maximum over the six within-arm run pairs (C-C ×3, D-D ×3) of |Y − Y′|.
6. **Limits, stated wherever the statistic appears:** see section 1. Between rounds the prefix is rebuilt, so N
   includes prefill variation; with 3 runs per arm in one session with a fixed order, the resolution is coarse.

### 4.3 Termination screens [A22-2]

- **Paired differences reported:** d_r = Y(D_r) − Y(C_r) for r = 1, 2, 3 (C_r and D_r share a prefix construction and
  a salt), with whether all three share a sign.
- **Beyond-null screen:** |Δ| > 2 × max(N, 0.05 nats).
- **Separation screen:** all three Y(D) lie on one side of all three Y(C).
- Both are **operational screens**. No exact permutation level or significance is claimed: the rounds are paired and
  the order is fixed within one session, so the 3-vs-3 permutation set is not the design's randomization. (The pre-22
  "1/20" statement is withdrawn.)
- The termination criterion is met when both screens hold.

### 4.4 Per-fixture and direction screens

Unchanged and operational: per fixture |Δ_f| ≥ 0.5, |Δ_f| > 2 × max(N_f, 0.05) and separation; direction: at least
6 of 8 fixtures share the sign of Δ (⌈0.75 n⌉ of n in the R2 sensitivity).

### 4.5 Selected keys

Unchanged definition; the outcome "insufficient selected-key exposure" covers either part failing:
1. **Exposure:** in every D run, selected corrupted window pools (n_sel_win_corrupt) > 0 at designated positions; in C
   runs exactly 0 (gated).
2. **Divergence beyond the null:** every one of the 9 D-C Jaccard distances between final selection sets (selection-
   record positions × layers × ranks × fixtures) exceeds every one of the 6 within-arm distances, and the mean D-C
   distance exceeds 2 × the largest within-arm one.
- **[A22-6] Reported quantitatively by run × fixture × layer:** n_sel, n_sel_win, n_sel_win_corrupt, n_win,
  n_win_corrupt and the selected-corrupted share, for C, D, LC and LD runs.

### 4.6 Reported separately (fixed definitions; none is primary) [A22-1]

Each as a full contrast (Δ, paired d_r, N, both screens):
- **end-position-excluded** (Q_f without o = W) — added;
- **question-weighted** (doc 79's three fixtures averaged first: 6 question units);
- **near-close** (5) and **active** (3) groups;
- the tail T_f (512 positions) and the end position alone;
- positions measured in steps with all 8 requests in flight and no padding (each run's own subset of Q_f);
- floored (ln 1e-9) and unfloored mean ln P;
- **`</think>` after top_p**: ln mean p_post, and the eligibility shares (definitely / possibly in the nucleus);
- **each stop token separately** (154820, 154827, 154829): pre-top_p, post-top_p and eligibility. Reasoning close and
  generation stop are never combined; premature stopping is not successful completion;
- the R2 leave-flagged-out contrast and direction screen;
- the **common-tail construction comparison** (section 6).

### 4.7 Readings [A22-8, exactly]

Only if every gate holds (section 5), each arm has 3 valid runs (else INCOMPLETE; never applied to 2 runs) and
N ≤ 0.25 nats (else no reading; the variation is reported).

| Outcome | Reading |
|---|---|
| Selected keys and close probabilities both change beyond the null | supported local computational mechanism candidate; behavioural relevance assessed via post-top_p and eligibility; no GPQA explanation |
| Selected keys change, close probabilities do not resolvably change | no termination-propensity effect detected under this exposure and design; the mechanism is not excluded |
| Insufficient selected-key exposure | uninformative |

**Direction:**
- If D2 increases close propensity, that is consistent with more corrupted old-engine computation helping
  termination.
- If it decreases it, that supports a harmful D2 effect; stock 0.7.0 finishing more often would then need an
  interaction or another contributor.

"No effect detected by the prespecified screen" replaces "equivalent within resolution". Flagged fixtures never carry a
mechanism claim; every reading is repeated in the R2 sensitivity. Rejection-dependent signals are not attributed to the
served D6 mechanism without the oracle-draft bracket.

---

## 5. Validity, nulls, flags and gates

### 5.1 Nulls measured in-session

| Null | Source |
|---|---|
| Termination | C-C and D-D run pairs |
| Selection | Within-arm Jaccard |
| Index-K decode null N_dd | C1 vs C2 window dumps (same composition) |
| Index-K prefill null N_pp **[A22-3]** | LW1 vs LW2: build lengths P + 7,168 and P + 8,192, shared causal positions [P, P_L) |

**[A22-3] Correction.** The pre-22 text said "when N_pp is 0, the content limit falls to the 0.1 % floor". That is
wrong when N_dd is nonzero: L = max(2 × max(N_dd, N_pp), 0.1 %), so N_pp = 0 gives L = max(2 × N_dd, 0.1 %). The floor
applies only when both nulls are below 0.05 %. Unit test U1 checks it.

### 5.2 Index-K content gate (21g formula)

C1 and C2 decode-written pools vs LW1's prefill-written pools over the matched units (pools wholly inside [P, P_L)).
- **Rule:** gross ≤ max(2 × max(N_dd, N_pp), 0.1 %), pooled over units, coverage ≥ 99 % and every fixture.
- **Beside it (reported):** the original criterion (gross ≤ 0.1 %), the same rule against LW2, gross per fixture ×
  layer with both nulls, crossing-neighbourhood content, and fixture flags by layer-unit and position (R3).

### 5.3 GATE-1: the expanded envelope, validated before any injected batch [A22-4, A22-5]

Computed by the runner's CPU gate step from stage A's records copied out of the container (`analysis/gate_fc.py`). All
gated rows must pass, else no D batch runs.

| Row | Kind | Check |
|---|---|---|
| G1-V0 | structural | Unkeyed probes bitwise equal |
| G1-coverage | structural | Every planned stage-A request present, without error, with its hook record |
| G1-cache | structural | Exact cache accounting (2.5) |
| G1-wmax | structural | **WMAX 8,448 without truncation**: init on both ranks; every keyed request's hook record holds its full list |
| G1-abspos | structural | **Absolute positions on cached prefixes**: hook prompt length = planned P on both ranks; dump ranges and snapshot frontiers exact |
| G1-forcing | structural | ids_equal; every output index measured exactly once; every reference row once |
| G1-errors | structural | No fw_error / guard_error |
| G1-attrib | structural | **Exact attribution in flattened and padded steps**: step-matched exposure records in every tracked layer at every designated position, counted separately in FULL-graph-padded steps; selection records step-matched |
| G1-inflight | structural | All 8 decoding before output index 512 |
| G1-clean | structural | C runs: no injection; ledger clean; no corrupt pool present or selected |
| G1-idx-struct | structural | Dumps attributed; population ≥ 99 %; no page > 25 % gross; zero gross in DSA layers 3-15 |
| G1-align | numerical | C1 vs LW1 and C2 vs LW2 on o ≥ 512: pooled top-1 ≥ 0.95, \|Δlp\| quantiles within max(2 × carried null, floors), per request aligned ≥ 0.90 and ±1 shifted ≤ 0.20 |
| G1-cross | numerical | **Both block crossings** of every window: ±128 output indices aligned ≥ 0.90, shifted ≤ 0.20; every crossing covered (M4's second via LW2) |
| G1-regime | numerical | **M3's 65,536 selection-regime crossing**: ±256 indices aligned, and every recorded final selection there holds 511 distinct causal pool ids |
| G1-content | numerical | 5.2 |
| G1-CV-slot, -book, -cross, -persist | structural | 3.7 |
| G1-CV-conv, G1-CV-ssm | numerical | 3.7, against poison |

**Pre-stated reading [A22-3, A22-4].** A failing numerical row means the long-context instrument is unvalidated under these
bounds. B1 is not thereby defective, and no threshold is revised after viewing data. A failing structural row makes the
run invalid. Either way no injected batch runs, and the spent time is reported (about 1.6 GPU-hours, section 8).

### 5.4 Flagged fixtures

R1-R3 and R5 of `fixtures/exclusion_rule.json` (amended [A22-7]): no fixture is excluded; in-session flags (GATE-1:
C1 and C2 against LW1, by layer-unit and by position, plus C1/C2 arm asymmetry) mark fixtures that carry no mechanism
claim and are dropped in the R2 sensitivity.

### 5.5 Per-run gates (exact; a failing run is INVALID, reported and excluded)

Every run: coverage, exact cache accounting, WMAX and absolute positions, forcing, no errors, attribution, in flight,
both ranks' termination rows identical. C and LC runs additionally: no injection, ledger clean, no corrupt pool. An arm
with fewer than 3 valid runs gives an INCOMPLETE contrast.

### 5.6 After injected traffic [A22-4]

| Row | Check |
|---|---|
| GATE1 | Passed in-session before any injected batch and recomputes to PASS on the full data |
| H5a-fc | Every D / LD request injected at least its list's tokens; nothing else injected |
| H5g-fc | Guard per rank: tail mism == null == owned layers × injected tokens, attributed only to D / LD phases; zero decode, seed, prefill, D7 and position mismatches (the injection writes nothing into a shared cached prefix at mapping level) |
| H5f-fc | Ledger agrees with the bytes: D-run clean-labelled pools within the GATE-1 content limit (corrupt-labelled gross reported; D gross share reported against the 30 % mark) |
| PS-prefix | Section 3.8 |
| C3-residue | C3 valid and clean, aligned with LW1 / LW2 and within the GATE-1 content rule |

Any failure: no reading; results are instrument findings only.

### 5.7 Analysis order (frozen)

Gates → within-arm variation → exposure and selection → termination → LP construction comparison → readings.

---

## 6. Long-prompt / short-window control: a history-construction and accumulated-decode-state comparison

Unchanged definitions: y^L over the common positions L_f (tail ∪ Q_f in [7,680, W]) for both constructions, clean and
injected, 3 runs per cell. Reported contrasts, each with the section 4.3 screens:
- the construction effect (clean): C vs LC;
- the construction effect (injected): D vs LD;
- the injection effect in each construction;
- the interaction (D − C) − (LD − LC), against 2 × (2 × max(N, 0.05)).

Also reported per cell: decode-written and corrupted pool counts, selected corrupted pools by fixture × layer,
`</think>` before and after top_p with eligibility, and each stop separately.

Readings:
- A clean construction effect beyond the null: termination probability at a fixed position depends on how the
  history state was built. This is a construction sensitivity, not an engine defect.
- An injection effect larger in the long construction beyond the null: accumulated corrupted decode-written state
  matters beyond recent corruption.
- Otherwise: no construction effect detected by the prespecified screen. Nothing beyond the window is excluded.

---

## 7. Runner, client, gate and tests [A22-4, A22-9] (implemented; FREEZE.md lists the evidence)

| File | Role |
|---|---|
| `runner/run_fc.sh` | From the validated runner: budget gate (4.29), single attempt, image and hook pins (in-container hook sha and WMAX checked before traffic), frozen SHA256SUMS, disk check (≥ 200 GB on /data and the docker root), pre-flight that reports desktop graphics clients, watchdog, cleanup and cost line on every exit path, no `set -e` |
| `runner/client_fc.py` | The session order of 3.4; never truncates a batch (skips and records it when the time left cannot cover it, keeping the primary phases' reserve); structural stops after C1 and C2; GATE-1 wait |
| `runner/gate_step.sh` | GATE-1: copies stage-A records out (docker exec + tar), runs gate_fc.py in a CPU container |
| `runner/plan_check.py` | Rules P1-P13 on the client's plan (DRY_RUN included) |
| `analysis/fc_geometry.py`, `fc_common.py`, `gate_fc.py`, `analyze_fc.py` | One geometry for client, plan check and analyses; frozen helpers; GATE-1; the final analysis |

Tests (results in FREEZE.md):
- runner refusals with shims;
- tampered-plan negatives;
- hook CPU tests at WMAX 8,448 and on cached prefixes;
- analysis unit checks;
- synthetic dry runs (good, null, uninformative, noisy and one plant per gate);
- final DRY_RUN.

---

## 8. Cost [A22-9]

`cost/cost_fc.py` → `cost_fc.json`, `.out` (pre-22 model kept as `.pre22`). Occupied GPU-hours = 2 GPUs × wall from
server start to release.

**Measured inputs** (attempt 2 unless stated):

| Input | Value |
|---|---|
| Overhead (start → ready, gate, stop) | 766 s |
| V0 | 6.3 s |
| Phase gap | 6.05 s |
| Prefill at long context | 2,410 tok/s (d7) |
| Hooked forced decode, natural drafts, 8 in flight, below 5k | 41.6 tok/s per request |
| Poison | 11.6 steps/s with snapshots |
| Snapshot | ≤ 0.5 s |
| Window dumps | 110-150 MB per rank (dump time not measured: 3 s central, 8 s pessimistic) |

**Assumption (unmeasured):** hooked decode at 63-120k = 33 tok/s per request (0.8 × 41.6), 25 pessimistic, 40
optimistic.

**Session** (seconds, central / pessimistic):

| Part | Central | Pessimistic |
|---|---:|---:|
| Overhead + V0 | 778 | 812 |
| Stage A references (LW1, LW2) | 724 | 1,029 |
| Stage A primary (warms, C1, C2) | 1,200 | 1,652 |
| CV | 43 | 74 |
| Stage L (LC1, LC2) | 156 | 263 |
| GATE-1 wait (overlapped with stage L) | 0 | 200 |
| Stage B primary (D2, D1, s3 round, C3) | 1,505 | 2,108 |
| Stage B LP (LD2, LD1, LW3, LD3, LC3) | 661 | 1,026 |
| Prefix sentinel (3 sets) | 71 | 114 |
| One LP prefix evicted and rebuilt | 0 | 435 |
| **Total** | **5,138 s = 2.86 GPU-h** | **7,713 s = 4.29 GPU-h** |

- Optimistic: 2.52 GPU-h.
- The amendment-22 additions cost +0.035 GPU-hours centrally over pre-22 option B with the same inputs (2.82): LW2's
  extra 8k tokens, CV and the sentinel, partly offset by one fewer re-warm per round.
- **Ceiling.** The single-attempt ceiling max(2.86 × 1.3, 4.29) = **4.29 GPU-hours**, equal to the amendment-22
  target, with wall cap 7,714 s (4.29 × 3,600 / 2 = 7,722 s). Under the pre-22 formula, which adds the 0.42 GPU-hour
  invalid-start allowance, the ceiling would be **4.71, exceeding the target by 0.42**. With a single-attempt runner
  that allowance cannot be spent on a second start, so it is not carried; this is reported rather than absorbed
  (amendment 22: the allowance is never consumed silently). If the operator wants the allowance as a reserve, the total
  is 4.71.
- **Decode-speed sensitivity** (other inputs central): 20 tok/s 3.46, 25 tok/s 3.15, 33 tok/s 2.86, 41.6 tok/s 2.66
  GPU-hours, all within the wall cap. The client never truncates a run; under a binding cap it skips LP stage-B batches
  first (the primary phases' time is reserved).
- **Early stops:** structural stop after C1 about 1.17 GPU-hours; stop at GATE-1 (no injected batch) about 1.61.
- **Runner:** APPROVED_GPU_H ≤ 4.29; wall cap min(7,714 s, approved × 1,800); refused below MIN_WALL 4,477 s (the
  central estimate of the gates plus the whole primary contrast).

---

## 9. Preconditions and open risks

### 9.1 Preconditions before any launch

1. The package SHA256SUMS is recorded in an amendment; the CPU tests pass (FREEZE.md).
2. The operator authorizes the item fc1 with a GPU-hour total (4.29 single attempt, or 4.71 if the allowance is
   wanted as a reserve; the runner as frozen refuses above 4.29).
3. GPUs idle; desktop graphics clients on GPU 1 (Firefox, an Electron app, gnome-shell in attempt 2) are reported by
   the pre-flight and best closed.

### 9.2 Open risks

1. **The hooked decode rate at long context is unmeasured** (the main cost risk; covered by the ceiling down to about
   20 tok/s with central other inputs).
2. **New for the hook:** WMAX 8,448, 8.2k windows, two crossings, cached prefixes. Covered by CPU tests and GATE-1; a
   failure stops the session before any injected batch.
3. **The long-context numerical gates are first measurements**, so the session may end with "unvalidated under these
   bounds" (about 1.6 GPU-hours spent).
4. **Concurrency falls at window ends** under natural drafts (padded steps). Attribution is gated in padded steps
   separately, and the all-8-in-flight sensitivity (pre-22, kept) restricts each run to its positions measured in steps
   with 8 requests and no padding.
5. **Coarse resolution** with 3 runs per arm in one session; the screens are operational only.
6. **A null result** excludes only decode-window D2 exposure on these 8 re-tokenized fixtures below 131,072. It does
   not exclude full-history accumulation, D6, or ranges beyond 131k.
7. **Doc 79 contributes 3 fixtures;** A8 and L8 share positions; M3 crosses 65,536 (gated).
8. **Conv state at a crossing** is checked only against a decode reference in-session (prefill comparison from attempt-2
   data at 4,352).
9. **Disk:** about 45-60 GB of window dumps plus about 20 GB of CV snapshots in the container layer (<docker-root>),
   about 35 GB copied by GATE-1 and everything again after release; the runner requires ≥ 200 GB free (2.1 TB free
   today).
10. Stock 0.7.0, D6 / ring-4, depth-matched arms, whole-history replay and stratum S are out of scope.
