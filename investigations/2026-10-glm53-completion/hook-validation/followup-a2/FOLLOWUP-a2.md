# Hook validation, attempt 2: follow-up on three open items (CPU only)

2026-10-14. Follow-up to PLAN.md amendment 21j. The attempt-2 verdict (revised VALID, original INVALID) is frozen and
unchanged. Everything below is **post-inspection**. It is reported beside the frozen verdict and changes no gate.

How it was done:
- **No GPU work.** No GPU or glm53 serving container was started, and nothing used `--gpus`. Scripts ran in the B1
  image as a CPU container with `--network none` and `CUDA_VISIBLE_DEVICES=` (`frun.sh`), or as plain Python on the
  host.
- **No `sudo`.**
- **Frozen data untouched.** `run-a1/`, `run-a2/`, `diagnosis-a1/` and `analysis/` were mounted read-only. Every
  output is in `followup-a2/`.
- **Independent of the gate code.** The index-K units were recomputed from the raw dumps, not through
  `analyze_hookval.py`. The decoding helpers were copied verbatim from `analyze_hookval.py.a1`, and the pool range
  from `compare_windows`.

## Bottom line

| # | Item | Classification | Effect on the frozen verdict |
|---|---|---|---|
| 1 | Both B1 arms show 172/27,973 gross against the rebuild | **Coincidence. No duplicated or aliased comparison, and no analysis defect.** | None. IDX-content, N_dd and H5e-r reproduce exactly from raw bytes |
| 2 | H4-meta disagrees in 2 of 106 snapshots, in both attempts | **A benign metadata-timing artefact, and a mis-specified diagnostic expectation.** It is one snapshot on both ranks, at the align-mode block-boundary reset. **The committed state is correct.** | None. H4-slot and H4b-r hold at that snapshot |
| 3a | be2e9e349395 is outside its own matched null in both B1 arms | **Fixture-specific, reproducible decode-vs-prefill numerics in deep layers.** It is not an instrument defect | Keep flagged |
| 3b | 376073921446 shows arm asymmetry | **An artefact of counting layer-units.** It is one decode run's single-position event spanning 7 deep layers. At the position level the two arms are symmetric (5 vs 6 positions) | Keep flagged |

**Recommendation for the first contrast:**
- Keep both fixtures **in** the primary aggregate, flagged, with no mechanism claim (21g).
- Pre-register a leave-flagged-out sensitivity analysis to report beside it.
- No investigation is required before the first contrast (reasons in §3.3).

---

## 1. The equal B1 gross counts (172 / 172)

### 1.1 The two arms read different files with different contents

`f1_units.py` resolves every window dump exactly as `analyze_hookval.py` `win_dump` does. There is one window
snapshot per tag per rank: 41 tags × 2 ranks, with no duplicates. Every V4/V4n window dump was hashed
(`out/window-dump-sha256.txt`, 82 files). **No two are byte-identical.**

Per fixture, the B1 arms have:
- different files;
- different request seeds;
- different engine steps at the dump;
- different physical block tables.

The full table is `out/f1_dumps_table.md`. An excerpt:

| Fixture | Arm | sha256 r0 (first 12) | Seed | Batch | Step | Block table |
|---|---|---|---:|---|---:|---|
| be2e9e349395 | V4-b1 | 1d2fa4632c33 | 9820034 | the V4 batch of 8 | 3059 | f5a19207… |
| be2e9e349395 | V4n-b1b | d30f9dd20ff8 | 9820051 | b1b-0 | 4513 | 58e4e738… |
| be2e9e349395 | V4-rebuild | 0d0ba246c581 | 9820043 | (alone) | 3775 | 2614448b… |
| 376073921446 | V4-b1 | 1e6bd1519968 | 9820038 | the V4 batch of 8 | 2451 | 156b9b2b… |
| 376073921446 | V4n-b1b | c704a868926a | 9820054 | b1b-1 | 4644 | ee7c0827… |

**Both arms are compared with the same `V4-rebuild` dump by design.** That is the frozen rule: each arm against the
rebuild. It is not aliasing.

At the pool level, b1 and b1b are byte-identical in **0 of 27,973** units. The relative L2 between them is non-zero
at every gross unit (minimum 0.017).

### 1.2 Different unit sets, per fixture and per layer

**Recomputed from the raw bytes:**
- b1: 172/27,973;
- b1b: 172/27,973;
- N_dd: 162;
- N_pp: 128;
- all 27,973 units have both nulls.

These equal the frozen figures exactly.

**Overlap of the gross sets:**
- both arms: 90 units;
- b1 only: 82;
- b1b only: 82;
- Jaccard 0.354.

At units gross in both arms, the two decodes agree with each other (median b1-b1b relative L2 0.108), so the rebuild
is the odd one out. At units gross in one arm only, the median b1-b1b relative L2 is 0.30-0.34.

**Gross units per fixture × layer.** Each cell is b1 / b1b (units gross in both). Layers 3-15 are zero in every
column; layers 3-23 sit on rank 0 and 27-43 on rank 1. Source: `out/f1_table.md`, `out/f1_summary.json`.

| Fixture | n per layer | L3-15 | L19 | L23 | L27 | L31 | L35 | L39 | L43 | Total b1 / b1b (both) | N_dd | N_pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e875e978fd4 | 63 | 0 | 1 / 1 (1) | 3 / 2 (2) | 2 / 2 (1) | 1 / 0 (0) | 0 / 0 (0) | 1 / 1 (1) | 1 / 0 (0) | **9 / 6** (5) | 3 | 0 |
| 376073921446 | 113 | 0 | 2 / 2 (1) | 1 / 1 (0) | 1 / 1 (0) | 2 / 0 (0) | 3 / 1 (0) | 4 / 1 (0) | 3 / 1 (0) | **16 / 7** (1) | 17 | 0 |
| 4b92a6a90ec4 | 429 | 0 | 9 / 5 (3) | 6 / 3 (3) | 7 / 5 (3) | 2 / 3 (1) | 9 / 11 (7) | 15 / 15 (10) | 10 / 12 (7) | **58 / 54** (34) | 52 | 44 |
| 68a96f264430 | 306 | 0 | 1 / 2 (1) | 1 / 1 (1) | 1 / 2 (1) | 0 / 1 (0) | 4 / 6 (2) | 6 / 10 (5) | 6 / 6 (4) | **19 / 28** (14) | 18 | 21 |
| 6d18c40f58df | 267 | 0 | 3 / 2 (1) | 1 / 0 (0) | 1 / 1 (0) | 0 / 0 (0) | 0 / 3 (0) | 0 / 4 (0) | 2 / 6 (2) | **7 / 16** (3) | 20 | 21 |
| be2e9e349395 | 499 | 0 | 4 / 2 (2) | 5 / 4 (4) | 4 / 3 (3) | 0 / 0 (0) | 2 / 3 (2) | 3 / 3 (2) | 3 / 3 (1) | **21 / 18** (14) | 8 | 6 |
| cbac17c70af8 | 499 | 0 | 1 / 1 (1) | 1 / 1 (1) | 0 / 0 (0) | 1 / 0 (0) | 2 / 2 (0) | 4 / 1 (1) | 1 / 3 (1) | **10 / 8** (4) | 11 | 0 |
| ddfd287416a4 | 367 | 0 | 5 / 8 (4) | 5 / 4 (1) | 3 / 4 (0) | 1 / 1 (0) | 4 / 4 (2) | 5 / 6 (3) | 9 / 8 (5) | **32 / 35** (15) | 33 | 36 |
| **Pooled** | 2,543 | 0 | 26 / 23 (14) | 23 / 16 (12) | 19 / 18 (8) | 7 / 5 (1) | 24 / 30 (13) | 38 / 41 (22) | 35 / 39 (20) | **172 / 172** (90) | 162 | 128 |

The per-fixture differences (b1 − b1b) are +3, +9, +4, −9, −9, +3, +2 and −3. They sum to 0. The per-layer
distributions also differ.

### 1.3 The equality holds only at the frozen threshold

A by-construction equality would hold at every threshold. This one does not (`f1_thresholds.py`).

| Threshold | 0.15 | 0.20 | 0.24 | **0.25** | 0.26 | 0.30 | 0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| b1 vs rebuild | 483 | 256 | 182 | **172** | 160 | 115 | 46 |
| b1b vs rebuild | 487 | 292 | 188 | **172** | 158 | 116 | 53 |

### 1.4 How likely is the coincidence?

**Exchange null** (`f1_exchange.py`, seeded, 200,000 simulations). Under it, the two arms are exchangeable decode
replicates. Arm labels are swapped at random per unit, or per position (fixture, pool) because one position event
spans several layers.

| Swap unit | P(equal totals vs rebuild) |
|---|---:|
| Per unit | 0.062 |
| Per position | 0.041 |

**An additional equality, not part of the question, is reported for completeness.** Against the second rebuild the two
arms are also equal: 171 / 171, with different per-fixture splits (`f1_summary.json`).
- **Joint probability:** 0.0045 per unit, or 0.0021 per position.
- **Conditional:** given the first equality, about 5-7 %.
- This was found post hoc, among many possible equalities one could notice.
- No mechanism produces equal totals with different unit sets. An aliased or duplicated comparison would produce
  **identical** sets, and it would give equal counts at every threshold.

### 1.5 Classification and effect

**Coincidence. No duplicated, aliased or self-compared comparison, and no analysis defect.** Nothing needs
recomputing.

Independent recomputation, **post-inspection, beside the frozen values**. Every value is identical:

| Row | Frozen | Recomputed from raw dumps |
|---|---|---|
| IDX-content b1 | 0.6149 % (172/27,973) vs L 1.1583 % | 0.6149 % (172/27,973) vs L 1.1583 % |
| IDX-content b1b | 0.6149 % (172/27,973) vs L 1.1583 % | 0.6149 % (172/27,973) vs L 1.1583 % |
| N_dd / N_pp (pooled, matched) | 0.5791 % (162) / 0.4576 % (128) | 0.5791 % (162) / 0.4576 % (128) |
| H5e-r solo | 0.1822 % (10/5,489) vs pooled L 1.1583 % | 0.1822 % (10/5,489) vs 1.1583 % |
| Page maxima | b1 0.109 (7/64), b1b 0.094 (6/64) | The same page: 4b92a6, rank 1, L43, page 2 |

The two page maxima differ by one gross pool on the same page. That is consistent with the different unit sets.

**The pass margin does not depend on the coincidence.** Each arm is at 53 % of its limit.

---

## 2. H4-meta: the KDA metadata disagreements

### 2.1 Which snapshots

The 106 are 53 snapshots × 2 ranks. Both "disagreements" are **one snapshot on both ranks**, and it is the same one in
both attempts (`f2_meta.py` → `out/f2_meta.json`):

| | Attempt 1 | Attempt 2 |
|---|---|---|
| Request | V3-reject_2-57703691f509 (CROSS fixture, P 3,832, W 1,098) | same |
| Snapshot | F 521, engine step 14,438 | F 521, engine step 5,497 |
| Frontier / num_computed / total_len | 4,351 / 4,352 / 4,353 | same |
| Schedule | reject:2 (1 draft accepted per step) | same |
| Layers | all 34 KDA layers, both ranks | same |
| Hook accepted: previous step / this step | 1 / 1 | 1 / 1 |
| Model-state num_accepted (H4-slot) | 2 = accepted + 1 (exact) | 2 (exact) |
| Layer metadata num_accepted | **1** (diagnostic expected 2) | **1** |
| Mamba spec slots (g2) | block ids [1:5] of [408, 68, 407, 441, 349] | block ids [1:5] of [202, 203, 204, 364, 394] |

The other CROSS snapshot at frontier 4,351, V3-sched_7-53fa5b3e066f F521, is consistent. Its previous step accepted 0,
so it crossed the boundary one step earlier.

### 2.2 Mechanism, from source

Source: image `ac05230b…`. Copies are in `engine-src/`, with hashes in `engine-src/IMAGE-SOURCE-SHA256.txt`. The
server runs `mamba_cache_mode: 'align'` and `num_speculative_tokens: 3` (`run-a2/server.log`).

1. **The diagnostic compares the wrong quantity at a boundary.**
   - The snapshot is taken in `after_commit`, after this step's sampling, so it records this step's `attn_metadata`.
   - The layer metadata `num_accepted_tokens` is the value the forward **started** from. `MambaHybridModelState.prepare_attn`,
     `mamba_hybrid.py` 298-300, gathers `num_accepted_tokens_gpu`.
   - The model-state `num_accepted_tokens_gpu` is updated **after** this step by `postprocess_state` (388-395). H4-slot
     checks that value, and it is exact.
2. **`preprocess_state` runs before `prepare_attn`.** Its docstring, `mamba_hybrid.py` 218-223, says: "so the boundary
   reset is visible to the forward kernels".
3. **The align kernel resets the count at a block crossing.** `preprocess_mamba_align_fused_kernel`
   (`mamba_utils.py` 394-436):
   - records `src_off = num_accepted − 1` (the accepted bias, before any reset);
   - computes `new_state_idx = ceil((num_computed + query_len) / 4352) − 1`;
   - when the running block column changes, sets `num_accepted = 1`;
   - `precopy_mamba_align_fused_kernel` then migrates the committed state from the old column at `src_off` into slot 0
     of the new column.
   - The spec slots move with it. `mamba_get_block_table_tensor` (`v1/attention/backends/utils.py` 1004+) gathers block
     ids from `(seq_len − 1) // 4352`, which gives ids[1:5]. That is what the snapshot recorded.
4. **The arithmetic at this snapshot.**
   - Before the step, num_computed is 4,350, and the query is 1 token plus 3 drafts.
   - computed_after = 4,354, so the column is 1. At the previous step it was 4,352, so the column was 0.
   - The column changed, so the metadata is **1**. The previous step accepted 1 draft, so `src_off` = 1.

**Prediction test.** The rule is: metadata = 1 when the column changes, else the previous accepted + 1. It reproduces
the layer metadata in **106/106 snapshots in both attempts**, and it predicts exactly these two boundary resets
(`out/f2_meta.json` → `prediction`, `predicted_resets`).

### 2.3 Is the committed state correct at that snapshot?

Yes, on both ranks in both attempts.
- **H4-slot:** num_accepted is 2, equal to the hook's accepted count + 1.
- **H4b identification:**
  - The committed slot (index 1) is the slot nearest the prefill reference in **34/34 KDA layers**.
  - Layer 0: committed slot rel 0.0035-0.0037, against 0.25-0.37 for the other slots.
  - Layers 0-2: margin at least 39.
  - Deepest margin 2.0-2.5. Attempt 1's minimum at depth was 1.04.

If the migration had used the wrong source offset, the running state would be one token update off. That is about 0.25
relative at layer 0, the size of the other slots' distances. It is not.

### 2.4 Classification

**Benign: a metadata-timing artefact of the snapshot point under align mode.** The H4-meta expectation (previous
accepted + 1) omits the engine's documented boundary reset.

This is **not a state-commit issue**. The bookkeeping (H4-slot) and the committed state (H4b) are correct at the only
affected snapshot.

**Recorded limits:**
- The conv state at the boundary is not compared. The snapshot keeps only `conv[sidx[0]]`, and H4b uses the SSM
  sketch.
- The output alignment checks (H2/H3) cover the conv state only indirectly.

The revised expectation (§2.2) is post-inspection, reported, and not gated.

---

## 3. Fixture flags

Sources:
- `f3_cross.py` → `out/cross_units.jsonl`: the same units compared across sessions, using attempt-1 V4-b1 and
  V4-rebuild dumps of the identical windows.
- `f3_summary.py`, `f3_positions.py`, `f3_v3ref.py`, `f3_pagefirst.py` and `f3_reuse.py`.

**Cross-session prefill identity.** The prefill rebuilds of attempt 1 and attempt 2 are byte-identical in:
- **95.6 %** of be2e9e349395's units;
- **100 %** of 376073921446's units;
- 0-100 % elsewhere.

Attempt-1 b1 against attempt-1 rebuild is therefore a third, independent decode of the same window, in the same batch
configuration.

### 3.1 be2e9e349395 (outside its own null in both arms)

**Size of the excess.** The own-fixture flag limit is 0.29 % (N_dd 0.15 %, N_pp 0.11 %).
- b1: 21/5,489 (0.38 %);
- b1b: 18/5,489 (0.33 %);
- attempt-1 b1: 34/5,489 (0.62 %).

**Where the gross units sit:**
- 25 units gross in b1 or b1b, at 15 distinct positions;
- **deep layers only**: L19 4, L23 5, L27 4, L35 3, L39 4, L43 5, and none in L3-15. That matches the pooled depth
  profile (pooled gross rate by layer 0.4-2.2 % in L19-43, 0 in L3-15);
- ordinary norms (9.7-13.4) and scales (2^-7, 2^-6).

**Reproducible across three independent decodes.**
- 13 of the 25 units are gross in a2-b1, a2-b1b **and** a1-b1.
- Pairwise overlaps: b1 & b1b 14, b1 & a1 16, b1b & a1 14.
- At those units, the decodes are much closer to each other than to the prefill reference: decode-vs-decode rel
  median 0.06-0.08 (range 0.02-0.48), against decode-vs-prefill median 0.53-0.60 (range 0.26-1.52).

**Two kinds of position:**

| Positions | Units | What happens | Reading |
|---|---:|---|---|
| Pools 446, 474 and 485 (tokens 1,784-1,943; 188-344 before the frontier) | 6 | **The prefill reference is unstable.** N_pp (rebuild vs rebuild2) is itself gross at these units (0.33-0.83), while the decodes agree with each other (rel 0.02-0.12). Pools 486 and 509 nearby have N_pp elevated at 0.08-0.25 | The rebuild is the odd one out |
| Pools 145-231 (tokens 580-927) | 7 | Every decode differs from prefill (e.g. pool 211 L35: 1.26 / 1.33 / 1.28), while the two prefill builds are byte-identical there (N_pp 0) | A deterministic decode-vs-prefill gap at specific positions |

**The in-session N_pp understates this fixture's prefill fragility.** It uses one alternative build length (+200 ids).
Diagnosis-a1's multi-length prefill null (V3ref builds of the same trace) covers this fixture's pools 34-70 (407 units):
- there, 8 units (2.0 %) exceed 0.25 for some pair of build lengths, a mean of 0.50 % per pair;
- the in-session N_pp on the same units is 0 (`out/f3_v3ref.json`).

That coverage is only 7 % of the fixture's units.

**At the position level** (`out/f3_positions.json`), the excess persists:

| Comparison | Gross positions |
|---|---:|
| b1 | 14 |
| b1b | 10 |
| a1-b1 | 17 |
| N_dd | 7 |
| N_pp | 3 |

**No stale-block artefact.** V4-b1-be2e9e349395 reused the physical blocks of the earlier injected V4-d2 request of the
same fixture: the same block table, and the same pages in 5,489/5,489 units.
- b1 and d2 bytes are equal in 67 units, all in layer 3 and all d2-clean.
- cbac17c70af8, whose b1 and d2 used **different** blocks, shows the same rate: 68/5,489.
- So the equality comes from identical computation in the same batch shape, not stale bytes. b1b, in batches of 4,
  matches d2 in 0 units (`out/f3_reuse.json`).

**Classification:** fixture-specific numerical fragility, reproducible across batch compositions and sessions:
- deep layers only;
- partly a fragile prefill reference, partly a stable decode-vs-prefill gap at specific positions.

This is the signature diagnosis-a1 §2(ii) described. There is no addressing or writer signature:
- IDX-page and IDX-early pass;
- no page clustering;
- the decodes agree with each other.

### 3.2 376073921446 (arm asymmetry)

**The unit counts:**

| Comparison | Gross units |
|---|---:|
| b1 | 16/1,243 |
| b1b | 7 |
| N_dd | 17 |
| N_pp | 0 |
| attempt-1 b1 (same batch-of-8 configuration) | 20 |

**The flag is driven by one position.**
- **7 of b1's 16 units are pool 128 (tokens 512-515).** That is one position, gross in **every** deep layer from L19 to
  L43 (0.26-0.86) in a2-b1 only. a2-b1b and a1-b1 agree with the rebuild there (0.06-0.12).
- That run's divergence at this position starts between layers 11 and 15:
  - L15 rel 0.091 against 0.055 and 0.056;
  - L3-11 at baseline (0.014-0.028);
  - so it is an upstream activation difference in that one decode, not a write fault.
- **Other positions move between runs:**
  - pool 77: a2-b1 and a1-b1 gross;
  - pool 87: a2-b1b and a1-b1 gross;
  - pool 129: a2-b1 and a1-b1 gross, a2-b1b not.

**At the position level the arms are symmetric:** b1 at 5 positions, b1b at 6, a1-b1 at 11, N_dd at 7.

**Why the flag fired:**
- N_pp is 0, so the flag limit is the 0.1 % floor (1.2 units).
- The flag counts layer-units, and one position event can contribute 7.

**Page-first pools.** Pool 128 is the first pool of an index-K page. This was tested across all comparisons, counted by
position (`out/f3_pagefirst.json`):
- b1: 3 page-first positions against 1.56 expected (P = 0.21);
- b1b: 1 against 1.66;
- attempt 1: 2 against 1.81;
- N_pp: 2 against 1.19.

There is no page-boundary effect.

**Classification:** run-to-run decode numerics at a few fragile positions in deep layers. The asymmetry is an artefact
of counting correlated layer-units, not an arm-specific defect.

**Recorded limits:**
- No multi-length prefill null covers this fixture, because it is not a V3 trace.
- N_pp = 0 here rests on one build-length pair.

### 3.3 Recommendation for the first contrast

Per 21g, neither fixture can carry a mechanism claim. **Recommendation: keep both in the first contrast's primary
aggregate, flagged, and do not exclude them.** Pre-register beside it a leave-flagged-out sensitivity analysis, plus
the per-fixture results.

Reasons:
1. **Neither flag is an instrument defect.** Every structural check is clean, and the cross-session and cross-batch
   replicates behave as numerics, not as a mis-write.
2. **The contrast cancels the fragility.**
   - The first contrast compares clean with D2-injected decode under matched construction.
   - At be2e9e349395's fragile positions the decode arms agree with each other (N_dd 0.15 %), so the fragility is common
     to both arms.
   - D2's effect is about 60 % gross, more than 100× these excesses.
3. **Excluding them would lose data the control needs.** The long-prompt/short-output control (21g §6) is about history
   construction. be2e9e349395's reproducible decode-vs-prefill gap is exactly the kind of sensitivity that control is
   meant to measure. Excluding it after seeing pool data would select on an outcome-adjacent quantity.

Before the first contrast's analysis is frozen, two cheap rules are worth adding:
- **(a) Fixture flags at the position level.** Count gross (fixture, pool) positions as well as layer-units. The
  376073921446 flag shows that the ≥ 10-unit rule fires on two or three position events.
- **(b) Record be2e9e349395's 13 three-way-reproducible gross units** (`out/f3_summary.json` → `flagged`). Any
  pool-level contrast effect located at them can then be identified as pre-existing decode-vs-prefill numerics.

Nothing here needs investigation, or GPU time, before the first contrast.

---

## Files

All under `followup-a2/`. Run order:
1. `f1_units.py` and `f3_cross.py` (container, via `frun.sh`);
2. `f1_summary.py`, `f1_exchange.py`, `f1_thresholds.py`, `f1_table.py`, `f1_dumps_table.py`;
3. `f2_meta.py` (container);
4. `f3_summary.py`, `f3_positions.py`, `f3_v3ref.py`, `f3_pagefirst.py`;
5. `f3_reuse.py` (container).

| File | What it does |
|---|---|
| `frun.sh` | The CPU container runner: B1 image, `--network none`, no GPU, run data read-only |
| `x_inspect.py` | Inspects the dump structure |
| `f1_units.py` | Every window unit recomputed from raw dumps → `out/units.jsonl`, `out/dump_meta.json` |
| `f1_summary.py` | Overlap, per fixture × layer, byte identity, pages → `out/f1_summary.json` |
| `f1_exchange.py` | Label-swap null → `out/f1_exchange.json` |
| `f1_thresholds.py` | Threshold sweep → `out/f1_thresholds.json` |
| `f1_table.py`, `f1_dumps_table.py` | The markdown tables → `out/f1_table.md`, `out/f1_dumps_table.md` |
| `out/window-dump-sha256.txt` | sha256 of the 82 V4/V4n window dumps |
| `f2_meta.py` | H4-meta snapshots, source prediction, identification → `out/f2_meta.json` |
| `engine-src/` | Engine source copies from the image, with `IMAGE-SOURCE-SHA256.txt` |
| `f3_cross.py` | Attempt-1 against attempt-2 units → `out/cross_units.jsonl` |
| `f3_summary.py` | Flagged-fixture characterisation → `out/f3_summary.json` |
| `f3_positions.py` | Position-level counts → `out/f3_positions.json` |
| `f3_v3ref.py` | The multi-length prefill null join → `out/f3_v3ref.json` |
| `f3_pagefirst.py` | Page-first-pool test → `out/f3_pagefirst.json` |
| `f3_reuse.py` | The block-reuse (stale bytes) control → `out/f3_reuse.json` |
| `SHA256SUMS` | Every file above, this document included |

Fixtures are not comparable across model families, and every figure here is index-K pool relative L2 within one model
and one engine image.
