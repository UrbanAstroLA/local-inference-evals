# Live causal-length and port-01 invariant check (frozen proposal, amendment 22ao, 2026-10-22)

**Status:** frozen with the package `livecheck-22ao/` (SHA256SUMS) as amendment 22ao, before any data. Not
authorized: the GPU run needs the operator's authorization at ≤ **2.0 occupied GPU-hours**, one attempt, no retry.
It adopts the assessor's assessment of 22an. Changes from the review draft are listed in section 11.

## 1. Question and scope

**Question:** on stock tpurtell 0.9.1 (`a9e985c0`, DCP1, async scheduling, DFlash2 ×3), do speculative verify rows
consume per-token causal lengths that differ from the authoritative `p + 1`? If they do, does any consumed selected
entry fall outside the row's causal range? Measure the same on stock + port 01, where lengths come from positions.

**Not claimed, whatever the result:**
- an explanation of the completion difference;
- a property of stock 0.7.0. The same source branch exists there, but DCP2 uses rank-local lengths;
- an equivalent fix for 0.7.0;
- a D2 effect.

## 2. Three levels of evidence, reported separately

**Indexing:**
- A query row at zero-based logical position p may read keys k ≤ p; its exclusive length is L* = p + 1.
- A key k violates causality when k ≥ L*.
- Physical slots are never compared with positions. Every slot is mapped to (request, logical position) through the
  guard's own ledger of that request's group-0 blocks, built from the SchedulerOutput stream. Request identity,
  block ownership, rank and row (slot) reuse are kept with each record.

| Level | Site | Measurement | Reported as |
|---|---|---|---|
| **1. Metadata** | output of the B12x sparse-MLA metadata builder (`cache_seq_lens_per_token` for verify rows) | signed d = consumed length − (p + 1), per row | distribution of d (negative, zero, positive), joined with the previous step's rejected count for that request, row reuse and context regime |
| **2. Selection** | the read site (`check_selected`, existing mapguard): entries `sel[t, j]` for j < `nsa[t]`, after `_compact_dropped_selection` and `_mask_page_table_after_nsa_len`, i.e. what the kernel consumes | `sel_causal` (held, logical ≥ p + 1), `sel_foreign` (not this request's block), `masked_live` (j ≥ nsa not masked), `sel_len` (nsa > p + 1); new: signed `nsa − min(p + 1, width)` | counts with row identities |
| **3. Computation** | **not measured in this session** | if level 2 yields a validated violation on A, a follow-up captures the offending rows' inputs and compares the attention output with and without the offending entries | per the decision table (section 8): a minimal isolated causal-length reproduction and computational comparison |

**Expected from source (S / I), to be measured:**
- **Context below the selection width:** in the dense regime (`nsa` = min(L, width)), an inflated L can raise `nsa`
  above p + 1. What the extra columns hold decides level 2: -1 (masked), stale or foreign entries, or later positions.
- **Context above the width:** L is clamped by the width and level 2 cannot be reached through the count. The fixed
  tail columns (2,044-2,046 under the DCP1 tail rule) are checked through the same entries.

## 3. Arms and images
- **A, stock 0.9.1 + guard:** the B1 mapguard rebased onto the stock 0.9.1 files. Its read and write sites are
  unchanged.
- **C, stock 0.9.1 + port 01 + guard:**
  - port 01 exactly as submitted (PR #7), applied to the image's own files at build time (`--network none`);
  - a second application must be a no-op, and the two touched files must change;
  - port gate 01 passes on the built image (3/3 port expected), and on A it reports the stock omissions
    (`--expect-unpatched`, 3/3).
- **Stock tail-ring (D2) and other known violations on A** are recorded as POSITIVE FINDINGS (known D2), never as
  instrument failure.
- **Not included:**
  - a causal-length-only arm, until this result is read;
  - the MoE MAX_M=16 arm. It supplies no capture this check needs; the MoE replay stays separately bounded.

## 4. Workload and coverage
- **Client:** 24 requests from the 12-question panel × 2 fresh seeds, temperature 1.0, top_p 0.95, natural stopping,
  prefix caching as served. Exactly 8 outstanding; a new request is sent the moment one finishes.
- **Request mix, so that rows recycle:**
  - 8 long requests (`max_tokens` staggered 6,144 down to 2,560 in steps of 512, so the drain passes every padded
    size);
  - 16 short requests (`max_tokens` 768).

  Shorts finish early and take over rows freed by earlier requests, including rows that held longer ones (long
  requests are front-loaded).
- **Reporting:** steady state (8 in flight engine-side) and drain are reported separately.
- **Required exposure, fixed before data.** Counted engine-side per arm from guard records. A shortfall in any row
  gives INCOMPLETE for the dependent invariants.

| Path | Minimum count |
|---|---:|
| Verify rows following a step with ≥ 1 rejected draft (same request), dense regime (p + 1 < width) | 300 |
| Verify rows following an all-accepted step (control), dense regime | 100 |
| Verify rows past the width (clamped regime) | 300 |
| Row recycles where a request replaces a finished request in the same runner row; of those, a shorter-context replacing a longer-context occupant | 10; 5 |
| FULL replays at each of the padded decode sizes 4, 8, 16, 24, 32 | 20 each |
| Transitions between captured sizes (step to step) | 20 |

**Why these minima:** the rule of three gives 95% upper bounds of about 1% (300 rows) and 3% (100 rows) on a
per-row violation rate when none is seen. These are the bounds stated with CLEAN WITHIN COVERAGE.

## 5. Outcomes, per invariant and arm

| Outcome | Condition |
|---|---|
| INCOMPLETE | instrument failure, guard error, invalid attribution (row or request not joinable), missing required records, required exposure not met, or a failed in-session positive control |
| POSITIVE FINDING | a validated violation on that arm (the positive controls passed in the same session and attribution is valid) |
| CLEAN WITHIN COVERAGE | required exposure met, positive controls passed, no violation; reported with the rule-of-three bound |

## 6. Positive controls (in session; for the selection checker itself)
**Shadow plants:** on a guard-owned copy of the consumed tensors, never on engine memory, for a fixed deterministic
subset of verify rows (every 97th checked row):

| Plant | Checker that must fire |
|---|---|
| (a) one live column set to a slot of the same request at logical position p + 1 or later (inside its blocks) | `sel_causal` |
| (b) one live column set to a slot in another request's block | `sel_foreign` |
| (c) `nsa` raised by 1 above min(p + 1, width) | `sel_len` and the signed count |
| (d) one masked column left unmasked | `masked_live` |

- Each must be detected at every planted row (detected / planted reported), and plants are excluded from the engine
  counts.
- **The same plants are run on CPU first:** the existing mapguard tests (`test_mapguard.py`, `b0emu.py`) plus new
  cases for the signed metadata site.
- **The deterministic lookahead control** validates the draft writer only. It is not used here.

## 7. Observer must preserve the condition under test
- **Async scheduling stays on:** guard `init` record `async_scheduling: True`.
- **Graph modes unchanged:** FULL decode fraction and capture list equal to the unguarded served logs (O).
- **No forced synchronisation:** the guard's only host wait is an event wait when the GPU is ≥ 4 steps behind; the
  count of waits that actually blocked is recorded. If more than 1% of steps block, async exposure is reduced: the
  causal-length rows are then marked INCOMPLETE unless the metadata discrepancy is still observed at the required
  counts.
- **Throughput:** decode tok/s under the guard reported against unguarded served rates (descriptive).

## 8. Decision rules (adopted from the assessment)

| Live finding (level reached) | Next decision |
|---|---|
| No length discrepancy at required exposure (level 1 clean) | Rule down this metadata mechanism within coverage. MoE replay continues independently. |
| Discrepancy (level 1) but no consumed non-causal or foreign entry (level 2 clean) | Record the discrepancy, without claiming leakage. Identify whether another computational effect remains (for example tail-column compaction). |
| Consumed non-causal entry on A, absent on C, with level-3 output change | Prioritize a minimal isolated causal-length reproduction and computational comparison. Reprice the larger D2 experiment. |
| INCOMPLETE | Repair the specific deficiency; no mechanism claim. |

## 9. Cost and limits
- **Engineering, about 1-2 days:**
  - rebase the mapguard (`mapguard.diff`: `sparse_attn_indexer_kpool.py`, `b12x_mla_sparse.py`, `scheduler.py`)
    onto stock 0.9.1 and onto stock + port 01;
  - add the metadata site with its signed record, the row-reuse annotation, and the shadow plants
    (level-3 dumps dropped; see section 11);
  - CPU tests: plants, refusals, a session simulation with a watchdog stop, and a real `DRY_RUN=1`;
  - SHA256SUMS.
- **GPU, estimate:**

  | Item | Estimate |
  |---|---|
  | Traffic per arm: 8 long × 6,144 tokens at about 70 tok/s ≈ 1.5 min each unguarded, assumed 2-3× under the guard; the 16 shorts overlap them | about 5-8 min |
  | Per arm: about 13 min start-up and release warm-ups + traffic + 3 min margin | about 21-24 min wall ≈ 0.7-0.8 GPU-hours |
  | Two arms | **about 1.4-1.6 GPU-hours (ceiling 2.0)** |

  - The guard slowdown factor is the main uncertainty. Earlier mapguard sessions (B0 a3, B1 a1) ran a 2-5 min workload
    in 0.57-0.62 GPU-hours including start-up.
  - **Exposure check:** each request's first ~2k tokens give about 700 verify steps × 4 rows in the dense regime, so
    about 2,800 rows per request, far above the minima. Padded sizes 4-24 occur in the drain: long requests are
    staggered (`max_tokens` 6,144 down to 2,560 in steps of 512) so the drain passes every size; the session simulation must
    confirm 20 FULL replays per size.
  - The watchdog stops at 2.0 and an unfinished run is INCOMPLETE. If the CPU session simulation shows that the minima
    cannot be met within the ceiling, the proposal returns for a revised ceiling before freezing. It is not run with
    reduced minima.
- **Limits:**
  - coverage is the exercised shapes, rows, contexts and the 12-question panel only;
  - rank 1 reads are covered only through the owned layers (MLA ownership split:25);
  - a clean result says nothing about 0.7.0 or DCP2.

## 10. Package as frozen (`livecheck-22ao/`)

**Images**, built with `--network none` from the local stock image `a9e985c0`. Hashes are gated before and after copying.

| Arm | Image | Contents |
|---|---|---|
| A | `local/glm53-v091-22ao-a:local` `sha256:dd51ed73…` | stock + guard |
| C | `local/glm53-v091-22ao-c:local` `sha256:5617e0df…` | stock + port 01 + guard |

**Guard:** `guard/mapguard_hooks.py` `e2dcd964…`, built by `guard/make_guard_22ao.py` from the frozen B1 guard
(`a8e59fa5…`).
- Exact-match edits only. The removed lines are the extracted predicate block and the extended lists.
- **Additions:**
  - the `mla_len` site;
  - signed `nsa` counters;
  - the four shadow plants (1 token in 97);
  - a host blocked-wait counter (`uploads`, `blocked_waits`);
  - per-row length records (`lrec`, first owned MLA layer per rank).

**Hooked files:**
- `b12x_mla_sparse.py` and `scheduler.py`: B1's hooked files, whose bases equal stock byte for byte, plus one new call
  (`hook_mla_len`) where `per_token_cache` is formed.
- `sparse_attn_indexer_kpool.py`: stock plus the B1 guard's 9 hook lines. The prefill write is placed by hand with
  `ring_mode="none"`, because stock has no D7 tail ring. The change is purely additive.

**Workload:** `runner/client_22ao.py`.
- 24 chat requests from the private panel fixture. All 12 message hashes match the 0.9.1 record's request log.
- Seeds 72201-72224 (no prior use found under `<data-runs>`).
- Exactly 8 outstanding, with immediate backfill.
- 8 long requests (`max_tokens` 6,144-2,560) first, then 16 short (768).

**Runner:** `runner/run_22ao.sh`.
- Arms A then C, each with a wall cap of 1,740 s (2 × 0.967 GPU-hours ≤ 2.0).
- Budget gate, image pins, frozen hashes, pre-flight, readiness and guard-installed gates.
- A watchdog, an EXIT trap, and one cost line per arm.

**Analysis:** `analysis/analyze_22ao.py`, the outcome rules of section 5.
- Exposure is counted from engine-side records:
  - **verify rows:** batch rows with spec tokens;
  - **dense:** the guard's `dense` flag, p + 1 < width;
  - **previous rejections:** the scheduler-side `upd` record of the request's previous spec step;
  - **recycles:** runner rows changing occupant; "shorter" means the new occupant's position is below the previous
    occupant's last.
- Attribution is the share of length records joined to step rows with a matching tag (≥ 0.99 required).

**CPU tests** (`tests/out/`, all no-GPU, `--network none`):
- **`test-22ao.log`: 10 / 10:**
  - the signed length site against independent truth: exact; over by the previous step's rejected drafts
    (20 / 20 tokens); under by one (4,716 / 4,716 tokens);
  - all four plant classes detected at every planted token, at 1 / 7 and 1 / 97 (deterministic);
  - plants leave the frozen counters unchanged;
  - the frozen negative control is clean.
- **`out-frozen-suite/`, the frozen B1 guard suite on the 22ao module:**
  - `test_mapguard` 27 / 28. N1 fails only because the frozen rig does not drive the new `mla_len` site; every
    frozen site's counts are identical, and 26 of 28 result records are field-identical (the other one adds only the
    new key);
  - `test_d7_guard` 30 / 30, identical;
  - a2 regression identical.
- **`runner-tests.txt`: 17 / 17:**
  - 8 refusals;
  - a real `DRY_RUN` that writes nothing;
  - a mock session per arm: peak exactly 8 outstanding, 24 / 24 seeds served, two cost lines;
  - a watchdog stop with cost lines still written and no process left.
- **`port-gate.txt`:** C 3 / 3 port expected; A 3 / 3 stock omissions.

## 11. Changes from the review draft
- Level 3 is not attempted in-session (it would need in-graph dumps of q rows and KV records). It follows only a
  validated level-2 finding.
- The port-01 image check is the port gate plus build-time checks. "Equal to B1's served files" was wrong, because B1's
  copies of those files also carry port 05.
- A host blocked-wait counter was added to make the observer check in section 7 measurable.

## 12. Before any data
1. **PLAN amendment 22ao** adopting the assessment, with this proposal frozen (hashes).
2. **Frozen package** with the CPU tests passed (§6, §9).
3. **Operator authorization** at ≤ 2.0 GPU-hours.
