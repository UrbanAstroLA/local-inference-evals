# DIAGNOSIS-fc1: why the first long-context contrast stopped after C1

CPU-only, post-inspection diagnosis of `first-contrast/run-fc1`, per PLAN.md amendment 22b. Written 2026-10-15.
- No GPU was used, no glm53 container was started, and no frozen file was modified.
- CPU work ran either in host Python or in `--network none` containers of the B1 / B1-d7 images, without `--gpus`.
- Nothing here says anything about D2 or termination. The C1 per-fixture statistics in `fc_result.json` are not used
  or interpreted.

Outputs (all in `diagnosis-fc1/`):

| File | Content |
|---|---|
| `scripts/cache_rule.py` → `cache_rule.json`, `cache_rule.out` | quick_struct sub-checks; the cached-token rule; scheduler-log confirmation; the eviction evidence; the LRU horizon; the same rule in four earlier sessions |
| `scripts/idx_pages.py` → `idx_pages.json`, `idx_pages.out` | G1-idx-struct page and early-layer criteria on C1 vs LW1, C1 vs LW2, and the prefill null LW1 vs LW2 (frozen `fc_common` functions, unchanged) |
| `gate-asrun/` | The frozen `analysis/gate_fc.py`, re-run unchanged on run-fc1. Gives every GATE-1 row with full detail; `fc_result.json` keeps only the failing list |
| `scripts/cost.py` → `cost.json`, `cost.out` | Cost sketch for a new attempt, from measured phase times |
| `SHA256SUMS` | Hashes of the above |

---

## 1. Verdict

**Exactly one quick_struct sub-check failed: `cached_tokens == cached_expected`, for all 8 C1 requests.** Every request
returned without error. Every `ids_equal` was true. Every request committed its full forced list (`completion_tokens`
= W, finish `length`). The hook logs carried no `fw_error` or `guard_error`.

The 8 cache failures have two causes:

| Fixtures | Observed vs planned | Cause | Class |
|---|---|---|---|
| L1, L4, M4, L3, L5, A8, L8 (7) | exactly **one 4,352-token block fewer** than planned | The plan's expected value is wrong for this engine. B1's prefix cache drops the last matched block for EAGLE-type speculators (DFlash counts as one), and the drop is a whole 4,352-token block because fine-grained hits are disabled | **(a) plan/harness defect**, rooted in **(b) correct engine behaviour** the plan did not model |
| M3 (1) | **0** cached (rule value 56,576) | M3's warm prefix was evicted by LRU before C1 looked it up. The prefix-cache horizon on this hybrid model is about 0.66 M prompt tokens, smaller than one round's warm phase (0.74 M) | **(a) plan defect** (capacity model), rooted in **(b) correct engine behaviour** |

**No class (c) fault was found**, neither in the instrument nor in the engine. C1's forced windows pass every
hook-level structural check in the frozen gate:
- WMAX, absolute positions, forcing and once-only measurement;
- errors, attribution (flattened and padded steps), all-8-in-flight, and clean ledger.

C1 also passes the per-request alignment, crossing and regime checks against LW1 (§4).

**The frozen GATE-1 could not have passed on this engine configuration, even with C2 and CV run.** Three independent
reasons, each certain from the data in hand:
1. **G1-cache**: C2 and CV would also come in at least one block short (deterministic rule, §2). The PS sentinels'
   expected values carry the same error.
2. **G1-idx-struct**: it already FAILS on C1 alone. 8 judged pages exceed 25 % gross and 3 early-layer pools are gross.
   The prefill-vs-prefill null LW1 vs LW2 fails the same criteria (7 pages, 1 early pool), at the same M4 pages. The
   structural threshold, carried over from short-context attempt 1, is not met by the reference builds themselves (§5).
3. **Eviction**: with a 0.66 M-token horizon, re-warms after other salts' traffic, CV on s1's prefix, and the PS
   sentinels would mostly miss. That makes exact cache accounting non-deterministic (§3).

The stop was correct under the frozen rules (amendment 22 §9).

---

## 2. The 19-vs-20-block rule (seven fixtures)

### 2.1 Records

C1's prompt is **identical** to warm-s1's prompt: same `prompt_sha256`, same salt, P = warm P.

| Fixture | P | Planned c0 = ⌊P/4352⌋·4352 | Observed | Blocks obs / planned | (⌊(P−1)/4352⌋ − 1)·4352 |
|---|---:|---:|---:|---:|---:|
| L3 | 89,304 | 87,040 | 82,688 | 19 / 20 | 82,688 |
| M4 | 83,416 | 82,688 | 78,336 | 18 / 19 | 78,336 |
| A8 | 111,896 | 108,800 | 104,448 | 24 / 25 | 104,448 |
| L1 | 94,512 | 91,392 | 87,040 | 20 / 21 | 87,040 |
| L8 | 111,896 | 108,800 | 104,448 | 24 / 25 | 104,448 |
| M3 | 63,488 | 60,928 | **0** | 0 / 14 | 56,576 |
| L5 | 112,024 | 108,800 | 104,448 | 24 / 25 | 104,448 |
| L4 | 72,360 | 69,632 | 65,280 | 15 / 16 | 65,280 |

The engine scheduler log (`mapguard/ms_p274.jsonl`) gives each C1 request's `num_computed` before its first chunk.
It equals `cached_tokens` for all 8. For each of the 7 hit fixtures, C1 re-used exactly the first n − 1 MLA blocks of
its own warm request (n = ⌊(P−1)/4352⌋), and the n-th block was re-prefilled into a fresh block. For example, C1-L3
took warm-L3's 19 MLA block ids and a new block for index 19.

The LW references (all 16) and the warm-s1 requests (all 8) were cold, cached 0, exactly as planned.

### 2.2 Source: B1 image `vllm/v1/core`, extracted read-only

- **The last token is always recomputed:** `kv_cache_manager.get_computed_blocks` sets
  `max_cache_hit_length = num_tokens − 1`.
- **DFlash is "EAGLE":** `SpeculativeConfig.use_eagle()` is true for `method == "dflash"`.
  - `KVCacheCoordinator.__init__` marks EAGLE groups. For wrapped GLM this means the groups that carry draft
    (`model.layers.*`) layers.
  - `GLM53_DFLASH_BOUNDARY_LOOKUP=1` would turn the drop off. The run had it at `0`
    (`container-files-env.txt`), so the drop applied.
- **The drop is a whole 4,352-token block:** `HybridKVCacheCoordinator` logs "Disabling fine-grained prefix-cache hits
  because these KV cache managers require block-aligned lookups: KpoolTailManager" (seen in b0-mapguard's smoke log).
  - With fine-grained hits off, the alignment unit is `scheduler_block_size` = 4,352 (the LCM of 4,352, 1,088 and 8).
  - `FullAttentionManager.find_longest_cache_hit` matches ⌊(P−1)/4352⌋ full blocks, then
    `hit_length -= min(alignment_tokens, block_size)`, i.e. one whole 4,352-token block.
  - The sliding-window path pops one block and re-aligns down to 4,352, which gives the same result.
- **KDA state follows the reduced length:** `MambaManager.find_longest_cache_hit` (align mode) searches right-to-left
  for a cached state at a block boundary no longer than the reduced length. The KDA states at every block boundary had
  been cached (`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` unset, so dense retention), so the hit is
  (⌊(P−1)/4352⌋ − 1)·4352.

### 2.3 The rule was already in the investigation's own records

Exact-repeat prompts with P > 4,352 in four earlier sessions on this stack follow the same rule:

| Session | Repeats following the rule |
|---|---|
| b0-mapguard run-a3 | 34 / 34 |
| phase3 | 17 / 17 |
| d7 step2 run-b1-a1 | 34 / 34 |
| phase2-stage2 | 29 / 35 |

The six phase2-stage2 exceptions are warm requests at the start of a second server session in the same file (cold).

Example: the 120,217-token prompt repeated gives 113,152 cached (26 blocks), not 117,504.

PROPOSAL §2.5 instead took c0 from the generic "full blocks" reading, ⌊P/4352⌋·4352.

### 2.4 Correct invariant

For a request whose prompt equals a cached warm prompt, or extends it within the warm's last partial block (C, D, PS,
CV, re-warms), with no eviction in between:

> cached_tokens = max(0, (⌊(P_warm − 1)/4352⌋ − 1) · 4352)   (P_warm not a multiple of 4,352; true for every fixture
> here). Equivalently c0 − 4,352, or c0_L − 4,352 on LP salts.

What changes as a result:
- The batch prefills 4,352 + (P mod 4,352) tokens per request, not P mod 4,352. That is 8 × 4,352 ≈ 35 k extra
  tokens, about 13 s per batch at the measured 2,770 tok/s.
- **Cache safety (P6) is unaffected.** Reading fewer blocks than planned is always safe. Block [c0 − 4,352, c0) is
  recomputed in prefill (clean) under the same hash.
- **The PS sentinel design is affected.** Its dump range [c0 − 1,024, c0) lies in the block the PS request
  recomputes into a private block, not in the last shared cached block. The last shared cached block is
  [c0 − 8,704, c0 − 4,352).
  - Its "KDA state rebuilt from the cached boundary state" starts at c0 − 4,352, then prefills 4,352 + rem tokens.
  - As written, PS-prefix would test prefill determinism of a private block, not integrity of the shared cached block.

---

## 3. M3 = 0: eviction inside the warm phase

### 3.1 Evidence (scheduler log)

- warm-M3 was the first warm admitted and the first to finish (scheduler seq 1,624).
- At seq 1,966, before C1-M3's lookup at seq 1,973, **warm-M3's MLA block idx 7 (block 344) was re-allocated** to
  C1-L3's recomputed block. Re-allocation removes the block's hash, so M3's MLA chain was broken at block 7.
- The cached-block free queue is FIFO (`BlockPool.free_blocks`: blocks with a hash are appended; blocks without one
  are prepended and reused first). Freed blocks therefore leave in this order:
  1. M3's KDA boundary-state blocks, freed during its prefill (seq 1,579-1,622);
  2. then its MLA blocks, freed tail-first at 1,624.
  So M3's KDA states had already been evicted, and the KDA lookup found nothing, giving 0.
- **No other fixture's needed blocks were re-allocated** before its lookup. The two re-allocations seen for A8 / L8
  touched index 25, the partial tail block, which is never cached as a full block.

### 3.2 The horizon

The pool has **1,275 block ids shared by all groups** (log `glm53-placement kv-capacity`):
- 1 MLA group (12 layers);
- 7 KDA/Mamba groups, each needing one new state block per 4,352-token boundary in align mode;
- SWA and kpool groups.

Measured horizon:
- From warm-M3's free (seq 1,624) to the re-allocation of its block idx 7 (seq 1,966), the engine scheduled 657,147
  tokens.
- The free queue held a median of 1,154 blocks over that span.
- That is about **7.6 blocks churned per 4,352 prompt tokens**, consistent with 1 MLA + 7 KDA state blocks.
- **The effective prefix-cache horizon is about 0.66 M prompt tokens**:
  - one warm round is 0.74 M tokens (warm-s1: 738,896);
  - the proposal's capacity argument (PROPOSAL §3.4) assumed a 4.7 M-token KV cache holding about 3.4 M tokens of
    resident salts.

The "4,707,515 tokens" in the log is a per-request concurrency figure. It is not the reach of the prefix cache when
KDA checkpoints churn the shared pool.

### 3.3 Consequences for the frozen order (not observed, but they follow)

- **In any round, the first-warmed fixture is likely to miss.**
  - The engine order of warm admission and of batch admission is set by eight concurrent HTTP threads, so it is not
    reproducible: C1 was admitted in the order L4, L1, L3, A8, M3, M4, L8, L5.
  - The margin was thin. L1, the next most exposed fixture, had its needed state freed only about 94 k scheduled
    tokens after warm-M3's MLA (about 14 % of the horizon). L4's margin was about 155 k (24 %).
- **Re-warms after other salts' traffic would miss.** These are rewarm-l1a/l2a/s2/s1/l2b/l1b/s3/l3, plus CV on s1
  after warm-s2 + C2. Each miss turns a ~25 s re-warm into a ~270-300 s cold build.
- **A miss is safe** (recomputation; no decode-written block can be read). But it is a different prefix construction
  (in-batch prefill vs cache-restored). G1-cache's exact equality then fails non-deterministically.

### 3.4 Effect on C1's batch

M3 prefilled all 63,488 tokens inside the batch. The other seven prefilled 4,352 + rem. Every request still entered
decode before any other had passed output index 119 (scheduler log), so G1-inflight (< 512) passed.

---

## 4. C1's forced windows

From the frozen gate re-run unchanged on run-fc1 (`gate-asrun/gate1.json`):

| Row | Status on C1 | Note |
|---|---|---|
| G1-V0 | PASS | 4/4 probes bitwise equal |
| G1-wmax | PASS | 8,448 on both ranks; full planned lists in the hook records (no truncation) |
| G1-abspos | PASS | hook P = planned P on both ranks; dumps carry tag, P and range; window snapshots at F = W |
| G1-forcing | PASS | ids_equal 8/8; every output index measured exactly once; LW reference rows once each |
| G1-errors | PASS | no fw_error / guard_error |
| G1-attrib | PASS | 3,529 flattened, 1,535 padded designated records; 512 selection-record positions; padded steps exercised |
| G1-inflight | PASS | all 8 decoding before output index 512 (max lead 119) |
| G1-clean | PASS | d2_injected_tokens 0; shadow ledger clean; no corrupt pools selected or present |
| G1-align | C1 pairs pass; row FAIL only because the C2/LW2 pairs are absent | pooled top-1 0.984; per-request top-1 0.973-0.997, ±1 shift ≤ 0.016; \|Δlp\| q50/q90/q99 = 0.0002/0.092/0.369, within 0.01/0.30/1.5 |
| G1-cross | C1 windows all pass; FAIL only because M4's second crossing (o = 7,976) is covered by LW2/C2 by design (LW1 ends at o = 7,168) | 15 windows, top-1 0.961-1.0 |
| G1-regime | PASS | M3's 65,536 crossing: top-1 0.980, shift 0.008; 18 selection rows valid |
| G1-content | not evaluable | no N_dd without C2 |
| G1-idx-struct | FAIL on C1 itself; see §5 | |
| G1-coverage, G1-CV-* | FAIL | phases not run |

**Answer: C1's forced windows look valid.** ids equal forced, positions are absolute and exact, every index is measured
once, attribution is complete, and all 8 were in flight. Window dumps and snapshots are present for all 8 on both
ranks (forcewin `req_C1-*`, `snap_C1-*_F*win`).

---

## 5. G1-idx-struct: a second, independent defect

The row's structural sub-criteria are no judged 64-pool page above 25 % gross, and zero gross pools in DSA layers 3-15.
Here they are on C1 vs the prefill reference LW1, against the prefill null (`idx_pages.out`):

| Comparison | Pooled gross | Pages > 25 % gross | Early-layer gross | M4 gross |
|---|---:|---:|---:|---:|
| C1 vs LW1 | 1,206 / 157,696 = 0.765 % | 8 (all C1-M4, pages 334-340) | 3 (M4, layer 15) | 704 |
| C1 vs LW2 | 0.706 % | 8 (all M4) | 2 (M4, layer 15) | 684 |
| **LW1 vs LW2 (prefill vs prefill)** | **0.769 %** | **7 (same M4 pages)** | **1 (M4, layer 15)** | **733** |

- Per fixture, C1-vs-LW1 matches LW1-vs-LW2 closely: L1 47/52, L4 15/14, M3 49/45, L3 179/164, L5 24/18, A8 91/86,
  L8 97/101.
- M4's pages around o ≈ 2,100-3,900 are fragile under any rebuild.
- The frozen structural criteria are not met by two cold prefills of the same tokens. They are unattainable on this
  panel, which makes this a **(a) plan defect** (threshold calibration). It is not an instrument fault.
- For orientation only (post-inspection, not a gate result): under the 21g-1 content rule C1's pooled gross (0.765 %)
  sits at N_pp (0.769 %), well inside 2·N_pp. N_dd was never measured.

---

## 6. Why the CPU tests and the DRY_RUN did not catch it

1. **DRY_RUN was plan mode.** No server ran and no cached value was produced. P7 checked that the plan's expected
   values followed the plan's own rule, so the check was circular.
2. **Synthetic records** (`tests/make_synth_fc.py`) wrote `cached_tokens` equal to the planned c0. The single
   negative (g-cache: one request cold) proves only that the check fires.
3. **Hook CPU tests W4/W4b/W5/W6** inject `num_computed = 4,352` into a simulated engine. The scheduler's lookup is
   not modelled.
   - W4's case ("4,400-token prompt, 4,352 cached") is a state this engine would not produce: with the drop it gives 0.
   - These tests are valid for the hook. They say nothing about cache accounting.
4. **No engine-side model of the prefix cache was run.** The image's `KVCacheManager` / `HybridKVCacheCoordinator`
   are pure Python. A CPU replay with the logged KV-cache config (1,275 blocks, the group list), the session's prompts
   and salts, and the scheduler's chunking would have reproduced both the one-block drop and M3's eviction.
5. **The earlier sessions' records** (§2.3) showed the rule, but no test or review step compared the c0 rule against
   measured hits.
6. **The capacity argument** used the KV-token figure. It did not include the shared-pool churn from KDA checkpoints.
7. **G1-idx-struct's page and early thresholds** were inherited from attempt 1 (short windows) and never evaluated on
   a long-context prefill null before freezing. LW1 vs LW2 became available only in-session.

---

## 7. Is C1 usable as a clean run in hindsight?

**As a labelled post-inspection instrument run: yes.** It is **not** a pass under the frozen gate.

Its windows are valid (§4). Its prefix states are clean prefill states:
- 7 fixtures: [0, c0 − 4,352) restored from warm-s1's cache, plus [c0 − 4,352, P) prefilled in-batch.
- M3: the whole prefix prefilled in-batch.
- No decode-written block was readable.

Limits:
- It is one run, with no C2 (no N_dd), no CV and no D runs in the same session.
- It cannot be paired with injected runs from another session (amendment 22 §2: same-session, fixed-order rounds).
- M3's prefix construction differs from what a D1 under s1 would have had.

It can serve as instrument characterisation at long context. In that role it shows alignment, crossings, the regime
crossing and index-K content at the prefill-null level. It says nothing about D2 or termination.

---

## 8. Recommended fixes (post-inspection; for review, none applied)

| # | Fix |
|---|---|
| F1 | **Expected cached value.** cached_expected = max(0, (⌊(P_warm−1)/4352⌋ − 1)·4352) for C, D, PS, CV and re-warms (c0 − 4,352 / c0_L − 4,352); keep 0 for cold references. Update P7, client, gate and synthetic records together (one geometry definition), and cite the §2.3 records as the empirical basis |
| F2 | **Gate the safety property, not an exact count, where exactness is not needed.** Safety: cached ≤ bound, a block multiple, ≤ every earlier decode start under the salt (P6). Then classify hit / partial / miss per request and record it as a per-fixture covariate. Require exact equality only where a measurement depends on the restored state (PS; possibly CV), and design those to sit inside the horizon (F3). In-flight stays gated (G1-inflight already caught nothing wrong with M3's cold prefill) |
| F3 | **Design for the measured horizon (~0.66 M tokens, ~7.6 blocks per 4,352 tokens).** Options, each needing review: (i) keep each batch's warm volume plus intervening traffic ≤ about half the horizon, and drop cross-phase re-warm hits from the cost model; (ii) sparse KDA retention (`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` = k·4,352) so intermediate states carry no hash and are recycled LIFO. Source-reading caveat: the retained "replay boundary" state sits at ⌊(P−1)/4352⌋ blocks, one block past the EAGLE-reduced hit, so hits would fall back to the last retained segment boundary. That needs a CPU replay before use. (iii) `GLM53_DFLASH_BOUNDARY_LOOKUP=1` removes the drop (hit = c0) but changes prefix construction and the pinned configuration; validate first. Only (i) leaves the validated environment unchanged |
| F4 | **Deterministic admission.** Submit each batch in a fixed order with a small stagger, so cache outcomes and decode-start skew are reproducible and recorded (today eight threads race) |
| F5 | **PS sentinel.** Dump the last *shared* cached block [c0 − 8,704, c0 − 4,352) and state the KDA boundary as c0 − 4,352, or redesign PS around F2/F3 |
| F6 | **G1-idx-struct.** Make the page and early-layer criteria null-relative and numerical: compare against LW1 vs LW2 on the same units, as G1-content does. Keep only attribution and population as structural. The current thresholds fail on the prefill null (§5) |
| F7 | **CPU engine replay as a pre-launch test.** Drive the image's `KVCacheManager` on CPU with the logged KV-cache config, the session's token ids, salts and chunking, and assert the plan's cached values and evictions before freezing |
| F8 | Re-run the frozen gate code on run-fc1 (as in `gate-asrun/`) as a regression fixture for the revised gate. Under F1/F6, C1 should pass the rows it can evaluate |

---

## 9. What a new attempt would need

- **Authorization.** The single-attempt authorization (amendment 22a, 4.29 GPU-h) is spent: 1.0861 GPU-h was used and
  no retry is permitted under it. Any new attempt needs:
  - a new operator authorization;
  - given the gate and design changes (F1-F6), most likely an independent review of the revised package before the
    freeze.
- **Measured inputs** (run-fc1):
  - startup 732 s;
  - V0 6 s, LW1 298 s, LW2 301 s, warm-s1 267 s, C1 300 s;
  - dump pause 6 s per phase;
  - prefill 2,770 tok/s (cold warm phase);
  - hooked decode 36.6 tok/s per request with 8 in flight (planned central 33);
  - GATE-1 on CPU 54 s.
- **Scope and cost** (`cost.out`; seconds × 2 GPUs):

  | Scope | Contents | GPU-h |
  |---|---|---:|
  | A: stage-A-only validation | V0, LW1, LW2, warm + C1, warm + C2, CV (CV cold) | ≈ 1.58 |
  | B: full option B, eviction not designed out | every re-warm, CV and PS cold | ≈ 4.49 (exceeds the 4.29 ceiling) |
  | C: full option B, eviction designed out (F3) | re-warms, CV and PS hit on the corrected rule | ≈ 2.77 |

  - Scope A departs from amendment 22's "no separate validation session" and would itself need authorization.
  - Scope C is close to the proposal's 2.86 central, because measured decode was faster than assumed.
  - A ceiling would be re-derived by the new proposal (the current rule, max(1.3 × central, target), gives about 3.6
    for C).
- **Open before any launch:**
  - F1-F7 frozen and reviewed;
  - the CPU replay (F7) predicting every cached value in the new plan;
  - G1-idx-struct recalibrated on a null available before launch. LW1 vs LW2 from run-fc1 is one candidate, post-data
    and labelled as such.
