# Hook validation, attempt 2: revised criteria (DRAFT for independent review; NOT adopted)

> **Status (PA2-24, 2026-10-14):** reviewed and adopted with qualifications by PLAN.md amendment 21g. The adopted
> form, as implemented and frozen in `analysis/analyze_hookval.py`, is described in `ATTEMPT2-21G.md`. This draft is
> kept unchanged below as the reviewed text.

2026-10-14. CPU only. This is a proposal. The criteria take effect only if an amendment adopts them before any
attempt-2 data. Until then:
- the analysis (`analysis/analyze_hookval.py`) computes them as "DRAFT criteria (not adopted)";
- it computes **every original criterion as well**, by running the frozen attempt-1 code
  (`analysis/analyze_hookval.py.a1`) verbatim.

Evidence labels: **[M]** measured from attempt-1 records; **[R]** reproduced on CPU; **[I]** inferred.

## Sources

| What | Where |
|---|---|
| Every number below | `prep-a2/out/criteria_basis.json` (`prep-a2/criteria_basis.py`, run on `run-a1` and `diagnosis-a1/out`) |
| Each proposed rule, evaluated by the attempt-2 analysis on the real attempt-1 data | `prep-a2/replay-a1/verdict-a2.{json,txt}` |

The replay also reproduces attempt 1's original verdict check-for-check.

## Summary

| Area | Original criterion | Attempt-1 value | Measured null | Proposed (DRAFT) | Proposed rule on attempt-1 data |
|---|---|---|---|---|---|
| KDA state (H4b) | rel ≤ 0.05 and margin ≥ 3 | rel max 0.364, margin min 1.04 | No per-depth null measurable | Slot identification 100 %; layers 0-2: rel ≤ 0.02, margin ≥ 10 | **PASS** (3,604/3,604; 0.0089; 18.7) |
| Index-K addressing (H5b, H4c) | B1 gross ≤ 0.1 %; V3: no gross pool | 0.71 %; 1.49 % | Prefill vs prefill 0.53 % | No page > 25 % gross; 0 gross in layers 3-15; byte-stable over time; end pools excluded | **PASS** (page max 9.4 %; 0 of 10,172; 100 %) |
| Index-K content (H5b, H5e, H5d) | B1 and solo ≤ 0.1 %; D2 > 30 % | 0.71 %, 0.18 %, 60.9 % | Prefill vs prefill 0.53 %; decode b8 vs b1 0.11 % (one trace) | Gross ≤ max(2 × max(N_dd, N_pp), 0.1 %), both nulls in-session (V4n); D2 > 30 % | **PASS** with a1's N_pp (limit 1.06 %); literal "≤ 2 × N_dd" **FAILS** (see 3) |
| Ledger (H5f) | Clean → non-gross ≥ 99 %; corrupt → gross ≥ 95 % | Defective (B1 arm 1,243 "corrupt") | n/a | B1 arms 100 % clean; D2 clean-labelled gross ≤ content limit; corrupt reported | Not measurable on a1 (hook defect); CPU D7p-a 21/21 clean |
| Mask bounds (H1f) | No cell outside eligibility bounds | 224 cells, max kept p 2.6e-9 | n/a (engine edge case) | No cell with tracked mass > 1e-6; unmasked rows as their own class | **PASS** |
| Alignment power (H2b, H2c) | lp shift median ≥ 1 nat and ≥ 20×; per request 10× | Shift medians 0.032 nat | Argmax: aligned 0.983, shifted 0.007 | Per request: aligned top-1 ≥ 0.90 and shifted ≤ 0.20; lp rule reported | **PASS** (min 0.951; max 0.021) |

**Also changed, by analysis fixes rather than new thresholds** (no threshold loosened):
- H2a (join): attempt-1 replay passes.
- H5g (per rank, from the start dump): replay passes.
- H5h (excludes o = 0 and W ≤ 1; records must match the measurement step): replay fails on the hook defect, as it must.

**Unchanged:** V0, H1a-e, H1g-j, H3, H4a, H5a.

## 1. KDA committed state (H4b)

**Original.** Relative L2 of the committed slot against the prefill-built state ≤ 0.05 in every KDA layer, and the
committed slot at least 3× closer than any scratch slot.

**Attempt 1 [M].** n 3,604 (106 snapshots × 34 layers): relative L2 median 0.045, maximum 0.364; minimum margin 1.04.
FAIL.

**What was measured instead [M].**

| Layers | n | Identification (committed slot is the nearest) | Minimum margin | Comparisons with margin < 1.25 |
|---|---:|---:|---:|---:|
| 0-2 | 318 | 100 % | 18.7 | 0 |
| 3-17 | 1,166 | 100 % | 2.37 | 0 |
| 18-31 | 1,060 | 100 % | 1.28 | 0 |
| 32-44 | 1,060 | 100 % | 1.04 | 14 |

- Layers 0-2 see identical embeddings, so their gap is the KDA kernel alone (recurrent against chunked): relative L2
  maximum 0.0089, median 0.0051.
- The gap grows with depth to a median of about 0.13. One token's update at depth is about 0.2, so a 3× margin is
  unattainable there by construction.
- No per-depth null exists: attempt 1 never built two prefill states at the same frontier.

**Proposed.**
- Identification 100 % in every layer.
- Layers 0-2: relative L2 ≤ 0.02 and margin ≥ 10.
- Deeper layers: reported per depth, not gated.

**Justification.**
- A wrong slot or a wrong spec-row order moves every layer at once. Identification catches it. The layer 0-2
  tolerance catches a state that is close to the right slot but drifted.
- 0.02 is 2.2× the measured maximum. A margin of 10 is about half the measured minimum.

**Risk for review.** Deep layers have near-ties: 14 comparisons below 1.25, minimum 1.04. A fresh sample of about
3,600 comparisons could therefore fail "100 % in every layer" on noise, with no defect behind it.
- The analysis therefore also reports a variant, **H4b-alt**: identification 100 % in layers 0-17 (minimum margin
  2.37) and ≥ 99 % in deeper layers.
- The variant gives up nothing against systematic misidentification, because that hits all layers.
- **Recommendation: adopt the variant.** The stated form is gated as drafted, pending review.

## 2. Index-K addressing (structural; replaces H5b and H4c as the addressing control)

**Original.**
- H5b: B1-arm gross ≤ 0.1 %. This was the addressing positive control: a wrong page compares unrelated records.
- H4c: no gross pool near the V3 frontier.

**Attempt 1 [M].**

| Comparison | Gross |
|---|---|
| B1 arm vs rebuild | 0.71 % (n 27,973, excluding end pools) |
| V3 decode vs prefill | 1.49 % |
| **Null:** prefill vs prefill (identical tokens, different build lengths) | 0.53 % (n 106,898) |

Both checks failed against a threshold set below the engine's own prefill-vs-prefill noise.

**Proposed.** All four parts apply to the B1 arms, the solo injected request and V3.
- **(i) No page with more than 25 % gross pools.**
  - Each (request, rank, layer, pool) counts once; for V3 that is the latest snapshot.
  - A page is judged only when it has ≥ 16 compared pools.
- **(ii) Zero gross pools in DSA layers 3-15.**
- **(iii) Committed pools byte-identical between consecutive V3 snapshots of the same request.**
- **(iv) Window comparisons exclude pools completing at P + W − 1** (committed, never processed).

**Measured on the aligned data, against a planted wrong page [M].** The planted case compares each decode pool with
the same slot of the adjacent 64-pool page in the rebuild.

| Set | Pages judged | Maximum page gross | Layers 3-15 gross |
|---|---:|---:|---:|
| B1 vs rebuild | 517 | 9.4 % | 0 of 10,172 |
| Solo vs rebuild | 99 | 3.1 % | 0 of 1,996 |
| V3 vs prefill (deduplicated) | 33 | 10 % | 0 of 1,736 |
| **Null:** prefill vs prefill | 1,727 | 8.6 % | 0 of 38,872 |
| **Planted wrong page** | 495 | **100 % (all 495 > 25 %)** | **9,849 of 9,860** |

Two more measured facts:
- **Byte stability:** 106,898 pool pairs are 100 % identical (diagnosis), and the replay's 4,037 frontier-band pairs
  are 100 % identical.
- **End-of-window pools:** 11 per arm, none gross in B1.

**Justification.**
- Misaddressing makes whole pages unrelated in every layer. Fragility is scattered and confined to layers 19-43.
- The separation is complete in both directions:
  - the worst aligned page is 2.7× below the threshold;
  - every planted page is 4× above it.

**Conditions found while checking (needed for the rule to hold).**
- Without deduplication, overlapping V3 snapshots count the same pools repeatedly, and one V3 page reached 27.8 %.
- With deduplication but an 8-pool floor, the worst V3 page sits exactly at 25 %.
- The rule therefore needs both the deduplication and the 16-pool floor.
- A planted shift within a page was not tested: [I] it also compares unrelated pools, so condition (ii) catches it.

## 3. Index-K content (H5b, H5e, H5d; H5c retired as a gate)

**Original.**
- B1 arm ≤ 0.1 % gross; solo injected request ≤ 0.1 %; D2 at 8 in flight > 30 %.
- H5c: B1 99.9th percentile < 0.25.

**Attempt 1 [M].** B1 0.71 %, solo 0.18 %, D2 60.9 %. H5c's q999 was 1.09, against 0.67 for the prefill null, so no
per-pool threshold separates corruption from this noise.

**Nulls measured [M].**

| Null | Gross |
|---|---|
| N_pp: prefill vs prefill | 0.53 % (per trace 0-1.0 %) |
| N_dd: decode at batch 8 vs batch 1 | 0.11 % (**one trace only, cbac17**, n 5,489) |
| B1 vs rebuild, per trace | 0-1.6 % (cbac17 0.07 %) |

**The literal proposal, "B1 gross ≤ 2 × a second B1 arm at a different batch" (N_dd).**
- Evaluated with attempt 1's only decode-path null, it **fails**: 0.71 % > 0.22 %.
- That null comes from the least fragile trace, while B1's figure covers 8 traces.
- On the matched trace it passes: 0.07 % ≤ 0.22 %.
- Decode-vs-prefill also differs from decode-vs-decode in kernels. Attempt 1 measured decode-vs-prefill at 1-5× the
  prefill-vs-prefill rate per trace, and 1.35× in aggregate.
- So a decode-only null can understate the gap a correct engine shows.

**Proposed (DRAFT).** Both nulls are measured in session, on the same 8 windows (phase V4n):
- N_dd = B1 arm vs a second B1 arm in two batches of 4;
- N_pp = rebuild vs a second rebuild that is 124-200 ids longer.

| Check | Rule |
|---|---|
| IDX-content | Each B1 arm vs rebuild: gross ≤ L = max(2 × max(N_dd, N_pp), 0.1 %) |
| H5e-r | Solo injected request ≤ L |
| H5d-r | D2 at 8 in flight > 30 %, also reported as a multiple of L |
| H4c-content | V3 decode vs prefill reported against L, not gated (V3 references are built alone, on other traces) |
| H5c | Reported only |

**Justification and attempt-1 evaluation [M].**
- With attempt 1's N_pp, L = 1.06 %:
  - B1 0.71 % passes, at 1.35 × N_pp;
  - solo 0.18 % passes;
  - D2 is 58 × L.
- The 0.1 % floor stops a zero null from failing a correct instrument. It is the original threshold.
- **Risk:** each null rests on 8 windows, so it carries sampling noise. The second rebuild is batched like the first,
  whereas attempt 1's prefill null was built alone. Its value can differ from 0.53 %.

## 4. Shadow ledger (H5f)

**Original.** Clean → non-gross ≥ 99 % and corrupt → gross ≥ 95 %, over the injected arms pooled.

**Attempt 1.** The ledger was defective (PA2-2). B1-arm pools were labelled "corrupt" 1,243 times and "none" 3,597
times, although they were byte-correct. The replay shows 881 of 5,086 B1 records not clean, and D2 clean-labelled pools
3.5 % gross.

**Proposed (DRAFT).**
- **H5f-B1:** every processed window pool of both B1 arms is classified clean (100 %). This is the positive control
  for the fix.
- **H5f-D2:** D2 clean-labelled pools are gross at most at L, the content limit. A clean pool should be no grosser than
  a B1 pool.
- **Corrupt → gross:** reported only, not gated.
- The original pooled form is reported beside them.

**Justification.**
- **B1 arm:** with circular per-request tail blocks no other request writes a B1 slot, so every committed pool is
  clean by construction. CPU test D7p-a: 21/21 clean and byte-correct through padded steps [R]. The attempt-1 ledger
  fails the same test (D7pn) [R].
- **Why the D2 rule uses L:** the original "≥ 99 % non-gross" ignores the measured fragility. About 0.7 % of
  byte-correct pools are gross against a rebuild.
- **Why corrupt → gross is not gated:** no GPU measurement exists of how far a corrupt member moves a pool. CPU check
  D3 used toy keys.

## 5. top_p mask bounds (H1f)

**Original.** No tracked cell outside the eligibility bounds.

**Attempt 1 [M].**
- 224 cells on 56 rows, all "kept although not eligible", and none "masked although eligible".
- Every row has fp32 top-1 = 1.0 and was left entirely unmasked by the served Triton path (inferred cause: fp32
  overflow).
- The maximum kept tracked probability is 2.6e-9.

**Proposed.**
- Gate on no kept-not-eligible cell with p_post > 1e-6, and no masked-but-eligible cell with p_pre > 1e-6. The second
  half is my symmetric extension: a mass-only gate would otherwise pass masking errors, whose p_post is 0.
- Rows with p_post == p_pre on every tracked id are reported as their own class, with their kept mass.

**Justification.** 1e-6 is 380× the measured maximum, and orders of magnitude below any probability the termination
metrics act on.

## 6. Alignment power (H2b, H2c) and the H2 join (H2a)

**Original.**
- H2b: the lp gap at a one-position shift has median ≥ 1 nat and ≥ 20× the aligned median.
- H2c: per request, aligned median < 0.1 × the shifted medians.

**Attempt 1 [M].**
- Shift medians were 0.032 and 0.031 nat: about 90 % of forced tokens have p ≈ 1, so the forced lp barely moves.
- After the join fix (PI-1), 2 of 120 requests still fail the lp rule (V1-*-free6-doc4).

**Proposed.**
- Per request: aligned top-1 ≥ 0.90 and top-1 at ±1 ≤ 0.20 (H2c-r, invalid level).
- The same rule in aggregate (H2b-r, inconclusive level).
- The lp rule stays reported, labelled "uninformative when the forced lp is ~0".

**Measured [M].**

| Set | Per-request aligned minimum | Per-request shifted maximum | Aggregate |
|---|---:|---:|---|
| All 120 requests | 0.951 | 0.021 | 0.983 vs 0.007 |
| V2a natural + oracle (16 requests; the attempt-2 scope) | 0.976 | 0.014 | |

**H2a** keeps its frozen tolerances. The analysis fix:
- joins by position: reference row k = o + P − ref_from;
- counts top-1 only on finite reference rows.

**Change of provenance, for review.**
- Attempt 2 does not run V1. The H2 null (q50/q90/q99 6.7e-5/0.097/0.406, n 7,787) therefore carries over from attempt
  1 (`analysis/carryover-a1.json`, with source hashes).
- The frozen floors (0.01/0.30/1.50) bind at every quantile in any case.

## 7. Analysis fixes that change no threshold

| Check | Fix | Attempt-1 replay |
|---|---|---|
| H5g | Polls per `rank`, deltas from the client's start-dump poll; tail_w_stash mism == null == owned layers × injected tokens; only in V4-d2/V4-d2solo | **PASS**: rank 0 100,098 = 6 × 16,683; rank 1 83,415 = 5 × 16,683; pre-client warm-up pos_bounds 4 per rank excluded |
| H5h | Exclude o = 0 and W ≤ 1; a record must carry the position's measurement step (stricter: catches misattribution); selection records likewise | **FAIL** (hook defect): 13,935 missing and 3,443 misattributed of 20,263; 231 of 247 selections wrong |
| Window end | Pools completing at P + W − 1 excluded | 11 per arm; none gross |
| H3 | Unchanged; in scope only if the optional poison phase runs, and then judged on the frozen floors alone (no V2b null) | Attempt-1 values pass on the floors alone; otherwise carried over |

## 8. Verdict rule

- Precedence: INVALID > INCONCLUSIVE > VALID. Statuses CARRIED and REPORTED never gate.
- `verdict.json` holds the original verdict and `verdict-a2.json` the revised one. Both are written by every run.
- Synthetic dry run:
  - **good** gives revised VALID and original INVALID. This is by design: it contains attempt 1's non-defects.
  - **bad** gives revised INVALID, failing exactly the 10 planted checks.
