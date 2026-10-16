# Mechanism ranking: release differences relevant to long decode (amendment 22k)

2026-10-15. Zero GPU. No GPU container, no server, no `sudo`, nothing pushed or posted. Existing files were only read.
Everything new is in this folder. CPU-only containers (`--network none`, `CUDA_VISIBLE_DEVICES=` empty,
`TRITON_INTERPRET=1`) ran one tail-ring harness on three local images (`p2-exposure/`). No GPQA question, answer or
reply text appears here. Sources and hashes are in `SOURCES.md`.

**Labels.** [M] measured (in a record or in this folder's CPU runs); [R] reconstructed from records; [S] read from
source; [I] inferred. "Harness" means the synthetic-scheduler CPU harness running the images' real Triton kernels.
Its keys, drafts and acceptance are synthetic, so its percentages describe exposure, not served outcomes.

**Naming.** The defect index calls the ring-4 / depth-5 defect **P2** and the seed-stride defect **P1**. Amendment 20d
and `probe-proposal/` call the ring defect "D6", and the source review calls it "DCP-audit D6". None of these is
inv-D6, the warm-up stale tail columns. This file uses P1, P2, D2 and D6 as in `defect-repros/INDEX.md`.

## Bottom line

1. **P2 cannot be observed in served 0.7.0 by itself.** In the same ring, D2 (the tail ring addressed without
   positions) already makes 100% of decode-completed pools wrong. That holds at 1, 2 and 8 requests and at the served
   acceptance rate, with or without P2 [M harness]. At GPQA's served concurrency, P2's own signature (a pool reading
   its own rejected draft) shows up in 3 of 972 wrong pools; foreign-key reads dominate (754 of 972 are pure
   foreign) [M harness]. P2 only becomes a separate mechanism once D2 is fixed, and served 0.7.0 never ran in that
   state. The step-5 record agrees: 0.7.0 with the P1 and P2 fixes but D2 kept left decode-versus-prefill KL at stock
   level [M; protocol limits below].
2. **The atomic-slot merge is real, nondeterministic and small.** In each launch and on each rank, 0.7.0 drops one of
   its lowest-ranked selected pools, at most two half-pools per row and layer [M K replay; S]. B1 already corrects the
   class it belongs to (column 511), and the column-511 correction alone made no measurable GPQA difference (15 vs 16
   empty) [M]. Across releases, column-511 selection quality runs B1 > 0.7.0 > stock 0.9.1, while finishing runs
   0.7.0 > 0.9.1 > B1. The two orderings do not match [M/R].
3. **No economical mechanism experiment emerges for either candidate.** A B1 ring-4 reintroduction would cost about
   4-5 GPU-hours. Neither outcome would change the next decision, because served 0.7.0 never ran P2 without D2, and
   the expected effect is smaller than the D2 window injection's, which was itself below the screen (0.106 against
   0.200 nats, NO READING) [I]. An atomic-merge emulation on B1 cannot reproduce the cross-rank half-pool, and its
   expected effect is below D1's [I].
4. **What the record does single out is the combined decode-built index-K state (D2 ⊕ P2 ⊕ P1) over the whole
   history.** Exposure is 100% on 0.7.0, 78-90% on stock 0.9.1 at 8 in flight and 0% on B1, and that order matches
   the long-reply finishing order (observation only) [M/R]. New here [I, with M harness]: at concurrency 1 the steps
   4-6 KL pattern (stock 0.7.0 high; 0.7.0 with P1 and P2 fixed unchanged; 0.9.1 with 0.7.0's DCP2 layout raised;
   0.9.1 at DCP1 low) is the pattern D2 exposure predicts: 100 / 100 / 100 / 0%. The "layout bundle" signal of step 6
   is therefore not independent of D2. A whole-history test costs about as much as a replication and is weakly
   powered, so it ranks after the replication question.
5. **The decision returns to the 22k fallback:** fund a stock 0.7.0 replication (expected 22.97, valid-run ceiling
   26.45, maximum 35.0 GPU-hours [R]) or stop work on the historical gap. See the recommendation.

## Ranked table

Criteria: (1) mechanism, (2) exposure in served execution, (3) discriminating measurement, (4) costed experiment,
(5) a result that changes the next decision. ✓ met, ~ partial, ✗ not met.

| # | Mechanism (release difference) | (1) | (2) | (3) | (4) | (5) | Cheapest GPU cost (central / pessimistic) | Decision value |
|---|---|---|---|---|---|---|---|---|
| 1 | Whole-history decode-built index-K corruption, D2 ⊕ P2 ⊕ P1 (0.7.0 100%; stock 0.9.1 78-90% at 8 in flight; B1 0%) | ✓ [S/M] | ✓ [M harness, R served] | ~ window version unresolved; whole-history needs a behavioural run | ~ ≈ 19-31 GPU-h | ✓ only if the gap is real | 19 / 31 [I] | High if the gap reproduces; premature before that |
| 2 | Draft-slot sharing: DFlash2 sliding-window blocks recycled in shared MLA pages (0.9.1 family and B1 only); the upstream rejected-suffix guard is absent | ~ [S, source review] | ~ served on B1 and 0.9.1, rate grows with length; unguarded | ✓ one extra guard site | ✓ ≈ 0.6-1.0 GPU-h | ~ violation → new B1 defect; clean → retires a UI. Not about 0.7.0 | 0.6 / 1.0 [I from M analogues] | Low-moderate. The only cheap item, and it targets the 0.9.1/B1 side |
| 3 | DFlash depth ×5 (0.7.0) vs ×3 | ~ [S]: sampling distribution exact; changes route thresholds and P2 exposure only | ✓ [M log] | ~ B1 ×5 GPQA passes | ~ ≈ 6 GPU-h per pass | ✗ confounded with route; step 2 showed no KL difference | 6 / 7 per pass [R] | Low |
| 4 | **P2** ring-4 / verify-window-6 rejected-draft overwrite (0.7.0; fixed in 0.9.0) | ✓ [S/M] | ✗ masked by D2 in served 0.7.0 (marginal 0%) [M harness] | ✓ on B1: ring-4 reintroduction plus oracle-draft bracket | ✓ ≈ 4-5 GPU-h | ✗ neither outcome changes the decision | 4.2 / 5.6 [I] | Low |
| 5 | **Atomic-slot top-k merge** (0.7.0 DCP2): column-511 drop varies per launch and per rank | ✓ [S/M] | ✓ every decode row under DCP2 [M K, S] | ✗ the cross-rank half-pool cannot be reproduced on B1 (DCP1); only an emulation | ~ ≈ 4 GPU-h plus a new kernel post-process | ✗ | 4 / 5 [I] | Very low |

Ranked lower again in "Other release differences": DCP2/EP2 layout (as a channel separate from D2), KDA block migration
(3,584 vs 4,352), B12x fe054789 vs 7fcc094e, P1 alone, D6, P3 and vision.

## Served execution (shared facts)

| | Stock 0.7.0 (GPQA passes 2-3) | Stock 0.9.1 (GPQA) | Source |
|---|---|---|---|
| Layout | TP2, EP2, DCP2; B12X sparse MLA; kpool tail ring 4 | TP2, DCP1, MLA owners `split:25`; ring 8 | [M] server log; [S] `attention.py` v070:192, v091:191-203 |
| Draft | DFlash2 ×5 (verify window 6) | DFlash2 ×3 (window 4) | [M] server log |
| Per-position acceptance | 0.710, 0.503, 0.363, 0.269, 0.206; mean length 3.05; conditional ≈ 0.71-0.77 | 0.755, 0.579, 0.454; mean length 2.79; conditional ≈ 0.76-0.78 | [M] `served_spec_metrics.out` (2,751 and 1,995 metric lines) |
| Concurrency | 87% of status lines at ≥ 8 running; 6% at 1-2 | 81% / 92% at ≥ 8 (passes 2-3 / pass 1) | [M] same |
| KV usage peak | 85.8%, no preemption | 41.6% | [M] same; source review §2 |
| Long replies | 14 at ≥ 100k (11 questions), 10 finished; pass 1 unlogged | 25 at ≥ 100k, 10 finished | [M] amendment 20c (exploratory; about 10 events per release) |

Pass 1 of 0.7.0 has no server log. Its acceptance and concurrency are assumed equal to passes 2-3 [I].

## Candidate P2: rejected drafts overwrite committed tail slots (0.7.0)

**(1) Mechanism.**
- [S] 0.7.0 sizes the tail ring at `block_size = index_kpool` = 4 (`attention.py` v070:192).
- DFlash ×5 stashes all 6 verify-window tokens before acceptance. The ring slot is `pos % POOL_SIZE` (`kpool_compress.py`
  v070, decode stash).
- When a pool-completing draft is rejected, the later drafts in the same window wrap and overwrite committed slots.
  The redo then reads those slots.
- 0.9.0+ sizes the ring as kpool·2^⌈log₂⌈(kpool+K)/kpool⌉⌉ (`attention.py` v091:191-203): 8 at K=3 and 16 at K=5.
  The kernel docstring states the invariant (v091:578-580).
- [M harness] Ring 8 at K=5 also gave 0% wrong pools (`b1-r8-k5`), so the upstream sizing has margin in this harness.

**(2) Exposure in served execution.**
- [M harness, this folder, 0.7.0 image, acceptance 0.72 ≈ served] As released (D2 present): 100% of decode-completed
  pools wrong at 1, 2 and 8 requests.
- With positions passed (D2 removed): 57.5 / 52.5 / 45.7%, almost all `own-rejected-draft`. That is P2's
  counterfactual exposure. The P1 boundary pool adds 0.8-1.9% at acceptance 1.0 (earlier run).
- [M harness] In the released configuration the classes are mostly `foreign`. At 8 requests, 754 of 972 wrong pools
  are pure foreign, and pools carrying any own-rejected-draft read number 3 of 972. At 1 request they are 58
  of 120, mixed with `own-other-position`.
- [R] GPQA ran at ≥ 8 in flight for 87% of status lines. So in served 0.7.0, P2 adds 0 to the wrong-pool rate and
  barely changes which garbage a pool holds.
- [M, existing record] Step 5 (amendment 6): 0.7.0 with only the P1/P2 backports (`attention.py` ring sizing and the
  0.9.0 `kpool_compress.py`), D2 kept, ×5, concurrency 1. KL at 8,000-15,999 tokens: 0.0439 / 0.0390 against stock
  0.0558 / 0.0243, "remain at stock level".
  - This is the release-level P2 contrast, already paid for (1.615 GPU-hours).
  - [I] It is what masking predicts.
  - Limits: 16k only; the bin is forced post-answer continuation (amendment 7); KL is not termination.
- [M harness] Counterfactual on B1's own kernels (image ac05230b, positions passed), ring 4 reintroduced:

  | Depth | Acceptance | Wrong pools at 1 / 2 / 8 requests |
  |---|---|---|
  | K=3 | 0.77 | 24 / 21 / 18% |
  | K=3 | 0.6 / 0.85 | 37-42% / 12-16% |
  | K=5 | 0.72 | 58 / 52 / 45% |
  | Control: ring 8, K=3 | 0.77 | 0% |

  The shadow classification agreed with the real kernel for every pool.
- [I] **Interaction with D2 and what it implies.** P2 is observable only where D2 is absent: B1 with ring 4, or 0.7.0
  with the D2 fix and without the P2 fix. Neither configuration was ever served. A P2 effect measured on B1 therefore
  describes a computation stock 0.7.0 did not run, and cannot account for 0.7.0's served behaviour.
  - Removing P2 alone from 0.7.0 is predicted to be invisible, as step 5 found.
  - The meaningful served object is the combined state (candidate 1).

**(3) Discriminating measurement.** On B1 (D2 fixed, D7 fixed):
- Contrast B1 ×3 with B1-ring4 ×3 on the first-contrast fixtures: close propensity, selected corrupted window pools.
- Add an oracle-draft bracket: drafts set to the next history tokens, acceptance near 1. That removes P2's exposure
  (harness at acceptance 1.0: about 1%) while keeping everything else.
- Only a shift present in B1-ring4, absent under the oracle bracket, and beyond the within-arm null is P2-specific.
- The fixtures' forced acceptance (2.84 committed tokens per step [M], against a served 2.79 at ×3) is close to served.

**(4) Cost (measured rates; not run).**
- Base: run-fcr1 measured 2.93 GPU-hours for variant B. Primary-only A0 is 2.75 central / 3.72 pessimistic
  (`cost_fcr.out`).
- One more 3-run arm (the oracle bracket) ≈ +1.0 / +1.3 [I: (2.75 − 0.75 pre-traffic and reference cost) / 2].
- The two gates that failed in fcr1 (H5f and the KDA sentinel) need the prospective validation design of 22k first,
  as CPU work plus an integrated validation segment of about 0.4-0.8 [I, from the early-stop costs 0.42-0.83].
- The B1-ring4 build follows the B0-ring4 recipe in `probe-proposal/` (reverse the spec-ring hunks), with CPU checks
  only.
- **Total ≈ 4.2 central / 5.6 pessimistic GPU-hours.**

**(5) Would a result change the next decision?** No [I].
- *Shift found:* P2-class corruption moves close propensity on B1. That says nothing about served 0.7.0, where D2
  masked P2. At most it adds motivation to candidate 1, which is motivated already.
- *No shift (the likelier outcome):*
  - exposure is about a third of the D2 injection's (18-24% against about 70% gross window pools in fcr1, H5d-fc);
  - the content is the request's own draft tokens rather than foreign keys;
  - the D2 injection itself stayed below the screen.
  It would retire nothing that the D2 masking argument has not already settled.
- Neither outcome moves the replicate-or-stop choice. **Not economical.**

## Candidate: atomic-slot top-k merge (0.7.0, DCP2)

**(1) Mechanism** [S]:
- The decode indexer takes the B12x kpool route (`sparse_attn_indexer_kpool.py` v070:881-887). Under DCP2 it calls
  `_merge_b12x_dcp_topk` (:924) on that route regardless of the owner-exchange environment variables.
- That function all-gathers both ranks' top-512 lists, 1,024 candidates (`sparse_attn_indexer.py` v070:1139-1218), and
  runs `run_row_topk` (:1207). The function assigns output slots through shared-memory atomic counters (`tiled_topk.py`
  v070, per DIAGNOSIS-run1 §2).
- Each rank then truncates its own list to `pool_ids[:, :511]` (`sparse_attn_indexer_kpool.py` v070:1009). The MLA keeps
  only own-parity tokens (`owning_rank = tok % 2`).
- So which pool lands in column 511 varies per launch and per rank. The attended set becomes 510 whole pools plus half
  of two possibly different pools.

**(2) Exposure** [M K rerun, 0.1408 GPU-hours]:
- Every 0.7.0 cell shows launch nondeterminism: the relaunch control differs in every record.
- Column 511 differs between perspectives in 60-80% of rows, and about half of those are non-tied. In the tiled 48-row
  131k cell the real-input families give 222/288 and 171/288 rows (178 and 107 non-tied).
- The selected *set* differs only among exact ties.
- The minimum rank of a differing pool is 1-94 per cell. For real inputs, the CPU claim model puts column 511 in the
  final claim group: the lowest median 5-10 selected pools [R].
- The 0.7.0 counter for column 511 ranked better than 480 fires in 15% of rows (311 of 2,016) [M]. The exact families (the zero
  family) supply 288 of those rows; the real-input families only a few percent (modelled 19 and 11.5 of 576
  row-repeats in the two run-1 cells) [R, `crosscheck_col511.out`]. Compare D1's
  88.1% high-rank discards on stock 0.9.1's tiled route [M].
- Served: every decode row on the B12x route under DCP2, at all lengths [S + M server log].
- Magnitude: at most 2 tokens of each of 2 low-ranked pools per row and layer, out of 2,047 attended [I].

**(3) Discriminating measurement.**
- B1 runs DCP1, so the cross-rank half-pool cannot be reintroduced faithfully. The closest single reintroduction is B1
  with column 511 drawn per launch at random from the final claim group. That emulates the per-launch drop, not the
  half-pools.
- Its observable would be added within-arm variance, with no expected direction.
- A within-0.7.0 measurement would also show variance, not a cause.

**(4) Cost:** about 3.7-4.5 GPU-hours on the fcr1 instrument (A0 plus validation segment) [I], plus a new kernel
post-process and its CPU tests.

**(5) Decision value: very low** [I].
- The perturbation has no systematic direction, and it is smaller than D1. B1 already corrects D1, and the correction
  alone was measured null on completion (15 vs 16 of 594, one-sided p = 0.50).
- The column-511 quality ordering does not match the finishing ordering.
- A null changes nothing. A positive would be surprising and would still not give 0.7.0's direction. **Not
  economical.**

## Candidate 1 (comparator that subsumes P2): whole-history decode-built index-K state

**(1)** D2 (and, under it, P2, P1 and D6's stale columns) writes every decode-completed pool from the wrong rows
[S/M, defect-repros D2].

**(2)** Exposure:
- [M harness] 0.7.0 100% at every concurrency. Stock 0.9.1: 0% at 1 request, 47-57% at 2, 78-90% at 8. B1: 0%; in
  serving, its shadow ledger was clean over 10,172 pools (21j) [M].
- The ordering matches finishing at ≥ 100k: 0.7.0 0.71, stock 0.9.1 0.40, B0+B1 0.12, the last pooled and modified
  (19f/20c) [M counts; the association is an observation only].
- **Reconciliation, new here.**
  - [M harness, `v091-dcp2`] 0.9.1's kernels under 0.7.0's DCP2 mapping make 100% of decode pools wrong even at 1
    request.
  - So at the concurrency 1 of steps 4-6, D2 exposure is 0.7.0 100%, 0.7.0 with P1 and P2 fixed 100%, 0.9.1 at DCP2
    100% and 0.9.1 at DCP1 0%.
  - The KL aggregates follow exactly that split: 0.0243-0.0558, 0.0390-0.0439, 0.0309-0.0604 against 0.0119-0.0142
    [M, steps 4-6].
  - [I] Decode-versus-prefill disagreement is what D2 produces, because prefill rebuilds index-K correctly. This
    offers a mechanism for the step-6 "layout bundle" result that needs no layout effect, but it is untested.
  - The KL bins measured forced post-answer continuation, so this says nothing about completion.

**(3)** The window version ran: fcr1, NO READING, with a descriptive Δ of 0.106 nats toward higher close propensity
under injection. Its stated limit is that a negative window result cannot exclude corruption accumulated over a whole
history (21 §7). The whole-history discriminator is behavioural: B1 against B1 with decode-only D2 injection for the
whole generation (D7-correct prefill), scored on long-reply finishing.

**(4)** Cost [I from M]:
- 3 GPQA passes ≈ 18-19 GPU-hours (B1 pass 6.33, B0 pass 5.866), plus extending and validating the injection hook to
  the whole decode, about 1.
- Controls: historical B1 and B0 records, or about 12 more GPU-hours for the two missing B1 seeds.
- **Total ≈ 19-31.**
- Power for the long-reply quantity is weak: about 6-8 replies reach 100k per pass.

**(5)** If D2-class corruption raised finishing, that would name a concrete old-engine mechanism; the reading rule is
that corruption is never restored as a remedy. If not, the line closes. Either result changes the plan, but only if the
historical gap is real. At replication cost with weaker power, it should follow a replication, not precede it.

## Other release differences relevant to long decode (ranked)

1. **Draft-slot sharing in shared MLA pages** (B1 and stock 0.9.1 only) [S, source review §4]:
   - DFlash2 blocks (2,048-token window, 1,088-token blocks) are recycled about every 1,088 tokens, about 300 per long
     reply. The draft slot mapping (`dflash/speculator.py:544-574`) was not a guard site, and the upstream
     null-block / rejected-suffix guard is absent.
   - Low prior, but it is the one listed difference on the *0.9.1/B1* side, where non-termination would have to
     originate if 0.9.1 is the deviant.
   - Cheap: one guard site in a B1 guarded session. Analogues measured at 0.62-0.99 GPU-hours. Recycling starts within
     the first few thousand tokens, so long contexts are not needed [I].
2. **DFlash depth ×5 vs ×3:**
   - Sampling is distribution-exact at any K [S].
   - Step 2 found no KL difference on 0.9.1 [M].
   - Depth shifts the fused/tiled route boundary (1-2 vs 1-4 requests on fused) and P2's exposure, the latter only once
     D2 is fixed [S/M harness].
   - B1 ×5 was never served. A GPQA pass is about 6 GPU-hours, confounded with route.
3. **DCP2/EP2 vs DCP1 with MLA owners:**
   - Through D2, DCP2 turns 0.9.1's concurrency-dependent corruption into 100% (above).
   - Apart from that: the combine is exact up to rounding (DCP-audit D1, amendment 16) [M/S]. The pilot and
     replication showed no completion benefit (2/10 vs 0/10; 0 vs 0) [M]. The component screen on one question showed
     no candidate (EP2 3/24 vs 5/24) [M].
   - Treat it as a channel for D2, not as a separate mechanism.
4. **KDA block migration under speculation** (every 3,584 tokens on 0.7.0, 4,352 on B1):
   - The code is byte-identical [S].
   - The boundary rule is understood and the committed state correct at 4,351 (21k) [M].
   - The convolution-state boundary and the fcr1 sentinel failures are unresolved [M].
   - It is not a version mechanism unless an effect depends on block size or K. UI.
5. **B12x fe054789 vs 7fcc094e:**
   - Both select the correct top-512 set from 2k to 330k: B1 34/34; 0.7.0 meets C1-C4 in all 22 cells [M K].
   - The decode and merge rewrite is rounding only (DCP-audit D3) [S].
   - Remaining differences are the serial fused merge in 0.7.0 and the column-511 handling (the candidate above).
6. **D6 stale warm-up columns:** present in all releases and active through D2. For 0.7.0's 4-token ring, column 1
   covers positions 4-7 [I]. Not modelled for 0.7.0, and subsumed by candidate 1.
7. **P1 prompt-boundary seed stride:** one pool per request (18 of 18 boundary pools wrong in the harness). That is a
   negligible fraction of 25k-80k pools at long context, and masked by D2 in served 0.7.0 [M/I].
8. **P3** (DCP1 tail tokens dropped at contexts ≤ 2,043): not exercised by 0.7.0's DCP2 and irrelevant to long decode
   [S, INDEX].
9. **Vision on (0.7.0) vs off:** text-only requests; no long-decode path found [I].

D3, D4 and D5 are the same in every release (fixed in B1 for D3 and D4) and are not release differences.

## Recommendation

- **Do not fund GPU work on P2 or on the atomic-slot merge.** Both are verified as code facts, but neither meets
  criterion 5:
  - P2 is masked by D2 in every served 0.7.0 state (marginal exposure 0%, and step 5 already removed it without
    effect);
  - the merge is a small nondeterministic low-rank perturbation inside a class B1 already corrects.
  The CPU exposure numbers here close both as release mechanisms worth a contrast.
- **The decision therefore returns to 22k's fallback.**
  - *Stock 0.7.0 replication* (`rep070/PROPOSAL.md`, draft) [R]: expected 22.97 GPU-hours; valid-run ceiling 26.45;
    maximum with one invalid-run allowance 35.0.
    - Its primary (empty replies at the recorded seeds) is well powered: O1 82-87% if 0.7.0's record
      reflects its rates (94% in the 0.25 × 0.9.1 scenario); O2 88% if 0.7.0 behaves like 0.9.1's record [M simulation in rep070]. A halfway rate splits
      across outcomes.
    - The long-reply finishing contrast would be a secondary with about 21 at-risk replies. Power is about 0.59 even
      if 0.7.0's 0.71 is true, and lower with question clustering (`power_long_finish.out`) [I].
    - The draft predates the long-reply framing. Adding that secondary, and choosing recorded seeds (repeatability) or
      fresh seeds (population rate, same cost), are re-freeze items.
  - *Stop:* 0 GPU-hours. The verified corrections (D1-D4, D7 and the B1 checkpoint) stay reportable separately, as
    22k states.
- **Lean (for the operator's decision).** Every mechanism reading depends on whether the gap is real. The gap rests on
  one three-pass record per engine, about 10 finishing events each, and per-question cap outcomes that changed in 8 of
  14 questions between two same-seed passes.
  - The replication is the only next step whose result changes the plan either way:
    - O1 makes candidate 1 the concrete target, tested on 0.7.0 with D2, P2 and P1 all fixed, or on B1 with
      whole-history injection;
    - O2 ends the historical-gap line.
  - If 23-35 GPU-hours is not worth that answer, stop.
- Optional and independent of that choice: the draft-slot guard site on B1 (about 1 GPU-hour), only if hardening the
  checkpoint is wanted.

## What ran here, and limits

- `served_spec_metrics.py` read counts from three GPQA server logs.
- `p2-exposure/` holds a copy of the D2 two-rank harness with configuration overrides only (diff in its header). It ran
  8 configurations × 3 seeds × 1/2/8 requests on the 0.7.0, 0.9.1 and B1 images, CPU only, about 9 minutes.
- `power_long_finish.py` is a pure-Python Fisher simulation.
- Harness limits:
  - a synthetic scheduler, keys and drafts;
  - iid conditional acceptance;
  - no stale columns;
  - kernels run sequentially, whereas a GPU races requests sharing block 0;
  - B1 kernels come from the forcehook-mapguard image (B1 plus guard and hook) [R];
  - the 0.9.1 DCP2 page geometry is assumed (14 sub-pages); it does not affect the classification.
- GPU-hour estimates scale measured sessions. No new rate was measured. Long-context decode rates beyond 120k come from
  the fcr1 fixture range only.
- Nothing here explains or localises the completion gap. Non-completion is not looping.
- A read-only `nvidia-smi --query-compute-apps` call was issued once by mistake inside a shell line. Its output was
  discarded, and no GPU work was started.
