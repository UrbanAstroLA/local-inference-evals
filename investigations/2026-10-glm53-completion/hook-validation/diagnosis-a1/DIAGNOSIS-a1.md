# Hook validation, attempt 1: diagnosis (PLAN.md amendments 21, 21c, 21e)

2026-10-14. CPU only: CPU containers of the B1 image (`sha256:799d3cf8…`) and of the hooked validation image
(`sha256:1b9f8b15…`), `--network none`, no `--gpus`, `CUDA_VISIBLE_DEVICES=`. No GPU or glm53 container was
started, no `sudo` was used, and nothing in `hook-validation/` outside `diagnosis-a1/` was changed. Run data was
mounted read-only. Only ids, positions, metrics and bytes were read; no reasoning text.

Evidence labels: **[M]** measured here from the run's own records, **[R]** reproduced on CPU, **[S]** read from the
installed source, **[I]** inferred.

## 0. Bottom line

- **The original verdict, INVALID, stands.** Two failures are hook defects. Neither can be repaired offline, because
  the records they produced are misattributed, not just missing:
  - **H5h:** the in-graph exposure hook assumes the wrong row layout.
  - **H5f:** the shadow ledger mis-slices padded steps.
- **The forcing instrument itself passed** once analysis defects are corrected, labelled as post-inspection:
  - ids, once-per-position, schedule invariance and integer state;
  - decode-vs-prefill logits equal the in-session unhooked null (H2a passes);
  - the KDA committed slot is identified in 3,604 of 3,604 comparisons;
  - committed index-K pools stay byte-identical over time;
  - the guard pattern is exact on both ranks (H5g passes).
- **No new B1 engine defect was found.** The gross B1 index-K pools are not misaligned comparisons. They are not
  writer defects either: they are a deep-layer numerical fragility of the pooled fp8 key. **Two prefill builds of
  identical tokens show the same signature**: 0.53 % gross pools, 0 % in layers 3-15, up to 1.7 % in layer 39, and
  relative L2 up to 1.36, which means unrelated vectors.
- **One genuine but immaterial engine edge case (H1f).** On 56 near-one-hot rows, the served Triton top-p path left
  the row unmasked. The kept tail mass was at most 2.6e-9 per tracked token, and fp32 top-1 was 1.0.
- **Attempt 2 is warranted, but only after the fixes in section 8**, with the package re-frozen and verified and
  the revised criteria recorded in an amendment before any data.
  - It should be scoped: 0.62-0.76 GPU-hours estimated against 1.188 remaining.
  - The full attempt-1 plan (1.21 GPU-hours) no longer fits.
- **For the operator:** no stop-the-line finding. Two items should reach them before the contrast is designed
  (section 9).

## 1. Classification of every failed check

| Check | Class | One-line evidence | Post-inspection reading (beside the original) |
|---|---|---|---|
| H1f | **(c)** engine edge case, immaterial; plus **(b)**: no tolerance for underflow-level mass | 56 rows, all fp32 top-1 = 1.0, p_post == p_pre, Triton path (≥ 8 rows), max kept p 2.6e-9 [M]; emulated mechanism [R] | FAIL kept; reclassified |
| H2a | **(a)** analysis: wrong reference join, plus NaN reference rows counted as disagreements | top-1 0.845 → **0.983**; q50/q90/q99 1.1e-4/0.100/0.422 against null 6.7e-5/0.097/0.406 [M] | **PASS** under frozen tolerances |
| H2c | **(a)** for 4 requests (same join); **(b)** for 2 (lp shift rule weak on near-deterministic text) | after the join fix, only V1-\*-free6-doc4 fails (ratio 0.12 vs 0.10); argmax shift rule 0/120 [M] | lp rule 2/120 fail; argmax rule passes |
| H2b | **(b)** power control without measured basis | shifted lp medians 0.03 nat because about 90 % of forced tokens have p ≈ 1; argmax agreement aligned 0.983 vs shifted 0.007 [M] | INCONCLUSIVE kept; the argmax control separates |
| H4b | **(b)** tolerance and margin without measured basis; **no slot misidentification** | committed slot is the nearest to the reference in 3,604/3,604; rel grows from 0.005 (layers 0-2) to about 0.13 (layers 33-42) [M] | FAIL kept |
| H4c | **(b)** threshold without measured basis; not misaligned; no writer defect found | 1.49 % gross vs **0.89 %** for prefill-vs-prefill on the same pools [M] | FAIL kept, null reported |
| H5b | **(b)**, as H4c | 0.71 % gross vs 0.53 % prefill-vs-prefill null; layers 3-15 zero [M] | FAIL kept, null reported |
| H5c | **(b)** | B1 q999 1.09; null q999 0.67, max 1.36: no per-pool threshold separates corruption from this noise [M] | INCONCLUSIVE kept |
| H5e | **(b)** | 0.18 % gross, below the null's 0.53 % [M] | FAIL kept, null reported |
| H5f | **(a)** hook defect: the ledger mis-slices FULL-graph padded steps | B1 arm, which must be all clean: 1,243 "corrupt" and 3,597 "none" pools, byte-correct [M]; [R] | FAIL kept; needs fixed hook plus GPU data |
| H5g | **(a)** analysis: polls keyed by a missing `pid`, so one rank was read; a pre-client warm-up artefact was counted | per rank, mism = null = 6 × and 5 × 16,683 injected, only in V4-d2/V4-d2solo [M] | **PASS** |
| H5h | **(a)** hook defect: `on_selection` assumes `[B, next_n]` rows, but SM120 flattens them to `[B·4, 1]` | `use_flattening=True` in `server.log`; 3,457 of 6,342 present records are misattributed [M]; [R] | FAIL kept; needs fixed hook plus GPU data |

The full side-by-side is in `out/verdict-postA1.{txt,json}` (`postA1_verdict.py`).

## 2. Priority: H4c / H5b / H5e, the gross index-K pools

### (i) Misaligned comparison: ruled out [M]

- **Join.** Same layer name on the same rank. The page comes from the group-0 table (`ids[c·68] // 68 · 17 +
  (pool % 1088) // 64`), the same layout the B1 mapguard verified.
  - A wrong page would make all 64 pools of the page gross, in every layer. Instead:
    - gross pools are scattered;
    - layers 3, 7, 11 and 15 have **0 gross in every comparison** (B1, solo, V3 and both nulls);
    - no pool is all-zero on either side.
- **Window start.** `range((lo+3)//4, …)` starts at the first pool lying wholly inside the window. No prefill-seeded
  or straddling pool is compared.
- **Window end.** 11 pools per arm complete at P + W − 1, which is committed but never processed (1 past the
  frontier). None of them is gross.
  - That is a latent harness edge and should be excluded.
  - All 199 gross B1 pools lie strictly inside, from 92 to 1,940 positions before the frontier.
- **Byte stability.** The same decode request, snapshotted at different times (V3, 106,898 pool pairs), is
  **byte-identical in 100 %**. Committed pools are never rewritten, and no stray write lands on them.
- **Context.** All 484 pools with context ≥ 2,044 are non-gross. Gross pools sit at 344-1,900 positions, where every
  pool is attended. So no selection effect is involved.

### (ii) Expected decode-vs-prefill numerics, amplified in deep layers: yes [M]

- **They are not small-norm artefacts.** The gross pools have ordinary norms (‖x‖, ‖y‖ about 10-14) and ordinary fp8
  scales (2^-7, 2^-6). Their cosine can be about 0.
  - The pooled key is a per-dimension softmax(gate + ape) blend of four member keys, followed by a Hadamard
    transform and fp8 quantisation (`kpool_compress.py` 544-714 [S]).
  - This can turn a small upstream difference into a different vector.
- **The measured nulls carry the same signature** (`out/null_table.json`):

| Comparison | n | Gross | q50 | q999 | Layers 3-15 | Layer 39 |
|---|---:|---:|---:|---:|---:|---:|
| **Null:** prefill vs prefill (identical tokens, build lengths differ) | 106,898 | 0.53 % | 0.027 | 0.67 | 0 | 1.66 % |
| **Null:** decode (batch 8) vs decode (batch 1), cbac17 | 5,489 | 0.11 % | 0.029 | 0.30 | 0 | 0.60 % |
| **Null:** decode vs the same decode later | 106,898 | 0 | 0 | 0 | 0 | 0 |
| H5b: B1 decode vs rebuild | 27,984 | 0.71 % | 0.032 | 1.09 | 0 | 2.12 % |
| H5e: solo decode vs rebuild | 5,489 | 0.18 % | 0.031 | 0.35 | 0 | 0.80 % |
| H4c: V3 decode vs prefill | 4,774 | 1.49 % | 0.036 | 1.03 | 0 | 4.38 % |
| H5d: D2 at 8 in flight | 27,984 | 60.9 % | 0.327 | 1.17 | 54-63 % | 52 % |

- **The fragility is position-specific.** Of the B1 pools gross against the rebuild, **42 %** are also > 0.25 between
  two prefill builds. For non-gross pools the figure is 0.36 % (`out/h5_overlap.json`).
- **Per trace, the decode-vs-prefill rate tracks the prefill-vs-prefill rate:**
  - 34b209: 5.4 % vs 1.0 %;
  - be2e9e: 0.96 % vs 0.84 %;
  - 4b92a6: 0.85 % vs 0.57 %;
  - ddfd28: 1.65 % vs 0.48 %;
  - aa89ec: 0.98 % vs 0.22 %;
  - cbac17: 0 % vs 0.02 %.
- **Decode against prefill differs more than prefill against prefill**, by 1-5× per trace. That is expected: the
  paths share no kernels. The frozen criterion assumed the prefill rebuild is an exact reference to 0.1 %, and it is
  not even an exact reference for itself.
- **KDA shows the same depth profile** (section 4). The decode-vs-prefill output gap equals B1's unhooked null
  (section 5).

### (iii) A genuinely wrong decode writer: no evidence [M][S]

- **The pattern does not follow speculation.** The 'oracle' schedule, with every pool completed from members stashed
  in the same launch, has **0 of 704** gross. The single-advance schedules, with no in-launch member dependency,
  have 7/726 (poison) and 39/726 (reject:1). A stash-then-read race inside `_kpool_decode_update_batched_kernel`
  would invert that.
- **The kernel keeps the members in order.** One program per request iterates its rows in order: completion reads
  first, then stashes. The B1 CPU test D2 already showed decode pools equal compress(own keys).
- **The two decodes do not fail together.** b1 (batch 8) and solo (batch 1) are each gross against the rebuild at
  different pools, and they differ from each other at those pools (relative L2 0.05-0.95). This is
  batch-composition numerics, not a deterministic mis-write.
- **The mapping is verified.** The guard shows 0 `idx_w_dec`, `tail_w_seed`, `s_mism` and `d7_fallback` on both
  ranks over about 2.2 M (rank 0) and 1.9 M (rank 1) checked decode writes.
- **No clustering at boundaries.** Gross pools do not cluster at the ring half (0.70 % vs 0.72 %), the block boundary
  (none in range), the window ends or context ≥ 2,044.
- **An unexplained correlate.** Gross pools sit at positions where the **output** is near-deterministic: median
  max|Δlp| over the members is 5e-5 against 0.037 for non-gross, with similar entropy differences
  (`out/h5_context.json`). They are also at high acceptance, which correlates with that. The mechanism is not
  identified. It is reported as an observation; no defect is inferred from it.

## 3. H5h: why in-graph exposure records are missing

- **Source [S].** On SM120 with next_n = 4, the DSA indexer decode path flattens spec-verify rows. `server.log`
  line 243 reads `use_flattening=True supports_varlen=False (next_n=4)`.
  - `indexer.py` 920-995 rewrites `decode_lens` to all-1s, so the decode branch reaches `on_selection(..., batch_size
    = num_decode_tokens, next_n = 1)`.
  - The hook maps row t to batch row `t // next_n = t` and reads `S.bidx[t]`.
- **Effect:**
  - Only the batch-row-0 request's row-0 token is attributed correctly.
  - Rows t < num_reqs are written to the wrong request, with that request's prompt length, so at a wrong output
    index.
  - Rows t ≥ num_reqs are dropped.
- **Data [M]:**
  - 14,127 of 20,469 designated positions have no record.
  - Of the 6,342 present records, 2,811 carry the measurement step at row 0. The other 3,457 come from a different
    step, so they are misattributed.
  - Positions measured at rows 1-3 are about 85 % missing.
  - Separately, the 206 positions o = 0 (predicted by prefill) and the V3ref/V4-rebuild W = 1 requests can never
    have a decode record. The check should exclude them.
- **CPU reproduction [R]** (`r_h5h_flatten.py`, the real `on_selection` in the hooked image):
  - the native `[B, 4]` layout records 8/8 correctly (as test X2 did);
  - the flattened layout records 1/8 correctly and writes 1 misattributed record (request b, o = 66, from request a's
    row 1).
- **Scope.** The full-selection records (`selbuf`) share the defect. The pattern follows the row index, not the batch
  shape: about 70 % are missing in padded steps and about 68 % in unpadded ones.

## 4. H4b: KDA recurrent state

- **Identification is perfect.** Across 3,604 comparisons (106 snapshots × 34 KDA layers), the committed slot
  `num_accepted − 1` is the slot nearest the prefill-built state **every time**. There is no slot misidentification
  and no spec-row-order fault.
- **The gap grows with depth:**

  | Layers | Median rel L2 | Margin |
  |---|---|---|
  | 0-2 (identical embeddings: the gap is the KDA kernel, recurrent vs chunked) | 0.005 (max 0.009) | 18-83 |
  | 16-18 | 0.03-0.06 | |
  | 33-42 | 0.09-0.13 (max 0.36) | median 1.9-2.5, min 1.04 |

  - At depth, one token's update (the distance from the committed to the nearest scratch slot) is about 0.2. So a
    3× margin cannot be met there by construction.
- **Reading.** The upstream decode-vs-prefill divergence accumulates with depth, as for index-K. The tolerances 0.05
  and 3× had no measured basis.
- **Not a hook effect, and not a KDA kernel defect** [I]:
  - the layer-0 gap is 0.5 %;
  - output logits agree with B1's unhooked null.

  A per-depth null is not measurable from this run, because no two prefill builds share a frontier.

## 5. H2: forced decode against prefill

- **The join defect.** For the block-crossing traces 53fa5b3e066f and 57703691f509, the V2b forced requests start
  the window at the prompt (P = prompt length). The analysis paired them with **V2c**, whose window starts at 3,832.
  - Every position was compared with a different token's row, giving top-1 0.000 on 18,588 positions.
  - In addition, V2b positions beyond V2c's 2,500-token window, where the reference is NaN, were counted as argmax
    disagreements.
- **After the fix.** Re-joining at offset 3,832 − P, which is valid because a causal prefill computes the same rows,
  and counting top-1 only over existing reference rows:

  | Group | top-1 |
  |---|---|
  | All | 0.983 |
  | The four re-joined requests | 0.980-0.991 |
  | Phases | 0.981-0.984 |
  | Decode vs decode (H3) | 0.982 |
  | Historical | 0.972 |

  Quantiles are q50 1.1e-4, q90 0.100 and q99 0.42, against the in-session unhooked null of 6.7e-5, 0.097 and
  0.406. **The hooked decode is at the null.** It is not a long-context decode-vs-prefill difference beyond B1's own.
- **H2b is a weak control.** The forced-token log-probability varies slowly: about 90 % of forced tokens are the
  argmax with p ≈ 1, so |lp(o) − lp(o±1)| has a median of about 0.03 nat.
  - The 1-nat expectation was never measured.
  - Argmax agreement at ±1 is 0.007 against 0.983 aligned, a decisive shift control.
  - Per request, the aligned minimum is 0.951 and the shifted maximum is 0.021.

## 6. H1f: top_p mask outside the eligibility bounds

- **The rows.** 224 cells on 56 rows across 38 requests. All are "kept although not eligible", for all four
  termination ids at once, never the forced token.
  - Every row has fp32 top-1 p = 1.0 and p_post == p_pre: the row was **left entirely unmasked**.
  - Every row is in a batch of ≥ 8 rows, so the Triton path ran.
  - The maximum kept tracked probability is 2.6e-9, and the fp32 tail mass is below 6e-8.
  - 23,695 other fp32-one-hot Triton rows were masked correctly, so 0.24 % are affected.
- **Mechanism.** `topk_topp_triton.py` 594-853 [S]; emulated in fp32 with the GPU tile sizes, `r_h1f_emulate.py` [R].
  - The standalone top-p path computes exp(logit − max_sample), where max_sample = mean + 10·std of the first 8,192
    logits.
  - When the top logit exceeds max_sample by more than about 88.7, the sum overflows to inf and the search is
    skipped, giving final_pivot = inf.
  - The kernel's own `not (final_pivot < max_logit)` guard then **skips masking**.
  - The emulation masks correctly at top − max_sample ≤ 82.5 and reproduces the skip at ≥ 91.3, around the fp32
    overflow threshold ln(FLT_MAX) ≈ 88.7.
  - The real rows' absolute logits are not recorded, so overflow as the trigger is inferred, not proven per row.
  - The Triton interpreter cannot run the kernel on CPU (`row_id.to` on a Python int).
- **Bound computation.** Correct: the PyTorch rule `cumsum ≤ 1 − p` masked, and `elig_hi` = 0 at mass ≤ 1.7e-8.
- **Materiality.** None for sampling or for the termination metrics, below 1e-8.

## 7. H5f and H5g: the ledger and the guard

**H5f (hook defect) [M][R].**
- **Cause.** `d2_tail_mapping` and `note_tail_mapping` store the builder's `qsl[: n_reqs + 1]`, where `n_reqs`
  includes FULL-graph padding requests. `_shadow_step` slices the packed buffer with the real `nr`, so `acc` and
  `idx` are read at shifted offsets.
- **When it happens.** Padded steps are exactly the steps with 3, 5 or 7 requests in flight (12, 20 and 28 tokens
  padded to 16, 24 and 32).
- **Effects:**
  - every "none" pool (326 in B1, 341 in D2) completes in a padded step;
  - in padded steps, attributions shift by one request;
  - the polluted ownership then labels byte-correct B1 pools "corrupt": 1,243 pools, with median relative L2 0.030.
- **CPU reproduction** (`r_h5f_padded.py`): 3 requests, 6 steps.
  - Unpadded, each request's pools are classified under its own key.
  - Padded, request 0 receives request 1's pools, request 1 receives request 2's, and request 2 receives none.
- **Consequence.** In D2, "corrupt" is only 61 % gross, even restricted to all-unpadded pools (62 %). Under D2 the
  ledger still misses other requests' block-0 writes made in padded steps.
- **Untested part of the criterion.** Whether a truly corrupt member always moves the pool by more than 0.25 is
  untested on GPU, because the CPU check D3 used toy keys. The ≥ 95 % rule cannot be judged until the ledger is
  fixed.

**H5g (analysis defect) [M].**
- **Cause.** The guard polls carry `rank` but no `pid`. `by_rank_last` therefore held a single poll, rank 1's last.
- **Per rank:**
  - rank 0: `tail_w_stash` mism = null = 100,098 = 6 × 16,683, from 6 owned indexer layers;
  - rank 1: 83,415 = 5 × 16,683, from 5 owned layers;
  - the counts accrue only in V4-d2 (83,022 and 69,185) and V4-d2solo (17,076 and 14,230).
- **The "other violations" 4.** These are `pos.pos_bounds` = 4 per rank, at host steps 4-6 at t = …390, before the
  first request at t = …396. They come from the runner's fake `_warmup_` batches.
  - The same 4 appears in d7/step2/run-b1-a1 and b0-mapguard run-a2/a3, and is documented in
    `b0-mapguard/DIAGNOSIS-a2.md` section 5.
  - Taking per-phase deltas from the client's "start" dump excludes them.
  - With that, H5g passes.
  - `mla_r.sel_hole` and `inact_read` are non-zero here as in run-b1-a1. They are descriptive counters and not part
    of H5g.

## 8. Recommendation

### 8.1 Harness and analysis fixes (post-inspection; no criterion loosened)

1. **H2 join** (PI-1): re-join V2b-of-CROSS to V2c at offset 3,832 − P, and count top-1 only on finite reference rows.
2. **H2b/H2c power** (PI-2): add the argmax-agreement shift control beside the lp rule.
   - Proposed frozen form for attempt 2: per request, aligned top-1 ≥ 0.90 and shifted ≤ 0.20. The measured values
     are ≥ 0.951 and ≤ 0.021.
   - Keep the lp rule reported, labelled as "uninformative when forced lp ≈ 0".
3. **H5g** (PI-3): read polls per `rank`, and use deltas from the client's start dump.
4. **H5h analysis:** exclude o = 0 and W = 1 requests from the designated-position count.
5. **Window end:** drop pools completing at P + W − 1 from the window comparison.
6. **H1f** (PI-4): report rows the served kernel left unmasked (p_post == p_pre for every tracked id) as a separate
   engine class, with their kept mass. The check stays "no violation with p_post > 1e-6".
   - This is a proposed tolerance, justified by the measured maximum of 2.6e-9.

### 8.2 Hook fixes (require a rebuild, CPU tests, a re-freeze and coordinator verification)

1. **`on_selection`.** Map each selection row to its request through a per-token table.
   - `on_batch` fills a persistent `tok2row[num_tokens]` from `query_start_loc`, which is graph-safe because it is
     filled before the forward.
   - Under flattening the layout is `[B·(K+1), 1]`, so use `tok2row[t]`. In the native layout, keep `t // next_n`.
   - New CPU test X2f: the flattened layout must give 8/8 correct and 0 misattributed.
2. **Ledger.** Store the real request count with `d2_tok`: `n_reqs` and `n_tok` before padding, or slice the packed
   buffer with the stored lengths.
   - New CPU test D7p: a padded step must classify each request under its own key, as `r_h5f_padded.py` checks.
   - Assert that the B1 arm is all-clean in the analysis (a new positive control).

### 8.3 Criteria for attempt 2, to be frozen in an amendment before data

These follow the measured nulls. The original verdict of attempt 1 is reported beside them.

- **H4b:**
  - identification: the committed slot is nearest in 100 % of comparisons;
  - layers 0-2: rel ≤ 0.02 (measured maximum 0.009) with margin ≥ 10 (measured minimum 18.7);
  - deeper layers: reported per depth, not gated.
- **Index-K addressing positive control.** Replace "gross ≤ 0.1 %" with structural checks that actually discriminate
  misaddressing:
  - no page with more than 25 % of its pools gross;
  - 0 gross in layers 3-15;
  - committed pools byte-stable across two snapshot times;
  - end-of-window pools excluded.
- **Index-K content:**
  - B1-arm gross ≤ 2 × an **in-session** decode-path null, measured by a second B1 arm at a different batch
    composition;
  - also reported against an in-session prefill-vs-prefill null (a second rebuild at a different length);
  - the D2 arm against the same nulls (> 30 % kept);
  - the solo D2 run ≤ 2 × the decode null.
  - Per-pool 0.25 remains the reporting threshold, and H5c is retired as a gate.
- **H5f:** the B1 arm all clean (100 %); D2 clean → non-gross ≥ 99 %; corrupt → gross reported against the decode
  null, not gated at 95 % until a measured corrupt-member effect exists.

### 8.4 Attempt 2: warranted, scoped (`attempt2_cost.py`, attempt-1 measured phase times)

**Budget.** Attempt 1 used 1.2117 of 2.4 GPU-hours, leaving **1.188 GPU-hours**. The runner's gate then caps the
wall at **2,138 s**. Fixed overhead is 746 s, covering start-up, readiness, stop and the final dump.

| Scope | Client | Occupied | GPU-hours |
|---|---:|---:|---:|
| Minimal: V0; V2a natural + oracle; V2c; V4 with a second B1 arm and a second rebuild at a different length | 367 s | 1,113 s | **0.62** |
| + V3 snapshots (H4 prospectively) | 465 s | 1,211 s | **0.67** |
| + V2a poison (H3 on the new image) | 625 s | 1,371 s | **0.76** |
| Attempt-1 plan in full | 1,434 s | 2,180 s | 1.21 (**does not fit**) |

**Recommendation: the "+ V3" scope, 0.67 GPU-hours estimated.**
- It re-measures what changed: exposure, selections, ledger and V0 on the rebuilt image.
- It measures the nulls the revised criteria need.
- It confirms H4 prospectively.
- H2 (V1, V2b) and H3 carry over from attempt 1 under PI-1/PI-2, with provenance stated. The forcing code is
  unchanged, and the image differs only in `on_selection` and `_shadow_step`; V0 checks unkeyed equality.
- The client must be a scoped and frozen variant. With the old client the deadline would skip V2b, and coverage
  would read INCONCLUSIVE.
- This is the **last** attempt under the 21c authorization. Run it only after the fixes pass on CPU (X2f, D7p, the
  existing 66 + 16 + 6 suites and I1 inertness), the package hashes are re-frozen, the coordinator verifies them,
  and the post-inspection items and revised criteria are recorded in an amendment.

## 9. To the operator (before the contrast is designed; nothing requires stopping)

1. **No new B1 defect.**
   - The decode writer, tail ring, KDA rollback and forcing are clean on every structural test.
   - The failures were two hook bookkeeping bugs, several analysis joins, and thresholds that had never been
     measured.
2. **Pooled index-K keys are numerically fragile in deep layers (engine property).**
   - The same tokens prefilled at two lengths give about 0.5 % of pools unrelated in layers 19-43, rising to
     1.7 % in layer 39. Decode against prefill gives 1-5× that.
   - **This is the noise floor any D2 or index-K contrast must beat.** D2 at 8 in flight is far above it (61 %).
   - It is also a plausible contributor to the run-to-run volatility seen in amendment 21a, because selection reads
     these keys. That last point is [I] and not measured.
3. **The served Triton top-p kernel skips masking on some near-one-hot rows** (inferred: fp32 overflow).
   - It has no practical effect, below 1e-8 probability.
   - It is a possible upstream report. Any public action needs the operator's permission.

## 10. Files (all in `diagnosis-a1/`)

**Runners.**
- `drun.sh`: a B1 CPU container with run data read-only.
- `drun_hooked.sh`: the same, using the hooked validation image.

| Script | Check | Output in `out/` |
|---|---|---|
| `d_h5_pools.py` | Annotated per-pool V4 comparison | `h5_pools.jsonl` |
| `d_h5_decode_null.py` | b1 vs d2, b1 vs solo | `h5_decode_null.json` |
| `d_h5_context.py` | Step, row and acceptance structure; output divergence | `h5_context.json` |
| `d_h5_overlap.py` | Position-specific fragility | `h5_overlap.json` |
| `d_h4.py` | H4b KDA and H4c pools | `h4b_kda.json`, `h4c_idx.json` |
| `d_h4c_null.py` | Prefill-vs-prefill and decode-stability nulls | `h4c_nulls.json` |
| `d_null_table.py` | Per-layer table | `null_table.json` |
| `d_h5h_exposure.py` | H5h | `h5h_exposure_rows.json` |
| `r_h5h_flatten.py` | H5h CPU reproduction | `h5h_flatten_repro.json` |
| `d_h5f.py`, `d_h5f_ledger.py`, `d_h5f_ledger2.py`, `d_h5f_restricted.py` | H5f | `h5f_*.json` |
| `r_h5f_padded.py` | H5f CPU reproduction | `h5f_padded_repro.json` |
| `d_h5g_guard.py` (host) | H5g | `h5g_guard.json` |
| `d_h2.py`, `d_h2_corrected.py` | H2 | `h2_per_request.json`, `h2_corrected.json` |
| `d_h1f.py` | H1f | `h1f_rows.json` |
| `r_h1f_emulate.py`, `r_h1f_triton.py` (interpreter attempt; fails on CPU as noted) | H1f mechanism | `h1f_emulation.json` |
| `attempt2_cost.py` | Attempt-2 price | `attempt2_cost.json` |
| `postA1_verdict.py` | Original vs post-inspection | `verdict-postA1.{json,txt}` |

**Other.**
- `x_*.py`: exploratory summaries.
- `engine-src/`: copies of the installed `topk_topp_triton.py`, `topk_topp_sampler.py` and `kpool_compress.py`
  (read-only references).
