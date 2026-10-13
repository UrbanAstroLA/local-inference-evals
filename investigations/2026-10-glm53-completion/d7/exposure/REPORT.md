# D7 exposure accounting (PLAN.md amendment 18, step 0 "GPQA exposure")

CPU only. No GPU, no container, no change outside `d7/exposure/`. Outputs hold doc ids, seeds, hashes, token counts,
offsets, times, classes and empty flags only; no question, answer or reply text. (Text written by the exposure agent;
saved to this file by the coordinator because the agent could not write report files.)

Labels: [M] measured from a recorded artefact; [R] reconstructed, deterministic from measured inputs through arithmetic
verified on the measured run; [E] estimated, depends on varied unrecorded parameters, with bounds given; [I] inferred from
source reading or reasoning, not measured.

## 1. The simulator's 73 predicted versus 217 measured events in attempt 3

### What was measured [M]
Source: `extract_a3.py` → `a3_measured.json`.
- The scheduler stream (`ms_p274.jsonl`, 2,482 `schedule()` records) contains 218 prefill chunks that start at an
  unaligned position.
- One of them is a 4-token prompt tail (step 1315). With n ≤ 1+K it takes the indexer's decode path
  (`decode_threshold = next_n`, indexer.py lines 737-738), and the guard did not flag it.
- That leaves **217 exposure events**: step×request events in 50 requests, all joined one-to-one to the guard's
  `idx_w_pf` records on both ranks. Rank 0 holds 1,302 records (217 × 6 layers) and rank 1 holds 1,085 (217 × 5).
- Subtypes: 186 `missing_pred` and 31 `pool_cross`.
  - Rule, 217 / 217: an event is `pool_cross` exactly when another prefill-classified row precedes it in the worker's
    persistent-batch order.
  - Scheduler order does not predict the subtype: only 17 events had a prefill row ahead in scheduler order, against 31
    `pool_cross`.

Events by phase (requests with events in brackets):

| Phase | Events | Solo budget | Shared step |
|---|---|---|---|
| Stack release warm-up | 10 (2) | 10 | 0 |
| warm | 108 (8) | 108 | 0 |
| T0 / T1 | 20 / 20 | 3 / 3 | 17 / 17 |
| F0 / F1 | 9 / 9 | 3 / 3 | 6 / 6 |
| D | 22 | 2 | 20 |
| S | 19 | 3 | 16 |
| Q, P0, P1 | 0 | 0 | 0 |
| **Total** | **217** | **135** | **82** |

- The 82 shared-step events had decode rows ahead (40), prefill rows ahead (5), or both (37).
- Chunk start mod 4,352: 2,045 in 74 events, 4,090 in 61, the rest scattered.

### What the frozen simulator modelled [I], from `reach/sim_reach_mapguard.py`
- It reported 73 events: warm 0, T0 12, F0 10, T1 12, F1 10, D 12, S 17, Q 0. Same unit as above (step×request).
- Its rules versus B0's:
  - budget: flat 2,048 with no per-request draft-slot reservation, versus 2,048 − K per scheduled request with K = 3
    (solo budget 2,045);
  - decode row: costs 4, versus 4 + 3 = 7;
  - Mamba align: no clip at 4,352 boundaries, versus a clip;
  - row order: decodes first, versus running requests in admission order and then waiting requests (FCFS);
  - admission: instant, in client send order, versus engine arrival order, which the frontend reorders;
  - decode progress: fixed 2.6 accepted tokens per step;
  - workload: no stack warm-up traffic, no P0/P1, no S-phase offsets.

### Decomposition [R] (closed loop on attempt 3's exogenous inputs)
- **Frozen rules: 80 events, 0 real.** All of them come from leftovers in budgets that are multiples of 4.
- **Plus Mamba clip: 59.**
- **Plus draft-slot reservation: 217, all real.**
- **Plus real row order: 217.**
- **Corrected rules without the Mamba clip: 206** (19 real).
- **Corrected rules without the reservation: 59.**
- **Corrected rules with client send order: 230** (147 real).

The reservation decides the outcome. With it, every long solo prefill chunks as 2,045 | 2,045 | 262 per block, starting
at offsets 0, 2,045 and 4,090. The cold warm phase alone contributes 108 events.

### Corrected simulator (`sim_d7.py`)
- B0's scheduler.py is byte-identical to stock 0.9.1's; `sim_d7.py` re-implements its budget arithmetic in the standard
  library.
- **Replay [R]:** 14,712 / 14,712 rows exact, and no running prefill was left unscheduled while budget remained.
  - The replay also confirms three settings from data: partial-tail Mamba stops are off in B0 (a 4,240-token prompt
    chunked 2,045 | 2,045 | 150); decode rows always take 4 tokens; prefix hits are whole 4,352-token blocks.
- **Closed loop [R]** on exogenous inputs (measured step grid, arrival bounded by client send time + 0.07 s and by the
  measured first scheduling, measured engine arrival order, prompt length, hit, decode-step counts):
  - 217 / 217 events exact: same step, start and length;
  - 481 / 481 prefill chunks exact;
  - every phase matches exactly;
  - the result is unchanged for any frontend delay from 0 to 1 s.
- **Workload-only prediction [E]:** step-time fit 0.067 s + 0.27 ms per prefill token + 1.5 ms per decode row (RMSE
  0.13 s), client send order, acceptance 2.6-3.2, step-time scale ×0.7-1.4. Result: 224-227 events, of which 133-138 are
  real chunks.
- Not reproduced: the `missing_pred` / `pool_cross` subtype, which depends on batch row order.

## 2. Exact GPQA prompt-token counts (`render_tokens.py` → `gpqa_prompt_tokens.json`)
- The tokenizer is a pure-Python implementation of K3.25 `tokenizer.json` (sha256 19e77364…; added tokens, Split regex
  with Unicode classes, ByteLevel, BPE with `ignore_merges`).
- Rendering: jinja2 sandbox, `add_generation_prompt`, `enable_thinking: true`, no BOS.
- Templates: zai (0c4099f3…) for the 0.9.1 family, checkpoint 701cd745 (34d5ee66…) for 0.7.0. The two give identical
  ids.
- **Cross-check [M]:** 1,980 / 1,980 logged `prompt_tokens` equal (0.7.0 `+ps` 396; 0.9.1 198 + 396; col511 198 + 396;
  bundle 198 + 198).
- Lengths: minimum 70, median 204, maximum 2,795 (doc 127); next 887, 766, 619. Docs 0-7 total 1,714. Only doc 127
  exceeds 1,088 or 2,045 tokens.

## 3. GPQA exposure (`gpqa_exposure.py`)
**Chunking.**
- 0.9.1 family: K = 3, so a solo prefill gets 2,045, a decode row costs 7, and rows with n ≤ 4 take the decode path.
  Verified on B0 [R].
- 0.7.0: K = 5, so 2,043, 11 and n ≤ 6 [I]. Basis: its scheduler differs only in one eagle back-off line that is
  irrelevant for prompts under 4,352 tokens; `max_num_new_slots_for_drafting` is identical; DCP 2; adaptive MTP off.
  **Conditional on 0.7.0's writer sharing the assumption.**
- No Mamba clip and no prefix hits for GPQA prompts [M: P1 repeats showed cached 0].

**Admission data [M].**
- Each pass sends docs 0-7 together, then dispatches in doc order (1-4 inversions per pass).
- Concurrency stays at 8 or below, with 0 preemptions.
- Mid-pass co-sends within 1 s, over 1,900 requests: 0 for 1,822, 1 for 72, 2 for 6.
- At doc 127's send, in all 10 logged passes: 7 others in flight and no transition within ±1 s.

**Method.**
- Per-step budgets are not uniquely recorded, so two methods are combined.
- Analytic enumeration: every co-send within 1 s is a prefill row ahead, an earlier decode row, or absent; requests in
  flight are decode rows, and those ending within 1 s may already have left; long-prompt remainders run ahead. Each case
  uses the sequential scheduler arithmetic.
- Ensemble [E]: 160 schedules per logged pass.
  - step duration 0.5-2 × the mean decode step, which is 50.4 ms for 0.9.1 and 90.5 ms for 0.7.0, from the server-log
    means;
  - 4 grid phases;
  - frontend delay of 5, 30 or 80 ms, or random 5-350 ms;
  - engine finish 0 or 100 ms before the proxy end.
- Classes:
  - KNOWN: every admitting configuration gives an exposed start;
  - POSSIBLE: some configuration does;
  - NOT-unsplittable: the prompt fits in every configuration;
  - NOT-aligned-split: always chunked, always at a multiple of 4;
  - UNKNOWN: no timing.

**Results.**
- 0.9.1 family, 8 passes: per pass 197 NOT-unsplittable and doc 127 NOT-aligned-split. Its chunks are [0, 1,996) and
  [1,996, 2,795); 0 exposure events in all schedules.
  - The first chunk by number of decode rows d: d = 0-7 gives 2,045, 2,038, 2,031, 2,024, 2,017, 2,010, 2,003, 1,996.
    It is aligned only at d = 3 or 7.
- 0.7.0, passes 2 and 3: doc 127 KNOWN, chunked at 1,966 (pool covering positions 1,964-1,967, a `missing_pred` type
  under the B0 writer [I]); 1 event in all 160 schedules; 197 NOT.
  - Its first chunk is unaligned for d ∈ {0, 2, 3, 4, 6, 7}. The conclusion holds with pool modulus 8 as well.
- 0.7.0, pass 1 (no log): 82 NOT under the conservative bound and 116 UNKNOWN. Of those 116, 115 cannot be split under
  the measured co-send ceiling of 2. Doc 127 is exposed unless d ∈ {1, 5}.
- **Shared-step exposure is nil in these records.** No plausibly co-admitted set can exceed the budget. A request behind
  doc 127 is not admitted in that step, because doc 127 leaves exactly 0 budget.
- **Consistent with attempt 3 [M]:** the Q phase (docs 0-15, concurrency 8) had 0 events.

## 4. Exposure class versus empty (empty/answered; descriptive only)

| Record | Pass | Classes | Empty |
|---|---|---|---|
| 0.7.0 | p1 | NOT 0/82, UNKNOWN 0/116 | 0 |
| 0.7.0 | p2 | KNOWN 0/1, NOT 3/194 | 3 |
| 0.7.0 | p3 | KNOWN 0/1, NOT 1/196 | 1 |
| 0.9.1 | p1 | aligned-split 1/0, NOT 4/193 | 5 |
| 0.9.1 | p2 | aligned-split 0/1, NOT 5/192 | 5 |
| 0.9.1 | p3 | aligned-split 1/0, NOT 5/192 | 6 |
| col511 | p1 | aligned-split 1/0, NOT 3/194 | 4 |
| col511 | p2 | aligned-split 1/0, NOT 4/193 | 5 |
| col511 | p3 | aligned-split 1/0, NOT 5/192 | 6 |
| bundle | p1 | aligned-split 1/0, NOT 2/195 | 3 |
| bundle | p2 | aligned-split 0/1, NOT 3/194 | 3 |

- Pooled: 0.7.0 4/594, 0.9.1 16/594, col511 15/594, bundle 6/396.
- The doc 127 rows, with finish reason and token counts, are in `exposure_tables.md`.
- No test was run and no causal reading is drawn. Doc 127 is not taken as bounding any effect, and no 0.9.1-family
  request had an exposed start.

## Limits
- **Attempt-3 agreement is conditional on exogenous inputs.** The exact match assumes the measured step grid, arrival
  order and decode counts; without them the estimate is 224-227.
- **Event subtype is not modelled.**
- **0.7.0 is inferred.** Its arithmetic comes from source and config, not serving; exposure under 0.7.0 is conditional
  on its writer and DCP 2 semantics.
- **The n ≤ 1+K decode-path rule rests on one measured instance plus the source.**
- **GPQA engine timing is unrecorded.** It is covered by the enumeration and the ensemble; the classes would change only
  if d at doc 127 had differed from 7, which the logs exclude.
- **The tokenizer is a re-implementation**, validated exactly on 1,980 requests.
- **Not included:** K4 records, the pilot, and the 0.8.0 and 0.9.0 passes.
