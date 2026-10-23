# Per-version dispatch table: stock 0.7.0, stock 0.9.1, B1 (recon, 2026-10-22, CPU only)

**Evidence labels:**
- **O**: observed in served logs of that version;
- **OD**: observed on an instrumented derivative image, not the stock image;
- **S**: read in that version's source (`src/`, copied from the image by digest with `docker create`/`docker cp`;
  hashes equal the 2026-10 extractions);
- **H**: CPU harness measurement;
- **I**: inferred (by combining O and S, not observed per step).

Per-step dispatch was **never logged on a stock image**. Every per-step cell below for stock 0.7.0 and 0.9.1 is
S + I.

**B1 source:** B1 = stock 0.9.1 plus five ports, which touch only `indexer.py`, `mamba_hybrid.py`,
`sparse_attn_indexer_kpool.py`, `gate_linear.py` and `exl3.py` (`upstream-drafts/v2/recheck-output.txt`). Its shape,
graph and padding logic is 0.9.1's (S). B1's mapguard step streams are therefore OD evidence for 0.9.1's padded shapes
and graph modes.

## 1. Occupancy: where long replies actually run (O; time window I)

Long replies run at full occupancy. Share of each request's post-100k time spent at 8 in flight, client side
(`observed.json`, `tail_exposure`):

| Session | Requests past 100k | At 8 in flight | Capped replies: median mean in-flight |
|---|---:|---:|---:|
| 0.7.0 record p2-3 | 14 | 0.86 | 7.9 |
| 0.7.0 rep070 s1 / s2 | 6 / 23 | 0.91 / 0.94 | 8.0 / 8.0 |
| 0.9.1 record p1 / p2-3 | 7 / 18 | 0.91 / 0.78 | 7.6 / 8.0 |
| 0.9.1 D / D2 | 10 / 12 | 0.82 / 0.70 | 7.9 / 7.7 |
| B1 s1236 | 9 | 0.89 | 8.0 |

- **Correction to `audits/phase1-dispatch-storage.md` Q1:** its inference that long unanswered replies "spend their
  final stretch with few requests left" does not hold for most capped replies. The decisive cell is 8 in flight.
- **Window caveat:** the past-100k window assumes a constant decode rate within each request (I).
- **Client side, not engine side:** the in-flight count is the number of outstanding client requests. It is not the
  number of requests in a given uniform decode step: prefill, mixed and waiting steps differ (correction, 22an).
- **Throughput (correction, 22an):** the figure first reported here as decode speed past 100k equals T / secs, the
  whole-request average (`(T − 100000) / (secs × (1 − 100000 / T)) = T / secs`). It is not a measurement after 100k
  and does not corroborate faster decoding inside late loops. The field is now `whole_request_tok_s`.

## 2. Decode verify step at n requests in flight

Verify rows = n × (K+1). Captures (O): 0.7.0 [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96];
0.9.1 and B1 [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]. Graph mode FULL_AND_PIECEWISE on both (O). Uniform decode
batches replay FULL graphs at the padded size (S; OD for B1/0.9.1: `gate/RESULT.md` replay counts, mapguard
`nrows_pad`).

| n | 0.7.0 rows → padded | 0.9.1/B1 rows → padded | Indexer top-k route (S; `source-review-lc/thresholds.out`) | **MoE plan (S + O plan config; I per step)** | One-shot all-reduce ≤ 384 KB = 48 rows (S) |
|---:|---|---|---|---|---|
| 1 | 6 → 8 | 4 → 4 | both fused | both Trellis decode plan (block_m 8) | both |
| 2 | 12 → 16 | 8 → 8 | both fused | both Trellis | both |
| 3 | 18 → 24 | 12 → 16 | **0.7.0 tiled**; 0.9.1 fused | both Trellis | both |
| 4 | 24 → 24 | 16 → 16 | **0.7.0 tiled**; 0.9.1 fused | both Trellis | both |
| 5 | 30 → 32 | 20 → 24 | both tiled | both Trellis | both |
| 6 | 36 → 40 | 24 → 24 | both tiled | **0.7.0 prefill plan (block_m 64)**; 0.9.1 Trellis | both |
| 7 | 42 → 48 | 28 → 32 | both tiled | **0.7.0 prefill plan**; 0.9.1 Trellis | both |
| 8 | 48 → 48 | 32 → 32 | both tiled; 0.7.0 then DCP merge across ranks | **0.7.0 prefill plan**; 0.9.1 Trellis | both (0.7.0 exactly at the limit) |

**MoE plan:**
- Both servers log "EXL3 rank-sliced runtime planned: Trellis m=1..32 block_m=8, prefill trellis block_m=64" (O: 0.7.0
  `+ps.server.log:238`, 0.9.1 `+v0.9.1+ps.server.log:236`).
- The source sends m in 1..32 to the Trellis plan and m > 32 to the prefill plan (S: v070 `exl3.py:3099-3127`; v091
  `exl3.py:3188-3215`). The same rule is in both versions; only the row count differs.
- Inference: 0.7.0 decode at 6-8 in flight, about 88% of its time, runs routed experts through the prefill plan,
  while 0.9.1 decode never does.
- **Kernel schedule (S):** both plans build the mixed-Trellis W4A16 kernel with `schedule_whole_tiles=True`
  (`b12x/moe/_shared/kernels/w4a16/mixed_trellis.py:1913, 2186`; `kernel.py:1533-1538`). One CTA computes each output
  tile over the full K, so the split-K path is not used by either plan.
- **Differences (S; corrected 22an):** both plans use packed routes (direct routes are disabled for projection-mixed
  EXL3, `_MIXED_TRELLIS_DIRECT_ROUTE_LIMIT = 0`). The decode plan compiles an exact launch per width 1..32 (tile
  K128/N128 at 10-32 rows); the prefill plan compiles one 1,024-wide launch (tile K64/N256, block_m 64). At 8 in flight:
  0.9.1 runs an exact 32-wide K128/N128 block-8 launch; 0.7.0 runs the 1,024-wide K64/N256 block-64 launch
  (`b12x/moe/fused_moe/_impl.py` `_trellis_exact_launch_token_counts`, `_projection_mixed_tile_config`). Details:
  `../next/MOE-SWITCH-FEASIBILITY.md` (local).
- **Not established:** whether the two plans give bitwise-equal outputs for the same rows (unmeasured).

**Other kernel families per version:**

| Family | 0.7.0 | 0.9.1 / B1 | Label |
|---|---|---|---|
| Experts | EP2: 144 of 288 experts per GPU, replicated input, partial all-reduce; B12x MoE policy fallback fired | TP2 rank-sliced (half of each expert's intermediate per GPU) | O |
| Sparse-MLA decode | old `run_decode` 96 rows, forced 32 splits; DCP2 (half the KV per rank, LSE combine) | `rows64.w2048.rec528`, splits 32/32; MLA layer ownership `split:25` (rank 0 layers 3-23, rank 1 27-43) | O / S |
| Query absorption | B12x H64 fused at 2-16 rows, else bmm (576) | bmm (512) | S |
| Top-k merge | fused route: serial last-CTA; tiled: per-rank then DCP merge (`run_row_topk` over 1,024) | fused: cooperative; tiled: single rank | S |
| Router top-k kernel | block kernel at ≤ 1,024 rows | same | S (`e1-kvwriter`) |
| All-reduce transport | B12x one-shot, old pull protocol | B12x one-shot, TP2 remote push | O / S |
| KV record / pool | 656 B; 2,758,919 tokens; peak usage 85.8% | 528 B NOPE; 4,707,515 tokens; peak about 41% | O |
| Draft KV | own pages (sharing off under DCP2) | slot-shared in MLA pages, 1,088-token blocks | O |
| Index-K tail ring | ring 4 < verify window 6 (P2), no positions (D2) | ring 8, no positions (D2); B1 fixed | S |
| D2 wrong decode-built pools at 8 in flight | 100% | stock 78-90% (by acceptance); B1 0% | H (`defect-repros/D2-kpool-tail-ring/two-rank/`) |
| Sampler, rejection sampler, KDA code, stop rules | identical bytes | identical bytes | S (README config diff) |

## 2a. Causal-length source for multi-token verify rows (S; read in code, not measured; added 22an)

| | Stock 0.7.0 (DCP2) | Stock 0.9.1 (DCP1, no `positions` in the metadata) | B1 / stock + port 01 (DCP1, `positions` passed) |
|---|---|---|---|
| Per-token cache length on verify rows (B12x sparse-MLA builder) | from the CPU `seq_lens_cpu_upper_bound`, converted to DCP-local lengths | from the CPU `seq_lens_cpu_upper_bound`: `context = bound − q_len`, lengths `context + 1 … context + q_len` | `positions + 1` |

**What the bound is (S):**
- It is the scheduler's `num_computed_tokens` plus scheduled tokens (v091 `model_runner.py:1370-1378`).
- Under async scheduling the scheduler's count assumes the previous step's drafts were accepted. The builder's comment
  calls the bound "an optimistic CPU sequence-length bound … not a valid causal mask for multi-token verification"
  (v091 `b12x_mla_sparse.py:1439-1445`). The same branch is in v070 (`:1233-1239`).

**Inferred, not measured:**
- Without positions, verify-row lengths can exceed `positions + 1` by the previous step's rejected-draft count. At the
  served per-position acceptance (0.755 / 0.579 / 0.454) that count is positive in about half of decode steps.
- The decode kernel receives these lengths as `cache_seqlens` for verify batches (`:3178-3181`).
- Attention reads only indexer-selected entries, so a longer length matters only if a selection at or beyond a row's
  causal position can occur. That is the `sel_causal` invariant, checked on B0 / B1 (positions path) and never on a
  stock image.

**Scope note:** passing `positions` (port 01, tpurtell PR #7) changes this source as well as the kpool tail ring.

## 3. Context and pool-length thresholds (S; `thresholds.out`)

| Threshold | 0.7.0 (DCP2, local pool = half the tokens) | 0.9.1 / B1 (DCP1) |
|---|---|---|
| Tiled top-k: 2nd / 3rd slice live from | 131,072 / 262,144 tokens | **65,536 / 131,072** |
| Tiled supertile chunk 2 / 3 live from | 262,144 / 524,288 | **131,072 / 262,144** |
| Fused-route local trims (low n only) | from 94k (n=1), 45k (n=2) | from 96k (n=1), 47k (n=2), 22.5k (n=3-4) |
| KDA align migrations per 327,680-token reply | 91 (block 3,584) | 75 (block 4,352) |
| Prefill chunk / unaligned starts (D7) | 2,043 tokens; chunk % 4 = 3 | 2,045; chunk % 4 = 1 |

Past 100k, at 8 in flight (the tiled route), 0.9.1 enters a third tiled slice and a second supertile chunk at
131,072 tokens; 0.7.0 enters its second slice there. The K-selection replay (`k-selection/`, model-free) found B1's
selection exact to 331k. Stock 0.9.1 differs from B1 in this path only by D1 (column 511), which is fixed in B1.

## 4. Acceptance geometry (O; `observed.json`)

Per-position draft acceptance, aggregated from each session's `SpecDecoding metrics` lines (weights = draft blocks):

| Workload | 0.7.0 (×5) position 1 / 2 / 3 | 0.9.1 (×3) position 1 / 2 / 3 | B1 (×3) |
|---|---|---|---|
| GPQA records and fresh passes | 0.710-0.733 / 0.50-0.54 / 0.36-0.41 (3 sessions) | 0.755-0.780 / 0.58-0.62 / 0.45-0.50 (4 sessions) | 0.783 / 0.622 / 0.489 |
| B+ continuations (12 0.7.0-built prefixes) | 0.786 / 0.602 / 0.467 | 0.800 / 0.630 / 0.500 | 0.806 / 0.641 / 0.513 |
| E continuations (13 0.9.1-built prefixes) | 0.780 / 0.593 / 0.450 | 0.804 / 0.639 / 0.509 | — |
| Phase 1 step 2 (0.9.1, concurrency 1) | ×5: 0.723, 0.690 | ×3: 0.731 | — |
| Phase 1 step 3 (0.7.0 ×5) | 0.703, 0.699 | — | — |
| Phase 1 step 4 (16k, same prompts) | 0.633, 0.680 | 0.690, 0.697 | — |
| Step 5: 0.7.0 + kpool fixes (D2 kept) | 0.646, 0.650 | — | — |
| Step 6: 0.9.1 + 0.7.0's DCP2/EP2 layout (×3) | — | **0.651, 0.664** | — |

- **Draft depth accounts for part of the gap.** At matched depth and workload (steps 2-3, ×5), 0.7.0 and 0.9.1
  overlap.
- **Layout and D2 track the rest.** At matched depth ×3 and the same prompts (step 4 against step 6), moving 0.9.1 to
  0.7.0's layout lowers position-1 acceptance by about 0.035, to 0.7.0's level. Under that mapping D2 corrupts 100%
  of decode-built pools at concurrency 1 (H), against 0% for 0.9.1 as shipped. Steps 5 and 6 also carry 0.7.0-level
  decode-vs-prefill disagreement.
- **Caveats:**
  - Acceptance also depends on content: loops raise it, and capped sessions run faster.
  - GPQA sessions differ in their mix of long replies.
  - B0 sessions (D2 fixed) read 0.735-0.757, which does not fit a simple "fewer corrupt pools, higher acceptance" rule
    under GPQA's content noise.
  - Only the matched-prompt Phase 1 steps and the matched-prefix B+ and E sessions are like for like.

## 5. What differs in the cell where long replies run (n = 8, past 100k)

1. Routed-expert launch: 1,024-wide prefill launch, K64/N256, block_m 64 (0.7.0) against exact 32-wide decode launch, K128/N128, block_m 8 (0.9.1). Both packed routes and whole tiles. I per step; whether outputs differ is unmeasured.
2. Expert layout (EP2 replicated against TP rank-sliced) and its partial-sum all-reduce. O.
3. Sparse-MLA decode kernel family and KV split (DCP2 LSE combine against layer-owned DCP1). O/S.
4. Verify width 6 against 4, draft depth, and draft KV placement. O.
5. Decode-built index-K corruption: 100% against 78-90% (H). B1: 0%.
6. Tiled top-k slice and supertile geometry past 131k tokens. S; selection replay exact on B1.

Not different: sampler, rejection test, KDA code, stop rules, router kernel branch, graph mode class, all-reduce class
(one-shot).

## 6. Coverage gaps in this table
- **No per-step observation on either stock image:**
  - padded size, graph replay and route are inferred for 0.7.0;
  - OD only for 0.9.1 via B1 mapguard and `gate/`.
- **Drafter (DFlash2) graph replays per step:** not observed on any image.
- **MoE plan per step:** not observed on any image. The verify13 plan-branch counts are prefill-side only.
- **Kernel numerics of the two MoE plans on identical rows:** not measured.
- **0.7.0 pass 1 (2026-09-29):** no server log. Its configuration is assumed equal to passes 2-3.
