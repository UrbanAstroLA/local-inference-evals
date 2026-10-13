# Proposal: forced-decode termination probe, first GPU stages (PLAN.md amendment 20, 20a-20c)

2026-10-13. **Draft for independent review. Nothing here has run on a GPU.** CPU-only preparation: source reading, a
CPU feasibility check inside the 0.9.1 and 0.7.0 images (`--network none`, no `--gpus`), fixture selection and cost
arithmetic. No image was built and nothing under `<stacks>` was touched. This file holds no GPQA or reasoning
text: ids, positions, metrics and hashes only.

Inputs: PLAN.md amendments 19f-20c; `tail-audit/AUDIT.md`; `source-review-lc/SOURCE-REVIEW.md`;
`fixture-inventory/{INVENTORY.md, TERMINATION.md, inventory.json}`; coordinator additions on draft depth, D6 and
recurrent-state rollback (2026-10-13, relayed from the engine's author).

## 0. Summary

- **Feasibility.** No existing route reaches the decode path. All engines run Model Runner V2 (DFlash2 forces it),
  and V2 refuses custom logits processors. Prompt logprobs score prefill only. `logprob_token_ids` neither forces
  tokens nor is served by the V2 rejection sampler. A small engine hook is feasible on every arm from source: one
  module plus marked lines in three files that are byte-identical across 0.7.0, 0.9.1, B0 and B1. It runs after the
  real (rejection) sampler and overrides the committed tokens with the history's own. A draft is accepted iff it
  equals the history token, and committed state advances only by history tokens. The logic passed 14/14 CPU checks
  in the 0.7.0 image. In the 0.9.1 image it passed 13/14: the one failure is the plugin delivery route, which that
  image's `VLLM_PLUGINS` allowlist blocks, so the hook ships as file hooks instead
  ([`feasibility/FEASIBILITY.md`](feasibility/FEASIBILITY.md)).
- **Core measurement.** Prefill a fixed history prefix, then force 8,192 history tokens through the real decode and
  verify path at 8 requests in flight. Record, at every window position, the termination-token probabilities before
  masking, the tail mass below each, the post-top_p probabilities, the forced token's log-probability, argmax and
  entropy. The same end position is also scored by prefill.
- **Arms.**
  - Stock 0.7.0 x5 (old shipped computation in the window).
  - B0 x3 (corrected; GPQA-served depth).
  - B0 x5 (corrected; depth-matched to 0.7.0).
  - B0-minus-D2 x3: a diagnostic that isolates D2. Not a remedy.
  - Optional B0-ring4 x5: a diagnostic that isolates D6. Not a remedy.
  - Stock 0.9.1 x3 as a context arm.
- **Fixtures.** Eight long windows ending at 69k-125k: four natural closes from stock 0.7.0 and stock 0.9.1
  histories, the phase-2 r0 exhaustion, and two doc-79 pairs. Eight short full-replay controls (four with exact ids).
  Stage 2 adds sixteen more long windows. Every window ends below 131,072 tokens.
- **Stages, each separately authorized** (occupied GPU-hours, estimate / ceiling):

  | Stage | Content | Estimate / ceiling |
  |---|---|---|
  | 0 | Hook, builds, CPU tests, freeze | 0 |
  | 1 | Index-K bytes, harness validity, repeatability; four arms | 7.7 / 10.6 |
  | 1-min (alternative) | Without repeats | 5.0 / 7.0 |
  | 1b (optional) | D6 diagnostic | 1.7 / 2.6 |
  | 2 | More fixtures, composition, oracle-draft bracket; three arms | 7.6 / 10.4 |
  | 2c | Stock 0.9.1 context arm | 3.2 / 4.6 |
  | K | Model-free selection replay, which gates any extension past 131,072 | 0.3 / 0.5 |

- **Readings frozen here** (section 8):
  - Repeatability is established before any engine comparison is read.
  - The amendment-20 section-5 categories apply.
  - D2 (or D6) is *nominated*, not proven, when its diagnostic reproduces the old-versus-corrected difference.
  - No claim is made that a probability difference explains completion rates.

## 1. What this probe can and cannot answer

**It asks** whether, on identical active-reasoning token histories at 64k-125k, the engines assign different
termination probabilities when the window is computed by their real decode path. It also asks whether such a
difference appears only in prefill scoring, and whether a decode-only index-K corruption (D2, D6) reproduces it.

**It cannot answer:**
- Whether these differences cause the completion gap. The 0.7.0 advantage remains a hypothesis (amendment 20c: 10/14
  against 10/25 at ≥ 100k, wide interval, no range or change point).
- Anything about state accumulated over 100k tokens of generation. In a window, only W/4 = 2,048 of 16k-30k pools are
  decode-written, and the KDA state at the window start comes from prefill. The windows therefore under-represent
  D2 and D6 relative to a fully decode-built context. Section 6, Stage 3, prices a whole-context replay.
- Anything past 131,072 tokens, before Stage K.

A lower probability of finishing is not a higher probability of looping (amendment 20b). Every fixture carries its
class at the window (loop, exhaustion, natural close). All fixtures chosen are non-repeating at their window
(repetition indicators at the nearest checkpoints, `fixtures/fixtures.json`).

## 2. Feasibility, per engine (detail in `feasibility/FEASIBILITY.md`)

| Route | 0.7.0 `48e254d9` | 0.9.1 `a9e985c0`, B0 `08d1cb90`, B0-minus-D2, B1 `799d3cf8` |
|---|---|---|
| Custom v1 logits processor (record, then force) | No: V2 runner refuses it (`_validate_v2_model_runner`; check E1); DFlash2 forces V2 (E2) | No (same code) |
| Prompt logprobs | Prefill only; cannot see D2 or D6 | Same |
| `logprob_token_ids` | Reports, does not force; completions sets `logprobs=None` (E3c); not wired into `RejectionSampler` | Same |
| Grammar forcing | Text-level, several tokenizations allowed, applies after the reasoning close | Same |
| **Engine hook after the real sampler** | **Feasible** | **Feasible**; identical files |
| Hook delivery as a general plugin | Works (E4) | Blocked by image env `VLLM_PLUGINS=lora_filesystem_resolver` (E4, E4b). Hook lines in copied files instead, as in the guard images |

**Where forcing sits relative to rejection sampling.**
1. The real `RejectionSampler` runs unchanged: penalties, temperature and top_p, then rejection with seeded noise.
2. The hook then replaces the result for forced requests. Draft `j+1` is accepted iff it equals the history token
   predicted at row `j`. The tokens committed are history tokens `f_0..f_a`.
3. `num_sampled` and `num_rejected` follow from this. `post_update`, KDA rollback (`postprocess_state`), DFlash2
   proposal and scheduler accounting consume the overridden values.
4. Rows conditioned on a rejected draft are discarded, so each position is measured exactly once.

**What differs from real sampling:**
- **Acceptance rule.** Acceptance is "draft equals history" rather than probability p(draft).
  - Matching drafts are accepted, so forcing does not reject systematically.
  - A counterfactual or re-tokenized history is matched less often than the engine's own samples would be. The
    reject-and-redo path, where D6 lives and where D2's ring rewrites happen, is therefore exercised more than in
    serving.
  - Per window the hook measures this as accepted length, per-position acceptance, and the count of rejected
    pool-completing drafts (`pos % 4 == 3`). These are compared with serving: mean acceptance length 2.8 at x3 and
    3.2-3.4 at x5 (`SpecDecoding metrics`).
  - Stage 2's oracle-draft bracket replaces the drafts with the next history tokens (acceptance near 100%) to give
    the low-rejection extreme.
- **History tokens outside the top_p nucleus.** These states would never be reached in serving. They are flagged and
  counted.
- **Prefill-built prefix.** See section 1.
- **Co-runners.** They are fixtures or sampled background prompts, not GPQA questions.

**Recurrent-state rollback.**
- Two rollback paths exist in every tree: baseline full-state rollback and ReplaySSM (`--use-replayssm`, identical
  KDA kernel files in 0.7.0 and 0.9.1).
- Every GPQA record used here ran baseline. The DFlash2 profile defaults `USE_REPLAYSSM=0`, no server log carries
  `use_replayssm`, and the B1 start log states "baseline full-state rollback". ReplaySSM is reachable only by an
  explicit override.
- Neither path is referenced by any index-K, tail-ring, kpool or MLA file.
- Both consume the same `num_accepted` that the override sets.
- The probe uses the served baseline mode. ReplaySSM is out of scope and untested.

## 3. Measurement design

**Requests.**
- `/v1/completions` with token-id prompts.
- Sampling as in GPQA: T 1.0, top_p 0.95, an explicit per-request seed (recorded; it does not affect forced tokens),
  no stop strings.
- `max_tokens` = W + 1.
- `vllm_xargs.local_forcewin` = the forced ids (W + 1 tokens: the window plus the history's own next token at the
  end, which is `</think>` for close windows).
- A per-request `cache_salt`. Prefix caching stays on: turning it off also switches `--mamba-cache-mode align` to
  `none` and would change the KDA path. The salt isolates cache reuse instead.

**Phases per batch of 8.**
1. **Warm.** Each prefix, `max_tokens=1`, salt A. This builds the prefix by prefill, as cached 4,352-token blocks.
2. **Forced window.** Same salt A: cache hit plus a remainder chunk of at most 4,351 tokens. All 8 requests enter
   decode within seconds of each other. The first 512 window positions are lead-in and are not primary.
3. **Prefill rebuild.**
   - Prompt = prefix + window, `max_tokens=1`, fresh salt R.
   - The engine's prefill-scored distribution at the same end position.
   - Labelled "0.7.0 prefill computation" on 0.7.0 (source review, implications 1).
4. **Index-K dumps (Stage 1, run 1).**
   - At the end of the forced window and at the rebuild, copy the window's index-K pool rows for all 11 DSA layers
     (both ranks on 0.7.0 DCP2; about 50 MB per run).
   - Decode-written bytes are compared with prefill-rebuilt bytes per pool.

**Concurrency and route.**
- 8 forced requests in flight throughout. The hook records requests, rows and co-scheduled prefill tokens at every
  measured step.
- The primary analysis uses positions recorded with 8 decoding requests and no co-scheduled prefill.
- Rows per step are 32 at x3 and 48 at x5. Both are on the tiled selection route (tiled from 5 requests at x3, 3 at
  x5), as in 81-88% of GPQA status lines.
- Composition runs (Stage 2) put one target with 7 sampled background requests (GPQA prompts, distinct seeds). They
  keep 8 in flight and measure the batch-composition effect.

**Repeatability first.**
- Stage 1 runs batch L three times per arm:
  - run 1 (salt A);
  - run 2 (salt A again: decode-only variation, same cached prefix);
  - run 3 (fresh salt B, with its own rebuild: prefill and decode variation).
- Engine comparisons are not read until the within-arm ranges are computed (section 8, rule R0).
- Long-context non-repeatability is known: in the bundle gate, greedy outputs from 4,240- and 16,240-token prompts
  diverged by token 3 on the release and the bundle alike.

**Measured per row.** For the four termination ids (`</think>` 154842; `<|endoftext|>` 154820; `<|user|>` 154827;
`<|observation|>` 154829) and for the forced token:
- log-probability before masking;
- mass strictly below, and mass at or below (top_p eligibility bounds);
- the served post-mask probability.

Also per row: entropy; whether the argmax is the forced token; whether the argmax is a termination id; top-1
probability; acceptance metadata. The close is a single token (TERMINATION.md). The `</s>` stop string and spelled-out
closes are multi-token and are not scored.

## 4. Arms and builds

| Arm | Image | Draft depth | Tail ring | Decode-only index-K defects | Build |
|---|---|---|---|---|---|
| Stock 0.7.0 | `48e254d9…` + probe hook | x5 (served) | 4 slots, wraps | D2 (positions/ownership) + D6 (ring wrap) + seed stride (DCP-audit D7) | Hook layer only |
| B0 | `08d1cb90…` (D1-D4) + hook | x3 (served) | 8 | none | Hook layer only |
| B0 x5 | Same image, `DFLASH_TOKENS=5` | x5 | 16, no wrap | none | None |
| B0-minus-D2 (diagnostic) | release `a9e985c0` + D1 + D3 + D4 + hook | x3 | 8, no wrap | D2 only (with D6-stale-ids reachable again) | New, diff-verified |
| B0-ring4 (diagnostic, Stage 1b) | B0 with the spec-ring port reverse-applied + hook | x5 | 4, wraps | D6 only | New, diff-verified |
| Stock 0.9.1 (context, Stage 2c) | `a9e985c0` + hook | x3 | 8 | D2 | Hook layer only |

**Why B0 is the corrected arm.**
- B1 = B0 + D7 differs only in how prefill builds pools at unaligned chunk starts, which here touches the prefix
  alone.
- B0-minus-D2 is defined against B0 (B1-minus-D2 is incoherent: D7's repair needs D2's ring).
- B0 and B0-minus-D2 share the prefix computation exactly, so their contrast isolates D2 in the decode window.
- If the reviewer prefers B1 as named in amendment 20, B1 replaces B0 at the same cost. B0-minus-D2 then compares
  against B0 only in an added session (+1.65 GPU-hours).

**B0-minus-D2 build** (no such image or patch set exists; `bundle-d124` is D1 + D2 + D4):
- Base: `FROM` the release digest.
- Gate 0: the replaced files must be the release files (`release-sha256.txt`).
- Copy B0's `sparse_attn_indexer_kpool.py` (`9104be37…`, D1), `exl3.py` (`b40344e2…`, D4) and `gate_linear.py`
  (`2499ee67…`, D3).
- Keep the release `indexer.py` (`67449677…`) and `mamba_hybrid.py` (`82968af5…`).
- Gates:
  - byte hashes;
  - `py_compile`;
  - D1 markers ×4 and the two `swiglu_limit` lines present;
  - the D2 strings (`own_block != 0`, `slot_mapping.copy_`, `positions=input_batch.positions`) absent.
- Whole-image manifest diff:
  - against B0: exactly `indexer.py`, `mamba_hybrid.py`, their bytecode and `/etc/hostname`;
  - against the release: exactly the three D1/D3/D4 files, bytecode and `/etc/hostname`.
- D1's prefill branch uses the model's own `positions`, not D2's metadata positions (`sparse_attn_indexer_kpool.py`
  forward). D1 therefore stays active without D2.
- CPU test: the existing `D2_scenarios` / `test-kpool-tail-positions.py --expect-unpatched` harness must reproduce
  the release behaviour on the installed files.

**B0-ring4 build (D6 only):**
- Reverse the `(old, new)` edits of `port-kpool-spec-ring-glm53.py` in B0's `attention.py` and `kpool_compress.py`.
  The script holds exact strings, and the seed-stride port stays.
- Diff-verify: only those hunks differ from B0.
- CPU test: `KpoolTailSpec.block_size == 4` at K=5, and the ring harness (`kdcp/tailfix/ring070`) reproduces the wrap
  signature under `TRITON_INTERPRET=1`.
- Diagnostic only; never a remedy.

**Probe hook layer:**
- Adds `vllm/local_forcewin.py` and marked hook lines in `v1/worker/gpu/{model_runner.py,
  spec_decode/rejection_sampler.py, sample/sampler.py}`. The hook is inert unless a request carries the key.
- One generator patches all arms. The 0.7.0 `model_runner.py` differs from 0.9.1's only by a logging call.
- Dockerfile gates as in `b0-mapguard/build`: pre-hashes, post-hashes, marker counts, layer ancestry.
- CPU tests run inside each derived image: `forcewin_logic` on the installed module, F/M checks, and a fake-tensor
  index-K page dump test after `b0-cache/fixture/b0views.py`.

## 5. Fixtures (`fixtures/fixtures.json`; generated by `fixtures/select_fixtures.py` from the sha256-checked inventory)

All long windows are re-tokenized text, so each is a shared counterfactual fixture and not a replay. Every window ends
below 131,072 tokens and none forces past a natural close.

| Batch | Id | Source (id) | Class at the window (20b) | Doc | Window end (reasoning index) | Prefix tokens |
|---|---|---|---|---|---|---|
| L | L1 | stock 0.7.0 (`ca7fd76b2b7a`) | natural close (positive control at e) | 13 | 102,579 | 94,518 |
| L | L2 | stock 0.7.0 (`e87744068243`, drift −8) | natural close | 88 | 121,500 | 113,537 |
| L | L3 | stock 0.9.1 B79 s5004 (`09ba09a88d7b`) | natural close | 79 | 97,408 | 89,307 |
| L | L4 | stock 0.9.1 B s5005 (`f0c2498e7b7f`) | natural close | 88 | 80,324 | 72,361 |
| L | L5 | pilot r0 history (`fc3010f3a808`) | active, exhaustion | 170 | 120,000 | 112,025 |
| L | L6 | stock 0.7.0 V79 s5004 (`a503e4fce96a`) | active, closes 51.7k later (pairs with L3 at the same index) | 79 | 97,408 | 89,307 |
| L | L7 / L8 | V79 / B79 s5010 (`c85856ae6b1f` / `77100ed719f5`) | active, exhaustion (doc-79 pair) | 79 | 120,000 | 111,899 |
| S | S1-S4 | exact-id step-5/6 traces (`57703691f509`, `53fa5b3e066f`, `be2e9e349395`, `4b92a6a90ec4`) | natural close; full replay from the prompt | 1, 3 | 1,722-4,795 | prompt |
| S | S5-S8 | text controls (`9143682aa636`, `7a64a6fa06d2`, `f4f28e056fb3`, `964a74b9edf1`) | natural close; full replay | 79, 55, 13, 109 | 6,885-11,981 | prompt |
| L2 (Stage 2) | M1-M8 | stock 0.7.0 docs 79, 88, 121, 127; stock 0.9.1 docs 79, 88×3 | natural close | | 69,265-124,781 | 61k-117k |
| L3 (Stage 2) | A1-A8 | L1-L4 histories ending 8,192 before their close; r1 end (64,000); doc-79 s5002 pair at 96,000; a V79 loop history 64k before its onset | active | | 64,000-120,000 | 56k-112k |

- **Primary positions** per window:
  - the end position e;
  - the last 512 positions;
  - a 64-token grid after the lead-in;
  - non-repeating commitment events inside the window (positions only).
- **Hashes.** Window-level hashes (prefix ids and forced ids) are computed at the Stage-0 freeze, when the ids are
  re-tokenized in a CPU container. History-level hashes are in `fixtures.json`.
- **Limits.**
  - The material comes from a few hard questions (13, 79, 88, 170, 121, 127).
  - The exact-id controls come from modified engines at top_p 1.0.
  - The ids immediately before a text fixture's close are reconstructed (whitespace stripped by the parser).

## 6. Stages, costs and ceilings (`cost/cost.py`, `cost/cost.out`)

**Planning rates, from measured receipts.**
- **Start to ready plus warm-up gate:** 13.5 min for the 0.9.1 family (measured 11.6-12.3) and 10.5 min for 0.7.0
  (measured 8.3-9.3).
- **Teardown:** 1.5 min.
- **Prefill:**
  - 0.9.1 family: 3,000 tok/s (measured 3,300-3,700 at 0-120k; 2,400-2,600 in guard images).
  - 0.7.0: 1,200 tok/s. It was measured at only 1,630-1,820 tok/s at ≤ 16k and never at 64-128k: **the largest
    cost uncertainty** (Stage 1 for 0.7.0 is 2.4-3.3 GPU-hours across 1,600-900 tok/s).
- **Decode per request at 8 in flight:** 45 tok/s for the 0.9.1 family (measured 54-64) and 27 tok/s for 0.7.0
  (measured 32-37).
- **Occupied GPU-hours:** 2 GPUs × wall time from server start to GPU release.
- **Ceiling:** 1.3 × estimate + one invalid-start allowance (0.5). At most one retry per stage. A watchdog stops the
  session at the ceiling.
- **Phase order:** most important first, so a stopped session still yields the index-K bytes and run 1.

| Stage | Arms and sessions | Workload per session | Estimate | Ceiling |
|---|---|---|---:|---:|
| **0 (CPU)** | none | Hook and generator; builds of B0-minus-D2 (and B0-ring4 if 1b is wanted) and four or five hook layers; CPU tests; fixture ids and hashes; frozen analysis script and dry runs | 0 GPU | 0 |
| **1** | 0.7.0 x5 (2.79); B0 x3 (1.65); B0-minus-D2 x3 (1.65); B0 x5 (1.65) | P0 health; S full replay + rebuild; L run 1 + dumps + rebuild; L run 2; L run 3 + rebuild | **7.74** | **10.6** |
| 1-min (alternative) | Same four arms | P0, S, L run 1 + dumps only (no repeats; repeats later cost another start, about 0.45 each) | 5.01 | 7.0 |
| **1b (optional)** | B0-ring4 x5 | As Stage 1 | 1.65 | 2.6 |
| **2** | 0.7.0 x5 (3.53); B0 x3 (2.04); B0-minus-D2 x3 (2.04) | L2 ×2 + rebuild; L3 ×2 + rebuild; composition (L3 and L6 targets, each with 7 background); batch L oracle-draft bracket | **7.62** | **10.4** |
| **2c** | Stock 0.9.1 x3, one session | Stage 1 + Stage 2 workloads | 3.17 | 4.6 |
| **K** | One GPU, offline container | Section 7 | 0.30 | 0.5 |
| 3 (priced only, not proposed) | Three arms | Whole-context replay: batch L forced from the prompt to the window end (about 120k decode per request) | about 6.9 | — |

- **Gates between stages.**
  - Stage 2 runs only if Stage 1 passes validity (section 9) and rule R0 leaves the comparisons interpretable.
  - Stage 2c is context and may be dropped.
  - Stage K is independent and gates only window ends at or beyond 131,072.
  - No stage, GPQA pass or ablation follows automatically (amendment 20, section 6).
- **Preconditions for every GPU stage.**
  - No GPQA pass running (one is running now).
  - The Live-tab preflight shows nothing else on the GPUs.
  - Frozen hashes check out (`SHA256SUMS` of the frozen package).

## 7. Stage K: model-free selection replay (separate small item)

**Method.**
- B0/B1's B12x selection kernels are byte-identical to 0.9.1's.
- Replay synthetic inputs and recorded indexer inputs at L = 128k, 160k, 200k, 262k and 330k.
- Recorded inputs exist only up to about 120k. To reach longer lengths, they are extended by tiling the recorded
  pools.
- Routes and row counts:
  - the tiled route at rows 24, 32 and 48 (x3 at 6-8 requests, x5 at 8);
  - the fused route at rows 4, 8 and 16 (forced with `B12X_FUSED_INDEXER=0` / `=1` as needed).
- Reference: fp64 top-512.

**Criteria.**
- Set equality, allowing exact boundary ties.
- The lowest-scoring selected pool sits in column 511 (D1).

**Coverage.** It exercises the untested regimes:
- supertile chunk ≥ 1 and the third slice from 131,072;
- 196,608 and 262,144;
- repeated fused trims.

**Tooling and cost.** The tooling is the phase-2 replay (`kdcp/fixtureB/replay.py`; stage-2 replays cost
0.016-0.049 GPU-hours each). Ceiling 0.5 GPU-hours on one GPU.

**Gate.** Required before any window ends at or beyond 131,072. It has no bearing on Stages 1-2.

## 8. Frozen interpretation (adds to amendment 20 section 5; unchanged in substance)

**Primary metrics, per fixture.**
- `M_end`: ln P_pre(`</think>`) and P_post(`</think>`) at e.
- `M_tail`: the mean of ln P_pre(`</think>`) over the last 512 positions, and the share of those positions where
  `</think>` is certainly eligible under top_p.
- `M_stop`: the maximum P_pre over the three stop ids in the tail. This measures premature-stop propensity, which is
  not a valid finish.

**Secondary metrics.** Forced-token log-probability, argmax agreement and entropy. Results are reported by fixture
class: natural close, active-before-close, exhaustion and loop pre-onset are never pooled into "non-completion".

**R0 (repeatability first).**
- The within-arm range for a fixture is taken over Stage 1 runs 1-3; in Stage 2, the composition runs are added.
- For arms where the range of `M_tail` exceeds 0.5 nat on 4 or more of the 8 batch-L fixtures, cross-arm comparisons
  involving that arm are "not interpretable at this n" and are reported without reading.

**Per-fixture difference.**
- Arm X differs from arm Y on a fixture only if every run of X lies outside the range of every run of Y. This is the
  amendment-4 rule.
- A consistent difference requires the same direction on at least 6 of 8 fixtures in a batch.
- The same rule applies to prefill-rebuild values, using runs 1 and 3.

**Readings (amendment 20, section 5):**
1. **Prefill-only difference** (rebuild values differ, decode-window values do not): investigate the prefill path. No
   generation-time claim.
2. **Difference in the decode window:** a mechanism is nominated and checked against within-engine variation,
   including batch composition. The primary old-versus-corrected comparison is 0.7.0 x5 against B0 x5 (same draft
   depth); 0.7.0 x5 against B0 x3 is reported alongside.
3. **No difference:** narrows the hypothesis for 8k decode-built windows at 64k-125k on these fixtures. It does not
   exclude corruption accumulated over a whole generated context, rare effects, or effects past 131,072.
4. **Stock 0.7.0 favours termination** (higher P(`</think>`), or eligibility where B0 has none): establish whether
   this is intended computation or known corruption, using the diagnostics:
   - If B0-minus-D2 x3 against B0 x3 reproduces the direction of 0.7.0 x5 against B0 x5 (consistent by the rule
     above), **D2 is nominated, not proven**, as the mechanism.
   - If B0-ring4 x5 against B0 x5 reproduces it, **D6 is nominated, not proven**.
   - If B0 x5 against B0 x3 reproduces it, draft depth is a factor in its own right and is reported as such.
   - If none reproduces it, the source is unidentified among the remaining 0.7.0 differences (DCP2 geometry, B12x
     version, seed stride, the D1 policy, D3/D4).

   A corrupted path is never restored as a remedy.
5. **Index-K bytes (Stage 1).**
   - Measure: per-pool relative L2 error of dequantized decode-written index-K against the prefill rebuild, per arm and
     layer. "Gross" means above 0.25 or non-finite (frozen).
   - B0's distribution shows the rounding-only spread. If B0's 99.9th percentile exceeds 0.25, the threshold does not
     discriminate and only distributions are reported. If B0 shows gross pools, the probe stops: the reference
     writer would then be broken.
   - Expectation, from the CPU harness and not live: 0.7.0 about 100%; B0-minus-D2 at 8 in flight 78-90%; B0 0%.
     Whether a measured share matters for selection is not inferred from bytes.
6. **Throughout:** no statement that a probability difference explains completion rates. Every comparison names its
   fixture class, and "0.7.0 prefill computation" labels every 0.7.0 prefill value.

## 9. Validity gates (checked by the frozen analysis before any reading)

Validity:
- The returned token ids equal the forced ids in 100% of forced requests.
- Every forced position has exactly one measured row.
- Per-step committed-token hashes are equal on both ranks.
- At least 90% of primary positions were recorded with 8 decoding requests and no co-scheduled prefill.
- The served-mask cross-check holds (`p_post > 0` exactly where the eligibility bounds allow).
- No NaN in the metrics.
- Dumps are complete.

Reported but not gating:
- Mean forced acceptance length against serving (2.8 at x3; 3.2-3.4 at x5), and the count of rejected
  pool-completing drafts. These are the distortion measures; if forced acceptance is below 0.7 × serving, the window
  is labelled "reject-path heavy".

A session that fails validity is reported as invalid. It may be retried once within the stage ceiling.

## 10. Open risks

- **0.7.0 prefill rate at 64-128k is unmeasured.** It drives 0.7.0's cost; the ceiling and phase order contain it.
- **Forcing distortion.** Counterfactual and re-tokenized histories lower acceptance and over-exercise reject/restore
  (D6's path), so 0.7.0's D6 exposure in the windows may exceed its exposure in serving. This is measured, and
  bracketed in Stage 2.
- **Dilution.** Windows exercise decode-written state on only 2,048 pools. A null result here does not speak to whole
  contexts (Stage 3 is priced).
- **Cross-request writes.** On arms with D2 (stock 0.9.1, B0-minus-D2, 0.7.0), unmodified 0.9.1 was observed writing
  into other requests' attention blocks (6 events in a session). Co-running fixtures can therefore contaminate each
  other. This is served behaviour, but it couples fixtures within a batch, so composition runs and per-fixture
  reporting matter.
- **Non-determinism.** Long-context decode was not bitwise repeatable even on the release (bundle gate). R0 may leave
  comparisons uninterpretable at n = 3.
- **B0 x5 and B0-ring4 x5 have never been served.** Their start-up, memory and throughput are unmeasured. B0-ring4
  also changes the tail-cache page size.
- **Hook correctness.** It touches the token-commit path. It is guarded by the CPU tests, the inert-unless-keyed
  design, and validity gates that compare returned ids with forced ids.
- **Fixture breadth.** Six hard questions, re-tokenized text and few stock-0.7.0 closes (14 at 64k-128k).
  Same-seed-across-engines pairs (doc 79) are not matched trajectories.
- **Existing tooling is not a substitute.** The check images' observers (NaN-byte index rows) exist only for the
  0.9.1 family, measure a corruption signature rather than equality with prefill, and need no build. They could give
  a cheaper, partial first look on D1-only against B0 (about 1.1 GPU-hours per session), but not for 0.7.0.

## Files

| Path | What it is |
|---|---|
| `PROPOSAL.md` | This document |
| `feasibility/FEASIBILITY.md` | Feasibility from source, per engine; draft depth, D6 and rollback notes |
| `feasibility/forcewin_logic.py`, `test_forcewin_cpu.py`, `plugin_startup_check.sh`, `cpu-check.out` | Reference forcing and metric logic, and the CPU checks run in both images |
| `fixtures/select_fixtures.py`, `fixtures.json`, `select_fixtures.out` | Fixture list: ids, positions, classes and hashes only |
| `cost/cost.py`, `cost.out`, `cost.json` | Cost arithmetic and its sources |
| `SHA256SUMS` | Hashes of the above |
