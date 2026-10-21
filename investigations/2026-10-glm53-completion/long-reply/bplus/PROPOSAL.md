# B+: fixed-prefix continuation of long GPQA replies on three engines (frozen before data)

2026-10-16. **Prepared, not run. Zero GPU was used to build this package.** CPU-only containers (`--network none`,
`CUDA_VISIBLE_DEVICES=`, no `--gpus`) on existing local images did the tokenization, source reading and analysis dry runs.
No model server was started and no request was sent to one.

Labels: **[M]** measured here by running code; **[S]** read from source or configuration; **[I]** inferred or assumed.

**Privacy.** Prefix token ids and the original prompt messages live only in `PRIVATE/` (mode 0700, files 0600). No
question, answer, answer letter or reply text appears in any other file. `checks/privacy-scan.txt` searched every other
file for prompt text windows and reasoning id runs: 0 hits [M].

**Status of every rule below:** frozen with `SHA256SUMS`. No continuation outcome exists yet. A change after freezing
needs a dated amendment.

## 1. Question

The question is why long GPQA replies finish within the 327,680-token cap more often on stock tpurtell 0.7.0 than on
0.9.1. Matched on question, 0.7.0 finished 23 of 36 replies longer than 100k tokens, against 18 of 58 for the 0.9.1
family (discovery T11) [M].

At 100k, 8 of 10 eventual 0.7.0 capped replies are not yet repeating. This experiment holds the trajectory's state at
100k fixed: identical token-id prefixes are continued on three engines. It asks whether the outcome is set by the state
(S) or by the engine's continuation from it (E).

**Engines** (the operator's option B+):
- stock 0.7.0;
- stock 0.9.1;
- B1 = 0.9.1 + the verified fixes D1-D4 and D7.

## 2. Prefixes

### 2.1 Selection rule (`scripts/select_prefixes.py` → `out/selection.json`)

The rule uses only discovery outputs (ids, lengths, R50 at 100k, panel counts). No continuation outcome exists. Its input
files are hash-checked.

- **Pool:** the 29 rep070-F replies with T ≥ 100,000.
- **E1:** re-tokenized reasoning longer than 100,000 tokens, so the cut lies strictly inside the reasoning.
- **E2:** the discovery's R50 at g = 100,000 (`replies_all.json` `R50_at["100000"]`) is defined and below 0.05.
- **Strata:** the original outcome, capped or finished.
- **One reply per question;** questions are never shared between strata, and the capped stratum is chosen first.
- **Question rank:** the long-reply panel count over all 15 logged passes (discovery T7 "all"), descending. Ties go to
  the lower sha256("bplus-2026-10-16:q<doc>").
- **Reply within a question:** the lowest sha256("bplus-2026-10-16:" + id). The tie-break is outcome-blind.
- **Take** 6 questions per stratum.

**Result** [M]:
- 25 of 29 replies are eligible.
- Excluded:
  - q7p3: reasoning ends at 99,548;
  - q147p3: R50 at 100k is undefined;
  - q88p2 and q127p3: R50 at 100k = 1.0.
- 12 prefixes on 12 distinct questions: 6 capped and 6 finished.
- Only 6 questions have an eligible capped reply, so the capped stratum is all of them.
- The finished stratum takes the top 6 of 11 questions by panel count. The 7th-ranked question has count 2, so there is
  no tie at the boundary.

**Covariates at 100k** [M]. They are computed in `build_prefixes.py` on the 100k-token reasoning prefix:
- chars/token = characters covered by the 100k tokens / 100,000;
- interjection = the discovery's `CHK_WAIT` matches per 1k tokens;
- digit = the digit share of characters;
- R50 back = R50 of the last valid window inside the prefix (positions 95,855-99,950, prefix tokens only).

| id | reply | stratum | original T | R50 @100k (discovery) | R50 back | prompt | prefix | chars/tok | interj./1k | digit | drift | prefix ids sha256 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P01 | q79p2 | capped | 327,680 | 0.000 | 0.003 | 91 | 100,091 | 2.26 | 9.3 | 0.163 | 2 | c7ade2776edc |
| P02 | q88p1 | capped | 327,680 | 0.000 | 0.000 | 229 | 100,229 | 2.76 | 25.2 | 0.057 | 0 | 92ed266a1e4f |
| P03 | q69p2 | capped | 327,680 | 0.002 | 0.000 | 148 | 100,148 | 2.31 | 49.0 | 0.065 | 1 | 6ea0c2f4040c |
| P04 | q120p3 | capped | 327,680 | 0.019 | 0.001 | 276 | 100,276 | 2.64 | 6.1 | 0.082 | 1 | 2fa27f96ecfc |
| P05 | q8p2 | capped | 327,680 | 0.000 | 0.000 | 238 | 100,238 | 2.81 | 8.6 | 0.075 | 1 | 8bcdf999e240 |
| P06 | q158p3 | capped | 327,680 | 0.035 | 0.034 | 195 | 100,195 | 1.70 | 3.6 | 0.427 | 1 | 61bf8d323ce7 |
| P07 | q127p2 | finished | 206,688 | 0.004 | 0.012 | 2,795 | 102,795 | 2.14 | 4.2 | 0.077 | 3 | 0d950ff4191e |
| P08 | q147p1 | finished | 150,616 | 0.007 | 0.034 | 191 | 100,191 | 2.60 | 11.0 | 0.097 | 3 | ac44215a8eea |
| P09 | q81p2 | finished | 126,276 | 0.000 | 0.003 | 766 | 100,766 | 3.22 | 2.6 | 0.036 | 0 | 7a4fd30b9b7f |
| P10 | q71p1 | finished | 124,709 | 0.000 | 0.000 | 178 | 100,178 | 3.15 | 3.8 | 0.036 | 4 | 2c1382082d01 |
| P11 | q186p1 | finished | 124,517 | 0.000 | 0.000 | 308 | 100,308 | 2.84 | 10.4 | 0.090 | 5 | 82578d0bb0c3 |
| P12 | q145p2 | finished | 142,105 | 0.000 | 0.010 | 285 | 100,285 | 3.27 | 2.2 | 0.028 | 3 | 1e01607ee993 |

Notes on the table:
- **Drift** is the reply-level re-tokenization drift from the discovery (0-8 tokens). It bounds how far the re-tokenized
  ids can differ from the sampled ones.
- **Original continuations:** 5 of the 6 finished originals finished within +64k of the cut (the exception is P07, at
  +106k). No capped original finished. Under a pure-state reading, 0.7.0's expected rate in this sample is about 0.42
  [I]; section 5.3 uses this as a sensitivity case.
- **Precursor windows:** none of the 12 has a high-repetition episode inside its first 100k [M].
  - P07's maximum in-prefix R50 is 0.69. That is the recovered q127 episode near 50k, which lasted under 2,048
    positions at threshold 0.5.
  - The others stay at or below 0.072.

### 2.2 Construction (`scripts/build_prefixes.py`, CPU container, 0.7.0 image)

The original generated ids were not retained. Each prefix is built in three steps:
1. **Prompt:** the original request's messages and `chat_template_kwargs {"enable_thinking": true}`, rendered with the
   0.7.0 engine's own template (checkpoint 701cd745 `chat_template.jinja`, sha256 34d5ee66…; the 0.7.0 server passes no
   `--chat-template`) and tokenizer, with `add_generation_prompt` and `add_special_tokens=False` as in vLLM's chat path.
   The prompt ends `<|assistant|><think>` [M].
2. **Reasoning:** the response `reasoning` text, re-tokenized whole with the K3.25 `tokenizer.json` and
   `add_special_tokens=False`, as in the discovery.
3. **Cut rule:** keep the first 100,000 reasoning ids. The cut is a token boundary of the whole-text tokenization: the
   100,000th id may be a word piece. Ids are sent, so the cut text is never re-tokenized.

`prefix = prompt ids + reasoning ids[:100000]`. The hash is sha256 of `",".join(ids)`.

**Checks** (86 / 86 in build) [M]:
- the messages hash equals the raw record and the request log;
- the rendered prompt length equals the server's `prompt_tokens` in both the raw response and the rep070-F request log;
- the prompt ids equal d7's independent pure-Python tokenization (sha256 of `json.dumps(ids)`);
- `transformers` and `tokenizers` give identical prompt and reasoning ids;
- the reasoning length equals the discovery's N_reasoning;
- the reasoning part contains no stop id and no `</think>`;
- the prompt carries one stop-class id, the template's `<|user|>` turn marker. Stop ids act on generated tokens only.

**Reproducibility** [M]: two builds (before and after the prompts file was added) wrote the same `PRIVATE/prefixes.json`
sha256 `4abffaf1…`. The verify mode rebuilt every prefix in each of the three images, twice, and matched it id for id:
170 / 170 checks each.

**Limit (stated):** re-tokenized ids can differ from the originally sampled ids by a few tokens. The discovery measured a
drift of 0-8 tokens per whole reply. All engines receive the same ids, so this changes the state relative to the
original run, not between engines.

### 2.3 Template, tokenizer and endpoint findings

**Templates differ as files but not as output** [M, S]:
- 0.7.0 serves the checkpoint template (34d5ee66…).
- 0.9.1 and B1 serve `--chat-template /opt/glm53/templates/glm53-zai-a5b45eb.jinja` (0c4099f3…, inside the image),
  with snapshot 0490d2f7.
- Snapshots 701cd745 and 0490d2f7 differ only in `README.md` and `chat_template.jinja`. Weights, tokenizer and configs
  are the same blobs.
- Rendered with each image's own `transformers` 5.15.1 / `tokenizers` 0.22.2 and its served template and snapshot, all
  12 prompts give byte-identical rendered text and ids (`checks/prefix-verify-{v070,v091,b1}.json`).
- d7 found the same for all 198 GPQA prompts.

**Decision:** every engine receives the 0.7.0-rendered ids (the prefix source). At run time the client also asks each
server to render each original prompt (`/v1/chat/completions/render`, no generation) and records whether the result
equals the prompt part. This check is recorded, not a refusal.

**`/v1/completions` in all three images** [S] (`checks/source-check.txt`):
- The relevant source files are byte-identical in the three images: completion and chat protocol and serving files,
  `input_processor.py`, `sampling_params.py`, the samplers, `config/model.py`.
- **Token-id prompts:** `CompletionRequest.prompt` accepts `list[int]`.
- **Sampling parameters:** `to_sampling_params` passes `seed`, `temperature`, `top_p`, `max_tokens`, `stop` and
  `stop_token_ids` into `SamplingParams`.
- **Per-request seeds** drive a per-request `torch.Generator` in both the rejection sampler (speculative decoding) and the
  top-p sampler.
- **Chat vs completions:** the two `to_sampling_params` differ only in logprob, echo and output-kind plumbing. Both take
  server defaults from `get_diff_sampling_param()` (generation_config: temperature 1.0, top_p 0.95).
  `repetition_detection` and `thinking_token_budget` default to None in both.
- **Stop tokens:**
  - The engine input processor, shared by both endpoints, adds the tokenizer EOS 154820 and generation_config
    `eos_token_id` [154820, 154827, 154829] as stop tokens.
  - `</think>` 154842 is a non-special added token and is not a stop token.
  - Every request also carries the GPQA task's stop string `["</s>"]`, as every logged chat request did [M: 594 / 594].
- **`return_token_ids`** is output-only: it is not referenced in `to_sampling_params`. It returns the generated ids and
  the prompt ids the server received. The first-contrast sessions already used `/v1/completions` with token ids, seed and
  `return_token_ids` on a B1-based image [M, earlier record].
- **Reasoning parser:** the completion endpoint has no reasoning-parser reference (0 matches). The parser only splits chat
  output text (the glm45 adapter starts in the reasoning state), so it is not needed: the continuation is raw ids/text,
  split at the first 154842.
- **The release warm-up of every image** sends rendered chat prompts as token ids to `/v1/completions`. It describes
  this as "the normal rendered-chat token-id path" [S].

## 3. Continuation

### 3.1 Requests (`out/plan.json`)

The body is `{model, prompt: ids, max_tokens: 65536, temperature: 1.0, top_p: 0.95, seed, stop: ["</s>"],
return_token_ids: true, stream: false}`. Nothing else is set.
- Its parameter hash (`c53443e3…`) is checked per request.
- Context: the longest prefix plus horizon is 102,795 + 65,536 = 168,331 tokens, below `max_model_len` 1,048,576.

### 3.2 Seeds

**Scheme:** seed = 8,160,000 + 10,000·engine + 100·prefix + j, with engine 1-3, prefix 1-12 and j 1-4. This gives
144 distinct seeds in 8,170,101-8,191,204, recorded per request.

**None has been used before** [M]:
- `rep070/refreeze/checks/seed_scan.py` was run unchanged with the 144 candidates. It scans request logs, lm-eval
  configs, published rows and a broad text scan of 10,710 files: all 144 are eligible (`checks/seed-inventory.json`).
- The text scan cannot see formula-based seeds, so the scripts' seed bases were also grepped (7.81M, 8.81M, 9.40M,
  9.85M; CPU fixture seeds ≥ 8.48M): none intersects (`checks/seed-formula-scan.txt`).

**Common random numbers: not used (distinct seeds)** [S/I]:
- vLLM draws its sampling noise from the request's seed, so a shared seed aligns the random streams only while the
  engines' logits are identical.
- They are not: different speculative depth (K5 vs K3) consumes the stream differently, and DCP layout and numerics
  differ.
- Shared seeds would therefore add an uncontrolled, early-trajectory correlation between engines without making their
  continuations comparable token by token.
- Distinct seeds keep the four seeds of a prefix exchangeable within an engine. They also keep the prefix as the paired
  unit across engines.

### 3.3 Concurrency and submission order

**Order:** 8 in flight, matching GPQA serving. One fixed order is used on every engine:
- 4 rounds (j = 1..4); in each round every prefix appears once.
- Within a round, the two strata are each shuffled with `random.Random(20261016 + j)` and interleaved capped/finished.
  The first stratum alternates by round.
- The first 8 in flight are 8 different prefixes, 4 per stratum.
- Seeds 2-4 of a prefix are submitted after its seed 1.
- When a request completes, the next in order is submitted. The actual schedule therefore differs by engine speed; that is
  part of "engine".

### 3.4 Engine order and sessions

**One runner invocation, three sessions, under one approval.** The order is stock 0.7.0, then stock 0.9.1, then B1. Each
session runs on a fresh server.

**Why one invocation:**
- one `APPROVED_GPU_H` and one wall-clock watchdog;
- identical pre-flight and package verification;
- one finalize step;
- no operator hand-off between engines, so less machinery and less room for drift.

**Why this order:**
1. 0.7.0 is the prefix source and the slowest engine with the widest cost uncertainty. Running it first exposes a
   failure or budget problem before the cheaper sessions are spent.
2. 0.9.1 completes the primary contrast second.
3. B1 (secondary contrasts) runs last. If the budget rule stops before B1, the frozen C1-ONLY fallback still gives a
   primary reading.

Order is confounded with time of day and GPU thermal state. The runner logs GPU memory, utilization, temperature, power,
SM/memory clocks and P-state every 60 s, plus the GPU state before and after each session.

### 3.5 Prefix caching: as shipped

All three launchers enable prefix caching with `--mamba-cache-mode align` [S]. Behaviour on each engine:
- **Cache hits:** seeds 2-4 of a prefix can hit the cache of seed 1's blocks.
  - The hit is the largest whole block below the prompt minus one block (EAGLE back-off), in blocks of 3,584 tokens on
    0.7.0 and 4,352 on 0.9.1 / B1 (server logs).
  - So a hit reuses about 91-93% of the prefix KV, and the remainder (about 7-9k tokens) is prefilled from a
    block-aligned start.
  - 0.9.1 adds an opt-in boundary lookup (`GLM53_DFLASH_BOUNDARY_LOOKUP`, default 0 in its launcher, so it is off). With
    it off, the back-off is the same as 0.7.0's [S].
- **Measurement:** whether hits happen is recorded per request where the engine reports it (`cached_tokens`; 0.9.1 and B1
  run `--enable-prompt-tokens-details`, 0.7.0 does not). For all engines, the server log's "Prefix cache hit rate" lines
  are recorded.
- **Why as shipped:** caching is part of how each engine serves. With hits, the four seeds share the very same prefix KV
  bytes on an engine, which is the "same state" intended.
- **Pool headroom** [I]: 8 × 168k = 1.35M tokens in use against pools of 2.76M (0.7.0) and 4.71M (0.9.1 / B1). Eviction
  of cached prefixes is possible late on 0.7.0, so hit rates are reported and not assumed.

## 4. Known asymmetries (part of "engine", not controlled)

### 4.1 Prefill rebuild

Every engine rebuilds the 100k prefix KV by **prefill**.
- **D2 (decode path):** in the original 0.7.0 run, the prefix KV was written through decode, with the decode-path
  index-K defect D2 present. Here D2's decode-path state is absent from the prefix on all engines. It is present only in
  the continuation's decode on stock 0.7.0 and stock 0.9.1; B1 has the D2 fix.
- **Decode-built state:** the original decode-built state is not reproduced. Discovery option C would test that.

### 4.2 D7 in the prefix prefill (`scripts/d7_prefill_exposure.py` → `out/d7_prefill_exposure.json`) [E]

Stock 0.7.0 and stock 0.9.1 still carry D7, the index-K prefill writer that assumes 4-aligned chunk starts. The d7
corrected scheduler simulator (hash-checked, imported read-only) chunks each prefix in the plan's order:
- Solo chunks are 2,045 tokens on 0.9.1 (2,048 − 3) and 2,043 on 0.7.0.
- A decode row in the step costs 7 tokens on 0.9.1 and 11 on 0.7.0.
- Chunks are clipped at Mamba-align blocks.

| | cold prefill beside D = 0 … 7 decoding requests (events over 12 prefixes) | plan, central (cache hits) | per continuation: affected share of the prefix's index-K pools |
|---|---|---|---|
| 0.7.0 (block 3,584) | 337, 0, 337, 337, 337, 0, 337, 337 (about 1 per block) | 476 unaligned starts; all 48 continuations exposed | mean 0.14%, max 0.18% (≤ 47 pools of ~25,000) |
| 0.9.1 (block 4,352) | 553, 277, 553, 0, 553, 277, 553, 0 (about 2 per block) | 248 starts, all in the first wave; 32 of 48 continuations exposed through the cache | mean 0.08%, max 0.20% (≤ 49 pools) |
| B1 | same chunking as 0.9.1 | every straddling pool repaired by the D7 fix (verified CLEAN in d7 step 2) | 0 |

How the exposure falls in the plan:
- **0.9.1:** a cold prefill is aligned at D = 3 and 7. Later requests arrive beside 7 decoding requests, so on 0.9.1 only
  the first-wave builders (D = 0-7 at admission) are exposed, and their siblings inherit the same pools through the cache.
  Without cache hits, only those 8 continuations are exposed.
- **0.7.0:** prefills are unaligned at every D except 1 and 5, so every 0.7.0 continuation carries about 1 affected pool
  per 3,584-token block.
- **Where an event lands:** on every MLA indexer layer (0.9.1: on its owner GPU; 0.7.0: on the straddling pool's DCP
  owner rank).
- **Limit:** the decode rows are assumed to keep decoding through a prefill, and 0.7.0's chunking is source-read, as
  in d7.

### 4.3 Layout

- 0.7.0 runs TP2 + EP2 with DCP2 and DFlash2 × 5.
- 0.9.1 and B1 run TP2 with DCP1 + MLA ownership and DFlash2 × 3.
- Different attention block sizes (3,584 vs 4,352), KV pools (2.76M vs 4.71M tokens), CUDA-graph captures and compile
  caches.

**These are all part of "engine".** A contrast that separates engines does not say which component is responsible.

## 5. Outcomes (frozen; `scripts/bplus_stats.py`, `scripts/analysis_bplus.py`)

### 5.1 Primary

Per engine, P(finish within +64k).
- **Finish:** `finish_reason` "stop" with `stop_reason` a stop-token id (154820 / 154827 / 154829) before
  `max_tokens` = 65,536.
- **Unit:** the prefix. Average its 4 seeds, then average the 12 prefixes.

### 5.2 Contrasts

The three contrasts are paired by prefix (d_i = rate_A,i − rate_B,i):
- **C1 = 0.7.0 − 0.9.1** (primary);
- **C2 = B1 − 0.9.1;**
- **C3 = 0.7.0 − B1.**

C2 and C3 are secondary, with no multiplicity correction.

Statistics:
- **Interval:** percentile bootstrap over prefixes, 20,000 draws, seed 20261016.
- **Test:** exact two-sided sign-flip over all 2^12 sign patterns of d_i.

### 5.3 Operational screens and readings

Each screen has a stated operational meaning, not a causal boundary.
- **EFFECT(A−B):** mean ≥ +0.15, 95% interval lower bound > 0 and sign-flip p ≤ 0.05.
- **REVERSE:** the mirror of EFFECT.
- **SMALL:** |mean| ≤ 0.10 and the 90% interval inside ±0.20.
- **PREFIX_HIGH:** η² ≥ 0.30. η² is the share of the variance of all 144 outcomes explained by the prefix (12 groups of
  3 engines × 4 seeds).

Readings:

| reading | rule | meaning |
|---|---|---|
| **R-E** | EFFECT(C1) | engine continuation effect supported: from identical 0.7.0-built 100k states, 0.7.0's continuation finishes more often, by at least half the descriptive gap |
| **R-E−** | REVERSE(C1) | the opposite direction is supported |
| **R-S** | SMALL for C1, C2 and C3, and PREFIX_HIGH | state-dominated: the 100k state sets the outcome and the continuations do not separate (small is not proof of equality) |
| **R-I** | otherwise | inconclusive under these rules |

- **C2 and C3** are reported as EFFECT, REVERSE, SMALL or unresolved.
- **C1-ONLY fallback** (frozen): applies only if 0.7.0 and 0.9.1 complete and B1 was not started because of the budget
  rule. The reading is then R-E, R-E− or R-I on C1 alone, and R-S is unavailable.
- **Ceiling and floor cases:** a run where every continuation finishes, or none does, reads R-I. With no variance,
  PREFIX_HIGH is undefined.

**Threshold choice and power** (`scripts/power_bplus.py` → `out/power.json`; 2,000 simulations per scenario, 2,000
bootstrap draws inside each; the frozen screens are imported) [M on the model, I on the model's inputs]:

*Model:*
- The prefix has a latent logit state a_i ~ N(μ, σ²).
- Engine shifts δ_e are calibrated so that the mean over prefixes of p_ie hits the target rate.
- Each of the 4 seeds is an independent Bernoulli draw.

*Inputs:*
- 0.7.0: +64k finishing from 100k = 0.55 (rep-F at risk).
- 0.9.1: 0.31 to the cap, from the question-matched table, scaled by 0.7.0's +64k / to-cap ratio (0.55 / 0.66), giving
  0.26.
- The full gap is therefore 0.29.

| scenario (0.7.0 = 0.55; B1 = 0.9.1) | σ = 0 | σ = 1 | σ = 2.5 |
|---|---|---|---|
| full gap (0.9.1 = 0.26): P(R-E) | 0.70 | 0.77 | 0.81 |
| half gap (0.40): P(R-E) | 0.18 | 0.21 | 0.24 |
| no gap: P(R-E) / P(R-E−) | 0.011 / 0.013 | 0.011 / 0.016 | 0.008 / 0.006 |
| no gap: P(R-S) | 0.00 | 0.04 | 0.40 |
| full gap: P(R-S) (false state reading) | 0.00 | 0.00 | 0.002 |
| median η² under the scenario (no gap) | 0.07 | 0.22 | 0.50 |

Sensitivity:
- **Base rate 0.42** (this sample's own expectation if the state dominates):
  - P(R-E) at the full gap: 0.80-0.82;
  - P(R-S) at no gap and σ = 2.5: 0.43;
  - type I ≤ 0.012.
- **C2 power** when the fixes close the full gap (B1 = 0.7.0): 0.74.

Why these thresholds:
- **Δ_E = 0.15** is half the descriptive gap. Power is flat over 0.10-0.20 (0.74-0.78 at the full gap, σ = 1), and
  0.15 keeps P(R-E) under no gap at about 1%.
- **η² ≥ 0.30** lies above the 95th percentile of η² with no state dependence (0.12-0.13). With no gap, P(R-S) is 0 at
  σ = 0, 0.03 at σ = 1 and 0.50 at σ = 2.5 for η² thresholds 0.20-0.40.
- **R-S power is limited** by the equivalence requirement (90% interval within ±0.20 on all three contrasts with
  12 prefixes). The design detects a full continuation effect well (0.70-0.81) and a half effect poorly (about 0.2). It
  supports a state reading only when the state effect is strong (σ ≈ 2.5: 0.40). Otherwise the likeliest reading is R-I.
  This is stated in advance.

### 5.4 Secondary (descriptive, never the reading)

- **Finishing measures:**
  - P(any stop), including the `</s>` stop string;
  - median tokens to finish;
  - P by original stratum.
- **Loop measures:**
  - **Onset:** loop onset in the continuation with the discovery's frozen R50 rule. `trajectory.py` is a byte-identical
    copy (sha256 c24267b1…, imported and hash-checked). The rule runs on prefix reasoning + continuation reasoning, so
    repetition of prefix text counts as in the discovery. The onset is reported as tokens after the cut.
  - **Loop type:** the discovery's E / P rule (short-period exact, long-period exact, fragmentary, none).
  - The count of finishes from inside terminal repetition.
- **Text and agreement measures:**
  - chars/token of the continuation reasoning;
  - within-prefix agreement (prefixes with unanimous seeds, ICC(1) per engine and pooled, η²);
  - covariates at 100k against the pooled prefix rate (Spearman, n = 12);
  - prefix-cache hits, decode tok/s, and preemption lines.
- **Caveats:**
  - All states are 0.7.0-built, one per question.
  - The 12 questions are not a random sample of GPQA. They are the long-reply panel's most frequent long questions.

## 6. Validity gates (any failure = INVALID, no reading; `analysis_bplus.py`)

Per engine:
1. **Coverage:** every planned (k, prefix, j) has exactly one record, and no other record exists. All are HTTP 200 with
   no error.
2. **Seeds:** seed as planned, eligible in the pre-run inventory, and all 144 distinct.
3. **Prompts:**
   - the sent ids hash and length equal the frozen prefix;
   - the server-echoed `prompt_token_ids` hash equals it, so the ids received are byte-identical across engines;
   - `prompt_tokens` equals the prefix length.
4. **Parameters:** the sampling-parameter hash equals the plan's.
5. **Finish fields:** `finish_reason` is stop or length, and length only at 65,536. The generated ids count equals
   `completion_tokens`. The raw file holds one record per request, whose ids and text hash to the request log.
6. **Configuration identity** with the engine's reference record:
   - the server log's non-default args, vLLM version, model snapshots, logged environment names, KV pool tokens and
     CUDA-graph mode;
   - references: 0.7.0 = the rep070-F server logs; 0.9.1 = the 0.9.1 record's server log; B1 = the B1 screen's server
     log;
   - the image Id from `docker inspect` equals the pinned image;
   - ports are published on 127.0.0.1 only.
7. **Warm-up gates** (as rep070):
   - one engine start;
   - 9 release warm-up lines ending with the ready line;
   - exactly the warm-up's own 8 `/v1/completions` requests before the ready line;
   - the first client request after the last timestamped line before it;
   - the runner's ready-file + API gate.
8. **Error lines:** no illegal-memory-access or NaN warning lines during the requests. Preemption lines are counted and
   reported, not invalidating.
9. **Order and pre-flight:** requests submitted in plan order, at most 8 in flight, and the GPU pre-flight passed.

**Incomplete path** (as rep070, never classify a partial run):
- **INCOMPLETE:** any session not started (other than the C1-ONLY budget case), interrupted, watchdog-stopped, failed by
  the client, or with missing records. It reports completed counts only: no contrast, screen or reading.
- No automatic retry. Stop and amend.

## 7. Runner (`runner/run_bplus.sh`)

It reuses the rep070 runner's safety design.

**Refusals** (nothing is created or started before all pass):
- `APPROVED_GPU_H` must be in (0, 31.6].
- The package and its private data:
  - the package SHA256SUMS verifies;
  - pinned recipe env files, both stacks' launchers and the cache-inventory script are unchanged;
  - `PRIVATE/` matches the manifest, with modes 0700 / 0600;
  - the three snapshot listings are unchanged.
- The launch configuration:
  - no launcher override variable set by the caller (both stacks' variables, plus IMAGE, PORT and
    GPU_MEMORY_UTILIZATION);
  - each launcher resolves to the pinned image and 127.0.0.1:8001;
  - image Ids are checked (the 0.7.0 and 0.9.1 digests, the B1 id), and the B1 tag must not have moved.
- The machine state:
  - no run folder exists;
  - no GLM container, and no container of the three images;
  - no B+ client and no lm_eval process;
  - port 8001 free and ≥ 5 GiB free on /data;
  - the GPU pre-flight (desktop graphics clients filtered as rep070; GPU0 ≤ 1 GiB, GPU1 ≤ 4 GiB, utilization ≤ 10%).

**`DRY_RUN=1`** runs every refusal read-only, then exits.

**Each session:**
1. GPU pre-flight, with up to 300 s for release after the previous session.
2. Budget rule: start only if the session's frozen ceiling wall still fits (v070 25,485 s, v091 15,449 s, b1 15,581 s).
3. Start a fresh server by digest through the stack's own launcher, via the recipe's `env.sh`.
4. Ready-file + `/v1/models` gate, with a 40-minute limit and engine-failure patterns as in the eval chain.
5. Record the startup markers and compute apps.
6. Run the client.
7. Save the server log and `docker inspect`, then stop the server.
8. Wait for GPU release, write the cost line, and run the compile-cache inventory and diff.
9. A failed session stops the run.

**Run-wide:**
- **GPU log** every 60 s.
- **Watchdog:** it kills the client and stops the server at APPROVED_GPU_H / 2 GPUs of wall time from script start.
- **Cleanup:** a trap runs on EXIT, INT and TERM: client, server, cost line, finalize. The script does not use `set -e`.
- **Finalize:** the frozen analysis runs in a CPU-only container.

**Client** (`scripts/client_bplus.py`, standard library only):
- It refuses unless the prefix hashes, the plan parameters and the served model match.
- It records the server-side render check.
- It sends 48 requests in the fixed order at 8 in flight, with no retry.
- Per request it logs (no text):
  - prefix id, j, seed, and the sent and server-echoed prompt-id hashes;
  - HTTP status, finish_reason and stop_reason;
  - prompt, completion and cached tokens, and the `</think>` position;
  - hashes of the generated ids and text, and timings.
- Generated text and ids go to `run/raw/<engine>.raw.jsonl.gz`: mode 0600 in a 0700 folder, never published, one gzip
  member per record.
- The prompt ids are not stored again; only their hash is.

## 8. Cost (`scripts/cost_bplus.py` → `out/cost.json`; rates from `scripts/rates_from_logs.py` → `checks/rates.out`)

**Session model:**
- **Wall:** start-up + all prefill (conservatively treated as stalling every slot) + the decode makespan of 48
  continuations list-scheduled on 8 slots in plan order + shutdown.
- **Occupied GPU-hours** = 2 × wall.
- **Decode:** a continuation decodes min(finish, 65,536) tokens. Finishers are uniform over the horizon (rep-F: 44% of
  the +64k finishers in the first half).
- **Prefill tokens** per engine: 1.39-1.45M with cache hits, 4.82M without.

**Inputs:**

| | per-request decode, central / pessimistic (tok/s) | prefill (tok/s) | start-up to ready (s) | P(finish +64k) central |
|---|---|---|---|---|
| 0.7.0 | 32.7 / 22.4 | 2,410 / 1,200 [I] | 540 / 810 | 0.55 |
| 0.9.1 | 52.9 / 37.8 | 2,770 / 2,410 | 760 / 1,140 | 0.26 |
| B1 | 52.3 / 37.5 | 2,770 / 2,410 | 760 / 1,140 | 0.26 |

Where the inputs come from:
- **Central decode** [M]: the per-request rate with ≥ 7 requests running and total batch context below 200k tokens. That
  bin is the least inflated by the high speculative acceptance of loops: rates rise with batch context in these logs
  because long contexts are mostly loops. It is assumed flat in context for this sparse-attention model [I].
- **Pessimistic decode:** 0.8 × the slowest lifetime rate of a finished ≥ 100k reply (0.7.0 28.0, 0.9.1 47.2, B0 46.9).
  This sits at or below the instrumented B1 measurement at 8 × 92k contexts (36.5 tok/s median, 29.6 minimum).
- **Start-up** [M]: 464-493 s on 0.7.0 and 675-732 s on 0.9.1 / B1, from server logs.
- **0.7.0 long-prompt prefill** was never measured [I].

**Results:**

| | central (median) | pessimistic (90th pct., pessimistic inputs, no cache hits, 0.26 everywhere) | ceiling (99th pct. of the full-horizon session at pessimistic inputs) |
|---|---|---|---|
| 0.7.0 | 6.15 | 12.74 | 14.16 |
| 0.9.1 | 4.72 | 7.72 | 8.58 |
| B1 | 4.76 | 7.78 | 8.66 |
| **total (with inter-session gaps)** | **15.8 GPU-h** | **28.4 GPU-h** | **31.6 GPU-h** (`PLAN_MAX_GPU_H`, watchdog at 56,880 s) |

- **P(exceeding the ceiling)** [M on the model]: < 0.1% for the run, ≤ 0.2% per session. This comes from a run with the
  session rate lognormal (median central, 5th percentile pessimistic) and outcomes at the central rates. Its 95th
  percentile is 20.3 GPU-h.
- **Against the earlier estimate:** the discovery's B+ estimate of 7.8 / 10.3 GPU-h assumed concurrency 16 (3 waves) and
  lifetime decode rates that include loop-inflated acceptance (41 / 67 / 74 tok/s). At the specified concurrency 8 and
  novel-text rates, the central cost is about 2× that figure.
- **Concurrency 16** (not the frozen design; rates × 0.8): about 11.8 GPU-h central. That is the operator's decision.
- **Partial approval:** an approval of at least about 23 GPU-h (0.7.0 + 0.9.1 ceilings) lets the primary contrast
  complete under the budget rule (C1-ONLY fallback).
- **Disk:** at most 144 × 65,536 tokens of raw text and ids, below 115 MB uncompressed (about 25-40 MB gzip). Server logs
  are 1-2 MB per session. 1.9 TB is free on /data [M].

## 9. Files

| path | contents |
|---|---|
| `PROPOSAL.md`, `CHECKS.md` | this design; the CPU checks and their results |
| `scripts/` | selection, prefix build / verify, plan, seeds (via rep070's `seed_scan.py`), D7 exposure, power, cost, rates, client, frozen stats and analysis, `run_cpu.sh`, `source_check.sh`, byte-identical `trajectory.py` and `analysis_col511.py` |
| `runner/run_bplus.sh` | the runner |
| `tests/` | synthetic-run generator and analysis dry runs, refusal tests, session simulation with shims and a fake server; their outputs |
| `out/` | selection, prefix manifest (no text, no reasoning ids), plan, D7 exposure, power, cost |
| `checks/` | source check, prefix verification per image, seed inventory and formula scan, snapshot listing 0490d2f7, rates, privacy scan |
| `PRIVATE/` | prefix token ids, original prompt messages (0700 / 0600; never published) |
| `SHA256SUMS` | every file above (the run folder is created by the runner) |
