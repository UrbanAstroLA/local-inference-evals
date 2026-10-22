# E: the prefix-source x engine cross (frozen before data; re-frozen under amendment 22ah)

2026-10-21, re-frozen 2026-10-22. **Prepared, not run. Zero GPU was used to build this package.** CPU-only containers
(`--network none`, `CUDA_VISIBLE_DEVICES=`, no `--gpus`) on the existing local images did the tokenization, source
reading, gate calibration, power simulation and analysis dry runs. No model server was started and no request was sent
to one.

Labels: **[M]** measured here by running code; **[S]** read from source or configuration; **[R]** reproduced against an
earlier package's frozen output; **[I]** inferred or assumed.

**Re-freeze (amendment 22ah).** The first freeze (22ac, SHA256SUMS 0be97e4c…) had D's 8 prefixes. Under 22ad / 22ag the
operator added D2's prefixes, and exactly four things changed:
1. **Prefix set:** 13 = D's 8 (unchanged, P01-P08) + 5 from D2 (P09-P13).
2. **Seeds:** the 64 seeds of P01-P08 are kept; 40 new seeds cover P09-P13.
3. **KV pool:** reported, with D's functional floor, not identity-gated. Every other gate is unchanged.
4. **Power, cost and the runner limit** are recomputed for 13 prefixes; the thresholds are unchanged.

Every changed file keeps a `.pre22ah` copy, and `CHANGES-22ah.json` lists before / after sha256.

**Privacy.** Prefix token ids and the original prompt messages live only in `PRIVATE/` (mode 0700, files 0600), including
the per-source copies in `PRIVATE/src-d091/` and `PRIVATE/src-d091b/`. No question, answer, answer letter or reply text
appears in any other file. Outside `PRIVATE/` there are reply ids, hashes, counts and positions only.
`checks/privacy-scan.txt` records the scan.

**Status of every rule below:** frozen with `SHA256SUMS`. No continuation outcome exists yet. A change after freezing
needs a dated amendment.

**Scope of the numbers.** Every rate here is one model family (GLM-5.3-Flash EXL3 K3.25) and one tokenizer. Rates are not
comparable with other families' fixtures.

## 1. Question

B+ (amendments 22s-22w) continued 12 prefixes **built by stock 0.7.0** at 100k tokens. The engines did not separate:
0.7.0 0.417, 0.9.1 0.417, C1 = +0.000 (95% −0.083 to +0.083), prefix η² 0.67, reading R-I [M, B+ `run/result-22v.md`].
Options D and D2 (22x-22ag) each ran one stock 0.9.1 pass with full text and exported **0.9.1-built prefixes** under
B+'s rule.

E continues the 13 0.9.1-built prefixes on **stock 0.7.0 and stock 0.9.1**. Everything else is exactly as in B+. Two
questions follow:
- **Primary:** from identical 0.9.1-built states, does 0.7.0's continuation finish more often than 0.9.1's (E1)?
- **Descriptive cross:** does the outcome follow the engine that **built** the prefix (source) or the engine that
  **continues** it, and does the B+ engine null (C1 = 0) hold on 0.9.1-built states (interaction E1 − C1)?

## 2. Prefixes

### 2.1 Selection (`out/selection.json`; `scripts/compose_prefixes_e.py`)

**Rule (22ad, result fixed by 22ag):**
- D's 8 prefixes keep their place and their ids.
- D2's selected replies on questions not already in D's set are added in D2's selection order, as P09-P13.
- Both sources applied B+'s rule unchanged (d091 check 7 reproduced B+'s selection byte for byte) [R].

**D** (seed 70704): 8 of 10 eligible, all taken. Excluded: q127p4 (R50 at 100k 0.71) and q149p4 (R50 1.0).

**D2** (seed 70705): 10 of 12 eligible (excluded: q8p5, R50 undefined; q127p5, R50 0.57). D2's own selection took 8:
- three are on questions D already has (q79p5, q81p5, q170p5), so they are not added;
- the other five are added: **q69p5, q120p5, q186p5 (capped), q88p5, q71p5 (finished)**.

The composer copies each entry verbatim from its source's byte-identical manifest and private files, changing only the
prefix id. It checks every id list against its source manifest hash: 35 / 35 checks (`checks/compose-prefixes.out`).

| id | reply | source | stratum | original T | R50 @100k | R50 back | prompt | prefix | chars/tok | interj./1k | digit | drift | prefix ids sha256 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P01 | q79p4 | D | capped | 327,680 | 0.017 | 0.039 | 91 | 100,091 | 2.20 | 7.8 | 0.180 | 0 | a5c9d9297b04 |
| P02 | q81p4 | D | capped | 327,680 | 0.000 | 0.010 | 766 | 100,766 | 2.41 | 9.5 | 0.091 | 0 | 04e436121fa3 |
| P03 | q170p4 | D | capped | 327,680 | 0.000 | 0.004 | 217 | 100,217 | 3.13 | 21.0 | 0.042 | 0 | 78709874fc4f |
| P04 | q109p4 | D | capped | 327,680 | 0.000 | 0.003 | 478 | 100,478 | 3.42 | 9.3 | 0.054 | 2 | 7c8d4660c334 |
| P05 | q13p4 | D | capped | 327,680 | 0.003 | 0.000 | 131 | 100,131 | 4.22 | 6.5 | 0.002 | −1 | 277345e19c41 |
| P06 | q145p4 | D | finished | 120,630 | 0.000 | 0.000 | 285 | 100,285 | 3.42 | 2.7 | 0.021 | 2 | 1b2f92c71ec9 |
| P07 | q121p4 | D | finished | 106,839 | 0.041 | 0.050 | 190 | 100,190 | 3.12 | 14.4 | 0.033 | 0 | 9540bbf99a7f |
| P08 | q27p4 | D | finished | 114,599 | 0.000 | 0.000 | 308 | 100,308 | 4.31 | 1.4 | 0.013 | 4 | 0edde6326ad0 |
| P09 | q69p5 | D2 | capped | 327,680 | 0.000 | 0.000 | 148 | 100,148 | 2.80 | 16.3 | 0.064 | 1 | 1b25420a6ae9 |
| P10 | q120p5 | D2 | capped | 327,680 | 0.025 | 0.058 | 276 | 100,276 | 2.50 | 7.6 | 0.103 | 2 | 99072f96cbe6 |
| P11 | q186p5 | D2 | capped | 327,680 | 0.000 | 0.000 | 308 | 100,308 | 2.80 | 10.7 | 0.091 | 2 | 5ef0afb00f19 |
| P12 | q88p5 | D2 | finished | 137,924 | 0.000 | 0.000 | 229 | 100,229 | 3.57 | 2.3 | 0.021 | 6 | eed6e6e6c6be |
| P13 | q71p5 | D2 | finished | 124,363 | 0.000 | 0.000 | 178 | 100,178 | 3.21 | 3.5 | 0.033 | 5 | 07d73045479c |

- **Strata:** 8 originally capped, 5 originally finished, on 13 distinct questions.
- **Original continuations:** all 5 finished originals finished within +64k of the cut (+6,241 to +37,617 reasoning
  tokens). No capped original finished; their loop onsets fell at 124,796-241,801. Under a pure-state reading the
  expected rate in this sample is 5 / 13 = 0.385 [I]; section 5.3 uses it as a sensitivity case.
- **Shared questions with B+ (the question-matched rows):** q79, q81, q145, q69, q120, q186, q88 and q71. Original
  outcomes in E vs B+:
  - capped in both: q79, q69, q120;
  - finished in both: q145, q71;
  - opposite: q81 (E capped, B+ finished), q186 (E capped, B+ finished), q88 (E finished, B+ capped).
- **Noted, not acted on:** D2 also had two eligible capped replies outside its own 6-per-stratum take: q109p5 (a question
  D already has) and q161p5 (a new question). 22ag fixes the set to D2's selected replies, so q161 is not included.

### 2.2 Construction and reproduction

Both sources built their prefixes with **B+'s `build_prefixes.py`, unchanged** (1ebbec2e…), in the **0.7.0 image**, with
the 0.7.0 template (34d5ee66…) and tokenizer: prompt ids + the first 100,000 re-tokenized reasoning ids. They ran through
d091's wrapper `prefix_build_d091.py`, which sets only `REQLOGS`.

This package rebuilt both sources in scratch folders under `PRIVATE/` with that code [M]:
- **D** (`tests/prefix-build-repro.txt`): 58 / 58 build checks; the manifest is **byte-identical** to d091's
  (5961c780…); the private files are identical (d15434d3… / 3642dcf6…). 10 / 10.
- **D2** (`tests/prefix-build-repro-d2.txt`): 58 / 58 build checks; the manifest is **byte-identical** to d091b's
  (2ebb5223…); the private files are identical (d24d660c… / a89e5626…). 8 / 8.

### 2.3 Rendering on both engines' images

B+'s verify mode, unchanged, rebuilt every prefix with each image's own `transformers` 5.15.1 / `tokenizers` 0.22.2 and
the template and snapshot each engine **serves** (0.7.0: checkpoint template 34d5ee66…, snapshot 701cd745; 0.9.1:
`--chat-template glm53-zai-a5b45eb.jinja` 0c4099f3…, snapshot 0490d2f7) [M]:
- **D's 8:** 0.7.0 114 / 114, 0.9.1 114 / 114 (`checks/prefix-verify-v070.json`, `-v091.json`);
- **D2's 8:** 0.7.0 114 / 114, 0.9.1 114 / 114 (`checks/prefix-verify-d2-v070.json`, `-d2-v091.json`). On the **5 new
  prompts** the result is 70 / 70 on each image.
- Each 0.9.1 file is byte-identical to its source's.

**Decision (as B+):** every engine receives the sources' ids, and the hash gates are B+'s:
- the runner refuses unless `PRIVATE/prefixes.json` and `prompts.json` hash to the manifest;
- the client refuses unless every prefix hashes to its manifest entry;
- the analysis requires the sent and the server-echoed prompt-id hashes to equal the manifest on every request.

## 3. Continuation

### 3.1 Requests (`out/plan.json`): identical to B+

Body `{model, prompt: ids, max_tokens: 65536, temperature: 1.0, top_p: 0.95, seed, stop: ["</s>"],
return_token_ids: true, stream: false}`. The parameter hash is **c53443e3…, equal to B+'s** [M]. Longest context:
100,766 + 65,536 = 166,302 tokens. The client is B+'s `client_bplus.py`, byte-identical.

### 3.2 Seeds

**Scheme:** seed = **8,260,000** + 10,000·engine + 100·prefix + j (engine 1 = 0.7.0, 2 = 0.9.1; prefix 1-13; j 1-4):
104 distinct seeds in **8,270,101-8,281,304**. **P01-P08 keep the first freeze's 64 seeds**; P09-P13 add 40.

**None has been used before** [M], re-scanned after D2:
- `checks/seed_scan_e.py` (rep070's `seed_scan.py`, only `SKIP_DIRS` changed) covers 11,622 files and 1,348 values,
  including D2's request logs: **104 / 104 eligible**. No value lies anywhere in 8,200,000-8,480,000.
- `checks/raw_seed_scan_e.py` (d091's, only `SKIP` changed) finds no candidate in any raw-body file, D2's included.
- `checks/seed_cross_e.py` → `checks/seed-formula-scan.txt` finds no intersection with B+'s, D's, D2's or rep070-F's
  planned seeds. No seed-line integer lies within 20,000 of the range, and no known formula family intersects it.

**Distinct seeds, no common random numbers:** as B+.

### 3.3 Concurrency and submission order

8 in flight. One fixed order on both engines, using B+'s rule:
- 4 rounds; in each round both strata are shuffled with `random.Random(20261016 + j)` and interleaved, the first stratum
  alternating by round.
- With 8 capped and 5 finished, the longer stratum's remainder follows the 5 interleaved pairs.
- The first 8 in flight are 8 different prefixes, 4 per stratum.
- Every seed 1 is in round 1, before any seed 2.

### 3.4 Sessions

One runner invocation, two sessions, one approval: **stock 0.7.0, then stock 0.9.1**, each on a fresh server started by
digest through the stack's own launcher (127.0.0.1:8001). The order is B+'s. It is confounded with time of day and GPU
thermal state; the runner logs GPU state every 60 s.

### 3.5 Prefix caching: as shipped (as B+)

Seeds 2-4 of a prefix can hit seed 1's cached blocks. B+ measured `cached_tokens` 0 on every 0.9.1 request, so the cache's
role is reported, not assumed.

## 4. Known asymmetries (part of "engine", not controlled)

- **Prefill rebuild:** both engines rebuild the 100k prefix KV by prefill. The original decode-built state is not
  reproduced.
- **D7 in the prefix prefill** (B+'s `d7_prefill_exposure.py`, byte-identical, on E's plan →
  `out/d7_prefill_exposure.json`; its labels say 12 / 48, the counts are over E's 13 prefixes / 52 requests) [E]:
  - 0.7.0: 52 / 52 continuations exposed, mean 0.14% (max 0.18%) of the prefix's index-K pools;
  - 0.9.1: 32 / 52 exposed through the cache, mean 0.07% (max 0.19%).
- **Layout:** 0.7.0 runs DCP2 / EP2 / DFlash ×5; 0.9.1 runs DCP1 / ×3. Attention blocks (3,584 vs 4,352), pools and
  graphs differ.
- **Source and session:** B+'s 0.7.0-built cells ran 2026-10-20/21; E's cells run later, with other seeds. D's and D2's
  prefixes come from two passes at different request seeds.

## 5. Outcomes (frozen; `scripts/analysis_e.py`, using B+'s `analysis_bplus_22v.py` and `bplus_stats.py` unchanged)

### 5.1 Primary

Per engine, P(finish within +64k) on the 13 0.9.1-built prefixes.
- **Finish:** `finish_reason` "stop" with `stop_reason` a stop-token id (154820 / 154827 / 154829) before 65,536.
- **Unit:** the prefix. Average its 4 seeds, then average the 13 prefixes.

### 5.2 Contrast

**E1 = 0.7.0 − 0.9.1**, paired by prefix, with B+'s `contrast()`:
- interval: percentile bootstrap over prefixes, 20,000 draws, seed 20261016;
- test: the exact two-sided sign-flip over all 2^13 patterns.

### 5.3 Screens, reading and power

**Screens: B+'s, unchanged.**
- **EFFECT:** mean ≥ +0.15, 95% lower bound > 0 and sign-flip p ≤ 0.05.
- **REVERSE:** the mirror of EFFECT.
- **SMALL:** |mean| ≤ 0.10 and the 90% interval inside ±0.20.
- **PREFIX_HIGH:** η² ≥ 0.30, over 104 outcomes in 13 groups of 2 engines × 4 seeds.

**Reading: B+'s frozen C1-ONLY rule applied to E1.** With two engines the reading is R-E (EFFECT), R-E− (REVERSE) or R-I
(otherwise). R-S is unavailable. SMALL(E1) and PREFIX_HIGH are reported as screens, not as a reading. Ceiling and floor
runs read R-I.

**Power** (`scripts/power_e.py` → `out/power.json`). The model is B+'s generative model with 13 prefixes, 2,000
simulations per scenario and 2,000 bootstrap draws inside each. The inputs are B+'s: 0.7.0 = 0.55, 0.9.1 = 0.26, full gap
0.29. [M on the model, I on its inputs.]

| scenario (0.7.0 = 0.55) | σ = 0 | σ = 1 | σ = 2.5 | first freeze, 8 prefixes (σ 0 / 1 / 2.5) | B+, 12 prefixes |
|---|---|---|---|---|---|
| full gap (0.9.1 = 0.26): P(R-E) | 0.75 | 0.81 | 0.88 | 0.40 / 0.42 / 0.37 | 0.70 / 0.77 / 0.81 |
| half gap (0.40): P(R-E) | 0.21 | 0.23 | 0.27 | 0.08 / 0.07 / 0.06 | 0.18 / 0.21 / 0.24 |
| no gap: P(R-E) / P(R-E−) | 0.013 / 0.012 | 0.012 / 0.006 | 0.007 / 0.006 | ≤ 0.012 | ≤ 0.016 |
| no gap: P(SMALL E1) | 0.38 | 0.51 | 0.76 | 0.17 / 0.27 / 0.53 | |
| no gap: P(SMALL E1 and PREFIX_HIGH) | 0.00 | 0.17 | 0.75 | 0.00 / 0.12 / 0.50 | |
| median 95% interval width of E1 | 0.35-0.37 | 0.31-0.33 | 0.25-0.31 | 0.31-0.44 | |

**Sensitivity:** at base 0.42 (B+'s observed level) or 0.385 (this sample's pure-state expectation), P(R-E) is 0.87-0.89
at the full gap and 0.25-0.31 at half. Type I is ≤ 0.015 per direction.

**What this means, stated before any data:**
- **With 13 prefixes the design detects the full discovery gap well** (0.75-0.89), comparable to B+.
- **It detects half the gap poorly** (about 1 run in 4). An R-I is then likely and would not be evidence of no effect.
- **False positives stay low:** ≤ 1.5% per direction.
- **The thresholds are B+'s** and were not changed.
- **The interaction (descriptive).** With B+'s C1 prefix differences held fixed, the interaction's 95% interval excludes 0
  in 0.82-0.96 of runs under the full gap and in 0.02-0.05 under no gap.

### 5.4 The cross (descriptive, prespecified; `analysis_e.py` `cross()`)

**Inputs:** E's 104 outcomes and B+'s 96 v070 / v091 outcomes from `ref/bplus-result-22v.json`. That file is a
byte-identical, hash-checked copy of B+'s amended result; it must be COMPLETE with every check passing.
- **Cells:** prefix source (0.7.0-built, 12 prefixes; 0.9.1-built, 13) × engine (v070, v091). Each cell gives P with a
  95% bootstrap interval over its prefixes.
- **Source contrast:** 0.9.1-built minus 0.7.0-built, pooled over engines and per engine. It uses a two-sample bootstrap
  and an exact permutation p over all C(25, 13) splits. **The prefix sets differ by question (8 of 17 questions shared),
  so this contrast is confounded with question.**
- **Question-matched sub-table:** the 8 shared questions. Each row gives both prefixes' original strata and the four
  per-engine rates, plus the mean pooled source difference. It is descriptive only, with no test: the same question gives
  different trajectories, and on 3 of the 8 questions the two prefixes have opposite original outcomes.
- **Interaction:** (E1) − (B+'s C1 = +0.000), with a two-sample bootstrap interval and an exact permutation p.
- **Outcome stratum:** P for originally capped vs originally finished prefixes, per source × engine and pooled.

No reading is drawn from the cross.

### 5.5 Secondary (descriptive, as B+)

- any-stop finishing, and median tokens to finish;
- loop onset in the continuation and loop type (B+'s `loop_metrics()`), and finishes from inside terminal repetition;
- chars/token of the continuation reasoning;
- within-prefix agreement (unanimous prefixes, ICC(1), η²);
- covariates at 100k against the pooled prefix rate (Spearman, n = 13);
- cache hits, decode tok/s and preemption lines.

## 6. Validity gates (any failure = INVALID, no reading)

**B+'s `gates()` runs unchanged** (analysis_bplus_22v.py, sha256 ed915a07…) for both engines, plus a check that seeds are
distinct across the run.

**Every gate except the KV pool is as before:**
- coverage and HTTP 200;
- seeds as planned and inventory-eligible;
- sent and server-echoed prompt-id hashes, and lengths;
- the parameter hash, the finish fields, and generated ids vs the raw file;
- non-default args, vLLM version, snapshots, logged environment names and CUDA-graph mode equal to the reference record;
- one engine start, 9 release warm-up lines ending with the ready line, and only the warm-up's 8 `/v1/completions` before
  ready;
- the first request after the last timestamped line before ready;
- no illegal-memory-access or NaN lines;
- image id and loopback publish;
- plan order, ≤ 8 in flight, GPU pre-flight and the ready gate.

**KV pool (22ah, as D under 22ad): reported, not identity-gated.** It is gated only on D's functional floor:
- the formula: 8 requests × attention block × ceil((cap 327,680 + longest record prompt 2,795 + 16) / block);
- each engine's own block: **0.7.0 2,666,496** (3,584-token blocks), **0.9.1 2,646,016** (4,352-token blocks);
- the check passes if the server log shows exactly one pool line and that pool is at least the floor.

`analysis_e.py` supplies this rule to the unchanged `gates()` through its KV accepted-set parameter (`KVFloor`) and
renames the check so its label states the rule. The reported pool and the floor are written to `session_extras`.

**Incomplete path** (as B+): any session not started (including a v091 budget stop), interrupted, watchdog-stopped,
failed or with missing records reads INCOMPLETE, with counts only.

### 6.1 Gate calibration (`checks/gate_calibration_e.py` → `checks/gate-calibration.txt`, `.json`) [M]

The unchanged `gates()`, with E's configuration, ran on every reference session with a server log. Each session is
presented as a run folder with its real server log and inspect record, and one stand-in request spanning its real request
window.
- **0.7.0:** rep070-F sessions 1 and 2, the 0.7.0 record's +ps session, and B+'s 0.7.0 session: 13 / 13, 13 / 13, 11 / 11
  (no inspect kept), 13 / 13.
- **0.9.1:** the 0.9.1 record's two sessions, B+'s 0.9.1 session, **D's session and D2's session**: 11 / 11, 11 / 11,
  13 / 13, 13 / 13, 13 / 13.
- **Every configuration gate passes on every reference session.**
- **Markers-only starts:** all 11 (7 on 0.7.0, 4 on 0.9.1) are above the floor. Margins: 0.7.0 +83,103 to +101,744
  tokens; 0.9.1 +2,061,499.
- **Floor inputs** are measured from the references: the attention blocks (3,584 and 4,352) and the longest record prompt
  (2,795).
- **What the floor fixes:** B+'s 22v set still rejects the 0.7.0 image's own 2,749,599 starts of 2026-09-30 and 10-01. The
  first E freeze accepted only the three observed sizes. With the floor, a fourth 0.7.0 size can no longer invalidate the
  run (dry run `kv-unseen-above-floor`).

## 7. Runner (`runner/run_e.sh`)

This is B+'s runner. The changes from B+ (`checks/runner.diff`):
- `set -u`;
- engines v070 v091;
- the B1 entries, B1 pin and B1 tag check dropped;
- a package-path guard;
- finalize runs `scripts/analysis_e.py`;
- under 22ah, **`PLAN_MAX_GPU_H` 19.9** and per-session budget walls of **20,700 s (v070) and 14,941 s (v091)**.

All refusals, pins, snapshot listings, digest checks, the GPU pre-flight, the watchdog at APPROVED_GPU_H / 2 of wall, the
budget rule, the cleanup trap and no-retry are B+'s.

## 8. Cost (`scripts/cost_e.py` → `out/cost.json`, `checks/cost.out`)

The inputs are B+'s measured sessions: 5.531 GPU-h (v070) and 3.879 GPU-h (v091) for 48 continuations each, with
start-ups of 467 and 680 s and 48 measured request durations each. E has 52 continuations per engine. GPU-hours =
2 × session wall.

| | start-up (s) | central: B+ scaled to 52 | simulated median (p95) | pessimistic (p95, all to horizon) | ceiling (p99.9 pessimistic) | P(exceed ceiling): central / pessimistic model |
|---|---:|---:|---:|---:|---:|---|
| 0.7.0 | 467 (pess. 810) | 5.97 | 6.25 (7.83) | 9.44 | **11.5** | < 0.005% / 0.07% |
| 0.9.1 | 680 (pess. 1,140) | 4.17 | 4.33 (5.41) | 6.78 | **8.3** | < 0.005% / 0.09% |
| **run** | | **10.2 GPU-h** | 10.65 (12.5) | 15.4 | **19.9** (watchdog 35,820 s) | 0 of 20,000 / 0 of 20,000 |

- **Models:** the simulation list-schedules 52 requests on 8 slots in plan order, with durations resampled from B+'s 48
  and a lognormal session speed factor (5th percentile 0.8). The pessimistic model runs every continuation to the
  horizon, with B+'s pessimistic start-up.
- **For reference:** if all 52 ran at the slowest measured full-horizon duration, the run would cost 9.96 + 7.05 = 17.0
  GPU-h.
- **Approval:** E1 needs both sessions. An approval below about 19.9 can let the budget rule stop v091, which reads
  INCOMPLETE.
- **Disk:** at most 104 × 65,536 tokens of raw text and ids, about 25-40 MB gzip.

## 9. Files and what differs from B+

`checks/identity.txt` (`checks/identity_check.sh`) proves by sha256 that these are byte-identical:
- **to B+:** `trajectory.py`, `analysis_col511.py`, `bplus_stats.py`, `analysis_bplus_22v.py`, `build_prefixes.py`,
  `client_bplus.py`, `d7_prefill_exposure.py`, `tests/fake_server.py`, `checks/snapshot-0490d2f7.txt`, and the `ref/`
  copies of B+'s result-22v, manifest, plan and power;
- **to D:** `prefix_build_d091.py`, `ref/d091-selection.json`, `ref/d091-prefix_manifest.json`, `PRIVATE/src-d091/*`,
  `checks/prefix-verify-v091.json`;
- **to D2:** `ref/d091b-selection.json`, `ref/d091b-prefix_manifest.json`, `PRIVATE/src-d091b/*`,
  `checks/prefix-verify-d2-v091.json`;
- **to B+'s request definition:** the plan's model, params, params hash and order seed.

The composed `out/` and `PRIVATE/` prefix files are checked entry by entry against their sources
(`checks/compose-prefixes.out`).

**Modified copies,** each with its diff in `checks/`:
- `plan.py`, `run_cpu.sh`, `run_e.sh`, `source_check.sh`, `make_sums.sh`;
- the tests;
- the seed and privacy scans.

**New code:**
- `analysis_e.py`, `power_e.py`, `cost_e.py`, `compose_prefixes_e.py`;
- `gate_calibration_e.py`, `seed_cross_e.py`, `identity_check.sh`;
- `prefix_build_repro.sh`, `prefix_build_repro_d2.sh`.

| path | contents |
|---|---|
| `PROPOSAL.md`, `CHECKS.md`, `CHANGES-22ah.json` | this design; the CPU checks; before / after sha256 of every file changed by 22ah |
| `scripts/` | plan, client, analysis (E + B+'s unchanged modules), composer, power, cost, run_cpu, source check, prefix build wrapper, D7 exposure |
| `runner/run_e.sh` | the runner |
| `tests/` | synthetic runs and analysis dry runs, prefix build repros (D, D2), refusal tests, session simulation, real dry run; outputs |
| `out/` | composed selection and manifest, plan, power, cost, D7 exposure |
| `checks/` | seed scans, gate calibration, prefix verification per source and image, source check, identity, compose, diffs, privacy scan |
| `ref/` | byte-identical copies of B+'s result-22v, manifest, plan and power, and of D's and D2's selection and manifest (no text) |
| `PRIVATE/` | prefix token ids and prompt messages, composed and per source (0700 / 0600; never published) |
| `*.pre22ah` | the first freeze's version of every file 22ah changed |
| `SHA256SUMS` | every file above except `run/`, test scratch and `FROZEN-VERIFY.txt` |

## 10. Not verified on CPU

- Actual prefix-cache hits on 0.7.0 (it does not report `cached_tokens`).
- Decode rates on 0.9.1-built prefixes. The costing assumes B+'s per-request durations carry over.
- The render endpoint's output on each live server (recorded at run time, not gated).
- 0.7.0's scheduler chunking in the D7 estimate (source-read, as in B+ and d7).
- Whether a future session could log more than one KV pool line. The floor check then fails, as in D.
