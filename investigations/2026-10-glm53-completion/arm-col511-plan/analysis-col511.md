# Frozen analysis: column-511 GPQA completion arm (amendments 11 and 11a)

Written 2026-10-11, before any arm data. Implemented by `analysis_col511.py` (same directory); its sha256 and this
file's are recorded in `SHA256SUMS`. Changing either file after the first arm request is sent requires a dated
amendment; numbers produced by a changed file are labelled as such.

## 1. What is compared

- **Arm:** tpurtell GLM-5.3-Flash 0.9.1 (release image `sha256:a9e985c0...`) plus only the column-511 correction (one
  file, `sparse_attn_indexer_kpool.py`, `phase3/fix.diff`, no probe code), K3.25 weights, three complete GPQA Diamond
  passes under `protocols/gpqa-diamond/v1`: pass p sends request seed 1233 + p, 8 concurrent, temperature 1.0,
  top_p 0.95, 327,680-token cap, shipped graph mode (FULL_AND_PIECEWISE), lm-eval `local-chat-completions` through the
  passive request proxy (`record_proxy.py`, required: finish reasons and token counts come only from it), startup
  warm-up gate as in the eval chain.
- **Comparator (fixed, historical):** the clean 0.9.1 three-pass record: lm-eval directories
  `glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps__pass{1,2,3}`; request logs `...+v0.9.1.requests.jsonl` (pass 1) and
  `...+v0.9.1+ps.requests.jsonl` (passes 2-3); server logs `...+v0.9.1.server.log` (pass 1) and
  `...+v0.9.1+ps.server.log` (passes 2-3; first server block cited); published rows
  `runs/2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3_gpqa-diamond`.
- Unit of pairing: (question, pass), which share prompt, answer order, target and request seed. The two generations
  are not matched trajectories (different engine code); pairing only aligns prompts and sampler seeds.

## 2. Definitions (unchanged from the protocol and the existing tools)

- **Empty reply:** the `flexible-extract` sample's response is empty after whitespace stripping (export_evals.py
  `empty`); it scores as wrong.
- **Raw correct:** lm-eval `flexible-extract` exact match. **Stated correct:** the protocol's audited method: the
  adjudicator copied verbatim from `<local exporter export_evals.py>` lines 233-259 (sha256 `731f27d4...`), replies read
  by hand as committing to no single option counted as not correct. The arm's stated figure is final only after every
  arm reply where stated and raw disagree has been read by hand (the script counts them). Empty replies are not
  correct under either.
- **Cap exhaustion:** an empty reply with `finish_reason` `length` (or completion tokens >= 327,680). **Premature
  stop:** an empty reply with `finish_reason` `stop`. Other or missing finish reasons are reported as such.
- All 594 question-passes per arm are analysed; nothing is excluded. Accuracy is over all requests, not answered-only.

## 3. Primary analysis (direction and test fixed before data)

- Per question q (198): `e_q(arm)`, `e_q(cmp)` = number of its three passes with an empty reply.
- `d_q = e_q(cmp) - e_q(arm)`; statistic `T = sum_q d_q` (= comparator empties - arm empties).
- **Test:** exact one-sided sign-flip (randomisation) test, questions as the unit. Null: each `d_q` is symmetric about
  0 (the two arms' outcomes are exchangeable within a question). Exact enumeration over all `2^m` sign patterns of the
  `m` questions with `d_q != 0` (questions with `d_q = 0` contribute nothing); one-sided p = share of patterns with
  `T* >= T_obs`. Alternative: fewer empties in the arm. alpha = 0.05. The two-sided p is printed for reference only
  (it equals `tools/analyze.py signflip_p` on the same d).
- **Effect:** reduction in empty rate `T / 594`, 95% percentile interval from a question bootstrap (198 questions
  resampled with replacement, each carrying all three passes of both arms; 20,000 replicates;
  `random.Random(20261011)`; percentile indexing as `tools/analyze.py`). Relative reduction `1 - arm/cmp` with a
  bootstrap interval (seed 20261014) as a descriptor.
- **Validity gate (computed, not discretionary):** each pass has 198 questions with both filters; every logged request
  has HTTP 200, the pass's seed, max_tokens 327,680, temperature 1.0, top_p 0.95, no parse error; no re-sent request
  (lm-eval re-sends silently after the 10,800 s timeout; a duplicate message hash within a pass fails); request-log
  content hashes equal the lm-eval reply hashes; every (question, pass) pair agrees on seed, prompt hash, doc hash,
  target hash and message hash; server-rendered prompt token counts agree. A failed check marks the run invalid
  (`valid: false`, exit code 3); its numbers are descriptive only and the cause goes to an amendment before any re-run.
- An incomplete arm (fewer than 594 question-passes, e.g. stopped at the GPU-hour ceiling) is not analysed by the
  primary test; completed passes are reported descriptively. No stopping or extension on interim scores.

### Reading rules (amendment 11; the bands are operational thresholds, not causal boundaries)

Bands on the arm's total empties of 594 (comparator 16): <= 6, 7-12, >= 13.

| Arm empties | One-sided p | Reading |
|---|---|---|
| <= 6 | < 0.05 | R1: large reduction supported by the prespecified analysis = evidence that the correction contributes to better completion relative to the recorded baseline |
| <= 6 | >= 0.05 | R2: inconclusive; no automatic further pass |
| 7-12 | >= 0.05 | R2: moderate reduction without precision = inconclusive; no automatic further pass |
| 7-12 | < 0.05 | R2b (amendment 11b, operator: strong signal required): a modest, statistically detectable reduction; NOT support for the correction as an explanation of the gap; reported as a contribution of modest size; no automatic further pass |
| >= 13 | any | R3: little improvement or deterioration = the localized correction stands but is not supported as the dominant completion explanation |

Illustration (exact, frozen test): if every arm empty falls on a question where the comparator had none, one per
question, 5 arm empties give one-sided p = 0.035 and 6 give p = 0.056 (R2, not R1). The same count placed on the
comparator's own questions gives a smaller p.

## 4. Secondary analyses (descriptive; no reading rule attaches)

1. Raw and stated correctness over all 594 requests: per pass and mean; paired arm - comparator difference in points
   with a 95% question-bootstrap interval (seeds 20261012 raw, 20261013 stated); two-sided sign-flip p for reference.
   **A nonsignificant accuracy test does not establish no accuracy cost**; the interval is the result.
2. Rescued empties (comparator empty, arm answered; arm raw/stated correct among them) and newly introduced empties
   (comparator answered, arm empty; comparator correct answers lost), both-empty pairs, raw accuracy where both
   answered. Ids only.
3. How empty replies ended: cap exhaustion vs premature stop vs other; all finish reasons; answered replies ending on
   `length`.
4. Completion-length distribution: nearest-rank quantiles of completion tokens (answered; all logged), counts of
   answered replies above 32k/64k/128k/200k tokens, paired per-question mean difference (bootstrap seed 20261015).
5. Cost and throughput: per-pass request window, completion tokens, aggregate and per-request tokens/s; occupied
   GPU-hours = 2 x server-session wall-hours from the eval chain log (first `chain start` after the previous
   `chain done` to `chain done`, so failed starts count) or the ledger value when given; raw and stated correct per
   GPU-hour. Server log inside the request windows: max running, waiting lines, peak KV usage, mean generation tok/s
   with >= 8 running, preemption / illegal-memory-access / NaN-warning lines; number of server starts per log.
6. Kernel-route exposure without changing execution: client-side in-flight count from request start and end times
   (share of busy time with 1-4 vs 5+ in flight; per request, share of its life at 1-4, empty vs answered) and the
   server's `Running:` status lines. On 0.9.1 the fused top-k route serves <= 16 decode rows (<= 4 requests x 4
   positions), the tiled route more (amendment 10a); in-flight counts include prefilling requests, so this
   approximates, and does not measure, the in-engine route.
7. Corrupted/NaN index rows: not measurable from GPQA outputs without instrumentation the arm must not carry. Where
   the gate's guard or probe data exist they are reported separately and enter no number above.

## 5. Assumptions and limitations (amendment 11a)

- **Historical control; exchangeability assumed, not designed.** The exact sign-flip test against the historical
  record relies on an exchangeability assumption: under the null each question's difference `d_q` is symmetric about
  0. Matching the recipe, image ancestry, weights, prompts, sampler settings and request seeds does **not** make this a
  randomized experiment: the arm runs later, in different server sessions, with uncontrolled other GPU load, cache
  state and timing, and any undetected drift is confounded with the correction. The p-value is valid only to the
  extent that assumption holds.
- **A positive result supports improvement relative to the recorded baseline only.** It does not show that the
  column-511 drop explains the 0.7.0-0.9.1 completion gap; a contemporaneous control is priced separately before any
  such claim.
- **Tail-ring defect.** The unresolved kpool tail-ring defect (attention metadata built without positions; tail
  writes to the shared null block and occasionally into live blocks) is present in both the comparator and the arm.
  This limits what a negative completion result could exclude: it cannot rule out that the column-511 correction
  improves completion in an engine whose tail ring is also correct, or an interaction between the two defects.
- **Corrupted/NaN index rows** are reported separately where measurable (4.7); the correction does not repair them.
- Empty replies are a low-rate tail (16 of 594 in the comparator, on 11 questions): power for moderate reductions is
  limited (section 6). The arm's stated correctness depends on a hand read of disagreements.
- The same seed under a different engine is not a matched trajectory.

## 6. Power (CPU simulation, `power_col511.py`, seed 20261011, 20,000 simulations per alternative)

- **Model.** Per-question empty probability `p_q`; empirical-Bayes beta-binomial: Beta(a, b) fitted by maximum marginal
  likelihood to the comparator's per-question empty counts (of 3 passes): mean 0.0261, a + b = 2.04 (a = 0.053,
  b = 1.988), intra-question correlation 0.33 (empties cluster strongly by question). Each simulation draws every
  `p_q` from its posterior Beta(a + k_q, b + 3 - k_q) (question-level clustering) and the three passes independently
  given `p_q` (pass-level independence). The frozen test (`analysis_col511.signflip`) is applied at alpha 0.05.
- **Alternatives (arm).** Null: arm `p_q` = comparator `p_q`. (i-a) 0.7.0's per-question EB estimates under 0.7.0's
  own fit: its 4 empties on 4 questions show no overdispersion, the fit sits at the bound and every question gets the
  pooled rate 0.0067. (i-b) 0.7.0's counts with the comparator fit's concentration (heterogeneity kept). (ii) every
  comparator `p_q` x 0.5. (iii) every comparator `p_q` x 0.25 (also "0.7.0's overall rate on 0.9.1's question profile").
- **Comparator handling.** Primary: **conditional on the observed comparator** (its 594 outcomes fixed; only the arm is
  simulated, the comparator's `p_q` uncertainty propagated through the posterior). Justification: the comparator is
  fixed historical data and the real analysis conditions on it, so the relevant operating characteristic is the
  probability of each reading given that record. Reference: **resample both** arms from the same `p_q` draw (power of a
  hypothetical replication with a contemporaneous comparator of the same size).

| Alternative (arm) | Mean arm empties | Conditional power p < 0.05 (MC se) | P(R1) | P(<= 6 / 7-12 / >= 13) | Resample-both power (MC se) |
|---|---:|---:|---:|---|---:|
| Null (same p_q) | 15.8 | 0.023 (0.001) = size | 0.006 | 0.006 / 0.205 / 0.789 | 0.033 (0.001) |
| (i-a) 0.7.0 EB, own fit | 4.0 | 0.819 (0.003) | 0.819 | 0.891 / 0.109 / 0.000 | 0.709 (0.003) |
| (i-b) 0.7.0, comparator concentration | 4.0 | 0.871 (0.002) | 0.855 | 0.875 / 0.125 / 0.000 | 0.720 (0.003) |
| (ii) 50% reduction | 7.9 | 0.514 (0.004) | 0.334 | 0.335 / 0.605 / 0.060 | 0.472 (0.004) |
| (iii) 75% reduction | 3.9 | 0.949 (0.002) | 0.897 | 0.898 / 0.101 / 0.000 | 0.845 (0.003) |

A 0.7.0-sized reduction is detected about 4 times in 5 under the conditional design (more when the remaining empties
sit on the comparator's hard questions, (iii); less when they fall elsewhere, (i-a)); a halving is detected about half
the time and lands in the large band a third of the time. Under the null the test is conservative in this model
(conditional size 0.023). The figures rest on a beta-binomial fitted to 16 events on 11 questions and are approximate.

## 7. Files and commands

- `analysis_col511.py`, this file, `power_col511.py`, `run_dryruns.sh`; sha256 in `SHA256SUMS`.
- Arm invocation after the run: the comparator block of `run_dryruns.sh` unchanged, with `--arm-pass` the three arm
  pass directories, `--arm-reqlog` the arm's proxy log(s), `--arm-serverlog`, `--arm-chainlog`, and `--arm-gpu-hours`
  from the cost ledger.
- Dry runs on existing records (pipeline proof, not results): 0.7.0 as the "arm" against 0.9.1; 0.9.1 against itself.
