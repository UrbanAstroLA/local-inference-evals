# Ledger

What was asked, run, found, corrected and withdrawn, in date order.

- Results: [Results page](https://urbanastrola.github.io/local-inference-evals/).
- Every statement, graded: [`FINDINGS.md`](FINDINGS.md).
- What else can move a result: [`CONFOUNDS.md`](CONFOUNDS.md).
- File-by-file history: [`CHANGELOG.md`](CHANGELOG.md).

## The questions

1. Which weights and engine release to run on two RTX PRO 6000 GPUs: accuracy, completion, speed, KV capacity.
2. Why some hard GPQA questions do not finish within the 327,680-token budget
   ([investigation](investigations/2026-10-glm53-looping)).
3. Which factors besides the configuration move a result, and how to control them ([`CONFOUNDS.md`](CONFOUNDS.md)).
4. Why tpurtell 0.7.0 left fewer GPQA answers empty than 0.9.1 ([investigation](investigations/2026-10-glm53-completion/README.md)).

## Timeline

**2026-09-29 to 10-01. First GPQA runs.**
- GPQA Diamond on 3.25bpw with tpurtell 0.7.0 and 0.8.0, then 4bpw TR3 (Brandon) on 0.8.0.
- Each run had three passes.

**2026-09-30 to 10-05. Hard-question screens and the kpool fixes.**
- Screens repeated a few hard questions many times per configuration.
- First form of question 2: do runtime issues fixed upstream explain it?
  ([kpool investigation](investigations/2026-10-glm53-kpool-tail)).
- Upstream vLLM's kpool kernel tests: 29 of 33 pass on the 0.7.0 and 0.8.0 images, 33 of 33 with two upstream fixes.
- The upstream vLLM fixes were ported and contributed as [tpurtell PR #5](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/5) and
  shipped in tpurtell 0.9.0.
- This repository was first published on 2026-10-05.

**2026-10-06. A seed correction.**
- The GPQA protocol had said passes used seeds 1235-1237.
- In fact every request carried seed 1234; lm-eval's `--seed` never reaches the request.
- Corrected the same day. Configuration labels were introduced.

**2026-10-07. The DCP1 tail masking issue.**
- A preregistered layout screen ran on tpurtell 0.9.0.
- Reading the DCP1 decode path showed it skipped the newest 1-3 tokens at causal lengths up to 2,043 not divisible by 4.
- An index check on the image's own kernels confirmed it. Decode-vs-prefill KL below 2,044 tokens: 0.066 → 0.010 with
  a fix.

**2026-10-08. The fix ships.**
- Contributed as [tpurtell PR #6](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/6); released in
  tpurtell 0.9.1 the same day.
- Noted: lm-eval's raw `flexible-extract` filter reads some correct answers as wrong.

**2026-10-09. A fixed-seed control, and two withdrawals.**
- A new screen gave every repeat its own request seed.
- A control arm sent seed 1234 on every repeat, as the earlier screens had: 11 of 12 failed, against 3 of 12 with
  distinct seeds (Fisher p = 0.003).
- So repeats that share a seed are not a real sample. Two sets of results were withdrawn (below).
- Also: an audited stated-answer score beside the raw score; kernel tests on the 0.9.0 and 0.9.1 release images
  (33 of 33).

**2026-10-10. The GPQA redo.**
- Three passes per configuration, one request seed per pass, for three configurations.
- One configuration's passes differ by 0.5 to 3.5 points.
- Documents reorganized around results, with this ledger and the confounds page.
- New [completion investigation](investigations/2026-10-glm53-completion/README.md), Phase 0 (existing data, no GPU):
  0.7.0's higher GPQA score comes entirely from question-passes 0.9.1 left empty. Where both answered (577 of 594),
  raw accuracy is identical (516 vs 516).
- Completion investigation, [Phase 1](investigations/2026-10-glm53-completion/README.md#phase-1) (12 decode-vs-prefill
  runs, 7.18 GPU-hours): no step found 0.9.1's decode agreeing worse with its prefill than 0.7.0's; at 8,000-15,999
  tokens 0.7.0's agreed worse. Below 2,044 tokens disagreement is higher with speculation on, in both engines.
- Completion investigation, Phase 1 [steps 5-6](investigations/2026-10-glm53-completion/README.md#step-5) (4 runs, 3.37
  GPU-hours): with the two kpool fixes, 0.7.0's disagreement at 8,000-15,999 tokens stayed at stock level. 0.9.1 with
  0.7.0's parallel layout rose to stock 0.7.0's level: that layout bundle can reproduce the difference on newer code,
  without saying which part of it does. Neither step measured completion.
- Correction: earlier text said both servers loaded the same vendored chat template. They did not. 0.7.0 served the
  checkpoint's own template, 0.9.1 the corrected Z.ai template. Both render all 198 GPQA prompts byte-identically, so
  no number changes.

**2026-10-10, evening. What the long-context measure covered; a completion pilot.**
- Trace audit of the 24 traces kept by Phase 1 steps 5-6 (CPU;
  [amendment 7](investigations/2026-10-glm53-completion/PLAN.md#dated-amendments)): in every trace the model had closed
  its reasoning and ended its reply at 637-5,377 generated tokens. The protocol keeps generating past the end
  (`ignore_eos`), so the 8,000-15,999-token bin of steps 4-6 is entirely continuation after the reply had ended. That
  measure says nothing about non-completion. Step 4 kept no traces; the same pattern is likely there, not verified.
- [Completion pilot](investigations/2026-10-glm53-completion/README.md#completion-pilot-and-replication) (amendment 8,
  manifest frozen before the run): 0.9.1 at its default layout against the 0.9.1 EP2/DCP2 diagnostic bundle (step 6's
  layout), three target questions and a control, original and fresh seeds, 8 requests in flight, normal stopping, caps
  of 65,536 or 131,072 tokens. Original seeds: no reply unanswered at the cap in either layout. Fresh seeds: 2 unanswered
  at the cap with the default layout (q109, q170), 0 with the bundle; one draw each, within chance. 2.01 GPU-hours.

**2026-10-11. Replication.**
- Four new fresh seeds each on q109 and q170, same arms and caps (amendment 9): 0 vs 0 unanswered. With the pilot's
  fresh seeds, 2 of 10 vs 0 of 10 target draws (one-sided p ≈ 0.24): no useful separation. 1.54 GPU-hours; the
  investigation's total 14.10.

**2026-10-11. Source audit, and which pool the 511-pool slice drops.**
- [Kernel and DCP source audit](investigations/2026-10-glm53-completion/README.md#kernel-and-dcp-source-audit)
  (amendments 10-10a; CPU; three independent audits of the 0.7.0, 0.8.0 and 0.9.1 source): no defect unique to 0.9.1 in
  attention, the DCP combine, record readers, graphs or scratch. The known indexer write defects are 0.7.0's and
  0.8.0's and fixed in 0.9.1. Common to every version: the top-k selects 512 compressed-key pools and the engine keeps
  output columns 0-510 to make room for the newest tokens. No top-k path sorts its output, so the discarded pool (the
  one in column 511) is not guaranteed to be the weakest. Upstream vLLM keeps all 512.
- Correction: the looping investigation's code question 6 said these releases' top-k emits winners first, so the
  discarded pool would be among the lowest-scored. No top-k path sorts its output; a dated note on that question now
  points here ([looping, open questions](investigations/2026-10-glm53-looping/README.md#open-questions)).
- [Stage 1](investigations/2026-10-glm53-completion/README.md#phase-2-stage-1-which-pool-is-dropped) (each release's
  top-k kernels on frozen synthetic inputs; one GPU, no model; 0.10 GPU-hours): on 0.9.1's tiled route (5-8 requests in
  flight at 3 draft tokens) the discarded pool ranked better than 480 of 512 in 94-98% of rows at 1,024-16,384 pools;
  on the fused route (1-4 requests) 0-1%; on 0.7.0's path 3-21%.
- [Stage 2](investigations/2026-10-glm53-completion/README.md#phase-2-stage-2-real-inputs) (real serving inputs
  captured on 0.9.1 in eager mode, replayed through each release's kernels; 1.16 GPU-hours): 88.1% of 9,944 live tiled
  decode rows; fused 0.5%; 0.7.0's path on the identical bytes 6.9%. The route, which follows the number of requests in
  flight, alone decides it. Selected sets matched an fp64 reference.
- [Phase 3](investigations/2026-10-glm53-completion/README.md#phase-3-column-511-intervention) (0.58 GPU-hours):
  switching the two-level fold off did not meet the criterion (the weakest pool in column 511 in 39.8% of tiled rows);
  moving the lowest-scoring selected pool into column 511 did, in 100% of rows offline and in place, with the selected
  set unchanged. This is a correction to the engine's 511-pool selection policy, not a restoration of upstream's
  512-pool policy.

**2026-10-11. The kpool tail ring.**
- In stage 2's second capture, 16 index-K pools were byte-identical across two different requests, some decoding as
  NaN. A CPU forensic pass found raw indexer key and gate rows for tail-ring slots 4-7, written into blocks then owned
  by other requests (amendment 10d).
- Cause (amendment 10e; CPU trace with the real kernels): this build builds attention metadata without token
  positions, which upstream vLLM has passed since GLM-5.3-Flash support landed. Without them, every tail write from
  position 8 goes to a shared null block, and through stale ids left by the runner's start-up warm-up, occasionally
  into another request's blocks. The same omission is in 0.7.0's source. Amendment 10d had nominated a stale request
  row or a block-table staging race; no staging race was found (25,000 steps against a CPU shadow).
- [Verification](investigations/2026-10-glm53-completion/README.md#kpool-tail-ring) (amendment 10f; 1.16 GPU-hours):
  unmodified 0.9.1 in its shipped graph mode sent 98% of tail writes to the null block on each rank, and 6 into live
  attention blocks. With positions passed and a graph-safe write: 0 and 0; 104 of 104 requests completed; throughput
  unchanged.
- A CPU harness running each release's real kernels: 0.7.0 gets 100% of decode-completed pools wrong at 1, 2 and 8
  requests in flight; 0.9.1 0% at 1, 47-57% at 2 and 78-90% at 8. 0.7.0 is the more affected engine; that does not
  exclude a role in its better completion, which is untested (corrected; see 2026-10-11, late). The investigation's
  total: 17.11 GPU-hours.

**2026-10-11. Upstream comparison, gate, completed tail fix, and the completion arm.**
- [Comparison with upstream vLLM](investigations/2026-10-glm53-completion/README.md#comparison-with-upstream-vllm)
  (CPU): all three releases carry the same vLLM development commit, not an ancestor of upstream main. No further
  reachable correctness divergence on the GLM-5.3 paths. Two divergences outside them affect every release and are
  untested: router logits computed in bf16 on this GPU family (upstream enabled the fp32 path for it, vllm#54048), and
  no SwiGLU clamp on the routed experts.
- An independent assessment was adopted (amendment 11): validate the column-511 correction in the shipped graph
  configuration first, then test completion.
- [Gate](investigations/2026-10-glm53-completion/README.md#gate-column-511-correction-in-the-shipped-graph-configuration)
  (amendments 11-11a; 0.72 GPU-hours): passed all four conditions in one full server run under the shipped CUDA graphs,
  with an observe-only device checker. It found the prefill branch affected too (rows with exactly 512 candidates came
  back in index order, so the newest pool was dropped; 2,200 rows); the frozen correction covers it. At least one
  selected pool with NaN bytes in 7.9% (rank 0) and 32.5% (rank 1) of sampled decode rows: the tail-ring defect, not
  repaired by this correction.
- [Tail fix completed](investigations/2026-10-glm53-completion/tail-fix/RESULT.md) (amendment 11; 1.15 GPU-hours):
  positions passed, upstream's ownership guard and the graph-safe write, in two variants (with the sparse-MLA
  causal-length source unchanged, or switched as upstream does). One shipped-graph run each: 0 null-block, 0 foreign
  and 0 attention-block writes on both ranks. The investigation's total before the arm: 18.97 GPU-hours.
- [Comparator audit](investigations/2026-10-glm53-completion/arm-col511-plan/COMPARATOR-AUDIT.md) and
  [frozen analysis](investigations/2026-10-glm53-completion/arm-col511-plan/analysis-col511.md) (amendment 11b, before
  any arm data): the comparator is the existing clean 0.9.1 three-pass record, a historical control; no material
  unexplained configuration difference. Primary test: one-sided exact sign-flip on per-question differences in empty
  replies. Reading: at most 6 empties with p < 0.05 supports a contribution relative to the recorded baseline; 7-12 with
  p < 0.05 is a modest reduction, not support for explaining the gap; otherwise inconclusive or not supported. Power
  0.82-0.87 if the arm matches 0.7.0's per-question rates, 0.95 for a 75% reduction, 0.51 for 50%.
- Correction (cost): amendment 11 asked for up to 13 GPU-hours for the arm. That estimate left out pass 1's separate
  server session; the comparator's three passes had occupied 16.59 GPU-hours (chain start to chain done; 16.63 from
  server start to stop, reconstructed later: see 2026-10-11, late). The ceiling was raised to 18.5 before launch.
- Correction (server starts): notes made while preparing the arm described the comparator's passes 2-3 server log as
  holding more than one server start ("first server block" in the gate result and the frozen analysis). It holds one;
  the frozen analysis script counts one start in each comparator log (dry runs). Nothing compared changes.
- [Completion arm](investigations/2026-10-glm53-completion/README.md#completion-arm) started in the afternoon:
  0.9.1 with the column-511 correction only, three GPQA passes (request seeds 1234-1236, 8 concurrent, 327,680-token
  cap, shipped graph mode), the comparator's session split mirrored, ceiling 18.5 GPU-hours. It finished in the evening
  (next entry). Nothing about its outcome is reported until it has finished and been analysed with the frozen script and
  read by the frozen rules.

**2026-10-11, evening. The completion arm's result.**
- [Completion arm](investigations/2026-10-glm53-completion/README.md#completion-arm) finished and was analysed with the
  frozen script, run unchanged ([receipts](investigations/2026-10-glm53-completion/arm-col511-result)). Every frozen
  validity check passed: seeds and sampling as specified, no re-sent request, all 594 (question, pass) pairs matching on
  prompt, doc, target and message hashes and prompt token counts, the same server-argument fingerprint, and the
  comparator's published rows reproduced.
- Empty replies 15 of 594 (2.53%) with the correction vs 16 (2.69%) on the comparator; per pass 4/5/6 vs 5/5/6; all 15
  at the token cap. One-sided exact sign-flip p = 0.50 over 13 questions that differ; reduction +0.17 points, 95%
  question-bootstrap interval -1.01 to +1.35. Reading R3: the localized correction stands but is not supported as the
  dominant completion explanation. Empties moved between questions: 10 question-passes rescued, 9 newly empty.
- Secondary: raw accuracy 85.0% vs 86.9% (-1.85 points, 95% -5.05 to +1.35), stated 86.7% vs 87.9% (-1.18, -4.04 to
  +1.52; final after the hand read of 12 arm rows, see 2026-10-11, late); not significant, which does not establish no
  accuracy cost.
  Answered completion lengths unchanged (median 3,387 vs 3,508 tokens). 76-84% of busy time had 5 or more requests in
  flight, so the corrected tiled path was heavily exercised.
- In short: the column-511 change is a correctness fix that did not move completion behaviour. Limits: a historical
  control (exchangeability assumed, not randomized); the tail-ring defect, present in both records, limits what this
  negative result can exclude.
- 17.11 GPU-hours (ceiling 18.5). The investigation's cumulative total: 36.08 GPU-hours. Nothing further is scheduled.

**2026-10-11, late. Finishing work (amendment 12; CPU only, no GPU).**
- An independent assessment was adopted (amendment 12): the column-511 completion experiment is closed; the correction
  and its reproduction evidence are kept; no further completion run is bought for it alone.
- Stated answers, final: the 12 arm rows where the stated answer differs from raw were read by hand under the
  protocol's rules. All 12 commit to a single option, the one the adjudicator read; none is hedged. The frozen script,
  re-run unchanged, gives the same figure: stated 515 of 594 (86.7%) vs 522 (87.9%), -1.18 points (95% -4.04 to +1.52).
  Ids and categories are in the
  [completion arm section](investigations/2026-10-glm53-completion/README.md#completion-arm).
- [Occupied GPU-hours reconstructed](investigations/2026-10-glm53-completion/README.md#occupied-gpu-hours-of-the-three-k325-records)
  from start-to-stop timings (2 GPUs x server start to server stop, per session): 0.7.0 record 22.97 (pass 1 7.44,
  passes 2-3 15.53), 0.9.1 record 16.63 (5.15 + 11.48), completion arm 17.11 (5.74 + 11.37). The cost ledger is
  unchanged: cumulative 36.08 GPU-hours.
- Correction (cost): the 0.9.1 record's 16.59 GPU-hours ran from chain start to chain done and left out the server's
  shutdown; start to stop it is 16.63. A planning figure of about 15.5 GPU-hours for the 0.7.0 three-pass record came
  from its passes 2-3 session alone; with pass 1's session it occupied 22.97.
- Correction (interpretation): the corrected 0.9.1 arm's 15 empty replies were described as 0.9.1 reproducing its own
  rate. The arm resembles the 0.9.1 record's rate (16), but it ran a modified engine; it is not a replication of the
  unmodified 0.9.1 record.
- Correction (interpretation): the tail-ring defect was described as not explaining 0.7.0's better completion because
  0.7.0 is the more affected engine. Greater corruption in 0.7.0 does not exclude an accidental completion benefit from
  incorrect computation; the defect's role in completion is untested.
- Proposed, not authorized: a repeatability run of stock 0.7.0's record (original image and settings, the recorded
  per-pass seeds, the original session split), with its analysis frozen before any run. Nothing is running.

## Withdrawn, and what replaced it

> **Withdrawal notice (2026-10-09).** The earlier hard-question screens (`hard-prompt-screen` v0 and v1, 2026-09-30 to
> 10-08) sent request seed 1234 on every repeat. Their failure rates, the layout-bisection statistics and the
> question-level observations drawn from them are withdrawn.
> **Replaced by** the [component screen](investigations/2026-10-glm53-looping/README.md#non-completion-what-the-clean-data-shows)
> (`hard-prompt-screen/v2`): one question per run, a distinct request seed per repeat.

> **GPQA passes that repeated one seed** (published before 2026-10-09) are withdrawn.
> **Replaced by** the [three-pass GPQA records](https://urbanastrola.github.io/local-inference-evals/#gpqa), one request
> seed per pass ([`comparisons/glm53-flash-gpqa-records`](comparisons/glm53-flash-gpqa-records)).

<details>
<summary>Evidence and detail</summary>

- Batching made shared-seed repeats vary, but every repeat drew on the same sampler noise. That variation is not a
  meaningful sample, and the effective sample behind each rate was far smaller than its request count.
- Evidence: question 88 failed 3 of 12 times with distinct seeds and 11 of 12 with seed 1234 on every repeat, all else
  equal. In the eight earlier fixed-seed screens of 3.25bpw DCP1 configurations it had failed 62 of 64 repeats.
- Withdrawn bisection statistics: as released vs the v0.7.0-layout control, with vs without the tail fix, and the
  verdicts built on them.
- Statements built on the shared-seed GPQA passes were rewritten from pass 1 on 2026-10-09
  ([`protocols/gpqa-diamond/v1.md`](protocols/gpqa-diamond/v1.md)).
- Finding 2 of the [kpool investigation](investigations/2026-10-glm53-kpool-tail) rested on the screens and was
  withdrawn with them.
- The withdrawn rows remain in the git history (commit `4fbaad2`). The bisection's preregistration is kept unchanged,
  marked [withdrawn](investigations/2026-10-glm53-looping/preregistration/WITHDRAWN.md).
- **The rule now.** Screens: a distinct seed per repeat (5000 + repeat), one question per run, the question as the unit
  (`protocols/hard-prompt-screen/v2.md`). GPQA: pass *p* sends 1233 + *p*. `tools/verify.py` rejects repeated seeds in
  a v2 run unless it is labelled a fixed-seed control, rejects GPQA passes that share a seed, keeps only repeat 1 in
  v0/v1 runs, and rejects comparisons of v0/v1 runs.

</details>

<a id="single-draws"></a>

<details>
<summary>Single draws kept from the earlier screens</summary>

Repeat 1 of each question from each configuration's first screen. One draw per question: loops and exhaustion occur in
every configuration tested. Not a rate (`tools/analyze.py screen-single`).

| Configuration | Date | q13 | q79 | q88 | q121 | q127 |
|---|---|---|---|---|---|---|
| `3.25bpw · tpurtell 0.7.0 · DFlash2 ×5` | 09-30 | ok | ok | loop | ok | exhaust |
| `3.25bpw · tpurtell 0.8.0 · DFlash2 ×3` | 09-30 | loop | loop | loop | ok | ok |
| `3.25bpw · tpurtell 0.8.0 · DFlash2 ×1` | 10-04 | ok | loop | ok | ok | exhaust |
| `3.25bpw · tpurtell 0.8.0 + kpool fixes ≈ 0.9.0 · DFlash2 ×3` | 10-04 | ok | ok | loop | ok | ok |
| `3.25bpw · tpurtell 0.7.0 + kpool fixes · DFlash2 ×5` | 10-05 | ok | ok | loop | ok | exhaust |
| `4bpw TR3 (Brandon) · tpurtell 0.8.0 + kpool fixes ≈ 0.9.0 · DFlash2 ×3` | 10-05 | ok | loop | loop | ok | ok |
| `3.25bpw · tpurtell 0.9.0 · DFlash2 ×3` | 10-07 | loop | exhaust | loop | ok | ok |
| `3.25bpw · tpurtell 0.9.0 · 0.7.0 layout (DCP2, EP2) · DFlash2 ×3` | 10-07 | ok | ok | loop | ok | ok |
| `3.25bpw · tpurtell 0.9.0 + DCP1 tail fix ≈ 0.9.1 · DFlash2 ×3` | 10-07 | ok | loop | loop | ok | loop |

`4bpw TR3 (Brandon) · tpurtell 0.7.0 + kpool fixes · DFlash2 ×5` could not run at this concurrency: the weights leave a
437,563-token KV pool and the engine crashed when it filled (run kept as INVALID).

</details>

<details>
<summary>Figures corrected along the way</summary>

| Figure | Was | Now | Why |
|---|---|---|---|
| GPQA request seed per pass | 1235-1237 | 1234 on every request before 2026-10-09; 1233 + *p* since | lm-eval's `--seed` never reaches the request |
| How low raw `flexible-extract` runs | about 0.7, then about 1.6 points per pass | 0.5 to 3.5 points per pass, 1.5 on average (16 passes) | the first checked only replies where two filters disagree; the second included passes no longer published |
| Kpool fixes in the 0.9.0 and 0.9.1 releases | kernel files byte-identical to the tested build | kernel tests run on both release images: 33 of 33 | tested directly on 2026-10-09 |
| Run-to-run noise of one GPQA pass | passes sharing seed 1234 | three passes with their own seeds: 0.5 to 3.5 points | a shared seed understates the spread |
| Which pool the 511-pool slice discards | among the lowest-scored, because the top-k emits winners first (code reading) | not guaranteed: no top-k path sorts its output; on 0.9.1's tiled route a pool ranked better than 480 of 512 in 88.1% of live decode rows | measured on 2026-10-11 (completion investigation, stage 2) |
| Completion investigation, cumulative GPU-hour ceiling after amendment 9 | 15.65 | 17.1 (the approved ceilings added) | amendment 9 counted Tranche 1 from consumption rather than from the discovery allocation (amendment 10) |
| GPU-hour ceiling for the completion arm | 13 | 18.5 | the estimate left out pass 1's server session; the comparator's three passes occupied 16.59 (amendment 11b) |
| Occupied GPU-hours of the 0.9.1 K3.25 three-pass record | 16.59 | 16.63 | 16.59 ran from chain start to chain done; reconstructed from server start to stop (2026-10-11) |
| Occupied GPU-hours of the 0.7.0 K3.25 three-pass record | about 15.5 (planning figure) | 22.97 | the planning figure was the passes 2-3 session alone; pass 1 ran in its own session (7.44) |

</details>
