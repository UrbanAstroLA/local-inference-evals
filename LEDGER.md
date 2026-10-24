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

<a id="2026-10-12-early-morning-stage-a-verification-of-d3-d4-d5-in-progress"></a>

**2026-10-12, early morning. Stage A verification of D3, D4, D5 (amendments 13-14).**
- One observe-only server session (shipped graph mode) on the gate fixture and offline comparisons on identical captured
  inputs. D5: omission confirmed, not exercised in this workload (resizes only during profiling). D4: omission and kernel
  support for the limit confirmed; effect not measured. D3: the bf16 fallback is confirmed in the server.
- Offline attempt 3 stopped at control (c): calling the router on the 64-row capture window does not reproduce the
  consumed weights, because the served router's weights depend on batch size (new observation E1; expert selection
  unaffected). Amendment 14, written before any D3 comparison was computed, has control (c) and the corrected comparison
  invoke the router as serving does; the criterion stays bitwise.
- Offline attempt 4: controls (a)-(d) pass on all 1,593 records (103,608 token × layer rows). The fp32-output computation
  selects a different top-8 expert set on 3.15% of rows (95% over 176 requests 3.01-3.29%; over 8 histories
  2.90-3.50%), different weights on every row; by layer 1.2-6.6%, by execution type 3.0-3.2%. The corrected computation
  lies within τ_d of fp64 on every logit. FULL-graph decode is covered by source inspection only. Completion untested.
- Two tooling faults (pageable capture buffer; offline mount over /run) were found by the smoke and offline steps and
  fixed before any comparison.
- 0.6929 of 2.0 GPU-hours (the 0.68 logged while in progress plus 0.0106 for attempt 4).
  [Stage A](investigations/2026-10-glm53-completion/README.md#stage-a-verification-d3-d4-d5).

**2026-10-12, early morning. Correctness bundle gate (amendments 14-14d).**
- An independent assessment was adopted (amendment 14): a minimal verified bundle on 0.9.1 (D1 column-511 correction, D2
  tail-ring positions, D3 fp32-output router, D4 SwiGLU limit 10), a gate, then one GPQA pass. D5 was left out (not
  exercised; its backport reorders warm-up and capture); D6 is unreachable under D2. The operator authorized the gate and
  the screen (amendment 14a).
- s1 (all corrections, observers on) passed every check as specified apart from two: D1 on every checked row including
  FULL-graph replays, D2 with 0 stray tail writes, D3 with fp32-output dispatch on every layer, D4 plans at 10, no
  execution, memory or stopping errors, configuration identical to the historical 0.9.1 record.
- Failed as specified: "D4 compiled kernels" and "P1 probe repeatable" (the 4,240- and 16,240-token probe prompts are not
  greedy-repeatable, in the bundle and in a stock-release control run for the review). Both criteria were revised after
  the gate data were inspected (amendment 14c): D4 by session-specific compiled-kernel evidence (66 kernels with limit 10
  written during s1 by the candidate's plans, none before), P1 by short-prompt repeatability. The original criteria are
  not described as met. D4 is reachable: a limit of 1 changes 6 of 6 probe outputs.
- Before launch (amendment 14d): D3 verified as served (bitwise reproduction of the served logits on 1,000 records, 0
  logits outside τ_d of fp64); the production image's engine files equal the frozen candidate's; the GPQA-shaped phase's
  early stops all have non-empty final content.
- Recorded separately: long-context greedy non-repeatability of the stock release; NaN-byte index rows 0 with the bundle
  against thousands in the 2026-10-11 gate (the tail-ring fix inferred as the cause).
- Correction (gate preparation): the D4 compiled-kernel check walked the plan objects for the kernel's limit fields. It had
  been CPU-tested only with a stand-in kernel object placed in a plan; that the runtime keeps the compiled kernel objects
  was assumed, not verified. It does not keep them, so the criterion could not be met as specified.
- Correction (coordinator): the coordinator's first description of the probe data gave the 16,240-token prompt's
  prefix-cache coverage as identical in both rounds. It differed in s3 and in the release control (0 vs 8,704 tokens cached) and was identical
  only in s1. The 4,240-token prompt's coverage was identical in every session, and it diverged in every session.
- Correction (count): the gate review (amendment 14c) counted three early stops in the GPQA-shaped phase; the client
  records hold four (docs 16, 19, 21, 22). All four have non-empty final content.
- 1.6403 of 2.0 GPU-hours (s1 0.7711, s3 0.4506, release control 0.4111, D3 as served 0.0075; one failed start at 0).
  [Bundle gate](investigations/2026-10-glm53-completion/README.md#bundle-gate-d1d2d3d4-on-091).

**2026-10-12, morning. One-pass GPQA screen of the bundle (amendments 14d-14e).**
- Frozen before generation: image, combined diff, recipe, analysis and runner hashes, and resource-allocation rules by the
  number of empty replies E of 198 (clear promising E ≤ 2 without a regression flag or accuracy concern; similar / no
  clear separation E = 3 without either; similar or worse E ≥ 4).
- Valid. 3 of 198 replies empty, all at the 327,680-token cap, no premature stop; no regression flag, no accuracy
  concern; stated-answer adjudication complete (2 rows, neither hedged). Category: **SIMILAR / NO CLEAR SEPARATION**.
- Against pass 1 (seed 1234) of the 0.9.1 record (5 empty): 3 rescued, 1 newly empty; of the column-511 arm (4 empty): 1
  rescued, 0 newly empty; 0.7.0's pass 1 (context): 0 empty. Secondary, single pass, not a decision input: raw 174,
  stated 176 of 198.
- Under the frozen rules the bundle is deprioritised as a completion remedy within this allocation; this is not a claim
  that the included corrections lack behavioural effects. The plan records an operator note: another pass is considered
  warranted (seed-1234 luck across earlier GPQA runs), to be raised with the independent assessor; it is not an
  authorization.
- 4.84 of 7.5 GPU-hours. The investigation's cumulative total: 43.25 GPU-hours. Nothing further is scheduled.
  [Screen](investigations/2026-10-glm53-completion/README.md#one-pass-gpqa-screen).

**2026-10-12, afternoon. Second GPQA pass of the bundle, seed 1235 (amendments 15-15c).**
- The independent assessment proposed one more pass of the unchanged bundle for seed sensitivity (amendment 15); the
  operator authorized one pass, no retry and no third pass (amendment 15a). The seed-1234 screen keeps its category.
- Frozen before generation (amendment 15b): runner and analysis hashes; session conditions matching the comparators'
  seed-1235 passes (protocol pass 2, first pass of a fresh server); primary comparator the stock 0.9.1 record's seed-1235
  pass (5 empty); accuracy thresholds as an absolute standard from the six stock-0.9.1 passes, with offsets from the single
  matched pass reported but not deciding (that pass is the family's best); new category names.
- Valid. 3 of 198 replies empty, all at the cap; no material or accuracy concern on the family standard (raw 168, stated
  169; 5 rows adjudicated, one hedged). Category: **MODEST IMPROVEMENT, NO CLEAR SEPARATION**. Against the record's
  seed-1235 pass: 2 rescued, 0 newly empty, raw −8, stated −9; the matched-offset readout (not deciding) shows "concern".
  Against the column-511 arm's: 5 rescued, 3 newly empty. 0.7.0's seed-1235 pass (context): 3 empty.
- EXPLORATORY (chosen after the first screen; historical controls; seeds reused): both bundle passes against the matching
  stock passes, 6 vs 10 empty of 396, reduction 1.01 points (95% 0.00 to 2.27), one-sided p = 0.11; raw −0.76 points
  (−3.79 to +2.27). Against the arm: 6 vs 9; raw +2.02 (−1.77 to +6.06).
- Per amendment 15: direction and uncertainty reported; no automatic further pass. The plan notes that any confirmation
  would prefer a contemporaneous stock control on new prespecified seeds.
- 4.541 of 7.5 GPU-hours (one session, 2 GPUs × 8,174 s). The investigation's cumulative total: 47.79 GPU-hours. Nothing
  further is scheduled. [Second pass](investigations/2026-10-glm53-completion/README.md#second-pass-seed-1235).

**2026-10-12, evening. Checkpoint B0 and targeted invariants (amendments 16-16e).**
- An independent assessment (amendment 16) froze the correctness bundle as checkpoint B0 and moved the work from
  behavioural passes to targeted invariants: a future intervention is B0 plus one verified change (B1); stock releases
  stay historical references; another behavioural evaluation needs a new verified intervention or a priced confirmation
  objective. Each item had a costed proposal and fixed its reference and verdict rules before data.
- **Two-GPU all-reduce** (authorized 16a, ceiling 0.5): CLEAN. B0's graph all-reduce transport (B12x PCIe peer-push)
  returned the exact bf16 sum on both GPUs in all 2,040,800 calls over 26,650 forwards; the epoch counter matched the
  scheduled calls; the PyNCCL reference arm exact; no hangs. 0.07 GPU-hours.
- **KV writer and query projection, B0 vs 0.7.0** (authorized 16a, ceiling 0.25): within expected rounding under the
  corrected criteria. Owned: before any GPU data, a CPU dry run of the frozen analysis on defect-free outputs found three
  errors in the frozen criteria (a projection bound half the correct half-ulp; a dequantisation tolerance inconsistent
  with the frozen error model; a byte-difference rate that omitted exact E4M3 ties). The errata were recorded and hashed
  before the run; the analysis reports both verdicts; the operator chose the corrected one (16b). As first written, the
  verdict is "beyond expected rounding", with failures equal to the dry run's defect-free counts. The runner's analysis
  step also failed to load its own result files (a version object rejected by weights-only loading); the frozen analysis
  was re-run unchanged through a one-type loader wrapper before any analysis output existed. 0.0069 GPU-hours.
- **E1** (CPU): expected numerical variation of the upstream router kernel used above 1,024 rows; selection unaffected;
  same kernel in every release; no workaround.
- **Cache isolation and high-offset addressing** (authorized 16c, ceiling 0.5): every driven kernel passed. Owned: the
  first launch was refused by the runner's own pre-flight, which counted the desktop's graphics client on GPU 0 (no
  container, 0 GPU-hours); the check was narrowed to type-G desktop clients before any data (16d). The first run with
  data had four failures: three fixture faults (an unsupported extend width; a scratch buffer compared with itself;
  never-written ring slots compared) and an obsolete upstream test. They were fixed in fixture code after inspecting the
  data, with no criterion or tolerance loosened; rerun 1 exposed two more fixture faults, and rerun 2 passed 80 of 80
  checks (16e). 0.0392 GPU-hours.
- No defect found, so no B1 and no behavioural evaluation. Not covered: block-table and slot-mapping construction under
  real scheduling, the index-K readers, extend under graph replay, other communication paths, the mixed-quant MoE path.
- 0.1161 GPU-hours in all. The investigation's cumulative total: 47.91 GPU-hours. Nothing further is scheduled.
  [Checkpoint B0](investigations/2026-10-glm53-completion/README.md#checkpoint-b0-and-targeted-invariants-amendment-16).

**2026-10-12 to 10-13, night. Live slot mappings on B0 (amendments 17-17e).**
- An independent assessment (amendment 17) asked whether real scheduling gives B0's attention and index-K kernels the
  cache slots they should get: the class of the tail-ring defect, which amendment 16's fixtures bypassed. The method was an
  observe-only guard checking each mapping as the replayed CUDA graphs consume it, against an expectation rebuilt
  independently from the scheduler's output. CPU fault-injection tests passed 28 of 28 and 28 coverage minimums were
  frozen before GPU use. Authorized at 1.0 GPU-hours with one invalid-run allowance (17a).
- **Found before GPU data** (17b, CPU): the index-K prefill writer assumes chunk starts aligned to its 4-token pools,
  which the scheduler does not guarantee. Registered as candidate D7 with a pre-registered VIOLATION.
- Owned: the first launch was refused by the runner's own pre-flight, which counted desktop graphics clients on GPU 1
  (no container, 0 GPU-hours; filter widened before data, 17c).
- Owned: attempt 2 (0.4211 GPU-hours) was invalid because of the investigation's tooling, not the engine. The runner's
  start-up abort matched the healthy field `"guard_errors": 0`, so the workload never ran. The guard expected block ids
  where the index-K reader consumes index-page ids, and the analysis had two window and timing artefacts. All were fixed
  after inspection with no rule loosened (17d). Before attempt 3, everything was re-checked with zero GPU use, including
  the client against a mock server, and the ceiling was raised to 1.45 (17e).
- **Attempt 3** (0.5722 GPU-hours): every coverage minimum met, nothing invalid, probes identical to the earlier B0
  record. VIOLATION, all of it D7: 217 events in 50 requests, identical on both GPUs; 186 pools never written and 31
  filled from the previous request's tokens. Every one is at a prefill chunk start that is not a multiple of 4. Every
  other mapping was clean. Effect on selection or completion not measured.
- 0.9933 GPU-hours in all; cumulative 48.90. Next under the plan: minimise D7, verify its smallest correction, then
  B1 = B0 + that correction; not yet authorized.
  [Live slot mappings](investigations/2026-10-glm53-completion/README.md#live-slot-mappings-on-b0-amendment-17).

**2026-10-13. D7: exposure, correction and verification (amendments 18-18c).**
- An independent review (amendment 18) asked for the mechanism per release, a comparison with upstream, exact GPQA
  exposure with stated uncertainty, the smallest correction, and verification in serving before any behavioural spend.
- **Releases and upstream** (CPU): the writer is byte-identical in 0.7.0, 0.8.0 and 0.9.1, and every served
  configuration reaches unaligned starts, with patterns that differ by draft slots, block size and DCP. Upstream vLLM main
  keeps the assumption; SGLang repairs it writer-side.
- **Exposure** (CPU):
  - The earlier simulation's 73 predicted against 217 measured events is explained by the scheduler's per-request
    draft-slot reservation, which the simulation lacked; with it, the 217 are reproduced exactly.
  - In the GPQA records there is no exposure in the 0.9.1 family, and doc 127 is exposed in 0.7.0 passes 2-3, where it
    finished. This is a reconstruction.
- **Owned:**
  - my first proposal compared prompt lengths in characters and called doc 127 "always exposed"; it is always split,
    but on 0.9.1 at a pool boundary;
  - I stated that no server log existed for the stock 0.9.1 record; both exist;
  - amendment 17e stated distinct seeds for all 64 sampled requests; 8 seeds are reused between two phases (no effect on
    the mapping checks).
  All three are corrected in amendment 18a or 18c.
- **Correction and verification:** a writer-side correction passed 28/28 CPU tests and the D1-D4 regressions. The guard
  was amended with an independent byte oracle and a positive control, and all required fault classes were detected on
  CPU. Authorized up to 2.0 GPU-hours with one retry (18b); one session, **CLEAN**: 219 straddling pools byte-verified,
  0 mismatches over 1.53 million control pools, every other mapping clean, 0.6244 GPU-hours. An appended phase measured
  doc 127's split at 2,045 − 7·D for D = 0-7.
- **Outcome:** B1 = B0 + D7 is the correctness checkpoint. D7's behavioural investigation is closed for now; there is no
  GPQA pass and completion impact is untested. Cumulative 49.52 GPU-hours.
  [D7 correction](investigations/2026-10-glm53-completion/README.md#d7-correction-and-verification-amendment-18).

**2026-10-13, afternoon. Checkpoint status, and the B1 GPQA pass (amendments 19-19b), results pending.**
- **Where the corrections stand.** Each correction in the package is validated and measured on its own terms:
  - D1 column-511 selection: verified offline, in place and in shipped graphs;
  - D2 tail ring: 0 null-block or foreign writes;
  - D3 router: fp32-output logits within tolerance of fp64, as served;
  - D4 SwiGLU limit: reaches the served kernels; whether it engages at 10 is not measured;
  - D7 unaligned chunk starts: verified in serving with byte-level checks.
- **What GPQA shows so far** (B0 = D1-D4, two passes): 3 and 3 of 198 answers empty against 5 and 5 for stock 0.9.1 at
  the same seeds. That is a modest, consistent reduction without clear separation, and it does not reach stock 0.7.0's
  record. The 0.7.0-vs-0.9.1 completion gap is not explained.
- **B1 = B0 + D7** is the correctness checkpoint. It is built from B0's image and differs only in the D7 file; D1-D4
  regressions pass on it.
- **One GPQA pass of B1** (seed 1236; primary reference: stock 0.9.1 at seed 1236, 6 empty) is running under rules
  frozen before generation, with a 7.5 GPU-hour watchdog and no retry. It characterizes the checkpoint; it does not
  attribute an effect to D7, and it stays a separate record from B0's passes.
- **Owned:** the seed-1235 table published stock 0.7.0's stated count as 178; the published per-question rows give 177.
  It is corrected in the README, in a context column, with no category change.
- **Upstream:** a pull request for the engine (D1-D4 and D7, each with a minimal reproduction) and a vLLM report on D7
  are drafted and held until the investigation concludes.
- Cumulative 49.52 GPU-hours before this pass.

**2026-10-13, afternoon and evening. B1 GPQA result, the review step, and the long-context tail (amendments 19c-19f).**
- **B1 result** (seed 1236): valid; 9 of 198 empty, all at the cap, against 6 for stock 0.9.1; raw and stated accuracy
  unchanged. Under the reading frozen before generation it is unfavourable, which triggers review; the verified
  corrections stand. 6.33 GPU-hours.
- **Post-hoc token reading:** the extra tokens are the extra capped replies, and finished replies are no longer. The four
  newly empty documents are short single-chunk prompts on which, per the exposure reconstruction, the D7 path is not
  expected to run.
- **Owned:** the identical-reply counts first quoted between builds at the same seed (2-5 of 198) were all pairs of
  empty replies; no non-empty reply was ever identical. The frozen B0 analysis counts non-empty identity only.
- **Review step:** a B0 pass at seed 1236 (near-replicate of B1) is running under readings frozen before generation.
- **Long-context tail** (exploratory, zero GPU, chosen after the B1 result): replies reach 100k tokens at similar rates,
  but fewer of 0.9.1's long replies finished before the cap than 0.7.0's (stock 0.9.1 10 of 25, 0.7.0 10 of 14, at
  ≥ 100k). This is a hypothesis to examine, not a localisation. The published summary first said it "located" the gap;
  see the next entry.
- **Next:** a proposal for a long-context termination probe, discovery first, for independent review. Cumulative 55.85
  GPU-hours.
  [B1 pass and long-context tail](investigations/2026-10-glm53-completion/README.md#b1-checkpoint-pass-and-the-long-context-tail-amendment-19).

**2026-10-13, evening. Where things stand, a correction of emphasis, and the next plan (amendment 20).**
- **Established** (each with its own reproduction and verification):
  - corrections for D1 (column-511 selection), D2 (tail ring without positions), D3 (router logits in bf16 on SM12x),
    D4 (routed SwiGLU limit not passed, reachability only) and D7 (index-K prefill writer at unaligned chunk starts,
    verified in serving);
  - D7 is also present in upstream vLLM main;
  - the targeted invariants of amendments 16-17 (all-reduce, KV writer, cache addressing, live slot mappings) were
    clean apart from D7.
- **Not established:**
  - what produces the 0.7.0-vs-0.9.1 completion gap (4 vs 16 empty of 594, one three-pass record per engine);
  - whether any correction changes completion;
  - whether the corrected builds are more accurate (accuracy is similar).
  
  The D1-D4 bundle's lower empty count at two seeds (3 and 3 vs 5 and 5) did not persist once B1's seed-1236 pass (9 vs
  6) is included; that pass reads unfavourable under its frozen rule and is under review.
- **In progress:** a B0 pass at seed 1236 (the review step; readings frozen before generation) and zero-GPU discovery
  under amendment 20: a tail-accounting audit with stock releases reported separately, an inventory of existing long
  histories, termination-token semantics, and a focused source review of context-dependent paths.
- **Hypothesis, not a result:** fewer long replies finishing on 0.9.1 than on 0.7.0 (amendment 19f). Finishing less
  often is not the same as looping more often. A capped reply may be a loop or an exhaustion still reasoning, and the
  GPQA records cannot tell which. Where text was kept (the completion pilot), capped replies were not loops.
- **Tail audit (amendment 20c), stock releases only:**
  - 0.7.0 finished 10 of 14 replies still generating at 100k; 0.9.1 finished 10 of 25. The difference is +0.31
    [0.03, 0.55], question-clustered.
  - Reaching 100k is equally common.
  - It reproduces the earlier counts. **Owned:** a "6 of about 100 past 150k" figure summed at-risk counts across bands;
    the unique count is 6 of 36.
  - No specific context range is supported.

**2026-10-13, night. Discovery for the termination probe, and the first-stage proposal (amendment 20d).**
- **Where things stand** (unchanged from the entry above):
  - the corrections are established;
  - the completion gap's cause is not;
  - fewer long replies finishing on stock 0.9.1 than on 0.7.0 is a hypothesis (stock-only audit: 10/25 against 10/14 at
    ≥ 100k);
  - finishing less is not looping more;
  - a B0 pass at seed 1236 (the review of B1's result) is still running.
- **Termination semantics:** stopping rules are identical across all four builds, so any difference is in the model's
  output probabilities, not the stop logic ([`fixture-inventory/TERMINATION.md`](investigations/2026-10-glm53-completion/fixture-inventory/TERMINATION.md)).
- **Existing histories:** exact token ids exist only below 5k tokens. Long histories are text, usable as shared
  re-tokenized fixtures, and classed as natural close, loop or exhaustion
  ([`fixture-inventory/INVENTORY.md`](investigations/2026-10-glm53-completion/fixture-inventory/INVENTORY.md)).
- **Source review of context-length-dependent paths** ([`source-review-lc/SOURCE-REVIEW.md`](investigations/2026-10-glm53-completion/source-review-lc/SOURCE-REVIEW.md)):
  - the leading candidate is the known D2: decode-written index-K pools are wrong on 0.7.0, often wrong on stock 0.9.1
    depending on concurrency, and correct on B0/B1;
  - its rank order matches the finishing rates, which is an observation, not evidence;
  - D6, the 0.7.0 tail-ring wrap on rejected drafts at ×5, is the second known decode-only index-K corruption;
  - kernel regime changes past about 120k and KDA block migration are untested invariants;
  - prefill scoring cannot see D2 or D6, so prefill-only probes would miss them.
- **Proposal, awaiting independent review** ([`probe-proposal/PROPOSAL.md`](investigations/2026-10-glm53-completion/probe-proposal/PROPOSAL.md)):
  a force-decoded window probe at 64-128k context, with 0.7.0 against B0 at matched draft depth plus D2 and D6
  diagnostic arms (diagnostic only, never a remedy), staged and costed. Draft depth and the reject/restore path were
  added as factors. No GPU use is authorized yet, and no probability difference will be read as explaining completion
  rates.

**2026-10-13 to 10-14. Review result, selection kernels, and the first hook validation (amendments 21-21f).**
- **Where things stand:**
  - corrections established;
  - the completion gap's cause not established;
  - fewer long replies finishing on stock 0.9.1 than on 0.7.0 is a hypothesis;
  - finishing less is not looping more.
- **B0 at seed 1236** (the review of B1's 9): 5 of 198 empty. Frozen reading (b): B1's excess is D7 or run-to-run
  variation, and no D7 effect is inferred. 0 identical non-empty replies between B0 and B1, and 8 of 14 capped questions
  changed status at the same seed. Single passes cannot resolve differences this size. 5.866 GPU-hours.
- **An independent assessment (amendment 21)** revised the next stage:
  - validate instruments first: K, then the hook;
  - use B1 ×3 as the corrected baseline;
  - test D2 by injecting the old tail mapping only during a forced decode window, never by removing D2 globally while
    keeping D7.
- **K:**
  - B1's selection is correct at 1-330k tokens (34 of 34).
  - Stock 0.7.0 selects correct sets, but its merge drops an arbitrary pool at column 511, differently per launch and per
    GPU.
  - **Owned:** the first run's plan gave 0.7.0 a negative-length row, which aborted most of its cells; the plan was fixed
    after inspection and the rerun completed.
  - 0.2825 GPU-hours.
- **Hook validation attempt 1: INVALID.**
  - The forcing logic is validated.
  - Two hook bookkeeping bugs, analysis join errors, and thresholds without a measured basis account for every failure.
  - No new B1 defect was found.
  - Deep-layer pooled index-K keys vary grossly in about 0.5% of pools even between two prefills of the same tokens.
  - 1.2117 GPU-hours.
- **Next:** hook fixes and revised criteria go to independent review before the last authorized attempt. Cumulative
  63.22 GPU-hours.
  [Instrument validation](investigations/2026-10-glm53-completion/README.md#instrument-validation-k-and-the-force-history-hook-amendment-21).

**2026-10-14. Hook validation attempt 2, and the first-contrast proposal (amendments 21g-21k).**
- **Where things stand:**
  - corrections established;
  - the completion gap's cause not established;
  - fewer long replies finishing on stock 0.9.1 than on 0.7.0 is a hypothesis;
  - finishing less is not looping more.
- **Revised criteria:** an independent review accepted them for the last validation attempt, all post-inspection:
  - an index-K content gate against matched in-session nulls, plus exact structural checks;
  - KDA commit bookkeeping kept exact, separate from a numerical identification screen;
  - attempt 1's alignment evidence carried over, with explicit provenance.
  Originals are reported beside the revised readings.
- **Attempt 2: VALID** under those criteria (0.795 GPU-hours).
  - **Validated:** the forced-decode instrument (forcing, alignment, attribution, a clean ledger, complete exposure
    records) and the decode-window D2 injection (59.7% corrupted pools against a 1.16% matched limit).
  - **Not established:** full-history equivalence, the absence of every deep-layer numerical issue, or any link to GPQA
    completion.
  - **Follow-ups:** found no defect.
- **Next:** a first contrast of B1 against B1 with the decode-window D2 injection, with a long-prompt control. It is
  proposed and costed (about 2.8 GPU-hours, ceiling 4.3) and awaits independent review and authorization. Cumulative
  64.01 GPU-hours.

**2026-10-15. First long-context contrast, attempt 1: stopped, no reading (amendments 22-22c).**
- **Amendment 22,** from independent review, set the design: option B, in-session validation before any injected batch,
  a stronger prefill null, a convolution-state boundary control, corrected readings, and no permutation p-value.
- **Session:** a structural stop after C1, so no injected run, no gate and no reading; 1.0861 GPU-hours. The
  single-attempt authorization is spent.
- **Owned:**
  - The plan's cache expectation was one 4,352-token block too high. This stack reuses (⌊(P−1)/4352⌋ − 1) blocks on an
    exact repeat, and our own earlier records already showed that rule. The CPU checks tested the plan against itself.
  - The plan also assumed a prefix-cache capacity the shared KDA state blocks do not leave (reach about 0.66 M tokens),
    so one warm prefix was evicted.
  - A page criterion calibrated at short context fails between two identical long prefills.
- **No instrument or engine fault:** C1's forced windows look valid post-inspection.
- **Observation:** deep-layer pooled index-K is more fragile at long context.
- **Next:** fixes and a new attempt go to independent review and new authorization. Cumulative 65.10 GPU-hours.
- **Correction (amendment 22e):** a CPU replay of the image's own cache code shows M3 was evicted by C1's own earlier
  admissions (HTTP admission race), not because the warm round exceeded the cache reach as first written. The retry
  package (variant B, about 3.0 GPU-hours, ceiling 4.18, single attempt) awaits review.
  [First contrast](investigations/2026-10-glm53-completion/README.md#instrument-validation-k-and-the-force-history-hook-amendment-21).

**2026-10-16. First long-context contrast, retry: complete session, no reading (amendments 22f-22j).**
- **Review and authorization (22f-22i):** independent review corrected the sentinel (SSM and convolution state judged
  separately), F6 page reporting, LP same-depth pairs and cache wording before any data. The operator authorized variant
  B with a 6.2 GPU-hour ceiling and a batch guard after stress replays showed 4.18 could not cover a complete session.
- **Session:** every planned phase ran (224 requests), 2.9317 GPU-hours, no guard stop. GATE-1R, run validity,
  injection placement, cache safety and balance, and the sentinel's coverage, block identity and bytes all pass.
- **Result: NO READING.** Three structural gates failed: H5f-fc (ledger against bytes) and the sentinel's SSM and
  convolution boundary-state rows. The session supports no statement about D2 and reasoning close propensity.
- **Descriptive only:** Δ = 0.106 nats against a beyond-null threshold of 0.200 (screen not met), with the three injected
  runs on one side of the three clean runs; selection divergence within the within-arm null.
- **Post-hoc diagnostics (after the data; they do not change the verdict):** the H5f failures are one high-variance
  fixture's pools, gross in the uninjected runs too, inside a small clean-labelled subset. The sentinel failures sit
  where the two pre-injection baselines happened to be unusually close; the same 2× rule applied to a pre-injection
  baseline fires as often. Neither shows contamination of the cached prefix; both point to gate limits built on too
  little baseline data.
- **Next:** whether to re-run with gates corrected before new data, or close the contrast line, goes to independent
  review. Cumulative 68.03 GPU-hours.
  [First contrast](investigations/2026-10-glm53-completion/README.md#instrument-validation-k-and-the-force-history-hook-amendment-21).

**2026-10-16, later. Contrast line closed as unresolved; mechanism ranking; replication route (amendments 22k-22l).**
- **Disposition (22k, independent review):** no repeat of the full contrast. The line closes as unresolved, not as
  evidence against whole-history D2 effects.
- **Bounded CPU review** of the failed gates:
  - On the 176 units H5f judged, the injected runs' gross count (10 of 286 unit-runs) lies within the clean arm's and
    the nulls' (1-6 of 176). The gross units are 4 fragile M4 units, gross without injection too.
  - Judged symmetrically, the post-injection sentinel state is the odd one out no more often than the pre-injection
    ones.
  - This bears on calibration only; it shows neither that the ledger is right nor that contamination was absent.
  - A prospective instrument design is documented, not implemented.
- **Zero-GPU mechanism ranking (22l):** 0.7.0's ring-4 / depth-5 rejected-draft defect is masked by D2 in served
  0.7.0, and its atomic-slot merge sits inside a class B1 already corrects. Neither earns a GPU experiment.
- **New open question:** D2 exposure at one request (CPU harness) splits steps 4-6 the same way as their measured
  disagreement, so step 6's layout signal may not be separate from D2. Untested.
- **Next:** the operator chose the stock 0.7.0 replication route. Its draft (expected about 23 GPU-hours) is re-frozen
  and goes to review and authorization before any data. No GPU-hours added (cumulative 68.03).
  [Investigation](investigations/2026-10-glm53-completion/README.md#open-questions).

**2026-10-16. Stock 0.7.0 replication frozen before data (amendments 22m-22n).**
- **What it tests:** whether stock 0.7.0's low empty-reply count holds at fresh request seeds (70701 / 70702 / 70703).
  Task, prompts and answer order are gated identical to both historical records.
- **Why fresh seeds:** same-seed passes measured as coupled, with reply-length correlation 0.66-0.91 against 0.37-0.48
  for different seeds.
- **Reporting:** outcome categories as drafted, with clarified meanings. Both comparisons are always reported, and the
  long-reply finishing rate is a descriptive secondary.
- **Budget:** 36.5 GPU-hour watchdog; incomplete runs reported, never classified.
- **Status:** independent review done; awaiting GPU authorization. No GPU-hours added (cumulative 68.03).
  [Replication](investigations/2026-10-glm53-completion/rep070/refreeze/PROPOSAL-v2.md).

**2026-10-17. Stock 0.7.0 replication: valid, inconclusive (O3); the historical-gap line closes (amendments 22p-22q).**
- **Run:** stock 0.7.0, three passes of GPQA Diamond at fresh request seeds, configuration-identical to its record.
  Valid on every frozen check; 23.93 GPU-hours.
- **Result:** **10 of 594 replies empty**, against 4 in its own record and 16 in the 0.9.1 record. It separates from
  neither (one-sided p 0.122 against 0.9.1, 0.108 against old 0.7.0), so the reading is O3. The original 4-versus-16
  contrast looks partly like a favourable draw; whether a smaller real gap remains is not resolved.
- **Accuracy:** stated 87.9%, the same as the 0.9.1 record's; the old 0.7.0 record had 89.4%.
- **Descriptive:** of replies still generating at 100k tokens, 19 of 29 finished (0.66), against 0.40 (0.9.1) and
  0.71 (old 0.7.0). This is not a causal effect and too few events to adjudicate a mechanism.
- **Next:** the 4-versus-16 line closes. Long-reply completion becomes its own question, starting with zero-GPU
  discovery. Cumulative 91.95 GPU-hours.
  [Result](investigations/2026-10-glm53-completion/rep070/RESULT.md).

**2026-10-21. Long-reply completion: discovery, a fixed-prefix continuation (B+) and a text-logged 0.9.1 pass (amendments 22r-22ab).**
- **Discovery** (zero GPU): stock 0.7.0's capped long replies mostly end in exact short-period loops that begin well
  after 100k tokens; finished ones stay novel. On matched questions, 0.7.0 finished more long replies (23 / 36 against
  18 / 58).
- **B+** (13.06 GPU-hours): 12 token prefixes generated by 0.7.0 at 100k, reconstructed by each engine's prefill;
  4 seeds; +64k on three engines.
  - The frozen analysis was INVALID on an over-strict KV-pool gate. Both verdicts are kept.
  - Under a disclosed, outcome-blind amendment to the KV-capacity gate, both stock engines stopped within the
    continuation budget on 20/48 requests. The estimated difference was 0.000, with a prefix-bootstrap 95% interval of
    −0.083 to +0.083. The overall reading remained R-I.
  - B1 stopped on 25/48. η² 0.67 describes between-prefix variation in the observed sample.
- **Option D** (5.88 GPU-hours, descriptive): one stock 0.9.1 pass with full text. 4 / 10 long replies finished;
  capped replies are the same loops as 0.7.0's, with different early signatures.
- **So far:** this points to the difference arising before 100k rather than in continuation. That is not established.
- **Next:** option E, 0.9.1-built prefixes on both engines. Cumulative 110.90 GPU-hours.
  [Long-reply](investigations/2026-10-glm53-completion/long-reply/README.md).

**2026-10-22. Long-reply completion: a second text-logged 0.9.1 pass; the prefix cross frozen (amendments 22ac-22ah).**
- **Option E as first frozen** (8 0.9.1-built prefixes) had power of only 0.37-0.42 at the full gap. The operator chose
  to add prefixes first.
- **D2** (7.16 GPU-hours, descriptive): a second stock 0.9.1 pass at a fresh seed. 8 of 198 empty; 4 / 12 replies past
  100k finished. The digit-heavy early signature of capped replies appears here as well. It adds 5 prefixes on new
  questions.
- **E re-frozen** on 13 prefixes: power 0.75-0.88 at the full gap, cost central 10.2 GPU-hours. The KV pool is now
  reported with a floor, not gated. Not yet run.
- Cumulative 118.07 GPU-hours.
  [Long-reply](investigations/2026-10-glm53-completion/long-reply/README.md).

**2026-10-22. Long-reply completion: the prefix-source cross (E), valid, inconclusive (amendments 22ai-22ak).**
- **Review:** an independent review (22ai) accepted E's design and fixed its interpretation before data: effects of
  the engine bundle after prefill reconstruction of token prefixes, not effects from identical internal states.
- **Run:** 13 token prefixes generated by 0.9.1, +64k on stock 0.7.0 and stock 0.9.1; 10.24 GPU-hours.
- **Reading: R-I.** Stop-token endpoint 0.423 against 0.288. The difference is +0.135 (95% +0.038 to +0.250;
  sign-flip p 0.0625, the floor of the exact test with 5 discordant prefixes), below the prespecified 0.15 threshold.
- **Descriptive:** the two engines stopped alike on 0.7.0-generated prefixes (B+) but differed on 0.9.1-generated ones.
  This is consistent with 0.9.1 histories reaching states its engine bundle then handles worse; it is not established.
- **Answer audit:** every stopped continuation gave a final letter; 12 / 22 and 10 / 15 were correct.
- Cumulative 128.31 GPU-hours.
  [Long-reply](investigations/2026-10-glm53-completion/long-reply/README.md).

**2026-10-22. Long-reply completion closed as unresolved, with positive findings (amendment 22al).**
- **Decision:** on independent review, the allocation closes. E is kept as suggestive evidence requiring independent
  confirmation; its R-I reading is unchanged.
- **Summary:** in B+, the stock engines had identical aggregate stopping rates on selected 0.7.0-generated prefixes.
  In E, 0.7.0 stopped more often on selected 0.9.1-generated prefixes (+13.5 points; five prefix-level differences
  favouring 0.7.0, none favouring 0.9.1). Neither met its frozen criterion. A dependence on prefix source is
  suggested, not established, and its cause is not located.
- **Established:**
  - the same terminal-loop forms on both stock engines;
  - observable differences that precede terminal repetition;
  - prefix identity strongly predicting outcomes in these samples;
  - prefill-reconstructed long histories finishing on every tested engine.
- **Correctness:** extra stops were not comparably more correct answers. Over all attempts in E, 23.1% were correct on
  0.7.0 and 19.2% on 0.9.1, against a stopping difference of +13.5 points.
- **Correction:** this supersedes the earlier pooled B0 + B1 figure (0.12 on 17 replies). B1 stopped on 25 / 48 B+
  continuations.
- **Any further interaction study** needs a new proposal with a fresh cohort and a decision that depends on it.
- No GPU-hours added (cumulative 128.31). [Close-out](investigations/2026-10-glm53-completion/long-reply/CLOSEOUT.md).

**2026-10-22. Reconnaissance after close: the stock contrast pooled; dispatch, a draft-slot check (amendment 22am).**
- **CPU only;** no frozen reading changes.
- **Pooled stock passes (post hoc):** 0.7.0 left 14 of 1,188 replies empty, 0.9.1 30 of 990 (−1.85 points, −3.23 to
  −0.66). Every 0.7.0 pass and session sits below every 0.9.1 one. Fresh seeds alone: −1.85 (−4.12 to +0.17).
- **Where it sits:** a clear difference in finishing once past 100k tokens (29 / 43 against 18 / 47); no clear
  difference in reaching 100k (wording corrected in 22an).
- **Dispatch:** long replies run mostly at 8 requests in flight. There the engines' decode batches (48 against 32 rows)
  take different routed-expert plans; their top-k routes differ only at 3-4 requests in flight. Per-step dispatch was never logged on a stock image.
- **Draft slot mapping (stock 0.9.1, CPU):** no violation in the exercised states; planted faults detected.
- **Errata (owned):** a sign in the investigation README; a loop-onset count (6, not 8) in the long-reply discovery
  report.
- No GPU-hours added (cumulative 128.31). [Reconnaissance](investigations/2026-10-glm53-completion/recon-2026-10-22/README.md).

**2026-10-22. Review of the reconnaissance: accepted with corrections (amendment 22an).**
- **Corrections (owned):**
  - exact permutation p is reported one- and two-sided (session level 0.0143 / 0.0286) with its assumption; the
    within-question test is Monte Carlo;
  - "no clear difference" in reaching 100k, not "no difference";
  - a throughput figure was a whole-request average, not a post-100k measurement;
  - the routed-expert launch geometry statement was wrong and is corrected.
- **NO READING stays NO READING:** the first contrast's D2 window figures are not a validated negative.
- **Added:**
  - a deterministic positive control for the draft-slot check (12 / 12);
  - from the code: passing `positions` also changes the per-token causal-length source for speculative verify rows,
    so the tail-ring correction is not a single isolated change. Not measured.
- No GPU-hours added (cumulative 128.31).

**2026-10-23. A live causal-length check, frozen before data (amendment 22ao).**
- **Why first:** on review, the causal-length source for speculative verify rows (22an, read in code) gets first
  measurement priority. It is a concrete path from scheduling metadata to attention. It is not shown to change
  outputs, and it is not an explanation of the completion difference.
- **The design**, pre-registered: stock 0.9.1 against stock 0.9.1 + port 01. It measures:
  - the consumed length against p + 1 (signed);
  - whether any consumed entry falls outside the row's causal range;
  - with required coverage of rejections, recycled rows and padded shapes, in-session checks of the checker itself,
    and three outcome classes (INCOMPLETE, POSITIVE FINDING, CLEAN WITHIN COVERAGE).
- **Not run:** ceiling 2.0 GPU-hours, pending authorization. [Package](investigations/2026-10-glm53-completion/livecheck-22ao/PROPOSAL.md).

**2026-10-23. The live causal-length check, run (amendment 22ap).**
- **Stock 0.9.1:** speculative verify rows consumed per-token causal lengths above p + 1 on 672,155 of 1,506,571
  checked rows, never below; the excess equalled the previous step's rejected-draft count. No consumed entry fell
  beyond the causal range or belonged to another request (0 of 48,782 rows; nominal 3/N bound, see 22aq): the extra
  in-count columns were empty.
- **With port 01 (PR #7):** no excess on any row; every selection check clean within coverage.
- **Not measured:** whether the empty columns change the attention output. Not shown to relate to the completion
  difference.
- **Disclosed amendments:** the port-01 arm's pre-flight re-samples utilization after a transient refusal (written
  before the stock arm's results were read); its first attempt stopped before traffic on an instrument race, which was
  fixed and the arm retried with the operator's authorization after the stock arm's results were read.
- GPU-hours 1.51 (cumulative 129.82). [Results](investigations/2026-10-glm53-completion/livecheck-22ao/run-1/result.md).

**2026-10-23. Level 3: the inflated count leaves the attention output unchanged in the tested configuration (amendment 22aq).**
- **A subsequent investigation**, run after the live check was read; not pre-registered.
- **Compaction** gave identical consumed inputs in the tested cases (12,282 CPU cases).
- **The stock decode and extend kernels** gave bit-identical outputs for the exact and inflated counts (64 / 64 rows,
  r = 1-3), and a planted control was detected.
- **Ruled down** as a computational contributor in the tested 0.9.1 configuration and under the verified
  masking/compaction conditions. Not extended to 0.7.0, untested configurations or arbitrary cache contents.
- **Correction (owned):** the 22ap rule-of-three bounds assume independent rows and are nominal.
- GPU-hours 0.0022 (cumulative 129.82). [Level 3](investigations/2026-10-glm53-completion/livecheck-22ao/level3/README.md).

**2026-10-23. The cache slot level 3 relies on stays zero on the served path (amendment 22ar).**
- **A bounded audit and a static writer census** (760 call sites; 0 that can write non-finite or arbitrary bytes)
  find that the slot-0 bytes the attention kernels read for empty entries stay all zero on the served 0.9.1 path.
  That is the case level 3 measured bit-equal.
- **Two allocator assumptions remain inference.**
- **Erratum (owned):** level 3 cited the V1 runner for the zero-filled cache; the served V2 runner also zero-fills.
- No GPU-hours added (cumulative 129.82). [Level 3](investigations/2026-10-glm53-completion/livecheck-22ao/level3/README.md).

**2026-10-23. The two routed-expert plans differ reproducibly from 16 rows up (amendment 22as).**
- **Setup:** each engine's served EXL3 code ran identical synthetic rows through the decode plan (0.9.1 at 8 in
  flight) and the prefill plan (0.7.0 at 6-8 in flight).
- **Results:**
  - bit-equal at 1 and 8 rows (same tile);
  - from 16 rows, nearly every row differs (bf16 relative L2 about 8.4e-4), in both engines and all three layers
    tested;
  - each plan is deterministic, and the two B12x versions agree bit-for-bit.
- **Not measured:** real inputs, the cross-GPU partition, and any effect on generation. Not shown to explain the
  completion difference.
- GPU-hours 0.1458 (cumulative 129.97). [Result](investigations/2026-10-glm53-completion/moe-plan-p3/RESULT.md).

**2026-10-23. On real decode rows the MoE plan difference reproduces, and its source is the tile (amendment 22at).**
- **Captured serving inputs, replayed offline** after reproducing each served output bit-for-bit:
  - the decode and prefill plans agree at 4-8 rows and differ on 89-100 % of rows from 16 rows up (relative L2 about
    6e-4 to 8.4e-4);
  - forcing the decode plan onto the prefill tile makes them bit-identical.
- **Source:** tile choice is the cause on the tested inputs. Its size is explained by source analysis and emulation
  (fp32 summation order amplified by shared fp16 roundings). No accuracy advantage is established (wording corrected in
  22au).
- **Attention:** the stock inflated count gives bit-identical outputs to the exact count on captured dense and clamped
  rows; the boundary case is incomplete.
- **A failed first capture is kept and explained:** a poller started before CUDA-graph capture.
- **Not measured:** any effect on generation.
- GPU-hours 0.8031 (cumulative 130.77). [Result](investigations/2026-10-glm53-completion/captured-p3b/RESULT.md).

- **Correction of emphasis (owned):** the summaries published earlier today said the tail analysis "located" or
  "localised" the gap and stated pooled figures, including a B0+B1 rate built on 17 replies, without enough
  qualification. They also said the D7 path "cannot run" where the basis is a reconstruction. These are reworded in the
  README, FINDINGS (item 4m, now graded exploratory and hypothesis-generating) and this ledger.
- **Next (amendment 20, adopted from an independent assessment):**
  - finish the B0 review under its frozen rules;
  - complete the zero-GPU discovery;
  - then propose a bounded fixed-history probe: stock 0.7.0 against B1 on identical token prefixes of about 64k-128k
    plus a short control, scoring reasoning-close and end-of-turn probabilities, sampler eligibility and argmax
    agreement, with a decode-path control and repeatability established first.
  
  No claim that a probability difference would explain completion rates. Each GPU stage is proposed and authorized
  separately.

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
| GPU-hour ceiling for the bundle gate | 1.5 | 2.0 | 1.5 was set before the gate design existed; its estimate for the two required sessions was 1.25-1.69 (amendment 14b, before any gate data) |
| Early stops in the bundle gate's GPQA-shaped phase | three | four (docs 16, 19, 21, 22) | counted from the client records before launch (amendment 14d) |
| Prefix-cache coverage of the 16,240-token probe prompt between rounds | identical | identical only in s1; 0 vs 8,704 tokens cached in s3 and in the release control | misstated in the coordinator's first description of the gate data |

</details>
