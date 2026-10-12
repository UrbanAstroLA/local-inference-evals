<!-- Frozen 2026-10-10. The plan text below is unchanged from the version approved on 2026-10-10. Changes go in "Dated amendments" at the end. -->

# Proposal: explain the GLM-5.3-Flash 0.7.0 completion advantage

**Prepared:** 2026-10-10  
**Audience:** the agent maintaining `UrbanAstroLA/local-inference-evals` and the tpurtell engine investigations  
**Status:** proposed investigation, not a report of new GPU experiments

## Objective and success criteria

Determine why the published 0.7.0 configuration returns more final answers than 0.9.1 with the same preferred EXL3 K3.25, 3:5:8 projection allocation. Identify the smallest defensible change that improves completion while preserving target-model fidelity. Treat quantization choice as fixed for this investigation.

A successful outcome is either:

1. A reproducible engine discrepancy, a localized cause, a corrective change, and evidence that the change improves completion on fresh prompt/seed pairs; or
2. A bounded explanation that the advantage depends on an old incorrect computation, speculative execution, numerical topology, or sampled trajectories, with the remaining uncertainty stated precisely.

Closing the accuracy gap is a hypothesis to test, not a promised consequence of improving completion. A completed answer can still be wrong.

## Evidence motivating this plan

The three-pass K3.25 records contain 594 question-runs per configuration [S1–S2]. A direct join on `(doc_id, pass)`, with prompt and seed identity verified, gives:

| Outcome | 0.7.0, DFlash2 ×5 | 0.9.1, DFlash2 ×3 |
| --- | ---: | ---: |
| Raw correct, all runs | 524/594 | 516/594 |
| Empty final answers | 4/594 | 16/594 |
| Raw correct in the 577 pairs where both answered | 516/577 | 516/577 |
| Audited correct in those same 577 pairs | 523/577 | 522/577 |

The entire eight-answer raw advantage is on pairs where 0.7.0 is correct and 0.9.1 is empty. This decomposition localizes the observed score gap to completion, but conditioning on completion is not a causal experiment. The published clustered test supports a completion difference (`p=0.009`); the overall raw accuracy difference remains uncertain (`p=0.41`).

The comparison changes draft depth, expert partitioning, attention topology, backend kernels, cache allocation, and vision together. The newer defaults were selected on uniform K4 and short coding outputs, not this mixed quantization and long GPQA reasoning [S3–S5].

Do not reuse withdrawn fixed-seed failure rates as population estimates. Fixed failing seeds remain valid debugging fixtures. The current small component screen excludes only very large effects and couples some switches [S3].

## Operating rules

- Keep the exact K3.25 weight payload, tokenizer, rendered prompts, answer ordering, chat template, and sampling settings fixed. Verify hashes rather than model names alone.
- Preserve all known correctness fixes in proposed candidate releases. Stock 0.7.0 is a diagnostic anchor, not a recommendation to restore corrupted caches.
- Separate debugging, exploratory screening, and independent confirmation. Do not use a discovered fixture as untouched confirmation data.
- Change one independently controllable component at a time. Report coupled switches as a bundle; do not assign its result to one component.
- Compare frozen inputs and committed states before spending compute on final-answer rates.
- Count occupied GPU-hours, including model loading, compilation, warmup, invalid runs, and retries. For a two-GPU allocation, `GPU-hours = 2 × allocated wall-hours`; do not sum concurrent request durations.
- Record engineering hours separately. A new hook that takes days to implement is not automatically economical because its runtime is short.
- Do not launch a new full GPQA pass until a specific candidate or unresolved branch justifies it.
- Preserve GPQA's existing publication practice: publish ids, hashes, metrics, and code; retain restricted prompt text, token sequences, and intermediate fixtures locally [S6].

## Initial resource envelope

Use **12 occupied GPU-hours** as an initial discovery ceiling: approximately six hours with both GPUs allocated. These are proposed ceilings, not runtime forecasts or guarantees of sufficient statistical power.

| Work | Initial ceiling | Required output |
| --- | ---: | --- |
| Existing-data audit | CPU only | Verified outcome decomposition and effective configuration diff |
| Fixture capture and isolated replay | 4 GPU-hours | Reusable failing or unstable fixture, or bounded null findings |
| Controlled model replay and localization | 4 GPU-hours | First material discrepancy and nominated component |
| Targeted behavioral pilot | 4 GPU-hours | Candidate ranking and cost/power estimate for confirmation |

Reallocate unspent time toward the leading hypothesis; do not mechanically run every arm below. Stop at the ceiling and report evidence, remaining branches, and the cheapest next experiment. Confirmation has a separate, precomputed budget.

## Phase 0 — extract information already paid for

1. Pin the evidence snapshot. This proposal inspected evaluation commit `5901e6b5591fd882c8da71e627b9d311d180d667` and recipe commit `0e4a5820cf1fde29a5fb6afc36a53e08760a9cbb`. Record any newer checkout and describe changes before comparing numbers.
2. From the evaluation repository root, run the existing checks:

   ```bash
   python3 tools/verify.py
   python3 tools/analyze.py gpqa-records
   ```

3. Add a small analysis joining the two K3.25 record files on `(doc_id, pass)`. Verify matching seed and prompt hashes. Reproduce the 577-pair decomposition above and list the eight score-driving pairs by id and seed.
4. Inventory locally retained original responses and request logs. Published hashes alone cannot reconstruct histories or states. Mark missing token traces explicitly; do not silently regenerate them as though they were the original executions.
5. Separate budget exhaustion, detected repetition, premature stop, server/transport error, and answer-extraction error. Audit the single short empty response separately from long exhaustion.
6. Produce an effective configuration diff from container arguments, environment, installed-file hashes, and startup logs. Include speculative method/depth, EP/TP/DCP, ownership, draft-slot sharing, cache dtype/record, recurrent-state mode, graphs, prefix caching, stop/EOS handling, and parser behavior.
7. Before rerunning a slow fixture, inspect its existing paired outcomes. Prefer a known score-driving pair and a normal completed control over arbitrarily choosing question 79 again.

**Gate:** do not run GPU comparisons with ambiguous effective settings or mismatched prompts. Repair measurement ambiguity first.

## Phase 1 — capture once, replay cheaply

Start with one resident 0.9.1 process. Use a small completed control and one existing problematic prompt/seed as diagnostic inputs. Keep initial capture bounded; do not immediately regenerate a 327K failure.

Extend the existing clients/backends with opt-in debugging hooks, not a separate full evaluation framework. Reuse:

- `tools/clients/decode_prefill_consistency.py`
- `tools/kernel/kpool_tail_index_check.py`
- `tools/kernel/kpool_rejected_draft_test.py`
- `tools/clients/hard_prompt_screen.py`

Capture token ids, effective configuration, kernel path/shape, selected indexer scores and pools, routing metadata, and narrowly selected state snapshots. Record checksums during routine execution; dump tensors only around a discrepancy. Measure instrumentation overhead against an uninstrumented short control. Verify the hooks have not changed ordinary outputs beyond the stack's measured variation.

### 1A. Speculative commit/rollback invariants — first priority

Construct controlled accepted-token histories for no speculation, ×3, and ×5. Define whether an acceptance count includes a bonus token; make accounting identical in the reference and tested paths.

Force acceptance of zero, one, intermediate, and all draft tokens. Include a rejected pool-completing token and a following step. Cross replay-ring wrap and actual allocator-page boundaries derived from the configuration. Check cancellation/reallocation as a separate case.

Compare post-commit state with ordinary incremental target execution of exactly the same accepted token ids:

- KDA recurrent and convolution state;
- kpool committed keys, tail entries, and indexer state;
- target and draft cache mappings and valid lengths;
- next-token logits after the rejection and on subsequent steps.

Integer mappings and ownership must agree exactly. Compare floating tensors with a declared reference tolerance and measured repeat variation, not an arbitrary bitwise requirement. Rejected drafts must not affect subsequent committed execution.

**Actionable positive:** a repeatable mismatch tied to an acceptance length or boundary. Reduce it to the smallest sequence/layer and patch that path before behavioral screening.

**Bounded negative:** invariants pass for the tested shapes and boundaries. This does not certify all long-context behavior.

### 1B. Sparse selection and the retained-pool contract

Freeze actual score tensors and metadata. Repeat the selection kernel on identical inputs and compare against deterministic top-k with a declared tie rule. Test unique scores, threshold ties, near ties, incomplete tails, and relevant context/batch shapes. Generate larger-context synthetic fixtures without requiring model generation, then validate representative shapes against captured real inputs.

Measure separately:

| Measurement | What it distinguishes |
| --- | --- |
| Selected-set changes | Different historical evidence reaches attention |
| Ordering changes with identical sets | Potential reduction-order variation |
| Pool removed by the 511-pool slice | Interaction between selector ordering and its consumer |
| Downstream attention difference on frozen query/cache | Numerical consequence of the selection discrepancy |

Audit causal bounds, duplicates, sentinel handling, and tail inclusion. Do not assume a new deterministic ordering is compatible with the existing slice: establish the intended retained-set contract first [S3].

**Actionable positive:** identical inputs produce semantically different retained sets, or selection violates the reference contract. First reproduce it; then test deterministic selection/explicit retained-set handling as the isolated candidate.

### 1C. Mixed-projection MoE fidelity

Capture a few real activation/router inputs, including mixed gate/up/down tiers and high-activation cases. Audit per-projection K3/K4 maps, rotations, scales, global/local expert ids, and TP rank slices.

Use the same reconstructed quantized weights for a high-precision reference, EP2, and TP2. Compare projection outputs, expert outputs, weighted expert combination, and the downstream residual. Do not compare against unquantized BF16 weights when the question is runtime fidelity to this fixed quant.

Check `swiglu_limit=10.0` handling explicitly. It is a shared fidelity question, not an independently sufficient explanation for the version difference. Treat native-reference clamping as a separate change unless evidence establishes an interaction.

**Actionable positive:** mapping/shape/activation-semantic error, or substantial excess error associated with a specific path. Ordinary small EP/TP rounding differences nominate further investigation only if they propagate materially.

## Phase 2 — locate the first model-level discrepancy

Replace independently sampled continuations with **teacher-forced incremental replay of one identical token sequence**. Compare both configurations on that sequence. For speculative comparisons, also force identical accepted histories and rejection schedules.

1. Establish repeat variation within each configuration on frozen inputs before comparing configurations.
2. Start with short prefixes and targeted boundaries. Add 8K/32K and then longer prefixes only when evidence indicates context dependence.
3. Compare ordinary incremental decode, speculative verification, and prefill. Prefill is another path to validate, not an unquestioned oracle.
4. Compare next-token distribution, reference-token log probability, and probability mass for the tokenizer's actual reasoning-close and termination tokens. Preserve full-vocabulary normalization; shared-top-20 KL alone can hide disagreement outside its intersection.
5. On the first material discrepancy, capture layer outputs and state around that position. Trace attention selection, KDA state, expert output, and residual processing to identify the first affected operation.
6. Compare eager versus graph execution if the discrepancy is execution-mode dependent. Use targeted sanitizers on a reduced reproducer when evidence points to invalid access or synchronization; do not sanitize a full long-generation run [S8].

Use progressive localization rather than collecting all intermediate tensors at every token. Record magnitude relative to the same-path noise floor. A changed top-1 token with a tiny margin is not by itself an engine defect or a completion explanation.

Do not create a long recurrent-state reference solely by prefill and assume it equals sequential decode. Prepare the accepted history through the appropriate incremental path. Reuse compatible local snapshots where possible; do not transplant cache tensors across versions/layouts without a validated conversion.

### Version bridge, only when needed

Use these arms to distinguish release bundles. Every row serves the same K3.25 payload:

| Arm | Engine | Draft depth | Purpose |
| --- | --- | ---: | --- |
| O | Stock 0.7.0 | 5 | Historical anchor; known incorrect cache behavior |
| F | 0.7.0 + applicable correctness backports | 5 | Does the advantage depend on old bugs? |
| N5 | 0.9.1 | 5 | Compare corrected generations at matched draft depth |
| N3 | 0.9.1 | 3 | Isolate draft-depth change within the new engine |
| N0 | 0.9.1 | Off | Remove speculative execution |
| F0 | Corrected 0.7.0 | Off | Ordinary-decode bridge, if N0 still differs |

Select the smallest comparison supported by Phases 1–2; do not launch the whole matrix by default. Verify backport applicability and installed hashes. Corrected 0.7.0 is not equivalent to 0.9.1. The DCP1-specific fix is irrelevant to a DCP2 path unless that path is changed.

If only stock O behaves better, investigate accidental effects of the incorrect computation. Do not turn a beneficial benchmark artifact into a proposed correctness regression.

## Phase 3 — test whether the nominated change affects completion

Compare **0.9.1 baseline versus one candidate**, with the candidate changing only the localized component. A mechanically correct patch and a behavioral improvement are separate claims.

### Exploratory pilot

Freeze a manifest before running. Suggested starting scope: eight prompt ids × two fresh seeds × two arms, with a 16K output cap. Include score-driving/hard cases and ordinary controls. Label this an enriched diagnostic set, not a benchmark-wide estimate. Select seeds not used to discover the candidate; use the same seed set across arms without assuming trajectories will match.

Run at matched concurrency that both arms support without KV waiting/preemption. Start with low-concurrency localization; reproduce a promising result at the original C8 serving condition before a release-level claim. Record actual batch composition and request order. Counterbalance baseline/candidate order where practical and amortize loading within each immutable configuration.

Collect final-answer completion, finish reason, reasoning closure, repetition indicators, answer correctness, output tokens, and occupied GPU-hours. Report throughput separately from completion.

At 16K, an unfinished request is **right-censored**, not a failure under the original 327,680-token budget. Use 16K and, if justified, a predeclared 32K follow-up to rank candidates. Do not equate shorter-budget completion with recovered original-budget accuracy. Question 79 includes an observed successful 314,625-token response [S7].

If early loop detection is used, freeze the classifier and report detector-triggered stops separately. Test a sample of flagged continuations without early cancellation before treating the detector as a reliable terminal-failure proxy.

**Advance:** candidate resolves the localized discrepancy and shows enough behavioral promise to justify a long-budget test. Do not require a significant pilot p-value.

**Do not advance:** no reproducible mechanism and only a few favorable sampled outcomes. Report the pilot and investigate another branch within the remaining budget.

## Phase 4 — independent confirmation at the original budget

Before launching, write a dated protocol containing the candidate, holdout prompt/seed manifest, cap, concurrency, minimum relevant completion improvement, analysis, exclusions, and compute ceiling. Estimate power and token cost from existing rows and pilot data, with questions as clusters. A roughly two-point absolute failure-rate change may require hundreds or more paired observations; do not promise that another 12-repeat screen can resolve it.

Use the original 327,680 cap and C8 where both arms can support it. Fresh seeds on previously used prompts can test those prompts' failure probabilities, but do not make those prompts an untouched benchmark sample. To claim a benchmark-wide effect, use a representative design; report enriched and representative results separately.

Primary outcome: empty final answer at the original budget. Secondary outcomes: audited and raw accuracy, correct completed answers per occupied GPU-hour, output-token cost, and finish categories. Preserve the existing scoring protocol. Record request retries explicitly; errors must not silently become new draws.

Use a fixed terminal analysis, or specify an appropriate sequential method before starting. Do not repeatedly inspect ordinary p-values and stop when favorable. Report effect intervals and uncertainty, not just a pass/fail threshold.

Only after confirmation should a full benchmark rerun establish a new headline score. Retain the historical records; do not overwrite them with the candidate's results.

## Required artifacts and reporting

Create a new dated investigation directory, following repository conventions. Suggested contents:

```text
README.md                 # question, current verdict, evidence, limits
PLAN.md                   # frozen plan and dated amendments
manifest.json             # immutable inputs, arms, fixtures, seeds, settings
costs.jsonl               # allocation/startup/warmup/test costs per arm
checks.jsonl              # invariant and frozen-input comparison results
candidate.md              # localized cause, proposed patch, residual risks
recompute.py              # reproduces public tables from public numeric rows
local-fixture-index.json  # hashes and local references, without publishing restricted bytes
```

Keep raw private fixture bytes outside the public checkout. Reuse the repository's `run.json`, result, and summary conventions where possible. Publish new protocols for changed probes rather than silently changing existing v1/v2 measurements.

Each experiment report must state: hypothesis; exact controlled difference; reference/noise floor; measured discrepancy or bounded null result; compute cost; next decision. Verification should reproduce the proposal's baseline table and all new numerical claims.

## What to do next

1. Recompute the score-driving pairs and effective configuration diff.
2. Implement the smallest reusable forced-acceptance/state-comparison hook.
3. Run ×3/×5 rollback cases around pool/ring/page boundaries on 0.9.1.
4. Capture one actual indexer score fixture and test set/order/511-slice stability.
5. Follow the first reproducible discrepancy into a minimal candidate.
6. Spend long-generation compute only on that candidate or a clearly documented unresolved branch.

If all short invariants pass, prioritize matched-depth corrected-old versus new incremental replay. If that also shows no material difference, invest in a small long-context fixture and explicitly revisit whether the completion advantage is robust across fresh seeds. Passing local checks does not prove long-generation equivalence.

## Sources

The snapshot identities above govern this proposal; source links below are convenient navigation links and may advance after preparation.

- **S1:** [Findings and qualifications](https://github.com/UrbanAstroLA/local-inference-evals/blob/main/FINDINGS.md).
- **S2:** [Three-pass record comparison](https://github.com/UrbanAstroLA/local-inference-evals/tree/main/comparisons/glm53-flash-gpqa-records); source rows under `runs/2026-09-29_glm53-flash_k3.25-v0.7.0-dflash5_gpqa-diamond/` and `runs/2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3_gpqa-diamond/`.
- **S3:** [Completion investigation](https://github.com/UrbanAstroLA/local-inference-evals/blob/main/investigations/2026-10-glm53-looping/README.md), including code questions; [confounds](https://github.com/UrbanAstroLA/local-inference-evals/blob/main/CONFOUNDS.md); [ledger](https://github.com/UrbanAstroLA/local-inference-evals/blob/main/LEDGER.md).
- **S4:** [Engine provenance and correctness fixes](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/blob/main/PROVENANCE.md).
- **S5:** [Default-selection development arms](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/blob/main/benchmarks/v0.8.0-dev/README.md); [engine README](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx).
- **S6:** [GPQA protocol](https://github.com/UrbanAstroLA/local-inference-evals/blob/main/protocols/gpqa-diamond/v1.md); [decode/prefill protocol](https://github.com/UrbanAstroLA/local-inference-evals/blob/main/protocols/decode-prefill-consistency/v1.md); [distinct-seed screen protocol](https://github.com/UrbanAstroLA/local-inference-evals/blob/main/protocols/hard-prompt-screen/v2.md).
- **S7:** [Question-79 0.9.1 screen rows](https://github.com/UrbanAstroLA/local-inference-evals/blob/main/runs/2026-10-09_glm53-flash_k3.25-v0.9.1-dflash3_screen-doc79/results.jsonl).
- **S8:** [NVIDIA Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

The common-completion table is a derived calculation from S2. Experiment priorities, resource ceilings, and advancement rules are proposals, not findings established by these sources.

## Dated amendments

**Amendment 1 (2026-10-10, written before any Phase 1 data).** First GPU step: decode-vs-prefill consistency
(`decode-prefill-consistency/v1`, 12 prompts, seeds 1234 + prompt, 2,600 tokens, concurrency 1, prefix caching off,
speculation off) on the tpurtell 0.7.0 image, two runs, plus one fresh run on 0.9.1 to add to its two runs of 2026-10-08.
Question: does the 0.7.0 generation path agree with its own prefill re-scoring better or worse than 0.9.1's?
Reading rule: compare mean shared-top-20 KL below 2,044 and from 2,048 tokens. A difference counts only if every 0.7.0
run lies outside the range of all 0.9.1 runs in the same region (and vice versa); otherwise "no difference beyond
run-to-run variation". Speculation off isolates the attention and expert paths from speculative execution; draft depth
is a separate step. Budget: about 1 wall-hour (about 2 occupied GPU-hours), counted in costs.jsonl.

**Amendment 2 (2026-10-10, after step 1, before step 2 data).** Step 1 result (rule of amendment 1): below 2,044 tokens
the 0.7.0 and 0.9.1 ranges overlap (0.0061-0.0104 vs 0.0072-0.0077); from 2,048 tokens both 0.7.0 runs (0.0142, 0.0145)
lie just above all three 0.9.1 runs (0.0053-0.0139), a hairline separation in the direction of 0.7.0 agreeing worse.
Cost 0.77 wall-hours (1.54 GPU-hours). Step 2: the same test on 0.9.1 with speculation on, DFlash2 x5 (two runs) vs x3
(one new run plus the 2026-10-08 run), prefix caching off. Question: does deeper speculation change what the model
computes along its own trajectory? Rule: a difference counts only if both x5 runs lie outside the range of both x3 runs
in the same region; the speculation-off runs of step 1 (0.9.1) give the reference range. Budget about 1 wall-hour.

**Amendment 3 (2026-10-10, after step 2, before step 3 data).** Step 2 result (rule of amendment 2): DFlash2 x5 and x3
overlap in both regions (below 2,044: x5 0.0100-0.0206, x3 0.0105-0.0131; from 2,048: x5 0.0066-0.0268, x3 0.0115-0.0186).
Not targeted by the rule but recorded here before further data: below 2,044 tokens every speculation-on run on 0.9.1
(0.0100-0.0206, four runs) lies above every speculation-off run (0.0072-0.0077, three runs); from 2,048 they overlap.
Cost 0.80 wall-hours (1.61 GPU-hours). Step 3: the same test on tpurtell 0.7.0 as shipped, speculation on, DFlash2 x5,
prefix caching off, two runs. Question: is 0.7.0's speculation-on agreement better than 0.9.1's? Rule: a difference
counts only if both 0.7.0 runs lie outside the range of all four 0.9.1 speculation-on runs (x3 and x5) in the same region;
the comparison is engine as shipped vs engine as shipped plus x5 on 0.9.1. Budget about 0.6 wall-hours.

**Amendment 4 (2026-10-10, after step 3, before step 4 data).** Step 3 result (rule of amendment 3): 0.7.0 with
speculation on (x5) lies inside the range of 0.9.1's four speculation-on runs in both regions (below 2,044: 0.0106-0.0138
vs 0.0100-0.0206; from 2,048: 0.0135-0.0165 vs 0.0066-0.0268): no difference. Speculation raises the disagreement in both
engines (0.7.0 below 2,044: off 0.0061-0.0104, on 0.0106-0.0138). Cost 0.42 wall-hours (0.84 GPU-hours).
Step 4: the same client at 16,000 generated tokens, prompts 0-5, both engines as shipped (0.7.0 DFlash2 x5 with its
layout; 0.9.1 DFlash2 x3), prefix caching off, two runs each, alternating engines. Positions are binned 0-2,043,
2,048-7,999 and 8,000-15,999. Question: does 0.9.1's decode drift from its prefill more than 0.7.0's as context grows?
Rule: a difference counts only if both runs of one engine lie outside the range of both runs of the other in the
8,000-15,999 bin; the other bins are reported. Budget about 1.5 wall-hours.

**Amendment 5 (2026-10-10, after step 4, before step 5 data; adopts the independent assessment of 2026-10-10).**
Wording: step 4 shows that stock 0.7.0 has greater disagreement between speculative generation and chunked-prefill
rescoring on its own generated histories; steps 1-3 did not separate the configurations under their screening rules.
Prefill is not a correctness oracle, overlapping ranges are not equivalence, and lower KL is not evidence of better completion.
Budget: about 4.82 GPU-hours remain; reserve about 2.2 for step 5 and spend the rest on one follow-up chosen from its result.
No full GPQA pass in this stage.

Step 5. Question: does applying the two kpool fixes reduce stock 0.7.0's long-context disagreement? Comparison: stock 0.7.0
(DFlash2 x5, EP2/DCP2; the two step-4 runs) vs the same stack with only the two kpool backports (the locally built
kpool-backport image, sha256:f68e47a0…; verified 2026-10-10: only attention.py (tail-ring sizing) and kpool_compress.py
differ from stock, the latter byte-identical to the 0.9.0/0.9.1 file; kernel tests 33/33). 0.9.1 is context only.
Conditions: protocol decode-prefill-consistency/v2 (16,000 tokens, prompts 0-5, DFlash2 x5, prefix caching off,
concurrency 1, existing seeds), two runs; measurement starts only after the container's startup warm-up has finished
(its ready file exists), checked in the log. Primary clean subset: prompts 1-5 for stock, corrected and 0.9.1 (step 4's
prompt 0 overlapped warm-up); prompt 0 reported separately; step 4's preregistered verdict is unchanged. The client keeps
token ids and decode/prefill log probabilities locally (trace mode); only hashes and numeric summaries are published.
Primary bin: 8,000-15,999. Report aggregate KL for prompts 1-5, per-prompt KL per repeat, sampled-token log-probability
disagreement, top-1 agreement, other bins as secondary. Reading rule (exploratory screen): if both corrected aggregates lie
below both stock aggregates and the reduction holds in most prompts (at least 4 of 5 per repeat), nominate the kpool
bundle as affecting the measured discrepancy; if ranges overlap or reductions are inconsistent, unresolved under this
screen; if corrected values remain at stock level, these repairs did not explain the discrepancy under this protocol.
Nearness to 0.9.1 is descriptive only. Follow-ups: A (falls) targeted stock-vs-corrected behavioral pilot on 2-3
score-driving prompt/seed pairs plus one control, x5 and layout fixed, GPQA sampling, pre-set 16K or 32K cap, unfinished =
right-censored; B (stays elevated) frozen-score indexer fixture; C (mixed) fixed-history rescoring across builds.

**Amendment 6 (2026-10-10, after step 5, before step 6 data).** Step 5 result (rule of amendment 5; warm-up gated,
first request 0.4-0.6 s after the ready file): prompts 1-5, positions 8,000-15,999, aggregate KL corrected 0.7.0 0.0439 and
0.0390 vs stock 0.0558 and 0.0243 (0.9.1 context 0.0142, 0.0119); lower than both stock repeats in 2 and 1 of 5 prompts.
Verdict: the corrected values remain at stock level; these repairs did not explain the discrepancy under this protocol.
Cost 0.81 wall-hours (1.62 GPU-hours); about 3.2 GPU-hours remain. Branch B applies. Before the engineering-heavy
frozen-score fixture, step 6 (no new code): tpurtell 0.9.1 image with v0.7.0's parallel layout (ENABLE_EXPERT_PARALLEL=1,
DECODE_CONTEXT_PARALLEL_SIZE=2, so MLA ownership off and 656-byte records; VLLM_B12X_DCP_TOPK_OWNER_EXCHANGE=0, the
global top-k merge fallback, because the owner exchange fails on this fork) at DFlash2 x3, protocol v2, two runs,
warm-up gated, traces local. Comparison: 0.9.1 at its default DCP1 layout (step 4, x3). Question: does the v0.7.0 layout
on 0.9.1 kernels reproduce stock 0.7.0's long-context disagreement? Reading rule (prompts 1-5, 8,000-15,999): if both
step-6 aggregates lie above both 0.9.1-default aggregates and inside or above the stock-0.7.0 range, with the increase in
at least 4 of 5 prompts per repeat, nominate the layout bundle (DCP2/EP2/ownership off/record format/top-k merge path) as
affecting the discrepancy; if both stay within the 0.9.1-default range, nominate the 0.7.0 kernel/engine code instead;
otherwise unresolved. The bundle is not split into components by this step.

**Amendment 6a (2026-10-10, written while step 6 was running, before any step-6 data was inspected; replaces the
interpretation clauses of amendment 6, not its measurement or thresholds).** If step-6 disagreement rises as defined: the
old layout bundle can reproduce the discrepancy on newer code; this does not identify which layout component causes it.
If it stays within the 0.9.1-default range: the bundle alone does not reproduce the discrepancy under these conditions;
old kernel code or a kernel-layout interaction remains possible. Otherwise: unresolved under this screen.

**Amendment 7 (2026-10-10; adopts the independent next-stage assessment of 2026-10-10).** Priorities: a targeted
behavioral comparison before any further subdivision of the generation-vs-prefill (KL) discrepancy; GPU time in tranches
(Tranche 1: up to +4 GPU-hours, ceiling 16; Tranche 2 conditional, up to +8, ceiling 24); the step-6 configuration is
named the "0.9.1 EP2/DCP2 diagnostic bundle" (global top-k merge fallback instead of stock's owner exchange; x3 drafts).
Exploratory CPU trace audit (existing step-5 and step-6 traces, 24 traces; primary results unchanged): ignore_eos in this
vLLM build only disables stopping (no end-token masking without min_tokens). Termination tokens: <|endoftext|> 154820,
<|user|> 154827, <|observation|> 154829; reasoning close </think> 154842. In all 24 traces the model closed its reasoning
and emitted a terminator at 637-5,377 generated tokens; 100% of the 8,000-15,999 bin is forced continuation after the first
natural termination (repeated synthetic turns, up to 93 terminators). Step-5 and step-6 disagreement in that bin: 0.0404
and 0.0403. Step-4 runs kept no traces; the same prompts, seeds and settings make the same pattern likely there but it is
not verified. Decision per the assessment: the long-context discrepancy of steps 4-6 measures forced post-answer
continuation; its component bisection is deprioritized as an explanation for non-completion. The behavioral pilot remains
necessary; its manifest will be frozen in amendment 8 before it runs.

**Amendment 8 (2026-10-10, written by the operator before the pilot; recorded verbatim in substance).** Completion pilot
and staged GPU authorization. Objective: determine whether the 0.9.1 diagnostic EP2/DCP2 bundle changes useful completion
behaviour relative to 0.9.1's default layout; the earlier post-completion KL discrepancy does not establish a mechanism.
Tranche 1 approved: up to 4 additional occupied GPU-hours; earlier GPU time stays in the cumulative ledger; this pilot has a
3 GPU-hour ceiling including startup, warm-up, measurement, shutdown and invalid runs. No further tranche is authorized.
Frozen manifest (case, original seed, fresh seed, output cap, both arms): q55 1235 7055 65,536; q109 1234 7109 65,536;
q170 1236 7170 131,072; q144 control 1234 7144 65,536. Arms: A 0.9.1 default layout; B 0.9.1 diagnostic EP2/DCP2 bundle.
Both: identical weights, rendered prompts, parser and sampler; DFlash2 x3; temperature 1.0, top_p 0.95; normal stopping, no
ignore_eos, no early loop termination; warm-up gated; concurrency 8 (the original evaluation's), the two seed groups being
analysis groups submitted together. Record admission order, active-request counts, cache waits and interruptions.
Primary: valid final answer by each case's cap; output tokens to first valid final answer. Secondary: completion by 32k and
64k (and 128k for q170); correctness, reasoning-close position, finish reason; repetition vs substantive reasoning; GPU-hours
and tokens. Unanswered at the cap = right-censored, not equivalent to the original 327,680-token failures. Arms compared at
common thresholds within each case; caps not pooled. Interpretation and next-expenditure rules as written by the operator
(multiple-case improvement incl. fresh seeds -> replication or focused component test; single original seed -> fresh-seed
replication first; both finish -> original conditions not reproduced; both censored -> inspect traces before longer runs;
B worse -> deprioritize; control changes -> investigate before interpreting rescues). No KL experiments, full GPQA pass or
new tranche on an inconclusive result. Report: 1 reproducible behavioural candidate, 2 fixture-specific observation, or
3 no useful separation within the window.

Pre-launch record (coordinator, before data): measured throughput 446 tok/s aggregate with 8 running (0.9.1 default, its
GPQA passes, median of 1,616 status lines); single-stream 113 tok/s (default) vs 98-100 (bundle) at 16k. Worst case per arm
(every request to its cap: 655,360 tokens): about 20 min with 8 running, then the two q170 requests alone about 11-13 min,
plus about 9-10 min startup: about 40-45 min per arm, about 2.7-3.0 GPU-hours for both. To hold the 3 GPU-hour ceiling,
each arm is stopped at 45 minutes after its server start; requests still running then are recorded as censored by the time
limit (distinct from the token cap). KV: default 4,707,515 and bundle 3,199,879 tokens vs at most about 660,000 needed;
max_model_len 1,048,576 covers prompt plus 131,072. Client: streaming chat client (manifest seeds and caps, early stop off,
stop ['</s>'] as in the GPQA task, local reasoning text). Manifest sha256 prefix 33d2f8fb5cd5b60d.

Launch note (coordinator): after the runner started and before arm A's client started, the client was changed to record
requests open at the time limit as finish_reason 'time_limit' with partial text, and to record time to first chunk. Request
bodies, seeds, caps and order were unchanged. Neither arm reached its time limit. Results: `pilot/table.md`, `pilot/report.json`.

**Amendment 9 (2026-10-11, approved by the operator before data).** Fresh-seed replication of the pilot's two target
observations (q109 and q170: arm A censored at the cap, arm B answered, one draw each). Arms, weights, prompts, sampler,
draft depth, stopping, warm-up gate, concurrency 8 and caps are exactly as in amendment 8. Frozen manifest (sha256 prefix
c6b10710ce414be4), fresh seeds following one rule (7000 + 100k + question, k = 2..5): q170 seeds 7270, 7370, 7470, 7570, cap
131,072; q109 seeds 7209, 7309, 7409, 7509, cap 65,536; submitted together in that order. No control: the pilot's control
was unchanged in both arms, and adding it would put more than 8 requests in flight; execution is checked by the gate record
and scheduling counts.

GPU provisioning: up to 1.1 GPU-hours beyond Tranche 1 (12.56 used, 1.99 left in Tranche 1; cumulative ceiling 15.65). Estimate from
the pilot's measurements: about 58 tok/s per request with 8 running and about 75 with 4, startup 10-13 min per arm;
expected about 2.5 GPU-hours, worst case (every request to its cap) about 3.1. Each arm is stopped 46 min after server
start; requests open then are recorded as censored by the time limit, distinct from the token cap. Replication ceiling
3.1 GPU-hours.

Analysis, fixed before data. Per case and arm: valid final answers by the cap out of 4, and tokens to answer, compared at
the common thresholds (32k and 64k; 128k for q170). Primary: the 8 new requests per arm. Secondary: pooled with the pilot's
fresh seed (5 per case per arm), labelled as pooled. Reported with the exact one-sided test over requests. Classification:
(1) reproducible behavioural candidate if arm B has fewer censored replies than arm A in both cases and the one-sided p
over the 8 new requests per arm is below 0.05; (2) observation still requiring replication if the direction holds in both
cases without reaching that; (3) no useful separation otherwise. Requests are draws within two fixtures; results do not
generalize beyond these questions, and the same seed on different layouts is not a matched trajectory.

**Amendment 10 (2026-10-11, written by the operator before data).** Kernel and DCP investigation. Objective: explain why
stock 0.7.0 completed more GPQA replies than 0.9.1 on the same K3.25 weights, and identify a correct intervention if
possible, by direct examination of attention/indexer kernels and DCP execution. No further speculative-decoding pilot at
this stage. Basis: draft depth 3 to 5 (with sharing changed) did not improve completion in the surviving 0.8.0 GPQA
comparison; the serving and short-context probes found no speculative mechanism; the rejected-draft kpool corruption is
repaired and tested; the old-layout behavioural intervention gave no reproducible benefit; the long-context KL result
measured forced continuation after natural completion. Kernel/DCP mechanisms remain plausible because they have not been
isolated; limited exploration is a reason to inspect them, not evidence that they are the cause.

Budget reconciliation: read additively, the approved ceilings are 12 + 4 + 1.1 = 17.1 GPU-hours; 14.10 consumed, 3.0
remain. (Amendment 9's "cumulative ceiling 15.65" counted Tranche 1 from consumption rather than from the discovery
allocation; the additive reading is the operator's and supersedes it.) No new GPU time requested. Allocation: Phase 1
source and provenance audit, CPU only; Phase 2, one discriminating kernel/DCP fixture, ceiling 1.5 GPU-hours; Phase 3,
replication or intervention test, up to the remaining 1.5, only if Phase 2 is actionable. Ceilings include startup,
warm-up, measurement, shutdown and invalid runs; engineering time tracked separately.

Phase 1: source-backed comparison of 0.7.0, 0.8.0 and 0.9.1 covering (1) sparse MLA indexer selection: pool eligibility
and incomplete tail, selected-pool count including the 511-pool slice, local vs global top-k, ties, ordering and
duplicates, owner exchange vs global-merge fallback; (2) DCP attention: token ownership and positions, cache writes, reads
and addressing, pool/ring/page boundaries, normalization and cross-rank combination, accumulation precision and reduction
order; (3) kernel dispatch and storage: kernels actually executed per configuration, KV record readers and writers,
graph vs eager, scratch initialization and reuse, kernel changes in the newer backend. Expert partitioning kept separate
from attention partitioning; settings that must change together documented. Per difference: source location and version,
existing tests and coverage, a concrete failure mechanism, the cheapest distinguishing test. No assumption that a newer
kernel can be transplanted into the older engine.

Phase 2: one controlled fixture on identical inputs and histories, in priority order where the audit supports it:
(A) indexer selection on frozen scores and eligibility masks against a small explicit reference (ties, tails, count
boundaries, cross-rank merge, owner exchange vs fallback; a different tie choice is a defect only if it violates the
intended rule); (B) DCP attention on semantically identical queries, caches and positions, comparing eligible and
selected tokens, normalization, rank-local contributions and combined output against a validated reference (neither
prefill nor the older engine is automatically an oracle); (C) committed-state replay of one identical history, only for
accumulated-write, ownership or reuse mechanisms (a newly built cache tests reads and selection, not incremental writes).
Controls: effective configuration verified from logs and dispatch; repeatability or tolerance within each path; weights,
history and inputs fixed; one compatible component at a time; inseparable settings labelled as a bundle; legitimate
floating-point variation distinguished from invariant failure; intermediate receipts and trace hashes kept. The
generation-vs-prefill KL sweep is not repeated; forced post-completion continuations are not evidence about non-completion.

Phase 3: the reserve is spent only on an actionable result (reproducible invariant violation; incorrect selection or
addressing against a validated reference; a difference localized to one operation beyond its measured repeatability and
tolerance): reproduce on another input or boundary, test the smallest correct intervention, confirm it removes the
localized problem. Expected numerical variation is documented, not followed by an automatic behavioural pilot. With no
actionable difference, the tranche stops and remaining uncertainty is reported. Further GPU time requires a concrete
proposal (mechanism and evidence, competing explanations, the distinguishing observation, measured throughput and maximum
GPU-hours, why a shorter fixture cannot answer). If the remaining question is whether legitimate numerical differences
change rare completion tails, that is stated explicitly and needs a separately budgeted distributional comparison.
Conclusions kept separate: (1) a kernel or state-management defect exists; (2) its correction changes completion on
selected fixtures; (3) it explains or closes the representative GPQA completion gap.

**Amendment 10a (2026-10-11, coordinator, Phase 1 complete, before Phase 2 data).** Phase 1 result: three independent
source audits (indexer selection; DCP attention; kernel dispatch and storage) found no defect unique to 0.9.1 in the
attention math, DCP combine, record readers or graph/scratch handling; attention differs between 0.7.0 and 0.9.1 only by
floating-point ordering. Known indexer write defects (tail-seed stride, 4-slot tail ring with five drafts) are 0.7.0/0.8.0
only and fixed in 0.9.1, so they cannot explain 0.9.1 completing less. One selection defect candidate is common to all
versions and was flagged by all three audits: the engine asks the top-k kernels for 512 compressed-key pools, then keeps
output columns 0-510 to make room for the 0-3 newest tail tokens. No top-k path sorts its output, so the dropped column is
not guaranteed to be the lowest-scoring pool; which kernel path supplies it differs by version and by the number of
requests in flight (0.9.1: rewritten fused cooperative merge with up to 4 requests, tiled with 5-8; 0.7.0: per-rank top-k
then a cross-rank merge). Upstream vLLM keeps all 512 pools (2,051 columns), so the drop is specific to this engine.
Other candidates recorded and not selected for Phase 2: the default-on two-GPU remote-push all-reduce transport new in
0.8.0 (no runnable correctness test); a teacher-forced NLL rise of about 0.007 from 0.7.1 to 0.8.0+ in the engine's own K4
records (one 0.7.1 pass); 528-byte record writer equivalence (unverified, expected rounding only).

Phase 2 fixture (type A, indexer selection), chosen because it is the only identified mechanism that changes which context
is attended at long lengths, whose supplying path differs between the engines, and that can be tested on frozen inputs
against an explicit reference. Reference, fixed now: the intended selection is the 511 highest-scoring complete pools by
exact fp32 score, plus the 0-3 tail tokens; a tie at the 511/512 boundary may be resolved either way. Stage 1 (one GPU, no
model, ceiling 0.2 GPU-hours): drive each release's real top-k paths (0.7.0: tiled and fused serial per rank plus the
cross-rank merge; 0.9.1: fused cooperative and tiled) with frozen inputs at pool counts from 512 to 30,000 and 1-64 rows,
including constructed boundary rows (exactly 512 candidates above a coarse bin edge; plateaus straddling rank 512), each
input repeated to measure run-to-run variation; record the reference rank of the pool in column 511 and selected-set
equality. Actionable: column 511 holds a pool ranked well inside the top 511 (not a boundary tie) at a non-negligible rate
on any shipped path. Not actionable: column 511 always holds the 512th pool or a boundary tie. Stage 2, only if Stage 1 is
actionable (ceiling 1.3 GPU-hours, remainder of Phase 2): the same measurement in place on real serving inputs through an
observe-only hook on 0.9.1 and 0.7.0, at the concurrency mix of a GPQA pass. A finding would establish conclusion (1) at
most; it would not by itself show a completion effect.

**Amendment 10b (2026-10-11, coordinator, after stage 1 and before stage 2 data).** Stage 1 met its criterion on synthetic
inputs (0.9.1 tiled path: the dropped pool ranked better than 480 in 94-98% of rows at 1,024-16,384 pools; 0.9.1 fused
0-1%; 0.7.0 per-rank plus merge 3-21%; no set mismatches beyond boundary ties). Stage 2 design (ceiling 1.3 GPU-hours,
startup, warm-up and invalid runs included):
- Capture, one 0.9.1 server run on both GPUs: the release image plus an observe-only hook in the decode branch of the
  sparse indexer (a local derived image; no kernel or selection code changed). For sampled decode rows the hook stores the
  real query bytes, head weights, the row's index-K pool entries, sequence length, row count and the kernel's output pool
  order, and computes the rank of the column-511 pool from the stored bytes. Graph mode piecewise, so the hook runs at
  every step (the indexer is an eager-break op there; the fused/tiled route still follows the live row count).
- Histories: real model reasoning from the pilot's retained traces, appended to their rendered GPQA prompts at lengths
  from about 4,000 to about 120,000 tokens; each history is prefilled once (prefix cache) and then decoded briefly with 8
  requests together (tiled path) and with 4 or fewer (fused path).
- Replay, one GPU: the same stored bytes through 0.9.1 fused and tiled and 0.7.0 per-rank top-k plus cross-rank merge
  (stage 1 driver), against the fp64 reference from the same bytes.
Actionable: on real inputs, 0.9.1's tiled path drops a pool ranked better than 480 (not a boundary tie) in at least 10% of
sampled rows, and 0.7.0's path on the identical inputs does so materially less often. Then Phase 3 tests the smallest
correct intervention (discard the lowest-scoring selected pool instead of column 511) on the same stored inputs and in
place. Not actionable: real inputs leave column 511 at or near the boundary on every path; the difference is then
documented and the tranche stops.

**Amendment 10c (2026-10-11, coordinator, after stage 2 and before Phase 3 data).** Stage 2 met its criterion on real
inputs: 0.9.1's tiled path dropped a pool ranked better than 480 in 88.1% of 9,944 live decode rows (replay of the same
bytes 87.9-88.2%; 13% at about 120,000 tokens, 96-97% from 4,000 to 64,000); 0.9.1's fused path 0.4-0.5%; 0.7.0's path
on the identical bytes 6.9%. The route alone decides it. Deviations: graph mode NONE (the indexer is not an eager-break op
in this build, so under piecewise graphs a hook would not run); two server runs (the second the permitted retry). An
incidental observation, not part of any rule: in run 2, sixteen index-K pools at the same in-page offsets were
byte-identical across two different requests on all five DSA layers owned by GPU 1, some decoding as NaN.
Phase 3 (reserve; ceiling 0.7 GPU-hours for this part): the smallest correct intervention for the column-511 drop,
tested first offline on the stored stage-2 inputs (one GPU), then in place (one probe-image server run): (i) configuration
only: B12X_INDEXER_TWO_LEVEL_FOLD=0; (ii) code: request the kernel's selected scores and move the lowest-scoring selected
pool into column 511 before the slice (selection set unchanged). Pass: on the tiled route, column 511 holds the
lowest-scoring selected pool or a boundary tie in at least 99% of rows, the selected set is unchanged, the fused route is
unchanged, and decode throughput at 8 requests is reported. Whichever passes with less change is the candidate. This
establishes at most that the localized problem is removed; completion effect and the GPQA gap remain separate questions.
The pool corruption gets a CPU-only forensic pass on the stored captures first; any GPU reproduction is a separate
decision by the operator.

**Amendment 10d (2026-10-11; operator authorization: "This is significant enough finding that it warrants extending GPU
time where necessary for proper verifications").** CPU forensic result on the stage-2 corruption: the corrupted index-K
bytes are raw bf16 indexer key and gate rows for tail-ring slots 4-7, written through the tail-ring view (its padded
per-block stride, ring of 8) into two indexer blocks then owned by other requests, on rank 1 only, in all five DSA layers
that rank owns; compressing those rows as the kpool kernel does reproduces byte-for-byte the pool of shared prompt tokens
12-15, which is why the bytes matched across two questions. The seed-stride and spec-ring fixes behave as intended; the
DCP1 tail-compaction fix writes no cache. Excluded by layout or address arithmetic: draft slot sharing, KDA state, MLA
record writer, block-pool double ownership (GPU 0 clean at the same block ids). Nominated mechanism: a wrong tail-block id
reached the tail write path once on rank 1 (a stale request row in the tail group's block table, or a CPU/GPU race in the
model runner's staged block-table updates). Competing explanations: an artefact of eager mode with the probe's per-step
synchronisation; a scheduler-side id error that rank 0 masked. Seen once in two capture runs. Not explained: why only
slots 4-7 (a 16-token write ending at position 15).
Extension: up to 4 GPU-hours beyond the 17.1 already approved (cumulative ceiling 21.1), for defect verification only:
(T0, CPU) read the runner's tail-group row writes for new, resumed and reused requests and its asynchronous staging;
(T1) an observe-only guard at the tail seed and stash call sites that checks, per step and rank, the tail-block ids used
against that step's scheduled tail-group ids and against its attention-block ids, logging any violation with rank, step,
request row, mapping, positions and chunk length; run on the stage-2 workload in eager mode and in the shipped graph mode
(device-side check), repeated to estimate a rate; (T2) a single-GPU unit test reproducing the nominated row-reuse or race
condition; (T3) if a source is located, the smallest correct fix, verified by the same guard. Column 511 continues under
amendment 10c; its in-place check under the shipped graph mode may use this extension. Shorter fixtures cannot answer:
the event depends on live scheduling, request-row reuse and rank-local runner state. Each run is costed before launch
from measured startup (about 10-13 minutes) and workload time (about 15 minutes per capture run, two GPUs). Completion
effects (conclusions 2 and 3) are not covered; they need their own proposal.

**Amendment 10e (2026-10-11, coordinator, T0 result, before T1-T3 GPU data).** T0 (CPU, the real Triton tail kernels
run in the interpreter): in this build the hybrid model state's attention-metadata call omits `positions`
(`v1/worker/gpu/model_states/mamba_hybrid.py`, `build_attn_metadata(...)` without `positions=input_batch.positions`;
upstream vLLM has passed it since GLM-5.3-Flash support landed, commit 98ed0856f3, and still does). Without positions the
kpool tail group's intended slot mapping is never built; the generic per-group mapping is used instead, so positions 0-7
go to the request's own tail block and every position from 8 goes to column pos // 8 of the tail table, normally the
null block 0 and occasionally a stale id. Reproduced on CPU: a stale column 1 plus a prefill from position 0 writes
exactly the stage-2 signature (slots 4-7 of positions 12-15 into another request's first indexer page); concurrently
decoding requests share block 0's ring, so a pool completed across two steps can be compressed from another request's
keys (relative error 0.9 vs 0.0 with positions passed). The same omission is in the 0.7.0 source (inferred for 0.8.0);
by itself it is not a version difference. Sequence under amendment 10d: T2 on one GPU (real kernels and block tables;
graph-capture smoke of the guard); T1 guard runs on the unmodified 0.9.1 code, shipped graph mode first, then eager;
T3 = the one-line upstream backport (pass positions), verified by the same guard in both modes and checked for graph
safety (the tail mapping must be read by replayed graphs). Pass for T3: zero tail writes outside each request's own tail
block and zero null-block aliasing, with no new guard violations. Effects on selection quality and completion are not
covered.

**Amendment 10f (2026-10-11, operator decision, before T1-T3 server data).** The tail-ring omission is in 0.7.0's source as
well, so by itself it does not explain the version difference; its verification is trimmed to the GPU unit test, one
shipped-graph-mode guard run on unmodified 0.9.1 and one with the positions backport (eager runs dropped; ceiling 2.0
GPU-hours for T1-T3). In parallel, a CPU-only audit compares the fork's vLLM with upstream vLLM on the GLM-5.3-Flash paths,
because every defect found so far sits where the fork departs from or lags upstream. A completion test of the column-511
fix needs its own proposal afterwards.

**Amendment 11 (2026-10-11; adopts the independent assessment "validate column-511 correction, then test completion").**
Terminology: the intervention is a correction to the engine's 511-pool selection policy (discard the lowest-scoring
selected pool), not a restoration of upstream's 512-pool policy. Established: the engine truncates an unordered 512-pool
selection to 511; the discarded pool differs substantially between kernel paths; identical-input route interventions
reproduce the difference; moving the lowest-scoring selected pool last removes the localized error. Not established: its
effect on completion or accuracy, or how much of the engine gap it explains.
Within the remaining defect-verification authorization (3.99 GPU-hours), in priority order: (1) column-511 correction
verified in a full server under the shipped graph configuration (gate, below); (2) the tail fix completed with
upstream's ownership guard and with the causal-length change isolated; (3) CPU confirmation of 0.7.0's two-rank
tail-ring mechanism. Deferred: SwiGLU NLL study, router precision, workspace diagnostics (absent a concrete suspected
violation).
Gate before the GPQA arm: the column-511-only build in a full server, shipped graph mode, existing fixtures (no new
free-running pilot), covering tiled and fused routes, transitions between them as requests finish, the captured batch
sizes, mixed prefill/decode execution, valid unfilled slots and boundary ties. Confirm: the selected set before
truncation is preserved; the discarded pool is the lowest-scoring valid selected pool within the documented tolerance;
graph replay uses current scores and indices; no unintended patch or effective configuration change. Corrupted/NaN rows
reported separately (the correction does not repair the tail ring).
Completion arm (needs operator authorization of up to 13 additional GPU-hours; startup, warm-up, shutdown and invalid runs
count): 0.9.1 K3.25 with the column-511 correction only; three complete GPQA passes; original protocol, seeds,
concurrency, sampling and 327,680-token cap; shipped graph mode; tail, router and clamp changes excluded. Image, patch,
configuration and analysis frozen before generation; no stopping or extension on interim scores. Comparator: the existing
clean 0.9.1 three-pass record, with an explicit historical-control limitation, after verifying image ancestry and
installed dependencies, weights, rendered prompts, tokenizer, parser, effective server settings, sampling, admission
policy, warm-up procedure and GPU/software environment; execution differences recorded. A positive result supports
improvement relative to the recorded baseline only; a contemporaneous control is priced separately before any definitive
gap-explanation claim.
Analysis: primary = empty replies across all 594 question-passes, question-clustered comparison with the baseline, effect
estimate with uncertainty interval; exact test and direction fixed before data (a power statement must state its
alternative and clustering model; total counts alone do not determine a clustered p-value). Secondary: raw and stated
correctness across all requests (not answered-only), rescued and newly introduced empties, cap exhaustion vs premature
stopping, completion-length distribution, GPU-hours and throughput, kernel-route exposure where measurable without
changing execution. Reading rules: large reduction supported by the prespecified analysis = evidence the correction
contributes to better completion relative to the recorded baseline; moderate reduction without precision = inconclusive,
no automatic further pass; little improvement or deterioration = the localized correction stands but is not supported as
the dominant completion explanation. The 6 / 7-12 / 13 bands are operational thresholds, not causal boundaries. A
nonsignificant accuracy test does not establish no accuracy cost.
Separately: finish the tail fix; prepare minimal reproductions and patch evidence for both defects; any issue or PR is a
separate operator-authorized action, stating completion impact as untested.

**Amendment 11a (2026-10-11, operator, before gate data).** Conditional authorization: if the gate passes and the
comparator audit is clean, proceed with the three-pass, correction-only GPQA evaluation under a 13 GPU-hour ceiling; no
further small completion pilot. If the gate fails, the remaining verification allocation is spent resolving that failure
before the long run. The gate must establish: (1) the instrumentation observes actual shipped graph execution without
forcing an eager fallback; (2) the frozen patch covers every affected selection branch, preserving the selection before
truncation and discarding the weakest selected pool; (3) corrupted/NaN rows remain separately reported; the unresolved
tail-ring defect limits what a negative completion result could exclude; (4) the comparator audit finds no material
unexplained configuration difference. Analysis note: the exact sign-flip test against the historical record relies on an
exchangeability assumption; matching recipes and seeds does not make the comparison a randomized experiment, and the
write-up says so.

**Amendment 11b (2026-10-11, operator decisions and freeze, before any arm data).** Gate result: PASS on all four
amendment 11a conditions (graph-mode observation without eager fallback; every selection branch covered, the prefill
branch included after the gate found rows with exactly 512 candidates returned in index order; NaN-byte rows reported
separately; server-level configuration identical to the historical run). Comparator audit: no material unexplained
configuration difference, given the two conditions below. Operator decisions: ceiling raised to 18.5 occupied GPU-hours
(the comparator's three passes measured 16.59; the earlier 13 rested on an estimate that omitted pass 1's session);
the comparator's session split is mirrored (pass 1 on a fresh server, then passes 2-3 on a fresh server); a strong
signal is required: 7-12 empties with one-sided p < 0.05 reads as a modest, statistically detectable reduction and NOT
as support for the correction explaining the gap (only at most 6 empties with p < 0.05 reads as support); desktop
graphics load on GPU 1 accepted and logged. Frozen: image id sha256:047ea60d1b40bdd93782459c43889c5718bb102a51887dfc3cf8788421715678
(FROM release a9e985c0; changed file sparse_attn_indexer_kpool.py sha256
9104be37c8cb876a436f5fac11401c4abdb254529a6c37f3dca8121f41a5b61d; diff sha256
1ca405f24a439ff66366de2081bcbaafb027f11cf4d6e8c0ea36e02ff54e42dc); recipe glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix
(env.sh sha256 0ac18e584ceff1cc7bc6aac642923e7b792a9415589aa879ac0d909ca8a58b9d; only IMAGE differs from the comparator's
recipe); runner sha256 05024b67d6ff9ca1a89e3302991229c9d49f7024a90fa82fd245127e5df121bf; analysis spec sha256
523ec22806059f2eb384fa2225af2db4d8b3c64a3b3f4d94ebed8312bb12d610 and script sha256
b5b13ce2155085ca46fb24877f12b7cd0a5cdd16eb4cf19012958e7b86ae0033 (primary: one-sided exact sign-flip on per-question
differences in empty replies, comparator minus arm, over 198 questions; effect with a 20,000-replicate question bootstrap,
seed 20261011; exchangeability assumption and historical-control limitation stated). The arm starts after the tail-fix
verification runs release the GPUs.

**Amendment 11c (2026-10-11, coordinator erratum, before any arm result was examined).** The frozen analysis
specification's description of the treatment (section 1) cites the phase-3 decode-branch diff (`phase3/fix.diff`). The
frozen treatment is the image recorded in amendment 11b, whose correction also covers the prefill branch (`gate/candidate.diff`,
frozen sha256 1ca405f2…). The image id and diff hash in amendment 11b govern; the analysis itself (statistics, data, reading
rules) does not depend on this reference and is unchanged. The frozen file is not edited.

**Amendment 12 (2026-10-11; adopts the independent assessment "establish whether the old-engine advantage
reproduces").** The column-511 completion experiment is closed: the localized selection correction and its reproduction
evidence are kept; no further completion run is bought for this correction alone (cap exhaustions 15 vs 15, empties 15 vs
16, rescued and newly empty replies largely offset, no demonstrated accuracy benefit). CPU-only finishing work: complete
the 12 stated-answer adjudications; finalize the result and cost ledger; replace obsolete runtime estimates with
reconstructed occupied GPU-hours; correct two interpretation statements (the corrected 0.9.1 arm resembles the historical
rate but is not an unmodified-engine replication; greater corruption in 0.7.0 does not exclude an accidental completion
benefit from incorrect computation). Defect reproductions prepared for review; sending them upstream is a separate
operator decision (held: more data and further correctness fixes may be bundled). Proposed next GPU work, not authorized:
replicate stock 0.7.0's representative completion record (original image and effective settings incl. draft depth 5,
same weights and rendered prompts, original protocol, concurrency and cap, verified warm-up, documented session
lifecycle, three complete passes, the original per-pass seeds; described as repeatability under the recorded seeds, not a
fresh-seed estimate), with analysis and interpretation frozen before running, and a request for time only after the old
run's start-to-stop timings are reconstructed, the earlier estimate verified, a maximum budget stated (startup and
invalid runs included) and any discrepancy explained. Outcomes: substantially fewer empties again → replication support,
then the tail-mapping/DCP interaction on 0.7.0 becomes the concrete candidate (smallest valid correction, its separate ring
and seed-stride defects accounted for; corruption is never restored as a remedy); movement toward the newer engine's
count → the original gap is reported as not reproduced, without claiming equivalence, and no expensive bisection without a
reproducible target; ambiguous → report uncertainty and stop this allocation. Kept separate: correcting real defects;
repeatability of the engine difference; its mechanism; improving representative completion while preserving correctness.

**Amendment 13 (2026-10-11; adopts the independent assessment "verify remaining defects cheaply"; the operator agrees
verification comes before another baseline).** Objective: determine which remaining implementation omissions are exercised
and alter computation, and use that to choose the next behavioural experiment. The full 0.7.0 replication (amendment 12)
is deferred until this bounded verification is complete. Order: D3 router precision (verify the intended fp32 router path
on SM12x; identical captured inputs through the shipped and the corrected computation, not a cast of rounded logits;
report top-8 membership and routing-weight differences); D4 routed-expert SwiGLU clamp (confirm the patch reaches the
projection-mixed K3.25 path; check it against the intended clamp semantics; identical frozen inputs, unclamped vs limit 10;
how often the clamp engages and its output effect; a limit-1 smoke test establishes reachability only; no NLL campaign
yet); D5 workspace locking (observe resizing after graph capture on existing fixtures; if absent, report "not exercised in
this workload", not "ruled out"); D2/D6 close verification (reuse existing evidence; document the final patch and
stale-entry handling; add tests only for uncovered conditions). Efficiency: source checks and fixtures before any GPU
allocation; reuse captured inputs; combine compatible observe-only captures in one server session; isolated comparisons
offline; no in-place modification of captured execution without validated graph safety. A costed proposal of at most 2
occupied GPU-hours (startup and warm-up included) is requested before running; no GPU work is authorized now. If
substantial new instrumentation is needed, stop and report the engineering cost first. Deliverable per defect:
confirmed implementation omission; exercised in the tested workload; demonstrated computational effect; verified
correction; completion effect untested. Then one next experiment is selected (stock 0.7.0 reproduction, or a narrowly
defined corrected 0.9.1 candidate). No completion benefit is inferred from expert flips, activation differences or
likelihood changes alone; bundling fixes into one explanatory experiment loses individual attribution and must say so.
