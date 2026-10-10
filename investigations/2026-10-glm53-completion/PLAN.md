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

None yet.
