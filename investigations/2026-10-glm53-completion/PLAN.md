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

**Amendment 13a (2026-10-11; independent review of the amendment 13 proposal, before any stage-A data).** Recommended:
Stage A only, at most 2.0 occupied GPU-hours (startup, warm-up, smoke, offline work and invalid attempts included; the
runner's refusal to exceed the ceiling kept); Stage B and the stock 0.7.0 replication not approved (the 35 GPU-hour
maximum is not authorized; any later request uses reconstructed timings and handles invalid runs separately, with no
automatic large retry allocation). D3 controls: label captured logits by computation ("bf16-result logits cast to fp32"
vs logits from the intended fp32-output GEMM); verify actual dispatch and matching weights, that the offline shipped
computation reproduces the captured computation, that captured logits reproduce the routing actually consumed, and that
the corrected computation approaches the fp64 reference within tolerances stated before data; if an exact-reproduction
control fails, investigate batching, shape, dispatch and repeatability rather than relaxing it after seeing results;
report membership flips and routing-weight changes by layer and execution type, near-boundary margins, descriptive row
counts and uncertainty that respects shared request/history dependence (rows are not independent serving draws); FULL-graph
decode coverage is by source inspection only unless measured, stated explicitly. D4: stage A establishes omission and
reachability, not the real-weight effect; frozen inputs sufficient for the intended comparison are preserved with the
serving path that produced them documented. D5: resizing observed across capture, post-capture warm-up and serving; for any
event, establish whether it invalidates an address retained by a captured graph (a resize alone does not establish graph
corruption); no event = "not exercised in this workload". Stage B afterwards: not built automatically; first price the
smallest comparison using existing B12x/runtime facilities (unclamped vs limit 10; limit 1 as reachability control; limits
7 and 5 deferred); if a new harness is still needed, state what decision its output would change and report engineering
cost separately from GPU cost; element-level engagement and an EXL3 decoder deferred. Deliverable: an updated defect
matrix (measured vs inferred execution coverage, correction verification, computational effect, remaining completion
uncertainty, actual GPU and engineering costs) and the smallest next step that can change a concrete decision, naming one
next large experiment with its exact objective (stock 0.7.0 replication for repeatability, or a corrected 0.9.1 arm for a
specific fix's behavioural value); computational effects alone do not establish a completion benefit or explain the gap.

**Amendment 13b (2026-10-11, operator authorization).** Stage A authorized as recommended in amendment 13a: at most 2.0
occupied GPU-hours including startup, warm-up, smoke, offline work and invalid attempts, with the runner's budget gate.
Stage B and the stock 0.7.0 replication are not authorized. Stage A runs after the amendment 13a controls are built in and
CPU-tested.

**Amendment 14 (2026-10-12; adopts the independent assessment "verified correctness bundle, one-pass screen", after the
operator's input; recorded before any D3 precision-comparison result is computed).**
(1) D3: control (c) and the corrected comparison invoke the router as serving does (one call on the serving M rows, rows in
place, server alignment and router configuration), keeping the bitwise criterion and the stop rule; every record is
verified (the 60-record diagnostic is supporting evidence only); normalization and the location of the x2.5 scaling are
documented for the reference and serving computations; offline attempt 4 runs within stage A's existing ceiling; E1 is
recorded separately and no router batch-shape workaround enters the bundle without an independent justification.
(2) Minimal verified bundle on 0.9.1: D1 column-511 correction; D2 canonical positions/ownership/graph-safe mapping
correction; D3 fp32-output router correction if its verification passes; D4 the checkpoint's SwiGLU limit 10 on the
projection-mixed path after a minimal real-weight execution check (limit 1 may establish reachability; 10 is the
candidate); D5 only if its backport is straightforward and verified against the runner's actual workspace lifecycle,
otherwise left out and documented as unexercised in stage A; D6 not separately prioritised (unreachable under D2). No limit
tuning, alternative routing heuristics or unrelated performance changes. D1 stays despite its negative standalone result
because its interaction with corrected context contents is untested.
(3) Gate before the screen: each correction verified active; actual K3.25 weights and shipped graph execution; short
contexts, mixed batches and request reuse where relevant; no new execution, memory or stopping errors; throughput measured
to price the run; image, diff, hashes and effective configuration frozen. Prefer existing runtime fixtures over a one-day
standalone harness for D4.
(4) One representative GPQA pass, separately costed and authorized: all 198 questions, original sampling, parser,
concurrency and 327,680-token cap, a prespecified seed matching an existing baseline pass, normal stopping, the documented
server lifecycle and warm-up gate; budget from reconstructed timings and the candidate's measured throughput (a
baseline-sized pass is roughly 5-6 GPU-hours; identical runtime is not assumed); traces kept locally where permitted.
Primary: empty replies with cap exhaustion separated from premature stopping; rescued and newly empty replies. Secondary:
raw and adjudicated correctness on every request, completion lengths, occupied GPU-hours, execution validity, throughput.
Screening decision rules frozen before generation.
(5) After the screen: clear promising behaviour without an obvious correctness regression → confirm before attribution
(a prespecified additional seed/control comparison; contemporaneous stock control vs more representative passes chosen on
measured cost), then ablate by removing one component at a time from the bundle, accounting for dependent fixes and
interactions; similar or worse → deprioritise the bundle as the next completion remedy, start other source and
frozen-input investigations, no serial small completion pilots, and no conclusion that included corrections lack effects,
that interactions are absent, or that a new kernel or MoE defect exists; mixed (e.g. fewer empties, worse accuracy) →
report the tradeoff and uncertainty, not success. The screen supports a resource-allocation decision, not equivalence.
(6) A successful bundle can give a useful corrected engine without explaining the historical gap; the stock 0.7.0
replication stays relevant to that question and is deferred, not automatically funded; deeper kernel/MoE work targets
specific uncovered invariants or observed discrepancies.

**Amendment 14a (2026-10-12, operator authorization).** The operator authorizes running the bundle gate and the one-pass
GPQA screen as planned in amendment 14 without a separate proposal review. Ceilings set by the coordinator from measured
timings, before any bundle data: gate at most 1.5 occupied GPU-hours (the comparable gate session measured 0.70); screen
at most 7.5 occupied GPU-hours (pass-1 sessions measured 5.15 for the 0.9.1 record and 5.74 for the column-511 arm; about
30% margin for an unmeasured throughput change, startup, warm-up and shutdown). One invalid gate attempt may be repeated
within its ceiling; an invalid screen session stops the screen and is reported, with no automatic retry. Sequence: gate on
the candidate (D1+D2+D3+D4, plus D5 only if the bundle preparation justifies it); if the gate passes, the image, diff,
hashes, effective configuration and screening decision rules are frozen here before generation; then one pass. If the gate
fails, no screen runs.

**Amendment 14b (2026-10-12, coordinator, before any gate data).** The gate design returned by the bundle preparation
estimates 1.25-1.69 occupied GPU-hours for the two required sessions (s1: all corrections active, short contexts, mixed
batches, request reuse, errors, throughput lower bound; s3: the D4 limit-1 reachability check against limit 10), above the
1.5 set in amendment 14a without that design. The gate ceiling is set to 2.0; the optional production-image throughput
session (s2) is not run (the screen measures its own throughput and its 7.5 ceiling is enforced by a watchdog). D5 is left
out of the bundle (unexercised in stage A; the backport moves kernel warm-up before capture and would turn any unforeseen
workspace growth into an error); D6 is unreachable under D2. Candidate: D1+D2+D3+D4 (image id 08d1cb90…).

**Amendment 14c (2026-10-12; independent review of the bundle gate, adopted by the operator; before any screen data).**
Gate verdict: PASS for proceeding to the one-pass screen, subject to the checks below. The original gate failures are
retained: "D4 compiled kernels" (runtime-walk criterion) and "P1 probe repeatable" (long-context) failed as specified, for
the candidate and, for P1, also for the stock release; both criterion revisions were made after inspecting gate data and
the original criteria are not described as met. Accepted revisions: D4, session-specific compiled-kernel evidence in place
of an unreachable runtime attribute, provided the cache keys belong to the candidate's actual compiled plans and not to
pre-existing entries (combined evidence: serving plans carry limit 10; corresponding compiled kernels carry (True, 10);
the limit-1 intervention changes repeatable short-prompt outputs); P1, short-prompt repeatability, with stock-release
long-context non-repeatability documented as a limitation (no claim of long-context numerical equivalence, of no additional
bundle variability, or of a demonstrated near-tie mechanism). To resolve before launch from existing records: (1) classify
the three early stops in the gate's GPQA-shaped phase (valid answered stops → throughput unusable, readiness intact;
missing answers or parser errors → investigate first); (2) confirm completed D3 numerical verification of the fp32 path as
served (dispatch tier and non-bf16-valued outputs alone are insufficient); (3) confirm the production image's engine files
match the verified candidate with observers absent. No extra server session solely for throughput; the 7.5 GPU-hour
watchdog stays and actual cost is reported.
Screening rules (resource-allocation rules, not significance or equivalence claims): E = empty replies of 198. Clear
promising: E ≤ 2, no regression flag and no accuracy concern. Mixed: E ≤ 3 with a regression flag or an accuracy concern.
Similar / no clear separation: E = 3 without either. Similar or worse: E ≥ 4 (any accompanying regression reported
separately). Regression flags: raw correct ≤ 158, stated ≤ 159, or 2+ premature stops; accuracy concern: raw 159-163
(an otherwise promising result with raw 159-163 is mixed). Stated-answer adjudication completed before the final category.
Simulation reporting distinguishes the probability of meeting the empty-count threshold from the probability of the
complete clear-promising rule, which is labelled as such only if accuracy and stopping conditions are modelled.
Follow-up: clear promising → prespecified confirmation before ablation, stronger confirmation standard, historical-control
limitation reported; mixed → resolve the correctness/stopping concern before attribution work; similar or worse →
deprioritise the bundle as a completion remedy within this allocation, start other source or controlled-input work, no
claim that every included fix lacks behavioural effects. An invalid or watchdog-truncated run is not a completed one-pass
result; no automatic retry.

**Amendment 14d (2026-10-12, coordinator; pre-launch items resolved and screen frozen, before generation).** (1) The
GPQA-shaped gate phase had four early stops, not three (docs 16, 19, 21, 22: 726-4,096 tokens): each has non-empty final
content and reasoning, no request or HTTP error; the gate client stored no reply text, so "answered" means non-empty final
content (the answer parser was not run); these questions stop at similar lengths, correctly, in the 0.9.1 record and the
column-511 arm. Readiness intact; that phase's throughput is unusable. (2) D3 as served: on 1,000 captured records (69,344
rows, both ranks), fp32-output dispatch on all layers; offline torch.mm(x, W.T, out_dtype=fp32) at the serving batch size
reproduces the served logits bitwise (1,000/1,000); 0 logits outside τ_d of fp64 (max error/S 6.8e-7), 0 top-8
disagreements; routing at serving batch size reproduces the consumed routing bitwise (69,344/69,344); 0.0075 GPU-hours.
(3) The production image's five engine files equal the frozen candidate files, with no observer code; the check image
differs only by additions, inert unless enabled. (4) The 66 kernel cache entries with (True, 10.0) were written during s1
by the candidate's plans (inside s1's window; none with limit 10.0 existed before; same shapes as release keys apart from
the limit). Gate total 1.6403 of 2.0 GPU-hours. Frozen: image sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548;
combined diff 949b2301c906e0d499c17978b5adf4bd8a0ce03ed01011073c51bdb99e6ddad8; recipe
glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234 (env.sh 3125683b01ba74a2d41745d8000fe40918c031511d8d43e098c08f8e845b15c5;
only IMAGE differs); screen_analysis.py 9c799f260ddcc54739fea3864884f55f33cf0e2cef1fcab32758d6a4e59106cb (amendment 14c
categories; PROVISIONAL until stated adjudication); runner gpqa-bundle-screen.sh
e8caf211698071d7bbbd51d8d5c00a7015ec1bca26106238b55dd70a574c6746 (one pass, seed 1234, 7.5 GPU-hour watchdog, no retry).
Power reported as P(E ≤ 2) only (0.078 no change; 0.85-0.86 0.7.0-like; 0.50 for 50% fewer); accuracy and stopping not
modelled.

**Amendment 14e (2026-10-12; screen result recorded after the frozen analysis, and operator note).** The one-pass screen
(seed 1234, all 198 questions, original protocol, frozen image and analysis) is valid: empty replies 3 of 198, all at the
cap, no premature stop; no regression flag and no accuracy concern; stated-answer adjudication complete (2 rows, neither
hedged). Category under amendment 14c: SIMILAR / NO CLEAR SEPARATION (E = 3). Against pass 1 of the 0.9.1 record: 3
rescued, 1 newly empty; against pass 1 of the column-511 arm: 1 rescued, 0 newly empty; 0.7.0 pass 1 (context): 0 empty.
Secondary (single pass, not a decision input): raw correct 174 (record pass 1 169, arm pass 1 164, 0.7.0 pass 1 174);
stated 176. Cost 4.84 occupied GPU-hours (ceiling 7.5). Per amendment 14c the bundle is deprioritised as a completion
remedy within this allocation; this is not a claim that the included fixes lack behavioural effects.
Operator note (not an authorization): the operator considers another pass warranted, because seed 1234 has shown
seed-luck effects across the many GPQA runs performed, and will convey this to the independent assessor. No further GPU
work is authorized by this note.

**Amendment 15 (2026-10-12; independent assessment "one additional bundle pass for seed sensitivity"; proposal pending a
separate operator budget decision; recorded before any seed-1235 bundle data).** The seed-1234 screen stays classified
SIMILAR / NO CLEAR SEPARATION; nothing is reclassified. Proposed: the unchanged frozen bundle (image 08d1cb90…) on all 198
questions with seed 1235 (the next protocol seed, chosen by protocol order, not by scores); primary comparator the stock
0.9.1 record's seed-1235 pass; secondary the column-511 arm's seed-1235 pass; original sampling, parser, concurrency and
cap; shipped graph mode; fresh server and verified warm-up; matching the comparator's session conditions (documented);
traces and complete logs kept; stated adjudication before interpretation. Budget: one pass, about 5-6 occupied GPU-hours,
7.5 watchdog, startup/warm-up/shutdown included, no automatic retry or third pass. A new-seed stock-plus-bundle comparison
(two passes) is deferred as a possible confirmation step. Analysis frozen before generation: primary empties (cap
exhaustion vs premature stop) and rescued/newly empty against the matching-seed baseline; secondary raw and stated
correctness on all requests, completion lengths, GPU-hours and throughput; accuracy-concern and regression thresholds stated
relative to the matching baseline or an absolute standard, with the choice explained (seed-1234 count thresholds not reused
silently); the new pass reported separately; both bundle passes against their matching stock passes, question-clustered,
labelled exploratory (the follow-up was chosen after inspecting the first screen). Decision rules: at most 2 empties with no
accuracy/stopping concern → promising follow-up signal, stronger confirmation proposed before ablation or any
completion-benefit claim; modest improvement without clear separation → report direction and uncertainty, no automatic pass;
similar or worse, or a material accuracy concern → deprioritise the bundle as a completion remedy for now and continue
targeted source/frozen-input work without another serial screen. No outcome attributes effects to an individual correction.
Limits: does not establish that seed 1234 was biased, prove a general improvement, explain the 0.7.0 advantage, or
establish equivalence if negative; any confirmation prefers a contemporaneous control on prespecified seeds.

**Amendment 15a (2026-10-12, operator authorization).** One bundle pass at seed 1235 is authorized as proposed in
amendment 15: about 5-6 occupied GPU-hours, ceiling 7.5 (watchdog), startup, warm-up and shutdown included, no automatic
retry and no third pass. It launches after the runner and the analysis are prepared, checked and their hashes recorded here.

**Amendment 15b (2026-10-12, coordinator; freeze before seed-1235 generation).** Session conditions: the bundle pass runs
as pass 2 on a fresh server (the eval chain skips the existing pass-1 directory), request seed 1235, lm-eval seed 1236,
task hash 87e807b3b27c, concurrency 8, timeout 10,800, 3 retries, 327,680 cap — the same as the stock record's and the
column-511 arm's seed-1235 passes (each pass 2, first on a fresh server); documented differences: the comparators continued
to pass 3 (after pass 2 ended), repository commit, and the record pass 1's implicit seed. Primary comparator: record pass 2
(5 empties, all cap; raw 176; stated 178); secondary: arm pass 2 (5; 170; 176); context: 0.7.0 pass 2 (3; 175; 178).
Accuracy thresholds (deciding): a stated absolute standard from the six stock-family passes (record and arm passes 1-3;
raw mean 170.2, SD 3.9; stated mean 172.8, SD 4.6): material concern raw ≤ 158 or stated ≤ 159 (mean − 3 SD) or 2+ premature
stops; accuracy concern raw 159-163 (below every family pass). Offsets from the single matched baseline are not deciding
because record pass 2 is the family's best pass (176 raw) and would place a typical pass in the concern band about half
the time with no accuracy change (the dry run on the arm's pass 2 shows this); the offset readout (flag raw ≤ 165 or stated
≤ 167; concern raw 166-170) is reported alongside. Categories: SIMILAR OR WORSE for a material concern or E ≥ 5; PROMISING
FOLLOW-UP SIGNAL for E ≤ 2 without an accuracy concern; MODEST IMPROVEMENT, NO CLEAR SEPARATION for E = 3-4 or E ≤ 2 with a
concern. Combined analysis of both bundle passes against the matching stock passes, question-clustered, labelled
EXPLORATORY. Frozen: runner gpqa-bundle-screen-s1235.sh sha256 39ff7616e385546c35939eed2c914b7fc8478145347314ee6f4a014391b2c0f7;
analysis screen2_analysis.py sha256 cc3d300032cd831deea233c7e2251477b807170cbee4f80a4bd611efafef11c7; image and recipe as in
amendment 14d. Budget estimate 5.0-6.0 occupied GPU-hours; ceiling 7.5.

**Amendment 15c (2026-10-12; seed-1235 result recorded after the frozen analysis).** The bundle's seed-1235 pass (pass 2,
fresh server, frozen image, runner and analysis) is valid; pass 1 unchanged. Empty replies 3 of 198, all at the cap; no
material or accuracy concern under the family standard (raw 168, stated 169 after adjudication of 5 rows, one hedged);
the matched-offset readout (not deciding) shows "concern". Category: MODEST IMPROVEMENT, NO CLEAR SEPARATION (E = 3).
Against record pass 2 (seed 1235, 5 empties): 2 rescued, 0 newly empty; raw −8, stated −9. Against arm pass 2: 5 rescued,
3 newly empty. 0.7.0 pass 2 (context): 3 empties. EXPLORATORY combined (both bundle passes vs the matching stock passes,
396 question-passes per side): empties 6 vs 10, reduction 1.01 points (95% question bootstrap 0.00 to 2.27), one-sided
sign-flip p = 0.11; raw −0.76 points (−3.79 to +2.27). Cost 4.54 occupied GPU-hours. Per amendment 15: report direction and
uncertainty; no automatic additional pass.

**Amendment 16 (2026-10-12; adopts the independent assessment "targeted invariants from the correctness-bundle
checkpoint").** Checkpoint B0 is frozen: the D1-D4 correctness bundle as gated and screened — image
sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548, combined diff 949b2301c906e0d499c17978b5adf4bd8a0ce03ed01011073c51bdb99e6ddad8,
K3.25 weights snapshot 0490d2f7, recipe glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234 (env.sh 3125683b…; only IMAGE
differs from the stock 0.9.1 recipe), effective configuration as fingerprinted in the gate and both screens (non-default
args 7025a968…, FULL_AND_PIECEWISE, DCP1, owners split:25, NOPE record, DFlash2 x3, KV pool 4,707,515); its gates and two
GPQA passes retained unchanged. Future interventions are B0 plus one justified change; verified corrections are not removed
to recreate stock behaviour; stock releases remain historical references, and explaining their completion difference is
separate from improving B0. No ablation of the weak bundle completion signal.
First priority: TP2 all-reduce correctness. CPU preparation confirms the transport B0 actually dispatches under shipped
graph execution, its supported message sizes, dtypes and graph-padded batch sizes, the current initialization/replay API,
and a validated reference for the exact reduction semantics; obsolete test interfaces are repaired; the fixture exercises
the real implementation. GPU fixture on two GPUs without the model: small and boundary message sizes, changing and
distinguishable inputs every replay, repeated captured-graph execution, supported synchronization and buffer-reuse
patterns, both ranks' outputs; comparison with the explicit reference and, where useful, the alternative validated
transport; exact comparison where the arithmetic warrants, otherwise a tolerance frozen before data; arithmetic
differences distinguished from stale values, cross-step contamination, wrong-rank data and synchronization/addressing
errors. A costed proposal precedes any run; an initial ceiling of 0.5 occupied GPU-hours is proposed only if compilation
and the fixture fit (initialization and cleanup included). No serving A/B follows automatically. Failure → minimize and
reproduce, verify the smallest correction or a validated transport substitution, create B1 from B0, then propose an
end-to-end test. Clean → retire the tested failure modes without claiming all communication paths or numerical effects are
excluded.
Second priority: cache isolation and high-offset addressing, model-free, with the bundle's actual cache geometry and
kernels (known records and sentinel regions; MLA, draft and recurrent-state write isolation; actual padded strides;
high-address blocks; slot reuse and applicable graph replay; reads against a validated logical-record reference; selected
addresses and allocation patterns shown reachable in serving); priced separately; no long GPQA replies to exercise an
addressing invariant.
Parallel CPU work: E1 from existing captures (absolute as well as relative routing-weight differences, normalization and
scaling semantics, expected numerical error per batch-size path; no workaround merely for lower accuracy against fp64);
prepare the cheapest KV-writer/query-projection comparison for a later stage; the mixed-quant MoE reference harness is
deferred unless evidence points there.
Gate for another behavioural evaluation: a new verified intervention (B0 vs B1, other corrections fixed, the changed
mechanism stated, completion, adjusted correctness and compute cost prespecified) or a clearly priced confirmation
objective. The adjudicated scoring method is preserved. Deliverable per item: tested invariant and execution coverage,
reference and tolerance, reproducibility, any localized failure and correction, GPU and engineering cost, what it rules
down, and the next smallest discriminating test.

**Amendment 16a (2026-10-12, operator authorization).** Authorized: (1) the B0 TP2 all-reduce fixture as proposed
(b0-allreduce/PROPOSAL.md; scripts frozen in its SHA256SUMS): smoke then full, ceiling 0.5 occupied GPU-hours including
initialization, compile and cleanup; reference exact bitwise equality with (a.float()+b.float()).to(bf16), identical on both
ranks; PyNCCL cross-check arm; verdicts INCOMPLETE / INVALID / FAIL / CLEAN fixed in analyze.py; a fixture-code bug in the
smoke may be fixed and the smoke repeated within the ceiling; no serving A/B follows automatically. (2) The KV-writer /
query-projection comparison (e1-kvwriter/KVWRITER-PREP.md; predictions and criteria frozen there): one GPU per image
(0.7.0 and B0), model-free, ceiling 0.25 occupied GPU-hours; it runs after the all-reduce fixture has released the GPUs.

**Amendment 16b (2026-10-12; results recorded after the frozen analyses, and operator decision).** (1) B0 TP2 all-reduce
fixture: CLEAN. 2,040,800 B12x peer-push all-reduces over 26,650 graph forwards (FULL and piecewise; every engine size and
geometry edge; slot and address reuse; rank skew; prefill interleaving) plus 900 PyNCCL reference forwards; the B12x epoch
counter advanced by exactly the scheduled calls in every phase; 0 mismatched elements in all 41 cases on both ranks against
the exact bf16 reference; no hangs. Cost 0.07 GPU-hours (smoke 0.0139, full 0.0561). The tested transport failure modes are
retired; contention with model kernels, the drafter's exact sequence, epoch rollover, other communication paths and
numerics elsewhere are not excluded. No B1. (2) KV-writer / query-projection comparison (one GPU per image, B0 and stock
0.7.0; 0.0069 GPU-hours): before GPU data, a CPU dry run on defect-free outputs of the frozen rounding model found three
criterion errors (P2 bound half the correct half-ulp; W4 tolerance inconsistent with the frozen error model; W3 rate
prediction omitting exact E4M3 ties), recorded as errata in the freeze manifest before the run. The runner's analysis step
failed to load the result files (torch weights-only loading rejects a version object written by the investigation's own
script); the frozen analysis was re-run unchanged through a loader wrapper that allows that one type, before any analysis
output existed; frozen file hashes verified. Verdicts: "with errata" not beyond expected rounding (no failures); "frozen as
written" beyond expected rounding, with failures that reproduce exactly the dry run's defect-free counts (W3 rates
6.48e-4-1.12e-3; P2 counts 5,691-104,184, identical across images and between bmm and H64). Operator decision: the
"with errata" verdict governs. Result: B0's 528-byte record writer and 0.7.0's 656-byte writer agree within rounding
(differences only at exact E4M3 ties, one code); the query projections are equivalent on the shared lanes; the record and
projection paths move from "floating-point only, unverified" to "measured equal within rounding" and are ruled down as
0.7.0-vs-B0 mechanisms. No B1. (3) E1 (CPU): expected numerical variation of the upstream router warp kernel (unbiased
score rebuilt as (s+b)-b in fp32 with biases 5.0-14.5); selection unaffected; same kernel binary in every release; no
workaround.

**Amendment 16c (2026-10-12, operator authorization).** The B0 cache isolation and high-offset fixture
(b0-cache/PROPOSAL.md; fixture hashes in b0-cache/fixture.sha256; criteria fixed in the fixture code before data) is
authorized: one GPU, ceiling 0.5 occupied GPU-hours including startup, compile and cleanup, no automatic retry.

**Amendment 16d (2026-10-12, coordinator; operational fix before any cache-fixture data).** The first launch was refused by
the runner's own pre-flight (no container started, 0 GPU-hours): its GPU-idle check counted the desktop's gnome-shell
graphics client (type G, 8 MiB, 0% utilization) on GPU 0. The check now ignores type-G desktop graphics clients
(gnome-shell, Xwayland, xdg-desktop-portal) only; the memory (1 GiB) and utilization (5%) limits and the refusal for any
compute process or other client are unchanged, matching the other fixture runners. Runner sha256 0cb18df9… → d6fac94f270581276b1314e48967d66f91e6d6859c457ff275521a30bb3e8f70. No
criterion, kernel call or fixture code changed.

**Amendment 16e (2026-10-12; cache fixture results, recorded after the runs).** Run 2 (0.0278 GPU-hours): 15 parts passed
byte-exact against the shadow image with sentinels intact (draft writer eager and graph replay, FA2 reader, five KDA
kernels and page copy-on-write, MLA writer, kpool seed and compress, zeroer, cross-group block reuse); four failures, each
diagnosed from stored outputs and source as a fixture fault or an obsolete upstream test, none high-address specific:
D2 requested an extend width (64) the GLM_NEXT extend kernel rejects (supported widths 512-2112); D3 compared the decode
binding's own scratch output buffer with itself, then (rerun 1) the fixture's writer replays overwrote its reader sets;
E3 compared unwritten sentinel ring slots (positions 12-15 write slots 4-7 only); the original B12x GLM_NEXT high-page test
uses the obsolete writer signature (the repaired copy passed). Fixes made after inspecting the data (fixture code only; no
criterion or tolerance loosened; hashes in b0-cache/fixture-post-run2.sha256; runner gained partial-part and per-run
ceiling options). Reruns 1 and 2 (0.0056 + 0.0058 GPU-hours): D1, D2 (decode and extend at the serving width 2048, plus
decode at 64), D3 (graph replay equals the eager read of the same selection; the control follows the new selection) and E3
pass; 80 checks, 0 failed. Total 0.0392 of 0.5 GPU-hours. Ruled down for the driven kernels: 32-bit truncation at blocks
934-1274 and the record straddling 2^31, writes outside a group's page or into padding, padded-stride errors, stale data
across block reuse, and replayed graphs using capture-time addresses (D8/D9 retired for these kernels). Not ruled down:
block-table and slot-mapping construction under real scheduling, the index-K readers, extend under graph replay, the mamba
postprocess kernel, numerics, rank 1. No B1.

**Amendment 16f (2026-10-12, coordinator erratum to 16b and 16e, before publication).** 16b's "differences only at exact
E4M3 ties, one code" overstated the KV-writer result: measured, 90-99% of cross-writer byte differences fall at exact ties,
plus 11-88 non-tie differences per set (rates 9e-6 to 6.5e-5), all one code and inside the frozen reciprocal bracket, as the
errata rule allows; the verdict is unchanged. 16e's count of the cache fixture's run-2 failures as "fixture faults or an
obsolete upstream test" stands, but in total five fixture faults were found (three in run 2, two more exposed by rerun 1)
plus one obsolete upstream test.

**Amendment 17 (2026-10-12; adopts the independent assessment "live scheduling and consumed slot mappings on B0", with the
operator's agreement).** Objective: test whether real scheduling supplies correct MLA and index-K mappings on B0. Kernel
addressing tests that already passed are not repeated; B0 is not changed; no GPQA screen. Preparation (CPU) before any GPU
allocation: extend the existing D2 guard where practical; define an independent expected mapping from request identity and
logical token position, current allocation ownership and KV group, committed vs speculative token state, and rank
ownership with the actual cache geometry — never a reference copied from the same mapping builder; CPU-test with injected
wrong-request and stale block ids, wrong logical positions, row-reuse errors and incorrect rank/group ownership, and with
valid padding, masked rows and speculative slots that must not be flagged. Serving coverage: one short session of the
unchanged B0 computation, both ranks, shipped graphs, existing active-reasoning prefixes with bounded continuations,
exercising prefill/decode interleaving, request completion and row reuse, partial prefix-cache hits where applicable,
speculative acceptance/rejection, fused/tiled transitions and actually reachable high-address blocks; the device mappings
consumed during graph replay are checked, not only host-side metadata prepared before capture; minimum coverage frozen
before running (missing required coverage = incomplete for that condition, not clean). Invariants per driven access: the
physical block belongs to the intended request/group; the logical position maps to the intended record; inactive rows and
masked accesses do not become live; replayed graphs consume current mappings; permitted speculative accesses stay separate
from committed-state claims. On a violation the smallest useful snapshot is kept (request/row, phase, rank, position, block
ownership, mapping). Instrumentation: verified not to alter model outputs or force eager fallback; overhead reported; a
clean run cannot exclude every contention-dependent race. Provisioning: a concrete proposal (expected ~0.6 occupied
GPU-hours from existing sessions; explicit maximum, startup and invalid-run allowance; engineering estimate and reused
components), preferring extension of the existing guard over a general tracing framework; no GPU work authorized now.
Outcomes: violation → minimize and reproduce, verify a correction, B1 = B0 + that correction only, then propose a
behavioural comparison; clean with required coverage → retire the tested scheduling/mapping failures, no automatic
completion pass, next prepare the smallest mixed-quant MoE fidelity fixture (identical inputs, routing and quantized
weights); incomplete → identify the exact missing exposure and price only that.

**Amendment 17a (2026-10-12, operator authorization "proceed as described").** The slot-mapping guard session is
authorized within the described envelope: one short B0 serving session (both ranks, shipped graphs), expected about 0.6
occupied GPU-hours, hard ceiling 1.0 including startup, warm-up, shutdown and one invalid-run allowance; it runs only after
the guard, the independent expected mapping, the CPU fault-injection tests, the frozen coverage minimums and the
instrumentation checks are prepared and pass. If the prepared proposal exceeds this envelope, needs substantial new
instrumentation, or cannot meet its coverage minimums by design, it returns to the operator before any GPU use.

**Amendment 17b (2026-10-12, coordinator; before any guard GPU data).** Preparation complete within the 17a envelope
(expected 0.55-0.63 occupied GPU-hours; budget gate 1.0 including one invalid-run allowance; 31/31 CPU tests; image
sha256:1f55fe65… = B0 plus the guard module and hook lines in three files; all files in b0-mapguard/SHA256SUMS). Independent
expected mapping: a worker-side ledger rebuilt from the SchedulerOutput stream only, cross-checked offline against digests
read directly from the KV cache manager in the engine-core process; eight graph-safe device check sites at the point of
consumption. Coverage minimums frozen for 28 conditions; copy-on-write prefix paths are unreachable here (all prefix hits are
block-aligned) and listed as not covered. Finding before GPU data (CPU, B0's real kernels under the Triton interpreter): the
index-K prefill writer (_kpool_compress_insert, sparse_attn_indexer_kpool.py:102-147) pools tokens t-3..t and assumes
pool-aligned chunk starts, while the scheduler does not align prefill chunks to 4 tokens (its mamba split aligns to 4,352-token
blocks; source reading). For an unaligned chunk start, (a) the first prefill request's pool completing in its first three
tokens is never written, and (b) a later request in the same step has its first pool compressed from the previous request's
tokens (stored record byte-equal to that cross-request pooling). The allocator simulation of this workload predicts 73
unaligned chunk starts (inference). The guard counts both cases; VIOLATION is pre-registered for this class whatever the
other checks show; no selection or completion effect is claimed. Candidate defect D7, pending confirmation in serving.

**Amendment 17c (2026-10-13, coordinator; operational fix before any guard GPU data).** The first launch was refused by the
runner's own pre-flight (no container started, 0 GPU-hours): it allowed only shell graphics clients, while GPU 1 carried the
desktop's Firefox and snap-store graphics clients and the Claude Desktop GPU process (the same desktop load present in every
earlier B0 session and accepted in amendment 11b, and ignored by the other runners). The check now also ignores those three
desktop clients; any other process, and the memory and utilization limits, still refuse. Runner sha256 59af5d51… → 4dd40ed67e02e605dc89c80ac0825c52fcdd6b99edcaae2364b24f438ee3198f. No
guard, workload, coverage minimum, invariant or analysis changed.

**Amendment 17d (2026-10-13; guard attempt-2 diagnosis, recorded after inspection).** Attempt 2 (0.4211 GPU-hours) never
ran its client: the runner's start-up abort matched the key "guard_errors": 0 in every readout (pattern fault). The guard
recorded only the stack's release warm-up traffic (84 requests, about 570 steps per rank, shipped graphs) and the runner's
six fake warm-up steps. Classified after inspection: idx_r_dec/idx_r_pf "rtable" flags (86,736 and 84) are a guard
expectation fault (the reader consumes index-page ids 17·block+j from a 256-token builder block size, indexer.py
1089-1096 and 1268-1290; the guard expected block ids), so the index-K read sites are void (INCOMPLETE) for attempt 2 —
stored records re-checked against the corrected expectation are consistent; the in-graph count off-by-one, the missing
baseline and the position/attribution flags are analysis or runner-warm-up artefacts. Engine finding: D7 confirmed in
serving — every long-prompt prefill chunk is 2,045 tokens (input budget 2,048 minus 3 draft slots, scheduler.py:563), so
chunk starts fall at 2,045 and 4,090 tokens into each 4,352-token block, and index-K pools 2044-2047 and 4088-4091 of every
block of a long prompt are never written (10 events per rank, every unaligned first-row start; none at aligned starts;
pool_cross 0 because only one request prefilled per step); a later prefill read consumed an unwritten pool. Scope
(inference): GPQA prompts are mostly a single chunk; effect on selection or completion not measured. Post-inspection fixes
(labelled P1-P6; guard read unit, exact baseline, device-executed counts, scheduler-side completions, window rule; runner
pattern and timing options; frozen criteria unchanged) with 28/28 + 4/4 CPU tests and an attempt-2 regression test; rebuilt
guard image sha256:e48738c6…; hashes in b0-mapguard/SHA256SUMS-postA2. Corrected re-analysis of attempt 2: VIOLATION
(D7 only), no INVALID; INCOMPLETE: mixed steps, prefix hits, 5-8 in flight, route transitions, high-address blocks (ids
reached only 528), index-K reads.

**Amendment 17e (2026-10-13; before data).** Operator authorization: rerun with the full frozen workload; data quality
and isolation take priority over GPU time. Total ceiling for the guard item raised from 1.0 to 1.45 GPU-hours
(attempt 2 spent 0.4211). Attempt 3 runs the full phase list (P0 warm T0 F0 T1 F1 D S Q P1) with the wall cap raised
from 1,440 s to 1,800 s (at most 1.0 GPU-hour). From earlier B0 sessions, readiness takes about 780-800 s and the
workload about 300 s, so the larger cap only adds margin; the expected spend is about 0.65 GPU-hours. Zero-GPU checks
before launch: file hashes match SHA256SUMS-postA2; the image id matches images.txt (e48738c6…); the post-a2 CPU tests
pass 28/28, and the a2 regression test passes on rerun (a logged rc=1 did not reproduce); the start-up abort pattern
matches only guard-error records; and the client completes every phase against a local mock server, with distinct
seeds for all 64 sampled requests and no request errors. If attempt 3 is invalid, there is no automatic retry: it is
diagnosed and returned to the operator. Frozen criteria and coverage minimums are unchanged.

**Amendment 18 (2026-10-13; adopts the independent review "D7 — verified B1 and exposure before behavioural spending",
relayed by the operator; before any D7 data).** Proceed with CPU discovery, reproduction and correction preparation.

Step 0, mechanism and exposure:
- **Release comparison.** For each release, test the actual writer and the relevant scheduling and DCP semantics. Report
  separately: whether the faulty assumption exists; whether execution can reach an unaligned start; and how the affected
  pools differ by request order, chunking and rank. A shared source defect does not exclude an engine difference.
- **Upstream comparison.** Prefer a verified upstream correction where one applies. Check its assumptions against B0's
  chunking, prefix caching, tail-ring state and speculative execution; upstream code is evidence, not proof of
  compatibility with this fork.
- **GPQA exposure.**
  - Use exact rendered prompt-token counts, not character lengths.
  - First explain the simulator's 73 predicted versus 217 measured events in attempt 3.
  - If recorded admission times and prompt lengths cannot uniquely reconstruct per-step budgets, label exposure as
    estimated, give bounds or sensitivity across plausible schedules, and separate known, possible and unknown exposure.
  - Exposure-versus-empty associations stay descriptive. Doc 127 is not taken to bound the possible effect until
    shared-step exposure has been assessed.

Step 1, the smallest correct intervention:
- **Writer-side repair is preferred** if the preceding 1-3 keys and gates are available with verified ownership and
  positions.
- **Its tests cover:** every chunk-start residue modulo 4; first and later requests in a packed batch; short chunks and
  successive chunk boundaries; block boundaries, prefix hits and applicable resume paths; pooled bytes against an
  independent reference; aligned inputs unchanged.
- **If scheduler alignment is chosen:** define alignment by logical pool position; verify progress when fewer than four
  budget tokens remain; cover shared-step splits and final chunks; document that it prevents exposure rather than making
  the writer correct for arbitrary unaligned inputs.
- B1 = B0 plus the chosen D7 intervention only.

Step 2, serving verification:
- **Recommended:** conditional authorization at a ceiling of 1.0 occupied GPU-hours, including startup, warm-up and
  invalid attempts. It runs only after the correction and the existing regression tests pass, and only once the operator
  has authorized it.
- **Conditions:** the frozen guard workload on both ranks with shipped graphs; all applicable coverage minimums; zero
  missing or foreign pools; correct ownership and logical position; D1-D4 gates unchanged; short single-chunk probes
  unchanged under matching conditions.
- **Writer-side repair:** unaligned starts must still be exercised; a clean run without exposure would not verify it.
- **Scheduler prevention:** demonstrate that the prohibited starts are prevented, and do not describe it as writer
  correctness on unaligned inputs.
- Throughput and instrumentation overhead are recorded. Lightweight exposure counters may be collected during step 2 if
  that is safe, never at the expense of graph-mode verification.

Step 3, localized computational effect, is a separate costed decision:
- identical queries, head weights and token histories;
- index-K states that differ only in the affected pool, with a reference for the correctly pooled keys;
- affected-pool inclusion on B0 and B1, final attended pool and token sets after the D1 truncation and tail handling,
  prefill as well as decode selection, and repeated baseline calls to separate top-k variability and boundary ties;
- selected-pool rank alone does not establish output impact.

Outcomes:
- **Verified correction:** B1 becomes the correctness checkpoint; completion impact stays untested.
- **Material computational effect with credible GPQA exposure:** propose a targeted B0/B1 behavioural comparison with
  long-tail requests costed explicitly and cases, seeds and outcomes frozen before running.
- **Narrow or uncertain exposure:** keep the correctness improvement; no full GPQA pass merely because the defect is real.
- **Shared defect across releases:** investigate differential exposure before deciding its relevance to the gap.

No behavioural evaluation follows automatically.

**Amendment 18a (2026-10-13; adopts the independent review "D7 Step 2 — recommended conditional authorization", relayed by
the operator; before any B1 data).** Steps 0-1 are complete (CPU only).

Step 0-1 results:
- **Writer:** the D7 writer is byte-identical in 0.7.0, 0.8.0 and 0.9.1, and every served configuration can reach an
  unaligned start. Exposure patterns differ by draft slots, block size and DCP.
- **Upstream:** vLLM main keeps the assumption; SGLang repairs it writer-side from the tail ring.
- **Simulator:** with the draft-slot reservation added, it reproduces attempt 3's 217 events exactly (given the recorded
  step timing).
- **GPQA exposure:** none in the 0.9.1 family under the reconstruction (doc 127 split at 1,996 with 7 decode rows), and
  doc 127 exposed in 0.7.0 passes 2-3, where it finished.
- **Correction:** writer-side `d7-fix.diff`, 28/28 CPU fix tests, regressions pass.

Correction to an earlier statement: stock 0.9.1's GPQA server logs exist (pass 1 and passes 2-3) and show a 2,048-token
budget and 3 draft tokens; "no server log found" was wrong. Stock 0.7.0 passes 2-3 show 2,048 and 5.

Authorized for preparation: 2a (guard amendment for B1's write rule) and 2b (B1 = B0 + D7 only, plus a guard image).
Recommended: 2c with the appended phase 2d, maximum 1.0 occupied GPU-hours including startup, warm-up, verification, the
appended phase and shutdown, no automatic retry, launched only after the CPU tests pass and hashes are frozen, and only
with the operator's go-ahead. No Step 3 or behavioural run.

Requirements before freezing:
- **Independent content oracle.** Straddling-pool references are built from request keys and gates tracked
  independently by logical position, not from the ring members B1 gathers.
- **Fault injection** must detect missing repaired pools, foreign keys, stale keys from the same request and incorrect
  logical positions.
- **Byte oracle on GPU-faithful conversion and compression semantics.** The Triton interpreter's conversions are not the
  byte oracle.
- **No-ring fallback.** Establish whether unaligned prefills can reach it under B1's configuration. If unreachable,
  document and assert it where practical; if reachable, repair it or scope it out before claiming arbitrary
  unaligned-input correctness. External-token and KV-connector paths stay outside the claim.

Serving criteria: as proposed — both ranks, shipped graphs, existing coverage minimums, at least 100 exercised
straddling-pool events, zero missing, foreign or byte-incorrect pools, all other mapping checks clean, D1-D4 gates
preserved, and short single-chunk probes unchanged. No long-prompt transcript parity is expected.

Phase 2d: for each case, record the requested background concurrency, the actual scheduled decode count D at the
relevant prefill step, the token budget and reservations, and the actual chunk start and length. Evaluate 2,045 − 7·D
against the actual D; a difference between the requested and actual D is reported as such, not as an arithmetic
contradiction. This validates the scheduling arithmetic under measured conditions and does not measure historical GPQA
exposure. Exposure reporting keeps measured chunking arithmetic, the reconstructed historical decode count and the
exposure inferred from those two separate; the reconstruction is not promoted to certainty.

Outcomes:
- **CLEAN with the required exposure:** the D7 repair is verified for the covered serving configuration; B1 is frozen as
  B0 + D7 only; B0 and its records are preserved; completion impact stays untested and no GPQA pass follows.
- **Violation:** the failing inputs are preserved and minimised; no verified B1.
- **INCOMPLETE:** the missing condition is identified and a narrowly costed follow-up is returned.
- **If exposure stays narrow and opposite to the gap:** D7's behavioural investigation is closed for now; the correctness
  repair and the upstream reproduction are kept; work moves to another specific untested invariant.

**Amendment 18b (2026-10-13; operator authorization, before any B1 data).** Step 2c with the appended phase 2d is
authorized up to **2.0 occupied GPU-hours in total**, with **at most one retry** (two attempts in all).
- **Precondition:** launch only if the 2a/2b CPU tests verifiably pass and the hashes are frozen.
- **Retry:** allowed only after the failed attempt is diagnosed and recorded, with no rule loosened. Any change to frozen
  files is recorded and re-hashed before the retry.
- **Unchanged:** the per-attempt wall cap and all criteria of amendment 18a. Startup, warm-up, the appended phase and
  shutdown count toward the total.
- **Not authorized:** Step 3 or any behavioural run.

**Amendment 18c (2026-10-13; frozen step-2 package, before any B1 GPU data).** The CPU preparation passed and was
verified by the coordinator against the frozen hashes. d7/step2/SHA256SUMS (95 files) has sha256 97c26d11…; FROZEN.md
holds the rules.

Images:
- **B1:** sha256:799d3cf8…. A full filesystem diff against B0 shows only `sparse_attn_indexer_kpool.py` and its bytecode
  changed.
- **Guard image:** sha256:4d7bd73e….
- **Recipe fork:** `+bundle-d1234+d7`, draft-untested.

Guard amendment:
- **Write rule:** idx_w_pf models B1's rule from the writer's actual ring status.
- **Independent oracle:**
  - inputs are a guard-owned shadow of each request's last ≤3 keys and gates, keyed by layer, request and ledger
    position — never the tail ring;
  - the expected record is recomputed with the engine's own compress kernel into guard scratch, so GPU semantics come by
    construction.
- **Positive control:** every aligned pool must match exactly. Any mismatch makes that rank's straddling byte checks
  INVALID, not violations.
- **New criteria:**
  - C13: at least 100 straddling pools per rank, byte-verified on every owned layer;
  - C14: at least 10,000 positive-control pools per rank;
  - zero missing, mismatched or fallback events;
  - probes identical to attempt 3's.
- **No-ring fallback:** unreachable in B1's configuration (source and environment). Any fallback event is a violation,
  and the skip-tail-cache variable makes the run INVALID.

CPU tests:
- new guard tests: 30/30 (all four fault classes detected; an unfaithful oracle turns byte checks INVALID);
- existing suite: 28/28;
- a2 regression: ok;
- smoke tests: pass;
- fix tests on the B1 image: 28/28;
- D1-D4 regressions: pass;
- D7 pipeline dry run: 0 violations, every straddling pool verified (the dry run is INVALID only for having no real graph
  capture or server arguments).

Phase 2d is analysed separately, after the window's last readout.

Budget: APPROVED_GPU_H 2.0 for the item, at most 2 attempts, wall cap 1,740 s per attempt.

Correction: amendment 17e's "distinct seeds for all 64 sampled requests" is wrong. Phases T1 and D reuse seeds
8,850,000-8,850,007 on the same prefixes, in attempt 3 as in the frozen workload. This does not affect mapping checks.
The workload is kept identical to attempt 3 for comparability.

**Amendment 19 (2026-10-13; adopts the proposal "B1 checkpoint benchmark before publication", relayed by the operator;
preparation only, no GPU use authorized).**

Objective: representative quality, completion and runtime data for the verified B1 package (B0's D1-D4 bundle plus the
D7 correction) before deciding on upstream publication. This is checkpoint characterization, not a D7 attribution
experiment.

Composition of B1, verified before this amendment:
- image sha256:799d3cf8… was built from B0's image;
- a full filesystem diff (150,806 entries) shows only `sparse_attn_indexer_kpool.py` and its bytecode changed;
- B0 → B1 in that file is exactly `d7-fix.diff`, so B0's column-511 helpers are intact;
- D1-D4 CPU regressions pass on the B1 serving image.

Proposed run:
- one GPQA pass of all 198 questions at seed 1236 with the original protocol, sampling, parser, concurrency and
  327,680-token cap;
- a fresh server with verified warm-up, shipped graph configuration, and frozen B1 image, diff and effective settings;
- primary reference: the existing stock 0.9.1 seed-1236 pass; context: the column-511 arm and stock 0.7.0 at seed 1236;
- B0 has no seed-1236 pass, so no D7 individual effect is inferred.

Cost: a runner and costed proposal come before launch. Planning uses the recent bundle passes (4.54-4.84 occupied
GPU-hours), with a proposed 7.5 GPU-hour watchdog. No automatic retry or additional pass.

Measurements:
- empty replies, separating cap exhaustion from premature stopping;
- adjudicated stated-answer correctness across all requests, with raw correctness kept for continuity;
- generated tokens across all requests;
- occupied GPU-hours and throughput;
- execution errors, preemption and validity checks.

The interpretation is frozen before generation. A favourable result supports further evaluation or publication claims;
an unfavourable one triggers review and does not by itself invalidate independently verified corrections.

Reporting: checkpoint records stay separate (B0: seeds 1234 and 1235; B1: seed 1236). Pooled scores are not labelled a
three-pass B1 result, and any family-level combination is exploratory and mixes two implementations.

Publication preparation, during the run:
- drafts of a tpurtell PR (exact changes, minimal reproductions, verification coverage and limits, B0/B1 results by
  checkpoint, no claim that the gap is explained);
- a vLLM bug report on D7 (minimal upstream reproduction, expected versus actual pooled contents, scheduler conditions,
  correction approach, engine GPQA results as context only).
Publication stays held for the operator's review after results.

Operator note: if the pass is strong, an additional checkpoint pass may be warranted. That is a separate decision.

**Amendment 19a (2026-10-13; frozen B1 screen package, before any generation).** b1-screen/SHA256SUMS has sha256
73fa6661…, covering the runner (a4bb8e6b…), the analysis (79268c47…), the unchanged column-511 adjudication helper and
FROZEN-INTERPRETATION.md (bbfb63d7…).

Runner (a copy of the seed-1235 runner with B1's recipe, image id and env.sh pinned):
- **Seed 1236:** reached as pass 3 through temporary empty pass-1/2 placeholders, which are removed on every exit path.
  Pass 3 runs as the first pass of a fresh server.
- **Protocol and limits:** original protocol; 7.5 GPU-hour watchdog; no retry.
- **Added refusals:** frozen hashes, prior B1 data, launcher overrides, ports, and GPU occupancy.
- **Correction:** the earlier runners' watchdog `pkill` pattern never matched the chain, because `+` in the recipe name
  acts as a regex operator. lm_eval and the server would still have been stopped, and no watchdog ever fired in a B0
  session. The new runner escapes the pattern.

References, with request seeds confirmed from the proxy logs (each was the second pass of its server session; B1 will be
the first of a fresh one):

| Pass | Role | Empty | Raw | Stated |
|---|---|---:|---:|---:|
| stock 0.9.1 pass 3 | primary | 6 | 171 | 174 |
| column-511 arm pass 3 | context | 6 | 171 | 174 |
| stock 0.7.0 pass 3 | context | 1 | 175 | 177 |

Frozen categories: amendment 15b's scheme with E = 6 as the baseline (SIMILAR OR WORSE / MODEST IMPROVEMENT / PROMISING,
the material concern and accuracy concern on the six-pass family standard). For amendment 19 this reads as:
- **favourable** = PROMISING;
- **unfavourable** = a material concern, or E ≥ 7 (above every stock-family pass), which triggers review;
- **similar** = otherwise.
There is no significance or equivalence claim and no D7 effect. B0 and B1 records stay separate; the three-seed view is
exploratory.

Dry runs:
- analysis with the arm as a labelled stand-in: valid 41/41;
- category boundaries: 16/16;
- refusal paths: 23/23;
- session simulation (normal end, SIGTERM, watchdog): 30/30;
- `DRY_RUN=1` on the real machine: passes.

Expected cost: 4.6-6.0 occupied GPU-hours (point estimate about 5.1); watchdog 7.5.

Correction: the seed-1235 table published the 0.7.0 stated count as 178; the published rows give 177 (context column;
no category change).

GPU use awaits the operator's authorization.

**Amendment 19b (2026-10-13; operator authorization, before generation).** "Launch": one B1 seed-1236 GPQA pass under the
frozen package of amendment 19a, with a 7.5 occupied GPU-hour watchdog, one session and no retry. Upstream contributions
stay as drafts only, with no GitHub PR, issue or other publishing.

**Amendment 19c (2026-10-13; B1 seed-1236 result, recorded after the run).**
- **Validity:** valid on every check.
- **Result:** 9 of 198 replies empty, all at the cap, and 0 stopped early. Raw 173 and stated 174; one reply was
  hand-read and found not hedged. 4.81 M tokens were generated. 6.33 occupied GPU-hours, with the watchdog not reached.
- **Against stock 0.9.1 at seed 1236** (6 empty, raw 171, stated 174): doc 170 rescued; docs 8, 69, 130 and 145 newly
  empty.
- **Frozen reading:** SIMILAR OR WORSE (amendment 15b); for amendment 19, **UNFAVOURABLE**, because E = 9 is above every
  stock-family pass. This triggers review and does not invalidate the verified corrections.
- **Exploratory three-seed view** (B0 at 1234 and 1235 with B1 at 1236): 15 against 16 empty of 594.
- **Context:** builds that differ numerically share few identical replies at the same seed (B1 against stock 0.9.1: 5 of
  198).
- **Review route chosen by the operator:** a B0 pass at seed 1236. B0 and B1 differ only in the D7 file, and D7 is
  reconstructed as not reaching GPQA prompts, so B0 at 1236 is a near-replicate of B1. It measures run-to-run variation
  at a fixed seed and completes B0's three-seed record. One pair of passes cannot fully separate D7 from that variation.

**Amendment 19d (2026-10-13; operator authorization "proceed with B0 seed 1236 pass", before generation).**
- **Scope:** one B0 GPQA pass at seed 1236, all 198 questions, original protocol, fresh server.
- **Limits:** 7.5 occupied GPU-hour watchdog, one session, no retry.
- **Launch:** only after its package (runner, frozen analysis including the B0-against-B1 identity and per-question
  measures, and a frozen interpretation) passes its dry runs and its hashes are frozen and verified by the coordinator.

**Amendment 19e (2026-10-13; frozen B0 seed-1236 package, verified by the coordinator, before generation).**

Package: b0-screen-s1236/SHA256SUMS has sha256 5615737c…, covering the runner (211bfff4…), the analysis (f707d200…) and
FROZEN-INTERPRETATION.md (42262e39…), with the external pins identical to the B1 package.
- Pass 3 of the B0 recipe gives seed 1236 naturally, so no placeholders are used. B0's passes 1 and 2 are pinned by their
  samples hashes.
- Primary comparison: B0 against B1 at seed 1236.
- Frozen readings:
  - **(a)** B0 E ≥ 8: the seed-1236 excess is a property of the corrected family at this seed, not attributable to D7;
  - **(b)** B0 E ≤ 6: either D7, which would contradict the zero-exposure reconstruction, or run-to-run variation; one
    pair of passes cannot separate them. A weak lean comes from the count of identical non-empty replies (I_ne): ≥ 20
    triggers a CPU re-check of the reconstruction, ≤ 2 means run-to-run variation suffices, and 3-19 gives no lean;
  - **(c)** E = 7: no reading.
- Continuity: amendment 15b's category against stock 0.9.1 at seed 1236. B0's three-seed record is exploratory, and B1
  stays separate.
- Dry runs pass: two stand-ins (49/49 each), reading boundaries 41/41, refusals 30/30, session simulation 34/34, and
  `DRY_RUN=1` on the real machine.
- Expected cost: 4.6-6.4 GPU-hours; watchdog 7.5.

Correction (coordinator): earlier "identical replies" counts between builds at the same seed (for example B1 against
stock 0.9.1, 5 of 198) were all pairs of empty replies at the cap. The count of identical non-empty replies is 0 in
every earlier pair. The frozen analysis reads the non-empty count.

Post-hoc (operator request, after B1's result; not frozen): `b1-screen/run-s1236/token-analysis-posthoc.md`.
- B1's +0.56 M generated tokens against stock at seed 1236 equal its 3 extra capped replies; its finished replies total
  0.43 M fewer tokens.
- The per-document deltas over 188 mutually finished documents have median −30, the same spread as every other pair.
- B0 at seeds 1234 and 1235 generated fewer tokens than stock.
- The four newly capped documents are short single-chunk prompts, classed unsplittable.

**Amendment 19f (2026-10-13; exploratory zero-GPU tail analysis, chosen after the B1 seed-1236 result; not
pre-registered).** Inputs: the recording-proxy logs of the 11 logged K3.25 GPQA passes (0.7.0 seeds 1235-1236; stock
0.9.1 and the column-511 arm at 1234-1236; B0 at 1234-1235; B1 at 1236). Counts only. Older records kept no
per-reply token counts.

Results:
- **Reaching long:** replies reaching ≥ 100k tokens are similar in number per pass: 0.7.0 7.0, the 0.9.1 family 8.3,
  B0+B1 5.7.
- **Finishing once long** (bootstrap over questions):
  - P(finish | reached 100k): 0.7.0 0.71 [0.46, 0.93]; 0.9.1 family 0.40 [0.26, 0.57]; B0+B1 0.12 [0.00, 0.27];
  - at 200k: 4 of 8, 3 of 33 and 0 of 15.
- **Finishing hazard by length:** 0.9.1-family replies almost never finish beyond about 150k tokens (6 of about 100
  still running); 0.7.0's still do (4 of 8 at 200-260k).
- **Question as the unit:** 8 questions reached 100k on both engines; 0.7.0 finished a larger share on 4, a smaller
  share on 1, and the same on 3.

Reading (exploratory): the completion gap is not in how often replies get long but in whether long replies terminate,
which points to decode behaviour at contexts of about 64-100k tokens and beyond. B0 and B1 are more correct (verified
corrections) but not shown more accurate, and their long replies rarely finish. The cause may be on either side: 0.9.1
failing to terminate, or 0.7.0 terminating for a reason that could include its own defects (D2, D6).

Limits: 0.7.0 has 14 replies at ≥ 100k over 11 questions; its pass 1 is unlogged; seeds are reused; controls are
historical.

Next: a proposal for a long-context termination probe (CPU discovery first), for independent review.

**Amendment 20 (2026-10-13; adopts the independent assessment "long-context termination localisation", relayed by the
operator; before any of its data).**

Objective: determine whether engines assign different termination probabilities on identical active-reasoning
histories, and whether those differences depend on the execution path or the context range. No claim is made that a
probability difference explains completion rates.

1. **Finish the current review.** B0's seed-1236 result is recorded under the frozen rules, with unchanged categories
   and these qualifications:
   - high B0 censoring means D7 is not necessary for the excess;
   - reply identity measures transcript repeatability, not outcome independence;
   - one B0/B1 pair cannot isolate a D7 effect;
   - B0, B1 and stock records stay separate.
2. **Discovery and fixture inventory (zero GPU):**
   - **Tail-accounting audit.** Stock releases are reported separately before any pooling of modified configurations.
     For each threshold and band: unique requests at risk, unique finishing events, question identities, valid
     final-answer completion against other stopping, session position and seed. 327,680-token exhaustions are
     administrative censoring in survival terms, kept separately as unanswered benchmark replies. No precise change
     point is inferred from thresholds chosen after the results.
   - **Existing histories before new ones.** Inventory retained pilot token traces (including censored reasoning beyond
     100k), successful traces approaching a reasoning close, and any raw reasoning text from later GPQA runs. Exact
     token ids are preferred; re-tokenized text is a shared counterfactual fixture, not a replay of the generated
     sequence. Forced continuation after natural completion is excluded.
   - **Focused source review** of context-dependent boundaries and dispatch: pool-count and fold transitions;
     sequence-length and index representation limits; sparse selection and attention paths at long lengths; DCP
     partition, merge and cache geometry; recurrent-state precision and update paths; long-context speculative
     verification and stopping/logits processing. Each item is classed as a known bug, expected numerical variation or
     an untested invariant; this is not a broad release audit.
   - **Termination semantics.** Verify the actual token ids and processing for reasoning close, end of turn / EOS and
     the transition into final-answer content. Reasoning close and generation stop are different events; a
     multi-token marker is scored as a sequence.
3. **First GPU proposal: a bounded fixed-history probe.** Existing histories spanning roughly 64k-128k, a shorter
   control, and windows near tentative conclusions or reasoning closes, rather than starting uniformly at 300k. Stock
   0.7.0 against B1 on identical token prefixes, labelled old shipped computation against corrected computation.
   - **Measures:** reasoning-close probability (or sequence probability); EOS / end-of-turn probabilities; sampler
     eligibility after masking and top_p; selected next-token log-probabilities; argmax agreement; repeated-run
     variation.
   - Shared-top-20 KL is not the primary measure, because it can omit the termination tokens themselves.
   - Only designated positions and metrics are captured, not full-vocabulary outputs at every token.
4. **Execution-path control.** Prefill scoring is a first comparison, not the decisive test. At a small subset of
   prefixes, prefill scoring is compared with target decode or verification scoring of the same next-token state,
   using a bounded forced-history window if existing tooling supports it. The report states how committed state is
   advanced, and repeatability is established before engine differences are interpreted. Full incremental replay that
   needs substantial new instrumentation is priced separately; prefill is never presented as decode fidelity. A cold
   prefill does not reconstruct the state accumulated during 100k tokens of speculative generation.
5. **Interpretation:**
   - a difference confined to prefill → investigate that path, with no generation-time claim;
   - a difference also present in target decode → nominate a mechanism, and check it against within-engine variation;
   - no difference on the tested prefixes → narrow that hypothesis, without excluding history-dependent state
     corruption or rare effects;
   - stock 0.7.0 favouring termination → establish whether that is intended computation or known corruption, and never
     restore an incorrect path as a remedy.
6. **Extension gate.** Extend toward 200k-300k only when a source boundary or an initial difference justifies it,
   suitable fixtures exist, and the observation would distinguish the stated alternatives. Costs come from measured
   throughput. Each GPU stage needs its own proposal and authorization; no full GPQA pass or ablation follows
   automatically.

**Amendment 20a (2026-10-13; correction of emphasis, coordinator).** Amendment 19f's reading ("the completion gap is
not in how often replies get long but in whether long replies terminate") is hypothesis-generating only. It was chosen
after the B1 result, its thresholds and bands came after the data, and its pooled groups mix stock with modified
configurations, with small counts. It does not localise the gap. Public summaries that said "located" or "localised" are
reworded, and FINDINGS item 4m is regraded as exploratory. "The D7 path cannot run" on the newly empty documents is a
reconstruction (exposure class unsplittable), not an observation. Amendment 20's audit, with stock releases reported
separately, governs any further use of these figures.

**Amendment 20b (2026-10-13; operator clarification).** A lower probability of finishing before the cap is not a higher
probability of looping. Cap exhaustion is censoring: the capped reply may be a loop (repetition) or an exhaustion
(continued varied reasoning), and GPQA records keep no reasoning text to distinguish them. Requirements:
- **Reporting:** no summary or proposal under amendment 20 equates non-completion with looping.
- **Fixture inventory:** classify each retained history with text as loop, exhaustion or natural close, using the
  repetition indicators already used by the pilot (repeated-window share, compression).
- **Probe proposal:** when windows are chosen near tentative conclusions, state for each fixture whether it was
  repeating at that point. A termination-probability difference measured on a looping history and one measured on a
  varied-reasoning history answer different questions.

**Amendment 20c (2026-10-13; tail-accounting audit result, zero GPU).** Receipts are in tail-audit/ (124 checks pass;
counts, ids and flags only).

Reproduction:
- The audit reproduces amendment 19f's counts byte for byte.
- One counting error: "6 of about 100 still running beyond 150k" summed the at-risk counts of three bands. The unique
  count is 6 of 36 (18 questions); stock 0.9.1 alone is 3 of 18, and stock 0.7.0 5 of 9.
- A valid final answer (finish stop, non-empty, an extractable letter, not ruled hedged) accounts for every finish at
  64k or more. Only four stops in all logged data are not valid answers, all below 64k.

Stock-only results, question-clustered bootstrap:

| | Stock 0.7.0 | Stock 0.9.1 | Difference |
|---|---|---|---|
| Finished at ≥ 100k | 10/14, 0.71 | 10/25, 0.40 | +0.31 [0.03, 0.55] |
| Finished at ≥ 200k | 4/8 | 2/17 | +0.38 [−0.15, 0.81] |
| Reaching 100k | 0.035 | 0.042 | −0.007 [−0.026, 0.015] |
| Capped | 4 of 396 | 15 of 594 | — |

- At matched seeds (1235-1236) the 100k difference is +0.33 [0.03, 0.57].
- Question as the unit, all seeds: 0.7.0 finished a larger share on 3 questions, a smaller share on 1 and the same on 3.
- The per-band shares do not support a specific range such as "64-100k". The 200-260k contrast rests on 4 events from 3
  questions.
- Session position and seed are confounded: every second-of-session pass is seed 1236. B1, a first-of-session pass, has
  the most caps.
- `reasoning_tokens` is null in every log. Using re-tokenized content as the axis does not change the 100k or 200k
  counts.

Reading: the hypothesis that long replies finish less often on stock 0.9.1 than on stock 0.7.0 survives the stock-only
audit, with a wide interval and small counts (about 10 finishing events per release). It remains a hypothesis, and no
range or change point is claimed.

**Amendment 20d (2026-10-13; zero-GPU discovery results and the first-stage proposal, before any probe data).**
Receipts are in tail-audit/ (20c), fixture-inventory/, source-review-lc/ and probe-proposal/.

Termination semantics:
- Stopping rules are byte-identical across 0.7.0, 0.9.1, B0 and B1: parser, sampler, speculative verification,
  tokenizer and generation config.
- `</think>` (154842) switches the output to content and does not stop generation. 154820, 154827 and 154829 stop it.
- top_p 0.95 is the only mask that can remove a termination token.
- So any engine difference in termination is in the logits.

Fixtures:
- Exact generated ids exist only for histories under 5k tokens.
- Long histories exist as text: 901 complete reasoning texts, usable as re-tokenized counterfactual fixtures, not
  replays.
- Classified per 20b: 527 natural closes, 224 loops, 92 exhaustions.
- No reasoning text exists for any GPQA pass.

Focused source review, ranked:
1. **D2 (known):** decode-written index-K pools are wrong on 0.7.0 at every concurrency; on stock 0.9.1 at 0% for 1
   request, 47-57% for 2 and 78-90% for 8; on B0/B1 never. The rank order matches the finishing rates, which is an
   observation only. Prefill cannot see it.
2. **Selection-kernel regime changes untested with live data past about 120k:** on B1's tiled route, at 65,536 and
   131,072 tokens.
3. **KDA block migration under speculation:** untested past 16k.
4. **D6 (known):** on 0.7.0 at ×5 the 4-slot tail ring wraps on rejected drafts. Decode-only; fixed from 0.9.0.

Settled by source:
- no RoPE in attention or the indexer;
- no representation limits;
- no precision change between 16k and 300k;
- the ReplaySSM rollback path was never served.

Proposal (probe-proposal/PROPOSAL.md) for independent review:
- **Instrument:** a force-decoded window. A small hook after the rejection sampler commits the history's own tokens, so
  the window's index-K, tail-ring, KDA and speculative state come from the real decode path. It is needed because Model
  Runner V2 refuses custom logits processors and prompt logprobs cover prefill only.
- **Measures:** termination-token probabilities before and after top_p, plus eligibility, at designated positions.
- **Arms** at 8 in flight: stock 0.7.0 ×5; B0 ×3; B0 ×5; the diagnostics B0 − D2 ×3 and an optional B0 ring-4 ×5;
  stock 0.9.1 as context.
- **Fixtures:** windows of 8,192 forced tokens ending at 80k-121.5k, plus short controls.
- **Stages** each separately authorized:

  | Stage | Estimate / ceiling (GPU-hours) |
  |---|---|
  | 1 | 7.74 / 10.6 |
  | 1-min | 5.0 / 7.0 |
  | 1b | 1.65 |
  | 2 | 7.62 |
  | K (kernel replay) | 0.3 |

- **Frozen readings:** repeatability first; a consistent direction on at least 6 of 8 windows; D2, D6 or draft depth
  nominated (not proven) only if its diagnostic contrast reproduces the 0.7.0-vs-B0 direction; a corrupted path never
  restored as a remedy; no claim that a probability difference explains completion rates.

No GPU use is authorized by this amendment.

**Amendment 21 (2026-10-13; adopts the independent assessment "revised first stage: validate the instrument and isolate
long-context effects", relayed by the operator; before any probe data).** Stage 1 and 1-min are not run as written.
Revised CPU preparation proceeds. K is proposed first, then a bounded GPU validation of the hook, each needing its own
authorization. D6/ring-4, extra context arms and the larger fixture campaign are deferred.

1. **Corrected baseline:** B1 ×3. B0's unfixed D7 is exposed by long cold prefills, so B0 is not "corrected" for this
   preparation. Stock 0.7.0, if compared at all, is labelled shipped computation with its known prefill and decode
   defects documented; it is not a fidelity reference. No difference is attributed to D2 or D6 without isolation.
2. **D2 diagnostic by decode-window-only fault injection:**
   - prepare the prefix with verified B1 computation;
   - introduce the old tail-mapping behaviour only during the forced decode window;
   - keep D7-correct prefill and every other correction;
   - verify the injected mapping and the resulting corruption with the existing guards.
   This is controlled fault injection, not the stock engine. If it cannot be implemented and validated cheaply, the D2
   contrast is deferred; D2 is never removed globally while D7 is retained.
3. **K, model-free selection validation (proposed ceiling 0.5 GPU-hours):**
   - covers both sides of the fold and supertile transitions, the corresponding old-engine DCP2 boundaries, and contexts
     up to the equivalent of 330k;
   - uses the actual kernels on finite quantized inputs against the validated reference, with explicit treatment of
     ties and rounding;
   - checks the selected top-512 set, the final attended selection after the truncation and tail policy, eligibility
     masks and output ordering.
   A clean result retires the tested selection failures; it does not establish termination equivalence.
4. **Hook validation before any probability is interpreted:**
   - short exact-id histories, covering zero accepted drafts, fully matching drafts, rejection at different draft
     positions, bonus-token accounting, and block and ring boundaries;
   - compared against a validated single-token history-advance reference;
   - committed state checked: token positions, KDA recurrent and convolution state, MLA records, index-K pools and tail
     contents, kept distinct from permitted speculative scratch, with justified tolerances;
   - returned ids matching the fixture are necessary, not sufficient;
   - it must be shown that designated logits are measured on the intended prefix, once per position, before forcing
     changes the decision.
5. **First long-context contrast (priced after validation):** B1 ×3 against the validated decode-only D2 injection at
   ×3, on existing diverse histories with a fixed batch composition. Per-window results are preserved and batch-level
   dependence is accounted for, since D2 couples requests. Repeated measurements in each arm; no range rule is applied
   to a single run. A stock-0.7.0/B1 comparison may follow if it answers a distinct question; depth-matched and ring-4
   contrasts are conditional follow-ups.
6. **Metrics:**
   - primary candidate: reasoning-close probability at prespecified active-reasoning positions, including positions near
     natural closes;
   - reported separately: `</think>` probability and top_p eligibility, each generation-stop token, forced-token
     likelihood, entropy and argmax agreement;
   - reasoning close and premature generation stop are never combined into one score;
   - window aggregation and the primary contrast are specified before data;
   - the 0.5-nat and 6-of-8 rules are operational screens, not statistical proof or equivalence.
7. **Window limitation:**
   - measure how many final attended pools were written during the decode window, including corrupted ones, because the
     fraction of all pools is not enough when recent pools may be selected disproportionately;
   - a negative window result cannot exclude corruption accumulated over a whole generated history;
   - introduce the oracle-draft bracket before attributing a rejection-dependent signal to the served D6 mechanism.
8. **Extension gate:**
   - before additional arms or whole-history replay, report hook validity, within-arm variation, actual
     corrupted-state exposure, the first contrast with its limits, and measured cost;
   - extend only when the next observation distinguishes stated alternatives;
   - account for the 58 histories not in the 527 / 224 / 92 classification before freezing fixtures.
   No probability difference alone explains benchmark completion rates.

**Amendment 21a (2026-10-13; B0 seed-1236 result under the frozen rules of amendment 19e, recorded after the run).**
- **Validity:** valid. 5.866 occupied GPU-hours, watchdog not reached.
- **Result:** 5 of 198 replies empty, all at the cap, 0 stopped early. Raw 169 and stated 173; four replies were hand-read
  and none was hedged. 3.98 M tokens were generated.
- **B0 against B1 at seed 1236:** 6 rescued (docs 8, 69, 88, 127, 130, 147), 2 newly empty (docs 160, 182), 3 empty in
  both (docs 79, 81, 145). Identical non-empty replies: 0.
- **Frozen reading (b):** B0 is within the stock range, so the B1/B0 difference is either D7, which would contradict the
  zero-exposure reconstruction, or run-to-run variation. One pair of passes cannot separate them. I_ne = 0 gives the
  weak lean that same-seed passes behave as distinct draws, so variation alone suffices; no D7 effect is inferred.
- **Amendment 20 §1 qualifications:** B0's censoring is not high, so the D7-not-necessary qualification does not apply.
  Reply identity measures transcript repeatability, not outcome independence. One B0/B1 pair cannot isolate D7.
- **Continuity** (15b against stock 0.9.1 at seed 1236, 6 empty): MODEST IMPROVEMENT, NO CLEAR SEPARATION; no accuracy
  or material concern.
- **B0 three-seed record against stock 0.9.1 at the same seeds** (exploratory: historical controls, seed reuse, and this
  pass chosen after inspecting B1's): 11 against 16 empty of 594; reduction 0.84 points [−0.34, 2.02]; one-sided
  p = 0.13.
- **Observation:** at a fixed seed on near-identical computation, 8 of the 14 questions that capped in either pass
  changed status. Per-question cap outcomes are highly volatile between runs.
- B0, B1 and stock records stay separate.

**Amendment 21b (2026-10-13; operator authorization "proceed", before any K data).** K runs as frozen in k-selection/,
with its SHA256SUMS verified by the coordinator.
- **What runs:** each image's own selection code (the B12x tiled and fused routes, 0.7.0's DCP2 merge, B1's column-511
  policy, expand plus tail) on both sides of every fold, supertile and trim boundary, at lengths from 1 to 331,779 tokens.
- **Inputs:** real, synthetic, exact-arithmetic adversarial and poison families.
- **Criteria:** C1-C7 as frozen.
- **Limits:** one GPU, ceiling 0.5 occupied GPU-hours, one session; cells cut by the time guard are reported as not run.
- **What a clean result means:** the tested selection failures are retired. It establishes nothing about termination,
  decode-written index-K bytes or prefill top-k.

**Amendment 21c (2026-10-13; K run 1 diagnosis, post-inspection fixes and operator authorizations, before any rerun
data).**

K run 1 (0.1417 GPU-hours):
- **B1:** 34 of 34 cells PASS. This retires the tested B1 selection failures from about 2k to 330k tokens, on the tiled
  route at 24 and 32 rows and the fused route at 4, 8 and 16 rows.
- **0.7.0:** 2 FAIL (C5), 1 ERROR, 19 NOT-RUN.

Diagnosis (k-selection/DIAGNOSIS-run1.md, CPU only):
- **The ERROR was a plan defect.** The shortest target gave 0.7.0 (K = 5) a row with L = −1, and the resulting error
  aborted the loop. It is a harness fault, not an engine result.
- **The C5 failures were not a harness artefact.** Both perspectives saw identical inputs in engine rank order. 0.7.0's
  `run_row_topk` assigns output slots through shared-memory atomics, so the output order varies between launches, and
  the selected set varies only among exact ties at the threshold.
- **Column 511 is a genuine 0.7.0 cross-rank inconsistency**, within the known column-511 class (D1). Each rank drops
  its own column 511, so the ranks can drop different low-ranked pools. The modelled exposure is about 88% of rows in
  the tested cells, a race among the lowest few selected pools. How often the ranks actually diverge is not measured.

Post-inspection fixes, with no criterion changed:
- **PI-1:** the minimum target accounts for draft slots.
- **PI-2:** the descriptive counter is guarded.
- **PI-3:** a cell's error no longer aborts the cells after it.
- **PI-4:** perspective mismatches are classified as order-only, tie-set or column-511, and recorded with a relaunch
  control.
- **PI-5:** 0.7.0-only rerun settings.

C5 is expected to fail again in the rerun, and the report will quote the PI-4 breakdown beside it.

Operator authorizations:
- **K:** cumulative 0.65 occupied GPU-hours, with the 0.7.0-only rerun capped at 0.5.
- **Hook validation session:** 2.4 GPU-hours, at most 2 attempts. It runs only after the forcing CPU test F7 (unforced
  co-runners unchanged with the hook on) passes for an understood reason and the package's hashes are frozen and
  verified.

**Amendment 21d (2026-10-14; K rerun result, 0.7.0 only, recorded after the run).** All 22 cells ran; 0.1408 GPU-hours,
for a K total of 0.2825 of the 0.65 approved.

**Tally:** 10 FAIL (tiled; C5 only) and 12 INVALID-ROUTE (fused; C5 plus C7). C1-C4 are met in every cell: the
selected-id validity, the top-512 set within the tie rule, and the expand-and-tail reference.

**C5 breakdown (PI-4):**
- **Set differences:** only among exact ties at the threshold (non-tie 0 in every cell).
- **Column 511:** differs between the two rank perspectives in about 60-80% of rows; about half of those are non-tied
  pools. The pool landing in column 511 can be highly ranked (minimum rank 1-94 per cell).
- **Relaunch control:** relaunching perspective 0 on identical inputs also differs in every record. The variation is
  launch nondeterminism of 0.7.0's atomic-slot merge, not a rank effect.

**Reading:**
- In stock 0.7.0 (DCP2) each rank drops an arbitrary selected pool, and differently on each launch.
- Its attended set therefore varies run to run and between its two GPUs: 510 whole pools plus half of two possibly
  different pools.
- This extends the known D1 (column-511) class to a cross-rank inconsistency. It is a property of the old engine's
  path; B1 is unaffected.
- No termination mechanism is inferred.

**C7 on the fused cells:** the route was fused as planned, but 0.7.0's planner exposes no CTA count (`None`), so the
trim geometry the cells target cannot be confirmed. Those cells' C1-C4 results stand; their boundary placement is
unverified.

**K summary:** B1's selection is verified at 2k-330k (34/34). 0.7.0's selection is correct as a set, but
nondeterministic and inconsistent across ranks at column 511.

**Amendment 21e (2026-10-14; frozen hook-validation package, verified by the coordinator, before any GPU data).**
Package: hook-validation/SHA256SUMS (100 files) has sha256 67d8e54f…. The hook module is `forcewin_hooks.py` (published name)
c079560e…, in image sha256:1b9f8b15… (B1-mapguard plus the hook; its `.py` manifest differs from the base in 6 files).

CPU results:
- forcing tests 66/66 across 9 shards;
- D2 window injection 16/16;
- inertness check I1: identical outputs with the hook off, unset, or on without a keyed request.

**Post-inspection change to F7** (test only; hook unchanged):
- **Before:** the co-runner's full sampler row was compared hook on against hook off.
- **Why it failed:** entries past `num_sampled` are uninitialised scratch (`torch.empty`) that differ even between two
  hook-off calls.
- **After:** over 4 steps, the committed tokens, `num_sampled` and `num_rejected` must be equal, plus the new negative
  control F7n.
- **Supporting evidence:** the co-runner's processed (top_p) logits are bitwise equal with the hook on and off, and its
  committed state is identical over 6 steps. Nothing the engine consumes reads the scratch.

The 58 non-classified histories are K4, so none is a K3.25 fixture.

The session runs under the 2.4 GPU-hour authorization of amendment 21c: at most 2 attempts, a 3,240 s wall cap per
attempt, 1.8 GPU-hours per attempt.

**Amendment 21f (2026-10-14; hook-validation attempt 1 result, recorded after the run; diagnosis pending).** Verdict
INVALID; 1.2117 GPU-hours of the 2.4 authorized.

Passed:
- V0: unkeyed probes bitwise equal to B1's earlier record;
- H1a-e and H1g-j: forced ids committed exactly, each position measured once, post-mask values from the served logits,
  history correct after every step, both ranks identical, acceptance accounting correct, no hook errors;
- H3: every draft schedule agrees with the single-token advance within the decode-vs-decode null;
- H4a: integer committed state;
- H5a: injection exactly on the intended tokens;
- H5d: injected arm 61% gross pools at 8 in flight.

Failed:
- **H1f:** top_p eligibility bounds, 5 requests.
- **H2a:** forced decode against prefill, top-1 0.845.
- **H2b/H2c:** the alignment-power controls.
- **H4b:** KDA state against the prefill build, median 4.5% relative, maximum 36%.
- **H4c and H5b:** index-K pools against the prefill rebuild on B1 without injection, 0.7-1.5% gross.
- **H5c:** threshold discrimination.
- **H5e:** the solo injected request, 0.18% gross.
- **H5f:** the shadow ledger, corrupt agreement 0.61.
- **H5g:** the guard count rule.
- **H5h:** in-graph exposure records missing at 14,127 of 20,469 positions.

Nothing from this run is interpreted. A CPU-only diagnosis classifies each failure as a harness defect, a criterion
without a measured basis, or a genuine engine finding. H4c/H5b take priority, because gross B1 pools could be a
comparison misalignment, an expected writer-numerics gap, or a new decode-path defect. There is no attempt 2 before the
diagnosis is reviewed; one attempt (about 1.19 GPU-hours) remains under the authorization.

**Amendment 21g (2026-10-14; adopts the independent assessment "attempt-2 validation and first-contrast guidance",
relayed by the operator; before any attempt-2 data).** Proceed with the scoped attempt-2 package after recording these
qualifications. The existing 2.4 GPU-hour ceiling and the last-attempt rule stand; no additional GPU time is authorized.
Attempt 1 remains INVALID. Attempt 2 tests the repaired instrument under revised, explicitly post-inspection criteria.
The PA2-1 to PA2-15 changes are recorded in hook-validation/prep-a2/PA2-changes.json.

1. **Index-K content gate.** gross ≤ max(2 × max(in-session decode null, in-session prefill null), 0.1%) is accepted as a
   coarse numerical compatibility screen, alongside the exact mapping and ledger checks.
   - **Conditions:**
     - the nulls cover the same fixture set, layers and overlapping token positions as the tested comparison (a single
       trace is insufficient);
     - aggregation, denominators and exclusions are frozen before attempt 2;
     - gross counts and rates are reported by fixture and layer as well as pooled;
     - the original criteria and results are kept beside the revised ones;
     - the 0.1% floor is an operational tolerance, not a measured universal noise floor.
   - A fixture substantially outside its null stays flagged and cannot carry a mechanism claim.
   - **Structural checks retained:** exact request/token attribution and mappings; B1 shadow ledger entirely clean;
     committed pools byte-stable over time; zero gross pools in layers 3-15; no judged page above 25% gross (unique
     pools, minimum page population).
   - The wrong-page and attribution negative controls must stay demonstrably sensitive under the revised analysis
     (offline fault injection on existing captures where appropriate).
2. **KDA gate.** Two claims are kept separate: which slot the engine commits, and which slot is numerically nearest the
   prefill-built reference.
   - Commit bookkeeping and expected-slot checks stay exact; no wrong-slot commits are permitted.
   - **Nearest-reference identification:** 100% in layers 0-17 and ≥ 99% deeper, as a diagnostic screen. Every
     disagreement is reported with its distances and margin. Deep-layer disagreements are unresolved numerical
     comparisons, and a comfortably separated wrong-slot match requires investigation.
   - Relative error ≤ 0.02 with margin ≥ 10 applies to layers 0-2 only.
   - The original 100% criterion and result are kept beside the revised reading.
3. **H2 and H3 carry-over.** Acceptable because the forcing and commit code is unchanged, provided:
   - the image diff confirms that scope;
   - provenance and corrected joins are explicit;
   - the results are labelled carried-over evidence;
   - current-image V0, forcing, attribution, ledger and state checks pass.
   Current-image alignment and the argmax shift control are recomputed from the overlapping V2a/V2c reference rows. The
   two repaired bookkeeping paths must be revalidated on GPU. The poison phase is retained if it fits the frozen budget;
   required coverage is never dropped silently.
4. **Top-p edge case.** The unmasked near-one-hot rows are a separately reported class. The bound applies to the tracked
   tokens; total retained mass outside the intended nucleus is reported if existing records allow. There is no sampler
   change during validation.
5. **What a pass establishes:** forced-token commitment and measurement alignment; exposure and selection attribution;
   verified D2 injection; numerical comparisons read against measured controls.
   **What it does not establish:** full-history replay equivalence; the absence of every deep-layer numerical issue;
   that D2 explains GPQA completion rates.
   No automatic retry or behavioural evaluation follows.
6. **Long-prompt / short-output control** (first-contrast design, costed separately, not from the attempt-2 allowance).
   - **Construction:** the same token history and final context position built two ways: a long prefill with a short
     forced window, and a shorter prefill with the planned longer window. Compared over common final positions, on clean
     B1 and on the decode-window D2 injection, with matched batch composition and concurrency. Clean and injected
     conditions run separately.
   - **Recorded:** decode-written pool counts, corrupted pool counts, selection of corrupted pools, reasoning-close
     probability before and after top-p, and EOS and other stops separately.
   - It distinguishes sensitivity to history construction and to accumulated decode-written state at fixed total
     context. It is not a pure test of output length.
7. **Reading the first contrast.**
   - The primary measure is `</think>` probability and nucleus eligibility. EOS and the other stops are separate
     outcomes, and premature stopping is not successful completion.
   - If D2 changes selected keys and termination probabilities beyond the matched null, it is a supported local mechanism
     candidate.
   - If the result is null, only this decode-window exposure is excluded; accumulated full-history D2 effects are not.
   - No full GPQA pass is purchased solely because the instrument validates.

**Amendment 21h (2026-10-14; coordinator, applying amendment 21g's wording; before any attempt-2 data).** The attempt-2
package implements 21g (PA2-16 to PA2-25; hook-validation/ATTEMPT2-21G.md; image unchanged, sha256:ac05230b…; hook
module 15e93e23…). Recorded limits and one interpretation:
- **H5e-r** (the solo injected request) is gated on the pooled matched null of the eight-fixture set, which is 21g's
  aggregate gate. The comparison against its own fixture's null is reported as a flag, not a failure, following 21g:
  "An aggregate pass does not clear every fixture. A fixture substantially outside its corresponding null remains
  flagged." This avoids a false INVALID when a single fixture's decode null happens to be low.
- **The decode null shares an arm** with each tested comparison. A defect confined to one arm raises its own limit and
  is absorbed by the content gate; such defects are caught by the exact structural checks and the arm-asymmetry flag.
  Recorded as a limit of the content gate.
- **V3 index-K content** has no matched null; it stays descriptive and ungated.
- **Attempt 1 had no V4n data**, so the content gate has been exercised on synthetic data only. The fault-injection base
  used proxy nulls.
- **KDA metadata** disagreed in 2 of 106 attempt-1 snapshots, both at frontier 4,351 next to a block boundary. It is
  reported as an unexplained, ungated diagnostic and investigated if it recurs.

**Amendment 21i (2026-10-14; frozen attempt-2 package, verified by the coordinator, before any attempt-2 data).**
- **Package:** hook-validation/SHA256SUMS (333 files) has sha256 4ae4ad22….
- **H5e-r (PA2-26):** gated on the pooled eight-fixture matched null, with H5e-fixture reported only and H5e-pooled kept
  beside them.
- **Reruns:** unchanged verdicts on the synthetic good and bad runs, the attempt-1 replay and all ten fault injections.
  Two fault cases flip H5e-r to PASS because a one-arm fault inflates the pooled null (the design limit recorded in
  21h); both still fail their target checks.
- **Image:** sha256:ac05230b…, unchanged.
- **Budget:** spent 1.2117, cap 1.188 GPU-hours for this attempt, wall cap 2,138 s.
- **Launch:** attempt 2 of 2, under the 2.4 GPU-hour authorization of amendment 21c.

**Amendment 21j (2026-10-14; hook-validation attempt 2 result, recorded after the run).**
- **Verdicts:** the revised verdict (21g/21h criteria) is **VALID**. The original attempt-1 criteria read INVALID and are
  reported beside it.
- **Cost:** 0.795 GPU-hours; the item total is 2.0067 of 2.4. No attempts remain.

Revised checks (all pass):
- V0 unkeyed probes bitwise equal to B1, and full coverage;
- forcing H1a-e and H1g-j on the current image (73 requests);
- alignment against prefill on the current image (top-1 ≥ 0.95 within the null) and the argmax shift control
  (aligned ≥ 0.90, shifted ≤ 0.20);
- H3: the natural and oracle schedules agree with poison (top-1 0.984);
- KDA: exact slot bookkeeping, and nearest-reference identification 100% with 0 disagreements;
- index-K structure: exact attribution, no page above 25%, zero gross in layers 3-15, 4,037 pool pairs byte-identical
  over time;
- index-K content: the B1 arms 0.61% against a matched limit of 1.16% (N_dd 0.58%, N_pp 0.46%); the solo injected
  request 0.18% against the pooled limit;
- D2 injection at 8 in flight: 59.7% gross, 51.6 times the limit, with the guard pattern exact per rank;
- B1 shadow ledger all clean (10,172 pools);
- exposure records at every designated position (7,922, none missing).

Reported, not gated:
- the solo request against its own fixture's null: not flagged (0.18% against 0.40%);
- fixture flags under 21g: fixture be2e9e349395 is outside its own matched null in both B1 arms, and fixture
  376073921446 shows arm asymmetry. Neither may carry a mechanism claim without further examination;
- KDA metadata disagrees in 2 of 106 snapshots again, so it recurs and is to be investigated (21h);
- the unmasked top-p rows: 18 rows, tracked kept mass ≤ 2.4e-9, total mass outside the nucleus ≤ 1.0e-7;
- V3 decode against prefill: 1.78%, descriptive only.

Open check: both B1 arms show the same pooled gross count against the rebuild (172 of 27,973) while their page maxima
differ. This is to be confirmed as coincidence, not a duplicated comparison, before the instrument is used.

Per 21g §5, a pass establishes forced-token commitment and alignment, exposure and selection attribution, the verified
D2 injection, and numerical comparisons read against measured controls. It does not establish full-history replay
equivalence, the absence of every deep-layer numerical issue, or that D2 explains GPQA completion.

**Amendment 21k (2026-10-14; attempt-2 follow-up and the first-contrast proposal, before any first-contrast data).**

Follow-up (hook-validation/followup-a2/, CPU only, post-inspection):
1. **The two B1 arms' equal pooled gross count (172) is a coincidence, not an aliased comparison.** The arms used
   different dumps and share 0 of 27,973 byte-identical units. Their gross sets overlap with Jaccard 0.35, and the
   counts are equal only at the 0.25 threshold. Every frozen value reproduces from raw bytes. The 21j open check is
   closed.
2. **The KDA metadata disagreement is explained.** It is one snapshot on both ranks at frontier 4,351. In align mode, a
   block-column change moves the committed state and sets `num_accepted` to 1; that rule predicts 106 of 106 snapshots
   in both attempts. The committed state there is correct (34 of 34 layers). The convolution state at the boundary is
   not compared.
3. **Both fixture flags are numerical fragility, not instrument defects.** be2e9e349395 is deep-layer only, and 13 of
   its gross units reproduce across three independent decodes. 376073921446's asymmetry is one position counted per
   layer against a zero prefill null; by position the arms are symmetric. These fixtures belong to attempt 2's
   short-trace set and are not in the first-contrast fixtures.

First-contrast proposal (first-contrast/PROPOSAL.md, for independent review; not authorized):
- **Primary:** B1 ×3 clean against B1 with the decode-window D2 injection ×3, on the 8 first-contrast fixtures
  (re-tokenized 63k-120k histories, windows of about 8.2k forced tokens). One session, 8 in flight, clean and injected
  in separate batches, distinct cache salts. A warm prefix request per round keeps the batch's decode starts aligned.
- **Order:** C1, C2 (repeatability gate), D2, D1, D3, C3 (residue check).
- **Long-prompt control (21g §6):** the same history and final positions built with a long prefill and a 1,025-token
  window, ×3 per condition.
- **Primary statistic:** per fixture, ln of the window-mean pre-top_p P(`</think>`) over prespecified active positions;
  the run is the unit.
  - **Beyond the null:** |Δ| > 2·max(N, 0.05) and the D and C runs fully separated. 3 against 3 gives a best exact
    one-sided level of 1/20, reported as a bound.
  - **No reading** if the within-arm null exceeds 0.25 nat.
  - **Selected keys changed:** corrupted window pools selected in every D run and in none of the C runs.
  - Readings per 21g §7.
- **Fixture rule:** a pre-data exclusion rule file (empty), and in-session flags with a leave-flagged-out sensitivity,
  counted by position and by layer-unit.
- **Environment:** `WMAX` 8448, required because the validated sessions ran with 5120.
- **Cost (GPU-hours, estimate / ceiling):**

  | Option | Contents | Estimate / ceiling |
  |---|---|---|
  | A0 | primary only | 1.96 / 2.99 |
  | A | primary plus long-context instrument checks | 2.36 / 3.57 |
  | B (recommended) | primary plus the long-prompt control | 2.82 / 4.29 |
  | B2 | the control ×2 per condition | 2.53 / 3.85 |

- **Open risks:**
  - decode speed at long context is unmeasured;
  - the instrument is validated only below 5k tokens, and windows will cross two blocks;
  - forced requests on a cached prefix are new;
  - fewer than 8 in flight at window ends;
  - the prefill null may be 0;
  - fixture overlaps;
  - 45-60 GB of window dumps.

**Amendment 22 (2026-10-14; adopts the independent assessment "first long-context contrast: recommended amendments",
relayed by the operator; before any first-contrast data).**

**Option B:** the primary contrast plus the long-prompt / short-window control, with three repeats per condition and a
budgeting target of a 4.29 occupied GPU-hour ceiling. Any added reference build or boundary check is repriced before
authorization; the invalid-start allowance is not consumed silently. There is no separate validation session:
validation is integrated, and the run stops before further injected runs if it fails.

1. **Primary statistic kept,** with its limits stated: it measures close propensity on this counterfactual fixture
   panel, not a finishing hazard or GPQA completion probability.
   - **Kept:** per-fixture results, a question-weighted sensitivity, near-close and active groups, the common-tail
     construction comparison, post-top_p probability and eligibility, and each actual stopping token separately.
   - **Added:** an end-position-excluded sensitivity.
   - A pre-mask change that stays outside the nucleus is a computational finding whose relevance to sampled termination
     is unestablished.
2. **No exact permutation claim.** C_r and D_r share a prefix construction, which makes three paired rounds, and the
   order is fixed within one session. The three paired differences are reported explicitly. Complete separation and the
   beyond-null rule are operational screens; no exact significance is claimed.
3. **Stronger prefill null before launch.** A prefill reference at a different build length, or a validated different
   batch shape, compared only on shared causal positions. An already planned reference warm is adapted if its cache
   safety can be shown; otherwise one more independently salted rebuild is priced.
   - **Kept:** fixture × layer × shared-position matching, exact cache accounting, per-fixture flags, and the pooled
     gate with the original criteria beside it.
   - **Correction:** N_pp = 0 does not force the limit to 0.1% when N_dd is nonzero.
   - If the numerical gate fails, the long-context instrument is unvalidated under these bounds. B1 is not
     automatically defective, and no threshold is revised after viewing the contrast.
4. **Expanded envelope validated in-session, before the injected batches.** Long-context alignment and content checks
   come from the clean runs and the reference builds, with reference warms moved earlier where practical. Explicit
   checks:
   - `WMAX` 8448 without truncation;
   - absolute positions on cached prefixes;
   - exact attribution in flattened and padded steps;
   - both block crossings, and M3's selection-regime crossing;
   - a clean prefix state after injected traffic.
   The guard establishes that the injection cannot write into the shared cached prefix or its committed recurrent state,
   with sampled before/after state checks if that protection is incomplete.
5. **Convolution-state boundary coverage.** A small control around a block crossing, reusing the snapshot capability,
   checks the actual convolution-state slot and its migration alongside KDA state and bookkeeping, against a matched
   decode reference where feasible. If it needs substantial new instrumentation, it is costed separately and the
   limitation is stated.
6. **Long-prompt control kept.** It is named a history-construction and accumulated-decode-state comparison, not an
   output-length experiment. Selected corrupted pools are reported quantitatively by fixture and layer.
7. **Short stratum S omitted.** A labelled exploratory short-context comparison may later use attempt-2 records, after
   the first-contrast definitions are frozen.
8. **Readings:**

   | Outcome | Reading |
   |---|---|
   | Selected keys and close probabilities both change beyond the null | supported local computational mechanism candidate; behavioural relevance assessed via post-top_p and eligibility; no GPQA explanation |
   | Selected keys change, close probabilities do not resolvably change | no termination-propensity effect detected under this exposure and design; the mechanism is not excluded |
   | Insufficient selected-key exposure | uninformative |

   Direction matters. If D2 increases close propensity, that is consistent with more corrupted old-engine computation
   helping termination. If it decreases it, that supports a harmful D2 effect, and stock 0.7.0 finishing more often
   would then need an interaction or another contributor. "No effect detected by the prespecified screen" replaces
   "equivalent within resolution".
9. **Budget and stopping:**
   - the amended analysis, reference design and cost are frozen before launch;
   - the run stops on structural or long-context instrument failure, and an incomplete contrast stays incomplete;
   - there is no automatic additional attempt, and no full GPQA pass follows from instrument validity or a local logit
     effect.

**Amendment 22a (2026-10-15; operator authorization "proceed", before any first-contrast data).** The first contrast runs
as frozen: first-contrast/SHA256SUMS (186 files) has sha256 2bf5e4b2…, verified by the coordinator, and the DRY_RUN
passed (240 requests, 0 plan failures).
- **Pins:** image ac05230b…, hook 15e93e23….
- **Order:** stage A (V0, LW1, LW2, C1, C2, CV); GATE-1 before any injected batch; stage B (D2, D1, LD2, LD1); then the
  round-3 sentinel and residue checks.
- **Limits:** ceiling 4.29 occupied GPU-hours, single attempt, no retry.
- **Desktop load:** the usual graphics load on GPU 1 (Firefox and the desktop app's GPU process) is present and
  recorded by the pre-flight, within the attempt-2 tolerance.

**Amendment 22b (2026-10-15; first-contrast session result, recorded after the run).** The session stopped at the
structural gate after C1: the client's quick structural check failed for all 8 C1 requests, with no hook errors.
- **Not run:** C2, the CV control, GATE-1 and every injected batch.
- **Reading:** NO READING. The contrast is incomplete and stays incomplete (amendment 22 §9).
- **Cost:** 1.0861 occupied GPU-hours of the 4.29 single-attempt authorization. No retry is permitted under it.
- **Initial observations, before diagnosis:**
  - every request completed;
  - C1-L3 committed 8,196 tokens, with 82,688 cached tokens on an 89,304-token warm prefix (19 whole blocks);
  - several gate inputs (prefix sentinel dumps) were absent because their phases never ran.
- A CPU diagnosis classifies the failure as a harness, plan or cache-accounting defect, or as an engine finding. Nothing
  is interpreted about D2 or termination.

**Amendment 22c (2026-10-15; diagnosis of the first-contrast structural stop, recorded after inspection;
first-contrast/diagnosis-fc1/DIAGNOSIS-fc1.md, CPU only).** No instrument or engine fault was found. Every C1 request
completed, `ids_equal` held 8 of 8, and there were 0 hook errors. The only failing sub-check was
`cached_tokens == cached_expected`.

1. **The plan expected one block too many from the cache** (a plan defect rooted in correct engine behaviour). The
   engine recomputes the last token. DFlash counts as an EAGLE-type speculator, so the last matched block is dropped
   (`GLM53_DFLASH_BOUNDARY_LOOKUP=0`). Prefix hits are whole 4,352-token blocks.
   - **Correct invariant:** cached = max(0, (⌊(P−1)/4352⌋ − 1)·4352). The plan used ⌊P/4352⌋·4352.
   - **Already in our records:** exact repeats in earlier sessions follow the rule (34/34, 17/17, 34/34).
   - **Owned:** the plan was not checked against those records.
   - Cache safety (P6) is unaffected. The prefix-sentinel dump range was placed in the privately recomputed block.
2. **Eviction.** The 7 KDA groups store a state block at every 4,352-token boundary in the shared 1,275-block pool. That
   limits prefix-cache reach to about 0.66 M tokens, against the plan's assumption of about 3.4 M. A warm round is
   0.74 M, so M3's warm prefix was evicted before C1 read it.
3. **The frozen G1-idx-struct page criterion fails even between two cold prefills of the same tokens at this context
   length** (7 pages above 25% gross at the same M4 pages; pooled 0.77%). It was calibrated at short context, so it is
   a calibration defect. As an observation in its own right, deep-layer pooled index-K is more fragile at long context.
4. **C1's forced windows look valid on post-inspection re-check:** WMAX, absolute positions, once-only measurement,
   attribution, all 8 in flight, the clean ledger, M3's regime crossing, and alignment against LW1 (top-1 0.973-0.997).
   C1 is usable as a labelled instrument run, not as a pass. There is no C2, CV or D data.
5. **The CPU tests missed it** because they checked the plan against its own rule using synthetic records. Nobody
   compared the rule with measured cache hits or modelled KDA churn in the shared pool.

**Fixes proposed for review (none applied):**
- **F1:** the corrected cache rule everywhere.
- **F2:** gate cache safety, and require exact counts only where a measurement depends on restored state.
- **F3:** design within the measured reach.
- **F4:** a fixed, staggered submission order.
- **F5:** the prefix sentinel moved to the real last shared block.
- **F6:** index-K page and early-layer criteria made relative to the long-context prefill null.
- **F7:** a CPU replay of the session's requests through the engine's own KV-cache manager before freezing.
- **F8:** the as-run gate kept as a regression fixture.

**Cost and authorization:** the spent attempt cost 1.0861 GPU-hours, and the single-attempt authorization is
exhausted. A new attempt needs a new operator authorization and independent review. Full option B with eviction
designed out is estimated at about 2.77 GPU-hours.

**Amendment 22d (2026-10-15; adopts the independent review "first-contrast retry: review and required preparation",
relayed by the operator; before any retry data).** Proceed with CPU corrections and replay. No launch from the current
cost sketch; no new GPU authorization is implied. The preferred form is one session with an integrated validation gate
followed conditionally by the contrast. B1, the hook and the served configuration stay unchanged unless that proves
infeasible.

1. **F1 accepted, scoped.** The corrected rule applies to exact repeats without eviction; every other case uses the
   image's actual cache lookup. Tests cover exact repeats, extensions of warm prompts, lengths immediately below, at
   and above block boundaries, cold and partially or fully evicted prefixes, and the actual C, D, CV and sentinel
   request forms. They are validated against earlier measured records, not only synthetic ones.
2. **F2 accepted, with construction accounting kept.** Exact safety checks stay: block alignment, no hit beyond the
   permissible prefix, no decode-written history admitted as clean prefill state, and correct restored-state and
   request attribution.
   - Hit, partial hit or miss is recorded for every request.
   - If C and D cache construction differs, that is reported, its distribution across conditions assessed, and
     prefix-reconstruction variability included in the clean null. Such a pair is not described as sharing an identical
     prefix.
   - Cache behaviour must not differ systematically by treatment.
3. **F3: strategy (a), demonstrated as feasible.** The revised proposal shows:
   - the actual request and warm order;
   - live and cached block occupancy by group;
   - which boundary states survive each lookup;
   - expected hits and permitted misses;
   - all eight requests decoding together before the lead-in ends;
   - survival of the shared state the sentinels need.
   Adjacent C/D pairs are tested for reduced churn, without assuming they eliminate it. Batches are not split into
   four-request batches, because D2 exposure depends on concurrency. If eviction cannot be eliminated economically, an
   explicitly designed, balanced partial-cache experiment with costed cold work is acceptable. 2.77 GPU-hours stays
   provisional.
4. **F4 accepted, without claiming determinism.** Actual admission and scheduling order are recorded, and the gates test
   the properties the experiment needs, not agreement with a timing prediction.
5. **F5 accepted.** The sentinel is defined from the actual restored boundary c and the shared physical block
   [c − 4,352, c). Block identity is verified as well as bytes. A missing shared-block measurement is incomplete
   coverage, not a successful sentinel.
6. **F6 accepted.**
   - The old numerical page and early-layer thresholds are replaced by frozen, matched long-context null comparisons,
     reported by fixture, layer and page alongside the pooled results.
   - Exact structural requirements stay: request identity, token position, physical mapping, ledger attribution,
     prohibited writes, and stable committed bytes.
   - A noisy M4 null does not excuse failures elsewhere. Fixture flags and the leave-flagged-out sensitivity stay, and
     the original criteria and failures are kept.
   - The reading is "compatible with measured long-context numerical variation". Planted wrong-page, foreign-request
     and stale-state controls must stay detectable.
7. **F7 required.** A CPU replay through the image's actual cache-manager code first reproduces run-fc1 (the dropped
   block, M3's eviction, the sentinel's private recompute), then exercises the proposed session under plausible
   interleavings and resource pressure, with conservative occupancy bounds where exact prediction is unavailable. This
   is required for future experiments whose validity or costing depends on cache residency, restoration or eviction.
8. **F8 accepted.** The original as-run verdict stays unchanged. Revised checks on C1 are labelled post-inspection.
9. **Integrated gate.** Reference builds and clean checks come before injected traffic. The session stops early on
   unsafe restoration, insufficient concurrent decoding, a long-context alignment failure, uninterpretable numerical
   content, or missing boundary or shared-prefix coverage. The ordering is repriced from measured rates and expected
   misses.
10. **C1's permitted use:** a labelled long-context instrument-characterization run (forcing, position and attribution,
    alignment, observed crossings, numerical calibration). It is not a clean-repeatability estimate, not pooled with or
    paired against new runs, and not evidence about D2.

**Deliverable before GPU authorization:** one revised package with the feasible schedule; the CPU replay against
historical records and the proposed workload; cache-construction and concurrency acceptance rules; frozen long-context
numerical criteria; sentinel and boundary coverage; measured-rate costing including misses and early-stop cost; and a
new explicit ceiling and attempt policy.

**Amendment 22e (2026-10-15; retry package prepared under 22d, CPU only; for independent review, not authorized).**
first-contrast/retry/: SHA256SUMS (287 files) has sha256 c7374d75…; changes R-1 to R-9 are in CHANGES-R.json. B1, the
hook and the served configuration are unchanged (image ac05230b…, hook 15e93e23…).

- **F7 replay.** The image's own scheduler, KV-cache manager, coordinator, per-type managers and block pool, run in a CPU
  container of the serving image.
  - It reproduces run-fc1 exactly: 5,609 scheduler steps including block ids and free counts, all 32 cached counts, the
    one-block drop, and the sentinel's private-recompute location.
  - It matches historical records: b0-mapguard run-a3 at all 2,482 steps; 253 measured cached counts across 5
    sessions. Four deliberately wrong configurations are each detected.
- **Correction to 22c.** M3's warm state survived the warm phase. It was evicted by C1's own earlier admissions, because
  the HTTP race admitted M3 fifth. The replayed counterfactuals agree: recorded order gives the miss, warm-completion
  order gives 8/8 hits, reverse order gives 2 misses. The 22c statement that the warm round exceeded the cache reach is
  superseded.
- **F1:** c = max(0, (min(⌊P_build/4352⌋, ⌊(P−1)/4352⌋) − 1)·4352), for same-stream, same-salt builds without eviction;
  the image's lookup decides elsewhere. 70/70 tests against the image's lookup, and all 253 measured counts match.
- **F3 feasible schedule (strategy (a), configuration unchanged).**
  - Each salt is built in two halves of 4 prompts, each followed by a sequential touch pass.
  - LP runs reuse the primary builds. C2/D2 share s2 across the gate; D3/C3 share s3.
  - Construction is balanced by arm, and there are no 4-request batches.
  - Every lookup is a full hit in 25 replayed scenarios (orders, staggers, jitter, pools of 1,100-1,275 blocks, a long
    gate wait), with a survival margin of at least 359 blocks.
  - All 8 requests decode by output index 95 (159 for LP), and the sentinels' shared state survives with at least 438
    blocks of margin. No partial-cache design is needed.
- **F2:** exact safety gates (block-aligned; within prefill coverage; no decode-written block restored; num_computed
  equals cached, with attribution from the scheduler log). Hit, partial, miss or cold recorded per request. On stage A
  every lookup must hit, or the session stops. C and D (LC and LD) must have equal non-hit counts. "Identical prefix" is
  used only when block identity is verified.
- **F4:** a fixed staggered order, with actual admission recorded; gates test properties.
- **F5:** sentinels PS-a, PS-b and PS-c dump exactly [c − 4,352, c), checking block identity, MLA and index-K bytes and
  the KDA boundary state. A missing measurement stops the session.
- **F6:** per page ≤ max(2 × the matched prefill null on that page, 0.25); layers 3-15 ≤ max(2 × prefill null, 0.1%);
  the pooled rule keeps both nulls; the originals are reported beside.
  - **Recorded pre-freeze change:** the page and early-layer rules use the prefill null only. A synthetic planted page
    fault slipped through when the decode null (which contains the faulty run) was included.
  - **Calibration on run-fc1 (post-inspection):** C1 passes (pooled 0.765% against 1.54%; 0 of 2,486 pages; 3 of
    57,344 early pools); the originals still fail and are kept beside. All 6 planted faults are detected.
- **F8:** the frozen attempt-1 gate reproduces gate-asrun exactly (27 rows, FAIL).
- **Gate order:** pre-traffic geometry; per phase structural and unsafe-restoration stops; an all-8 lead ≥ 512 per batch;
  GATE-1R before any injected traffic; the sentinel after injected traffic.
- **Cost:**

  | Variant | Central | Pessimistic = proposed ceiling |
  |---|---|---|
  | B | 3.00 GPU-hours | 4.18 |
  | A0 | 2.75 GPU-hours | 3.72 |

  Early-stop costs (central/pessimistic): 0.42/0.46 before traffic; 1.19/1.47 after C1; 1.75/2.34 if GATE-1R fails.
  The recommended attempt policy is single.
- **Tests:** F1 70/70; units 25/25; tampered plans 19/19; runner refusals 23/23; synthetic dry runs 27/27; F6 planted
  faults 6/6; F8 exact; DRY_RUN B and A0 pass.
- **Limits:**
  - KDA block ids are not in the guard's record, so the sentinel certifies KDA state by restored length and snapshot
    bytes;
  - the EAGLE flag on the KDA groups is not identifiable;
  - the long-context gates have never seen an in-session decode null;
  - the replay's decode acceptance comes from C1 alone;
  - GATE-1R idles the GPUs about 270 s.

**Amendment 22f (2026-10-15; adopts the independent review "retry review: recommended disposition", relayed by the
operator; before any retry data).**

1. **Deliverable substantially complete.** Eviction is described as "avoided in the replayed scenarios, with measured
   survival margins"; the actual-session gates remain authoritative. The corrected account of M3 is accepted.
2. **Sentinel analysis correction before launch.** The SSM and convolution components are evaluated separately, each
   with its own PS-a/PS-b numerical baseline, and each PS-c comparison must meet its own criterion. Only the intended
   committed state and relevant convolution entries are compared. Negative controls cover convolution-only corruption,
   SSM-only corruption, and a non-identical baseline followed by excessive convolution change. This is an analysis
   change only: no hook change and no GPU revalidation.
3. **Acceptance stress scenarios (CPU):** single-token advance throughout, maximum accepted advance throughout, and
   strongly unequal acceptance between fixtures. Each checks cache survival and that all eight enter decode before the
   lead-in ends, with any material runtime effect in the cost. These are stress scenarios, not predictions of D2.
4. **F6 accepted** (prefill-only page and early-layer nulls). A page whose null makes its threshold non-discriminating is
   reported as such and not called independently verified. A pass means compatibility with measured long-context
   variation.
5. **KDA sentinel accepted** after correction 2. Stated limits: MLA/index-K shared-block identity is directly checked;
   KDA restoration and captured-state consistency are checked; physical KDA block identity and the complete unsampled
   state are not established.
6. **LP restore depth.** The primary contrast is unaffected, but LP attribution is limited. Added: a descriptive
   same-depth comparison (LC1 against LD1, LC2 against LD2), with LC3 against LD3 reported separately as depth-mismatched.
   The three-repeat rule is not applied to the two matched pairs. If the LP conclusion depends on the mismatched pair, it
   is reported as unresolved construction sensitivity.
7. **Recommended GPU scope:** variant B, ceiling 4.18 occupied GPU-hours, single attempt, existing early-stop gates, no
   automatic retry. It proceeds after the CPU corrections, stress scenarios and an updated freeze pass, subject to
   operator authorization.
8. **Interpretation stays bounded.** A valid session can establish whether decode-window D2 exposure changes selected
   keys and close propensity on these histories, not that D2 explains the completion gap. C1 stays outside the primary
   comparison.

**Amendment 22g (2026-10-15; operator authorization "go with variant b", before any retry data).** The first-contrast
retry is authorized as variant B, ceiling 4.18 occupied GPU-hours, single attempt, existing early-stop gates, no
automatic retry. It is conditional on the 22f CPU corrections and stress scenarios passing, and on the coordinator
verifying the re-frozen package. If the stress scenarios show the ceiling does not hold, it returns to the operator
before launch.

**Amendment 22h (2026-10-15; operator decision "both", before any retry data).**

The 22f stress scenarios (R-11) showed the 4.18 ceiling does not cover a complete session under single-token or
strongly unequal draft acceptance:

| Acceptance | Central / pessimistic GPU-hours |
|---|---|
| Single-token | 4.33 / 6.08 |
| Strongly unequal | 4.40 / 6.18 |

Under 4.18, the session would stop before C3 at central rates, or be killed partway through D1 at pessimistic rates.
Cache safety and all-eight concurrency held in all 10 scenarios.

The operator chose both remedies:
1. **The ceiling is raised to 6.2 occupied GPU-hours** for the single retry attempt (variant B, no automatic retry).
2. **A batch guard is added to the runner and client:** a batch starts only if its projected completion fits in the
   remaining budget and wall time, using the decode rate measured in the session's earlier batches with a conservative
   margin. A batch is never cut mid-way. A stop by the guard is recorded as an incomplete contrast.

The guard is a runner and client change only. The design, order, gates and analysis reviewed in 22f are unchanged, and
the hook, image and served configuration are unchanged. The package is re-frozen and coordinator-verified before
launch.

**Amendment 22i (2026-10-15; frozen retry package, verified by the coordinator, before any retry data).**
- **Package:** first-contrast/retry/SHA256SUMS (381 files) has sha256 fca158ca…; the attempt-1 imports verify.
- **Changes:** R-16 (ceiling 6.2 occupied GPU-hours, wall cap 11,160 s) and R-17 (batch guard). The guard projects
  each batch at 1.25 × (uncached prefill ÷ the slowest measured prefill rate + max_tokens ÷ the slowest measured decode
  rate) + 30 s. Before the first batch of a form it uses a prior of 2,410 tok/s prefill and 11.0 tok/s decode. A
  primary batch that does not fit stops the session (rc 17, "incomplete contrast (batch guard)"). LP batches keep the
  later primary phases in reserve and are otherwise skipped. The wall-cap watchdog is the only mid-batch cut.
- **Stress replay under 6.2 with the guard:** the primary contrast completes in every scenario, with no mid-batch cut
  and no guard stop. The LP control is incomplete only under pessimistic single-token or unequal acceptance, where LD
  batches are skipped. Normal acceptance: 3.00 GPU-hours central, 4.18 pessimistic, both complete.
- **Tests:** units 39/39, runner refusals 27/27, synthetic dry runs 33/33, tampered plans 19/19.
- **DRY_RUN:** variant B, policy single, approved 6.2: passes. The runner is invoked through bash, because the frozen
  file is not executable.
- **Launch:** under amendments 22g and 22h.

**Amendment 22j (2026-10-16; first-contrast retry result, recorded by the coordinator after the session).**
- **Session `run-fcr1`:** variant B, single attempt, 2026-10-16 00:54-02:22 UTC, **2.9317 occupied GPU-hours** of the
  6.2 approved. Every planned phase ran (224 requests); no batch-guard stop, no LP batch skipped, no wall-cap cut.
- **Passed:** GATE-1R (in session, before any injected batch, and recomputed); all 12 runs valid; injection checks
  H5a-fc and H5g-fc; R-cache-safe (224 requests), R-cache-balance (no non-hit fixture-run in any arm); sentinel
  coverage, block identity and bytes (7,480 MLA and index-K pages, 0 mismatches); C3 residue.
- **Failed:** H5f-fc (ledger vs bytes), R-PS-state-ssm and R-PS-state-conv (sentinel boundary state).
- **Frozen verdict: NO READING.** The results below are instrument findings only. No statement about D2 and
  reasoning close propensity is drawn from this session.
- **Descriptive numbers, not a reading:** Y_C −11.610, −11.710, −11.675; Y_D −11.561, −11.549, −11.569; Δ 0.106 nats,
  within-arm null N 0.100, beyond-null threshold 0.200: the beyond-null screen is not met; the three D runs lie on one
  side of the three C runs. Selection: injected pools were selected (exposure met); selection divergence did not exceed
  the within-arm null. LP same-depth pairs (descriptive): −0.003 and −0.009 nats; depth-mismatched pair −0.020. The
  analysis prints a direction line in every outcome; under NO READING it is not a finding.
- **Post-hoc diagnostics (written after the data; they do not change the verdict)**, in `retry/diagnosis-fcr1/`:
  - *H5f-fc.* Only 7-11 window pools per layer are clean-labelled in each D run (77-121 per run, against about
    4,000 corrupt-labelled). Every clean-labelled gross pool is fixture M4, window pool 3 (once pool 5), at layers 23, 27
    and 43, in all three D runs. The same pools are gross in the uninjected C2 and C3. M4 is the fixture already
    reported as having non-discriminating F6 pages (22f s.4). The limit (1.51 %, a pooled rate over all window pools)
    was applied to a small, non-random subset that includes those pools.
  - *Sentinel state.* The gate's failing list is capped at 10; the full counts are 3 of 544 SSM rows and 13 of 544
    convolution rows. In the 48 rows per component where PS-a and PS-b are bitwise equal, PS-c is bitwise equal too.
    In the other 496, the median ratio of the PS-a/PS-c to the PS-a/PS-b difference is 0.98 (SSM) and 0.93 (conv). The
    failures sit where PS-a and PS-b happened to be unusually close. The mirror comparison under the same 2× rule (PS-b
    farther from PS-a than 2× PS-c) fires 2 times (SSM) and 22 times (conv), on two phases that both ran before the
    injected batch. A limit set from one baseline pair per row does not separate PS-c from the baselines' own spread.
- **Not established:** whether either gate would pass under a redesign; no redesign is adopted here. Any further
  session needs a new dated amendment before data, as before.

**Amendment 22k (2026-10-16; adopts the independent review "disposition after first-contrast retry", relayed by the
operator, before any further work).**
- **Decision:** the full contrast is not repeated solely to repair the failed gates. The frozen NO READING verdict, the
  descriptive measurements, the post-hoc diagnoses (labelled as such) and the verified B1 checkpoint and corrections are
  retained. This experiment line closes as **unresolved**, not as evidence against full-history D2 effects.
- **Why no repeat now:** the termination contrast (0.106 against 0.200 nats) and selection divergence (not beyond the
  within-arm null) were also unmet. They stay descriptive, because structural gates failed, but they give no compelling
  signal that a near-identical repeat would resolve the mechanism. Complete separation of three runs per arm does not
  override those screens.
- **What the session establishes.** *Operationally:* the long-window execution and cache schedule worked; every planned
  request completed; forcing, attribution, injection placement and cache safety passed; the measured shared MLA /
  index-K pages were unchanged; the remaining numerical criteria were inadequate or unresolved. *Scientifically:* no
  supported D2-to-termination reading; no exclusion of accumulated full-history D2 effects; no explanation of 0.7.0's
  completion advantage.
- **Bounded CPU review (existing records only), to test whether the failures fit the proposed explanations:**
  - *H5f, like with like:* the exact clean-labelled fixture / rank / layer / pool units, compared with the same units in
    the clean arm and in the prefill reference; counts and identities, not percentages alone; M4's earlier fragility
    flags kept. No pool is removed and H5f is not dropped because it failed; H5g (placement) does not replace the
    content check.
  - *Sentinel:* the original component-wise failures retained; exact-equality cases, per-component distances, both
    baseline orientations, fixture and layer concentration, and whether failures coincide with unusually small baseline
    denominators. The mirror comparison is evidence that the single-pair ratio gate is unstable, not proof that
    post-injection contamination was absent. No retrospective PASS is sought.
- **Instrument:** a prospective validation design is documented for future use (exact structural checks; content
  controls matched to the judged subset; multiple clean sentinel baselines; separate SSM and convolution limits;
  planted-fault sensitivity; direct inspection of cached state before and after treatment where feasible). It is not
  implemented or GPU-validated until a specified experiment needs it.
- **Next work targets a different uncertainty:** a zero-GPU ranking of mechanisms that differ between releases and
  matter for long decode, starting with 0.7.0's ring-4 / depth-5 rejected-draft defect (the index-K tail ring of 4
  entries against a 6-token verify window at 5 drafts, fixed in engine 0.9.0) and its atomic-slot top-k merge.
  Each needs a concrete mechanism, evidence of exposure in served execution, a discriminating measurement,
  a costed experiment, and a result that would change the next decision. Neither arm is launched automatically. If no
  economical mechanism experiment emerges, the decision returns to funding a stock 0.7.0 replication or stopping work
  on the historical gap.
- **Reporting:** verified correctness defects are reportable independently of the unresolved completion question, each
  separating the demonstrated defect and correction, the verified serving coverage, and the unestablished completion
  impact. Publication stays the operator's decision.

**Amendment 22l (2026-10-16; zero-GPU mechanism ranking and operator decision, before any further GPU work).**
- **Ranking** (`mechanism-ranking/`, CPU only; the tail-ring harness of `tail-fix/two-rank/` with configuration
  overrides, on the 0.7.0, 0.9.1 and B1 images):
  - *Ring-4 / depth-5 rejected-draft defect (0.7.0, fixed in 0.9.0):* at acceptance 0.72 and 1, 2 and 8 requests, 0.7.0
    as released writes 100% of decode-completed pools wrong with or without it, because of D2. With positions passed
    (D2 removed) it leaves 57.5 / 52.5 / 45.7% wrong. At 8 requests, 3 of 972 wrong pools carry its signature
    [M harness]. Step 5 removed it from 0.7.0 with D2 kept and measured no change in disagreement [M]. Criterion 5 not
    met: not funded.
  - *Atomic-slot top-k merge (0.7.0):* real and launch-dependent, inside the column-511 class B1 corrects; that
    correction alone was null on empty replies (15 vs 16). Criterion 5 not met: not funded.
  - *Observation added as an open question [I, with M harness]:* 0.9.1's kernels under 0.7.0's DCP2 mapping write 100%
    of decode pools wrong at one request, so D2 exposure across steps 4-6 matches their disagreement split. Step 6's
    rise may not be separate from D2. Untested; it says nothing about completion.
  - The harness's scheduler, keys and drafts are synthetic; its percentages describe exposure, not served outcomes.
- **Operator decision:** "publish and proceed", taken as the replication route the ranking leaves. The stock 0.7.0
  replication draft (expected 22.97 GPU-hours, valid-run ceiling 26.45, maximum 35.0 as drafted before the long-reply
  framing) is re-frozen: seed policy (recorded or fresh), the long-reply finishing secondary and its power, and
  current tooling. It goes to independent review and a GPU authorization before any data.

**Amendment 22m (2026-10-16; stock 0.7.0 replication package re-frozen for review, before any GPU use).**
- **Package:** `rep070/refreeze/` (SHA256SUMS sha256 29a2fb11…; the 2026-10-11 draft verifies unchanged). Nothing run
  on the GPUs; no file of the new run tags exists.
- **Re-verified against the current machine:** image by digest, weights, draft and template blobs, stack and recipe,
  lm-eval install and task, proxy, host kernel / driver / runtime, port binding on 127.0.0.1. Draft correction: the
  stack's start-up script carries a one-line TileLang cache addition dated 2026-09-05, present in both record sessions.
  The compile cache gained 61 files after the record ended, none since the draft.
- **Added:** a gating warm-up check (specified in the draft, not implemented there) and a configuration-identity check
  against the record's passes 2-3 server log; an eval-chain copy differing from the original in exactly the seed list
  and log name (frozen diff).
- **Measured coupling [M]:** across 24 pairs of existing passes, no same-seed pair shares a byte-identical answered
  reply, but the correlation of log reply length over questions is 0.66-0.91 for same-seed pairs against 0.37-0.48
  for different seeds. Same-seed passes are not independent draws.
- **Seed policy, for decision:** R = recorded seeds 1234 / 1235 / 1236 (does this record repeat?) or F = fresh seeds
  70701 / 70702 / 70703, unused on this machine (does the engine leave fewer empties?). Under the power model with a
  coupling sensitivity, if 0.7.0 truly behaves like the 0.9.1 record, R reads O2 with probability 0.88 / 0.70 / 0.44
  at no / one-quarter / one-half inherited outcomes; F is not exposed to this. Same cost.
- **Secondary (descriptive, never the reading):** P(finish | still generating at 100,000 tokens). Power about 0.59
  (new run finishing like the old 0.7.0 record) and 0.41 (like 0.9.1), intervals 0.5-0.65 wide.
- **Budget, for decision:** expected 22.97, valid-run ceiling 26.45, maximum 35.0 GPU-hours (option A, as drafted).
  If the run leaves 0.9.1-like numbers of capped replies, expected cost is 26.4-31.2 and up to 11% of such runs would
  reach a 35.0 watchdog, so A is most likely to cut off the runs that would read O2. Option B: valid-run ceiling 36.5,
  maximum 50.0; or B-lean: watchdog 36.5, no pre-authorized invalid run.
- **CPU checks:** dry runs O1 / O2 as drafted and a fresh-seed relabel valid; 6 negative cases invalid; runner refusals
  17 / 17 (Docker read-only); session simulation 42 / 42 (one runner logging fault found and fixed before freezing).
- **Not determinable:** the record's pass-1 configuration and warm-up; coupling strength for empty replies
  specifically.

**Amendment 22n (2026-10-16; adopts the independent review "stock 0.7.0 replication: review disposition", relayed by
the operator; package changes before any GPU use).**
- **Recommended choices:**
  - seed policy F (request seeds 70701 / 70702 / 70703);
  - budget B-lean (36.5 occupied GPU-hours, no pre-authorized invalid-run retry);
  - the frozen three-pass analysis and outcome rules retained.
  Operator authorization is still required.
- **Seeds and task:**
  - Only the request sampling seed changes. lm-eval's process/task seed stays 1234 + pass (1235 / 1236 / 1237).
  - Unchanged, and checked against both records as gating checks: task hash 87e807b3b27c8661, rendered prompts and
    answer order, question / target / doc / message hashes, sampling, stopping, parser, concurrency 8, cap 327,680.
  - Both seeds are recorded per pass.
  - Session 1 = pass 1 and session 2 = passes 2 then 3, as in the record. Known differences: the record's pass 1 had
    no proxy, its first chain was interrupted and re-attached, and the publish address is now loopback.
- **Budget:** watchdog at 36.5. Estimated P(total ≥ 36.5): 0 under the old 0.7.0 record's rates; 0 or 0.044 under
  0.9.1-like rates (empirical or token-based slope). P(session 2 not started) is 0 in every scenario. Seeds, cap and
  requests are never changed to fit the budget.
- **Watchdog interruption:**
  - Completed passes and partial requests are preserved.
  - The run is labelled INCOMPLETE with its cause.
  - Completed, protocol-cap, interrupted (started, no response) and unstarted requests are counted separately, from a
    package-local proxy copy that adds a start / interruption sidecar. The main request log stays byte-compatible.
  - Interrupted or unstarted requests are never counted as empty replies, and no 594-based classification is applied.
  - No automatic restart or replacement run.
- **Category meanings (rules unchanged):**
  - O1: the fresh-seed run sustains a low empty count and separates from the historical 0.9.1 comparator under the
    specified test. This is not equivalence to the original 0.7.0 rate, nor a contemporaneous causal engine advantage.
  - O2: the original low-empty result is not sustained under the rules. This does not necessarily mean no advantage
    over 0.9.1 remains.
  - O3: inconclusive.
  - Both comparisons, counts, effects and uncertainty are always reported. The sign-flip tests assume
    historical-comparator and question-level exchangeability; "exact" refers to the calculation under those
    assumptions. The coupling power figures are sensitivity assumptions, not measured dependence of empty outcomes.
- **Tail secondary:** descriptive. It is reported with numerator, denominator and question-clustered intervals. It
  cannot adjudicate a termination mechanism, and reaching 100k selects different trajectories on each engine.
- **Durable records (local, never published):**
  - full request and response bodies (reasoning and final text, finish reasons, usage), mode 0600;
  - timings; whole server logs, including admission, concurrency and preemption lines;
  - configuration and response hashes;
  - a compile-cache inventory before session 1 and a diff after each session (cache not cleared).
  Generated token ids are not retained: the 0.7.0 image returns them only on a request flag, and requests are not
  changed. No engine hook. Scoring is unchanged.
- **Prespecified next steps:**
  - O1: further investigation, with a contemporaneous comparison considered before another broad mechanism search.
  - O2: reassess whether the 4-versus-16 contrast remains an appropriate target; the verified fixes keep their value.
  - O3: report the uncertainty and make a new budget decision; no automatic extra passes.
- **Package:** rep070/refreeze/ SHA256SUMS sha256 131e72ea… (110 files; 33 changed files with `.pre22n` copies and
  before/after hashes in CHANGES-22n.json; 12 new). Checks:
  - runner refusals 18/18;
  - proxy copy 17/17;
  - incomplete-run cases 14/14;
  - session simulation 61/61;
  - chain seed 6/6;
  - dry runs D1 O1, D2 O2, D3 O2 valid, and N1-N8 invalid (including a wrong task hash and a target mismatch);
  - DRY_RUN of F at 36.5 passes and creates nothing.

**Amendment 22o (2026-10-16; operator authorization "yes, launch, use the gpu", before any replication data).**
Variant F (request seeds 70701 / 70702 / 70703), up to 36.5 occupied GPU-hours, single attempt, no retry, package
rep070/refreeze/ SHA256SUMS 131e72ea… as frozen in 22n.

**Amendment 22p (2026-10-17; stock 0.7.0 replication result, recorded by the coordinator after the run).**
- **Run:** variant F, two sessions as frozen (2026-10-16 06:16-18:14 local), **23.926 occupied GPU-hours** of 36.5
  (session 1 6.693, session 2 17.233). No watchdog stop, all 594 requests HTTP 200 with the variant's seeds, warm-up
  and configuration identity with the old 0.7.0 record's passes 2-3 server log confirmed, no preemption, no illegal
  memory access or NaN warning lines, compile cache unchanged apart from one file in session 1. Every frozen validity
  check passes: **valid**.
- **Reading: O3 (inconclusive)** under the frozen rules.
  - Empty replies 10 of 594 (2 / 4 / 4 by pass): old 0.7.0 record 4, 0.9.1 record 16.
  - C1 (fewer than the 0.9.1 record): one-sided exact sign-flip p = 0.122; reduction +1.01 points (95% question
    bootstrap −0.34 to +2.53).
  - C2 (more than the old 0.7.0 record): one-sided p = 0.108; increase +1.01 points (95% −0.17 to +2.36).
  - Both tests assume historical-comparator and question-level exchangeability; "exact" refers to the calculation.
- **Descriptive:** raw (flexible) accuracy 87.2% (518 / 594) against 86.9% (0.9.1 record) and 88.2% (old 0.7.0
  record); the stated figure waits for the hand read. No identical answered reply with the old record in any pass;
  one question-pass empty in both.
- **Secondary (descriptive, never the reading):** P(finish | still generating at 100,000 tokens) 19 / 29 (18
  questions; 0.66) against 10 / 25 (0.40) for the 0.9.1 record and 10 / 14 (0.71) for the old 0.7.0 record's passes
  2-3. Differences: −0.06 (95% −0.35 to +0.27) vs old 0.7.0, +0.26 (+0.01 to +0.53) vs 0.9.1. About 10-19 events
  per record; reaching 100k selects different trajectories on each engine, so this is not a causal engine effect.
- **Prespecified next step (O3):** report the uncertainty and make a new budget decision; no automatic extra passes.

**Amendment 22q (2026-10-17; stated figure final; operator decision after O3).**
- **Hand read** (protocol): all 13 replication rows where the stated answer differed from flexible-extract were read.
  3 commit to no single option (passes 1 and 3 of one question, pass 2 of another) and are scored incorrect as stated.
  Final stated accuracy 87.9% (522 / 594), against 87.9% for the 0.9.1 record and 89.4% for the old 0.7.0 record
  (`rep070/run-F/result-rep070-F-final.*`). The reading is unchanged: O3.
- **Operator decision:** the historical-gap line (4 versus 16 empty) closes as inconclusive. The replication is reported;
  the verified correctness fixes keep their value; upstream reporting stays the operator's decision.
- **New question, taken up separately:** what drives completion of long replies (P(finish | still generating at
  100,000 tokens)), the measure on which 0.7.0 still looks different descriptively. Discovery comes first, zero GPU;
  any experiment is proposed, reviewed and authorized before data.

**Amendment 22r (2026-10-17; long-reply completion: discovery result and operator choice, before any GPU use).**
- **Discovery** (zero GPU; `long-reply/discovery/`, definitions hashed before any metric was computed): on the
  replication's 29 replies past 100,000 tokens:
  - 8 of 10 capped replies end in an exact loop (period 2-1,001 tokens); 2 do not loop. Only 2 were already repeating
    at 100k; the rest began 43k-188k tokens later. No capped reply recovered from a terminal loop.
  - 15 of 19 finished replies produce new text to the end of their reasoning.
  - Capped replies show lower characters per token in their first 100k tokens (2.31 against 3.15), before any
    repetition.
  - Onset positions show no clustering on the attention block, prefill chunk or powers of two.
  - On the 14 questions long in both engine families, the 0.7.0 family finished 23 / 36 long replies against 18 / 58
    for the 0.9.1 family: the finishing gap is not explained by which questions go long. All descriptive.
- **Open:** is the outcome set by the state a trajectory has reached at 100k, or by how the engine continues from it?
- **Operator choice:** option B+. 12 identical 0.7.0 prefixes cut at 100k generated tokens, 4 seeds each, continued
  +64k tokens on stock 0.7.0, stock 0.9.1 and B1 (about 8-10 GPU-hours, upper bound). The design, analysis and costing
  are frozen in a package, reviewed independently and authorized before any data.

**Amendment 22s (2026-10-17; B+ fixed-prefix continuation package frozen for review, before any GPU use).**
- **Package:** `long-reply/bplus/` (SHA256SUMS sha256 a71aa141…, 85 files); 16 / 16 CPU checks.
- **Prefixes:** the 29 replication replies past 100k, kept if R50 at 100k is below 0.05; one reply per question;
  questions ranked by panel long count; ties broken by salted sha256.
  - originally capped: q79p2, q88p1, q69p2, q120p3, q8p2, q158p3;
  - originally finished: q127p2, q147p1, q81p2, q71p1, q186p1, q145p2.
  Each is cut at exactly 100,000 reasoning tokens and re-tokenized from text, because the original ids were not kept.
  Every engine receives the same token ids through `/v1/completions`, gated by hash.
- **Continuation:** 4 distinct seeds per prefix per engine, +65,536 tokens, temperature 1.0, top_p 0.95, 8 in flight.
  Fresh server per engine, in the order stock 0.7.0, stock 0.9.1, B1. Prefix caching as shipped. Serving and sampling
  source is byte-identical across the three images; stop ids 154820 / 154827 / 154829; `</think>` does not stop.
- **Known asymmetries (part of "engine", not controlled):**
  - prefix KV is rebuilt by prefill on every engine;
  - D7 in the prefix prefill: 0.7.0 mean 0.14% of a prefix's index-K pools; 0.9.1 0.08% in 32 of 48 continuations;
    B1 repaired;
  - layout: 0.7.0 DCP2 / EP2 / DFlash ×5; 0.9.1 and B1 DCP1 / ×3.
- **Primary:** P(finish within +64k) per engine, prefix as the unit.
  - *Engine effect supported:* 0.7.0 − 0.9.1 ≥ 0.15, 95% interval excluding 0, and sign-flip p ≤ 0.05. Power 0.70-0.81
    at the full discovery gap, about 0.2 at half the gap; false positive ≤ 0.016 with no gap.
  - *State-dominated:* all three contrasts within ±0.20 (90% interval) and the prefix explaining ≥ 30% of outcome
    variance.
- **Cost:** central 15.8, pessimistic 28.4, ceiling 31.6 GPU-hours; P(ceiling) < 0.1%. The discovery estimate (7.8)
  assumed 16 in flight; 16 in flight would be about 11.8 central. A frozen fallback reads the primary contrast within
  about 23.
- **Open, not verified on CPU:** actual prefix-cache hits; 0.7.0's long-prompt prefill rate; decode rates at 8 ×
  100-165k contexts.

**Amendment 22t (2026-10-20; operator authorization "full send with the 31.6 ceiling", before any B+ data).**
The B+ package of 22s (SHA256SUMS a71aa141…, verified at launch, DRY_RUN passed) is authorized at up to 31.6 occupied
GPU-hours, single attempt, no retry. By the operator's decision it launches without a separate independent review of the
package; the frozen design, thresholds and analysis are unchanged from 22s.

**Amendment 22u (2026-10-21; B+ run completed; frozen verdict INVALID on one configuration gate; recorded before any
outcome was read).**
- **Run:** all three sessions complete, 144 / 144 requests HTTP 200, no preemption, 9 warm-ups per session, loopback
  publish. **13.064 occupied GPU-hours** of 31.6 (v070 5.531, v091 3.879, b1 3.654). The server renderings of all 12
  prefixes equal the frozen token ids, and the first record was checked against the plan (prompt hash, seed, parameters).
- **Frozen verdict: INVALID, no reading.** The only failed check is that the v070 server log's KV pool must equal the
  reference record's:
  - this run reported 2,768,240 tokens against 2,758,919;
  - "Available KV cache memory" was 15.78 GiB in both;
  - 2,768,240 is exactly the old 0.7.0 record's own pass-1 session value (2026-09-29). The rep070 comparator audit
    listed this difference as an existing run-to-run variation, "not gating";
  - peak KV use at 8 × about 165k tokens stays far below either pool.
- **No outcome has been read.** Whether to amend this gate goes to the operator, decided blind to outcomes.

**Amendment 22v (2026-10-21; operator decision "first option": KV pool gate amended, decided before any outcome was
read).**
- **The change:** the v070 KV pool gate accepts either size the old 0.7.0 record itself produced: 2,758,919
  (2026-10-10 and the rep070-F replication) or 2,768,240 (2026-09-29). Every other check, threshold, screen and
  analysis step is unchanged.
- **Implementation:** `scripts/analysis_bplus_22v.py` (sha256 ed915a07…), a copy of the frozen `analysis_bplus.py`
  (613019d9…) differing only in this gate. The frozen file and its INVALID result (`run/result.md`) are kept. The
  amended analysis writes `run/result-22v.*`.
- **Labelling:** any reading is reported as read after a disclosed gate amendment.

**Amendment 22w (2026-10-21; B+ result under the 22v gate amendment).**
- **Valid** under 22v; every other check unchanged and passing. **Reading: R-I, inconclusive** under the prespecified
  rules.
- **Primary, P(finish within +64k), prefix as the unit:** stock 0.7.0 0.417 (20 / 48), stock 0.9.1 0.417 (20 / 48),
  B1 0.521 (25 / 48).
  - C1 (0.7.0 − 0.9.1): +0.000 (95% −0.083 to +0.083; sign-flip p 1.00), screen SMALL.
  - C2 (B1 − 0.9.1): +0.104 (95% +0.000 to +0.208; p 0.19).
  - C3 (0.7.0 − B1): −0.104 (95% −0.208 to −0.021; p 0.13).
  - Prefix share of outcome variance η² 0.67 (PREFIX_HIGH).
- **Why not R-S:** C2 and C3 miss the SMALL rule (|mean| ≤ 0.10) by 0.004. Their 90% intervals lie inside ±0.20.
- **Descriptive:**
  - originally capped prefixes finish 0.13 / 0.17 / 0.29, originally finished prefixes 0.71 / 0.67 / 0.75 (0.7.0 /
    0.9.1 / B1);
  - loop onsets in the continuation: 5 / 6 / 12 of 48;
  - within-prefix ICC 0.62 / 0.57 / 0.71;
  - decode 34.7 / 48.4 / 48.1 tok/s.
- **Caveats:** one 0.7.0-built state per question; prefixes rebuilt by prefill (decode-built state absent); re-tokenized
  ids; "engine" bundles decode path, D2 / D7 exposure, layout and speculation depth.
- **Cost:** 13.064 GPU-hours (v070 5.531, v091 3.879, b1 3.654).

**Amendment 22x (2026-10-21; operator choice after B+: option D, before any GPU use).**
- **Design:** one stock 0.9.1 GPQA Diamond pass on the standard protocol (198 questions, 8 in flight, cap 327,680,
  temperature 1.0, top_p 0.95, task hash 87e807b3…) at a fresh request seed. Full request and response bodies are kept
  locally, as in the rep070 replication.
- **Analysis:** the replies are run through the discovery pipeline unchanged (definitions hashed in
  `long-reply/discovery/PRESPEC.md`).
- **Question:** do 0.9.1's long replies reach the states that 0.7.0's capped replies showed before 100k (lower
  characters per token, more interjections), and are 0.9.1's capped replies the same exact loops?
- **Also:** the pass supplies 0.9.1-built prefixes for a possible prefix-source × engine cross-continuation.
- **Descriptive only.** Package, costing and checks are frozen before authorization.

**Amendment 22y (2026-10-21; option-D package frozen, before any GPU use).**
- **Package:** `long-reply/d091/` (SHA256SUMS sha256 b558b584…, 147 files).
- **Run:** stock 0.9.1 (image a9e985c0…), one pass, request seed 70704 (checked unused across request logs, results,
  published rows, a text scan and the raw-body files), lm-eval process seed 1235, standard protocol.
- **Gates:** identity gates accept only values the 0.9.1 record's own two sessions produced; calibrated by running the
  gate code on those sessions and on B+'s 0.9.1 session (12 / 12 pass on all three). The KV pool is reported, not
  identity-gated, with a floor of 2,646,016 tokens (8 requests at the cap).
- **Analysis:** discovery's `trajectory.py`, `periods.py` and B+'s `build_prefixes.py` run byte-unchanged. Their
  rep070-specific input stages run through adapters proven to reproduce the originals' outputs byte for byte.
- **Cost:** central 5.15, pessimistic 7.9, ceiling 10.0 GPU-hours; P(ceiling) 0.004% central, 6% at twice 0.9.1's cap
  rate.
- **Checks:** analysis dry runs 17 / 17; discovery pipeline on rep070-F 12 / 12 outputs byte-identical; runner refusals
  22 / 22; proxy 17 / 17; session simulation 67 / 67; privacy scan clean; DRY_RUN passes.

**Amendment 22z (2026-10-21; operator authorization "Authorized", before any option-D data).** One stock 0.9.1 pass, up to
10.0 occupied GPU-hours, single attempt, no retry; package long-reply/d091/ SHA256SUMS b558b584… verified at launch, DRY_RUN
passed.

**Amendment 22aa (2026-10-21; option-D result, descriptive).**
- **Run:** valid on every frozen check. One stock 0.9.1 pass at seed 70704, 5.884 GPU-hours of 10.0, no preemption,
  KV pool 4,707,515 (the record's size).
- **The pass:**
  - empty 6 of 198, all cap exhaustion, inside the record's per-pass range (5-6);
  - raw 86.4%; stated 88.4% (final after the hand read of 6 rows, 1 committing to no single option);
  - 10 replies past 100k.
- **P(finish | still generating at 100k):** 4 / 10 (0.40), the same as the 0.9.1 record's 10 / 25. rep070-F (0.7.0)
  had 19 / 29 (0.66); difference −0.26 (95% −0.63 to +0.11).
- **Capped replies:** all 6 end in short-period exact loops (four of period 2), the same kind as 0.7.0's. Onsets fall
  at 76k-238k, and 1 was already repeating at 100k.
- **First 100k tokens:**
  - characters per token: capped 2.97 against finished 3.27 (AUC 0.42). 0.7.0 showed 2.31 against 3.15 (AUC 0.12).
    The digit-heavy early signature seen on 0.7.0 is weak here;
  - interjection rate: capped 9.4 against finished 2.7 per 1k tokens (0.7.0: 9.0 against 6.0).
- **Prefixes:** 8 0.9.1-built prefixes exported under B+'s rule (ids and hashes only outside PRIVATE).
- **Limits:** descriptive only. One pass, 10 replies at risk, seeds different from every comparator, and reaching 100k
  selects different trajectories on each engine.

**Amendment 22ab (2026-10-21; operator decision "publish, let's do E", before any E data).**
- **Published:** discovery, B+ and option D (`long-reply/`), with prefixes, prompts, token ids and raw bodies withheld.
- **Option E, the prefix-source × engine cross:** the 8 0.9.1-built prefixes from option D, 4 seeds each, continued
  +65,536 tokens on stock 0.7.0 and stock 0.9.1 under B+'s design and analysis (22v gate).
  - Primary: E1 = 0.7.0 − 0.9.1 on 0.9.1-built prefixes, with B+'s screens.
  - Descriptive cross table: prefix source × engine, with question-matched rows.
  - The package is frozen and costed before authorization.

**Amendment 22ac (2026-10-21; option-E package frozen for review, before any GPU use).**
- **Package:** `long-reply/e-cross/` (SHA256SUMS sha256 0be97e4c…, 135 files).
- **Prefixes:** option D's 8 0.9.1-built prefixes, rebuilt byte-identically.
  - originally capped: q79p4, q81p4, q170p4, q109p4, q13p4;
  - originally finished: q145p4, q121p4, q27p4.
  - q79, q81 and q145 also appear in B+ and form the question-matched rows.
- **Run:** seeds 8,270,101-8,280,804, verified unused; stock 0.7.0 and stock 0.9.1, 4 seeds per prefix, +65,536 tokens.
  Renderings 114 / 114 identical on both images.
- **Gates:** calibrated on 8 reference sessions. The stock 0.7.0 image produced three KV pool sizes in nine starts
  (2,749,599; 2,758,919; 2,768,240); 22v accepted two, E accepts all three.
- **Reading:** E1 under B+'s two-engine rule (EFFECT / REVERSE / SMALL; R-S unavailable).
- **Power:** P(R-E) 0.37-0.42 at the full discovery gap and 0.06-0.08 at half; ≤ 0.012 per direction with no gap. The
  likely reading is R-I, which would not be evidence of no effect.
- **Cost:** central 6.5, pessimistic 10.2, ceiling 13.2 GPU-hours.
- **Checks:** identity 21 / 21, analysis dry runs 19 / 19, refusals 24 / 24, session simulation 62 / 62, privacy clean,
  DRY_RUN passes.

**Amendment 22ad (2026-10-21; operator decision "get more prefixes", before any new data).**
- **D2:** a second text-logged stock 0.9.1 GPQA pass, identical to option D apart from a fresh request seed and run
  tag. Package `long-reply/d091b/`, copied from `d091/` with only those changed. Same terms as D: up to 10.0 GPU-hours,
  single attempt.
- **E, revised before any E data:**
  - **Prefix set:** the union of D's and D2's eligible 0.9.1-built replies under B+'s rule (R50 < 0.05 at 100k), one
    per question across both passes, stratified by original outcome, ranked and tie-broken as in B+. D's 8 prefixes
    keep their place, with D2 adding questions not already present.
  - **KV pool:** reported with a floor (8 requests at the cap), not identity-gated, as in D.
  - **Re-freeze:** power and cost are recomputed for the resulting number of prefixes, and E is re-frozen before its
    run.

**Amendment 22ae (2026-10-21; D2 package frozen, before any GPU use).**
- **Package:** `long-reply/d091b/` (SHA256SUMS sha256 1b080fb5…, 150 files). It is a copy of `d091/` with the seed,
  tag and self-paths changed; every changed file has before/after sha256 in its PROPOSAL §13. Its gate calibration adds
  D's 2026-10-21 session (12 / 12 pass on all four).
- **Request seed 70705:** absent from every request log, results file, published row and raw body. Its only 10 text hits
  are option D's own negative dry run N1, which relabelled stand-in rows to 70705 without sending a request (listed in
  `checks/seed-text-hits.txt`). 70706 is clean under the strict text rule as well.
- **Checks:** analysis dry runs 17 / 17, refusals 22 / 22, session simulation 67 / 67, privacy clean, DRY_RUN at 10.0
  passes.

**Amendment 22af (2026-10-21; operator: "keep 70705, launch", before any D2 data).** D2 authorized at up to 10.0
occupied GPU-hours, single attempt, seed 70705 kept with the disclosed text-scan exception; package SHA256SUMS 1b080fb5…
verified at launch.

**Amendment 22ag (2026-10-22; D2 result, descriptive; E prefix set fixed by the 22ad rule).**
- **Run:** valid. Stock 0.9.1 at seed 70705, 7.163 GPU-hours of 10.0, no preemption, KV pool 4,707,515.
- **The pass:**
  - empty 8 of 198, all cap exhaustion. This is above the record's per-pass range (5-6); D had 6;
  - raw 84.3%; stated 85.4% (final after the hand read of 2 rows, both committing to one option);
  - 12 replies past 100k, of which 4 finished (0.33).
- **Capped replies:** 7 of 8 end in exact loops; none was repeating at 100k.
- **First 100k tokens, capped against finished:**
  - characters per token 2.68 against 3.37 (AUC 0.25);
  - digit share 0.072 against 0.030;
  - interjections 11.6 against 2.3 per 1k (AUC 1.00).
  - Here the digit-heavy signature that D showed only weakly is present.
- **E prefix set under 22ad:** D's 8 plus D2's eligible replies on new questions: q69p5, q120p5 and q186p5 (capped),
  q88p5 and q71p5 (finished). That makes 13 prefixes (8 capped, 5 finished). Eight of their questions (79, 81, 145,
  69, 120, 186, 88, 71) also appear in B+, which gives the question-matched rows.
- **Limits:** descriptive only, one pass.

**Amendment 22ah (2026-10-22; E re-frozen on 13 prefixes, before any E data).**
- **Package:** `long-reply/e-cross/` (SHA256SUMS sha256 60a10748…). `CHANGES-22ah.json` lists 85 changed, 27 new and
  6 removed files; every changed or removed file keeps a `.pre22ah` copy.
- **Prefixes:** P01-P08 from D, unchanged, plus P09-P13 from D2.
  - D2's prefixes: q69p5, q120p5 and q186p5 (capped), q88p5 and q71p5 (finished).
  - All are rebuilt byte-identically; renderings are identical on both images.
  - D2's eligible q161p5 lies outside D2's own per-stratum cap and is excluded under 22ag.
- **Seeds:** 8,270,101-8,281,304 (104), verified unused, including in D2's logs and raw bodies.
- **Gates:** KV pool reported, gated only on the floor of 8 requests at the cap (v070 2,666,496; v091 2,646,016). All
  other gates are unchanged and pass on 9 reference sessions, D and D2 included.
- **Power (thresholds unchanged):** P(R-E) 0.75-0.88 at the full discovery gap and 0.21-0.27 at half; ≤ 0.013 per
  direction with no gap.
- **Cost:** central 10.2, pessimistic 15.4, ceiling 19.9 GPU-hours.
- **Checks:** identity 26 / 26, dry runs 21 / 21 (a KV pool below the floor gives INVALID; an unseen size above it is
  valid), refusals 24 / 24, session simulation 62 / 62, privacy clean, DRY_RUN passes.

**Amendment 22ai (2026-10-22; adopts the independent review "review disposition: amendments 22n-22ah and proposed run
E", relayed by the operator, before any E data; wording only).**
- **E proceeds** with its numerical rules, 13-prefix set (q161 excluded as documented), KV floor, budget (central
  10.2, ceiling 19.9, one attempt) and the two-engine reading rule (R-E / R-E− / R-I; SMALL reported; R-S
  unavailable) unchanged. Incomplete runs stay unclassified. The frozen package files are not changed. This amendment
  supersedes their interpretive wording.
- **Interpretation corrections:**
  - "identical 0.9.1-built states" → identical token prefixes generated by 0.9.1, independently reconstructed by each
    engine's prefill path;
  - "continuation engine effect" → effect of the engine bundle used for prefill reconstruction and continuation;
  - "finishing" → the frozen stop-token endpoint within +65,536 tokens; a valid final answer is verified separately.
  - E does not preserve the original internal state or hold reconstructed state constant. Prefill differences,
    including D7 exposure, are part of its engine contrast.
  - EFFECT requires a point estimate ≥ 0.15 with an interval excluding 0; it does not establish a true effect of at
    least 15 points.
  - "Exact" sign-flip refers to the calculation under exchangeability assumptions. One session per engine in fixed
    order limits causal attribution. Power and false-positive rates are simulation results under stated assumptions.
- **Result statements, fixed now:**

  | Result | Supported | Not established |
  |---|---|---|
  | R-E | On these selected 0.9.1-origin histories, the 0.7.0 engine bundle produced more stop-token completions within +64k under the frozen rule | a kernel cause; superiority from identical internal states; better answer accuracy |
  | R-E− | the corresponding advantage favours 0.9.1 | general superiority of the newer engine |
  | SMALL with substantial prefix dependence | across both tested prefix sources, large stock-engine differences after prefill reconstruction are less plausible; prefix-associated behaviour is important | outcomes irreversibly set before 100k; original decode-built state irrelevant |
  | R-I without SMALL | the experiment did not discriminate | equivalence or state dominance |
  | floor or ceiling | little discrimination at this horizon | engine equivalence |

  - Limits on any "before 100k" reading: prefill reconstruction may remove a generation-time state difference on
    both engines; the endpoint covers finishing by about 164k, not by 327k.
  - The 8 matched questions and cross-source comparisons stay descriptive.
- **Supplementary answer audit** (descriptive, never the primary): for every stopped continuation, whether the text
  after the reasoning close yields a final letter under the frozen extractor (`analysis_col511.final_answer`), and
  separately whether that letter is correct. Reported per engine. It runs on the local raw outputs and publishes
  counts only.
- **B+ citation** (both the INVALID and the amended verdicts are kept): "Under a disclosed, outcome-blind amendment to
  the KV-capacity gate, both stock engines stopped within the continuation budget on 20/48 requests. The estimated
  difference was 0.000, with a prefix-bootstrap 95% interval of −0.083 to +0.083. The overall reading remained R-I."
  η² = 0.67 describes between-prefix variation in the observed sample, not a causal share.
- **C:** not authorized after E. It comes only through a result-dependent design review, with the review's
  requirements: the forced-decode length specified, a contemporaneous B1 prefill control, independent validation of
  the hook, gates calibrated on separate control data, a recalculated cost, and prespecified decision value.

**Amendment 22aj (2026-10-22; operator: "yes to both", before any E data).** E authorized at up to 19.9 occupied
GPU-hours, single attempt, no retry; package SHA256SUMS 60a10748… verified at launch; interpretation under 22ai.

**Amendment 22ak (2026-10-22; E result).**
- **Run:** valid on every frozen check. 104 / 104 requests HTTP 200, all 13 renderings equal, no preemption, 9 warm-ups
  per session, loopback publish. 10.245 GPU-hours of 19.9 (v070 5.918, v091 4.327).
- **Reading: R-I** (two-engine rule). E1 is not EFFECT and not SMALL ("unresolved").
- **Stop-token endpoint within +64k, prefix as the unit:**

  | engine | P | stopped / 52 | originally capped | originally finished |
  |---|---:|---:|---:|---:|
  | stock 0.7.0 | 0.423 | 22 | 0.125 | 0.900 |
  | stock 0.9.1 | 0.288 | 15 | 0.031 | 0.700 |

- **E1 = +0.135** (95% +0.038 to +0.250; 90% +0.058 to +0.231; two-sided sign-flip p 0.0625; prefixes 5 / 0 / 8).
  - The point estimate is below the 0.15 EFFECT threshold.
  - With 5 discordant prefixes, all in one direction, 0.0625 is the smallest two-sided p the exact test can give.
    This is a property of the design, not a change to the reading.
- **Between-prefix variation:** η² 0.72 in the observed sample.
- **Cross (descriptive, prespecified):**
  - 0.7.0-generated prefixes (B+): 0.417 on both engines;
  - 0.9.1-generated prefixes (E): 0.423 (v070) and 0.288 (v091);
  - interaction E1 − C1: +0.135 (95% +0.013 to +0.271; permutation p 0.10);
  - source contrast, confounded with question: −0.061 (95% −0.369 to +0.260);
  - question-matched rows (8 questions) mostly follow each prefix's original outcome.
- **Supplementary answer audit (22ai; counts only):** every stopped continuation closed its reasoning and gave a final
  letter under the frozen extractor (22 / 22 and 15 / 15). Correct: 12 / 22 (v070) and 10 / 15 (v091). Continuations
  of originally capped prefixes that stopped (4 and 1) were all incorrect.
- **Limits:**
  - per 22ai, this is an effect of the engine bundle after prefill reconstruction of token prefixes, not from identical
    internal states;
  - one session per engine in fixed order;
  - re-tokenized ids;
  - the endpoint covers about 164k tokens.

**Amendment 22al (2026-10-22; adopts the independent review "review disposition: run E, amendments 22ai-22ak",
relayed by the operator; the long-reply allocation closes).**
- **Decision:** close the current long-reply line as **unresolved, with positive findings**. E is kept as suggestive
  evidence requiring independent confirmation. Readings are unchanged: B+ R-I (after 22v), E R-I without SMALL. No
  threshold relaxed, no one-sided test.
- **Summary wording, primary statement:** "In B+, the stock engines had identical aggregate stopping rates on selected
  0.7.0-generated prefixes. In E, 0.7.0 stopped more often on selected 0.9.1-generated prefixes: +13.5 percentage
  points, with five prefix-level differences favoring 0.7.0 and none favoring 0.9.1. Neither experiment met its frozen
  criterion for an engine effect or state dominance. The cross-experiment pattern suggests possible dependence on
  prefix source, but does not establish an interaction or locate its cause."
- **Summary wording, separate hypothesis:** "One hypothesis is that the engine bundles respond differently to features
  of 0.9.1-generated histories after prefill reconstruction. Differences in selected histories, reconstruction, and
  sessions remain alternative explanations."
- **"Reaching states its engine handles worse"** is withdrawn from the main finding as too mechanistic.
- **E1 statistics:** its bootstrap interval and exact sign-flip p use different constructions; both are reported. The
  exact-test floor explains the discreteness and does not override the rule.
- **Correction to earlier framing:** these results replace any suggestion of a near-universal completion cliff on the
  corrected engines (the pooled B0 + B1 figure of 0.12 in 19f rested on 17 replies). B1 stopped on 25 / 48 B+
  continuations.
- **Answer audit over all attempts (E):**
  - correct 12 / 52 (23.1%) against 10 / 52 (19.2%), a difference of +3.8 points against a stopping difference of +13.5;
  - the seven extra stops and two extra correct answers are aggregate differences, not paired rescues;
  - all 4 / 1 capped-origin stops were incorrect.
- **C stays unfunded.** Any interaction study needs a new proposal meeting the review's requirements: a fresh
  confirmatory cohort; multiple source-generation sessions; a question-matched core; fixed strata weighting; multiple
  balanced continuation sessions; more histories rather than more seeds; clustering-aware analysis; power ≥ 80% for a
  decision-relevant effect; and a decision that depends on it (e.g. correct answers per GPU-hour).
- **Close-out document:** `long-reply/CLOSEOUT.md`.
