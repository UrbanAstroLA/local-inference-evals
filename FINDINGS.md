# Findings: GLM-5.3-Flash on 2x RTX PRO 6000 (2026-09 to 2026-10)

Every statement from this evaluation, numbered and graded, with its receipt.

- Plain-language summary and result tables: [Results page](https://urbanastrola.github.io/local-inference-evals/).
- What else can move a result: [`CONFOUNDS.md`](CONFOUNDS.md). What changed and when: [`LEDGER.md`](LEDGER.md).
- Labels: [key](#labels). Terms: [glossary](README.md#glossary).

**Grades.** **Supported**: deterministic, or statistically clear. **Descriptive**: what the data shows, without a test
that separates it from chance. **Open**: not answered by the data. "Not graded" marks scoring and context notes.

<details>
<summary>How to recompute</summary>

`python3 tools/verify.py` recomputes every summary. `python3 tools/analyze.py <subcommand>` (named per row)
recomputes the analyses. `python3 investigations/2026-10-glm53-looping/recompute.py` prints every number of the
looping investigation; `python3 investigations/2026-10-glm53-completion/recompute.py` those of the completion
investigation. Standard library only.

</details>

## 1. Engine issues found during evaluation, and their fixes

| # | Statement | Grade | Evidence |
|---|---|---|---|
| 1 | Two upstream vLLM kpool issues were present in the tpurtell 0.7.0 and 0.8.0 images; the fixes remove them. Upstream's tests: 29 of 33 pass without, 33 of 33 with. | **Supported** | `runs/*_kpool-kernel-tests`, [kpool investigation](investigations/2026-10-glm53-kpool-tail) |
| 2 | Under the DCP1 layout of tpurtell 0.8.0 and 0.9.0, decode attention skipped the newest 1-3 tokens at causal lengths up to 2,043 not divisible by 4. The DCP1 tail fix removes it. | **Supported** | [investigation](investigations/2026-10-glm53-looping/README.md#the-dcp1-tail-issue-and-its-fix) |
| 3 | With the fix, decode agrees much better with prefill below 2,044 tokens: mean KL 0.066 → 0.010, 0.008 and 0.006. | **Supported** (for this large effect) | [`glm53-flash-v090-tailfix-decode-prefill`](comparisons/glm53-flash-v090-tailfix-decode-prefill), [`glm53-flash-v091-decode-prefill-repeat`](comparisons/glm53-flash-v091-decode-prefill-repeat) |
| 4 | The fix's effect on answers: GPQA and tool calling show no difference beyond noise at their sizes. | **Descriptive** (effect unmeasured) | [`glm53-flash-k4-v090-v091-gpqa`](comparisons/glm53-flash-k4-v090-v091-gpqa), [`glm53-flash-v090-tailfix-tool-eval`](comparisons/glm53-flash-v090-tailfix-tool-eval) |
| 4a | The engine keeps 511 of the 512 pools its top-k selects, and no top-k path sorts its output. On 0.9.1's tiled route (5-8 requests in flight) the discarded pool ranked better than 480 of 512 in 88.1% of 9,944 live decode rows; on 0.7.0's path, 6.9% of the same rows. Moving the lowest-scoring selected pool into the discarded column removes it (100% of rows, selection unchanged), also under the shipped CUDA graphs. | **Supported** (localized defect); no measurable change in completion (15 vs 16 empty of 594, **descriptive**) | [stage 2](investigations/2026-10-glm53-completion/README.md#phase-2-stage-2-real-inputs), [phase 3](investigations/2026-10-glm53-completion/README.md#phase-3-column-511-intervention), [gate](investigations/2026-10-glm53-completion/README.md#gate-column-511-correction-in-the-shipped-graph-configuration) |
| 4b | tpurtell 0.7.0 through 0.9.1 build attention metadata without token positions, which upstream vLLM passes. The kpool tail ring is then written to a shared null block from position 8 (98% of tail writes on 0.9.1 in shipped graph mode) and occasionally into other requests' blocks. Passing positions as upstream does, with a write that replayed graphs read, removes it (0 stray writes). | **Supported** (localized defect); effect on completion **open** | [tail ring](investigations/2026-10-glm53-completion/README.md#kpool-tail-ring), [`tail-fix/`](investigations/2026-10-glm53-completion/tail-fix) |
| 4c | Every release computes the MoE router's logits in bf16 on this GPU family, although the checkpoint asks for fp32 routing; upstream vLLM enabled the fp32-output path for it (vllm#54048). On 1,593 captured records of 0.9.1 (103,608 token × layer rows), the fp32-output computation selects a different top-8 expert set on 3.15% of rows and different routing weights on every row. The correction lies within the stated tolerance of an fp64 reference on every logit, offline and as served. | **Supported** (computational effect); effect on completion **open** | [stage A](investigations/2026-10-glm53-completion/README.md#stage-a-verification-d3-d4-d5), [`verify-stage-a/`](investigations/2026-10-glm53-completion/verify-stage-a) |
| 4d | Every release leaves the routed experts' SwiGLU clamp (`swiglu_limit` 10) unpassed on the projection-mixed EXL3 path. Passing it reaches the served kernels: a limit of 1 changes greedy outputs, and the served kernels compiled with the fix carry the limit 10. Whether 10 engages on real activations was not measured. | **Supported** (omission and reachability); effect **open** | [bundle gate](investigations/2026-10-glm53-completion/README.md#bundle-gate-d1d2d3d4-on-091) |
| 4e | 0.9.1 with the column-511, tail-ring, router and clamp corrections together (items 4a-4d) passed a gate under the shipped graphs. Two GPQA passes, each read by rules frozen before generation: seed 1234, 3 of 198 replies empty against 5 in the matching 0.9.1 record pass (similar, no clear separation); seed 1235, 3 against 5 (modest improvement, no clear separation). All at the cap. | **Descriptive** (two passes, historical controls; no clear separation; combination exploratory) | [screen](investigations/2026-10-glm53-completion/README.md#one-pass-gpqa-screen), [second pass](investigations/2026-10-glm53-completion/README.md#second-pass-seed-1235), [`bundle/`](investigations/2026-10-glm53-completion/bundle) |
| 4f | On 0.9.1 the served fused MoE router computes routing weights differently above about 1,024 batch rows than below (expert ids unchanged). This is expected numerical variation of the upstream kernel used for large batches, not a defect: it rebuilds the unbiased score from the biased one in fp32, which loses precision with this model's large correction biases, and its own arithmetic reproduces the served weights. The same kernel is in every release. | **Descriptive** (expected numerical variation; effect on outputs untested) | [observations](investigations/2026-10-glm53-completion/README.md#observations-recorded-separately), [E1 assessment](investigations/2026-10-glm53-completion/e1-kvwriter/README.md#e1-batch-size-dependent-routing-weights), [confounds](CONFOUNDS.md#17-batch-size-dependent-routing-weights-e1) |
| 4g | On checkpoint B0 (0.9.1 with the corrections of items 4a-4d), the two-GPU all-reduce transport the engine dispatches under its CUDA graphs (B12x PCIe peer-push) returned the exact bf16 sum on both GPUs in all 2,040,800 calls of a model-free fixture covering every graph size and launch geometry, back-to-back replays, address reuse and rank skew; no hangs. | **Supported** (the tested path; other communication paths not covered) | [checkpoint B0](investigations/2026-10-glm53-completion/README.md#checkpoint-b0-and-targeted-invariants-amendment-16), [`b0-allreduce/`](investigations/2026-10-glm53-completion/b0-allreduce) |
| 4h | B0's KV record writer (528-byte record) and query projection agree with 0.7.0's (656-byte record, H64 projection) within expected rounding: stored bytes differ on 0.06-0.11% of elements, each by one E4M3 code and each explained by the rounding model fixed before data; the two projections are bitwise equal on the shared lanes. Read under criteria errata recorded before GPU data. | **Supported** (these inputs and shapes) | [checkpoint B0](investigations/2026-10-glm53-completion/README.md#checkpoint-b0-and-targeted-invariants-amendment-16), [`e1-kvwriter/`](investigations/2026-10-glm53-completion/e1-kvwriter) |
| 4i | With B0's real cache geometry (a 2.93 GB shared tensor whose pages cross 2³¹ bytes), every cache writer and reader driven stayed in its own page and addressed correctly at the highest blocks and on the record straddling 2³¹, eager and under graph replay, after fixture faults found in the data were fixed. | **Supported** (driven kernels only) | [checkpoint B0](investigations/2026-10-glm53-completion/README.md#checkpoint-b0-and-targeted-invariants-amendment-16), [`b0-cache/`](investigations/2026-10-glm53-completion/b0-cache) |
| 4j | On checkpoint B0, under real scheduling, the index-K prefill writer leaves the pool that straddles a prefill chunk start not aligned to 4 tokens unwritten, or fills it from the previous request's tokens in the batch (D7): 217 events in 50 requests of one long-prefix serving session, identical on both GPUs, every one at an unaligned chunk start. Every other slot mapping the attention and index-K kernels consumed was correct over the frozen coverage. Effect on selection, outputs or completion not measured. | **Supported** (the defect, in serving; its effect untested) | [live slot mappings](investigations/2026-10-glm53-completion/README.md#live-slot-mappings-on-b0-amendment-17), [`b0-mapguard/`](investigations/2026-10-glm53-completion/b0-mapguard) |
| 4k | D7's writer is byte-identical in tpurtell 0.7.0, 0.8.0 and 0.9.1 and keeps the same assumption in upstream vLLM main. A writer-side correction, which reads a straddling pool's earlier members from the request's own tail ring, was verified in serving as B1 = B0 + D7: at 219 unaligned chunk starts every straddling pool was written and byte-identical to an independent recompute, with 0 mismatches over 1.53 million aligned control pools and every other mapping clean. In the GPQA records the exposure is narrow and opposite to the completion gap (reconstructed: none in the 0.9.1 family; one question in 0.7.0, which finished). Completion impact untested. | **Supported** (the correction, for the covered serving configuration); exposure **descriptive** (reconstruction) | [D7 correction](investigations/2026-10-glm53-completion/README.md#d7-correction-and-verification-amendment-18), [`d7/`](investigations/2026-10-glm53-completion/d7) |
| 4l | One GPQA pass of B1 (B0 + D7) at seed 1236 left 9 of 198 replies empty, all at the cap, against 6 for stock 0.9.1 at that seed, with raw and stated accuracy unchanged (173 / 174 against 171 / 174). Under rules frozen before generation this reads unfavourable, which triggers review; it does not invalidate the verified corrections. The extra generated tokens are the extra capped replies, and finished replies are no longer. | **Descriptive** (one pass; historical control; seed reused; review in progress) | [B1 pass](investigations/2026-10-glm53-completion/README.md#b1-checkpoint-pass-and-the-long-context-tail-amendment-19), [`b1-screen/`](investigations/2026-10-glm53-completion/b1-screen) |
| 4m | Across the 11 logged K3.25 GPQA passes, replies reach 100k tokens at similar rates on every engine, but the share of those that finish before the cap is 0.71 on 0.7.0, 0.40 on the 0.9.1 family and 0.12 on B0 + B1; beyond about 150k, 0.9.1-family replies almost never finish. The completion gap sits in whether long replies terminate. | **Exploratory** (chosen after the B1 result; small counts for 0.7.0; historical controls) | [long-context tail](investigations/2026-10-glm53-completion/README.md#b1-checkpoint-pass-and-the-long-context-tail-amendment-19), [`tail-analysis/`](investigations/2026-10-glm53-completion/tail-analysis) |

<details>
<summary>Numbers and qualifications, items 1-4m</summary>

1. vllm-project/vllm#57477: every prefill wrote 2 KB of keys into another block's indexer region. vllm-project/vllm#58454: a rejected
   pool-completing draft could overwrite committed keys at 2 or more draft tokens. 29 of 33 pass on both images.
   Ported in tpurtell PR #5, shipped in tpurtell 0.9.0. The tests ran on the 0.7.0 and 0.8.0 release images, on local
   builds with the fixes, and (2026-10-09) on the 0.9.0 and 0.9.1 release images, which pass 33 of 33 and reproduce no
   rejected-draft corruption at 2, 3, 5 or 7 draft tokens.
2. Index check on the image's own kernels: the tail is dropped in 9 of 23 packed cases without the fix, 0 with it; rows
   the stock code already handled are byte-identical. DCP1 with MLA layer ownership is tpurtell's layout choice and
   frees KV memory (item 17); the masking path already existed in the vendored attention code, and only this layout
   exercises it. Merged as tpurtell PR #6, released in tpurtell 0.9.1.
3. Six prompts every run shares: 0.066 without the fix (one run), 0.010 with the local build of the fix, 0.008 and 0.006
   in two runs of the 0.9.1 release. Top-1 agreement 93.8% → 97.6%; lower in 6 of 6 prompts. The two 0.9.1 runs, same
   prompts and seeds, are the noise floor: means differ by up to about 2x, single prompts by up to about 6x. From 2,048
   tokens, runs with the fix range 0.006 to 0.019 and the one run without it is 0.031: no effect claimed there. With
   speculation on (DFlash2 ×3, one run), 0.9.1 decode sits somewhat above both speculation-off runs, within that spread
   ([`glm53-flash-v091-decode-prefill-speculation`](comparisons/glm53-flash-v091-decode-prefill-speculation);
   descriptive).
4. GPQA: 4bpw TR3 (Brandon) 84.8% on tpurtell 0.9.1 vs 84.3% on 0.9.0, one pass each at 8 concurrent requests with a
   KV pool that holds about 4 requests at the token cap; well inside the 0.5-3.5 points by which passes of one
   configuration differ (item 5). tool-eval-bench: TC-80 and TC-88 pass in both repeats with the fix and fail in both
   without it (157 and 157 of 176 points → 159 and 163), but that is two repeats per build, and ten other scenarios
   flip between repeats of the same build.

**4a.** Found by the [completion investigation](investigations/2026-10-glm53-completion/README.md) (2026-10-11). The
engine asks the top-k for 512 compressed-key pools and keeps output columns 0-510 to make room for the newest 1-3
tokens; the pool in column 511 is discarded. Upstream vLLM keeps all 512. On 0.9.1, decode batches of up to 16 rows
(1-4 requests in flight at 3 draft tokens) take a fused top-k route that discarded a pool ranked better than 480 in
0.5% of live rows; larger batches take a tiled route (88.1%). By context length: 96-97% from 4,000 to 64,000 tokens,
13% at about 120,000. Synthetic inputs: 94-98% at 1,024-16,384 pools. Selected sets matched an fp64 reference in every
clean row; only which selected pool is discarded differs. The real-input capture ran in eager mode (a probe cannot run
inside the indexer under CUDA graphs; the route follows the row count). The gate, under the shipped graphs, found the
prefill branch affected too (rows with exactly 512 candidates come back in index order, so the newest pool was
discarded) and the frozen correction covers it; after correction, 0 changed selections and 0 rows with a non-lowest
pool in column 511. The correction changes the engine's 511-pool selection policy to discard the lowest-scoring
selected pool; it is not a restoration of upstream's 512-pool policy. Prepared locally, not released. It is the only
defect found that differs between the engines in the direction of the completion gap (item 7). A three-pass GPQA arm
with this correction alone (2026-10-11, against the existing 0.9.1 record as a historical control, analysis frozen
before data) left 15 of 594 replies empty against 16 (one-sided p = 0.50; reduction +0.17 points, 95% -1.01 to +1.35),
all 15 at the token cap: a correctness fix that did not move completion behaviour, so it is not supported as the
dominant completion explanation. Raw accuracy 85.0% vs 86.9% (-1.85 points, 95% -5.05 to +1.35); not significant,
which does not establish no accuracy cost
([completion arm](investigations/2026-10-glm53-completion/README.md#completion-arm)).

**4b.** Found by the same investigation in a capture where 16 index-K pools were byte-identical across two different
requests, some decoding as NaN: raw indexer key and gate rows written through the tail ring into blocks then owned by
other requests, from stale tail ids left by the runner's start-up warm-up. Upstream vLLM has passed positions since
GLM-5.3-Flash support landed. On a CPU harness with each release's real kernels, the share of decode-completed pools
that differ from the request's own keys is 0% / 47-57% / 78-90% on 0.9.1 at 1 / 2 / 8 requests, and 100% on 0.7.0 at
all three; 0.7.0 is the more affected engine, which does not exclude an accidental completion benefit from incorrect
computation: the defect's role in completion is untested. Shipped graph
mode, unmodified 0.9.1: 6 events wrote into live attention blocks. The fix (positions, upstream's ownership guard and
a graph-safe write) gave 0 null-block, 0 foreign and 0 attention-block writes in two variants; all requests completed;
throughput unchanged. The gate found at least one selected pool with NaN bytes in 7.9% (rank 0) and 32.5% (rank 1) of
sampled decode rows, attributed by inference to this defect; the bundle gate of 2026-10-12, with this fix, found 0 in
every branch (the fix inferred as the cause; not varied alone). The kernel tests of item 1 exercise the kernels, not the
metadata call where this omission sits. Prepared locally, not released, excluded from the completion arm and included in
the bundle (item 4e).

**4c.** Stage A of the completion investigation (amendments 13-14). Controls on all 1,593 records: dispatch and weight
hashes; the offline shipped computation reproduces the captured logits bitwise; the engine's router, invoked as serving
does (one call on the serving batch rows), reproduces the consumed routing bitwise; the corrected computation lies within
τ_d = 2⁻¹⁴·S + 2⁻²⁴·|ref| of fp64 on all 29,839,104 logits. The serving-shaped router invocation was adopted before any
comparison was computed, after a 64-row invocation failed the bitwise control (item 4f). Membership changes: 95% bootstrap
3.01-3.29% over 176 requests, 2.90-3.50% over 8 histories; by layer 1.2-6.6%; by execution type 3.0-3.2%; on flipped
rows the 8th-9th score gap has median 0.00014-0.00034. FULL-graph decode is covered by source inspection only. As
served in the bundle gate: on 1,000 records the fp32-output computation reproduces the served logits bitwise, with 0
logits outside τ_d of fp64. Flips and weight changes are not evidence of a completion effect.

**4d.** The gate's original criterion, a host-side walk finding the limit on compiled kernel objects, failed: the runtime
does not retain those objects. It was replaced after the gate data were inspected by the compile-cache evidence: 66
kernels with (True, 10.0) written during the limit-10 session by the fixed plans, none before it. A limit of 1 changed all
6 probe outputs within 16 tokens; on the 4 short, bitwise-repeatable prompts at tokens 0-2.

**4e.** Gate (amendments 14-14d): two original criteria failed as specified (item 4d's walk; repeatability of all 6
greedy probes, which the stock release also fails at long context, item 19) and were revised after the gate data were
inspected; the original criteria are not described as met. Screen (amendment 14e): all 198 questions, original
protocol, analysis and category rules frozen before generation; valid; category similar / no clear separation (E = 3;
clear promising needed E ≤ 2 with no regression flag). Against the 0.9.1 record's pass 1: 3 rescued (q12, q109, q120),
1 newly empty (q79); against the column-511 arm's pass 1: 1 rescued (q170), 0 newly empty. Secondary, single pass, not a
decision input and without a significance claim: raw 174, stated 176 of 198 (record pass 1 169 / 170, arm 164 / 165,
0.7.0 174 / 177). One pass has little power for a low-rate outcome: simulated P(E ≤ 2) is 0.078 with no change and
0.85-0.86 at 0.7.0-like rates. Not evidence that the included corrections lack effects. 4.84 GPU-hours. Second pass
(amendments 15-15c; proposed by the independent assessment for seed sensitivity, chosen after the first screen, which
keeps its category): the unchanged bundle at seed 1235, protocol pass 2 on a fresh server as in the comparators'
seed-1235 passes; valid; E = 3, all at the cap; category modest improvement, no clear separation (amendment 15's rules,
under which E = 3-4 against a baseline of 5 carries this name). Accuracy thresholds are an absolute standard from the six
stock-0.9.1 passes (material concern raw ≤ 158 or stated ≤ 159; concern raw 159-163): none met (raw 168, stated 169; one
of 5 adjudicated rows hedged). Against the matched record pass (the family's best, raw 176): 2 rescued (q55, q81), 0
newly empty, raw −8, stated −9; the matched-offset readout, reported but not deciding, shows "concern". Against the
column-511 arm's pass: 5 rescued, 3 newly empty; 0.7.0's seed-1235 pass (context): 3 empty. EXPLORATORY, both bundle
passes against the matching stock passes (historical controls; seeds reused): 6 vs 10 empty of 396, reduction 1.01
points (95% 0.00 to 2.27), one-sided p = 0.11; raw −0.76 points (−3.79 to +2.27); against the arm 6 vs 9, raw +2.02
(−1.77 to +6.06). 4.541 GPU-hours.

**4f.** Stage A, engine observation E1: the switch lies between 1,025 and 1,139 rows; against fp64, median relative
weight error 1.2e-7 at ≤ 1,024 rows and 1.4e-6 above (maximum 1.1e-3). Inferred exposure: prefill chunks only. CPU
assessment (amendment 16b): above 1,024 rows vLLM's grouped top-k router runs a warp kernel that rebuilds the unbiased
sigmoid as (s + b) − b in fp32; with correction biases of 5.0-14.5 that puts it on a 2⁻²¹-2⁻²⁰ grid. An emulation of that
arithmetic reproduces the served weights bitwise on 98% of rows (the residual from hardware approximations inside the
sigmoid), and 0 of 729,600 weights exceed its stated error bound (largest ratio 0.83); the block kernel's arithmetic
reproduces none of those rows. Selection cannot be affected (both kernels select from the same biased scores; 0
differences on 138,280 rows, stock 0.9.1 and B0). p99 weight change 5.6e-7, against about 1.6e-3 for item 4c. Same
library in every release. No workaround.

**4g.** Amendments 16-16b. The transport's epoch counter advanced by exactly the scheduled peer-push calls in every phase;
41 cases (FULL 4-48 rows, PIECEWISE 1-48, every launch-geometry edge, rank skew, interleaved eager prefill), 26,650
forwards, 2.05·10¹¹ checked elements; reference exact bitwise `(a.float()+b.float()).to(bf16)`; 900 forwards through the
in-graph PyNCCL reference also exact. Retired: stale, torn, cross-step, wrong-rank, addressing and slot-parity errors and
hangs on this path. Not covered: contention with model kernels, the drafter's exact sequence, epoch rollover, PyNCCL
paths beyond a cross-check, the owner exchange. 0.07 GPU-hours.

**4h.** Amendments 16a-16b. One GPU per image (B0 and stock 0.7.0), no model; real-weight proxies at MLA layers 3, 23 and
43 plus a synthetic set; references fp64. Before GPU data, a CPU dry run of the frozen analysis on defect-free outputs of
the frozen rounding model failed three criteria (a projection bound half the correct half-ulp; a dequantisation tolerance
inconsistent with the error model; a byte-difference rate that omitted exact E4M3 ties); errata were recorded before the
run, the analysis reports both verdicts, and the operator chose the corrected one. Under the criteria as first written
the verdict is "beyond expected rounding", with failures equal to the dry run's defect-free counts. 90-99% of byte
differences are at exact ties, which the writers break differently; scales within 1 ulp; 0.7.0's RoPE tail all zero.
Not covered: latents captured from a server, the record readers, rounding-level effects on completion. 0.0069 GPU-hours.

**4i.** Amendments 16c-16e. One GPU, no model, B0's own kernels: draft, KDA, MLA and kpool writers and readers,
copy-on-write, the new-block zeroer, cross-group reuse of a block id and graph replay; every operation checked byte-exact
against a full-size shadow image. Addresses shown reachable in serving by a CPU simulation with the real allocator. The
first run with data passed 15 of 18 parts; three parts failed on fixture faults and one upstream test on an obsolete
interface; the fixture was fixed after inspecting the data (no criterion loosened), rerun 1 exposed two more fixture
faults, and rerun 2 passed 80 of 80 checks. Retires the source audit's draft-slot-sharing and above-2³¹ items for these
kernels. Not covered: block-table and slot-mapping construction under real scheduling (the class of item 4b), the index-K
readers, extend under graph replay, the mamba postprocess kernel, rank 1. 0.0392 GPU-hours.

**4j.** Amendments 17-17e. An observe-only guard on B0 (inert unless enabled) checked each mapping at the point of
consumption inside the replayed CUDA graphs. The expectation was rebuilt independently from the scheduler's output
stream. CPU fault-injection tests passed 28 of 28 before GPU use, and 28 coverage minimums were frozen.

- **Attempts.** Attempt 1 was refused by the runner's pre-flight (0 GPU-hours). Attempt 2 was invalid because of
  investigation tooling faults (the runner's abort pattern, the guard's index-K read unit, two analysis artefacts); these
  were fixed after inspection with no rule loosened (0.4211 GPU-hours). Attempt 3 met every minimum and nothing was
  invalid (0.5722 GPU-hours).
- **Coverage in attempt 3:** 2,076 graph-replay steps, 132 mixed steps, 32 prefix hits, 18 route transitions, 26,016
  accepted and 16,677 rejected drafts, and accesses above 2³¹ bytes on every site. Short greedy probes were identical to
  the earlier B0 record.
- **Violations,** all at the index-K prefill write: 186 unwritten pools and 31 cross-request pools. A post-hoc join places
  every per-layer record in the pool straddling an unaligned chunk start. Most such starts are at 2,045 and 4,090 tokens
  into a block (the prefill budget of 2,048 less 3 draft slots); the others fall in steps shared with other requests.
  The prediction was registered before GPU data, and a CPU test reproduces both cases with B0's kernel.
- **Not covered:** copy-on-write prefix paths (unreachable), contention-dependent races, the values the index-K readers
  compute, and the mixed-quant MoE. Whether 0.7.0 shares D7 was not checked.

0.9933 GPU-hours.

**4k.** Amendments 18-18c.

- **Release and upstream comparison** (CPU, each image's own code):
  - the writer is identical across releases, and every served configuration reaches unaligned starts;
  - the pattern differs: 0.7.0 has 5 draft slots, a 2,043-token solo chunk, a 3,584-token block and DCP2, against 3,
    2,045 and 4,352 tokens and DCP1 for 0.8.0/0.9.1;
  - SGLang repairs the defect writer-side.
- **Exposure:**
  - the rebuilt scheduler simulation reproduces the B0 session's 217 events exactly;
  - exact prompt-token counts for 1,980 logged GPQA requests match the server's;
  - in the 0.9.1 family doc 127 is split at 1,996, a pool boundary, given 7 decode rows at admission, which is
    reconstructed from proxy timestamps;
  - an appended serving phase measured the split at 2,045 − 7·D for D = 0-7.
- **Correction:** 28/28 CPU tests against an independent reference; D1-D4 regressions pass.
- **Serving:**
  - the guard was amended with an independent byte oracle: its own shadow of each request's keys, recomputed with the
    engine's compress kernel;
  - all four fault classes were detected on CPU;
  - all coverage minimums were met and the short probes were identical to B0's.

0.6244 GPU-hours, one attempt.

</details>

## 2. GPQA Diamond

198 questions; pass *p* sends request seed 1233 + *p*. Raw = `flexible-extract` (the headline); stated =
`correct_stated` (audited). 8 concurrent requests unless the label says otherwise. Tables:
[Results page](https://urbanastrola.github.io/local-inference-evals/#gpqa),
[`glm53-flash-gpqa-records`](comparisons/glm53-flash-gpqa-records) (three passes),
[`glm53-flash-gpqa-configs`](comparisons/glm53-flash-gpqa-configs) (pass 1).

| # | Statement | Grade | Evidence |
|---|---|---|---|
| 5 | One pass of one configuration varies by 0.5 to 3.5 points between passes with their own request seeds. | **Descriptive** | `analyze.py gpqa-passes` |
| 6 | 3.25bpw and 4bpw TR3 (Brandon) on tpurtell 0.9.1 show no measurable difference in accuracy (86.9% vs 85.7%) or completion (16 vs 17 empty of 594). | **Descriptive** | `analyze.py gpqa-records` |
| 7 | tpurtell 0.7.0 as shipped left fewer GPQA questions unanswered than tpurtell 0.9.1 (3.25bpw, three passes each): 4 vs 16 empty answers of 594. What produces the difference is open: two localized defects have been found (items 4a-4b); a GPQA arm with the column-511 correction alone left 15 vs 16 empty, so that defect is not supported as the explanation; two passes with four corrections together (items 4a-4d) left 3 and 3 empty against 5 and 5 in the matching 0.9.1 passes, with no clear separation (item 4e). | **Supported** (clustered p = 0.009); cause **open** | `analyze.py gpqa-records`, [confounds](CONFOUNDS.md#7-several-changes-between-releases-at-once) |
| 7a | 3.25bpw on tpurtell 0.7.0 scored highest over three passes. Raw / stated means: 88.2 / 89.4% (0.7.0), 86.9 / 87.9% (3.25bpw 0.9.1), 85.7 / 87.7% (4bpw 0.9.1). The lead is 1.3-2.5 points raw, consistent on both scores but within pass-to-pass noise (clustered p = 0.12-0.41; every 95% interval includes zero), so not yet established. Against 3.25bpw 0.9.1, the higher score comes entirely from questions 0.9.1 left unanswered: where both answered (577 of 594), raw accuracy is identical (516 vs 516; stated 523 vs 522). Why 0.7.0 leads is open. | **Descriptive**; cause **open** | `analyze.py gpqa-records`, `gpqa-join`, [investigation](investigations/2026-10-glm53-completion/README.md) |
| 8 | Empty answers concentrate on a few questions. | **Descriptive** | `analyze.py gpqa-empty` |
| 9 | One pass per configuration does not separate these configurations in accuracy: pass 1 lands at 84.3-87.9% raw. | **Descriptive** | `analyze.py gpqa-table`, `gpqa-pairs` |
| 10 | Empty answers in pass 1 are 0 to 9 of 198 per run. | **Descriptive** | `analyze.py gpqa-table` |
| 11 | Scoring note: raw `flexible-extract` reads some correct answers as wrong, a net 0.5 to 3.5 points per pass. | Not graded (scoring note) | [`protocols/gpqa-diamond/v1.md`](protocols/gpqa-diamond/v1.md) |
| 12 | Published scores (92.1 NVIDIA, 90.6 Red Hat) are context only. | Not graded (context only) | [`glm53-flash-gpqa-published-context`](comparisons/glm53-flash-gpqa-published-context) |

<details>
<summary>Numbers and qualifications, items 5-12</summary>

5. Three configurations, three passes each. As large as the spread of pass 1 across all ten configurations
   (84.3-87.9%, 3.6 points). Of 198 questions, 154-160 are right in all three passes of a configuration, 10-12 in none,
   28-32 vary between passes.
6. Over three passes each: B - A -1.2 points (-4.2 to +1.7; clustered p = 0.51); stated 87.9% vs 87.7%; empty answers
   9 questions each way (clustered p = 1.00). The 4bpw record ran 4 requests at once, the 3.25bpw record 8. With three
   passes, accuracy differences smaller than about 3 points are neither shown nor excluded.
7. On 4 vs 11 questions; 0.9.1 had more empty answers on 10 questions and fewer on 1 (exact sign-flip test over
   questions p = 0.009; +2.0 points, 95% interval over questions +0.7 to +3.5). The question was raised by pass 1
   (0 vs 5); passes 2 and 3 alone, run after it was raised, point the same way (4 vs 11 on 1 vs 7 questions, clustered
   p = 0.06). Three record comparisons were made without correction (Bonferroni over three: p = 0.03). Accuracy does not
   differ measurably (88.2% vs 86.9%, clustered p = 0.41). Where a request log exists, every one of these empty answers
   ran to the 327,680-token cap except one, which ended after 38 tokens. 0.7.0 differs from 0.9.1 in draft depth (5 vs
   3 tokens), parallel layout (DCP2 with EP2 experts vs DCP1 with MLA layer ownership), kernel and engine code, vision
   (on vs off) and KV pool (2,758,919 vs 4,707,515 tokens) at once. The two model revisions share the same weight files.
   Their chat templates differ (0.7.0 served the checkpoint's own template, 0.9.1 the corrected Z.ai template), but both render all 198 GPQA prompts byte-identically. None was varied alone in GPQA;
   the component screen at three draft tokens found no tested part that moved questions 88 or 79 (item 14). The 4bpw
   TR3 (Brandon) record on 0.9.1 also left more empty answers than 0.7.0 (17, on 14 questions; clustered p = 0.006), but
   it differs in weights and concurrency as well.
   The [completion investigation](investigations/2026-10-glm53-completion/README.md) (from 2026-10-10) has since found no step where 0.9.1's decode agreed worse with
   its own prefill than 0.7.0's; its long-context measure turned out to cover continuation after the reply had ended,
   so it does not bear on completion. A completion pilot and replication of 0.9.1's default layout against 0.7.0's
   layout bundle on 0.9.1 gave no useful separation. A source audit found no defect unique to 0.9.1 in attention or the
   DCP combine, and two localized defects (items 4a-4b): the column-511 drop, which differs between the engines in the
   direction of the gap, and the tail-ring omission, which affects 0.7.0 more (its role in completion is untested). A
   three-pass GPQA arm on 0.9.1 with the
   column-511 correction alone, against this 0.9.1 record as a historical control and read by rules frozen before data,
   left 15 empty replies vs 16 (one-sided p = 0.50); empties moved between questions (10 rescued, 9 newly empty). The
   column-511 drop is therefore not supported as the explanation of the gap
   ([completion arm](investigations/2026-10-glm53-completion/README.md#completion-arm)). The gap itself rests on one
   three-pass record per engine. The corrected arm's 15 resembles the 0.9.1 record's 16, but it ran a modified engine
   and is not a replication of the unmodified record; whether either engine's rate reproduces unmodified has not been
   tested. Stage A then measured the router's bf16 logits (item 4c), and one GPQA pass of 0.9.1 with the column-511,
   tail-ring, router and clamp corrections together left 3 of 198 empty against 5 in the 0.9.1 record's pass 1 and 4 in
   the arm's (0.7.0's pass 1: 0): similar, no clear separation under rules frozen before generation (item 4e). A second
   pass at seed 1235 left 3 against 5 in the record's matching pass (0.7.0's: 3): modest improvement, no clear
   separation; both passes together, exploratory, 6 vs 10 of 396 (one-sided p = 0.11). Router
   precision and the clamp are the same in every release, so neither can explain a difference between 0.7.0 and 0.9.1.
   Item 7a, pair by pair (same question, pass and request seed): 0.7.0 was right where 0.9.1 came back empty in 8
   question-passes (q12 pass 1, q55 pass 2, q81 pass 3, q88 passes 1 and 3, q109 pass 1, q120 pass 1, q170 pass 3), and
   never the other way. This splits the score by completion; it is not a causal comparison.
8. Questions 79 and 81 came back empty in all three records; 88, 127 and 147 in both 0.9.1 records and in no pass of
   0.7.0; question 88 in two of three passes of each 0.9.1 record. Questions are doc ids; the questions themselves are
   not published.
9. 85.4-89.4% stated. Within the 0.5-3.5 points by which passes of one configuration differ (item 5); every
   question-paired difference is consistent with noise. Each comparison resolves only differences of about 5 points, so
   smaller effects are neither shown nor excluded.
10. No paired pass-1 comparison of empty answers reaches p < 0.05; one pass per configuration does not separate
    releases (the three-pass records do for 0.7.0 vs 0.9.1, item 7). Of the 43 published empty answers with a request
    log (four runs), 42 ran to the 327,680-token cap and one ended after 38 tokens (`finish_reason` stop).
11. Misread when a reply mentions other options' labels or chemistry notation after its answer, or states its answer as
    a boxed or bold letter. Over the 16 published passes, 49 correct answers are scored wrong and 3 replies whose stated
    answer is not the target are credited: net 0.5 to 3.5 points per pass (1.5 on average). The audited
    `correct_stated` score is published per row beside it; raw scores stay the headline so runs remain comparable.
12. Raw score: NVIDIA's 92.1 (BF16 and NVFP4) lies above every local run's interval; Red Hat's 90.6 (NVFP4) lies inside
    the intervals of three runs, all at DFlash2 ×5 (the three-pass tpurtell 0.7.0 record among them), and above the
    others. Stated-answer score: 90.6 lies inside nine runs' intervals and 92.1 inside three. Those numbers used other
    weights and harnesses whose details are not fully published; no higher-precision reference was run here, so these
    runs cannot separate quantization, harness, scoring and runtime effects.

</details>

## 3. Non-completion on hard questions

Details: [investigation](investigations/2026-10-glm53-looping/README.md#non-completion-what-the-clean-data-shows).

| # | Statement | Grade | Evidence |
|---|---|---|---|
| 13 | On tpurtell 0.9.1, question 88 fails to finish in 1-3 of 12 draws in every tested arm; question 79 in 9 of 12 on both tpurtell 0.9.1 and the tpurtell 0.7.0 image at three draft tokens. | **Descriptive** | `analyze.py screen-v2`, [`glm53-flash-v091-component-screen-doc88`](comparisons/glm53-flash-v091-component-screen-doc88), [`glm53-flash-doc79-image-screen`](comparisons/glm53-flash-doc79-image-screen) |
| 14 | None of the tested runtime parts moved either question at this size, and only very large effects could have shown. | **Descriptive** | `analyze.py screen-v2`, `screen-power` |
| 15 | Repeats that share one request seed vary only through small numerical differences (batching and engine nondeterminism), which is not a meaningful sample of how often a question fails: 11 of 12 failed with seed 1234 on every repeat vs 3 of 12 with distinct seeds. | **Supported** (Fisher p = 0.003) | [`glm53-flash-fixed-seed-control`](comparisons/glm53-flash-fixed-seed-control) |
| 16 | Loops are stopped far beyond the 2,044-token region where the tail issue acted: an estimated 106,000-187,000 tokens. | **Descriptive** | `analyze.py screen-anatomy` |

<details>
<summary>Numbers and qualifications, items 13-16</summary>

13. Component screen, 2026-10-09, preregistered: one question per run, distinct request seeds 5001-5012. Held fixed in
    every arm: 3.25bpw weights, DFlash2 ×3, temperature 1.0 / top_p 0.95, the 327,680-token budget, 12 concurrent
    requests, one question per arm, a fresh server. Both questions were chosen as hard from GPQA runs and screens that
    all sent request seed 1234; with distinct seeds question 88 fails far less often than those screens suggested. Two
    questions are not a benchmark-wide rate. Per-arm counts with 95% Wilson intervals: in the investigation.
14. Question 88: EP2 routed experts 3 vs 5 failures of 24 (Fisher p = 0.70), NOPE records off 4 vs 4 (p = 1.00), MLA
    ownership tp 3 vs 5 (p = 0.70). Arms EO and NO also turned draft-slot sharing off, so ownership and sharing are not
    separated; the preregistered rule found no candidate. Question 79: 9 vs 9; the rule's verdict is "unresolved".
    Minimum detectable differences (two-sided Fisher, p < 0.05, 80% power; `analyze.py screen-power`): one arm against
    another on question 88 (12 vs 12, from 25%) only a rise of about 60 points, and no drop at any size; a switch on vs
    off (24 vs 24, from 17%) a rise of about 41 points; question 79 (from 75%) a drop of about 60 points. Draft depth,
    quantization and sampling settings were not varied.
15. Same configuration and question (88), 12 repeats each. The earlier hard-question screens sent seed 1234 on every
    repeat; their rates, the layout-bisection statistics and the question-level observations drawn from them were
    withdrawn on 2026-10-09 and replaced by the component screen ([ledger](LEDGER.md#withdrawn-and-what-replaced-it)).
    Repeat 1 of each question from each configuration's first screen is kept as a single draw
    (`analyze.py screen-single`): loops or exhaustion occur in every configuration tested.
16. Component screen, early-stop detector. Question 88 fails by looping, question 79 mostly by exhaustion (varied
    reasoning until the budget runs out).

</details>

## 4. Serving

Tables: [Results page](https://urbanastrola.github.io/local-inference-evals/#serving).

| # | Statement | Grade | Evidence |
|---|---|---|---|
| 17 | KV capacity depends on the layout; tpurtell's default DCP1 layout with MLA layer ownership holds the most: 4,707,515 tokens for 3.25bpw on 0.9.0/0.9.1. | **Supported** | server logs (`server_log.jsonl`) |
| 18 | Neither fix shows a speed cost, in single runs. | **Descriptive** | [`glm53-flash-serving-probe`](comparisons/glm53-flash-serving-probe) |
| 19 | The engine is not bitwise reproducible, even one request at a time. | **Supported** | [`glm53-flash-serving-probe`](comparisons/glm53-flash-serving-probe), [`glm53-flash-v091-decode-prefill-repeat`](comparisons/glm53-flash-v091-decode-prefill-repeat) |
| 19a | Greedy decoding of the stock 0.9.1 release, one request at a time, repeats bitwise on short prompts (124-158 tokens, 96 tokens) but not at long context: prompts of 4,240 and 16,240 tokens diverge at token 3. | **Descriptive** (one control session; mechanism open) | [observations](investigations/2026-10-glm53-completion/README.md#observations-recorded-separately), [confounds](CONFOUNDS.md#16-long-context-greedy-non-repeatability) |

<details>
<summary>Numbers and qualifications, items 17-19a</summary>

17. Reported by the server at start-up, same memory setting. 3.25bpw: with v0.7.0's layout on the 0.9.0 image
    3,165,056; component-screen arms on 0.9.1 3,992,056 (EP2, NOPE records off), 2,215,158 (EP2, ownership tp) and
    1,851,617 (NOPE records off, ownership tp); the tpurtell 0.7.0 image 2,894,456 at three draft tokens and 2,758,919 at
    five. 4bpw TR3 (Brandon) on tpurtell 0.9.1: 1,377,179 tokens, which holds 4.17 requests at the token cap (327,680
    generated plus the longest prompt, 2,795). At 8 concurrent GPQA requests the pool was full for much of the run and
    requests waited (653 of 1,417 status lines); at 4 none waited (peak usage 91.7%, no preemption). On tpurtell 0.7.0 +
    kpool fixes it has 437,563 tokens, and the engine crashed when the pool filled at 8 concurrent requests.
18. Serving probe, one run per configuration, no noise floor. Kpool fixes: per-request decode speed -3.5% to +1.7%,
    acceptance within 0.004. DCP1 tail fix: 147.3 vs 146.9 tok/s at concurrency 1 and 62.5 vs 64.9 at 8, acceptance
    0.534 vs 0.526 and 0.529 vs 0.550. Draft acceptance at 8 concurrent requests was 0.5285 on unpatched tpurtell 0.9.0
    against 0.548-0.552 on the other DFlash2 ×3 builds (single runs).
19. The same configuration run twice with greedy decoding, one request at a time, diverges after a median of 318
    characters. Two decode-vs-prefill runs of tpurtell 0.9.1 with the same prompts and seeds, one request at a time,
    first differ in their per-position values after 1 to 130 generated tokens. Batching cannot explain this; at 8 or 12
    concurrent requests batching adds further variation. Greedy parity therefore cannot certify speculative exactness
    on this stack.

**19a.** Release control of the completion investigation's bundle gate (2026-10-12), stock 0.9.1 in shipped graph mode,
temperature 0, two rounds in one server session. The 4,240-token prompt diverged with identical prefix-cache
coverage in both rounds; log-probabilities differed on every token before divergence. The same pattern held in two
sessions of the corrected bundle (first divergence at tokens 3-75). Divergence fell at steps where the chosen token
had probability 0.30-0.66; the runner-up was not recorded, so the margin is bounded, not measured. Any single-pass
or repeat comparison inherits this ([confounds](CONFOUNDS.md#16-long-context-greedy-non-repeatability)).

</details>

## Open questions

Stated with their evidence. Grade: **open**.

| Question | Evidence so far |
|---|---|
| What drives non-completion on questions 88 and 79? | Items 13-14: no tested runtime part moved either, at a size where only very large effects could show. |
| What makes tpurtell 0.7.0 as shipped leave fewer GPQA questions unanswered than 0.9.1? | Item 7: 4 vs 16 of 594, almost all at the 327,680-token cap; several differences at once, none varied alone in GPQA. The component screen ran the 0.7.0 image at three draft tokens on question 79 only, where it failed as often as 0.9.1 (9 of 12 each). The [completion investigation](investigations/2026-10-glm53-completion/README.md) found two localized defects (items 4a-4b): the column-511 drop, the only one that differed in the direction of the gap, is not supported as its explanation (arm: 15 vs 16 empty); the tail ring affects 0.7.0 more, which does not exclude an accidental completion benefit; its role is untested. A pilot of 0.7.0's layout bundle on 0.9.1 gave no useful separation; router precision and the SwiGLU clamp are the same in all releases. Two GPQA passes of 0.9.1 with four corrections together left 3 and 3 of 198 empty against 5 and 5 in the matching 0.9.1 passes, with no clear separation (item 4e). As evidence only: the bundle corrects verified defects and shows a modest, consistent reduction in empty replies at two seeds; not established. Stock 0.7.0's lower rate (0 and 3 at the same seeds) is not reached, and three of the four corrections make 0.9.1 more correct than 0.7.0 rather than more like it, so other contributors to the historical gap likely remain unidentified; a cumulative effect of several small corrections is a working hypothesis, not a result. The KV record writer and query projection, which differ between the engines, agree within rounding (item 4h), so they are ruled down as a mechanism beyond rounding. The gap rests on one three-pass record per engine. The corrected 0.9.1 arm (15) resembles the 0.9.1 record (16), with empties moving between questions, but it ran a modified engine and is not a replication of the unmodified record; whether either engine's rate reproduces unmodified has not been tested. Which difference matters is inconclusive. |
| Does the column-511 correction change completion or accuracy? | Item 4a: the localized defect is removed. A three-pass GPQA arm with this correction alone, against the existing 0.9.1 record (a historical control) and read by rules frozen before data, left 15 vs 16 empty replies of 594 (one-sided p = 0.50): not supported as the dominant completion explanation. Raw accuracy -1.85 points (95% -5.05 to +1.35), stated -1.18 (95% -4.04 to +1.52, final after the hand read); inconclusive for accuracy. The tail-ring defect, present in both records, limits what this negative result can exclude ([completion arm](investigations/2026-10-glm53-completion/README.md#completion-arm)). |
| Do the tail-ring fix, the router's bf16 logits or the routed-expert SwiGLU clamp change completion or accuracy? | All three affect every release (the tail ring 0.7.0 more). Their computational effects are measured or reachable (items 4b-4d); none was tested alone in GPQA. Together with the column-511 correction, two passes: 3 and 3 of 198 empty against 5 and 5 in the matching 0.9.1 passes; raw 174 and 168, stated 176 and 169 (single passes, secondary); no clear separation in either (item 4e). Untested individually ([confounds](CONFOUNDS.md#10-kpool-tail-ring-positions-omission), [13](CONFOUNDS.md#13-router-precision-and-the-routed-expert-swiglu-clamp)). |
| Does generation-vs-prefill disagreement at long context bear on completion? | The completion investigation's 8,000-15,999-token measure covers continuation after the reply had ended (its amendment 7); no evidence either way ([confounds](CONFOUNDS.md#15-long-context-decode-vs-prefill-measures-continuation-after-the-reply)). |
| Do other releases or layouts differ in non-completion across the benchmark? | Only pass 1 exists for the other configurations: 0 to 9 empty answers of 198, no paired difference at p < 0.05 (item 10). 3.25bpw and 4bpw TR3 (Brandon) on 0.9.1: no measurable difference over three passes (16 vs 17; item 6). |
| Does the DCP1 tail fix change answers or how often hard questions fail to finish? | It changes decode numerics in the first 2,044 tokens; the loops observed are stopped far later; GPQA and tool calling show no difference beyond noise at their sizes (items 3, 4, 16). |
| Why is the engine not bitwise reproducible one request at a time? | Item 19: greedy reruns diverge after a median of 318 characters; same-seed decode-vs-prefill runs first differ after 1 to 130 tokens, with no concurrent requests. Item 19a: short prompts repeat bitwise for 96 tokens, long-context prompts diverge at token 3 on the stock release, at low-margin steps (margin bounded, not measured). Mechanism not identified. |
| Do batch-size-dependent routing weights (item 4f) change outputs? | Weights differ by up to 1.1e-3 relative above about 1,024 rows (p99 5.6e-7); expert ids unchanged; inferred exposure prefill only; expected numerical variation of the upstream kernel, present in every release. Untested. |
| Does D7 (items 4j-4k) affect selection, outputs or completion? | Corrected and verified in B1. Present in every release. In the GPQA records exposure is narrow and opposite to the gap (reconstructed). The selection effect was not measured. Untested; behavioural investigation closed for now. |
| What do the targeted checks on the corrected build leave unexamined? | Items 4g-4i found no defect; item 4j checked slot-mapping construction under real scheduling and found only D7. Not covered: copy-on-write prefix paths, contention-dependent races, the values the index-K readers compute, extend under graph replay, communication paths other than the tested transport, and the mixed-quant MoE path, whose reference harness was deferred. No evidence yet points to any of these. |
| Do these quants cost accuracy? | No higher-precision reference (BF16 or NVFP4) was run on this hardware. The published 90.6-92.1 used other weights and harnesses that are not fully published, so the gap to them is not a measure of quantization (item 12). |
| Code-level questions: the 511-pool slice, `swiglu_limit` on the routed experts, top-k ties between layouts | Read in code ([investigation](investigations/2026-10-glm53-looping/README.md#open-questions)). Since measured: the 511-pool slice (item 4a). `swiglu_limit` on the routed experts: omission confirmed and the fix reachable in served kernels (item 4d); whether the limit engages is untested. Top-k ties: the selected set matches an exact top 512 apart from boundary ties; not measured further. |

## Labels

Configurations are named **weights · engine version · speculation**, built from the config files by one rule
([`SCHEMA.md`](SCHEMA.md#labels)). The engine name comes first because engines number their versions independently.

<details>
<summary>What each label means</summary>

| Label | Meaning |
|---|---|
| **tpurtell 0.7.0** | Release v0.7.0 of [tpurtell/glm-5.3-flash-ext3-2x-rtx](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx), an engine built on vLLM, as published |
| **tpurtell 0.7.0 + kpool fixes** | v0.7.0 with the kpool fixes applied locally. A backport: not a release, and not 0.9.0 |
| **tpurtell 0.8.0** | Release v0.8.0 as published (no kpool fixes) |
| **tpurtell 0.8.0 + kpool fixes ≈ 0.9.0** | v0.8.0 with the kpool fixes applied locally. **The same engine as tpurtell 0.9.0 for every measurement here:** its kpool kernel files are byte-for-byte identical to the 0.9.0 release, and 0.9.0's other two changes (an opt-in boundary prefix-cache lookup, off by default, and usage reporting) do not affect these measurements. Measured before 0.9.0 was released |
| **tpurtell 0.9.0** | Release v0.9.0 as published, at its defaults (includes the kpool fixes) |
| **tpurtell 0.9.0 + DCP1 tail fix ≈ 0.9.1** | v0.9.0 with the fix of [tpurtell/glm-5.3-flash-ext3-2x-rtx#6](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/6) applied locally, measured before the fix was merged. **The same engine as tpurtell 0.9.1 for every measurement here:** 0.9.1 (released 2026-10-08 from that merge) ships the fixed attention file byte-for-byte as built from the pull request, which differs from the build measured here only in identifier names; its other kpool and indexer files are those of 0.9.0 |
| **tpurtell 0.9.1** | Release v0.9.1 as published: v0.9.0 plus the DCP1 tail fix |
| Layout segments | Shown only when a configuration runs a parallel layout other than its release's default: `0.7.0 layout (DCP2, EP2)` on the 0.9.0 image, and the component-screen arms `EP2 experts, NOPE records off`, `EP2 experts, MLA owners tp` and `NOPE records off, MLA owners tp` on 0.9.1. Screen arms and diagnostic controls, not recommendations |
| concurrency N | Shown only when the client kept N requests in flight instead of the protocol's setting (GPQA: 8): `4bpw TR3 (Brandon) · tpurtell 0.9.1 · DFlash2 ×3 · concurrency 4`, because that KV pool holds about 4 requests at the token cap |
| kpool fixes | Upstream vLLM fixes vllm-project/vllm#57477 and vllm-project/vllm#58454, ported in [tpurtell/glm-5.3-flash-ext3-2x-rtx#5](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/5) (commit 5a366b5) and shipped in v0.9.0 |
| 3.25bpw | tpurtell's K3.25 checkpoint [wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1) (EXL3, mixed K3/K4 routed experts) |
| 4bpw TR3 (Brandon) | Brandon M. Music's TR3 checkpoint [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) (EXL3, uniform K4 routed experts) |
| DFlash2 ×N | DFlash2 speculative decoding (incoai/GLM-5.3-Flash-DFlash2), N draft tokens per step; "no speculation" = plain decoding |

All engines here are tpurtell builds. 0.7.0 and the 0.8.0-0.9.1 line also differ in their default parallel layout
(0.7.0: EP2 + DCP2, vision on; 0.8.0 onward: EP1 + DCP1 with MLA layer ownership, vision off), so "engine" here means
the release as shipped.

</details>
