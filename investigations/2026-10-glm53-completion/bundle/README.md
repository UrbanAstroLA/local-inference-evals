# Correctness bundle on 0.9.1: gate and one-pass GPQA screen (amendments 14-14e)

**Result.** A bundle of four corrections on tpurtell 0.9.1 K3.25 (D1 column-511 selection, D2 tail-ring positions, D3
fp32-output router, D4 routed SwiGLU limit 10) passed its gate for proceeding to a screen, with two criteria revised after
the gate data were inspected (below). One GPQA pass with the bundle (seed 1234, all 198 questions, original protocol)
was valid and left **3 of 198 replies empty, all at the 327,680-token cap**: category **SIMILAR / NO CLEAR SEPARATION**
under rules frozen before generation. Per those rules the bundle is deprioritised as a completion remedy within this
allocation. This is not a finding that the included corrections lack behavioural effects. Cost: gate 1.6403 of 2.0,
screen 4.841 of 7.5 occupied GPU-hours.

Plan and rules: [`../PLAN.md`](../PLAN.md#dated-amendments), amendments 14-14e. Design and the gate review:
[`PROPOSAL.md`](PROPOSAL.md) (with two addenda). Investigation page:
[bundle gate](../README.md#bundle-gate-d1d2d3d4-on-091) and [screen](../README.md#one-pass-gpqa-screen).

## Bundle

| Id | Correction | Files | In the bundle |
|---|---|---|---|
| D1 | column-511: discard the lowest-scoring selected pool (a correction to the engine's 511-pool selection policy, not a restoration of upstream's 512-pool policy) | `sparse_attn_indexer_kpool.py` | yes |
| D2 | pass token positions to the attention metadata, upstream's ownership guard, graph-safe tail-slot write (variant U) | `indexer.py`, `mamba_hybrid.py` | yes |
| D3 | router GEMM with fp32 output on SM12x (port of vllm#54048) | `gate_linear.py` | yes (its offline verification passed first) |
| D4 | pass the checkpoint's `swiglu_limit` (10) to the projection-mixed EXL3 routed experts | `exl3.py` | yes |
| D5 | lock the workspace after graph capture (port of vllm#55341) | | no: not exercised in stage A, and the backport reorders kernel warm-up and capture |
| D6 | stale warm-up tail block ids | | no: unreachable under D2 (stale columns never read; 16 stale rows per rank, 0 foreign writes) |

Frozen candidate: image `sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548`, built from the 0.9.1
release digest; combined diff `949b2301…` (5 files, +66/−22; [`freeze/combined-d1234.diff`](freeze/combined-d1234.diff)),
component diffs in [`freeze/patches/`](freeze/patches). A whole-image file manifest shows no differences from the release
other than these files, their bytecode and `/etc/hostname` ([`freeze/manifests/`](freeze/manifests)). CPU tests on the
installed files of each image: [`cpu-tests/summary.txt`](cpu-tests/summary.txt). The gate ran a verification image built
from the candidate that adds observers only (additions only, inert unless enabled); the screen ran the candidate image.

## Gate

Three server sessions in shipped graph mode (FULL_AND_PIECEWISE, DFlash2 ×3) and one offline check. Runner, client,
analysis and verdict scripts: [`gate/`](gate). Verdict outputs: [`gate-runs/`](gate-runs).

| Session | Image | What it checks | Occupied GPU-hours |
|---|---|---|---:|
| s1 | verification image, all observers | D1-D4 active, short contexts, mixed batches, request reuse, errors, throughput lower bound | 0.7711 |
| s3 | verification image, D4 limit overridden to 1 | D4 real-weight reachability (6-prompt greedy probe vs s1) | 0.4506 |
| rel | stock 0.9.1 release, no observers | probe control, added during the gate review | 0.4111 |
| D3 as served | offline, on s1's capture (attempt 1 did not start: a mount path; 0 s) | fp32 router path as served | 0.0075 |
| **Total** | | | **1.6403** (ceiling 2.0) |

**Original criteria that failed, as specified.**
- "D4 compiled kernels": a host-side walk of the plan objects was to find `has_swiglu_limit=True` with only the value 10
  on the compiled kernels. It found 10.0 on every plan object (6 per rank) but could not reach kernel objects: the
  projection-mixed runtime does not retain them. The walk had been CPU-tested only on a stand-in kernel object placed in
  a plan, an unverified assumption made in preparing the gate.
- "P1 probe repeatable": both greedy rounds identical on all 6 probe prompts. The 4 short GPQA prompts were bitwise
  repeatable in every session; the 2 long-context prompts (4,240 and 16,240 tokens) were not, in the bundle sessions
  and in the stock release alike.

**Revisions, made after inspecting gate data and accepted in amendment 14c.** The original criteria are not described as
met.
- D4: session-specific compiled-kernel evidence replaces the unreachable attribute. The b12x on-disk compile cache holds
  66 mixed-trellis kernels with (has_swiglu_limit, swiglu_limit) = (True, 10.0) in all three kernel keys, 33 per GPU,
  written inside s1's window; none with limit 10 existed before s1; with the limit fields neutralised all 66 equal
  pre-gate release keys; s3 wrote the same shapes with (True, 1.0)
  ([`d4-compile-cache-evidence.txt`](gate-runs/d4-compile-cache-evidence.txt),
  [`d4-cache-provenance.txt`](gate-runs/d4-cache-provenance.txt)).
- P1: within-session repeatability on the short prompts, where the release is repeatable. Met in s1 and s3. Long-context
  non-repeatability is reported as a limitation (below), with no claim of long-context numerical equivalence, of no
  additional bundle variability, or of a demonstrated near-tie mechanism.

**Checks that passed as specified (s1, both ranks).**

| Check | Result |
|---|---|
| Validity and errors | every planned phase ran; 0 of 140 request errors; fixed-length requests finished by length; 0 stopping errors; no traceback, memory, CUDA, workspace, assertion, preemption or NaN lines; one server start; configuration identical to the historical 0.9.1 record (23/23 arguments, 33/33 engine fields, graph captures 11/9/9); served engine files equal the frozen files |
| D1 | column 511 holds the lowest selected pool on every checked row, set unchanged, consumers read it; fp64 re-score agrees on every clean sampled row (decode fused and tiled, prefill, including 75,306 rows from FULL-graph replays); 0 checker errors or eager fallbacks |
| D2 | circular tail-slot mapping in use; 998,640 and 832,200 tail writes; 0 null-block, foreign, attention-block or tail collisions |
| D3 | fp32-output dispatch on 42 layers × 2 ranks; 0 of 69,344 sampled rows bf16-valued |
| D4 plans | layer limit 10; every plan carries 10.0 |
| D5 (excluded) | no workspace resize after capture: not exercised in this workload |

**D4 reachability.** Under the frozen no-control rule, 6 of 6 probe prompts differ between limit 1 (s3) and limit 10
(s1), all within the first 16 tokens. The informative ones are the 4 short prompts: bitwise repeatable within each
session, diverging at tokens 0-2 with token-0 log-probability shifts of 0.03-0.71. Reachable. Whether limit 10 changes
outputs (whether the clamp engages) was not measured.

**D3 as served** ([`d3-served-a2/`](gate-runs/d3-served-a2/d3_served_summary.json)). 1,000 captured records, 69,344 rows,
both ranks: fp32-output dispatch and matching weight hashes on 84 of 84 layer-ranks; `torch.mm(x, W.T, out_dtype=fp32)`
at the serving batch size reproduces the served logits bitwise (1,000 of 1,000); 0 of 19,971,072 logits outside τ_d of
fp64 (maximum error/S 6.8e-7), 0 top-8 disagreements; routing at the serving batch size reproduces the consumed routing
bitwise (69,344 of 69,344 rows).

**Pre-launch items (amendment 14d).**
- The gate's GPQA-shaped phase had four early stops, not three (docs 16, 19, 21, 22; 726-4,096 tokens). Each has non-empty
  final content and reasoning and no request or HTTP error; the client kept lengths only, so "answered" means non-empty
  final content (the answer parser was not run). These questions stop at similar lengths in the 0.9.1 record and the
  column-511 arm. That phase's throughput (244.7 tok/s aggregate, mostly 5 or fewer in flight) is unusable for pricing.
- The production image's five engine files equal the frozen candidate files, with no observer code; the verification
  image differs only by additions ([`production-image-check.txt`](gate-runs/production-image-check.txt)).
- Throughput with all observers on: 82.2 / 55.1 tok/s on the long-context T/F phases, 0.93 of the release-with-guard
  reference; a lower bound that does not separate observer cost from bundle cost.

### Recorded separately (evidence only)

- **Long-context greedy non-repeatability of the stock 0.9.1 release.** In the release control, the 4 short prompts
  (124-158 tokens) give bitwise identical 96-token greedy outputs in both rounds; the 4,240- and 16,240-token prompts
  diverge at token 3, with nonzero log-probability differences on every token before divergence. The 4,240-token prompt
  diverges with identical prefix-cache coverage in both rounds (0 cached), in every session. In the bundle sessions first
  divergence comes at tokens 3-75. Divergence was observed at steps where the chosen token's probability was about
  0.3-0.66; only the chosen token's log-probability was recorded, so the margin to the runner-up is bounded, not
  measured. Mechanism not identified ([`probe-divergence.json`](gate-runs/probe-divergence.json); token strings withheld
  because the long prompts are model reasoning on GPQA questions).
- **NaN-byte index rows.** 0 in every branch in s1, against thousands of sampled rows in the 2026-10-11 gate (column-511
  correction only, otherwise the release: 7.9% / 32.5% of sampled decode rows on ranks 0 / 1). D2 is inferred as the
  cause; it was not varied alone.
- **E1** (stage A, [`../verify-stage-a/`](../verify-stage-a)): the served fused router's routing weights depend on the
  batch row count; no workaround entered the bundle.

## Screen

[`screen/`](screen). One pass, frozen before generation (amendment 14d): candidate image above, recipe
`glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234`, all 198 questions, temperature 1.0, top_p 0.95, 8 concurrent,
327,680-token cap, seed 1234 (pass 1 of the 0.9.1 record, of the column-511 arm and of 0.7.0), the documented server
lifecycle and warm-up gate, a 7.5 GPU-hour watchdog, no retry. Analysis `screen_analysis.py` (local sha256
`9c799f26…`), runner `gpqa-bundle-screen.sh` (`e8caf211…`).

Rules (amendment 14c; resource-allocation rules, not significance or equivalence claims): E = empty replies of 198.
Clear promising: E ≤ 2, no regression flag and no accuracy concern. Mixed: E ≤ 3 with a flag or concern. Similar / no
clear separation: E = 3 without either. Similar or worse: E ≥ 4. Regression flags: raw correct ≤ 158, stated ≤ 159, or 2
or more premature stops; accuracy concern: raw 159-163. Simulated P(E ≤ 2) (accuracy and stopping not modelled): 0.078
with no change, 0.85-0.86 at 0.7.0-like rates, 0.50 at half the 0.9.1 rates ([`power-onepass.txt`](screen/power-onepass.txt)).

**Result** ([`run-d1234/screen-result-final.json`](screen/run-d1234/screen-result-final.json)). Valid (198 requests,
none re-sent, all HTTP 200, seed 1234, one server start, no preemption, illegal-memory or NaN lines). Stated-answer
adjudication complete: 2 rows (docs 58 and 98), each commits to a single option, neither hedged
([`adjudication.json`](screen/run-d1234/adjudication.json); ids and verdicts only).

| Pass 1, seed 1234 | Bundle (this screen) | 0.9.1 record | Column-511 arm | 0.7.0 (context) |
|---|---:|---:|---:|---:|
| Empty replies | **3** (3 cap, 0 premature) | 5 (4 cap, 1 premature) | 4 (4 cap) | 0 |
| Rescued by the bundle | | q12, q109, q120 | q170 | |
| Newly empty with the bundle | | q79 | none | q79, q88, q127 |
| Empty in both | | q88, q127 | q79, q88, q127 | |
| Raw correct (secondary) | 174 | 169 | 164 | 174 |
| Stated correct (secondary) | 176 | 170 | 165 | 177 |

- Category: **SIMILAR / NO CLEAR SEPARATION** (E = 3; no regression flag, no accuracy concern).
- Secondary figures are a single pass and were not a decision input; no significance claim is made for them. The frozen
  output also carries paired reference tests, reported there as references only.
- Answered completion lengths: median 3,907 tokens, 90th percentile 31,682, maximum 101,893; 19 above 32,000, 8 above
  64,000.
- Request window 2.22 h at 414 tok/s aggregate; server mean 449 tok/s with 8 or more running; 87% of busy time had 5 or
  more requests in flight.
- Cost 4.841 occupied GPU-hours (2 GPUs × 8,713 s, server start to stop); ceiling 7.5.
- Limits: one pass of a low-rate outcome; historical controls with exchangeability assumed; a matched seed matches the
  sampling stream only until the first computational divergence, and D3 alone changes about 3% of expert sets, so the
  replies are not paired copies; long-context greedy decoding of the release is itself not repeatable (above).
- The plan records an operator note after the result: the operator considers another pass warranted, citing seed-1234
  luck across earlier GPQA runs, and will raise it with the independent assessor. The note is not an authorization, and
  no further GPU work is authorized.

## Files

- [`PROPOSAL.md`](PROPOSAL.md): build, CPU tests, gate and screen design, D5/D6 decisions, the gate review and the
  pre-launch items.
- [`freeze/`](freeze): combined and component diffs, Dockerfiles, image ids, verification-image observer diffs and the
  D4 inspection module (`d4limit.py`) with its tests, build and test scripts, whole-image manifests.
- [`cpu-tests/`](cpu-tests): per-image CPU test outputs and the summary.
- [`gate/`](gate): runner, client, analysis, verdict, D3-as-served check, fixture builder, image list, unchanged tool
  copies. The GPQA fixture itself (benchmark questions) is local only; its hash is in the raw list.
- [`gate-runs/`](gate-runs): per session `verdict.txt`/`.json`, `gate.txt`, configuration comparison and, for s1, the
  gate summary, column-511 report, tail-guard and D3/D5 readouts and throughput; D3 as served; D4 cache evidence;
  production-image check; probe divergence.
- [`screen/`](screen): analysis, runner, power simulation, dry runs on the column-511 arm's pass 1, the result before
  (`screen-result.*`) and after (`screen-result-final.*`) adjudication, the adjudication ids, cost.
- [`costs.jsonl`](costs.jsonl), [`screen/costs.jsonl`](screen/costs.jsonl): cost lines as logged.
- [`RAW-SHA256SUMS-bundle.txt`](RAW-SHA256SUMS-bundle.txt): sha256 of files kept local (server logs, request files,
  client and probe records, capture and observer records, container inspection, pre-flight listings, build logs, full
  engine files the diffs reproduce, the GPQA fixture).
- Public copies replace local paths, image-name prefixes and module names (e.g. `local/…` images, `d4limit.py`,
  `col511check`, `kpool_tailguard`, `verify13_capture`), so they are not byte-identical to the files that ran or were
  frozen; [`../PUBLIC-COPIES-14.txt`](../PUBLIC-COPIES-14.txt) lists both hashes. The frozen-file hashes in amendment 14d
  refer to the local files.
