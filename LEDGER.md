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
- Correction: earlier text said both servers loaded the same vendored chat template. They did not. 0.7.0 served the
  checkpoint's own template, 0.9.1 the corrected Z.ai template. Both render all 198 GPQA prompts byte-identically, so
  no number changes.

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

</details>
