# local-inference-evals

Maintained by Michael M: [UrbanAstroLA](https://github.com/UrbanAstroLA) on GitHub, [@UrbanAstroFella](https://x.com/UrbanAstroFella) on X.

Receipts for evaluations of locally served LLMs: what was run, on exactly which software and hardware, the raw
per-item results, and the scripts that recompute every published number. Null and negative results are kept.

The current subject is GLM-5.3-Flash EXL3 quants on 2x RTX PRO 6000, served by builds of tpurtell's vLLM-based engine
[tpurtell/glm-5.3-flash-ext3-2x-rtx](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx). It is a public ledger of
evaluating the runtimes and quants, and the confounds that can move accuracy or speed.

## Read

| Page | What it answers | Source |
|---|---|---|
| [Results](https://urbanastrola.github.io/local-inference-evals/) | What to run on 2x RTX PRO 6000: GPQA, speed, KV capacity, kernel tests. **The summary is here.** | [`FINDINGS.md`](FINDINGS.md): every statement, graded |
| [Confounds](https://urbanastrola.github.io/local-inference-evals/confounds.html) | What else can move a result, and how the runs control for it | [`CONFOUNDS.md`](CONFOUNDS.md) |
| [Ledger](https://urbanastrola.github.io/local-inference-evals/ledger.html) | What was run, found, corrected and withdrawn, and when | [`LEDGER.md`](LEDGER.md) |
| [Investigation](https://urbanastrola.github.io/local-inference-evals/looping.html) | Why some hard questions do not finish | [`investigations/2026-10-glm53-looping`](investigations/2026-10-glm53-looping) |
| [Completion audit](investigations/2026-10-glm53-completion/README.md) | Why 0.7.0 left fewer GPQA answers empty than 0.9.1 (Phase 0: existing data; Phase 1: decode vs prefill on both engines) | [`investigations/2026-10-glm53-completion`](investigations/2026-10-glm53-completion) |
| [Method](https://urbanastrola.github.io/local-inference-evals/method.html) | Grades, scoring, intervals, receipts, glossary | [`protocols/`](protocols), [`DATASHEET.md`](DATASHEET.md) |

**The DCP1 tail issue and its fix** (tpurtell PR #6, released in tpurtell 0.9.1):
[record](investigations/2026-10-glm53-looping/README.md#the-dcp1-tail-issue-and-its-fix).

## Labels

Configurations are named **weights · engine version · speculation**, for example
`4bpw TR3 (Brandon) · tpurtell 0.9.0 · DFlash2 ×3`. Every engine so far is a build of tpurtell's engine.

<details>
<summary>Label details</summary>

- The engine name comes first because engines number their versions independently.
- `tpurtell 0.8.0 + kpool fixes ≈ 0.9.0` is 0.8.0 with the two upstream kpool fixes that later shipped in 0.9.0. It is
  the same engine as `tpurtell 0.9.0` for every measurement here, and was measured before 0.9.0 was released.
- `tpurtell 0.7.0 + kpool fixes` is a separate backport of the same fixes: not a release, and not 0.9.0.
- `tpurtell 0.9.0 + DCP1 tail fix ≈ 0.9.1` is 0.9.0 with the fix of tpurtell/glm-5.3-flash-ext3-2x-rtx#6, applied locally
  and measured before the fix was merged. The fix shipped in 0.9.1 (2026-10-08); it is the same engine as
  `tpurtell 0.9.1` for every measurement here.
- A layout segment such as `0.7.0 layout (DCP2, EP2)` or `EP2 experts, NOPE records off` appears only when a
  configuration runs a parallel layout other than its release's default (screen arms and diagnostic controls).
- Plain `tpurtell 0.7.0`, `0.8.0`, `0.9.0` and `0.9.1` are the releases as published.
- A trailing `· concurrency N` appears only when the client kept N requests in flight instead of the protocol's setting
  (GPQA: 8), as for `4bpw TR3 (Brandon) · tpurtell 0.9.1 · DFlash2 ×3 · concurrency 4`.
- Weights: `3.25bpw` is tpurtell's K3.25 checkpoint wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1; `4bpw TR3 (Brandon)` is
  Brandon M. Music's TR3 checkpoint brandonmusic/GLM-5.3-Flash-tr3-4bpw. Speculation: `DFlash2 ×N` is DFlash2 with N
  draft tokens per step.
- Full key: [`FINDINGS.md`](FINDINGS.md#labels). Labels are built from config fields by one rule, so results from
  another engine carry its own name and version without code changes ([`SCHEMA.md`](SCHEMA.md#labels)).

</details>

## Layout

<details>
<summary>Where things are</summary>

```
protocols/<benchmark>/v<N>.md     frozen, versioned evaluation procedures (settings, scoring, dataset revision)
configs/<model-family>/<id>.json  exact serving configurations: engine image digest, model revision, overrides, hardware, labels
runs/<run-id>/                    one evaluation of one config under one protocol
    run.json                      manifest: config id, protocol id, dates, software versions
    results.jsonl                 one line per item (question and pass, request, test) - no benchmark text
    summary.json                  aggregates, recomputable from results.jsonl (and server_log.jsonl)
    server_log.jsonl              optional: numeric fields parsed from the engine's server log (KV pool, throughput)
investigations/<yyyy-mm>-<topic>/ narrative, preregistration and decision rules for a line of work, linking its runs
comparisons/<id>/                 apples-to-apples analyses of several runs (see the rules below)
tools/verify.py                   checks manifests, recomputes every summary, and enforces the comparison rules
docs/                             the results site (GitHub Pages), built from runs/ by tools/site.py
tools/analyze.py                  recomputes the analyses in FINDINGS.md from published rows
SCHEMA.md, DATASHEET.md           field definitions; provenance, terms and intended uses
FINDINGS.md, CONFOUNDS.md         graded statements; confounds and their controls
LEDGER.md, CHANGELOG.md           dated account in plain language; technical change history
```

</details>

## Comparability rules

<details>
<summary>What may be compared, and how</summary>

A comparison may only line up runs that share the same **protocol id and version** and the same **hardware**.
`comparison.json` lists the runs and names the configuration fields that are meant to differ; `tools/verify.py`
fails if anything else differs. Cross-protocol or cross-hardware tables are allowed only when marked
`"comparable": false` with a stated reason, and are reported as context, never as a ranking.

Every accuracy is reported with a 95% confidence interval. Differences inside the interval are ties. Runs that send
one request seed on every repeat (`hard-prompt-screen` v0 and v1) are single draws per question and cannot be compared;
GPQA passes must each carry their own request seed. `tools/verify.py` enforces both. Runs with several passes of the
same questions are compared with questions as clusters.

</details>

## Benchmark data

<details>
<summary>Why there is no question text</summary>

Some benchmarks ask that their questions not be published in plain text (GPQA does). Results here therefore carry
item ids and hashes (`doc_hash`, `prompt_hash`, `target_hash` as logged by lm-evaluation-harness) instead of text.
Anyone with the dataset can regenerate the prompts from the protocol and confirm the hashes match.

Two kinds of figure rest on text that cannot be published (model output): greedy shared-prefix lengths
(`comparisons/glm53-flash-serving-probe/parity.json`) and the per-row audited `correct_stated` judgement; for both, the
outputs' hashes are published.

</details>

## Verify

```bash
python3 tools/verify.py            # all runs and comparisons
python3 tools/verify.py runs/<id>  # one run
```
Python 3.10+, standard library only.

## Glossary

<details>
<summary>Terms used across the repository</summary>

- **MLA** (multi-head latent attention): GLM-5.3-Flash's attention in 11 of its layers (the others are KDA layers). Keys
  and values are cached as a compressed latent, and a sparse-attention indexer picks which earlier tokens each decode
  step attends.
- **kpool:** the indexer's pooled key cache: earlier tokens are grouped in pools of 4, and each decode step selects up to
  512 pools (2,048 tokens). The **kpool tail** is the newest, incomplete pool: the current token and up to two before it.
- **DCP1 / DCP2:** decode context parallelism. DCP2 splits each MLA layer's KV cache across both GPUs by token; DCP1
  does not split it.
- **MLA ownership:** tpurtell's DCP1 layout from 0.8.0 on: each MLA layer, with its indexer and caches, lives on one GPU
  only (`split:25`: layers 3-23 on the first GPU, 27-43 on the second) and runs all heads there. `tp` is the setting
  without ownership.
- **EP2 / TP2:** routed experts placed whole on one GPU each (expert parallel) or each expert split across both GPUs
  (tensor parallel).
- **NOPE record:** the KV cache record of an MLA latent. tpurtell 0.8.0 on stores a 528-byte record (512 FP8 bytes and
  four FP32 scales); "NOPE records off" uses the 656-byte record with an unused rotary tail. tpurtell reports that outputs
  match within BF16 rounding.
- **DFlash2 ×N:** speculative decoding with the draft model incoai/GLM-5.3-Flash-DFlash2, which proposes N tokens per
  step; the served model checks them in one forward pass and keeps those it accepts. Acceptance rate = accepted /
  drafted tokens.
- **Draft-slot sharing:** from tpurtell 0.8.0 on, the DFlash2 draft cache is stored inside the MLA cache's slot tensors
  instead of separate tensors; "sharing off" disables it.
- **flexible-extract:** lm-evaluation-harness's GPQA answer filter: the last parenthesised capital letter in the reply.
  The headline (raw) GPQA score. **`correct_stated`** is the secondary, audited score: whether the reply's stated final
  answer is right.
- **Empty answer:** a GPQA reply with no content after its reasoning: the reasoning hit the 327,680-token budget, or ended
  without an answer. Scored wrong.
- **Loop:** a request that does not finish and repeats itself: stopped by the screen's early-stop detector, or ending at
  the budget with a tail that compresses below 15% of its size.
- **Exhaustion:** a request that runs to the 327,680-token budget with varied, non-repetitive reasoning.
- **Pass:** one run of all 198 GPQA questions. Pass *p* sends request seed 1233 + *p* (1234, 1235, 1236), so the passes of
  one configuration are independent draws, and pass *p* of two configurations shares its seed, which pairs them question
  by question.
- **KV pool / KV-saturated:** the KV cache capacity in tokens that the server reports at start-up. When the running
  requests fill it, further requests wait, so fewer run at once than the client sends (KV-saturated).
- **Question-clustered test:** a comparison that counts a question answered in several passes once, not once per pass:
  here an exact sign-flip test over questions (each question's difference, summed over its passes, keeps or flips its
  sign) with intervals that resample questions. A pooled test that counts every question-pass separately overstates the
  evidence and is shown only for reference.

</details>

## Licenses

Code: Apache-2.0 (`LICENSE`). Results and documentation: CC BY 4.0 (`LICENSE-DATA.md`). Model, engine and dataset
licenses belong to their owners and are linked from each config and protocol.
