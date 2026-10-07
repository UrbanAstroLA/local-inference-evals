# local-inference-evals

Maintained by Michael M: [UrbanAstroLA](https://github.com/UrbanAstroLA) on GitHub, [@UrbanAstroFella](https://x.com/UrbanAstroFella) on X.

Receipts for evaluations of locally served LLMs: what was run, on exactly which software and hardware, the raw
per-item results, and the scripts that recompute every published number. Null and negative results are kept.

Start with [`FINDINGS.md`](FINDINGS.md) for results and [`DATASHEET.md`](DATASHEET.md) for what the data is and is not.
Results site: <https://urbanastrola.github.io/local-inference-evals/>.

## Labels

Configurations are named **weights · engine version · speculation**, for example `4bpw · tpurtell 0.9.0 · 3 drafts`.
The engine name comes first because engines number their versions independently. Every engine so far is a build of
[tpurtell/glm-5.3-flash-ext3-2x-rtx](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx), which is built on vLLM:

- `tpurtell 0.8.0 + kpool fixes ≈ 0.9.0` is 0.8.0 with the two upstream kpool fixes that later shipped in 0.9.0. It is
  the same engine as `tpurtell 0.9.0` for every measurement here, and was measured before 0.9.0 was released.
- `tpurtell 0.7.0 + kpool fixes` is a separate backport of the same fixes: not a release, and not 0.9.0.
- Plain `tpurtell 0.7.0`, `0.8.0` and `0.9.0` are the releases as published.

Full key: [`FINDINGS.md`](FINDINGS.md#labels). Labels are built from config fields by one rule, so results from
another engine carry its own name and version without code changes ([`SCHEMA.md`](SCHEMA.md#labels)).

## Layout

```
protocols/<benchmark>/v<N>.md     frozen, versioned evaluation procedures (settings, scoring, dataset revision)
configs/<model-family>/<id>.json  exact serving configurations: engine image digest, model revision, overrides, hardware, labels
runs/<run-id>/                    one evaluation of one config under one protocol
    run.json                      manifest: config id, protocol id, dates, software versions
    results.jsonl                 one line per item (question, request, test) - no benchmark text
    summary.json                  aggregates, recomputable from results.jsonl
investigations/<yyyy-mm>-<topic>/ narrative, preregistration and decision rules for a line of work, linking its runs
comparisons/<id>/                 apples-to-apples analyses of several runs (see the rules below)
tools/verify.py                   checks manifests, recomputes every summary, and enforces the comparison rules
docs/                             the results site (GitHub Pages), built from runs/ by tools/site.py
tools/analyze.py                  recomputes the analyses in FINDINGS.md from published rows
SCHEMA.md, DATASHEET.md           field definitions; provenance, terms and intended uses
```

## Comparability rules

A comparison may only line up runs that share the same **protocol id and version** and the same **hardware**.
`comparison.json` lists the runs and names the configuration fields that are meant to differ; `tools/verify.py`
fails if anything else differs. Cross-protocol or cross-hardware tables are allowed only when marked
`"comparable": false` with a stated reason, and are reported as context, never as a ranking.

Every accuracy is reported with a 95% confidence interval. Differences inside the interval are ties.

## Benchmark data

Some benchmarks ask that their questions not be published in plain text (GPQA does). Results here therefore carry
item ids and hashes (`doc_hash`, `prompt_hash`, `target_hash` as logged by lm-evaluation-harness) instead of text.
Anyone with the dataset can regenerate the prompts from the protocol and confirm the hashes match.

## Verify

```bash
python3 tools/verify.py            # all runs and comparisons
python3 tools/verify.py runs/<id>  # one run
```
Python 3.10+, standard library only.

## Licenses

Code: Apache-2.0 (`LICENSE`). Results and documentation: CC BY 4.0 (`LICENSE-DATA.md`). Model, engine and dataset
licenses belong to their owners and are linked from each config and protocol.
