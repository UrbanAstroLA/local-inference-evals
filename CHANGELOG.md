# Changelog

## 2026-10-09 (review): receipts for every figure, qualified determinations

Contents now: 46 runs, 12 comparisons.

- **Decode vs prefill on the tpurtell 0.9.1 release, published:** two runs of one configuration with the same prompts
  and seeds (the noise floor) and one with DFlash2 ×3 on, 12 prompts each, protocol `decode-prefill-consistency/v1`.
  New comparisons `glm53-flash-v091-decode-prefill-repeat` and `glm53-flash-v091-decode-prefill-speculation`. Only the
  large tail-fix effect below 2,044 tokens is claimed; the fix's effect on answers is stated as unmeasured.
- **Server-log figures now have receipts:** runs whose notes quote KV pool size, throughput, acceptance or waiting
  requests publish `server_log.jsonl` (numeric fields parsed from the engine's log), and `tools/verify.py` recomputes a
  `server` block in their summaries. The figures that still rest on unpublished model output (greedy prefix lengths,
  the audited `correct_stated` judgement) are named as such.
- **GPQA: secondary, audited score `correct_stated`** per row (whether the reply's stated final answer is right) and
  accuracy among answered questions, beside the raw `flexible-extract` headline. The scoring note now uses the published
  runs: 0.5 to 2.0 points low per run.
- **Qualifications:** the shared pass-1 seed likely understates run-to-run spread; questions 88 and 79 were chosen
  from seed-1234 data; the component screen's fixed settings, its pairing of ownership with slot sharing, and its
  minimum detectable differences (`tools/analyze.py screen-power`); kernel tests cover the 0.7.0/0.8.0 images and
  local builds, not the 0.9.0/0.9.1 release images; speed figures are single runs; the engine is not bitwise
  reproducible even one request at a time.
- **Open questions added:** 0.7.0 as shipped and empty answers; single-request nondeterminism; no higher-precision
  reference on this hardware.
- Labels: `, prefix cache off` is appended when prefix caching is disabled. Glossary in `README.md` and on the site's
  method page, linked from first use on each site page; the overview leads with its summary.

## 2026-10-09: hard-question screens withdrawn as rate measures; component screen added

Contents now: 43 runs of GLM-5.3-Flash on 2x RTX PRO 6000: 9 GPQA Diamond runs (pass 1 each); 7 component-screen runs
(`hard-prompt-screen/v2`); 10 earlier screens kept as single draws (one kept as INVALID with its cause); 7 serving
probes; 4 kpool kernel-test runs; 2 kpool tail index checks; 2 decode-vs-prefill runs; 2 tool-eval-bench runs. 10
comparisons.

- **Withdrawn: the hard-question screen rates.** Every repeat of the `hard-prompt-screen/v0` and `v1` screens
  (2026-09-30 to 2026-10-08) sent request seed 1234. Concurrent batching still made the repeats vary, but every repeat
  drew on the same sampler noise, so that variation was not statistically meaningful and the effective sample behind
  each rate was far smaller than its request count. Evidence: on `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3`, question 88 failed 3 of 12
  times with distinct seeds and 11 of 12 with seed 1234 on every repeat, all else equal. Withdrawn: the screen failure
  rates and intervals, the layout-bisection statistics and verdicts, and the question-level observations drawn from
  those screens, together with the text built on them in `FINDINGS.md`, the investigation, the comparisons and the site.
  Kept: repeat 1 of each question from each configuration's first screen, as a single draw. Removed runs (in git
  history): the 2026-10-04 `screen-a`, the 2026-10-05 `k3.25-v0.7.0-dflash5_screen-v07pair` and the three
  `screen-bisect2` runs; removed comparisons: `glm53-flash-engine-screen-v0`, `glm53-flash-kpool-screen`,
  `glm53-flash-kpool-screen-v070`, `glm53-flash-v090-layout-screen`, `glm53-flash-v090-tailfix-screen`. Notice:
  `investigations/2026-10-glm53-looping`, section 4. The layout-bisection preregistration is kept unchanged and marked
  withdrawn in a separate `WITHDRAWN.md`.
- **New protocol `hard-prompt-screen/v2`:** one question per run, 12 repeats with distinct request seeds (5000 +
  repeat), 12 concurrent, same early stop and classifier. `tools/clients/hard_prompt_screen.py` gains `SEED_BASE` and
  `CONC`; its v1 defaults are unchanged.
- **New data: the 2026-10-09 component screen** on tpurtell 0.9.1 (preregistered; plan and amendment in
  `investigations/2026-10-glm53-looping/component-screen`): question 88 under three layout switches (arms B, EN, EO,
  NO), question 79 on tpurtell 0.9.1 and on the tpurtell 0.7.0 image at three draft tokens (B79, V79), and a
  fixed-seed control (S1234). Five new configurations; new comparisons `glm53-flash-v091-component-screen-doc88`,
  `glm53-flash-doc79-image-screen`, `glm53-flash-fixed-seed-control`.
- GPQA runs now publish pass 1 only; further passes will use per-pass request seeds (1233 + pass). New: pass 1 of
  `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3` and `4bpw TR3 (Brandon) · tpurtell 0.9.1 · DFlash2 ×3`, with `finish_reason`
  and `completion_tokens` from a passive request log. GPQA rows' `seed` is now the request seed.
- Decode vs prefill: each build was measured once, and a later repeat of the measurement varied by up to about 2x
  between runs; only the large effect below 2,044 tokens (KL 0.066 → 0.010) is claimed.
- `FINDINGS.md` rewritten to what the data supports, each statement graded supported / descriptive / open; the
  investigation reframed around what drives non-completion on hard questions; `tools/analyze.py` subcommands
  `gpqa-table`, `gpqa-pairs` (question-paired), `screen-v2`, `screen-single`, `screen-anatomy`, `screen-speed`;
  `tools/verify.py` enforces pass 1 for GPQA runs, repeat 1 for v0/v1 screens, distinct seeds in v2 runs, and no
  comparisons of v0/v1 runs; site rebuilt.

## Unreleased (first public version)
Contents: 39 runs of GLM-5.3-Flash on 2x RTX PRO 6000:
- 7 GPQA Diamond runs: 5 full 3-pass and 2 single-pass;
- 15 hard-question screens, one kept as INVALID with its cause;
- 7 serving probes;
- 4 kpool kernel-test runs;
- 2 kpool tail index checks;
- 2 decode-vs-prefill consistency runs;
- 2 tool-eval-bench runs.

Also: 11 comparisons, two investigations with their preregistrations, `FINDINGS.md`, `DATASHEET.md`, `SCHEMA.md`, and the
results site.

- Scoring note revised (2026-10-08): `flexible-extract` underscores by 0.5 to 2.5 points per pass, about 1.6, not about
  0.7. Most of it comes from replies that state the answer and then mention other options' labels; the earlier figure
  checked only replies where the two filters disagree. Notes added on `strict-match` (a phrasing check, not accuracy)
  and on answer order (fixed by the first `--seed 1235` load, cached). `tools/verify.py` now checks that compared GPQA
  runs share prompt and target hashes. No scores changed.
- Scoring note (2026-10-08): `flexible-extract` matches any parenthesised capital letter, so notation in a reply can
  be read as the answer. A deterministic misread that only lowers scores, about 0.7 points per pass. Stated in the GPQA
  protocol, DATASHEET, SCHEMA and site; no scores changed. FINDINGS 6 now credits unanswered questions at their own
  observed accuracy (0.8-2.8 points instead of 0.9-3.4).
- tpurtell 0.9.1 (2026-10-08): the DCP1 tail fix was merged (tpurtell/glm-5.3-flash-ext3-2x-rtx#6) and released in
  tpurtell 0.9.1. The tail-fix configurations, measured on a local build before the merge, are now labelled
  `tpurtell 0.9.0 + DCP1 tail fix ≈ 0.9.1` with the basis in `engine.equivalent_to`, as `0.8.0 + kpool fixes ≈ 0.9.0`
  is. Engine series `tpurtell 0.8.0-0.9.0` renamed `tpurtell 0.8.0-0.9.1`. Glossaries, the investigation and the site
  define 0.9.1. No data changed.
- Looping investigation (2026-10-08): `investigations/2026-10-glm53-looping`, and a site page.
  - **Layout bisection** on tpurtell 0.9.0, preregistered with three amendments: six screens across three
    configurations.
    - 0.9.0 as released vs v0.7.0's layout as a diagnostic control: 41 vs 29 failures of 80, p = 0.079 (not yet
      significant at this size).
    - With vs without the DCP1 tail fix (tpurtell/glm-5.3-flash-ext3-2x-rtx#6): 33 vs 41, p = 0.27, lower in both screens
      with the fix, not yet significant at this size.
  - **Tail-fix checks:** index check, decode-vs-prefill consistency (KL 0.066 → 0.010 below 2,044 tokens), serving
    probe, and tool-eval-bench (157/157 → 159/163 of 176).
  - New protocols `decode-prefill-consistency/v1`, `kpool-tail-index/v1` and `tool-eval-bench/v1`, with their clients.
  - New config field `serving.layout`. Its label appears only for non-default layouts.
  - `tools/analyze.py screen-pool`, and `recompute.py` in the investigation.
  - `FINDINGS.md` gains inference 8.
- Labels (2026-10-06): configurations are named `weights · engine version · speculation`, engine name first (for
  example `4bpw TR3 (Brandon) · tpurtell 0.8.0 + kpool fixes ≈ 0.9.0 · DFlash2 ×3`), built from new config fields (`model.label`,
  `engine.name`, `engine.built_on`, `engine.patches[].name`, `engine.equivalent_to`, `engine.series`,
  `serving.speculative.label`) by one rule in
  `tools/verify.py`, which also fails if a label is ambiguous. Defined in `SCHEMA.md`; key at the top of `FINDINGS.md`,
  in `README.md` and on every site page. Replaces "0.8.0 + patches ≈ 0.9.0". Brandon M. Music's TR3 checkpoint is labelled
  `4bpw TR3 (Brandon)`; speculation labels name the method (`DFlash2 ×3`, `no speculation`). Config and run notes use the labels.
- Consistency corrections (2026-10-06), no data changed: the 0.7.0-vs-0.8.0 completion gap is about a third on GPQA
  (6 vs 20 of 594, Fisher p = 0.009) and about half on the screen (12 vs 24 of 40); the earlier text merged the two and
  called overlapping exact intervals non-overlapping. "No completion cost" from the kpool fixes is now "no detectable
  change" with its interval (-3 to +19 more empty answers on 4bpw TR3 (Brandon), tpurtell 0.9.0). The gap to published scores is no longer
  called "not a runtime problem". Greedy divergence is stated as measured (median 318 characters). The serving probe's
  fix effect is given as measured (-7% to +3%, single runs). The published-context comparison covers all five full runs.
  See the commit history for each change.
- `tools/analyze.py` gains `gpqa-passes`, `gpqa-pairs` and `screen-speed`, so every number in `FINDINGS.md` is
  recomputable.
- Correction: the GPQA protocol said passes used seeds 1235-1237. Those are lm-eval's `--seed`; every request
  carried `seed: 1234`. Passes differ through batching nondeterminism. Protocol and schema now say so; no number changes.
- Results site in `docs/` (GitHub Pages), generated from the data by `tools/site.py`: overview, GPQA, screens,
  speed and acceptance, kernel tests, and a reading guide; interval plots with 95% intervals, table views, links to receipts.
- 4bpw TR3 (Brandon) on tpurtell 0.8.0 + kpool fixes ≈ 0.9.0, hard-question screen: 15/40.
- 4bpw TR3 (Brandon) on tpurtell 0.9.0 (official release): full GPQA Diamond run (84.7%, 23 of 594 empty), replacing the earlier
  partial exports. New comparison `glm53-flash-k4-v080-v090-gpqa`: no detectable difference from tpurtell 0.8.0 in
  accuracy (p = 0.90) or empty answers (23 vs 15, p = 0.25).
- Site labels runs with fewer than 3 passes by their pass count; accuracy ranges on the site are computed from the data.
- Client fix before release: a stream that ends without a finish reason is an `error`, not an `exhaust`. Affected only
  the INVALID run (engine crash); all other screens were audited and have finish reasons on every row.
