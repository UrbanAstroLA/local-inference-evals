# Changelog

## Unreleased (first public version)
Contents: 25 runs of GLM-5.3-Flash on 2x RTX PRO 6000 (7 GPQA Diamond runs: 5 full 3-pass and 2 single-pass;
9 hard-question screens, one kept as INVALID with its cause; 5 serving probes; 4 kpool kernel-test runs), 7 comparisons,
the kpool investigation with its preregistrations, `FINDINGS.md`, `DATASHEET.md`, `SCHEMA.md`, and the results site.

- Labels (2026-10-06): configurations are named `weights · engine version · speculation`, engine name first (for
  example `4bpw · tpurtell 0.8.0 + kpool fixes ≈ 0.9.0 · 3 drafts`), built from new config fields (`model.label`,
  `engine.name`, `engine.built_on`, `engine.patches[].name`, `engine.equivalent_to`, `engine.series`) by one rule in
  `tools/verify.py`, which also fails if a label is ambiguous. Defined in `SCHEMA.md`; key at the top of `FINDINGS.md`,
  in `README.md` and on every site page. Replaces "0.8.0 + patches ≈ 0.9.0". Config and run notes use the labels.
- Consistency corrections (2026-10-06), no data changed: the 0.7.0-vs-0.8.0 completion gap is about a third on GPQA
  (6 vs 20 of 594, Fisher p = 0.009) and about half on the screen (12 vs 24 of 40); the earlier text merged the two and
  called overlapping exact intervals non-overlapping. "No completion cost" from the kpool fixes is now "no detectable
  change" with its interval (-3 to +19 more empty answers on 4bpw 0.9.0). The gap to published scores is no longer
  called "not a runtime problem". Greedy divergence is stated as measured (median 318 characters). The serving probe's
  fix effect is given as measured (-7% to +3%, single runs). The published-context comparison covers all five full runs.
  See the commit history for each change.
- `tools/analyze.py` gains `gpqa-passes`, `gpqa-pairs` and `screen-speed`, so every number in `FINDINGS.md` is
  recomputable.
- Correction: the GPQA protocol said passes used seeds 1235-1237. Those are lm-eval's `--seed`; every request
  carried `seed: 1234`. Passes differ through batching nondeterminism. Protocol and schema now say so; no number changes.
- Results site in `docs/` (GitHub Pages), generated from the data by `tools/site.py`: overview, GPQA, screens,
  speed and acceptance, kernel tests, and a reading guide; interval plots with 95% intervals, table views, links to receipts.
- 4bpw on tpurtell 0.8.0 + kpool fixes ≈ 0.9.0, hard-question screen: 15/40.
- 4bpw on tpurtell 0.9.0 (official release): full GPQA Diamond run (84.7%, 23 of 594 empty), replacing the earlier
  partial exports. New comparison `glm53-flash-k4-v080-v090-gpqa`: no detectable difference from 4bpw tpurtell 0.8.0 in
  accuracy (p = 0.90) or empty answers (23 vs 15, p = 0.25).
- Site labels runs with fewer than 3 passes by their pass count; accuracy ranges on the site are computed from the data.
- Client fix before release: a stream that ends without a finish reason is an `error`, not an `exhaust`. Affected only
  the INVALID run (engine crash); all other screens were audited and have finish reasons on every row.
