# Changelog

## Unreleased (first public version)
- GLM-5.3-Flash on 2x RTX PRO 6000: four 3-pass GPQA Diamond runs plus three partial runs (two 1-pass, one in progress), eight hard-question screens (one
  kept as INVALID with its cause), five serving probes, four kpool kernel-test runs, six comparisons, the kpool
  investigation with its preregistrations, `FINDINGS.md`, `DATASHEET.md`, `SCHEMA.md`.
- Results site in `docs/` (GitHub Pages), generated from the data by `tools/site.py`: overview, GPQA, screens,
  speed and acceptance, kernel tests, and a reading guide; interval plots with 95% intervals, table views, links to receipts.
- 4bpw on v0.8.0 + patches (= v0.9.0 defaults) hard-question screen: 15/40.
- 4bpw on the official v0.9.0 release: GPQA Diamond passes 1-2 (84.3%, 17 of 396 empty), replacing the pass-1-only run; pass 3 to follow. Paired comparison with unpatched 4bpw 0.8.0 in `FINDINGS.md`.
- Site labels runs with fewer than 3 passes by their pass count; accuracy ranges on the site are computed from the data.
- Engine labels: patched 0.8.0 rows read "0.8.0 + patches ≈ 0.9.0", with a note on every page (same kpool kernel files as the 0.9.0 release).
- Client fix before release: a stream that ends without a finish reason is an `error`, not an `exhaust`. Affected only
  the INVALID run (engine crash); all other screens were audited and have finish reasons on every row.
