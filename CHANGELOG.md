# Changelog

## Unreleased (first public version)
- GLM-5.3-Flash on 2x RTX PRO 6000: four 3-pass and two 1-pass GPQA Diamond runs, eight hard-question screens (one
  kept as INVALID with its cause), five serving probes, four kpool kernel-test runs, six comparisons, the kpool
  investigation with its preregistrations, `FINDINGS.md`, `DATASHEET.md`, `SCHEMA.md`.
- Client fix before release: a stream that ends without a finish reason is an `error`, not an `exhaust`. Affected only
  the INVALID run (engine crash); all other screens were audited and have finish reasons on every row.
