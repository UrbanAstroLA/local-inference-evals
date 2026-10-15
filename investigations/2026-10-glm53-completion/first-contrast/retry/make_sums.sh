#!/bin/bash
# SHA256SUMS of the retry package (retry/). Covers every design, code, replay, calibration, cost and evidence file.
# Excluded: the item ledger costs.jsonl, session outputs run-fcr*/, DRYRUN.out, bytecode, the synthetic runs' raw data
# (only each variant's gate1r.json, fcr_result.json and logs are hashed), and SHA256SUMS itself (its own sha256 is in
# DRYRUN.out). Attempt-1 files are not copied or re-hashed here: they are pinned by ../SHA256SUMS, which the runner
# checks too (they are imported read-only).
cd "$(dirname "$0")" || exit 1
find . -type f \
  ! -path './costs.jsonl' ! -path './run-fcr*' ! -name 'DRYRUN.out' ! -name 'SHA256SUMS' ! -path '*/__pycache__/*' \
  ! -name '*.pyc' \
  \( ! -path './tests/out/synth-*/full/*' -o -name 'gate1r.json' -o -name 'fcr_result.json' \) \
  ! -path './tests/out/synth-*/full/gate1r/client.jsonl' \
  | grep -v -E '^\./tests/out/synth-[^/]+/full/(forcewin|mapguard|gate1r/forcewin|gate1r/mapguard)/' \
  | sort | xargs sha256sum > SHA256SUMS
echo "$(wc -l < SHA256SUMS) files; SHA256SUMS sha256 $(sha256sum SHA256SUMS | cut -d' ' -f1)"
