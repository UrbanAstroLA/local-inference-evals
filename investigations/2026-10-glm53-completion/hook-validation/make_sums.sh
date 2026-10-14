#!/bin/bash
# SHA256SUMS of every file that runs or was produced here (excluding caches, run outputs and the cost ledger).
cd "$(dirname "$0")" || exit 1
find . -type f ! -path '*/__pycache__/*' ! -name '*.pyc' ! -name SHA256SUMS ! -name costs.jsonl \
  ! -path './run-a*' ! -path './tests/out/analysis-dryrun/*' | sort | xargs sha256sum > SHA256SUMS
wc -l SHA256SUMS
