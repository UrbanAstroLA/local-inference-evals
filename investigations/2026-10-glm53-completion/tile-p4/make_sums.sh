#!/bin/bash
# SHA256SUMS of the frozen package (everything except run outputs, the ledger, test outputs, caches and PRIVATE-*).
# The session runner checks this list before every GPU session.
cd "$(dirname "$0")" || exit 2
find . -type f \
  ! -path './runs/*' ! -path './PRIVATE-*' ! -path './tests/out/*' ! -path './analysis/out/*' ! -path '*/__pycache__/*' \
  ! -name 'costs.jsonl' ! -name 'SHA256SUMS' ! -name '*.pyc' | sort | xargs sha256sum > SHA256SUMS
echo "$(wc -l < SHA256SUMS) files hashed"
