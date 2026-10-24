#!/bin/bash
# SHA256SUMS of the frozen package (everything except run outputs, the ledgers, test outputs, caches and PRIVATE-*).
# 2026-10-24: also excludes the stage-0 ledger and run outputs (costs-stage0.jsonl, runs-stage0/).
# The session runner checks this list before every GPU session.
cd "$(dirname "$0")" || exit 2
find . -type f \
  ! -path './runs/*' ! -path './runs-stage0/*' ! -path './PRIVATE-*' ! -path './tests/out/*' ! -path './analysis/out/*' \
  ! -path '*/__pycache__/*' ! -name 'costs.jsonl' ! -name 'costs-stage0.jsonl' ! -name 'SHA256SUMS' ! -name '*.pyc' | sort | xargs sha256sum > SHA256SUMS
echo "$(wc -l < SHA256SUMS) files hashed"
