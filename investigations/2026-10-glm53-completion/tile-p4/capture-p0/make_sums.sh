#!/bin/bash
# capture-p0/SHA256SUMS: the frozen package (code, docs, build record, cost estimate).  The session runner refuses to
# start a GPU step when any listed file differs (sha256sum -c).  Excludes test outputs, run outputs and the ledger.
cd "$(dirname "$0")" || exit 2
find . -type f \( -path './tests/out/*' -o -path './runs/*' -o -name 'costs-cap0.jsonl' -o -name 'SHA256SUMS' \
     -o -path '*/__pycache__/*' -o -name '*.pyc' \) -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum > SHA256SUMS
echo "SHA256SUMS: $(wc -l < SHA256SUMS) files"
