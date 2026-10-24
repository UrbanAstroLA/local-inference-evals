#!/bin/bash
# replay-topk/SHA256SUMS: the frozen package (code, docs, tests and their results, cost estimate).  The runner refuses
# to start a GPU step when any listed file differs (sha256sum -c).  Excludes test outputs, run outputs, the compile
# cache (regenerable, withheld from publication) and the ledger.
cd "$(dirname "$0")" || exit 2
find . -type f \( -path './tests/out/*' -o -path './runs/*' -o -path './cache/*' -o -name 'costs-replay.jsonl' \
     -o -name 'SHA256SUMS' -o -path '*/__pycache__/*' -o -name '*.pyc' \) -prune -o -type f -print | LC_ALL=C sort \
  | xargs sha256sum > SHA256SUMS
echo "SHA256SUMS: $(wc -l < SHA256SUMS) files"
