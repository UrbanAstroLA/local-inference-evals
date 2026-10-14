#!/bin/bash
# SHA256SUMS of every file that runs or was produced here (excluding caches, run outputs and the cost ledger).
# Attempt 2 (PA2-14; attempt-1 script kept as make_sums.sh.a1, attempt-1 sums as SHA256SUMS.a1): also excludes the
# regenerated analysis dry runs in tests/out-a2/analysis-dryrun and DRYRUN-a2.out (written by the final DRY_RUN of
# the frozen package, after this file). The *.a1 copies, diagnosis-a1/ and prep-a2/ are covered.
# PA2-25 (amendment 21g; pre-21g script kept as make_sums.sh.pre21g, pre-21g sums as SHA256SUMS.pre21g): also excludes
# the regenerated synthetic run trees in tests/out-21g/analysis-dryrun (their logs and verdicts are reproduced by
# tests/run_21g.sh). The *.pre21g copies, prep-a2/out-21g, prep-a2/faultinj-a1 and prep-a2/replay-a1-21g are covered.
cd "$(dirname "$0")" || exit 1
find . -type f ! -path '*/__pycache__/*' ! -name '*.pyc' ! -name SHA256SUMS ! -name costs.jsonl \
  ! -path './run-a*' ! -path './tests/out/analysis-dryrun/*' ! -path './tests/out-a2/analysis-dryrun/*' \
  ! -path './tests/out-21g/analysis-dryrun/*' \
  ! -name DRYRUN-a2.out | sort | xargs sha256sum > SHA256SUMS
wc -l SHA256SUMS
