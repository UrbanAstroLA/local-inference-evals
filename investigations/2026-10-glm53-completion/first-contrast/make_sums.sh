#!/bin/bash
# Freeze the first-contrast package: SHA256SUMS over every design, code, fixture, cost and evidence file.
# Excluded (mutable or bulky): costs.jsonl (the item ledger), run-fc*/ (session outputs), DRYRUN.out, __pycache__,
# the synthetic runs' raw data (tests/out/synth-*/full/* except their gate1.json and fc_result.json).
cd "$(dirname "$0")" || exit 1
{
  ls PROPOSAL.md PROPOSAL.md.pre22 FREEZE.md SHA256SUMS.pre22 make_sums.sh 2>/dev/null
  find analysis runner fixtures cost prior-null conv-a2 tests -type f \
    \( -path '*/__pycache__/*' -o -path 'tests/out/synth-*/full/*' -o -path 'tests/out/plan-dryrun/client.jsonl' \) -prune -o -type f -print
  find tests/out -path 'tests/out/synth-*/full/gate1/gate1.json' -o -path 'tests/out/synth-*/full/fc_result.json'
} | grep -v '__pycache__' | sed 's#^#./#' | sort -u | xargs sha256sum > SHA256SUMS.new || exit 2
mv SHA256SUMS.new SHA256SUMS
echo "SHA256SUMS: $(wc -l < SHA256SUMS) files, sha256 $(sha256sum SHA256SUMS | cut -c1-16)"
