#!/usr/bin/env bash
set -u
# Writes SHA256SUMS over every file of this package (relative paths, sorted), excluding SHA256SUMS itself, __pycache__,
# PRIVATE/ (local-only, never hashed into a shareable list), out/ and run/ (run-time outputs) and tests/sim-work/ (scratch).
# Not set -e.
cd "$(dirname "$0")" || exit 1
case "$PWD" in <runs>/long-reply/d091|<runs>/long-reply/d091/tests/sim-work/pkg) ;; *) echo "ABORT: unexpected folder $PWD"; exit 1 ;; esac
find . -type f ! -name SHA256SUMS ! -path '*/__pycache__/*' ! -path './PRIVATE/*' ! -path './out/*' ! -path './run/*' ! -path './tests/sim-work/*' -printf '%P\n' \
  | LC_ALL=C sort | xargs -d '\n' sha256sum > SHA256SUMS
echo "SHA256SUMS: $(wc -l < SHA256SUMS) files; sha256 $(sha256sum SHA256SUMS | cut -c1-64)"
