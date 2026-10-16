#!/usr/bin/env bash
# Writes SHA256SUMS over every file in this package (relative paths, sorted), excluding SHA256SUMS itself. Not set -e.
cd "$(dirname "$0")" || exit 1
find . -type f ! -name SHA256SUMS ! -path '*/__pycache__/*' -printf '%P\n' | LC_ALL=C sort | xargs -d '\n' sha256sum > SHA256SUMS
echo "SHA256SUMS: $(wc -l < SHA256SUMS) files; sha256 $(sha256sum SHA256SUMS | cut -c1-64)"
