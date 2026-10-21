#!/usr/bin/env bash
# Writes SHA256SUMS over every file of the package except run/ (created by the runner), SHA256SUMS itself and bytecode.
cd "$(dirname "$0")" || exit 1
find . -type f ! -path './run/*' ! -name SHA256SUMS ! -name FROZEN-VERIFY.txt ! -path '*/__pycache__/*' ! -name '*.pyc' | LC_ALL=C sort | xargs -d '\n' sha256sum > SHA256SUMS
echo "SHA256SUMS: $(wc -l < SHA256SUMS) files; sha256 $(sha256sum SHA256SUMS | cut -c1-64)"
