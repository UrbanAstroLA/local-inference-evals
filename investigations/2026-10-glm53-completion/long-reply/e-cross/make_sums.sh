#!/usr/bin/env bash
set -u
# Writes SHA256SUMS over every file of the package except run/ (created by the runner), SHA256SUMS itself and bytecode.
# [E copy of bplus/make_sums.sh: set -u; package-path guard; the transient test scratch folders PRIVATE/scratch-*/ and
#  FROZEN-VERIFY.txt (written after the freeze) excluded.]
cd "$(dirname "$0")" || exit 1
case "$(pwd)" in <runs>/long-reply/e-cross|<runs>/long-reply/e-cross/*) ;; *) echo "ABORT: not inside e-cross"; exit 1 ;; esac
find . -type f ! -path './run/*' ! -path './PRIVATE/scratch-*' ! -name SHA256SUMS ! -name FROZEN-VERIFY.txt ! -path '*/__pycache__/*' ! -name '*.pyc' | LC_ALL=C sort | xargs -d '\n' sha256sum > SHA256SUMS
echo "SHA256SUMS: $(wc -l < SHA256SUMS) files; sha256 $(sha256sum SHA256SUMS | cut -c1-64)"
