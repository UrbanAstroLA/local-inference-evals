#!/bin/bash
# capture-p0 analysis in a CPU container (docker run --rm --network none, no --gpus, CUDA_VISIBLE_DEVICES=, host
# uid): analysis/analyze_cap0.py AO PO.  The code is mounted read-only; AO receives cap0-result.{json,md}; PO (0700)
# receives cap0-detail.json.  Uses the pinned stage-0 image (numpy) by id.  No `set -e`.
#   analysis/run_cap0_analysis.sh <AO: runs/session-attempt-N> <PO: ../PRIVATE-capture-p0/session-attempt-N/p4>
set -u
[ "$#" = 2 ] || { echo "usage: run_cap0_analysis.sh AO PO"; exit 2; }
PKG=$(cd "$(dirname "$0")/.." && pwd)
AO=$(cd "$1" && pwd) || exit 2
PO=$(cd "$2" && pwd) || exit 2
IMG=sha256:7408f893dfe35f1fbd330153e34ea81e83460fe5e2fff200ab87ea992565d3c7
docker run --rm --pull=never --network none -e CUDA_VISIBLE_DEVICES= --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PKG":/c:ro -v "$AO":/ao -v "$PO":/po --entrypoint python3 "$IMG" /c/analysis/analyze_cap0.py /ao /po
