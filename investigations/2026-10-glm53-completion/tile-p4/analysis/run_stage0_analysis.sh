#!/bin/bash
# Stage-0 gates, CPU only: one `docker run --rm --network none` CPU container of the local stock image (torch reads
# the forcewin records).  Reads the attempt's public logs and its PRIVATE twin read-only; writes stage0-result.json /
# .md (public aggregates) into the attempt directory.  Called by runner/stage0/run_stage0.sh after the GPUs are released.
#   analysis/run_stage0_analysis.sh <ROOT>/runs-stage0/stage0-attempt-N        (ROOT_OVERRIDE for mock roots)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BROOT=${ROOT_OVERRIDE:-$ROOT}
STOCK=sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
AO=$(cd "$1" 2>/dev/null && pwd) || { echo "usage: run_stage0_analysis.sh ATTEMPT_DIR"; exit 2; }
case "$AO" in "$BROOT"/runs-stage0/stage0-attempt-*) ;; *) echo "REFUSED: $AO is not a stage-0 attempt under $BROOT"; exit 2;; esac
PO=$BROOT/PRIVATE-runs-stage0/$(basename "$AO")
[ -d "$PO" ] || { echo "private twin $PO missing"; exit 3; }
docker run --rm --pull=never --network none -e CUDA_VISIBLE_DEVICES= -e HOME=/tmp --user "$(id -u):$(id -g)" \
  -v "$ROOT/analysis":/an:ro -v "$AO":/ao -v "$PO":/po:ro --entrypoint python3 "$STOCK" \
  /an/analyze_stage0.py /ao /po
