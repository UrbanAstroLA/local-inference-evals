#!/bin/bash
# P4 analysis, CPU only: one `docker run --rm --network none` CPU container of the local stock image (torch reads the
# forcewin records).  Reads runs/ and PRIVATE-runs/ (read-only), writes analysis/out/ (public aggregates only).
#   analysis/run_analysis.sh [--a RUNDIR --b RUNDIR --a2 RUNDIR]     (default: latest gates-ok attempt per session)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
STOCK=sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
mkdir -p "$ROOT/analysis/out"
ARGS=(); for a in "$@"; do ARGS+=("${a//$ROOT//p4}"); done
docker run --rm --pull=never --network none -e CUDA_VISIBLE_DEVICES= -e HOME=/tmp --user "$(id -u):$(id -g)" \
  -v "$ROOT":/p4:ro -v "$ROOT/analysis/out":/out --entrypoint python3 "$STOCK" \
  /p4/analysis/analyze_p4.py /p4 /out "${ARGS[@]}"
