#!/bin/bash
# P4 fixture build, CPU only: one `docker run --rm --network none` CPU container of the LOCAL stock 0.9.1 image
# (CUDA_VISIBLE_DEVICES empty, no --gpus) runs select_windows.py and extract_sched.py.  Reads long-reply/ d091 + d091b
# raw bodies and hook-validation/run-a2/forcewin (read-only mounts); writes windows/out/ (public) and
# PRIVATE-fixtures/ (0700 dir, 0600 files).  Never overwrites (the scripts use O_EXCL).  No `set -e`.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
INV=<runs>
IMG=sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
REPO=<models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1
mkdir -p "$HERE/out" || exit 2
( umask 077; mkdir -p "$ROOT/PRIVATE-fixtures" ) && chmod 700 "$ROOT/PRIVATE-fixtures" || exit 2
MODE=${1:-build}   # build | dry
ARGS=""; [ "$MODE" = dry ] && ARGS="--dry-sources"
docker run --rm --pull=never --network none -e CUDA_VISIBLE_DEVICES= -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --user "$(id -u):$(id -g)" \
  -v "$HERE":/w:ro -v "$INV/long-reply":/lr:ro -v "$REPO":/repo:ro -v "$INV/hook-validation/run-a2/forcewin":/hv:ro \
  -v "$HERE/out":/out -v "$ROOT/PRIVATE-fixtures":/priv \
  --entrypoint sh "$IMG" -c "python3 /w/select_windows.py $ARGS; echo select rc=\$?; \
     if [ '$MODE' = build ]; then python3 /w/extract_sched.py; echo sched rc=\$?; fi"
chmod 600 "$ROOT"/PRIVATE-fixtures/* 2>/dev/null
ls -la "$ROOT/PRIVATE-fixtures"
