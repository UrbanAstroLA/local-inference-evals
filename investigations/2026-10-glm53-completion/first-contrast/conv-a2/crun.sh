#!/bin/bash
# CPU-only runner for conv-a2 scripts: B1 image as a CPU container, no network, no GPU; attempt-2 run data read-only.
HV=<runs>/hook-validation
FC=<runs>/first-contrast
IMG=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a   # local/glm53-v0.9.1-b1-d7:local
S=$1; shift
docker run --rm --network none --user $(id -u):$(id -g) -e HOME=/tmp -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 \
  -v $HV/run-a2:/run:ro -v $FC/conv-a2:/c -v $FC/analysis:/a:ro \
  -w /c --entrypoint python3 $IMG "$S" "$@" 2>&1 | grep -v cpp_extension
