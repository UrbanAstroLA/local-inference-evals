#!/bin/bash
# CPU-only runner for follow-up-a2 scripts: B1 image as a CPU container, no network, no GPU; run data read-only.
HV=<runs>/hook-validation
IMG=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a   # local/glm53-v0.9.1-b1-d7:local
S=$1; shift
docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 \
  -v $HV/run-a2:/run:ro -v $HV/run-a1:/run1:ro -v $HV/followup-a2:/f -v $HV/analysis:/a:ro \
  -v $HV/diagnosis-a1:/d1:ro -v $HV/src:/src:ro -v $HV/hook:/hook:ro \
  -w /f --entrypoint python3 $IMG /f/$S "$@" 2>&1 | grep -v cpp_extension
