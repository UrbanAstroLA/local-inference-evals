#!/bin/bash
# CPU-only runner for diagnosis scripts: B1 image as a CPU container, no network, no GPU; run data read-only.
HV=<runs>/hook-validation
IMG=sha256:1b9f8b15d63197ad8b2ef2410321baedf9dcd55a112603bc77b887e8dac293d5   # forcehook-mg validation image (hook installed), CPU only
S=$1; shift
docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e TRITON_INTERPRET=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v $HV/run-a1:/run:ro -v $HV/diagnosis-a1:/d -v $HV/hook:/hook:ro -v $HV/src:/src:ro -v $HV/analysis:/a:ro \
  -v $HV/tests:/tests:ro -v <runs>/d7/step2/guard:/guard:ro \
  -w /d --entrypoint python3 $IMG /d/$S "$@" 2>&1 | grep -v cpp_extension
