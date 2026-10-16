#!/bin/bash
# [R-10] Sentinel boundary-state tests (test_sentinel_fcr.py) in a CPU container of the B1 image: --network none, no GPU,
# host uid; the package and attempt-2 run-a2 snapshots read-only. -> tests/out/sentinel-tests.log, sentinel_tests.json
RETRY=$(cd "$(dirname "$0")/.." && pwd)
FC=$(cd "$RETRY/.." && pwd)
HV=<runs>/hook-validation/run-a2/forcewin
IMG=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a
docker run --rm --network none --name "fcr-sentinel-$$" --user "$(id -u):$(id -g)" -e HOME=/tmp -e CUDA_VISIBLE_DEVICES= \
  -e PYTHONDONTWRITEBYTECODE=1 -v "$FC":/fc:ro -v "$HV":/hv:ro -v "$RETRY/tests/out":/out --entrypoint python3 "$IMG" \
  /fc/retry/tests/test_sentinel_fcr.py /fc /hv /out 2>&1 | grep -v cpp_extension
exit "${PIPESTATUS[0]}"
