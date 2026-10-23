#!/bin/bash
# Regression: the frozen B1 guard suite (d7/step2, read-only) against the 22ao guard module, in the B1 guard image.
# NO GPU, --network none, TRITON_INTERPRET=1. Outputs only in livecheck-22ao/tests/out-frozen-suite/.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
D7=<runs>/d7/step2
O=$HERE/tests/out-frozen-suite; mkdir -p "$O"; : > "$O/summary.txt"
IMG=local/glm53-v0.9.1-b1-mapguard:local
G=$HERE/guard/mapguard_hooks.py
VP=/usr/local/lib/python3.12/dist-packages/vllm
echo "image $(docker image inspect $IMG --format '{{.Id}}') guard $(sha256sum "$G" | cut -c1-16)" | tee -a "$O/summary.txt"
run() { local label=$1; shift
  docker run --rm --network none --user "$(id -u):$(id -g)" -e CUDA_VISIBLE_DEVICES= -e TRITON_INTERPRET=1 -e HOME=/tmp \
    -e VLLM_GLM53_MLA_OWNERS=split:25 -v "$D7":/w:ro -v "$O":/o -v "$G":$VP/mapguard_hooks.py:ro -w /w \
    --entrypoint python3 "$IMG" "$@" 2>&1 | grep -v "^WARNING\|cpp_extension\|No CUDA runtime" > "$O/$label.log"
  local rc=${PIPESTATUS[0]}
  printf '%-16s rc=%-3s %s\n' "$label" "$rc" "$(grep -v '^\s*$' "$O/$label.log" | tail -n 1 | cut -c1-140)" | tee -a "$O/summary.txt"
}
run test_mapguard tests/test_mapguard.py /o/cpu-results.json
run test_d7_guard tests/test_d7_guard.py /o/d7-guard.json
run a2_regression tests/test_a2_regression.py /o/a2-regression.json
