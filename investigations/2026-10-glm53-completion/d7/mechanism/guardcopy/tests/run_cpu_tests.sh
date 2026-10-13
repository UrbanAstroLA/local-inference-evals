#!/bin/bash
# All CPU tests inside the derived image (no GPU, no network, no server).  Outputs in tests/out/.
HERE=$(cd "$(dirname "$0")/.." && pwd)
IMG=${IMG:-local/glm53-v0.9.1-b0-mapguard:local}
mkdir -p "$HERE/tests/out"
run() {  # $1 = GLM53_MAPGUARD value, rest = command
  local m=$1; shift
  docker run --rm --name mgtest --network none --user "$(id -u):$(id -g)" -e CUDA_VISIBLE_DEVICES= -e TRITON_INTERPRET=1 \
    -e HOME=/tmp -e GLM53_MAPGUARD="$m" -e GLM53_MAPGUARD_DIR=/tmp/mgsmoke -e VLLM_GLM53_MLA_OWNERS=split:25 \
    -v "$HERE":/w -w /w --entrypoint python3 "$IMG" "$@"
}
docker image inspect "$IMG" --format '{{.Id}}' > "$HERE/tests/out/image-id.txt"
run "" tests/test_image_smoke.py tests/out/image-smoke-unset.json > tests/out/image-smoke-unset.log 2>&1; echo "smoke unset rc=$?"
run device tests/test_image_smoke.py tests/out/image-smoke-device.json > tests/out/image-smoke-device.log 2>&1; echo "smoke device rc=$?"
run "" tests/test_kpool_unaligned.py tests/out/kpool-unaligned.json > tests/out/kpool-unaligned.log 2>&1; echo "kpool unaligned rc=$?"
run "" tests/test_mapguard.py tests/out/cpu-results.json > tests/out/test_mapguard.log 2>&1; echo "mapguard tests rc=$?"
run "" tests/test_a2_regression.py tests/out/a2-regression.json > tests/out/a2-regression.log 2>&1; echo "a2 regression rc=$?"
rm -rf tests/out/dryrun; mkdir -p tests/out/dryrun
run "" tests/test_pipeline_dryrun.py tests/out/dryrun 0 > tests/out/dryrun-r0.log 2>&1; echo "dryrun r0 rc=$?"
run "" tests/test_pipeline_dryrun.py tests/out/dryrun 1 > tests/out/dryrun-r1.log 2>&1; echo "dryrun r1 rc=$?"
python3 analysis/analyze_mapguard.py tests/out/dryrun > tests/out/dryrun-verdict.log 2>&1; echo "dryrun analysis rc=$?"
