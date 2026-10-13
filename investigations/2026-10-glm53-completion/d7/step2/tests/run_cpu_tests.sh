#!/bin/bash
# All step-2 CPU tests (PLAN.md amendment 18a, 2a/2b).  NO GPU: no --gpus, CUDA_VISIBLE_DEVICES empty,
# --network none, TRITON_INTERPRET=1; no server; no glm53* container.  Outputs in tests/out/ and a summary.
# No `set -e`: every exit code is recorded in tests/out/summary.txt.
#   (1) B1 serving image: the D7 fix tests (d7/mechanism/tests/test_d7_fix.py, read-only) on the image that
#       will serve, i.e. B1 = B0 + d7-fix.diff as built (not the CPU-test image)
#   (2) B1 guard image: image smoke (inert unset / device), the frozen guard suite (28 tests, unmodified
#       copies), a2 regression, kpool-unaligned check, the B1-amendment tests (test_d7_guard.py), the frozen
#       pipeline dry run and the B1 pipeline dry run, each followed by the amended analysis; the analysis'
#       phase-2d / window-cut tests (pure Python)
HERE=$(cd "$(dirname "$0")/.." && pwd)
IMG=${IMG:-local/glm53-v0.9.1-b1-mapguard:local}
B1=local/glm53-v0.9.1-b1-d7:local
MECH=<runs>/d7/mechanism
O=$HERE/tests/out; mkdir -p "$O"; SUM=$O/summary.txt; : > "$SUM"
for i in "$IMG" "$B1"; do echo "image $i $(docker image inspect "$i" --format '{{.Id}}')" | tee -a "$SUM"; done
docker image inspect "$IMG" --format '{{.Id}}' > "$O/image-id.txt"
filt() { grep -v "^WARNING\|cpp_extension\|No CUDA runtime"; }
run() {  # $1 = label, $2 = GLM53_MAPGUARD value, rest = command
  local label=$1 m=$2; shift 2
  docker run --rm --name "mgtest-$label" --network none --user "$(id -u):$(id -g)" -e CUDA_VISIBLE_DEVICES= \
    -e TRITON_INTERPRET=1 -e HOME=/tmp -e GLM53_MAPGUARD="$m" -e GLM53_MAPGUARD_DIR=/tmp/mgsmoke \
    -e VLLM_GLM53_MLA_OWNERS=split:25 -v "$HERE":/w -w /w --entrypoint python3 "$IMG" "$@" 2>&1 | filt > "$O/$label.log"
  local rc=${PIPESTATUS[0]}
  printf '%-22s rc=%-3s %s\n' "$label" "$rc" "$(grep -v '^\s*$' "$O/$label.log" | tail -n 1 | cut -c1-140)" | tee -a "$SUM"
}
# (1) D7 fix tests on the B1 serving image
docker run --rm --name mgtest-d7fix --network none --user "$(id -u):$(id -g)" -e CUDA_VISIBLE_DEVICES= \
  -e TRITON_INTERPRET=1 -e HOME=/tmp -e D7_EXPECT=b1 -v "$MECH/tests":/t:ro -v "$MECH/patch/b0/sparse_attn_indexer_kpool.py":/p/b0_kpool.py:ro \
  -v "$O":/o -w /t --entrypoint python3 "$B1" test_d7_fix.py /o/d7fix-b1-serving-image.json 2>&1 | filt > "$O/d7fix-b1-serving-image.log"
printf '%-22s rc=%-3s %s\n' d7fix_b1_image "${PIPESTATUS[0]}" "$(tail -n 1 "$O/d7fix-b1-serving-image.log" | cut -c1-140)" | tee -a "$SUM"
# (2) guard image
run smoke_unset "" tests/test_image_smoke.py tests/out/image-smoke-unset.json
run smoke_device device tests/test_image_smoke.py tests/out/image-smoke-device.json
run kpool_unaligned "" tests/test_kpool_unaligned.py tests/out/kpool-unaligned.json
run test_mapguard "" tests/test_mapguard.py tests/out/cpu-results.json
run a2_regression "" tests/test_a2_regression.py tests/out/a2-regression.json
run test_d7_guard "" tests/test_d7_guard.py tests/out/d7-guard.json
for v in dryrun dryrun-d7; do
  rm -rf "$O/$v"; mkdir -p "$O/$v"
  s=tests/test_pipeline_dryrun.py; [ $v = dryrun-d7 ] && s=tests/test_pipeline_dryrun_d7.py
  run "$v-r0" "" "$s" "tests/out/$v" 0
  run "$v-r1" "" "$s" "tests/out/$v" 1
  python3 "$HERE/analysis/analyze_mapguard.py" "$O/$v" > "$O/$v-verdict.log" 2>&1
  printf '%-22s rc=%-3s %s\n' "$v-analysis" "$?" "$(head -n 1 "$O/$v-verdict.log")" | tee -a "$SUM"
done
python3 "$HERE/tests/test_analysis_2d.py" "$O/analysis-2d.json" > "$O/analysis-2d.log" 2>&1
printf '%-22s rc=%-3s %s\n' analysis_2d "$?" "$(tail -n 1 "$O/analysis-2d.log" | cut -c1-140)" | tee -a "$SUM"
echo "done $(date -u +%FT%TZ)" | tee -a "$SUM"
