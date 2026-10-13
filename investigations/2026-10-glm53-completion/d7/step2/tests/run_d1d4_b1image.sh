#!/bin/bash
# Step 2b: the D1-D4 (and D5/D6 still-reproduced) CPU regressions of d7/mechanism/tests/run_regressions.sh part (1),
# re-run on the B1 SERVING image (local/glm53-v0.9.1-b1-d7:local) instead of the CPU-test image. Copy of the
# frozen script's part (1) only; outputs in d7/step2/tests/out/regress-b1image/. CPU only (no --gpus,
# CUDA_VISIBLE_DEVICES empty, --network none, TRITON_INTERPRET=1 where the frozen script sets it).
#  (1) bundle/freeze/cpu_tests.sh, the d1234 part, re-run with the D7 CPU-test image in place of B0
#      (same defect-repros scripts, installed files, nothing mounted over engine files);
#  (2) the bundle check-image tests (c511check, tailguard, verify13, d4limit, test_fix U) in the d1234-check
#      image with its observer file + d7-fix.diff mounted over the installed copy;
#  (3) the b0-mapguard CPU suite (copied to guardcopy/; frozen originals untouched) in the guard image with
#      the guard's file + d7-fix.diff mounted over the installed copy.
# No `set -e`; every exit code is recorded in out/regress/summary.txt.
M=$(cd "$(dirname "$0")/.." && pwd)   # d7/step2
R=<runs>
DR=$R/defect-repros
OUT=$M/tests/out/regress-b1image; mkdir -p "$OUT"; SUM=$OUT/summary${SUMSUFFIX}.txt; : > "$SUM"
B1=local/glm53-v0.9.1-b1-d7:local
REL=ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
CHK=local/glm53-v0.9.1-bundle-d1234-check:local
GRD=local/glm53-v0.9.1-b0-mapguard:local
CKPT=<models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1/snapshots/0490d2f708b12145f6516555ab066aaeb401cd21
VP=/usr/local/lib/python3.12/dist-packages/vllm
KP=model_executor/layers/sparse_attn_indexer_kpool.py
for img in "$B1" "$REL"; do echo "image $img $(docker image inspect "$img" --format '{{.Id}}')" | tee -a "$SUM"; done
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
filt() { grep -v "^WARNING\|cpp_extension\|No CUDA runtime"; }
RUN=(docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1
     --user "$(id -u):$(id -g)" -e HOME=/tmp --entrypoint python3)
chk() {  # label outdir command...
  local label=$1 tag=$2; shift 2
  mkdir -p "$OUT/$tag"
  "$@" 2>&1 | filt > "$OUT/$tag/$label.txt"; local rc=${PIPESTATUS[0]}
  printf '%-34s %-8s rc=%-3s %s\n' "$label" "$tag" "$rc" "$(grep -v '^\s*$' "$OUT/$tag/$label.txt" | tail -n 1 | cut -c1-120)" | tee -a "$SUM"
}
CID=$(docker create "$REL") && docker cp "$CID:$VP/$KP" "$W/release_kpool.py" && docker rm "$CID" >/dev/null

# ---- (1) bundle cpu_tests.sh, d1234 part, on the D7 image (SKIP1=1 skips this section)
if [ -z "$SKIP1" ]; then
t=b1
chk D1_test_col511_cpu $t "${RUN[@]}" -v "$DR/D1-col511-selection":/w:ro -v "$W/release_kpool.py":/p/release.py:ro "$B1" \
    /w/test_col511_cpu.py /p/release.py $VP/$KP
chk D1_fixture_cpucheck $t "${RUN[@]}" -v "$DR/D1-col511-selection":/w:ro -v "$OUT/$t":/o "$B1" \
    /w/topk_col511_fixture.py --version v091 --mode cpucheck --out /o
chk D2_test_fix_U $t "${RUN[@]}" -e TRITON_INTERPRET=1 -e TAILFIX_EXPECT=U -v "$DR/D2-kpool-tail-ring":/w:ro "$B1" /w/test_fix.py
chk D2_scenarios $t "${RUN[@]}" -e TRITON_INTERPRET=1 -v "$DR/D2-kpool-tail-ring":/w:ro "$B1" /w/tailring_scenarios.py --cpu-emulate
CK=(-v "$(dirname "$(dirname "$CKPT")")":/ckptrepo:ro)
CA=(--ckpt "/ckptrepo/snapshots/$(basename "$CKPT")" --layer 5)
chk D3_installed_expect_patched $t "${RUN[@]}" -v "$DR/D3-router-logits-bf16":/w:ro "${CK[@]}" "$B1" \
    /w/repro_router_logits.py --expect patched "${CA[@]}"
CF=(-v "$CKPT/config.json":/cfg/config.json:ro)
chk D4_installed_expect_patched $t "${RUN[@]}" -v "$DR/D4-routed-swiglu-clamp":/w:ro "${CF[@]}" "$B1" \
    /w/check_swiglu_plumbing.py --expect patched --config /cfg/config.json
chk D5_excluded_expect_release $t "${RUN[@]}" -v "$DR/D5-workspace-lock":/w:ro "$B1" /w/check_workspace_lock.py
chk D6_excluded_expect_release $t "${RUN[@]}" -e TRITON_INTERPRET=1 -v "$DR/D6-warmup-stale-tail-columns":/w:ro "$B1" \
    /w/check_warmup_tail_columns.py

fi

echo "done $(date -u +%FT%TZ)" | tee -a "$SUM"
