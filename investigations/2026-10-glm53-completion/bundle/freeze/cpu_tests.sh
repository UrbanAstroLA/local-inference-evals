#!/bin/bash
# CPU-only checks of the bundle images (PLAN.md amendment 14 (2)); no GPU, no network inside the containers.
# Runs the existing defect-repros checks against the INSTALLED files of each image (nothing mounted over them),
# and the release behaviour on the release image. Outputs: $OUT/<image>/<check>.txt and $OUT/summary.txt.
# No `set -e`: each exit code is recorded explicitly.
R=<runs>
DR=$R/defect-repros
OUT=${OUT:-$R/bundle/cpu-tests}
REL=ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
D124=local/glm53-v0.9.1-bundle-d124:local
D1234=local/glm53-v0.9.1-bundle-d1234:local
CKPT=${CKPT:-<models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1/snapshots/0490d2f708b12145f6516555ab066aaeb401cd21}
VP=/usr/local/lib/python3.12/dist-packages/vllm
KP=model_executor/layers/sparse_attn_indexer_kpool.py
mkdir -p "$OUT"; SUM=$OUT/summary.txt; : > "$SUM"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
filt() { grep -v "^WARNING\|cpp_extension\|No CUDA runtime"; }
RUN=(docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1
     --user "$(id -u):$(id -g)" -e HOME=/tmp --entrypoint python3)
# one check: label image expect_rc command...
chk() {
  local label=$1 img=$2 tag=$3; shift 3
  mkdir -p "$OUT/$tag"
  "$@" 2>&1 | filt > "$OUT/$tag/$label.txt"; local rc=${PIPESTATUS[0]}
  printf '%-34s %-8s rc=%-3s %s\n' "$label" "$tag" "$rc" "$(grep -v '^\s*$' "$OUT/$tag/$label.txt" | tail -n 1 | cut -c1-110)" | tee -a "$SUM"
}
for img in "$REL" "$D124" "$D1234"; do echo "image $img $(docker image inspect "$img" --format '{{.Id}}')" | tee -a "$SUM"; done
# release kpool file for D1's side-by-side test
CID=$(docker create "$REL") && docker cp "$CID:$VP/$KP" "$W/release_kpool.py" && docker rm "$CID" >/dev/null
for pair in "d124:$D124" "d1234:$D1234"; do
  t=${pair%%:*}; img=${pair#*:}
  CID=$(docker create "$img") && docker cp "$CID:$VP/$KP" "$W/${t}_kpool.py" && docker rm "$CID" >/dev/null
done

# ---- D1 column-511 (test_col511_cpu.py: release slice vs the INSTALLED corrected helpers; fixture self-check)
for pair in "d124:$D124" "d1234:$D1234"; do
  t=${pair%%:*}; img=${pair#*:}
  chk D1_test_col511_cpu "$img" "$t" "${RUN[@]}" -v "$DR/D1-col511-selection":/w:ro \
      -v "$W/release_kpool.py":/p/release.py:ro "$img" /w/test_col511_cpu.py /p/release.py $VP/$KP
  chk D1_fixture_cpucheck "$img" "$t" "${RUN[@]}" -v "$DR/D1-col511-selection":/w:ro -v "$OUT/$t":/o "$img" \
      /w/topk_col511_fixture.py --version v091 --mode cpucheck --out /o
done

# ---- D2 tail ring (installed files; TRITON_INTERPRET runs the real tail kernels)
chk D2_test_fix_release "$REL" release "${RUN[@]}" -e TRITON_INTERPRET=1 -e TAILFIX_EXPECT=release \
    -v "$DR/D2-kpool-tail-ring":/w:ro "$REL" /w/test_fix.py
chk D2_scenarios_release "$REL" release "${RUN[@]}" -e TRITON_INTERPRET=1 -v "$DR/D2-kpool-tail-ring":/w:ro "$REL" \
    /w/tailring_scenarios.py --cpu-emulate
mkdir -p "$OUT/release/mech"
chk D2_mechanism_release "$REL" release "${RUN[@]}" -e TRITON_INTERPRET=1 -v "$DR/D2-kpool-tail-ring":/w:ro \
    -v "$OUT/release/mech":/o -w /o "$REL" /w/mechanism_check.py
for pair in "d124:$D124" "d1234:$D1234"; do
  t=${pair%%:*}; img=${pair#*:}
  chk D2_test_fix_U "$img" "$t" "${RUN[@]}" -e TRITON_INTERPRET=1 -e TAILFIX_EXPECT=U \
      -v "$DR/D2-kpool-tail-ring":/w:ro "$img" /w/test_fix.py
  chk D2_scenarios "$img" "$t" "${RUN[@]}" -e TRITON_INTERPRET=1 -v "$DR/D2-kpool-tail-ring":/w:ro "$img" \
      /w/tailring_scenarios.py --cpu-emulate
done

# ---- D3 router (release behaviour on release and d124; corrected dispatch on d1234), synthetic + one real layer
CK=(-v "$(dirname "$(dirname "$CKPT")")":/ckptrepo:ro)
CA=(--ckpt "/ckptrepo/snapshots/$(basename "$CKPT")" --layer 5)
chk D3_release "$REL" release "${RUN[@]}" -v "$DR/D3-router-logits-bf16":/w:ro "${CK[@]}" "$REL" /w/repro_router_logits.py "${CA[@]}"
chk D3_absent_expect_release "$D124" d124 "${RUN[@]}" -v "$DR/D3-router-logits-bf16":/w:ro "${CK[@]}" "$D124" /w/repro_router_logits.py "${CA[@]}"
chk D3_installed_expect_patched "$D1234" d1234 "${RUN[@]}" -v "$DR/D3-router-logits-bf16":/w:ro "${CK[@]}" "$D1234" /w/repro_router_logits.py --expect patched "${CA[@]}"

# ---- D4 swiglu_limit plumbing (static AST checks of the installed exl3.py + model config)
CF=(-v "$CKPT/config.json":/cfg/config.json:ro)
chk D4_release "$REL" release "${RUN[@]}" -v "$DR/D4-routed-swiglu-clamp":/w:ro "${CF[@]}" "$REL" /w/check_swiglu_plumbing.py --config /cfg/config.json
for pair in "d124:$D124" "d1234:$D1234"; do
  t=${pair%%:*}; img=${pair#*:}
  chk D4_installed_expect_patched "$img" "$t" "${RUN[@]}" -v "$DR/D4-routed-swiglu-clamp":/w:ro "${CF[@]}" "$img" \
      /w/check_swiglu_plumbing.py --expect patched --config /cfg/config.json
done

# ---- D5 / D6 excluded: the release behaviour must still be present in the bundles
for pair in "d124:$D124" "d1234:$D1234"; do
  t=${pair%%:*}; img=${pair#*:}
  chk D5_excluded_expect_release "$img" "$t" "${RUN[@]}" -v "$DR/D5-workspace-lock":/w:ro "$img" /w/check_workspace_lock.py
  chk D6_excluded_expect_release "$img" "$t" "${RUN[@]}" -e TRITON_INTERPRET=1 -v "$DR/D6-warmup-stale-tail-columns":/w:ro "$img" \
      /w/check_warmup_tail_columns.py
done
echo "done $(date -u +%FT%TZ)" | tee -a "$SUM"
