#!/bin/bash
# All CPU tests of the captured-p3b package (no GPU anywhere: every container is `docker run --rm --network none
# -e CUDA_VISIBLE_DEVICES=` without --gpus).  Writes tests/RESULTS.txt (summary) and tests/out/* (details).
set -u
T=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$T/.." && pwd)
O=$T/out
mkdir -p "$O"
IMG91=ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
CAPIMG=local/glm53-v091-p3b-capture:local
U="$(id -u):$(id -g)"
RES=$T/RESULTS.txt
{ echo "captured-p3b CPU tests $(date -u +%FT%TZ) (host $(hostname); no GPU used)"; echo; } > "$RES"
step() {  # name command...
  local name=$1; shift
  local out rc
  out=$("$@" 2>&1); rc=$?
  echo "$out" > "$O/$(echo "$name" | tr ' /' '__').log"
  local tail_; tail_=$(echo "$out" | grep -E "passed|^PASS|^FAIL|rc=" | tail -3 | tr '\n' ' ')
  printf '%-58s rc=%s  %s\n' "$name" "$rc" "$tail_" | tee -a "$RES"
}
step "1 additive hook gate (make_hooked.py --check)" python3 "$ROOT/build/make_hooked.py" --check
step "2 capture module, CPU (T1-T11)" docker run --rm --network none -e CUDA_VISIBLE_DEVICES= --user "$U" -e HOME=/tmp \
  -v "$ROOT":/p:ro -v "$O":/o --entrypoint python3 "$IMG91" /p/tests/test_capture_cpu.py /p/build/files/vllm/p3b_capture_hooks.py /o
step "3 capture module AS INSTALLED in the built image (T1-T11)" docker run --rm --network none -e CUDA_VISIBLE_DEVICES= \
  --user "$U" -e HOME=/tmp -v "$ROOT":/p:ro -v "$O":/o --entrypoint python3 "$CAPIMG" /p/tests/test_capture_cpu.py \
  /usr/local/lib/python3.12/dist-packages/vllm/p3b_capture_hooks.py /o
step "4 register()/install() vs the image's vLLM classes (G1-G4)" docker run --rm --network none -e CUDA_VISIBLE_DEVICES= \
  --user "$U" -e HOME=/tmp -v "$ROOT":/p:ro -v "$O":/o --entrypoint python3 "$IMG91" /p/tests/test_register_cpu.py \
  /p/build/files/vllm/p3b_capture_hooks.py /o
step "5 replay scripts --dry on fabricated captures" bash "$T/test_replay_dry.sh"
step "6 runner + replay wrapper (refusals, budget, MOCK sessions)" bash "$T/test_wrappers.sh"
step "7 post-build image gates (build/post-build-gates.txt)" sh -c "cat '$ROOT/build/post-build-gates.txt'; grep -c '^PASS' '$ROOT/build/post-build-gates.txt' | sed 's/$/ passed/'"
step "8 seeds 72301-72316 unused under <data-runs>" sh -c "
  hits=\$(grep -rlE '(seed[\"'\"'\"' :=]+|SEED0 *= *)723(0[1-9]|1[0-6])\b' <data-runs> --include=*.py --include=*.json --include=*.jsonl 2>/dev/null | grep -v '/captured-p3b/' | head -5)
  [ -z \"\$hits\" ] && echo 'PASS no prior use of 72301-72316 1 passed' || { echo \"FAIL seeds used in: \$hits\"; exit 1; }"
echo >> "$RES"
grep -q "rc=[1-9]" "$RES" && { echo "SOME FAILED" | tee -a "$RES"; exit 1; }
echo "ALL PASSED" | tee -a "$RES"
