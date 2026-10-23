#!/bin/bash
# Level 3 GPU run (22ap): kernel_eq.py on GPU0 only, stock 0.9.1 base image, --network none, no model.
# Authorized 2026-10-23 by the operator ("Approve"): <= 0.35 GPU-h on GPU0. Wall cap 1200 s x 1 GPU = 0.333 GPU-h.
# One attempt: refuses if a cost line already exists. No `set -e`.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
IMG=ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
COSTS=$HERE/costs.jsonl
CAP=1200
touch "$COSTS" || exit 2
if [ -s "$COSTS" ]; then echo "REFUSED: a level-3 cost line already exists (one attempt)"; exit 2; fi
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "ABORT: base image missing"; exit 2; }
if docker ps --format '{{.Names}}' | grep -qE '^glm53|b0cache|mgtest'; then echo "ABORT: inference container running"; exit 3; fi
echo "== preflight $(date -u +%FT%TZ)" | tee "$HERE/preflight.txt"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a "$HERE/preflight.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a "$HERE/preflight.txt"
ok=0
for try in $(seq 1 12); do
  read -r MEM UTIL < <(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
  if [ "$MEM" -gt 1024 ]; then echo "ABORT: GPU0 memory ${MEM} MiB"; exit 3; fi
  echo "GPU0 sample $try: util ${UTIL}% mem ${MEM} MiB" | tee -a "$HERE/preflight.txt"
  [ "$UTIL" -le 5 ] && { ok=1; break; }
  sleep 5
done
[ "$ok" = 1 ] || { echo "ABORT: GPU0 busy"; exit 3; }
START=$(date -u +%FT%TZ); T0=$(date +%s)
timeout --kill-after=30 "$CAP" docker run --rm --name l3kernel --network none --gpus device=0 --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$HERE":/o --entrypoint python3 "$IMG" /o/kernel_eq.py /o/result.json 2>&1 | grep -v -i 'warning' | tee "$HERE/run.log"
RC=${PIPESTATUS[0]}
docker rm -f l3kernel >/dev/null 2>&1
T1=$(date +%s); SECS=$((T1 - T0))
GH=$(python3 -c "print(round($SECS/3600, 4))")
printf '{"item":"level3-kernel-eq","start":"%s","end":"%s","seconds":%s,"gpus":1,"gpu_hours":%s,"image":"%s","rc":%s,"wall_cap":%s,"purpose":"22ap level 3: inflated nsa vs p+1, stock kernels, synthetic"}\n' \
  "$START" "$(date -u +%FT%TZ)" "$SECS" "$GH" "${IMG##*@}" "$RC" "$CAP" >> "$COSTS"
echo "level3 rc=$RC, ${SECS}s x 1 GPU = $GH GPU-h"
