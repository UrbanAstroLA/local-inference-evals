#!/bin/bash
# D3 served-path numerical verification (PLAN.md amendment 14c item 2): one GPU (GPU0), offline, on the s1 capture,
# inside the bundle d1234-check image (its patched GateLinear + the stage-A helpers mounted read-only).
# Budget gate against bundle/costs.jsonl (gate ceiling 2.0 GPU-h incl. this step's cap), pre-flight, container removed
# on every exit path, one cost line. No `set -e`.  usage: run_d3_served.sh [ATTEMPT]
ATTEMPT=${1:-1}
D=$(cd "$(dirname "$0")" && pwd)
ROOT=<runs>/bundle
S1=$ROOT/gate-runs/s1-d1234-a1
MREPO=<models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1
MREV=0490d2f708b12145f6516555ab066aaeb401cd21
IMG=local/glm53-v0.9.1-bundle-d1234-check:local
IMG_EXP=sha256:b61793817388bc42b2477df4bbde5aca61df0567cf7c88ab29061633b06282ba
BUDGET=${BUDGET:-2.0}; WALL_CAP=${WALL_CAP:-900}; GPUS=1
CAP_GH=$(python3 -c "print(round($GPUS*$WALL_CAP/3600,3))")
OUT=$ROOT/gate-runs/d3-served-a$ATTEMPT
[ -e "$OUT" ] && { echo "ABORT: $OUT exists"; exit 2; }
mkdir -p "$OUT" || exit 2
USED=$(python3 -c "
import json; s=0.0
for l in open('$ROOT/costs.jsonl'):
    s+=float(json.loads(l).get('gpu_hours',0))
print(round(s,4))")
python3 -c "import sys; sys.exit(0 if $USED + $CAP_GH <= $BUDGET + 1e-9 else 1)" || { echo "ABORT: budget used $USED + cap $CAP_GH > $BUDGET" | tee "$OUT/budget.txt"; exit 2; }
echo "budget: used $USED + cap $CAP_GH <= $BUDGET GPU-h" | tee "$OUT/budget.txt"
[ "$(docker image inspect "$IMG" --format '{{.Id}}' 2>/dev/null)" = "$IMG_EXP" ] || { echo "ABORT: image id"; exit 2; }
echo "== preflight $(date -u +%FT%TZ)" | tee "$OUT/preflight.txt"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a "$OUT/preflight.txt"
docker ps --format '{{.Names}}' | grep -q '^glm53' && { echo "ABORT: glm53* container running"; exit 3; }
read -r MEM UTIL < <(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
{ [ "$MEM" -le 1024 ] && [ "$UTIL" -le 10 ]; } || { echo "ABORT: GPU0 busy mem=$MEM util=$UTIL"; exit 3; }
CN=bundle-d3-served-a$ATTEMPT
START=$(date -u +%Y-%m-%dT%H:%M:%SZ); T0=$(date +%s); RC_MAIN=null; DONE=0
cleanup() {
  [ "$DONE" = 1 ] && return; DONE=1
  docker rm -f "$CN" >/dev/null 2>&1
  T1=$(date +%s); SECS=$((T1 - T0)); GH=$(python3 -c "print(round($GPUS*$SECS/3600,4))")
  printf '{"start":"%s","end":"%s","seconds":%s,"gpus":%s,"gpu_hours":%s,"mode":"d3-served","attempt":%s,"image":"%s","rc":%s,"purpose":"D3 fp32 path as served: offline controls A0/Aprime/D/C on the s1 capture (PLAN.md amendment 14c item 2), GPU0"}\n' \
    "$START" "$(date -u -d @"$T1" +%Y-%m-%dT%H:%M:%SZ)" "$SECS" "$GPUS" "$GH" "$ATTEMPT" "$IMG_EXP" "$RC_MAIN" >> "$ROOT/costs.jsonl"
  echo "occupied ${SECS}s x $GPUS GPU = $GH GPU-h"
}
trap cleanup EXIT
trap 'echo signal; exit 4' INT TERM
timeout "$WALL_CAP" docker run --rm --name "$CN" --gpus '"device=0"' --network none --ipc host \
  -v "$D:/d3:ro" -v "$S1:/capture:ro" -v "$MREPO:/model-repo:ro" -v "$OUT:/out" --entrypoint python3 "$IMG" \
  /d3/d3_served_check.py /capture /model-repo/snapshots/$MREV /out 2>&1 | grep -v "^WARNING\|^W1012" | tee "$OUT/run.log"
RC_MAIN=${PIPESTATUS[0]}
echo "rc=$RC_MAIN"
exit "$RC_MAIN"
