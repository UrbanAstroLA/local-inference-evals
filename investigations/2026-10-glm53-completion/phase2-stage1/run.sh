#!/bin/bash
# usage: run.sh <v070|v091> <cpucheck|smoke|full> <purpose...>
# Pre-flight GPU checks, then one container run; logs a JSON line to costs.jsonl.
FX=$(cd "$(dirname "$0")" && pwd)
VER=$1; MODE=$2; shift 2; PURPOSE="$*"
case $VER in
  v070) IMG=ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629;;
  v091) IMG=ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07;;
  *) echo "bad version"; exit 2;;
esac
GPUARGS=()
GPU=no
if [ "$MODE" != "cpucheck" ]; then
  GPU=yes
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee "$FX/preflight-$VER-$MODE.txt"
  docker ps --format '{{.Names}} {{.Image}}' | tee -a "$FX/preflight-$VER-$MODE.txt"
  if docker ps --format '{{.Names}}' | grep -q '^glm53'; then echo "ABORT: glm53* container running"; exit 3; fi
  read -r MEM UTIL < <(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
  if [ "$MEM" -gt 1024 ] || [ "$UTIL" -gt 5 ]; then echo "ABORT: GPU0 busy mem=${MEM}MiB util=${UTIL}%"; exit 3; fi
  GPUARGS=(--gpus '"device=0"')
fi
mkdir -p "$FX/cache/$VER" "$FX/home" "$FX/results"
START=$(date -u +%Y-%m-%dT%H:%M:%SZ); T0=$(date +%s.%N)
docker run --rm --network none "${GPUARGS[@]}" --user "$(id -u):$(id -g)" \
  -e HOME=/fx/home -e B12X_COMPILE_CACHE_DIR=/fx/cache/$VER -e TRITON_CACHE_DIR=/fx/cache/$VER/triton \
  -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python3 -v "$FX":/fx "$IMG" /fx/fixture.py --version "$VER" --mode "$MODE" --out /fx/results
RC=$?
T1=$(date +%s.%N); END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
SECS=$(python3 -c "print(round($T1-$T0,1))")
printf '{"start":"%s","end":"%s","seconds":%s,"gpu0":"%s","version":"%s","mode":"%s","rc":%s,"purpose":"%s"}\n' \
  "$START" "$END" "$SECS" "$GPU" "$VER" "$MODE" "$RC" "$PURPOSE" >> "$FX/costs.jsonl"
echo "rc=$RC seconds=$SECS"
exit $RC
