#!/bin/bash
# B0 TP2 all-reduce fixture runner (PLAN.md amendment 16, first priority).
#
#   run.sh cpucheck "<purpose>"   CPU only: cpu_tests.py inside the B0 image, no GPUs
#   run.sh smoke    "<purpose>"   two GPUs, smoke profile (wall cap 6 min)
#   run.sh full     "<purpose>"   two GPUs, full profile  (wall cap 10 min)
#   run.sh preflight "<purpose>"  pre-flight and budget gate for a full run only; no GPU use
#
# Every GPU run: pre-flight (no glm53* or fixture container, GPU0 idle, GPU1 only the
# desktop), budget gate (cost so far + this run's wall cap <= CEILING_GPU_H, default 0.5),
# one container with both GPUs and --ipc host, wall cap enforced by docker kill, a cost
# line in costs.jsonl and container removal on every exit path, then the CPU-only
# analysis (analyze.py) after the GPUs are released.
# No `set -e`: status queries are checked explicitly.

FX=$(cd "$(dirname "$0")" && pwd)
MODE=$1; shift; PURPOSE="$*"
IMG=sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548
NAME=b0-allreduce-fixture
CEILING_GPU_H=${CEILING_GPU_H:-0.5}
COSTS="$FX/costs.jsonl"
case "$MODE" in
  cpucheck) CAP_S=900;;
  smoke) CAP_S=${CAP_S:-360};;
  full|preflight) CAP_S=${CAP_S:-600};;
  *) echo "usage: run.sh cpucheck|smoke|full|preflight <purpose>"; exit 2;;
esac
[ -n "$PURPOSE" ] || { echo "purpose required"; exit 2; }

if [ "$(docker image inspect --format '{{.Id}}' "$IMG" 2>/dev/null)" != "$IMG" ]; then
  echo "ABORT: B0 image $IMG not present"; exit 3
fi
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$FX/runs/$MODE-$STAMP"
mkdir -p "$OUT/home" "$FX/cache" || { echo "ABORT: cannot create $OUT"; exit 3; }
( cd "$FX" && sha256sum fixture.py patterns.py analyze.py cpu_tests.py run.sh ) > "$OUT/SHA256SUMS"

if [ "$MODE" = cpucheck ]; then
  docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/out/home \
    -e PYTHONDONTWRITEBYTECODE=1 -v "$FX":/fx:ro -v "$OUT":/out \
    --entrypoint python3 "$IMG" /fx/cpu_tests.py /out/cpu_tests.json > "$OUT/cpu_tests.log" 2>&1
  RC=$?
  tail -3 "$OUT/cpu_tests.log"; echo "rc=$RC out=$OUT"
  exit $RC
fi

# ---------------------------------------------------------------- pre-flight
PRE="$OUT/preflight.txt"
{
  date -u +%FT%TZ
  docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader
} > "$PRE" 2>&1
if docker ps --format '{{.Names}}' | grep -Eq "^glm53|^$NAME\$"; then
  echo "ABORT: a glm53* or fixture container is running"; cat "$PRE"; exit 3
fi
MAXU0=0; MAXU1=0
for i in 1 2 3; do
  U0=$(nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  U1=$(nvidia-smi -i 1 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  [ "${U0:-100}" -gt "$MAXU0" ] && MAXU0=${U0:-100}
  [ "${U1:-100}" -gt "$MAXU1" ] && MAXU1=${U1:-100}
  sleep 1
done
M0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
M1=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
echo "max util gpu0=$MAXU0 gpu1=$MAXU1; mem gpu0=$M0 gpu1=$M1 MiB" | tee -a "$PRE"
UUID0=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader | tr -d ' ')
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | tr -d ' ' | grep -q "$UUID0"; then
  echo "ABORT: compute process on GPU0"; exit 3
fi
# GPU1 carries the desktop (graphics clients are not listed by --query-compute-apps).
if [ "${M0:-99999}" -gt 1024 ] || [ "$MAXU0" -gt 5 ] || [ "${M1:-99999}" -gt 4096 ] || [ "$MAXU1" -gt 15 ]; then
  echo "ABORT: GPUs not idle (gpu0 <=1024 MiB/<=5%, gpu1 desktop only <=4096 MiB/<=15%)"; exit 3
fi

# ---------------------------------------------------------------- budget gate
USED=$(python3 -c "
import json,sys
t=0.0
try:
    for l in open('$COSTS'):
        l=l.strip()
        if l: t+=float(json.loads(l).get('gpu_hours',0))
except FileNotFoundError: pass
print(round(t,4))")
CAP_H=$(python3 -c "print(round($CAP_S*2/3600,4))")
OKB=$(python3 -c "print(1 if $USED + $CAP_H <= $CEILING_GPU_H + 1e-9 else 0)")
echo "budget: used $USED + cap $CAP_H <= $CEILING_GPU_H GPU-h ? $OKB" | tee "$OUT/budget.txt"
[ "$OKB" = 1 ] || { echo "ABORT: budget gate"; exit 3; }
if [ "$MODE" = preflight ]; then echo "pre-flight and budget gate passed; out=$OUT"; exit 0; fi

# ---------------------------------------------------------------- run
START=$(date -u +%FT%TZ); T0=$(date +%s)
RC=99; LOGGED=0; SMI_PID=""
cleanup() {
  [ -n "$SMI_PID" ] && kill "$SMI_PID" 2>/dev/null
  if docker ps -a --format '{{.Names}}' | grep -q "^$NAME\$"; then
    docker logs "$NAME" > "$OUT/container.log" 2>&1
    docker kill "$NAME" >/dev/null 2>&1
    docker rm -f "$NAME" >/dev/null 2>&1
  fi
  if [ "$LOGGED" = 0 ]; then
    LOGGED=1
    T1=$(date +%s); SECS=$((T1 - T0))
    GH=$(python3 -c "print(round($SECS*2/3600,4))")
    printf '{"start":"%s","end":"%s","seconds":%s,"gpus":2,"gpu_hours":%s,"mode":"%s","image":"%s","rc":%s,"run_dir":"%s","purpose":"%s"}\n' \
      "$START" "$(date -u +%FT%TZ)" "$SECS" "$GH" "$MODE" "$IMG" "$RC" "$OUT" "$PURPOSE" >> "$COSTS"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > "$OUT/postflight.txt" 2>&1
  fi
}
trap 'RC=130; cleanup; exit 130' INT TERM
trap cleanup EXIT

( while true; do nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv,noheader; sleep 5; done ) > "$OUT/nvidia-smi.log" 2>&1 &
SMI_PID=$!

docker run -d --name "$NAME" --gpus all --ipc host --network none \
  --user "$(id -u):$(id -g)" \
  -e HOME=/out/home -e XDG_CACHE_HOME=/cache/xdg \
  -e B12X_COMPILE_CACHE_DIR=/cache/b12x -e TRITON_CACHE_DIR=/cache/triton \
  -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo -e NCCL_DEBUG=WARN \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=384KB -e VLLM_B12X_PCIE_EAGER=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$FX":/fx:ro -v "$OUT":/out -v "$FX/cache":/cache \
  --entrypoint python3 "$IMG" /fx/fixture.py --profile "$MODE" --out /out > "$OUT/container.id" 2>&1
if [ $? -ne 0 ]; then RC=98; cat "$OUT/container.id"; exit $RC; fi

DEADLINE=$((T0 + CAP_S))
while [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ]; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "wall cap ${CAP_S}s reached: stopping"
    RC=124
    break
  fi
  sleep 5
done
if [ "$RC" != 124 ]; then
  RC=$(docker inspect -f '{{.State.ExitCode}}' "$NAME" 2>/dev/null || echo 97)
fi
cleanup
trap - EXIT
echo "fixture rc=$RC; GPU part done; cost logged"

# ---------------------------------------------------------------- analysis (CPU only)
docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/out/home \
  -e PYTHONDONTWRITEBYTECODE=1 -v "$FX":/fx:ro -v "$OUT":/out \
  --entrypoint python3 "$IMG" /fx/analyze.py /out > "$OUT/analyze.log" 2>&1
echo "analysis rc=$?: $(head -1 "$OUT/analyze.log")"
echo "out=$OUT"
exit "$RC"
