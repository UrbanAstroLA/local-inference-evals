#!/usr/bin/env bash
# KV-writer / query-projection comparison runner (PLAN.md amendment 16a). GPU0 only, one container at a time.
#   1. bounded wait for the B0 all-reduce fixture (RESULT.md of a full run, or no fixture container and both
#      GPUs idle for 2 minutes), at most WAIT_MAX seconds;
#   2. per image: pre-flight, budget gate (ceiling 0.25 occupied GPU-h incl. startup/compile), run with a wall cap,
#      cost line to costs.jsonl, container removed on every exit path; one repeat of an invalid attempt if budget allows;
#   3. CPU analysis (no GPU).
# No set -e: status queries are checked explicitly.
W=<runs>/e1-kvwriter
AR=<runs>/b0-allreduce
COSTS=$W/costs.jsonl
CEILING=0.25
WALL_CAP=${WALL_CAP:-360}            # seconds per container (0.1 GPU-h)
WAIT_MAX=${WAIT_MAX:-21600}          # 6 h bounded wait
IMG_B0=sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548
IMG_070=$(docker image inspect 48e254d94f58 --format '{{.Id}}' 2>/dev/null)
[ -n "$IMG_070" ] || { echo "ABORT: 0.7.0 image 48e254d94f58 missing"; exit 2; }
RUNS=$W/kvw-runs
mkdir -p "$RUNS"
LOG=$RUNS/runner.log
CN=""
cleanup() { [ -n "$CN" ] && docker rm -f "$CN" >/dev/null 2>&1; CN=""; }
trap 'cleanup; echo "runner exit $(date -u +%FT%TZ)" >> "$LOG"' EXIT
trap 'echo "interrupted" >> "$LOG"; exit 130' INT TERM
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

# ---- 0. freeze check: the files whose hashes were recorded before GPU data must be unchanged
( cd "$W" && sha256sum -c freeze/pre-gpu.sha256 >> "$LOG" 2>&1 ) || { say "ABORT: frozen files changed"; exit 2; }

# ---- 1. bounded wait for the all-reduce fixture
gpus_idle() {
  local busy=0
  while read -r IDX MEM UTIL; do
    LIM=1024; [ "$IDX" = 1 ] && LIM=4096
    if [ "$MEM" -gt "$LIM" ] || [ "$UTIL" -gt 10 ]; then busy=1; fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
  return $busy
}
T_WAIT0=$(date +%s); IDLE_SINCE=""
while true; do
  if ls "$AR"/runs/full-*/RESULT.md >/dev/null 2>&1 && ! docker ps --format '{{.Names}}' | grep -q 'allreduce\|fixture'; then
    say "all-reduce full run finished: $(ls "$AR"/runs/full-*/RESULT.md | tr '\n' ' ')"; break
  fi
  if ! docker ps --format '{{.Names}}' | grep -q 'allreduce\|fixture\|^glm53' && gpus_idle; then
    [ -z "$IDLE_SINCE" ] && IDLE_SINCE=$(date +%s)
    if [ $(( $(date +%s) - IDLE_SINCE )) -ge 120 ]; then say "no fixture container and GPUs idle for 2 min"; break; fi
  else
    IDLE_SINCE=""
  fi
  if [ $(( $(date +%s) - T_WAIT0 )) -ge "$WAIT_MAX" ]; then say "ABORT: wait exceeded ${WAIT_MAX}s"; exit 4; fi
  sleep 30
done

# ---- 2. per-image runs
used_gpu_h() { python3 -c "
import json,sys
t=0.0
try:
    for l in open('$COSTS'):
        l=l.strip()
        if l: t+=json.loads(l)['gpu_hours']
except FileNotFoundError: pass
print(round(t,4))"; }

preflight() {   # $1 = out file
  {
    echo "== preflight $(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
    nvidia-smi | tail -n 15
    docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
  } > "$1" 2>&1
  if docker ps --format '{{.Names}}' | grep -q '^glm53\|allreduce\|fixture'; then echo "glm53*/fixture container running"; return 1; fi
  local NONDESK=""
  for P in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    if tr '\0' ' ' < "/proc/$P/cmdline" 2>/dev/null | grep -q -- '--type=gpu-process'; then
      echo "ignoring desktop GPU process pid $P" >> "$1"
    else NONDESK="$NONDESK $P"; fi
  done
  if [ -n "$NONDESK" ]; then echo "compute processes:$NONDESK"; return 1; fi
  read -r MEM UTIL < <(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
  if [ "$MEM" -gt 1024 ] || [ "$UTIL" -gt 10 ]; then echo "GPU0 busy mem=${MEM}MiB util=${UTIL}%"; return 1; fi
  return 0
}

run_image() {   # $1 tag, $2 image id
  local TAG=$1 IMG=$2 ATT RC
  for ATT in 1 2; do
    local USED; USED=$(used_gpu_h)
    if ! python3 -c "import sys; sys.exit(0 if $USED + $WALL_CAP/3600 <= $CEILING + 1e-9 else 1)"; then
      say "BUDGET: used $USED + cap $(python3 -c "print(round($WALL_CAP/3600,4))") > $CEILING; not starting $TAG a$ATT"; return 5
    fi
    echo "used $USED + cap $(python3 -c "print(round($WALL_CAP/3600,4))") <= $CEILING" > "$RUNS/$TAG-a$ATT.budget.txt"
    local WHY; WHY=$(preflight "$RUNS/$TAG-a$ATT.preflight.txt") || { say "ABORT preflight $TAG: $WHY"; return 3; }
    CN="kvw-$TAG-a$ATT"
    local START T0 T1
    START=$(date -u +%FT%TZ); T0=$(date +%s)
    say "start $TAG a$ATT image $IMG"
    timeout "$WALL_CAP" docker run --rm --name "$CN" --gpus '"device=0"' --network none \
      --user "$(id -u):$(id -g)" -e HOME=/w/kvw-runs/home -e XDG_CACHE_HOME=/w/kvw-runs/cache/$TAG/xdg \
      -e B12X_COMPILE_CACHE_DIR=/w/kvw-runs/cache/$TAG/b12x -e TRITON_CACHE_DIR=/w/kvw-runs/cache/$TAG/triton \
      -e PYTHONDONTWRITEBYTECODE=1 -e IMAGE_TAG="$TAG" -e IMAGE_ID="$IMG" \
      -v "$W:/w" --entrypoint python3 "$IMG" /w/kvw_gpu_compare.py /w/kvw-inputs/kvw_inputs.pt "/w/kvw-runs/$TAG.pt" \
      > "$RUNS/$TAG-a$ATT.log" 2>&1
    RC=$?
    cleanup
    T1=$(date +%s)
    python3 -c "
import json
s=$T1-$T0
print(json.dumps(dict(start='$START', end='$(date -u +%FT%TZ)', seconds=s, gpus=1, gpu_hours=round(s/3600,4), mode='kvw',
    image='$IMG', tag='$TAG', attempt=$ATT, rc=$RC,
    purpose='KV-writer/query-projection comparison (PLAN.md amendment 16a), GPU0, model-free')))" >> "$COSTS"
    say "end $TAG a$ATT rc=$RC seconds=$((T1-T0))"
    if [ "$RC" = 0 ] && [ -s "$RUNS/$TAG.pt" ]; then return 0; fi
    say "invalid attempt $TAG a$ATT (rc=$RC)"
  done
  return 1
}

run_image b0 "$IMG_B0"; RB0=$?
run_image v070 "$IMG_070"; R070=$?
say "runs: b0 rc=$RB0 v070 rc=$R070 used $(used_gpu_h) GPU-h"

# ---- 3. CPU analysis
FILES=""
[ -s "$RUNS/b0.pt" ] && FILES="$FILES /w/kvw-runs/b0.pt"
[ -s "$RUNS/v070.pt" ] && FILES="$FILES /w/kvw-runs/v070.pt"
if [ -n "$FILES" ]; then
  docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$W:/w" -w /w/kvw-runs --entrypoint python3 \
    local/glm53-v0.9.1-bundle-d1234:local /w/kvw_analyze.py analyze /w/kvw-inputs/kvw_inputs.pt $FILES \
    > "$RUNS/analysis.log" 2>&1
  say "analysis rc=$?"
fi
( cd "$RUNS" && sha256sum *.pt *.json *.log 2>/dev/null > SHA256SUMS )
say "done"
