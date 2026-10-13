#!/bin/bash
# B1 mapping-guard session (PLAN.md amendments 18a/18b, step 2c + appended phase 2d; copy of the frozen
# b0-mapguard runner with the a3 fixes).  PREPARED BY CPU WORK; the GPU run is launched by the coordinator.
#   APPROVED_GPU_H=2.0 run_mapguard.sh [ATTEMPT]      (DRY_RUN=1: budget gate + image pin only)
# Budget (amendment 18b): item "b1-mapguard", total <= APPROVED_GPU_H summed over every cost line of the item
# (invalid attempts included), at most 2 attempts in all; per attempt the wall cap is sized for one complete
# session (WALL_CAP 1740 s x 2 GPUs = 0.967 GPU-h); refused if the remaining budget cannot fit MIN_WALL.
# One server: the B1 computation (derived image = B1 + inert observe-only hooks, GLM53_MAPGUARD=device),
# recipe glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234+d7 with IMAGE overridden, both ranks, shipped
# graph mode (no EXTRA_SERVE_ARGS), DFlash2 x3.  Workload: client_mapguard.py (existing fixtures + 2d).
# Unattended-safe: budget gate (all b1-mapguard cost lines, invalid runs included, + this attempt's wall cap
# <= APPROVED_GPU_H), image-id pin, pre-flight (no glm53* container; GPUs idle; only desktop type-G graphics
# clients gnome-shell/Xwayland/xdg-desktop-portal ignored, as amendment 16d), guard-installed check before
# traffic, wall cap, stop on every exit path (EXIT/INT/TERM trap), docker cp of the guard output, one cost line
# counted to the moment the GPUs are released, then the frozen CPU analysis.  No `set -e`.
ATTEMPT=${1:-1}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=<runs>/d7/step2
COSTS=$ROOT/costs.jsonl
ITEM=b1-mapguard
MAX_ATTEMPTS=2                   # amendment 18b: at most one retry
REQ=<runs>/phase2-stage2/requests.json
ST=<stacks>/glm53-flash-v0.9.0
RC=<recipes>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234+d7
CN=glm53-flash-exl3-b12x-vllm
IMG=local/glm53-v0.9.1-b1-mapguard:local
GPUS=2
WALL_CAP=${WALL_CAP:-1740}       # 29 min from server start to forced stop -> 0.967 GPU-h per attempt (a3: ready 713 s,
                                 # frozen phases ~300 s, cleanup ~70 s; + 2d ~60-120 s; 2d is skipped near the deadline)
MIN_WALL=${MIN_WALL:-1260}       # below 21 min the frozen phases cannot fit (readiness up to ~800 s + ~300 s + reserve)
RESERVE=${RESERVE:-150}          # seconds kept after the client deadline for readout + stop
# Post-a2 trimmed rerun (amendment 17c follow-up): MIN_WALL=960 RESERVE=95 PHASES="P0 warm T0 F0 T1 S P1"
# (a2 measured: cleanup -> GPUs released in 66 s; readiness ~800 s).  PHASES is read by the client.
GATE_CAP=${GATE_CAP:-1020}       # readiness (start-up incl. graph capture and release warm-up) cap
DEBUG="GLM53_MAPGUARD=device GLM53_MAPGUARD_DIR=/tmp/mapguard GLM53_MAPGUARD_POLL_SECS=20"
OUT=$ROOT/run-b1-a$ATTEMPT
if [ -e "$OUT" ]; then echo "ABORT: $OUT exists (use another ATTEMPT)"; exit 2; fi
mkdir -p "$ROOT" && touch "$COSTS" || exit 2

# ---- budget gate (total approved for this item, invalid runs included)
if [ -z "${APPROVED_GPU_H:-}" ]; then echo "REFUSED: set APPROVED_GPU_H (amendment 18b: 2.0 for the item)"; exit 2; fi
if python3 -c "import sys; sys.exit(0 if float('$APPROVED_GPU_H') <= 2.0 else 1)"; then :; else
  echo "REFUSED: APPROVED_GPU_H=$APPROVED_GPU_H exceeds the authorized item total 2.0 (amendment 18b)"; exit 2; fi
read -r SPENT NATT < <(python3 -c "
import json
s=0.0; n=0
for l in open('$COSTS'):
    l=l.strip()
    if l:
        d=json.loads(l)
        if d.get('item')=='$ITEM': s+=float(d.get('gpu_hours',0)); n+=1
print(round(s,4), n)")
if [ "$NATT" -ge "$MAX_ATTEMPTS" ]; then echo "REFUSED: $NATT attempts already recorded for $ITEM (max $MAX_ATTEMPTS, amendment 18b)"; exit 2; fi
LEFT_S=$(python3 -c "print(int(($APPROVED_GPU_H - $SPENT) * 3600 / $GPUS))")
if [ "$LEFT_S" -lt "$WALL_CAP" ]; then WALL_CAP=$LEFT_S; fi
if [ "$WALL_CAP" -lt "$MIN_WALL" ]; then
  echo "REFUSED: spent ${SPENT} of ${APPROVED_GPU_H} GPU-h leaves ${WALL_CAP}s wall < ${MIN_WALL}s needed for a complete session"; exit 2
fi
CAP_GH=$(python3 -c "print(round($GPUS*$WALL_CAP/3600,3))")
mkdir -p "$OUT" && cd "$OUT" || exit 2
echo "budget: spent $SPENT + this attempt's cap $CAP_GH (wall ${WALL_CAP}s) <= approved $APPROVED_GPU_H GPU-h" | tee "$OUT/budget.txt"

# ---- image identity (id recorded at build time; a rebuilt image is refused)
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "ABORT: image $IMG missing"; exit 2; }
IMG_ID=$(docker image inspect "$IMG" --format '{{.Id}}')
EXP=$(awk -v t="$IMG" '$1==t {print $2}' "$HERE/images.txt")
echo "$IMG $IMG_ID" > "$OUT/image-id.txt"
if [ -z "$EXP" ] || [ "$IMG_ID" != "$EXP" ]; then echo "ABORT: $IMG is $IMG_ID, expected '$EXP'"; exit 2; fi
( cd "$HERE" && sha256sum client_mapguard.py run_mapguard.sh images.txt ../analysis/analyze_mapguard.py ) > "$OUT/scripts.sha256"
# frozen hashes (d7/step2/SHA256SUMS): every file that runs must match before any GPU use
( cd "$ROOT" && sha256sum -c --quiet SHA256SUMS ) > "$OUT/sha256-check.txt" 2>&1 || { echo "ABORT: frozen hashes differ (see $OUT/sha256-check.txt)"; cat "$OUT/sha256-check.txt"; exit 2; }
[ -f <gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1__pass1/wrldsuksgo2mars__GLM-5.3-Flash-EXL3-K3.25-v1/samples_gpqa_diamond_thinking_2026-10-08T20-01-05.904768.jsonl ] || { echo "ABORT: no GPQA samples for 2d"; exit 2; }
cp "$REQ" "$OUT/requests.json" || { echo "ABORT: no requests.json"; exit 2; }
[ -f <runs>/bundle/gate/gpqa_fixture.LOCAL.json ] || { echo "ABORT: no GPQA fixture"; exit 2; }
if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY_RUN: budget gate, attempt count, frozen hashes and image check passed"; rm -rf "$OUT"; exit 0; fi

# ---- pre-flight
echo "== preflight $(date -u +%FT%TZ)" | tee "$OUT/preflight.txt"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a "$OUT/preflight.txt"
nvidia-smi | tail -n 15 >> "$OUT/preflight.txt"
docker ps --format '{{.Names}} {{.Image}} {{.Status}}' | tee -a "$OUT/preflight.txt"
if docker ps --format '{{.Names}}' | grep -qE '^glm53|b0cache|mgtest'; then echo "ABORT: inference/fixture container running"; exit 3; fi
PROCS=$(nvidia-smi 2>/dev/null | awk '/Processes:/{p=1;next} p && /^\|/ && $2 ~ /^[0-9]+$/ {print}' \
  | grep -vE '[[:space:]]G[[:space:]].*(gnome-shell|Xwayland|xdg-desktop-portal|firefox|snap-store)' \
  | grep -vE '[[:space:]]C\+G[[:space:]].*--disable-logging')   # desktop clients as in the other runners (amendment 17c)
if [ -n "$PROCS" ]; then echo "ABORT: GPU clients other than desktop type-G graphics:"; echo "$PROCS"; exit 3; fi
while read -r IDX MEM UTIL; do
  LIM=1024; [ "$IDX" = 1 ] && LIM=4096          # desktop graphics on GPU 1 (amendment 11b)
  if [ "$MEM" -gt "$LIM" ] || [ "$UTIL" -gt 5 ]; then echo "ABORT: GPU$IDX busy mem=${MEM}MiB util=${UTIL}%"; exit 3; fi
done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')

START=$(date -u +%Y-%m-%dT%H:%M:%SZ); T0=$(date +%s); TEND=""; RC_MAIN=""; DONE=0
cost_line() {
  local T1=${TEND:-$(date +%s)}
  local SECS=$((T1 - T0))
  local GH=$(python3 -c "print(round($GPUS*$SECS/3600,4))")
  printf '{"item":"b1-mapguard","start":"%s","end":"%s","seconds":%s,"gpus":%s,"gpu_hours":%s,"attempt":%s,"image":"%s","rc":%s,"wall_cap":%s,"purpose":"B1 live mapping guard session + phase 2d (PLAN.md amendments 18a/18b)"}\n' \
    "$START" "$(date -u -d @"$T1" +%Y-%m-%dT%H:%M:%SZ)" "$SECS" "$GPUS" "$GH" "$ATTEMPT" "$IMG_ID" "${RC_MAIN:-null}" "$WALL_CAP" >> "$COSTS"
  echo "occupied ${SECS}s x $GPUS GPU = ${GH} GPU-h"
}
cleanup() {
  [ "$DONE" = 1 ] && return
  DONE=1
  [ -n "${WD:-}" ] && kill "$WD" 2>/dev/null
  echo "== cleanup $(date -u +%FT%TZ)"
  if [ "$(docker inspect --format '{{.State.Running}}' "$CN" 2>/dev/null)" = true ]; then
    docker exec "$CN" sh -c 'mkdir -p /tmp/mapguard && touch /tmp/mapguard/DUMP' >/dev/null 2>&1
    sleep 8                                       # poll threads check the trigger every 2 s
    docker logs "$CN" > "$OUT/server.log" 2>&1
    docker exec "$CN" sh -c 'cd /usr/local/lib/python3.12/dist-packages/vllm && sha256sum mapguard_hooks.py model_executor/layers/sparse_attn_indexer_kpool.py v1/attention/backends/mla/b12x_mla_sparse.py v1/core/sched/scheduler.py; env | grep -E "^(GLM53|VLLM)" | sort' > "$OUT/container-files-env.txt" 2>&1
  fi
  ( cd "$ST" && ./stop.sh ); echo "stop rc=$?"
  if docker ps --format '{{.Names}}' | grep -q "^$CN\$"; then docker stop -t 30 "$CN"; echo "docker stop rc=$?"; fi
  docker ps --format '{{.Names}}' | grep '^glm53' && echo "WARNING: glm53* container still running" || echo "no glm53* container running"
  TEND=$(date +%s)                                # GPUs released; everything below is CPU-only
  cost_line
  mkdir -p "$OUT/mapguard"; docker cp "$CN:/tmp/mapguard/." "$OUT/mapguard/" >/dev/null 2>&1; echo "docker cp rc=$?"
  [ -s "$OUT/server.log" ] || docker logs "$CN" > "$OUT/server.log" 2>&1
  python3 "$HERE/../analysis/analyze_mapguard.py" "$OUT" 2>&1 | tee "$OUT/verdict-console.txt"
}
trap cleanup EXIT
trap 'echo "signal"; RC_MAIN=130; exit 4' INT TERM
# hard wall cap: a watchdog stops the server even if this script hangs
# watchdog fires early enough that the stop (~60-70 s, run-a2) completes within the cap
( sleep $(( WALL_CAP > 300 ? WALL_CAP - 75 : WALL_CAP )); echo "WALL CAP reached"; docker stop -t 30 "$CN" >/dev/null 2>&1 ) &
WD=$!

echo "== start $START image=$IMG ($IMG_ID) cap ${WALL_CAP}s"
( unset EXTRA_SERVE_ARGS; cd "$ST" && source "$RC/env.sh" && export SPECULATIVE_METHOD=dflash2 DFLASH_TOKENS=3 \
    IMAGE="$IMG" DEBUG_ENV="$DEBUG" && ./start.sh ) > "$OUT/start.log" 2>&1
SRC=$?
cat "$OUT/start.log"
if [ $SRC -ne 0 ]; then echo "start failed rc=$SRC"; RC_MAIN=5; kill $WD 2>/dev/null; exit 5; fi
echo "== readiness gate (cap ${GATE_CAP}s)"
READY=0
while [ $(( $(date +%s) - T0 )) -lt "$GATE_CAP" ]; do
  RUN=$(docker inspect --format '{{.State.Running}} {{.RestartCount}}' "$CN" 2>/dev/null)
  if [ "$RUN" != "true 0" ]; then echo "container state: '$RUN' -> abort"; RC_MAIN=6; kill $WD 2>/dev/null; exit 6; fi
  if docker logs "$CN" 2>&1 | grep -qE 'Engine core initialization failed|CUDA out of memory|Traceback'; then
    echo "engine error during start-up -> abort"; RC_MAIN=6; kill $WD 2>/dev/null; exit 6
  fi
  # early abort on any guard error during start-up (graph capture included): keeps an invalid
  # attempt short enough for the invalid-run allowance.  Pattern = the quoted record kind; the bare
  # word also matched the "guard_errors": 0 key of every readout and aborted run-a2 (post-inspection fix)
  NEARLY=$(docker exec "$CN" sh -c 'grep -h "\"guard_error\"" /tmp/mapguard/*.jsonl 2>/dev/null | wc -l' 2>/dev/null)
  if [ "${NEARLY:-0}" -gt 0 ]; then
    docker exec "$CN" sh -c 'grep -h "\"guard_error\"" /tmp/mapguard/*.jsonl | head -n 5' 2>&1 | tee -a "$OUT/gate.txt"
    echo "INVALID: guard error during start-up -> abort"; RC_MAIN=8; kill $WD 2>/dev/null; exit 8
  fi
  if docker exec "$CN" test -f /tmp/glm53-release-ready 2>/dev/null && curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null; then
    READY=1; break
  fi
  sleep 10
done
if [ $READY -ne 1 ]; then echo "readiness timeout"; RC_MAIN=7; kill $WD 2>/dev/null; exit 7; fi
echo "ready after $(( $(date +%s) - T0 ))s" | tee "$OUT/gate.txt"
docker logs "$CN" 2>&1 | grep -n "cudagraph_mode\|Graph capturing finished\|BREAKABLE\|GPU KV cache size\|guard_error" | cut -c1-300 | tail -n 20 | tee -a "$OUT/gate.txt"
# instrumentation present on both worker ranks and the scheduler side before any client traffic
NW=$(docker exec "$CN" sh -c 'grep -l "\"kind\": \"init\"" /tmp/mapguard/mg_p*.jsonl 2>/dev/null | wc -l')
NS=$(docker exec "$CN" sh -c 'grep -l "installed_scheduler" /tmp/mapguard/ms_p*.jsonl 2>/dev/null | wc -l')
NE=$(docker exec "$CN" sh -c 'grep -h "\"guard_error\"" /tmp/mapguard/*.jsonl 2>/dev/null | wc -l')
docker exec "$CN" sh -c 'grep -h "\"kind\": \"init\"" /tmp/mapguard/mg_p*.jsonl' 2>&1 | cut -c1-600 | tee -a "$OUT/gate.txt"
echo "guard: worker ranks $NW, scheduler side $NS, guard errors $NE" | tee -a "$OUT/gate.txt"
if [ "${NW:-0}" -lt 2 ] || [ "${NS:-0}" -lt 1 ] || [ "${NE:-0}" -gt 0 ]; then
  echo "INVALID: guard not installed on both ranks and the scheduler, or guard errors at start-up"; RC_MAIN=8; kill $WD 2>/dev/null; exit 8
fi
DEADLINE=$((T0 + WALL_CAP - RESERVE))
echo "== client (deadline in $((DEADLINE - $(date +%s)))s)"
timeout $((DEADLINE - $(date +%s) + 15)) python3 "$HERE/client_mapguard.py" "$OUT" "$DEADLINE" 2>&1 | tee -a "$OUT/client.log"
RC_MAIN=${PIPESTATUS[0]}
echo "client rc=$RC_MAIN"
kill $WD 2>/dev/null
exit 0
