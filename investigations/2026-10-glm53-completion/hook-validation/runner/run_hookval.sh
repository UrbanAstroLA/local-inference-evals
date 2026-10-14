#!/bin/bash
# Hook-validation GPU session (PLAN.md amendment 21 section 4; hook-validation/VALIDATION.md). PREPARED ON CPU;
# NOT RUN. Launch only after the operator authorizes the item "hookval" with a GPU-hour total:
#   APPROVED_GPU_H=2.4 run_hookval.sh [ATTEMPT]          (DRY_RUN=1: budget gate, attempt count, hashes, image pin)
# Budget: item "hookval", total <= APPROVED_GPU_H (proposed 2.4) summed over every cost line of the item, invalid
# attempts included; at most 2 attempts. Per attempt WALL_CAP 3,240 s x 2 GPUs = 1.80 GPU-h; refused if the
# remaining budget cannot fit MIN_WALL (start-up + V0 + V2a + V2c + V4 essentials).
# One server: B1 + inert mapguard + forcewin hook (local/glm53-v0.9.1-b1-forcehook-mg:local, pinned id),
# recipe glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234+d7 with IMAGE overridden, shipped graph mode,
# DFlash2 x3, both ranks; hook ON (FORCEWIN=on, D2 injection allowed for keyed requests, shadow ledger and
# snapshots on), guard ON (GLM53_MAPGUARD=device). Workload: client_hookval.py (existing exact-id fixtures only).
# Unattended-safe: budget gate, image-id pin, frozen-hash check, pre-flight (no glm53* container, GPUs idle),
# hook + guard installed on both ranks before traffic, watchdog at the wall cap, stop on every exit path
# (EXIT/INT/TERM trap), docker cp of hook + guard outputs, one cost line to the moment the GPUs are released, then
# the frozen CPU analysis. No `set -e`. <stacks> is only read (start/stop scripts), never modified.
#
# ATTEMPT-2 VARIANT (PA2-10; attempt-1 runner kept as run_hookval.sh.a1; ATTEMPT2-PREP.md). Differences:
#   image local/glm53-v0.9.1-b1-forcehook-mg2:local (pinned in images.txt); ATTEMPT must be 2 and the cost
#   ledger must hold exactly the one attempt-1 line of the item (2 of 2 = the last attempt under amendment 21c);
#   the budget gate is cumulative over the item (2.4 GPU-h approved; attempt 1 spent 1.2117), so the wall cap is
#   min(WALL_CAP, remaining x 3600 / 2 GPUs) = 2,138 s; MIN_WALL 1,500 s (scoped plan estimate 1,260 s occupied
#   without the optional poison phase, runner/cost_a2.json); the client is the scoped attempt-2 variant.
ATTEMPT=${1:-2}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
COSTS=$ROOT/costs.jsonl
ITEM=hookval
MAX_ATTEMPTS=2
ST=<stacks>/glm53-flash-v0.9.0
RC=<recipes>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234+d7
CN=glm53-flash-exl3-b12x-vllm
IMG=local/glm53-v0.9.1-b1-forcehook-mg2:local
GPUS=2
CEILING=2.4                     # authorized item total (PLAN.md amendment 21c), both attempts together
WALL_CAP=${WALL_CAP:-3240}      # capped below by the remaining item budget (2,138 s after attempt 1)
MIN_WALL=${MIN_WALL:-1500}      # attempt 2: scoped plan, 1,260 s occupied estimated (runner/cost_a2.json)
REQ_ATTEMPT=2                   # attempt 2 of 2
RESERVE=${RESERVE:-180}         # seconds after the client deadline for readouts, copies and the stop
GATE_CAP=${GATE_CAP:-1080}
DEBUG="FORCEWIN=on FORCEWIN_D2=1 FORCEWIN_SHADOW=1 FORCEWIN_SNAP=1 FORCEWIN_DIR=/tmp/forcewin FORCEWIN_WMAX=5120 GLM53_MAPGUARD=device GLM53_MAPGUARD_DIR=/tmp/mapguard GLM53_MAPGUARD_POLL_SECS=20"
OUT=$ROOT/run-a$ATTEMPT
if [ "$ATTEMPT" != "$REQ_ATTEMPT" ]; then echo "REFUSED: this runner variant is attempt $REQ_ATTEMPT only (got '$ATTEMPT')"; exit 2; fi
if [ -e "$OUT" ]; then echo "ABORT: $OUT exists (use another ATTEMPT)"; exit 2; fi
touch "$COSTS" || exit 2

# ---- budget gate
if [ -z "${APPROVED_GPU_H:-}" ]; then echo "REFUSED: set APPROVED_GPU_H (authorized item total, at most $CEILING)"; exit 2; fi
if python3 -c "import sys; sys.exit(0 if float('$APPROVED_GPU_H') <= $CEILING else 1)"; then :; else
  echo "REFUSED: APPROVED_GPU_H=$APPROVED_GPU_H exceeds the proposed item ceiling $CEILING"; exit 2; fi
read -r SPENT NATT < <(python3 -c "
import json
s=0.0; n=0
for l in open('$COSTS'):
    l=l.strip()
    if l:
        d=json.loads(l)
        if d.get('item')=='$ITEM': s+=float(d.get('gpu_hours',0)); n+=1
print(round(s,4), n)")
if [ "$NATT" -ge "$MAX_ATTEMPTS" ]; then echo "REFUSED: $NATT attempts already recorded for $ITEM (max $MAX_ATTEMPTS)"; exit 2; fi
if [ "$NATT" -ne $((ATTEMPT - 1)) ]; then echo "REFUSED: cost ledger holds $NATT $ITEM attempt(s); attempt $ATTEMPT needs exactly $((ATTEMPT - 1))"; exit 2; fi
LEFT_S=$(python3 -c "print(int(($APPROVED_GPU_H - $SPENT) * 3600 / $GPUS))")
if [ "$LEFT_S" -lt "$WALL_CAP" ]; then WALL_CAP=$LEFT_S; fi
if [ "$WALL_CAP" -lt "$MIN_WALL" ]; then
  echo "REFUSED: spent ${SPENT} of ${APPROVED_GPU_H} GPU-h leaves ${WALL_CAP}s wall < ${MIN_WALL}s needed"; exit 2
fi
CAP_GH=$(python3 -c "print(round($GPUS*$WALL_CAP/3600,3))")
mkdir -p "$OUT" && cd "$OUT" || exit 2
echo "budget: spent $SPENT + this attempt's cap $CAP_GH (wall ${WALL_CAP}s) <= approved $APPROVED_GPU_H GPU-h" | tee "$OUT/budget.txt"

# ---- pins
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "ABORT: image $IMG missing"; exit 2; }
IMG_ID=$(docker image inspect "$IMG" --format '{{.Id}}')
EXP=$(awk -v t="$IMG" '$1==t {print $2}' "$HERE/images.txt")
echo "$IMG $IMG_ID" > "$OUT/image-id.txt"
if [ -z "$EXP" ] || [ "$IMG_ID" != "$EXP" ]; then echo "ABORT: $IMG is $IMG_ID, expected '$EXP'"; exit 2; fi
( cd "$ROOT" && sha256sum -c --quiet SHA256SUMS ) > "$OUT/sha256-check.txt" 2>&1 || { echo "ABORT: frozen hashes differ"; cat "$OUT/sha256-check.txt"; exit 2; }
[ -f <runs>/bundle/gate/gpqa_fixture.LOCAL.json ] || { echo "ABORT: no GPQA probe fixture"; exit 2; }
python3 "$HERE/client_hookval.py" "$OUT/plan" 0 --plan > "$OUT/plan.log" 2>&1 || { echo "ABORT: client plan failed"; tail "$OUT/plan.log"; exit 2; }
if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY_RUN: budget gate, attempt count, frozen hashes, image pin and client plan passed"; rm -rf "$OUT"; exit 0; fi

# ---- pre-flight (Live-tab equivalent; nvidia-smi --query-compute-apps misses graphics clients, so the process
#      table is read too)
echo "== preflight $(date -u +%FT%TZ)" | tee "$OUT/preflight.txt"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a "$OUT/preflight.txt"
nvidia-smi | tail -n 15 >> "$OUT/preflight.txt"
docker ps --format '{{.Names}} {{.Image}} {{.Status}}' | tee -a "$OUT/preflight.txt"
if docker ps --format '{{.Names}}' | grep -qE '^glm53|b0cache|mgtest'; then echo "ABORT: inference/fixture container running"; exit 3; fi
PROCS=$(nvidia-smi 2>/dev/null | awk '/Processes:/{p=1;next} p && /^\|/ && $2 ~ /^[0-9]+$/ {print}' \
  | grep -vE '[[:space:]]G[[:space:]].*(gnome-shell|Xwayland|xdg-desktop-portal|firefox|snap-store)' \
  | grep -vE '[[:space:]]C\+G[[:space:]].*--disable-logging')
if [ -n "$PROCS" ]; then echo "ABORT: GPU clients other than desktop type-G graphics:"; echo "$PROCS"; exit 3; fi
while read -r IDX MEM UTIL; do
  LIM=1024; [ "$IDX" = 1 ] && LIM=4096
  if [ "$MEM" -gt "$LIM" ] || [ "$UTIL" -gt 5 ]; then echo "ABORT: GPU$IDX busy mem=${MEM}MiB util=${UTIL}%"; exit 3; fi
done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')

START=$(date -u +%Y-%m-%dT%H:%M:%SZ); T0=$(date +%s); TEND=""; RC_MAIN=""; DONE=0
cost_line() {
  local T1=${TEND:-$(date +%s)}
  local SECS=$((T1 - T0))
  local GH=$(python3 -c "print(round($GPUS*$SECS/3600,4))")
  printf '{"item":"%s","start":"%s","end":"%s","seconds":%s,"gpus":%s,"gpu_hours":%s,"attempt":%s,"image":"%s","rc":%s,"wall_cap":%s,"purpose":"forced-history hook validation attempt 2 of 2, scoped (PLAN.md amendment 21 section 4; ATTEMPT2-PREP.md)"}\n' \
    "$ITEM" "$START" "$(date -u -d @"$T1" +%Y-%m-%dT%H:%M:%SZ)" "$SECS" "$GPUS" "$GH" "$ATTEMPT" "$IMG_ID" "${RC_MAIN:-null}" "$WALL_CAP" >> "$COSTS"
  echo "occupied ${SECS}s x $GPUS GPU = ${GH} GPU-h"
}
cleanup() {
  [ "$DONE" = 1 ] && return
  DONE=1
  [ -n "${WD:-}" ] && kill "$WD" 2>/dev/null
  echo "== cleanup $(date -u +%FT%TZ)"
  if [ "$(docker inspect --format '{{.State.Running}}' "$CN" 2>/dev/null)" = true ]; then
    docker exec "$CN" sh -c 'mkdir -p /tmp/mapguard && touch /tmp/mapguard/DUMP' >/dev/null 2>&1
    sleep 8
    docker logs "$CN" > "$OUT/server.log" 2>&1
    docker exec "$CN" sh -c 'cd /usr/local/lib/python3.12/dist-packages/vllm && sha256sum forcewin_hooks.py mapguard_hooks.py model_executor/layers/sparse_attn_indexer_kpool.py v1/worker/gpu/model_runner.py v1/worker/gpu/spec_decode/rejection_sampler.py v1/worker/gpu/sample/sampler.py v1/attention/backends/mla/indexer.py; env | grep -E "^(GLM53|VLLM|FORCEWIN)" | sort' > "$OUT/container-files-env.txt" 2>&1
  fi
  ( cd "$ST" && ./stop.sh ); echo "stop rc=$?"
  if docker ps --format '{{.Names}}' | grep -q "^$CN\$"; then docker stop -t 30 "$CN"; echo "docker stop rc=$?"; fi
  docker ps --format '{{.Names}}' | grep '^glm53' && echo "WARNING: glm53* container still running" || echo "no glm53* container running"
  TEND=$(date +%s)                                 # GPUs released; everything below is CPU-only
  cost_line
  mkdir -p "$OUT/forcewin" "$OUT/mapguard"
  docker cp "$CN:/tmp/forcewin/." "$OUT/forcewin/" >/dev/null 2>&1; echo "docker cp forcewin rc=$?"
  docker cp "$CN:/tmp/mapguard/." "$OUT/mapguard/" >/dev/null 2>&1; echo "docker cp mapguard rc=$?"
  [ -s "$OUT/server.log" ] || docker logs "$CN" > "$OUT/server.log" 2>&1
  # frozen analysis on CPU (torch for the .pt records): a CPU container of B1, no network, no GPU
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v "$OUT":/run -v "$ROOT/analysis":/a:ro \
    --entrypoint python3 sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a \
    /a/analyze_hookval.py /run 2>&1 | grep -v "cpp_extension" | tee "$OUT/verdict-console.txt"
}
trap cleanup EXIT
trap 'echo "signal"; RC_MAIN=130; exit 4' INT TERM
( sleep $(( WALL_CAP > 300 ? WALL_CAP - 75 : WALL_CAP )); echo "WALL CAP reached"; docker stop -t 30 "$CN" >/dev/null 2>&1 ) &
WD=$!

echo "== start $START image=$IMG ($IMG_ID) cap ${WALL_CAP}s"
( unset EXTRA_SERVE_ARGS; cd "$ST" && source "$RC/env.sh" && export SPECULATIVE_METHOD=dflash2 DFLASH_TOKENS=3 \
    IMAGE="$IMG" DEBUG_ENV="$DEBUG" && ./start.sh ) > "$OUT/start.log" 2>&1
SRC=$?
cat "$OUT/start.log"
if [ $SRC -ne 0 ]; then echo "start failed rc=$SRC"; RC_MAIN=5; exit 5; fi
echo "== readiness gate (cap ${GATE_CAP}s)"
READY=0
while [ $(( $(date +%s) - T0 )) -lt "$GATE_CAP" ]; do
  RUN=$(docker inspect --format '{{.State.Running}} {{.RestartCount}}' "$CN" 2>/dev/null)
  if [ "$RUN" != "true 0" ]; then echo "container state: '$RUN' -> abort"; RC_MAIN=6; exit 6; fi
  if docker logs "$CN" 2>&1 | grep -qE 'Engine core initialization failed|CUDA out of memory|Traceback'; then
    echo "engine error during start-up -> abort"; RC_MAIN=6; exit 6
  fi
  NE=$(docker exec "$CN" sh -c 'grep -h "\"fw_error\"\|\"guard_error\"" /tmp/forcewin/*.jsonl /tmp/mapguard/*.jsonl 2>/dev/null | wc -l' 2>/dev/null)
  if [ "${NE:-0}" -gt 0 ]; then echo "INVALID: hook/guard error during start-up"; RC_MAIN=8; exit 8; fi
  if docker exec "$CN" test -f /tmp/glm53-release-ready 2>/dev/null && curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null; then
    READY=1; break
  fi
  sleep 10
done
if [ $READY -ne 1 ]; then echo "readiness timeout"; RC_MAIN=7; exit 7; fi
echo "ready after $(( $(date +%s) - T0 ))s" | tee "$OUT/gate.txt"
docker logs "$CN" 2>&1 | grep -n "cudagraph_mode\|Graph capturing finished\|GPU KV cache size\|Using V2 Model Runner\|State rollback" | cut -c1-300 | tail -n 20 | tee -a "$OUT/gate.txt"
# both instrumentations registered on both worker ranks BEFORE graph capture finished (forcewin "registered" is
# logged from capture_model; a hook absent at capture would be missing from every replayed graph)
NF=$(docker exec "$CN" sh -c 'grep -l "\"registered\"" /tmp/forcewin/fw_r*.jsonl 2>/dev/null | wc -l')
NW=$(docker exec "$CN" sh -c 'grep -l "\"kind\": \"init\"" /tmp/mapguard/mg_p*.jsonl 2>/dev/null | wc -l')
echo "forcewin registered on $NF rank files; guard init on $NW" | tee -a "$OUT/gate.txt"
if [ "${NF:-0}" -lt 2 ] || [ "${NW:-0}" -lt 2 ]; then echo "INVALID: instrumentation not registered on both ranks"; RC_MAIN=8; exit 8; fi
DEADLINE=$((T0 + WALL_CAP - RESERVE))
echo "== client (deadline in $((DEADLINE - $(date +%s)))s)"
timeout $((DEADLINE - $(date +%s) + 15)) python3 "$HERE/client_hookval.py" "$OUT" "$DEADLINE" 2>&1 | tee -a "$OUT/client.log"
RC_MAIN=${PIPESTATUS[0]}
echo "client rc=$RC_MAIN"
exit 0
