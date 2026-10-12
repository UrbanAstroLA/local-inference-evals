#!/bin/bash
# Bundle gate (PLAN.md amendment 14 (3)). PREPARED, NOT RUN: every step needs operator authorization.
#   run_bundle_gate.sh s1 [BUNDLE] [ATTEMPT]  verification image, observers on (col511 checker, tail guard, verify13 capture,
#                                             D4 plan inspection, VLLM_DEBUG_WORKSPACE), limit 10: activation of D1-D4,
#                                             short contexts / mixed batches / request reuse, errors, throughput (lower bound)
#   run_bundle_gate.sh s2 [BUNDLE] [ATTEMPT]  the PRODUCTION bundle image (the screen's image), no observers: start-up smoke,
#                                             clean T/F and GPQA-shaped throughput, D4 probe at limit 10 (cross-session control).
#                                             CONDITIONAL: run only if s1's T/F throughput is below 95% of the release reference
#                                             or the operator wants the production image started before the screen.
#   run_bundle_gate.sh s3 [BUNDLE] [ATTEMPT]  verification image, observers off, GLM53_D4_LIMIT_OVERRIDE=1: D4 real-weight
#                                             reachability (probe vs s1 at limit 10)
#   run_bundle_gate.sh rel release [ATTEMPT] RELEASE CONTROL: stock release image, no observers/override, probe phase only
#   BUNDLE = d1234 (default; with D3) or d124 (without D3). Images are pinned by the ids in images.txt.
# Shipped graph mode (no EXTRA_SERVE_ARGS), recipe glm53-flash-exl3-k3.25@tpurtell+v0.9.1 with IMAGE overridden, DFlash2 x3.
# Unattended-safe: budget gate (logged GPU-h + this step's cap <= BUDGET, default 3.0), image-id check, pre-flight (no glm53*
# container, no non-desktop compute process, GPU0 <= 1 GiB, GPU1 <= 4 GiB, util <= 10%), EXIT/INT/TERM trap that stops the
# server on every exit path, one cost line per step in bundle/costs.jsonl counted to the moment the GPUs are released.
# No `set -e`: every status query is checked explicitly.
MODE=${1:-}
BUNDLE=${2:-d1234}
ATTEMPT=${3:-1}
G=$(cd "$(dirname "$0")" && pwd)
ROOT=<runs>/bundle
REQ=<runs>/phase2-stage2/requests.json
ST=<stacks>/glm53-flash-v0.9.0
RC=<recipes>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1
CN=glm53-flash-exl3-b12x-vllm
BUDGET=${BUDGET:-3.0}
case "$BUNDLE" in d1234|d124|release) ;; *) echo "BUNDLE must be d1234, d124 (or release with mode rel)"; exit 2 ;; esac
REL_IMG=ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
CHECK_IMG=local/glm53-v0.9.1-bundle-$BUNDLE-check:local
PROD_IMG=local/glm53-v0.9.1-bundle-$BUNDLE:local
case "$MODE" in
  s1) [ "$BUNDLE" != release ] || { echo "release only with mode rel"; exit 2; }; IMG=$CHECK_IMG; WALL_CAP=${WALL_CAP:-2400}; GATE_CAP=${GATE_CAP:-1500}
      PHASES="warm TF DS Q G P"; REPEATS=${REPEATS:-4}
      DEBUG="GLM53_COL511CHECK=device GLM53_COL511CHECK_DIR=/tmp/col511check GLM53_COL511CHECK_POLL_SECS=30 GLM53_TAILGUARD=device GLM53_TAILGUARD_DIR=/tmp/tailguard GLM53_TAILGUARD_POLL_SECS=20 GLM53_VERIFY13=1 GLM53_VERIFY13_DIR=/tmp/verify13 GLM53_VERIFY13_EVERY=4 GLM53_VERIFY13_STRIDE=21 GLM53_VERIFY13_OUT=0 GLM53_VERIFY13_MAX_MB=1024 GLM53_VERIFY13_SLABS=96 GLM53_D4_INSPECT=1 GLM53_D4_LIMIT_DIR=/tmp/d4limit VLLM_DEBUG_WORKSPACE=1" ;;
  s2) [ "$BUNDLE" != release ] || { echo "release only with mode rel"; exit 2; }; IMG=$PROD_IMG; WALL_CAP=${WALL_CAP:-2000}; GATE_CAP=${GATE_CAP:-1200}
      PHASES="warm TF G P"; REPEATS=${REPEATS:-8}; DEBUG="" ;;
  s3) [ "$BUNDLE" != release ] || { echo "release only with mode rel"; exit 2; }; IMG=$CHECK_IMG; WALL_CAP=${WALL_CAP:-1800}; GATE_CAP=${GATE_CAP:-1500}
      PHASES="P"; REPEATS=0
      DEBUG="GLM53_D4_LIMIT_OVERRIDE=1 GLM53_D4_INSPECT=1 GLM53_D4_LIMIT_DIR=/tmp/d4limit" ;;
  rel) # release control (added 2026-10-12 after the gate's P1 failure): the s3 client sequence (probe only) on the
      # STOCK release image, no observers, no override; answers whether long-context greedy non-repeatability is pre-existing
      [ "$BUNDLE" = release ] || { echo "mode rel requires BUNDLE=release"; exit 2; }
      IMG=$REL_IMG; WALL_CAP=${WALL_CAP:-1350}; GATE_CAP=${GATE_CAP:-1100}
      PHASES="P"; REPEATS=0; DEBUG="" ;;
  *) echo "usage: $0 s1|s2|s3 [d1234|d124] [ATTEMPT]  |  $0 rel release [ATTEMPT]"; exit 2 ;;
esac
GPUS=2
CAP_GH=$(python3 -c "print(round($GPUS*$WALL_CAP/3600,3))")
OUT=$ROOT/gate-runs/$MODE-$BUNDLE-a$ATTEMPT
if [ -e "$OUT" ]; then echo "ABORT: $OUT exists (use another ATTEMPT)"; exit 2; fi
mkdir -p "$OUT" || exit 2
cd "$OUT" || exit 2
# ---- budget gate (logged GPU-h of every bundle step, invalid runs included, plus this step's cap)
USED=$(python3 -c "
import json,os
p='$ROOT/costs.jsonl'; s=0.0
if os.path.exists(p):
    for l in open(p):
        try: s+=float(json.loads(l).get('gpu_hours',0))
        except Exception: pass
print(round(s,3))")
if python3 -c "import sys; sys.exit(0 if $USED + $CAP_GH <= $BUDGET + 1e-9 else 1)"; then
  echo "budget: used $USED + cap $CAP_GH <= $BUDGET GPU-h" | tee "$OUT/budget.txt"
else
  echo "ABORT: budget: used $USED + cap $CAP_GH > $BUDGET GPU-h (lower WALL_CAP or ask the operator)" | tee "$OUT/budget.txt"; exit 2
fi
# ---- image identity (ids recorded at build time; a rebuilt image is refused)
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "ABORT: image $IMG missing"; exit 2; }
IMG_ID=$(docker image inspect "$IMG" --format '{{.Id}}')
EXP=$(awk -v t="$IMG" '$1==t {print $2}' "$G/images.txt")
echo "$IMG $IMG_ID" > "$OUT/image-id.txt"
if [ -z "$EXP" ] || [ "$IMG_ID" != "$EXP" ]; then echo "ABORT: $IMG is $IMG_ID, expected '$EXP' (images.txt)"; exit 2; fi
cp "$REQ" "$OUT/requests.json" || { echo "ABORT: no requests.json"; exit 2; }
[ -f "$G/gpqa_fixture.LOCAL.json" ] || { echo "ABORT: no gpqa_fixture.LOCAL.json (run make_gpqa_fixture.py)"; exit 2; }
# ---- pre-flight
echo "== preflight $(date -u +%FT%TZ) mode=$MODE bundle=$BUNDLE" | tee "$OUT/preflight.txt"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a "$OUT/preflight.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv | tee -a "$OUT/preflight.txt"
nvidia-smi | tail -n 15 >> "$OUT/preflight.txt"   # also lists graphics clients (browsers, Electron)
docker ps --format '{{.Names}} {{.Image}} {{.Status}}' | tee -a "$OUT/preflight.txt"
if docker ps --format '{{.Names}}' | grep -q '^glm53'; then echo "ABORT: glm53* container running"; exit 3; fi
NONDESK=""
for P in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  if tr '\0' ' ' < "/proc/$P/cmdline" 2>/dev/null | grep -q -- '--type=gpu-process'; then
    echo "preflight: ignoring desktop GPU process pid $P" | tee -a "$OUT/preflight.txt"
  else
    NONDESK="$NONDESK $P"
  fi
done
if [ -n "$NONDESK" ]; then echo "ABORT: compute processes on the GPUs:$NONDESK"; exit 3; fi
while read -r IDX MEM UTIL; do
  LIM=1024; [ "$IDX" = 1 ] && LIM=4096        # desktop graphics on GPU 1 accepted (amendment 11b)
  if [ "$MEM" -gt "$LIM" ] || [ "$UTIL" -gt 10 ]; then echo "ABORT: GPU$IDX busy mem=${MEM}MiB util=${UTIL}%"; exit 3; fi
done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')

START=$(date -u +%Y-%m-%dT%H:%M:%SZ); T0=$(date +%s); TEND=""
DONE_CLEANUP=0
RC_MAIN=""
cost_line() {
  local T1=${TEND:-$(date +%s)}
  local SECS=$((T1 - T0))
  local GH=$(python3 -c "print(round($GPUS*$SECS/3600,4))")
  printf '{"start":"%s","end":"%s","seconds":%s,"gpus":%s,"gpu_hours":%s,"mode":"%s","bundle":"%s","attempt":%s,"image":"%s","rc":%s,"purpose":"bundle gate %s (PLAN.md amendment 14 (3))"}\n' \
    "$START" "$(date -u -d @"$T1" +%Y-%m-%dT%H:%M:%SZ)" "$SECS" "$GPUS" "$GH" "$MODE" "$BUNDLE" "$ATTEMPT" "$IMG_ID" "${RC_MAIN:-null}" "$MODE" >> "$ROOT/costs.jsonl"
  echo "occupied ${SECS}s x $GPUS GPU = ${GH} GPU-h"
}
snapshot() {
  # observer flush triggers, then logs and identity evidence while the container still runs
  docker exec "$CN" sh -c 'mkdir -p /tmp/col511check /tmp/tailguard /tmp/verify13; touch /tmp/col511check/DUMP_RING /tmp/tailguard/DUMP /tmp/verify13/DISARM' >/dev/null 2>&1
  for i in $(seq 1 45); do
    if [ "$MODE" != s1 ]; then break; fi
    N=$(docker exec "$CN" sh -c 'ls /tmp/verify13/FLUSHED_r* 2>/dev/null | wc -l' 2>/dev/null)
    [ "${N:-0}" -ge 2 ] && break
    sleep 2
  done
  sleep 8   # checker / guard poll threads check their triggers every 2 s
  docker logs "$CN" > "$OUT/server.log" 2>&1
  docker exec "$CN" sh -c 'cd /usr/local/lib/python3.12/dist-packages/vllm && sha256sum model_executor/layers/sparse_attn_indexer_kpool.py v1/attention/backends/mla/indexer.py v1/worker/gpu/model_states/mamba_hybrid.py model_executor/layers/quantization/exl3.py model_executor/layers/fused_moe/router/gate_linear.py local_*.py 2>/dev/null; env | grep -E "^(GLM53|VLLM|B12X)" | sort' > "$OUT/container-files-env.txt" 2>&1
  docker inspect "$CN" --format '{{.Image}} {{json .Config.Env}} {{json .Args}}' > "$OUT/container-inspect.txt" 2>&1
}
cleanup() {
  [ "$DONE_CLEANUP" = 1 ] && return
  DONE_CLEANUP=1
  echo "== cleanup $(date -u +%FT%TZ)"
  if docker inspect "$CN" >/dev/null 2>&1 && [ "$(docker inspect --format '{{.State.Running}}' "$CN" 2>/dev/null)" = true ]; then
    snapshot
  fi
  ( cd "$ST" && ./stop.sh ); echo "stop rc=$?"
  if docker ps --format '{{.Names}}' | grep -q "^$CN\$"; then
    echo "stop.sh left $CN running; docker stop"; docker stop -t 30 "$CN"; echo "docker stop rc=$?"
  fi
  docker ps --format '{{.Names}}' | grep '^glm53' && echo "WARNING: glm53* container still running" || echo "no glm53* container running"
  TEND=$(date +%s)                     # GPUs released here; everything below uses no GPU
  cost_line
  # observer output from the retained stopped container (stop.sh keeps it), then CPU-only analysis
  for d in col511check tailguard verify13 d4limit; do
    mkdir -p "$OUT/$d"; docker cp "$CN:/tmp/$d/." "$OUT/$d/" >/dev/null 2>&1; echo "copied $d rc=$?"
  done
  [ -s "$OUT/server.log" ] || docker logs "$CN" > "$OUT/server.log" 2>&1
  python3 "$G/analyze_bundle_gate.py" "$OUT" "$MODE" "$BUNDLE" 2>&1 | tee "$OUT/verdict.txt"
}
trap cleanup EXIT
trap 'echo "signal"; exit 4' INT TERM

echo "== start $START mode=$MODE bundle=$BUNDLE image=$IMG ($IMG_ID)"
( unset EXTRA_SERVE_ARGS; cd "$ST" && source "$RC/env.sh" && export SPECULATIVE_METHOD=dflash2 DFLASH_TOKENS=3 IMAGE="$IMG" \
    DEBUG_ENV="$DEBUG" && ./start.sh ) > "$OUT/start.log" 2>&1
SRC=$?
cat "$OUT/start.log"
if [ $SRC -ne 0 ]; then echo "start failed rc=$SRC"; RC_MAIN=5; exit 5; fi

echo "== readiness gate (cap ${GATE_CAP}s; release warm-up marker + API)"
READY=0
while [ $(( $(date +%s) - T0 )) -lt "$GATE_CAP" ]; do
  RUN=$(docker inspect --format '{{.State.Running}} {{.RestartCount}}' "$CN" 2>/dev/null)
  if [ "$RUN" != "true 0" ]; then echo "container state: '$RUN' -> abort"; RC_MAIN=6; exit 6; fi
  if docker logs "$CN" 2>&1 | grep -qE 'Engine core initialization failed|CUDA out of memory|Workspace is locked|Traceback'; then
    echo "engine error during start-up -> abort"; RC_MAIN=6; exit 6
  fi
  if docker exec "$CN" test -f /tmp/glm53-release-ready 2>/dev/null && curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null; then
    READY=1; break
  fi
  sleep 10
done
if [ $READY -ne 1 ]; then echo "readiness timeout"; RC_MAIN=7; exit 7; fi
TR=$(date +%s); echo "ready after $((TR - T0))s" | tee "$OUT/gate.txt"
docker logs "$CN" 2>&1 | grep -n "cudagraph_mode\|Graph capturing finished\|BREAKABLE\|WORKSPACE DEBUG\|EXL3 rank-sliced runtime planned\|GPU KV cache size" | cut -c1-300 | tail -n 30 | tee -a "$OUT/gate.txt"
if [ "$MODE" = s1 ]; then
  # all observers installed on both ranks before any client traffic; otherwise the run is invalid (stop now)
  NC=$(docker exec "$CN" sh -c 'grep -l "\"installed\"" /tmp/col511check/*.jsonl 2>/dev/null | wc -l')
  NT=$(docker exec "$CN" sh -c 'grep -l "\"installed\"" /tmp/tailguard/tg_p*.jsonl 2>/dev/null | wc -l')
  NV=$(docker exec "$CN" sh -c 'grep -l "\"kind\": \"installed\"" /tmp/verify13/v13_p*.jsonl 2>/dev/null | wc -l')
  ND=$(docker exec "$CN" sh -c 'grep -l "\"installed\"" /tmp/d4limit/*.jsonl 2>/dev/null | wc -l')
  echo "observers installed: col511check $NC, tailguard $NT, verify13 $NV, d4limit $ND (files; >= 2 ranks each)" | tee -a "$OUT/gate.txt"
  if [ "${NC:-0}" -lt 2 ] || [ "${NT:-0}" -lt 2 ] || [ "${NV:-0}" -lt 2 ] || [ "${ND:-0}" -lt 2 ]; then
    echo "INVALID: an observer is not installed on both ranks"; RC_MAIN=8; exit 8
  fi
  docker exec "$CN" sh -c 'touch /tmp/verify13/ARM'
fi
if [ "$MODE" = s3 ]; then
  ND=$(docker exec "$CN" sh -c 'grep -h "override_applied" /tmp/d4limit/*.jsonl 2>/dev/null | wc -l')
  docker exec "$CN" sh -c 'cat /tmp/d4limit/*.jsonl' 2>&1 | cut -c1-300 | tee -a "$OUT/gate.txt"
  if [ "${ND:-0}" -lt 2 ]; then echo "INVALID: limit override not applied on both ranks"; RC_MAIN=8; exit 8; fi
fi
DEADLINE=$((T0 + WALL_CAP - 180))
echo "== client phases '$PHASES' (deadline in $((DEADLINE - $(date +%s)))s)"
PHASES="$PHASES" REPEATS=$REPEATS timeout $((DEADLINE - $(date +%s) + 60)) python3 "$G/client_bundle.py" "$OUT" "$DEADLINE" 2>&1 | tee -a "$OUT/client.log"
RC_MAIN=${PIPESTATUS[0]}
echo "client rc=$RC_MAIN"
exit 0
