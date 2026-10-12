#!/bin/bash
# PLAN.md amendment 13 / 13a Stage A (only): D3 router precision, D4 omission + MoE input freeze, D5 workspace.
# One script, three steps, run in this order (each needs operator authorization; none run yet):
#   run_verify13.sh smoke   [ATTEMPT]   1 GPU (GPU0), ~2-6 min: hook host-sync / graph-capture smoke and
#                                       B12x's in-tree projection-mixed swiglu_limit kernel tests
#   run_verify13.sh server  [ATTEMPT]   2 GPUs, ~22 min: one server from the verify13 image, SHIPPED graph
#                                       mode (no EXTRA_SERVE_ARGS), capture armed after readiness, the
#                                       col511-gate fixture (client_v13.py), VLLM_DEBUG_WORKSPACE=1
#   run_verify13.sh offline [ATTEMPT]   1 GPU (GPU0), ~5 min: D3 comparison on the captured inputs, D4
#                                       inventory, D5 analysis
# Unattended-safe: pre-flight aborts if any glm53* container runs or a GPU is busy; the server/containers
# are stopped on every exit path (EXIT/INT/TERM trap); hard wall caps; one cost line per step appended to
# $ROOT/costs.jsonl; a step refuses to start if the cumulative logged GPU-hours plus its cap exceed
# BUDGET (2.0 occupied GPU-hours, amendment 13).
# No `set -e`: every status query is checked explicitly.
MODE=${1:-}
ATTEMPT=${2:-1}
V=$(cd "$(dirname "$0")" && pwd)
ROOT=<runs>/verify13
REQ=<runs>/phase2-stage2/requests.json
ST=<stack>
RC=<recipe>
MREPO=<models>   # snapshot files are symlinks into blobs/
MREV=0490d2f708b12145f6516555ab066aaeb401cd21
CN=glm53-flash-exl3-b12x-vllm
IMG=local/glm53-v0.9.1-verify313:local
IMG_ID_EXPECTED=$(cat "$V/image-id.txt" 2>/dev/null)
BUDGET=${BUDGET:-2.0}
case "$MODE" in
  smoke)   GPUS=1; WALL_CAP=${WALL_CAP:-900} ;;
  server)  GPUS=2; WALL_CAP=${WALL_CAP:-1800}; GATE_CAP=${GATE_CAP:-1200}; REPEATS=${REPEATS:-6} ;;
  offline) GPUS=1; WALL_CAP=${WALL_CAP:-1200} ;;
  *) echo "usage: $0 smoke|server|offline [ATTEMPT]"; exit 2 ;;
esac
CAP_GH=$(python3 -c "print(round($GPUS*$WALL_CAP/3600,3))")
OUT=$ROOT/$MODE-a$ATTEMPT
if [ -e "$OUT" ]; then echo "ABORT: $OUT exists (use another ATTEMPT)"; exit 2; fi
mkdir -p "$OUT" || exit 2
cd "$OUT" || exit 2

# ---- budget gate (amendment 13: <= 2.0 occupied GPU-hours, invalid runs included)
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

# ---- image identity
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "ABORT: image $IMG missing"; exit 2; }
IMG_ID=$(docker image inspect "$IMG" --format '{{.Id}}')
echo "$IMG_ID" > "$OUT/image-id.txt"
if [ -n "$IMG_ID_EXPECTED" ] && [ "$IMG_ID" != "$IMG_ID_EXPECTED" ]; then
  echo "ABORT: $IMG is $IMG_ID, expected $IMG_ID_EXPECTED (rebuilt?)"; exit 2
fi

# ---- pre-flight
echo "== preflight $(date -u +%FT%TZ) mode=$MODE" | tee "$OUT/preflight.txt"
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
OCN="verify13-$MODE-a$ATTEMPT"               # name of the one-GPU containers (smoke/offline)
cost_line() {
  local T1=${TEND:-$(date +%s)}
  local SECS=$((T1 - T0))
  local GH=$(python3 -c "print(round($GPUS*$SECS/3600,4))")
  printf '{"start":"%s","end":"%s","seconds":%s,"gpus":%s,"gpu_hours":%s,"mode":"%s","attempt":%s,"image":"%s","rc":%s,"purpose":"verify13 %s (PLAN.md amendment 13): %s"}\n' \
    "$START" "$(date -u -d @"$T1" +%Y-%m-%dT%H:%M:%SZ)" "$SECS" "$GPUS" "$GH" "$MODE" "$ATTEMPT" "$IMG_ID" "${RC_MAIN:-null}" "$MODE" "$1" >> "$ROOT/costs.jsonl"
  echo "occupied ${SECS}s x $GPUS GPU = ${GH} GPU-h"
}

# ======================================================================= smoke / offline (1 GPU)
if [ "$MODE" = smoke ] || [ "$MODE" = offline ]; then
  cleanup1() {
    [ "$DONE_CLEANUP" = 1 ] && return
    DONE_CLEANUP=1
    docker rm -f "$OCN" >/dev/null 2>&1
    TEND=$(date +%s)
    if [ "$MODE" = smoke ]; then cost_line "hook host-sync/graph smoke + B12x swiglu_limit kernel tests, GPU0"
    else cost_line "D3 offline comparison + D4 inventory + D5 analysis, GPU0"; fi
  }
  trap cleanup1 EXIT
  trap 'echo "signal"; exit 4' INT TERM
  if [ "$MODE" = smoke ]; then
    timeout "$WALL_CAP" docker run --rm --name "$OCN" --gpus '"device=0"' --network none --ipc host \
      -e GLM53_VERIFY13=1 -e GLM53_VERIFY13_STRIDE=1 -e GLM53_VERIFY13_DIR=/out/hooksmoke \
      -v "$OUT:/out" --entrypoint bash "$IMG" -c '
        set -o pipefail
        cd /opt/verify13 && python3 smoke_gpu.py 2>&1 | tee /out/smoke_gpu.log; R1=${PIPESTATUS[0]}
        cd /opt/b12x && timeout 700 python3 -m pytest -q -p no:cacheprovider \
          tests/moe/test_w4a16_mixed_trellis.py -k "swiglu_limit_matches_clipped_serial" 2>&1 | tail -n 40 | tee /out/b12x_swiglu_tests.log; R2=${PIPESTATUS[0]}
        echo "smoke_gpu rc=$R1 b12x_swiglu rc=$R2"; [ $R1 -eq 0 ] && [ $R2 -eq 0 ]' 2>&1 | tee "$OUT/run.log"
    RC_MAIN=${PIPESTATUS[0]}
  else
    SRV=${SRV:-$ROOT/server-a1}
    [ -d "$SRV/verify13" ] || { echo "ABORT: no $SRV/verify13 (server step not run?)"; RC_MAIN=2; exit 2; }
    timeout "$WALL_CAP" docker run --rm --name "$OCN" --gpus '"device=0"' --network none --ipc host \
      -v "$SRV:/capture:ro" -v "$MREPO:/model-repo:ro" -v "$OUT:/out" -e MREV="$MREV" --entrypoint bash "$IMG" -c '
        set -o pipefail
        cd /opt/verify13
        python3 d3_router_compare.py /capture /model-repo/snapshots/$MREV /out 2>&1 | tee /out/d3.log; R1=${PIPESTATUS[0]}
        python3 analyze_d5.py /capture > /out/d5_summary.json 2>/out/d5.err; R2=$?
        echo "d3 rc=$R1 d5 rc=$R2"; [ $R1 -eq 0 ] && [ $R2 -eq 0 ]' 2>&1 | tee "$OUT/run.log"
    RC_MAIN=${PIPESTATUS[0]}
  fi
  echo "rc=$RC_MAIN"
  exit "$RC_MAIN"
fi

# ======================================================================= server (2 GPUs)
cp "$REQ" "$OUT/requests.json" || { echo "ABORT: no requests.json"; exit 2; }
DEBUG="GLM53_VERIFY13=1 GLM53_VERIFY13_DIR=/tmp/verify13 GLM53_VERIFY13_EVERY=${V13_EVERY:-1} GLM53_VERIFY13_STRIDE=${V13_STRIDE:-21} GLM53_VERIFY13_FULL_ROWS=256 GLM53_VERIFY13_WIN_ROWS=64 GLM53_VERIFY13_MAX_MB=${V13_MAX_MB:-3072} GLM53_VERIFY13_SLABS=192 VLLM_DEBUG_WORKSPACE=1"
FLUSHED=0
flush_capture() {
  # disarm, wait for both ranks' writers to drain (FLUSHED_r0/_r1), snapshot logs
  docker exec "$CN" sh -c 'mkdir -p /tmp/verify13 && touch /tmp/verify13/DISARM' >/dev/null 2>&1
  for i in $(seq 1 45); do
    N=$(docker exec "$CN" sh -c 'ls /tmp/verify13/FLUSHED_r* 2>/dev/null | wc -l' 2>/dev/null)
    [ "${N:-0}" -ge 2 ] && { FLUSHED=1; break; }
    sleep 2
  done
  echo "flushed=$FLUSHED after $((i*2))s"
  docker logs "$CN" > "$OUT/server.log" 2>&1
  docker exec "$CN" sh -c 'du -sh /tmp/verify13; ls /tmp/verify13' > "$OUT/capture-dir.txt" 2>&1
  docker exec "$CN" sh -c 'sha256sum /usr/local/lib/python3.12/dist-packages/vllm/verify13_capture.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/router/gate_linear.py; env | grep -E "^(GLM53|VLLM|B12X)" | sort' > "$OUT/container-files-env.txt" 2>&1
  docker inspect "$CN" --format '{{.Image}} {{json .Config.Env}} {{json .Args}}' > "$OUT/container-inspect.txt" 2>&1
}
cleanup() {
  [ "$DONE_CLEANUP" = 1 ] && return
  DONE_CLEANUP=1
  echo "== cleanup $(date -u +%FT%TZ)"
  if docker inspect "$CN" >/dev/null 2>&1 && [ "$(docker inspect --format '{{.State.Running}}' "$CN" 2>/dev/null)" = true ]; then
    flush_capture
  fi
  ( cd "$ST" && ./stop.sh ); echo "stop rc=$?"
  if docker ps --format '{{.Names}}' | grep -q "^$CN\$"; then
    echo "stop.sh left $CN running; docker stop"; docker stop -t 30 "$CN"; echo "docker stop rc=$?"
  fi
  docker ps --format '{{.Names}}' | grep '^glm53' && echo "WARNING: glm53* container still running" || echo "no glm53* container running"
  TEND=$(date +%s)                     # GPUs released here; the copy below uses no GPU
  cost_line "one server, verify13 image (observe-only capture armed after readiness), shipped graph mode, DFlash2 x3, col511-gate fixture x$REPEATS, VLLM_DEBUG_WORKSPACE=1"
  # the stopped container is retained by stop.sh: copy the capture out after the GPUs are free
  mkdir -p "$OUT/verify13"
  docker cp "$CN:/tmp/verify13/." "$OUT/verify13/"; echo "docker cp rc=$?"
  du -sh "$OUT/verify13" 2>/dev/null
  # D4 serving-path provenance: the engine's own EXL3 log lines (exl3.py:727 / :2536 / :3118 in 0.9.1)
  grep -n "EXL3 standard fused MoE\|EXL3 projection-mixed Trellis\|EXL3 rank-sliced runtime planned\|EXL3 mixed Trellis runtime planned" \
    "$OUT/server.log" | cut -c1-400 > "$OUT/verify13/serving_path_log_lines.txt" 2>/dev/null; echo "serving-path lines: $(wc -l < "$OUT/verify13/serving_path_log_lines.txt" 2>/dev/null)"
  python3 "$V/offline/analyze_d5.py" "$OUT" > "$OUT/d5_summary.json" 2>"$OUT/d5.err"; echo "d5 analysis rc=$?"
  # effective configuration vs the historical clean 0.9.1 run and the col511 gate run (copies of
  # kdcp/gate/config_compare.py); decode throughput vs the gate run (copy of tailguard/throughput_tg.py)
  python3 "$V/offline/config_compare.py" "$OUT/server.log" <gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.server.log > "$OUT/config-compare-historical.txt" 2>&1
  python3 "$V/offline/config_compare.py" "$OUT/server.log" <runs>/gate/server-a1/server.log > "$OUT/config-compare-gate.txt" 2>&1
  python3 "$V/offline/throughput_tg.py" "$OUT" <runs>/gate/server-a1 > "$OUT/throughput.txt" 2>&1
  python3 - "$OUT" <<'PY' 2>&1 | tee "$OUT/summary.txt"
import glob, json, os, sys
out = sys.argv[1]
ev = [json.loads(l) for f in glob.glob(f"{out}/verify13/v13_p*.jsonl") for l in open(f)]
summ = [d for d in ev if d.get("kind") == "summary"]
for r in sorted({d.get("rank") for d in summ}, key=str):
    last = [d for d in summ if d.get("rank") == r][-1]
    print("rank", r, json.dumps(last.get("counters")), "bytes", last.get("bytes"), "errors", last.get("errors"))
for k in ("gate_info", "router_info", "experts_info"):
    xs = [d for d in ev if d.get("kind") == k and d.get("rank") == 0]
    if xs:
        print(k, json.dumps({kk: xs[0].get(kk) for kk in list(xs[0])[:24]}, default=str)[:900])
print("records r0:", len(glob.glob(f"{out}/verify13/rec_r0/*.pt")), "r1:", len(glob.glob(f"{out}/verify13/rec_r1/*.pt")))
PY
}
trap cleanup EXIT
trap 'echo "signal"; exit 4' INT TERM

echo "== start $START image=$IMG ($IMG_ID)"
( unset EXTRA_SERVE_ARGS; cd "$ST" && source "$RC/env.sh" && export SPECULATIVE_METHOD=dflash2 DFLASH_TOKENS=3 IMAGE="$IMG" \
    DEBUG_ENV="$DEBUG" && ./start.sh ) > "$OUT/start.log" 2>&1
SRC=$?
cat "$OUT/start.log"
if [ $SRC -ne 0 ]; then echo "start failed rc=$SRC"; RC_MAIN=5; exit 5; fi

echo "== gate (cap ${GATE_CAP}s)"
READY=0
while [ $(( $(date +%s) - T0 )) -lt "$GATE_CAP" ]; do
  RUN=$(docker inspect --format '{{.State.Running}} {{.RestartCount}}' "$CN" 2>/dev/null)
  if [ "$RUN" != "true 0" ]; then echo "container state: '$RUN' -> abort"; RC_MAIN=6; exit 6; fi
  if docker exec "$CN" test -f /tmp/glm53-release-ready 2>/dev/null && curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null; then
    READY=1; break
  fi
  sleep 10
done
if [ $READY -ne 1 ]; then echo "gate timeout"; RC_MAIN=7; exit 7; fi
TR=$(date +%s); echo "ready after $((TR - T0))s" | tee "$OUT/gate.txt"
docker logs "$CN" 2>&1 | grep -n "cudagraph_mode\|Capturing CUDA graphs\|Graph capturing finished\|BREAKABLE\|WORKSPACE DEBUG" | cut -c1-300 | tail -n 30 | tee -a "$OUT/gate.txt"
# install / inertness evidence before arming: both ranks installed, no hook errors, nothing sampled yet
docker exec "$CN" sh -c 'grep -h "\"installed\"\|hook_error\|capture_model_end" /tmp/verify13/v13_p*.jsonl | cut -c1-400 | head -n 12' 2>&1 | tee -a "$OUT/gate.txt"
NI=$(docker exec "$CN" sh -c 'grep -l "\"kind\": \"installed\"" /tmp/verify13/v13_p*.jsonl 2>/dev/null | wc -l')
if [ "${NI:-0}" -lt 2 ]; then echo "ABORT: verify13 hooks not installed on both ranks (found $NI)"; RC_MAIN=8; exit 8; fi

echo "== arm"
docker exec "$CN" sh -c 'touch /tmp/verify13/ARM'
sleep 3
docker exec "$CN" sh -c 'grep -h "\"armed\"" /tmp/verify13/v13_p*.jsonl' 2>&1 | tee -a "$OUT/gate.txt"

DEADLINE=$((T0 + WALL_CAP - 240))
echo "== client (deadline in $((DEADLINE - $(date +%s)))s, repeats=$REPEATS)"
REPEATS=$REPEATS timeout $((DEADLINE - $(date +%s) + 60)) python3 "$V/client_v13.py" "$OUT" "$DEADLINE" 2>&1 | tee -a "$OUT/client.log"
RC_MAIN=${PIPESTATUS[0]}
echo "client rc=$RC_MAIN"
flush_capture
exit 0
