#!/bin/bash
# First-contrast RETRY GPU session (PLAN.md amendment 22d; retry/RETRY-PROPOSAL.md, RETRY-FREEZE.md).
# PREPARED ON CPU; NOT RUN. Launch only after the operator authorizes the item "fcr1" with a GPU-hour total, a variant
# and an attempt policy:
#   APPROVED_GPU_H=6.2 VARIANT=B APPROVED_POLICY=single run_fcr.sh       (DRY_RUN=1: every pre-launch check, no GPU)
# Derived from ../../runner/run_fc.sh (attempt 1, unchanged): image-id pin, frozen SHA256SUMS, pre-flight, readiness
# gate, hook and guard registered on both ranks, in-container hook sha and WMAX 8448 before traffic, watchdog at the
# wall cap, EXIT/INT/TERM trap, one cost line at GPU release, docker cp after release, frozen CPU analysis in a CPU
# container. No `set -e`. B1, the hook and the served configuration are unchanged (environment as attempt 1).
# Changed (R-n = RETRY-PROPOSAL.md s.0):
#   R-8  item fcr1 with its own ledger retry/costs.jsonl; VARIANT B (option B) or A0 (primary only); ceilings from
#        cost/cost_fcr.json (B 4.18, A0 3.72 GPU-h); wall cap = min(ceiling, approved) x 1800 s; refused below the
#        variant's MIN_WALL (gates + whole primary contrast, central)
#   R-8  attempt policy APPROVED_POLICY: "single" (one attempt; refused if any fcr1 line exists) or "invalid-start"
#        (a second attempt ONLY if every earlier attempt stopped before any client traffic: rc 5, 6, 7, 8 or 16; the
#        approval must then cover the spent startup); never an automatic retry
#   R-1  pre-traffic replay-geometry check (rc 16): the served KV-cache geometry must be the one the F7 replay used
#        (glm53-placement kv-capacity: the 10 groups and block sizes, num_blocks >= 1,150, where the replay keeps every
#        lookup a hit with >= 409 blocks of margin), GLM53_DFLASH_BOUNDARY_LOOKUP=0, prefix-cache retention dense
#        (VLLM_PREFIX_CACHE_RETENTION_INTERVAL unset), mamba_cache_mode align, max_num_batched_tokens 2048,
#        max_num_seqs 16, DFlash K = 3
#   R-7  client_fcr.py (integrated GATE-1R via gate_step_fcr.sh; in-session early stops), analyze_fcr.py at the end
#   R-16 [PLAN.md amendment 22h s.1] variant B ceiling 6.2 occupied GPU-hours (APPROVED_GPU_H <= 6.2; wall cap
#        6.2 x 3600 / 2 = 11,160 s). SUPERSEDED, kept for the record: B 4.18 GPU-h / wall cap 7,523 s (22e-22g, the
#        cost model's max(1.3 x central, pessimistic)). A0 unchanged (3.72 / 6,696 s). Single attempt, no automatic
#        retry (APPROVED_POLICY as before).
#   R-17 [22h s.2] batch guard in the client (runner/batch_guard_fcr.py) against DEADLINE = T0 + WALL_CAP - RESERVE,
#        where WALL_CAP = min(variant wall cap, remaining approved budget): no phase starts unless its projected end
#        fits; a guard stop is rc 17. The client's own timeout now coincides with the hard wall-cap watchdog (the last
#        resort, which may cut mid-phase) instead of firing at the deadline + 15 s.
ATTEMPT=${1:-1}
HERE=$(cd "$(dirname "$0")" && pwd)
RETRY=$(cd "$HERE/.." && pwd)
ROOT=$(cd "$RETRY/.." && pwd)
COSTS=$RETRY/costs.jsonl
ITEM=fcr1
VARIANT=${VARIANT:-B}
POLICY=${APPROVED_POLICY:-}
ST=<stacks>/glm53-flash-v0.9.0
RC=<recipes>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234+d7
CN=glm53-flash-exl3-b12x-vllm
IMG=local/glm53-v0.9.1-b1-forcehook-mg2:local
IMG_CPU=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a   # B1 (CPU analysis container)
HOOK_SHA=15e93e236067567ffb9db7116986e8b732ef679e672c5b12b9d78f5884899d83
GPUS=2
case "$VARIANT" in
  B)  CEILING=6.2; WALL_MAX=11160; MIN_WALL=${MIN_WALL:-4942} ;;   # R-16 (22h); superseded: CEILING=4.18 WALL_MAX=7523
  A0) CEILING=3.72; WALL_MAX=6696; MIN_WALL=${MIN_WALL:-4945} ;;
  *)  echo "REFUSED: VARIANT must be B or A0 (got '$VARIANT')"; exit 2 ;;
esac
MIN_BLOCKS=1150
RESERVE=${RESERVE:-240}
GATE_CAP=${GATE_CAP:-1080}
DISK_MIN_GB=${DISK_MIN_GB:-200}
DEBUG="FORCEWIN=on FORCEWIN_D2=1 FORCEWIN_SHADOW=1 FORCEWIN_SNAP=1 FORCEWIN_DIR=/tmp/forcewin FORCEWIN_WMAX=8448 GLM53_MAPGUARD=device GLM53_MAPGUARD_DIR=/tmp/mapguard GLM53_MAPGUARD_POLL_SECS=20"
OUT=$RETRY/run-fcr$ATTEMPT
case "$ATTEMPT" in 1|2) ;; *) echo "REFUSED: attempt must be 1 or 2 (got '$ATTEMPT')"; exit 2 ;; esac
if [ -e "$OUT" ]; then echo "ABORT: $OUT exists"; exit 2; fi
touch "$COSTS" || exit 2

# ---- budget and attempt policy
if [ -z "${APPROVED_GPU_H:-}" ]; then echo "REFUSED: set APPROVED_GPU_H (authorized item total, at most $CEILING for $VARIANT)"; exit 2; fi
if python3 -c "import sys; v=float('$APPROVED_GPU_H'); sys.exit(0 if 0 < v <= $CEILING + 0.46 else 1)" 2>/dev/null; then :; else
  echo "REFUSED: APPROVED_GPU_H=$APPROVED_GPU_H is not in (0, $CEILING (+ 0.46 invalid-start allowance)]"; exit 2; fi
case "$POLICY" in single|invalid-start) ;; *) echo "REFUSED: set APPROVED_POLICY=single or invalid-start"; exit 2 ;; esac
read -r SPENT NATT PRE_ONLY < <(python3 -c "
import json
s=0.0; n=0; pre=True
for l in open('$COSTS'):
    l=l.strip()
    if l:
        d=json.loads(l)
        if d.get('item')=='$ITEM':
            s+=float(d.get('gpu_hours',0)); n+=1
            pre = pre and d.get('rc') in (5, 6, 7, 8, 16)
print(round(s,4), n, 1 if pre else 0)")
if [ "$NATT" -ge 1 ]; then
  if [ "$POLICY" != invalid-start ]; then echo "REFUSED: $NATT attempt(s) already recorded for $ITEM (policy single: no further attempt)"; exit 2; fi
  if [ "$NATT" -ge 2 ] || [ "$PRE_ONLY" != 1 ] || [ "$ATTEMPT" != 2 ]; then
    echo "REFUSED: invalid-start policy allows attempt 2 only after an attempt that stopped before any client traffic (rc 5-8, 16)"; exit 2; fi
elif [ "$ATTEMPT" != 1 ]; then echo "REFUSED: no attempt recorded; this must be attempt 1"; exit 2; fi
if python3 -c "import sys; sys.exit(0 if float('$APPROVED_GPU_H') <= $CEILING or '$POLICY'=='invalid-start' else 1)"; then :; else
  echo "REFUSED: APPROVED_GPU_H above the $VARIANT ceiling $CEILING requires APPROVED_POLICY=invalid-start"; exit 2; fi
LEFT_S=$(python3 -c "print(int(($APPROVED_GPU_H - $SPENT) * 3600 / $GPUS))")
WALL_CAP=$WALL_MAX
if [ "$LEFT_S" -lt "$WALL_CAP" ]; then WALL_CAP=$LEFT_S; fi
if [ "$WALL_CAP" -lt "$MIN_WALL" ]; then
  echo "REFUSED: approved ${APPROVED_GPU_H} GPU-h (spent $SPENT) gives ${WALL_CAP}s wall < ${MIN_WALL}s needed for the gates and the primary contrast"; exit 2
fi
CAP_GH=$(python3 -c "print(round($GPUS*$WALL_CAP/3600,3))")

# ---- disk (fail early)
for D in /data "$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo <docker-root>)"; do
  AV=$(df -BG --output=avail "$D" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ -z "$AV" ] || [ "$AV" -lt "$DISK_MIN_GB" ]; then echo "ABORT: $D has ${AV:-?} GB free < $DISK_MIN_GB GB (window dumps, gate copy, final copy)"; exit 2; fi
done
mkdir -p "$OUT" && cd "$OUT" || exit 2
echo "budget: variant $VARIANT policy $POLICY attempt $ATTEMPT; spent $SPENT + this attempt's cap $CAP_GH (wall ${WALL_CAP}s) <= approved $APPROVED_GPU_H GPU-h; disk ok (>= $DISK_MIN_GB GB)" | tee "$OUT/budget.txt"

# ---- pins and plan
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "ABORT: image $IMG missing"; exit 2; }
IMG_ID=$(docker image inspect "$IMG" --format '{{.Id}}')
EXP=$(awk -v t="$IMG" '$1==t {print $2}' "$HERE/images.txt")
echo "$IMG $IMG_ID" > "$OUT/image-id.txt"
if [ -z "$EXP" ] || [ "$IMG_ID" != "$EXP" ]; then echo "ABORT: $IMG is $IMG_ID, expected '$EXP'"; exit 2; fi
( cd "$RETRY" && sha256sum -c --quiet SHA256SUMS ) > "$OUT/sha256-check.txt" 2>&1 || { echo "ABORT: frozen hashes differ"; cat "$OUT/sha256-check.txt"; exit 2; }
( cd "$ROOT" && sha256sum -c --quiet SHA256SUMS ) >> "$OUT/sha256-check.txt" 2>&1 || { echo "ABORT: attempt-1 frozen files (imported read-only) differ"; cat "$OUT/sha256-check.txt"; exit 2; }
[ -f <runs>/bundle/gate/gpqa_fixture.LOCAL.json ] || { echo "ABORT: no GPQA probe fixture"; exit 2; }
python3 "$HERE/client_fcr.py" "$OUT/plan" 0 --plan --variant "$VARIANT" > "$OUT/plan.log" 2>&1 || { echo "ABORT: client plan failed"; tail "$OUT/plan.log"; exit 2; }
python3 "$HERE/plan_check_fcr.py" "$OUT/plan/plan.json" --variant "$VARIANT" > "$OUT/plan-check.txt" 2>&1 || { echo "ABORT: plan check failed"; cat "$OUT/plan-check.txt"; exit 2; }
if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY_RUN: budget gate, attempt policy, disk, frozen hashes (retry + attempt-1 imports), image pin, client plan and plan check passed ($(tail -1 "$OUT/plan-check.txt"))"; rm -rf "$OUT"; exit 0; fi

# ---- pre-flight (as attempt 1: every desktop graphics client is REPORTED)
echo "== preflight $(date -u +%FT%TZ)" | tee "$OUT/preflight.txt"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a "$OUT/preflight.txt"
nvidia-smi | tail -n 15 >> "$OUT/preflight.txt"
docker ps --format '{{.Names}} {{.Image}} {{.Status}}' | tee -a "$OUT/preflight.txt"
if docker ps --format '{{.Names}}' | grep -qE '^glm53|b0cache|mgtest'; then echo "ABORT: inference/fixture container running"; exit 3; fi
TABLE=$(nvidia-smi 2>/dev/null | awk '/Processes:/{p=1;next} p && /^\|/ && $2 ~ /^[0-9]+$/ {print}')
DESK=$(echo "$TABLE" | grep -E '[[:space:]](G|C\+G)[[:space:]]' | grep -iE 'firefox|electron|--disable-logging|gnome-shell|Xwayland|xdg-desktop-portal|snap-store|code|chrome')
if [ -n "$DESK" ]; then
  echo "NOTE: desktop graphics clients on the GPUs (tolerated within the memory limits; close them for a quiet GPU 1):" | tee -a "$OUT/preflight.txt"
  echo "$DESK" | tee -a "$OUT/preflight.txt"
fi
PROCS=$(echo "$TABLE" | grep -vE '[[:space:]]G[[:space:]].*(gnome-shell|Xwayland|xdg-desktop-portal|firefox|snap-store)' \
  | grep -vE '[[:space:]]C\+G[[:space:]].*--disable-logging' | grep -v '^$')
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
  printf '{"item":"%s","start":"%s","end":"%s","seconds":%s,"gpus":%s,"gpu_hours":%s,"attempt":%s,"image":"%s","rc":%s,"wall_cap":%s,"variant":"%s","policy":"%s","purpose":"first long-context contrast retry under PLAN.md amendment 22d"}\n' \
    "$ITEM" "$START" "$(date -u -d @"$T1" +%Y-%m-%dT%H:%M:%SZ)" "$SECS" "$GPUS" "$GH" "$ATTEMPT" "$IMG_ID" "${RC_MAIN:-null}" "$WALL_CAP" "$VARIANT" "$POLICY" >> "$COSTS"
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
    docker exec "$CN" sh -c 'cd /usr/local/lib/python3.12/dist-packages/vllm && sha256sum forcewin_hooks.py mapguard_hooks.py model_executor/layers/sparse_attn_indexer_kpool.py v1/worker/gpu/model_runner.py v1/worker/gpu/spec_decode/rejection_sampler.py v1/worker/gpu/sample/sampler.py v1/attention/backends/mla/indexer.py v1/core/kv_cache_manager.py v1/core/kv_cache_coordinator.py v1/core/single_type_kv_cache_manager.py v1/core/block_pool.py v1/core/sched/scheduler.py; env | grep -E "^(GLM53|VLLM|FORCEWIN)" | sort' > "$OUT/container-files-env.txt" 2>&1
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
  du -sh "$OUT" | tee "$OUT/disk-used.txt"
  docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e CUDA_VISIBLE_DEVICES= -v "$OUT":/run \
    -v "$ROOT":/fc:ro --entrypoint python3 "$IMG_CPU" /fc/retry/analysis/analyze_fcr.py /run /fc 2>&1 | grep -v "cpp_extension" \
    | tee "$OUT/verdict-console.txt"
}
trap cleanup EXIT
trap 'echo "signal"; RC_MAIN=130; exit 4' INT TERM
( sleep $(( WALL_CAP > 300 ? WALL_CAP - 75 : WALL_CAP )); echo "WALL CAP reached"; docker stop -t 30 "$CN" >/dev/null 2>&1 ) &
WD=$!

echo "== start $START image=$IMG ($IMG_ID) variant $VARIANT cap ${WALL_CAP}s"
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
docker logs "$CN" 2>&1 | grep -n "cudagraph_mode\|Graph capturing finished\|GPU KV cache size\|Using V2 Model Runner\|State rollback\|glm53-placement kv-capacity" | cut -c1-400 | tail -n 20 | tee -a "$OUT/gate.txt"
NF=$(docker exec "$CN" sh -c 'grep -l "\"registered\"" /tmp/forcewin/fw_r*.jsonl 2>/dev/null | wc -l')
NW=$(docker exec "$CN" sh -c 'grep -l "\"kind\": \"init\"" /tmp/mapguard/mg_p*.jsonl 2>/dev/null | wc -l')
echo "forcewin registered on $NF rank files; guard init on $NW" | tee -a "$OUT/gate.txt"
if [ "${NF:-0}" -lt 2 ] || [ "${NW:-0}" -lt 2 ]; then echo "INVALID: instrumentation not registered on both ranks"; RC_MAIN=8; exit 8; fi
HS=$(docker exec "$CN" sh -c 'sha256sum /usr/local/lib/python3.12/dist-packages/vllm/forcewin_hooks.py' 2>/dev/null | cut -d' ' -f1)
WM=$(docker exec "$CN" sh -c 'grep -h "\"kind\": \"init\"" /tmp/forcewin/fw_r*.jsonl' 2>/dev/null | python3 -c "import sys,json; print(sorted({json.loads(l)['wmax'] for l in sys.stdin if l.strip()}))")
echo "hook sha256 $HS; hook init wmax $WM" | tee -a "$OUT/gate.txt"
if [ "$HS" != "$HOOK_SHA" ]; then echo "INVALID: in-container hook $HS != $HOOK_SHA"; RC_MAIN=8; exit 8; fi
if [ "$WM" != "[8448]" ]; then echo "INVALID: hook WMAX $WM != [8448]"; RC_MAIN=8; exit 8; fi
# [R-1] the served KV-cache geometry and cache configuration are the ones the F7 replay used, before any traffic
docker logs "$CN" > "$OUT/server-start.log" 2>&1
docker exec "$CN" sh -c 'env | grep -E "^(GLM53|VLLM|FORCEWIN)" | sort' > "$OUT/container-env-start.txt" 2>&1
python3 "$HERE/geometry_check_fcr.py" "$OUT/server-start.log" "$OUT/container-env-start.txt" "$MIN_BLOCKS" | tee "$OUT/geometry-check.txt"
if [ "${PIPESTATUS[0]}" != 0 ]; then echo "INVALID: served cache geometry differs from the replayed one (no traffic sent)"; RC_MAIN=16; exit 16; fi
DEADLINE=$((T0 + WALL_CAP - RESERVE))
echo "== client (deadline in $((DEADLINE - $(date +%s)))s)"
# [R-17] the batch guard plans against DEADLINE; the client is only killed together with the hard watchdog (last resort)
timeout $((T0 + (WALL_CAP > 300 ? WALL_CAP - 75 : WALL_CAP) - $(date +%s))) python3 "$HERE/client_fcr.py" "$OUT" "$DEADLINE" --variant "$VARIANT" 2>&1 | tee -a "$OUT/client.log"
RC_MAIN=${PIPESTATUS[0]}
echo "client rc=$RC_MAIN (0 complete; 9 structural; 10 GATE-1R not PASS: no injected batch ran; 12 unsafe restoration; 13 insufficient concurrent decoding; 14 missing sentinel coverage; 17 incomplete contrast (batch guard): stopped before a phase whose projected end did not fit; 124 killed by the wall-cap watchdog)"
exit 0
