#!/usr/bin/env bash
# K: model-free selection-kernel validation (PLAN.md amendment 21 section 3). ONE GPU, no model,
# no server, no network. PREPARED, NOT RUN. Requires an explicit operator budget decision:
#   APPROVED_GPU_H=0.5 ./run_k_selection.sh            full run (B1 container, then 0.7.0 container)
#   APPROVED_GPU_H=0.5 K_SMOKE=1 ./run_k_selection.sh  one batch of the first cell per route (counts
#                                                      against the same approval)
#   APPROVED_GPU_H=0.5 DRY_RUN=1 ./run_k_selection.sh  gates, pins and frozen-file check only; no GPU
# Optional: K_GPU_INDEX (default 0), K_ENGINES ("b1 v070" default, run in that order).
# Guards: frozen-file check (SHA256SUMS), image-id pins, budget gate against costs.jsonl (ceiling
# 0.5 GPU-h incl. start-up, compile and clean-up), GPU-idle pre-flight (no inference/fixture container,
# memory <= 1 GiB, utilisation <= 5 %, no process except desktop graphics clients; the plain nvidia-smi
# report is parsed because --query-compute-apps misses graphics clients), wall caps (timeout + docker
# kill), container removal and a cost line on every exit path. No `set -e`: every status is checked.
set -u
umask 022

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$HERE"
if [ -n "${K_TEST_ROOT:-}" ]; then
  # refusal/cleanup tests only: redirect costs and outputs into a test directory
  case "$K_TEST_ROOT" in
    */refusal-tests/*) BASE="$K_TEST_ROOT" ;;
    *) echo "REFUSED: K_TEST_ROOT must lie under a refusal-tests directory"; exit 2 ;;
  esac
fi
COSTS="$BASE/costs.jsonl"
ITEM="k-selection"
CEILING_GPU_H=0.5                              # PI-5 (amendment 21c): v070-only rerun; spent 0.1417 + 0.5 <= 0.65 approved
WALL_CAP_S=1800                                # whole run, 1 GPU (PI-5: = 0.5 GPU-h)
declare -A IMAGE=(
  [b1]="sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a"
  [v070]="sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629"
)
declare -A CAP_S=([b1]=850 [v070]=1740)       # per-container wall caps (PI-5: v070 alone, 1740 <= WALL_CAP_S - 60)
MIN_CAP_S=120
if [ -n "${K_TEST_ROOT:-}" ] && [ -n "${K_TEST_CAP_S:-}" ]; then CAP_S=([b1]=$K_TEST_CAP_S [v070]=$K_TEST_CAP_S); MIN_CAP_S=1; fi
GPU_INDEX="${K_GPU_INDEX:-0}"
ENGINES="${K_ENGINES:-v070}"                   # PI-5: rerun scope is 0.7.0 only (B1 34/34 PASS stands)
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$BASE/run-$STAMP"
START=""
STATUS="not-started"
NAMES=()

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

cost_line() {
  [ -z "$START" ] && return 0
  local end wall gh
  end=$(date +%s)
  wall=$(( end - START ))
  gh=$(python3 -c "print(round(1*$wall/3600, 4))")
  printf '{"item": "%s", "run": "%s", "engines": "%s", "smoke": %s, "status": "%s", "wall_s": %d, "gpus": 1, "gpu_index": %s, "gpu_hours": %s, "ceiling_gpu_hours": %s, "images": ["%s", "%s"]}\n' \
    "$ITEM" "$STAMP" "$ENGINES" "${K_SMOKE:-0}" "$STATUS" "$wall" "$GPU_INDEX" "$gh" "$CEILING_GPU_H" \
    "${IMAGE[b1]}" "${IMAGE[v070]}" >> "$COSTS"
  log "cost: ${wall}s wall, ${gh} GPU-hours (status $STATUS)"
}

CHILD=""
cleanup() {
  local n
  if [ -n "$CHILD" ] && kill -0 "$CHILD" 2>/dev/null; then kill -TERM "$CHILD" 2>/dev/null; fi
  for n in "${NAMES[@]}"; do
    docker rm -f "$n" >/dev/null 2>&1
  done
  cost_line
  START=""
}
trap cleanup EXIT
trap 'STATUS="interrupted"; exit 130' INT TERM

case "$GPU_INDEX" in 0|1) ;; *) echo "REFUSED: K_GPU_INDEX must be 0 or 1"; exit 2 ;; esac
for e in $ENGINES; do
  [ -n "${IMAGE[$e]:-}" ] || { echo "REFUSED: unknown engine '$e'"; exit 2; }
done
mkdir -p "$BASE" || { echo "cannot create $BASE"; exit 2; }
touch "$COSTS" || { echo "cannot write $COSTS"; exit 2; }

# ---------------------------------------------------------------- frozen files
if ! ( cd "$HERE" && sha256sum --quiet -c SHA256SUMS ) ; then
  echo "REFUSED: fixture files differ from SHA256SUMS (criteria are frozen; re-freeze deliberately)"; exit 2
fi
log "frozen files verified against SHA256SUMS"

# ---------------------------------------------------------------- budget gate
if [ -z "${APPROVED_GPU_H:-}" ]; then
  echo "REFUSED: set APPROVED_GPU_H to the operator-approved total for item K (proposed 0.5)."; exit 2
fi
if ! python3 -c "import sys; float(sys.argv[1])" "$APPROVED_GPU_H" 2>/dev/null; then
  echo "REFUSED: APPROVED_GPU_H is not a number"; exit 2
fi
SPENT=$(python3 - "$COSTS" "$ITEM" <<'EOF'
import json, sys
t = 0.0
for l in open(sys.argv[1]):
    l = l.strip()
    if l:
        d = json.loads(l)
        if d.get("item") == sys.argv[2]:
            t += float(d.get("gpu_hours", 0))
print(t)
EOF
)
if [ -z "$SPENT" ]; then echo "REFUSED: cannot read $COSTS"; exit 2; fi
if ! python3 -c "import sys; sys.exit(0 if $SPENT + $CEILING_GPU_H <= float(sys.argv[1]) + 1e-9 else 1)" "$APPROVED_GPU_H"; then
  echo "REFUSED: spent ${SPENT} + ceiling ${CEILING_GPU_H} > approved ${APPROVED_GPU_H} GPU-hours"; exit 2
fi
log "budget: spent ${SPENT}, this run ceiling ${CEILING_GPU_H}, approved ${APPROVED_GPU_H}"

# ---------------------------------------------------------------- image pins
for e in $ENGINES; do
  got=$(docker image inspect "${IMAGE[$e]}" --format '{{.Id}}' 2>/dev/null)
  if [ "$got" != "${IMAGE[$e]}" ]; then echo "REFUSED: $e image ${IMAGE[$e]} not present (got '${got}')"; exit 2; fi
done
log "images pinned: $(for e in $ENGINES; do printf '%s=%s ' "$e" "${IMAGE[$e]:7:12}"; done)"
if [ "${DRY_RUN:-0}" = "1" ]; then
  log "DRY_RUN: frozen files, budget gate and image pins passed; stopping before the GPU pre-flight"; exit 0
fi

# ---------------------------------------------------------------- GPU-idle pre-flight
gpu_idle() {
  local running q used util procs
  running=$(docker ps --format '{{.Names}} {{.Image}}' 2>/dev/null)
  if printf '%s\n' "$running" | grep -q -i -E 'glm53|vllm|kselect|fixture|b0cache'; then
    echo "REFUSED: an inference/fixture container is running:"; printf '  %s\n' "$running"; return 1
  fi
  q=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  if [ -z "$q" ]; then echo "REFUSED: nvidia-smi query failed"; return 1; fi
  used=$(echo "$q" | cut -d, -f1 | tr -d ' '); util=$(echo "$q" | cut -d, -f2 | tr -d ' ')
  procs=$(nvidia-smi -i "$GPU_INDEX" 2>/dev/null | awk '/Processes:/{p=1;next} p && /^\|/ && $2 ~ /^[0-9]+$/ {print}' \
    | grep -vE '[[:space:]]G[[:space:]].*(gnome-shell|Xwayland|xdg-desktop-portal)')
  if [ "${used:-99999}" -gt 1024 ] || [ "${util:-100}" -gt 5 ] || [ -n "$procs" ]; then
    echo "REFUSED: GPU $GPU_INDEX not idle (used ${used} MiB, util ${util}%)"; [ -n "$procs" ] && echo "$procs"; return 1
  fi
  log "pre-flight ok: GPU $GPU_INDEX used ${used} MiB, util ${util}%"
  return 0
}
gpu_idle || exit 2

mkdir -p "$OUT" && chmod 777 "$OUT"
( cd "$HERE" && cp SHA256SUMS "$OUT/frozen-SHA256SUMS" )

# ---------------------------------------------------------------- run
START=$(date +%s)
STATUS="running"
K_MAX_BATCHES=0
K_CELLS=""
if [ "${K_SMOKE:-0}" = "1" ]; then K_MAX_BATCHES=1; K_CELLS='/(tiled(32|48)|fused(4|8))/b(16384|24064|11776)$'; fi
for e in $ENGINES; do
  now=$(date +%s)
  left=$(( WALL_CAP_S - (now - START) - 60 ))
  cap=${CAP_S[$e]}
  [ "$left" -lt "$cap" ] && cap=$left
  if [ "$cap" -lt "$MIN_CAP_S" ]; then log "skip $e: ${cap}s left under the wall cap"; STATUS="partial-wall-cap"; continue; fi
  if [ "$e" != "${ENGINES%% *}" ]; then gpu_idle || { STATUS="refused-mid-run"; break; }; fi
  name="kselect-$e-$$"
  NAMES+=("$name")
  mkdir -p "$OUT/$e" && chmod 777 "$OUT/$e"
  log "starting $name (${IMAGE[$e]:7:12}, wall cap ${cap}s)"
  # Background + wait: a trapped INT/TERM interrupts `wait` at once, so cleanup removes the container
  # immediately instead of after the cap.
  timeout --signal=TERM --kill-after=30 "$cap" \
    docker run --rm --name "$name" --gpus "device=$GPU_INDEX" --network none --memory 32g --shm-size 2g \
      --user "$(id -u):$(id -g)" -e HOME=/tmp -e TRITON_CACHE_DIR=/tmp/triton -e PYTHONDONTWRITEBYTECODE=1 \
      -e K_IMAGE_ID="${IMAGE[$e]}" -e K_ENGINE="$e" -e K_MAX_SECONDS="$(( cap - 45 ))" -e K_MAX_BATCHES="$K_MAX_BATCHES" -e K_CELLS="$K_CELLS" \
      -v "$HERE/kselect":/k/kselect:ro -v "$HERE/plan.json":/k/plan.json:ro -v "$HERE/bank":/k/bank:ro \
      -v "$OUT/$e":/o --entrypoint bash "${IMAGE[$e]}" /k/kselect/inner.sh \
      > "$OUT/$e/container.log" 2>&1 &
  CHILD=$!
  wait "$CHILD"
  rc=$?
  CHILD=""
  docker rm -f "$name" >/dev/null 2>&1
  if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then STATUS="wall-cap-$e"; log "$e hit the wall cap (rc $rc)";
  else log "$e container finished: rc=$rc"; fi
  cat "$OUT/$e/exit.txt" 2>/dev/null
done
[ "$STATUS" = "running" ] && STATUS="done"
python3 "$HERE/kselect/report.py" "$OUT" 2>&1 | tee "$OUT/REPORT.txt"
log "results in $OUT"
exit 0
