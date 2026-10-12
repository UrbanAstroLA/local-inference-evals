#!/usr/bin/env bash
# B0 cache isolation + high-offset addressing fixture runner (ONE GPU, model-free).
# PREPARED, NOT RUN. Requires an explicit operator budget decision:
#   APPROVED_GPU_H=0.5 ./run_b0_cache_fixture.sh
# Steps (one container, B0 image, GPU 0 only, no network, no server):
#   1. run_fixture.py            -> results.json   (parts A-G, Z; see PROPOSAL.md)
#   2. existing B12x >2^31 tests -> pytest-*.txt   (sm120 GLM_NSA high-pages; GLM_NEXT high-page
#                                                   as shipped (expected TypeError) and repaired)
# Guards: image id pin, GPU-0 idle pre-flight (incl. graphics clients), budget gate against
# costs.jsonl, wall cap (timeout + docker kill), cost line + container removal on every exit path.
# No `set -e` (status queries are checked explicitly).
set -u
umask 022

IMAGE_ID="sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE=<runs>/b0-cache
COSTS="$BASE/costs.jsonl"
CEILING_GPU_H=0.5            # this run's ceiling (1 GPU x 30 min), startup/compile/cleanup included
WALL_CAP_S=1800              # hard wall cap for the whole container
FIXTURE_CAP_S=1380           # inner cap for step 1 (leaves time for step 2 and cleanup)
PYTEST_CAP_S=360             # inner cap for step 2
# Post-run-2 (amendment 16e): a partial rerun may declare a smaller per-run ceiling so that
# spent + ceiling stays within the approved total; the wall cap follows the ceiling (1 GPU).
if [ -n "${RUN_CEILING_GPU_H:-}" ]; then
  CEILING_GPU_H="$RUN_CEILING_GPU_H"
  WALL_CAP_S=$(python3 -c "print(int(float('$RUN_CEILING_GPU_H') * 3600))")
  FIXTURE_CAP_S=$(( WALL_CAP_S - 120 ))
fi
GPU_INDEX=0
NAME="b0cache-fixture-$$"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$BASE/run-$STAMP"
START=""
STATUS="not-started"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

cost_line() {
  # Written on every exit path once the GPU was claimed (START set).
  [ -z "$START" ] && return 0
  local end wall gh
  end=$(date +%s)
  wall=$(( end - START ))
  gh=$(python3 -c "print(round(1*$wall/3600, 4))")
  printf '{"item": "b0-cache-fixture", "run": "%s", "status": "%s", "wall_s": %d, "gpus": 1, "gpu_hours": %s, "ceiling_gpu_hours": %s, "image": "%s"}\n' \
    "$STAMP" "$STATUS" "$wall" "$gh" "$CEILING_GPU_H" "$IMAGE_ID" >> "$COSTS"
  log "cost: ${wall}s wall, ${gh} GPU-hours (status $STATUS)"
}

cleanup() {
  docker rm -f "$NAME" >/dev/null 2>&1
  cost_line
  START=""
}
trap cleanup EXIT
trap 'STATUS="interrupted"; exit 130' INT TERM

mkdir -p "$BASE" || { echo "cannot create $BASE"; exit 2; }
touch "$COSTS"

# ---------------------------------------------------------------- budget gate
if [ -z "${APPROVED_GPU_H:-}" ]; then
  echo "REFUSED: set APPROVED_GPU_H to the operator-approved total for this item (proposed 0.5)."; exit 2
fi
SPENT=$(python3 - "$COSTS" <<'EOF'
import json, sys
t = 0.0
for l in open(sys.argv[1]):
    l = l.strip()
    if l:
        d = json.loads(l)
        if d.get("item") == "b0-cache-fixture":
            t += float(d.get("gpu_hours", 0))
print(t)
EOF
)
if ! python3 -c "import sys; sys.exit(0 if $SPENT + $CEILING_GPU_H <= ${APPROVED_GPU_H} + 1e-9 else 1)"; then
  echo "REFUSED: spent ${SPENT} + ceiling ${CEILING_GPU_H} > approved ${APPROVED_GPU_H} GPU-hours"; exit 2
fi
log "budget: spent ${SPENT}, this run ceiling ${CEILING_GPU_H}, approved ${APPROVED_GPU_H}"

# ---------------------------------------------------------------- pre-flight
got=$(docker image inspect "$IMAGE_ID" --format '{{.Id}}' 2>/dev/null)
if [ "$got" != "$IMAGE_ID" ]; then echo "REFUSED: B0 image $IMAGE_ID not present"; exit 2; fi
if [ "${DRY_RUN:-0}" = "1" ]; then
  log "DRY_RUN: budget gate and image check passed; stopping before the GPU pre-flight"; exit 0
fi
if docker ps --format '{{.Names}}' | grep -q .; then
  echo "running containers:"; docker ps --format '  {{.Names}}  {{.Image}}'
  if docker ps --format '{{.Names}}' | grep -q -E 'glm53|vllm|b0cache'; then
    echo "REFUSED: an inference/fixture container is running"; exit 2
  fi
fi
q=$(nvidia-smi -i $GPU_INDEX --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
if [ -z "$q" ]; then echo "REFUSED: nvidia-smi query failed"; exit 2; fi
used=$(echo "$q" | cut -d, -f1 | tr -d ' '); util=$(echo "$q" | cut -d, -f2 | tr -d ' ')
# Process table from the plain report (also lists graphics clients, which
# --query-compute-apps misses).
procs=$(nvidia-smi -i $GPU_INDEX 2>/dev/null | awk '/Processes:/{p=1;next} p && /^\|/ && $2 ~ /^[0-9]+$/ {print}' \
  | grep -vE '[[:space:]]G[[:space:]].*(gnome-shell|Xwayland|xdg-desktop-portal)' )  # desktop graphics clients only (amendment 16d)
if [ "${used:-99999}" -gt 1024 ] || [ "${util:-100}" -gt 5 ] || [ -n "$procs" ]; then
  echo "REFUSED: GPU $GPU_INDEX not idle (used ${used} MiB, util ${util}%)"; echo "$procs"; exit 2
fi
log "pre-flight ok: GPU $GPU_INDEX used ${used} MiB, util ${util}%"

mkdir -p "$OUT" && chmod 777 "$OUT"
( cd "$HERE" && sha256sum b0geom.py b0shadow.py b0views.py run_fixture.py test_repaired_high_page.py inner.sh "$(basename "$0")" ) > "$OUT/fixture.sha256"

# ---------------------------------------------------------------- run
START=$(date +%s)
STATUS="running"
log "starting container $NAME (wall cap ${WALL_CAP_S}s)"
timeout --signal=TERM --kill-after=30 "$WALL_CAP_S" \
  docker run --rm --name "$NAME" --gpus "device=$GPU_INDEX" --network none \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e TRITON_CACHE_DIR=/tmp/triton \
    -e VLLM_GLM53_NOPE_RECORD=1 -e PYTHONPATH=/opt/b12x -e B0_IMAGE_ID="$IMAGE_ID" \
    -e FIXTURE_CAP_S="$FIXTURE_CAP_S" -e PYTEST_CAP_S="$PYTEST_CAP_S" -e FIXTURE_PARTS="${FIXTURE_PARTS:-}" \
    -v "$HERE":/f:ro -v "$OUT":/o -w /f --entrypoint bash "$IMAGE_ID" /f/inner.sh \
    > "$OUT/container.log" 2>&1
rc=$?
if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then STATUS="wall-cap"; else STATUS="exit-$rc"; fi
log "container finished: rc=$rc"
cat "$OUT/exit.txt" 2>/dev/null

if [ -f "$OUT/results.json" ]; then
  python3 - "$OUT/results.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
for k, v in d["parts"].items():
    print(f"{k:32s} ok={v.get('ok')} secs={v.get('secs')} {v.get('error', '')[:120]}")
bad = [c for c in d["checks"] if not c["ok"]]
print(f"failed checks: {len(bad)}")
for c in bad[:20]:
    print("  ", c.get("part"), c.get("label"), c.get("where", "")[:2] if c.get("where") else "")
EOF
fi
log "results in $OUT"
exit 0
