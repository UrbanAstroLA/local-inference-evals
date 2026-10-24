#!/bin/bash
# replay-topk GPU runner (PROTOCOL.md section 8).  Single GPU (GPU 0), no server, the stock images by digest,
# --network none.  Not run by the package build or the CPU tests (they use FAKE=1).
#
#   APPROVED_GPU_H=<h> ATTEMPT_REASON="..." runner/run_replay_topk.sh            run v091 then v070
#   ... IMAGES=v091 runner/run_replay_topk.sh                                     one image only
#   ... DRY_RUN=1 runner/run_replay_topk.sh                                       every gate, then the plan; starts nothing
#   FAKE=1 ROOT_OVERRIDE=<dir under tests/out> ...   CPU wrapper test: fake docker / nvidia-smi on PATH
#
# Gates, all before any GPU container: ATTEMPT_REASON; frozen package (sha256sum -c SHA256SUMS); images present by
# digest (no pulls) and equal to runner/images.txt; budget/attempt gate (budget_replay.py, APPROVED_GPU_H required);
# no inference or check container running; GPU-0 idle preflight (preflight_gpu.py, reports graphics clients).
# One cost line per image container on every exit path (incl. interrupt); `docker rm -f` on exit.  No `set -e`.
set -u
umask 022
HERE=$(cd "$(dirname "$0")" && pwd)
PKG=$(cd "$HERE/.." && pwd)
ROOT=${ROOT_OVERRIDE:-$PKG}
COSTS=$ROOT/costs-replay.jsonl
IMAGES=${IMAGES:-v091,v070}
DRY_RUN=${DRY_RUN:-0}
ATTEMPT_REASON=${ATTEMPT_REASON:-}
STOP_MARGIN_S=20
SOFT_MARGIN_S=40
SERVED_ENV=(
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_LAUNCH_BLOCKING=0 -e CUDA_MODULE_LOADING=LAZY
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 -e VLLM_USE_B12X_KPOOL_INDEXER=1
)
refuse() { echo "REFUSED: $*"; exit 2; }
abort()  { echo "ABORT: $*"; exit 3; }

[ -n "$ATTEMPT_REASON" ] || refuse "ATTEMPT_REASON is not set (say why this attempt runs)"
if [ "${FAKE:-0}" = 1 ]; then case "$ROOT" in */tests/out/*) ;; *) refuse "FAKE=1 needs ROOT_OVERRIDE under tests/out";; esac; fi
( cd "$PKG" && sha256sum -c --quiet SHA256SUMS ) >/dev/null 2>&1 || refuse "package differs from SHA256SUMS (frozen package; re-freeze only with a protocol amendment)"
declare -A IMG ID
while read -r v ref id; do [ -n "$v" ] && IMG[$v]=$ref && ID[$v]=$id; done < "$HERE/images.txt"
for v in ${IMAGES//,/ }; do
  [ -n "${IMG[$v]:-}" ] || refuse "unknown image key $v"
  got=$(docker image inspect "${IMG[$v]}" --format '{{.Id}}' 2>/dev/null)
  [ "$got" = "${ID[$v]}" ] || abort "image $v is '${got:-absent}', expected ${ID[$v]} (no pulls)"
done
BUD=$(python3 "$HERE/budget_replay.py" "$ROOT" "$IMAGES")
set -- $BUD
[ "${1:-}" = OK ] || refuse "budget/attempt gate: $BUD"
USED_GH=$2; LEFT_GH=$3; ATTEMPT=$4; CAP_v091=$5; CAP_v070=$6
echo "budget: approved ${APPROVED_GPU_H} GPU-h; used $USED_GH; left $LEFT_GH; attempt $ATTEMPT; caps v091 ${CAP_v091}s v070 ${CAP_v070}s (1 GPU)"
AO=$ROOT/runs/attempt-$ATTEMPT
[ -e "$AO" ] && refuse "$AO exists (never overwritten)"
if docker ps --format '{{.Names}}' | grep -qE '^glm53|vllm|^p3-|^p3b-|^p4-|^rtk-'; then
  docker ps --format '{{.Names}}'; abort "an inference or check container is running"
fi
PRE=$(python3 "$HERE/preflight_gpu.py" --samples 12 --interval "$([ "$DRY_RUN" = 1 ] && echo 0 || echo 5)" 2>&1)
PRC=$?
echo "$PRE"
[ "$PRC" = 0 ] || abort "GPU preflight failed (see above)"
UIDGID="$(id -u):$(id -g)"
CC=$ROOT/cache          # compile caches: regenerable, excluded from SHA256SUMS and from publication
if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN: all gates passed; no container started, nothing written."
  for v in ${IMAGES//,/ }; do
    cap=CAP_$v
    echo "plan: $v timeout $(( ${!cap} - STOP_MARGIN_S ))s: docker run --rm --pull=never --name rtk-$ATTEMPT-$v --network none --gpus device=0 --user $UIDGID -e HOME=/tmp -e B12X_COMPILE_CACHE_DIR=/cc/b12x -e DG_JIT_CACHE_DIR=/cc/deep_gemm -e TRITON_CACHE_DIR=/cc/triton <served env> -v $PKG/scripts:/s:ro -v $AO/$v:/o -v $CC/$v:/cc --entrypoint sh ${IMG[$v]} /s/in_container.sh $v <soft>"
  done
  echo "then (CPU): analysis/analyze_replay.py $AO"
  exit 0
fi
mkdir -p "$AO" || abort "cannot create $AO"
echo "$PRE" > "$AO/preflight.txt"
echo "replay-topk attempt $ATTEMPT; reason: $ATTEMPT_REASON; images $IMAGES" > "$AO/attempt.txt"
cp "$PKG/SHA256SUMS" "$AO/package-SHA256SUMS"
seed_cache() {  # $1 version: seed once from the serving caches (read only there); a cache hit cannot change results
  local v=$1 b12x dg
  [ -d "$CC/$v" ] && return 0
  mkdir -p "$CC/$v/b12x" "$CC/$v/deep_gemm" "$CC/$v/triton"
  case $v in v091) b12x=<cache>/vllm-v090/b12x/compile; dg=<cache>/vllm-v090/vllm/deep_gemm;;
             v070) b12x=<cache>/vllm/b12x/compile; dg="";; esac
  [ "${FAKE:-0}" = 1 ] && return 0
  [ -r "$b12x" ] && cp -r "$b12x/." "$CC/$v/b12x/" 2>/dev/null
  [ -n "$dg" ] && [ -r "$dg/cache" ] && cp -r "$dg/." "$CC/$v/deep_gemm/" 2>/dev/null
  echo "seeded compile cache $v: $(du -sh "$CC/$v" 2>/dev/null | cut -f1)"
}
CURRENT=""; CUR_T0=0; CUR_START=""; COSTED=" "; CUR_CAP=0
cost_line() {  # version seconds rc start end cap
  local v=$1
  RT_REASON="$ATTEMPT_REASON" python3 - "$COSTS" "$ATTEMPT" "$v" "$2" "$3" "$4" "$5" "$6" "${ID[$v]}" <<'PYEOF'
import json, os, sys
c, att, v, secs, rc, start, end, cap, img = sys.argv[1:]
open(c, "a").write(json.dumps(dict(item="replay-topk", attempt=int(att), engine=v, attempt_reason=os.environ["RT_REASON"],
    start=start, end=end, seconds=int(secs), gpus=1, gpu_hours=round(int(secs) / 3600.0, 4), image=img,
    rc=None if rc == "null" else int(rc), wall_cap=int(cap), purpose="replay-topk single-GPU replay (GPU0)")) + "\n")
PYEOF
  COSTED="$COSTED$v "
}
CPID=""
on_exit() {
  for v in ${IMAGES//,/ }; do docker rm -f "rtk-$ATTEMPT-$v" >/dev/null 2>&1; done
  [ -n "$CPID" ] && kill "$CPID" 2>/dev/null
  if [ -n "$CURRENT" ] && [[ "$COSTED" != *" $CURRENT "* ]]; then
    cost_line "$CURRENT" $(( $(date +%s) - CUR_T0 )) 130 "$CUR_START" "$(date -u +%FT%TZ)" "$CUR_CAP"
  fi
}
trap on_exit EXIT
trap 'echo "interrupted"; exit 130' INT TERM
OVERALL=0
for v in ${IMAGES//,/ }; do
  cap=CAP_$v; CAP=${!cap}
  seed_cache "$v"
  mkdir -p "$AO/$v"
  TMO=$((CAP - STOP_MARGIN_S)); SOFT=$((TMO - SOFT_MARGIN_S))
  CURRENT=$v; CUR_CAP=$CAP; CUR_START=$(date -u +%FT%TZ); CUR_T0=$(date +%s)
  echo "== $v start $CUR_START timeout ${TMO}s soft ${SOFT}s"
  # backgrounded + wait, so an interrupt is handled at once (bash defers traps while a foreground child runs)
  timeout --kill-after=15 "$TMO" docker run --rm --pull=never --name "rtk-$ATTEMPT-$v" --network none --gpus device=0 \
    --user "$UIDGID" -e HOME=/tmp -e B12X_COMPILE_CACHE_DIR=/cc/b12x -e DG_JIT_CACHE_DIR=/cc/deep_gemm \
    -e TRITON_CACHE_DIR=/cc/triton "${SERVED_ENV[@]}" -v "$PKG/scripts":/s:ro -v "$AO/$v":/o -v "$CC/$v":/cc \
    --entrypoint sh "${IMG[$v]}" /s/in_container.sh "$v" "$SOFT" > "$AO/$v.log" 2>&1 &
  CPID=$!
  wait "$CPID"
  RC=$?
  CPID=""
  docker rm -f "rtk-$ATTEMPT-$v" >/dev/null 2>&1
  cost_line "$v" $(( $(date +%s) - CUR_T0 )) "$RC" "$CUR_START" "$(date -u +%FT%TZ)" "$CAP"
  CURRENT=""
  echo "$v rc=$RC, $(( $(date +%s) - CUR_T0 ))s x 1 GPU"
  grep -E '^\{"version"|elapsed|REFUSED' "$AO/$v.log" | tail -3
  [ "$RC" = 0 ] || OVERALL=4
done
python3 "$PKG/analysis/analyze_replay.py" "$AO" 2>&1 | tee "$AO/analysis-console.txt"
python3 - "$COSTS" "${APPROVED_GPU_H}" <<'PYEOF'
import json, sys
s = sum(float(json.loads(l)["seconds"]) * float(json.loads(l).get("gpus", 1)) for l in open(sys.argv[1]) if l.strip())
print(f"replay-topk ledger: {s / 3600:.4f} GPU-h used of {float(sys.argv[2]):.4f} approved")
PYEOF
exit $OVERALL
