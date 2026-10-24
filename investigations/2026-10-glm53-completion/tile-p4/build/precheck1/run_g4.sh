#!/bin/bash
# P4 gate G4 (PROPOSAL.md section 3): offline single-GPU check that the patched image with the P4 env unset reproduces
# captured-p3b's served stock MoE outputs bit for bit (and, reported, that GLM53_P4_TILE gives the prefill-plan
# arithmetic).  GPU0 only; `docker run --rm --network none --gpus device=0`; modeled on captured-p3b/replay/run_replay.sh.
#
#   ATTEMPT_REASON="..." g4/run_g4.sh            run (needs a passed precheck; budget_p4.py step g4)
#   ATTEMPT_REASON="..." DRY_RUN=1 g4/run_g4.sh  gates and plan only; starts nothing, writes nothing
#   FAKE=1 ...   CPU-only wrapper test: fake docker / nvidia-smi on PATH, ROOT_OVERRIDE under tests/out
# One cost line per attempt (1 GPU) on every exit path; `docker rm -f` on exit.  No `set -e`.
set -u
umask 022
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT_OVERRIDE:-<runs>/tile-p4}
INV=<runs>
COSTS=$ROOT/costs.jsonl
IMG=local/glm53-v091-p4-tile:local
P3B=$INV/captured-p3b
P3=$INV/moe-plan-p3
CAPDIR=$P3B/PRIVATE-capture/capture-attempt-2
CAPLOG=$P3B/runs/capture-attempt-2/caplog
MODEL_REPO=<models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1
P3_HARNESS_SHA=15ce45d8e0ab5caff09016f8f11d3c6c3e26b51499a9b22438b987a5d130b5a0
STOP_MARGIN_S=20
DRY_RUN=${DRY_RUN:-0}
ATTEMPT_REASON=${ATTEMPT_REASON:-}
SERVED_ENV=(
  -e VLLM_EXL3_TRELLIS_MIN_M=1 -e VLLM_EXL3_TRELLIS_MAX_M=32 -e VLLM_EXL3_TRELLIS_BLOCK_M=8
  -e VLLM_EXL3_PREFILL_TRELLIS=1 -e VLLM_EXL3_PREFILL_BLOCK_M=64 -e VLLM_EXL3_PREFILL_CAPACITY=1024
  -e B12X_EXL3_BF16_EPILOGUE=0 -e B12X_EXL3_BF16_GEMV=1
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_LAUNCH_BLOCKING=0 -e CUDA_MODULE_LOADING=LAZY
)
refuse() { echo "REFUSED: $*"; exit 2; }
abort()  { echo "ABORT: $*"; exit 3; }
[ -n "$ATTEMPT_REASON" ] || refuse "ATTEMPT_REASON is not set (say why this attempt runs)"
if [ "${FAKE:-0}" = 1 ]; then case "$ROOT" in */tests/out/*) ;; *) refuse "FAKE=1 needs ROOT_OVERRIDE under tests/out";; esac; fi
BUD=$(python3 "$HERE/../runner/budget_p4.py" "$ROOT" g4)
set -- $BUD
[ "${1:-}" = OK ] || refuse "budget/order gate: $BUD"
USED_GH=$2; LEFT_GH=$3; CAP=$4; ATTEMPT=$5
echo "budget: ceiling 7.0 GPU-h; used $USED_GH; left $LEFT_GH; G4 attempt $ATTEMPT wall cap ${CAP}s x 1 GPU"
AO=$ROOT/runs/g4-attempt-$ATTEMPT
[ -e "$AO" ] && refuse "$AO exists (never overwritten)"
[ -d "$CAPDIR/moe" ] || abort "captured-p3b MoE captures missing under $CAPDIR"
ls "$CAPLOG"/cap_*.jsonl >/dev/null 2>&1 || abort "captured-p3b capture log missing"
[ "$(sha256sum "$P3/harness/moe_plan_eq.py" | cut -d' ' -f1)" = "$P3_HARNESS_SHA" ] || abort "P3 harness is not the frozen one"
ID=$(docker image inspect "$IMG" --format '{{.Id}}' 2>/dev/null)
EXP=$(awk -v t="$IMG" '$1==t {print $2}' "$HERE/../runner/images.txt")
[ -n "$EXP" ] && [ "$ID" = "$EXP" ] || abort "$IMG is '$ID', expected '$EXP' (runner/images.txt)"
if docker ps --format '{{.Names}}' | grep -qE '^glm53|vllm|^p3-|^p3b-|^p4-'; then abort "an inference or check container is running"; fi
PRE=$( { echo "== preflight $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv; } 2>&1 ) || abort "nvidia-smi failed"
ok=0
for try in $(seq 1 12); do
  read -r MEM UTIL < <(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
  case "${MEM:-x}${UTIL:-x}" in *[!0-9]*) abort "cannot read GPU0 state";; esac
  if [ "$MEM" -gt 1024 ]; then echo "$PRE"; abort "GPU0 memory ${MEM} MiB > 1024 MiB"; fi
  PRE="$PRE
GPU0 sample $try: util ${UTIL}% mem ${MEM} MiB"
  [ "$UTIL" -le 5 ] && { ok=1; break; }
  [ "$DRY_RUN" = 1 ] || sleep 5
done
echo "$PRE"
[ "$ok" = 1 ] || abort "GPU0 busy after 12 samples"
UIDGID="$(id -u):$(id -g)"
CC=$ROOT/runs/g4-compile-cache   # seeded once from captured-p3b (a cache hit cannot change results)
NAME=p4-g4-$ATTEMPT
MOUNTS=(-v "$HERE":/g:ro -v "$P3B/replay":/r:ro -v "$P3/harness":/p3h:ro -v "$MODEL_REPO":/model-repo:ro
        -v "$CAPDIR":/cap:ro -v "$CAPLOG":/caplog:ro -v "$AO":/o -v "$CC":/cc)
if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN: all gates passed; no container started, nothing written."
  echo "plan: attempt $ATTEMPT -> $AO; timeout $((CAP - STOP_MARGIN_S))s; docker run --rm --pull=never --name $NAME --network none --gpus device=0 --user $UIDGID <served env> ${MOUNTS[*]} --entrypoint sh $IMG /g/in_container_g4.sh <soft>"
  exit 0
fi
mkdir -p "$AO" || abort "cannot create $AO"
echo "$PRE" > "$AO/preflight.txt"
echo "G4 attempt $ATTEMPT; reason: $ATTEMPT_REASON" > "$AO/attempt.txt"
( cd "$HERE" && sha256sum g4_moe.py in_container_g4.sh g4_verdict.py run_g4.sh ) > "$AO/scripts.sha256"
if [ ! -d "$CC" ]; then mkdir -p "$CC"; [ -d "$P3B/replay/compile-cache/v091" ] && cp -r "$P3B/replay/compile-cache/v091/." "$CC/"; fi
START=$(date -u +%FT%TZ); T0=$(date +%s); COSTED=0; RC=null
cost_line() {
  [ "$COSTED" = 1 ] && return 0
  COSTED=1
  P4_REASON="$ATTEMPT_REASON" python3 - "$COSTS" "$ATTEMPT" "$START" "$(date -u +%FT%TZ)" "$(( $(date +%s) - T0 ))" "$RC" "$ID" "$CAP" <<'PYEOF'
import json, os, sys
c, att, start, end, secs, rc, img, cap = sys.argv[1:]
open(c, "a").write(json.dumps(dict(item="p4-g4", attempt=int(att), attempt_reason=os.environ["P4_REASON"], start=start, end=end,
    seconds=int(secs), gpus=1, gpu_hours=round(int(secs) / 3600, 4), image=img, rc=None if rc == "null" else int(rc),
    wall_cap=int(cap), purpose="P4 G4 offline MoE reproduction (GPU0)")) + "\n")
PYEOF
}
on_exit() { docker rm -f "$NAME" >/dev/null 2>&1; cost_line; }
trap on_exit EXIT
trap 'echo "interrupted"; exit 130' INT TERM
TMO=$((CAP - STOP_MARGIN_S)); SOFT=$((TMO - 30))
echo "== G4 start $START timeout ${TMO}s soft ${SOFT}s"
timeout --kill-after=20 "$TMO" docker run --rm --pull=never --name "$NAME" --network none --gpus device=0 --user "$UIDGID" \
  -e HOME=/tmp -e B12X_COMPILE_CACHE_DIR=/cc "${SERVED_ENV[@]}" "${MOUNTS[@]}" \
  --entrypoint sh "$IMG" /g/in_container_g4.sh "$SOFT" > "$AO/g4.log" 2>&1
RC=$?
docker rm -f "$NAME" >/dev/null 2>&1
cost_line
echo "G4 container rc=$RC, $(( $(date +%s) - T0 ))s x 1 GPU"
grep -E '^\{"part"|rc=|elapsed' "$AO/g4.log" | tail -6
python3 "$HERE/g4_verdict.py" "$AO"
