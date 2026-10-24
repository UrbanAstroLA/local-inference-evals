#!/bin/bash
# P4 single-GPU pre-check (about 0.01 GPU-h): CUDA-graph capture with every P4 hook active, no poller (see
# precheck_graph.py).  GPU0; `docker run --rm --network none --gpus device=0` of the built P4 image; synthetic tensors
# only (no model, no benchmark data).  First GPU step of P4 (budget_p4.py order: precheck -> g4 -> A -> B -> A2).
#
#   ATTEMPT_REASON="..." precheck/run_precheck.sh            run
#   ATTEMPT_REASON="..." DRY_RUN=1 precheck/run_precheck.sh  gates and plan only; starts nothing, writes nothing
#   FAKE=1 ...   CPU-only wrapper test: fake docker / nvidia-smi on PATH, ROOT_OVERRIDE under tests/out
# One cost line per attempt (1 GPU) on every exit path; `docker rm -f` on exit.  No `set -e`.
set -u
umask 022
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT_OVERRIDE:-<runs>/tile-p4}
COSTS=$ROOT/costs.jsonl
IMG=local/glm53-v091-p4-tile:local
DRY_RUN=${DRY_RUN:-0}
ATTEMPT_REASON=${ATTEMPT_REASON:-}
refuse() { echo "REFUSED: $*"; exit 2; }
abort()  { echo "ABORT: $*"; exit 3; }
[ -n "$ATTEMPT_REASON" ] || refuse "ATTEMPT_REASON is not set (say why this attempt runs)"
if [ "${FAKE:-0}" = 1 ]; then case "$ROOT" in */tests/out/*) ;; *) refuse "FAKE=1 needs ROOT_OVERRIDE under tests/out";; esac; fi
BUD=$(python3 "$HERE/../runner/budget_p4.py" "$ROOT" precheck)
set -- $BUD
[ "${1:-}" = OK ] || refuse "budget/order gate: $BUD"
USED_GH=$2; LEFT_GH=$3; CAP=$4; ATTEMPT=$5
echo "budget: ceiling 7.0 GPU-h; used $USED_GH; left $LEFT_GH; precheck attempt $ATTEMPT wall cap ${CAP}s x 1 GPU"
AO=$ROOT/runs/precheck-attempt-$ATTEMPT
[ -e "$AO" ] && refuse "$AO exists (never overwritten)"
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
NAME=p4-precheck-$ATTEMPT
if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN: all gates passed; no container started, nothing written."
  echo "plan: attempt $ATTEMPT -> $AO; timeout $((CAP - 20))s; docker run --rm --pull=never --name $NAME --network none --gpus device=0 -e GLM53_P4=on -e GLM53_P4_DIR=/tmp/p4pre -v $HERE:/pc:ro -v $AO:/o --entrypoint python3 $IMG /pc/precheck_graph.py /o/precheck.json"
  exit 0
fi
mkdir -p "$AO" || abort "cannot create $AO"
echo "$PRE" > "$AO/preflight.txt"
echo "precheck attempt $ATTEMPT; reason: $ATTEMPT_REASON" > "$AO/attempt.txt"
( cd "$HERE" && sha256sum precheck_graph.py run_precheck.sh ) > "$AO/scripts.sha256"
START=$(date -u +%FT%TZ); T0=$(date +%s); COSTED=0; RC=null
cost_line() {
  [ "$COSTED" = 1 ] && return 0
  COSTED=1
  P4_REASON="$ATTEMPT_REASON" python3 - "$COSTS" "$ATTEMPT" "$START" "$(date -u +%FT%TZ)" "$(( $(date +%s) - T0 ))" "$RC" "$ID" "$CAP" <<'PYEOF'
import json, os, sys
c, att, start, end, secs, rc, img, cap = sys.argv[1:]
open(c, "a").write(json.dumps(dict(item="p4-precheck", attempt=int(att), attempt_reason=os.environ["P4_REASON"], start=start,
    end=end, seconds=int(secs), gpus=1, gpu_hours=round(int(secs) / 3600, 4), image=img, rc=None if rc == "null" else int(rc),
    wall_cap=int(cap), purpose="P4 single-GPU graph-capture pre-check (GPU0)")) + "\n")
PYEOF
}
on_exit() { docker rm -f "$NAME" >/dev/null 2>&1; cost_line; }
trap on_exit EXIT
trap 'echo "interrupted"; exit 130' INT TERM
timeout --kill-after=20 $((CAP - 20)) docker run --rm --pull=never --name "$NAME" --network none --gpus device=0 --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e GLM53_P4=on -e GLM53_P4_DIR=/tmp/p4pre -v "$HERE":/pc:ro -v "$AO":/o \
  --entrypoint python3 "$IMG" /pc/precheck_graph.py /o/precheck.json > "$AO/precheck.log" 2>&1
RC=$?
docker rm -f "$NAME" >/dev/null 2>&1
cost_line
grep -E "^(PASS|FAIL|VERDICT)" "$AO/precheck.log"
python3 - "$AO" <<'PY'
import json, os, sys
ao = sys.argv[1]
p = os.path.join(ao, "precheck.json")
v = json.load(open(p)).get("verdict", "FAIL") if os.path.exists(p) else "FAIL"
json.dump(dict(verdict=v), open(os.path.join(ao, "verdict.json"), "w"))
print("precheck verdict", v)
PY
