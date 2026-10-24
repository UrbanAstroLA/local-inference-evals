#!/bin/bash
# capture-p0 single-GPU pre-check (PROTOCOL.md section 6; cap 180 s x 1 GPU): point-0 capture on a served-layout
# synthetic KV cache, CUDA-graph capture with the localisation hooks installed, an armed 2,045-token hashing step at
# the served row widths, negative control -- precheck/precheck_cap0.py.  GPU0; `docker run --rm --network none
# --gpus device=0` of the cap0 image.  No model, no benchmark data.  First GPU step (budget order: precheck -> session).
#
#   APPROVED_GPU_H=<approval> ATTEMPT_REASON="..." runner/run_precheck_cap0.sh          run
#   ... DRY_RUN=1 ...          gates and plan only; starts nothing, writes nothing
#   FAKE=1 ...                 CPU-only wrapper test: fake docker / nvidia-smi on PATH, ROOT_OVERRIDE under tests/out
# Budget: refuses unless APPROVED_GPU_H is set (no default) and when cap + used (ledger costs-cap0.jsonl) would exceed
# it.  One cost line per attempt (1 GPU) on every exit path.  No `set -e`.
set -u
umask 022
HERE=$(cd "$(dirname "$0")" && pwd)
PKG_REAL=$(cd "$HERE/.." && pwd)
PKG=${ROOT_OVERRIDE:-$PKG_REAL}
PC=$PKG_REAL/precheck
COSTS=$PKG/costs-cap0.jsonl
IMG=local/glm53-v091-p4-cap0-a2:local
DRY_RUN=${DRY_RUN:-0}
ATTEMPT_REASON=${ATTEMPT_REASON:-}
refuse() { echo "REFUSED: $*"; exit 2; }
abort()  { echo "ABORT: $*"; exit 3; }
[ -n "$ATTEMPT_REASON" ] || refuse "ATTEMPT_REASON is not set (say why this attempt runs)"
[ -n "${APPROVED_GPU_H:-}" ] || refuse "APPROVED_GPU_H is not set: capture-p0 needs an explicit operator approval in GPU-hours (no default)"
if [ "${FAKE:-0}" = 1 ]; then case "$PKG" in "$PKG_REAL"/tests/out/*) ;; *) refuse "FAKE=1 needs ROOT_OVERRIDE under tests/out";; esac; fi
EXP=$(awk -v t="$IMG" '$1==t {print $2}' "$HERE/images.txt")
BUD=$(APPROVED_GPU_H="$APPROVED_GPU_H" python3 "$HERE/budget_cap0.py" "$PKG" precheck "$EXP")
set -- $BUD
[ "${1:-}" = OK ] || refuse "budget/order gate: $BUD"
USED_GH=$2; LEFT_GH=$3; CAP=$4; ATTEMPT=$5
echo "budget: approved $APPROVED_GPU_H GPU-h (costs-cap0.jsonl); used $USED_GH; left $LEFT_GH; precheck attempt $ATTEMPT wall cap ${CAP}s x 1 GPU"
AO=$PKG/runs/precheck-attempt-$ATTEMPT
[ -e "$AO" ] && refuse "$AO exists (never overwritten)"
if [ "${FAKE:-0}" != 1 ]; then
  SHAOUT=$( cd "$PKG_REAL" && sha256sum -c --quiet SHA256SUMS 2>&1 ) || { echo "$SHAOUT"; abort "frozen hashes differ (capture-p0/SHA256SUMS)"; }
fi
ID=$(docker image inspect "$IMG" --format '{{.Id}}' 2>/dev/null)
[ -n "$EXP" ] && [ "$ID" = "$EXP" ] || abort "$IMG is '$ID', expected '$EXP' (runner/images.txt)"
if docker ps --format '{{.Names}}' | grep -qE '^glm53|vllm|^p3-|^p3b-|^p4-|^cap0-'; then abort "an inference or check container is running"; fi
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
NAME=cap0-precheck-$ATTEMPT
ENVS="-e GLM53_P4=on -e GLM53_P4_S0=on -e GLM53_CAP0=on -e GLM53_CAP0_LOC=on -e GLM53_P4_DIR=/tmp/cap0pre"
if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN: all gates passed; no container started, nothing written."
  echo "plan: attempt $ATTEMPT -> $AO; timeout $((CAP - 20))s; docker run --rm --pull=never --name $NAME --network none --gpus device=0 $ENVS -v $PC:/pc:ro -v $AO:/o --entrypoint python3 $IMG /pc/precheck_cap0.py /o/precheck.json"
  exit 0
fi
mkdir -p "$AO" || abort "cannot create $AO"
echo "$PRE" > "$AO/preflight.txt"
echo "cap0 precheck attempt $ATTEMPT; reason: $ATTEMPT_REASON; approved $APPROVED_GPU_H GPU-h" > "$AO/attempt.txt"
( cd "$PC" && sha256sum precheck_cap0.py; cd "$HERE" && sha256sum run_precheck_cap0.sh budget_cap0.py ) > "$AO/scripts.sha256"
START=$(date -u +%FT%TZ); T0=$(date +%s); COSTED=0; RC=null
cost_line() {
  [ "$COSTED" = 1 ] && return 0
  COSTED=1
  CAP0_REASON="$ATTEMPT_REASON" python3 - "$COSTS" "$ATTEMPT" "$START" "$(date -u +%FT%TZ)" "$(( $(date +%s) - T0 ))" "$RC" "$ID" "$CAP" "$APPROVED_GPU_H" "${FAKE:-0}" <<'PYEOF'
import json, os, sys
c, att, start, end, secs, rc, img, cap, appr, fake = sys.argv[1:]
open(c, "a").write(json.dumps(dict(item="cap0-precheck", attempt=int(att), attempt_reason=os.environ["CAP0_REASON"], start=start,
    end=end, seconds=int(secs), gpus=1, gpu_hours=round(int(secs) / 3600, 4), image=img, rc=None if rc == "null" else int(rc),
    wall_cap=int(cap), approved_gpu_h=float(appr), fake=fake == "1",
    purpose="capture-p0 single-GPU pre-check (GPU0): capture addressing, graph capture with hooks, hashing cost")) + "\n")
PYEOF
}
on_exit() { docker rm -f "$NAME" >/dev/null 2>&1; cost_line; }
trap on_exit EXIT
trap 'echo "interrupted"; exit 130' INT TERM
timeout --kill-after=20 $((CAP - 20)) docker run --rm --pull=never --name "$NAME" --network none --gpus device=0 --user "$(id -u):$(id -g)" \
  -e HOME=/tmp $ENVS -v "$PC":/pc:ro -v "$AO":/o --entrypoint python3 "$IMG" /pc/precheck_cap0.py /o/precheck.json > "$AO/precheck.log" 2>&1
RC=$?
docker rm -f "$NAME" >/dev/null 2>&1
cost_line
grep -E "^(PASS|FAIL|VERDICT)" "$AO/precheck.log"
FAKE="${FAKE:-0}" python3 - "$AO" "$ID" <<'PY'
import json, os, sys
ao = sys.argv[1]
p = os.path.join(ao, "precheck.json")
d = json.load(open(p)) if os.path.exists(p) else {}
# a pre-check counts only if it ran its graph checks on the GPU (a CPU fallback is not a pass), except in FAKE tests
v = "PASS" if d.get("verdict") == "PASS" and (d.get("gpu") or os.environ.get("FAKE") == "1") else "FAIL"
json.dump(dict(verdict=v, gpu=d.get("gpu"), times=d.get("times"), image=sys.argv[2]), open(os.path.join(ao, "verdict.json"), "w"))
print("cap0 precheck verdict", v, json.dumps(d.get("times")))
PY
