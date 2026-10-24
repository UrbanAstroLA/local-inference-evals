#!/bin/bash
# captured-p3b, step B: offline replay on GPU0 (single GPU; stock images by digest; --network none).  Not run by the
# package build.  Modeled on moe-plan-p3/harness/run_p3.sh.
#
#   ATTEMPT_REASON="..." CAPTURE_ATTEMPT=N replay/run_replay.sh          replay capture attempt N
#   ATTEMPT_REASON="..." CAPTURE_ATTEMPT=N DRY_RUN=1 replay/run_replay.sh gates and plan only; nothing started
#   FAKE=1 ...   CPU-only wrapper test: fake docker / nvidia-smi on PATH, scratch ROOT_OVERRIDE (tests/test_wrappers.sh)
#
# Budget: the SAME ledger and ceiling as the capture (costs.jsonl, 1.0 occupied GPU-h total incl. retries).  This
# attempt's wall cap = min(600 s, (1.0 - used) x 3600 s / 1 GPU); refused below 120 s.
# Order (priority): image v091 (attention replay, MoE baseline + plan comparison, forced-tile confirmation), then
# image v070 (MoE plan comparison on its ep2r0 slice) only if >= 120 s remain, then a CPU-only summary.
# One cost line per container (incl. skipped / interrupted); `docker rm -f` on every exit path.  No `set -e`.
set -u
umask 022
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT_OVERRIDE:-<runs>/captured-p3b}
COSTS=$ROOT/costs.jsonl
P3=${P3_OVERRIDE:-<runs>/moe-plan-p3}
MODEL_REPO=<models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1
MODEL_REV=0490d2f708b12145f6516555ab066aaeb401cd21
REPO=ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx
IMG_v091=$REPO@sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
IMG_v070=$REPO@sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629
P3_HARNESS_SHA=15ce45d8e0ab5caff09016f8f11d3c6c3e26b51499a9b22438b987a5d130b5a0
ITEM=p3b-replay
BUDGET_GH=1.0   # raised from 0.65 by the operator after attempt 1 (AUTHORIZATION.txt line 2)
CAP_MAX=600
MIN_S=120
V070_MIN_S=120
STOP_MARGIN_S=20
DRY_RUN=${DRY_RUN:-0}
ATTEMPT_REASON=${ATTEMPT_REASON:-}
CAPTURE_ATTEMPT=${CAPTURE_ATTEMPT:-}
SERVED_ENV=(
  -e VLLM_EXL3_TRELLIS_MIN_M=1 -e VLLM_EXL3_TRELLIS_MAX_M=32 -e VLLM_EXL3_TRELLIS_BLOCK_M=8
  -e VLLM_EXL3_PREFILL_TRELLIS=1 -e VLLM_EXL3_PREFILL_BLOCK_M=64 -e VLLM_EXL3_PREFILL_CAPACITY=1024
  -e B12X_EXL3_BF16_EPILOGUE=0 -e B12X_EXL3_BF16_GEMV=1
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_LAUNCH_BLOCKING=0 -e CUDA_MODULE_LOADING=LAZY
)
refuse() { echo "REFUSED: $*"; exit 2; }
abort()  { echo "ABORT: $*"; exit 3; }

[ -n "$ATTEMPT_REASON" ] || refuse "ATTEMPT_REASON is not set (say why this attempt runs)"
case "$CAPTURE_ATTEMPT" in ''|*[!0-9]*) refuse "CAPTURE_ATTEMPT=N (the capture attempt to replay) is required";; esac
CAPDIR=$ROOT/PRIVATE-capture/capture-attempt-$CAPTURE_ATTEMPT
CAPLOG=$ROOT/runs/capture-attempt-$CAPTURE_ATTEMPT/caplog
[ -d "$CAPDIR/moe" ] || [ -d "$CAPDIR/att" ] || abort "no captured tensors under $CAPDIR"
ls "$CAPLOG"/cap_*.jsonl >/dev/null 2>&1 || abort "no capture log under $CAPLOG"
[ -f "$P3/harness/moe_plan_eq.py" ] && [ "$(sha256sum "$P3/harness/moe_plan_eq.py" | cut -d' ' -f1)" = "$P3_HARNESS_SHA" ] \
  || abort "P3 harness missing or not the frozen one ($P3_HARNESS_SHA)"
if [ -e "$COSTS" ]; then [ -w "$COSTS" ] || abort "cannot write $COSTS"; fi

BUD=$(python3 - "$COSTS" "$BUDGET_GH" "$CAP_MAX" "$ITEM" <<'PYEOF'
import json, os, sys
path, budget, capmax, item = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
used_h = 0.0
attempts = set()
for n, line in enumerate(open(path) if os.path.exists(path) else [], 1):
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except Exception:
        print(f"ERR malformed ledger line {n}"); sys.exit(0)
    used_h += float(d["seconds"]) * float(d.get("gpus", 1)) / 3600.0 if d.get("seconds") is not None else float(d.get("gpu_hours") or 0)
    if d.get("item") == item:
        attempts.add(d.get("attempt"))
left_h = budget - used_h
print(f"OK {used_h:.4f} {left_h:.4f} {int(min(capmax, max(0.0, left_h) * 3600.0))} {len(attempts)}")
PYEOF
)
set -- $BUD
[ "${1:-}" = OK ] || abort "cannot read $COSTS: $BUD"
USED_GH=$2; LEFT_GH=$3; CAP=$4; PRIOR=$5
ATTEMPT=$((PRIOR + 1))
echo "budget: ceiling $BUDGET_GH GPU-h (capture + replay); used $USED_GH; left $LEFT_GH"
echo "replay attempt $ATTEMPT of capture attempt $CAPTURE_ATTEMPT: wall cap ${CAP}s x 1 GPU"
[ "$CAP" -ge "$MIN_S" ] || refuse "wall cap ${CAP}s < ${MIN_S}s (budget left $LEFT_GH GPU-h)"
AO=$ROOT/runs/replay-attempt-$ATTEMPT
PO=$ROOT/PRIVATE-capture/replay-attempt-$ATTEMPT
[ -e "$AO" ] && refuse "$AO exists (never overwritten)"
[ -e "$PO" ] && refuse "$PO exists (never overwritten)"
[ -d "$MODEL_REPO/snapshots/$MODEL_REV" ] || abort "model snapshot missing"
for e in v091 v070; do
  v=IMG_$e; docker image inspect "${!v}" >/dev/null 2>&1 || abort "image missing locally: ${!v} (no pulls)"
done
if docker ps --format '{{.Names}}' | grep -qE '^glm53|vllm|b0cache|mgtest|^p3-|^p3b-'; then
  docker ps --format '{{.Names}}'; abort "an inference or P3 container is running"
fi
PRE=$( { echo "== preflight $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
         nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv; } 2>&1 ) || abort "nvidia-smi failed"
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
CC=$ROOT/replay/compile-cache
mounts() {  # $1 engine
  echo -v "$HERE":/r:ro -v "$P3/harness":/p3h:ro -v "$P3/out/attempt-2":/p3out:ro -v "$MODEL_REPO":/model-repo:ro \
       -v "$CAPDIR":/cap:ro -v "$CAPLOG":/caplog:ro -v "$AO":/o -v "$PO":/po -v "$CC/$1":/cc
}
if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN: all gates passed; no container started, nothing written."
  echo "plan: attempt $ATTEMPT -> $AO (public) and $PO (private, 0700); wall cap ${CAP}s"
  echo "  v091: timeout <cap-left - ${STOP_MARGIN_S}s (minus ${V070_MIN_S}s for v070 when it fits)> docker run --rm --name p3b-r${ATTEMPT}-v091 --network none --gpus device=0 --user $UIDGID <served env> $(mounts v091) --entrypoint sh $IMG_v091 /r/in_container.sh v091 <soft> attn,moe,forced"
  echo "  v070: only if >= ${V070_MIN_S}s left: ... --entrypoint sh $IMG_v070 /r/in_container.sh v070 <soft> moe"
  echo "  then (CPU only, no GPU): replay/summarize_p3b.py $AO"
  exit 0
fi

mkdir -p "$AO" || abort "cannot create $AO"
( umask 077; mkdir -p "$PO" ) && chmod 700 "$ROOT/PRIVATE-capture" "$PO"
echo "$PRE" > "$AO/preflight.txt"
echo "replay attempt $ATTEMPT of capture attempt $CAPTURE_ATTEMPT; reason: $ATTEMPT_REASON" > "$AO/attempt.txt"
( cd "$ROOT" && sha256sum replay/*.py replay/*.sh ) > "$AO/scripts.sha256"
for e in v091 v070; do   # seed the compile cache from P3's (read only there); a cache hit cannot change results
  if [ ! -d "$CC/$e" ]; then mkdir -p "$CC/$e"; [ -d "$P3/out/compile-cache/$e" ] && cp -r "$P3/out/compile-cache/$e/." "$CC/$e/"; fi
done
T_ATT=$(date +%s); CURRENT=""; CUR_T0=0; CUR_START=""; COSTED=" "
cost_line() {  # engine seconds rc skipped reason start end timeout
  local e=$1 v=IMG_$1
  P3B_REASON="$ATTEMPT_REASON" python3 - "$COSTS" "$ITEM" "$ATTEMPT" "$CAPTURE_ATTEMPT" "$e" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "${!v##*@}" "$CAP" <<'PYEOF'
import json, os, sys
c, item, att, capatt, eng, secs, rc, skipped, reason, start, end, tmo, img, cap = sys.argv[1:]
line = dict(item=item, attempt=int(att), capture_attempt=int(capatt), attempt_reason=os.environ["P3B_REASON"], engine=eng,
            start=start, end=end, seconds=int(secs), gpus=1 if skipped != "true" else 0, gpu_hours=round(int(secs) / 3600.0, 4),
            image=img, rc=None if rc == "null" else int(rc), skipped=skipped == "true", skip_reason=reason or None,
            wall_cap_total=int(cap), image_timeout=int(tmo), purpose="captured-p3b offline replay (GPU0)")
open(c, "a").write(json.dumps(line) + "\n")
PYEOF
  COSTED="$COSTED$e "
}
on_exit() {
  for e in v091 v070; do docker rm -f "p3b-r${ATTEMPT}-$e" >/dev/null 2>&1; done
  if [ -n "$CURRENT" ] && [[ "$COSTED" != *" $CURRENT "* ]]; then
    cost_line "$CURRENT" $(( $(date +%s) - CUR_T0 )) 130 false "wrapper interrupted" "$CUR_START" "$(date -u +%FT%TZ)" 0
  fi
  [ "$COSTED" = " " ] && cost_line v091 0 null true "attempt aborted before any GPU container" "$(date -u +%FT%TZ)" "$(date -u +%FT%TZ)" 0
}
trap on_exit EXIT
trap 'echo "interrupted"; exit 130' INT TERM

run_image() {  # engine parts reserve_after min_left
  local e=$1 parts=$2 reserve=$3 minleft=$4 v=IMG_$1 now LEFT TMO SOFT name
  now=$(date +%s); LEFT=$((CAP - (now - T_ATT)))
  TMO=$((LEFT - STOP_MARGIN_S - reserve))
  if [ "$TMO" -lt 60 ] || [ "$LEFT" -lt "$minleft" ]; then
    echo "SKIP $e: ${LEFT}s left of ${CAP}s"
    cost_line "$e" 0 null true "skipped: ${LEFT}s left of ${CAP}s cap" "$(date -u +%FT%TZ)" "$(date -u +%FT%TZ)" 0
    return 0
  fi
  SOFT=$((TMO - 25))
  name=p3b-r${ATTEMPT}-$e
  CURRENT=$e; CUR_START=$(date -u +%FT%TZ); CUR_T0=$(date +%s)
  echo "== $e start $CUR_START timeout ${TMO}s soft ${SOFT}s parts $parts"
  # shellcheck disable=SC2046
  timeout --kill-after=20 "$TMO" docker run --rm --name "$name" --network none --gpus device=0 --user "$UIDGID" \
    -e HOME=/tmp -e B12X_COMPILE_CACHE_DIR=/cc "${SERVED_ENV[@]}" $(mounts "$e") \
    --entrypoint sh "${!v}" /r/in_container.sh "$e" "$SOFT" "$parts" > "$AO/$e.log" 2>&1
  local RC=$?
  docker rm -f "$name" >/dev/null 2>&1
  cost_line "$e" $(( $(date +%s) - CUR_T0 )) "$RC" false "" "$CUR_START" "$(date -u +%FT%TZ)" "$TMO"
  CURRENT=""
  echo "$e rc=$RC, $(( $(date +%s) - CUR_T0 ))s x 1 GPU"; grep -E '^\{"part"|elapsed' "$AO/$e.log" | tail -4
}
R=0; [ "$CAP" -ge $((V070_MIN_S * 2 + 60)) ] && R=$V070_MIN_S
PARTS_V091=${PARTS_V091:-attn,moe,forced}   # replay 2: attention only (MoE/forced complete in replay 1)
run_image v091 "$PARTS_V091" "$R" 0
[ "${SKIP_V070:-0}" = 1 ] || run_image v070 moe 0 $((V070_MIN_S + STOP_MARGIN_S))
python3 "$HERE/summarize_p3b.py" "$AO" "$ROOT/runs/capture-attempt-$CAPTURE_ATTEMPT" "$P3/out/attempt-2" 2>&1 | tee "$AO/summary-console.txt"
python3 -c "
import json
s=sum(float(json.loads(l)['seconds'])*float(json.loads(l).get('gpus',1)) for l in open('$COSTS') if l.strip())
print(f'budget used so far (capture + replay): {s/3600:.4f} GPU-h of $BUDGET_GH')"
