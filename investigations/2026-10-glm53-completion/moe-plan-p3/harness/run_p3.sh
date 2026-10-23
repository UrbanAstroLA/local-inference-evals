#!/bin/bash
# P3 Stage 1 GPU wrapper: moe_plan_eq.py on GPU0 only, stock images by digest,
# --network none, model snapshot read-only.  Modeled on
# ../../livecheck-22ao/level3/run_level3.sh.
#
# Authorization: ../AUTHORIZATION.txt (GPU0 only, <= 0.5 occupied GPU-hours incl.
# load/plan/compile).  Retries are allowed (AUTHORIZATION.txt line 2, operator
# 2026-10-23: "retry to test properly is approved") under a budget gate: the SUM
# of gpu_hours already in ../costs.jsonl plus this attempt's wall cap must stay
# <= 0.5 GPU-h; each attempt is logged with its reason.
#
#   ATTEMPT_REASON="..." harness/run_p3.sh            full run: v091 then v070
#   ATTEMPT_REASON="..." SMOKE=1 harness/run_p3.sh    v091 only, layer 3, few fixtures
#   ATTEMPT_REASON="..." DRY_RUN=1 harness/run_p3.sh  all gates + plan, no GPU container
#
# Optional: SMOKE_CAP (default 420 s).  Not overridable: the 0.5 GPU-h budget,
# the 1800 s per-attempt ceiling, the 300 s minimum.
# No `set -e`: every status query is checked explicitly.
set -u
umask 022

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
COSTS="$ROOT/costs.jsonl"
OUTROOT="$ROOT/out"
MODEL_REPO=<models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1
MODEL_REV=0490d2f708b12145f6516555ab066aaeb401cd21
REPO=ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx
IMG_v091=$REPO@sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
IMG_v070=$REPO@sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629
BUDGET_GPU_H=0.5        # operator ceiling, all attempts together
ATTEMPT_MAX_S=1800      # per-attempt ceiling (0.5 GPU-h x 1 GPU)
MIN_S=300               # refuse to start (or skip an image) below this
STOP_MARGIN_S=30        # docker stop/rm headroom inside the cap
SOFT_MARGIN_S=60        # harness stops starting new work this long before its timeout
FIRST_SHARE_PCT=55      # full mode: the first image's SOFT deadline is <= 55% of the cap, so the
                        # second image's own-slice run is not starved by the first image's
                        # optional E-version work (the hard timeout is still the whole remainder)
DRY_RUN=${DRY_RUN:-0}
SMOKE=${SMOKE:-0}
SMOKE_CAP=${SMOKE_CAP:-420}
ATTEMPT_REASON=${ATTEMPT_REASON:-}

# Served env (README "Served values"); passed explicitly although both images bake the same defaults.
SERVED_ENV=(
  -e VLLM_EXL3_TRELLIS_MIN_M=1 -e VLLM_EXL3_TRELLIS_MAX_M=32 -e VLLM_EXL3_TRELLIS_BLOCK_M=8
  -e VLLM_EXL3_PREFILL_TRELLIS=1 -e VLLM_EXL3_PREFILL_BLOCK_M=64 -e VLLM_EXL3_PREFILL_CAPACITY=1024
  -e B12X_EXL3_BF16_EPILOGUE=0 -e B12X_EXL3_BF16_GEMV=1
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_LAUNCH_BLOCKING=0 -e CUDA_MODULE_LOADING=LAZY
)

refuse() { echo "REFUSED: $*"; exit 2; }
abort()  { echo "ABORT: $*"; exit 3; }

# ---------------------------------------------------------------- gates (no GPU)
[ -n "$ATTEMPT_REASON" ] || refuse "ATTEMPT_REASON is not set (say why this attempt runs)"
case "$DRY_RUN$SMOKE" in 00|01|10|11) ;; *) refuse "DRY_RUN and SMOKE must be 0 or 1";; esac
case "$SMOKE_CAP" in ''|*[!0-9]*) refuse "SMOKE_CAP must be an integer number of seconds";; esac
command -v python3 >/dev/null 2>&1 || abort "python3 (host) is needed to read costs.jsonl"
if [ -e "$COSTS" ]; then [ -w "$COSTS" ] || abort "cannot write $COSTS"; else [ -w "$ROOT" ] || abort "cannot write $ROOT"; fi

# Budget: seconds already used (prefer the seconds field), distinct attempts so far.
BUDGET_OUT=$(python3 - "$COSTS" "$BUDGET_GPU_H" <<'PYEOF'
import json, sys
import os
path, budget_h = sys.argv[1], float(sys.argv[2])
used_s = 0.0
attempts = set()
for n, line in enumerate(open(path) if os.path.exists(path) else [], 1):
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except Exception:
        print(f"ERR malformed line {n}")
        sys.exit(0)
    if d.get("seconds") is not None:
        used_s += float(d["seconds"]) * float(d.get("gpus", 1))
    else:
        used_s += float(d.get("gpu_hours") or 0.0) * 3600.0
    attempts.add(d.get("attempt", f"legacy-{d.get('start', n)}"))
remaining = int(budget_h * 3600.0 - used_s)
print(f"OK {used_s:.0f} {remaining} {len(attempts)}")
PYEOF
)
set -- $BUDGET_OUT
[ "${1:-}" = OK ] || abort "cannot read $COSTS: $BUDGET_OUT"
USED_S=$2; REMAINING_S=$3; PRIOR_ATTEMPTS=$4
ATTEMPT=$((PRIOR_ATTEMPTS + 1))
if [ "$SMOKE" = 1 ]; then MODE=smoke; WANT=$SMOKE_CAP; else MODE=full; WANT=$ATTEMPT_MAX_S; fi
[ "$WANT" -le "$ATTEMPT_MAX_S" ] || WANT=$ATTEMPT_MAX_S
CAP=$WANT; [ "$REMAINING_S" -lt "$CAP" ] && CAP=$REMAINING_S
echo "budget: ${BUDGET_GPU_H} GPU-h; used ${USED_S} s over ${PRIOR_ATTEMPTS} prior attempt(s); remaining ${REMAINING_S} s"
echo "attempt ${ATTEMPT} (${MODE}): wall cap ${CAP} s (requested ${WANT} s)"
[ "$REMAINING_S" -ge "$MIN_S" ] || refuse "budget exhausted: ${REMAINING_S} s remaining < ${MIN_S} s"
[ "$CAP" -ge "$MIN_S" ] || refuse "wall cap ${CAP} s < ${MIN_S} s"

ADIR="$OUTROOT/attempt-$ATTEMPT"
[ -e "$ADIR" ] && refuse "$ADIR already exists (attempt numbering out of step with costs.jsonl; never overwritten)"

[ -d "$MODEL_REPO/snapshots/$MODEL_REV" ] || abort "model snapshot missing: $MODEL_REPO/snapshots/$MODEL_REV"
[ -f "$HERE/moe_plan_eq.py" ] || abort "harness missing"
if [ "$MODE" = smoke ]; then ENGINES=(v091); else ENGINES=(v091 v070); fi
declare -A LABEL
for e in "${ENGINES[@]}"; do
  img_var=IMG_$e; img=${!img_var}
  docker image inspect "$img" >/dev/null 2>&1 || abort "image missing locally: $img (no pulls)"
  LABEL[$e]=$(docker image inspect "$img" --format '{{index .Config.Labels "io.tpurtell.b12x.commit"}}' 2>/dev/null)
  echo "image $e: ${img##*@} b12x ${LABEL[$e]}"
done
if docker ps --format '{{.Names}}' | grep -qE '^glm53|vllm|b0cache|mgtest|^p3-'; then
  docker ps --format '{{.Names}}'
  abort "an inference or P3 container is running"
fi

PRE=$(mktemp) || abort "mktemp failed"
trap 'rm -f "$PRE"' EXIT
{
  echo "== preflight $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
} >"$PRE" 2>&1 || abort "nvidia-smi failed: $(tail -2 "$PRE")"
ok=0
for try in $(seq 1 12); do
  read -r MEM UTIL < <(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
  case "${MEM:-x}${UTIL:-x}" in *[!0-9]*) abort "cannot read GPU0 state (mem='${MEM:-}' util='${UTIL:-}')";; esac
  if [ "$MEM" -gt 1024 ]; then cat "$PRE"; abort "GPU0 memory ${MEM} MiB > 1024 MiB"; fi
  echo "GPU0 sample $try: util ${UTIL}% mem ${MEM} MiB" >>"$PRE"
  [ "$UTIL" -le 5 ] && { ok=1; break; }
  [ "$DRY_RUN" = 1 ] || sleep 5
done
cat "$PRE"
[ "$ok" = 1 ] || abort "GPU0 busy after 12 samples"

UIDGID="$(id -u):$(id -g)"
if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN: all gates passed; no container started, nothing written."
  echo "plan: attempt ${ATTEMPT} -> $ADIR; total wall cap ${CAP} s; remaining budget ${REMAINING_S} s"
  for e in "${ENGINES[@]}"; do
    img_var=IMG_$e
    echo "  $e: timeout <cap-left - ${STOP_MARGIN_S}> docker run --rm --name p3-a${ATTEMPT}-$e --network none --gpus device=0" \
         "--user $UIDGID -e HOME=/tmp <served env> -v $HERE:/h:ro -v $ADIR:/o -v $MODEL_REPO:/model-repo:ro" \
         "-v $OUTROOT/compile-cache/$e:/cc --entrypoint python3 ${!img_var##*@} /h/moe_plan_eq.py --engine $e --out /o/$e.json" \
         "$([ "$MODE" = smoke ] && echo --smoke) --deadline-s <timeout - ${SOFT_MARGIN_S}$([ "$MODE" = full ] && [ "$e" = v091 ] && echo ", <= ${FIRST_SHARE_PCT}% of cap")>"
  done
  [ "$MODE" = full ] && echo "  then (CPU only, no GPU): compare_eversion.py v091.json v070.json eversion.json"
  exit 0
fi

# ---------------------------------------------------------------- the attempt
mkdir -p "$OUTROOT" || abort "cannot create $OUTROOT"
mkdir "$ADIR" || refuse "cannot create $ADIR (exists?)"
cp "$PRE" "$ADIR/preflight.txt"
for e in "${ENGINES[@]}"; do mkdir -p "$OUTROOT/compile-cache/$e"; done
T_ATTEMPT=$(date +%s)
CURRENT=""; CUR_T0=0; CUR_START=""; CUR_TIMEOUT=0
COSTED=" "

cost_line() {  # engine seconds rc skipped reason start end timeout
  local e=$1 secs=$2 rc=$3 skipped=$4 reason=$5 start=$6 end=$7 tmo=$8 img_var=IMG_$1
  P3_REASON="$ATTEMPT_REASON" python3 - "$COSTS" "$ATTEMPT" "$MODE" "$e" "$start" "$end" "$secs" \
    "${!img_var##*@}" "${LABEL[$e]:-}" "$rc" "$skipped" "$reason" "$CAP" "$tmo" <<'PYEOF'
import json, os, sys
(costs, attempt, mode, engine, start, end, secs, image, b12x, rc, skipped, reason,
 cap, tmo) = sys.argv[1:]
line = dict(item="p3-stage1-moe-plan", attempt=int(attempt), attempt_reason=os.environ["P3_REASON"],
            mode=mode, engine=engine, start=start, end=end, seconds=int(secs), gpus=1,
            gpu_hours=round(int(secs) / 3600.0, 4), image=image, b12x_commit=b12x,
            rc=None if rc == "null" else int(rc), skipped=skipped == "true",
            skip_reason=reason or None, wall_cap_total=int(cap), image_timeout=int(tmo),
            out=f"out/attempt-{attempt}/{engine}.json")
with open(costs, "a") as fh:
    fh.write(json.dumps(line) + "\n")
PYEOF
  COSTED="$COSTED$e "
}

on_exit() {
  rm -f "$PRE"
  for e in "${ENGINES[@]}"; do docker rm -f "p3-a${ATTEMPT}-$e" >/dev/null 2>&1; done
  if [ -n "$CURRENT" ] && [[ "$COSTED" != *" $CURRENT "* ]]; then
    local t1; t1=$(date +%s)
    cost_line "$CURRENT" $((t1 - CUR_T0)) 130 false "wrapper interrupted" "$CUR_START" "$(date -u +%FT%TZ)" "$CUR_TIMEOUT"
  fi
  # keep attempt numbering in step with out/attempt-N even if nothing ran
  if [ "$COSTED" = " " ]; then
    cost_line "${ENGINES[0]}" 0 null true "attempt aborted before any GPU container" \
      "$(date -u +%FT%TZ)" "$(date -u +%FT%TZ)" 0
  fi
}
trap on_exit EXIT
trap 'echo "interrupted"; exit 130' INT TERM

for e in "${ENGINES[@]}"; do
  img_var=IMG_$e; img=${!img_var}
  now=$(date +%s); LEFT=$((CAP - (now - T_ATTEMPT)))
  if [ "$LEFT" -lt "$MIN_S" ]; then
    echo "SKIP $e: ${LEFT} s left of the ${CAP} s cap (< ${MIN_S} s)"
    cost_line "$e" 0 null true "skipped: ${LEFT} s left of ${CAP} s cap" "$(date -u +%FT%TZ)" "$(date -u +%FT%TZ)" 0
    continue
  fi
  TMO=$((LEFT - STOP_MARGIN_S)); SOFT=$((TMO - SOFT_MARGIN_S))
  if [ "$MODE" = full ] && [ "$e" = "${ENGINES[0]}" ]; then
    SHARE=$((CAP * FIRST_SHARE_PCT / 100)); [ "$SOFT" -gt "$SHARE" ] && SOFT=$SHARE
  fi
  EXTRA=(); [ "$MODE" = smoke ] && EXTRA=(--smoke)
  name="p3-a${ATTEMPT}-$e"
  CURRENT=$e; CUR_START=$(date -u +%FT%TZ); CUR_T0=$(date +%s); CUR_TIMEOUT=$TMO
  echo "== $e start $CUR_START timeout ${TMO} s soft ${SOFT} s"
  timeout --kill-after=20 "$TMO" docker run --rm --name "$name" --network none --gpus device=0 \
    --user "$UIDGID" -e HOME=/tmp -e B12X_COMPILE_CACHE_DIR=/cc "${SERVED_ENV[@]}" \
    -v "$HERE":/h:ro -v "$ADIR":/o -v "$MODEL_REPO":/model-repo:ro -v "$OUTROOT/compile-cache/$e":/cc \
    --entrypoint python3 "$img" /h/moe_plan_eq.py --engine "$e" --out "/o/$e.json" \
    --deadline-s "$SOFT" "${EXTRA[@]}" >"$ADIR/$e.log" 2>&1
  RC=$?
  docker rm -f "$name" >/dev/null 2>&1
  T1=$(date +%s)
  cost_line "$e" $((T1 - CUR_T0)) "$RC" false "" "$CUR_START" "$(date -u +%FT%TZ)" "$TMO"
  CURRENT=""
  echo "$e rc=$RC, $((T1 - CUR_T0)) s x 1 GPU"
  grep -v -i 'warn' "$ADIR/$e.log" | tail -3
done

# CPU only, not GPU time: E-version comparison inside the 0.9.1 image without --gpus.
if [ "$MODE" = full ] && [ -f "$ADIR/v091.json" ] && [ -f "$ADIR/v070.json" ]; then
  timeout 600 docker run --rm --name "p3-a${ATTEMPT}-cmp" --network none -e CUDA_VISIBLE_DEVICES= \
    --user "$UIDGID" -e HOME=/tmp -v "$HERE":/h:ro -v "$ADIR":/o --entrypoint python3 "$IMG_v091" \
    /h/compare_eversion.py /o/v091.json /o/v070.json /o/eversion.json >"$ADIR/eversion.log" 2>&1
  echo "eversion compare rc=$? (CPU only)"
  docker rm -f "p3-a${ATTEMPT}-cmp" >/dev/null 2>&1
fi
USED_NOW=$(python3 -c "
import json
s=sum(float(json.loads(l).get('seconds') or 0) for l in open('$COSTS') if l.strip())
print(f'{s:.0f} s = {s/3600:.4f} GPU-h of $BUDGET_GPU_H')")
echo "attempt $ATTEMPT done; budget used so far: $USED_NOW"
