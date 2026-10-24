#!/bin/bash
# Stage-0 determinism probe (STAGE0-PROTOCOL.md): ONE serving session on 2 GPUs, arm A only (GLM53_P4_TILE unset),
# one batch of 8 windows, V1 solitary x 2 then V2 co-batched x 2, each replicate independently warmed under its own
# cache salt.  Image local/glm53-v091-p4-s0:local (stage-0 recording: GLM53_P4_S0=on).  Not run by the package build.
#
#   APPROVED_GPU_H=<operator approval> ATTEMPT_REASON="..." S0_PROTOCOL_ACK=<12 hex> runner/stage0/run_stage0.sh
#   ... DRY_RUN=1 ...     gates and plan only; starts nothing, writes nothing
#   MOCK=1 ...            CPU simulation against tests/mock_server_stage0.py (no docker container, no GPU; MOCK_ROOT
#                         must be under tests/out/)
#
# Budget (coordinator 2026-10-24; AUTHORIZATION.txt: the 7.0 ceiling is VOID): runner/stage0/budget_stage0.py refuses
# unless APPROVED_GPU_H is set explicitly (no default, no fallback) and refuses when this attempt's cap plus the GPU-h
# already in costs-stage0.jsonl (the new approval's own ledger) would exceed it.  Run-1 costs.jsonl is not touched.
# Unattended-safe: ATTEMPT_REASON, protocol acknowledgement, budget/order gate (needs a passed stage-0 pre-check),
# image pin, frozen hashes, fixture hash, pre-flight, start-up gates (hooks + stage-0 recording registered on both
# ranks before capture, entry-digest coverage, G3 vs captured-p3b) BEFORE any traffic, client deadline, flush wait,
# wall-cap watchdog, stop on every exit path, one cost line per attempt counted to the GPU release, copy-out, then the
# stage-0 analysis in a CPU container.  No `set -e`.
set -u
umask 022
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT_OVERRIDE:-<runs>/tile-p4}
[ -d "$ROOT" ] || { echo "path check failed"; exit 2; }
INV=<runs>
GPUS=2
GATE_CAP=${GATE_CAP:-960}
RESERVE=${RESERVE:-150}            # wall kept after the client: flush wait, logs, stop.sh, docker stop
ST=<stacks>/glm53-flash-v0.9.0
RC=<recipes>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1
CN=glm53-flash-exl3-b12x-vllm
IMG=local/glm53-v091-p4-s0:local
P3B_CAPLOG=$INV/captured-p3b/runs/capture-attempt-2/caplog/cap_p370.jsonl
MOCK=${MOCK:-0}
DRY_RUN=${DRY_RUN:-0}
ATTEMPT_REASON=${ATTEMPT_REASON:-}
BROOT=$ROOT
if [ "$MOCK" = 1 ]; then
  BROOT=${MOCK_ROOT:-$ROOT/tests/out/stage0-mock-root}
  case "$BROOT" in "$ROOT"/tests/out/*) ;; *) echo "REFUSED: MOCK_ROOT outside tests/out"; exit 2;; esac
fi
COSTS=$BROOT/costs-stage0.jsonl
OUTROOT=$BROOT/runs-stage0
PRIVROOT=$BROOT/PRIVATE-runs-stage0
WINDOWS=$BROOT/PRIVATE-fixtures/windows.json
SCHED=$BROOT/PRIVATE-fixtures/sched.json
BASE=http://127.0.0.1:8001
[ "$MOCK" = 1 ] && { BASE=http://127.0.0.1:${MOCK_PORT:-18051}; GATE_CAP=${GATE_CAP_MOCK:-20}; RESERVE=${RESERVE_MOCK:-15}; }
DEBUG="GLM53_P4=on GLM53_P4_S0=on GLM53_P4_DIR=/tmp/p4 FORCEWIN=on FORCEWIN_DIR=/tmp/p4/PRIVATE-fw FORCEWIN_WMAX=520 FORCEWIN_NDESIG=1 FORCEWIN_NSEL=1"

refuse() { echo "REFUSED: $*"; exit 2; }
abort()  { echo "ABORT: $*"; exit 3; }

[ -n "$ATTEMPT_REASON" ] || refuse "ATTEMPT_REASON is not set (say why this attempt runs)"
case "$DRY_RUN$MOCK" in 00|01|10|11) ;; *) refuse "DRY_RUN and MOCK must be 0 or 1";; esac
[ -n "${APPROVED_GPU_H:-}" ] || refuse "APPROVED_GPU_H is not set: stage 0 needs an explicit operator approval in GPU-hours (no default; the 7.0 ceiling is VOID)"
PROTO_SHA=$(sha256sum "$ROOT/STAGE0-PROTOCOL.md" 2>/dev/null | cut -c1-12)
[ -n "$PROTO_SHA" ] || refuse "STAGE0-PROTOCOL.md missing"
[ "${S0_PROTOCOL_ACK:-}" = "$PROTO_SHA" ] || refuse "S0_PROTOCOL_ACK must be $PROTO_SHA (first 12 hex of sha256(STAGE0-PROTOCOL.md)): the operator accepts the stage-0 protocol"
if [ -e "$COSTS" ]; then [ -w "$COSTS" ] || abort "cannot write $COSTS"; else [ -w "$(dirname "$COSTS")" ] || abort "cannot write $(dirname "$COSTS")"; fi

BUD=$(APPROVED_GPU_H="$APPROVED_GPU_H" python3 "$HERE/budget_stage0.py" "$BROOT" stage0)
set -- $BUD
[ "${1:-}" = OK ] || refuse "budget/order gate: $BUD"
USED_GH=$2; LEFT_GH=$3; CAP=$4; ATTEMPT=$5
echo "budget: approved $APPROVED_GPU_H GPU-h (costs-stage0.jsonl); used $USED_GH; left $LEFT_GH; stage0 attempt $ATTEMPT: wall cap ${CAP}s x $GPUS GPUs"
AO=$OUTROOT/stage0-attempt-$ATTEMPT
PO=$PRIVROOT/stage0-attempt-$ATTEMPT
[ -e "$AO" ] && refuse "$AO exists (attempt numbering out of step with the ledger; never overwritten)"
[ -e "$PO" ] && refuse "$PO exists (never overwritten)"

if [ "$MOCK" != 1 ]; then
  SHAOUT=$( cd "$ROOT" && sha256sum -c --quiet SHA256SUMS 2>&1 ) || { echo "$SHAOUT"; abort "frozen hashes differ (SHA256SUMS)"; }
  ID=$(docker image inspect "$IMG" --format '{{.Id}}' 2>/dev/null)
  EXP=$(awk -v t="$IMG" '$1==t {print $2}' "$HERE/images.txt")
  [ -n "$EXP" ] && [ "$ID" = "$EXP" ] || abort "$IMG is '$ID', expected '$EXP' (runner/stage0/images.txt)"
  [ -d "$ST" ] && [ -f "$RC/env.sh" ] || abort "stack or recipe missing"
  [ -f "$P3B_CAPLOG" ] || abort "captured-p3b capture log (G3 reference) missing"
  FX_EXP=$(python3 -c "import json;print(json.load(open('$ROOT/windows/out/window-manifest.json'))['private_fixture_sha256'])")
  [ "$(sha256sum "$WINDOWS" 2>/dev/null | cut -d' ' -f1)" = "$FX_EXP" ] || abort "windows.json hash differs from windows/out/window-manifest.json"
fi
[ -f "$WINDOWS" ] && [ -f "$SCHED" ] || abort "fixtures missing"
if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN: ATTEMPT_REASON, APPROVED_GPU_H, protocol ack, budget/order gate, attempt numbering, frozen hashes, image pin and fixtures passed."
  echo "plan: stage0 attempt $ATTEMPT -> $AO (logs) and $PO (private, 0700); wall cap ${CAP}s; readiness by ${GATE_CAP}s;"
  echo "      client deadline T0+$((CAP - RESERVE))s; watchdog at T0+$((CAP - 60))s; image $IMG; DEBUG_ENV '$DEBUG'"
  P4_WINDOWS="$WINDOWS" P4_SCHED="$SCHED" P4_PLAN_ONLY=1 python3 "$HERE/client_stage0.py" x 0 x
  exit 0
fi

mkdir -p "$AO" || abort "cannot create $AO"
( cd "$ROOT" && sha256sum runner/stage0/*.py runner/stage0/*.sh runner/stage0/images.txt build/files/vllm/p4_hooks.py \
    build/files/vllm/forcewin_hooks.py analysis/analyze_stage0.py STAGE0-PROTOCOL.md ) > "$AO/scripts.sha256" 2>/dev/null
echo "stage0 attempt $ATTEMPT reason: $ATTEMPT_REASON; approved $APPROVED_GPU_H GPU-h; protocol ack $PROTO_SHA" > "$AO/attempt.txt"

preflight() {
  [ "$MOCK" = 1 ] && return 0
  echo "== preflight $(date -u +%FT%TZ)" | tee -a "$AO/preflight.txt"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a "$AO/preflight.txt"
  docker ps --format '{{.Names}} {{.Image}} {{.Status}}' | tee -a "$AO/preflight.txt"
  if docker ps --format '{{.Names}}' | grep -qE '^glm53|b0cache|mgtest|^p3-|^p3b-|^p4-'; then echo "ABORT: inference/fixture container running"; return 3; fi
  local PROCS
  PROCS=$(nvidia-smi 2>/dev/null | awk '/Processes:/{p=1;next} p && /^\|/ && $2 ~ /^[0-9]+$/ {print}' \
    | grep -vE '[[:space:]]G[[:space:]].*(gnome-shell|Xwayland|xdg-desktop-portal|firefox|snap-store)' \
    | grep -vE '[[:space:]]C\+G[[:space:]].*--disable-logging')
  if [ -n "$PROCS" ]; then echo "ABORT: GPU clients other than desktop type-G graphics:"; echo "$PROCS"; return 3; fi
  local IDX MEM UTIL LIM busy try
  for try in $(seq 1 12); do
    busy=""
    while read -r IDX MEM UTIL; do
      LIM=1024; [ "$IDX" = 1 ] && LIM=4096
      if [ "$MEM" -gt "$LIM" ]; then echo "ABORT: GPU$IDX busy mem=${MEM}MiB util=${UTIL}%"; return 3; fi
      [ "$UTIL" -gt 5 ] && busy="$busy GPU$IDX:${UTIL}%"
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
    echo "utilization sample $try:${busy:- idle}" | tee -a "$AO/preflight.txt"
    [ -z "$busy" ] && return 0
    sleep 5
  done
  echo "ABORT: utilization above 5% in 12 samples:$busy"; return 3
}

MOCKCTR=$AO/mock-ctr
c_cat() { if [ "$MOCK" = 1 ]; then cat "$MOCKCTR/$1" 2>/dev/null; else docker exec "$CN" sh -c "cat /tmp/p4/$1" 2>/dev/null; fi; }
c_cp()  { if [ "$MOCK" = 1 ]; then [ -e "$MOCKCTR/$1" ] && cp -r "$MOCKCTR/$1/." "$2/" 2>/dev/null; else docker cp "$CN:/tmp/p4/$1/." "$2/" >/dev/null 2>&1; fi; }
cost_line() {  # secs rc note
  S0_REASON="$ATTEMPT_REASON" python3 - "$COSTS" "$ATTEMPT" "${START:-}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$1" "$GPUS" "${IMG_ID:-}" "$2" "$CAP" "$MOCK" "$APPROVED_GPU_H" "$3" <<'PYEOF'
import json, os, sys
c, att, start, end, secs, gpus, img, rc, cap, mock, appr, note = sys.argv[1:]
secs = int(secs)
line = dict(item="s0-stage0", attempt=int(att), attempt_reason=os.environ["S0_REASON"], start=start or end, end=end,
            seconds=secs, gpus=int(gpus) if secs > 0 else 0, gpu_hours=round(int(gpus) * secs / 3600, 4) if secs > 0 else 0.0,
            image=img or None, rc=None if rc == "null" else int(rc), wall_cap=int(cap), mock=mock == "1",
            approved_gpu_h=float(appr), note=note or None,
            purpose="P4 stage-0 determinism probe (arm A, one batch, V1 + V2, stock 0.9.1 + forcewin port + local-p4 s0)")
open(c, "a").write(json.dumps(line) + "\n")
PYEOF
}

START=""; T0=0; RC_MAIN=""; WD=""; ACTIVE=0; IMG_ID=""; CPID=""; STAGE="start"
cleanup() {
  [ "$ACTIVE" = 1 ] || return 0
  ACTIVE=0
  [ -n "$WD" ] && kill "$WD" 2>/dev/null
  [ -n "$CPID" ] && kill "$CPID" 2>/dev/null
  echo "== cleanup $(date -u +%FT%TZ) (stage $STAGE)"
  if [ "$MOCK" = 1 ]; then
    [ -n "${MOCKPID:-}" ] && { kill "$MOCKPID" 2>/dev/null; wait "$MOCKPID" 2>/dev/null; }
  else
    if [ "$(docker inspect --format '{{.State.Running}}' "$CN" 2>/dev/null)" = true ]; then
      docker logs "$CN" > "$AO/server.log" 2>&1
      docker exec "$CN" sh -c 'cd /usr/local/lib/python3.12/dist-packages/vllm && sha256sum p4_hooks.py forcewin_hooks.py model_executor/layers/quantization/exl3.py models/glm5next/nvidia/model.py v1/worker/gpu/model_runner.py v1/worker/gpu/spec_decode/rejection_sampler.py v1/worker/gpu/sample/sampler.py; env | grep -E "^(GLM53|VLLM|FORCEWIN|B12X)" | sort' > "$AO/container-files-env.txt" 2>&1
    fi
    ( cd "$ST" && ./stop.sh ) >> "$AO/stop.log" 2>&1
    docker ps --format '{{.Names}}' | grep -q "^$CN\$" && docker stop -t 30 "$CN" >> "$AO/stop.log" 2>&1
  fi
  local SECS=$(( $(date +%s) - T0 ))
  cost_line "$SECS" "${RC_MAIN:-null}" "stage $STAGE"
  echo "stage0 attempt $ATTEMPT occupied ${SECS}s x $GPUS GPU = $(python3 -c "print(round($GPUS*$SECS/3600,4))") GPU-h (rc ${RC_MAIN:-null})"
  mkdir -p "$AO/p4log"
  c_cp log "$AO/p4log"
  ( umask 077; mkdir -p "$PO/p4" "$PO/fw" ) && chmod 700 "$PRIVROOT" "$PO" "$PO/p4" "$PO/fw"
  ( umask 077; c_cp PRIVATE "$PO/p4"; c_cp PRIVATE-fw "$PO/fw" )
  chmod -R go-rwx "$PO" 2>/dev/null
  echo "private files: $(find "$PO" -type f | wc -l) under $PO (mode 0700)"
  python3 "$ROOT/runner/gates_p4.py" --arm A --binds "$AO/p4log" --p3b "$P3B_CAPLOG" --out "$AO/gates-final.json" --stage final 2>&1 | tee -a "$AO/gate.txt"
  python3 "$HERE/outcome_stage0.py" "$AO" "$ATTEMPT" 2>&1 | tee "$AO/outcome-console.txt"
  if [ "$STAGE" = flush ] || [ "$STAGE" = done ]; then
    ROOT_OVERRIDE="$BROOT" "$ROOT/analysis/run_stage0_analysis.sh" "$AO" 2>&1 | tail -n 12 | tee "$AO/analysis-console.txt"
  fi
}
on_exit() { cleanup; }
trap on_exit EXIT
trap 'echo "signal"; exit 4' INT TERM

if ! preflight; then
  echo "preflight failed" | tee "$AO/refused.txt"
  START=$(date -u +%Y-%m-%dT%H:%M:%SZ); cost_line 0 3 "refused by pre-flight; no container started"
  exit 3
fi
START=$(date -u +%Y-%m-%dT%H:%M:%SZ); T0=$(date +%s); ACTIVE=1
[ "$MOCK" = 1 ] || IMG_ID=$(docker image inspect "$IMG" --format '{{.Id}}' 2>/dev/null)
( trap 'kill "$SP" 2>/dev/null; exit 0' TERM; sleep $(( CAP > 300 ? CAP - 60 : CAP )) & SP=$!; wait "$SP" || exit 0; \
  echo "WALL CAP reached"; \
  if [ "$MOCK" = 1 ]; then pkill -f "tests/mock_server_stage0[.]py" 2>/dev/null; else docker stop -t 30 "$CN" >/dev/null 2>&1; fi ) &
WD=$!
echo "== start $START image=$IMG ($IMG_ID) cap ${CAP}s stage0 attempt $ATTEMPT"
if [ "$MOCK" = 1 ]; then
  mkdir -p "$MOCKCTR"
  python3 "$ROOT/tests/mock_server_stage0.py" "$MOCKCTR" "${MOCK_PORT:-18051}" > "$AO/mock-server.log" 2>&1 &
  MOCKPID=$!; sleep 1
else
  ( unset EXTRA_SERVE_ARGS; cd "$ST" && source "$RC/env.sh" && export SPECULATIVE_METHOD=dflash2 DFLASH_TOKENS=3 \
      IMAGE="$IMG" DEBUG_ENV="$DEBUG" && ./start.sh ) > "$AO/start.log" 2>&1
  SRC=$?
  if [ $SRC -ne 0 ]; then echo "start failed rc=$SRC"; RC_MAIN=5; exit 5; fi
fi
STAGE="readiness"
READY=0
while [ $(( $(date +%s) - T0 )) -lt "$GATE_CAP" ]; do
  if [ "$MOCK" != 1 ]; then
    RUN=$(docker inspect --format '{{.State.Running}} {{.RestartCount}}' "$CN" 2>/dev/null)
    if [ "$RUN" != "true 0" ]; then echo "container state '$RUN'"; RC_MAIN=6; exit 6; fi
    if docker logs "$CN" 2>&1 | grep -qE 'Engine core initialization failed|CUDA out of memory|Traceback'; then
      echo "engine error during start-up"; RC_MAIN=6; exit 6; fi
    if docker exec "$CN" test -f /tmp/glm53-release-ready 2>/dev/null && curl -sf -m 5 "$BASE/v1/models" >/dev/null; then
      READY=1; break; fi
  else
    curl -sf -m 2 "$BASE/v1/models" >/dev/null && { READY=1; break; }
  fi
  sleep 10
done
if [ $READY -ne 1 ]; then echo "readiness timeout"; RC_MAIN=7; exit 7; fi
echo "ready after $(( $(date +%s) - T0 ))s" | tee "$AO/gate.txt"
STAGE="startup-gates"
[ "$MOCK" = 1 ] || docker logs "$CN" 2>&1 | grep -n "cudagraph_mode\|Graph capturing finished\|GPU KV cache size\|async_scheduling" | cut -c1-300 | tail -n 12 >> "$AO/gate.txt"
mkdir -p "$AO/p4log-startup"
c_cp log "$AO/p4log-startup"
# G-instr: hooks and stage-0 recording registered on both ranks before capture; entry-digest coverage per KV group
INIT=$(python3 - "$AO/p4log-startup" <<'PY'
import glob, json, os, sys
recs = []
for p in glob.glob(os.path.join(sys.argv[1], "p4_r*_p*.jsonl")):
    recs += [json.loads(l) for l in open(p) if l.strip()]
init = {r.get("rank"): r for r in recs if r.get("kind") == "init"}
ent = {r.get("rank"): r for r in recs if r.get("kind") == "entry_init"}
errs = [r for r in recs if r.get("kind") == "p4_error"]
ok = set(init) == {0, 1} and not errs and not init[0].get("issues") and init[0].get("stores_mib", 0) > 0 \
     and len(init[0].get("stage_modules", {})) == 6 and all(i.get("s0") and i.get("schema") == "local-p4/2" for i in init.values()) \
     and init[0].get("n_digest_layers", 0) > 0 and not (init.get(0) or {}).get("tile_env")
cov_ok = set(ent) == {0, 1}
groups = {}
for rk, e in ent.items():
    ng = int(e.get("n_groups", 0))
    cg = {int(t["group"]) for t in e.get("tensors", []) if t.get("covered")}
    groups[rk] = (len(cg), ng, int(e.get("covered", 0)), len(e.get("uncovered", [])))
    cov_ok = cov_ok and ng > 0 and len(cg) >= 1 and int(e.get("covered", 0)) >= 1   # uncovered groups: listed, reported
print(("OK" if ok and cov_ok else "FAIL") + " p4 init ranks %s s0 %s issues %s stage_modules %d digest_layers %s errors %d; entry coverage per rank (groups covered, groups, tensors covered, uncovered) %s" % (
    sorted(init), [i.get("s0") for i in init.values()], (init.get(0) or {}).get("issues"),
    len((init.get(0) or {}).get("stage_modules", {})), (init.get(0) or {}).get("n_digest_layers"), len(errs), groups))
PY
)
echo "G-instr: $INIT" | tee -a "$AO/gate.txt"
if [ "$MOCK" != 1 ]; then
  FWREG=$(docker exec "$CN" sh -c 'cat /tmp/p4/PRIVATE-fw/fw_r*_p*.jsonl 2>/dev/null' | grep -c '"kind": "registered"')
  FWERR=$(docker exec "$CN" sh -c 'cat /tmp/p4/PRIVATE-fw/fw_r*_p*.jsonl 2>/dev/null' | grep -c '"fw_error"')
else
  FWREG=$(cat "$MOCKCTR"/PRIVATE-fw/fw_r*_p*.jsonl 2>/dev/null | grep -c '"kind": "registered"')
  FWERR=$(cat "$MOCKCTR"/PRIVATE-fw/fw_r*_p*.jsonl 2>/dev/null | grep -c '"fw_error"')
fi
echo "forcewin: registered records $FWREG (expected 2), errors $FWERR" | tee -a "$AO/gate.txt"
case "$INIT" in OK*) ;; *) echo "INCOMPLETE: stage-0 recording not installed as designed (G-instr)"; RC_MAIN=8; exit 8;; esac
[ "$FWREG" = 2 ] && [ "$FWERR" = 0 ] || { echo "INCOMPLETE: forcewin not registered on both ranks"; RC_MAIN=8; exit 8; }
python3 "$ROOT/runner/gates_p4.py" --arm A --binds "$AO/p4log-startup" --p3b "$P3B_CAPLOG" --out "$AO/gates-startup.json" --stage startup 2>&1 | tee -a "$AO/gate.txt"
GRC=${PIPESTATUS[0]}
[ "$GRC" = 0 ] || { echo "INCOMPLETE: G3 failed before traffic"; RC_MAIN=8; exit 8; }
STAGE="client"
DEADLINE=$((T0 + CAP - RESERVE))
echo "== client (deadline in $((DEADLINE - $(date +%s)))s)"
timeout $((DEADLINE - $(date +%s) + 30)) env P4_WINDOWS="$WINDOWS" P4_SCHED="$SCHED" S0_BATCH="${S0_BATCH:-1}" \
  python3 "$HERE/client_stage0.py" "$AO" "$DEADLINE" "$BASE" > "$AO/client.log" 2>&1 &
CPID=$!
wait "$CPID"; RC_MAIN=$?
CPID=""
echo "client rc=$RC_MAIN" | tee -a "$AO/gate.txt"
STAGE="flush"
NOK=$(python3 -c "
import json
print(sum(1 for l in open('$AO/client.jsonl') if json.loads(l).get('kind')=='forced' and json.loads(l).get('ok')))" 2>/dev/null)
for i in $(seq 1 60); do
  FL=$(c_cat "log/status_r0.json" | python3 -c "import json,sys; print(json.load(sys.stdin).get('flushed',0))" 2>/dev/null)
  [ "${FL:-0}" -ge "${NOK:-0}" ] && break
  sleep 1
done
echo "flushed ${FL:-0} of ${NOK:-0} completed forced requests (wait ${i}s)" | tee -a "$AO/gate.txt"
STAGE="done"
exit 0
