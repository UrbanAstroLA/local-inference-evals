#!/bin/bash
# P4 serving session (2 GPUs): stock 0.9.1 + forcewin (stock port) + local-p4 recorder/stages, one server start.
# Adapted from captured-p3b/runner/run_p3b.sh (itself from livecheck-22ao/runner/run_22ao.sh, 22ap re-sampled
# pre-flight).  Not run by the package build.  Launched only after the operator's authorization:
#
#   SESSION=A|B|A2 ATTEMPT_REASON="..." P4_AMENDMENTS_ACK=<12 hex> runner/run_p4.sh      a session (GPU)
#   ... DRY_RUN=1 runner/run_p4.sh      gates and plan only; starts nothing, writes nothing
#   MOCK=1 ...                          CPU session simulation against tests/mock_server_p4.py (no docker, no GPU;
#                                       writes only under tests/out/)
#
# Arms (one image, PROPOSAL.md section 2): A and A2 = stock tile selection (GLM53_P4_TILE unset); B = GLM53_P4_TILE=
# 64,256,64,256.  Order A -> B -> A2 is enforced by runner/budget_p4.py, which also needs a passed single-GPU
# pre-check (precheck/run_precheck.sh) and a passed G4 (g4/run_g4.sh) before A.
# Budget (AUTHORIZATION.txt): 4.0 occupied GPU-hours in TOTAL, retries and single-GPU checks included, ledger
# costs.jsonl.  Wall cap per attempt = min(2350 s, (left - 1.25 x later sessions) x 3600 / 2) (budget_p4.py).
# Unattended-safe: ATTEMPT_REASON, amendments acknowledgement, budget/order gate, image pin, frozen hashes, fixture
# hashes, pre-flight (no glm53* container, GPUs idle with re-sampled utilization, no foreign GPU client),
# start-up gates (hooks registered on both ranks before capture, recorder coverage, G1/G2 or G3 or the A2 control
# evaluated from the bound launches BEFORE any traffic), client deadline, flush wait, wall-cap watchdog, stop on
# every exit path, one cost line per attempt counted to the moment the GPUs are released.  No `set -e`.
set -u
umask 022
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT_OVERRIDE:-<runs>/tile-p4}
[ -d "$ROOT" ] || { echo "path check failed"; exit 2; }
INV=<runs>
COSTS=$ROOT/costs.jsonl
GPUS=2
GATE_CAP=${GATE_CAP:-960}          # readiness must come by T0 + GATE_CAP (measured 722-874 s)
RESERVE=${RESERVE:-120}            # wall kept after the client for the flush wait, logs, stop.sh, docker stop
ST=<stacks>/glm53-flash-v0.9.0
RC=<recipes>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1
CN=glm53-flash-exl3-b12x-vllm
IMG=local/glm53-v091-p4-tile-r3:local   # precheck fixes (r2: capture check first; r3: slot release on refusal)
P3B_CAPLOG=$INV/captured-p3b/runs/capture-attempt-2/caplog/cap_p370.jsonl
WINDOWS=$ROOT/PRIVATE-fixtures/windows.json
SCHED=$ROOT/PRIVATE-fixtures/sched.json
MOCK=${MOCK:-0}
DRY_RUN=${DRY_RUN:-0}
SESSION=${SESSION:-}
ATTEMPT_REASON=${ATTEMPT_REASON:-}
OUTROOT=$ROOT/runs
PRIVROOT=$ROOT/PRIVATE-runs
BASE=http://127.0.0.1:8001
DEBUG_COMMON="GLM53_P4=on GLM53_P4_DIR=/tmp/p4 FORCEWIN=on FORCEWIN_DIR=/tmp/p4/PRIVATE-fw FORCEWIN_WMAX=520 FORCEWIN_NDESIG=1 FORCEWIN_NSEL=1"
BROOT=$ROOT
if [ "$MOCK" = 1 ]; then   # everything under one mock root (tests/test_runner.sh); never the real ledger or runs
  BROOT=${MOCK_ROOT:-$ROOT/tests/out/mock-root}
  OUTROOT=$BROOT/runs; PRIVROOT=$BROOT/PRIVATE-runs; COSTS=$BROOT/costs.jsonl
  WINDOWS=$BROOT/PRIVATE-fixtures/windows.json; SCHED=$BROOT/PRIVATE-fixtures/sched.json
  BASE=http://127.0.0.1:${MOCK_PORT:-18041}; GATE_CAP=${GATE_CAP_MOCK:-20}; RESERVE=${RESERVE_MOCK:-15}
  case "$BROOT" in "$ROOT"/tests/out/*) ;; *) echo "REFUSED: MOCK_ROOT outside tests/out"; exit 2;; esac
fi

refuse() { echo "REFUSED: $*"; exit 2; }
abort()  { echo "ABORT: $*"; exit 3; }

[ -n "$ATTEMPT_REASON" ] || refuse "ATTEMPT_REASON is not set (say why this attempt runs)"
case "$SESSION" in A|B|A2) ;; *) refuse "SESSION must be A, B or A2 (run in that order)";; esac
case "$DRY_RUN$MOCK" in 00|01|10|11) ;; *) refuse "DRY_RUN and MOCK must be 0 or 1";; esac
TILE=""; [ "$SESSION" = B ] && TILE="64,256,64,256"
AMD_SHA=$(sha256sum "$ROOT/PROPOSAL-AMENDMENTS.md" 2>/dev/null | cut -c1-12)
[ -n "$AMD_SHA" ] || refuse "PROPOSAL-AMENDMENTS.md missing"
[ "${P4_AMENDMENTS_ACK:-}" = "$AMD_SHA" ] || refuse "P4_AMENDMENTS_ACK must be $AMD_SHA (the first 12 hex of sha256(PROPOSAL-AMENDMENTS.md)): the operator accepts the amendments it lists"
if [ -e "$COSTS" ]; then [ -w "$COSTS" ] || abort "cannot write $COSTS"; else [ -w "$(dirname "$COSTS")" ] || abort "cannot write $(dirname "$COSTS")"; fi

# ---- budget / order / attempt gate (budget_p4.py; MOCK reads the mock ledger and mock run dirs)
BUD=$(python3 "$HERE/budget_p4.py" "$BROOT" "$SESSION")
set -- $BUD
[ "${1:-}" = OK ] || refuse "budget/order gate: $BUD"
USED_GH=$2; LEFT_GH=$3; CAP=$4; ATTEMPT=$5
echo "budget: ceiling 7.0 GPU-h; used $USED_GH; left $LEFT_GH; session $SESSION attempt $ATTEMPT: wall cap ${CAP}s x $GPUS GPUs"
AO=$OUTROOT/session-$SESSION-attempt-$ATTEMPT
PO=$PRIVROOT/session-$SESSION-attempt-$ATTEMPT
[ -e "$AO" ] && refuse "$AO exists (attempt numbering out of step with the ledger; never overwritten)"
[ -e "$PO" ] && refuse "$PO exists (never overwritten)"

# ---- frozen hashes, image pin, fixtures
if [ "$MOCK" != 1 ]; then
  SHAOUT=$( cd "$ROOT" && sha256sum -c --quiet SHA256SUMS 2>&1 ) || { echo "$SHAOUT"; abort "frozen hashes differ (SHA256SUMS)"; }
  ID=$(docker image inspect "$IMG" --format '{{.Id}}' 2>/dev/null)
  EXP=$(awk -v t="$IMG" '$1==t {print $2}' "$HERE/images.txt")
  [ -n "$EXP" ] && [ "$ID" = "$EXP" ] || abort "$IMG is '$ID', expected '$EXP' (runner/images.txt)"
  [ -d "$ST" ] && [ -f "$RC/env.sh" ] || abort "stack or recipe missing"
  [ -f "$P3B_CAPLOG" ] || abort "captured-p3b capture log (G3 reference) missing"
  FX_EXP=$(python3 -c "import json;print(json.load(open('$ROOT/windows/out/window-manifest.json'))['private_fixture_sha256'])")
  [ "$(sha256sum "$WINDOWS" 2>/dev/null | cut -d' ' -f1)" = "$FX_EXP" ] || abort "windows.json hash differs from windows/out/window-manifest.json"
fi
[ -f "$WINDOWS" ] && [ -f "$SCHED" ] || abort "fixtures missing"
REF_A=""
if [ "$SESSION" != A ]; then
  REF_A=$(python3 - "$OUTROOT" <<'PY'
import glob, json, os, sys
best = None
for p in sorted(glob.glob(os.path.join(sys.argv[1], "session-A-attempt-*", "outcome.json"))):
    o = json.load(open(p))
    if o.get("gates_ok") and int(o.get("batches_done", 0)) >= 1:
        best = os.path.join(os.path.dirname(p), "p4log")
print(best or "")
PY
)
  [ -n "$REF_A" ] && [ -d "$REF_A" ] || abort "no completed session A bind log to compare with"
fi
if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN: ATTEMPT_REASON, amendments ack, budget/order gate, attempt numbering, frozen hashes, image pin and fixtures passed."
  echo "plan: session $SESSION attempt $ATTEMPT -> $AO (logs) and $PO (private, 0700); wall cap ${CAP}s; readiness by ${GATE_CAP}s;"
  echo "      client deadline T0+$((CAP - RESERVE))s; watchdog at T0+$((CAP - 60))s; image $IMG;"
  echo "      DEBUG_ENV '$DEBUG_COMMON${TILE:+ GLM53_P4_TILE=$TILE}'; gates: $([ "$SESSION" = A ] && echo G3 vs captured-p3b || ([ "$SESSION" = B ] && echo G1 + G2 vs $REF_A || echo A2 control vs $REF_A))"
  P4_WINDOWS="$WINDOWS" P4_SCHED="$SCHED" P4_PLAN_ONLY=1 python3 "$HERE/client_p4.py" x 0 x "$SESSION"
  exit 0
fi

mkdir -p "$AO" || abort "cannot create $AO"
( cd "$ROOT" && sha256sum runner/*.py runner/*.sh runner/images.txt build/files/vllm/p4_hooks.py \
    build/files/vllm/forcewin_hooks.py analysis/*.py ) > "$AO/scripts.sha256" 2>/dev/null
echo "session $SESSION attempt $ATTEMPT reason: $ATTEMPT_REASON; tile env '${TILE}'; amendments ack $AMD_SHA" > "$AO/attempt.txt"

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
c_cat() {  # print a file under the container's /tmp/p4 (mock: the mock server's directory)
  if [ "$MOCK" = 1 ]; then cat "$MOCKCTR/$1" 2>/dev/null; else docker exec "$CN" sh -c "cat /tmp/p4/$1" 2>/dev/null; fi
}
c_cp() {   # copy the container's /tmp/p4/$1 into $2
  if [ "$MOCK" = 1 ]; then [ -e "$MOCKCTR/$1" ] && cp -r "$MOCKCTR/$1/." "$2/" 2>/dev/null; else docker cp "$CN:/tmp/p4/$1/." "$2/" >/dev/null 2>&1; fi
}
cost_line() {  # secs rc note
  P4_REASON="$ATTEMPT_REASON" python3 - "$COSTS" "p4-$SESSION" "$ATTEMPT" "${START:-}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$1" "$GPUS" "${IMG_ID:-}" "$2" "$CAP" "$MOCK" "$SESSION" "$TILE" "$3" <<'PYEOF'
import json, os, sys
c, item, att, start, end, secs, gpus, img, rc, cap, mock, sess, tile, note = sys.argv[1:]
secs = int(secs)
line = dict(item=item, attempt=int(att), session=sess, attempt_reason=os.environ["P4_REASON"], start=start or end, end=end,
            seconds=secs, gpus=int(gpus) if secs > 0 else 0, gpu_hours=round(int(gpus) * secs / 3600, 4) if secs > 0 else 0.0,
            image=img or None, rc=None if rc == "null" else int(rc), wall_cap=int(cap), mock=mock == "1",
            tile_env=tile or None, note=note or None, purpose="P4 serving session (stock 0.9.1 + forcewin port + local-p4)")
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
    [ -n "${MOCKPID:-}" ] && kill "$MOCKPID" 2>/dev/null
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
  echo "session $SESSION attempt $ATTEMPT occupied ${SECS}s x $GPUS GPU = $(python3 -c "print(round($GPUS*$SECS/3600,4))") GPU-h (rc ${RC_MAIN:-null})"
  # copy-out after the GPUs are released (the stopped container is retained by the stack until the next start)
  mkdir -p "$AO/p4log"
  c_cp log "$AO/p4log"
  ( umask 077; mkdir -p "$PO/p4" "$PO/fw" ) && chmod 700 "$PRIVROOT" "$PO" "$PO/p4" "$PO/fw"
  ( umask 077; c_cp PRIVATE "$PO/p4"; c_cp PRIVATE-fw "$PO/fw" )
  chmod -R go-rwx "$PO" 2>/dev/null
  echo "private files: $(find "$PO" -type f | wc -l) under $PO (mode 0700)"
  local GA=(--arm "$SESSION" --binds "$AO/p4log" --out "$AO/gates-final.json" --stage final)
  [ "$SESSION" = A ] && GA+=(--p3b "$P3B_CAPLOG")
  [ -n "$REF_A" ] && GA+=(--ref-a "$REF_A")
  python3 "$HERE/gates_p4.py" "${GA[@]}" 2>&1 | tee -a "$AO/gate.txt"
  python3 "$HERE/outcome_p4.py" "$AO" "$SESSION" "$ATTEMPT" 2>&1 | tee "$AO/outcome-console.txt"
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
DEBUG="$DEBUG_COMMON"; [ -n "$TILE" ] && DEBUG="$DEBUG GLM53_P4_TILE=$TILE"
# watchdog: stops the server even if this script hangs
( trap 'kill "$SP" 2>/dev/null; exit 0' TERM; sleep $(( CAP > 300 ? CAP - 60 : CAP )) & SP=$!; wait "$SP" || exit 0; \
  echo "WALL CAP reached"; \
  if [ "$MOCK" = 1 ]; then pkill -f "tests/mock_server_p4[.]py" 2>/dev/null; else docker stop -t 30 "$CN" >/dev/null 2>&1; fi ) &
WD=$!
echo "== start $START image=$IMG ($IMG_ID) cap ${CAP}s session $SESSION attempt $ATTEMPT tile '${TILE}'"
if [ "$MOCK" = 1 ]; then
  mkdir -p "$MOCKCTR"
  MOCK_TILE="$TILE" python3 "$ROOT/tests/mock_server_p4.py" "$MOCKCTR" "${MOCK_PORT:-18041}" "$SESSION" > "$AO/mock-server.log" 2>&1 &
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
# hooks registered on both ranks before capture; no hook error; forcewin registered on both ranks
INIT=$(python3 - "$AO/p4log-startup" <<'PY'
import glob, json, os, sys
recs = []
for p in glob.glob(os.path.join(sys.argv[1], "p4_r*_p*.jsonl")):
    recs += [json.loads(l) for l in open(p) if l.strip()]
init = {r.get("rank"): r for r in recs if r.get("kind") == "init"}
errs = [r for r in recs if r.get("kind") == "p4_error"]
ok = set(init) == {0, 1} and not errs and not init[0].get("issues") and init[0].get("stores_mib", 0) > 0 \
     and len(init[0].get("stage_modules", {})) == 6
print(("OK" if ok else "FAIL") + " p4 init ranks %s issues %s stores_MiB %s stage_modules %d errors %d tile_env %r" % (
    sorted(init), (init.get(0) or {}).get("issues"), (init.get(0) or {}).get("stores_mib"),
    len((init.get(0) or {}).get("stage_modules", {})), len(errs), (init.get(0) or {}).get("tile_env")))
PY
)
echo "hooks: $INIT" | tee -a "$AO/gate.txt"
if [ "$MOCK" != 1 ]; then
  FWREG=$(docker exec "$CN" sh -c 'cat /tmp/p4/PRIVATE-fw/fw_r*_p*.jsonl 2>/dev/null' | grep -c '"kind": "registered"')
  FWERR=$(docker exec "$CN" sh -c 'cat /tmp/p4/PRIVATE-fw/fw_r*_p*.jsonl 2>/dev/null' | grep -c '"fw_error"')
else
  FWREG=$(cat "$MOCKCTR"/PRIVATE-fw/fw_r*_p*.jsonl 2>/dev/null | grep -c '"kind": "registered"')
  FWERR=$(cat "$MOCKCTR"/PRIVATE-fw/fw_r*_p*.jsonl 2>/dev/null | grep -c '"fw_error"')
fi
echo "forcewin: registered records $FWREG (expected 2), errors $FWERR" | tee -a "$AO/gate.txt"
case "$INIT" in OK*) ;; *) echo "INCOMPLETE: hooks not installed as designed"; RC_MAIN=8; exit 8;; esac
[ "$FWREG" = 2 ] && [ "$FWERR" = 0 ] || { echo "INCOMPLETE: forcewin not registered on both ranks"; RC_MAIN=8; exit 8; }
GA=(--arm "$SESSION" --binds "$AO/p4log-startup" --out "$AO/gates-startup.json" --stage startup)
[ "$SESSION" = A ] && GA+=(--p3b "$P3B_CAPLOG")
[ -n "$REF_A" ] && GA+=(--ref-a "$REF_A")
python3 "$HERE/gates_p4.py" "${GA[@]}" 2>&1 | tee -a "$AO/gate.txt"
GRC=${PIPESTATUS[0]}
[ "$GRC" = 0 ] || { echo "INCOMPLETE: launch-verification gates failed before traffic (no measurement read)"; RC_MAIN=8; exit 8; }
STAGE="client"
DEADLINE=$((T0 + CAP - RESERVE))
echo "== client (deadline in $((DEADLINE - $(date +%s)))s)"
timeout $((DEADLINE - $(date +%s) + 30)) env P4_WINDOWS="$WINDOWS" P4_SCHED="$SCHED" \
  python3 "$HERE/client_p4.py" "$AO" "$DEADLINE" "$BASE" "$SESSION" > "$AO/client.log" 2>&1 &
CPID=$!
wait "$CPID"; RC_MAIN=$?
CPID=""
echo "client rc=$RC_MAIN" | tee -a "$AO/gate.txt"
STAGE="flush"
NOK=$(python3 -c "
import json
n=0
for l in open('$AO/client.jsonl'):
    d=json.loads(l)
    if d.get('kind')=='forced' and d.get('status')==200: n+=1
print(n)" 2>/dev/null)
for i in $(seq 1 60); do
  FL=$(c_cat "log/status_r0.json" | python3 -c "import json,sys; print(json.load(sys.stdin).get('flushed',0))" 2>/dev/null)
  [ "${FL:-0}" -ge "${NOK:-0}" ] && break
  sleep 1
done
echo "flushed ${FL:-0} of ${NOK:-0} completed forced requests (wait ${i}s)" | tee -a "$AO/gate.txt"
exit 0
