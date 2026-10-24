#!/bin/bash
# captured-p3b, step A: capture run on stock 0.9.1 + observe-only graph-safe capture hooks (2 GPUs while serving).
# Adapted from livecheck-22ao/runner/run_22ao.sh (22ap version: re-sampled pre-flight).  Not run by the package
# build.  Launched only after the operator's authorization (AUTHORIZATION.txt):
#
#   ATTEMPT_REASON="..." runner/run_p3b.sh          the capture attempt (GPU)
#   ATTEMPT_REASON="..." DRY_RUN=1 runner/run_p3b.sh gates and plan only; starts nothing, writes nothing
#   MOCK=1 ...                                      CPU session simulation against tests/mock_server_p3b.py
#                                                   (no docker, no GPU; writes only under tests/out/)
#
# Budget (AUTHORIZATION.txt): 0.65 occupied GPU-hours TOTAL for the capture and the replay, including retries.
# The ledger is costs.jsonl (items p3b-capture and p3b-replay).  A capture attempt's wall cap is
#   min(CAP_MAX = 1080 s, (0.65 - spent - REPLAY_RESERVE 0.05) x 3600 / 2 GPUs)
# and an attempt is refused when that is below MIN_WALL (990 s): start-up alone took 743-874 s in 22ao, so an
# attempt with less wall time cannot reach traffic.  Retries are allowed (attempt-N directories) whenever the gate
# passes; in practice only an attempt that failed in its first ~1-2 minutes leaves room for another.
# Unattended-safe: ATTEMPT_REASON, budget gate, image pin, frozen hashes, pre-flight (no glm53* container, GPUs
# idle with re-sampled utilization), capture-installed gate before traffic, coverage watch (stops the client when
# the quotas are met), FLUSH handshake, wall-cap watchdog, stop on every exit path, one cost line per attempt
# counted to the moment the GPUs are released.  No `set -e`.
set -u
umask 022
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT_OVERRIDE:-<runs>/captured-p3b}
[ -d "$ROOT" ] || { echo "path check failed"; exit 2; }
COSTS=$ROOT/costs.jsonl
ITEM=p3b-capture
BUDGET_GH=0.65
REPLAY_RESERVE_GH=0.05
GPUS=2
CAP_MAX=1080
MIN_WALL=${MIN_WALL:-990}
GATE_CAP=${GATE_CAP:-930}          # readiness must come by T0 + GATE_CAP (22ao: 743 s and 874 s)
RESERVE=${RESERVE:-110}            # wall kept for FLUSH (<= 45 s), logs, stop.sh, docker stop
ST=<stacks>/glm53-flash-v0.9.0
RC=<recipes>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1
CN=glm53-flash-exl3-b12x-vllm
FIXTURE=<runs>/livecheck-22ao/PRIVATE/panel_fixture.LOCAL.json  # path only; read by the client
IMG=local/glm53-v091-p3b-capture:local
DEBUG="GLM53_P3B=capture GLM53_P3B_DIR=/tmp/p3b GLM53_P3B_POLL_SECS=1"
MOCK=${MOCK:-0}
DRY_RUN=${DRY_RUN:-0}
ATTEMPT_REASON=${ATTEMPT_REASON:-}
OUTROOT=$ROOT/runs
PRIVROOT=$ROOT/PRIVATE-capture
BASE=http://127.0.0.1:8001
if [ "$MOCK" = 1 ]; then
  OUTROOT=$ROOT/tests/out/mock-runs; PRIVROOT=$ROOT/tests/out/mock-PRIVATE; COSTS=${MOCK_COSTS:-$ROOT/tests/out/mock-costs.jsonl}
  FIXTURE=${MOCK_FIXTURE:-$ROOT/tests/fixture_fake.json}; BASE=http://127.0.0.1:${MOCK_PORT:-18011}
  GATE_CAP=${GATE_CAP_MOCK:-20}
fi

refuse() { echo "REFUSED: $*"; exit 2; }
abort()  { echo "ABORT: $*"; exit 3; }

[ -n "$ATTEMPT_REASON" ] || refuse "ATTEMPT_REASON is not set (say why this attempt runs)"
case "$DRY_RUN$MOCK" in 00|01|10|11) ;; *) refuse "DRY_RUN and MOCK must be 0 or 1";; esac
if [ -e "$COSTS" ]; then [ -w "$COSTS" ] || abort "cannot write $COSTS"; else [ -w "$(dirname "$COSTS")" ] || abort "cannot write $(dirname "$COSTS")"; fi

# ---- budget gate (all items in the ledger count: capture and replay share the 0.65 GPU-h ceiling)
BUD=$(python3 - "$COSTS" "$BUDGET_GH" "$REPLAY_RESERVE_GH" "$GPUS" "$CAP_MAX" "$ITEM" <<'PYEOF'
import json, os, sys
path, budget, reserve, gpus, capmax, item = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
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
    if d.get("seconds") is not None:
        used_h += float(d["seconds"]) * float(d.get("gpus", 1)) / 3600.0
    else:
        used_h += float(d.get("gpu_hours") or 0.0)
    if d.get("item") == item:
        attempts.add(d.get("attempt"))
left_h = budget - used_h
cap = int(min(capmax, max(0.0, (left_h - reserve)) * 3600.0 / gpus))
print(f"OK {used_h:.4f} {left_h:.4f} {cap} {len(attempts)}")
PYEOF
)
set -- $BUD
[ "${1:-}" = OK ] || abort "cannot read $COSTS: $BUD"
USED_GH=$2; LEFT_GH=$3; CAP=$4; PRIOR=$5
ATTEMPT=$((PRIOR + 1))
echo "budget: ceiling $BUDGET_GH GPU-h (capture + replay); used $USED_GH; left $LEFT_GH; replay reserve $REPLAY_RESERVE_GH"
echo "capture attempt $ATTEMPT: wall cap ${CAP}s x $GPUS GPUs = $(python3 -c "print(round($CAP*$GPUS/3600,4))") GPU-h"
[ "$CAP" -ge "$MIN_WALL" ] || refuse "wall cap ${CAP}s < MIN_WALL ${MIN_WALL}s (budget left $LEFT_GH GPU-h after the replay reserve)"
AO=$OUTROOT/capture-attempt-$ATTEMPT
PO=$PRIVROOT/capture-attempt-$ATTEMPT
[ -e "$AO" ] && refuse "$AO exists (attempt numbering out of step with the ledger; never overwritten)"
[ -e "$PO" ] && refuse "$PO exists (never overwritten)"

# ---- frozen hashes, image pin, fixture (path only)
if [ "$MOCK" != 1 ]; then
  SHAOUT=$( cd "$ROOT" && sha256sum -c --quiet SHA256SUMS 2>&1 ) || { echo "$SHAOUT"; abort "frozen hashes differ (SHA256SUMS)"; }
  ID=$(docker image inspect "$IMG" --format '{{.Id}}' 2>/dev/null)
  EXP=$(awk -v t="$IMG" '$1==t {print $2}' "$HERE/images.txt")
  [ -n "$EXP" ] && [ "$ID" = "$EXP" ] || abort "$IMG is '$ID', expected '$EXP' (runner/images.txt)"
  [ -d "$ST" ] && [ -f "$RC/env.sh" ] || abort "stack or recipe missing"
fi
[ -f "$FIXTURE" ] || abort "panel fixture missing"
if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN: ATTEMPT_REASON, budget gate, attempt numbering, frozen hashes, image pin and fixture passed."
  echo "plan: attempt $ATTEMPT -> $AO (logs) and $PO (captured tensors, 0700); wall cap ${CAP}s; readiness by ${GATE_CAP}s;"
  echo "      client deadline T0+$((CAP - RESERVE))s; watchdog at T0+$((CAP - 60))s; image $IMG; DEBUG_ENV '$DEBUG'"
  exit 0
fi

# the panel fixture is read (hashed) only here, on a real attempt, never by DRY_RUN: it must be the file 22ao froze
if [ "$MOCK" != 1 ]; then
  FIX_EXP=$(awk '$2 == "PRIVATE/panel_fixture.LOCAL.json" {print $1}' <runs>/livecheck-22ao/SHA256SUMS)
  FIX_SHA=$(sha256sum "$FIXTURE" 2>/dev/null | cut -d' ' -f1)
  [ -n "$FIX_EXP" ] && [ "$FIX_SHA" = "$FIX_EXP" ] || abort "panel fixture hash differs from livecheck-22ao/SHA256SUMS"
fi
mkdir -p "$AO" || abort "cannot create $AO"
( cd "$ROOT" && sha256sum runner/client_p3b.py runner/run_p3b.sh runner/images.txt runner/coverage_p3b.py \
    build/files/vllm/p3b_capture_hooks.py ) > "$AO/scripts.sha256" 2>/dev/null
echo "attempt $ATTEMPT reason: $ATTEMPT_REASON" > "$AO/attempt.txt"

preflight() {
  [ "$MOCK" = 1 ] && return 0
  echo "== preflight $(date -u +%FT%TZ)" | tee -a "$AO/preflight.txt"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a "$AO/preflight.txt"
  docker ps --format '{{.Names}} {{.Image}} {{.Status}}' | tee -a "$AO/preflight.txt"
  if docker ps --format '{{.Names}}' | grep -qE '^glm53|b0cache|mgtest|^p3-|^p3b-'; then echo "ABORT: inference/fixture container running"; return 3; fi
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

cexec() { [ "$MOCK" = 1 ] && return 1; docker exec "$CN" "$@"; }
status_json() {  # prints the poller's status.json (mock: the mock server's)
  if [ "$MOCK" = 1 ]; then cat "$AO/mock-status.json" 2>/dev/null; else docker exec "$CN" cat /tmp/p3b/log/status.json 2>/dev/null; fi
}

START=""; T0=0; RC_MAIN=""; WD=""; ACTIVE=0; IMG_ID=""; CPID=""
cleanup() {
  [ "$ACTIVE" = 1 ] || return 0
  ACTIVE=0
  [ -n "$WD" ] && kill "$WD" 2>/dev/null
  [ -n "$CPID" ] && kill "$CPID" 2>/dev/null
  echo "== cleanup $(date -u +%FT%TZ)"
  if [ "$MOCK" = 1 ]; then
    [ -n "${MOCKPID:-}" ] && kill "$MOCKPID" 2>/dev/null
  else
    if [ "$(docker inspect --format '{{.State.Running}}' "$CN" 2>/dev/null)" = true ]; then
      docker exec "$CN" sh -c 'touch /tmp/p3b/FLUSH' >/dev/null 2>&1
      local i
      for i in $(seq 1 45); do
        docker exec "$CN" test -f /tmp/p3b/log/FLUSHED.json 2>/dev/null && break
        sleep 1
      done
      echo "flush wait ${i}s" | tee -a "$AO/gate.txt"
      docker logs "$CN" > "$AO/server.log" 2>&1
      docker exec "$CN" sh -c 'cd /usr/local/lib/python3.12/dist-packages/vllm && sha256sum p3b_capture_hooks.py model_executor/layers/quantization/exl3.py v1/attention/backends/mla/b12x_mla_sparse.py; env | grep -E "^(GLM53|VLLM)" | sort' > "$AO/container-files-env.txt" 2>&1
    fi
    ( cd "$ST" && ./stop.sh ) >> "$AO/stop.log" 2>&1
    docker ps --format '{{.Names}}' | grep -q "^$CN\$" && docker stop -t 30 "$CN" >> "$AO/stop.log" 2>&1
  fi
  local TEND SECS GH
  TEND=$(date +%s); SECS=$((TEND - T0))
  GH=$(python3 -c "print(round($GPUS*$SECS/3600,4))")
  P3B_REASON="$ATTEMPT_REASON" python3 - "$COSTS" "$ITEM" "$ATTEMPT" "$START" "$(date -u -d @"$TEND" +%Y-%m-%dT%H:%M:%SZ)" \
      "$SECS" "$GPUS" "$GH" "$IMG_ID" "${RC_MAIN:-null}" "$CAP" "$MOCK" <<'PYEOF'
import json, os, sys
c, item, att, start, end, secs, gpus, gh, img, rc, cap, mock = sys.argv[1:]
line = dict(item=item, attempt=int(att), attempt_reason=os.environ["P3B_REASON"], start=start, end=end,
            seconds=int(secs), gpus=int(gpus), gpu_hours=float(gh), image=img, rc=None if rc == "null" else int(rc),
            wall_cap=int(cap), mock=mock == "1", purpose="captured-p3b capture run (stock 0.9.1 + capture hooks)")
open(c, "a").write(json.dumps(line) + "\n")
PYEOF
  echo "capture attempt $ATTEMPT occupied ${SECS}s x $GPUS GPU = ${GH} GPU-h (rc ${RC_MAIN:-null})"
  # copy-out after the GPUs are released (the stopped container is retained by the stack until the next start)
  ( umask 077; mkdir -p "$PO" ) && chmod 700 "$PRIVROOT" "$PO"
  if [ "$MOCK" != 1 ]; then
    mkdir -p "$AO/caplog"
    docker cp "$CN:/tmp/p3b/log/." "$AO/caplog/" >/dev/null 2>&1
    ( umask 077; docker cp "$CN:/tmp/p3b/PRIVATE-capture/." "$PO/" >/dev/null 2>&1 )
  else
    mkdir -p "$AO/caplog"
    [ -d "$AO/mock-caplog" ] && cp -r "$AO/mock-caplog/." "$AO/caplog/" 2>/dev/null
    [ -d "$AO/mock-private" ] && ( umask 077; cp -r "$AO/mock-private/." "$PO/" )
  fi
  chmod -R go-rwx "$PO" 2>/dev/null
  echo "captured files: $(find "$PO" -type f | wc -l) under $PO (mode 0700)"
  python3 "$HERE/coverage_p3b.py" "$AO" 2>&1 | tee "$AO/coverage-console.txt"
}
on_exit() { cleanup; }
trap on_exit EXIT
trap 'echo "signal"; exit 4' INT TERM

if ! preflight; then
  echo "preflight failed" | tee "$AO/refused.txt"
  # zero-GPU cost line keeps the attempt numbering in step with runs/capture-attempt-N
  P3B_REASON="$ATTEMPT_REASON" python3 -c "
import json, os, sys, time
t = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
open(sys.argv[1], 'a').write(json.dumps(dict(item=sys.argv[2], attempt=int(sys.argv[3]), attempt_reason=os.environ['P3B_REASON'],
    start=t, end=t, seconds=0, gpus=0, gpu_hours=0.0, image=None, rc=3, wall_cap=int(sys.argv[4]), mock=sys.argv[5] == '1',
    purpose='captured-p3b capture run: refused by pre-flight, no container started')) + chr(10))
" "$COSTS" "$ITEM" "$ATTEMPT" "$CAP" "$MOCK"
  exit 3
fi
START=$(date -u +%Y-%m-%dT%H:%M:%SZ); T0=$(date +%s); ACTIVE=1
[ "$MOCK" = 1 ] || IMG_ID=$(docker image inspect "$IMG" --format '{{.Id}}' 2>/dev/null)
# watchdog: stops the server even if this script hangs; kills its own sleep on TERM
( trap 'kill "$SP" 2>/dev/null; exit 0' TERM; sleep $(( CAP > 300 ? CAP - 60 : CAP )) & SP=$!; wait "$SP" || exit 0; \
  echo "WALL CAP reached"; \
  if [ "$MOCK" = 1 ]; then pkill -f "tests/mock_server_p3b[.]py" 2>/dev/null; else docker stop -t 30 "$CN" >/dev/null 2>&1; fi ) &
WD=$!
echo "== start $START image=$IMG ($IMG_ID) cap ${CAP}s attempt $ATTEMPT"
if [ "$MOCK" = 1 ]; then
  python3 "$ROOT/tests/mock_server_p3b.py" "$AO" "${MOCK_PORT:-18011}" > "$AO/mock-server.log" 2>&1 &
  MOCKPID=$!; sleep 1
else
  ( unset EXTRA_SERVE_ARGS; cd "$ST" && source "$RC/env.sh" && export SPECULATIVE_METHOD=dflash2 DFLASH_TOKENS=3 \
      IMAGE="$IMG" DEBUG_ENV="$DEBUG" && ./start.sh ) > "$AO/start.log" 2>&1
  SRC=$?
  if [ $SRC -ne 0 ]; then echo "start failed rc=$SRC"; RC_MAIN=5; exit 5; fi
fi
READY=0
while [ $(( $(date +%s) - T0 )) -lt "$GATE_CAP" ]; do
  if [ "$MOCK" != 1 ]; then
    RUN=$(docker inspect --format '{{.State.Running}} {{.RestartCount}}' "$CN" 2>/dev/null)
    if [ "$RUN" != "true 0" ]; then echo "container state '$RUN'"; RC_MAIN=6; exit 6; fi
    if docker logs "$CN" 2>&1 | grep -qE 'Engine core initialization failed|CUDA out of memory|Traceback'; then
      echo "engine error during start-up"; RC_MAIN=6; exit 6; fi
    NE=$(docker exec "$CN" sh -c 'grep -h "\"capture_error\"" /tmp/p3b/log/cap_*.jsonl 2>/dev/null | wc -l' 2>/dev/null)
    if [ "${NE:-0}" -gt 0 ]; then echo "INCOMPLETE: capture error during start-up"; RC_MAIN=8; exit 8; fi
    if docker exec "$CN" test -f /tmp/glm53-release-ready 2>/dev/null && curl -sf -m 5 "$BASE/v1/models" >/dev/null; then
      READY=1; break; fi
  else
    curl -sf -m 2 "$BASE/v1/models" >/dev/null && { READY=1; break; }
  fi
  sleep 10
done
if [ $READY -ne 1 ]; then echo "readiness timeout"; RC_MAIN=7; exit 7; fi
echo "ready after $(( $(date +%s) - T0 ))s" | tee "$AO/gate.txt"
if [ "$MOCK" != 1 ]; then
  docker logs "$CN" 2>&1 | grep -n "cudagraph_mode\|Graph capturing finished\|GPU KV cache size\|async_scheduling\|capture_error" | cut -c1-300 | tail -n 20 >> "$AO/gate.txt"
  # capture installed: rank 0 registered with rings, rank 1 inert, and every hook site seen at graph capture
  GATE=$(docker exec "$CN" sh -c 'cat /tmp/p3b/log/cap_*.jsonl 2>/dev/null' | python3 -c "
import json, sys
recs = [json.loads(l) for l in sys.stdin if l.strip()]
init = [r for r in recs if r.get('kind') == 'init']
r0 = [r for r in init if r.get('rank') == 0 and 'ring_bytes' in r]
r1 = [r for r in init if r.get('rank') == 1 and r.get('inert')]
sites = {tuple(r['key'][:2]) for r in recs if r.get('kind') == 'site'}
moe_ok = all(('moe', L) in sites for L in (3, 23, 44))
att_ok = all((k, L) in sites for L in (3, 23) for k in ('att_pre', 'att_in', 'att_out'))
issues = r0[0].get('issues') if r0 else ['no rank-0 init']
cap = [r for r in recs if r.get('kind') == 'site' and r['key'][0] == 'moe' and not r.get('captured')]
ok = bool(r0) and bool(r1) and moe_ok and att_ok and not issues and r0[0].get('async_scheduling') is True
print('OK' if ok else 'FAIL', 'rank0', bool(r0), 'rank1_inert', bool(r1), 'moe_sites', moe_ok, 'att_sites', att_ok,
      'issues', issues, 'ring_MiB', r0[0].get('ring_mib_total') if r0 else None,
      'async', r0[0].get('async_scheduling') if r0 else None, 'moe_sites_not_captured', [r['key'] for r in cap][:6])
")
  echo "capture gate: $GATE" | tee -a "$AO/gate.txt"
  case "$GATE" in OK*) ;; *) echo "INCOMPLETE: capture not installed as designed"; RC_MAIN=8; exit 8;; esac
  docker exec "$CN" touch /tmp/p3b/PHASE_TRAFFIC || { RC_MAIN=8; exit 8; }
else
  touch "$AO/PHASE_TRAFFIC"
fi
DEADLINE=$((T0 + CAP - RESERVE))
echo "== client (deadline in $((DEADLINE - $(date +%s)))s)"
timeout $((DEADLINE - $(date +%s) + 15)) env PANEL_FIXTURE="$FIXTURE" python3 "$HERE/client_p3b.py" "$AO" "$DEADLINE" "$BASE" \
  > "$AO/client.log" 2>&1 &
CPID=$!
MET=0
while kill -0 "$CPID" 2>/dev/null; do
  sleep 5
  S_=$(status_json)
  if [ -n "$S_" ] && echo "$S_" | python3 -c "import json,sys; sys.exit(0 if json.load(sys.stdin).get('quotas_met') else 1)" 2>/dev/null; then
    [ "$MET" = 0 ] && { echo "quotas met at +$(( $(date +%s) - T0 ))s: stopping the client" | tee -a "$AO/gate.txt"; touch "$AO/STOP"; MET=1; }
  fi
done
wait "$CPID"; RC_MAIN=$?
CPID=""
echo "client rc=$RC_MAIN (quotas met: $MET)" | tee -a "$AO/gate.txt"
exit 0
