#!/bin/bash
# capture-p0 session (PROTOCOL.md): ONE serving session on 2 GPUs, image local/glm53-v091-p4-cap0:local with
# GLM53_P4=on GLM53_P4_S0=on GLM53_CAP0=on GLM53_CAP0_LOC=on|off, N independently warmed replicates of R01 (13,057-token
# warm, then the R01-early capture request whose first step is point 0), one at a time, each drained by a sentinel.
#
#   APPROVED_GPU_H=<approval> ATTEMPT_REASON="..." CAP0_PROTOCOL_ACK=<12 hex> runner/run_cap0.sh
#   ... DRY_RUN=1 ...     gates and plan only; starts nothing, writes nothing
#   MOCK=1 ...            CPU simulation against tests/mock_server_cap0.py (no container, no GPU; MOCK_ROOT under
#                         tests/out; synthetic fixture; private outputs under MOCK_ROOT/private-out-synthetic)
#   CAP0_LOC=off          the localisation option off (hooks not installed; every replicate "N")
#   CAP0_REPS=...         replicate plan (default r1:L,r2:L,r3:N,r4:N,r5:L,r6:L)
#   CAP0_PRIOR_ACK=<12 hex>  needed after a completed attempt whose analysis read INCOMPLETE (budget_cap0.py)
#
# Unattended-safe: ATTEMPT_REASON, protocol acknowledgement, budget/order gate (needs a passed pre-check; ledger
# costs-cap0.jsonl only), image pin, frozen hashes (capture-p0/SHA256SUMS), fixture hash, pre-flight, start-up gates
# (G-instr: local-p4 schema 2 + stage-0 recording + cap0 targets/layout/hooks on both ranks; G3 vs captured-p3b)
# BEFORE any traffic, client deadline, flush wait, wall-cap watchdog, stop on every exit path, one cost line per
# attempt counted to the GPU release, private copy-out (0700), outcome, then the analysis in a CPU container.  No `set -e`.
set -u
umask 022
HERE=$(cd "$(dirname "$0")" && pwd)
PKG=$(cd "$HERE/.." && pwd)
TILE=$(cd "$PKG/.." && pwd)
INV=<runs>
GPUS=2
GATE_CAP=${GATE_CAP:-960}
RESERVE=${RESERVE:-90}            # client deadline T0 + cap - 90 s; the watchdog stops the container at T0 + cap - 60 s
ST=<stacks>/glm53-flash-v0.9.0
RC=<recipes>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1
CN=glm53-flash-exl3-b12x-vllm
IMG=local/glm53-v091-p4-cap0-a2:local
P3B_CAPLOG=$INV/captured-p3b/runs/capture-attempt-2/caplog/cap_p370.jsonl
MOCK=${MOCK:-0}
DRY_RUN=${DRY_RUN:-0}
CAP0_LOC=${CAP0_LOC:-on}
ATTEMPT_REASON=${ATTEMPT_REASON:-}
BROOT=$PKG
PRIVROOT=$TILE/PRIVATE-capture-p0
WINDOWS=$TILE/PRIVATE-fixtures/windows.json
if [ "$MOCK" = 1 ]; then
  BROOT=${MOCK_ROOT:-$PKG/tests/out/cap0-mock-root}
  case "$BROOT" in "$PKG"/tests/out/*) ;; *) echo "REFUSED: MOCK_ROOT outside tests/out"; exit 2;; esac
  PRIVROOT=$BROOT/private-out-synthetic
  WINDOWS=$BROOT/fixtures-synthetic/windows.json
fi
COSTS=$BROOT/costs-cap0.jsonl
OUTROOT=$BROOT/runs
BASE=http://127.0.0.1:8001
[ "$MOCK" = 1 ] && { BASE=http://127.0.0.1:${MOCK_PORT:-18061}; GATE_CAP=${GATE_CAP_MOCK:-20}; RESERVE=${RESERVE_MOCK:-15}; }
case "$CAP0_LOC" in on|off) ;; *) echo "REFUSED: CAP0_LOC must be on or off"; exit 2;; esac
DEBUG="GLM53_P4=on GLM53_P4_S0=on GLM53_P4_DIR=/tmp/p4 GLM53_CAP0=on GLM53_CAP0_LOC=$CAP0_LOC FORCEWIN=on FORCEWIN_DIR=/tmp/p4/PRIVATE-fw FORCEWIN_WMAX=520 FORCEWIN_NDESIG=1 FORCEWIN_NSEL=1"

refuse() { echo "REFUSED: $*"; exit 2; }
abort()  { echo "ABORT: $*"; exit 3; }

[ -n "$ATTEMPT_REASON" ] || refuse "ATTEMPT_REASON is not set (say why this attempt runs)"
case "$DRY_RUN$MOCK" in 00|01|10|11) ;; *) refuse "DRY_RUN and MOCK must be 0 or 1";; esac
[ -n "${APPROVED_GPU_H:-}" ] || refuse "APPROVED_GPU_H is not set: capture-p0 needs an explicit operator approval in GPU-hours (no default)"
PROTO_SHA=$(sha256sum "$PKG/PROTOCOL.md" 2>/dev/null | cut -c1-12)
[ -n "$PROTO_SHA" ] || refuse "PROTOCOL.md missing"
[ "${CAP0_PROTOCOL_ACK:-}" = "$PROTO_SHA" ] || refuse "CAP0_PROTOCOL_ACK must be $PROTO_SHA (first 12 hex of sha256(capture-p0/PROTOCOL.md)): the operator accepts the protocol"
mkdir -p "$BROOT" 2>/dev/null
if [ -e "$COSTS" ]; then [ -w "$COSTS" ] || abort "cannot write $COSTS"; else [ -w "$(dirname "$COSTS")" ] || abort "cannot write $(dirname "$COSTS")"; fi

EXP=$(awk -v t="$IMG" '$1==t {print $2}' "$HERE/images.txt")
BUD=$(APPROVED_GPU_H="$APPROVED_GPU_H" python3 "$HERE/budget_cap0.py" "$BROOT" session "$EXP")
set -- $BUD
[ "${1:-}" = OK ] || refuse "budget/order gate: $BUD"
USED_GH=$2; LEFT_GH=$3; CAP=$4; ATTEMPT=$5
echo "budget: approved $APPROVED_GPU_H GPU-h (costs-cap0.jsonl); used $USED_GH; left $LEFT_GH; session attempt $ATTEMPT: wall cap ${CAP}s x $GPUS GPUs"
AO=$OUTROOT/session-attempt-$ATTEMPT
PO=$PRIVROOT/session-attempt-$ATTEMPT
[ -e "$AO" ] && refuse "$AO exists (attempt numbering out of step with the ledger; never overwritten)"
[ -e "$PO" ] && refuse "$PO exists (never overwritten)"

if [ "$MOCK" != 1 ]; then
  SHAOUT=$( cd "$PKG" && sha256sum -c --quiet SHA256SUMS 2>&1 ) || { echo "$SHAOUT"; abort "frozen hashes differ (capture-p0/SHA256SUMS)"; }
  ID=$(docker image inspect "$IMG" --format '{{.Id}}' 2>/dev/null)
  [ -n "$EXP" ] && [ "$ID" = "$EXP" ] || abort "$IMG is '$ID', expected '$EXP' (runner/images.txt)"
  [ -d "$ST" ] && [ -f "$RC/env.sh" ] || abort "stack or recipe missing"
  [ -f "$P3B_CAPLOG" ] || abort "captured-p3b capture log (G3 reference) missing"
  FX_EXP=$(python3 -c "import json;print(json.load(open('$TILE/windows/out/window-manifest.json'))['private_fixture_sha256'])")
  [ "$(sha256sum "$WINDOWS" 2>/dev/null | cut -d' ' -f1)" = "$FX_EXP" ] || abort "windows.json hash differs from windows/out/window-manifest.json"
fi
[ -f "$WINDOWS" ] || abort "fixture missing"
if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN: ATTEMPT_REASON, APPROVED_GPU_H, protocol ack, budget/order gate, attempt numbering, frozen hashes, image pin and fixture passed."
  echo "plan: session attempt $ATTEMPT -> $AO (logs) and $PO (private, 0700); wall cap ${CAP}s; readiness by ${GATE_CAP}s;"
  echo "      client deadline T0+$((CAP - RESERVE))s; watchdog at T0+$((CAP - 60))s; image $IMG; DEBUG_ENV '$DEBUG'"
  CAP0_WINDOWS="$WINDOWS" CAP0_LOC="$CAP0_LOC" CAP0_PLAN_ONLY=1 python3 "$HERE/client_cap0.py" x 0 x
  exit 0
fi

mkdir -p "$AO" || abort "cannot create $AO"
( cd "$PKG" && sha256sum runner/*.py runner/*.sh runner/images.txt build/files/vllm/*.py analysis/*.py PROTOCOL.md ) > "$AO/scripts.sha256" 2>/dev/null
echo "cap0 session attempt $ATTEMPT reason: $ATTEMPT_REASON; approved $APPROVED_GPU_H GPU-h; protocol ack $PROTO_SHA; loc $CAP0_LOC" > "$AO/attempt.txt"

preflight() {
  [ "$MOCK" = 1 ] && return 0
  echo "== preflight $(date -u +%FT%TZ)" | tee -a "$AO/preflight.txt"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a "$AO/preflight.txt"
  docker ps --format '{{.Names}} {{.Image}} {{.Status}}' | tee -a "$AO/preflight.txt"
  if docker ps --format '{{.Names}}' | grep -qE '^glm53|b0cache|mgtest|^p3-|^p3b-|^p4-|^cap0-'; then echo "ABORT: inference/fixture container running"; return 3; fi
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
c_cat() { if [ "$MOCK" = 1 ]; then cat "$MOCKCTR"/$1 2>/dev/null; else docker exec "$CN" sh -c "cat /tmp/p4/$1" 2>/dev/null; fi; }
c_cp()  { if [ "$MOCK" = 1 ]; then [ -e "$MOCKCTR/$1" ] && cp -r "$MOCKCTR/$1/." "$2/" 2>/dev/null; else docker cp "$CN:/tmp/p4/$1/." "$2/" >/dev/null 2>&1; fi; }
cost_line() {  # secs rc note
  CAP0_REASON="$ATTEMPT_REASON" python3 - "$COSTS" "$ATTEMPT" "${START:-}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$1" "$GPUS" "${IMG_ID:-}" "$2" "$CAP" "$MOCK" "$APPROVED_GPU_H" "$3" "$CAP0_LOC" <<'PYEOF'
import json, os, sys
c, att, start, end, secs, gpus, img, rc, cap, mock, appr, note, loc = sys.argv[1:]
secs = int(secs)
line = dict(item="cap0-session", attempt=int(att), attempt_reason=os.environ["CAP0_REASON"], start=start or end, end=end,
            seconds=secs, gpus=int(gpus) if secs > 0 else 0, gpu_hours=round(int(gpus) * secs / 3600, 4) if secs > 0 else 0.0,
            image=img or None, rc=None if rc == "null" else int(rc), wall_cap=int(cap), mock=mock == "1",
            approved_gpu_h=float(appr), note=note or None, loc=loc,
            purpose="capture-p0: point-0 exact-byte capture + warm-prefill localisation (stock 0.9.1, R01, TP 2)")
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
      docker exec "$CN" sh -c 'cd /usr/local/lib/python3.12/dist-packages/vllm && sha256sum p4_hooks.py cap0_hooks.py cap0_hooks_layout.py forcewin_hooks.py models/glm5next/nvidia/model.py models/glm5next/nvidia/kda.py v1/worker/gpu/model_runner.py; env | grep -E "^(GLM53|VLLM|FORCEWIN|B12X)" | sort' > "$AO/container-files-env.txt" 2>&1
    fi
    ( cd "$ST" && ./stop.sh ) >> "$AO/stop.log" 2>&1
    docker ps --format '{{.Names}}' | grep -q "^$CN\$" && docker stop -t 30 "$CN" >> "$AO/stop.log" 2>&1
  fi
  local SECS=$(( $(date +%s) - T0 ))
  cost_line "$SECS" "${RC_MAIN:-null}" "stage $STAGE"
  echo "cap0 session attempt $ATTEMPT occupied ${SECS}s x $GPUS GPU = $(python3 -c "print(round($GPUS*$SECS/3600,4))") GPU-h (rc ${RC_MAIN:-null})"
  mkdir -p "$AO/p4log"
  c_cp log "$AO/p4log"
  ( umask 077; mkdir -p "$PO/p4" "$PO/fw" ) && chmod 700 "$PRIVROOT" "$PO" "$PO/p4" "$PO/fw"
  ( umask 077; c_cp PRIVATE "$PO/p4"; c_cp PRIVATE-fw "$PO/fw" )
  chmod -R go-rwx "$PO" 2>/dev/null
  echo "private files: $(find "$PO" -type f | wc -l) under $PO (mode 0700)"
  if [ "$MOCK" != 1 ] || [ -f "$P3B_CAPLOG" ]; then
    python3 "$TILE/runner/gates_p4.py" --arm A --binds "$AO/p4log" --p3b "$P3B_CAPLOG" --out "$AO/gates-final.json" --stage final 2>&1 | tee -a "$AO/gate.txt"
  fi
  python3 "$HERE/outcome_cap0.py" "$AO" "$ATTEMPT" 2>&1 | tee "$AO/outcome-console.txt"
  if [ "$STAGE" = flush ] || [ "$STAGE" = done ]; then
    "$PKG/analysis/run_cap0_analysis.sh" "$AO" "$PO/p4" 2>&1 | tail -n 12 | tee "$AO/analysis-console.txt"
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
  if [ "$MOCK" = 1 ]; then pkill -f "tests/mock_server_cap0[.]py" 2>/dev/null; else docker stop -t 30 "$CN" >/dev/null 2>&1; fi ) &
WD=$!
echo "== start $START image=$IMG ($IMG_ID) cap ${CAP}s session attempt $ATTEMPT loc $CAP0_LOC"
if [ "$MOCK" = 1 ]; then
  mkdir -p "$MOCKCTR"
  python3 "$PKG/tests/mock_server_cap0.py" "$MOCKCTR" "${MOCK_PORT:-18061}" "$CAP0_LOC" > "$AO/mock-server.log" 2>&1 &
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
# G-instr: local-p4 schema 2 + stage-0 recording on both ranks; cap0 targets resolved with the served layout on both
# ranks (rank 0: MLA/indexer 3 and 7 + KDA 0/2/4; rank 1: KDA 0/2/4, MLA owner map split:25); no hook error.  These
# stop the session before traffic.  The localisation hooks (rank 0: 12, rank 1: 11 -- no indexer on the non-owner
# rank) and the conv wrapper are checked too, but a shortfall only DEGRADES the option: the core capture proceeds and
# the analysis's G-loc marks the localisation INCOMPLETE (a start-up is not spent on the optional part).
INIT=$(CAP0_LOC="$CAP0_LOC" python3 - "$AO/p4log-startup" <<'PY'
import glob, json, os, sys
recs = []
for p in glob.glob(os.path.join(sys.argv[1], "p4_r*_p*.jsonl")) + glob.glob(os.path.join(sys.argv[1], "cap0_r*_p*.jsonl")):
    recs += [json.loads(l) for l in open(p) if l.strip()]
loc = os.environ["CAP0_LOC"] == "on"
init = {r.get("rank"): r for r in recs if r.get("kind") == "init"}
c0 = {r.get("rank"): r for r in recs if r.get("kind") == "cap0_init"}
errs = [r for r in recs if r.get("kind") in ("p4_error", "cap0_hook_problem")]
why, warn = [], []
if set(init) != {0, 1} or not all(i.get("s0") and i.get("schema") == "local-p4/2" for i in init.values()):
    why.append("local-p4 schema 2 / stage-0 recording not on both ranks")
if set(c0) != {0, 1}:
    why.append("cap0_init missing on a rank")
want = {0: {"mla3", "mla7", "idx3", "idx7", "kda0", "kda2", "kda4"}, 1: {"kda0", "kda2", "kda4"}}
hooks = {0: 12, 1: 11}
for rk, r in c0.items():
    t = r.get("targets", {})
    if not want[rk] <= set(t) or not all(v.get("ok") for v in t.values()):
        why.append("rank %d targets %s (layout ok %s)" % (rk, sorted(t), {k: v.get("ok") for k, v in t.items()}))
    if r.get("mla_owners") != "split:25":
        why.append("rank %d MLA owner map %r (expected split:25)" % (rk, r.get("mla_owners")))
    if bool(r.get("loc")) != loc or (not loc and (r.get("hooks") or r.get("conv_patched"))):
        why.append("rank %d localisation switch %s, hooks %d (session loc %s)" % (rk, r.get("loc"), len(r.get("hooks", [])), loc))
    elif loc and (len(r.get("hooks", [])) != hooks[rk] or not r.get("conv_patched")):
        warn.append("rank %d hooks %d of %d, conv_patched %s" % (rk, len(r.get("hooks", [])), hooks[rk], r.get("conv_patched")))
if [e for e in errs if e.get("kind") == "p4_error"]:
    why.append("%d hook errors: %s" % (len(errs), str(errs[:2])[:300]))
elif errs:
    warn.append("hook problems: %s" % str(errs[:2])[:300])
print(("OK" if not why else "FAIL") + " p4 init ranks %s; cap0_init ranks %s; %s%s" % (
    sorted(init), sorted(c0), "; ".join(why) or "core targets and layout as designed",
    ("; LOCALISATION DEGRADED (the core capture proceeds; the analysis marks localisation INCOMPLETE): " + "; ".join(warn)) if warn else ""))
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
case "$INIT" in OK*) ;; *) echo "INCOMPLETE: capture instrumentation not installed as designed (G-instr)"; RC_MAIN=8; exit 8;; esac
[ "$FWREG" = 2 ] && [ "$FWERR" = 0 ] || { echo "INCOMPLETE: forcewin not registered on both ranks"; RC_MAIN=8; exit 8; }
python3 "$TILE/runner/gates_p4.py" --arm A --binds "$AO/p4log-startup" --p3b "$P3B_CAPLOG" --out "$AO/gates-startup.json" --stage startup 2>&1 | tee -a "$AO/gate.txt"
GRC=${PIPESTATUS[0]}
[ "$GRC" = 0 ] || { echo "INCOMPLETE: G3 failed before traffic"; RC_MAIN=8; exit 8; }
STAGE="client"
DEADLINE=$((T0 + CAP - RESERVE))
echo "== client (deadline in $((DEADLINE - $(date +%s)))s)"
if [ "$MOCK" = 1 ]; then TESTENV="CAP0_TEST_MOCK=1"; else TESTENV="CAP0_TEST_MOCK=0 CAP0_TEST_WARM_TOKENS="; fi
timeout $((DEADLINE - $(date +%s) + 30)) env $TESTENV CAP0_WINDOWS="$WINDOWS" CAP0_LOC="$CAP0_LOC" ${CAP0_REPS:+CAP0_REPS="$CAP0_REPS"} \
  python3 "$HERE/client_cap0.py" "$AO" "$DEADLINE" "$BASE" > "$AO/client.log" 2>&1 &
CPID=$!
wait "$CPID"; RC_MAIN=$?
CPID=""
echo "client rc=$RC_MAIN" | tee -a "$AO/gate.txt"
STAGE="flush"
EXPN=$(python3 -c "
import json
L=[json.loads(l) for l in open('$AO/client.jsonl')]
done=[d['replicate'] for d in L if d.get('kind')=='replicate_done']
loc={r['name']:r['loc'] for d in L if d.get('kind')=='start' for r in d['replicates']}
print(2*len(done)+2*sum(1 for r in done if loc.get(r)))" 2>/dev/null)
for i in $(seq 1 60); do
  GOT=$(c_cat "log/cap0_r*_p*.jsonl" | grep -cE '"kind": "cap0_(capture|loc_flushed)"')
  [ "${GOT:-0}" -ge "${EXPN:-0}" ] && break
  sleep 1
done
echo "cap0 records flushed ${GOT:-0} of ${EXPN:-0} expected (wait ${i}s)" | tee -a "$AO/gate.txt"
STAGE="done"
exit 0
