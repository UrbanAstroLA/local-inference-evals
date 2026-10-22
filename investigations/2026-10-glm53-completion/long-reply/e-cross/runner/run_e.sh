#!/usr/bin/env bash
set -u
# [E copy of bplus/runner/run_bplus.sh (diff in checks/runner.diff): set -u; engines v070 v091; E's ceilings; the B1
#  entries, B1 pin and B1 tag check dropped with the engine; package-path guard; finalize runs scripts/analysis_e.py.
#  Amendment 22ah: 13 prefixes, 52 continuations per session; ceiling and budget walls from the recomputed out/cost.json.]
# E prefix-source x engine continuation runner (PROPOSAL.md section 7). PREPARED, NOT RUN; needs the operator's GPU authorization
# and the frozen package (SHA256SUMS).
#   APPROVED_GPU_H  occupied GPU-hours the operator approved (required; 0 < x <= PLAN_MAX_GPU_H). The watchdog stops
#                   everything at APPROVED_GPU_H / 2 GPUs of wall time from script start (the only early stop).
#   DRY_RUN=1       every refusal check, read-only (docker ps / docker image inspect / nvidia-smi / ss / df); then exits
#                   before creating anything or starting any process or container.
# One invocation, two sessions, one after another, each on a FRESH server started by digest through the engine's own
# stack launcher (loopback publish 127.0.0.1:8001): 1 stock 0.7.0, 2 stock 0.9.1. Per session: GPU pre-flight
# (desktop graphics clients filtered, as rep070), budget rule (a session starts only if its frozen ceiling wall still fits
# in the approved wall), start, release warm-up gate (ready file + /v1/models, as the eval chain), the B+ client
# (scripts/client_bplus.py: 52 continuations, 8 in flight, fixed order), server log + docker inspect, stop, GPU release,
# cost line, compile-cache inventory. A failed session stops the run (no retry, no replacement, later engines not
# started). At the end the frozen analysis runs in a CPU-only container: COMPLETE -> E1 screen and reading; anything else
# -> INCOMPLETE or INVALID with no reading. Modelled on rep070/refreeze/runner/run_rep070.sh (refusal set, desktop filter,
# escaped watchdog pattern, cleanup on every exit path). Not set -e: every status query's exit code is checked.
PLAN_MAX_GPU_H=19.9          # out/cost.json ceiling (sum of the per-session 99.9th percentiles of the full-horizon session at pessimistic rates,
                             # both engines, inter-session gaps); no invalid-run allowance
declare -A SESSION_MIN_S=([v070]=20700 [v091]=14941)   # per-session ceiling walls (out/cost.json)
H=$(cd "$(dirname "$0")" && pwd); PKG=$(dirname "$H")
EC=<runs>/long-reply/e-cross
case "$PKG" in "$EC"|"$EC"/*) ;; *) echo "ABORT: package path $PKG is not e-cross or inside it"; exit 1 ;; esac
RUN="$PKG/run"; LEDGER="$RUN/costs.jsonl"
case "$RUN" in "$EC"/run|"$EC"/*/run) ;; *) echo "ABORT: run folder $RUN is not inside e-cross"; exit 1 ;; esac
C=glm53-flash-exl3-b12x-vllm
ENGINES="v070 v091"
declare -A RECIPE=([v070]="glm53-flash-exl3-k3.25@tpurtell" [v091]="glm53-flash-exl3-k3.25@tpurtell+v0.9.1")
declare -A SD=([v070]=<stacks>/glm53-flash-v0.7.0 [v091]=<stacks>/glm53-flash-v0.9.0)
declare -A IMG_REF=([v070]="ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629"
                    [v091]="ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07")
declare -A IMG_ID=([v070]="sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629"
                   [v091]="sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07")
declare -A CACHE=([v070]=<cache>/vllm [v091]=<cache>/vllm-v090)
READY_FILE=/tmp/glm53-release-ready
CI=<runs>/rep070/refreeze/runner/cache_inventory.py
HUB=<models>
declare -A PIN=(
  ["<recipes>/glm53-flash-exl3-k3.25@tpurtell/env.sh"]=ed50bc7d83751258b4707e3b9d3c5dba43b41b4d6587d5d81277d02758cea2b6
  ["<recipes>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1/env.sh"]=6186bbb66ed982ac942d62ce1dda0a2b66cc63e7af015de717e66043f263414c
  ["<stacks>/glm53-flash-v0.7.0/start.sh"]=c852511c1b13264aa750f78ed78ffccc15a3a67569179e2d84344a678378ca57
  ["<stacks>/glm53-flash-v0.7.0/local-env.sh"]=f1f3ced5dc85968cb31aabd102575d121f673e1a0f619919c5855d8fed8fb4d0
  ["<stacks>/glm53-flash-v0.7.0/start.sh"]=869b0c0af97865c637986f5dfe6be9886e511514cb3b24818208052b505b5fe3
  ["<stacks>/glm53-flash-v0.7.0/model-profiles.sh"]=13c07d5b61953174340ab8c000d68bd84368ad653bbfe66b60bcd80dc4cd17b3
  ["<stacks>/glm53-flash-v0.9.0/start.sh"]=e1196cbcd6b4744cb41fdebf3d12d8586dbea464e702af5de47617837efe0e5e
  ["<stacks>/glm53-flash-v0.9.0/local-env.sh"]=f4658b81eeb4c4d5403a95795f10cf5962eedd8cc5161372356ac725a1404fe6
  ["<stacks>/glm53-flash-v0.9.0/start.sh"]=869b0c0af97865c637986f5dfe6be9886e511514cb3b24818208052b505b5fe3
  ["<stacks>/glm53-flash-v0.9.0/model-profiles.sh"]=a4ccf85dc1e1dd75acdb8549ecaec82137c9f77b47e5bb6464d82f04ef5537f1
  ["$CI"]=d317acd0ae949337a61f4a970c7091e57e1e932a4e1463c4c98539cf8a4dfec5
)
declare -A SNAPLIST=(
  ["$HUB/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1/snapshots/701cd7456c13d87bf0147ad946f828a999afb59c"]=<runs>/rep070/refreeze/checks/snapshot-701cd745.txt
  ["$HUB/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1/snapshots/0490d2f708b12145f6516555ab066aaeb401cd21"]="$PKG/checks/snapshot-0490d2f7.txt"
  ["$HUB/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a"]=<runs>/rep070/refreeze/checks/snapshot-dflash2-bf582e4e.txt
)
[ -n "${APPROVED_GPU_H:-}" ] || { echo "ABORT: APPROVED_GPU_H is not set (the operator's approved occupied GPU-hours)"; exit 1; }
python3 -c "import sys; x=float(sys.argv[1]); sys.exit(0 if 0 < x <= float(sys.argv[2]) else 1)" "$APPROVED_GPU_H" "$PLAN_MAX_GPU_H" 2>/dev/null \
  || { echo "ABORT: APPROVED_GPU_H=$APPROVED_GPU_H must be a number in (0, $PLAN_MAX_GPU_H] (the plan's ceiling)"; exit 1; }
CAP_S=$(python3 -c "import sys; print(int(float(sys.argv[1])*3600/2))" "$APPROVED_GPU_H")   # 2 GPUs
T0=$(date +%s)
echo "$(date -Is) E runner start; engines $ENGINES; approved $APPROVED_GPU_H GPU-h (${CAP_S} s wall)${DRY_RUN:+; DRY_RUN=$DRY_RUN}"
# ---------------------------------------------------------------- refusals (nothing is created or started before all pass)
[ -f "$PKG/SHA256SUMS" ] || { echo "ABORT: $PKG/SHA256SUMS missing (package not frozen)"; exit 1; }
( cd "$PKG" && sha256sum -c --quiet SHA256SUMS ) > /dev/null 2>&1 || { echo "ABORT: $PKG/SHA256SUMS does not verify:"; ( cd "$PKG" && sha256sum -c --quiet SHA256SUMS ) 2>&1 | head -20; exit 1; }
for F in "${!PIN[@]}"; do
  [ -f "$F" ] || { echo "ABORT: pinned file missing: $F"; exit 1; }
  [ "$(sha256sum "$F" | cut -c1-64)" = "${PIN[$F]}" ] || { echo "ABORT: pinned file changed: $F"; exit 1; }
done
PRIV_SHA=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['private_file_sha256'], d['private_prompts_sha256'])" "$PKG/out/prefix_manifest.json" 2>/dev/null)
[ "$PRIV_SHA" = "$(sha256sum "$PKG/PRIVATE/prefixes.json" 2>/dev/null | cut -c1-64) $(sha256sum "$PKG/PRIVATE/prompts.json" 2>/dev/null | cut -c1-64)" ] \
  || { echo "ABORT: PRIVATE/prefixes.json or prompts.json does not match the frozen manifest"; exit 1; }
[ "$(stat -c %a "$PKG/PRIVATE")" = 700 ] && [ "$(stat -c %a "$PKG/PRIVATE/prefixes.json")" = 600 ] || { echo "ABORT: PRIVATE/ must be mode 0700 and its files 0600"; exit 1; }
for D in "${!SNAPLIST[@]}"; do
  LST=$(cd "$D" 2>/dev/null && find . \( -type l -o -type f \) | LC_ALL=C sort | while read -r f; do echo "$f -> $(basename "$(readlink "$f")")"; done)
  [ -n "$LST" ] && [ "$LST" = "$(cat "${SNAPLIST[$D]}")" ] || { echo "ABORT: snapshot listing differs from the frozen one: $D"; exit 1; }
done
# shipped configuration: no launcher variable of either stack may be set by the caller (exempt: MODEL_PROFILE, CACHE_DIR,
# HF_HOME, which the env files export unconditionally). IMAGE / PORT / GPU_MEMORY_UTILIZATION from the caller are refused.
VARS=$(cat <stacks>/glm53-flash-v0.7.0/start.sh <stacks>/glm53-flash-v0.9.0/start.sh | grep -oE '\$\{[A-Z_][A-Z0-9_]*(:-|-)' | sed -E 's/^\$\{//; s/(:-|-)$//' | sort -u)
[ -n "$VARS" ] || { echo "ABORT: could not read the launchers' variables"; exit 1; }
SETV=""
for v in $(printf '%s\n' $VARS IMAGE PORT GPU_MEMORY_UTILIZATION | sort -u); do
  case "$v" in MODEL_PROFILE|CACHE_DIR|HF_HOME) continue ;; esac
  [ -n "${!v+x}" ] && SETV="$SETV $v"
done
[ -n "$SETV" ] && { echo "ABORT: override variables set in the environment (not the shipped configuration):$SETV"; exit 1; }
for e in $ENGINES; do
  RES=$(env -u IMAGE -u PORT bash -c "source '<recipes>/${RECIPE[$e]}/env.sh' && source '${SD[$e]}/local-env.sh' && echo \"\$IMAGE \$PORT\""); [ $? -eq 0 ] || { echo "ABORT: cannot resolve $e's launcher IMAGE/PORT"; exit 1; }
  [ "$RES" = "${IMG_REF[$e]} 127.0.0.1:8001" ] || { echo "ABORT: $e launcher resolves to '$RES' (expected the pinned image and 127.0.0.1:8001)"; exit 1; }
  GOT=$(docker image inspect "${IMG_REF[$e]}" --format '{{.Id}}' 2>/dev/null)
  [ "$GOT" = "${IMG_ID[$e]}" ] || { echo "ABORT: $e image ${IMG_REF[$e]} has Id '${GOT:-missing}', not ${IMG_ID[$e]}"; exit 1; }
done
[ -e "$RUN" ] && { echo "ABORT: $RUN exists (an earlier attempt; no automatic retry)"; exit 1; }
DPS=$(docker ps --format '{{.Names}} {{.Image}}'); [ $? -eq 0 ] || { echo "ABORT: docker ps failed"; exit 1; }
echo "$DPS" | grep -qE '^glm53' && { echo "ABORT: a GLM container is running:"; echo "$DPS" | grep -E '^glm53'; exit 1; }
for e in $ENGINES; do echo "$DPS" | grep -q "${IMG_ID[$e]:7:12}" && { echo "ABORT: a container of the $e image is running"; exit 1; }; done
CLX='scripts/client_bplus.py'
pgrep -f -- "$CLX" > /dev/null 2>&1 && { echo "ABORT: a B+ client process is running"; exit 1; }
pgrep -f -- 'bin/lm_eval --model' > /dev/null 2>&1 && { echo "ABORT: an lm_eval process is running"; exit 1; }
PL=$(ss -ltnH '( sport = :8001 )'); [ $? -eq 0 ] || { echo "ABORT: ss failed"; exit 1; }
[ -n "$PL" ] && { echo "ABORT: port 8001 is in use:"; echo "$PL"; exit 1; }
FREE_KB=$(df -Pk /data | awk 'NR==2{print $4}'); [ -n "$FREE_KB" ] && [ "$FREE_KB" -ge 5242880 ] || { echo "ABORT: less than 5 GiB free on /data"; exit 1; }
gpu_idle() {  # 0 if idle: no compute process except the Claude Desktop GPU process; no GPU client except the desktop graphics
              # clients (amendments 16d/17c filter); GPU0 <= 1 GiB, GPU1 <= 4 GiB, util <= 10%; exactly 2 GPUs
  nvidia-smi -L > /dev/null 2>&1 || { echo "nvidia-smi failed"; return 1; }
  local CA P PROCS NG=0 IDX MEM UTIL LIM
  CA=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); [ $? -eq 0 ] || { echo "compute-apps query failed"; return 1; }
  for P in $CA; do tr '\0' ' ' < "/proc/$P/cmdline" 2>/dev/null | grep -q -- '--type=gpu-process' || { echo "compute process $P on the GPUs"; return 1; }; done
  PROCS=$(nvidia-smi 2>/dev/null | awk '/Processes:/{p=1;next} p && /^\|/ && $2 ~ /^[0-9]+$/ {print}' \
    | grep -vE '[[:space:]]G[[:space:]].*(gnome-shell|Xwayland|xdg-desktop-portal|firefox|snap-store)' \
    | grep -vE '[[:space:]]C\+G[[:space:]].*--disable-logging')
  [ -n "$PROCS" ] && { echo "GPU clients other than the desktop graphics clients: $PROCS"; return 1; }
  while read -r IDX MEM UTIL; do
    NG=$((NG+1)); LIM=1024; [ "$IDX" = 1 ] && LIM=4096
    { [ "$MEM" -le "$LIM" ] && [ "$UTIL" -le 10 ]; } || { echo "GPU$IDX busy mem=${MEM}MiB util=${UTIL}%"; return 1; }
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
  [ "$NG" = 2 ] || { echo "expected 2 GPUs, saw $NG"; return 1; }
  return 0
}
MSG=$(gpu_idle) || { echo "ABORT: GPU pre-flight: $MSG"; exit 1; }
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "DRY_RUN: all refusal checks passed. A real run would now: create $RUN (raw/ mode 0700); start the GPU logger (60 s)"
  echo "  and the watchdog (${CAP_S} s from script start); then for each engine in order $ENGINES: GPU pre-flight; budget rule"
  echo "  (start only if >= the session's ceiling wall remains: v070 ${SESSION_MIN_S[v070]} s, v091 ${SESSION_MIN_S[v091]} s);"
  echo "  fresh server by digest via the stack launcher (127.0.0.1:8001); ready-file + API gate; client_bplus.py (52 requests,"
  echo "  8 in flight, fixed order, seeds from out/plan.json); save server log and docker inspect; stop; GPU release; cost line;"
  echo "  compile-cache inventory. Then the frozen analysis in a CPU-only container. Nothing was created or started."
  exit 0
fi
# ---------------------------------------------------------------- run
mkdir -p "$RUN" || { echo "ABORT: cannot create $RUN"; exit 1; }
( umask 077; mkdir -p "$RUN/raw" ) && chmod 700 "$RUN/raw" || { echo "ABORT: cannot create $RUN/raw"; exit 1; }
( cd "$PKG" && sha256sum -c SHA256SUMS ) > "$RUN/package-verify.txt" 2>&1
for e in $ENGINES; do echo "$e ${IMG_REF[$e]} ${IMG_ID[$e]}"; done > "$RUN/image-ids.txt"
( SL=""; trap '[ -n "$SL" ] && kill $SL 2>/dev/null; exit 0' TERM INT
  while true; do echo "$(date -Is) $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate --format=csv,noheader | tr '\n' ' ') apps: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')"; sleep 60 & SL=$!; wait $SL; done ) > "$RUN/nvidia-smi.log" 2>&1 &
GPULOG=$!
RX=$(printf '%s' "$PKG/$CLX" | sed 's/[][\.*^$+?(){}|]/\\&/g')   # literal match of the client's command line
( SL=""; trap '[ -n "$SL" ] && kill $SL 2>/dev/null; exit 0' TERM INT
  while true; do sleep 60 & SL=$!; wait $SL; [ $(( $(date +%s)-T0 )) -ge $CAP_S ] && { echo "$(date -Is) CEILING REACHED: stopping"; pkill -f -- "$RX"; docker stop $C; break; }; done ) >> "$RUN/watchdog.log" 2>&1 &
WD=$!
CLP=""; STARTED=0; COSTED=1; DONE=0; FINAL=0; CRC=""; s0=""; E=""; TOTAL_GH=0
sess() {  # engine key=value...  merge into RUN/engine/session.json
  local e=$1; shift
  python3 - "$RUN/$e/session.json" "$@" <<'EOF'
import json, os, sys
p = sys.argv[1]
d = json.load(open(p)) if os.path.exists(p) else {}
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    try:
        d[k] = json.loads(v)
    except ValueError:
        d[k] = v
json.dump(d, open(p, "w"), indent=1)
EOF
}
cost_line() {   # once per session: 2 GPUs x (session start -> server stopped)
  [ "$COSTED" = 1 ] && return; COSTED=1
  local w=$(( $(date +%s)-s0 )); local gh; gh=$(python3 -c "import sys; print(round(2*int(sys.argv[1])/3600,3))" "$w")
  local wdf=false; grep -q "CEILING REACHED" "$RUN/watchdog.log" 2>/dev/null && wdf=true
  echo "{\"engine\": \"$E\", \"recipe\": \"${RECIPE[$E]}\", \"image\": \"${IMG_ID[$E]}\", \"start\": $s0, \"stop\": $(date +%s), \"wall_s\": $w, \"gpus\": 2, \"gpu_hours\": $gh, \"client_rc\": ${CRC:-null}, \"watchdog_fired\": $wdf}" >> "$LEDGER"
  TOTAL_GH=$(python3 -c "import sys; print(round(float(sys.argv[1])+float(sys.argv[2]),3))" "$TOTAL_GH" "$gh")
  sess "$E" "watchdog=$wdf" "gpu_hours=$gh" "wall_s=$w"
}
stop_server() {
  [ -s "$RUN/$E/server.log" ] || docker logs $C > "$RUN/$E/server.log" 2>&1
  [ -s "$RUN/$E/inspect.json" ] || docker inspect $C > "$RUN/$E/inspect.json" 2>/dev/null
  docker stop $C > /dev/null 2>&1; local src=$?; echo "$(date -Is) $E: docker stop rc=$src"
}
finalize() {   # runs once, on every exit path after the first session started
  [ "$FINAL" = 1 ] && return; FINAL=1
  RUN_DIR="$RUN" bash "$PKG/scripts/run_cpu.sh" v070 scripts/analysis_e.py /work/run/result.json > "$RUN/result.md" 2> "$RUN/result.err"
  echo "$(date -Is) analysis: $(grep -m1 '^# E ' "$RUN/result.md"); $(grep -oE '^exit [0-9]+' "$RUN/result.md" | tail -1) (0 complete, 3 invalid, 4 incomplete)"
}
cleanup() {
  [ "$DONE" = 1 ] && return; DONE=1
  local KIDS; KIDS=$(pgrep -P "$GPULOG" 2>/dev/null; pgrep -P "$WD" 2>/dev/null)
  kill $GPULOG $WD 2>/dev/null; [ -n "$KIDS" ] && kill $KIDS 2>/dev/null
  if [ -n "$CLP" ] && kill -0 $CLP 2>/dev/null; then pkill -f -- "$RX"; echo "$(date -Is) client interrupted"; fi
  if [ "$STARTED" = 1 ] && [ "$COSTED" != 1 ]; then
    stop_server; cost_line; sess "$E" "completed=false"
    echo "$(date -Is) cleanup after an interrupted $E session: server stopped, cost line written"
  fi
  [ -n "$E" ] && finalize
  docker ps --format '{{.Names}}' | grep -qE '^glm53' && echo "WARN: GLM container still running" || echo "no GLM container running"
}
trap cleanup EXIT
trap 'echo "$(date -Is) signal received"; exit 4' INT TERM
python3 -B "$CI" snap <cache>/vllm "$RUN/cache-v070-before.json"; python3 -B "$CI" snap <cache>/vllm-v090 "$RUN/cache-v09x-before.json"
STOPPED=""
for e in $ENGINES; do
  mkdir -p "$RUN/$e"
  if [ -n "$STOPPED" ]; then sess "$e" "budget_stop=$([ "$STOPPED" = budget ] && echo true || echo false)" "started=false" "not_started_because=\"$STOPPED\""; continue; fi
  LEFT=$(( CAP_S - ($(date +%s)-T0) ))
  if [ "$LEFT" -lt "${SESSION_MIN_S[$e]}" ]; then
    echo "$(date -Is) $e: ${LEFT} s of approved wall left < ${SESSION_MIN_S[$e]} s (its ceiling): not started (budget stop)"
    sess "$e" "budget_stop=true" "started=false"; STOPPED=budget; continue
  fi
  OKG=0
  for i in $(seq 1 30); do MSG=$(gpu_idle) && { OKG=1; break; }; sleep 10; done
  sess "$e" "preflight_ok=$([ $OKG = 1 ] && echo true || echo false)" "preflight_msg=\"${MSG:-ok}\"" "gpu_state_before=\"$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader | tr '\n' ';')\""
  [ "$OKG" = 1 ] || { echo "$(date -Is) $e: GPU pre-flight failed ($MSG): stopping"; STOPPED=preflight; continue; }
  E=$e; s0=$(date +%s); COSTED=0; CRC=""; STARTED=1
  sess "$e" "started=true" "budget_stop=false" "t_start=$s0"
  echo "$(date -Is) $e: starting ${RECIPE[$e]} (${IMG_REF[$e]}) via ${SD[$e]}/start.sh"
  ( cd "${SD[$e]}" && bash -c "source '<recipes>/${RECIPE[$e]}/env.sh' && ./start.sh" ) > "$RUN/$e/start.log" 2>&1
  SRC=$?
  READY=0; t1=$(date +%s)
  if [ "$SRC" = 0 ]; then
    while true; do
      if docker exec $C test -f "$READY_FILE" 2>/dev/null && curl -sf -m 5 http://127.0.0.1:8001/v1/models > /dev/null; then READY=1; break; fi
      [ "$(docker inspect -f '{{.State.Running}}' $C 2>/dev/null)" = true ] || { echo "$(date -Is) $e: CONTAINER DIED"; break; }
      if docker logs $C 2>&1 | grep -qE 'is less than desired GPU memory utilization|Engine core initialization failed|CUDA out of memory'; then echo "$(date -Is) $e: ENGINE FAILED TO START"; break; fi
      [ $(( $(date +%s)-t1 )) -gt 2400 ] && { echo "$(date -Is) $e: NOT READY in 40 min"; break; }
      sleep 15
    done
  else
    echo "$(date -Is) $e: launcher failed (rc $SRC)"
  fi
  sess "$e" "ready=$([ $READY = 1 ] && echo true || echo false)" "ready_after_s=$(( $(date +%s)-t1 ))" "launcher_rc=$SRC"
  if [ "$READY" = 1 ]; then
    docker logs $C 2>&1 | grep -E "GPU KV cache size|Maximum concurrency|Available KV cache memory|Loading weights took|attention block size" | tail -8 > "$RUN/$e/startup-markers.txt"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > "$RUN/$e/compute-apps-at-ready.txt" 2>&1
    echo "$(date -Is) $e: ready; client starts"
    python3 -B "$PKG/$CLX" "$e" "$RUN" > "$RUN/$e/client.out" 2>&1 &
    CLP=$!; wait $CLP; CRC=$?; CLP=""
    echo "$(date -Is) $e: client rc=$CRC ($(tail -1 "$RUN/$e/client.out"))"
  fi
  stop_server
  python3 - "$RUN/$e/inspect.json" "${IMG_ID[$e]}" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))[0]
    b = (d.get("HostConfig") or {}).get("PortBindings") or {}
    hosts = sorted({x.get("HostIp") for v in b.values() for x in (v or [])})
    print(f"  inspect: image {'OK' if d.get('Image') == sys.argv[2] else 'MISMATCH ' + str(d.get('Image'))}; published on {hosts} "
          f"({'loopback only' if hosts == ['127.0.0.1'] else 'NOT loopback only'})")
except Exception as e:  # noqa: BLE001
    print(f"  inspect: unreadable ({e!r})")
EOF
  for i in $(seq 1 30); do gpu_idle > /dev/null && break; sleep 10; done
  sess "$e" "gpu_state_after=\"$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader | tr '\n' ';')\""
  cost_line; STARTED=0
  PREV="$RUN/cache-v070-before.json"; [ "$e" = v091 ] && PREV="$RUN/cache-v09x-before.json"
  python3 -B "$CI" snap "${CACHE[$e]}" "$RUN/cache-$e-after.json"
  python3 -B "$CI" diff "$PREV" "$RUN/cache-$e-after.json" "$RUN/cache-diff-$e.json"
  echo "  $e: warm-up lines $(grep -c 'GLM release startup ' "$RUN/$e/server.log"); preemption lines $(grep -ciE 'preempt' "$RUN/$e/server.log")"
  if [ "$READY" = 1 ] && [ "$CRC" = 0 ] && ! grep -q "CEILING REACHED" "$RUN/watchdog.log" 2>/dev/null; then
    sess "$e" "completed=true" "client_rc=$CRC"
  else
    sess "$e" "completed=false" "client_rc=${CRC:-null}"
    echo "$(date -Is) $e: session did not complete: stopping (no automatic retry; later engines not started)"; STOPPED=failure
  fi
done
finalize
echo "$(date -Is) E DONE; total wall $(( $(date +%s)-T0 ))s, $TOTAL_GH GPU-h (ledger $LEDGER)"
