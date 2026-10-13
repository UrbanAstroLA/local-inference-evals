#!/usr/bin/env bash
# B1 checkpoint pass, seed 1236 (PLAN.md amendment 19). PREPARED, NOT RUN; needs the operator's GPU authorization.
# Copy of ../bundle/screen2/gpqa-bundle-screen-s1235.sh (sha256 39ff7616...), changed only for:
#   - checkpoint B1: recipe glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234+d7 (env.sh differs from bundle-d1234's
#     only in IMAGE and comments), image id pinned, refusal if the tag local/glm53-v0.9.1-b1-d7:local moved;
#   - seed 1236 = protocol pass 3. eval-chain.sh sends request seed 1233+pass (GPQA_SEED_PER_PASS=1) and lm-eval
#     --seed 1234+pass, and skips any existing <recipe>__pass<N> directory. screen2 reached pass 2 because B0's real
#     __pass1 existed. B1 has no pass 1 or 2, so this runner creates two EMPTY placeholder directories __pass1 and
#     __pass2 just before the chain and removes them (rmdir: empty only) on every exit path. GPQA_PASSES=3 then runs
#     pass 3 alone, first pass of a FRESH server: request seed 1236, lm-eval --seed 1237 -- the comparators' seed-1236
#     passes carry the same pair (record +ps__pass3, arm __pass3, 0.7.0 +ps__pass3; each was the SECOND pass of its server
#     session). The new directory is <B1>__pass3: no existing directory has the B1 recipe name;
#   - output in b1-screen/run-s1236, request log <B1>.s1.requests.jsonl (B1 session 1), cost line in b1-screen/costs.jsonl;
#   - frozen analysis b1screen_analysis.py (also given the container inspect record);
#   - pre-flight additions: b1-screen/SHA256SUMS (every file that runs, incl. the chain, proxy and launcher) must verify;
#     no earlier B1 session (costs.jsonl empty); no launcher/chain override variable set in the environment (shipped
#     configuration); ports 8001/8002 free; the recording proxy must be up before the chain starts; GPU idle check with
#     the amendment 16d/17c desktop filter (type-G gnome-shell/Xwayland/xdg-desktop-portal/firefox/snap-store and the
#     Claude Desktop C+G GPU process) in addition to screen2's compute-app check and memory/utilization limits;
#   - watchdog: the chain-kill pattern is regex-escaped (the earlier runners' unescaped "+" never matched the chain);
#   - cleanup on every exit path (EXIT/INT/TERM): chain (run in the background and waited for, so a signal is handled at
#     once), proxy, loggers, watchdog, server stop, one cost line, placeholders.
#   - DRY_RUN=1 runs every refusal check, then exits before creating anything or starting any process or container.
# Protocol gpqa-diamond/v1 unchanged: 198 questions, concurrency 8, cap 327,680, temperature 1.0, top_p 0.95, thinking on,
# normal stopping, recording proxy, shipped graph configuration (no EXTRA_SERVE_ARGS), release warm-up marker gate of the
# chain (ready file + API). Traces stay local. Only early stop: the 7.5 occupied-GPU-hour watchdog (2 GPUs x wall from
# script start; 13,500 s). No automatic retry and no further pass. Afterwards b1screen_analysis.py runs (PROVISIONAL until
# the stated-answer adjudication file). Not set -e: every status query's exit code is checked explicitly.
R="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234+d7"; G=<gpqa-runs>; CH=<stacks>/glm53-flash-v0.8.0/eval-chain.sh
IMG_ID=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a
IMG_TAG=local/glm53-v0.9.1-b1-d7:local
ENV_SHA=49f1c57cce0e3c4f022a1ff846362cdd21e9e59d633ec021be8cacdb561d2bf7
B1S=<runs>/b1-screen
SA=$B1S/b1screen_analysis.py
SA_SHA=79268c4730d3c0f371cbc9afafa51680ff3d7bfbbfb1befd3827b957f2465b9c
C=glm53-flash-exl3-b12x-vllm; OUT=$B1S/run-s1236; REQLOG="$G/${R}.s1.requests.jsonl"
PH1="$G/${R}__pass1"; PH2="$G/${R}__pass2"; P3="$G/${R}__pass3"
CEIL_GH=${CEIL_GH:-7.5}                                   # occupied GPU-hours, start-up and shutdown included
python3 -c "import sys; sys.exit(0 if 0 < float('$CEIL_GH') <= 7.5 else 1)" 2>/dev/null || { echo "ABORT: CEIL_GH=$CEIL_GH must be in (0, 7.5]"; exit 1; }
CAP_S=$(python3 -c "print(int($CEIL_GH*3600/2))")       # 2 GPUs: 7.5 GPU-h = 13,500 s wall
T0=$(date +%s)
echo "$(date -Is) B1 checkpoint pass (seed 1236) start; recipe $R; ceiling $CEIL_GH GPU-h ($CAP_S s wall)${DRY_RUN:+; DRY_RUN=$DRY_RUN}"
# ---- refusals (nothing is created or started before all of them pass)
[ -e "$OUT" ] && { echo "ABORT: $OUT exists (an earlier attempt; no automatic retry)"; exit 1; }
[ -s "$B1S/costs.jsonl" ] && { echo "ABORT: $B1S/costs.jsonl already records a B1 session (no retry, no further pass)"; exit 1; }
[ -f "<recipes>/$R/env.sh" ] || { echo "ABORT: recipe $R does not exist"; exit 1; }
[ "$(sha256sum "<recipes>/$R/env.sh" | cut -c1-64)" = "$ENV_SHA" ] || { echo "ABORT: recipe env.sh is not the frozen one"; exit 1; }
[ "$(bash -c "source <recipes>/$R/env.sh; echo \$IMAGE")" = "$IMG_ID" ] || { echo "ABORT: recipe IMAGE is not $IMG_ID"; exit 1; }
[ "$(docker image inspect "$IMG_ID" --format '{{.Id}}' 2>/dev/null)" = "$IMG_ID" ] || { echo "ABORT: image $IMG_ID missing (or docker unavailable)"; exit 1; }
TAG_ID=$(docker image inspect "$IMG_TAG" --format '{{.Id}}' 2>/dev/null)
[ "$TAG_ID" = "$IMG_ID" ] || { echo "ABORT: tag $IMG_TAG is '${TAG_ID:-missing}', not $IMG_ID (tag moved; the frozen image is pinned by id)"; exit 1; }
( cd "$B1S" && sha256sum -c --quiet SHA256SUMS ) > /dev/null 2>&1 || { echo "ABORT: $B1S/SHA256SUMS does not verify:"; ( cd "$B1S" && sha256sum -c --quiet SHA256SUMS ) 2>&1 | head -20; exit 1; }
[ "$(sha256sum "$SA" | cut -c1-64)" = "$SA_SHA" ] || { echo "ABORT: b1screen_analysis.py is not the frozen version"; exit 1; }
# pass state: no B1 pass directory of any number or tag, no B1 request log
EXIST=$(ls -d "$G/${R}"__pass* "$G/${R}"+*__pass* 2>/dev/null)
[ -n "$EXIST" ] && { echo "ABORT: B1 pass directories exist (no retry, no further pass):"; echo "$EXIST"; exit 1; }
[ -e "$REQLOG" ] && { echo "ABORT: $REQLOG exists"; exit 1; }
# shipped configuration: no launcher (start.sh) or chain override variable may be set by the caller. Exempt: variables
# that env.sh or local-env.sh export unconditionally (IMAGE, MODEL_PROFILE, GPU_MEMORY_UTILIZATION, CACHE_DIR, HF_HOME).
VARS=$(grep -oE '\$\{[A-Z_][A-Z0-9_]*(:-|-)' <stacks>/glm53-flash-v0.9.0/start.sh | sed -E 's/^\$\{//; s/(:-|-)$//' | sort -u)
[ -n "$VARS" ] || { echo "ABORT: could not read the launcher's variables"; exit 1; }
SETV=""
for v in $VARS GPQA_TAG NO_RESTART GPQA_MAX_TOKENS GPQA_TIMEOUT; do
  case "$v" in IMAGE|MODEL_PROFILE|GPU_MEMORY_UTILIZATION|CACHE_DIR|HF_HOME) continue ;; esac
  [ -n "${!v+x}" ] && SETV="$SETV $v"
done
[ -n "$SETV" ] && { echo "ABORT: override variables set in the environment (not the shipped configuration):$SETV"; exit 1; }
DPS=$(docker ps --format '{{.Names}}'); [ $? -eq 0 ] || { echo "ABORT: docker ps failed"; exit 1; }
echo "$DPS" | grep -qE '^glm53' && { echo "ABORT: a GLM container is running"; exit 1; }
LST=$(ss -ltnH '( sport = :8001 or sport = :8002 )'); [ $? -eq 0 ] || { echo "ABORT: ss failed"; exit 1; }
[ -n "$LST" ] && { echo "ABORT: port 8001 or 8002 is in use:"; echo "$LST"; exit 1; }
# pre-flight GPUs: no compute process except the Claude Desktop GPU process (screen2 rule); no GPU client other than the
# desktop graphics clients (amendments 16d/17c filter); GPU0 <= 1 GiB, GPU1 <= 4 GiB (desktop, amendment 11b), util <= 10%
nvidia-smi -L > /dev/null 2>&1 || { echo "ABORT: nvidia-smi failed"; exit 1; }
for P in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  tr '\0' ' ' < "/proc/$P/cmdline" 2>/dev/null | grep -q -- '--type=gpu-process' || { echo "ABORT: compute process $P on the GPUs"; exit 1; }
done
PROCS=$(nvidia-smi 2>/dev/null | awk '/Processes:/{p=1;next} p && /^\|/ && $2 ~ /^[0-9]+$/ {print}' \
  | grep -vE '[[:space:]]G[[:space:]].*(gnome-shell|Xwayland|xdg-desktop-portal|firefox|snap-store)' \
  | grep -vE '[[:space:]]C\+G[[:space:]].*--disable-logging')
[ -n "$PROCS" ] && { echo "ABORT: GPU clients other than the desktop graphics clients:"; echo "$PROCS"; exit 1; }
NG=0
while read -r IDX MEM UTIL; do
  NG=$((NG+1)); LIM=1024; [ "$IDX" = 1 ] && LIM=4096
  { [ "$MEM" -le "$LIM" ] && [ "$UTIL" -le 10 ]; } || { echo "ABORT: GPU$IDX busy mem=${MEM}MiB util=${UTIL}%"; exit 1; }
done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
[ "$NG" = 2 ] || { echo "ABORT: expected 2 GPUs, saw $NG"; exit 1; }
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "DRY_RUN: all refusal checks passed. A real run would now: create $OUT; start the GPU logger and the watchdog"
  echo "  (${CAP_S} s); start the recording proxy 8002 -> 8001 ($REQLOG); create the empty placeholders __pass1/__pass2;"
  echo "  run GPQA_PASSES=3 GPQA_SEED_PER_PASS=1 GPQA_CONCURRENCY=8 GPQA_BASE=http://127.0.0.1:8002 $CH $R gpqa"
  echo "  (fresh server, warm-up gate, pass 3 only: request seed 1236, lm-eval --seed 1237); stop the server; write one cost"
  echo "  line; remove the placeholders; run the frozen analysis. Nothing was created or started."
  exit 0
fi
# ---- session
mkdir -p "$OUT" || { echo "ABORT: cannot create $OUT"; exit 1; }
( cd "$B1S" && sha256sum gpqa-b1-screen-s1236.sh b1screen_analysis.py analysis_col511.py FROZEN-INTERPRETATION.md SHA256SUMS ) > "$OUT/scripts.sha256"
echo "$IMG_TAG $TAG_ID" > "$OUT/image-id.txt"
( while true; do echo "$(date -Is) $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ')"; sleep 300; done ) > "$OUT/nvidia-smi.log" 2>&1 &
GPULOG=$!
RX=$(printf '%s' "eval-chain.sh $R" | sed 's/[][\.*^$+?(){}|]/\\&/g')   # literal match of the chain's command line
( while sleep 60; do [ $(( $(date +%s)-T0 )) -ge $CAP_S ] && { echo "$(date -Is) CEILING REACHED: stopping"; pkill -f -- "$RX"; pkill -f "lm_eval"; docker stop $C; break; }; done ) >> "$OUT/watchdog.log" 2>&1 &
WD=$!
PROXY=""; CHP=""; STARTED=0; COSTED=0; PH_MADE=0; DONE=0; CRC=""; s0=""
cost_line() {
  [ "$COSTED" = 1 ] && return; COSTED=1
  local w=$(( $(date +%s)-s0 )); local gh; gh=$(python3 -c "print(round(2*$w/3600,3))")
  local wdf=false; grep -q "CEILING REACHED" "$OUT/watchdog.log" 2>/dev/null && wdf=true
  echo "{\"session\": 1, \"checkpoint\": \"B1\", \"recipe\": \"$R\", \"image\": \"$IMG_ID\", \"passes\": \"3\", \"seed\": 1236, \"wall_s\": $w, \"gpus\": 2, \"gpu_hours\": $gh, \"chain_rc\": ${CRC:-null}, \"watchdog_fired\": $wdf}" >> "$B1S/costs.jsonl"
  GH=$gh
}
cleanup() {
  [ "$DONE" = 1 ] && return; DONE=1
  kill $GPULOG $WD 2>/dev/null
  [ -n "$PROXY" ] && kill $PROXY 2>/dev/null
  if [ -n "$CHP" ] && kill -0 $CHP 2>/dev/null; then pkill -f -- "$RX"; pkill -f "lm_eval"; echo "$(date -Is) chain interrupted"; fi
  if [ "$STARTED" = 1 ] && [ "$COSTED" != 1 ]; then
    [ -s "$OUT/server-s1.log" ] || docker logs $C > "$OUT/server-s1.log" 2>&1
    [ -s "$OUT/inspect-s1.json" ] || docker inspect $C > "$OUT/inspect-s1.json" 2>/dev/null
    docker stop $C >/dev/null 2>&1
    cost_line
    echo "$(date -Is) cleanup after an interrupted session: server stopped, cost line written (${GH} GPU-h)"
  fi
  if [ "$PH_MADE" = 1 ]; then rmdir "$PH1" "$PH2" 2>/dev/null; PH_MADE=0; fi
  ls -d "$PH1" "$PH2" 2>/dev/null && echo "WARN: a placeholder directory remains (not empty?)"
}
trap cleanup EXIT
trap 'echo "$(date -Is) signal received"; exit 4' INT TERM
s0=$(date +%s)
python3 <eval-tools>/diag/record_proxy.py 8002 http://127.0.0.1:8001 "$REQLOG" & PROXY=$!
PUP=0
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 1
  kill -0 $PROXY 2>/dev/null || break
  [ -n "$(ss -ltnH '( sport = :8002 )' 2>/dev/null)" ] && { PUP=1; break; }
done
[ "$PUP" = 1 ] || { echo "ABORT: recording proxy did not come up on 8002 (no container was started)"; exit 1; }
mkdir "$PH1" "$PH2" || { echo "ABORT: cannot create the pass-1/2 placeholders (no container was started)"; rmdir "$PH1" "$PH2" 2>/dev/null; exit 1; }
PH_MADE=1
echo "$(date -Is) placeholders (empty, removed at exit): $PH1 $PH2" | tee "$OUT/placeholders.txt"
STARTED=1
GPQA_PASSES=3 GPQA_SEED_PER_PASS=1 GPQA_CONCURRENCY=8 GPQA_BASE=http://127.0.0.1:8002 bash $CH "$R" gpqa &
CHP=$!
wait $CHP
CRC=$?
CHP=""
echo "$(date -Is) chain rc=$CRC"
kill $PROXY 2>/dev/null; PROXY=""
docker logs $C > "$OUT/server-s1.log" 2>&1; echo "docker logs rc=$?"
docker inspect $C > "$OUT/inspect-s1.json" 2>/dev/null; echo "docker inspect rc=$?"
docker stop $C >/dev/null 2>&1; echo "docker stop rc=$?"
cost_line
rmdir "$PH1" "$PH2" 2>/dev/null && PH_MADE=0 && echo "placeholders removed"
python3 $SA --pass-dir "$P3" --reqlog "$REQLOG" --serverlog "$OUT/server-s1.log" --inspect "$OUT/inspect-s1.json" --gpu-hours "$GH" \
  --label "B1 checkpoint pass (seed 1236)" --out "$OUT/b1screen-result.json" > "$OUT/b1screen-result.txt" 2>&1
echo "analysis rc=$? (3 = invalid pass; category PROVISIONAL until --adjudication)"
docker ps --format '{{.Names}}' | grep -E '^glm53' && echo "WARN: GLM container still running" || echo "no GLM container running"
echo "$(date -Is) B1 PASS DONE; total wall $(( $(date +%s)-T0 ))s, $GH GPU-h"
