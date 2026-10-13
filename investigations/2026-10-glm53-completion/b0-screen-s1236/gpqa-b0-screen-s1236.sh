#!/usr/bin/env bash
# B0 checkpoint pass, seed 1236 (review step after the UNFAVOURABLE B1 seed-1236 reading). PREPARED, NOT RUN; needs the
# operator's GPU authorization.
# Copy of ../b1-screen/gpqa-b1-screen-s1236.sh (sha256 a4bb8e6b...), changed only for:
#   - checkpoint B0: recipe glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234 (env.sh 3125683b..., recipe.json
#     318996c4...), image id 08d1cb90... pinned, refusal if the tag local/glm53-v0.9.1-bundle-d1234:local moved;
#   - seed 1236 = protocol pass 3, reached NATURALLY: B0's real __pass1 (seed 1234) and __pass2 (seed 1235) exist, so
#     eval-chain.sh skips them ("pass N exists, skip") and GPQA_PASSES=3 runs pass 3 alone, first pass of a FRESH
#     server: request seed 1233+3 = 1236, lm-eval --seed 1234+3 = 1237 (as B1's pass and the comparators' seed-1236
#     passes). The placeholder logic of the B1 runner is REMOVED; instead the runner refuses unless __pass1 and __pass2
#     are present with their frozen samples files (sha256 8c413a97..., d3eb4fa7...), and re-checks them afterwards;
#   - pass-state refusals: any B0 pass-3 (or later) directory; the session-3 request log; any existing B0 request log
#     (<B0>.*requests.jsonl) holding a seed-1236 request. (B1's "tagged pass directory" glob <R>+*__pass* is dropped: for
#     the B0 name it matches B1's own +d7__pass3. GPQA_TAG stays refused, so the chain cannot write a tagged directory.)
#   - output in b0-screen-s1236/run-s1236, request log <B0>.s3.requests.jsonl (B0 session 3; s1 = seed 1234, s2 = seed
#     1235), server-s3.log / inspect-s3.json, cost line in b0-screen-s1236/costs.jsonl;
#   - frozen analysis b0s1236_analysis.py (B0 vs stock 0.9.1 seed 1236 for continuity; B0 vs B1 at seed 1236; B0
#     three-seed exploratory view).
# Unchanged from the B1 runner: every other refusal (SHA256SUMS incl. chain/proxy/launcher; no earlier session; no
# launcher/chain override variable; ports 8001/8002 free; no GLM container; GPU idle check with the 16d/17c desktop
# filter), the recording proxy up before the chain, the escaped watchdog pattern, chain in the background and waited for,
# cleanup on every exit path (EXIT/INT/TERM: chain, proxy, loggers, watchdog, server stop, one cost line).
# DRY_RUN=1 runs every refusal check, then exits before creating anything or starting any process or container.
# Protocol gpqa-diamond/v1 unchanged: 198 questions, concurrency 8, cap 327,680, temperature 1.0, top_p 0.95, thinking on,
# normal stopping, recording proxy, shipped graph configuration (no EXTRA_SERVE_ARGS), release warm-up marker gate of the
# chain (ready file + API). Traces stay local. Only early stop: the 7.5 occupied-GPU-hour watchdog (2 GPUs x wall from
# script start; 13,500 s). No automatic retry and no further pass. Afterwards b0s1236_analysis.py runs (PROVISIONAL until
# the stated-answer adjudication file). Not set -e: every status query's exit code is checked explicitly.
R="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234"; G=<gpqa-runs>; CH=<stacks>/glm53-flash-v0.8.0/eval-chain.sh
IMG_ID=sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548
IMG_TAG=local/glm53-v0.9.1-bundle-d1234:local
ENV_SHA=3125683b01ba74a2d41745d8000fe40918c031511d8d43e098c08f8e845b15c5
B0S=<runs>/b0-screen-s1236
SA=$B0S/b0s1236_analysis.py
SA_SHA=f707d2003ddc7553235db8ef31071ce26e2adaec91294e43098cd945eb3a7477
C=glm53-flash-exl3-b12x-vllm; OUT=$B0S/run-s1236; REQLOG="$G/${R}.s3.requests.jsonl"
P1="$G/${R}__pass1"; P2="$G/${R}__pass2"; P3="$G/${R}__pass3"
P1_SAMPLES_SHA=8c413a97e412ea156ffc7093b3e85a5d04ff06e3fbba66b6f3077f18114caad8   # B0 pass 1 (seed 1234)
P2_SAMPLES_SHA=d3eb4fa745c70ee8a91133e3d12e238a83e8aa9487f7989e3b42d6f157fa9c92   # B0 pass 2 (seed 1235)
CEIL_GH=${CEIL_GH:-7.5}                                   # occupied GPU-hours, start-up and shutdown included
python3 -c "import sys; sys.exit(0 if 0 < float('$CEIL_GH') <= 7.5 else 1)" 2>/dev/null || { echo "ABORT: CEIL_GH=$CEIL_GH must be in (0, 7.5]"; exit 1; }
CAP_S=$(python3 -c "print(int($CEIL_GH*3600/2))")       # 2 GPUs: 7.5 GPU-h = 13,500 s wall
T0=$(date +%s)
echo "$(date -Is) B0 checkpoint pass (seed 1236) start; recipe $R; ceiling $CEIL_GH GPU-h ($CAP_S s wall)${DRY_RUN:+; DRY_RUN=$DRY_RUN}"
# ---- refusals (nothing is created or started before all of them pass)
[ -e "$OUT" ] && { echo "ABORT: $OUT exists (an earlier attempt; no automatic retry)"; exit 1; }
[ -s "$B0S/costs.jsonl" ] && { echo "ABORT: $B0S/costs.jsonl already records a B0 seed-1236 session (no retry, no further pass)"; exit 1; }
[ -f "<recipes>/$R/env.sh" ] || { echo "ABORT: recipe $R does not exist"; exit 1; }
[ "$(sha256sum "<recipes>/$R/env.sh" | cut -c1-64)" = "$ENV_SHA" ] || { echo "ABORT: recipe env.sh is not the frozen one"; exit 1; }
[ "$(bash -c "source <recipes>/$R/env.sh; echo \$IMAGE")" = "$IMG_ID" ] || { echo "ABORT: recipe IMAGE is not $IMG_ID"; exit 1; }
[ "$(docker image inspect "$IMG_ID" --format '{{.Id}}' 2>/dev/null)" = "$IMG_ID" ] || { echo "ABORT: image $IMG_ID missing (or docker unavailable)"; exit 1; }
TAG_ID=$(docker image inspect "$IMG_TAG" --format '{{.Id}}' 2>/dev/null)
[ "$TAG_ID" = "$IMG_ID" ] || { echo "ABORT: tag $IMG_TAG is '${TAG_ID:-missing}', not $IMG_ID (tag moved; the frozen image is pinned by id)"; exit 1; }
( cd "$B0S" && sha256sum -c --quiet SHA256SUMS ) > /dev/null 2>&1 || { echo "ABORT: $B0S/SHA256SUMS does not verify:"; ( cd "$B0S" && sha256sum -c --quiet SHA256SUMS ) 2>&1 | head -20; exit 1; }
[ "$(sha256sum "$SA" | cut -c1-64)" = "$SA_SHA" ] || { echo "ABORT: b0s1236_analysis.py is not the frozen version"; exit 1; }
# pass state: B0 passes 1 and 2 present and unchanged (the chain skips them, so pass 3 = seed 1236 is reached
# naturally); no B0 pass 3 or later; no session-3 request log; no B0 request log with a seed-1236 request
for PP in "$P1:$P1_SAMPLES_SHA" "$P2:$P2_SAMPLES_SHA"; do
  PD=${PP%:*}; PS=${PP##*:}
  F=$(ls "$PD"/*/samples_*.jsonl 2>/dev/null)
  { [ -n "$F" ] && [ "$(echo "$F" | wc -l)" = 1 ] && [ "$(sha256sum "$F" | cut -c1-64)" = "$PS" ]; } || { echo "ABORT: $PD missing or not the frozen B0 record (pass 3 would not be reached naturally)"; exit 1; }
done
EXIST=$(ls -d "$G/${R}"__pass[3-9]* "$G/${R}"__pass[1-9][0-9]* 2>/dev/null)
[ -n "$EXIST" ] && { echo "ABORT: a B0 pass-3 (or later) directory exists (no retry, no further pass):"; echo "$EXIST"; exit 1; }
[ -e "$REQLOG" ] && { echo "ABORT: $REQLOG exists"; exit 1; }
S1236=$(python3 -c "
import glob, json, sys
hits = []
for f in sorted(glob.glob(sys.argv[1] + '.*requests.jsonl')):
    for line in open(f):
        if line.strip() and json.loads(line).get('seed') == 1236:
            hits.append(f); break
print(' '.join(hits))" "$G/$R"); [ $? -eq 0 ] || { echo "ABORT: could not read the B0 request logs"; exit 1; }
[ -n "$S1236" ] && { echo "ABORT: a B0 request log already holds seed-1236 requests: $S1236"; exit 1; }
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
  echo "  (${CAP_S} s); start the recording proxy 8002 -> 8001 ($REQLOG);"
  echo "  run GPQA_PASSES=3 GPQA_SEED_PER_PASS=1 GPQA_CONCURRENCY=8 GPQA_BASE=http://127.0.0.1:8002 $CH $R gpqa"
  echo "  (fresh server, warm-up gate; the existing __pass1/__pass2 are skipped, pass 3 only: request seed 1236, lm-eval"
  echo "  --seed 1237); stop the server; write one cost line; re-check passes 1-2; run the frozen analysis."
  echo "  Nothing was created or started."
  exit 0
fi
# ---- session
mkdir -p "$OUT" || { echo "ABORT: cannot create $OUT"; exit 1; }
( cd "$B0S" && sha256sum gpqa-b0-screen-s1236.sh b0s1236_analysis.py analysis_col511.py FROZEN-INTERPRETATION.md SHA256SUMS ) > "$OUT/scripts.sha256"
echo "$IMG_TAG $TAG_ID" > "$OUT/image-id.txt"
( while true; do echo "$(date -Is) $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ')"; sleep 300; done ) > "$OUT/nvidia-smi.log" 2>&1 &
GPULOG=$!
RX=$(printf '%s' "eval-chain.sh $R" | sed 's/[][\.*^$+?(){}|]/\\&/g')   # literal match of the chain's command line
( while sleep 60; do [ $(( $(date +%s)-T0 )) -ge $CAP_S ] && { echo "$(date -Is) CEILING REACHED: stopping"; pkill -f -- "$RX"; pkill -f "lm_eval"; docker stop $C; break; }; done ) >> "$OUT/watchdog.log" 2>&1 &
WD=$!
PROXY=""; CHP=""; STARTED=0; COSTED=0; DONE=0; CRC=""; s0=""
cost_line() {
  [ "$COSTED" = 1 ] && return; COSTED=1
  local w=$(( $(date +%s)-s0 )); local gh; gh=$(python3 -c "print(round(2*$w/3600,3))")
  local wdf=false; grep -q "CEILING REACHED" "$OUT/watchdog.log" 2>/dev/null && wdf=true
  echo "{\"session\": 3, \"checkpoint\": \"B0\", \"recipe\": \"$R\", \"image\": \"$IMG_ID\", \"passes\": \"3\", \"seed\": 1236, \"wall_s\": $w, \"gpus\": 2, \"gpu_hours\": $gh, \"chain_rc\": ${CRC:-null}, \"watchdog_fired\": $wdf}" >> "$B0S/costs.jsonl"
  GH=$gh
}
cleanup() {
  [ "$DONE" = 1 ] && return; DONE=1
  kill $GPULOG $WD 2>/dev/null
  [ -n "$PROXY" ] && kill $PROXY 2>/dev/null
  if [ -n "$CHP" ] && kill -0 $CHP 2>/dev/null; then pkill -f -- "$RX"; pkill -f "lm_eval"; echo "$(date -Is) chain interrupted"; fi
  if [ "$STARTED" = 1 ] && [ "$COSTED" != 1 ]; then
    [ -s "$OUT/server-s3.log" ] || docker logs $C > "$OUT/server-s3.log" 2>&1
    [ -s "$OUT/inspect-s3.json" ] || docker inspect $C > "$OUT/inspect-s3.json" 2>/dev/null
    docker stop $C >/dev/null 2>&1
    cost_line
    echo "$(date -Is) cleanup after an interrupted session: server stopped, cost line written (${GH} GPU-h)"
  fi
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
STARTED=1
GPQA_PASSES=3 GPQA_SEED_PER_PASS=1 GPQA_CONCURRENCY=8 GPQA_BASE=http://127.0.0.1:8002 bash $CH "$R" gpqa &
CHP=$!
wait $CHP
CRC=$?
CHP=""
echo "$(date -Is) chain rc=$CRC"
kill $PROXY 2>/dev/null; PROXY=""
docker logs $C > "$OUT/server-s3.log" 2>&1; echo "docker logs rc=$?"
docker inspect $C > "$OUT/inspect-s3.json" 2>/dev/null; echo "docker inspect rc=$?"
docker stop $C >/dev/null 2>&1; echo "docker stop rc=$?"
cost_line
for PP in "$P1:$P1_SAMPLES_SHA" "$P2:$P2_SAMPLES_SHA"; do
  PD=${PP%:*}; PS=${PP##*:}; F=$(ls "$PD"/*/samples_*.jsonl 2>/dev/null)
  [ -n "$F" ] && [ "$(sha256sum "$F" | cut -c1-64)" = "$PS" ] && echo "$(basename "$PD") unchanged" || echo "WARN: $(basename "$PD") samples changed"
done
python3 $SA --pass-dir "$P3" --reqlog "$REQLOG" --serverlog "$OUT/server-s3.log" --inspect "$OUT/inspect-s3.json" --gpu-hours "$GH" \
  --label "B0 checkpoint pass (seed 1236)" --out "$OUT/b0s1236-result.json" > "$OUT/b0s1236-result.txt" 2>&1
echo "analysis rc=$? (3 = invalid pass; category and reading PROVISIONAL until --adjudication)"
docker ps --format '{{.Names}}' | grep -E '^glm53' && echo "WARN: GLM container still running" || echo "no GLM container running"
echo "$(date -Is) B0 PASS DONE; total wall $(( $(date +%s)-T0 ))s, $GH GPU-h"
