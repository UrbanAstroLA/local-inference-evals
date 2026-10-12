#!/usr/bin/env bash
# One-pass GPQA screen of the correctness bundle (PLAN.md amendments 14, 14a-14c). READY; NOT RUN by its author -- the
# coordinator launches it after reviewing the amendment 14c pre-launch items. Recipe fork
# glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234 (env.sh: only IMAGE differs from +v0.9.1; IMAGE=sha256:08d1cb90...).
# Protocol gpqa-diamond/v1 unchanged: all 198 questions, seed 1234 (GPQA_SEED_PER_PASS=1, pass 1), concurrency 8,
# 327,680-token cap, temperature 1.0, top_p 0.95, thinking on, normal stopping, through the recording proxy; fresh server;
# release warm-up marker gate (eval-chain.sh). Traces stay local (<gpqa-runs>). No stopping or extension on
# interim scores. The only early stop is the 7.5 occupied-GPU-hour watchdog; a watchdog-truncated or invalid run is not a
# completed one-pass result and is NOT retried automatically (the script refuses to start if a pass directory exists).
# Afterwards screen_analysis.py runs without hand reads, so its category is PROVISIONAL until the stated-answer
# adjudication file is supplied (amendment 14c). Not set -e.
BUNDLE=${BUNDLE:-d1234}
case "$BUNDLE" in
  d1234) IMG_ID=sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548 ;;
  d124)  IMG_ID=sha256:0e970204890e4f353ff375b53f2307c8738e5fa15d8195029f950d990c9ebbfe ;;
  *) echo "BUNDLE must be d1234 or d124"; exit 2 ;;
esac
R="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-$BUNDLE"; G=<gpqa-runs>; CH=<stacks>/glm53-flash-v0.8.0/eval-chain.sh
C=glm53-flash-exl3-b12x-vllm; OUT=<runs>/bundle/screen/run-$BUNDLE; mkdir -p $OUT
CEIL_GH=${CEIL_GH:-7.5}                                   # occupied GPU-hours, start-up and shutdown included
CAP_S=$(python3 -c "print(int($CEIL_GH*3600/2))")       # 2 GPUs: 7.5 GPU-h = 13,500 s wall
T0=$(date +%s)
echo "$(date -Is) screen start; recipe $R; ceiling $CEIL_GH GPU-h ($CAP_S s wall)"
[ -f "<recipes>/$R/env.sh" ] || { echo "ABORT: recipe $R does not exist (create the fork first)"; exit 1; }
[ "$(bash -c "source <recipes>/$R/env.sh; echo \$IMAGE")" = "$IMG_ID" ] || { echo "ABORT: recipe IMAGE is not $IMG_ID"; exit 1; }
[ "$(docker image inspect "$IMG_ID" --format '{{.Id}}' 2>/dev/null)" = "$IMG_ID" ] || { echo "ABORT: image $IMG_ID missing"; exit 1; }
docker ps --format '{{.Names}}' | grep -qE '^glm53' && { echo "ABORT: a GLM container is running"; exit 1; }
# pre-flight: no non-desktop compute process; GPU0 <= 1 GiB, GPU1 <= 4 GiB (desktop accepted, amendment 11b), util <= 10%
for P in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  tr '\0' ' ' < "/proc/$P/cmdline" 2>/dev/null | grep -q -- '--type=gpu-process' || { echo "ABORT: compute process $P on the GPUs"; exit 1; }
done
while read -r IDX MEM UTIL; do
  LIM=1024; [ "$IDX" = 1 ] && LIM=4096
  { [ "$MEM" -le "$LIM" ] && [ "$UTIL" -le 10 ]; } || { echo "ABORT: GPU$IDX busy mem=${MEM}MiB util=${UTIL}%"; exit 1; }
done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
SA=<runs>/bundle/screen/screen_analysis.py
[ "$(sha256sum $SA | cut -c1-64)" = "${SCREEN_ANALYSIS_SHA:-9c799f260ddcc54739fea3864884f55f33cf0e2cef1fcab32758d6a4e59106cb}" ] || { echo "ABORT: screen_analysis.py is not the frozen version"; exit 1; }
ls -d $G/${R}__pass* 2>/dev/null && { echo "ABORT: screen pass directory already exists (an earlier attempt: see invalid-run rules)"; exit 1; }
( while true; do echo "$(date -Is) $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ')"; sleep 300; done ) > $OUT/nvidia-smi.log 2>&1 &
GPULOG=$!
( while sleep 60; do [ $(( $(date +%s)-T0 )) -ge $CAP_S ] && { echo "$(date -Is) CEILING REACHED: stopping"; pkill -f "eval-chain.sh $R"; pkill -f "lm_eval"; docker stop $C; break; }; done ) >> $OUT/watchdog.log 2>&1 &
WD=$!
PROXY=""
trap 'kill $GPULOG $WD 2>/dev/null; [ -n "$PROXY" ] && kill $PROXY 2>/dev/null; docker stop $C >/dev/null 2>&1' EXIT
s0=$(date +%s)
python3 <eval-tools>/diag/record_proxy.py 8002 http://127.0.0.1:8001 "$G/${R}.s1.requests.jsonl" & PROXY=$!
GPQA_PASSES=1 GPQA_SEED_PER_PASS=1 GPQA_CONCURRENCY=8 GPQA_BASE=http://127.0.0.1:8002 bash $CH "$R" gpqa
echo "$(date -Is) chain rc=$?"
kill $PROXY 2>/dev/null; PROXY=""
docker logs $C > $OUT/server-s1.log 2>&1; docker inspect $C > $OUT/inspect-s1.json 2>/dev/null; docker stop $C >/dev/null 2>&1
w=$(( $(date +%s)-s0 ))
echo "{\"session\": 1, \"passes\": \"1\", \"bundle\": \"$BUNDLE\", \"wall_s\": $w, \"gpus\": 2, \"gpu_hours\": $(python3 -c "print(round(2*$w/3600,3))")}" >> $OUT/../costs.jsonl
python3 <runs>/bundle/screen/screen_analysis.py \
  --pass-dir "$G/${R}__pass1" --reqlog "$G/${R}.s1.requests.jsonl" --serverlog "$OUT/server-s1.log" \
  --gpu-hours "$(python3 -c "print(round(2*$w/3600,3))")" --label "bundle $BUNDLE screen pass (seed 1234)" --out "$OUT/screen-result.json" \
  > "$OUT/screen-result.txt" 2>&1; echo "analysis rc=$? (3 = invalid pass)"
docker ps --format '{{.Names}}' | grep -E '^glm53' && echo "WARN: GLM container still running" || echo "no GLM container running"
echo "$(date -Is) SCREEN DONE; total wall $(( $(date +%s)-T0 ))s"
