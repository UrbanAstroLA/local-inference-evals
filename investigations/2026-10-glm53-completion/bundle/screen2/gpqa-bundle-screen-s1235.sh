#!/usr/bin/env bash
# Second pass of the UNCHANGED frozen bundle, seed 1235 (PLAN.md amendment 15). PREPARED, NOT RUN; needs the operator's
# budget decision. Same logic as ../screen/gpqa-bundle-screen.sh (sha256 e8caf211...), changed only to run protocol pass 2:
#   GPQA_PASSES=2 with the existing __pass1 directory present -> eval-chain.sh skips pass 1 ("pass 1 exists, skip")
#   and runs pass 2 as the first and only pass of a FRESH server session: request seed 1233+2 = 1235 (GPQA_SEED_PER_PASS=1),
#   lm-eval --seed 1234+2 = 1236 -- the same conditions as the comparators' seed-1235 passes (stock record +ps__pass2,
#   2026-10-09; column-511 arm __pass2, 2026-10-11): each was the first pass after a fresh server start (see README.md).
# Protocol gpqa-diamond/v1 unchanged: 198 questions, concurrency 8, cap 327,680, temperature 1.0, top_p 0.95, thinking on,
# normal stopping, recording proxy (NEW request log <recipe>.s2.requests.jsonl), release warm-up marker gate. Traces local.
# Only early stop: the 7.5 occupied-GPU-hour watchdog. No automatic retry and no third pass: refuses if __pass2 (or any
# __pass3+) exists, if the s2 request log exists, or if pass 1 differs from the frozen screen-1 record; GPQA_PASSES=2 makes
# a third pass impossible. Afterwards screen2_analysis.py runs (PROVISIONAL until the stated-answer adjudication file).
# Not set -e.
R="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234"; G=<gpqa-runs>; CH=<stacks>/glm53-flash-v0.8.0/eval-chain.sh
IMG_ID=sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548
ENV_SHA=3125683b01ba74a2d41745d8000fe40918c031511d8d43e098c08f8e845b15c5
P1_SAMPLES_SHA=8c413a97e412ea156ffc7093b3e85a5d04ff06e3fbba66b6f3077f18114caad8   # screen-1 pass-1 samples (seed 1234)
S2=<runs>/bundle/screen2
SA=$S2/screen2_analysis.py
SA_SHA=${SCREEN2_ANALYSIS_SHA:-cc3d300032cd831deea233c7e2251477b807170cbee4f80a4bd611efafef11c7}
C=glm53-flash-exl3-b12x-vllm; OUT=$S2/run-s1235; REQLOG="$G/${R}.s2.requests.jsonl"
CEIL_GH=${CEIL_GH:-7.5}                                   # occupied GPU-hours, start-up and shutdown included
CAP_S=$(python3 -c "print(int($CEIL_GH*3600/2))")       # 2 GPUs: 7.5 GPU-h = 13,500 s wall
T0=$(date +%s)
echo "$(date -Is) second bundle pass (seed 1235) start; recipe $R; ceiling $CEIL_GH GPU-h ($CAP_S s wall)"
[ -e "$OUT" ] && { echo "ABORT: $OUT exists (an earlier attempt; no automatic retry)"; exit 1; }
[ -f "<recipes>/$R/env.sh" ] || { echo "ABORT: recipe $R does not exist"; exit 1; }
[ "$(sha256sum "<recipes>/$R/env.sh" | cut -c1-64)" = "$ENV_SHA" ] || { echo "ABORT: recipe env.sh is not the frozen one"; exit 1; }
[ "$(bash -c "source <recipes>/$R/env.sh; echo \$IMAGE")" = "$IMG_ID" ] || { echo "ABORT: recipe IMAGE is not $IMG_ID"; exit 1; }
[ "$(docker image inspect "$IMG_ID" --format '{{.Id}}' 2>/dev/null)" = "$IMG_ID" ] || { echo "ABORT: image $IMG_ID missing"; exit 1; }
[ "$(sha256sum $SA | cut -c1-64)" = "$SA_SHA" ] || { echo "ABORT: screen2_analysis.py is not the frozen version"; exit 1; }
# pass state: pass 1 present and unchanged; no pass 2 (or later); no s2 request log
P1=$(ls $G/${R}__pass1/*/samples_*.jsonl 2>/dev/null)
[ "$(echo "$P1" | wc -l)" = 1 ] && [ "$(sha256sum "$P1" | cut -c1-64)" = "$P1_SAMPLES_SHA" ] || { echo "ABORT: __pass1 missing or not the frozen screen-1 record"; exit 1; }
ls -d $G/${R}__pass[2-9]* 2>/dev/null && { echo "ABORT: a pass-2 (or later) directory exists; no retry, no third pass"; exit 1; }
[ -e "$REQLOG" ] && { echo "ABORT: $REQLOG exists"; exit 1; }
docker ps --format '{{.Names}}' | grep -qE '^glm53' && { echo "ABORT: a GLM container is running"; exit 1; }
# pre-flight: no non-desktop compute process; GPU0 <= 1 GiB, GPU1 <= 4 GiB (desktop accepted, amendment 11b), util <= 10%
for P in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  tr '\0' ' ' < "/proc/$P/cmdline" 2>/dev/null | grep -q -- '--type=gpu-process' || { echo "ABORT: compute process $P on the GPUs"; exit 1; }
done
while read -r IDX MEM UTIL; do
  LIM=1024; [ "$IDX" = 1 ] && LIM=4096
  { [ "$MEM" -le "$LIM" ] && [ "$UTIL" -le 10 ]; } || { echo "ABORT: GPU$IDX busy mem=${MEM}MiB util=${UTIL}%"; exit 1; }
done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
mkdir -p $OUT
( while true; do echo "$(date -Is) $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ')"; sleep 300; done ) > $OUT/nvidia-smi.log 2>&1 &
GPULOG=$!
( while sleep 60; do [ $(( $(date +%s)-T0 )) -ge $CAP_S ] && { echo "$(date -Is) CEILING REACHED: stopping"; pkill -f "eval-chain.sh $R"; pkill -f "lm_eval"; docker stop $C; break; }; done ) >> $OUT/watchdog.log 2>&1 &
WD=$!
PROXY=""
trap 'kill $GPULOG $WD 2>/dev/null; [ -n "$PROXY" ] && kill $PROXY 2>/dev/null; docker stop $C >/dev/null 2>&1' EXIT
s0=$(date +%s)
python3 <eval-tools>/diag/record_proxy.py 8002 http://127.0.0.1:8001 "$REQLOG" & PROXY=$!
GPQA_PASSES=2 GPQA_SEED_PER_PASS=1 GPQA_CONCURRENCY=8 GPQA_BASE=http://127.0.0.1:8002 bash $CH "$R" gpqa
echo "$(date -Is) chain rc=$?"
kill $PROXY 2>/dev/null; PROXY=""
docker logs $C > $OUT/server-s2.log 2>&1; docker inspect $C > $OUT/inspect-s2.json 2>/dev/null; docker stop $C >/dev/null 2>&1
w=$(( $(date +%s)-s0 ))
GH=$(python3 -c "print(round(2*$w/3600,3))")
echo "{\"session\": 2, \"passes\": \"2\", \"seed\": 1235, \"bundle\": \"d1234\", \"wall_s\": $w, \"gpus\": 2, \"gpu_hours\": $GH}" >> $S2/costs.jsonl
[ "$(sha256sum "$P1" | cut -c1-64)" = "$P1_SAMPLES_SHA" ] && echo "pass 1 unchanged" || echo "WARN: pass 1 samples changed"
python3 $SA --pass-dir "$G/${R}__pass2" --reqlog "$REQLOG" --serverlog "$OUT/server-s2.log" --gpu-hours "$GH" \
  --label "bundle d1234 second pass (seed 1235)" --out "$OUT/screen2-result.json" > "$OUT/screen2-result.txt" 2>&1
echo "analysis rc=$? (3 = invalid pass; category PROVISIONAL until --adjudication)"
docker ps --format '{{.Names}}' | grep -E '^glm53' && echo "WARN: GLM container still running" || echo "no GLM container running"
echo "$(date -Is) SECOND PASS DONE; total wall $(( $(date +%s)-T0 ))s, $GH GPU-h"
