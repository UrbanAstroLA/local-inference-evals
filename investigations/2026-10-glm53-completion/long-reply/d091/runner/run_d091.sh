#!/usr/bin/env bash
set -u
# Option D runner (PROPOSAL.md section 5): ONE stock tpurtell 0.9.1 GPQA Diamond pass on the standard protocol at request
# seed 70704, full request/response bodies kept locally. PREPARED, NOT RUN; needs the operator's GPU authorization and the
# frozen package (dated amendment with the SHA256SUMS hash).
#   APPROVED_GPU_H  occupied GPU-hours the operator approved (required; 0 < x <= PLAN_MAX_GPU_H); the watchdog stops
#                   everything at APPROVED_GPU_H / 2 GPUs of wall time from script start (the only early stop)
#   DRY_RUN=1       every refusal check, read-only (docker ps / docker image inspect / nvidia-smi / ss / df / git); then
#                   exits before creating anything or starting any process or container
# One session, as the 0.9.1 record's pass 1 (2026-10-08, <eval-tools>/diag/gpqa-k325-v091-pass1.sh): fresh server,
# release warm-up gate (ready file + /v1/models, inside the eval chain), pass 1 (lm-eval --seed 1235, request seed 70704 in
# every request), stop. Launched by digest through the 0.9.0 stack's own launcher with the recipe's env.sh (IMAGE = the
# 0.9.1 digest reference, PORT = 127.0.0.1:8001), through the recording proxy copy (127.0.0.1:8002 -> 8001).
# Derived from rep070/refreeze/runner/run_rep070.sh (refusal set, desktop-graphics filter, escaped watchdog pattern,
# server stopped before the proxy, cleanup and finalize on every exit path); differences in PROPOSAL.md section 5.1.
# Reused unchanged: the rep070 eval-chain copy (the record's chain + GPQA_SEED_LIST + run-tag log name) and the rep070
# recording-proxy copy (main log byte-compatible with the record's proxy + start/interrupted sidecar + 0600 raw bodies),
# both verified against their originals by the frozen diffs and against rep070's copies by hash.
# At the end: complete -> analysis_d091.py (frozen; exit 3 = INVALID, no reading); valid -> scripts/run_pipeline.sh (CPU
# only: discovery pipeline, side-by-side tables, prefix export); otherwise incomplete_d091.py (INCOMPLETE, cause, never
# classified). No automatic retry or replacement run. Not set -e: every status query's exit code is checked explicitly.
PLAN_MAX_GPU_H=10.0          # cost/cost.json ceiling (section 8); no invalid-run allowance
R="glm53-flash-exl3-k3.25@tpurtell+v0.9.1"; G=<gpqa-runs>; C=glm53-flash-exl3-b12x-vllm
H=$(cd "$(dirname "$0")" && pwd); PKG=$(dirname "$H")
case "$PKG" in <runs>/long-reply/d091) ;; *) echo "ABORT: unexpected package path '$PKG'"; exit 1 ;; esac
CH="$H/local-eval-chain-rep070.sh"; PX="$H/record_proxy_rep070.py"; CI="$H/cache_inventory.py"
SD=<stacks>/glm53-flash-v0.9.0
CACHE=<cache>/vllm-v090
IMG_REF="ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07"
IMG_ID="sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07"
SEED=70704; TAG="+d091"; RT="$R$TAG"
OUT="$PKG/run"; RAWD="$PKG/PRIVATE/raw"; LEDGER="$OUT/costs.jsonl"
declare -A PIN=(
  ["<recipes>/$R/env.sh"]=6186bbb66ed982ac942d62ce1dda0a2b66cc63e7af015de717e66043f263414c
  ["$SD/start.sh"]=e1196cbcd6b4744cb41fdebf3d12d8586dbea464e702af5de47617837efe0e5e
  ["$SD/local-env.sh"]=f4658b81eeb4c4d5403a95795f10cf5962eedd8cc5161372356ac725a1404fe6
  ["$SD/start.sh"]=869b0c0af97865c637986f5dfe6be9886e511514cb3b24818208052b505b5fe3
  ["$SD/model-profiles.sh"]=a4ccf85dc1e1dd75acdb8549ecaec82137c9f77b47e5bb6464d82f04ef5537f1
  ["<stacks>/glm53-flash-v0.8.0/eval-chain.sh"]=c4dbb500220772e8bcb1c4a4b6f2e3715176c011d012e8bd95db560ccd8b0cdd
  ["<eval-tools>/diag/record_proxy.py"]=946d880e1df584abbf03b6ab79cc2587dd447c800d1ead6dc3bfaf9b70394d76
  ["<eval-tools>/local-gpqa-score"]=e13116ca27db071254787043eabe13da497d3cb3cd0d069bc45d5eefa924f8e7
  ["<tools>/lm-eval/lm_eval/tasks/gpqa/cot_zeroshot/gpqa_diamond_thinking.yaml"]=149658b4bbfe36e59875a8a462cc3bd351b664d5a46176157af67b78f9a2a0c4
  ["<tools>/lm-eval/lm_eval/tasks/gpqa/cot_zeroshot/utils.py"]=b603828710f1fb3e1b8fe42726b188544ef8dbd12d25f4ca676e42ca4d1a251b
  ["<models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1/snapshots/0490d2f708b12145f6516555ab066aaeb401cd21/chat_template.jinja"]=0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5
  ["$CH"]=81ef338653a2f9cbffff62123113238953f232c268c8e3f02be9e66080d26b9e
  ["$PX"]=ce09016bfba71d241bd88d25a773fb61af149fecc5aa42094cfd30a2a38c6d06
  ["$CI"]=d317acd0ae949337a61f4a970c7091e57e1e932a4e1463c4c98539cf8a4dfec5
)
LMEVAL_HEAD=b954108c9baaaa934b4ad842033b31a97ee30816
[ -n "${APPROVED_GPU_H:-}" ] || { echo "ABORT: APPROVED_GPU_H is not set (the operator's approved occupied GPU-hours)"; exit 1; }
python3 -c "import sys; x=float(sys.argv[1]); sys.exit(0 if 0 < x <= float(sys.argv[2]) else 1)" "$APPROVED_GPU_H" "$PLAN_MAX_GPU_H" 2>/dev/null \
  || { echo "ABORT: APPROVED_GPU_H=$APPROVED_GPU_H must be a number in (0, $PLAN_MAX_GPU_H] (the plan's ceiling)"; exit 1; }
CAP_S=$(python3 -c "import sys; print(int(float(sys.argv[1])*3600/2))" "$APPROVED_GPU_H")   # 2 GPUs
T0=$(date +%s)
echo "$(date -Is) d091 start; seed $SEED; tag $TAG; approved $APPROVED_GPU_H GPU-h (${CAP_S} s wall)${DRY_RUN:+; DRY_RUN=$DRY_RUN}"
# ---------------------------------------------------------------- refusals (nothing is created or started before all pass)
[ -f "$PKG/SHA256SUMS" ] || { echo "ABORT: $PKG/SHA256SUMS missing (package not frozen)"; exit 1; }
( cd "$PKG" && sha256sum -c --quiet SHA256SUMS ) > /dev/null 2>&1 || { echo "ABORT: $PKG/SHA256SUMS does not verify:"; ( cd "$PKG" && sha256sum -c --quiet SHA256SUMS ) 2>&1 | head -20; exit 1; }
for F in "${!PIN[@]}"; do
  [ -f "$F" ] || { echo "ABORT: pinned file missing: $F"; exit 1; }
  [ "$(sha256sum "$F" | cut -c1-64)" = "${PIN[$F]}" ] || { echo "ABORT: pinned file changed: $F"; exit 1; }
done
for PAIR in "<stacks>/glm53-flash-v0.8.0/eval-chain.sh:$CH:$H/local-eval-chain-rep070.diff" \
            "<eval-tools>/diag/record_proxy.py:$PX:$H/record_proxy_rep070.diff"; do
  IFS=: read -r ORIGF COPYF DIFFF <<< "$PAIR"
  DIFFOUT=$(diff "$ORIGF" "$COPYF" 2>&1); DRC=$?
  { [ "$DRC" = 1 ] && [ "$DIFFOUT" = "$(cat "$DIFFF")" ]; } || { echo "ABORT: $(basename "$COPYF") differs from its original by more than the frozen diff"; exit 1; }
done
J=$(python3 -c "
import json, sys
l = (json.load(open(sys.argv[1])).get('launch') or {})
print(l.get('stack_dir'), l.get('container') or 'glm53-flash-exl3-b12x-vllm', l.get('ready_file', '/tmp/glm53-release-ready'))" "<recipes>/$R/recipe.json"); [ $? -eq 0 ] || { echo "ABORT: cannot read recipe.json"; exit 1; }
[ "$J" = "$SD $C /tmp/glm53-release-ready" ] || { echo "ABORT: recipe launch fields changed: $J"; exit 1; }
LH=$(git -C <tools>/lm-eval rev-parse HEAD 2>/dev/null); [ "$LH" = "$LMEVAL_HEAD" ] || { echo "ABORT: lm-eval HEAD is '$LH', not $LMEVAL_HEAD"; exit 1; }
LS=$(git -C <tools>/lm-eval status --porcelain 2>/dev/null); [ $? -eq 0 ] || { echo "ABORT: git status failed on lm-eval"; exit 1; }
[ "$LS" = "?? lm_eval/tasks/gpqa/cot_zeroshot/gpqa_diamond_thinking.yaml" ] || { echo "ABORT: lm-eval working tree changed:"; echo "$LS"; exit 1; }
# shipped configuration: no launcher or chain override variable may be set by the caller (exempt: variables env.sh or
# local-env.sh export unconditionally and the login profile sets: MODEL_PROFILE, CACHE_DIR, HF_HOME). IMAGE, PORT and
# GPU_MEMORY_UTILIZATION from the caller are refused.
VARS=$(grep -oE '\$\{[A-Z_][A-Z0-9_]*(:-|-)' "$SD/start.sh" | sed -E 's/^\$\{//; s/(:-|-)$//' | sort -u)
[ -n "$VARS" ] || { echo "ABORT: could not read the launcher's variables"; exit 1; }
SETV=""
for v in $(printf '%s\n' $VARS IMAGE PORT GPU_MEMORY_UTILIZATION GPQA_TAG NO_RESTART GPQA_MAX_TOKENS GPQA_TIMEOUT GPQA_BASE GPQA_PASSES GPQA_SEED_PER_PASS GPQA_SEED_LIST GPQA_CONCURRENCY | sort -u); do
  case "$v" in MODEL_PROFILE|CACHE_DIR|HF_HOME) continue ;; esac
  [ -n "${!v+x}" ] && SETV="$SETV $v"
done
[ -n "$SETV" ] && { echo "ABORT: override variables set in the environment (not the shipped configuration):$SETV"; exit 1; }
RES=$(env -u IMAGE -u PORT -u GPU_MEMORY_UTILIZATION bash -c "source <recipes>/$R/env.sh && source $SD/local-env.sh && echo \"\$IMAGE \$PORT \$GPU_MEMORY_UTILIZATION \$MODEL_PROFILE \$CACHE_DIR\""); [ $? -eq 0 ] || { echo "ABORT: cannot resolve the launcher's IMAGE/PORT"; exit 1; }
[ "$RES" = "$IMG_REF 127.0.0.1:8001 0.95 k325 $CACHE" ] || { echo "ABORT: launcher resolves to '$RES' (expected the digest reference, 127.0.0.1:8001, 0.95, k325, $CACHE)"; exit 1; }
GOT=$(docker image inspect "$IMG_REF" --format '{{.Id}}' 2>/dev/null)
[ "$GOT" = "$IMG_ID" ] || { echo "ABORT: image $IMG_REF has Id '${GOT:-missing}', not $IMG_ID"; exit 1; }
# weights and draft snapshots: every file resolves to the frozen blob (listings in checks/)
LST=$(cd <models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1/snapshots/0490d2f708b12145f6516555ab066aaeb401cd21 2>/dev/null && find . \( -type l -o -type f \) | LC_ALL=C sort | while read -r f; do echo "$f -> $(basename "$(readlink "$f")")"; done)
[ -n "$LST" ] && [ "$LST" = "$(cat "$PKG/checks/snapshot-0490d2f7.txt")" ] || { echo "ABORT: weights snapshot listing differs from the frozen one"; exit 1; }
DLST=$(cd <models>/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a 2>/dev/null && find . \( -type l -o -type f \) | LC_ALL=C sort | while read -r f; do echo "$f -> $(basename "$(readlink "$f")")"; done)
[ -n "$DLST" ] && [ "$DLST" = "$(cat "$PKG/checks/snapshot-dflash2-bf582e4e.txt")" ] || { echo "ABORT: draft snapshot listing differs from the frozen one"; exit 1; }
# run state: nothing of this run may exist (no retry over an earlier attempt; no record directory is touched)
[ -e "$OUT" ] && { echo "ABORT: $OUT exists (an earlier attempt; no automatic retry)"; exit 1; }
[ -e "$RAWD" ] && { echo "ABORT: $RAWD exists (an earlier attempt)"; exit 1; }
[ -n "$(ls -A "$PKG/out" 2>/dev/null)" ] && { echo "ABORT: $PKG/out is not empty (an earlier pipeline run)"; exit 1; }
[ "$(stat -c %a "$PKG/PRIVATE" 2>/dev/null)" = 700 ] || { echo "ABORT: $PKG/PRIVATE must exist with mode 0700"; exit 1; }
EX=$(ls -d "$G/${RT}"__pass* "$G/${RT}".* <data-runs>/benchmarks/*"-${RT}" 2>/dev/null)
[ -n "$EX" ] && { echo "ABORT: files of run tag $TAG already exist:"; echo "$EX"; exit 1; }
DPS=$(docker ps --format '{{.Names}} {{.Image}}'); [ $? -eq 0 ] || { echo "ABORT: docker ps failed"; exit 1; }
echo "$DPS" | grep -qE '^glm53' && { echo "ABORT: a GLM container is running:"; echo "$DPS" | grep -E '^glm53'; exit 1; }
echo "$DPS" | grep -q "a9e985c09885" && { echo "ABORT: a container of the 0.9.1 image is running"; exit 1; }
LMX='bin/lm_eval --model'      # the lm_eval executable's command line (not any process that merely mentions the name)
LMP=$(pgrep -f -- "$LMX" 2>/dev/null); [ -n "$LMP" ] && { echo "ABORT: an lm_eval process is running (pids $LMP)"; exit 1; }
PL=$(ss -ltnH '( sport = :8001 or sport = :8002 )'); [ $? -eq 0 ] || { echo "ABORT: ss failed"; exit 1; }
[ -n "$PL" ] && { echo "ABORT: port 8001 or 8002 is in use:"; echo "$PL"; exit 1; }
FREE_KB=$(df -Pk /data | awk 'NR==2{print $4}'); [ -n "$FREE_KB" ] && [ "$FREE_KB" -ge 5242880 ] || { echo "ABORT: less than 5 GiB free on /data"; exit 1; }
# pre-flight GPUs: no compute process except the Claude Desktop GPU process; no GPU client other than the desktop graphics
# clients (amendments 16d/17c filter); GPU0 <= 1 GiB, GPU1 <= 4 GiB (desktop, amendment 11b), util <= 10%; exactly 2 GPUs
nvidia-smi -L > /dev/null 2>&1 || { echo "ABORT: nvidia-smi failed"; exit 1; }
CA=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); [ $? -eq 0 ] || { echo "ABORT: nvidia-smi compute-apps query failed"; exit 1; }
for P in $CA; do
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
  echo "DRY_RUN: all refusal checks passed. A real run would now: create $OUT and $RAWD (mode 0700); start the GPU logger"
  echo "  (300 s) and the watchdog (${CAP_S} s from script start); compile-cache inventory of $CACHE; one session: recording"
  echo "  proxy copy 127.0.0.1:8002 -> 8001 ($G/${RT}.s1.requests.jsonl, sidecar .events.jsonl, raw bodies $RAWD/s1.raw.jsonl.gz, 0600),"
  echo "  GPQA_TAG=$TAG GPQA_PASSES=1 GPQA_SEED_LIST='$SEED' GPQA_CONCURRENCY=8 GPQA_BASE=http://127.0.0.1:8002 $CH $R gpqa"
  echo "  (fresh server by digest via $SD, ready-file + API gate; pass 1, request seed $SEED, lm-eval --seed 1235); save server"
  echo "  log and docker inspect; stop the server, then the proxy; cost line; cache diff. Then the frozen analysis and, if valid,"
  echo "  the CPU-only long-reply pipeline. Nothing was created or started."
  exit 0
fi
# ---------------------------------------------------------------- run
mkdir -p "$OUT" || { echo "ABORT: cannot create $OUT"; exit 1; }
( umask 077; mkdir -p "$RAWD" ) && chmod 700 "$RAWD" || { echo "ABORT: cannot create $RAWD"; exit 1; }
python3 -c '
import json, sys
json.dump({"task": "gpqa_diamond_thinking", "task_hash": "87e807b3b27c86618285efdab99466609ad12f022ae453146595fc8306b053b0",
           "passes": [{"pass": 1, "session": 1, "request_seed": int(sys.argv[2]), "lm_eval_process_seed": 1235, "max_gen_toks": 327680,
                       "temperature": 1.0, "top_p": 0.95, "num_concurrent": 8, "timeout": 10800, "max_retries": 3}]},
          open(sys.argv[1], "w"), indent=1)' "$OUT/seeds-planned.json" "$SEED"
python3 -B "$CI" snap "$CACHE" "$OUT/cache-s0.json"
( cd "$PKG" && sha256sum -c SHA256SUMS ) > "$OUT/package-verify.txt" 2>&1
echo "$IMG_REF $GOT" > "$OUT/image-id.txt"
( SL=""; trap '[ -n "$SL" ] && kill $SL 2>/dev/null; exit 0' TERM INT
  while true; do echo "$(date -Is) $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader | tr '\n' ' ') apps: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')"; sleep 300 & SL=$!; wait $SL; done ) > "$OUT/nvidia-smi.log" 2>&1 &
GPULOG=$!
RX=$(printf '%s' "local-eval-chain-rep070.sh $R" | sed 's/[][\.*^$+?(){}|]/\\&/g')   # literal match of the chain's command line
( SL=""; trap '[ -n "$SL" ] && kill $SL 2>/dev/null; exit 0' TERM INT
  while true; do sleep 60 & SL=$!; wait $SL; [ $(( $(date +%s)-T0 )) -ge $CAP_S ] && { echo "$(date -Is) CEILING REACHED: stopping"; pkill -f -- "$RX"; pkill -f -- "$LMX"; docker stop $C; break; }; done ) >> "$OUT/watchdog.log" 2>&1 &
WD=$!
PROXY=""; CHP=""; STARTED=0; COSTED=1; DONE=0; CRC=""; s0=""; SN=0; TOTAL_GH=0; FINAL=0
REQ="$G/${RT}.s1.requests.jsonl"; EVT="$G/${RT}.s1.requests.events.jsonl"; PD="$G/${RT}__pass1"
cost_line() {   # once: 2 GPUs x (session start -> server stopped)
  [ "$COSTED" = 1 ] && return; COSTED=1
  local w=$(( $(date +%s)-s0 )); local gh; gh=$(python3 -c "import sys; print(round(2*int(sys.argv[1])/3600,3))" "$w")
  local wdf=false; grep -q "CEILING REACHED" "$OUT/watchdog.log" 2>/dev/null && wdf=true
  echo "{\"session\": $SN, \"recipe\": \"$R\", \"tag\": \"$TAG\", \"image\": \"$IMG_ID\", \"passes\": \"1\", \"seed\": $SEED, \"start\": $s0, \"stop\": $(date +%s), \"wall_s\": $w, \"gpus\": 2, \"gpu_hours\": $gh, \"chain_rc\": ${CRC:-null}, \"watchdog_fired\": $wdf}" >> "$LEDGER"
  TOTAL_GH=$(python3 -c "import sys; print(round(float(sys.argv[1])+float(sys.argv[2]),3))" "$TOTAL_GH" "$gh")
}
stop_server() {
  [ -s "$OUT/server-s1.log" ] || docker logs $C > "$OUT/server-s1.log" 2>&1
  [ -s "$OUT/inspect-s1.json" ] || docker inspect $C > "$OUT/inspect-s1.json" 2>/dev/null
  docker stop $C >/dev/null 2>&1; local src=$?; echo "$(date -Is) session 1: docker stop rc=$src"
}
stop_proxy() {   # after the server: in-flight requests are written to the sidecar as interrupted (SIGTERM handler)
  [ -n "$PROXY" ] || return
  kill -TERM $PROXY 2>/dev/null
  for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 $PROXY 2>/dev/null || break; sleep 1; done
  kill -9 $PROXY 2>/dev/null; PROXY=""
}
finalize() {   # complete -> frozen analysis (+ CPU pipeline if valid); otherwise the INCOMPLETE report. Runs once.
  [ "$FINAL" = 1 ] && return; FINAL=1
  local CL; CL=$(ls <data-runs>/benchmarks/*"-${RT}/chain.log" 2>/dev/null)
  python3 -B "$PKG/incomplete_d091.py" probe --pass-dir "$PD" --ledger "$LEDGER" --watchdog-log "$OUT/watchdog.log" > "$OUT/probe.json" 2>&1
  local PRC=$?
  if [ "$PRC" = 0 ]; then
    python3 -B "$PKG/analysis_d091.py" --pass-dir "$PD" --reqlog "$REQ" --serverlog "$OUT/server-s1.log" --inspect "$OUT/inspect-s1.json" \
      --chainlog $CL --gpu-hours "$TOTAL_GH" --rows-out "$OUT/rows-d091.jsonl" --out "$OUT/result-d091.json" > "$OUT/result-d091.md" 2> "$OUT/result-d091.err"
    ARC=$?; echo "$(date -Is) COMPLETE run: frozen analysis rc=$ARC (0 valid; 3 INVALID; any other = analysis error; only 0 is read; stated figure provisional until the hand read)"
    if [ "$ARC" = 0 ]; then
      bash "$PKG/scripts/run_pipeline.sh" > "$OUT/pipeline.out" 2>&1
      echo "$(date -Is) long-reply pipeline (CPU only) rc=$? (see $PKG/out/pipeline.log)"
    else
      echo "$(date -Is) INVALID run: the long-reply pipeline is not run; stop and amend"
    fi
  else
    python3 -B "$PKG/incomplete_d091.py" report --pass-dir "$PD" --reqlog "$REQ" --events "$EVT" --ledger "$LEDGER" \
      --watchdog-log "$OUT/watchdog.log" --artifacts "$G/${RT}*" "$OUT/server-s1.log" "$OUT/inspect-s1.json" "$RAWD" \
      --out "$OUT/incomplete-d091.json" > "$OUT/incomplete-d091.md" 2> "$OUT/incomplete-d091.err"
    ARC=$?; echo "$(date -Is) INCOMPLETE run ($(tr -d '\n' < "$OUT/probe.json" | cut -c1-200)): report rc=$ARC; never classified; stop and amend"
  fi
}
cleanup() {
  [ "$DONE" = 1 ] && return; DONE=1
  local KIDS; KIDS=$(pgrep -P "$GPULOG" 2>/dev/null; pgrep -P "$WD" 2>/dev/null)   # their sleep children
  kill $GPULOG $WD 2>/dev/null; [ -n "$KIDS" ] && kill $KIDS 2>/dev/null
  if [ -n "$CHP" ] && kill -0 $CHP 2>/dev/null; then pkill -f -- "$RX"; pkill -f -- "$LMX"; echo "$(date -Is) chain interrupted"; fi
  if [ "$STARTED" = 1 ] && [ "$COSTED" != 1 ]; then
    stop_server; stop_proxy; cost_line
    echo "$(date -Is) cleanup after an interrupted session: server stopped, proxy stopped after it, cost line written"
  fi
  stop_proxy
  [ -f "$RAWD/s1.raw.jsonl.gz" ] && chmod 600 "$RAWD/s1.raw.jsonl.gz"
  [ "$SN" -ge 1 ] && finalize
  docker ps --format '{{.Names}}' | grep -qE '^glm53' && echo "WARN: GLM container still running" || echo "no GLM container running"
}
trap cleanup EXIT
trap 'echo "$(date -Is) signal received"; exit 4' INT TERM
# ---------------------------------------------------------------- the session
SN=1; s0=$(date +%s); COSTED=0; CRC=""
python3 -B "$PX" 8002 http://127.0.0.1:8001 "$REQ" "$RAWD/s1.raw.jsonl.gz" & PROXY=$!
UP=0
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 1; kill -0 $PROXY 2>/dev/null || break
  [ -n "$(ss -ltnH '( sport = :8002 )' 2>/dev/null)" ] && { UP=1; break; }
done
[ "$UP" = 1 ] || { echo "ABORT: recording proxy did not come up on 8002 (no container started)"; COSTED=1; exit 1; }
STARTED=1
echo "$(date -Is) session 1: pass 1, seed $SEED, requests -> $REQ"
GPQA_TAG=$TAG GPQA_PASSES=1 GPQA_SEED_LIST="$SEED" GPQA_CONCURRENCY=8 GPQA_BASE=http://127.0.0.1:8002 bash "$CH" "$R" gpqa &
CHP=$!; wait $CHP; CRC=$?; CHP=""
echo "$(date -Is) session 1: chain rc=$CRC"
stop_server; stop_proxy
python3 - "$OUT/inspect-s1.json" "$IMG_ID" <<'EOF'
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
cost_line; STARTED=0
python3 -B "$CI" snap "$CACHE" "$OUT/cache-s1.json"
python3 -B "$CI" diff "$OUT/cache-s0.json" "$OUT/cache-s1.json" "$OUT/cache-diff-s1.json"
echo "  warm-up lines: $(grep -c 'GLM release startup ' "$OUT/server-s1.log"); preemption lines: $(grep -ciE 'preempt' "$OUT/server-s1.log")"
P1S=$(ls "$PD"/*/samples_*.jsonl 2>/dev/null | wc -l)
[ "$CRC" = 0 ] && [ "$P1S" = 1 ] || echo "$(date -Is) the session did not produce pass 1 (chain rc $CRC, samples files $P1S): no retry"
finalize
echo "$(date -Is) D091 DONE; total wall $(( $(date +%s)-T0 ))s, $TOTAL_GH GPU-h (ledger $LEDGER)"
