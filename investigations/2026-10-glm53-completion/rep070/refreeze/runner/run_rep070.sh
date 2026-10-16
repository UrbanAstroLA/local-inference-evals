#!/usr/bin/env bash
# Replication of stock tpurtell 0.7.0's K3.25 GPQA record (PROPOSAL-v2.md). PREPARED, NOT RUN; needs the operator's GPU
# authorization and a frozen package (dated amendment with SHA256SUMS).
#   VARIANT=R  recorded per-pass seeds 1234/1235/1236 (run tag +rep070-R)
#   VARIANT=F  fresh per-pass seeds 70701/70702/70703 (run tag +rep070-F)
#   APPROVED_GPU_H  occupied GPU-hours the operator approved (required; 0 < x <= PLAN_MAX_GPU_H); the watchdog stops
#                   everything at APPROVED_GPU_H / 2 GPUs of wall time from script start (the only early stop)
#   DRY_RUN=1  every refusal check, read-only (docker ps / docker image inspect / nvidia-smi / ss); then exits before
#              creating anything or starting any process or container
# Two sessions, as the record: session 1 = fresh server, warm-up gate, pass 1, stop; session 2 = fresh server, warm-up gate,
# passes 2-3 back to back, stop. Launched by image digest through the stack's own launcher (local-env.sh default IMAGE
# = the digest reference, PORT = 127.0.0.1:8001), through the passive recording proxy (127.0.0.1:8002 -> 8001).
# Modelled on <eval-tools>/diag/gpqa-col511-arm.sh (session split, proxy, GPU log, ceiling watchdog, server log and
# docker inspect per session, cost line per session) and the B0/B1 screen runners (refusal set, desktop-graphics filter,
# escaped watchdog pattern, cleanup on every exit path). Protocol gpqa-diamond/v1: 198 questions, concurrency 8, cap
# 327,680, temperature 1.0, top_p 0.95, lm-eval --seed 1234+pass. No automatic retry; no stopping or extension on interim
# scores. Not set -e: every status query's exit code is checked explicitly.
# Change set 22n: recording proxy = runner/record_proxy_rep070.py (frozen diff against the original): main log unchanged,
# plus a sidecar of start/interrupted events and a LOCAL raw file of request and response bodies (OUT/raw/, mode 0600,
# never published); the server is stopped before the proxy, so in-flight requests are recorded as interrupted; compile-
# cache inventory before session 1 and after each session; planned seeds written per pass; at the end the run is probed:
# complete -> the frozen analysis, otherwise incomplete_rep070.py (INCOMPLETE, cause, completed passes reported, no
# O1/O2/O3). Every partial artifact is kept on every exit path.
PLAN_MAX_GPU_H=36.5          # budget B-lean (change set 22n): valid-run ceiling = maximum; NO invalid-run allowance (an
                             # invalid or incomplete session means stop and amend; no automatic restart or replacement run)
R="glm53-flash-exl3-k3.25@tpurtell"; G=<gpqa-runs>; C=glm53-flash-exl3-b12x-vllm
H=$(cd "$(dirname "$0")" && pwd); PKG=$(dirname "$H"); BASE=$(dirname "$PKG")       # refreeze/ and rep070/
CH="$H/local-eval-chain-rep070.sh"; PX="$H/record_proxy_rep070.py"
SD=<stacks>/glm53-flash-v0.7.0
IMG_REF="ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629"
IMG_ID="sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629"
declare -A PIN=(
  ["<recipes>/$R/env.sh"]=ed50bc7d83751258b4707e3b9d3c5dba43b41b4d6587d5d81277d02758cea2b6
  ["$SD/start.sh"]=c852511c1b13264aa750f78ed78ffccc15a3a67569179e2d84344a678378ca57
  ["$SD/local-env.sh"]=f1f3ced5dc85968cb31aabd102575d121f673e1a0f619919c5855d8fed8fb4d0
  ["$SD/start.sh"]=869b0c0af97865c637986f5dfe6be9886e511514cb3b24818208052b505b5fe3
  ["$SD/model-profiles.sh"]=13c07d5b61953174340ab8c000d68bd84368ad653bbfe66b60bcd80dc4cd17b3
  ["<stacks>/glm53-flash-v0.8.0/eval-chain.sh"]=c4dbb500220772e8bcb1c4a4b6f2e3715176c011d012e8bd95db560ccd8b0cdd
  ["<eval-tools>/diag/record_proxy.py"]=946d880e1df584abbf03b6ab79cc2587dd447c800d1ead6dc3bfaf9b70394d76
  ["<eval-tools>/local-gpqa-score"]=e13116ca27db071254787043eabe13da497d3cb3cd0d069bc45d5eefa924f8e7
  ["<tools>/lm-eval/lm_eval/tasks/gpqa/cot_zeroshot/gpqa_diamond_thinking.yaml"]=149658b4bbfe36e59875a8a462cc3bd351b664d5a46176157af67b78f9a2a0c4
  ["<tools>/lm-eval/lm_eval/tasks/gpqa/cot_zeroshot/utils.py"]=b603828710f1fb3e1b8fe42726b188544ef8dbd12d25f4ca676e42ca4d1a251b
  ["<models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1/snapshots/701cd7456c13d87bf0147ad946f828a999afb59c/chat_template.jinja"]=34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891
)
LMEVAL_HEAD=b954108c9baaaa934b4ad842033b31a97ee30816
case "${VARIANT:-}" in
  R) SEEDS="1234 1235 1236" ;;
  F) SEEDS="70701 70702 70703" ;;
  *) echo "ABORT: VARIANT must be R or F (got '${VARIANT:-}')"; exit 1 ;;
esac
TAG="+rep070-$VARIANT"; RT="$R$TAG"; OUT="$BASE/run-$VARIANT"; LEDGER="$OUT/costs.jsonl"
[ -n "${APPROVED_GPU_H:-}" ] || { echo "ABORT: APPROVED_GPU_H is not set (the operator's approved occupied GPU-hours)"; exit 1; }
python3 -c "import sys; x=float(sys.argv[1]); sys.exit(0 if 0 < x <= float(sys.argv[2]) else 1)" "$APPROVED_GPU_H" "$PLAN_MAX_GPU_H" 2>/dev/null \
  || { echo "ABORT: APPROVED_GPU_H=$APPROVED_GPU_H must be a number in (0, $PLAN_MAX_GPU_H] (the plan's maximum)"; exit 1; }
CAP_S=$(python3 -c "import sys; print(int(float(sys.argv[1])*3600/2))" "$APPROVED_GPU_H")   # 2 GPUs
S2_MIN_S=27955                     # record's session-2 wall (7.765 h): session 2 is not started with less wall left
T0=$(date +%s)
echo "$(date -Is) rep070 variant $VARIANT start; seeds $SEEDS; tag $TAG; approved $APPROVED_GPU_H GPU-h (${CAP_S} s wall)${DRY_RUN:+; DRY_RUN=$DRY_RUN}"
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
# local-env.sh export unconditionally: MODEL_PROFILE, CACHE_DIR, HF_HOME). IMAGE and PORT are conditional defaults in
# local-env.sh, so a caller's value would override the digest or the loopback publish: refused.
VARS=$(grep -oE '\$\{[A-Z_][A-Z0-9_]*(:-|-)' "$SD/start.sh" | sed -E 's/^\$\{//; s/(:-|-)$//' | sort -u)
[ -n "$VARS" ] || { echo "ABORT: could not read the launcher's variables"; exit 1; }
SETV=""
for v in $(printf '%s\n' $VARS IMAGE PORT GPQA_TAG NO_RESTART GPQA_MAX_TOKENS GPQA_TIMEOUT GPQA_BASE GPQA_PASSES GPQA_SEED_PER_PASS GPQA_SEED_LIST GPQA_CONCURRENCY | sort -u); do
  case "$v" in MODEL_PROFILE|CACHE_DIR|HF_HOME) continue ;; esac
  [ -n "${!v+x}" ] && SETV="$SETV $v"
done
[ -n "$SETV" ] && { echo "ABORT: override variables set in the environment (not the shipped configuration):$SETV"; exit 1; }
RES=$(env -u IMAGE -u PORT bash -c "source <recipes>/$R/env.sh && source $SD/local-env.sh && echo \"\$IMAGE \$PORT\""); [ $? -eq 0 ] || { echo "ABORT: cannot resolve the launcher's IMAGE/PORT"; exit 1; }
[ "$RES" = "$IMG_REF 127.0.0.1:8001" ] || { echo "ABORT: launcher resolves to '$RES' (expected the digest reference and 127.0.0.1:8001)"; exit 1; }
GOT=$(docker image inspect "$IMG_REF" --format '{{.Id}}' 2>/dev/null)
[ "$GOT" = "$IMG_ID" ] || { echo "ABORT: image $IMG_REF has Id '${GOT:-missing}', not $IMG_ID"; exit 1; }
# weights snapshot: every file resolves to the frozen blob (listing in checks/snapshot-701cd745.txt)
LST=$(cd <models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1/snapshots/701cd7456c13d87bf0147ad946f828a999afb59c 2>/dev/null && find . \( -type l -o -type f \) | LC_ALL=C sort | while read -r f; do echo "$f -> $(basename "$(readlink "$f")")"; done)
[ "$LST" = "$(cat "$PKG/checks/snapshot-701cd745.txt")" ] || { echo "ABORT: weights snapshot listing differs from the frozen one"; exit 1; }
DLST=$(cd <models>/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a 2>/dev/null && find . \( -type l -o -type f \) | LC_ALL=C sort | while read -r f; do echo "$f -> $(basename "$(readlink "$f")")"; done)
[ "$DLST" = "$(cat "$PKG/checks/snapshot-dflash2-bf582e4e.txt")" ] || { echo "ABORT: draft snapshot listing differs from the frozen one"; exit 1; }
# pass state: nothing of this run tag may exist (no retry over an earlier attempt; no record directory is touched)
[ -e "$OUT" ] && { echo "ABORT: $OUT exists (an earlier attempt; no automatic retry)"; exit 1; }
EX=$(ls -d "$G/${RT}"__pass* "$G/${RT}".* "<data-runs>/benchmarks/"*"-${RT}" 2>/dev/null)
[ -n "$EX" ] && { echo "ABORT: files of run tag $TAG already exist:"; echo "$EX"; exit 1; }
DPS=$(docker ps --format '{{.Names}} {{.Image}}'); [ $? -eq 0 ] || { echo "ABORT: docker ps failed"; exit 1; }
echo "$DPS" | grep -qE '^glm53' && { echo "ABORT: a GLM container is running:"; echo "$DPS" | grep -E '^glm53'; exit 1; }
echo "$DPS" | grep -q "48e254d94f58" && { echo "ABORT: a container of the 0.7.0 image is running"; exit 1; }
LMX='bin/lm_eval --model'      # the lm_eval executable's command line (not any process that merely mentions the name)
LMP=$(pgrep -f -- "$LMX" 2>/dev/null); [ -n "$LMP" ] && { echo "ABORT: an lm_eval process is running (pids $LMP)"; exit 1; }
PL=$(ss -ltnH '( sport = :8001 or sport = :8002 )'); [ $? -eq 0 ] || { echo "ABORT: ss failed"; exit 1; }
[ -n "$PL" ] && { echo "ABORT: port 8001 or 8002 is in use:"; echo "$PL"; exit 1; }
FREE_KB=$(df -Pk /data | awk 'NR==2{print $4}'); [ -n "$FREE_KB" ] && [ "$FREE_KB" -ge 5242880 ] || { echo "ABORT: less than 5 GiB free on /data"; exit 1; }
# pre-flight GPUs: no compute process except the Claude Desktop GPU process (screen2 rule); no GPU client other than the
# desktop graphics clients (amendments 16d/17c filter); GPU0 <= 1 GiB, GPU1 <= 4 GiB (desktop, amendment 11b), util <= 10%
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
  echo "DRY_RUN: all refusal checks passed. A real run would now: create $OUT; start the GPU logger (300 s) and the"
  echo "  watchdog (${CAP_S} s from script start); compile-cache inventory; session 1: recording proxy copy 127.0.0.1:8002 -> 8001"
  echo "  ($G/${RT}.s1.requests.jsonl, sidecar .events.jsonl, local raw bodies in $OUT/raw, mode 0600),"
  echo "  GPQA_TAG=$TAG GPQA_PASSES=1 GPQA_SEED_LIST='$SEEDS' GPQA_CONCURRENCY=8 GPQA_BASE=http://127.0.0.1:8002 $CH $R gpqa"
  echo "  (fresh server by digest, ready-file + API gate; pass 1, request seed ${SEEDS%% *}); save server log and docker inspect;"
  echo "  stop; cost line. Session 2 (if session 1 produced its pass and >= ${S2_MIN_S} s of approved wall remain): same with"
  echo "  GPQA_PASSES=3 (pass 1 skipped; passes 2-3) and $G/${RT}.s2.requests.jsonl. Then the analysis (variant $VARIANT)."
  echo "  Nothing was created or started."
  exit 0
fi
# ---------------------------------------------------------------- run
mkdir -p "$OUT" || { echo "ABORT: cannot create $OUT"; exit 1; }
( umask 077; mkdir -p "$OUT/raw" ) && chmod 700 "$OUT/raw" || { echo "ABORT: cannot create $OUT/raw"; exit 1; }
python3 -c '
import json, sys
out, v, s = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]]
json.dump({"variant": v, "task": "gpqa_diamond_thinking", "task_hash": "87e807b3b27c86618285efdab99466609ad12f022ae453146595fc8306b053b0",
           "passes": [{"pass": p, "session": 1 if p == 1 else 2, "request_seed": s[p - 1], "lm_eval_process_seed": 1234 + p,
                       "max_gen_toks": 327680, "temperature": 1.0, "top_p": 0.95, "num_concurrent": 8} for p in (1, 2, 3)]},
          open(out, "w"), indent=1)' "$OUT/seeds-planned.json" "$VARIANT" $SEEDS
python3 "$H/cache_inventory.py" snap <cache>/vllm "$OUT/cache-s0.json"
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
cost_line() {   # once per session: 2 GPUs x (session start -> server stopped)
  [ "$COSTED" = 1 ] && return; COSTED=1
  local w=$(( $(date +%s)-s0 )); local gh; gh=$(python3 -c "import sys; print(round(2*int(sys.argv[1])/3600,3))" "$w")
  local wdf=false; grep -q "CEILING REACHED" "$OUT/watchdog.log" 2>/dev/null && wdf=true
  echo "{\"session\": $SN, \"variant\": \"$VARIANT\", \"recipe\": \"$R\", \"tag\": \"$TAG\", \"image\": \"$IMG_ID\", \"passes\": \"$PASSES\", \"seeds\": \"$SEEDS\", \"start\": $s0, \"stop\": $(date +%s), \"wall_s\": $w, \"gpus\": 2, \"gpu_hours\": $gh, \"chain_rc\": ${CRC:-null}, \"watchdog_fired\": $wdf}" >> "$LEDGER"
  TOTAL_GH=$(python3 -c "import sys; print(round(float(sys.argv[1])+float(sys.argv[2]),3))" "$TOTAL_GH" "$gh")
}
stop_server() {
  [ -s "$OUT/server-s$SN.log" ] || docker logs $C > "$OUT/server-s$SN.log" 2>&1
  [ -s "$OUT/inspect-s$SN.json" ] || docker inspect $C > "$OUT/inspect-s$SN.json" 2>/dev/null
  docker stop $C >/dev/null 2>&1; local src=$?; echo "$(date -Is) session $SN: docker stop rc=$src"
}
stop_proxy() {   # after the server: in-flight requests are written to the sidecar as interrupted (SIGTERM handler)
  [ -n "$PROXY" ] || return
  kill -TERM $PROXY 2>/dev/null
  for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 $PROXY 2>/dev/null || break; sleep 1; done
  kill -9 $PROXY 2>/dev/null; PROXY=""
}
finalize() {   # complete -> frozen analysis; otherwise the INCOMPLETE report. Runs once, on every exit path after a start.
  [ "$FINAL" = 1 ] && return; FINAL=1
  local PD=("$G/${RT}__pass1" "$G/${RT}__pass2" "$G/${RT}__pass3")
  local CL; CL=$(ls "<data-runs>/benchmarks/"*"-${RT}/chain.log" 2>/dev/null)
  python3 "$PKG/incomplete_rep070.py" probe --variant "$VARIANT" --rep-pass "${PD[@]}" --ledger "$LEDGER" --watchdog-log "$OUT/watchdog.log" > "$OUT/probe.json" 2>&1
  local PRC=$?
  if [ "$PRC" = 0 ]; then
    python3 "$PKG/analysis_rep070.py" --variant "$VARIANT" --rep-pass "${PD[@]}" \
      --rep-reqlog "$G/${RT}.s1.requests.jsonl" "$G/${RT}.s2.requests.jsonl" --rep-serverlog "$OUT/server-s1.log" "$OUT/server-s2.log" \
      --rep-chainlog $CL --rep-gpu-hours "$TOTAL_GH" --out "$OUT/result-rep070-$VARIANT.json" > "$OUT/result-rep070-$VARIANT.md" 2> "$OUT/result-rep070-$VARIANT.err"
    ARC=$?; echo "$(date -Is) COMPLETE run: frozen analysis rc=$ARC (3 = invalid run; stated figure provisional until the hand read)"
  else
    python3 "$PKG/incomplete_rep070.py" report --variant "$VARIANT" --rep-pass "${PD[@]}" \
      --rep-reqlog "$G/${RT}.s1.requests.jsonl" "$G/${RT}.s2.requests.jsonl" \
      --rep-events "$G/${RT}.s1.requests.events.jsonl" "$G/${RT}.s2.requests.events.jsonl" --ledger "$LEDGER" \
      --watchdog-log "$OUT/watchdog.log" --artifacts "$G/${RT}*" "$OUT/server-s*.log" "$OUT/inspect-s*.json" \
      --out "$OUT/incomplete-rep070-$VARIANT.json" > "$OUT/incomplete-rep070-$VARIANT.md" 2> "$OUT/incomplete-rep070-$VARIANT.err"
    ARC=$?; echo "$(date -Is) INCOMPLETE run ($(tr -d '\n' < "$OUT/probe.json" | cut -c1-200)): report rc=$ARC; no outcome category; stop and amend"
  fi
}
cleanup() {
  [ "$DONE" = 1 ] && return; DONE=1
  local KIDS; KIDS=$(pgrep -P "$GPULOG" 2>/dev/null; pgrep -P "$WD" 2>/dev/null)   # their sleep children
  kill $GPULOG $WD 2>/dev/null; [ -n "$KIDS" ] && kill $KIDS 2>/dev/null
  if [ -n "$CHP" ] && kill -0 $CHP 2>/dev/null; then pkill -f -- "$RX"; pkill -f -- "$LMX"; echo "$(date -Is) chain interrupted"; fi
  if [ "$STARTED" = 1 ] && [ "$COSTED" != 1 ]; then
    stop_server; stop_proxy; cost_line
    echo "$(date -Is) cleanup after an interrupted session $SN: server stopped, proxy stopped after it, cost line written"
  fi
  stop_proxy
  [ "$SN" -ge 1 ] && finalize
  docker ps --format '{{.Names}}' | grep -qE '^glm53' && echo "WARN: GLM container still running" || echo "no GLM container running"
}
trap cleanup EXIT
trap 'echo "$(date -Is) signal received"; exit 4' INT TERM
session() { # n passes
  SN=$1; PASSES=$2; local req="$G/${RT}.s$1.requests.jsonl"
  s0=$(date +%s); COSTED=0; CRC=""
  python3 "$PX" 8002 http://127.0.0.1:8001 "$req" "$OUT/raw/s$1.raw.jsonl.gz" & PROXY=$!
  local up=0
  for i in 1 2 3 4 5 6 7 8 9 10; do
    sleep 1; kill -0 $PROXY 2>/dev/null || break
    [ -n "$(ss -ltnH '( sport = :8002 )' 2>/dev/null)" ] && { up=1; break; }
  done
  [ "$up" = 1 ] || { echo "ABORT: recording proxy did not come up on 8002 (session $1; no container started)"; COSTED=1; exit 1; }
  STARTED=1
  echo "$(date -Is) session $1: passes up to $2, seeds $SEEDS, requests -> $req"
  GPQA_TAG=$TAG GPQA_PASSES=$2 GPQA_SEED_LIST="$SEEDS" GPQA_CONCURRENCY=8 GPQA_BASE=http://127.0.0.1:8002 bash "$CH" "$R" gpqa &
  CHP=$!; wait $CHP; CRC=$?; CHP=""
  echo "$(date -Is) session $1: chain rc=$CRC"
  stop_server; stop_proxy
  python3 - "$OUT/inspect-s$1.json" "$IMG_ID" <<'EOF'
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
  python3 "$H/cache_inventory.py" snap <cache>/vllm "$OUT/cache-s$1.json"
  python3 "$H/cache_inventory.py" diff "$OUT/cache-s$(( $1 - 1 )).json" "$OUT/cache-s$1.json" "$OUT/cache-diff-s$1.json"
  echo "  warm-up lines: $(grep -c 'GLM release startup ' "$OUT/server-s$1.log"); preemption lines: $(grep -ciE 'preempt' "$OUT/server-s$1.log")"
}
session 1 1
P1S=$(ls "$G/${RT}__pass1"/*/samples_*.jsonl 2>/dev/null | wc -l)
if [ "$CRC" != 0 ] || [ "$P1S" != 1 ]; then
  echo "$(date -Is) session 1 did not produce pass 1 (chain rc $CRC, samples files $P1S): stopping (no automatic retry)"
elif [ $(( CAP_S - ($(date +%s)-T0) )) -lt $S2_MIN_S ]; then
  echo "$(date -Is) less than ${S2_MIN_S} s of approved wall remains: session 2 not started (budget stop)"
else
  session 2 3
fi
finalize
echo "$(date -Is) REP070 DONE; total wall $(( $(date +%s)-T0 ))s, $TOTAL_GH GPU-h (ledger $LEDGER)"
