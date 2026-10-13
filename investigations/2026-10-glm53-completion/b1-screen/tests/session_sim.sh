#!/usr/bin/env bash
# Session-path simulation of gpqa-b1-screen-s1236.sh with NO GPU and NO container (CPU only).
#   session_sim.sh WORK_DIR
# A copy of the frozen runner with B1S, G and CH pointing into WORK; CH is a fake eval-chain.sh (same file name,
# so the watchdog's escaped pattern is exercised) that only creates the pass-3 directory and sleeps. docker and nvidia-smi
# are shims: read-only queries answer as on a healthy machine; logs/inspect/stop are recorded and answered with stubs; any
# other docker call is refused. The real recording proxy binds 127.0.0.1:8002 (no upstream is contacted).
# Scenarios: (a) normal completion; (b) SIGTERM to the runner during the chain; (c) watchdog at a 0.04 GPU-h ceiling.
# Checks after each: placeholders removed, exactly one cost line, server stopped (shim), chain gone, proxy gone.
# Refuses to run if any lm_eval process exists (the watchdog and cleanup use pkill -f lm_eval). Not set -e.
WORK=${1:?usage: session_sim.sh WORK_DIR}
HERE=$(cd "$(dirname "$0")" && pwd); B1S=$(dirname "$HERE"); RUNNER=$B1S/gpqa-b1-screen-s1236.sh
R="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234+d7"
IMG=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a
pgrep -f lm_eval >/dev/null && { echo "REFUSED: an lm_eval process exists"; exit 2; }
[ -n "$(ss -ltnH '( sport = :8001 or sport = :8002 )')" ] && { echo "REFUSED: port 8001/8002 in use"; exit 2; }
rm -rf "$WORK"; mkdir -p "$WORK/shim" "$WORK/stack" || exit 2
SHIMLOG=$WORK/shim-calls.log; : > "$SHIMLOG"
cat > "$WORK/shim/docker" <<EOF
#!/usr/bin/env bash
echo "\$(date +%s) docker \$*" >> "$SHIMLOG"
case "\$1" in
  image) echo $IMG; exit 0 ;;
  ps) echo open-webui; exit 0 ;;
  logs) echo "stub server log"; exit 0 ;;
  inspect) echo "[]"; exit 0 ;;
  stop) echo "\$2"; exit 0 ;;
esac
echo "REFUSED docker \$*" >> "$SHIMLOG"; exit 99
EOF
cat > "$WORK/shim/nvidia-smi" <<EOF
#!/usr/bin/env bash
case "\$*" in
  -L) echo "GPU 0: shim"; echo "GPU 1: shim" ;;
  *query-compute-apps*) : ;;
  *query-gpu=index,memory.used,utilization.gpu*) echo "0, 20, 0"; echo "1, 2246, 2" ;;
  "") echo "| Processes: |" ;;
  *) : ;;
esac
exit 0
EOF
cat > "$WORK/stack/eval-chain.sh" <<EOF
#!/usr/bin/env bash
# FAKE chain: records its environment, creates the pass-3 directory only if passes 1-2 are skipped, then sleeps.
echo "fake chain \$* GPQA_PASSES=\$GPQA_PASSES SEED_PER_PASS=\$GPQA_SEED_PER_PASS CONC=\$GPQA_CONCURRENCY BASE=\$GPQA_BASE" >> "$WORK/chain.log"
for p in 1 2 3; do d="$WORK/gpqa/\${1}__pass\$p"; [ -d "\$d" ] && { echo "pass \$p exists, skip" >> "$WORK/chain.log"; continue; }; mkdir "\$d"; echo "pass \$p run, seed \$((1233+p))" >> "$WORK/chain.log"; done
sleep "\${FAKE_CHAIN_SECS:-2}"
EOF
chmod +x "$WORK/shim/"* "$WORK/stack/eval-chain.sh"
PASS=0; FAIL=0
ok() { if eval "$2"; then PASS=$((PASS+1)); echo "  ok    $1"; else FAIL=$((FAIL+1)); echo "  FAIL  $1"; fi; }
setup() {
  CP=$WORK/b1s; GW=$WORK/gpqa; rm -rf "$CP" "$GW" "$WORK/chain.log"; mkdir -p "$CP" "$GW"
  cp "$B1S/b1screen_analysis.py" "$B1S/analysis_col511.py" "$B1S/FROZEN-INTERPRETATION.md" "$CP/"
  sed -e "s#^B1S=.*#B1S=$CP#" -e "s#R=\"$R\"; G=<gpqa-runs>; CH=[^ ]*#R=\"$R\"; G=$GW; CH=$WORK/stack/eval-chain.sh#" "$RUNNER" > "$CP/run.sh"
  cp "$CP/run.sh" "$CP/gpqa-b1-screen-s1236.sh"
  ( cd "$B1S" && grep -v -e ' gpqa-b1-screen-s1236.sh$' -e 'eval-chain.sh$' SHA256SUMS ) > "$CP/SHA256SUMS"
  ( cd "$CP" && sha256sum gpqa-b1-screen-s1236.sh >> SHA256SUMS )
}
post() {
  ok "placeholders __pass1/__pass2 removed" "[ ! -e '$GW/${R}__pass1' ] && [ ! -e '$GW/${R}__pass2' ]"
  ok "pass-3 directory created by the (fake) chain; passes 1-2 skipped" "[ -d '$GW/${R}__pass3' ] && grep -q 'pass 1 exists, skip' '$WORK/chain.log' && grep -q 'pass 2 exists, skip' '$WORK/chain.log' && grep -q 'pass 3 run, seed 1236' '$WORK/chain.log'"
  ok "chain called with GPQA_PASSES=3 SEED_PER_PASS=1 CONC=8 BASE=proxy" "grep -q 'GPQA_PASSES=3 SEED_PER_PASS=1 CONC=8 BASE=http://127.0.0.1:8002' '$WORK/chain.log'"
  ok "exactly one cost line" "[ \$(wc -l < '$CP/costs.jsonl') = 1 ]"
  ok "server stopped (docker stop glm53-flash-exl3-b12x-vllm)" "grep -q 'docker stop glm53-flash-exl3-b12x-vllm' '$SHIMLOG'"
  ok "no refused docker call" "! grep -q REFUSED '$SHIMLOG'"
  ok "fake chain gone" "! pgrep -f '$WORK/stack/eval-chain.sh' >/dev/null"
  ok "proxy gone (8002 free)" "[ -z \"\$(ss -ltnH '( sport = :8002 )')\" ]"
}
echo "== (a) normal completion"
setup; FAKE_CHAIN_SECS=2 PATH="$WORK/shim:$PATH" bash "$CP/run.sh" > "$WORK/a.out" 2>&1; echo "  runner rc=$?"
post; ok "cost line: watchdog_fired false, chain_rc 0" "grep -q '\"chain_rc\": 0, \"watchdog_fired\": false' '$CP/costs.jsonl'"
ok "analysis ran and reported invalid (empty pass dir) with rc 3" "grep -q 'analysis rc=3' '$WORK/a.out'"
sleep 2
echo "== (b) SIGTERM to the runner during the chain"
setup; : > "$SHIMLOG"
FAKE_CHAIN_SECS=601 PATH="$WORK/shim:$PATH" bash "$CP/run.sh" > "$WORK/b.out" 2>&1 & RP=$!
for i in $(seq 1 30); do sleep 1; grep -q 'pass 3 run' "$WORK/chain.log" 2>/dev/null && break; done
sleep 2; kill -TERM $RP; wait $RP; echo "  runner rc=$?"
post; ok "runner reported the signal and the interrupted-session cleanup" "grep -q 'signal received' '$WORK/b.out' && grep -q 'cleanup after an interrupted session' '$WORK/b.out'"
ok "cost line has chain_rc null" "grep -q '\"chain_rc\": null' '$CP/costs.jsonl'"
sleep 2
echo "== (c) watchdog at CEIL_GH=0.04 (72 s wall; checked every 60 s)"
setup; : > "$SHIMLOG"
CEIL_GH=0.04 FAKE_CHAIN_SECS=601 PATH="$WORK/shim:$PATH" timeout 400 bash "$CP/run.sh" > "$WORK/c.out" 2>&1; echo "  runner rc=$?"
post; ok "watchdog fired and the cost line records it" "grep -q 'CEILING REACHED' '$CP/run-s1236/watchdog.log' && grep -q '\"watchdog_fired\": true' '$CP/costs.jsonl'"
ok "watchdog's escaped pattern killed the chain (wall < 200 s)" "python3 -c \"import json,sys; sys.exit(0 if json.loads(open('$CP/costs.jsonl').read())['wall_s'] < 200 else 1)\""
pkill -f -x "sleep 601" 2>/dev/null   # orphaned sleeps of the killed fake chains
echo "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
