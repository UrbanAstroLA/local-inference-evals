#!/usr/bin/env bash
# Session-path simulation of runner/run_rep070.sh with NO GPU and NO container (CPU only), after ../../b0-screen-s1236/tests/
# session_sim.sh. A copy of the package goes to WORK/pkg; in the runner copy only G points into WORK; the chain at the
# runner's CH path is a FAKE (same file name, so the watchdog's escaped pattern is exercised) that applies the real chain's
# skip rule, creates the missing pass directories with an empty placeholder samples file (no GPQA content), records its
# environment and sleeps; the package's chain diff and SHA256SUMS are regenerated for the copy, so every refusal still
# runs. docker and nvidia-smi are shims: read-only queries answer as on a healthy machine, logs/inspect/stop are recorded
# and answered with stubs, any other docker call is refused (exit 99). The real recording proxy binds 127.0.0.1:8002 (no
# request is sent). Scenarios: (a) normal completion of both sessions; (b) SIGTERM to the runner during session 2;
# (c) watchdog at 0.04 GPU-h; (d) approved budget too small for session 2; (e) session 1 fails (chain rc 1, no pass);
# (f) change set 22n: the watchdog fires while a request is in flight through the proxy copy (the fake chain starts a
# hanging fake upstream on 8001 and sends one request via 8002; the docker shim's 'stop' kills the fake upstream, as a
# stopped server would): the sidecar shows start + interrupted, the raw file holds the interrupted record (mode 0600),
# the main log has no row, and the INCOMPLETE report counts it as interrupted (never as empty).
# Refuses to run if an lm_eval process exists or port 8001/8002 is in use. Not set -e.
WORK=${1:?usage: session_sim.sh WORK_DIR}
HERE=$(cd "$(dirname "$0")" && pwd); PKG0=$(dirname "$HERE")
R="glm53-flash-exl3-k3.25@tpurtell"; IMG=sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629
pgrep -f -- 'bin/lm_eval --model' >/dev/null && { echo "REFUSED: an lm_eval process exists"; exit 2; }
[ -n "$(ss -ltnH '( sport = :8001 or sport = :8002 )')" ] && { echo "REFUSED: port 8001/8002 in use"; exit 2; }
rm -rf "$WORK"; mkdir -p "$WORK/shim" || exit 2
SHIMLOG=$WORK/shim-calls.log; : > "$SHIMLOG"
cat > "$WORK/shim/docker" <<EOF
#!/usr/bin/env bash
echo "\$(date +%s) docker \$*" >> "$SHIMLOG"
case "\$1" in
  image) echo $IMG; exit 0 ;;
  ps) echo "open-webui 8afd2d774834"; exit 0 ;;
  logs) echo "stub server log"; exit 0 ;;
  inspect) echo '[{"Image": "$IMG", "HostConfig": {"PortBindings": {"8001/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8001"}]}}}]'; exit 0 ;;
  stop) pkill -f fake_upstream_rep070 2>/dev/null; echo "\$2"; exit 0 ;;
esac
echo "REFUSED docker \$*" >> "$SHIMLOG"; exit 99
EOF
cat > "$WORK/shim/nvidia-smi" <<EOF
#!/usr/bin/env bash
case "\$*" in
  -L) echo "GPU 0: shim"; echo "GPU 1: shim" ;;
  *query-compute-apps*) : ;;
  *query-gpu=index,memory.used,utilization.gpu,temperature*) echo "0, 20 MiB, 0 %, 30, 20 W"; echo "1, 2246 MiB, 2 %, 30, 20 W" ;;
  *query-gpu=index,memory.used,utilization.gpu*) echo "0, 20, 0"; echo "1, 2246, 2" ;;
  "") echo "| Processes: |" ;;
  *) : ;;
esac
exit 0
EOF
chmod +x "$WORK/shim/"*
cat > "$WORK/fake_upstream_rep070.py" <<'EOF'
import sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def do_POST(self):
        time.sleep(3600)
    def log_message(self, *a):
        pass
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
EOF
PASS=0; FAIL=0
ok() { if eval "$2"; then PASS=$((PASS+1)); echo "  ok    $1"; else FAIL=$((FAIL+1)); echo "  FAIL  $1"; fi; }
setup() { # fake-chain mode: ok | fail
  local P=$WORK/pkg; GW=$WORK/gpqa; rm -rf "$P" "$GW" "$WORK/run-"* "$WORK/chain.log"; mkdir -p "$GW"
  cp -a "$PKG0" "$P"; rm -rf "$P/dryruns" "$P/power"
  cat > "$P/runner/local-eval-chain-rep070.sh" <<EOF
#!/usr/bin/env bash
# FAKE chain (session simulation): skip rule of the real chain, placeholder pass directories, sleep.
echo "fake chain \$* TAG=\$GPQA_TAG GPQA_PASSES=\$GPQA_PASSES SEED_LIST=\$GPQA_SEED_LIST CONC=\$GPQA_CONCURRENCY BASE=\$GPQA_BASE" >> "$WORK/chain.log"
[ "$1" = fail ] && { echo "fake chain: START FAILED" >> "$WORK/chain.log"; exit 1; }
if [ "$1" = inflight ]; then
  python3 "$WORK/fake_upstream_rep070.py" 8001 &
  for i in 1 2 3 4 5 6 7 8 9 10; do ss -ltnH '( sport = :8001 )' | grep -q . && break; sleep 0.5; done
  curl -s -m 600 -o /dev/null -X POST http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages": [{"role": "user", "content": "synthetic session-sim request"}], "seed": 70701, "max_tokens": 327680}' &
  echo "fake chain: one request in flight via the proxy" >> "$WORK/chain.log"
  sleep 299; exit 0
fi
for p in \$(seq 1 \${GPQA_PASSES:-3}); do d="$GW/\${1}\${GPQA_TAG}__pass\$p"; [ -d "\$d" ] && { echo "pass \$p exists, skip" >> "$WORK/chain.log"; continue; }
  mkdir -p "\$d/model"; : > "\$d/model/samples_placeholder.jsonl"; echo "pass \$p run, seed \$(echo "\$GPQA_SEED_LIST" | cut -d' ' -f\$p)" >> "$WORK/chain.log"; done
sleep "\${FAKE_CHAIN_SECS:-2}"
EOF
  diff <stacks>/glm53-flash-v0.8.0/eval-chain.sh "$P/runner/local-eval-chain-rep070.sh" > "$P/runner/local-eval-chain-rep070.diff"
  sed -i -e "s#^R=\"glm53-flash-exl3-k3.25@tpurtell\"; G=<gpqa-runs>; #R=\"glm53-flash-exl3-k3.25@tpurtell\"; G=$GW; #" "$P/runner/run_rep070.sh"
  grep -q "G=$GW;" "$P/runner/run_rep070.sh" || { echo "setup: could not redirect G"; exit 2; }
  ( cd "$P" && ./make_sums.sh > /dev/null )
  RUNNER=$P/runner/run_rep070.sh
}
common() { # scenario output, expected sessions with a cost line
  local out=$1 n=$2
  ok "exactly $n cost line(s), 2 GPUs each" "[ \$(wc -l < '$WORK/run-$V/costs.jsonl' 2>/dev/null || echo 0) = $n ] && [ \$(grep -c '\"gpus\": 2' '$WORK/run-$V/costs.jsonl' 2>/dev/null || echo 0) = $n ]"
  ok "no refused docker call" "! grep -q REFUSED '$SHIMLOG'"
  ok "fake chain gone" "! pgrep -f '$WORK/pkg/runner/local-eval-chain-rep070.sh' >/dev/null"
  ok "proxy gone (8002 free)" "[ -z \"\$(ss -ltnH '( sport = :8002 )')\" ]"
  ok "no runner process left (GPU logger, watchdog)" "! pgrep -f '$RUNNER' >/dev/null"
  ok "no orphaned logger/watchdog sleep left (children of the runner's loops)" "[ -z \"\$(ps -eo pid=,args= | awk '\$2==\"sleep\" && (\$3==\"300\" || \$3==\"60\")' | while read -r pid rest; do [ \"\$(readlink /proc/\$pid/cwd)\" = '$PWD' ] && echo \$pid; done)\" ]"
}
V=R
echo "== (a) normal completion, variant R (two sessions)"
setup ok; : > "$SHIMLOG"
FAKE_CHAIN_SECS=2 VARIANT=R APPROVED_GPU_H=26.45 PATH="$WORK/shim:$PATH" bash "$RUNNER" > "$WORK/a.out" 2>&1; echo "  runner rc=$?"
common "$WORK/a.out" 2
ok "session 1: GPQA_PASSES=1, seeds 1234 1235 1236, concurrency 8, via the proxy, tag +rep070-R" "grep -q 'TAG=+rep070-R GPQA_PASSES=1 SEED_LIST=1234 1235 1236 CONC=8 BASE=http://127.0.0.1:8002' '$WORK/chain.log'"
ok "session 2: GPQA_PASSES=3; pass 1 skipped; passes 2 and 3 run at 1235 and 1236" "grep -q 'GPQA_PASSES=3 SEED_LIST=1234 1235 1236' '$WORK/chain.log' && grep -q 'pass 1 exists, skip' '$WORK/chain.log' && grep -q 'pass 2 run, seed 1235' '$WORK/chain.log' && grep -q 'pass 3 run, seed 1236' '$WORK/chain.log'"
ok "server stopped after each session (2 docker stop calls)" "[ \$(grep -c 'docker stop glm53-flash-exl3-b12x-vllm' '$SHIMLOG') = 2 ]"
ok "server log and docker inspect saved per session" "[ -s '$WORK/run-R/server-s1.log' ] && [ -s '$WORK/run-R/server-s2.log' ] && [ -s '$WORK/run-R/inspect-s1.json' ] && [ -s '$WORK/run-R/inspect-s2.json' ]"
ok "inspect check reports the digest and loopback-only publish" "[ \$(grep 'image OK; published on' '$WORK/a.out' | grep -c '(loopback only)') = 2 ]"
ok "cost lines: chain_rc 0, watchdog_fired false" "[ \$(grep -c '\"chain_rc\": 0, \"watchdog_fired\": false' '$WORK/run-R/costs.jsonl') = 2 ]"
ok "finalize ran: placeholder passes probed INCOMPLETE, report written, no outcome category" "grep -q 'INCOMPLETE run' '$WORK/a.out' && [ -s '$WORK/run-R/incomplete-rep070-R.json' ] && ! grep -q '\"outcome\"' '$WORK/run-R/incomplete-rep070-R.json'"
ok "cache inventory before session 1 and after each session, with diffs" "[ -s '$WORK/run-R/cache-s0.json' ] && [ -s '$WORK/run-R/cache-s2.json' ] && [ -s '$WORK/run-R/cache-diff-s1.json' ] && [ -s '$WORK/run-R/cache-diff-s2.json' ]"
ok "planned seeds written per pass (request seed and lm-eval --seed)" "python3 -c \"import json,sys;d=json.load(open('$WORK/run-R/seeds-planned.json'));sys.exit(0 if [(x['request_seed'],x['lm_eval_process_seed']) for x in d['passes']]==[(1234,1235),(1235,1236),(1236,1237)] else 1)\""
ok "raw directory is mode 0700" "[ \"\$(stat -c %a '$WORK/run-R/raw')\" = 700 ]"
ok "package verification saved" "grep -q ': OK' '$WORK/run-R/package-verify.txt'"
echo "== (b) SIGTERM to the runner during session 2"
setup ok; : > "$SHIMLOG"
( FAKE_CHAIN_SECS=12 VARIANT=R APPROVED_GPU_H=26.45 PATH="$WORK/shim:$PATH" bash "$RUNNER" > "$WORK/b.out" 2>&1; echo "$?" > "$WORK/b.rc" ) &
for i in $(seq 1 60); do grep -q 'session 2: passes up to 3' "$WORK/b.out" 2>/dev/null && break; sleep 1; done
sleep 3; pkill -TERM -f "bash $RUNNER"; wait; echo "  runner rc=$(cat "$WORK/b.rc")"
common "$WORK/b.out" 2
ok "runner exited 4 and reported the interrupted-session cleanup" "[ \"\$(cat '$WORK/b.rc')\" = 4 ] && grep -q 'cleanup after an interrupted session 2' '$WORK/b.out'"
ok "session-2 cost line has chain_rc null" "tail -1 '$WORK/run-R/costs.jsonl' | grep -q '\"session\": 2.*\"chain_rc\": null'"
ok "server stopped in the cleanup (2 docker stop calls)" "[ \$(grep -c 'docker stop glm53-flash-exl3-b12x-vllm' '$SHIMLOG') = 2 ]"
V=F
echo "== (c) watchdog at APPROVED_GPU_H=0.04 (72 s wall; checked every 60 s), variant F"
setup ok; : > "$SHIMLOG"
FAKE_CHAIN_SECS=150 VARIANT=F APPROVED_GPU_H=0.04 PATH="$WORK/shim:$PATH" bash "$RUNNER" > "$WORK/c.out" 2>&1; echo "  runner rc=$?"
common "$WORK/c.out" 1
ok "watchdog fired, killed the chain and stopped the server" "grep -q 'CEILING REACHED' '$WORK/run-F/watchdog.log' && grep -q 'docker stop glm53-flash-exl3-b12x-vllm' '$SHIMLOG'"
ok "cost line: watchdog_fired true; variant F seeds" "grep -q '\"watchdog_fired\": true' '$WORK/run-F/costs.jsonl' && grep -q '\"seeds\": \"70701 70702 70703\"' '$WORK/run-F/costs.jsonl'"
ok "session 2 not started" "! grep -q 'GPQA_PASSES=3' '$WORK/chain.log'"
echo "== (d) approved budget too small for session 2 (15 GPU-h = 27,000 s < 27,955 s)"
setup ok; : > "$SHIMLOG"
FAKE_CHAIN_SECS=1 VARIANT=F APPROVED_GPU_H=15 PATH="$WORK/shim:$PATH" bash "$RUNNER" > "$WORK/d.out" 2>&1; echo "  runner rc=$?"
common "$WORK/d.out" 1
ok "budget stop reported; session 2 not started" "grep -q 'session 2 not started (budget stop)' '$WORK/d.out' && ! grep -q 'GPQA_PASSES=3' '$WORK/chain.log'"
echo "== (e) session 1 fails (chain rc 1, no pass 1)"
setup fail; : > "$SHIMLOG"
VARIANT=F APPROVED_GPU_H=26.45 PATH="$WORK/shim:$PATH" bash "$RUNNER" > "$WORK/e.out" 2>&1; echo "  runner rc=$?"
common "$WORK/e.out" 1
ok "stopped after session 1, no retry" "grep -q 'session 1 did not produce pass 1' '$WORK/e.out' && [ \$(grep -c 'TAG=+rep070' '$WORK/chain.log') = 1 ]"
ok "cost line has chain_rc 1" "grep -q '\"chain_rc\": 1' '$WORK/run-F/costs.jsonl'"
echo "== (f) watchdog while a request is in flight through the proxy copy, variant F"
setup inflight; : > "$SHIMLOG"
VARIANT=F APPROVED_GPU_H=0.04 PATH="$WORK/shim:$PATH" bash "$RUNNER" > "$WORK/f.out" 2>&1; echo "  runner rc=$?"
common "$WORK/f.out" 1
EV="$GW/glm53-flash-exl3-k3.25@tpurtell+rep070-F.s1.requests.events.jsonl"; MAINF="$GW/glm53-flash-exl3-k3.25@tpurtell+rep070-F.s1.requests.jsonl"
ok "watchdog fired; fake upstream stopped by the docker shim" "grep -q 'CEILING REACHED' '$WORK/run-F/watchdog.log' && ! pgrep -f fake_upstream_rep070 >/dev/null"
ok "sidecar: start then interrupted for seed 70701" "python3 -c \"import json,sys;e=[json.loads(l) for l in open('$EV')];sys.exit(0 if [x['event'] for x in e]==['start','interrupted'] and e[1]['seed']==70701 else 1)\""
ok "no main-log row for the interrupted request" "[ ! -s '$MAINF' ]"
ok "raw file mode 0600 with the interrupted record" "[ \"\$(stat -c %a '$WORK/run-F/raw/s1.raw.jsonl.gz')\" = 600 ] && python3 -c \"import gzip,json,sys;r=[json.loads(l) for l in gzip.open('$WORK/run-F/raw/s1.raw.jsonl.gz','rt')];sys.exit(0 if len(r)==1 and r[0]['kind']=='interrupted' else 1)\""
ok "INCOMPLETE report: watchdog cause; pass 1 interrupted 1, unstarted 197, never counted empty" "python3 -c \"import json,sys;d=json.load(open('$WORK/run-F/incomplete-rep070-F.json'));p=d['passes']['pass1'];sys.exit(0 if any('watchdog' in c for c in d['causes']) and p['interrupted_started_no_response']==1 and p['unstarted']==197 and p['responses_logged']==0 else 1)\""
pkill -f "synthetic session-sim request" 2>/dev/null; for P in $(ps -eo pid=,args= | awk '$2=="sleep" && $3=="299" {print $1}'); do [ "$(readlink /proc/$P/cwd)" = "$PWD" ] && kill $P; done   # the fake chain's orphaned sleep
echo "session simulation: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ] && rm -rf "$WORK"
[ "$FAIL" = 0 ]
