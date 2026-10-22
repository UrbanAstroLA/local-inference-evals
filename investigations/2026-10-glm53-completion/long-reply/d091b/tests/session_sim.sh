#!/usr/bin/env bash
set -u
# Session-path simulation of runner/run_d091.sh with NO GPU and NO container (CPU only), after rep070/refreeze/tests/
# session_sim.sh. A copy of the frozen package (the files listed in SHA256SUMS, plus empty PRIVATE/ and out/) goes to
# WORK/pkg. In the runner copy only three lines differ: G points into WORK, the package-path guard accepts the copy's path,
# and the chain pin is the fake chain's hash. The chain at the runner's CH path is a FAKE (same file name, so the watchdog's
# escaped pattern is exercised) that applies the real chain's skip rule, creates the pass directory (an empty placeholder
# samples file, or for scenario f symlinks to the 0.9.1 record's pass-1 files with the seed relabelled in a copy of its
# results json; no GPQA text is written), records its environment and sleeps; the copy's chain diff and SHA256SUMS are
# regenerated, so every refusal still runs. docker and nvidia-smi are shims: read-only queries answer as on a healthy
# machine, logs / inspect / stop are recorded and answered with stubs, any other docker call is refused (exit 99). The real
# recording proxy copy binds 127.0.0.1:8002. Scenarios: (a) normal end of the session; (b) SIGTERM to the runner during the
# session; (c) watchdog at 0.04 GPU-h; (d) the session fails (chain rc 1, no pass); (e) the watchdog fires while a request
# is in flight through the proxy copy (a hanging fake upstream on 8001; the docker shim's 'stop' kills it, as a stopped
# server would); (f) a complete pass directory without proxy rows: the probe says COMPLETE, the frozen analysis says
# INVALID, and the long-reply pipeline is not run.
# Refuses to run if an lm_eval process exists or port 8001/8002 is in use. Not set -e.
H=$(cd "$(dirname "$0")" && pwd); PKG0=$(dirname "$H")
case "$PKG0" in <runs>/long-reply/d091b) ;; *) echo "ABORT: unexpected package path"; exit 1 ;; esac
WORK="$PKG0/tests/sim-work"
[ -n "$WORK" ] && case "$WORK" in <runs>/long-reply/d091b/tests/sim-work) ;; *) exit 1 ;; esac
R="glm53-flash-exl3-k3.25@tpurtell+v0.9.1"; IMG=sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
V091=<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1
pgrep -f -- 'bin/lm_eval --model' >/dev/null && { echo "REFUSED: an lm_eval process exists"; exit 2; }
[ -n "$(ss -ltnH '( sport = :8001 or sport = :8002 )')" ] && { echo "REFUSED: port 8001/8002 in use"; exit 2; }
[ -f "$PKG0/SHA256SUMS" ] || { echo "REFUSED: no SHA256SUMS to copy the package from"; exit 2; }
rm -rf -- "$WORK"; mkdir -p "$WORK/shim" || exit 2
SHIMLOG=$WORK/shim-calls.log; : > "$SHIMLOG"
cat > "$WORK/shim/docker" <<EOF
#!/usr/bin/env bash
set -u
echo "\$(date +%s) docker \$*" >> "$SHIMLOG"
case "\$1" in
  image) echo $IMG; exit 0 ;;
  ps) echo "open-webui 8afd2d774834"; exit 0 ;;
  logs) echo "stub server log"; exit 0 ;;
  inspect) echo '[{"Image": "$IMG", "HostConfig": {"PortBindings": {"8001/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8001"}]}}}]'; exit 0 ;;
  stop) pkill -f fake_upstream_d091 2>/dev/null; echo "\$2"; exit 0 ;;
esac
echo "REFUSED docker \$*" >> "$SHIMLOG"; exit 99
EOF
cat > "$WORK/shim/nvidia-smi" <<EOF
#!/usr/bin/env bash
set -u
case "\$*" in
  -L) echo "GPU 0: shim"; echo "GPU 1: shim" ;;
  *query-compute-apps*) : ;;
  *query-gpu=index,memory.used,utilization.gpu,temperature*) echo "0, 20 MiB, 0 %, 30, 20 W"; echo "1, 2038 MiB, 2 %, 30, 20 W" ;;
  *query-gpu=index,memory.used,utilization.gpu*) echo "0, 20, 0"; echo "1, 2038, 2" ;;
  "") echo "| Processes: |" ;;
  *) : ;;
esac
exit 0
EOF
chmod +x "$WORK/shim/"*
cat > "$WORK/fake_upstream_d091.py" <<'EOF'
import sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def do_POST(self):
        time.sleep(3600)
    def log_message(self, *a):
        pass
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
EOF
cat > "$WORK/fake_pass.py" <<'EOF'
import glob, json, os, sys
dst, base, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
src = f"{base}+ps__pass1"
pd = os.path.join(dst, "model"); os.makedirs(pd, exist_ok=True)
sf = glob.glob(src + "/**/samples_*.jsonl", recursive=True)[0]; rf = glob.glob(src + "/**/results_*.json", recursive=True)[0]
os.symlink(sf, os.path.join(pd, os.path.basename(sf)))
r = json.load(open(rf)); r["config"]["model_args"]["seed"] = seed
json.dump(r, open(os.path.join(pd, os.path.basename(rf)), "w"))
EOF
PASS=0; FAIL=0
ok() { if eval "$2"; then PASS=$((PASS+1)); echo "  ok    $1"; else FAIL=$((FAIL+1)); echo "  FAIL  $1"; fi; }
P=$WORK/pkg; GW=$WORK/gpqa
setup() { # fake-chain mode: ok | fail | inflight | complete
  [ -n "$P" ] && case "$P" in "$WORK"/pkg) ;; *) exit 2 ;; esac
  rm -rf -- "$P" "$GW" "$WORK/chain.log"; mkdir -p "$GW" "$P" || exit 2
  ( cd "$PKG0" && cut -c67- SHA256SUMS | while read -r f; do mkdir -p "$P/$(dirname "$f")" && cp -p "$f" "$P/$f"; done )
  ( umask 077; mkdir -p "$P/PRIVATE" ) && chmod 700 "$P/PRIVATE"; mkdir -p "$P/out"
  rm -rf -- "$P/tests/dryruns" "$P/tests/repro-discovery" "$P/tests/prefix-repro" "$P/tests/index-repro" "$P/tests/standin"
  cat > "$P/runner/local-eval-chain-rep070.sh" <<EOF
#!/usr/bin/env bash
set -u
# FAKE chain (session simulation): skip rule of the real chain, pass directory, sleep.
echo "fake chain \$* TAG=\$GPQA_TAG GPQA_PASSES=\$GPQA_PASSES SEED_LIST=\$GPQA_SEED_LIST CONC=\$GPQA_CONCURRENCY BASE=\$GPQA_BASE" >> "$WORK/chain.log"
[ "$1" = fail ] && { echo "fake chain: START FAILED" >> "$WORK/chain.log"; exit 1; }
if [ "$1" = inflight ]; then
  python3 "$WORK/fake_upstream_d091.py" 8001 &
  for i in 1 2 3 4 5 6 7 8 9 10; do ss -ltnH '( sport = :8001 )' | grep -q . && break; sleep 0.5; done
  curl -s -m 600 -o /dev/null -X POST http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages": [{"role": "user", "content": "synthetic session-sim request"}], "seed": 70705, "max_tokens": 327680}' &
  echo "fake chain: one request in flight via the proxy" >> "$WORK/chain.log"
  sleep 299; exit 0
fi
d="$GW/\${1}\${GPQA_TAG}__pass1"; [ -d "\$d" ] && { echo "pass 1 exists, skip" >> "$WORK/chain.log"; exit 0; }
if [ "$1" = complete ]; then python3 "$WORK/fake_pass.py" "\$d" "$V091" "\$GPQA_SEED_LIST"; else mkdir -p "\$d/model"; : > "\$d/model/samples_placeholder.jsonl"; fi
echo "pass 1 run, seed \$(echo "\$GPQA_SEED_LIST" | cut -d' ' -f1)" >> "$WORK/chain.log"
sleep "\${FAKE_CHAIN_SECS:-2}"
EOF
  diff <stacks>/glm53-flash-v0.8.0/eval-chain.sh "$P/runner/local-eval-chain-rep070.sh" > "$P/runner/local-eval-chain-rep070.diff"
  local FH; FH=$(sha256sum "$P/runner/local-eval-chain-rep070.sh" | cut -c1-64)
  sed -i -e "s#^R=\"glm53-flash-exl3-k3.25@tpurtell+v0.9.1\"; G=<gpqa-runs>; #R=\"glm53-flash-exl3-k3.25@tpurtell+v0.9.1\"; G=$GW; #" \
         -e "s#^case \"\$PKG\" in <runs>/long-reply/d091b) #case \"\$PKG\" in $P) #" \
         -e "s#^  \[\"\$CH\"\]=81ef338653a2f9cbffff62123113238953f232c268c8e3f02be9e66080d26b9e#  [\"\$CH\"]=$FH#" "$P/runner/run_d091.sh"
  grep -q "G=$GW;" "$P/runner/run_d091.sh" && grep -q "in $P) ;;" "$P/runner/run_d091.sh" && grep -q "\[\"\$CH\"\]=$FH" "$P/runner/run_d091.sh" \
    || { echo "setup: could not patch the runner copy"; exit 2; }
  sed -i -e "s#^case \"\$PKG\" in <runs>/long-reply/d091b) #case \"\$PKG\" in $P) #" "$P/scripts/run_pipeline.sh"
  ( cd "$P" && bash ./make_sums.sh > /dev/null )
  RUNNER=$P/runner/run_d091.sh
}
common() { # expected cost lines
  local n=$1
  ok "exactly $n cost line(s), 2 GPUs each" "[ \$(wc -l < '$P/run/costs.jsonl' 2>/dev/null || echo 0) = $n ] && [ \$(grep -c '\"gpus\": 2' '$P/run/costs.jsonl' 2>/dev/null || echo 0) = $n ]"
  ok "no refused docker call" "! grep -q REFUSED '$SHIMLOG'"
  ok "fake chain gone" "! pgrep -f '$P/runner/local-eval-chain-rep070.sh' >/dev/null"
  ok "proxy gone (8002 free)" "[ -z \"\$(ss -ltnH '( sport = :8002 )')\" ]"
  ok "no runner process left (GPU logger, watchdog)" "! pgrep -f '$RUNNER' >/dev/null"
  ok "raw directory under PRIVATE/ is mode 0700; any raw file 0600" "[ \"\$(stat -c %a '$P/PRIVATE/raw')\" = 700 ] && { [ ! -e '$P/PRIVATE/raw/s1.raw.jsonl.gz' ] || [ \"\$(stat -c %a '$P/PRIVATE/raw/s1.raw.jsonl.gz')\" = 600 ]; }"
  ok "long-reply pipeline not run (out/ empty)" "[ -z \"\$(ls -A '$P/out')\" ]"
}
echo "== (a) normal end of the session (placeholder pass)"
setup ok; : > "$SHIMLOG"
FAKE_CHAIN_SECS=2 APPROVED_GPU_H=10 PATH="$WORK/shim:$PATH" bash "$RUNNER" > "$WORK/a.out" 2>&1; echo "  runner rc=$?"
common 1
ok "chain: tag +d091b, GPQA_PASSES=1, seed list 70705, concurrency 8, via the proxy" "grep -q 'TAG=+d091b GPQA_PASSES=1 SEED_LIST=70705 CONC=8 BASE=http://127.0.0.1:8002' '$WORK/chain.log'"
ok "server stopped once" "[ \$(grep -c 'docker stop glm53-flash-exl3-b12x-vllm' '$SHIMLOG') = 1 ]"
ok "server log and docker inspect saved" "[ -s '$P/run/server-s1.log' ] && [ -s '$P/run/inspect-s1.json' ]"
ok "inspect check reports the digest and loopback-only publish" "grep -q 'image OK; published on .*(loopback only)' '$WORK/a.out'"
ok "cost line: chain_rc 0, watchdog_fired false, seed 70705" "grep -q '\"seed\": 70705.*\"chain_rc\": 0, \"watchdog_fired\": false' '$P/run/costs.jsonl'"
ok "finalize: placeholder pass probed INCOMPLETE, report written, never classified" "grep -q 'INCOMPLETE run' '$WORK/a.out' && [ -s '$P/run/incomplete-d091.json' ] && [ ! -e '$P/run/result-d091.json' ]"
ok "cache inventory before and after, with diff" "[ -s '$P/run/cache-s0.json' ] && [ -s '$P/run/cache-s1.json' ] && [ -s '$P/run/cache-diff-s1.json' ]"
ok "planned seeds: request 70705, lm-eval process 1235" "python3 -c \"import json,sys;d=json.load(open('$P/run/seeds-planned.json'));sys.exit(0 if [(x['request_seed'],x['lm_eval_process_seed']) for x in d['passes']]==[(70705,1235)] else 1)\""
ok "package verification saved" "grep -q ': OK' '$P/run/package-verify.txt'"
echo "== (b) SIGTERM to the runner during the session"
setup ok; : > "$SHIMLOG"
( FAKE_CHAIN_SECS=20 APPROVED_GPU_H=10 PATH="$WORK/shim:$PATH" bash "$RUNNER" > "$WORK/b.out" 2>&1; echo "$?" > "$WORK/b.rc" ) &
for i in $(seq 1 60); do grep -q 'session 1: pass 1, seed 70705' "$WORK/b.out" 2>/dev/null && break; sleep 1; done
sleep 3; pkill -TERM -f "bash $RUNNER"; wait; echo "  runner rc=$(cat "$WORK/b.rc")"
common 1
ok "runner exited 4 and reported the interrupted-session cleanup" "[ \"\$(cat '$WORK/b.rc')\" = 4 ] && grep -q 'cleanup after an interrupted session' '$WORK/b.out'"
ok "cost line has chain_rc null" "grep -q '\"chain_rc\": null' '$P/run/costs.jsonl'"
ok "server stopped in the cleanup" "[ \$(grep -c 'docker stop glm53-flash-exl3-b12x-vllm' '$SHIMLOG') -ge 1 ]"
ok "INCOMPLETE report with the interruption as a cause; never classified" "python3 -c \"import json,sys;d=json.load(open('$P/run/incomplete-d091.json'));sys.exit(0 if d['status']=='INCOMPLETE' and any('interrupted' in c for c in d['causes']) else 1)\" && [ ! -e '$P/run/result-d091.json' ]"
echo "== (c) watchdog at APPROVED_GPU_H=0.04 (72 s wall; checked every 60 s)"
setup ok; : > "$SHIMLOG"
FAKE_CHAIN_SECS=150 APPROVED_GPU_H=0.04 PATH="$WORK/shim:$PATH" bash "$RUNNER" > "$WORK/c.out" 2>&1; echo "  runner rc=$?"
common 1
ok "watchdog fired, killed the chain and stopped the server" "grep -q 'CEILING REACHED' '$P/run/watchdog.log' && grep -q 'docker stop glm53-flash-exl3-b12x-vllm' '$SHIMLOG'"
ok "cost line: watchdog_fired true" "grep -q '\"watchdog_fired\": true' '$P/run/costs.jsonl'"
ok "INCOMPLETE report with the watchdog cause" "python3 -c \"import json,sys;d=json.load(open('$P/run/incomplete-d091.json'));sys.exit(0 if any('watchdog' in c for c in d['causes']) else 1)\""
echo "== (d) the session fails (chain rc 1, no pass)"
setup fail; : > "$SHIMLOG"
APPROVED_GPU_H=10 PATH="$WORK/shim:$PATH" bash "$RUNNER" > "$WORK/d.out" 2>&1; echo "  runner rc=$?"
common 1
ok "stopped after the session, no retry" "grep -q 'did not produce pass 1' '$WORK/d.out' && [ \$(grep -c 'TAG=+d091b' '$WORK/chain.log') = 1 ]"
ok "cost line has chain_rc 1; INCOMPLETE report" "grep -q '\"chain_rc\": 1' '$P/run/costs.jsonl' && [ -s '$P/run/incomplete-d091.json' ]"
echo "== (e) watchdog while a request is in flight through the proxy copy"
setup inflight; : > "$SHIMLOG"
APPROVED_GPU_H=0.04 PATH="$WORK/shim:$PATH" bash "$RUNNER" > "$WORK/e.out" 2>&1; echo "  runner rc=$?"
common 1
EV="$GW/$R+d091b.s1.requests.events.jsonl"; MAINF="$GW/$R+d091b.s1.requests.jsonl"
ok "watchdog fired; fake upstream stopped by the docker shim" "grep -q 'CEILING REACHED' '$P/run/watchdog.log' && ! pgrep -f fake_upstream_d091 >/dev/null"
ok "sidecar: start then interrupted for seed 70705" "python3 -c \"import json,sys;e=[json.loads(l) for l in open('$EV')];sys.exit(0 if [x['event'] for x in e]==['start','interrupted'] and e[1]['seed']==70705 else 1)\""
ok "no main-log row for the interrupted request" "[ ! -s '$MAINF' ]"
ok "raw file in PRIVATE/raw, mode 0600, holds the interrupted record" "[ \"\$(stat -c %a '$P/PRIVATE/raw/s1.raw.jsonl.gz')\" = 600 ] && python3 -c \"import gzip,json,sys;r=[json.loads(l) for l in gzip.open('$P/PRIVATE/raw/s1.raw.jsonl.gz','rt')];sys.exit(0 if len(r)==1 and r[0]['kind']=='interrupted' else 1)\""
ok "INCOMPLETE report: watchdog cause; pass 1 interrupted 1, unstarted 197, no response logged, never counted empty" "python3 -c \"import json,sys;d=json.load(open('$P/run/incomplete-d091.json'));p=d['pass1'];sys.exit(0 if any('watchdog' in c for c in d['causes']) and p['interrupted_started_no_response']==1 and p['unstarted']==197 and p['responses_logged']==0 else 1)\""
pkill -f "synthetic session-sim request" 2>/dev/null; for Q in $(ps -eo pid=,args= | awk '$2=="sleep" && $3=="299" {print $1}'); do [ "$(readlink /proc/$Q/cwd)" = "$PWD" ] && kill $Q; done
echo "== (f) complete pass directory without proxy rows: COMPLETE probe -> frozen analysis INVALID -> no pipeline"
setup complete; : > "$SHIMLOG"
FAKE_CHAIN_SECS=1 APPROVED_GPU_H=10 PATH="$WORK/shim:$PATH" bash "$RUNNER" > "$WORK/f.out" 2>&1; echo "  runner rc=$?"
common 1
ok "probe COMPLETE, frozen analysis ran and returned INVALID (rc 3: no proxy request log, no proxy rows)" "grep -q 'COMPLETE run: frozen analysis rc=3' '$WORK/f.out' && python3 -c \"import json,sys;d=json.load(open('$P/run/result-d091.json'));f=[c[0] for c in d['checks'] if not c[1]];sys.exit(0 if d['valid'] is False and 'proxy request log present' in f else 1)\""
ok "INVALID run: pipeline explicitly not run" "grep -q 'INVALID run: the long-reply pipeline is not run' '$WORK/f.out'"
echo "session simulation: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ] && rm -rf -- "$WORK"
[ "$FAIL" = 0 ]
