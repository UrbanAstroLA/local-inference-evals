#!/usr/bin/env bash
# Session-path simulation of runner/run_bplus.sh with NO GPU and NO model server (CPU only), after
# rep070/refreeze/tests/session_sim.sh. A copy of the package goes to WORK/pkg; in the runner copy only the per-session
# minimum walls are lowered (SESSION_MIN_S) and the copy's SHA256SUMS is regenerated, so every refusal still runs.
# docker and nvidia-smi are shims: read-only queries answer as on a healthy machine; the REAL stack launchers run
# unmodified and their 'docker run --detach' (with --gpus) is answered by starting tests/fake_server.py on 127.0.0.1:8001
# (no model); 'docker logs' returns the engine's real reference server log; 'docker stop' kills the fake server; the
# analysis' CPU-only container ('docker run --rm --network none', no --gpus) passes through to the real docker; any
# other docker call is refused (exit 99). Scenarios:
#   (a) normal completion of all three sessions -> analysis COMPLETE
#   (b) SIGTERM to the runner during session 2 -> cleanup, INCOMPLETE
#   (c) watchdog at APPROVED_GPU_H=0.04 (72 s wall) while requests are in flight -> INCOMPLETE
#   (d) budget rule stops B1 (its minimum wall raised in this scenario) -> C1-ONLY
#   (e) session 1 launcher fails -> stop, later engines not started, INCOMPLETE
#   (f) one request answered HTTP 500 in session 2 -> client rc 3, stop, INCOMPLETE
# Refuses to run if port 8001 is in use or a GLM container runs. Not set -e.
WORK=${1:?usage: session_sim.sh WORK_DIR}
HERE=$(cd "$(dirname "$0")" && pwd); PKG0=$(dirname "$HERE")
[ -n "$(ss -ltnH '( sport = :8001 )')" ] && { echo "REFUSED: port 8001 in use"; exit 2; }
docker ps --format '{{.Names}}' | grep -qE '^glm53' && { echo "REFUSED: a GLM container runs"; exit 2; }
REAL_DOCKER=$(command -v docker)
rm -rf "$WORK"; mkdir -p "$WORK/shim" "$WORK/state" || exit 2
SHIMLOG=$WORK/shim-calls.log; : > "$SHIMLOG"
cat > "$WORK/shim/docker" <<EOF
#!/usr/bin/env bash
ST=$WORK/state; PKGC=$WORK/pkg
echo "\$(date +%s) docker \$*" >> "$SHIMLOG"
idof() { case "\$1" in *48e254d9*) echo sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629 ;;
  *a9e985c0*) echo sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07 ;;
  *799d3cf8*|local/glm53-v0.9.1-b1-d7:local) echo sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a ;;
  *) return 1 ;; esac; }
case "\$1" in
  ps) [ -f "\$ST/running" ] && echo "glm53-flash-exl3-b12x-vllm \$(cat "\$ST/image")"; echo "open-webui 8afd2d774834"; exit 0 ;;
  image) id=\$(idof "\$3") || exit 1; [[ "\$*" == *--format* ]] && echo "\$id"; exit 0 ;;
  inspect)
    if [ "\$2" = -f ] || [ "\$2" = --format ]; then [ -f "\$ST/running" ] && { echo true; exit 0; }; [ -f "\$ST/exists" ] && { echo false; exit 0; }; exit 1; fi
    [ -f "\$ST/exists" ] || exit 1
    pub=\$(cat "\$ST/publish"); echo "[{\"Image\": \"\$(idof "\$(cat "\$ST/image")")\", \"HostConfig\": {\"PortBindings\": {\"8001/tcp\": [{\"HostIp\": \"\${pub%:*}\", \"HostPort\": \"8001\"}]}}}]"; exit 0 ;;
  rm) rm -f "\$ST/exists"; exit 0 ;;
  run)
    if [[ " \$* " == *" --detach "* ]]; then
      [ -n "\$SIM_LAUNCH_FAIL" ] && { echo "sim: launcher docker run fails" >&2; exit 125; }
      pub=""; img=""; prev=""
      for a in "\$@"; do [ "\$prev" = --publish ] && pub=\${a%:8001}; idof "\$a" > /dev/null && img=\$a; prev=\$a; done
      [ -n "\$img" ] && [ -n "\$pub" ] || { echo "REFUSED docker run (no image/publish)" >> "$SHIMLOG"; exit 99; }
      echo "\$img" > "\$ST/image"; echo "\$pub" > "\$ST/publish"; touch "\$ST/exists" "\$ST/running"
      echo "\$*" > "\$ST/last-run-args"
      fkv=SIM_FAIL_K_\$(idof "\$img" | cut -c8-15)
      FAKE_FAIL_K=\${!fkv:-0} setsid python3 -B "$HERE/fake_server.py" 8001 "\$PKGC" > "\$ST/fake.log" 2>&1 &
      echo \$! > "\$ST/pid"
      for i in \$(seq 1 50); do ss -ltnH '( sport = :8001 )' | grep -q . && break; sleep 0.1; done
      echo fakecontainerid; exit 0
    fi
    if [[ " \$* " == *" --rm "* ]] && [[ " \$* " == *" --network none "* ]] && [[ " \$* " != *" --gpus"* ]]; then exec "$REAL_DOCKER" "\$@"; fi
    echo "REFUSED docker \$*" >> "$SHIMLOG"; exit 99 ;;
  exec) [ -f "\$ST/running" ] && exit 0; exit 1 ;;
  logs) case "\$(cat "\$ST/image" 2>/dev/null)" in
          *48e254d9*) cat <runs>/rep070/run-F/server-s1.log ;;
          *a9e985c0*) cat <gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1.server.log ;;
          *799d3cf8*) cat <runs>/b1-screen/run-s1236/server-s1.log ;;
        esac; exit 0 ;;
  stop) [ -f "\$ST/pid" ] && kill "\$(cat "\$ST/pid")" 2>/dev/null; rm -f "\$ST/running" "\$ST/pid"; echo "\$2"; exit 0 ;;
esac
echo "REFUSED docker \$*" >> "$SHIMLOG"; exit 99
EOF
cat > "$WORK/shim/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  -L) echo "GPU 0: shim"; echo "GPU 1: shim" ;;
  *query-compute-apps*) : ;;
  *query-gpu=index,memory.used,utilization.gpu,temperature*) echo "0, 20 MiB, 0 %, 30, 20 W, 210 MHz, 405 MHz, P8"; echo "1, 2246 MiB, 2 %, 30, 20 W, 210 MHz, 405 MHz, P8" ;;
  *query-gpu=index,memory.used,utilization.gpu*) echo "0, 20, 0"; echo "1, 2246, 2" ;;
  "") echo "| Processes: |" ;;
  *) : ;;
esac
exit 0
EOF
chmod +x "$WORK/shim/"*
PASS=0; FAIL=0
ok() { if eval "$2"; then PASS=$((PASS+1)); echo "  ok    $1"; else FAIL=$((FAIL+1)); echo "  FAIL  $1"; fi; }
setup() { # minimum walls v070 v091 b1
  local P=$WORK/pkg; rm -rf "$P" "$WORK/state"/*
  cp -a "$PKG0" "$P"; rm -rf "$P/run" "$P/tests/out"
  sed -i -E "s/^declare -A SESSION_MIN_S=.*/declare -A SESSION_MIN_S=([v070]=$1 [v091]=$2 [b1]=$3)   # SIM/" "$P/runner/run_bplus.sh"
  grep -q '# SIM' "$P/runner/run_bplus.sh" || { echo "setup: could not set SESSION_MIN_S"; exit 2; }
  ( cd "$P" && ./make_sums.sh > /dev/null )
  RUNNER=$P/runner/run_bplus.sh; R=$P/run
}
common() {
  local n=$1
  ok "exactly $n cost line(s), 2 GPUs each" "[ \$(wc -l < '$R/costs.jsonl' 2>/dev/null || echo 0) = $n ] && [ \$(grep -c '\"gpus\": 2' '$R/costs.jsonl' 2>/dev/null || echo 0) = $n ]"
  ok "no refused docker call" "! grep -q REFUSED '$SHIMLOG'"
  ok "every launcher docker run published on 127.0.0.1:8001 by the pinned image" "! grep -E 'docker run .*--detach' '$SHIMLOG' | grep -vqE -- '--publish 127.0.0.1:8001:8001 .*(48e254d9|a9e985c0|799d3cf8)'"
  ok "fake server gone (8001 free), no client left" "[ -z \"\$(ss -ltnH '( sport = :8001 )')\" ] && ! pgrep -f '$WORK/pkg/scripts/client_bplus.py' >/dev/null"
  ok "no runner process left" "! pgrep -f '$RUNNER' >/dev/null"
  ok "raw directory 0700, raw files 0600" "[ \"\$(stat -c %a '$R/raw')\" = 700 ] && ! find '$R/raw' -type f ! -perm 600 | grep -q ."
  ok "analysis ran and wrote result.json" "[ -s '$R/result.json' ]"
}
state() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['state'], d.get('reading'))" "$R/result.json" 2>/dev/null; }
echo "== (a) normal completion of all three sessions"
setup 10 10 10; : > "$SHIMLOG"
FAKE_SECS=0.05 APPROVED_GPU_H=31.6 PATH="$WORK/shim:$PATH" bash "$RUNNER" > "$WORK/a.out" 2>&1; echo "  runner rc=$? result: $(state)"
common 3
ok "three sessions completed, 48 requests each, 8-in-flight cap held" "python3 -c \"import json,sys
for e in ('v070','v091','b1'):
    s=json.load(open('$R/'+e+'/session.json')); r=[json.loads(l) for l in open('$R/'+e+'/requests.jsonl')]
    assert s['completed'] and s['ready'] and s['preflight_ok'] and s['client_rc']==0, e
    assert sum(x['kind']=='request' for x in r)==48 and sum(x['kind']=='render_check' and x['equal'] for x in r)==12, e\""
ok "server stopped after each session (3 docker stop calls)" "[ \$(grep -c 'docker stop glm53-flash-exl3-b12x-vllm' '$SHIMLOG') = 3 ]"
ok "server log and docker inspect saved per session; inspect reports loopback only" "[ -s '$R/v070/server.log' ] && [ -s '$R/b1/inspect.json' ] && [ \$(grep -c '(loopback only)' '$WORK/a.out') = 3 ]"
ok "analysis: COMPLETE with a reading (all gates passed on the fake outputs)" "state | grep -qE '^COMPLETE R-'"
ok "cache inventory before and after each session" "[ -s '$R/cache-v070-before.json' ] && [ -s '$R/cache-diff-b1.json' ]"
ok "GPU log and package verification written" "[ -s '$R/nvidia-smi.log' ] && grep -q ': OK' '$R/package-verify.txt'"
echo "== (b) SIGTERM to the runner during session 2"
setup 10 10 10; : > "$SHIMLOG"
( FAKE_SECS=3 APPROVED_GPU_H=31.6 PATH="$WORK/shim:$PATH" bash "$RUNNER" > "$WORK/b.out" 2>&1; echo "$?" > "$WORK/b.rc" ) &
for i in $(seq 1 240); do grep -q 'v091: ready; client starts' "$WORK/b.out" 2>/dev/null && break; sleep 1; done
sleep 4; pkill -TERM -f "bash $RUNNER"; wait; echo "  runner rc=$(cat "$WORK/b.rc") result: $(state)"
common 2
ok "runner exited 4 and reported the interrupted-session cleanup" "[ \"\$(cat '$WORK/b.rc')\" = 4 ] && grep -q 'cleanup after an interrupted v091 session' '$WORK/b.out'"
ok "analysis INCOMPLETE, no reading" "state | grep -q '^INCOMPLETE None'"
ok "B1 never started" "! grep -q 'b1: starting' '$WORK/b.out'"
echo "== (c) watchdog at APPROVED_GPU_H=0.04 (72 s wall; checked every 60 s) with requests in flight"
setup 10 10 10; : > "$SHIMLOG"
FAKE_SECS=400 APPROVED_GPU_H=0.04 PATH="$WORK/shim:$PATH" bash "$RUNNER" > "$WORK/c.out" 2>&1; echo "  runner rc=$? result: $(state)"
common 1
ok "watchdog fired, killed the client and stopped the server" "grep -q 'CEILING REACHED' '$R/watchdog.log' && grep -q 'docker stop glm53-flash-exl3-b12x-vllm' '$SHIMLOG'"
ok "cost line: watchdog_fired true" "grep -q '\"watchdog_fired\": true' '$R/costs.jsonl'"
ok "analysis INCOMPLETE, no reading; later engines not started" "state | grep -q '^INCOMPLETE None' && ! grep -q 'v091: starting' '$WORK/c.out'"
echo "== (d) budget rule: B1's minimum wall exceeds what is left -> B1 not started, C1-ONLY"
setup 10 10 999999; : > "$SHIMLOG"
FAKE_SECS=0.05 APPROVED_GPU_H=31.6 PATH="$WORK/shim:$PATH" bash "$RUNNER" > "$WORK/d.out" 2>&1; echo "  runner rc=$? result: $(state)"
common 2
ok "budget stop reported for b1; b1 session.json budget_stop true" "grep -q 'b1: .*not started (budget stop)' '$WORK/d.out' && grep -q '\"budget_stop\": true' '$R/b1/session.json'"
ok "analysis C1-ONLY with a C1 reading" "state | grep -qE '^C1-ONLY R-'"
echo "== (e) session 1 launcher fails"
setup 10 10 10; : > "$SHIMLOG"
SIM_LAUNCH_FAIL=1 APPROVED_GPU_H=31.6 PATH="$WORK/shim:$PATH" bash "$RUNNER" > "$WORK/e.out" 2>&1; echo "  runner rc=$? result: $(state)"
common 1
ok "stopped after session 1; v091 and b1 not started; no client ran" "grep -q 'v070: session did not complete' '$WORK/e.out' && ! grep -q 'v091: starting' '$WORK/e.out' && [ ! -e '$R/v070/requests.jsonl' ]"
ok "analysis INCOMPLETE" "state | grep -q '^INCOMPLETE None'"
echo "== (f) one request answered HTTP 500 in session 2"
setup 10 10 10; : > "$SHIMLOG"
SIM_FAIL_K_a9e985c0=17 FAKE_SECS=0.05 APPROVED_GPU_H=31.6 PATH="$WORK/shim:$PATH" bash "$RUNNER" > "$WORK/f.out" 2>&1; echo "  runner rc=$? result: $(state)"
common 2
ok "client rc 3 in v091; run stopped; b1 not started" "grep -q 'v091: client rc=3' '$WORK/f.out' && ! grep -q 'b1: starting' '$WORK/f.out'"
ok "the failed request is logged with its HTTP status (no retry)" "[ \$(grep -c '\"http_status\": 500' '$R/v091/requests.jsonl') = 1 ] && [ \$(grep -c '\"kind\": \"request\"' '$R/v091/requests.jsonl') = 48 ]"
ok "analysis INCOMPLETE" "state | grep -q '^INCOMPLETE None'"
echo "session simulation: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ] && rm -rf "$WORK"
[ "$FAIL" = 0 ]
