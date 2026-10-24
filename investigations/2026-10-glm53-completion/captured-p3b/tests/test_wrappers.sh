#!/bin/bash
# CPU-only tests of runner/run_p3b.sh (MOCK sessions against tests/mock_server_p3b.py, no docker, no GPU) and of
# replay/run_replay.sh (fake `docker` / `nvidia-smi` on PATH, scratch ROOT_OVERRIDE).  Nothing here starts a
# container or touches a GPU.  Writes tests/out/wrappers.txt.
# Needs tests/out/fake (tests/make_fake_capture.py) for the copy-out / coverage / replay-input paths.
set -u
T=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$T/.." && pwd)
O=$T/out
LOG=$O/wrappers.txt
mkdir -p "$O"
: > "$LOG"
pass=0; fail=0
note() { echo "$*" | tee -a "$LOG"; }
expect() {  # name want_rc pattern -- command...
  local name=$1 want=$2 pat=$3; shift 3
  local out rc
  out=$("$@" 2>&1); rc=$?
  { echo "----- $name (rc=$rc, want $want)"; echo "$out"; } >> "$LOG"
  if [ "$rc" = "$want" ] && echo "$out" | grep -qE "$pat"; then pass=$((pass + 1)); note "PASS $name"
  else fail=$((fail + 1)); note "FAIL $name (rc=$rc want=$want pattern=$pat)"; echo "$out" | tail -5 | sed 's/^/    /' | tee -a "$LOG"; fi
}
check() {  # name python-condition-script
  local name=$1; shift
  if python3 -c "$1" >> "$LOG" 2>&1; then pass=$((pass + 1)); note "PASS $name"; else fail=$((fail + 1)); note "FAIL $name"; fi
}
[ -d "$O/fake/cap" ] || { echo "run tests/make_fake_capture.py first"; exit 2; }
R=$ROOT/runner/run_p3b.sh
mreset() { rm -rf "$O/mock-runs" "$O/mock-PRIVATE"; rm -f "$O/mock-costs.jsonl"; pkill -f "tests/mock_server_p3b[.]py" 2>/dev/null; sleep 0.3; }
export MOCK_FAKE=$O/fake

# ---------------------------------------------------------------- capture runner: refusals and DRY_RUN
mreset
expect "R1 refuse without ATTEMPT_REASON" 2 "REFUSED: ATTEMPT_REASON" env MOCK=1 "$R"
echo 'not json' > "$O/mock-costs.jsonl"
expect "R2 malformed ledger aborts" 3 "malformed ledger" env MOCK=1 ATTEMPT_REASON=t "$R"
mreset
echo '{"item":"p3b-capture","attempt":1,"seconds":1080,"gpus":2,"gpu_hours":0.6}' > "$O/mock-costs.jsonl"
expect "R3 refuse when budget minus replay reserve < MIN_WALL" 2 "REFUSED: wall cap 0s" env MOCK=1 ATTEMPT_REASON=t "$R"
mreset
expect "R4 MOCK DRY_RUN prints the plan, writes nothing" 0 "DRY_RUN: ATTEMPT_REASON, budget gate" env MOCK=1 DRY_RUN=1 ATTEMPT_REASON=t "$R"
check "R4b DRY_RUN wrote nothing" "import os,sys; sys.exit(0 if not os.path.exists('$O/mock-runs') and not os.path.exists('$O/mock-costs.jsonl') else 1)"
mkdir -p "$O/mock-runs/capture-attempt-1"
expect "R5 refuse an existing attempt directory" 2 "exists \(attempt numbering" env MOCK=1 ATTEMPT_REASON=t "$R"

# ---------------------------------------------------------------- capture runner: MOCK sessions
mreset
expect "R6 MOCK session, quotas met -> client stopped early" 0 "quotas met at|capture attempt 1 occupied" \
  env MOCK=1 ATTEMPT_REASON="mock early stop" MOCK_QUOTA_AFTER=6 MOCK_TOK_S=60 RAMP_S=0.5 "$R"
A=$O/mock-runs/capture-attempt-1
check "R6b early-stop session records (stop record, ramp then 8 in flight, seeds 72301.., cost line, copy-out 0700, coverage)" "
import json, os, stat, sys
cl = [json.loads(l) for l in open('$A/client.jsonl')]
st = json.load(open('$A/mock-stats.json'))
costs = [json.loads(l) for l in open('$O/mock-costs.jsonl')]
sent = [c for c in cl if c['kind'] == 'sent']
ok = any(c['kind'] == 'stop' for c in cl) and st['peak'] == 8 and [c['inflight_at_send'] for c in sent][:2] == [1, 2]
ok &= all(s >= 72301 and s <= 72316 for s in st['seeds']) and len(costs) == 1 and costs[0]['item'] == 'p3b-capture'
ok &= costs[0]['attempt'] == 1 and costs[0]['mock'] is True and costs[0]['attempt_reason'] == 'mock early stop'
po = '$O/mock-PRIVATE/capture-attempt-1'
ok &= stat.S_IMODE(os.stat(po).st_mode) == 0o700 and len([f for _, _, fs in os.walk(po) for f in fs]) == 35
ok &= os.path.exists('$A/coverage.json') and os.path.exists('$A/STOP')
print('R6b', ok, st['peak'], len(sent), costs[0]['seconds']); sys.exit(0 if ok else 1)"
check "R6c no mock server or client process left" "
import subprocess, sys
import time; time.sleep(2)
out = subprocess.run(['pgrep', '-af', 'mock_server_p3b[.]py|client_p3b[.]py'], capture_output=True, text=True).stdout
print('left:', out.strip())
sys.exit(0 if not out.strip() else 1)"
expect "R6d second attempt numbered 2 (budget gate counts the first)" 0 "capture attempt 2:" \
  env MOCK=1 DRY_RUN=1 ATTEMPT_REASON=t "$R"

mreset
expect "R7 MOCK session to natural completion (16 requests, drain)" 0 "capture attempt 1 occupied" \
  env MOCK=1 ATTEMPT_REASON="mock full" MOCK_QUOTA_AFTER=0 MOCK_TOK_S=1000 RAMP_S=0.3 "$R"
check "R7b full session: 16 distinct seeds 72301-72316 served, peak exactly 8, ramp 1,2,4,6,8, long requests first" "
import json, sys
A = '$O/mock-runs/capture-attempt-1'
cl = [json.loads(l) for l in open(A + '/client.jsonl')]
st = json.load(open(A + '/mock-stats.json'))
req = [c for c in cl if c['kind'] == 'request']
sent = [c for c in cl if c['kind'] == 'sent']
ok = sorted(st['seeds']) == list(range(72301, 72317)) and st['peak'] == 8 and len(req) == 16
ok &= all(c['error'] is None for c in req) and [c['kind_req'] for c in sent][:8] == ['long'] * 8
ok &= max(c['inflight_at_send'] for c in sent[:1]) == 1 and any(c['kind'] == 'done' for c in cl)
print('R7b', ok, st['peak'], len(req)); sys.exit(0 if ok else 1)"

mreset
echo '{"item":"p3b-capture","attempt":1,"seconds":1035,"gpus":2,"gpu_hours":0.575,"rc":7}' > "$O/mock-costs.jsonl"
mkdir -p "$O/mock-runs/capture-attempt-1"
expect "R8 watchdog: wall cap reached, server stopped, cost line written" 0 "WALL CAP reached" \
  env MOCK=1 ATTEMPT_REASON="mock watchdog" MIN_WALL=20 RESERVE=4 MOCK_QUOTA_AFTER=0 MOCK_TOK_S=5 RAMP_S=0.2 "$R"
check "R8b watchdog attempt 2: cost line seconds <= cap + 15, no process left" "
import json, subprocess, sys
c = [json.loads(l) for l in open('$O/mock-costs.jsonl')][-1]
import time; time.sleep(2)
out = subprocess.run(['pgrep', '-af', 'mock_server_p3b[.]py|client_p3b[.]py'], capture_output=True, text=True).stdout
print('left:', out.strip())
ok = c['attempt'] == 2 and c['seconds'] <= c['wall_cap'] + 15 and not out.strip()
print('R8b', ok, c['seconds'], c['wall_cap']); sys.exit(0 if ok else 1)"
mreset

# ---------------------------------------------------------------- replay wrapper (fake docker / nvidia-smi)
SCR=$O/scratch-replay
rm -rf "$SCR"; mkdir -p "$SCR/fakebin" "$SCR/root/replay" "$SCR/root/PRIVATE-capture/capture-attempt-1" "$SCR/root/runs/capture-attempt-1/caplog"
cp "$ROOT"/replay/*.py "$ROOT"/replay/*.sh "$SCR/root/replay/"
cp -r "$O/fake/cap/." "$SCR/root/PRIVATE-capture/capture-attempt-1/"
cp -r "$O/fake/caplog/." "$SCR/root/runs/capture-attempt-1/caplog/"
cat > "$SCR/fakebin/docker" <<'EOF'
#!/bin/bash
echo "docker $*" >> "$FAKE_LOG"
case "$1" in
  image) exit 0 ;;
  ps) exit 0 ;;
  rm) exit 0 ;;
  run) echo "RUNARGS $*" >> "$FAKE_LOG"; [ -n "${FAKE_RUN_SLEEP:-}" ] && sleep "$FAKE_RUN_SLEEP"; exit 0 ;;
esac
exit 0
EOF
cat > "$SCR/fakebin/nvidia-smi" <<'EOF'
#!/bin/bash
case "$*" in
  *noheader*) echo "${FAKE_MEM:-3}, ${FAKE_UTIL:-0}" ;;
  *query-compute-apps*) echo "gpu_uuid, pid, process_name, used_memory" ;;
  *) echo "index, name, memory.used [MiB], utilization.gpu [%]"; echo "0, FAKE GPU, 3 MiB, 0 %" ;;
esac
EOF
chmod +x "$SCR/fakebin/"*
RR=$SCR/root/replay/run_replay.sh
FENV=(env PATH="$SCR/fakebin:$PATH" FAKE_LOG="$SCR/fake.log" ROOT_OVERRIDE="$SCR/root")
expect "Q1 replay: refuse without ATTEMPT_REASON" 2 "REFUSED: ATTEMPT_REASON" "${FENV[@]}" CAPTURE_ATTEMPT=1 "$RR"
expect "Q2 replay: refuse without CAPTURE_ATTEMPT" 2 "CAPTURE_ATTEMPT=N" "${FENV[@]}" ATTEMPT_REASON=t "$RR"
expect "Q3 replay: abort when the capture attempt has no tensors" 3 "no captured tensors" "${FENV[@]}" ATTEMPT_REASON=t CAPTURE_ATTEMPT=7 "$RR"
echo '{"item":"p3b-capture","attempt":1,"seconds":1150,"gpus":2}' > "$SCR/root/costs.jsonl"
expect "Q4 replay: refuse when < 120 s of budget remain" 2 "REFUSED: wall cap" "${FENV[@]}" ATTEMPT_REASON=t CAPTURE_ATTEMPT=1 "$RR"
rm -f "$SCR/root/costs.jsonl"
expect "Q5 replay: GPU0 busy aborts" 3 "GPU0 memory" "${FENV[@]}" FAKE_MEM=5000 ATTEMPT_REASON=t CAPTURE_ATTEMPT=1 DRY_RUN=1 "$RR"
expect "Q6 replay DRY_RUN prints the plan, starts nothing" 0 "DRY_RUN: all gates passed" "${FENV[@]}" ATTEMPT_REASON=t CAPTURE_ATTEMPT=1 DRY_RUN=1 "$RR"
check "Q6b replay DRY_RUN wrote nothing" "import os,sys; sys.exit(0 if not os.path.exists('$SCR/root/runs/replay-attempt-1') and not os.path.exists('$SCR/root/costs.jsonl') else 1)"
: > "$SCR/fake.log"
echo '{"item":"p3b-capture","attempt":1,"seconds":900,"gpus":2}' > "$SCR/root/costs.jsonl"
expect "Q7 replay fake run: both images, cost lines, summary" 0 "budget used so far" "${FENV[@]}" ATTEMPT_REASON="fake run" CAPTURE_ATTEMPT=1 "$RR"
check "Q7b docker run argv: --network none, --gpus device=0, --user, ro mounts, served env, digests, parts; 2 cost lines" "
import json, sys
runs = [l for l in open('$SCR/fake.log') if l.startswith('RUNARGS')]
costs = [json.loads(l) for l in open('$SCR/root/costs.jsonl')][1:]
need = ['--network none', '--gpus device=0', '--user ', ':/cap:ro', ':/caplog:ro', ':/p3h:ro', ':/model-repo:ro', ':/r:ro',
        'VLLM_EXL3_TRELLIS_MAX_M=32', 'B12X_EXL3_BF16_GEMV=1', 'B12X_COMPILE_CACHE_DIR=/cc', '--entrypoint sh']
ok = len(runs) == 2 and all(all(n in r for n in need) for r in runs)
ok &= 'a9e985c0' in runs[0] and 'attn,moe,forced' in runs[0] and '48e254d9' in runs[1] and ' moe' in runs[1]
ok &= [c['engine'] for c in costs] == ['v091', 'v070'] and all(c['item'] == 'p3b-replay' and c['capture_attempt'] == 1 for c in costs)
ok &= '--gpus' not in ''.join(l for l in open('$SCR/fake.log') if 'summarize' in l)
import os
ok &= os.path.exists('$SCR/root/runs/replay-attempt-1/summary.md')
ok &= oct(os.stat('$SCR/root/PRIVATE-capture/replay-attempt-1').st_mode)[-3:] == '700'
print('Q7b', ok, len(runs), [c['engine'] for c in costs]); sys.exit(0 if ok else 1)"
printf '%s\n' '{"item":"p3b-capture","attempt":1,"seconds":1105,"gpus":2}' '{"item":"p3b-replay","attempt":1,"seconds":0,"gpus":0,"engine":"v091"}' > "$SCR/root/costs.jsonl"
expect "Q8 replay with 130 s left: v091 runs, v070 skipped with a cost line" 0 "SKIP v070" \
  "${FENV[@]}" FAKE_RUN_SLEEP=4 ATTEMPT_REASON="small budget" CAPTURE_ATTEMPT=1 "$RR"
check "Q8b skipped v070 recorded (gpus 0, skipped true), attempt 2" "
import json, sys
c = [json.loads(l) for l in open('$SCR/root/costs.jsonl')][2:]
ok = c[0]['engine'] == 'v091' and c[0]['attempt'] == 2 and c[1]['engine'] == 'v070' and c[1]['skipped'] is True and c[1]['gpus'] == 0
print('Q8b', ok, c); sys.exit(0 if ok else 1)"

note "wrapper tests: $pass passed, $fail failed"
[ "$fail" = 0 ]
