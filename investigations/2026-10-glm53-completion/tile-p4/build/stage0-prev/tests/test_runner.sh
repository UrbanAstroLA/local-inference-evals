#!/bin/bash
# CPU tests of the P4 GPU wrappers (no GPU, no real docker container, no real ledger):
#   R*  runner/run_p4.sh refusals, budget/order gate, DRY_RUN writes nothing, MOCK sessions A -> B -> A2 against
#       tests/mock_server_p4.py (start-up gates from fabricated bind logs, client, flush wait, copy-out at 0700,
#       cost line, outcome), a mock start-up gate failure (INCOMPLETE, no traffic), order and re-run refusals
#   Q*  client deadline handling (skips a batch it cannot finish)
#   W*  precheck/run_precheck.sh and g4/run_g4.sh with FAKE=1: fake docker / nvidia-smi on PATH (argv recorded:
#       --network none, --gpus device=0, read-only mounts), budget order (g4 needs a passed precheck), verdicts, cost
#       lines, DRY_RUN
# Everything is written under tests/out/runner-scratch.  No `set -e`.
T=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$T/.." && pwd)
S=$T/out/runner-scratch
rm -rf "$S"; mkdir -p "$S"
NP=0; NF=0
ok()  { echo "PASS $*"; NP=$((NP + 1)); }
bad() { echo "FAIL $*"; NF=$((NF + 1)); }
expect() { if [ "$1" = 0 ]; then ok "$2"; else bad "$2 ${3:-}"; fi; }
ACK=$(sha256sum "$ROOT/PROPOSAL-AMENDMENTS.md" | cut -c1-12)
R=$ROOT/runner/run_p4.sh
MR=$T/out/runner-scratch/mock-root
export MOCK_ROOT=$MR MOCK_PORT=18041

# ---- refusals
out=$(SESSION=A MOCK=1 P4_AMENDMENTS_ACK=$ACK $R 2>&1); [[ "$out" == REFUSED*ATTEMPT_REASON* ]]; expect $? "R1 refuses without ATTEMPT_REASON"
out=$(SESSION=C MOCK=1 ATTEMPT_REASON=t P4_AMENDMENTS_ACK=$ACK $R 2>&1); [[ "$out" == REFUSED*SESSION* ]]; expect $? "R2 refuses an unknown session"
out=$(SESSION=A MOCK=1 ATTEMPT_REASON=t P4_AMENDMENTS_ACK=000000000000 $R 2>&1); [[ "$out" == REFUSED*AMENDMENTS* ]]; expect $? "R3 refuses a wrong amendments acknowledgement"
python3 "$T/make_mock_root.py" "$MR" > /dev/null
out=$(SESSION=A MOCK=1 ATTEMPT_REASON=t P4_AMENDMENTS_ACK=$ACK $R 2>&1); [[ "$out" == *"need a passed precheck"* ]]; expect $? "R4 session A refused before a passed precheck and G4"
python3 "$T/make_mock_root.py" "$MR" --checks-passed > /dev/null
out=$(SESSION=B MOCK=1 ATTEMPT_REASON=t P4_AMENDMENTS_ACK=$ACK $R 2>&1); [[ "$out" == *"needs session A done first"* ]]; expect $? "R5 session B refused before A"
BEFORE=$(find "$MR" | sort | sha256sum)
out=$(SESSION=A MOCK=1 DRY_RUN=1 ATTEMPT_REASON=t P4_AMENDMENTS_ACK=$ACK $R 2>&1)
AFTER=$(find "$MR" | sort | sha256sum)
[[ "$out" == *"DRY_RUN"* && "$out" == *total_est_s* && "$BEFORE" == "$AFTER" ]]; expect $? "R6 DRY_RUN passes the gates, prints the plan and writes nothing"
echo "$out" | grep -E "budget:|plan:" | sed 's/^/    /'
out=$(SESSION=A MOCK=1 ATTEMPT_REASON=t P4_AMENDMENTS_ACK=$ACK MOCK_ROOT=/tmp/elsewhere $R 2>&1); [[ "$out" == *"outside tests/out"* ]]; expect $? "R7 MOCK refuses a mock root outside tests/out"

# ---- MOCK sessions
run_mock() {  # session tile_env bad_gate
  MOCK_TILE=$2 MOCK_BAD_GATE=$3 SESSION=$1 MOCK=1 ATTEMPT_REASON="mock $1" P4_AMENDMENTS_ACK=$ACK timeout 300 $R > "$S/mock-$1-$4.log" 2>&1
  echo $?
}
rc=$(run_mock A "" 0 1)
AO=$MR/runs/session-A-attempt-1; PO=$MR/PRIVATE-runs/session-A-attempt-1
V=$(python3 -c "import json;print(json.load(open('$AO/outcome.json'))['verdict'])" 2>/dev/null)
[ "$V" = COMPLETE ]; expect $? "R8 MOCK session A: COMPLETE (G3 vs captured-p3b, 96 forced requests, flushed)" "verdict $V rc $rc"
[ "$(stat -c %a "$PO")" = 700 ] && [ "$(find "$PO" -type f | wc -l)" -ge 96 ]; expect $? "R8 private copy-out at 0700 ($(find "$PO" -type f 2>/dev/null | wc -l) files)"
python3 -c "
import json
L=[json.loads(l) for l in open('$MR/costs.jsonl')]
s=[l for l in L if l['item']=='p4-A']
assert len(s)==1 and s[0]['mock'] and s[0]['gpus']==2 and s[0]['seconds']>0 and s[0]['tile_env'] is None
" ; expect $? "R8 exactly one cost line for session A (2 GPUs, no tile env)"
grep -q "GATES PASS A startup" "$AO/gate.txt"; expect $? "R8 start-up gates evaluated before traffic"
python3 -c "
import json
first_forced=min(json.loads(l)['t'] for l in open('$AO/client.jsonl') if json.loads(l).get('kind')=='forced')
import os; assert os.path.getmtime('$AO/gates-startup.json') < first_forced
"; expect $? "R8 gates-startup.json written before the first forced request"
rc=$(run_mock B "" 1 bad)
AO=$MR/runs/session-B-attempt-1
V=$(python3 -c "import json;print(json.load(open('$AO/outcome.json'))['verdict'])" 2>/dev/null)
[ "$V" = INCOMPLETE ] && [ ! -s "$AO/client.jsonl" ] && grep -q "GATES FAIL B startup" "$AO/gate.txt"; expect $? "R9 a mock start-up G1 failure stops session B before traffic (INCOMPLETE)" "verdict $V"
python3 -c "
import json
s=[json.loads(l) for l in open('$MR/costs.jsonl') if json.loads(l)['item']=='p4-B']
assert len(s)==1 and s[0]['rc']==8 and s[0]['tile_env']=='64,256,64,256'
"; expect $? "R9 the failed attempt still has its cost line (rc 8, tile env recorded)"
rc=$(run_mock B "64,256,64,256" 0 2)
AO=$MR/runs/session-B-attempt-2
V=$(python3 -c "import json;print(json.load(open('$AO/outcome.json'))['verdict'])" 2>/dev/null)
[ "$V" = COMPLETE ] && grep -q "GATES PASS B final" "$AO/gate.txt"; expect $? "R10 MOCK session B attempt 2: COMPLETE with G1 + G2 vs session A" "verdict $V"
out=$(SESSION=B MOCK=1 ATTEMPT_REASON=t P4_AMENDMENTS_ACK=$ACK $R 2>&1); [[ "$out" == *"already done"* ]]; expect $? "R11 a done session is never re-run"
rc=$(run_mock A2 "" 0 1)
V=$(python3 -c "import json;print(json.load(open('$MR/runs/session-A2-attempt-1/outcome.json'))['verdict'])" 2>/dev/null)
[ "$V" = COMPLETE ]; expect $? "R12 MOCK session A2: COMPLETE with the session control vs A" "verdict $V"
pgrep -f "mock_server_p4[.]py" > /dev/null; [ $? != 0 ]; expect $? "R12 no mock server left running"
# budget: a nearly spent ledger refuses a session
MR2=$S/mock-root2; python3 "$T/make_mock_root.py" "$MR2" --checks-passed --ledger-gh 6.2 > /dev/null   # ceiling 7.0 (operator raise, AUTHORIZATION.txt line 2)
out=$(MOCK_ROOT=$MR2 SESSION=A MOCK=1 ATTEMPT_REASON=t P4_AMENDMENTS_ACK=$ACK $R 2>&1); [[ "$out" == *"wall cap"*"< 1300s"* ]]; expect $? "R13 budget gate refuses a session that cannot fit (6.2 of 7.0 GPU-h used + reserve)"
out=$(python3 - "$ROOT" <<'PY'
import json, os, subprocess, sys, tempfile
root = sys.argv[1]
d = tempfile.mkdtemp(dir=os.path.join(root, "tests", "out", "runner-scratch"))
os.makedirs(os.path.join(d, "runs", "precheck-attempt-1")); os.makedirs(os.path.join(d, "runs", "g4-attempt-1"))
json.dump({"verdict": "PASS"}, open(os.path.join(d, "runs", "precheck-attempt-1", "verdict.json"), "w"))
json.dump({"G4": "PASS"}, open(os.path.join(d, "runs", "g4-attempt-1", "verdict.json"), "w"))
def cap(step, used):
    open(os.path.join(d, "costs.jsonl"), "w").write(json.dumps(dict(item="x", attempt=1, seconds=int(used * 1800), gpus=2)) + "\n")
    return subprocess.run([sys.executable, os.path.join(root, "runner", "budget_p4.py"), d, step], capture_output=True, text=True).stdout.split()
a = cap("A", 0.06)          # fresh: share cap
c = cap("A", 1.2)           # ceiling 7.0: one failed A attempt of 1.14 GPU-h still leaves room for a retry -> share cap
b = cap("A", 4.2)           # ceiling 7.0: 4.2 used, reserve 2.54 for B and A2 -> 0.26 left -> refused
ok = a[0] == "OK" and int(a[3]) == 2350 and c[0] == "OK" and int(c[3]) == 2350 and b[0] == "REFUSE"
print(("OK" if ok else "BAD"), a, b[:3])
PY
)
[[ "$out" == OK* ]]; expect $? "R14 budget arithmetic (ceiling 7.0): fresh A cap 2350 s; after a 1.14 GPU-h failure a retry still gets 2350 s; at 4.2 used A is refused (reserve for B and A2)" "$out"

# ---- Q1 client deadline: a batch that cannot finish is skipped and recorded
python3 "$T/mock_server_p4.py" "$S/q1-ctr" 18042 A > /dev/null 2>&1 &
MP=$!; sleep 1
mkdir -p "$S/q1"
P4_WINDOWS=$MR/PRIVATE-fixtures/windows.json P4_SCHED=$MR/PRIVATE-fixtures/sched.json \
  python3 "$ROOT/runner/client_p4.py" "$S/q1" $(( $(date +%s) + 3 )) http://127.0.0.1:18042 A > /dev/null 2>&1
kill $MP 2>/dev/null
python3 -c "
import json
L=[json.loads(l) for l in open('$S/q1/client.jsonl')]
sk=[l for l in L if l['kind']=='skip']
assert sk and sk[0]['reason']=='time' and any(l['kind']=='end' for l in L)
"; expect $? "Q1 client skips the batches it cannot finish before the deadline and ends cleanly"

# ---- W: precheck and G4 wrappers with fake docker / nvidia-smi
FB=$S/fakebin; mkdir -p "$FB"
cat > "$FB/docker" <<'EOF'
#!/bin/bash
echo "docker $*" >> "$FAKE_LOG"
case "$1" in
  image) awk -v t="$3" '$1==t {print $2}' "$FAKE_IMAGES" ;;
  ps) ;;
  rm) ;;
  run)
    O=""; for a in "$@"; do case "$a" in *:/o) O=${a%:/o};; esac; done
    case "$*" in
      *precheck_graph.py*) echo '{"verdict": "PASS", "results": []}' > "$O/precheck.json"; echo "PASS C1"; echo "VERDICT PASS" ;;
      *in_container_g4.sh*) python3 - "$O" <<'PY'
import json, sys
o = sys.argv[1]
ent = lambda m: dict(m=m, phase="traffic", baseline=dict(bitequal_all_rows=True, repeat_bitequal=True, apply_bitequal_served=True),
                     between=dict(n_rows_differ=0))
lay = {L: dict(entries=[ent(8), ent(32)], launch_map={"decode:m16": dict(fc1_tile_k=64, fc1_tile_n=256, fc2_tile_k=64, fc2_tile_n=256)})
       for L in ("3", "23", "44")}
json.dump(dict(status="complete", layers=lay), open(o + "/moe-g4a.json", "w"))
json.dump(dict(status="complete", layers=lay), open(o + "/moe-g4b.json", "w"))
PY
      echo "G4a rc=0" ;;
    esac ;;
esac
exit 0
EOF
cat > "$FB/nvidia-smi" <<'EOF'
#!/bin/bash
case "$*" in *noheader*) echo "3, 0";; *) echo "index, name, memory.used [MiB], utilization.gpu [%]"; echo "0, RTX, 3 MiB, 0 %";; esac
EOF
chmod +x "$FB/docker" "$FB/nvidia-smi"
WR=$S/wrap-root; mkdir -p "$WR/runs"
export FAKE_LOG=$S/fake-docker.log FAKE_IMAGES=$ROOT/runner/images.txt
out=$(PATH=$FB:$PATH FAKE=1 ROOT_OVERRIDE=$WR ATTEMPT_REASON=t "$ROOT/g4/run_g4.sh" 2>&1); [[ "$out" == *"needs a passed precheck"* ]]; expect $? "W1 G4 refused before a passed precheck"
out=$(PATH=$FB:$PATH FAKE=1 ROOT_OVERRIDE=$WR DRY_RUN=1 ATTEMPT_REASON=t "$ROOT/precheck/run_precheck.sh" 2>&1); [[ "$out" == *DRY_RUN* && ! -e "$WR/runs/precheck-attempt-1" ]]; expect $? "W2 precheck DRY_RUN writes nothing"
out=$(PATH=$FB:$PATH FAKE=1 ROOT_OVERRIDE=$WR ATTEMPT_REASON=t "$ROOT/precheck/run_precheck.sh" 2>&1)
[[ "$out" == *"precheck verdict PASS"* ]] && grep -q -- "--network none --gpus device=0" "$FAKE_LOG" && grep -q ":/pc:ro" "$FAKE_LOG"; expect $? "W3 precheck (fake docker): --network none, --gpus device=0, read-only code mount, verdict PASS"
out=$(PATH=$FB:$PATH FAKE=1 ROOT_OVERRIDE=$WR ATTEMPT_REASON=t "$ROOT/precheck/run_precheck.sh" 2>&1); [[ "$out" == *"already passed"* ]]; expect $? "W4 a passed precheck is not re-run"
out=$(PATH=$FB:$PATH FAKE=1 ROOT_OVERRIDE=$WR ATTEMPT_REASON=t "$ROOT/g4/run_g4.sh" 2>&1)
[[ "$out" == *"G4 PASS"* ]] && grep -q ":/cap:ro" "$FAKE_LOG" && grep -q ":/r:ro" "$FAKE_LOG"; expect $? "W5 G4 (fake docker): captures and replay code mounted read-only; verdict PASS" "$(echo "$out" | tail -2)"
python3 -c "
import json
L=[json.loads(l) for l in open('$WR/costs.jsonl')]
assert [l['item'] for l in L]==['p4-precheck','p4-g4'] and all(l['gpus']==1 for l in L)
"; expect $? "W6 one 1-GPU cost line each for precheck and G4"
out=$(PATH=$FB:$PATH ATTEMPT_REASON=t FAKE=1 "$ROOT/g4/run_g4.sh" 2>&1); [[ "$out" == *"ROOT_OVERRIDE under tests/out"* ]]; expect $? "W7 FAKE mode refuses the real package root"
echo "test_runner: $NP PASS, $NF FAIL"
[ "$NF" = 0 ]
