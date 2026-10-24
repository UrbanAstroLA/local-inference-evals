#!/bin/bash
# CPU tests of the P4 GPU wrappers (no GPU, no real docker container, no real ledger):
#   2026-10-24: the run-1 ceiling is VOID; the V* tests check that every run-1 GPU step is refused (session runner,
#   pre-check and G4 wrappers, mock or real root).  The older R4-R14 / W1-W7 mock-session tests are kept in
#   build/stage0-prev/tests/test_runner.sh (they need a budget gate that can say OK).
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
S=${OUT_OVERRIDE:-$T/out}/runner-scratch
rm -rf "$S"; mkdir -p "$S"
NP=0; NF=0
ok()  { echo "PASS $*"; NP=$((NP + 1)); }
bad() { echo "FAIL $*"; NF=$((NF + 1)); }
expect() { if [ "$1" = 0 ]; then ok "$2"; else bad "$2 ${3:-}"; fi; }
ACK=$(sha256sum "$ROOT/PROPOSAL-AMENDMENTS.md" | cut -c1-12)
R=$ROOT/runner/run_p4.sh
MR=$S/mock-root
export MOCK_ROOT=$MR MOCK_PORT=18041

# ---- refusals
out=$(SESSION=A MOCK=1 P4_AMENDMENTS_ACK=$ACK $R 2>&1); [[ "$out" == REFUSED*ATTEMPT_REASON* ]]; expect $? "R1 refuses without ATTEMPT_REASON"
out=$(SESSION=C MOCK=1 ATTEMPT_REASON=t P4_AMENDMENTS_ACK=$ACK $R 2>&1); [[ "$out" == REFUSED*SESSION* ]]; expect $? "R2 refuses an unknown session"
out=$(SESSION=A MOCK=1 ATTEMPT_REASON=t P4_AMENDMENTS_ACK=000000000000 $R 2>&1); [[ "$out" == REFUSED*AMENDMENTS* ]]; expect $? "R3 refuses a wrong amendments acknowledgement"
python3 "$T/make_mock_root.py" "$MR" --checks-passed > /dev/null
# ---- VOID (2026-10-24): the run-1 ceiling is void; budget_p4.py refuses every step, mock or real, with no override.
#      (The pre-void mock-session tests R4-R14 / W1-W7 are kept in build/stage0-prev/tests/test_runner.sh.)
LEDGER_BEFORE=$(sha256sum "$ROOT/costs.jsonl" | cut -d' ' -f1); MLEDGER_BEFORE=$(sha256sum "$MR/costs.jsonl" | cut -d' ' -f1)
RUNS_BEFORE=$(ls "$ROOT/runs" | sort | sha256sum)
for SS in A B A2; do
  out=$(SESSION=$SS MOCK=1 ATTEMPT_REASON=t P4_AMENDMENTS_ACK=$ACK $R 2>&1); [[ "$out" == REFUSED*VOID* ]]; expect $? "V1 MOCK session $SS refused: run-1 ceiling VOID" "$out"
  out=$(SESSION=$SS DRY_RUN=1 ATTEMPT_REASON=t P4_AMENDMENTS_ACK=$ACK $R 2>&1); [[ "$out" == REFUSED*VOID* ]]; expect $? "V2 real-root DRY_RUN session $SS refused: run-1 ceiling VOID" "$out"
done
for st in precheck g4 A B A2; do
  out=$(python3 "$ROOT/runner/budget_p4.py" "$MR" $st); [[ "$out" == REFUSE*VOID* ]]; expect $? "V3 budget_p4.py refuses step $st (mock root with passed checks)" "$out"
done
[ ! -e "$MR/runs/session-A-attempt-1" ] && [ "$(sha256sum "$MR/costs.jsonl" | cut -d' ' -f1)" = "$MLEDGER_BEFORE" ]; expect $? "V4 a refused session writes no attempt dir and no cost line"
FB=$S/fakebin; mkdir -p "$FB"
printf '#!/bin/bash\necho "docker $*" >> "$FAKE_LOG"\nexit 0\n' > "$FB/docker"; printf '#!/bin/bash\necho "0, 3, 0"\n' > "$FB/nvidia-smi"; chmod +x "$FB/docker" "$FB/nvidia-smi"
WR=$S/wrap-root; mkdir -p "$WR/runs"; export FAKE_LOG=$S/fake-docker.log
out=$(PATH=$FB:$PATH FAKE=1 ROOT_OVERRIDE=$WR ATTEMPT_REASON=t "$ROOT/precheck/run_precheck.sh" 2>&1); [[ "$out" == REFUSED*VOID* ]]; expect $? "V5 run-1 precheck wrapper refused (VOID)" "$out"
out=$(PATH=$FB:$PATH FAKE=1 ROOT_OVERRIDE=$WR ATTEMPT_REASON=t "$ROOT/g4/run_g4.sh" 2>&1); [[ "$out" == REFUSED*VOID* ]]; expect $? "V6 run-1 G4 wrapper refused (VOID)" "$out"
[ ! -s "$FAKE_LOG" ] && [ ! -e "$WR/costs.jsonl" ]; expect $? "V7 refused wrappers never called docker and wrote no cost line"
[ "$(sha256sum "$ROOT/costs.jsonl" | cut -d' ' -f1)" = "$LEDGER_BEFORE" ] && [ "$(ls "$ROOT/runs" | sort | sha256sum)" = "$RUNS_BEFORE" ]; expect $? "V8 the run-1 ledger and runs/ are unchanged"

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

echo "test_runner: $NP PASS, $NF FAIL"
[ "$NF" = 0 ]
