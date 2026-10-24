#!/bin/bash
# CPU tests of the stage-0 GPU wrappers (no GPU, no real serving container, never the real ledgers):
#   S*  runner/stage0/run_stage0.sh: refusals (ATTEMPT_REASON, APPROVED_GPU_H unset / empty / non-numeric / <= 0,
#       protocol ack, mock root outside tests/out), budget arithmetic (cap + used > approved refused; never shrunk),
#       the new ledger only (a run-1 ledger is neither read nor written), order (needs a passed stage-0 pre-check),
#       DRY_RUN writes nothing, a MOCK session end to end against tests/mock_server_stage0.py (G-instr + G3 before
#       traffic, V1 one request at a time, V2 as ONE multi-prompt request, flush wait, 0700 private copy-out, one cost
#       line in costs-stage0.jsonl, outcome, the stage-0 analysis in a CPU container), and no re-run after a
#       client-complete attempt
#   P*  runner/stage0/run_precheck_s0.sh with FAKE=1 (fake docker / nvidia-smi): refusals, cost line, mounts, verdict
# Everything is written under ${OUT_OVERRIDE:-tests/out}/stage0-runner-scratch (inside tests/out).  No `set -e`.
T=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$T/.." && pwd)
S=${OUT_OVERRIDE:-$T/out}/stage0-runner-scratch
case "$S" in "$ROOT"/tests/out/*) ;; *) echo "scratch must be under tests/out"; exit 2;; esac
rm -rf "$S"; mkdir -p "$S"
NP=0; NF=0
ok()  { echo "PASS $*"; NP=$((NP + 1)); }
bad() { echo "FAIL $*"; NF=$((NF + 1)); }
expect() { if [ "$1" = 0 ]; then ok "$2"; else bad "$2 ${3:-}"; fi; }
ACK=$(sha256sum "$ROOT/STAGE0-PROTOCOL.md" | cut -c1-12)
R=$ROOT/runner/stage0/run_stage0.sh
B=$ROOT/runner/stage0/budget_stage0.py
MR=$S/mock-root
export MOCK_ROOT=$MR MOCK_PORT=18051
python3 "$T/make_mock_root.py" "$MR" > /dev/null
REAL_LEDGER=$(sha256sum "$ROOT/costs.jsonl" | cut -d' ' -f1)
REAL_S0=$( [ -e "$ROOT/costs-stage0.jsonl" ] && sha256sum "$ROOT/costs-stage0.jsonl" | cut -d' ' -f1 || echo none)

# ---- refusals
out=$(MOCK=1 APPROVED_GPU_H=2 S0_PROTOCOL_ACK=$ACK $R 2>&1); [[ "$out" == REFUSED*ATTEMPT_REASON* ]]; expect $? "S1 refuses without ATTEMPT_REASON"
out=$(env -u APPROVED_GPU_H MOCK=1 ATTEMPT_REASON=t S0_PROTOCOL_ACK=$ACK $R 2>&1); [[ "$out" == REFUSED*APPROVED_GPU_H* ]]; expect $? "S2 refuses with APPROVED_GPU_H unset (no default, no 7.0 fallback)" "$out"
out=$(MOCK=1 APPROVED_GPU_H= ATTEMPT_REASON=t S0_PROTOCOL_ACK=$ACK $R 2>&1); [[ "$out" == REFUSED*APPROVED_GPU_H* ]]; expect $? "S2 refuses with APPROVED_GPU_H empty"
for v in abc 0 -1 nan inf; do
  out=$(APPROVED_GPU_H=$v python3 "$B" "$MR" precheck); [[ "$out" == REFUSE*APPROVED_GPU_H* ]]; expect $? "S2 budget gate refuses APPROVED_GPU_H=$v" "$out"
done
out=$(MOCK=1 APPROVED_GPU_H=2 ATTEMPT_REASON=t S0_PROTOCOL_ACK=000000000000 $R 2>&1); [[ "$out" == REFUSED*S0_PROTOCOL_ACK* ]]; expect $? "S3 refuses a wrong protocol acknowledgement"
out=$(MOCK=1 APPROVED_GPU_H=2 ATTEMPT_REASON=t S0_PROTOCOL_ACK=$ACK MOCK_ROOT=/tmp/elsewhere $R 2>&1); [[ "$out" == *"outside tests/out"* ]]; expect $? "S3 MOCK refuses a mock root outside tests/out"
out=$(MOCK=1 APPROVED_GPU_H=2 ATTEMPT_REASON=t S0_PROTOCOL_ACK=$ACK $R 2>&1); [[ "$out" == *"needs a passed stage-0 pre-check"* ]]; expect $? "S4 stage 0 refused before a passed stage-0 pre-check" "$out"

# ---- budget arithmetic on the new ledger only
mkdir -p "$MR/runs-stage0/precheck-attempt-1"; echo '{"verdict": "PASS"}' > "$MR/runs-stage0/precheck-attempt-1/verdict.json"
echo '{"item": "p4-A", "attempt": 1, "seconds": 12000, "gpus": 2}' > "$MR/costs.jsonl"         # a run-1 ledger at 6.7 GPU-h
L1=$(sha256sum "$MR/costs.jsonl" | cut -d' ' -f1)
out=$(APPROVED_GPU_H=1.2 python3 "$B" "$MR" stage0); [[ "$out" == REFUSE*"> approved 1.2000"* ]]; expect $? "S5 cap 1.25 GPU-h > approved 1.2: refused (cap never shrunk)" "$out"
echo '{"item": "s0-precheck", "attempt": 1, "seconds": 360, "gpus": 1}' > "$MR/costs-stage0.jsonl"   # 0.1 GPU-h used
out=$(APPROVED_GPU_H=1.34 python3 "$B" "$MR" stage0); [[ "$out" == REFUSE*"used 0.1000"* ]]; expect $? "S5 used 0.10 + cap 1.25 > approved 1.34: refused" "$out"
echo '{"item": "s0-precheck", "attempt": 1, "seconds": 288, "gpus": 1}' > "$MR/costs-stage0.jsonl"   # 0.08 GPU-h used
out=$(APPROVED_GPU_H=1.34 python3 "$B" "$MR" stage0); set -- $out
[ "$1" = OK ] && [ "$4" = 2250 ] && [ "$6" = 2 ]; expect $? "S5 used 0.08 + cap 1.25 <= approved 1.34: OK, cap 2250 s on 2 GPUs; the run-1 ledger (6.7 GPU-h) is not read" "$out"
BEFORE=$(find "$MR" | sort | sha256sum)
out=$(MOCK=1 DRY_RUN=1 APPROVED_GPU_H=1.34 ATTEMPT_REASON=t S0_PROTOCOL_ACK=$ACK $R 2>&1)
AFTER=$(find "$MR" | sort | sha256sum)
[[ "$out" == *DRY_RUN* && "$out" == *prompt_keys_unique\":\ true* && "$BEFORE" == "$AFTER" ]]; expect $? "S6 DRY_RUN passes the gates, prints the plan and writes nothing" "$(echo "$out" | tail -2)"

# ---- MOCK session
rm -f "$MR/costs-stage0.jsonl"
timeout 600 env MOCK=1 APPROVED_GPU_H=1.34 ATTEMPT_REASON="mock stage 0" S0_PROTOCOL_ACK=$ACK $R > "$S/mock-1.log" 2>&1
RC=$?
AO=$MR/runs-stage0/stage0-attempt-1; PO=$MR/PRIVATE-runs-stage0/stage0-attempt-1
V=$(python3 -c "import json;print(json.load(open('$AO/outcome.json'))['verdict'])" 2>/dev/null)
[ "$V" = COMPLETE ]; expect $? "S7 MOCK stage 0: COMPLETE (G-instr + G3, 4 replicates, 32 forced ok, flushed, entry both ranks)" "verdict $V rc $RC $(tail -3 "$S/mock-1.log")"
grep -q "G-instr: OK" "$AO/gate.txt" && grep -q "GATES PASS A startup" "$AO/gate.txt"; expect $? "S7 G-instr and G3 evaluated before traffic"
python3 -c "
import json, os
first = min(json.loads(l)['t'] for l in open('$AO/client.jsonl') if json.loads(l).get('kind') in ('warm', 'forced'))
assert os.path.getmtime('$AO/gates-startup.json') < first
"; expect $? "S7 gates-startup.json written before the first request"
python3 -c "
import json
L = [json.loads(l) for l in open('$AO/client.jsonl')]
m = [d for d in L if d['kind'] == 'multi']
v1 = [d for d in L if d['kind'] == 'forced' and d['variant'] == 'V1']
assert len(m) == 2 and all(d['n_prompts'] == 8 and d['ok'] == 8 for d in m), m
assert len(v1) == 16 and all(d['ok'] for d in v1)
salts = {d['replicate']: d['salt'] for d in L if d['kind'] == 'warm'}
assert sorted(set(salts.values())) == ['s0-V1-r1', 's0-V1-r2', 's0-V2-r1', 's0-V2-r2'], salts
seeds = {d['seed'] for d in L if d['kind'] == 'forced'}
assert seeds == {73000}, seeds
t = [(d['t_send'], d['t_done']) for d in v1]
assert all(t[i][1] <= t[i + 1][0] for i in range(len(t) - 1)), 'V1 requests overlap'
"; expect $? "S7 V2 = ONE multi-prompt request per replicate (8 prompts, 8 ok); V1 one request at a time; a salt per replicate; one seed for every forced request"
L=$(python3 -c "
import json
L=[json.loads(l) for l in open('$MR/costs-stage0.jsonl')]
assert len(L) == 1 and L[0]['item'] == 's0-stage0' and L[0]['gpus'] == 2 and L[0]['seconds'] > 0 and L[0]['approved_gpu_h'] == 1.34 and L[0]['mock']
print('ok')" 2>&1)
[ "$L" = ok ] && [ "$(sha256sum "$MR/costs.jsonl" | cut -d' ' -f1)" = "$L1" ]; expect $? "S7 one cost line in costs-stage0.jsonl (2 GPUs, approval recorded); the run-1 ledger unchanged" "$L"
[ "$(stat -c %a "$PO")" = 700 ] && [ "$(find "$PO" -type f | wc -l)" -ge 32 ]; expect $? "S7 private copy-out at 0700 ($(find "$PO" -type f 2>/dev/null | wc -l) files)"
[ -s "$AO/stage0-result.json" ] && grep -q "STAGE0:" "$AO/analysis-console.txt"; expect $? "S7 the stage-0 analysis ran in a CPU container after the GPUs were released" "$(tail -2 "$AO/analysis-console.txt" 2>/dev/null)"
out=$(MOCK=1 APPROVED_GPU_H=5 ATTEMPT_REASON=t S0_PROTOCOL_ACK=$ACK $R 2>&1); [[ "$out" == *"already completed its client"* ]]; expect $? "S8 no further stage-0 attempt after a client-complete one"
for i in 1 2 3 4 5; do pgrep -f "mock_server_stage0[.]py" > /dev/null || break; sleep 1; done
pgrep -f "mock_server_stage0[.]py" > /dev/null; [ $? != 0 ]; expect $? "S8 no mock server left running"

# ---- P: pre-check wrapper with fake docker / nvidia-smi
FB=$S/fakebin; mkdir -p "$FB"
cat > "$FB/docker" <<'EOF'
#!/bin/bash
echo "docker $*" >> "$FAKE_LOG"
case "$1" in
  image) awk -v t="$3" '$1==t {print $2}' "$FAKE_IMAGES" ;;
  run) O=""; for a in "$@"; do case "$a" in *:/o) O=${a%:/o};; esac; done
       case "$*" in *precheck_s0.py*) echo '{"verdict": "PASS", "results": []}' > "$O/precheck.json"; echo "PASS C1"; echo "VERDICT PASS" ;; esac ;;
esac
exit 0
EOF
cat > "$FB/nvidia-smi" <<'EOF'
#!/bin/bash
case "$*" in *noheader*) echo "3, 0";; *) echo "index, name, memory.used [MiB], utilization.gpu [%]"; echo "0, RTX, 3 MiB, 0 %";; esac
EOF
chmod +x "$FB/docker" "$FB/nvidia-smi"
WR=$S/wrap-root; mkdir -p "$WR"
export FAKE_LOG=$S/fake-docker.log FAKE_IMAGES=$ROOT/runner/stage0/images.txt
PR=$ROOT/runner/stage0/run_precheck_s0.sh
out=$(env -u APPROVED_GPU_H PATH=$FB:$PATH FAKE=1 ROOT_OVERRIDE=$WR ATTEMPT_REASON=t $PR 2>&1); [[ "$out" == REFUSED*APPROVED_GPU_H* ]]; expect $? "P1 pre-check refused without APPROVED_GPU_H"
out=$(PATH=$FB:$PATH FAKE=1 ROOT_OVERRIDE=$WR APPROVED_GPU_H=0.05 ATTEMPT_REASON=t $PR 2>&1); [[ "$out" == REFUSED*"> approved"* ]]; expect $? "P2 pre-check refused when its cap (0.083 GPU-h) exceeds the approval" "$out"
out=$(PATH=$FB:$PATH FAKE=1 ROOT_OVERRIDE=$WR APPROVED_GPU_H=1.34 DRY_RUN=1 ATTEMPT_REASON=t $PR 2>&1); [[ "$out" == *DRY_RUN* && ! -e "$WR/runs-stage0" ]]; expect $? "P3 pre-check DRY_RUN writes nothing" "$out"
out=$(PATH=$FB:$PATH FAKE=1 ROOT_OVERRIDE=$WR APPROVED_GPU_H=1.34 ATTEMPT_REASON=t $PR 2>&1)
[[ "$out" == *"precheck verdict PASS"* ]] && grep -q -- "--network none --gpus device=0" "$FAKE_LOG" && grep -q ":/pc:ro" "$FAKE_LOG" && grep -q "GLM53_P4_S0=on" "$FAKE_LOG"; expect $? "P4 pre-check (fake docker): --network none, --gpus device=0, read-only code, GLM53_P4_S0=on, verdict PASS" "$out"
python3 -c "
import json
L=[json.loads(l) for l in open('$WR/costs-stage0.jsonl')]
assert [l['item'] for l in L]==['s0-precheck'] and L[0]['gpus']==1 and L[0]['approved_gpu_h']==1.34
"; expect $? "P5 one 1-GPU cost line in costs-stage0.jsonl"
out=$(PATH=$FB:$PATH FAKE=1 ROOT_OVERRIDE=$WR APPROVED_GPU_H=1.34 ATTEMPT_REASON=t $PR 2>&1); [[ "$out" == *"already passed"* ]]; expect $? "P6 a passed stage-0 pre-check is not re-run"
out=$(PATH=$FB:$PATH FAKE=1 APPROVED_GPU_H=1.34 ATTEMPT_REASON=t $PR 2>&1); [[ "$out" == *"ROOT_OVERRIDE under tests/out"* ]]; expect $? "P7 FAKE mode refuses the real package root"
[ "$(sha256sum "$ROOT/costs.jsonl" | cut -d' ' -f1)" = "$REAL_LEDGER" ] && [ "$( [ -e "$ROOT/costs-stage0.jsonl" ] && sha256sum "$ROOT/costs-stage0.jsonl" | cut -d' ' -f1 || echo none)" = "$REAL_S0" ]
expect $? "P8 the real run-1 ledger and the real stage-0 ledger are untouched by these tests"
echo "test_stage0_runner: $NP PASS, $NF FAIL"
[ "$NF" = 0 ]
