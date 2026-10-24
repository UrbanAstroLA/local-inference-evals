#!/bin/bash
# CPU tests of the capture-p0 GPU wrappers (no GPU, no serving container, never the real ledger):
#   S*  runner/run_cap0.sh: refusals (ATTEMPT_REASON, APPROVED_GPU_H, protocol ack, mock root outside tests/out,
#       CAP0_LOC value), order (needs a passed pre-check), DRY_RUN writes nothing, a MOCK session end to end against
#       tests/mock_server_cap0.py (G-instr + G3 before traffic, replicates one request at a time with a salt each,
#       capture / loc records flushed, 0700 private copy-out under a non-PRIVATE synthetic path, one cost line, outcome,
#       the analysis in a CPU container -> INCOMPLETE because a mock carries no bytes: missing evidence is never a
#       negative), no re-run after a client-complete session; CAP0_LOC=off plans every replicate "N"
#   P*  runner/run_precheck_cap0.sh with FAKE=1 (fake docker / nvidia-smi): refusals, mounts and env, cost line,
#       verdict, a CPU-only pre-check never counts as a pass
# Scratch: ${OUT_OVERRIDE:-tests/out}/runner-scratch (inside tests/out).  No `set -e`.
T=$(cd "$(dirname "$0")" && pwd)
PKG=$(cd "$T/.." && pwd)
S=${OUT_OVERRIDE:-$T/out}/runner-scratch
case "$S" in "$PKG"/tests/out/*) ;; *) echo "scratch must be under tests/out"; exit 2;; esac
rm -rf "$S"; mkdir -p "$S"
NP=0; NF=0
ok()  { echo "PASS $*"; NP=$((NP + 1)); }
bad() { echo "FAIL $*"; NF=$((NF + 1)); }
expect() { if [ "$1" = 0 ]; then ok "$2"; else bad "$2 ${3:-}"; fi; }
ACK=$(sha256sum "$PKG/PROTOCOL.md" | cut -c1-12)
R=$PKG/runner/run_cap0.sh
MR=$S/mock-root
export MOCK_ROOT=$MR MOCK_PORT=18061
python3 "$T/make_mock_root_cap0.py" "$MR" > /dev/null
REAL=$( [ -e "$PKG/costs-cap0.jsonl" ] && sha256sum "$PKG/costs-cap0.jsonl" | cut -d' ' -f1 || echo none)
PIN=$(awk '{print $2}' "$PKG/runner/images.txt")
pass_precheck() { mkdir -p "$1/runs/precheck-attempt-1"; echo "{\"verdict\": \"PASS\", \"image\": \"$PIN\"}" > "$1/runs/precheck-attempt-1/verdict.json"; }

out=$(MOCK=1 APPROVED_GPU_H=2 CAP0_PROTOCOL_ACK=$ACK $R 2>&1); [[ "$out" == REFUSED*ATTEMPT_REASON* ]]; expect $? "S1 refuses without ATTEMPT_REASON"
out=$(env -u APPROVED_GPU_H MOCK=1 ATTEMPT_REASON=t CAP0_PROTOCOL_ACK=$ACK $R 2>&1); [[ "$out" == REFUSED*APPROVED_GPU_H* ]]; expect $? "S2 refuses with APPROVED_GPU_H unset (no default)"
out=$(MOCK=1 APPROVED_GPU_H=2 ATTEMPT_REASON=t CAP0_PROTOCOL_ACK=000000000000 $R 2>&1); [[ "$out" == REFUSED*CAP0_PROTOCOL_ACK* ]]; expect $? "S3 refuses a wrong protocol acknowledgement"
out=$(MOCK=1 APPROVED_GPU_H=2 ATTEMPT_REASON=t CAP0_PROTOCOL_ACK=$ACK MOCK_ROOT=/tmp/elsewhere $R 2>&1); [[ "$out" == *"outside tests/out"* ]]; expect $? "S3 MOCK refuses a mock root outside tests/out"
out=$(MOCK=1 APPROVED_GPU_H=2 ATTEMPT_REASON=t CAP0_PROTOCOL_ACK=$ACK CAP0_LOC=maybe $R 2>&1); [[ "$out" == *"CAP0_LOC must be"* ]]; expect $? "S3 refuses a CAP0_LOC other than on/off"
out=$(MOCK=1 APPROVED_GPU_H=2 ATTEMPT_REASON=t CAP0_PROTOCOL_ACK=$ACK $R 2>&1); [[ "$out" == *"pre-check passed for image"* ]]; expect $? "S4 session refused before a passed pre-check" "$out"
mkdir -p "$MR/runs/precheck-attempt-1"; echo '{"verdict": "PASS", "image": "sha256:0ld"}' > "$MR/runs/precheck-attempt-1/verdict.json"
out=$(MOCK=1 APPROVED_GPU_H=2 ATTEMPT_REASON=t CAP0_PROTOCOL_ACK=$ACK $R 2>&1); [[ "$out" == *"pre-check passed for image"* ]]; expect $? "S4 a pre-check passed for ANOTHER image does not count" "$out"
pass_precheck "$MR"
echo '{"item": "cap0-precheck", "attempt": 1, "seconds": 30, "gpus": 1}' > "$MR/costs-cap0.jsonl"
out=$(MOCK=1 APPROVED_GPU_H=0.6 ATTEMPT_REASON=t CAP0_PROTOCOL_ACK=$ACK $R 2>&1); [[ "$out" == REFUSED*"> approved"* ]]; expect $? "S5 cap 0.611 + used 0.008 > 0.6: refused (cap never shrunk)" "$out"
BEFORE=$(find "$MR" -printf '%p %s\n' | sort | sha256sum)
out=$(MOCK=1 DRY_RUN=1 APPROVED_GPU_H=0.67 ATTEMPT_REASON=t CAP0_PROTOCOL_ACK=$ACK $R 2>&1)
AFTER=$(find "$MR" -printf '%p %s\n' | sort | sha256sum)
[[ "$out" == *DRY_RUN* && "$out" == *'"warm_tokens": 17409'* && "$out" == *'"expected_cap_cached": 13056'* && "$BEFORE" == "$AFTER" ]]; expect $? "S6 DRY_RUN passes the gates, prints the plan (warm 17,409 tokens -> capture cached 13,056) and writes nothing" "$(echo "$out" | tail -2)"
out=$(MOCK=1 DRY_RUN=1 CAP0_LOC=off APPROVED_GPU_H=0.67 ATTEMPT_REASON=t CAP0_PROTOCOL_ACK=$ACK $R 2>&1)
[[ "$out" == *'"loc": false'* && "$out" != *'"loc": true'* ]]; expect $? "S6 CAP0_LOC=off: every replicate planned without localisation"

timeout 300 env MOCK=1 APPROVED_GPU_H=0.67 ATTEMPT_REASON="mock capture" CAP0_PROTOCOL_ACK=$ACK $R > "$S/mock-1.log" 2>&1
RC=$?
AO=$MR/runs/session-attempt-1; PO=$MR/private-out-synthetic/session-attempt-1
V=$(python3 -c "import json;print(json.load(open('$AO/outcome.json'))['verdict'])" 2>/dev/null)
[ "$V" = COMPLETE ]; expect $? "S7 MOCK session: COMPLETE (G-instr + G3, 6 replicates, captures and loc records on both ranks)" "verdict $V rc $RC $(tail -5 "$S/mock-1.log")"
grep -q "G-instr: OK" "$AO/gate.txt" && grep -q "GATES PASS A startup" "$AO/gate.txt"; expect $? "S7 G-instr and G3 evaluated before traffic"
python3 -c "
import json, os
L = [json.loads(l) for l in open('$AO/client.jsonl')]
first = min(d['t_send'] for d in L if d['kind'] in ('warm', 'cap'))
assert os.path.getmtime('$AO/gates-startup.json') < first
reqs = [d for d in L if d['kind'] in ('warm', 'cap')]
assert len(reqs) == 12 and all(d['ok'] for d in reqs), [d for d in reqs if not d['ok']]
t = [(d['t_send'], d['t_done']) for d in reqs]
assert all(t[i][1] <= t[i + 1][0] for i in range(len(t) - 1)), 'requests overlap'
salts = sorted({d['salt'] for d in reqs}); assert salts == ['cap0-r%d' % k for k in range(1, 7)], salts
assert {d['seed'] for d in reqs if d['kind'] == 'warm'} == {72999} and {d['seed'] for d in reqs if d['kind'] == 'cap'} == {73000}
assert len({d['prompt_sha256'] for d in reqs}) == 1, 'warm and capture prompts must be the same 17,409 ids'
assert all(d['cached_tokens'] == (0 if d['kind'] == 'warm' else 13056) for d in reqs), [(d['kind'], d['cached_tokens']) for d in reqs]
assert all(d['prompt_tokens'] == 17409 for d in reqs)
tags = [d['tag'] for d in reqs if d['kind'] == 'warm']; assert tags == ['CW-r1-L', 'CW-r2-L', 'CW-r3-N', 'CW-r4-N', 'CW-r5-L', 'CW-r6-L'], tags
"; expect $? "S7 gates before the first request; 12 requests one at a time; a salt per replicate; one seed per kind; warm = capture prompt (17,409); mock cache gives warm 0, capture 13,056; tags L,L,N,N,L,L"
python3 -c "
import json
L=[json.loads(l) for l in open('$MR/costs-cap0.jsonl')]
s=[l for l in L if l['item']=='cap0-session']
assert len(s) == 1 and s[0]['gpus'] == 2 and s[0]['seconds'] > 0 and s[0]['approved_gpu_h'] == 0.67 and s[0]['mock'] and s[0]['loc'] == 'on'
"; expect $? "S7 one 2-GPU cost line in costs-cap0.jsonl (approval and loc recorded)"
[ "$(stat -c %a "$PO")" = 700 ] && [ "$(find "$PO" -type f | wc -l)" -ge 20 ] && [ ! -e "$MR/PRIVATE-capture-p0" ]; expect $? "S7 private copy-out at 0700 under the synthetic path ($(find "$PO" -type f 2>/dev/null | wc -l) files); nothing under a PRIVATE path"
[ -s "$AO/cap0-result.json" ] && python3 -c "
import json; d = json.load(open('$AO/cap0-result.json')); assert d['cls'] == 'incomplete', d['cls']
f = [x for g in d['gates'].values() for x in g['failures']]
assert not [x for x in f if x.startswith('G-client')], [x for x in f if x.startswith('G-client')]
assert all(any('no scheduler log at all' in x for x in g['failures']) for g in d['gates'].values())
"; expect $? "S7 the analysis reads the mock INCOMPLETE for the RIGHT reasons: G-client passes; the mock has no scheduler log (only the real flush path writes one) and no bytes" "$(tail -3 "$AO/analysis-console.txt" 2>/dev/null)"
out=$(MOCK=1 APPROVED_GPU_H=5 ATTEMPT_REASON=t CAP0_PROTOCOL_ACK=$ACK $R 2>&1); [[ "$out" == *"needs CAP0_PRIOR_ACK="* ]]; expect $? "S8 after a completed attempt that read INCOMPLETE: refused without CAP0_PRIOR_ACK" "$out"
PACK=$(sha256sum "$AO/cap0-result.json" | cut -c1-12)
out=$(MOCK=1 DRY_RUN=1 CAP0_PRIOR_ACK=000000000000 APPROVED_GPU_H=5 ATTEMPT_REASON=t CAP0_PROTOCOL_ACK=$ACK $R 2>&1); [[ "$out" == *"needs CAP0_PRIOR_ACK="* ]]; expect $? "S8 a wrong CAP0_PRIOR_ACK is refused"
out=$(MOCK=1 DRY_RUN=1 CAP0_PRIOR_ACK=$PACK APPROVED_GPU_H=5 ATTEMPT_REASON=t CAP0_PROTOCOL_ACK=$ACK $R 2>&1); [[ "$out" == *"DRY_RUN:"*"session attempt 2"* ]]; expect $? "S8 with CAP0_PRIOR_ACK of the INCOMPLETE result: attempt 2 passes the gates" "$(echo "$out" | head -3)"
cp "$AO/cap0-result.json" "$S/keep.json"; python3 -c "
import json; d=json.load(open('$AO/cap0-result.json')); d['cls']='located'; json.dump(d, open('$AO/cap0-result.json','w'))"
out=$(MOCK=1 DRY_RUN=1 CAP0_PRIOR_ACK=$(sha256sum "$AO/cap0-result.json" | cut -c1-12) APPROVED_GPU_H=5 ATTEMPT_REASON=t CAP0_PROTOCOL_ACK=$ACK $R 2>&1); [[ "$out" == *"not INCOMPLETE"* ]]; expect $? "S8 a completed attempt with a valid reading is never re-run, even with an acknowledgement" "$out"
cp "$S/keep.json" "$AO/cap0-result.json"
for i in 1 2 3 4 5; do curl -sf -m 1 "http://127.0.0.1:$MOCK_PORT/v1/models" > /dev/null || break; sleep 1; done
curl -sf -m 1 "http://127.0.0.1:$MOCK_PORT/v1/models" > /dev/null; [ $? != 0 ]; expect $? "S8 no mock server left listening"

# ---- S9: a localisation-hook shortfall degrades the option; it does not spend the session
MR2=$S/mock-root-2
python3 "$T/make_mock_root_cap0.py" "$MR2" > /dev/null
pass_precheck "$MR2"
timeout 300 env MOCK=1 MOCK_ROOT=$MR2 MOCK_PORT=18062 MOCK_HOOKS_R0=10 APPROVED_GPU_H=0.67 ATTEMPT_REASON="mock degraded" CAP0_PROTOCOL_ACK=$ACK $R > "$S/mock-2.log" 2>&1
grep -q "G-instr: OK.*LOCALISATION DEGRADED.*rank 0 hooks 10 of 12" "$MR2/runs/session-attempt-1/gate.txt" && [ -s "$MR2/runs/session-attempt-1/client.jsonl" ]
expect $? "S9 10 of 12 hooks on rank 0: G-instr OK with LOCALISATION DEGRADED; the client still runs" "$(grep G-instr "$MR2/runs/session-attempt-1/gate.txt" 2>/dev/null)"

# ---- S10: the attempt-1 construction (13,057-token warm) against the mock's one-block drop -> early abort after r1
MR3=$S/mock-root-3
python3 "$T/make_mock_root_cap0.py" "$MR3" > /dev/null
pass_precheck "$MR3"
timeout 300 env MOCK=1 MOCK_ROOT=$MR3 MOCK_PORT=18063 CAP0_TEST_WARM_TOKENS=13057 APPROVED_GPU_H=0.67 ATTEMPT_REASON="mock attempt-1 construction" CAP0_PROTOCOL_ACK=$ACK $R > "$S/mock-3.log" 2>&1
AO3=$MR3/runs/session-attempt-1
python3 -c "
import json
L = [json.loads(l) for l in open('$AO3/client.jsonl')]
ab = [d for d in L if d['kind'] == 'abort']
assert len(ab) == 1 and 'capture cached_tokens 8704 (expected 13056)' in ab[0]['reason'], ab
assert [d['replicate'] for d in L if d['kind'] == 'replicate_done'] == ['r1']
o = json.load(open('$AO3/outcome.json')); assert o['aborted'] and o['verdict'] != 'COMPLETE', o
r = json.load(open('$AO3/cap0-result.json')); f = r['gates']['r1']['failures']
assert any('capture cached_tokens 8704 (must be 13056)' in x for x in f) and any('prompt_tokens 13057 (must be 17409' in x for x in f), f
"; expect $? "S10 attempt-1 construction: the mock drops a block (8,704), the client aborts after r1 with the reason, the analysis names it" "$(tail -4 "$S/mock-3.log")"
out=$(env CAP0_TEST_WARM_TOKENS=13057 CAP0_TEST_MOCK=0 CAP0_WINDOWS=$MR3/fixtures-synthetic/windows.json CAP0_PLAN_ONLY=1 python3 "$PKG/runner/client_cap0.py" x 0 x)
[[ "$out" == *'"warm_tokens": 17409'* ]]; expect $? "S10 outside a mock the test-only warm override is ignored" "$out"
grep -q 'TESTENV="CAP0_TEST_MOCK=0 CAP0_TEST_WARM_TOKENS="' "$R"; expect $? "S10 the GPU runner clears the test-only override for real sessions"

# ---- P: pre-check wrapper with fake docker / nvidia-smi
FB=$S/fakebin; mkdir -p "$FB"
cat > "$FB/docker" <<'EOF2'
#!/bin/bash
echo "docker $*" >> "$FAKE_LOG"
case "$1" in
  image) awk -v t="$3" '$1==t {print $2}' "$FAKE_IMAGES" ;;
  run) O=""; for a in "$@"; do case "$a" in *:/o) O=${a%:/o};; esac; done
       case "$*" in *precheck_cap0.py*) echo "{\"verdict\": \"PASS\", \"gpu\": $FAKE_GPU, \"times\": {}}" > "$O/precheck.json"; echo "PASS K1"; echo "VERDICT PASS" ;; esac ;;
esac
exit 0
EOF2
cat > "$FB/nvidia-smi" <<'EOF2'
#!/bin/bash
case "$*" in *noheader*) echo "3, 0";; *) echo "index, name, memory.used [MiB], utilization.gpu [%]"; echo "0, RTX, 3 MiB, 0 %";; esac
EOF2
chmod +x "$FB/docker" "$FB/nvidia-smi"
WR=$S/wrap-root; mkdir -p "$WR"
export FAKE_LOG=$S/fake-docker.log FAKE_IMAGES=$PKG/runner/images.txt FAKE_GPU=true
PR=$PKG/runner/run_precheck_cap0.sh
out=$(env -u APPROVED_GPU_H PATH=$FB:$PATH FAKE=1 ROOT_OVERRIDE=$WR ATTEMPT_REASON=t $PR 2>&1); [[ "$out" == REFUSED*APPROVED_GPU_H* ]]; expect $? "P1 pre-check refused without APPROVED_GPU_H"
out=$(PATH=$FB:$PATH FAKE=1 ROOT_OVERRIDE=$WR APPROVED_GPU_H=0.04 ATTEMPT_REASON=t $PR 2>&1); [[ "$out" == REFUSED*"> approved"* ]]; expect $? "P2 pre-check refused when its cap (0.05 GPU-h) exceeds the approval" "$out"
out=$(PATH=$FB:$PATH FAKE=1 ROOT_OVERRIDE=$WR APPROVED_GPU_H=0.67 DRY_RUN=1 ATTEMPT_REASON=t $PR 2>&1); [[ "$out" == *DRY_RUN* && ! -e "$WR/runs" ]]; expect $? "P3 pre-check DRY_RUN writes nothing" "$out"
FAKE_GPU=false
export FAKE_GPU
out=$(PATH=$FB:$PATH FAKE=1 ROOT_OVERRIDE=$WR APPROVED_GPU_H=0.67 ATTEMPT_REASON=t $PR 2>&1)
[[ "$out" == *"precheck verdict PASS"* ]]; expect $? "P4 FAKE pre-check reports its verdict" "$out"
grep -q -- "--network none --gpus device=0" "$FAKE_LOG" && grep -q ":/pc:ro" "$FAKE_LOG" && grep -q "GLM53_CAP0=on" "$FAKE_LOG" && grep -q "GLM53_CAP0_LOC=on" "$FAKE_LOG"
expect $? "P4 pre-check container: --network none, --gpus device=0, read-only code, GLM53_CAP0=on + LOC"
python3 -c "
import json
L=[json.loads(l) for l in open('$WR/costs-cap0.jsonl')]
assert [l['item'] for l in L]==['cap0-precheck'] and L[0]['gpus']==1 and L[0]['approved_gpu_h']==0.67 and L[0]['fake']
v=json.load(open('$WR/runs/precheck-attempt-1/verdict.json')); assert v['image'] == '$PIN', v
"; expect $? "P5 one 1-GPU cost line in costs-cap0.jsonl; the verdict records the image it ran"
out=$(PATH=$FB:$PATH FAKE=1 ROOT_OVERRIDE=$WR APPROVED_GPU_H=0.67 ATTEMPT_REASON=t $PR 2>&1); [[ "$out" == *"already passed"* ]]; expect $? "P6 a passed pre-check is not re-run"
out=$(PATH=$FB:$PATH FAKE=1 APPROVED_GPU_H=0.67 ATTEMPT_REASON=t $PR 2>&1); [[ "$out" == *"ROOT_OVERRIDE under tests/out"* ]]; expect $? "P7 FAKE mode refuses the real package root"
python3 - "$PKG/runner/run_precheck_cap0.sh" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
assert 'd.get("gpu") or os.environ.get("FAKE") == "1"' in s
PY
expect $? "P8 outside FAKE, a pre-check that did not run on the GPU is FAIL (rule present in the wrapper)"
[ "$( [ -e "$PKG/costs-cap0.jsonl" ] && sha256sum "$PKG/costs-cap0.jsonl" | cut -d' ' -f1 || echo none)" = "$REAL" ]
expect $? "P9 the real capture-p0 ledger is untouched by these tests"
echo "test_runner_cap0: $NP PASS, $NF FAIL"
[ "$NF" = 0 ]
