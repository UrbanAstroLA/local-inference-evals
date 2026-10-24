#!/bin/bash
# Host test of the GPU runner and its gates (no GPU, no real container): fake docker and nvidia-smi on PATH, every
# run under tests/out/runner/<case> (ROOT_OVERRIDE, FAKE=1).  Needs the package frozen (SHA256SUMS current).
#   tests/test_runner.sh HARNESS_DIR      (HARNESS_DIR/h-order supplies the fake container outputs)
# Prints PASS/FAIL per case; exit 1 on any FAIL.  No `set -e`.
set -u
T=$(cd "$(dirname "$0")" && pwd)
PKG=$(cd "$T/.." && pwd)
HD=$1
OUT=$T/out/runner
rm -rf "$OUT"; mkdir -p "$OUT"
export PATH="$T/fake:$PATH"
export FAKE_IMAGES=$PKG/runner/images.txt
FAILS=0
pass() { echo "PASS runner $1"; }
fail() { echo "FAIL runner $1: $2"; FAILS=$((FAILS + 1)); }
Q_IDLE=$OUT/q-idle; printf '0, 20, 0\n1, 1995, 1\n' > "$Q_IDLE"
Q_BUSY=$OUT/q-busy; printf '0, 20, 60\n1, 1995, 1\n' > "$Q_BUSY"
Q_MEM=$OUT/q-mem; printf '0, 4000, 0\n1, 1995, 1\n' > "$Q_MEM"
mk_table() {  # file, extra process rows
  { cat <<'EOF'
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           89543      G   /usr/bin/gnome-shell                      8MiB |
|    1   N/A  N/A           91069      G   .../8969/usr/lib/firefox/firefox        391MiB |
|    1   N/A  N/A           99539    C+G   ...88185955192 --disable-logging        281MiB |
EOF
    printf '%s\n' "${@:2}"
    echo "+-----------------------------------------------------------------------------------------+"; } > "$1"
}
TB_OK=$OUT/t-ok; mk_table "$TB_OK"
TB_COMP=$OUT/t-comp; mk_table "$TB_COMP" "|    0   N/A  N/A           12345      C   python3                                 900MiB |"
TB_GFX=$OUT/t-gfx; mk_table "$TB_GFX" "|    0   N/A  N/A           23456      G   /usr/lib/firefox/firefox                300MiB |"
export FAKE_SMI_QUERY=$Q_IDLE FAKE_SMI_TABLE=$TB_OK

run() {  # case, expect-substring, expect-rc-class (ok|refused|abort), env...
  local c=$1 want=$2 cls=$3; shift 3
  mkdir -p "$OUT/$c"
  local out rc
  out=$(cd "$PKG" && env FAKE=1 ROOT_OVERRIDE="$OUT/$c" FAKE_LOG="$OUT/$c/docker.log" "$@" bash runner/run_replay_topk.sh 2>&1)
  rc=$?
  echo "$out" > "$OUT/$c/console.txt"
  local okrc=0
  case $cls in ok) [ $rc = 0 ] && okrc=1;; refused) [ $rc = 2 ] && okrc=1;; abort) [ $rc = 3 ] && okrc=1;; any) okrc=1;; esac
  if [ $okrc = 1 ] && echo "$out" | grep -qF -- "$want"; then pass "$c"; else fail "$c" "rc=$rc; want '$want'; got: $(echo "$out" | tail -2 | tr '\n' ' ')"; fi
}
A=(APPROVED_GPU_H=0.2 ATTEMPT_REASON=test)
run no-reason "ATTEMPT_REASON is not set" refused APPROVED_GPU_H=0.2
run fake-root "FAKE=1 needs ROOT_OVERRIDE under tests/out" refused ATTEMPT_REASON=x APPROVED_GPU_H=0.2 ROOT_OVERRIDE=/tmp/x
run no-approval "APPROVED_GPU_H is not set" refused ATTEMPT_REASON=x
run approval-nan "must be a positive finite number" refused ATTEMPT_REASON=x APPROVED_GPU_H=nan
run approval-inf "must be a positive finite number" refused ATTEMPT_REASON=x APPROVED_GPU_H=inf
run approval-zero "must be a positive finite number" refused ATTEMPT_REASON=x APPROVED_GPU_H=0
run approval-text "is not a number" refused ATTEMPT_REASON=x APPROVED_GPU_H=abc
run approval-small "> approved" refused ATTEMPT_REASON=x APPROVED_GPU_H=0.15
run approval-one-image "DRY_RUN: all gates passed" ok ATTEMPT_REASON=x APPROVED_GPU_H=0.12 IMAGES=v091 DRY_RUN=1
run bad-image-key "unknown image key" refused "${A[@]}" IMAGES=v080
run image-mismatch "expected sha256:a9e985c0" abort "${A[@]}" FAKE_BAD_IMAGE=1
run container-running "an inference or check container is running" abort "${A[@]}" FAKE_PS=glm53-flash-exl3-b12x-vllm
run gpu-compute "compute process on GPU 0" abort "${A[@]}" DRY_RUN=1 FAKE_SMI_TABLE=$TB_COMP
run gpu-graphics "graphics client on GPU 0 above 64 MiB" abort "${A[@]}" DRY_RUN=1 FAKE_SMI_TABLE=$TB_GFX
run gpu-busy "utilisation 60% > 5% after 12 samples" abort "${A[@]}" DRY_RUN=1 FAKE_SMI_QUERY=$Q_BUSY
run gpu-mem "memory 4000 MiB > 1024 MiB" abort "${A[@]}" DRY_RUN=1 FAKE_SMI_QUERY=$Q_MEM
run dry "graphics clients reported: 3 (1 on GPU 0)" ok "${A[@]}" DRY_RUN=1
[ -e "$OUT/dry/runs" ] || [ -e "$OUT/dry/costs-replay.jsonl" ] && fail "dry wrote nothing" "runs/ or ledger exists" || pass "dry wrote nothing"
# full fake run: outputs from the harness h-order case, one ledger line per image, analysis run
run full "v091 R1 [K: A-ORDER" ok "${A[@]}" FAKE_OUT_SRC="$HD/h-order"
L=$OUT/full/costs-replay.jsonl
if [ -f "$L" ] && [ "$(wc -l < "$L")" = 2 ] && python3 -c "
import json,sys
ls=[json.loads(l) for l in open('$L')]
assert [l['engine'] for l in ls]==['v091','v070'] and all(l['gpus']==1 and l['item']=='replay-topk' and l['attempt']==1 for l in ls)
assert ls[0]['wall_cap']==400 and ls[1]['wall_cap']==300"; then pass "full: ledger lines"; else fail "full: ledger lines" "$(cat "$L" 2>/dev/null)"; fi
[ -f "$OUT/full/runs/attempt-1/replay-result.json" ] && pass "full: analysis written" || fail "full: analysis written" "missing"
grep -q -- "--network none --gpus device=0" "$OUT/full/docker.log" && grep -q "rm -f rtk-1-v091" "$OUT/full/docker.log" \
  && pass "full: docker flags and cleanup" || fail "full: docker flags and cleanup" "$(head -3 "$OUT/full/docker.log")"
grep -q -- "B12X_COMPILE_CACHE_DIR=/cc/b12x" "$OUT/full/docker.log" && pass "full: compile cache env" || fail "full: compile cache env" ""
# a complete reading is not re-run
run rerun-complete "produced a complete reading" refused "${A[@]}" ROOT_OVERRIDE="$OUT/full" DRY_RUN=1
# an INCOMPLETE reading needs the operator's acknowledgement
mkdir -p "$OUT/inc"; cp -r "$OUT/full/runs" "$OUT/inc/"; cp "$L" "$OUT/inc/"
python3 "$PKG/analysis/analyze_replay.py" "$OUT/inc/runs/attempt-1" >/dev/null 2>&1
rm -f "$OUT/inc/runs/attempt-1/v070/r2.json"; python3 "$PKG/analysis/analyze_replay.py" "$OUT/inc/runs/attempt-1" >/dev/null
ACK=$(sha256sum "$OUT/inc/runs/attempt-1/replay-result.json" | cut -c1-12)
run rerun-noack "needs REPLAY_PRIOR_ACK=$ACK" refused "${A[@]}" ROOT_OVERRIDE="$OUT/inc" DRY_RUN=1
run rerun-ack "attempt 2;" ok "${A[@]}" ROOT_OVERRIDE="$OUT/inc" DRY_RUN=1 REPLAY_PRIOR_ACK=$ACK APPROVED_GPU_H=0.4
run rerun-budget "> approved" refused "${A[@]}" ROOT_OVERRIDE="$OUT/inc" DRY_RUN=1 REPLAY_PRIOR_ACK=$ACK APPROVED_GPU_H=0.19
# attempt directory never overwritten
mkdir -p "$OUT/exists/runs/attempt-1"     # an aborted attempt (no analysis): the next one is numbered 2
run attempt-exists "attempt 2;" ok "${A[@]}" ROOT_OVERRIDE="$OUT/exists" DRY_RUN=1
# interrupt during a container: cost line with rc 130, container removed
mkdir -p "$OUT/intr"
( cd "$PKG" && env FAKE=1 ROOT_OVERRIDE="$OUT/intr" FAKE_LOG="$OUT/intr/docker.log" FAKE_SLEEP=30 "${A[@]}" \
    bash runner/run_replay_topk.sh > "$OUT/intr/console.txt" 2>&1 ) &
RP=$!
for i in $(seq 1 40); do [ -d "$OUT/intr/runs/attempt-1/v091" ] && break; sleep 0.25; done
sleep 1
pkill -TERM -P "$RP" -f run_replay_topk.sh 2>/dev/null || kill -TERM "$(pgrep -P "$RP" -f run_replay_topk.sh)" 2>/dev/null
T0=$(date +%s); wait "$RP"; T1=$(date +%s)
LI=$OUT/intr/costs-replay.jsonl
if [ -f "$LI" ] && python3 -c "
import json
ls=[json.loads(l) for l in open('$LI')]
assert len(ls)==1 and ls[0]['engine']=='v091' and ls[0]['rc']==130, ls"; then pass "interrupt: cost line rc 130"; else fail "interrupt: cost line rc 130" "$(cat "$LI" 2>/dev/null)"; fi
[ $((T1 - T0)) -lt 20 ] && pass "interrupt handled promptly ($((T1 - T0))s)" || fail "interrupt handled promptly" "$((T1 - T0))s"
grep -q "rm -f rtk-1-v091" "$OUT/intr/docker.log" && pass "interrupt: container removed" || fail "interrupt: container removed" ""
# tampered package: the SHA gate refuses (on a copy)
CP=$OUT/pkgcopy; rm -rf "$CP"; mkdir -p "$CP"
(cd "$PKG" && tar cf - --exclude=./tests/out --exclude=./runs --exclude=./cache .) | (cd "$CP" && tar xf -)
echo "# tampered" >> "$CP/scripts/r2_mla.py"
mkdir -p "$CP/tests/out/x"
out=$(cd "$CP" && env FAKE=1 ROOT_OVERRIDE="$CP/tests/out/x" "${A[@]}" bash runner/run_replay_topk.sh 2>&1); rc=$?
[ $rc = 2 ] && echo "$out" | grep -q "package differs from SHA256SUMS" && pass "tampered package refused" || fail "tampered package refused" "$out"
# budget gate units
B() { env "$@" python3 "$PKG/runner/budget_replay.py" "$OUT/budget" "${IMGS:-v091,v070}"; }
mkdir -p "$OUT/budget"
echo '{"item":"replay-topk","attempt":1,"engine":"v091","seconds":360,"gpus":1}' > "$OUT/budget/costs-replay.jsonl"
r=$(B APPROVED_GPU_H=0.3); case "$r" in "OK 0.1000 0.2000 1 400 300") pass "budget: used counted";; *) fail "budget: used counted" "$r";; esac
r=$(B APPROVED_GPU_H=0.29); case "$r" in REFUSE*) pass "budget: caps never shrunk";; *) fail "budget: caps never shrunk" "$r";; esac
echo 'garbage' >> "$OUT/budget/costs-replay.jsonl"
r=$(B APPROVED_GPU_H=1); case "$r" in "REFUSE malformed ledger line 2"*) pass "budget: malformed ledger refused";; *) fail "budget: malformed ledger" "$r";; esac
r=$(env APPROVED_GPU_H=1 python3 "$PKG/runner/budget_replay.py" "$OUT/budget" v091,v091); case "$r" in REFUSE*) pass "budget: repeated image refused";; *) fail "budget: repeated image" "$r";; esac
echo "runner: $FAILS failure(s)"
[ "$FAILS" = 0 ]
