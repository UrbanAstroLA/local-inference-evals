#!/bin/bash
# CPU-only tests of harness/run_p3.sh on a scratch copy.  `docker` and
# `nvidia-smi` are replaced by fakes on PATH: no container starts, no GPU is
# queried.  The fake `docker run` writes a stub JSON into the /o mount and logs
# its argument vector so the real command line can be checked.
# Usage: tests/test_wrapper.sh   (writes tests/out-wrapper.txt)
set -u
T=$(cd "$(dirname "$0")" && pwd)
SCR="$T/scratch-wrapper"
LOG="$T/out-wrapper.txt"
rm -rf "$SCR"
mkdir -p "$SCR/harness" "$SCR/fakebin" "$SCR/out"
cp "$T/../harness/run_p3.sh" "$T/../harness/moe_plan_eq.py" "$T/../harness/compare_eversion.py" "$SCR/harness/"

cat >"$SCR/fakebin/docker" <<'EOF'
#!/bin/bash
echo "docker $*" >>"$FAKE_LOG"
case "$1" in
  image) [[ "$*" == *--format* ]] && echo fakecommit; exit 0 ;;
  ps) exit 0 ;;
  rm) exit 0 ;;
  run)
    out=""; odir=""; prev=""
    for a in "$@"; do
      [ "$prev" = "--out" ] && out=$a
      [ "$prev" = "-v" ] && [[ "$a" == *:/o ]] && odir=${a%:/o}
      prev=$a
    done
    echo "RUNARGS $*" >>"$FAKE_LOG"
    [ -n "${FAKE_RUN_SLEEP:-}" ] && [[ "$*" == *--gpus* ]] && sleep "$FAKE_RUN_SLEEP"
    if [ -n "$out" ] && [ -n "$odir" ]; then echo '{"status":"fake"}' >"$odir/$(basename "$out")"; fi
    exit 0 ;;
esac
exit 0
EOF
cat >"$SCR/fakebin/nvidia-smi" <<'EOF'
#!/bin/bash
echo "nvidia-smi $*" >>"$FAKE_LOG"
case "$*" in
  *noheader*) echo "${FAKE_MEM:-3}, ${FAKE_UTIL:-0}" ;;
  *query-compute-apps*) echo "gpu_uuid, pid, process_name, used_memory" ;;
  *) echo "index, name, memory.used [MiB], utilization.gpu [%]"; echo "0, FAKE GPU, 3 MiB, 0 %" ;;
esac
EOF
chmod +x "$SCR/fakebin/"*
export PATH="$SCR/fakebin:$PATH"
export FAKE_LOG="$SCR/fake.log"
W="$SCR/harness/run_p3.sh"
pass=0; fail=0
: >"$LOG"
note() { echo "$*" | tee -a "$LOG"; }
expect() {  # name expected_rc grep_pattern -- command...
  local name=$1 want=$2 pat=$3; shift 3
  local out rc
  out=$("$@" 2>&1); rc=$?
  { echo "----- $name (rc=$rc, want $want)"; echo "$out"; } >>"$LOG"
  if [ "$rc" = "$want" ] && echo "$out" | grep -qE "$pat"; then
    pass=$((pass + 1)); note "PASS $name"
  else
    fail=$((fail + 1)); note "FAIL $name (rc=$rc want=$want pattern=$pat)"
  fi
}
check() {  # name condition-command...
  local name=$1; shift
  if "$@"; then pass=$((pass + 1)); note "PASS $name"; else fail=$((fail + 1)); note "FAIL $name"; fi
}
ncost() { awk 'NF{n++} END{print n+0}' "$SCR/costs.jsonl" 2>/dev/null || echo 0; }

# 1. refuse without ATTEMPT_REASON
expect "refuse-without-reason" 2 "REFUSED: ATTEMPT_REASON" env -u ATTEMPT_REASON DRY_RUN=1 "$W"
# 2. DRY_RUN on an empty ledger: gates pass, budget printed, nothing started or written
: >"$FAKE_LOG"
expect "dryrun-prints-budget" 0 "remaining 1800 s" env ATTEMPT_REASON="test dry" DRY_RUN=1 "$W"
check "dryrun-no-container" bash -c "! grep -q '^docker run' '$FAKE_LOG'"
check "dryrun-no-attempt-dir" test ! -e "$SCR/out/attempt-1"
check "dryrun-no-cost-line" test ! -e "$SCR/costs.jsonl"
expect "dryrun-plans-attempt-1-cap-1800" 0 "attempt 1 \(full\): wall cap 1800 s" env ATTEMPT_REASON="test dry" DRY_RUN=1 "$W"
expect "dryrun-smoke-cap-420" 0 "attempt 1 \(smoke\): wall cap 420 s" env ATTEMPT_REASON="test dry" DRY_RUN=1 SMOKE=1 "$W"
# 3. GPU0 busy / occupied -> abort (fake values)
expect "abort-gpu0-memory" 3 "GPU0 memory 5000 MiB" env ATTEMPT_REASON="t" DRY_RUN=1 FAKE_MEM=5000 "$W"
expect "abort-gpu0-busy" 3 "GPU0 busy" env ATTEMPT_REASON="t" DRY_RUN=1 FAKE_UTIL=80 "$W"
# 4. first attempt (SMOKE, fake docker): attempt-1, one cost line, v091 only, smoke flags
: >"$FAKE_LOG"
expect "smoke-attempt-1" 0 "attempt 1 done" env ATTEMPT_REASON="smoke: validate GPU path" SMOKE=1 "$W"
check "smoke-dir-attempt-1" test -f "$SCR/out/attempt-1/v091.json"
check "smoke-one-cost-line" test "$(ncost)" = 1
check "smoke-cost-fields" python3 -c "
import json; d=json.loads(open('$SCR/costs.jsonl').readline())
assert d['attempt']==1 and d['mode']=='smoke' and d['engine']=='v091' and d['gpus']==1
assert d['attempt_reason']=='smoke: validate GPU path' and d['rc']==0 and d['wall_cap_total']==420
assert d['image']=='sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07'
for k in ('seconds','gpu_hours','start','end','image_timeout'): assert k in d"
check "smoke-runargs" python3 -c "
import re; runs=[l for l in open('$FAKE_LOG') if l.startswith('RUNARGS')]
assert len(runs)==1, runs; r=runs[0]
for need in ('--rm','--network none','--gpus device=0','--user','-e HOME=/tmp',
             '-e VLLM_EXL3_TRELLIS_MAX_M=32','-e VLLM_EXL3_PREFILL_CAPACITY=1024','-e B12X_EXL3_BF16_EPILOGUE=0',
             ':/model-repo:ro','/harness:/h:ro','--engine v091','--smoke','--deadline-s',
             'a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07'):
    assert need in r, need
assert 'CUDA_VISIBLE_DEVICES' not in r"
# 5. second attempt (full) goes to attempt-2; both images; E-version compare runs CPU-only
: >"$FAKE_LOG"
expect "full-attempt-2" 0 "attempt 2 done" env ATTEMPT_REASON="full run" "$W"
check "full-dir-attempt-2" test -f "$SCR/out/attempt-2/v091.json" -a -f "$SCR/out/attempt-2/v070.json"
check "attempt-1-untouched" test -f "$SCR/out/attempt-1/v091.json" -a ! -e "$SCR/out/attempt-1/v070.json"
check "full-cost-lines" python3 -c "
import json; L=[json.loads(l) for l in open('$SCR/costs.jsonl')]
assert len(L)==3; a2=[d for d in L if d['attempt']==2]
assert [d['engine'] for d in a2]==['v091','v070'] and all(d['mode']=='full' for d in a2)
assert a2[1]['image']=='sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629'"
check "first-image-soft-share" python3 -c "
import re; runs=[l for l in open('$FAKE_LOG') if l.startswith('RUNARGS') and '--gpus' in l]
d={re.search(r'--engine (\\w+)', r).group(1): int(re.search(r'--deadline-s (\\d+)', r).group(1)) for r in runs}
assert d['v091']==990 and d['v070']>990, d"
check "compare-runs-without-gpu" python3 -c "
runs=[l for l in open('$FAKE_LOG') if l.startswith('RUNARGS')]
assert len(runs)==3, len(runs)
cmp=[r for r in runs if 'compare_eversion.py' in r]
assert len(cmp)==1 and '--gpus' not in cmp[0] and 'CUDA_VISIBLE_DEVICES=' in cmp[0] and '--network none' in cmp[0]"
# 6. DRY_RUN after two attempts: budget reduced, next would be attempt-3
expect "dryrun-after-attempts" 0 "over 2 prior attempt\(s\)" env ATTEMPT_REASON="t" DRY_RUN=1 "$W"
# 7. never overwrite: a stray attempt-3 dir makes the next attempt refuse
mkdir "$SCR/out/attempt-3"
expect "refuse-existing-attempt-dir" 2 "attempt-3 already exists" env ATTEMPT_REASON="t" "$W"
rmdir "$SCR/out/attempt-3"
# 8. partial budget: cap shrinks to the remainder
cp "$SCR/costs.jsonl" "$SCR/costs.keep"
echo '{"item":"synthetic","attempt":90,"seconds":1000,"gpus":1,"gpu_hours":0.2778}' >>"$SCR/costs.jsonl"
expect "cap-shrinks-to-remaining" 0 "wall cap (79[0-9]|800) s" env ATTEMPT_REASON="t" DRY_RUN=1 "$W"
# 9. budget exhausted -> refuse
echo '{"item":"synthetic","attempt":91,"seconds":620,"gpus":1,"gpu_hours":0.1722}' >>"$SCR/costs.jsonl"
expect "refuse-budget-exhausted" 2 "REFUSED: budget exhausted" env ATTEMPT_REASON="t" DRY_RUN=1 "$W"
expect "refuse-budget-exhausted-smoke" 2 "REFUSED: budget exhausted" env ATTEMPT_REASON="t" DRY_RUN=1 SMOKE=1 "$W"
# 10. second image skipped (recorded) when < 300 s remain after the first
cp "$SCR/costs.keep" "$SCR/costs.jsonl"
echo '{"item":"synthetic","attempt":92,"seconds":1488,"gpus":1,"gpu_hours":0.4133}' >>"$SCR/costs.jsonl"
before=$(ncost)
expect "second-image-skipped" 0 "SKIP v070" env ATTEMPT_REASON="skip test" FAKE_RUN_SLEEP=15 "$W"
check "skip-cost-line" python3 -c "
import json; L=[json.loads(l) for l in open('$SCR/costs.jsonl')][$before:]
assert [d['engine'] for d in L]==['v091','v070'], L
assert L[1]['skipped'] is True and L[1]['seconds']==0 and 'left of' in L[1]['skip_reason']"
# 11. malformed ledger -> abort (cannot verify budget)
echo 'not json' >>"$SCR/costs.jsonl"
expect "abort-malformed-ledger" 3 "malformed line" env ATTEMPT_REASON="t" DRY_RUN=1 "$W"

note "wrapper tests: $pass passed, $fail failed"
[ "$fail" = 0 ]
