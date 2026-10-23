#!/bin/bash
# All CPU tests for P3 Stage 1.  No GPU: every container runs with
# --network none, -e CUDA_VISIBLE_DEVICES= and no --gpus; wrapper tests use
# fake docker/nvidia-smi.  Outputs land in tests/ (out-*.json, *.log,
# out-wrapper.txt, RESULTS.txt).
set -u
T=$(cd "$(dirname "$0")" && pwd)
H=$(cd "$T/../harness" && pwd)
MR=<models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1
REPO=ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx
declare -A IMG=([v091]=$REPO@sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
                [v070]=$REPO@sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629)
ENVS=(-e VLLM_EXL3_TRELLIS_MIN_M=1 -e VLLM_EXL3_TRELLIS_MAX_M=32 -e VLLM_EXL3_TRELLIS_BLOCK_M=8
      -e VLLM_EXL3_PREFILL_TRELLIS=1 -e VLLM_EXL3_PREFILL_BLOCK_M=64 -e VLLM_EXL3_PREFILL_CAPACITY=1024
      -e B12X_EXL3_BF16_EPILOGUE=0 -e B12X_EXL3_BF16_GEMV=1)
RES="$T/RESULTS.txt"
: >"$RES"
pass=0; fail=0
rec() { if [ "$2" = ok ]; then pass=$((pass+1)); else fail=$((fail+1)); fi; echo "$2 $1 ${3:-}" | tee -a "$RES"; }
cpu_run() {  # image, then args after the image
  local img=$1; shift
  timeout 1800 docker run --rm --network none -e CUDA_VISIBLE_DEVICES= --user "$(id -u):$(id -g)" \
    -e HOME=/tmp "${ENVS[@]}" -v "$H":/h:ro -v "$T":/t -v "$MR":/model-repo:ro \
    --entrypoint python3 "$img" "$@"
}
echo "== CPU tests $(date -u +%FT%TZ)" | tee -a "$RES"

# 1. import/signature/source checks
for e in v091 v070; do
  cpu_run "${IMG[$e]}" /t/check_signatures.py "$e" "/t/out-signatures-$e.json" >"$T/signatures-$e.log" 2>&1
  rc=$?; [ $rc = 0 ] && rec "signatures-$e" ok || rec "signatures-$e" FAIL "rc=$rc"
done
# 2. full dry run (fixtures, comparison self-check, CPU weight-slice probe) in both images
for e in v091 v070; do
  cpu_run "${IMG[$e]}" /h/moe_plan_eq.py --engine "$e" --out "/t/out-dry-$e.json" --dry >"$T/dry-$e.log" 2>&1
  rc=$?
  chk=$(python3 -c "
import json; d=json.load(open('$T/out-dry-$e.json')); s=d['dry_selfcheck']
print('ok' if d['status']=='complete' and s['pass'] and s['membership_matches_served_logs'] and s['weights_probe_layers']==6 else 'FAIL', 'fixtures=%d'%s['n_fixtures'], 'membership=%s'%s['membership_matches_served_logs'], 'probe=%s'%s['weights_probe_reached_prepare'])" 2>&1)
  [ $rc = 0 ] && [ "${chk%% *}" = ok ] && rec "dry-$e" ok "$chk" || rec "dry-$e" FAIL "rc=$rc $chk"
done
# 3. smoke dry (v091): one layer, own slice only
cpu_run "${IMG[v091]}" /h/moe_plan_eq.py --engine v091 --out /t/out-dry-smoke-v091.json --dry --smoke >"$T/dry-smoke-v091.log" 2>&1
rc=$?
chk=$(python3 -c "
import json; d=json.load(open('$T/out-dry-smoke-v091.json'))
ok = d['status']=='complete' and d['dry_selfcheck']['pass'] and list(d['slices'])==['tp2r0'] and list(d['slices']['tp2r0']['layers'])==['3']
print('ok' if ok else 'FAIL', 'fixtures=%d'%d['dry_selfcheck']['n_fixtures'])")
[ $rc = 0 ] && [ "${chk%% *}" = ok ] && rec "dry-smoke-v091" ok "$chk" || rec "dry-smoke-v091" FAIL "rc=$rc $chk"
# 4. soft deadline -> partial result written, rc 4
cpu_run "${IMG[v091]}" /h/moe_plan_eq.py --engine v091 --out /t/out-dry-deadline-v091.json --dry --dry-skip-weights --deadline-s 12 >"$T/dry-deadline-v091.log" 2>&1
rc=$?
chk=$(python3 -c "
import json; d=json.load(open('$T/out-dry-deadline-v091.json'))
n=sum(len(l['fixtures']) for s in d['slices'].values() for l in s['layers'].values())
print('ok' if d['status']=='partial' and d['stopped_reason']=='soft_deadline' and n>0 else 'FAIL', 'fixtures_done=%d'%n)")
[ $rc = 4 ] && [ "${chk%% *}" = ok ] && rec "deadline-partial" ok "$chk" || rec "deadline-partial" FAIL "rc=$rc $chk"
# 5. wrong image for the engine -> refused (rc 3), nothing measured
cpu_run "${IMG[v091]}" /h/moe_plan_eq.py --engine v070 --out /t/out-dry-wrongimage.json --dry >"$T/dry-wrongimage.log" 2>&1
rc=$?
[ $rc = 3 ] && grep -q "does not match engine" "$T/out-dry-wrongimage.json" && rec "wrong-image-refused" ok || rec "wrong-image-refused" FAIL "rc=$rc"
# 6. E-version comparison on the two dry results (inputs must be bit-identical across images)
cpu_run "${IMG[v091]}" /h/compare_eversion.py /t/out-dry-v091.json /t/out-dry-v070.json /t/out-eversion-dry.json >"$T/eversion-dry.log" 2>&1
rc=$?
chk=$(python3 -c "
import json; d=json.load(open('$T/out-eversion-dry.json'))
ok = all(s['n_common']==54 and s['n_input_mismatch']==0 and all(p['row_bitequal_frac']==1.0 for p in s['plans'].values()) for s in d['slices'].values())
print('ok' if ok else 'FAIL', {k:(v['n_common'],v['n_input_mismatch']) for k,v in d['slices'].items()})")
[ $rc = 0 ] && [ "${chk%% *}" = ok ] && rec "eversion-dry" ok "$chk" || rec "eversion-dry" FAIL "rc=$rc $chk"
# 7. wrapper gates on a scratch copy (fake docker / nvidia-smi)
"$T/test_wrapper.sh" >/dev/null 2>&1
rc=$?
[ $rc = 0 ] && rec "wrapper" ok "$(tail -1 "$T/out-wrapper.txt")" || rec "wrapper" FAIL "$(tail -1 "$T/out-wrapper.txt")"

echo "== $pass passed, $fail failed" | tee -a "$RES"
[ "$fail" = 0 ]
