#!/bin/bash
# Final CPU evidence run for the freeze (no GPU; CPU containers --network none; no sudo). No `set -e`.
FC=$(cd "$(dirname "$0")/.." && pwd)
cd "$FC" || exit 1
echo "== conv-a2"; ( cd conv-a2 && ./crun.sh conv_a2.py > conv_a2.out 2>&1; echo "rc=$?" >> conv_a2.out; head -3 conv_a2.out )
echo "== units"; docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e CUDA_VISIBLE_DEVICES= -v "$FC":/fc:ro \
  --entrypoint python3 sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a /fc/tests/test_fc_units.py 2>&1 \
  | grep -v cpp_ext > tests/out/test_fc_units.log; tail -1 tests/out/test_fc_units.log
echo "== plan"; rm -rf tests/out/plan-dryrun; python3 runner/client_fc.py tests/out/plan-dryrun 0 --plan > tests/out/plan-dryrun.log 2>&1
python3 runner/plan_check.py tests/out/plan-dryrun/plan.json > tests/out/plan-check.log 2>&1; tail -1 tests/out/plan-check.log
python3 tests/test_plan_fc.py tests/out/plan-dryrun/plan.json > tests/out/plan-tamper-tests.log 2>&1; tail -1 tests/out/plan-tamper-tests.log
echo "== synthetic"; rm -f tests/out/synth-batch-*.txt
V1="good null uninformative noisy g-wmax g-abspos g-cache"; V2="g-attrib g-align g-cross g-regime g-content g-clean g-cvmig"
V3="g-cvslot g-cvbook a-ps a-guard a-incomplete a-c3"
for B in "$V1" "$V2" "$V3"; do
  for V in $B; do tests/run_synth.sh $V > tests/out/synth-batch-$V.txt 2>&1 & done; wait
done
python3 tests/check_synth.py tests/out > tests/out/synth-check.log 2>&1; tail -1 tests/out/synth-check.log
echo "== gate step"; tests/test_gate_step.sh > /dev/null 2>&1; tail -1 tests/out/gate-step-tests.log
echo "== hook fast"; tests/run_fc_cpu_tests.sh fast > tests/out/run_fc_cpu_tests_fast.console 2>&1; grep "checks passed" tests/out/test_fc_hook_fast.log
echo "== done $(date -u +%FT%TZ)"
