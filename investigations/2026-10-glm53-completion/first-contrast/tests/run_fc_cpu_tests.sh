#!/bin/bash
# CPU-only hook tests for the first-contrast envelope: the attempt-2 image (forcehook-mg2, the validated hook installed)
# as a CPU container (--network none, no --gpus, CUDA_VISIBLE_DEVICES=, TRITON_INTERPRET=1), the validated scaffolding
# (hook-validation/tests) mounted read-only. No image is built; the hook is not changed. No `set -e`.
# usage: run_fc_cpu_tests.sh [fast|long|all]   -> tests/out/test_fc_hook_<part>.log
FC=<runs>/first-contrast
HV=<runs>/hook-validation
IMGF=sha256:ac05230bd94b3114cb6fb202b06402f9e2d45a61b20f041f314e045f967e97ee   # local/glm53-v0.9.1-b1-forcehook-mg2:local
OUT=$FC/tests/out; mkdir -p "$OUT"
for PART in "${@:-fast long}"; do
  echo "== test_fc_hook $PART $(date -u +%FT%TZ)"
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e TRITON_INTERPRET=1 -e FW_THREADS=${FW_THREADS:-8} \
    -v $HV/tests:/hv/tests:ro -v $FC/tests:/fct:ro --entrypoint sh "$IMGF" -c \
    "cmp -s /usr/local/lib/python3.12/dist-packages/vllm/forcewin_hooks.py /dev/null; sha256sum /usr/local/lib/python3.12/dist-packages/vllm/forcewin_hooks.py; cd /hv/tests && python3 /fct/test_fc_hook.py $PART" \
    > "$OUT/test_fc_hook_$PART.log" 2>&1
  RC=$?
  grep -E "^(PASS|FAIL)|checks passed|Traceback|Error|forcewin_hooks.py" "$OUT/test_fc_hook_$PART.log" | grep -v Warning | cut -c1-220
  echo "rc=$RC" | tee -a "$OUT/test_fc_hook_$PART.log"
done
