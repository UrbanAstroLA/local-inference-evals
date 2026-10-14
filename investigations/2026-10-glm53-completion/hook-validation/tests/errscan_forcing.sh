#!/bin/bash
# PA2-3 supplement: rerun each forcing shard and count hook error records ("fw_error") the run wrote. CPU only:
# B1 image as a CPU container, --network none, no --gpus. Logs: tests/out-a2/errscan_forcing_<shard>.log
HV=<runs>/hook-validation
IMG=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a
OUT=$HV/tests/out-a2
for P in natural oracle poison reject:1 reject:2 reject:3 sched:7 seed rest; do
  ( docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e TRITON_INTERPRET=1 -e FORCEWIN=on -e FW_THREADS=2 \
      -e FORCEWIN_DIR=/tmp/fwerr -v $HV/hook:/hv/hook:ro -v $HV/tests:/hv/tests:ro \
      -v <runs>/d7/step2/guard:/hv/guard:ro --entrypoint /bin/bash "$IMG" -c \
      "/hv/tests/inner.sh hooked test_forcing.py $P; echo rc=\$?; echo fw_error_records=\$(cat /tmp/fwerr/*.jsonl 2>/dev/null | grep -c '\"fw_error\"'); cat /tmp/fwerr/*.jsonl 2>/dev/null | grep '\"fw_error\"' | head -3" \
      > "$OUT/errscan_forcing_${P/:/}.log" 2>&1 ) &
done; wait
grep -hE "checks passed|fw_error_records" $OUT/errscan_forcing_*.log
