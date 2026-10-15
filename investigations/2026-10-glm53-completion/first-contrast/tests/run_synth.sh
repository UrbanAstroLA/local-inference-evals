#!/bin/bash
# Synthetic dry runs of the GATE-1 and final analyses (CPU container: B1 image, --network none, no GPU).
# usage: run_synth.sh VARIANT [VARIANT...]   -> tests/out/synth-<variant>/ (gate1.json, fc_result.json, logs)
# Each run: client --plan (the session's own request list) -> make_synth_fc.py -> gate_fc.py on a stage-A copy
# (hard links of the stage-A tags only, as gate_step.sh copies) -> analyze_fc.py on the full directory.
FC=<runs>/first-contrast
HV=<runs>/hook-validation
IMG=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a
OUT=$FC/tests/out
mkdir -p "$OUT"
PL=$OUT/plan-dryrun
if [ ! -s "$PL/plan.json" ]; then rm -rf "$PL"; python3 $FC/runner/client_fc.py "$PL" 0 --plan > "$OUT/plan-dryrun.log" 2>&1 || exit 2; fi
for V in "$@"; do
  D=$OUT/synth-$V
  rm -rf "$D"; mkdir -p "$D"
  docker run --rm --network none --user $(id -u):$(id -g) -e HOME=/tmp -e CUDA_VISIBLE_DEVICES= -e PYTHONHASHSEED=0 -v $FC:/fc:ro -v $HV/hook:/hv/hook:ro \
    -v "$D":/run -v "$PL":/plan:ro --entrypoint sh $IMG -c \
    "python3 /fc/tests/make_synth_fc.py /plan/plan.json /run/full $V && mkdir -p /run/full/gate1/forcewin /run/full/gate1/mapguard && \
     cd /run/full && grep -v -E '\"phase\": \"(rewarm|warm-s3|LC|LD|D[0-9]|PS|C3|LW3)' client.jsonl > gate1/client.jsonl; \
     for f in forcewin/*; do b=\$(basename \$f); case \$b in req_LW1-*|req_LW2-*|req_C1-*|req_C2-*|req_CV-*|snap_LW1-*|snap_LW2-*|snap_C1-*|snap_C2-*|snap_CV-*|fw_r*) ln \$f gate1/forcewin/\$b;; esac; done; \
     ln mapguard/* gate1/mapguard/; \
     python3 /fc/analysis/gate_fc.py /run/full/gate1 /fc > /run/gate1.log 2>&1; echo gate_rc=\$? >> /run/gate1.log; \
     python3 /fc/analysis/analyze_fc.py /run/full /fc > /run/analysis.log 2>&1; echo analysis_rc=\$? >> /run/analysis.log" \
    2>&1 | grep -v cpp_extension > "$D/make.log"
  echo "== $V: $(head -1 $D/gate1.log | cut -c1-200)"; grep -E "^FAIL" "$D/gate1.log" | cut -c1-60
  echo "   $(head -1 $D/analysis.log | cut -c1-220)"
done
