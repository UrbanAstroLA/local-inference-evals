#!/bin/bash
# Synthetic dry runs of GATE-1R and the final retry analysis (CPU container: B1 image, --network none, no GPU).
# usage: run_synth_fcr.sh VARIANT [VARIANT...]   -> retry/tests/out/synth-<variant>/ (gate1r.json, fcr_result.json)
# Each run: client_fcr --plan (the session's own request list) -> make_synth_fcr.py -> gate_fcr.py on a stage-A copy
# (hard links of exactly the files gate_step_fcr.sh copies) -> analyze_fcr.py on the full directory.
RETRY=$(cd "$(dirname "$0")/.." && pwd)
FC=$(cd "$RETRY/.." && pwd)
HV=<runs>/hook-validation
IMG=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a
OUT=$RETRY/tests/out
mkdir -p "$OUT"
PL=$OUT/plan-dryrun
if [ ! -s "$PL/plan.json" ]; then rm -rf "$PL"; python3 $RETRY/runner/client_fcr.py "$PL" 0 --plan > "$OUT/plan-dryrun.log" 2>&1 || exit 2; fi
for V in "$@"; do
  D=$OUT/synth-$V
  rm -rf "$D"; mkdir -p "$D"
  docker run --rm --network none --name "fcr-synth-$$" --user $(id -u):$(id -g) -e HOME=/tmp -e CUDA_VISIBLE_DEVICES= -e PYTHONHASHSEED=0 \
    -v $FC:/fc:ro -v $HV/hook:/hv/hook:ro -v "$D":/run -v "$PL":/plan:ro --entrypoint sh $IMG -c \
    "python3 /fc/retry/tests/make_synth_fcr.py /plan/plan.json /run/full $V && mkdir -p /run/full/gate1r/forcewin /run/full/gate1r/mapguard && \
     cd /run/full && python3 -c \"
import json
keep=('V0','LW1','LW2','warm-s1-a','T-s1-a','warm-s1-b','T-s1','LC1','LC2','C1','CV','warm-s2-a','T-s2-a','warm-s2-b','T-s2','C2','PS-a','PS-b')
out=open('gate1r/client.jsonl','w')
for l in open('client.jsonl'):
    d=json.loads(l)
    if d.get('kind')=='request' and d.get('phase') not in keep: continue
    if d.get('kind')=='phase' and d.get('tag') not in keep: continue
    out.write(l)
\"; \
     for f in forcewin/*; do b=\$(basename \$f); case \$b in req_LW1-*|req_LW2-*|req_LC1-*|req_LC2-*|req_C1-*|req_C2-*|req_CV-*|req_PS-a-*|req_PS-b-*|snap_LW1-*|snap_LW2-*|snap_LC1-*|snap_LC2-*|snap_C1-*|snap_C2-*|snap_CV-*|snap_PS-a-*|snap_PS-b-*|fw_r*) ln \$f gate1r/forcewin/\$b;; esac; done; \
     ln mapguard/* gate1r/mapguard/; \
     python3 /fc/retry/analysis/gate_fcr.py /run/full/gate1r /fc > /run/gate1r.log 2>&1; echo gate_rc=\$? >> /run/gate1r.log; \
     python3 /fc/retry/analysis/analyze_fcr.py /run/full /fc > /run/analysis.log 2>&1; echo analysis_rc=\$? >> /run/analysis.log" \
    2>&1 | grep -v cpp_extension > "$D/make.log"
  echo "== $V: $(head -1 $D/gate1r.log | cut -c1-200)"; grep -E "^FAIL" "$D/gate1r.log" | cut -c1-60
  echo "   $(head -1 $D/analysis.log | cut -c1-220)"
done
