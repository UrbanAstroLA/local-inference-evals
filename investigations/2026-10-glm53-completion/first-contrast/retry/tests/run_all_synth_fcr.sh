#!/bin/bash
# All synthetic retry variants, 4 CPU containers at a time; then the expectation check.
HERE=$(cd "$(dirname "$0")" && pwd)
VS=(good null uninformative noisy g-wmax g-abspos g-cache g-attrib g-align g-cross g-regime g-content g-clean g-cvmig g-cvslot g-cvbook a-ps a-guard a-incomplete a-c3 r-cache-model r-cache-unsafe r-cache-attrib r-ps-cover r-ps-ident r-f6-page r-balance r-ps-conv r-ps-ssm r-ps-nonid r-ps-nonid-ok r-f6-nd r-guard-stop)
N=${PAR:-4}; i=0
for v in "${VS[@]}"; do
  "$HERE/run_synth_fcr.sh" "$v" > "$HERE/out/synth-batch-$v.txt" 2>&1 &
  i=$((i+1)); if (( i % N == 0 )); then wait; fi
done
wait
python3 "$HERE/check_synth_fcr.py" "$HERE/out" | tee "$HERE/out/synth-check.log"
