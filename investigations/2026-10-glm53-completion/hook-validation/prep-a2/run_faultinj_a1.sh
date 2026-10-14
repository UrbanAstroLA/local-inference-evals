#!/bin/bash
# PA2-21: offline fault injection on attempt-1 captures under the REVISED analysis (prep-a2/faultinj_a1.py).
# CPU only: CPU containers of the B1 image, --network none, no --gpus, CUDA_VISIBLE_DEVICES=, run-a1 read-only.
# Scratch trees (proxy copies of window dumps, ~5 GB) live on /data, outside the package, and are removed on every
# exit path. Outputs: prep-a2/faultinj-a1/<case>/verdict{,-a2}.{json,txt}, build.json, summary.{json,txt}.
# No `set -e`: each step's rc is checked.
HV=$(cd "$(dirname "$0")/.." && pwd)
IMG=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a   # local/glm53-v0.9.1-b1-d7:local
T=$(dirname "$HV")/tmp-faultinj-21g
OUT=$HV/prep-a2/faultinj-a1
trap 'rm -rf "$T"' EXIT
rm -rf "$T" "$OUT"; mkdir -p "$T" "$OUT" || exit 1
DR=(docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e CUDA_VISIBLE_DEVICES=
    -e PYTHONDONTWRITEBYTECODE=1 -v "$HV/run-a1":/run:ro -v "$HV/analysis":/a:ro)
"${DR[@]}" -v "$HV/prep-a2":/p:ro -v "$T":/t --entrypoint python3 "$IMG" /p/faultinj_a1.py build /t \
  2>&1 | grep -v cpp_extension > "$OUT/build.log"
[ -s "$T/build.json" ] || { echo "build failed"; tail "$OUT/build.log"; exit 1; }
cp "$T/build.json" "$OUT/"
CASES=$(python3 -c "import json; print(' '.join(json.load(open('$T/build.json'))['cases']))")
N=0
for C in $CASES; do
  mkdir -p "$OUT/$C"
  ( "${DR[@]}" -v "$T":/t:ro -v "$OUT/$C":/o -e HV_ANALYSIS_OUT=/o --entrypoint python3 "$IMG" \
      /a/analyze_hookval.py "/t/$C" 2>&1 | grep -v cpp_extension > "$OUT/$C/console.txt" ) &
  N=$((N + 1)); [ $((N % 5)) -eq 0 ] && wait
done
wait
python3 "$HV/prep-a2/faultinj_a1.py" summarize "$OUT"
