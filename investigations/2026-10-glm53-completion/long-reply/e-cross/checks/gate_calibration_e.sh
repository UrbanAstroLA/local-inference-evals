#!/usr/bin/env bash
set -u
# [22ah: D2's run folder mounted read-only.]
# E gate calibration (checks/gate_calibration_e.py) in a CPU-only container: --network none, CUDA_VISIBLE_DEVICES empty,
# no --gpus, every reference folder mounted read-only; the package is /work (writes checks/gate-calibration.json).
# usage: checks/gate_calibration_e.sh > checks/gate-calibration.txt
H=$(cd "$(dirname "$0")" && pwd); PKG=$(dirname "$H")
EC=<runs>/long-reply/e-cross
case "$PKG" in "$EC") ;; *) echo "ABORT: unexpected package path $PKG"; exit 2 ;; esac
B=<runs>
docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  --user "$(id -u):$(id -g)" --entrypoint python3 \
  -v <gpqa-runs>:/gpqa:ro -v "$B/rep070/run-F":/refs/runF:ro -v "$B/long-reply/bplus/run":/refs/bplus_run:ro \
  -v "$B/long-reply/d091/run":/refs/d091_run:ro -v "$B/long-reply/d091b/run":/refs/d091b_run:ro -v <data-runs>/benchmarks:/refs/bench:ro -v "$PKG":/work \
  sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629 \
  /work/checks/gate_calibration_e.py /work/checks/gate-calibration.json
rc=$?
echo "exit $rc"
exit $rc
