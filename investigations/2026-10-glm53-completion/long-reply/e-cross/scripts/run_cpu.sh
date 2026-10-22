#!/usr/bin/env bash
set -u
# [E copy of bplus/scripts/run_cpu.sh: set -u added; /raw and /disc point at d091's prefix source (its raw bodies and its
# index, as d091's run_pipeline.sh mounted them); the package folder must be e-cross or a copy inside it.]
# CPU-only container wrapper for the E package: --network none, CUDA_VISIBLE_DEVICES empty, no --gpus, read-only
# inputs, the package folder as /work. Usage: run_cpu.sh ENGINE SCRIPT [ARGS...]   (ENGINE = v070 | v091 | b1)
# Never starts a model server; runs one python3 script and exits.
B=<runs>
W=$(cd "$(dirname "$0")/.." && pwd)        # the package folder this script belongs to
EC=$B/long-reply/e-cross
case "$W" in "$EC"|"$EC"/*) ;; *) echo "ABORT: package folder $W is not e-cross or inside it"; exit 2 ;; esac
case "${1:-}" in
  v070) IMG=sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629 ;;
  v091) IMG=sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07 ;;
  b1)   IMG=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a ;;
  *) echo "usage: run_cpu.sh v070|v091|b1 SCRIPT [ARGS...]"; exit 2 ;;
esac
ENGINE=$1; shift
SCRIPT=${1:?usage: run_cpu.sh ENGINE SCRIPT [ARGS...]}; shift
EXTRA=()
[ -n "${RUN_DIR:-}" ] && EXTRA+=(-v "$RUN_DIR":/run:ro)
docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e HOME=/tmp -e ENGINE="$ENGINE" \
  -e PYTHONDONTWRITEBYTECODE=1 ${BPLUS_ENV:+-e BPLUS_ENV="$BPLUS_ENV"} \
  --user "$(id -u):$(id -g)" --entrypoint python3 \
  -v "$B/long-reply/d091/PRIVATE/raw":/raw:ro \
  -v <models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1:/repo:ro \
  -v <gpqa-runs>:/gpqa:ro \
  -v "$B/d7/exposure/gpqa_prompt_tokens.json":/d7/gpqa_prompt_tokens.json:ro \
  -v "$B/long-reply/d091/out":/disc:ro \
  -v "$W":/work "${EXTRA[@]}" "$IMG" "/work/$SCRIPT" "$@"
rc=$?
echo "exit $rc"
exit $rc
