#!/bin/bash
# Runs INSIDE a CPU container of the stock 0.9.1 image (--network none, no --gpus, CUDA_VISIBLE_DEVICES empty).
# hooked: installs the P4 files (build/files/vllm, mounted at /p4files) over the container's ephemeral site-packages;
# orig: leaves the stock files; image: runs in the built image as is.  Then runs one test.
# Args: <variant: hooked|orig|image> <dir: t (tile-p4/tests) | hv (hook-validation/tests)> <script> [args]
VAR=$1; D=$2; shift 2
P=/usr/local/lib/python3.12/dist-packages/vllm
if [ "$VAR" = hooked ]; then
  ( cd /p4files && find . -name '*.py' ) | while read -r f; do cp "/p4files/$f" "$P/$f" || exit 3; done
  find $P -name '__pycache__' \( -path '*worker/gpu*' -o -path '*quantization*' -o -path '*glm5next*' \) -prune -exec rm -rf {} + 2>/dev/null
fi
case $D in t) cd /t ;; hv) cd /hv/tests ;; *) echo "bad dir"; exit 2;; esac
exec python3 "$@"
