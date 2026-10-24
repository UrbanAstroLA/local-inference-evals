#!/bin/bash
# Runs INSIDE a CPU container (--network none, no --gpus, CUDA_VISIBLE_DEVICES empty).
# hooked: copies the capture-p0 vllm files (mounted at /c/build/files/vllm) over the stage-0 image's ephemeral
# site-packages; image: runs in the built cap0 image as is.  Then runs one script; files a test writes under /o are
# handed back to the host user (HOST_UID:HOST_GID).
# Args: <variant: hooked|image> <script> [args]
VAR=$1; shift
P=/usr/local/lib/python3.12/dist-packages/vllm
if [ "$VAR" = hooked ]; then
  for f in p4_hooks.py cap0_hooks.py cap0_hooks_layout.py; do cp "/c/build/files/vllm/$f" "$P/$f" || exit 3; done
  rm -rf "$P/__pycache__/local_"* 2>/dev/null
fi
cd /c/tests || exit 2
python3 "$@"
RC=$?
[ -n "${HOST_UID:-}" ] && [ -d /o ] && chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" /o 2>/dev/null
exit $RC
