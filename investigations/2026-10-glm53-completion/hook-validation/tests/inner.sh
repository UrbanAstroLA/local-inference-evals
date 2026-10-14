#!/bin/bash
# Runs INSIDE a CPU container of the B1 image (--network none, no --gpus). Installs the hooked copies over the
# container's (ephemeral) site-packages, then runs one test. Args: <variant: hooked|orig> <test.py> [args]
VAR=$1; shift
P=/usr/local/lib/python3.12/dist-packages/vllm
if [ "$VAR" = hooked ] || [ "$VAR" = hooked-mapguard ]; then
  SRC=/hv/hook/hooked-b1; [ "$VAR" = hooked-mapguard ] && SRC=/hv/hook/hooked-b1-mapguard
  cp /hv/hook/forcewin_hooks.py $P/forcewin_hooks.py || exit 3
  ( cd $SRC && find . -name '*.py' ) | while read -r f; do cp "$SRC/$f" "$P/$f" || exit 3; done
  find $P -name '__pycache__' -path '*worker/gpu*' -prune -exec rm -rf {} + 2>/dev/null
fi
cd /hv/tests && exec python3 "$@"
