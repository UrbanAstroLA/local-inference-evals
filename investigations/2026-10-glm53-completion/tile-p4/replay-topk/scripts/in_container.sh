#!/bin/sh
# Runs INSIDE one stock image (launched by runner/run_replay_topk.sh).  Mounts: /s scripts (ro), /o this image's output
# directory, /cc compile caches (B12X_COMPILE_CACHE_DIR, DG_JIT_CACHE_DIR, TRITON_CACHE_DIR point into it).
#   in_container.sh VERSION SOFT_S
VERSION=$1; SOFT=$2
T0=$(date +%s)
umask 022
echo "== replay-topk $VERSION soft ${SOFT}s"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
python3 /s/replay.py --version "$VERSION" --out /o --deadline-s "$SOFT"
RC=$?
echo "replay rc=$RC"
echo "in-container elapsed $(( $(date +%s) - T0 ))s"
exit $RC
