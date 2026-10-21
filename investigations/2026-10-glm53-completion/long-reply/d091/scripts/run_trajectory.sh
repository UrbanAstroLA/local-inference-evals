#!/usr/bin/env bash
# CPU-only, no network, no GPU. Usage: run_trajectory.sh [capped|all]
D=<runs>/long-reply/discovery
RAW=<runs>/rep070/run-F/raw
REPO=<models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1
IMG=local/glm53-v0.7.0-kpoolring:local
MODE=${1:-all}
if [ "$MODE" = capped ]; then ENVS="-e ONLY=capped -e TAG=capped"; else ENVS="-e TAG=all"; fi
docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e HOME=/tmp $ENVS --user "$(id -u):$(id -g)" \
  --entrypoint python3 -v "$RAW":/raw:ro -v "$REPO":/repo:ro -v "$D":/work "$IMG" /work/scripts/trajectory.py
rc=$?
echo "exit $rc"
exit $rc
