#!/bin/bash
# Run a replay script in a CPU container of the serving image: no network, no GPU, host uid, read-only inputs.
# usage: run_cpu.sh SCRIPT [ARGS...]     (SCRIPT relative to retry/replay)
HERE=$(cd "$(dirname "$0")" && pwd)
FC=$(cd "$HERE/../.." && pwd)
INV=$(cd "$FC/.." && pwd)
IMG=$(docker image inspect local/glm53-v0.9.1-b1-forcehook-mg2:local --format '{{.Id}}')
case "$IMG" in sha256:ac05230bd94b*) ;; *) echo "ABORT: image id $IMG is not the pinned ac05230b..."; exit 2;; esac
exec docker run --rm --network none --name "fcretry-replay-$$" --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 ${REPLAY_ENV:-} \
  -v "$INV":/inv:ro -v "$HERE/out":/out -v "$HERE":/replay:ro -w /replay \
  --entrypoint python3 "$IMG" "/replay/$1" "${@:2}"
