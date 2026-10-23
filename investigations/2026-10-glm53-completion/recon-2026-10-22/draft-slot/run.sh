#!/bin/bash
# CPU-only: stock tpurtell 0.9.1 image by digest, no network, no GPU (default runtime runc, no --gpus), read-only
# inputs, writes only to ./out. Usage: ./run.sh [quick]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
INV=<runs>
IMG=sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
[ -n "$HERE" ] && [ -d "$INV/b0-mapguard/reach" ] || { echo "path check failed"; exit 2; }
mkdir -p "$HERE/out"
Q=""; NAME=result.json
if [ "${1:-}" = "quick" ]; then Q="-e QUICK=1"; NAME=result-quick.json; fi
docker image inspect "$IMG" --format '{{.Id}}' > "$HERE/out/image-id.txt"
docker run --rm --network none --user "$(id -u):$(id -g)" -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void \
  -e TRITON_INTERPRET=1 -e HOME=/tmp $Q \
  -v "$INV":/inv:ro -v "$HERE":/w:ro -v "$HERE/out":/out --entrypoint python3 "$IMG" /w/harness.py "/out/$NAME"
echo "rc=$?"
