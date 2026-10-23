#!/bin/bash
# Builds the two 22ao images from the local stock 0.9.1 image. No network: --network none (base is local).
set -u
cd "$(dirname "$0")" || exit 2
docker image inspect ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07 >/dev/null || { echo "base image missing"; exit 3; }
for arm in A C; do
  tag=local/glm53-v091-22ao-${arm,,}:local
  docker build --network none --pull=false -f Dockerfile.$arm -t "$tag" . > build-$arm.log 2>&1
  rc=$?; echo "arm $arm build rc=$rc"
  [ $rc = 0 ] && docker image inspect "$tag" --format '{{.Id}}' > image-id-$arm.txt && cat image-id-$arm.txt
done
