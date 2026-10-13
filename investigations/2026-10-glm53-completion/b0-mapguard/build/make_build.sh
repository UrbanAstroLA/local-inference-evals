#!/bin/bash
# Regenerate the hook copies from the pinned B0 files, write the Dockerfile with the resulting hashes,
# build the derived image FROM B0 (by id) and record its id.  CPU only; no GPU, no server.
HERE=$(cd "$(dirname "$0")" && pwd)
B0=sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548
TAG=local/glm53-v0.9.1-b0-mapguard:local
B0TAG=local/glm53-v0.9.1-bundle-d1234:local
# BuildKit cannot resolve FROM <image id>; build FROM the B0 tag after checking it IS the B0 id
[ "$(docker image inspect "$B0TAG" --format '{{.Id}}' 2>/dev/null)" = "$B0" ] || { echo "ABORT: $B0TAG is not $B0"; exit 1; }
python3 "$HERE/../guard/make_patch.py" || exit 1
cd "$HERE" || exit 1
H() { sha256sum "files/vllm/$1" | cut -d' ' -f1; }
cat > Dockerfile <<EOF
# B0 + observe-only mapping guard (PLAN.md amendment 17).  Inert unless GLM53_MAPGUARD=device.
# Base = checkpoint B0 ($B0TAG, verified to be image $B0 before the build; release 0.9.1 + D1-D4).  Adds vllm/mapguard_hooks.py and hook lines in
# three files (mapguard.diff); every hook is behind _MAPGUARD_ON.  B0's computation is otherwise unchanged.
FROM $B0TAG
ARG VP=/usr/local/lib/python3.12/dist-packages/vllm
# Gate 0: the replaced files are B0's.
RUN printf '%s\n' \\
    "9104be37c8cb876a436f5fac11401c4abdb254529a6c37f3dca8121f41a5b61d  \${VP}/model_executor/layers/sparse_attn_indexer_kpool.py" \\
    "6e74c42cf4385bb0451243098ec8b4a84005675fa776169fddb2fb230bab0157  \${VP}/v1/attention/backends/mla/b12x_mla_sparse.py" \\
    "e405e84eb3674b04bc6251b2736fe0dce8f66a907123c82dd31b4c2ebc73841f  \${VP}/v1/core/sched/scheduler.py" \\
  | sha256sum -c - && ! test -e \${VP}/mapguard_hooks.py
COPY files/vllm/mapguard_hooks.py \${VP}/mapguard_hooks.py
COPY files/vllm/model_executor/layers/sparse_attn_indexer_kpool.py \${VP}/model_executor/layers/sparse_attn_indexer_kpool.py
COPY files/vllm/v1/attention/backends/mla/b12x_mla_sparse.py \${VP}/v1/attention/backends/mla/b12x_mla_sparse.py
COPY files/vllm/v1/core/sched/scheduler.py \${VP}/v1/core/sched/scheduler.py
RUN printf '%s\n' \\
    "$(H mapguard_hooks.py)  \${VP}/mapguard_hooks.py" \\
    "$(H model_executor/layers/sparse_attn_indexer_kpool.py)  \${VP}/model_executor/layers/sparse_attn_indexer_kpool.py" \\
    "$(H v1/attention/backends/mla/b12x_mla_sparse.py)  \${VP}/v1/attention/backends/mla/b12x_mla_sparse.py" \\
    "$(H v1/core/sched/scheduler.py)  \${VP}/v1/core/sched/scheduler.py" \\
  | sha256sum -c - \\
 && python3 -m py_compile \${VP}/mapguard_hooks.py \${VP}/model_executor/layers/sparse_attn_indexer_kpool.py \${VP}/v1/attention/backends/mla/b12x_mla_sparse.py \${VP}/v1/core/sched/scheduler.py \\
 && test "\$(grep -c "mapguard" \${VP}/model_executor/layers/sparse_attn_indexer_kpool.py)" = 8 \\
 && test "\$(grep -c "mapguard" \${VP}/v1/attention/backends/mla/b12x_mla_sparse.py)" = 4 \\
 && test "\$(grep -c "mapguard" \${VP}/v1/core/sched/scheduler.py)" = 2 \\
 && grep -q "col511-fix" \${VP}/model_executor/layers/sparse_attn_indexer_kpool.py
LABEL io.local.base="B0 $B0" io.local.purpose="observe-only mapping guard, PLAN.md amendment 17" \\
      io.local.patch="mapguard.diff $(sha256sum mapguard.diff | cut -c1-16)"
EOF
docker build -t "$TAG" . > build.log 2>&1; rc=$?
tail -n 5 build.log
[ $rc -eq 0 ] || exit $rc
ID=$(docker image inspect "$TAG" --format '{{.Id}}')
PARENT=$(docker image inspect "$TAG" --format '{{index .RootFS.Layers 0}}')
# ancestry: every B0 layer is a prefix of the derived image's layers
python3 - "$B0" "$TAG" <<'PY' || { echo "ABORT: derived image does not extend B0's layers"; exit 1; }
import json, subprocess, sys
L = lambda i: json.loads(subprocess.check_output(["docker", "image", "inspect", i, "--format", "{{json .RootFS.Layers}}"]))
b, d = L(sys.argv[1]), L(sys.argv[2])
sys.exit(0 if d[:len(b)] == b and len(d) > len(b) else 1)
PY
echo "$TAG $ID" | tee image-id.txt
