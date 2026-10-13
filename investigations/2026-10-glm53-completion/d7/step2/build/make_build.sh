#!/bin/bash
# B1 guard image (PLAN.md amendment 18a step 2b; copy of the frozen b0-mapguard/build/make_build.sh with B1 as
# the base).  Regenerate the hook copies (B1's index-K file, B0's other two), write the Dockerfile with the
# resulting hashes, build FROM B1 (tag verified to be the B1 id) and record the id.  CPU only; no GPU, no server.
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a   # B1 = B0 + d7-fix.diff
TAG=local/glm53-v0.9.1-b1-mapguard:local
BASETAG=local/glm53-v0.9.1-b1-d7:local
# BuildKit cannot resolve FROM <image id>; build FROM the B1 tag after checking it IS the B1 id
[ "$(docker image inspect "$BASETAG" --format '{{.Id}}' 2>/dev/null)" = "$BASE" ] || { echo "ABORT: $BASETAG is not $BASE"; exit 1; }
python3 "$HERE/../guard/make_patch.py" || exit 1
cd "$HERE" || exit 1
H() { sha256sum "files/vllm/$1" | cut -d' ' -f1; }
cat > Dockerfile <<EOF
# B1 + observe-only mapping guard, B1 amendment (PLAN.md amendments 17, 18a).  Inert unless GLM53_MAPGUARD=device.
# Base = B1 ($BASETAG, verified to be image $BASE before the build; B0 + d7-fix.diff).  Adds vllm/mapguard_hooks.py and hook lines in
# three files (mapguard.diff); every hook is behind _MAPGUARD_ON.  B1's computation is otherwise unchanged.
FROM $BASETAG
ARG VP=/usr/local/lib/python3.12/dist-packages/vllm
# Gate 0: the replaced files are B1's.
RUN printf '%s\n' \\
    "146a7c6b525d56e17e56b5c55d81cb0c19bfe7e1cfe3304a4974c1047246c6e3  \${VP}/model_executor/layers/sparse_attn_indexer_kpool.py" \\
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
 && test "\$(grep -c "mapguard" \${VP}/model_executor/layers/sparse_attn_indexer_kpool.py)" = 9 \\
 && test "\$(grep -c "mapguard" \${VP}/v1/attention/backends/mla/b12x_mla_sparse.py)" = 4 \\
 && test "\$(grep -c "mapguard" \${VP}/v1/core/sched/scheduler.py)" = 2 \\
 && grep -q "col511-fix" \${VP}/model_executor/layers/sparse_attn_indexer_kpool.py \\
 && test "\$(grep -c "local-d7" \${VP}/model_executor/layers/sparse_attn_indexer_kpool.py)" = 2
LABEL io.local.base="B1 $BASE" io.local.purpose="observe-only mapping guard + B1 amendment, PLAN.md amendments 17/18a" \\
      io.local.patch="mapguard.diff $(sha256sum mapguard.diff | cut -c1-16)"
EOF
docker build -t "$TAG" . > build.log 2>&1; rc=$?
tail -n 5 build.log
[ $rc -eq 0 ] || exit $rc
ID=$(docker image inspect "$TAG" --format '{{.Id}}')
PARENT=$(docker image inspect "$TAG" --format '{{index .RootFS.Layers 0}}')
# ancestry: every B1 layer is a prefix of the derived image's layers
python3 - "$BASE" "$TAG" <<'PY' || { echo "ABORT: derived image does not extend B1's layers"; exit 1; }
import json, subprocess, sys
L = lambda i: json.loads(subprocess.check_output(["docker", "image", "inspect", i, "--format", "{{json .RootFS.Layers}}"]))
b, d = L(sys.argv[1]), L(sys.argv[2])
sys.exit(0 if d[:len(b)] == b and len(d) > len(b) else 1)
PY
echo "$TAG $ID" | tee image-id.txt
