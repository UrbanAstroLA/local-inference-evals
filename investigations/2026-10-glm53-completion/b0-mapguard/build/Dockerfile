# B0 + observe-only mapping guard (PLAN.md amendment 17).  Inert unless GLM53_MAPGUARD=device.
# Base = checkpoint B0 (local/glm53-v0.9.1-bundle-d1234:local, verified to be image sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548 before the build; release 0.9.1 + D1-D4).  Adds vllm/mapguard_hooks.py and hook lines in
# three files (mapguard.diff); every hook is behind _MAPGUARD_ON.  B0's computation is otherwise unchanged.
FROM local/glm53-v0.9.1-bundle-d1234:local
ARG VP=/usr/local/lib/python3.12/dist-packages/vllm
# Gate 0: the replaced files are B0's.
RUN printf '%s\n' \
    "9104be37c8cb876a436f5fac11401c4abdb254529a6c37f3dca8121f41a5b61d  ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py" \
    "6e74c42cf4385bb0451243098ec8b4a84005675fa776169fddb2fb230bab0157  ${VP}/v1/attention/backends/mla/b12x_mla_sparse.py" \
    "e405e84eb3674b04bc6251b2736fe0dce8f66a907123c82dd31b4c2ebc73841f  ${VP}/v1/core/sched/scheduler.py" \
  | sha256sum -c - && ! test -e ${VP}/mapguard_hooks.py
COPY files/vllm/mapguard_hooks.py ${VP}/mapguard_hooks.py
COPY files/vllm/model_executor/layers/sparse_attn_indexer_kpool.py ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py
COPY files/vllm/v1/attention/backends/mla/b12x_mla_sparse.py ${VP}/v1/attention/backends/mla/b12x_mla_sparse.py
COPY files/vllm/v1/core/sched/scheduler.py ${VP}/v1/core/sched/scheduler.py
RUN printf '%s\n' \
    "3fa979c8fcb3588275ca92a17db4a633b21977382823adff36bb158e709ac065  ${VP}/mapguard_hooks.py" \
    "0ac09f5c270ee485ed388f6c2e1cfa8dbac1b082091b0589b440f5e94fa68e0b  ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py" \
    "7124975df27de3f31b7e5ae16d51c9863e33cab66b77ef7c9322e3a765c408a1  ${VP}/v1/attention/backends/mla/b12x_mla_sparse.py" \
    "755916f22df976c58d1dd68996482f12366dd49102275a6e21e89dab7bd4b108  ${VP}/v1/core/sched/scheduler.py" \
  | sha256sum -c - \
 && python3 -m py_compile ${VP}/mapguard_hooks.py ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py ${VP}/v1/attention/backends/mla/b12x_mla_sparse.py ${VP}/v1/core/sched/scheduler.py \
 && test "$(grep -c "mapguard" ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py)" = 8 \
 && test "$(grep -c "mapguard" ${VP}/v1/attention/backends/mla/b12x_mla_sparse.py)" = 4 \
 && test "$(grep -c "mapguard" ${VP}/v1/core/sched/scheduler.py)" = 2 \
 && grep -q "col511-fix" ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py
LABEL io.local.base="B0 sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548" io.local.purpose="observe-only mapping guard, PLAN.md amendment 17" \
      io.local.patch="mapguard.diff 30c9e031cbf327d6"
