# B1 + observe-only mapping guard, B1 amendment (PLAN.md amendments 17, 18a).  Inert unless GLM53_MAPGUARD=device.
# Base = B1 (local/glm53-v0.9.1-b1-d7:local, verified to be image sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a before the build; B0 + d7-fix.diff).  Adds vllm/mapguard_hooks.py and hook lines in
# three files (mapguard.diff); every hook is behind _MAPGUARD_ON.  B1's computation is otherwise unchanged.
FROM local/glm53-v0.9.1-b1-d7:local
ARG VP=/usr/local/lib/python3.12/dist-packages/vllm
# Gate 0: the replaced files are B1's.
RUN printf '%s\n' \
    "146a7c6b525d56e17e56b5c55d81cb0c19bfe7e1cfe3304a4974c1047246c6e3  ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py" \
    "6e74c42cf4385bb0451243098ec8b4a84005675fa776169fddb2fb230bab0157  ${VP}/v1/attention/backends/mla/b12x_mla_sparse.py" \
    "e405e84eb3674b04bc6251b2736fe0dce8f66a907123c82dd31b4c2ebc73841f  ${VP}/v1/core/sched/scheduler.py" \
  | sha256sum -c - && ! test -e ${VP}/mapguard_hooks.py
COPY files/vllm/mapguard_hooks.py ${VP}/mapguard_hooks.py
COPY files/vllm/model_executor/layers/sparse_attn_indexer_kpool.py ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py
COPY files/vllm/v1/attention/backends/mla/b12x_mla_sparse.py ${VP}/v1/attention/backends/mla/b12x_mla_sparse.py
COPY files/vllm/v1/core/sched/scheduler.py ${VP}/v1/core/sched/scheduler.py
RUN printf '%s\n' \
    "a8e59fa53590253584ccb1057ddc57b1cbf3946a1b9c143a3f76ab7ac65aa663  ${VP}/mapguard_hooks.py" \
    "30ff8bdea2bfba185bf0bf5b591e7999d451baad12238761681f60d56b55b35f  ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py" \
    "7124975df27de3f31b7e5ae16d51c9863e33cab66b77ef7c9322e3a765c408a1  ${VP}/v1/attention/backends/mla/b12x_mla_sparse.py" \
    "755916f22df976c58d1dd68996482f12366dd49102275a6e21e89dab7bd4b108  ${VP}/v1/core/sched/scheduler.py" \
  | sha256sum -c - \
 && python3 -m py_compile ${VP}/mapguard_hooks.py ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py ${VP}/v1/attention/backends/mla/b12x_mla_sparse.py ${VP}/v1/core/sched/scheduler.py \
 && test "$(grep -c "mapguard" ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py)" = 9 \
 && test "$(grep -c "mapguard" ${VP}/v1/attention/backends/mla/b12x_mla_sparse.py)" = 4 \
 && test "$(grep -c "mapguard" ${VP}/v1/core/sched/scheduler.py)" = 2 \
 && grep -q "col511-fix" ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py \
 && test "$(grep -c "local-d7" ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py)" = 2
LABEL io.local.base="B1 sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a" io.local.purpose="observe-only mapping guard + B1 amendment, PLAN.md amendments 17/18a" \
      io.local.patch="mapguard.diff ecf89fae855ad0c9"
