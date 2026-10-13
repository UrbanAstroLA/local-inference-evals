# B1 = B0 + d7-fix.diff ONLY (PLAN.md amendment 18/18a step 2b): the D7 writer-side correction of the index-K
# prefill writer. Base = B0 (local/glm53-v0.9.1-bundle-d1234:local, verified to be image sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548 before the build).
FROM local/glm53-v0.9.1-bundle-d1234:local
ARG VP=/usr/local/lib/python3.12/dist-packages/vllm
RUN echo "9104be37c8cb876a436f5fac11401c4abdb254529a6c37f3dca8121f41a5b61d  ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py" | sha256sum -c -
COPY sparse_attn_indexer_kpool.py ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py
RUN echo "146a7c6b525d56e17e56b5c55d81cb0c19bfe7e1cfe3304a4974c1047246c6e3  ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py" | sha256sum -c - && python3 -m py_compile ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py
LABEL io.local.base="B0 sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548" io.local.purpose="B1 = B0 + D7 writer correction (PLAN.md amendment 18/18a)" \
      io.local.patch="d7-fix.diff cbdd2ada6f5a9932"
