# B1 + inert mapguard (d7/step2) + inert local-forcewin hook, attempt 2 (PA2-1/PA2-2; hook-validation/ATTEMPT2-PREP.md).
# Base verified to be sha256:4d7bd73e6f2ef6485289dcb5bcb2f2a65f1846a5959cde7cbd9c275471c72e16 before the build. Every forcewin call site is behind _FW_ON (env FORCEWIN=on).
FROM local/glm53-v0.9.1-b1-mapguard:local
ARG VP=/usr/local/lib/python3.12/dist-packages/vllm
# Gate 0: the replaced files are the B1-mapguard image's (make_patch.py PRE['b1-mapguard']).
RUN printf '%s\n' \
    "101e456b28fbebd0cb73e3f43ae38b4726844a43ee96ec61239bbe455853205f  ${VP}/v1/worker/gpu/model_runner.py" \
    "308d4171f892f27231655289ceb03cd4566533c7d26b8f98bd62ef0a34f986f0  ${VP}/v1/worker/gpu/spec_decode/rejection_sampler.py" \
    "76070739447fc9f7563ef164ef80c0777208d1ea07dec3a6f6d2a0755188d7ae  ${VP}/v1/worker/gpu/sample/sampler.py" \
    "00989c503028216cb349b506ccd7d4dc752bc9a54425a25b901c4905d95f5e0b  ${VP}/v1/attention/backends/mla/indexer.py" \
    "30ff8bdea2bfba185bf0bf5b591e7999d451baad12238761681f60d56b55b35f  ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py" \
  | sha256sum -c - && ! test -e ${VP}/forcewin_hooks.py
COPY files/vllm/forcewin_hooks.py ${VP}/forcewin_hooks.py
COPY files/vllm/v1/worker/gpu/model_runner.py ${VP}/v1/worker/gpu/model_runner.py
COPY files/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py ${VP}/v1/worker/gpu/spec_decode/rejection_sampler.py
COPY files/vllm/v1/worker/gpu/sample/sampler.py ${VP}/v1/worker/gpu/sample/sampler.py
COPY files/vllm/v1/attention/backends/mla/indexer.py ${VP}/v1/attention/backends/mla/indexer.py
COPY files/vllm/model_executor/layers/sparse_attn_indexer_kpool.py ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py
RUN printf '%s\n' \
    "15e93e236067567ffb9db7116986e8b732ef679e672c5b12b9d78f5884899d83  ${VP}/forcewin_hooks.py" \
    "8d276b3aac4bd6b5b2ef2c1f4b501140a4f1cd4e0dd1eb95f5a48d93e3a81d65  ${VP}/v1/worker/gpu/model_runner.py" \
    "f5a875ef0af0f8e5ceda52d1272c56e2badd07893192f44a0421183d29f75539  ${VP}/v1/worker/gpu/spec_decode/rejection_sampler.py" \
    "bbb1df8171c4750bea4f89ff76277a54b4b839caa25ef57d59313b2ce1e146c7  ${VP}/v1/worker/gpu/sample/sampler.py" \
    "6a1ec616ddbdecd6467e25a04bf012b50efb373dcfdf39f7fba6f8dc87980b86  ${VP}/v1/attention/backends/mla/indexer.py" \
    "f570b068d5515a16d013d258f9db6c5a816cd5ec8ff90d04c09d3f19ee1a0cbe  ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py" \
  | sha256sum -c - \
 && python3 -m py_compile ${VP}/forcewin_hooks.py ${VP}/v1/worker/gpu/model_runner.py ${VP}/v1/worker/gpu/spec_decode/rejection_sampler.py ${VP}/v1/worker/gpu/sample/sampler.py ${VP}/v1/attention/backends/mla/indexer.py ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py  \
 && test "$(grep -c local-forcewin ${VP}/v1/worker/gpu/model_runner.py)" = 12 \
 && test "$(grep -c local-forcewin ${VP}/v1/worker/gpu/spec_decode/rejection_sampler.py)" = 7 \
 && test "$(grep -c local-forcewin ${VP}/v1/worker/gpu/sample/sampler.py)" = 6 \
 && test "$(grep -c local-forcewin ${VP}/v1/attention/backends/mla/indexer.py)" = 5 \
 && test "$(grep -c local-forcewin ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py)" = 5 \
 && test "$(grep -c local-mapguard ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py)" = 9 \
 && test "$(grep -c local-d7 ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py)" = 2
LABEL io.local.base="B1-mapguard sha256:4d7bd73e6f2ef6485289dcb5bcb2f2a65f1846a5959cde7cbd9c275471c72e16" io.local.purpose="forced-history hook validation attempt 2, PLAN.md amendment 21"
