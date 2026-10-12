# GLM-5.3-Flash 0.9.1 (tpurtell release) + minimal correctness bundle d124 (PLAN.md amendment 14 (2)). PREPARED; not run on a GPU.
# D1 column-511 correction (gate/freeze/candidate.diff, decode + prefill at DCP1); D2 kpool tail ring, variant U
# (positions to build_attn_metadata, own_block != 0 guard, circular mapping written into the persistent slot-mapping buffer);
# D4 routed-expert swiglu_limit passed to the B12x planner (projection-mixed) and compile_mixed_trellis (expert-mixed); D3 NOT included; 
# D5 workspace lock NOT included (left out, see PROPOSAL.md); D6 not separately included (unreachable under D2). No observers.
FROM ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
ARG VP=/usr/local/lib/python3.12/dist-packages/vllm
# Gate 0: the files being replaced are the release files the patches were generated from.
RUN printf '%s\n' \
    "e7935f2a9b8ecbc75410defff571d9940ad2abf511286bf0936d0c324936df39  ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py" \
    "67449677b787f37e31f69de97269097907ebda395879eb67d0b2ebb991a2f0dc  ${VP}/v1/attention/backends/mla/indexer.py" \
    "82968af5e104f99f8f9d15c5eb8c15c13640f7b6ab8e2bd1580b0c49e9cf9968  ${VP}/v1/worker/gpu/model_states/mamba_hybrid.py" \
    "3b4ca47cb1168bffebcfe3cf1db48daf86691c140cbf40f9c1f3aa6ad6a444d9  ${VP}/model_executor/layers/quantization/exl3.py" \
  | sha256sum -c -
COPY files/vllm/model_executor/layers/sparse_attn_indexer_kpool.py ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py
COPY files/vllm/v1/attention/backends/mla/indexer.py ${VP}/v1/attention/backends/mla/indexer.py
COPY files/vllm/v1/worker/gpu/model_states/mamba_hybrid.py ${VP}/v1/worker/gpu/model_states/mamba_hybrid.py
COPY files/vllm/model_executor/layers/quantization/exl3.py ${VP}/model_executor/layers/quantization/exl3.py
# Gates: byte-exact files, bytecode regenerated, markers present, no observer modules or hooks in the image.
RUN printf '%s\n' \
    "9104be37c8cb876a436f5fac11401c4abdb254529a6c37f3dca8121f41a5b61d  ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py" \
    "00989c503028216cb349b506ccd7d4dc752bc9a54425a25b901c4905d95f5e0b  ${VP}/v1/attention/backends/mla/indexer.py" \
    "049238f99b3c77cb23dc0abbbf25d140817c40c5abbf5cfe6a319f8f410ca085  ${VP}/v1/worker/gpu/model_states/mamba_hybrid.py" \
    "b40344e2b4bd583a670ddecc643de01afbfa0e35e6a960a259275dedf8e246cb  ${VP}/model_executor/layers/quantization/exl3.py" \
  | sha256sum -c - \
 && python3 -m py_compile ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py ${VP}/v1/attention/backends/mla/indexer.py ${VP}/v1/worker/gpu/model_states/mamba_hybrid.py ${VP}/model_executor/layers/quantization/exl3.py \
 && test "$(grep -c "col511-fix" ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py)" = 4 \
 && grep -q "torch.where(own_block != 0, slots, -1)" ${VP}/v1/attention/backends/mla/indexer.py \
 && grep -q "common_attn_metadata.slot_mapping.copy_(slot_mapping)" ${VP}/v1/attention/backends/mla/indexer.py \
 && test "$(grep -c "positions=input_batch.positions," ${VP}/v1/worker/gpu/model_states/mamba_hybrid.py)" = 1 \
 && test "$(grep -c "swiglu_limit=getattr(layer, \"swiglu_limit\", None)" ${VP}/model_executor/layers/quantization/exl3.py)" = 2 \
 && ! grep -q "_router_gemm_cublas_capable" ${VP}/model_executor/layers/fused_moe/router/gate_linear.py && echo "6fca8c7fba361ac79418dd56d6d3c281ef04f34ebfdd2044cb1d12e1fa9e70d7  ${VP}/model_executor/layers/fused_moe/router/gate_linear.py" | sha256sum -c - \
 && ! ls ${VP}/local_*.py 2>/dev/null \
 && ! grep -rqs "kpool_tailguard\|col511check\|verify13_capture\|col511-probe" ${VP}/model_executor ${VP}/v1 \
 && ! grep -q "lock_workspace" ${VP}/v1/worker/gpu/model_runner.py
LABEL io.local.bundle="d124" \
      io.local.patch="D1 col511 (candidate.diff 1ca405f2) + D2 tail ring U (tailfix-U.diff 9d5a7514) + D4 swiglu_limit (0091e194); D5/D6 excluded; combined-d124.diff 1a30a9f8161f0c4b" \
      io.local.base="ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07 (release 0.9.1)"
