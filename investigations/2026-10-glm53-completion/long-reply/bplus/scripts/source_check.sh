#!/usr/bin/env bash
# B+ source reading of the three engine images (PROPOSAL.md section 2.3). CPU-only containers (--network none,
# CUDA_VISIBLE_DEVICES empty, no --gpus); reads installed Python sources, prints hashes and the lines the proposal cites.
# Usage: source_check.sh > checks/source-check.txt
declare -A IM=([v070]=sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629
               [v091]=sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
               [b1]=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a)
V=vllm
FILES="entrypoints/openai/completion/protocol.py entrypoints/openai/completion/serving.py
entrypoints/openai/chat_completion/protocol.py entrypoints/openai/chat_completion/serving.py
v1/engine/input_processor.py sampling_params.py config/model.py v1/sample/sampler.py v1/sample/rejection_sampler.py
v1/sample/ops/topk_topp_sampler.py v1/core/sched/scheduler.py v1/core/kv_cache_coordinator.py v1/core/kv_cache_utils.py
config/cache.py"
for k in v070 v091 b1; do
  echo "################ $k ${IM[$k]}"
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= --entrypoint bash "${IM[$k]}" -c "
    cd /usr/local/lib/python3.12/dist-packages/$V || exit 3
    python3 -c 'import vllm; print(\"vllm\", vllm.__version__)' 2>/dev/null | tail -1
    for f in $(echo $FILES); do sha256sum \$f; done
    sha256sum /usr/local/bin/glm53-entrypoint /usr/local/bin/glm53-release-warmup.py
    echo '--- [S1] /v1/completions prompt types (token-id lists accepted)'
    grep -n -A7 '^    prompt: ($' entrypoints/openai/completion/protocol.py | head -8
    echo '--- [S2] per-request seed, temperature, top_p, max_tokens, stop_token_ids passed into SamplingParams (completion)'
    sed -n '/    def to_sampling_params/,/^    @model_validator/p' entrypoints/openai/completion/protocol.py | grep -nE 'seed=|temperature=|top_p=|max_tokens=|stop_token_ids=|stop=|skip_special_tokens=|thinking_token_budget=|repetition_detection='
    echo '--- [S3] completion fields whose default changes nothing (return_token_ids output-only; repetition_detection / thinking_token_budget default None)'
    grep -nE '^    (return_token_ids|repetition_detection|thinking_token_budget|skip_special_tokens|stop_token_ids|ignore_eos|min_tokens|top_k|min_p|repetition_penalty|n):' entrypoints/openai/completion/protocol.py
    grep -n -A2 'return_token_ids: bool | None = Field' entrypoints/openai/completion/protocol.py | head -3
    echo '    to_sampling_params mentions return_token_ids:' \$(sed -n '/    def to_sampling_params/,/^    @model_validator/p' entrypoints/openai/completion/protocol.py | grep -c return_token_ids)
    grep -nE 'request.return_token_ids' entrypoints/openai/completion/serving.py | head -8
    echo '--- [S4] chat vs completion to_sampling_params: lines that differ (sampling-relevant only?)'
    diff <(sed -n '/    def to_sampling_params/,/^    @model_validator/p' entrypoints/openai/completion/protocol.py) <(sed -n '/    def to_sampling_params/,/^    @model_validator/p' entrypoints/openai/chat_completion/protocol.py) | grep -E '^[<>]'
    echo '--- [S5] server defaults from generation_config, identical in both endpoints'
    grep -nE 'default_sampling_params = self.model_config.get_diff_sampling_param' entrypoints/openai/completion/serving.py entrypoints/openai/chat_completion/serving.py
    echo '--- [S6] stop tokens: engine input processor (shared by chat and completions) adds tokenizer EOS + generation_config eos_token_id'
    grep -nE 'update_from_generation_config|update_from_tokenizer' v1/engine/input_processor.py
    sed -n '/    def update_from_generation_config/,/    def update_from_tokenizer/p' sampling_params.py | grep -nE 'eos_ids|stop_token_ids|_all_stop_token_ids'
    echo '--- [S7] reasoning parser references in the completion endpoint (expected 0)'
    grep -ciE 'reasoning' entrypoints/openai/completion/serving.py
    echo '--- [S8] per-request torch.Generator used by both samplers (seeded requests)'
    grep -nE 'generators' v1/sample/rejection_sampler.py | head -3; grep -nE 'generators' v1/sample/ops/topk_topp_sampler.py | head -2
    echo '--- [S9] prefix-cache EAGLE back-off and DFlash boundary lookup'
    grep -n -B1 -A3 'Back off only while\|if self.use_eagle:' v1/core/sched/scheduler.py | head -8
    grep -n 'GLM53_DFLASH_BOUNDARY_LOOKUP' v1/core/kv_cache_coordinator.py
  " 2>&1
done
echo "################ generation_config.json (K3.25, identical blob in snapshots 701cd745 and 0490d2f7)"
M=<models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1/snapshots
sha256sum $M/701cd7456c13d87bf0147ad946f828a999afb59c/generation_config.json $M/0490d2f708b12145f6516555ab066aaeb401cd21/generation_config.json
python3 -c "import json;print(json.load(open('$M/701cd7456c13d87bf0147ad946f828a999afb59c/generation_config.json')))"
python3 -c "
import json; tj=json.load(open('$M/701cd7456c13d87bf0147ad946f828a999afb59c/tokenizer.json'))
print({a['id']: (a['content'], 'special' if a['special'] else 'not special') for a in tj['added_tokens'] if a['id'] in (154820, 154827, 154828, 154829, 154841, 154842)})
print('tokenizer_config eos_token:', json.load(open('$M/701cd7456c13d87bf0147ad946f828a999afb59c/tokenizer_config.json')).get('eos_token'))"
echo "################ launcher defaults: prefix caching and boundary lookup (stack start.sh)"
grep -nE '^ENABLE_PREFIX_CACHING=|^GLM53_DFLASH_BOUNDARY_LOOKUP=|^DFLASH_TOKENS=|^DECODE_CONTEXT_PARALLEL_SIZE=|^ENABLE_EXPERT_PARALLEL=|^CHAT_TEMPLATE=|enable-prompt-tokens-details' <stacks>/glm53-flash-v0.7.0/start.sh <stacks>/glm53-flash-v0.9.0/start.sh
echo "################ snapshot difference 701cd745 vs 0490d2f7 (files whose blob differs)"
diff <(cd $M/701cd7456c13d87bf0147ad946f828a999afb59c && find . \( -type l -o -type f \) | LC_ALL=C sort | while read -r f; do echo "$f -> $(basename "$(readlink "$f")")"; done) \
     <(cd $M/0490d2f708b12145f6516555ab066aaeb401cd21 && find . \( -type l -o -type f \) | LC_ALL=C sort | while read -r f; do echo "$f -> $(basename "$(readlink "$f")")"; done)
echo "done"
