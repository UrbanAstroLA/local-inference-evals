#!/usr/bin/env python3
"""Assign every census4 sink to exactly one writer group (first matching rule wins)."""
import json, re, collections, sys
d = json.load(open("<census-work>/out/census4.json"))
B = "vllm/v1/attention/backends/mla/b12x_mla_sparse.py"
R = [  # (group, file regex, line lo, line hi)
 ("G1", re.escape(B), 2108, 2108),
 ("G2", r"b12x/attention/_shared/mla/kv_cache\.py", 213, 443),
 ("G2", r"b12x/attention/sparse_mla/_preparation\.py", 304, 304),
 ("G2", r"b12x/attention/sparse_mla/_preparation\.py", 350, 350),
 ("G1", r"b12x/attention/_shared/mla/kv_cache\.py", 1, 900),
 ("G1", r"b12x/attention/sparse_mla/api\.py", 133, 165),
 ("G1", r"b12x/attention/sparse_mla/_preparation\.py", 280, 400),
 ("G2", re.escape(B), 2109, 2160),
 ("G2", r"vllm/v1/attention/backend\.py", 1, 99999),
 ("G2", r"b12x/attention/_shared/mla/kv_cache\.py", 901, 99999),
 ("G2", r"vllm/_custom_ops\.py", 2780, 2860),
 ("G3", re.escape(B), 100, 230),
 ("G4", re.escape(B), 520, 545),
 ("G4", re.escape(B), 2560, 2760),
 ("G4", re.escape(B), 3100, 3160),
 ("G3", re.escape(B), 3080, 3095),
 ("G3", re.escape(B), 3180, 3240),
 ("G3", r"b12x/attention/(sparse_mla|_shared/mla|dense_mla)/.*", 1, 99999),
 ("G3", r"b12x/attention/_shared/workspace\.py", 1, 99999),
 ("G5", r"vllm/model_executor/layers/attention/mla_attention\.py", 1, 99999),
 ("G5", r"vllm/model_executor/layers/attention/pcp\.py", 1, 99999),
 ("G16", r"vllm/model_executor/layers/attention/attention\.py", 745, 745),
 ("G17", r"vllm/model_executor/layers/attention/attention\.py", 791, 791),
 ("G16", r"vllm/v1/attention/backends/flash_attn\.py", 1210, 1210),
 ("G16", r"vllm/_custom_ops\.py", 2641, 2641),
 ("G16", r"vllm/model_executor/models/qwen3_dflash\.py", 1, 99999),
 ("G17", r"vllm/v1/attention/backends/flash_attn\.py", 1, 99999),
 ("G17", r"vllm/v1/worker/cp_utils\.py", 1, 99999),
 ("G6", r"vllm/models/glm5next/nvidia/kda\.py", 513, 547),
 ("G6", r"vllm/model_executor/layers/mamba/ops/causal_conv1d\.py", 1, 99999),
 ("G8", r"vllm/models/glm5next/nvidia/kda\.py", 591, 591),
 ("G8", r"vllm/models/glm5next/nvidia/kda\.py", 657, 657),
 ("G8", r"vllm/third_party/flash_linear_attention/ops/kda_replayssm_spec_decode\.py", 1, 99999),
 ("G7", r"vllm/models/glm5next/nvidia/kda\.py", 600, 720),
 ("G7", r"vllm/third_party/flash_linear_attention/ops/(kda|fused_recurrent)\.py", 1, 99999),
 ("G7", r"vllm/model_executor/layers/mamba/ops/(scatter_states|gather_initial_states)\.py", 1, 99999),
 ("G9", r"vllm/models/glm5next/nvidia/kda\.py", 721, 760),
 ("G11", r"vllm/v1/worker/mamba_utils\.py", 1080, 1300),
 ("G10", r"vllm/v1/worker/mamba_utils\.py", 1, 1079),
 ("G12", r"vllm/v1/worker/gpu/model_runner\.py", 1170, 1170),
 ("G12", r"vllm/v1/worker/utils\.py", 623, 660),
 ("G13", r"vllm/v1/worker/utils\.py", 60, 270),
 ("G14", r"vllm/v1/worker/utils\.py", 561, 622),
 ("G14", r"vllm/v1/worker/gpu/attn_utils\.py", 1, 99999),
 ("G14", r"vllm/v1/worker/gpu/model_runner\.py", 1, 99999),
 ("G14", r"vllm/model_executor/layers/mamba/abstract\.py", 1, 99999),
 ("G14", r"vllm/v1/worker/kv_connector_model_runner_mixin\.py", 1, 99999),
 ("G15", r"vllm/v1/worker/gpu/kv_connector\.py", 1, 99999),
 ("G15", r"vllm/model_executor/layers/attention/kv_transfer_utils\.py", 1, 99999),
 ("G18", r"vllm/model_executor/layers/sparse_attn_indexer(_kpool)?\.py", 1, 99999),
 ("G18", r"vllm/models/glm5next/nvidia/ops/kpool_compress\.py", 1, 99999),
 ("G18", r"b12x/attention/dsa_indexer/.*", 1, 99999),
 ("G18", r"vllm/_custom_ops\.py", 2950, 3100),
 ("G19", r"vllm/model_executor/layers/mamba/gdn/.*", 1, 99999),
 ("G19", r"vllm/model_executor/layers/attention/(static_sink|cross)_attention\.py", 1, 99999),
 ("G19", r"vllm/model_executor/warmup/.*", 1, 99999),
 ("G19", r"vllm/third_party/flash_linear_attention/ops/(chunk|fused_recurrent_replayssm|fused_sigmoid_gating|gdn_replayssm_spec_decode)\.py", 1, 99999),
 ("G19", r"vllm/_custom_ops\.py", 1, 2780),
 ("G20", r"vllm/compilation/passes/fusion/.*", 1, 99999),
 ("G21", r"b12x/_lib/.*", 1, 99999),
]
out = collections.defaultdict(list); un = []
for s in d["sinks"]:
    for g, rx, lo, hi in R:
        if re.fullmatch(rx, s["file"]) and lo <= s["line"] <= hi:
            out[g].append(s); break
    else:
        un.append(s)
for g in sorted(out, key=lambda x: int(x[1:])):
    ss = out[g]
    kinds = collections.Counter(x["kind"] for x in ss)
    files = sorted({x["file"] for x in ss})
    print(f"{g}\t{len(ss)}\t{dict(kinds)}\t{len(files)} files")
print("UNASSIGNED", len(un))
for s in un: print("  ", s["file"], s["line"], s["callee"])
json.dump({g: [f"{x['file']}:{x['line']}|{x['kind']}|{x['callee']}" for x in v] for g, v in out.items()},
          open("<census-work>/out/groups.json", "w"), indent=1)
