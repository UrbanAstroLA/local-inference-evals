# A: hybrid Mamba/GDN/KDA state, mamba-align prefix caching, MRV2 model state (read-only audit)

Fork paths are $S/v091/<repo path>. v070 and v080 are byte-identical to v091 for mamba_hybrid.py, worker/mamba_utils.py, gdn_attn.py, mamba_attn.py, single_type_kv_cache_manager.py and layers/mamba/mamba_utils.py. Only kv_cache_coordinator.py and scheduler.py differ (0.9.x adds port-dflash2-boundary-prefix-cache: an env-gated block plus the back-off condition).

## Deployed-run facts that decide reachability
- All six <gpqa-runs>/*.server.log show `Prefix cache hit rate: 0.0%` on every line and 0 preemptions. Running is mostly 8 (max 16).
- Mamba block = attention block = 4352 (log: "Setting attention block size to 4352"). Every GPQA prompt (≤2800 tokens) is shorter than one block, so no prefix-cache hit and no resume with num_computed>0 can happen.
- What is exercised: align-mode boundary crossings during long decodes (precopy in preprocess_state, align copy in postprocess_state), and spec-decode rows with K=3. Zero-draft or partial-draft decode rows only occur when the token or input budget runs out behind a running prefill chunk (scheduler.py:557-562). GPQA prompts need at most 2 chunks, so this is effectively absent (inferred).
- The DFlash SW draft group sets dcp_shard_count_override=1 unconditionally (qwen3_dflash.py:239), so it does not participate in prefix caching (kv_cache_interface.py:682-687). As a result cache_config.block_size = min over participating groups = 4352 = mamba_block_size (engine/core.py:322-331).

## Ranked table: see final report (same content).

## Fork-only change attribution
- mamba_hybrid.py ReplaySSM fields/branches: vllm-replayssm-spec.patch. Gated on use_replayssm_spec (False), so inert.
- worker/mamba_utils.py TEMPORAL_ACCEPT_BIAS / get_replayssm_spec_copy_funcs: vllm-replayssm-spec.patch. Default True gives identical behaviour.
- layers/mamba/mamba_utils.py: only added ReplaySSM ring shape/dtype helpers and get_current_block_copy_spec, so inert.
- gdn_attn.py: ReplaySSM is gated. The elif restructure is equivalent. The glm5_next non_spec_chunk_indices/offsets come from port-glm53-sm12-stability.py and are active.
- mamba_attn.py: Mamba2 ReplaySSM only. The KDA path does not use it.
- single_type_kv_cache_manager.py:1420 dropped `assert dcp_world_size == 1` (port-b12x-dcp-glm53.py).
- kv_cache_coordinator.py:
  - eagle_group_ids scoping: port-glm53-mtp-prefix-cache.py
  - GLM53_DFLASH_BOUNDARY_LOOKUP block: port-dflash2-boundary-prefix-cache.py (0.9.x only, env default 0)
  - per-group dcp_shard_count_override, DCP SW assert, and skipping non-participating groups in cache_blocks: port-dflash2-replicated-dcp.py
- scheduler.py:
  - AdaptiveMTP: port-adaptive-mtp-glm53.py. It raises unless method=='mtp', so it is inert under dflash.
  - back-off condition at :410: port-dflash2-boundary-prefix-cache.py
