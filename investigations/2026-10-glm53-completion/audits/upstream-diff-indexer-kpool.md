# notes-C-indexer: sparse-MLA / DSA indexer / kpool / DCP / KV-layout upstream diff (CPU-only, read-only)

S=<scratch>. Fork paths: $S/v0XX/vllm/<repo path minus "vllm/">.
Commit diffs saved: $S/upstream-diff/c/<sha>.diff; scan log $S/upstream-diff/c/scanlog.txt.

## TOP FINDING: kpool tail ring collapses onto the null block (all versions, MRV2 hybrid)
Chain (all files byte-identical v070/v080/v091 unless noted):
1. GLM-5.3 is_hybrid (models/glm5next/nvidia/model.py:1056) and defines no get_model_state_cls ->
   MambaHybridModelState (v1/worker/gpu/model_states/__init__.py:36-39).
2. MambaHybridModelState.prepare_attn calls build_attn_metadata WITHOUT positions
   (model_states/mamba_hybrid.py:353-370); build_attn_metadata default positions=None (attn_utils.py:610, 657).
   Upstream added positions=input_batch.positions in 98ed0856f3 (trees/main mamba_hybrid.py:339). Known defect.
3. KpoolTailMetadataBuilder.build only applies the circular per-request mapping if positions is not None
   (v1/attention/backends/mla/indexer.py:613-625); otherwise it returns the generic MRV2 slot mapping.
4. Generic MRV2 slot kernel (v1/worker/gpu/block_table.py:262-328; fork adds an in_range guard) computes
   slot = bt[req][pos // 8] * 8 + pos % 8 for the tail group (block 8, row width cdiv(1M, 8) per
   model_runner.py:516-539, not 1). KpoolTailManager writes only column 0 (single_type_kv_cache_manager.py:1097+;
   StagedWriteTensor zero-init) -> for pos >= ring every request's tail slot is in block 0 (null block).
   The fork's own docstring (indexer.py:545-552) describes exactly this collapse.
5. Consumers: kpool_seed_tail_cache (prefill seed; kpool_compress.py:449-521) and
   _kpool_decode_update_batched_kernel (kpool_compress.py ~560-700) derive block = tail_slot // RING ->
   block 0 for all requests; kernel docstring assumes "distinct tail blocks" per program.
   0.7.0 DCP2 additionally: generic CP path PADs tokens with pos % 2 != rank (block_table.py:318-326),
   so each rank stashes only same-parity tokens; pool completion reads never-written (zero) or foreign
   ring entries for the other parity -> every decode-built pool is wrong even at concurrency 1.
CPU transcription (python heredoc, not saved): 0.9.1-like (ring 8, DCP1, K=3) decode-built pools with
a corrupted ring read: 0% @1 req, 56% @2, 79% @4, 88% @8 concurrent. 0.7.0-like (ring 4, DCP2, K=5):
100% at 1 and 8 reqs, both ranks. Generic slots for own block 100: pos 5->805, 7->807, 8->0, 9->1, 4100->4.
Effect: index-K of pools completed during decode (reply history + prompt boundary pool) is a gate-softmax
blend including other requests' (or zero) K/gate rows -> wrong pool scores -> wrong top-512 selection
once context > 2047 tokens. Prompt pools (prefill _kpool_compress_insert) and MLA latents unaffected.
Invisible to prefill-only teacher-forced NLL and to single-request probes on DCP1.
Fix needs THREE parts (upstream has all): positions in mamba_hybrid (98ed0856f3); persistent
slot_mapping_buffer for the tail builder (98ed0856f3; fork's compute_kpool_tail_slot_mapping returns a
fresh clone, indexer.py:566, unsafe once FULL graphs bake the indexer op - inferred); own_block != 0
guard (240785b82). Under DCP the circular mapping is rank-agnostic, matching upstream dcp_sharded=False
(30d403236).
Side effect of the same missing positions (MLA area, not mine): b12x_mla_sparse.py:1439-1446 falls back
from exact GPU positions to seq_lens_cpu_upper_bound on DCP1, which its own comment says can lag
under async scheduling for multi-token verify.

## Per-commit verdicts (see final report table)
39724758e a-NA (supports_varlen False on SM120, log L9:243; decode_indices None; fallback path rarely used)
70df48dc3 b/related: fork has in_range guard + full-width tail row, no OOB; generic output still used -> TOP
240785b82 a, latent until positions fixed (indexer.py:566-568 no own_block!=0 guard)
1b9fa3eaa c (perf; fork torch path)
2edbb32e5 a-NA (SM90); no B12x analogue (B12x never uses tokens_per_state; nsa_len from width)
133b71e0b a-NA (vendored DSA sparse_attn_indexer.py; kpool path flatten -> per-token rows, write uses
          per_req_decode_lens + scatter, _decode_topk_seq_lens consistent)
586f652f8 a-NA (PCP plan, interleave>1; fork refuses interleave>1, indexer.py:700-707)
241e9391e a-NA/c (PCP, FlashInfer TRTLLM workspace, sync removal; empty-shard guard = crash not silent)
22099afc6 b-equivalent (fork mla_attention.py:1008-1017 combines all reqs for B12x since decode is None)
0ecc28479 a-NA (interleave 1: old/new math identical: slot = bt[pos//128]*64 + (pos%128)//2)
83990f5bc a-NA (PCP/ubatch partition; fork pads dcp_local_seq_lens to num_reqs_padded, model_runner v070:1331-1339)
29468dde8 a-NA/c (FlashInfer/FlashMLA flat layouts; B12x uses real stride(0); idx page 143,616 not 512-aligned
          but DeepGEMM paged path unreachable)
8c51b9265 a-NA (cfg None -> interleave 1 == deployed)
dc07f1638 d (Glm5NextConfig forwards attrs to text_config, transformers_utils/configs/glm5_next.py:366-387)
01e56caaf a-NA (B12X supports_dense_mha_prefill=False, b12x_mla_sparse.py:1566)
58aa1e3d2 a-NA (FlashInfer SM120 backend not deployed)
62af3df73 c (memory sizing)
51918e252 a-NA (index_group/hisparse)
c37f86e57 c (workspace sizing; fork consistent, oversized)
30d403236 c vs fork-own DCP2 (fork: pool p on rank p%2 via compressor_utils.py fork diff, interleave 1, MLA tokens t%2;
          upstream: interleave multiple of kpool so pool + its MLA tokens co-located; tail dcp_sharded=False)
02e55d2af c (refactor; ring formula same as fork port attention.py:191-201)
36fa72d2d c, 2bbdfcfcf c, 70bdce613 c, bf01fc4a3 c, 3e8e1364f c, ebe1dec2d a-NA (PCP), 273e09969 a-NA (ROCm)
9f1c78388 d (test tolerance: bf16 midpoint rounding of pooled K vs CPU ref -> 1 fp8 step; FP only)

## csrc (compiled from 487ecf187)
indexer_k_quant_and_cache / cp_gather_indexer_k_quant_cache: no upstream change base..main.
top_k_per_row_prefill (sampler.cu): only ROCm TopK512 path + warnings since base; CUDA semantics unchanged.
persistent_topk/cooperative_topk negative-length clamp (97388c44f) absent in base (persistent_topk.cuh:909)
  - only reached by DeepGEMM decode fallback (unreachable) -> a-NA.
concat_and_cache_ds_mla pow2 scale (a7fda4c88) absent; B12x reads ARBITRARY_FP32 scales -> a-NA.
pe_dim==0 support (c843f0cab) absent; fork passes zero RoPE pad (0.7.0) or uses B12x writer -> equivalent.

## Other upstream fixes seen in scan (outside this scope, pointers only)
d095e61c0 DFlash2/EAGLE3 aux hidden-state capture with upstream config; 0eac15270 route every
speculation-capable row through spec path (recurrent state); d8fa463e2 hybrid Mamba prefix-cache
boundaries; ae53b0689 KDA fused recurrent gridDim.z; c9b34fdb2 dense MLP on SP shard.
