# B: spec-decode / MRV2 / CUDA-graph / sampling audit (fork 0.9.1 vs upstream)

Paths: F=$S/v091/vllm, base trees $S/upstream-diff/trees/{base,main}/vllm. Spec-decode files (gumbel.py, dflash/, dflash2/, speculator.py, rejection_sampler*.py, block_table.py, cudagraph_utils.py, mamba_hybrid.py) are byte-identical in 0.7.0/0.8.0/0.9.1.

Deployed facts used (GPQA logs): draft_sample_method unset -> "greedy" (config/speculative.py:291) so draft_logits=None (spec_decode/speculator.py:146), DFlash2 walk runs with temp 0 (dflash2/speculator.py:62); target T=1.0 top_p=0.95, per-request seeds; FA version 2 (log line 203) -> aot_schedule False (flash_attn.py:445); PIECEWISE(breakable) mixed + FULL decode graphs (log 353/359/399); prefix-cache hit 0.0% throughout, 0 preemptions; GLM53_DFLASH_BOUNDARY_LOOKUP inert (no log line); slot sharing on (log 240: block 1088, page 2228224/2297856).

Ranked findings: see final report (SubagentHandback). Key code refs:
- workspace unlocked after capture: gpu_worker.py:715-719 (capture) vs :796-797 (warmup_kernels after); no lock_workspace in V2 (only gpu_model_runner.py:7180 V1).
- DFlash null-block/rejected-suffix guards (upstream 97388c44f) absent: dflash/speculator.py:544-555, 563-575.
- d2983f2f1 absent: block_table.py:300-328 (no is_real_req), speculator _copy_request_inputs no dummy_run.
- fedbc3b56 absent: rejection_sampler.py:259-260; trigger scheduler.py:925-939,1119-1122.
- fe755c889 absent: gumbel.py:85-121 (no salt); dflash2/speculator.py:54-64 (draft key = sample_pos-1 = target resample key, rejection_sampler_utils.py:826-841).
- aux taps (port-dflash2-glm-eagle3.py) == upstream 2274e8684 semantics; mhc_post out-of-place (kernels/mhc/tilelang.py:410-430).
