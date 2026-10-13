# Forced-decode window: feasibility from source (amendment 20, section 4)

2026-10-13. CPU only. Source trees as extracted for the source review (`scratchpad/kdcp/{v070,v091}`, B0 =
`kdcp/b0-cache/src`); installed-module checks ran in CPU containers (`--network none`, no `--gpus`, read-only
mount) of the 0.9.1 release image (`a9e985c0`) and the stock 0.7.0 image (`48e254d9`). Receipts: `cpu-check.out`
(14 checks per image), `test_forcewin_cpu.py`, `forcewin_logic.py`, `plugin_startup_check.sh`.

## Verdict per engine

| Engine (image) | Custom v1 logits processor | Prompt-logprobs / `logprob_token_ids` | Grammar forcing | Engine hook (proposed) |
|---|---|---|---|---|
| stock 0.7.0 (`48e254d9`) | **not available**: the engine runs Model Runner V2, which refuses custom logits processors | prefill only; cannot see decode-written state | not usable | **feasible**, one hook for all engines |
| stock 0.9.1 (`a9e985c0`) | not available (same) | same | not usable | feasible |
| B0 (`08d1cb90`), B0-minus-D2 (to build), B1 (`799d3cf8`) | not available (same) | same | not usable | feasible |

Nothing reaches the decode path without new instrumentation. The instrumentation is small (one module plus hook lines
in three files that are byte-identical across all engines); it is priced in the proposal as Stage 0 engineering.

## Why the existing routes fail

1. **Custom logits processors.** Every engine runs Model Runner V2 (`v1/worker/gpu/`; "Using V2 Model Runner" in the
   server logs). DFlash2 forces it: `VllmConfig.use_v2_model_runner` returns True because "the DFlash2 candidate
   selector exists only in the V2 speculator" (check E2). `_get_v2_model_runner_unsupported_features` lists
   "custom logits processors" whenever `--logits-processors` or a `vllm.logits_processors` entry point is present,
   and `_validate_v2_model_runner` raises (check E1, both images). Falling back to the V1 runner would drop DFlash2 to
   DFlash1 silently and change the decode path; not acceptable.
2. **Prompt logprobs.** `prompt_logprob.py` scores prompt positions from the prefill forward. Prefill rebuilds
   index-K through `_kpool_compress_insert`, so it cannot see inv-D2 or D6 (source review section 3).
3. **`logprob_token_ids`.** It exists (`SamplingParams.logprob_token_ids`, up to 128 ids) and would report the four
   termination ids at every generated position, but (a) it does not force tokens, (b) the completions protocol sets
   `logprobs=None` when it is given (check E3c), and `RejectionSampler._get_logprobs_tensors` never receives the
   per-request id state, so under speculative decoding the ids are not served (source reading; not run on a GPU), and
   (c) it cannot give the tail mass below a token, entropy or argmax.
4. **Grammar / structured output.** Constrains text, not ids (several tokenizations remain allowed), adds a bitmask to
   the sampled path, and with the reasoning parser it applies only after the reasoning close. Rejected.
5. **Offline `LLM` runner.** Same engine core and workers; it needs the same hook and leaves the documented server
   lifecycle. No advantage.

## The hook

**Where it runs.** After the real sampler, on the GPU, with no host synchronisation per step:
- `RejectionSampler._verify` (speculative steps) and `Sampler.sample` (steps without drafts: the first token after
  prefill, prefill-scoring requests): stash, for forced rows only, the raw target logits and the served processed
  logits (after penalties, temperature and top_p, i.e. exactly what the rejection sampler used).
- `GPUModelRunner.sample`: after the real call returns, override `sampled_token_ids`, `num_sampled` and
  `num_rejected` of forced requests (`forced_override`), and write the measurements of valid rows into a per-request
  GPU buffer indexed by output position (`row_metrics`).
- Request removal: one device-to-host copy of the buffer per request; optionally the index-K pages of the window
  (Stage 1 dump; page addressing as in `kdcp/b0-cache/fixture/b0views.py` and the phase-2 capture hook).

**Keying.** A request is forced only if its `vllm_xargs` carries the forced id list (a JSON list of ints; check E3a
shows it reaching `SamplingParams.extra_args` on both images). Other requests (background co-runners) are untouched
(check F2). `cache_salt` (check E3b) isolates every request's prefix-cache entries while leaving prefix caching on,
because turning it off also switches `--mamba-cache-mode align` to `none` in both start scripts and would change the
KDA migration path.

**Delivery.** A general plugin would leave every engine file unmodified, and it is discovered on 0.7.0 (E4 PASS), but
the 0.9.1-family images set `VLLM_PLUGINS=lora_filesystem_resolver`, an allowlist that filters it out (E4 and E4b
FAIL). Changing image environment is avoidable, so the hook ships as in the guard images: a new module
`vllm/local_forcewin.py` and marked hook lines in `v1/worker/gpu/model_runner.py`,
`v1/worker/gpu/spec_decode/rejection_sampler.py` and `v1/worker/gpu/sample/sampler.py`, each inert unless a request
carries the key. The whole `v1/worker/gpu/` tree is byte-identical between 0.7.0 and 0.9.1 except a logging call in
`model_runner.py`; sampler, rejection sampler, sampling states, input batch and plugin loader are md5-identical; B0 and
B1 change none of them. One patch generator therefore serves every arm.

## How committed state is advanced

Per step and per forced request with rows `j = 0..n-1` (row 0 is fed the last committed token, rows `1..n-1` the
drafts), row `j` predicts output index `o_j`; `f_j` is the history token at `o_j`.

- Draft `j+1` is **accepted iff it equals `f_j`** (the longest matching prefix, `a` drafts).
- The committed tokens are `f_0..f_a` (`a+1` tokens; clamped at the window end). The engine's own `post_update`,
  KDA rollback (`postprocess_state(num_sampled)`), draft proposal and scheduler accounting then run unchanged on these
  values, so **committed state advances only by history tokens** (checks F1-F4, K=3 and K=5).
- Rows `0..a` were conditioned only on history tokens and are the measurements; rows after the first mismatch were
  conditioned on a rejected draft and are discarded. Each window position is measured exactly once.
- The forward pass is the real decode/verify pass: K+1 query rows per request, the served route (tiled at 8 in
  flight), index-K pools completed by the decode writer including writes for rejected drafts and their redo, the tail
  ring, KDA sequential recurrence with block migrations, DFlash2 drafting from target hidden states, CUDA graphs as
  served.

## What differs from real sampling

1. The committed tokens are a fixed history, not samples: re-tokenized text (and, for some fixtures, another engine's
   generation) is a shared counterfactual fixture, not a replay.
2. **Acceptance.** Real verification accepts a greedy draft x with probability p(x) after top_p; forcing accepts it
   iff the history has x there. If the history were drawn from the probe engine's own distribution the expected
   acceptance would be the same; for a counterfactual or re-tokenized history it is lower, so the reject-and-redo path
   (the path where D6 lives, and where D2's ring rewrites happen) is exercised **more** than in serving, by an amount
   measured per window (accepted length, per-position acceptance, and pool-completing draft rejections, i.e. a
   rejected draft at a position with `pos % 4 == 3`). Matching drafts **are** accepted (check F1), so forcing does not
   reject systematically. The oracle-draft bracket in Stage 2 (drafts replaced by the next history tokens, acceptance
   near 100%) gives the low-rejection extreme.
3. Random draws are made by the real sampler and discarded; they do not touch state.
4. A history token can be outside the top_p nucleus (served probability 0): a state the engine would not reach. Such
   positions are flagged (`p_post = 0` for the forced token) and counted per window.
5. The prefix is built by prefill (2,048-token chunks and 4,352-token prefix-cache blocks), not by generation. Only
   the window's W/4 pools are decode-written (2,048 of 16k-30k pools for W = 8,192); KDA state at the window start is
   the prefill state. A cold prefill does not reconstruct the state accumulated over 100k tokens of speculative
   generation (amendment 20 section 4); the windows therefore under-represent D2/D6 relative to a whole decode-built
   context. Stage 1's index-K dumps measure the per-pool corruption, and a long-window or full-replay extension is
   priced separately.
6. Co-runners are other fixtures (or sampled background prompts in the composition runs), not GPQA questions.

## Recorded per measured row (no text)

Pre-mask log-probability of `</think>` 154842, `<|endoftext|>` 154820, `<|user|>` 154827, `<|observation|>` 154829
and of the forced token; probability mass strictly below and at-or-below each (top_p eligibility bounds; the served
mask is read directly from the processed logits, check M1-M2); post-mask probability of each; entropy; argmax id
equality with the forced token and whether the argmax is a termination id; top-1 probability; step metadata (requests
and rows in the batch, draft count, accepted count, row index, prefill tokens co-scheduled, output index). Token ids of
the history are kept local; published outputs carry metrics and positions only.

## Recurrent-state rollback (coordinator item 3)

- **Two paths exist in every tree.** Baseline full-state rollback: `fused_recurrent_kda` writes one fp32 state per
  speculative token and the next step reads slot `num_accepted - 1` (`kda.py`, source review section 5). ReplaySSM:
  `--use-replayssm` (`cache_config.use_replayssm_spec`; `kda_replayssm_spec_decode.py`, md5-identical in 0.7.0 and
  0.9.1; `materialize_kda_replayssm_state`; extra ring state in `MambaStateShapeCalculator.append_kda_replayssm_spec_ring`)
  replays accepted steps from a buffer of length `REPLAYSSM_BUFFER_LEN` (default 10).
- **Selection in the served configs.** Both start scripts set `USE_REPLAYSSM=1` only for the MTP profile and `0` for
  DFlash2 (a comment says DFlash2 "favors baseline rollback"; ReplaySSM "remains available with USE_REPLAYSSM=1").
  No GPQA server log (0.7.0, stock 0.9.1 passes 1-3, B1) has `use_replayssm` in its non-default args; the B1 start log
  states "State rollback: baseline full-state rollback". ReplaySSM is therefore reachable only by an explicit
  override and was never served in any record this investigation uses.
- **Interaction with attention/index-K restoration.** None in code: no index-K, tail-ring, kpool-compress or MLA
  file references either rollback mode (grep over `indexer.py`, `sparse_attn_indexer_kpool.py`, `kpool_compress.py`,
  `attention.py`, `b12x_mla_sparse.py` in B0). Both modes consume the same `num_accepted` that the forcing override
  sets, so forcing is consistent with either. The probe uses the served baseline mode; ReplaySSM is out of scope and
  untested here.

## Draft depth and the tail ring (coordinator items 1-2)

`Glm5NextTailCache.get_kv_cache_spec` (B0/0.9.1 `models/glm5next/nvidia/attention.py`, marker `kpool-spec-ring`,
port of vllm#58454) sizes the ring as `kpool * next_pow2(ceil((kpool + K) / kpool))`: **8 slots at K=3, 16 at K=5**,
so a pool plus every draft fits and a rejected pool-completing draft's redo reads intact keys. 0.7.0's spec has
`block_size = sliding_window = index_kpool` (**4 slots**) and its `kpool_compress.py` lacks both the spec-ring and the
seed-stride (vllm#57477) markers: with 5 drafts the ring wraps (**D6**). `DFLASH_TOKENS` is an existing start-script
variable (1-7; default 3 in the v0.9.0 stack, 5 in the v0.7.0 stack), so B0 at x5 needs no build; its capture size
follows (`MAX_NUM_SEQS * (K + 1)`), and 4,352 is a multiple of 16. A D6-only diagnostic build (B0 with the spec-ring
port reverse-applied to `attention.py` and `kpool_compress.py`, ring 4) is described in the proposal.
