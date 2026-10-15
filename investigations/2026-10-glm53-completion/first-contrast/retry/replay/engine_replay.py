#!/usr/bin/env python3
"""F7 [R-1]: CPU replay of the B1 serving image's ACTUAL scheduler and KV-cache manager.

Runs INSIDE a CPU container of the serving image (local/glm53-v0.9.1-b1-forcehook-mg2:local,
sha256:ac05230b...), `--network none`, no `--gpus`, CUDA_VISIBLE_DEVICES empty. Nothing here is a re-implementation:
the classes driven are the image's own
    vllm.v1.core.sched.async_scheduler.AsyncScheduler  (-> Scheduler.schedule / update_from_output)
    vllm.v1.core.kv_cache_manager.KVCacheManager
    vllm.v1.core.kv_cache_coordinator.HybridKVCacheCoordinator
    vllm.v1.core.single_type_kv_cache_manager.{FullAttentionManager, KpoolTailManager, MambaManager,
                                               SlidingWindowManager}
    vllm.v1.core.block_pool.BlockPool, vllm.v1.request.Request, kv_cache_utils.get_request_block_hasher
Only three things are supplied from outside:
  1. the KV-cache config: the logged geometry of every B1 session (glm53-placement kv-capacity line + the guard's init
     record): 1,275 shared block ids; group 0 MLA+index-K (UniformTypeKVCacheSpecs of MLAAttentionSpec, block 4,352),
     group 1 kpool tail (KpoolTailSpec, block 8, opts out of prefix caching), groups 2-8 seven identical KDA MambaSpec
     groups (block 4,352, mamba_cache_mode align, 3 speculative blocks), group 9 the DFlash2 draft window
     (SlidingWindowSpec 2,048, block 1,088, dcp_shard_count_override 1 -> opts out of prefix caching);
  2. a configuration namespace holding exactly the served values (max_num_batched_tokens 2048, max_num_seqs 16,
     async scheduling, Model Runner V2, DFlash K=3, prefix caching on, mamba align, max_model_len 1,048,576);
  3. the "model": per scheduled request, the number of tokens the step commits (0 for a non-final prefill chunk;
     accepted + 1 for a decode step) and their ids (the forced history for keyed requests; filler otherwise).
The engine loop mirrors EngineCore.step_with_batch_queue with batch queue 2 (async scheduling): schedule N+1 is
called before update_from_output(N).

Correctness of (1)-(3) is not assumed: reproduce_fc1.py replays the run-fc1 scheduler log and requires every step's
scheduled set, token counts, spec counts, group-0/1 block ids, per-group block-list lengths and free-block count to be
identical (5,609 steps).
"""
import collections
import gzip
import hashlib
import json
import os
import random
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("GLM53_DFLASH_BOUNDARY_LOOKUP", "0")   # the served value (run-fc1 container-files-env.txt)

import torch  # noqa: E402

from vllm.v1.kv_cache_interface import (  # noqa: E402
    KVCacheConfig, KVCacheGroupSpec, KpoolTailSpec, MLAAttentionSpec, MambaSpec, SlidingWindowSpec,
    UniformTypeKVCacheSpecs)

BLOCK = 4352
NUM_BLOCKS = 1275
SPEC_K = 3
SERVED = dict(max_num_batched_tokens=2048, max_num_seqs=16, max_model_len=1048576, async_scheduling=True,
              use_v2_model_runner=True, method="dflash", num_speculative_tokens=SPEC_K, enable_prefix_caching=True,
              mamba_cache_mode="align", block_size=BLOCK, num_blocks=NUM_BLOCKS,
              boundary_lookup=os.environ["GLM53_DFLASH_BOUNDARY_LOOKUP"],
              retention_interval=os.environ.get("VLLM_PREFIX_CACHE_RETENTION_INTERVAL"))
MLA_LAYERS = [3, 7, 11, 15, 19, 23]          # rank-0 owners (split:25); names only gate eagle-group detection
KDA_PAGE = 2297856
GROUP_NAMES = ["mla+idx", "tail", "kda0", "kda1", "kda2", "kda3", "kda4", "kda5", "kda6", "draft"]


class Strict(types.SimpleNamespace):
    """Config namespace that fails loudly on any attribute the image reads but we did not set."""

    def __getattr__(self, k):
        raise AttributeError("replay config: unset attribute %r" % k)


def kv_cache_config(num_blocks=NUM_BLOCKS, mla_prefix="language_model.model.layers.",
                    kda_prefix="language_model.model.layers."):
    mla = {}
    for i in MLA_LAYERS:
        mla[f"{mla_prefix}{i}.self_attn.indexer.k_cache"] = MLAAttentionSpec(
            block_size=BLOCK, num_kv_heads=1, head_size=132, dtype=torch.uint8, compress_ratio=4)
        mla[f"{mla_prefix}{i}.self_attn.attn"] = MLAAttentionSpec(
            block_size=BLOCK, num_kv_heads=1, head_size=656, dtype=torch.uint8, cache_dtype_str="fp8_ds_mla")
    tail = {f"{mla_prefix}{i}.self_attn.indexer.tail": KpoolTailSpec(
        block_size=8, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, sliding_window=8,
        page_size_padded=143616) for i in MLA_LAYERS}
    kda = MambaSpec(block_size=BLOCK, shapes=((16,), (16,)), dtypes=(torch.float32, torch.float32),
                    page_size_padded=KDA_PAGE, mamba_cache_mode="align", num_speculative_blocks=SPEC_K)
    kda_layers = [i for i in range(47) if i % 4 != 3][:34]
    draft = {f"model.layers.{i}.self_attn.attn": SlidingWindowSpec(
        block_size=1088, num_kv_heads=4, head_size=128, dtype=torch.bfloat16, sliding_window=2048,
        page_size_padded=KDA_PAGE, dcp_shard_count_override=1) for i in range(5)}
    groups = [KVCacheGroupSpec(list(mla), UniformTypeKVCacheSpecs(block_size=BLOCK, kv_cache_specs=mla)),
              KVCacheGroupSpec(list(tail), UniformTypeKVCacheSpecs(block_size=8, kv_cache_specs=tail))]
    names = [f"{kda_prefix}{i}.linear_attn" for i in kda_layers]
    for g in range(7):
        groups.append(KVCacheGroupSpec(names[g::7], kda))
    groups.append(KVCacheGroupSpec(list(draft), UniformTypeKVCacheSpecs(block_size=1088, kv_cache_specs=draft)))
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups)


def vllm_config(kvc, max_num_batched_tokens=2048, max_num_seqs=16):
    from vllm.config.vllm import VllmConfig
    spec = Strict(method="dflash", num_speculative_tokens=SPEC_K, num_speculative_tokens_per_batch_size=None,
                  parallel_drafting=True, draft_model_config=None)
    spec.use_eagle = lambda: True
    spec.use_dflash = lambda: True
    spec.use_dspark = lambda: False
    spec.uses_draft_model = lambda: False
    spec.use_gemma4_mtp = lambda: False
    spec.max_num_new_slots_for_drafting = SPEC_K          # DFlash: K (SpeculativeConfig property, same value)
    sched = Strict(max_num_batched_tokens=max_num_batched_tokens, max_num_scheduled_tokens=None,
                   max_num_seqs=max_num_seqs, long_prefill_token_threshold=0, policy="fcfs",
                   scheduler_reserve_full_isl=True, watermark=0.0, enable_chunked_prefill=True,
                   disable_chunked_mm_input=False, async_scheduling=True, max_num_partial_prefills=1)
    cache = Strict(block_size=BLOCK, enable_prefix_caching=True, mamba_cache_mode="align", num_gpu_blocks=kvc.num_blocks,
                   prefix_caching_hash_algo="sha256", prefix_match_unit=None, kv_cache_groups=kvc.kv_cache_groups)
    par = Strict(decode_context_parallel_size=1, prefill_context_parallel_size=1, pipeline_parallel_size=1,
                 data_parallel_index=0, data_parallel_size=1)
    model = Strict(max_model_len=1048576, is_encoder_decoder=False, is_diffusion=False, multimodal_config=None,
                   enable_return_routed_experts=False, runner_type="generate", is_multimodal_model=False,
                   hf_config=Strict(), skip_tokenizer_init=False)
    obs = Strict(kv_cache_metrics=False, kv_cache_metrics_sample=0.01, enable_mfu_metrics=False)
    ecm = Strict()
    ecm.get_encoder_cache_manager_obj = lambda: None
    cfg = Strict(scheduler_config=sched, cache_config=cache, lora_config=None, kv_events_config=None,
                 parallel_config=par, observability_config=obs, model_config=model, kv_transfer_config=None,
                 ec_transfer_config=None, speculative_config=spec, diffusion_config=None, ec_manager_config=ecm,
                 is_encoder_only=False, is_ec_producer_only=False, use_v2_model_runner=True)
    cfg.max_concurrent_batches = 2                         # VllmConfig: async + V2 -> pp_size + 1
    cfg.max_in_flight_tokens = 2 * max_num_batched_tokens
    cfg.num_speculative_tokens = SPEC_K
    cfg.num_lookahead_tokens = SPEC_K + 1                  # VllmConfig.num_lookahead_tokens for dflash
    # cross-check the derived values against the image's own property code
    assert VllmConfig.num_lookahead_tokens.fget(cfg) == cfg.num_lookahead_tokens
    assert VllmConfig.max_concurrent_batches.fget(cfg) == 2
    return cfg


class NoStructuredOutput:
    """No request in these sessions uses structured output; the manager is only asked these questions."""

    @staticmethod
    def should_advance(*a, **k):
        return False

    @staticmethod
    def grammar_bitmask(*a, **k):
        return None

    @staticmethod
    def clear_backend():
        return None


class FakeMM:
    @staticmethod
    def supports_multimodal_inputs(model_config):
        return False


def make_scheduler(num_blocks=NUM_BLOCKS, max_num_batched_tokens=2048, max_num_seqs=16, mla_prefix=None,
                   kda_prefix=None):
    from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config, resolve_kv_cache_block_sizes
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    kw = {}
    if mla_prefix is not None:
        kw["mla_prefix"] = mla_prefix
    if kda_prefix is not None:
        kw["kda_prefix"] = kda_prefix
    worker_kvc = kv_cache_config(num_blocks, **kw)
    # the image's own EngineCore step: the scheduler sees the first spec of every uniform group
    kvc = generate_scheduler_kv_cache_config([worker_kvc])
    cfg = vllm_config(kvc, max_num_batched_tokens, max_num_seqs)
    try:
        register_all_kvcache_specs(cfg)
    except Exception:  # noqa: BLE001  (platform hook; the built-in registrations are done first)
        pass
    sbs, hbs = resolve_kv_cache_block_sizes(kvc, cfg)
    s = AsyncScheduler(vllm_config=cfg, kv_cache_config=kvc, structured_output_manager=NoStructuredOutput(), block_size=sbs,
                       hash_block_size=hbs, mm_registry=FakeMM(), log_stats=False)
    s._replay_block_sizes = (sbs, hbs)
    s._replay_cfg = cfg
    return s


def placement_summary(s):
    """The image's 'glm53-placement kv-capacity' group summary recomputed from the replay's config (compared with the
    logged line of each session)."""
    from vllm.utils.math_utils import cdiv
    out = []
    for g in s.kv_cache_config.kv_cache_groups:
        sp = g.kv_cache_spec
        out.append(dict(spec=type(sp).__name__, block_size=sp.block_size, page_bytes=sp.page_size_bytes,
                        blocks_per_max_request=cdiv(sp.max_memory_usage_bytes(s._replay_cfg), sp.page_size_bytes)))
    return dict(num_blocks=s.kv_cache_config.num_blocks, groups=out)


_HASHER = {}


STOP_ID = 154829          # a real generation-stop id (DFlash config eos list); filler never produces it


def make_request(rid, prompt, max_tokens, salt, hbs, arrival=0.0, stop=False):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request
    if hbs not in _HASHER:
        fn = get_hash_fn_by_name("sha256")
        init_none_hash(fn)
        _HASHER[hbs] = get_request_block_hasher(hbs, fn)
    sp = SamplingParams(max_tokens=max_tokens, temperature=1.0, top_p=0.95, ignore_eos=True,
                        stop_token_ids=[STOP_ID] if stop else None)
    return Request(request_id=rid, prompt_token_ids=list(prompt), sampling_params=sp, pooling_params=None,
                   arrival_time=arrival, cache_salt=salt, block_hasher=_HASHER[hbs])


def model_output(so, commits):
    """commits: req_id -> list of token ids this step commits (empty for a non-final prefill chunk)."""
    from vllm.v1.outputs import ModelRunnerOutput
    ids = list(so.num_scheduled_tokens)
    return ModelRunnerOutput(req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
                             sampled_token_ids=[list(commits.get(r, [])) for r in ids])


# ------------------------------------------------------------------ observation helpers
def block_ids(s, rid):
    return [list(x) for x in s.kv_cache_manager.get_block_ids(rid)]


def group_occupancy(s):
    """Live blocks per group (held by running requests, null excluded), cached-and-free blocks per group (hash in the
    pool's map and ref_cnt 0), free total."""
    km = s.kv_cache_manager
    live = [0] * len(km.coordinator.single_type_managers)
    for g, m in enumerate(km.coordinator.single_type_managers):
        seen = set()
        for blks in m.req_to_blocks.values():
            for b in blks:
                if not b.is_null and b.block_id not in seen:
                    seen.add(b.block_id)
        live[g] = len(seen)
    cached = [0] * len(live)
    cached_live = [0] * len(live)
    pool = km.block_pool
    for key, blk in _iter_cached(pool):
        g = _group_of(key)
        for b in (blk.values() if isinstance(blk, dict) else [blk]):
            if b.ref_cnt == 0:
                cached[g] += 1
            else:
                cached_live[g] += 1
    return dict(live=live, cached_free=cached, cached_live=cached_live, free=pool.get_num_free_blocks())


def _iter_cached(pool):
    m = pool.cached_block_hash_to_block
    d = getattr(m, "_cache", None)
    if d is None:
        d = m
    for k, v in d.items():
        yield k, v


def _group_of(key):
    from vllm.v1.core.kv_cache_utils import get_group_id
    return get_group_id(key)


class Engine:
    """EngineCore.step_with_batch_queue with batch queue 2 around the image's AsyncScheduler."""

    def __init__(self, sched):
        self.s = sched
        self.queue = collections.deque()       # (so, step_no)
        self.step_no = 0
        self.lookups = {}                      # rid -> dict(hit, seq, ...)
        self._wrap_lookup()

    def _wrap_lookup(self):
        km = self.s.kv_cache_manager
        orig = km.get_computed_blocks
        eng = self

        def gcb(request):
            blocks, n, spb = orig(request)
            d = eng.lookups.setdefault(request.request_id, dict(tries=0))
            d["tries"] += 1
            d.update(hit=n, at_step=eng.step_no + 1, shared_prefix_boundary=spb,
                     mla_hit_blocks=len(blocks.blocks[0]) if blocks.blocks else 0)
            return blocks, n, spb
        km.get_computed_blocks = gcb

    def schedule(self):
        self.step_no += 1
        so = self.s.schedule()
        self.queue.appendleft((so, self.step_no))
        return so

    def update(self, commit_fn):
        so, n = self.queue.pop()
        commits = {r: commit_fn(r, so, n) for r in so.num_scheduled_tokens}
        self.s.update_from_output(so, model_output(so, commits))
        return so, n


def load_ids(fcroot):
    out = {}
    man = json.load(open(os.path.join(fcroot, "fixtures", "ids_manifest.json")))
    for n, m in man["fixtures"].items():
        p = os.path.join(fcroot, "fixtures", m["file"])
        raw = open(p, "rb").read()
        assert hashlib.sha256(raw).hexdigest() == m["file_sha256"], n
        out[n] = json.loads(gzip.decompress(raw))
    return out


def filler(n, seed):
    r = random.Random(seed)
    return [r.randrange(1000, 150000) for _ in range(n)]


def json_fp(so):
    """The guard's step fingerprint (mapguard_hooks.fingerprint, copied verbatim): scheduled counts, finished ids and
    num_computed per scheduled request. Equal fingerprints = the same step as recorded."""
    nc = [(str(nr.req_id), int(nr.num_computed_tokens)) for nr in so.scheduled_new_reqs]
    cr = so.scheduled_cached_reqs
    nc += [(str(r), int(c)) for r, c in zip(cr.req_ids, cr.num_computed_tokens)]
    key = json.dumps([sorted((str(k), int(v)) for k, v in so.num_scheduled_tokens.items()),
                      sorted(str(x) for x in (so.finished_req_ids or ())), sorted(nc)])
    return hashlib.sha1(key.encode()).hexdigest()[:16]


class Tracker:
    """Provenance of every cached block, from the image's own BlockPool calls (wrapped, behaviour unchanged):
    owner request, block index, group, and whether the block's token range was written by prefill (end <= the owner's
    prompt length) or includes decode-written history. Logs every eviction (which cached block, whose, at which step)
    and every lookup's restored blocks with their provenance."""

    def __init__(self, sched, engine, label_of):
        self.s, self.eng, self.label_of = sched, engine, label_of
        self.owner = {}            # block_id -> (label, idx, group, kind, P_owner, hash)
        self.evictions = []        # (step, block_id, owner tuple)
        self.restores = {}         # rid -> list of (group, idx, block_id, owner)
        pool = sched.kv_cache_manager.block_pool
        o_cache, o_evict = pool.cache_full_blocks, pool._maybe_evict_cached_block
        trk = self

        def cache_full_blocks(request, blocks, num_cached_blocks, num_full_blocks, block_size, kv_cache_group_id,
                              block_mask=None):
            before = {b.block_id: b.block_hash for b in blocks[num_cached_blocks:num_full_blocks] if not b.is_null}
            r = o_cache(request, blocks, num_cached_blocks, num_full_blocks, block_size, kv_cache_group_id,
                        block_mask=block_mask)
            for i in range(num_cached_blocks, num_full_blocks):
                b = blocks[i]
                if b.is_null or b.block_hash is None or before.get(b.block_id) is not None:
                    continue
                end = (i + 1) * block_size
                kind = "prefill" if end <= request.num_prompt_tokens else "decode"
                trk.owner[b.block_id] = (trk.label_of(request.request_id), i, kv_cache_group_id, kind,
                                         request.num_prompt_tokens, b.block_hash)
            return r

        def evict(block):
            h = block.block_hash
            ow = trk.owner.get(block.block_id)
            ev = o_evict(block)
            if ev:
                trk.evictions.append((trk.eng.step_no, block.block_id, ow))
                trk.owner.pop(block.block_id, None)
            return ev
        pool.cache_full_blocks = cache_full_blocks
        pool._maybe_evict_cached_block = evict
        km = sched.kv_cache_manager
        o_gcb = km.get_computed_blocks

        def gcb(request):
            blocks, n, spb = o_gcb(request)
            lst = []
            for g, bl in enumerate(blocks.blocks):
                for i, b in enumerate(bl):
                    if not b.is_null:
                        lst.append((g, i, b.block_id, trk.owner.get(b.block_id)))
            trk.restores[request.request_id] = dict(hit=n, step=trk.eng.step_no + 1, blocks=lst)
            return blocks, n, spb
        km.get_computed_blocks = gcb

    def free_queue_position(self, block_ids):
        """Position (0 = next to be reused) of each given block in the free queue; None if live or absent."""
        pool = self.s.kv_cache_manager.block_pool
        q = pool.free_block_queue
        pos, i = {}, 0
        b = q.fake_free_list_head.next_free_block
        want = set(block_ids)
        while b is not None and b is not q.fake_free_list_tail:
            if b.block_id in want:
                pos[b.block_id] = i
            i += 1
            b = b.next_free_block
        return {x: pos.get(x) for x in block_ids}
