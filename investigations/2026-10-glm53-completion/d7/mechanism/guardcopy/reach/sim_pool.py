# CPU-only reachability simulation for checkpoint B0's KV block pool.
# Drives the REAL vLLM KVCacheManager / HybridKVCacheCoordinator / BlockPool of the
# B0 image (no GPU, no model) with the shipped group geometry (10 groups, 1,275 blocks)
# and the token counts of the two B0 GPQA passes (requests.jsonl from the record proxy).
# Reports which block ids each group is handed, when ids >= 935 (MLA page start > 2^31 B)
# are first allocated, and how often a block id moves between groups (cross-group reuse
# of the same bytes in a shared slot tensor).
# Inference, stated: the scheduler loop is approximated (concurrency 8 from the eval
# client, chunked prefill 2048, decode step = 1 + 3 drafts scheduled, a fixed accepted
# count per step); block-pool behaviour itself is the real code.
import json, os, random, sys
from collections import defaultdict

import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config, get_request_block_hasher, init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, KpoolTailSpec, MLAAttentionSpec,
    MambaSpec, SlidingWindowSpec, UniformTypeKVCacheSpecs,
)
from vllm.v1.request import Request

NUM_BLOCKS = 1275
MLA_PAGE = 2_297_856
HIGH = 935  # first block whose MLA-tensor page starts above 2^31 bytes


def build_config():
    mla = MLAAttentionSpec(block_size=4352, num_kv_heads=1, head_size=512,
                           dtype=torch.uint8, cache_dtype_str="fp8_ds_mla")
    idx = MLAAttentionSpec(block_size=4352, num_kv_heads=1, head_size=132,
                           dtype=torch.uint8, compress_ratio=4)
    attn = UniformTypeKVCacheSpecs.from_specs({"mla.3": mla, "idx.3": idx})
    tail = KpoolTailSpec(block_size=8, num_kv_heads=1, head_size=128,
                         dtype=torch.bfloat16, sliding_window=8, page_size_padded=143_616)
    tailu = UniformTypeKVCacheSpecs.from_specs({"tail.3": tail})
    mamba = MambaSpec(block_size=4352, shapes=((32 * 128 * 128,), (3 * 12288,)),
                      dtypes=(torch.float32, torch.float32), page_size_padded=MLA_PAGE,
                      mamba_cache_mode="align", num_speculative_blocks=3)
    draft = SlidingWindowSpec(block_size=1088, num_kv_heads=4, head_size=128,
                              dtype=torch.bfloat16, sliding_window=2048,
                              page_size_padded=MLA_PAGE)
    draftu = UniformTypeKVCacheSpecs.from_specs({"draft.0": draft})
    groups = [KVCacheGroupSpec(["mla.3", "idx.3"], attn), KVCacheGroupSpec(["tail.3"], tailu)]
    groups += [KVCacheGroupSpec([f"kda.{g}"], mamba) for g in range(7)]
    groups += [KVCacheGroupSpec(["draft.0"], draftu)]
    names = ["MLA+idx", "tail"] + [f"KDA{g}" for g in range(7)] + ["draft"]
    cfg = KVCacheConfig(num_blocks=NUM_BLOCKS, kv_cache_tensors=[], kv_cache_groups=groups)
    # the scheduler sees each uniform group as its first spec (kv_cache_utils.py:2226-2245)
    return generate_scheduler_kv_cache_config([cfg]), names


def main(reqlog, out, accepted_per_step=2.6, concurrency=8, seed=0):
    rng = random.Random(seed)
    cfg, names = build_config()
    hash_fn = get_hash_fn_by_name("sha256")
    init_none_hash(hash_fn)
    hash_block = 1088  # gcd of participating group block sizes (resolve_kv_cache_block_sizes)
    mgr = KVCacheManager(cfg, max_model_len=1_048_576, scheduler_block_size=4352,
                         hash_block_size=hash_block, max_in_flight_tokens=2048,
                         enable_caching=True, use_eagle=True)
    hasher = get_request_block_hasher(hash_block, hash_fn)
    recs = [json.loads(l) for l in open(reqlog)]
    recs.sort(key=lambda r: r["t_start"])
    pending = list(enumerate(recs))
    running = []  # [req, remaining_output]
    owner = {}  # block id -> last group index that held it
    first_high_step = {}
    group_ids = defaultdict(set)
    transitions = defaultdict(int)
    reuse_same_group = 0
    allocs = 0
    step = 0
    tok_total = 0
    prev = defaultdict(set)  # request -> set of (group, block) currently held
    violations = []
    max_live_id = 0
    while pending or running:
        mgr.new_step_starts() if hasattr(mgr, "new_step_starts") else None
        while pending and len(running) < concurrency:
            i, r = pending.pop(0)
            p = r["prompt_tokens"]
            # distinct prompts: GPQA prompts share only a short template prefix (< 1088 tokens)
            ids = [rng.randrange(1000, 150000) for _ in range(p)]
            req = Request(f"r{i}", ids, SamplingParams(max_tokens=r["max_tokens"]), None,
                          block_hasher=hasher)
            comp, n_hit = mgr.get_computed_blocks(req)[:2]
            req._sim_new_computed = (comp, n_hit)
            running.append([req, r["completion_tokens"]])
        budget = 2048
        for ent in list(running):
            req, rem = ent
            if req.num_computed_tokens < req.num_prompt_tokens:
                n = min(req.num_prompt_tokens - req.num_computed_tokens, budget)
                if n <= 0:
                    continue
                comp, n_hit = req._sim_new_computed if req.num_computed_tokens == 0 else (None, 0)
                blocks = mgr.allocate_slots(req, n, num_new_computed_tokens=n_hit,
                                            new_computed_blocks=comp, num_lookahead_tokens=4)
                if blocks is None:
                    raise SystemExit("allocation failed (prefill)")
                req.num_computed_tokens += n
                budget -= n
                if req.num_computed_tokens >= req.num_prompt_tokens:
                    req.append_output_token_ids(rng.randrange(1000, 150000))
                    ent[1] -= 1
            else:
                blocks = mgr.allocate_slots(req, 4, num_lookahead_tokens=4)
                if blocks is None:
                    raise SystemExit("allocation failed (decode)")
                acc = min(max(1, int(accepted_per_step + rng.random())), 4, max(ent[1], 1))
                req.num_computed_tokens += acc
                for _ in range(acc):
                    req.append_output_token_ids(rng.randrange(1000, 150000))
                ent[1] -= acc
            held = set()
            for g, ids_ in enumerate(mgr.get_block_ids(req.request_id)):
                for b in ids_:
                    if b == 0:
                        continue
                    held.add((g, b))
            for (g, b) in held - prev[req.request_id]:
                allocs += 1
                group_ids[g].add(b)
                if b >= HIGH and g not in first_high_step:
                    first_high_step[g] = (step, tok_total)
                last = owner.get(b)
                if last is not None:
                    if last == g:
                        reuse_same_group += 1
                    else:
                        transitions[(names[last], names[g])] += 1
                owner[b] = g
            prev[req.request_id] = held
            if ent[1] <= 0:
                running.remove(ent)
                mgr.free(req)
                del prev[req.request_id]
            tok_total += 0
        # exclusivity: a block id may be held by at most one group at a time
        live = defaultdict(set)
        for rid, hs in prev.items():
            for (g, b) in hs:
                live[b].add(g)
        for b, gs in live.items():
            if len(gs) > 1:
                violations.append((step, b, sorted(names[g] for g in gs)))
        max_live_id = max([max_live_id] + list(live))
        step += 1
        tok_total = step
    res = {
        "reqlog": reqlog, "requests": len(recs), "steps": step,
        "accepted_per_step": accepted_per_step, "concurrency": concurrency,
        "num_blocks": NUM_BLOCKS, "high_threshold_block": HIGH,
        "per_group": {
            names[g]: {"distinct_ids": len(s), "max_id": max(s) if s else None,
                       "ids_ge_935": sum(1 for b in s if b >= HIGH)}
            for g, s in sorted(group_ids.items())
        },
        "first_step_with_id_ge_935": {names[g]: v[0] for g, v in first_high_step.items()},
        "all_ids_reached": len(set().union(*group_ids.values())),
        "block_allocations": allocs,
        "cross_group_simultaneous_holds": len(violations),
        "cross_group_examples": violations[:5],
        "first_step_fraction_ge_935": {names[g]: round(v[0] / step, 4) for g, v in first_high_step.items()},
        "reallocated_to_same_group": reuse_same_group,
        "reallocated_across_groups": {f"{a}->{b}": n for (a, b), n in sorted(transitions.items())},
    }
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 2.6)
