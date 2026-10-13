# CPU-only reachability estimate for the b0-mapguard workload (PLAN.md amendment 17, item 4).
# Drives the REAL B0 KVCacheManager / HybridKVCacheCoordinator / BlockPool (10-group B0 scheduler
# config from b0-cache/reachability/sim_pool.py:build_config) with the proposed client phases.
# Long-context prompts use the REAL stage-2 token ids (so prefix-cache hits and partial-block
# hits are the real ones); GPQA prompts use random ids of the recorded lengths (no hits, < 1088).
# Scheduler approximation (inference, as in sim_pool.py): chunked prefill 2048 per step shared by
# the running requests in admission order, decode = 1 + 3 drafts scheduled (+4 lookahead), a fixed
# mean accepted count per step. Reports when group 0 (MLA+index-K) and the tail group are first
# handed block ids >= 935 and how many >= 935 blocks they hold per phase.
import json, random, sys
from collections import defaultdict
sys.path.insert(0, ".")
from sim_pool import build_config, NUM_BLOCKS, HIGH  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.utils.hashing import get_hash_fn_by_name  # noqa: E402
from vllm.v1.core.kv_cache_manager import KVCacheManager  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
from vllm.v1.request import Request  # noqa: E402

GPQA_LENS = [124, 134, 158, 133, 365, 181, 172, 447, 238, 131, 376, 146, 322, 131, 283, 204]
GPQA_GEN = [1024, 1828, 651, 1270, 1044, 1025, 1128, 2048, 2048, 678, 2048, 1027, 2048, 2048, 743, 2048]


def phases(variant, stage2, qcap):
    L = {r["id"]: r["ids"] for r in stage2["requests"]}
    A8 = ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"]
    ph = []
    ph.append(("warm", [(L[r], 1) for r in ["r7", "r6", "r5", "r4", "r3", "r2", "r1", "r0"]], 1))
    reps = 2 if variant != "gate" else 4
    for k in range(reps):
        ph.append((f"T{k}", [(L[r], 200) for r in A8], 8))
        ph.append((f"F{k}", [(L[r], 128) for r in ["r0", "r2", "r4", "r7"]], 4))
    ph.append(("D", [(L[r], 40 * (8 - k)) for k, r in enumerate(A8)], 8))
    ph.append(("S", [(L["r0"], 300), (L["r2"], 300), (L["r4"], 300), (L["r7"], 300),
                     (L["r1"], 120), (L["r3"], 120), (L["r5"], 120), (L["r6"], 120)], 8))
    q = [([1000 + 7 * d + i for i in range(GPQA_LENS[d])], min(GPQA_GEN[d], qcap)) for d in range(16)]
    ph.append(("Q", q, 8))
    if variant == "gate":
        ph.append(("Qreuse", q, 8))
    if variant == "rerun":   # trimmed rerun: warm, T0, F0, T1, F1 only
        ph = ph[:5]
    return ph


def precycle(mgr, hasher, target):
    """Advance the free-queue head to ~target (models the stack's release warm-up, which handed
    out ids up to 528 in run-a2) with hashed requests the workload never sees."""
    k = 0
    while True:
        ids = [900000 + 7919 * k + i for i in range(4352 * 2)]
        req = Request(f"pc{k}", ids, SamplingParams(max_tokens=1), None, block_hasher=hasher)
        mgr.new_step_starts()
        n = 0
        while n < len(ids):
            c = min(2048, len(ids) - n)
            assert mgr.allocate_slots(req, c, num_lookahead_tokens=4) is not None
            req.num_computed_tokens += c
            n += c
        mx = max(max(g) if g else 0 for g in mgr.get_block_ids(req.request_id))
        mgr.free(req)
        k += 1
        if mx >= target:
            return mx


def main(variant="mapguard", acc=2.6, qcap=1024, seed=0, pre=0):
    rng = random.Random(seed)
    stage2 = json.load(open("stage2_requests.json"))
    cfg, names = build_config()
    hash_fn = get_hash_fn_by_name("sha256")
    init_none_hash(hash_fn)
    mgr = KVCacheManager(cfg, max_model_len=1_048_576, scheduler_block_size=4352,
                         hash_block_size=1088, max_in_flight_tokens=2048,
                         enable_caching=True, use_eagle=True)
    hasher = get_request_block_hasher(1088, hash_fn)
    pre_max = precycle(mgr, hasher, pre) if pre else 0
    step = 0
    first_high = {}
    held_high = defaultdict(lambda: defaultdict(set))  # phase -> group -> ids >= HIGH held
    prev = {}
    phase_steps = {}
    partial_hits = []
    hits = []
    unaligned = []
    ids_of = lambda r: r.prompt_token_ids
    rid = 0
    max_id = defaultdict(int)
    for tag, jobs, width in phases(variant, stage2, qcap):
        s0 = step
        pending = list(jobs)
        running = []
        while pending or running:
            while pending and len(running) < width:
                ids, gen = pending.pop(0)
                req = Request(f"q{rid}", list(ids), SamplingParams(max_tokens=gen), None,
                              block_hasher=hasher)
                rid += 1
                req._hit = None
                running.append([req, gen])
            mgr.new_step_starts()
            budget = 2048
            pf_rank = 0
            # decode rows first (as the scheduler schedules running decodes before prefills)
            for ent in sorted(running, key=lambda e: e[0].num_computed_tokens < e[0].num_prompt_tokens):
                req, rem = ent
                if req.num_computed_tokens < req.num_prompt_tokens:
                    first = req.num_computed_tokens == 0
                    if first:
                        comp, n_hit = mgr.get_computed_blocks(req)[:2]
                    else:
                        comp, n_hit = None, 0
                    n = min(req.num_prompt_tokens - req.num_computed_tokens - n_hit, budget)
                    if n <= 0:
                        continue
                    blocks = mgr.allocate_slots(req, n, num_new_computed_tokens=n_hit,
                                                new_computed_blocks=comp, num_lookahead_tokens=4)
                    start_pos = req.num_computed_tokens + n_hit
                    if blocks is not None and start_pos % 4 != 0:
                        unaligned.append((tag, step, pf_rank, start_pos))
                    if blocks is None and first:
                        continue  # deferred to the next step (mamba: hit cached this step)
                    if blocks is None:
                        raise SystemExit(f"allocation failed (prefill) phase={tag} step={step} len={req.num_prompt_tokens} computed={req.num_computed_tokens} n={n} hit={n_hit} free={mgr.block_pool.get_num_free_blocks()}")
                    if first and n_hit:
                        hits.append((tag, len(ids_of(req)), n_hit, n_hit % 4352 != 0))
                    req.num_computed_tokens += n + n_hit
                    pf_rank += 1
                    budget -= n
                    if req.num_computed_tokens >= req.num_prompt_tokens:
                        req.append_output_token_ids(rng.randrange(1000, 150000))
                        ent[1] -= 1
                else:
                    if budget < 4:
                        continue
                    blocks = mgr.allocate_slots(req, 4, num_lookahead_tokens=4)
                    if blocks is None:
                        raise SystemExit("allocation failed (decode)")
                    budget -= 4
                    a = min(max(1, int(acc + rng.random())), 4, max(ent[1], 1))
                    req.num_computed_tokens += a
                    for _ in range(a):
                        req.append_output_token_ids(rng.randrange(1000, 150000))
                    ent[1] -= a
                held = set()
                for g, ids_ in enumerate(mgr.get_block_ids(req.request_id)):
                    for b in ids_:
                        if b:
                            held.add((g, b))
                            max_id[g] = max(max_id[g], b)
                            if b >= HIGH:
                                held_high[tag][names[g]].add(b)
                                if g not in first_high:
                                    first_high[g] = (step, tag, step - s0)
                prev[req.request_id] = held
                if ent[1] <= 0:
                    running.remove(ent)
                    mgr.free(req)
                    prev.pop(req.request_id, None)
            step += 1
        phase_steps[tag] = (s0, step)
    res = dict(variant=variant, accepted_per_step=acc, qcap=qcap, precycled_to=pre_max, steps=step,
               phase_steps=phase_steps,
               first_id_ge_935={names[g]: dict(step=v[0], phase=v[1], step_in_phase=v[2])
                                 for g, v in sorted(first_high.items())},
               blocks_ge_935_held_per_phase={t: {n: len(s) for n, s in d.items()} for t, d in held_high.items()},
               max_id={names[g]: v for g, v in sorted(max_id.items())},
               prefix_hits=hits[:40], n_prefix_hits=len(hits), n_partial_mla_block_hits=sum(1 for h in hits if h[3]),
               unaligned_prefill_chunk_starts=len(unaligned),
               unaligned_first_in_slice=sum(1 for u in unaligned if u[2] == 0),
               unaligned_later_in_slice=sum(1 for u in unaligned if u[2] > 0),
               unaligned_by_phase={t: sum(1 for u in unaligned if u[0] == t) for t in phase_steps},
               unaligned_examples=unaligned[:10])
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    v = sys.argv[1] if len(sys.argv) > 1 else "mapguard"
    acc = float(sys.argv[2]) if len(sys.argv) > 2 else 2.6
    pre = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    out = main(v, acc, pre=pre)
    json.dump(out, open(f"out-{v}-acc{acc}" + (f"-pre{pre}" if pre else "") + ".json", "w"), indent=1)
