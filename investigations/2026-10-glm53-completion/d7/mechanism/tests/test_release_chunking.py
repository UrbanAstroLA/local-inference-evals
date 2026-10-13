#!/usr/bin/env python3
"""D7 step 0a, item 2: can a release's served configuration start a prefill chunk off a 4-token pool
boundary? CPU only, no engine start.

REAL code: the image's ``Scheduler._mamba_block_aligned_split`` (called unbound on a stand-in ``self``
carrying the served values). EMULATED from source: the per-request budget lines of
``Scheduler.schedule`` (running: scheduler.py:562-564 and 689-690; waiting: 916 and 1115-1116 in
0.9.1/B0; one line lower before 409 in 0.7.0/0.8.0): ``num_new = min(remaining, token_budget,
input_budget - draft_slots)``, then the split; ``token_budget -= num_new``;
``input_budget -= num_new + draft_slots``. Decoding requests are scheduled first with 1 + K tokens.

usage: test_release_chunking.py <label> <block_size> <draft_slots K> <out.json>
"""
import inspect
import json
import sys
from types import SimpleNamespace as NS

from vllm.v1.core.sched.scheduler import Scheduler

label, block_size, K, out_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
BUDGET = 2048  # max_num_batched_tokens (served); max_num_scheduled_tokens logged as 2048
split = Scheduler._mamba_block_aligned_split
src_lines = inspect.getsourcelines(split)


def fake_self(eagle_groups, partial_hits, hash_bs):
    return NS(cache_config=NS(block_size=block_size), use_eagle=True,
              kv_cache_manager=NS(coordinator=NS(eagle_group_ids=eagle_groups)),
              max_num_scheduled_tokens=BUDGET, scheduler_config=NS(long_prefill_token_threshold=0),
              mamba_partial_cache_hit=partial_hits, hash_block_size=hash_bs)


def chunk(selfv, req, remaining_budget_tokens, input_budget):
    n = min(req.num_tokens - req.num_computed_tokens, remaining_budget_tokens, input_budget - K)
    return split(selfv, req, n) if n > 0 else 0


def run_prompt(selfv, L, n_decodes=0, prefix_hit=0, first_prefill=0):
    """Prefill one prompt of L tokens to completion. Each step: n_decodes running decodes (1+K tokens
    each) and optionally another prefill taking ``first_prefill`` tokens are scheduled first."""
    req = NS(num_computed_tokens=prefix_hit, num_prompt_tokens=L, num_tokens=L, shared_prefix_boundary=0)
    starts, steps = [], 0
    while req.num_computed_tokens < L and steps < 1000:
        tb, ib = BUDGET, BUDGET
        for _ in range(n_decodes):
            tb -= 1 + K
            ib -= 1 + K + K
        if first_prefill:
            tb -= first_prefill
            ib -= first_prefill + K
        n = chunk(selfv, req, tb, ib)
        if n <= 0:
            break
        starts.append((req.num_computed_tokens, n))
        req.num_computed_tokens += n
        steps += 1
    return starts


out = {"label": label, "block_size": block_size, "K": K, "budget": BUDGET,
       "split_source": f"{inspect.getsourcefile(split)}:{src_lines[1]}", "runs": []}
variants = [("eagle_groups=[0]", [0], False, block_size), ("eagle_groups=[]", [], False, block_size),
            ("partial_hits hash=1088", [0], True, 1088)]
for vname, eg, ph, hb in variants:
    s = fake_self(eg, ph, hb)
    for L in (1500, 2795, 9000, 20000):
        for nd in (0, 1, 2, 5):
            st = run_prompt(s, L, n_decodes=nd)
            un = [(a, n) for a, n in st if a % 4]
            out["runs"].append({"variant": vname, "L": L, "n_decodes": nd, "chunks": st,
                                "unaligned_starts": un, "residues": sorted({a % 4 for a, _ in un})})
    # a second prefill sharing the step (first request takes X tokens)
    for X in (21, 100, 1127):
        st = run_prompt(s, 9000, first_prefill=X)
        un = [(a, n) for a, n in st if a % 4]
        out["runs"].append({"variant": vname, "L": 9000, "shared_first_prefill": X, "chunks": st[:4],
                            "unaligned_starts": un[:6], "residues": sorted({a % 4 for a, _ in un})})
    # prefix hit at a block boundary, then continuation
    st = run_prompt(s, 3 * block_size + 1500, prefix_hit=2 * block_size)
    out["runs"].append({"variant": vname, "L": 3 * block_size + 1500, "prefix_hit": 2 * block_size,
                        "chunks": st, "unaligned_starts": [(a, n) for a, n in st if a % 4]})

alone = [r for r in out["runs"] if r.get("n_decodes") == 0 and "shared_first_prefill" not in r
         and "prefix_hit" not in r and r["variant"] == "eagle_groups=[0]"]
out["single_request_unaligned_starts"] = {r["L"]: r["unaligned_starts"] for r in alone}
out["any_unaligned"] = any(r["unaligned_starts"] for r in out["runs"])
out["all_residues_seen"] = sorted({x for r in out["runs"] for x in r.get("residues", [])})
print(json.dumps({k: out[k] for k in ("label", "block_size", "K", "split_source",
                                       "single_request_unaligned_starts", "any_unaligned",
                                       "all_residues_seen")}, indent=1))
json.dump(out, open(out_path, "w"), indent=1)
