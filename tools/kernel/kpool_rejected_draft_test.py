"""Does tpurtell v0.8.0's kpool decode kernel corrupt a pool when a
pool-completing speculative draft is rejected? (vLLM PR #58454 scenario)

GLM-5.3-Flash: index_kpool = 4, index_head_dim = 128. Pool W = positions 4..7.
Positions 4, 5 accepted earlier. A verify step stashes 6 plus num_spec drafts.
If draft 7 (pool-completing) is rejected, drafts beyond it land in ring slots
p % 4 - i.e. on top of the committed keys of 4, 5, ... - before step t+1
re-verifies from 7 and recompresses pool W.

Reference = the accepted tokens fed one per step, no speculation.
Adapted from the standalone repro in vllm-project/vllm#58454.
"""
import inspect
import torch
from vllm.models.glm5next.nvidia.ops.kpool_compress import (
    kpool_decode_update_and_maybe_write_cache_batched as update,
)

D, P, PAGE, BLOCK, POOL_LOC = 128, 4, 64, 3, 5
PATCHED = "ring = tail_kv_cache.shape[2]" in inspect.getsource(update)


def ring_for(num_spec):
    """Unpatched: one pool (the wrapper asserts it). Patched: the #58454 sizing."""
    if not PATCHED:
        return P
    n = -(-(P + num_spec) // P)
    return P * (1 << (n - 1).bit_length())
dev = "cuda"
torch.manual_seed(0)
rnd = lambda: torch.randn(D, dtype=torch.bfloat16, device=dev)  # noqa: E731
keys = {p: rnd() for p in range(4, 20)}
gates = {p: rnd() for p in range(4, 20)}
dkeys = {p: rnd() for p in range(4, 20)}   # wrong (draft) values
dgates = {p: rnd() for p in range(4, 20)}
ape = torch.randn(P, D, dtype=torch.float32, device=dev)


def fresh(ring):
    return (torch.zeros(4, PAGE, D + 4, dtype=torch.uint8, device=dev),
            torch.zeros(8, 2, ring, D, dtype=torch.bfloat16, device=dev))


def step(kv, tail, positions, K, G):
    n = len(positions); RING = tail.shape[2]
    t = lambda xs: torch.tensor([xs], dtype=torch.int32, device=dev)  # noqa: E731
    update(kv, tail, t([BLOCK * RING + p % RING for p in positions]),
           torch.stack(K).view(1, n, D), torch.stack(G).view(1, n, D), ape,
           t([POOL_LOC if p == 7 else (99 if p % P == P - 1 else -1) for p in positions]),
           t(positions), pool_size=P, head_dim=D)


def pool(kv):
    return kv.view(-1, D + 4)[POOL_LOC].clone()


# reference: no speculation, accepted tokens one per step
kv, tail = fresh(P)
for p in (4, 5, 6, 7):
    step(kv, tail, [p], [keys[p]], [gates[p]])
ref = pool(kv)


def run(num_spec, reject):
    kv, tail = fresh(ring_for(num_spec))
    for p in (4, 5):
        step(kv, tail, [p], [keys[p]], [gates[p]])
    pos = list(range(6, 6 + 1 + num_spec))           # 6 + num_spec drafts
    if not reject:                                   # every draft is correct
        step(kv, tail, pos, [keys[p] for p in pos], [gates[p] for p in pos])
    else:                                            # 6 ok, draft 7 onward rejected
        step(kv, tail, pos, [keys[6]] + [dkeys[p] for p in pos[1:]],
             [gates[6]] + [dgates[p] for p in pos[1:]])
        pos2 = list(range(7, 7 + 1 + num_spec))      # re-verify from the true 7
        step(kv, tail, pos2, [keys[7]] + [dkeys[p] for p in pos2[1:]],
             [gates[7]] + [dgates[p] for p in pos2[1:]])
    got = pool(kv)
    bad = (ref[:D] != got[:D]).sum().item()
    return torch.equal(ref, got), bad, tail.shape[2]


print(f"kpool={P}; kernel {'PATCHED (ring sized for drafts)' if PATCHED else 'unpatched (ring = one pool)'}\n")
print(f"{'scenario':58s} {'ring':>4s}  {'pool W correct?':16s} key bytes wrong")
cases = [
    ("control: 3 drafts, all accepted", 3, False),
    ("1 draft token (DFLASH_TOKENS=1), pool-completing draft rejected", 1, True),
    ("2 draft tokens, pool-completing draft rejected", 2, True),
    ("3 draft tokens (v0.8.0 default), rejected", 3, True),
    ("5 draft tokens (v0.7.0 setting), rejected", 5, True),
    ("7 draft tokens (checkpoint max), rejected", 7, True),
]
for name, n, rej in cases:
    ok, bad, ring = run(n, rej)
    print(f"{name:58s} {ring:4d}  {'yes' if ok else 'NO - CORRUPT':16s} {bad}/{D}")
