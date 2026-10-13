#!/usr/bin/env python3
"""CPU check (Triton interpreter, NO GPU) of B0's index-K prefill write at an unaligned chunk start.

B0's _kpool_compress_insert (sparse_attn_indexer_kpool.py:102-147) pools the 4 batch tokens
t-3..t of the prefill slice for every token whose pool-granular slot is valid, and masks
local index < 3 ("Assumes pool-aligned chunk starts").  The scheduler does not align chunk
starts to 4 (scheduler.py:378-440, mamba split only aligns to 4,352-token blocks), so:
  (a) first prefill row of the slice, start % 4 != 0: the pool completing at local index < 3
      is never written;
  (b) a later row starting unaligned: its first completing pool is compressed from the
      previous row's last tokens.
This test runs the REAL B0 functions on a small index-K cache and compares with a reference
that pools each request's own four tokens.  It also checks that the guard flags both cases.
"""
import json
import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch  # noqa: E402

import vllm.model_executor.layers.sparse_attn_indexer  # noqa: E402,F401
import vllm.models.glm5next.nvidia.ops.kpool_compress  # noqa: E402,F401
from vllm.model_executor.layers.sparse_attn_indexer_kpool import _kpool_compress_insert  # noqa: E402
from vllm.models.glm5next.nvidia.ops.kpool_compress import kpool_compress_and_write_cache  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "guard"))
try:
    from vllm import mapguard_hooks as mg
except Exception:  # noqa: BLE001
    import mapguard_hooks as mg  # noqa: E402

PPB, KP, HD = 1088, 4, 128
NBLK = 4
g = torch.Generator().manual_seed(5)


def cache():
    return torch.full((NBLK * 17, 64, 132), 0xA5, dtype=torch.uint8)


def rec(c, loc):
    """Index-K record of pool slot loc: page-planar page of 64 records = 64 x 128 E4M3 K bytes,
    then 64 fp32 scales at byte 8,192 (b0-cache/PROPOSAL.md section 1)."""
    f = c.reshape(-1)
    pg, r = loc // 64, loc % 64
    base = pg * 8448
    return torch.cat([f[base + r * 128: base + r * 128 + 128], f[base + 8192 + r * 4: base + 8192 + r * 4 + 4]]).clone()


# request A (first prefill row): previous chunk ended at position 2047 -> this chunk starts at
# 2047 (unaligned, 2047 % 4 == 3); its tokens 2047..2052 live in block 1.
# request B (second row): chunk starts at 6 (6 % 4 == 2), tokens 6..13, block 2.
A_blk, B_blk = 1, 2
A_pos = list(range(2047, 2053))
B_pos = list(range(6, 14))
pos = A_pos + B_pos
n = len(pos)
blk = [A_blk] * len(A_pos) + [B_blk] * len(B_pos)
slot = torch.tensor([b * PPB + (p // KP) % PPB if (p + 1) % KP == 0 else -1 for b, p in zip(blk, pos)],
                    dtype=torch.int64)
k_all = torch.randn(n, HD, generator=g).to(torch.bfloat16)
gate_all = torch.randn(n, HD, generator=g).to(torch.bfloat16)
ape = (torch.randn(KP, HD, generator=g) * 0.1).float()
# each request's own earlier tokens (from its previous chunk), needed by the reference
A_prev = torch.randn(3, HD, generator=g).to(torch.bfloat16)      # positions 2044..2046
B_prev = torch.randn(2, HD, generator=g).to(torch.bfloat16)      # positions 4..5
gA_prev = torch.randn(3, HD, generator=g).to(torch.bfloat16)
gB_prev = torch.randn(2, HD, generator=g).to(torch.bfloat16)

c = cache()
_kpool_compress_insert(k_all, gate_all, ape, c, slot, KP, HD, round_scale=True)

out = {}
# (a) A's pool 511 (positions 2044..2047) completes at local index 0 -> masked
locA = A_blk * PPB + 2047 // KP
out["a_pool_written"] = not bool((rec(c, locA) == 0xA5).all())
# (b) B's pool 1 (positions 4..7) completes at batch token 7 (B local 1) and is pooled from
#     batch tokens 4..7 = A's 2051, 2052 and B's 6, 7
locB = B_blk * PPB + 7 // KP
ref = cache()
kB = torch.cat([B_prev, k_all[6:8]])[None]
gB = torch.cat([gB_prev, gate_all[6:8]])[None]
kpool_compress_and_write_cache(ref, kB, gB, ape, torch.tensor([locB], dtype=torch.int64), pool_size=KP,
                               head_dim=HD, round_scale=True)
contam = cache()
kpool_compress_and_write_cache(contam, k_all[4:8][None], gate_all[4:8][None], ape,
                               torch.tensor([locB], dtype=torch.int64), pool_size=KP, head_dim=HD,
                               round_scale=True)
out["b_pool_written"] = not bool((rec(c, locB) == 0xA5).all())
out["b_equals_own_tokens_reference"] = bool(torch.equal(rec(c, locB), rec(ref, locB)))
out["b_equals_cross_request_pooling"] = bool(torch.equal(rec(c, locB), rec(contam, locB)))
# aligned pools later in the chunk are correct (control): A pool 512 = positions 2048..2051
locA2 = A_blk * PPB + 2051 // KP
refA = cache()
kpool_compress_and_write_cache(refA, k_all[1:5][None], gate_all[1:5][None], ape,
                               torch.tensor([locA2], dtype=torch.int64), pool_size=KP, head_dim=HD,
                               round_scale=True)
out["control_aligned_pool_correct"] = bool(torch.equal(rec(c, locA2), rec(refA, locA2)))

# guard on the same inputs
geo = mg.Geo(nb=1275, R=16, W0=241)
dev = mg.Dev(geo, torch.device("cpu"))
a, _ = (None, None)
lay = dev.lay
arr = torch.zeros(lay.size, dtype=torch.int64)
v = lay.np_views(arr.numpy())
v["hdr"][mg.H_ARMED], v["hdr"][mg.H_STEP], v["hdr"][mg.H_NROWS], v["hdr"][mg.H_NTOK] = 1, 1, 2, n
v["qsl"][:] = n
v["qsl"][0], v["qsl"][1] = 0, len(A_pos)
v["tag"][:2] = [1, 2]
v["nsched"][:2] = [len(A_pos), len(B_pos)]
v["lo"][:2] = v["hi"][:2] = [2047, 6]
v["bt0"][0, :1] = [A_blk]
v["bt0"][1, :1] = [B_blk]
v["bt0"][0, 0] = A_blk
v["nb0"][:2] = [1, 1]
v["col0"][:] = -1
v["grp"][:] = -1
v["col0"][A_blk] = 0
v["col0"][B_blk] = 0
v["grp"][A_blk] = v["grp"][B_blk] = 0
v["tagx"][A_blk], v["tagx"][B_blk] = 1, 2
dev.E.copy_(arr)
mg.ENABLED = True
mg.S.dev, mg.S.rank = dev, 0
mg.hook_idx_prefill_write("model.layers.3.x", slot, torch.tensor(pos), 0, n, KP)
cnt = dev.cnt[mg.S_IDX_W_PF]
out["guard_missing_pred"] = int(cnt[mg.CI["missing_pred"]])
out["guard_pool_cross"] = int(cnt[mg.CI["pool_cross"]])
out["guard_other_violations"] = {c_: int(cnt[mg.CI[c_]]) for c_ in mg.VIOLATION_COUNTERS
                                 if c_ not in ("missing_pred", "pool_cross") and int(cnt[mg.CI[c_]])}
ok = (not out["a_pool_written"] and out["b_pool_written"] and not out["b_equals_own_tokens_reference"]
      and out["b_equals_cross_request_pooling"] and out["control_aligned_pool_correct"]
      and out["guard_missing_pred"] == 1 and out["guard_pool_cross"] == 1 and not out["guard_other_violations"])
out["ok"] = ok
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "kpool-unaligned.json"), "w"), indent=1)
