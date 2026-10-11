#!/usr/bin/env python3
"""CPU-only mechanism check for the rank-1 tail-ring corruption (T0).

Runs inside the 0.9.1 release image with no GPU (TRITON_INTERPRET=1 runs the
real Triton kernels on CPU).  Three checks:

 A. Static: does runner-v2's MambaHybridModelState.prepare_attn pass
    `positions` to build_attn_metadata?  KpoolTailMetadataBuilder.build only
    applies compute_kpool_tail_slot_mapping when positions is not None.
 B. Real KpoolTailMetadataBuilder.build on CPU tensors with positions=None vs
    positions given: which tail slot mapping reaches the indexer op.
 C. Real _kpool_tail_seed_kernel (Triton interpreter) over a prefill chunk
    whose tail slot mapping is the GENERIC per-group mapping
    (persistent_tail_bt[row][pos // 8] * 8 + pos % 8, transcribed from
    _compute_slot_mappings_kernel, CP_SIZE == 1) with column 1 of the row
    holding a foreign block id X.  Which (block, slot, position) does it write?
"""
import ast
import inspect
import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402

out = {}

# ---------------------------------------------------------------- A. static
import vllm.v1.worker.gpu.model_states.mamba_hybrid as mh  # noqa: E402
import vllm.v1.attention.backends.mla.indexer as idx  # noqa: E402

src = inspect.getsource(mh.MambaHybridModelState.prepare_attn)
tree = ast.parse(src.lstrip() if not src.startswith(" ") else "class _X:\n" + src)
calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
         and getattr(n.func, "id", getattr(n.func, "attr", "")) == "build_attn_metadata"]
kw = sorted(k.arg for k in calls[0].keywords) if calls else None
out["A_mamba_hybrid_build_attn_metadata_kwargs"] = kw
out["A_positions_passed"] = bool(kw and "positions" in kw)
bsrc = inspect.getsource(idx.KpoolTailMetadataBuilder.build)
out["A_tail_build_gated_on_positions"] = "if positions is not None" in bsrc
import vllm.v1.worker.gpu.attn_utils as au  # noqa: E402
sig = inspect.signature(au.build_attn_metadata)
out["A_build_attn_metadata_positions_default"] = repr(sig.parameters["positions"].default)

# ---------------------------------------------------------------- B. builder
from vllm.v1.attention.backend import CommonAttentionMetadata  # noqa: E402

RING = 8
KPOOL = 4
# two requests: row0 prefill chunk [0, 24) (24 tokens), row1 decode 4 tokens at 100..103
q = torch.tensor([0, 24, 28], dtype=torch.int32)
pos = torch.cat([torch.arange(0, 24), torch.arange(100, 104)]).to(torch.int64)
num_tok = 28
width = 131072
persist = torch.zeros(16, width, dtype=torch.int32)
persist[3, 0] = 77   # row (req_state idx) 3 -> own tail block 77
persist[5, 0] = 88   # row 5 -> own tail block 88
persist[3, 1] = 483  # stale foreign id in column 1 of row 3
idx_mapping = torch.tensor([3, 5])


def generic_slots(persist, idx_mapping, qsl, pos, block_size=RING):
    """Transcription of _compute_slot_mappings_kernel (CP_SIZE == 1)."""
    s = torch.full((pos.numel(),), -1, dtype=torch.int64)
    for b in range(idx_mapping.numel()):
        r = int(idx_mapping[b])
        for t in range(int(qsl[b]), int(qsl[b + 1])):
            p = int(pos[t])
            bi, bo = p // block_size, p % block_size
            if bi < persist.shape[1]:
                s[t] = int(persist[r, bi]) * block_size + bo
    return s


gen = generic_slots(persist, idx_mapping, q, pos)
gathered = torch.zeros(2, width, dtype=torch.int32)
gathered[0, 0], gathered[1, 0] = 77, 88  # gather copies num_blocks (=1) columns


def build_with(positions):
    b = object.__new__(idx.KpoolTailMetadataBuilder)

    class _Spec:
        block_size = RING

    b.kv_cache_spec = _Spec()
    cm = CommonAttentionMetadata(
        query_start_loc=q, query_start_loc_cpu=q, seq_lens=torch.tensor([24, 104]),
        num_reqs=2, num_actual_tokens=num_tok, max_query_len=24, max_seq_len=104,
        block_table_tensor=gathered, slot_mapping=gen.clone(), positions=positions,
    )
    try:
        md = idx.KpoolTailMetadataBuilder.build(b, 0, cm)
        return md.slot_mapping
    except Exception as e:  # noqa: BLE001
        return repr(e)


sm_none = build_with(None)
sm_pos = build_with(pos)
if isinstance(sm_none, torch.Tensor):
    out["B_slot_mapping_positions_None"] = sm_none.tolist()
    out["B_blocks_positions_None"] = (sm_none // RING).tolist()
else:
    out["B_slot_mapping_positions_None"] = sm_none
if isinstance(sm_pos, torch.Tensor):
    out["B_blocks_positions_given"] = (sm_pos // RING).tolist()
else:
    out["B_slot_mapping_positions_given"] = sm_pos

# ---------------------------------------------------------------- C. seed kernel
from vllm.models.glm5next.nvidia.ops.kpool_compress import kpool_seed_tail_cache  # noqa: E402

PAGE = 143616  # bytes per idx page (= padded tail page)
NB = 600
raw = torch.zeros(NB * PAGE, dtype=torch.uint8)
# as_strided bf16 tail view, exactly like attn_utils._reshape_attention_kv_cache with page_size_padded
tail = torch.as_strided(raw.view(torch.bfloat16), size=(NB, 2, RING, 128),
                        stride=(PAGE // 2, RING * 128, 128, 1))
n_pref = 24
key = (torch.arange(n_pref, dtype=torch.float32)[:, None] + 1).repeat(1, 128).to(torch.bfloat16)
gate = -key
ts = gen[:n_pref].clone()
seed_err = None
try:
    kpool_seed_tail_cache(tail, key, gate, ts, KPOOL, 128)
except Exception as e:  # noqa: BLE001
    seed_err = repr(e)
out["C_seed_error"] = seed_err
writes = []
for blk in sorted({int(x) // RING for x in ts.tolist()} | {0, 77, 483}):
    for half in (0, 1):
        for slot in range(RING):
            v = float(tail[blk, half, slot, 0])
            if v != 0.0:
                writes.append(dict(block=blk, half="K" if half == 0 else "score", slot=slot,
                                   position=int(abs(v)) - 1,
                                   byte_offset_in_page=half * RING * 128 * 2 + slot * 256))
out["C_tail_slot_mapping_prefill"] = ts.tolist()
out["C_writes"] = writes
# byte map of block 483 page: which 256-B units are nonzero
pg = raw[483 * PAGE: 483 * PAGE + 4096].view(16, 256)
out["C_block483_nonzero_256B_units"] = [i for i in range(16) if bool(pg[i].any())]

import json  # noqa: E402
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpu_mechanism.json"), "w"), indent=1)
for k, v in out.items():
    if k == "C_writes":
        print(k, [(w["block"], w["half"], w["slot"], w["position"]) for w in v])
    elif k == "C_tail_slot_mapping_prefill":
        print(k, v)
    else:
        print(k, v)
ok = (not out["A_positions_passed"]) and out["A_tail_build_gated_on_positions"] and \
    out["C_block483_nonzero_256B_units"] == [4, 5, 6, 7, 12, 13, 14, 15]
print("MECHANISM_REPRODUCED" if ok else "MECHANISM_NOT_REPRODUCED")
sys.exit(0)
