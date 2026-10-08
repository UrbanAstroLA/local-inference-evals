#!/usr/bin/env python3
"""kpool-tail-index/v1 check (protocols/kpool-tail-index/v1.md): which tokens a DCP1 sparse-MLA decode step attends.

Runs the image's own Triton kernels, extracted from the installed vLLM sources by name, in the order of the DCP1
logical-id branch with an identity block table (no model, no vLLM import):
  - kpool_compress._expand_pools_and_append_tail_kernel (decode history + tail expansion);
  - b12x_mla_sparse.*_compact_dropped_selection(_kernel) (the DCP1 tail fix, tpurtell/glm-5.3-flash-ext3-2x-rtx#6;
    present only in a patched image, found by name suffix);
  - b12x_mla_sparse._mask_page_table_after_nsa_len(_kernel) (the stock positional mask).
Prints JSON: per case the attended set summary, and a hash of the final index row.

GPU (as published): docker run --rm --gpus all --network none -e TRITON_INTERPRET=0 -v "$PWD":/w:ro \
                        --entrypoint python3 IMAGE /w/kpool_tail_index_check.py
CPU (Triton interpreter): the same with -e TRITON_INTERPRET=1 and no --gpus.
"""

import ast
import hashlib
import json
import os
import random

os.environ.setdefault("TRITON_INTERPRET", "1")   # GPU mode: pass -e TRITON_INTERPRET=0 and --gpus
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"
KPOOL = f"{VLLM}/models/glm5next/nvidia/ops/kpool_compress.py"
MLA = f"{VLLM}/v1/attention/backends/mla/b12x_mla_sparse.py"


def load(path, names):
    """Copy the named top-level functions into a temp module and import it: the Triton
    interpreter re-reads kernel source through inspect, so it needs a real file."""
    import importlib.util
    import tempfile
    src = open(path).read()
    tree = ast.parse(src)
    segs, found = ["import torch", "import triton", "import triton.language as tl", ""], []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
            segs.append("\n".join(src.splitlines()[start:node.end_lineno]) + "\n")
            found.append(node.name)
    fd, mod_path = tempfile.mkstemp(suffix=".py", prefix="kpool_tail_")
    with os.fdopen(fd, "w") as fh:
        fh.write("\n\n".join(segs))
    spec = importlib.util.spec_from_file_location(os.path.basename(mod_path)[:-3], mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return vars(mod), found


kp, _ = load(KPOOL, {"_expand_pools_and_append_tail_kernel"})
FIX = sorted({n.name for n in ast.parse(open(MLA).read()).body if isinstance(n, ast.FunctionDef)
              and n.name.endswith(("_compact_dropped_selection", "_compact_dropped_selection_kernel"))})
ml, found = load(MLA, {"_mask_page_table_after_nsa_len_kernel", "_mask_page_table_after_nsa_len", *FIX})
FIX_FN = next((n for n in found if n.endswith("_compact_dropped_selection")), None)
patched = FIX_FN is not None

DEV = "cpu" if os.environ["TRITON_INTERPRET"] == "1" else "cuda"
POOL, NGROUPS, WIDTH = 4, 512, 2048          # index_kpool, select_k pools, topk buffer width
rng = random.Random(1234)


def pool_topk(L, layout):
    """[1, 512] pool ids as the decode pool top-k could return them."""
    pool_len = L // POOL
    if pool_len <= NGROUPS:
        ids = list(range(pool_len)) + [-1] * (NGROUPS - pool_len)
        if layout == "scattered":
            rng.shuffle(ids)
        else:
            rng.shuffle(ids[:pool_len])
    else:
        ids = rng.sample(range(pool_len), NGROUPS)
    return torch.tensor([ids], dtype=torch.int64, device=DEV)


def run(L, layout):
    pools = pool_topk(L, layout)[:, : NGROUPS - 1]                 # kpool:1009 slice
    seq = torch.tensor([L], dtype=torch.int32, device=DEV)
    topk = (NGROUPS - 1) * POOL
    out_cols = topk + POOL - 1
    out = torch.empty((1, out_cols), dtype=torch.int32, device=DEV)
    kp["_expand_pools_and_append_tail_kernel"][(1, triton.cdiv(out_cols, 128))](
        pools, seq, out, topk, out_cols, POOL_SIZE=POOL, BLOCK_COLS=128,
        pid_s0=pools.stride(0), out_s0=out.stride(0))
    buf = torch.full((1, WIDTH), -1, dtype=torch.int32, device=DEV)             # kpool:461 init
    buf[:, :out_cols] = out                                         # kpool:1019
    sel = buf.clone()                                               # identity logical->physical
    nsa = torch.tensor([min(L, WIDTH)], dtype=torch.int32, device=DEV)          # copy_ + clamp_max_
    if patched:
        ml[FIX_FN](sel, nsa)
    ml["_mask_page_table_after_nsa_len"](sel, nsa)
    row = sel[0, : int(nsa[0])].tolist()
    picked = sorted(x for x in row if x >= 0)
    return row, picked, int(nsa[0]), out[0].tolist()


lengths = [5, 6, 7, 8, 1001, 1002, 1003, 1004, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047,
           2048, 2049, 2050, 2051, 2052, 4097, 100003]
report = {"image_patched": patched, "device": DEV, "rows": {}, "checks": {}}
fail = []
for layout in ("packed", "scattered"):
    for L in lengths:
        row, picked, nsa, expanded = run(L, layout)
        selected = sorted(x for x in expanded if x >= 0)   # what the indexer chose
        tail = list(range((L // POOL) * POOL, L))
        dropped = sorted(set(selected) - set(picked))
        key = f"{layout}:{L}"
        report["rows"][key] = hashlib.sha256(json.dumps([row, nsa]).encode()).hexdigest()[:16]
        report["checks"][key] = {"nsa_len": nsa, "tail_cols_before_mask": expanded[2044:2047],
                                 "tail_cols_after_mask": (row + [-1] * 2048)[2044:2047], "selected": len(selected), "attended": len(picked),
                                 "tail": tail, "tail_attended": all(t in picked for t in tail),
                                 "dropped": dropped[:8], "n_dropped": len(dropped),
                                 "equals_dense_prefix": picked == list(range(L)) if L <= 2047 else None}
        if patched and dropped:
            fail.append(f"{key}: patched image still drops {dropped[:8]}")
        # "packed" = valid pools ahead of -1 padding (what the pool top-k is expected to emit).
        # "scattered" is a synthetic worst case: there the [:511] slice itself can cut a valid
        # pool before any masking (a separate issue, reported as equals_dense_prefix only).
        if L <= 2047 and patched and layout == "packed" and picked != list(range(L)):
            fail.append(f"{key}: patched selection != tokens 0..L-1")
report["fail"] = fail
print(json.dumps(report, indent=1))
