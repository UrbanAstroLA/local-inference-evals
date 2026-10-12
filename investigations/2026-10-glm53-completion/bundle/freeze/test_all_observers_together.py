#!/usr/bin/env python3
"""CPU: all four observers enabled in ONE process (as in gate session s1) install without conflict.
Run inside the check image with GLM53_COL511CHECK=device GLM53_TAILGUARD=device GLM53_VERIFY13=1 GLM53_D4_INSPECT=1."""
import os, sys, types
import torch
fails = []
def check(n, ok, info=""):
    print(("PASS " if ok else "FAIL ") + n + (f"  [{info}]" if info else ""), flush=True)
    if not ok: fails.append(n)
import vllm.models.glm5next.nvidia.attention  # noqa: F401  (imports the kpool module the way the model does)
import vllm.model_executor.layers.sparse_attn_indexer_kpool as k
import vllm.col511check as c, vllm.kpool_tailguard as t
from vllm.model_executor.layers.fused_moe.router import gate_linear as gl
import vllm.v1.worker.gpu.block_table as bt
check("col511 checker and tail guard enabled in the kpool module", k._C511_ON and k._TAILGUARD_ON)
check("both BlockTables wrappers installed (chained)", getattr(bt.BlockTables, "_col511check", False) and getattr(bt.BlockTables, "_tailguard", False))
import vllm.distributed.parallel_state as ps
ps._TP = types.SimpleNamespace(rank_in_group=0, world_size=1)
g = gl.GateLinear(64, 8, out_dtype=torch.float32, params_dtype=torch.bfloat16, prefix="language_model.model.layers.3.mlp.gate")
from vllm.model_executor.layers.quantization import exl3 as e
import vllm.verify13_capture as v
check("verify13 installed on GateLinear and its deferred wrappers", getattr(gl.GateLinear, "_v13", False) and "deferred" in v.S.installed, sorted(v.S.installed))
check("d4limit (inspect) wrapped both EXL3 planners after verify13's apply wrapper",
      hasattr(e.Exl3MoEMethod._rank_sliced_runtime, "_d4limit_orig") and hasattr(e.Exl3MoEMethod._mixed_rank_sliced_runtime, "_d4limit_orig")
      and getattr(e.Exl3MoEMethod.apply, "__qualname__", "") != "Exl3MoEMethod.apply")
x = torch.randn(5, 64, dtype=torch.bfloat16)
out = g(x)
ref = gl.GateLinear.__mro__[1].forward(g, x)[0].to(torch.float32) if False else None
check("hooked GateLinear forward runs with every observer enabled (CPU)", isinstance(out, tuple) and out[0].shape == (5, 8))
print(f"done: {len(fails)} failed {fails}")
sys.exit(1 if fails else 0)
