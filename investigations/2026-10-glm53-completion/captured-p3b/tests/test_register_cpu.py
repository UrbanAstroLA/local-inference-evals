#!/usr/bin/env python3
"""CPU test of p3b_capture_hooks.register() and install() against the image's real vLLM classes (no GPU, no
engine): a fake runner whose model tree has the served layer names, real B12xMLASparseImpl instances (attributes
set by hand, no __init__), a draft-named subtree that must be ignored, and a second runner with a duplicate
candidate that must be reported as an issue (the capture gate then fails before traffic).

usage: test_register_cpu.py MODULE_PATH OUT_DIR       (stock image, CUDA_VISIBLE_DEVICES=, --network none)
"""
import importlib.util
import json
import os
import shutil
import sys
from types import SimpleNamespace

MOD, OUT = sys.argv[1], sys.argv[2]
CAPDIR = os.path.join(OUT, "capdir-register")
shutil.rmtree(CAPDIR, ignore_errors=True)
os.environ.update(GLM53_P3B="capture", GLM53_P3B_DIR=CAPDIR)

import torch  # noqa: E402
import vllm.distributed as vdist  # noqa: E402
from b12x.attention import sparse_mla  # noqa: E402
from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseImpl  # noqa: E402

vdist.get_tensor_model_parallel_rank = lambda: 0
RES = []


def result(name, ok, **kw):
    RES.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + "  " + json.dumps(kw, default=str)[:700], flush=True)


def load():
    spec = importlib.util.spec_from_file_location("p3b_capture_hooks_r", MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def impl():
    o = object.__new__(B12xMLASparseImpl)
    caps = sparse_mla.Caps(device="cpu", num_q_heads=64, max_q_rows=64, max_width=2048, softmax_scale=0.0625 * 1.0000001,
                           dtype=torch.bfloat16, kv_dtype=torch.uint8, head_dim=512, v_head_dim=512,
                           model_type=int(sparse_mla.ModelType.GLM_NEXT), cache_record_bytes=528, mode="decode",
                           max_batch=64, max_chunks_per_row=32, page_size=64, head_major_output=True, return_lse=False,
                           lse_scale="natural")
    for k, v in dict(_input_num_heads=64, _kernel_num_heads=64, q_head_dim=512, kv_lora_rank=512, _kv_record_bytes=528,
                     topk_tokens=2048, block_size=64, _pad_heads=False, need_to_return_lse_for_decode=False,
                     _head_major_mla_output=True, _decode_max_rows=64, _num_splits_cap=32, spec_decode_max_q=8,
                     spec_extend_as_decode=True, _indexer_outputs_physical_slots=False, _glm_next_record=True,
                     dcp_world_size=1, scale=0.0625 * 1.0000001, _decode_plan=SimpleNamespace(caps=caps)).items():
        object.__setattr__(o, k, v)
    return o


def layer_tree(n=45, dup=False):
    layers = torch.nn.ModuleList()
    for i in range(n):
        lay = torch.nn.Module()
        lay.mlp = torch.nn.Module()
        lay.mlp.experts = torch.nn.Module()
        lay.mlp.experts.exl3_trellis_weights = object()
        lay.mlp.experts.hidden_size = 4096
        lay.self_attn = torch.nn.Module()
        lay.self_attn.attn = torch.nn.Module()
        lay.self_attn.attn.impl = impl()
        layers.append(lay)
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = layers
    root.draft = torch.nn.Module()            # a draft-named subtree with the same layer numbers: must be ignored
    root.draft.layers = torch.nn.ModuleList([layer_tree_one() for _ in range(4)])
    if dup:
        root.other = torch.nn.Module()        # a second target-like candidate for layer 3: reported, not guessed
        root.other.layers = torch.nn.ModuleList([layer_tree_one() for _ in range(4)])
    return root


def layer_tree_one():
    lay = torch.nn.Module()
    lay.mlp = torch.nn.Module()
    lay.mlp.experts = torch.nn.Module()
    lay.mlp.experts.exl3_trellis_weights = object()
    lay.self_attn = torch.nn.Module()
    lay.self_attn.attn = torch.nn.Module()
    lay.self_attn.attn.impl = impl()
    return lay


def runner(dup=False):
    return SimpleNamespace(device=torch.device("cpu"), model=layer_tree(dup=dup),
                           scheduler_config=SimpleNamespace(async_scheduling=True),
                           input_buffers=SimpleNamespace(positions=torch.zeros(4096, dtype=torch.int64)))


def init_records():
    out = []
    for f in os.listdir(os.path.join(CAPDIR, "log")):
        if f.startswith("cap_"):
            out += [json.loads(l) for l in open(os.path.join(CAPDIR, "log", f)) if '"init"' in l]
    return out


cap = load()
r = runner()
cap.register(r)
cap.S.stop.set()
init = init_records()[-1]
target_moe = {L: f"model.layers.{L}.mlp.experts" for L in (3, 23, 44)}
ok = (init["moe_modules"] == {str(k): v for k, v in target_moe.items()} or init["moe_modules"] == target_moe)
ok &= sorted(int(k) for k in init["att_modules"]) == [3, 23] and init["issues"] == [] and init["async_scheduling"] is True
ok &= all(id(r.model.model.layers[L].mlp.experts) in cap.S.moe_mod for L in (3, 23, 44))
ok &= not any(id(m.mlp.experts) in cap.S.moe_mod for m in r.model.draft.layers)
q = cap.S.att[3].t["q"]
ok &= tuple(q.shape) == (cap.RA + 1, cap.NA, 64, 512) and tuple(cap.S.att[3].t["rec"].shape) == (cap.RA + 1, cap.NA, 2048, 528)
ok &= tuple(cap.S.moe[44].t["x"].shape) == (cap.RM + 1, cap.MMAX, 4096)
dc = init["att_info"]["3"]["decode_caps"]
ok &= float.fromhex(dc["softmax_scale"]["float_hex"]) == 0.0625 * 1.0000001 and dc["model_type"] == int(sparse_mla.ModelType.GLM_NEXT)
result("G1 register(): target layers found by name (draft subtree ignored), rings sized from the impl, served Caps logged",
       ok, ring_mib=init["ring_mib_total"], att=init["att_modules"], issues=init["issues"])
result("G2 register(): ring memory within the 256 MiB budget of PROPOSAL.md (rank 0)", init["ring_mib_total"] <= 256,
       ring_mib=init["ring_mib_total"])
cap2 = load()
cap2.register(runner(dup=True))
cap2.S.stop.set()
iss = init_records()[-1]["issues"]
result("G3 register(): a second candidate for a captured layer is an issue (the start-up gate then refuses traffic)",
       any(i[0] == "moe_duplicate" for i in iss) and any(i[0] == "att_duplicate" for i in iss), issues=iss)
os.environ["GLM53_P3B"] = ""
cap3 = load()
result("G4 GLM53_P3B unset: hooks are no-ops even if called", not cap3.ENABLED and cap3._att_layer(object()) is None
       and cap3.hook_moe(object(), None, None, None, None) is None)
shutil.rmtree(CAPDIR, ignore_errors=True)
print(f"{sum(RES)}/{len(RES)} passed")
sys.exit(0 if all(RES) else 1)
