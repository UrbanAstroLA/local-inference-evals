#!/usr/bin/env python3
"""verify13 GPU smoke (ONE GPU, ~1 min, before the server session; NOT a server).

In the verify13 image with GLM53_VERIFY13=1 and a writable GLM53_VERIFY13_DIR:
  1. tier: the image's GateLinear(4096, 288, out_dtype=fp32), bf16 weight, on this GPU -> flags
     (expect tier 6 on SM120) and output values bf16-representable.
  2. host-sync check: an armed eager forward (M=64 full rows and M=300 window) and a wrapped
     apply run under torch.cuda.set_sync_debug_mode("error"); any synchronising call inside
     the hooks raises, is caught by the hook and counted -> require errors == 0 and 2 records.
  3. graph safety: CUDA-graph capture of the wrapped forward and apply -> the hooks skip
     (gate_capturing / apply_capturing counted, nothing sampled); replay on new data equals
     the unhooked eager forward bitwise.
  4. records on disk equal the source tensors bitwise.
Prints one JSON line with the results; exit 0 only if every check holds.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import types

assert os.environ.get("GLM53_VERIFY13") == "1", "run with GLM53_VERIFY13=1"
assert os.environ.get("GLM53_VERIFY13_STRIDE") == "1", "run with GLM53_VERIFY13_STRIDE=1"
DIR = os.environ.get("GLM53_VERIFY13_DIR", "/tmp/verify13-smoke")
os.makedirs(DIR, exist_ok=True)

import torch  # noqa: E402
import vllm.distributed.parallel_state as ps  # noqa: E402

ps._TP = types.SimpleNamespace(rank_in_group=0, world_size=1)
from vllm.model_executor.layers.fused_moe.router import gate_linear as gl  # noqa: E402
import vllm.verify13_capture as v13  # noqa: E402

res = dict(ok=True, checks={})


def chk(name, cond, **info):
    res["checks"][name] = dict(ok=bool(cond), **info)
    res["ok"] = res["ok"] and bool(cond)


dev = torch.device("cuda", 0)
torch.cuda.set_device(dev)
torch.manual_seed(0)
prev = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
gate = gl.GateLinear(4096, 288, out_dtype=torch.float32, prefix="language_model.model.layers.3.mlp.gate")
torch.set_default_dtype(prev)
gate = gate.to(dev)
with torch.no_grad():
    gate.weight.normal_(0, 0.02)
gate.e_score_correction_bias = torch.nn.Parameter(torch.zeros(288, device=dev), requires_grad=False)
flags = {k: getattr(gate, k, None) for k in ("allow_ll_bf16_gemm", "allow_specialized_router_gemm",
         "allow_dsv3_router_gemm", "allow_fp32_router_gemm", "allow_bf16x3_router_gemm",
         "allow_cublas_router_gemm")}
cap = torch.cuda.get_device_capability(dev)
x64 = torch.randn(64, 4096, device=dev).to(torch.bfloat16)
out_unhooked = gl.GateLinear._v13_orig_forward(gate, x64)[0]
chk("tier6_on_this_gpu", not any(flags.values()) and out_unhooked.dtype == torch.float32
    and torch.equal(out_unhooked, out_unhooked.to(torch.bfloat16).float()), capability=list(cap), flags=flags)


class FakeExperts:
    layer_name = "language_model.model.layers.3.mlp.experts"
    swiglu_limit = 10.0


class FakeMethod:
    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        return x * 2


v13.wrap_apply(FakeMethod)
v13._runtime_mode = lambda: "NONE"           # eager serving step
_ = gate(x64)                                # first call: layer info, writer thread, S.dev (not armed)
open(os.path.join(DIR, "ARM"), "w").close()
t0 = time.time()
while not v13.S.armed and time.time() - t0 < 30:
    time.sleep(0.05)
chk("armed", v13.S.armed and v13.S.pool_pinned, pool_pinned=v13.S.pool_pinned, dev=str(v13.S.dev))

ids = torch.randint(0, 288, (300, 8), device=dev, dtype=torch.int64)
w = torch.rand(300, 8, device=dev)
x300 = torch.randn(300, 4096, device=dev).to(torch.bfloat16)
e0 = v13.S.c["errors"]
srcs = {}
torch.cuda.synchronize()
torch.cuda.set_sync_debug_mode("error")
try:
    for x in (x64, x300):                       # run with GLM53_VERIFY13_STRIDE=1: every call sampled
        o = gate(x)
        out = FakeMethod().apply(layer=FakeExperts(), x=x, topk_weights=w[: x.shape[0]],
                                 topk_ids=ids[: x.shape[0]], shared_experts=None, shared_experts_input=None)
        ok, a, r = v13.select(v13.S.fwd, 3, x.shape[0])
        srcs[f"f{v13.S.fwd:07d}_l03.pt"] = dict(x=x[a:a + r], logits=o[0][a:a + r], ids=ids[a:a + r],
                                                w=w[a:a + r], out=out[a:a + r])
finally:
    torch.cuda.set_sync_debug_mode(0)
chk("no_host_sync_in_hooks", v13.S.c["errors"] == e0, errors=v13.S.errors[-5:])

# graph capture: hooks must skip; replay equals unhooked eager
xs = torch.randn(64, 4096, device=dev).to(torch.bfloat16)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2):
        gl.GateLinear._v13_orig_forward(gate, xs)
torch.cuda.current_stream().wait_stream(s)
before = dict(v13.S.c)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    yo = gate(xs)[0]
    yy = FakeMethod().apply(layer=FakeExperts(), x=xs, topk_weights=w[:64], topk_ids=ids[:64],
                            shared_experts=None, shared_experts_input=None)
xs.copy_(torch.randn(64, 4096, device=dev).to(torch.bfloat16))
g.replay()
torch.cuda.synchronize()
ref = gl.GateLinear._v13_orig_forward(gate, xs)[0]
chk("graph_capture_skips_hooks",
    v13.S.c["gate_capturing"] > before["gate_capturing"] and v13.S.c["apply_capturing"] > before["apply_capturing"]
    and v13.S.c["sampled"] == before["sampled"])
chk("graph_replay_equals_eager", torch.equal(yo, ref))

# D5 observer on a real capture: a workspace view bound inside torch.cuda.graph capture is recorded,
# and a later growth that frees that buffer is classified as invalidating a captured address.
from vllm.v1.worker.workspace import WorkspaceManager  # noqa: E402
wm = WorkspaceManager(dev)
wm.get_simultaneous(((1 << 20,), torch.uint8))            # allocate before capture
nb0 = v13.S.c["ws_capture_bindings"]
g2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g2):
    (buf,) = wm.get_simultaneous(((4096,), torch.uint8))  # fits: binding only, no resize in capture
    buf.add_(1)
v13.S.captured_done = True
inv0 = v13.S.c["ws_invalidating_resizes"]
wm.get_simultaneous(((8 << 20,), torch.uint8))            # growth after capture frees the bound buffer
chk("d5_capture_binding_and_invalidation", v13.S.c["ws_capture_bindings"] == nb0 + 1
    and v13.S.c["ws_invalidating_resizes"] == inv0 + 1)

open(os.path.join(DIR, "DISARM"), "w").close()
t0 = time.time()
while not glob.glob(os.path.join(DIR, "FLUSHED_r*")) and time.time() - t0 < 60:
    time.sleep(0.1)
files = {os.path.basename(f): f for f in glob.glob(os.path.join(DIR, "rec_r*", "*.pt"))}
good = len(files) == 2
for name, src in srcs.items():
    if name not in files:
        good = False
        continue
    r = torch.load(files[name])
    good = good and all(torch.equal(r[k].to(dev), src[k2]) for k, k2 in
                        (("x", "x"), ("logits_shipped", "logits"), ("topk_ids", "ids"), ("topk_weights", "w"),
                         ("out", "out")))
chk("records_bitwise", good, files=sorted(files))
res["counters"] = dict(v13.S.c)
print(json.dumps(res, default=str))
sys.exit(0 if res["ok"] else 1)
