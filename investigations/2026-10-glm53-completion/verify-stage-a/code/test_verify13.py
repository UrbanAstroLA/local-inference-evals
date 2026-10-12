#!/usr/bin/env python3
"""CPU / meta unit tests for vllm/verify13_capture.py (amendment 13 capture image).

Run inside the image with CUDA hidden:
  CUDA_VISIBLE_DEVICES= GLM53_VERIFY13=1 python3 test_verify13.py
The first test re-runs the import in a subprocess WITHOUT GLM53_VERIFY13 to show inertness.
Prints PASS/FAIL per test and 'N passed, M failed'; exit code 1 on any failure.
"""
from __future__ import annotations

import ast
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

TMP = tempfile.mkdtemp(prefix="v13test_")
os.environ["GLM53_VERIFY13"] = "1"
os.environ["GLM53_VERIFY13_DIR"] = TMP
os.environ["GLM53_VERIFY13_FULL_ROWS"] = "16"
os.environ["GLM53_VERIFY13_WIN_ROWS"] = "4"
os.environ["GLM53_VERIFY13_STRIDE"] = "2"
os.environ["GLM53_VERIFY13_EVERY"] = "1"
os.environ["GLM53_VERIFY13_SLABS"] = "8"

import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from vllm.model_executor.layers.fused_moe.router import gate_linear as gl  # noqa: E402
import vllm.verify13_capture as v13  # noqa: E402

RESULTS = []

# Outside a server there is no TP group: GateLinear (ReplicatedLinear) only asks for rank/size.
import types  # noqa: E402
import vllm.distributed.parallel_state as _ps  # noqa: E402
_ps._TP = types.SimpleNamespace(rank_in_group=0, world_size=1)


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"PASS {name}", flush=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        RESULTS.append((name, False, repr(exc)))
        print(f"FAIL {name}: {exc!r}", flush=True)
        traceback.print_exc()


def make_gate(layer=3, h=64, e=16):
    g = gl.GateLinear(h, e, out_dtype=torch.float32, params_dtype=torch.bfloat16,
                      prefix=f"language_model.model.layers.{layer}.mlp.gate")
    torch.manual_seed(layer)
    with torch.no_grad():
        g.weight.normal_()
    g.e_score_correction_bias = torch.nn.Parameter(torch.randn(e, dtype=torch.float32))
    return g


class FakeExperts:
    def __init__(self, layer):
        self.layer_name = f"language_model.model.layers.{layer}.mlp.experts"
        self.swiglu_limit = 10.0


class FakeMethod:
    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        return (x.float() * topk_weights.sum(-1, keepdim=True)).to(x.dtype)


v13.wrap_apply(FakeMethod)


class OpLog(TorchDispatchMode):
    """Logs every aten op and the storages written by mutating ops."""

    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        written = []
        for i, a in enumerate(func._schema.arguments):
            if a.alias_info is not None and a.alias_info.is_write:
                t = args[i] if i < len(args) else kwargs.get(a.name)
                if isinstance(t, torch.Tensor):
                    written.append(t.untyped_storage().data_ptr())
        self.ops.append((str(func), tuple(written)))
        return out


def arm():
    open(os.path.join(TMP, "ARM"), "w").close()
    t0 = time.time()
    while not v13.S.armed and time.time() - t0 < 10:
        time.sleep(0.05)
    assert v13.S.armed, "writer did not arm"


def eager(fn):
    """Run fn with the forward-context mode forced to NONE (eager serving step)."""
    old = v13._runtime_mode
    v13._runtime_mode = lambda: "NONE"
    try:
        return fn()
    finally:
        v13._runtime_mode = old


def in_pool(ptr):
    base = v13.S.pool.untyped_storage().data_ptr()
    return base <= ptr < base + v13.S.pool.untyped_storage().nbytes()


# ------------------------------------------------------------------ tests
def t_inert_default():
    env = {k: v for k, v in os.environ.items() if not k.startswith("GLM53_VERIFY13")}
    code = ("import sys; from vllm.model_executor.layers.fused_moe.router import gate_linear as g; "
            "assert not getattr(g.GateLinear, '_v13', False); "
            "assert 'vllm.verify13_capture' not in sys.modules; "
            "import vllm.v1.worker.workspace as w; assert not getattr(w.WorkspaceManager, '_v13', False); "
            "print('inert')")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "inert" in r.stdout, r.stderr[-2000:]


def t_install():
    assert gl.GateLinear._v13
    make_gate(3)
    need = {"gate_linear", "deferred", "apply:Exl3MoEMethod", "router:FusedMoERouter", "workspace",
            "runner_v2_capture_model", "worker_warmup_kernels", "runner_v2_prepare_inputs"}
    missing = need - v13.S.installed
    assert not missing, f"missing installs {missing}; errors {v13.S.errors}"


def t_capture_device_and_pool_target():
    """Smoke attempt 1 regression: a GateLinear built on the CPU (moved to the GPU later) must not
    make the CPU the capture device, and with CUDA present the pool is pinned or not allocated."""
    make_gate(3)                          # CPU construction -> __init__ wrapper -> _ensure_writer(cpu)
    assert v13.S.dev is None, v13.S.dev
    v13._ensure_writer(torch.device("cpu"))
    assert v13.S.dev is None
    assert v13.pool_target(None, True) is None
    assert v13.pool_target(torch.device("cpu"), True) is None
    assert v13.pool_target(torch.device("cuda", 0), True) == "pinned"
    assert v13.pool_target(None, False) == "pageable"
    assert v13.pool_target(torch.device("cpu"), False) == "pageable"
    src = open(v13.__file__).read()
    assert 'if (S.dev is None or S.dev.type != "cuda") and x.device.type == "cuda":' in src


def t_not_armed_no_record():
    g = make_gate(3)
    x = torch.randn(8, 64).to(torch.bfloat16)
    eager(lambda: g(x))
    assert not v13.S.pending and v13.S.c["sampled"] == 0


def t_arm():
    arm()
    assert v13.S.pool is not None and v13.S.pool.shape[0] == 8


def t_select_rule():
    seen = set()
    for f in range(1, 40):
        for layer in range(3, 45):
            ok, a, r = v13.select(f, layer, 1000)
            assert ok == ((layer + f) % 2 == 0)
            if ok:
                assert r == 4 and 0 <= a <= 1000 - 4
                seen.add(a)
            ok2, a2, r2 = v13.select(f, layer, 10)
            assert ok2 == ok and (not ok2 or (a2 == 0 and r2 == 10))
    assert len(seen) > 20  # windows rotate
    assert v13.select(5, 7, 1000) == v13.select(5, 7, 1000)  # same on every rank


def t_observe_only_gate_and_apply():
    g3 = make_gate(3)
    g5 = make_gate(5)
    m = FakeMethod()
    x = torch.randn(10, 64).to(torch.bfloat16)
    x_ref = x.clone()
    # reference: original forward, no hook
    with OpLog() as A:
        ref3 = gl.GateLinear._v13_orig_forward(g3, x)
    # forward counter: layer 3 increments fwd; choose fwd so that layer 3 is sampled
    v13.S.fwd = 0  # next layer-3 call -> fwd 1 -> (3 + 1) % 2 == 0 -> sampled
    import numpy as np
    v13.S.batch = types.SimpleNamespace(   # mixed batch: 2 decode rows (req 0, 1), 8 prefill rows (req 2)
        num_tokens_after_padding=10, num_tokens=10, num_reqs=3, req_ids=["cmpl-a", "cmpl-b", "cmpl-c"],
        query_start_loc_np=np.array([0, 1, 2, 10, 10]), is_prefilling_np=np.array([False, False, True]),
        prefill_len_np=np.array([4240, 8240, 16478]), num_computed_tokens_np=np.array([5000, 9000, 2048]))
    with OpLog() as B:
        out3 = eager(lambda: g3(x))
    assert torch.equal(out3[0], ref3[0]) and out3[0].dtype == torch.float32
    assert torch.equal(x, x_ref), "input modified"
    na = len(A.ops)
    assert [o[0] for o in B.ops[:na]] == [o[0] for o in A.ops], "engine op sequence changed"
    extra = B.ops[na:]
    assert extra, "hook did not copy"
    for op, written in extra:
        for p in written:
            assert in_pool(p), f"hook wrote outside its pool: {op}"
    assert 3 in v13.S.pending
    # apply for layer 3 (same x) and a non-sampled layer 5 gate/apply pair
    ids = torch.randint(0, 16, (10, 8), dtype=torch.int64)
    w = torch.rand(10, 8)
    w_ref, ids_ref = w.clone(), ids.clone()
    with OpLog() as C:
        o = m.apply(layer=FakeExperts(3), x=x, topk_weights=w, topk_ids=ids,
                    shared_experts=None, shared_experts_input=None)
    assert torch.equal(w, w_ref) and torch.equal(ids, ids_ref)
    for op, written in C.ops:
        for p in written:
            assert in_pool(p), f"apply path wrote outside the hook pool: {op}"
    assert 3 not in v13.S.pending
    eager(lambda: g5(x))  # fwd stays 1 -> (5 + 1) % 2 == 0 -> sampled too
    m.apply(layer=FakeExperts(5), x=x.clone(), topk_weights=w, topk_ids=ids,
            shared_experts=None, shared_experts_input=None)  # different x -> x_apply stored
    t_observe_only_gate_and_apply.expect = dict(
        x=x[0:10], logits=ref3[0][0:10], ids=ids, w=w, out=o, g5=g5(x)[0])


def t_records_on_disk():
    exp = t_observe_only_gate_and_apply.expect
    open(os.path.join(TMP, "DISARM"), "w").close()
    t0 = time.time()
    while not glob.glob(os.path.join(TMP, "FLUSHED_r*")) and time.time() - t0 < 20:
        time.sleep(0.1)
    assert glob.glob(os.path.join(TMP, "FLUSHED_r*")), "writer did not flush"
    files = sorted(glob.glob(os.path.join(TMP, "rec_r*", "*.pt")))
    assert len(files) == 2, files
    r3 = torch.load([f for f in files if f.endswith("_l03.pt")][0])
    assert r3["meta"]["same_x"] is True and r3["meta"]["complete"] is True
    assert torch.equal(r3["x"], exp["x"]) and torch.equal(r3["logits_shipped"], exp["logits"])
    assert "logits" not in r3 and r3["meta"]["labels"]["logits_shipped"].startswith("shipped: bf16-result GEMM")
    assert set(r3["meta"]["labels"]) >= {"x", "logits_shipped", "topk_ids", "topk_weights", "out"}
    rows = r3["meta"]["rows"]
    assert rows["batch_kind"] == "mixed" and rows["row_req"] == [0, 1] + [2] * 8, rows
    assert rows["reqs"][2]["prefill_len"] == 16478 and rows["reqs"][0]["is_prefilling"] is False
    assert "plan_branch" in r3["meta"]["d4"]
    assert torch.equal(r3["topk_ids"], exp["ids"]) and torch.equal(r3["topk_weights"], exp["w"])
    assert torch.equal(r3["out"], exp["out"])
    r5 = torch.load([f for f in files if f.endswith("_l05.pt")][0])
    assert r5["meta"]["same_x"] is False and "x_apply" in r5
    assert torch.equal(r5["logits_shipped"], exp["g5"])
    log = [json.loads(l) for f in glob.glob(os.path.join(TMP, "v13_p*.jsonl")) for l in open(f)]
    kinds = {d["kind"] for d in log}
    assert {"gate_info", "experts_info", "record", "armed", "disarmed"} <= kinds, kinds
    gi = [d for d in log if d["kind"] == "gate_info" and d["layer"] == 3][0]
    assert gi["tier_from_flags"].startswith("6"), gi  # CPU: no specialized tiers
    ei = [d for d in log if d["kind"] == "experts_info" and d["layer"] == 3][0]
    assert ei["swiglu_limit_on_layer"] == 10.0
    sha = [d for d in log if d["kind"] == "gate_weight_sha"]
    assert sha and all("weight_sha256" in d for d in sha)


def t_capture_skip():
    g = make_gate(3)
    x = torch.randn(4, 64).to(torch.bfloat16)
    before = dict(v13.S.c)
    old = v13._capturing
    v13._capturing = lambda: True
    try:
        v13.S.fwd = 1
        eager(lambda: g(x))
        FakeMethod().apply(layer=FakeExperts(3), x=x, topk_weights=torch.rand(4, 8),
                           topk_ids=torch.zeros(4, 8, dtype=torch.int64),
                           shared_experts=None, shared_experts_input=None)
    finally:
        v13._capturing = old
    assert v13.S.c["gate_capturing"] == before["gate_capturing"] + 1
    assert v13.S.c["apply_capturing"] == before["apply_capturing"] + 1
    assert v13.S.c["sampled"] == before["sampled"] and v13.S.fwd == 1


def t_mode_not_none_and_no_ctx():
    g = make_gate(3)
    x = torch.randn(4, 64).to(torch.bfloat16)
    before = dict(v13.S.c)
    old = v13._runtime_mode
    v13._runtime_mode = lambda: "FULL"
    try:
        g(x)
    finally:
        v13._runtime_mode = old
    g(x)  # real _runtime_mode(): no forward context in this test -> None
    assert v13.S.c["gate_mode_not_none"] == before["gate_mode_not_none"] + 1
    assert v13.S.c["gate_no_ctx"] == before["gate_no_ctx"] + 1
    assert v13.S.c["sampled"] == before["sampled"]


def t_meta_tensors():
    g = gl.GateLinear(64, 16, out_dtype=torch.float32, params_dtype=torch.bfloat16,
                      prefix="language_model.model.layers.3.mlp.gate").to("meta")
    x = torch.empty(4, 64, dtype=torch.bfloat16, device="meta")
    before = dict(v13.S.c)
    out = eager(lambda: g(x))
    ref = gl.GateLinear._v13_orig_forward(g, x)
    assert out[0].device.type == "meta" and out[0].shape == ref[0].shape and out[0].dtype == ref[0].dtype
    assert v13.S.c["gate_other_device"] == before["gate_other_device"] + 1
    assert v13.S.c["errors"] == before["errors"]


def t_workspace_observer():
    from vllm.v1.worker.workspace import WorkspaceManager
    assert getattr(WorkspaceManager, "_v13", False)
    mgr = WorkspaceManager(torch.device("cpu"))
    v13.S.captured_done = False
    n0 = v13.S.c["ws_resizes"]
    mgr.get_simultaneous(((64,), torch.uint8))
    mgr.get_simultaneous(((32,), torch.uint8))   # fits: no resize
    v13.S.captured_done = True
    mgr.get_simultaneous(((8192,), torch.uint8))  # grows after "capture"
    assert v13.S.c["ws_resizes"] == n0 + 2, v13.S.c
    log = [json.loads(l) for f in glob.glob(os.path.join(TMP, "v13_p*.jsonl")) for l in open(f)]
    rs = [d for d in log if d["kind"] == "ws_resize"][-2:]
    assert rs[0]["after_capture"] is False and rs[1]["after_capture"] is True
    assert rs[1]["before"][0][1] == 256 and rs[1]["after"][0][1] == 8192
    assert rs[1]["caller"].startswith("test_verify13.py"), rs[1]["caller"]
    assert rs[1]["phase"] == "serving" and rs[1]["classification"] == "resize, no captured-graph address invalidated"
    # a view bound during capture, then growth: the freed range holds the captured address
    mgr2 = WorkspaceManager(torch.device("cpu"))
    v13.S.captured_done = False
    v13.S.in_capture_model = True
    old_cap = v13._capturing
    v13._capturing = lambda: True
    try:
        mgr2.get_simultaneous(((128,), torch.uint8))       # first allocation inside "capture" + binding
        mgr2.get_simultaneous(((64,), torch.uint8))        # fits: binding only
    finally:
        v13._capturing = old_cap
        v13.S.in_capture_model = False
    v13.S.captured_done = True
    v13.S.in_warmup_kernels = True
    mgr2.get_simultaneous(((1 << 20,), torch.uint8))      # post-capture warm-up growth
    v13.S.in_warmup_kernels = False
    log = [json.loads(l) for f in glob.glob(os.path.join(TMP, "v13_p*.jsonl")) for l in open(f)]
    last = [d for d in log if d["kind"] == "ws_resize"][-1]
    assert last["phase"] == "post_capture_warmup", last
    assert last["classification"] == "captured-graph address invalidated" and last["n_invalidated"] >= 1, last
    assert v13.S.c["ws_invalidating_resizes"] >= 1
    first = [d for d in log if d["kind"] == "ws_resize"][-2]
    assert first["phase"] == "graph_capture" and first["classification"].startswith("resize, no")


def t_runner_and_router_wrappers():
    class R:
        def capture_model(self):
            return 7
    v13.wrap_runner(R)
    v13.S.captured_done = False
    assert R().capture_model() == 7 and v13.S.captured_done is True

    class Router:
        top_k, global_num_experts, num_expert_group, topk_group = 8, 288, 1, 1
        renormalize, scoring_func, routed_scaling_factor = True, "sigmoid", 2.5
        e_score_correction_bias = torch.zeros(288)

        def select_experts(self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
            return router_logits[:, :8], torch.zeros_like(router_logits[:, :8], dtype=torch.int64)
    v13.wrap_router(Router)
    rt = Router()
    a = rt.select_experts(hidden_states=torch.zeros(2, 4), router_logits=torch.ones(2, 288),
                          topk_indices_dtype=torch.int64, input_ids=None)
    rt.select_experts(torch.zeros(2, 4), torch.ones(2, 288))
    assert torch.equal(a[0], torch.ones(2, 8))
    log = [json.loads(l) for f in glob.glob(os.path.join(TMP, "v13_p*.jsonl")) for l in open(f)]
    ri = [d for d in log if d["kind"] == "router_info" and d["cls"] == "Router"]
    assert len(ri) == 1 and ri[0]["routed_scaling_factor"] == 2.5


HOT = {"on_gate", "on_apply", "_put", "_submit", "select", "_num_decode_tokens", "_runtime_mode",
       "_capturing", "_layer_from", "forward", "apply", "select_experts", "on_router", "_layer_info",
       "_apply_info"}
FORBIDDEN = {"item", "cpu", "tolist", "numpy", "synchronize", "nonzero", "masked_select",
             "is_pinned", "query", "wait_event", "wait_stream"}


def t_ast_no_host_sync():
    src = open(v13.__file__).read()
    tree = ast.parse(src)
    bad = []
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name in HOT:
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in FORBIDDEN:
                        bad.append(f"{fn.name}:{node.lineno}:{node.func.attr}")
                if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Compare):
                    bad.append(f"{fn.name}:{node.lineno}:boolean-mask indexing")
    assert not bad, bad


def t_no_engine_writes_in_module():
    """Every in-place tensor op in the module writes into a hook-owned tensor
    (dst = pool view or host buffer); checked by name."""
    src = open(v13.__file__).read()
    tree = ast.parse(src)
    inplace = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                node.func.attr.endswith("_") and not node.func.attr.startswith("_"):
            tgt = ast.unparse(node.func.value)
            inplace.append((node.lineno, tgt, node.func.attr))
    allowed_targets = {"dst", "host"}
    bad = [i for i in inplace if i[1] not in allowed_targets]
    assert not bad, bad


for name, fn in [("inert_by_default", t_inert_default), ("install", t_install),
                 ("capture_device_and_pool_target", t_capture_device_and_pool_target),
                 ("not_armed_no_record", t_not_armed_no_record), ("arm", t_arm),
                 ("select_rule", t_select_rule),
                 ("observe_only_gate_and_apply", t_observe_only_gate_and_apply),
                 ("records_on_disk", t_records_on_disk), ("capture_skip", t_capture_skip),
                 ("mode_not_none_and_no_ctx", t_mode_not_none_and_no_ctx),
                 ("meta_tensors", t_meta_tensors), ("workspace_observer", t_workspace_observer),
                 ("runner_and_router_wrappers", t_runner_and_router_wrappers),
                 ("ast_no_host_sync", t_ast_no_host_sync),
                 ("no_engine_writes_in_module", t_no_engine_writes_in_module)]:
    check(name, fn)

npass = sum(1 for r in RESULTS if r[1])
print(f"{npass} passed, {len(RESULTS) - npass} failed (tmp {TMP})")
sys.exit(0 if npass == len(RESULTS) else 1)
