#!/usr/bin/env python3
"""CPU tests of d4limit inside the verification image (no GPU). Run: python3 test_d4limit.py"""
import ast, importlib, inspect, os, subprocess, sys, tempfile

fails = []
def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)
    if not ok:
        fails.append(name)

# 1. inert without the variable (fresh interpreter)
code = ("import os, sys; os.environ.pop('GLM53_D4_LIMIT_OVERRIDE', None); os.environ.pop('GLM53_D4_INSPECT', None); "
        "from vllm.model_executor.layers.fused_moe.router import gate_linear as g; "
        "from vllm.model_executor.layers.quantization import exl3 as e; "
        "assert not getattr(g.GateLinear, '_d4limit', False); "
        "assert 'vllm.d4limit' not in sys.modules; "
        "assert e.Exl3MoEMethod._rank_sliced_runtime.__qualname__ == 'Exl3MoEMethod._rank_sliced_runtime'; "
        "assert not hasattr(e.Exl3MoEMethod._rank_sliced_runtime, '_d4limit_orig'); print('inert')")
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
check("inert by default (no import, no wrapper)", r.returncode == 0 and "inert" in r.stdout, r.stderr.strip()[-300:])

# 2. the D4-patched planners read the limit inside exactly the two wrapped methods
from vllm.model_executor.layers.quantization import exl3 as e
src = inspect.getsource(e)
tree = ast.parse(src)
readers = []
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "Exl3MoEMethod":
        for f in node.body:
            if isinstance(f, ast.FunctionDef) and 'getattr(layer, "swiglu_limit", None)' in (ast.get_source_segment(src, f) or ""):
                readers.append(f.name)
check("limit is read only inside _rank_sliced_runtime and _mixed_rank_sliced_runtime", sorted(readers) == ["_mixed_rank_sliced_runtime", "_rank_sliced_runtime"], str(readers))

# 3. override semantics on a fake method class (values 1.0 and none), attribute restored afterwards
for raw, expect in (("1", 1.0), ("none", None), ("10", 10.0)):
    with tempfile.TemporaryDirectory() as td:
        os.environ["GLM53_D4_LIMIT_OVERRIDE"] = raw
        os.environ["GLM53_D4_LIMIT_DIR"] = td
        sys.modules.pop("vllm.d4limit", None)
        m = importlib.import_module("vllm.d4limit")
        seen = {}

        class FakeMethod:
            def _rank_sliced_runtime(self, layer, x, topk_ids):
                seen["proj"] = getattr(layer, "swiglu_limit", "absent")
                return "rt"
            def _mixed_rank_sliced_runtime(self, layer, x, topk_ids):
                seen["mixed"] = getattr(layer, "swiglu_limit", "absent")
                return "mrt"

        class L:  # stand-in RoutedExperts
            swiglu_limit = 10.0
        lay = L()
        m.install(FakeMethod)
        fm = FakeMethod()
        ok = fm._rank_sliced_runtime(lay, None, None) == "rt" and fm._mixed_rank_sliced_runtime(lay, None, None) == "mrt"
        lay2 = type("L2", (), {})()
        fm._rank_sliced_runtime(lay2, None, None)
        logs = open(os.path.join(td, os.listdir(td)[0])).read()
        check(f"override {raw!r}: planners see {expect!r}; attribute restored; absent attribute stays absent",
              ok and seen["proj"] == expect and seen["mixed"] == expect and lay.swiglu_limit == 10.0
              and not hasattr(lay2, "swiglu_limit") and '"installed"' in logs and '"override_applied"' in logs, str(seen))

# 3b. inspect mode: no override, plan walk finds the limit wherever it sits in the runtime's object graph
with tempfile.TemporaryDirectory() as td:
    os.environ.pop("GLM53_D4_LIMIT_OVERRIDE", None)
    os.environ["GLM53_D4_INSPECT"] = "1"
    os.environ["GLM53_D4_LIMIT_DIR"] = td
    sys.modules.pop("vllm.d4limit", None)
    m = importlib.import_module("vllm.d4limit")
    class K:  # stand-in compiled kernel
        __slots__ = ("has_swiglu_limit", "swiglu_limit", "w")
        def __init__(self):
            self.has_swiglu_limit, self.swiglu_limit, self.w = True, 10.0, None
    class Caps:
        def __init__(self):
            self.swiglu_limit = 10.0
    class Plan:
        def __init__(self):
            self.caps, self.launch_plan = Caps(), {"k": [K()]}
    class FakeM:
        def _rank_sliced_runtime(self, layer, x, topk_ids):
            seen2["l"] = getattr(layer, "swiglu_limit", "absent")
            return {"trellis_plan": Plan(), "prefill_plan": Plan(), "loop": None}
        def _mixed_rank_sliced_runtime(self, layer, x, topk_ids):
            return {}
    seen2 = {}
    m.install(FakeM)
    class L3:
        swiglu_limit = 10.0
    rt = FakeM()._rank_sliced_runtime(L3(), None, None)
    FakeM()._rank_sliced_runtime(L3(), None, None)
    lines = [l for l in open(os.path.join(td, os.listdir(td)[0])) if '"plan_limits"' in l]
    import json as _j
    rec = _j.loads(lines[0]) if lines else {}
    check("inspect mode: no override applied, plan walk logs caps and kernel limits (values 10.0 / True)",
          seen2["l"] == 10.0 and len(lines) == 2 and rec.get("n_hits") == 6 and set(rec.get("values", [])) == {"10.0", "True"},
          str(rec)[:300])
    os.environ.pop("GLM53_D4_INSPECT", None)

# 4. invalid values are refused at import
os.environ["GLM53_D4_LIMIT_OVERRIDE"] = "-3"
sys.modules.pop("vllm.d4limit", None)
try:
    importlib.import_module("vllm.d4limit")
    check("negative override refused", False)
except ValueError:
    check("negative override refused", True)

# 5. guarded block present in the installed gate_linear.py, and deferred install through GateLinear construction
os.environ["GLM53_D4_LIMIT_OVERRIDE"] = "1"
gl_src = open(sys.modules["vllm.model_executor.layers.quantization.exl3"].__file__.replace(
    "quantization/exl3.py", "fused_moe/router/gate_linear.py")).read()
check("gate_linear.py carries the guarded d4limit block", gl_src.count("_d4l.install_on_gate(GateLinear)") == 1)
code = ("import os; os.environ['GLM53_D4_LIMIT_OVERRIDE']='1'; os.environ['GLM53_D4_LIMIT_DIR']='/tmp/d4t'; "
        "from vllm.model_executor.layers.fused_moe.router import gate_linear as g; "
        "from vllm.model_executor.layers.quantization import exl3 as e; "
        "assert g.GateLinear._d4limit; "
        "assert not hasattr(e.Exl3MoEMethod._rank_sliced_runtime, '_d4limit_orig'); "
        "import types, torch, vllm.distributed.parallel_state as ps; ps._TP = types.SimpleNamespace(rank_in_group=0, world_size=1); "
        "g.GateLinear(64, 8, out_dtype=torch.float32, params_dtype=torch.bfloat16, prefix='language_model.model.layers.3.mlp.gate'); "
        "assert hasattr(e.Exl3MoEMethod._rank_sliced_runtime, '_d4limit_orig'); "
        "assert hasattr(e.Exl3MoEMethod._mixed_rank_sliced_runtime, '_d4limit_orig'); print('deferred ok')")
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
check("deferred install at first GateLinear construction wraps both planners", r.returncode == 0 and "deferred ok" in r.stdout, (r.stdout + r.stderr).strip()[-400:])

print(f"done: {len(fails)} failed {fails}")
sys.exit(1 if fails else 0)
