#!/usr/bin/env python3
"""CPU dry run of precheck/precheck_s0.py (the script's bookkeeping, not the GPU behaviour): CUDA graphs are replaced
by a fake whose replay re-runs the hooks, `_capturing()` is True inside the fake capture, device = CPU.  C1-C4 must
pass; N1 is a GPU-only negative control and is expected to FAIL here (a CPU copy inside a fake capture cannot fail)."""
import contextlib
import os
import sys

os.environ["P4_PRECHECK_DEV"] = "cpu"
os.environ.setdefault("GLM53_P4", "on")
os.environ.setdefault("GLM53_P4_S0", "on")
os.environ.setdefault("GLM53_P4_DIR", "/tmp/p4pre0-cpu")
import torch  # noqa: E402
from vllm import p4_hooks as p4  # noqa: E402

STATE = {"capturing": False}
sys.argv = ["precheck_s0.py", "/tmp/p4pre0-cpu-result.json"]
sys.path.insert(0, "/p4pc")
import precheck_s0 as ps  # noqa: E402


class FakeGraph:
    def replay(self):
        ps.forward(ps.G["mods"], ps.G["layers"], ps.G["st"])


@contextlib.contextmanager
def fake_graph(g):
    STATE["capturing"] = True
    try:
        yield
    finally:
        STATE["capturing"] = False


torch.cuda.CUDAGraph = FakeGraph
torch.cuda.graph = fake_graph
p4._capturing = lambda: STATE["capturing"]
ps.main()
names = {r["test"][:2]: r["ok"] for r in ps.RESULTS}
ok = all(names.get(k) for k in ("C1", "C2", "C3", "C4"))
print(("PASS" if ok else "FAIL") + " precheck_s0.py dry run on CPU: C1-C4 pass (N1 is GPU-only, here: %s)" % names.get("N1"))
sys.exit(0 if ok else 1)
