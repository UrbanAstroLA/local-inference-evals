#!/usr/bin/env python3
"""CPU dry run of precheck/precheck_graph.py (the script itself, not the GPU behaviour): CUDA graphs are replaced by a
fake whose replay re-runs the hooks, `_capturing()` is True inside the fake capture, device = CPU.  C1-C3 must pass
(the bookkeeping that compares stores with the step inputs is right); N1 is a GPU-only negative control and is
expected to FAIL here (a CPU copy inside a fake capture does not fail)."""
import contextlib
import json
import os
import sys

os.environ["P4_PRECHECK_DEV"] = "cpu"
os.environ.setdefault("GLM53_P4", "on")
os.environ.setdefault("GLM53_P4_DIR", "/tmp/p4pre-cpu")
import torch  # noqa: E402
from vllm import p4_hooks as p4  # noqa: E402

STATE = {"capturing": False}
sys.argv = ["precheck_graph.py", "/tmp/p4pre-cpu-result.json"]
sys.path.insert(0, "/p4pc")
import precheck_graph as pg  # noqa: E402  (main() is not run on import)


class FakeGraph:
    def replay(self):
        pg.forward(pg.G["mods"], pg.G["st"])


@contextlib.contextmanager
def fake_graph(g):
    STATE["capturing"] = True
    try:
        yield
    finally:
        STATE["capturing"] = False


torch.cuda.CUDAGraph = FakeGraph
torch.cuda.graph = fake_graph
torch.cuda.synchronize = lambda *a, **k: None
p4._capturing = lambda: STATE["capturing"]
pg.main()
names = {r["test"][:2]: r["ok"] for r in pg.RESULTS}
ok = bool(names.get("C1") and names.get("C2") and names.get("C3"))
print(("PASS" if ok else "FAIL") + " precheck_graph.py dry run on CPU: C1, C2, C3 pass (N1 is GPU-only, here: %s)" % names.get("N1"))
sys.exit(0 if ok else 1)
