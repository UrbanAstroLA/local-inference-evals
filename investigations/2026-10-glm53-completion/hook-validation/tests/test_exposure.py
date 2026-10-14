#!/usr/bin/env python3
"""Window-exposure instrumentation (PLAN.md amendment 21 section 7), CPU.

  X1  exposure_counts == a brute-force count over random selections, prompt lengths, positions and pool classes:
      selected pools, selected pools written in the decode window, of which corrupt / race; window pools that
      exist at the row's position, of which corrupt / race.
  X2  on_selection glue (the call in the index-K decode branch): rows of a forced request land at
      (request, output index, layer slot) with the step id; unforced, padding and disarmed rows write nothing.
  X3  graph-safety lint of the in-graph hook path (on_selection, exposure_counts): no host synchronisation
      (.item / .tolist / .cpu / .numpy / nonzero / bool() / Python branching on tensor values).
"""
import ast
import inspect
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("FORCEWIN", "on")
os.environ.setdefault("FORCEWIN_WMAX", "512")
os.environ.setdefault("FORCEWIN_DIR", "/tmp/fwtest")

from fwtest_common import check, summary  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from vllm import forcewin_hooks as fw  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402

CPU = torch.device("cpu")


def brute(pids, pos, P, cls, pool0):
    out = []
    for r in range(len(pids)):
        sel = [p for p in pids[r] if p >= 0]
        win = [p for p in sel if 4 * p + 3 >= P[r]]
        c = lambda p: cls[r][min(max(p - pool0[r], 0), len(cls[r]) - 1)]  # noqa: E731
        ncomp = (pos[r] + 1) // 4
        wpools = [i for i in range(P[r] // 4, ncomp)]
        out.append([len(sel), len(win), sum(1 for p in win if c(p) == fw.PC_CORRUPT),
                    sum(1 for p in win if c(p) == fw.PC_RACE), len(wpools),
                    sum(1 for i in wpools if c(i) == fw.PC_CORRUPT), sum(1 for i in wpools if c(i) == fw.PC_RACE)])
    return out


def t_counts():
    g = torch.Generator().manual_seed(5)
    ok = True
    for trial in range(40):
        rows, S, NPW = 12, 64, 200
        P = torch.randint(100, 300, (rows,), generator=g)
        pos = P + torch.randint(0, 600, (rows,), generator=g)
        pids = torch.full((rows, S), -1, dtype=torch.int64)
        for r in range(rows):
            n = int(torch.randint(0, S + 1, (1,), generator=g))
            ncomp = int(pos[r] + 1) // 4
            if ncomp > 0 and n > 0:
                pids[r, :n] = torch.randint(0, ncomp, (n,), generator=g)
        cls = torch.randint(0, 5, (rows, NPW), generator=g).to(torch.int8)
        pool0 = torch.div(P, 4, rounding_mode="floor")
        got = fw.exposure_counts(pids, pos, P, cls, pool0).tolist()
        exp = brute(pids.tolist(), pos.tolist(), P.tolist(), cls.tolist(), pool0.tolist())
        ok &= got == exp
    check("X1 exposure_counts == brute force (40 random batches x 12 rows)", ok)


def t_glue():
    fw.S.__init__()
    runner = types.SimpleNamespace(device=CPU, num_speculative_steps=3, max_num_reqs=4,
                                   model_config=types.SimpleNamespace(get_vocab_size=lambda: 154880))
    P = 101
    fw.add_request(runner, "a", 2, P, types.SimpleNamespace(sampling_params=SamplingParams(
        max_tokens=300, extra_args={"forcewin_hooks": [7] * 300, "local_fw_tag": "a",
                                    "local_fw_desig": [5, 41, 200], "local_fw_selrec": [41]})))
    fw.add_request(runner, "b", 0, 77, types.SimpleNamespace(sampling_params=SamplingParams(max_tokens=10)))
    ib = types.SimpleNamespace(num_reqs=2, idx_mapping=torch.tensor([2, 0], dtype=torch.int32),
                               idx_mapping_np=np.array([2, 0]), is_prefilling_np=np.zeros(2, dtype=bool),
                               num_tokens=8, num_draft_tokens=6)
    fw.on_batch(runner, ib)
    # window pools of request a: pool index i >= P // 4 = 25; mark some corrupt / race
    fw.S.pcls[2, 3] = fw.PC_CORRUPT      # pool 28
    fw.S.pcls[2, 5] = fw.PC_RACE         # pool 30
    next_n = 4
    pos_a = [P + 40 + j for j in range(4)]
    pos_b = [77 + 10 + j for j in range(4)]
    dec_seq = torch.tensor([p + 1 for p in pos_a + pos_b] + [0] * 4, dtype=torch.int32)   # + one padding request
    S_ = 16
    pids = torch.full((12, S_), -1, dtype=torch.int64)
    pids[:4, :6] = torch.tensor([3, 26, 28, 30, 31, 10])
    pids[4:8, :3] = torch.tensor([1, 2, 3])
    fw.on_selection("model.layers.3.self_attn.indexer.k_cache", pids, dec_seq, 3, next_n)
    xb = fw.S.xbuf
    rec = xb[2, 1, 0].tolist()             # row j=0 of a predicts output index pos+1-P = 41 = designated #1
    exp = [fw.S.step] + brute([pids[0].tolist()], [pos_a[0]], [P],
                              [fw.S.pcls[2].tolist()], [P // 4])[0]
    check("X2a forced row lands at (request, output index, layer slot) with the step id", rec == exp,
          "got %s exp %s" % (rec, exp))
    others = int((xb[0] >= 0).sum()) + int((xb[1] >= 0).sum()) + int((xb[3] >= 0).sum())
    nondes = int((xb[2, 0] >= 0).sum()) + int((xb[2, 2] >= 0).sum())
    check("X2b unforced, padding and non-designated rows write nothing", others == 0 and nondes == 0,
          "stray cells %d, non-designated %d" % (others, nondes))
    selrow = fw.S.selbuf[2, 0, 0, : 1 + 16].tolist()
    check("X2d full final selection recorded at the selection-record position (step id + pool ids)",
          selrow == [fw.S.step] + pids[0].tolist(), str(selrow[:8]))
    fw.S.xarmed.fill_(0)
    before = xb[2].clone()
    pids2 = pids.clone()
    pids2[:4, :6] = torch.tensor([25, 26, 27, 28, 29, 30])
    fw.on_selection("model.layers.3.self_attn.indexer.k_cache", pids2, dec_seq, 3, next_n)
    check("X2c disarmed (outside a real forward: capture, dummy runs) writes nothing",
          torch.equal(before, fw.S.xbuf[2]))


BANNED = {"item", "tolist", "cpu", "numpy", "nonzero", "synchronize"}


def lint(fn):
    src = inspect.getsource(fn)
    tree = ast.parse(src.lstrip() if src.startswith(" ") else src)
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr in BANNED:
            bad.append(n.attr)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "bool":
            bad.append("bool()")
    return bad


def t_lint():
    res = {f.__name__: lint(f) for f in (fw.on_selection, fw.exposure_counts, fw.d2_mapping, fw.forced_override,
                                         fw.row_metrics, fw.post_probs, fw.draft_rows, fw.row_plan)}
    check("X3 no host-sync constructs in the in-graph / per-step device functions", all(not v for v in res.values()),
          str({k: v for k, v in res.items() if v}))


if __name__ == "__main__":
    t_counts()
    t_glue()
    t_lint()
    ok = summary("test_exposure")
    sys.exit(0 if ok else 1)
