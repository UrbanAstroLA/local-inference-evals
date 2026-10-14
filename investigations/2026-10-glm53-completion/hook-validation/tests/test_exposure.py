#!/usr/bin/env python3
"""Window-exposure instrumentation (PLAN.md amendment 21 section 7), CPU.

  X1  exposure_counts == a brute-force count over random selections, prompt lengths, positions and pool classes:
      selected pools, selected pools written in the decode window, of which corrupt / race; window pools that
      exist at the row's position, of which corrupt / race.
  X2  on_selection glue (the call in the index-K decode branch): rows of a forced request land at
      (request, output index, layer slot) with the step id; unforced, padding and disarmed rows write nothing.
  X3  graph-safety lint of the in-graph hook path (on_selection, exposure_counts): no host synchronisation
      (.item / .tolist / .cpu / .numpy / nonzero / bool() / Python branching on tensor values).

Attempt 2 (post-inspection PA2-3; attempt-1 file kept as test_exposure.py.a1; X1-X3 unchanged except that the X2
input-batch stand-in now carries query_start_loc_np, as the engine's InputBatch does):
  X2f flattened indexer layout (SM120 use_flattening=True, next_n=4 -> rows [B*4, 1], on_selection(..., batch_size
      = num_decode_tokens, next_n = 1)) with a FULL-graph padding request: every forced token's row lands at its own
      (request, output index) with the brute-force counts; nothing for unforced or padding rows; full selections at
      the selection-record positions; also with non-uniform decode lengths, and the native [B, 4] layout with a
      padding request still exact.
  X2fn negative control: the attempt-1 module (forcewin_hooks.py.a1) on the same flattened input fails X2f
      (records missing and misattributed, as DIAGNOSIS-a1 section 3 measured on GPU).
  X0  no unexpected hook errors in the suite.
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
                               num_tokens=8, num_draft_tokens=6, query_start_loc_np=np.array([0, 4, 8]))
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


# ---------------------------------------------------------------- attempt 2: flattened layout (PA2-3)
ERRS = []
_orig_err = fw._err


def _count_err(where, exc):
    ERRS.append((where, repr(exc)[:200]))
    _orig_err(where, exc)


fw._err = _count_err


def load_a1():
    import importlib.machinery
    import importlib.util
    path = os.path.join(HERE, "..", "hook", "forcewin_hooks.py.a1")
    loader = importlib.machinery.SourceFileLoader("forcewin_hooks_a1", path)
    spec = importlib.util.spec_from_loader("forcewin_hooks_a1", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


LAYER = "model.layers.3.self_attn.indexer.k_cache"


def flat_case(mod, lens, flat, pad_tokens=4):
    """3 requests in batch order a (req_idx 2, P 101, forced), b (req_idx 0, P 77, forced), c (req_idx 1, P 90,
    unforced) with `lens` decode tokens each, plus one FULL-graph padding request of pad_tokens tokens. Returns
    (expected {(name, o): row}, recorded {(name, o): record}, selections {(name, o): row values}, rows, pids,
    dec_seq, P)."""
    mod.S.__init__()
    runner = types.SimpleNamespace(device=CPU, num_speculative_steps=3, max_num_reqs=4, max_num_tokens=64,
                                   model_config=types.SimpleNamespace(get_vocab_size=lambda: 154880))
    P = {"a": 101, "b": 77, "c": 90}
    des = list(range(0, 80))
    for name, idx, sel in (("a", 2, [41, 42]), ("b", 0, [12])):
        mod.add_request(runner, name, idx, P[name], types.SimpleNamespace(sampling_params=SamplingParams(
            max_tokens=300, extra_args={"forcewin_hooks": [7] * 300, "local_fw_tag": name,
                                        "local_fw_desig": des, "local_fw_selrec": sel})))
    mod.add_request(runner, "c", 1, P["c"], types.SimpleNamespace(sampling_params=SamplingParams(max_tokens=10)))
    names, idxs = ["a", "b", "c"], [2, 0, 1]
    qsl = np.concatenate([[0], np.cumsum(lens)])
    ntok = int(qsl[-1])
    ib = types.SimpleNamespace(num_reqs=3, idx_mapping=torch.tensor(idxs, dtype=torch.int32),
                               idx_mapping_np=np.array(idxs), is_prefilling_np=np.zeros(3, dtype=bool),
                               num_tokens=ntok, num_draft_tokens=ntok - 3, query_start_loc_np=qsl)
    mod.on_batch(runner, ib)
    mod.S.pcls[2, 3] = mod.PC_CORRUPT        # request a: pool 28 corrupt, pool 30 race
    mod.S.pcls[2, 5] = mod.PC_RACE
    mod.S.pcls[0, 2] = mod.PC_CORRUPT        # request b: pool 21 corrupt
    start = {"a": P["a"] + 40, "b": P["b"] + 10, "c": P["c"] + 5}
    pos = []
    owner = []
    for name, n in zip(names, lens):
        for j in range(n):
            pos.append(start[name] + j)
            owner.append(name)
    g = torch.Generator().manual_seed(11)
    S_ = 16
    if flat:
        rows = ntok + pad_tokens
        dec_seq = torch.tensor([p + 1 for p in pos] + [1] * pad_tokens, dtype=torch.int32)   # padding: position 0
        row_owner = owner + [None] * pad_tokens
        row_pos = pos + [0] * pad_tokens
    else:                                    # native padded [B, 4] grid (+ one padding request row block)
        rows = 4 * 4
        dec_seq = torch.zeros(rows, dtype=torch.int32)
        row_owner, row_pos = [None] * rows, [0] * rows
        k = 0
        for b, (name, n) in enumerate(zip(names, lens)):
            for j in range(n):
                dec_seq[4 * b + j] = pos[k] + 1
                row_owner[4 * b + j], row_pos[4 * b + j] = name, pos[k]
                k += 1
    pids = torch.full((rows, S_), -1, dtype=torch.int64)
    for r in range(rows):
        ncomp = (row_pos[r] + 1) // 4
        n = int(torch.randint(1, S_ + 1, (1,), generator=g))
        if ncomp > 0:
            pids[r, :n] = torch.randint(0, ncomp, (n,), generator=g)
    if flat:
        mod.on_selection(LAYER, pids, dec_seq, rows, 1)
    else:
        mod.on_selection(LAYER, pids, dec_seq, 4, 4)
    rid = {"a": 2, "b": 0, "c": 1}
    expected = {}
    for r in range(rows):
        nm = row_owner[r]
        if nm in ("a", "b"):
            expected[(nm, row_pos[r] + 1 - P[nm])] = r
    recorded, sels = {}, {}
    for nm in ("a", "b"):
        i = rid[nm]
        for k, o in enumerate(des):
            if int(mod.S.xbuf[i, k, 0, 0]) >= 0:
                recorded[(nm, o)] = mod.S.xbuf[i, k, 0].tolist()
        sp = {"a": [41, 42], "b": [12]}[nm]
        for k, o in enumerate(sp):
            if int(mod.S.selbuf[i, k, 0, 0]) >= 0:
                sels[(nm, o)] = mod.S.selbuf[i, k, 0, : 1 + S_].tolist()
    stray = int((mod.S.xbuf[1] >= 0).sum()) + int((mod.S.xbuf[3] >= 0).sum())
    exp_rec = {}
    for (nm, o), r in expected.items():
        i = rid[nm]
        exp_rec[(nm, o)] = [mod.S.step] + brute([pids[r].tolist()], [row_pos[r]], [P[nm]],
                                                [mod.S.pcls[i].tolist()], [P[nm] // 4])[0]
    exp_sel = {(nm, o): [mod.S.step] + pids[r].tolist() for (nm, o), r in expected.items()
               if o in {"a": [41, 42], "b": [12]}[nm]}
    return dict(expected=expected, recorded=recorded, exp_rec=exp_rec, sels=sels, exp_sel=exp_sel, stray=stray)


def judge(res):
    exp, got = set(res["exp_rec"]), set(res["recorded"])
    correct = {k for k in exp & got if res["recorded"][k] == res["exp_rec"][k]}
    return dict(n_expected=len(exp), correct=len(correct), missing=sorted(exp - got),
                misattributed=sorted(got - exp), wrong_content=sorted((exp & got) - correct), stray=res["stray"],
                sel_ok=res["sels"] == res["exp_sel"])


def flat_ok(j):
    return (j["correct"] == j["n_expected"] and not j["missing"] and not j["misattributed"]
            and not j["wrong_content"] and j["stray"] == 0 and j["sel_ok"])


def t_flat():
    j = judge(flat_case(fw, [4, 4, 4], flat=True))
    check("X2f-a flattened [B*4,1] + padding request: every forced row at its own (request, output index) with "
          "the brute-force counts; selections at the record positions; nothing for unforced/padding rows",
          flat_ok(j) and j["n_expected"] == 8, str(j))
    j2 = judge(flat_case(fw, [4, 1, 4], flat=True, pad_tokens=7))
    check("X2f-b flattened, non-uniform decode lengths (4, 1, 4) + 7 padding tokens: exact",
          flat_ok(j2) and j2["n_expected"] == 5, str(j2))
    j3 = judge(flat_case(fw, [4, 4, 4], flat=False))
    check("X2f-c native [B,4] layout with a padding request: still exact (regression)",
          flat_ok(j3) and j3["n_expected"] == 8, str(j3))
    a1 = load_a1()
    a1._err = lambda where, exc: None
    ja = judge(flat_case(a1, [4, 4, 4], flat=True))
    check("X2fn negative control: the attempt-1 hook fails X2f on the same flattened input (missing and "
          "misattributed records, as measured on GPU)",
          not flat_ok(ja) and len(ja["missing"]) > 0 and len(ja["misattributed"]) > 0,
          "a1: correct %d of %d, missing %d, misattributed %s" % (ja["correct"], ja["n_expected"],
                                                                 len(ja["missing"]), ja["misattributed"]))
    ja3 = judge(flat_case(a1, [4, 4, 4], flat=False))
    check("X2fn-b the attempt-1 hook was exact in the native layout (why CPU test X2 passed in attempt 1)",
          flat_ok(ja3), str(ja3))


if __name__ == "__main__":
    t_counts()
    t_glue()
    t_lint()
    t_flat()
    check("X0 no hook errors during the suite", not ERRS, str(ERRS[:3]))
    ok = summary("test_exposure")
    sys.exit(0 if ok else 1)
