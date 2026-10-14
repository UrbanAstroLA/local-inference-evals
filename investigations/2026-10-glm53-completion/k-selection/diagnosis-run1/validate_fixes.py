#!/usr/bin/env python3
"""Diagnosis run 1: CPU checks of the ERROR and of the proposed post-inspection fixes (no GPU).
Runs inside the pinned 0.7.0 image (--network none, no --gpus), once against the FROZEN kselect and
once against the PATCHED copy (post-inspection-fixes.diff applied):

  python3 validate_fixes.py <kselect_dir> <plan.json> <bank_dir> frozen|patched

E1  run_k.run_batch on v070/tiled48/b512 batch 0 (family real_L3). Frozen: expect the run's exact
    RuntimeError at run_k.py:266 BEFORE any driver call. Patched (with the regenerated plan): the
    reference phase completes and the first driver call is reached (a sentinel driver stops there).
E2  run_k.run_cells over three cells where the middle cell's run_batch raises. Frozen: expect
    [PASS, ERROR, NOT-RUN] (the run-1 abort). Patched: expect [PASS, ERROR, PASS].
E3  (patched) persp_record on synthetic perspective pairs: order-only, column-511 swap (non-tie) and
    a set difference among exact ties; verdict counters must be untouched.
"""
import json
import os
import sys
import tempfile
import time

KD, PLAN, BANK, MODE = sys.argv[1:5]
sys.path.insert(0, KD)
import torch  # noqa: E402

import kinputs  # noqa: E402
import kref  # noqa: E402
import run_k  # noqa: E402

dev = torch.device("cpu")
plan = json.load(open(PLAN))
eng = plan["meta"]["engines"]["v070"]
out = {}


class Sentinel(Exception):
    pass


class StopDriver:
    def topk(self, *a, **k):
        raise Sentinel("driver reached")


# ---- E1
cell = next(c for c in plan["cells"] if c["id"] == "v070/tiled48/b512")
banks = {"real_L3": kinputs.Bank(os.path.join(BANK, "bank_L3.pt"), dev)}
agg = run_k.Agg()
timing = dict(build=0.0, ref=0.0, kernel=0.0, check=0.0)
try:
    run_k.run_batch(StopDriver(), eng, cell, "real_L3", False, 0, cell["batches"][0], banks, dev, agg,
                    open(os.devnull, "w"), [], timing)
    e1 = "no exception"
except Sentinel:
    e1 = "reference phase completed; first driver call reached"
except RuntimeError as e:
    import traceback
    tb = traceback.extract_tb(e.__traceback__)[-1]
    e1 = f"RuntimeError at {os.path.basename(tb.filename)}:{tb.lineno}: {str(e)[:90]}"
out["E1"] = dict(batch0=cell["batches"][0], result=e1, poison_rows=agg.desc["poison_rows"])

# ---- E2
calls = []


def fake_run_batch(drv, eng_, c, fam, exact, bi, batch, banks_, dev_, agg_, res_f, fails, timing_):
    calls.append(c["id"])
    if c["id"].endswith("/bad"):
        raise RuntimeError("synthetic harness fault")
    agg_.bump("rows", fam)


cells = [dict(id=f"cpu/v070/tiled48/{n}", kind="cpu", route="fused", rows=48, batches=[[1]],
              expect=dict(route="paged_tiled", ctas_per_group=1, fold=None)) for n in ("ok1", "bad", "ok2")]
saved = run_k.run_batch
run_k.run_batch = fake_run_batch
try:
    with tempfile.TemporaryDirectory() as td:
        summ = run_k.run_cells(None, dict(meta=dict(FAMILIES=["edge"], EXACT_FAMILIES=["edge"])), eng, "v070", cells,
                               {}, dev, td, time.time(), 1e9, 0, lambda m: None, dict(test=True))
        out["E2"] = dict(verdicts=[c["verdict"] for c in summ["cells"]], files=sorted(os.listdir(td)),
                         run_error=bool(summ.get("error")))
finally:
    run_k.run_batch = saved

# ---- E3
if MODE == "patched":
    P = 2000
    s = torch.arange(P, dtype=torch.float64)          # distinct scores: top-512 = ids 1488..1999
    s[1480:1500] = 1490.0                               # 20 exact ties straddling rank 512 (t = 1490)
    bb = torch.zeros(P, dtype=torch.float64)
    top = torch.arange(P - 1, P - 513, -1)              # 1999 .. 1488 (descending)
    rec = dict(cell="synthetic", fam="edge")

    def run(a, b, fam="edge"):
        agg = run_k.Agg()
        fails = []
        A = a[None, :].to(torch.int32)
        B = b[None, :].to(torch.int32)
        Z = torch.zeros_like(A, dtype=torch.float32)
        run_k.persp_record(agg, fam, rec, [(A, Z), (B, Z)], 1, lambda ri: (s, bb), fails)
        d = {k: v for k, v in agg.desc.items() if k.startswith("persp") and k != "persp_by_family"}
        return d, sum(agg.n.values()), fails[0]["info"] if fails else None
    base = top.clone()
    o = base.clone()
    o[[0, 1]] = o[[1, 0]]                               # order only, column 511 untouched
    c = base.clone()
    c[[0, 511]] = c[[511, 0]]                           # column 511 now holds the top pool
    tie_ids = torch.arange(1480, 1500)
    in_a = tie_ids[torch.isin(tie_ids, base)]
    out_a = tie_ids[~torch.isin(tie_ids, base)]
    t = base.clone()
    pos = (t == in_a[0]).nonzero().item()
    t[pos] = out_a[0]                                   # swap one tied pool for another tied pool
    out["E3"] = {
        "order_only": run(base, o),
        "col511_swap_nontie": run(base, c),
        "set_diff_tie": run(base, t),
    }

print(json.dumps(dict(mode=MODE, kselect=KD, **out), indent=1, default=str))
