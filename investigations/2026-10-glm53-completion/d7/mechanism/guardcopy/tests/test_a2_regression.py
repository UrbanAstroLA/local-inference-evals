#!/usr/bin/env python3
"""Regression for the run-a2 index-K read finding (CPU, NO GPU).

With the indexer read tables produced by B0's real builder geometry (b0emu.IDX_FACTOR=4,
page ids 17*block+j), the a2 guard module (guard/mapguard_hooks.py.a2, sha 551faad6...) flags
every decode/prefill index-K read of a clean scenario as `rtable` with consumed == 17 x block
(the a2 signature), and the corrected guard flags none.
"""
import importlib.util
import json
import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = [sys.argv[0]] + sys.argv[1:]
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "a2-regression.json")
import test_mapguard as T  # noqa: E402

from importlib.machinery import SourceFileLoader  # noqa: E402
_path = os.path.join(HERE, "..", "guard", "mapguard_hooks.py.a2")
spec = importlib.util.spec_from_loader("mg_a2", SourceFileLoader("mg_a2", _path))
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
res = {}
for name, mod in (("a2_guard", old), ("fixed_guard", T.mg)):
    T.mg = mod
    T.VIOL = mod.VIOLATION_COUNTERS
    sc = T.scenario(max_steps=14)
    tot = T.summarize(sc)
    sig = []
    for st in sc["steps"]:
        for r in st["recs"]:
            d = dict(zip(mod.REC_COLS, r))
            if mod.SITES[d["site"]] in ("idx_r_dec", "idx_r_pf"):
                sig.append(d["consumed"])
    res[name] = dict(violations=tot, idx_read_consumed_examples=sig[:6],
                     consumed_multiple_of_17=all(c % 17 == 0 for c in sig) if sig else None)
ok = (res["a2_guard"]["violations"].get("idx_r_dec.rtable", 0) > 0
      and res["a2_guard"]["consumed_multiple_of_17"] is True
      and not res["fixed_guard"]["violations"])
res["ok"] = ok
print(json.dumps(res, indent=1))
json.dump(res, open(OUT, "w"), indent=1)
sys.exit(0 if ok else 1)
