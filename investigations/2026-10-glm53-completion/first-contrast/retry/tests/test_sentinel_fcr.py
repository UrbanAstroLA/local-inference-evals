#!/usr/bin/env python3
"""[R-10] Sentinel boundary-state tests (PLAN.md amendment 22f s.2): the SSM and convolution components are evaluated
separately, each against its own PS-a / PS-b baseline; only the intended committed state and the relevant convolution
entries are compared. CPU container of the B1 image, --network none; inputs read-only.

Template: REAL non-speculative KDA prefill snapshots from attempt-2 run-a2 (hook-validation/run-a2/forcewin/
snap_V3ref-*_F1_r{0,1}.pt: 34 KDA layers per rank, ssm_sketch 1 x 256 fp32, conv 6 x 12,288 bf16 with rows 3..5 not
written by the prefill kernel), so the layer names, shapes, dtypes, layout and unwritten rows are the engine's own.
PS-a is the template; PS-b and PS-c are derived from it per case. Every case runs through sentinel.check() end to end
(a minimal run object; coverage / identity / bytes rows are not exercised here and are ignored) for both the current
module and the pre-22f module (sentinel.py.pre22f), whose result is recorded as evidence of what it missed.

Cases (expected for the CURRENT module: R-PS-state-ssm / R-PS-state-conv)
  clean
    K1  identical baseline, PS-c identical                                                  PASS / PASS
    K2  non-identical baseline in both components (rel ~1e-3), PS-c within 1.5 x            PASS / PASS
    K3  identical baseline, PS-c differs only in the conv rows the prefill does not write   PASS / PASS
    K4  identical baseline, conv stored dim-first (transposed layout), PS-c identical       PASS / PASS
    K5  speculative-format snapshots: PS-c differs only in a NON-committed SSM slot          PASS / PASS
  negative
    N1  convolution-only corruption (identical baseline; one relevant conv entry, one layer) PASS / FAIL
    N2  SSM-only corruption (identical baseline; committed state, one layer)                 FAIL / PASS
    N3  non-identical baseline (SSM only), then excessive convolution change in PS-c         PASS / FAIL
    N4  non-identical baseline (both components), then excessive convolution change         PASS / FAIL
    N5  non-identical baseline (both components), then excessive SSM change                  FAIL / PASS
    N6  speculative-format snapshots: the COMMITTED SSM slot of PS-c corrupted                FAIL / PASS
    N7  PS-c snapshot of a different frontier (not the intended state)                       FAIL / FAIL
    N8  PS-c snapshot missing its KDA conv tensor                                            PASS / FAIL
    N9  conv stored dim-first; convolution-only corruption of a relevant entry               PASS / FAIL
usage: test_sentinel_fcr.py FC_ROOT HV_FORCEWIN OUT_DIR  -> OUT_DIR/sentinel-tests.log, sentinel_tests.json; rc 0 iff
every case behaves as expected
"""
import copy
import glob
import importlib.machinery
import importlib.util
import json
import os
import sys

import torch

FC, HV, OUT = sys.argv[1:4]
sys.path.insert(0, os.path.join(FC, "retry", "analysis"))
sys.path.insert(0, os.path.join(FC, "analysis"))
import sentinel as cur  # noqa: E402


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path, loader=importlib.machinery.SourceFileLoader(name, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


old = load_mod(os.path.join(FC, "retry", "analysis", "sentinel.py.pre22f"), "sentinel_pre22f")


class Rows:
    def __init__(self):
        self.rows = []

    def add(self, i, kind, status, name, detail, ref=""):
        self.rows.append(dict(id=i, kind=kind, status=status, name=name, detail=detail, ref=ref))

    def st(self, i):
        return next((r["status"] for r in self.rows if r["id"] == i), None)


class FakeRun:
    """the attributes sentinel.check() reads; snapshots held in memory ('path' = key)."""

    def __init__(self, snaps):
        self.store = {}
        self.reqs, self.by_tag, self._sn = [], {}, {}
        for ph, ranks in snaps.items():
            tag = f"{ph}-X"
            self.reqs.append(dict(phase=ph, fixture="X", tag=tag))
            d = {}
            for rk, s in ranks.items():
                key = f"{tag}/r{rk}"
                self.store[key] = s
                d[rk] = key
            self._sn[tag] = [(1, d)]

    def win_dump(self, tag):
        return None

    def snaps(self, tag):
        return self._sn.get(tag, [])

    def load(self, p):
        return self.store[p]


tmpl_paths = sorted(glob.glob(os.path.join(HV, "snap_V3ref-*_F1_r0.pt")))
assert tmpl_paths, "no real V3ref snapshot found"
p0 = tmpl_paths[0]
T = {0: torch.load(p0, weights_only=False), 1: torch.load(p0.replace("_r0.pt", "_r1.pt"), weights_only=False)}
P = int(T[0]["frontier_pos"]) + 1
assert all(int(T[r]["frontier_pos"]) == P - 1 and not any(L.get("is_spec") for L in cur.fc.kda_layers(T[r]).values())
           for r in (0, 1)), "template must be a non-speculative prefill snapshot"
KL = sorted(cur.fc.kda_layers(T[0]))
G = dict(order=["X"], fx={"X": dict(P=P)})
gen = torch.Generator().manual_seed(20261015)


def clone():
    return {r: copy.deepcopy(T[r]) for r in (0, 1)}


def noisy(t, rel):
    """t + Gaussian noise of relative L2 size `rel` (float32; cast back to t's dtype)."""
    x = t.float()
    n = torch.randn(x.shape, generator=gen)
    y = x + n * (rel * float(x.norm()) / max(float(n.norm()), 1e-12))
    return y.to(t.dtype)


def edit(snaps, fn, layers=None, ranks=(0, 1)):
    for r in ranks:
        for nm in (layers or KL):
            fn(snaps[r]["layers"][nm], snaps[r])
    return snaps


def ssm_rel(rel):
    def f(L, s):
        L["ssm_sketch"] = L["ssm_sketch"].clone()
        L["ssm_sketch"][0] = noisy(L["ssm_sketch"][0], rel)
    return f


def conv_rel(rel, rows=(0, 1, 2)):
    def f(L, s):
        c = L["conv"].clone()
        c[list(rows)] = noisy(c[list(rows)], rel)
        L["conv"] = c
    return f


def conv_flip(L, s):
    c = L["conv"].clone()
    c[1, 100] = c[1, 100] + 0.5 if float(c[1, 100]) != 0 else torch.tensor(1.0, dtype=c.dtype)
    L["conv"] = c


def conv_flip_t(L, s):
    """conv_flip for the dim-first layout: (dim 100, row 1); row 1 is a relevant entry, dim 100 is beyond the first 3."""
    c = L["conv"].clone()
    c[100, 1] = c[100, 1] + 0.5 if float(c[100, 1]) != 0 else torch.tensor(1.0, dtype=c.dtype)
    L["conv"] = c


def ssm_flip(L, s):
    x = L["ssm_sketch"].clone()
    x[0, 7] = x[0, 7] * 1.5 + 0.25
    L["ssm_sketch"] = x


def transpose_conv(L, s):
    L["conv"] = L["conv"].t().contiguous()


def to_spec(slot_count=4, na=2):
    """speculative-format copy: ssm_sketch K + 1 slots with the real state in the committed slot na - 1, the other slots
    unrelated; conv rows mapped for num_accepted = na (fc_common.conv_entries: rows 0..na + 1 committed, positions
    p0 - 2 + e with p0 = frontier - (na - 1))."""
    def f(L, s):
        real = L["ssm_sketch"][0].clone()
        sk = torch.stack([noisy(real, 0.8) for _ in range(slot_count)])
        sk[na - 1] = real
        L["ssm_sketch"] = sk
        L["is_spec"] = True
        s["num_accepted"] = na
    return f


def other_slot(L, s):
    x = L["ssm_sketch"].clone()
    x[0] = x[0] * -3.0          # slot 0 is not the committed slot when num_accepted = 2
    L["ssm_sketch"] = x


def committed_slot(L, s):
    x = L["ssm_sketch"].clone()
    x[1, 3] = x[1, 3] + 1.0
    L["ssm_sketch"] = x


def drop_conv(L, s):
    L.pop("conv", None)


def frontier_shift(snaps):
    for r in (0, 1):
        snaps[r]["frontier_pos"] = P - 1 + 4352
    return snaps


one = [KL[len(KL) // 2]]
cases = {}
# clean
cases["K1 identical baseline, PS-c identical"] = (clone(), clone(), clone(), ("PASS", "PASS"))
b = edit(edit(clone(), ssm_rel(1e-3)), conv_rel(1e-3))
cases["K2 non-identical baseline (both), PS-c within 1.5 x"] = (
    clone(), b, edit(edit(clone(), ssm_rel(1.5e-3)), conv_rel(1.5e-3)), ("PASS", "PASS"))
cases["K3 PS-c differs only in conv rows the prefill does not write (3..5)"] = (
    clone(), clone(), edit(clone(), conv_rel(0.5, rows=(3, 4, 5))), ("PASS", "PASS"))
cases["K4 conv stored dim-first (transposed layout), PS-c identical"] = (
    edit(clone(), transpose_conv), edit(clone(), transpose_conv), edit(clone(), transpose_conv), ("PASS", "PASS"))
sp = lambda: edit(clone(), to_spec())  # noqa: E731
torch.manual_seed(0)
spa = sp()
cases["K5 speculative format: PS-c differs only in a NON-committed SSM slot"] = (
    spa, copy.deepcopy(spa), edit(copy.deepcopy(spa), other_slot), ("PASS", "PASS"))
# negatives
cases["N9 conv stored dim-first, convolution-only corruption of a relevant entry"] = (
    edit(clone(), transpose_conv), edit(clone(), transpose_conv),
    edit(edit(clone(), transpose_conv), conv_flip_t, layers=one, ranks=(0,)), ("PASS", "FAIL"))
cases["N1 convolution-only corruption (identical baseline, one entry, one layer)"] = (
    clone(), clone(), edit(clone(), conv_flip, layers=one, ranks=(1,)), ("PASS", "FAIL"))
cases["N2 SSM-only corruption (identical baseline, committed state, one layer)"] = (
    clone(), clone(), edit(clone(), ssm_flip, layers=one, ranks=(0,)), ("FAIL", "PASS"))
bs = edit(clone(), ssm_rel(1e-3))
cases["N3 non-identical baseline (SSM only), then excessive conv change"] = (
    clone(), bs, edit(edit(clone(), ssm_rel(1e-3)), conv_rel(5e-2), layers=one), ("PASS", "FAIL"))
cases["N4 non-identical baseline (both), then excessive conv change"] = (
    clone(), b, edit(edit(edit(clone(), ssm_rel(1e-3)), conv_rel(1e-3)), conv_rel(5e-2), layers=one), ("PASS", "FAIL"))
cases["N5 non-identical baseline (both), then excessive SSM change"] = (
    clone(), b, edit(edit(edit(clone(), ssm_rel(1e-3)), conv_rel(1e-3)), ssm_rel(5e-2), layers=one), ("FAIL", "PASS"))
cases["N6 speculative format: the COMMITTED SSM slot of PS-c corrupted"] = (
    spa, copy.deepcopy(spa), edit(copy.deepcopy(spa), committed_slot, layers=one), ("FAIL", "PASS"))
cases["N7 PS-c snapshot of a different frontier (not the intended state)"] = (
    clone(), clone(), frontier_shift(clone()), ("FAIL", "FAIL"))
cases["N8 PS-c snapshot missing its KDA conv tensor (one layer)"] = (
    clone(), clone(), edit(clone(), drop_conv, layers=one), ("PASS", "FAIL"))

res, n_ok = {}, 0
lines = ["template: %s (P %d, %d KDA layers per rank)" % (os.path.basename(p0), P, len(KL))]
for name, (a, b_, c, exp) in cases.items():
    run = FakeRun({"PS-a": a, "PS-b": b_, "PS-c": c})
    R = Rows()
    cur.check(run, G, R, [], phases=("PS-a", "PS-b", "PS-c"), build_users=(), prefix="R-PS")
    got = (R.st("R-PS-state-ssm"), R.st("R-PS-state-conv"))
    rows_present = all(R.st(i) is not None for i in ("R-PS-coverage", "R-PS-identity", "R-PS-bytes", "R-PS-limits"))
    Ro = Rows()
    try:
        old.check(run, G, Ro, [], phases=("PS-a", "PS-b", "PS-c"), build_users=(), prefix="R-PS")
        got_old = Ro.st("R-PS-state")
    except Exception as exc:  # noqa: BLE001
        got_old = "raised %r" % exc
    should_fail = "FAIL" in exp
    ok = got == exp and rows_present             # the unchanged coverage / identity / bytes rows are still emitted
    n_ok += ok
    old_note = ("pre-22f %s (%s)" % (got_old, "missed" if should_fail and got_old == "PASS" else
                                     ("false alarm" if not should_fail and got_old != "PASS" else "agrees")))
    lim = [r for r in R.rows if r["id"] == "R-PS-limits"]
    res[name] = dict(expected=exp, got=got, ok=ok, pre22f=got_old, limits_row=bool(lim),
                     ssm=next(r["detail"] for r in R.rows if r["id"] == "R-PS-state-ssm"),
                     conv=next(r["detail"] for r in R.rows if r["id"] == "R-PS-state-conv"))
    lines.append("%s %-82s ssm %s conv %s (expected %s / %s; coverage / identity / bytes / limits rows present %s); %s"
                 % ("PASS" if ok else "FAIL", name, got[0], got[1], exp[0], exp[1], rows_present, old_note))
neg = [k for k in cases if k.startswith("N")]
missed_old = [k for k in neg if res[k]["pre22f"] == "PASS"]
lines.append("sentinel tests: %d / %d PASS (clean %d, negative %d); limits row present in every case: %s; the pre-22f "
             "module passes %d of the %d negatives: %s" % (
                 n_ok, len(cases), len(cases) - len(neg), len(neg), all(r["limits_row"] for r in res.values()),
                 len(missed_old), len(neg), [k.split(" ")[0] for k in missed_old]))
os.makedirs(OUT, exist_ok=True)
json.dump(dict(template=os.path.basename(p0), P=P, cases=res), open(os.path.join(OUT, "sentinel_tests.json"), "w"),
          indent=1, default=str)
open(os.path.join(OUT, "sentinel-tests.log"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(0 if n_ok == len(cases) else 1)
