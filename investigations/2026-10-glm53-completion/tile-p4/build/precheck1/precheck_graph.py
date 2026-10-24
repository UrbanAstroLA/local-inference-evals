#!/usr/bin/env python3
"""P4 single-GPU pre-check (PROPOSAL.md section 7; the captured-p3b attempt-1 lesson).  Runs in the built P4 image on
GPU0, no model, synthetic tensors at the served sizes (hidden 4096, vocab 154,880, 16 request slots, 8 stores).

  C1  CUDA-graph capture of all P4 in-graph hooks (MoE out x3, mHC hidden x2, final norm) at m = 32, with the
      module registered exactly as in serving (stages and stores allocated BEFORE capture, no poller anywhere), and
      a flush attempted INSIDE the capture: the capture succeeds, the flush is refused and writes nothing.
  C2  replay 5 steps (8 forced requests x 4 rows, the fixed accepted-count list), after each replay the eager bridge
      (on_logits with fp32 logits, on_commit of rows 0..a, on_step), then removal flushes: every stored row equals
      the step's static input of its token row bit for bit (MoE every recorded position, hidden at o % 8 == 0,
      logits at o % 64 == 0), have == 1 exactly there, and no p4_error record.
  C3  the same replay without the bridge leaves the stores untouched (in-graph hooks write stages only).
  N1  negative control, run last: a device-to-host copy inside a capture invalidates it (the failure mode the
      flush rule prevents).

usage: precheck_graph.py OUT_JSON     (env GLM53_P4=on GLM53_P4_DIR=/tmp/p4pre)
"""
import glob
import json
import os
import shutil
import sys
import time
import traceback
import types

OUT_JSON = sys.argv[1]
os.environ.setdefault("GLM53_P4", "on")
os.environ.setdefault("GLM53_P4_DIR", "/tmp/p4pre")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from vllm import p4_hooks as p4  # noqa: E402

DEV = torch.device(os.environ.get("P4_PRECHECK_DEV", "cuda:0"))   # cpu only in tests/test_precheck_cpu.py
G = {}
H, V, R = 4096, 154880, 16
M = 32
SCHED = [3, 1, 2, 0, 3]
RESULTS = []


def result(name, ok, **kw):
    RESULTS.append(dict(test=name, ok=bool(ok), **kw))
    print(("PASS " if ok else "FAIL ") + name + "  " + json.dumps(kw, default=str)[:400], flush=True)


class Mod:
    def __init__(self, mhc=False):
        self.mhc, self.n = mhc, 4

    def hc_post(self, h, r, post, comb):
        return (h.unsqueeze(1) * post.unsqueeze(-1)).to(h.dtype)    # [m, 4, H]; with post == 1 the contract == h


def setup():
    shutil.rmtree(p4.OUT_DIR, ignore_errors=True)
    p4.S.__init__()
    p4.S.rank = 0
    p4._alloc(DEV, H, V, R)
    mods = [Mod(), Mod(), Mod(), Mod(True), Mod(True), Mod()]
    for k, m in enumerate(mods):
        p4.S.ids[id(m)] = k
    p4.S.registered = True
    p4._mkdir_private(p4.PRIV_DIR)
    return mods


def forward(mods, st):
    for k in range(3):
        p4.hook_moe_out(mods[k], st["x"][k])
    for k in (3, 4):
        p4.hook_hidden(mods[k], st["x"][k], None, st["post"], None)
    p4.hook_final(mods[5], st["x"][5])


def main():
    t0 = time.time()
    mods = setup()
    st = dict(x=[torch.zeros(M, H, dtype=torch.bfloat16, device=DEV) for _ in range(6)],
              post=torch.ones(M, 4, dtype=torch.bfloat16, device=DEV))
    G.update(mods=mods, st=st)
    for _ in range(2):            # warm-up outside capture (allocator, kernels)
        forward(mods, st)
    torch.cuda.synchronize()
    p4.on_add(15, "inside-capture", 512)
    err = None
    refused_before = len(glob.glob(os.path.join(p4.PRIV_DIR, "*")))
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            forward(mods, st)
            p4.S.serving = True               # as if serving: the capture check alone must refuse the flush
            p4.on_remove(15, "inside-capture", 512)
            p4.S.serving = False
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)[:300] + traceback.format_exc()[-600:]
    files_after = len(glob.glob(os.path.join(p4.PRIV_DIR, "*")))
    logs = "".join(open(p).read() for p in glob.glob(os.path.join(p4.LOG_DIR, "p4_r0_*.jsonl")))
    result("C1 capture with every in-graph hook succeeds; a flush inside the capture is refused and writes nothing",
           err is None and files_after == refused_before and "flush_refused" in logs, error=err)
    if err:
        return
    # serving: first real batch outside capture
    p4.on_batch(types.SimpleNamespace(num_reqs=8, is_prefilling_np=np.zeros(8), num_tokens=M), 1)
    tags = [f"pc-{i}" for i in range(8)]
    for i, t in enumerate(tags):
        p4.on_add(i, t, 512)
    gen = torch.Generator(device=DEV).manual_seed(7)
    nout = [0] * 8             # committed output count per request (o of row 0)
    expect = {}                # (tag, kind, k, o) -> tensor
    for step in range(len(SCHED)):
        a = SCHED[step]
        for k in range(6):
            st["x"][k].copy_(torch.randn(M, H, generator=gen, device=DEV).to(torch.bfloat16))
        logits = torch.randn(M, V, generator=gen, device=DEV)
        g.replay()
        # sc as forcewin builds it: 8 requests x 4 rows; row j predicts o = nout + j
        ridx = torch.arange(8, device=DEV).repeat_interleave(4)
        j = torch.arange(4, device=DEV).repeat(8)
        o = torch.tensor([nout[i] for i in range(8)], device=DEV).repeat_interleave(4) + j
        sc = dict(L=M, ridx=ridx, o=o, fwin=torch.ones(M, dtype=torch.bool, device=DEV))
        ib = types.SimpleNamespace(logits_indices=torch.arange(M, device=DEV))
        p4.on_logits(sc, logits, ib)
        valid = j <= a
        p4.on_commit(sc, valid)
        p4.on_step(step + 1, torch.arange(8, device=DEV), o.view(8, 4)[:, 0], torch.full((8,), 4, device=DEV),
                   torch.full((8,), a, device=DEV))
        for i in range(8):
            for jj in range(a + 1):
                oo = nout[i] + jj
                row = 4 * i + jj
                for k in range(3):
                    expect[(tags[i], "moe", k, oo)] = st["x"][k][row].clone()
                if oo % 8 == 0:
                    for kk, k in enumerate((3, 4, 5)):
                        expect[(tags[i], "hid", kk, oo)] = st["x"][k][row].clone()
                if oo % 64 == 0:
                    expect[(tags[i], "logits", 0, oo)] = logits[row].clone()
            nout[i] += a + 1
        p4.S.last_step = step + 1
    torch.cuda.synchronize()
    store_before = p4.S.store.clone()
    g.replay()                                            # C3: replay without the bridge
    torch.cuda.synchronize()
    c3 = torch.equal(store_before, p4.S.store)
    for i, t in enumerate(tags):
        p4.on_remove(i, t, 512)
    bad, checked = [], 0
    for i, t in enumerate(tags):
        z = np.load(os.path.join(p4.PRIV_DIR, f"p4_{t}_r0.npz"))
        n = nout[i]
        have_ok = bool((z["moe_have"][:, :n] == 1).all()) and bool((z["moe_have"][:, n:] == 0).all())
        if not have_ok:
            bad.append((t, "moe_have"))
        for (tt, kind, k, oo), ref in expect.items():
            if tt != t:
                continue
            checked += 1
            if kind == "moe":
                got = z["moe"][k, oo]
                ok = np.array_equal(got, ref.view(torch.int16).cpu().numpy().view(np.uint16)) and z["moe_have"][k, oo] == 1
            elif kind == "hid":
                got = z["hid"][k, oo // 8]
                ok = np.array_equal(got, ref.view(torch.int16).cpu().numpy().view(np.uint16)) and z["hid_have"][k, oo // 8] == 1
            else:
                ok = np.array_equal(z["logits"][oo // 64], ref.cpu().numpy()) and z["logits_have"][oo // 64] == 1
            if not ok:
                bad.append((t, kind, k, oo))
    errs = [l for p in glob.glob(os.path.join(p4.LOG_DIR, "p4_r0_*.jsonl")) for l in open(p) if '"p4_error"' in l]
    result("C2 replays + eager bridge: every stored row == its step's input bit for bit; have exact; no p4_error",
           not bad and checked > 0 and not errs, checked=checked, bad=bad[:5], errors=errs[:3])
    result("C3 a replay without the bridge leaves the stores untouched", c3)
    # N1 negative control (last: may leave the context in an error state)
    err = None
    try:
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            forward(mods, st)
            _ = p4.S.stage[0, 0].cpu()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)[:200]
    result("N1 negative control: a device-to-host copy inside a capture fails it", err is not None, error=err)
    print("secs", round(time.time() - t0, 1))


if __name__ == "__main__":
    try:
        main()
    finally:
        ok = bool(RESULTS) and all(r["ok"] for r in RESULTS)
        json.dump(dict(results=RESULTS, verdict="PASS" if ok and len(RESULTS) == 4 else "FAIL"), open(OUT_JSON, "w"),
                  indent=1)
        print("VERDICT", "PASS" if ok and len(RESULTS) == 4 else "FAIL", flush=True)
        os._exit(0 if ok else 1)
