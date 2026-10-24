#!/usr/bin/env python3
"""GPU check of the attempt-1 fix (one GPU, no model, synthetic tensors).  Runs in the capture image.

  G1  reproduce: with the poller running (as attempt 1 started it, at registration), capture a CUDA graph that
      contains the MoE and attention hooks plus enough work to span several poll periods. Expected: the capture
      fails (cudaErrorStreamCaptureInvalidated or a capture-time error).
  G2  fix: fresh module, no poller during capture. Capture the same graph, then for 3 steps: arm on the host
      (_upload, as prepare_attn does before a replay), fill the static inputs, replay, disarm. Then start the poller
      (as the fix does after the first non-dummy forward) and flush. Expected: capture and replays succeed, no
      capture_error, and every captured MoE entry equals the inputs of its step bit for bit.

usage: gpu_graph_test.py MOD_PATH OUT_JSON
"""
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback

MOD_PATH, OUT_JSON = sys.argv[1], sys.argv[2]
CAPDIR = "/tmp/p3b-gputest"
os.environ["GLM53_P3B"] = "capture"
os.environ["GLM53_P3B_DIR"] = CAPDIR
os.environ["GLM53_P3B_POLL_SECS"] = "0.05"
import numpy as np  # noqa: E402
import torch  # noqa: E402

DEV = torch.device("cuda:0")
HID, TOPK, W, RB, HEADS, QD, VD = 256, 8, 2048, 528, 8, 64, 64
RESULTS = []


def result(name, ok, **kw):
    RESULTS.append(dict(test=name, ok=None if ok is None else bool(ok), **kw))
    print(("INFO " if ok is None else ("PASS " if ok else "FAIL ")) + name + "  " + json.dumps(kw, default=str)[:500],
          flush=True)


class Obj:
    pass


class FakeBatch:
    def __init__(self, n, nreq):
        self.num_tokens = self.num_tokens_after_padding = n
        self.num_reqs = nreq
        self.has_prefill = False
        self.num_draft_tokens = n - nreq
        per = n // nreq
        self.query_start_loc_np = np.array([i * per for i in range(nreq + 1)])
        self.num_computed_tokens_np = np.array([100 + 37 * i for i in range(nreq)])
        self.req_ids = [f"chatcmpl-{i}" for i in range(nreq)]


def fresh():
    shutil.rmtree(CAPDIR, ignore_errors=True)
    spec = importlib.util.spec_from_file_location(f"cap{time.time_ns()}", MOD_PATH)
    cap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cap)
    S = cap.S
    S.rank = 0
    layers = {L: Obj() for L in (3, 23, 44)}
    impls = {L: Obj() for L in (3, 23)}
    for L, o in layers.items():
        S.moe_mod[id(o)] = L
    for L, o in impls.items():
        S.att_impl[id(o)] = L
        S.att_info[L] = dict(input_num_heads=HEADS, kernel_num_heads=HEADS, q_heads_cap=HEADS, q_head_dim=QD,
                             v_head_dim=VD, record_bytes=RB, topk_tokens=W, returns_lse=False)
    pos = torch.zeros(4096, dtype=torch.int64, device=DEV)
    cap._setup(DEV, {L: f"model.layers.{L}.mlp.experts" for L in layers}, HID, TOPK, pos)
    cap._mkdir_private(cap.PRIV_DIR)
    return cap, layers, impls


def static_inputs(m=32, n=32, nslots=64 * 64):
    g = torch.Generator(device=DEV).manual_seed(5)
    t = dict(x={L: torch.zeros(m, HID, dtype=torch.bfloat16, device=DEV) for L in (3, 23, 44)},
             ids={L: torch.zeros(m, TOPK, dtype=torch.int32, device=DEV) for L in (3, 23, 44)},
             w={L: torch.zeros(m, TOPK, device=DEV) for L in (3, 23, 44)},
             out={L: torch.zeros(m, HID, dtype=torch.bfloat16, device=DEV) for L in (3, 23, 44)},
             logical=torch.randint(-1, 3000, (n, W), generator=g, dtype=torch.int32, device=DEV),
             sel=torch.randint(0, nslots, (n, W), generator=g, dtype=torch.int32, device=DEV),
             L=torch.randint(10, 3000, (n,), generator=g, dtype=torch.int32, device=DEV),
             kv=torch.randint(0, 256, (nslots // 64, 64, RB), generator=g, dtype=torch.uint8, device=DEV),
             q=torch.randn(n, HEADS, QD, generator=g, device=DEV).to(torch.bfloat16),
             aout=torch.randn(n, HEADS, VD, generator=g, device=DEV).to(torch.bfloat16),
             a=torch.randn(2048, 2048, device=DEV))
    return t


def forward(cap, layers, impls, t, heavy=0):
    for L, lay in layers.items():
        cap.hook_moe(lay, t["x"][L], t["w"][L], t["ids"][L], t["out"][L])
    for L, imp in impls.items():
        n = t["q"].shape[0]
        cap.hook_attn_pre(imp, t["logical"], t["sel"], t["L"], n)
        nsa = t["L"].clamp(max=W) - 1
        cap.hook_attn_in(imp, t["q"], t["kv"], t["sel"], nsa, t["L"].clone(), n)
        cap.hook_attn_out(imp, t["aout"], None)
    b = t["a"]
    for _ in range(heavy):          # keeps the capture open across several poll periods
        b = torch.tanh(b @ t["a"]) * 0.5
    return b


def g1():
    cap, layers, impls = fresh()
    t = static_inputs()
    cap._ensure_poller()                  # attempt-1 behaviour: poller running before graph capture
    time.sleep(0.3)
    err = None
    try:
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            forward(cap, layers, impls, t, heavy=0)   # no cuBLAS inside capture (first call needs a warm-up)
            time.sleep(0.5)               # host time inside the capture: several 0.05 s poll periods
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)[:300]
    cap.S.stop.set()
    result("G1 poller running during capture invalidates it (attempt-1 reproduction)", err is not None, error=err)
    try:
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


def g2():
    cap, layers, impls = fresh()
    S = cap.S
    t = static_inputs()
    err = None
    try:
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            forward(cap, layers, impls, t, heavy=0)
            time.sleep(0.5)               # same host time inside the capture as G1, without a poller
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)[:300] + traceback.format_exc()[-600:]
    result("G2a capture with the hooks and no poller succeeds", err is None, error=err)
    if err:
        return
    S.phase = "traffic"
    g = torch.Generator(device=DEV).manual_seed(9)
    sent = {}
    for step in range(3):
        S.in_exec = True
        cap._upload(None, FakeBatch(32, 8))
        for L in (3, 23, 44):
            t["x"][L].copy_(torch.randn(32, HID, generator=g, device=DEV).to(torch.bfloat16))
            t["ids"][L].copy_(torch.randint(0, 288, (32, TOPK), generator=g, device=DEV, dtype=torch.int32))
            t["w"][L].copy_(torch.rand(32, TOPK, generator=g, device=DEV))
            t["out"][L].copy_(torch.randn(32, HID, generator=g, device=DEV).to(torch.bfloat16))
        ps = S.pending_step
        if ps and ps.get("arm_moe"):
            sent[ps["seq_moe"]] = {L: (t["x"][L].cpu().clone(), t["ids"][L].cpu().clone()) for L in (3, 23, 44)}
        gr.replay()
        S.meta[cap.M_ARM_MOE:cap.M_ARM_MOE + 1].zero_()
        S.meta[cap.M_ARM_ATT:cap.M_ARM_ATT + 1].zero_()
        S.in_exec = False
        S.pending_step = None
        torch.cuda.synchronize()
    cap._ensure_poller()                  # the fix: started after the first non-dummy forward
    time.sleep(1.5)
    st = cap.poll("final")
    cap.S.stop.set()
    errs = []
    for p in sorted(os.listdir(os.path.join(CAPDIR, "log"))) if os.path.isdir(os.path.join(CAPDIR, "log")) else []:
        for line in open(os.path.join(CAPDIR, "log", p)):
            if '"capture_error"' in line:
                errs.append(line[:300])
    matched, checked = 0, 0
    for seq, per in sent.items():
        for L, (x, ids) in per.items():
            f = os.path.join(cap.PRIV_DIR, "moe", f"L{L}", f"s{seq:05d}.npz")
            if os.path.exists(f):
                d = dict(np.load(f))
                checked += 1
                ok = "x" in d and "ids" in d
                if ok:
                    xa, ia = d["x"].reshape(-1, d["x"].shape[-1]), d["ids"].reshape(-1, d["ids"].shape[-1])
                    m = x.shape[0]
                    xbits = xa[:m].view(np.int16) if xa.dtype.itemsize == 2 else None
                    ok = (xbits is not None and np.array_equal(xbits, x.view(torch.int16).numpy())
                          and np.array_equal(ia[:m, :ids.shape[1]].astype(np.int64), ids.numpy().astype(np.int64)))
                matched += int(ok)
    result("G2b replays recorded; no capture_error records", not errs and checked > 0, capture_errors=errs[:3],
           armed_steps=len(sent), entries_checked=checked)
    result("G2c captured MoE entries equal the step inputs bit for bit", checked > 0 and matched == checked,
           matched=matched, checked=checked, status_keys=sorted(st.keys())[:12] if isinstance(st, dict) else None)


t0 = time.time()
g2()                     # run the fix first, on a clean CUDA context
g1()
ok = all(r["ok"] for r in RESULTS if r["ok"] is not None)
json.dump(dict(results=RESULTS, ok=ok, secs=round(time.time() - t0, 1)), open(OUT_JSON, "w"), indent=1)
print("ALL PASS" if ok else "SOME FAILED", flush=True)
os._exit(0 if ok else 1)
