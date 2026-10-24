#!/usr/bin/env python3
"""CPU tests of p3b_capture_hooks (no GPU, no vLLM runner): run inside the stock image with
CUDA_VISIBLE_DEVICES= and --network none (tests/run_cpu_tests.sh).

usage: test_capture_cpu.py MODULE_PATH OUT_DIR

Every test drives the module's own functions (setup, _upload arming, hooks, poll, poller loop) with CPU tensors
and fake runner objects.  The data are random fabrications, never captured inputs.
  T1  ring primitives: target slot (armed -> (seq-1)%R, disarmed -> sink), narrowed index_copy_, seqlock header,
      device commit counter
  T2  MoE capture end to end: arming quotas/stride, hooks on 3 layers, poll -> .npz byte-exact vs the inputs,
      disarmed steps write only the sink, ring wrap over 3 x RM entries, every entry emitted exactly once
  T3  backpressure: without polls the host stops arming after R unresolved entries; nothing is overwritten
  T4  abandoned entries: an armed step whose hook never commits (one layer skipped, or m > MMAX) is resolved as
      abandoned once done_step passes it; it never blocks the ring
  T5  attention H1/H2/H3: pre-compaction table, post table, nsa, cache lengths, positions, consumed L, q, KV records
      gathered by the PRE table (-1 -> slot 0, out-of-range clamped and counted), slot 0, output; H2 without H1
      and an H1-only (extend path) step never commit
  T6  graph-static: the aten op sequence of every hook is identical for armed / disarmed / different seq / rows /
      meta values (what a FULL graph captures once is valid for every replay), and contains no host-sync or
      data-dependent-shape op; Python-level tensor->host conversions raise during hooks and are never hit
  T7  poller race safety: (a) the 22ap allocation race: buffers allocated by _alloc_pins under inference_mode in
      one thread are writable from another (the pre-22ap allocation reproduces the failure); (b) a torn copy
      (header changed between the two header reads) is detected, not emitted, and re-read later; (c) an engine
      thread and the real poller thread run concurrently over many steps: every committed entry emitted exactly
      once, byte-exact, no torn emission, no overwrite before read
  T8  attention row choice: categories (dense / boundary / clamped) and quota deficits
  T9  FLUSH protocol through the real poller loop: FLUSHED.json once every armed entry is resolved
  T10 privacy: PRIVATE dirs 0700 / files 0600; log records hold no tensor values and no positions or lengths
  T11 Caps serialization round trip (float hex, dtype, device, enum)
"""
import dataclasses
import enum
import glob
import importlib.util
import json
import os
import shutil
import stat
import sys
import threading
import time
import traceback

MOD_PATH, OUT = sys.argv[1], sys.argv[2]
CAPDIR = os.path.join(OUT, "capdir")
shutil.rmtree(CAPDIR, ignore_errors=True)
os.environ["GLM53_P3B"] = "capture"
os.environ["GLM53_P3B_DIR"] = CAPDIR
os.environ["GLM53_P3B_POLL_SECS"] = "0.05"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

RESULTS = []


def load_module():
    spec = importlib.util.spec_from_file_location("p3b_capture_hooks", MOD_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def result(name, ok, **kw):
    RESULTS.append(dict(test=name, ok=bool(ok), **kw))
    print(("PASS " if ok else "FAIL ") + name + "  " + json.dumps(kw, default=str)[:600], flush=True)


# ---------------------------------------------------------------- fakes
class FakeLayer:
    pass


class FakeImpl:
    pass


W = 2048
RB = 528
HEADS, QD, VD = 8, 64, 64          # small heads/dims keep the CPU test fast; widths and record bytes as served
HID, TOPK = 256, 8


class FakeBatch:
    def __init__(self, n, npad, nreq, prefill=False, ndraft=None, nc=None, qsl=None, req_ids=None):
        self.num_tokens, self.num_tokens_after_padding, self.num_reqs = n, npad, nreq
        self.has_prefill = prefill
        self.num_draft_tokens = (n - nreq) if ndraft is None else ndraft
        per = max(n // max(nreq, 1), 1)
        self.query_start_loc_np = np.array(qsl if qsl is not None else [i * per for i in range(nreq + 1)])
        self.num_computed_tokens_np = np.array(nc if nc is not None else [100 + 37 * i for i in range(nreq)])
        self.req_ids = req_ids or [f"chatcmpl-{i}" for i in range(nreq)]


MODULES = []


def fresh(m, moe_layers=(3, 23, 44), att_layers=(3, 23)):
    """A new module instance with rings set up on CPU (as register() does, without a vLLM runner)."""
    for old in MODULES:                      # stop earlier instances' poller threads (same output dir)
        old.S.stop.set()
        if old.S.poller is not None:
            old.S.poller.join(timeout=5)
    shutil.rmtree(CAPDIR, ignore_errors=True)
    cap = load_module()
    MODULES.append(cap)
    S = cap.S
    S.rank = 0
    layers = {L: FakeLayer() for L in moe_layers}
    impls = {L: FakeImpl() for L in att_layers}
    for L, o in layers.items():
        S.moe_mod[id(o)] = L
    for L, o in impls.items():
        S.att_impl[id(o)] = L
        S.att_info[L] = dict(input_num_heads=HEADS, kernel_num_heads=HEADS, q_heads_cap=HEADS, q_head_dim=QD,
                             v_head_dim=VD, record_bytes=RB, topk_tokens=W, returns_lse=False)
    pos = torch.zeros(4096, dtype=torch.int64)
    cap._setup(torch.device("cpu"), {L: f"model.layers.{L}.mlp.experts" for L in moe_layers}, HID, TOPK, pos)
    cap._mkdir_private(cap.PRIV_DIR)
    return cap, layers, impls, pos


def moe_inputs(g, m):
    x = (torch.randn(m, HID, generator=g)).to(torch.bfloat16)
    ids = torch.randint(0, 288, (m, TOPK), generator=g, dtype=torch.int32)
    w = torch.rand(m, TOPK, generator=g)
    out = (torch.randn(m, HID, generator=g)).to(torch.bfloat16)
    return x, ids, w, out


def run_step(cap, layers, batch, g, record=None, skip_layer=None, m_override=None):
    """One engine step: execute_model-like arming (prepare_attn -> _upload), MoE hooks, disarm."""
    cap.S.in_exec = True
    cap._upload(None, batch)
    m = m_override or batch.num_tokens_after_padding
    ins = {}
    for L, lay in layers.items():
        x, ids, w, out = moe_inputs(g, m)
        ins[L] = (x, ids, w, out)
        if L != skip_layer:
            cap.hook_moe(lay, x, w, ids, out)
        elif L == cap.S.last_moe_layer:
            cap.S.done_step.copy_(cap.S.meta[cap.M_STEP:cap.M_STEP + 1])
    cap.S.meta[cap.M_ARM_MOE:cap.M_ARM_MOE + 1].zero_()
    cap.S.meta[cap.M_ARM_ATT:cap.M_ARM_ATT + 1].zero_()
    cap.S.in_exec = False
    if record is not None:
        ps = cap.S.pending_step
        if ps and ps["arm_moe"]:
            record[ps["seq_moe"]] = ins
    cap.S.pending_step = None
    return ins


def load_entry(cap, kind, L, seq):
    p = os.path.join(cap.PRIV_DIR, kind, f"L{L}", f"s{seq:05d}.npz")
    return dict(np.load(p)) if os.path.exists(p) else None


def bf16_np(t):
    return t.view(torch.int16).numpy().view(np.uint16)


# ---------------------------------------------------------------- tests
def t1():
    cap, layers, impls, pos = fresh(1)
    meta = cap.S.meta
    ring = cap.S.moe[3]
    meta[cap.M_ARM_MOE] = 1
    ok = True
    for seq in range(1, 2 * cap.RM + 2):
        meta[cap.M_SEQ_MOE] = seq
        tgt = cap.target_slot(meta, cap.M_ARM_MOE, cap.M_SEQ_MOE, ring.R)
        ok &= int(tgt[0]) == (seq - 1) % cap.RM
    meta[cap.M_ARM_MOE] = 0
    tgt = cap.target_slot(meta, cap.M_ARM_MOE, cap.M_SEQ_MOE, ring.R)
    ok_sink = int(tgt[0]) == cap.RM
    # narrowed index_copy_ (m < MMAX) touches only rows [0, m) of the slot
    meta[cap.M_ARM_MOE], meta[cap.M_SEQ_MOE], meta[cap.M_STEP] = 1, 3, 77
    tgt = cap.target_slot(meta, cap.M_ARM_MOE, cap.M_SEQ_MOE, ring.R)
    ring.t["x"].fill_(5.0)
    src = torch.full((7, HID), 2.0, dtype=torch.bfloat16)
    cap.hdr_begin(ring, tgt, meta, cap.M_SEQ_MOE, 7, 0)
    hb = ring.t["hdr"][2].clone()
    cap.put(ring.t["x"], tgt, src)
    cap.hdr_end(ring, tgt, meta, cap.M_SEQ_MOE, cap.M_ARM_MOE)
    x = ring.t["x"]
    ok_narrow = bool((x[2, :7] == 2).all() and (x[2, 7:] == 5).all() and (x[1] == 5).all() and (x[3] == 5).all())
    ok_hdr = (int(hb[cap.H_BEGIN]) == 3 and int(hb[cap.H_END]) == -1 and int(ring.t["hdr"][2, cap.H_END]) == 3
              and int(ring.t["hdr"][2, cap.H_STEP]) == 77 and int(ring.t["hdr"][2, cap.H_M]) == 7)
    c0 = int(ring.cnt[0])
    meta[cap.M_ARM_MOE] = 0
    tgt = cap.target_slot(meta, cap.M_ARM_MOE, cap.M_SEQ_MOE, ring.R)
    cap.hdr_end(ring, tgt, meta, cap.M_SEQ_MOE, cap.M_ARM_MOE)
    ok_cnt = c0 == 1 and int(ring.cnt[0]) == 1
    result("T1 ring primitives (slot, sink, narrowed copy, seqlock header, device counter)",
           ok and ok_sink and ok_narrow and ok_hdr and ok_cnt, slot_ok=ok, sink=ok_sink, narrow=ok_narrow,
           hdr=ok_hdr, counter=ok_cnt)


def t2():
    cap, layers, impls, pos = fresh(2)
    cap.S.phase = "traffic"
    g = torch.Generator().manual_seed(2)
    rec = {}
    sizes = [32] * 70 + [24] * 40 + [16] * 40 + [8] * 20 + [4] * 20 + [64] * 12 + [1] * 12
    n_polls = 0
    sink_before = {L: cap.S.moe[L].t["x"][cap.RM].clone() for L in layers}
    for i, npad in enumerate(sizes):
        nreq = max(npad // 4, 1)
        run_step(cap, layers, FakeBatch(n=npad - (1 if npad > 4 else 0), npad=npad, nreq=nreq), g, rec)
        if i % 5 == 4:
            cap.poll("test")
            n_polls += 1
    cap.poll("final")
    st = cap._status()
    armed = sorted(cap.S.armed_moe)
    exact = True
    missing = 0
    for seq in armed:
        for L in layers:
            e = load_entry(cap, "moe", L, seq)
            if e is None:
                missing += 1
                continue
            x, ids, w, out = rec[seq][L]
            m = x.shape[0]
            exact &= (np.array_equal(e["x"][:m], bf16_np(x)) and np.array_equal(e["ids"][:m], ids.numpy().astype(np.int64))
                      and np.array_equal(e["w"][:m], w.numpy()) and np.array_equal(e["out"][:m], bf16_np(out))
                      and int(e["hdr"][cap.H_BEGIN]) == seq and int(e["hdr"][cap.H_END]) == seq
                      and int(e["hdr"][cap.H_M]) == m)
    files = glob.glob(os.path.join(cap.PRIV_DIR, "moe", "L*", "*.npz"))
    want = {32: 10, 24: 6, 16: 6, 8: 4, 4: 4, 64: 2, 1: 2}
    counts = {}
    for seq in armed:
        counts[cap.S.armed_moe[seq]["npad"]] = counts.get(cap.S.armed_moe[seq]["npad"], 0) + 1
    strides_ok = True
    by = {}
    for seq in armed:
        a = cap.S.armed_moe[seq]
        by.setdefault(a["npad"], []).append(a["step"])
    for v in by.values():
        strides_ok &= all(b - a >= cap.MOE_STRIDE for a, b in zip(v, v[1:]))
    dev_commits = {L: int(cap.S.moe[L].cnt[0]) for L in layers}
    sink_written = any(not torch.equal(sink_before[L], cap.S.moe[L].t["x"][cap.RM]) for L in layers)
    ok = (exact and missing == 0 and len(files) == 3 * len(armed) and counts == want and strides_ok
          and all(v == len(armed) for v in dev_commits.values()) and len(armed) > 3 * cap.RM and st["moe_quotas_met"]
          and sink_written and st["torn"] == 0)
    result("T2 MoE capture end to end (quotas, stride, byte-exact npz, wrap > 3R, sink for disarmed steps)", ok,
           armed=len(armed), files=len(files), missing=missing, exact=exact, counts=counts, strides_ok=strides_ok,
           dev_commits=dev_commits, wraps=len(armed) / cap.RM, quotas_met=st["moe_quotas_met"], polls=n_polls)


def t3():
    cap, layers, impls, pos = fresh(3)
    cap.S.phase = "traffic"
    g = torch.Generator().manual_seed(3)
    rec = {}
    for _ in range(200):
        run_step(cap, layers, FakeBatch(n=31, npad=32, nreq=8), g, rec)
    armed_before = len(cap.S.armed_moe)
    bp = cap.S.arm_skipped_backpressure
    # nothing overwritten: the ring still holds exactly entries 1..R
    hdr_ok = all(int(cap.S.moe[3].t["hdr"][(s - 1) % cap.RM, cap.H_BEGIN]) == s for s in range(1, armed_before + 1))
    cap.poll("drain")
    for _ in range(200):
        run_step(cap, layers, FakeBatch(n=31, npad=32, nreq=8), g, rec)
    cap.poll("drain2")
    armed_after = len(cap.S.armed_moe)
    exact = all(np.array_equal(load_entry(cap, "moe", 3, s)["x"][:32], bf16_np(rec[s][3][0]))
                for s in range(1, armed_after + 1))
    ok = armed_before == cap.RM and bp > 0 and hdr_ok and armed_after == 10 and exact
    result("T3 backpressure (no arming past R unresolved; resumes after a poll; nothing overwritten)", ok,
           armed_without_poll=armed_before, backpressure_skips=bp, armed_after_poll=armed_after, exact=exact)


def t4():
    cap, layers, impls, pos = fresh(4)
    cap.S.phase = "traffic"
    g = torch.Generator().manual_seed(4)
    run_step(cap, layers, FakeBatch(n=31, npad=32, nreq=8), g, skip_layer=23)
    seq1 = cap.S.seq_moe
    step1 = cap.S.armed_moe[seq1]["step"]
    cap.S.done_step.fill_(step1 - 1)          # as if the GPU had not finished step 1 yet
    cap.poll("p1")
    pending_before = seq1 not in cap.S.resolved_moe and (seq1, 23) not in cap.S.abandoned["moe"]
    cap.S.done_step.fill_(step1)
    cap.poll("p2")
    ab = cap.S.abandoned["moe"]
    em = cap.S.emitted["moe"]
    ok = ((seq1, 23) in ab and (seq1, 3) in em and (seq1, 44) in em and seq1 in cap.S.resolved_moe)
    # the next armed entry that reuses the slot (seq1 + RM) is not blocked by the abandoned layer
    # m > MMAX is never armed, and done_step still advances (the last MoE layer's hook runs for every m)
    cap2, layers2, _, _ = fresh(4)
    cap2.S.phase = "traffic"
    run_step(cap2, layers2, FakeBatch(n=100, npad=128, nreq=8), g)
    ok2 = cap2.S.seq_moe == 0 and int(cap2.S.done_step[0]) == 1
    # an entry abandoned on EVERY layer gives its quota back (the host re-arms that padded size)
    cap3, layers3, _, _ = fresh(4)
    cap3.S.phase = "traffic"
    cap3.S.in_exec = True
    cap3._upload(None, FakeBatch(n=31, npad=32, nreq=8))
    before = cap3.S.moe_armed_count[("traffic", 32)]
    cap3.S.done_step.fill_(1)
    cap3.S.meta[cap3.M_ARM_MOE] = 0
    cap3.poll("all-abandoned")
    ok2 &= before == 1 and cap3.S.moe_armed_count[("traffic", 32)] == 0 and 1 in cap3.S.resolved_moe
    result("T4 abandoned entries: pending until done_step reaches the step, then resolved; m > MMAX never armed",
           ok and ok2 and pending_before, pending_before_done=pending_before, abandoned=list(ab), emitted=len(em),
           done_step_m128=int(cap2.S.done_step[0]))


def att_case(cap, impls, g, n=32, nslots=64 * 64, rows=None):
    """Fabricate one verify step's attention tensors and run H1 -> (compaction) -> H2 -> H3."""
    L3 = impls[3]
    topk_logical = torch.randint(-1, 3000, (n, W), generator=g, dtype=torch.int32)
    pre = torch.randint(0, nslots, (n, W), generator=g, dtype=torch.int32)
    pre[:, W - 100:] = -1
    pre[0, 5] = nslots + 7         # out of range: must be clamped and counted
    L = torch.randint(10, 3000, (n,), generator=g, dtype=torch.int32)
    kv = torch.randint(0, 256, (nslots // 64, 64, RB), generator=g, dtype=torch.uint8)
    q = torch.randn(n, HEADS, QD, generator=g).to(torch.bfloat16)
    out = torch.randn(n, HEADS, VD, generator=g).to(torch.bfloat16)
    pos = cap.S.pos_buf
    pos[:n] = torch.randint(0, 5000, (n,), generator=g)
    sel = pre.clone()
    cap.hook_attn_pre(L3, topk_logical, sel, L, n)
    post = sel.clone()
    post[:, ::3] = -1               # "compaction + mask" changes the table in place after H1
    sel.copy_(post)
    nsa = L.clamp(max=W) - 1
    clen = L.clone()
    cap.hook_attn_in(L3, q, kv, sel, nsa, clen, n)
    cap.hook_attn_out(L3, out, None)
    return dict(logical=topk_logical, pre=pre, post=post, L=L, kv=kv, q=q, out=out, nsa=nsa, clen=clen,
                pos=pos[:n].clone())


def t5():
    cap, layers, impls, pos = fresh(5, att_layers=(3,))
    cap.S.phase = "traffic"
    g = torch.Generator().manual_seed(5)
    nreq, per = 8, 4
    nc = [100, 190, 400, 2100, 700, 1000, 1500, 60]
    b = FakeBatch(n=32, npad=32, nreq=nreq, nc=nc, qsl=[i * per for i in range(nreq + 1)])
    cap.S.in_exec = True
    cap._upload(None, b)
    rows = [int(v) for v in cap.S.meta[cap.M_ROWS:].tolist() if v >= 0]
    seq = cap.S.seq_att
    case = att_case(cap, impls, g)
    cap.S.done_step.copy_(cap.S.meta[cap.M_STEP:cap.M_STEP + 1])
    cap.S.meta[cap.M_ARM_ATT] = 0
    cap.poll("t5")
    e = load_entry(cap, "att", 3, seq)
    ok = e is not None and seq == 1 and len(rows) > 0
    checks = {}
    if ok:
        r = torch.tensor(rows + [0] * (cap.NA - len(rows)))
        nv = len(rows)
        nslots = case["kv"].shape[0] * 64
        flat = case["kv"].reshape(-1, RB)
        pre_r = case["pre"][r].to(torch.int64)
        exp_rec = flat[pre_r.clamp(0, nslots - 1).reshape(-1)].reshape(cap.NA, W, RB)
        checks = dict(
            pre=np.array_equal(e["pre"][:nv], case["pre"][rows].numpy()),
            post=np.array_equal(e["post"][:nv], case["post"][rows].numpy()),
            logical=np.array_equal(e["logical"][:nv], case["logical"][rows].numpy()),
            q=np.array_equal(e["q"][:nv], bf16_np(case["q"][rows])),
            out=np.array_equal(e["out"][:nv], bf16_np(case["out"][rows])),
            rec=np.array_equal(e["rec"][:nv], exp_rec[:nv].numpy()),
            slot0=np.array_equal(e["slot0"][0], flat[0].numpy()),
            pos=np.array_equal(e["scal"][:nv, cap.S_POS], case["pos"][rows].numpy()),
            L=np.array_equal(e["scal"][:nv, cap.S_L], case["L"][rows].numpy()),
            nsa=np.array_equal(e["scal"][:nv, cap.S_NSA], case["nsa"][rows].numpy()),
            clen=np.array_equal(e["scal"][:nv, cap.S_CLEN], case["clen"][rows].numpy()),
            valid=bool((e["scal"][:nv, cap.S_VALID] == 1).all() and (e["scal"][nv:, cap.S_VALID] == 0).all()),
            rows=np.array_equal(e["scal"][:nv, cap.S_ROW], np.array(rows)),
            nslots_in_hdr=int(e["hdr"][cap.H_AUX]) == nslots,
            oob_counted=int(cap.S.att[3].oob[0]) == (1 if 0 in rows else 0))
        ok = all(checks.values())
    # H2 without H1, and an H1-only (extend-path) step: neither commits
    cap.S.in_exec = True
    for _ in range(cap.ATT_STRIDE):
        cap._upload(None, b)
    seq2 = cap.S.seq_att
    c_before = int(cap.S.att[3].cnt[0])
    L3 = impls[3]
    n = 32
    cap.hook_attn_in(L3, torch.zeros(n, HEADS, QD, dtype=torch.bfloat16), torch.zeros(64, 64, RB, dtype=torch.uint8),
                     torch.zeros(n, W, dtype=torch.int32), torch.zeros(n, dtype=torch.int32),
                     torch.zeros(n, dtype=torch.int32), n)
    cap.hook_attn_out(L3, torch.zeros(n, HEADS, VD, dtype=torch.bfloat16), None)
    cap.hook_attn_pre(L3, torch.zeros(n, W, dtype=torch.int32), torch.zeros(n, W, dtype=torch.int32),
                      torch.zeros(n, dtype=torch.int32), n)
    cap.S.done_step.copy_(cap.S.meta[cap.M_STEP:cap.M_STEP + 1])
    cap.S.meta[cap.M_ARM_ATT] = 0
    cap.poll("t5b")
    no_commit = int(cap.S.att[3].cnt[0]) == c_before and (seq2, 3) in cap.S.abandoned["att"] and seq2 > seq
    result("T5 attention H1/H2/H3 capture (pre/post tables, KV by PRE table, slot 0, scalars, out; no commit "
           "without H1->H2->H3)", ok and no_commit, rows=len(rows), checks=checks, no_commit=no_commit)


class OpRecorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func.overloadpacket.__name__) + ":" + ",".join(
            "x".join(map(str, a.shape)) + str(a.dtype)[6:] for a in args if isinstance(a, torch.Tensor)))
        return func(*args, **(kwargs or {}))


FORBIDDEN = {"_local_scalar_dense", "nonzero", "masked_select", "_unique2", "unique_dim", "unique_consecutive",
             "is_nonzero", "equal", "item", "nonzero_static", "_assert_async"}


class HostSyncTrap:
    """Python-level host conversions raise while active (a hook must never take a data-dependent host path)."""

    def __enter__(self):
        self.saved = {}
        for name in ("__bool__", "item", "tolist", "numpy", "__int__", "__float__", "__index__"):
            self.saved[name] = getattr(torch.Tensor, name)

            def boom(*a, _n=name, **k):
                raise RuntimeError(f"host conversion {_n} inside a hook")
            setattr(torch.Tensor, name, boom)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(torch.Tensor, k, v)
        return False


def t6():
    cap, layers, impls, pos = fresh(6)
    g = torch.Generator().manual_seed(6)
    seqs = {}
    variants = [dict(arm_moe=0, seq_moe=0, arm_att=0, seq_att=0, rows=[-1] * 8, step=1),
                dict(arm_moe=1, seq_moe=5, arm_att=1, seq_att=3, rows=[0, 1, 2, 3, 8, 9, 10, 11], step=9),
                dict(arm_moe=1, seq_moe=13, arm_att=1, seq_att=7, rows=[4, 5, 6, 7, -1, -1, -1, -1], step=33),
                dict(arm_moe=0, seq_moe=99, arm_att=1, seq_att=1, rows=[31, 30, 29, 28, 27, 26, 25, 24], step=2)]
    n, m = 32, 32
    x, ids, w, out = moe_inputs(g, m)
    att_inputs = None
    errors = []
    for vi, v in enumerate(variants):
        meta = cap.S.meta
        meta[cap.M_ARM_MOE], meta[cap.M_SEQ_MOE] = v["arm_moe"], v["seq_moe"]
        meta[cap.M_ARM_ATT], meta[cap.M_SEQ_ATT] = v["arm_att"], v["seq_att"]
        meta[cap.M_STEP], meta[cap.M_NTOK], meta[cap.M_NTOK_PAD] = v["step"], n - 1, n
        meta[cap.M_ROWS:] = torch.tensor(v["rows"])
        if att_inputs is None:
            att_inputs = dict(logical=torch.randint(-1, 3000, (n, W), generator=g, dtype=torch.int32),
                              pre=torch.randint(-1, 4096, (n, W), generator=g, dtype=torch.int32),
                              L=torch.randint(1, 3000, (n,), generator=g, dtype=torch.int32),
                              kv=torch.randint(0, 255, (64, 64, RB), generator=g, dtype=torch.uint8),
                              q=torch.randn(n, HEADS, QD, generator=g).to(torch.bfloat16),
                              out=torch.randn(n, HEADS, VD, generator=g).to(torch.bfloat16))
        a = att_inputs
        rec = OpRecorder()
        before = cap.S.errors
        try:
            with HostSyncTrap(), rec:
                cap.hook_moe(layers[3], x, w, ids, out)
                cap.hook_moe(layers[44], x, w, ids, out)
                sel = a["pre"].clone()
                cap.hook_attn_pre(impls[3], a["logical"], sel, a["L"], n)
                cap.hook_attn_in(impls[3], a["q"], a["kv"], sel, a["L"], a["L"], n)
                cap.hook_attn_out(impls[3], a["out"], None)
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))
        if cap.S.errors != before:
            errors.append(f"variant {vi}: hook error count {cap.S.errors - before}")
        seqs[vi] = rec.ops
    base = seqs[0]
    same = all(seqs[k] == base for k in seqs)
    bad = sorted({o.split(":")[0] for o in base if o.split(":")[0] in FORBIDDEN})
    result("T6 graph-static hooks: identical aten op sequence for all meta values, no host-sync / data-dependent ops,"
           " no Python host conversion", same and not bad and not errors and len(base) > 20,
           n_ops=len(base), identical=same, forbidden=bad, errors=errors[:3],
           op_kinds=sorted({o.split(':')[0] for o in base}))


def t7():
    cap, layers, impls, pos = fresh(7)
    # (a) the 22ap allocation race
    got = {}

    def alloc_in_inference():
        with torch.inference_mode():
            got["old"] = [torch.empty(4)]                  # pre-22ap: allocated under inference mode
            got["new"] = cap._alloc_pins([torch.empty(4)], pin=False)

    th = threading.Thread(target=alloc_in_inference)
    th.start()
    th.join()
    try:
        got["old"][0].copy_(torch.ones(4))
        old_fails = False
    except RuntimeError:
        old_fails = True
    try:
        got["new"][0].copy_(torch.ones(4))
        new_ok = True
    except RuntimeError:
        new_ok = False
    # (b) torn copy: the header changes between the first and the last header copy
    cap.S.phase = "traffic"
    g = torch.Generator().manual_seed(7)
    rec = {}
    run_step(cap, layers, FakeBatch(n=31, npad=32, nreq=8), g, rec)
    orig = cap._copy_rows
    state = {"n": 0}

    def tearing(tensors, rows):
        out_ = orig(tensors, rows)
        if isinstance(rows[0], int) and state["n"] == 0:      # an entry copy (the snapshot uses slices)
            state["n"] += 1
            out_[-1] = out_[-1].clone()
            out_[-1][cap.H_BEGIN] += cap.RM          # as if entry seq + R overwrote the slot during the copy
        return out_
    cap._copy_rows = tearing
    cap.poll("tear")
    torn1 = cap.S.torn
    emitted_after_tear = len(cap.S.emitted["moe"])
    cap._copy_rows = orig
    cap.poll("retry")
    e = load_entry(cap, "moe", 3, 1)
    ok_b = torn1 == 1 and emitted_after_tear == 2 and e is not None and np.array_equal(e["x"][:32], bf16_np(rec[1][3][0]))
    # (c) engine thread vs the real poller thread
    cap2, layers2, _, _ = fresh(7)
    cap2.S.phase = "traffic"
    rec2 = {}
    stop = threading.Event()
    g2 = torch.Generator().manual_seed(71)
    sizes = [32, 24, 16, 8, 4]

    def engine():
        i = 0
        while not stop.is_set() and i < 1500:
            npad = sizes[(i // 7) % len(sizes)]
            run_step(cap2, layers2, FakeBatch(n=npad, npad=npad, nreq=max(npad // 4, 1)), g2, rec2)
            i += 1
            if i % 3 == 0:
                time.sleep(0.001)
        stop.set()

    cap2.S.poller = None
    cap2._ensure_poller()
    te = threading.Thread(target=engine)
    te.start()
    te.join(timeout=120)
    time.sleep(0.6)
    armed = sorted(cap2.S.armed_moe)
    emitted = cap2.S.emitted["moe"]
    exact = True
    for seq in armed:
        for L in layers2:
            ent = load_entry(cap2, "moe", L, seq)
            if ent is None:
                exact = False
                continue
            x = rec2[seq][L][0]
            exact &= np.array_equal(ent["x"][:x.shape[0]], bf16_np(x)) and int(ent["hdr"][cap2.H_BEGIN]) == seq
    files = glob.glob(os.path.join(cap2.PRIV_DIR, "moe", "L*", "*.npz"))
    ok_c = (exact and len(emitted) == 3 * len(armed) == len(files) and cap2.S.torn == 0 and len(armed) > cap2.RM
            and cap2.S.errors == 0)
    result("T7 poller race safety (22ap allocation race; torn copy detected and re-read; concurrent engine/poller: "
           "every entry exactly once, byte-exact)", old_fails and new_ok and ok_b and ok_c,
           old_alloc_fails=old_fails, alloc_pins_ok=new_ok, torn_detected=torn1, retry_exact=ok_b,
           concurrent=dict(armed=len(armed), emitted=len(emitted), files=len(files), torn=cap2.S.torn,
                           errors=cap2.S.errors, exact=exact, backpressure=cap2.S.arm_skipped_backpressure))


def t8():
    cap, layers, impls, pos = fresh(8)
    # requests: dense (c=100), boundary (c=125 -> p+1 126..129 crosses 128), clamped (c=2500), near-W (2040: skipped)
    nc = [100, 125, 2500, 2040, 300, 2600, 61, 900]
    b = FakeBatch(n=32, npad=32, nreq=8, nc=nc, qsl=[4 * i for i in range(9)])
    rows, cats = cap._att_choose(b)
    ok1 = len(rows) == 8 and len(cats) == 2 and "boundary" in cats   # boundary has the largest deficit (16)
    cap.S.att_armed_groups = {"dense": 0, "boundary": 16, "clamped": 0}
    rows2, cats2 = cap._att_choose(b)
    ok2 = set(cats2) <= {"dense", "clamped"} and len(cats2) == 2
    cap.S.att_armed_groups = {"dense": 12, "boundary": 16, "clamped": 10}
    rows3, cats3 = cap._att_choose(b)
    ok3 = rows3 == [] and cats3 == []
    whole = all(r // 4 == rows[0] // 4 for r in rows[:4]) and all(r // 4 == rows[4] // 4 for r in rows[4:])
    result("T8 attention row choice (whole request groups, category by deficit, near-width skipped, stops at target)",
           ok1 and ok2 and ok3 and whole, first=cats, second=cats2, at_target=cats3, whole_groups=whole)


def t9():
    cap, layers, impls, pos = fresh(9)
    cap.S.phase = "traffic"
    g = torch.Generator().manual_seed(9)
    for _ in range(30):
        run_step(cap, layers, FakeBatch(n=31, npad=32, nreq=8), g)
    cap.S.poller = None
    cap._ensure_poller()
    open(os.path.join(cap.OUT_DIR, "FLUSH"), "w").close()
    t0 = time.time()
    fl = os.path.join(cap.LOG_DIR, "FLUSHED.json")
    while not os.path.exists(fl) and time.time() - t0 < 20:
        time.sleep(0.05)
    d = json.load(open(fl)) if os.path.exists(fl) else {}
    ok = bool(d) and d.get("pending_at_flush") == 0 and d.get("emitted_moe") == 3 * len(cap.S.armed_moe)
    result("T9 FLUSH -> FLUSHED.json through the real poller loop", ok, flushed=bool(d),
           pending=d.get("pending_at_flush"), emitted=d.get("emitted_moe"), wait_s=round(time.time() - t0, 2))


def t10():
    cap, layers, impls, pos = fresh(10, att_layers=(3,))
    cap.S.phase = "traffic"
    g = torch.Generator().manual_seed(10)
    for _ in range(12):
        run_step(cap, layers, FakeBatch(n=31, npad=32, nreq=8), g)
    nc = [100, 190, 400, 2100, 700, 1000, 1500, 60]
    b = FakeBatch(n=32, npad=32, nreq=8, nc=nc, qsl=[4 * i for i in range(9)])
    cap.S.in_exec = True
    cap._upload(None, b)
    case = att_case(cap, impls, g)
    # distinctive values that must never appear in a log
    cap.S.done_step.copy_(cap.S.meta[cap.M_STEP:cap.M_STEP + 1])
    cap.poll("t10")
    bad_modes = []
    for root, dirs, fs in os.walk(cap.PRIV_DIR):
        if stat.S_IMODE(os.stat(root).st_mode) != 0o700:
            bad_modes.append(root)
        for f in fs:
            if stat.S_IMODE(os.stat(os.path.join(root, f)).st_mode) != 0o600:
                bad_modes.append(f)
    logtext = ""
    for f in glob.glob(os.path.join(cap.LOG_DIR, "*")):
        logtext += open(f).read()
    allowed = {"kind", "ring", "layer", "seq", "step", "m_static", "ntok", "ntok_pad", "phase", "cats", "nrows",
               "file", "sha256", "shapes", "t", "rank"}
    entry_keys_ok = True
    for f in glob.glob(os.path.join(cap.LOG_DIR, "cap_*.jsonl")):
        for line in open(f):
            d = json.loads(line)
            if d.get("kind") == "entry":
                entry_keys_ok &= set(d) <= allowed
    pos_vals = [int(v) for v in case["pos"].tolist() if v > 999 and v not in (2048, 4096)]
    leaked = [v for v in pos_vals if f" {v}," in logtext or f"[{v}," in logtext or f" {v}]" in logtext]
    n_files = len(glob.glob(os.path.join(cap.PRIV_DIR, "*", "L*", "*.npz")))
    result("T10 privacy (0700 dirs / 0600 files under PRIVATE-capture; entry records hold shapes/digests only; "
           "positions never in logs)", not bad_modes and entry_keys_ok and not leaked and n_files > 0,
           bad_modes=bad_modes[:4], entry_keys_ok=entry_keys_ok, leaked=leaked[:4], files=n_files)


def t11():
    cap = load_module()

    class E(enum.IntEnum):
        GLM_NEXT = 7

    @dataclasses.dataclass(frozen=True)
    class Caps:
        device: torch.device
        num_q_heads: int
        softmax_scale: float
        dtype: torch.dtype
        model_type: int
        mode: str
        scale_format: object = None
        head_major_output: bool = True

    c = Caps(torch.device("cpu"), 64, 0.0625 * 1.000000001, torch.bfloat16, E.GLM_NEXT, "decode")
    d = cap.serialize_caps(c)
    back = float.fromhex(d["softmax_scale"]["float_hex"])
    ok = (back == c.softmax_scale and d["dtype"] == {"dtype": "torch.bfloat16"} and d["device"] == {"device": "cpu"}
          and d["model_type"]["value"] == 7 and d["mode"] == "decode" and d["head_major_output"] is True
          and d["scale_format"] is None)
    json.dumps(d)
    result("T11 Caps serialization round trip", ok, caps=d)


def main():
    os.makedirs(OUT, exist_ok=True)
    for t in (t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11):
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            result(t.__name__ + " crashed", False, err=repr(exc), tb=traceback.format_exc()[-1500:])
    n_ok = sum(r["ok"] for r in RESULTS)
    print(f"{n_ok}/{len(RESULTS)} passed")
    json.dump(RESULTS, open(os.path.join(OUT, "test_capture_cpu.json"), "w"), indent=1, default=str)
    for old in MODULES:
        old.S.stop.set()
    shutil.rmtree(CAPDIR, ignore_errors=True)
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
