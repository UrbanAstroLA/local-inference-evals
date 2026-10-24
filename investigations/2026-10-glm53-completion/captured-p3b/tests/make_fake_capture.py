#!/usr/bin/env python3
"""Fabricate a capture in exactly the format the served capture writes, by driving the real capture module's hooks
and poller on CPU with random data (no GPU; run in the stock image with TRITON_INTERPRET=1 so that the image's own
compaction kernels build the POST tables).  Used by tests/test_replay_dry.sh only; the data are fabricated, never
served inputs.

usage: make_fake_capture.py CAPTURE_MODULE P3_HARNESS OUTDIR
writes OUTDIR/cap (PRIVATE-capture layout) and OUTDIR/caplog (log layout + fabricated init/site/bind records),
and OUTDIR/expect.json (what the dry replays must find).

Injected faults: MoE layer 23, the first m = 24 entry: served row 3 changed by one bf16 ulp (BASELINE must report
not reproduced).  Attention layer 3, entry 1, row slot 2: served output changed (that row must be NOT ANCHORED).
"""
import importlib.util
import json
import os
import shutil
import sys

MOD, P3H, OUTD = sys.argv[1], sys.argv[2], sys.argv[3]
CAPDIR = os.path.join(OUTD, "capdir")
shutil.rmtree(OUTD, ignore_errors=True)
os.makedirs(OUTD)
os.environ["GLM53_P3B"] = "capture"
os.environ["GLM53_P3B_DIR"] = CAPDIR

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "replay"))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


cap = load(MOD, "p3b_capture_hooks")
P = load(P3H, "moe_plan_eq")
import replay_attn as RA  # noqa: E402
import vllm.v1.attention.backends.mla.b12x_mla_sparse as bm  # noqa: E402

W, RB, HEADS, QD, VD, HID = 2048, 528, 8, 64, 64, 256
S = cap.S
S.rank = 0


class Obj:
    pass


moe_layers = {L: Obj() for L in (3, 23, 44)}
att_impls = {L: Obj() for L in (3, 23)}
for L, o in moe_layers.items():
    S.moe_mod[id(o)] = L
for L, o in att_impls.items():
    S.att_impl[id(o)] = L
    S.att_info[L] = dict(input_num_heads=HEADS, kernel_num_heads=HEADS, q_heads_cap=HEADS, q_head_dim=QD,
                         v_head_dim=VD, record_bytes=RB, topk_tokens=W, returns_lse=False, num_splits_cap=32,
                         decode_caps={})
pos_buf = torch.zeros(4096, dtype=torch.int64)
cap._setup(torch.device("cpu"), {L: f"model.layers.{L}.mlp.experts" for L in moe_layers}, HID, 8, pos_buf)
cap._mkdir_private(cap.PRIV_DIR)
S.phase = "traffic"
g = torch.Generator().manual_seed(1234)
expect = dict(moe={}, att={})
step = [0]


def arm(kind, seq, rows=None, n=32, npad=32):
    step[0] += 1
    S.meta.zero_()
    S.meta[cap.M_ROWS:] = -1
    S.meta[cap.M_STEP] = step[0]
    S.meta[cap.M_NTOK], S.meta[cap.M_NTOK_PAD] = n, npad
    if kind == "moe":
        S.meta[cap.M_ARM_MOE], S.meta[cap.M_SEQ_MOE] = 1, seq
        S.armed_moe[seq] = dict(step=step[0], phase=phase_now[0], npad=npad)
    else:
        S.meta[cap.M_ARM_ATT], S.meta[cap.M_SEQ_ATT] = 1, seq
        for j, r in enumerate(rows):
            S.meta[cap.M_ROWS + j] = r
        S.armed_att[seq] = dict(step=step[0], phase="traffic", cats=["dense", "boundary"], nrows=len(rows))


def finish():
    S.meta[cap.M_ARM_MOE] = 0
    S.meta[cap.M_ARM_ATT] = 0
    S.done_step.fill_(step[0])
    cap.poll("fake")


phase_now = ["traffic"]
# ---------------------------------------------------------------- MoE
plan_of = lambda m: "decode" if m <= 32 else "prefill"  # noqa: E731
seq = 0
injected_moe = None
for m, n_real, phase in [(32, 32, "traffic"), (32, 30, "traffic"), (32, 32, "traffic"), (24, 22, "traffic"),
                         (24, 24, "traffic"), (16, 15, "traffic"), (16, 16, "traffic"), (8, 7, "traffic"),
                         (4, 4, "traffic"), (48, 44, "startup")]:
    seq += 1
    phase_now[0] = phase
    arm("moe", seq, n=n_real, npad=m)
    for L, lay in moe_layers.items():
        x = (torch.randn(m, HID, generator=g)).to(torch.bfloat16)
        ids = torch.randint(0, 288, (m, 8), generator=g, dtype=torch.int32)
        w = torch.rand(m, 8, generator=g)
        w = w / w.sum(1, keepdim=True)
        out, _ = P.dry_kernel(torch, "tp2r0", plan_of(m), x, w, ids.to(torch.int64))
        if L == 23 and m == 24 and injected_moe is None:
            out = out.clone()
            out.view(torch.int16)[3, 0] += 1
            injected_moe = seq
        cap.hook_moe(lay, x, w, ids, out)
    finish()
expect["moe"] = dict(entries=seq, injected=dict(layer=23, seq=injected_moe))

# ---------------------------------------------------------------- attention
kv = torch.randint(0, 256, (64 * 4, 64, RB), generator=g, dtype=torch.uint8)
nslots = kv.shape[0] * 64
flat = kv.reshape(-1, RB)
ai = S.att_info[3]
kern = RA.Kernel(torch, True, ai, torch.device("cpu"), {})


def group(p0, r, base):
    """4 verify rows of one request: positions p0..p0+3, consumed L = p + 1 + r, front-packed PRE tables."""
    rows = []
    for j in range(4):
        p = p0 + j
        Lc = p + 1 + r
        pre = torch.full((W,), -1, dtype=torch.int32)
        k = min(p + 1, W)
        if p + 1 <= W:
            pre[:k] = torch.arange(base, base + k, dtype=torch.int32)
        else:
            pre[:] = torch.arange(base + (p + 1 - W), base + (p + 1), dtype=torch.int32)
        rows.append((p, Lc, pre))
    return rows


specs = {
    3: [[(126, 2, 100), (2100, 1, 3000)], [(500, 1, 6000), (700, 0, 7000)], [(1000, 3, 1000), (2500, 2, 8000)]],
    23: [[(190, 3, 100), (2200, 0, 3000)], [(63, 1, 6000), (1500, 2, 9000)]],
}
aseq = 0
expect["att"] = {}
for L, entries in specs.items():
    for ent_i, groups in enumerate(entries):
        aseq += 1
        n = 32
        rows_idx = [0, 1, 2, 3, 12, 13, 14, 15]
        arm("att", aseq, rows=rows_idx, n=n, npad=n)
        rowsdat = group(*groups[0]) + group(*groups[1])
        pre_all = torch.full((n, W), -1, dtype=torch.int32)
        Lall = torch.ones(n, dtype=torch.int32)
        for j, ri in enumerate(rows_idx):
            p, Lc, pre = rowsdat[j]
            pre_all[ri] = pre
            Lall[ri] = Lc
            pos_buf[ri] = p
        logical = pre_all.clone()
        sel = pre_all.clone()
        cap.hook_attn_pre(att_impls[L], logical, sel, Lall, n)
        nsa = Lall.clamp(max=W).clone()
        bm._compact_dropped_selection(sel, nsa)
        bm._mask_page_table_after_nsa_len(sel, nsa)
        q = torch.randn(n, HEADS, QD, generator=g).to(torch.bfloat16)
        out, _ = kern.run(q, kv, sel, Lall, nsa)
        if L == 3 and ent_i == 0:
            out = out.clone()
            out.view(torch.int16)[2, 0, 0] += 1     # row slot 2 (batch row 2): must be NOT ANCHORED
        cap.hook_attn_in(att_impls[L], q, kv, sel, nsa, Lall, n)
        cap.hook_attn_out(att_impls[L], out, None)
        finish()
        expect["att"].setdefault(str(L), []).append(dict(seq=aseq, groups=groups))

# ---------------------------------------------------------------- capture log: init / sites / binds
logdir = os.path.join(CAPDIR, "log")
recs = [dict(kind="init", rank=0, schema=cap.SCHEMA, att_info={str(L): v for L, v in S.att_info.items()},
             ring_bytes={}, ring_mib_total=0, issues=[], async_scheduling=True, env={})]
for L in moe_layers:
    for m in (32, 24, 16, 8, 4, 48):
        recs.append(dict(kind="site", key=["moe", L, m], layer=L, m=m, captured=True, ids=[[m, 8], "torch.int32"]))
        recs.append(dict(kind="site", key=["moe_bind", L, m, plan_of(m)], layer=L, m=m, plan=plan_of(m), launch={}))
for L in att_impls:
    recs.append(dict(kind="site", key=["att_in", L, 32], layer=L, q=[[32, HEADS, QD], "torch.bfloat16"]))
    recs.append(dict(kind="site", key=["att_out", L, 32], layer=L, out=[[32, HEADS, VD], "torch.bfloat16", []]))
with open(os.path.join(logdir, "cap_fake_init.jsonl"), "w") as fh:
    for r in recs:
        fh.write(json.dumps(r) + "\n")
shutil.copytree(os.path.join(CAPDIR, "PRIVATE-capture"), os.path.join(OUTD, "cap"))
shutil.copytree(logdir, os.path.join(OUTD, "caplog"))
expect["att_injected"] = dict(layer=3, seq=1, row_slot=2)
expect["status"] = cap._status()
json.dump(expect, open(os.path.join(OUTD, "expect.json"), "w"), indent=1, default=str)
S.stop.set()
print(json.dumps(dict(moe_entries=seq, att_entries=aseq, files=sum(len(f) for _, _, f in os.walk(os.path.join(OUTD, "cap"))),
                      torn=S.torn, errors=S.errors)))
