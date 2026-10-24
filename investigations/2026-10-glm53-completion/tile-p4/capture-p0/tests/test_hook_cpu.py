#!/usr/bin/env python3
"""CPU tests of cap0_hooks (point-0 capture and localisation) through p4_hooks, in a CPU container
(--network none, CUDA_VISIBLE_DEVICES=) of the stage-0 image with the capture-p0 files installed over it
("hooked") or of the built cap0 image ("image").  SYNTHETIC tensors only (random bytes in served layouts, a toy
decoder model); nothing here is model data.

Env: GLM53_P4=on GLM53_P4_S0=on GLM53_CAP0=on GLM53_CAP0_LOC=on GLM53_CAP0_LOC_TOKENS=200 GLM53_P4_DIR=/tmp/cap0test
     CAP0_TEST_OUT (optional: copy the synthetic loc/capture files there for the analysis format test)

  C1  capture through p4_hooks._s0_batch on a served-layout fake KV cache (rank 0: MLA 3/7, indexer 3/7, KDA
      0/2/4 co-owning the layer-3 MLA storage): every slice == the bytes at the independently computed offsets;
      view copy == raw copy; device two-sum == host two-sum; pools 511/1022 split out; npz sha == log record
  C2  capture fires once, at the CAP request's first (prefilling) step only; never for other tags
  C3  rank-1 layout (no MLA 3/7 on this rank): KDA slices only, MLA listed as missing, capture ok
  N1  wrong recurrent offset in the view -> layout_ok False -> capture not ok (refuses to call it valid)
  N2  kernel pages of a block not b*68+k -> kernel_pages_contiguous False
  N3  n_computed not a multiple of 4352 -> capture not ok, nothing written
  L1  localisation on a toy model with the real hook installation: every boundary's per-row sha256 == host
      sha256 of the toy's own trace; identities hold; state_end col / sha == the column written
  L2  a second replicate perturbed at one token of the layer-3 MoE output: points before it equal the
      reference, L3.moe_out differs at exactly that token, raw cur/ref rows saved and correct; the analysis
      loader's loc_first() names L3.moe_out at that position (writer and analysis agree on the file format)
  L3  inert: not armed / capturing / graph mode not "none" -> no record; "-N" warm, 2-row batch, <= 64 tokens
      -> not armed
  P1  p4_hooks call sites: the 21 marked lines are where the protocol says (AST), and the module minus them is
      the stage-0 module
"""
import ast
import glob
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import types

os.environ.setdefault("GLM53_P4", "on")
os.environ.setdefault("GLM53_P4_S0", "on")
os.environ.setdefault("GLM53_CAP0", "on")
os.environ.setdefault("GLM53_CAP0_LOC", "on")
os.environ.setdefault("GLM53_CAP0_LOC_TOKENS", "200")
os.environ.setdefault("GLM53_P4_DIR", "/tmp/cap0test")
shutil.rmtree(os.environ["GLM53_P4_DIR"], ignore_errors=True)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from vllm import p4_hooks as p4  # noqa: E402

C = p4._cap0
LY = C.LY if C is not None else None
N = {"PASS": 0, "FAIL": 0}
P = 2297856
R = 16


def check(name, ok, detail=""):
    N["PASS" if ok else "FAIL"] += 1
    print(("PASS " if ok else "FAIL ") + name + ("  " + str(detail)[:400] if detail and not ok else ""), flush=True)


def logs():
    out = []
    for p in glob.glob(os.path.join(p4.LOG_DIR, "cap0_r*_p*.jsonl")):
        out += [json.loads(l) for l in open(p)]
    return out


def clear_logs():
    for fh in p4.S.fh.values():
        fh.close()
    p4.S.fh.clear()
    shutil.rmtree(p4.LOG_DIR, ignore_errors=True)


def sha(b):
    return hashlib.sha256(memoryview(np.ascontiguousarray(b)).cast("B")).hexdigest()


# ------------------------------------------------------------------ served-layout fake KV cache
NBLK = 12
G0 = [5, 2, 7, 1]          # MLA/indexer group: logical 0..3
KDA = {0: (2, [0, 0, 3, 8]), 2: (4, [0, 0, 4, 9]), 4: (5, [0, 0, 6, 10])}   # layer -> (group, row)
PFX = "language_model.model.layers."


def make_kv(rank=0, bad_rec_offset=False, seed=7):
    g = torch.Generator().manual_seed(seed)
    A = torch.randint(0, 256, (NBLK * P,), generator=g, dtype=torch.uint8)       # MLA L3 + KDA 0/2/4 co-owners
    B = torch.randint(0, 256, (NBLK * P,), generator=g, dtype=torch.uint8)       # MLA L7
    I3 = torch.randint(0, 256, (NBLK * 143616,), generator=g, dtype=torch.uint8)
    I7 = torch.randint(0, 256, (NBLK * 143616,), generator=g, dtype=torch.uint8)
    conv = torch.as_strided(A.view(torch.bfloat16), (NBLK, 6, 12288), (P // 2, 12288, 1), 0)
    rec = torch.as_strided(A.view(torch.float32), (NBLK, 32, 128, 128), (P // 4, 16384, 128, 1),
                           0 if bad_rec_offset else 147456 // 4)
    ctx, groups = {}, [[] for _ in range(6)]
    if rank == 0:
        for L, raw in ((3, A), (7, B)):
            ctx[f"{PFX}{L}.self_attn.attn"] = types.SimpleNamespace(kv_cache=[raw.view(NBLK * 68, 64, 528)])
            groups[0].append(f"{PFX}{L}.self_attn.attn")
        for L, raw in ((3, I3), (7, I7)):
            ctx[f"{PFX}{L}.self_attn.indexer.k_cache"] = types.SimpleNamespace(kv_cache=[raw.view(NBLK * 17, 64, 132)])
            groups[0].append(f"{PFX}{L}.self_attn.indexer.k_cache")
    for L, (gi, _row) in KDA.items():
        ctx[f"{PFX}{L}.self_attn"] = types.SimpleNamespace(kv_cache=(conv, rec))
        groups[gi].append(f"{PFX}{L}.self_attn")
    return dict(A=A, B=B, I3=I3, I7=I7, conv=conv, rec=rec, ctx=ctx, groups=groups)


def make_runner(kv, req_idx=3, g0=None, model=None, prompt=None):
    g0 = list(G0 if g0 is None else g0)
    bpk = [68, 1, 1, 1, 1, 1]
    rows = [np.array([b * 68 + k for b in g0 for k in range(68)], dtype=np.int32)] + [np.zeros(0, np.int32)] * 5
    for L, (gi, row) in KDA.items():
        rows[gi] = np.array(row, dtype=np.int32)
    width = max(len(r) for r in rows)
    bts = []
    nbk = np.zeros((6, R), dtype=np.int32)
    for gi, r in enumerate(rows):
        t = torch.zeros(R, width, dtype=torch.int32)
        t[req_idx, :len(r)] = torch.from_numpy(r)
        nbk[gi, req_idx] = len(r)
        bts.append(types.SimpleNamespace(gpu=t))
    bt = types.SimpleNamespace(num_kv_cache_groups=6, blocks_per_kv_block=bpk, block_tables=bts,
                               num_blocks=types.SimpleNamespace(np=nbk), block_sizes=[4352] * 6)
    kvc = types.SimpleNamespace(num_blocks=NBLK, kv_cache_groups=[types.SimpleNamespace(layer_names=n) for n in kv["groups"]])
    ids = torch.zeros(R, 20000, dtype=torch.int32)
    ids[req_idx, :17409] = torch.from_numpy(np.asarray(prompt if prompt is not None else np.arange(17409) % 1000, dtype=np.int32))
    return types.SimpleNamespace(kv_cache_config=kvc, compilation_config=types.SimpleNamespace(static_forward_context=kv["ctx"]),
                                 block_tables=bt, device=torch.device("cpu"), model=model or nn.Module(),
                                 req_states=types.SimpleNamespace(all_token_ids=types.SimpleNamespace(gpu=ids)))


def fake_ib(nt, req_idx=3, n_comp=13056, pf=True, tag_rows=1, ids=None, pos0=0):
    nr = tag_rows
    return types.SimpleNamespace(
        num_reqs=nr, num_reqs_after_padding=nr, num_tokens=nt, num_tokens_after_padding=nt, num_draft_tokens=0,
        logits_indices=torch.tensor([nt - 1]), cu_num_logits_np=np.array([0, 1] + [1] * nr), req_ids=["r%d" % i for i in range(nr)],
        idx_mapping_np=np.array([req_idx] + [req_idx + 1 + i for i in range(nr - 1)]),
        num_scheduled_tokens=np.array([nt] * nr), num_computed_tokens_np=np.array([n_comp] * nr),
        num_computed_prefill_tokens_np=np.array([n_comp] * nr), prefill_len_np=np.array([17409] * nr),
        is_prefilling_np=np.array([pf] * nr), num_draft_tokens_per_req=np.array([0] * nr),
        input_ids=torch.as_tensor(ids if ids is not None else np.arange(nt) % 977, dtype=torch.int32),
        positions=torch.arange(pos0, pos0 + nt, dtype=torch.int64))


def reset(rank=0):
    C.S.__init__()
    p4.S.rank = rank
    p4.S.registered = True
    p4.S.serving = True
    p4.S.mode = "none"


def add(req_idx, tag):
    p4.S.uid_ctr += 1
    p4.S.uid_host[req_idx] = p4.S.uid_ctr
    p4.S.tag_of[req_idx] = tag
    return p4.S.uid_ctr


def expected_slices(kv, rank=0):
    """Independent arithmetic (not the layout module): the bytes each slice must hold."""
    A, B, I3, I7 = (kv[k].numpy() for k in ("A", "B", "I3", "I7"))
    out = {}
    if rank == 0:
        for L, raw in ((3, A), (7, B)):
            for j in range(3):
                out[f"mla{L}_b{j}"] = raw[G0[j] * P:(G0[j] + 1) * P]
        for L, raw in ((3, I3), (7, I7)):
            for j in range(3):
                out[f"idx{L}_b{j}"] = raw[G0[j] * 143616:(G0[j] + 1) * 143616]
    for L in (0, 4):
        b = KDA[L][1][2]
        out[f"kda{L}_conv_live"] = A[b * P:b * P + 3 * 24576]
        out[f"kda{L}_conv_inactive"] = A[b * P + 3 * 24576:b * P + 6 * 24576]
    for L in (2, 4):
        b = KDA[L][1][2]
        out[f"kda{L}_rec"] = A[b * P + 147456:b * P + 147456 + 2097152]
    return out


def test_capture():
    if C is None:
        check("C0 cap0_hooks imported with GLM53_CAP0=on", False, p4._CAP0_ERR)
        return
    reset(0)
    kv = make_kv(0)
    runner = make_runner(kv)
    C.S.conv_patched = True                      # no toy model here: skip the real kda patch
    p4.S.runner = runner
    C.register(runner)
    init = [r for r in logs() if r["kind"] == "cap0_init"]
    check("C1 init: 7 targets resolved, all layout_ok", bool(init) and len(init[-1]["targets"]) == 7 and
          all(t["ok"] for t in init[-1]["targets"].values()), init[-1]["targets"] if init else None)
    add(3, "CAP-r1")
    p4._s0_batch(fake_ib(2045), 101)
    caps = [r for r in logs() if r["kind"] == "cap0_capture"]
    check("C1 one capture record, ok", len(caps) == 1 and caps[0]["ok"], caps[-1] if caps else None)
    if not caps:
        return
    c = caps[0]
    f = os.path.join(p4.PRIV_DIR, "cap0_CAP-r1_r0.npz")
    check("C1 npz written 0600 with the logged sha256", os.path.exists(f) and oct(os.stat(f).st_mode & 0o777) == "0o600"
          and sha(np.fromfile(f, dtype=np.uint8)) == c["file_sha256"])
    z = np.load(f)
    exp = expected_slices(kv, 0)
    bad = [k for k, v in exp.items() if k not in z.files or not np.array_equal(z[k], v)]
    check("C1 every slice == the bytes at the independently computed offsets (%d slices)" % len(exp), not bad, bad)
    check("C1 view copy == raw copy for every slice", all(d["view_equal"] for d in c["slices"].values()))
    ts = [(k, d) for k, d in c["slices"].items() if "twosum_device" in d]
    check("C1 device two-sum == host two-sum (%d slices)" % len(ts), bool(ts) and all(d["twosum_device"] == d["twosum_host"] for _k, d in ts))
    pr = exp["idx7_b0"]
    p511 = np.concatenate([pr[67200:67328], pr[67580:67584]])
    check("C1 indexer pool 511 split out (values + scale, 132 B)", "idx7_b0_pool511" in z.files and np.array_equal(z["idx7_b0_pool511"], p511))
    check("C1 pools 511 / 1022 listed for blocks 0-2", all(f"pool{q}_sha256" in c["slices"][f"idx{L}_b{j}"]
                                                          for L in (3, 7) for j in range(3) for q in (511, 1022)))
    dst = c["slices"].get("kda4_rec_dst", {})
    check("C1 pre-copy destination column 3 = block 10 hashed, not stored", dst.get("b") == 10 and "kda4_rec_dst" not in z.files)
    check("C1 physical ids recorded (MLA 7 block 0 = 5, KDA 4 col 2 = 6)", c["slices"]["mla7_b0"]["b"] == 5 and c["slices"]["kda4_rec"]["b"] == 6)
    check("C1 engine-side prompt ids hashed (cached span)", c.get("cached_ids_sha256") == sha((np.arange(17409) % 1000).astype(np.int32)[:13056]))
    # C2: once
    p4._s0_batch(fake_ib(2045, n_comp=15101), 102)
    add(5, "W-other")
    p4._s0_batch(fake_ib(2045, req_idx=5), 103)
    check("C2 no second capture for the same request; none for other tags",
          len([r for r in logs() if r["kind"] == "cap0_capture"]) == 1)


def test_rank1_and_negatives():
    reset(1)
    kv = make_kv(1)
    runner = make_runner(kv)
    C.S.conv_patched = True
    p4.S.runner = runner
    clear_logs()
    C.register(runner)
    init = [r for r in logs() if r["kind"] == "cap0_init"][-1]
    check("C3 rank 1: KDA targets only, MLA/indexer 3/7 listed missing", set(init["targets"]) == {"kda0", "kda2", "kda4"}
          and len(init["missing"]) == 4)
    add(3, "CAP-r2")
    p4._s0_batch(fake_ib(2045), 201)
    c = [r for r in logs() if r["kind"] == "cap0_capture"]
    z = np.load(os.path.join(p4.PRIV_DIR, "cap0_CAP-r2_r1.npz")) if c else None
    exp = expected_slices(kv, 1)
    check("C3 rank-1 capture ok and bytes correct", bool(c) and c[0]["ok"] and all(np.array_equal(z[k], v) for k, v in exp.items()))
    # N1 wrong recurrent offset
    reset(0)
    clear_logs()
    kv = make_kv(0, bad_rec_offset=True)
    runner = make_runner(kv)
    C.S.conv_patched = True
    p4.S.runner = runner
    C.register(runner)
    init = [r for r in logs() if r["kind"] == "cap0_init"][-1]
    add(3, "CAP-r3")
    p4._s0_batch(fake_ib(2045), 301)
    c = [r for r in logs() if r["kind"] == "cap0_capture"]
    check("N1 recurrent view at the wrong offset -> target not layout_ok, capture not ok",
          not init["targets"]["kda4"]["ok"] and bool(c) and not c[0]["ok"] and not c[0]["slices"]["kda4_rec"]["view_equal"])
    # N2 kernel pages not contiguous
    reset(0)
    clear_logs()
    kv = make_kv(0)
    runner = make_runner(kv)
    t = runner.block_tables.block_tables[0].gpu
    t[3, 5], t[3, 6] = int(t[3, 6]), int(t[3, 5])
    C.S.conv_patched = True
    p4.S.runner = runner
    C.register(runner)
    add(3, "CAP-r4")
    p4._s0_batch(fake_ib(2045), 401)
    c = [r for r in logs() if r["kind"] == "cap0_capture"]
    check("N2 swapped kernel pages -> kernel_pages_contiguous False for block 0",
          bool(c) and c[0]["slices"]["mla7_b0"]["kernel_pages_contiguous"] is False and
          c[0]["slices"]["mla7_b1"]["kernel_pages_contiguous"] is True)
    # N3 n_computed not a block multiple
    add(3, "CAP-r5")
    p4._s0_batch(fake_ib(2045, n_comp=13000), 402)
    c = [r for r in logs() if r["kind"] == "cap0_capture" and r["tag"] == "CAP-r5"]
    check("N3 n_computed 13,000 -> capture not ok, nothing written", bool(c) and not c[0]["ok"]
          and not os.path.exists(os.path.join(p4.PRIV_DIR, "cap0_CAP-r5_r0.npz")))


# ------------------------------------------------------------------ toy model for the localisation hooks
H, NS, W = 64, 4, 32
TRACE = {}
KMOD = types.SimpleNamespace()
PERTURB = {}


def _conv(x, *a, **k):
    return torch.nn.functional.silu(x.float()).to(x.dtype)


KMOD.causal_conv1d_fn = _conv


class FusedOp(nn.Module):
    def forward(self, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, **kw):
        r = (residual.float() + x.float().unsqueeze(1) * post_layer_mix + comb_res_mix.sum(-1, keepdim=True)).to(residual.dtype)
        y = (r.float().mean(1) * fn[0, :H]).to(x.dtype)
        return r, torch.sigmoid(y.float()[:, :NS]).unsqueeze(-1), torch.softmax(y.float()[:, :NS * NS].view(-1, NS, NS), -1), y


class Indexer(nn.Module):
    def __init__(self):
        super().__init__()
        self.buf = torch.full((4096, W), -1, dtype=torch.int32)

    def forward(self, h, qr, positions, rope):
        m = h.shape[0]
        self.buf[:m] = torch.argsort(h.float()[:, :W], dim=1).to(torch.int32)
        return self.buf


class MLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.indexer = Indexer()
        self.w = nn.Parameter(torch.randn(H, H, generator=torch.Generator().manual_seed(3)) / 8)

    def forward(self, hidden_states, positions):
        self.indexer(hidden_states, None, positions, None)
        return (hidden_states.float() @ self.w).to(hidden_states.dtype)


class MoE(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L

    def forward(self, x):
        out = (x.float() * 1.5).to(x.dtype)
        if self.L == 3:
            if PERTURB.get("on"):
                out[PERTURB["row"], 0] = out[PERTURB["row"], 0] + 1.0
            TRACE["L3.moe_out.x"] = out.clone()
        return out


class ONorm(nn.Module):
    def forward(self, o, g):
        return o * 2


class KDAMod(nn.Module):
    def __init__(self, L, conv, rec, col):
        super().__init__()
        self.prefix, self.kv_cache, self.L, self.col = f"{PFX}{L}.self_attn", (conv, rec), L, col
        self.o_norm = ONorm()

    def forward(self, hidden_states, positions):
        m = hidden_states.shape[0]
        qkv = hidden_states.repeat(1, 3)
        out = KMOD.causal_conv1d_fn(qkv.transpose(0, 1)).transpose(0, 1)
        core = out[:, :H].reshape(1, m, 4, H // 4)
        conv, rec = self.kv_cache
        rec[self.col].copy_(rec[self.col] * 0.5 + float(out.float().mean()))
        conv[self.col, :3] = out[-1, :1].to(torch.bfloat16)
        if self.L == 4:
            TRACE.update({"L4.conv_out.qkv": out.clone(), "L4.rec_out.o": core.clone(),
                          "L4.state_end": (rec[self.col].clone(), conv[self.col, :3].clone())})
        o = self.o_norm(core, None)
        res = o.reshape(m, -1)
        if self.L == 4:
            TRACE["L4.attn_out.x"] = res.clone()
        return res


class Glm5NextDecoderLayer(nn.Module):
    def __init__(self, L, attn, mlp):
        super().__init__()
        self.layer_idx, self.self_attn, self.mlp = L, attn, mlp
        self.mhc_fused_post_pre_op = FusedOp()
        g = torch.Generator().manual_seed(10 + L)
        self.hc_attn_fn = nn.Parameter(torch.rand(2, H, generator=g) + 0.5)
        self.hc_ffn_fn = nn.Parameter(torch.rand(2, H, generator=g) + 0.5)
        self.sc = nn.Parameter(torch.ones(3))

    def forward(self, positions, hidden_states, residual, post, comb):
        if self.layer_idx in (3, 4):
            TRACE.update({f"L{self.layer_idx}.in.{k}": v.clone() for k, v in
                          zip(("x", "residual", "post", "comb"), (hidden_states, residual, post, comb))})
        residual, post, comb, x = self.mhc_fused_post_pre_op(x=hidden_states, residual=residual, post_layer_mix=post,
                                                             comb_res_mix=comb, fn=self.hc_attn_fn, hc_scale=self.sc, hc_base=self.sc)
        if self.layer_idx in (3, 4):
            TRACE.update({f"L{self.layer_idx}.pre_attn.{k}": v.clone() for k, v in
                          zip(("residual", "post", "comb", "y"), (residual, post, comb, x))})
        x = self.self_attn(hidden_states=x, positions=positions)
        if self.layer_idx == 3:
            TRACE["L3.attn_out.x"] = x.clone()
            TRACE["L3.topk.ids"] = self.self_attn.indexer.buf[: x.shape[0]].clone()
        residual, post, comb, x = self.mhc_fused_post_pre_op(x=x, residual=residual, post_layer_mix=post, comb_res_mix=comb,
                                                             fn=self.hc_ffn_fn, hc_scale=self.sc, hc_base=self.sc)
        if self.layer_idx == 3:
            TRACE.update({f"L3.pre_ffn.{k}": v.clone() for k, v in zip(("residual", "post", "comb", "y"), (residual, post, comb, x))})
        x = self.mlp(x)
        return x, residual, post, comb


class Toy(nn.Module):
    def __init__(self, kv):
        super().__init__()
        conv, rec = kv["conv"], kv["rec"]
        self.layers = nn.ModuleList([Glm5NextDecoderLayer(2, KDAMod(2, conv, rec, 4), MoE(2)),
                                     Glm5NextDecoderLayer(3, MLA(), MoE(3)),
                                     Glm5NextDecoderLayer(4, KDAMod(4, conv, rec, 6), MoE(4))])

    def forward(self, x, residual, post, comb):
        pos = torch.arange(x.shape[0])
        for layer in self.layers:
            x, residual, post, comb = layer(pos, x, residual, post, comb)
        return x


def toy_inputs(m, pos0):
    g = torch.Generator().manual_seed(1000 + pos0)
    return (torch.randn(m, H, generator=g).to(torch.bfloat16), torch.randn(m, NS, H, generator=g).to(torch.bfloat16),
            torch.rand(m, NS, 1, generator=g), torch.rand(m, NS, NS, generator=g))


def run_warm(model, tag, req_idx, step0, m=100, steps=2, perturb_row=None):
    uid = add(req_idx, tag)
    for k in range(steps):
        pos0 = k * m
        PERTURB.clear()
        if perturb_row is not None and pos0 <= perturb_row < pos0 + m:
            PERTURB.update(on=True, row=perturb_row - pos0)
        p4.S.mode = "none"
        p4._s0_batch(fake_ib(m, req_idx=req_idx, n_comp=pos0, ids=(np.arange(m) + pos0) % 977, pos0=pos0), step0 + k)
        TRACE.clear()
        with torch.no_grad():
            model(*toy_inputs(m, pos0))
        TRACES[(tag, k)] = dict(TRACE)
        C.after_forward()
    C.on_remove(req_idx, tag, uid)
    return uid


TRACES = {}


def test_loc():
    import vllm.forward_context as fc
    reset(0)
    clear_logs()
    kv = make_kv(0)
    model = Toy(kv)
    runner = make_runner(kv, model=model)
    p4.S.runner = runner
    C.patch_conv(KMOD)
    fc.get_forward_context = lambda: types.SimpleNamespace(attn_metadata={
        f"{PFX}2.self_attn": types.SimpleNamespace(non_spec_state_indices_tensor=torch.tensor([4])),
        f"{PFX}4.self_attn": types.SimpleNamespace(non_spec_state_indices_tensor=torch.tensor([6]))})
    C.register(runner)
    init = [r for r in logs() if r["kind"] == "cap0_init"][-1]
    check("L1 hooks installed on the toy (12) and the conv wrapper", len(init["hooks"]) == 12 and init["conv_patched"], init["hooks"])
    run_warm(model, "CW-r1-L", 3, 1000)
    f = os.path.join(p4.PRIV_DIR, "loc_CW-r1-L_r0.npz")
    ok = os.path.exists(f)
    check("L1 loc file written", ok)
    if not ok:
        return
    z = np.load(f)
    meta = json.loads(str(z["meta"]))
    bad = []
    for point, tens, _d in LY.LOC_ORDER:
        for t in tens:
            src = {"L3.attn_in.x": "L3.pre_attn.y", "L4.in.x": "L3.moe_out.x"}.get(f"{point}.{t}", f"{point}.{t}")
            want = []
            for k in range(2):
                v = TRACES[("CW-r1-L", k)].get(src)
                if v is None:
                    bad.append("trace " + src)
                    break
                v = v[0] if v.dim() >= 3 and v.shape[0] == 1 else v
                v = v.reshape(v.shape[0], -1).contiguous()
                want += [hashlib.sha256(v[i].view(torch.uint8).numpy().tobytes()).digest() for i in range(v.shape[0])]
            got = z.get(f"dig:{point}.{t}")
            if got is None or [bytes(r) for r in got] != want:
                bad.append(f"{point}.{t}")
    check("L1 every boundary's per-row sha256 == host sha256 of the toy's trace (%d tensors)" %
          sum(len(t) for _p, t, _d in LY.LOC_ORDER), not bad, bad)
    check("L1 identities hold (L3.pre_attn.y == L3.attn_in.x, L4.in == L3 outputs)",
          all(np.array_equal(z[f"dig:{a}.{ta}"], z[f"dig:{b}.{tb}"]) for (a, ta), (b, tb) in LY.LOC_IDENTITIES))
    st = [d for d in meta["states"] if d["point"] == "L4.state_end"]
    rb, cb = TRACES[("CW-r1-L", 1)]["L4.state_end"]
    check("L1 L4.state_end: column 6, sha of the column written in the last step",
          len(st) == 2 and st[-1]["col"] == 6 and st[-1]["rec_sha256"] == sha(rb.view(torch.uint8).numpy()) and
          st[-1]["conv_live_sha256"] == sha(cb.contiguous().view(torch.uint8).numpy()))
    check("L1 raw rows kept for positions < RAW_FIRST and the boundary", list(z["rawpos:L3.moe_out.x"][:3]) == [0, 1, 2]
          and 199 in list(z["rawpos:L3.moe_out.x"]))
    check("L1 steps recorded eager, engine-side ids hashed", len(meta["steps"]) == 2 and all(s["mode"] == "none" and s["ids_sha256"] for s in meta["steps"]))
    # L2: perturbed replicate
    model2 = model
    run_warm(model2, "CW-r2-L", 4, 2000, perturb_row=137)
    z2 = np.load(os.path.join(p4.PRIV_DIR, "loc_CW-r2-L_r0.npz"))
    m2 = json.loads(str(z2["meta"]))
    nd = {k: v["ndiff_vs_ref"] for k, v in m2["points"].items()}
    before = [f"{p}.{t}" for p, tens, _d in LY.LOC_ORDER[:6] for t in tens]
    check("L2 boundaries before the MoE equal the reference", all(nd.get(k) == 0 for k in before), nd)
    check("L2 L3.moe_out differs at exactly one token", nd.get("L3.moe_out.x") == 1 and list(z2["diffpos:L3.moe_out.x"]) == [137], nd)
    cur, ref = z2["diffcur:L3.moe_out.x"][0], z2["diffref:L3.moe_out.x"][0]
    t1 = TRACES[("CW-r1-L", 1)]["L3.moe_out.x"][37].view(torch.uint8).numpy()
    t2 = TRACES[("CW-r2-L", 1)]["L3.moe_out.x"][37].view(torch.uint8).numpy()
    check("L2 raw cur/ref rows of the differing token == the two traces", np.array_equal(cur, t2) and np.array_equal(ref, t1))
    spec = importlib.util.spec_from_file_location("A", os.environ.get("CAP0_ANALYSIS", "/c/analysis/analyze_cap0.py"))
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    da = {0: A.load_npz(f)}
    db = {0: A.load_npz(os.path.join(p4.PRIV_DIR, "loc_CW-r2-L_r0.npz"))}
    first = A.loc_first(da, db, (0,))
    check("L2 analysis loc_first() on the hook's own files: L3.moe_out at position 137",
          first is not None and first["point"] == "L3.moe_out" and first["hits"][0]["first_pos"] == 137, first)
    mag = A.raw_magnitude(db[0][0], db[0][1], "L3.moe_out", "x")
    check("L2 analysis raw_magnitude reads the saved rows (|diff| ~ 1)", mag is not None and 0.5 < mag["max_abs"] < 2.0, mag)
    out = os.environ.get("CAP0_TEST_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        for p in glob.glob(os.path.join(p4.PRIV_DIR, "*.npz")):
            shutil.copy(p, out)
    # L3 inert
    n0 = len(C.S.loc)
    C.S.arm = None
    with torch.no_grad():
        model(*toy_inputs(80, 0))
    check("L3 not armed -> no record", len(C.S.loc) == n0)
    uid = add(5, "CW-r3-L")
    p4._s0_batch(fake_ib(100, req_idx=5, n_comp=0), 3000)
    real = C._capturing
    C._capturing = lambda: True
    with torch.no_grad():
        model(*toy_inputs(100, 0))
    C._capturing = real
    p4.S.mode = "FULL"
    with torch.no_grad():
        model(*toy_inputs(100, 0))
    p4.S.mode = "none"
    check("L3 capturing / graph mode FULL -> no rows recorded", not C.S.loc.get(uid, {}).get("dig"))
    C.after_forward()
    add(6, "CW-r4-N")
    p4._s0_batch(fake_ib(100, req_idx=6, n_comp=0), 3001)
    a1 = C.S.arm
    add(7, "CW-r5-L")
    p4._s0_batch(fake_ib(100, req_idx=7, n_comp=0, tag_rows=2), 3002)
    a2 = C.S.arm
    add(8, "CW-r6-L")
    p4._s0_batch(fake_ib(60, req_idx=8, n_comp=0), 3003)
    a3 = C.S.arm
    check("L3 '-N' warm, 2-row batch, 60-token step -> not armed", a1 is None and a2 is None and a3 is None)
    check("L3 the 60-token step is logged as skipped", any(r["kind"] == "cap0_loc_skip" for r in logs()))
    check("L3 no p4/cap0 errors in the whole run", C.S.errors == 0 and p4.S.errors == 0, (C.S.errors, p4.S.errors))


def test_callsites():
    src = open(p4.__file__).read()
    tree = ast.parse(src)
    where = {}
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                        isinstance(node.func.value, ast.Name) and node.func.value.id == "_cap0":
                    where.setdefault(node.func.attr, []).append(fn.name)
    check("P1 cap0 calls: register in register, on_batch in _s0_batch, after_forward in on_logits, on_remove in on_remove",
          where == {"register": ["register"], "on_batch": ["_s0_batch"], "after_forward": ["on_logits"], "on_remove": ["on_remove"]}, where)
    fl = [i for i, l in enumerate(src.splitlines()) if l.strip().startswith("flush_steps()") and l.endswith("  # local-cap0")]
    guard = src.splitlines()[fl[0] - 1] if fl else ""
    check("P1 the cap0 step/sched flush in on_remove is rank-0 only and guarded by flush_allowed()",
          len(fl) == 1 and "_rank() == 0" in guard and "flush_allowed()" in guard and "_cap0 is not None" in guard, guard)
    lines = src.splitlines(True)
    marked = [l for l in lines if l.rstrip("\n").endswith("  # local-cap0")]
    ref = os.environ.get("CAP0_S0_P4", "/s0p4/p4_hooks.py")
    if os.path.exists(ref):
        check("P1 p4_hooks minus its 21 marked lines == the stage-0 module", len(marked) == 21 and
              "".join(l for l in lines if l not in marked) == open(ref).read())
    else:
        check("P1 stage-0 module available for the strip comparison", False, ref)


if __name__ == "__main__":
    test_capture()
    test_rank1_and_negatives()
    test_loc()
    test_callsites()
    print("cap0 hook tests: %d PASS, %d FAIL" % (N["PASS"], N["FAIL"]))
    sys.exit(1 if N["FAIL"] else 0)
