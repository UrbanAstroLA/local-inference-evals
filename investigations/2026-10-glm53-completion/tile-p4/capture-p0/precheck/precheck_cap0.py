#!/usr/bin/env python3
"""capture-p0 single-GPU pre-check (PROTOCOL.md section 6).  Runs in the cap0 image on GPU0: no model, synthetic
tensors only, env GLM53_P4=on GLM53_P4_S0=on GLM53_CAP0=on GLM53_CAP0_LOC=on (set below).  P4_PRECHECK_DEV=cpu runs
the same checks on CPU (graph checks skipped) as a CPU test.

  K1  point-0 capture on a served-layout synthetic KV cache on the device (MLA 3/7 [nblk*68, 64, 528] uint8,
      indexer 3/7 [nblk*17, 64, 132], KDA 0/2/4 conv [nblk, 6, 12288] bf16 / recurrent [nblk, 32, 128, 128] fp32
      as strided views of the layer-3 MLA storage): every slice == the host copy of the bytes at the formula
      offsets; view == raw; device two-sum == host two-sum; capture ok
  K2  CUDA-graph capture of a toy model with every localisation hook and the conv wrapper installed, once
      unarmed and once with a (stale) arm set: the capture succeeds, records nothing, and replays equal the eager
      output bit for bit
  K3  an armed eager step at the served sizes (2,045 tokens; every boundary's served row width): per-row sha256
      == host recomputation on every row; the wall time per step is measured (the localisation cost input)
  K4  wall time of one full point-0 capture at the served slice set
  N1  negative control, last: a device-to-host copy inside a capture fails it (the property the hooks rely on)

usage: precheck_cap0.py OUT_JSON
"""
import hashlib
import json
import os
import sys
import time
import traceback
import types

OUT_JSON = sys.argv[1]
for k, v in (("GLM53_P4", "on"), ("GLM53_P4_S0", "on"), ("GLM53_CAP0", "on"), ("GLM53_CAP0_LOC", "on"),
             ("GLM53_P4_DIR", "/tmp/cap0pre")):
    os.environ.setdefault(k, v)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from vllm import p4_hooks as p4  # noqa: E402

C = p4._cap0
DEV = torch.device(os.environ.get("P4_PRECHECK_DEV", "cuda:0"))
GPU = DEV.type == "cuda"
P = 2297856
NBLK = 12
G0 = [5, 2, 7, 1]
KDA = {0: (2, [0, 0, 3, 8]), 2: (4, [0, 0, 4, 9]), 4: (5, [0, 0, 6, 10])}
PFX = "language_model.model.layers."
RESULTS = []
TIMES = {}


def result(name, ok, **kw):
    RESULTS.append(dict(test=name, ok=bool(ok), **kw))
    print(("PASS " if ok else "FAIL ") + name + "  " + json.dumps(kw, default=str)[:400], flush=True)


def sync():
    if GPU:
        torch.cuda.synchronize(DEV)


def kv_and_runner(model=None):
    g = torch.Generator().manual_seed(5)
    host = {k: torch.randint(0, 256, (n,), generator=g, dtype=torch.uint8)
            for k, n in (("A", NBLK * P), ("B", NBLK * P), ("I3", NBLK * 143616), ("I7", NBLK * 143616))}
    dev = {k: v.to(DEV) for k, v in host.items()}
    A = dev["A"]
    conv = torch.as_strided(A.view(torch.bfloat16), (NBLK, 6, 12288), (P // 2, 12288, 1), 0)
    rec = torch.as_strided(A.view(torch.float32), (NBLK, 32, 128, 128), (P // 4, 16384, 128, 1), 147456 // 4)
    ctx, groups = {}, [[] for _ in range(6)]
    for L, k in ((3, "A"), (7, "B")):
        ctx[f"{PFX}{L}.self_attn.attn"] = types.SimpleNamespace(kv_cache=[dev[k].view(NBLK * 68, 64, 528)])
        groups[0].append(f"{PFX}{L}.self_attn.attn")
    for L, k in ((3, "I3"), (7, "I7")):
        ctx[f"{PFX}{L}.self_attn.indexer.k_cache"] = types.SimpleNamespace(kv_cache=[dev[k].view(NBLK * 17, 64, 132)])
        groups[0].append(f"{PFX}{L}.self_attn.indexer.k_cache")
    for L, (gi, _r) in KDA.items():
        ctx[f"{PFX}{L}.self_attn"] = types.SimpleNamespace(kv_cache=(conv, rec))
        groups[gi].append(f"{PFX}{L}.self_attn")
    rows = [np.array([b * 68 + k for b in G0 for k in range(68)], dtype=np.int32)] + [np.zeros(0, np.int32)] * 5
    for L, (gi, row) in KDA.items():
        rows[gi] = np.array(row, dtype=np.int32)
    width = max(len(r) for r in rows)
    nbk = np.zeros((6, 16), dtype=np.int32)
    bts = []
    for gi, r in enumerate(rows):
        t = torch.zeros(16, width, dtype=torch.int32)
        t[3, :len(r)] = torch.from_numpy(r)
        nbk[gi, 3] = len(r)
        bts.append(types.SimpleNamespace(gpu=t.to(DEV)))
    ids = torch.zeros(16, 17409, dtype=torch.int32)
    ids[3] = torch.arange(17409, dtype=torch.int32) % 1000
    runner = types.SimpleNamespace(
        kv_cache_config=types.SimpleNamespace(num_blocks=NBLK, kv_cache_groups=[types.SimpleNamespace(layer_names=n) for n in groups]),
        compilation_config=types.SimpleNamespace(static_forward_context=ctx), device=DEV, model=model or nn.Module(),
        block_tables=types.SimpleNamespace(num_kv_cache_groups=6, blocks_per_kv_block=[68, 1, 1, 1, 1, 1], block_tables=bts,
                                           num_blocks=types.SimpleNamespace(np=nbk)),
        req_states=types.SimpleNamespace(all_token_ids=types.SimpleNamespace(gpu=ids.to(DEV))))
    return host, dict(conv=conv, rec=rec), runner


def row(tag, uid, n_sched=2045, n_comp=13056, pf=1):
    return ["r", 3, uid, tag, n_sched, n_comp, n_comp, 17409, pf, 0]


def k1_capture():
    C.S.__init__()
    p4.S.rank = 0
    p4.S.mode = "none"
    host, _kv, runner = kv_and_runner()
    C.S.conv_patched = True
    C.register(runner)
    t0 = time.time()
    C.on_batch(types.SimpleNamespace(num_tokens=2045, num_tokens_after_padding=2045), 7, [row("CAP-pre", 1)])
    sync()
    TIMES["capture_s"] = round(time.time() - t0, 4)
    rec = [json.loads(l) for l in open(_logfile()) if '"cap0_capture"' in l]
    z = np.load(os.path.join(p4.PRIV_DIR, "cap0_CAP-pre_r0.npz"))
    A, B, I3, I7 = (host[k].numpy() for k in ("A", "B", "I3", "I7"))
    exp = {}
    for L, raw in ((3, A), (7, B)):
        for j in range(3):
            exp[f"mla{L}_b{j}"] = raw[G0[j] * P:(G0[j] + 1) * P]
    for L, raw in ((3, I3), (7, I7)):
        for j in range(3):
            exp[f"idx{L}_b{j}"] = raw[G0[j] * 143616:(G0[j] + 1) * 143616]
    for L in (0, 4):
        b = KDA[L][1][2]
        exp[f"kda{L}_conv_live"] = A[b * P:b * P + 73728]
        exp[f"kda{L}_conv_inactive"] = A[b * P + 73728:b * P + 147456]
    for L in (2, 4):
        b = KDA[L][1][2]
        exp[f"kda{L}_rec"] = A[b * P + 147456:b * P + 147456 + 2097152]
    bad = [k for k, v in exp.items() if k not in z.files or not np.array_equal(z[k], v)]
    ok = bool(rec) and rec[-1]["ok"] and not bad and all(d["view_equal"] for d in rec[-1]["slices"].values()) and \
        all(d["twosum_device"] == d["twosum_host"] for d in rec[-1]["slices"].values() if "twosum_device" in d)
    result("K1 point-0 capture on a served-layout device KV cache: %d slices exact, view == raw, two-sums equal" % len(exp),
           ok, bad=bad, secs=TIMES["capture_s"])


def _logfile():
    import glob
    return sorted(glob.glob(os.path.join(p4.LOG_DIR, "cap0_r0_p*.jsonl")))[-1]


# ------------------------------------------------------------------ toy model (device)
H = 128
KM = types.SimpleNamespace(causal_conv1d_fn=lambda x, *a, **k: torch.nn.functional.silu(x))


class FusedOp(nn.Module):
    def forward(self, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, **kw):
        r = residual + (x.unsqueeze(1) * post_layer_mix.to(x.dtype))
        y = r.mean(1) * fn[0].to(x.dtype)
        return r, torch.sigmoid(y[:, :4].float()).unsqueeze(-1), torch.softmax(y[:, :16].float().view(-1, 4, 4), -1), y


class Indexer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("buf", torch.full((256, 32), -1, dtype=torch.int32))

    def forward(self, h, qr, positions, rope):
        self.buf[: h.shape[0]] = (h[:, :32].float() * 64).to(torch.int32)      # capture-safe stand-in for top-k
        return self.buf


class MLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.indexer = Indexer()
        self.lin = nn.Linear(H, H, bias=False)

    def forward(self, hidden_states, positions):
        self.indexer(hidden_states, None, positions, None)
        return self.lin(hidden_states)


class ONorm(nn.Module):
    def forward(self, o, g):
        return o * 2


class KDAMod(nn.Module):
    def __init__(self, L, kv, col):
        super().__init__()
        self.prefix, self.kv_cache, self.col = f"{PFX}{L}.self_attn", (kv["conv"], kv["rec"]), col
        self.o_norm = ONorm()

    def forward(self, hidden_states, positions):
        m = hidden_states.shape[0]
        out = KM.causal_conv1d_fn(hidden_states.repeat(1, 3).transpose(0, 1)).transpose(0, 1)
        self.kv_cache[1][self.col, 0, 0, :H].copy_(out[-1, :H].float())
        return self.o_norm(out[:, :H].reshape(1, m, 4, H // 4), None).reshape(m, -1)


class MLP(nn.Module):
    def forward(self, x):
        return x * 1.5


class Glm5NextDecoderLayer(nn.Module):
    def __init__(self, L, attn):
        super().__init__()
        self.layer_idx, self.self_attn, self.mlp = L, attn, MLP()
        self.mhc_fused_post_pre_op = FusedOp()
        self.hc_attn_fn = nn.Parameter(torch.rand(2, H) + 0.5)
        self.hc_ffn_fn = nn.Parameter(torch.rand(2, H) + 0.5)
        self.sc = nn.Parameter(torch.ones(3))

    def forward(self, positions, hidden_states, residual, post, comb):
        residual, post, comb, x = self.mhc_fused_post_pre_op(x=hidden_states, residual=residual, post_layer_mix=post,
                                                             comb_res_mix=comb, fn=self.hc_attn_fn, hc_scale=self.sc, hc_base=self.sc)
        x = self.self_attn(hidden_states=x, positions=positions)
        residual, post, comb, x = self.mhc_fused_post_pre_op(x=x, residual=residual, post_layer_mix=post, comb_res_mix=comb,
                                                             fn=self.hc_ffn_fn, hc_scale=self.sc, hc_base=self.sc)
        return self.mlp(x), residual, post, comb


class Toy(nn.Module):
    def __init__(self, kv):
        super().__init__()
        self.layers = nn.ModuleList([Glm5NextDecoderLayer(2, KDAMod(2, kv, 4)), Glm5NextDecoderLayer(3, MLA()),
                                     Glm5NextDecoderLayer(4, KDAMod(4, kv, 6))])

    def forward(self, x, residual, post, comb):
        pos = torch.arange(x.shape[0], device=x.device)
        for layer in self.layers:
            x, residual, post, comb = layer(pos, x, residual, post, comb)
        return x


def k2_graph():
    import vllm.forward_context as fc
    C.S.__init__()
    p4.S.mode = "none"
    _host, kv, _r = kv_and_runner()
    model = Toy(kv).to(DEV).to(torch.bfloat16)
    runner = kv_and_runner(model)[2]
    C.patch_conv(KM)
    fc.get_forward_context = lambda: types.SimpleNamespace(attn_metadata={
        f"{PFX}2.self_attn": types.SimpleNamespace(non_spec_state_indices_tensor=torch.tensor([4], device=DEV)),
        f"{PFX}4.self_attn": types.SimpleNamespace(non_spec_state_indices_tensor=torch.tensor([6], device=DEV))})
    C.register(runner)
    init = [json.loads(l) for l in open(_logfile()) if '"cap0_init"' in l][-1]
    result("K2 the toy carries the served call pattern: 12 hooks + the conv wrapper installed", len(init["hooks"]) == 12
           and init["conv_patched"], hooks=init["hooks"])
    m = 32
    g = torch.Generator(device=DEV).manual_seed(3)
    inp = [torch.randn(m, H, generator=g, device=DEV).to(torch.bfloat16), torch.randn(m, 4, H, generator=g, device=DEV).to(torch.bfloat16),
           torch.rand(m, 4, 1, generator=g, device=DEV), torch.rand(m, 4, 4, generator=g, device=DEV)]
    if not GPU:                                  # CPU: the same hooks, eager only (graph capture needs the GPU)
        for armed in (False, True):
            C.S.loc[999] = C._new_acc("CW-x-L")
            C.S.arm = dict(uid=999, tag="CW-x-L", step=0, pos0=0, n=m, m=m, t0=0, secs=0.0, ids_sha256=None,
                           pos_sha256=None, points=[]) if armed else None
            with torch.no_grad():
                model(*inp)
            C.S.arm = None
            pts = {"%s.%s" % k for k in C.S.loc[999]["dig"]}
            want = {f"{p}.{t}" for p, ts, _d in C.LY.LOC_ORDER for t in ts}
            if armed:
                result("K2 (CPU, eager) armed toy step: every boundary recorded, both state points", pts == want and
                       len(C.S.loc[999]["states"]) == 2, missing=sorted(want - pts))
            else:
                result("K2 (CPU, eager) unarmed toy step: nothing recorded", not pts and not C.S.loc[999]["states"])
        return
    oks = []
    for armed in (False, True):
        C.S.arm = dict(uid=999, tag="CW-x-L", step=0, pos0=0, n=m, m=m, t0=0, secs=0.0, ids_sha256=None,
                       pos_sha256=None, points=[]) if armed else None
        C.S.loc[999] = C._new_acc("CW-x-L")
        with torch.no_grad():
            s = torch.cuda.Stream(DEV)
            s.wait_stream(torch.cuda.current_stream(DEV))
            with torch.cuda.stream(s):
                for _ in range(2):
                    model(*inp)
            torch.cuda.current_stream(DEV).wait_stream(s)
            C.S.loc[999] = C._new_acc("CW-x-L")
            graph = torch.cuda.CUDAGraph()
            err = None
            try:
                with torch.cuda.graph(graph):
                    out = model(*inp)
            except Exception as exc:  # noqa: BLE001
                err = repr(exc)[:300]
            recorded = bool(C.S.loc[999]["dig"]) or bool(C.S.loc[999]["states"])
            C.S.arm = None
            if err is None:
                graph.replay()
                sync()
                ref = model(*inp)
                sync()
                same = bool(torch.equal(out, ref))
            else:
                same = False
        oks.append((err is None, not recorded, same))
        result("K2 graph capture with hooks installed (%s): captured, nothing recorded, replay == eager" %
               ("stale arm set" if armed else "unarmed"), err is None and not recorded and same, err=err,
               recorded=recorded, replay_equal=same)


def k3_timing():
    C.S.__init__()
    p4.S.mode = "none"
    C.S.registered = True
    C.S.p4 = p4
    m = 2045
    widths = {"x": (4096,), "residual": (4, 4096), "post": (4, 1), "comb": (4, 4), "y": (4096,), "ids": (2048,),
              "qkv": (12288,), "o": (32, 128)}
    g = torch.Generator(device=DEV).manual_seed(9)

    def mk(t):
        shp = (m,) + widths[t]
        if t == "ids":
            return torch.randint(0, 100000, shp, generator=g, device=DEV, dtype=torch.int32)
        dt = torch.float32 if t in ("post", "comb") else torch.bfloat16
        return torch.randn(shp, generator=g, device=DEV).to(dt)
    tens = {(p, t): mk(t) for p, ts, _d in C.LY.LOC_ORDER for t in ts}
    C.S.loc[1] = C._new_acc("CW-time-L")
    C.S.arm = dict(uid=1, tag="CW-time-L", step=1, pos0=0, n=m, m=m, t0=time.time(), secs=0.0, ids_sha256=None,
                   pos_sha256=None, points=[])
    C.S.ref_uid = 1
    sync()
    t0 = time.time()
    for p, ts, _d in C.LY.LOC_ORDER:
        C._rec(p, [(t, tens[(p, t)]) for t in ts])
    sync()
    secs = time.time() - t0
    nbytes = sum(v.numel() * v.element_size() for v in tens.values())
    bad = []
    for (p, t), v in tens.items():
        h = v.reshape(m, -1).contiguous().view(torch.uint8).cpu().numpy()
        dig = C.S.loc[1]["dig"][(p, t)][0]
        if any(hashlib.sha256(h[i]).digest() != bytes(dig[i]) for i in range(m)):
            bad.append(f"{p}.{t}")
    TIMES.update(loc_step_s=round(secs, 3), loc_step_bytes=nbytes, loc_bytes_per_token=nbytes // m,
                 loc_gbps=round(nbytes / secs / 1e9, 3), hold_mb=round(C.S.hold_bytes / 2 ** 20, 1))
    C.S.arm = None
    result("K3 armed eager step at served sizes (2,045 tokens, %d tensors): every row's sha256 == host" % len(tens),
           not bad, bad=bad, **{k: TIMES[k] for k in ("loc_step_s", "loc_bytes_per_token", "loc_gbps", "hold_mb")})


def n1_negative():
    if not GPU:
        result("N1 negative control: D2H inside a capture fails (skipped on CPU)", True, skipped=True)
        return
    x = torch.ones(8, device=DEV)
    g = torch.cuda.CUDAGraph()
    failed = False
    try:
        with torch.cuda.graph(g):
            y = x * 2
            y.cpu()
    except Exception:  # noqa: BLE001
        failed = True
    try:
        torch.cuda.synchronize(DEV)
    except Exception:  # noqa: BLE001
        pass
    result("N1 negative control: a device-to-host copy inside a capture fails it", failed)


def main():
    t0 = time.time()
    if C is None:
        result("K0 cap0_hooks imported", False, err=p4._CAP0_ERR)
    else:
        for f in (k1_capture, k2_graph, k3_timing, n1_negative):
            try:
                f()
            except Exception:  # noqa: BLE001
                result(f.__name__ + " raised", False, tb=traceback.format_exc()[-1500:])
    n_exp = 6 if GPU else 6
    ok = len(RESULTS) == n_exp and all(r["ok"] for r in RESULTS) and (C is not None and C.S.errors == 0) and p4.S.errors == 0
    doc = dict(verdict="PASS" if ok else "FAIL", device=str(DEV), gpu=GPU, results=RESULTS, times=TIMES,
               cap0_errors=None if C is None else C.S.errors, p4_errors=p4.S.errors, wall_s=round(time.time() - t0, 2))
    json.dump(doc, open(OUT_JSON, "w"), indent=1, default=str)
    print("VERDICT", doc["verdict"], json.dumps(TIMES))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
