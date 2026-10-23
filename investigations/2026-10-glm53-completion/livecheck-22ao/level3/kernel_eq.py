#!/usr/bin/env python3
"""Level 3 (22ap): does an inflated consumed count (nsa) with -1 holes change the stock sparse-MLA output?

Stock 0.9.1 base image, synthetic tensors, no model. Runs the stock b12x GLM_NEXT decode and extend kernels with the
served plan shape (64 heads, 64 rows, width 2048, 32 decode splits) on rows whose p + 1 valid slots are front-packed
and the rest -1, once with nsa = p + 1 (A) and once with nsa = min(p + 1 + r, 2048) (B), r = 1, 2, 3, and compares
the outputs bit for bit. cache_lengths follows the stock value (p + 1 for A, p + 1 + r for B).

  D0  determinism: A twice, bit-equal (decode, extend)
  E   A vs B bit-equal on every row, for r = 1, 2, 3, with slot 0 zero (served: null block) and with slot 0 real
  P   positive control: B with one live non-causal slot planted at column p + 1; must differ on every row
  N   probe (diagnostic, not a pass criterion): slot 0 = NaN bytes; NaN rows in A and B, and A vs B bit-equality

usage: kernel_eq.py OUT_JSON [--dry]   (--dry: CPU layout checks only, no kernels)
"""
import json
import os
import sys
import time

os.environ.setdefault("B12X_MLA_SM120_NUM_SPLITS", "32")   # vLLM pins 32 around plan prime (b12x_mla_sparse.py:203-215)
import torch  # noqa: E402

OUT = sys.argv[1]
DRY = "--dry" in sys.argv
PAGE, REC, HD, W, ROWS, HEADS, NPAGES = 64, 528, 512, 2048, 64, 64, 64
NREC = NPAGES * PAGE
SCALE = 256 ** -0.5
EDGES = [1, 2, 63, 64, 65, 127, 128, 129, 191, 192, 1023, 1024, 1025, 1983, 1984, 1985, 2043, 2044, 2045, 2046, 2047]
RES = dict(started=time.time(), dry=DRY, tests=[], info={})


def result(name, ok, **kw):
    RES["tests"].append(dict(test=name, ok=None if ok is None else bool(ok), **kw))
    tag = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    print(f"{tag} {name}  " + json.dumps(kw, default=str)[:400], flush=True)


# ---- layout (CPU, deterministic)
g = torch.Generator().manual_seed(22_0_3)
NR = ROWS - len(EDGES)
near = (torch.randint(1, W // 64, (NR // 2,), generator=g) * 64 - torch.randint(0, 3, (NR // 2,), generator=g)).tolist()
plen = EDGES + near + torch.randint(1, W, (NR - NR // 2,), generator=g).tolist()   # p + 1 per row, all < 2048;
# half the random rows sit 0-2 below a multiple of 64, so inflation by r = 1-3 crosses a chunk boundary
sel = torch.full((ROWS, W), -1, dtype=torch.int32)
spare = torch.empty(ROWS, dtype=torch.int32)
for t, n in enumerate(plen):
    perm = torch.randperm(NREC - 1, generator=g)[: n + 1] + 1          # distinct slots, never slot 0
    sel[t, :n] = perm[:n].to(torch.int32)
    spare[t] = int(perm[n])
sel_ctrl = sel.clone()
for t, n in enumerate(plen):
    sel_ctrl[t, n] = spare[t]                                            # live non-causal entry at column p + 1
nA = torch.tensor(plen, dtype=torch.int32)
nB = {r: torch.clamp(nA + r, max=W) for r in (1, 2, 3)}
cross = {r: int(((nB[r] + 63) // 64 > (nA + 63) // 64).sum()) for r in (1, 2, 3)}
lay_ok = all(len(set(sel[t, :n].tolist())) == n and 0 not in sel[t, :n].tolist() and int(spare[t]) not in sel[t, :n].tolist()
             and bool((sel[t, n:] == -1).all()) for t, n in enumerate(plen))
result("L0 layout: front-packed distinct non-zero slots, -1 after, spare outside the row", lay_ok,
       rows=ROWS, p1_min=min(plen), p1_max=max(plen), rows_crossing_64_boundary=cross)
if DRY:
    RES["ok"] = lay_ok
    json.dump(RES, open(OUT, "w"), indent=1)
    sys.exit(0 if lay_ok else 1)

from b12x.attention import sparse_mla  # noqa: E402
from b12x.attention._shared.mla import kernel as K  # noqa: E402
from b12x.attention._shared.mla.traits import ModelType  # noqa: E402

dev = torch.device("cuda:0")
RES["info"]["device"] = torch.cuda.get_device_name(dev)
RES["info"]["capability"] = torch.cuda.get_device_capability(dev)
gd = torch.Generator(device=dev).manual_seed(22_0_4)
latent = (torch.randn((NREC, HD), generator=gd, device=dev) / 4).to(torch.bfloat16)
q = (torch.randn((ROWS, HEADS, HD), generator=gd, device=dev) / 4).to(torch.bfloat16)
sel_d, ctrl_d = sel.to(dev), sel_ctrl.to(dev)
nA_d = nA.to(dev)
nB_d = {r: v.to(dev) for r, v in nB.items()}
LA_d = nA.to(dev)                                       # stock cache_lengths for A: p + 1
LB_d = {r: (nA + r).to(dev) for r in (1, 2, 3)}         # stock cache_lengths for B: p + 1 + r (unclamped)


def make_cache(slot0):
    cache = torch.zeros((NPAGES, PAGE, REC), dtype=torch.uint8, device=dev)   # vLLM allocates the KV cache zeroed
    lo = 0 if slot0 == "real" else 1                                          # slot 0 stays zero unless "real"
    kv, slots = latent[lo:].contiguous(), torch.arange(lo, NREC, dtype=torch.int64, device=dev)
    sparse_mla.concat_and_cache_glm_next_mla(kv, cache, slots,             # the stock GLM_NEXT writer, as vLLM
                                             plan=sparse_mla.plan_cache_writer(kv, cache, slots))
    if slot0 == "nan":
        cache[0, 0, :] = 0x7F
    torch.cuda.synchronize(dev)
    return cache


def make_plan(mode):
    return sparse_mla.plan(sparse_mla.Caps(            # as vLLM _make_plan (b12x_mla_sparse.py:1839-1880), fp8_ds_mla
        device=dev, num_q_heads=HEADS, max_q_rows=ROWS, max_width=W, softmax_scale=SCALE,
        dtype=torch.bfloat16, kv_dtype=torch.uint8, head_dim=HD, v_head_dim=HD,
        model_type=ModelType.GLM_NEXT, scale_format=None, cache_record_bytes=REC, fp8_rope=None,
        latent_scale_per_token=False, mode=mode, max_batch=ROWS, max_chunks_per_row=W // 64, page_size=PAGE,
        head_major_output=(mode == "decode"), return_lse=(mode != "decode"), lse_scale="natural"))


PLANS = {}


def run(mode, cache, selected, active, clen):
    if mode not in PLANS:
        t0 = time.time()
        PLANS[mode] = make_plan(mode)
        RES["info"][f"plan_{mode}_s"] = round(time.time() - t0, 1)
    plan = PLANS[mode]
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=dev)
    b = sparse_mla.bind(plan, scratch=scratch, q=q, kv_cache=cache, selected_indices=selected,
                        cache_lengths=clen, selected_lengths=active)
    r = sparse_mla.run(b)
    torch.cuda.synchronize(dev)
    if isinstance(r, tuple):
        return r[0].contiguous().clone(), r[1].contiguous().clone()
    return r.contiguous().clone(), None


def rows_diff(a, b):
    """Rows (dim 0) whose bits differ; outputs are bf16 [rows, heads, dim], LSE fp32 [rows, heads]."""
    ia = a.view(torch.int16 if a.dtype == torch.bfloat16 else torch.int32)
    ib = b.view(torch.int16 if b.dtype == torch.bfloat16 else torch.int32)
    return [int(i) for i in torch.nonzero((ia != ib).reshape(a.shape[0], -1).any(1)).flatten().tolist()]


def maxabs(a, b):
    d = (a.float() - b.float()).abs()
    return float(torch.nan_to_num(d, nan=float("inf")).max())


def nan_rows(a):
    return int(torch.isnan(a.float()).reshape(a.shape[0], -1).any(1).sum())


t_start = time.time()
with torch.inference_mode():
    caches = {v: make_cache(v) for v in ("zero", "real")}
    outA = {}
    for mode in ("decode", "extend"):
        a1 = run(mode, caches["zero"], sel_d, nA_d, LA_d)
        a2 = run(mode, caches["zero"], sel_d, nA_d, LA_d)
        d = rows_diff(a1[0], a2[0]) + (rows_diff(a1[1], a2[1]) if a1[1] is not None else [])
        result(f"D0 {mode}: A run twice is bit-equal", not d, rows_differing=len(d), out_shape=list(a1[0].shape))
        if mode == "decode":
            RES["info"]["last_decode_plan"] = {k: str(v) for k, v in dict(K.LAST_DECODE_PLAN).items()}
    for v in ("zero", "real"):
        for mode in ("decode", "extend"):
            a = run(mode, caches[v], sel_d, nA_d, LA_d)
            outA[(v, mode)] = a
            for r in (1, 2, 3):
                b = run(mode, caches[v], sel_d, nB_d[r], LB_d[r])
                d_out = rows_diff(a[0], b[0])
                d_lse = rows_diff(a[1], b[1]) if a[1] is not None else []
                result(f"E {mode} slot0={v} r={r}: A vs B bit-equal on every row", not d_out and not d_lse,
                       rows_out_differing=len(d_out), rows_lse_differing=len(d_lse), max_abs=maxabs(a[0], b[0]),
                       rows_crossing_64=cross[r], nan_rows_A=nan_rows(a[0]))
    for mode in ("decode", "extend"):
        a = outA[("zero", mode)]
        c = run(mode, caches["zero"], ctrl_d, nB_d[1], LB_d[1])
        d = rows_diff(a[0], c[0])
        result(f"P {mode}: planted live non-causal entry changes every row", len(d) == ROWS,
               rows_differing=len(d), of=ROWS, max_abs=maxabs(a[0], c[0]))
    cn = make_cache("nan")
    for mode in ("decode", "extend"):
        a = run(mode, cn, sel_d, nA_d, LA_d)
        b = run(mode, cn, sel_d, nB_d[1], LB_d[1])
        result(f"N {mode} slot0=NaN (diagnostic)", None, nan_rows_A=nan_rows(a[0]), nan_rows_B=nan_rows(b[0]),
               rows_bits_differing=len(rows_diff(a[0], b[0])))
RES["info"]["kernel_section_s"] = round(time.time() - t_start, 1)
RES["ok"] = all(x["ok"] for x in RES["tests"] if x["ok"] is not None)
RES["finished"] = time.time()
json.dump(RES, open(OUT, "w"), indent=1, default=str)
print("ALL PASS" if RES["ok"] else "SOME FAILED", flush=True)
sys.exit(0 if RES["ok"] else 1)
