"""CPU emulation (torch fp32, tile order as on GPU: BLOCK_SIZE 8192, BLOCK_SIZE_TRUNC 4096) of the STANDALONE TOP-P
path of the served Triton kernel (vllm/v1/sample/ops/topk_topp_triton.py lines 594-853; copied to engine-src/).
The Triton interpreter cannot run the kernel on CPU (row_id.to on a Python int), so the arithmetic is transcribed.
Question: on which near-one-hot rows does the kernel skip masking ('not (final_pivot < max_logit)')?"""
import json, math, torch
import vllm.v1.sample.ops.topk_topp_triton as T
SIG = torch.tensor(T._NORMAL_CDF_TO_SIGMA_TABLE, dtype=torch.float32)
BS, BT = 8192, 4096
def topp_row(x, p=0.95):
    x = x.float(); V = x.numel(); f32 = lambda v: torch.tensor(v, dtype=torch.float32)
    b0 = x[:BS]; fin = torch.isfinite(b0)
    avg = b0[fin].sum() / fin.sum(); sq = (b0[fin] * b0[fin]).sum() / fin.sum()
    std = torch.sqrt(torch.clamp(sq - avg * avg, min=0.0)); ms = avg + std * 10.0
    S = f32(0.0); mx = f32(-math.inf); mn = f32(math.inf)
    for i in range(0, V, BS):
        t = x[i:i + BS]; mx = torch.maximum(mx, t.max()); mn = torch.minimum(mn, t.min()); S = S + torch.exp(t - ms).sum()
    mn = torch.minimum(mn, mx)
    idx = max(0, min(int(p * 200), 199)); sg = SIG[idx]; sg = sg + abs(sg) * -0.25
    op = avg + std * sg; oprob = torch.exp(op - ms) / S
    probs = torch.exp(x - ms) / S
    om = probs > oprob; sop = probs[om].sum() if om.any() else f32(0.0); buf = probs[om]
    max_range = torch.exp(mx - ms) / S; min_range = torch.exp(mn - ms) / S
    p_pivot = f32(1.0); mlp = f32(1.0); nml = 0; pps = f32(0.0); it = 0; branch = "else"
    if bool(sop > p):
        branch = "search"; min_range = oprob
        while True:
            pv = (max_range - min_range) * 0.5 + min_range
            ab = buf > pv; ps = (buf * ab).sum()
            ml = buf[ab].min() if ab.any() else f32(1.0); nm = int((buf[ab] == ml).sum()) if ab.any() else 0
            if bool(ps >= p) and bool(ps - ml * nm < p):
                p_pivot, mlp, nml, pps = pv, ml, nm, ps; break
            if bool(ps > p): min_range = pv
            elif bool(ps < p): max_range = pv
            it += 1
            if bool((max_range - min_range) < 1e-9) or it >= 18:
                p_pivot, mlp, nml, pps = (max_range + min_range) / 2.0, ml, nm, ps; break
    final = torch.log(p_pivot * S) + ms
    skip = not bool(final < mx)
    return dict(branch=branch, skip=skip, S=float(S), ms=float(ms), top_minus_ms=float(mx - ms), final_pivot=float(final),
                max_logit=float(mx), sum_outlier=float(sop))
torch.manual_seed(0)
V = 154880
rows = []
for sd in (1.0, 2.0, 3.0):
    for top in (30.0, 50.0, 70.0, 90.0, 110.0, 130.0):
        x = torch.randn(V) * sd - 5.0
        x[1234] = top
        r = topp_row(x); r.update(sd=sd, top=top, top1_p=float(torch.softmax(x, -1).max()))
        rows.append(r); print(r)
json.dump(rows, open("/d/out/h1f_emulation.json", "w"), indent=0)
