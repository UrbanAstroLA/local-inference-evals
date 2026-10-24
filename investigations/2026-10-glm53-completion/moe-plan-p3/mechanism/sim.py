# CPU emulation of the B12x mixed-Trellis w4a16 data path, two K partitions.
# Random weights; dims from the tp2r0 slice (H=4096, I=1024, topk=8).
# Plan A = K64/N256 tile (2 k-warp rows), plan B = K128/N128 tile (4 k-warp rows).
import sys, torch, json
torch.manual_seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
H, I, TOPK, M = 4096, 1024, 8, 16
E = TOPK  # concentrated: every token routes to the same 8 experts
f16, f32, f64 = torch.float16, torch.float32, torch.float64

def had128(x):  # orthonormal blockwise H128 on last dim, fp32
    n = x.shape[-1]
    h = torch.ones(1, 1, dtype=f64)
    while h.shape[0] < 128:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    h = (h / 128 ** 0.5).to(f32)
    return (x.reshape(*x.shape[:-1], n // 128, 128) @ h).reshape(x.shape)

def rz(s64):  # fp32 round-toward-zero emulation of an accumulate
    r = s64.to(f32)
    over = r.to(f64).abs() > s64.abs()
    return torch.where(over, torch.nextafter(r, torch.zeros_like(r)), r)

def gemm(a16, w16, rows, mode):
    # a16 [m,K] fp16, w16 [K,N] fp16. Each MMA k16 chunk is summed exactly and
    # added to its k-warp row's fp32 accumulator with one rounding; rows own
    # chunks c with (c mod 2*rows) in {2r, 2r+1}; then the CTA fold.
    m, K = a16.shape; N = w16.shape[1]; C = K // 16
    chunks = torch.einsum('mck,ckn->cmn', a16.to(f64).reshape(m, C, 16),
                          w16.to(f64).reshape(C, 16, N))
    acc = [torch.zeros(m, N, dtype=f32) for _ in range(rows)]
    for c in range(C):
        r = (c % (2 * rows)) // 2
        s = acc[r].to(f64) + chunks[c]
        acc[r] = rz(s) if mode == 'rz' else s.to(f32)
    if rows == 2:
        return acc[0] + acc[1]
    return acc[0] + ((acc[1] + acc[3]) + acc[2])

x = (torch.randn(M, H)).to(torch.bfloat16)
suh_g = torch.randint(0, 2, (E, H)).float() * 2 - 1
suh_u = torch.randint(0, 2, (E, H)).float() * 2 - 1
svg = torch.randint(0, 2, (E, I)).float() * 2 - 1
svu = torch.randint(0, 2, (E, I)).float() * 2 - 1
sd = torch.randint(0, 2, (E, I)).float() * 2 - 1
svh = torch.randint(0, 2, (E, H)).float() * 2 - 1
W1 = torch.randn(E, H, 2 * I).to(f16); g1 = 1.0 / H ** 0.5
W2 = torch.randn(E, I, H).to(f16); g2 = 1.0 / I ** 0.5
tw = torch.full((M, TOPK), 1.0 / TOPK)

def run(rows, mode, r16=(True, True, True)):
    rnd = lambda t, on: t.to(f16).to(f32) if on else t
    out = torch.zeros(M, H, dtype=f32)
    for e in range(E):
        x16 = x.to(f32).to(f16).to(f32)
        ag = had128((x16 * suh_g[e]).to(f16).to(f32)).to(f16)  # input rotation
        au = had128((x16 * suh_u[e]).to(f16).to(f32)).to(f16)
        fg = rnd(gemm(ag, W1[e][:, :I], rows, mode) * g1, r16[0])  # FC1 store
        fu = rnd(gemm(au, W1[e][:, I:], rows, mode) * g1, r16[0])
        ig = had128(fg) * svg[e]; iu = had128(fu) * svu[e]
        a = had128(torch.nn.functional.silu(ig) * iu * sd[e])
        a = rnd(a, r16[1])                                          # activated
        if r16[1]:
            y = gemm(a.to(f16), W2[e], rows, mode)
        else:  # keep fp32 activations: emulate with fp64 chunks of fp32 a
            m_, K = a.shape; C = K // 16
            ch = torch.einsum('mck,ckn->cmn', a.to(f64).reshape(m_, C, 16), W2[e].to(f64).reshape(C, 16, H))
            acc = [torch.zeros(m_, H, dtype=f32) for _ in range(rows)]
            for c in range(C):
                r = (c % (2 * rows)) // 2; acc[r] = (acc[r].to(f64) + ch[c]).to(f32)
            y = acc[0] + acc[1] if rows == 2 else acc[0] + ((acc[1] + acc[3]) + acc[2])
        y = rnd(y * g2, r16[2])                                     # FC2 store
        out += (had128(y) * svh[e]) * tw[:, e:e + 1]                # fp32 top-k sum
    return out

def cmp(a, b):
    d = (a - b).norm() / b.norm()
    ab, bb = a.to(torch.bfloat16), b.to(torch.bfloat16)
    db = (ab.float() - bb.float()).norm() / bb.float().norm()
    fr = (ab != bb).float().mean()
    return dict(fp32_rel_l2=float(d), bf16_rel_l2=float(db), bf16_elem_frac=float(fr),
                fp32_elem_frac=float((a != b).float().mean()))

res = {}
for name, mode, r16 in [('served_rne', 'rne', (True, True, True)),
                        ('served_rz', 'rz', (True, True, True)),
                        ('all_fp32', 'rne', (False, False, False)),
                        ('only_fc1_f16', 'rne', (True, False, False)),
                        ('fc1+act_f16', 'rne', (True, True, False)),
                        ('only_fc2_f16', 'rne', (False, False, True))]:
    A = run(2, mode, r16); B = run(4, mode, r16)
    res[name] = cmp(B, A)
    print(name, json.dumps(res[name]), flush=True)
