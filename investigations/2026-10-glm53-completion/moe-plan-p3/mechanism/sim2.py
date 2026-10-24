# Per-stage fp16 flip fractions (FC1 store only), plans K64/N256 vs K128/N128.
import torch, importlib.util, sys
sys.argv = ['x', '0']
src = open('/w/sim.py').read().split('res = {}')[0]
exec(src)
for mode in ('rne', 'rz'):
    e = 0
    x16 = x.to(f32).to(f16).to(f32)
    ag = had128((x16 * suh_g[e]).to(f16).to(f32)).to(f16)
    A = (gemm(ag, W1[e][:, :I], 2, mode) * g1); B = (gemm(ag, W1[e][:, :I], 4, mode) * g1)
    pre = float(((A - B).norm() / A.norm()))
    a16, b16 = A.to(f16), B.to(f16)
    print(mode, 'fc1 fp32 pre-round rel_l2 %.2e' % pre, 'fc1 fp16 elem differ frac %.4f' % float((a16 != b16).float().mean()),
          'max ulp diff', int(((a16.view(torch.int16).int() - b16.view(torch.int16).int()).abs()).max()))
