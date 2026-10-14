"""CPU reproduction (Triton interpreter) of the served top-p kernel on near-one-hot rows (H1f).
Rows: 8 x V. Row 0..3: one-hot-like (top logit +g above a random tail); rows 4..7: ordinary. top_p 0.95.
Reports, per row, whether tokens outside the exact top-p set survive, and the surviving tail mass."""
import torch, json
import vllm.v1.sample.ops.topk_topp_triton as T
T.num_compute_units = lambda *a, **k: 8          # CPU shim: no device SM count
V = 154880
torch.manual_seed(0)
res = []
for gap in (8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 35.0, 40.0):
    x = torch.randn(8, V) * 2.0
    for r in range(4):
        x[r, r * 7] = x[r].max() + gap
    p = torch.full((8,), 0.95)
    y = T.apply_top_k_top_p_triton(x.clone(), None, p)
    pre = torch.softmax(x.double(), -1)
    for r in range(8):
        kept = torch.isfinite(y[r])
        s = torch.sort(pre[r], descending=False)
        cs = torch.cumsum(s.values, 0)
        exact_keep = torch.zeros(V, dtype=torch.bool)
        exact_keep[s.indices[cs > 0.05]] = True
        exact_keep[s.indices[-1]] = True
        extra = kept & ~exact_keep
        res.append(dict(gap=gap, row=r, top1=float(pre[r].max()), n_kept=int(kept.sum()), n_exact=int(exact_keep.sum()),
                        n_extra=int(extra.sum()), extra_mass=float(pre[r][extra].sum()),
                        missing=int((exact_keep & ~kept).sum())))
for z in res:
    if z["row"] in (0, 4):
        print(z)
json.dump(res, open("/d/out/h1f_triton_repro.json", "w"), indent=0)
