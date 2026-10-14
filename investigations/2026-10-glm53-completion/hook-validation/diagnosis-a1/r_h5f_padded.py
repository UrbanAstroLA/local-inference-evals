"""CPU reproduction (hooked validation image, real forcewin_hooks._shadow_step + ShadowRing) of the H5f ledger defect.
d2_tail_mapping / note_tail_mapping store the builder's qsl[: n_reqs + 1], where n_reqs includes FULL-graph padding
requests (3 real requests x 4 rows = 12 tokens run in the 16-token graph: n_reqs 4). _shadow_step packs
[final(n), positions(n), qsl, acc(nr), idx(nr)] and slices with the REAL nr, so acc and idx are read at shifted
offsets. Same three B1 (uninjected, circular) requests driven for 6 steps with all drafts accepted, unpadded vs padded."""
import os, types, json
os.environ.setdefault("FORCEWIN", "on"); os.environ["FORCEWIN_SHADOW"] = "1"
os.environ.setdefault("FORCEWIN_WMAX", "512"); os.environ.setdefault("FORCEWIN_DIR", "/tmp/fwtest")
import numpy as np, torch
from vllm import forcewin_hooks as fw
def drive(padded):
    fw.S.__init__(); fw.S.shadow = fw.ShadowRing()
    P = [104, 208, 312]; idx = [0, 1, 2]; blk = [5, 9, 13]
    fw.S.reqs = {}
    for i, p in zip(idx, P):
        r = fw._Req(); r.idx = i; r.P = p; r.F = [1] * 64; fw.S.reqs[i] = r
    fw.S.pcls = torch.zeros(4, 64, dtype=torch.int8); fw.S.npw = 64
    for step in range(6):
        pos, slot, q = [], [], [0]
        for b, (i, p) in enumerate(zip(idx, P)):
            p0 = p + 4 * step
            for j in range(4):
                pos.append(p0 + j); slot.append(blk[b] * 8 + (p0 + j) % 8)
            q.append(q[-1] + 4)
        n_reqs, ntok = 3, 12
        if padded:                       # one padding request of 4 dummy tokens (slot -1, position 0)
            pos += [0] * 4; slot += [-1] * 4; q.append(q[-1] + 4); n_reqs, ntok = 4, 16
        fw.S.d2_tok = (torch.tensor(slot), torch.tensor(pos), torch.tensor(q[: n_reqs + 1]))
        sc = dict(nr=3, acc=torch.tensor([3, 3, 3]), idx=torch.tensor(idx))
        fw._shadow_step(None, sc)
    return {i: sorted({int(k): int(v) for k, v in fw.S.shadow.cls.get(i, {}).items()}.items()) for i in idx}
res = {"unpadded": drive(False), "padded": drive(True)}
print(json.dumps(res))
json.dump(res, open("/d/out/h5f_padded_repro.json", "w"), indent=1)
