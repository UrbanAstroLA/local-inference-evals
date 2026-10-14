"""K: input families, poison and paged packing (device-agnostic torch).

Families (kplan.FAMILIES):
  real_L3, real_L43  real index-K pools and real query/weight rows from bank/ (stage-2 captures on stock
                     0.9.1, sanitised to finite bytes). A request draws one contiguous run of the bank's
                     pool stream from a seeded offset; when the stream wraps, the reused pools get a
                     fixed per-wrap e4m3 sign flip on 64 of the 128 dims (a derived pool, so lengths past
                     the bank's 58,599 distinct pools do not manufacture exact duplicate scores).
  gauss              engine-style quantisation of Gaussian K and Q (phase2-stage1/fixture.py:56-104):
                     Hadamard-128, bf16, ue8m0 absmax scale, e4m3; w = w_raw * q_scale * 128^-.5 * 32^-.5.
  plateau/zero/ulp/edge  exact-arithmetic constructions. Head 0 only (w = [1, 0, ...]),
                     q0 = [8, 2^-9 x 6, 0...]; pool p has k0 in {4, 2, 1, -4} (bands 32 / 16 / 8 / relu 0)
                     and an integer m < 8^6 written as six base-8 digits, each digit an e4m3 byte on
                     dims 1..6 (every d * 8^j, d < 8, j < 6 is an e4m3 value in 2^-9 units). Score =
                     scale * (8 k0 + m 2^-18), scale = 1. Every partial sum is a multiple of 2^-18 below
                     64, so any fp32 evaluation order is exact; in the top band consecutive m are exactly
                     1 ulp apart (ulp family). plateau/zero/edge space their m by 128 (exact with a 17-bit
                     accumulator too), ulp uses consecutive m (needs a full fp32 accumulator).
Poison (every family): pools that would outscore every live pool are written into (a) the unused slots
of each request's last live page, (b) two extra pages appended to its block-table row, (c) the null
page 0 that every remaining block-table entry and every zero-length padding row points to. All of them
are ineligible (pool id >= the row's live count), so any poison id in an output is a C1 failure.
"""
from __future__ import annotations

import math

import torch

HEADS, HD, KPOOL, PAGE = 32, 128, 4, 64
PB = PAGE * (HD + 4)
FP8_MAX = 448.0
SLICE, SUPERTILE = 16384, 32768


def _gen(seed, device):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) % (2 ** 63))
    return g


# ------------------------------------------------------------------------------------------ e4m3 tables
_E4 = {}


def e4m3_byte_of_units():
    """dict: integer value in 2^-9 units -> e4m3 byte (positive finite values only)."""
    if "u" not in _E4:
        b = torch.arange(256, dtype=torch.uint8)
        v = b.view(torch.float8_e4m3fn).to(torch.float64)
        d = {}
        for i in range(256):
            x = float(v[i])
            if math.isfinite(x) and x > 0:
                u = x * 512.0
                assert u == int(u)
                d.setdefault(int(u), i)
        _E4["u"] = d
    return _E4["u"]


def e4m3_byte(x: float) -> int:
    t = torch.tensor([x], dtype=torch.float32).to(torch.float8_e4m3fn)
    assert float(t.to(torch.float32)) == x, x
    return int(t.view(torch.uint8))


def digit_table(device):
    """(6, 8) uint8: byte encoding d * 8^j (2^-9 units); d = 0 -> byte 0."""
    key = ("dt", str(device))
    if key not in _E4:
        u = e4m3_byte_of_units()
        t = torch.zeros((6, 8), dtype=torch.uint8)
        for j in range(6):
            for d in range(1, 8):
                t[j, d] = u[d * 8 ** j]
        _E4[key] = t.to(device)
    return _E4[key]


def exact_pools(k0: torch.Tensor, m: torch.Tensor, device):
    """k0 (n,) float values in {4,2,1,-4,448}, m (n,) int64 in [0, 8^6) -> K8 (n,128) uint8."""
    n = k0.shape[0]
    K = torch.zeros((n, HD), dtype=torch.uint8, device=device)
    for val in torch.unique(k0).tolist():
        K[k0 == val, 0] = e4m3_byte(val)
    dt = digit_table(device)
    mm = m.to(torch.int64).clone()
    assert int(mm.min()) >= 0 and int(mm.max()) < 8 ** 6
    for j in range(6):
        K[:, 1 + j] = dt[j][mm % 8]
        mm //= 8
    return K


def exact_query(rows, device):
    q = torch.zeros((rows, HEADS, HD), dtype=torch.float32)
    q[:, 0, 0] = 8.0
    q[:, 0, 1:7] = 2.0 ** -9
    w = torch.zeros((rows, HEADS), dtype=torch.float32)
    w[:, 0] = 1.0
    return q.to(torch.float8_e4m3fn).view(torch.uint8).to(device), w.to(device)


# ------------------------------------------------------------------------------------------ hot positions
def hot_local(p, b, route, rows, ctas, slice_rows=SLICE):
    """Boundary-sensitive local pool ids for a row with p live pools around local boundary b."""
    if p <= 0:
        return []
    h = {0, 1, 2, p - 1, p - 2, p - 3, b - 2, b - 1, b, b + 1, b + 2,
         64 * ((p - 1) // 64), 64 * ((p - 1) // 64) - 1, 512 * ((p - 1) // 512), 512 * ((p - 1) // 512) - 1}
    if route == "tiled":
        for step in (slice_rows, SUPERTILE):
            k = step
            while k < p + 2:
                h.update((k - 1, k, k + 1))
                k += step
    else:
        pages = -(-p // PAGE)
        ppc = -(-pages // max(1, ctas))
        for j in range(1, ctas + 1):
            s = j * ppc * PAGE
            h.update((s - 1, s))
    return sorted(x for x in h if 0 <= x < p)


def hot_global(P, b, route, rows, ctas, dcp):
    """Global ids, highest first: the newest pools and the pools at / after the boundary get the
    designated ranks before the low positions do."""
    if dcp == 1:
        return sorted(hot_local(P, b, route, rows, ctas), reverse=True)
    out = set()
    for r in range(2):
        pr = P // 2 + (1 if r < P % 2 else 0)
        out.update(2 * x + r for x in hot_local(pr, b, route, rows, ctas))
    return sorted((x for x in out if x < P), reverse=True)


# ------------------------------------------------------------------------------------------ families
def fwht_quant(x_bf16, H):
    """phase2-stage1/fixture.py:63-71 (emulates fwht128_quant_fp8, kpool_compress.py v091:79-152)."""
    x = x_bf16.float() @ H
    x = (x * 0.08838834764831845).to(torch.bfloat16).float()
    absmax = x.abs().amax(-1).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(absmax * (1.0 / FP8_MAX))))
    y = (x / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return y.view(torch.uint8), scale


def hadamard128(device):
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < HD:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h.to(device)


class Bank:
    def __init__(self, path, device):
        d = torch.load(path, weights_only=False, map_location="cpu")
        self.layer = d["layer"]
        self.k = d["pools_k"].to(device)
        self.s = d["pools_s"].to(device)
        order = torch.argsort(d["q_seq"], descending=True, stable=True)
        self.q = d["q"][order][:256].to(device)
        self.w = d["w"][order][:256].to(device)
        self.n = int(self.k.shape[0])
        self.device = device
        assert self.n > 0
        assert not bool(((self.k == 0x7F) | (self.k == 0xFF)).any())
        assert bool(torch.isfinite(self.s).all()) and bool((self.s > 0).all())

    def pools(self, n, gen):
        off = int(torch.randint(0, self.n, (1,), generator=gen, device="cpu"))
        idx = (torch.arange(n, device=self.device) + off)
        wrap = idx // self.n
        idx = idx % self.n
        K = self.k[idx].clone()
        S = self.s[idx].clone()
        if int(wrap.max()) > 0:
            for wv in range(1, int(wrap.max()) + 1):
                g = _gen(1_000_003 * wv + 17, "cpu")
                flip = (torch.randperm(HD, generator=g)[: HD // 2]).to(self.device)
                rows = (wrap == wv).nonzero().flatten()
                sub = K[rows]
                sub[:, flip] ^= 0x80
                K[rows] = sub
        return K, S, int(wrap.max())

    def queries(self, nrows, gen):
        off = int(torch.randint(0, self.q.shape[0] - nrows, (1,), generator=gen, device="cpu"))
        return self.q[off: off + nrows].clone(), self.w[off: off + nrows].clone()


def poison_general(q8_rows, w_rows, S):
    """A pool aligned with sum_h max(w_h, 0) q_h of the request's first row, max magnitude, large scale."""
    q = q8_rows[0].view(torch.float8_e4m3fn).float()
    v = (w_rows[0].clamp_min(0)[:, None] * q).sum(0)
    k = torch.where(v >= 0, torch.full_like(v, FP8_MAX), torch.full_like(v, -FP8_MAX))
    kb = k.to(torch.float8_e4m3fn).view(torch.uint8)
    ps = float(S.max()) * 16.0 if S.numel() else 1.0
    return kb, ps


def build_requests(family, Ls_last, K, b, route, rows, ctas, dcp, seed, device, banks):
    """One batch's requests. Ls_last: last-row token length per request. Returns list of dicts
    (q8 (K+1,32,128) uint8, w (K+1,32) f32, K8 (P,128) uint8, S (P,) f32, P, poison_k, poison_s, wraps)."""
    reqs = []
    H = hadamard128(device) if family == "gauss" else None
    for ri, Llast in enumerate(Ls_last):
        P = Llast // KPOOL
        g = _gen(seed * 1009 + ri, "cpu")
        gd = _gen(seed * 1009 + ri, device) if device.type == "cuda" else g
        wraps = 0
        if family.startswith("real_"):
            bank = banks[family]
            K8, S, wraps = bank.pools(P, g)
            q8, w = bank.queries(K + 1, g)
            pk, ps = poison_general(q8, w, S)
        elif family == "gauss":
            kb = torch.randn((P, HD), generator=gd, device=device).to(torch.bfloat16)
            K8, S = fwht_quant(kb, H) if P else (torch.zeros((0, HD), dtype=torch.uint8, device=device),
                                                torch.zeros((0,), device=device))
            qb = torch.randn(((K + 1) * HEADS, HD), generator=gd, device=device).to(torch.bfloat16)
            q8, qs = fwht_quant(qb, H)
            q8 = q8.reshape(K + 1, HEADS, HD)
            wraw = torch.randn((K + 1, HEADS), generator=gd, device=device).to(torch.bfloat16).float()
            w = (wraw * qs.reshape(K + 1, HEADS) * (HD ** -0.5) * (HEADS ** -0.5)).float().contiguous()
            pk, ps = poison_general(q8, w, S)
        else:
            q8, w = exact_query(K + 1, device)
            k0, m = exact_scores(family, P, b, route, rows, ctas, dcp, g)
            K8 = exact_pools(k0.to(device), m.to(device), device)
            S = torch.ones((P,), dtype=torch.float32, device=device)
            pk = exact_pools(torch.tensor([448.0]), torch.tensor([8 ** 6 - 1]), device)[0]
            ps = 16.0
        reqs.append(dict(q8=q8.contiguous(), w=w.contiguous(), K8=K8.contiguous(), S=S.contiguous(), P=P,
                         poison_k=pk, poison_s=ps, wraps=wraps, L=Llast))
    return reqs


def exact_scores(family, P, b, route, rows, ctas, dcp, g):
    """(k0 (P,), m (P,)) for an exact family; positions chosen relative to the hot set."""
    k0 = torch.full((P,), 2.0)                        # mid band default
    m = torch.randint(0, 1000, (P,), generator=g) * 128
    if P == 0:
        return k0, m
    hot = hot_global(P, b, route, rows, ctas, dcp)
    perm = torch.randperm(P, generator=g).tolist()
    hotset = set(hot)
    rest = [x for x in perm if x not in hotset]

    def take(lst, n):
        out, lst[:] = lst[:n], lst[n:]
        return out
    if family == "plateau":
        hot_l = list(hot)
        plat = take(hot_l, 20)
        plat += take(rest, 40 - len(plat))
        top = take(rest, 500)
        for i, x in enumerate(top):
            k0[x], m[x] = 4.0, 128 * (600 + i)
        for x in plat:
            k0[x], m[x] = 4.0, 128 * 100
    elif family == "zero":
        hot_l = list(hot)
        pos = take(hot_l, min(len(hot_l), 30)) + take(rest, 270)
        k0[:] = -4.0
        for i, x in enumerate(pos):
            k0[x], m[x] = 4.0, 128 * (i + 1)
    elif family == "ulp":
        M0 = 100_000
        hot_l = list(hot)
        near = take(hot_l, 21)                         # ranks around 512: i = 80..100
        top_hot = take(hot_l, 10)                      # very top
        others = take(rest, 600 - len(near) - len(top_hot))
        ivals = list(range(80, 80 + len(near))) + list(range(599, 599 - len(top_hot), -1))
        used = set(ivals)
        free = [i for i in range(600) if i not in used]
        for x, i in zip(near + top_hot, ivals):
            k0[x], m[x] = 4.0, M0 + i
        for x, i in zip(others, free):
            k0[x], m[x] = 4.0, M0 + i
    elif family == "edge":
        hot_l = list(hot)
        # interleave: newest / boundary pools alternately get a top rank and a rank at the 512 cut
        ranks = [0, 511, 1, 512, 2, 510, 3, 513, 4, 509, 5, 514, 6, 508, 7, 515, 8, 507, 9, 516,
                 505, 506, 517, 518, 519]
        hsel = take(hot_l, len(ranks))
        assigned = {}
        for x, r in zip(hsel, ranks):
            assigned[r] = x
        free_r = [r for r in range(520) if r not in assigned]
        for r, x in zip(free_r, take(rest, len(free_r))):
            assigned[r] = x
        for r, x in assigned.items():
            k0[x], m[x] = 4.0, 128 * (1000 - r)
    else:
        raise ValueError(family)
    return k0, m


# ------------------------------------------------------------------------------------------ packing
def pack(reqs_local, width, device):
    """reqs_local: list of (K8 (n,128) uint8, S (n,) f32, poison_k (128,), poison_s). Page 0 = poison
    null page. Request i: ceil(n/64) live pages (unused slots of the last one poisoned), then two
    poison pages; every later block-table entry points at page 0. Returns cache (N,64,132) uint8 view
    with planar pages (64x128 K bytes then 64 fp32 scales) and bt (nreq, width) int32."""
    def page_block(Kb, Sb):
        p = Kb.shape[0] // PAGE
        return torch.cat([Kb.reshape(p, PAGE * HD), Sb.contiguous().view(torch.uint8).reshape(p, PAGE * 4)], 1)

    blocks = []
    bt = torch.zeros((len(reqs_local), width), dtype=torch.int32)
    nxt = 1
    pk0, ps0 = reqs_local[0][2], reqs_local[0][3]
    blocks.append(page_block(pk0[None, :].expand(PAGE, HD).contiguous(),
                             torch.full((PAGE,), ps0, dtype=torch.float32, device=device)))
    for i, (K8, S, pk, ps) in enumerate(reqs_local):
        n = K8.shape[0]
        npg = -(-n // PAGE)
        tot = (npg + 2) * PAGE
        Kb = pk[None, :].expand(tot, HD).contiguous().clone()
        Sb = torch.full((tot,), ps, dtype=torch.float32, device=device)
        Kb[:n] = K8
        Sb[:n] = S
        blocks.append(page_block(Kb, Sb))
        assert npg + 2 <= width
        bt[i, : npg + 2] = torch.arange(nxt, nxt + npg + 2, dtype=torch.int32)
        nxt += npg + 2
    cache = torch.cat(blocks, 0).contiguous()
    assert cache.shape[1] == PB
    return cache.view(-1, PAGE, HD + 4), bt.to(device)


def unpack_pools(cache3d, bt_row, n):
    """Inverse of pack for one request (CPU tests): first n pools as (K8, S)."""
    flat = cache3d.reshape(-1, PB)
    npg = -(-n // PAGE)
    pages = flat[bt_row[:npg].long()]
    K = pages[:, : PAGE * HD].reshape(-1, HD)[:n]
    S = pages[:, PAGE * HD:].contiguous().view(torch.float32).reshape(-1)[:n]
    return K, S


def local_split(K8, S, r):
    return K8[r::2].contiguous(), S[r::2].contiguous()
