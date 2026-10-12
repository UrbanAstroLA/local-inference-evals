"""B0 TP2 all-reduce fixture: input patterns, reference and mismatch classifier.

Shared by the GPU fixture (in-graph generation and checking), the CPU tests and the
post-run analysis.  Every function here is plain torch and gives bit-identical results on
CPU and CUDA: only int64 arithmetic that never overflows (every product < 2**63), exact
small-integer conversions and multiplications by exact powers of two.

Values come from two fixed random tables per rank (built once from a hash of the table
index, rank and family), each TABLE_LEN entries:
  int   : integers in [-127, 127].  The sum of two is an integer of magnitude <= 254 and is
          exactly representable in bf16, so the observed peer value of any output word can
          be read back exactly and attributed to a source.
  arith : normal bf16 values (+/-)(128+m)*2**(e-17), m in [0,127], e in [0,20]
          (magnitudes 2**-10 .. ~2**11), plus -0.0 lanes (1 in 29).  Sums need rounding
          (exponent gaps up to 20 include full absorption of the smaller operand).
Element i of rank r's input for the call at fixture step s, layer l is
  table[r][fam(s,l)*TABLE_LEN + (off(s,l) + i) % TABLE_LEN],
with fam(s,l) = (s+l) % 2 and off(s,l) a 32-bit hash of (s, l) reduced mod TABLE_LEN.
Every replay therefore sees new, distinguishable inputs; a value read from another
step, layer, rank or index is identified by regenerating that candidate exactly.
Owner/zero emulation (MLA owner layers and the split embedding add a rank's +0.0): at
layers l % 3 == 2 one rank, chosen by (s + l) // 3 parity, contributes all +0.0.

Reference: out = bf16_RN(a + b) for every element, the same on both ranks.  Computed here
as (a.float() + b.float()).to(bf16): an fp32 RN add followed by an RN conversion.  Double
rounding through fp32 (24-bit significand) is innocuous for addition into bf16 (8-bit)
because 24 >= 2*8 + 2 (Figueroa 1995); the CPU tests also check it against exact rational
rounding.  B12x computes add.rn.bf16x2 (single RN rounding of the exact sum; operand order
is local + peer, and addition is commutative), PyNCCL/NCCL sums two bf16 operands once.
No NaN, Inf or subnormal value is generated, so exact (bitwise) comparison is the
criterion; no tolerance is used.
"""
from __future__ import annotations

from fractions import Fraction

import torch

HIDDEN = 4096
TABLE_LEN = 1 << 22          # entries per family per rank (> 64 rows * 4096 = 262144)
K_S = 0x5851F42D             # step multiplier (odd, < 2**31)
K_L = 0x0F4243               # layer multiplier (1000003)
K_R = 0x14057B7F             # rank multiplier (< 2**31)
K_F = 0x3C6EF372             # family multiplier (< 2**31)
H1 = 0x7FEB352D              # mixer multipliers (< 2**31 so x * H < 2**63 for x < 2**32)
H2 = 0x2C1B3C6D
MASK32 = 0xFFFFFFFF
E_SPAN = 21                  # arith exponent index 0..20
OWNER_PERIOD = 3


def mix32(key: torch.Tensor) -> torch.Tensor:
    """32-bit avalanche hash of an int64 tensor (result in [0, 2**32))."""
    x = key & MASK32
    x = x ^ (x >> 16)
    x = (x * H1) & MASK32
    x = x ^ (x >> 15)
    x = (x * H2) & MASK32
    return x ^ (x >> 16)


def make_tables(device=None) -> torch.Tensor:
    """bf16 [2 ranks, 2 * TABLE_LEN]: int family then arith family, per rank."""
    idx = torch.arange(TABLE_LEN, dtype=torch.int64, device=device)
    pow2 = torch.tensor([2.0 ** (e - 17) for e in range(E_SPAN)], dtype=torch.float32,
                        device=device)
    out = torch.empty((2, 2 * TABLE_LEN), dtype=torch.bfloat16, device=device)
    for rank in (0, 1):
        x = mix32(idx + rank * K_R)
        out[rank, :TABLE_LEN] = (torch.remainder(x, 255) - 127).to(torch.bfloat16)
        x = mix32(idx + rank * K_R + K_F)
        sign = (1 - 2 * (x & 1)).to(torch.float32)
        e_idx = torch.remainder(x >> 1, E_SPAN)
        mant = (((x >> 8) & 127) + 128).to(torch.float32)
        v = sign * mant * pow2[e_idx]
        v = torch.where(torch.remainder(x >> 16, 29) == 0, torch.full_like(v, -0.0), v)
        out[rank, TABLE_LEN:] = v.to(torch.bfloat16)
    return out


def _as_step(s, device) -> torch.Tensor:
    if isinstance(s, torch.Tensor):
        return s
    return torch.tensor(int(s), dtype=torch.int64, device=device)


def call_index(s, lvec: torch.Tensor, iota: torch.Tensor) -> torch.Tensor:
    """int64 [L, N] table positions for the calls (s, lvec) (rank independent)."""
    s_t = _as_step(s, lvec.device)
    lsl = s_t + lvec
    off = torch.remainder(mix32(s_t * K_S + lvec * K_L), TABLE_LEN)       # [L]
    fam = torch.remainder(lsl, 2) * TABLE_LEN                              # [L]
    return torch.remainder(iota[None, :] + off[:, None], TABLE_LEN) + fam[:, None]


def generate(tables: torch.Tensor, s, rank: int, lvec: torch.Tensor,
             index: torch.Tensor) -> torch.Tensor:
    """bf16 [L, N] input of `rank` for the calls (s, lvec); index from call_index.

    Graph-capturable: no host synchronisation, no data-dependent shapes.
    """
    s_t = _as_step(s, lvec.device)
    L, N = index.shape
    v = torch.index_select(tables[int(rank)], 0, index.reshape(-1)).view(L, N)
    lsl = s_t + lvec
    owner = (torch.remainder(lvec, OWNER_PERIOD) == OWNER_PERIOD - 1) & (
        torch.remainder(torch.div(lsl, OWNER_PERIOD, rounding_mode="floor"), 2) == int(rank))
    return torch.where(owner[:, None], torch.zeros_like(v), v)


def reference_sum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 RN(a + b): fp32 add then RN conversion (innocuous double rounding)."""
    return (a.float() + b.float()).to(torch.bfloat16)


_CPU_TABLES: torch.Tensor | None = None


def cpu_tables() -> torch.Tensor:
    global _CPU_TABLES
    if _CPU_TABLES is None:
        _CPU_TABLES = make_tables("cpu")
    return _CPU_TABLES


def generate_layer(s: int, layer: int, rank: int, elements: int) -> torch.Tensor:
    """One call's flat bf16 input on CPU (used by tests and the classifier)."""
    lvec = torch.tensor([layer], dtype=torch.int64)
    iota = torch.arange(elements, dtype=torch.int64)
    return generate(cpu_tables(), s, rank, lvec, call_index(s, lvec, iota))[0]


# ----------------------------------------------------------------------------- exact RN
def bf16_bits_to_fraction(bits: int) -> Fraction:
    bits &= 0xFFFF
    sign = -1 if bits >> 15 else 1
    exp = (bits >> 7) & 0xFF
    frac = bits & 0x7F
    if exp == 0:
        return sign * Fraction(frac, 1 << 7) * Fraction(1, 1 << 126)
    if exp == 0xFF:
        raise ValueError("inf/nan not expected")
    value = Fraction(128 + frac, 128)
    return sign * (value * (Fraction(2) ** (exp - 127)))


def exact_rn_bf16(value: Fraction) -> float:
    """Round an exact rational to bf16 (round to nearest, ties to even), normal range."""
    if value == 0:
        return 0.0
    sign = -1 if value < 0 else 1
    mag = abs(value)
    exp = mag.numerator.bit_length() - mag.denominator.bit_length()
    while Fraction(2) ** exp > mag:
        exp -= 1
    while Fraction(2) ** (exp + 1) <= mag:
        exp += 1
    ulp = Fraction(2) ** (exp - 7)
    q, r = divmod(mag, ulp)
    q = int(q)
    if r > ulp / 2 or (r == ulp / 2 and q % 2 == 1):
        q += 1
    return float(sign * q * ulp)


# ------------------------------------------------------------------------ classification
CLASS_ORDER = (
    "ok",
    "arithmetic",          # right operands, different rounding (RZ / 1-ulp) of their sum
    "wrong-rank",          # peer word replaced by this rank's own word
    "missing-peer",        # peer word read as +0.0 (unpublished / cleared record)
    "stale-same-slot",     # peer word of the B12x call 2k earlier (same incoming slot)
    "stale-other-slot",    # peer word of the B12x call 2k+1 earlier (slot parity error)
    "cross-step",          # peer word of the same layer, earlier replay
    "addressing",          # peer word from a shifted index of the right call
    "local-stale",         # own operand from an earlier call, right peer word
    "unexplained",
)


def _word_view(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def _shift(t: torch.Tensor, d: int) -> tuple[torch.Tensor, torch.Tensor]:
    """t shifted by d elements and the mask of positions that have a source."""
    out = torch.zeros_like(t)
    valid = torch.zeros(t.numel(), dtype=torch.bool)
    n = t.numel()
    if 0 < d < n:
        out[d:] = t[:-d]
        valid[d:] = True
    elif 0 < -d < n:
        out[:d] = t[-d:]
        valid[:d] = True
    return out, valid


def _fit(t: torch.Tensor | None, n: int):
    """Candidate source as n elements plus its validity mask: truncated, or padded when
    the earlier call was shorter (positions beyond its length are not candidates)."""
    if t is None:
        return None, None
    if t.numel() >= n:
        return t[:n], None
    out = torch.zeros(n, dtype=t.dtype)
    out[:t.numel()] = t
    valid = torch.zeros(n, dtype=torch.bool)
    valid[:t.numel()] = True
    return out, valid


def rz_sum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 round-toward-zero of the fp32 sum (an alternative rounding)."""
    f = (a.float() + b.float()).view(torch.int32)
    return (f & ~0xFFFF).view(torch.float32).to(torch.bfloat16)


def classify(y: torch.Tensor, local: torch.Tensor, peer: torch.Tensor, *,
             prev_calls_peer: list[torch.Tensor | None],
             prev_steps_peer: list[torch.Tensor | None],
             prev_calls_local: list[torch.Tensor | None],
             shifts: tuple[int, ...] = (8, -8, 16, -16, HIDDEN, -HIDDEN),
             max_examples: int = 8) -> dict:
    """Attribute every 32-bit output word (two bf16, the wire payload unit) to a source.

    y, local, peer: flat bf16 [N] observed output, this rank's input and the peer's input
    of the failing call.  prev_calls_peer[k-1]: the peer input of the B12x call k calls
    earlier in the global push sequence (None if unknown).  prev_steps_peer[j-1]: the
    peer input of the same layer j replays earlier.  prev_calls_local: this rank's input of
    the previous calls (local-operand staleness).

    For every word the set of candidate classes it matches is recorded; its label is the
    first match in CLASS_ORDER.  A 16-byte pack (four words) is labelled with the first
    class all four words match; a pack with no common class is a torn pack.  Coincidental
    word matches (e.g. a stale -0.0 lane equals a missing peer) are therefore reported in
    `word_matches` without splitting a pack.
    """
    n = y.numel()
    yw = _word_view(y)
    nwords = yw.numel()
    ncls = len(CLASS_ORDER)
    match = torch.zeros((nwords, ncls), dtype=torch.bool)
    ci = {name: i for i, name in enumerate(CLASS_ORDER)}

    def mark(name: str, cand_out: torch.Tensor | None, valid=None) -> None:
        if cand_out is None:
            return
        hit = _word_view(cand_out) == yw
        if valid is not None:
            hit &= valid.view(-1, 2).all(dim=1)
        match[:, ci[name]] |= hit

    ref = reference_sum(local, peer)
    mark("ok", ref)
    mark("arithmetic", rz_sum(local, peer))
    ref_bits = ref.view(torch.int16).to(torch.int32)
    y_bits = y.view(torch.int16).to(torch.int32)
    diff = (ref_bits - y_bits).abs().view(-1, 2)
    match[:, ci["arithmetic"]] |= (diff <= 1).all(dim=1) & (diff == 1).any(dim=1)
    mark("wrong-rank", reference_sum(local, local))
    mark("missing-peer", reference_sum(local, torch.zeros_like(local)))
    for k, cand in enumerate(prev_calls_peer, start=1):
        cand, valid = _fit(cand, n)
        if cand is not None:
            mark("stale-same-slot" if k % 2 == 0 else "stale-other-slot",
                 reference_sum(local, cand), valid)
    for cand in prev_steps_peer:
        cand, valid = _fit(cand, n)
        if cand is not None:
            mark("cross-step", reference_sum(local, cand), valid)
    for d in shifts:
        shifted, valid = _shift(peer, d)
        mark("addressing", reference_sum(local, shifted), valid)
    for cand in prev_calls_local:
        cand, valid = _fit(cand, n)
        if cand is not None:
            mark("local-stale", reference_sum(cand, peer), valid)
    match[:, ci["unexplained"]] = ~match[:, :ci["unexplained"]].any(dim=1)

    first = torch.argmax(match.to(torch.int8), dim=1)          # first True in order
    word_counts = {name: int((first == i).sum()) for i, name in enumerate(CLASS_ORDER)}
    bad_words = first != ci["ok"]
    word_matches = {name: int((match[:, i] & bad_words).sum())
                    for i, name in enumerate(CLASS_ORDER) if i != ci["ok"]}
    pack_all = match.view(-1, 4, ncls).all(dim=1)               # [packs, classes]
    has = pack_all.any(dim=1)
    pfirst = torch.argmax(pack_all.to(torch.int8), dim=1)
    pack_counts: dict[str, int] = {}
    for i, name in enumerate(CLASS_ORDER):
        c = int(((pfirst == i) & has).sum())
        if c:
            pack_counts[name] = c
    torn = int((~has).sum())
    if torn:
        pack_counts["torn-pack (mixed sources)"] = torn
    bad = torch.nonzero(bad_words).view(-1)[:max_examples]
    examples = []
    for w in bad.tolist():
        e0 = 2 * w
        examples.append({
            "word": w, "row": e0 // HIDDEN, "col": e0 % HIDDEN,
            "class": CLASS_ORDER[int(first[w])],
            "also_matches": [CLASS_ORDER[i] for i in range(ncls)
                             if bool(match[w, i]) and i != int(first[w])],
            "observed": [float(v) for v in y[e0:e0 + 2].float()],
            "expected": [float(v) for v in ref[e0:e0 + 2].float()],
            "local": [float(v) for v in local[e0:e0 + 2].float()],
            "peer": [float(v) for v in peer[e0:e0 + 2].float()],
        })
    return {"words": nwords, "word_classes": {k: v for k, v in word_counts.items() if v},
            "word_matches": {k: v for k, v in word_matches.items() if v},
            "pack_classes": pack_counts, "examples": examples}
