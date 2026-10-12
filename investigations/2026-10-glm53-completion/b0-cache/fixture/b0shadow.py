"""Sentinel pattern, shadow image and payload tags for the B0 cache fixture.

Device-agnostic torch (CPU for the CPU tests, CUDA in the fixture).

Model: the raw int8 cache tensor `raw` is filled with a deterministic per-8-byte-word
pattern derived from (word offset, salt). A same-size `shadow` tensor holds the bytes the
cache MUST contain. Every driven operation updates the shadow through its reference
effect (exact copy, zeroing, or "accept the kernel's bytes inside the declared footprint
after they passed their own check"). After every operation `diff(raw, shadow)` must be
empty: any differing byte is a write outside the operation's declared footprint, a missing
write, or wrong content. Differences are reported with block / region / payload decoding.
"""
from __future__ import annotations

import torch

import b0geom as G

_M1 = 0x9E3779B97F4A7C15 - (1 << 64)  # as signed int64
_M2 = 0xBF58476D1CE4E5B9 - (1 << 64)
CHUNK_WORDS = 1 << 26  # 512 MiB of int64 per chunk


def _mix(w: torch.Tensor, salt: int) -> torch.Tensor:
    # splitmix-style; int64 wraparound multiply is two's complement on CPU and CUDA.
    x = w * _M1 + salt
    x = x ^ (x >> 31)
    x = x * _M2
    x = x ^ (x >> 29)
    return x


def fill_pattern(raw: torch.Tensor, salt: int, base_word: int = 0) -> None:
    """Fill a contiguous int8 tensor (size % 8 == 0) with the sentinel pattern."""
    assert raw.dtype == torch.int8 and raw.is_contiguous() and raw.numel() % 8 == 0
    words = raw.view(torch.int64)
    n = words.numel()
    for s in range(0, n, CHUNK_WORDS):
        e = min(n, s + CHUNK_WORDS)
        idx = torch.arange(base_word + s, base_word + e, dtype=torch.int64, device=raw.device)
        words[s:e] = _mix(idx, salt)


def pattern_bytes(start: int, end: int, salt: int, device="cpu") -> torch.Tensor:
    """Expected sentinel bytes for byte range [start, end) (for decoding reports)."""
    w0, w1 = start // 8, (end + 7) // 8
    idx = torch.arange(w0, w1, dtype=torch.int64, device=device)
    b = _mix(idx, salt).view(torch.int8)
    return b[start - w0 * 8 : end - w0 * 8]


def diff(raw: torch.Tensor, shadow: torch.Tensor, limit: int = 64) -> dict:
    """Compare raw and shadow; return count and the first differing byte runs."""
    assert raw.shape == shadow.shape
    rw, sw = raw.view(torch.int64), shadow.view(torch.int64)
    n = rw.numel()
    total = 0
    runs: list[tuple[int, int]] = []
    for s in range(0, n, CHUNK_WORDS):
        e = min(n, s + CHUNK_WORDS)
        ne = rw[s:e] != sw[s:e]
        c = int(ne.sum())
        if c:
            total += c
            if len(runs) < limit:
                wi = torch.nonzero(ne).flatten()[: limit * 4].tolist()
                for w in wi:
                    b = (s + w) * 8
                    if runs and runs[-1][1] == b:
                        runs[-1] = (runs[-1][0], b + 8)
                    else:
                        runs.append((b, b + 8))
    return {"differing_words": total, "runs": runs[:limit]}


def locate(byte_off: int, kind: str = "mla") -> dict:
    """Describe a byte offset of the MLA slot tensor or the idx tensor."""
    if kind == "mla":
        b, r = divmod(byte_off, G.MLA_PAGE)
        d = {"tensor": "mla_slot", "block": b, "page_off": r,
             "mla_slot": byte_off // G.MLA_RECORD, "mla_tok": r // G.MLA_RECORD,
             "draft_tok": r // G.DRAFT_TOKEN_BYTES if r < G.DRAFT_REAL else None,
             "kda_region": ("conv" if r < G.KDA_CONV_BYTES else "ssm" if r < G.KDA_REAL else "pad"),
             "above_2^31": byte_off >= G.TWO31}
    else:
        b, r = divmod(byte_off, G.IDX_PAGE)
        kp, kr = divmod(r, G.IDX_KPAGE_BYTES)
        d = {"tensor": "idx", "block": b, "page_off": r, "kpage": kp,
             "region": ("K" if kr < G.IDX_KPAGE * 128 else "scale"),
             "tail_ring": r < G.TAIL_REAL}
    return d


# ---------------------------------------------------------------- payload tags
# Exact-representable payloads that identify owner group, block, slot/token and epoch.
GROUPS = {"mla": 1, "idx": 2, "tail": 3, "kda": 4, "draft": 5}


def tag_code(group: str, block: int, tok: int, epoch: int) -> int:
    """24-bit code: group(3) | epoch(3) | block(11) | tok%128(7). Exact in fp32."""
    return (GROUPS[group] << 21) | ((epoch & 7) << 18) | ((block & 2047) << 7) | (tok & 127)


def untag_code(c: int) -> dict:
    inv = {v: k for k, v in GROUPS.items()}
    return {"group": inv.get(c >> 21, "?"), "epoch": (c >> 18) & 7,
            "block": (c >> 7) & 2047, "tok_mod128": c & 127}


def bf16_tag_values(group: str, block: int, tok: int, epoch: int, n: int,
                    device="cpu") -> torch.Tensor:
    """n bf16 values, all exact small integers, whose first 4 elements carry the tag in
    6-bit digits (each |v| < 64 is exact in bf16) and the rest a position ramp."""
    c = tag_code(group, block, tok, epoch)
    digits = [(c >> (6 * i)) & 63 for i in range(4)]
    v = torch.empty(n, dtype=torch.float32)
    v[:4] = torch.tensor(digits, dtype=torch.float32)
    if n > 4:
        v[4:] = (torch.arange(4, n, dtype=torch.float32) % 61) - 30
    return v.to(torch.bfloat16).to(device)


def bf16_untag(vals: torch.Tensor) -> dict:
    d = [int(x) for x in vals[:4].float().tolist()]
    c = sum((d[i] & 63) << (6 * i) for i in range(4))
    return untag_code(c)


def fp32_tag_values(group: str, block: int, tok: int, epoch: int, n: int,
                    device="cpu") -> torch.Tensor:
    c = tag_code(group, block, tok, epoch)
    v = torch.empty(n, dtype=torch.float32)
    v[0] = float(c)
    if n > 1:
        v[1:] = (torch.arange(1, n, dtype=torch.float32) % 1021) * 0.5 - 255.0
    return v.to(device)


def e4m3_exact_latent(group_tag: int, n_groups: int = 4, group: int = 128,
                      device="cpu", scale_pow2: int = 0) -> torch.Tensor:
    """A 512-wide bf16 latent whose per-128 groups have amax 448 and otherwise small
    E4M3-exact integers (|v| <= 15), so the GLM_NEXT writer's quantization is exact up to
    the writer's scale rounding (amax * fp32(1/448), rcp.approx). The 24-bit tag is carried
    in elements 1..6 of group 0 as 4-bit digits (0..15, exact in E4M3)."""
    v = torch.zeros(n_groups * group, dtype=torch.float32)
    for gi in range(n_groups):
        base = gi * group
        v[base : base + group] = (torch.arange(group, dtype=torch.float32) % 15) - 7
        v[base] = 448.0
    v[1:7] = torch.tensor([(group_tag >> (4 * i)) & 15 for i in range(6)], dtype=torch.float32)
    return (v * (2.0 ** scale_pow2)).to(torch.bfloat16).to(device)


def e4m3_untag(dequant: torch.Tensor, scale_pow2: int = 0) -> dict:
    d = [int(round(x / (2.0 ** scale_pow2))) & 15 for x in dequant[1:7].float().tolist()]
    return untag_code(sum(d[i] << (4 * i) for i in range(6)))


class Shadow:
    """raw + shadow pair over one cache tensor."""

    def __init__(self, raw: torch.Tensor, salt: int, kind: str):
        self.raw, self.kind, self.salt = raw, kind, salt
        fill_pattern(raw, salt)
        self.shadow = raw.clone()
        self.log: list[dict] = []

    # reference effects -------------------------------------------------------
    def expect_bytes(self, start: int, data: torch.Tensor) -> None:
        d = data.contiguous().view(torch.int8).flatten()
        self.shadow[start : start + d.numel()] = d.to(self.shadow.device)

    def expect_zero(self, start: int, end: int) -> None:
        self.shadow[start:end] = 0

    def expect_copy(self, src: int, dst: int, n: int) -> None:
        self.shadow[dst : dst + n] = self.shadow[src : src + n].clone()

    def accept_actual(self, start: int, end: int) -> None:
        """Adopt the kernel's bytes inside a declared footprint (after they were checked)."""
        self.shadow[start:end] = self.raw[start:end]

    def check(self, label: str, limit: int = 32) -> dict:
        r = diff(self.raw, self.shadow, limit)
        r["label"] = label
        r["ok"] = r["differing_words"] == 0
        r["where"] = [dict(locate(s, self.kind), run=(s, e)) for s, e in r["runs"][:8]]
        self.log.append(r)
        if not r["ok"]:
            # resynchronise so later checks report only new damage
            self.shadow.copy_(self.raw)
        return r
