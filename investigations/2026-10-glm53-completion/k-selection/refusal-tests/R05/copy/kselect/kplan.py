#!/usr/bin/env python3
"""K: boundary and cell plan for the model-free selection-kernel fixture (stdlib only).

Every geometry constant is copied from the source cited next to it (trees as in
source-review-lc/SOURCE-REVIEW.md: v070 = 0.7.0 with B12x fe054789; v091 = 0.9.1 with B12x 7fcc094e;
B1 = B0 + d7-fix.diff, whose B12x files are byte-identical to 0.9.1's).

Units. L = token length of one query row (= position + 1). P = L // 4 = pool rows of that row
(indexer.py b0:1110 `compressed_seq_lens = seq_lens // compress_ratio`). Under 0.7.0's DCP2 (interleave
1, server log "compress_ratio=4, world_size=2, interleave=1") rank r holds global pools p = 2*i + r,
so its local pool count is P_r = P // 2 + (1 if r < P % 2 else 0) (utils.py v070:964-1000,
get_dcp_local_seq_lens). Kernel regime boundaries are in LOCAL pool rows b; a regime with boundary b
is live in a row iff P_local > b.

Rows per decode step = requests x (K + 1), padded to the next captured size (thresholds.py).

usage: python3 kplan.py [out.json]   (writes plan.json next to this file by default)
"""
from __future__ import annotations

import json
import os
import sys

TOPK = 512                  # index_topk 2048 / index_kpool 4 (sparse_attn_indexer_kpool.py:855)
KEEP = TOPK - 1             # pool_ids[:, : select_k - 1] (kpool.py b0:1043-1044)
KPOOL = 4
PAGE = 64                   # PAGED_INDEX_PAGE_SIZE (paged.py v091:48)
SLICE = 16384               # _TWO_LEVEL_SLICE_TOKENS, in pool rows (paged.py v091:36)
MAX_SLICES = 32             # _TWO_LEVEL_MAX_SLICES (paged.py v091:37)
SUPERTILE = 32768           # _PAGED_INDEX_SUPERTILE_K_DEFAULT (paged.py v091:51)
BLOCK_K = 512               # _PAGED_INDEX_TILE_BLOCK_K (paged.py v091:53)
FOLD_BUDGET = 256 << 20     # _TWO_LEVEL_FOLD_MAX_MIB_DEFAULT (paged.py v091:40)
SLACK = 448                 # _BATCH_SLACK, 32 heads / top-k 512 (fused_indexer.py v091:1137, :1141-1151)
CARRY_CAP = TOPK + SLACK    # 960 (fused_indexer.py v091:1605)
FUSED_MAX_ROWS = 16         # _C4_B12X_FUSED_MAX_ROWS, SM120 (fused_indexer.py v091:140, :917-926)
SMS = 188                   # RTX PRO 6000 Blackwell Max-Q (HANDOFF.md); re-read on the GPU at run time
L_MAX = 330_000             # "the equivalent of 330k" (amendment 21 section 3): prompt + 327,680 cap
CAPTURE = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96]

ENGINES = {
    "b1": dict(
        label="B1 = 0.9.1 + bundle d1234 + D7 fix (corrected computation)",
        image="sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a",
        image_tag="local/glm53-v0.9.1-b1-d7:local",
        K=3, dcp=1,
        width_pages=4097,       # 241 blocks x 17 pages (SOURCE-REVIEW geometry table; stage-2 captures: width 4097)
        col511_fix=True,        # _move_lowest_selected_pool_last (kpool.py b0:63-76, :966)
        tiled_rows=(32, 24),    # 8 and 5-6 requests in flight (GPQA ran mostly at 8)
        fused_rows=(16, 8, 4),  # 3-4, 2 and 1 requests
    ),
    "v070": dict(
        label="stock 0.7.0 (shipped computation; DCP2 per-rank geometry + merge)",
        image="sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629",
        image_tag="ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx (0.7.0)",
        K=5, dcp=2,
        width_pages=2058,       # per rank: 147 blocks x 14 pages (SOURCE-REVIEW geometry table)
        col511_fix=False,       # stock kpool.py (e7935f2a) keeps pool_ids[:, :511] as merged
        tiled_rows=(48, 24),    # 8 and 3-4 requests
        fused_rows=(16, 8),     # 2 and 1 requests (rows 12 -> 16, 6 -> 8 padded)
    ),
}

# Input families, in run order inside a cell. 'real_*' = real index-K / query bytes from the stage-2
# captures (bank/), 'gauss' = engine-style FWHT-quantised Gaussian (stage-1 generator), the rest are
# exact-arithmetic adversarial constructions (scores exactly representable and computed exactly in
# any fp32 summation order; validated on CPU).
FAMILIES = ("real_L3", "real_L43", "gauss", "plateau", "zero", "ulp", "edge")
EXACT_FAMILIES = ("plateau", "zero", "ulp", "edge")
REPEATS = 3                 # scratch pre-filled 0x00 / 0xFF / 0x7F on repeats 0 / 1 / 2


def cdiv(a, b):
    return -(-a // b)


def padded_rows(rows, cap_max):
    for c in CAPTURE:
        if rows <= c <= cap_max:
            return c
    return rows


def local_pools(P, dcp, r):
    if dcp == 1:
        return P
    return P // 2 + (1 if r < P % 2 else 0)


def fold_plan(width_pages, rows):
    """Mirror of _plan_two_level_fold (paged.py v091:150-230; identical in v070 apart from the
    allow_transient switch, which the eager engine path leaves at its default True)."""
    width_tokens = width_pages * PAGE
    if width_tokens < 2 * SLICE:
        return dict(on=False, reason="below threshold")
    slice_tokens = SLICE
    min_slice = cdiv(width_tokens, MAX_SLICES)
    if min_slice > slice_tokens:
        slice_tokens = cdiv(min_slice, PAGE) * PAGE
    sup_pages = SUPERTILE // PAGE
    n_chunks = cdiv(width_pages, sup_pages)
    slices, total = [], 0
    for c in range(n_chunks):
        pages = min(sup_pages, width_pages - c * sup_pages)
        n = max(1, cdiv(pages * PAGE, slice_tokens))
        slices.append([n, total])
        total += n
    nbytes = rows * (total * TOPK * 8 + 4)
    if nbytes > FOLD_BUDGET:
        return dict(on=False, reason="slab over budget", candidate_nbytes=nbytes)
    # per-chunk split extent (paged.py v091:1175-1177)
    ext = []
    for c in range(n_chunks):
        pages = min(sup_pages, width_pages - c * sup_pages)
        e = cdiv(cdiv(pages * PAGE, slices[c][0]), BLOCK_K) * BLOCK_K
        ext.append(e)
    return dict(on=True, slice_rows=slice_tokens, n_chunks=n_chunks, slices=slices, total_slices=total,
                split_extent=ext, candidate_nbytes=nbytes)


def fused_ctas(width_pages, rows, sms=SMS):
    # scratch.py v091:1020-1026 / _resolve_default_ctas_per_group (fused_indexer.py v091:966-976)
    return max(1, min(width_pages, sms // max(1, rows)))


def expected_route(rows):
    return "paged_fused" if rows <= FUSED_MAX_ROWS else "paged_tiled"


def tiled_boundaries(e):
    """Local pool-row boundaries of the tiled route: level-1 slice starts and supertile chunk starts."""
    fp = fold_plan(e["width_pages"], max(e["tiled_rows"]))
    assert fp["on"], fp
    pmax = local_pools(L_MAX // KPOOL, e["dcp"], 0)
    out = {}
    for b in range(fp["slice_rows"], pmax + 1, fp["slice_rows"]):
        out.setdefault(b, []).append(f"slice{b // fp['slice_rows'] + 1}")
    for b in range(SUPERTILE, pmax + 1, SUPERTILE):
        out.setdefault(b, []).append(f"chunk{b // SUPERTILE + 1}")
    return [(b, "+".join(k)) for b, k in sorted(out.items())]


def fused_boundaries(e, rows, sms=SMS):
    """Per-CTA share boundaries (fused_indexer.py v091:1843-1850, :2328-2370): a CTA holding k pages
    first trims at the end when k >= 9 (k*64 > 512), first trims mid-loop when k >= 16 (deferred flow:
    trim at iteration p when 64p + 64 > 960) and a second time when k >= 23. ceil(ceil(P/64)/c) >= k
    <=> P > 64 (k-1) c."""
    c = fused_ctas(e["width_pages"], rows, sms)
    pmax = local_pools(L_MAX // KPOOL, e["dcp"], 0)
    out = []
    for k, name in ((9, "final-trim"), (16, "mid-trim-1"), (23, "mid-trim-2")):
        b = 64 * (k - 1) * c
        if b <= pmax:
            out.append((b, f"{name}(k={k},ctas={c})"))
    return out


def controls(e):
    pmax_local = local_pools(L_MAX // KPOOL, e["dcp"], 0)
    return [(512, "short(P~512)"),
            (30_000 if e["dcp"] == 1 else 15_000, "overlap(~120k tokens; stage-2 real-input range)"),
            (pmax_local, "max(330k-equivalent)")]


def targets(e, b, kind, ctas=None):
    """Global pool counts of each request's LAST row around local boundary b (12 per boundary; 16 for
    the fused trim boundaries, which also bracket the neighbouring per-CTA page counts k-1 and k+1 at
    b -/+ 64*ctas local pools, in case the source reading of the trim condition is off by one page)."""
    if e["dcp"] == 1:
        G = b
        offs = [-512, -64, -2, -1, 0, 1, 2, 3, 64, 65, 512, 1024]
        if ctas and "trim" in kind:
            offs += [-64 * ctas, -64 * ctas + 1, 64 * ctas, 64 * ctas + 1]
    else:
        # rank 0 crosses at P = 2b+1, rank 1 at P = 2b+2; K+1 = 6 rows span <= 2 pool values
        G = 2 * b
        offs = [-1024, -128, -2, -1, 0, 1, 2, 3, 4, 129, 1024, 2049]
        if ctas and "trim" in kind:
            offs += [-128 * ctas + 1, -128 * ctas + 2, 128 * ctas + 1, 128 * ctas + 2]
    if kind.startswith("max"):
        G = L_MAX // KPOOL
        offs = [-1024, -64, -3, -2, -1, 0, -5, -4, -128, -65, -512, -2048]
    # Post-inspection change PI-1 (DIAGNOSIS-run1.md): every row of a request (L-K..L) must have L >= 1,
    # so the lowest last-row target is cdiv(K+1, KPOOL) pools (was 1: v070 K=5 produced a row with L=-1).
    # Changes only the four v070 short(P~512) cells; B1 (K=3) is unchanged.
    lo = cdiv(e["K"] + 1, KPOOL)
    out = [max(lo, G + o) for o in offs]
    return out


def make_batches(e, rows, tgt):
    """Pack targets into fixed-shape batches. Request i of a batch: last-row token length
    L = 4*T + phase (phase cycles 0..3 so every tail count occurs); its K+1 rows have L-K..L."""
    K = e["K"]
    rpb = rows // (K + 1)
    pad = rows - rpb * (K + 1)
    seq = []
    for i, T in enumerate(tgt):
        seq.append(4 * T + (i % 4))
    nb = cdiv(len(seq), rpb)
    while len(seq) < nb * rpb:          # keep every batch full: reuse targets with the next phase
        j = len(seq) - len(tgt)
        seq.append(4 * tgt[j % len(tgt)] + ((j + 1) % 4))
    batches = [seq[i * rpb:(i + 1) * rpb] for i in range(nb)]
    return batches, rpb, pad


def row_lengths(e, batch, pad):
    K = e["K"]
    Ls = []
    for Llast in batch:
        Ls += [Llast - K + j for j in range(K + 1)]
    Ls += [0] * pad                       # zero-length padding rows (assumed padding representation)
    return Ls


def build(sms=SMS):
    cells = []
    for name, e in ENGINES.items():
        order = [("tiled", r) for r in e["tiled_rows"]] + [("fused", r) for r in e["fused_rows"]]
        for route, rows in order:
            assert expected_route(rows) == ("paged_fused" if route == "fused" else "paged_tiled")
            bnds = tiled_boundaries(e) if route == "tiled" else fused_boundaries(e, rows, sms)
            for b, kind in bnds + controls(e):
                tgt = targets(e, b, kind, fused_ctas(e["width_pages"], rows, sms) if route == "fused" else None)
                batches, rpb, pad = make_batches(e, rows, tgt)
                # PI-1: no request row may have a non-positive token length (padding rows are separate)
                assert all(L >= 1 for bt in batches for L in row_lengths(e, bt, 0)), (name, route, rows, b)
                # coverage: some row's local pool count <= b and some > b, on every rank
                cov = {}
                for r in range(e["dcp"]):
                    loc = [local_pools(L // KPOOL, e["dcp"], r) for bt in batches for L in row_lengths(e, bt, pad)
                           if L > 0]
                    cov[r] = dict(min=min(loc), max=max(loc), at_or_below=sum(1 for x in loc if x <= b),
                                  above=sum(1 for x in loc if x > b))
                cell = dict(
                    id=f"{name}/{route}{rows}/b{b}", engine=name, route=route, rows=rows, boundary_local=b,
                    kind=kind, boundary_tokens_global=b * KPOOL * e["dcp"], requests_per_batch=rpb,
                    pad_rows=pad, batches=batches, coverage=cov,
                    expect=dict(route=expected_route(rows),
                                ctas_per_group=(fused_ctas(e["width_pages"], rows, sms) if route == "fused" else 1),
                                fold=(fold_plan(e["width_pages"], rows) if route == "tiled" else None)),
                    max_tokens=max(max(bt) for bt in batches))
                if not kind.startswith(("short", "overlap", "max")):
                    for r, c in cov.items():
                        assert c["at_or_below"] > 0 and c["above"] > 0, (cell["id"], r, c)
                cells.append(cell)
    return cells


def summary(cells):
    lines = []
    for c in cells:
        nb = len(c["batches"])
        lines.append(f"{c['id']:28s} {c['kind']:44s} tokens~{c['boundary_tokens_global']:>8,d} "
                     f"batches {nb:2d} x {len(FAMILIES)} families x {REPEATS} repeats; "
                     f"max L {c['max_tokens']:,}; route {c['expect']['route']}"
                     + (f" ctas {c['expect']['ctas_per_group']}" if c['route'] == 'fused' else
                        f" fold slices {c['expect']['fold']['total_slices']}"))
    return lines


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan.json")
    cells = build()
    meta = dict(TOPK=TOPK, KEEP=KEEP, KPOOL=KPOOL, PAGE=PAGE, SLICE=SLICE, SUPERTILE=SUPERTILE, SLACK=SLACK,
                SMS=SMS, L_MAX=L_MAX, FAMILIES=FAMILIES, EXACT_FAMILIES=EXACT_FAMILIES, REPEATS=REPEATS,
                engines={k: {kk: vv for kk, vv in v.items()} for k, v in ENGINES.items()},
                fold={k: fold_plan(v["width_pages"], max(v["tiled_rows"])) for k, v in ENGINES.items()})
    json.dump(dict(meta=meta, cells=cells), open(out, "w"), indent=1)
    for line in summary(cells):
        print(line)
    tot = sum(len(c["batches"]) for c in cells) * len(FAMILIES)
    print(f"{len(cells)} cells, {tot} batches x {REPEATS} repeats = {tot * REPEATS} kernel calls "
          f"(v070 calls are per rank, plus one merge per perspective)")
