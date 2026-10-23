# SPDX-License-Identifier: Apache-2.0
"""local-mapguard: observe-only check that the MLA / index-K / kpool-tail mappings the
kernels CONSUME agree with an expected mapping derived independently from scheduler truth.

PLAN.md amendment 17 (B0).  Extension of the D2 tail guard (local_tailguard.py): same
architecture (env-gated, inert by default; graph-safe device counters + bounded records in
guard-owned device buffers; background readout on a side stream; DUMP trigger file), more
hook sites, and an expected mapping that does NOT come from the runner.

Inert unless GLM53_MAPGUARD=device.  Installed as ``vllm/mapguard_hooks.py`` in a derived
diagnostic image; imported by the patched sparse_attn_indexer_kpool.py, b12x_mla_sparse.py
(worker processes) and v1/core/sched/scheduler.py (EngineCore process).  It never writes an
engine tensor: engine tensors are only read; every store goes to guard-owned buffers.

Where the expected mapping comes from (independence)
  * Scheduler side (EngineCore): the KV cache manager allocates block ids per request and
    group; ``SchedulerOutput`` carries them to every worker (new requests: the full
    per-group lists from ``KVCacheManager.get_blocks(..).get_block_ids()``; running requests:
    the newly appended ids).  The guard's EngineCore hook logs, per schedule() call, a step
    fingerprint, a digest of every scheduled request's group-0/group-1 lists read directly
    from ``kv_cache_manager.get_block_ids(req_id)`` (and the full lists whenever they change),
    and at update_from_output the per-request draft/accepted counts.
  * Worker side: a guard-private Ledger rebuilds request -> per-group block lists from the
    SchedulerOutput stream alone (finished / preempted / new / resumed / appended), assigns a
    tag per request incarnation and keeps scheduler num_computed / scheduled / spec counts.
    It never reads the runner's RequestState rows, BlockTables (staged writes, persistent or
    gathered tables), slot-mapping kernels, metadata builders or idx_mapping.  The only runner
    value it uses is the batch ORDER of request ids (input_batch.req_ids, itself a sort of the
    scheduler's keys) to say which request a batch row is supposed to be.
  * The ledger digest is logged per step on the worker and compared offline with the
    EngineCore digest of the KV cache manager's own lists (transmission check).
  * Expected values per driven access: geometry (block 4352 tokens, MLA kernel page 64,
    kpool 4 -> 1088 index-K pools per block, tail ring 8) checked against kv_cache_config at
    registration; MLA slot = blk[p // 4352] * 4352 + p % 4352; index-K pool slot (only when
    (p + 1) % 4 == 0) = blk[p // 4352] * 1088 + (p // 4) % 1088; tail slot = tailblk * 8 +
    p % 8; reads: page-table rows == the request's own block list, per-token lengths ==
    causal (p + 1) [MLA] or (p + 1) // 4 pools [index-K]; selected MLA slots inside the
    request's own blocks at logical positions <= p.  Positions themselves are checked against
    scheduler state (contiguity per row; first position within [num_computed - drafts of the
    request's previous step, num_computed] under async scheduling; exact offline join).
  * Rank ownership: MLA layer owners are recomputed from VLLM_GLM53_MLA_OWNERS (split:N) and
    the per-layer device write counters must match exactly.

22ao AMENDMENT (live causal-length check on stock 0.9.1 and stock + port 01).  Base = the frozen B1 guard
(d7/step2/guard/mapguard_hooks.py a8e59fa5...).  Additions only; no frozen check changes semantics:
  * site "mla_len": the per-token causal length the B12x sparse-MLA builder hands the consumer
    (cache_seq_lens_per_token) against the authoritative p + 1, signed (len_pos / len_neg on every owned
    layer; a [-8, 8] histogram and one record per batch row per step on the rank's first owned layer).
  * mla_r: the signed consumed count nsa - min(p + 1, width) (nsa_pos / nsa_neg; histogram).
  * Shadow plants (positive controls for the selection checker): on guard-owned copies, for a deterministic
    subset of checked tokens ((step * 131 + token) % 97 == 0), the checks are re-run after planting
    (a) a same-request future key at p + 1, (b) a block of another batch row, (c) an over-long count on
    an exact dense row, (d) an unmasked column.  Ground truth comes from a full scan of the row's
    expected block table, not from the checker's col0 map.  Plants never touch engine memory and are
    counted only in pa/pb/pc/pd counters.
  * check_selected's predicates are factored into _sel_masks (pure extraction, the same expressions) so
    the plants exercise the code that judges the engine.

Graph safety: hooks use fixed-shape device ops only (no .item()/.tolist()/nonzero/boolean
indexing/host tensor creation); per-step values come from one guard-owned device buffer E
that is refreshed by an H2D copy before every real forward (prepare_attn) and disarmed after
it, so FULL-graph replays read the current expectation and the current engine buffers at the
point of consumption.  A replay that consumed a stale mapping shows up as a mismatch.

B1 AMENDMENT (PLAN.md amendment 18a, step 2a; d7/step2/FROZEN.md).  B1 = B0 + d7-fix.diff: the index-K
prefill writer takes the members of a pool that straddles an unaligned chunk start from the request's
tail ring.  Changes against the frozen B0 module (b0-mapguard/guard/mapguard_hooks.py 3fa979c8...):
  * idx_w_pf write rule.  The call site tells the hook what the writer receives (ring_mode "ring" with
    the writer's per-token tail slots, or "none").  Tokens whose writer has a ring: every completing
    pool is EXPECTED written (no missing_pred, no pool_cross from the predicate model).  Tokens without a
    ring (call without ring, or tail slot < 0): B0's predicate as frozen, and a completing pool that
    straddles the chunk start is counted `d7_fallback` (VIOLATION).  Legacy calls without ring_mode
    (the frozen CPU suite) keep B0's predicate exactly and are counted `legacy` (must be 0 in serving).
  * Independent content oracle (straddling pools).  Logical positions come from the ledger
    (scheduler num_computed + offset in the row; prefill rows only, lo == hi), not from engine buffers.
    At the point of consumption the post-write hook snapshots each row's trailing incomplete-pool keys
    and gates into a guard-owned shadow keyed by (layer, request tag, logical position).  A straddling
    pool's earlier members come from that shadow (tag and position verified), its own members from the
    row's batch tokens by ledger position; never from the tail ring the writer reads.
  * Byte semantics.  The expected record is produced by the engine's own compress kernel
    (kpool_compress_and_write_cache, same constexpr specialisation) run on the oracle's members into a
    guard-owned scratch page set: on GPU its bf16/softmax/Hadamard/E4M3/scale semantics are the GPU
    kernel's by construction.  Positive control: every ALIGNED completing pool (all members in the
    chunk, written by the unchanged path) is recomputed the same way and must be byte-identical
    (pc_checked / pc_mism).  pc_mism > 0 => straddling byte checks INVALID, not violations.
  * Outcomes per straddling pool (pre-write bytes snapshotted before the writer): s_verified (stored ==
    expected != pre), s_indet (stored == expected == pre: write not observable), s_missing (stored ==
    pre != expected), s_mism (stored != expected, != pre: foreign / stale / wrong-position members),
    s_unavail (shadow lacks a member: not checked).  Each straddling event is logged (slog) for the
    offline "exercised" count; byte snapshots of failures in bsnap.
  * Everything else (sites, invariants, classes, coverage minimums) unchanged.

Env:
  GLM53_MAPGUARD            "" (off) | "device"
  GLM53_MAPGUARD_DIR        output dir (default /tmp/mapguard)
  GLM53_MAPGUARD_CAP        violation record capacity (default 8192)
  GLM53_MAPGUARD_POLL_SECS  background readout interval (default 20; DUMP trigger and exit too)
  GLM53_MAPGUARD_RING       steps kept in the device position ring (default 16384)
  GLM53_MAPGUARD_SEL_TILE   rows per tile in the selected-slot check (default 256)
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import threading
import time

import numpy as np
import torch

MODE = os.environ.get("GLM53_MAPGUARD", "").strip().lower()
ENABLED = MODE == "device"
OUT_DIR = os.environ.get("GLM53_MAPGUARD_DIR", "/tmp/mapguard")
CAP = int(os.environ.get("GLM53_MAPGUARD_CAP", "8192"))
POLL_SECS = float(os.environ.get("GLM53_MAPGUARD_POLL_SECS", "20"))
RING = int(os.environ.get("GLM53_MAPGUARD_RING", "16384"))
SEL_TILE = int(os.environ.get("GLM53_MAPGUARD_SEL_TILE", "256"))
NLAYERS = 64

# ------------------------------------------------------------------ sites / counters / flags
SITES = ["mla_w", "idx_w_pf", "idx_w_dec", "tail_w_seed", "tail_w_stash",
         "idx_r_dec", "idx_r_pf", "mla_r", "pos", "mla_len"]
S_MLA_W, S_IDX_W_PF, S_IDX_W_DEC, S_TAIL_SEED, S_TAIL_STASH, S_IDX_R_DEC, S_IDX_R_PF, \
    S_MLA_R, S_POS, S_MLA_LEN = range(len(SITES))
CNT = ["exec", "armed", "checked", "written", "spec", "high",          # activity / coverage
       "mism", "mism_spec", "missing", "missing_pred", "unexp", "inact",  # write violations
       "foreign", "stale", "wgroup", "null", "ownoff", "oor", "shared",  # ownership classes
       "pos_contig", "pos_bounds", "qsl",                               # positions / rows
       "rlen", "rtable", "sel_foreign", "sel_causal", "masked_live", "sel_len", "grouppos",
       "pool_cross", "sel_hole", "inact_read",                          # info (not violations)
       # B1 amendment (idx_w_pf): straddling-pool content oracle, positive control, fallback
       "s_events", "s_verified", "s_indet", "s_unavail", "s_missing", "s_mism",
       "pc_checked", "pc_mism", "d7_fallback", "legacy",
       # 22ao amendment (append-only)
       "len_checked", "len_pos", "len_neg", "nsa_pos", "nsa_neg",
       "pa_planted", "pa_det", "pb_planted", "pb_det", "pc_planted", "pc_det", "pd_planted", "pd_det"]
CI = {n: i for i, n in enumerate(CNT)}
VIOLATION_COUNTERS = ["mism", "missing", "missing_pred", "pool_cross", "unexp", "inact", "shared",
                      "pos_contig", "pos_bounds", "qsl", "rlen", "rtable", "sel_foreign",
                      "sel_causal", "masked_live", "sel_len", "grouppos",
                      "s_missing", "s_mism", "d7_fallback"]
BYTE_COUNTERS = ["s_missing", "s_mism"]          # void (INVALID) if the positive control fails
INFO_COUNTERS = ["sel_hole", "inact_read", "s_events", "s_verified", "s_indet", "s_unavail",
                 "pc_checked", "pc_mism", "legacy"]
B = {n: 1 << i for i, n in enumerate(
    ["mism", "missing", "pred", "unexp", "inact", "foreign", "stale", "wgroup", "null",
     "ownoff", "oor", "shared", "posc", "posb", "qsl", "rlen", "rtable", "self", "selc",
     "maskl", "sellen", "gpos", "spec", "pcross", "smiss", "smism", "fallback", "pcmism"])}
# straddling-event log outcome codes (slog column "outcome")
D7_OUT = {"verified": 1, "indet": 2, "unavail": 3, "missing": 4, "mism": 5, "fallback": 6}
SLOG_COLS = ["step", "layer", "tag", "position", "outcome", "rank", "row", "slot"]
SH = 64            # shadow request slots per layer (tag % SH; tag and position verified on read)
SCAP = int(os.environ.get("GLM53_MAPGUARD_SCAP", "65536"))   # straddling-event log rows
BCAP = 256         # byte snapshots (expected / stored / pre-write) of failing or pc-failing pools
LCAP = int(os.environ.get("GLM53_MAPGUARD_LCAP", "131072"))   # 22ao: per-row length records
LREC_COLS = ["step", "batch_row", "tag", "position", "consumed_len", "d", "dense", "row_tokens"]
PLANT_MOD = 97     # 22ao: one checked token in 97 carries the shadow plants
HBINS = 17         # 22ao: signed-difference histograms over [-8, 8] (clamped)
ORACLE_MAX = 4096  # initial oracle scratch records (max_num_batched_tokens 2048; grows if a call is larger)
REC_COLS = ["site", "rank", "layer", "step", "token", "row", "tag", "position", "consumed",
            "expected", "flags", "aux"]
NREC = len(REC_COLS)
H_ARMED, H_STEP, H_NROWS, H_NTOK, H_WARM, H_VERSION = range(6)


def flag_names(f: int) -> list[str]:
    return [n for n, b in B.items() if f & b]


# ------------------------------------------------------------------ geometry
class Geo:
    """B0 cache geometry (b0-cache/PROPOSAL.md section 1); verified at registration."""

    def __init__(self, nb: int, R: int, W0: int, bs: int = 4352, kbs: int = 64, kpool: int = 4,
                 ring: int = 8, high: int = 935, g_attn: int = 0, g_tail: int = 1):
        self.nb, self.R, self.W0 = int(nb), int(R), int(W0)
        self.bs, self.kbs, self.kpool, self.ring, self.high = bs, kbs, kpool, ring, high
        self.ppb = bs // kpool          # index-K pool records per block (1088)
        # index-K READ tables are in index-page units, not KV blocks (post-inspection fix,
        # run-a2): the indexer builder's spec block = compressed_kernel_block_size (64 pools)
        # x compress_ratio (worker/utils.py:302-311), so its table is
        # block_table[:, ::4] // 4 = 17*block + j (indexer.py:1089-1096, 1268-1290): one entry
        # per 64-pool page, 17 pages per 4,352-token block.
        self.ipage = 64                 # pools per index-K page
        self.ipb = self.ppb // self.ipage  # index-K pages per block (17)
        self.g_attn, self.g_tail = g_attn, g_tail

    def as_dict(self):
        return dict(nb=self.nb, R=self.R, W0=self.W0, bs=self.bs, kbs=self.kbs, kpool=self.kpool,
                    ring=self.ring, ppb=self.ppb, ipage=self.ipage, ipb=self.ipb, high=self.high,
                    g_attn=self.g_attn,
                    g_tail=self.g_tail)


class ELayout:
    """Flat int64 layout of the per-step expectation buffer E."""

    def __init__(self, geo: Geo):
        R, W0, NB1 = geo.R, geo.W0, geo.nb + 1
        self.off = {}
        o = 0
        for name, n in (("hdr", 8), ("qsl", R + 1), ("tag", R), ("lo", R), ("hi", R),
                        ("nspec", R), ("nsched", R), ("nb0", R), ("bt1", R), ("bt0", R * W0),
                        ("col0", NB1), ("tagx", NB1), ("grp", NB1)):
            self.off[name] = (o, n)
            o += n
        self.size = o
        self.geo = geo

    def np_views(self, a: np.ndarray) -> dict:
        v = {k: a[o:o + n] for k, (o, n) in self.off.items()}
        v["bt0"] = v["bt0"].reshape(self.geo.R, self.geo.W0)
        return v


class EV:
    """Named device views of E (fixed addresses: safe to capture)."""

    def __init__(self, E: torch.Tensor, lay: ELayout):
        g = lay.geo
        for k, (o, n) in lay.off.items():
            setattr(self, k, E[o:o + n])
        self.bt0 = self.bt0.view(g.R, g.W0)
        self.armed = self.hdr[H_ARMED]
        self.step = self.hdr[H_STEP]
        self.nrows = self.hdr[H_NROWS]
        self.ntok = self.hdr[H_NTOK]
        self.warm = self.hdr[H_WARM]


class Dev:
    def __init__(self, geo: Geo, device):
        self.geo = geo
        self.lay = ELayout(geo)
        self.E = torch.zeros(self.lay.size, dtype=torch.int64, device=device)
        self.ev = EV(self.E, self.lay)
        self.cnt = torch.zeros(len(SITES), len(CNT), dtype=torch.int64, device=device)
        self.rec = torch.zeros(CAP + 1, NREC, dtype=torch.int64, device=device)
        self.nrec = torch.zeros(1, dtype=torch.int64, device=device)
        self.layer = torch.zeros(NLAYERS, len(SITES), 2, dtype=torch.int64, device=device)
        self.ring_step = torch.full((RING,), -1, dtype=torch.int64, device=device)
        self.ring_p0 = torch.full((RING, geo.R), -1, dtype=torch.int64, device=device)
        self.device = device
        # B1 amendment: straddling-pool oracle state (guard-owned; never an engine tensor)
        nsh = NLAYERS * SH * geo.kpool
        self.sh_dump = nsh                      # write sink for tokens not snapshotted
        self.shk = torch.zeros(nsh + 1, 128, dtype=torch.bfloat16, device=device)
        self.shg = torch.zeros(nsh + 1, 128, dtype=torch.bfloat16, device=device)
        self.shtag = torch.full((nsh + 1,), -1, dtype=torch.int64, device=device)
        self.shpos = torch.full((nsh + 1,), -1, dtype=torch.int64, device=device)
        self.slog = torch.zeros(SCAP + 1, len(SLOG_COLS), dtype=torch.int64, device=device)
        self.nslog = torch.zeros(1, dtype=torch.int64, device=device)
        self.bsnap = torch.zeros(BCAP + 1, 3, 132, dtype=torch.uint8, device=device)
        self.bkey = torch.zeros(BCAP + 1, 6, dtype=torch.int64, device=device)
        self.nbsnap = torch.zeros(1, dtype=torch.int64, device=device)
        self.scratch = None                     # oracle record pages, allocated on first use
        # 22ao amendment
        self.lrec = torch.zeros(LCAP + 1, len(LREC_COLS), dtype=torch.int64, device=device)
        self.nlrec = torch.zeros(1, dtype=torch.int64, device=device)
        self.lenhist = torch.zeros(HBINS, dtype=torch.int64, device=device)
        self.nsahist = torch.zeros(HBINS, dtype=torch.int64, device=device)


# ------------------------------------------------------------------ device primitives
def record_rows(dev: Dev, mask: torch.Tensor, rows: torch.Tensor) -> None:
    """Fixed-shape append of rows[mask] (as local_tailguard.record_rows); overflow -> row CAP."""
    m = mask.to(torch.int64)
    idx = dev.nrec[0] + torch.cumsum(m, 0) - 1
    ok = mask & (idx < CAP)
    tgt = torch.where(ok, idx, torch.full_like(idx, CAP))
    dev.rec.index_copy_(0, tgt, rows)
    dev.nrec.add_(m.sum())


def _append(buf: torch.Tensor, n: torch.Tensor, cap: int, mask: torch.Tensor, rows: torch.Tensor) -> None:
    """record_rows for any guard-owned log buffer (row `cap` is the overflow sink)."""
    m = mask.to(torch.int64)
    idx = n[0] + torch.cumsum(m, 0) - 1
    tgt = torch.where(mask & (idx < cap), idx, torch.full_like(idx, cap))
    buf.index_copy_(0, tgt, rows)
    n.add_(m.sum())


def _acc_exec(dev: Dev, site: int, layer: int, armed: torch.Tensor) -> None:
    """One execution of a check site (counted in-graph: proves replayed graphs ran it)."""
    row = dev.cnt[site]
    row[CI["exec"]] += 1
    row[CI["armed"]] += armed
    if 0 <= layer < NLAYERS:
        dev.layer[layer, site, 0] += 1
        dev.layer[layer, site, 1] += armed


def _acc_counts(dev: Dev, site: int, armed: torch.Tensor, counts: dict) -> None:
    row = dev.cnt[site]
    for name, v in counts.items():
        row[CI[name]] += v.to(torch.int64).sum() * armed


def _acc(dev: Dev, site: int, layer: int, armed: torch.Tensor, counts: dict) -> None:
    _acc_exec(dev, site, layer, armed)
    _acc_counts(dev, site, armed, counts)


def rows_of(ev: EV, tok: torch.Tensor, R: int) -> torch.Tensor:
    r = torch.searchsorted(ev.qsl, tok, right=True) - 1
    return r.clamp(0, R - 1)


def owner_bits(ev: EV, geo: Geo, blk: torch.Tensor, row: torch.Tensor, group: int) -> torch.Tensor:
    """Classify a (wrong) block id for batch row `row` in KV group `group` (bit flags)."""
    nb = geo.nb
    oor = (blk < 0) | (blk > nb - 1)
    b = blk.clamp(0, nb)
    g = ev.grp[b]
    if group == geo.g_attn:
        c0 = ev.col0[b]
        own = (c0 >= 0) & (c0 < ev.nb0[row]) & (ev.bt0[row, c0.clamp(0, geo.W0 - 1)] == blk)
    else:
        own = ev.bt1[row] == blk
    null = blk == 0
    live = ~null & ~oor
    foreign = live & ~own & (g == group)
    wgroup = live & ~own & (g >= 0) & (g != group)
    stale = live & ~own & (g < 0)
    ownoff = live & own
    return (foreign.to(torch.int64) * B["foreign"] + stale.to(torch.int64) * B["stale"]
            + wgroup.to(torch.int64) * B["wgroup"] + null.to(torch.int64) * B["null"]
            + ownoff.to(torch.int64) * B["ownoff"] + oor.to(torch.int64) * B["oor"])


def _class_counts(bits: torch.Tensor, bad: torch.Tensor) -> dict:
    return {n: bad & ((bits & B[n]) != 0) for n in ("foreign", "stale", "wgroup", "null",
                                                     "ownoff", "oor")}


def check_writes(dev: Dev, site: int, layer: int, rank: int, kind: str, consumed: torch.Tensor,
                 pos: torch.Tensor, tok: torch.Tensor, written: torch.Tensor | None = None,
                 want_extra: torch.Tensor | None = None, active_extra: torch.Tensor | None = None,
                 pred_ok: torch.Tensor | None = None, pool_span: int = 0,
                 pool_mask: torch.Tensor | None = None) -> None:
    """One write site: consumed[i] is the slot the kernel uses for batch token tok[i] at
    position pos[i].  kind: 'mla' (token slots, group 0), 'idx' (pool slots, group 0),
    'tail' (ring slots, group 1).  written: the kernel's own write predicate (default
    consumed >= 0); want_extra: additional 'must be written' predicate (tail seed);
    pred_ok: tokens the kernel's predicate allows (idx prefill: local index >= kpool-1).
    pool_mask (B1 amendment): tokens to which the B0 batch-gathering rule applies (writer without a
    ring); pool_cross is evaluated only there.  None = every token (frozen B0 behaviour)."""
    geo, ev = dev.geo, dev.ev
    R, W0 = geo.R, geo.W0
    armed = ev.armed
    c = consumed.to(torch.int64)
    p = pos.to(torch.int64)
    t = tok.to(torch.int64)
    row = rows_of(ev, t, R)
    active = (t < ev.ntok) & (t >= 0) & (row < ev.nrows)
    if active_extra is not None:
        active = active & active_extra
    pc = p.clamp(min=0)
    if kind == "tail":
        unit, group = geo.ring, geo.g_tail
        tb = ev.bt1[row]
        want = active & (p >= 0) & (tb != 0)
        exp = torch.where(want, tb * geo.ring + torch.remainder(pc, geo.ring), torch.full_like(c, -1))
        hi_blk = tb
    else:
        group = geo.g_attn
        col = torch.div(pc, geo.bs, rounding_mode="floor")
        colv = active & (p >= 0) & (col < ev.nb0[row])
        eb = ev.bt0[row, col.clamp(0, W0 - 1)]
        if kind == "mla":
            unit = geo.bs
            want = colv
            exp = torch.where(want, eb * geo.bs + torch.remainder(pc, geo.bs), torch.full_like(c, -1))
        else:  # idx
            unit = geo.ppb
            complete = torch.remainder(pc + 1, geo.kpool) == 0
            want = colv & complete
            exp = torch.where(want, eb * geo.ppb + torch.remainder(
                torch.div(pc, geo.kpool, rounding_mode="floor"), geo.ppb), torch.full_like(c, -1))
        hi_blk = eb
    if want_extra is not None:
        want = want & want_extra
    if written is None:
        written = c >= 0
    if pred_ok is not None:
        kernel_written = written & pred_ok
    else:
        kernel_written = written
    cb = torch.div(c.clamp(min=0), unit, rounding_mode="floor")
    cb = torch.where(c >= 0, cb, torch.full_like(cb, -1))
    match = c == exp
    mism = active & kernel_written & (exp >= 0) & ~match
    # writes the expectation does not call for (unallocated column, non-completing pool,
    # null tail block, tail token outside the seeded set) -- including null-block aliasing
    unexp = active & kernel_written & (exp < 0)
    if want_extra is not None:
        unexp = unexp | (active & kernel_written & (exp >= 0) & match & ~want)
    missing_pred = (want & written & match & ~kernel_written) if pred_ok is not None \
        else torch.zeros_like(want)
    missing = want & ~written
    # pool_span > 0 (index-K prefill compress): the kernel pools batch tokens t-span..t; they
    # must be this request's own tokens (positions p-span..p), never the previous row's
    if pool_span:
        pool_cross = active & kernel_written & (exp >= 0) & ((t - pool_span) < ev.qsl[row])
        if pool_mask is not None:
            pool_cross = pool_cross & pool_mask
    else:
        pool_cross = torch.zeros_like(want)
    inact = ~active & kernel_written & (cb != 0) & (t >= 0)
    bad_blk = mism | unexp | inact
    bits = owner_bits(ev, geo, cb, row, group)
    shared = active & kernel_written & match & (exp >= 0) & (ev.tagx[cb.clamp(0, geo.nb)] == -2)
    spec = active & ((t - ev.qsl[row]) >= (ev.nsched[row] - ev.nspec[row]))
    flags = (mism.to(torch.int64) * B["mism"] + missing.to(torch.int64) * B["missing"]
             + missing_pred.to(torch.int64) * B["pred"] + unexp.to(torch.int64) * B["unexp"]
             + inact.to(torch.int64) * B["inact"] + shared.to(torch.int64) * B["shared"]
             + pool_cross.to(torch.int64) * B["pcross"]
             + torch.where(bad_blk, bits, torch.zeros_like(bits)))
    viol = (mism | missing | missing_pred | pool_cross | unexp | inact | shared) & (armed != 0)
    flags = flags + spec.to(torch.int64) * B["spec"]
    good = active & kernel_written & match & (exp >= 0)
    counts = dict(checked=active, written=active & kernel_written, spec=spec & kernel_written,
                  high=good & (hi_blk >= geo.high), mism=mism, mism_spec=mism & spec,
                  missing=missing, missing_pred=missing_pred, pool_cross=pool_cross, unexp=unexp,
                  inact=inact, shared=shared)
    counts.update(_class_counts(bits, bad_blk))
    _acc(dev, site, layer, armed, counts)
    rows = torch.stack([
        torch.full_like(t, site), torch.full_like(t, rank), torch.full_like(t, layer),
        ev.step.expand_as(t), t, row, ev.tag[row], p, c, exp, flags,
        ev.tagx[cb.clamp(0, geo.nb)] * 64 + (ev.grp[cb.clamp(0, geo.nb)] + 4)], dim=1)
    record_rows(dev, viol, rows)


def check_positions(dev: Dev, rank: int, layer: int, positions: torch.Tensor,
                    qsl_consumed: torch.Tensor | None, write_ring: bool = True) -> None:
    """Positions consumed by the forward vs scheduler state; consumed query_start_loc vs the
    expected row split; p0 per row into the device ring for the exact offline join."""
    geo, ev = dev.geo, dev.ev
    R = geo.R
    armed = ev.armed
    n = positions.shape[0]
    t = torch.arange(n, device=positions.device, dtype=torch.int64)
    p = positions.to(torch.int64)
    row = rows_of(ev, t, R)
    active = (t < ev.ntok) & (row < ev.nrows)
    start = ev.qsl[row]
    first_idx = start.clamp(0, max(n - 1, 0))
    p_first = p[first_idx]
    contig = active & (p != p_first + (t - start))
    is_first = active & (t == start)
    bounds = is_first & ((p < ev.lo[row]) | (p > ev.hi[row]))
    flags = contig.to(torch.int64) * B["posc"] + bounds.to(torch.int64) * B["posb"]
    counts = dict(checked=active, pos_contig=contig, pos_bounds=bounds)
    rr = torch.arange(R, device=positions.device, dtype=torch.int64)
    rvalid = rr < ev.nrows
    if qsl_consumed is not None:
        m = min(int(qsl_consumed.shape[0]), R + 1)
        q = qsl_consumed[:m].to(torch.int64)
        qe = ev.qsl[:m]
        ri = torch.arange(m, device=positions.device, dtype=torch.int64)
        qbad = (ri <= ev.nrows) & (q != qe) & (armed != 0)
        counts["qsl"] = qbad
        qrows = torch.stack([
            torch.full_like(ri, S_POS), torch.full_like(ri, rank), torch.full_like(ri, layer),
            ev.step.expand_as(ri), torch.full_like(ri, -1), ri, ev.tag[ri.clamp(0, R - 1)],
            torch.full_like(ri, -1), q, qe, torch.full_like(ri, B["qsl"]), torch.full_like(ri, 0)],
            dim=1)
        record_rows(dev, qbad, qrows)
    _acc(dev, S_POS, layer, armed, counts)
    rows = torch.stack([
        torch.full_like(t, S_POS), torch.full_like(t, rank), torch.full_like(t, layer),
        ev.step.expand_as(t), t, row, ev.tag[row], p, p_first, ev.hi[row], flags, ev.lo[row]],
        dim=1)
    record_rows(dev, (contig | bounds) & (armed != 0), rows)
    if write_ring:
        p0 = torch.where(rvalid, p[ev.qsl[:R].clamp(0, max(n - 1, 0))], torch.full_like(rr, -1))
        slot = torch.remainder(ev.step, RING).reshape(1)
        on = armed != 0
        dev.ring_p0.index_copy_(0, slot, torch.where(on, p0, torch.full_like(p0, -1)).reshape(1, R))
        dev.ring_step.index_copy_(0, slot, torch.where(on, ev.step, torch.full_like(ev.step, -1)).reshape(1))


def check_table_read(dev: Dev, site: int, layer: int, rank: int, table: torch.Tensor,
                     lens: torch.Tensor, tok: torch.Tensor, active_extra: torch.Tensor | None,
                     exp_len: torch.Tensor, per_block: int) -> None:
    """Index-K page-table read: row i serves batch token tok[i]; it reads page columns
    c < cdiv(lens[i], ipage) (lens in pools).  Column c must be the request's own page
    17 * blk[c // 17] + c % 17 (index-page units; see Geo.ipage).  `per_block` is kept for
    the call signature (pools per block, 1088)."""
    geo, ev = dev.geo, dev.ev
    R, W0 = geo.R, geo.W0
    ipb, ipage = geo.ipb, geo.ipage
    armed = ev.armed
    t = tok.to(torch.int64)
    row = rows_of(ev, t, R)
    active = (t < ev.ntok) & (t >= 0) & (row < ev.nrows)
    if active_extra is not None:
        active = active & active_extra
    L = lens.to(torch.int64)
    need_cols = torch.div(L.clamp(min=0) + ipage - 1, ipage, rounding_mode="floor")
    W = int(table.shape[1])
    Wc = min(W, W0 * ipb)
    cols = torch.arange(Wc, device=table.device, dtype=torch.int64)
    tab = table[:, :Wc].to(torch.int64)
    need = active[:, None] & (cols[None, :] < need_cols[:, None])
    bcol = torch.div(cols, ipb, rounding_mode="floor")
    expb = ev.bt0[row][:, bcol] * ipb + torch.remainder(cols, ipb)[None, :]
    colv = bcol[None, :] < ev.nb0[row][:, None]
    ok = colv & (tab == expb)
    bad = need & ~ok
    beyond = active & (need_cols > Wc)          # needs columns the check cannot see
    rlen = active & (L != exp_len.to(torch.int64))
    inact_read = ~active & (L > 0) & (t >= 0)
    rowbad = bad.any(dim=1) | beyond
    first_bad = torch.argmax(bad.to(torch.int8), dim=1)
    bad_val = tab.gather(1, first_bad[:, None]).squeeze(1)
    bits = owner_bits(ev, geo, torch.div(bad_val, ipb, rounding_mode="floor"), row, geo.g_attn)
    high = (need & ok & (tab >= geo.high * ipb)).sum(dim=1)
    flags = (rowbad.to(torch.int64) * B["rtable"] + rlen.to(torch.int64) * B["rlen"]
             + torch.where(rowbad, bits, torch.zeros_like(bits)))
    counts = dict(checked=active, rtable=rowbad, rlen=rlen, inact_read=inact_read, high=high)
    counts.update(_class_counts(bits, rowbad))
    _acc(dev, site, layer, armed, counts)
    rows = torch.stack([
        torch.full_like(t, site), torch.full_like(t, rank), torch.full_like(t, layer),
        ev.step.expand_as(t), t, row, ev.tag[row], L, bad_val, exp_len.to(torch.int64), flags,
        first_bad.to(torch.int64)], dim=1)
    record_rows(dev, (rowbad | rlen) & (armed != 0), rows)


def check_selected(dev: Dev, layer: int, rank: int, sel: torch.Tensor, nsa: torch.Tensor,
                   pos: torch.Tensor, tok: torch.Tensor) -> None:
    """MLA sparse read: sel[t, j] (j < nsa[t]) are the physical token slots the kernel reads
    for token tok[t] at position pos[t].  Each must lie in the request's own group-0 blocks
    at a logical position <= pos[t]; entries j >= nsa[t] must be masked (-1)."""
    geo, ev = dev.geo, dev.ev
    R, W0 = geo.R, geo.W0
    armed = ev.armed
    n, K = int(sel.shape[0]), int(sel.shape[1])
    tile = max(1, SEL_TILE)
    _acc_exec(dev, S_MLA_R, layer, armed)
    for a in range(0, n, tile):
        b_ = min(n, a + tile)
        s = sel[a:b_].to(torch.int64)
        L = nsa[a:b_].to(torch.int64)
        p = pos[a:b_].to(torch.int64)
        t = tok[a:b_].to(torch.int64)
        row = rows_of(ev, t, R)
        active = (t < ev.ntok) & (row < ev.nrows)
        (live, valid, blk, held, logical, act, sel_foreign, sel_causal, sel_hole, masked_live,
         sel_inact, sel_len, high) = _sel_masks(dev, s, L, p, row, active, K)
        counts = dict(checked=active, sel_foreign=sel_foreign, sel_causal=sel_causal,
                      sel_hole=sel_hole, masked_live=masked_live, inact_read=sel_inact,
                      sel_len=sel_len, high=high)
        _acc_counts(dev, S_MLA_R, armed, counts)
        badm = sel_foreign | sel_causal | masked_live
        rowbad = badm.any(dim=1) | sel_len
        jb = torch.argmax(badm.to(torch.int8), dim=1)
        sv = s.gather(1, jb[:, None]).squeeze(1)
        bits = owner_bits(ev, geo, torch.div(sv.clamp(min=0), geo.bs, rounding_mode="floor"),
                          row, geo.g_attn)
        fl = (sel_foreign.any(dim=1).to(torch.int64) * B["self"]
              + sel_causal.any(dim=1).to(torch.int64) * B["selc"]
              + masked_live.any(dim=1).to(torch.int64) * B["maskl"]
              + sel_len.to(torch.int64) * B["sellen"]
              + torch.where(sel_foreign.any(dim=1), bits, torch.zeros_like(bits)))
        rows = torch.stack([
            torch.full_like(t, S_MLA_R), torch.full_like(t, rank), torch.full_like(t, layer),
            ev.step.expand_as(t), t, row, ev.tag[row], p, sv,
            logical.gather(1, jb[:, None]).squeeze(1), fl, jb * 4096 + L.clamp(max=4095)], dim=1)
        record_rows(dev, rowbad & (armed != 0), rows)

        # ---- 22ao: signed consumed count, and shadow plants for the selection checker ----
        expn = torch.minimum(p + 1, torch.full_like(p, K))
        dn = L - expn
        _acc_counts(dev, S_MLA_R, armed, dict(nsa_pos=active & (dn > 0), nsa_neg=active & (dn < 0)))
        if layer == S.first_layer:
            _hist(dev.nsahist, dn, active, armed)
        _plants(dev, s, L, p, t, row, active, K, armed)


def _sel_masks(dev: Dev, s: torch.Tensor, L: torch.Tensor, p: torch.Tensor, row: torch.Tensor,
               active: torch.Tensor, K: int):
    """check_selected's predicates (extracted unchanged; 22ao) so the shadow plants run the same code."""
    geo, ev = dev.geo, dev.ev
    W0 = geo.W0
    j = torch.arange(K, device=s.device, dtype=torch.int64)
    live = j[None, :] < L[:, None]
    valid = s >= 0
    blk = torch.div(s.clamp(min=0), geo.bs, rounding_mode="floor")
    bclamp = blk.clamp(0, geo.nb)
    c0 = ev.col0[bclamp]
    rr = row[:, None].expand_as(c0)
    held = (c0 >= 0) & (c0 < ev.nb0[row][:, None]) & \
        (ev.bt0[rr, c0.clamp(0, W0 - 1)] == blk) & (blk < geo.nb)
    logical = c0 * geo.bs + torch.remainder(s.clamp(min=0), geo.bs)
    act = active[:, None]
    sel_foreign = act & live & valid & ~held
    sel_causal = act & live & valid & held & (logical > p[:, None])
    sel_hole = act & live & ~valid
    masked_live = ~live & valid
    sel_inact = ~act & live & valid & (blk != 0)
    sel_len = active & (L > p + 1)
    high = act & live & valid & held & (blk >= geo.high)
    return (live, valid, blk, held, logical, act, sel_foreign, sel_causal, sel_hole, masked_live,
            sel_inact, sel_len, high)


def _hist(buf: torch.Tensor, d: torch.Tensor, mask: torch.Tensor, armed: torch.Tensor) -> None:
    b = d.clamp(-(HBINS // 2), HBINS // 2) + HBINS // 2
    buf.index_add_(0, b, mask.to(torch.int64) * armed)


def _row_holds(ev: EV, geo: Geo, row: torch.Tensor, blk: torch.Tensor) -> torch.Tensor:
    """Independent ground truth (22ao plants): does batch row `row` hold block `blk` in group 0?
    A full scan of the row's expected table (columns < nb0), not the col0 map the checker uses."""
    cols = torch.arange(geo.W0, device=blk.device, dtype=torch.int64)
    tab = ev.bt0[row]
    inrow = cols[None, :] < ev.nb0[row][:, None]
    return ((tab == blk[:, None]) & inrow).any(dim=1)


def _plants(dev: Dev, s: torch.Tensor, L: torch.Tensor, p: torch.Tensor, t: torch.Tensor,
            row: torch.Tensor, active: torch.Tensor, K: int, armed: torch.Tensor) -> None:
    """22ao shadow plants on guard-owned copies; never engine memory."""
    geo, ev = dev.geo, dev.ev
    W0, bs = geo.W0, geo.bs
    plant = active & (torch.remainder(ev.step * 131 + t, PLANT_MOD) == 0)
    j = torch.arange(K, device=s.device, dtype=torch.int64)
    col0 = (j == 0)[None, :]
    has0 = s[:, 0] >= 0
    # (a) same-request future key at logical q = p + 1 (q's column exists in the row's table)
    q = p + 1
    cq = torch.div(q, bs, rounding_mode="floor")
    blkq = ev.bt0[row, cq.clamp(0, W0 - 1)]
    oka = plant & (L >= 1) & (cq < ev.nb0[row]) & (blkq > 0) & _row_holds(ev, geo, row, blkq)
    sa = torch.where(col0 & oka[:, None], (blkq * bs + torch.remainder(q, bs))[:, None], s)
    ma = _sel_masks(dev, sa, L, p, row, active, K)
    deta = oka & ma[7][:, 0]
    # (b) foreign: the next batch row's first block, when this row does not hold it
    nr = ev.nrows.clamp(min=1)
    orow = torch.remainder(row + 1, nr)
    blkf = ev.bt0[orow, 0]
    okb = plant & (L >= 1) & (ev.nrows > 1) & (blkf > 0) & (blkf < geo.nb) & \
        ~_row_holds(ev, geo, row, blkf)
    sb = torch.where(col0 & okb[:, None], (blkf * bs)[:, None], s)
    mb = _sel_masks(dev, sb, L, p, row, active, K)
    detb = okb & mb[6][:, 0]
    # (c) over-long count on an exact dense row (L == p + 1 < K): planted L + 1
    okc = plant & (L == p + 1) & (p + 1 < K)
    Lc = torch.where(okc, L + 1, L)
    mc = _sel_masks(dev, s, Lc, p, row, active, K)
    detc = okc & mc[11]
    # (d) unmask column L with column 0's valid slot (L < K)
    okd = plant & (L >= 1) & (L < K) & has0
    colL = j[None, :] == L[:, None]
    sd = torch.where(colL & okd[:, None], s[:, 0:1].expand_as(s), s)
    md = _sel_masks(dev, sd, L, p, row, active, K)
    detd = okd & (md[9] & colL).any(dim=1)
    _acc_counts(dev, S_MLA_R, armed, dict(pa_planted=oka, pa_det=deta, pb_planted=okb, pb_det=detb,
                                          pc_planted=okc, pc_det=detc, pd_planted=okd, pd_det=detd))


def check_len(dev: Dev, layer: int, rank: int, Lc: torch.Tensor, pos: torch.Tensor,
              tok: torch.Tensor, width: int, record: bool) -> None:
    """22ao: the consumed per-token causal length against the authoritative p + 1 (signed)."""
    ev = dev.ev
    R = dev.geo.R
    armed = ev.armed
    _acc_exec(dev, S_MLA_LEN, layer, armed)
    Lv = Lc.to(torch.int64)
    p = pos.to(torch.int64)
    t = tok
    row = rows_of(ev, t, R)
    active = (t < ev.ntok) & (row < ev.nrows)
    d = Lv - (p + 1)
    _acc_counts(dev, S_MLA_LEN, armed, dict(len_checked=active, len_pos=active & (d > 0),
                                             len_neg=active & (d < 0)))
    if record:
        _hist(dev.lenhist, d, active, armed)
        first = active & (t == ev.qsl[row])
        dense = ((p + 1) < int(width)).to(torch.int64)
        rows = torch.stack([ev.step.expand_as(t), row, ev.tag[row], p, Lv, d, dense,
                            ev.qsl[(row + 1).clamp(max=R)] - ev.qsl[row]], dim=1)
        _append(dev.lrec, dev.nlrec, LCAP, first & (armed != 0), rows)


# ------------------------------------------------------------------ host side: ledger
def fingerprint(so) -> str:
    """Step key computed identically from the same SchedulerOutput in EngineCore and workers:
    scheduled counts, finished ids and the scheduler's num_computed per scheduled request
    (changes every step for a running request, so decode streaks do not collide)."""
    nc = [(str(nr.req_id), int(nr.num_computed_tokens)) for nr in so.scheduled_new_reqs]
    cr = so.scheduled_cached_reqs
    nc += [(str(r), int(c)) for r, c in zip(cr.req_ids, cr.num_computed_tokens)]
    key = json.dumps([sorted((str(k), int(v)) for k, v in so.num_scheduled_tokens.items()),
                      sorted(str(x) for x in (so.finished_req_ids or ())), sorted(nc)])
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def lists_digest(items) -> str:
    """items: iterable of (req_id, g0 list, g1 list) -- same function on both sides."""
    h = hashlib.sha1()
    for rid, g0, g1 in sorted(items, key=lambda x: str(x[0])):
        h.update(f"{rid}|{','.join(map(str, g0))}|{','.join(map(str, g1))};".encode())
    return h.hexdigest()[:16]


class Ledger:
    """Request -> per-group block lists, rebuilt from SchedulerOutput only."""

    def __init__(self, geo: Geo):
        self.geo = geo
        self.reqs: dict[str, dict] = {}
        self.next_tag = 1
        self.version = 0
        self.step = 0
        self.freed: dict[int, tuple] = {}
        self.events: list = []
        self.cur: dict[str, dict] = {}
        self._maps_version = -1
        self._maps = None

    def _new(self, rid, blocks):
        tag = self.next_tag
        self.next_tag += 1
        st = dict(tag=tag, blocks=[list(map(int, g)) for g in blocks], last=-10, k1=0,
                  warm=str(rid).startswith("_warmup"), born=self.step)
        self.reqs[rid] = st
        self.version += 1
        return st

    def _free(self, rid):
        st = self.reqs.pop(rid, None)
        if st is None:
            return
        self.version += 1
        held = set()
        for o in self.reqs.values():
            for g in o["blocks"]:
                held.update(g)
        for g in st["blocks"]:
            for b in g:
                if b and b not in held:
                    self.freed[b] = (st["tag"], self.step)
        self.events.append(("free", rid, st["tag"]))

    def update(self, so) -> dict:
        """Apply one SchedulerOutput; returns per-scheduled-request state for this step."""
        self.step += 1
        self.events = []
        gone = set(so.finished_req_ids or ())
        if getattr(so, "preempted_req_ids", None):
            gone |= set(so.preempted_req_ids)
        for rid in gone:
            self._free(rid)
        nc = {}
        for nr in so.scheduled_new_reqs:
            if nr.req_id in self.reqs:
                self._free(nr.req_id)
            st = self._new(nr.req_id, nr.block_ids)
            nc[nr.req_id] = int(nr.num_computed_tokens)
            self.events.append(("new", nr.req_id, st["tag"], int(nr.num_computed_tokens),
                                len(nr.prompt_token_ids or ())))
        cr = so.scheduled_cached_reqs
        resumed = set(getattr(cr, "resumed_req_ids", ()) or ())
        for i, rid in enumerate(cr.req_ids):
            st = self.reqs.get(rid)
            nb = cr.new_block_ids[i]
            if st is None:
                st = self._new(rid, nb if nb is not None else [[] for _ in range(10)])
                self.events.append(("unknown_cached", rid, st["tag"]))
            elif rid in resumed and nb is not None:
                st["blocks"] = [list(map(int, g)) for g in nb]
                self.version += 1
            elif nb is not None:
                if any(len(g) for g in nb):
                    for g, ids in enumerate(nb):
                        if g < len(st["blocks"]):
                            st["blocks"][g].extend(int(x) for x in ids)
                        else:
                            st["blocks"].append([int(x) for x in ids])
                    self.version += 1
            nc[rid] = int(cr.num_computed_tokens[i])
        self.cur = {}
        spec = so.scheduled_spec_decode_tokens or {}
        for rid, n in so.num_scheduled_tokens.items():
            st = self.reqs.get(rid)
            if st is None:
                st = self._new(rid, [[] for _ in range(10)])
                self.events.append(("unknown_sched", rid, st["tag"]))
            k = len(spec.get(rid, ()))
            hi = nc.get(rid, st.get("nc", 0))
            # async scheduling: the previous step's drafts of this request may still be
            # rejected when this step runs (only if it was scheduled in the previous step)
            lo = hi - (st["k1"] if st["last"] == self.step - 1 else 0)
            st.update(nc=hi, nsched=int(n), nspec=k, lo=lo, hi=hi)
            st["k1"] = k
            st["last"] = self.step
            self.cur[rid] = st
        return self.cur

    def digest(self, rids) -> str:
        ga, gt = self.geo.g_attn, self.geo.g_tail
        items = []
        for rid in rids:
            st = self.reqs.get(rid)
            if st is None:
                continue
            bl = st["blocks"]
            items.append((rid, bl[ga] if ga < len(bl) else [], bl[gt] if gt < len(bl) else []))
        return lists_digest(items)

    def maps(self):
        """Block -> (column in group 0, holder tag (-2 shared), group (-1 free))."""
        if self._maps_version == self.version and self._maps is not None:
            return self._maps
        NB1 = self.geo.nb + 1
        col0 = np.full(NB1, -1, np.int64)
        tagx = np.zeros(NB1, np.int64)
        grp = np.full(NB1, -1, np.int64)
        conflicts = 0
        order = [self.geo.g_attn, self.geo.g_tail]
        # pass 1: attention and tail groups (append-only lists: exact).  pass 2: the other
        # groups only fill ids still free -- their ledger lists can be stale (sliding-window
        # and mamba-align managers release blocks without a SchedulerOutput delta), so they
        # are used for classification only, never for an expectation.
        for pas in (0, 1):
            for st in self.reqs.values():
                ng = len(st["blocks"])
                groups = order if pas == 0 else [x for x in range(ng) if x not in order]
                for g in groups:
                    if g >= ng:
                        continue
                    for c, b in enumerate(st["blocks"][g]):
                        if b <= 0 or b >= NB1:
                            continue
                        if grp[b] == -1:
                            grp[b] = g
                            tagx[b] = st["tag"]
                            if g == self.geo.g_attn:
                                col0[b] = c
                        elif pas == 0 and grp[b] == g:
                            if tagx[b] != st["tag"]:
                                tagx[b] = -2
                        elif pas == 0:
                            conflicts += 1
        self._maps = (col0, tagx, grp, conflicts)
        self._maps_version = self.version
        return self._maps


def build_expected(lay: ELayout, ledger: Ledger, req_ids, step: int, out: np.ndarray | None = None):
    """Fill the flat expectation array for one step (batch rows in runner order)."""
    geo = lay.geo
    a = out if out is not None else np.zeros(lay.size, np.int64)
    a[:] = 0
    v = lay.np_views(a)
    R, W0 = geo.R, geo.W0
    n = len(req_ids)
    host_issues = []
    if n > R:
        host_issues.append(("rows_gt_R", n))
        n = R
    q = 0
    v["qsl"][0] = 0
    warm = 0
    for r in range(n):
        rid = req_ids[r]
        st = ledger.cur.get(rid) or ledger.reqs.get(rid)
        if st is None:
            host_issues.append(("unknown_row", rid))
            v["qsl"][r + 1] = q
            continue
        warm |= int(st["warm"])
        q += int(st.get("nsched", 0))
        v["qsl"][r + 1] = q
        v["tag"][r] = st["tag"]
        v["lo"][r], v["hi"][r] = st.get("lo", 0), st.get("hi", 0)
        v["nspec"][r], v["nsched"][r] = st.get("nspec", 0), st.get("nsched", 0)
        bl = st["blocks"]
        g0 = bl[geo.g_attn] if geo.g_attn < len(bl) else []
        if len(g0) > W0:
            host_issues.append(("blocks_gt_W0", rid, len(g0)))
        m = min(len(g0), W0)
        v["bt0"][r, :m] = g0[:m]
        v["nb0"][r] = m
        g1 = bl[geo.g_tail] if geo.g_tail < len(bl) else []
        v["bt1"][r] = g1[0] if len(g1) else 0
        if len(g1) > 1:
            host_issues.append(("tail_gt_1", rid, len(g1)))
    v["qsl"][n + 1:] = q
    col0, tagx, grp, conflicts = ledger.maps()
    v["col0"][:] = col0
    v["tagx"][:] = tagx
    v["grp"][:] = grp
    h = v["hdr"]
    h[H_ARMED] = 1
    h[H_STEP] = step
    h[H_NROWS] = n
    h[H_NTOK] = q
    h[H_WARM] = warm
    h[H_VERSION] = ledger.version
    if conflicts:
        host_issues.append(("group01_conflicts", conflicts))
    return a, host_issues


# ------------------------------------------------------------------ process state
class _State:
    def __init__(self):
        self.runner = None
        self.dev: Dev | None = None
        self.geo: Geo | None = None
        self.ledger: Ledger | None = None
        self.rank = -1
        self.step = 0
        self.fh = None
        self.errors = 0
        self.logged_once = set()
        self.impl_layer: dict[int, int] = {}
        self.owned_layers: list[int] = []
        self.first_layer = -1
        self.pos_buf = None
        self.qsl_buf = None
        self.staging = None
        self.events = None
        self.sidx = 0
        self.armed_host = False
        self.pending = None
        self.mode = "?"
        self.poller = None
        self.side = None
        self.emitted = 0
        self.ring_seen = -1
        self.row_hist: dict[int, str] = {}
        self.row_reuse = 0
        self.uploads = 0          # 22ao
        self.blocked_waits = 0    # 22ao
        self.poll_lock = threading.Lock()   # 22ap: engine-thread baseline and poller thread poll one at a time
        self.sched = None
        self.cur_fp = None
        self.baseline_done = False
        self.d7_pending = None
        self.slog_emitted = 0
        self.bsnap_emitted = 0
        self.lrec_emitted = 0


S = _State()


def _capturing() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False


def _log(rec: dict, prefix: str = "mg") -> None:
    """mg_p<pid>.jsonl: worker-side records; ms_p<pid>.jsonl: EngineCore scheduler side."""
    try:
        if S.fh is None:
            S.fh = {}
        fh = S.fh.get(prefix)
        if fh is None:
            os.makedirs(OUT_DIR, exist_ok=True)
            fh = S.fh[prefix] = open(os.path.join(OUT_DIR, f"{prefix}_p{os.getpid()}.jsonl"), "a")
        rec.setdefault("t", round(time.time(), 4))
        rec.setdefault("rank", S.rank)
        fh.write(json.dumps(rec, default=str) + "\n")
        fh.flush()
    except Exception:  # noqa: BLE001
        S.errors += 1


def _err(where: str, exc: Exception) -> None:
    S.errors += 1
    if S.errors <= 50:
        _log(dict(kind="guard_error", where=where, err=repr(exc)[:400]))


def _layer_idx(name) -> int:
    try:
        parts = str(name).split(".")
        return int(parts[parts.index("layers") + 1])
    except Exception:  # noqa: BLE001
        return -1


def mla_owner_layers(spec: str, mla_layers, rank: int) -> list[int]:
    """Independent owner rule (placement.py _parse: split:N -> int(layer >= N))."""
    spec = (spec or "tp").strip()
    if spec.startswith("split:"):
        bnd = int(spec.split(":", 1)[1])
        return [L for L in mla_layers if int(L >= bnd) == rank]
    if spec.startswith("map:"):
        m = {int(a): int(b) for a, b in (x.split("=") for x in spec.split(":", 1)[1].split(","))}
        return [L for L in mla_layers if m.get(L) == rank]
    return list(mla_layers)


# ------------------------------------------------------------------ worker registration
def _register_runner(runner) -> None:
    from vllm.distributed import get_tensor_model_parallel_rank
    S.runner = runner
    S.rank = int(get_tensor_model_parallel_rank())
    cfg = runner.kv_cache_config
    groups = cfg.kv_cache_groups
    names = [type(g.kv_cache_spec).__name__ for g in groups]
    bt = runner.block_tables
    g_attn = 0
    g_tail = next((i for i, g in enumerate(groups) if "KpoolTail" in type(g.kv_cache_spec).__name__
                   or any("KpoolTail" in type(s).__name__ for s in
                          getattr(g.kv_cache_spec, "kv_cache_specs", {}).values())), 1)
    bpk = bt.blocks_per_kv_block[g_attn]
    W0 = int(bt.block_tables[g_attn].gpu.shape[1]) // max(bpk, 1)
    geo = Geo(nb=int(cfg.num_blocks), R=int(runner.max_num_reqs), W0=W0, g_attn=g_attn,
              g_tail=g_tail)
    issues = []
    if int(bt.block_sizes[g_attn]) != geo.bs or int(bt.kernel_block_sizes[g_attn]) != geo.kbs:
        issues.append(("attn_geometry", bt.block_sizes[g_attn], bt.kernel_block_sizes[g_attn]))
    if int(bt.block_sizes[g_tail]) != geo.ring:
        issues.append(("tail_geometry", bt.block_sizes[g_tail]))
    try:   # index-K page (pools) as the builder derives it (worker/utils.py:302-311)
        from vllm.v1.worker.utils import compressed_kernel_block_size
        specs = getattr(groups[g_attn].kv_cache_spec, "kv_cache_specs", {}) or {}
        comp = [sp for sp in specs.values() if getattr(sp, "compress_ratio", 1) > 1]
        if comp:
            pg = int(compressed_kernel_block_size(comp[0]))
            if pg != geo.ipage or int(comp[0].storage_block_size) != geo.ppb:
                issues.append(("index_page", pg, int(comp[0].storage_block_size)))
        else:
            issues.append(("index_spec_not_found", [type(v).__name__ for v in specs.values()][:4]))
    except Exception as exc:  # noqa: BLE001
        issues.append(("index_page_check_error", repr(exc)[:120]))
    S.geo = geo
    S.ledger = Ledger(geo)
    S.dev = Dev(geo, runner.device)
    S.pos_buf = runner.input_buffers.positions
    S.qsl_buf = runner.input_buffers.query_start_loc
    pin = torch.cuda.is_available() and S.dev.E.is_cuda
    S.staging = [torch.zeros(S.dev.lay.size, dtype=torch.int64).pin_memory() if pin else
                 torch.zeros(S.dev.lay.size, dtype=torch.int64) for _ in range(4)]
    S.events = [None] * 4
    # MLA impl -> layer index; owned layers on this rank
    try:
        from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseImpl
        for name, mod in runner.model.named_modules():
            impl = getattr(mod, "impl", None)
            if isinstance(impl, B12xMLASparseImpl):
                S.impl_layer[id(impl)] = _layer_idx(name)
    except Exception as exc:  # noqa: BLE001
        _err("impl_map", exc)
    S.owned_layers = sorted(set(S.impl_layer.values()))
    S.first_layer = S.owned_layers[0] if S.owned_layers else -1
    hf = runner.vllm_config.model_config.hf_text_config
    try:
        mla_layers = [i for i in range(hf.num_hidden_layers) if not hf.is_kda_layer(i)]
    except Exception:  # noqa: BLE001
        mla_layers = []
    owners = mla_owner_layers(os.environ.get("VLLM_GLM53_MLA_OWNERS", "tp"), mla_layers, S.rank)
    _log(dict(kind="init", mode=MODE, geo=geo.as_dict(), groups=names, issues=issues,
              block_sizes=list(bt.block_sizes), kernel_block_sizes=list(bt.kernel_block_sizes),
              owned_layers_impl=S.owned_layers, owned_layers_rule=owners,
              owners_env=os.environ.get("VLLM_GLM53_MLA_OWNERS"), mla_layers=mla_layers,
              async_scheduling=bool(getattr(runner.scheduler_config, "async_scheduling", None)),
              adaptive_verification=runner.adaptive_verification is not None
              if hasattr(runner, "adaptive_verification") else None,
              e_size=S.dev.lay.size, cap=CAP, ring=RING, pid=os.getpid(),
              b1_amendment=dict(sh=SH, scap=SCAP, bcap=BCAP, oracle_max=ORACLE_MAX,
                                skip_tail_env=os.environ.get("VLLM_KPOOL_SKIP_TAIL_CACHE"))))


def _upload(runner, input_batch) -> None:
    led, dev = S.ledger, S.dev
    S.step += 1
    k = S.sidx
    S.sidx = (k + 1) % len(S.staging)
    ev = S.events[k]
    if ev is not None:
        S.uploads += 1
        if not ev.query():                     # 22ao: this wait blocks the host (GPU >= 4 steps behind)
            S.blocked_waits += 1
        ev.synchronize()   # only waits if the GPU is >= 4 steps behind the host
    buf = S.staging[k].numpy()
    req_ids = list(input_batch.req_ids)
    _, issues = build_expected(dev.lay, led, req_ids, S.step, out=buf)
    dev.E.copy_(S.staging[k], non_blocking=True)
    if dev.E.is_cuda:
        e = torch.cuda.Event()
        e.record()
        S.events[k] = e
    S.armed_host = True
    # host-side attribution checks (runner counts vs scheduler counts)
    try:
        ns = np.asarray(input_batch.num_scheduled_tokens)[: len(req_ids)]
        exp_ns = np.array([led.cur.get(r, {}).get("nsched", -1) for r in req_ids])
        if ns.shape == exp_ns.shape and (ns != exp_ns).any():
            issues.append(("nsched_mismatch", int((ns != exp_ns).sum())))
        if int(input_batch.num_tokens) != int(exp_ns.sum()):
            issues.append(("ntok_mismatch", int(input_batch.num_tokens), int(exp_ns.sum())))
    except Exception as exc:  # noqa: BLE001
        issues.append(("host_check_error", repr(exc)[:120]))
    rows = []
    idxm = getattr(input_batch, "idx_mapping_np", None)
    for r, rid in enumerate(req_ids):
        st = led.cur.get(rid) or led.reqs.get(rid) or {}
        bl = st.get("blocks", [[], []])
        g0 = bl[led.geo.g_attn] if len(bl) > led.geo.g_attn else []
        g1 = bl[led.geo.g_tail] if len(bl) > led.geo.g_tail else []
        rrow = int(idxm[r]) if idxm is not None and r < len(idxm) else -1
        prev = S.row_hist.get(rrow)
        reuse = prev is not None and prev != rid
        if reuse:
            S.row_reuse += 1
        S.row_hist[rrow] = rid
        rows.append([str(rid), st.get("tag", 0), st.get("lo", 0), st.get("hi", 0), st.get("nsched", 0),
                     st.get("nspec", 0), rrow, len(g0), max(g0) if g0 else 0,
                     g1[0] if g1 else 0, int(reuse)])
    S.pending = dict(kind="step", step=S.step, sched_step=led.step, fp=S.cur_fp, rows=rows,
                     ntok=int(input_batch.num_tokens),
                     ntok_pad=int(getattr(input_batch, "num_tokens_after_padding", -1)),
                     nrows_pad=int(getattr(input_batch, "num_reqs_after_padding", -1)),
                     warm=int(any(led.reqs.get(r, {}).get("warm", False) for r in req_ids)),
                     dig=led.digest(req_ids), ver=led.version, issues=issues, ev=led.events)


def _maybe_baseline(so) -> None:
    """Exact readout just before the first forward that serves a real (non-_warmup_) request:
    separates the runner's own warm-up from serving traffic (post-inspection change after
    run-a2, whose only pre-traffic readout existed by chance).  One host sync at a step
    boundary, outside any capture; it changes no engine tensor or kernel input."""
    try:
        rids = list(so.num_scheduled_tokens)
        if rids and any(not str(r).startswith("_warmup") for r in rids):
            S.baseline_done = True
            if S.dev.cnt.is_cuda and not _capturing():
                torch.cuda.synchronize()
            poll("baseline")
    except Exception as exc:  # noqa: BLE001
        _err("baseline", exc)


def _ensure_poller() -> None:
    if S.poller is None and S.dev is not None:
        S.poller = threading.Thread(target=_poller_loop, name="local-mapguard-poll", daemon=True)
        S.poller.start()


def _alloc_pins(tensors, pin):
    """22ap: host copy buffers as normal tensors, whichever thread (and inference mode) allocates them."""
    with torch.inference_mode(False):
        return [torch.empty(t.shape, dtype=t.dtype, pin_memory=pin) for t in tensors]


def _snapshot():
    dev = S.dev
    # B1 amendment: log counters are copied BEFORE their buffers (side-stream copies run in order;
    # the engine stream writes a log row before it advances the counter)
    tensors = [dev.cnt, dev.rec, dev.nrec, dev.layer, dev.ring_step, dev.ring_p0,
               dev.nslog, dev.slog, dev.nbsnap, dev.bsnap, dev.bkey,
               dev.nlrec, dev.lrec, dev.lenhist, dev.nsahist]   # 22ao: counter before buffer
    if dev.cnt.is_cuda:
        if S.side is None:
            S.side = torch.cuda.Stream(device=dev.cnt.device)
            S.pins = _alloc_pins(tensors, pin=True)
        with torch.cuda.stream(S.side):
            for p_, t in zip(S.pins, tensors):
                p_.copy_(t, non_blocking=True)
            e = torch.cuda.Event()
            e.record(S.side)
        e.synchronize()
        return [p_.clone() for p_ in S.pins]
    return [t.clone() for t in tensors]


def poll(reason: str) -> None:
    if not S.poll_lock.acquire(timeout=60.0):   # 22ap: bounded, so the exit poll cannot hang shutdown
        _err("poll_lock", RuntimeError("poll lock not acquired in 60 s (" + reason + ")"))
        return
    try:
        _poll_locked(reason)
    finally:
        S.poll_lock.release()


def _poll_locked(reason: str) -> None:
    if S.dev is None:
        return
    try:
        cnt, rec, nrec, layer, rstep, rp0, nslog, slog, nbs, bsnap, bkey, nlrec, lrec, lhist, nhist = _snapshot()
        nl = min(int(nlrec[0]), LCAP)
        lnew = lrec[S.lrec_emitted:nl].tolist()
        S.lrec_emitted = nl
        ns = min(int(nslog[0]), SCAP)
        snew = slog[S.slog_emitted:ns].tolist()
        S.slog_emitted = ns
        nb = min(int(nbs[0]), BCAP)
        bnew = [dict(key=bkey[i].tolist(), expected=bsnap[i, 0].numpy().tobytes().hex(),
                     stored=bsnap[i, 1].numpy().tobytes().hex(), pre=bsnap[i, 2].numpy().tobytes().hex())
                for i in range(S.bsnap_emitted, nb)]
        S.bsnap_emitted = nb
        n = int(nrec[0])
        new = rec[S.emitted:min(n, CAP)].tolist()
        S.emitted = min(n, CAP)
        rs = rstep.tolist()
        ring = []
        for i, s in enumerate(rs):
            if s > S.ring_seen:
                ring.append([s] + rp0[i].tolist())
        if ring:
            S.ring_seen = max(r[0] for r in ring)
        _log(dict(kind="poll", reason=reason, host_step=S.step,
                  counters={SITES[i]: {CNT[j]: int(cnt[i, j]) for j in range(len(CNT))}
                            for i in range(len(SITES))},
                  n_records=n, records=[dict(zip(REC_COLS, r)) for r in new],
                  layers={str(L): {SITES[s]: [int(layer[L, s, 0]), int(layer[L, s, 1])]
                                   for s in range(len(SITES)) if int(layer[L, s, 0])}
                          for L in range(NLAYERS) if int(layer[L].sum())},
                  ring=sorted(ring), row_reuse=S.row_reuse, guard_errors=S.errors,
                  uploads=S.uploads, blocked_waits=S.blocked_waits,
                  n_slog=int(nslog[0]), slog=snew, n_bsnap=int(nbs[0]), bsnap=bnew,
                  n_lrec=int(nlrec[0]), lrec=lnew, lenhist=lhist.tolist(), nsahist=nhist.tolist()))
    except Exception as exc:  # noqa: BLE001
        _err("poll", exc)


def _poller_loop():
    try:
        if S.dev is not None and S.dev.cnt.is_cuda:
            torch.cuda.set_device(S.dev.cnt.device)
    except Exception:  # noqa: BLE001
        pass
    trig = os.path.join(OUT_DIR, "DUMP")
    last_trig, last = None, time.time()
    while True:
        time.sleep(2.0)
        try:
            t = os.path.getmtime(trig) if os.path.exists(trig) else None
        except OSError:
            t = None
        if t is not None and t != last_trig:
            last_trig = t
            poll("trigger")
            last = time.time()
        elif time.time() - last >= POLL_SECS:
            poll("periodic")
            last = time.time()


# ------------------------------------------------------------------ worker hook entry points
def _ok() -> bool:
    return ENABLED and S.dev is not None


def hook_mla_write(impl, kv_cache, slot_mapping) -> None:
    if not _ok() or kv_cache.numel() == 0:
        return
    try:
        layer = S.impl_layer.get(id(impl), -1)
        sm = slot_mapping.reshape(-1)
        n = int(sm.shape[0])
        tok = torch.arange(n, device=sm.device, dtype=torch.int64)
        pos = S.pos_buf[:n]
        check_writes(S.dev, S_MLA_W, layer, S.rank, "mla", sm, pos, tok)
        if layer == S.first_layer:
            check_positions(S.dev, S.rank, layer, pos, S.qsl_buf)
    except Exception as exc:  # noqa: BLE001
        _err("mla_write", exc)


def hook_idx_prefill_write(layer_name, slot_slice, positions, num_decode_tokens: int,
                           num_tokens: int, kpool: int, k=None, gate=None, ape=None, kv_cache=None,
                           round_scale: bool = True, ring_mode: str | None = None,
                           ring_tslot=None) -> None:
    """Index-K prefill write, BEFORE the writer runs.  ring_mode (B1 amendment) is what the writer
    receives: "ring" (ring_tslot = the writer's per-token tail slots) or "none"; None = legacy caller
    (frozen B0 predicate, counted `legacy`).  With k/gate/ape/kv_cache the straddling-pool content
    oracle is prepared (pre-write bytes snapshotted) and completed by hook_idx_prefill_post."""
    S.d7_pending = None
    if not _ok():
        return
    try:
        dev = S.dev
        ev = dev.ev
        n = int(slot_slice.shape[0])
        if n == 0:
            return
        layer = _layer_idx(layer_name)
        loc = torch.arange(n, device=slot_slice.device, dtype=torch.int64)
        tok = loc + int(num_decode_tokens)
        pos = positions[num_decode_tokens:num_decode_tokens + n]
        b0pred = loc >= (kpool - 1) if n >= kpool else torch.zeros_like(loc, dtype=torch.bool)
        if ring_mode is None:            # legacy call: frozen B0 behaviour, counted
            _acc_counts(dev, S_IDX_W_PF, ev.armed, dict(legacy=torch.ones_like(loc[:1], dtype=torch.bool)))
            check_writes(dev, S_IDX_W_PF, layer, S.rank, "idx", slot_slice, pos,
                         tok, pred_ok=b0pred, pool_span=kpool - 1)
            return
        if ring_mode == "ring" and ring_tslot is not None:
            has_ring = ring_tslot[:n].to(torch.int64) >= 0
        else:
            has_ring = torch.zeros_like(loc, dtype=torch.bool)
        # B1 write rule: with a ring every completing pool is written (fix: valid & all members own
        # or at position >= 0); without one, B0's rule and B0's batch gathering
        check_writes(dev, S_IDX_W_PF, layer, S.rank, "idx", slot_slice, pos, tok,
                     pred_ok=has_ring | b0pred, pool_span=kpool - 1, pool_mask=~has_ring)
        row, active, pl, exact, expv, want, straddle, p0 = _d7_rows(dev, tok)
        fb = straddle & ~has_ring
        armed = ev.armed
        on = armed != 0
        _acc_counts(dev, S_IDX_W_PF, armed, dict(d7_fallback=fb))
        rows = torch.stack([
            torch.full_like(tok, S_IDX_W_PF), torch.full_like(tok, S.rank), torch.full_like(tok, layer),
            ev.step.expand_as(tok), tok, row, ev.tag[row], pl, slot_slice.to(torch.int64), expv,
            torch.full_like(tok, B["fallback"]), torch.full_like(tok, -1)], dim=1)
        record_rows(dev, fb & on, rows)
        _slog(dev, layer, fb & on, row, pl, expv, torch.full_like(tok, D7_OUT["fallback"]))
        if k is None or gate is None or ape is None or kv_cache is None or _capturing():
            return
        if not (0 <= layer < NLAYERS) or k.dtype != torch.bfloat16 \
                or gate.dtype != torch.bfloat16 or int(k.shape[-1]) != 128:
            _acc_counts(dev, S_IDX_W_PF, armed, dict(s_events=straddle, s_unavail=straddle))
            _slog(dev, layer, straddle & on, row, pl, expv, torch.full_like(tok, D7_OUT["unavail"]))
            return
        S.d7_pending = dict(layer=layer, n=n, ndt=int(num_decode_tokens), k=k[:n], gate=gate[:n],
                            ape=ape, rs=bool(round_scale), tok=tok, row=row, active=active, pl=pl,
                            exact=exact, expv=expv, want=want, straddle=straddle, p0=p0,
                            pre=_rec_bytes(kv_cache, expv.clamp(min=0)))
    except Exception as exc:  # noqa: BLE001
        S.d7_pending = None
        _err("idx_prefill_write", exc)


def _d7_rows(dev: Dev, tok: torch.Tensor):
    """Ledger view of prefill tokens (B1 amendment): batch row, active, LEDGER logical position
    (scheduler num_computed + offset in the row; never the engine's positions), exact row (prefill:
    lo == hi and no drafts), expected pool slot of a completing pool (-1 otherwise), want (completing
    pool with an allocated column) and straddle (its first member precedes the chunk start)."""
    geo, ev = dev.geo, dev.ev
    R, W0, kp = geo.R, geo.W0, geo.kpool
    row = rows_of(ev, tok, R)
    active = (tok < ev.ntok) & (tok >= 0) & (row < ev.nrows)
    p0 = ev.hi[row]
    pl = p0 + (tok - ev.qsl[row])
    exact = active & (ev.lo[row] == ev.hi[row]) & (ev.nspec[row] == 0)
    plc = pl.clamp(min=0)
    col = torch.div(plc, geo.bs, rounding_mode="floor")
    colv = active & (pl >= 0) & (col < ev.nb0[row])
    eb = ev.bt0[row, col.clamp(0, W0 - 1)]
    complete = torch.remainder(plc + 1, kp) == 0
    expv = torch.where(colv & complete, eb * geo.ppb + torch.remainder(
        torch.div(plc, kp, rounding_mode="floor"), geo.ppb), torch.full_like(pl, -1))
    want = active & complete & (expv >= 0)
    straddle = want & ((pl - (kp - 1)) < p0)
    return row, active, pl, exact, expv, want, straddle, p0


def _rec_bytes(cache: torch.Tensor, loc: torch.Tensor) -> torch.Tensor:
    """[n, 132] bytes of index-K records at pool slots `loc`, addressed exactly as the compress kernel
    writes them (page = loc // page_size; K at page*stride0 + r*128; fp32 scale at
    page*stride0 + page_size*128 + 4r)."""
    ps, s0 = int(cache.shape[1]), int(cache.stride(0))
    size = (int(cache.shape[0]) - 1) * s0 + ps * int(cache.shape[2])
    flat = torch.as_strided(cache, (size,), (1,), cache.storage_offset())
    page = torch.div(loc, ps, rounding_mode="floor")
    r = torch.remainder(loc, ps)
    ko = (page * s0 + r * 128)[:, None] + torch.arange(128, device=loc.device, dtype=torch.int64)[None, :]
    so = (page * s0 + ps * 128 + r * 4)[:, None] + torch.arange(4, device=loc.device, dtype=torch.int64)[None, :]
    return flat[torch.cat([ko, so], dim=1)]


def _scratch(dev: Dev, like: torch.Tensor, n: int = 0) -> torch.Tensor:
    """Guard-owned oracle pages with the engine cache's page size, row width and page stride (so the
    compress kernel runs with the engine call's constexpr specialisation); >= max(n, ORACLE_MAX) records."""
    ps, w, s0 = int(like.shape[1]), int(like.shape[2]), int(like.stride(0))
    pages = max(int(n), ORACLE_MAX) // ps + 2
    key = (ps, w, s0, str(like.device))
    if dev.scratch is None or dev.scratch[0] != key or int(dev.scratch[1].shape[0]) < pages:
        raw = torch.zeros(pages * max(s0, ps * w), dtype=torch.uint8, device=like.device)
        dev.scratch = (key, torch.as_strided(raw, (pages, ps, w), (s0, int(like.stride(1)), int(like.stride(2)))))
    return dev.scratch[1]


def _oracle_compress(scratch, slot_k, slot_score, ape, loc, kpool, mask, round_scale) -> None:
    """The engine's compress kernel on the ORACLE's members, into guard scratch (GPU semantics)."""
    from vllm.models.glm5next.nvidia.ops.kpool_compress import kpool_compress_and_write_cache
    kpool_compress_and_write_cache(scratch, slot_k, slot_score, ape, loc, pool_size=kpool,
                                   head_dim=int(slot_k.shape[2]), write_mask=mask,
                                   round_scale=round_scale, write_cache=True, return_compressed=False)


def _slog(dev: Dev, layer: int, mask, row, pl, slot, outcome) -> None:
    ev = dev.ev
    t = row
    rows = torch.stack([ev.step.expand_as(t), torch.full_like(t, layer), ev.tag[row], pl, outcome,
                        torch.full_like(t, S.rank), row, slot.to(torch.int64)], dim=1)
    _append(dev.slog, dev.nslog, SCAP, mask, rows)


def hook_idx_prefill_post(layer_name, kv_cache) -> None:
    """Index-K prefill write, AFTER the writer: content oracle for every completing pool of an exact
    prefill row (straddling pools: outcome classes; aligned pools: positive control), then the
    shadow snapshot of each row's trailing incomplete pool (for its next chunk)."""
    P, S.d7_pending = S.d7_pending, None
    if not _ok() or P is None:
        return
    try:
        dev = S.dev
        ev, geo = dev.ev, dev.geo
        layer = _layer_idx(layer_name)
        if layer != P["layer"]:
            raise RuntimeError(f"post hook layer {layer} != pre hook layer {P['layer']}")
        n, kp, ndt = P["n"], geo.kpool, P["ndt"]
        armed = ev.armed
        on = armed != 0
        row, pl, p0, tok = P["row"], P["pl"], P["p0"], P["tok"]
        tag = ev.tag[row]
        offs = torch.arange(kp, device=pl.device, dtype=torch.int64)
        q = pl[:, None] - (kp - 1) + offs[None, :]                  # member logical positions
        from_sh = q < p0[:, None]                                    # before this chunk -> shadow
        li = (ev.qsl[row][:, None] + (q - p0[:, None]) - ndt).clamp(0, n - 1)   # own batch token
        base = layer * SH * kp
        shi = base + torch.remainder(tag, SH)[:, None] * kp + torch.remainder(q.clamp(min=0), kp)
        sh_ok = (dev.shtag[shi] == tag[:, None]) & (dev.shpos[shi] == q)
        avail = (~from_sh | sh_ok).all(dim=1)
        need = P["want"] & P["exact"]
        sel = from_sh[..., None]
        mk = torch.where(sel, dev.shk[shi], P["k"][li])
        mgt = torch.where(sel, dev.shg[shi], P["gate"][li])
        omask = need & avail
        scr = _scratch(dev, kv_cache, n)
        idx = torch.arange(n, device=pl.device, dtype=torch.int64)
        _oracle_compress(scr, mk.contiguous(), mgt.contiguous(), P["ape"], idx, kp, omask, P["rs"])
        orc = _rec_bytes(scr, idx)
        post = _rec_bytes(kv_cache, P["expv"].clamp(min=0))
        pre = P["pre"]
        eq = (orc == post).all(dim=1)
        unch = (post == pre).all(dim=1)
        preq = (pre == orc).all(dim=1)
        strad = P["straddle"]
        chk = strad & P["exact"] & avail
        s_unav = strad & ~(P["exact"] & avail)
        s_ver, s_ind = chk & eq & ~preq, chk & eq & preq
        s_mis, s_mm = chk & ~eq & unch, chk & ~eq & ~unch
        aligned = need & ~from_sh.any(dim=1)
        pc_mm = aligned & ~eq
        _acc_counts(dev, S_IDX_W_PF, armed, dict(
            s_events=strad, s_verified=s_ver, s_indet=s_ind, s_unavail=s_unav, s_missing=s_mis,
            s_mism=s_mm, pc_checked=aligned, pc_mism=pc_mm))
        code = torch.full_like(pl, D7_OUT["unavail"])
        for name, m in (("verified", s_ver), ("indet", s_ind), ("missing", s_mis), ("mism", s_mm)):
            code = torch.where(m, torch.full_like(code, D7_OUT[name]), code)
        _slog(dev, layer, strad & on, row, pl, P["expv"], code)
        bad = s_mis | s_mm
        flags = s_mis.to(torch.int64) * B["smiss"] + s_mm.to(torch.int64) * B["smism"]
        rows = torch.stack([
            torch.full_like(tok, S_IDX_W_PF), torch.full_like(tok, S.rank), torch.full_like(tok, layer),
            ev.step.expand_as(tok), tok, row, tag, pl, P["expv"], P["expv"], flags,
            (from_sh.to(torch.int64) * (1 << offs)[None, :]).sum(dim=1)], dim=1)
        record_rows(dev, bad & on, rows)
        snap = bad | pc_mm
        bcode = torch.where(s_mis, torch.full_like(code, 4), torch.where(s_mm, torch.full_like(code, 5),
                                                                         torch.full_like(code, 7)))
        sm = (snap & on).to(torch.int64)
        bi = dev.nbsnap[0] + torch.cumsum(sm, 0) - 1
        bt = torch.where((sm > 0) & (bi < BCAP), bi, torch.full_like(bi, BCAP))
        dev.bsnap.index_copy_(0, bt, torch.stack([orc, post, pre], dim=1))
        dev.bkey.index_copy_(0, bt, torch.stack([ev.step.expand_as(tok), torch.full_like(tok, layer), tag,
                                                 pl, bcode, P["expv"]], dim=1))
        dev.nbsnap.add_(sm.sum())
        # shadow snapshot: the trailing incomplete pool of every exact row (its next chunk's members)
        e = p0 + ev.nsched[row] - 1
        store = P["exact"] & on & (pl <= e) & (torch.remainder(e + 1, kp) != 0) & \
            (torch.div(pl, kp, rounding_mode="floor") == torch.div(e, kp, rounding_mode="floor"))
        tgt = torch.where(store, base + torch.remainder(tag, SH) * kp + torch.remainder(pl.clamp(min=0), kp),
                          torch.full_like(pl, dev.sh_dump))
        dev.shk.index_copy_(0, tgt, P["k"])
        dev.shg.index_copy_(0, tgt, P["gate"])
        dev.shtag.index_copy_(0, tgt, tag)
        dev.shpos.index_copy_(0, tgt, pl)
    except Exception as exc:  # noqa: BLE001
        _err("idx_prefill_post", exc)


def hook_tail_seed(layer_name, tail_slot_slice, positions, num_decode_tokens: int,
                   kpool: int) -> None:
    if not _ok():
        return
    try:
        dev = S.dev
        n = int(tail_slot_slice.shape[0])
        if n == 0:
            return
        loc = torch.arange(n, device=tail_slot_slice.device, dtype=torch.int64)
        tok = loc + int(num_decode_tokens)
        ts = tail_slot_slice.to(torch.int64)
        # the seed kernel's own write predicate (_kpool_tail_seed_kernel, as in
        # local_tailguard.write_mask_and_blocks('seed'))
        ring = dev.geo.ring
        ahead = torch.full_like(ts, -1)
        if n > kpool:
            ahead[: n - kpool] = ts[kpool:]
        blk = torch.div(ts.clamp(min=0), ring, rounding_mode="floor")
        ablk = torch.div(ahead.clamp(min=0), ring, rounding_mode="floor")
        written = (ts >= 0) & ((ahead < 0) | (ablk != blk))
        # expected seeded set: the last min(kpool, len) tokens of every request's chunk
        row = rows_of(dev.ev, tok, dev.geo.R)
        end = dev.ev.qsl[(row + 1).clamp(max=dev.geo.R)]
        want = tok >= (end - kpool)
        pos = positions[num_decode_tokens:num_decode_tokens + n]
        check_writes(dev, S_TAIL_SEED, _layer_idx(layer_name), S.rank, "tail", ts, pos, tok,
                     written=written, want_extra=want)
    except Exception as exc:  # noqa: BLE001
        _err("tail_seed", exc)


def hook_idx_decode_write(layer_name, dec_slot, dec_pos, dec_tail_slot, positions,
                          num_decode_tokens: int) -> None:
    """Grouped [B, lmax] decode-write inputs (index-K pool slot and tail slot per token)."""
    if not _ok():
        return
    try:
        dev = S.dev
        ev = dev.ev
        Bn, lm = int(dec_pos.shape[0]), int(dec_pos.shape[1])
        if Bn == 0:
            return
        b = torch.arange(Bn, device=dec_pos.device, dtype=torch.int64)
        j = torch.arange(lm, device=dec_pos.device, dtype=torch.int64)
        bb = b[:, None].expand(Bn, lm).reshape(-1)
        jj = j[None, :].expand(Bn, lm).reshape(-1)
        bcl = bb.clamp(max=dev.geo.R - 1)
        tok = ev.qsl[bcl] + jj
        act = (bb < ev.nrows) & (jj < ev.nsched[bcl])
        tok = torch.where(act, tok, torch.full_like(tok, 1 << 40))
        dp = dec_pos.reshape(-1).to(torch.int64)
        # grouping check: the grouped position equals the token's consumed position
        npos = int(positions.shape[0])
        tpos = positions[tok.clamp(0, max(npos - 1, 0))].to(torch.int64)
        gbad = act & (dp != tpos) & (ev.armed != 0)
        _acc_counts(dev, S_IDX_W_DEC, ev.armed, dict(grouppos=gbad))
        rows = torch.stack([
            torch.full_like(tok, S_IDX_W_DEC), torch.full_like(tok, S.rank),
            torch.full_like(tok, _layer_idx(layer_name)), ev.step.expand_as(tok), tok, bb,
            ev.tag[bcl], dp, tpos, tpos, torch.full_like(tok, B["gpos"]), jj], dim=1)
        record_rows(dev, gbad, rows)
        written_idx = (dp >= 0) & (dec_slot.reshape(-1) >= 0)
        check_writes(dev, S_IDX_W_DEC, _layer_idx(layer_name), S.rank, "idx", dec_slot.reshape(-1),
                     dp, tok, written=written_idx, active_extra=act)
        if dec_tail_slot is not None:
            dts = dec_tail_slot.reshape(-1)
            written_t = (dp >= 0) & (dts >= 0)
            check_writes(dev, S_TAIL_STASH, _layer_idx(layer_name), S.rank, "tail", dts, dp, tok,
                         written=written_t, active_extra=act)
    except Exception as exc:  # noqa: BLE001
        _err("idx_decode_write", exc)


def hook_idx_decode_read(layer_name, block_table, seq_lens, positions, num_rows: int) -> None:
    """Paged index-K read (B12x paged top-k or DeepGEMM): page-table rows and pool lengths."""
    if not _ok():
        return
    try:
        dev = S.dev
        ev = dev.ev
        kp = dev.geo.kpool
        if seq_lens.dim() == 2:
            Bn, nn = int(seq_lens.shape[0]), int(seq_lens.shape[1])
            b = torch.arange(Bn, device=seq_lens.device, dtype=torch.int64)
            j = torch.arange(nn, device=seq_lens.device, dtype=torch.int64)
            bb = b[:, None].expand(Bn, nn).reshape(-1)
            jj = j[None, :].expand(Bn, nn).reshape(-1)
            bcl = bb.clamp(max=dev.geo.R - 1)
            tok = ev.qsl[bcl] + jj
            act = (bb < ev.nrows) & (jj < ev.nsched[bcl])
            tab = block_table[:Bn].repeat_interleave(nn, dim=0) if int(block_table.shape[0]) == Bn \
                else block_table[: Bn * nn]
            lens = seq_lens.reshape(-1)
        else:
            m = int(num_rows)
            tok = torch.arange(m, device=seq_lens.device, dtype=torch.int64)
            act = None
            tab = block_table[:m]
            lens = seq_lens[:m]
        npos = int(positions.shape[0])
        p = positions[tok.clamp(0, max(npos - 1, 0))].to(torch.int64)
        exp_len = torch.div(p + 1, kp, rounding_mode="floor")
        check_table_read(dev, S_IDX_R_DEC, _layer_idx(layer_name), S.rank, tab, lens, tok, act,
                         exp_len, dev.geo.ppb)
    except Exception as exc:  # noqa: BLE001
        _err("idx_decode_read", exc)


def hook_idx_prefill_read(layer_name, chunk, positions) -> None:
    """cp_gather of one prefill chunk: rows = the chunk's requests (batch rows from token_start)."""
    if not _ok():
        return
    try:
        dev = S.dev
        ev = dev.ev
        tab = chunk.block_table
        cu = chunk.local_cu_seq_lens if chunk.local_cu_seq_lens is not None else chunk.cu_seq_lens
        m = int(tab.shape[0])
        if m == 0:
            return
        start_row = rows_of(ev, torch.full((1,), int(chunk.token_start), device=tab.device,
                                           dtype=torch.int64), dev.geo.R)
        brow = start_row + torch.arange(m, device=tab.device, dtype=torch.int64)
        bcl = brow.clamp(max=dev.geo.R - 1)
        last_tok = ev.qsl[(bcl + 1).clamp(max=dev.geo.R)] - 1
        npos = int(positions.shape[0])
        p_last = positions[last_tok.clamp(0, max(npos - 1, 0))].to(torch.int64)
        lens = (cu[1:m + 1] - cu[:m]).to(torch.int64)
        exp_len = torch.div(p_last + 1, dev.geo.kpool, rounding_mode="floor")
        act = brow < ev.nrows
        check_table_read(dev, S_IDX_R_PF, _layer_idx(layer_name), S.rank, tab, lens, last_tok, act,
                         exp_len, dev.geo.ppb)
    except Exception as exc:  # noqa: BLE001
        _err("idx_prefill_read", exc)


def hook_mla_read(layer, selected, nsa, num_tokens: int) -> None:
    if not _ok():
        return
    try:
        n = int(num_tokens)
        if n == 0:
            return
        L = _layer_idx(getattr(layer, "layer_name", "")) if not isinstance(layer, int) else layer
        tok = torch.arange(n, device=selected.device, dtype=torch.int64)
        check_selected(S.dev, L, S.rank, selected[:n], nsa[:n], S.pos_buf[:n], tok)
    except Exception as exc:  # noqa: BLE001
        _err("mla_read", exc)


def hook_mla_len(layer, per_token, num_tokens: int, width: int) -> None:
    """22ao: the consumed per-token causal length (B12x sparse-MLA forward, before clamping)."""
    if not _ok():
        return
    try:
        n = int(num_tokens)
        if n == 0:
            return
        L = _layer_idx(getattr(layer, "layer_name", "")) if not isinstance(layer, int) else layer
        tok = torch.arange(n, device=per_token.device, dtype=torch.int64)
        check_len(S.dev, L, S.rank, per_token[:n], S.pos_buf[:n], tok, int(width),
                  record=(L == S.first_layer))
    except Exception as exc:  # noqa: BLE001
        _err("mla_len", exc)


# ------------------------------------------------------------------ worker install
def install() -> None:
    from vllm.v1.worker.gpu import model_runner as mrmod
    from vllm.v1.worker.gpu import cudagraph_utils as cgmod
    from vllm.v1.worker.gpu import states as stmod

    MR = mrmod.GPUModelRunner
    if getattr(MR, "_mapguard_hooks", False):
        return
    MR._mapguard_hooks = True
    o_init_kv, o_exec, o_prep = MR.initialize_kv_cache, MR.execute_model, MR.prepare_attn

    def initialize_kv_cache(self, *a, **kw):
        r = o_init_kv(self, *a, **kw)
        try:
            _register_runner(self)
        except Exception as exc:  # noqa: BLE001
            _err("register", exc)
        return r

    def execute_model(self, scheduler_output, *a, **kw):
        dummy = kw.get("dummy_run", a[1] if len(a) > 1 else False)
        if not dummy and S.dev is not None and not S.baseline_done:
            _maybe_baseline(scheduler_output)
        if not dummy and S.ledger is not None:
            try:
                S.cur_fp = fingerprint(scheduler_output)
                S.ledger.update(scheduler_output)
                S.mode = "none"
                S.pending = None
            except Exception as exc:  # noqa: BLE001
                _err("ledger", exc)
        try:
            return o_exec(self, scheduler_output, *a, **kw)
        finally:
            if S.armed_host and S.dev is not None:
                try:
                    S.dev.E[H_ARMED].zero_()   # stream-ordered after the forward
                except Exception as exc:  # noqa: BLE001
                    _err("disarm", exc)
                S.armed_host = False
            if S.pending is not None:
                S.pending["cg"] = S.mode
                _log(S.pending)
                S.pending = None
                _ensure_poller()

    def prepare_attn(self, input_batch):
        r = o_prep(self, input_batch)
        if S.dev is not None and not _capturing():
            try:
                _upload(self, input_batch)
            except Exception as exc:  # noqa: BLE001
                _err("upload", exc)
        return r

    MR.initialize_kv_cache = initialize_kv_cache
    MR.execute_model = execute_model
    MR.prepare_attn = prepare_attn

    for cls in (cgmod.CudaGraphManager, getattr(cgmod, "ModelCudaGraphManager", None)):
        if cls is None:
            continue
        if "run_fullgraph" in cls.__dict__:
            o_full = cls.__dict__["run_fullgraph"]

            def run_fullgraph(self, *a, _o=o_full, **kw):
                S.mode = "FULL"
                return _o(self, *a, **kw)
            cls.run_fullgraph = run_fullgraph
        if "run_pw_graph" in cls.__dict__:
            o_pw = cls.__dict__["run_pw_graph"]

            def run_pw_graph(self, *a, _o=o_pw, **kw):
                S.mode = "PIECEWISE"
                return _o(self, *a, **kw)
            cls.run_pw_graph = run_pw_graph

    RS = stmod.RequestState
    o_add, o_rm = RS.add_request, RS.remove_request

    def add_request(self, req_id, *a, **kw):
        o_add(self, req_id, *a, **kw)
        try:
            _log(dict(kind="row_add", req_id=str(req_id), row=int(self.req_id_to_index[req_id]),
                      step=S.step))
        except Exception as exc:  # noqa: BLE001
            _err("row_add", exc)

    def remove_request(self, req_id):
        r = o_rm(self, req_id)
        if r is not None:
            _log(dict(kind="row_free", req_id=str(req_id), row=int(r), step=S.step))
        return r
    RS.add_request = add_request
    RS.remove_request = remove_request

    from vllm.v1.attention.backends.mla import b12x_mla_sparse as bmod
    IMPL = bmod.B12xMLASparseImpl
    o_upd = IMPL.do_kv_cache_update

    def do_kv_cache_update(self, kv_c_normed, k_pe, kv_cache, slot_mapping, *a, **kw):
        hook_mla_write(self, kv_cache, slot_mapping)
        return o_upd(self, kv_c_normed, k_pe, kv_cache, slot_mapping, *a, **kw)
    IMPL.do_kv_cache_update = do_kv_cache_update
    atexit.register(lambda: poll("exit"))
    _log(dict(kind="installed", mode=MODE, pid=os.getpid()))


# ------------------------------------------------------------------ EngineCore (scheduler) side
class _Sched:
    def __init__(self):
        self.seq = 0
        self.last: dict[str, str] = {}
        self.ga, self.gt = 0, 1


def install_scheduler_side(SchedulerCls) -> None:
    if not ENABLED or getattr(SchedulerCls, "_mapguard_hooks", False):
        return
    SchedulerCls._mapguard_hooks = True
    SS = _Sched()
    o_sched, o_upd = SchedulerCls.schedule, SchedulerCls.update_from_output

    def schedule(self, *a, **kw):
        so = o_sched(self, *a, **kw)
        try:
            SS.seq += 1
            km = self.kv_cache_manager
            reqs = {}
            items = []
            changed = {}
            for rid, n in so.num_scheduled_tokens.items():
                ids = km.get_block_ids(rid)
                g0 = list(ids[SS.ga]) if len(ids) > SS.ga else []
                g1 = list(ids[SS.gt]) if len(ids) > SS.gt else []
                items.append((rid, g0, g1))
                d = lists_digest([(rid, g0, g1)])
                if SS.last.get(rid) != d:
                    SS.last[rid] = d
                    changed[rid] = [g0, g1, [len(x) for x in ids]]
                req = self.requests.get(rid)
                reqs[rid] = [int(req.num_computed_tokens) if req is not None else -1, int(n),
                             len((so.scheduled_spec_decode_tokens or {}).get(rid, ()))]
            for rid in so.finished_req_ids or ():
                SS.last.pop(rid, None)
            rec = dict(kind="sched", seq=SS.seq, fp=fingerprint(so),
                       reqs=reqs, dig=lists_digest(items), changed=changed,
                       finished=sorted(map(str, so.finished_req_ids or ())),
                       free_blocks=int(km.block_pool.get_num_free_blocks()))
            if SS.seq % 16 == 1:
                rec["excl"] = _exclusivity(km)
            _log(rec, prefix="ms")
        except Exception as exc:  # noqa: BLE001
            _err("sched", exc)
        return so

    def update_from_output(self, scheduler_output, model_runner_output, *a, **kw):
        try:
            idx = getattr(model_runner_output, "req_id_to_index", None) or {}
            sampled = getattr(model_runner_output, "sampled_token_ids", None) or []
            spec = scheduler_output.scheduled_spec_decode_tokens or {}
            rej = {}
            for rid, toks in spec.items():
                if rid in idx and idx[rid] < len(sampled):
                    gen = len(sampled[idx[rid]])
                    acc = max(gen - 1, 0)
                    rej[rid] = [len(toks), acc, len(toks) - acc]
            _log(dict(kind="upd", fp=fingerprint(scheduler_output), rej=rej),
                 prefix="ms")
        except Exception as exc:  # noqa: BLE001
            _err("upd", exc)
        return o_upd(self, scheduler_output, model_runner_output, *a, **kw)

    SchedulerCls.schedule = schedule
    SchedulerCls.update_from_output = update_from_output
    _log(dict(kind="installed_scheduler", pid=os.getpid()), prefix="ms")


def _exclusivity(km) -> dict:
    """Live holders per block id across all groups, straight from the managers' req_to_blocks."""
    holders: dict[int, set] = {}
    shared0 = 0
    for g, m in enumerate(km.coordinator.single_type_managers):
        for rid, blks in m.req_to_blocks.items():
            for blk in blks:
                b = int(getattr(blk, "block_id", blk))
                if b == 0:
                    continue
                holders.setdefault(b, set()).add(g)
    multi = {b: sorted(gs) for b, gs in holders.items() if len(gs) > 1}
    return dict(live_blocks=len(holders), multi_group=len(multi),
                examples=dict(list(multi.items())[:5]), max_id=max(holders) if holders else 0,
                ge_935=sum(1 for b in holders if b >= 935), shared0=shared0)


def ensure_worker_install() -> None:
    """Called by the patched model-layer module at import (worker processes only)."""
    if not ENABLED:
        return
    try:
        install()
    except Exception as exc:  # noqa: BLE001
        _err("install", exc)
