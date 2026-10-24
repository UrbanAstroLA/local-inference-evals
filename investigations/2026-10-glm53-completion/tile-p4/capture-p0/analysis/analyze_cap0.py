#!/usr/bin/env python3
"""capture-p0 analysis (PROTOCOL.md sections 4-5).  Written and tested on synthetic fixtures BEFORE any GPU data.

usage: analyze_cap0.py AO PO [--quiet] [--out DIR]
  AO  public attempt directory (client.jsonl, p4log/{p4,sched,cap0}_r*.jsonl)
  PO  private attempt directory (cap0_<tag>_r<rank>.npz, loc_<tag>_r<rank>.npz); never written to except
      PO/cap0-detail.json (per-token tables)
writes AO/cap0-result.json and AO/cap0-result.md (counts, positions, classes; no bytes, ids or text).
--out DIR: read-only re-analysis -- the two result files go to DIR, nothing is written under AO or PO.

Rules fixed before data:
  * Missing or unverifiable evidence is INCOMPLETE, never a negative.  A replicate counts only when every gate
    of section 4 passes for it; fewer than 2 counted replicates -> INCOMPLETE.
  * Comparison is exact (bytes); equal means byte-identical.  Magnitudes are reported, never used as a threshold.
  * Valid state slices (used for the reading): MLA KV of layer 7 (blocks 0-2, rank 0), KDA layer-4 recurrent
    state and live conv rows 0-2 (both ranks), KDA layer-0 live conv rows (both ranks); controls MLA layer 3,
    KDA layer-2 recurrent.  Reported but not "valid state": conv rows 3-5 (inactive), indexer pool records
    511 / 1022 (consumed, never written by the warm), the rest of the indexer blocks.
  * Localisation reading: the first warm step (chunk order) and, inside it, the first boundary in computation
    order (cap0_hooks_layout LOC_ORDER, with L2.state_end first and L4.state_end last) at which the token-row
    sha256 differ between two localisation replicates, on either rank.  The operation between that boundary and
    the previous one is the reading, if every differing localisation pair names the same boundary.
"""
import glob
import hashlib
import importlib.util
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "cap0_layout", os.path.join(HERE, "..", "build", "files", "vllm", "cap0_hooks_layout.py"))
LY = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LY)

CLASSES = {
    "located": "operation located",
    "slice": "slice located, operation not located",
    "equal": "valid state equal this time (non-reproduced)",
    "incomplete": "INCOMPLETE (instrumentation/addressing invalid)",
}
ASSESSOR = {   # 22ax decision table row and the next action it implies
    "located": ("Valid state differs, reproducible local operation defect found",
                "the producing operation is located in serving but not yet shown repeatable: replay that operation "
                "with the captured inputs per assessor section 5 before attributing propagation"),
    "slice": ("Valid state differs, bounded replay does not locate it",
              "preserve the finding (the exact differing slice); stop expanding this diagnostic branch and plan "
              "the behavioural A/B"),
    "equal": ("(no row: the Stage-0 difference did not reproduce in this construction)",
              "report non-reproduction; one equal pair does not establish repeatability; do not expand; plan the "
              "behavioural A/B"),
    "incomplete": ("Instrumentation or addressing remains invalid",
                   "repair the specific measurement named in the failed gate before interpreting anything"),
}
CHUNKS = LY.warm_chunks(LY.WARM_TOKENS)
CACHED_STEPS = [c for c in CHUNKS if c[0] < LY.N_CACHED]          # the 9 warm steps that build blocks 0-2
NB = LY.N_CACHED // LY.BLOCK_TOKENS
REQ_SLICES = {0: [f"mla{L}_b{j}" for L in (3, 7) for j in range(NB)] + [f"idx{L}_b{j}" for L in (3, 7) for j in range(NB)]
              + ["kda0_conv_live", "kda0_conv_inactive", "kda2_rec", "kda4_rec", "kda4_conv_live", "kda4_conv_inactive",
                 "kda4_rec_dst"],
              1: ["kda0_conv_live", "kda0_conv_inactive", "kda2_rec", "kda4_rec", "kda4_conv_live", "kda4_conv_inactive",
                  "kda4_rec_dst"]}
VALID = [(0, f"mla7_b{j}") for j in range(NB)] + [(r, s) for r in (0, 1) for s in ("kda4_rec", "kda4_conv_live", "kda0_conv_live")]
CONTROL = [(0, f"mla3_b{j}") for j in range(NB)] + [(r, "kda2_rec") for r in (0, 1)]
OTHER = [(r, s) for r in (0, 1) for s in ("kda0_conv_inactive", "kda4_conv_inactive")] + \
        [(0, f"idx{L}_b{j}") for L in (3, 7) for j in range(NB)]
COMP_ORDER = ["L2.state_end"] + [p for p, _t, _d in LY.LOC_ORDER] + ["L4.state_end"]
LOC_TENSORS = {p: t for p, t, _d in LY.LOC_ORDER}
E4M3 = np.array([np.nan if v is None else v for v in LY.e4m3_table()], dtype=np.float64)


# ------------------------------------------------------------------ loading
def jl(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        for line in open(p):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    out.append(dict(kind="_malformed", file=os.path.basename(p)))
    return out


def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_npz(p):
    z = np.load(p, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    meta = json.loads(str(d.pop("meta"))) if "meta" in d else {}
    return d, meta


def safe(s):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))[:80]


class Run:
    def __init__(self, ao, po):
        self.ao, self.po = ao, po
        self.client = jl(os.path.join(ao, "client.jsonl"))
        self.p4 = jl(os.path.join(ao, "p4log", "p4_r*_p*.jsonl"))
        self.cap0 = jl(os.path.join(ao, "p4log", "cap0_r*_p*.jsonl"))
        self.sched = [s for r in jl(os.path.join(ao, "p4log", "sched_r0_p*.jsonl")) if r.get("kind") == "sched"
                      for s in r.get("steps", [])]
        start = [d for d in self.client if d.get("kind") == "start"]
        self.start = start[-1] if start else {}
        self.reps = [r["name"] for r in self.start.get("replicates", [])]
        self.loc = {r["name"]: bool(r.get("loc")) for r in self.start.get("replicates", [])}

    def crec(self, kind, rep):
        x = [d for d in self.client if d.get("kind") == kind and d.get("replicate") == rep]
        return x[-1] if x else None

    def caprec(self, tag, rank):
        x = [d for d in self.cap0 if d.get("kind") == "cap0_capture" and d.get("tag") == tag and d.get("rank") == rank]
        return x

    def locrec(self, tag, rank):
        x = [d for d in self.cap0 if d.get("kind") == "cap0_loc_flushed" and d.get("tag") == tag and d.get("rank") == rank]
        return x

    def init(self, rank):
        x = [d for d in self.cap0 if d.get("kind") == "cap0_init" and d.get("rank") == rank]
        return x[-1] if x else None

    def blocks(self, tag):
        x = [d for d in self.p4 if d.get("kind") == "blocks" and d.get("tag") == tag and d.get("rank", 0) == 0]
        return x[-1] if x else None

    def steps_of(self, tag):
        out = []
        for s in self.sched:
            for row in s.get("rows", []):
                if len(row) > 8 and row[3] == tag:
                    out.append(dict(step=s.get("step"), mode=s.get("mode"), nr=s.get("nr"), n_sched=row[4],
                                    n_computed=row[5], prefilling=row[8], uid=row[2]))
        return out


def tags(rep, loc):
    return f"CW-{rep}-{'L' if loc else 'N'}", f"CAP-{rep}"


# ------------------------------------------------------------------ gates (section 4)
def gate_replicate(R, rep):
    """-> (ok, failures list, info).  Every check names its evidence; missing evidence fails the gate."""
    fail, info = [], {}
    loc = R.loc.get(rep, False)
    wtag, ctag = tags(rep, loc)
    w, c = R.crec("warm", rep), R.crec("cap", rep)
    # client
    if w is None:
        fail.append("G-client: warm request missing")
    elif w.get("cached_tokens") not in (0, None) or w.get("prompt_tokens") != LY.WARM_TOKENS:
        fail.append("G-client: warm cached_tokens %s (must be 0) / prompt_tokens %s (must be %d: a shorter warm caches "
                    "fewer blocks and the capture loses one more, amendment A1)" % (
                        w.get("cached_tokens"), w.get("prompt_tokens"), LY.WARM_TOKENS))
    elif w.get("cached_tokens") is None:
        fail.append("G-client: warm cached_tokens not reported")
    elif not w.get("ok"):
        fail.append("G-client: warm request not ok (status %s, error %s)" % (w.get("status"), w.get("error")))
    if c is None:
        fail.append("G-client: capture request missing")
    elif c.get("cached_tokens") != LY.N_CACHED or c.get("prompt_tokens") != LY.P_EARLY:
        fail.append("G-client: capture cached_tokens %s (must be %d) / prompt_tokens %s" % (
            c.get("cached_tokens"), LY.N_CACHED, c.get("prompt_tokens")))
    elif not c.get("ok"):
        fail.append("G-client: capture request not ok (status %s, error %s)" % (c.get("status"), c.get("error")))
    info["salt"] = (w or {}).get("salt")
    info["prompt_sha256"] = ((w or {}).get("prompt_sha256"), (c or {}).get("prompt_sha256"))
    # execution: the warm's steps that build the cached blocks
    st = R.steps_of(wtag)
    got = [(s["n_computed"], s["n_sched"]) for s in st if s["n_computed"] < LY.N_CACHED]
    if not R.sched:
        fail.append("G-exec: no scheduler log at all (p4log/sched_r0_*.jsonl missing): the step rows were never "
                    "flushed, so the warm chunking cannot be verified")
    elif got != CACHED_STEPS:
        fail.append("G-exec: warm chunks %s != %s" % (got[:12], CACHED_STEPS))
    if any(s["nr"] != 1 or s["mode"] != "none" or not s["prefilling"] for s in st if s["n_computed"] < LY.N_CACHED):
        fail.append("G-exec: a cached-span warm step was not solitary / eager / prefilling")
    cst = R.steps_of(ctag)
    if not cst or cst[0]["n_computed"] != LY.N_CACHED or not cst[0]["prefilling"] or cst[0]["nr"] != 1:
        fail.append("G-exec: the capture request's first step is not a solitary prefill from n_computed %d" % LY.N_CACHED)
    info["cap_first_step"] = cst[0] if cst else None
    # capture records and files, both ranks
    caps = {}
    for rank in (0, 1):
        recs = R.caprec(ctag, rank)
        if len(recs) != 1:
            fail.append(f"G-capture: rank {rank}: {len(recs)} capture records (need exactly 1)")
            continue
        cr = recs[0]
        caps[rank] = cr
        if not cr.get("ok"):
            fail.append(f"G-capture: rank {rank}: capture not ok ({cr.get('why') or cr.get('problems')})")
        if cr.get("n_computed") != LY.N_CACHED or cr.get("n_sched") != LY.CHUNK_BUDGET:
            fail.append(f"G-capture: rank {rank}: n_computed {cr.get('n_computed')} n_sched {cr.get('n_sched')}")
        if cst and cr.get("step") is not None and R.sched and cr.get("step") != cst[0]["step"]:
            fail.append(f"G-capture: rank {rank}: captured at step {cr.get('step')}, not the request's first step {cst[0]['step']}")
        sl = cr.get("slices") or {}
        for k in REQ_SLICES[rank]:
            d = sl.get(k)
            if d is None:
                fail.append(f"G-capture: rank {rank}: slice {k} missing")
                continue
            if not d.get("layout_ok") or not d.get("view_equal"):
                fail.append(f"G-addr: rank {rank}: {k} layout_ok {d.get('layout_ok')} view_equal {d.get('view_equal')}")
            if "twosum_device" in d and d.get("twosum_device") != d.get("twosum_host"):
                fail.append(f"G-addr: rank {rank}: {k} device two-sum != host two-sum of the copied bytes")
            if k.startswith(("mla", "idx")) and not d.get("kernel_pages_contiguous"):
                fail.append(f"G-addr: rank {rank}: {k} kernel pages not b*bpk + k")
        p = os.path.join(R.po, f"cap0_{safe(ctag)}_r{rank}.npz")
        if not os.path.exists(p):
            fail.append(f"G-capture: rank {rank}: {os.path.basename(p)} missing")
        elif file_sha(p) != cr.get("file_sha256"):
            fail.append(f"G-capture: rank {rank}: {os.path.basename(p)} sha256 != its log record")
        d4, dd = sl.get("kda4_rec"), sl.get("kda4_rec_dst")
        if d4 and dd and (d4.get("sha256") == dd.get("sha256") or d4.get("b") == dd.get("b")):
            fail.append(f"G-timing: rank {rank}: pre-copy destination column equals the cached column (read after the pre-copy?)")
    if len(caps) == 2:
        s0, s1 = caps[0].get("cached_ids_sha256"), caps[1].get("cached_ids_sha256")
        if s0 is None or s0 != s1:
            fail.append("G-input: engine-side cached prompt ids sha256 missing or differ between ranks")
        info["cached_ids_sha256"] = s0
    # provenance: the captured blocks are the ones this replicate's warm computed
    bw = R.blocks(wtag)
    if bw is None:
        fail.append("G-prov: no block record for the warm request")
    for rank, cr in caps.items():
        init = R.init(rank)
        if init is None:
            fail.append(f"G-prov: rank {rank}: no cap0_init record")
            continue
        if bw is None:
            continue
        groups, wrow = cr.get("groups") or [], bw.get("groups", [])
        for key, tg in (init.get("targets") or {}).items():
            g = tg.get("group")
            cols = list(range(NB)) if tg.get("kind") in ("mla", "idx") else [LY.cached_state_col(LY.N_CACHED)]
            if g is None or g >= len(groups) or g >= len(wrow):
                fail.append(f"G-prov: rank {rank}: group {g} of {key} missing in the capture or the warm's block record")
                continue
            got = [groups[g].get("logical", [None] * 8)[j] if j < len(groups[g].get("logical", [])) else None for j in cols]
            exp = [wrow[g][j] if j < len(wrow[g]) else None for j in cols]
            if got != exp or None in got:
                fail.append(f"G-prov: rank {rank}: {key} group {g} columns {cols}: captured {got} != the warm's {exp}")
    info["physical"] = {rk: {k: d.get("b") for k, d in (cr.get("slices") or {}).items()} for rk, cr in caps.items()}
    return not fail, fail, info


def gate_loc(R, rep, ref_sha=None):
    """Localisation evidence of one loc replicate (both ranks).  -> (ok, failures, per-rank (arrays, meta))."""
    fail, data = [], {}
    wtag, _ = tags(rep, True)
    for rank in (0, 1):
        recs = R.locrec(wtag, rank)
        p = os.path.join(R.po, f"loc_{safe(wtag)}_r{rank}.npz")
        if len(recs) != 1 or not os.path.exists(p):
            fail.append(f"G-loc: rank {rank}: localisation record/file missing")
            continue
        if file_sha(p) != recs[0].get("file_sha256"):
            fail.append(f"G-loc: rank {rank}: file sha256 != its log record")
            continue
        arr, meta = load_npz(p)
        data[rank] = (arr, meta)
        steps = sorted(meta.get("steps", []), key=lambda s: s["pos0"])
        cover = [(s["pos0"], s["n"]) for s in steps]
        want = [(s, min(n, meta.get("loc_tokens", LY.N_CACHED) - s)) for s, n in CHUNKS if s < meta.get("loc_tokens", LY.N_CACHED)]
        if cover != want:
            fail.append(f"G-loc: rank {rank}: steps cover {cover[:4]}... not {want[:4]}...")
        if any(s.get("mode") != "none" for s in steps):
            fail.append(f"G-loc: rank {rank}: a recorded step was not eager")
        if meta.get("errors"):
            fail.append(f"G-loc: rank {rank}: recording errors {meta['errors'][:3]}")
        for p_, tens in LOC_TENSORS.items():
            if p_ == "L3.topk" and rank != 0:
                continue
            for t in tens:
                k = f"dig:{p_}.{t}"
                if k not in arr:
                    fail.append(f"G-loc: rank {rank}: {p_}.{t} not recorded")
                    continue
                pos = arr[f"pos:{p_}.{t}"]
                if not np.array_equal(pos, np.arange(pos.size)) or pos.size != meta.get("loc_tokens", LY.N_CACHED):
                    fail.append(f"G-loc: rank {rank}: {p_}.{t} positions not 0..loc_tokens-1")
        for (pa, ta), (pb, tb) in LY.LOC_IDENTITIES:
            ka, kb = f"dig:{pa}.{ta}", f"dig:{pb}.{tb}"
            if ka in arr and kb in arr and not np.array_equal(arr[ka], arr[kb]):
                fail.append(f"G-loc: rank {rank}: identity {pa}.{ta} == {pb}.{tb} violated (instrument)")
        for L in (2, 4):
            ends = sorted(d["pos_end"] for d in meta.get("states", []) if d.get("point") == f"L{L}.state_end")
            if ends != [s + n for s, n in want]:
                fail.append(f"G-loc: rank {rank}: L{L}.state_end at {ends[:4]}... not every step end")
        ids = [s.get("ids_sha256") for s in steps]
        if any(x is None for x in ids):
            fail.append(f"G-loc: rank {rank}: a step's engine-side input ids were not hashed")
        data.setdefault("ids", {})[rank] = ids
    if 0 in data and 1 in data and data["ids"].get(0) != data["ids"].get(1):
        fail.append("G-loc: input ids differ between ranks")
    return not fail, fail, data


# ------------------------------------------------------------------ comparisons
@np.errstate(all="ignore")
def mla_diff(a, b, j):
    A, B = a.reshape(LY.BLOCK_TOKENS, LY.MLA_REC), b.reshape(LY.BLOCK_TOKENS, LY.MLA_REC)
    lat = np.any(A[:, :LY.MLA_LATENT] != B[:, :LY.MLA_LATENT], axis=1)
    sc = np.any(A[:, LY.MLA_LATENT:] != B[:, LY.MLA_LATENT:], axis=1)
    tok = np.nonzero(lat | sc)[0]
    out = dict(tokens_differ=int(tok.size), latent_only=int(np.sum(lat & ~sc)), scale_only=int(np.sum(sc & ~lat)),
               both=int(np.sum(lat & sc)))
    if tok.size:
        pos = j * LY.BLOCK_TOKENS + tok
        out["first_pos"] = int(pos[0])
        out["by_chunk"] = {str(k): int(np.sum([LY.chunk_of(int(p), CHUNKS) == k for p in pos])) for k in
                           sorted({LY.chunk_of(int(p), CHUNKS) for p in pos})}
        sa = A[tok, LY.MLA_LATENT:].copy().view(np.float32).reshape(-1, LY.MLA_NGROUP)
        sb = B[tok, LY.MLA_LATENT:].copy().view(np.float32).reshape(-1, LY.MLA_NGROUP)
        va = E4M3[A[tok, :LY.MLA_LATENT]].reshape(-1, LY.MLA_NGROUP, LY.MLA_GROUP) * sa[:, :, None]
        vb = E4M3[B[tok, :LY.MLA_LATENT]].reshape(-1, LY.MLA_NGROUP, LY.MLA_GROUP) * sb[:, :, None]
        d = np.abs(va - vb)
        out["groups_differ"] = int(np.sum(np.any(A[tok, :LY.MLA_LATENT].reshape(-1, 4, 128) !=
                                                  B[tok, :LY.MLA_LATENT].reshape(-1, 4, 128), axis=2)))
        out["max_abs_decoded"] = float(np.nanmax(d)) if np.isfinite(d).any() else None
        out["positions"] = [int(p) for p in pos[:64]]
    return out


def idx_records(u8):
    """[1088, 132] (128 E4M3 values + 4 scale bytes) per pool record of one indexer block."""
    pages = u8.reshape(LY.IDX_PPB, LY.IDX_POOL_PAGE)
    vals = pages[:, :LY.IDX_SCALE_OFF].reshape(LY.IDX_PPB, LY.IDX_ROWS, LY.IDX_VAL)
    sc = pages[:, LY.IDX_SCALE_OFF:].reshape(LY.IDX_PPB, LY.IDX_ROWS, 4)
    return np.concatenate([vals, sc], axis=2).reshape(LY.IDX_POOLS, LY.IDX_VAL + 4)


def idx_diff(a, b, j):
    A, B = idx_records(a), idx_records(b)
    q = np.nonzero(np.any(A != B, axis=1))[0]
    pools = LY.never_written_pools(CHUNKS, j)
    out = dict(records_differ=int(q.size), first_pool=int(q[0]) if q.size else None,
               unwritten_pools={str(p): dict(zero_a=bool(not A[p].any()), zero_b=bool(not B[p].any()),
                                             equal=bool(np.array_equal(A[p], B[p]))) for p in pools})
    rest = np.ones(LY.IDX_POOLS, dtype=bool)
    rest[pools] = False
    out["rest_records_differ"] = int(np.sum(np.any(A[rest] != B[rest], axis=1)))
    return out


@np.errstate(all="ignore")
def words_diff(a, b, dtype):
    A, B = a.view(dtype), b.view(dtype)
    neq = A != B
    out = dict(words=int(A.size), words_differ=int(neq.sum()))
    if neq.any():
        fa, fb = A.astype(np.float64), B.astype(np.float64)
        out["max_abs"] = float(np.nanmax(np.abs(fa - fb)))
    return out


def bf16(u8):
    return (u8.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


@np.errstate(all="ignore")
def conv_diff(a, b):
    A, B = bf16(a).reshape(-1, LY.CONV_ROW_BYTES // 2), bf16(b).reshape(-1, LY.CONV_ROW_BYTES // 2)
    rows = [int(np.sum(A[r] != B[r])) for r in range(A.shape[0])]
    out = dict(rows_words_differ=rows)
    if any(rows):
        out["max_abs"] = float(np.nanmax(np.abs(A.astype(np.float64) - B.astype(np.float64))))
    return out


def compare_slice(name, a, b):
    if a.size != b.size:
        return dict(equal=False, size_mismatch=[int(a.size), int(b.size)])
    eq = bool(np.array_equal(a, b))
    out = dict(equal=eq)
    if eq:
        return out
    if name.startswith("mla"):
        out.update(mla_diff(a, b, int(name.split("_b")[1])))
    elif name.startswith("idx"):
        out.update(idx_diff(a, b, int(name.split("_b")[1])))
    elif "conv" in name:
        out.update(conv_diff(a, b))
    elif "rec" in name:
        out.update(words_diff(a, b, np.float32))
        A, B = a.view(np.float32).reshape(32, -1), b.view(np.float32).reshape(32, -1)
        out["heads_differ"] = int(np.sum(np.any(A != B, axis=1)))
    return out


def loc_first(da, db, ranks):
    """First (step, point, tensor, rank) at which the token-row digests of two loc replicates differ, with the
    differing positions of that step and point.  da/db: {rank: (arrays, meta)}."""
    steps = [(s, n) for s, n in CHUNKS if s < LY.N_CACHED]
    for s, n in steps:
        for point in COMP_ORDER:
            hits = []
            for rank in ranks:
                if rank not in da or rank not in db:
                    continue
                (aa, ma), (ab, mb) = da[rank], db[rank]
                if point.endswith("state_end"):
                    sa = {d["pos_end"]: d for d in ma.get("states", []) if d.get("point") == point}
                    sb = {d["pos_end"]: d for d in mb.get("states", []) if d.get("point") == point}
                    e = min(s + n, LY.N_CACHED)
                    if e in sa and e in sb:
                        for f in ("rec_sha256", "conv_live_sha256"):
                            if sa[e].get(f) != sb[e].get(f):
                                hits.append(dict(rank=rank, tensor=f.replace("_sha256", ""), positions=[], pos_end=e))
                    continue
                for t in LOC_TENSORS[point]:
                    k = f"dig:{point}.{t}"
                    if k not in aa or k not in ab:
                        continue
                    lo, hi = s, min(s + n, aa[k].shape[0], ab[k].shape[0])
                    neq = np.nonzero(np.any(aa[k][lo:hi] != ab[k][lo:hi], axis=1))[0]
                    if neq.size:
                        hits.append(dict(rank=rank, tensor=t, n_tokens=int(neq.size), first_pos=int(lo + neq[0]),
                                         positions=[int(lo + x) for x in neq[:32]]))
            if hits:
                return dict(step_pos0=s, step_n=n, chunk=LY.chunk_of(s, CHUNKS), point=point, hits=hits,
                            operation=LY.LOC_OPERATION.get(point))
    return None


@np.errstate(all="ignore")
def raw_magnitude(arr, meta, point, tensor):
    """From a non-reference replicate's file: max |cur - ref| over its saved first differing rows (decoded)."""
    nm = f"{point}.{tensor}"
    if f"diffcur:{nm}" not in arr or f"diffref:{nm}" not in arr:
        return None
    dt = (meta.get("points", {}).get(nm) or {}).get("dtype", "")
    a, b = arr[f"diffcur:{nm}"], arr[f"diffref:{nm}"]
    if "bfloat16" in dt:
        fa, fb = bf16(a.reshape(-1).copy()), bf16(b.reshape(-1).copy())
    elif "float32" in dt:
        fa, fb = a.reshape(-1).copy().view(np.float32), b.reshape(-1).copy().view(np.float32)
    elif "int32" in dt:
        fa, fb = a.reshape(-1).copy().view(np.int32), b.reshape(-1).copy().view(np.int32)
    else:
        return None
    d = np.abs(fa.astype(np.float64) - fb.astype(np.float64))
    return dict(rows=int(a.shape[0]), positions=[int(x) for x in arr[f"diffpos:{nm}"]],
                max_abs=float(np.nanmax(d)) if d.size else None, words_differ=int(np.sum(fa != fb)))


def topk_sets(arr):
    """Amendment A2 (2026-10-24, from the descriptive look at attempt 1, before any attempt-2 data): for the saved first
    differing top-k rows (current vs reference replicate), is it the same SET of token ids in another order, or a
    different selection?  -> dict or None."""
    if "diffcur:L3.topk.ids" not in arr or "diffref:L3.topk.ids" not in arr:
        return None
    cur = arr["diffcur:L3.topk.ids"].reshape(arr["diffcur:L3.topk.ids"].shape[0], -1).copy().view(np.int32)
    ref = arr["diffref:L3.topk.ids"].reshape(arr["diffref:L3.topk.ids"].shape[0], -1).copy().view(np.int32)
    same, diff = 0, 0
    for a, b in zip(cur, ref):
        if set(a[a >= 0].tolist()) == set(b[b >= 0].tolist()):
            same += 1
        else:
            diff += 1
    return dict(rows=same + diff, same_set_other_order=same, different_selection=diff,
                positions=[int(x) for x in arr["diffpos:L3.topk.ids"]])


# ------------------------------------------------------------------ main reading
def analyse(ao, po):
    R = Run(ao, po)
    res = dict(schema="cap0-result/1", replicates=R.reps, loc=R.loc, gates={}, counted=[], pairs={}, notes=[])
    detail = {}
    if not R.reps:
        res.update(cls="incomplete", why=["no client start record"])
        return res, detail
    infos = {}
    for rep in R.reps:
        ok, fail, info = gate_replicate(R, rep)
        res["gates"][rep] = dict(ok=ok, failures=fail)
        infos[rep] = info
        if ok:
            res["counted"].append(rep)
    counted = res["counted"]
    # cross-replicate input identity (client prompt ids and the engine's cached ids)
    for k in ("prompt_sha256", "cached_ids_sha256"):
        vals = {json.dumps(infos[r].get(k)) for r in counted}
        if len(vals) > 1:
            res.setdefault("why", []).append(f"G-input: {k} differs between counted replicates")
    salts = [infos[r].get("salt") for r in counted]
    if len(set(salts)) != len(salts):
        res.setdefault("why", []).append("G-input: replicates share a cache salt")
    if len(counted) < 2 or res.get("why"):
        res.update(cls="incomplete")
        res.setdefault("why", []).append("fewer than 2 replicates passed every gate" if len(counted) < 2 else "")
        res["why"] = [w for w in res["why"] if w]
        return res, detail
    # load the captures
    cap = {}
    for rep in counted:
        _, ctag = tags(rep, R.loc[rep])
        for rank in (0, 1):
            arrs, meta = load_npz(os.path.join(po, f"cap0_{safe(ctag)}_r{rank}.npz"))
            cap[(rep, rank)] = arrs
            if meta.get("rank") != rank or meta.get("tag") != ctag:
                res.setdefault("why", []).append(f"G-capture: {rep} rank {rank}: the file's own tag/rank "
                                                 f"({meta.get('tag')}, {meta.get('rank')}) differ from its name")
            for (rk, s_) in VALID + CONTROL:
                if rk == rank and s_ in arrs and not arrs[s_].any():
                    res.setdefault("why", []).append(f"G-content: {rep} rank {rank}: {s_} is all zero (a written "
                                                     "slice cannot be; the addressing is suspect)")
    if res.get("why"):
        res.update(cls="incomplete")
        return res, detail
    # loc evidence
    locd, locok = {}, {}
    for rep in counted:
        if R.loc[rep]:
            ok, fail, data = gate_loc(R, rep)
            locok[rep] = ok
            res["gates"][rep]["loc_ok"] = ok
            res["gates"][rep]["loc_failures"] = fail
            if ok:
                locd[rep] = data
    ids = {rep: json.dumps(d.get("ids", {}).get(0)) for rep, d in locd.items()}
    if len(set(ids.values())) > 1:
        res["notes"].append("localisation INCOMPLETE: the warm steps' engine-side input ids differ between "
                            "localisation replicates %s" % sorted(ids))
        for rep in locd:
            res["gates"][rep]["loc_ok"] = False
            res["gates"][rep]["loc_failures"].append("G-loc: input ids differ between localisation replicates")
        locd = {}
    # per pair
    any_valid = False
    for ra, rb in itertools.combinations(counted, 2):
        pr = dict(same_alloc=None, valid_differ=[], control_differ=[], other_differ=[], slices={})
        same = True
        for (rank, s) in VALID + CONTROL + OTHER:
            a, b = cap[(ra, rank)].get(s), cap[(rb, rank)].get(s)
            if a is None or b is None:
                pr["slices"][f"r{rank}:{s}"] = dict(missing=True)
                continue
            c = compare_slice(s, a, b)
            ba, bb = infos[ra]["physical"].get(rank, {}).get(s), infos[rb]["physical"].get(rank, {}).get(s)
            c["physical"] = [ba, bb]
            same = same and ba == bb
            pr["slices"][f"r{rank}:{s}"] = c
            if not c["equal"]:
                (pr["valid_differ"] if (rank, s) in VALID else pr["control_differ"] if (rank, s) in CONTROL
                 else pr["other_differ"]).append(f"r{rank}:{s}")
        pr["same_alloc"] = same
        pr["loc_pair"] = ra in locd and rb in locd
        if pr["valid_differ"]:
            any_valid = True
        if pr["loc_pair"]:
            first = loc_first(locd[ra], locd[rb], (0, 1))
            pr["loc_first"] = first
            if first is not None:
                nonref = rb if locd[rb][0][1].get("is_ref") is False else ra
                mags = {}
                for h in first["hits"]:
                    if h.get("positions") and h["rank"] in locd[nonref]:
                        arr, meta = locd[nonref][h["rank"]]
                        m = raw_magnitude(arr, meta, first["point"], h["tensor"])
                        if m:
                            mags[f"r{h['rank']}:{h['tensor']}"] = m
                pr["loc_first_magnitude"] = mags
            # cross-check: the warm's last write of the cached state == the bytes at point 0 (same replicate)
        res["pairs"][f"{ra}-{rb}"] = pr
    # same replicate: last state write vs point-0 bytes
    for rep in locd:
        for rank in (0, 1):
            if rank not in locd[rep]:
                continue
            meta = locd[rep][rank][1]
            end = [d for d in meta.get("states", []) if d.get("point") == "L4.state_end" and d.get("pos_end") == LY.N_CACHED]
            c = cap[(rep, rank)]
            if end and "kda4_rec" in c and "kda4_conv_live" in c:
                same_bytes = (end[0].get("rec_sha256") == hashlib.sha256(c["kda4_rec"].tobytes()).hexdigest() and
                              end[0].get("conv_live_sha256") == hashlib.sha256(c["kda4_conv_live"].tobytes()).hexdigest())
                same_col = end[0].get("col") == infos[rep]["physical"].get(rank, {}).get("kda4_rec")
                res.setdefault("lifecycle", {})[f"{rep}:r{rank}"] = dict(last_write_equals_point0=same_bytes,
                                                                          same_column=same_col)
    # prior holders (keyed requests only) of every captured physical block: allocation history for the readings
    order = [d for d in R.p4 if d.get("kind") == "blocks" and d.get("rank", 0) == 0]
    ph = {}
    for rep in counted:
        wtag, _c = tags(rep, R.loc[rep])
        wpos = next((i for i, d in enumerate(order) if d.get("tag") == wtag), None)
        if wpos is None:
            continue
        for rank in (0, 1):
            for sname, b in (infos[rep]["physical"].get(rank) or {}).items():
                prev = sorted({d["tag"] for d in order[:wpos] if any(b in g for g in d.get("groups", []))})
                if prev:
                    ph.setdefault(rep, {})[f"r{rank}:{sname}"] = dict(b=b, earlier_holders=prev)
    res["prior_holders"] = dict(by_replicate=ph, note="keyed requests of this session only; start-up warm-up and "
                                "sentinels are not recorded")
    # ---------------- reading
    pairs = res["pairs"]
    if any(p["control_differ"] for p in pairs.values()):
        res["notes"].append("a control slice differs (MLA layer 3 or KDA layer 2): the layer-order premise (layers 0-3 "
                            "equal) does not hold in this run")
    l0 = [k for k, p in pairs.items() if any(s.endswith("kda0_conv_live") for s in p["valid_differ"])]
    l0i = [k for k, p in pairs.items() if any(s.endswith("kda0_conv_inactive") for s in p["other_differ"])]
    res["conv_rows_test"] = dict(pairs_live_rows_0_2_differ=l0, pairs_inactive_rows_3_5_differ=l0i,
                                 reading=("stale-row inference supported: layer-0 conv differences only in rows 3-5"
                                          if l0i and not l0 else "stale-row inference refuted: live rows 0-2 of layer 0 differ"
                                          if l0 else "no layer-0 conv difference in this run"))
    lc = res.get("lifecycle", {})
    if any(not v["last_write_equals_point0"] for v in lc.values()):
        res["notes"].append("the warm's last write of the layer-4 cached state differs from the bytes read at point 0 "
                            "in the same replicate: the state changed between its write and point 0 (lifecycle)")
    if not any_valid:
        res["cls"] = "equal"
    else:
        lp = {k: p for k, p in pairs.items() if p["loc_pair"] and p["valid_differ"]}
        firsts = {k: (p.get("loc_first") or {}).get("point") for k, p in lp.items()}
        res["loc_reading"] = dict(differing_loc_pairs=sorted(lp), first_point=firsts)
        if not lp:
            res["cls"] = "slice"
            any_lp = [k for k, p in pairs.items() if p["loc_pair"]]
            res["why_not_located"] = ("no localisation pair passed its gates (localisation not run, missing or "
                                      "invalid): the operation cannot be read" if not any_lp else
                                      "the pairs that differ in valid state have no localisation on both sides")
        elif any(v is None for v in firsts.values()):
            res["cls"] = "slice"
            silent = [k for k, v in firsts.items() if v is None]
            reps_silent = {r for k in silent for r in k.split("-")}
            if any(not v["last_write_equals_point0"] for key, v in lc.items() if key.split(":")[0] in reps_silent):
                res["why_not_located"] = ("no warm boundary or state write differs, but the layer-4 cached state at point 0 "
                                          "is not the bytes the warm last wrote (lifecycle): the producing event lies "
                                          "between the warm's write and point 0, outside the warm prefill")
            else:
                res["why_not_located"] = ("a localisation pair differs in valid state but no measured boundary differs: "
                                          "the difference arises outside layers 3-4 / the measured tokens")
        elif len(set(firsts.values())) > 1:
            res["cls"] = "slice"
            res["why_not_located"] = "localisation pairs disagree on the first differing boundary: %s" % firsts
        else:
            pt = next(iter(firsts.values()))
            if pt in ("L3.attn_in", "L4.in"):
                res["cls"] = "incomplete"
                res["why"] = [f"first differing boundary {pt} is an identity of an earlier boundary (instrument)"]
            elif pt in ("L3.in", "L2.state_end"):
                res["cls"] = "slice"
                res["why_not_located"] = "the difference is already present at %s: it enters before layer 3" % pt
            else:
                res["cls"] = "located"
                f0 = next(iter(lp.values()))["loc_first"]
                res["located"] = dict(boundary=pt, operation=LY.LOC_OPERATION.get(pt), chunk=f0["chunk"],
                                      step_pos0=f0["step_pos0"], hits=f0["hits"])
                if pt == "L3.topk":
                    ts = [t for t in (topk_sets(locd[r][0][0]) for r in locd if not locd[r][0][1].get("is_ref")) if t]
                    if ts:
                        tot = {k: sum(t[k] for t in ts) for k in ("rows", "same_set_other_order", "different_selection")}
                        tot["reading"] = ("the indexer emits the SAME token set in a different ORDER; the sparse-MLA "
                                          "consumer is order-sensitive (the replay must test the top-k kernel's output "
                                          "order and the attention's dependence on it)" if tot["rows"] and
                                          not tot["different_selection"] else
                                          "the indexer SELECTS different tokens (check the never-written pool records "
                                          "and allocation dependence)" if tot["different_selection"] else "no rows saved")
                        res["located"]["topk_sets"] = tot
                # causality: no earlier MLA-7 token may differ than the located step's first differing token
                firsts_l7 = [c.get("first_pos") for p in lp.values() for k, c in p["slices"].items()
                             if "mla7" in k and not c.get("equal", True) and c.get("first_pos") is not None]
                fp = min([h.get("first_pos", f0["step_pos0"]) for h in f0["hits"]] or [f0["step_pos0"]])
                if firsts_l7 and min(firsts_l7) < fp:
                    res["cls"] = "slice"
                    res["why_not_located"] = ("causality check failed: layer-7 KV differs at position %d, before the "
                                              "located boundary's first differing token %d" % (min(firsts_l7), fp))
    # allocation reading (assessor section 5)
    sa = [k for k, p in pairs.items() if p["same_alloc"]]
    da = [k for k, p in pairs.items() if not p["same_alloc"]]
    res["allocation"] = dict(
        same_allocation_pairs=sa, different_allocation_pairs=da,
        same_alloc_differ=[k for k in sa if pairs[k]["valid_differ"]],
        diff_alloc_differ=[k for k in da if pairs[k]["valid_differ"]],
        reading=("same inputs and the same allocation differ: execution nondeterminism or incomplete initialisation"
                 if any(pairs[k]["valid_differ"] for k in sa) else
                 "allocation-dependent: same-allocation pairs equal, relocated pairs differ" if sa and
                 any(pairs[k]["valid_differ"] for k in da) else
                 "every pair has a different allocation: allocation dependence and execution nondeterminism are not "
                 "separated by this run" if not sa else "no valid difference"))
    lp_loc = [k for k, p in pairs.items() if p["loc_pair"]]
    nl = [k for k, p in pairs.items() if not R.loc[k.split("-")[0]] and not R.loc[k.split("-")[1]]]
    res["instrumentation"] = dict(
        loc_pairs=lp_loc, loc_pairs_differ=[k for k in lp_loc if pairs[k]["valid_differ"]],
        noloc_pairs=nl, noloc_pairs_differ=[k for k in nl if pairs[k]["valid_differ"]],
        reading=("localisation hooks implicated: loc pairs and no-loc pairs disagree on whether valid state differs"
                 if lp_loc and nl and bool([k for k in lp_loc if pairs[k]["valid_differ"]]) !=
                 bool([k for k in nl if pairs[k]["valid_differ"]]) else
                 "loc and no-loc pairs agree" if lp_loc and nl else "no loc/no-loc contrast in this run"))
    detail = {k: v for k, v in pairs.items()}
    return res, detail


def render(res):
    cls = res.get("cls", "incomplete")
    row, nxt = ASSESSOR[cls]
    L = [f"# capture-p0 result: {CLASSES[cls]}", "",
         f"- Assessor 22ax table row: **{row}**", f"- Next action: {nxt}",
         f"- Replicates: {res.get('replicates')} (localisation: {res.get('loc')}); counted: {res.get('counted')}", ""]
    for w in res.get("why", []):
        L.append(f"- INCOMPLETE because: {w}")
    L += ["## Gates", "", "| Replicate | Counted | Failures |", "|---|---|---|"]
    for rep, g in res.get("gates", {}).items():
        f = "; ".join(g.get("failures", []) + g.get("loc_failures", []))[:600] or "none"
        L.append(f"| {rep} | {'yes' if g.get('ok') else 'no'} | {f} |")
    if res.get("pairs"):
        L += ["", "## Pairs (exact bytes)", "", "| Pair | Same allocation | Valid slices differing | Controls differing | "
              "Other differing | First differing boundary |", "|---|---|---|---|---|---|"]
        for k, p in res["pairs"].items():
            fl = p.get("loc_first") or {}
            L.append(f"| {k} | {p['same_alloc']} | {', '.join(p['valid_differ']) or 'none'} | "
                     f"{', '.join(p['control_differ']) or 'none'} | {', '.join(p['other_differ']) or 'none'} | "
                     f"{(fl.get('point') + ' (chunk ' + str(fl.get('chunk')) + ')') if fl else ('n/a' if not p['loc_pair'] else 'none')} |")
        first = next(iter(res["pairs"].values()))
        for s in ("r0:mla7_b0", "r0:mla7_b1", "r0:mla7_b2"):
            c = first["slices"].get(s, {})
            if c and not c.get("equal", True):
                L.append(f"- {s} (first pair): {c.get('tokens_differ')} of 4352 token records differ; first position "
                         f"{c.get('first_pos')}; by warm chunk {c.get('by_chunk')}; latent-only {c.get('latent_only')}, "
                         f"scale-only {c.get('scale_only')}, both {c.get('both')}; max |decoded diff| {c.get('max_abs_decoded')}")
    for k in ("conv_rows_test", "allocation", "instrumentation", "located", "loc_reading", "lifecycle"):
        if k in res:
            L.append(f"- **{k}**: `{json.dumps(res[k], default=str)[:900]}`")
    if res.get("why_not_located"):
        L.append(f"- Operation not located: {res['why_not_located']}")
    for n in res.get("notes", []):
        L.append(f"- Note: {n}")
    L += ["", "Limits: equal bytes in one pair do not establish repeatability; a located boundary is where the "
              "difference first appears in this serving run, not a replay of that operation and not a clearance of "
              "the rest of the serving pipeline (PROTOCOL.md section 8)."]
    return "\n".join(L) + "\n"


def main():
    ao, po = sys.argv[1], sys.argv[2]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    res, detail = analyse(ao, po)
    res["class"] = CLASSES[res.get("cls", "incomplete")]
    res["assessor_row"], res["next_action"] = ASSESSOR[res.get("cls", "incomplete")]
    dest = out or ao
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "cap0-result.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    with open(os.path.join(dest, "cap0-result.md"), "w") as f:
        f.write(render(res))
    if out is None and os.path.isdir(po):
        old = os.umask(0o077)
        try:
            with open(os.path.join(po, "cap0-detail.json"), "w") as f:
                json.dump(detail, f, indent=1, default=str)
        finally:
            os.umask(old)
    if "--quiet" not in sys.argv:
        print("CAP0: %s (%s)" % (res["class"], res["assessor_row"]))
        for w in res.get("why", [])[:6]:
            print("  why:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
