"""Shared, frozen helpers of the first-contrast gate (gate_fc.py) and analysis (analyze_fc.py). CPU only.

Provenance: the index-K decoding helpers (idx_pool, attn_blocks, idx_layers) are copied verbatim from the validated
attempt-1/2 analysis (hook-validation/analysis/analyze_hookval.py.a1, sha256 recorded in FREEZE.md); the content
rule, matching, flags, page/early structural rules, the H2-style alignment rule and the KDA identification screen
carry the attempt-2 (21g/21h) thresholds unchanged. New here: the conv-state functions of the [A22-5] boundary
control (validated on attempt-2 V3 snapshots in conv-a2/), position-level flags (R3) and the [A22-4] prefix sentinel.
"""
import collections
import glob
import json
import os
import re

import numpy as np
import torch

# ---------------------------------------------------------------- frozen constants (attempt-2 values, unchanged)
CLOSE = 154842
STOPS = (154820, 154827, 154829)
TERM_COL = {154842: 0, 154820: 1, 154827: 2, 154829: 3}   # hook column index of each tracked termination id
GROSS = 0.25                 # index-K pool rel L2 above which a pool is gross (attempt-1 frozen)
CONTENT_FACTOR = 2.0         # gross <= max(2 x max(N_dd, N_pp), floor)   (21g-1)
CONTENT_FLOOR = 0.001        # operational tolerance, not a measured noise floor
ORIG_B1_GROSS_MAX = 0.001    # ORIGINAL attempt-1 criterion (reported beside the 21g rule)
NULL_COVER_MIN = 0.99
POP_MIN = 0.99
FLAG_MIN_GROSS = 10          # fixture flag: >= 10 gross layer-units ...
FLAG_MIN_POS = 3             # ... or (R3) >= 3 gross positions
FLAG_FACTOR = 2.0
PAGE_MINP, PAGE_MAX_GROSS = 16, 0.25
EARLY_IDX_LAYERS = 15
TOP1_MIN = 0.95              # H2a: pooled aligned top-1
SHIFT_ALIGNED_MIN = 0.90     # H2c-r per request (and per crossing window here)
SHIFT_SHIFTED_MAX = 0.20
NULL_FACTOR = 2.0
FLOOR = {0.5: 0.01, 0.9: 0.30, 0.99: 1.50}
H2_NULL = {0.5: 6.659261998720467e-05, 0.9: 0.09682364463806155, 0.99: 0.40648360252380433}  # carryover-a1.json
KDA_SHALLOW_MAX, KDA_DEEP_IDENT_MIN = 17, 0.99
KDA_EARLY_LAYERS, KDA_EARLY_REL, KDA_EARLY_MARGIN, KDA_SEP_FLAG = (0, 1, 2), 0.02, 10.0, 1.25
K = 3                        # DFlash2 draft depth
CONV_STATE_LEN = 6           # KDA conv state rows: conv kernel 4 - 1 + num_spec 3 (kda.py get_state_shape)


def q(a, p):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, p)) if a.size else float("nan")


def jl(p):
    out = []
    if os.path.exists(p):
        for ln in open(p):
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    return out


def safe(s):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))[:80]


def layer_no(name):
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


# ---------------------------------------------------------------- run loader
class Run:
    def __init__(self, d):
        self.d = d
        self.CL = jl(os.path.join(d, "client.jsonl"))
        self.reqs = [r for r in self.CL if r.get("kind") == "request"]
        self.by_tag = {}
        for r in self.reqs:
            self.by_tag.setdefault(r["tag"], r)
        self.REQP = {}
        for p in glob.glob(os.path.join(d, "forcewin", "req_*_r*.pt")):
            m = re.search(r"req_(.*)_r(\d+)\.pt$", os.path.basename(p))
            self.REQP.setdefault(m.group(1), {})[int(m.group(2))] = p
        self.SNAP = {}
        for p in glob.glob(os.path.join(d, "forcewin", "snap_*_r*.pt")):
            m = re.search(r"snap_(.*)_F(\d+)(win)?_r(\d+)\.pt$", os.path.basename(p))
            self.SNAP.setdefault((m.group(1), int(m.group(2)), bool(m.group(3))), {})[int(m.group(4))] = p
        self.FWLOG = [r for p in sorted(glob.glob(os.path.join(d, "forcewin", "fw_r*.jsonl"))) for r in jl(p)]
        self.POLLS = sorted([r for p in glob.glob(os.path.join(d, "mapguard", "mg_p*.jsonl")) for r in jl(p)
                             if r.get("kind") == "poll"], key=lambda r: r.get("t", 0))
        self._cache = collections.OrderedDict()

    def load(self, p):
        if p in self._cache:
            self._cache.move_to_end(p)
            return self._cache[p]
        x = torch.load(p, weights_only=False)
        self._cache[p] = x
        while len(self._cache) > 48:
            self._cache.popitem(last=False)
        return x

    def req(self, tag, rank=0):
        d = self.REQP.get(safe(tag))
        if not d:
            return None
        return self.load(d[rank] if rank in d else next(iter(d.values())))

    def req_ranks(self, tag):
        return {k: self.load(p) for k, p in (self.REQP.get(safe(tag)) or {}).items()}

    def win_dump(self, tag):
        k = [v for kk, v in self.SNAP.items() if kk[0] == safe(tag) and kk[2]]
        return k[0] if k else None

    def snaps(self, tag):
        """[(F, {rank: path})] of the non-window snapshots of a request, by F."""
        return sorted((kk[1], v) for kk, v in self.SNAP.items() if kk[0] == safe(tag) and not kk[2])

    def steps_table(self, rank=0):
        """step -> (num_reqs, num_tokens, num_draft_tokens, n_prefilling, padded_tokens) from the hook step log."""
        t = {}
        for r in self.FWLOG:
            if r.get("kind") == "steps" and r.get("rank", 0) == rank:
                for row in r["rows"]:
                    t[int(row[0])] = tuple(row[1:])
        return t

    def hook_records(self, kind):
        return [r for r in self.FWLOG if r.get("kind") == kind]


def cols(rec):
    return {n: i for i, n in enumerate(rec["cols"])}


def metrics(rec, n=None):
    M = rec["metrics"].numpy().astype(np.float64)
    return M if n is None else M[:n]


# ---------------------------------------------------------------- index-K pages (verbatim from analyze_hookval.py.a1)
def idx_pool(pages, blk_of, pool):
    """Dequantized index-K record of logical pool `pool` (page-planar 64 x 128 E4M3 + 64 x fp32 scales)."""
    blk = blk_of((4 * pool + 3) // 4352)
    pg = blk * 17 + (pool % 1088) // 64
    t = pages.get(pg)
    if t is None:
        return None
    raw = t.reshape(-1).view(torch.uint8)
    rr = pool % 64
    v = raw[rr * 128:(rr + 1) * 128].view(torch.float8_e4m3fn).float()
    s = raw[8192 + rr * 4: 8192 + rr * 4 + 4].view(torch.float32)
    return v * s


def attn_blocks(rec):
    for g in rec["groups"]:
        if g["bpk"] > 1:
            ids = g["ids"].tolist()
            bpk = g["bpk"]
            return lambda c, ids=ids, bpk=bpk: ids[c * bpk] // bpk if c * bpk < len(ids) else -1
    return lambda c: -1


def idx_layers(rec):
    return {n: L for n, L in rec["layers"].items() if L.get("kind") == "paged" and L.get("shape", [0])[-1:] == [132]}


def paged_layers(rec):
    return {n: L for n, L in rec["layers"].items() if L.get("kind") == "paged"}


# ---------------------------------------------------------------- window comparisons
def compare_pools(run, tagA, tagB, pool_lo, pool_hi, attrib=None, pop=None, name=None):
    """{(rank, layer, pool): rel} over pools [pool_lo, pool_hi) between two window dumps (rel = ||A-B|| / ||B||)."""
    A, B = run.win_dump(tagA), run.win_dump(tagB)
    out, expected = {}, 0
    if not A or not B:
        if pop is not None:
            pop.setdefault(name, [0, 0, []])[2].append("%s vs %s: dump missing" % (tagA, tagB))
        return out
    for rank in sorted(A):
        a = run.load(A[rank])
        b = run.load(B[rank]) if rank in B else None
        if attrib is not None:
            attrib(tagA, rank, a)
            if b is not None:
                attrib(tagB, rank, b)
        fa = attn_blocks(a)
        fb = attn_blocks(b) if b is not None else None
        for nm, La in idx_layers(a).items():
            expected += max(0, pool_hi - pool_lo)
            Lb = b["layers"].get(nm) if b is not None else None
            if Lb is None:
                continue
            for pool in range(pool_lo, pool_hi):
                x, y = idx_pool(La["pages"], fa, pool), idx_pool(Lb["pages"], fb, pool)
                if x is None or y is None:
                    continue
                out[(rank, layer_no(nm), pool)] = float((x - y).norm() / max(float(y.norm()), 1e-12))
    if pop is not None:
        e = pop.setdefault(name, [0, 0, []])
        e[0] += len(out)
        e[1] += expected
    return out


def gross(vals):
    vals = list(vals)
    return float(np.mean([v > GROSS for v in vals])) if vals else float("nan")


def limit_of(units):
    """21g-1: L = max(2 x max(N_dd, N_pp), 0.1 %) on the matched units. [A22-3] correction: N_pp = 0 does NOT put
    the limit on the floor when N_dd > 0: the max over the two nulls is taken first."""
    ndd, npp = gross(u[3] for u in units), gross(u[4] for u in units)
    lim = max(CONTENT_FACTOR * max(ndd, npp), CONTENT_FLOOR) if units else float("nan")
    return lim, ndd, npp


def matched(tested, dd, pp):
    """tested/dd/pp: {fixture: {key: rel}} -> [(fixture, key, rel, rel_dd, rel_pp)] on keys present in BOTH nulls."""
    units, n_tested, fx = [], 0, set()
    for f, rows in tested.items():
        d_, p_ = dd.get(f, {}), pp.get(f, {})
        for k, rel in rows.items():
            n_tested += 1
            fx.add(f)
            if k in d_ and k in p_:
                units.append((f, k, rel, d_[k], p_[k]))
    return units, n_tested, sorted(fx)


def content(tested, dd, pp):
    units, n_tested, fx = matched(tested, dd, pp)
    lim, ndd, npp = limit_of(units)
    g = gross(u[2] for u in units)
    have = {u[0] for u in units}
    cov = len(units) / n_tested if n_tested else 0.0
    return dict(tested=n_tested, evaluated=len(units), coverage=cov, fixtures=fx,
                fixtures_without_null=[f for f in fx if f not in have], gross=g,
                gross_n=int(sum(u[2] > GROSS for u in units)), N_dd=ndd, N_pp=npp,
                N_dd_n=int(sum(u[3] > GROSS for u in units)), N_pp_n=int(sum(u[4] > GROSS for u in units)),
                limit=lim, orig_pass=bool(units) and g <= ORIG_B1_GROSS_MAX,
                cover_ok=bool(units) and cov >= NULL_COVER_MIN and not [f for f in fx if f not in have],
                _units=units)


def table(units):
    by_fl, by_f = collections.defaultdict(list), collections.defaultdict(list)
    for u in units:
        by_fl[(u[0], u[1][1])].append(u)
        by_f[u[0]].append(u)

    def summ(us):
        n = len(us)
        g = [int(sum(x[i] > GROSS for x in us)) for i in (2, 3, 4)]
        return dict(n=n, gross_n=g[0], gross=g[0] / n if n else None, N_dd_n=g[1], N_pp_n=g[2])
    return dict(pooled=summ(units), per_fixture={t: summ(v) for t, v in sorted(by_f.items())},
                per_fixture_layer={"%s|L%d" % (t, L): summ(v) for (t, L), v in sorted(by_fl.items())})


def positions_of(units, i):
    """R3: gross positions = (fixture, pool) gross (column i) in >= 1 DSA layer of either rank; denominators are
    the distinct (fixture, pool) of the units."""
    tot, g = collections.defaultdict(set), collections.defaultdict(set)
    for u in units:
        tot[u[0]].add(u[1][2])
        if u[i] > GROSS:
            g[u[0]].add(u[1][2])
    return tot, g


def flags(units):
    """21g fixture flag 'outside its own matched null', evaluated with layer-units (>= 10 gross units) and with
    positions (>= 3 gross positions) [R3]. A flag on either count marks the fixture."""
    out = []
    by_f = collections.defaultdict(list)
    for u in units:
        by_f[u[0]].append(u)
    for t, us in sorted(by_f.items()):
        lim_f, ndd, npp = limit_of(us)
        gn = int(sum(u[2] > GROSS for u in us))
        lim_flag = max(FLAG_FACTOR * max(ndd, npp), CONTENT_FLOOR)
        by_unit = gn >= FLAG_MIN_GROSS and gn / len(us) > lim_flag
        tot, gpos = positions_of(us, 2)
        _, gdd = positions_of(us, 3)
        _, gpp = positions_of(us, 4)
        npos = len(tot[t])
        rp, rdd, rpp = len(gpos[t]) / npos, len(gdd[t]) / npos, len(gpp[t]) / npos
        lim_pos = max(FLAG_FACTOR * max(rdd, rpp), CONTENT_FLOOR)
        by_pos = len(gpos[t]) >= FLAG_MIN_POS and rp > lim_pos
        if by_unit or by_pos:
            out.append(dict(fixture=t, n_units=len(us), gross_units=gn, unit_limit=lim_flag, by_unit=by_unit,
                            n_positions=npos, gross_positions=len(gpos[t]), position_limit=lim_pos, by_position=by_pos,
                            mechanism_claim=False))
    return out


def asym_flags(units_a, units_b):
    """Arm asymmetry between two clean decode arms (C1, C2) on units both arms and N_pp share; layer-units and
    positions [R3]."""
    out = []
    ua = collections.defaultdict(dict)
    for arm, us in (("a", units_a), ("b", units_b)):
        for t, k, rel, dd, pp in us:
            ua[t].setdefault(k, {})[arm] = (rel, pp)
    for t, d in sorted(ua.items()):
        ks = [k for k, v in d.items() if len(v) == 2]
        if not ks:
            continue
        na = sum(d[k]["a"][0] > GROSS for k in ks)
        nb = sum(d[k]["b"][0] > GROSS for k in ks)
        npp = float(np.mean([d[k]["a"][1] > GROSS for k in ks]))
        lim = max(FLAG_FACTOR * npp, CONTENT_FLOOR)
        hi_n, lo_n = max(na, nb), min(na, nb)
        by_unit = hi_n >= FLAG_MIN_GROSS and hi_n > FLAG_FACTOR * lo_n and hi_n / len(ks) > lim
        pa = {k[2] for k in ks if d[k]["a"][0] > GROSS}
        pb = {k[2] for k in ks if d[k]["b"][0] > GROSS}
        ppp = {k[2] for k in ks if d[k]["a"][1] > GROSS}
        npos = len({k[2] for k in ks})
        hp, lp = max(len(pa), len(pb)), min(len(pa), len(pb))
        lim_p = max(FLAG_FACTOR * len(ppp) / npos, CONTENT_FLOOR)
        by_pos = hp >= FLAG_MIN_POS and hp > FLAG_FACTOR * lp and hp / npos > lim_p
        if by_unit or by_pos:
            out.append(dict(fixture=t, n=len(ks), a_gross_units=int(na), b_gross_units=int(nb), a_positions=len(pa),
                            b_positions=len(pb), by_unit=by_unit, by_position=by_pos, mechanism_claim=False))
    return out


def page_stats(rows_by_tag):
    pg = collections.defaultdict(list)
    for tag, rows in rows_by_tag.items():
        for (rank, layer, pool), rel in rows.items():
            pg[(tag, rank, layer, pool // 64)].append(rel > GROSS)
    judged = {k: v for k, v in pg.items() if len(v) >= PAGE_MINP}
    worst = sorted(((float(np.mean(v)), len(v), k) for k, v in judged.items()), reverse=True)[:3]
    return dict(pages_judged=len(judged), page_max=worst[0][0] if worst else None,
                over=sum(1 for v in judged.values() if np.mean(v) > PAGE_MAX_GROSS),
                worst=[(round(f, 4), n, list(map(str, k))) for f, n, k in worst])


def early_gross(rows_by_tag):
    g = n = 0
    for rows in rows_by_tag.values():
        for (rk, L, p), rel in rows.items():
            if 3 <= L <= EARLY_IDX_LAYERS:
                n += 1
                g += rel > GROSS
    return g, n


# ---------------------------------------------------------------- H2-style alignment against a prefill reference
def align(dec, ref, o_lo, o_hi, shift=0):
    """decode row o (predicts P + o) vs reference row k = o (predicts ref + k; ref = P): forced-token |dlp| and argmax
    agreement on rows where the reference is finite. Returns (dlp array, agree array, n)."""
    Cc = cols(dec)
    M = metrics(dec)
    Rf = metrics(ref)
    o = np.arange(max(0, o_lo), min(o_hi, M.shape[0]))
    k = o + shift
    ok = (k >= 0) & (k < Rf.shape[0])
    o, k = o[ok], k[ok]
    y = Rf[k, Cc["lp_pre_4"]]
    fin = np.isfinite(y) & np.isfinite(M[o, Cc["lp_pre_4"]])
    d = np.abs(M[o, Cc["lp_pre_4"]] - y)[fin]
    agree = (M[o, Cc["argmax"]] == Rf[k, Cc["argmax"]])[fin]
    return d, agree, int(fin.sum())


def align_summary(dec, ref, o_lo, o_hi):
    d, a, n = align(dec, ref, o_lo, o_hi)
    _, ap, _ = align(dec, ref, o_lo, o_hi, +1)
    _, am, _ = align(dec, ref, o_lo, o_hi, -1)
    t1 = float(np.mean(a)) if a.size else float("nan")
    sh = max(float(np.mean(ap)) if ap.size else 0.0, float(np.mean(am)) if am.size else 0.0)
    return dict(n=n, top1=t1, shifted=sh, lp={str(p): q(d, p) for p in FLOOR}, _d=d, _a=a)


def align_tol():
    return {p: max(NULL_FACTOR * H2_NULL[p], FLOOR[p]) for p in FLOOR}


# ---------------------------------------------------------------- KDA conv state (A22-5; validated in conv-a2/)
def kda_layers(snap):
    return {n: L for n, L in snap["layers"].items() if L.get("kind") == "kda"}


def conv_entries(snap, L):
    """{absolute position: conv input row} of the COMMITTED conv-state entries recorded in a snapshot.
    Speculative (decode) snapshot, after a step that accepted a drafts (num_accepted = a + 1, model state after the
    step): state rows e = 0..5 hold the inputs of positions p0 - 2 + e with p0 = frontier - a; rows 0..a+2 are committed
    (positions <= frontier). Non-speculative (prefill reference / sentinel) snapshot: rows 0..2 hold the last three
    prompt positions frontier - 2 .. frontier (rows 3..5 are not written by the prefill kernel). Established on
    attempt-2 V3 snapshots (conv-a2/conv_a2.json): layer-0 rel 0.003 at the mapped rows, >= 0.8 elsewhere."""
    c = L["conv"]
    if c.dim() != 2:
        return {}
    if c.shape[0] != CONV_STATE_LEN and c.shape[1] == CONV_STATE_LEN:      # dim-first layout -> (state rows, dim)
        c = c.t()
    f = int(snap["frontier_pos"])
    if L.get("is_spec"):
        na = snap.get("num_accepted")
        if not isinstance(na, int) or na < 1:
            return {}
        a = na - 1
        p0 = f - a
        return {p0 - 2 + e: c[e] for e in range(0, min(a + 3, c.shape[0]))}
    return {f - 2 + e: c[e] for e in range(3)}


def rel(x, y):
    x, y = x.float(), y.float()
    return float((x - y).norm() / max(float(y.norm()), 1e-12))


def conv_ident(snapX, snapR):
    """Nearest-entry identification of X's committed conv rows against the reference R's committed rows, per KDA
    layer and shared position: [(layer, position, rel_same, nearest_position, margin)]."""
    out = []
    LX, LR = kda_layers(snapX), kda_layers(snapR)
    for nm, Lx in LX.items():
        Lr = LR.get(nm)
        if Lr is None or "conv" not in Lx or "conv" not in Lr:
            continue
        ex, er = conv_entries(snapX, Lx), conv_entries(snapR, Lr)
        for p, v in ex.items():
            if p not in er:
                continue
            d = {pr: rel(v, w) for pr, w in er.items()}
            near = min(d, key=d.get)
            others = [x for pr, x in d.items() if pr != p]
            out.append(dict(layer=layer_no(nm), pos=p, rel=d[p], nearest=near, ident=near == p,
                            margin=(min(others) / max(d[p], 1e-12)) if others else float("inf")))
    return out


def conv_persist(snaps_sorted):
    """Exact persistence: for consecutive snapshots of one request, conv rows of shared committed positions are
    bitwise identical in every KDA layer (a committed position's input is written once and only moved by byte copies,
    including the align-mode block-column migration). Returns (pairs, rows_compared, rows_identical, straddles)."""
    pairs, n, same, straddle, bad = 0, 0, 0, 0, []
    for (Fa, a), (Fb, b) in zip(snaps_sorted, snaps_sorted[1:]):
        la, lb = kda_layers(a), kda_layers(b)
        ca, cb = col_of(a), col_of(b)
        shared_any = False
        for nm, A in la.items():
            B = lb.get(nm)
            if B is None:
                continue
            ea, eb = conv_entries(a, A), conv_entries(b, B)
            for p in sorted(set(ea) & set(eb)):
                shared_any = True
                n += 1
                ok = bool(torch.equal(ea[p], eb[p]))
                same += ok
                if not ok and len(bad) < 10:
                    bad.append((Fa, Fb, layer_no(nm), p))
        pairs += 1
        if shared_any and ca is not None and cb is not None and ca != cb:
            straddle += 1
    return dict(pairs=pairs, rows=n, identical=same, straddling_pairs=straddle, mismatches=bad)


def col_of(snap):
    """Align-mode running block column of a speculative snapshot's step: (p0 + K) // 4352, p0 = frontier - a."""
    na = snap.get("num_accepted")
    if not isinstance(na, int):
        return None
    p0 = int(snap["frontier_pos"]) - (na - 1)
    return (p0 + K) // 4352


def slot_check(snap):
    """[A22-5] actual conv-state slot: every KDA layer's speculative slots equal the run of K + 1 block ids starting
    at the running column in exactly one mamba group (block size 4352, the align layout of mamba_get_block_table_
    tensor); the conv row snapshot is slots[0] = spec_state_indices_tensor[:, 0] = the conv_state_indices the engine
    passes to causal_conv1d_update (kda.py). Returns list of violations."""
    bad = []
    col = col_of(snap)
    groups = [g for g in snap["groups"] if g["block_size"] == 4352 and g["bpk"] == 1]
    for nm, L in kda_layers(snap).items():
        if not L.get("is_spec") or "slots" not in L:
            bad.append((layer_no(nm), "not speculative"))
            continue
        s = L["slots"].tolist()
        hits = [g["g"] for g in groups if g["ids"].tolist()[col: col + len(s)] == s]
        if len(hits) != 1:
            bad.append((layer_no(nm), "slots %s not the column-%s run of exactly one group (%s)" % (s, col, hits)))
    return bad


def kda_ident(snapX, snapR):
    """H4b-style nearest-slot identification of X's committed SSM slot (num_accepted - 1) against reference R's
    committed state (R speculative: its committed slot; R non-speculative: its single slot)."""
    out = []
    na = snapX.get("num_accepted")
    for nm, Lx in kda_layers(snapX).items():
        Lr = kda_layers(snapR).get(nm)
        if Lr is None or "ssm_sketch" not in Lx or "ssm_sketch" not in Lr or not isinstance(na, int):
            continue
        if Lr.get("is_spec"):
            nr = snapR.get("num_accepted")
            ref = Lr["ssm_sketch"][nr - 1]
        else:
            ref = Lr["ssm_sketch"][0]
        dist = [float((s - ref).norm() / max(float(ref.norm()), 1e-12)) for s in Lx["ssm_sketch"]]
        c = dist[na - 1]
        others = [x for k_, x in enumerate(dist) if k_ != na - 1]
        near = int(np.argmin(dist))
        out.append(dict(layer=layer_no(nm), rel=c, committed=na - 1, nearest=near, ident=near == na - 1,
                        dists=[round(x, 6) for x in dist],
                        margin=(min(others) / max(c, 1e-12)) if others else float("inf"),
                        sep=(c / max(min(others), 1e-12)) if others else 0.0))
    return out


def ident_screen(rows):
    """21g-2 screen over identification rows: 100 % in layers 0-17, >= 99 % deeper; layers 0-2 rel <= 0.02 and
    margin >= 10. Returns (ok, summary)."""
    if not rows:
        return False, dict(n=0)
    sh = [r for r in rows if r["layer"] <= KDA_SHALLOW_MAX]
    dp = [r for r in rows if r["layer"] > KDA_SHALLOW_MAX]
    ea = [r for r in rows if r["layer"] in KDA_EARLY_LAYERS]
    i_sh = float(np.mean([r["ident"] for r in sh])) if sh else None
    i_dp = float(np.mean([r["ident"] for r in dp])) if dp else None
    erel = max([r["rel"] for r in ea], default=float("nan"))
    emar = min([r["margin"] for r in ea], default=float("nan"))
    ok = (i_sh == 1.0 and (i_dp is None or i_dp >= KDA_DEEP_IDENT_MIN) and bool(ea) and erel <= KDA_EARLY_REL
          and emar >= KDA_EARLY_MARGIN)
    dis = [r for r in rows if not r["ident"]]
    return ok, dict(n=len(rows), ident_0_17=i_sh, ident_18_plus=i_dp, early_rel_max=erel, early_margin_min=emar,
                    disagreements=len(dis), disagreement_examples=dis[:5],
                    rel_q50=q([r["rel"] for r in rows], 0.5), rel_max=max(r["rel"] for r in rows))
