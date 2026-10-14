"""K: fp64 reference, fp32 evaluation bound and the frozen selection criteria (device-agnostic torch).

Score (the quantity every B12x route ranks; attention.py v091:63-67, :391-396; b12x reference.py):
    s(p) = scale_p * sum_h w_h * relu( sum_d q_hd * k_pd )
with q, k e4m3 bytes, w and scale fp32. The reference decodes the exact e4m3/fp32 values to fp64.

fp32 evaluation bound (phase2-stage1/fixture.py:171-186, unchanged): any fp32 evaluation order of the
128-term dots, the relu, the 32-term head sum and two multiplies differs from the exact value by at most
    B(p) = 162 * 2^-24 * |scale_p| * sum_h |w_h| * sum_d |q_hd| |k_pd|.
Measured kernel score error on stage 1 was <= 0.0047 B. fp64 rounding of the reference itself is
<= 162 * 2^-53 * (same sum), i.e. 2^-29 B, and is ignored. For the exact families every partial sum
is exactly representable in fp32 (validated on CPU), so B is replaced by 0 there.

Frozen criteria (PROPOSAL.md section 4), per row:
  C1 structure/eligibility: exactly n = min(n_slots, P) ids in [0, P); no duplicates; the rest -1;
     nothing < -1. (P = the row's live pool count; everything >= P is ineligible, incl. poison.)
  C2 set: let t = n-th largest reference score.
     exact families : every s > t selected; nothing with s < t selected  (ties at t: any subset).
     other families : every s > t + B + B_t selected; nothing with s < t - B - B_t selected.
  C2b kernel scores of selected ids: exact families equal the reference; others within B.
  C3 (B1 policy) column 511 after _move_lowest_selected_pool_last: the multiset is unchanged; the
     column-511 id carries the minimum kernel score; its reference score is the selected minimum
     (exact) or within B of it; when P < 512 column 511 is -1; and columns 0..510 pass C1/C2 as a
     top-min(511, P) set.
  C4 expanded output (rows, 2047) == independent reference of expand + tail.
"""
from __future__ import annotations

import torch

TOPK, KEEP, HEADS, HD, KPOOL = 512, 511, 32, 128, 4
EPS32 = 2.0 ** -24
BOUND_C = 162


def e4m3(u8: torch.Tensor) -> torch.Tensor:
    return u8.contiguous().view(torch.float8_e4m3fn).to(torch.float64)


def ref_scores(q8, w, K8, S, chunk=8192):
    """q8 (R,32,128) uint8, w (R,32) f32, K8 (P,128) uint8, S (P,) f32 ->
    s64 (R,P) float64, bound (R,P) float64 (fp32 evaluation bound)."""
    R, P = q8.shape[0], K8.shape[0]
    q = e4m3(q8)                                   # (R,32,128)
    qa = q.abs()
    wd = w.to(torch.float64)
    wa = wd.abs()
    s_out = torch.empty((R, P), dtype=torch.float64, device=q8.device)
    b_out = torch.empty((R, P), dtype=torch.float64, device=q8.device)
    for a in range(0, P, chunk):
        e = min(P, a + chunk)
        k = e4m3(K8[a:e])                          # (n,128)
        sc = S[a:e].to(torch.float64)
        dots = torch.einsum("rhd,nd->rhn", q, k)
        s_out[:, a:e] = (torch.relu(dots) * wd[:, :, None]).sum(1) * sc[None, :]
        ad = torch.einsum("rhd,nd->rhn", qa, k.abs())
        b_out[:, a:e] = (ad * wa[:, :, None]).sum(1) * sc.abs()[None, :] * (BOUND_C * EPS32)
    return s_out, b_out


def nth_largest(s: torch.Tensor, n: int):
    v, i = torch.topk(s, n, largest=True, sorted=True)
    return v[n - 1], i[n - 1]


def check_set_ref(ids: torch.Tensor, s: torch.Tensor, b: torch.Tensor | None, exact: bool, n_slots: int):
    """ids (n_slots,) int; s (P,) fp64 reference over the row's eligible pools; b (P,) bound or None.
    Returns a dict with C1 (struct_ok) and C2 (set_ok) verdicts plus diagnostics."""
    ids = ids.to(torch.int64)
    P = int(s.shape[0])
    n = min(n_slots, P)
    valid = ids >= 0
    bad_neg = int((ids < -1).sum())
    oob = int((valid & (ids >= P)).sum())
    inr = valid & (ids < P)
    sel = ids[inr]
    n_inr = int(sel.numel())
    dup = n_inr - int(torch.unique(sel).numel())
    struct_ok = (n_inr == n) and oob == 0 and dup == 0 and bad_neg == 0
    out = dict(P=P, n=n, n_valid=n_inr, oob=oob, dup=dup, bad_neg=bad_neg, struct_ok=bool(struct_ok))
    if n == 0:
        out.update(set_ok=bool(struct_ok), missing=0, forbidden=0, n_band=0, t=None)
        return out
    t, it = nth_largest(s, n)
    mask = torch.zeros(P, dtype=torch.bool, device=s.device)
    mask[sel] = True
    if exact or b is None:
        must = s > t
        allowed = s >= t
    else:
        bt = b[it]
        must = s > t + b + bt
        allowed = s >= t - b - bt
    missing = int((must & ~mask).sum())
    forbidden = int((mask & ~allowed).sum())
    out.update(set_ok=bool(struct_ok and missing == 0 and forbidden == 0), missing=missing,
               forbidden=forbidden, n_band=int((allowed & ~must).sum()), t=float(t),
               n_eq_t=int((s == t).sum()))
    if P > n:
        t2, _ = nth_largest(s, n + 1)
        out["gap_n_n1"] = float(t - t2)
    return out


def check_kernel_scores_ref(ids, ks, s, b, exact):
    """C2b: kernel scores of the selected ids against the reference. Returns max err/B (inf if a
    nonzero error meets B == 0) and exact-mismatch count; plus the unfilled-slot score values."""
    ids = ids.to(torch.int64)
    P = int(s.shape[0])
    inr = (ids >= 0) & (ids < P)
    out = dict(n=int(inr.sum()))
    if out["n"]:
        idx = ids[inr]
        k = ks[inr].to(torch.float64)
        r = s[idx]
        err = (k - r).abs()
        nonfinite = int((~torch.isfinite(k)).sum())
        out["nonfinite"] = nonfinite
        out["exact_mismatch"] = int((k != r).sum())
        if b is not None:
            bb = b[idx]
            ratio = torch.where(bb > 0, err / bb, torch.where(err > 0, torch.full_like(err, float("inf")),
                                                             torch.zeros_like(err)))
            out["err_over_bound_max"] = float(ratio.max())
        out["ok"] = bool(nonfinite == 0 and (out["exact_mismatch"] == 0 if exact else
                                             out.get("err_over_bound_max", 0.0) <= 1.0))
    else:
        out.update(nonfinite=0, exact_mismatch=0, ok=True)
    unf = ks[~(ids >= 0)]
    out["unfilled"] = int(unf.numel())
    out["unfilled_all_neginf"] = bool(unf.numel() == 0 or bool(torch.isneginf(unf).all()))
    return out


def check_col511_ref(pre_ids, pre_ks, post_ids, s, b, exact):
    """C3 for the B1 policy. pre_* = kernel output before _move_lowest_selected_pool_last."""
    pre_ids = pre_ids.to(torch.int64)
    post_ids = post_ids.to(torch.int64)
    P = int(s.shape[0])
    out = {}
    out["multiset_same"] = bool(torch.equal(torch.sort(pre_ids).values, torch.sort(post_ids).values))
    c = int(post_ids[KEEP])
    out["col511_id"] = c
    n = min(TOPK, P)
    if n < TOPK:
        # fewer live pools than slots: an unfilled slot (-1) must end in column 511 so that columns
        # 0..510 keep every live pool (checked separately as the first-511 set). Kernel scores of
        # unfilled slots are reported by check_kernel_scores, not required here.
        out["col511_kernel_min"] = True
        out["col511_unfilled_ok"] = c == -1
        out["fp64_min_ok"] = True
    else:
        m = pre_ks.min()
        pos = (pre_ids == c).nonzero().flatten()
        out["col511_kernel_min"] = bool(pos.numel() > 0 and bool((pre_ks[pos] == m).any()))
        out["col511_unfilled_ok"] = True
        sel = post_ids[(post_ids >= 0) & (post_ids < P)]
        if 0 <= c < P and sel.numel():
            ss = s[sel]
            j = int(torch.argmin(ss))
            smin = ss[j]
            if exact or b is None:
                ok = bool(s[c] == smin)
            else:
                ok = bool(s[c] <= smin + b[c] + b[sel[j]])
            out["fp64_min_ok"] = ok
            out["col511_rank_in_selected"] = int((ss > s[c]).sum())   # 511 = lowest
        else:
            out["fp64_min_ok"] = False
    out["ok"] = bool(out["multiset_same"] and out["col511_kernel_min"] and out["col511_unfilled_ok"]
                     and out["fp64_min_ok"])
    return out


def expand_ref(ids511: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Independent reference of expand_pools_to_tokens + append_tail_to_topk (identity path):
    history columns 4g+o = 4*id+o for 0 <= id < L//4 else -1; tail columns 2044..2046 =
    4*(L//4)+t for t < L%4 else -1. ids511 (R,511), L (R,) token lengths."""
    ids = ids511.to(torch.int64)
    R = ids.shape[0]
    L = L.to(torch.int64)
    P = (L // KPOOL)[:, None]
    o = torch.arange(KPOOL, device=ids.device)
    hist = ids[:, :, None] * KPOOL + o
    ok = ((ids >= 0) & (ids < P))[:, :, None].expand(-1, -1, KPOOL)
    hist = torch.where(ok, hist, torch.full_like(hist, -1)).reshape(R, -1)
    t = torch.arange(KPOOL - 1, device=ids.device)[None, :]
    tail = torch.where(t < (L % KPOOL)[:, None], P * KPOOL + t, torch.full_like(t, -1).expand(R, -1))
    return torch.cat([hist, tail], 1).to(torch.int32)


def check_expanded(E: torch.Tensor, ids511: torch.Tensor, L: torch.Tensor):
    """C4: exact equality with expand_ref, plus semantic checks of the attended token set per row
    (every token < L, no token twice, exactly L % 4 tail tokens). One host transfer."""
    ref = expand_ref(ids511, L)
    E = E.to(torch.int64)
    L = L.to(torch.int64)
    eq = (E == ref.to(torch.int64)).all()
    live = E >= 0
    too_big = (live & (E >= L[:, None])).any(1)
    srt = torch.sort(torch.where(live, E, torch.full_like(E, -1)), dim=1).values
    dup = ((srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0)).any(1)
    tail_bad = (E[:, KEEP * KPOOL:] >= 0).sum(1) != (L % KPOOL)
    bad = too_big | dup | tail_bad
    v = torch.cat([eq.reshape(1).to(torch.int64), bad.to(torch.int64)]).tolist()
    out = dict(equal=bool(v[0]), rows_bad=[i for i, x in enumerate(v[1:]) if x])
    out["ok"] = out["equal"] and not out["rows_bad"]
    return out


# ------------------------------------------------------------------------------------------------
# Sync-light versions used by the GPU run (one host transfer per call). Same keys and semantics as the
# *_ref versions above; test_cpu.py T3b checks they agree on random and mutated rows.
# ------------------------------------------------------------------------------------------------
def _d(x):
    return x.to(torch.float64).reshape(())


def check_set(ids, s, b, exact, n_slots):
    ids = ids.to(torch.int64)
    P = int(s.shape[0])
    n = min(n_slots, P)
    valid = ids >= 0
    bad_neg = (ids < -1).sum()
    oob = (valid & (ids >= P)).sum()
    inr = valid & (ids < P)
    n_inr = inr.sum()
    srt = torch.sort(torch.where(inr, ids, torch.full_like(ids, -1))).values
    dup = ((srt[1:] == srt[:-1]) & (srt[1:] >= 0)).sum()
    if n == 0:
        v = torch.stack([_d(n_inr), _d(oob), _d(dup), _d(bad_neg)]).tolist()
        n_v, oob_v, dup_v, neg_v = (int(x) for x in v)
        ok = n_v == 0 and oob_v == 0 and dup_v == 0 and neg_v == 0
        return dict(P=P, n=0, n_valid=n_v, oob=oob_v, dup=dup_v, bad_neg=neg_v, struct_ok=ok, set_ok=ok,
                    missing=0, forbidden=0, n_band=0, t=None)
    vals, idx = torch.topk(s, min(n + 1, P), largest=True, sorted=True)
    t, it = vals[n - 1], idx[n - 1]
    cnt = torch.zeros(P, dtype=torch.int32, device=s.device).index_add_(0, ids.clamp(0, P - 1),
                                                                         inr.to(torch.int32))
    mask = cnt > 0
    if exact or b is None:
        must = s > t
        allowed = s >= t
    else:
        bt = b[it]
        must = s > t + b + bt
        allowed = s >= t - b - bt
    gap = (t - vals[n]) if P > n else torch.zeros((), dtype=torch.float64, device=s.device)
    v = torch.stack([_d(n_inr), _d(oob), _d(dup), _d(bad_neg), _d((must & ~mask).sum()), _d((mask & ~allowed).sum()),
                     _d((allowed & ~must).sum()), _d((s == t).sum()), _d(t), _d(gap)]).tolist()
    n_v, oob_v, dup_v, neg_v, miss, forb, band, neq = (int(x) for x in v[:8])
    struct_ok = n_v == n and oob_v == 0 and dup_v == 0 and neg_v == 0
    out = dict(P=P, n=n, n_valid=n_v, oob=oob_v, dup=dup_v, bad_neg=neg_v, struct_ok=struct_ok,
               set_ok=struct_ok and miss == 0 and forb == 0, missing=miss, forbidden=forb, n_band=band,
               t=v[8], n_eq_t=neq)
    if P > n:
        out["gap_n_n1"] = v[9]
    return out


def check_kernel_scores(ids, ks, s, b, exact):
    ids = ids.to(torch.int64)
    P = int(s.shape[0])
    filled = ids >= 0
    unf_bad = (~filled & ~torch.isneginf(ks)).sum()
    n_unf = (~filled).sum()
    if P == 0:
        v = torch.stack([_d(n_unf), _d(unf_bad)]).tolist()
        return dict(n=0, nonfinite=0, exact_mismatch=0, ok=True, unfilled=int(v[0]), unfilled_all_neginf=int(v[1]) == 0)
    inr = filled & (ids < P)
    safe = ids.clamp(0, P - 1)
    k = ks.to(torch.float64)
    r = s[safe]
    err = (k - r).abs()
    nonfinite = (inr & ~torch.isfinite(k)).sum()
    mism = (inr & (k != r)).sum()
    if b is not None:
        bb = b[safe]
        ratio = torch.where(bb > 0, err / bb, torch.where(err > 0, torch.full_like(err, float("inf")),
                                                         torch.zeros_like(err)))
        ratio = torch.where(inr & torch.isfinite(k), ratio, torch.zeros_like(ratio))
        rmax = ratio.max()
    else:
        rmax = torch.zeros((), dtype=torch.float64, device=s.device)
    v = torch.stack([_d(inr.sum()), _d(nonfinite), _d(mism), _d(rmax), _d(n_unf), _d(unf_bad)]).tolist()
    n, nf, mm = int(v[0]), int(v[1]), int(v[2])
    out = dict(n=n, nonfinite=nf, exact_mismatch=mm)
    if n and b is not None:
        out["err_over_bound_max"] = v[3]
    out["ok"] = (nf == 0 and (mm == 0 if exact else out.get("err_over_bound_max", 0.0) <= 1.0)) if n else True
    out["unfilled"] = int(v[4])
    out["unfilled_all_neginf"] = int(v[5]) == 0
    return out


def check_col511(pre_ids, pre_ks, post_ids, s, b, exact):
    pre_ids = pre_ids.to(torch.int64)
    post_ids = post_ids.to(torch.int64)
    P = int(s.shape[0])
    same = torch.equal(torch.sort(pre_ids).values, torch.sort(post_ids).values)
    c_t = post_ids[KEEP]
    n = min(TOPK, P)
    out = dict(multiset_same=bool(same))
    if n < TOPK:
        c = int(c_t)
        out.update(col511_id=c, col511_kernel_min=True, col511_unfilled_ok=c == -1, fp64_min_ok=True)
    else:
        kmin = ((pre_ids == c_t) & (pre_ks == pre_ks.min())).any()
        sel = (post_ids >= 0) & (post_ids < P)
        safe = post_ids.clamp(0, P - 1)
        ss = torch.where(sel, s[safe], torch.full_like(s[safe], float("inf")))
        j = torch.argmin(ss)
        smin = ss[j]
        cs = c_t.clamp(0, P - 1)
        sc = s[cs]
        if exact or b is None:
            okm = sc == smin
        else:
            okm = sc <= smin + b[cs] + b[safe[j]]
        cval = (c_t >= 0) & (c_t < P) & sel.any()
        rank = (torch.where(sel, s[safe], torch.full_like(ss, float("-inf"))) > sc).sum()
        v = torch.stack([_d(c_t), _d(kmin), _d(okm & cval), _d(rank), _d(cval)]).tolist()
        out.update(col511_id=int(v[0]), col511_kernel_min=bool(v[1]), col511_unfilled_ok=True,
                   fp64_min_ok=bool(v[2]))
        if v[4]:
            out["col511_rank_in_selected"] = int(v[3])
    out["ok"] = bool(out["multiset_same"] and out["col511_kernel_min"] and out["col511_unfilled_ok"]
                     and out["fp64_min_ok"])
    return out
# tampered
