#!/usr/bin/env python3
"""K CPU tests, run inside each pinned image with no GPU (CUDA_VISIBLE_DEVICES='', TRITON_INTERPRET=1,
--network none). usage: test_cpu.py <b1|v070> <out.json>

  T1 reference: fp64 reference vs the image's own b12x torch reference (packing + eligibility), vs exact
     rationals (fractions.Fraction), fp32 bound validity under three summation orders, exactness of
     the exact families under those orders (incl. one full 330k-equivalent row).
  T2 family construction: plateau straddles rank 512, zero has < 512 positive, ulp neighbours are
     exactly 1 ulp apart at ranks 511/512/513, edge hot positions sit at the designed ranks, poison
     only beyond the live count, bank wrap derivation creates no exact duplicates.
  T3 checker mutation tests (each criterion must fire on its own fault, ties must be accepted).
  T3b the sync-light checkers used on the GPU agree with the straightforward reference checkers.
  T4 consumer: the image's expand_pools_and_append_tail (Triton, interpreted) == the image's torch
     composition == kref.expand_ref; B1's _move_lowest_selected_pool_last satisfies C3; 0.7.0 lacks it.
  T5 DCP harness: get_dcp_local_seq_lens == kplan.local_pools; the image's Triton convert / unpack /
     gather kernels (interpreted) and the real _merge_b12x_dcp_topk with the two-rank stub.
  T6 plan constants vs the image's b12x functions (fold plan, fused routing, batch slack, constants).
  T7 packing round trip and poison placement.
  T8 end-to-end harness (run_k.run_batch) with a fake kernel built on the image's b12x torch reference
     (reduced page-table width): the honest fake must PASS every family; four faulty fakes per image must
     be caught by the intended criterion.
  T9 run_k.run_cells on CPU: verdicts and output files; the time guard leaves NOT-RUN cells and still
     writes the summary.
Triton interpreter caveat: kernels execute through numpy on the host; integer index arithmetic and float
copies (the only operations in the interpreted kernels here) are exact, but the interpreter does not
model GPU memory ordering, races or out-of-bounds faults, so T4/T5 validate semantics, not the compiled
kernels. The CuTe DSL selection kernels cannot run on CPU at all; they run only in the GPU stage.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import traceback
from fractions import Fraction

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import kinputs  # noqa: E402
import kplan  # noqa: E402
import kref  # noqa: E402

ENGINE = sys.argv[1]
OUT = sys.argv[2]
DEV = torch.device("cpu")
EXACT = kplan.EXACT_FAMILIES
RES = dict(engine=ENGINE, tests=[], started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
_KD = os.path.dirname(os.path.abspath(__file__))
RES["code_sha256"] = {f: __import__("hashlib").sha256(open(os.path.join(_KD, f), "rb").read()).hexdigest()
                      for f in sorted(os.listdir(_KD)) if f.endswith((".py", ".sh", ".json"))}
torch.manual_seed(0)


def record(name, ok, **info):
    RES["tests"].append(dict(name=name, ok=bool(ok), **info))
    print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(info, default=str)[:300]}", flush=True)


def run(name, fn):
    t = time.time()
    try:
        fn()
    except Exception:
        record(name + " (exception)", False, tb=traceback.format_exc()[-2000:])
    RES.setdefault("secs", {})[name] = round(time.time() - t, 1)


ENG = kplan.ENGINES[ENGINE]
BANKDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bank")
BANKS = {f: kinputs.Bank(os.path.join(BANKDIR, f"bank_L{f.split('_L')[1]}.pt"), DEV)
         for f in kplan.FAMILIES if f.startswith("real_")}


def fp32_orders(q8, w, K8, S):
    """Three fp32 evaluation orders of the score (R,P): einsum; per-dim forward accumulation;
    per-dim reverse accumulation with the head sum reversed."""
    q = q8.view(torch.float8_e4m3fn).float()
    k = K8.view(torch.float8_e4m3fn).float()
    out = []
    d1 = torch.einsum("rhd,pd->rhp", q, k)
    out.append((torch.relu(d1) * w[:, :, None]).sum(1) * S[None, :])
    acc = torch.zeros(q.shape[0], q.shape[1], k.shape[0])
    for d in range(q.shape[2]):
        acc = acc + q[:, :, d:d + 1] * k[None, None, :, d]
    t = torch.relu(acc) * w[:, :, None]
    hs = torch.zeros(q.shape[0], k.shape[0])
    for h in range(q.shape[1]):
        hs = hs + t[:, h]
    out.append(hs * S[None, :])
    acc = torch.zeros(q.shape[0], q.shape[1], k.shape[0])
    for d in reversed(range(q.shape[2])):
        acc = acc + q[:, :, d:d + 1] * k[None, None, :, d]
    t = torch.relu(acc) * w[:, :, None]
    hs = torch.zeros(q.shape[0], k.shape[0])
    for h in reversed(range(q.shape[1])):
        hs = hs + t[:, h]
    out.append(hs * S[None, :])
    return out


def exact_score(q8r, wr, krow, s):
    q = q8r.view(torch.float8_e4m3fn).double().tolist()
    k = krow.view(torch.float8_e4m3fn).double().tolist()
    tot = Fraction(0)
    for h in range(len(q)):
        if float(wr[h]) == 0.0:
            continue
        d = sum((Fraction(q[h][i]) * Fraction(k[i]) for i in range(len(k)) if q[h][i] and k[i]), Fraction(0))
        if d > 0:
            tot += Fraction(float(wr[h])) * d
    return tot * Fraction(float(s))


def reqs_for(fam, Ls, b=1024, route="tiled", rows=8, ctas=23, seed=5):
    return kinputs.build_requests(fam, Ls, ENG["K"], b, route, rows, ctas, ENG["dcp"], seed, DEV, BANKS)


# ------------------------------------------------------------------------------------------ T1
def t1_reference():
    from b12x.attention.dsa_indexer.paged import paged_index_logits_reference
    for fam in ("real_L3", "real_L43", "gauss", "plateau", "ulp"):
        reqs = reqs_for(fam, [4 * 700 + 1, 4 * 1500 + 2])
        cache, bt = kinputs.pack([(r["K8"], r["S"], r["poison_k"], r["poison_s"]) for r in reqs], 40, DEV)
        worst, inf_ok = 0.0, True
        for i, r in enumerate(reqs):
            R = r["q8"].shape[0]
            P = r["P"]
            lg = paged_index_logits_reference(
                q_fp8=r["q8"].view(torch.float8_e4m3fn), weights=r["w"], index_k_cache=cache.reshape(-1, kinputs.PB),
                real_page_table=bt[i:i + 1].expand(R, -1).contiguous(), query_row_to_batch=torch.arange(R, dtype=torch.int32),
                seqlens_per_query=torch.full((R,), P, dtype=torch.int32))
            s, b = kref.ref_scores(r["q8"], r["w"], r["K8"], r["S"])
            e = (lg[:, :P].double() - s).abs()
            ratio = torch.where(b > 0, e / b, torch.where(e > 0, torch.full_like(e, float("inf")), torch.zeros_like(e)))
            worst = max(worst, float(ratio.max()))
            inf_ok &= bool(torch.isneginf(lg[:, P:]).all())
        record(f"T1a ref vs b12x torch reference [{fam}]", worst <= 1.0 and inf_ok, worst_err_over_bound=worst,
               beyond_live_all_neginf=inf_ok)
    # exact rationals: fp64 reference error and ordering
    for fam in ("real_L3", "gauss", "plateau", "zero", "ulp", "edge"):
        r = reqs_for(fam, [4 * 900 + 3], seed=11)[0]
        s, b = kref.ref_scores(r["q8"][:1], r["w"][:1], r["K8"], r["S"])
        s = s[0]
        order = torch.argsort(s, descending=True)
        pick = order[:24].tolist() + order[500:530].tolist() + random.Random(3).sample(range(r["P"]), 16)
        maxrel, ord_ok = 0.0, True
        ex = {}
        for p in pick:
            ex[p] = exact_score(r["q8"][0], r["w"][0], r["K8"][p], r["S"][p])
            d = abs(Fraction(float(s[p])) - ex[p])
            if fam in EXACT:
                ord_ok &= d == 0
            else:
                lim = Fraction(float(b[0, p])) * Fraction(1, 2 ** 29)
                maxrel = max(maxrel, float(d / lim) if lim else (0.0 if d == 0 else math.inf))
        sp = sorted(pick, key=lambda p: (-ex[p], p))
        sq = sorted(pick, key=lambda p: (-float(s[p]), p))
        ord_ok &= [ex[p] for p in sp] == [Fraction(float(s[p])) for p in sq] if fam in EXACT else \
            all(ex[sp[i]] >= ex[sp[i + 1]] for i in range(len(sp) - 1))
        record(f"T1b fp64 ref vs exact rationals [{fam}]", ord_ok and maxrel <= 1.0,
               exact_equal=(fam in EXACT), max_err_over_fp64_bound=maxrel, n=len(pick))
    # fp32 bound validity and exactness of the exact families under three orders
    for fam in kplan.FAMILIES:
        r = reqs_for(fam, [4 * 2500 + 1], seed=13)[0]
        s, b = kref.ref_scores(r["q8"], r["w"], r["K8"], r["S"])
        worst, exact_ok = 0.0, True
        for s32 in fp32_orders(r["q8"], r["w"], r["K8"], r["S"]):
            e = (s32.double() - s).abs()
            worst = max(worst, float(torch.where(b > 0, e / b, torch.zeros_like(e)).max()))
            if fam in EXACT:
                exact_ok &= bool((s32.double() == s).all())
        record(f"T1c fp32 orders within bound{' and exact' if fam in EXACT else ''} [{fam}]",
               worst <= 1.0 and exact_ok, worst_err_over_bound=worst, exact=exact_ok if fam in EXACT else None)
    # one full-length exact row (330k equivalent) in fp32 einsum order
    for fam in EXACT:
        r = reqs_for(fam, [kplan.L_MAX], b=81920, rows=32, seed=17)[0]
        q = r["q8"][-1:]
        w = r["w"][-1:]
        s, _ = kref.ref_scores(q, w, r["K8"], r["S"])
        qf = q.view(torch.float8_e4m3fn).float()[:, 0, :8]
        kf = r["K8"][:, :8].view(torch.float8_e4m3fn).float()
        d32 = torch.relu(kf @ qf[0]) * r["S"]
        record(f"T1d exact family exact at 330k-equivalent length [{fam}]", bool((d32.double() == s[0]).all()),
               P=r["P"], distinct=int(torch.unique(s[0]).numel()))


# ------------------------------------------------------------------------------------------ T2
def t2_families():
    b = 16384 if ENG["dcp"] == 1 else 16384
    Ls = [4 * (b * ENG["dcp"]) + 2, 4 * (b * ENG["dcp"] + 1) + 3]
    for fam in EXACT:
        reqs = reqs_for(fam, Ls, b=b, route="tiled", rows=32, seed=19)
        info = []
        ok = True
        for r in reqs:
            s, _ = kref.ref_scores(r["q8"][-1:], r["w"][-1:], r["K8"], r["S"])
            s = s[0]
            srt = torch.sort(s, descending=True).values
            t = srt[511]
            if fam == "plateau":
                good = int((s == t).sum()) >= 2 and srt[0] > t and srt[540] < t
            elif fam == "zero":
                good = int((s > 0).sum()) < 512 and float(t) == 0.0
            elif fam == "ulp":
                u = 2.0 ** -18
                good = float(srt[510] - srt[511]) == u and float(srt[511] - srt[512]) == u
            else:
                hot = kinputs.hot_global(r["P"], b, "tiled", 32, 1, ENG["dcp"])
                rk = {int(i): k for k, i in enumerate(torch.argsort(s, descending=True, stable=True)[:520])}
                at = sorted(rk[h] for h in hot if h in rk)
                good = any(505 <= x <= 519 for x in at) and any(x < 10 for x in at) and 0 in at
            ok &= bool(good)
            info.append(dict(P=r["P"], t=float(t), eq_t=int((s == t).sum())))
        record(f"T2 family property [{fam}]", ok, rows=info)
    # poison never inside the live range, and it outranks the live pools for exact families
    r = reqs_for("plateau", [4 * 3000 + 1])[0]
    ps, _ = kref.ref_scores(r["q8"][:1], r["w"][:1], r["poison_k"][None], torch.tensor([r["poison_s"]]))
    s, _ = kref.ref_scores(r["q8"][:1], r["w"][:1], r["K8"], r["S"])
    record("T2 poison outranks live pools (exact)", bool(ps[0, 0] > s.max()), poison=float(ps[0, 0]), live_max=float(s.max()))
    # bank wrap derivation: no exact duplicate pool between a wrapped copy and its source
    bk = BANKS["real_L3"]
    g = kinputs._gen(1, "cpu")
    K, S, wr = bk.pools(bk.n + 5000, g)
    a, bb = K[:5000], K[bk.n:bk.n + 5000]
    record("T2 bank wrap derivation creates no duplicates", wr == 1 and int((a == bb).all(1).sum()) == 0, wraps=wr)


# ------------------------------------------------------------------------------------------ T3
def t3_checker():
    g = torch.Generator().manual_seed(23)
    P = 4000
    s = torch.randn(P, generator=g, dtype=torch.float64)
    b = torch.full((P,), 1e-9, dtype=torch.float64)
    order = torch.argsort(s, descending=True)
    perfect = order[:512].clone()
    cases = []

    def chk(ids, exact=True, bnd=None):
        return kref.check_set(ids, s, bnd, exact, 512)
    r = chk(perfect)
    cases.append(("perfect", r["struct_ok"] and r["set_ok"]))
    r = chk(perfect[torch.randperm(512, generator=g)])
    cases.append(("permuted", r["struct_ok"] and r["set_ok"]))
    x = perfect.clone(); x[7] = order[600]
    r = chk(x)
    cases.append(("swap-out-lower -> C2", r["struct_ok"] and not r["set_ok"] and r["missing"] == 1 and r["forbidden"] == 1))
    x = perfect.clone(); x[7] = x[8]
    r = chk(x)
    cases.append(("duplicate -> C1", not r["struct_ok"] and r["dup"] == 1))
    x = perfect.clone(); x[3] = P
    r = chk(x)
    cases.append(("ineligible id P -> C1", not r["struct_ok"] and r["oob"] == 1))
    x = perfect.clone(); x[3] = -2
    r = chk(x)
    cases.append(("id -2 -> C1", not r["struct_ok"] and r["bad_neg"] == 1))
    x = perfect.clone(); x[3] = -1
    r = chk(x)
    cases.append(("missing slot -> C1", not r["struct_ok"]))
    # tie plateau: values equal at the threshold, any subset accepted, one below rejected
    s2 = s.clone()
    plat = order[500:540]
    s2[plat] = float(s[order[520]])
    s2[order[540:]] = torch.clamp(s2[order[540:]], max=float(s[order[520]]) - 1.0)
    o2 = torch.argsort(s2, descending=True, stable=True)
    ok_all = True
    for _ in range(50):
        pl = plat[torch.randperm(40, generator=g)[:12]]
        ids = torch.cat([o2[:500], pl])
        ok_all &= kref.check_set(ids, s2, None, True, 512)["set_ok"]
    ids = torch.cat([o2[:500], plat[:11], o2[600:601]])
    rej = not kref.check_set(ids, s2, None, True, 512)["set_ok"]
    cases.append(("plateau any subset accepted / below rejected", ok_all and rej))
    # signed zero ties
    s3 = torch.zeros(1000, dtype=torch.float64)
    s3[:300] = torch.arange(300, 0, -1, dtype=torch.float64)
    s3[300:650] = -0.0
    ids = torch.cat([torch.arange(300), torch.arange(700, 912)])
    cases.append(("+0/-0 tie accepted", kref.check_set(ids, s3, None, True, 512)["set_ok"]))
    # tolerance band
    s4 = s.clone()
    bb = torch.full((P,), 1e-3, dtype=torch.float64)
    t = float(s4[order[511]])
    s4[order[512]] = t - 1e-4                          # inside the band below t
    x = perfect.clone(); x[511] = order[512]
    cases.append(("tol: swap inside band accepted", kref.check_set(x, s4, bb, False, 512)["set_ok"]))
    x = perfect.clone(); x[0] = order[900]
    cases.append(("tol: swap outside band rejected", not kref.check_set(x, s4, bb, False, 512)["set_ok"]))
    # kernel scores
    ks = s[perfect].float()
    cs = kref.check_kernel_scores(perfect, ks, s.float().double(), b, True)
    cases.append(("C2b exact equal", cs["ok"]))
    ks2 = ks.clone(); ks2[5] = torch.nextafter(ks2[5], torch.tensor(1e9))
    cs = kref.check_kernel_scores(perfect, ks2, s.float().double(), b, True)
    cases.append(("C2b 1-ulp off -> exact mismatch", not cs["ok"] and cs["exact_mismatch"] == 1))
    bb2 = torch.full((P,), 1e-3, dtype=torch.float64)
    ks3 = (s[perfect] + 0.5e-3).float()
    cases.append(("C2b within bound ok", kref.check_kernel_scores(perfect, ks3, s, bb2, False)["ok"]))
    ks4 = (s[perfect] + 3e-3).float()
    cases.append(("C2b 3x bound rejected", not kref.check_kernel_scores(perfect, ks4, s, bb2, False)["ok"]))
    # col511
    pre = perfect[torch.randperm(512, generator=g)].clone()
    pks = s[pre].float()
    post = pre.clone()
    j = int(torch.argmin(pks)); post[[j, 511]] = post[[511, j]]
    cases.append(("C3 min last accepted", kref.check_col511(pre, pks, post, s, None, True)["ok"]))
    post2 = pre.clone()
    if int(torch.argmin(pks)) == 511:
        post2[[0, 511]] = post2[[511, 0]]
    cases.append(("C3 non-min last rejected", not kref.check_col511(pre, pks, post2, s, None, True)["ok"]))
    s5 = s[:300]
    pre5 = torch.cat([torch.argsort(s5, descending=True), torch.full((212,), -1)])[torch.randperm(512, generator=g)]
    pk5 = torch.where(pre5 >= 0, s5[pre5.clamp(min=0)].float(), torch.tensor(float("-inf")))
    post5 = pre5.clone(); j = int(torch.argmin(pk5)); post5[[j, 511]] = post5[[511, j]]
    cases.append(("C3 P<512: -1 last accepted", kref.check_col511(pre5, pk5, post5, s5, None, True)["ok"]))
    post6 = pre5.clone()
    jv = int((post6 >= 0).nonzero()[0]); post6[[jv, 511]] = post6[[511, jv]]
    cases.append(("C3 P<512: valid last rejected", not kref.check_col511(pre5, pk5, post6, s5, None, True)["ok"]))
    # expanded
    ids511 = torch.stack([torch.randperm(700, generator=g)[:511] - 1 for _ in range(6)])
    L = torch.tensor([1600, 1601, 1602, 1603, 400, 9])
    E = kref.expand_ref(ids511, L)
    cases.append(("C4 reference self-consistent", kref.check_expanded(E, ids511, L)["ok"]))
    E2 = E.clone(); E2[0, 5] = 3
    cases.append(("C4 corrupted entry rejected", not kref.check_expanded(E2, ids511, L)["ok"]))
    E3 = E.clone(); E3[1, -3] = -1
    cases.append(("C4 missing tail rejected", not kref.check_expanded(E3, ids511, L)["ok"]))
    for name, ok in cases:
        record(f"T3 {name}", ok)


def t3b_fast_equals_ref():
    """The sync-light checkers (used on GPU) agree with the straightforward *_ref versions."""
    g = torch.Generator().manual_seed(37)
    mism = []
    n_cases = 0
    for P in (0, 1, 100, 511, 512, 513, 4000):
        for trial in range(12):
            s = torch.randn(P, generator=g, dtype=torch.float64)
            if trial % 3 == 0 and P:
                s = torch.round(s * 2) / 2                                  # heavy exact ties
            if trial % 4 == 1 and P:
                s[torch.randperm(P, generator=g)[: P // 2]] = 0.0           # zero plateau
            b = torch.rand(P, generator=g, dtype=torch.float64) * 1e-3
            exact = trial % 2 == 0
            n = min(512, P)
            order = torch.argsort(s, descending=True, stable=True)
            ids = torch.cat([order[:n], torch.full((512 - n,), -1, dtype=torch.int64)])
            ids = ids[torch.randperm(512, generator=g)]
            mut = trial % 6
            if mut == 1 and P:
                ids[3] = P + 2                                              # ineligible
            elif mut == 2 and n > 2:
                k = int((ids >= 0).nonzero()[0]); k2 = int((ids >= 0).nonzero()[1]); ids[k] = ids[k2]  # duplicate
            elif mut == 3:
                ids[5] = -2
            elif mut == 4 and P > n:
                k = int((ids >= 0).nonzero()[0]) if n else 0
                ids[k] = order[min(P - 1, n + 3)]                           # swap in a lower pool
            ks = torch.where(ids >= 0, s[ids.clamp(0, max(P - 1, 0))].float() if P else torch.zeros(512),
                             torch.tensor(float("-inf")))
            if trial % 5 == 2:
                ks[7] = float("nan")
            if trial % 7 == 3 and P:
                ks[(ids >= 0)] = (ks[(ids >= 0)].double() + 2e-4).float()
            for name, f, r, args in (
                    ("set", kref.check_set, kref.check_set_ref, (ids, s, b, exact, 512)),
                    ("set511", kref.check_set, kref.check_set_ref, (ids[:511], s, None, True, 511)),
                    ("scores", kref.check_kernel_scores, kref.check_kernel_scores_ref, (ids, ks, s, b, exact))):
                a1, a2 = f(*args), r(*args)
                if name == "scores" and a2.get("nonfinite"):
                    a1.pop("err_over_bound_max", None); a2.pop("err_over_bound_max", None)
                n_cases += 1
                if a1 != a2:
                    mism.append((name, P, trial, {k: (a1.get(k), a2.get(k)) for k in set(a1) | set(a2) if a1.get(k) != a2.get(k)}))
            post = ids.clone()
            j = int(torch.argmin(torch.nan_to_num(ks, nan=-float("inf")))) if trial % 2 else int(torch.randint(0, 512, (1,), generator=g))
            post[[j, 511]] = post[[511, j]]
            a1 = kref.check_col511(ids, ks, post, s, b, exact)
            a2 = kref.check_col511_ref(ids, ks, post, s, b, exact)
            n_cases += 1
            if a1 != a2:
                mism.append(("col511", P, trial, {k: (a1.get(k), a2.get(k)) for k in set(a1) | set(a2) if a1.get(k) != a2.get(k)}))
    record("T3b sync-light checkers == reference checkers", not mism, cases=n_cases, mismatches=mism[:5])


# ------------------------------------------------------------------------------------------ T4
def t4_consumer():
    from kdrive import Driver
    drv = Driver(ENG, DEV)
    KC = drv.KC
    g = torch.Generator().manual_seed(29)
    allok = True
    for rows in (1, 5, 24, 48):
        L = torch.randint(1, 4 * 3000, (rows,), generator=g)
        L[0] = 4 * 600
        full = torch.randint(-1, 3500, (rows, 512), generator=g).to(torch.int32)
        E = drv.expand(full, L)
        ref = kref.expand_ref(full[:, :511], L)
        pool_len = (L // 4).to(torch.int32)
        ids64 = full.to(torch.int64)[:, :511]
        valid = (ids64 >= 0) & (ids64 < pool_len[:, None])
        comp = KC.append_tail_to_topk(KC.expand_pools_to_tokens(torch.where(valid, ids64, -1), valid, 511 * 4, 4),
                                      L.to(torch.int32), pool_len, 4)
        allok &= bool(torch.equal(E, ref)) and bool(torch.equal(comp, ref))
    record("T4 expand kernel (interpreted) == torch composition == expand_ref", allok)
    if ENG["col511_fix"]:
        ok = True
        for trial in range(40):
            P = 3000 if trial % 4 else 300
            s = torch.randn(P, generator=g, dtype=torch.float64)
            if trial % 5 == 0:
                s = torch.round(s * 4) / 4                  # many exact ties
            order = torch.argsort(s, descending=True)
            n = min(512, P)
            ids = torch.cat([order[:n], torch.full((512 - n,), -1)])[torch.randperm(512, generator=g)].to(torch.int32)
            ks = torch.where(ids >= 0, s[ids.long().clamp(min=0)].float(), torch.tensor(float("-inf")))
            pre, pks = ids.clone(), ks.clone()
            ids2, ks2 = ids[None].clone(), ks[None].clone()
            drv.col511_fix(ids2, ks2)
            r = kref.check_col511(pre, pks, ids2[0], s, None, True)
            f = kref.check_set(ids2[0, :511], s, None, True, 511)
            ok &= r["ok"] and f["set_ok"]
        record("T4 B1 _move_lowest_selected_pool_last satisfies C3 and first-511", ok)
    else:
        record("T4 0.7.0 kpool has no column-511 fix (policy pin)", not drv.has_fix)


# ------------------------------------------------------------------------------------------ T5
def stub_run_row_topk(row_logits, lengths, topk, output_values, output_indices, **kw):
    x = row_logits.clone()
    ar = torch.arange(x.shape[1])[None, :]
    x = torch.where(ar < lengths[:, None].long(), x, torch.tensor(float("-inf")))
    v, i = torch.topk(x, topk, dim=1)
    output_values.copy_(v)
    output_indices.copy_(i.to(torch.int32))


def t5_dcp():
    from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
    Ps = torch.tensor([0, 1, 2, 3, 511, 512, 513, 32767, 32768, 32769, 65537, 82500])
    ok = True
    for r in range(2):
        got = get_dcp_local_seq_lens(Ps, 2, r, 1).tolist()
        ok &= got == [kplan.local_pools(int(p), 2, r) for p in Ps]
    record("T5 get_dcp_local_seq_lens == kplan.local_pools", ok)
    from vllm.v1.attention.backends.mla.b12x_dcp_topk import (triton_convert_dcp_local_topk_to_global,
                                                              triton_gather_topk_ids_by_position)
    g = torch.Generator().manual_seed(31)
    ok = True
    for r in range(2):
        loc = torch.randint(-1, 4000, (6, 512), generator=g).to(torch.int32)
        sc = torch.randn((6, 512), generator=g)
        i2, s2 = loc.clone(), sc.clone()
        triton_convert_dcp_local_topk_to_global(i2, s2, dcp_world_size=2, dcp_rank=r, cp_kv_cache_interleave_size=1)
        exp = torch.where(loc >= 0, loc * 2 + r, -1)
        ok &= bool(torch.equal(i2, exp)) and bool(torch.equal(torch.isneginf(s2), loc < 0)) and \
            bool(torch.equal(s2[loc >= 0], sc[loc >= 0]))
    record("T5 convert kernel (interpreted): local i -> 2i+r, -1 -> -1 with -inf", ok)
    import vllm.model_executor.layers.sparse_attn_indexer as SAI
    rows = 6
    gathered = torch.randint(-5, 5000, (2 * rows, 2, 512), generator=g).to(torch.int32)
    ci = torch.empty((rows, 1024), dtype=torch.int32)
    cs = torch.empty((rows, 1024), dtype=torch.int32)
    SAI._unpack_b12x_dcp_gathered_candidates(gathered, ci, cs, dcp_world_size=2, topk_tokens=512)
    ci2 = torch.empty_like(ci); cs2 = torch.empty_like(cs)
    SAI._b12x_dcp_unpack_gathered_candidates_kernel[(rows, 1)](gathered, ci2, cs2, rows, 512, 1024, BLOCK_K=1024)
    record("T5 unpack: Triton kernel (interpreted) == torch fallback", bool(torch.equal(ci, ci2) and torch.equal(cs, cs2)))
    pos = torch.randint(0, 1024, (rows, 512), generator=g).to(torch.int32)
    out = torch.empty((rows, 512), dtype=torch.int32)
    triton_gather_topk_ids_by_position(ci, pos, out)
    record("T5 gather-by-position kernel (interpreted)", bool(torch.equal(out, torch.gather(ci, 1, pos.long()))))
    # real merge with the two-rank stub; run_row_topk (CuTe) replaced by a torch top-k on CPU only
    import b12x.attention.dsa_indexer.tiled_topk as TT
    saved = TT.run_row_topk
    TT.run_row_topk = stub_run_row_topk
    try:
        from kdrive import Driver
        drv = Driver(ENG, DEV)
        P = 5000
        s = torch.randn((rows, P), generator=g, dtype=torch.float64)
        ids_l, sc_l = [], []
        for r in range(2):
            sl = s[:, r::2]
            v, i = torch.topk(sl, 512, dim=1)
            perm = torch.randperm(512, generator=g)
            ids_l.append(i[:, perm].to(torch.int32).contiguous())
            sc_l.append(v[:, perm].float().contiguous())
        persp, ag_ok, cand = drv.merge_dcp2(ids_l, sc_l, 0xFF)
        ok = ag_ok and torch.equal(persp[0][0], persp[1][0])
        for row in range(rows):
            ok &= kref.check_set(persp[0][0][row], s[row].float().double(), None, True, 512)["set_ok"]
        record("T5 real _merge_b12x_dcp_topk via stub: global top-512, perspectives equal", bool(ok), allgather_ok=ag_ok)
    finally:
        TT.run_row_topk = saved


# ------------------------------------------------------------------------------------------ T6
def t6_plan():
    import b12x.attention.dsa_indexer.paged as PG
    import b12x.attention.dsa_indexer.fused_indexer as FI
    ok = True
    det = []
    for width in (4097, 2058, 2048):
        for rows in (4, 8, 16, 24, 32, 48):
            fp = PG._plan_two_level_fold(q_rows=rows, topk=512, page_size=64, page_table_width=width,
                                         supertile_pages=512, output_physical_slots=False)
            mine = kplan.fold_plan(width, rows)
            same = (list(map(list, fp.slices)) == mine["slices"]) and int(fp.total_slices) == mine["total_slices"]
            ok &= same
            det.append((width, rows, int(fp.total_slices), same))
    record("T6 fold plan == image _plan_two_level_fold", ok, detail=det[:6])
    route = [FI.resolve_fused_indexer_path(topk=512, num_rows=r, width=ENG["width_pages"], num_heads=32,
                                           compute_capability=(12, 0)) for r in (4, 6, 8, 12, 16, 24, 32, 48)]
    record("T6 fused routing rows<=16 on sm120", route == [True] * 5 + [False] * 3, route=route)
    slack = FI._resolve_fused_batch_slack(kv_layout=FI.KV_LAYOUT_PAGED, num_heads=32, topk=512)
    record("T6 fused carry slack (topk 512, 32 heads) == 448", slack == kplan.SLACK, slack=slack)
    consts = dict(slice=PG._TWO_LEVEL_SLICE_TOKENS == kplan.SLICE, max_slices=PG._TWO_LEVEL_MAX_SLICES == kplan.MAX_SLICES,
                  supertile=PG._PAGED_INDEX_SUPERTILE_K_DEFAULT == kplan.SUPERTILE, page=PG.PAGED_INDEX_PAGE_SIZE == kplan.PAGE,
                  fused_rows=FI._C4_B12X_FUSED_MAX_ROWS == kplan.FUSED_MAX_ROWS)
    record("T6 b12x constants == kplan", all(consts.values()), **consts)
    thr = {}
    for c in (11, 23, 47):
        try:
            if hasattr(FI, "resolve_fused_merge_threshold"):        # 0.9.1 / B1
                thr[c] = FI.resolve_fused_merge_threshold("auto", ctas_per_group=c, num_heads=32, topk=512)
            else:                                                    # 0.7.0
                thr[c] = FI._resolve_default_merge_threshold(ctas_per_group=c, num_heads=32, topk=512)
        except Exception as e:  # noqa
            thr[c] = repr(e)[:120]
    force_last = getattr(FI, "_FORCE_LAST_CTA", None)
    if ENG["dcp"] == 1:
        ok = all(thr[c] == 0 for c in thr)                           # cooperative merge at every length
    else:
        ok = all(thr[c] == force_last for c in (11, 23))              # serial last-CTA at the served ctas
    record("T6 fused cross-CTA merge arm (B1 cooperative; 0.7.0 serial at 11/23 CTAs)", ok, thresholds=thr,
           force_last_cta=force_last)


# ------------------------------------------------------------------------------------------ T7
def t7_pack():
    reqs = reqs_for("gauss", [4 * 100 + 1, 4 * 64, 4 * 1000 + 3])
    cache, bt = kinputs.pack([(r["K8"], r["S"], r["poison_k"], r["poison_s"]) for r in reqs], 40, DEV)
    ok = True
    for i, r in enumerate(reqs):
        K, S = kinputs.unpack_pools(cache, bt[i], r["P"])
        ok &= bool(torch.equal(K, r["K8"])) and bool(torch.equal(S, r["S"]))
        npg = -(-r["P"] // 64)
        flat = cache.reshape(-1, kinputs.PB)
        pages = flat[bt[i, :npg + 2].long()]
        Kall = pages[:, :64 * 128].reshape(-1, 128)
        ok &= bool((Kall[r["P"]:] == r["poison_k"]).all())
        ok &= bool((bt[i, npg + 2:] == 0).all())
    ok &= bool((cache.reshape(-1, kinputs.PB)[0, :64 * 128].reshape(64, 128) == reqs[0]["poison_k"]).all())
    record("T7 pack round trip; poison exactly beyond the live count; null page poisoned", ok)


# ------------------------------------------------------------------------------------------ T8
class FakeMixin:
    """Selection by the image's b12x torch reference (fp32) + torch.topk, output order shuffled
    (no B12x route orders its output). Faults: 'noelig' ignores the live length (poison reachable),
    'dropslice' never selects local pools >= drop_at, 'nofix' skips the column-511 fix, 'bf16' ranks
    and returns bf16-rounded scores, 'skew' perturbs one merge perspective."""
    fault = None
    drop_at = None

    def topk(self, q8, w, cache3d, bt_rows, seqlens_pool, pattern):
        from b12x.attention.dsa_indexer.paged import paged_index_logits_reference
        rows = q8.shape[0]
        sl = seqlens_pool.to(torch.int32)
        if self.fault == "noelig":
            sl = torch.where(sl > 0, torch.full_like(sl, bt_rows.shape[1] * 64), sl)
        lg = paged_index_logits_reference(q_fp8=q8.view(torch.float8_e4m3fn), weights=w, index_k_cache=cache3d.reshape(-1, kinputs.PB),
                                          real_page_table=bt_rows, query_row_to_batch=torch.arange(rows, dtype=torch.int32),
                                          seqlens_per_query=sl)
        if self.fault == "dropslice" and self.drop_at is not None:
            lg[:, self.drop_at:] = float("-inf")
        if self.fault == "bf16":
            lg = lg.to(torch.bfloat16).float()
        v, i = torch.topk(lg, 512, dim=1)
        i = torch.where(torch.isneginf(v), torch.full_like(i, -1), i).to(torch.int32)
        g = torch.Generator().manual_seed(int(pattern) + 7)
        perm = torch.stack([torch.randperm(512, generator=g) for _ in range(rows)])
        meta = dict(route=self.expect_route, ctas_per_group=self.expect_ctas, merge_threshold=None)
        return torch.gather(i, 1, perm), torch.gather(v, 1, perm), meta

    def col511_fix(self, ids, sc):
        if self.fault != "nofix":
            super().col511_fix(ids, sc)

    def merge_dcp2(self, ids_loc, sc_loc, pattern):
        res, ag, cand = super().merge_dcp2(ids_loc, sc_loc, pattern)
        if self.fault == "skew":
            res[1][0][0, 0] = res[1][0][0, 1]
        return res, ag, cand


def t8_end_to_end():
    import run_k
    from kdrive import Driver
    import b12x.attention.dsa_indexer.tiled_topk as TT
    saved = TT.run_row_topk
    TT.run_row_topk = stub_run_row_topk

    class Fake(FakeMixin, Driver):
        pass
    eng = dict(ENG)
    eng["width_pages"] = 160                          # reduced width: CPU reference cost
    rows = 8 if ENG["dcp"] == 1 else 16
    b = 1024
    tgt = kplan.targets(eng, b, "test")[2:8]
    batches, rpb, pad = kplan.make_batches(eng, rows, tgt)
    cell = dict(id=f"cpu/{ENGINE}/tiled{rows}/b{b}", route="tiled", rows=rows, boundary_local=b, kind="cpu",
                pad_rows=pad, batches=batches[:2], expect=dict(route="paged_tiled", ctas_per_group=1))
    try:
        def go(fault, fams, drop_at=None):
            d = Fake(ENG, DEV)
            d.fault, d.drop_at = fault, drop_at
            d.expect_route, d.expect_ctas = "paged_tiled", 1
            agg = run_k.Agg()
            fails, timing = [], dict(build=0.0, ref=0.0, kernel=0.0, check=0.0)
            sink = open(os.devnull, "w")
            for fam in fams:
                for bi, bt in enumerate(cell["batches"]):
                    run_k.run_batch(d, eng, cell, fam, fam in EXACT, bi, bt, BANKS, DEV, agg, sink, fails, timing)
            return run_k.verdict(agg, True), agg
        v, agg = go(None, kplan.FAMILIES)
        record("T8 honest fake kernel: every family PASS", v == "PASS", verdict=v,
               counts={k: x for k, x in agg.n.items() if x}, desc=agg.desc)
        v, agg = go("noelig", ("plateau", "real_L3"))
        key = "c1" if ENG["dcp"] == 1 else "rank_c1"
        record("T8 fault: eligibility ignored (poison reachable) -> FAIL via C1", v == "FAIL" and agg.n[key] > 0,
               verdict=v, counts={k: x for k, x in agg.n.items() if x})
        local_b = b
        v, agg = go("dropslice", ("edge",), drop_at=local_b - 1)
        key = "c2" if ENG["dcp"] == 1 else "rank_c2"
        record("T8 fault: newly live region dropped -> FAIL via C2", v == "FAIL" and agg.n[key] > 0, verdict=v,
               counts={k: x for k, x in agg.n.items() if x})
        v, agg = go("bf16", ("ulp", "gauss"))
        record("T8 fault: bf16-rounded ranking -> FAIL / FAIL-NUMERIC", v in ("FAIL", "FAIL-NUMERIC"), verdict=v,
               counts={k: x for k, x in agg.n.items() if x})
        if ENG["col511_fix"]:
            v, agg = go("nofix", ("real_L43", "plateau"))
            record("T8 fault: column-511 fix skipped -> FAIL via C3", v == "FAIL" and agg.n["c3"] > 0, verdict=v,
                   counts={k: x for k, x in agg.n.items() if x})
        else:
            v, agg = go("skew", ("gauss",))
            record("T8 fault: merge perspectives differ -> FAIL via persp", v == "FAIL" and agg.n["persp"] > 0, verdict=v,
                   counts={k: x for k, x in agg.n.items() if x})
    finally:
        TT.run_row_topk = saved


def t9_run_cells():
    """run_k.run_cells end to end on CPU (honest fake, reduced width): summary verdicts, results and
    failures files; and the time guard (max_seconds=0) leaving every cell NOT-RUN with files written."""
    import tempfile
    import run_k
    from kdrive import Driver
    import b12x.attention.dsa_indexer.tiled_topk as TT
    saved = TT.run_row_topk
    TT.run_row_topk = stub_run_row_topk

    class Fake(FakeMixin, Driver):
        pass
    try:
        eng = dict(ENG)
        eng["width_pages"] = 160
        rows = 8 if ENG["dcp"] == 1 else 16
        cells = []
        for b in (512, 1024):
            batches, rpb, pad = kplan.make_batches(eng, rows, kplan.targets(eng, b, "test")[3:6])
            cells.append(dict(id=f"cpu/{ENGINE}/tiled{rows}/b{b}", engine=ENGINE, route="tiled", rows=rows,
                              boundary_local=b, kind="cpu", pad_rows=pad, batches=batches[:1],
                              expect=dict(route="paged_tiled", ctas_per_group=1, fold=kplan.fold_plan(160, rows))))
        plan = dict(meta=dict(FAMILIES=["gauss", "edge"], EXACT_FAMILIES=list(EXACT)), cells=cells)
        d = Fake(ENG, DEV)
        d.fault, d.expect_route, d.expect_ctas = None, "paged_tiled", 1
        with tempfile.TemporaryDirectory() as td:
            summ = run_k.run_cells(d, plan, eng, ENGINE, cells, BANKS, DEV, td, time.time(), 1e9, 0,
                                   lambda m: None, dict(test=True))
            files = sorted(os.listdir(td))
            ok = [c["verdict"] for c in summ["cells"]] == ["PASS", "PASS"] and \
                files == sorted([f"failures-{ENGINE}.pt", f"results-{ENGINE}.jsonl", f"summary-{ENGINE}.json"])
            nres = sum(1 for _ in open(os.path.join(td, f"results-{ENGINE}.jsonl")))
            record("T9 run_cells: honest fake PASS, files written", ok and nres == 2 * 2 * 3, verdicts=[c["verdict"] for c in summ["cells"]], error=str(summ.get("error", ""))[-600:],
                   results_lines=nres)
        with tempfile.TemporaryDirectory() as td:
            summ = run_k.run_cells(d, plan, eng, ENGINE, cells, BANKS, DEV, td, time.time(), 0.0, 0,
                                   lambda m: None, dict(test=True))
            ok = summ["stopped_by_time_guard"] and [c["verdict"] for c in summ["cells"]] == ["NOT-RUN", "NOT-RUN"] \
                and os.path.exists(os.path.join(td, f"summary-{ENGINE}.json"))
            record("T9 run_cells: time guard -> NOT-RUN, summary still written", ok, tally=summ["tally"])
    finally:
        TT.run_row_topk = saved


for name, fn in [("T1", t1_reference), ("T2", t2_families), ("T3", t3_checker), ("T3b", t3b_fast_equals_ref), ("T4", t4_consumer),
                 ("T5", t5_dcp), ("T6", t6_plan), ("T7", t7_pack), ("T8", t8_end_to_end),
                 ("T9", t9_run_cells)]:
    run(name, fn)
RES["n_pass"] = sum(1 for t in RES["tests"] if t["ok"])
RES["n_fail"] = sum(1 for t in RES["tests"] if not t["ok"])
json.dump(RES, open(OUT, "w"), indent=1, default=str)
print(f"{ENGINE}: {RES['n_pass']} pass, {RES['n_fail']} fail", flush=True)
sys.exit(0 if RES["n_fail"] == 0 else 1)
