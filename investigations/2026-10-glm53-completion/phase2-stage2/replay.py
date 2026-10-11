"""Phase 2 / Stage 2 replay: run captured real decode-step indexer inputs through a release's
top-k paths (stage-1 driver, $S/fixtureA/fixture.py) and score column 511 against an fp64
reference computed from the same bytes.

Runs inside a release image (v091 or v070) on one GPU, no network.
  python3 replay.py --version v091|v070 --probe DIR [DIR..] --out OUT.jsonl [--repeats 3]
  python3 replay.py --version v091|v070 --selftest --out OUT.jsonl
Capture format: written by the col511-probe hook (sparse_attn_indexer_kpool.py).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/fa")
import fixture as FX  # noqa: E402  (stage-1 driver: b12x_paged_topk, dcp2_merge)

TOPK, KEEP, HEADS, HD, PAGE = 512, 511, 32, 128, 64
PB = PAGE * (HD + 4)  # 8448 bytes per page: 64x128 e4m3, then 64 fp32 scales
EPS32 = 2.0 ** -24
SEL = {}


def page_split(pages):
    """(U,8448) uint8 -> k8 (U,64,128) uint8, ks (U,64) fp32."""
    U = pages.shape[0]
    k8 = pages[:, : PAGE * HD].reshape(U, PAGE, HD)
    ks = pages[:, PAGE * HD:].contiguous().view(torch.float32).reshape(U, PAGE)
    return k8, ks


def load_capture(path, dev):
    r = torch.load(path, weights_only=False, map_location="cpu")
    R = int(r["rows"])
    q = r["q"].view(torch.float8_e4m3fn).reshape(R, HEADS, HD).to(dev).contiguous()
    w = r["w"].float().reshape(R, HEADS).to(dev).contiguous()
    sl = r["seq_lens"].to(torch.int64)
    bt = r["bt"].to(torch.int64)
    pid = r["page_ids"].to(torch.int64)
    pages = r["pages"].to(dev)
    # remap physical page ids -> 1..U (0 = null page, as vLLM pads)
    loc = torch.searchsorted(pid, bt.clamp(min=0)) + 1
    npages = (sl + PAGE - 1) // PAGE
    need = torch.arange(bt.shape[1]).unsqueeze(0) < npages.unsqueeze(1)
    ok = need & (loc - 1 < pid.numel())
    ok &= torch.where(ok, pid[(loc - 1).clamp(max=pid.numel() - 1)] == bt, torch.zeros_like(ok))
    assert bool((ok == need).all()), "page remap failed"
    bt_loc = torch.where(need, loc, torch.zeros_like(loc))
    return r, R, q, w, sl, bt_loc, pages


def build_v091_cache(R, sl, bt_loc, pages, width, dev):
    U = pages.shape[0]
    cache = torch.zeros((U + 1, PB), dtype=torch.uint8, device=dev)
    cache[1:] = pages
    bt = torch.zeros((R, width), dtype=torch.int32)
    bt[:, : bt_loc.shape[1]] = bt_loc.to(torch.int32)
    return cache, bt.to(dev).contiguous(), sl.to(torch.int32).to(dev)


def row_pools(i, sl, bt_loc, k8, ks):
    n = int(sl[i])
    np_ = (n + PAGE - 1) // PAGE
    pg = bt_loc[i, :np_] - 1
    K = k8[pg.to(k8.device)].reshape(-1, HD)[:n]
    S = ks[pg.to(ks.device)].reshape(-1)[:n]
    return K, S


def reference(q, w, sl, bt_loc, pages):
    """Per row: fp64 score, fp32 bound, sorted scores and boundary stats."""
    k8, ks = page_split(pages)
    refs = []
    for i in range(q.shape[0]):
        Kb, sc = row_pools(i, sl, bt_loc, k8, ks)
        K = Kb.view(torch.float8_e4m3fn).double()
        qi = q[i].double()
        wi = w[i].double()
        sc = sc.double()
        dots = qi @ K.T
        s = (torch.relu(dots) * wi[:, None]).sum(0) * sc
        b = ((qi.abs() @ K.abs().T) * wi.abs()[:, None]).sum(0) * sc * (162 * EPS32)
        # Pools whose stored bytes decode to NaN (e4m3 0x7F/0xFF) are never selected by any kernel
        # path; the reference treats them as excluded (-inf) and counts them per row.
        nan = torch.isnan(s)
        n_nan = int(nan.sum())
        s = torch.where(nan, torch.full_like(s, float("-inf")), s)
        b = torch.where(torch.isnan(b), torch.zeros_like(b), b)
        srt = torch.sort(s, descending=True, stable=True)
        n = s.numel()
        k512 = min(TOPK, n) - 1
        t512 = srt.values[k512]
        b512 = b[srt.indices[k512]]
        refs.append(dict(
            S=s, B=b, n=n, t512=t512, b512=b512,
            n_above=int((s > t512).sum()), n_above_tol=int((s > t512 + b + b512).sum()),
            stats=dict(
                n=n, n_nan=n_nan, t512=float(t512),
                gap_512_513=float(srt.values[k512] - srt.values[k512 + 1]) if n > TOPK else None,
                gap_511_512=float(srt.values[k512 - 1] - srt.values[k512]),
                n_eq_t512=int((s == t512).sum()), n_zero=int((s == 0).sum()),
                n_tol_t512=int(((s - t512).abs() <= b + b512).sum()),
                n_eq_any_top520=int(520 - torch.unique(srt.values[:520]).numel()) if n >= 520 else None,
                zeros_in_top512=int((srt.values[:TOPK] == 0).sum()),
                s_max=float(srt.values[0]), s_min=float(srt.values[-1]))))
    return refs


def row_metrics(ref, ids):
    """ids: (512,) int64 on device. Column-511 rank and set checks (fp64 reference)."""
    S, B, n = ref["S"], ref["B"], ref["n"]
    valid = (ids >= 0) & (ids < n)
    safe = ids.clamp(0, n - 1)
    vs = torch.where(valid, S[safe], torch.full_like(S[safe], float("nan")))
    vb = B[safe]
    nvalid = int(valid.sum())
    uniq = torch.unique(ids[valid]).numel()
    t, bt_ = ref["t512"], ref["b512"]
    sel_above = int((valid & (vs > t)).sum())
    sel_above_tol = int((valid & (S[safe] > t + B[safe] + bt_)).sum())
    low_ex = int((valid & (vs < t)).sum())
    low_tol = int((valid & (vs < t - vb - bt_)).sum())
    base = nvalid == min(TOPK, n) and uniq == nvalid
    set_ok_exact = base and sel_above == ref["n_above"] and low_ex == 0
    set_ok_tol = base and sel_above_tol == ref["n_above_tol"] and low_tol == 0
    c = int(ids[KEEP])
    if 0 <= c < n:
        cs = S[c]
        rank = int((S > cs).sum())
        tie_ex = bool(cs == t)
        tie_tol = bool((cs - t).abs() <= B[c] + bt_)
    else:
        rank, tie_ex, tie_tol = None, False, False
    return dict(c=c, rank=rank, tie_exact=tie_ex, tie_tol=tie_tol, set_ok_exact=bool(set_ok_exact),
                set_ok_tol=bool(set_ok_tol), nvalid=nvalid, dup=nvalid - uniq)


def score(refs, out):
    return [row_metrics(refs[i], out[i].long()) for i in range(out.shape[0])]


def per_rank(R, sl, bt_loc, pages, width_r, rank, dev):
    """0.7.0 DCP2 split: pool p -> rank p % 2, local id p // 2 (as stage 1)."""
    k8, ks = page_split(pages)
    # one K per distinct page-table row (rows of one request share it; a row whose page count
    # differs simply gets its own copy -- each row only ever sees its own first n pools)
    keys = {}
    for i in range(R):
        np_ = int((sl[i] + PAGE - 1) // PAGE)
        keys.setdefault(tuple(bt_loc[i, :np_].tolist()), []).append(i)
    groups = [dict(key=k, rows=v) for k, v in keys.items()]
    pages_r, bt_r = [], torch.zeros((R, width_r), dtype=torch.int32)
    nxt = 1
    sl_r = torch.zeros(R, dtype=torch.int32)
    for g in groups:
        nmax = max(int(sl[i]) for i in g["rows"])
        pg = torch.tensor(g["key"], dtype=torch.int64) - 1
        Kb = k8[pg.to(dev)].reshape(-1, HD)[:nmax][rank::2]
        Sc = ks[pg.to(dev)].reshape(-1)[:nmax][rank::2]
        m = Kb.shape[0]
        p = (m + PAGE - 1) // PAGE
        kb = torch.zeros((p * PAGE, HD), dtype=torch.uint8, device=dev)
        kb[:m] = Kb
        sb = torch.zeros((p * PAGE,), dtype=torch.float32, device=dev)
        sb[:m] = Sc
        blk = torch.cat([kb.reshape(p, PAGE * HD), sb.view(torch.uint8).reshape(p, PAGE * 4)], 1)
        pages_r.append(blk)
        for i in g["rows"]:
            bt_r[i, :p] = torch.arange(nxt, nxt + p, dtype=torch.int32)
            n = int(sl[i])
            sl_r[i] = (n - rank + 1) // 2
        nxt += p
    cache = torch.cat([torch.zeros((1, PB), dtype=torch.uint8, device=dev)] + pages_r, 0)
    return cache, bt_r.to(dev).contiguous(), sl_r.to(dev)


def run_paths(version, r, R, q, w, sl, bt_loc, pages, refs, repeats, dev):
    width = int(r["width"])
    res = {}
    if version == "v091":
        cache, bt, sld = build_v091_cache(R, sl, bt_loc, pages, width, dev)
        plans = [("v091_auto", [list(range(R))], False)]
        if R <= 16:
            plans.append(("v091_tiled_forced", [list(range(R))], True))
        else:
            plans.append(("v091_fused_split16", [list(range(s, min(R, s + 16))) for s in range(0, R, 16)], False))
        for name, chunks, forced in plans:
            reps, routes = [], set()
            for _ in range(repeats):
                full = torch.full((R, TOPK), -1, dtype=torch.int32, device=dev)
                for ch in chunks:
                    ix = torch.tensor(ch, device=dev)
                    o, _, meta = FX.b12x_paged_topk(q[ix].contiguous(), w[ix].contiguous(), cache,
                                                    bt[ix].contiguous(), sld[ix].contiguous(), False, forced)
                    full[ix] = o
                    routes.add(meta["route"])
                torch.cuda.synchronize()
                reps.append(full.clone())
            res[name] = (reps, sorted(routes))
    else:
        wr = width // 2
        caches = [per_rank(R, sl, bt_loc, pages, wr, rk, dev) for rk in (0, 1)]
        reps, routes = [], set()
        for _ in range(repeats):
            idx_r, sc_r = [], []
            for rk in (0, 1):
                cache, bt, sld = caches[rk]
                o, s, meta = FX.b12x_paged_topk(q, w, cache, bt, sld, True, False)
                idx_r.append(o)
                sc_r.append(s)
                routes.add(meta["route"])
            merged, _ = FX.dcp2_merge(idx_r, sc_r)
            torch.cuda.synchronize()
            reps.append(merged.clone())
        res["v070_dcp2_merge"] = (reps, sorted(routes))
    return res


def process(version, path, repeats, dev, fout, with_insitu):
    t0 = time.time()
    r, R, q, w, sl, bt_loc, pages = load_capture(path, dev)
    refs = reference(q, w, sl, bt_loc, pages)
    live = r["out"].to(dev)
    paths = run_paths(version, r, R, q, w, sl, bt_loc, pages, refs, repeats, dev)
    rec = {k: r[k] for k in ("tag", "rank", "layer", "call", "rows", "num_decodes", "next_n", "has_prefill",
                             "route", "ctas_per_group", "merge_threshold", "width", "file")}
    rec["version"] = version
    rec["path"] = path
    rec["seq_lens"] = sl.tolist()
    rec["dec_seq"] = r["dec_seq"].tolist()
    rec["ref_stats"] = [x["stats"] for x in refs]
    if with_insitu:
        rec["insitu"] = score(refs, live)
    rec["paths"] = {}
    for name, (reps, routes) in paths.items():
        ent = dict(routes=routes, reps=[score(refs, o) for o in reps])
        # agreement with the live order (same bytes): col511 id, full order, set
        ent["vs_live"] = [dict(
            col511_same=[bool(o[i, KEEP] == live[i, KEEP]) for i in range(R)],
            order_same=[bool(torch.equal(o[i], live[i])) for i in range(R)],
            set_same=[bool(torch.equal(torch.sort(o[i]).values, torch.sort(live[i]).values)) for i in range(R)])
            for o in reps]
        # run-to-run: distinct col511 ids across repeats per row
        st = torch.stack([o[:, KEEP] for o in reps])
        ent["rtr_col511_distinct"] = [int(torch.unique(st[:, i]).numel()) for i in range(R)]
        rec["paths"][name] = ent
    SEL[path] = {name: torch.stack(reps).to(torch.int16).cpu() for name, (reps, _) in paths.items()}
    rec["secs"] = round(time.time() - t0, 3)
    fout.write(json.dumps(rec) + "\n")
    fout.flush()
    return rec


def selftest_captures(version, dev, d):
    """Fabricate captures in the hook's format from random inputs (stage-1 generator) and the
    0.9.1 auto path as the 'live' output, to validate replay plumbing before the server run."""
    os.makedirs(d, exist_ok=True)
    files = []
    for rows, n in [(16, 3000), (32, 5000)]:
        inp = FX.make_random_inputs(n, rows, "gauss", 12345 + rows, dev)
        width = 4096
        cache, bt, sl = FX.pack_cache(inp, rows, width, dev)
        live, _, meta = FX.b12x_paged_topk(inp["q"], inp["w"], cache, bt, sl, False, False) \
            if version == "v091" else (torch.full((rows, TOPK), -1, dtype=torch.int32, device=dev), None,
                                       dict(route="n/a"))
        npg = (n + PAGE - 1) // PAGE
        btc = bt[:, :npg].long().cpu()
        pid = torch.unique(btc)
        rec = dict(tag="selftest", rank=0, layer="selftest", call=0, rows=rows, num_decodes=rows // 4, next_n=4,
                   has_prefill=False, route=meta["route"], ctas_per_group=None, merge_threshold=None, width=width,
                   q=inp["q"].view(torch.uint8).cpu(), w=inp["w"].cpu(), seq_lens=sl.long().cpu(),
                   dec_seq=(sl.long() * 4).cpu(), bt=btc, page_ids=pid, pages=cache[pid.to(dev)].cpu(),
                   out=live.cpu())
        rec["file"] = f"self_{rows}.pt"
        p = os.path.join(d, rec["file"])
        torch.save(rec, p)
        files.append(p)
        # reference cross-check against the stage-1 reference
        ref1 = FX.reference_scores(inp, dev)
        r2, R2, q2, w2, sl2, bl2, pg2 = load_capture(p, dev)
        refs = reference(q2, w2, sl2, bl2, pg2)
        md = max(float((refs[i]["S"] - ref1[i][0]).abs().max()) for i in range(rows))
        print(f"selftest rows={rows} n={n}: max |ref - stage1 ref| = {md:.3g}", flush=True)
        assert md == 0.0
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["v070", "v091"])
    ap.add_argument("--probe", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--list", default="", help="text file of capture paths (container paths)")
    ap.add_argument("--max-seconds", type=float, default=1500)
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    t0 = time.time()
    import b12x
    p = torch.cuda.get_device_properties(dev)
    print(f"torch {torch.__version__} b12x {b12x.__file__} {p.name} sm{p.major}{p.minor}", flush=True)
    if a.selftest:
        files = selftest_captures(a.version, dev, os.path.join(os.path.dirname(a.out), "selftest-caps"))
    elif a.list:
        files = [l.strip() for l in open(a.list) if l.strip()]
    else:
        files = sorted(f for d in a.probe for f in glob.glob(os.path.join(d, "c511_*.pt")))
    if a.limit:
        files = files[: a.limit]
    print(f"{len(files)} captures", flush=True)
    with open(a.out, "w") as fout:
        for k, f in enumerate(files):
            if time.time() - t0 > a.max_seconds:
                print("STOP: time guard", flush=True)
                break
            rec = process(a.version, f, a.repeats, dev, fout, with_insitu=(a.version == "v091"))
            if k < 3 or k % 50 == 0:
                summ = {n: (e["routes"], sum(1 for m in e["reps"][0] if m["rank"] is not None and m["rank"] < 480
                                             and not m["tie_exact"])) for n, e in rec["paths"].items()}
                print(f"[{time.time()-t0:.0f}s] {k} {os.path.basename(f)} rows={rec['rows']} route={rec['route']} "
                      f"layer={rec['layer']} n={min(rec['seq_lens'])}-{max(rec['seq_lens'])} "
                      f"rank<480 rep0 {summ} {rec['secs']}s", flush=True)
    if not a.selftest:
        torch.save(SEL, a.out.replace(".jsonl", "-selections.pt"))
    print(f"done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
