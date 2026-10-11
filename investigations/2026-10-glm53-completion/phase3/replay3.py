"""Phase 3 offline/in-place scoring of the column-511 interventions (PLAN.md amendment 10c).

Offline (GPU0, 0.9.1 release image):
  python3 replay3.py --list LIST --out OUT.jsonl [--repeats 3] [--graph-test N]
  Per stored stage-2 capture, all rows in one call (route by rows, as served), each repeat:
    base    unpatched kernel call (out_scores=None, two-level fold auto)       -- the reference kernel
    fold0   (i)  same call with B12X_INDEXER_TWO_LEVEL_FOLD=0
    sc_raw  (ii) same call with an out_scores buffer (kernel order, before the swap)
    fix     (ii) sc_raw + _move_lowest_selected_pool_last (source taken from the patched engine file)
In-place (CPU is fine):
  python3 replay3.py --insitu --probe DIR --out OUT.jsonl --device cpu
  Scores the live 512-column output the probe stored (post-fix order when the fix image runs).
Reference: stage-2 fp64 reference from the same bytes (fixtureB/replay.py: NaN-byte pools excluded).
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/fa")
sys.path.insert(0, "/fb")
import replay as RP  # noqa: E402  stage-2 helpers: load_capture, reference, build_v091_cache

TOPK, KEEP, HEADS = 512, 511, 32
BASE_REPEATS = 3
FOLD_ENV = "B12X_INDEXER_TWO_LEVEL_FOLD"
PATCHED = "/fc/sparse_attn_indexer_kpool.fix.py"


def load_fix_helper():
    """Exec the exact helper text from the patched engine file."""
    src = open(PATCHED).read()
    mod = ast.parse(src)
    fn = [n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == "_move_lowest_selected_pool_last"][0]
    code = ast.get_source_segment(src, fn)
    ns = {"torch": torch}
    exec(code, ns)
    return ns["_move_lowest_selected_pool_last"], code


def metrics(ref, ids):
    """Stage-2 row metrics + 'column 511 is the lowest-scoring selected pool'."""
    m = RP.row_metrics(ref, ids)
    S, B, n = ref["S"], ref["B"], ref["n"]
    valid = (ids >= 0) & (ids < n)
    v = ids[valid]
    c = m["c"]
    if 0 <= c < n and v.numel():
        sv = S[v]
        mn = sv.min()
        am = v[int(torch.argmin(sv))]
        m["lowest"] = bool(S[c] == mn)
        m["lowest_tol"] = bool(S[c] - mn <= B[c] + B[am])
        m["n_below_c"] = int((sv < S[c]).sum())  # selected pools scoring below col 511
    else:
        m["lowest"], m["lowest_tol"], m["n_below_c"] = False, False, None
    m["pass"] = bool(m["lowest"] or m["tie_exact"])
    m["pass_tol"] = bool(m["lowest_tol"] or m["tie_tol"])
    return m


def fold_plan(rows, width):
    from b12x.attention.dsa_indexer import paged as P
    st = P._resolve_supertile_k(None, page_size=64)
    p = P._plan_two_level_fold(q_rows=rows, topk=TOPK, page_size=64, page_table_width=width,
                               supertile_pages=max(1, st // 64), output_physical_slots=False)
    return dict(reason=p.reason, total_slices=p.total_slices, chunks=-(-width // max(1, st // 64)))


def call(q, w, cache, bt, sld, scores, fold0):
    old = os.environ.get(FOLD_ENV)
    if fold0:
        os.environ[FOLD_ENV] = "0"
    try:
        o, s, meta = RP.FX.b12x_paged_topk(q, w, cache, bt, sld, scores, False)
        meta["fold"] = fold_plan(int(q.shape[0]), int(bt.shape[1]))
    finally:
        if fold0:
            if old is None:
                os.environ.pop(FOLD_ENV, None)
            else:
                os.environ[FOLD_ENV] = old
    return o, s, meta


def graph_test(fix, q, w, cache, bt, sld, refs):
    """Capture (ii) = index_topk_fp8(out_scores) + helper in one CUDA graph; replay; score."""
    from b12x.attention.dsa_indexer.paged import PAGED_INDEX_PAGE_SIZE, index_topk_fp8
    from b12x.attention.dsa_indexer.scratch import (INDEXER_SOURCE_LAYOUT_PAGED, B12XIndexerScratchCaps,
                                                    plan_indexer_scratch)
    res = {}
    rows = int(q.shape[0])
    plan = plan_indexer_scratch(B12XIndexerScratchCaps(
        device=q.device, source_layout=INDEXER_SOURCE_LAYOUT_PAGED, num_q_heads=HEADS, max_q_rows=rows,
        max_page_table_width=int(bt.shape[1]), topk=TOPK, mode="decode", shared_page_table=False))
    scratch = [torch.empty(s, dtype=d, device=q.device) for s, d in plan.shapes_and_dtypes()]
    binding = plan.bind(scratch=scratch, real_page_table=bt, cache_seqlens_int32=sld, active_width=None,
                        schedule_metadata=None, expected_num_q_heads=HEADS, shared_page_table=False,
                        output_physical_slots=False)
    out = torch.full((rows, TOPK), -1, dtype=torch.int32, device=q.device)
    sc = torch.empty((rows, TOPK), dtype=torch.float32, device=q.device)

    def body():
        out.fill_(-1)
        index_topk_fp8(q_fp8=q, weights=w, index_k_cache=cache, binding=binding, page_size=PAGED_INDEX_PAGE_SIZE,
                       expected_num_q_heads=HEADS, out_indices=out, out_scores=sc)
        fix(out, sc)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    eager = out.clone()
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        reps = []
        for _ in range(3):
            out.fill_(-7)
            g.replay()
            torch.cuda.synchronize()
            reps.append(out.clone())
        res["full_capture"] = "ok"
        res["full_reps"] = [[metrics(refs[i], o[i].long())["pass"] for i in range(rows)] for o in reps]
        res["full_set_same_as_eager"] = [all(torch.equal(torch.sort(o[i]).values, torch.sort(eager[i]).values)
                                             for i in range(rows)) for o in reps]
    except Exception as e:  # noqa: BLE001
        res["full_capture"] = "error: " + repr(e)[:300]
    # helper alone on static buffers
    try:
        ids0 = torch.stack([torch.randperm(5000, device=q.device)[:TOPK] for _ in range(rows)]).int()
        sc0 = torch.randn((rows, TOPK), device=q.device)
        ids_s, sc_s = ids0.clone(), sc0.clone()
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            fix(ids_s, sc_s)
        for t in range(3):
            ids1 = torch.stack([torch.randperm(5000, device=q.device)[:TOPK] for _ in range(rows)]).int()
            sc1 = torch.randn((rows, TOPK), device=q.device)
            ids_s.copy_(ids1)
            sc_s.copy_(sc1)
            g2.replay()
            exp = ids1.clone()
            fix(exp, sc1.clone())
            ok_eq = torch.equal(ids_s, exp)
            last_low = bool((sc1.gather(1, torch.argmin(sc1, 1, keepdim=True)).squeeze(1) ==
                             sc1.gather(1, (ids1.unsqueeze(2) == ids_s[:, KEEP:].unsqueeze(1)).float()
                                        .argmax(1)).squeeze(1)).all())
            same_set = torch.equal(torch.sort(ids_s, 1).values, torch.sort(ids1, 1).values)
            res.setdefault("helper_graph", []).append(dict(eq_eager=ok_eq, last_is_min=last_low, set_same=same_set))
    except Exception as e:  # noqa: BLE001
        res["helper_graph"] = "error: " + repr(e)[:300]
    return res


def process(path, repeats, dev, fout, fix, graph):
    t0 = time.time()
    r, R, q, w, sl, bt_loc, pages = RP.load_capture(path, dev)
    refs = RP.reference(q, w, sl, bt_loc, pages)
    width = int(r["width"])
    cache, bt, sld = RP.build_v091_cache(R, sl, bt_loc, pages, width, dev)
    rec = {k: r[k] for k in ("tag", "rank", "layer", "call", "rows", "route", "width", "file")}
    rec["path"] = path
    rec["seq_lens"] = sl.tolist()
    rec["n_nan"] = [x["stats"]["n_nan"] for x in refs]
    rec["live"] = [metrics(refs[i], r["out"][i].to(dev).long()) for i in range(R)]
    outs = {k: [] for k in ("base", "fold0", "sc_raw", "fix")}
    metas = {}
    score_err = []
    argmin_agree = []
    for k in range(repeats):
        if k < BASE_REPEATS:
            o, _, m = call(q, w, cache, bt, sld, False, False)
            outs["base"].append(o.clone()); metas["base"] = m
        o, _, m = call(q, w, cache, bt, sld, False, True)
        outs["fold0"].append(o.clone()); metas["fold0"] = m
        o, s, m = call(q, w, cache, bt, sld, True, False)
        outs["sc_raw"].append(o.clone()); metas["sc_raw"] = m
        fx = o.clone(); fs = s.clone()
        fix(fx, fs)
        torch.cuda.synchronize()
        outs["fix"].append(fx)
        # kernel scores vs fp64 reference at the returned ids (relative to the fp32 bound)
        for i in range(R):
            ids = o[i].long()
            ok = (ids >= 0) & (ids < refs[i]["n"])
            S = refs[i]["S"][ids[ok]]
            B = refs[i]["B"][ids[ok]]
            ks = s[i][ok].double()
            fin = torch.isfinite(S)
            if fin.any():
                score_err.append(float(((ks[fin] - S[fin]).abs() / B[fin].clamp(min=1e-30)).max()))
            # does the kernel-score argmin coincide with the fp64 argmin among the selected?
            if ok.any():
                a_k = ids[ok][int(torch.argmin(s[i][ok]))]
                a_r = ids[ok][int(torch.argmin(S))]
                argmin_agree.append(bool(a_k == a_r or refs[i]["S"][a_k] == refs[i]["S"][a_r]))
    rec["meta"] = {k: dict(route=v["route"], fold=v["fold"]) for k, v in metas.items()}
    rec["paths"] = {}
    for name, reps in outs.items():
        base0 = outs["base"][0]
        ent = dict(reps=[[metrics(refs[i], o[i].long()) for i in range(R)] for o in reps])
        ent["set_same_vs_base0"] = [[bool(torch.equal(torch.sort(o[i]).values, torch.sort(base0[i]).values))
                                     for i in range(R)] for o in reps]
        st = torch.stack([o[:, KEEP] for o in reps])
        ent["rtr_col511_distinct"] = [int(torch.unique(st[:, i]).numel()) for i in range(R)]
        rec["paths"][name] = ent
    # sc_raw vs base: does requesting out_scores change the kernel's output order?
    rec["fix_vs_raw_set_same"] = [[bool(torch.equal(torch.sort(outs["fix"][k][i]).values,
                                                    torch.sort(outs["sc_raw"][k][i]).values)) for i in range(R)]
                                  for k in range(repeats)]
    rec["score_err_max_rel_bound"] = max(score_err) if score_err else None
    rec["argmin_agree"] = [sum(argmin_agree), len(argmin_agree)]
    if graph:
        rec["graph"] = graph_test(fix, q, RP_w(w), cache, bt, sld, refs)
    rec["secs"] = round(time.time() - t0, 3)
    fout.write(json.dumps(rec) + "\n")
    fout.flush()
    return rec


def RP_w(w):
    return w.contiguous()


def insitu(paths, dev, fout):
    for k, path in enumerate(paths):
        r, R, q, w, sl, bt_loc, pages = RP.load_capture(path, dev)
        refs = RP.reference(q, w, sl, bt_loc, pages)
        rec = {k2: r[k2] for k2 in ("tag", "rank", "layer", "call", "rows", "route", "width", "file", "t")}
        rec["path"] = path
        rec["seq_lens"] = sl.tolist()
        rec["n_nan"] = [x["stats"]["n_nan"] for x in refs]
        rec["ref_stats"] = [x["stats"] for x in refs]
        rec["live"] = [metrics(refs[i], r["out"][i].to(dev).long()) for i in range(R)]
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        if k % 25 == 0:
            p = sum(m["pass"] for m in rec["live"])
            print(f"{k} {os.path.basename(path)} rows={R} route={rec['route']} pass={p}/{R}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="")
    ap.add_argument("--probe", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--graph-test", type=int, default=0, help="run the CUDA-graph test on the first N captures")
    ap.add_argument("--insitu", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-seconds", type=float, default=900)
    ap.add_argument("--base-repeats", type=int, default=3)
    a = ap.parse_args()
    global BASE_REPEATS
    BASE_REPEATS = a.base_repeats
    t0 = time.time()
    if a.list:
        files = [l.strip() for l in open(a.list) if l.strip()]
    else:
        files = sorted(f for d in a.probe for f in glob.glob(os.path.join(d, "c511_*.pt")))
    if a.limit:
        files = files[: a.limit]
    print(f"{len(files)} captures", flush=True)
    if a.insitu:
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        with open(a.out, "w") as fout:
            insitu(files, torch.device(a.device), fout)
        print(f"done in {time.time()-t0:.1f}s", flush=True)
        return
    dev = torch.device("cuda", 0)
    import b12x
    p = torch.cuda.get_device_properties(dev)
    print(f"torch {torch.__version__} b12x {b12x.__file__} {p.name} sm{p.major}{p.minor}", flush=True)
    fix, code = load_fix_helper()
    print("helper source:\n" + code, flush=True)
    with open(a.out, "w") as fout:
        for k, f in enumerate(files):
            if time.time() - t0 > a.max_seconds:
                print("STOP: time guard", flush=True)
                break
            rec = process(f, a.repeats, dev, fout, fix, graph=(k < a.graph_test))
            if k < 3 or k % 50 == 0:
                s = {n: sum(m["pass"] for m in e["reps"][0]) for n, e in rec["paths"].items()}
                print(f"[{time.time()-t0:.0f}s] {k} {os.path.basename(f)} rows={rec['rows']} route={rec['route']} "
                      f"meta={rec['meta']['base']['route']}/{rec['meta']['fold0']['fold']['reason']} "
                      f"pass rep0 {s} {rec['secs']}s graph={rec.get('graph', {}).get('full_capture')}", flush=True)
    print(f"done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
