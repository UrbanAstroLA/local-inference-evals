#!/usr/bin/env python3
"""K GPU entry (runs INSIDE one pinned release image, one GPU, --network none). PREPARED, NOT RUN.

  python3 run_k.py --engine b1|v070 --plan /k/plan.json --bank /k/bank --out /o --max-seconds N
                   [--cells REGEX] [--max-batches N]

Per cell of plan.json for this engine (tiled cells first): for every family and batch, build the
inputs, compute the fp64 reference once, then run the image's own selection path REPEATS times with
the scratch pre-filled 0x00 / 0xFF / 0x7F, and apply the frozen criteria (kref.py docstring,
PROPOSAL.md section 4). Writes results.jsonl (one line per batch x repeat), summary.json (per-cell
verdicts) and failures.pt (ids of failing rows, capped). Stops cleanly at --max-seconds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import kinputs  # noqa: E402
import kplan  # noqa: E402
import kref  # noqa: E402

FORBIDDEN_ENV = ("B12X_FUSED_INDEXER", "B12X_INDEXER_TWO_LEVEL_FOLD", "B12X_INDEXER_TWO_LEVEL_FOLD_MAX_MIB",
                 "B12X_PAGED_INDEX_SUPERTILE_K")
PATTERNS = (0x00, 0xFF, 0x7F)
FAIL_CAP = 400


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def seed_of(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


class Agg:
    KEYS = ("rows", "c1", "c2", "c2_exact_numeric", "c2b_bound", "c3", "c3_first511", "c4", "rank_c1", "rank_c2",
            "rank_c2b_bound", "rank_c2_exact_numeric", "persp", "allgather", "route")

    def __init__(self):
        self.n = {k: 0 for k in self.KEYS}
        self.by_family = {}
        self.desc = dict(stock_col511_rank_lt480=0, stock_col511_rows=0, err_over_bound_max=0.0,
                         poison_beats_live_rows=0, poison_rows=0, wraps_max=0, n_band_max=0,
                         unfilled_neginf_false=0)

    def bump(self, key, fam, k=1):
        if k:
            self.n[key] += k
            self.by_family.setdefault(fam, {}).setdefault(key, 0)
            self.by_family[fam][key] += k


def verdict(a: Agg, complete: bool):
    n = a.n
    if not complete:
        return "NOT-RUN" if n["rows"] == 0 else "PARTIAL"
    if n["route"]:
        return "INVALID-ROUTE"
    hard = n["c1"] + n["c2"] + n["c3"] + n["c3_first511"] + n["c4"] + n["rank_c1"] + n["rank_c2"] + n["persp"] \
        + n["allgather"]
    if hard:
        return "FAIL"
    if n["c2_exact_numeric"] + n["rank_c2_exact_numeric"]:
        return "FAIL-NUMERIC"
    if n["c2b_bound"] + n["rank_c2b_bound"]:
        return "INCONCLUSIVE-BOUND"
    return "PASS"


def set_and_scores(ids, ks, s, b, exact):
    """C1/C2/C2b for one row. Exact families: if the kernel's scores are not bit-exact, the row is
    'numeric' and C2 is judged with the fp32 bound instead (a set failure there is still a FAIL)."""
    cs = kref.check_kernel_scores(ids, ks, s, b, exact)
    numeric = False
    if exact and cs["exact_mismatch"]:
        numeric = True
        st = kref.check_set(ids, s, b, False, ids.shape[0])
    else:
        st = kref.check_set(ids, s, b, exact, ids.shape[0])
    bound_bad = (not exact) and cs.get("err_over_bound_max", 0.0) > 1.0
    return st, cs, numeric, bound_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=list(kplan.ENGINES))
    ap.add_argument("--plan", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seconds", type=float, required=True)
    ap.add_argument("--cells", default="")
    ap.add_argument("--max-batches", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, f"log-{a.engine}.txt"), "w")

    def log(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    plan = json.load(open(a.plan))
    eng = plan["meta"]["engines"][a.engine]
    pins = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pins.json")))[a.engine]
    header = dict(engine=a.engine, image_env=os.environ.get("K_IMAGE_ID"), argv=sys.argv)
    # ---- pre-flight inside the container: files, env, device
    bad = {p: sha(p) for p, h in pins["files"].items() if not os.path.exists(p) or sha(p) != h}
    header["file_pins_ok"] = not bad
    if bad:
        log(f"ABORT: engine files differ from pins: {bad}")
        json.dump(dict(header=header, aborted="file pins"), open(os.path.join(a.out, f"summary-{a.engine}.json"), "w"))
        return 3
    env_bad = [k for k in FORBIDDEN_ENV if k in os.environ]
    header["env"] = {k: v for k, v in os.environ.items() if "B12X" in k}
    if env_bad:
        log(f"ABORT: routing env set: {env_bad}")
        json.dump(dict(header=header, aborted="env"), open(os.path.join(a.out, f"summary-{a.engine}.json"), "w"))
        return 3
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        log(f"ABORT: need exactly one visible CUDA device, have "
            f"{torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        json.dump(dict(header=header, aborted="device"), open(os.path.join(a.out, f"summary-{a.engine}.json"), "w"))
        return 3
    dev = torch.device("cuda", 0)
    pr = torch.cuda.get_device_properties(dev)
    header.update(device=pr.name, sm=f"{pr.major}{pr.minor}", sms=pr.multi_processor_count,
                  torch=torch.__version__)
    if pr.multi_processor_count != plan["meta"]["SMS"] or (pr.major, pr.minor) != (12, 0):
        log(f"ABORT: plan assumes {plan['meta']['SMS']} SMs on sm120, device has "
            f"{pr.multi_processor_count} on sm{pr.major}{pr.minor}")
        return 3
    if torch.cuda.memory_allocated(dev) != 0:
        log("ABORT: device memory already allocated in this process")
        return 3
    from kdrive import Driver
    drv = Driver(eng, dev)
    header["pkg_index_topk_is_paged"] = drv.pkg_same_function
    log(json.dumps(header))
    banks = {f: kinputs.Bank(os.path.join(a.bank, f"bank_L{f.split('_L')[1]}.pt"), dev)
             for f in plan["meta"]["FAMILIES"] if f.startswith("real_")}
    for f, bk in banks.items():
        log(f"bank {f}: {bk.n} pools, {bk.q.shape[0]} query rows")

    cells = [c for c in plan["cells"] if c["engine"] == a.engine and (not a.cells or re.search(a.cells, c["id"]))]
    run_cells(drv, plan, eng, a.engine, cells, banks, dev, a.out, t0, a.max_seconds, a.max_batches, log, header)
    return 0


def run_cells(drv, plan, eng, engine, cells, banks, dev, out, t0, max_seconds, max_batches, log, header):
    """Run the cells in order under the time guard; always writes summary/results/failures."""
    res_f = open(os.path.join(out, f"results-{engine}.jsonl"), "w")
    fails = []
    summary = dict(header=header, cells=[])
    stopped = False
    timing = dict(build=0.0, ref=0.0, kernel=0.0, check=0.0)
    fams = plan["meta"]["FAMILIES"]
    try:
        for cell in cells:
            agg = Agg()
            nb = len(cell["batches"]) if not max_batches else min(max_batches, len(cell["batches"]))
            n_planned = nb * len(fams)
            n_done = 0
            tc = time.time()
            # route / fold plan expectation (C7)
            fold_seen = drv.fold_plan(cell["rows"], eng["width_pages"]) if cell["route"] == "tiled" else None
            fold_ok = True
            if fold_seen is not None:
                exp = cell["expect"]["fold"] or {}
                fold_ok = (fold_seen.get("total_slices") == (exp.get("total_slices", 0) if exp.get("on") else 0)
                           and fold_seen.get("slices") == (exp.get("slices") if exp.get("on") else None))
            if not fold_ok:
                agg.n["route"] += 1
            # PI-3 (DIAGNOSIS-run1.md): a harness exception is confined to its cell (verdict ERROR, traceback
            # kept) and the next cell runs; the time guard still stops the run; a CUDA context that no longer
            # synchronises stops the run (remaining cells NOT-RUN), because later results would be meaningless.
            cell_err = None
            try:
                for fam in fams:
                    exact = fam in plan["meta"]["EXACT_FAMILIES"]
                    for bi, batch in enumerate(cell["batches"][:nb]):
                        if time.time() - t0 > max_seconds:
                            stopped = True
                            raise TimeoutError("time guard")
                        run_batch(drv, eng, cell, fam, exact, bi, batch, banks, dev, agg, res_f, fails, timing)
                        n_done += 1
            except TimeoutError:
                raise
            except Exception:
                cell_err = traceback.format_exc()
                log(f"ERROR in {cell['id']} (confined to this cell)\n{cell_err}")
            finally:
                v = "ERROR" if cell_err is not None else verdict(agg, n_done == n_planned)
                summary["cells"].append(dict(id=cell["id"], kind=cell["kind"], verdict=v, counts=agg.n,
                                             by_family=agg.by_family, desc=agg.desc, batches=n_done,
                                             batches_planned=n_planned, fold_seen=fold_seen, fold_ok=fold_ok,
                                             secs=round(time.time() - tc, 2),
                                             error=cell_err[-4000:] if cell_err else None))
                log(f"{cell['id']:24s} {v:18s} rows={agg.n['rows']} fails="
                    f"{ {k: x for k, x in agg.n.items() if x and k != 'rows'} } "
                    f"err/B={agg.desc['err_over_bound_max']:.3g} {time.time() - tc:.1f}s "
                    f"timing={ {k: round(x, 1) for k, x in timing.items()} }")
            if cell_err is not None:
                sync(dev)       # raises if the CUDA context is unusable -> outer handler, remaining NOT-RUN
    except TimeoutError:
        log("STOPPED: time guard")
    except Exception:
        log("ERROR\n" + traceback.format_exc())
        summary["error"] = traceback.format_exc()
    finally:
        done_ids = {c["id"] for c in summary["cells"]}
        for c in cells:
            if c["id"] not in done_ids:
                summary["cells"].append(dict(id=c["id"], kind=c["kind"], verdict="NOT-RUN"))
        summary["stopped_by_time_guard"] = stopped
        summary["timing"] = timing
        summary["secs"] = round(time.time() - t0, 1)
        tally = {}
        for c in summary["cells"]:
            tally[c["verdict"]] = tally.get(c["verdict"], 0) + 1
        summary["tally"] = tally
        json.dump(summary, open(os.path.join(out, f"summary-{engine}.json"), "w"), indent=1)
        torch.save(fails, os.path.join(out, f"failures-{engine}.pt"))
        res_f.close()
        log(f"tally {tally}; done in {time.time() - t0:.1f}s")
    return summary


PERSP_CAP = 120


def persp_record(agg, fam, rec, persp, rows, row_ref, fails):
    """PI-4 (descriptive only; verdict() and C5 are unchanged): classify each row whose two merge
    perspectives differ. ORDER only (same set), SET differs (only among ties at t / inside the fp32 band, or
    not), and whether column 511 - the one column stock 0.7.0 drops per rank - holds a different pool
    (tie / non-tie). Keeps both perspectives of the first PERSP_CAP differing rows in failures."""
    exact = fam in kplan.EXACT_FAMILIES
    (a_ids, a_sc), (b_ids, b_sc) = persp
    row_ne = (a_ids != b_ids).any(1)
    set_ne = (torch.sort(a_ids, 1).values != torch.sort(b_ids, 1).values).any(1)
    c511_ne = a_ids[:, kref.KEEP] != b_ids[:, kref.KEEP]
    v = torch.stack([row_ne.sum(), (row_ne & ~set_ne & ~c511_ne).sum()]).tolist()   # one sync
    d = agg.desc
    pf = d.setdefault("persp_by_family", {}).setdefault(fam, {})

    def bump(k, n=1):
        d[k] = d.get(k, 0) + n
        pf[k] = pf.get(k, 0) + n
    bump("persp_rows_differ", int(v[0]))
    bump("persp_rows_order_only_cols_0_510", int(v[1]))      # same set, same column 511
    n_kept = sum(1 for f in fails if f.get("kind") == "persp")
    for ri in (set_ne | c511_ne).nonzero().flatten().tolist():
        s, bb = row_ref(ri)
        P = int(s.shape[0])
        a, b = a_ids[ri].long(), b_ids[ri].long()
        n = min(kref.TOPK, P)
        if n == 0:
            continue
        vals, idx = torch.topk(s, n, largest=True, sorted=True)
        t, bt = vals[n - 1], bb[idx[n - 1]]

        def tie(x):           # x: ids of pools; True where the pool is free under C2 (tie at t / band)
            if exact:
                return s[x] == t
            return (s[x] - t).abs() <= bb[x] + bt
        av, bv = a[(a >= 0) & (a < P)], b[(b >= 0) & (b < P)]
        only_a = av[~torch.isin(av, bv)]
        only_b = bv[~torch.isin(bv, av)]
        set_diff = bool(only_a.numel() or only_b.numel())
        info = dict(set_differs=set_diff, n_only_a=int(only_a.numel()), n_only_b=int(only_b.numel()))
        if set_diff:
            bump("persp_rows_set_differ")
            nontie = bool((~tie(only_a)).any() or (~tie(only_b)).any()) if (only_a.numel() and only_b.numel()) \
                else True
            info["set_nontie"] = nontie
            if nontie:
                bump("persp_rows_set_differ_nontie")
        c0, c1 = int(a[kref.KEEP]), int(b[kref.KEEP])
        if c0 != c1:
            bump("persp_rows_col511_differ")
            info["col511"] = [c0, c1]
            if 0 <= c0 < P and 0 <= c1 < P:
                cc = torch.tensor([c0, c1], device=s.device)
                sel = av
                ranks = [int((s[sel] > s[c]).sum()) for c in (c0, c1)]
                info["col511_rank"] = ranks
                d["persp_col511_rank_min"] = min(d.get("persp_col511_rank_min", kref.TOPK), *ranks)
                same = bool(s[c0] == s[c1]) if exact else bool((s[c0] - s[c1]).abs() <= bb[c0] + bb[c1])
                info["col511_tie"] = same
                if not same:
                    bump("persp_rows_col511_differ_nontie")
                info["col511_ref"] = s[cc].tolist()
            else:
                bump("persp_rows_col511_differ_nontie")
        if n_kept < PERSP_CAP:
            n_kept += 1
            fails.append(dict(kind="persp", rec=dict(rec), row=ri, info=info, ids0=a_ids[ri].cpu(),
                              ids1=b_ids[ri].cpu(), sc0=a_sc[ri].cpu(), sc1=b_sc[ri].cpu()))


def run_batch(drv, eng, cell, fam, exact, bi, batch, banks, dev, agg, res_f, fails, timing):
    K, dcp, rows, pad = eng["K"], eng["dcp"], cell["rows"], cell["pad_rows"]
    b, route = cell["boundary_local"], cell["route"]
    ctas = cell["expect"]["ctas_per_group"]
    seed = seed_of(cell["id"], fam, bi)
    tb = time.time()
    reqs = kinputs.build_requests(fam, batch, K, b, route, rows, ctas, dcp, seed, dev, banks)
    Ls = kplan.row_lengths(eng, batch, pad)
    if min(Ls[:len(batch) * (K + 1)]) < 1:      # PI-1 guard: a request row with L <= 0 is not a decode row
        raise ValueError(f"plan error: request row with L < 1 in {cell['id']} batch {bi}: {Ls[:K + 1]}")
    L = torch.tensor(Ls, dtype=torch.int64, device=dev)
    P = L // kplan.KPOOL
    nreq = len(reqs)
    q8 = torch.zeros((rows, kref.HEADS, kref.HD), dtype=torch.uint8, device=dev)
    w = torch.zeros((rows, kref.HEADS), dtype=torch.float32, device=dev)
    for i, r in enumerate(reqs):
        q8[i * (K + 1):(i + 1) * (K + 1)] = r["q8"]
        w[i * (K + 1):(i + 1) * (K + 1)] = r["w"]
    req_of_row = [i // (K + 1) for i in range(nreq * (K + 1))] + [-1] * pad
    agg.desc["wraps_max"] = max(agg.desc["wraps_max"], max(r["wraps"] for r in reqs))
    sync(dev)
    timing["build"] += time.time() - tb
    # ---- fp64 reference (once per batch; per row sliced to its live pools)
    tr = time.time()
    S64, B64 = [], []
    for i, r in enumerate(reqs):
        s, bb = kref.ref_scores(r["q8"], r["w"], r["K8"], r["S"])
        S64.append(s)
        B64.append(bb)
        ps, _ = kref.ref_scores(r["q8"], r["w"], r["poison_k"][None, :], torch.tensor([r["poison_s"]], device=dev))
        for j in range(K + 1):
            pj = int(P[i * (K + 1) + j])
            if pj > 0:          # PI-2: descriptive counter only; was `if pj:` (pj = -1 passed it)
                agg.desc["poison_rows"] += 1
                agg.desc["poison_beats_live_rows"] += int(bool(ps[j, 0] > s[j, :pj].max()))

    def row_ref(ri):
        rq = req_of_row[ri]
        if rq < 0:
            z = torch.zeros(0, dtype=torch.float64, device=dev)
            return z, z
        j = ri - rq * (K + 1)
        pj = int(P[ri])
        return S64[rq][j, :pj], B64[rq][j, :pj]
    sync(dev)
    timing["ref"] += time.time() - tr

    if dcp == 1:
        cache, btq = kinputs.pack([(r["K8"], r["S"], r["poison_k"], r["poison_s"]) for r in reqs],
                                  eng["width_pages"], dev)
        bt_rows = torch.zeros((rows, eng["width_pages"]), dtype=torch.int32, device=dev)
        for ri, rq in enumerate(req_of_row):
            if rq >= 0:
                bt_rows[ri] = btq[rq]
    else:
        caches = []
        for r_ in range(2):
            cache_r, btq_r = kinputs.pack([(*kinputs.local_split(r["K8"], r["S"], r_), r["poison_k"], r["poison_s"])
                                           for r in reqs], eng["width_pages"], dev)
            bt_r = torch.zeros((rows, eng["width_pages"]), dtype=torch.int32, device=dev)
            for ri, rq in enumerate(req_of_row):
                if rq >= 0:
                    bt_r[ri] = btq_r[rq]
            Pr = torch.tensor([kplan.local_pools(int(p), 2, r_) for p in P.tolist()], dtype=torch.int32, device=dev)
            caches.append((cache_r, bt_r, Pr))

    for rep, pat in enumerate(PATTERNS):
        tk = time.time()
        rec = dict(cell=cell["id"], fam=fam, batch=bi, rep=rep, pattern=pat, L=Ls, seed=seed)
        if dcp == 1:
            ids, sc, meta = drv.topk(q8, w, cache, bt_rows, P.to(torch.int32), pat)
            drv.merge_dcp1(ids, sc)
            pre_ids, pre_sc = ids.clone(), sc.clone()
            drv.col511_fix(ids, sc)
            E = drv.expand(ids, L)
            sync(dev)
            timing["kernel"] += time.time() - tk
            tc = time.time()
            route_ok = meta["route"] == cell["expect"]["route"] and (
                cell["route"] != "fused" or meta["ctas_per_group"] == cell["expect"]["ctas_per_group"])
            rec["meta"] = meta
            if not route_ok:
                agg.bump("route", fam)
            bad_rows = []
            for ri in range(rows):
                s, bb = row_ref(ri)
                agg.bump("rows", fam)
                st, cs, numeric, bound_bad = set_and_scores(pre_ids[ri], pre_sc[ri], s, bb, exact)
                agg.desc["err_over_bound_max"] = max(agg.desc["err_over_bound_max"], cs.get("err_over_bound_max", 0.0))
                agg.desc["n_band_max"] = max(agg.desc["n_band_max"], st.get("n_band", 0))
                bf = agg.desc.setdefault("n_band_max_by_family", {})
                bf[fam] = max(bf.get(fam, 0), st.get("n_band", 0))
                if cs["unfilled"] and not cs["unfilled_all_neginf"]:
                    agg.desc["unfilled_neginf_false"] += 1
                c3 = kref.check_col511(pre_ids[ri], pre_sc[ri], ids[ri], s, bb, exact and not numeric)
                f511 = kref.check_set(ids[ri, :kref.KEEP], s, bb, exact and not numeric, kref.KEEP)
                flags = dict(c1=not st["struct_ok"], c2=st["struct_ok"] and not st["set_ok"],
                             c2_exact_numeric=numeric, c2b_bound=bound_bad, c3=not c3["ok"],
                             c3_first511=not f511["set_ok"])
                for k_, v_ in flags.items():
                    agg.bump(k_, fam, int(bool(v_)))
                # descriptive: what stock 0.9.1 (same B12x files, no fix) would have dropped
                if int(P[ri]) >= kref.TOPK:
                    c = int(pre_ids[ri, kref.KEEP])
                    if 0 <= c < s.shape[0]:
                        sel = pre_ids[ri][(pre_ids[ri] >= 0) & (pre_ids[ri] < s.shape[0])].long()
                        rank = int((s[sel] > s[c]).sum())
                        agg.desc["stock_col511_rows"] += 1
                        agg.desc["stock_col511_rank_lt480"] += int(rank < 480)
                if any(flags.values()):
                    bad_rows.append(ri)
                    if len(fails) < FAIL_CAP:
                        fails.append(dict(rec=dict(rec), row=ri, flags=flags, st=st, cs=cs, c3=c3, f511=f511,
                                          pre_ids=pre_ids[ri].cpu(), pre_sc=pre_sc[ri].cpu(), post_ids=ids[ri].cpu(),
                                          L=Ls[ri]))
            c4 = kref.check_expanded(E, ids[:, :kref.KEEP], L)
            if not c4["ok"]:
                agg.bump("c4", fam)
                rec["c4"] = c4
            rec["bad_rows"] = bad_rows
        else:
            ids_l, sc_l, metas = [], [], []
            for r_ in range(2):
                cache_r, bt_r, Pr = caches[r_]
                i_, s_, m_ = drv.topk(q8, w, cache_r, bt_r, Pr, pat)
                ids_l.append(i_)
                sc_l.append(s_)
                metas.append(m_)
            persp, ag_ok, _ = drv.merge_dcp2(ids_l, sc_l, pat)
            # PI-4 control (descriptive): the same merge relaunched on bit-identical inputs. If perspective 0
            # differs from its own relaunch, perspective differences are launch nondeterminism, not rank.
            persp_re, _, _ = drv.merge_dcp2(ids_l, sc_l, pat)
            ids, sc = persp[0]
            E = drv.expand(ids, L)
            sync(dev)
            timing["kernel"] += time.time() - tk
            tc = time.time()
            rec["meta"] = metas
            route_ok = all(m_["route"] == cell["expect"]["route"] and (
                cell["route"] != "fused" or m_["ctas_per_group"] == cell["expect"]["ctas_per_group"]) for m_ in metas)
            if not route_ok:
                agg.bump("route", fam)
            p_eq = bool(torch.equal(persp[0][0], persp[1][0]) and
                        torch.equal(persp[0][1].view(torch.int32), persp[1][1].view(torch.int32)))
            if not p_eq:
                agg.bump("persp", fam)
            if not ag_ok:
                agg.bump("allgather", fam)
            relaunch_eq = bool(torch.equal(persp[0][0], persp_re[0][0]) and
                               torch.equal(persp[0][1].view(torch.int32), persp_re[0][1].view(torch.int32)))
            agg.desc["persp0_relaunch_differs"] = agg.desc.get("persp0_relaunch_differs", 0) + int(not relaunch_eq)
            if not p_eq:
                persp_record(agg, fam, rec, persp, rows, row_ref, fails)
            bad_rows = []
            for ri in range(rows):
                s, bb = row_ref(ri)
                agg.bump("rows", fam)
                flags = {}
                # per rank (local pools p_local <-> global 2*p_local + r)
                for r_ in range(2):
                    sl, bl = s[r_::2], bb[r_::2]
                    st, cs, numeric, bound_bad = set_and_scores(ids_l[r_][ri], sc_l[r_][ri], sl, bl, exact)
                    flags["rank_c1"] = flags.get("rank_c1", False) or not st["struct_ok"]
                    flags["rank_c2"] = flags.get("rank_c2", False) or (st["struct_ok"] and not st["set_ok"])
                    flags["rank_c2_exact_numeric"] = flags.get("rank_c2_exact_numeric", False) or numeric
                    flags["rank_c2b_bound"] = flags.get("rank_c2b_bound", False) or bound_bad
                st, cs, numeric, bound_bad = set_and_scores(ids[ri], sc[ri], s, bb, exact)
                agg.desc["err_over_bound_max"] = max(agg.desc["err_over_bound_max"], cs.get("err_over_bound_max", 0.0))
                agg.desc["n_band_max"] = max(agg.desc["n_band_max"], st.get("n_band", 0))
                bf = agg.desc.setdefault("n_band_max_by_family", {})
                bf[fam] = max(bf.get(fam, 0), st.get("n_band", 0))
                flags.update(c1=not st["struct_ok"], c2=st["struct_ok"] and not st["set_ok"],
                             c2_exact_numeric=numeric, c2b_bound=bound_bad)
                for k_, v_ in flags.items():
                    agg.bump(k_, fam, int(bool(v_)))
                # descriptive: the shipped 0.7.0 policy drops column 511 as merged
                if int(P[ri]) >= kref.TOPK:
                    c = int(ids[ri, kref.KEEP])
                    if 0 <= c < s.shape[0]:
                        sel = ids[ri][(ids[ri] >= 0) & (ids[ri] < s.shape[0])].long()
                        rank = int((s[sel] > s[c]).sum())
                        agg.desc["stock_col511_rows"] += 1
                        agg.desc["stock_col511_rank_lt480"] += int(rank < 480)
                if any(flags.values()):
                    bad_rows.append(ri)
                    if len(fails) < FAIL_CAP:
                        fails.append(dict(rec=dict(rec), row=ri, flags=flags, st=st, cs=cs, ids=ids[ri].cpu(),
                                          rank_ids=[x[ri].cpu() for x in ids_l], L=Ls[ri]))
            c4 = kref.check_expanded(E, ids[:, :kref.KEEP], L)
            if not c4["ok"]:
                agg.bump("c4", fam)
                rec["c4"] = c4
            rec["bad_rows"] = bad_rows
            rec["persp_equal"] = p_eq
            rec["persp0_relaunch_equal"] = relaunch_eq
            rec["allgather_ok"] = ag_ok
        timing["check"] += time.time() - tc
        res_f.write(json.dumps(rec) + "\n")
    res_f.flush()


if __name__ == "__main__":
    sys.exit(main())
