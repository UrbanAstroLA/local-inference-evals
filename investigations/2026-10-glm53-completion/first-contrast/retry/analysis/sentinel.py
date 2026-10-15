"""F5 [R-5]: prefix sentinel defined from the ACTUAL restored boundary c and the shared physical block [c - 4352, c).

Attempt 1 dumped [c0 - 1024, c0), which lies in the block the sentinel request recomputes privately (diagnosis-fc1
s.2.4). Here, for every fixture and every sentinel request (PS-a, PS-b before the injected batch on s2; PS-c after it):
  coverage (structural; missing = incomplete coverage, never a successful sentinel)
      the request restored exactly c (cached == the F1 value, class hit); its dump exists on both ranks with range
      exactly [c - 4352, c); its committed-state snapshot exists on both ranks
  identity (structural)
      the group-0 physical block id at index c / 4352 - 1 is the same in PS-a, PS-b and PS-c, equals the id that the
      s2 build wrote there and that C2 and D2 restored (scheduler log), and equals the id in each dump's block table
  bytes (structural)
      every MLA and index-K page of that physical block byte-identical across PS-a, PS-b, PS-c (both ranks)
  boundary state (structural)
      the KDA state rebuilt from the cached boundary state at c (snapshot after the prefill step) bitwise equal in
      PS-c and PS-a when PS-a and PS-b are bitwise equal, else within 2 x the PS-a / PS-b difference
      (the KDA block id itself is not in the guard's scheduler record: its restoration is certified by the restored
      length c, which the coordinator returns only when the boundary state at c exists in all 7 KDA groups)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "analysis"))
import fc_common as fc  # noqa: E402

BLOCK = 4352


def shared_range(P):
    c = max(0, ((P - 1) // BLOCK - 1) * BLOCK)
    return c - BLOCK, c, c


def g0_ids(rec):
    grp = [g for g in rec["groups"] if g["bpk"] > 1][0]
    return grp["ids"].tolist(), grp["bpk"]


def check(run, G, R, cache_rows, phases=("PS-a", "PS-b", "PS-c"), build_users=("warm-s2", "C2", "D2"), prefix="R-PS"):
    """cache_rows: cache_gate.evaluate rows of the session (for scheduler-side identity)."""
    have = [p for p in phases if any(r["phase"] == p for r in run.reqs)]
    cover, ident, page_bad, kda = [], [], [], []
    page_n = 0
    by = {(r["phase"], r["fixture"]): r for r in cache_rows}
    for n in G["order"]:
        P = G["fx"][n]["P"]
        lo, hi, c = shared_range(P)
        idx = c // BLOCK - 1
        recs_all = {}
        for ph in have:
            tag = f"{ph}-{n}"
            cr = by.get((ph, n))
            cl = run.by_tag.get(tag)
            if cl is None or cr is None or cr["cls"] != "hit" or cr["cached"] != c:
                cover.append((tag, "restore", None if cr is None else (cr["cls"], cr["cached"]), c))
                continue
            if list(cl.get("dumprange") or []) != [lo, hi]:
                cover.append((tag, "dumprange", cl.get("dumprange"), [lo, hi]))
            D = run.win_dump(tag) or {}
            if set(D) != {0, 1}:
                cover.append((tag, "dump ranks", sorted(D)))
            S = run.snaps(tag)
            if not S or not all({0, 1} <= set(rk) for _, rk in S):
                cover.append((tag, "snapshot missing"))
            for rank, p in D.items():
                rec = run.load(p)
                if list(rec.get("range") or []) != [lo, hi] or rec.get("P") != P:
                    cover.append((tag, rank, "dump range/P", rec.get("range"), rec.get("P")))
                recs_all.setdefault(rank, {})[ph] = rec
        # identity: scheduler side (restored ids) and dump block tables
        sched_ids = {}
        for ph in list(have) + [u for u in build_users]:
            cr = by.get((ph, n))
            if cr is None:
                continue
            ids = cr.get("restored_ids") or []
            if ph.startswith("warm"):
                continue
            if idx < len(ids):
                sched_ids[ph] = ids[idx]
        wrote = [r for r in cache_rows if r["fixture"] == n and r["phase"].startswith("warm-s2")]
        for w in wrote:
            if idx < len(w.get("g0_final") or []):
                sched_ids[w["phase"] + " (writer)"] = w["g0_final"][idx]
        if len(set(sched_ids.values())) != 1 or not sched_ids:
            ident.append((n, "scheduler ids differ or missing", sched_ids))
        for rank, recs in recs_all.items():
            dump_ids = {}
            for ph, rec in recs.items():
                ids, bpk = g0_ids(rec)
                dump_ids[ph] = ids[idx * bpk] // bpk if idx * bpk < len(ids) else None
            vals = set(dump_ids.values())
            if len(vals) != 1:
                ident.append((n, rank, "dump block tables differ", dump_ids))
            elif sched_ids and vals != set(sched_ids.values()):
                ident.append((n, rank, "dump block id != scheduler id", dump_ids, sched_ids))
            if len(recs) < 2:
                continue
            first = recs[have[0]] if have[0] in recs else next(iter(recs.values()))
            for nm, La in fc.paged_layers(first).items():
                typ = La.get("type", "")
                if "Tail" in typ:
                    continue
                pages = set()
                ids, bpk = g0_ids(first)
                if La.get("shape", [0])[-1:] == [132]:
                    pages |= {(ids[(p // BLOCK) * bpk] // bpk) * 17 + ((p // 4) % 1088) // 64 for p in range(lo, hi, 4)}
                elif "MLA" in typ:
                    pages |= {ids[p // 64] for p in range(lo, hi, 64)}
                else:
                    continue                    # draft window pages: request-local, not prefix state
                for pg in sorted(pages):
                    xs = [r_["layers"].get(nm, {}).get("pages", {}).get(pg) for r_ in recs.values()]
                    page_n += 1
                    if any(x is None for x in xs) or not all(np.array_equal(xs[0].numpy(), x.numpy()) for x in xs[1:]):
                        page_bad.append((n, rank, fc.layer_no(nm), pg))
        # boundary state rebuilt from the cached state at c
        S = {}
        for ph in have:
            sn = run.snaps(f"{ph}-{n}")
            S[ph] = {rk: run.load(p) for F, ranks in sn for rk, p in ranks.items()} if sn else {}
        if len(have) >= 2:
            a, b = have[0], have[1]
            for rank in (0, 1):
                if not all(rank in S[k] for k in have):
                    kda.append(dict(fixture=n, rank=rank, missing=True))
                    continue
                for nm, La in fc.kda_layers(S[a][rank]).items():
                    Lb = fc.kda_layers(S[b][rank]).get(nm)
                    if Lb is None:
                        kda.append(dict(fixture=n, rank=rank, layer=nm, missing=True))
                        continue
                    eq_ab = bool(np.array_equal(La["ssm_sketch"].numpy(), Lb["ssm_sketch"].numpy())) and \
                        bool(np.array_equal(La["conv"].float().numpy()[:3], Lb["conv"].float().numpy()[:3]))
                    r_ab = fc.rel(Lb["ssm_sketch"][0], La["ssm_sketch"][0])
                    row = dict(fixture=n, rank=rank, layer=fc.layer_no(nm), eq_ab=eq_ab, rel_ab=r_ab, ok=True)
                    for c_ in have[2:]:
                        Lc = fc.kda_layers(S[c_][rank]).get(nm)
                        if Lc is None:
                            row.update(missing=True, ok=False)
                            continue
                        eq_ac = bool(np.array_equal(La["ssm_sketch"].numpy(), Lc["ssm_sketch"].numpy())) and \
                            bool(np.array_equal(La["conv"].float().numpy()[:3], Lc["conv"].float().numpy()[:3]))
                        r_ac = fc.rel(Lc["ssm_sketch"][0], La["ssm_sketch"][0])
                        row.update(eq_ac=eq_ac, rel_ac=r_ac, ok=eq_ac if eq_ab else (r_ac <= 2 * r_ab + 1e-7))
                    kda.append(row)
    kbad = [k for k in kda if k.get("missing") or not k.get("ok")]
    R.add(f"{prefix}-coverage", "structural", "PASS" if have and not cover else "FAIL", "sentinel coverage: every "
          "sentinel request restored exactly c, dumped exactly the shared block [c - 4352, c) on both ranks and took its "
          "boundary-state snapshot (a missing measurement is incomplete coverage, not a successful sentinel)",
          dict(phases=have, violations=cover[:12]), "R-5 / 22d F5")
    R.add(f"{prefix}-identity", "structural", "PASS" if have and not ident else "FAIL", "block identity: the physical "
          "group-0 block at index c / 4352 - 1 is the same in every sentinel request, in C2 / D2 and in each dump's "
          "block table", ident[:12], "R-5")
    R.add(f"{prefix}-bytes", "structural", "PASS" if page_n > 0 and not page_bad else "FAIL", "bytes: every MLA and "
          "index-K page of the shared block identical across the sentinel requests (both ranks)",
          dict(pages=page_n, mismatches=page_bad[:12]), "R-5")
    R.add(f"{prefix}-state", "structural", "PASS" if kda and not kbad else "FAIL", "boundary state: the KDA state "
          "rebuilt from the cached state at c bitwise equal across the sentinels (PS-c vs PS-a when PS-a = PS-b, else "
          "within 2 x the PS-a / PS-b difference)", dict(rows=len(kda), bitwise_ab=sum(1 for k in kda if
                                                                                         k.get("eq_ab")),
                                                        failing=kbad[:10]), "R-5")
    return dict(coverage=cover, identity=ident, page_bad=page_bad, kda_bad=kbad)
