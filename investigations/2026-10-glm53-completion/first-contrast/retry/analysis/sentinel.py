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
  boundary state (structural) [R-10: PLAN.md amendment 22f s.2; analysis change only, no hook change]
      the KDA state rebuilt from the cached boundary state at c (snapshot after the prefill step), per fixture x rank x
      KDA layer, with the SSM and convolution components evaluated SEPARATELY, each against its OWN PS-a / PS-b baseline:
        SSM   (R-PS-state-ssm)  only the intended committed state: the snapshot's single non-speculative slot (slot
              num_accepted - 1 if a snapshot were speculative); the other slots are not compared
        conv  (R-PS-state-conv) only the relevant convolution entries: the committed rows that fc_common.conv_entries
              maps to positions P - 3 .. P - 1 (the kernel window at the boundary; non-speculative: rows 0..2,
              layout-normalised); the rows the prefill kernel does not write are not compared
      a snapshot whose frontier is not P - 1 (not the intended state) or whose component cannot be identified is a
      failure; per component, PS-c must be bitwise equal to PS-a when that component's PS-a and PS-b are bitwise
      equal, else within 2 x that component's own PS-a / PS-b relative difference. One component's baseline never
      licenses the other component's change.
  limits (reported, 22f s.5): LIMITS below. The KDA block id is not in the guard's scheduler record; restoration of the
      KDA state at c is certified by the restored length c, which the coordinator returns only when the boundary state
      at c exists in all 7 KDA groups.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "analysis"))
import fc_common as fc  # noqa: E402

BLOCK = 4352
STATE_FACTOR = 2.0          # PS-c within 2 x the component's own PS-a / PS-b difference when that baseline is not bitwise
LIMITS = ("MLA / index-K shared-block identity is directly checked; KDA restoration and captured-state consistency are "
          "checked; physical KDA block identity and the complete unsampled state are not established")   # 22f s.5


def shared_range(P):
    c = max(0, ((P - 1) // BLOCK - 1) * BLOCK)
    return c - BLOCK, c, c


def g0_ids(rec):
    grp = [g for g in rec["groups"] if g["bpk"] > 1][0]
    return grp["ids"].tolist(), grp["bpk"]


def committed_ssm(snap, L):
    """[R-10] the intended committed SSM state of a KDA layer in a snapshot: the single slot of a non-speculative
    (prefill / sentinel) snapshot; slot num_accepted - 1 of a speculative one. None when it cannot be identified."""
    s = L.get("ssm_sketch")
    if s is None or s.dim() < 2 or s.shape[0] < 1:
        return None
    if L.get("is_spec"):
        na = snap.get("num_accepted")
        if not isinstance(na, int) or not 1 <= na <= s.shape[0]:
            return None
        return s[na - 1]
    return s[0]


def relevant_conv(snap, L):
    """[R-10] the relevant (committed) convolution entries {absolute position: row} (fc_common.conv_entries, layout-
    normalised; uncommitted / unwritten rows excluded)."""
    if "conv" not in L:
        return {}
    return fc.conv_entries(snap, L)


def _cmp(xa, xb):
    """(bitwise equal, relative L2 difference of xb against xa); NaN difference when not finite."""
    eq = bool(torch.equal(xa, xb))
    r = fc.rel(xb, xa)
    return eq, (r if np.isfinite(r) else float("nan"))


def _component(xs, ref, base, others):
    """one component row: xs = {phase: tensor or None}; ref / base = the PS-a / PS-b phases; others = PS-c ..."""
    if any(xs.get(k) is None for k in [ref, base] + list(others)):
        return dict(ok=False, missing=sorted(k for k in [ref, base] + list(others) if xs.get(k) is None))
    if any(xs[k].shape != xs[ref].shape for k in [base] + list(others)):
        return dict(ok=False, shape_mismatch={k: list(xs[k].shape) for k in [ref, base] + list(others)})
    eq_ab, r_ab = _cmp(xs[ref], xs[base])
    row = dict(eq_ab=eq_ab, rel_ab=r_ab, ok=True)
    for c_ in others:
        eq_ac, r_ac = _cmp(xs[ref], xs[c_])
        ok = eq_ac if eq_ab else bool(r_ac <= STATE_FACTOR * r_ab)     # NaN compares False: fails closed
        row.update(eq_ac=eq_ac, rel_ac=r_ac, limit=0.0 if eq_ab else STATE_FACTOR * r_ab, ok=ok)
    return row


def state_rows(n, P, have, S):
    """[R-10] per rank x KDA layer: separate SSM and conv rows for the sentinel phases `have` (have[0] = PS-a the
    reference, have[1] = PS-b the baseline, have[2:] = PS-c ... compared). S = {phase: {rank: snapshot}}."""
    out = []
    a = have[0]
    for rank in (0, 1):
        if not all(rank in S[k] for k in have):
            miss = dict(ok=False, missing=[k for k in have if rank not in S[k]])
            out.append(dict(fixture=n, rank=rank, layer=None, ssm=dict(miss), conv=dict(miss)))
            continue
        fr = {k: int(S[k][rank].get("frontier_pos", -1)) for k in have}
        layers = sorted(set().union(*[set(fc.kda_layers(S[k][rank])) for k in have]))
        if not layers:
            miss = dict(ok=False, missing="no KDA layer in the snapshots")
            out.append(dict(fixture=n, rank=rank, layer=None, ssm=dict(miss), conv=dict(miss)))
            continue
        for nm in layers:
            Ls = {k: fc.kda_layers(S[k][rank]).get(nm) for k in have}
            row = dict(fixture=n, rank=rank, layer=fc.layer_no(nm))
            if any(v != P - 1 for v in fr.values()):
                bad = dict(ok=False, not_intended_state=dict(frontier=fr, expected=P - 1))
                row.update(ssm=dict(bad), conv=dict(bad))
                out.append(row)
                continue
            ssm = {k: (committed_ssm(S[k][rank], L) if L is not None else None) for k, L in Ls.items()}
            row["ssm"] = _component({k: (v.float() if v is not None else None) for k, v in ssm.items()},
                                    a, have[1], have[2:])
            ent = {k: (relevant_conv(S[k][rank], L) if L is not None else {}) for k, L in Ls.items()}
            want = [P - 3, P - 2, P - 1]                # the conv kernel's window at the boundary (kernel 4 - 1)
            if any(not set(want) <= set(e) for e in ent.values()):
                row["conv"] = dict(ok=False, positions={k: sorted(e) for k, e in ent.items()}, expected=want)
            else:
                xs = {k: torch.stack([e[p].float() for p in want]) for k, e in ent.items()}
                row["conv"] = _component(xs, a, have[1], have[2:])
                row["conv"]["positions"] = want
            out.append(row)
    for row in out:
        for comp in ("ssm", "conv"):
            row[comp].update(fixture=row["fixture"], rank=row["rank"], layer=row["layer"])
    return out


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
        # boundary state rebuilt from the cached state at c [R-10]: SSM and conv separately, each its own baseline
        S = {}
        for ph in have:
            sn = run.snaps(f"{ph}-{n}")
            S[ph] = {rk: run.load(p) for F, ranks in sn for rk, p in ranks.items()} if sn else {}
        if len(have) >= 2:
            kda.extend(state_rows(n, P, have, S))
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
    for comp, cname in (("ssm", "SSM: the intended committed state only"),
                        ("conv", "convolution: the relevant committed entries only (positions P - 3 .. P - 1)")):
        rows_c = [k[comp] for k in kda]
        bad_c = [k[comp] for k in kda if not k[comp].get("ok")]
        R.add(f"{prefix}-state-{comp}", "structural", "PASS" if rows_c and not bad_c else "FAIL", "boundary state, "
              f"{cname}: KDA state rebuilt from the cached state at c, per fixture x rank x layer; PS-c bitwise equal "
              "to PS-a where this component's PS-a = PS-b bitwise, else within 2 x this component's own PS-a / PS-b "
              "difference", dict(rows=len(rows_c), bitwise_ab=sum(1 for k in rows_c if k.get("eq_ab")),
                                 compared_c=sum(1 for k in rows_c if "rel_ac" in k), failing=bad_c[:10]), "R-10 / 22f s.2")
    R.add(f"{prefix}-limits", "reported", "REPORTED", "sentinel limits (22f s.5): " + LIMITS, dict(limits=LIMITS),
          "R-13 / 22f s.5")
    kbad = [k for k in kda if not (k["ssm"].get("ok") and k["conv"].get("ok"))]
    return dict(coverage=cover, identity=ident, page_bad=page_bad, kda_bad=kbad)
