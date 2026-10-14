#!/usr/bin/env python3
"""PA2-21 (PLAN.md amendment 21g item 1, last paragraph): negative-control sensitivity of the REVISED analysis,
shown by offline fault injection on the existing attempt-1 captures. CPU only; attempt-1 data read-only.

  faultinj_a1.py build /t        (in a CPU container of the B1 image, run-a1 mounted read-only at /run, a scratch
                                  directory on /data at /t): writes one run tree per case under /t
  faultinj_a1.py summarize OUT   (host python, json only): reads OUT/<case>/verdict-a2.json, writes OUT/summary.json
                                  and OUT/summary.txt; rc 0 iff the base passes every targeted check and every fault
                                  fails each of its targeted checks
Driven by prep-a2/run_faultinj_a1.sh, which runs analysis/analyze_hookval.py on every tree (HV_ANALYSIS_OUT).

A tree is a run directory of symlinks to run-a1 plus the few files a case changes; nothing in run-a1 is modified.

BASE (the unfaulted reference; every change is stated, none touches a targeted quantity's measured bytes):
  * attempt 1 has no V4n phases: V4n-b1b and V4n-rebuild2 are PROXIES = byte copies of V4-b1 and V4-rebuild with
    their tag changed (client records and phase records added, marked "proxy"). Their nulls are therefore exactly 0,
    so the content limit is the 0.1 % floor and IDX-content / H5e-r FAIL in the base: the content gate is NOT
    demonstrated here (attempt 1 has no matched nulls; it is demonstrated on the synthetic runs). The structural
    checks need the B1 sets present and are evaluated on real attempt-1 bytes.
  * the B1 arms' ledger is REPAIRED to its by-construction value (every processed window pool clean; PA2-2, CPU test
    D7p-a). Attempt 1's real ledger was defective (padded steps shifted attributions by one request).
  * the D2 arm's ledger is replaced by ORACLE labels from the measured bytes (a processed window pool is corrupt iff
    any of its (rank, layer) units is gross against the rebuild). This makes H5f-D2 pass by construction in the base;
    the cases below only show that a SHIFTED ledger is detected, not that the hook's ledger is right.
FAULTS (each must fail its targeted revised checks):
  F1-wrongpage-b1    one fixture's B1-arm window dump: every index-K page holds the adjacent page's contents
  F1-wrongpage-v3    one V3 request's latest snapshot: the same page shift
  F2-attrib-b1       one fixture's B1-arm window dump carries another fixture's pages (tag, P, range kept)
  F2-attrib-v2a      one V2a request's hook record carries another request's measurements (same W)
  F2-attrib-v3       one V3 snapshot carries another request's state (tag, P, F kept)
  F3-ledger-a1       the B1 arms' ledger as attempt 1 actually recorded it (the real padded-step request shift)
  F3-ledger-poolshift-b1   the repaired B1 ledger shifted by one pool
  F3-ledger-poolshift-d2   the D2 ledger shifted by one pool
  F3-ledger-reqshift-d2    the D2 ledger of each request given to the next request of the batch
"""
import glob
import json
import os
import re
import sys

TARGETS = {
    "base": ["IDX-attrib", "IDX-page", "IDX-early", "IDX-stable", "H5f-B1", "H5f-D2", "H2a-r", "H2c-r", "H4-slot",
             "H4b-r", "H4b-sep"],
    "F1-wrongpage-b1": ["IDX-page", "IDX-early"],
    "F1-wrongpage-v3": ["IDX-stable", "IDX-early"],
    "F2-attrib-b1": ["IDX-page", "IDX-early"],
    "F2-attrib-v2a": ["H2c-r"],
    "F2-attrib-v3": ["H4b-r"],
    "F3-ledger-a1": ["H5f-B1"],
    "F3-ledger-poolshift-b1": ["H5f-B1"],
    "F3-ledger-poolshift-d2": ["H5f-D2"],
    "F3-ledger-reqshift-d2": ["H5f-D2"],
}


def summarize(out):
    res, ok_all = {}, True
    for case, tg in TARGETS.items():
        p = os.path.join(out, case, "verdict-a2.json")
        if not os.path.exists(p):
            res[case] = dict(error="no verdict")
            ok_all = False
            continue
        v = json.load(open(p))
        st = {r["id"]: r["status"] for r in v["rows"]}
        want = "PASS" if case == "base" else "FAIL"
        got = {t: st.get(t) for t in tg}
        ok = all(s == want for s in got.values())
        ok_all &= ok
        failing = [r["id"] for r in v["rows"] if r["status"] == "FAIL"]
        res[case] = dict(targeted=got, expected=want, ok=ok, all_failing=failing,
                         raised=[r["id"] for r in v["rows"] if r["id"].endswith("-raised")])
    base_fail = set(res.get("base", {}).get("all_failing", []))
    for case in res:
        if case != "base" and "all_failing" in res[case]:
            res[case]["newly_failing_vs_base"] = sorted(set(res[case]["all_failing"]) - base_fail)
    json.dump(dict(ok=ok_all, cases=res), open(os.path.join(out, "summary.json"), "w"), indent=1)
    lines = [f"fault injection on attempt-1 captures under the REVISED analysis: {'ALL AS EXPECTED' if ok_all else 'NOT AS EXPECTED'}"]
    for case, r in res.items():
        if "error" in r:
            lines.append(f"ERROR {case}: {r['error']}")
            continue
        lines.append("%-5s %-24s targeted %s (expected %s)%s" % (
            "OK" if r["ok"] else "WRONG", case, r["targeted"], r["expected"],
            "" if case == "base" else f"; newly failing vs base {r['newly_failing_vs_base']}"))
    lines.append(f"base fails (not targeted; see the docstring): {sorted(base_fail)}")
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return ok_all


def build(T):
    import numpy as np
    import torch
    RUN = "/run"
    FW = f"{RUN}/forcewin"
    load = lambda p: torch.load(p, weights_only=False)  # noqa: E731
    CL = [json.loads(ln) for ln in open(f"{RUN}/client.jsonl")]
    REQ = {r["tag"]: r for r in CL if r.get("kind") == "request"}
    st = [r for r in CL if r.get("kind") == "start"][0]
    d2set = st["d2set"]

    def snapfile(tag, rank, win=True):
        f = sorted(glob.glob(f"{FW}/snap_{tag}_F*{'win' if win else ''}_r{rank}.pt"))
        f = [x for x in f if bool(re.search(r"_F\d+win_r", x)) == win]
        return f[-1]

    def tree(name, base=None):
        d = os.path.join(T, name)
        os.makedirs(os.path.join(d, "forcewin"), exist_ok=True)
        src = base or RUN
        for f in ("client.jsonl", "image-id.txt", "mapguard"):
            if not os.path.lexists(os.path.join(d, f)):
                os.symlink(os.path.join(src, f), os.path.join(d, f))
        sfw = os.path.join(src, "forcewin")
        for f in os.listdir(sfw):
            if not os.path.lexists(os.path.join(d, "forcewin", f)):
                os.symlink(os.path.join(sfw, f), os.path.join(d, "forcewin", f))
        return d

    def put(d, rel, obj):
        p = os.path.join(d, rel)
        if os.path.lexists(p):
            os.remove(p)
        torch.save(obj, p)

    def put_json(d, rows):
        p = os.path.join(d, "client.jsonl")
        if os.path.lexists(p):
            os.remove(p)
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def idx_layers(rec):
        return [n for n, L in rec["layers"].items() if L.get("kind") == "paged" and L.get("shape", [0])[-1:] == [132]]

    def window_pools(P, W):
        return range((P + 3) // 4, (P + W - 1) // 4)

    # ---------------------------------------------------------------- BASE
    B = tree("base")
    cl = list(CL)
    t_reb = max(r["t1"] for r in CL if r.get("kind") == "phase" and r["tag"] == "V4-rebuild")
    for k, (arm, src_arm, ph) in enumerate((("b1b", "b1", "V4n"), ("rebuild2", "rebuild", "V4n"))):
        for tid in d2set:
            s = dict(REQ[f"V4-{src_arm}-{tid}"])
            s.update(tag=f"V4n-{arm}-{tid}", phase="V4n", batch="proxy", proxy=True)
            if arm == "rebuild2":
                w = REQ[f"V4-b1-{tid}"]
                s["dumprange"] = [w["P"], w["P"] + w["W"]]
            cl.append(s)
            for rank in (0, 1):
                f = snapfile(f"V4-{src_arm}-{tid}", rank)
                a = load(f)
                a["tag"] = f"V4n-{arm}-{tid}"
                put(B, "forcewin/" + os.path.basename(f).replace(f"V4-{src_arm}-", f"V4n-{arm}-"), a)
        cl.append(dict(kind="phase", tag=f"V4n-{arm}", t0=t_reb + k, t1=t_reb + k + 0.5, secs=0.5, proxy=True))
    put_json(B, cl)
    # B1 ledger repaired (both arms; b1b is a copy of b1)
    rep_b1 = {}
    for tid in d2set:
        r = REQ[f"V4-b1-{tid}"]
        for rank in (0, 1):
            a = load(f"{FW}/req_V4-b1-{tid}_r{rank}.pt")
            pc = a["pool_class"].clone()
            for i in window_pools(r["P"], r["W"]):
                pc[i - r["P"] // 4] = 1
            a["pool_class"] = pc
            rep_b1[(tid, rank)] = a
            put(B, f"forcewin/req_V4-b1-{tid}_r{rank}.pt", a)
            b = dict(a)
            b["tag"] = f"V4n-b1b-{tid}"
            put(B, f"forcewin/req_V4n-b1b-{tid}_r{rank}.pt", b)
            c = load(f"{FW}/req_V4-rebuild-{tid}_r{rank}.pt")
            c["tag"] = f"V4n-rebuild2-{tid}"
            put(B, f"forcewin/req_V4n-rebuild2-{tid}_r{rank}.pt", c)
    # D2 ledger: oracle labels from the measured bytes
    import importlib.machinery
    import importlib.util
    sys.argv = ["x", RUN]
    L_ = importlib.machinery.SourceFileLoader("a1", "/a/analyze_hookval.py.a1")
    sp = importlib.util.spec_from_loader("a1", L_)
    A1 = importlib.util.module_from_spec(sp)
    L_.exec_module(A1)
    orc = {}
    for tid in d2set:
        r = REQ[f"V4-d2-{tid}"]
        bad = set()
        for rank in (0, 1):
            a, b = load(snapfile(f"V4-d2-{tid}", rank)), load(snapfile(f"V4-rebuild-{tid}", rank))
            fa, fb = A1.attn_blocks(a), A1.attn_blocks(b)
            for nm in idx_layers(a):
                if nm not in b["layers"]:
                    continue
                for pool in window_pools(r["P"], r["W"]):
                    x = A1.idx_pool(a["layers"][nm]["pages"], fa, pool)
                    y = A1.idx_pool(b["layers"][nm]["pages"], fb, pool)
                    if x is not None and y is not None and float((x - y).norm() / max(float(y.norm()), 1e-12)) > 0.25:
                        bad.add(pool)
        for rank in (0, 1):
            a = load(f"{FW}/req_V4-d2-{tid}_r{rank}.pt")
            pc = a["pool_class"].clone()
            for i in window_pools(r["P"], r["W"]):
                pc[i - r["P"] // 4] = 2 if i in bad else 1
            a["pool_class"] = pc
            orc[(tid, rank)] = a
            put(B, f"forcewin/req_V4-d2-{tid}_r{rank}.pt", a)
    print("base built:", {t: int((orc[(t, 0)]["pool_class"] == 2).sum()) for t in d2set})

    # ---------------------------------------------------------------- faults
    def shift_pages(rec):
        for nm in idx_layers(rec):
            pg = rec["layers"][nm]["pages"]
            ks = sorted(pg)
            if len(ks) < 2:
                continue
            new = {ks[i]: pg[ks[i + 1]] for i in range(len(ks) - 1)}
            new[ks[-1]] = pg[ks[-2]]
            rec["layers"][nm]["pages"] = new
        return rec

    X = d2set[1]
    D = tree("F1-wrongpage-b1", B)
    for rank in (0, 1):
        f = snapfile(f"V4-b1-{X}", rank)
        put(D, "forcewin/" + os.path.basename(f), shift_pages(load(f)))
    v3tag = sorted({re.search(r"snap_(V3-[^_]+(?:_\d+)?-[0-9a-f]+)_F", os.path.basename(p)).group(1)
                    for p in glob.glob(f"{FW}/snap_V3-natural-*_r0.pt")})[0]
    D = tree("F1-wrongpage-v3", B)
    for rank in (0, 1):
        f = snapfile(v3tag, rank, win=False)
        put(D, "forcewin/" + os.path.basename(f), shift_pages(load(f)))
    A_, B_ = d2set[2], d2set[3]
    D = tree("F2-attrib-b1", B)
    for rank in (0, 1):
        fa, fb = snapfile(f"V4-b1-{A_}", rank), snapfile(f"V4-b1-{B_}", rank)
        a, b = load(fa), load(fb)
        a["layers"], a["groups"] = b["layers"], b["groups"]
        put(D, "forcewin/" + os.path.basename(fa), a)
    same_w = sorted([t for t, r in REQ.items() if t.startswith("V2a-natural-")], key=lambda t: (-REQ[t]["W"], t))
    va, vb = same_w[0], same_w[1]
    assert REQ[va]["W"] == REQ[vb]["W"], (va, vb)
    D = tree("F2-attrib-v2a", B)
    for rank in (0, 1):
        a, b = load(f"{FW}/req_{va}_r{rank}.pt"), load(f"{FW}/req_{vb}_r{rank}.pt")
        a["metrics"] = b["metrics"]
        put(D, f"forcewin/req_{va}_r{rank}.pt", a)
    v3b = sorted({re.search(r"snap_(V3-[^_]+(?:_\d+)?-[0-9a-f]+)_F", os.path.basename(p)).group(1)
                  for p in glob.glob(f"{FW}/snap_V3-oracle-*_r0.pt")})[0]
    D = tree("F2-attrib-v3", B)
    for rank in (0, 1):
        fa, fb = snapfile(v3tag, rank, win=False), snapfile(v3b, rank, win=False)
        a, b = load(fa), load(fb)
        a["layers"], a["groups"] = b["layers"], b["groups"]
        put(D, "forcewin/" + os.path.basename(fa), a)
    D = tree("F3-ledger-a1", B)
    for tid in d2set:
        for rank in (0, 1):
            for arm in ("V4-b1", "V4n-b1b"):
                a = load(f"{FW}/req_V4-b1-{tid}_r{rank}.pt")
                a["tag"] = f"{arm}-{tid}"
                put(D, f"forcewin/req_{arm}-{tid}_r{rank}.pt", a)

    def shifted(a, by):
        a = dict(a)
        pc = a["pool_class"].clone()
        pc[by:] = a["pool_class"][:-by]
        pc[:by] = 0
        a["pool_class"] = pc
        return a
    D = tree("F3-ledger-poolshift-b1", B)
    for (tid, rank), a in rep_b1.items():
        put(D, f"forcewin/req_V4-b1-{tid}_r{rank}.pt", shifted(a, 1))
    D = tree("F3-ledger-poolshift-d2", B)
    for (tid, rank), a in orc.items():
        put(D, f"forcewin/req_V4-d2-{tid}_r{rank}.pt", shifted(a, 1))
    D = tree("F3-ledger-reqshift-d2", B)
    for j, tid in enumerate(d2set):
        nxt = d2set[(j + 1) % len(d2set)]
        for rank in (0, 1):
            a = dict(orc[(tid, rank)])
            a["pool_class"] = orc[(nxt, rank)]["pool_class"].clone()
            put(D, f"forcewin/req_V4-d2-{tid}_r{rank}.pt", a)
    json.dump(dict(cases=list(TARGETS), wrongpage_b1=X, wrongpage_v3=v3tag, attrib_b1=[A_, B_], attrib_v2a=[va, vb],
                   attrib_v3=[v3tag, v3b], d2_oracle_corrupt_pools={t: int((orc[(t, 0)]["pool_class"] == 2).sum())
                                                                    for t in d2set}),
              open(os.path.join(T, "build.json"), "w"), indent=1)
    print("built", list(TARGETS))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2])
    else:
        sys.exit(0 if summarize(sys.argv[2]) else 1)
