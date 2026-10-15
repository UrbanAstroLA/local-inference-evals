#!/usr/bin/env python3
"""Plan check of the first-contrast session (PROPOSAL.md s.7.4; amendment 22 items 3, 4, 5 and 9). CPU only, no network.

usage: plan_check.py PLAN_JSON        rc 0 iff every rule holds; each failure is printed as "PLAN-FAIL <rule> ..."
The runner runs it on the client's --plan output before anything is started (DRY_RUN included); tests/ feed it
tampered plans (each must fail).

Rules (frozen):
  P1  every keyed forced list is <= WMAX + 1 = 8,449 ids AND has exactly its planned length (no truncation anywhere:
      C/D W + 1; LC/LD W - 7,168 + 1; CV 72; PS 1; references 0)
  P2  prompt lengths: C/D/PS = P; LC/LD and warms of LP salts = P_L (LW1/LW3 P_L; LW2 P + 8,192 [A22-3]); CV = B1 - 63;
      P and P_L are 0 mod 8; every predicted position < 131,072
  P3  ids: prompt and forced-list sha256 equal the frozen ids manifest for the request's construction
  P4  designated <= 1,024 and selection records <= 64; C/D designated set = position_plan's (sha); near-close C/D
      forced lists end in </think> 154842
  P5  salts: role partition (s1-s3 only on primary/CV/PS requests and their warms, l1-l3 only on LW/LC/LD and their
      warms); every run's 8 requests share one salt equal to its warm's; LW salts are the LP salts of their rounds
  P6  cache-safety invariant [A22-3]: under one salt, every request issued after a decode window started under that
      salt satisfies floor(prompt_len / 4352) * 4352 <= that decode start (prompt length of the decode request)
  P7  exact cache accounting: cached_expected = 0 for the first request of a salt; c0 = floor(P/4352)*4352 for every
      later primary / CV / PS request; c0_L for every later LP request; warms record their hit value
  P8  distinct seeds
  P9  the D2 key is present, = [0, len(F) + 1], on every D / LD request and on no other request
  P10 identical composition and order (batch_order) in every C / D / LC / LD run
  P11 order [A22-4]: no injected request before the GATE-1 record point, i.e. every D / LD request comes after LW1,
      LW2, C1, C2 and CV; PS-a and PS-b precede D3; PS-c follows D3 and precedes C3; C1, C2 precede D1, D2
  P12 references: LW requests are keyed with local_fw_ref = P and dumprange [P, P_L]; LW1 and LW2 differ in build
      length (the prefill null is never at equal build length) and share P (shared causal positions)
  P13 CV: two fixtures x four schedules (poison, oracle, reject:2, natural), snapshot targets as fc_geometry, no D2
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FCR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(FCR, "analysis"))
import fc_geometry as geo  # noqa: E402

FAIL = []


def bad(rule, msg):
    FAIL.append((rule, msg))


def main(path):
    plan = [d for d in json.load(open(path)) if d.get("kind") == "request"]
    G = geo.load(FCR)
    man = json.load(open(os.path.join(FCR, "fixtures", "ids_manifest.json")))["fixtures"]
    order = G["order"]
    if not plan:
        bad("P0", "empty plan")
    for d in plan:
        n, ph, role = d.get("fixture"), d.get("phase", ""), d.get("role")
        f = G["fx"].get(n)
        m = man.get(n)
        if f is None:
            bad("P0", "%s unknown fixture %s" % (d.get("tag"), n))
            continue
        W = d.get("W", 0)
        # ---- P1
        if d.get("keyed") and W > geo.WMAX + 1:
            bad("P1", "%s forced list %d > WMAX+1" % (d["tag"], W))
        exp_W = {"C": f["L"], "D": f["L"], "LC": f["L_s"], "LD": f["L_s"], "CV": geo.CV_LIST, "PS": 1, "ref": 0,
                 "warm": 0}.get(role)
        if exp_W is None or W != exp_W:
            bad("P1", "%s forced list %s != planned %s" % (d["tag"], W, exp_W))
        # ---- P2 / P3
        P = d["P"]
        if role in ("C", "D", "PS"):
            ok = P == f["P"] and d["prompt_sha256"] == m["prefix_sha256"]
            if role in ("C", "D"):
                ok &= d["forced_sha256"] == m["forced_sha256"]
            if not ok:
                bad("P2/P3", "%s prompt/forced length or ids differ from the manifest" % d["tag"])
        elif role in ("LC", "LD"):
            if not (P == f["P_L"] and d["prompt_sha256"] == m["lp_prefix_sha256"]
                    and d["forced_sha256"] == m["lp_forced_sha256"]):
                bad("P2/P3", "%s LP prompt/forced length or ids differ" % d["tag"])
        elif role == "ref":
            want = f["ref2_len"] if ph == "LW2" else f["P_L"]
            wsha = m["ref2_prompt_sha256"] if ph == "LW2" else m["lp_prefix_sha256"]
            if P != want or d["prompt_sha256"] != wsha:
                bad("P2/P3", "%s reference build length %d != %d or ids differ" % (d["tag"], P, want))
        elif role == "warm":
            want = f["P"] if d["salt_name"].startswith("s") else f["P_L"]
            wsha = m["prefix_sha256"] if d["salt_name"].startswith("s") else m["lp_prefix_sha256"]
            if P != want or d["prompt_sha256"] != wsha:
                bad("P2/P3", "%s warm prompt differs" % d["tag"])
        elif role == "CV":
            if P != f["cv"]["P_cv"]:
                bad("P2", "%s CV prompt %d != B1 - 63 = %d" % (d["tag"], P, f["cv"]["P_cv"]))
        if role in ("C", "D", "LC", "LD") and P % 8:
            bad("P2", "%s P %% 8 = %d" % (d["tag"], P % 8))
        if P + W >= 131072:
            bad("P2", "%s reaches %d >= 131072" % (d["tag"], P + W))
        # ---- P4
        if d.get("desig_n", 0) > geo.NDESIG or d.get("selrec_n", 0) > geo.NSEL:
            bad("P4", "%s designated %d / selrec %d" % (d["tag"], d.get("desig_n"), d.get("selrec_n")))
        if role in ("C", "D"):
            import hashlib
            if d.get("desig_sha") != hashlib.sha256(json.dumps(f["desig"]).encode()).hexdigest()[:16]:
                bad("P4", "%s designated set differs from position_plan" % d["tag"])
            if f["near_close"] and d.get("last_forced") != geo.CLOSE:
                bad("P4", "%s near-close list does not end in </think>" % d["tag"])
        # ---- P9
        injected = role in ("D", "LD")
        if injected and d.get("d2") != [0, W + 1]:
            bad("P9", "%s injected request without the D2 key [0, W+1] (got %s)" % (d["tag"], d.get("d2")))
        if not injected and d.get("d2") is not None:
            bad("P9", "%s carries the D2 key but is not an injected request" % d["tag"])
        # ---- P12
        if role == "ref" and (d.get("ref_from") != f["P"] or d.get("dumprange") != [f["P"], f["P_L"]]):
            bad("P12", "%s reference keys ref=%s dumprange=%s" % (d["tag"], d.get("ref_from"), d.get("dumprange")))
    # ---- P5 salts / roles
    for d in plan:
        sn = d.get("salt_name", "")
        lp = d["role"] in ("ref", "LC", "LD") or (d["role"] == "warm" and sn.startswith("l"))
        if lp != sn.startswith("l") or not (sn[1:] in ("1", "2", "3") and sn[0] in "sl"):
            bad("P5", "%s salt %s outside its role partition" % (d["tag"], sn))
        if d["salt"] != geo.salt(sn):
            bad("P5", "%s salt string %s != %s" % (d["tag"], d["salt"], geo.salt(sn)))
    if len({geo.salt(x) for x in ("s1", "s2", "s3", "l1", "l2", "l3")} & {d["salt"] for d in plan}) != 6 or \
            len({d["salt"] for d in plan}) != 6:
        bad("P5", "salts are not exactly six distinct round salts: %s" % sorted({d["salt"] for d in plan}))
    runs = collections.OrderedDict()
    for d in plan:
        if d["role"] in ("C", "D", "LC", "LD"):
            runs.setdefault(d["phase"], []).append(d)
    for ph, rs in runs.items():
        if [r["fixture"] for r in rs] != order:
            bad("P10", "%s composition/order %s" % (ph, [r["fixture"] for r in rs]))
        if len({r["salt"] for r in rs}) != 1:
            bad("P5", "%s mixes salts" % ph)
    # ---- P6 / P7 cache invariant and expected cached values (in issue order)
    first, starts = set(), collections.defaultdict(list)
    for d in plan:
        sn, n = d["salt_name"], d["fixture"]
        f = G["fx"][n]
        key = (sn, n)
        for s0 in starts[key]:
            if (d["P"] // geo.BLOCK) * geo.BLOCK > s0:
                bad("P6", "%s prompt %d covers a full block past decode start %d under %s" % (d["tag"], d["P"], s0, sn))
        if key not in first:
            if d["cached_expected"] != 0:
                bad("P7", "%s first request of %s expects %s cached (not 0)" % (d["tag"], sn, d["cached_expected"]))
            first.add(key)
        else:
            want = f["c0"] if sn.startswith("s") else f["c0_L"]
            if d["cached_expected"] != want:
                bad("P7", "%s expects %s cached, rule gives %s" % (d["tag"], d["cached_expected"], want))
        if d["role"] in ("C", "D", "LC", "LD", "CV", "PS"):
            starts[key].append(d["P"])
    # ---- P8
    seeds = [d["seed"] for d in plan]
    if len(seeds) != len(set(seeds)):
        bad("P8", "duplicate seeds: %d of %d distinct" % (len(set(seeds)), len(seeds)))
    # ---- P11 order
    pos = {}
    for i, d in enumerate(plan):
        pos.setdefault(d["phase"], []).append(i)
    first_inj = min([i for i, d in enumerate(plan) if d["role"] in ("D", "LD")], default=None)
    for need in ("LW1", "LW2", "C1", "C2", "CV"):
        if need not in pos:
            bad("P11", "phase %s missing" % need)
        elif first_inj is not None and max(pos[need]) > first_inj:
            bad("P11", "injected request before %s completes" % need)

    def before(a, b):
        return a in pos and b in pos and max(pos[a]) < min(pos[b])
    for a, b in (("PS-a", "D3"), ("PS-b", "D3"), ("D3", "PS-c"), ("PS-c", "C3"), ("C1", "D1"), ("C2", "D2")):
        if not before(a, b):
            bad("P11", "%s must precede %s" % (a, b))
    # ---- P12 build lengths
    lw = {d["phase"]: {} for d in plan if d["role"] == "ref"}
    for d in plan:
        if d["role"] == "ref":
            lw[d["phase"]][d["fixture"]] = d["P"]
    if "LW1" in lw and "LW2" in lw:
        if any(lw["LW1"].get(n) == lw["LW2"].get(n) for n in order):
            bad("P12", "LW1 and LW2 at equal build length: the prefill null would be at the same build length")
    # ---- P13 CV
    cv = [d for d in plan if d["role"] == "CV"]
    got = sorted((d["fixture"], d.get("cv_mode")) for d in cv)
    want = sorted((n, m) for n in geo.CV_FIXTURES for m in geo.CV_MODES)
    if got != want:
        bad("P13", "CV requests %s != %s" % (got, want))
    for d in cv:
        t = G["fx"][d["fixture"]]["cv"]["targets"][d["cv_mode"]]
        if d.get("snap") != t or d.get("mode") != d.get("cv_mode") or d.get("d2") is not None:
            bad("P13", "%s snapshot targets / schedule / D2 key" % d["tag"])
    for r_, m_ in FAIL:
        print("PLAN-FAIL", r_, m_)
    print("plan_check: %d requests, %d failures" % (len(plan), len(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
