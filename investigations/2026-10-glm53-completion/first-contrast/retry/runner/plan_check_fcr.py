#!/usr/bin/env python3
"""Plan check of the first-contrast RETRY session (RETRY-PROPOSAL.md s.7; amendment 22d). CPU only, no network.

usage: plan_check_fcr.py PLAN_JSON [--variant B|A0]     rc 0 iff every rule holds; failures "PLAN-FAIL <rule> ..."
The runner runs it on the client's --plan output before anything starts (DRY_RUN included); tests/ feed it tampered
plans (each must fail with its rule). Every expectation is recomputed here from the frozen ids and the frozen geometry
(attempt-1 fc_geometry for lengths and positions), not copied from the plan.

Rules (frozen):
  Q1  every forced list <= WMAX + 1 = 8,449 ids and exactly its planned length (C/D W + 1; LC/LD W - 7,168 + 1; CV 72;
      PS 1; LW / warm / touch 0)
  Q2  prompt lengths: C/D/PS/warm/touch = P; LC/LD = P_L; LW = P_L except LW2 = P + 8,192; CV = B1 - 63; 0 mod 8 where
      attempt 1 required it (P, P_L, P + 8,192); every predicted position < 131,072
  Q3  prompt and forced-list sha256 equal the frozen ids for the request's form
  Q4  designated <= 1,024 and selection records <= 64; C/D designated set = position_plan's; near-close C/D lists end
      in </think>
  Q5  salts: one salt per build family; LW salts only on LW requests, distinct from each other and from every run salt;
      every request of a run / touch / sentinel pass on its build's salt; the salt string = fcr_geometry.salt(name)
  Q6  [R-2] cache-safety invariant per token stream under a salt (with cross-stream reach through common prefixes,
      lcp from the ids): no prompt covers a full block reaching an earlier decode start
  Q7  [R-2] cached_expected = the F1 value from the prefill coverage of the stream under the salt (0 for the first
      request, rule_c(longest earlier prompt, P) after); recomputed here
  Q8  distinct seeds
  Q9  the D2 key = [0, len(F) + 1] on every D / LD request and on no other
  Q10 every C / D / LC / LD run holds exactly the 8 frozen fixtures
  Q11 order: GATE-1R precedes every injected request; LW1, LW2, C1, C2, CV, PS-a, PS-b precede GATE-1R; PS-a, PS-b
      precede D2 and PS-c follows it; D3 precedes C3; C2 and D2 share one salt, D3 and C3 share one salt
  Q12 references: LW keyed with local_fw_ref = P and dumprange [P, P_L]; LW1 and LW2 differ in build length
  Q13 CV: two fixtures x four schedules, snapshot targets as fc_geometry, no D2 key
  Q14 [R-3] every run salt is built in two cold halves (4 longest, then 4 shortest), each followed by a touch pass (the
      second over all 8), and no run / sentinel request on the salt precedes its final touch pass
  Q15 [R-5] sentinels: three passes PS-a, PS-b, PS-c on C2's salt; each dumps exactly [c - 4,352, c) with
      c = its cached_expected; one forced token; snapshot [1]
  Q16 [R-4] submission: builds staggered 1.0 s in the plan's fixed order; runs staggered 0.5 s; touch and sentinel
      passes sequential
"""
import collections
import gzip
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RETRY = os.path.dirname(HERE)
FCR = os.path.dirname(RETRY)
sys.path.insert(0, os.path.join(RETRY, "analysis"))
import fcr_geometry as geo  # noqa: E402

FAIL = []


def bad(rule, msg):
    FAIL.append((rule, msg))


def sha_full(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def lcp_table(ids, order):
    out = {}
    for a in order:
        for b in order:
            if a == b:
                continue
            x, y = ids[a]["prefix"] + ids[a]["forced"], ids[b]["prefix"] + ids[b]["forced"]
            k = 0
            m = min(len(x), len(y))
            while k < m and x[k] == y[k]:
                k += 1
            out[(a, b)] = k
    return out


def main(path, variant):
    data = json.load(open(path))
    plan = [d for d in data if d.get("kind") == "request"]
    phases = [d for d in data if d.get("kind") == "phase"]
    G = geo.load(FCR)
    order = G["order"]
    man = json.load(open(os.path.join(FCR, "fixtures", "ids_manifest.json")))["fixtures"]
    ids = {n: json.loads(gzip.open(os.path.join(FCR, "fixtures", man[n]["file"])).read()) for n in order}
    gp = geo.phases(G, variant)
    pby = {p["tag"]: p for p in gp}
    if not plan:
        bad("Q0", "empty plan")
    seq = [d.get("tag") or d.get("gate") for d in data if d.get("kind") in ("request", "gate")]
    gate_i = next((i for i, d in enumerate(data) if d.get("kind") == "gate"), None)
    if gate_i is None:
        bad("Q11", "no GATE-1R record point in the plan")
    for i, d in enumerate(data):
        if d.get("kind") != "request":
            continue
        n, form, tag = d.get("fixture"), d.get("form"), d.get("tag")
        f = G["fx"].get(n)
        if f is None:
            bad("Q0", "%s unknown fixture %s" % (tag, n))
            continue
        pre, fo = ids[n]["prefix"], ids[n]["forced"]
        W = d["W"]
        # Q1
        expW = {"C": f["L"], "D": f["L"], "LC": f["L_s"], "LD": f["L_s"], "CV": geo.g1.CV_LIST, "PS": 1}.get(form, 0)
        if W > geo.WMAX + 1 or W != expW:
            bad("Q1", "%s forced list %d (planned %d, max %d)" % (tag, W, expW, geo.WMAX + 1))
        # Q2 / Q3
        if form in ("C", "D", "PS", "warm", "touch"):
            want_p, want_f = pre, fo[:W]
        elif form in ("LC", "LD"):
            want_p, want_f = pre + fo[:geo.LP_SHIFT], fo[geo.LP_SHIFT:geo.LP_SHIFT + W]
        elif form == "LW":
            ln = f["P"] + (geo.REF2_LEN if d["phase"] == "LW2" else geo.LP_SHIFT)
            want_p, want_f = pre + fo[:ln - f["P"]], []
        elif form == "CV":
            cv = f.get("cv")
            if cv is None:
                bad("Q13", "%s CV on a non-CV fixture" % tag)
                continue
            want_p, want_f = pre + fo[:cv["x"]], fo[cv["x"]:cv["x"] + cv["list_len"]]
        else:
            bad("Q0", "%s unknown form %s" % (tag, form))
            continue
        if d["P"] != len(want_p) or d["prompt_sha256"] != sha_full(want_p) or d["forced_sha256"] != sha_full(want_f):
            bad("Q2/Q3", "%s prompt/forced length or ids differ from the frozen ids (%s)" % (tag, form))
        if form in ("C", "D", "PS", "LW", "LC", "LD") and d["P"] % 8:
            bad("Q2", "%s P %% 8 = %d" % (tag, d["P"] % 8))
        if d["P"] + W >= 131072:
            bad("Q2", "%s reaches %d" % (tag, d["P"] + W))
        # Q4
        if d.get("desig_n", 0) > geo.g1.NDESIG or d.get("selrec_n", 0) > geo.g1.NSEL:
            bad("Q4", "%s designated %s selrec %s" % (tag, d.get("desig_n"), d.get("selrec_n")))
        if form in ("C", "D"):
            if d.get("desig_sha") != hashlib.sha256(json.dumps(f["desig"]).encode()).hexdigest()[:16]:
                bad("Q4", "%s designated set differs from position_plan" % tag)
            if f["near_close"] and d.get("last_forced") != geo.g1.CLOSE:
                bad("Q4", "%s near-close list does not end in </think>" % tag)
        # Q9
        inj = form in ("D", "LD")
        if inj and d.get("d2") != [0, W + 1]:
            bad("Q9", "%s injected request without the D2 key [0, W+1] (got %s)" % (tag, d.get("d2")))
        if not inj and d.get("d2"):
            bad("Q9", "%s carries the D2 key but is not injected" % tag)
        # Q11 injected after the gate
        if inj and (gate_i is None or i < gate_i):
            bad("Q11", "%s injected before GATE-1R" % tag)
        # Q12
        if form == "LW" and (d.get("ref_from") != f["P"] or d.get("dumprange") != [f["P"], f["P_L"]] or not d["keyed"]):
            bad("Q12", "%s reference keys ref=%s dumprange=%s" % (tag, d.get("ref_from"), d.get("dumprange")))
        # Q13
        if form == "CV":
            if d.get("d2") or d.get("snap") != f["cv"]["targets"].get(d.get("cv_mode")):
                bad("Q13", "%s snapshot targets / schedule / D2 key" % tag)
        # Q15
        if form == "PS":
            c = d["cached_expected"]
            if d.get("dumprange") != [c - geo.BLOCK, c] or d.get("snap") != [1] or W != 1 or c <= 0:
                bad("Q15", "%s sentinel dumprange %s != [c-4352, c) with c=%s" % (tag, d.get("dumprange"), c))
        # Q5 salt string
        if d.get("salt") != geo.salt(d.get("salt_name", "")):
            bad("Q5", "%s salt string %s" % (tag, d.get("salt")))
    # Q5 partition
    by_phase = collections.defaultdict(list)
    for d in plan:
        by_phase[d["phase"]].append(d)
    lw_salts = {d["salt_name"] for d in plan if d["form"] == "LW"}
    other = {d["salt_name"] for d in plan if d["form"] != "LW"}
    if lw_salts & other:
        bad("Q5", "LW salts also used by non-reference requests: %s" % sorted(lw_salts & other))
    if len({by_phase[p][0]["salt_name"] for p in ("LW1", "LW2") if p in by_phase}) != 2:
        bad("Q5", "LW1 and LW2 must have distinct salts")
    for ph, rs in by_phase.items():
        if len({r["salt_name"] for r in rs}) != 1:
            bad("Q5", "%s mixes salts" % ph)
        p = pby.get(ph)
        if p is None and not ph.startswith("CV"):
            bad("Q11", "phase %s not in the frozen plan" % ph)
        if p is not None and p.get("salt") and rs[0]["salt_name"] != p["salt"]:
            bad("Q5", "%s salt %s != plan %s" % (ph, rs[0]["salt_name"], p["salt"]))
    # Q6 / Q7
    lcp = lcp_table(ids, order)
    pl6 = [dict(tag=ph, requests=[dict(salt=r["salt_name"], fixture=r["fixture"], P=r["P"],
                                       forced=(0, r["W"]) if r["W"] else None) for r in rs])
           for ph, rs in by_phase.items()]
    # issue order (plan order of first appearance)
    first = {}
    for i, d in enumerate(plan):
        first.setdefault(d["phase"], i)
    pl6.sort(key=lambda p: first[p["tag"]])
    for v in geo.cache_invariant(pl6, lcp):
        bad("Q6", "%s: %s prompt covers %d past decode start %d of %s under %s" % (v[0], v[1], v[4], v[5], v[2], v[3]))
    cov = {}
    for d in plan:
        key = (d["salt_name"], d["fixture"])
        prev = cov.get(key, 0)
        want = geo.rule_c(prev, d["P"]) if prev else 0
        if d["cached_expected"] != want:
            bad("Q7", "%s expects %s cached, F1 gives %s" % (d["tag"], d["cached_expected"], want))
        cov[key] = max(prev, d["P"])
    # Q8
    seeds = [d["seed"] for d in plan]
    if len(set(seeds)) != len(seeds):
        bad("Q8", "duplicate seeds: %d of %d distinct" % (len(set(seeds)), len(seeds)))
    # Q10
    for ph, rs in by_phase.items():
        if rs[0]["form"] in ("C", "D", "LC", "LD") and sorted(r["fixture"] for r in rs) != sorted(order):
            bad("Q10", "%s composition %s" % (ph, sorted(r["fixture"] for r in rs)))
    # Q11 order
    pos = {t: i for i, t in enumerate(seq)}

    def phase_pos(ph, last=False):
        idx = [i for i, d in enumerate(data) if d.get("kind") == "request" and d["phase"] == ph]
        return (max(idx) if last else min(idx)) if idx else None
    need = ["LW1", "LW2", "C1", "C2", "PS-a", "PS-b"] + (["LC1", "LC2"] if variant == "B" else [])
    for p in need:
        a = phase_pos(p, last=True)
        if a is None:
            bad("Q11", "phase %s missing" % p)
        elif gate_i is not None and a > gate_i:
            bad("Q11", "%s after GATE-1R" % p)
    cvp = [i for i, d in enumerate(data) if d.get("kind") == "request" and d.get("form") == "CV"]
    if not cvp or (gate_i is not None and max(cvp) > gate_i):
        bad("Q11", "CV missing or after GATE-1R")

    def before(a, b):
        x, y = phase_pos(a, last=True), phase_pos(b)
        if x is None or y is None or x > y:
            bad("Q11", "%s must precede %s" % (a, b))
    before("PS-a", "D2")
    before("PS-b", "D2")
    before("D2", "PS-c")
    before("D3", "C3")
    for a, b in (("C2", "D2"), ("D3", "C3"), ("C2", "PS-a")):
        if a in by_phase and b in by_phase and by_phase[a][0]["salt_name"] != by_phase[b][0]["salt_name"]:
            bad("Q11", "%s and %s must share a salt (shared build)" % (a, b))
    # Q12 build lengths
    if "LW1" in by_phase and "LW2" in by_phase:
        if by_phase["LW1"][0]["P"] == by_phase["LW2"][0]["P"]:
            bad("Q12", "LW1 and LW2 at equal build length")
    # Q13 CV composition
    got = sorted((d["fixture"], d.get("cv_mode")) for d in plan if d["form"] == "CV")
    want = sorted((n, m) for n in geo.CV_FIXTURES for m in geo.CV_MODES)
    if got != want:
        bad("Q13", "CV requests %s != %s" % (got, want))
    # Q14 split builds and touch passes
    run_salts = {d["salt_name"] for d in plan if d["form"] in ("C", "D", "LC", "LD", "PS", "CV")}
    longest = sorted(order, key=lambda n: -G["fx"][n]["P"])[:4]
    for s in sorted(run_salts):
        seqs = [(first[ph], ph, rs) for ph, rs in by_phase.items() if rs[0]["salt_name"] == s]
        seqs.sort()
        kinds = [(ph, rs[0]["form"]) for _, ph, rs in seqs]
        if len(kinds) < 4 or [k for _, k in kinds[:4]] != ["warm", "touch", "warm", "touch"]:
            bad("Q14", "salt %s does not start warm(a), touch(a), warm(b), touch(all): %s" % (s, kinds[:4]))
            continue
        a, ta, b, tb = (seqs[j][2] for j in range(4))
        if sorted(r["fixture"] for r in a) != sorted(longest) or sorted(r["fixture"] for r in ta) != sorted(longest):
            bad("Q14", "salt %s first half / touch is not the 4 longest prompts" % s)
        if sorted(r["fixture"] for r in b) != sorted(set(order) - set(longest)) or \
                sorted(r["fixture"] for r in tb) != sorted(order):
            bad("Q14", "salt %s second half / final touch composition" % s)
    # Q15 sentinel passes
    ps = [p for p in ("PS-a", "PS-b", "PS-c") if p in by_phase]
    if ps != ["PS-a", "PS-b", "PS-c"] or len({by_phase[p][0]["salt_name"] for p in ps}) != 1 or \
            ("C2" in by_phase and by_phase["PS-a"][0]["salt_name"] != by_phase["C2"][0]["salt_name"]):
        bad("Q15", "sentinel passes %s not all on C2's salt" % ps)
    # Q16 submission policy (phase records)
    for p in phases:
        g = pby.get(p["tag"])
        if g is None:
            continue
        if g["kind"] == "build" and (p.get("stagger") != geo.BUILD_STAGGER_S or p.get("sequential")):
            bad("Q16", "%s build stagger %s" % (p["tag"], p.get("stagger")))
        if g["kind"] == "run" and (p.get("stagger") != geo.RUN_STAGGER_S or p.get("sequential")):
            bad("Q16", "%s run stagger %s" % (p["tag"], p.get("stagger")))
        if g["kind"] in ("touch", "ps") and not p.get("sequential"):
            bad("Q16", "%s pass not sequential" % p["tag"])
        if g["kind"] == "build":
            want_order = ["%s-%s" % (p["tag"], r["fixture"]) for r in g["requests"]]
            if p.get("submission_order") != want_order:
                bad("Q16", "%s build submission order %s != fixed %s" % (p["tag"], p.get("submission_order"),
                                                                           want_order))
    for r, m in FAIL:
        print("PLAN-FAIL", r, m)
    print("plan check (%s): %d requests, %d phases, %d failures" % (variant, len(plan), len(by_phase), len(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    v = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else "B"
    sys.exit(main(sys.argv[1], v))
