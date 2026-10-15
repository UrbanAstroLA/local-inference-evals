"""First-contrast RETRY session geometry (RETRY-PROPOSAL.md; PLAN.md amendment 22d). ONE definition of the session
plan, every request form, its salt, its build, its expected cache value, the sentinel ranges and the submission policy,
imported by the replay (replay/session_sim.py), the client (runner/client_fcr.py), the plan check
(runner/plan_check_fcr.py), the gate (analysis/gate_fcr.py) and the cost model (cost/cost_fcr.py).

Per-fixture lengths, positions and CV targets come unchanged from the frozen attempt-1 geometry
(../analysis/fc_geometry.py, imported read-only). Changes are labelled R-n (RETRY-PROPOSAL.md s.0).

[R-2] Cache rule (F1, scoped). For a request whose prompt's full 4,352-token blocks were all written by PREFILL of
an earlier build request under the same salt (exact repeat, or a prompt that extends or is a prefix of the build's
prompt), with no eviction in between, the image restores
        c = max(0, (min(floor(P_build / 4352), floor((P_req - 1) / 4352)) - 1) * 4352)
(last token recomputed; DFlash = EAGLE-type -> the last matched block is dropped; block-aligned lookups). This
reduces to the run-fc1 rule max(0, (floor((P - 1) / 4352) - 1) * 4352) for exact repeats. It is NOT used for
anything else: every other case (partial or full eviction, prompts reaching decode-written blocks, other salts) is
decided by the image's own lookup, replayed on CPU (replay/), and in-session by the recorded value.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RETRY = os.path.dirname(HERE)
FC = os.path.dirname(RETRY)
sys.path.insert(0, os.path.join(FC, "analysis"))
import fc_geometry as g1  # noqa: E402  (attempt-1 geometry, frozen, read-only)

BLOCK = g1.BLOCK
LP_SHIFT, REF2_LEN, WMAX, LEAD = g1.LP_SHIFT, g1.REF2_LEN, g1.WMAX, g1.LEAD
CV_FIXTURES, CV_MODES = g1.CV_FIXTURES, g1.CV_MODES
SALT_NONCE = "20261015-fcr1"

# [R-4] F4 submission policy: fixed order, fixed stagger. Builds are submitted shortest-first (deterministic FCFS
# completion order); the runs that use a build are submitted in the build's OBSERVED completion order (client t1),
# so the prefix freed first (oldest in the FIFO free queue) is restored first.
BUILD_STAGGER_S = 1.0
RUN_STAGGER_S = 0.5
GAP_S = 6.3               # measured client gap per phase in run-fc1 (dump pause 6 s + HTTP)


def rule_c(P_build, P_req):
    """[R-2] expected restored length (tokens) for a request on a prefill-built prefix of the same salt."""
    m = min(P_build // BLOCK, (P_req - 1) // BLOCK)
    return max(0, (m - 1) * BLOCK)


def rule_repeat(P):
    return rule_c(P, P)


def salt(name):
    return "fcr1-%s-%s" % (name, SALT_NONCE)


def load(fc_root=FC):
    return g1.load(fc_root)


# ------------------------------------------------------------------ the session plan
def _req(G, n, form, salt_name, build, extra=None):
    f = G["fx"][n]
    P, L = f["P"], f["L"]
    d = dict(fixture=n, form=form, salt=salt_name, build=build)
    if form == "warm":                      # unkeyed build, prompt = prefix
        d.update(P=P, prompt=("prefix", 0), forced=None, max_tokens=1, keyed=False)
    elif form == "LW":                      # keyed prefill reference build (prompt = prefix + forced[:k])
        ln = extra["build_len"]
        d.update(P=ln, prompt=("prefix+forced", ln - P), forced=None, max_tokens=1, keyed=True,
                 ref_from=P, dumprange=[P, f["P_L"]])
    elif form in ("C", "D"):                # long construction: prompt P, forced list L
        d.update(P=P, prompt=("prefix", 0), forced=(0, L), max_tokens=L, keyed=True, d2=form == "D")
    elif form in ("LC", "LD"):              # short construction: prompt P_L, forced list L_s
        d.update(P=f["P_L"], prompt=("prefix+forced", LP_SHIFT), forced=(LP_SHIFT, f["L_s"]), max_tokens=f["L_s"],
                 keyed=True, d2=form == "LD")
    elif form == "CV":
        cv = f["cv"]
        d.update(P=cv["P_cv"], prompt=("prefix+forced", cv["x"]), forced=(cv["x"], cv["list_len"]),
                 max_tokens=cv["list_len"], keyed=True, d2=False, cv_mode=extra["mode"],
                 snap=cv["targets"][extra["mode"]])
    elif form == "touch":                   # [R-3] unkeyed refresh of a build's prefix (no measurement)
        d.update(P=P, prompt=("prefix", 0), forced=None, max_tokens=1, keyed=False)
    elif form == "PS":                      # [R-5] sentinel: prompt P, one forced token, dump the LAST SHARED block
        c = rule_repeat(P)
        d.update(P=P, prompt=("prefix", 0), forced=(0, 1), max_tokens=1, keyed=True, d2=False, snap=[1],
                 dumprange=[c - BLOCK, c], shared_block=[c - BLOCK, c], restored_boundary=c)
    else:
        raise ValueError(form)
    return d


def phases(G, variant="B"):
    """Ordered session plan [R-3]. Each phase: tag, kind, salt, requests (fixture order = SUBMISSION order where fixed;
    runs are re-ordered by the client to the build's observed completion order [R-4]), and the build it restores.

    Builds are cold prefills of the primary prompt P under a fresh salt (0.74 M tokens: inside the measured reach;
    the replay shows every fixture's boundary state surviving to its lookups). LP runs (prompt P_L = P + 7,168) use
    the SAME builds, before any primary decode on that build (cache safety: a later prompt never reaches a
    decode-written block of its stream). LW1 / LW2 are cold prefill references only (no run restores them: the
    0.81 / 0.85 M-token LW builds exceed the reach and lose their earliest-finished fixture, replay/out).

      s1: warm-s1 -> T-s1 -> LC1 -> LC2 -> C1 -> CV                    (clean)
      s2: warm-s2 -> T-s2 -> C2 -> PS-a -> PS-b -> [GATE-1R] -> D2 -> PS-c
      s4: warm-s4 -> T-s4 -> LD1 -> LD2 -> D1
      s3: warm-s3 -> T-s3 -> LD3 -> LC3 -> D3 -> C3
    Submission order of every run / touch / sentinel pass [R-4]: oldest last-freed first, i.e. ascending completion
    time of the most recent earlier request of the same fixture under the same salt (client t1).

    Variant "B": primary contrast + long-prompt control (amendment 22 option B). Variant "A0": primary only.
    """
    order = list(G["order"])
    by_len = sorted(order, key=lambda n: (G["fx"][n]["P"], order.index(n)))      # shortest first (builds)
    lp = variant == "B"
    out = []

    def build(tag, form, s, build_len=None, fixtures=None):
        fx = fixtures or by_len
        reqs = [_req(G, n, form, s, None, dict(build_len=(G["fx"][n]["P"] + build_len) if build_len else None))
                for n in fx]
        out.append(dict(tag=tag, kind="build", form=form, salt=s, requests=reqs, stagger=BUILD_STAGGER_S,
                        reference=form == "LW"))

    def split_build(name, s):
        """[R-3] a salt's build in two halves with a touch pass between them: half a = the 4 longest prompts,
        touch a, half b = the 4 shortest, touch all 8 (oldest last-freed first). The volume built after any
        fixture's last refresh is then <= one half (<= 0.43 M tokens), well inside the measured reach (~0.75 M),
        instead of up to 0.67 M (replay: first-user margin 30 blocks -> >= 500 blocks; a 25-block smaller pool
        loses M3 without the split)."""
        half_a = list(reversed(by_len))[:4]
        half_b = by_len[:4]
        build(name + "-a", "warm", s, fixtures=sorted(half_a, key=by_len.index))
        touch("T-" + s + "-a", s, name + "-a", half_a)
        build(name + "-b", "warm", s, fixtures=half_b)
        touch("T-" + s, s, name + "-b", order)

    def run(tag, form, s, b):
        reqs = [_req(G, n, form, s, b) for n in order]
        out.append(dict(tag=tag, kind="run", form=form, salt=s, build=b, requests=reqs, stagger=RUN_STAGGER_S,
                        order_from=b))

    def touch(tag, s, b, fixtures):
        """[R-3] sequential touch pass right after a build: one unkeyed max_tokens-1 request per fixture (prompt P),
        in the build's completion order. Each restores its prefix [0, c) and frees it to the tail of the FIFO free
        queue (refreshed), creating only a few hashed blocks; the following batch's admissions then draw on blocks
        older than every needed boundary state (replay: margins from 0-30 blocks to >= 500)."""
        out.append(dict(tag=tag, kind="touch", form="touch", salt=s, build=b, sequential=True, order_from=b,
                        requests=[_req(G, n, "touch", s, b) for n in fixtures]))

    def ps(tag, s, b):
        out.append(dict(tag=tag, kind="ps", form="PS", salt=s, build=b, sequential=True,
                        requests=[_req(G, n, "PS", s, b) for n in order]))

    out.append(dict(tag="V0", kind="probe", requests=[]))
    # ---- stage A: references and clean checks before any injected traffic
    build("LW1", "LW", "l1", LP_SHIFT)
    build("LW2", "LW", "l2", REF2_LEN)
    split_build("warm-s1", "s1")
    if lp:
        run("LC1", "LC", "s1", "warm-s1-b")
        run("LC2", "LC", "s1", "warm-s1-b")
    run("C1", "C", "s1", "warm-s1-b")
    out.append(dict(tag="CV", kind="run", form="CV", salt="s1", build="warm-s1-b", stagger=RUN_STAGGER_S,
                    order_from=None,
                    requests=[_req(G, n, "CV", "s1", "warm-s1-b", dict(mode=m)) for n in CV_FIXTURES for m in CV_MODES]))
    split_build("warm-s2", "s2")
    run("C2", "C", "s2", "warm-s2-b")
    ps("PS-a", "s2", "warm-s2-b")
    ps("PS-b", "s2", "warm-s2-b")
    out.append(dict(tag="GATE-1R", kind="gate", requests=[]))
    # ---- stage B: injected batches only after GATE-1R = PASS
    run("D2", "D", "s2", "warm-s2-b")                      # shares s2's build with C2 (adjacent across the gate)
    ps("PS-c", "s2", "warm-s2-b")
    split_build("warm-s4", "s4")
    if lp:
        run("LD1", "LD", "s4", "warm-s4-b")
        run("LD2", "LD", "s4", "warm-s4-b")
    run("D1", "D", "s4", "warm-s4-b")
    split_build("warm-s3", "s3")
    if lp:
        run("LD3", "LD", "s3", "warm-s3-b")
        run("LC3", "LC", "s3", "warm-s3-b")
    run("D3", "D", "s3", "warm-s3-b")
    run("C3", "C", "s3", "warm-s3-b")                      # residue check: clean after injected traffic, same build
    expected_cache(out)
    return out


def expected_cache(plan):
    """[R-2] expected restored length of every request, from the PREFILL coverage of its token stream under its salt:
    the longest earlier prompt of the same fixture under the same salt (every prompt is prefill-written; no prompt
    reaches a decode-written block, cache_invariant). Builds and references are cold (0)."""
    cov = {}
    for p in plan:
        for r in p["requests"]:
            key = (r["salt"], r["fixture"])
            prev = cov.get(key, 0)
            r["cached_expected"] = rule_c(prev, r["P"]) if prev else 0
            r["c_bound"] = r["cached_expected"]
            cov[key] = max(prev, r["P"])
    return plan


def construction_table(plan):
    """[R-3] construction class of every primary and LP run: which build, and how many earlier users of that build
    (runs, CV, sentinels) restored it before this run."""
    users = {}
    out = {}
    for p in plan:
        if p["kind"] in ("run", "ps", "touch"):
            b = p.get("build")
            prior = list(users.get(b, []))
            if p["kind"] == "run" and p["form"] in ("C", "D", "LC", "LD"):
                out[p["tag"]] = dict(build=b, prior_users=prior, cls=("first user" if not prior else
                                                                       "later user (after %s)" % "+".join(prior)))
            users.setdefault(b, []).append(p["tag"])
    return out


def cache_invariant(plan, lcp=None):
    """P6 restated for the retry plan: under one salt, no request issued after a decode window started on a token
    stream may cover a full block that reaches that decode start: floor(P / 4352) * 4352 <= every earlier decode
    start of the same stream under the salt. Block hashes are chained, so another fixture's decode-written block can
    only be reached through a common token prefix: lcp[(a, b)] (from the frozen ids; plan_check_fcr.py supplies it)
    extends the check across streams (a request of fixture a is also bounded by a decode start s of fixture b when
    s < lcp(a, b)). Returns the violations (empty = safe)."""
    starts = {}
    bad = []
    for p in plan:
        for r in p["requests"]:
            s, a = r["salt"], r["fixture"]
            cover = (r["P"] // BLOCK) * BLOCK
            for (b, ds) in starts.get(s, []):
                same = b == a or (lcp is not None and ds < lcp.get((a, b), 0))
                if same and cover > ds:
                    bad.append((p["tag"], a, b, s, cover, ds))
            if r.get("forced"):
                starts.setdefault(s, []).append((a, r["P"]))
    return bad


if __name__ == "__main__":
    G = load()
    for v in ("B", "A0"):
        pl = phases(G, v)
        print(v, [p["tag"] for p in pl])
        print("  invariant violations:", cache_invariant(pl))
        print("  construction:", json.dumps(construction_table(pl)))
