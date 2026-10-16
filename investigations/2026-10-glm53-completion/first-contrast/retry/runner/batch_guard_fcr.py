#!/usr/bin/env python3
"""[R-17] Batch guard (PLAN.md amendment 22h s.2): a phase (build, touch pass, sentinel pass, C / D / LC / LD / CV run
batch) starts only if its projected completion fits before the client deadline. The runner sets that deadline to
T0 + min(wall cap, remaining approved budget) - the cleanup reserve, so one comparison covers both the remaining budget
and the wall cap. A phase is never cut mid-way by this guard; the runner's hard wall-cap watchdog stays the last-resort
safety. Replaces the client's "never truncate" rule, which assumed 25 tok/s per request whatever the session measured.

Projection of a phase, from rates measured in THIS session:
    projected_s = MARGIN x (uncached prompt tokens / prefill rate + max_tokens / decode rate) + fixed overhead
  * prefill rate: the slowest prompt-token rate measured on this session's completed builds; before any build, the
    cost model's pessimistic 2,410 tok/s (PRIOR_PF);
  * decode rate, per run form (C, D, LC, LD, CV separately: an injected arm may decode differently from a clean one,
    and LP batches differ in length): the slowest rate measured on this session's completed batches of that form,
    each batch's rate = max_tokens / (batch seconds - its uncached prefill at the measured prefill rate); before the
    form's first batch, PRIOR_DEC = 11.0 tok/s per request, below the single-token-advance rate
    of the 22f stress replay (11.9-12.3 tok/s, replay/out/F3_stress.out) -- a conservative prior;
  * MARGIN 1.25 on the variable part; a fixed 30 s per phase (the measured 6.3 s dump pause after a phase, the
    submission stagger of at most 7 s, and slack). The attempt-1 need rule's 8 x 6.3 s + 30 s per phase made the LP
    reserve skip every LP batch at pessimistic rates although the whole session fits (cost/cost_fcr.out, R-18).
Decision: a primary phase starts iff now + projected <= deadline; otherwise the session stops BEFORE it (rc 17, reading
"incomplete contrast (batch guard)"). An LP batch additionally keeps the projected time of every later primary phase
(and GATE_S for a GATE-1R not yet run) in reserve; if it does not fit it is skipped, and so is every later LP batch
(the existing LP-first rule). Pure functions; used by runner/client_fcr.py and cost/cost_fcr.py (the stress replay).
"""
PRIOR_PF = 2410.0
PRIOR_DEC = 11.0
MARGIN = 1.25
GATE_S = 600.0               # GATE-1R pessimistic (cost model), reserved by LP batches before the gate
GUARD_RC = 17
READING = "incomplete contrast (batch guard)"
DUMP_S = 6.3
FIXED_S = 30.0               # per phase: dump pause DUMP_S, submission stagger <= 7 s, slack
DECODE_FORMS = ("C", "D", "LC", "LD", "CV")


def is_lp(ph):
    return ph.get("form") in ("LC", "LD")


def uncached_tokens(ph):
    rs = ph["requests"]
    if ph["kind"] == "build":
        return sum(r["P"] for r in rs)
    return sum(r["P"] - r["cached_expected"] for r in rs)


def decode_tokens(ph):
    if ph["kind"] != "run":
        return 0
    return max(r["max_tokens"] for r in ph["requests"])


class Rates:
    """Rates measured in this session's completed phases."""

    def __init__(self, prior_pf=PRIOR_PF, prior_dec=PRIOR_DEC, margin=MARGIN):
        self.prior_pf, self.prior_dec, self.margin = prior_pf, prior_dec, margin
        self.pf = []
        self.dec = {f: [] for f in DECODE_FORMS}

    def pf_rate(self):
        return min(self.pf) if self.pf else self.prior_pf

    def dec_rate(self, form):
        v = self.dec.get(form) or []
        return min(v) if v else self.prior_dec

    def observe(self, ph, secs):
        """record a completed phase's measured seconds (client: send of the first request to the last response)."""
        if secs is None or secs <= 0:
            return None
        if ph["kind"] == "build":
            r = uncached_tokens(ph) / secs
            self.pf.append(r)
            return dict(prefill_tok_s=round(r, 1))
        if ph["kind"] == "run" and ph.get("form") in DECODE_FORMS:
            pre = uncached_tokens(ph) / self.pf_rate()
            r = decode_tokens(ph) / max(secs - pre, 1e-3)
            self.dec[ph["form"]].append(r)
            return dict(decode_tok_s=round(r, 2), form=ph["form"])
        return None

    def project(self, ph):
        if ph["kind"] in ("probe", "gate"):
            return 0.0
        var = uncached_tokens(ph) / self.pf_rate()
        if ph["kind"] == "run":
            var += decode_tokens(ph) / self.dec_rate(ph.get("form"))
        return self.margin * var + FIXED_S

    def reserve(self, later, gate_pending):
        """projected seconds of every later PRIMARY phase (LP excluded) plus GATE_S while GATE-1R is still to run."""
        s = sum(self.project(p) for p in later if p["kind"] not in ("probe", "gate") and not is_lp(p))
        return s + (GATE_S if gate_pending and any(p["kind"] == "gate" for p in later) else 0.0)

    def snapshot(self):
        return dict(prefill_tok_s=round(self.pf_rate(), 1),
                    decode_tok_s={f: round(self.dec_rate(f), 2) for f in DECODE_FORMS},
                    observed={f: len(v) for f, v in self.dec.items()}, builds_observed=len(self.pf))


def decide(rates, plan, i, left_s, lp_off):
    """the guard's decision for plan[i] with left_s seconds to the deadline: dict(action=start | skip | stop, ...)."""
    ph = plan[i]
    proj = rates.project(ph)
    if is_lp(ph) and lp_off:
        return dict(action="skip", phase=ph["tag"], why="an earlier LP batch was skipped", projected=round(proj, 1))
    gate_pending = any(p["kind"] == "gate" for p in plan[i + 1:])
    res = rates.reserve(plan[i + 1:], gate_pending) if is_lp(ph) else 0.0
    d = dict(phase=ph["tag"], projected=round(proj, 1), reserve=round(res, 1), left=round(left_s, 1),
             rates=rates.snapshot())
    if left_s >= proj + res:
        return dict(d, action="start")
    if is_lp(ph):
        return dict(d, action="skip", why="batch guard: LP batch skipped, primary phases reserved")
    return dict(d, action="stop", why="batch guard: projected completion of %s (%.0f s) does not fit the %.0f s left "
                "before the deadline" % (ph["tag"], proj, left_s))
