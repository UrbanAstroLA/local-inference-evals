#!/usr/bin/env python3
"""P4 analysis: endpoints E1-E5 and the decision rules of PROPOSAL.md sections 5-6.  CPU only (runs in a CPU
container of the stock image: torch reads the forcewin records).  Specified before any session ran.

usage: analyze_p4.py ROOT OUT_DIR [--a RUNDIR] [--b RUNDIR] [--a2 RUNDIR] [--g4 VERDICT_JSON]
  RUNDIR = ROOT/runs/session-<S>-attempt-<N> (public logs); its private twin ROOT/PRIVATE-runs/session-<S>-attempt-<N>
  holds fw/req_<tag>_r<rank>.pt (forcewin), p4/p4_<tag>_r0.npz (stores) and p4/steprec_*.npz.  Default: per session
  the latest attempt whose outcome.json has gates_ok.  Writes OUT_DIR/result.json and OUT_DIR/result.md (public:
  counts, quantiles, fractions, digests; no ids, text, logits or hidden states).

Units.  Window w (48), repeat r (1, 2), session s (A, B, A2).  Position o in [0, 512) of a window.  Every
comparison pairs the SAME window and position:
  AB_r    A rep r vs B rep r           (the intervention)
  AA      A rep 1 vs A rep 2           (within-arm, in-session repeat; likewise BB and A2A2)
  AA2_r   A rep r vs A2 rep r          (session control: A repeated in a third session)
A position enters a comparison only if both sides measured it exactly once (forcewin nmeas == 1; store have == 1
for E1/E2/E4) and the two steps that measured it have the same STEP KEY: the host step facts (requests, tokens,
padded tokens, draft tokens, prefilling requests, graph mode) and the composition of forced requests in the step
(window, row-0 output index, logits rows, accepted count), with the session label removed (and, for the within-arm
comparison, the repeat label).  Other positions are excluded and counted by reason.

Endpoints (PROPOSAL.md section 5):
  E1  MoE output after the TP all-reduce, layers 3/23/44, every position: bit-equality and rel L2 (bf16 -> fp32).
  E2  hidden state at layers 12, 34 and the final norm, every 8th position: bit-equality and rel L2.
  E3  forcewin row metrics (full-normalized pre-mask log-probabilities): d ln P(forced), d ln P(</think>),
      d ln P(<|endoftext|>, <|user|>, <|observation|>), entropy, top-1 p, argmax change, top_p 0.95 nucleus
      eligibility (elig_lo / elig_hi) of the tracked tokens, post-mask probabilities.
  E4  full-vocabulary KL(A||B), KL(B||A) and TV on the fp32 logits at every 64th position; the renormalized
      top-20 intersection distance is reported as a RESTRICTED DIAGNOSTIC only.
  E5  step time (host inter-step interval), graph mode, occupancy, acceptance (must equal the fixed schedule).

Reproducible difference (fixed before data).  For a metric vector v at position o (E3: the tracked-metric
columns; E1/E2: the stored row; E4: the logits row):
  A-stable(o)   v_A1 == v_A2 == v_A2_1 == v_A2_2 bitwise (all four measured and key-matched),
  D(o)          A-stable(o) and v_B1 != v_A1 and v_B2 != v_A2  (B differs in both repeats).
  Fallback when fewer than half the compared positions are A-stable: "beyond" positions,
  min_r |v_Br - v_Ar| > max(|v_A1 - v_A2|, |v_A1 - v_A2_1|, |v_A2 - v_A2_2|) for scalar metrics, counted for AB and
  for the null pairing (A2 in B's place, B's repeat spread in A2's), and reported side by side.
Decision rules (PROPOSAL.md section 6):
  any gate (G1-G4) failed or a session INCOMPLETE  -> INCOMPLETE; repair; no reading
  E1 has D positions but E3 has none (and no excess of beyond positions)  -> "No demonstrated propagation on these
      windows; investigate attenuation before any behavioural run"
  E3 has D positions (or, in the fallback, more beyond positions than the null pairing)  -> "Computational
      propagation established; behavioural direction untested"
  D positions of E3 on terminal-token log-probabilities in near-close windows are reported as such, still not a
      completion claim.  No threshold dismisses a reproducible small difference; a changed argmax shows
      sensitivity only.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

W = 512
HID_EVERY, LOGIT_EVERY = 8, 64
E3_COLS = {"lp_forced": "lp_pre_4", "lp_think_close": "lp_pre_0", "lp_endoftext": "lp_pre_1", "lp_user": "lp_pre_2",
           "lp_observation": "lp_pre_3", "entropy": "entropy", "top1_p": "top1_p", "p_post_forced": "p_post_4",
           "p_post_think_close": "p_post_0"}
E3_BOOL = {"argmax": "argmax", "elig_lo_forced": "elig_lo_4", "elig_hi_forced": "elig_hi_4",
           "elig_lo_think_close": "elig_lo_0", "elig_hi_think_close": "elig_hi_0"}
TERMINAL = ("lp_think_close", "lp_endoftext", "lp_user", "lp_observation")
MOE_NAMES = ("moe3", "moe23", "moe44")
HID_NAMES = ("hid12", "hid34", "final")


# ------------------------------------------------------------------ loading
def jl(path):
    out = []
    for p in sorted(glob.glob(path)):
        for line in open(p):
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def pick_attempt(root, s):
    best = None
    for p in sorted(glob.glob(os.path.join(root, "runs", f"session-{s}-attempt-*", "outcome.json")),
                    key=lambda x: int(x.split("attempt-")[-1].split("/")[0])):
        o = json.load(open(p))
        if o.get("gates_ok") and int(o.get("batches_done", 0)) >= 1:
            best = os.path.dirname(p)
    return best


def priv_of(rundir):
    return rundir.replace(os.sep + "runs" + os.sep, os.sep + "PRIVATE-runs" + os.sep)


class Session:
    def __init__(self, label, rundir):
        self.label, self.rundir, self.priv = label, rundir, priv_of(rundir)
        self.outcome = json.load(open(os.path.join(rundir, "outcome.json"))) if rundir and \
            os.path.exists(os.path.join(rundir, "outcome.json")) else {}
        self.client = jl(os.path.join(rundir, "client.jsonl")) if rundir else []
        self.steps = {}
        for r in jl(os.path.join(rundir, "p4log", "steps_r0_p*.jsonl")) if rundir else []:
            cols = r["cols"]
            for row in r["rows"]:
                d = dict(zip(cols, row))
                self.steps[int(d["step"])] = d
        self.comp = {}
        for p in sorted(glob.glob(os.path.join(self.priv, "p4", "steprec_*_r0.npz"))):
            z = np.load(p, allow_pickle=False)
            tags = {int(t.split(":", 1)[0]): t.split(":", 1)[1] for t in z["tags"].tolist()}
            for rows in z["rows"]:
                ent = []
                st = None
                for row in rows:
                    if int(row[0]) < 0:
                        continue
                    st = int(row[0])
                    tag = tags.get(int(row[1]))
                    ent.append((strip_session(tag) if tag else "unforced", int(row[2]) if tag else -1,
                                int(row[3]), int(row[4])))
                if st is not None:
                    self.comp[st] = tuple(sorted(ent))

    def tag(self, wid, rep):
        return f"{self.label}-{wid}-r{rep}"

    def fw(self, wid, rep, rank=0):
        p = os.path.join(self.priv, "fw", f"req_{self.tag(wid, rep)}_r{rank}.pt")
        if not os.path.exists(p):
            return None
        import torch
        rec = torch.load(p, map_location="cpu", weights_only=False)
        return dict(cols=list(rec["cols"]), m=rec["metrics"].numpy(), nmeas=rec["nmeas"].numpy(),
                    acc_hist=rec["acc_hist"].numpy(), inv=rec["inv"].numpy(), errors=rec.get("errors", 0))

    def p4(self, wid, rep):
        p = os.path.join(self.priv, "p4", f"p4_{self.tag(wid, rep)}_r0.npz")
        if not os.path.exists(p):
            return None
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}

    def step_key(self, step, strip_rep=False):
        h = self.steps.get(int(step))
        host = None if h is None else (h["num_reqs"], h["num_tokens"], h["num_tokens_padded"], h["num_draft_tokens"],
                                       h["n_prefilling"], h["graph_mode"])
        comp = self.comp.get(int(step))
        if comp is not None and strip_rep:
            comp = tuple(sorted((c[0].rsplit("-r", 1)[0], c[1], c[2], c[3]) for c in comp))
        return (host, comp)


def strip_session(tag):
    return tag.split("-", 1)[1] if tag and "-" in tag else tag


# ------------------------------------------------------------------ elementary comparisons
def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def rel_l2(x, y):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    den = np.linalg.norm(y)
    return float(np.linalg.norm(x - y) / den) if den > 0 else (0.0 if np.linalg.norm(x) == 0 else float("inf"))


def quant(xs):
    if not len(xs):
        return None
    a = np.asarray(xs, dtype=np.float64)
    return dict(n=int(a.size), q50=float(np.quantile(a, 0.5)), q90=float(np.quantile(a, 0.9)),
                q99=float(np.quantile(a, 0.99)), max=float(a.max()), mean=float(a.mean()))


def softmax64(x):
    x = x.astype(np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def kl_tv(la, lb):
    p, q = softmax64(la), softmax64(lb)
    with np.errstate(divide="ignore", invalid="ignore"):
        kpq = float(np.sum(np.where(p > 0, p * (np.log(p) - np.log(np.maximum(q, 1e-300))), 0.0)))
        kqp = float(np.sum(np.where(q > 0, q * (np.log(q) - np.log(np.maximum(p, 1e-300))), 0.0)))
    return kpq, kqp, float(0.5 * np.abs(p - q).sum())


def topk_restricted(la, lb, k=20):
    """RESTRICTED DIAGNOSTIC: TV over the renormalized intersection of the two top-k lists."""
    ia, ib = set(np.argsort(-la)[:k].tolist()), set(np.argsort(-lb)[:k].tolist())
    inter = sorted(ia & ib)
    if not inter:
        return 1.0, 0
    pa = softmax64(la[inter])
    pb = softmax64(lb[inter])
    return float(0.5 * np.abs(pa - pb).sum()), len(inter)


# ------------------------------------------------------------------ per-window extraction
class WinData:
    """Everything of one window in one session and repeat, keyed by position."""

    def __init__(self, sess, wid, rep):
        self.ok = False
        self.reason = None
        fw = sess.fw(wid, rep)
        p4 = sess.p4(wid, rep)
        if fw is None:
            self.reason = "no forcewin record"
            return
        self.fw, self.p4 = fw, p4
        c = {n: i for i, n in enumerate(fw["cols"])}
        self.c = c
        m = fw["m"]
        self.nmeas = fw["nmeas"][:W]
        steps = m[:W, c["step"]]
        self.key = {}
        self.key_norep = {}
        for o in range(W):
            if self.nmeas[o] == 1 and steps[o] == steps[o]:
                self.key[o] = sess.step_key(int(steps[o]))
                self.key_norep[o] = sess.step_key(int(steps[o]), strip_rep=True)
        self.m = m[:W]
        self.ok = True

    def e3(self, o):
        c = self.c
        cols = [c[v] for v in E3_COLS.values()] + [c[v] for v in E3_BOOL.values()]
        return self.m[o, cols]

    def moe(self, k, o):
        if self.p4 is None or self.p4["moe_have"][k, o] != 1:
            return None
        return self.p4["moe"][k, o]

    def hid(self, k, o):
        if self.p4 is None or o % HID_EVERY or self.p4["hid_have"][k, o // HID_EVERY] != 1:
            return None
        return self.p4["hid"][k, o // HID_EVERY]

    def logits(self, o):
        if self.p4 is None or o % LOGIT_EVERY or self.p4["logits_have"][o // LOGIT_EVERY] != 1:
            return None
        return self.p4["logits"][o // LOGIT_EVERY]


def matched(x, y, o, norep=False):
    if not (x.ok and y.ok):
        return False, "window missing"
    if x.nmeas[o] != 1 or y.nmeas[o] != 1:
        return False, "not measured once"
    kx = (x.key_norep if norep else x.key).get(o)
    ky = (y.key_norep if norep else y.key).get(o)
    if kx is None or ky is None or kx[0] is None or ky[0] is None:
        return False, "no step record"
    if kx != ky:
        return False, "step key mismatch"
    return True, None


# ------------------------------------------------------------------ main analysis
def analyze(root, sess, g4_verdict):
    A, B, A2 = sess["A"], sess["B"], sess["A2"]
    windows = []
    plan_batches = []
    for s in (A, B, A2):
        for d in s.client:
            if d.get("kind") == "batch":
                plan_batches.append((d["batch"], tuple(d["windows"])))
    order = []
    for b, ws in sorted(set(plan_batches)):
        for w in ws:
            if w not in order:
                order.append(w)
    windows = order
    pairs = {"AB_1": (A, 1, B, 1, False), "AB_2": (A, 2, B, 2, False), "AA": (A, 1, A, 2, True),
             "BB": (B, 1, B, 2, True), "A2A2": (A2, 1, A2, 2, True), "AA2_1": (A, 1, A2, 1, False),
             "AA2_2": (A, 2, A2, 2, False)}
    excl = {p: {} for p in pairs}
    res = {p: dict(e1={n: dict(bitequal=0, n=0, rel=[]) for n in MOE_NAMES},
                   e2={n: dict(bitequal=0, n=0, rel=[]) for n in HID_NAMES},
                   e3={n: dict(n=0, equal=0, absd=[], signed=[]) for n in list(E3_COLS) + list(E3_BOOL)},
                   e4=dict(n=0, kl_ab=[], kl_ba=[], tv=[], topk_diag=[], equal=0)) for p in pairs}
    repro = {fam: dict(stable=0, compared=0, D=0, D_windows=set(), D_by_class={}) for fam in
             ["e1_" + n for n in MOE_NAMES] + ["e2_" + n for n in HID_NAMES] + ["e3_" + n for n in list(E3_COLS) + list(E3_BOOL)] + ["e4"]}
    beyond = {n: dict(ab=0, null=0, n=0) for n in E3_COLS}
    validity = dict(windows=len(windows), per_session={})
    for s in (A, B, A2):
        validity["per_session"][s.label] = dict(records=0, nmeas_bad=0, rank_mismatch=0, inv_bad=0, fw_errors=0)
    for wid in windows:
        cls = wid.split("-")[1]
        data = {}
        for s in (A, B, A2):
            for r in (1, 2):
                wd = WinData(s, wid, r)
                data[(s.label, r)] = wd
                v = validity["per_session"][s.label]
                if wd.ok:
                    v["records"] += 1
                    v["nmeas_bad"] += int((wd.nmeas != 1).sum())
                    v["inv_bad"] += int(wd.fw["inv"][1])
                    v["fw_errors"] = max(v["fw_errors"], int(wd.fw["errors"] or 0))
                    r1 = s.fw(wid, r, rank=1)
                    if r1 is None or not np.array_equal(np.nan_to_num(r1["m"][:W], nan=9.0), np.nan_to_num(wd.m, nan=9.0)):
                        v["rank_mismatch"] += 1
        for pname, (sx, rx, sy, ry, norep) in pairs.items():
            x, y = data[(sx.label, rx)], data[(sy.label, ry)]
            R = res[pname]
            for o in range(W):
                ok, why = matched(x, y, o, norep)
                if not ok:
                    excl[pname][why] = excl[pname].get(why, 0) + 1
                    continue
                for k, n in enumerate(MOE_NAMES):
                    a, b = x.moe(k, o), y.moe(k, o)
                    if a is None or b is None:
                        continue
                    R["e1"][n]["n"] += 1
                    R["e1"][n]["bitequal"] += int(np.array_equal(a, b))
                    R["e1"][n]["rel"].append(rel_l2(bf16_to_f32(b), bf16_to_f32(a)))
                for k, n in enumerate(HID_NAMES):
                    a, b = x.hid(k, o), y.hid(k, o)
                    if a is None or b is None:
                        continue
                    R["e2"][n]["n"] += 1
                    R["e2"][n]["bitequal"] += int(np.array_equal(a, b))
                    R["e2"][n]["rel"].append(rel_l2(bf16_to_f32(b), bf16_to_f32(a)))
                ex, ey = x.e3(o), y.e3(o)
                for i, n in enumerate(list(E3_COLS) + list(E3_BOOL)):
                    a, b = float(ex[i]), float(ey[i])
                    if a != a or b != b:
                        continue
                    e = R["e3"][n]
                    e["n"] += 1
                    e["equal"] += int(a == b)
                    if n in E3_COLS:
                        e["absd"].append(abs(b - a))
                        e["signed"].append(b - a)
                la, lb = x.logits(o), y.logits(o)
                if la is not None and lb is not None:
                    kab, kba, tv = kl_tv(la, lb)
                    d, _ = topk_restricted(la, lb)
                    R["e4"]["n"] += 1
                    R["e4"]["equal"] += int(np.array_equal(la, lb))
                    R["e4"]["kl_ab"].append(kab)
                    R["e4"]["kl_ba"].append(kba)
                    R["e4"]["tv"].append(tv)
                    R["e4"]["topk_diag"].append(d)
        # reproducible differences (four A-side measurements, two B-side)
        a1, a2, b1, b2 = data[("A", 1)], data[("A", 2)], data[("B", 1)], data[("B", 2)]
        c1, c2 = data[("A2", 1)], data[("A2", 2)]
        for o in range(W):
            okab1, _ = matched(a1, b1, o)
            okab2, _ = matched(a2, b2, o)
            okaa, _ = matched(a1, a2, o, True)
            okc1, _ = matched(a1, c1, o)
            okc2, _ = matched(a2, c2, o)
            if not (okab1 and okab2 and okaa and okc1 and okc2):
                continue

            def fam(name, get):
                vals = [get(w) for w in (a1, a2, c1, c2, b1, b2)]
                if any(v is None for v in vals):
                    return
                f = repro[name]
                f["compared"] += 1
                A_, A_b, C_, C_b, B_, B_b = vals
                stable = np.array_equal(A_, A_b) and np.array_equal(A_, C_) and np.array_equal(A_b, C_b)
                if stable:
                    f["stable"] += 1
                    if not np.array_equal(B_, A_) and not np.array_equal(B_b, A_b):
                        f["D"] += 1
                        f["D_windows"].add(wid)
                        f["D_by_class"][cls] = f["D_by_class"].get(cls, 0) + 1
            for k, n in enumerate(MOE_NAMES):
                fam("e1_" + n, lambda w, k=k: w.moe(k, o))
            for k, n in enumerate(HID_NAMES):
                fam("e2_" + n, lambda w, k=k: w.hid(k, o))
            for i, n in enumerate(list(E3_COLS) + list(E3_BOOL)):
                fam("e3_" + n, lambda w, i=i: (lambda v: None if v != v else np.float32(v))(float(w.e3(o)[i])))
            fam("e4", lambda w: w.logits(o))
            ea1, ea2, ec1, ec2, eb1, eb2 = (w.e3(o) for w in (a1, a2, c1, c2, b1, b2))
            for i, n in enumerate(E3_COLS):
                v = [float(e[i]) for e in (ea1, ea2, ec1, ec2, eb1, eb2)]
                if any(x != x for x in v):
                    continue
                spread_a = max(abs(v[0] - v[1]), abs(v[0] - v[2]), abs(v[1] - v[3]))
                ab = min(abs(v[4] - v[0]), abs(v[5] - v[1]))
                null = min(abs(v[2] - v[0]), abs(v[3] - v[1]))
                spread_null = max(abs(v[0] - v[1]), abs(v[4] - v[5]))
                beyond[n]["n"] += 1
                beyond[n]["ab"] += int(ab > spread_a)
                beyond[n]["null"] += int(null > spread_null)
    # summarize
    out = dict(windows=len(windows), validity=validity, exclusions=excl, comparisons={}, reproducible={},
               beyond_fallback=beyond)
    for p, R in res.items():
        out["comparisons"][p] = dict(
            e1={n: dict(n=v["n"], bitequal_frac=(v["bitequal"] / v["n"]) if v["n"] else None, rel_l2=quant(v["rel"]))
                for n, v in R["e1"].items()},
            e2={n: dict(n=v["n"], bitequal_frac=(v["bitequal"] / v["n"]) if v["n"] else None, rel_l2=quant(v["rel"]))
                for n, v in R["e2"].items()},
            e3={n: dict(n=v["n"], equal_frac=(v["equal"] / v["n"]) if v["n"] else None, absd=quant(v["absd"]),
                        mean_signed=(float(np.mean(v["signed"])) if v["signed"] else None)) for n, v in R["e3"].items()},
            e4=dict(n=R["e4"]["n"], equal=R["e4"]["equal"], kl_ab=quant(R["e4"]["kl_ab"]), kl_ba=quant(R["e4"]["kl_ba"]),
                    tv=quant(R["e4"]["tv"]), restricted_top20_tv_DIAGNOSTIC_ONLY=quant(R["e4"]["topk_diag"])))
    for k, f in repro.items():
        out["reproducible"][k] = dict(compared=f["compared"], a_stable=f["stable"], D=f["D"],
                                      D_windows=len(f["D_windows"]), D_by_class=f["D_by_class"])
    return out


def e5(sess, sched):
    out = {}
    for lab, s in sess.items():
        st = sorted(s.steps.values(), key=lambda d: d["step"])
        dt = []
        for a, b in zip(st, st[1:]):
            if b["step"] == a["step"] + 1 and a["num_reqs"] == 8 and a["n_prefilling"] == 0:
                dt.append(b["t"] - a["t"])
        modes = {}
        occ = {}
        for d in st:
            modes[d["graph_mode"]] = modes.get(d["graph_mode"], 0) + 1
            key = f"{d['num_reqs']}x{d['num_tokens']}"
            occ[key] = occ.get(key, 0) + 1
        out[lab] = dict(steps=len(st), step_time_s_8req_decode=quant(dt), graph_mode=modes,
                        occupancy_top=sorted(occ.items(), key=lambda kv: -kv[1])[:8])
    return out


def acceptance_check(sess, windows, sched):
    """Per request: accepted count of the k-th verify step == sched[k] (clamped at the window end)."""
    out = {}
    for lab, s in sess.items():
        bad = n = 0
        for wid in windows:
            for r in (1, 2):
                fw = s.fw(wid, r)
                if fw is None:
                    continue
                c = {x: i for i, x in enumerate(fw["cols"])}
                m = fw["m"][:W]
                per = {}
                for o in range(W):
                    if fw["nmeas"][o] == 1 and m[o, c["kind"]] == 0.0:
                        per[int(m[o, c["step"]])] = int(m[o, c["accepted"]])
                seq = [per[k] for k in sorted(per)]
                for k, a in enumerate(seq[:-1]):        # the last step clamps at the window end
                    n += 1
                    bad += int(a != sched[k % len(sched)])
        out[lab] = dict(steps_checked=n, not_as_scheduled=bad)
    return out


def decide(out, outcomes, g4_ok):
    gates = all(o.get("gates_ok") for o in outcomes.values()) and g4_ok
    if not gates:
        return "INCOMPLETE; repair; no reading (a launch-verification gate G1-G4 failed or a session is missing)"
    r = out["reproducible"]
    e1_D = sum(r["e1_" + n]["D"] for n in MOE_NAMES)
    e3_D = sum(v["D"] for k, v in r.items() if k.startswith("e3_"))
    stable_frac = min((v["a_stable"] / v["compared"]) if v["compared"] else 0.0 for k, v in r.items() if k.startswith("e3_"))
    fallback = stable_frac < 0.5
    e3_beyond = any(v["ab"] > v["null"] for v in out["beyond_fallback"].values())
    lines = []
    if (not fallback and e3_D > 0) or (fallback and e3_beyond):
        lines.append("E3 differs reproducibly (A != B where A = A, same positions, both repeats): computational "
                     "propagation established; behavioural direction untested")
        term = {k: v for k, v in r.items() if k[3:] in TERMINAL and v["D_by_class"].get("close")}
        if term:
            lines.append("E3 differences include terminal-token log-probabilities in near-close windows: reported "
                         "as such, still not a completion claim")
        if r["e3_argmax"]["D"]:
            lines.append("argmax changes reproducibly at %d positions: sensitivity only" % r["e3_argmax"]["D"])
    elif e1_D > 0:
        lines.append("E1 differs but E3 shows no A-vs-B difference beyond A-vs-A: no demonstrated propagation on "
                     "these windows; investigate attenuation before any behavioural run")
    else:
        lines.append("Neither E1 nor E3 shows a reproducible A-vs-B difference on these windows (the tile "
                     "intervention did not change the measured outputs; check G1 before reading further)")
    if fallback:
        lines.append(f"(fallback rule used: A is bit-stable on only {stable_frac:.2%} of compared positions)")
    return " | ".join(lines)


def to_md(doc):
    L = ["# P4 result (generated by analysis/analyze_p4.py; aggregates only)", "",
         f"**Reading:** {doc['reading']}", "", "## Sessions", ""]
    for k, v in doc["sessions"].items():
        L.append(f"- {k}: {v.get('rundir')} — {v.get('verdict')} (gates_ok {v.get('gates_ok')}, batches {v.get('batches_done')})")
    L += ["", f"G4: {doc.get('g4')}", "", "## Reproducible differences (A-stable positions where B differs in both repeats)", "",
          "| family | compared | A-stable | D | windows with D | D by window class |", "|---|---:|---:|---:|---:|---|"]
    for k, v in doc["result"]["reproducible"].items():
        L.append(f"| {k} | {v['compared']} | {v['a_stable']} | {v['D']} | {v['D_windows']} | {v['D_by_class']} |")
    L += ["", "## E1/E2 bit-equality and rel L2 (median / q90 / max)", "",
          "| pair | stage | n | bit-equal | rel L2 q50 | q90 | max |", "|---|---|---:|---:|---:|---:|---:|"]
    for p, c in doc["result"]["comparisons"].items():
        for fam in ("e1", "e2"):
            for n, v in c[fam].items():
                q = v["rel_l2"] or {}
                be = v["bitequal_frac"]
                L.append(f"| {p} | {n} | {v['n']} | {'' if be is None else f'{be:.4f}'} | {q.get('q50', '')} | "
                         f"{q.get('q90', '')} | {q.get('max', '')} |")
    L += ["", "## E3 (|delta|: median / q90 / max; equal fraction)", "",
          "| pair | metric | n | equal | q50 | q90 | max | mean B-A |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for p, c in doc["result"]["comparisons"].items():
        for n, v in c["e3"].items():
            q = v["absd"] or {}
            L.append(f"| {p} | {n} | {v['n']} | {'' if v['equal_frac'] is None else round(v['equal_frac'], 4)} | "
                     f"{q.get('q50', '')} | {q.get('q90', '')} | {q.get('max', '')} | {v.get('mean_signed')} |")
    L += ["", "## E4 full-vocabulary KL / TV (restricted top-20 distance: diagnostic only)", "",
          "| pair | n | KL(A‖B) q50 | q90 | TV q50 | q90 | top-20 diag q50 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for p, c in doc["result"]["comparisons"].items():
        e = c["e4"]
        g = lambda d, k: (d or {}).get(k, "")  # noqa: E731
        L.append(f"| {p} | {e['n']} | {g(e['kl_ab'], 'q50')} | {g(e['kl_ab'], 'q90')} | {g(e['tv'], 'q50')} | "
                 f"{g(e['tv'], 'q90')} | {g(e['restricted_top20_tv_DIAGNOSTIC_ONLY'], 'q50')} |")
    L += ["", "## Exclusions (positions)", "", "```", json.dumps(doc["result"]["exclusions"], indent=1), "```",
          "", "## E5", "", "```", json.dumps(doc["e5"], indent=1), json.dumps(doc["acceptance"], indent=1), "```",
          "", "## Validity", "", "```", json.dumps(doc["result"]["validity"], indent=1), "```"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--a2")
    ap.add_argument("--g4")
    ap.add_argument("--sched", default=None)
    a = ap.parse_args()
    dirs = dict(A=a.a or pick_attempt(a.root, "A"), B=a.b or pick_attempt(a.root, "B"), A2=a.a2 or pick_attempt(a.root, "A2"))
    sess = {k: Session(k, v) if v else Session(k, "") for k, v in dirs.items()}
    g4p = a.g4 or (sorted(glob.glob(os.path.join(a.root, "runs", "g4-attempt-*", "verdict.json"))) or [None])[-1]
    g4 = json.load(open(g4p)) if g4p and os.path.exists(g4p) else {}
    sched = json.load(open(a.sched or os.path.join(a.root, "PRIVATE-fixtures", "sched.json")))["sched"]
    os.makedirs(a.out, exist_ok=True)
    if not all(dirs.values()):
        doc = dict(reading="INCOMPLETE; no reading (session(s) missing: %s)" % [k for k, v in dirs.items() if not v],
                   sessions={k: dict(rundir=v) for k, v in dirs.items()})
        json.dump(doc, open(os.path.join(a.out, "result.json"), "w"), indent=1)
        open(os.path.join(a.out, "result.md"), "w").write("# P4 result\n\n**Reading:** %s\n" % doc["reading"])
        print(doc["reading"])
        return 1
    res = analyze(a.root, sess, g4)
    windows = []
    for d in sess["A"].client:
        if d.get("kind") == "batch":
            windows += [w for w in d["windows"] if w not in windows]
    doc = dict(result=res, e5=e5(sess, sched), acceptance=acceptance_check(sess, windows, sched), g4=g4.get("G4"),
               sessions={k: dict(rundir=os.path.relpath(v, a.root), verdict=sess[k].outcome.get("verdict"),
                                 gates_ok=sess[k].outcome.get("gates_ok"), batches_done=sess[k].outcome.get("batches_done"))
                         for k, v in dirs.items()})
    doc["reading"] = decide(res, {k: s.outcome for k, s in sess.items()}, g4.get("G4") == "PASS")
    json.dump(doc, open(os.path.join(a.out, "result.json"), "w"), indent=1, default=str)
    open(os.path.join(a.out, "result.md"), "w").write(to_md(doc))
    print("READING:", doc["reading"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
