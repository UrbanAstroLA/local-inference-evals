#!/usr/bin/env python3
"""D7 exposure accounting (PLAN.md amendment 18), steps 3-4: GPQA exposure per request and the descriptive table.

LOCAL ONLY while running (reads lm-eval samples to recover empty flags and message hashes); WRITES doc ids, seeds,
hashes, token counts, times, classes and flags only -- never question, answer or reply text.

Inputs (read-only)
  gpqa_prompt_tokens.json        exact rendered prompt-token counts (render_tokens.py; 1,980/1,980 logged requests equal)
  <gpqa-runs>/*requests.jsonl  record-proxy logs: t_start, secs, seed, messages_sha256, prompt_tokens
  <gpqa-runs>/<recipe>__passN/**/samples_*.jsonl   empty = reply text empty after strip (analysis_col511.py:219)

Engine chunking (sim_d7.Rules; scheduler arithmetic verified on attempt 3: 14,712/14,712 rows, 217/217 events)
  0.9.1 family (stock record, col511 arm, bundle): K = 3 -> solo budget 2045, decode row 4 + 3 = 7, decode path n <= 4
  0.7.0 record: K = 5 -> solo budget 2043, decode row 6 + 5 = 11, decode path n <= 6. CONDITIONAL: whether 0.7.0's writer
      shares the aligned-start assumption is checked elsewhere; 0.7.0 runs DCP2 (positions as logical prompt positions here)
  GPQA prompts are < 4,352 tokens: the Mamba clip never applies; no prefix hits (P0/P1 repeats in attempt 3: cached 0)

Per logged pass
  ENSEMBLE (ESTIMATED): time-driven simulation over plausible schedules. Exogenous, recorded: arrival order and proxy
  send time, request lifetime (t_start + secs), prompt length. Unrecorded, varied: frontend delay to the engine,
  engine-finish lead before the proxy end, step duration (scale x the record's mean decode step), step-grid phase.
  ANALYTIC (bounds): for each request, the rows that could precede it in its admission step:
      S = requests sent within W seconds before/after it (could be co-admitted ahead), plus doc 127's remainder chunk
          if doc 127 was admitted within W before it;
      decode rows d in [d_lo, d_hi]: others in flight at its send time not in S; those ending within W of it may
          already have left (d_lo) or still be scheduled (d_hi).
      NOT EXPOSED  the prompt fits the smallest plausible budget (all of S ahead, d = d_hi): it cannot be split.
      KNOWN        every plausible configuration (each subset of S ahead x each d) gives an exposed chunk start.
      POSSIBLE     some plausible configuration (or some ensemble schedule) gives one.
  Pass without request log (0.7.0 pass 1): UNKNOWN unless the dispatch-order bound (up to 7 neighbours in doc order
  co-admitted ahead) shows the prompt cannot be split -> NOT EXPOSED (bound).
"""
import collections
import glob
import hashlib
import itertools
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim_d7 import Rules, mamba_clip  # noqa: E402

G = "<gpqa-runs>"
CONTRIB = "<this-repo>"
W = 1.0                    # s, co-admission / transition window for the analytic bounds (generous: attempt 3's largest
                           # frontend spread was 0.34 s; a doc-127 prefill step about 0.6 s)
REC = [
    dict(record="stock 0.7.0 (DFlash2 x5, DCP2)", engine="0.7.0", passes=[
        dict(p=1, seed=1234, dir="glm53-flash-exl3-k3.25@tpurtell__pass1", log=None),
        dict(p=2, seed=1235, dir="glm53-flash-exl3-k3.25@tpurtell+ps__pass2", log="glm53-flash-exl3-k3.25@tpurtell+ps.requests.jsonl"),
        dict(p=3, seed=1236, dir="glm53-flash-exl3-k3.25@tpurtell+ps__pass3", log="glm53-flash-exl3-k3.25@tpurtell+ps.requests.jsonl")],
         published=f"{CONTRIB}/runs/2026-09-29_glm53-flash_k3.25-v0.7.0-dflash5_gpqa-diamond/results.jsonl"),
    dict(record="stock 0.9.1 (DFlash2 x3)", engine="0.9.1", passes=[
        dict(p=1, seed=1234, dir="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps__pass1", log="glm53-flash-exl3-k3.25@tpurtell+v0.9.1.requests.jsonl"),
        dict(p=2, seed=1235, dir="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps__pass2", log="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.requests.jsonl"),
        dict(p=3, seed=1236, dir="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps__pass3", log="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.requests.jsonl")],
         published=f"{CONTRIB}/runs/2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3_gpqa-diamond/results.jsonl"),
    dict(record="col511 arm (0.9.1 + column-511 correction)", engine="0.9.1", passes=[
        dict(p=1, seed=1234, dir="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix__pass1", log="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix.s1.requests.jsonl"),
        dict(p=2, seed=1235, dir="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix__pass2", log="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix.s2.requests.jsonl"),
        dict(p=3, seed=1236, dir="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix__pass3", log="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix.s2.requests.jsonl")],
         published=None),
    dict(record="bundle D1-D4 = B0 (0.9.1 + D1-D4)", engine="0.9.1", passes=[
        dict(p=1, seed=1234, dir="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234__pass1", log="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234.s1.requests.jsonl"),
        dict(p=2, seed=1235, dir="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234__pass2", log="glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234.s2.requests.jsonl")],
         published=None),
]
ENGINE = {"0.9.1": dict(K=3, step=451.5 / (8 * 2.845)), "0.7.0": dict(K=5, step=271.5 / (8 * 3.071))}
for e in ENGINE.values():          # mean decode-step duration at 8 running (record server-log means; ESTIMATE)
    e["dt"] = 1.0 / e.pop("step")
B_PER_PREFILL_TOKEN = 0.000274     # s/token, attempt-3 step-time fit (sim_a3_results.json), scaled with the step


def messages_hash(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def load_samples(d):
    fs = glob.glob(os.path.join(G, d, "**", "samples_*.jsonl"), recursive=True)
    assert len(fs) == 1, (d, fs)
    out = {}
    for line in open(fs[0]):
        r = json.loads(line)
        if r.get("filter") != "flexible-extract":
            continue
        txt = str(r["resps"][0][0]) if r["resps"] else ""
        out[r["doc_id"]] = dict(mh=messages_hash(json.loads(r["arguments"]["gen_args_0"]["arg_0"][0])), empty=not txt.strip())
    assert sorted(out) == list(range(198)), d
    return out, hashlib.sha256(open(fs[0], "rb").read()).hexdigest()


# ------------------------------------------------------------------ time-driven ensemble simulation
def simulate(reqs, K, dt, b, delay, lead, phase, rng=None, pool=4, adm=None):
    """reqs: list of dict(doc, L, t, end) in send order. Returns {doc: [(start, n), ...]} prefill chunks."""
    R = Rules(K=K)
    arr = []
    for i, q in enumerate(reqs):
        dl = delay if rng is None else rng.uniform(*delay)
        arr.append([q["t"] + dl, i, q])
    arr.sort(key=lambda x: (x[0], x[1]))
    chunks = collections.defaultdict(list)
    running = []        # dicts: q, computed, decode(bool), dep (engine-finish time)
    waiting = []
    i = 0
    t = arr[0][0] + phase * dt
    while i < len(arr) or waiting or running:
        while i < len(arr) and arr[i][0] <= t:
            waiting.append(arr[i])
            i += 1
        running = [r for r in running if not (r["decode"] and r["dep"] <= t)]
        tb, ib = R.max_sched, R.max_batched
        pf_tokens = 0
        nrows = 0
        ndec = npf = 0
        for r in running:
            if tb <= 0 or ib <= K:
                break
            n = (1 + K) if r["decode"] else r["q"]["L"] - r["computed"]
            n = min(n, tb, ib - K)
            if not r["decode"]:
                n = mamba_clip(r["computed"], n, r["q"]["L"], R)
            if n <= 0:
                continue
            nrows += 1
            ndec += 1 if r["decode"] else 0
            npf += 0 if r["decode"] else 1
            if not r["decode"]:
                chunks[r["q"]["doc"]].append((r["computed"], n))
                r["computed"] += n
                pf_tokens += n
                if r["computed"] >= r["q"]["L"]:
                    r["next_decode"] = True
            tb -= n
            ib -= n + K
        while waiting and tb > 0 and ib > K and len(running) < R.max_running:
            a, idx, q = waiting[0]
            n = min(q["L"], tb, ib - K)
            n = mamba_clip(0, n, q["L"], R)
            if n <= 0:
                break
            waiting.pop(0)
            if adm is not None:
                adm[q["doc"]] = dict(decode_rows_ahead=ndec, prefill_rows_ahead=npf,
                                     input_budget_left=ib - K, first_chunk=n, L=q["L"])
            r = dict(q=q, computed=n, decode=False, dep=q["end"] - lead)
            chunks[q["doc"]].append((0, n))
            if n >= q["L"]:
                r["next_decode"] = True
            running.append(r)
            nrows += 1
            npf += 1
            pf_tokens += n
            tb -= n
            ib -= n + K
        for r in running:
            if r.pop("next_decode", False):
                r["decode"] = True
        step = dt + b * pf_tokens
        busy = waiting or any(not r["decode"] for r in running)
        if busy or i >= len(arr):
            t += step
            if not busy and i >= len(arr):
                # only decodes left: jump to the last departure
                if running:
                    t = max(t, max(r["dep"] for r in running) + dt)
                    running = []
        else:
            # nothing to prefill: continue the decode grid up to the next arrival
            nxt = arr[i][0]
            if nxt > t:
                k = int((nxt - t) / dt) + 1
                t += k * dt
            else:
                t += step
    return chunks


def exposed_chunks(chs, K, pool=4):
    return [(s, n) for s, n in chs if s % pool and n > 1 + K]


def ensemble(reqs, K, dt0):
    scales = (0.5, 0.75, 1.0, 1.5, 2.0)
    phases = (0.0, 0.25, 0.5, 0.75)
    delays = (("fixed", 0.005), ("fixed", 0.03), ("fixed", 0.08), ("random", (0.005, 0.35)))
    leads = (0.0, 0.1)
    hits = collections.Counter()
    nevents = []
    n = 0
    detail = collections.defaultdict(collections.Counter)
    for sc, ph, (dm, dv), ld in itertools.product(scales, phases, delays, leads):
        rng = random.Random(hash((sc, ph, ld)) & 0xffffffff) if dm == "random" else None
        ch = simulate(reqs, K, dt0 * sc, B_PER_PREFILL_TOKEN * sc, dv, ld, ph, rng)
        n += 1
        tot = 0
        for doc, cs in ch.items():
            ex = exposed_chunks(cs, K)
            if ex:
                hits[doc] += 1
                tot += len(ex)
                detail[doc][tuple(ex)] += 1
        nevents.append(tot)
    return dict(schedules=n, exposed_fraction={d: hits[d] / n for d in hits}, events_min=min(nevents), events_max=max(nevents),
                events_mean=sum(nevents) / n, chunk_patterns={d: [[list(map(list, k)), v] for k, v in c.most_common(4)] for d, c in detail.items()})


# ------------------------------------------------------------------ analytic bounds
def admit_chunk(L, running_tokens, waiting_ahead, K, max_batched=2048):
    """Scheduler arithmetic for one step (sim_d7 / scheduler.py): running rows first (decode rows 1 + K tokens, a
    running prefill its remaining tokens), then waiting rows ahead in FIFO order, each taking min(len, budget - K) and
    reserving K. Returns the first chunk the request would get, or None if it is not admitted in this step (FIFO
    stops at the first request that cannot be admitted, so a request behind it waits too). Prompts < 4,352: no clip."""
    ib = max_batched
    for tok in running_tokens:
        if ib <= K:
            break
        n = min(tok, ib - K)
        ib -= n + K
    for Lo in waiting_ahead:
        if ib <= K:
            return None
        n = min(Lo, ib - K)
        ib -= n + K
    if ib <= K:
        return None
    return min(L, ib - K)


def classify_cfgs(L, cfgs, K):
    """cfgs: iterable of (running_tokens, waiting_ahead). Returns (class, first chunks seen when split)."""
    split_exposed, firsts, admitted_any = [], set(), False
    for run, wait in cfgs:
        c = admit_chunk(L, run, wait, K)
        if c is None:
            continue
        admitted_any = True
        if c >= L:
            split_exposed.append(None)
            continue
        firsts.add(c)
        split_exposed.append(c % 4 != 0 and L - c > 1 + K)
    splits = [x for x in split_exposed if x is not None]
    if not splits:
        cls = "NOT-unsplittable"
    elif len(splits) == len(split_exposed) and all(splits):
        cls = "KNOWN"
    elif any(splits):
        cls = "POSSIBLE"
    else:
        cls = "NOT-aligned-split"
    return cls, sorted(firsts), admitted_any


def big_remainders(Lbig, K, dmax=7):
    """Remaining tokens of a prompt longer than one step's budget after its first chunk, over plausible first chunks
    (admitted after 0..dmax decode rows; or behind up to two co-admitted prompts, covered by the caller's S)."""
    return sorted({Lbig - min(Lbig, 2048 - K - (1 + 2 * K) * d) for d in range(dmax + 1)} - {0})


def analytic(reqs, K):
    """Per request, over every plausible configuration of the rows that precede it in its admission step:
       S   requests sent within W of it: each is, at its admission step, a waiting prefill row ahead (it arrived first),
           or -- if sent earlier -- already a decode row, or not ahead at all;
       d   others in flight at its send time and not in S: decode rows; those ending within W may already have left;
       a prompt longer than one step's budget sent within W before it may still have its remainder running ahead."""
    out = {}
    Lmax_step = 2048 - K - (1 + 2 * K) * 7 - 2 * K
    for q in reqs:
        t = q["t"]
        S = [o for o in reqs if o is not q and abs(o["t"] - t) <= W]
        inflight = [o for o in reqs if o is not q and o["t"] < t - W and o["end"] > t]
        sure = [o for o in inflight if o["end"] > t + W]
        maybe_gone = [o for o in inflight if o["end"] <= t + W]
        rem_opts = [0]
        for o in reqs:
            if o is not q and o["L"] > Lmax_step and t - 2 * W <= o["t"] < t:
                rem_opts += big_remainders(o["L"], K)
        cfgs = []
        states = [("ahead", "decode", "none") if o["t"] < t else ("ahead", "none") for o in S]
        for combo in (itertools.product(*states) if states else [()]):
            ahead = [o for o, st in zip(S, combo) if st == "ahead"]
            ndec_S = sum(1 for st in combo if st == "decode")
            for d in range(len(sure), len(sure) + len(maybe_gone) + 1):
                for r in rem_opts:
                    run = [1 + K] * (d + ndec_S) + ([r] if r else [])
                    for perm in (itertools.permutations(ahead) if len(ahead) <= 3 else [ahead]):
                        cfgs.append((run, [o["L"] for o in perm]))
        cls, firsts, adm = classify_cfgs(q["L"], cfgs, K)
        out[q["doc"]] = dict(cls=cls, n_S=len(S), d_range=[len(sure), len(sure) + len(maybe_gone)],
                             big_remainder_options=rem_opts[1:], first_chunks=firsts[:12], n_configs=len(cfgs))
    return out


def analytic_unlogged(Ls, K, m_plausible):
    """No admission times. Dispatch is in doc order (at most 4 adjacent inversions per logged pass); at most 8 in
    flight, so at most 7 others share a request's admission step (running decode rows, or waiting rows ahead).
      conservative: any number (0..7) of the 14 nearest neighbours in dispatch order co-admitted ahead -- bounded
                    by the worst case: the longest ones, longest first, rest decode rows; or a long prompt's remainder
      plausible:    at most m_plausible co-admitted ahead (the largest mid-pass co-send count in the logged passes)
      pass start:   docs 0-7 are sent together (every logged pass), so co-admission among them is enumerated exactly.
    Requests unsplittable under the conservative bound are NOT-unsplittable; the rest UNKNOWN (timing missing), with
    the plausible-bound outcome and, for prompts longer than one step's budget, the first chunk per decode-row count."""
    out = {}
    Lmax_step = 2048 - K - (1 + 2 * K) * 7 - 2 * K
    bigs = [j for j in range(198) if Ls[j] > Lmax_step]
    for d in range(198):
        nb = [j for j in range(max(0, d - 7), min(198, d + 8)) if j != d]
        rem_opts = [0] + [r for j in bigs if j in nb or j == d - 8 for r in big_remainders(Ls[j], K)]
        def cfgs_for(m):
            for k in range(0, m + 1):
                for sub in itertools.combinations(nb, k):
                    for perm in (itertools.permutations(sub) if k <= 2 else [tuple(sorted(sub, key=lambda j: -Ls[j]))]):
                        for r in rem_opts:
                            run = [1 + K] * (7 - k - (1 if r else 0)) + ([r] if r else [])
                            yield run, [Ls[j] for j in perm]
        if d < 8:
            first8 = [j for j in range(8) if j != d]
            cf = [([], [Ls[j] for j in perm]) for k in range(0, 8) for perm in itertools.permutations(first8, k)] if False else \
                 [([1 + K] * (7 - k), [Ls[j] for j in sorted(sub, key=lambda j: -Ls[j])]) for k in range(0, 8)
                  for sub in itertools.combinations(first8, k)]
            cls_c, firsts_c, _ = classify_cfgs(Ls[d], cf, K)
            cls_p = cls_c
        else:
            # conservative worst case: the 7 longest neighbours ahead, longest first (most budget consumed), any count
            nbs = sorted(nb, key=lambda j: -Ls[j])
            cf = [([1 + K] * (7 - k), [Ls[j] for j in nbs[:k]]) for k in range(0, 8)]
            cf += [([1 + K] * (6 - k) + [r], [Ls[j] for j in nbs[:k]]) for k in range(0, 7) for r in rem_opts if r]
            cls_c, firsts_c, _ = classify_cfgs(Ls[d], cf, K)
            cls_p, _, _ = classify_cfgs(Ls[d], list(cfgs_for(m_plausible)), K)
        if cls_c == "NOT-unsplittable":
            out[d] = dict(cls="NOT-unsplittable", bound="conservative dispatch-order bound")
        else:
            out[d] = dict(cls="UNKNOWN", bound="no admission times", conservative_bound=cls_c, plausible_bound=cls_p,
                          solo_first_chunk_by_decode_rows={dd: 2048 - K - (1 + 2 * K) * dd for dd in range(8)}
                          if Ls[d] > Lmax_step else None)
    return out


def cosend_stats():
    """Largest number of OTHER requests sent within W of a request, mid-pass (excluding the 8-request pass start),
    over every logged pass (MEASURED from the proxy logs)."""
    mx, hist = 0, collections.Counter()
    for rec in REC:
        for P in rec["passes"]:
            if not P["log"]:
                continue
            ts = sorted(json.loads(l)["t_start"] for l in open(os.path.join(G, P["log"])) if l.strip() and json.loads(l).get("seed") == P["seed"])
            for i, t in enumerate(ts[8:], 8):
                k = sum(1 for u in ts[8:] if u != t and abs(u - t) <= W)
                hist[k] += 1
                mx = max(mx, k)
    return mx, dict(sorted(hist.items()))


M_PLAUSIBLE = None


def main():
    global M_PLAUSIBLE
    M_PLAUSIBLE, cohist = cosend_stats()
    print(f"co-sends within {W} s (mid-pass, all logged passes): max {M_PLAUSIBLE}, histogram {cohist}")
    tok = json.load(open(os.path.join(HERE, "gpqa_prompt_tokens.json")))
    Ls = {r["doc_id"]: r["B_zai_0490d2f7"]["tokens"] for r in tok["docs"]}
    LsA = {r["doc_id"]: r["A_ckpt_701cd745"]["tokens"] for r in tok["docs"]}
    assert Ls == LsA
    h2d = {r["messages_sha256"]: r["doc_id"] for r in tok["docs"]}
    results = []
    per_req = []
    checks = []
    for rec in REC:
        K = ENGINE[rec["engine"]]["K"]
        dt0 = ENGINE[rec["engine"]]["dt"]
        pub = None
        if rec["published"]:
            pub = {(r["doc_id"], r["pass"]): r for r in (json.loads(l) for l in open(rec["published"]) if l.strip())}
        for P in rec["passes"]:
            smp, smp_sha = load_samples(P["dir"])
            if pub:
                bad = sum(1 for d in range(198) if pub[(d, P["p"])]["empty"] != smp[d]["empty"])
                checks.append(f"{rec['record']} pass {P['p']}: empty flags vs published rows: {198 - bad}/198 equal")
            reqs = None
            if P["log"]:
                logs = [json.loads(l) for l in open(os.path.join(G, P["log"])) if l.strip()]
                logs = [r for r in logs if r.get("seed") == P["seed"]]
                assert len(logs) == 198, (P, len(logs))
                reqs = []
                for r in sorted(logs, key=lambda r: r["t_start"]):
                    d = h2d[r["messages_sha256"]]
                    assert r["prompt_tokens"] == Ls[d]
                    assert smp[d]["mh"] == r["messages_sha256"]
                    reqs.append(dict(doc=d, L=Ls[d], t=r["t_start"], end=r["t_start"] + r["secs"],
                                     finish=r.get("finish_reason"), ctoks=r.get("completion_tokens")))
                order = [q["doc"] for q in reqs]
                inv = sum(1 for a, b in zip(order, order[1:]) if b < a)
                checks.append(f"{rec['record']} pass {P['p']}: dispatch order inversions vs doc order: {inv}")
                ens = ensemble(reqs, K, dt0)
                central_adm = {}
                simulate(reqs, K, dt0, B_PER_PREFILL_TOKEN, 0.03, 0.0, 0.5, adm=central_adm)
                ana = analytic(reqs, K)
                t0 = reqs[0]["t"]
            else:
                ens = None
                central_adm = {}
                ana = analytic_unlogged(Ls, K, M_PLAUSIBLE)
            cls_count = collections.Counter()
            table = collections.Counter()
            for d in range(198):
                a = ana[d]
                cls = a["cls"]
                fr = (ens or {}).get("exposed_fraction", {}).get(d, 0.0) if ens else None
                if ens is not None and cls.startswith("NOT") and fr > 0:
                    cls = "POSSIBLE"           # the ensemble found a schedule the bound missed (should not happen)
                if ens is not None and cls == "KNOWN" and fr < 1:
                    cls = "POSSIBLE"
                cls_count[cls] += 1
                table[(cls, smp[d]["empty"])] += 1
                q = next((x for x in reqs if x["doc"] == d), None) if reqs else None
                per_req.append(dict(record=rec["record"], engine=rec["engine"], **{"pass": P["p"]}, seed=P["seed"], doc_id=d,
                                    prompt_tokens=Ls[d], empty=smp[d]["empty"], exposure=cls,
                                    ensemble_exposed_fraction=None if ens is None else round(fr, 4),
                                    analytic=a, central_schedule_admission=central_adm.get(d), send_t_rel=None if q is None else round(q["t"] - t0, 3),
                                    secs=None if q is None else round(q["end"] - q["t"], 3),
                                    finish_reason=None if q is None else q["finish"],
                                    completion_tokens=None if q is None else q["ctoks"]))
            results.append(dict(record=rec["record"], engine=rec["engine"], **{"pass": P["p"]}, seed=P["seed"],
                                logged=bool(P["log"]), samples_sha256=smp_sha,
                                empties=sum(smp[d]["empty"] for d in range(198)), classes=dict(cls_count),
                                class_x_empty={f"{c}|{'empty' if e else 'answered'}": v for (c, e), v in sorted(table.items())},
                                ensemble=None if ens is None else {k: ens[k] for k in ("schedules", "events_min", "events_max", "events_mean")},
                                ensemble_exposed={str(d): round(f, 4) for d, f in sorted((ens or {}).get("exposed_fraction", {}).items())},
                                ensemble_chunk_patterns=(ens or {}).get("chunk_patterns")))
            r = results[-1]
            print(f"{rec['record']:45s} p{P['p']} seed {P['seed']} logged={r['logged']!s:5s} E={r['empties']:2d} "
                  f"classes={r['classes']} ens_events={r['ensemble'] and (r['ensemble']['events_min'], r['ensemble']['events_max'])} "
                  f"exposed_docs={r['ensemble_exposed']}")
    for c in checks:
        print("check:", c)
    json.dump(dict(window_s=W, engines=ENGINE, cosend_max=M_PLAUSIBLE, cosend_hist=cohist, results=results, checks=checks), open(os.path.join(HERE, "gpqa_exposure.json"), "w"), indent=1)
    with open(os.path.join(HERE, "gpqa_exposure_per_request.jsonl"), "w") as f:
        for r in per_req:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
