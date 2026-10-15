"""F2 [R-2]: cache-safety gates and construction accounting from the session's own records (client.jsonl and the
guard's EngineCore scheduler log mapguard/ms_p*.jsonl). Pure python (json only); used by gate_fcr.py (GATE-1R),
analyze_fcr.py (whole session) and the client's per-batch quick check.

Per request (every build, touch, reference, run, CV and sentinel request):
  cached            the API's cached_tokens (= the scheduler's num_computed before the first chunk; checked)
  expected          the F1 rule value from the plan (fcr_geometry.expected_cache)
  class             cold (expected 0, cached 0) | hit (cached == expected) | partial (0 < cached < expected) |
                    miss (cached 0 < expected) | OVER (cached > expected)
Exact safety (structural, gated):
  S1 block-aligned        cached % 4352 == 0
  S2 permissible prefix   cached <= expected (the prefill coverage of the stream under the salt; never beyond)
  S3 no decode history    every restored MLA/index-K block (group 0, index < cached / 4352) is a block that an
                          earlier request of the same fixture and salt held at the same index INSIDE its prompt
                          (prefill-written), and never a block it held at an index at or past its decode start
  S4 attribution          the scheduler record of the request shows num_computed == cached before its first chunk,
                          and its restored block ids are found as in S3 (request -> earlier writer)
Construction accounting (reported; balance gated post-session): restore source (which earlier request wrote the
restored blocks), shared-build identity across users, class counts per run and per arm.
"""
import collections
import glob
import json
import os

BLOCK = 4352


def jl(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def load_sched(run_dir):
    """All EngineCore scheduler records ('sched' with num_computed / scheduled / block ids) in file order."""
    recs = []
    for p in sorted(glob.glob(os.path.join(run_dir, "mapguard", "ms_p*.jsonl"))):
        recs += [r for r in jl(p) if r.get("kind") == "sched"]
    recs.sort(key=lambda r: (r.get("t", 0), r.get("seq", 0)))
    return recs


def sched_requests(recs):
    """engine request id -> dict(first_seq, first_t, computed_before, g0 (latest full group-0 list), g0_first (list at
    the first scheduled step), P_rec (num_computed when decode began or the last value), fin_seq)."""
    info = collections.OrderedDict()
    for r in recs:
        for q, v in r["reqs"].items():
            d = info.get(q)
            if d is None:
                d = info[q] = dict(first_seq=r["seq"], first_t=r.get("t"), computed_before=v[0] - v[1], g0=None,
                                   g0_first=None, last=v[0], first_dec=None)
            d["last"] = v[0]
            if v[2] > 0 and d["first_dec"] is None:
                d["first_dec"] = v[0] - v[1]
                d["first_dec_seq"] = r["seq"]
            ch = r.get("changed", {}).get(q)
            if ch:
                d["g0"] = ch[0]
                if d["g0_first"] is None:
                    d["g0_first"] = ch[0]
        for q in r.get("finished", []):
            if q in info:
                info[q]["fin_seq"] = r["seq"]
    for d in info.values():
        d["P_rec"] = d["first_dec"] if d["first_dec"] is not None else d["last"]
    return info


def map_client(reqs, info, tags_from_hook=None):
    """client request tag -> engine request id. Order of evidence: (1) the response id recorded by the retry client
    (engine ids are '<response id>-<choice>-<suffix>'); (2) the hook's add records (keyed requests); (3) prompt length
    within the request's [t0, t1] window, ties broken by completion order."""
    out = {}
    used = set()
    hook = tags_from_hook or {}
    by_resp = collections.defaultdict(list)
    for q in info:
        by_resp[q.rsplit("-", 2)[0] if q.count("-") >= 3 else q].append(q)
    for r in reqs:
        rid = r.get("resp_id")
        if rid and len(by_resp.get(rid, [])) == 1:
            out[r["tag"]] = by_resp[rid][0]
            used.add(out[r["tag"]])
    for q, tag in hook.items():
        if tag not in out and q in info and q not in used:
            out[tag] = q
            used.add(q)
    rest = sorted([r for r in reqs if r["tag"] not in out and r.get("t0") is not None], key=lambda r: r["t1"])
    for r in rest:
        cand = [q for q, d in info.items() if q not in used and d["first_t"] is not None and
                r["t0"] - 1.0 <= d["first_t"] <= r["t1"] + 1.0 and d["P_rec"] in (r["P"], r["P"] + 1)]
        if cand:
            q = min(cand, key=lambda q: (info[q].get("fin_seq") or 1 << 40, info[q]["first_seq"]))
            out[r["tag"]] = q
            used.add(q)
    return out


def classify(cached, expected):
    if cached is None:
        return "missing"
    if expected == 0:
        return "cold" if cached == 0 else "OVER"
    if cached == expected:
        return "hit"
    if cached == 0:
        return "miss"
    return "partial" if cached < expected else "OVER"


def evaluate(reqs, info, mapping):
    """Per-request rows in client order (t0). Each row carries S1-S4 and the restore source."""
    rows = []
    history = collections.defaultdict(list)       # (salt, fixture) -> [(tag, P, decode_start or None, g0 list)]
    for r in sorted([x for x in reqs if x.get("t0") is not None], key=lambda x: (x["t0"], x["tag"])):
        key = (r.get("salt"), r.get("fixture"))
        cached, exp = r.get("cached_tokens"), r.get("cached_expected", 0)
        q = mapping.get(r["tag"])
        d = info.get(q) if q else None
        row = dict(tag=r["tag"], phase=r["phase"], fixture=r.get("fixture"), salt=r.get("salt_name"), P=r["P"],
                   cached=cached, expected=exp, cls=classify(cached, exp), engine_id=q)
        row["S1_aligned"] = cached is not None and cached % BLOCK == 0
        row["S2_within_bound"] = cached is not None and cached <= exp
        nb = (cached or 0) // BLOCK
        src, s3, s4 = [], True, True
        if d is None:
            s4 = False
            row["S4_note"] = "no scheduler record mapped"
        else:
            if d["computed_before"] != cached:
                s4 = False
                row["S4_note"] = "scheduler num_computed %s != cached %s" % (d["computed_before"], cached)
            g0 = d["g0_first"] or []
            for i in range(nb):
                if i >= len(g0):
                    s3 = False
                    src.append((i, None, "no id"))
                    continue
                who = None
                for (tag2, P2, ds2, g2) in reversed(history[key]):
                    if g2 and i < len(g2) and g2[i] == g0[i]:
                        lim = ds2 if ds2 is not None else P2
                        if (i + 1) * BLOCK <= min(P2, lim):
                            who = (tag2, "prefill")
                        else:
                            who = (tag2, "decode")
                        break
                if who is None or who[1] != "prefill":
                    s3 = False
                src.append((i, g0[i], who))
        row["S3_prefill_only"] = s3
        row["S4_attributed"] = s4 and all(w is not None for _, _, w in src)
        row["restore_source"] = sorted({w[0] for _, _, w in src if w})
        row["restored_ids"] = [b for _, b, _ in src]
        row["g0_final"] = list((d or {}).get("g0") or [])
        row["safe"] = row["S1_aligned"] and row["S2_within_bound"] and s3 and row["S4_attributed"]
        rows.append(row)
        ds = r["P"] if (r.get("W") or 0) > 0 else None
        history[key].append((r["tag"], r["P"], ds, (d or {}).get("g0") or []))
    return rows


def run_summary(rows, phases):
    out = {}
    for ph in phases:
        rs = [x for x in rows if x["phase"] == ph]
        out[ph] = dict(n=len(rs), classes=dict(collections.Counter(x["cls"] for x in rs)),
                       unsafe=[x["tag"] for x in rs if not x["safe"]])
    return out


def balance(rows, arms):
    """arms: {arm: [phases]}. Non-hit fixture-runs per arm (partial + miss + OVER); gated post-session: equal counts
    across arms (cache behaviour must not differ systematically by treatment); any non-hit = construction-mismatch
    flag on that fixture (it is analysed in the construction-matched sensitivity)."""
    per = {}
    fx = set()
    for arm, phs in arms.items():
        nh = [(x["phase"], x["fixture"], x["cls"]) for x in rows if x["phase"] in phs and x["cls"] != "hit"]
        per[arm] = dict(non_hits=len(nh), list=nh)
        fx |= {f for _, f, _ in nh}
    counts = [v["non_hits"] for v in per.values()]
    return dict(per_arm=per, balanced=len(set(counts)) <= 1, mismatch_fixtures=sorted(fx))


def identity(rows, groups):
    """groups: list of phase lists that share a build; per fixture, the restored id lists of every user must agree on
    their common length (block identity)."""
    out = {}
    for grp in groups:
        for n in sorted({x["fixture"] for x in rows if x["phase"] in grp}):
            ls = [(x["phase"], x["restored_ids"]) for x in rows if x["phase"] in grp and x["fixture"] == n]
            if len(ls) < 2:
                continue
            m = min(len(v) for _, v in ls)
            out["%s:%s" % ("+".join(grp), n)] = dict(identical=all(v[:m] == ls[0][1][:m] for _, v in ls), blocks=m)
    return out
