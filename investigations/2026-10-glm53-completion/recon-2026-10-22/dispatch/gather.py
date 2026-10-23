#!/usr/bin/env python3
"""Recon dispatch (2026-10-22): observed per-session facts from server and request logs. CPU only, read-only inputs.

Observed here (from logs): capture sizes, KV pool, graph mode, draft acceptance per position (aggregate and by the
engine's own 'Running' count), occupancy shares, and per-request timing. One derived quantity is labelled INFERRED:
the time window a request spends past 100k tokens assumes a constant decode rate within the request.

Writes only ./observed.json
"""
import bisect, json, os, re, statistics

INV = "<runs>"
G = "<gpqa-runs>"
S = [  # label, engine, K, workload, server log, request logs
    ("0.7.0 record p2-3", "0.7.0", 5, "gpqa", f"{G}/glm53-flash-exl3-k3.25@tpurtell+ps.server.log", [f"{G}/glm53-flash-exl3-k3.25@tpurtell+ps.requests.jsonl"]),
    ("0.7.0 rep070 s1", "0.7.0", 5, "gpqa", f"{INV}/rep070/run-F/server-s1.log", [f"{G}/glm53-flash-exl3-k3.25@tpurtell+rep070-F.s1.requests.jsonl"]),
    ("0.7.0 rep070 s2", "0.7.0", 5, "gpqa", f"{INV}/rep070/run-F/server-s2.log", [f"{G}/glm53-flash-exl3-k3.25@tpurtell+rep070-F.s2.requests.jsonl"]),
    ("0.9.1 record p1", "0.9.1", 3, "gpqa", f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1.server.log", [f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1.requests.jsonl"]),
    ("0.9.1 record p2-3", "0.9.1", 3, "gpqa", f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.server.log", [f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.requests.jsonl"]),
    ("0.9.1 D", "0.9.1", 3, "gpqa", f"{INV}/long-reply/d091/run/server-s1.log", [f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+d091.s1.requests.jsonl"]),
    ("0.9.1 D2", "0.9.1", 3, "gpqa", f"{INV}/long-reply/d091b/run/server-s1.log", [f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+d091b.s1.requests.jsonl"]),
    ("col511 arm s1", "0.9.1+D1", 3, "gpqa", f"{INV}/arm-col511/server-s1.log", []),
    ("col511 arm s2", "0.9.1+D1", 3, "gpqa", f"{INV}/arm-col511/server-s2.log", []),
    ("B0 s1234", "B0", 3, "gpqa", f"{INV}/bundle/screen/run-d1234/server-s1.log", []),
    ("B0 s1235", "B0", 3, "gpqa", f"{INV}/bundle/screen2/run-s1235/server-s2.log", []),
    ("B0 s1236", "B0", 3, "gpqa", f"{INV}/b0-screen-s1236/run-s1236/server-s3.log", []),
    ("B1 s1236", "B1", 3, "gpqa", f"{INV}/b1-screen/run-s1236/server-s1.log", [f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234+d7.s1.requests.jsonl"]),
    ("B+ v070", "0.7.0", 5, "cont-B+ (0.7.0 prefixes)", f"{INV}/long-reply/bplus/run/v070/server.log", [f"{INV}/long-reply/bplus/run/v070/requests.jsonl"]),
    ("B+ v091", "0.9.1", 3, "cont-B+ (0.7.0 prefixes)", f"{INV}/long-reply/bplus/run/v091/server.log", [f"{INV}/long-reply/bplus/run/v091/requests.jsonl"]),
    ("B+ b1", "B1", 3, "cont-B+ (0.7.0 prefixes)", f"{INV}/long-reply/bplus/run/b1/server.log", [f"{INV}/long-reply/bplus/run/b1/requests.jsonl"]),
    ("E v070", "0.7.0", 5, "cont-E (0.9.1 prefixes)", f"{INV}/long-reply/e-cross/run/v070/server.log", [f"{INV}/long-reply/e-cross/run/v070/requests.jsonl"]),
    ("E v091", "0.9.1", 3, "cont-E (0.9.1 prefixes)", f"{INV}/long-reply/e-cross/run/v091/server.log", [f"{INV}/long-reply/e-cross/run/v091/requests.jsonl"]),
]
MET = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d) \[metrics\.py:\d+\] SpecDecoding metrics:.*?Accepted: (\d+) tokens, Drafted: (\d+) tokens, "
                 r"Per-position acceptance rate: ([0-9., ]+), Avg")
LOG = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d) \[loggers\.py:\d+\] .*?Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([0-9.]+)%")
CAP = re.compile(r"cudagraph_capture_sizes': \[([0-9, ]+)\]")
KV = re.compile(r"GPU KV cache size: ([0-9,]+) tokens")
MODE = re.compile(r"cudagraph_mode': <CUDAGraphMode\.(\w+)")


def bucket(n):
    return "0" if n == 0 else "1-2" if n <= 2 else "3-5" if n <= 5 else "6-8" if n <= 8 else ">8"


def parse_server(path, K):
    caps, kv, mode = None, [], None
    status = {}
    acc = {"lines": 0, "accepted": 0, "drafted": 0, "pos": [0.0] * K}
    by = {}
    mets = []
    for line in open(path, errors="replace"):
        m = CAP.search(line)
        if m and caps is None:
            caps = [int(x) for x in m.group(1).split(",")]
        m = KV.search(line)
        if m:
            kv.append(int(m.group(1).replace(",", "")))
        m = MODE.search(line)
        if m and mode is None:
            mode = m.group(1)
        m = LOG.search(line)
        if m:
            status[m.group(1)] = int(m.group(2))
        m = MET.search(line)
        if m:
            mets.append((m.group(1), int(m.group(2)), int(m.group(3)), [float(x) for x in m.group(4).split(",")]))
    occ = {}
    for t, n in status.items():
        occ[bucket(n)] = occ.get(bucket(n), 0) + 1
    for t, a, d, pp in mets:
        if len(pp) != K or d == 0:
            continue
        blocks = d / K
        for tgt in (acc, by.setdefault(bucket(status.get(t, -1)) if t in status else "unpaired", {"lines": 0, "accepted": 0, "drafted": 0, "pos": [0.0] * K})):
            tgt["lines"] += 1
            tgt["accepted"] += a
            tgt["drafted"] += d
            for i, r in enumerate(pp):
                tgt["pos"][i] += r * blocks

    def fin(x):
        if not x["drafted"]:
            return None
        blocks = x["drafted"] / K
        return {"lines": x["lines"], "drafted": x["drafted"], "accept_rate": round(x["accepted"] / x["drafted"], 4),
                "mean_accepted_per_block": round(x["accepted"] / blocks, 3),
                "per_position": [round(p / blocks, 4) for p in x["pos"]]}

    tot = sum(occ.values()) or 1
    return {"capture_sizes": caps, "kv_pool_tokens": sorted(set(kv)), "cudagraph_mode": mode,
            "status_lines": tot, "occupancy_share_by_running": {k: round(v / tot, 4) for k, v in sorted(occ.items())},
            "acceptance": fin(acc), "acceptance_by_running": {k: fin(v) for k, v in sorted(by.items())}}


def parse_requests(paths):
    R = []
    for p in paths:
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                if r.get("t_start") is not None and r.get("secs") is not None:
                    R.append(r)
    return R


def tail_exposure(R):
    """For requests with completion_tokens >= 100k: time-weighted distribution of requests in flight (client side)
    during the part generated after token 100,000 (INFERRED window: constant rate within the request)."""
    ev = sorted([(r["t_start"], 1) for r in R] + [(r["t_start"] + r["secs"], -1) for r in R])
    times, counts, c = [], [], 0
    for t, d in ev:
        c += d
        times.append(t)
        counts.append(c)

    def inflight_profile(a, b):
        out = {}
        i = bisect.bisect_right(times, a) - 1
        t = a
        while t < b:
            n = counts[i] if i >= 0 else 0
            nxt = times[i + 1] if i + 1 < len(times) else b
            seg = min(nxt, b) - t
            if seg > 0:
                out[n] = out.get(n, 0) + seg
            t = min(nxt, b)
            i += 1
        return out

    agg, per = {}, []
    for r in R:
        T = r.get("completion_tokens") or 0
        if T < 100000:
            continue
        a = r["t_start"] + r["secs"] * 100000 / T
        b = r["t_start"] + r["secs"]
        prof = inflight_profile(a, b)
        tot = sum(prof.values())
        for n, s in prof.items():
            agg[n] = agg.get(n, 0) + s
        per.append({"T": T, "finish": r.get("finish_reason"), "secs_after_100k": round(b - a, 1),
                    "mean_inflight": round(sum(n * s for n, s in prof.items()) / tot, 2) if tot else None,
                    "whole_request_tok_s": round(T / r["secs"], 1) if r["secs"] else None})  # = (T-1e5)/(b-a) under the constant-rate window; not a post-100k measurement
    tot = sum(agg.values())
    return {"long_requests": len(per), "inflight_time_share": {str(n): round(s / tot, 4) for n, s in sorted(agg.items())} if tot else {},
            "per_request": per, "label": "INFERRED window (constant decode rate within a request); client-side in-flight count"}


out = {"sessions": []}
for label, eng, K, wl, sl, rl in S:
    if not os.path.exists(sl):
        out["sessions"].append({"label": label, "missing": sl})
        continue
    s = {"label": label, "engine": eng, "K": K, "workload": wl, "server_log": sl, **parse_server(sl, K)}
    if rl and all(os.path.exists(p) for p in rl):
        R = parse_requests(rl)
        s["requests_logged"] = len(R)
        s["tail_exposure"] = tail_exposure(R)
    out["sessions"].append(s)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "observed.json"), "w"), indent=1)
for s in out["sessions"]:
    if "missing" in s:
        print("MISSING", s["label"]); continue
    a = s["acceptance"] or {}
    te = s.get("tail_exposure", {})
    print(f"{s['label']:20s} K{s['K']} caps={len(s['capture_sizes'] or [])} kv={s['kv_pool_tokens']} pos={a.get('per_position')} "
          f"rate={a.get('accept_rate')} occ={s['occupancy_share_by_running']} long={te.get('long_requests')} tailinflight={te.get('inflight_time_share')}")
