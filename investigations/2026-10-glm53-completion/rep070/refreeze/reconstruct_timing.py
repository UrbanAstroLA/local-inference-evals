#!/usr/bin/env python3
"""Reconstruct occupied GPU-hours of the K3.25 GPQA records (amendment 12). CPU only, read-only on the logs.
occupied = 2 GPUs x (server start to server stop). Start = the eval chain's 'chain start' (it stops and removes any old
container, then starts the server). Stop = the first logged event at or after which the container was stopped (driver
'docker stop' after the chain, or the next chain's start, which stops it). Times are local (-07:00)."""
import json, datetime as dt
G = "<gpqa-runs>"; B = "<data-runs>/benchmarks"; I = "<runs>/arm-col511"
T = lambda s: dt.datetime.fromisoformat(s)
TZ = dt.timezone(dt.timedelta(hours=-7))

def reqwin(path, seeds):
    q = [json.loads(l) for l in open(path)]
    q = [x for x in q if x["seed"] in seeds]
    a = min(x["t_start"] for x in q); b = max(x["t_start"] + x["secs"] for x in q)
    f = lambda t: dt.datetime.fromtimestamp(t, TZ).isoformat(timespec="seconds")
    return f(a), f(b), len(q)

S = []
def add(record, session, passes, start, ready, first, last, stop, src, nreq=None, note=""):
    w = (T(stop) - T(start)).total_seconds()
    S.append(dict(record=record, session=session, passes=passes, start=start, ready=ready, first_request=first,
                  last_request_end=last, stop=stop, wall_s=int(w), wall_h=round(w / 3600, 4), gpu_hours=round(2 * w / 3600, 3),
                  requests=nreq, sources=src, note=note))

# 0.7.0 K3.25, pass 1, 2026-09-29 (no server log, no request log)
add("0.7.0 K3.25 DFlash2 x5", "2026-09-29", "1", "2026-09-29T06:51:39-07:00", "2026-09-29T07:00:54-07:00",
    "2026-09-29T07:01:03-07:00", "2026-09-29T10:34:37-07:00", "2026-09-29T10:34:58-07:00",
    ["eval-chain log 2026-09-29 (chain start 06:51:39 started the server; a second chain attached at 06:59:53 with NO_RESTART=1, "
     "'ready after 61s'; chain done 10:34:38)", "lm-eval log (Running generate_until 07:01:03; progress bar 198/198 at 3:33:34; "
     "results saved 10:34:37; total_evaluation_time_seconds 12822)", "next eval chain (4bpw, 0.8.0) started 10:34:58 and stops "
     "the container first (same container name)", "server-log extract in the chain log (weights loaded 06:54:12, KV sized 06:57:16, "
     "memory report 06:59:12; times converted from UTC)"], 198,
    "stop = the next chain's start, whose first action stops the container (the stop's own seconds not timed); no request "
    "log, so first/last request come from lm-eval")
# 0.7.0 K3.25, passes 2-3, 2026-10-10
f, l, n = reqwin(f"{G}/glm53-flash-exl3-k3.25@tpurtell+ps.requests.jsonl", {1235, 1236})
add("0.7.0 K3.25 DFlash2 x5", "2026-10-10", "2-3", "2026-10-10T00:59:36-07:00", "2026-10-10T01:07:54-07:00", f, l,
    "2026-10-10T08:45:31-07:00",
    ["eval-chain log 2026-10-10+ps (chain start 00:59:36, 'ready after 498s', chain done 08:45:29)", "request proxy log (+ps)",
     "re-run driver log 2026-10-09 (chain rc 08:45:29; server log saved and container stopped; 'CLEAN RERUNS DONE' 08:45:31)"], n)
# 0.9.1 K3.25, pass 1, 2026-10-08
f, l, n = reqwin(f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1.requests.jsonl", {1234})
add("0.9.1 K3.25 DFlash2 x3", "2026-10-08", "1", "2026-10-08T17:26:47-07:00", "2026-10-08T17:38:52-07:00", f, l,
    "2026-10-08T20:01:12-07:00",
    ["eval-chain log 2026-10-08 (chain start 17:26:47, 'ready after 725s', chain done 20:01:06)", "request proxy log",
     "pass-1 runner log ('GPQA pass 1 rc=0' 20:01:06; 'DONE (server stopped)' 20:01:12)"], n)
# 0.9.1 K3.25, passes 2-3, 2026-10-09
f, l, n = reqwin(f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.requests.jsonl", {1235, 1236})
add("0.9.1 K3.25 DFlash2 x3", "2026-10-09", "2-3", "2026-10-09T12:18:31-07:00", "2026-10-09T12:30:20-07:00", f, l,
    "2026-10-09T18:02:53-07:00",
    ["eval-chain log 2026-10-09+ps (chain start 12:18:31, 'ready after 709s', chain done 18:02:00)", "request proxy log (+ps)",
     "re-run driver log 2026-10-09 (driver noticed the chain end 18:02:46 on its 60 s poll, saved the server log and stopped "
     "the container; next line 18:02:53)"], n,
    "stop is an upper bound by up to 7 s; the server idled up to 53 s after the chain finished")
# column-511 arm, 2026-10-11
f, l, n = reqwin(f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix.s1.requests.jsonl", {1234})
add("0.9.1 K3.25 DFlash2 x3 + column-511 correction (arm)", "2026-10-11 s1", "1", "2026-10-11T12:59:09-07:00",
    "2026-10-11T13:10:45-07:00", f, l, "2026-10-11T15:51:20-07:00",
    ["arm runner log and costs.jsonl (session clock from runner start 12:59:09, chain start 12:59:10, 'ready after 695s', "
     "chain rc 15:51:15, server log saved and container stopped before session 2 at 15:51:20)", "request proxy log s1"], n)
f, l, n = reqwin(f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix.s2.requests.jsonl", {1235, 1236})
add("0.9.1 K3.25 DFlash2 x3 + column-511 correction (arm)", "2026-10-11 s2", "2-3", "2026-10-11T15:51:20-07:00",
    "2026-10-11T16:03:40-07:00", f, l, "2026-10-11T21:32:22-07:00",
    ["arm runner log and costs.jsonl (chain start 15:51:20, 'ready after 740s', chain rc 21:32:17, 'ARM DONE' 21:32:22 after "
     "the container stop)", "request proxy log s2"], n)

tot = {}
for s in S:
    t = tot.setdefault(s["record"], {"wall_s": 0, "gpu_hours": 0.0, "sessions": 0})
    t["wall_s"] += s["wall_s"]; t["gpu_hours"] = round(2 * t["wall_s"] / 3600, 3); t["sessions"] += 1
out = {"definition": "occupied GPU-hours = 2 GPUs x (server start to server stop); start = eval-chain start (server launched), "
                     "stop = container stopped (first logged event at or after the stop)", "sessions": S, "totals": tot}
json.dump(out, open(__file__.replace("reconstruct_timing.py", "timing-reconstruction.json"), "w"), indent=1)
hm = lambda s: s[11:19]
print("| record | session (passes) | start | ready | first request | last request end | stop | wall h | GPU-h |")
print("|---|---|---|---|---|---|---|---:|---:|")
for s in S:
    print(f"| {s['record']} | {s['session']} ({s['passes']}) | {hm(s['start'])} | {hm(s['ready'])} | {hm(s['first_request'])} | "
          f"{hm(s['last_request_end'])} | {hm(s['stop'])} | {s['wall_h']:.3f} | {s['gpu_hours']:.2f} |")
for k, t in tot.items():
    print(f"total {k}: {t['sessions']} sessions, {t['wall_s']/3600:.3f} h wall, {t['gpu_hours']:.2f} GPU-h")
