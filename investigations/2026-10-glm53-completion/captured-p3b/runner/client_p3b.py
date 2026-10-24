#!/usr/bin/env python3
"""captured-p3b capture-run workload client.  No GPQA text is written anywhere.

usage: client_p3b.py OUT_DIR DEADLINE_EPOCH [BASE_URL]

Workload (PROPOSAL.md section 4), adapted from the 22ao client (runner/client_22ao.py there):
  * 16 chat requests from the 12-question GPQA panel (private fixture, read at run time from PANEL_FIXTURE; the
    same fixture 22ao used, hash-checked there against the 0.9.1 record's request log), temperature / top_p /
    chat_template_kwargs exactly as in the fixture, natural stopping (no stop strings added).
  * Fresh seeds 72301-72316, one per request, recorded (distinct from 22ao's 72201-72224; checked unused under
    <data-runs> when the package was built).
  * Queue: 8 long requests first (max_tokens 2816 down to 1024 in steps of 256; q127, the panel's longest prompt,
    is long #1 so clamped-regime rows (p + 1 >= 2048) exist from the first seconds), then 8 short (512).
  * In flight: a short ramp 1, 2, 4, 6, 8 (RAMP_S seconds per level) so the padded decode sizes 4, 8, 16 and 24
    are served early, then exactly 8 outstanding with immediate backfill, then the natural drain.
  * Stops early when OUT_DIR/STOP appears (the runner writes it once the capture quotas are met): no new requests,
    the in-flight ones are abandoned (connection closed; the server aborts them).  Requests are not sent past the
    deadline.
Records (client.jsonl): plan, ids, seeds, times, token counts, finish reasons, sha256 of the reply only.
"""
import hashlib
import json
import os
import queue
import sys
import threading
import time

import requests

OUT, DEADLINE = sys.argv[1], float(sys.argv[2])
BASE = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8001"
FIXTURE = os.environ.get("PANEL_FIXTURE",
                         "<runs>/livecheck-22ao/PRIVATE/panel_fixture.LOCAL.json")
WIDTH = 8
RAMP = (1, 2, 4, 6, 8)
RAMP_S = float(os.environ.get("RAMP_S", "4"))
PANEL = [8, 69, 71, 79, 81, 88, 109, 120, 121, 127, 147, 170]
LONG_DOCS = [127, 8, 69, 71, 79, 81, 88, 109]
SHORT_DOCS = [120, 121, 147, 170, 8, 69, 71, 79]
LONG = [2816 - 256 * i for i in range(8)]                   # 2816 .. 1024
SHORT = 512
SEED0 = 72301


def plan():
    """16 jobs in queue order: (doc, seed, max_tokens, kind)."""
    jobs = []
    seed = SEED0
    for doc, mt in zip(LONG_DOCS, LONG):
        jobs.append((doc, seed, mt, "long"))
        seed += 1
    for doc in SHORT_DOCS:
        jobs.append((doc, seed, SHORT, "short"))
        seed += 1
    assert len(jobs) == 16 and len({j[1] for j in jobs}) == 16 and set(LONG_DOCS) | set(SHORT_DOCS) == set(PANEL)
    return jobs


def main():
    fx = {i["doc_id"]: i for i in json.load(open(FIXTURE))["items"]}
    assert sorted(fx) == sorted(PANEL), "panel fixture does not match the panel"
    model = requests.get(f"{BASE}/v1/models", timeout=30).json()["data"][0]["id"]
    log = open(os.path.join(OUT, "client.jsonl"), "a")
    lock = threading.Lock()
    inflight = [0]
    peak = [0]
    stop_file = os.path.join(OUT, "STOP")
    t_start = time.time()

    def rec(d):
        with lock:
            log.write(json.dumps(d) + "\n")
            log.flush()

    def cap_now():
        k = int((time.time() - t_start) // RAMP_S)
        return RAMP[k] if k < len(RAMP) else WIDTH

    def stopped():
        return os.path.exists(stop_file) or time.time() > DEADLINE

    q = queue.Queue()
    for j in plan():
        q.put(j)
    rec(dict(kind="plan", jobs=plan(), width=WIDTH, ramp=list(RAMP), ramp_s=RAMP_S, panel=PANEL,
             t=round(time.time(), 3)))

    def worker(idx):
        while True:
            reserved = False
            while not stopped():                       # ramp: reserve a slot only while the cap admits one
                with lock:
                    if inflight[0] < cap_now():
                        inflight[0] += 1
                        peak[0] = max(peak[0], inflight[0])
                        n_at_send = inflight[0]
                        reserved = True
                        break
                time.sleep(0.05)
            if not reserved:
                return
            try:
                doc, seed, mt, kind = q.get_nowait()
            except queue.Empty:
                with lock:
                    inflight[0] -= 1
                return
            it = fx[doc]
            body = dict(model=model, messages=it["messages"], max_tokens=mt, temperature=it["temperature"],
                        top_p=it["top_p"], seed=seed, chat_template_kwargs=it["chat_template_kwargs"])
            t0 = time.time()
            rec(dict(kind="sent", doc=doc, seed=seed, max_tokens=mt, kind_req=kind, t0=round(t0, 3),
                     inflight_at_send=n_at_send, worker=idx))
            try:
                r = requests.post(f"{BASE}/v1/chat/completions", json=body, timeout=max(60, DEADLINE - t0 + 30))
                j = r.json()
                err = None if "choices" in j else (j.get("error") or j.get("message") or f"http {r.status_code}")
            except Exception as e:  # noqa: BLE001
                j, err = {}, repr(e)[:200]
            t1 = time.time()
            with lock:
                inflight[0] -= 1
            ch = (j.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            text = (msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")
            u = j.get("usage") or {}
            rec(dict(kind="request", doc=doc, seed=seed, max_tokens=mt, kind_req=kind, t0=round(t0, 3),
                     t1=round(t1, 3), secs=round(t1 - t0, 2), inflight_at_send=n_at_send,
                     finish=ch.get("finish_reason"), completion_tokens=u.get("completion_tokens"),
                     prompt_tokens=u.get("prompt_tokens"),
                     reply_sha256=hashlib.sha256(text.encode()).hexdigest() if not err else None, error=err))
            print(time.strftime("%H:%M:%S"), f"q{doc} seed {seed} {kind} gen={u.get('completion_tokens')} "
                  f"finish={ch.get('finish_reason')} {round(t1 - t0, 1)}s err={err}", flush=True)

    ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(WIDTH)]
    for t in ths:
        t.start()
    while any(t.is_alive() for t in ths):
        if os.path.exists(stop_file):
            with lock:
                n = inflight[0]
            rec(dict(kind="stop", reason="STOP file (capture quotas met)", inflight=n, queued=q.qsize(),
                     peak_inflight=peak[0], t=round(time.time(), 3)))
            log.flush()
            print(f"STOP: abandoning {n} in-flight request(s), {q.qsize()} queued", flush=True)
            os._exit(0)                                    # closes the sockets; the server aborts the requests
        time.sleep(0.2)
    rec(dict(kind="done", peak_inflight=peak[0], queued=q.qsize(), t=round(time.time(), 3)))


if __name__ == "__main__":
    main()
