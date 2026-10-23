#!/usr/bin/env python3
"""22ao live-check workload client (one arm). Frozen with the package; no GPQA text is written.

usage: client_22ao.py OUT_DIR DEADLINE_EPOCH [BASE_URL]

Workload (LIVE-CHECK-PROPOSAL.md section 4): the 12-question panel x 2 fresh seeds = 24 chat requests,
temperature / top_p / chat_template_kwargs exactly as lm-eval sent them (panel fixture, hash-checked against
the 0.9.1 record's request log), natural stopping. Exactly 8 outstanding: a new request is sent the moment one
finishes, from a queue that front-loads 8 long requests (max_tokens 6,144 down to 2,560 in steps of 512) and
then 16 short ones (768), so runner rows recycle (shorter after longer) and the drain passes every padded size.
Seeds 72201-72224, one per request, recorded; the same list on both arms. Requests are not sent past the
deadline. Records: ids, seeds, times, token counts, finish reasons and a sha256 of the reply only.
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
PANEL = [8, 69, 71, 79, 81, 88, 109, 120, 121, 127, 147, 170]
LONG = [6144 - 512 * i for i in range(8)]                    # 6144 .. 2560
SHORT = 768
SEED0 = 72201


def plan():
    """24 jobs in queue order: (doc, seed, max_tokens, kind)."""
    jobs = []
    seed = SEED0
    for i, doc in enumerate(PANEL[:8]):                    # first seed: 8 long
        jobs.append((doc, seed, LONG[i], "long")); seed += 1
    for doc in PANEL[8:]:                                  # first seed: 4 short
        jobs.append((doc, seed, SHORT, "short")); seed += 1
    for doc in PANEL:                                      # second seed: 12 short
        jobs.append((doc, seed, SHORT, "short")); seed += 1
    assert len(jobs) == 24 and len({j[1] for j in jobs}) == 24
    return jobs


def main():
    fx = {i["doc_id"]: i for i in json.load(open(FIXTURE))["items"]}
    assert sorted(fx) == sorted(PANEL), "panel fixture does not match the panel"
    model = requests.get(f"{BASE}/v1/models", timeout=30).json()["data"][0]["id"]
    log = open(os.path.join(OUT, "client.jsonl"), "a")
    lock = threading.Lock()
    inflight = [0]

    def rec(d):
        with lock:
            log.write(json.dumps(d) + "\n")
            log.flush()

    q = queue.Queue()
    for j in plan():
        q.put(j)
    rec(dict(kind="plan", jobs=plan(), width=WIDTH, panel=PANEL, t=round(time.time(), 3)))

    def worker():
        while True:
            try:
                doc, seed, mt, kind = q.get_nowait()
            except queue.Empty:
                return
            if time.time() > DEADLINE:
                rec(dict(kind="skipped", doc=doc, seed=seed, reason="deadline", t=round(time.time(), 3)))
                continue
            it = fx[doc]
            body = dict(model=model, messages=it["messages"], max_tokens=mt, temperature=it["temperature"],
                        top_p=it["top_p"], seed=seed, chat_template_kwargs=it["chat_template_kwargs"])
            with lock:
                inflight[0] += 1
                n_at_send = inflight[0]
            t0 = time.time()
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
            rec(dict(kind="request", doc=doc, seed=seed, max_tokens=mt, kind_req=kind, t0=round(t0, 3), t1=round(t1, 3),
                     secs=round(t1 - t0, 2), inflight_at_send=n_at_send, finish=ch.get("finish_reason"),
                     completion_tokens=u.get("completion_tokens"), prompt_tokens=u.get("prompt_tokens"),
                     reply_sha256=hashlib.sha256(text.encode()).hexdigest() if not err else None, error=err))
            print(time.strftime("%H:%M:%S"), f"q{doc} seed {seed} {kind} gen={u.get('completion_tokens')} "
                  f"finish={ch.get('finish_reason')} {round(t1 - t0, 1)}s err={err}", flush=True)

    ths = [threading.Thread(target=worker) for _ in range(WIDTH)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    rec(dict(kind="done", t=round(time.time(), 3)))


if __name__ == "__main__":
    main()
