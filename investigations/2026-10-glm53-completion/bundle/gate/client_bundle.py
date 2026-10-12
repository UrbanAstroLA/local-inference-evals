#!/usr/bin/env python3
"""Bundle gate client (PLAN.md amendment 14 (3)). PREPARED, NOT RUN.

usage: client_bundle.py OUTDIR DEADLINE_EPOCH
env:   PHASES      space-separated subset of: warm TF DS Q G P   (default: all, in that order)
       REPEATS     T/F repeats (default 4)        DS_REPEATS  D/S repeats (default 2)
       FIXTURE     long-context fixture (default OUTDIR/requests.json = phase2-stage2/requests.json)
       GPQA        short-context fixture (default ../gpqa_fixture.LOCAL.json next to this script; LOCAL ONLY)

Phases (every request: one JSON line in OUTDIR/client.jsonl with its seed; every phase: a phase line):
  warm  prefix-cache warm of the 8 long-context requests (as stage-2 run 2 / tail guard / col511 gate)
  TF    T = 8 long-context requests x 200 tokens, F = 4 x 128 tokens, temperature 1.0 -- byte-identical
        workload and seeds to tailguard/graph-a1 and tailfix/{T,U}-graph-a1 (throughput reference 88.3 / 59.3)
  DS    D = drain (8 requests, 40..320 tokens, finishing one by one: route transitions 8->1 in flight),
        S = staggered admission (4 running, 4 admitted 3-12 s later: mixed prefill/decode batches) -- as col511 gate
  Q     short contexts: 16 GPQA prompts (docs 0-15, 124-600 prompt tokens), chat template with thinking,
        temperature 1.0, top_p 0.95, max_tokens 2048, 8 in flight with continuous admission (a new request is
        admitted when one finishes: mixed prefill/decode, contexts below the 2,176-token index width), then
        Q-reuse = the same 16 prompts again with new seeds (prefix-cache hits on the prompts)
  G     GPQA-shaped throughput: docs 16-23 at once, 8 in flight, max_tokens 6144 (early-GPQA regime)
  P     D4 probe: 4 GPQA prompts (docs 0-3, chat) + 2 long-context prompts (r7 4,240 / r5 16,240 tokens,
        completions), each ALONE (1 in flight), temperature 0, seed 4242, max_tokens 96, logprobs; each prompt
        twice (within-session repeatability control); OUTDIR/probe.jsonl holds token strings, logprobs and a
        canonical sha256 per request for cross-session comparison (S1 vs S2 vs S3)
After each phase: col511 checker and tail guard poll triggers (harmless when the observers are off).
"""
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time

import requests

OUT, DEADLINE = sys.argv[1], float(sys.argv[2])
PHASES = os.environ.get("PHASES", "warm TF DS Q G P").split()
REPEATS = int(os.environ.get("REPEATS", "4"))
DS_REPEATS = int(os.environ.get("DS_REPEATS", "2"))
BASE = "http://127.0.0.1:8001"
CONTAINER = "glm53-flash-exl3-b12x-vllm"
HERE = os.path.dirname(os.path.abspath(__file__))
FIX = json.load(open(os.environ.get("FIXTURE", f"{OUT}/requests.json")))
BYID = {r["id"]: r for r in FIX["requests"]}
GQ = {i["doc_id"]: i for i in json.load(open(os.environ.get("GPQA", f"{HERE}/gpqa_fixture.LOCAL.json")))["items"]}
LOG = open(f"{OUT}/client.jsonl", "a")
PROBE = open(f"{OUT}/probe.jsonl", "a")
lock = threading.Lock()


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def rec(d, fh=None):
    with lock:
        (fh or LOG).write(json.dumps(d) + "\n")
        (fh or LOG).flush()


def dump(tag):
    p = subprocess.run(["docker", "exec", CONTAINER, "sh", "-c",
                        "mkdir -p /tmp/col511check /tmp/tailguard && touch /tmp/col511check/DUMP /tmp/tailguard/DUMP"],
                       capture_output=True, text=True)
    rec(dict(kind="dump", tag=tag, t=round(time.time(), 3), rc=p.returncode))


MODEL = requests.get(f"{BASE}/v1/models", timeout=30).json()["data"][0]["id"]
log(f"model {MODEL}; phases {PHASES}")


def _post(path, body):
    t0 = time.time()
    try:
        j = requests.post(f"{BASE}{path}", json=body, timeout=3600).json()
        err = (j.get("error") or j.get("message")) if "choices" not in j else None
    except Exception as e:  # noqa: BLE001
        j, err = {}, repr(e)
    return j, err, t0, time.time()


def complete_long(rid, max_tokens, temperature, seed, phase, extra=None):
    r = BYID[rid]
    body = dict(model=MODEL, prompt=r["ids"], max_tokens=max_tokens, temperature=temperature,
                top_p=0.95 if temperature > 0 else 1.0, seed=seed, **(extra or {}))
    j, err, t0, t1 = _post("/v1/completions", body)
    ch = (j.get("choices") or [{}])[0]
    u = j.get("usage") or {}
    d = dict(kind="request", phase=phase, id=rid, doc=r["doc"], L=r["L"], total=len(r["ids"]), seed=seed,
             max_tokens=max_tokens, temperature=temperature, t0=round(t0, 3), t1=round(t1, 3), secs=round(t1 - t0, 2),
             finish=ch.get("finish_reason"), completion_tokens=u.get("completion_tokens"),
             prompt_tokens=u.get("prompt_tokens"),
             cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"), error=err)
    rec(d)
    log(f"{phase} {rid} L={r['L']} cached={d['cached_tokens']}/{d['prompt_tokens']} gen={d['completion_tokens']} "
        f"fin={d['finish']} {d['secs']}s err={err}")
    return d, ch


def complete_chat(doc, max_tokens, temperature, seed, phase, extra=None):
    it = GQ[doc]
    body = dict(model=MODEL, messages=it["messages"], max_tokens=max_tokens, temperature=temperature,
                top_p=it["top_p"] if temperature > 0 else 1.0, seed=seed,
                chat_template_kwargs=it["chat_template_kwargs"], **(extra or {}))
    j, err, t0, t1 = _post("/v1/chat/completions", body)
    ch = (j.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    u = j.get("usage") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    d = dict(kind="request", phase=phase, id=f"q{doc}", doc=doc, messages_sha256=it["messages_sha256"], seed=seed,
             max_tokens=max_tokens, temperature=temperature, t0=round(t0, 3), t1=round(t1, 3), secs=round(t1 - t0, 2),
             finish=ch.get("finish_reason"), completion_tokens=u.get("completion_tokens"),
             prompt_tokens=u.get("prompt_tokens"),
             cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
             content_chars=len(content), reasoning_chars=len(reasoning),
             empty_reply=not content.strip(), error=err)
    rec(d)
    log(f"{phase} q{doc} prompt={d['prompt_tokens']} cached={d['cached_tokens']} gen={d['completion_tokens']} "
        f"fin={d['finish']} {d['secs']}s err={err}")
    return d, ch


def phase_rec(tag, t0, **kw):
    rec(dict(kind="phase", tag=tag, secs=round(time.time() - t0, 2), t0=round(t0, 3), t1=round(time.time(), 3), **kw))
    log(f"phase {tag} done in {time.time() - t0:.1f}s")
    dump(tag)


def run_threads(fns):
    ths = [threading.Thread(target=f) for f in fns]
    [t.start() for t in ths]
    [t.join() for t in ths]


def pool(tag, jobs, width):
    """continuous admission: at most `width` in flight; a new job starts when one finishes."""
    q = queue.Queue()
    for j in jobs:
        q.put(j)

    def worker():
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            j()
    t0 = time.time()
    run_threads([worker] * width)
    return t0


def need(seconds, what):
    if time.time() > DEADLINE - seconds:
        log(f"deadline: skipping {what}")
        return False
    return True


ALL8 = ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"]
for ph in PHASES:
    if ph == "warm":
        if not need(300, "warm"):
            break
        t0 = time.time()
        for i, rid in enumerate(["r7", "r6", "r5", "r4", "r3", "r2", "r1", "r0"]):
            complete_long(rid, 1, 0.0, 7810000 + i, "warm")
        phase_rec("warm", t0)
    elif ph == "TF":
        for rep in range(REPEATS):
            for tag, ids, sb, mt in (("T", ALL8, 7840000, 200), ("F", ["r0", "r2", "r4", "r7"], 7842000, 128)):
                if not need(100, f"{tag}-rep{rep}"):
                    break
                t0 = time.time()
                base = sb + 10000 * rep   # = client_tg.py seeds (distinct per request and repeat)
                run_threads([lambda rid=rid, k=k, tag=tag, mt=mt, base=base:
                             complete_long(rid, mt, 1.0, base + k, f"{tag}-rep{rep}") for k, rid in enumerate(ids)])
                phase_rec(f"{tag}-rep{rep}", t0, ids=ids, seed_base=base)
    elif ph == "DS":
        PLANS = {"D": ([(r, 40 * (8 - k), 0) for k, r in enumerate(ALL8)], 7850000, 40),
                 "S": ([("r0", 300, 0), ("r2", 300, 0), ("r4", 300, 0), ("r7", 300, 0),
                        ("r1", 120, 3), ("r3", 120, 6), ("r5", 120, 9), ("r6", 120, 12)], 7860000, 45)}
        for rep in range(DS_REPEATS):
            for tag in ("D", "S"):
                plan, sb, nd = PLANS[tag]
                if not need(nd + 60, f"{tag}-rep{rep}"):
                    break
                t0 = time.time()
                base = sb + 10000 * rep
                run_threads([lambda rid=rid, mt=mt, k=k, delay=delay, tag=tag, base=base: (
                    time.sleep(delay), complete_long(rid, mt, 1.0, base + k, f"{tag}-rep{rep}"))
                    for k, (rid, mt, delay) in enumerate(plan)])
                phase_rec(f"{tag}-rep{rep}", t0, plan=plan, seed_base=base)
    elif ph == "Q":
        for tag, sb in (("Q", 9100000), ("Qreuse", 9200000)):
            if not need(420, tag):
                break
            t0 = pool(tag, [lambda d=d, tag=tag, sb=sb: complete_chat(d, 2048, 1.0, sb + d, tag) for d in range(16)], 8)
            phase_rec(tag, t0, docs=list(range(16)), seed_base=sb, width=8, max_tokens=2048)
    elif ph == "G":
        if not need(360, "G"):
            continue
        t0 = time.time()
        run_threads([lambda d=d: complete_chat(d, 6144, 1.0, 9300000 + d, "G") for d in range(16, 24)])
        phase_rec("G", t0, docs=list(range(16, 24)), seed_base=9300000, max_tokens=6144)
    elif ph == "P":
        if not need(150, "P"):
            continue
        t0 = time.time()
        for rnd in (1, 2):
            for d in range(4):
                r, ch = complete_chat(d, 96, 0.0, 4242, f"P{rnd}", extra=dict(logprobs=True, top_logprobs=1))
                lp = ((ch.get("logprobs") or {}).get("content")) or []
                toks = [(x.get("token"), x.get("logprob")) for x in lp]
                msg = ch.get("message") or {}
                canon = json.dumps(dict(c=msg.get("content"), r=msg.get("reasoning_content") or msg.get("reasoning"),
                                        t=toks), sort_keys=True)
                rec(dict(kind="probe", round=rnd, id=f"q{d}", n_tokens=len(toks), tokens=toks,
                         sha256=hashlib.sha256(canon.encode()).hexdigest(), finish=r["finish"], error=r["error"]), PROBE)
            for rid in ("r7", "r5"):
                r, ch = complete_long(rid, 96, 0.0, 4242, f"P{rnd}", extra=dict(logprobs=1))
                lpo = ch.get("logprobs") or {}
                toks = list(zip(lpo.get("tokens") or [], lpo.get("token_logprobs") or []))
                canon = json.dumps(dict(text=ch.get("text"), t=toks), sort_keys=True)
                rec(dict(kind="probe", round=rnd, id=rid, n_tokens=len(toks), tokens=toks,
                         sha256=hashlib.sha256(canon.encode()).hexdigest(), finish=r["finish"], error=r["error"]), PROBE)
        phase_rec("P", t0)
log("CLIENT DONE")
