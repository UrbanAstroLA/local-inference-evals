#!/usr/bin/env python3
"""B1 mapping-guard workload (PLAN.md amendment 18a, step 2c + appended phase 2d).  Copy of the frozen
b0-mapguard client: the frozen phases are unchanged (same requests, seeds, order); changes are
  * probe records keep sha256 / n_tokens and per-token hashes, no token text (identity unchanged);
  * after P1: a readout with a 6 s wait, the "phase2d_start" record (the analysis window ends there),
    then phase 2d (outside every coverage count and the verdict), then the final readout.
Existing fixtures only.

usage: client_mapguard.py OUTDIR DEADLINE_EPOCH
env:   FIXTURE (default OUTDIR/requests.json = phase2-stage2/requests.json: active-reasoning prefixes)
       GPQA    (default bundle/gate/gpqa_fixture.LOCAL.json; LOCAL ONLY, never published)
       PHASES  subset/order of: P0 warm T0 F0 T1 F1 D S Q P1 (default: all; the post-a2 trimmed
               rerun uses "P0 warm T0 F0 T1 S P1")

Phases (functions copied from bundle/gate/client_bundle.py; seeds distinct per request, recorded):
  P0      short greedy probes q0-q3 (chat, temperature 0, seed 4242, 96 tokens, each alone) --
          output-identity check against the s1 B0-check-image record (short prompts only)
  warm    8 stage-2 prefixes, 1 token each, one at a time (prefix-cache population; partial hits
          r6<-r7, r5<-r6, r2<-r3, r1<-r0 class; pool cycling towards block ids >= 935)
  T0 F0 T1 F1   T = 8 prefixes x 200 tokens at once (5-8 in flight), F = 4 x 128 (<= 4 in flight):
          prefix hits on every request, mixed prefill/decode as remainders prefill, DFlash2
          acceptance/rejection, fused/tiled route transitions, request completion and row reuse;
          high block ids are reached during T1 in the CPU allocator simulation (reach/)
  D       drain: 8 requests finishing one by one (40..320 tokens): 8 -> 1 in flight
  S       staggered admission: 4 running + 4 admitted 3-12 s later (mixed prefill/decode)
  Q       16 GPQA prompts, 8 in flight, continuous admission, max_tokens 1024 (bounded):
          short contexts (< 2,176), row reuse, mixed batches
  P1      the four probes again (within-session repeat)
  2d      (appended, amendment 18a) doc-127 chunk-boundary arithmetic: for D = 0..7, D background
          streaming completions of fixture prefix r7 (4,240 tokens, never a prefix hit: < one 4,352-token
          block; max_tokens 1024, temperature 1.0, distinct recorded seeds) are brought into decode
          (>= 2 streamed tokens each), then the doc-127 GPQA chat prompt (2,795 tokens; messages read
          locally from the pass-1 lm-eval samples and verified by sha256 before sending) is admitted
          with max_tokens 8; the background streams are then closed (server-side abort) and the server
          is drained before the next case.  Recorded: requested D, background requests in decode at
          send, seeds, times, server prompt_tokens / finish -- token counts, hashes and offsets only.
          The ACTUAL scheduled decode count and chunk lengths come from the guard's EngineCore records
          (analysis/analyze_mapguard.py phase2d).  Skipped (recorded) if the deadline is near.
Before the first request and after every phase the guard readout is triggered (DUMP).
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
BASE = "http://127.0.0.1:8001"
CONTAINER = "glm53-flash-exl3-b12x-vllm"
GATE = "<runs>/bundle/gate"
FIX = json.load(open(os.environ.get("FIXTURE", f"{OUT}/requests.json")))
BYID = {r["id"]: r for r in FIX["requests"]}
GQ = {i["doc_id"]: i for i in json.load(open(os.environ.get("GPQA", f"{GATE}/gpqa_fixture.LOCAL.json")))["items"]}
LOG = open(f"{OUT}/client.jsonl", "a")
PROBE = open(f"{OUT}/probe.jsonl", "a")
lock = threading.Lock()
ALL8 = ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"]
PHASES = os.environ.get("PHASES", "P0 warm T0 F0 T1 F1 D S Q P1 2d").split()


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def rec(d, fh=None):
    with lock:
        (fh or LOG).write(json.dumps(d) + "\n")
        (fh or LOG).flush()


def dump(tag, wait=0.0):
    p = subprocess.run(["docker", "exec", CONTAINER, "sh", "-c",
                        "mkdir -p /tmp/mapguard && touch /tmp/mapguard/DUMP"],
                       capture_output=True, text=True)
    rec(dict(kind="dump", tag=tag, t=round(time.time(), 3), rc=p.returncode))
    if wait:
        time.sleep(wait)


MODEL = requests.get(f"{BASE}/v1/models", timeout=30).json()["data"][0]["id"]


def _post(path, body):
    t0 = time.time()
    try:
        j = requests.post(f"{BASE}{path}", json=body, timeout=1800).json()
        err = (j.get("error") or j.get("message")) if "choices" not in j else None
    except Exception as e:  # noqa: BLE001
        j, err = {}, repr(e)
    return j, err, t0, time.time()


def complete_long(rid, max_tokens, temperature, seed, phase):
    r = BYID[rid]
    body = dict(model=MODEL, prompt=r["ids"], max_tokens=max_tokens, temperature=temperature,
                top_p=0.95 if temperature > 0 else 1.0, seed=seed)
    j, err, t0, t1 = _post("/v1/completions", body)
    ch = (j.get("choices") or [{}])[0]
    u = j.get("usage") or {}
    d = dict(kind="request", phase=phase, id=rid, L=r["L"], seed=seed, max_tokens=max_tokens,
             temperature=temperature, t0=round(t0, 3), t1=round(t1, 3), secs=round(t1 - t0, 2),
             finish=ch.get("finish_reason"), completion_tokens=u.get("completion_tokens"),
             prompt_tokens=u.get("prompt_tokens"),
             cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"), error=err)
    rec(d)
    log(f"{phase} {rid} cached={d['cached_tokens']}/{d['prompt_tokens']} gen={d['completion_tokens']} "
        f"{d['secs']}s err={err}")
    return d


def complete_chat(doc, max_tokens, temperature, seed, phase, extra=None):
    it = GQ[doc]
    body = dict(model=MODEL, messages=it["messages"], max_tokens=max_tokens, temperature=temperature,
                top_p=it["top_p"] if temperature > 0 else 1.0, seed=seed,
                chat_template_kwargs=it["chat_template_kwargs"], **(extra or {}))
    j, err, t0, t1 = _post("/v1/chat/completions", body)
    ch = (j.get("choices") or [{}])[0]
    u = j.get("usage") or {}
    d = dict(kind="request", phase=phase, id=f"q{doc}", seed=seed, max_tokens=max_tokens,
             temperature=temperature, t0=round(t0, 3), t1=round(t1, 3), secs=round(t1 - t0, 2),
             finish=ch.get("finish_reason"), completion_tokens=u.get("completion_tokens"),
             prompt_tokens=u.get("prompt_tokens"),
             cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"), error=err)
    rec(d)
    log(f"{phase} q{doc} prompt={d['prompt_tokens']} gen={d['completion_tokens']} {d['secs']}s err={err}")
    return d, ch


def probes(rnd):
    t0 = time.time()
    for d in range(4):   # identical payload/canonicalisation to client_bundle.py phase P
        r, ch = complete_chat(d, 96, 0.0, 4242, f"P{rnd}", extra=dict(logprobs=True, top_logprobs=1))
        lp = ((ch.get("logprobs") or {}).get("content")) or []
        toks = [(x.get("token"), x.get("logprob")) for x in lp]
        msg = ch.get("message") or {}
        canon = json.dumps(dict(c=msg.get("content"), r=msg.get("reasoning_content") or msg.get("reasoning"),
                                t=toks), sort_keys=True)
        rec(dict(kind="probe", round=rnd, id=f"q{d}", n_tokens=len(toks),
                 tok_sha=[hashlib.sha256(json.dumps(t).encode()).hexdigest()[:12] for t in toks],
                 sha256=hashlib.sha256(canon.encode()).hexdigest(), finish=r["finish"], error=r["error"]), PROBE)
    phase_rec(f"P{rnd}", t0)


def phase_rec(tag, t0, **kw):
    rec(dict(kind="phase", tag=tag, secs=round(time.time() - t0, 2), t0=round(t0, 3), t1=round(time.time(), 3), **kw))
    log(f"phase {tag} done in {time.time() - t0:.1f}s")
    dump(tag)


def run_threads(fns):
    ths = [threading.Thread(target=f) for f in fns]
    [t.start() for t in ths]
    [t.join() for t in ths]


def pool(jobs, width):
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
    run_threads([worker] * width)


def need(seconds, what):
    if time.time() > DEADLINE - seconds:
        log(f"deadline: skipping {what}")
        rec(dict(kind="skipped", what=what, t=round(time.time(), 3)))
        return False
    return True


log(f"model {MODEL}")
dump("baseline", wait=6.0)     # guard readout before any client traffic (analysis baseline)
if "P0" in PHASES and need(60, "P0"):
    probes(0)
if "warm" in PHASES and need(150, "warm"):
    t0 = time.time()
    for i, rid in enumerate(["r7", "r6", "r5", "r4", "r3", "r2", "r1", "r0"]):
        complete_long(rid, 1, 0.0, 8810000 + i, "warm")
    phase_rec("warm", t0)
for rep in range(2):
    for tag, ids, sb, mt in (("T", ALL8, 8840000, 200), ("F", ["r0", "r2", "r4", "r7"], 8842000, 128)):
        if f"{tag}{rep}" not in PHASES or not need(30, f"{tag}{rep}"):
            continue
        t0 = time.time()
        base = sb + 10000 * rep
        run_threads([lambda rid=rid, k=k, mt=mt, base=base, tag=tag:
                     complete_long(rid, mt, 1.0, base + k, f"{tag}{rep}") for k, rid in enumerate(ids)])
        phase_rec(f"{tag}{rep}", t0, ids=ids, seed_base=base)
PLANS = {"D": ([(r, 40 * (8 - k), 0) for k, r in enumerate(ALL8)], 8850000, 45),
         "S": ([("r0", 300, 0), ("r2", 300, 0), ("r4", 300, 0), ("r7", 300, 0),
                ("r1", 120, 3), ("r3", 120, 6), ("r5", 120, 9), ("r6", 120, 12)], 8860000, 45)}
for tag in ("D", "S"):
    plan, sb, nd = PLANS[tag]
    if tag not in PHASES or not need(nd, tag):
        continue
    t0 = time.time()
    run_threads([lambda rid=rid, mt=mt, k=k, delay=delay, tag=tag, sb=sb: (
        time.sleep(delay), complete_long(rid, mt, 1.0, sb + k, tag)) for k, (rid, mt, delay) in enumerate(plan)])
    phase_rec(tag, t0, plan=plan, seed_base=sb)
if "Q" in PHASES and need(75, "Q"):
    t0 = time.time()
    pool([lambda d=d: complete_chat(d, 1024, 1.0, 9400000 + d, "Q") for d in range(16)], 8)
    phase_rec("Q", t0, docs=list(range(16)), seed_base=9400000, width=8, max_tokens=1024)
if "P1" in PHASES and need(25, "P1"):
    probes(1)

# ---------------------------------------------------------------- phase 2d (appended; outside the verdict)
SAMPLES_2D = ("<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1__pass1/wrldsuksgo2mars__GLM-5.3-Flash-EXL3-K3.25-v1/"
              "samples_gpqa_diamond_thinking_2026-10-08T20-01-05.904768.jsonl")
SAMPLES_2D_SHA = "be5c567defff894ac898e7e90ea2149e4fd5ff64486216a6c26f736efb447fc1"
DOC127_MSG_SHA = "b41ce17bb0d4ce27af85de277fdd16b86cb88ad66683d95c799146dced26ad62"
DOC127_TOKENS = 2795
D2_BG = "r7"                      # 4,240 tokens < one 4,352-token block: never a prefix hit
D2_SEED_BG, D2_SEED_DOC = 9700000, 9710000
D2_NEED_S = float(os.environ.get("D2_NEED_S", "240"))   # 2d starts only with this much time left
D2_CASE_S = float(os.environ.get("D2_CASE_S", "40"))    # and each case only with this much left


def doc127_messages():
    """Doc 127 as the GPQA client sent it (lm-eval sample arguments), verified by hash; LOCAL ONLY."""
    if hashlib.sha256(open(SAMPLES_2D, "rb").read()).hexdigest() != SAMPLES_2D_SHA:
        return None, "samples file hash differs"
    for line in open(SAMPLES_2D):
        x = json.loads(line)
        if x.get("doc_id") == 127:
            m = json.loads(x["arguments"]["gen_args_0"]["arg_0"][0])
            h = hashlib.sha256(json.dumps(m, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            return (m, None) if h == DOC127_MSG_SHA else (None, "doc 127 messages hash differs")
    return None, "doc 127 not found"


def running_requests():
    try:
        t = requests.get(f"{BASE}/metrics", timeout=5).text
    except Exception:  # noqa: BLE001
        return None
    n = 0.0
    for line in t.splitlines():
        if line.startswith(("vllm:num_requests_running", "vllm:num_requests_waiting")):
            try:
                n += float(line.rsplit(" ", 1)[1])
            except Exception:  # noqa: BLE001
                return None
    return n


def drain(timeout=45.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        n = running_requests()
        if n is None:
            time.sleep(3.0)
            return dict(idle=None, secs=round(time.time() - t0, 2))
        if n == 0:
            return dict(idle=True, secs=round(time.time() - t0, 2))
        time.sleep(0.25)
    return dict(idle=False, secs=round(time.time() - t0, 2))


def background(j, seed, st, stop):
    """Streaming completion kept in decode until `stop` is set; then the stream is closed (abort)."""
    r = BYID[D2_BG]
    body = dict(model=MODEL, prompt=r["ids"], max_tokens=1024, temperature=1.0, top_p=0.95, seed=seed, stream=True)
    st[j] = dict(seed=seed, n=0, t_first=None, t_second=None, error=None)
    try:
        with requests.post(f"{BASE}/v1/completions", json=body, stream=True, timeout=600) as resp:
            for line in resp.iter_lines():
                if stop.is_set():
                    break
                if not line or not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                st[j]["n"] += 1
                if st[j]["t_first"] is None:
                    st[j]["t_first"] = time.time()
                elif st[j]["t_second"] is None:
                    st[j]["t_second"] = time.time()
    except Exception as e:  # noqa: BLE001
        st[j]["error"] = repr(e)[:200]


def phase_2d():
    msgs, why = doc127_messages()
    if msgs is None:
        rec(dict(kind="skipped", what="2d", why=why, t=round(time.time(), 3)))
        log(f"2d skipped: {why}")
        return
    it = GQ[0]                                   # GPQA request parameters (template kwargs, top_p)
    t0 = time.time()
    rec(dict(kind="phase2d_start", t=round(t0, 3), bg=D2_BG, doc=127, doc_tokens_expected=DOC127_TOKENS,
             doc_messages_sha256=DOC127_MSG_SHA))
    for D in range(8):
        if not need(D2_CASE_S, f"2d case D={D}"):
            break
        stop = threading.Event()
        st = {}
        seeds = [D2_SEED_BG + 10 * D + j for j in range(D)]
        ths = [threading.Thread(target=background, args=(j, seeds[j], st, stop), daemon=True) for j in range(D)]
        [t.start() for t in ths]
        t_w = time.time()
        while time.time() - t_w < 60 and not all(st.get(j, {}).get("t_second") or st.get(j, {}).get("error")
                                                 for j in range(D)):
            time.sleep(0.05)
        in_dec = sum(1 for j in range(D) if st.get(j, {}).get("t_second"))
        body = dict(model=MODEL, messages=msgs, max_tokens=8, temperature=1.0, top_p=it["top_p"],
                    seed=D2_SEED_DOC + D, chat_template_kwargs=it["chat_template_kwargs"])
        j_, err, ts, td = _post("/v1/chat/completions", body)
        u = j_.get("usage") or {}
        ch = (j_.get("choices") or [{}])[0]
        stop.set()
        [t.join(timeout=20) for t in ths]
        dr = drain()
        rec(dict(kind="phase2d_case", case=D, requested_D=D, bg_in_decode=in_dec, seeds=dict(bg=seeds, doc=D2_SEED_DOC + D),
                 t_send=round(ts, 3), t_done=round(td, 3), prompt_tokens=u.get("prompt_tokens"),
                 completion_tokens=u.get("completion_tokens"), finish=ch.get("finish_reason"), error=err,
                 bg=[dict(seed=st[j]["seed"], streamed=st[j]["n"], error=st[j]["error"]) for j in range(D) if j in st],
                 drain=dr))
        log(f"2d D={D} in_decode={in_dec} prompt_tokens={u.get('prompt_tokens')} err={err} drain={dr}")
    phase_rec("2d", t0, bg=D2_BG, max_tokens_doc=8)


if "2d" in PHASES:
    dump("pre2d", wait=6.0)                      # readout that closes the frozen window (B1-5)
    if need(D2_NEED_S, "2d"):
        phase_2d()
dump("final", wait=6.0)
log("CLIENT DONE")
