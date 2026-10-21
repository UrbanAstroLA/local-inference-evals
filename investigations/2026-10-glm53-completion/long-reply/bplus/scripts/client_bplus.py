#!/usr/bin/env python3
"""B+ continuation client for ONE engine session (PROPOSAL.md section 7). Host python, standard library only.
Runs only from runner/run_bplus.sh after the operator's GPU authorization, against the server it started on
127.0.0.1:8001. Never retries a request; never stops early on content.

usage: client_bplus.py ENGINE RUN_DIR [--base URL]
  ENGINE   v070 | v091 | b1 (seeds from out/plan.json for this engine)
  RUN_DIR  the run folder; writes RUN_DIR/ENGINE/requests.jsonl (ids, seeds, hashes, counts, timings; no text) and
           RUN_DIR/raw/ENGINE.raw.jsonl.gz (generated text and token ids; mode 0600 in a 0700 folder; never published)
Behaviour
  - first asks the server to render each original chat prompt (/v1/chat/completions/render, no generation) and records
    whether its ids equal the prefix's prompt part (template check on the serving engine itself; recorded, not a refusal)
  - refuses to send anything unless every prefix in PRIVATE/prefixes.json hashes to the frozen manifest, the plan's
    params hash matches, and the server's /v1/models lists the expected model
  - 8 requests in flight; requests are submitted in the plan's fixed order k = 1..48; when one completes, the next is
    submitted (the same order on every engine)
  - body: model, prompt (token ids), max_tokens 65,536, temperature 1.0, top_p 0.95, seed, stop ["</s>"],
    return_token_ids true, stream false; nothing else (server defaults otherwise, identical in chat and completions)
  - per request: k, prefix id, j, seed, sha256 of the sent ids, sha256 of the server-echoed prompt ids, HTTP status,
    finish_reason, stop_reason, prompt / completion / cached tokens, position of the first </think> in the generated
    ids, sha256 of the generated ids and text, character count, submit / end times
  - exit 0 only if all 48 returned HTTP 200 with a choice; 2 refusal before sending; 3 one or more requests failed
"""
import gzip
import hashlib
import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request

os.umask(0o077)
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ENGINE, RUN = sys.argv[1], sys.argv[2]
BASE = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "http://127.0.0.1:8001"
CONC = 8
TIMEOUT = 10800
THINK_CLOSE = 154842
sys.path.insert(0, HERE)
import plan as PL  # noqa: E402


def ids_sha(ids):
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


def main():
    if ENGINE not in ("v070", "v091", "b1"):
        print("usage: client_bplus.py v070|v091|b1 RUN_DIR")
        return 2
    plan = json.load(open(f"{PKG}/out/plan.json"))
    man = {m["prefix_id"]: m for m in json.load(open(f"{PKG}/out/prefix_manifest.json"))["prefixes"]}
    pre = json.load(open(f"{PKG}/PRIVATE/prefixes.json"))["prefixes"]
    bad = [p for p, m in man.items() if ids_sha(pre[p]) != m["prefix_ids_sha256"] or len(pre[p]) != m["prefix_len"]]
    if bad or plan["params_sha256"] != PL.params_sha(plan["model"], plan["params"]) or plan["params"] != PL.PARAMS:
        print(f"REFUSE: prefix hashes or plan params do not match the frozen files ({bad})")
        return 2
    out = os.path.join(RUN, ENGINE)
    os.makedirs(out, exist_ok=True)
    rawdir = os.path.join(RUN, "raw")
    os.makedirs(rawdir, exist_ok=True)
    os.chmod(rawdir, 0o700)
    log_path = os.path.join(out, "requests.jsonl")
    raw_path = os.path.join(rawdir, f"{ENGINE}.raw.jsonl.gz")
    if os.path.exists(log_path) or os.path.exists(raw_path):
        print("REFUSE: this engine's request log or raw file already exists (no retry)")
        return 2
    lock = threading.Lock()

    def rec(d):
        with lock:
            with open(log_path, "a") as f:
                f.write(json.dumps(d, sort_keys=True) + "\n")

    def raw(d):
        with lock:
            with gzip.open(raw_path, "ab") as f:      # one gzip member per record: a crash loses at most one record
                f.write((json.dumps(d) + "\n").encode())
            os.chmod(raw_path, 0o600)

    try:
        with urllib.request.urlopen(f"{BASE}/v1/models", timeout=30) as r:
            models = [m["id"] for m in json.load(r)["data"]]
    except Exception as e:  # noqa: BLE001
        print(f"REFUSE: /v1/models unreachable ({e!r})")
        return 2
    if plan["model"] not in models:
        print(f"REFUSE: server model {models} is not {plan['model']}")
        return 2
    rec({"kind": "start", "engine": ENGINE, "t": round(time.time(), 3), "base": BASE, "model": plan["model"],
         "params_sha256": plan["params_sha256"], "concurrency": CONC, "planned": len(plan["order"])})
    # run-time template check (no generation): the server's own chat rendering of each original prompt vs the prompt
    # part of the prefix that will be sent. Recorded, not a refusal: every engine receives the 0.7.0-rendered ids.
    prompts = json.load(open(f"{PKG}/PRIVATE/prompts.json"))["prompts"]
    for p in sorted(man):
        pl = man[p]["prompt_len"]
        try:
            body = {"model": plan["model"], "messages": prompts[p]["messages"], **({"chat_template_kwargs": prompts[p]["chat_template_kwargs"]}
                                                                                  if prompts[p]["chat_template_kwargs"] else {})}
            req = urllib.request.Request(f"{BASE}/v1/chat/completions/render", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                tid = json.load(resp).get("token_ids") or []
            rec({"kind": "render_check", "prefix_id": p, "n_tokens": len(tid), "equal": tid == pre[p][:pl],
                 "rendered_ids_sha256": ids_sha(tid), "error": None})
        except Exception as e:  # noqa: BLE001
            rec({"kind": "render_check", "prefix_id": p, "equal": None, "error": repr(e)[:200]})
    q = queue.Queue()
    for r in plan["order"]:
        q.put(r)
    failures = []

    def worker():
        while True:
            try:
                r = q.get_nowait()
            except queue.Empty:
                return
            p = r["prefix_id"]
            seed = r["seeds"][ENGINE]
            ids = pre[p]
            body = {"model": plan["model"], "prompt": ids, "seed": seed, **plan["params"]}
            sent_sha = ids_sha(body["prompt"])
            d = {"kind": "request", "engine": ENGINE, "k": r["k"], "prefix_id": p, "j": r["j"], "seed": seed,
                 "prompt_len": len(ids), "prompt_ids_sha256": sent_sha, "params_sha256": PL.params_sha(body["model"], {
                     k: v for k, v in body.items() if k not in ("model", "prompt", "seed")})}
            t0 = time.time()
            status, j, err = None, {}, None
            try:
                req = urllib.request.Request(f"{BASE}/v1/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                    status = resp.status
                    j = json.load(resp)
            except urllib.error.HTTPError as e:
                status, err = e.code, f"HTTPError {e.code}"
            except Exception as e:  # noqa: BLE001
                err = repr(e)[:300]
            t1 = time.time()
            ch = (j.get("choices") or [{}])[0] if isinstance(j, dict) else {}
            u = (j.get("usage") or {}) if isinstance(j, dict) else {}
            gen = ch.get("token_ids") or []
            ptid = ch.get("prompt_token_ids") or (j.get("prompt_token_ids") if isinstance(j, dict) else None)
            text = ch.get("text") or ""
            tc = gen.index(THINK_CLOSE) if THINK_CLOSE in gen else None
            d.update(t0=round(t0, 3), t1=round(t1, 3), secs=round(t1 - t0, 2), http_status=status, error=err,
                     resp_id=j.get("id") if isinstance(j, dict) else None, finish_reason=ch.get("finish_reason"),
                     stop_reason=ch.get("stop_reason"), prompt_tokens=u.get("prompt_tokens"),
                     completion_tokens=u.get("completion_tokens"),
                     cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                     returned_prompt_ids_sha256=ids_sha(ptid) if ptid else None, gen_ids_len=len(gen),
                     gen_ids_sha256=ids_sha(gen), text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                     text_chars=len(text), think_close_at=tc)
            if status != 200 or err or not ch:
                failures.append(r["k"])
            raw({"engine": ENGINE, "k": r["k"], "prefix_id": p, "j": r["j"], "seed": seed, "text": text, "token_ids": gen})
            rec(d)

    th = [threading.Thread(target=worker, daemon=True) for _ in range(CONC)]
    for t in th:
        t.start()
        time.sleep(0.2)          # fixed small stagger: submission order follows k within the first wave
    for t in th:
        t.join()
    rec({"kind": "end", "engine": ENGINE, "t": round(time.time(), 3), "failures": failures})
    print(f"client {ENGINE}: {len(plan['order']) - len(failures)} / {len(plan['order'])} requests returned 200; failures {failures}")
    return 3 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
