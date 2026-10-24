#!/usr/bin/env python3
"""P4 session client (PROPOSAL.md section 4; PROPOSAL-AMENDMENTS.md).  Host python3, stdlib only.

usage: client_p4.py AO DEADLINE_EPOCH BASE_URL SESSION     (env: P4_WINDOWS, P4_SCHED; optional P4_PLAN_ONLY=1)

For each batch of the frozen plan (PRIVATE-fixtures/windows.json; 6 batches x 8 windows, identical in every session):
  1. warm: one unkeyed request per reply in the batch, prompt = that reply's longest prefix needed by the batch,
     max_tokens 1 (fills the prefix cache; its token is discarded);
  2. rep 1: the 8 forced requests, released together from a barrier (8 in flight), each
       prompt = prompt ids + reasoning ids[:o_start]   (P tokens, built by prefill)
       forced = reasoning ids[o_start : o_start + 512] (forcewin_hooks), draft mode "list" with the fixed `sched`
       list (local_fw_sched), max_tokens 512, T 1.0, top_p 0.95 (the GPQA sampler: the recorded post-mask
       probabilities and nucleus eligibility are those of the served sampler), seed = 73000 + 2 * window + rep
       (the same in every session), cache_salt "p4", return_token_ids;
  3. rep 2: the same 8 requests again (fresh requests; the prefix cache serves the prefill-built prefix).
Before a batch starts, its time is estimated (prefill at RATE tok/s for the warm prefixes and both passes'
uncached remainders, plus DECODE_S per pass), scaled by the ratio of measured to estimated time of the batches
already run (clamped to 0.6-2.0); a batch that cannot finish before the deadline is skipped and recorded, so a
session never runs past its cap and every session covers the same leading batches.  A STOP file in
AO ends the session after the current batch.  At the end one unkeyed sentinel request makes the engine remove the
last finished requests (their stores are flushed at removal).

Writes AO/client.jsonl (public): per request tag, window, rep, seed, timing, prompt/cached/completion token counts,
finish reason, ids_equal (returned ids == forced ids), errors.  No token ids, no text, no positions.
"""
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

W = 512
RATE = 2400.0          # planning prefill rate (tok/s; measured on B1 2,750-2,770 tok/s at 70k-120k)
DECODE_S = 20.0        # planning time of one 8-request forced pass (512 positions at ~3.06 tokens/step;
                       # first-contrast forced decode on B1: 34 tok/s/request -> 15.1 s)
BLOCK = 4352
SEED0 = 73000
SALT = "p4"


def cached_expected(P):
    return max(0, ((P - 1) // BLOCK - 1) * BLOCK)


class Client:
    def __init__(self, ao, deadline, base, session):
        self.ao, self.deadline, self.base, self.session = ao, float(deadline), base.rstrip("/"), session
        self.lock = threading.Lock()
        self.fh = open(os.path.join(ao, "client.jsonl"), "a")
        self.model = None

    def rec(self, d):
        d = dict(d, t=round(time.time(), 3))
        with self.lock:
            self.fh.write(json.dumps(d) + "\n")
            self.fh.flush()

    def post(self, path, body, timeout):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=max(5.0, timeout)) as r:
                return json.loads(r.read().decode()), None, t0, time.time(), r.status
        except urllib.error.HTTPError as e:
            return None, f"HTTP {e.code}", t0, time.time(), e.code
        except Exception as e:  # noqa: BLE001
            return None, repr(e)[:200], t0, time.time(), None

    def get_model(self):
        with urllib.request.urlopen(self.base + "/v1/models", timeout=30) as r:
            self.model = json.loads(r.read().decode())["data"][0]["id"]

    def left(self):
        return self.deadline - time.time()

    def warm(self, batch, reply, ids, P):
        body = dict(model=self.model, prompt=list(ids[:P]), max_tokens=1, temperature=1.0, top_p=0.95,
                    seed=SEED0 - 1, cache_salt=SALT)
        j, err, t0, t1, st = self.post("/v1/completions", body, self.left())
        u = (j or {}).get("usage") or {}
        self.rec(dict(kind="warm", session=self.session, batch=batch, reply=reply, status=st, error=err,
                      secs=round(t1 - t0, 2), prompt_tokens=u.get("prompt_tokens"),
                      cached_tokens=((u.get("prompt_tokens_details") or {}).get("cached_tokens")),
                      ok=err is None and u.get("prompt_tokens") == P))
        return err is None

    def forced(self, batch, widx, wid, rep, prompt, F, sched, barrier, results):
        tag = f"{self.session}-{wid}-r{rep}"
        seed = SEED0 + 2 * widx + rep
        body = dict(model=self.model, prompt=list(prompt), max_tokens=len(F), temperature=1.0, top_p=0.95,
                    seed=seed, cache_salt=SALT, return_token_ids=True,
                    vllm_xargs={"forcewin_hooks": list(F), "local_fw_tag": tag, "local_fw_draft": "list",
                                "local_fw_sched": list(sched)})
        try:
            barrier.wait(timeout=60)
        except threading.BrokenBarrierError:
            pass
        j, err, t0, t1, st = self.post("/v1/completions", body, self.left())
        ch = ((j or {}).get("choices") or [{}])[0]
        u = (j or {}).get("usage") or {}
        got = ch.get("token_ids")
        ids_equal = got is not None and list(got) == list(F)
        d = dict(kind="forced", session=self.session, batch=batch, window=wid, widx=widx, rep=rep, tag=tag, seed=seed,
                 status=st, error=err, t0=round(t0, 3), t1=round(t1, 3), secs=round(t1 - t0, 2),
                 prompt_tokens=u.get("prompt_tokens"), prompt_ok=u.get("prompt_tokens") == len(prompt),
                 cached_tokens=((u.get("prompt_tokens_details") or {}).get("cached_tokens")),
                 cached_expected=cached_expected(len(prompt)), completion_tokens=u.get("completion_tokens"),
                 finish_reason=ch.get("finish_reason"), ids_equal=ids_equal,
                 returned_ids_sha256=hashlib.sha256(",".join(map(str, got)).encode()).hexdigest() if got else None,
                 ok=err is None and ids_equal and ch.get("finish_reason") == "length")
        self.rec(d)
        results[tag] = d

    def run(self, fx, sched):
        replies, plan = fx["replies"], fx["plan"]
        order = [w for bp in plan for w in bp["windows"]]
        widx_of = {w: i for i, w in enumerate(order)}
        self.rec(dict(kind="start", session=self.session, deadline=self.deadline, n_batches=len(plan),
                      n_windows=len(order), W=W, sched_len=len(sched),
                      sched_sha256=hashlib.sha256(json.dumps(sched).encode()).hexdigest()))
        self.get_model()
        done_batches = []
        est_sum = act_sum = 0.0
        for bp in plan:
            b = bp["batch"]
            if os.path.exists(os.path.join(self.ao, "STOP")):
                self.rec(dict(kind="skip", batch=b, reason="STOP file"))
                break
            est0 = (bp["warm_tokens"] + 2 * sum(bp["remainder_tokens"])) / RATE + 2 * DECODE_S
            ratio = min(2.0, max(0.6, act_sum / est_sum)) if est_sum > 0 else 1.0
            est = est0 * ratio
            if time.time() + est > self.deadline:
                self.rec(dict(kind="skip", batch=b, reason="time", est_s=round(est), ratio=round(ratio, 3),
                              left_s=round(self.left())))
                break
            tb = time.time()
            self.rec(dict(kind="batch", batch=b, phase="warm", est_s=round(est), windows=bp["windows"]))
            ws = [threading.Thread(target=self.warm, args=(b, w["reply"], replies[w["reply"]]["ids"], w["P"]))
                  for w in bp["warm"]]
            for t in ws:
                t.start()
            for t in ws:
                t.join()
            for rep in (1, 2):
                if self.left() < 30:
                    self.rec(dict(kind="skip", batch=b, rep=rep, reason="deadline"))
                    break
                barrier = threading.Barrier(len(bp["windows"]))
                results, ths = {}, []
                for wid in bp["windows"]:
                    rid, kind = wid.split("-")
                    r = replies[rid]
                    P = r["windows"][kind]["P"]
                    F = r["ids"][P:P + W]
                    ths.append(threading.Thread(target=self.forced, args=(b, widx_of[wid], wid, rep, r["ids"][:P], F,
                                                                          sched, barrier, results)))
                for t in ths:
                    t.start()
                for t in ths:
                    t.join()
                self.rec(dict(kind="pass", batch=b, rep=rep, n=len(results),
                              ok=sum(1 for d in results.values() if d["ok"]),
                              secs=round(max([d["t1"] for d in results.values()] or [time.time()]) -
                                         min([d["t0"] for d in results.values()] or [time.time()]), 1)))
            done_batches.append(b)
            est_sum += est0
            act_sum += time.time() - tb
            self.rec(dict(kind="batch_done", batch=b, secs=round(time.time() - tb, 1), est0_s=round(est0)))
        # sentinel: an unkeyed request so the engine removes (and the hooks flush) the last finished requests
        j, err, t0, t1, st = self.post("/v1/completions", dict(model=self.model, prompt=[1, 2, 3, 4], max_tokens=1,
                                                              temperature=0.0, cache_salt="p4-sentinel"), 60)
        self.rec(dict(kind="sentinel", status=st, error=err))
        self.rec(dict(kind="end", batches_done=done_batches, n_batches=len(plan)))
        return 0 if len(done_batches) == len(plan) else 4


def main():
    ao, deadline, base, session = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
    fx = json.load(open(os.environ["P4_WINDOWS"]))
    sched = json.load(open(os.environ["P4_SCHED"]))["sched"]
    if os.environ.get("P4_PLAN_ONLY") == "1":
        out = []
        for bp in fx["plan"]:
            out.append(dict(batch=bp["batch"], windows=bp["windows"], warm_prefixes=len(bp["warm"]),
                            est_s=round((bp["warm_tokens"] + 2 * sum(bp["remainder_tokens"])) / RATE + 2 * DECODE_S)))
        print(json.dumps(dict(batches=out, total_est_s=sum(b["est_s"] for b in out))))
        return 0
    c = Client(ao, deadline, base, session)
    return c.run(fx, sched)


if __name__ == "__main__":
    sys.exit(main())
