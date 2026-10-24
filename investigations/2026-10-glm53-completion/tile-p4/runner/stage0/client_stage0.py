#!/usr/bin/env python3
"""Stage-0 client (STAGE0-PROTOCOL.md sections 2-3).  Host python3, stdlib only.

usage: client_stage0.py AO DEADLINE_EPOCH BASE_URL      env: P4_WINDOWS, P4_SCHED, S0_BATCH (default 1),
                                                             optional P4_PLAN_ONLY=1, S0_ONLY (e.g. "V1-r1,V1-r2")

Arm A only, ONE batch of 8 windows (default batch 1 of the frozen plan).  Four independently warmed replicates, in this
fixed order: V1-r1, V1-r2, V2-r1, V2-r2.  Every replicate:
  salt   cache_salt "s0-<variant>-r<rep>": no replicate can reuse another replicate's cached prefix (it does NOT reset
         the allocator's history, the physical block ids, or D2's shared tail block; those are recorded, not removed).
  warm   one request per reply of the batch, in plan order, each awaited before the next (solitary chunked prefill):
         prompt = the reply's longest prefix needed by the batch, max_tokens 1, seed SEED_WARM.  Warm requests are
         KEYED with an empty forced list (forcewin_hooks [], tag "W-<variant>-r<rep>-<reply>") so the engine-side
         recorder logs their steps and block ids (provenance); nothing is forced or stored for them.
  forced V1 (solitary): the 8 windows one at a time, each awaited; order = replies in plan order, within a reply
         descending P (close, deep, early), so no forced request computes blocks inside a later window's cached span.
         V2 (co-batched): ONE multi-prompt request carrying the 8 prompts in plan order; every engine request picks
         its own forced list and tag by its prompt key (local_fw_multi).  The engine-side enqueue order and the
         scheduler's execution are recorded by the server (sched log); the client records the submission.
         Both: draft mode "list" with the fixed acceptance list, max_tokens 512, T 1.0, top_p 0.95,
         seed SEED for every forced request of every replicate (exact repeats share the seed), return_token_ids.
  sentinel an unkeyed 4-token request (salt "s0-sentinel") so the engine removes, and the hooks flush, the last
         finished requests before the next replicate starts.  The next replicate's warm starts after it returns.
A replicate that cannot finish before the deadline is skipped and recorded (replicates run in the fixed order, so a
short session loses the trailing replicates); a STOP file in AO ends the session after the current replicate.

Writes AO/client.jsonl (public): per request kind, replicate, tags, order, send / done times, status, token counts,
finish reasons, ids_equal per choice, sha256 of the returned ids.  No token ids, no text.
"""
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

W = 512
SEED = 73000
SEED_WARM = 72999
REPLICATES = [("V1", 1), ("V1", 2), ("V2", 1), ("V2", 2)]
EST_S = {"V1": 320.0, "V2": 202.0}         # runner/stage0/cost_stage0.json, planning case
KIND_ORDER = {"close": 0, "deep": 1, "early": 2}
BLOCK = 4352


def multi_key(prompt_ids):
    """Identical to forcewin_hooks.multi_key (stage-0 port P16)."""
    p = [int(x) for x in prompt_ids]
    return "%d:%s:%s" % (len(p), ",".join(map(str, p[:8])), ",".join(map(str, p[-8:])))


def cached_expected(P):
    return max(0, ((P - 1) // BLOCK - 1) * BLOCK)


def sha_ids(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest() if ids else None


def batch_plan(fx, batch):
    bp = next(b for b in fx["plan"] if int(b["batch"]) == int(batch))
    replies = fx["replies"]
    wins = []
    for wid in bp["windows"]:
        rid, kind = wid.split("-")
        P = int(replies[rid]["windows"][kind]["P"])
        wins.append(dict(wid=wid, reply=rid, kind=kind, P=P))
    reply_order = []
    for w in wins:
        if w["reply"] not in reply_order:
            reply_order.append(w["reply"])
    v1 = sorted(wins, key=lambda w: (reply_order.index(w["reply"]), KIND_ORDER.get(w["kind"], 9)))
    warm = [dict(reply=w["reply"], P=int(w["P"])) for w in bp["warm"]]
    return bp, wins, v1, warm


class Client:
    def __init__(self, ao, deadline, base):
        self.ao, self.deadline, self.base = ao, float(deadline), base.rstrip("/")
        self.fh = open(os.path.join(ao, "client.jsonl"), "a")
        self.model = None

    def rec(self, d):
        d = dict(d, t=round(time.time(), 3))
        self.fh.write(json.dumps(d) + "\n")
        self.fh.flush()

    def left(self):
        return self.deadline - time.time()

    def post(self, body, timeout):
        req = urllib.request.Request(self.base + "/v1/completions", data=json.dumps(body).encode(),
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

    def warm(self, rname, salt, reply, ids, P):
        tag = f"W-{rname}-{reply}"
        body = dict(model=self.model, prompt=list(ids[:P]), max_tokens=1, temperature=1.0, top_p=0.95,
                    seed=SEED_WARM, cache_salt=salt,
                    vllm_xargs={"forcewin_hooks": [], "local_fw_tag": tag})
        j, err, t0, t1, st = self.post(body, self.left())
        u = (j or {}).get("usage") or {}
        self.rec(dict(kind="warm", replicate=rname, reply=reply, tag=tag, salt=salt, status=st, error=err,
                      t_send=round(t0, 3), t_done=round(t1, 3), prompt_tokens=u.get("prompt_tokens"),
                      cached_tokens=((u.get("prompt_tokens_details") or {}).get("cached_tokens")),
                      ok=err is None and u.get("prompt_tokens") == P))
        return err is None

    def forced_one(self, rname, salt, w, prompt, F, sched, order):
        tag = f"{rname.split('-')[0]}-{w['wid']}-r{rname.split('-r')[1]}"
        body = dict(model=self.model, prompt=list(prompt), max_tokens=len(F), temperature=1.0, top_p=0.95,
                    seed=SEED, cache_salt=salt, return_token_ids=True,
                    vllm_xargs={"forcewin_hooks": list(F), "local_fw_tag": tag, "local_fw_draft": "list",
                                "local_fw_sched": list(sched)})
        j, err, t0, t1, st = self.post(body, self.left())
        ch = ((j or {}).get("choices") or [{}])[0]
        u = (j or {}).get("usage") or {}
        got = ch.get("token_ids")
        eq = got is not None and list(got) == list(F)
        d = dict(kind="forced", replicate=rname, variant="V1", window=w["wid"], tag=tag, order=order, seed=SEED,
                 salt=salt, status=st, error=err, t_send=round(t0, 3), t_done=round(t1, 3),
                 prompt_tokens=u.get("prompt_tokens"), prompt_ok=u.get("prompt_tokens") == len(prompt),
                 cached_tokens=((u.get("prompt_tokens_details") or {}).get("cached_tokens")),
                 cached_expected=cached_expected(len(prompt)), completion_tokens=u.get("completion_tokens"),
                 finish_reason=ch.get("finish_reason"), ids_equal=eq, returned_ids_sha256=sha_ids(got),
                 ok=err is None and eq and ch.get("finish_reason") == "length")
        self.rec(d)
        return d["ok"]

    def forced_multi(self, rname, salt, wins, prompts, Fs, sched):
        var, rep = rname.split("-r")
        tags = [f"{var}-{w['wid']}-r{rep}" for w in wins]
        keys = [multi_key(p) for p in prompts]
        if len(set(keys)) != len(keys):
            self.rec(dict(kind="refused", replicate=rname, reason="prompt keys not unique"))
            return False
        spec = json.dumps(dict(entries=[dict(key=k, F=list(F), tag=t) for k, F, t in zip(keys, Fs, tags)]))
        body = dict(model=self.model, prompt=[list(p) for p in prompts], max_tokens=W, temperature=1.0, top_p=0.95,
                    seed=SEED, cache_salt=salt, return_token_ids=True,
                    vllm_xargs={"local_fw_multi": spec, "local_fw_draft": "list", "local_fw_sched": list(sched)})
        j, err, t0, t1, st = self.post(body, self.left())
        chs = sorted((j or {}).get("choices") or [], key=lambda c: int(c.get("index", 0)))
        u = (j or {}).get("usage") or {}
        oks = 0
        for i, (w, F, tag) in enumerate(zip(wins, Fs, tags)):
            ch = chs[i] if i < len(chs) else {}
            got = ch.get("token_ids")
            eq = got is not None and list(got) == list(F)
            ok = err is None and eq and ch.get("finish_reason") == "length"
            oks += int(ok)
            self.rec(dict(kind="forced", replicate=rname, variant="V2", window=w["wid"], tag=tag, order=i, seed=SEED,
                          salt=salt, status=st, error=err, t_send=round(t0, 3), t_done=round(t1, 3),
                          prompt_tokens_P=len(prompts[i]), finish_reason=ch.get("finish_reason"), ids_equal=eq,
                          returned_ids_sha256=sha_ids(got), ok=ok))
        self.rec(dict(kind="multi", replicate=rname, n_prompts=len(prompts), status=st, error=err,
                      t_send=round(t0, 3), t_done=round(t1, 3), prompt_tokens=u.get("prompt_tokens"),
                      cached_tokens=((u.get("prompt_tokens_details") or {}).get("cached_tokens")),
                      completion_tokens=u.get("completion_tokens"), choices=len(chs), ok=oks,
                      order=[w["wid"] for w in wins]))
        return oks == len(wins)

    def sentinel(self, rname):
        j, err, t0, t1, st = self.post(dict(model=self.model, prompt=[1, 2, 3, 4], max_tokens=1, temperature=0.0,
                                            cache_salt="s0-sentinel"), 60)
        self.rec(dict(kind="sentinel", replicate=rname, status=st, error=err))

    def run(self, fx, sched, batch, only=None):
        bp, wins, v1, warm = batch_plan(fx, batch)
        replies = fx["replies"]
        self.rec(dict(kind="start", batch=batch, windows=[w["wid"] for w in wins], v1_order=[w["wid"] for w in v1],
                      warm=[(x["reply"], x["P"]) for x in warm], replicates=[f"{v}-r{r}" for v, r in REPLICATES],
                      W=W, seed=SEED, sched_len=len(sched),
                      sched_sha256=hashlib.sha256(json.dumps(sched).encode()).hexdigest(), deadline=self.deadline))
        self.get_model()
        done, ratio_num, ratio_den = [], 0.0, 0.0
        for var, rep in REPLICATES:
            rname = f"{var}-r{rep}"
            if only and rname not in only:
                continue
            if os.path.exists(os.path.join(self.ao, "STOP")):
                self.rec(dict(kind="skip", replicate=rname, reason="STOP file"))
                break
            ratio = min(2.0, max(0.6, ratio_num / ratio_den)) if ratio_den > 0 else 1.0
            est = EST_S[var] * ratio
            if time.time() + est > self.deadline:
                self.rec(dict(kind="skip", replicate=rname, reason="time", est_s=round(est), left_s=round(self.left())))
                break
            salt = f"s0-{var}-r{rep}"
            t_r = time.time()
            self.rec(dict(kind="replicate", replicate=rname, phase="warm", salt=salt, est_s=round(est)))
            ok = all(self.warm(rname, salt, x["reply"], replies[x["reply"]]["ids"], x["P"]) for x in warm)
            self.rec(dict(kind="replicate", replicate=rname, phase="forced", warm_ok=ok))
            if var == "V1":
                fok = 0
                for k, w in enumerate(v1):
                    ids = replies[w["reply"]]["ids"]
                    fok += int(self.forced_one(rname, salt, w, ids[:w["P"]], ids[w["P"]:w["P"] + W], sched, k))
                fok = fok == len(v1)
            else:
                prompts = [replies[w["reply"]]["ids"][:w["P"]] for w in wins]
                Fs = [replies[w["reply"]]["ids"][w["P"]:w["P"] + W] for w in wins]
                fok = self.forced_multi(rname, salt, wins, prompts, Fs, sched)
            self.sentinel(rname)
            secs = time.time() - t_r
            ratio_num += secs
            ratio_den += EST_S[var]
            self.rec(dict(kind="replicate_done", replicate=rname, secs=round(secs, 1), est0_s=EST_S[var],
                          warm_ok=ok, forced_ok=bool(fok)))
            done.append(rname)
        self.rec(dict(kind="end", replicates_done=done))
        want = [f"{v}-r{r}" for v, r in REPLICATES if not only or f"{v}-r{r}" in only]
        return 0 if done == want else 4


def main():
    ao, deadline, base = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    fx = json.load(open(os.environ["P4_WINDOWS"]))
    sched = json.load(open(os.environ["P4_SCHED"]))["sched"]
    batch = int(os.environ.get("S0_BATCH", "1"))
    only = [x for x in os.environ.get("S0_ONLY", "").split(",") if x]
    if os.environ.get("P4_PLAN_ONLY") == "1":
        bp, wins, v1, warm = batch_plan(fx, batch)
        keys = [multi_key(fx["replies"][w["reply"]]["ids"][:w["P"]]) for w in wins]
        print(json.dumps(dict(batch=batch, windows=[w["wid"] for w in wins], v1_order=[w["wid"] for w in v1],
                              warm_prefixes=len(warm), warm_tokens=sum(x["P"] for x in warm),
                              replicates=[f"{v}-r{r}" for v, r in REPLICATES], prompt_keys_unique=len(set(keys)) == 8,
                              est_s=sum(EST_S[v] for v, _ in REPLICATES))))
        return 0
    c = Client(ao, deadline, base)
    return c.run(fx, sched, batch, only)


if __name__ == "__main__":
    sys.exit(main())
