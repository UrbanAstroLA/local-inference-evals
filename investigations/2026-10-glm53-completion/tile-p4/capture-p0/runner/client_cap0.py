#!/usr/bin/env python3
"""capture-p0 client (PROTOCOL.md section 3).  Host python3, stdlib only.

usage: client_cap0.py AO DEADLINE_EPOCH BASE_URL
env:   CAP0_WINDOWS   the frozen window fixture (PRIVATE windows.json; R01's ids)
       CAP0_REPS      replicate plan, default "r1:L,r2:L,r3:N,r4:N,r5:L,r6:L" (L = localisation on for its warm)
       CAP0_LOC       "on" (default) | "off": with "off" every replicate is N (the hooks are not installed)
       CAP0_PLAN_ONLY=1   print the plan (counts and hashes only) and exit

Per replicate, in order, each request awaited before the next (nothing else is in flight):
  warm     prompt = R01 ids[:17,409] (the full capture prompt; amendment A1), max_tokens 1, T 1.0, top_p 0.95, seed 72,999, cache_salt "cap0-<rk>"; keyed with
           an empty forced list and tag "CW-<rk>-<L|N>" (the engine-side recorder logs its steps and blocks; the
           localisation hooks record it when the tag ends in "-L").  Expected: cached_tokens 0.  It caches 4 blocks;
           the capture request then gets 4 - 1 = 3 (one block is dropped on a hit: PROPOSAL-AMENDMENTS.md A3).
  capture  prompt = R01 ids[:17,409] (the R01-early window prompt), max_tokens 1, seed 73,000, same salt, keyed with
           an empty forced list and tag "CAP-<rk>": its first step is point 0 (n_computed 13,056).  Expected:
           cached_tokens 13,056.
  sentinel an unkeyed 4-token request (salt "cap0-sentinel") so the engine removes the capture request and the
           hooks flush before the next replicate.
Replicates run in the fixed order; one that cannot finish before the deadline is skipped and recorded (a STOP file
in AO ends the session after the current replicate).  EARLY ABORT: if the FIRST replicate's warm is not uncached or
its capture request does not get exactly 13,056 cached tokens, the construction is wrong for every replicate: the
client records kind "abort" with the reason and ends the session at once (exit 5).
CAP0_TEST_WARM_TOKENS (honoured only with CAP0_TEST_MOCK=1; the runner never sets either) replaces the warm length,
so the runner tests can replay the attempt-1 construction against the mock.  Writes AO/client.jsonl (public): kinds, tags, salts, status,
token counts, cached tokens, sha256 of the prompt ids sent.  No token ids, no text.
"""
import hashlib
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("LY", os.path.join(HERE, "..", "build", "files", "vllm", "cap0_hooks_layout.py"))
LY = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LY)
SEED_WARM, SEED_CAP = 72999, 73000
WARM_TOKENS = LY.WARM_TOKENS
if os.environ.get("CAP0_TEST_MOCK") == "1" and os.environ.get("CAP0_TEST_WARM_TOKENS"):
    WARM_TOKENS = int(os.environ["CAP0_TEST_WARM_TOKENS"])
DEFAULT_REPS = "r1:L,r2:L,r3:N,r4:N,r5:L,r6:L"
EST_S = {"L": 22.0, "N": 15.0}          # cost_cap0.json planning: core 14.6 s, + 6.5 s with localisation


def sha_ids(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def plan():
    loc_on = os.environ.get("CAP0_LOC", "on").strip().lower() == "on"
    reps = []
    for item in os.environ.get("CAP0_REPS", DEFAULT_REPS).split(","):
        name, _, mode = item.strip().partition(":")
        if not name.startswith("r") or mode not in ("L", "N"):
            raise SystemExit(f"bad CAP0_REPS item {item!r}")
        reps.append(dict(name=name, loc=loc_on and mode == "L", salt=f"cap0-{name}"))
    if len({r["name"] for r in reps}) != len(reps):
        raise SystemExit("CAP0_REPS names repeat")
    return reps


class Client:
    def __init__(self, ao, deadline, base):
        self.ao, self.deadline, self.base = ao, float(deadline), base.rstrip("/")
        self.fh = open(os.path.join(ao, "client.jsonl"), "a")
        self.model = None
        self.last = []

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

    def one(self, kind, rep, tag, ids, seed, want_cached):
        body = dict(model=self.model, prompt=list(ids), max_tokens=1, temperature=1.0, top_p=0.95, seed=seed,
                    cache_salt=rep["salt"], vllm_xargs={"forcewin_hooks": [], "local_fw_tag": tag})
        j, err, t0, t1, st = self.post(body, self.left())
        u = (j or {}).get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        ok = err is None and u.get("prompt_tokens") == len(ids) and cached == want_cached
        d = dict(kind=kind, replicate=rep["name"], tag=tag, salt=rep["salt"], loc=rep["loc"], seed=seed,
                 status=st, error=err, t_send=round(t0, 3), t_done=round(t1, 3), prompt_tokens=u.get("prompt_tokens"),
                 cached_tokens=cached, cached_expected=want_cached, prompt_sha256=sha_ids(ids), ok=ok)
        self.rec(d)
        self.last.append(d)
        return ok

    def sentinel(self, rep):
        j, err, t0, t1, st = self.post(dict(model=self.model, prompt=[1, 2, 3, 4], max_tokens=1, temperature=0.0,
                                            cache_salt="cap0-sentinel"), 60)
        self.rec(dict(kind="sentinel", replicate=rep["name"], status=st, error=err))
        return err is None

    def run(self, ids, reps):
        warm_ids, cap_ids = ids[:WARM_TOKENS], ids[:LY.P_EARLY]
        self.rec(dict(kind="start", replicates=reps, warm_tokens=len(warm_ids), cap_prompt_tokens=len(cap_ids),
                      n_cached_expected=LY.N_CACHED, seed_warm=SEED_WARM, seed_cap=SEED_CAP,
                      warm_sha256=sha_ids(warm_ids), cap_sha256=sha_ids(cap_ids), deadline=self.deadline))
        self.get_model()
        done = []
        for rep in reps:
            if os.path.exists(os.path.join(self.ao, "STOP")):
                self.rec(dict(kind="skip", replicate=rep["name"], reason="STOP file"))
                break
            est = EST_S["L" if rep["loc"] else "N"]
            if time.time() + est > self.deadline:
                self.rec(dict(kind="skip", replicate=rep["name"], reason="time", est_s=est, left_s=round(self.left())))
                break
            t_r = time.time()
            self.last = []
            wtag = f"CW-{rep['name']}-{'L' if rep['loc'] else 'N'}"
            ok_w = self.one("warm", rep, wtag, warm_ids, SEED_WARM, 0)
            ok_c = self.one("cap", rep, f"CAP-{rep['name']}", cap_ids, SEED_CAP, LY.N_CACHED) if ok_w else False
            ok_s = self.sentinel(rep)
            self.rec(dict(kind="replicate_done", replicate=rep["name"], secs=round(time.time() - t_r, 1),
                          warm_ok=ok_w, cap_ok=ok_c, sentinel_ok=ok_s))
            done.append(rep["name"])
            if len(done) == 1 and not (ok_w and ok_c):
                last = {d["kind"]: d for d in self.last}
                self.rec(dict(kind="abort", replicate=rep["name"],
                              reason="first replicate: warm cached_tokens %s (expected 0), capture cached_tokens %s "
                                     "(expected %d), prompt_tokens %s / %s: the construction is wrong for every "
                                     "replicate; session ended early" % (
                                         (last.get("warm") or {}).get("cached_tokens"), (last.get("cap") or {}).get("cached_tokens"),
                                         LY.N_CACHED, (last.get("warm") or {}).get("prompt_tokens"),
                                         (last.get("cap") or {}).get("prompt_tokens"))))
                self.rec(dict(kind="end", replicates_done=done, aborted=True))
                return 5
        self.rec(dict(kind="end", replicates_done=done))
        return 0 if done == [r["name"] for r in reps] else 4


def main():
    ao, deadline, base = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    fx = json.load(open(os.environ["CAP0_WINDOWS"]))
    r01 = fx["replies"]["R01"]
    ids = [int(x) for x in r01["ids"]]
    reps = plan()
    if int(r01["windows"]["early"]["P"]) != LY.P_EARLY or len(ids) < LY.P_EARLY:
        print(json.dumps(dict(refused="R01-early P %s != %d or too few ids" % (r01["windows"]["early"]["P"], LY.P_EARLY))))
        return 2
    if os.environ.get("CAP0_PLAN_ONLY") == "1":
        print(json.dumps(dict(replicates=reps, warm_tokens=WARM_TOKENS, cap_prompt_tokens=LY.P_EARLY,
                              expected_cap_cached=LY.cache_hit(WARM_TOKENS, LY.P_EARLY),
                              n_cached_expected=LY.N_CACHED, est_s=sum(EST_S["L" if r["loc"] else "N"] for r in reps),
                              warm_sha256=sha_ids(ids[:WARM_TOKENS])[:16])))
        return 0
    return Client(ao, deadline, base).run(ids, reps)


if __name__ == "__main__":
    sys.exit(main())
