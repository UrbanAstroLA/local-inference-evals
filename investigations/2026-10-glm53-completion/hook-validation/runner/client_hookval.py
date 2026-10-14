#!/usr/bin/env python3
"""Hook-validation workload (PLAN.md amendment 21 section 4; VALIDATION.md). Runs AFTER authorization, against
the B1 forcehook-mg server started by run_hookval.sh. Existing exact-id histories only (24 step-5/6 traces < 5k)
plus 8 histories B1 samples itself in-session. No reasoning text is read or written: ids, positions, hashes.

usage: client_hookval.py OUTDIR DEADLINE_EPOCH         (GPU session; server at 127.0.0.1:8001)
       client_hookval.py OUTDIR 0 --plan               (CPU: build every request, write plan.json, no network)

Phases in priority order (a stopped session keeps the most important results):
  V0   four greedy chat probes (unkeyed), compared with B1's run-b1-a1 probe hashes: the hooked image with the hook
       ON computes like B1 for unkeyed requests.
  V2a  schedule matrix: 8 traces x 7 draft schedules (natural, oracle, poison, reject:1-3, sched). Two windows
       cross the 4,352-token block boundary (prefix 3,832 = 0 mod 8); six start at the prompt (prompt lengths
       124-365: unaligned to 4 and 8, so the first window pools straddle the prefill-seeded ring).
  V2c  prefill references for every V2 history: prompt + history, every position recorded (local_fw_ref).
  V1   B1's own histories: 8 free runs (T 1.0, top_p 0.95, distinct seeds, logprobs=1, return_token_ids), then
       each forced back (natural, poison) and prefill-referenced. The free run's returned logprobs are B1's own
       unhooked decode measurement of the same tokens.
  V4   D2 decode-window injection: one batch of 8 windows injected (local_fw_d2 = whole window), the same batch
       uninjected, one injected request alone; every request dumps its window's pages at its last commit;
       prefill rebuilds of prompt + window dump the same positions. Guard on (mapping-level verification).
  V3   committed-state snapshots: 8 requests, frontiers around the block boundary and at all residues mod 8,
       mixed schedules; then prefill references at the ACTUAL frontiers (read from the hook log).
  V2b  all 24 traces x {natural, oracle} (full histories up to the natural close).
Before the first request and after every phase the guard readout is triggered (DUMP).
"""
import gzip
import hashlib
import json
import os
import subprocess
import sys
import threading
import time

OUT, DEADLINE = sys.argv[1], float(sys.argv[2])
PLAN = "--plan" in sys.argv
ROOT = "<runs>"
INV = json.load(open(f"{ROOT}/fixture-inventory/inventory.json"))
GPQA = os.environ.get("GPQA", f"{ROOT}/bundle/gate/gpqa_fixture.LOCAL.json")
B1_PROBES = f"{ROOT}/d7/step2/run-b1-a1/probe.jsonl"
BASE = "http://127.0.0.1:8001"
CONTAINER = "glm53-flash-exl3-b12x-vllm"
CLOSE = 154842
os.makedirs(OUT, exist_ok=True)
LOG = open(f"{OUT}/client.jsonl", "a")
lock = threading.Lock()
SCHEDULES = ["natural", "oracle", "poison", "reject:1", "reject:2", "reject:3", "sched:20261013"]
PLANNED = []


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def rec(d):
    with lock:
        LOG.write(json.dumps(d) + "\n")
        LOG.flush()


def sha(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ fixtures (exact ids)
def load_traces():
    T = {}
    for et in INV["exact_traces"]:
        d = json.load(gzip.open(os.path.join("<data-runs>", et["path"])))
        c = int(et["natural_close_pos"])
        gen = d["gen_ids"]
        assert gen[c] == CLOSE, (et["id"], gen[c])
        assert sha(d["prompt_ids"]) and len(d["prompt_ids"]) == et["prompt_len"]
        T[et["id"][:12]] = dict(id=et["id"][:12], doc=et["doc"], prompt=d["prompt_ids"], hist=gen[: c + 1],
                                after=gen[c + 1], engine=et["engine"])
    return T


TR = load_traces()
CROSS = ["53fa5b3e066f", "57703691f509"]          # histories long enough to cross the 4,352 block boundary
MATRIX = CROSS + ["cbac17c70af8", "be2e9e349395", "4b92a6a90ec4", "ddfd287416a4", "34b209357e82", "aa89ec370e9c"]


def window(tid, cap=2500, cross=False):
    """(prefix ids, forced window ids). cross: prefix = prompt + history[:m] with P = 3,832 (0 mod 8), window
    capped at 1,100 tokens (crosses position 4,352 at window index 520)."""
    t = TR[tid]
    if cross:
        m = 3832 - len(t["prompt"])
        return t["prompt"] + t["hist"][:m], t["hist"][m: m + 1100]
    return list(t["prompt"]), t["hist"][:cap]


# ------------------------------------------------------------------ HTTP
MODEL = None


def _post(path, body):
    import requests
    t0 = time.time()
    try:
        j = requests.post(f"{BASE}{path}", json=body, timeout=3600).json()
        err = (j.get("error") or j.get("message")) if "choices" not in j else None
    except Exception as e:  # noqa: BLE001
        j, err = {}, repr(e)
    return j, err, t0, time.time()


SEED = [9_820_000]


def next_seed():
    SEED[0] += 1
    return SEED[0]


def forced(phase, tag, prefix, F, mode="natural", salt=None, xa=None, max_tokens=None, logprobs=None):
    """One forced request (token-id prompt); returns the record. Seeds are distinct per request and recorded."""
    seed = next_seed()
    xargs = {"forcewin_hooks": list(F), "local_fw_tag": tag, "local_fw_draft": mode}
    xargs.update(xa or {})
    body = dict(model=MODEL, prompt=list(prefix), max_tokens=max_tokens or len(F), temperature=1.0, top_p=0.95,
                seed=seed, cache_salt=salt or f"{phase}:{tag}:{seed}", vllm_xargs=xargs, return_token_ids=True)
    if logprobs is not None:
        body["logprobs"] = logprobs
    d = dict(kind="request", phase=phase, tag=tag, mode=mode, seed=seed, salt=body["cache_salt"], P=len(prefix),
             W=len(F), P_mod8=len(prefix) % 8, prefix_sha=sha(list(prefix)), forced_sha=sha(list(F)),
             keys=sorted(xargs))
    if PLAN:
        PLANNED.append(d)
        return d
    j, err, t0, t1 = _post("/v1/completions", body)
    ch = (j.get("choices") or [{}])[0]
    u = j.get("usage") or {}
    ids = ch.get("token_ids") or []
    d.update(t0=round(t0, 3), t1=round(t1, 3), secs=round(t1 - t0, 2), finish=ch.get("finish_reason"),
             completion_tokens=u.get("completion_tokens"), prompt_tokens=u.get("prompt_tokens"),
             cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
             returned_sha=sha(ids), ids_equal=(ids == list(F)) if "forcewin_hooks" in xargs and F else None,
             error=err)
    rec(d)
    return d


def free(phase, tag, prompt, n):
    seed = next_seed()
    body = dict(model=MODEL, prompt=list(prompt), max_tokens=n, temperature=1.0, top_p=0.95, seed=seed,
                logprobs=1, return_token_ids=True, cache_salt=f"{phase}:{tag}:{seed}")
    d = dict(kind="free", phase=phase, tag=tag, seed=seed, P=len(prompt), max_tokens=n)
    if PLAN:
        PLANNED.append(d)
        return d, list(range(1, n + 1)), [0.0] * n
    j, err, t0, t1 = _post("/v1/completions", body)
    ch = (j.get("choices") or [{}])[0]
    ids = ch.get("token_ids") or []
    lps = ((ch.get("logprobs") or {}).get("token_logprobs")) or []
    d.update(secs=round(t1 - t0, 2), finish=ch.get("finish_reason"), n=len(ids), returned_sha=sha(ids), error=err)
    rec(d)
    return d, ids, lps


def batch(fns):
    ths = [threading.Thread(target=f) for f in fns]
    [t.start() for t in ths]
    [t.join() for t in ths]


def dump(tag):
    if PLAN:
        return
    p = subprocess.run(["docker", "exec", CONTAINER, "sh", "-c", "mkdir -p /tmp/mapguard && touch /tmp/mapguard/DUMP"],
                       capture_output=True, text=True)
    rec(dict(kind="dump", tag=tag, t=round(time.time(), 3), rc=p.returncode))
    time.sleep(6)


def phase(tag, t0):
    rec(dict(kind="phase", tag=tag, t0=round(t0, 3), t1=round(time.time(), 3), secs=round(time.time() - t0, 1)))
    log(f"phase {tag} done in {time.time() - t0:.0f}s")
    dump(tag)


def left():
    return DEADLINE - time.time() if not PLAN else 1e9


# ------------------------------------------------------------------ phases
def V0():
    t0 = time.time()
    if PLAN:
        PLANNED.append(dict(kind="probe", n=4))
        return
    G = {i["doc_id"]: i for i in json.load(open(GPQA))["items"]}
    exp = {}
    for ln in open(B1_PROBES):
        r = json.loads(ln)
        if r.get("kind") == "probe" and r.get("round") == 0:
            exp[r["id"]] = r["sha256"]
    for d in range(4):   # payload and canonicalisation as client_mapguard.py probes()
        it = G[d]
        body = dict(model=MODEL, messages=it["messages"], max_tokens=96, temperature=0.0, top_p=1.0, seed=4242,
                    chat_template_kwargs=it["chat_template_kwargs"], logprobs=True, top_logprobs=1)
        j, err, a, b = _post("/v1/chat/completions", body)
        ch = (j.get("choices") or [{}])[0]
        lp = ((ch.get("logprobs") or {}).get("content")) or []
        toks = [(x.get("token"), x.get("logprob")) for x in lp]
        msg = ch.get("message") or {}
        canon = json.dumps(dict(c=msg.get("content"), r=msg.get("reasoning_content") or msg.get("reasoning"),
                                t=toks), sort_keys=True)
        h = hashlib.sha256(canon.encode()).hexdigest()
        rec(dict(kind="probe", id=f"q{d}", sha256=h, expected=exp.get(f"q{d}"), equal=h == exp.get(f"q{d}"),
                 n_tokens=len(toks), error=err))
    phase("V0", t0)


def V2a():
    for mode in SCHEDULES:
        if left() < 240:
            rec(dict(kind="skip", phase="V2a", mode=mode))
            continue
        t0 = time.time()
        fns = []
        for tid in MATRIX:
            pre, F = window(tid, cross=tid in CROSS)
            fns.append(lambda tid=tid, pre=pre, F=F, mode=mode: forced(
                "V2a", f"V2a-{mode}-{tid}", pre, F, mode,
                xa={"local_fw_selrec": [o for o in (64, 256, 520, 1024) if o < len(F)]}))
        batch(fns)
        phase(f"V2a-{mode}", t0)


def refs(phase_tag, items):
    """Prefill references (every position): prompt + history in one request, fresh salt, max_tokens 1."""
    t0 = time.time()
    fns = []
    for tag, pre, F in items:
        fns.append(lambda tag=tag, pre=pre, F=F: forced(
            phase_tag, f"{phase_tag}-{tag}", list(pre) + list(F), [], "natural",
            xa={"local_fw_ref": len(pre)}, max_tokens=1))
    for k in range(0, len(fns), 8):
        batch(fns[k:k + 8])
    phase(phase_tag, t0)


def V2c():
    refs("V2c", [(tid, *window(tid, cross=tid in CROSS)) for tid in MATRIX])


FREE = {}


def V1():
    t0 = time.time()
    docs = sorted({(TR[t]["doc"], t) for t in TR})
    prompts = []
    seen = {}
    for doc, tid in docs:
        if seen.get(doc, 0) < (2 if doc in (0, 1) else 1):
            seen[doc] = seen.get(doc, 0) + 1
            prompts.append((f"free{len(prompts)}-doc{doc}", TR[tid]["prompt"]))
    prompts = prompts[:8]
    res = {}
    batch([lambda tag=tag, p=p: res.__setitem__(tag, free("V1", tag, p, 2000)) for tag, p in prompts])
    for tag, p in prompts:
        d, ids, lps = res[tag]
        if CLOSE in ids:
            ids = ids[: ids.index(CLOSE) + 1]
            lps = lps[: len(ids)]
        FREE[tag] = (p, ids, lps)
        rec(dict(kind="free_hist", tag=tag, n=len(ids), sha=sha(ids), lps_sha=sha([round(x or 0, 6) for x in lps]),
                 lps=lps))            # B1's own decode log-probabilities of its sampled tokens (no text)
    phase("V1-free", t0)
    for mode in ("natural", "poison"):
        t0 = time.time()
        batch([lambda tag=tag, mode=mode: forced("V1", f"V1-{mode}-{tag}", FREE[tag][0], FREE[tag][1], mode)
               for tag in FREE])
        phase(f"V1-{mode}", t0)
    refs("V1-ref", [(tag, FREE[tag][0], FREE[tag][1]) for tag in FREE])


D2SET = ["cbac17c70af8", "be2e9e349395", "4b92a6a90ec4", "ddfd287416a4", "68a96f264430", "376073921446",
         "1e875e978fd4", "6d18c40f58df"]


def V4():
    if left() < 420:
        rec(dict(kind="skip", phase="V4"))
        return
    items = [(tid, *window(tid, cap=2000)) for tid in D2SET]
    for arm in ("d2", "b1"):
        t0 = time.time()
        fns = []
        for tid, pre, F in items:
            xa = {"local_fw_dumpwin": 1, "local_fw_selrec": [o for o in (256, 512, 1024, 1536) if o < len(F)]}
            if arm == "d2":
                xa["local_fw_d2"] = [0, len(F) + 1]
            fns.append(lambda tid=tid, pre=pre, F=F, xa=xa, arm=arm: forced("V4", f"V4-{arm}-{tid}", pre, F,
                                                                             "natural", xa=xa))
        batch(fns)
        phase(f"V4-{arm}", t0)
    t0 = time.time()
    tid, pre, F = items[0]
    forced("V4", f"V4-d2solo-{tid}", pre, F, "natural", xa={"local_fw_dumpwin": 1, "local_fw_d2": [0, len(F) + 1]})
    phase("V4-d2solo", t0)
    t0 = time.time()
    fns = []
    for tid, pre, F in items:    # prefill rebuild of prompt + window: dumps the same positions, written by prefill
        nxt = TR[tid]["hist"][len(F)] if len(F) < len(TR[tid]["hist"]) else TR[tid]["after"]
        fns.append(lambda tid=tid, pre=pre, F=F, nxt=nxt: forced(
            "V4", f"V4-rebuild-{tid}", list(pre) + list(F), [nxt], "natural",
            xa={"local_fw_dumpwin": 1, "local_fw_dumprange": [len(pre), len(pre) + len(F)]}))
    batch(fns)
    phase("V4-rebuild", t0)


def V3():
    if left() < 300:
        rec(dict(kind="skip", phase="V3"))
        return
    t0 = time.time()
    plan = []
    modes = ["sched:7", "reject:2", "oracle", "poison", "natural", "reject:3", "reject:1", "sched:11"]
    for k, tid in enumerate(MATRIX):
        cross = tid in CROSS
        pre, F = window(tid, cross=cross, cap=600)
        if cross:   # processed frontier P + F - 2 at 4,350 .. 4,354 (P = 3,832)
            targets = [4350 - 3832 + 2 + i for i in range(5)]
        else:       # all residues of the frontier mod 8, after the first window pools
            targets = [101 + i * 9 for i in range(8)]
        targets = [t for t in targets if t < len(F)]
        plan.append((tid, pre, F, modes[k], targets))
    batch([lambda tid=tid, pre=pre, F=F, m=m, tg=tg: forced("V3", f"V3-{m}-{tid}", pre, F, m,
                                                             xa={"local_fw_snap": tg}) for tid, pre, F, m, tg in plan])
    phase("V3-forced", t0)
    t0 = time.time()
    actual = []
    if not PLAN:
        p = subprocess.run(["docker", "exec", CONTAINER, "sh", "-c", "cat /tmp/forcewin/fw_r0_*.jsonl"],
                           capture_output=True, text=True)
        for ln in p.stdout.splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("kind") == "snap" and str(r.get("tag", "")).startswith("V3-") and r["targets"] != ["win"]:
                actual.append((r["tag"], int(r["F"])))
    else:
        actual = [(f"V3-{m}-{tid}", t) for tid, pre, F, m, tg in plan for t in tg[:1]]
    fns = []
    byid = {f"V3-{m}-{tid}": (pre, F) for tid, pre, F, m, tg in plan}
    for tag, Fa in sorted(set(actual)):
        pre, F = byid[tag]
        fns.append(lambda tag=tag, Fa=Fa, pre=pre, F=F: forced(
            "V3", f"V3ref-F{Fa}-{tag}", list(pre) + list(F[: Fa - 1]), [F[Fa - 1]], "natural",
            xa={"local_fw_snap": [1]}))
    for f in fns:        # one at a time: the reference's KDA state is then the step's only non-spec row
        f()
    rec(dict(kind="snap_refs", n=len(fns), actual=actual))
    phase("V3-refs", t0)


def V2b():
    order = sorted(TR, key=lambda t: len(TR[t]["hist"]))
    for mode in ("natural", "oracle"):
        for k in range(0, 24, 8):
            if left() < 200:
                rec(dict(kind="skip", phase="V2b", mode=mode, k=k))
                continue
            t0 = time.time()
            batch([lambda tid=tid, mode=mode: forced("V2b", f"V2b-{mode}-{tid}", TR[tid]["prompt"], TR[tid]["hist"],
                                                     mode) for tid in order[k:k + 8]])
            phase(f"V2b-{mode}-{k}", t0)
    refs("V2b-ref", [(tid, TR[tid]["prompt"], TR[tid]["hist"]) for tid in order if tid not in MATRIX])


if __name__ == "__main__":
    if not PLAN:
        import requests
        MODEL = requests.get(f"{BASE}/v1/models", timeout=30).json()["data"][0]["id"]
    rec(dict(kind="start", t=round(time.time(), 3), plan=PLAN, deadline=DEADLINE,
             traces=sorted(TR), matrix=MATRIX, d2set=D2SET))
    dump("start")
    for ph in (V0, V2a, V2c, V1, V4, V3, V2b):
        if left() < 60:
            rec(dict(kind="skip", phase=ph.__name__))
            continue
        ph()
    rec(dict(kind="end", t=round(time.time(), 3)))
    if PLAN:
        json.dump(PLANNED, open(f"{OUT}/plan.json", "w"), indent=0)
        n = sum(1 for p in PLANNED if p.get("kind") == "request")
        tok = sum(p.get("W", 0) for p in PLANNED if p.get("kind") == "request")
        pf = sum(p.get("P", 0) for p in PLANNED if p.get("kind") == "request")
        print(f"plan: {n} forced/ref requests, {tok} forced tokens, {pf} prompt tokens; "
              f"{sum(1 for p in PLANNED if p.get('kind') == 'free')} free runs")
