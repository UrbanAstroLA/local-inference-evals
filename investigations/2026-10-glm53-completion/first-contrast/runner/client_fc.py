#!/usr/bin/env python3
"""First long-context contrast workload (PROPOSAL.md s.3, s.7; PLAN.md amendments 21g s.6-7 and 22). Derived from the
validated hook-validation/runner/client_hookval.py (attempt-2 variant): forced(), dump(), phase(), batch() and the
coverage / skip records are kept; payload keys, seeds scheme and record fields follow attempt 2. The hook is not
changed. Runs AFTER authorization against the B1 forcehook-mg2 server started by run_fc.sh.

usage: client_fc.py OUTDIR DEADLINE_EPOCH            (GPU session; server at 127.0.0.1:8001)
       client_fc.py OUTDIR 0 --plan                  (CPU: builds every request, writes plan.json; no network)

Session order [A22-4: validation BEFORE any injected batch; reference warms first]:
  stage A   V0; LW1, LW2 (prefill references, cold, own salts; LW2 at build length P + 8192 = the different-build-
            length prefill null [A22-3]); warm s1 -> C1 [quick structural stop]; warm s2 -> C2 [quick structural
            stop]; CV (conv-state boundary control, clean, on s1's cached prefix) [A22-5]
  GATE-1    runner/gate_step.sh copies stage A's records out of the container and runs analysis/gate_fc.py in a CPU
            container (no network, no GPU) WHILE stage L runs; the client waits for its verdict before any D batch
  stage L   re-warm l1 -> LC1; re-warm l2 -> LC2 (clean LP runs; no injection)
  -- stop here unless GATE-1 = PASS (no injected batch runs on a failed or missing gate) --
  stage B   re-warm s2 -> D2; re-warm s1 -> D1; re-warm l2 -> LD2; re-warm l1 -> LD1;
            warm s3 -> PS-a, PS-b (prefix sentinel, sequential) -> D3 -> PS-c -> re-warm s3 -> C3 (residue check);
            LW3 -> LD3 -> re-warm l3 -> LC3
Cache-safety invariant [A22-3, restated]: under one salt, no request issued after a decode window started under that
salt may cover a full 4,352-token block that reaches the decode start, i.e. floor(len / 4352) * 4352 <= every earlier
decode start under the salt. Primary salts carry P (C/D/PS) and P' = B1 - 63 (CV; below B1); LP salts carry P_L and,
BEFORE any decode under l2, LW2's P + 8192. runner/plan_check.py enforces it on the plan.
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
HERE = os.path.dirname(os.path.abspath(__file__))
FCR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(FCR, "analysis"))
import fc_geometry as geo  # noqa: E402

ROOT = "<runs>"
GPQA = os.environ.get("GPQA", f"{ROOT}/bundle/gate/gpqa_fixture.LOCAL.json")
B1_PROBES = f"{ROOT}/d7/step2/run-b1-a1/probe.jsonl"
BASE = "http://127.0.0.1:8001"
CONTAINER = "glm53-flash-exl3-b12x-vllm"
GATE_STEP = os.environ.get("FC_GATE_STEP", os.path.join(HERE, "gate_step.sh"))
os.makedirs(OUT, exist_ok=True)
LOG = open(f"{OUT}/client.jsonl", "a")
lock = threading.Lock()
G = geo.load(FCR)
ORDER = G["order"]
MAN = json.load(open(os.path.join(FCR, "fixtures", "ids_manifest.json")))
DONE = {}
PLANNED = []
PHASES = []                    # phase order as executed (or planned)
REQUIRED = ["V0", "LW1", "LW2", "C1", "C2", "CV", "GATE1", "LC1", "LC2", "D2", "D1", "LD2", "LD1", "PS-a", "PS-b",
            "D3", "PS-c", "C3", "LW3", "LD3", "LC3"]
# planning rates for the never-truncate rule (pessimistic, cost/cost_fc.py)
PF_RATE, DEC_RATE, DUMP_S = 1800.0, 22.0, 8.0


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def rec(d):
    with lock:
        LOG.write(json.dumps(d) + "\n")
        LOG.flush()


def sha(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


def sha_full(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


# ------------------------------------------------------------------ frozen ids
def load_ids():
    I = {}
    for n in ORDER:
        m = MAN["fixtures"][n]
        p = os.path.join(FCR, "fixtures", m["file"])
        if hashlib.sha256(open(p, "rb").read()).hexdigest() != m["file_sha256"]:
            raise SystemExit("ids file hash mismatch: %s" % n)
        d = json.loads(gzip.open(p).read())
        assert sha_full(d["prefix"]) == m["prefix_sha256"] and sha_full(d["forced"]) == m["forced_sha256"], n
        assert len(d["prefix"]) == G["fx"][n]["P"] and len(d["forced"]) == G["fx"][n]["L"], n
        I[n] = d
    return I


IDS = load_ids()

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


SEED_PHASES = ["LW1", "LW2", "warm-s1", "C1", "warm-s2", "C2", "CV", "rewarm-l1a", "LC1", "rewarm-l2a", "LC2",
               "rewarm-s2", "D2", "rewarm-s1", "D1", "rewarm-l2b", "LD2", "rewarm-l1b", "LD1", "warm-s3", "PS-a",
               "PS-b", "D3", "PS-c", "rewarm-s3", "C3", "LW3", "LD3", "rewarm-l3", "LC3"]


def seed_of(phase, fx, extra=0):
    """Distinct, recorded, deterministic per (phase, fixture, schedule) from a FIXED phase list: a skipped phase never
    shifts the seeds of later phases, and the plan's seeds equal the session's."""
    return 9_830_000 + 1000 * SEED_PHASES.index(phase) + 10 * ORDER.index(fx) + extra


def send(phase, tag, fx, prompt, F, salt_name, role, cached_expected, xa=None, max_tokens=None, keyed=True,
         seed_extra=0, batch_id=None, extra=None):
    """One request. Keyed requests carry forcewin_hooks (possibly empty for references); unkeyed warms carry no
    vllm_xargs at all. Every request records its expected cached-token count (exact cache accounting)."""
    seed = seed_of(phase, fx, seed_extra)
    salt = geo.salt(salt_name)
    body = dict(model=MODEL, prompt=list(prompt), max_tokens=max_tokens or max(len(F), 1), temperature=1.0,
                top_p=0.95, seed=seed, cache_salt=salt, return_token_ids=True)
    xargs = None
    if keyed:
        xargs = {"forcewin_hooks": list(F), "local_fw_tag": tag}
        xargs.update(xa or {})
        body["vllm_xargs"] = xargs
    d = dict(kind="request", phase=phase, tag=tag, fixture=fx, role=role, salt=salt, salt_name=salt_name, seed=seed,
             P=len(prompt), W=len(F), P_mod8=len(prompt) % 8, prefix_sha=sha(list(prompt)), forced_sha=sha(list(F)),
             prompt_sha256=sha_full(list(prompt)), forced_sha256=sha_full(list(F)),
             last_forced=(F[-1] if F else None), keyed=keyed, keys=sorted(xargs) if xargs else [],
             mode=(xargs or {}).get("local_fw_draft"), d2=(xargs or {}).get("local_fw_d2"),
             ref_from=(xargs or {}).get("local_fw_ref"), dumprange=(xargs or {}).get("local_fw_dumprange"),
             dumpwin=(xargs or {}).get("local_fw_dumpwin"), snap=(xargs or {}).get("local_fw_snap"),
             desig_n=len((xargs or {}).get("local_fw_desig") or []),
             desig_sha=sha((xargs or {}).get("local_fw_desig")) if (xargs or {}).get("local_fw_desig") is not None
             else None, selrec_n=len((xargs or {}).get("local_fw_selrec") or []),
             cached_expected=cached_expected, max_tokens=body["max_tokens"], batch=batch_id)
    if extra:
        d.update(extra)
    if PLAN:
        PLANNED.append(d)
        return d
    j, err, t0, t1 = _post("/v1/completions", body)
    ch = (j.get("choices") or [{}])[0]
    u = j.get("usage") or {}
    ids = ch.get("token_ids") or []
    d.update(t0=round(t0, 3), t1=round(t1, 3), secs=round(t1 - t0, 2), finish=ch.get("finish_reason"),
             completion_tokens=u.get("completion_tokens"), prompt_tokens=u.get("prompt_tokens"),
             cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"), returned_sha=sha(ids),
             ids_equal=(ids == list(F)) if (keyed and F) else None, error=err)
    rec(d)
    return d


def batch(fns):
    if PLAN:                    # plan mode: submission order is the record order
        for f in fns:
            f()
        return
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
    DONE[tag] = round(time.time() - t0, 1)
    PHASES.append(tag)
    rec(dict(kind="phase", tag=tag, t0=round(t0, 3), t1=round(time.time(), 3), secs=round(time.time() - t0, 1)))
    log(f"phase {tag} done in {time.time() - t0:.0f}s")
    dump(tag)


def left():
    return DEADLINE - time.time() if not PLAN else 1e9


RESERVE_S = [0.0]           # seconds kept for higher-priority phases still to come (primary before LP)


def fits(tag, need):
    """Never start a batch that cannot finish: skip (recorded) when the time left cannot cover it plus the reserve
    of the higher-priority phases that follow it."""
    if left() >= need + RESERVE_S[0]:
        return True
    rec(dict(kind="skip", phase=tag, left=round(left(), 1), need=round(need, 1), reserve=round(RESERVE_S[0], 1)))
    log(f"SKIP {tag}: left {left():.0f}s < need {need:.0f}s")
    return False


# ------------------------------------------------------------------ request builders
def prim_xa(n, d2=False):
    f = G["fx"][n]
    xa = {"local_fw_draft": "natural", "local_fw_desig": f["desig"], "local_fw_selrec": f["selrec"],
          "local_fw_dumpwin": 1}
    if d2:
        xa["local_fw_d2"] = [0, f["L"] + 1]           # whole window, attempt-2 V4 convention
    return xa


def lp_xa(n, d2=False):
    f = G["fx"][n]
    xa = {"local_fw_draft": "natural", "local_fw_desig": f["desig_lp"], "local_fw_selrec": f["selrec_lp"],
          "local_fw_dumpwin": 1}
    if d2:
        xa["local_fw_d2"] = [0, f["L_s"] + 1]
    return xa


def V0():
    t0 = time.time()
    if PLAN:
        PLANNED.append(dict(kind="probe", n=4))
        phase("V0", t0)
        return
    G_ = {i["doc_id"]: i for i in json.load(open(GPQA))["items"]}
    exp = {}
    for ln in open(B1_PROBES):
        r = json.loads(ln)
        if r.get("kind") == "probe" and r.get("round") == 0:
            exp[r["id"]] = r["sha256"]
    for d in range(4):   # payload and canonicalisation as client_hookval.py V0 (validated)
        it = G_[d]
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


def LW(r, build_len_key="P_L", tag=None):
    """Prefill reference warm (keyed, cold, own LP salt): prompt = prefix + forced[:n] with n = 7,168 (LW1, LW3) or
    8,192 (LW2 [A22-3]: different build length); records every prompt row predicting position >= P (local_fw_ref)
    and dumps the prefill-written pages over [P, P_L] (identical range for LW1 / LW2 / LW3: the N_pp comparison)."""
    tag = tag or f"LW{r}"
    need = sum(G["fx"][n][build_len_key] for n in ORDER) / PF_RATE + 8 * DUMP_S + 30
    if not fits(tag, need):
        return False
    t0 = time.time()
    fns = []
    for n in ORDER:
        f = G["fx"][n]
        ln = f[build_len_key]
        prompt = IDS[n]["prefix"] + IDS[n]["forced"][: ln - f["P"]]
        xa = {"local_fw_ref": f["P"], "local_fw_dumpwin": 1, "local_fw_dumprange": [f["P"], f["P_L"]]}
        fns.append(lambda n=n, prompt=prompt, xa=xa: send(tag, f"{tag}-{n}", n, prompt, [], f"l{r}", "ref", 0, xa=xa,
                                                         max_tokens=1, extra=dict(build_len=len(prompt))))
    batch(fns)
    phase(tag, t0)
    return True


def warm(salt_name, construction, tag, hit):
    """Unkeyed warm (max_tokens 1): prefills each prompt once under the round salt so every batch prefills only the
    uncached remainder. hit=False: first use of the salt (cold, 0 cached expected); hit=True: re-warm, a cache hit
    unless evicted (cached_tokens recorded and classified, never assumed)."""
    t0 = time.time()
    fns = []
    for n in ORDER:
        f = G["fx"][n]
        if construction == "long":
            prompt, c0 = IDS[n]["prefix"], f["c0"]
        else:
            prompt, c0 = IDS[n]["prefix"] + IDS[n]["forced"][: geo.LP_SHIFT], f["c0_L"]
        fns.append(lambda n=n, prompt=prompt, c0=c0: send(tag, f"{tag}-{n}", n, prompt, [], salt_name, "warm",
                                                          c0 if hit else 0, keyed=False, max_tokens=1,
                                                          extra=dict(rewarm=hit, hit_value=c0)))
    batch(fns)
    phase(tag, t0)


def run_batch(tag, arm, construction, salt_name):
    """One run = the 8 fixtures in batch_order, submitted together, identical composition in every run."""
    f0 = [G["fx"][n] for n in ORDER]
    if construction == "long":
        need = sum(f["rem"] for f in f0) / PF_RATE + max(f["L"] for f in f0) / DEC_RATE + 8 * DUMP_S + 30
    else:
        need = sum(f["P_L"] % geo.BLOCK for f in f0) / PF_RATE + max(f["L_s"] for f in f0) / DEC_RATE + 8 * DUMP_S + 30
    if not fits(tag, need):
        return False
    t0 = time.time()
    fns = []
    for n in ORDER:
        f = G["fx"][n]
        d2 = arm == "D"
        if construction == "long":
            prompt, F, xa, c0 = IDS[n]["prefix"], IDS[n]["forced"], prim_xa(n, d2), f["c0"]
        else:
            prompt = IDS[n]["prefix"] + IDS[n]["forced"][: geo.LP_SHIFT]
            F, xa, c0 = IDS[n]["forced"][geo.LP_SHIFT:], lp_xa(n, d2), f["c0_L"]
        fns.append(lambda n=n, prompt=prompt, F=F, xa=xa, c0=c0: send(
            tag, f"{tag}-{n}", n, prompt, F, salt_name, arm if construction == "long" else "L" + arm, c0, xa=xa,
            batch_id=tag, extra=dict(construction=construction, arm=arm)))
    batch(fns)
    phase(tag, t0)
    return True


def CV():
    """[A22-5] conv-state boundary control: fixtures L5, L1 x schedules poison / oracle / reject:2 / natural, one batch
    of 8 (clean, no injection), on s1's cached prefix: prompt P' = B1 - 63 (prefix + forced[:x]), forced list = the
    next 72 history ids; committed-state snapshots (local_fw_snap; existing capability) at frontiers B1 - 8 .. B1 + 4
    chosen so that poison (single-token advance, the matched decode reference) shares every F with oracle and
    reject:2, and consecutive poison snapshots straddle the align-mode column change. No hook change."""
    if not fits("CV", 4289 * 8 / PF_RATE + 90 + 120):
        return False
    t0 = time.time()
    fns = []
    for n in geo.CV_FIXTURES:
        f = G["fx"][n]
        cv = f["cv"]
        prompt = IDS[n]["prefix"] + IDS[n]["forced"][: cv["x"]]
        F = IDS[n]["forced"][cv["x"]: cv["x"] + cv["list_len"]]
        for k, m in enumerate(geo.CV_MODES):
            xa = {"local_fw_draft": m, "local_fw_snap": cv["targets"][m], "local_fw_desig": [],
                  "local_fw_selrec": []}
            fns.append(lambda n=n, m=m, k=k, prompt=prompt, F=F, xa=xa, c0=f["c0"], B=cv["B"]: send(
                "CV", f"CV-{m.replace(':', '_')}-{n}", n, prompt, F, "s1", "CV", c0, xa=xa, seed_extra=k,
                batch_id="CV", extra=dict(cv_mode=m, B=B)))
    batch(fns)
    phase("CV", t0)
    return True


def PS(tag):
    """[A22-4] prefix sentinel on s3's cached prefix, one request at a time (deterministic batch shape): prompt P,
    forced list [forced[0]], a committed-state snapshot after the prefill step (KDA conv + recurrent sketch of the
    state at P - 1, built from the cached boundary state) and a dump of the LAST CACHED (shared) block's pages
    [c0 - 1024, c0]. PS-a / PS-b before D3 (PS-b = in-session determinism null), PS-c after D3."""
    if not fits(tag, 8 * 6 + 30):
        return False
    t0 = time.time()
    for n in ORDER:
        f = G["fx"][n]
        xa = {"local_fw_draft": "natural", "local_fw_snap": [1], "local_fw_dumpwin": 1,
              "local_fw_dumprange": [f["c0"] - geo.PS_DUMP, f["c0"]], "local_fw_desig": [], "local_fw_selrec": []}
        send(tag, f"{tag}-{n}", n, IDS[n]["prefix"], IDS[n]["forced"][:1], "s3", "PS", f["c0"], xa=xa, max_tokens=1)
    phase(tag, t0)
    return True


# ------------------------------------------------------------------ in-session checks
def quick_struct(tag):
    """Client-side structural stop (from client records and the hook log; no analysis): every request of the run
    returned without error, ids_equal, cached_tokens == expected; no fw_error / guard_error in the hook logs."""
    if PLAN:
        return True
    rs = [json.loads(ln) for ln in open(f"{OUT}/client.jsonl") if '"kind": "request"' in ln]
    rs = [r for r in rs if r.get("phase") == tag]
    bad = [r["tag"] for r in rs if r.get("error") or r.get("ids_equal") is not True
           or r.get("cached_tokens") != r.get("cached_expected")]
    p = subprocess.run(["docker", "exec", CONTAINER, "sh", "-c",
                        'grep -h "\\"fw_error\\"\\|\\"guard_error\\"" /tmp/forcewin/*.jsonl /tmp/mapguard/*.jsonl '
                        '2>/dev/null | wc -l'], capture_output=True, text=True)
    nerr = int((p.stdout or "0").strip() or 0)
    ok = len(rs) == len(ORDER) and not bad and nerr == 0
    rec(dict(kind="quick_struct", run=tag, ok=ok, n=len(rs), bad=bad[:8], hook_errors=nerr))
    return ok


GATE = {}


def gate_start():
    """GATE-1 (runner's CPU gate step) in the background: copies stage-A records and runs analysis/gate_fc.py in a
    CPU container. Its verdict file is OUT/gate1/gate1.json."""
    rec(dict(kind="gate_start", t=round(time.time(), 3)))
    if PLAN:
        GATE["rc"] = 0
        return None

    def go():
        t = time.time()
        p = subprocess.run(["bash", GATE_STEP, OUT], capture_output=True, text=True)
        GATE.update(rc=p.returncode, secs=round(time.time() - t, 1), tail=(p.stdout + p.stderr)[-1500:])
    th = threading.Thread(target=go)
    th.start()
    return th


def gate_wait(th):
    t = time.time()
    if th is not None:
        th.join(timeout=max(60.0, left() - 60.0))
    v = None
    if not PLAN:
        try:
            v = json.load(open(f"{OUT}/gate1/gate1.json")).get("verdict")
        except Exception:  # noqa: BLE001
            v = None
    ok = PLAN or (GATE.get("rc") == 0 and v == "PASS")
    rec(dict(kind="gate", gate="GATE1", ok=ok, verdict=v, rc=GATE.get("rc"), secs=GATE.get("secs"),
             waited=round(time.time() - t, 1), tail=GATE.get("tail")))
    DONE["GATE1"] = GATE.get("secs", 0.0)
    PHASES.append("GATE1")
    return ok


# ------------------------------------------------------------------ main
def stop(why, rc):
    rec(dict(kind="stop", why=why, phases=PHASES, t=round(time.time(), 3)))
    end(rc)


def end(rc=0):
    rec(dict(kind="coverage", required=REQUIRED, ran=[p for p in REQUIRED if p in DONE],
             not_run=[p for p in REQUIRED if p not in DONE], order=PHASES, left=round(left(), 1)))
    rec(dict(kind="end", t=round(time.time(), 3)))
    if PLAN:
        json.dump(PLANNED, open(f"{OUT}/plan.json", "w"), indent=0)
        n = sum(1 for p in PLANNED if p.get("kind") == "request")
        tok = sum(p.get("W", 0) for p in PLANNED if p.get("kind") == "request")
        pf = sum(p.get("P", 0) for p in PLANNED if p.get("kind") == "request")
        print(f"plan: {n} requests, {tok} forced tokens, {pf} prompt tokens (before cache hits); phases {PHASES}")
    sys.exit(rc)


def main():
    global MODEL
    if not PLAN:
        import requests
        MODEL = requests.get(f"{BASE}/v1/models", timeout=30).json()["data"][0]["id"]
    rec(dict(kind="start", t=round(time.time(), 3), plan=PLAN, deadline=DEADLINE, scope="first-contrast fc1",
             order=ORDER, wmax=geo.WMAX, salts={k: geo.salt(k) for k in ("s1", "s2", "s3", "l1", "l2", "l3")}))
    dump("start")
    # ---------------- stage A
    V0()
    if not LW(1):
        stop("LW1 does not fit", 11)
    if not LW(2, "ref2_len"):
        stop("LW2 does not fit", 11)
    warm("s1", "long", "warm-s1", False)
    if not run_batch("C1", "C", "long", "s1"):
        stop("C1 does not fit", 11)
    if not quick_struct("C1"):
        stop("structural stop after C1", 9)
    warm("s2", "long", "warm-s2", False)
    if not run_batch("C2", "C", "long", "s2"):
        stop("C2 does not fit", 11)
    if not quick_struct("C2"):
        stop("structural stop after C2", 9)
    if not CV():
        stop("CV does not fit", 11)
    th = gate_start()
    # ---------------- stage L (clean; concurrent with the gate)
    warm("l1", "short", "rewarm-l1a", True)
    run_batch("LC1", "C", "short", "l1")
    warm("l2", "short", "rewarm-l2a", True)
    run_batch("LC2", "C", "short", "l2")
    if not gate_wait(th):
        stop("GATE-1 not PASS: no injected batch is run", 10)
    # ---------------- stage B (injected batches only after GATE-1 = PASS)
    warm("s2", "long", "rewarm-s2", True)
    run_batch("D2", "D", "long", "s2")
    warm("s1", "long", "rewarm-s1", True)
    run_batch("D1", "D", "long", "s1")
    # LD2 / LD1 run here (their LP prefixes are still cached) only if the remaining PRIMARY phases (s3 round: warm,
    # PS-a, PS-b, D3, PS-c, re-warm, C3) still fit afterwards: the primary contrast has priority
    f0 = [G["fx"][n] for n in ORDER]
    RESERVE_S[0] = (sum(f["P"] for f in f0) / PF_RATE + 3 * (8 * 6 + 30)
                    + 2 * (sum(f["rem"] for f in f0) / PF_RATE + max(f["L"] for f in f0) / DEC_RATE + 8 * DUMP_S + 30)
                    + sum(f["rem"] for f in f0) / PF_RATE + 60)
    warm("l2", "short", "rewarm-l2b", True)
    run_batch("LD2", "D", "short", "l2")
    warm("l1", "short", "rewarm-l1b", True)
    run_batch("LD1", "D", "short", "l1")
    RESERVE_S[0] = 0.0
    warm("s3", "long", "warm-s3", False)
    PS("PS-a")
    PS("PS-b")
    run_batch("D3", "D", "long", "s3")
    PS("PS-c")
    warm("s3", "long", "rewarm-s3", True)
    run_batch("C3", "C", "long", "s3")
    if LW(3):
        run_batch("LD3", "D", "short", "l3")
        warm("l3", "short", "rewarm-l3", True)
        run_batch("LC3", "C", "short", "l3")
    end(0)


if __name__ == "__main__":
    main()
