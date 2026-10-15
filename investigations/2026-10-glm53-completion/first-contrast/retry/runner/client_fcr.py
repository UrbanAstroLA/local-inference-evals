#!/usr/bin/env python3
"""First-contrast RETRY workload (RETRY-PROPOSAL.md; PLAN.md amendment 22d). Derived from ../../runner/client_fc.py
(attempt 1, unchanged and not imported): the request payloads, keys, seed scheme, record fields, dump() and
structural stop are kept; the session order, submission policy, cache expectations and in-session checks are new and
come from ONE definition, analysis/fcr_geometry.py. The hook is not changed. Runs AFTER authorization against the B1
forcehook-mg2 server started by run_fcr.sh.

usage: client_fcr.py OUTDIR DEADLINE_EPOCH [--variant B|A0]        (GPU session; server at 127.0.0.1:8001)
       client_fcr.py OUTDIR 0 --plan [--variant B|A0]               (CPU: builds every request, writes plan.json)

Changes (R-n = RETRY-PROPOSAL.md s.0):
  R-2  every request records cached_expected = the F1 value from the plan (prefill coverage of its stream under its
       salt), its class (cold / hit / partial / miss / OVER) and the response id (engine request id prefix)
  R-3  the plan: split builds (4 longest, touch, 4 shortest, touch all), LP runs on the primary builds before any
       primary decode on them, C2 / D2 sharing s2's build across the gate, D3 / C3 sharing s3's
  R-4  submission policy: builds in a fixed shortest-first order with a fixed 1.0 s stagger; every run / touch /
       sentinel pass in ascending completion time of the most recent earlier request of the same fixture and salt
       (oldest last-freed first), runs with a fixed 0.5 s stagger, passes strictly sequential; the actual submission
       times are recorded, and the gates test the needed properties, not agreement with a timing prediction
  R-5  sentinel PS-a / PS-b / PS-c on s2: dump the shared block [c - 4352, c) of the restored boundary c
  R-7  in-session early stops (rc): 9 structural (error / ids), 12 unsafe restoration, 13 insufficient concurrent
       decoding (lead >= 512 in a run batch), 14 missing sentinel coverage after injected traffic, 10 GATE-1R not PASS,
       11 a required phase does not fit
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
VARIANT = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else "B"
HERE = os.path.dirname(os.path.abspath(__file__))
RETRY = os.path.dirname(HERE)
FCR = os.path.dirname(RETRY)
sys.path.insert(0, os.path.join(RETRY, "analysis"))
import cache_gate as cg  # noqa: E402
import fcr_geometry as geo  # noqa: E402

ROOT = "<runs>"
GPQA = os.environ.get("GPQA", f"{ROOT}/bundle/gate/gpqa_fixture.LOCAL.json")
B1_PROBES = f"{ROOT}/d7/step2/run-b1-a1/probe.jsonl"
BASE = "http://127.0.0.1:8001"
CONTAINER = "glm53-flash-exl3-b12x-vllm"
GATE_STEP = os.environ.get("FCR_GATE_STEP", os.path.join(HERE, "gate_step_fcr.sh"))
os.makedirs(OUT, exist_ok=True)
LOG = open(f"{OUT}/client.jsonl", "a")
lock = threading.Lock()
G = geo.load(FCR)
ORDER = G["order"]
MAN = json.load(open(os.path.join(FCR, "fixtures", "ids_manifest.json")))
PLANNED, PHASES, DONE = [], [], {}
LAST = {}                    # (salt_name, fixture) -> t1 of the most recent completed request
PF_RATE, DEC_RATE, DUMP_S = 2410.0, 25.0, 6.3      # the cost model's pessimistic rates (never-truncate rule)
LP_OFF = [False]             # once an LP batch is skipped, every later LP batch is skipped (pairs stay coherent)


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


def load_ids():
    out = {}
    for n in ORDER:
        m = MAN["fixtures"][n]
        p = os.path.join(FCR, "fixtures", m["file"])
        raw = open(p, "rb").read()
        if hashlib.sha256(raw).hexdigest() != m["file_sha256"]:
            raise SystemExit("ids file hash mismatch: %s" % n)
        d = json.loads(gzip.decompress(raw))
        assert sha_full(d["prefix"]) == m["prefix_sha256"] and sha_full(d["forced"]) == m["forced_sha256"], n
        out[n] = d
    return out


IDS = load_ids()
PLAN_PHASES = geo.phases(G, VARIANT)
PHASE_LIST = [p["tag"] for p in PLAN_PHASES]
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


def seed_of(phase, fx, extra=0):
    """Distinct, recorded, deterministic per (phase, fixture, schedule) from the FIXED plan phase list."""
    return 9_850_000 + 1000 * PHASE_LIST.index(phase) + 10 * ORDER.index(fx) + extra


def prim_xa(n, d2=False):
    f = G["fx"][n]
    xa = {"local_fw_draft": "natural", "local_fw_desig": f["desig"], "local_fw_selrec": f["selrec"],
          "local_fw_dumpwin": 1}
    if d2:
        xa["local_fw_d2"] = [0, f["L"] + 1]
    return xa


def lp_xa(n, d2=False):
    f = G["fx"][n]
    xa = {"local_fw_draft": "natural", "local_fw_desig": f["desig_lp"], "local_fw_selrec": f["selrec_lp"],
          "local_fw_dumpwin": 1}
    if d2:
        xa["local_fw_d2"] = [0, f["L_s"] + 1]
    return xa


ROLE = {"LW": "ref", "warm": "warm", "touch": "touch", "C": "C", "D": "D", "LC": "LC", "LD": "LD", "CV": "CV",
        "PS": "PS"}


def build_request(ph, r, k=0):
    """(prompt ids, forced ids, xargs or None, record dict) for one planned request."""
    n = r["fixture"]
    f = G["fx"][n]
    pre, fo = IDS[n]["prefix"], IDS[n]["forced"]
    kind, x = r["prompt"]
    prompt = pre + fo[:x] if kind == "prefix+forced" else list(pre)
    F = fo[r["forced"][0]: r["forced"][0] + r["forced"][1]] if r.get("forced") else []
    form = r["form"]
    tag = "CV-%s-%s" % (r["cv_mode"].replace(":", "_"), n) if form == "CV" else "%s-%s" % (ph["tag"], n)
    xa = None
    if form == "LW":
        xa = {"local_fw_ref": f["P"], "local_fw_dumpwin": 1, "local_fw_dumprange": [f["P"], f["P_L"]]}
    elif form in ("C", "D"):
        xa = prim_xa(n, form == "D")
    elif form in ("LC", "LD"):
        xa = lp_xa(n, form == "LD")
    elif form == "CV":
        xa = {"local_fw_draft": r["cv_mode"], "local_fw_snap": r["snap"], "local_fw_desig": [],
              "local_fw_selrec": []}
    elif form == "PS":
        xa = {"local_fw_draft": "natural", "local_fw_snap": [1], "local_fw_dumpwin": 1,
              "local_fw_dumprange": list(r["dumprange"]), "local_fw_desig": [], "local_fw_selrec": []}
    keyed = xa is not None
    if keyed:
        xa = dict({"forcewin_hooks": list(F), "local_fw_tag": tag}, **xa)
    seed = seed_of(ph["tag"], n, k)
    d = dict(kind="request", phase=ph["tag"], tag=tag, fixture=n, role=ROLE[form], form=form, salt=geo.salt(r["salt"]),
             salt_name=r["salt"], seed=seed, P=len(prompt), W=len(F), P_mod8=len(prompt) % 8, prefix_sha=sha(prompt),
             forced_sha=sha(F), prompt_sha256=sha_full(prompt), forced_sha256=sha_full(F),
             last_forced=(F[-1] if F else None), keyed=keyed, keys=sorted(xa) if xa else [],
             mode=(xa or {}).get("local_fw_draft"), d2=(xa or {}).get("local_fw_d2"),
             ref_from=(xa or {}).get("local_fw_ref"), dumprange=(xa or {}).get("local_fw_dumprange"),
             dumpwin=(xa or {}).get("local_fw_dumpwin"), snap=(xa or {}).get("local_fw_snap"),
             desig_n=len((xa or {}).get("local_fw_desig") or []),
             desig_sha=sha((xa or {}).get("local_fw_desig")) if (xa or {}).get("local_fw_desig") is not None
             else None, selrec_n=len((xa or {}).get("local_fw_selrec") or []),
             cached_expected=r["cached_expected"], c_bound=r["c_bound"], max_tokens=r["max_tokens"],
             batch=ph["tag"] if form in ("C", "D", "LC", "LD", "CV") else None, build=r.get("build"),
             build_len=len(prompt) if form == "LW" else None, construction="long" if form in ("C", "D") else
             ("short" if form in ("LC", "LD") else None), arm=form if form in ("C", "D", "LC", "LD") else None,
             cv_mode=r.get("cv_mode"), shared_block=r.get("shared_block"))
    assert len(prompt) == r["P"], (tag, len(prompt), r["P"])
    return prompt, F, xa, d


def send(prompt, F, xa, d, delay=0.0):
    if PLAN:
        PLANNED.append(d)
        return d
    if delay:
        time.sleep(delay)
    body = dict(model=MODEL, prompt=list(prompt), max_tokens=d["max_tokens"], temperature=1.0, top_p=0.95,
                seed=d["seed"], cache_salt=d["salt"], return_token_ids=True)
    if xa is not None:
        body["vllm_xargs"] = xa
    j, err, t0, t1 = _post("/v1/completions", body)
    ch = (j.get("choices") or [{}])[0]
    u = j.get("usage") or {}
    ids = ch.get("token_ids") or []
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    d = dict(d, t0=round(t0, 3), t1=round(t1, 3), secs=round(t1 - t0, 2), finish=ch.get("finish_reason"),
             completion_tokens=u.get("completion_tokens"), prompt_tokens=u.get("prompt_tokens"), cached_tokens=cached,
             cls=cg.classify(cached, d["cached_expected"]), resp_id=j.get("id"), returned_sha=sha(ids),
             ids_equal=(ids == list(F)) if (d["keyed"] and F) else None, error=err)
    rec(d)
    with lock:
        LAST[(d["salt_name"], d["fixture"])] = t1
    return d


def dump(tag):
    if PLAN:
        return
    p = subprocess.run(["docker", "exec", CONTAINER, "sh", "-c", "mkdir -p /tmp/mapguard && touch /tmp/mapguard/DUMP"],
                       capture_output=True, text=True)
    rec(dict(kind="dump", tag=tag, t=round(time.time(), 3), rc=p.returncode))
    time.sleep(6)


def left():
    return DEADLINE - time.time() if not PLAN else 1e9


def need_s(ph):
    """Pessimistic seconds for a phase (never-truncate rule)."""
    rs = ph["requests"]
    if ph["kind"] == "build":
        return sum(r["P"] for r in rs) / PF_RATE + 8 * DUMP_S + 30
    if ph["kind"] in ("touch", "ps"):
        return sum(r["P"] - r["cached_expected"] for r in rs) / PF_RATE + 8 * 2.0 + 30
    if ph["kind"] == "run":
        return sum(r["P"] - r["cached_expected"] for r in rs) / PF_RATE + max(r["max_tokens"] for r in rs) / DEC_RATE + \
            8 * DUMP_S + 30
    return 60.0


def is_lp(ph):
    return ph.get("form") in ("LC", "LD")


def reserve_after(i):
    """seconds kept for the PRIMARY phases after plan index i (LP batches are skipped first when time is short)."""
    return sum(need_s(p) for p in PLAN_PHASES[i + 1:] if p["kind"] != "gate" and p["tag"] != "V0" and not is_lp(p))


def ordered(ph):
    """[R-4] submission order of a run / touch / sentinel pass: oldest last-freed first (client t1 of the latest
    earlier request of the same fixture and salt); builds keep the plan's fixed order."""
    rs = list(ph["requests"])
    if ph["kind"] in ("run", "touch", "ps") and ph.get("form") != "CV":
        rs.sort(key=lambda r: (LAST.get((r["salt"], r["fixture"]), 0.0), ORDER.index(r["fixture"])))
    return rs


def V0():
    t0 = time.time()
    if PLAN:
        PLANNED.append(dict(kind="probe", n=4))
        phase_done("V0", t0)
        return
    G_ = {i["doc_id"]: i for i in json.load(open(GPQA))["items"]}
    exp = {}
    for ln in open(B1_PROBES):
        r = json.loads(ln)
        if r.get("kind") == "probe" and r.get("round") == 0:
            exp[r["id"]] = r["sha256"]
    for d in range(4):   # payload and canonicalisation as client_fc.py / client_hookval.py V0 (validated)
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
    phase_done("V0", t0)


def phase_done(tag, t0, extra=None):
    DONE[tag] = round(time.time() - t0, 1)
    PHASES.append(tag)
    d = dict(kind="phase", tag=tag, t0=round(t0, 3), t1=round(time.time(), 3), secs=round(time.time() - t0, 1))
    d.update(extra or {})
    if PLAN:
        PLANNED.append(d)
    rec(d)
    log(f"phase {tag} done in {time.time() - t0:.0f}s")
    dump(tag)


def run_phase(ph):
    t0 = time.time()
    rs = ordered(ph)
    items = []
    for k, r in enumerate(rs):
        extra = list(geo.CV_MODES).index(r["cv_mode"]) if r["form"] == "CV" else 0
        items.append(build_request(ph, r, extra))
    order = [d["tag"] for _, _, _, d in items]
    out = []
    if PLAN or ph.get("sequential"):
        for prompt, F, xa, d in items:
            out.append(send(prompt, F, xa, d))
    else:
        st = ph.get("stagger", 0.0)
        res = [None] * len(items)

        def go(i, it):
            res[i] = send(*it, delay=i * st)
        ths = [threading.Thread(target=go, args=(i, it)) for i, it in enumerate(items)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        out = res
    phase_done(ph["tag"], t0, dict(submission_order=order, stagger=ph.get("stagger"),
                                   sequential=bool(ph.get("sequential"))))
    return out


# ------------------------------------------------------------------ in-session quick checks (R-7)
def sched_records():
    """The guard's EngineCore scheduler log, read from the serving container (no file is modified)."""
    p = subprocess.run(["docker", "exec", CONTAINER, "sh", "-c", "cat /tmp/mapguard/ms_p*.jsonl"], capture_output=True,
                       text=True)
    return [r for r in (json.loads(x) for x in p.stdout.splitlines() if x.strip()) if r.get("kind") == "sched"]


def quick_check(ph, out):
    """rc 0 ok; 9 structural; 12 unsafe restoration; 13 insufficient concurrent decoding; 14 missing sentinel
    coverage. Writes a 'quick' record with the scheduler-side evidence."""
    if PLAN:
        return 0
    bad = [d["tag"] for d in out if d.get("error") or (d["keyed"] and d["W"] and d.get("ids_equal") is not True)]
    unsafe = [d["tag"] for d in out if d.get("cached_tokens") is None or d["cached_tokens"] % geo.BLOCK
              or d["cached_tokens"] > d["c_bound"]]
    recs = sched_records()
    info = cg.sched_requests(recs)
    allreq = [json.loads(x) for x in open(f"{OUT}/client.jsonl") if '"kind": "request"' in x]
    m = cg.map_client(allreq, info)
    rows = {x["tag"]: x for x in cg.evaluate(allreq, info, m)}
    s3 = [t for t in (d["tag"] for d in out) if t in rows and not (rows[t]["S3_prefill_only"] and
                                                                  rows[t]["S4_attributed"])]
    lead = None
    if ph["kind"] == "run" and ph.get("form") in ("C", "D", "LC", "LD"):
        ids = [m.get(d["tag"]) for d in out]
        if all(ids):
            first = {q: info[q]["first_dec_seq"] for q in ids if info[q].get("first_dec_seq") is not None}
            if len(first) == len(ids):
                last = max(first.values())
                P = {m[d["tag"]]: d["P"] for d in out}
                lead = 0
                for r in recs:
                    if r["seq"] == last:
                        for q, v in r["reqs"].items():
                            if q in P:
                                lead = max(lead, v[0] - v[1] - P[q])
                        break
    sentinel_missing = [d["tag"] for d in out if ph["kind"] == "ps" and d.get("cls") != "hit"]
    rc = 9 if bad else 12 if (unsafe or s3) else 13 if (lead is not None and lead >= geo.LEAD) else \
        14 if (sentinel_missing and ph["tag"] == "PS-c") else 0
    rec(dict(kind="quick", phase=ph["tag"], rc=rc, structural=bad[:8], unsafe_client=unsafe[:8],
             unsafe_scheduler=s3[:8], lead=lead, classes={d["tag"]: d.get("cls") for d in out},
             sentinel_missing=sentinel_missing, mapped=sum(1 for d in out if m.get(d["tag"]))))
    return rc


GATE = {}


def gate():
    """GATE-1R: runner/gate_step_fcr.sh copies stage A's records and runs analysis/gate_fcr.py in a CPU container.
    The engine stays IDLE while it runs (s2's build must survive to D2: no clean work is overlapped)."""
    t = time.time()
    rec(dict(kind="gate_start", t=round(t, 3)))
    if PLAN:
        PHASES.append("GATE-1R")
        DONE["GATE-1R"] = 0.0
        PLANNED.append(dict(kind="gate", gate="GATE-1R"))
        return True
    try:
        p = subprocess.run(["bash", GATE_STEP, OUT], capture_output=True, text=True,
                           timeout=max(60.0, left() - 60.0))
    except subprocess.TimeoutExpired as exc:
        p = subprocess.CompletedProcess(exc.cmd, 124, stdout=str(exc.stdout or ""), stderr="gate step timed out")
    v = None
    try:
        v = json.load(open(f"{OUT}/gate1r/gate1r.json")).get("verdict")
    except Exception:  # noqa: BLE001
        v = None
    ok = p.returncode == 0 and v == "PASS"
    rec(dict(kind="gate", gate="GATE-1R", ok=ok, verdict=v, rc=p.returncode, secs=round(time.time() - t, 1),
             tail=(p.stdout + p.stderr)[-1500:]))
    DONE["GATE-1R"] = round(time.time() - t, 1)
    PHASES.append("GATE-1R")
    return ok


def stop(why, rc):
    rec(dict(kind="stop", why=why, rc=rc, phases=PHASES, t=round(time.time(), 3)))
    end(rc)


def end(rc=0):
    req = [p for p in PHASE_LIST]
    rec(dict(kind="coverage", required=req, ran=[p for p in req if p in DONE], not_run=[p for p in req if p not in DONE],
             order=PHASES, left=round(left(), 1)))
    rec(dict(kind="end", t=round(time.time(), 3), rc=rc))
    if PLAN:
        json.dump(PLANNED, open(f"{OUT}/plan.json", "w"), indent=0)
        n = sum(1 for p in PLANNED if p.get("kind") == "request")
        tok = sum(p.get("W", 0) for p in PLANNED if p.get("kind") == "request")
        pf = sum(p.get("P", 0) - p.get("cached_expected", 0) for p in PLANNED if p.get("kind") == "request")
        print(f"plan ({VARIANT}): {n} requests, {tok} forced tokens, {pf} prompt tokens to prefill after the expected "
              f"restores; phases {PHASES}")
    sys.exit(rc)


def main():
    global MODEL
    if not PLAN:
        import requests
        MODEL = requests.get(f"{BASE}/v1/models", timeout=30).json()["data"][0]["id"]
    rec(dict(kind="start", t=round(time.time(), 3), plan=PLAN, deadline=DEADLINE, scope="first-contrast retry fcr1",
             variant=VARIANT, order=ORDER, wmax=geo.WMAX, phases=PHASE_LIST,
             salts={s: geo.salt(s) for s in sorted({p.get("salt") for p in PLAN_PHASES if p.get("salt")})}))
    dump("start")
    for i, ph in enumerate(PLAN_PHASES):
        if ph["kind"] == "probe":
            V0()
            continue
        if ph["kind"] == "gate":
            if not gate():
                stop("GATE-1R not PASS: no injected batch is run", 10)
            continue
        need = need_s(ph)
        res = reserve_after(i) if is_lp(ph) else 0.0
        if is_lp(ph) and LP_OFF[0]:
            rec(dict(kind="skip", phase=ph["tag"], why="an earlier LP batch was skipped"))
            continue
        if left() < need + res:
            rec(dict(kind="skip", phase=ph["tag"], left=round(left(), 1), need=round(need, 1), reserve=round(res, 1)))
            if is_lp(ph):
                LP_OFF[0] = True
                log(f"SKIP {ph['tag']} and every later LP batch (primary phases reserved)")
                continue
            stop("%s does not fit" % ph["tag"], 11)
        out = run_phase(ph)
        rc = quick_check(ph, out)
        if rc:
            stop("in-session stop after %s (rc %d)" % (ph["tag"], rc), rc)
    end(0)


if __name__ == "__main__":
    main()
