#!/usr/bin/env python3
"""Synthetic B+ run folders for analysis dry runs (CHECKS.md). Host, standard library. No GPU, no server.
Generated continuations are random token ids with placeholder text (no GPQA content). The real reference server logs
of each engine stand in for the session's server log (config identity and warm-up checks see real logs); request times
are placed after each log's ready line.

usage: make_synthetic_run.py SCENARIO OUT_DIR
  scenarios: all-finish | none-finish | engine-effect | state-dominated | null
             neg-wrong-seed | neg-prompt-hash | neg-missing-request | neg-raw-mismatch | neg-image | neg-no-ready
             c1-only | neg-watchdog
"""
import datetime
import gzip
import hashlib
import json
import os
import random
import re
import shutil
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = "<runs>"
REFLOG = {"v070": f"{B}/rep070/run-F/server-s1.log", "v091": "<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1.server.log",
          "b1": f"{B}/b1-screen/run-s1236/server-s1.log"}
IMAGE = {"v070": "sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629",
         "v091": "sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07",
         "b1": "sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a"}
H = 65536
THINK_CLOSE = 154842
READY = "GLM release startup warmup complete; container is ready."


def ids_sha(ids):
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


def ts_utc(line):
    m = re.search(r" (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) ", line)
    return datetime.datetime(2026, *map(int, m.groups()), tzinfo=datetime.timezone.utc).timestamp() if m else None


def p_finish(scn, e, pid):
    n = int(pid[1:])
    if scn == "all-finish":
        return 1.0
    if scn == "none-finish":
        return 0.0
    if scn == "state-dominated":
        return 0.0 if n <= 6 else 1.0
    if scn == "null":
        return 0.45
    return {"v070": 0.75, "v091": 0.2, "b1": 0.2}[e]     # engine-effect and the negative cases


def main():
    scn, out = sys.argv[1], sys.argv[2]
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "raw"))
    os.chmod(os.path.join(out, "raw"), 0o700)
    plan = json.load(open(f"{PKG}/out/plan.json"))
    man = {m["prefix_id"]: m for m in json.load(open(f"{PKG}/out/prefix_manifest.json"))["prefixes"]}
    rng = random.Random(int(hashlib.sha256(scn.encode()).hexdigest()[:8], 16))
    for e in ("v070", "v091", "b1"):
        d = os.path.join(out, e)
        os.makedirs(d)
        if scn == "c1-only" and e == "b1":
            json.dump({"engine": e, "budget_stop": True, "started": False}, open(f"{d}/session.json", "w"))
            continue
        lines = open(REFLOG[e], errors="replace").read().splitlines()
        if scn == "neg-no-ready" and e == "v091":
            lines = [l for l in lines if READY not in l]
        open(f"{d}/server.log", "w").write("\n".join(lines) + "\n")
        r0 = [i for i, l in enumerate(lines) if READY in l]
        t_ready = [t for t in (ts_utc(l) for l in (lines[:r0[0]] if r0 else lines)) if t][-1]
        img = IMAGE[e] if not (scn == "neg-image" and e == "b1") else IMAGE["v091"]
        json.dump([{"Image": img, "HostConfig": {"PortBindings": {"8001/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8001"}]}}}],
                  open(f"{d}/inspect.json", "w"))
        sess = {"engine": e, "started": True, "ready": True, "preflight_ok": True, "completed": True, "client_rc": 0,
                "watchdog": scn == "neg-watchdog" and e == "b1", "budget_stop": False}
        json.dump(sess, open(f"{d}/session.json", "w"))
        recs = [{"kind": "start", "engine": e, "t": t_ready + 30}]
        slots = [t_ready + 60 + 0.2 * i for i in range(8)]
        raw = []
        for r in plan["order"]:
            p, seed = r["prefix_id"], r["seeds"][e]
            fin = rng.random() < p_finish(scn, e, p)
            if fin:
                L = rng.randint(2000, H - 1000)
                gen = [rng.randint(1000, 120000) for _ in range(L)] + [THINK_CLOSE] + [rng.randint(1000, 120000) for _ in range(300)] + [154827]
                gen = gen[:H]
                fr, sr = "stop", 154827
            else:
                if rng.random() < 0.5:
                    base = [rng.randint(1000, 120000) for _ in range(rng.randint(5000, 40000))]
                    unit = [rng.randint(1000, 120000) for _ in range(rng.choice((2, 35, 514)))]
                    gen = (base + unit * (H // len(unit) + 1))[:H]
                else:
                    gen = [rng.randint(1000, 120000) for _ in range(H)]
                fr, sr = "length", None
            tc = gen.index(THINK_CLOSE) if THINK_CLOSE in gen else None
            text = "x" * (3 * (tc if tc is not None else len(gen))) + ("</think>" + "y" * 900 if tc is not None else "")
            i = min(range(8), key=lambda q: slots[q])
            t0 = max(slots[i], recs[-1].get("t0", 0) + 0.01 if len(recs) > 1 else slots[i])
            secs = len(gen) / rng.uniform(25, 55)
            slots[i] = t0 + secs + 0.01
            ps = man[p]["prefix_ids_sha256"]
            rec = {"kind": "request", "engine": e, "k": r["k"], "prefix_id": p, "j": r["j"], "seed": seed, "prompt_len": man[p]["prefix_len"],
                   "prompt_ids_sha256": ps, "params_sha256": plan["params_sha256"], "t0": round(t0, 3), "t1": round(t0 + secs, 3),
                   "secs": round(secs, 2), "http_status": 200, "error": None, "resp_id": f"cmpl-{e}-{r['k']}", "finish_reason": fr,
                   "stop_reason": sr, "prompt_tokens": man[p]["prefix_len"], "completion_tokens": len(gen),
                   "cached_tokens": None if e == "v070" else 0, "returned_prompt_ids_sha256": ps, "gen_ids_len": len(gen),
                   "gen_ids_sha256": ids_sha(gen), "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "text_chars": len(text),
                   "think_close_at": tc}
            if scn == "neg-wrong-seed" and e == "v091" and r["k"] == 7:
                rec["seed"] = seed + 1
            if scn == "neg-prompt-hash" and e == "v070" and r["k"] == 3:
                rec["returned_prompt_ids_sha256"] = "0" * 64
            recs.append(rec)
            rw = {"engine": e, "k": r["k"], "prefix_id": p, "j": r["j"], "seed": rec["seed"], "text": text, "token_ids": gen}
            if scn == "neg-raw-mismatch" and e == "b1" and r["k"] == 11:
                rw["token_ids"] = gen[:-1]
            raw.append(rw)
        if scn == "neg-missing-request" and e == "v091":
            recs = [x for x in recs if x.get("k") != 20]
            raw = [x for x in raw if x["k"] != 20]
        recs.append({"kind": "end", "engine": e, "t": max(x.get("t1", 0) for x in recs) + 1, "failures": []})
        with open(f"{d}/requests.jsonl", "w") as f:
            for x in sorted(recs, key=lambda x: (x.get("t0", x.get("t", 0)))):
                f.write(json.dumps(x, sort_keys=True) + "\n")
        with gzip.open(os.path.join(out, "raw", f"{e}.raw.jsonl.gz"), "wt") as f:
            for x in raw:
                f.write(json.dumps(x) + "\n")
    print(f"synthetic run {scn} -> {out}")


if __name__ == "__main__":
    sys.exit(main())
