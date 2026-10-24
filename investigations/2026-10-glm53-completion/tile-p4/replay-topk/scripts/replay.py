#!/usr/bin/env python3
"""replay-topk driver, run INSIDE one stock image by in_container.sh (PROTOCOL.md section 6).

usage: replay.py --version v091|v070 --out DIR [--deadline-s S] [--dry]

The configuration below is frozen with the protocol (pre-registered); there are no knobs on the command line other
than the soft deadline.  GPU mode writes DIR/env.json, DIR/r1.json, DIR/r2.json (R3 inside r2.json for v091).
--dry (CUDA hidden): imports, the served window builder and index construction under TRITON_INTERPRET=1 on CPU
tensors, argument and shape construction; no CUDA op, no compile.  Writes DIR/dry-<version>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SEED = 20261024
CONFIG = {
    "v091": dict(
        r1=dict(regimes=["K", "C", "T"], n_calls=24, n_warm=3, keep_for_r3="K", seed=SEED),
        r2=dict(q_regimes=[("flat", 1.0), ("peaked", 4.0)], r2_warm=2, r2_repeats=8, r2_reloc=3, ref_rows=64,
                r3_calls=8, seed=SEED),
        r1_share=0.35),
    "v070": dict(
        r1=dict(regimes=["C", "T"], n_calls=20, n_warm=3, keep_for_r3=None, seed=SEED),
        r2=dict(q_regimes=[("flat", 1.0), ("peaked", 4.0)], r2_warm=2, r2_repeats=8, r2_reloc=3, ref_rows=64,
                r3_calls=0, seed=SEED),
        r1_share=0.20),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["v091", "v070"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline-s", type=float, default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.dry:
        os.environ["TRITON_INTERPRET"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import rt_common as rc
    import r1_topk
    import r2_mla
    cfg = CONFIG[a.version]
    if a.dry:
        doc = dict(version=a.version, mode="dry", config=cfg, cuda_available=torch.cuda.is_available(), errors=[])
        for name, fn in (("r1", lambda: r1_topk.dry_r1(torch, log)), ("r2", lambda: r2_mla.dry_r2(torch, a.version, log))):
            try:
                doc[name] = fn()
            except Exception as exc:  # noqa: BLE001
                doc["errors"].append(dict(part=name, error=repr(exc)[:400], tb=traceback.format_exc()[-1500:]))
        doc["status"] = "error" if doc["errors"] else "complete"
        rc.write_json(os.path.join(a.out, f"dry-{a.version}.json"), doc)
        print(json.dumps(dict(dry=a.version, status=doc["status"], errors=len(doc["errors"]))))
        return 0 if doc["status"] == "complete" else 5
    if not torch.cuda.is_available():
        print("REFUSED: no CUDA device visible (GPU mode)")
        return 2
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    t0 = time.time()
    env = rc.env_record(torch, dev)
    env.update(version=a.version, config=cfg, started=t0)
    rc.write_json(os.path.join(a.out, "env.json"), env)
    total = rc.Deadline(a.deadline_s)
    r1_budget = None if a.deadline_s is None else a.deadline_s * cfg["r1_share"]
    dl1 = rc.Deadline(r1_budget)
    log(f"R1 start ({a.version}), soft budget {r1_budget}")
    r1doc, kept = r1_topk.run_r1(torch, dev, cfg["r1"], dl1, log)
    r1doc.update(version=a.version, seconds=round(time.time() - t0, 2))
    if a.version == "v070":
        r1doc["scope_note"] = ("binary-identical confirmation at the v0.9.1 chunk geometry; NOT the v0.7.0 served "
                               "geometry (DCP=2 localises the window to <= 511 pools per rank for p < 4,096)")
    rc.write_json(os.path.join(a.out, "r1.json"), r1doc)
    if kept is None and a.version == "v091":
        log("R3 input: regime K unavailable; R3 is not run (no substitute regime: pre-registered)")
    t2 = time.time()
    log(f"R2 start ({a.version}), soft left {total.left():.0f}s")
    r2doc = r2_mla.run_r2(torch, dev, a.version, cfg["r2"], total, log,
                          r1_kept=kept if a.version == "v091" else None)
    r2doc.update(seconds=round(time.time() - t2, 2))
    rc.write_json(os.path.join(a.out, "r2.json"), r2doc)
    print(json.dumps(dict(version=a.version, r1=r1doc["status"], r2=r2doc["status"],
                          elapsed=round(time.time() - t0, 1))), flush=True)
    return 0 if (r1doc["status"] == "complete" and r2doc["status"] == "complete") else 4


if __name__ == "__main__":
    sys.exit(main())
