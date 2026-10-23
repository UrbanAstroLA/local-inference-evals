#!/usr/bin/env python3
"""CPU tests for the 22ao guard additions (no GPU, --network none, TRITON_INTERPRET=1).

Reuses the frozen B1 guard test rig (d7/step2/tests/test_mapguard.py: real B0/B1 KVCacheManager block ids,
real mapping builders under the Triton interpreter, the guard's own Ledger) and drives, per step, the real
22ao hook functions:
  L*  metadata site: consumed per-token length L against p + 1 (exact / over by the previous step's
      rejected drafts / under by one), counts and per-row records must equal independently computed truth.
  P*  shadow plants: every plant class must be planted and detected at every planted token, plants must
      not change any engine counter, and the planted subset must be deterministic.
  N*  signed consumed count: recorded, and zero violations from plants in the frozen counters.
Writes a JSON result file (argv[1]); exit 0 only if every test passes.
"""
import json
import os
import random
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
RIG = "/rig"     # d7/step2/tests (read-only mount)
sys.path.insert(0, RIG)
sys.path.insert(0, os.path.join(RIG, "..", "analysis"))

import torch  # noqa: E402
import test_mapguard as tm  # noqa: E402  (imports vllm.mapguard_hooks from the image: the 22ao module is mounted)
import b0emu  # noqa: E402
from b0emu import Emu  # noqa: E402

mg = tm.mg
assert "len_checked" in mg.CI, "22ao guard module not mounted"
RESULTS = []


def result(name, ok, **kw):
    RESULTS.append(dict(test=name, ok=bool(ok), **kw))
    print(("PASS " if ok else "FAIL ") + name + "  " + json.dumps(kw, default=str)[:500], flush=True)


def run(mode="exact", seed=1, max_steps=60, plant_mod=None):
    """One emulated session. mode: exact | over (L = p + 1 + rejected drafts of the request's previous step)
    | under (L = p + 1 - 1 on every 5th token)."""
    if plant_mod is not None:
        mg.PLANT_MOD = plant_mod
    rnd = random.Random(seed)
    emu = Emu(seed=seed, budget=2048, concurrency=3)
    g = tm.Guard()
    A = [700000 + i for i in range(9000)]
    emu.add(A, 6, rid="A")
    emu.add([710000 + i for i in range(5000)], 10, rid="B")
    emu.add(A[:8704] + [720000 + i for i in range(300)], 6, rid="C")
    emu.add([730000 + i for i in range(300)], 12, rid="D")
    emu.add([740000 + i for i in range(4400)], 5, rid="E")
    emu.add([750000 + i for i in range(132)], 8, rid="F")
    prev_rej = {}
    truth = dict(pos=0, neg=0, checked=0, rows=0)
    lrec0 = int(g.dev.nlrec[0])
    for s in range(1, max_steps + 1):
        if not (emu.running or emu.waiting):
            break
        so, plan = emu.schedule()
        if not so.num_scheduled_tokens:
            emu.complete(plan, {})
            continue
        g.begin(so)
        req_ids = b0emu.sort_batch_req_ids(so.num_scheduled_tokens, so.scheduled_spec_decode_tokens, 4)
        ns = [so.num_scheduled_tokens[r] for r in req_ids]
        uniform = all(so.scheduled_spec_decode_tokens.get(r) for r in req_ids) and len(set(ns)) == 1
        pad_to = None
        if uniform:
            ntok = sum(ns)
            pad_to = next(c for c in (4, 8, 16, 24, 32, 40, 48, 56, 64) if c >= ntok)
            pad_to = None if pad_to == ntok else pad_to
        out = emu.runner_step(so, pad_to=pad_to)
        g.upload(out["req_ids"])
        # the consumed length tensor, built from the runner's own positions and per-request offsets
        pos = out["pos"].to(torch.int64)
        qsl = out["qsl_consumed"].to(torch.int64).tolist()
        L = pos + 1
        n_real = int(out["ntok"])
        width = int(out["sel"][0].shape[1])
        for i, rid in enumerate(out["req_ids"]):
            a, b = qsl[i], qsl[i + 1]
            if mode == "over":
                L[a:b] += prev_rej.get(rid, 0)
            elif mode == "under":
                for t in range(a, b):
                    if t % 5 == 0:
                        L[t] -= 1
            truth["rows"] += 1
        d = (L[:n_real] - (pos[:n_real] + 1))
        truth["pos"] += int((d > 0).sum())
        truth["neg"] += int((d < 0).sum())
        truth["checked"] += n_real
        before = g.dev.cnt.clone()

        class Lyr:
            layer_name = f"model.layers.{g.layers[0]}.self_attn.mla_attn"
        mg.S.pos_buf = out["pos"]
        mg.hook_mla_len(Lyr(), L.to(torch.int32), int(out["ntok_pad"]), width)
        dlen = (g.dev.cnt - before)[mg.S_MLA_LEN]
        if any(int(dlen[mg.CI[c]]) for c in ("len_checked",)) and int(dlen[mg.CI["len_checked"]]) != n_real:
            truth.setdefault("checked_mismatch_steps", []).append(s)
        g.run_checks(out)
        accepted = {}
        for rid, pl in plan.items():
            if pl["k"] or (pl["st"]["prefill_done"] and not pl["st"].get("first_decode")):
                accepted[rid] = rnd.randint(1, pl["k"] + 1)
        emu.commit(out, accepted, plan)
        rej = emu.complete(plan, accepted)
        prev_rej = {rid: int(v) for rid, v in (rej or {}).items()}
    cnt = g.dev.cnt
    get = lambda site, c: int(cnt[site, mg.CI[c]])
    lrec = g.dev.lrec[lrec0:int(g.dev.nlrec[0])].tolist()
    return dict(g=g, truth=truth, lrec=lrec,
                len={c: get(mg.S_MLA_LEN, c) for c in ("len_checked", "len_pos", "len_neg")},
                nsa={c: get(mg.S_MLA_R, c) for c in ("nsa_pos", "nsa_neg")},
                plants={c: get(mg.S_MLA_R, c) for c in ("pa_planted", "pa_det", "pb_planted", "pb_det",
                                                        "pc_planted", "pc_det", "pd_planted", "pd_det")},
                viol={f"{mg.SITES[s_]}.{c}": int(cnt[s_, mg.CI[c]]) for s_ in range(len(mg.SITES))
                      for c in mg.VIOLATION_COUNTERS if int(cnt[s_, mg.CI[c]])},
                hist=g.dev.lenhist.tolist())


def main():
    out_path = sys.argv[1]
    # L1: exact lengths -> no discrepancy; every active token checked; one record per batch row per step
    r = run("exact")
    t = r["truth"]
    result("L1 exact lengths: len_pos = len_neg = 0, all tokens checked", r["len"]["len_pos"] == 0 and r["len"]["len_neg"] == 0
           and r["len"]["len_checked"] == t["checked"] and "checked_mismatch_steps" not in t,
           len=r["len"], truth=t)
    result("L1b exact lengths: per-row records = batch rows, all d = 0", len(r["lrec"]) == t["rows"]
           and all(row[mg.LREC_COLS.index("d")] == 0 for row in r["lrec"]), records=len(r["lrec"]), rows=t["rows"])
    # L2: over by the previous step's rejected drafts (the stock upper-bound hypothesis)
    r2 = run("over")
    t2 = r2["truth"]
    result("L2 over-estimate: len_pos equals truth and > 0; len_neg = 0", r2["len"]["len_pos"] == t2["pos"] > 0
           and r2["len"]["len_neg"] == 0, len=r2["len"], truth=t2, hist=r2["hist"])
    dvals = [row[mg.LREC_COLS.index("d")] for row in r2["lrec"]]
    result("L2b per-row records carry the signed d (> 0 present, none < 0)", any(v > 0 for v in dvals)
           and not any(v < 0 for v in dvals), d_hist={v: dvals.count(v) for v in sorted(set(dvals))})
    # L3: under-estimate is reported with its sign
    r3 = run("under")
    t3 = r3["truth"]
    result("L3 under-estimate: len_neg equals truth and > 0; len_pos = 0", r3["len"]["len_neg"] == t3["neg"] > 0
           and r3["len"]["len_pos"] == 0, len=r3["len"], truth=t3)
    # P1: plants at a dense rate: every class planted and detected at every planted token
    rp = run("exact", plant_mod=7)
    pl = rp["plants"]
    ok = all(pl[f"p{c}_planted"] > 0 and pl[f"p{c}_det"] == pl[f"p{c}_planted"] for c in "abcd")
    result("P1 every plant class planted and detected at every planted token (PLANT_MOD 7)", ok, plants=pl)
    # P2: plants never change engine counters (same session with plants off)
    rq = run("exact", plant_mod=10 ** 9)
    result("P2 frozen violation counters identical with plants on and off", rp["viol"] == rq["viol"],
           with_plants=rp["viol"], without=rq["viol"])
    result("P2b frozen negative control still clean (no violation counter set)", rp["viol"] == {},
           viol=rp["viol"])
    # P3: the default rate plants deterministically (same counts twice)
    ra = run("exact", plant_mod=97)
    rb = run("exact", plant_mod=97)
    result("P3 default PLANT_MOD 97: deterministic planted subset, all detected", ra["plants"] == rb["plants"]
           and all(ra["plants"][f"p{c}_det"] == ra["plants"][f"p{c}_planted"] for c in "abcd"), plants=ra["plants"])
    # N1: signed consumed count recorded (exact lengths; reported, not asserted zero: emu nsa is the real kernel's)
    result("N1 signed consumed count counters present", all(k in r["nsa"] for k in ("nsa_pos", "nsa_neg")), nsa=r["nsa"])
    ok_all = all(x["ok"] for x in RESULTS)
    json.dump(dict(source=tm.SRC, guard_sha=None, results=RESULTS, ok=ok_all), open(out_path, "w"), indent=1)
    print("ALL PASS" if ok_all else "SOME FAILED", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
