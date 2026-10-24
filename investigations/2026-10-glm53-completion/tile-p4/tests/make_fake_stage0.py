#!/usr/bin/env python3
"""Fabricate stage-0 attempts in the real record formats (client.jsonl; p4log req / blocks / entry_init / sched records;
PRIVATE fw add log, forcewin req_<tag>_r0.pt, p4_<tag>_r0.npz with layer digests, entry_<tag>_r<rank>.npz) for the
dry runs of analysis/analyze_stage0.py.  CPU container (torch writes the .pt files).  No benchmark data.

usage: make_fake_stage0.py OUTROOT  -> OUTROOT/<scenario>/{runs-stage0/stage0-attempt-1, PRIVATE-runs-stage0/...}

Scenarios:
  pass        V1 and V2 identical between repeats -> both REPRODUCIBLE (R03-close's row holds a null-block
              placeholder whose digests differ by repeat: skipped, as the shared null block is not the request's state)
  v2enq       V2 repeat 2: one scheduled-token count differs at forced step offset 4 -> V2 fails G-enq (V1 passes)
  v2order     V2 repeat 2: two forced requests enqueued in the other order -> V2 fails G-enq ("enqueue order")
  entry       V1 repeat 2: tensor "mla.0" logical block 3 differs at point 1 (point 0 equal); V2 repeat 2: "kda.0#1"
              differs at point 0 -> both fail G-entry (V1 after equal cached state, V2 in the cached state)
  prov        V1 repeat 1: a forced request's cached block is not its warm request's block -> V1 fails G-entry
  out         both variants: E3 entropy differs at o = 7 of R02-deep in repeat 2, layer digests differ from the third
              digest layer on -> both fail G-out; first differing layer state reported; steps equal
  outcov      V2 repeat 1: E1 not staged at o = 0 for R01-early -> V2 fails G-out (missing coverage never passes)
  missing     V2 repeat 2 not run -> V2 INCOMPLETE
  stop        V1 repeat 2: G-enq differs AND an entry digest differs -> G-entry / G-out "not evaluated"
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_fake_sessions as MF  # noqa: E402  (forcewin COLS)

WINDOWS = ["R01-early", "R01-deep", "R01-close", "R02-early", "R02-deep", "R02-close", "R03-deep", "R03-close"]
P = {"early": 17409, "deep": 95745, "close": 113940}
KIND = {"close": 0, "deep": 1, "early": 2}
V1_ORDER = sorted(WINDOWS, key=lambda w: (w.split("-")[0], KIND[w.split("-")[1]]))
REPLIES = ["R01", "R02", "R03"]
BS = 4352
N = 96                       # stored positions per window (>= 64)
H = 8
LAYERS = [3, 12, 23, 34]
TENS = [("mla.0", 0, 4), ("idx.0", 0, 2), ("kda.0#1", 1, 1)]
NE_BLK = 48
NPT = 4                      # 3 prefill points + the entry point
SCHED_COLS = ["req_id", "req_idx", "uid", "tag", "n_sched", "n_computed", "n_computed_prefill", "prefill_len",
              "prefilling", "n_draft"]


def rng(*key):
    return np.random.default_rng(abs(hash(("s0fake",) + key)) % (2 ** 31))


def cached_blocks(wid):
    p = P[wid.split("-")[1]]
    return (p - 1) // BS - 1


def write(root, scen):
    ao = os.path.join(root, scen, "runs-stage0", "stage0-attempt-1")
    po = os.path.join(root, scen, "PRIVATE-runs-stage0", "stage0-attempt-1")
    for d in (os.path.join(ao, "p4log"), os.path.join(po, "fw"), os.path.join(po, "p4")):
        os.makedirs(d, exist_ok=True)
    cl, logs0, logs1, sched, adds = [], [], [], [], []
    cl.append(dict(kind="start", batch=1, windows=WINDOWS, v1_order=V1_ORDER))
    for rk, lg in ((0, logs0), (1, logs1)):
        lg.append(dict(kind="entry_init", rank=rk, covered=len(TENS), uncovered=[], n_groups=2,
                       tensors=[dict(layer=n, group=g, covered=True, ppb=p) for n, g, p in TENS]))
    uid = [0]
    step = [0]
    reps = [("V1", 1), ("V1", 2), ("V2", 1), ("V2", 2)]
    if scen == "missing":
        reps = reps[:3]
    for var, rep in reps:
        rname = f"{var}-r{rep}"
        phys0 = 1000 * (len(cl) + 1)        # physical block ids differ per replicate (allocation history)
        warm_ids = {}
        for k, reply in enumerate(REPLIES):
            tag = f"W-{rname}-{reply}"
            uid[0] += 1
            logs0.append(dict(kind="req", uid=uid[0], tag=tag, req_idx=k, W=0))
            adds.append(dict(kind="add", tag=tag, idx=k))
            nb = 27
            ids = [phys0 + 100 * k + i for i in range(nb)]
            warm_ids[reply] = ids
            for c in range(2):
                step[0] += 1
                sched.append(dict(step=step[0], nt=2048, ntp=2048, nr=1, nrp=1, ndraft=0, mode="none", binds=[["prefill", 2048]],
                                  rows=[[f"cmpl-w{uid[0]}-0", k, uid[0], tag, 2048, 2048 * c, 2048 * c, 113940, 1, 0]]))
            logs0.append(dict(kind="blocks", tag=tag, uid=uid[0], groups=[ids, ids], block_sizes=[BS, BS]))
            cl.append(dict(kind="warm", replicate=rname, tag=tag, ok=True))
        order = V1_ORDER if var == "V1" else list(WINDOWS)
        if scen == "v2order" and rname == "V2-r2":
            order = list(WINDOWS)
            order[1], order[2] = order[2], order[1]
        tags = {w: f"{var}-{w}-r{rep}" for w in WINDOWS}
        uids = {}
        for i, w in enumerate(order):
            uid[0] += 1
            uids[w] = uid[0]
            logs0.append(dict(kind="req", uid=uid[0], tag=tags[w], req_idx=i, W=512))
            adds.append(dict(kind="add", tag=tags[w], idx=i))
        groups = (lambda ws: [ws]) if var == "V1" else (lambda ws: [ws])
        batches = [[w] for w in order] if var == "V1" else [order]
        steps_of = {}
        for bw in batches:
            for c in range(3):                       # remainder prefill chunks
                step[0] += 1
                rows = []
                for w in bw:
                    ns = 2048 if c < 2 else 257
                    if scen == "v2enq" and rname == "V2-r2" and c == 1 and w == bw[2]:
                        ns = 2047
                    rows.append([f"cmpl-x-{uids[w]}", order.index(w), uids[w], tags[w], ns, 0, 2048 * c,
                                 P[w.split("-")[1]], 1, 0])
                sched.append(dict(step=step[0], nt=2048 * len(bw), ntp=2048 * len(bw), nr=len(bw), nrp=len(bw), ndraft=0,
                                  mode="none", binds=[["prefill", 2048]], rows=rows))
                for w in bw:
                    steps_of.setdefault(w, []).append(step[0])
            for d in range(30):                      # decode
                step[0] += 1
                nt = 4 * len(bw)
                if scen == "stop" and rname == "V1-r2" and d == 3 and bw == [order[0]]:
                    nt = 5
                sched.append(dict(step=step[0], nt=nt, ntp=nt, nr=len(bw), nrp=len(bw), ndraft=3 * len(bw), mode="FULL",
                                  binds=[], rows=[[f"cmpl-x-{uids[w]}", order.index(w), uids[w], tags[w], 4, 0, 0,
                                                   P[w.split("-")[1]], 0, 3] for w in bw]))
                for w in bw:
                    steps_of.setdefault(w, []).append(step[0])
        for w in WINDOWS:
            tag = tags[w]
            kc = cached_blocks(w)
            reply = w.split("-")[0]
            ids = warm_ids[reply][:kc] + [phys0 + 900 + 10 * WINDOWS.index(w) + i for i in range(3)]
            if scen == "prov" and rname == "V1-r1" and w == "R02-deep":
                ids[2] = 77
            if w == "R03-close":                    # a null-block placeholder in the row (both repeats): skipped
                ids[5] = 0
            logs0.append(dict(kind="blocks", tag=tag, uid=uids[w], groups=[ids, ids], block_sizes=[BS, BS]))
            # forcewin record (E3): identical between repeats except the planted difference
            g = rng(var, w, "fw")
            m = np.zeros((N, len(MF.COLS)), np.float32)
            m[:, :20] = g.standard_normal((N, 20)).astype(np.float32)
            m[:, MF.C["entropy"]] = g.random(N).astype(np.float32)
            st_ids = steps_of[w]
            for o in range(N):
                m[o, MF.C["step"]] = st_ids[min(2 + o // 4, len(st_ids) - 1)]
                m[o, MF.C["row"]] = o % 4
            if scen == "out" and rep == 2 and w == "R02-deep":
                m[7, MF.C["entropy"]] += np.float32(1e-3)
            rec = dict(tag=tag, cols=MF.COLS, metrics=torch.from_numpy(m), nmeas=torch.ones(N, dtype=torch.int32))
            torch.save(rec, os.path.join(po, "fw", f"req_{tag}_r0.pt"))
            gm = rng(var, w, "p4")
            moe = gm.integers(0, 2 ** 16, (3, N, H)).astype(np.uint16)
            have = np.ones((3, N), np.int32)
            if scen == "outcov" and rname == "V2-r1" and w == "R01-early":
                have[:, 0] = 0
            ld = gm.integers(-2 ** 40, 2 ** 40, (N, len(LAYERS), 2)).astype(np.int64)
            if scen == "out" and rep == 2 and w == "R02-deep":
                ld[7, 2:] += 1
            np.savez(os.path.join(po, "p4", f"p4_{tag}_r0.npz"), moe=moe, moe_have=have, ldig=ld,
                     ldig_have=np.ones(N, np.int32), ldig_layers=np.array(LAYERS, dtype=np.int64))
            for rk in (0, 1):
                ge = rng(var, w, "entry", rk)
                ed = ge.integers(-2 ** 40, 2 ** 40, (NPT, len(TENS), NE_BLK + 1, 2)).astype(np.int64)
                if w == "R03-close":                # null placeholder column: shared scratch, differs by repeat
                    ed[:, :, 5, :] += rep
                if scen in ("entry", "stop") and rname == "V1-r2" and w == "R01-close" and rk == 0:
                    ed[1, 0, 3, 0] += 1
                if scen == "entry" and rname == "V2-r2" and w == "R03-deep" and rk == 1:
                    ed[0, 2, 1, 1] += 1
                eb = np.full((NPT, 2, NE_BLK), -1, np.int32)
                for pt in range(NPT):
                    for gg in range(2):
                        eb[pt, gg, :len(ids)] = ids
                pts = np.array([[st_ids[0], 0, kc * BS, kc * BS, 2048], [st_ids[1], 0, kc * BS + 2048, kc * BS + 2048, 2048],
                                [st_ids[2], 0, kc * BS + 4096, kc * BS + 4096, 257],
                                [st_ids[3], 1, kc * BS + 4353, kc * BS + 4353, 4]], dtype=np.int64)
                np.savez(os.path.join(po, "p4", f"entry_{tag}_r{rk}.npz"), edig=ed, eblk=eb, points=pts,
                         point_cols=np.array(["step", "kind", "n_computed", "n_computed_prefill", "n_sched"]),
                         tensors=np.array([t[0] for t in TENS]), tensor_group=np.array([t[1] for t in TENS]),
                         tensor_ppb=np.array([t[2] for t in TENS]))
                (logs0 if rk == 0 else logs1).append(dict(kind="entry_flushed", tag=tag, uid=uids[w], points=NPT))
            cl.append(dict(kind="forced", replicate=rname, variant=var, window=w, tag=tag, ok=True))
        cl.append(dict(kind="replicate_done", replicate=rname, warm_ok=True, forced_ok=True))
    cl.append(dict(kind="end", replicates_done=[f"{v}-r{r}" for v, r in reps]))
    with open(os.path.join(ao, "client.jsonl"), "w") as f:
        for d in cl:
            f.write(json.dumps(d) + "\n")
    with open(os.path.join(ao, "p4log", "p4_r0_p1.jsonl"), "w") as f:
        for d in logs0:
            f.write(json.dumps(d) + "\n")
    with open(os.path.join(ao, "p4log", "p4_r1_p2.jsonl"), "w") as f:
        for d in logs1:
            f.write(json.dumps(d) + "\n")
    with open(os.path.join(ao, "p4log", "sched_r0_p1.jsonl"), "w") as f:
        f.write(json.dumps(dict(kind="sched", cols=SCHED_COLS, steps=sched)) + "\n")
    with open(os.path.join(po, "fw", "fw_r0_p1.jsonl"), "w") as f:
        for d in adds:
            f.write(json.dumps(d) + "\n")
    return ao


def main():
    out = sys.argv[1]
    for scen in ("pass", "v2enq", "v2order", "entry", "prov", "out", "outcov", "missing", "stop"):
        write(out, scen)
        print("fabricated", scen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
