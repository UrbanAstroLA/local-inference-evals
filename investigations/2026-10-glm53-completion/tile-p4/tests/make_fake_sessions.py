#!/usr/bin/env python3
"""Fabricate P4 session outputs in the real record formats (forcewin .pt, local-p4 .npz and steprec, step logs,
client.jsonl, outcome.json, G4 verdict) for the analysis dry runs.  CPU container (torch writes the .pt files).

usage: make_fake_sessions.py OUTROOT      -> OUTROOT/<scenario>/{runs,PRIVATE-runs,PRIVATE-fixtures}

Scenarios (each a separate root; 4 windows in 2 batches of 2, W = 512, small hidden / vocab):
  propagation   A == A == A2 bitwise; B differs in E1 (every row of decode steps), E2, E3 (all positions) and E4
  attenuation   B differs in E1 only; E2, E3 and E4 equal to A
  null          B == A everywhere
  gatefail      as propagation, but session B's outcome has gates_ok false
  noisy         A not bit-stable (A vs A2 E3 noise at most positions); B shifted far beyond the noise
  noisynull     A not bit-stable; B carries only the same noise (no shift), E1 shifted
  shapes        as propagation, plus 40 steps of B logged with another graph mode: their positions are excluded
"""
import json
import os
import sys

import numpy as np
import torch

W, H, V = 512, 16, 1000
NT = 5
COLS = (["lp_pre_%d" % i for i in range(NT)] + ["mass_lt_%d" % i for i in range(NT)]
        + ["mass_le_%d" % i for i in range(NT)] + ["p_post_%d" % i for i in range(NT)]
        + ["elig_lo_%d" % i for i in range(NT)] + ["elig_hi_%d" % i for i in range(NT)]
        + ["entropy", "argmax", "top1_p", "argmax_is_forced", "argmax_is_term", "step", "row", "n_rows",
           "accepted", "kind", "post_ok", "out_of_nucleus"])
C = {n: i for i, n in enumerate(COLS)}
WINDOWS = ["R01-early", "R01-close", "R02-deep", "R02-close"]
BATCHES = [["R01-early", "R01-close"], ["R02-deep", "R02-close"]]
SCHED = [3, 2, 3, 1]


def base_metrics(wid):
    g = np.random.default_rng(abs(hash(wid)) % (2 ** 31))
    m = np.zeros((W, len(COLS)), np.float32)
    m[:, :NT] = -g.random((W, NT)).astype(np.float32) * 10
    m[:, C["entropy"]] = g.random(W).astype(np.float32)
    m[:, C["top1_p"]] = g.random(W).astype(np.float32)
    m[:, C["argmax"]] = g.integers(0, V, W).astype(np.float32)
    m[:, C["p_post_4"]] = g.random(W).astype(np.float32)
    m[:, C["p_post_0"]] = g.random(W).astype(np.float32) * 1e-3
    m[:, C["elig_lo_4"]] = 1.0
    m[:, C["elig_hi_4"]] = 1.0
    m[:, C["post_ok"]] = 1.0
    return m, g


def steps_of(o):
    """Steps follow SCHED: step s commits SCHED[s % 4] + 1 positions (o = 0 from the prefill step)."""
    out, s, pos = {}, 0, 1
    out[0] = (0, 0, 0)
    while pos < W:
        a = SCHED[s % len(SCHED)]
        n = min(a + 1, W - pos)
        for j in range(n):
            out[pos + j] = (s + 1, j, a)
        pos += n
        s += 1
    return out


STEPMAP = steps_of(0)
NSTEPS = max(v[0] for v in STEPMAP.values()) + 1


def write_session(root, label, scen, rng_noise):
    run = os.path.join(root, "runs", f"session-{label}-attempt-1")
    priv = os.path.join(root, "PRIVATE-runs", f"session-{label}-attempt-1")
    for d in (run, os.path.join(run, "p4log"), os.path.join(priv, "fw"), os.path.join(priv, "p4")):
        os.makedirs(d, exist_ok=True)
    gates_ok = not (scen == "gatefail" and label == "B")
    json.dump(dict(session=label, attempt=1, verdict="COMPLETE" if gates_ok else "INCOMPLETE", gates_ok=gates_ok,
                   batches_done=2), open(os.path.join(run, "outcome.json"), "w"))
    cl = open(os.path.join(run, "client.jsonl"), "w")
    cl.write(json.dumps(dict(kind="start", n_batches=2)) + "\n")
    steprows, steplog = [], []
    step0 = 1000 if label == "B" else (5000 if label == "A2" else 1)     # absolute step numbers differ per session
    st = step0
    for b, ws in enumerate(BATCHES, 1):
        cl.write(json.dumps(dict(kind="batch", batch=b, windows=ws)) + "\n")
        for rep in (1, 2):
            base_step = st
            for s in range(NSTEPS):
                gm = "FULL" if s > 0 else "none"
                if scen == "shapes" and label == "B" and rep == 1 and b == 1 and 10 <= s < 50:
                    gm = "PIECEWISE"
                steplog.append([base_step + s, 0.0 + 0.05 * (base_step + s), 2, 8 if s else 2, 8 if s else 2,
                                6 if s else 0, 0 if s else 2, gm])
                rows = np.full((16, 5), -1, np.int32)
                for i, wid in enumerate(ws):
                    o0 = 0 if s == 0 else min(o for o, v in STEPMAP.items() if v[0] == s)
                    a = 0 if s == 0 else SCHED[(s - 1) % len(SCHED)]
                    rows[i] = [base_step + s, 10 + i, o0, 1 if s == 0 else 4, a]
                steprows.append(rows)
            tags = [f"{10 + i}:{label}-{w}-r{rep}" for i, w in enumerate(ws)]
            np.savez(os.path.join(priv, "p4", f"steprec_{base_step:07d}_{base_step + NSTEPS - 1:07d}_r0.npz"),
                     rows=np.stack(steprows[-NSTEPS:]), cols=np.array(["step", "req_idx", "o0", "n_rows", "accepted"]),
                     tags=np.array(tags))
            for wid in ws:
                m, g = base_metrics(wid)
                for o in range(W):
                    m[o, C["step"]] = base_step + STEPMAP[o][0]
                    m[o, C["row"]] = STEPMAP[o][1]
                    m[o, C["accepted"]] = STEPMAP[o][2]
                    m[o, C["kind"]] = 0.0 if o else 1.0
                    m[o, C["n_rows"]] = 4 if o else 1
                moe = g.standard_normal((3, W, H)).astype(np.float32)
                hid = g.standard_normal((3, W // 8, H)).astype(np.float32)
                lg = g.standard_normal((W // 64, V)).astype(np.float32)
                noisy = scen in ("noisy", "noisynull")
                if noisy and label in ("A2",) or (noisy and rep == 2 and label in ("A", "B")):
                    m[:, :NT] += rng_noise.normal(0, 1e-3, (W, NT)).astype(np.float32)
                if label == "B" and scen in ("propagation", "gatefail", "noisy", "shapes"):
                    m[:, :NT] += np.float32(0.05)
                    m[:, C["entropy"]] += np.float32(0.01)
                    hid += np.float32(0.01)
                    lg += rng_noise.normal(0, 0.02, lg.shape).astype(np.float32) if rep == 1 else 0
                    lg[:, 0] += np.float32(0.1)
                if label == "B" and scen in ("propagation", "gatefail", "noisy", "shapes", "attenuation", "noisynull"):
                    moe[:, 1:] += np.float32(0.01)       # decode rows (o >= 1); o = 0 came from the prefill step
                u16 = lambda a: (torch.from_numpy(a).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16))  # noqa: E731
                tag = f"{label}-{wid}-r{rep}"
                np.savez(os.path.join(priv, "p4", f"p4_{tag}_r0.npz"), moe=u16(moe), moe_have=np.ones((3, W), np.int32),
                         hid=u16(hid), hid_have=np.ones((3, W // 8), np.int32), logits=lg,
                         logits_have=np.ones(W // 64, np.int32))
                rec = dict(tag=tag, cols=COLS, metrics=torch.from_numpy(m), nmeas=torch.ones(W, dtype=torch.int32),
                           acc_hist=torch.zeros(5, dtype=torch.int64), inv=torch.zeros(4, dtype=torch.int64), errors=0)
                for rank in (0, 1):
                    torch.save(rec, os.path.join(priv, "fw", f"req_{tag}_r{rank}.pt"))
            st += NSTEPS + 3
    cl.close()
    with open(os.path.join(run, "p4log", "steps_r0_p1.jsonl"), "w") as f:
        f.write(json.dumps(dict(kind="steps", cols=["step", "t", "num_reqs", "num_tokens", "num_tokens_padded",
                                                    "num_draft_tokens", "n_prefilling", "graph_mode"], rows=steplog)) + "\n")


def main():
    out = sys.argv[1]
    for scen in ("propagation", "attenuation", "null", "gatefail", "noisy", "noisynull", "shapes"):
        root = os.path.join(out, scen)
        os.makedirs(os.path.join(root, "PRIVATE-fixtures"), exist_ok=True)
        json.dump(dict(sched=SCHED), open(os.path.join(root, "PRIVATE-fixtures", "sched.json"), "w"))
        os.makedirs(os.path.join(root, "runs", "g4-attempt-1"), exist_ok=True)
        json.dump(dict(G4="PASS"), open(os.path.join(root, "runs", "g4-attempt-1", "verdict.json"), "w"))
        rng = np.random.default_rng(7)
        for label in ("A", "B", "A2"):
            write_session(root, label, scen, rng)
        print("fabricated", scen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
