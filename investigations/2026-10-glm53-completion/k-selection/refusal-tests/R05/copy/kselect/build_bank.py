#!/usr/bin/env python3
"""K: build the realistic input bank from the stage-2 real decode captures (CPU only).

Source: phase2-stage2/probe-a1 and probe-a2 (c511_*.pt, written by the col511 probe hook on stock
0.9.1, real GPQA-style decode steps, 4k-120k tokens). Each record holds the real fp8 query rows,
fp32 head weights, pool-granular seq_lens, the page table and the raw index-K pages it read.

Output (per DSA layer in LAYERS): bank_L<layer>.pt with
  pools_k  (N,128) uint8   e4m3 bytes of every distinct live pool, request runs kept in order
  pools_s  (N,)    float32 per-pool scale
  runs     list of (start, length, source_file, request_first_page) - one per distinct request
  q        (M,32,128) uint8 e4m3 query rows (distinct), w (M,32) float32 weights, q_seq (M,) pool count
plus bank_manifest.json (counts, sanitisation, sha256 of every output).

Sanitisation (the fixture uses FINITE inputs only; amendment 21 section 3):
  e4m3fn NaN bytes (0x7F, 0xFF) -> 0x00 (counted); pools with a non-finite or non-positive scale and
  query rows with a non-finite weight are dropped (counted). Nothing else is altered.

Origin caveat: the bytes come from stock 0.9.1 decode, so some pools were built by the inv-D2 path.
For a selection-kernel test only the score distribution shape matters (gaps, ReLU zeros, exact ties),
not whether a pool summarises the right tokens.

usage: build_bank.py <probe_dir> [<probe_dir> ...] <out_dir>
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys

import torch

LAYERS = (3, 23, 43)
PAGE, HD, HEADS = 64, 128, 32
PB = PAGE * (HD + 4)


def layer_of(name):
    m = re.search(r"layers\.(\d+)\.", str(name))
    return int(m.group(1)) if m else -1


def main():
    dirs, out = sys.argv[1:-1], sys.argv[-1]
    os.makedirs(out, exist_ok=True)
    files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "c511_*.pt")))
    # per layer: request key -> (n_pools, k bytes, scales, file); keep the longest per request
    reqs = {L: {} for L in LAYERS}
    qrows = {L: {} for L in LAYERS}
    stats = {L: dict(files=0, nan_k_bytes=0, nan_q_bytes=0, bad_scale_pools=0, bad_w_rows=0) for L in LAYERS}
    for f in files:
        r = torch.load(f, weights_only=False, map_location="cpu")
        L = layer_of(r["layer"])
        if L not in reqs:
            continue
        stats[L]["files"] += 1
        R = int(r["rows"])
        sl = r["seq_lens"].to(torch.int64)
        bt = r["bt"].to(torch.int64)
        pid = r["page_ids"].to(torch.int64)
        pages = r["pages"]
        k8 = pages[:, : PAGE * HD].reshape(-1, PAGE, HD)
        ks = pages[:, PAGE * HD:].contiguous().view(torch.float32).reshape(-1, PAGE)
        loc = torch.searchsorted(pid, bt.clamp(min=0))
        for i in range(R):
            n = int(sl[i])
            if n <= 0:
                continue
            np_ = (n + PAGE - 1) // PAGE
            pg = loc[i, :np_]
            if not bool((pid[pg.clamp(max=pid.numel() - 1)] == bt[i, :np_]).all()):
                continue  # page missing from the capture: skip this row
            key = int(bt[i, 0])  # first physical page identifies the request within a server run
            key = (os.path.basename(os.path.dirname(f)), int(r["rank"]), key)
            old = reqs[L].get(key)
            if old is None or old[0] < n:
                K = k8[pg].reshape(-1, HD)[:n].clone()
                S = ks[pg].reshape(-1)[:n].clone()
                reqs[L][key] = (n, K, S, os.path.basename(f))
            q = r["q"].reshape(R, HEADS, HD)[i].clone()
            w = r["w"].float().reshape(R, HEADS)[i].clone()
            h = hashlib.sha256(q.numpy().tobytes() + w.numpy().tobytes()).hexdigest()
            if h not in qrows[L]:
                qrows[L][h] = (q, w, n, os.path.basename(f))
    manifest = dict(source_dirs=dirs, n_files=len(files), layers={})
    for L in LAYERS:
        st = stats[L]
        # distinct requests (dedupe identical content across TP ranks / probe dirs)
        seen, runs, ks_, ss_ = set(), [], [], []
        start = 0
        for key in sorted(reqs[L], key=lambda k: -reqs[L][k][0]):
            n, K, S, src = reqs[L][key]
            h = hashlib.sha256(K[: min(n, 256)].numpy().tobytes()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            nan = (K == 0x7F) | (K == 0xFF)
            st["nan_k_bytes"] += int(nan.sum())
            K = torch.where(nan, torch.zeros_like(K), K)
            good = torch.isfinite(S) & (S > 0)
            st["bad_scale_pools"] += int((~good).sum())
            K, S = K[good], S[good]
            runs.append([start, int(K.shape[0]), src, list(key)])
            ks_.append(K)
            ss_.append(S)
            start += int(K.shape[0])
        qs, ws, qn, qsrc = [], [], [], []
        for h, (q, w, n, src) in sorted(qrows[L].items(), key=lambda kv: -kv[1][2]):
            if not bool(torch.isfinite(w).all()):
                st["bad_w_rows"] += 1
                continue
            nan = (q == 0x7F) | (q == 0xFF)
            st["nan_q_bytes"] += int(nan.sum())
            qs.append(torch.where(nan, torch.zeros_like(q), q))
            ws.append(w)
            qn.append(n)
            qsrc.append(src)
        bank = dict(layer=L, pools_k=torch.cat(ks_) if ks_ else torch.empty(0, HD, dtype=torch.uint8),
                    pools_s=torch.cat(ss_) if ss_ else torch.empty(0), runs=runs,
                    q=torch.stack(qs), w=torch.stack(ws), q_seq=torch.tensor(qn), q_src=qsrc)
        p = os.path.join(out, f"bank_L{L}.pt")
        torch.save(bank, p)
        sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
        s = bank["pools_s"]
        manifest["layers"][str(L)] = dict(
            file=os.path.basename(p), sha256=sha, n_pools=int(bank["pools_k"].shape[0]), n_requests=len(runs),
            longest_run=max((r[1] for r in runs), default=0), n_query_rows=int(bank["q"].shape[0]),
            query_seq_max=int(max(qn)) if qn else 0, scale_min=float(s.min()) if s.numel() else None,
            scale_max=float(s.max()) if s.numel() else None, **st)
        print(json.dumps({L: manifest["layers"][str(L)]}), flush=True)
    json.dump(manifest, open(os.path.join(out, "bank_manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
