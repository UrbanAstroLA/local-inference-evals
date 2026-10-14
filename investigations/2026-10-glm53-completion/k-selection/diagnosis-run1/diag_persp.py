#!/usr/bin/env python3
"""Diagnosis run 1, item 2 (CPU only; runs inside the pinned 0.7.0 image with --network none, no --gpus).

The run did not save the second perspective, so the observed differences cannot be read back. This
script rebuilds the exact inputs of the two FAIL cells (same seeds, same kinputs/kref code) and models,
per row, the DETERMINISTIC part of the merge's run_row_topk (tiled_topk.py v070): which radix pass
claims which candidates. Output order inside one pass is decided by shared-memory atomics
(`_smem_xadd(ctr)` -> `s_out[pos]`, tiled_topk.py:894-895, :913-914, :946-947, :973-974, :1076-1077,
:1114-1115; tie fill `s_out[topk_static - old_rem]`, :1118-1124) and is NOT modelled; only the size of
the last-claimed group is. Column 511 is always the last slot claimed, so:

  final group size 1                -> column 511 is the same pool on every launch (and every rank)
  final group size > 1, no tie      -> same SET on every launch; column 511 is a race among the group
  round-3 exact tie (n_eq > slots)  -> the SET itself is a race among exactly equal scores

Families: the exact families (plateau, zero, ulp, edge) are bit-exact on the GPU (C2b passed), so their
model is exact. real_L3 / real_L43 use the fp64 reference rounded to fp32 as a proxy for the kernel's
fp32 scores (|err| <= 0.0039 B observed). gauss is skipped: on the GPU its inputs come from a CUDA
generator (kinputs.py:212, :220-226) that the CPU cannot reproduce.

usage (inside the image): python3 diag_persp.py /k /o
"""
from __future__ import annotations

import json
import os
import sys
import time

KROOT = sys.argv[1] if len(sys.argv) > 1 else "/k"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/o"
sys.path.insert(0, os.path.join(KROOT, "kselect"))
import torch  # noqa: E402

import kinputs  # noqa: E402
import kplan  # noqa: E402
import kref  # noqa: E402
from run_k import seed_of  # noqa: E402  (same seed derivation as the run)

TOPK = 512
COARSE_BITS = 10
CELLS = ("v070/tiled48/b16384", "v070/tiled48/b32768")
FAMS = ("real_L3", "real_L43", "plateau", "zero", "ulp", "edge")
EXACT = ("plateau", "zero", "ulp", "edge")


def key32(v32: torch.Tensor) -> torch.Tensor:
    """_convert_to_uint32 (tiled_topk.py:251-260) as int64."""
    b = v32.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    neg = (b & 0x80000000) != 0
    return torch.where(neg, (~b) & 0xFFFFFFFF, b | 0x80000000)


def coarse_bin(v32: torch.Tensor) -> torch.Tensor:
    """_convert_to_coarse_bin (tiled_topk.py:238-248): cvt.rn.f16.f32, sign-flip key, top 10 bits."""
    h = v32.to(torch.float16).view(torch.int16).to(torch.int64) & 0xFFFF
    neg = (h & 0x8000) != 0
    k16 = torch.where(neg, 0xFFFF ^ h, h | 0x8000)
    return (k16 >> (16 - COARSE_BITS)) & ((1 << COARSE_BITS) - 1)


def threshold(bins: torch.Tensor, nbins: int, k: int):
    """Bin t with count(bin >= t) > k >= count(bin >= t+1) (tiled_topk.py:859-870, :1031-1049)."""
    h = torch.bincount(bins, minlength=nbins)
    suf = torch.flip(torch.cumsum(torch.flip(h, [0]), 0), [0]).tolist() + [0]
    for t in range(nbins):
        if suf[t] > k >= suf[t + 1]:
            return t, suf[t + 1]
    raise AssertionError("no threshold bin")


def model_row(v32: torch.Tensor, topk: int = TOPK):
    """Claim structure of run_row_topk over one row of candidates (need_radix: len > topk)."""
    n = v32.numel()
    assert n > topk
    cb = coarse_bin(v32)
    t, above = threshold(cb, 1 << COARSE_BITS, topk)
    rem = topk - above
    if rem == 0:                                            # tiled_topk.py:872 coarse quick exit
        return dict(path="coarse-quick", final_group=topk, tie_n=0, tie_slots=0)
    cand = key32(v32[cb == t])
    if cand.numel() > 8192:                                  # overflow fallback (:1148-1175); impossible at 1024
        return dict(path="overflow-fallback", final_group=None, tie_n=None, tie_slots=None)
    for r in range(4):
        byte = (cand >> (24 - 8 * r)) & 0xFF
        st, above = threshold(byte, 256, rem)
        rem2 = rem - above
        if rem2 == 0:                                        # quick exit in round r (:1051-1081)
            return dict(path=f"quick-r{r}", final_group=above, tie_n=0, tie_slots=0)
        if r == 3:                                           # exact tie: fill from the end (:1118-1124)
            n_eq = int((byte == st).sum())
            return dict(path="tie-r3", final_group=n_eq, tie_n=n_eq, tie_slots=rem2)
        cand = cand[byte == st]
        rem = rem2
    raise AssertionError


def main():
    t0 = time.time()
    dev = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    plan = json.load(open(os.path.join(KROOT, "plan.json")))
    eng = plan["meta"]["engines"]["v070"]
    K = eng["K"]
    banks = {f: kinputs.Bank(os.path.join(KROOT, "bank", f"bank_L{f.split('_L')[1]}.pt"), dev)
             for f in FAMS if f.startswith("real_")}
    rows_out = []
    for cid in CELLS:
        cell = next(c for c in plan["cells"] if c["id"] == cid)
        b, route, rows = cell["boundary_local"], cell["route"], cell["rows"]
        ctas = cell["expect"]["ctas_per_group"]
        for fam in FAMS:
            exact = fam in EXACT
            for bi, batch in enumerate(cell["batches"]):
                seed = seed_of(cell["id"], fam, bi)
                reqs = kinputs.build_requests(fam, batch, K, b, route, rows, ctas, eng["dcp"], seed, dev, banks)
                Ls = kplan.row_lengths(eng, batch, cell["pad_rows"])
                for i, r in enumerate(reqs):
                    s64, _ = kref.ref_scores(r["q8"], r["w"], r["K8"], r["S"])
                    for j in range(K + 1):
                        ri = i * (K + 1) + j
                        P = Ls[ri] // kplan.KPOOL
                        s = s64[j, :P]
                        per_rank = []
                        for rk in range(2):
                            sl = s[rk::2]
                            assert sl.numel() >= TOPK
                            per_rank.append(torch.topk(sl, TOPK, largest=True, sorted=False).values)
                        cand64 = torch.cat(per_rank)
                        v32 = cand64.to(torch.float32)
                        if exact:
                            assert torch.equal(v32.to(torch.float64), cand64), "exact family not fp32-exact"
                        m = model_row(v32)
                        srt = torch.sort(v32, descending=True).values
                        m.update(cell=cid, fam=fam, batch=bi, row=ri, L=Ls[ri], P=P,
                                 tie_512_513=bool(srt[TOPK - 1] == srt[TOPK]),
                                 # global descriptive rank of the bottom of the final group
                                 final_group_gt32=bool(m["final_group"] is not None and m["final_group"] > 32))
                        rows_out.append(m)
        print(f"[{time.time() - t0:6.1f}s] {cid} done", flush=True)
    json.dump(rows_out, open(os.path.join(OUT, "diag_persp_rows.json"), "w"))

    # ---- summary
    def tally(sel):
        d = {}
        for m in sel:
            d[m["path"]] = d.get(m["path"], 0) + 1
        return dict(sorted(d.items()))
    lines = []
    for cid in CELLS:
        lines.append(f"== {cid}  (rows per family = 2 batches x 48 = 96; each row ran 3 repeats)")
        lines.append(f"   {'family':9s} {'paths':58s} {'col511 race (grp>1)':>20s} {'set race (tie)':>15s} "
                     f"{'grp>32':>7s} {'median grp':>10s}")
        for fam in FAMS:
            sel = [m for m in rows_out if m["cell"] == cid and m["fam"] == fam]
            grp = sorted(m["final_group"] for m in sel)
            race = sum(1 for m in sel if m["final_group"] > 1)
            tie = sum(1 for m in sel if m["path"] == "tie-r3")
            gt32 = sum(1 for m in sel if m["final_group_gt32"])
            lines.append(f"   {fam:9s} {json.dumps(tally(sel)):58s} {race:>12d}/{len(sel):<7d} {tie:>8d}/{len(sel):<6d} "
                         f"{gt32:>7d} {grp[len(grp) // 2]:>10d}")
        sel = [m for m in rows_out if m["cell"] == cid]
        lines.append(f"   all six modelled families: rows {len(sel)}, col511-race rows "
                     f"{sum(1 for m in sel if m['final_group'] > 1)}, set-race (exact tie) rows "
                     f"{sum(1 for m in sel if m['path'] == 'tie-r3')}, final group > 32 rows "
                     f"{sum(1 for m in sel if m['final_group_gt32'])} (x3 repeats = "
                     f"{3 * sum(1 for m in sel if m['final_group_gt32'])} row-repeats)")
        ties = [m for m in sel if m["path"] == "tie-r3"]
        if ties:
            lines.append("   exact-tie rows: tie group n_eq / slots: " + ", ".join(
                sorted({f"{m['fam']}:{m['tie_n']}/{m['tie_slots']}" for m in ties})))
    lines.append(f"done in {time.time() - t0:.1f}s")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "diag_persp.out"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
