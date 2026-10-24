#!/usr/bin/env python3
"""SYNTHETIC capture-p0 session records for the analysis tests (numpy; run in a CPU container).  Every byte here is
fabricated (seeded random bytes in the served layouts, fabricated digests); nothing is model data.  Output goes under
the tests/out directory given, in <case>/ao (public-format records) and <case>/synthetic-po (the private-format files)
-- never under a PRIVATE path.

build(root, spec) -> (ao, po).  spec keys (all optional):
  reps       [("r1", True), ("r2", True), ("r3", False), ("r4", False)]   (name, localisation on)
  variant    {rep: int}   0 = the reference content; a different int = a different differing variant
  loc_diff   {rep: (point, first_pos)}   digests from that boundary on (computation order), positions >= first_pos
  core_diff  {rep: [(rank, slice, first_pos)]}   bytes of that slice differ (MLA: records >= first_pos)
  same_ids   [rep, ...]   these replicates get replicate r1's physical ids
  break      {rep: [what, ...]}   planted defects: "cap_rank1", "view", "ncached", "chunks", "prompt", "cached_tokens",
             "timing", "npz_missing", "loc_missing", "identity", "lifecycle", "twosum", "prov"
  pools      {rep: value}  write this byte into the layer-7 indexer pool 511 / 1022 records of block 0
"""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("LY", os.path.join(HERE, "..", "build", "files", "vllm", "cap0_hooks_layout.py"))
LY = importlib.util.module_from_spec(_s)
_s.loader.exec_module(LY)
CHUNKS = LY.warm_chunks(LY.WARM_TOKENS)
ORDER = ["L2.state_end"] + [p for p, _t, _d in LY.LOC_ORDER] + ["L4.state_end"]
TENS = {p: t for p, t, _d in LY.LOC_ORDER}
GROUP = {"mla3": 0, "mla7": 0, "idx3": 0, "idx7": 0, "kda0": 2, "kda2": 4, "kda4": 5}
SIZES = {"mla": LY.P, "idx": LY.IDX_PAGE, "conv_live": LY.CONV_LIVE_BYTES, "conv_inactive": LY.CONV_LIVE_BYTES,
         "rec": LY.REC_BYTES}
NB = 3


def seed_of(*a):
    return int.from_bytes(hashlib.sha256(repr(a).encode()).digest()[:4], "little")


def sha(b):
    return hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest()


def slice_kind(name):
    if name.startswith("mla"):
        return "mla"
    if name.startswith("idx"):
        return "idx"
    if "conv_live" in name:
        return "conv_live"
    if "conv_inactive" in name:
        return "conv_inactive"
    return "rec"


def slice_names(rank):
    out = ["kda0_conv_live", "kda0_conv_inactive", "kda2_rec", "kda4_rec", "kda4_conv_live", "kda4_conv_inactive"]
    if rank == 0:
        out = [f"mla{L}_b{j}" for L in (3, 7) for j in range(NB)] + [f"idx{L}_b{j}" for L in (3, 7) for j in range(NB)] + out
    return out


def ids_for(k):
    base = 100 * k
    return dict(g0=[base + 1, base + 2, base + 3, base + 4], kda0=base + 10, kda2=base + 11, kda4=base + 12, dst=base + 20)


def slice_bytes(name, rank, variant, diff_pos):
    kind = slice_kind(name)
    a = np.random.default_rng(seed_of("slice", name, rank)).integers(0, 256, SIZES[kind], dtype=np.uint8)
    if kind == "idx":                                    # never-written pool records are zero by default
        recs_zero = [LY.idx_pool(q) for q in LY.never_written_pools(CHUNKS, int(name.split("_b")[1]))]
        for (v0, v1), (s0, s1) in recs_zero:
            a[v0:v1] = 0
            a[s0:s1] = 0
    if variant and diff_pos is not None:
        if kind == "mla":
            j = int(name.split("_b")[1])
            r = a.reshape(LY.BLOCK_TOKENS, LY.MLA_REC)
            lo = max(0, diff_pos - j * LY.BLOCK_TOKENS)
            if lo < LY.BLOCK_TOKENS:
                r[lo:, 5] ^= np.uint8(variant)
        else:
            a[7] ^= np.uint8(variant)
    return a


def digests(point, tensor, rank, variant, diff):
    """[13056, 32] fabricated row digests; rows from the perturbed boundary on differ by variant."""
    src = (point, tensor)
    for (a, ta), (b, tb) in LY.LOC_IDENTITIES:      # identities: the derived boundary carries its source's digests
        if (b, tb) == src or (a, ta) == src:
            src = min((a, ta), (b, tb), key=lambda x: ORDER.index(x[0]))
    d = np.random.default_rng(seed_of("dig", src, rank)).integers(0, 256, (LY.N_CACHED, 32), dtype=np.uint8)
    if variant and diff is not None:
        dp, pos = diff
        if ORDER.index(src[0]) >= ORDER.index(dp):       # an identity follows its source boundary
            d[pos:, 0] ^= np.uint8(variant)
    return d


def build(root, spec):
    reps = spec.get("reps", [("r1", True), ("r2", True), ("r3", False), ("r4", False)])
    variant = spec.get("variant", {})
    brk = spec.get("break", {})
    ao, po = os.path.join(root, "ao"), os.path.join(root, "synthetic-po")
    os.makedirs(os.path.join(ao, "p4log"), exist_ok=True)
    os.makedirs(po, exist_ok=True)
    client = [dict(kind="start", SYNTHETIC=True, replicates=[dict(name=n, loc=l, salt=f"cap0-{n}") for n, l in reps])]
    p4, sched, cap0 = [], [], {0: [], 1: []}
    for rank in (0, 1):
        tk = ["mla3", "mla7", "idx3", "idx7", "kda0", "kda2", "kda4"] if rank == 0 else ["kda0", "kda2", "kda4"]
        cap0[rank].append(dict(kind="cap0_init", rank=rank, SYNTHETIC=True, mla_owners="split:25", loc=True,
                               targets={k: dict(kind=k[:3], layer=int(k[3:]), group=GROUP[k], ok=True) for k in tk}))
    step, uid = 0, 0
    for k, (rep, loc) in enumerate(reps, 1):
        b = brk.get(rep, [])
        v = variant.get(rep, 0)
        ids = ids_for(1 if rep in spec.get("same_ids", []) else k)
        wtag, ctag = f"CW-{rep}-{'L' if loc else 'N'}", f"CAP-{rep}"
        client.append(dict(kind="warm", replicate=rep, tag=wtag, salt=f"cap0-{rep}", ok=True, prompt_tokens=LY.WARM_TOKENS,
                           cached_tokens=0, prompt_sha256="P" * 8 + ("x" if "prompt" in b else ""), SYNTHETIC=True))
        client.append(dict(kind="cap", replicate=rep, tag=ctag, salt=f"cap0-{rep}", ok=True, prompt_tokens=LY.P_EARLY,
                           cached_tokens=8704 if "cached_tokens" in b else LY.N_CACHED, prompt_sha256="C" * 8, SYNTHETIC=True))
        client.append(dict(kind="replicate_done", replicate=rep))
        uid += 1
        chunks = [(0, 2048)] + CHUNKS[1:] if "chunks" in b else CHUNKS
        for s, n in chunks:
            step += 1
            sched.append(dict(step=step, mode="none", nr=1, rows=[["q", 3, uid, wtag, n, s, s, LY.WARM_TOKENS, 1, 0]]))
        g0 = ids["g0"]
        groups = [g0, [], [0, 0, ids["kda0"], 999], [], [0, 0, ids["kda2"], 998], [0, 0, ids["kda4"], ids["dst"]]]
        p4.append(dict(kind="blocks", tag=wtag, uid=uid, rank=0, groups=groups if "prov" not in b else
                       [[7, 7, 7, 7]] + groups[1:]))
        uid += 1
        step += 1
        ncomp = 8704 if "ncached" in b else LY.N_CACHED
        sched.append(dict(step=step, mode="none", nr=1, rows=[["q", 3, uid, ctag, 2045, ncomp, ncomp, LY.P_EARLY, 1, 0]]))
        cdiff = {(r, s): p for r, s, p in spec.get("core_diff", {}).get(rep, [])}
        for rank in (0, 1):
            if rank == 1 and "cap_rank1" in b:
                continue
            arrays, sl = {}, {}
            for name in slice_names(rank):
                key = name.split("_")[0]
                pos = cdiff.get((rank, name))
                a = slice_bytes(name, rank, v if pos is not None else 0, pos)
                if name == "idx7_b0" and rep in spec.get("pools", {}):
                    for q in (511, 1022):
                        (v0, v1), (s0, s1) = LY.idx_pool(q)
                        a[v0:v1] = spec["pools"][rep]
                pb = g0[int(name.split("_b")[1])] if key.startswith(("mla", "idx")) else ids[key]
                if "zero" in b and name == "mla7_b0":
                    a = np.zeros_like(a)
                arrays[name] = a
                d = dict(b=pb, sha256=sha(a), view_equal="view" not in b, layout_ok=True,
                         twosum_device=[1, 2], twosum_host=[1, 2] if "twosum" not in b else [1, 3])
                if key.startswith(("mla", "idx")):
                    d["kernel_pages_contiguous"] = True
                sl[name] = d
            sl["kda4_rec_dst"] = dict(b=ids["dst"], sha256=sl["kda4_rec"]["sha256"] if "timing" in b else "f" * 64,
                                      view_equal=True, layout_ok=True)
            arrays["meta"] = np.array(json.dumps(dict(SYNTHETIC=True, tag=ctag, rank=1 - rank if "meta_rank" in b else rank)))
            path = os.path.join(po, f"cap0_{ctag}_r{rank}.npz")
            np.savez(path, **arrays)
            fsha = hashlib.sha256(open(path, "rb").read()).hexdigest()
            if "npz_missing" in b and rank == 0:
                os.remove(path)
            cap0[rank].append(dict(kind="cap0_capture", tag=ctag, uid=uid, step=step, rank=rank, ok=True, n_computed=ncomp,
                                   n_sched=2045, slices=sl, file_sha256=fsha, cached_ids_sha256="I" * 8,
                                   groups=[dict(nbk=len(g) * (68 if gi == 0 else 1), bpk=68 if gi == 0 else 1,
                                                logical=list(g)) for gi, g in enumerate(groups)], SYNTHETIC=True))
            if loc and "loc_missing" not in b:
                ld = spec.get("loc_diff", {}).get(rep)
                arrays = {}
                for p, ts in TENS.items():
                    if p == "L3.topk" and rank != 0:
                        continue
                    for t in ts:
                        dd = digests(p, t, rank, v, ld)
                        if "identity" in b and (p, t) == ("L3.attn_in", "x"):
                            dd = dd.copy()
                            dd[3, 3] ^= 1
                        arrays[f"dig:{p}.{t}"] = dd
                        arrays[f"pos:{p}.{t}"] = np.arange(LY.N_CACHED, dtype=np.int32)
                states, steps = [], []
                for s, n in [c for c in CHUNKS if c[0] < LY.N_CACHED]:
                    steps.append(dict(pos0=s, n=n, mode="none", ids_sha256="S%d" % s + ("x" if "loc_ids" in b else "")))
                    for L in (2, 4):
                        e = s + n
                        tag_ = "base"
                        if v and ld is not None and ORDER.index(f"L{L}.state_end") >= ORDER.index(ld[0]) and e > ld[1]:
                            tag_ = "v%d" % v
                        rs = hashlib.sha256(f"{rank}{L}{e}{tag_}".encode()).hexdigest()
                        if L == 4 and e == LY.N_CACHED:
                            rs = sha(slice_bytes("kda4_rec", rank, v if cdiff.get((rank, "kda4_rec")) is not None else 0,
                                                 cdiff.get((rank, "kda4_rec"))))
                            if "lifecycle" in b or "loc_state_equal" in b:
                                rs = "0" * 64
                        cs = rs[:16]
                        if L == 4 and e == LY.N_CACHED:
                            cs = sha(slice_bytes("kda4_conv_live", rank, v if cdiff.get((rank, "kda4_conv_live")) is not None
                                                 else 0, cdiff.get((rank, "kda4_conv_live"))))
                        states.append(dict(point=f"L{L}.state_end", pos_end=e, col=ids["kda4"] if L == 4 else ids["kda2"],
                                           rec_sha256=rs, conv_live_sha256=cs))
                meta = dict(SYNTHETIC=True, tag=wtag, uid=uid - 1, rank=rank, is_ref=k == 1, steps=steps, states=states,
                            points={}, errors=[], loc_tokens=LY.N_CACHED)
                arrays["meta"] = np.array(json.dumps(meta))
                path = os.path.join(po, f"loc_{wtag}_r{rank}.npz")
                np.savez(path, **arrays)
                cap0[rank].append(dict(kind="cap0_loc_flushed", tag=wtag, rank=rank, file_sha256=hashlib.sha256(
                    open(path, "rb").read()).hexdigest(), SYNTHETIC=True))
    client.append(dict(kind="end", replicates_done=[r for r, _l in reps]))
    with open(os.path.join(ao, "client.jsonl"), "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in client)
    with open(os.path.join(ao, "p4log", "p4_r0_p1.jsonl"), "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in p4)
    with open(os.path.join(ao, "p4log", "sched_r0_p1.jsonl"), "w") as f:
        f.write(json.dumps(dict(kind="sched", steps=sched)) + "\n")
    for rank in (0, 1):
        with open(os.path.join(ao, "p4log", f"cap0_r{rank}_p1.jsonl"), "w") as f:
            f.writelines(json.dumps(d) + "\n" for d in cap0[rank])
    return ao, po


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "SYNTHETIC-README.txt"), "w").write(
        "Synthetic capture-p0 fixtures for the analysis tests: fabricated bytes and digests, no model data.\n")
    print("synthetic fixture root", out)
