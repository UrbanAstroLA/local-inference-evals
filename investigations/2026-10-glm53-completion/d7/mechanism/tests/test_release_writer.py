#!/usr/bin/env python3
"""D7 step 0a, items 1 and 3: the installed prefill index-K writer of one release image at unaligned
chunk starts, on CPU under the Triton interpreter (no GPU).

Runs the image's REAL ``_kpool_compress_insert`` + ``kpool_compress_and_write_cache`` and, for DCP2,
the image's REAL ``get_compressed_slot_mapping`` (per rank, interleave 1). Every pool completing in
the step is compared with an independent plain-torch pooling of the request's own four tokens
(d7lib.ref_record). Classes: ok / unwritten / cross (== pooling of the batch rows t-3..t) / other.

usage: test_release_writer.py <label> <out.json>
"""
import json
import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import vllm.model_executor.layers.sparse_attn_indexer  # noqa: E402,F401
import vllm.models.glm5next.nvidia.ops.kpool_compress  # noqa: E402,F401
from vllm.model_executor.layers import sparse_attn_indexer_kpool as kp  # noqa: E402
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping  # noqa: E402

from d7lib import HD, KP, Req, classify, make_caches, rec, ref_record, SENT  # noqa: E402

label, out_path = sys.argv[1], sys.argv[2]
gen = torch.Generator().manual_seed(7)
ape = (torch.randn(KP, HD, generator=gen) * 0.1).float()
results = {"label": label, "writer_file": kp.__file__, "cases": []}
# measured: reference with interpreter casts vs round-to-nearest-even casts, 16 random pools
_d = []
for _ in range(16):
    _k = torch.randn(KP, HD, generator=gen).to(torch.bfloat16)
    _g = torch.randn(KP, HD, generator=gen).to(torch.bfloat16)
    _d.append(int((ref_record(_k, _g, ape, mode="interp")[:128] != ref_record(_k, _g, ape, mode="rne")[:128]).sum()))
results["ref_interp_casts_vs_rne_casts_differing_bytes_per_pool"] = _d


def run_dcp1(start, role, chunk_len=12):
    """One step. role 'first': the unaligned request is the prefill slice's first row.
    role 'later': another request P (aligned, 12 tokens at 100..111) precedes it."""
    raw, _, _ = make_caches(8, 8)
    A = Req("A", start + chunk_len + 8, [1, 2, 3], 6, gen)
    P = Req("P", 200, [4], 7, gen)
    segs = ([] if role == "first" else [(P, list(range(100, 112)))]) + [(A, list(range(start, start + chunk_len)))]
    k = torch.cat([r.k[ps] for r, ps in segs])
    g = torch.cat([r.g[ps] for r, ps in segs])
    slot = torch.tensor([r.slot(p) for r, ps in segs for p in ps], dtype=torch.int64)
    kp._kpool_compress_insert(k, g, ape, raw.view(-1, 64, 132)[: 8 * 17], slot, KP, HD, round_scale=True)
    base = 0 if role == "first" else 12
    out = []
    for i, p in enumerate(range(start, start + chunk_len)):
        if (p + 1) % KP:
            continue
        pool = p // KP
        row = base + i
        cross = None
        if row >= 3:
            cross = (k[row - 3: row + 1], g[row - 3: row + 1])
        out.append({"pool": pool, "positions": [pool * KP, pool * KP + 3], "straddles_start": pool * KP < start,
                    "class": classify(raw, A, pool, ape, cross)})
    # the preceding request's own pools must be unaffected
    p_ok = all(classify(raw, P, q, ape) == "ok" for q in range(25, 28)) if role == "later" else None
    return out, p_ok


for start in (2043, 2044, 2045, 2046, 2047, 4090, 4351, 4353):
    for role in ("first", "later"):
        pools, p_ok = run_dcp1(start, role)
        st = [x for x in pools if x["straddles_start"]]
        results["cases"].append({
            "dcp": 1, "start": start, "residue": start % KP, "role": role,
            "straddling_pool": st[0] if st else None,
            "other_pools_ok": all(x["class"] == "ok" for x in pools if not x["straddles_start"]),
            "n_pools_checked": len(pools), "preceding_request_pools_ok": p_ok})


def run_dcp2(start, role, chunk_len=12, ppb_rank=896):
    """DCP2, interleave 1 (0.7.0 served topology): each rank runs the writer on the full batch (the
    indexer K is replicated); the real compressed slot mapping keeps only the rank's pools."""
    A = Req("A", start + chunk_len + 8, [1, 2, 3], 6, gen)
    P = Req("P", 200, [4], 7, gen)
    segs = ([] if role == "first" else [(P, 100, 12)]) + [(A, start, chunk_len)]
    k = torch.cat([r.k[s0:s0 + n] for r, s0, n in segs])
    g = torch.cat([r.g[s0:s0 + n] for r, s0, n in segs])
    qsl = torch.tensor([0] + [sum(n for _, _, n in segs[: i + 1]) for i in range(len(segs))], dtype=torch.int32)
    seq_lens = torch.tensor([s0 + n for _, s0, n in segs], dtype=torch.int32)
    bt = torch.tensor([r.idx_blocks + [0] * (4 - len(r.idx_blocks)) for r, _, _ in segs], dtype=torch.int32)
    per_rank = {}
    for rank in (0, 1):
        raw, _, _ = make_caches(8, 8)
        sm = get_compressed_slot_mapping(int(qsl[-1]), qsl, seq_lens, bt, ppb_rank, KP, dcp_world_size=2,
                                         dcp_rank=rank, cp_kv_cache_interleave_size=1)
        kp._kpool_compress_insert(k, g, ape, raw.view(-1, 64, 132)[: 8 * 17], sm.to(torch.int64), KP, HD,
                                  round_scale=True)
        base = 0 if role == "first" else 12
        rows = []
        for i, p in enumerate(range(start, start + chunk_len)):
            if (p + 1) % KP:
                continue
            pool, row = p // KP, base + i
            owner = pool % 2
            loc = int(sm[row])
            exp_loc = A.idx_blocks[(pool // 2) // ppb_rank] * ppb_rank + (pool // 2) % ppb_rank if owner == rank else -1
            ref = ref_record(A.k[pool * KP: pool * KP + KP], A.g[pool * KP: pool * KP + KP], ape)
            if loc < 0:
                cls = "not_owned"
            else:
                got = rec(raw, loc)
                if torch.equal(got, ref):
                    cls = "ok"
                elif bool((got == SENT).all()):
                    cls = "unwritten"
                elif row >= 3 and torch.equal(got, ref_record(k[row - 3: row + 1], g[row - 3: row + 1], ape)):
                    cls = "cross"
                else:
                    cls = "other"
            rows.append({"pool": pool, "owner_rank": owner, "slot": loc, "slot_as_expected": loc == exp_loc,
                         "straddles_start": pool * KP < start, "class": cls})
        per_rank[rank] = rows
    return per_rank


for start in (2043, 2045, 2046, 2047, 5627):
    for role in ("first", "later"):
        pr = run_dcp2(start, role)
        results["cases"].append({
            "dcp": 2, "start": start, "residue": start % KP, "role": role,
            "straddling_pool_by_rank": {r: [x for x in rows if x["straddles_start"]] for r, rows in pr.items()},
            "other_owned_pools_ok": all(x["class"] in ("ok", "not_owned") for rows in pr.values() for x in rows
                                        if not x["straddles_start"]),
            "slots_as_expected": all(x["slot_as_expected"] for rows in pr.values() for x in rows)})

# summary: the defect signature per case
sig = []
for c in results["cases"]:
    if c["dcp"] == 1:
        s = c["straddling_pool"]["class"] if c["straddling_pool"] else "none(aligned)"
    else:
        s = {r: [x["class"] for x in v] for r, v in c["straddling_pool_by_rank"].items()}
    sig.append((c["dcp"], c["start"], c["residue"], c["role"], s))
results["signature"] = sig
expect_ok = True
for c in results["cases"]:
    if c["dcp"] == 1:
        if c["residue"] == 0:
            expect_ok &= c["straddling_pool"] is None
        else:
            expect_ok &= c["straddling_pool"]["class"] == ("unwritten" if c["role"] == "first" else "cross")
        expect_ok &= c["other_pools_ok"] and (c["preceding_request_pools_ok"] in (None, True))
    else:
        for r, v in c["straddling_pool_by_rank"].items():
            for x in v:
                want = ("unwritten" if c["role"] == "first" else "cross") if x["owner_rank"] == r else "not_owned"
                expect_ok &= x["class"] == want
        expect_ok &= c["other_owned_pools_ok"] and c["slots_as_expected"]
results["defect_reproduced_as_described"] = bool(expect_ok)
for row in sig:
    print(row)
print("defect_reproduced_as_described:", bool(expect_ok))
json.dump(results, open(out_path, "w"), indent=1, default=str)
