#!/usr/bin/env python3
"""D7 step 1: CPU tests of the writer-side correction (d7-fix.diff) -- no GPU, Triton interpreter.

Drives the INSTALLED ``sparse_attn_indexer_kpool`` op (real call site: prefill index-K insert, then the
tail-ring seed) step by step, with the index-K slot mapping, the REAL tail slot mapping builder
(``compute_kpool_tail_slot_mapping``, the D2-corrected one in B0) and positions, and stops the op right
after its writes (the top-k buffer raises on first use). Every pool a request has computed is compared
byte-for-byte with an independent plain-torch pooling of the request's OWN four tokens chosen by logical
position (d7lib.ref_record). Stray writes are checked: every index-K record that no request owns must
keep the sentinel.

D7_EXPECT=b1 (patched image): every check must hold. D7_EXPECT=b0 (unpatched B0): the unaligned checks
are expected to FAIL (they document the defect); aligned checks must hold.
Function-level aligned identity compares the installed writer with B0's ``_kpool_compress_insert``
taken verbatim (AST) from /p/b0_kpool.py.

usage: test_d7_fix.py <out.json>
"""
import ast
import json
import os
import random
import sys
import time
from types import SimpleNamespace as NS

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

torch.Tensor.pin_memory = lambda self, *a, **k: self
import vllm.model_executor.layers.sparse_attn_indexer  # noqa: E402,F401
import vllm.models.glm5next.nvidia.ops.kpool_compress as kc  # noqa: E402
from vllm.model_executor.layers import sparse_attn_indexer_kpool as kp  # noqa: E402
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping  # noqa: E402
from vllm.v1.attention.backends.mla.indexer import (  # noqa: E402
    DeepseekV32IndexerMetadata,
    compute_kpool_tail_slot_mapping,
)

from d7lib import HD, KP, PAGE, SENT, make_caches, rec, ref_record  # noqa: E402

EXPECT = os.environ.get("D7_EXPECT", "b1")
RING, PPB, BLK_TOK = 8, 1088, 4352
gen = torch.Generator().manual_seed(11)
APE = (torch.randn(KP, HD, generator=gen) * 0.1).float()
RESULTS = []


def check(name, ok, detail="", unaligned=False):
    """unaligned=True: the check exercises an unaligned start (expected to fail on b0)."""
    expect = True if (EXPECT == "b1" or not unaligned) else False
    met = bool(ok) == expect
    RESULTS.append({"name": name, "holds": bool(ok), "expected": expect, "met": met, "detail": detail})
    print(f"{'PASS' if met else 'FAIL'} {name}: {'holds' if ok else 'does not hold'} "
          f"(expected {'hold' if expect else 'not hold'}) {detail}")


import inspect  # noqa: E402

_HAS_D7 = "tail_slot_mapping" in inspect.signature(kp._kpool_compress_insert).parameters


def insert(k, g, idx, sm, pos, tail, tsm):
    """Installed writer; ring arguments only where the installed signature has them (B1)."""
    if _HAS_D7:
        kp._kpool_compress_insert(k, g, APE, idx, sm, KP, HD, round_scale=True, positions=pos,
                                  tail_kv_cache=tail, tail_slot_mapping=tsm)
    else:
        kp._kpool_compress_insert(k, g, APE, idx, sm, KP, HD, round_scale=True)


class Req:
    def __init__(self, name, n_tokens, idx_blocks, tail_block):
        self.name, self.n = name, n_tokens
        self.k = torch.randn(n_tokens, HD, generator=gen).to(torch.bfloat16)
        self.g = torch.randn(n_tokens, HD, generator=gen).to(torch.bfloat16)
        self.idx_blocks, self.tail_block = list(idx_blocks), tail_block
        self.done = set()  # computed positions

    def loc(self, pool):
        return self.idx_blocks[pool // PPB] * PPB + pool % PPB

    def slot(self, p):
        return self.loc(p // KP) if (p + 1) % KP == 0 else -1

    def ref(self, pool):
        return ref_record(self.k[pool * KP: pool * KP + KP], self.g[pool * KP: pool * KP + KP], APE)


class Abort(Exception):
    pass


class StopBuf:
    shape = (1, 2048)

    def __setitem__(self, *_):
        raise Abort()


class World:
    def __init__(self, nb=48):
        self.raw, self.tail, self.idx = make_caches(nb, RING)
        self.reqs = []

    def add(self, r):
        self.reqs.append(r)
        return r

    def step(self, segs, n_decode_rows=0, tail=True, positions_ok=True):
        """segs: [(req, start, length)] prefill rows in batch order. n_decode_rows leading 1-token decode
        rows of a throwaway request (excluded from the prefill slice; never checked)."""
        rows = []
        dec = None
        if n_decode_rows:
            dec = Req("dec", 64, [47], 46)
            rows += [(dec, 40 + i, 1) for i in range(n_decode_rows)]
        rows += list(segs)
        pos = torch.cat([torch.arange(s, s + n) for _, s, n in rows]).to(torch.int64)
        k = torch.cat([r.k[s:s + n] for r, s, n in rows])
        g = torch.cat([r.g[s:s + n] for r, s, n in rows])
        sm = torch.tensor([r.slot(int(p)) for (r, s, n) in rows for p in range(s, s + n)], dtype=torch.int64)
        qsl = torch.tensor([0] + list(torch.cumsum(torch.tensor([n for _, _, n in rows]), 0)), dtype=torch.int32)
        bt = torch.zeros(len(rows), 4, dtype=torch.int32)
        bt[:, 0] = torch.tensor([r.tail_block for r, _, _ in rows], dtype=torch.int32)
        tsm = compute_kpool_tail_slot_mapping(torch.full((len(pos),), -1, dtype=torch.int64), bt, qsl, pos,
                                              len(pos), len(rows), RING)
        ndt = n_decode_rows
        md = dict(seq_lens=torch.tensor([s + n for _, s, n in rows], dtype=torch.int32), max_seq_len=0,
                  num_decodes=n_decode_rows, num_decode_tokens=ndt, num_prefills=len(segs),
                  num_prefill_tokens=len(pos) - ndt)
        meta = {"idx": DeepseekV32IndexerMetadata(slot_mapping=sm, **md)}
        if tail:
            meta["tail"] = DeepseekV32IndexerMetadata(slot_mapping=tsm, **md)
        kp.get_forward_context = lambda: NS(attn_metadata=meta)
        try:
            kp.sparse_attn_indexer_kpool(
                torch.zeros(len(pos), 1), "idx", self.idx, torch.zeros(len(pos), 32, HD), None, k, None, 128,
                "ue8m0", 2048, HD, 1 << 20, 1 << 16, StopBuf(), False, False, gate_score=g, compress_ape=APE,
                index_kpool=KP, positions=pos if positions_ok else None,
                tail_kv_cache=self.tail if tail else None, tail_prefix="tail" if tail else None)
        except Abort:
            pass
        for r, s, n in segs:
            r.done.update(range(s, s + n))

    def verify(self, reqs=None):
        """Per request: classes of every fully computed pool; plus stray-write check."""
        out = {}
        owned = set()
        for r in self.reqs:
            if r.name == "dec":
                continue
            for pool in range(r.n // KP):
                if all(p in r.done for p in range(pool * KP, pool * KP + KP)):
                    owned.add(r.loc(pool))
        for r in (reqs or self.reqs):
            bad = []
            for pool in range(r.n // KP):
                if not all(p in r.done for p in range(pool * KP, pool * KP + KP)):
                    continue
                got = rec(self.raw, r.loc(pool))
                if not torch.equal(got, r.ref(pool)):
                    bad.append((pool, "unwritten" if bool((got == SENT).all()) else "wrong"))
            out[r.name] = bad
        # stray writes: index-K records not owned by any computed pool must still be the sentinel,
        # except tail-ring blocks (they alias index pages) and prefilled prefix pools
        tail_blocks = {r.tail_block for r in self.reqs} | {46}
        skip = set(getattr(self, "prefix_locs", set()))
        stray = 0
        nb = self.raw.numel() // PAGE
        for blk in range(nb):
            if blk in tail_blocks:
                continue
            for j in range(PPB):
                loc = blk * PPB + j
                if loc in owned or loc in skip:
                    continue
                if not bool((rec(self.raw, loc) == SENT).all()):
                    stray += 1
        return out, stray


def n_bad(v):
    return sum(len(x) for x in v.values())


# ---------------------------------------------------------------- M1: every chunk-start residue, alone
for r in range(4):
    w = World()
    A = w.add(Req("A", 4200, [1, 2], 30))
    s = 2044 + r
    w.step([(A, 0, s)])
    w.step([(A, s, 4200 - s)])
    v, stray = w.verify()
    check(f"M1 residue {r}: first request alone, start {s}", n_bad(v) == 0 and stray == 0,
          f"bad={v['A']} stray={stray}", unaligned=r != 0)

# ---------------------------------------------------------------- M2: packed batch, later and first
for r in range(4):
    w = World()
    A = w.add(Req("A", 3000, [1], 30))
    B = w.add(Req("B", 700, [2], 31))
    s = 2044 + r
    w.step([(A, 0, s)])
    w.step([(B, 0, 600), (A, s, 3000 - s)])  # A is the later request
    v, stray = w.verify()
    check(f"M2 residue {r}: later request in packed batch (after aligned B)", n_bad(v) == 0 and stray == 0,
          f"bad={v} stray={stray}", unaligned=r != 0)
for r1, r2 in ((1, 2), (3, 1), (2, 3)):
    w = World()
    A = w.add(Req("A", 3000, [1], 30))
    B = w.add(Req("B", 3000, [2], 31))
    w.step([(A, 0, 1000 + r1)])
    w.step([(B, 0, 1200 + r2)])
    w.step([(A, 1000 + r1, 900), (B, 1200 + r2, 700)])  # both unaligned: first and later
    w.step([(B, 1900 + r2, 1100 - r2), (A, 1900 + r1, 1100 - r1)])  # order swapped
    v, stray = w.verify()
    check(f"M2b first+later both unaligned (residues {r1},{r2}), order swapped next step",
          n_bad(v) == 0 and stray == 0, f"bad={v} stray={stray}", unaligned=True)

# ---------------------------------------------------------------- M3: short chunks, successive boundaries
w = World()
C = w.add(Req("C", 2200, [1], 30))
for s, n in ((0, 2045), (2045, 1), (2046, 2), (2048, 3), (2051, 1), (2052, 4), (2056, 3), (2059, 141)):
    w.step([(C, s, n)])
v, stray = w.verify()
check("M3 short chunks 1-4 tokens alone, successive unaligned boundaries", n_bad(v) == 0 and stray == 0,
      f"bad={v['C']} stray={stray}", unaligned=True)
w = World()
C = w.add(Req("C", 2200, [1], 30))
D = w.add(Req("D", 2200, [2], 31))
w.step([(C, 0, 2045)])
sD = 0
for n in (1, 2, 3, 1, 2):
    s = max(C.done) + 1
    w.step([(D, sD, 64), (C, s, n)])
    sD += 64
w.step([(D, sD, 64), (C, max(C.done) + 1, 2200 - max(C.done) - 1)])
v, stray = w.verify()
check("M3b short chunks (1-3 tokens) as later request in packed batches", n_bad(v) == 0 and stray == 0,
      f"bad={v} stray={stray}", unaligned=True)

# ---------------------------------------------------------------- M4: 4,352-token block boundaries
w = World()
E = w.add(Req("E", 9000, [1, 2, 3], 30))
for s, n in ((0, 2045), (2045, 2045), (4090, 262), (4352, 2045), (6397, 2045), (8442, 262), (8704, 296)):
    w.step([(E, s, n)])
v, stray = w.verify()
check("M4 0.9.1/B0 served chunking of a 9,000-token prompt (starts 2045,4090,6397,8442)",
      n_bad(v) == 0 and stray == 0, f"bad={v['E']} stray={stray}", unaligned=True)
for split in (4349, 4350, 4351, 4353, 4354, 4355):
    w = World()
    E = w.add(Req("E", 4600, [1, 2], 30))
    w.step([(E, 0, split)])
    w.step([(E, split, 4600 - split)])
    v, stray = w.verify()
    check(f"M4b chunk start {split} at the block-0/1 boundary (pool {split // 4})", n_bad(v) == 0 and stray == 0,
          f"bad={v['E']} stray={stray}", unaligned=split % 4 != 0)

# ---------------------------------------------------------------- M5: prefix hit (aligned) + continuation
w = World()
X = w.add(Req("X", 300, [5], 31))
F = w.add(Req("F", 7000, [1, 2], 31))  # F reuses X's freed tail block 31 (stale ring contents)
w.step([(X, 0, 300)])  # X seeds ring block 31 with its own tokens; X then finishes, block 31 is reused
w.prefix_locs = set()
for pool in range(BLK_TOK // KP):  # F's cached prefix: 4,352 tokens (block-aligned hit), pools from cache
    loc = F.loc(pool)
    pg, rr = loc // 64, loc % 64
    ref = F.ref(pool)
    w.raw[pg * 8448 + rr * 128: pg * 8448 + rr * 128 + 128] = ref[:128]
    w.raw[pg * 8448 + 8192 + rr * 4: pg * 8448 + 8192 + rr * 4 + 4] = ref[128:]
    w.prefix_locs.add(loc)
F.done.update(range(BLK_TOK))
w.step([(F, 4352, 1127)])  # r1 pattern: cached block + 1,127
w.step([(F, 5479, 7000 - 5479)])
v, stray = w.verify()
check("M5 prefix hit at 4,352 on a reused (stale) ring block, then unaligned continuation 5,479",
      n_bad(v) == 0 and stray == 0, f"bad={v['F']} stray={stray}", unaligned=True)

# ---------------------------------------------------------------- M6: preemption + resume (recompute)
w = World()
G = w.add(Req("G", 3000, [1], 30))
H = w.add(Req("H", 200, [2], 31))
w.step([(G, 0, 2045)])
w.step([(H, 0, 37)])  # H is not G; separate block
# G preempted: its tail block 30 is freed and reused by H2, which seeds it
H2 = w.add(Req("H2", 200, [3], 30))
w.step([(H2, 0, 123)])
# G resumes on a new tail block 32 with no prefix hit (num_computed_tokens = 0) and recomputes
G.tail_block = 32
G.done.clear()
w.step([(G, 0, 2045)])
w.step([(G, 2045, 955)])
v, stray = w.verify()
check("M6 preempted request resumed on a new ring block, recompute from 0, unaligned continuation",
      n_bad(v) == 0 and stray == 0, f"bad={v} stray={stray}", unaligned=True)

# ---------------------------------------------------------------- M7: ring unavailable -> B0 rule
w = World()
A = w.add(Req("A", 2600, [1], 30))
w.step([(A, 0, 2045)], tail=False)
w.step([(A, 2045, 555)], tail=False)
v, _ = w.verify()
check("M7 tail ring absent: straddling pool left unwritten exactly as B0 (documented fallback)",
      v["A"] == [(511, "unwritten")], f"bad={v['A']}")

# ---------------------------------------------------------------- M8: decode rows ahead of the prefill slice
w = World()
A = w.add(Req("A", 2600, [1], 30))
B = w.add(Req("B", 400, [2], 31))
w.step([(A, 0, 2045)])
w.step([(B, 0, 102), (A, 2045, 555)], n_decode_rows=3)
v, stray = w.verify([A, B])
check("M8 three decode rows before the prefill slice; later request unaligned", n_bad(v) == 0 and stray == 0,
      f"bad={v} stray={stray}", unaligned=True)

# ---------------------------------------------------------------- M9: random serving-like schedules
rng = random.Random(5)
tot_bad = tot_pools = n_unaligned = 0
for trial in range(6):
    w = World()
    reqs = [w.add(Req(f"R{i}", rng.randint(300, 6000), [1 + 3 * i, 2 + 3 * i, 3 + 3 * i], 30 + i))
            for i in range(4)]
    comp = {r.name: 0 for r in reqs}
    while any(comp[r.name] < r.n for r in reqs):
        budget = 2045 - 7 * rng.randint(0, 4)
        segs = []
        for r in rng.sample(reqs, len(reqs)):
            left = r.n - comp[r.name]
            if left <= 0 or budget <= 4:
                continue
            n = min(left, budget, rng.choice([budget, rng.randint(1, 900)]))
            segs.append((r, comp[r.name], n))
            n_unaligned += comp[r.name] % 4 != 0
            comp[r.name] += n
            budget -= n + 3
        w.step(segs)
    v, stray = w.verify()
    tot_bad += n_bad(v) + stray
    tot_pools += sum(r.n // KP for r in reqs)
check(f"M9 random shared-step schedules (6 trials, 4 requests, {n_unaligned} unaligned starts)", tot_bad == 0,
      f"bad+stray={tot_bad} of {tot_pools} pools", unaligned=True)

# ---------------------------------------------------------------- M10: aligned identity with B0 (function level)
src = open("/p/b0_kpool.py").read()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_kpool_compress_insert")
ns = {"torch": torch, "kpool_compress_and_write_cache": kc.kpool_compress_and_write_cache}
exec(compile(ast.Module([fn], []), "/p/b0_kpool.py", "exec"), ns)
b0_insert = ns["_kpool_compress_insert"]
same = trials = 0
for trial in range(24):
    nreq = rng.randint(1, 4)
    segs = []
    for i in range(nreq):
        s = 4 * rng.randint(0, 1500)
        n = rng.choice([4 * rng.randint(1, 120), rng.randint(1, 200)])  # aligned starts; any length
        segs.append((i, s, n))
    if trial % 3 == 0:
        segs[0] = (0, segs[0][1], 4 * rng.randint(1, 50))
    reqs = [Req(f"Q{i}", 8000, [1 + 2 * i, 2 + 2 * i], 30 + i) for i in range(nreq)]
    pos = torch.cat([torch.arange(s, s + n) for _, s, n in segs]).to(torch.int64)
    k = torch.cat([reqs[i].k[s:s + n] for i, s, n in segs])
    g = torch.cat([reqs[i].g[s:s + n] for i, s, n in segs])
    sm = torch.tensor([reqs[i].slot(int(p)) for i, s, n in segs for p in range(s, s + n)], dtype=torch.int64)
    qsl = torch.tensor([0] + list(torch.cumsum(torch.tensor([n for _, _, n in segs]), 0)), dtype=torch.int32)
    bt = torch.zeros(nreq, 4, dtype=torch.int32)
    bt[:, 0] = torch.tensor([30 + i for i, _, _ in segs], dtype=torch.int32)
    tsm = compute_kpool_tail_slot_mapping(torch.full((len(pos),), -1, dtype=torch.int64), bt, qsl, pos,
                                          len(pos), nreq, RING)
    raw0, tail0, idx0 = make_caches(48, RING)
    raw1, tail1, idx1 = make_caches(48, RING)
    gt = torch.randn(tail0.shape, generator=gen).to(torch.bfloat16)  # arbitrary (stale) ring contents
    tail0.copy_(gt)
    tail1.copy_(gt)
    b0_insert(k, g, APE, idx0, sm, KP, HD, round_scale=True)
    insert(k, g, idx1, sm, pos, tail1, tsm)
    trials += 1
    same += bool(torch.equal(raw0, raw1))
check(f"M10 aligned starts: patched writer byte-identical to B0 writer ({trials} random packed batches, "
      "stale ring contents)", same == trials, f"{same}/{trials} identical")

# ---------------------------------------------------------------- M11: op-level aligned workload hashes
import hashlib  # noqa: E402

hashes = []
w2 = World()
A2 = w2.add(Req("A", 5000, [1, 2], 30))
B2 = w2.add(Req("B", 3000, [3], 31))
aligned_steps = [[(0, 0, 2044)], [(1, 0, 1000), (0, 2044, 1000)], [(1, 1000, 2000)], [(0, 3044, 1956)]]
for st in aligned_steps:
    w2.step([([A2, B2][i], s, n) for i, s, n in st])
    hashes.append(hashlib.sha256(w2.raw.numpy().tobytes()).hexdigest())
v, stray = w2.verify()
check("M11 op-level aligned workload: all pools correct (hashes compared across images by the runner)",
      n_bad(v) == 0 and stray == 0, f"bad={v} stray={stray}")

# ---------------------------------------------------------------- M12: DCP2 geometry (informative)
nfix = nbad = 0
for start in (2043, 2045, 2046):
    raw_r, tail_r = {}, {}
    A = Req("A", 2600, [1, 2], 30)
    P = Req("P", 200, [4], 31)
    for rank in (0, 1):
        raw, tail, idx = make_caches(48, RING)
        for segs in ([(A, 0, start)], [(P, 100, 12), (A, start, 300)]):
            pos = torch.cat([torch.arange(s, s + n) for _, s, n in segs]).to(torch.int64)
            k = torch.cat([r.k[s:s + n] for r, s, n in segs])
            g = torch.cat([r.g[s:s + n] for r, s, n in segs])
            qsl = torch.tensor([0] + list(torch.cumsum(torch.tensor([n for _, _, n in segs]), 0)), dtype=torch.int32)
            seq = torch.tensor([s + n for _, s, n in segs], dtype=torch.int32)
            ibt = torch.tensor([r.idx_blocks + [0] * (4 - len(r.idx_blocks)) for r, _, _ in segs], dtype=torch.int32)
            sm = get_compressed_slot_mapping(len(pos), qsl, seq, ibt, 896, KP, dcp_world_size=2, dcp_rank=rank,
                                             cp_kv_cache_interleave_size=1).to(torch.int64)
            bt = torch.zeros(len(segs), 4, dtype=torch.int32)
            bt[:, 0] = torch.tensor([r.tail_block for r, _, _ in segs], dtype=torch.int32)
            tsm = compute_kpool_tail_slot_mapping(torch.full((len(pos),), -1, dtype=torch.int64), bt, qsl, pos,
                                                  len(pos), len(segs), RING)
            insert(k, g, idx, sm, pos, tail, tsm)
            kc.kpool_seed_tail_cache(tail, k, g, tsm, KP, HD)
        pool = start // 4
        if pool % 2 == rank:
            lp = pool // 2
            loc = A.idx_blocks[lp // 896] * 896 + lp % 896
            ok = torch.equal(rec(raw, loc), A.ref(pool))
            nfix += ok
            nbad += not ok
check("M12 DCP2 interleave-1 geometry: straddling pool correct on its owner rank (informative; B1 is DCP1)",
      nbad == 0, f"ok={nfix} bad={nbad}", unaligned=True)

summary = {"expect": EXPECT, "installed_writer_has_d7_args": _HAS_D7, "writer_file": kp.__file__, "n": len(RESULTS),
           "met": sum(r["met"] for r in RESULTS), "results": RESULTS, "m11_hashes": hashes,
           "ref_mode": os.environ.get("D7_REF_MODE", "interp")}
print(f"{summary['met']}/{summary['n']} met expectation ({EXPECT})")
json.dump(summary, open(sys.argv[1], "w"), indent=1)
sys.exit(0 if summary["met"] == summary["n"] else 1)
