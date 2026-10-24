#!/usr/bin/env python3
"""Addressing tests for cap0_hooks_layout (host python3, stdlib only; reads the public Stage-0 records read-only).

  A1  constants: P, the indexer page, the KDA page use (audit-s0/AUDIT.md tasks 3-4)
  A2  AUDIT task 4 byte ranges for physical ids 436 / 762 (layer-7 indexer block, pools 511 / 1,022) and the
      layer-7 MLA block / 528-B records
  A3  KDA conv rows 0-2 / 3-5 and the recurrent range ([b*P + 147,456, +2,097,152)), AUDIT task 3
  A4  warm chunking of the 13,057-token warm == the Stage-0 recorded warm rows of W-V1-r1-R01 (first 9 steps:
      2,045 / 2,045 / 262 per block) and never-written pools == (511, 1,022) in blocks 0, 1, 2
  A5  KDA columns: cached-state column 2, pre-copy destination 3 (AUDIT task 3: V1-r1 group 2 440 -> 715)
  A6  physical(): the Stage-0 warm's recorded logical ids expanded to kernel entries (bpk 68) map back; the
      R01-early capture plan of AUDIT task 4 (ids 436 / 762) gives the AUDIT byte ranges
  A7  E4M3 decode table spot values
"""
import glob
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
TILE = os.path.dirname(PKG)
spec = importlib.util.spec_from_file_location("LY", os.path.join(PKG, "build", "files", "vllm", "cap0_hooks_layout.py"))
LY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LY)
N = {"PASS": 0, "FAIL": 0}


def check(name, ok, detail=""):
    N["PASS" if ok else "FAIL"] += 1
    print(("PASS " if ok else "FAIL ") + name + ("  " + str(detail) if detail and not ok else ""))


check("A1 P = 68 x 64 x 528 = 2,297,856", LY.P == 2297856 == 68 * 64 * 528)
check("A1 indexer page 17 x 8,448 = 143,616", LY.IDX_PAGE == 143616 == 17 * 8448 and LY.IDX_POOL_PAGE == 8448)
check("A1 KDA page use 2,244,608 of P", LY.KDA_USED == 2244608 and LY.KDA_USED < LY.P)
check("A1 1,088 pools per block = 17 x 64", LY.IDX_POOLS == 1088 == LY.IDX_PPB * LY.IDX_ROWS)

check("A2 indexer block 436 = [62,616,576, 62,760,192)", LY.idx_block(436) == (62616576, 62760192), LY.idx_block(436))
check("A2 indexer block 762 = [109,435,392, 109,579,008)", LY.idx_block(762) == (109435392, 109579008), LY.idx_block(762))
check("A2 pool 511: values [67,200, 67,328), scale [67,580, 67,584)", LY.idx_pool(511) == ((67200, 67328), (67580, 67584)),
      LY.idx_pool(511))
check("A2 pool 1,022: values [134,656, 134,784), scale [135,160, 135,164)",
      LY.idx_pool(1022) == ((134656, 134784), (135160, 135164)), LY.idx_pool(1022))
check("A2 pool 0 and pool 1,087 bound the block", LY.idx_pool(0)[0][0] == 0 and LY.idx_pool(1087)[1][1] == LY.IDX_PAGE)
check("A2 two never-written pools = 264 B", sum(v1 - v0 + s1 - s0 for (v0, v1), (s0, s1) in map(LY.idx_pool, (511, 1022))) == 264)
check("A2 MLA block 436 = [436 P, 437 P)", LY.mla_block(436) == (436 * 2297856, 437 * 2297856))
check("A2 MLA record of in-block token p at b*P + 528 p", all(LY.mla_record(762, p) == (762 * LY.P + 528 * p, 762 * LY.P + 528 * (p + 1))
                                                            for p in (0, 63, 64, 2044, 2045, 4351)))

b = 440
check("A3 conv rows 0-2 [b P, +73,728) / rows 3-5 [+73,728, +147,456)",
      LY.kda_conv(b) == ((b * LY.P, b * LY.P + 73728), (b * LY.P + 73728, b * LY.P + 147456)))
check("A3 recurrent [b P + 147,456, +2,097,152)", LY.kda_rec(b) == (b * LY.P + 147456, b * LY.P + 147456 + 2097152))

ch = LY.warm_chunks(LY.WARM_TOKENS)
exp = [(0, 2045), (2045, 2045), (4090, 262), (4352, 2045), (6397, 2045), (8442, 262), (8704, 2045), (10749, 2045),
       (12794, 262), (13056, 2045), (15101, 2045), (17146, 262), (17408, 1)]
check("A4 warm chunks of 17,409 tokens (amendment A1: the warm is the full capture prompt)", ch == exp, ch)
sched = sorted(glob.glob(os.path.join(TILE, "runs-stage0", "stage0-attempt-1", "p4log", "sched_r0_p*.jsonl")))
rows = []
for p in sched:
    for line in open(p):
        r = json.loads(line)
        for s in r.get("steps", []):
            for row in s.get("rows", []):
                if row[3] == "W-V1-r1-R01":
                    rows.append((row[5], row[4]))
check("A4 == Stage-0 recorded warm rows of W-V1-r1-R01 (first 12 steps: 4 blocks of 2,045 / 2,045 / 262)",
      bool(rows) and rows[:12] == exp[:12], rows[:13])
for j in range(4):
    check(f"A4 never-written pools of block {j} == (511, 1022)", LY.never_written_pools(ch, j) == [511, 1022],
          LY.never_written_pools(ch, j))
check("A4 an aligned chunking writes every pool", LY.never_written_pools([(0, 4352)], 0) == [])

check("A5 cached-state column of n_cached 13,056 = 2", LY.cached_state_col(13056) == 2)
check("A5 pre-copy destination (13,056 + 2,045) = column 3", LY.precopy_dst_col(13056, 2045) == 3)
check("A5 n_cached for P_early 17,409 = 13,056", ((LY.P_EARLY - 1) // 4352 - 1) * 4352 == LY.N_CACHED)
check("A8 cache rule: the 17,409-token warm gives the capture 13,056 cached tokens", LY.cache_hit(LY.WARM_TOKENS, LY.P_EARLY) == LY.N_CACHED)
check("A8 cache rule reproduces attempt 1 (13,057-token warm -> 8,704) and Stage 0 (113,940 -> 13,056)",
      LY.cache_hit(13057, 17409) == 8704 and LY.cache_hit(113940, 17409) == 13056)
a1 = sorted(glob.glob(os.path.join(PKG, "runs", "session-attempt-1", "client.jsonl")))
if a1:
    caps = [json.loads(l) for l in open(a1[0]) if '"kind": "cap"' in l]
    warms = [json.loads(l) for l in open(a1[0]) if '"kind": "warm"' in l]
    check("A8 the rule matches attempt 1's measured cached_tokens on all 6 replicates",
          len(caps) == 6 and all(c["cached_tokens"] == LY.cache_hit(w["prompt_tokens"], c["prompt_tokens"]) for w, c in zip(warms, caps)))

blocks = {}
for p in sorted(glob.glob(os.path.join(TILE, "runs-stage0", "stage0-attempt-1", "p4log", "p4_r0_p*.jsonl"))):
    for line in open(p):
        r = json.loads(line)
        if r.get("kind") == "blocks":
            blocks[r["tag"]] = r
w1 = blocks.get("W-V1-r1-R01")
check("A6 Stage-0 W-V1-r1-R01 group-0 logical block 0 = 436 (AUDIT task 4)", w1 is not None and w1["groups"][0][0] == 436)
check("A6 Stage-0 W-V1-r1-R01 group-2 column 2 = 440 (AUDIT task 3)", w1 is not None and w1["groups"][2][2] == 440)
w2 = blocks.get("W-V1-r2-R01")
check("A6 Stage-0 W-V1-r2-R01 group-0 logical block 0 = 762", w2 is not None and w2["groups"][0][0] == 762)
ok = True
for w in (w1, w2):
    ids = w["groups"][0][:3]
    ent = [x * 68 + k for x in ids for k in range(68)]
    ok = ok and [LY.physical(ent, j, 68) for j in range(3)] == ids
check("A6 physical() inverts the kernel-entry expansion (bpk 68)", ok)
for w, rng in ((w1, (62616576, 62760192)), (w2, (109435392, 109579008))):
    check(f"A6 capture plan for {w['tag']}: layer-7 indexer logical 0 -> AUDIT range",
          LY.idx_block(LY.physical([w['groups'][0][0] * 68 + k for k in range(68)], 0, 68)) == rng)

t = LY.e4m3_table()
check("A7 E4M3 0x38 = 1.0, 0x7e = 448, 0x01 = 2^-9, 0xb8 = -1.0, 0x7f = NaN",
      t[0x38] == 1.0 and t[0x7e] == 448.0 and t[0x01] == 2.0 ** -9 and t[0xb8] == -1.0 and t[0x7f] is None)
check("A7 LOC_OPERATION covers every boundary", all(p in LY.LOC_OPERATION for p, _t, _d in LY.LOC_ORDER) and
      all(p in LY.LOC_OPERATION for p in LY.STATE_POINTS))
print("layout tests: %d PASS, %d FAIL" % (N["PASS"], N["FAIL"]))
sys.exit(1 if N["FAIL"] else 0)
